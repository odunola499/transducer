from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional, Union

import librosa
import numpy as np
from torch import distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from transformers import AutoFeatureExtractor

from transducer.dataset.config import DatasetConfig
from transducer.processor import Processor, Tokenizer, NemoFeatureExtractor

FEATURE_EXTRACTORS = {
    "wav2vec2": AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base"),
    "wav2vecbert": AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0"),
    "parakeet": NemoFeatureExtractor(),
}


class BaseDataset(Dataset, ABC):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config
        tokenizer_config = config.tokenizer_config
        tokenizer = Tokenizer(tokenizer_config)
        feature_extractor = FEATURE_EXTRACTORS[config.feature_extractor_type]
        processor = Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.processor = processor

        self.sample_rate = config.sample_rate

    def load_audio(
        self, audio: Union[np.ndarray, bytes, str], orig_sr: Optional[int] = None
    ) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            assert orig_sr is not None, "orig_sr must be provided for type numpy array"

        elif isinstance(audio, str):
            audio, orig_sr = librosa.load(audio, sr=self.sample_rate)
            return audio

        elif isinstance(audio, bytes):
            audio, orig_sr = librosa.load(BytesIO(audio), sr=self.sample_rate)
            return audio
        else:
            raise ValueError("Unknown audio type")

        if orig_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
        audio = audio[: int(self.sample_rate * self.config.max_audio_length_ms * 0.001)]
        return audio

    @abstractmethod
    def _collate_fn(self, batch):
        raise NotImplementedError


class StreamingBaseDataset(IterableDataset, ABC):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config
        tokenizer_config = config.tokenizer_config
        tokenizer = Tokenizer(tokenizer_config)
        feature_extractor = FEATURE_EXTRACTORS[config.feature_extractor_type]
        processor = Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.processor = processor

        self.sample_rate = config.sample_rate

    def should_yield(self, index: int) -> bool:
        worker_info = get_worker_info()
        if not dist.is_available() or not dist.is_initialized():
            return True
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            if worker_info is None:
                worker_id = 0
                num_workers = 1
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

            global_worker_id = rank * num_workers + worker_id
            total_workers = world_size * num_workers

            if index % total_workers == global_worker_id:
                return True
            return False

    def load_audio(
        self, audio: Union[np.ndarray, bytes, str], orig_sr: Optional[int] = None
    ) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            assert orig_sr is not None, "orig_sr must be provided for type numpy array"

        elif isinstance(audio, str):
            audio, orig_sr = librosa.load(audio, sr=self.sample_rate)
            return audio

        elif isinstance(audio, bytes):
            audio, orig_sr = librosa.load(BytesIO(audio), sr=self.sample_rate)
            return audio
        else:
            raise ValueError("Unknown audio type")

        if orig_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)

        audio = audio[: int(self.sample_rate * self.config.max_audio_length_ms * 0.001)]
        return audio

    @abstractmethod
    def _collate_fn(self, batch):
        raise NotImplementedError
