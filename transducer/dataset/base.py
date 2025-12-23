from abc import abstractmethod, ABC

import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from typing import Union
from io import BytesIO
import librosa
from typing import Optional
from transducer.dataset.config import DatasetConfig
from transducer.processor import Processor, Tokenizer
from transformers import AutoFeatureExtractor

FEATURE_EXTRACTORS = {
    'wav2vec2': AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base'),
    'wav2vec-bert': AutoFeatureExtractor.from_pretrained('facebook/w2v-bert')
}
class BaseDataset(Dataset, ABC):
    def __init__(self, config:DatasetConfig) -> None:
        super().__init__()
        self.config = config
        tokenizer_config = config.tokenizer_config
        tokenizer = Tokenizer(tokenizer_config)
        feature_extractor = FEATURE_EXTRACTORS[tokenizer_config.feature_extractor_type]
        processor = Processor(feature_extractor = feature_extractor, tokenizer = tokenizer)
        self.processor = processor

        self.sample_rate = config.sample_rate

    def load_audio(self, audio:Union[np.ndarray, bytes, str], orig_sr:Optional[int] = None) -> np.ndarray:
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
        return audio


    @abstractmethod
    def __collate_fn(self, batch):
        pass

class BaseDataset(IterableDataset, ABC):
    def __init__(self, config:DatasetConfig) -> None:
        super().__init__()
        self.config = config
        tokenizer_config = config.tokenizer_config
        tokenizer = Tokenizer(tokenizer_config)
        feature_extractor = FEATURE_EXTRACTORS[tokenizer_config.feature_extractor_type]
        processor = Processor(feature_extractor = feature_extractor, tokenizer = tokenizer)
        self.processor = processor

        self.sample_rate = config.sample_rate

    def load_audio(self, audio:Union[np.ndarray, bytes, str], orig_sr:Optional[int] = None) -> np.ndarray:
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
        return audio


    @abstractmethod
    def __collate_fn(self, batch):
        pass

