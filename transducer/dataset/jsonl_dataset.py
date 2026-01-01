import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from transducer.dataset.base import BaseDataset
from transducer.dataset.config import DatasetConfig, JsonlDatasetStruct


class JsonlDataset(BaseDataset):
    def __init__(self, dataset: JsonlDatasetStruct, config: DatasetConfig):
        super().__init__(config)
        if not dataset.jsonl_filepath:
            raise ValueError("jsonl_filepath must be provided for JsonlDataset.")
        self.dataset_config = dataset
        self.data = self._load_jsonl(Path(dataset.jsonl_filepath))

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"jsonl_filepath does not exist: {path}")
        samples: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    def __len__(self):
        return len(self.data)

    def _prepare_audio(self, audio: Any) -> np.ndarray:
        # Support common jsonl encodings: path, raw bytes, or array with sampling_rate metadata.
        if isinstance(audio, dict):
            if "path" in audio:
                return self.load_audio(audio["path"])
            if "bytes" in audio:
                return self.load_audio(audio["bytes"])
            if "array" in audio:
                array = np.asarray(audio["array"])
                orig_sr = audio.get("sampling_rate", self.sample_rate)
                return self.load_audio(array, orig_sr=orig_sr)
        elif isinstance(audio, np.ndarray):
            return self.load_audio(audio, orig_sr=self.sample_rate)
        return self.load_audio(audio)

    def __getitem__(self, idx):
        row = self.data[idx]
        audio_value = row[self.dataset_config.audio_column_name]
        raw_text = row[self.dataset_config.text_column_name]

        audio = self._prepare_audio(audio_value)
        text = self.processor.tokenize(raw_text)
        return {
            "audio": audio,
            "text": text,
            "raw_text": raw_text,
            "index": idx,
        }

    def _collate_fn(self, batch):
        audios = [i["audio"] for i in batch]
        texts = [i["text"] for i in batch]
        indices = [i["index"] for i in batch]
        raw_texts = [i.get("raw_text") for i in batch]

        features = self.processor.extract_features(
            audios, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        if isinstance(features, dict):
            features = features["input_values"]
        else:
            features = features.input_values
        labels = pad_sequence(
            texts, batch_first=True, padding_value=self.processor.pad_id
        )
        label_lens = (labels != self.processor.pad_id).sum(dim=-1)
        return {
            "audio_features": features,
            "labels": labels,
            "label_lens": label_lens,
            "audio_lens": None,
            "indices": indices,
            "texts": raw_texts,
        }

    def get_loader(self, batch_size):
        num_workers = self.config.num_workers
        pin_memory = self.config.pin_memory

        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )
        return loader
