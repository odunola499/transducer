from transducer.dataset.config import (
    DatasetConfig,
    DatasetStruct,
    HFDatasetStruct,
    JsonlDatasetStruct,
    TokenizerConfig,
)

__all__ = [
    "DatasetConfig",
    "DatasetStruct",
    "HFDatasetStruct",
    "JsonlDatasetStruct",
    "TokenizerConfig",
    "BaseDataset",
    "StreamingBaseDataset",
    "HFDataset",
    "StreamingHFDataset",
    "JsonlDataset",
]


def __getattr__(name):
    if name in {"BaseDataset", "StreamingBaseDataset"}:
        from transducer.dataset.base import BaseDataset, StreamingBaseDataset

        return BaseDataset if name == "BaseDataset" else StreamingBaseDataset
    if name in {"HFDataset", "StreamingHFDataset"}:
        from transducer.dataset.hf_dataset import HFDataset, StreamingHFDataset

        return HFDataset if name == "HFDataset" else StreamingHFDataset
    if name == "JsonlDataset":
        from transducer.dataset.jsonl_dataset import JsonlDataset

        return JsonlDataset
    raise AttributeError(name)
