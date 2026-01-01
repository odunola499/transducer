from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from transducer.dataset.config import DatasetConfig
    from transducer.models.config import ModelConfig
    from transducer.train.config import TrainConfig


from transducer.commons.config import Args


def _ensure_forward_refs():
    from transducer.dataset.config import DatasetConfig
    from transducer.models.config import ModelConfig
    from transducer.train.config import TrainConfig

    globals().update(
        TrainConfig=TrainConfig,
        ModelConfig=ModelConfig,
        DatasetConfig=DatasetConfig,
    )
    Config.model_rebuild()
    return TrainConfig, ModelConfig, DatasetConfig


class Config(Args):
    train: "TrainConfig"
    model: "ModelConfig"
    dataset: "DatasetConfig"

    def model_post_init(self, __context):
        if self.train.strategy == "fsdp":
            raise NotImplementedError("FSDP is not supported yet; use strategy='ddp'.")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False)

    @classmethod
    def from_yaml(cls, content: str) -> "Config":
        _ensure_forward_refs()
        data = yaml.safe_load(content)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "Config":
        path = Path(path)
        return cls.from_yaml(path.read_text())
