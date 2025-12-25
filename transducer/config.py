from pathlib import Path
import yaml
from transducer.config_base import Args
from transducer.dataset.config import DatasetConfig
from transducer.models.config import ModelConfig
from transducer.train.config import TrainConfig


class Config(Args):
    train: TrainConfig
    model: ModelConfig
    dataset: DatasetConfig

    def model_post_init(self, __context):
        if self.train.strategy == "fsdp":
            raise NotImplementedError("FSDP is not supported yet; use strategy='ddp'.")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False)

    @classmethod
    def from_yaml(cls, content: str) -> "Config":
        data = yaml.safe_load(content)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "Config":
        path = Path(path)
        return cls.from_yaml(path.read_text())
