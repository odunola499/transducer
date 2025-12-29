from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor, nn

from transducer.commons import ModelOutput
from transducer.models.config import ModelConfig


class BaseModel(nn.Module, ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_loss(
        self, lattice: Tensor, labels: Tensor, act_lens: Tensor, label_lens: Tensor
    ):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        audio_features: Tensor,
        labels: Tensor,
        label_lens: Tensor,
        audio_lens: Optional[Tensor] = None,
    ) -> ModelOutput:
        pass
