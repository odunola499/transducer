from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor, nn

from transducer.commons.config import ModelConfig
from transducer.commons.types import ModelOutput


class BaseModel(nn.Module, ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def compute_loss(
        self, lattice: Tensor, labels: Tensor, act_lens: Tensor, label_lens: Tensor
    ): ...

    @abstractmethod
    def forward(
        self,
        audio_features: Tensor,
        labels: Tensor,
        label_lens: Tensor,
        audio_lens: Optional[Tensor] = None,
    ) -> ModelOutput:
        raise NotImplementedError
