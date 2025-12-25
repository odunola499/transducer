import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod
from transducer.models.config import ModelConfig
from transducer.commons import ModelOutput
from typing import Optional

class BaseModel(nn.Module, ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_loss(self, lattice:Tensor, labels:Tensor, act_lens:Tensor, label_lens:Tensor):
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

