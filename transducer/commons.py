from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn
from pydantic import BaseModel

class Args(BaseModel):
    def to_dict(self):
        return self.model_dump()
    def to_json(self):
        return self.model_dump_json()


@dataclass
class EncoderOutput:
    last_hidden_state: Tensor


class Encoder(nn.Module, ABC):
    @abstractmethod
    def _load_hf_weights(self, load_into_params=True):
        pass


class Predictor(nn.Module, ABC):
    @abstractmethod
    def step(self, input_ids: Tensor, **kwargs):
        pass


class Joiner(nn.Module, ABC):
    ...


class Loss(nn.modules.loss._Loss, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
