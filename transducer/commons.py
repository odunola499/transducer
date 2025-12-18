from dataclasses import dataclass
from torch import Tensor, nn
from abc import ABC, abstractmethod

@dataclass
class EncoderOutput:
    last_hidden_state:Tensor


class Encoder(nn.Module, ABC):

    @abstractmethod
    def _load_hf_weights(self, load_into_params = True):
        pass

class Predictor(nn.Module, ABC):

    @abstractmethod
    def step(self, input_ids:Tensor, **kwargs):
        pass

class Joiner(nn.Module, ABC):
    ...

class Loss(nn.modules.loss._Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
