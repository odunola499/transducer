from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

from torch import Tensor, nn


@dataclass
class EncoderOutput:
    last_hidden_state: Tensor

@dataclass
class Hypothesis:
    tokens:Tensor
    pred_out:Tensor
    pred_state:Tensor
    score:Tensor

@dataclass
class GenerationOutput:
    ids:Tensor
    labels: Union[List[str], List[int]]

@dataclass
class ModelOutput:
    loss:Tensor
    lattice:Tensor

class Encoder(nn.Module, ABC):
    @abstractmethod
    def _load_hf_weights(self, load_into_params=True):
        raise NotImplementedError

    def encoder_name(self):
        raise NotImplementedError

class Predictor(nn.Module, ABC):
    @abstractmethod
    def step(self, input_ids: Tensor, state: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def init_state(self, batch_size: int):
        raise NotImplementedError

class Joiner(nn.Module, ABC):
    ...


class Loss(nn.modules.loss._Loss, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
