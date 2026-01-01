from dataclasses import dataclass
from typing import List, Union

from torch import Tensor


@dataclass
class EncoderOutput:
    last_hidden_state: Tensor


@dataclass
class Hypothesis:
    tokens: Tensor
    pred_out: Tensor
    pred_state: Tensor
    score: Tensor


@dataclass
class GenerationOutput:
    ids: Tensor
    labels: Union[List[str], List[int]]


@dataclass
class ModelOutput:
    loss: Tensor
    lattice: Tensor
