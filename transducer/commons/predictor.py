from abc import ABC, abstractmethod

from torch import Tensor, nn


class Predictor(nn.Module, ABC):
    @abstractmethod
    def step(self, input_ids: Tensor, state: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def init_state(self, batch_size: int):
        raise NotImplementedError
