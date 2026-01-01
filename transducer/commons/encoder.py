from abc import ABC, abstractmethod

from torch import nn


class Encoder(nn.Module, ABC):
    @abstractmethod
    def _load_hf_weights(self, load_into_params=True):
        raise NotImplementedError

    @abstractmethod
    def encoder_name(self):
        raise NotImplementedError
