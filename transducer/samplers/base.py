import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Union, List
from transducer.commons import GenerationOutput, Predictor, Joiner, Encoder




class BaseSampler(ABC):
    def __init__(self, max_symbols_per_timestep:int = 5, blank_id:int = 0):
        super().__init__()
        self.max_symbols_per_timestep = max_symbols_per_timestep
        self.blank_id = blank_id

    @abstractmethod
    def get_feature_extractor(self):
        pass

    @abstractmethod
    def get_tokenizer(self):
        pass

    @abstractmethod
    def get_encoder(self) -> Encoder:
        pass
    @abstractmethod
    def get_joiner(self) -> Joiner:
        pass
    @abstractmethod
    def get_predictor(self) -> Predictor:
        pass

    @abstractmethod
    def offline_decode(self, *args, **kwargs) -> GenerationOutput:
        pass

    @abstractmethod
    def decode_frame(self, *args, **kwargs):
        pass
