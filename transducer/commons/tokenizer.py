from abc import ABC, abstractmethod
from typing import List, Union


class TokenizerBase(ABC):
    @abstractmethod
    def encode(self, texts: Union[str, List[str]]):
        raise NotImplementedError

    @abstractmethod
    def decode(self, texts: Union[List[int], List[List[int]]]) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_id(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def unk_id(self) -> int:
        raise NotImplementedError
