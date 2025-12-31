from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class FeatureExtractor(ABC):
    model_input_names = []

    @abstractmethod
    def __call__(
        self,
        audio: Any,
        lengths: Optional[Any] = None,
        return_tensors: str | None = None,
        padding: bool = True,
    ) -> Dict[str, Any]:
        raise NotImplementedError
