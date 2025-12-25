from typing import Any, Dict, Optional, Union

from transformers import AutoFeatureExtractor


class Processor:
    def __init__(self, feature_extractor, tokenizer):
        if feature_extractor is None:
            raise ValueError("feature_extractor is required")
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def extract_features(
        self,
        audio,
        sampling_rate: Optional[int] = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        if sampling_rate is not None:
            kwargs["sampling_rate"] = sampling_rate
        return self.feature_extractor(audio, return_tensors=return_tensors, padding = True, **kwargs)

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def pad_id(self):
        return self.tokenizer.pad_id

    def __call__(
        self,
        audio: Optional[Any] = None,
        text: Optional[Union[str, list]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> Dict[str, Any]:
        if audio is None and text is None:
            raise ValueError("audio or text must be provided")
        output: Dict[str, Any] = {}
        if audio is not None:
            output["features"] = self.extract_features(
                audio, sampling_rate=sampling_rate, return_tensors=return_tensors, **kwargs
            )
        if text is not None:
            output["input_ids"] = self.tokenize(text)
        return output
