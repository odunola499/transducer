from typing import Any, Dict, Optional

from transformers import AutoFeatureExtractor

from transducer.commons import FeatureExtractor

WAV2VEC2_MODEL_ID = "facebook/wav2vec2-base"
WAV2VEC2BERT_MODEL_ID = "facebook/w2v-bert-2.0"


class DawnFeatureExtractor(FeatureExtractor):
    def __init__(self, model_id: str):
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model_input_names = self.extractor.model_input_names

    def __call__(
        self,
        audio: Any,
        lengths: Optional[Any] = None,
        return_tensors: str | None = None,
        padding: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        sampling_rate = kwargs.pop("sampling_rate", None)
        return self.extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors=return_tensors,
            padding=padding,
            **kwargs,
        )


FEATURE_EXTRACTORS = {
    "wav2vec2": DawnFeatureExtractor(WAV2VEC2_MODEL_ID),
    "wav2vecbert": DawnFeatureExtractor(WAV2VEC2BERT_MODEL_ID),
}
