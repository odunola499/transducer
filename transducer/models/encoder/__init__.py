from transformers import AutoFeatureExtractor

from transducer.models.encoder.wav2vec import Wav2VecModel
from transducer.models.encoder.wav2vec2bert import Wav2Vec2BertModel

FEATURE_EXTRACTORS = {
    'wav2vec': AutoFeatureExtractor.from_pretrained('facebook/wav2vec'),
    'wav2vecbert': AutoFeatureExtractor.from_pretrained('facebook/wav2vec')
}

__all__ = [
    "Wav2VecModel",
    "Wav2Vec2BertModel",
    "FEATURE_EXTRACTORS",
]
