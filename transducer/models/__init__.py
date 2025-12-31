__all__ = [
    "BaseModel",
    "DawnDecoderConfig",
    "DawnModel",
    "DawnModelConfig",
    "FastConformerConfig",
    "Parakeet",
    "ParakeetDecoderConfig",
    "ParakeetModelConfig",
    "Wav2Vec2BertConfig",
    "Wav2VecLargeConfig",
    "Wav2VecSmallConfig",
]


def __getattr__(name):
    if name == "BaseModel":
        from transducer.commons.modeling import BaseModel

        return BaseModel
    if name in {
        "DawnDecoderConfig",
        "DawnModel",
        "DawnModelConfig",
        "Wav2Vec2BertConfig",
        "Wav2VecLargeConfig",
        "Wav2VecSmallConfig",
    }:
        from transducer.models import dawn

        return getattr(dawn, name)
    if name in {
        "FastConformerConfig",
        "Parakeet",
        "ParakeetDecoderConfig",
        "ParakeetModelConfig",
    }:
        from transducer.models import parakeet

        return getattr(parakeet, name)
    raise AttributeError(name)
