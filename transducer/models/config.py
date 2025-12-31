from transducer.commons.config import DecoderConfig, EncoderConfig, ModelConfig as BaseModelConfig
from transducer.models.dawn.config import (
    DawnDecoderConfig,
    DawnModelConfig,
    Wav2Vec2BertConfig,
    Wav2VecConfig,
    Wav2VecLargeConfig,
    Wav2VecSmallConfig,
)
from transducer.models.parakeet.config import (
    FastConformerConfig,
    ParakeetDecoderConfig,
    ParakeetModelConfig,
    StreamingConfig,
)

ModelConfig = DawnModelConfig | ParakeetModelConfig

__all__ = [
    "BaseModelConfig",
    "DecoderConfig",
    "DawnDecoderConfig",
    "EncoderConfig",
    "FastConformerConfig",
    "ModelConfig",
    "ParakeetDecoderConfig",
    "ParakeetModelConfig",
    "StreamingConfig",
    "Wav2Vec2BertConfig",
    "Wav2VecConfig",
    "Wav2VecLargeConfig",
    "Wav2VecSmallConfig",
]
