from typing import Literal, Optional

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr

from transducer.commons.config import Args, DecoderConfig, EncoderConfig, ModelConfig


class FastConformerConfig(EncoderConfig):
    hidden_size: StrictInt = 512
    num_hidden_layers: StrictInt = 17
    ff_expansion_factor: StrictInt = 4
    feat_in: int = 128
    use_bias: StrictBool = False
    model_name: StrictStr = "fastconformer"
    spec_augment_freq_masks: int = 2
    spec_augment_time_masks: int = 2
    spec_augment_freq_widths: int = 2
    spec_augment_time_widths: float = 0.05

    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    causal_subsampling: bool = False

    num_heads: int = 8
    att_context_size: list = Field(default_factory=lambda: [70, 1])
    att_context_style: Literal["regular", "chunked_limited"] = "chunked_limited"
    pos_emb_max_len: int = 5000

    conv_kernel_size: StrictInt = 9
    conv_context_size: list = None
    dropout: float = 0.1
    dropout_pre_encoder: float = 0.1
    dropout_emb: float = 0.0
    dropout_att: float = 0.1


class ParakeetDecoderConfig(DecoderConfig):
    rnn_type: Literal["gru", "lstm"] = "lstm"
    embed_dim: StrictInt = 512
    hidden_dim: StrictInt = 512
    pred_dim: StrictInt = 640
    joint_dim: StrictInt = 640
    num_layers: StrictInt = 1
    dropout: StrictFloat = 0.2
    vocab_size: StrictInt = 1026


class StreamingConfig(Args):
    chunk_size: list = Field(default_factory=lambda: [9, 16])
    shift_size: list = Field(default_factory=lambda: [9, 16])
    cache_drop_size: int = 0
    last_channel_cache_size: int = 70
    valid_out_len: int = 2
    pre_encode_cache_size: list = Field(default_factory=lambda: [0, 9])
    drop_extra_pre_encoded: int = 2
    last_channel_num: int = 0
    last_time_num: int = 0


class ParakeetModelConfig(ModelConfig):
    model_name: StrictStr = "parakeet"
    tokenizer_path: Optional[StrictStr] = None
    encoder_config: FastConformerConfig
    decoder_config: ParakeetDecoderConfig
