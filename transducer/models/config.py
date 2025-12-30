from typing import Literal, Optional

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr

from transducer.config import Args


class EncoderConfig(Args):
    attn_impl: Literal["flash_attn", "sdpa", "math"] = "sdpa"

    @property
    def embed_dim(self):
        return getattr(self, "hidden_size", None)


class Wav2VecSmallConfig(EncoderConfig):
    model_name: StrictStr = "wav2vec2small"
    dropout: StrictFloat = 0.1
    conv_dim: list[StrictInt] = Field(
        default_factory=lambda: [512, 512, 512, 512, 512, 512, 512]
    )
    conv_kernel: list[StrictInt] = Field(default_factory=lambda: [10, 3, 3, 3, 3, 2, 2])
    conv_stride: list[StrictInt] = Field(default_factory=lambda: [5, 2, 2, 2, 2, 2, 2])
    num_feat_extract_layers: StrictInt = 7
    final_dropout: StrictFloat = 0.0
    hidden_size: StrictInt = 768
    intermediate_size: StrictInt = 3072
    layer_norm_eps: StrictFloat = 1e-5
    layerdrop: StrictFloat = 0.0
    num_attention_heads: StrictInt = 12
    num_conv_pos_embedding_groups: StrictInt = 16
    num_conv_pos_embeddings: StrictInt = 128
    num_hidden_layers: StrictInt = 12
    conv_bias: StrictBool = False
    feat_proj_dropout: StrictFloat = 0.1
    activation_dropout: StrictFloat = 0.0
    hidden_dropout: StrictFloat = 0.1


class Wav2VecLargeConfig(Wav2VecSmallConfig):
    model_name: StrictStr = "wav2veclarge"
    activation_dropout: StrictFloat = 0.1
    apply_spec_augment: StrictBool = True
    attention_dropout: StrictFloat = 0.1
    feat_extract_activation: StrictStr = "gelu"
    feat_extract_dropout: StrictFloat = 0.0
    feat_extract_norm: StrictStr = "group"
    final_dropout: StrictFloat = 0.1
    hidden_act: StrictStr = "gelu"
    hidden_dropout: StrictFloat = 0.1
    hidden_size: StrictInt = 1024
    intermediate_size: StrictInt = 4096
    layerdrop: StrictFloat = 0.1
    mask_feature_length: StrictInt = 10
    mask_feature_prob: StrictFloat = 0.0
    mask_time_length: StrictInt = 10
    mask_time_prob: StrictFloat = 0.05
    num_attention_heads: StrictInt = 16
    num_hidden_layers: StrictInt = 24
    proj_codevector_dim: StrictInt = 768


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
    att_context_size: list = Field(default_factory=lambda: [-1, -1])
    pos_emb_max_len: int = 5000

    conv_kernel_size: StrictInt = 9
    conv_context_size: list = None
    dropout: float = 0.1
    dropout_pre_encoder: float = 0.1
    dropout_emb: float = 0.0
    dropout_att: float = 0.1


Wav2VecConfig = Wav2VecSmallConfig | Wav2VecLargeConfig


class Wav2Vec2BertConfig(EncoderConfig):
    model_name: StrictStr = "wav2vec2bert"
    activation_dropout: StrictFloat = 0.0
    apply_spec_augment: StrictBool = False
    attention_dropout: StrictFloat = 0.0
    conv_depthwise_kernel_size: StrictInt = 31
    conformer_conv_dropout: StrictFloat = 0.1
    feat_proj_dropout: StrictFloat = 0.0
    feature_projection_input_dim: StrictInt = 160
    hidden_act: StrictStr = "swish"
    hidden_dropout: StrictFloat = 0.0
    hidden_size: StrictInt = 1024
    intermediate_size: StrictInt = 4096
    layer_norm_eps: StrictFloat = 1e-5
    layerdrop: StrictFloat = 0.1
    left_max_position_embeddings: StrictInt = 64
    mask_feature_length: StrictInt = 10
    mask_feature_min_masks: StrictInt = 0
    mask_feature_prob: StrictFloat = 0.0
    mask_time_length: StrictInt = 10
    mask_time_min_masks: StrictInt = 2
    mask_time_prob: StrictFloat = 0.05
    num_attention_heads: StrictInt = 16
    num_hidden_layers: StrictInt = 24
    position_embeddings_type: StrictStr = "relative_key"
    right_max_position_embeddings: StrictInt = 8


class DecoderConfig(Args):
    rnn_type: Literal["gru", "lstm"] = "gru"
    embed_dim: StrictInt
    hidden_dim: StrictInt = 1024
    pred_dim: StrictInt = 640
    joint_dim: StrictInt = 640
    num_layers: StrictInt
    dropout: StrictFloat
    vocab_size: StrictInt = 1024


class ParakeetDecoderConfig(DecoderConfig):
    rnn_type: Literal["gru", "lstm"] = "lstm"
    embed_dim: StrictInt = 512
    hidden_dim: StrictInt = 512
    pred_dim: StrictInt = 640
    joint_dim: StrictInt = 640
    num_layers: StrictInt = 1
    dropout: StrictFloat = 0.2
    vocab_size: StrictInt = 1024


class ModelConfig(Args):
    model_name: StrictStr  # Used for logging etc
    loss_type: Literal["tdt", "tdt_triton", "rnnt", "rnnt_triton"] = "rnnt"
    loss_duration: Optional[list[StrictInt]] = None
    fastemit_lambda: StrictFloat = 0.0
    # If None, blank_id will be set to decoder_config.vocab_size (i.e., after tokens).
    blank_id: Optional[StrictInt] = None
    loss_reduction: Literal["sum", "mean"] = "sum"
    sampler_type: Literal["greedy_search", "beam_search"] = "greedy_search"
    encoder_config: (
        Wav2VecSmallConfig
        | Wav2VecLargeConfig
        | Wav2Vec2BertConfig
        | FastConformerConfig
    )
    decoder_config: DecoderConfig | ParakeetDecoderConfig
