from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EncoderConfig:
    attn_impl: Literal["flash_attn", "sdpa", "math"] = "sdpa"

    @property
    def embed_dim(self):
        return getattr(self, "hidden_size", None)


@dataclass
class Wav2VecSmallConfig(EncoderConfig):
    dropout: float = 0.1
    conv_dim: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 512, 512, 512])
    conv_kernel: list[int] = field(default_factory=lambda: [10, 3, 3, 3, 3, 2, 2])
    conv_stride: list[int] = field(default_factory=lambda: [5, 2, 2, 2, 2, 2, 2])
    num_feat_extract_layers: int = 7
    final_dropout: float = 0.0
    hidden_size: int = 768
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-5
    layerdrop: float = 0.0
    num_attention_heads: int = 12
    num_conv_pos_embedding_groups: int = 16
    num_conv_pos_embeddings: int = 128
    num_hidden_layers: int = 12
    conv_bias: bool = False
    feat_proj_dropout: float = 0.1
    activation_dropout: float = 0.0
    hidden_dropout: float = 0.1


@dataclass
class Wav2VecLargeConfig(Wav2VecSmallConfig):
    activation_dropout: float = 0.1
    apply_spec_augment: bool = True
    attention_dropout: float = 0.1
    feat_extract_activation: str = "gelu"
    feat_extract_dropout: float = 0.0
    feat_extract_norm: str = "group"
    final_dropout: float = 0.1
    hidden_act: str = "gelu"
    hidden_dropout: float = 0.1
    hidden_size: int = 1024
    intermediate_size: int = 4096
    layerdrop: float = 0.1
    mask_feature_length: int = 10
    mask_feature_prob: float = 0.0
    mask_time_length: int = 10
    mask_time_prob: float = 0.05
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    proj_codevector_dim: int = 768
    vocab_size: int = 32


Wav2VecConfig = Wav2VecSmallConfig | Wav2VecLargeConfig


@dataclass
class Wav2Vec2BertConfig(EncoderConfig):
    activation_dropout: float = 0.0
    apply_spec_augment: bool = False
    attention_dropout: float = 0.0
    conv_depthwise_kernel_size: int = 31
    conformer_conv_dropout: float = 0.1
    feat_proj_dropout: float = 0.0
    feature_projection_input_dim: int = 160
    hidden_act: str = "swish"
    hidden_dropout: float = 0.0
    hidden_size: int = 1024
    intermediate_size: int = 4096
    layer_norm_eps: float = 1e-5
    layerdrop: float = 0.1
    left_max_position_embeddings: int = 64
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    mask_feature_prob: float = 0.0
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_time_prob: float = 0.05
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    position_embeddings_type: str = "relative_key"
    right_max_position_embeddings: int = 8


@dataclass
class DecoderConfig:
    embed_dim: int
    hidden_dim: int
    pred_dim: int
    joint_dim: int
    num_layers: int
    dropout: int
    blank_id: int
    vocab_size:int = 1024
