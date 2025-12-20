from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from transducer.commons import Encoder
from transducer.config import Wav2Vec2BertConfig
from transducer.models.attention import ATTN_FUNCTIONS


def _get_activation(name: str):
    if name == "gelu":
        return nn.GELU()
    if name in {"swish", "silu"}:
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    return nn.Identity()


class Wav2Vec2BertFeatureProjection(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            config.feature_projection_input_dim, eps=config.layer_norm_eps
        )
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states: Tensor):
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class Wav2Vec2BertAttention(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = config.attention_dropout
        self.config = config

        num_positions = (
            config.left_max_position_embeddings + config.right_max_position_embeddings + 1
        )
        self.distance_embedding = nn.Embedding(num_positions, self.head_dim)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None):
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        additive_mask = None
        if attention_mask is not None:
            additive_mask = attention_mask.to(query.dtype)

        position_ids_l = torch.arange(seq_len, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(seq_len, device=hidden_states.device).view(1, -1)
        distance = position_ids_r - position_ids_l
        distance = torch.clamp(
            distance,
            -self.config.left_max_position_embeddings,
            self.config.right_max_position_embeddings,
        )
        positional_embedding = self.distance_embedding(
            distance + self.config.left_max_position_embeddings
        ).to(dtype=query.dtype)
        relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
        relative_position_attn_weights = relative_position_attn_weights * self.scaling
        additive_mask = (
            relative_position_attn_weights
            if additive_mask is None
            else additive_mask + relative_position_attn_weights
        )

        attention_function = ATTN_FUNCTIONS[self.config.attn_impl]
        attn_output, attn_weights = attention_function(
            self,
            query=query,
            key=key,
            value=value,
            attention_mask=additive_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = self.linear_out(attn_output)
        return attn_output, attn_weights


class Wav2Vec2BertFeedForward(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act = _get_activation(config.hidden_act)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2BertConvolutionModule(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError(
                "`conv_depthwise_kernel_size` should be an odd number for 'SAME' padding"
            )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )
        self.depthwise_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = _get_activation(config.hidden_act)
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(config.conformer_conv_dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None):
        hidden_states = self.layer_norm(hidden_states)
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)

        hidden_states = torch.nn.functional.pad(
            hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0)
        )
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2BertEncoderLayer(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout

        self.ffn1_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ffn1 = Wav2Vec2BertFeedForward(config)

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = Wav2Vec2BertAttention(config)

        self.conv_module = Wav2Vec2BertConvolutionModule(config)

        self.ffn2_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ffn2 = Wav2Vec2BertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        conv_attention_mask: Optional[Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class Wav2Vec2BertEncoder(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2BertEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None):
        conv_attention_mask = attention_mask
        additive_mask = None
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            additive_mask = 1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)
            additive_mask = additive_mask * torch.finfo(hidden_states.dtype).min

        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if self.training and torch.rand([]) < self.config.layerdrop:
                continue
            hidden_states = layer(
                hidden_states, attention_mask=additive_mask, conv_attention_mask=conv_attention_mask
            )
        return hidden_states


def _compute_mask_indices(
    shape: tuple[int, int], mask_prob: float, mask_length: int, min_masks: int = 0
):
    batch_size, seq_length = shape
    if mask_length < 1 or mask_length > seq_length:
        raise ValueError("Invalid mask_length for mask computation.")

    num_masks = max(int(mask_prob * seq_length / mask_length), min_masks)
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    for b in range(batch_size):
        if num_masks == 0:
            continue
        starts = torch.randperm(seq_length - mask_length + 1)[:num_masks]
        for s in starts:
            mask[b, s : s + mask_length] = True
    return mask


class Wav2Vec2BertModel(Encoder):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        self.config = config
        self.feature_projection = Wav2Vec2BertFeatureProjection(config)
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())
        else:
            self.masked_spec_embed = None
        self.encoder = Wav2Vec2BertEncoder(config)

    def _mask_hidden_states(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_time_indices: Optional[Tensor] = None,
    ):
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        batch_size, sequence_length, hidden_size = hidden_states.size()
        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                min_masks=self.config.mask_time_min_masks,
            ).to(hidden_states.device)
            hidden_states[mask] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            mask = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            ).to(hidden_states.device)
            mask = mask[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask] = 0
        return hidden_states

    def forward(self, input_features: Tensor, attention_mask: Optional[Tensor] = None):
        hidden_states, _ = self.feature_projection(input_features)
        if self.masked_spec_embed is not None:
            hidden_states = self._mask_hidden_states(hidden_states, attention_mask=attention_mask)
        hidden_states = self.encoder(hidden_states, attention_mask=attention_mask)
        return hidden_states

    def _load_hf_weights(
        self, load_into_params: bool = True, repo_id: str = "facebook/w2v-bert-2.0"
    ):
        local_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        weights = torch.load(local_path, map_location="cpu")
        weights = remap_hf_state_dict_wav2vec2bert(weights)
        if load_into_params:
            self.load_state_dict(weights, strict=False)
        return weights


def remap_hf_state_dict_wav2vec2bert(state_dict):
    remapped = {}
    temp = {}

    for key, value in state_dict.items():
        if key.startswith("wav2vec2_bert."):
            new_key = key[len("wav2vec2_bert.") :]
        elif key.startswith("wav2vec2."):
            new_key = key[len("wav2vec2.") :]
        else:
            new_key = key

        if new_key.startswith(("adapter.", "projector", "tdnn", "lm_head", "layer_weights")):
            continue

        temp[new_key] = value

    layer_ids = set()
    for key in temp:
        if key.startswith("encoder.layers.") and ".self_attn.linear_q.weight" in key:
            layer_ids.add(key.split(".")[2])

    for layer_id in layer_ids:
        prefix = f"encoder.layers.{layer_id}.self_attn."
        q_w = temp.pop(prefix + "linear_q.weight")
        k_w = temp.pop(prefix + "linear_k.weight")
        v_w = temp.pop(prefix + "linear_v.weight")
        remapped[prefix + "qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = temp.pop(prefix + "linear_q.bias")
        k_b = temp.pop(prefix + "linear_k.bias")
        v_b = temp.pop(prefix + "linear_v.bias")
        remapped[prefix + "qkv_proj.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        remapped[prefix + "linear_out.weight"] = temp.pop(prefix + "linear_out.weight")
        remapped[prefix + "linear_out.bias"] = temp.pop(prefix + "linear_out.bias")

    for key, value in temp.items():
        if ".self_attn.linear_pos." in key or ".self_attn.pos_bias_" in key:
            continue
        remapped[key] = value

    return remapped


if __name__ == "__main__":
    config = Wav2Vec2BertConfig(num_hidden_layers=2)
    model = Wav2Vec2BertModel(config)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num params: {num_params}")

    tensor = torch.randn(2, 200, config.feature_projection_input_dim)
    with torch.no_grad():
        out = model(tensor)
    print(f"output shape: {out.shape}")

    from transformers import Wav2Vec2BertModel as HFWav2Vec2BertModel

    hf_model = HFWav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    hf_model.eval()
    state_dict = remap_hf_state_dict_wav2vec2bert(hf_model.state_dict())
    model.load_state_dict(state_dict, strict=False)
    with torch.no_grad():
        hf_out = hf_model(tensor).last_hidden_state
        print(torch.max(torch.abs(out - hf_out)))
