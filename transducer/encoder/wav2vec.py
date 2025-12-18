import torch
from torch import Tensor, nn
from transducer.config import Wav2VecConfig, Wav2VecSmallConfig, Wav2VecLargeConfig
from transducer.model.modules.attention import ATTN_FUNCTIONS
from huggingface_hub import hf_hub_download
from transducer.commons import Encoder

def remap_hf_state_dict(state_dict):
    remapped = {}
    temp = {}
    for key, value in state_dict.items():
        if key == "masked_spec_embed":
            continue
        new_key = key.replace("encoder.", "", 1)
        temp[new_key] = value

    layer_ids = set()
    for key in temp:
        if key.startswith("layers.") and ".attention.q_proj.weight" in key:
            layer_ids.add(key.split(".")[1])

    for layer_id in layer_ids:
        prefix = f"layers.{layer_id}.attention."
        q_w = temp.pop(prefix + "q_proj.weight")
        k_w = temp.pop(prefix + "k_proj.weight")
        v_w = temp.pop(prefix + "v_proj.weight")
        remapped[prefix + "qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = temp.pop(prefix + "q_proj.bias")
        k_b = temp.pop(prefix + "k_proj.bias")
        v_b = temp.pop(prefix + "v_proj.bias")
        remapped[prefix + "qkv_proj.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

    for key, value in temp.items():
        remapped[key] = value

    return remapped


class NormConv(nn.Module):
    def __init__(self, config: Wav2VecConfig, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size = config.conv_kernel[layer_id],
            stride = config.conv_stride[layer_id],
            bias = config.conv_bias
        )
        self.activation = nn.GELU()

    def forward(self, hidden_states:Tensor):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class GroupNormConv(nn.Module):
    def __init__(self, config: Wav2VecConfig, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )

        self.activation = nn.GELU()
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine = True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class PositionalConvEmbedding(nn.Module):
    def __init__(self, config: Wav2VecConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding = config.num_conv_pos_embeddings // 2,
            groups = config.num_conv_pos_embedding_groups
        )
        weight_norm = nn.utils.parametrizations.weight_norm
        self.conv = weight_norm(self.conv, name = 'weight', dim = 2)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)[:, :,:-1]
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

class FeatureEncoder(nn.Module):
    def __init__(self, config: Wav2VecConfig):
        super().__init__()
        conv_layers = [GroupNormConv(config, 0)] + [
            NormConv(config, i + 1) for i in range(0, config.num_feat_extract_layers -1)
        ]
        self.conv_layers = nn.ModuleList(conv_layers)

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, input_values:Tensor):
        hidden_states = input_values[:, None]

        if self.training:
            hidden_states = hidden_states.requires_grad_()

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states

class FeatureProjection(nn.Module):
    def __init__(self, config: Wav2VecConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps = config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states:Tensor):
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states

class Attention(nn.Module):
    def __init__(
            self,
            config: Wav2VecConfig,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.config = config

        self.scaling= self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
            self,
            hidden_states:Tensor,
    ):
        bsz, seq_len = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_state, key_state, value_state = qkv[0], qkv[1], qkv[2]

        attention_function = ATTN_FUNCTIONS[self.config.attn_impl]
        attn_output, attn_weights = attention_function(
            self,
            query=query_state,
            key = key_state,
            value = value_state,
            attention_mask = None,
            scaling=self.scaling,
            dropout = 0.0 if not self.training else self.dropout
        )

        attn_output = attn_output.reshape(bsz, seq_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights, None

class FeedForward(nn.Module):
    def __init__(self, config: Wav2VecConfig):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states

class EncoderLayer(nn.Module):
    def __init__(self, config: Wav2VecConfig):
        super().__init__()
        self.attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class Wav2VecModel(Encoder):
    def __init__(self, config: Wav2VecConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureEncoder(config)
        self.masked_spec_embed = nn.Identity()
        self.feature_projection = FeatureProjection(config)
        self.pos_conv_embed = PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def freeze_feature_encoder(self):
        self.feature_extractor._freeze_parameters()

    def forward(self, input_values:Tensor):
        hidden_states = self.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states, _ = self.feature_projection(hidden_states)
        hidden_states = hidden_states + self.pos_conv_embed(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states

    def _load_hf_weights(self, load_into_params = True):
        if isinstance(self.config, Wav2VecSmallConfig):
            hf_repo = 'facebook/wav2vec2-base'

        elif isinstance(self.config, Wav2VecLargeConfig):
            hf_repo = 'facebook/wav2vec2-large'

        else:
            raise AttributeError("Unknown Config class")

        local_path = hf_hub_download(
            repo_id = hf_repo,
            filename = 'pytorch_model.bin'
        )
        weights = torch.load(local_path, map_location="cpu")
        weights = remap_hf_state_dict(weights)
        if load_into_params:
            self.load_state_dict(weights)
        return weights






def spec_augment(hidden_states, time_mask_prob=0.05, time_mask_width=10, freq_mask_prob=0.05, freq_mask_width=10):
    batch, seq, feat = hidden_states.shape
    device = hidden_states.device
    time_mask_width = max(int(time_mask_width), 1)
    freq_mask_width = max(int(freq_mask_width), 1)
    num_time_masks = max(int(seq * time_mask_prob), 0)
    num_freq_masks = max(int(feat * freq_mask_prob), 0)
    for b in range(batch):
        for _ in range(num_time_masks):
            start = torch.randint(0, max(seq - time_mask_width, 1), (1,), device=device).item()
            hidden_states[b, start:start + time_mask_width] = 0
        for _ in range(num_freq_masks):
            start = torch.randint(0, max(feat - freq_mask_width, 1), (1,), device=device).item()
            hidden_states[b, :, start:start + freq_mask_width] = 0
    return hidden_states


if __name__ == '__main__':
    config = Wav2VecSmallConfig()
    model = Wav2VecModel(config)
    model.eval()
    num_params = sum([p.numel() for p in model.parameters()])
    print(num_params)
    tensor = torch.randn(2, 16000)

    print(tensor.shape)

    from transformers import Wav2Vec2Model

    hf_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    hf_model.eval()
    hf_num_params = sum([p.numel() for p in model.parameters()])
    print(f"num params: {num_params} hf_params; {hf_num_params}")

    state_dict = remap_hf_state_dict(hf_model.state_dict())
    model.load_state_dict(state_dict)
    with torch.no_grad():
        out = model(tensor)
        ref = hf_model(tensor).last_hidden_state
        print(torch.max(torch.abs(out - ref)))

        feat_extract_output = model.feature_extractor(tensor).transpose(1,2)
        hf_feat_extract_output = hf_model.feature_extractor(tensor).transpose(1,2)

        feat_project = model.feature_projection(feat_extract_output)
        hf_feat_project = hf_model.feature_projection(hf_feat_extract_output)

        print(torch.max(torch.abs(feat_extract_output - hf_feat_extract_output)))
        print(torch.max(torch.abs(feat_project[0]- hf_feat_project[0])))

        pre_block = model.dropout(model.layer_norm(feat_project[0] + model.pos_conv_embed(feat_project[0])))
        hf_pre_block= hf_model.encoder.dropout(hf_model.encoder.layer_norm(hf_feat_project[0] + hf_model.encoder.pos_conv_embed(feat_project[0])))
        print(torch.max(torch.abs(pre_block - hf_pre_block)))

        model_layer = model.layers[0]
        hf_model_layer = hf_model.encoder.layers[0]

        model_layer_out = model_layer(pre_block)
        hf_model_layer_out = hf_model_layer(hf_pre_block)[0]

        print(torch.max(torch.abs(model_layer_out - hf_model_layer_out)))

        layer_attn_block = model_layer.attention
        layer_attn_block_output = layer_attn_block(model_layer_out)[0]

        hf_model_attn_block = hf_model_layer.attention
        hf_layer_attn_block_output = hf_model_attn_block(hf_model_layer_out)[0]
        print(torch.max(torch.abs(layer_attn_block_output - hf_layer_attn_block_output)))




