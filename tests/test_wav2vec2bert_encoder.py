import torch

from transducer.models.config import Wav2Vec2BertConfig
from models.encoder import wav2vec2bert


def test_feature_projection_shapes():
    config = Wav2Vec2BertConfig()
    proj = wav2vec2bert.Wav2Vec2BertFeatureProjection(config)
    batch, seq_len = 2, 8
    inputs = torch.randn(batch, seq_len, config.feature_projection_input_dim)
    projected, normed = proj(inputs)

    assert projected.shape == (batch, seq_len, config.hidden_size)

    assert normed.shape == (batch, seq_len, config.feature_projection_input_dim)


def test_encoder_forward_shape():
    config = Wav2Vec2BertConfig(
        num_hidden_layers=2, position_embeddings_type="relative_key", layerdrop=0.0
    )

    model = wav2vec2bert.Wav2Vec2BertModel(config)

    batch, seq_len = 1, 16

    inputs = torch.randn(batch, seq_len, config.feature_projection_input_dim)

    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

    outputs = model(inputs, attention_mask=attention_mask)

    assert outputs.shape == (batch, seq_len, config.hidden_size)


def test_remap_hf_state_dict_fuses_qkv():
    q_weight = torch.randn(4, 4)

    k_weight = torch.randn(4, 4)

    v_weight = torch.randn(4, 4)

    q_bias = torch.randn(4)

    k_bias = torch.randn(4)

    v_bias = torch.randn(4)

    state_dict = {
        "encoder.layers.0.self_attn.linear_q.weight": q_weight,
        "encoder.layers.0.self_attn.linear_k.weight": k_weight,
        "encoder.layers.0.self_attn.linear_v.weight": v_weight,
        "encoder.layers.0.self_attn.linear_q.bias": q_bias,
        "encoder.layers.0.self_attn.linear_k.bias": k_bias,
        "encoder.layers.0.self_attn.linear_v.bias": v_bias,
        "encoder.layers.0.self_attn.linear_out.weight": torch.randn(4, 4),
        "encoder.layers.0.self_attn.linear_out.bias": torch.randn(4),
    }

    remapped = wav2vec2bert.remap_hf_state_dict_wav2vec2bert(state_dict)

    fused_w = remapped["encoder.layers.0.self_attn.qkv_proj.weight"]

    fused_b = remapped["encoder.layers.0.self_attn.qkv_proj.bias"]

    assert torch.equal(fused_w, torch.cat([q_weight, k_weight, v_weight], dim=0))

    assert torch.equal(fused_b, torch.cat([q_bias, k_bias, v_bias], dim=0))
