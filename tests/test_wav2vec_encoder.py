import torch

from transducer.config import Wav2VecSmallConfig
from transducer.encoder import wav2vec


def _expected_conv_length(config: Wav2VecSmallConfig, input_length: int) -> int:
    """Compute sequence length after successive 1D conv layers."""
    length = input_length
    for kernel, stride in zip(config.conv_kernel, config.conv_stride):
        length = (length - kernel) // stride + 1
    return length


def test_feature_encoder_shapes():
    """FeatureEncoder should produce expected channel count and length."""
    config = Wav2VecSmallConfig()
    encoder = wav2vec.FeatureEncoder(config)
    batch, input_length = 2, 400
    inputs = torch.randn(batch, input_length)

    output = encoder(inputs)

    expected_length = _expected_conv_length(config, input_length)
    assert output.shape == (batch, config.conv_dim[-1], expected_length)


def test_wav2vec_model_forward_shape():
    """Full model returns hidden states with expected hidden size and sequence length."""
    config = Wav2VecSmallConfig()
    model = wav2vec.Wav2VecModel(config)
    batch, input_length = 2, 3200
    inputs = torch.randn(batch, input_length)

    hidden_states = model(inputs)

    expected_length = _expected_conv_length(config, input_length)
    assert hidden_states.shape == (batch, expected_length, config.hidden_size)


def test_attention_dispatch(monkeypatch):
    """Attention layer should call the configured attention implementation."""
    config = Wav2VecSmallConfig(attn_impl="eager")
    attention = wav2vec.Attention(config)
    batch, seq_len = 1, 3
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)

    called = {}

    def dummy_attention(module, query, key, value, attention_mask, scaling, dropout):
        called["ran"] = True
        return torch.zeros_like(query), None

    monkeypatch.setitem(wav2vec.ATTN_FUNCTIONS, "eager", dummy_attention)
    attention(hidden_states)

    assert called.get("ran") is True


def test_spec_augment_masks_values():
    """Spec augment should introduce zeros when masking probabilities are positive."""
    torch.manual_seed(0)
    hidden_states = torch.ones(1, 4, 4)

    augmented = wav2vec.spec_augment(
        hidden_states.clone(),
        time_mask_prob=1.0,
        time_mask_width=1,
        freq_mask_prob=0.5,
        freq_mask_width=1,
    )

    assert augmented.shape == hidden_states.shape
    assert (augmented == 0).any()
    assert not torch.equal(augmented, torch.zeros_like(augmented))


def test_remap_hf_state_dict_strips_prefix_and_fuses_qkv():
    """remap_hf_state_dict drops encoder prefix and fuses q/k/v projections."""
    q_weight = torch.randn(4, 4)
    k_weight = torch.randn(4, 4)
    v_weight = torch.randn(4, 4)
    q_bias = torch.randn(4)
    k_bias = torch.randn(4)
    v_bias = torch.randn(4)
    state_dict = {
        "encoder.layers.0.attention.q_proj.weight": q_weight,
        "encoder.layers.0.attention.k_proj.weight": k_weight,
        "encoder.layers.0.attention.v_proj.weight": v_weight,
        "encoder.layers.0.attention.q_proj.bias": q_bias,
        "encoder.layers.0.attention.k_proj.bias": k_bias,
        "encoder.layers.0.attention.v_proj.bias": v_bias,
        "encoder.layer_norm.weight": torch.tensor([1.0]),
        "masked_spec_embed": torch.tensor([3.0]),
    }

    remapped = wav2vec.remap_hf_state_dict(state_dict)

    fused_weight = remapped["layers.0.attention.qkv_proj.weight"]
    fused_bias = remapped["layers.0.attention.qkv_proj.bias"]
    assert torch.equal(fused_weight, torch.cat([q_weight, k_weight, v_weight], dim=0))
    assert torch.equal(fused_bias, torch.cat([q_bias, k_bias, v_bias], dim=0))
    assert "encoder.layer_norm.weight" not in remapped
    assert "masked_spec_embed" not in remapped
    assert "layer_norm.weight" in remapped


def test_attention_qkv_projection_chunks_correctly(monkeypatch):
    """qkv projection should split into query/key/value in head-major form."""
    config = Wav2VecSmallConfig(hidden_size=4, num_attention_heads=2, attn_impl="eager")
    attention = wav2vec.Attention(config)
    with torch.no_grad():
        eye = torch.eye(config.hidden_size)
        attention.qkv_proj.weight.zero_()
        attention.qkv_proj.bias.zero_()
        attention.qkv_proj.weight[: config.hidden_size] = eye
        attention.qkv_proj.weight[config.hidden_size : 2 * config.hidden_size] = eye
        attention.qkv_proj.weight[2 * config.hidden_size :] = eye

    batch, seq_len = 1, 2
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)

    def assert_qkv(module, query, key, value, attention_mask, scaling, dropout):
        expected = hidden_states.view(batch, seq_len, config.num_attention_heads, -1).permute(0, 2, 1, 3)
        assert torch.allclose(query, expected)
        assert torch.allclose(key, expected)
        assert torch.allclose(value, expected)
        return torch.zeros_like(query), None

    monkeypatch.setitem(wav2vec.ATTN_FUNCTIONS, "eager", assert_qkv)
    attention(hidden_states)
