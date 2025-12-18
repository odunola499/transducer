import torch
from torch.nn import functional as F
from torch import nn
from typing import Optional
from torch.nn.attention import SDPBackend


def _flash_status(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
):
    if query.device.type != "cuda":
        return False, "flash attention requires CUDA; falling back to non-flash kernel"
    if not torch.backends.cuda.is_built():
        return False, "CUDA is not built with PyTorch; falling back to non-flash kernel"
    if attention_mask is not None:
        return False, "attention mask provided; flash kernel will fall back to math implementation"
    if not torch.backends.cuda.sdp_kernel.is_flash_available(query, key, value):
        return False, "flash kernel not available for input dtype/shape; using fallback"
    return True, ""

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def flash_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    can_flash, reason = _flash_status(query, key, value)
    if not can_flash:
        print(f"[flash_attention] Falling back: {reason}")

    with torch.nn.attention.sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=False,
            scale=scaling,
        )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None

ATTN_FUNCTIONS = {
    'flash_attn': flash_attention_forward,
    'sdpa': sdpa_attention_forward,
    'eager': eager_attention_forward
}

if __name__ == "__main__":
    query = torch.randn(2, 6, 4, 8)
    key = torch.randn(2, 6, 4, 8)
    value = torch.randn(2, 6, 4, 8)

    batch_size = query.shape[0]
    seq_len = query.shape[-2]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device))[None, None, :, :].expand(
        batch_size, 1, seq_len, seq_len
    )
    causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float).min

    module = nn.Module()
    module.num_key_value_groups = 1

    output, _ = eager_attention_forward(
        module, query, key, value, attention_mask=causal_mask, scaling=1
    )
    sdpa_output, _ = sdpa_attention_forward(
        module, query, key, value, attention_mask=causal_mask, scaling=1
    )
    print(output.shape)
    print(sdpa_output.shape)
    result = torch.allclose(output, sdpa_output)
    print(result)
