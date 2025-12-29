import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend

INF_VAL = 10000.0


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
        return (
            False,
            "attention mask provided; flash kernel will fall back to math implementation",
        )
    if not torch.backends.cuda.sdp_kernel.is_flash_available(query, key, value):
        return False, "flash kernel not available for input dtype/shape; using fallback"
    return True, ""


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0
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
    scaling: Optional[float] = None,
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
    "flash_attn": flash_attention_forward,
    "sdpa": sdpa_attention_forward,
    "eager": eager_attention_forward,
}


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout=0.1,
        max_len: int = 5000,
        dropout_rate_emb: float = 0.0,
        xscale:bool = False
    ):
        super().__init__()
        self.d_model = hidden_size
        self.xscale = xscale
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.dropout_emb = nn.Dropout(p=dropout_rate_emb)

    def create_pe(self, positions):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(
                0, self.d_model, 2, dtype=torch.float32, device=positions.device
            )
            * -(math.log(INF_VAL) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(positions.device)
        self.register_buffer("pe", pe, persistent=False)

    def extend_pe(self, length, device):
        needed_size = 2 * length - 1
        if hasattr(self, "pe") and self.pe.size(1) >= needed_size:
            return
        positions = torch.arange(
            length - 1, -length, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.create_pe(positions=positions)

    def forward(self, x: Tensor, cache_len=0):
        if self.xscale:
            x = x * self.xscale

        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


class FastConformerAttention(nn.Module):
    def __init__(
        self, num_heads, hidden_size: int, dropout: float, use_bias: bool = True, attn_impl = 'eager'
    ):
        super().__init__()
        self.d_k = hidden_size // num_heads
        self.s_d_k = math.sqrt(self.d_k)
        self.num_heads = num_heads
        self.dropout = dropout
        self.linear_q = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_out = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        self.linear_pos = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.num_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.num_heads, self.d_k))
        self.attn_impl = attn_impl

    def rel_shift(self, x: Tensor):
        B, H, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0))
        x = x.view(B, H, -1, qlen)[:, :, 1:].view(B, H, qlen, pos_len)
        return x

    def forward(self, x: Tensor, pos_emb: Tensor):
        """
        query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            pos_emb (torch.Tensor) : (batch, time1, size)
        """
        B, T = x.shape[:2]

        query = self.linear_q(x).view(B, -1, self.num_heads, self.d_k)
        key = self.linear_k(x).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(x).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        p = (
            self.linear_pos(pos_emb)
            .view(pos_emb.shape[0], -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        n_batch = x.shape[0]

        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        scale_factor = 1 / math.sqrt(q_with_bias_u.size(-1))
        matrix_bd = matrix_bd[:, :, :, : key.size(-2)] * scale_factor

        if self.attn_impl == 'eager':

            output, _ = eager_attention_forward(
                self,
                query=q_with_bias_u,
                key=key,
                value=value,
                attention_mask=matrix_bd,
                scaling=1/self.s_d_k,
                dropout=self.dropout if self.training else 0.0,
            )
        else:
            output = sdpa_attention_forward(
                self,
                query = q_with_bias_u,
                key = key,
                value = value,
                attention_mask = matrix_bd,
                scaling = None,
                dropout=self.dropout if self.training else 0.0,
            )

        output = output.reshape(n_batch, -1, self.num_heads * self.d_k)
        return self.linear_out(output)


if __name__ == "__main__":
    query = torch.randn(2, 6, 8)
    pos = torch.randn(2, 6, 8)

    attn = FastConformerAttention(
        num_heads=4,
        hidden_size=8,
        dropout=0.1,
    )
    print(attn(query, pos))
