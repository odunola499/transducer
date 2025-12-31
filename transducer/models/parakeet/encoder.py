import torch
from torch import Tensor, nn

from transducer.models.modules import ConvSubsampling, FeedForward, PositionalEncoding
from transducer.models.modules.attention import FastConformerAttention
from transducer.models.modules.causal_convs import ConformerConvolution
from transducer.models.parakeet.config import FastConformerConfig, StreamingConfig


class ParakeetConvolutionModule(ConformerConvolution):
    pass


class ParakeetAttention(FastConformerAttention):
    pass


class ConformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        d_ff: int,
        num_heads: int,
        dropout: float,
        dropout_att: float,
        conv_kernel_size: int = 31,
        use_bias: bool = True,
        attn_impl: str = "eager",
    ):
        super().__init__()
        self.d_model = hidden_size
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.dropout_att = dropout_att
        self.use_bias = use_bias

        self.norm_feed_forward1 = nn.LayerNorm(hidden_size)
        self.feed_forward1 = FeedForward(
            hidden_size,
            d_ff,
            dropout=dropout,
            use_bias=use_bias,
        )
        self.norm_conv = nn.LayerNorm(hidden_size)
        self.conv = ParakeetConvolutionModule(
            hidden_size=hidden_size,
            kernel_size=conv_kernel_size,
        )

        self.norm_self_att = nn.LayerNorm(hidden_size)
        self.self_attn = ParakeetAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout_att,
            use_bias=use_bias,
            attn_impl=attn_impl,
        )
        self.norm_feed_forward2 = nn.LayerNorm(hidden_size)
        self.feed_forward2 = FeedForward(
            hidden_size=hidden_size,
            d_ff=d_ff,
            dropout=dropout,
            use_bias=use_bias,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, x, pos_emb=None, pad_mask=None, att_mask=None, cache=None):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual += self.dropout(x) * 0.5

        x = self.norm_self_att(residual)
        x = self.self_attn(x, pos_emb, att_mask=att_mask)
        residual += self.dropout(x)

        x = self.norm_conv(residual)
        if cache:
            x, _ = self.conv(x, pad_mask=pad_mask, cache=cache)
        else:
            x = self.conv(x, pad_mask=pad_mask)
        residual += self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual += self.dropout(x) * 0.5
        x = self.norm_out(residual)
        return x


class ConformerEncoder(nn.Module):
    def __init__(self, config: FastConformerConfig):
        super().__init__()
        self.config = config
        self.stream_config = StreamingConfig()

        d_ff = config.hidden_size * config.ff_expansion_factor
        self.d_model = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_heads
        self._feat_in = config.feat_in
        self.subsampling_factor = config.subsampling_factor

        self.pre_encode = ConvSubsampling(config)
        self._feat_out = config.hidden_size

        self.pos_emb_max_len = config.pos_emb_max_len
        self.pos_enc = PositionalEncoding(
            hidden_size=config.hidden_size,
            dropout=config.dropout_pre_encoder,
            max_len=self.pos_emb_max_len,
            dropout_rate_emb=config.dropout_emb,
        )
        self.pos_enc.extend_pe(
            self.pos_emb_max_len, device=next(self.parameters()).device
        )

        self.layers = nn.ModuleList()
        attn_impl = config.attn_impl
        if attn_impl == "math":
            attn_impl = "eager"
        elif attn_impl not in {"eager", "sdpa"}:
            attn_impl = "eager"
        for _i in range(config.num_hidden_layers):
            layer = ConformerLayer(
                hidden_size=config.hidden_size,
                d_ff=d_ff,
                num_heads=config.num_heads,
                dropout=config.dropout,
                dropout_att=config.dropout_att,
                conv_kernel_size=config.conv_kernel_size,
                use_bias=config.use_bias,
                attn_impl=attn_impl,
            )
            self.layers.append(layer)

        self.att_context_size = config.att_context_size
        self.att_context_style = config.att_context_style

    def forward(self, x, length: Tensor = None, return_lengths: bool = False):
        if length is None:
            length = x.new_full(
                (x.size(0),), x.size(-1), dtype=torch.int64, device=x.device
            )

        x = x.transpose(1, 2)
        x, length = self.pre_encode(x, lengths=length)
        x, pos_emb = self.pos_enc(x=x)

        pad_mask = None
        att_mask = None
        if length is not None:
            pad_mask, att_mask = self._create_masks(
                padding_length=length,
                max_audio_length=x.shape[1],
                device=x.device,
            )

        for layer in self.layers:
            x = layer(x=x, pos_emb=pos_emb, pad_mask=pad_mask, att_mask=att_mask)

        x = x.transpose(1, 2)

        if return_lengths:
            return x, length
        return x

    def _normalize_att_context_size(self):
        att_context_size = self.att_context_size
        if isinstance(att_context_size, (list, tuple)) and len(att_context_size) > 0:
            if isinstance(att_context_size[0], (list, tuple)):
                att_context_size = att_context_size[0]
        if not isinstance(att_context_size, (list, tuple)) or len(att_context_size) != 2:
            att_context_size = [-1, -1]
        return list(att_context_size)

    def _create_masks(self, padding_length, max_audio_length, device):
        att_context_size = self._normalize_att_context_size()
        att_mask = torch.ones(
            1, max_audio_length, max_audio_length, dtype=torch.bool, device=device
        )
        if self.att_context_style == "regular":
            if att_context_size[0] >= 0:
                att_mask = att_mask.triu(diagonal=-att_context_size[0])
            if att_context_size[1] >= 0:
                att_mask = att_mask.tril(diagonal=att_context_size[1])
        elif self.att_context_style == "chunked_limited":
            if att_context_size[1] == -1:
                if att_context_size[0] >= 0:
                    att_mask = att_mask.triu(diagonal=-att_context_size[0])
            else:
                chunk_size = att_context_size[1] + 1
                left_chunks_num = (
                    att_context_size[0] // chunk_size if att_context_size[0] >= 0 else 10000
                )
                chunk_idx = torch.arange(
                    0, max_audio_length, dtype=torch.int, device=device
                )
                chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                chunked_limited_mask = torch.logical_and(
                    torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
                )
                att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))

        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(
            pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2)
        )
        att_mask = torch.logical_and(
            pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device)
        )
        att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask


FastConformerEncoder = ConformerEncoder

__all__ = [
    "ConformerEncoder",
    "FastConformerEncoder",
    "ParakeetAttention",
    "ParakeetConvolutionModule",
]


if __name__ == "__main__":
    config = FastConformerConfig()
    encoder = ConformerEncoder(config)
    module = encoder
    num_params = sum([p.numel() for p in module.parameters()])
    print(num_params)
