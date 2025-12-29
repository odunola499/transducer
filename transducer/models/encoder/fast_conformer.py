import torch
from torch import Tensor, nn

from transducer import config
from transducer.models.config import FastConformerConfig
from transducer.models.modules import (
    ConformerConvolution,
    ConvSubsampling,
    FastConformerAttention,
    FeedForward,
    PositionalEncoding,
)


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
        self.conv = ConformerConvolution(
            hidden_size=hidden_size,
            kernel_size=conv_kernel_size,
        )

        self.norm_self_att = nn.LayerNorm(hidden_size)
        self.self_attn = FastConformerAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout_att,
            use_bias=use_bias,
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

    def forward(self, x, pos_emb=None, pad_mask=None, cache=None):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual += self.dropout(x) * 0.5

        x = self.norm_self_att(residual)
        x = self.self_attn(x, pos_emb)
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
        for _i in range(config.num_hidden_layers):
            layer = ConformerLayer(
                hidden_size=config.hidden_size,
                d_ff=d_ff,
                num_heads=config.num_heads,
                dropout=config.dropout,
                dropout_att=config.dropout_att,
                conv_kernel_size=config.conv_kernel_size,
                use_bias=config.use_bias,
            )
            self.layers.append(layer)

        self.att_context_size = config.att_context_size

    def forward(self, x, length: Tensor = None, return_lengths: bool = False):
        if length is None:
            length = x.new_full(
                (x.size(0),), x.size(-1), dtype=torch.int64, device=x.device
            )

        x = x.transpose(1, 2)
        x, length = self.pre_encode(x, lengths=length)
        x, pos_emb = self.pos_enc(x=x)

        pad_mask = None
        if length is not None:
            max_len = x.size(1)
            pad_mask = torch.arange(0, x.shape[1], device=x.device).expand(
            length.size(0), -1
        ) < length.unsqueeze(-1)
            pad_mask = ~pad_mask

        for layer in self.layers:
            x = layer(x=x, pos_emb=pos_emb, pad_mask=pad_mask)

        x = x.transpose(1, 2)

        if return_lengths:
            return x, length
        return x


if __name__ == "__main__":
    config = FastConformerConfig()
    encoder = ConformerEncoder(config)
    module = encoder
    num_params = sum([p.numel() for p in module.parameters()])
    print(num_params)
