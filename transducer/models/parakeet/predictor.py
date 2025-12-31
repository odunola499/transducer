from typing import Optional

import torch
from torch import Tensor, nn

from transducer.commons import Predictor
from transducer.commons.config import DecoderConfig


class NemoPredictor(Predictor):
    class LSTMDropout(nn.Module):
        def __init__(
            self, input_size: int, hidden_size: int, num_layers: int, dropout: float
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.dropout = nn.Dropout(p=dropout) if dropout else None

        def forward(self, x: Tensor, h=None):
            x, h = self.lstm(x, h)
            if self.dropout:
                x = self.dropout(x)
            return x, h

    def __init__(
        self,
        config: DecoderConfig,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        pred_dim = config.pred_dim
        num_layers = config.num_layers
        blank_idx = vocab_size
        dropout = config.dropout
        self.pred_dim = pred_dim
        self.num_layers = num_layers
        self.blank_idx = blank_idx

        layers = nn.ModuleDict(
            {
                "embed": nn.Embedding(vocab_size + 1, pred_dim, padding_idx=blank_idx),
                "dec_rnn": self.LSTMDropout(
                    input_size=pred_dim,
                    hidden_size=pred_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                ),
            }
        )
        self.prediction = layers

    def init_state(self, batch_size: int):
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype
        return (
            torch.zeros(
                self.num_layers,
                batch_size,
                self.pred_dim,
                dtype=dtype,
                device=device,
            ),
            torch.zeros(
                self.num_layers,
                batch_size,
                self.pred_dim,
                dtype=dtype,
                device=device,
            ),
        )

    def step(self, input_ids: Tensor, state: Optional[Tensor] = None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (B,) or (B, 1)")

        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        input_ids = input_ids.to(device=device)
        y = self.prediction["embed"](input_ids).to(dtype=dtype)

        if state is None:
            state = self.init_state(y.size(0))

        y = y.transpose(0, 1)
        g, hidden = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)
        return g[:, -1, :], hidden

    def forward(
        self, targets: Tensor, target_length: Optional[Tensor] = None, states=None
    ):
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        if targets.dim() != 2:
            raise ValueError("targets must have shape (B, U)")

        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        targets = targets.to(device=device)
        y = self.prediction["embed"](targets).to(dtype=dtype)

        bsz, seq_len, hidden = y.shape
        start = torch.zeros((bsz, 1, hidden), dtype=dtype, device=device)
        y = torch.concat([start, y], dim=1).contiguous()

        if states is None:
            states = self.init_state(bsz)

        y = y.transpose(0, 1)
        g, hidden = self.prediction["dec_rnn"](y, states)
        g = g.transpose(0, 1)

        if target_length is None:
            target_length = torch.full((bsz,), seq_len, dtype=torch.long, device=device)

        return g.transpose(1, 2), target_length, hidden
