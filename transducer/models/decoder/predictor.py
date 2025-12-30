from typing import Optional

import torch
from torch import Tensor, nn

from transducer.commons import Predictor
from transducer.models.config import DecoderConfig


class RNNPredictor(Predictor):
    def __init__(self, config: DecoderConfig, vocab_size: Optional[int] = None):
        super().__init__()
        effective_vocab_size = (
            vocab_size if vocab_size is not None else config.vocab_size
        )
        self.text_embed = nn.Embedding(effective_vocab_size, config.embed_dim)
        self.rnn_type = config.rnn_type
        if "gru" in config.rnn_type:
            self.rnn = nn.GRU(
                input_size=config.embed_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=0.0,
            )
        elif "lstm" in config.rnn_type:
            self.rnn = nn.LSTM(
                input_size=config.embed_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=0.0,
            )
        else:
            raise NotImplementedError

        self.proj = nn.Linear(config.hidden_dim, config.pred_dim)
        self.config = config
        self.vocab_size = effective_vocab_size

    def forward(self, input_ids: Tensor, state: Optional[Tensor] = None):
        input_embeds = self.text_embed(input_ids)
        out, new_state = self.rnn(input_embeds, state)
        output = self.proj(out)
        return output, new_state

    @torch.no_grad()
    def step(self, input_ids: Tensor, state: Optional[Tensor] = None):
        input_embeds = self.text_embed(input_ids).unsqueeze(1)
        output, new_state = self.rnn(input_embeds, state)
        output = self.proj(output.squeeze(1))
        return output, new_state

    def init_state(self, batch_size: int = 1):
        hidden_size = self.rnn.hidden_size
        if "lstm" in self.rnn_type:
            h0 = torch.zeros(self.config.num_layers, batch_size, hidden_size)
            c0 = torch.zeros(self.config.num_layers, batch_size, hidden_size)
            return (h0, c0)
        return torch.zeros(self.config.num_layers, batch_size, hidden_size)


class NemoPredictor(Predictor):
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

        layers = nn.ModuleDict(
            {
                "embed": nn.Embedding(vocab_size + 1, pred_dim, padding_idx=blank_idx),
                "dec_rnn": nn.LSTM(
                    input_size=pred_dim,
                    hidden_size=pred_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                ),
            }
        )
        self.prediction = layers
        self.pred_dim = pred_dim
        self.num_layers = num_layers

    def init_state(self, y: Tensor):
        state = (
            torch.zeros(
                self.num_layers,
                y.size(0),
                self.pred_dim,
                dtype=y.dtype,
                device=y.device,
            ),
            torch.zeros(
                self.num_layers,
                y.size(0),
                self.pred_dim,
                dtype=y.dtype,
                device=y.device,
            ),
        )
        return state

    def step(
        self,
        y: Tensor,
        state: Optional[Tensor] = None,
    ):
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        B, U, H = y.shape
        start = torch.zeros((B, 1, H), dtype=dtype, device=device)
        y = torch.concat([start, y], dim=1).contiguous()

        if state is None:
            state = self.init_state(y)

        y = y.transpose(0, 1)
        g, hidden = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)

        del y, start, state
        return g, hidden

    def forward(self, targets: Tensor, target_length: Tensor, states=None):
        g, states = self.step(targets, state=states)
        return g, target_length, states
