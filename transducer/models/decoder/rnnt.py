from typing import Optional

import torch
from torch import Tensor, nn

from transducer.commons import Joiner, Predictor
from transducer.models.config import DecoderConfig


class RNNTClassicPredictor(Predictor):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.text_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.gru = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.proj = nn.Linear(config.hidden_dim, config.pred_dim)

    def forward(self, input_ids: Tensor, state: Optional[Tensor] = None):
        input_embeds = self.text_embed(input_ids)
        out, new_state = self.gru(input_embeds, state)
        output = self.proj(out)
        return output, state

    @torch.no_grad()
    def step(self, input_ids: Tensor, state: Optional[Tensor] = None):
        input_embeds = self.embed(input_ids).unsqueeze(1)
        output, new_state = self.gru(input_embeds, state)
        output = self.proj(output.squeeze(1))
        return output, new_state


class RNNTJoiner(Joiner):
    def __init__(self, encoder_dim: int, config: DecoderConfig):
        super().__init__()
        joint_dim = config.joint_dim
        predictor_dim = config.pred_dim
        vocab_size = config.vocab_size

        if joint_dim != encoder_dim:
            self.enc_proj = nn.Linear(encoder_dim, joint_dim, bias=False)
        else:
            self.enc_proj = nn.Identity()

        if joint_dim != encoder_dim:
            self.pred_proj = nn.Linear(predictor_dim, joint_dim, bias=False)
        else:
            self.pred_proj = nn.Identity()

        self.bias = nn.Parameter(torch.zeros(joint_dim))
        self.out = nn.Linear(joint_dim, vocab_size)

    def forward(self, encoder_output: Tensor, predictor_output: Tensor):
        enc_proj = self.enc_proj(encoder_output).unsqueeze(2)
        pred_proj = self.pred_proj(predictor_output).unsqueeze(1)

        lattice = enc_proj + pred_proj + self.bias
        x = torch.tanh(lattice)
        logits = self.out(x)
        return logits
