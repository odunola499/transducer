import torch
from torch import Tensor, nn

from transducer.commons import Joiner
from transducer.models.config import DecoderConfig


class SimpleJoiner(Joiner):
    def __init__(self, vocab_size: int, encoder_dim: int, config: DecoderConfig):
        super().__init__()
        joint_dim = config.joint_dim
        predictor_dim = config.pred_dim

        if joint_dim != encoder_dim:
            self.enc_proj = nn.Linear(encoder_dim, joint_dim, bias=False)
        else:
            self.enc_proj = nn.Identity()

        if joint_dim != predictor_dim:
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


class NemoJoiner(Joiner):
    def __init__(
        self,
        encoder_dim,
        pred_dim,
        joint_dim,
        dropout: float = 0.2,
        num_classes: int = 1024,
    ):
        super().__init__()

        self._vocab_size = num_classes
        self._num_classes = num_classes + 1
        self.encoder_dim = encoder_dim
        self.pred_dim = pred_dim
        self.joint_dim = joint_dim
        self.dropout = dropout

        self.pred = nn.Linear(pred_dim, joint_dim)
        self.enc = nn.Linear(encoder_dim, joint_dim)
        activation = nn.ReLU(inplace=True)

        layers = (
            [activation]
            + [nn.Dropout(p=dropout)]
            + [nn.Linear(joint_dim, num_classes + 1)]
        )
        self.joint_net = nn.Sequential(*layers)

    def forward(
        self,
        encoder_output: Tensor,  # B, d1, T
        predictor_output: Tensor,  # B, d2, U
    ):
        encoder_output = self.pred(encoder_output.transpose(1, 2)).unsqueeze(2)
        decoder_output = self.enc(predictor_output.transpose(1, 2)).unsqueeze(1)
        joined = encoder_output + decoder_output
        joined = self.joint_net(joined)
        return joined
