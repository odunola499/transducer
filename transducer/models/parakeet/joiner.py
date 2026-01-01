from torch import Tensor, nn

from transducer.commons import Joiner


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
        if encoder_output.dim() != 3:
            raise ValueError("encoder_output must have shape (B, T, D) or (B, D, T)")
        if encoder_output.shape[-1] == self.encoder_dim:
            pass
        elif encoder_output.shape[1] == self.encoder_dim:
            encoder_output = encoder_output.transpose(1, 2)
        else:
            raise ValueError("encoder_output last dim must match encoder_dim")

        if predictor_output.dim() == 2:
            predictor_output = predictor_output.unsqueeze(1)
        elif predictor_output.dim() != 3:
            raise ValueError(
                "predictor_output must have shape (B, D), (B, U, D), or (B, D, U)"
            )

        if predictor_output.shape[-1] == self.pred_dim:
            pass
        elif predictor_output.shape[1] == self.pred_dim:
            predictor_output = predictor_output.transpose(1, 2)
        else:
            raise ValueError("predictor_output last dim must match pred_dim")

        encoder_output = self.enc(encoder_output).unsqueeze(2)
        decoder_output = self.pred(predictor_output).unsqueeze(1)
        joined = encoder_output + decoder_output
        return self.joint_net(joined)
