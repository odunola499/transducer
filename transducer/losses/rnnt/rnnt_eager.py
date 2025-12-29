from typing import Literal

import torch
from torch import Tensor

from transducer.commons import Loss


class EagerRNNTLoss(Loss):
    # Ported from https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/losses/rnnt_pytorch.py#L24
    def __init__(self, blank_id: int, reduction: Literal["mean", "sum", "none"]):
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction

    def forward(
        self, lattice: Tensor, labels: Tensor, lattice_lens: Tensor, label_lens: Tensor
    ):
        losses = -(self.compute_forward_prob(lattice, labels, lattice_lens, label_lens))
        if self.reduction == "mean":
            return torch.mean(losses)
        elif self.reduction == "sum":
            return torch.sum(losses)
        elif self.reduction == "none":
            return losses
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

    def compute_forward_prob(
        self, lattice: Tensor, labels: Tensor, lattice_lens: Tensor, label_lens: Tensor
    ):
        B, T, U, _ = lattice.shape

        log_alpha = torch.zeros(B, T, U)
        log_alpha = log_alpha.to(lattice.device)

        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        log_alpha[:, t, u] = 0.0
                    else:
                        log_alpha[:, t, u] = (
                            log_alpha[:, t - 1, u] + lattice[:, t - 1, u, self.blank_id]
                        )
                else:
                    if t == 0:
                        gathered = torch.gather(
                            lattice[:, t, u - 1],
                            dim=1,
                            index=labels[:, u - 1].view(-1, 1),
                        ).reshape(-1)
                        log_alpha[:, t, u] = log_alpha[:, t, u - 1] + gathered

                    else:
                        log_alpha[:, t, u] = torch.logsumexp(
                            torch.stack(
                                [
                                    log_alpha[:, t - 1, u]
                                    + lattice[:, t - 1, u, self.blank_id],
                                    log_alpha[:, t, u - 1]
                                    + torch.gather(
                                        lattice[:, t, u - 1],
                                        dim=1,
                                        index=labels[:, u - 1].view(-1, 1),
                                    ).reshape(-1),
                                ]
                            ),
                            dim=0,
                        )

        log_probs = []
        for b in range(B):
            to_append = (
                log_alpha[b, lattice_lens[b] - 1, label_lens[b]]
                + lattice[b, lattice_lens[b] - 1, label_lens[b], self.blank_id]
            )
            log_probs.append(to_append)
        log_probs = torch.stack(log_probs)
        return log_probs


if __name__ == "__main__":
    rnnt_loss = EagerRNNTLoss(blank_id=0, reduction="mean")
    B, T, U, vocab_size = 4, 10, 7, 16
    lattice = torch.randn(B, T, U, vocab_size)
    lattice = torch.log_softmax(lattice, dim=-1)
    labels = torch.randint(0, vocab_size, (B, U))
    lattice_lens = torch.randint(2, T, (B,))
    label_lens = torch.randint(2, U, (B,))

    print(lattice_lens, label_lens)
    output = rnnt_loss(lattice, labels, lattice_lens, label_lens)
    print(output)
