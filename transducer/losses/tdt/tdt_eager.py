from typing import List, Literal

import torch
from torch import Tensor

from transducer.commons import Loss

# https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/losses/rnnt_pytorch.py


class TDTLossPytorch(Loss):
    def __init__(
        self,
        blank_id: int,
        durations: List[int] = [],
        reduction: Literal["mean", "sum", "none"] = "sum",
        sigma: float = 0.0,
    ):
        super().__init__()

        self.blank = blank_id

        self.durations = durations

        self.n_durations = len(durations)

        self.reduction = reduction

        self.sigma = sigma

    def forward(self, acts: Tensor, labels: Tensor, act_lens: Tensor, label_lens: Tensor):
        label_acts = acts[:, :, :, : -self.n_durations]

        duration_acts = acts[:, :, :, -self.n_durations :]

        label_acts = torch.log_softmax(label_acts, -1) - self.sigma

        duration_acts = torch.log_softmax(duration_acts, -1)

        forward_logprob, _ = self.compute_forward_prob(
            label_acts, duration_acts, labels, act_lens, label_lens
        )

        losses = -forward_logprob

        if self.reduction == "mean":
            return torch.mean(losses)

        elif self.reduction == "sum":
            return torch.sum(losses)

        elif self.reduction == "none":
            return losses

        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        return losses

    def logsumexp(self, a, b):
        ret = torch.logsumexp(torch.stack([a, b]), dim=0)

        return ret

    def compute_forward_prob(self, acts, duration_acts, labels, act_lens, label_lens):
        B, T, U, _ = acts.shape

        log_alpha = torch.zeros(B, T, U)

        for b in range(B):
            for t in range(T):
                for u in range(U):
                    if u == 0:
                        if t == 0:
                            log_alpha[b, t, u] = 0.0

                        else:
                            log_alpha[b, t, u] = -1000.0

                            for n, l in enumerate(self.durations):
                                if t - l >= 0 and l > 0:
                                    tmp = (
                                        log_alpha[b, t - l, u]
                                        + acts[b, t - l, u, self.blank]
                                        + duration_acts[b, t - l, u, n]
                                    )

                                    log_alpha[b, t, u] = self.logsumexp(
                                        tmp, 1.0 * log_alpha[b, t, u]
                                    )

                    else:
                        log_alpha[b, t, u] = -1000.0

                        for n, l in enumerate(self.durations):
                            if t - l >= 0:
                                if l > 0:
                                    tmp = (
                                        log_alpha[b, t - l, u]
                                        + acts[b, t - l, u, self.blank]
                                        + duration_acts[b, t - l, u, n]
                                    )

                                    log_alpha[b, t, u] = self.logsumexp(
                                        tmp, 1.0 * log_alpha[b, t, u]
                                    )

                                tmp = (
                                    log_alpha[b, t - l, u - 1]
                                    + acts[b, t - l, u - 1, labels[b, u - 1]]
                                    + duration_acts[b, t - l, u - 1, n]
                                )

                                log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

        log_probs = []

        for b in range(B):
            tt = torch.Tensor([-1000.0])[0]

            for n, l in enumerate(self.durations):
                if act_lens[b] - l >= 0 and l > 0:
                    bb = (
                        log_alpha[b, act_lens[b] - l, label_lens[b]]
                        + acts[b, act_lens[b] - l, label_lens[b], self.blank]
                        + duration_acts[b, act_lens[b] - l, label_lens[b], n]
                    )

                    tt = self.logsumexp(bb, 1.0 * tt)

            log_probs.append(tt)

        log_prob = torch.stack(log_probs)

        return log_prob, log_alpha


if __name__ == "__main__":
    rnnt_loss = TDTLossPytorch(blank_id=0, reduction="mean")

    B, T, U, vocab_size = 4, 10, 7, 16

    lattice = torch.randn(B, T, U, vocab_size)

    lattice = torch.log_softmax(lattice, dim=-1)

    labels = torch.randint(0, vocab_size, (B, U))

    lattice_lens = torch.randint(2, T, (B,))

    label_lens = torch.randint(2, U, (B,))

    print(lattice_lens, label_lens)

    output = rnnt_loss(lattice, labels, lattice_lens, label_lens)

    print(output)
