from __future__ import annotations

from typing import Optional

import torch
from torch.autograd import Function

from transducer.commons import Loss
from transducer.kernels.triton_kernels.rnnt import rnnt_loss_triton


class _RNNTTriton(Function):
    @staticmethod
    def forward(
        ctx,
        acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
        blank: int,
        reduction: str,
        fastemit_lambda: float,
        clamp: float,
    ):
        _log_probs, _alphas, _betas, grads, loglikes = rnnt_loss_triton(
            acts, labels, act_lens, label_lens, blank, fastemit_lambda, clamp
        )
        costs = -loglikes * (1.0 + fastemit_lambda)
        ctx.reduction = reduction
        ctx.save_for_backward(grads)
        ctx.blank = blank
        ctx.fastemit_lambda = fastemit_lambda
        ctx.clamp = clamp
        ctx.batch_size = acts.shape[0]

        if reduction == "sum":
            return costs.sum().unsqueeze(-1)
        if reduction == "mean":
            return costs.mean().unsqueeze(-1)
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        (grads,) = ctx.saved_tensors
        reduction = ctx.reduction

        if reduction in {"sum", "mean"}:
            scale = grad_output
            if reduction == "mean":
                scale = scale / ctx.batch_size
            grads = grads * scale
        else:
            grads = grads * grad_output.view(-1, 1, 1, 1)

        return grads, None, None, None, None, None, None, None


class RNNTTritonLoss(Loss):
    def __init__(
        self,
        blank_id: int,
        reduction: str = "mean",
        fastemit_lambda: float = 0.0,
        clamp: float = 0.0,
    ):
        super().__init__()
        self.blank = blank_id
        self.reduction = reduction
        self.fastemit_lambda = fastemit_lambda
        self.clamp = float(clamp) if clamp > 0 else 0.0

    def forward(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: Optional[torch.Tensor],
        label_lens: Optional[torch.Tensor],
    ):
        if act_lens is None:
            batch_size, num_frames = acts.shape[:2]
            act_lens = torch.full(
                (batch_size,), num_frames, device=acts.device, dtype=torch.long
            )
        return _RNNTTriton.apply(
            acts,
            labels,
            act_lens,
            label_lens,
            self.blank,
            self.reduction,
            self.fastemit_lambda,
            self.clamp,
        )
