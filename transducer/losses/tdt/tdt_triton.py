from __future__ import annotations

from typing import Optional

import torch
from torch.autograd import Function

from transducer.commons import Loss
from transducer.kernels.triton_kernels.tdt import tdt_loss_triton


class _TDTTriton(Function):
    @staticmethod
    def forward(
        ctx,
        label_acts: torch.Tensor,
        duration_acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
        blank: int,
        durations,
        reduction: str,
        fastemit_lambda: float,
        clamp: float,
        sigma: float,
        omega: float,
    ):
        (
            _log_probs,
            _alphas,
            _betas,
            label_grads,
            duration_grads,
            loglikes,
        ) = tdt_loss_triton(
            label_acts,
            duration_acts,
            labels,
            act_lens,
            label_lens,
            blank,
            durations,
            fastemit_lambda,
            clamp,
            sigma,
            omega,
        )
        costs = -loglikes * (1.0 + fastemit_lambda)
        ctx.reduction = reduction
        ctx.save_for_backward(label_grads, duration_grads)
        ctx.batch_size = label_acts.shape[0]

        if reduction == "sum":
            return costs.sum().unsqueeze(-1)
        if reduction == "mean":
            return costs.mean().unsqueeze(-1)
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        label_grads, duration_grads = ctx.saved_tensors
        reduction = ctx.reduction

        if reduction in {"sum", "mean"}:
            scale = grad_output
            if reduction == "mean":
                scale = scale / ctx.batch_size
            label_grads = label_grads * scale
            duration_grads = duration_grads * scale
        else:
            scale = grad_output.view(-1, 1, 1, 1)
            label_grads = label_grads * scale
            duration_grads = duration_grads * scale

        return (
            label_grads,
            duration_grads,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class TDTTritonLoss(Loss):
    def __init__(
        self,
        blank_id: int,
        durations=None,
        reduction: str = "mean",
        fastemit_lambda: float = 0.0,
        clamp: float = 0.0,
        sigma: float = 0.0,
        omega: float = 0.0,
    ):
        super().__init__()
        self.blank = blank_id
        self.durations = durations if durations is not None else []
        self.reduction = reduction
        self.fastemit_lambda = fastemit_lambda
        self.clamp = float(clamp) if clamp > 0 else 0.0
        self.sigma = sigma
        self.omega = omega

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

        label_size = acts.shape[-1] - len(self.durations)
        label_acts, duration_acts = torch.split(
            acts, [label_size, len(self.durations)], dim=-1
        )
        label_acts = label_acts.contiguous()
        duration_acts = torch.nn.functional.log_softmax(
            duration_acts, dim=-1
        ).contiguous()

        return _TDTTriton.apply(
            label_acts,
            duration_acts,
            labels,
            act_lens,
            label_lens,
            self.blank,
            self.durations,
            self.reduction,
            self.fastemit_lambda,
            self.clamp,
            self.sigma,
            self.omega,
        )
