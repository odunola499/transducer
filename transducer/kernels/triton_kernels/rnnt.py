from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def _missing_triton(*_args, **_kwargs):
    raise ImportError("Triton is not available. Install triton to use triton kernels.")


if triton is None:

    class _TritonKernelStub:
        def __init__(self, _fn=None):
            self._fn = _fn

        def __call__(self, *_args, **_kwargs):
            _missing_triton()

        def __getitem__(self, _grid):
            def launcher(*_args, **_kwargs):
                _missing_triton()

            return launcher

    def triton_jit(*jit_args, **jit_kwargs):
        def decorator(_fn):
            return _TritonKernelStub(_fn)

        if jit_args and callable(jit_args[0]) and not jit_kwargs:
            return decorator(jit_args[0])
        return decorator

    def triton_cdiv(*_args, **_kwargs):
        _missing_triton()
else:
    triton_jit = triton.jit
    triton_cdiv = triton.cdiv


NEG_INF = -1.0e30


@triton_jit
def _logsumexp_pair(a, b):
    m = tl.maximum(a, b)
    return m + tl.log(tl.exp(a - m) + tl.exp(b - m))


@triton_jit
def _rnnt_alpha_kernel(
    log_probs_ptr,
    labels_ptr,
    xlen_ptr,
    ylen_ptr,
    alphas_ptr,
    loglikes_ptr,
    label_stride: tl.constexpr,
    maxT: tl.constexpr,
    maxU: tl.constexpr,
    V: tl.constexpr,
    blank_id: tl.constexpr,
    neg_inf: tl.constexpr,
):
    b = tl.program_id(0)
    T = tl.load(xlen_ptr + b).to(tl.int32)
    U = (tl.load(ylen_ptr + b) + 1).to(tl.int32)

    alpha_base = alphas_ptr + b * maxT * maxU
    logp_base = log_probs_ptr + b * maxT * maxU * V
    label_base = labels_ptr + b * label_stride

    tl.store(alpha_base + 0, 0.0)

    for t in range(maxT):
        t_t = tl.full((), t, tl.int32)
        for u in range(maxU):
            u_t = tl.full((), u, tl.int32)
            if not (t == 0 and u == 0):
                in_range = (t_t < T) & (u_t < U)
                if u == 0:
                    if t > 0:
                        prev = tl.load(
                            alpha_base + (t - 1) * maxU + 0,
                            mask=in_range,
                            other=neg_inf,
                        )
                        logp = tl.load(
                            logp_base + ((t - 1) * maxU + 0) * V + blank_id,
                            mask=in_range,
                            other=neg_inf,
                        )
                        tl.store(alpha_base + t * maxU + 0, prev + logp, mask=in_range)
                elif t == 0:
                    label_idx = tl.where(u_t > 0, u_t - 1, 0)
                    label = tl.load(
                        label_base + label_idx,
                        mask=in_range & (u_t > 0) & (u_t - 1 < (U - 1)),
                        other=0,
                    )
                    label = label.to(tl.int32)
                    prev = tl.load(
                        alpha_base + 0 * maxU + (u - 1), mask=in_range, other=neg_inf
                    )
                    logp = tl.load(
                        logp_base + (0 * maxU + (u - 1)) * V + label,
                        mask=in_range,
                        other=neg_inf,
                    )
                    tl.store(alpha_base + 0 * maxU + u, prev + logp, mask=in_range)
                else:
                    no_emit = tl.load(
                        alpha_base + (t - 1) * maxU + u, mask=in_range, other=neg_inf
                    )
                    no_emit = no_emit + tl.load(
                        logp_base + ((t - 1) * maxU + u) * V + blank_id,
                        mask=in_range,
                        other=neg_inf,
                    )
                    label_idx = tl.where(u_t > 0, u_t - 1, 0)
                    label = tl.load(
                        label_base + label_idx,
                        mask=in_range & (u_t > 0) & (u_t - 1 < (U - 1)),
                        other=0,
                    )
                    label = label.to(tl.int32)
                    emit = tl.load(
                        alpha_base + t * maxU + (u - 1), mask=in_range, other=neg_inf
                    )
                    emit = emit + tl.load(
                        logp_base + (t * maxU + (u - 1)) * V + label,
                        mask=in_range,
                        other=neg_inf,
                    )
                    out = _logsumexp_pair(emit, no_emit)
                    tl.store(alpha_base + t * maxU + u, out, mask=in_range)

    t_last = T - 1
    u_last = U - 1
    in_bounds = (t_last >= 0) & (u_last >= 0)
    loglike = tl.load(
        alpha_base + t_last * maxU + u_last, mask=in_bounds, other=neg_inf
    ) + tl.load(
        logp_base + (t_last * maxU + u_last) * V + blank_id,
        mask=in_bounds,
        other=neg_inf,
    )
    tl.store(loglikes_ptr + b, loglike, mask=in_bounds)


@triton_jit
def _rnnt_beta_kernel(
    log_probs_ptr,
    labels_ptr,
    xlen_ptr,
    ylen_ptr,
    betas_ptr,
    label_stride: tl.constexpr,
    maxT: tl.constexpr,
    maxU: tl.constexpr,
    V: tl.constexpr,
    blank_id: tl.constexpr,
    neg_inf: tl.constexpr,
):
    b = tl.program_id(0)
    T = tl.load(xlen_ptr + b).to(tl.int32)
    U = (tl.load(ylen_ptr + b) + 1).to(tl.int32)

    beta_base = betas_ptr + b * maxT * maxU
    logp_base = log_probs_ptr + b * maxT * maxU * V
    label_base = labels_ptr + b * label_stride

    for t in range(maxT - 1, -1, -1):
        t_t = tl.full((), t, tl.int32)
        for u in range(maxU - 1, -1, -1):
            u_t = tl.full((), u, tl.int32)
            in_range = (t_t < T) & (u_t < U)
            is_t_last = t_t == (T - 1)
            is_u_last = u_t == (U - 1)

            base_val = tl.full((), neg_inf, tl.float32)
            base_val = tl.where(
                is_t_last & is_u_last,
                tl.load(
                    logp_base + (t * maxU + u) * V + blank_id,
                    mask=in_range,
                    other=neg_inf,
                ),
                base_val,
            )

            has_t_next = t_t + 1 < T
            t_next = tl.where(has_t_next, t_t + 1, 0)
            has_u_next = u_t + 1 < U
            u_next = tl.where(has_u_next, u_t + 1, 0)

            no_emit = tl.load(
                beta_base + t_next * maxU + u,
                mask=in_range & has_t_next,
                other=neg_inf,
            )
            no_emit = no_emit + tl.load(
                logp_base + (t * maxU + u) * V + blank_id,
                mask=in_range,
                other=neg_inf,
            )
            label_idx = tl.where(u_t < (U - 1), u_t, 0)
            label = tl.load(
                label_base + label_idx,
                mask=in_range & (u_t < (U - 1)),
                other=0,
            )
            label = label.to(tl.int32)
            emit = tl.load(
                beta_base + t * maxU + u_next,
                mask=in_range & has_u_next,
                other=neg_inf,
            )
            emit = emit + tl.load(
                logp_base + (t * maxU + u) * V + label,
                mask=in_range,
                other=neg_inf,
            )
            out = _logsumexp_pair(emit, no_emit)

            val = tl.where(is_u_last & (~is_t_last), no_emit, out)
            val = tl.where(is_t_last & (~is_u_last), emit, val)
            val = tl.where(is_t_last & is_u_last, base_val, val)
            tl.store(beta_base + t * maxU + u, val, mask=in_range)


@triton_jit
def _rnnt_grad_kernel(
    log_probs_ptr,
    alphas_ptr,
    betas_ptr,
    loglikes_ptr,
    labels_ptr,
    xlen_ptr,
    ylen_ptr,
    grads_ptr,
    label_stride: tl.constexpr,
    maxT: tl.constexpr,
    maxU: tl.constexpr,
    V: tl.constexpr,
    blank_id: tl.constexpr,
    fastemit_lambda: tl.constexpr,
    clamp: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_tu = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_v = tl.program_id(2)

    t = pid_tu // maxU
    u = pid_tu - t * maxU
    t_t = tl.full((), t, tl.int32)
    u_t = tl.full((), u, tl.int32)

    T = tl.load(xlen_ptr + pid_b).to(tl.int32)
    U = (tl.load(ylen_ptr + pid_b) + 1).to(tl.int32)

    v_offsets = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offsets < V

    in_range = (t_t < T) & (u_t < U)
    valid = in_range & v_mask

    base_logp = log_probs_ptr + ((pid_b * maxT + t) * maxU + u) * V
    base_alpha = alphas_ptr + (pid_b * maxT + t) * maxU + u
    base_beta = betas_ptr + (pid_b * maxT + t) * maxU + u
    base_grad = grads_ptr + ((pid_b * maxT + t) * maxU + u) * V
    label_base = labels_ptr + pid_b * label_stride

    logpk = tl.load(base_logp + v_offsets, mask=valid, other=0.0)
    alpha = tl.load(base_alpha, mask=in_range, other=0.0)
    beta = tl.load(base_beta, mask=in_range, other=0.0)
    loglike = tl.load(loglikes_ptr + pid_b)

    grad = tl.where(
        valid,
        tl.exp(alpha + beta + logpk - loglike),
        0.0,
    )

    if fastemit_lambda > 0.0:
        label_idx = tl.where(u_t < (U - 1), u_t, 0)
        label = tl.load(
            label_base + label_idx,
            mask=in_range & (u_t < (U - 1)),
            other=0,
        )
        label = label.to(tl.int32)
        logp_label = tl.load(
            base_logp + label,
            mask=in_range & (u_t < (U - 1)),
            other=0.0,
        )
        has_u_next = u_t + 1 < U
        u_next = tl.where(has_u_next, u_t + 1, 0)
        beta_next = tl.load(
            betas_ptr + (pid_b * maxT + t) * maxU + u_next,
            mask=in_range & has_u_next,
            other=0.0,
        )
        fastemit_grad = tl.where(
            in_range & (u_t < (U - 1)),
            fastemit_lambda * tl.exp(alpha + logp_label + beta_next + logpk - loglike),
            0.0,
        )
        grad = grad + fastemit_grad

    if blank_id >= 0:
        is_blank = v_offsets == blank_id
        is_last = (t_t == (T - 1)) & (u_t == (U - 1))
        grad = tl.where(
            in_range & is_blank & is_last,
            grad - tl.exp(alpha + logpk - loglike),
            grad,
        )
        has_t_next = t_t + 1 < T
        t_next = tl.where(has_t_next, t_t + 1, 0)
        beta_down = tl.load(
            betas_ptr + (pid_b * maxT + t_next) * maxU + u,
            mask=in_range & has_t_next,
            other=0.0,
        )
        grad = tl.where(
            in_range & is_blank & (t_t < (T - 1)),
            grad - tl.exp(alpha + logpk - loglike + beta_down),
            grad,
        )

    label_idx = tl.where(u_t < (U - 1), u_t, 0)
    label = tl.load(
        label_base + label_idx,
        mask=in_range & (u_t < (U - 1)),
        other=0,
    )
    label = label.to(tl.int32)
    is_label = v_offsets == label
    has_u_next = u_t + 1 < U
    u_next = tl.where(has_u_next, u_t + 1, 0)
    beta_right = tl.load(
        betas_ptr + (pid_b * maxT + t) * maxU + u_next,
        mask=in_range & has_u_next,
        other=0.0,
    )
    label_term = tl.exp(
        tl.log(1.0 + fastemit_lambda) + alpha + logpk - loglike + beta_right
    )
    grad = tl.where(in_range & (u_t < (U - 1)) & is_label, grad - label_term, grad)

    if clamp > 0.0:
        grad = tl.minimum(grad, clamp)
        grad = tl.maximum(grad, -clamp)

    tl.store(base_grad + v_offsets, grad, mask=valid)


def _launch_alphas(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    blank_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, maxT, maxU, V = log_probs.shape
    alphas = torch.full(
        (B, maxT, maxU), NEG_INF, device=log_probs.device, dtype=log_probs.dtype
    )
    loglikes = torch.empty((B,), device=log_probs.device, dtype=log_probs.dtype)
    label_stride = labels.shape[1]
    _rnnt_alpha_kernel[(B,)](
        log_probs,
        labels,
        act_lens,
        label_lens,
        alphas,
        loglikes,
        label_stride=label_stride,
        maxT=maxT,
        maxU=maxU,
        V=V,
        blank_id=blank_id,
        neg_inf=NEG_INF,
    )
    return alphas, loglikes


def _launch_betas(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    B, maxT, maxU, V = log_probs.shape
    betas = torch.full(
        (B, maxT, maxU), NEG_INF, device=log_probs.device, dtype=log_probs.dtype
    )
    label_stride = labels.shape[1]
    _rnnt_beta_kernel[(B,)](
        log_probs,
        labels,
        act_lens,
        label_lens,
        betas,
        label_stride=label_stride,
        maxT=maxT,
        maxU=maxU,
        V=V,
        blank_id=blank_id,
        neg_inf=NEG_INF,
    )
    return betas


def _launch_grads(
    log_probs: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    loglikes: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    blank_id: int,
    fastemit_lambda: float,
    clamp: float,
    block_v: int = 128,
) -> torch.Tensor:
    B, maxT, maxU, V = log_probs.shape
    grads = torch.zeros_like(log_probs)
    # Put the largest dimension on grid.x to avoid the 65k grid.y limit.
    grid = (maxT * maxU, B, triton_cdiv(V, block_v))
    label_stride = labels.shape[1]
    _rnnt_grad_kernel[grid](
        log_probs,
        alphas,
        betas,
        loglikes,
        labels,
        act_lens,
        label_lens,
        grads,
        label_stride=label_stride,
        maxT=maxT,
        maxU=maxU,
        V=V,
        blank_id=blank_id,
        fastemit_lambda=float(fastemit_lambda),
        clamp=float(clamp),
        BLOCK_V=block_v,
    )
    return grads


def rnnt_loss_triton(
    acts: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    blank_id: int,
    fastemit_lambda: float,
    clamp: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise ImportError(
            "Triton is not available. Install triton to use triton kernels."
        )
    if acts.device.type != "cuda":
        raise RuntimeError("Triton RNNT only supports CUDA tensors.")
    if acts.dtype != torch.float32:
        raise RuntimeError("Triton RNNT requires float32 inputs.")
    if act_lens.device != acts.device:
        act_lens = act_lens.to(device=acts.device)
    if label_lens.device != acts.device:
        label_lens = label_lens.to(device=acts.device)
    log_probs = torch.log_softmax(acts, dim=-1)
    alphas, loglikes = _launch_alphas(log_probs, labels, act_lens, label_lens, blank_id)
    betas = _launch_betas(log_probs, labels, act_lens, label_lens, blank_id)
    grads = _launch_grads(
        log_probs,
        alphas,
        betas,
        loglikes,
        labels,
        act_lens,
        label_lens,
        blank_id,
        fastemit_lambda,
        clamp,
    )
    return log_probs, alphas, betas, grads, loglikes
