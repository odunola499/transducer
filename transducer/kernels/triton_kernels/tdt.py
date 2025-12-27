from __future__ import annotations

from typing import Tuple
import random

import torch
import triton
import triton.language as tl

from transducer.kernels.triton_kernels.rnnt import rnnt_loss_triton


NEG_INF = -1.0e30


@triton.jit
def _logsumexp_pair(a, b):
    m = tl.maximum(a, b)
    return m + tl.log(tl.exp(a - m) + tl.exp(b - m))


@triton.jit
def _tdt_alpha_kernel(
    log_probs_ptr,
    duration_acts_ptr,
    labels_ptr,
    xlen_ptr,
    ylen_ptr,
    alphas_ptr,
    loglikes_ptr,
    durations_ptr,
    label_stride: tl.constexpr,
    maxT: tl.constexpr,
    maxU: tl.constexpr,
    V: tl.constexpr,
    NUM_DURATIONS: tl.constexpr,
    blank_id: tl.constexpr,
    sigma: tl.constexpr,
    neg_inf: tl.constexpr,
):
    b = tl.program_id(0)
    T = tl.load(xlen_ptr + b).to(tl.int32)
    U = (tl.load(ylen_ptr + b) + 1).to(tl.int32)

    alpha_base = alphas_ptr + b * maxT * maxU
    logp_base = log_probs_ptr + b * maxT * maxU * V
    dur_base = duration_acts_ptr + b * maxT * maxU * NUM_DURATIONS
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
                        val = tl.full((), neg_inf, tl.float32)
                        for i in range(NUM_DURATIONS):
                            d = tl.load(durations_ptr + i).to(tl.int32)
                            t_minus = t_t - d
                            valid = in_range & (d > 0) & (t_minus >= 0)
                            alpha_prev = tl.load(
                                alpha_base + t_minus * maxU + 0,
                                mask=valid,
                                other=neg_inf,
                            )
                            logp_blank = tl.load(
                                logp_base + (t_minus * maxU + 0) * V + blank_id,
                                mask=valid,
                                other=neg_inf,
                            )
                            logp_dur = tl.load(
                                dur_base + (t_minus * maxU + 0) * NUM_DURATIONS + i,
                                mask=valid,
                                other=neg_inf,
                            )
                            candidate = alpha_prev + logp_blank - sigma + logp_dur
                            val = _logsumexp_pair(val, tl.where(valid, candidate, neg_inf))
                        tl.store(alpha_base + t * maxU + 0, val, mask=in_range)
                elif t == 0:
                    dur0 = tl.load(durations_ptr + 0).to(tl.int32)
                    use_d0 = dur0 == 0
                    label_idx = tl.where(u_t > 0, u_t - 1, 0)
                    label = tl.load(
                        label_base + label_idx,
                        mask=in_range & use_d0 & (u_t > 0) & (u_t - 1 < (U - 1)),
                        other=0,
                    )
                    label = label.to(tl.int32)
                    prev = tl.load(
                        alpha_base + 0 * maxU + (u - 1),
                        mask=in_range & use_d0,
                        other=neg_inf,
                    )
                    logp_label = tl.load(
                        logp_base + (0 * maxU + (u - 1)) * V + label,
                        mask=in_range & use_d0,
                        other=neg_inf,
                    )
                    logp_dur = tl.load(
                        dur_base + (0 * maxU + (u - 1)) * NUM_DURATIONS + 0,
                        mask=in_range & use_d0,
                        other=neg_inf,
                    )
                    val = prev + logp_label - sigma + logp_dur
                    val = tl.where(use_d0, val, neg_inf)
                    tl.store(alpha_base + 0 * maxU + u, val, mask=in_range)
                else:
                    no_emit = tl.full((), neg_inf, tl.float32)
                    for i in range(NUM_DURATIONS):
                        d = tl.load(durations_ptr + i).to(tl.int32)
                        t_minus = t_t - d
                        valid = in_range & (d > 0) & (t_minus >= 0)
                        alpha_prev = tl.load(
                            alpha_base + t_minus * maxU + u,
                            mask=valid,
                            other=neg_inf,
                        )
                        logp_blank = tl.load(
                            logp_base + (t_minus * maxU + u) * V + blank_id,
                            mask=valid,
                            other=neg_inf,
                        )
                        logp_dur = tl.load(
                            dur_base + (t_minus * maxU + u) * NUM_DURATIONS + i,
                            mask=valid,
                            other=neg_inf,
                        )
                        candidate = alpha_prev + logp_blank - sigma + logp_dur
                        no_emit = _logsumexp_pair(no_emit, tl.where(valid, candidate, neg_inf))

                    emit = tl.full((), neg_inf, tl.float32)
                    label_idx = tl.where(u_t > 0, u_t - 1, 0)
                    label = tl.load(
                        label_base + label_idx,
                        mask=in_range & (u_t > 0) & (u_t - 1 < (U - 1)),
                        other=0,
                    )
                    label = label.to(tl.int32)
                    for i in range(NUM_DURATIONS):
                        d = tl.load(durations_ptr + i).to(tl.int32)
                        t_minus = t_t - d
                        valid = in_range & (t_minus >= 0)
                        alpha_prev = tl.load(
                            alpha_base + t_minus * maxU + (u - 1),
                            mask=valid,
                            other=neg_inf,
                        )
                        logp_label = tl.load(
                            logp_base + (t_minus * maxU + (u - 1)) * V + label,
                            mask=valid,
                            other=neg_inf,
                        )
                        logp_dur = tl.load(
                            dur_base + (t_minus * maxU + (u - 1)) * NUM_DURATIONS + i,
                            mask=valid,
                            other=neg_inf,
                        )
                        candidate = alpha_prev + logp_label - sigma + logp_dur
                        emit = _logsumexp_pair(emit, tl.where(valid, candidate, neg_inf))

                    out = _logsumexp_pair(emit, no_emit)
                    tl.store(alpha_base + t * maxU + u, out, mask=in_range)

    t_last = T - 1
    u_last = U - 1
    in_bounds = (t_last >= 0) & (u_last >= 0)
    loglike = tl.full((), neg_inf, tl.float32)
    for i in range(NUM_DURATIONS):
        d = tl.load(durations_ptr + i).to(tl.int32)
        t_start = t_last - (d - 1)
        valid = in_bounds & (d > 0) & (t_start >= 0)
        alpha_val = tl.load(
            alpha_base + t_start * maxU + u_last, mask=valid, other=neg_inf
        )
        logp_blank = tl.load(
            logp_base + (t_start * maxU + u_last) * V + blank_id,
            mask=valid,
            other=neg_inf,
        )
        logp_dur = tl.load(
            dur_base + (t_start * maxU + u_last) * NUM_DURATIONS + i,
            mask=valid,
            other=neg_inf,
        )
        candidate = alpha_val + logp_blank - sigma + logp_dur
        loglike = _logsumexp_pair(loglike, tl.where(valid, candidate, neg_inf))

    tl.store(loglikes_ptr + b, loglike, mask=in_bounds)


@triton.jit
def _tdt_beta_kernel(
    log_probs_ptr,
    duration_acts_ptr,
    labels_ptr,
    xlen_ptr,
    ylen_ptr,
    betas_ptr,
    durations_ptr,
    label_stride: tl.constexpr,
    maxT: tl.constexpr,
    maxU: tl.constexpr,
    V: tl.constexpr,
    NUM_DURATIONS: tl.constexpr,
    blank_id: tl.constexpr,
    sigma: tl.constexpr,
    neg_inf: tl.constexpr,
):
    b = tl.program_id(0)
    T = tl.load(xlen_ptr + b).to(tl.int32)
    U = (tl.load(ylen_ptr + b) + 1).to(tl.int32)

    beta_base = betas_ptr + b * maxT * maxU
    logp_base = log_probs_ptr + b * maxT * maxU * V
    dur_base = duration_acts_ptr + b * maxT * maxU * NUM_DURATIONS
    label_base = labels_ptr + b * label_stride

    for t in range(maxT - 1, -1, -1):
        t_t = tl.full((), t, tl.int32)
        for u in range(maxU - 1, -1, -1):
            u_t = tl.full((), u, tl.int32)
            in_range = (t_t < T) & (u_t < U)
            is_t_last = t_t == (T - 1)
            is_u_last = u_t == (U - 1)

            val = tl.full((), neg_inf, tl.float32)

            base_val = tl.full((), neg_inf, tl.float32)
            for i in range(NUM_DURATIONS):
                d = tl.load(durations_ptr + i).to(tl.int32)
                is_d1 = d == 1
                logp_blank = tl.load(
                    logp_base + (t * maxU + u) * V + blank_id,
                    mask=in_range,
                    other=neg_inf,
                )
                logp_dur = tl.load(
                    dur_base + (t * maxU + u) * NUM_DURATIONS + i,
                    mask=in_range,
                    other=neg_inf,
                )
                candidate = logp_blank - sigma + logp_dur
                base_val = tl.where(is_d1 & in_range, candidate, base_val)

            val = tl.where(is_t_last & is_u_last, base_val, val)

            if u == maxU - 1:
                no_emit = tl.full((), neg_inf, tl.float32)
                for i in range(NUM_DURATIONS):
                    d = tl.load(durations_ptr + i).to(tl.int32)
                    t_plus = t_t + d
                    logp_blank = tl.load(
                        logp_base + (t * maxU + u) * V + blank_id,
                        mask=in_range,
                        other=neg_inf,
                    )
                    logp_dur = tl.load(
                        dur_base + (t * maxU + u) * NUM_DURATIONS + i,
                        mask=in_range,
                        other=neg_inf,
                    )
                    cand_lt = tl.load(
                        beta_base + t_plus * maxU + u,
                        mask=in_range & (d > 0) & (t_plus < T),
                        other=neg_inf,
                    )
                    cand_lt = cand_lt + logp_blank + logp_dur - sigma
                    cand_eq = logp_blank + logp_dur - sigma
                    no_emit = _logsumexp_pair(
                        no_emit,
                        tl.where(in_range & (d > 0) & (t_plus < T), cand_lt, neg_inf),
                    )
                    no_emit = _logsumexp_pair(
                        no_emit,
                        tl.where(in_range & (d > 0) & (t_plus == T), cand_eq, neg_inf),
                    )
                val = tl.where(is_u_last & (~is_t_last), no_emit, val)

            if u < maxU - 1:
                if t == maxT - 1:
                    dur0 = tl.load(durations_ptr + 0).to(tl.int32)
                    use_d0 = dur0 == 0
                    label = tl.load(
                        label_base + u_t,
                        mask=in_range & use_d0 & (u_t < (U - 1)),
                        other=0,
                    )
                    label = label.to(tl.int32)
                    logp_label = tl.load(
                        logp_base + (t * maxU + u) * V + label,
                        mask=in_range & use_d0,
                        other=neg_inf,
                    )
                    logp_dur = tl.load(
                        dur_base + (t * maxU + u) * NUM_DURATIONS + 0,
                        mask=in_range & use_d0,
                        other=neg_inf,
                    )
                    beta_next = tl.load(
                        beta_base + t * maxU + (u + 1),
                        mask=in_range & use_d0,
                        other=neg_inf,
                    )
                    val_tlast = beta_next + logp_label + logp_dur - sigma
                    val = tl.where(is_t_last & (~is_u_last) & use_d0, val_tlast, val)
                else:
                    no_emit = tl.full((), neg_inf, tl.float32)
                    for i in range(NUM_DURATIONS):
                        d = tl.load(durations_ptr + i).to(tl.int32)
                        t_plus = t_t + d
                        logp_blank = tl.load(
                            logp_base + (t * maxU + u) * V + blank_id,
                            mask=in_range,
                            other=neg_inf,
                        )
                        logp_dur = tl.load(
                            dur_base + (t * maxU + u) * NUM_DURATIONS + i,
                            mask=in_range,
                            other=neg_inf,
                        )
                        beta_down = tl.load(
                            beta_base + t_plus * maxU + u,
                            mask=in_range & (d > 0) & (t_plus < T),
                            other=neg_inf,
                        )
                        candidate = beta_down + logp_blank + logp_dur - sigma
                        no_emit = _logsumexp_pair(
                            no_emit,
                            tl.where(in_range & (d > 0) & (t_plus < T), candidate, neg_inf),
                        )

                    emit = tl.full((), neg_inf, tl.float32)
                    label = tl.load(
                        label_base + u_t, mask=in_range & (u_t < (U - 1)), other=0
                    )
                    label = label.to(tl.int32)
                    for i in range(NUM_DURATIONS):
                        d = tl.load(durations_ptr + i).to(tl.int32)
                        t_plus = t_t + d
                        logp_label = tl.load(
                            logp_base + (t * maxU + u) * V + label,
                            mask=in_range,
                            other=neg_inf,
                        )
                        logp_dur = tl.load(
                            dur_base + (t * maxU + u) * NUM_DURATIONS + i,
                            mask=in_range,
                            other=neg_inf,
                        )
                        beta_right = tl.load(
                            beta_base + t_plus * maxU + (u + 1),
                            mask=in_range & (t_plus < T),
                            other=neg_inf,
                        )
                        candidate = beta_right + logp_label + logp_dur - sigma
                        emit = _logsumexp_pair(
                            emit, tl.where(in_range & (t_plus < T), candidate, neg_inf)
                        )

                    out = _logsumexp_pair(emit, no_emit)
                    val = tl.where((~is_t_last) & (~is_u_last), out, val)

            tl.store(beta_base + t * maxU + u, val, mask=in_range)


@triton.jit
def _tdt_label_grad_kernel(
    log_probs_ptr,
    duration_acts_ptr,
    alphas_ptr,
    betas_ptr,
    loglikes_ptr,
    labels_ptr,
    xlen_ptr,
    ylen_ptr,
    durations_ptr,
    label_grads_ptr,
    label_stride: tl.constexpr,
    maxT: tl.constexpr,
    maxU: tl.constexpr,
    V: tl.constexpr,
    NUM_DURATIONS: tl.constexpr,
    blank_id: tl.constexpr,
    fastemit_lambda: tl.constexpr,
    clamp: tl.constexpr,
    sigma: tl.constexpr,
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
    base_grad = label_grads_ptr + ((pid_b * maxT + t) * maxU + u) * V
    base_dur = duration_acts_ptr + ((pid_b * maxT + t) * maxU + u) * NUM_DURATIONS
    label_base = labels_ptr + pid_b * label_stride

    logpk = tl.load(base_logp + v_offsets, mask=valid, other=0.0)
    alpha = tl.load(base_alpha, mask=in_range, other=0.0)
    beta = tl.load(base_beta, mask=in_range, other=0.0)
    loglike = tl.load(loglikes_ptr + pid_b)

    grad = tl.where(valid, tl.exp(alpha + beta + logpk - loglike), 0.0)

    if fastemit_lambda > 0.0:
        label_idx = tl.where(u_t < (U - 1), u_t, 0)
        label = tl.load(
            label_base + label_idx, mask=in_range & (u_t < (U - 1)), other=0
        )
        label = label.to(tl.int32)
        logp_label = tl.load(
            base_logp + label, mask=in_range & (u_t < (U - 1)), other=0.0
        )
        fastemit_grad = tl.full((BLOCK_V,), 0.0, tl.float32)
        for i in range(NUM_DURATIONS):
            d = tl.load(durations_ptr + i).to(tl.int32)
            t_plus = t_t + d
            has = in_range & (u_t < (U - 1)) & (t_plus < T)
            beta_next = tl.load(
                betas_ptr + (pid_b * maxT + t_plus) * maxU + (u + 1),
                mask=has,
                other=0.0,
            )
            logp_dur = tl.load(base_dur + i, mask=has, other=0.0)
            term = fastemit_lambda * tl.exp(
                alpha + logp_label + logp_dur + beta_next + logpk - sigma - loglike
            )
            fastemit_grad = fastemit_grad + tl.where(has, term, 0.0)
        grad = grad + fastemit_grad

    if blank_id >= 0:
        is_blank = v_offsets == blank_id
        is_last = (t_t == (T - 1)) & (u_t == (U - 1))
        for i in range(NUM_DURATIONS):
            d = tl.load(durations_ptr + i).to(tl.int32)
            t_plus = t_t + d
            logp_dur = tl.load(base_dur + i, mask=in_range & (d > 0), other=0.0)
            term_last = tl.exp(alpha + logpk - sigma - loglike + logp_dur)
            cond_last = in_range & is_blank & is_last & (d > 0) & (t_plus == T)
            grad = tl.where(cond_last, grad - term_last, grad)

            beta_down = tl.load(
                betas_ptr + (pid_b * maxT + t_plus) * maxU + u,
                mask=in_range & (d > 0) & (t_plus < T),
                other=0.0,
            )
            term_mid = tl.exp(alpha + logpk - sigma - loglike + beta_down + logp_dur)
            cond_mid = in_range & is_blank & (d > 0) & (t_plus < T)
            grad = tl.where(cond_mid, grad - term_mid, grad)

    label_idx = tl.where(u_t < (U - 1), u_t, 0)
    label = tl.load(label_base + label_idx, mask=in_range & (u_t < (U - 1)), other=0)
    label = label.to(tl.int32)
    is_label = v_offsets == label
    log1p_fastemit = tl.log(1.0 + fastemit_lambda)
    for i in range(NUM_DURATIONS):
        d = tl.load(durations_ptr + i).to(tl.int32)
        t_plus = t_t + d
        beta_right = tl.load(
            betas_ptr + (pid_b * maxT + t_plus) * maxU + (u + 1),
            mask=in_range & (t_plus < T),
            other=0.0,
        )
        logp_dur = tl.load(base_dur + i, mask=in_range, other=0.0)
        term = tl.exp(
            log1p_fastemit + alpha + logpk - sigma - loglike + beta_right + logp_dur
        )
        cond = in_range & (u_t < (U - 1)) & is_label & (t_plus < T)
        grad = tl.where(cond, grad - term, grad)

    if clamp > 0.0:
        grad = tl.minimum(grad, clamp)
        grad = tl.maximum(grad, -clamp)

    tl.store(base_grad + v_offsets, grad, mask=valid)


@triton.jit
def _tdt_duration_grad_kernel(
    log_probs_ptr,
    duration_acts_ptr,
    alphas_ptr,
    betas_ptr,
    loglikes_ptr,
    labels_ptr,
    xlen_ptr,
    ylen_ptr,
    durations_ptr,
    duration_grads_ptr,
    label_stride: tl.constexpr,
    maxT: tl.constexpr,
    maxU: tl.constexpr,
    V: tl.constexpr,
    NUM_DURATIONS: tl.constexpr,
    blank_id: tl.constexpr,
    sigma: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_tu = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_d = tl.program_id(2)

    t = pid_tu // maxU
    u = pid_tu - t * maxU
    t_t = tl.full((), t, tl.int32)
    u_t = tl.full((), u, tl.int32)

    T = tl.load(xlen_ptr + pid_b).to(tl.int32)
    U = (tl.load(ylen_ptr + pid_b) + 1).to(tl.int32)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < NUM_DURATIONS

    in_range = (t_t < T) & (u_t < U)
    valid = in_range & d_mask

    base_logp = log_probs_ptr + ((pid_b * maxT + t) * maxU + u) * V
    base_alpha = alphas_ptr + (pid_b * maxT + t) * maxU + u
    base_dur = duration_acts_ptr + ((pid_b * maxT + t) * maxU + u) * NUM_DURATIONS
    base_grad = duration_grads_ptr + ((pid_b * maxT + t) * maxU + u) * NUM_DURATIONS
    label_base = labels_ptr + pid_b * label_stride

    alpha = tl.load(base_alpha, mask=in_range, other=0.0)
    loglike = tl.load(loglikes_ptr + pid_b)
    logpk_blank = (
        tl.load(base_logp + blank_id, mask=in_range, other=0.0) - sigma
    )

    label_idx = tl.where(u_t < (U - 1), u_t, 0)
    label = tl.load(label_base + label_idx, mask=in_range & (u_t < (U - 1)), other=0)
    label = label.to(tl.int32)
    logpk_label = (
        tl.load(base_logp + label, mask=in_range & (u_t < (U - 1)), other=0.0) - sigma
    )

    d_vals = tl.load(durations_ptr + d_offsets, mask=d_mask, other=0).to(tl.int32)
    t_plus = t_t + d_vals

    grad = tl.full((BLOCK_D,), 0.0, tl.float32)

    beta_label = tl.load(
        betas_ptr + (pid_b * maxT + t_plus) * maxU + (u + 1),
        mask=valid & (u_t < (U - 1)) & (t_plus < T),
        other=0.0,
    )
    term_label = tl.exp(alpha + beta_label + logpk_label - loglike)
    grad = grad - tl.where(valid & (u_t < (U - 1)) & (t_plus < T), term_label, 0.0)

    beta_blank = tl.load(
        betas_ptr + (pid_b * maxT + t_plus) * maxU + u,
        mask=valid & (d_vals > 0) & (t_plus < T),
        other=0.0,
    )
    term_blank = tl.exp(alpha + beta_blank + logpk_blank - loglike)
    grad = grad - tl.where(valid & (d_vals > 0) & (t_plus < T), term_blank, 0.0)

    term_final = tl.exp(alpha + logpk_blank - loglike)
    grad = grad - tl.where(
        valid & (d_vals > 0) & (u_t == (U - 1)) & (t_plus == T), term_final, 0.0
    )

    logp_dur = tl.load(base_dur + d_offsets, mask=valid, other=0.0)
    grad = grad * tl.exp(logp_dur)
    tl.store(base_grad + d_offsets, grad, mask=valid)


def _launch_tdt_alphas(
    log_probs: torch.Tensor,
    duration_acts: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    durations: torch.Tensor,
    blank_id: int,
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, maxT, maxU, V = log_probs.shape
    alphas = torch.full(
        (B, maxT, maxU), NEG_INF, device=log_probs.device, dtype=log_probs.dtype
    )
    loglikes = torch.empty((B,), device=log_probs.device, dtype=log_probs.dtype)
    label_stride = labels.shape[1]
    num_durations = durations.numel()
    _tdt_alpha_kernel[(B,)](
        log_probs,
        duration_acts,
        labels,
        act_lens,
        label_lens,
        alphas,
        loglikes,
        durations,
        label_stride=label_stride,
        maxT=maxT,
        maxU=maxU,
        V=V,
        NUM_DURATIONS=num_durations,
        blank_id=blank_id,
        sigma=float(sigma),
        neg_inf=NEG_INF,
    )
    return alphas, loglikes


def _launch_tdt_betas(
    log_probs: torch.Tensor,
    duration_acts: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    durations: torch.Tensor,
    blank_id: int,
    sigma: float,
) -> torch.Tensor:
    B, maxT, maxU, V = log_probs.shape
    betas = torch.full(
        (B, maxT, maxU), NEG_INF, device=log_probs.device, dtype=log_probs.dtype
    )
    label_stride = labels.shape[1]
    num_durations = durations.numel()
    _tdt_beta_kernel[(B,)](
        log_probs,
        duration_acts,
        labels,
        act_lens,
        label_lens,
        betas,
        durations,
        label_stride=label_stride,
        maxT=maxT,
        maxU=maxU,
        V=V,
        NUM_DURATIONS=num_durations,
        blank_id=blank_id,
        sigma=float(sigma),
        neg_inf=NEG_INF,
    )
    return betas


def _launch_tdt_label_grads(
    log_probs: torch.Tensor,
    duration_acts: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    loglikes: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    durations: torch.Tensor,
    blank_id: int,
    fastemit_lambda: float,
    clamp: float,
    sigma: float,
    block_v: int = 128,
) -> torch.Tensor:
    B, maxT, maxU, V = log_probs.shape
    label_grads = torch.zeros_like(log_probs)
    num_durations = durations.numel()
    grid = (maxT * maxU, B, triton.cdiv(V, block_v))
    label_stride = labels.shape[1]
    _tdt_label_grad_kernel[grid](
        log_probs,
        duration_acts,
        alphas,
        betas,
        loglikes,
        labels,
        act_lens,
        label_lens,
        durations,
        label_grads,
        label_stride=label_stride,
        maxT=maxT,
        maxU=maxU,
        V=V,
        NUM_DURATIONS=num_durations,
        blank_id=blank_id,
        fastemit_lambda=float(fastemit_lambda),
        clamp=float(clamp),
        sigma=float(sigma),
        BLOCK_V=block_v,
    )
    return label_grads


def _launch_tdt_duration_grads(
    log_probs: torch.Tensor,
    duration_acts: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    loglikes: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    durations: torch.Tensor,
    blank_id: int,
    sigma: float,
    block_d: int = 32,
) -> torch.Tensor:
    B, maxT, maxU, _ = log_probs.shape
    num_durations = durations.numel()
    duration_grads = torch.zeros_like(duration_acts)
    if num_durations == 0:
        return duration_grads
    grid = (maxT * maxU, B, triton.cdiv(num_durations, block_d))
    label_stride = labels.shape[1]
    _tdt_duration_grad_kernel[grid](
        log_probs,
        duration_acts,
        alphas,
        betas,
        loglikes,
        labels,
        act_lens,
        label_lens,
        durations,
        duration_grads,
        label_stride=label_stride,
        maxT=maxT,
        maxU=maxU,
        V=log_probs.shape[-1],
        NUM_DURATIONS=num_durations,
        blank_id=blank_id,
        sigma=float(sigma),
        BLOCK_D=block_d,
    )
    return duration_grads


def tdt_loss_triton(
    label_acts: torch.Tensor,
    duration_acts: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    blank_id: int,
    durations: list[int],
    fastemit_lambda: float,
    clamp: float,
    sigma: float,
    omega: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if label_acts.device.type != "cuda":
        raise RuntimeError("Triton TDT only supports CUDA tensors.")
    if label_acts.dtype != torch.float32:
        raise RuntimeError("Triton TDT requires float32 label activations.")
    if duration_acts.dtype != torch.float32:
        raise RuntimeError("Triton TDT requires float32 duration activations.")
    if duration_acts.shape[:3] != label_acts.shape[:3]:
        raise RuntimeError("Label and duration activations must share [B, T, U].")
    if duration_acts.shape[-1] != len(durations):
        raise RuntimeError("Duration activations last dim must match durations length.")
    if len(durations) == 0:
        raise RuntimeError("Triton TDT requires a non-empty durations list.")
    if act_lens.device != label_acts.device:
        act_lens = act_lens.to(device=label_acts.device)
    if label_lens.device != label_acts.device:
        label_lens = label_lens.to(device=label_acts.device)

    if omega > 0.0 and random.random() < omega:
        log_probs, alphas, betas, label_grads, loglikes = rnnt_loss_triton(
            label_acts, labels, act_lens, label_lens, blank_id, fastemit_lambda, clamp
        )
        duration_grads = torch.zeros_like(duration_acts)
        return log_probs, alphas, betas, label_grads, duration_grads, loglikes

    durations_tensor = torch.tensor(
        durations, device=label_acts.device, dtype=torch.int32
    )
    log_probs = torch.log_softmax(label_acts, dim=-1)
    alphas, loglikes = _launch_tdt_alphas(
        log_probs,
        duration_acts,
        labels,
        act_lens,
        label_lens,
        durations_tensor,
        blank_id,
        sigma,
    )
    betas = _launch_tdt_betas(
        log_probs,
        duration_acts,
        labels,
        act_lens,
        label_lens,
        durations_tensor,
        blank_id,
        sigma,
    )
    label_grads = _launch_tdt_label_grads(
        log_probs,
        duration_acts,
        alphas,
        betas,
        loglikes,
        labels,
        act_lens,
        label_lens,
        durations_tensor,
        blank_id,
        fastemit_lambda,
        clamp,
        sigma,
    )
    duration_grads = _launch_tdt_duration_grads(
        log_probs,
        duration_acts,
        alphas,
        betas,
        loglikes,
        labels,
        act_lens,
        label_lens,
        durations_tensor,
        blank_id,
        sigma,
    )
    return log_probs, alphas, betas, label_grads, duration_grads, loglikes
