import time

import torch

from transducer.losses.tdt.tdt_loss import TDTLoss
from transducer.losses.tdt.tdt_triton import TDTTritonLoss


def _check_env():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")


def _make_data(batch, max_t, max_u, vocab, num_durations):
    device = torch.device("cuda")
    acts = torch.randn(
        batch, max_t, max_u, vocab + num_durations, device=device, dtype=torch.float32
    )
    labels = torch.randint(
        0, vocab - 1, (batch, max_u - 1), device=device, dtype=torch.long
    )
    act_lens = torch.randint(1, max_t + 1, (batch,), device=device, dtype=torch.long)
    label_lens = torch.randint(1, max_u, (batch,), device=device, dtype=torch.long)
    return acts, labels, act_lens, label_lens


def _make_training_like_data(
    batch,
    max_t,
    max_u,
    vocab,
    num_durations,
    pad_id=0,
    max_label_len=None,
    force_max_label=True,
    allow_zero_tokens=False,
):
    device = torch.device("cuda")
    blank_id = vocab - 1
    max_label_len = (max_u - 1) if max_label_len is None else max_label_len
    label_lens = torch.randint(1, max_label_len + 1, (batch,), device=device, dtype=torch.long)
    if force_max_label:
        label_lens[0] = max_label_len

    labels = torch.full((batch, max_u), pad_id, device=device, dtype=torch.long)
    low_token = 0 if allow_zero_tokens else 1
    for i, length in enumerate(label_lens.tolist()):
        labels[i, :length] = torch.randint(
            low_token, blank_id, (length,), device=device, dtype=torch.long
        )

    label_lens = (labels != pad_id).sum(dim=-1)
    acts = torch.randn(
        batch, max_t, max_u, vocab + num_durations, device=device, dtype=torch.float32
    )
    return acts, labels, None, label_lens


def _tensor_stats(tensor):
    finite = torch.isfinite(tensor)
    finite_ratio = finite.float().mean().item()
    if finite.any().item():
        finite_vals = tensor[finite]
        return {
            "min": finite_vals.min().item(),
            "max": finite_vals.max().item(),
            "mean": finite_vals.mean().item(),
            "absmax": finite_vals.abs().max().item(),
            "finite_ratio": finite_ratio,
        }
    return {
        "min": float("nan"),
        "max": float("nan"),
        "mean": float("nan"),
        "absmax": float("nan"),
        "finite_ratio": finite_ratio,
    }


def _print_nonfinite_debug(tag, acts, labels, act_lens, label_lens, blank_id, num_durations):
    batch, max_t, max_u, total_vocab = acts.shape
    vocab = total_vocab - num_durations
    act_lens_use = (
        torch.full((batch,), max_t, device=acts.device, dtype=torch.long)
        if act_lens is None
        else act_lens
    )
    label_acts = acts[:, :, :, :vocab]
    duration_acts = acts[:, :, :, vocab:]
    print(f"{tag} debug:")
    print(
        f"  shapes: acts={tuple(acts.shape)} labels={tuple(labels.shape)} "
        f"act_lens={tuple(act_lens_use.shape)} label_lens={tuple(label_lens.shape)}"
    )
    print(
        f"  lengths: act_lens=[{act_lens_use.min().item()},{act_lens_use.max().item()}] "
        f"label_lens=[{label_lens.min().item()},{label_lens.max().item()}]"
    )
    print(
        f"  label_range: min={labels.min().item()} max={labels.max().item()} "
        f"zeros={(labels == 0).sum().item()} blank_id={blank_id}"
    )
    print(
        f"  len_violations: label_lens>=U={(label_lens >= max_u).sum().item()} "
        f"act_lens>T={(act_lens_use > max_t).sum().item()}"
    )
    stats = _tensor_stats(label_acts)
    print(
        "  label_logits: "
        f"min={stats['min']:.3f} max={stats['max']:.3f} mean={stats['mean']:.3f} "
        f"absmax={stats['absmax']:.3f} finite_ratio={stats['finite_ratio']:.3f}"
    )
    d_stats = _tensor_stats(duration_acts)
    print(
        "  duration_logits: "
        f"min={d_stats['min']:.3f} max={d_stats['max']:.3f} mean={d_stats['mean']:.3f} "
        f"absmax={d_stats['absmax']:.3f} finite_ratio={d_stats['finite_ratio']:.3f}"
    )
    log_probs = torch.log_softmax(label_acts, dim=-1)
    lp_stats = _tensor_stats(log_probs)
    print(
        "  label_log_softmax: "
        f"min={lp_stats['min']:.3f} max={lp_stats['max']:.3f} mean={lp_stats['mean']:.3f} "
        f"absmax={lp_stats['absmax']:.3f} finite_ratio={lp_stats['finite_ratio']:.3f}"
    )
    dur_log_probs = torch.log_softmax(duration_acts, dim=-1)
    dl_stats = _tensor_stats(dur_log_probs)
    print(
        "  duration_log_softmax: "
        f"min={dl_stats['min']:.3f} max={dl_stats['max']:.3f} mean={dl_stats['mean']:.3f} "
        f"absmax={dl_stats['absmax']:.3f} finite_ratio={dl_stats['finite_ratio']:.3f}"
    )


def _compare_data(
    acts,
    labels,
    act_lens,
    label_lens,
    blank_id,
    durations,
    fastemit_lambda=0.0,
    clamp=0.0,
    sigma=0.0,
    omega=0.0,
):
    loss_cuda = TDTLoss(
        blank_id=blank_id,
        durations=durations,
        reduction="mean",
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        sigma=sigma,
        omega=omega,
    )
    loss_triton = TDTTritonLoss(
        blank_id=blank_id,
        durations=durations,
        reduction="mean",
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        sigma=sigma,
        omega=omega,
    )

    acts_cuda = acts.clone().requires_grad_(True)
    acts_triton = acts.clone().requires_grad_(True)

    loss_val_cuda = loss_cuda(acts_cuda, labels, act_lens=act_lens, label_lens=label_lens)
    loss_val_triton = loss_triton(
        acts_triton, labels, act_lens=act_lens, label_lens=label_lens
    )

    loss_val_cuda.backward()
    loss_val_triton.backward()

    torch.cuda.synchronize()
    loss_diff = (loss_val_cuda - loss_val_triton).abs().item()
    grad_diff = (acts_cuda.grad - acts_triton.grad).abs().max().item()
    cuda_loss_finite = torch.isfinite(loss_val_cuda).all().item()
    triton_loss_finite = torch.isfinite(loss_val_triton).all().item()
    cuda_grad_finite = torch.isfinite(acts_cuda.grad).all().item()
    triton_grad_finite = torch.isfinite(acts_triton.grad).all().item()
    if not (cuda_loss_finite and triton_loss_finite and cuda_grad_finite and triton_grad_finite):
        _print_nonfinite_debug(
            "non_finite",
            acts,
            labels,
            act_lens,
            label_lens,
            blank_id,
            len(durations),
        )
    return (
        loss_diff,
        grad_diff,
        cuda_loss_finite,
        triton_loss_finite,
        cuda_grad_finite,
        triton_grad_finite,
    )


def _compare_case(
    batch,
    max_t,
    max_u,
    vocab,
    durations,
    fastemit_lambda=0.0,
    clamp=0.0,
    sigma=0.0,
    omega=0.0,
):
    acts, labels, act_lens, label_lens = _make_data(
        batch, max_t, max_u, vocab, len(durations)
    )
    blank_id = vocab - 1
    return _compare_data(
        acts,
        labels,
        act_lens,
        label_lens,
        blank_id,
        durations,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        sigma=sigma,
        omega=omega,
    )


def _compute_grads(loss_fn, acts, labels, act_lens, label_lens):
    logits = acts.clone().requires_grad_(True)
    loss = loss_fn(logits, labels, act_lens=act_lens, label_lens=label_lens)
    loss.backward()
    return loss.detach(), logits.grad.detach()


def _autograd_parity(acts, labels, act_lens, label_lens, loss_ref, loss_new):
    loss_ref_val, grads_ref = _compute_grads(
        loss_ref, acts, labels, act_lens, label_lens
    )
    loss_new_val, grads_new = _compute_grads(
        loss_new, acts, labels, act_lens, label_lens
    )

    diff = (grads_ref - grads_new).abs()
    max_abs = diff.max().item()
    denom = torch.maximum(grads_ref.abs(), grads_new.abs())
    max_rel = (diff / (denom + 1e-6)).max().item()
    nonfinite_ref = (~torch.isfinite(grads_ref)).any().item()
    nonfinite_new = (~torch.isfinite(grads_new)).any().item()
    return {
        "loss_ref": loss_ref_val.item(),
        "loss_new": loss_new_val.item(),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nonfinite_ref": nonfinite_ref,
        "nonfinite_new": nonfinite_new,
        "grads_ref": grads_ref,
        "grads_new": grads_new,
    }


def _finite_diff_check(
    loss_fn, acts, labels, act_lens, label_lens, grads, num_checks=5, eps=1e-3
):
    bsz, t_sz, u_sz, v_sz = acts.shape
    max_abs = 0.0
    max_rel = 0.0
    sign_mismatch = 0
    nonfinite = 0

    for _ in range(num_checks):
        b = torch.randint(0, bsz, (1,)).item()
        t = torch.randint(0, t_sz, (1,)).item()
        u = torch.randint(0, u_sz, (1,)).item()
        v = torch.randint(0, v_sz, (1,)).item()

        acts_plus = acts.clone()
        acts_minus = acts.clone()
        acts_plus[b, t, u, v] += eps
        acts_minus[b, t, u, v] -= eps

        with torch.no_grad():
            loss_plus = loss_fn(
                acts_plus, labels, act_lens=act_lens, label_lens=label_lens
            ).item()
            loss_minus = loss_fn(
                acts_minus, labels, act_lens=act_lens, label_lens=label_lens
            ).item()

        num_grad = (loss_plus - loss_minus) / (2.0 * eps)
        auto_grad = grads[b, t, u, v].item()

        if not (torch.isfinite(torch.tensor(num_grad)) and torch.isfinite(torch.tensor(auto_grad))):
            nonfinite += 1
            continue

        abs_err = abs(num_grad - auto_grad)
        rel_err = abs_err / (abs(auto_grad) + 1e-6)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)

        if num_grad == 0.0 or auto_grad == 0.0:
            continue
        if (num_grad > 0) != (auto_grad > 0):
            sign_mismatch += 1

    return max_abs, max_rel, sign_mismatch, nonfinite


def _grad_stats(grads):
    bsz = grads.shape[0]
    flat = grads.view(bsz, -1)
    norms = flat.norm(dim=1)
    return norms.mean().item(), norms.var(unbiased=False).item()


def _warmup(loss_fn, acts, labels, act_lens, label_lens, iters=5):
    for _ in range(iters):
        logits = acts.clone().requires_grad_(True)
        loss = loss_fn(logits, labels, act_lens=act_lens, label_lens=label_lens)
        loss.backward()
    torch.cuda.synchronize()


def _measure_memory(loss_fn, acts, labels, act_lens, label_lens):
    device = acts.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    logits = acts.clone().requires_grad_(True)
    loss = loss_fn(logits, labels, act_lens=act_lens, label_lens=label_lens)
    loss.backward()
    torch.cuda.synchronize(device)
    return (
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
    )


def _benchmark(loss_fn, acts, labels, act_lens, label_lens, iters=20):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        logits = acts.clone().requires_grad_(True)
        loss = loss_fn(logits, labels, act_lens=act_lens, label_lens=label_lens)
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


def _format_mem(value_bytes):
    return f"{value_bytes / (1024 ** 2):.1f}MiB"


def main():
    _check_env()
    torch.manual_seed(0)

    durations = [0, 1, 2, 3, 4]
    random_cases = [
        (2, 32, 16, 64),
        (2, 64, 32, 128),
        (4, 64, 32, 256),
        (8, 64, 32, 512),
        (4, 200, 64, 1024),
    ]
    train_like_cases = [
        ("train_like", 4, 200, 64, 512, True, None, False),
        ("train_like_safe", 4, 200, 64, 512, False, 63, False),
        ("train_like_unk", 4, 200, 64, 512, False, 63, True),
    ]
    scaled_cases = [
        ("scaled_5x", 2, 200, 64, 512, 5.0),
        ("scaled_20x", 2, 200, 64, 512, 20.0),
        ("scaled_80x", 1, 200, 64, 512, 80.0),
    ]
    fastemit_lambda = 0.1
    clamp = 0.0
    sigma = 0.0
    omega = 0.0
    warmup_iters = 5
    timed_iters = 20

    print("TDT Triton vs CUDA correctness (random lengths)")
    for batch, max_t, max_u, vocab in random_cases:
        (
            loss_diff,
            grad_diff,
            cuda_loss_finite,
            triton_loss_finite,
            cuda_grad_finite,
            triton_grad_finite,
        ) = _compare_case(
            batch,
            max_t,
            max_u,
            vocab,
            durations,
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        print(
            f"case B={batch} T={max_t} U={max_u} V={vocab} "
            f"loss_diff={loss_diff:.6f} grad_max_diff={grad_diff:.6f} "
            f"cuda_finite={cuda_loss_finite and cuda_grad_finite} "
            f"triton_finite={triton_loss_finite and triton_grad_finite}"
        )

    print("\nTDT Triton gradient correctness (autograd parity + finite diff)")
    for batch, max_t, max_u, vocab in random_cases[:3]:
        acts, labels, act_lens, label_lens = _make_data(
            batch, max_t, max_u, vocab, len(durations)
        )
        blank_id = vocab - 1
        loss_ref = TDTLoss(
            blank_id=blank_id,
            durations=durations,
            reduction="mean",
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        loss_new = TDTTritonLoss(
            blank_id=blank_id,
            durations=durations,
            reduction="mean",
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        parity = _autograd_parity(acts, labels, act_lens, label_lens, loss_ref, loss_new)
        fd_abs, fd_rel, fd_sign, fd_nonfinite = _finite_diff_check(
            loss_new,
            acts,
            labels,
            act_lens,
            label_lens,
            parity["grads_new"],
            num_checks=5,
            eps=1e-3,
        )
        print(
            f"case B={batch} T={max_t} U={max_u} V={vocab} "
            f"max_abs={parity['max_abs']:.6f} max_rel={parity['max_rel']:.6f} "
            f"nonfinite_ref={parity['nonfinite_ref']} nonfinite_new={parity['nonfinite_new']}"
        )
        print(
            f"  finite_diff: max_abs={fd_abs:.6f} max_rel={fd_rel:.6f} "
            f"sign_mismatch={fd_sign} nonfinite={fd_nonfinite}"
        )
        mean_norm, var_norm = _grad_stats(parity["grads_new"])
        print(f"  grad_scale: mean_norm={mean_norm:.6f} var_norm={var_norm:.6f}")

    print("\nTDT Triton gradient scale sanity (increasing length)")
    length_cases = [
        (1, 32, 16, 64),
        (1, 64, 32, 128),
        (1, 128, 64, 256),
        (4, 200, 64, 512),
    ]
    for batch, max_t, max_u, vocab in length_cases:
        acts, labels, act_lens, label_lens = _make_data(
            batch, max_t, max_u, vocab, len(durations)
        )
        loss_triton = TDTTritonLoss(
            blank_id=vocab - 1,
            durations=durations,
            reduction="mean",
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        _, grads_new = _compute_grads(loss_triton, acts, labels, act_lens, label_lens)
        mean_norm, var_norm = _grad_stats(grads_new)
        print(
            f"case B={batch} T={max_t} U={max_u} V={vocab} "
            f"mean_norm={mean_norm:.6f} var_norm={var_norm:.6f}"
        )

    print("\nTDT Triton vs CUDA correctness (train-like padding)")
    for name, batch, max_t, max_u, vocab, force_max_label, max_label_len, allow_zero_tokens in train_like_cases:
        acts, labels, act_lens, label_lens = _make_training_like_data(
            batch,
            max_t,
            max_u,
            vocab,
            len(durations),
            pad_id=0,
            max_label_len=max_label_len,
            force_max_label=force_max_label,
            allow_zero_tokens=allow_zero_tokens,
        )
        blank_id = vocab - 1
        (
            loss_diff,
            grad_diff,
            cuda_loss_finite,
            triton_loss_finite,
            cuda_grad_finite,
            triton_grad_finite,
        ) = _compare_data(
            acts,
            labels,
            act_lens,
            label_lens,
            blank_id,
            durations,
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        print(
            f"case {name} B={batch} T={max_t} U={max_u} V={vocab} "
            f"label_lens=[{label_lens.min().item()},{label_lens.max().item()}] "
            f"loss_diff={loss_diff:.6f} grad_max_diff={grad_diff:.6f} "
            f"cuda_finite={cuda_loss_finite and cuda_grad_finite} "
            f"triton_finite={triton_loss_finite and triton_grad_finite}"
        )

    print("\nTDT Triton vs CUDA correctness (scaled logits)")
    for name, batch, max_t, max_u, vocab, scale in scaled_cases:
        acts, labels, act_lens, label_lens = _make_training_like_data(
            batch,
            max_t,
            max_u,
            vocab,
            len(durations),
            pad_id=0,
            max_label_len=max_u - 1,
            force_max_label=False,
            allow_zero_tokens=False,
        )
        acts = acts * scale
        blank_id = vocab - 1
        (
            loss_diff,
            grad_diff,
            cuda_loss_finite,
            triton_loss_finite,
            cuda_grad_finite,
            triton_grad_finite,
        ) = _compare_data(
            acts,
            labels,
            act_lens,
            label_lens,
            blank_id,
            durations,
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        print(
            f"case {name} B={batch} T={max_t} U={max_u} V={vocab} "
            f"scale={scale:.1f} loss_diff={loss_diff:.6f} grad_max_diff={grad_diff:.6f} "
            f"cuda_finite={cuda_loss_finite and cuda_grad_finite} "
            f"triton_finite={triton_loss_finite and triton_grad_finite}"
        )

    print("\nTDT Triton vs CUDA speed (ms, forward+backward)")
    for batch, max_t, max_u, vocab in random_cases:
        acts, labels, act_lens, label_lens = _make_data(
            batch, max_t, max_u, vocab, len(durations)
        )
        blank_id = vocab - 1
        loss_cuda = TDTLoss(
            blank_id=blank_id,
            durations=durations,
            reduction="mean",
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        loss_triton = TDTTritonLoss(
            blank_id=blank_id,
            durations=durations,
            reduction="mean",
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        _warmup(loss_cuda, acts, labels, act_lens, label_lens, iters=warmup_iters)
        _warmup(loss_triton, acts, labels, act_lens, label_lens, iters=warmup_iters)
        cuda_mem = _measure_memory(loss_cuda, acts, labels, act_lens, label_lens)
        triton_mem = _measure_memory(loss_triton, acts, labels, act_lens, label_lens)
        cuda_ms = _benchmark(loss_cuda, acts, labels, act_lens, label_lens, iters=timed_iters)
        triton_ms = _benchmark(
            loss_triton, acts, labels, act_lens, label_lens, iters=timed_iters
        )
        print(
            f"case B={batch} T={max_t} U={max_u} V={vocab} "
            f"cuda={cuda_ms:.2f}ms triton={triton_ms:.2f}ms",
        )
        print(
            f"  cuda_peak_alloc={_format_mem(cuda_mem[0])} "
            f"cuda_peak_reserved={_format_mem(cuda_mem[1])} "
            f"triton_peak_alloc={_format_mem(triton_mem[0])} "
            f"triton_peak_reserved={_format_mem(triton_mem[1])}"
        )

    print("\nTDT Triton vs CUDA speed (train-like padding)")
    for name, batch, max_t, max_u, vocab, force_max_label, max_label_len, allow_zero_tokens in train_like_cases:
        acts, labels, act_lens, label_lens = _make_training_like_data(
            batch,
            max_t,
            max_u,
            vocab,
            len(durations),
            pad_id=0,
            max_label_len=max_label_len,
            force_max_label=force_max_label,
            allow_zero_tokens=allow_zero_tokens,
        )
        blank_id = vocab - 1
        loss_cuda = TDTLoss(
            blank_id=blank_id,
            durations=durations,
            reduction="mean",
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        loss_triton = TDTTritonLoss(
            blank_id=blank_id,
            durations=durations,
            reduction="mean",
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        _warmup(loss_cuda, acts, labels, act_lens, label_lens, iters=warmup_iters)
        _warmup(loss_triton, acts, labels, act_lens, label_lens, iters=warmup_iters)
        cuda_mem = _measure_memory(loss_cuda, acts, labels, act_lens, label_lens)
        triton_mem = _measure_memory(loss_triton, acts, labels, act_lens, label_lens)
        cuda_ms = _benchmark(loss_cuda, acts, labels, act_lens, label_lens, iters=timed_iters)
        triton_ms = _benchmark(
            loss_triton, acts, labels, act_lens, label_lens, iters=timed_iters
        )
        print(
            f"case {name} B={batch} T={max_t} U={max_u} V={vocab} "
            f"cuda={cuda_ms:.2f}ms triton={triton_ms:.2f}ms",
        )
        print(
            f"  cuda_peak_alloc={_format_mem(cuda_mem[0])} "
            f"cuda_peak_reserved={_format_mem(cuda_mem[1])} "
            f"triton_peak_alloc={_format_mem(triton_mem[0])} "
            f"triton_peak_reserved={_format_mem(triton_mem[1])}"
        )


if __name__ == "__main__":
    main()
