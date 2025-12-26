import time

import torch

from transducer.losses.rnnt.rnnt_loss import RNNTLoss
from transducer.losses.rnnt.rnnt_triton import RNNTTritonLoss


def _check_env():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    try:
        import triton  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Triton is required for this benchmark.") from exc


def _make_data(batch, max_t, max_u, vocab):
    device = torch.device("cuda")
    acts = torch.randn(batch, max_t, max_u, vocab, device=device, dtype=torch.float32)
    labels = torch.randint(0, vocab - 1, (batch, max_u - 1), device=device, dtype=torch.long)
    act_lens = torch.randint(1, max_t + 1, (batch,), device=device, dtype=torch.long)
    label_lens = torch.randint(1, max_u, (batch,), device=device, dtype=torch.long)
    return acts, labels, act_lens, label_lens


def _compare_case(batch, max_t, max_u, vocab, fastemit_lambda=0.0, clamp=0.0):
    acts, labels, act_lens, label_lens = _make_data(batch, max_t, max_u, vocab)
    blank_id = vocab - 1

    loss_cuda = RNNTLoss(
        blank_id=blank_id,
        reduction="mean",
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
    )
    loss_triton = RNNTTritonLoss(
        blank_id=blank_id,
        reduction="mean",
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
    )

    acts_cuda = acts.clone().requires_grad_(True)
    acts_triton = acts.clone().requires_grad_(True)

    loss_val_cuda = loss_cuda(
        acts_cuda, labels, label_lens=label_lens, act_lens=act_lens
    )
    loss_val_triton = loss_triton(
        acts_triton, labels, act_lens=act_lens, label_lens=label_lens
    )

    loss_val_cuda.backward()
    loss_val_triton.backward()

    torch.cuda.synchronize()
    loss_diff = (loss_val_cuda - loss_val_triton).abs().item()
    grad_diff = (acts_cuda.grad - acts_triton.grad).abs().max().item()
    return loss_diff, grad_diff


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


def main():
    _check_env()
    torch.manual_seed(0)

    cases = [
        (2, 32, 16, 64),
        (2, 64, 32, 128),
        (4, 64, 32, 256),
    ]
    fastemit_lambda = 0.1
    clamp = 0.0

    print("RNNT Triton vs CUDA correctness")
    for batch, max_t, max_u, vocab in cases:
        loss_diff, grad_diff = _compare_case(
            batch, max_t, max_u, vocab, fastemit_lambda=fastemit_lambda, clamp=clamp
        )
        print(
            f"case B={batch} T={max_t} U={max_u} V={vocab} "
            f"loss_diff={loss_diff:.6f} grad_max_diff={grad_diff:.6f}"
        )

    print("\nRNNT Triton vs CUDA speed (ms, forward+backward)")
    for batch, max_t, max_u, vocab in cases:
        acts, labels, act_lens, label_lens = _make_data(batch, max_t, max_u, vocab)
        blank_id = vocab - 1
        loss_cuda = RNNTLoss(
            blank_id=blank_id,
            reduction="mean",
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
        )
        loss_triton = RNNTTritonLoss(
            blank_id=blank_id,
            reduction="mean",
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
        )
        cuda_ms = _benchmark(loss_cuda, acts, labels, act_lens, label_lens)
        triton_ms = _benchmark(loss_triton, acts, labels, act_lens, label_lens)
        print(
            f"case B={batch} T={max_t} U={max_u} V={vocab} "
            f"cuda={cuda_ms:.2f}ms triton={triton_ms:.2f}ms"
        )


if __name__ == "__main__":
    main()
