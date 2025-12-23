import multiprocessing

import torch
from numba import cuda
from torch import Tensor
from torch.autograd import Function

from transducer.commons import Loss
from transducer.kernels.gpu_kernels.gpu_tdt import GPUTDT
from transducer.kernels.gpu_kernels.helpers import flatten_tensor, get_workspace_size
from transducer.kernels.utils import RNNTStatus

def tdt_loss_gpu(
    label_acts: torch.Tensor,
    duration_acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    costs: torch.Tensor,
    label_grads: torch.Tensor,
    duration_grads: torch.Tensor,
    blank_label: int,
    durations: list,
    fastemit_lambda: float,
    clamp: float,
    num_threads: int,
    sigma: float,
    omega: float,
):
    minibatch_size = label_acts.shape[0]

    maxT = label_acts.shape[1]

    maxU = label_acts.shape[2]

    alphabet_size = label_acts.shape[3]

    if hasattr(cuda, "external_stream"):
        stream = cuda.external_stream(torch.cuda.current_stream(label_acts.device).cuda_stream)

    else:
        stream = cuda.default_stream()

    if num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    num_threads = max(1, num_threads)  # have to use at least 1 thread

    gpu_size, status = get_workspace_size(maxT, maxU, minibatch_size, gpu=True)

    if status != RNNTStatus.RNNT_STATUS_SUCCESS:
        raise RuntimeError("Invalid parameter passed when calculating working space memory")

    # Select GPU index

    cuda.select_device(label_acts.device.index)

    gpu_workspace = torch.zeros(
        gpu_size, device=label_acts.device, dtype=label_acts.dtype, requires_grad=False
    )

    tdt_workspace = torch.zeros(
        len(durations), device=label_acts.device, dtype=torch.long, requires_grad=False
    )

    for i in range(0, len(durations)):
        tdt_workspace[i] = durations[i]

    ### VIEW TENSORS AS VECTORS FOR POINTER INDEXING ###

    label_acts, label_acts_shape = flatten_tensor(label_acts)

    duration_acts, duration_acts_shape = flatten_tensor(duration_acts)

    wrapper = GPUTDT(
        minibatch=minibatch_size,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=alphabet_size,
        workspace=gpu_workspace,
        tdt_workspace=tdt_workspace,
        num_durations=len(durations),
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=num_threads,
        stream=stream,
        sigma=sigma,
        omega=omega,
    )

    if label_grads is None:
        status = wrapper.score_forward(
            label_acts=label_acts.data,
            duration_acts=duration_acts.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    else:
        ### FLATTEN GRAD TENSOR ###

        label_grads, label_grads_shape = flatten_tensor(label_grads)

        duration_grads, duration_grads_shape = flatten_tensor(duration_grads)

        status = wrapper.cost_and_grad(
            label_acts=label_acts.data,
            duration_acts=duration_acts.data,
            label_grads=label_grads.data,
            duration_grads=duration_grads.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    del gpu_workspace, tdt_workspace, wrapper

    return True


class _TDTNumba(Function):
    @staticmethod
    def forward(
        ctx,
        label_acts: Tensor,
        duration_acts: Tensor,
        labels: Tensor,
        act_lens: Tensor,
        label_lens: Tensor,
        blank: int,
        durations,
        reduction,
        fastemit_lambda,
        clamp,
        sigma,
        omega,
    ):
        is_cuda = label_acts.is_cuda

        if clamp < 0:
            raise ValueError("`clamp` must be 0.0 or positive float value.")

        if is_cuda:
            loss_func = tdt_loss_gpu

        else:
            raise ValueError("TDT is not yet implemented for non CUDA computation.")

        label_grads = torch.zeros_like(label_acts) if label_acts.requires_grad else None
        duration_grads = torch.zeros_like(duration_acts) if duration_acts.requires_grad else None
        minibatch_size = label_acts.size(0)

        costs = torch.zeros(minibatch_size, device=label_acts.device, dtype=label_acts.dtype)

        loss_func(
            label_acts,
            duration_acts,
            labels=labels,
            input_lengths=act_lens,
            label_lengths=label_lens,
            costs=costs,
            label_grads=label_grads,
            duration_grads=duration_grads,
            blank_label=blank,
            durations=durations,
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
            num_threads=0,
        )

        if reduction in ["sum", "mean"]:
            costs = costs.sum().unsqueeze_(-1)

            if reduction == "mean":
                costs /= minibatch_size

                if label_grads is not None:
                    label_grads /= minibatch_size

                    duration_grads /= minibatch_size

        ctx.save_for_backward(label_grads, duration_grads)

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        label_grads, duration_grads = ctx.saved_tensors

        if grad_output is not None and label_grads is not None:
            grad_output = grad_output.view(-1, 1, 1, 1).to(label_grads)

            return (
                label_grads.mul_(grad_output),
                duration_grads.mul_(grad_output),
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


class TDTLoss(Loss):
    def __init__(
        self,
        blank_id,
        durations=None,
        reduction="mean",
        fastemit_lambda: float = 0.0,
        clamp: float = -1,
        sigma: float = 0.0,
        omega: float = 0.0,
    ):
        super().__init__()

        self.blank = blank_id
        self.durations = durations if durations is not None else []
        self.fastemit_lambda = fastemit_lambda
        self.clamp = float(clamp) if clamp > 0 else 0.0
        self.reduction = reduction
        self.loss = _TDTNumba.apply
        self.sigma = sigma
        self.omega = omega

    def forward(self, acts, labels, act_lens, label_lens):
        # TODO(hainan): in the future, we could further optimize this so that we don't need to

        # Lazy, we take all frames as important for now.
        if act_lens is None:
            batch_size, num_frames = acts.shape[:-1]
            act_lens  = torch.tensor([num_frames]* batch_size, device=acts.device, dtype=torch.long)

        label_acts, duration_acts = torch.split(
            acts, [acts.shape[-1] - len(self.durations), len(self.durations)], dim=-1
        )

        label_acts = label_acts.contiguous()

        duration_acts = torch.nn.functional.log_softmax(duration_acts, dim=-1).contiguous()

        return self.loss(
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

if __name__ == "__main__":

    torch.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    durations = [0, 1, 2, 3, 4]
    label_vocab_size = 12
    B, T, U = 2, 48, 16

    lattice = torch.randn(B, T, U, label_vocab_size + len(durations), device=device)
    lattice = torch.log_softmax(lattice, dim=-1)

    labels = torch.randint(0, label_vocab_size, (B, U), device=device)
    lattice_lens = torch.randint(5, T, (B,), device=device)
    label_lens = torch.randint(5, U, (B,), device=device)

    loss_func = TDTLoss(
        blank_id=0,
        durations=durations,
        sigma=1.0,
        omega=0.5, 
        reduction="mean",
    )

    print("Input lengths:", lattice_lens)
    print("Label lengths:", label_lens)
    output = loss_func(lattice, labels, lattice_lens, label_lens)
    print("Loss:", output)
