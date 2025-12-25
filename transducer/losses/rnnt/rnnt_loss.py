import multiprocessing

import torch
from numba import cuda
from torch import Tensor
from torch.autograd import Function

from transducer.commons import Loss
from transducer.kernels.cpu_kernels.cpu_rnnt import CPURNNT
from transducer.kernels.gpu_kernels.gpu_rnnt import GPURNNT
from transducer.kernels.gpu_kernels.helpers import flatten_tensor, get_workspace_size
from transducer.kernels.utils import RNNTStatus


# https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/numba/rnnt_loss/rnnt.py
def rnnt_loss_cpu(
    acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    costs: torch.Tensor,
    grads: torch.Tensor,
    blank_label: int,
    fastemit_lambda: float,
    clamp: float,
    num_threads: int,
):
    log_probs = acts
    flat_labels = labels

    minibatch_size = log_probs.shape[0]
    maxT = log_probs.shape[1]
    maxU = log_probs.shape[2]
    alphabet_size = log_probs.shape[3]

    if num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    num_threads = max(1, num_threads)
    gpu_size, status = get_workspace_size(maxT, maxU, minibatch_size, gpu=False)
    if status != RNNTStatus.RNNT_STATUS_SUCCESS:
        raise RuntimeError("Invalid parameter passed when calculating working space memory")

    cpu_workspace = torch.zeros(
        gpu_size, device=log_probs.device, dtype=log_probs.dtype, requires_grad=False
    )

    log_probs, acts_shape = flatten_tensor(log_probs)
    flat_labels, labels_shape = flatten_tensor(flat_labels)

    wrapper = CPURNNT(
        minibatch=minibatch_size,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=alphabet_size,
        workspace=cpu_workspace,
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=num_threads,
        batch_first=True,
    )

    if grads is None:
        status = wrapper.score_forward(
            log_probs=log_probs.data,
            costs=costs,
            flat_labels=flat_labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    else:
        grads, grads_shape = flatten_tensor(grads)

        status = wrapper.cost_and_grad(
            log_probs=log_probs.data,
            grads=grads.data,
            costs=costs,
            flat_labels=flat_labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    del cpu_workspace, wrapper
    return True


def rnnt_loss_gpu(
    acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    costs: torch.Tensor,
    grads: torch.Tensor,
    blank_label: int,
    fastemit_lambda: float,
    clamp: float,
    num_threads: int,
):
    """
    Wrapper method for accessing GPU RNNT loss.

    CUDA implementation ported from [HawkAaron/warp-transducer](https://github.com/HawkAaron/warp-transducer).

    Args:
        acts: Activation tensor of shape [B, T, U, V+1].
        labels: Ground truth labels of shape [B, U].
        input_lengths: Lengths of the acoustic sequence as a vector of ints [B].
        label_lengths: Lengths of the target sequence as a vector of ints [B].
        costs: Zero vector of length [B] in which costs will be set.
        grads: Zero tensor of shape [B, T, U, V+1] where the gradient will be set.
        blank_label: Index of the blank token in the vocabulary.
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
        clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].
        num_threads: Number of threads for OpenMP.
    """
    minibatch_size = acts.shape[0]
    maxT = acts.shape[1]
    maxU = acts.shape[2]
    alphabet_size = acts.shape[3]

    if hasattr(cuda, "external_stream"):
        stream = cuda.external_stream(torch.cuda.current_stream(acts.device).cuda_stream)
    else:
        stream = cuda.default_stream()

    if num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    num_threads = max(1, num_threads)  # have to use at least 1 thread

    gpu_size, status = get_workspace_size(maxT, maxU, minibatch_size, gpu=True)
    if status != RNNTStatus.RNNT_STATUS_SUCCESS:
        raise RuntimeError("Invalid parameter passed when calculating working space memory")

    # Select GPU index
    cuda.select_device(acts.device.index)
    gpu_workspace = torch.zeros(
        gpu_size, device=acts.device, dtype=torch.float32, requires_grad=False
    )

    ### VIEW TENSORS AS VECTORS FOR POINTER INDEXING ###
    acts, acts_shape = flatten_tensor(acts)

    wrapper = GPURNNT(
        minibatch=minibatch_size,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=alphabet_size,
        workspace=gpu_workspace,
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=num_threads,
        stream=stream,
    )

    if grads is None:
        status = wrapper.score_forward(
            acts=acts.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    else:
        ### FLATTEN GRAD TENSOR ###
        grads, grads_shape = flatten_tensor(grads)

        status = wrapper.cost_and_grad(
            acts=acts.data,
            grads=grads.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    del gpu_workspace, wrapper
    return True


class _RNNTNumba(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, blank, reduction, fastemit_lambda, clamp):
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
        """
        is_cuda = acts.is_cuda

        if clamp < 0:
            raise ValueError("`clamp` must be 0.0 or positive float value.")

        loss_func = rnnt_loss_gpu if is_cuda else rnnt_loss_cpu
        grads = torch.zeros_like(acts) if acts.requires_grad else None
        minibatch_size = acts.size(0)
        costs = torch.zeros(minibatch_size, device=acts.device, dtype=torch.float32)

        loss_func(
            acts,
            labels=labels,
            input_lengths=act_lens,
            label_lengths=label_lens,
            costs=costs,
            grads=grads,
            blank_label=blank,
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            num_threads=0,
        )

        if reduction in ["sum", "mean"]:
            costs = costs.sum().unsqueeze_(-1)
            if reduction == "mean":
                costs /= minibatch_size

                if grads is not None:
                    grads /= minibatch_size

        ctx.save_for_backward(grads)

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        (grads,) = ctx.saved_tensors
        if grad_output is not None and grads is not None:
            grad_output = grad_output.view(-1, 1, 1, 1).to(grads)
            return grads.mul_(grad_output), None, None, None, None, None, None, None


class RNNTLoss(Loss):
    def __init__(
        self,
        blank_id,
        reduction="mean",
        fastemit_lambda: float = 0.0,
        clamp: float = 0.0,
    ):
        super().__init__()
        self.blank = blank_id
        self.reduction = reduction
        self.fastemit_lambda = fastemit_lambda
        self.clamp = clamp
        self.reduction = reduction
        self.loss = _RNNTNumba.apply

    def forward(self, acts: Tensor, labels: Tensor, label_lens, act_lens: Tensor = None):
        # Lazy, we take all frames as important for now.
        if act_lens is None:
            batch_size, num_frames = acts.shape[:2]
            act_lens  = torch.tensor([num_frames]* batch_size, device=acts.device, dtype=torch.long)
        if not acts.is_cuda:
            if acts.dtype == torch.float16:
                acts = acts.float()
            # NOTE manually done log_softmax for CPU version,
            # log_softmax is computed within GPU version.
            acts = torch.nn.functional.log_softmax(acts, -1)

        return self.loss(
            acts,
            labels,
            act_lens,
            label_lens,
            self.blank,
            self.reduction,
            self.fastemit_lambda,
            self.clamp,
        )


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loss_func = RNNTLoss(blank_id=0)
    B, T, U, vocab_size = 4, 10, 7, 16
    lattice = torch.randn(B, T, U, vocab_size).to(device)
    lattice = torch.log_softmax(lattice, dim=-1)
    labels = torch.randint(0, vocab_size, (B, U)).to(device)
    lattice_lens = torch.randint(2, T, (B,)).to(device)
    label_lens = torch.randint(2, U, (B,)).to(device)

    print(lattice_lens, label_lens)
    output = loss_func(lattice, labels, lattice_lens, label_lens)
    print(output)
