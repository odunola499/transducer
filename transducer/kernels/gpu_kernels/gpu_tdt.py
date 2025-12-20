import multiprocessing
import random
from typing import Optional, Tuple

import numba
import torch
from numba import cuda

from transducer.kernels import utils
from transducer.kernels.gpu_kernels import global_kernels as gpu_rnnt_kernel
from transducer.kernels.gpu_kernels import helpers as rnnt_helper
from transducer.kernels.gpu_kernels import reduce


class GPUTDT:
    def __init__(
        self,
        sigma: float,
        omega: float,
        num_durations: int,
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        workspace,
        tdt_workspace,
        blank: int,
        fastemit_lambda: float,
        clamp: float,
        num_threads: int,
        stream,
    ):
        super().__init__()

        self.minibatch_ = minibatch

        self.maxT_ = maxT

        self.maxU_ = maxU

        self.alphabet_size_ = alphabet_size

        self.gpu_workspace = cuda.as_cuda_array(
            workspace
        )  # a flat vector of floatX numbers that represents allocated memory slices

        self.blank_ = blank

        self.fastemit_lambda_ = fastemit_lambda

        self.clamp_ = abs(clamp)

        self.num_threads_ = num_threads

        self.stream_ = stream  # type: cuda.cudadrv.driver.Stream

        _torch_num_threads = torch.get_num_threads()

        if num_threads > 0:
            numba.set_num_threads(min(multiprocessing.cpu_count(), num_threads))

            self.num_threads_ = numba.get_num_threads()

        else:
            self.num_threads_ = numba.get_num_threads()

        torch.set_num_threads(_torch_num_threads)

        self.tdt_workspace = cuda.as_cuda_array(
            tdt_workspace
        )  # a flat vector of integer numbers that represents allocated memory slices

        self.num_durations = num_durations

        self.sigma = sigma

        self.omega = omega

    def log_softmax(self, acts: torch.Tensor, denom: torch.Tensor):
        # // trans_acts + pred_acts -> log_softmax denominator

        reduce.reduce_max(
            acts,
            denom,
            rows=self.alphabet_size_,
            cols=self.minibatch_ * self.maxT_ * self.maxU_,
            minus=False,
            stream=self.stream_,
        )

        reduce.reduce_exp(
            acts,
            denom,
            rows=self.alphabet_size_,
            cols=self.minibatch_ * self.maxT_ * self.maxU_,
            minus=True,
            stream=self.stream_,
        )

    def compute_cost_and_score(
        self,
        label_acts: torch.Tensor,
        duration_acts: torch.Tensor,
        label_grads: Optional[torch.Tensor],
        duration_grads: Optional[torch.Tensor],
        costs: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> utils.RNNTStatus:
        training = label_grads is not None

        if training:
            label_grads *= 0.0  # zero grads

            duration_grads *= 0.0  # zero grads

        _, (denom, alphas, betas, llForward, llBackward, durations) = self._prepare_workspace()

        ######## START EXECUTION ########

        self.log_softmax(label_acts, denom)

        r = random.uniform(0, 1)

        if r < self.omega:
            # Compute alphas

            gpu_rnnt_kernel.compute_alphas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                label_acts,
                denom,
                alphas,
                llForward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
            )

        else:
            # Compute alphas

            gpu_rnnt_kernel.compute_tdt_alphas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                label_acts,
                duration_acts,
                denom,
                self.sigma,
                alphas,
                llForward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
                durations,
                self.num_durations,
            )

        if training:
            # Compute betas

            if r < self.omega:
                gpu_rnnt_kernel.compute_betas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                    label_acts,
                    denom,
                    betas,
                    llBackward,
                    input_lengths,
                    label_lengths,
                    labels,
                    self.minibatch_,
                    self.maxT_,
                    self.maxU_,
                    self.alphabet_size_,
                    self.blank_,
                )

                # Compute gradient

                grad_blocks_per_grid = self.minibatch_ * self.maxT_ * self.maxU_

                grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE

                gpu_rnnt_kernel.compute_grad_kernel[
                    grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0
                ](
                    label_grads,
                    label_acts,
                    denom,
                    alphas,
                    betas,
                    llForward,
                    input_lengths,
                    label_lengths,
                    labels,
                    self.minibatch_,
                    self.maxT_,
                    self.maxU_,
                    self.alphabet_size_,
                    self.blank_,
                    self.fastemit_lambda_,
                    self.clamp_,
                )

            else:
                gpu_rnnt_kernel.compute_tdt_betas_kernel[
                    self.minibatch_, self.maxU_, self.stream_, 0
                ](
                    label_acts,
                    duration_acts,
                    denom,
                    self.sigma,
                    betas,
                    llBackward,
                    input_lengths,
                    label_lengths,
                    labels,
                    self.minibatch_,
                    self.maxT_,
                    self.maxU_,
                    self.alphabet_size_,
                    self.blank_,
                    durations,
                    self.num_durations,
                )

                # Compute gradient

                grad_blocks_per_grid = self.minibatch_ * self.maxT_ * self.maxU_

                grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE

                gpu_rnnt_kernel.compute_tdt_grad_kernel[
                    grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0
                ](
                    label_grads,
                    duration_grads,
                    label_acts,
                    duration_acts,
                    denom,
                    self.sigma,
                    alphas,
                    betas,
                    llForward,
                    input_lengths,
                    label_lengths,
                    labels,
                    self.minibatch_,
                    self.maxT_,
                    self.maxU_,
                    self.alphabet_size_,
                    self.blank_,
                    durations,
                    self.num_durations,
                    self.fastemit_lambda_,
                    self.clamp_,
                )

        # // cost copy, negate (for log likelihood) and update with additional regularizers

        # This needs to be done via CUDA, because we used temporary memory llForward

        # passed to alpha, which was updated with log likelihoods.

        # But copying this data into a pytorch pointer is more difficult (numba api is one way)

        # Therefore launch a pointwise CUDA kernel to update the costs inplace from data of llForward

        # Then negate to compute the loglikelihood.

        threadsperblock = min(costs.shape[0], 32)

        blockspergrid = (costs.shape[0] + (threadsperblock - 1)) // threadsperblock

        rnnt_helper.compute_costs_data[blockspergrid, threadsperblock, self.stream_, 0](
            llForward, costs, self.fastemit_lambda_
        )

        self.stream_.synchronize()

        return utils.RNNTStatus.RNNT_STATUS_SUCCESS

    def cost_and_grad(
        self,
        label_acts: torch.Tensor,
        duration_acts: torch.Tensor,
        label_grads: torch.Tensor,
        duration_grads: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        if (
            duration_acts is None
            or label_acts is None
            or label_grads is None
            or duration_grads is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return utils.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(
            label_acts,
            duration_acts,
            label_grads,
            duration_grads,
            costs,
            pad_labels,
            label_lengths,
            input_lengths,
        )

    def score_forward(
        self,
        label_acts: torch.Tensor,
        duration_acts: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        if (
            label_acts is None
            or duration_acts is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return utils.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(
            label_acts, duration_acts, None, None, costs, pad_labels, label_lengths, input_lengths
        )

    def _prepare_workspace(self) -> Tuple[int, Tuple[torch.Tensor, ...]]:
        used_offset, (denom, alphas, betas, llForward, llBackward) = super()._prepare_workspace()

        durations = self.tdt_workspace[: self.num_durations]

        return used_offset, (denom, alphas, betas, llForward, llBackward, durations)
