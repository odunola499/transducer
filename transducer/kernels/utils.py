from enum import Enum

import numpy as np
from numba import float32

# https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/numba/rnnt_loss/utils/global_constants.py

_THREADS_PER_BLOCK = 32
_WARP_SIZE = 32
_DTYPE = float32

FP32_INF = np.inf
FP32_NEG_INF = -np.inf
THRESHOLD = 1e-1


def threads_per_block():
    global _THREADS_PER_BLOCK
    return _THREADS_PER_BLOCK


def warp_size():
    global _WARP_SIZE
    return _WARP_SIZE


def dtype():
    global _DTYPE
    return _DTYPE


class RNNTStatus(Enum):
    RNNT_STATUS_SUCCESS = 0
    RNNT_STATUS_INVALID_VALUE = 1
