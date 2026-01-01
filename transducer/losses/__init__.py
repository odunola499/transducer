from transducer.losses.rnnt.rnnt_loss import RNNTLoss
from transducer.losses.rnnt.rnnt_triton import RNNTTritonLoss
from transducer.losses.tdt.tdt_loss import TDTLoss
from transducer.losses.tdt.tdt_triton import TDTTritonLoss

LOSSES = {
    "tdt": TDTLoss,
    "tdt_triton": TDTTritonLoss,
    "rnnt": RNNTLoss,
    "rnnt_triton": RNNTTritonLoss,
}
