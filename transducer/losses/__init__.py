from transducer.losses.rnnt.rnnt_loss import RNNTLoss
from transducer.losses.rnnt.rnnt_triton import RNNTTritonLoss
from transducer.losses.tdt.tdt_loss import TDTLoss


LOSSES = {
    'tdt': TDTLoss,
    'rnnt': RNNTLoss,
    'rnnt_triton': RNNTTritonLoss,
}
