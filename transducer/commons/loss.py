from torch import nn


class Loss(nn.modules.loss._Loss, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
