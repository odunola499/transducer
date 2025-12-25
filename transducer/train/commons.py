import torch
from transformers import get_linear_schedule_with_warmup

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'adamw': torch.optim.AdamW,

}

SCHEDULERS = {
    'linear': get_linear_schedule_with_warmup,
}