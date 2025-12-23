import os
from dataclasses import dataclass
from typing import Optional, Literal, Union
from datetime import datetime

from transducer.commons import Args

@dataclass
class TrainConfig(Args):
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    devices:Optional[int] = -1
    max_epochs:Optional[int] = 10
    max_steps:Optional[int] = 1000
    accumulate_grad_batches:Optional[int] = 8

    strategy: Literal['ddp','fsdp'] = 'ddp'
    log_to:Literal['none','wandb'] = 'wandb'
    wandb_name:str = 'Voicera'
    wandb_project:str = 'Voicera'

    log_every_num_steps:Optional[int] = 10


    num_sanity_val_steps:Optional[int] = 10
    check_val_every_n_epoch:Optional[int] = 1
    check_val_every_num_steps:Optional[int] = 10

    # checkpointing
    enable_checkpointing: bool = True
    monitor:Literal['val_loss', 'val_wer','train_loss', 'train_wer'] = 'val_wer'
    save_top_k:int = 5
    checkpoint_dir:str = './checkpoints'

    pretrained_checkpoint_dir:Optional[str] = None

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rank = os.getenv('RANK')
        self.wandb_name = f"{self.wandb_name}_{timestamp}_r{rank}"



