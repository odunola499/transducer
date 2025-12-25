import os
from typing import Optional, Literal, Union
from datetime import datetime

from pydantic import StrictBool, StrictInt, StrictStr, StrictFloat
from transducer.config_base import Args

class TrainConfig(Args):
    per_device_train_batch_size: StrictInt = 8
    per_device_eval_batch_size: StrictInt = 8
    devices:Optional[StrictInt] = -1
    max_steps:Optional[StrictInt] = 1000
    num_warmup_steps:Union[int, float] = 0.1
    accumulate_grad_batches:Optional[StrictInt] = 8

    lr:Optional[StrictFloat] = 3e-4
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    strategy: Literal['ddp','fsdp'] = 'ddp'
    log_to:Literal['none','wandb'] = 'wandb'
    wandb_name:StrictStr = 'Voicera'
    wandb_project:StrictStr = 'Voicera'
    wandb_entity: Optional[StrictStr] = None

    log_every_num_steps:Optional[StrictInt] = 10
    num_sanity_val_steps:Optional[StrictInt] = 10
    check_val_every_num_steps:Optional[StrictInt] = 10

    # checkpointing
    enable_checkpointing: StrictBool = True
    monitor:Literal['val_loss', 'val_wer','train_loss', 'train_wer'] = 'val_wer'
    save_top_k:StrictInt = 5
    checkpoint_dir:StrictStr = './checkpoints'

    pretrained_checkpoint_dir:Optional[StrictStr] = None

    scheduler:Literal['linear'] = 'linear'
    optimizer:Literal['adam','sgd','adamw'] = 'adam'

    print_predictions: StrictBool = False
    print_predictions_every_num_steps: StrictInt = 200
    max_print_predictions: StrictInt = 2

    log_indices: StrictBool = False
    log_indices_every_num_steps: StrictInt = 50
    max_log_indices: StrictInt = 32

    def model_post_init(self, __context):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rank = os.getenv('RANK')
        self.wandb_name = f"{self.wandb_name}_{timestamp}_r{rank}"
