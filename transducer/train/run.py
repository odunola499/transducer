import argparse
import os
from typing import Optional

import torch
import torch.distributed as dist
from rich.console import Console

from transducer.config import Config
from transducer.dataset import HFDataset
from transducer.dataset.config import DatasetConfig, HFDatasetStruct
from transducer.models import DawnModel
from transducer.train.train_module import TrainModule


def _init_distributed() -> Optional[int]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size == 1:
        return None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def _build_dataloaders(config: DatasetConfig, train_batch_size: int, eval_batch_size: int):
    if config.dataset_type != "hf":
        raise NotImplementedError("Only HF datasets are supported right now.")
    train_data = config.train_data
    val_data = config.val_data
    if not isinstance(train_data, HFDatasetStruct):
        raise ValueError("train_data must be HFDatasetStruct when dataset_type='hf'.")
    if not isinstance(val_data, HFDatasetStruct):
        raise ValueError("val_data must be HFDatasetStruct when dataset_type='hf'.")

    train_dataset = HFDataset(train_data, config)
    val_dataset = HFDataset(val_data, config)
    train_loader = train_dataset.get_loader(train_batch_size)
    val_loader = val_dataset.get_loader(eval_batch_size)
    return train_loader, val_loader, train_dataset


def _build_model(config: Config, vocab_size: int):
    model_config = config.model
    if model_config.decoder_config.vocab_size != vocab_size:
        model_config.decoder_config.vocab_size = vocab_size
    return DawnModel(vocab_size, model_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = Config.from_yaml_file(args.config)
    local_rank = _init_distributed()

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    console = Console()

    train_loader, val_loader, train_dataset = _build_dataloaders(
        config.dataset,
        config.train.per_device_train_batch_size,
        config.train.per_device_eval_batch_size,
    )

    model = _build_model(config, config.dataset.tokenizer_config.vocab_size)
    model.feature_extractor = train_dataset.processor.feature_extractor
    model.tokenizer = train_dataset.processor.tokenizer

    trainer = TrainModule(
        model=model,
        config=config.train,
        train_loader=train_loader,
        valid_loader=val_loader,
        device=device,
        console=console,
        local_rank=local_rank or 0,
    )
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
