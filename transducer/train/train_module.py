import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from jiwer import wer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from transducer.models import BaseModel
from transducer.train.commons import OPTIMIZERS, SCHEDULERS
from transducer.train.config import TrainConfig


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def _autocast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def _count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class TrainModule:
    def __init__(
        self,
        model: BaseModel,
        config: TrainConfig,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        device: torch.device,
        console: Optional[Console] = None,
        local_rank: int = 0,
    ):
        self.config = config
        self.device = device
        self.console = console or Console()
        self.local_rank = local_rank
        self.rank = _get_rank()
        self.world_size = _get_world_size()

        self.model = model.to(self.device)
        if is_dist() and config.strategy == "ddp":
            self.model = DDP(self.model, device_ids=[local_rank] if device.type == "cuda" else None)
        self.raw_model = self.model.module if isinstance(self.model, DDP) else self.model

        self.train_loader = self._configure_loader(train_loader, shuffle=True)
        self.valid_loader = (
            self._configure_loader(valid_loader, shuffle=False) if valid_loader else None
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16")
        self.autocast_dtype = _autocast_dtype(config.precision)
        self.autocast_enabled = self.autocast_dtype is not None and device.type == "cuda"

        self.global_step = 0
        self.configure_optimizers()
        self._setup_wandb()

    def _configure_loader(self, loader: DataLoader, shuffle: bool) -> DataLoader:
        if not is_dist():
            return loader
        dataset = loader.dataset
        if isinstance(dataset, Dataset):
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=shuffle,
            )
            return DataLoader(
                dataset=dataset,
                batch_size=loader.batch_size,
                num_workers=loader.num_workers,
                pin_memory=loader.pin_memory,
                collate_fn=loader.collate_fn,
                sampler=sampler,
            )
        if isinstance(dataset, IterableDataset):
            return loader
        raise ValueError("Unsupported dataset type for distributed loader")

    def configure_optimizers(self) -> None:
        optimizer_cls = OPTIMIZERS[self.config.optimizer]
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.config.lr)

        scheduler_cls = SCHEDULERS[self.config.scheduler]
        warmup_steps = self.config.num_warmup_steps
        if isinstance(warmup_steps, float):
            warmup_steps = int(self.config.max_steps * warmup_steps)
        if warmup_steps <= 0:
            warmup_steps = 0
        self.scheduler = scheduler_cls(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.config.max_steps,
        )

    def _setup_wandb(self) -> None:
        self.wandb = None
        self.wandb_run = None
        if self.config.log_to != "wandb":
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("Install wandb to enable logging.") from exc
        run_name = f"{self.config.wandb_name}_r{self.rank}"
        group_name = self.config.wandb_name
        self.wandb = wandb
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            group=group_name,
            reinit=True,
        )

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(metrics, step=self.global_step)

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if key == "indices":
                continue
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _decode_texts(self, labels: torch.Tensor, label_lens: torch.Tensor) -> list[str]:
        tokenizer = self.raw_model.get_tokenizer()
        texts = []
        for i in range(labels.size(0)):
            ids = labels[i, : label_lens[i]].tolist()
            texts.append(tokenizer.decode(ids))
        return texts

    def _predict_texts(self, features: torch.Tensor) -> list[str]:
        output = self.raw_model.decode_features(features)
        if isinstance(output.labels, list):
            return output.labels
        return [output.labels]

    def _print_predictions(self, preds: list[str], targets: list[str]) -> None:
        if self.rank != 0:
            return
        table = Table(title="Sample Predictions")
        table.add_column("Pred")
        table.add_column("Target")
        for pred, tgt in zip(preds, targets):
            table.add_row(pred, tgt)
        self.console.print(table)

    def _show_startup(self) -> None:
        if self.rank != 0:
            return
        total, trainable = _count_parameters(self.raw_model)
        table = Table(title="Training Summary")
        table.add_column("Item")
        table.add_column("Value")
        table.add_row("Model", self.raw_model.__class__.__name__)
        table.add_row("Total Params", f"{total:,}")
        table.add_row("Trainable Params", f"{trainable:,}")
        table.add_row("Precision", self.config.precision)
        table.add_row("Accumulation", str(self.config.accumulate_grad_batches))
        table.add_row("DDP", "on" if is_dist() else "off")
        self.console.print(table)

    def _validate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        if not self.valid_loader:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_wer = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_loader):
                batch = self._move_batch(batch)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=self.autocast_enabled,
                ):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                total_loss += loss.item()

                preds = self._predict_texts(batch["audio_features"])
                targets = self._decode_texts(batch["labels"], batch["label_lens"])
                total_wer += wer(targets, preds)
                num_batches += 1

                if max_batches and num_batches >= max_batches:
                    break
        self.model.train()
        if is_dist():
            stats = torch.tensor(
                [total_loss, total_wer, num_batches],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total_wer, num_batches = stats.tolist()
            num_batches = int(num_batches)

        if num_batches == 0:
            return {}
        metrics = {
            "val_loss": total_loss / num_batches,
            "val_wer": total_wer / num_batches,
        }
        if is_dist():
            stats = torch.tensor(
                [total_loss, total_wer, num_batches],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_loss, global_wer, global_batches = stats.tolist()
            global_batches = int(global_batches)
            if global_batches:
                metrics["val_loss_global"] = global_loss / global_batches
                metrics["val_wer_global"] = global_wer / global_batches
        return metrics

    def train(self) -> None:
        self._show_startup()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if self.valid_loader and self.config.num_sanity_val_steps:
            metrics = self._validate(max_batches=self.config.num_sanity_val_steps)
            if self.rank == 0 and metrics:
                self.console.print(
                    f"sanity_val_loss={metrics['val_loss']:.4f} sanity_val_wer={metrics['val_wer']:.4f}"
                )
            if metrics:
                self._log_metrics(
                    {
                        f"val/sanity_loss_rank_{self.rank}": metrics["val_loss"],
                        f"val/sanity_wer_rank_{self.rank}": metrics["val_wer"],
                    }
                )

        progress = None
        if self.rank == 0:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )
            task_id = progress.add_task("Training", total=self.config.max_steps)
            progress.start()
        else:
            task_id = None

        acc_steps = self.config.accumulate_grad_batches
        try:
            steps_per_epoch = len(self.train_loader)
        except TypeError:
            steps_per_epoch = self.config.max_steps

        for epoch in range(int(math.ceil(self.config.max_steps / steps_per_epoch))):
            if is_dist():
                sampler = self.train_loader.sampler
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)
            pending_steps = 0
            for batch_idx, batch in enumerate(self.train_loader):
                if self.global_step >= self.config.max_steps:
                    break
                batch_indices = batch.get("indices")
                batch = self._move_batch(batch)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=self.autocast_enabled,
                ):
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / acc_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                pending_steps = (batch_idx + 1) % acc_steps
                if pending_steps == 0:
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1

                    if self.rank == 0 and progress is not None:
                        progress.update(task_id, advance=1)

                    if (
                        self.config.log_every_num_steps
                        and self.global_step % self.config.log_every_num_steps == 0
                    ):
                        if self.rank == 0:
                            self.console.print(
                                f"step={self.global_step} loss={loss.item() * acc_steps:.4f}"
                            )
                        self._log_metrics(
                            {
                                f"train/loss_rank_{self.rank}": loss.item() * acc_steps,
                                f"train/lr_rank_{self.rank}": self.optimizer.param_groups[0]["lr"],
                            }
                        )

                    if (
                        self.config.log_indices
                        and batch_indices is not None
                        and self.global_step % self.config.log_indices_every_num_steps == 0
                    ):
                        indices = batch_indices[: self.config.max_log_indices]
                        self._log_metrics({f"train/indices_rank_{self.rank}": indices})

                    if (
                        self.config.print_predictions
                        and self.global_step % self.config.print_predictions_every_num_steps == 0
                        and self.rank == 0
                    ):
                        preds = self._predict_texts(batch["audio_features"])
                        targets = self._decode_texts(batch["labels"], batch["label_lens"])
                        self._print_predictions(
                            preds[: self.config.max_print_predictions],
                            targets[: self.config.max_print_predictions],
                        )

                    if (
                        self.valid_loader
                        and self.config.check_val_every_num_steps
                        and self.global_step % self.config.check_val_every_num_steps == 0
                    ):
                        metrics = self._validate()
                        if metrics:
                            self._log_metrics(
                                {
                                    f"val/loss_rank_{self.rank}": metrics["val_loss"],
                                    f"val/wer_rank_{self.rank}": metrics["val_wer"],
                                }
                            )
                        if self.rank == 0 and metrics:
                            loss_key = "val_loss_global" if "val_loss_global" in metrics else "val_loss"
                            wer_key = "val_wer_global" if "val_wer_global" in metrics else "val_wer"
                            self.console.print(
                                f"{loss_key}={metrics[loss_key]:.4f} {wer_key}={metrics[wer_key]:.4f}"
                            )
                            if loss_key != "val_loss" or wer_key != "val_wer":
                                self._log_metrics(
                                    {
                                        "val/loss_global": metrics.get("val_loss_global", metrics["val_loss"]),
                                        "val/wer_global": metrics.get("val_wer_global", metrics["val_wer"]),
                                    }
                                )

            if pending_steps != 0 and self.global_step < self.config.max_steps:
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
                if self.rank == 0 and progress is not None:
                    progress.update(task_id, advance=1)

            if self.global_step >= self.config.max_steps:
                break

        if progress is not None:
            progress.stop()
