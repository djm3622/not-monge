"""Custom trainer shared by OT and diffusion experiments."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol

import torch
from torch.utils.data import DataLoader

from src.utils import (
    PrecisionConfig,
    build_logger,
    infer_device,
    make_autocast_context,
    make_grad_scaler,
    save_checkpoint,
    seed_all,
)


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors to a device."""
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(value, device) for value in batch)
    return batch


def mean_metrics(metrics_list: list[Mapping[str, float]]) -> dict[str, float]:
    """Average a list of metric dictionaries."""
    if not metrics_list:
        return {}
    keys = set().union(*(metrics.keys() for metrics in metrics_list))
    return {
        key: sum(float(metrics.get(key, 0.0)) for metrics in metrics_list) / len(metrics_list)
        for key in keys
    }


class TrainTask(Protocol):
    """Protocol implemented by trainable tasks."""

    def to(self, device: torch.device) -> "TrainTask":
        ...

    def train(self, mode: bool = True) -> "TrainTask":
        ...

    def eval(self) -> "TrainTask":
        ...

    def configure_optimizers(self, total_steps: int) -> None:
        ...

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        ...

    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        ...

    def state_dict(self) -> MutableMapping[str, Any]:
        ...

    def compile_modules(self) -> None:
        ...


@dataclass
class TrainerConfig:
    """Configuration for the custom trainer."""

    seed: int
    max_epochs: int
    max_steps: int | None
    log_every_n_steps: int
    validate_every_n_epochs: int
    gradient_clip_norm: float | None
    precision: str
    compile: bool
    deterministic: bool
    device: str
    logging: Mapping[str, Any]
    checkpointing: Mapping[str, Any]


class Trainer:
    """Simple trainer with AMP, checkpointing, and metric logging."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_name: str,
        output_dir: str | Path,
        full_config: Mapping[str, Any],
    ) -> None:
        self.config = TrainerConfig(
            seed=int(config["seed"]),
            max_epochs=int(config["max_epochs"]),
            max_steps=int(config["max_steps"]) if config.get("max_steps") is not None else None,
            log_every_n_steps=int(config["log_every_n_steps"]),
            validate_every_n_epochs=int(config["validate_every_n_epochs"]),
            gradient_clip_norm=(
                float(config["gradient_clip_norm"])
                if config.get("gradient_clip_norm") is not None
                else None
            ),
            precision=str(config["precision"]),
            compile=bool(config["compile"]),
            deterministic=bool(config["deterministic"]),
            device=str(config["device"]),
            logging=dict(config["logging"]),
            checkpointing=dict(config["checkpointing"]),
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = infer_device(self.config.device)
        self.precision = PrecisionConfig(self.config.precision)
        self.scaler = make_grad_scaler(self.device, self.precision)
        self.logger = build_logger(
            backend=str(self.config.logging["backend"]),
            output_dir=self.output_dir,
            run_name=experiment_name,
            project=str(self.config.logging["project"]),
            config=full_config,
            wandb_mode=str(self.config.logging.get("wandb_mode", "online")),
        )
        self.best_metric: float | None = None
        self.global_step = 0

    def _save_epoch_checkpoint(
        self,
        task: TrainTask,
        epoch: int,
        val_metrics: Mapping[str, float] | None,
    ) -> None:
        checkpoint_dir = self.output_dir / str(self.config.checkpointing["dirpath"])
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "task": task.state_dict(),
            "val_metrics": dict(val_metrics or {}),
        }
        save_every_n_epochs = int(self.config.checkpointing.get("save_every_n_epochs", 1) or 0)
        if save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(state, checkpoint_path)
        monitor = str(self.config.checkpointing.get("monitor", ""))
        if val_metrics is None or monitor not in val_metrics:
            return
        mode = str(self.config.checkpointing.get("mode", "min")).lower()
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported checkpoint mode '{mode}'")
        current = float(val_metrics[monitor])
        is_better = (
            self.best_metric is None
            or (mode == "min" and current < self.best_metric)
            or (mode == "max" and current > self.best_metric)
        )
        if is_better:
            self.best_metric = current
            save_checkpoint(state, checkpoint_dir / "best.pt")

    def fit(
        self,
        task: TrainTask,
        train_loader: DataLoader[Mapping[str, torch.Tensor]],
        val_loader: DataLoader[Mapping[str, torch.Tensor]] | None = None,
    ) -> dict[str, float]:
        seed_all(self.config.seed, deterministic=self.config.deterministic)
        task.to(self.device)
        if self.config.compile:
            task.compile_modules()
        total_steps = self.config.max_steps or (self.config.max_epochs * len(train_loader))
        task.configure_optimizers(total_steps)
        final_val_metrics: dict[str, float] = {}

        for epoch in range(self.config.max_epochs):
            task.train(True)
            epoch_metrics: list[Mapping[str, float]] = []
            for batch in train_loader:
                batch = move_to_device(batch, self.device)
                metrics = task.training_step(
                    batch=batch,
                    scaler=self.scaler,
                    autocast_context=lambda: make_autocast_context(self.device, self.precision),
                    gradient_clip_norm=self.config.gradient_clip_norm,
                )
                epoch_metrics.append(metrics)
                self.global_step += 1
                if self.global_step % self.config.log_every_n_steps == 0:
                    self.logger.log_metrics(dict(metrics), step=self.global_step)
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    break
            self.logger.log_metrics(mean_metrics(epoch_metrics), step=self.global_step)

            if val_loader is not None and (epoch + 1) % self.config.validate_every_n_epochs == 0:
                task.eval()
                val_metrics = [task.validation_step(move_to_device(batch, self.device)) for batch in val_loader]
                final_val_metrics = mean_metrics(val_metrics)
                self.logger.log_metrics(final_val_metrics, step=self.global_step)
                self._save_epoch_checkpoint(task, epoch + 1, final_val_metrics)
            else:
                self._save_epoch_checkpoint(task, epoch + 1, None)

            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break

        self.logger.finalize()
        return final_val_metrics
