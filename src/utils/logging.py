"""Experiment logging backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class ExperimentLogger(ABC):
    """Abstract logging backend used by training and evaluation loops."""

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Persist scalar metrics."""

    @abstractmethod
    def log_artifact(self, name: str, value: Any, step: int) -> None:
        """Persist an artifact payload."""

    @abstractmethod
    def finalize(self) -> None:
        """Close logger resources."""


class ConsoleLogger(ExperimentLogger):
    """Minimal JSONL logger for local experiments."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        payload = {"step": step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def log_artifact(self, name: str, value: Any, step: int) -> None:
        artifact_dir = self.output_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{step:08d}_{name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, default=str)

    def finalize(self) -> None:
        return None


class WandbLogger(ExperimentLogger):
    """Weights & Biases logger with optional offline mode."""

    def __init__(
        self,
        project: str,
        name: str,
        output_dir: str | Path,
        config: Mapping[str, Any],
        mode: str = "online",
    ) -> None:
        import wandb

        self.run = wandb.init(
            project=project,
            name=name,
            dir=str(output_dir),
            config=dict(config),
            mode=mode,
        )

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self.run.log(dict(metrics), step=step)

    def log_artifact(self, name: str, value: Any, step: int) -> None:
        self.run.log({name: value}, step=step)

    def finalize(self) -> None:
        self.run.finish()


def build_logger(
    backend: str,
    output_dir: str | Path,
    run_name: str,
    project: str,
    config: Mapping[str, Any],
    wandb_mode: str = "online",
) -> ExperimentLogger:
    """Construct the requested logging backend."""
    if backend == "wandb":
        return WandbLogger(
            project=project,
            name=run_name,
            output_dir=output_dir,
            config=config,
            mode=wandb_mode,
        )
    return ConsoleLogger(output_dir=output_dir)
