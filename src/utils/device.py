"""Device, precision, and compilation helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class PrecisionConfig:
    """Precision policy used by the trainer."""

    mode: str = "fp32"

    @property
    def amp_enabled(self) -> bool:
        return self.mode in {"fp16", "bf16"}

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        if self.mode == "fp16":
            return torch.float16
        if self.mode == "bf16":
            return torch.bfloat16
        return None


def infer_device(requested: str = "auto") -> torch.device:
    """Select a training device with sensible fallbacks."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_autocast_context(
    device: torch.device,
    precision: PrecisionConfig,
) -> Iterator[None]:
    """Build an autocast context manager for the current backend."""
    if not precision.amp_enabled or precision.autocast_dtype is None:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=precision.autocast_dtype)
    if device.type == "cpu" and precision.mode == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def make_grad_scaler(device: torch.device, precision: PrecisionConfig) -> torch.amp.GradScaler:
    """Create a scaler only when CUDA fp16 is active."""
    enabled = device.type == "cuda" and precision.mode == "fp16"
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.GradScaler(device=scaler_device, enabled=enabled)


def maybe_compile_module(
    module: torch.nn.Module,
    enabled: bool,
    dynamic: bool = False,
    fullgraph: bool = False,
) -> torch.nn.Module:
    """Compile a module when supported and requested."""
    if not enabled or not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(module, dynamic=dynamic, fullgraph=fullgraph)
    except Exception:
        return module
