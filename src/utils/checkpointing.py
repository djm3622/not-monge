"""Checkpoint persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def save_checkpoint(state: Mapping[str, Any], path: str | Path) -> None:
    """Save a checkpoint atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state), tmp_path)
    tmp_path.replace(path)
