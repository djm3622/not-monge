"""Unified baseline evaluation entrypoint for OT solvers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import eval_baseline_run


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    checkpoint_path = config["evaluation"]["checkpoint_path"]
    if checkpoint_path is None:
        raise ValueError("evaluation.checkpoint_path must be provided")
    result = eval_baseline_run(
        config,
        checkpoint_path=checkpoint_path,
        output_root=ROOT / str(config["evaluation"]["output_dir"]),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
