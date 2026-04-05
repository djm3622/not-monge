"""Unified baseline training entrypoint for OT solvers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import train_baseline_run


@hydra.main(version_base=None, config_path="../configs", config_name="ot")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    result = train_baseline_run(config, output_root=ROOT / str(config["experiment"]["output_dir"]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
