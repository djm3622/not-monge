"""Run the 1-D Gaussian flatness sweep over K, eta_psi / eta_T and seeds.

Writes ``results.npz`` (keys ``K{K}|r{ratio:g}|{metric}``, arrays of shape
seeds x logged steps) and ``manifest.json`` (resolved config) to the sweep
output directory. Plots are made from these files by
``scripts/plot_flatness_1d.py`` without retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics.flatness_1d import run_flatness_1d  # noqa: E402


def _parse_list(value: str, cast: type) -> list:
    return [cast(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/flatness_1d.yaml")
    parser.add_argument("--inner-steps", default=None, help="Optional comma-separated K override.")
    parser.add_argument("--lr-ratios", default=None, help="Optional comma-separated eta_psi/eta_T override.")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed override.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    config = OmegaConf.to_container(OmegaConf.load(ROOT / args.config), resolve=True)
    sweep = config["sweep"]
    if args.inner_steps is not None:
        sweep["inner_steps"] = _parse_list(args.inner_steps, int)
    if args.lr_ratios is not None:
        sweep["lr_ratios"] = _parse_list(args.lr_ratios, float)
    if args.seeds is not None:
        sweep["seeds"] = _parse_list(args.seeds, int)
    if args.output_dir is not None:
        sweep["output_dir"] = args.output_dir
    output_dir = ROOT / sweep["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for inner_steps in sweep["inner_steps"]:
        for lr_ratio in sweep["lr_ratios"]:
            logs = []
            for seed in sweep["seeds"]:
                log = run_flatness_1d(config, inner_steps=inner_steps, lr_ratio=lr_ratio, seed=seed)
                logs.append(log)
                print(
                    f"K={inner_steps} ratio={lr_ratio:g} seed={seed} "
                    f"map_err={log['map_err'][-1]:.2e} grad_err={log['grad_err'][-1]:.2e} "
                    f"F={log['F'][-1]:.3f}",
                    flush=True,
                )
            for key in logs[0]:
                arrays[f"K{inner_steps}|r{lr_ratio:g}|{key}"] = np.array([log[key] for log in logs])

    np.savez(output_dir / "results.npz", **arrays)
    (output_dir / "manifest.json").write_text(json.dumps({"config": config}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "series": len(arrays)}, indent=2))


if __name__ == "__main__":
    main()
