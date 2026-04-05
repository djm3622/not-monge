"""Run smoke tests and short solver validation jobs."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import train_baseline_run


def build_config(solver_name: str, output_dir: str) -> dict:
    model_cfg = {
        "input_dim": 2,
        "output_dim": 2,
        "hidden_dims": [16, 16],
        "activation": "silu",
        "dropout": 0.0,
        "residual": True,
        "layer_norm": False,
    }
    solver_cfgs = {
        "minimax": {
            "name": "minimax",
            "critic_steps": 1,
            "map_lr": 1.0e-3,
            "potential_lr": 1.0e-3,
            "potential": {"kind": "mlp", "hidden_dims": [16, 16], "activation": "silu", "layer_norm": False},
        },
        "icnn": {
            "name": "icnn",
            "critic_steps": 1,
            "map_lr": 1.0e-3,
            "potential_lr": 1.0e-3,
            "potential": {"hidden_dims": [16, 16], "activation": "softplus", "strong_convexity": 0.1},
        },
        "otp": {
            "name": "otp",
            "critic_steps": 1,
            "map_lr": 1.0e-3,
            "potential_lr": 1.0e-3,
            "potential": {"kind": "mlp", "hidden_dims": [16, 16], "activation": "silu", "layer_norm": False},
            "smoothing": {"sigma_start": 0.05, "sigma_end": 0.0, "anneal_steps": 32},
            "plan": {"enabled": True, "reg": 1.0, "supervision_weight": 0.5, "entropy_weight": 0.0},
            "regularization": {"potential_gp_weight": 1.0, "potential_l2_weight": 1.0e-3},
        },
        "flow": {
            "name": "flow",
            "velocity": {"hidden_dims": [16, 16], "activation": "silu", "layer_norm": False},
            "lr": 1.0e-3,
            "integration": {"backend": "rk4", "method": "rk4", "steps": 6, "atol": 1.0e-5, "rtol": 1.0e-5, "use_adjoint": False},
            "plan": {"reg": 1.0},
            "loss": {"endpoint_weight": 1.0, "energy_weight": 0.1, "mmd_weight": 0.05},
        },
        "sinkhorn": {"name": "sinkhorn", "reg": 1.0, "fit_samples": 32, "kernel_bandwidth": 1.0},
    }
    return {
        "model": model_cfg,
        "solver": copy.deepcopy(solver_cfgs[solver_name]),
        "dataset": {
            "name": "synthetic_ot",
            "input_dim": 2,
            "n_train": 64,
            "n_val": 16,
            "n_test": 16,
            "batch_size": 8,
            "num_workers": 0,
            "source_distribution": "gaussian",
            "source_scale": 1.0,
            "mixture_components": 2,
            "generator_hidden_dims": [16, 16],
            "strong_convexity": 0.1,
            "seed": 7,
        },
        "training": {
            "seed": 123,
            "max_epochs": 4,
            "max_steps": 10,
            "log_every_n_steps": 1,
            "validate_every_n_epochs": 1,
            "gradient_clip_norm": 1.0,
            "precision": "fp32",
            "compile": False,
            "deterministic": True,
            "device": "cpu",
            "optimizer": {
                "lr": 1.0e-3,
                "weight_decay": 1.0e-4,
                "pct_start": 0.3,
                "div_factor": 25.0,
                "final_div_factor": 1000.0,
            },
            "fairness": {"batch_size": 8, "max_steps": 10, "optimizer": "adamw", "scheduler": "onecycle"},
            "logging": {"backend": "console", "project": "smoke", "wandb_mode": "offline"},
            "checkpointing": {"dirpath": "checkpoints", "save_every_n_epochs": 1, "monitor": "val/map_l2", "mode": "min"},
        },
        "experiment": {"id": "ot_recovery", "name": f"smoke_{solver_name}", "output_dir": output_dir},
        "evaluation": {"checkpoint_path": None, "output_dir": output_dir, "max_items": 64},
    }


def run_pytest_smoke() -> tuple[int, str]:
    command = [sys.executable, "-m", "pytest", "-v", "tests/smoke/"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    summary_text = completed.stdout + completed.stderr
    summary_match = re.search(r"=+ (.+?) in [0-9.]+s", summary_text)
    summary = summary_match.group(1) if summary_match is not None else "summary unavailable"
    print("Pytest smoke summary:", summary)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    return completed.returncode, summary


def run_solver_smoke_jobs() -> tuple[bool, list[dict[str, object]]]:
    results = []
    all_ok = True
    output_root = ROOT / "outputs" / "smoke_validation"
    output_root.mkdir(parents=True, exist_ok=True)
    for solver_name in ["minimax", "icnn", "otp", "flow", "sinkhorn"]:
        config = build_config(solver_name, output_dir=f"outputs/smoke_validation/{solver_name}")
        result = train_baseline_run(config, output_root=output_root / solver_name)
        metrics = result["metrics"]
        finite_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value == value and value not in {float("inf"), float("-inf")}
        }
        okay = len(finite_metrics) == len([value for value in metrics.values() if isinstance(value, (int, float))])
        all_ok = all_ok and okay
        summary = {
            "solver": solver_name,
            "status": "ok" if okay else "failed",
            "metrics": metrics,
        }
        results.append(summary)
        print(json.dumps(summary, indent=2))
    return all_ok, results


def main() -> None:
    start_time = time.perf_counter()
    pytest_code, summary = run_pytest_smoke()
    if pytest_code != 0:
        raise SystemExit(pytest_code)
    solvers_ok, _ = run_solver_smoke_jobs()
    total_runtime = time.perf_counter() - start_time
    print(json.dumps({"pytest": summary, "solvers_ok": solvers_ok, "runtime_seconds": total_runtime}, indent=2))
    if not solvers_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
