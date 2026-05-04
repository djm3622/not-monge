"""Run K and dual/primal timescale sweeps for direct OT formulations."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "notmonge_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_case1_formulation_suite import (  # noqa: E402
    DEFAULT_CACHE_VERSION,
    _build_run_config,
    _resolve_device,
    _solver_specs_for_dataset,
)

DEFAULT_SEEDS = list(range(10))
DEFAULT_SOLVERS = ["otp", "monge_map", "otm", "maxcorr"]
DEFAULT_K_VALUES = [1, 2, 5, 10, 20]
DEFAULT_RATIO_VALUES = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
FIXED_TRANSPORT_STEP_SOLVERS: set[str] = set()
FIXED_TRANSPORT_STEPS = 1
SUMMARY_METRICS = (
    "map_l2",
    "pushforward_w2",
    "d_kr",
    "mmd",
    "transport_cost",
    "optimal_transport_cost",
    "transport_cost_gap",
    "theorem_cost_equivalent",
    "theorem_gap",
    "dot_reward",
    "scaled_dot_reward",
    "transport_cos",
)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_plain_config(config: dict[str, Any]) -> dict[str, Any]:
    plain = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    assert isinstance(plain, dict)
    return plain


def _purge_process_caches() -> None:
    """Release Python objects and backend allocator caches after a run."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        # Cache cleanup is best-effort and must not hide the actual run result.
        return


def _run_training_in_process(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run one grid component in the current process."""
    from src.benchmarking import train_baseline_run

    try:
        return train_baseline_run(config, output_root=run_dir)
    finally:
        _purge_process_caches()


def _run_training_subprocess(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run one grid component in a short-lived child process.

    Exiting the child releases model weights, datasets, autograd graphs, and
    backend allocator arenas that a long-lived sweep process may otherwise keep.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="timescale_run_",
        delete=False,
    ) as handle:
        json.dump(_to_plain_config(config), handle)
        config_path = Path(handle.name)

    try:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-config-json",
                str(config_path),
                "--run-output-dir",
                str(run_dir),
            ],
            cwd=str(ROOT),
            check=True,
        )
        if not result_path.exists():
            raise FileNotFoundError(f"Expected child run to write {result_path}")
        return json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        config_path.unlink(missing_ok=True)
        _purge_process_caches()


def _run_single_config(config_path: Path, output_dir: Path) -> None:
    """Child-process entry point for exactly one training run."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _run_training_in_process(config, output_dir)


def _prune_run_checkpoints(config: dict[str, Any], run_dir: Path) -> None:
    checkpointing = config.get("training", {}).get("checkpointing", {})
    checkpoint_dir = run_dir / str(checkpointing.get("dirpath", "checkpoints"))
    if not checkpoint_dir.exists():
        return
    for checkpoint_path in checkpoint_dir.glob("*.pt"):
        if checkpoint_path.name != "best.pt":
            checkpoint_path.unlink(missing_ok=True)


def _run_max_steps(base_max_steps: int, k_value: int, budget_mode: str) -> int:
    if budget_mode == "outer":
        return base_max_steps
    return max(1, int(math.ceil(base_max_steps / max(k_value, 1))))


def _sweep_k_values(solver_name: str, k_values: list[int]) -> list[int]:
    if solver_name in FIXED_TRANSPORT_STEP_SOLVERS:
        return [FIXED_TRANSPORT_STEPS]
    return k_values


def _effective_transport_steps(solver_name: str, k_value: int) -> int:
    if solver_name in FIXED_TRANSPORT_STEP_SOLVERS:
        return FIXED_TRANSPORT_STEPS
    return int(k_value)


def _effective_potential_lr(
    *,
    ratio_value: float,
    transport_lr: float,
    potential_steps: int,
) -> float:
    return ratio_value * transport_lr / max(potential_steps, 1)


def _build_grid_config(
    *,
    solver_name: str,
    dataset_name: str,
    seed: int,
    output_dir: Path,
    base_spec: dict[str, Any],
    cache_version: str,
    device: str,
    eval_items: int,
    k_value: int,
    ratio_value: float,
    transport_lr: float,
    potential_steps: int,
    max_steps: int,
    batch_size: int | None,
    steps_per_epoch: int | None,
    disable_noise: bool,
    visualize: bool,
    save_epoch_checkpoints: bool,
    extra_overrides: list[str],
) -> dict[str, Any]:
    transport_steps = _effective_transport_steps(solver_name, k_value)
    potential_lr = _effective_potential_lr(
        ratio_value=ratio_value,
        transport_lr=transport_lr,
        potential_steps=potential_steps,
    )
    spec = copy.deepcopy(base_spec)
    spec["max_steps"] = int(max_steps)
    if batch_size is not None:
        spec["batch_size"] = int(batch_size)
    if steps_per_epoch is not None:
        spec["steps_per_epoch"] = int(steps_per_epoch)

    grid_overrides = [
        f"solver.transport_steps={int(transport_steps)}",
        f"solver.inner_steps={int(transport_steps)}",
        f"solver.potential_steps={int(potential_steps)}",
        f"solver.transport_lr={float(transport_lr)}",
        f"solver.potential_lr={float(potential_lr)}",
    ]
    if disable_noise:
        grid_overrides.extend(
            [
                "solver.noise.sigma_start=0.0",
                "solver.noise.sigma_end=0.0",
            ]
        )

    config = _build_run_config(
        solver_name=solver_name,
        dataset_name=dataset_name,
        seed=seed,
        output_dir=output_dir,
        spec=spec,
        cache_version=cache_version,
        device=device,
        eval_items=eval_items,
        visualization_items=512,
        saddle_examples=0,
        diagnostic_checkpoints=1,
        overrides=extra_overrides + grid_overrides,
    )
    config["experiment"]["name"] = (
        f"timescale_{dataset_name}_{solver_name}_k{transport_steps}_r{_slug_float(ratio_value)}_seed{seed}"
    )
    if dataset_name.startswith("synthetic_ot"):
        config["dataset"]["seed"] = int(seed)
    config["visualization"]["enabled"] = bool(visualize)
    config["training"]["fairness"]["batch_size"] = int(config["dataset"]["batch_size"])
    config["training"]["fairness"]["max_steps"] = int(config["training"]["max_steps"])
    if not save_epoch_checkpoints:
        config["training"]["checkpointing"]["save_every_n_epochs"] = 0
    return _to_plain_config(config)


def _flatten_result(
    result: dict[str, Any],
    *,
    run_dir: Path,
    k_value: int,
    ratio_value: float,
    transport_lr: float,
    potential_steps: int,
) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    transport_steps = _effective_transport_steps(str(result["solver_id"]), k_value)
    effective_ratio = potential_steps * float(metrics.get("configured_potential_lr", 0.0))
    effective_ratio = effective_ratio / max(transport_lr, 1.0e-12)
    row: dict[str, Any] = {
        "solver_id": result["solver_id"],
        "dataset_id": result["dataset_id"],
        "seed": result["seed"],
        "k": int(k_value),
        "transport_steps": int(transport_steps),
        "ratio": float(ratio_value),
        "transport_lr": float(transport_lr),
        "potential_steps": int(potential_steps),
        "effective_ratio": effective_ratio,
        "max_steps": result["max_steps"],
        "run_dir": str(run_dir),
    }
    for key, value in metrics.items():
        if isinstance(value, (int, float)) or value is None:
            row[key] = value
    return row


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["solver_id"]), int(row["k"]), float(row["ratio"]))
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (solver_id, k_value, ratio_value), group in sorted(grouped.items()):
        output: dict[str, Any] = {
            "solver_id": solver_id,
            "k": k_value,
            "ratio": ratio_value,
            "num_seeds": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            if not values:
                continue
            metric_mean, metric_std = _mean_std(values)
            output[f"{metric}_mean"] = metric_mean
            output[f"{metric}_std"] = metric_std
        aggregate_rows.append(output)
    return aggregate_rows


def _write_summary(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Timescale Grid Summary",
        "",
        "| solver | K | ratio | map_l2 | d_KR | MMD | theorem_gap | dot_reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["solver_id"]),
                    str(row["k"]),
                    f"{float(row['ratio']):g}",
                    _format_summary(row, "map_l2"),
                    _format_summary(row, "d_kr"),
                    _format_summary(row, "mmd"),
                    _format_summary(row, "theorem_gap"),
                    _format_summary(row, "dot_reward"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_summary(row: dict[str, Any], metric: str) -> str:
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    if mean_key not in row:
        return "---"
    return f"{float(row[mean_key]):.4g} +/- {float(row.get(std_key, 0.0)):.2g}"


def _load_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _close_float(left: Any, right: float, *, atol: float = 1.0e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _existing_matches_grid(
    result: dict[str, Any],
    *,
    solver_name: str,
    max_steps: int,
    transport_steps: int,
    potential_steps: int,
    transport_lr: float,
    potential_lr: float,
    ratio_value: float,
) -> bool:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return False
    if str(result.get("solver_id")) != solver_name or int(result.get("max_steps", -1)) != int(max_steps):
        return False

    runtime_keys = (
        "solver_transport_steps",
        "solver_potential_steps",
        "solver_transport_lr",
        "solver_potential_lr",
    )
    has_runtime_solver_config = all(key in metrics for key in runtime_keys)
    if has_runtime_solver_config:
        actual_solver_config_matches = (
            int(metrics.get("solver_transport_steps", -1)) == int(transport_steps)
            and int(metrics.get("solver_potential_steps", -1)) == int(potential_steps)
            and _close_float(metrics.get("solver_transport_lr"), transport_lr)
            and _close_float(metrics.get("solver_potential_lr"), potential_lr)
        )
    else:
        # Older OTP sweeps did not persist runtime solver_* fields. Those runs are still
        # reusable because OTP's configured K is the actual transport step count.
        actual_solver_config_matches = solver_name not in FIXED_TRANSPORT_STEP_SOLVERS

    configured_transport_steps = metrics.get("configured_transport_steps", metrics.get("configured_k"))
    configured_potential_steps = metrics.get(
        "configured_potential_steps",
        metrics.get("solver_potential_steps", -1),
    )
    return (
        actual_solver_config_matches
        and int(configured_transport_steps) == int(transport_steps)
        and int(metrics.get("configured_k", -1)) == int(transport_steps)
        and int(configured_potential_steps) == int(potential_steps)
        and _close_float(metrics.get("configured_transport_lr"), transport_lr)
        and _close_float(metrics.get("configured_potential_lr"), potential_lr)
        and _close_float(metrics.get("configured_ratio"), ratio_value)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K and dual/primal ratio grid sweeps.")
    parser.add_argument("--dataset", default="synthetic_ot")
    parser.add_argument("--solvers", default=",".join(DEFAULT_SOLVERS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--k-values", default=",".join(str(value) for value in DEFAULT_K_VALUES))
    parser.add_argument("--ratio-values", default=",".join(str(value) for value in DEFAULT_RATIO_VALUES))
    parser.add_argument("--transport-lr", type=float, default=5.0e-4)
    parser.add_argument("--potential-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--budget-mode", choices=["outer", "transport"], default="outer")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--eval-items", type=int, default=4096)
    parser.add_argument("--cache-version", default=DEFAULT_CACHE_VERSION)
    parser.add_argument("--output-root", default="outputs/timescale_grid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-config-noise", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--no-isolate-runs", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--run-config-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-output-dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_config_json is not None:
        if args.run_output_dir is None:
            raise ValueError("--run-output-dir is required with --run-config-json")
        _run_single_config(Path(str(args.run_config_json)), Path(str(args.run_output_dir)))
        return

    dataset_name = str(args.dataset)
    solvers = _parse_csv_strings(str(args.solvers))
    seeds = _parse_csv_ints(str(args.seeds))
    k_values = _parse_csv_ints(str(args.k_values))
    ratio_values = _parse_csv_floats(str(args.ratio_values))
    device = _resolve_device(str(args.device)).type
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    base_specs = _solver_specs_for_dataset(dataset_name)
    rows: list[dict[str, Any]] = []
    for solver_name in solvers:
        if solver_name not in base_specs:
            raise ValueError(f"Unknown solver '{solver_name}' for dataset '{dataset_name}'")
        for k_value in _sweep_k_values(solver_name, k_values):
            for ratio_value in ratio_values:
                transport_steps = _effective_transport_steps(solver_name, k_value)
                potential_lr = _effective_potential_lr(
                    ratio_value=ratio_value,
                    transport_lr=float(args.transport_lr),
                    potential_steps=int(args.potential_steps),
                )
                run_max_steps = _run_max_steps(int(args.max_steps), transport_steps, str(args.budget_mode))
                for seed in seeds:
                    run_dir = (
                        output_root
                        / solver_name
                        / f"k{k_value}_r{_slug_float(ratio_value)}"
                        / f"seed{seed}"
                    )
                    result_path = run_dir / "results.json"
                    existing = None if args.rerun else _load_existing(result_path)
                    if existing is not None and not _existing_matches_grid(
                        existing,
                        solver_name=solver_name,
                        max_steps=run_max_steps,
                        transport_steps=transport_steps,
                        potential_steps=int(args.potential_steps),
                        transport_lr=float(args.transport_lr),
                        potential_lr=potential_lr,
                        ratio_value=float(ratio_value),
                    ):
                        existing = None
                    if existing is None:
                        config = _build_grid_config(
                            solver_name=solver_name,
                            dataset_name=dataset_name,
                            seed=seed,
                            output_dir=run_dir,
                            base_spec=base_specs[solver_name],
                            cache_version=str(args.cache_version),
                            device=device,
                            eval_items=int(args.eval_items),
                            k_value=k_value,
                            ratio_value=ratio_value,
                            transport_lr=float(args.transport_lr),
                            potential_steps=int(args.potential_steps),
                            max_steps=run_max_steps,
                            batch_size=int(args.batch_size) if int(args.batch_size) > 0 else None,
                            steps_per_epoch=int(args.steps_per_epoch)
                            if int(args.steps_per_epoch) > 0
                            else None,
                            disable_noise=not bool(args.keep_config_noise),
                            visualize=bool(args.visualize),
                            save_epoch_checkpoints=bool(args.save_epoch_checkpoints),
                            extra_overrides=list(args.override),
                        )
                        if bool(args.no_isolate_runs):
                            result = _run_training_in_process(config, run_dir)
                        else:
                            result = _run_training_subprocess(config, run_dir)
                    else:
                        result = existing

                    metrics = result.setdefault("metrics", {})
                    metrics["configured_transport_steps"] = int(transport_steps)
                    metrics["configured_transport_lr"] = float(args.transport_lr)
                    metrics["configured_potential_lr"] = potential_lr
                    metrics["configured_potential_steps"] = int(args.potential_steps)
                    metrics["configured_k"] = int(transport_steps)
                    metrics["configured_ratio"] = float(ratio_value)
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    if (
                        existing is None
                        and not bool(args.keep_checkpoints)
                        and not bool(args.save_epoch_checkpoints)
                    ):
                        _prune_run_checkpoints(config, run_dir)
                    if existing is None:
                        del config
                        _purge_process_caches()
                    rows.append(
                        _flatten_result(
                            result,
                            run_dir=run_dir,
                            k_value=k_value,
                            ratio_value=ratio_value,
                            transport_lr=float(args.transport_lr),
                            potential_steps=int(args.potential_steps),
                        )
                    )

    aggregate_rows = _aggregate(rows)
    (output_root / "seed_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "aggregate_results.json").write_text(
        json.dumps(aggregate_rows, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_root / "seed_results.csv", rows)
    _write_csv(output_root / "aggregate_results.csv", aggregate_rows)
    _write_summary(output_root / "summary.md", aggregate_rows)

    run_spec = {
        "dataset": dataset_name,
        "solvers": solvers,
        "seeds": seeds,
        "k_values": k_values,
        "fixed_transport_step_solvers": sorted(FIXED_TRANSPORT_STEP_SOLVERS),
        "fixed_transport_steps": FIXED_TRANSPORT_STEPS,
        "ratio_values": ratio_values,
        "transport_lr": float(args.transport_lr),
        "potential_steps": int(args.potential_steps),
        "max_steps": int(args.max_steps),
        "budget_mode": str(args.budget_mode),
        "eval_items": int(args.eval_items),
        "device": device,
    }
    (output_root / "run_spec.json").write_text(json.dumps(run_spec, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
