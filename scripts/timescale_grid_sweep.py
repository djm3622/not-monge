"""Run K and dual/primal timescale sweeps for direct OT formulations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_case1_formulation_suite import (  # noqa: E402
    DEFAULT_CACHE_VERSION,
    _build_run_config,
    _resolve_device,
    _solver_specs_for_dataset,
)
from src.benchmarking import train_baseline_run  # noqa: E402


DEFAULT_SEEDS = list(range(10))
DEFAULT_SOLVERS = ["otp", "monge_map", "otm", "maxcorr"]
DEFAULT_K_VALUES = [1, 2, 5, 10, 20]
DEFAULT_RATIO_VALUES = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
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


def _run_max_steps(base_max_steps: int, k_value: int, budget_mode: str) -> int:
    if budget_mode == "outer":
        return base_max_steps
    return max(1, int(math.ceil(base_max_steps / max(k_value, 1))))


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
    potential_lr = ratio_value * k_value * transport_lr / max(potential_steps, 1)
    spec = copy.deepcopy(base_spec)
    spec["max_steps"] = int(max_steps)
    if batch_size is not None:
        spec["batch_size"] = int(batch_size)
    if steps_per_epoch is not None:
        spec["steps_per_epoch"] = int(steps_per_epoch)

    grid_overrides = [
        f"solver.transport_steps={int(k_value)}",
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
        overrides=grid_overrides + extra_overrides,
    )
    config["experiment"]["name"] = (
        f"timescale_{dataset_name}_{solver_name}_k{k_value}_r{_slug_float(ratio_value)}_seed{seed}"
    )
    config["visualization"]["enabled"] = bool(visualize)
    config["training"]["fairness"]["batch_size"] = int(config["dataset"]["batch_size"])
    config["training"]["fairness"]["max_steps"] = int(config["training"]["max_steps"])
    if not save_epoch_checkpoints:
        config["training"]["checkpointing"]["save_every_n_epochs"] = 0
    return config


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
    effective_ratio = potential_steps * float(metrics.get("configured_potential_lr", 0.0))
    effective_ratio = effective_ratio / max(k_value * transport_lr, 1.0e-12)
    row: dict[str, Any] = {
        "solver_id": result["solver_id"],
        "dataset_id": result["dataset_id"],
        "seed": result["seed"],
        "k": int(k_value),
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
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

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
        for k_value in k_values:
            for ratio_value in ratio_values:
                run_max_steps = _run_max_steps(int(args.max_steps), k_value, str(args.budget_mode))
                for seed in seeds:
                    run_dir = (
                        output_root
                        / solver_name
                        / f"k{k_value}_r{_slug_float(ratio_value)}"
                        / f"seed{seed}"
                    )
                    result_path = run_dir / "results.json"
                    existing = None if args.rerun else _load_existing(result_path)
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
                        result = train_baseline_run(config, output_root=run_dir)
                    else:
                        result = existing

                    metrics = result.setdefault("metrics", {})
                    metrics["configured_transport_lr"] = float(args.transport_lr)
                    metrics["configured_potential_lr"] = (
                        float(ratio_value)
                        * int(k_value)
                        * float(args.transport_lr)
                        / max(int(args.potential_steps), 1)
                    )
                    metrics["configured_potential_steps"] = int(args.potential_steps)
                    metrics["configured_k"] = int(k_value)
                    metrics["configured_ratio"] = float(ratio_value)
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
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
