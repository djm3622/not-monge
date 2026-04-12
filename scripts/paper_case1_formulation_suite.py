"""Run case-study-1 comparisons for the three direct OT formulation families."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import eval_baseline_run, resolve_ot_dataset, train_baseline_run
from src.diagnostics.case1_formulations import potential_seed_dispersion, run_case1_formulation_diagnostic
from src.utils.device import infer_device

DEFAULT_SEEDS = [686499, 928801, 48156]
DEFAULT_SOLVERS = ["otp", "monge_map", "otm", "maxcorr", "makkuva_icnn_cvx"]
DEFAULT_DATASET = "paper_mix3to10"
DEFAULT_CACHE_VERSION = "paper_ref_d64_b256_s25k"
DEFAULT_OUTPUT_ROOTS = {
    "paper_mix3to10": "outputs/paper_case1_formulations",
    "synthetic_ot": "outputs/synthetic_ot_formulations",
}
DEFAULT_VISUALIZATION_ITEMS = 512
DEFAULT_SADDLE_EXAMPLES = 3
DEFAULT_DIAGNOSTIC_CHECKPOINTS = 5
DEFAULT_DIAGNOSTIC_ITEMS = 512
DEFAULT_DIAGNOSTIC_NOISE_SCALE = 1.0e-2

SOLVER_SPECS: dict[str, dict[str, object]] = {
    "otp": {
        "max_steps": 4096,
        "batch_size": 256,
        "steps_per_epoch": 64,
        "extra_overrides": [
            "solver.transport_steps=1",
            "solver.transport_lr=5e-4",
            "solver.potential_lr=5e-4",
            "solver.noise.sigma_start=0.0",
            "solver.noise.sigma_end=0.0",
        ],
    },
    "monge_map": {
        "max_steps": 4096,
        "batch_size": 256,
        "steps_per_epoch": 64,
        "extra_overrides": [],
    },
    "otm": {
        "max_steps": 4096,
        "batch_size": 256,
        "steps_per_epoch": 64,
        "extra_overrides": [],
    },
    "maxcorr": {
        "max_steps": 4096,
        "batch_size": 256,
        "steps_per_epoch": 64,
        "extra_overrides": [
            "solver.transport_steps=1",
            "solver.transport_lr=5e-4",
            "solver.potential_lr=5e-4",
            "solver.transport_l2_weight=0.05",
            "solver.noise.sigma_start=0.0",
            "solver.noise.sigma_end=0.0",
        ],
    },
    "makkuva_icnn_cvx": {
        "max_steps": 4096,
        "batch_size": 256,
        "steps_per_epoch": 64,
        "extra_overrides": [
            "solver.lr=5e-4",
            "solver.inner_steps=4",
        ],
    },
}
SYNTHETIC_SOLVER_SPECS: dict[str, dict[str, object]] = {
    "otp": {
        "max_steps": 4096,
        "batch_size": 512,
        "steps_per_epoch": 128,
        "extra_overrides": [
            "solver.transport_steps=1",
            "solver.transport_lr=5e-4",
            "solver.potential_lr=5e-4",
            "solver.noise.sigma_start=0.0",
            "solver.noise.sigma_end=0.0",
        ],
    },
    "monge_map": {
        "max_steps": 4096,
        "batch_size": 512,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
    "otm": {
        "max_steps": 4096,
        "batch_size": 512,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
    "maxcorr": {
        "max_steps": 4096,
        "batch_size": 512,
        "steps_per_epoch": 128,
        "extra_overrides": [
            "solver.transport_steps=1",
            "solver.transport_lr=5e-4",
            "solver.potential_lr=5e-4",
            "solver.transport_l2_weight=0.05",
            "solver.noise.sigma_start=0.0",
            "solver.noise.sigma_end=0.0",
        ],
    },
    "makkuva_icnn_cvx": {
        "max_steps": 4096,
        "batch_size": 512,
        "steps_per_epoch": 128,
        "extra_overrides": [
            "solver.lr=5e-4",
            "solver.inner_steps=4",
        ],
    },
}
DATASET_SOLVER_SPECS: dict[str, dict[str, dict[str, object]]] = {
    "paper_mix3to10": SOLVER_SPECS,
    "synthetic_ot": SYNTHETIC_SOLVER_SPECS,
}
SUMMARY_METRICS = ("map_l2", "pushforward_w2", "transport_cos")
DIAGNOSTIC_METRICS = (
    "diagnostic_best_target_potential_centered_rmse",
    "diagnostic_best_target_potential_gradient_rmse",
    "diagnostic_best_flatness_std_F",
    "diagnostic_best_potential_centered_rmse",
)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _parse_spec_overrides(value: str) -> dict[str, dict[str, object]]:
    if not value.strip():
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("--spec-overrides must decode to a JSON object.")
    overrides: dict[str, dict[str, object]] = {}
    for solver_name, solver_override in loaded.items():
        if solver_name not in SOLVER_SPECS:
            raise ValueError(f"Unknown solver in --spec-overrides: '{solver_name}'")
        if not isinstance(solver_override, dict):
            raise ValueError(f"Override for solver '{solver_name}' must be a JSON object.")
        overrides[solver_name] = dict(solver_override)
    return overrides


def _solver_specs_for_dataset(dataset_name: str) -> dict[str, dict[str, object]]:
    if dataset_name not in DATASET_SOLVER_SPECS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
    return DATASET_SOLVER_SPECS[dataset_name]


def _merge_solver_specs(
    base_specs: dict[str, dict[str, object]],
    spec_overrides: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    merged = copy.deepcopy(base_specs)
    for solver_name, solver_override in spec_overrides.items():
        merged.setdefault(solver_name, {}).update(solver_override)
    return merged


def _scale_solver_specs(
    solver_specs: dict[str, dict[str, object]],
    *,
    budget_scale: float,
) -> dict[str, dict[str, object]]:
    if budget_scale <= 0.0:
        raise ValueError("--budget-scale must be positive.")
    scaled = copy.deepcopy(solver_specs)
    if math.isclose(budget_scale, 1.0):
        return scaled
    for spec in scaled.values():
        max_steps = int(spec.get("max_steps", 0))
        if max_steps <= 1:
            continue
        spec["max_steps"] = max(1, int(math.ceil(max_steps * budget_scale)))
    return scaled


def _resolve_output_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _default_output_root(dataset_name: str) -> str:
    if dataset_name not in DEFAULT_OUTPUT_ROOTS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
    return DEFAULT_OUTPUT_ROOTS[dataset_name]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("Requested device 'mps' but torch.backends.mps.is_available() is False.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device 'cuda' but torch.cuda.is_available() is False.")
    return device


def _diagnostic_checkpoint_interval(
    *,
    max_steps: int,
    steps_per_epoch: int,
    num_checkpoints: int,
) -> int:
    if num_checkpoints <= 0:
        return 0
    epochs = max(1, math.ceil(max_steps / max(steps_per_epoch, 1)))
    if epochs <= num_checkpoints:
        return 1
    return max(1, math.floor(epochs / num_checkpoints))


def _apply_override(config: dict[str, Any], override: str) -> None:
    key, raw_value = override.split("=", 1)
    value = OmegaConf.create(f"value: {raw_value}")["value"]
    node: dict[str, Any] = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _build_run_config(
    *,
    solver_name: str,
    dataset_name: str,
    seed: int,
    output_dir: Path,
    spec: dict[str, Any],
    cache_version: str,
    device: str,
    eval_items: int,
    visualization_items: int,
    saddle_examples: int,
    diagnostic_checkpoints: int,
    overrides: list[str],
) -> dict[str, Any]:
    model = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/model/ot_map.yaml"), resolve=True)
    solver = OmegaConf.to_container(OmegaConf.load(ROOT / f"configs/solver/{solver_name}.yaml"), resolve=True)
    dataset = OmegaConf.to_container(OmegaConf.load(ROOT / f"configs/dataset/{dataset_name}.yaml"), resolve=True)
    training = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/training/base.yaml"), resolve=True)

    assert isinstance(model, dict)
    assert isinstance(solver, dict)
    assert isinstance(dataset, dict)
    assert isinstance(training, dict)

    config: dict[str, Any] = {
        "model": model,
        "solver": solver,
        "dataset": dataset,
        "training": training,
        "experiment": {
            "id": "ot_recovery",
            "name": f"{dataset_name}_formulations",
            "output_dir": output_dir.as_posix(),
        },
        "visualization": {
            "enabled": True,
            "dirpath": "visualizations",
            "max_items": visualization_items,
            "saddle_examples": saddle_examples,
        },
        "evaluation": {"max_items": eval_items},
    }
    input_dim = int(dataset["input_dim"])
    config["model"]["input_dim"] = input_dim
    config["model"]["output_dim"] = input_dim
    config["dataset"]["batch_size"] = int(spec["batch_size"])
    config["dataset"]["steps_per_epoch"] = int(spec["steps_per_epoch"])
    if dataset_name == "paper_mix3to10":
        config["dataset"]["cache_version"] = cache_version
    config["training"]["seed"] = seed
    config["training"]["device"] = device
    config["training"]["max_steps"] = int(spec["max_steps"])
    config["training"]["max_epochs"] = 1000
    config["training"]["checkpointing"]["save_every_n_epochs"] = _diagnostic_checkpoint_interval(
        max_steps=int(spec["max_steps"]),
        steps_per_epoch=int(spec["steps_per_epoch"]),
        num_checkpoints=diagnostic_checkpoints,
    )
    config["training"]["checkpointing"]["monitor"] = "val/map_l2"
    config["training"]["checkpointing"]["mode"] = "min"

    for override in spec.get("extra_overrides", []):
        _apply_override(config, str(override))
    for override in overrides:
        _apply_override(config, override)
    return config


def _diagnostic_result_is_complete(metrics: dict[str, Any]) -> bool:
    plot_path = metrics.get("diagnostic_plot_path")
    results_path = metrics.get("diagnostic_results_path")
    return (
        metrics.get("formulation_case_name") is not None
        and bool(plot_path)
        and bool(results_path)
        and Path(str(plot_path)).exists()
        and Path(str(results_path)).exists()
    )


def _result_is_complete(result: dict[str, Any], *, saddle_examples: int) -> bool:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return False
    for metric_name in SUMMARY_METRICS:
        if metric_name not in metrics:
            return False
    transport_path = metrics.get("transport_figure_path") or metrics.get("visualization_path")
    saddle_path = metrics.get("saddle_figure_path")
    sample_paths = metrics.get("saddle_sample_paths", [])
    if not transport_path or not Path(str(transport_path)).exists():
        return False
    if not saddle_path or not Path(str(saddle_path)).exists():
        return False
    if not isinstance(sample_paths, list) or len(sample_paths) < saddle_examples:
        return False
    if not all(Path(str(path)).exists() for path in sample_paths[:saddle_examples]):
        return False
    return _diagnostic_result_is_complete(metrics)


def _run_formulation_diagnostic(
    *,
    result: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    max_items: int,
    noise_scale: float,
) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    if isinstance(metrics, dict) and _diagnostic_result_is_complete(metrics):
        return result

    dataset_bundle = resolve_ot_dataset(config)
    device = infer_device(str(config["training"].get("device", "auto")))
    diagnostic_metrics = run_case1_formulation_diagnostic(
        run_dir=output_dir,
        config=config,
        dataset_bundle=dataset_bundle,
        device=device,
        max_items=max_items,
        noise_scale=noise_scale,
        seed=int(config["training"]["seed"]),
    )
    updated = dict(result)
    updated_metrics = dict(metrics) if isinstance(metrics, dict) else {}
    updated_metrics.update(diagnostic_metrics)
    updated["metrics"] = updated_metrics
    (output_dir / "results.json").write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return updated


def _run_one(
    *,
    solver: str,
    dataset_name: str,
    seed: int,
    output_dir: Path,
    cache_version: str,
    eval_items: int,
    visualization_items: int,
    saddle_examples: int,
    diagnostic_checkpoints: int,
    diagnostic_items: int,
    diagnostic_noise_scale: float,
    spec: dict[str, Any],
    device: str,
    rerun: bool,
    overrides: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _build_run_config(
        solver_name=solver,
        dataset_name=dataset_name,
        seed=seed,
        output_dir=output_dir,
        spec=spec,
        cache_version=cache_version,
        device=device,
        eval_items=eval_items,
        visualization_items=visualization_items,
        saddle_examples=saddle_examples,
        diagnostic_checkpoints=diagnostic_checkpoints,
        overrides=overrides,
    )
    result_path = output_dir / "results.json"
    checkpoint_path = output_dir / "checkpoints" / "best.pt"

    if result_path.exists() and not rerun:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if _result_is_complete(existing, saddle_examples=saddle_examples):
            existing["run_dir"] = str(output_dir)
            return existing
        if checkpoint_path.exists():
            refreshed = eval_baseline_run(config, checkpoint_path=checkpoint_path, output_root=output_dir)
            refreshed = _run_formulation_diagnostic(
                result=refreshed,
                config=config,
                output_dir=output_dir,
                max_items=diagnostic_items,
                noise_scale=diagnostic_noise_scale,
            )
            refreshed["run_dir"] = str(output_dir)
            return refreshed

    result = train_baseline_run(config, output_root=output_dir)
    result = _run_formulation_diagnostic(
        result=result,
        config=config,
        output_dir=output_dir,
        max_items=diagnostic_items,
        noise_scale=diagnostic_noise_scale,
    )
    result["run_dir"] = str(output_dir)
    return result


def _aggregate(rows: list[dict[str, Any]], metric_names: tuple[str, ...]) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        solver = str(row["solver_id"])
        metrics = row.get("metrics", {})
        grouped.setdefault(solver, {})
        for metric in metric_names:
            value = metrics.get(metric) if isinstance(metrics, dict) else None
            if value is None:
                continue
            grouped[solver].setdefault(metric, []).append(float(value))

    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for solver, metrics in grouped.items():
        summary[solver] = {}
        for metric, values in metrics.items():
            summary[solver][metric] = (mean(values), pstdev(values) if len(values) > 1 else 0.0)
    return summary


def _format_summary_value(values: tuple[float, float] | None) -> str:
    if values is None:
        return "---"
    mean_value, std_value = values
    if not math.isfinite(mean_value) or not math.isfinite(std_value):
        return "---"
    return f"{mean_value:.4f} +- {std_value:.4f}"


def _select_representative_row(
    rows: list[dict[str, Any]],
    solver: str,
    metric: str = "map_l2",
) -> dict[str, Any] | None:
    solver_rows = [row for row in rows if row["solver_id"] == solver and row["metrics"].get(metric) is not None]
    if not solver_rows:
        return None
    target = mean(float(row["metrics"][metric]) for row in solver_rows)
    return min(solver_rows, key=lambda row: abs(float(row["metrics"][metric]) - target))


def _diagnostic_label(
    solver: str,
    diagnostic_summary: dict[str, dict[str, tuple[float, float]]],
) -> str:
    solver_summary = diagnostic_summary.get(solver, {})
    if "diagnostic_best_target_potential_gradient_rmse" in solver_summary:
        gradient_value = _format_summary_value(solver_summary.get("diagnostic_best_target_potential_gradient_rmse"))
        flatness_value = _format_summary_value(solver_summary.get("diagnostic_best_flatness_std_F"))
        return f"target_grad_rmse: {gradient_value}; flatness_std: {flatness_value}"
    if "diagnostic_best_flatness_std_F" in solver_summary:
        return f"flatness_std: {_format_summary_value(solver_summary.get('diagnostic_best_flatness_std_F'))}"
    if "diagnostic_best_potential_centered_rmse" in solver_summary:
        return (
            "potential_centered_rmse: "
            f"{_format_summary_value(solver_summary.get('diagnostic_best_potential_centered_rmse'))}"
        )
    return "---"


def _write_markdown(
    path: Path,
    *,
    summary: dict[str, dict[str, tuple[float, float]]],
    diagnostic_summary: dict[str, dict[str, tuple[float, float]]],
    solvers: list[str],
    representative_rows: dict[str, dict[str, Any]],
    dispersion_summary: dict[str, dict[str, Any]],
    seed_count: int,
) -> None:
    lines = [
        "# Case Study 1 Formulations",
        "",
        f"Average over `{seed_count}` seeds.",
        "",
        "| Solver | map_l2 | pushforward_w2 | transport_cos | diagnostic | seed_potential_dispersion |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for solver in solvers:
        solver_summary = summary.get(solver, {})
        dispersion = dispersion_summary.get(solver, {}).get("seed_potential_dispersion_rmse")
        lines.append(
            "| "
            + " | ".join(
                [
                    solver,
                    _format_summary_value(solver_summary.get("map_l2")),
                    _format_summary_value(solver_summary.get("pushforward_w2")),
                    _format_summary_value(solver_summary.get("transport_cos")),
                    _diagnostic_label(solver, diagnostic_summary),
                    f"{float(dispersion):.4f}" if dispersion is not None and math.isfinite(float(dispersion)) else "---",
                ]
            )
            + " |"
        )

    lines.extend(["", "Representative diagnostics", ""])
    for solver in solvers:
        row = representative_rows.get(solver)
        if row is None:
            continue
        metrics = row["metrics"]
        lines.append(f"- `{solver}` seed `{row['seed']}` transport: `{metrics.get('transport_figure_path')}`")
        lines.append(f"- `{solver}` seed `{row['seed']}` diagnostic: `{metrics.get('diagnostic_plot_path')}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct-formulation case-study-1 comparisons.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--solvers", default=",".join(DEFAULT_SOLVERS))
    parser.add_argument("--cache-version", default=DEFAULT_CACHE_VERSION)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--eval-items", type=int, default=4096)
    parser.add_argument("--visualization-items", type=int, default=DEFAULT_VISUALIZATION_ITEMS)
    parser.add_argument("--saddle-examples", type=int, default=DEFAULT_SADDLE_EXAMPLES)
    parser.add_argument("--diagnostic-checkpoints", type=int, default=DEFAULT_DIAGNOSTIC_CHECKPOINTS)
    parser.add_argument("--diagnostic-items", type=int, default=DEFAULT_DIAGNOSTIC_ITEMS)
    parser.add_argument("--diagnostic-noise-scale", type=float, default=DEFAULT_DIAGNOSTIC_NOISE_SCALE)
    parser.add_argument("--device", default=os.environ.get("TRAIN_DEVICE", "auto"))
    parser.add_argument("--spec-overrides", default="")
    parser.add_argument("--rerun", action="store_true", default=_env_flag("RERUN", False))
    parser.add_argument("--override", action="append", default=[], help="Additional config override applied to every run.")
    args = parser.parse_args()

    dataset_name = str(args.dataset)
    seeds = _parse_csv_ints(args.seeds)
    solvers = _parse_csv_strings(args.solvers)
    base_specs = _solver_specs_for_dataset(dataset_name)
    spec_overrides = _parse_spec_overrides(str(args.spec_overrides))
    solver_specs = _scale_solver_specs(
        _merge_solver_specs(base_specs, spec_overrides),
        budget_scale=float(args.budget_scale),
    )
    for solver in solvers:
        if solver not in solver_specs:
            raise ValueError(f"Unknown solver '{solver}'")

    output_root_arg = str(args.output_root).strip() or _default_output_root(dataset_name)
    output_root = _resolve_output_path(output_root_arg)
    output_root.mkdir(parents=True, exist_ok=True)
    training_device = _resolve_device(str(args.device)).type

    rows: list[dict[str, Any]] = []
    for solver in solvers:
        solver_output = output_root / solver
        solver_output.mkdir(parents=True, exist_ok=True)
        spec = copy.deepcopy(solver_specs[solver])
        for seed in seeds:
            run_dir = solver_output / f"{solver}_seed{seed}"
            rows.append(
                _run_one(
                    solver=solver,
                    dataset_name=dataset_name,
                    seed=seed,
                    output_dir=run_dir,
                    cache_version=str(args.cache_version),
                    eval_items=int(args.eval_items),
                    visualization_items=int(args.visualization_items),
                    saddle_examples=int(args.saddle_examples),
                    diagnostic_checkpoints=int(args.diagnostic_checkpoints),
                    diagnostic_items=int(args.diagnostic_items),
                    diagnostic_noise_scale=float(args.diagnostic_noise_scale),
                    spec=spec,
                    device=training_device,
                    rerun=bool(args.rerun),
                    overrides=list(args.override),
                )
            )

    summary = _aggregate(rows, SUMMARY_METRICS)
    diagnostic_summary = _aggregate(rows, DIAGNOSTIC_METRICS)
    representative_rows = {
        solver: row
        for solver in solvers
        if (row := _select_representative_row(rows, solver)) is not None
    }

    dispersion_summary: dict[str, dict[str, Any]] = {}
    for solver in solvers:
        solver_rows = [row for row in rows if row["solver_id"] == solver]
        if not solver_rows:
            continue
        reference_seed = int(solver_rows[0]["seed"])
        config = _build_run_config(
            solver_name=solver,
            dataset_name=dataset_name,
            seed=reference_seed,
            output_dir=output_root / solver / f"{solver}_seed{reference_seed}",
            spec=solver_specs[solver],
            cache_version=str(args.cache_version),
            device=training_device,
            eval_items=int(args.eval_items),
            visualization_items=int(args.visualization_items),
            saddle_examples=int(args.saddle_examples),
            diagnostic_checkpoints=int(args.diagnostic_checkpoints),
            overrides=list(args.override),
        )
        dataset_bundle = resolve_ot_dataset(config)
        device = infer_device(training_device)
        dispersion_summary[solver] = potential_seed_dispersion(
            run_dirs=[row["run_dir"] for row in solver_rows],
            config=config,
            dataset_bundle=dataset_bundle,
            device=device,
            max_items=int(args.diagnostic_items),
        )

    run_specs = {
        "_metadata": {
            "dataset": dataset_name,
            "cache_version": str(args.cache_version),
            "evaluation_max_items": int(args.eval_items),
            "visualization_max_items": int(args.visualization_items),
            "saddle_examples": int(args.saddle_examples),
            "diagnostic_checkpoints": int(args.diagnostic_checkpoints),
            "diagnostic_items": int(args.diagnostic_items),
            "diagnostic_noise_scale": float(args.diagnostic_noise_scale),
            "device": training_device,
            "budget_scale": float(args.budget_scale),
            "spec_overrides": spec_overrides,
            "global_overrides": list(args.override),
        },
        **{solver: solver_specs[solver] for solver in solvers},
    }
    (output_root / "run_specs.json").write_text(json.dumps(run_specs, indent=2), encoding="utf-8")
    (output_root / "seed_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "potential_dispersion_summary.json").write_text(
        json.dumps(dispersion_summary, indent=2), encoding="utf-8"
    )
    _write_markdown(
        output_root / "summary.md",
        summary=summary,
        diagnostic_summary=diagnostic_summary,
        solvers=solvers,
        representative_rows=representative_rows,
        dispersion_summary=dispersion_summary,
        seed_count=len(seeds),
    )


if __name__ == "__main__":
    main()
