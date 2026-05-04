"""Run the case study 1 paper suite and emit the final table artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import eval_baseline_run, resolve_ot_dataset, train_baseline_run
from src.diagnostics.case1_stability import run_case1_stability_diagnostic
from src.utils.device import infer_device

DEFAULT_SEEDS = [
    686499,
    928801,
    48156,
    431753,
    526655,
    480953,
    178898,
    700645,
    62001,
    192385,
    911170,
    307368,
    546159,
    381730,
    917274,
    910268,
    301123,
    218738,
    982909,
    870264,
    948176,
    164608,
    892045,
    767301,
    576270,
    525000,
    7029,
    644131,
    480217,
    389379,
]
DEFAULT_SOLVERS = ["gaussian", "mm", "mmv2", "tw2", "mm_b", "qc", "otp", "monge_map", "otm", "maxcorr", "makkuva_icnn_cvx"]
DEFAULT_CACHE_VERSION = "paper_ref_d64_b256_s25k"
DEFAULT_OUTPUT_ROOT = "outputs/paper_case1_suite"
DEFAULT_VISUALIZATION_ITEMS = 512
DEFAULT_SADDLE_EXAMPLES = 3
DEFAULT_STABILITY_CHECKPOINTS = 5
DEFAULT_STABILITY_ITEMS = 512
DEFAULT_STABILITY_NOISE_SCALE = 1.0e-2

SOLVER_SPECS: dict[str, dict[str, object]] = {
    "gaussian": {
        "max_steps": 1,
        "batch_size": 1024,
        "steps_per_epoch": 1,
        "extra_overrides": [],
    },
    "mm": {
        "max_steps": 4096,
        "batch_size": 2048,
        "steps_per_epoch": 128,
        "extra_overrides": [
            "training.gradient_clip_norm=0.5",
            "solver.forward_lr=5e-4",
            "solver.inverse_lr=5e-4",
            "solver.inner_steps=10",
            "solver.identity_pretrain_batch_size=2048",
        ],
    },
    "mmv2": {
        "max_steps": 4096,
        "batch_size": 1024,
        "steps_per_epoch": 128,
        "extra_overrides": [
            "training.gradient_clip_norm=0.5",
            "solver.forward_lr=1e-3",
            "solver.inverse_lr=1e-3",
            "solver.inner_steps=15",
            "solver.identity_pretrain_batch_size=2048",
        ],
    },
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
        "extra_overrides": [
            "solver.transport_steps=1",
            "solver.transport_lr=5e-4",
            "solver.potential_lr=5e-4",
            "solver.noise.sigma_start=0.0",
            "solver.noise.sigma_end=0.0",
        ],
    },
    "otm": {
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
    "tw2": {
        "max_steps": 10000,
        "batch_size": 512,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
    "mm_b": {
        "max_steps": 10000,
        "batch_size": 2048,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
    "qc": {
        "max_steps": 1024,
        "batch_size": 512,
        "steps_per_epoch": 64,
        "extra_overrides": [
            "solver.forward_lr=1e-4",
            "solver.regularization_weight=128.0",
            "solver.identity_pretrain_batch_size=2048",
        ],
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
SUMMARY_METRICS = ("map_l2", "l2_uvp", "transport_cos", "saddle_residual")
STABILITY_SUMMARY_METRICS = (
    "stability_best_potential_centered_rmse",
    "stability_best_forward_flatness_std_F",
)
LATEX_SOLVER_NAMES = {
    "gaussian": "Gaussian",
    "mm": "tMM",
    "mmv2": "tMMv2",
    "otp": "OTP",
    "monge_map": "MongeMap",
    "otm": "OTM",
    "tw2": "tW2",
    "mm_b": "tMM-B",
    "qc": "tQC",
    "maxcorr": "MaxCorr",
    "makkuva_icnn_cvx": "Makkuva-ICNN",
}
CASE1_STABILITY_SOLVERS = {"mm", "mmv2", "tw2", "mm_b", "qc"}
TABLE_COLUMNS = [
    ("map_l2", "Map L2 $\\downarrow$"),
    ("l2_uvp", "L2 UVP $\\downarrow$"),
    ("transport_cos", "Cosine Similarity $\\uparrow$"),
    ("saddle_residual", "Saddle Residual $\\downarrow$"),
]
STABILITY_TABLE_COLUMNS = [
    ("stability_best_potential_centered_rmse", "Centered Potential RMSE $\\downarrow$"),
    ("stability_best_forward_flatness_std_F", "Forward Flatness Std $\\downarrow$"),
]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _resolve_output_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def _supports_case1_stability(solver: str) -> bool:
    return solver in CASE1_STABILITY_SOLVERS


def _stability_checkpoint_interval(
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


def _merge_solver_specs(spec_overrides: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    merged = copy.deepcopy(SOLVER_SPECS)
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


def _apply_override(config: dict[str, Any], override: str) -> None:
    key, raw_value = override.split("=", 1)
    value = OmegaConf.create({"value": raw_value})["value"]
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
    seed: int,
    output_dir: Path,
    spec: dict[str, Any],
    cache_version: str,
    device: str,
    eval_items: int,
    visualization_items: int,
    saddle_examples: int,
    stability_checkpoints: int,
    overrides: list[str],
) -> dict[str, Any]:
    model = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/model/ot_map.yaml"), resolve=True)
    solver = OmegaConf.to_container(OmegaConf.load(ROOT / f"configs/solver/{solver_name}.yaml"), resolve=True)
    dataset = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/dataset/paper_mix3to10.yaml"), resolve=True)
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
            "name": "paper_case1_suite",
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
    config["model"]["input_dim"] = 64
    config["model"]["output_dim"] = 64
    config["dataset"]["batch_size"] = int(spec["batch_size"])
    config["dataset"]["steps_per_epoch"] = int(spec["steps_per_epoch"])
    config["dataset"]["cache_version"] = cache_version
    config["training"]["seed"] = seed
    config["training"]["device"] = device
    config["training"]["max_steps"] = int(spec["max_steps"])
    config["training"]["max_epochs"] = 1000
    config["training"]["checkpointing"]["save_every_n_epochs"] = (
        _stability_checkpoint_interval(
            max_steps=int(spec["max_steps"]),
            steps_per_epoch=int(spec["steps_per_epoch"]),
            num_checkpoints=stability_checkpoints,
        )
        if _supports_case1_stability(solver_name)
        else 0
    )
    config["training"]["checkpointing"]["monitor"] = "val/map_l2"
    config["training"]["checkpointing"]["mode"] = "min"

    for override in spec.get("extra_overrides", []):
        _apply_override(config, str(override))
    for override in overrides:
        _apply_override(config, override)
    return config


def _stability_result_is_complete(metrics: Mapping[str, Any]) -> bool:
    plot_path = metrics.get("stability_plot_path")
    results_path = metrics.get("stability_results_path")
    return (
        metrics.get("stability_best_potential_centered_rmse") is not None
        and metrics.get("stability_best_forward_flatness_std_F") is not None
        and bool(plot_path)
        and bool(results_path)
        and Path(str(plot_path)).exists()
        and Path(str(results_path)).exists()
    )


def _result_is_complete(
    result: dict[str, Any],
    *,
    solver: str,
    saddle_examples: int,
) -> bool:
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
    if not isinstance(sample_paths, list):
        return False
    if len(sample_paths) < saddle_examples:
        return False
    if not all(Path(str(path)).exists() for path in sample_paths[:saddle_examples]):
        return False
    if _supports_case1_stability(solver) and not _stability_result_is_complete(metrics):
        return False
    return True


def _keep_best_checkpoint_only(run_dir: Path) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return
    for checkpoint_path in checkpoint_dir.glob("*.pt"):
        if checkpoint_path.name != "best.pt":
            checkpoint_path.unlink(missing_ok=True)


def _run_one(
    *,
    solver: str,
    seed: int,
    output_dir: Path,
    cache_version: str,
    eval_items: int,
    visualization_items: int,
    saddle_examples: int,
    stability_checkpoints: int,
    stability_items: int,
    stability_noise_scale: float,
    spec: dict[str, Any],
    device: str,
    rerun: bool,
    overrides: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _build_run_config(
        solver_name=solver,
        seed=seed,
        output_dir=output_dir,
        spec=spec,
        cache_version=cache_version,
        device=device,
        eval_items=eval_items,
        visualization_items=visualization_items,
        saddle_examples=saddle_examples,
        stability_checkpoints=stability_checkpoints,
        overrides=overrides,
    )
    result_path = output_dir / "results.json"
    checkpoint_path = output_dir / "checkpoints" / "best.pt"

    if result_path.exists() and not rerun:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if _result_is_complete(existing, solver=solver, saddle_examples=saddle_examples):
            _keep_best_checkpoint_only(output_dir)
            return existing
        if checkpoint_path.exists():
            refreshed = eval_baseline_run(config, checkpoint_path=checkpoint_path, output_root=output_dir)
            refreshed = _run_case1_stability(
                result=refreshed,
                solver=solver,
                config=config,
                output_dir=output_dir,
                max_items=stability_items,
                noise_scale=stability_noise_scale,
            )
            _keep_best_checkpoint_only(output_dir)
            return refreshed

    result = train_baseline_run(config, output_root=output_dir)
    result = _run_case1_stability(
        result=result,
        solver=solver,
        config=config,
        output_dir=output_dir,
        max_items=stability_items,
        noise_scale=stability_noise_scale,
    )
    _keep_best_checkpoint_only(output_dir)
    return result


def _run_case1_stability(
    *,
    result: dict[str, Any],
    solver: str,
    config: Mapping[str, Any],
    output_dir: Path,
    max_items: int,
    noise_scale: float,
) -> dict[str, Any]:
    if not _supports_case1_stability(solver):
        return result

    metrics = result.get("metrics", {})
    if isinstance(metrics, dict) and _stability_result_is_complete(metrics):
        return result

    dataset_bundle = resolve_ot_dataset(config)
    device = infer_device(str(config["training"].get("device", "auto")))
    stability_metrics = run_case1_stability_diagnostic(
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
    updated_metrics.update(stability_metrics)
    updated["metrics"] = updated_metrics
    (output_dir / "results.json").write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return updated


def _aggregate_named_metrics(
    rows: list[dict[str, Any]],
    metric_names: tuple[str, ...],
) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        solver = str(row["solver_id"])
        grouped.setdefault(solver, {})
        for metric in metric_names:
            value = row["metrics"].get(metric)
            if value is None:
                continue
            grouped[solver].setdefault(metric, []).append(float(value))

    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for solver, metrics in grouped.items():
        summary[solver] = {}
        for metric, values in metrics.items():
            summary[solver][metric] = (mean(values), pstdev(values) if len(values) > 1 else 0.0)
    return summary


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, tuple[float, float]]]:
    return _aggregate_named_metrics(rows, SUMMARY_METRICS)


def _format_summary_value(values: tuple[float, float] | None, *, latex: bool) -> str:
    if values is None:
        return "---"
    mean_value, std_value = values
    if not math.isfinite(mean_value) or not math.isfinite(std_value):
        return "---"
    if latex:
        return f"{_format_sig(mean_value)} $\\pm$ {_format_sig(std_value)}"
    return f"{mean_value:.4f} +- {std_value:.4f}"


def _format_sig(value: float) -> str:
    if value == 0.0:
        return "0.000"
    exponent = math.floor(math.log10(abs(value)))
    digits_after_decimal = max(0, 2 - exponent)
    return f"{value:.{digits_after_decimal}f}"


def _metric_ranks(
    summary: dict[str, dict[str, tuple[float, float]]],
    metric: str,
) -> tuple[str | None, str | None]:
    values = [
        (solver, summary[solver][metric][0])
        for solver in summary
        if metric in summary[solver] and math.isfinite(summary[solver][metric][0])
    ]
    if not values:
        return None, None
    reverse = metric == "transport_cos"
    ordered = sorted(values, key=lambda item: item[1], reverse=reverse)
    best = ordered[0][0]
    second = ordered[1][0] if len(ordered) > 1 else None
    return best, second


def _write_latex_table(
    path: Path,
    *,
    summary: dict[str, dict[str, tuple[float, float]]],
    solvers: list[str],
    seed_count: int,
) -> None:
    rankings = {metric: _metric_ranks(summary, metric) for metric, _ in TABLE_COLUMNS}
    lines = [
        "\\begin{table*}[h]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "c" * len(TABLE_COLUMNS) + "}",
        "\\toprule",
        "Solver & " + " & ".join(label for _, label in TABLE_COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for solver in solvers:
        display_name = LATEX_SOLVER_NAMES.get(solver, solver)
        cells = [display_name]
        solver_metrics = summary.get(solver, {})
        for metric, _ in TABLE_COLUMNS:
            text = _format_summary_value(solver_metrics.get(metric), latex=True)
            best, second = rankings[metric]
            if solver == best and text != "---":
                text = f"\\textbf{{{text}}}"
            elif solver == second and text != "---":
                text = f"\\emph{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{Average over {seed_count} runs.}}",
            "\\label{tab:case1_broad_30seed_results}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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


def _write_markdown(
    path: Path,
    *,
    summary: dict[str, dict[str, tuple[float, float]]],
    solvers: list[str],
    representative_rows: dict[str, dict[str, Any]],
    seed_count: int,
    cache_version: str,
    saddle_examples: int,
) -> None:
    lines = [
        "# Case Study 1",
        "",
        f"Cache version: `{cache_version}`.",
        f"Average over `{seed_count}` seeds.",
        f"Per-seed artifacts: transport geometry, saddle geometry, and `{saddle_examples}` sampled saddle plots.",
        "",
        "| Solver | map_l2 | l2_uvp | transport_cos | saddle_residual |",
        "| --- | --- | --- | --- | --- |",
    ]
    for solver in solvers:
        display_name = LATEX_SOLVER_NAMES.get(solver, solver)
        metrics = summary.get(solver, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    display_name,
                    _format_summary_value(metrics.get("map_l2"), latex=False),
                    _format_summary_value(metrics.get("l2_uvp"), latex=False),
                    _format_summary_value(metrics.get("transport_cos"), latex=False),
                    _format_summary_value(metrics.get("saddle_residual"), latex=False),
                ]
            )
            + " |"
        )

    if representative_rows:
        lines.extend(["", "Representative runs", ""])
        for solver in solvers:
            row = representative_rows.get(solver)
            if row is None:
                continue
            display_name = LATEX_SOLVER_NAMES.get(solver, solver)
            metrics = row["metrics"]
            lines.append(
                f"- `{display_name}` seed `{row['seed']}` transport: `{metrics.get('transport_figure_path')}`"
            )
            lines.append(
                f"- `{display_name}` seed `{row['seed']}` saddle: `{metrics.get('saddle_figure_path')}`"
            )
            sample_paths = metrics.get("saddle_sample_paths", [])
            for sample_path in sample_paths[:saddle_examples]:
                lines.append(f"- `{display_name}` sampled saddle: `{sample_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stability_latex_table(
    path: Path,
    *,
    summary: dict[str, dict[str, tuple[float, float]]],
    solvers: list[str],
    seed_count: int,
) -> None:
    learned_solvers = [solver for solver in solvers if solver in summary]
    rankings = {metric: _metric_ranks(summary, metric) for metric, _ in STABILITY_TABLE_COLUMNS}
    lines = [
        "\\begin{table*}[h]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "c" * len(STABILITY_TABLE_COLUMNS) + "}",
        "\\toprule",
        "Solver & " + " & ".join(label for _, label in STABILITY_TABLE_COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for solver in learned_solvers:
        display_name = LATEX_SOLVER_NAMES.get(solver, solver)
        cells = [display_name]
        solver_metrics = summary.get(solver, {})
        for metric, _ in STABILITY_TABLE_COLUMNS:
            text = _format_summary_value(solver_metrics.get(metric), latex=True)
            best, second = rankings[metric]
            if solver == best and text != "---":
                text = f"\\textbf{{{text}}}"
            elif solver == second and text != "---":
                text = f"\\emph{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{Case-study-1 stability diagnostics averaged over {seed_count} runs.}}",
            "\\label{tab:case1_stability_results}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_stability_markdown(
    path: Path,
    *,
    summary: dict[str, dict[str, tuple[float, float]]],
    solvers: list[str],
    representative_rows: dict[str, dict[str, Any]],
    seed_count: int,
) -> None:
    learned_solvers = [solver for solver in solvers if solver in summary]
    lines = [
        "# Case Study 1 Stability",
        "",
        "These metrics are computed from the best checkpoint for each run, using the fixed-map",
        "forward-potential objective over saved checkpoints plus `last.pt`.",
        f"Average over `{seed_count}` seeds.",
        "",
        "| Solver | centered_potential_rmse | forward_flatness_std |",
        "| --- | --- | --- |",
    ]
    for solver in learned_solvers:
        display_name = LATEX_SOLVER_NAMES.get(solver, solver)
        metrics = summary.get(solver, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    display_name,
                    _format_summary_value(metrics.get("stability_best_potential_centered_rmse"), latex=False),
                    _format_summary_value(metrics.get("stability_best_forward_flatness_std_F"), latex=False),
                ]
            )
            + " |"
        )

    if representative_rows:
        lines.extend(["", "Representative stability plots", ""])
        for solver in learned_solvers:
            row = representative_rows.get(solver)
            if row is None:
                continue
            plot_path = row.get("metrics", {}).get("stability_plot_path")
            if plot_path:
                lines.append(
                    f"- `{LATEX_SOLVER_NAMES.get(solver, solver)}` seed `{row['seed']}` stability: `{plot_path}`"
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the case study 1 paper suite.")
    parser.add_argument("--seeds", default=os.environ.get("SEEDS", ",".join(str(seed) for seed in DEFAULT_SEEDS)))
    parser.add_argument("--solvers", default=os.environ.get("SOLVERS", ",".join(DEFAULT_SOLVERS)))
    parser.add_argument("--cache-version", default=os.environ.get("CACHE_VERSION", DEFAULT_CACHE_VERSION))
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--eval-items", type=int, default=int(os.environ.get("EVAL_ITEMS", "4096")))
    parser.add_argument(
        "--visualization-items",
        type=int,
        default=int(os.environ.get("VISUALIZATION_ITEMS", str(DEFAULT_VISUALIZATION_ITEMS))),
    )
    parser.add_argument(
        "--saddle-examples",
        type=int,
        default=int(os.environ.get("SADDLE_EXAMPLES", str(DEFAULT_SADDLE_EXAMPLES))),
    )
    parser.add_argument(
        "--stability-checkpoints",
        type=int,
        default=int(os.environ.get("STABILITY_CHECKPOINTS", str(DEFAULT_STABILITY_CHECKPOINTS))),
    )
    parser.add_argument(
        "--stability-items",
        type=int,
        default=int(os.environ.get("STABILITY_ITEMS", str(DEFAULT_STABILITY_ITEMS))),
    )
    parser.add_argument(
        "--stability-noise-scale",
        type=float,
        default=float(os.environ.get("STABILITY_NOISE_SCALE", str(DEFAULT_STABILITY_NOISE_SCALE))),
    )
    parser.add_argument("--device", default=os.environ.get("TRAIN_DEVICE", "auto"))
    parser.add_argument("--budget-scale", type=float, default=float(os.environ.get("BUDGET_SCALE", "1.0")))
    parser.add_argument("--spec-overrides", default=os.environ.get("SPEC_OVERRIDES", ""))
    parser.add_argument("--rerun", action="store_true", default=_env_flag("RERUN", False))
    parser.add_argument("--override", action="append", default=[], help="Additional config override applied to every run.")
    args = parser.parse_args()

    seeds = _parse_csv_ints(args.seeds)
    solvers = _parse_csv_strings(args.solvers)
    solver_specs = _scale_solver_specs(
        _merge_solver_specs(_parse_spec_overrides(str(args.spec_overrides))),
        budget_scale=float(args.budget_scale),
    )
    for solver in solvers:
        if solver not in solver_specs:
            raise ValueError(f"Unknown solver '{solver}'")

    output_root = _resolve_output_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    training_device = _resolve_device(str(args.device)).type

    rows: list[dict[str, Any]] = []
    for solver in solvers:
        solver_output = output_root / solver
        solver_output.mkdir(parents=True, exist_ok=True)
        spec = solver_specs[solver]
        for seed in seeds:
            run_dir = solver_output / f"{solver}_seed{seed}"
            rows.append(
                _run_one(
                    solver=solver,
                    seed=seed,
                    output_dir=run_dir,
                    cache_version=str(args.cache_version),
                    eval_items=int(args.eval_items),
                    visualization_items=int(args.visualization_items),
                    saddle_examples=int(args.saddle_examples),
                    stability_checkpoints=int(args.stability_checkpoints),
                    stability_items=int(args.stability_items),
                    stability_noise_scale=float(args.stability_noise_scale),
                    spec=spec,
                    device=training_device,
                    rerun=bool(args.rerun),
                    overrides=list(args.override),
                )
            )

    summary = _aggregate(rows)
    stability_summary = _aggregate_named_metrics(rows, STABILITY_SUMMARY_METRICS)
    representative_rows = {
        solver: row
        for solver in solvers
        if (row := _select_representative_row(rows, solver)) is not None
    }

    run_specs = {
        "_metadata": {
            "cache_version": str(args.cache_version),
            "checkpoint_monitor": "val/map_l2",
            "evaluation_max_items": int(args.eval_items),
            "visualization_max_items": int(args.visualization_items),
            "saddle_examples": int(args.saddle_examples),
            "stability_checkpoints": int(args.stability_checkpoints),
            "stability_items": int(args.stability_items),
            "stability_noise_scale": float(args.stability_noise_scale),
            "device": training_device,
            "budget_scale": float(args.budget_scale),
            "global_overrides": list(args.override),
            "spec_overrides": _parse_spec_overrides(str(args.spec_overrides)),
        },
        **{solver: solver_specs[solver] for solver in solvers},
    }
    (output_root / "run_specs.json").write_text(json.dumps(run_specs, indent=2), encoding="utf-8")
    (output_root / "seed_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "stability_seed_summary.json").write_text(json.dumps(stability_summary, indent=2), encoding="utf-8")
    (output_root / "representative_runs.json").write_text(
        json.dumps(representative_rows, indent=2), encoding="utf-8"
    )
    _write_latex_table(
        output_root / "case1_results_table.tex",
        summary=summary,
        solvers=solvers,
        seed_count=len(seeds),
    )
    _write_markdown(
        output_root / "seed_summary.md",
        summary=summary,
        solvers=solvers,
        representative_rows=representative_rows,
        seed_count=len(seeds),
        cache_version=str(args.cache_version),
        saddle_examples=int(args.saddle_examples),
    )
    _write_stability_latex_table(
        output_root / "case1_stability_table.tex",
        summary=stability_summary,
        solvers=solvers,
        seed_count=len(seeds),
    )
    _write_stability_markdown(
        output_root / "stability_seed_summary.md",
        summary=stability_summary,
        solvers=solvers,
        representative_rows=representative_rows,
        seed_count=len(seeds),
    )


if __name__ == "__main__":
    main()
