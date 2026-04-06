"""Run and report the case study 1 broad comparison."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import collect_ot_predictions, eval_baseline_run, load_solver_checkpoint, resolve_ot_dataset
from src.evaluation.visualization import save_ot_visualizations
from src.solvers.registry import build_solver

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
DEFAULT_SOLVERS = ["gaussian", "mm", "mmv2", "tw2", "mm_b", "qc"]
DEFAULT_FIGURE_SOLVERS = ["mm", "mmv2"]
DEFAULT_CACHE_VERSION = "paper_ref_d64_b256_s25k"

SOLVER_SPECS: dict[str, dict[str, object]] = {
    "gaussian": {
        "max_steps": 1,
        "batch_size": 1024,
        "steps_per_epoch": 1,
        "extra_overrides": [],
    },
    "mm": {
        "max_steps": 2048,
        "batch_size": 1024,
        "steps_per_epoch": 128,
        "extra_overrides": [
            "solver.forward_lr=1e-3",
            "solver.inverse_lr=1e-3",
            "solver.inner_steps=15",
        ],
    },
    "mmv2": {
        "max_steps": 2048,
        "batch_size": 1024,
        "steps_per_epoch": 128,
        "extra_overrides": [
            "solver.forward_lr=1e-3",
            "solver.inverse_lr=1e-3",
            "solver.inner_steps=15",
        ],
    },
    "tw2": {
        "max_steps": 2048,
        "batch_size": 1024,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
    "mm_b": {
        "max_steps": 2048,
        "batch_size": 1024,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
    "qc": {
        "max_steps": 2048,
        "batch_size": 64,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
}
METRIC_SPECS = {
    "map_l2": {},
    "l2_uvp": {},
    "transport_cos": {},
    "saddle_residual": {},
    "mmd": {},
}
LATEX_SOLVER_NAMES = {
    "gaussian": "Gaussian",
    "mm": "tMM",
    "mmv2": "tICNN",
    "tw2": "tW2",
    "mm_b": "tMM-B",
    "qc": "tQC",
}
TABLE_COLUMNS = [
    ("map_l2", "Map L2 $\\downarrow$"),
    ("l2_uvp", "L2 UVP $\\downarrow$"),
    ("transport_cos", "Cosine Similarity $\\uparrow$"),
    ("saddle_residual", "Saddle Residual $\\downarrow$"),
    ("mmd", "MMD $\\downarrow$"),
]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _run_one(
    *,
    python_exe: str,
    solver: str,
    seed: int,
    output_dir: Path,
    cache_version: str,
    eval_items: int,
    spec: dict[str, object],
    device: str,
    rerun: bool,
) -> dict[str, Any]:
    result_path = output_dir / "results.json"
    if result_path.exists() and not rerun:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        metric_keys = set(existing.get("metrics", {}))
        if set(METRIC_SPECS).issubset(metric_keys):
            return existing
        checkpoint_path = output_dir / "checkpoints" / "best.pt"
        if checkpoint_path.exists():
            config = _build_run_config(
                solver_name=solver,
                seed=seed,
                spec=spec,
                cache_version=cache_version,
                device="cpu",
                eval_items=eval_items,
            )
            return eval_baseline_run(
                config=config,
                checkpoint_path=checkpoint_path,
                output_root=output_dir,
            )
        return existing

    command = [
        python_exe,
        "scripts/train_baseline.py",
        f"solver={solver}",
        "dataset=paper_mix3to10",
        f"training.device={device}",
        "training.gradient_clip_norm=null",
        "visualization.enabled=true",
        f"training.seed={seed}",
        f"training.max_steps={int(spec['max_steps'])}",
        "training.max_epochs=1000",
        f"+evaluation.max_items={eval_items}",
        f"dataset.cache_version={cache_version}",
        f"dataset.batch_size={int(spec['batch_size'])}",
        f"dataset.steps_per_epoch={int(spec['steps_per_epoch'])}",
        "model.input_dim=64",
        "model.output_dim=64",
        f"experiment.output_dir={output_dir.as_posix()}",
    ]
    command.extend(str(item) for item in spec.get("extra_overrides", []))
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(result_path.read_text(encoding="utf-8"))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        solver = str(row["solver_id"])
        grouped.setdefault(solver, {})
        for metric in METRIC_SPECS:
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


def _write_markdown(
    path: Path,
    *,
    summary: dict[str, dict[str, tuple[float, float]]],
    specs: dict[str, dict[str, object]],
    representative_rows: dict[str, dict[str, Any]],
    cache_version: str,
) -> None:
    metric_columns = [metric for metric, _ in TABLE_COLUMNS]
    lines = [
        "# Paper Case Study 1 Broad Comparison",
        "",
        f"Dataset cache: `{cache_version}`.",
        "Checkpoint selection monitor: `val/map_l2`.",
        "",
        "| Solver | " + " | ".join(metric_columns) + " |",
        "| --- | " + " | ".join("---" for _ in metric_columns) + " |",
    ]

    for solver in specs:
        metrics = summary.get(solver, {})

        def fmt(metric: str) -> str:
            value, spread = metrics.get(metric, (math.nan, math.nan))
            if not math.isfinite(value) or not math.isfinite(spread):
                return "n/a"
            return f"{value:.4f} ± {spread:.4f}"

        formatted_metrics = " | ".join(fmt(metric) for metric in metric_columns)
        lines.append(f"| {LATEX_SOLVER_NAMES.get(solver, solver)} | {formatted_metrics} |")

    if representative_rows:
        lines.extend(
            [
                "",
                "Representative geometry figures use the seed closest to each solver's mean final `map_l2`.",
                "",
            ]
        )
        for solver, row in representative_rows.items():
            transport_path = Path(str(row["transport_figure_path"]))
            display_name = LATEX_SOLVER_NAMES.get(solver, solver)
            lines.append(f"- `{display_name}` seed `{row['seed']}` transport: `{transport_path.relative_to(ROOT).as_posix()}`")
            saddle_path = row.get("saddle_figure_path")
            if saddle_path:
                lines.append(
                    f"- `{display_name}` seed `{row['seed']}` saddle: `{Path(str(saddle_path)).relative_to(ROOT).as_posix()}`"
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_sig(value: float) -> str:
    if value == 0.0:
        return "0.000"
    exponent = math.floor(math.log10(abs(value)))
    digits_after_decimal = max(0, 2 - exponent)
    return f"{value:.{digits_after_decimal}f}"


def _format_mean_pm_std(values: tuple[float, float] | None) -> str:
    if values is None:
        return "n/a"
    mean_value, std_value = values
    if not math.isfinite(mean_value) or not math.isfinite(std_value):
        return "n/a"
    return f"{_format_sig(mean_value)} $\\pm$ {_format_sig(std_value)}"


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
) -> None:
    rankings = {metric: _metric_ranks(summary, metric) for metric, _ in TABLE_COLUMNS}
    lines = [
        "\\begin{table*}[t]",
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
            text = _format_mean_pm_std(solver_metrics.get(metric))
            best, second = rankings[metric]
            if solver == best and text != "n/a":
                text = f"\\textbf{{{text}}}"
            elif solver == second and text != "n/a":
                text = f"\\emph{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:case1_broad_30seed_results}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _select_representative_row(rows: list[dict[str, Any]], solver: str, metric: str = "map_l2") -> dict[str, Any] | None:
    solver_rows = [row for row in rows if row["solver_id"] == solver and row["metrics"].get(metric) is not None]
    if not solver_rows:
        return None
    target = mean(float(row["metrics"][metric]) for row in solver_rows)
    return min(solver_rows, key=lambda row: abs(float(row["metrics"][metric]) - target))


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
    spec: dict[str, Any],
    cache_version: str,
    device: str,
    eval_items: int,
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
            "name": "paper_case1_full_compare",
            "output_dir": "outputs/paper_case1_full_compare",
        },
        "visualization": {"enabled": False, "dirpath": "visualizations", "max_items": 512},
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
    config["training"]["gradient_clip_norm"] = None
    for override in spec.get("extra_overrides", []):
        _apply_override(config, str(override))
    return config


def _render_representative_geometry(
    *,
    output_dir: Path,
    solver_name: str,
    seed: int,
    spec: dict[str, Any],
    cache_version: str,
    device: torch.device,
    eval_items: int,
    max_items: int,
    checkpoint_path: Path,
) -> dict[str, Path | None]:
    config = _build_run_config(
        solver_name=solver_name,
        seed=seed,
        spec=spec,
        cache_version=cache_version,
        device=device.type,
        eval_items=eval_items,
    )
    dataset_bundle = resolve_ot_dataset(config)
    solver = build_solver(config["model"], config["solver"], config["training"]).to(device)
    load_solver_checkpoint(solver, checkpoint_path)
    solver.eval()
    _, _, test_loader = dataset_bundle.make_dataloaders()
    aggregated = collect_ot_predictions(solver, test_loader, device=device, max_items=eval_items)
    transport_path = save_ot_visualizations(aggregated, output_dir, max_items=max_items, solver=solver)
    saddle_path = output_dir / "saddle_geometry.png"
    return {
        "transport": transport_path,
        "saddle": saddle_path if saddle_path.exists() else None,
    }


def _prune_legacy_report_artifacts(figures_dir: Path) -> None:
    if not figures_dir.exists():
        return
    for pattern in ("final_*.png", "final_*.pdf", "trajectory_*.png", "trajectory_*.pdf"):
        for path in figures_dir.glob(pattern):
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--solvers", default=",".join(DEFAULT_SOLVERS))
    parser.add_argument("--figure-solvers", default=",".join(DEFAULT_FIGURE_SOLVERS))
    parser.add_argument("--cache-version", default=DEFAULT_CACHE_VERSION)
    parser.add_argument("--output-root", default="outputs/paper_case1_full_compare_broad_30seeds_v3")
    parser.add_argument("--eval-items", type=int, default=4096)
    parser.add_argument("--visualization-items", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    seeds = _parse_csv_ints(args.seeds)
    solvers = _parse_csv_strings(args.solvers)
    figure_solvers = _parse_csv_strings(args.figure_solvers)
    for solver in solvers:
        if solver not in SOLVER_SPECS:
            raise ValueError(f"Unknown solver '{solver}'")

    output_root = _resolve_output_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for solver in solvers:
        solver_output = output_root / solver
        solver_output.mkdir(parents=True, exist_ok=True)
        spec = SOLVER_SPECS[solver]
        for seed in seeds:
            run_dir = solver_output / f"{solver}_seed{seed}"
            rows.append(
                _run_one(
                    python_exe=sys.executable,
                    solver=solver,
                    seed=seed,
                    output_dir=run_dir,
                    cache_version=str(args.cache_version),
                    eval_items=int(args.eval_items),
                    spec=spec,
                    device=str(args.device),
                    rerun=bool(args.rerun),
                )
            )

    summary = _aggregate(rows)
    figures_dir = output_root / "figures"
    _prune_legacy_report_artifacts(figures_dir)

    representative_rows: dict[str, dict[str, Any]] = {}
    report_device = _resolve_device(str(args.device))
    for solver in figure_solvers:
        if solver not in solvers:
            continue
        row = _select_representative_row(rows, solver)
        if row is None:
            continue
        checkpoint_path = output_root / solver / f"{solver}_seed{row['seed']}" / "checkpoints" / "best.pt"
        if not checkpoint_path.exists():
            continue
        solver_figure_dir = figures_dir / solver
        rendered_paths = _render_representative_geometry(
            output_dir=solver_figure_dir,
            solver_name=solver,
            seed=int(row["seed"]),
            spec=SOLVER_SPECS[solver],
            cache_version=str(args.cache_version),
            device=report_device,
            eval_items=int(args.eval_items),
            max_items=int(args.visualization_items),
            checkpoint_path=checkpoint_path,
        )
        representative_rows[solver] = {
            "seed": int(row["seed"]),
            "map_l2": float(row["metrics"]["map_l2"]),
            "transport_figure_path": str(rendered_paths["transport"]),
            "saddle_figure_path": str(rendered_paths["saddle"]) if rendered_paths["saddle"] is not None else None,
        }

    run_specs = {
        "_metadata": {
            "cache_version": str(args.cache_version),
            "checkpoint_monitor": "val/map_l2",
            "evaluation_max_items": int(args.eval_items),
            "visualization_max_items": int(args.visualization_items),
            "figure_solvers": figure_solvers,
        },
        **{solver: SOLVER_SPECS[solver] for solver in solvers},
    }
    (output_root / "run_specs.json").write_text(json.dumps(run_specs, indent=2), encoding="utf-8")
    (output_root / "seed_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "representative_runs.json").write_text(json.dumps(representative_rows, indent=2), encoding="utf-8")
    _write_latex_table(output_root / "case1_results_table.tex", summary=summary, solvers=solvers)
    _write_markdown(
        output_root / "seed_summary.md",
        summary=summary,
        specs={solver: SOLVER_SPECS[solver] for solver in solvers},
        representative_rows=representative_rows,
        cache_version=str(args.cache_version),
    )


if __name__ == "__main__":
    main()
