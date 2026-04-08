"""Run the Makkuva checkerboard comparison across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOLVERS = ["makkuva_icnn_cvx", "makkuva_mlp_ablation"]
DEFAULT_SEEDS = [1, 2, 3]
METRICS = ["pushforward_w2", "mmd", "w2_estimate"]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_one(
    *,
    python_exe: str,
    solver: str,
    seed: int,
    output_dir: Path,
    eval_items: int,
    rerun: bool,
    overrides: list[str],
) -> dict[str, Any]:
    result_path = output_dir / "results.json"
    if result_path.exists() and not rerun:
        return json.loads(result_path.read_text(encoding="utf-8"))

    command = [
        python_exe,
        "scripts/train_baseline.py",
        "dataset=makkuva_checkerboard",
        "training=makkuva_case1",
        f"solver={solver}",
        "visualization.enabled=true",
        "model.input_dim=2",
        "model.output_dim=2",
        f"training.seed={seed}",
        f"+evaluation.max_items={eval_items}",
        f"experiment.output_dir={output_dir.as_posix()}",
    ]
    command.extend(overrides)
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(result_path.read_text(encoding="utf-8"))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        solver = str(row["solver_id"])
        grouped.setdefault(solver, {})
        for metric in METRICS:
            value = row.get("metrics", {}).get(metric)
            if value is None:
                continue
            grouped[solver].setdefault(metric, []).append(float(value))

    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for solver, metrics in grouped.items():
        summary[solver] = {}
        for metric, values in metrics.items():
            summary[solver][metric] = (mean(values), pstdev(values) if len(values) > 1 else 0.0)
    return summary


def _write_summary_csv(path: Path, summary: dict[str, dict[str, tuple[float, float]]]) -> None:
    rows = []
    for solver, metrics in summary.items():
        row: dict[str, Any] = {"solver": solver}
        for metric in METRICS:
            score = metrics.get(metric)
            row[f"{metric}_mean"] = score[0] if score is not None else None
            row[f"{metric}_std"] = score[1] if score is not None else None
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(
    path: Path,
    *,
    summary: dict[str, dict[str, tuple[float, float]]],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Makkuva Checkerboard Comparison",
        "",
        "| Solver | pushforward_w2 | mmd | w2_estimate | Representative plot |",
        "| --- | --- | --- | --- | --- |",
    ]
    representative = {str(row["solver_id"]): row for row in rows}
    for solver, metrics in summary.items():
        row = representative[solver]
        plot_path = row.get("metrics", {}).get("visualization_path", "")
        formatted = []
        for metric in METRICS:
            score = metrics.get(metric)
            formatted.append("n/a" if score is None else f"{score[0]:.6f} +/- {score[1]:.6f}")
        plot_cell = f"`{plot_path}`" if plot_path else ""
        lines.append(f"| {solver} | {formatted[0]} | {formatted[1]} | {formatted[2]} | {plot_cell} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Makkuva checkerboard comparison")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--solvers", default=",".join(DEFAULT_SOLVERS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--output-root", default="outputs/makkuva_case1_compare")
    parser.add_argument("--eval-items", type=int, default=2048)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for solver in _parse_csv_strings(args.solvers):
        for seed in _parse_csv_ints(args.seeds):
            run_output = output_root / solver / f"seed_{seed}"
            result = _run_one(
                python_exe=str(args.python),
                solver=solver,
                seed=seed,
                output_dir=run_output,
                eval_items=int(args.eval_items),
                rerun=bool(args.rerun),
                overrides=list(args.override),
            )
            rows.append(result)

    summary = _aggregate(rows)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_csv(output_root / "summary.csv", summary)
    _write_summary_markdown(output_root / "summary.md", summary=summary, rows=rows)


if __name__ == "__main__":
    main()
