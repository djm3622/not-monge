"""Run and aggregate paper-style case study 1 sweeps."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_one(
    python_exe: str,
    solver: str,
    seed: int,
    output_dir: Path,
    cache_version: str,
    batch_size: int | None,
    steps_per_epoch: int | None,
    max_steps: int,
    eval_items: int,
    extra_overrides: list[str],
) -> dict:
    command = [
        python_exe,
        "scripts/train_baseline.py",
        f"solver={solver}",
        "dataset=paper_mix3to10",
        "training.device=cpu",
        "training.gradient_clip_norm=null",
        "visualization.enabled=true",
        f"training.seed={seed}",
        f"training.max_steps={max_steps}",
        "training.max_epochs=1000",
        f"+evaluation.max_items={eval_items}",
        f"dataset.cache_version={cache_version}",
        "model.input_dim=64",
        "model.output_dim=64",
        f"experiment.output_dir={output_dir.as_posix()}",
    ]
    if batch_size is not None:
        command.append(f"dataset.batch_size={batch_size}")
    if steps_per_epoch is not None:
        command.append(f"dataset.steps_per_epoch={steps_per_epoch}")
    command.extend(extra_overrides)
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads((output_dir / "results.json").read_text())


def _aggregate(rows: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics_of_interest = ["map_l2", "pushforward_w2", "mmd", "gradient_error", "l2_uvp", "transport_cos"]
    for row in rows:
        solver = row["solver_id"]
        for metric in metrics_of_interest:
            value = row["metrics"].get(metric)
            if value is not None:
                grouped[solver][metric].append(float(value))
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for solver, metrics in grouped.items():
        summary[solver] = {}
        for metric, values in metrics.items():
            summary[solver][metric] = (mean(values), pstdev(values) if len(values) > 1 else 0.0)
    return summary


def _write_markdown(path: Path, summary: dict[str, dict[str, tuple[float, float]]]) -> None:
    lines = [
        "# Paper Case Study 1 Seed Sweep",
        "",
        "| Solver | map_l2 | l2_uvp | transport_cos | pushforward_w2 | mmd |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for solver in sorted(summary):
        metrics = summary[solver]
        def fmt(metric: str) -> str:
            value, spread = metrics.get(metric, (math.nan, math.nan))
            return f"{value:.4f} ± {spread:.4f}"
        lines.append(
            f"| {solver} | {fmt('map_l2')} | {fmt('l2_uvp')} | {fmt('transport_cos')} | {fmt('pushforward_w2')} | {fmt('mmd')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, summary: dict[str, dict[str, tuple[float, float]]]) -> None:
    solvers = sorted(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics = [("l2_uvp", "L2-UVP"), ("transport_cos", "Transport Cos")]
    for axis, (metric, title) in zip(axes, metrics):
        means = [summary[solver][metric][0] for solver in solvers]
        stds = [summary[solver][metric][1] for solver in solvers]
        axis.bar(solvers, means, yerr=stds, color=["#295D8A", "#D2693C", "#3F8F64", "#8B7D3A"][: len(solvers)])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1234,2024,3407")
    parser.add_argument("--solvers", default="gaussian,tw2,mmv2,mm")
    parser.add_argument("--cache-version", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-items", type=int, default=4096)
    parser.add_argument("--extra-override", action="append", default=[])
    args = parser.parse_args()

    seeds = [int(seed) for seed in _parse_csv(args.seeds)]
    solvers = _parse_csv(args.solvers)
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for solver in solvers:
        for seed in seeds:
            run_dir = output_root / f"{solver}_seed{seed}"
            rows.append(
                _run_one(
                    python_exe=sys.executable,
                    solver=solver,
                    seed=seed,
                    output_dir=run_dir,
                    cache_version=args.cache_version,
                    batch_size=args.batch_size,
                    steps_per_epoch=args.steps_per_epoch,
                    max_steps=args.max_steps,
                    eval_items=args.eval_items,
                    extra_overrides=list(args.extra_override),
                )
            )

    summary = _aggregate(rows)
    (output_root / "seed_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(output_root / "seed_summary.md", summary)
    _write_plot(output_root / "seed_summary.png", summary)


if __name__ == "__main__":
    main()
