"""Run and aggregate paper-style case study 1 comparisons."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
from PIL import Image, ImageDraw, ImageFont

DEFAULT_SEEDS = [1234, 2024, 3407, 4444, 5555, 6666, 7777, 8888, 9999, 11111]

SOLVER_SPECS: dict[str, dict[str, object]] = {
    "gaussian": {
        "max_steps": 1,
        "batch_size": 1024,
        "steps_per_epoch": 1,
        "extra_overrides": [],
    },
    "mm": {
        "max_steps": 512,
        "batch_size": 1024,
        "steps_per_epoch": 128,
        "extra_overrides": [
            "solver.forward_lr=3e-4",
            "solver.inverse_lr=3e-4",
            "solver.inner_steps=5",
        ],
    },
    "mmv2": {
        "max_steps": 512,
        "batch_size": 1024,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
    "tw2": {
        "max_steps": 512,
        "batch_size": 1024,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
    "mm_b": {
        "max_steps": 512,
        "batch_size": 1024,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
    "qc": {
        "max_steps": 256,
        "batch_size": 64,
        "steps_per_epoch": 128,
        "extra_overrides": [],
    },
}

METRICS = ["map_l2", "l2_uvp", "transport_cos", "pushforward_w2", "mmd", "gradient_error"]
CHART_COLORS = ["#215E8A", "#2F8A5B", "#C66531", "#8B7B3A", "#7A4E8A", "#8A2F5D"]
BACKGROUND = (248, 246, 240)
PANEL_BG = (255, 255, 255)
BORDER = (210, 205, 196)
TEXT = (28, 31, 36)
GRID = (232, 228, 220)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_one(
    python_exe: str,
    solver: str,
    seed: int,
    output_dir: Path,
    cache_version: str,
    eval_items: int,
    spec: dict[str, object],
    rerun: bool,
) -> dict:
    result_path = output_dir / "results.json"
    if result_path.exists() and not rerun:
        return json.loads(result_path.read_text())

    command = [
        python_exe,
        "scripts/train_baseline.py",
        f"solver={solver}",
        "dataset=paper_mix3to10",
        "training.device=cpu",
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
    return json.loads(result_path.read_text())


def _aggregate(rows: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        solver = row["solver_id"]
        grouped.setdefault(solver, {})
        for metric in METRICS:
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


def _write_markdown(path: Path, summary: dict[str, dict[str, tuple[float, float]]], specs: dict[str, dict[str, object]]) -> None:
    lines = [
        "# Paper Case Study 1 Full Comparison",
        "",
        "| Solver | Batch | Steps | map_l2 | l2_uvp | transport_cos | pushforward_w2 | mmd | gradient_error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for solver in specs:
        metrics = summary.get(solver, {})

        def fmt(metric: str) -> str:
            value, spread = metrics.get(metric, (math.nan, math.nan))
            return f"{value:.4f} ± {spread:.4f}"

        lines.append(
            f"| {solver} | {int(specs[solver]['batch_size'])} | {int(specs[solver]['max_steps'])} | "
            f"{fmt('map_l2')} | {fmt('l2_uvp')} | {fmt('transport_cos')} | {fmt('pushforward_w2')} | "
            f"{fmt('mmd')} | {fmt('gradient_error')} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_plot(path: Path, summary: dict[str, dict[str, tuple[float, float]]], solvers: list[str]) -> None:
    plotted = [
        ("l2_uvp", "L2-UVP"),
        ("transport_cos", "Transport Cos"),
        ("map_l2", "Map L2"),
    ]
    canvas = Image.new("RGB", (1620, 560), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (metric, title) in enumerate(plotted):
        box = (30 + index * 530, 36, 520 + index * 530, 520)
        _draw_panel(draw, box, title, font)
        plot_box = _plot_box(box)
        means = [summary[solver][metric][0] for solver in solvers]
        stds = [summary[solver][metric][1] for solver in solvers]
        _draw_bars(draw, plot_box, solvers, means, stds, font)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _write_visual_panel(path: Path, rows: list[dict], solvers: list[str]) -> None:
    selected: list[tuple[str, Path]] = []
    for solver in solvers:
        solver_rows = [row for row in rows if row["solver_id"] == solver]
        solver_rows.sort(key=lambda row: float(row["metrics"]["map_l2"]))
        if not solver_rows:
            continue
        selected.append((solver, Path(str(solver_rows[0]["metrics"]["visualization_path"]))))

    columns = len(selected)
    if columns == 0:
        return
    tile_width = 360
    tile_height = 360
    margin = 24
    title_height = 22
    canvas = Image.new(
        "RGB",
        (margin + columns * (tile_width + margin), margin * 2 + title_height + tile_height),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (solver, image_path) in enumerate(selected):
        left = margin + index * (tile_width + margin)
        draw.text((left, margin), solver, fill=TEXT, font=font)
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((tile_width, tile_height))
        tile = Image.new("RGB", (tile_width, tile_height), PANEL_BG)
        offset = ((tile_width - image.width) // 2, (tile_height - image.height) // 2)
        tile.paste(image, offset)
        canvas.paste(tile, (left, margin + title_height))
        draw.rectangle((left, margin + title_height, left + tile_width, margin + title_height + tile_height), outline=BORDER, width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(box, radius=16, fill=PANEL_BG, outline=BORDER, width=2)
    draw.text((box[0] + 16, box[1] + 12), title, fill=TEXT, font=font)
    _draw_grid(draw, _plot_box(box))


def _plot_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return left + 18, top + 44, right - 18, bottom - 28


def _draw_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    for fraction in (0.25, 0.5, 0.75):
        y = top + int(height * fraction)
        draw.line((left, y, right, y), fill=GRID, width=1)
    draw.rectangle(box, outline=BORDER, width=1)


def _draw_bars(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    solvers: list[str],
    means: list[float],
    stds: list[float],
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    min_value = min(min((mean - std for mean, std in zip(means, stds)), default=0.0), 0.0)
    max_value = max(max((mean + std for mean, std in zip(means, stds)), default=1.0), 1.0e-6)
    scale = max(max_value - min_value, 1.0e-6)
    zero_y = bottom - int(((0.0 - min_value) / scale) * (height - 36))
    zero_y = max(top, min(bottom, zero_y))
    draw.line((left, zero_y, right, zero_y), fill=TEXT, width=1)
    bar_width = max(12, int(width / max(1, len(solvers) * 2)))
    gap = bar_width
    x = left + gap // 2
    for index, (solver, mean_value, std_value) in enumerate(zip(solvers, means, stds)):
        y_end = bottom - int(((mean_value - min_value) / scale) * (height - 36))
        y0 = min(zero_y, y_end)
        y1 = max(zero_y, y_end)
        color = _hex_to_rgb(CHART_COLORS[index % len(CHART_COLORS)])
        draw.rectangle((x, y0, x + bar_width, y1), fill=color)
        error_top = bottom - int((((mean_value + std_value) - min_value) / scale) * (height - 36))
        error_bottom = bottom - int((((mean_value - std_value) - min_value) / scale) * (height - 36))
        center = x + bar_width // 2
        draw.line((center, error_top, center, error_bottom), fill=TEXT, width=1)
        draw.line((center - 4, error_top, center + 4, error_top), fill=TEXT, width=1)
        draw.line((center - 4, error_bottom, center + 4, error_bottom), fill=TEXT, width=1)
        draw.text((x - 4, bottom + 6), solver, fill=TEXT, font=font)
        x += bar_width + gap


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--solvers", default="gaussian,mm,mmv2,tw2,mm_b,qc")
    parser.add_argument("--cache-version", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--eval-items", type=int, default=4096)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    seeds = _parse_csv_ints(args.seeds)
    solvers = _parse_csv_strings(args.solvers)
    for solver in solvers:
        if solver not in SOLVER_SPECS:
            raise ValueError(f"Unknown solver '{solver}'")

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
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
                    cache_version=args.cache_version,
                    eval_items=args.eval_items,
                    spec=spec,
                    rerun=bool(args.rerun),
                )
            )

    summary = _aggregate(rows)
    (output_root / "run_specs.json").write_text(json.dumps({solver: SOLVER_SPECS[solver] for solver in solvers}, indent=2), encoding="utf-8")
    (output_root / "seed_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(output_root / "seed_summary.md", summary, {solver: SOLVER_SPECS[solver] for solver in solvers})
    _write_summary_plot(output_root / "seed_summary.png", summary, solvers)
    _write_visual_panel(output_root / "visual_panel.png", rows, solvers)


if __name__ == "__main__":
    main()
