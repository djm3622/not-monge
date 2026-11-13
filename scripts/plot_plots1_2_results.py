"""Plot saved ``outputs/plots1_2`` data in the post-eval figure style."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "notmonge_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.post_eval_timescale_metrics import (  # noqa: E402
    _format_mean_std_cell,
    _metric_slug,
    _to_float,
)
from src.evaluation.plotting import apply_publication_axes, save_png_and_pdf  # noqa: E402


DEFAULT_METRICS = ("map_error", "grad_error")
SUMMARY_AXIS_LABEL_SIZE = 20
SUMMARY_TICK_SIZE = 17
SUMMARY_LEGEND_SIZE = 19
METRIC_LABELS = {
    "map_error": "Map error",
    "grad_error": "Potential gradient error",
    "F": "Flatness objective",
    "W2_self": "Self W2",
    "W1": "W1",
    "psi_error": "Potential error",
}


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def _apply_summary_axes(axis: Any, *, xlabel: str, ylabel: str) -> None:
    apply_publication_axes(axis, xlabel=xlabel, ylabel=ylabel)
    axis.tick_params(axis="both", labelsize=SUMMARY_TICK_SIZE, length=6, width=1.1)
    axis.xaxis.label.set_size(SUMMARY_AXIS_LABEL_SIZE)
    axis.yaxis.label.set_size(SUMMARY_AXIS_LABEL_SIZE)
    axis.xaxis.labelpad = 8
    axis.yaxis.labelpad = 4
    axis.xaxis.get_offset_text().set_fontsize(SUMMARY_TICK_SIZE)
    axis.yaxis.get_offset_text().set_fontsize(SUMMARY_TICK_SIZE)


def _mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def _ordered_fieldnames(rows: list[dict[str, Any]], prefix: list[str]) -> list[str]:
    seen = set(prefix)
    rest = sorted({key for row in rows for key in row if key not in seen})
    return [key for key in prefix if any(key in row for row in rows)] + rest


def _write_csv(path: Path, rows: list[dict[str, Any]], *, prefix: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _ordered_fieldnames(rows, prefix) if rows else prefix
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_not_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows = json.loads((input_dir / "not_results.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list rows in {input_dir / 'not_results.json'}")
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        lr_t = float(row["lr_t"])
        lr_psi = float(row["lr_psi"])
        output.append(
            {
                "solver_id": "not",
                "seed": int(row["seed"]),
                "k": int(row["inner_steps"]),
                "inner_steps": int(row["inner_steps"]),
                "ratio": lr_psi / lr_t,
                "lr_t": lr_t,
                "lr_psi": lr_psi,
                **{
                    key: float(value)
                    for key, value in row.items()
                    if key not in {"seed", "inner_steps", "lr_t", "lr_psi"}
                    and _to_float(value) is not None
                },
            }
        )
    return output


def _aggregate_rows(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["inner_steps"]), float(row["ratio"])), []).append(row)

    output: list[dict[str, Any]] = []
    for (inner_steps, ratio), group in sorted(grouped.items()):
        record: dict[str, Any] = {
            "inner_steps": inner_steps,
            "ratio": ratio,
            "num_seeds": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if _to_float(row.get(metric)) is not None]
            if not values:
                continue
            metric_mean, metric_std = _mean_std(values)
            record[f"{metric}_mean"] = metric_mean
            record[f"{metric}_std"] = metric_std
        output.append(record)
    return output


def _values_by_grid(
    rows: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[tuple[int, float], list[float]]:
    grouped: dict[tuple[int, float], list[float]] = {}
    for row in rows:
        value = _to_float(row.get(metric))
        if value is None:
            continue
        grouped.setdefault((int(row["inner_steps"]), float(row["ratio"])), []).append(value)
    return grouped


def _plot_heatmap(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    output_dir: Path,
) -> list[dict[str, str]]:
    grouped = _values_by_grid(rows, metric=metric)
    if not grouped:
        return []

    inner_steps = sorted({key[0] for key in grouped})
    ratios = sorted({key[1] for key in grouped})
    mean_matrix: list[list[float]] = []
    std_matrix: list[list[float]] = []
    for inner in inner_steps:
        mean_row: list[float] = []
        std_row: list[float] = []
        for ratio in ratios:
            values = grouped.get((inner, ratio), [])
            if values:
                metric_mean, metric_std = _mean_std(values)
                mean_row.append(metric_mean)
                std_row.append(metric_std)
            else:
                mean_row.append(math.nan)
                std_row.append(math.nan)
        mean_matrix.append(mean_row)
        std_matrix.append(std_row)

    figure_width = max(5.0, 0.55 * len(ratios) + 2.5)
    figure_height = max(3.8, 0.42 * len(inner_steps) + 1.8)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    image = axis.imshow(mean_matrix, aspect="auto", cmap="viridis_r")
    figure.colorbar(image, ax=axis, label=f"{_metric_label(metric)} mean")
    axis.set_xticks(range(len(ratios)))
    axis.set_xticklabels([f"{ratio:g}" for ratio in ratios], rotation=45, ha="right")
    axis.set_yticks(range(len(inner_steps)))
    axis.set_yticklabels([str(inner) for inner in inner_steps])
    axis.set_xlabel("LR ratio")
    axis.set_ylabel("Inner steps")

    for row_index, mean_row in enumerate(mean_matrix):
        for column_index, mean_value in enumerate(mean_row):
            std_value = std_matrix[row_index][column_index]
            if math.isfinite(mean_value) and math.isfinite(std_value):
                axis.text(
                    column_index,
                    row_index,
                    _format_mean_std_cell(mean_value, std_value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )

    output_stem = output_dir / "not" / f"{_metric_slug(metric)}_heatmap"
    png_path, pdf_path = save_png_and_pdf(figure, output_stem)
    plt.close(figure)
    return [
        {
            "metric": metric,
            "kind": "heatmap",
            "png_path": str(png_path),
            "pdf_path": str(pdf_path),
        }
    ]


def _plot_vs_inner_steps(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    output_dir: Path,
) -> list[dict[str, str]]:
    grouped = _values_by_grid(rows, metric=metric)
    if not grouped:
        return []

    inner_steps = sorted({key[0] for key in grouped})
    ratios = sorted({key[1] for key in grouped})
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    plotted = False
    for ratio in ratios:
        x_values: list[int] = []
        y_values: list[float] = []
        lower_values: list[float] = []
        upper_values: list[float] = []
        for inner in inner_steps:
            values = grouped.get((inner, ratio), [])
            if not values:
                continue
            metric_mean, metric_std = _mean_std(values)
            x_values.append(inner)
            y_values.append(metric_mean)
            lower_values.append(metric_mean - metric_std)
            upper_values.append(metric_mean + metric_std)
        if not x_values:
            continue
        (line,) = axis.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
            markersize=4,
            markeredgewidth=0.8,
            label=rf"$\eta_\psi / \eta_t = {ratio:g}$",
        )
        axis.fill_between(
            x_values,
            lower_values,
            upper_values,
            color=line.get_color(),
            alpha=0.18,
            linewidth=0,
        )
        plotted = True

    if not plotted:
        plt.close(figure)
        return []

    apply_publication_axes(axis, xlabel="Inner steps", ylabel=_metric_label(metric))
    axis.legend(
        fontsize=8,
        ncols=min(max(len(ratios), 1), 3),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.45,
        labelspacing=0.3,
        columnspacing=1.0,
    )

    output_stem = output_dir / "not" / f"{_metric_slug(metric)}_vs_inner_steps"
    png_path, pdf_path = save_png_and_pdf(figure, output_stem)
    plt.close(figure)
    return [
        {
            "metric": metric,
            "kind": "inner_step_trend",
            "png_path": str(png_path),
            "pdf_path": str(pdf_path),
        }
    ]


def _plot_two_panel(
    rows: list[dict[str, Any]],
    *,
    metrics: list[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, len(metrics), figsize=(8.8 * len(metrics), 4.8), squeeze=False)
    legend_entries: dict[str, Any] = {}
    for axis, metric in zip(axes[0], metrics):
        grouped = _values_by_grid(rows, metric=metric)
        inner_steps = sorted({key[0] for key in grouped})
        ratios = sorted({key[1] for key in grouped})
        for ratio in ratios:
            x_values: list[int] = []
            y_values: list[float] = []
            lower_values: list[float] = []
            upper_values: list[float] = []
            for inner in inner_steps:
                values = grouped.get((inner, ratio), [])
                if not values:
                    continue
                metric_mean, metric_std = _mean_std(values)
                x_values.append(inner)
                y_values.append(metric_mean)
                lower_values.append(metric_mean - metric_std)
                upper_values.append(metric_mean + metric_std)
            if not x_values:
                continue
            (line,) = axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=3.0,
                markersize=7.5,
                markeredgewidth=1.1,
                label=rf"$\eta_\psi / \eta_t = {ratio:g}$",
            )
            axis.fill_between(
                x_values,
                lower_values,
                upper_values,
                color=line.get_color(),
                alpha=0.18,
                linewidth=0,
            )
            legend_entries.setdefault(line.get_label(), line)
        _apply_summary_axes(axis, xlabel="Inner steps", ylabel=_metric_label(metric))
    if legend_entries:
        figure.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            fontsize=SUMMARY_LEGEND_SIZE,
            ncols=min(max(len(legend_entries), 1), 3),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.89),
            frameon=False,
            borderaxespad=0.0,
            handlelength=2.0,
            handletextpad=0.5,
            labelspacing=0.25,
            columnspacing=1.25,
            markerscale=1.1,
        )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.2, top=0.84, wspace=0.18)
    png_path, pdf_path = save_png_and_pdf(figure, output_dir / "not_results_two_panel")
    plt.close(figure)
    return png_path, pdf_path


def _load_series(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(float)
    return np.asarray(data, dtype=float)


def _plot_saved_point_trajectory(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    series = [
        ("T_errors.pt", "Transport error"),
        ("grad_errors.pt", "Potential gradient error"),
        ("F_vals.pt", "Flatness objective"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(24.0, 4.6), squeeze=False)
    for axis, (filename, label) in zip(axes[0], series):
        values = _load_series(input_dir / filename)
        steps = np.arange(values.shape[0]) * 10
        axis.plot(steps, values, linewidth=3.0)
        if filename == "F_vals.pt":
            axis.axhline(
                2.125,
                color="#d62728",
                linestyle="--",
                linewidth=2.2,
                label=r"$W_2^2$",
            )
            axis.legend(
                fontsize=SUMMARY_LEGEND_SIZE,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                frameon=False,
                borderaxespad=0.0,
                handlelength=2.0,
                handletextpad=0.5,
            )
        _apply_summary_axes(axis, xlabel="Step", ylabel=label)
    figure.subplots_adjust(left=0.06, right=0.995, bottom=0.2, top=0.86, wspace=0.38)
    png_path, pdf_path = save_png_and_pdf(figure, output_dir / "saved_points_trajectory")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/plots1_2")
    parser.add_argument("--output-dir", default="outputs/plots1_2/formatted_plots")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args()

    input_dir = _resolve_path(str(args.input_dir))
    output_dir = _resolve_path(str(args.output_dir))
    metrics = _parse_csv_strings(str(args.metrics))

    rows = _load_not_rows(input_dir)
    aggregate_rows = _aggregate_rows(rows, metrics)
    _write_csv(
        output_dir / "not_result_rows.csv",
        rows,
        prefix=["solver_id", "seed", "inner_steps", "ratio", "lr_t", "lr_psi"],
    )
    _write_csv(
        output_dir / "not_result_summary.csv",
        aggregate_rows,
        prefix=["inner_steps", "ratio", "num_seeds"],
    )

    manifest: list[dict[str, str]] = []
    for metric in metrics:
        manifest.extend(_plot_heatmap(rows, metric=metric, output_dir=output_dir / "plots"))
        manifest.extend(_plot_vs_inner_steps(rows, metric=metric, output_dir=output_dir / "plots"))
    two_panel_png, two_panel_pdf = _plot_two_panel(rows, metrics=metrics[:2], output_dir=output_dir)
    trajectory_png, trajectory_pdf = _plot_saved_point_trajectory(input_dir, output_dir)
    manifest.extend(
        [
            {
                "kind": "two_panel",
                "png_path": str(two_panel_png),
                "pdf_path": str(two_panel_pdf),
            },
            {
                "kind": "saved_point_trajectory",
                "png_path": str(trajectory_png),
                "pdf_path": str(trajectory_pdf),
            },
        ]
    )
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": len(rows),
                "aggregate_rows": len(aggregate_rows),
                "plots": len(manifest),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
