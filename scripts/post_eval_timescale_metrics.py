"""Post-evaluate timescale grid metrics logs without rerunning checkpoints.

The timescale sweeps persist training and validation scalars in each run's
``metrics.jsonl``. This script reads those logs, joins them with the grid
metadata in ``results.json`` when available, and produces solver-specific plots
for the final logged validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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

from src.evaluation.plotting import apply_publication_axes, save_png_and_pdf  # noqa: E402


DEFAULT_OUTPUT_ROOTS = (
    "outputs/timescale_grid_synthetic_harder_mps_5seed_otp",
    "outputs/timescale_grid_synthetic_harder_mps_5seed_monge_map",
    "outputs/timescale_grid_synthetic_harder_mps_5seed_otm",
    "outputs/timescale_grid_synthetic_harder_mps_5seed_maxcorr",
)

METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "map": (
        "val/map_l2",
        "val/pushforward_w2",
        "val/d_kr",
        "val/mmd",
        "val/transport_cos_fwd",
    ),
    "potential": (
        "val/target_potential_centered_mse",
        "val/target_potential_gradient_mse",
        "val/potential_mean",
    ),
    "flatness": (
        "val/flatness_std_F",
        "val/flatness_range_F",
        "val/flatness_mean_abs_gap_to_current",
    ),
}

HIGHER_IS_BETTER_SUFFIXES = ("cos", "cos_fwd", "precision", "recall")

_GRID_RE = re.compile(r"^k(?P<k>[^_]+)_r(?P<ratio>.+)$")
_SEED_RE = re.compile(r"^seed(?P<seed>-?\d+)$")


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _slug_to_float(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def _metric_slug(metric: str) -> str:
    return (
        metric.replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "p")
        .strip("_")
    )


def _metric_label(metric: str) -> str:
    labels = {
        "val/map_l2": "Map L2",
        "val/pushforward_w2": "Pushforward W2",
        "val/d_kr": "KR distance",
        "val/mmd": "MMD",
        "val/transport_cos_fwd": "Transport cosine",
        "val/target_potential_centered_mse": "Potential MSE",
        "val/target_potential_gradient_mse": "Potential gradient MSE",
        "val/potential_mean": "Potential mean",
        "val/flatness_std_F": "Flatness std",
        "val/flatness_range_F": "Flatness range",
        "val/flatness_mean_abs_gap_to_current": "Flatness gap",
    }
    if metric in labels:
        return labels[metric]
    return metric.removeprefix("val/").replace("_", " ")


def _summary_key(metric: str, suffix: str) -> str:
    return f"{_metric_slug(metric)}_{suffix}"


def _metric_direction(metric: str) -> str:
    metric_tail = metric.rsplit("/", maxsplit=1)[-1].lower()
    if any(metric_tail.endswith(suffix) for suffix in HIGHER_IS_BETTER_SUFFIXES):
        return "max"
    return "min"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _infer_path_metadata(metrics_path: Path) -> dict[str, Any]:
    run_dir = metrics_path.parent
    metadata: dict[str, Any] = {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
    }

    seed_match = _SEED_RE.match(run_dir.name)
    grid_match = _GRID_RE.match(run_dir.parent.name)
    if seed_match is not None:
        metadata["seed"] = int(seed_match.group("seed"))
    if grid_match is not None:
        metadata["k"] = int(round(_slug_to_float(grid_match.group("k"))))
        metadata["ratio"] = _slug_to_float(grid_match.group("ratio"))
        metadata["solver_id"] = run_dir.parent.parent.name
    return metadata


def _run_metadata(metrics_path: Path) -> dict[str, Any]:
    metadata = _infer_path_metadata(metrics_path)
    result = _read_json(metrics_path.parent / "results.json")
    if result is None:
        return metadata

    result_metrics = result.get("metrics", {})
    if not isinstance(result_metrics, dict):
        result_metrics = {}

    metadata.update(
        {
            "solver_id": result.get("solver_id", metadata.get("solver_id")),
            "dataset_id": result.get("dataset_id"),
            "seed": result.get("seed", metadata.get("seed")),
            "max_steps": result.get("max_steps"),
            "batch_size": result.get("batch_size"),
            "k": result_metrics.get(
                "configured_k",
                result_metrics.get("configured_transport_steps", metadata.get("k")),
            ),
            "ratio": result_metrics.get("configured_ratio", metadata.get("ratio")),
            "transport_steps": result_metrics.get(
                "configured_transport_steps",
                result_metrics.get("solver_transport_steps"),
            ),
            "potential_steps": result_metrics.get(
                "configured_potential_steps",
                result_metrics.get("solver_potential_steps"),
            ),
            "transport_lr": result_metrics.get(
                "configured_transport_lr",
                result_metrics.get("solver_transport_lr"),
            ),
            "potential_lr": result_metrics.get(
                "configured_potential_lr",
                result_metrics.get("solver_potential_lr"),
            ),
        }
    )
    return metadata


def _discover_metrics_files(output_roots: list[Path], solvers: set[str] | None) -> list[Path]:
    paths: list[Path] = []
    for output_root in output_roots:
        for path in output_root.rglob("metrics.jsonl"):
            if solvers is not None:
                metadata = _run_metadata(path)
                if str(metadata.get("solver_id", "")) not in solvers:
                    continue
            paths.append(path)
    return sorted(set(paths))


def _select_metric_names(metric_groups: list[str], explicit_metrics: list[str]) -> list[str]:
    selected: list[str] = []
    for group in metric_groups:
        if group == "all":
            for metrics in METRIC_GROUPS.values():
                selected.extend(metrics)
            continue
        if group not in METRIC_GROUPS:
            raise ValueError(
                f"Unknown metric group '{group}'. Available groups: "
                f"{', '.join(sorted([*METRIC_GROUPS, 'all']))}"
            )
        selected.extend(METRIC_GROUPS[group])
    selected.extend(explicit_metrics)
    return list(dict.fromkeys(selected))


def collect_final_metric_rows(
    metrics_files: list[Path],
    metric_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect final logged validation metrics and per-run best metric rows."""
    final_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for metrics_path in metrics_files:
        metadata = _run_metadata(metrics_path)
        log_rows = _read_jsonl(metrics_path)
        candidate_rows: list[tuple[int, int, dict[str, float]]] = []
        metric_values: dict[str, list[tuple[int, float]]] = {metric: [] for metric in metric_names}

        for row_index, row in enumerate(log_rows):
            raw_step = _to_float(row.get("step"))
            step = int(raw_step) if raw_step is not None else row_index
            values: dict[str, float] = {}
            for metric in metric_names:
                if metric not in row:
                    continue
                value = _to_float(row[metric])
                if value is None:
                    continue
                values[metric] = value
                metric_values[metric].append((step, value))
            if values:
                candidate_rows.append((step, row_index, values))

        if candidate_rows:
            final_step, _, final_values = max(candidate_rows, key=lambda item: (item[0], item[1]))
            final_row = dict(metadata)
            final_row["final_step"] = final_step
            final_row.update(final_values)
            final_rows.append(final_row)

        for metric, values in metric_values.items():
            if not values:
                continue
            direction = _metric_direction(metric)
            selector = max if direction == "max" else min
            best_step, best_value = selector(values, key=lambda item: item[1])
            best_row = dict(metadata)
            best_row.update(
                {
                    "metric": metric,
                    "direction": direction,
                    "best_step": best_step,
                    "best_value": best_value,
                }
            )
            best_rows.append(best_row)

    return final_rows, best_rows


def aggregate_final_rows(rows: list[dict[str, Any]], metric_names: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        solver = str(row.get("solver_id", "unknown"))
        k_value = int(row["k"])
        ratio = float(row["ratio"])
        grouped.setdefault((solver, k_value, ratio), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (solver, k_value, ratio), group in sorted(grouped.items()):
        output: dict[str, Any] = {
            "solver_id": solver,
            "k": k_value,
            "ratio": ratio,
            "num_seeds": len(group),
        }
        dataset_ids = sorted({str(row.get("dataset_id")) for row in group if row.get("dataset_id")})
        if dataset_ids:
            output["dataset_id"] = ",".join(dataset_ids)
        for metric in metric_names:
            values = [float(row[metric]) for row in group if _to_float(row.get(metric)) is not None]
            if not values:
                continue
            metric_mean, metric_std = _mean_std(values)
            output[_summary_key(metric, "mean")] = metric_mean
            output[_summary_key(metric, "std")] = metric_std
        aggregate_rows.append(output)
    return aggregate_rows


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


def _metric_values_by_grid(
    rows: list[dict[str, Any]],
    *,
    solver: str,
    metric: str,
) -> dict[tuple[int, float], list[float]]:
    grouped: dict[tuple[int, float], list[float]] = {}
    for row in rows:
        if str(row.get("solver_id")) != solver:
            continue
        value = _to_float(row.get(metric))
        if value is None:
            continue
        grouped.setdefault((int(row["k"]), float(row["ratio"])), []).append(value)
    return grouped


def _format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if abs_value < 0.01 or abs_value >= 1000:
        return f"{value:.1e}"
    return f"{value:.3g}"


def _format_mean_std_cell(mean_value: float, std_value: float) -> str:
    return f"{_format_number(mean_value)}\n+/- {_format_number(std_value)}"


def _plot_heatmap(
    rows: list[dict[str, Any]],
    *,
    solver: str,
    metric: str,
    output_dir: Path,
) -> list[dict[str, str]]:
    grouped = _metric_values_by_grid(rows, solver=solver, metric=metric)
    if not grouped:
        return []

    k_values = sorted({key[0] for key in grouped})
    ratios = sorted({key[1] for key in grouped})
    mean_matrix: list[list[float]] = []
    std_matrix: list[list[float]] = []
    for k_value in k_values:
        mean_line: list[float] = []
        std_line: list[float] = []
        for ratio in ratios:
            values = grouped.get((k_value, ratio), [])
            if values:
                metric_mean, metric_std = _mean_std(values)
                mean_line.append(metric_mean)
                std_line.append(metric_std)
            else:
                mean_line.append(math.nan)
                std_line.append(math.nan)
        mean_matrix.append(mean_line)
        std_matrix.append(std_line)

    if not any(math.isfinite(value) for line in mean_matrix for value in line):
        return []

    figure_width = max(5.0, 0.55 * len(ratios) + 2.5)
    figure_height = max(3.8, 0.42 * len(k_values) + 1.8)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    cmap = "viridis" if _metric_direction(metric) == "max" else "viridis_r"
    image = axis.imshow(mean_matrix, aspect="auto", cmap=cmap)
    figure.colorbar(image, ax=axis, label=f"{_metric_label(metric)} mean")

    axis.set_xticks(range(len(ratios)))
    axis.set_xticklabels([f"{ratio:g}" for ratio in ratios], rotation=45, ha="right")
    axis.set_yticks(range(len(k_values)))
    axis.set_yticklabels([str(k_value) for k_value in k_values])
    axis.set_xlabel("LR ratio")
    axis.set_ylabel("Transport steps K")

    for row_index, line in enumerate(mean_matrix):
        for column_index, mean_value in enumerate(line):
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

    output_stem = output_dir / solver / f"{_metric_slug(metric)}_heatmap"
    png_path, pdf_path = save_png_and_pdf(figure, output_stem)
    plt.close(figure)
    return [
        {
            "solver_id": solver,
            "metric": metric,
            "kind": "heatmap",
            "png_path": str(png_path),
            "pdf_path": str(pdf_path),
        }
    ]


def _plot_ratio_trends(
    rows: list[dict[str, Any]],
    *,
    solver: str,
    metric: str,
    output_dir: Path,
) -> list[dict[str, str]]:
    grouped = _metric_values_by_grid(rows, solver=solver, metric=metric)
    if not grouped:
        return []

    k_values = sorted({key[0] for key in grouped})
    ratios = sorted({key[1] for key in grouped})
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    plotted = False

    for k_value in k_values:
        x_values: list[float] = []
        y_values: list[float] = []
        lower_values: list[float] = []
        upper_values: list[float] = []
        for ratio in ratios:
            values = grouped.get((k_value, ratio), [])
            if not values:
                continue
            metric_mean, metric_std = _mean_std(values)
            x_values.append(ratio)
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
            label=f"K = {k_value}",
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

    if ratios and all(ratio > 0 for ratio in ratios):
        axis.set_xscale("log")
    apply_publication_axes(
        axis,
        xlabel="LR ratio",
        ylabel=_metric_label(metric),
    )
    legend_columns = min(max(len(k_values), 1), 5)
    axis.legend(
        fontsize=8,
        ncols=legend_columns,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.45,
        labelspacing=0.3,
        columnspacing=1.0,
    )

    output_stem = output_dir / solver / f"{_metric_slug(metric)}_vs_ratio"
    png_path, pdf_path = save_png_and_pdf(figure, output_stem)
    plt.close(figure)
    return [
        {
            "solver_id": solver,
            "metric": metric,
            "kind": "ratio_trend",
            "png_path": str(png_path),
            "pdf_path": str(pdf_path),
        }
    ]


def write_plots(
    rows: list[dict[str, Any]],
    *,
    metric_names: list[str],
    output_dir: Path,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    solvers = sorted({str(row.get("solver_id", "unknown")) for row in rows})
    for solver in solvers:
        for metric in metric_names:
            manifest.extend(_plot_heatmap(rows, solver=solver, metric=metric, output_dir=output_dir))
            manifest.extend(
                _plot_ratio_trends(rows, solver=solver, metric=metric, output_dir=output_dir)
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read timescale sweep metrics.jsonl logs and plot grid effects without "
            "rerunning training or checkpoint evaluation."
        )
    )
    parser.add_argument("--output-roots", default=",".join(DEFAULT_OUTPUT_ROOTS))
    parser.add_argument("--solvers", default="", help="Optional comma-separated solver filter.")
    parser.add_argument(
        "--metric-groups",
        default="map,potential,flatness",
        help="Comma-separated groups: map, potential, flatness, or all.",
    )
    parser.add_argument("--metrics", default="", help="Optional comma-separated extra metric keys.")
    parser.add_argument(
        "--plot-root",
        default="outputs/timescale_grid_synthetic_harder_mps_5seed_post_eval",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_roots = [_resolve_path(value) for value in _parse_csv_strings(str(args.output_roots))]
    solvers = set(_parse_csv_strings(str(args.solvers))) if str(args.solvers).strip() else None
    metric_names = _select_metric_names(
        _parse_csv_strings(str(args.metric_groups)),
        _parse_csv_strings(str(args.metrics)),
    )
    plot_root = _resolve_path(str(args.plot_root))

    metrics_files = _discover_metrics_files(output_roots, solvers=solvers)
    print(f"Discovered {len(metrics_files)} metrics files.", flush=True)
    if args.dry_run:
        for path in metrics_files:
            print(path)
        return
    if not metrics_files:
        raise FileNotFoundError("No metrics.jsonl files found under the requested roots.")

    final_rows, best_rows = collect_final_metric_rows(metrics_files, metric_names)
    if not final_rows:
        raise ValueError("No requested metrics were found in the discovered metrics.jsonl files.")

    aggregate_rows = aggregate_final_rows(final_rows, metric_names)
    _write_csv(
        plot_root / "final_metric_rows.csv",
        final_rows,
        prefix=[
            "solver_id",
            "dataset_id",
            "k",
            "ratio",
            "seed",
            "final_step",
            "run_dir",
            "metrics_path",
        ],
    )
    _write_csv(
        plot_root / "grid_metric_summary.csv",
        aggregate_rows,
        prefix=["solver_id", "dataset_id", "k", "ratio", "num_seeds"],
    )
    _write_csv(
        plot_root / "best_metric_rows.csv",
        best_rows,
        prefix=[
            "solver_id",
            "dataset_id",
            "k",
            "ratio",
            "seed",
            "metric",
            "direction",
            "best_step",
            "best_value",
            "run_dir",
        ],
    )
    manifest = write_plots(final_rows, metric_names=metric_names, output_dir=plot_root / "plots")
    (plot_root / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    solver_counts: dict[str, int] = {}
    for row in final_rows:
        solver_counts[str(row.get("solver_id", "unknown"))] = (
            solver_counts.get(str(row.get("solver_id", "unknown")), 0) + 1
        )
    print(
        json.dumps(
            {
                "plot_root": str(plot_root),
                "runs": len(final_rows),
                "aggregate_rows": len(aggregate_rows),
                "plots": len(manifest),
                "runs_by_solver": solver_counts,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
