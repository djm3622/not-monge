#!/usr/bin/env python3
"""Plot scale-free potential errors for timescale sweeps without retraining.

Raw potential MSEs are not comparable across solvers because each reference has its own
scale. This divides the logged final errors by the reference variance on the same validation
pairs the training-time metric used. The gradient MSE is logged per coordinate, so its
denominator is per coordinate too. A value of 1 means no better than a constant predictor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import post_eval_timescale_metrics as post_eval  # noqa: E402
from scripts.timescale_grid_sweep import (  # noqa: E402
    DEFAULT_CACHE_VERSION,
    _build_grid_config,
    _solver_specs_for_dataset,
)
from src.benchmarking import (  # noqa: E402
    _collect_target_potential_metric_batch,
    _target_reference_potential_values_and_gradients,
    resolve_ot_dataset,
)
from src.solvers.registry import build_solver  # noqa: E402

VALUE_METRIC = "val/target_potential_centered_mse"
GRADIENT_METRIC = "val/target_potential_gradient_mse"
RELATIVE_VALUE_METRIC = "val/relative_potential_error"
RELATIVE_GRADIENT_METRIC = "val/relative_potential_gradient_error"
RELATIVE_METRICS = [RELATIVE_VALUE_METRIC, RELATIVE_GRADIENT_METRIC]
CONTEXT_METRICS = ["val/map_l2", "val/optimal_transport_cost"]
BATCH_CHECK_RTOL = 1.0e-3
# Ten times worse than a constant predictor; cells above this have diverged.
DIVERGED_THRESHOLD = 10.0


def _reference_variances(
    *,
    solver_name: str,
    dataset_name: str,
    seed: int,
    metric_items: int,
    bundles: dict[int, Any],
) -> dict[str, float]:
    """Return reference variances on the validation pairs used by the training-time metric."""
    config = _build_grid_config(
        solver_name=solver_name,
        dataset_name=dataset_name,
        seed=seed,
        output_dir=Path("unused"),
        base_spec=_solver_specs_for_dataset(dataset_name)[solver_name],
        cache_version=DEFAULT_CACHE_VERSION,
        device="cpu",
        eval_items=metric_items,
        k_value=1,
        ratio_value=1.0,
        transport_lr=5.0e-4,
        potential_steps=1,
        max_steps=1,
        batch_size=None,
        steps_per_epoch=None,
        disable_noise=True,
        visualize=False,
        save_epoch_checkpoints=False,
        extra_overrides=[],
        target_potential_metric_items=metric_items,
    )
    if seed not in bundles:
        bundles[seed] = resolve_ot_dataset(config)
    bundle = bundles[seed]
    _, val_loader, _ = bundle.make_dataloaders()
    batch = _collect_target_potential_metric_batch(
        val_loader,
        device=torch.device("cpu"),
        max_items=metric_items,
    )
    solver = build_solver(config["model"], config["solver"], config["training"])
    source = batch["source"]
    target = batch["ground_truth_map"]
    with torch.no_grad():
        values, gradients = _target_reference_potential_values_and_gradients(
            solver,
            bundle.ground_truth_potential,
            source=source,
            target=target,
        )
    return {
        "value_variance": float((values - values.mean()).pow(2).mean()),
        "gradient_variance": float((gradients - gradients.mean(dim=0)).pow(2).mean()),
        "optimal_transport_cost": float(0.5 * (source - target).pow(2).sum(dim=-1).mean()),
    }


def _add_relative_metrics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    metric_items: int,
) -> list[dict[str, Any]]:
    bundles: dict[int, Any] = {}
    variances: dict[tuple[str, int], dict[str, float]] = {}
    summary: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["solver_id"]), int(row["seed"]))
        if key not in variances:
            variances[key] = _reference_variances(
                solver_name=key[0],
                dataset_name=dataset_name,
                seed=key[1],
                metric_items=metric_items,
                bundles=bundles,
            )
            summary.append({"solver_id": key[0], "seed": key[1], **variances[key]})
        reference = variances[key]
        logged_cost = float(row["val/optimal_transport_cost"])
        if abs(logged_cost - reference["optimal_transport_cost"]) > BATCH_CHECK_RTOL * abs(logged_cost):
            raise ValueError(
                f"Validation batch mismatch for {key}: logged cost {logged_cost}, "
                f"recomputed {reference['optimal_transport_cost']}"
            )
        row[RELATIVE_VALUE_METRIC] = float(row[VALUE_METRIC]) / reference["value_variance"]
        row[RELATIVE_GRADIENT_METRIC] = float(row[GRADIENT_METRIC]) / reference["gradient_variance"]
    return summary


def _plot_capped_heatmap(
    rows: list[dict[str, Any]],
    *,
    solver: str,
    metric: str,
    output_dir: Path,
) -> list[dict[str, str]]:
    """Heatmap whose colour scale ignores diverged cells, which are hatched and keep their value labels."""
    grouped = post_eval._metric_values_by_grid(rows, solver=solver, metric=metric)
    if not grouped:
        return []
    k_values = sorted({key[0] for key in grouped})
    ratios = sorted({key[1] for key in grouped})
    means = np.full((len(k_values), len(ratios)), np.nan)
    stds = np.full_like(means, np.nan)
    for (k_value, ratio), values in grouped.items():
        row, column = k_values.index(k_value), ratios.index(ratio)
        means[row, column], stds[row, column] = post_eval._mean_std(values)

    finite = np.isfinite(means)
    diverged = finite & (means > DIVERGED_THRESHOLD)
    stable_values = means[finite & ~diverged]
    vmin = float(stable_values.min())
    cap = float(stable_values.max())
    use_log_scale = post_eval._should_use_log_scale(solver, stable_values.tolist())
    norm = (
        post_eval.mcolors.LogNorm(vmin=vmin, vmax=cap)
        if use_log_scale
        else post_eval.mcolors.Normalize(vmin=vmin, vmax=cap)
    )
    cmap = plt.get_cmap("viridis" if post_eval._metric_direction(metric) == "max" else "viridis_r").copy()
    cmap.set_over("#d9d9d9")

    figure_width = max(5.0, 0.55 * len(ratios) + 2.5)
    figure_height = max(3.8, 0.42 * len(k_values) + 1.8)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    image = axis.imshow(np.where(diverged, np.inf, means), aspect="auto", cmap=cmap, norm=norm)
    scale_note = " (log scale)" if use_log_scale else ""
    colorbar = figure.colorbar(
        image,
        ax=axis,
        extend="max" if diverged.any() else "neither",
        label=f"{post_eval._metric_label(metric)} mean{scale_note}",
    )
    if diverged.any():
        colorbar.ax.set_title(f"diverged\n(> {DIVERGED_THRESHOLD:g})", fontsize=7)

    axis.set_xticks(range(len(ratios)))
    axis.set_xticklabels([f"{ratio:g}" for ratio in ratios], rotation=45, ha="right")
    axis.set_yticks(range(len(k_values)))
    axis.set_yticklabels([str(k_value) for k_value in k_values])
    axis.set_xlabel("LR ratio")
    axis.set_ylabel("Transport steps K")

    for row, column in zip(*np.nonzero(finite)):
        label = post_eval._format_mean_std_cell(float(means[row, column]), float(stds[row, column]))
        if diverged[row, column]:
            axis.add_patch(
                plt.Rectangle(
                    (column - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    hatch="///",
                    edgecolor="#7f7f7f",
                    linewidth=0.0,
                )
            )
            label = f"diverged\n{label}"
            text_color = "black"
        else:
            red, green, blue, _ = cmap(norm(means[row, column]))
            text_color = "black" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.5 else "white"
        axis.text(column, row, label, ha="center", va="center", fontsize=7, color=text_color)

    output_stem = output_dir / solver / f"{post_eval._metric_slug(metric)}_heatmap"
    png_path, pdf_path = post_eval.save_png_and_pdf(figure, output_stem)
    plt.close(figure)
    return [
        {
            "solver_id": solver,
            "metric": metric,
            "kind": "heatmap",
            "scale": "log" if use_log_scale else "linear",
            "color_cap": str(cap),
            "diverged_cells": str(int(diverged.sum())),
            "png_path": str(png_path),
            "pdf_path": str(pdf_path),
        }
    ]


def _write_plots(rows: list[dict[str, Any]], *, output_dir: Path) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for solver in sorted({str(row["solver_id"]) for row in rows}):
        for metric in RELATIVE_METRICS:
            manifest.extend(_plot_capped_heatmap(rows, solver=solver, metric=metric, output_dir=output_dir))
            manifest.extend(post_eval._plot_ratio_trends(rows, solver=solver, metric=metric, output_dir=output_dir))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-roots", required=True)
    parser.add_argument("--plot-root", required=True)
    parser.add_argument("--dataset", default="synthetic_ot_harder")
    parser.add_argument("--exclude-ratios", default="")
    parser.add_argument("--metric-items", type=int, default=512)
    args = parser.parse_args()

    plot_root = post_eval._resolve_path(str(args.plot_root))
    if plot_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing plot root: {plot_root}")
    output_roots = [
        post_eval._resolve_path(value) for value in post_eval._parse_csv_strings(str(args.output_roots))
    ]
    discovered = post_eval._discover_metrics_files(output_roots, solvers=None)
    metrics_files = [path for path in discovered if (path.parent / "results.json").exists()]
    print(f"Using {len(metrics_files)} finished runs of {len(discovered)} discovered.", flush=True)

    metric_names = [VALUE_METRIC, GRADIENT_METRIC, *CONTEXT_METRICS]
    final_rows, _ = post_eval.collect_final_metric_rows(metrics_files, metric_names)
    excluded = post_eval._parse_ratio_filter(post_eval._parse_csv_strings(str(args.exclude_ratios)))
    final_rows = post_eval._exclude_ratios(final_rows, excluded)
    reference_rows = _add_relative_metrics(
        final_rows,
        dataset_name=str(args.dataset),
        metric_items=int(args.metric_items),
    )

    labels = {
        RELATIVE_VALUE_METRIC: "Relative potential error",
        RELATIVE_GRADIENT_METRIC: "Relative potential gradient error",
    }
    base_label = post_eval._metric_label
    post_eval._metric_label = lambda metric: labels.get(metric, base_label(metric))
    post_eval.LOG_SCALE_SOLVERS = {str(row["solver_id"]) for row in final_rows}

    summary_metrics = [*RELATIVE_METRICS, *metric_names]
    post_eval._write_csv(
        plot_root / "final_metric_rows.csv",
        final_rows,
        prefix=["solver_id", "k", "ratio", "seed", "final_step", "run_dir", "metrics_path"],
    )
    post_eval._write_csv(
        plot_root / "grid_metric_summary.csv",
        post_eval.aggregate_final_rows(final_rows, summary_metrics),
        prefix=["solver_id", "k", "ratio", "num_seeds"],
    )
    post_eval._write_csv(
        plot_root / "reference_variances.csv",
        reference_rows,
        prefix=["solver_id", "seed"],
    )
    manifest = _write_plots(final_rows, output_dir=plot_root / "plots")
    (plot_root / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"plot_root": str(plot_root), "runs": len(final_rows), "plots": len(manifest)}), flush=True)


if __name__ == "__main__":
    main()
