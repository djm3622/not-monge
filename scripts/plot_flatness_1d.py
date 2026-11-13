"""Plot the 1-D Gaussian flatness sweep from its saved ``results.npz``.

Reads the sweep output of ``scripts/flatness_1d_sweep.py`` (or the equivalent
notebook export) and draws, for each K, the transport error, potential
gradient error and objective gap |F - W_2^2| against potential steps, as the
mean +/- std over seeds for every eta_psi / eta_T. Nothing is retrained.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
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

from scripts.plot_plots1_2_results import SUMMARY_LEGEND_SIZE, _apply_summary_axes  # noqa: E402
from scripts.post_eval_timescale_metrics import _positive_floor  # noqa: E402
from src.evaluation.plotting import save_png_and_pdf  # noqa: E402


PANELS = (
    ("map_err", "Transport error"),
    ("grad_err", "Potential gradient error"),
    ("F_gap", r"Objective gap $|F - W_2^2|$"),
)
SHARED_RANGE_METRICS = ("map_err", "grad_err")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_results(results_dir: Path) -> tuple[dict[tuple[int, float], dict[str, np.ndarray]], int]:
    """Return ``{(K, ratio): {metric: seeds x steps}}`` and the perturbation step."""
    npz = np.load(results_dir / "results.npz")
    results: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    for key in npz.files:
        k_part, ratio_part, metric = key.split("|")
        results.setdefault((int(k_part[1:]), float(ratio_part[1:])), {})[metric] = npz[key]
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    if "config" in manifest:
        perturb_at = int(manifest["config"]["training"]["perturb_at"])
    else:  # notebook export
        perturb_at = int(manifest["args"]["perturb_at"])
    return results, perturb_at


def _mean_std_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    floor = _positive_floor(mean.tolist())
    return mean, np.maximum(mean - std, floor), mean + std


def plot_three_panel(
    results: dict[tuple[int, float], dict[str, np.ndarray]],
    *,
    inner_steps: int,
    perturb_at: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    ratios = sorted((ratio for k_value, ratio in results if k_value == inner_steps), reverse=True)
    figure, axes = plt.subplots(1, len(PANELS), figsize=(24.0, 4.6), squeeze=False)
    legend_entries: dict[str, Any] = {}
    shared_bounds: list[float] = []
    for axis, (metric, label) in zip(axes[0], PANELS):
        for ratio in ratios:
            series = results[(inner_steps, ratio)]
            steps = series["step"][0]
            mean, lower, upper = _mean_std_band(series[metric])
            (line,) = axis.plot(steps, mean, linewidth=3.0, label=rf"$\eta_\psi / \eta_t = {ratio:g}$")
            axis.fill_between(steps, lower, upper, color=line.get_color(), alpha=0.18, linewidth=0)
            legend_entries.setdefault(line.get_label(), line)
            if metric in SHARED_RANGE_METRICS:
                shared_bounds.extend([float(lower.min()), float(upper.max())])
        axis.axvline(perturb_at, color="#7f7f7f", linestyle=":", linewidth=2.0)
        axis.set_yscale("log")
        _apply_summary_axes(axis, xlabel="Potential steps", ylabel=label)
    for axis, (metric, _) in zip(axes[0], PANELS):
        if metric in SHARED_RANGE_METRICS:
            axis.set_ylim(min(shared_bounds) * 0.8, max(shared_bounds) * 1.25)
    figure.legend(
        list(legend_entries.values()),
        list(legend_entries.keys()),
        fontsize=SUMMARY_LEGEND_SIZE,
        ncols=len(legend_entries),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.89),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.25,
        columnspacing=1.25,
    )
    figure.subplots_adjust(left=0.06, right=0.995, bottom=0.2, top=0.84, wspace=0.3)
    png_path, pdf_path = save_png_and_pdf(figure, output_dir / f"flatness_1d_K{inner_steps}_three_panel")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="outputs/flatness_1d")
    parser.add_argument("--output-dir", default=None, help="Defaults to <results-dir>/formatted_plots.")
    parser.add_argument("--inner-steps", default="", help="Optional comma-separated K filter.")
    args = parser.parse_args()

    results_dir = _resolve_path(str(args.results_dir))
    output_dir = _resolve_path(str(args.output_dir)) if args.output_dir else results_dir / "formatted_plots"
    results, perturb_at = load_results(results_dir)
    k_values = sorted({k_value for k_value, _ in results})
    if args.inner_steps:
        k_values = [int(value) for value in str(args.inner_steps).split(",")]

    manifest = []
    for inner_steps in k_values:
        png_path, pdf_path = plot_three_panel(
            results, inner_steps=inner_steps, perturb_at=perturb_at, output_dir=output_dir
        )
        manifest.append({"inner_steps": inner_steps, "png_path": str(png_path), "pdf_path": str(pdf_path)})
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "plots": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
