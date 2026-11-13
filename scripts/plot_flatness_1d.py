"""Plot the 1-D Gaussian flatness sweep from its saved ``results.npz``.

Reads the sweep output of ``scripts/flatness_1d_sweep.py`` (or the equivalent
notebook export) and draws, for each K, one figure each for the transport error, potential
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

from scripts.post_eval_timescale_metrics import _positive_floor  # noqa: E402
from src.evaluation.plotting import apply_publication_axes, save_png_and_pdf  # noqa: E402


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


def plot_panels(
    results: dict[tuple[int, float], dict[str, np.ndarray]],
    *,
    inner_steps: int,
    perturb_at: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Write one appendix-style figure per panel metric for a single K."""
    ratios = sorted((ratio for k_value, ratio in results if k_value == inner_steps), reverse=True)
    bands = {
        (metric, ratio): _mean_std_band(results[(inner_steps, ratio)][metric])
        for metric, _ in PANELS
        for ratio in ratios
    }
    shared_values = [
        value
        for (metric, _), (_, lower, upper) in bands.items()
        if metric in SHARED_RANGE_METRICS
        for value in (float(lower.min()), float(upper.max()))
    ]
    manifest: list[dict[str, Any]] = []
    for panel_index, (metric, label) in enumerate(PANELS):
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        for ratio in ratios:
            steps = results[(inner_steps, ratio)]["step"][0]
            mean, lower, upper = bands[(metric, ratio)]
            (line,) = axis.plot(steps, mean, linewidth=1.8, label=rf"$\eta_\psi / \eta_t = {ratio:g}$")
            axis.fill_between(steps, lower, upper, color=line.get_color(), alpha=0.18, linewidth=0)
        axis.axvline(perturb_at, color="#7f7f7f", linestyle=":", linewidth=1.2)
        axis.set_yscale("log")
        if metric in SHARED_RANGE_METRICS:
            axis.set_ylim(min(shared_values) * 0.8, max(shared_values) * 1.25)
        apply_publication_axes(axis, xlabel="Potential steps", ylabel=label)
        axis.legend(
            fontsize=8,
            ncols=len(ratios),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.6,
            handletextpad=0.45,
            labelspacing=0.3,
            columnspacing=1.0,
        )
        stem = output_dir / f"flatness_1d_K{inner_steps}_{'abc'[panel_index]}_{metric}"
        png_path, pdf_path = save_png_and_pdf(figure, stem)
        plt.close(figure)
        manifest.append(
            {"inner_steps": inner_steps, "metric": metric, "png_path": str(png_path), "pdf_path": str(pdf_path)}
        )
    return manifest


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
        manifest.extend(
            plot_panels(results, inner_steps=inner_steps, perturb_at=perturb_at, output_dir=output_dir)
        )
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "plots": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
