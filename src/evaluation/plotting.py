"""Shared plotting helpers for publication-style evaluation figures."""

from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def apply_publication_axes(
    axis: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Apply a minimal publication-style axis treatment."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, color="#d9dde3", linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=9)
    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=10)
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=10)


def save_png_and_pdf(figure: Figure, output_stem: str | Path) -> tuple[Path, Path]:
    """Save a figure as both PNG and PDF using the same stem."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path
