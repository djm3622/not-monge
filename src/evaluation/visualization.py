"""Lightweight visualization helpers for OT benchmark outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from PIL import Image, ImageDraw, ImageFont


_BACKGROUND = (248, 246, 240)
_PANEL_BG = (255, 255, 255)
_BORDER = (210, 205, 196)
_TEXT = (28, 31, 36)
_SOURCE = (47, 85, 151)
_PRED = (204, 88, 40)
_TRUE = (46, 143, 82)
_GRID = (232, 228, 220)


def save_ot_visualizations(
    aggregated: Mapping[str, torch.Tensor],
    output_dir: str | Path,
    max_items: int = 512,
) -> Path:
    """Render a compact overview figure for OT predictions."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reference = aggregated.get("ground_truth_map", aggregated["target"])
    basis = _projection_basis(reference, max_items=max_items)
    source = _project_points(aggregated["source"], basis, max_items=max_items)
    prediction = _project_points(aggregated["prediction"], basis, max_items=max_items)
    target = _project_points(
        reference,
        basis,
        max_items=max_items,
    )
    source_extent = _compute_extent(source)
    target_extent = _compute_extent(prediction, target)
    predicted_transport_extent = _compute_extent(source, prediction)
    true_transport_extent = _compute_extent(source, target)

    canvas = Image.new("RGB", (1460, 1120), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = "OT Recovery Overview"
    subtitle = (
        "Projection uses principal components of the true target measure"
        if aggregated["source"].shape[-1] > 2
        else "Data shown in native 2D coordinates"
    )
    draw.text((48, 28), title, fill=_TEXT, font=font)
    draw.text((48, 52), subtitle, fill=_TEXT, font=font)

    panels = {
        "Source Samples": (40, 96, 700, 520),
        "Predicted vs True Targets": (720, 96, 1380, 520),
        "Predicted Transport Map": (40, 560, 700, 984),
        "True Transport Map": (720, 560, 1380, 984),
    }
    for panel_title, box in panels.items():
        _draw_panel_frame(draw, box, panel_title, font)

    _draw_scatter_panel(draw, panels["Source Samples"], source, source_extent, _SOURCE)
    _draw_overlay_panel(
        draw,
        panels["Predicted vs True Targets"],
        prediction,
        target,
        target_extent,
        font,
    )
    _draw_transport_panel(
        draw,
        panels["Predicted Transport Map"],
        source,
        prediction,
        predicted_transport_extent,
        _PRED,
    )
    _draw_transport_panel(
        draw,
        panels["True Transport Map"],
        source,
        target,
        true_transport_extent,
        _TRUE,
    )

    image_path = output_path / "ot_recovery_overview.png"
    canvas.save(image_path)
    return image_path


def _projection_basis(reference: torch.Tensor, max_items: int) -> torch.Tensor | None:
    clipped = reference[:max_items].detach().float().cpu()
    if clipped.ndim != 2:
        raise ValueError(f"Expected a rank-2 tensor of points, got shape {tuple(clipped.shape)}")
    if clipped.shape[-1] <= 2:
        return None
    return _principal_components(clipped)


def _project_points(points: torch.Tensor, basis: torch.Tensor | None, max_items: int) -> torch.Tensor:
    clipped = points[:max_items].detach().float().cpu()
    if clipped.ndim != 2:
        raise ValueError(f"Expected a rank-2 tensor of points, got shape {tuple(clipped.shape)}")
    if clipped.shape[-1] == 1:
        zeros = torch.zeros_like(clipped)
        return torch.cat([clipped, zeros], dim=-1)
    if clipped.shape[-1] == 2:
        return clipped
    if basis is None:
        raise ValueError("Expected a projection basis for inputs with dimension greater than 2")
    return clipped @ basis


def _principal_components(points: torch.Tensor) -> torch.Tensor:
    centered = points - points.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(centered, q=2)
    return basis[:, :2]


def _compute_extent(*point_sets: torch.Tensor) -> tuple[float, float, float, float]:
    all_points = torch.cat(point_sets, dim=0)
    mins = all_points.min(dim=0).values
    maxs = all_points.max(dim=0).values
    span = (maxs - mins).clamp_min(1e-3)
    padding = 0.08 * span
    lower = mins - padding
    upper = maxs + padding
    return float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])


def _draw_panel_frame(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=_PANEL_BG, outline=_BORDER, width=2)
    draw.text((box[0] + 20, box[1] + 14), title, fill=_TEXT, font=font)
    plot_box = _plot_box(box)
    _draw_grid(draw, plot_box)


def _draw_scatter_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    points: torch.Tensor,
    extent: tuple[float, float, float, float],
    color: tuple[int, int, int],
) -> None:
    plot_box = _plot_box(box)
    for point in points:
        x, y = _to_canvas(point, extent, plot_box)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def _draw_overlay_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    prediction: torch.Tensor,
    target: torch.Tensor,
    extent: tuple[float, float, float, float],
    font: ImageFont.ImageFont,
) -> None:
    plot_box = _plot_box(box)
    for point in target:
        x, y = _to_canvas(point, extent, plot_box)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=_TRUE)
    for point in prediction:
        x, y = _to_canvas(point, extent, plot_box)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=_PRED)

    legend_x = box[0] + 22
    legend_y = box[3] - 38
    draw.rectangle((legend_x, legend_y, legend_x + 10, legend_y + 10), fill=_PRED)
    draw.text((legend_x + 16, legend_y - 2), "predicted", fill=_TEXT, font=font)
    legend_x += 100
    draw.rectangle((legend_x, legend_y, legend_x + 10, legend_y + 10), fill=_TRUE)
    draw.text((legend_x + 16, legend_y - 2), "true", fill=_TEXT, font=font)


def _draw_transport_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    source: torch.Tensor,
    destination: torch.Tensor,
    extent: tuple[float, float, float, float],
    color: tuple[int, int, int],
) -> None:
    plot_box = _plot_box(box)
    for src_point, dst_point in zip(source, destination):
        x0, y0 = _to_canvas(src_point, extent, plot_box)
        x1, y1 = _to_canvas(dst_point, extent, plot_box)
        draw.line((x0, y0, x1, y1), fill=color, width=1)
        draw.ellipse((x0 - 2, y0 - 2, x0 + 2, y0 + 2), fill=_SOURCE)
        draw.ellipse((x1 - 2, y1 - 2, x1 + 2, y1 + 2), fill=color)


def _draw_grid(draw: ImageDraw.ImageDraw, plot_box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = plot_box
    width = right - left
    height = bottom - top
    for fraction in (0.25, 0.5, 0.75):
        x = left + int(width * fraction)
        y = top + int(height * fraction)
        draw.line((x, top, x, bottom), fill=_GRID, width=1)
        draw.line((left, y, right, y), fill=_GRID, width=1)
    draw.rectangle(plot_box, outline=_BORDER, width=1)


def _plot_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return left + 20, top + 48, right - 20, bottom - 20


def _to_canvas(
    point: torch.Tensor,
    extent: tuple[float, float, float, float],
    plot_box: tuple[int, int, int, int],
) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = extent
    left, top, right, bottom = plot_box
    x = float(point[0])
    y = float(point[1])
    norm_x = (x - min_x) / max(max_x - min_x, 1e-6)
    norm_y = (y - min_y) / max(max_y - min_y, 1e-6)
    canvas_x = left + norm_x * (right - left)
    canvas_y = bottom - norm_y * (bottom - top)
    return int(canvas_x), int(canvas_y)
