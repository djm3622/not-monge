"""Publication-style OT geometry visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np
import torch
from matplotlib.collections import LineCollection

from src.evaluation.plotting import apply_publication_axes, save_png_and_pdf

matplotlib.use("Agg")
from matplotlib import pyplot as plt


_SOURCE_COLOR = "#355c9a"
_PRED_COLOR = "#c66a33"
_TRUE_COLOR = "#3a8f5b"
_VECTOR_COLOR = "#7e4d9b"
_CONTOUR_COLOR = "#9aa4b1"
_CLOUD_ALPHA = 0.55
_CONTOUR_STYLE = (0, (2.0, 2.0))


@dataclass(frozen=True)
class ProjectionSpec:
    center: torch.Tensor
    basis: torch.Tensor


def save_ot_visualizations(
    aggregated: Mapping[str, torch.Tensor],
    output_dir: str | Path,
    max_items: int = 512,
    solver: object | None = None,
) -> Path:
    """Render transport geometry and, when available, a separate saddle slice."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clipped = _clip_aggregated(aggregated, max_items=max_items)
    reference = clipped.get("ground_truth_map", clipped["target"])
    projection = _build_projection(clipped["source"], clipped["prediction"], reference)
    projected = {
        "source": _project_points(clipped["source"], projection),
        "prediction": _project_points(clipped["prediction"], projection),
        "reference": _project_points(reference, projection),
    }
    selected_index = _select_representative_index(clipped["source"], clipped["prediction"], reference)

    has_saddle = (
        solver is not None
        and "ground_truth_map" in clipped
        and hasattr(solver, "forward_potential")
        and hasattr(solver, "compute_inverse_map")
    )
    figure, axis = plt.subplots(1, 1, figsize=(6.2, 5.2), dpi=200, constrained_layout=True)
    _draw_transport_geometry(
        axis,
        projected_source=projected["source"],
        projected_prediction=projected["prediction"],
        projected_reference=projected["reference"],
        selected_index=selected_index,
    )
    png_path, _ = save_png_and_pdf(figure, output_path / "transport_geometry")
    plt.close(figure)

    if has_saddle:
        saddle_figure, saddle_axis = plt.subplots(1, 1, figsize=(6.2, 5.2), dpi=200, constrained_layout=True)
        _draw_saddle_geometry(
            saddle_axis,
            solver=solver,
            projection=projection,
            reference_points=clipped["ground_truth_map"],
            projected_source=projected["source"],
            selected_index=selected_index,
        )
        save_png_and_pdf(saddle_figure, output_path / "saddle_geometry")
        plt.close(saddle_figure)
    return png_path


def _clip_aggregated(
    aggregated: Mapping[str, torch.Tensor],
    *,
    max_items: int,
) -> dict[str, torch.Tensor]:
    clipped: dict[str, torch.Tensor] = {}
    for key, value in aggregated.items():
        clipped[key] = value[:max_items].detach().float().cpu()
    return clipped


def _build_projection(*point_sets: torch.Tensor) -> ProjectionSpec:
    reference = torch.cat(point_sets, dim=0)
    input_dim = int(reference.shape[1])
    if input_dim == 1:
        return ProjectionSpec(
            center=torch.zeros(1, dtype=reference.dtype),
            basis=torch.tensor([[1.0, 0.0]], dtype=reference.dtype),
        )
    if input_dim == 2:
        return ProjectionSpec(
            center=torch.zeros(2, dtype=reference.dtype),
            basis=torch.eye(2, dtype=reference.dtype),
        )
    center = reference.mean(dim=0)
    centered = reference - center
    _, _, basis = torch.pca_lowrank(centered, q=2)
    return ProjectionSpec(center=center, basis=basis[:, :2])


def _project_points(points: torch.Tensor, projection: ProjectionSpec) -> torch.Tensor:
    if points.shape[1] == 1:
        zeros = torch.zeros(points.shape[0], 1, dtype=points.dtype)
        return torch.cat([points, zeros], dim=1)
    if points.shape[1] == 2 and torch.allclose(projection.basis, torch.eye(2, dtype=projection.basis.dtype)):
        return points
    return (points - projection.center) @ projection.basis


def _select_representative_index(
    source: torch.Tensor,
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> int:
    errors = torch.linalg.norm(prediction - reference, dim=1)
    transport = torch.linalg.norm(reference - source, dim=1)
    if len(errors) == 0:
        return 0
    mask = transport >= torch.quantile(transport, 0.6)
    if bool(mask.any()):
        candidate_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        candidate_errors = errors[candidate_indices]
    else:
        candidate_indices = torch.arange(len(errors))
        candidate_errors = errors
    target_error = torch.median(candidate_errors)
    choice = torch.argmin(torch.abs(candidate_errors - target_error))
    return int(candidate_indices[choice])


def _sample_indices(total: int, limit: int) -> np.ndarray:
    count = min(total, limit)
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.linspace(0, total - 1, num=count, dtype=np.int64)


def _compute_extent(*point_sets: torch.Tensor) -> tuple[float, float, float, float]:
    combined = torch.cat(point_sets, dim=0)
    mins = combined.min(dim=0).values
    maxs = combined.max(dim=0).values
    span = (maxs - mins).clamp_min(1.0e-3)
    padding = 0.08 * span
    lower = mins - padding
    upper = maxs + padding
    return float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])


def _draw_transport_geometry(
    axis: plt.Axes,
    *,
    projected_source: torch.Tensor,
    projected_prediction: torch.Tensor,
    projected_reference: torch.Tensor,
    selected_index: int,
) -> None:
    line_indices = _sample_indices(len(projected_source), limit=96)
    pred_segments = np.stack(
        [
            projected_source[line_indices].numpy(),
            projected_prediction[line_indices].numpy(),
        ],
        axis=1,
    )
    true_segments = np.stack(
        [
            projected_source[line_indices].numpy(),
            projected_reference[line_indices].numpy(),
        ],
        axis=1,
    )
    axis.add_collection(
        LineCollection(true_segments, colors=_TRUE_COLOR, linewidths=0.8, linestyles="dashed", alpha=0.18)
    )
    axis.add_collection(
        LineCollection(pred_segments, colors=_PRED_COLOR, linewidths=0.8, alpha=0.24)
    )

    axis.scatter(
        projected_source[:, 0].numpy(),
        projected_source[:, 1].numpy(),
        s=10,
        color=_SOURCE_COLOR,
        alpha=_CLOUD_ALPHA,
        label="Source",
    )
    axis.scatter(
        projected_reference[:, 0].numpy(),
        projected_reference[:, 1].numpy(),
        s=11,
        color=_TRUE_COLOR,
        alpha=_CLOUD_ALPHA,
        label="True Map",
    )
    axis.scatter(
        projected_prediction[:, 0].numpy(),
        projected_prediction[:, 1].numpy(),
        s=11,
        color=_PRED_COLOR,
        alpha=_CLOUD_ALPHA,
        label="Predicted Map",
    )

    extent = _compute_extent(projected_source, projected_prediction, projected_reference)
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal", adjustable="box")
    apply_publication_axes(axis, xlabel="PCA Compenent 1", ylabel="PCA Component 2")
    legend = axis.legend(frameon=True, fontsize=8, loc="best")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#c7cdd6")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_boxstyle("square", pad=0.25)


def _draw_saddle_geometry(
    axis: plt.Axes,
    *,
    solver: object,
    projection: ProjectionSpec,
    reference_points: torch.Tensor,
    projected_source: torch.Tensor,
    selected_index: int,
) -> None:
    device = _solver_device(solver)
    basis = projection.basis.to(device)
    selected_target = reference_points[selected_index].to(device)
    with torch.no_grad():
        inverse_point = solver.compute_inverse_map(selected_target.unsqueeze(0)).squeeze(0).detach()  # type: ignore[attr-defined]
    inverse_projected = _project_points(inverse_point.detach().cpu().unsqueeze(0), projection).squeeze(0)
    true_projected = projected_source[selected_index]

    plane_radius = _plane_radius(projected_source, true_projected, inverse_projected)
    grid_coords = np.linspace(-plane_radius, plane_radius, num=81, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(grid_coords, grid_coords)
    contour_values = _evaluate_plane_objective(
        solver=solver,
        target=selected_target,
        inverse_point=inverse_point,
        basis=basis,
        grid_x=grid_x,
        grid_y=grid_y,
    )
    contour_levels = _contour_levels(contour_values)
    if contour_levels is not None:
        axis.contour(
            inverse_projected[0].item() + grid_x,
            inverse_projected[1].item() + grid_y,
            contour_values,
            levels=contour_levels,
            colors=_CONTOUR_COLOR,
            linewidths=0.9,
            linestyles=[_CONTOUR_STYLE],
        )

    axis.scatter(
        projected_source[:, 0].numpy(),
        projected_source[:, 1].numpy(),
        s=9,
        color=_SOURCE_COLOR,
        alpha=0.16,
        label="Source",
    )
    field_x, field_y, grad_u, grad_v = _evaluate_plane_vector_field(
        solver=solver,
        target=selected_target,
        inverse_point=inverse_point,
        inverse_projected=inverse_projected,
        basis=basis,
        plane_radius=plane_radius,
    )
    axis.quiver(
        field_x,
        field_y,
        grad_u,
        grad_v,
        color=_VECTOR_COLOR,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0032,
        headwidth=3.8,
        headlength=4.8,
        headaxislength=4.2,
        alpha=0.68,
        zorder=1,
    )

    axis.scatter(
        [true_projected[0].item()],
        [true_projected[1].item()],
        s=62,
        facecolors="white",
        edgecolors=_SOURCE_COLOR,
        linewidths=1.6,
        label="Preimage",
    )
    axis.scatter(
        [inverse_projected[0].item()],
        [inverse_projected[1].item()],
        s=72,
        marker="*",
        color=_PRED_COLOR,
        linewidths=0.0,
        label="Inverse Response",
    )

    axis.set_xlim(inverse_projected[0].item() - plane_radius, inverse_projected[0].item() + plane_radius)
    axis.set_ylim(inverse_projected[1].item() - plane_radius, inverse_projected[1].item() + plane_radius)
    axis.set_aspect("equal", adjustable="box")
    apply_publication_axes(axis, xlabel="PCA Compenent 1", ylabel="PCA Component 2")
    legend = axis.legend(frameon=True, fontsize=8, loc="best")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#c7cdd6")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_boxstyle("square", pad=0.25)


def _solver_device(solver: object) -> torch.device:
    parameters = []
    if hasattr(solver, "parameters"):
        parameters = list(solver.parameters())  # type: ignore[attr-defined]
    if parameters:
        return parameters[0].device
    return torch.device("cpu")


def _plane_radius(
    projected_source: torch.Tensor,
    true_projected: torch.Tensor,
    inverse_projected: torch.Tensor,
) -> float:
    global_span = float(
        max(
            projected_source[:, 0].max().item() - projected_source[:, 0].min().item(),
            projected_source[:, 1].max().item() - projected_source[:, 1].min().item(),
        )
    )
    local_distances = torch.linalg.norm(projected_source - true_projected.unsqueeze(0), dim=1)
    local_radius = float(torch.quantile(local_distances, 0.3).item())
    inverse_gap = float(torch.linalg.norm(inverse_projected - true_projected).item())
    radius = max(0.08 * global_span, 0.9 * local_radius, 1.2 * inverse_gap, 0.15)
    return min(radius, max(0.22 * global_span, 0.2))


def _evaluate_plane_objective(
    *,
    solver: object,
    target: torch.Tensor,
    inverse_point: torch.Tensor,
    basis: torch.Tensor,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    offsets = torch.from_numpy(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)).to(target.device)
    points = inverse_point.unsqueeze(0) + offsets[:, :1] * basis[:, 0].unsqueeze(0) + offsets[:, 1:] * basis[:, 1].unsqueeze(0)
    values = []
    with torch.no_grad():
        for chunk in torch.split(points, 2048):
            objective = solver.forward_potential(chunk).view(-1) - (chunk * target.unsqueeze(0)).sum(dim=1)  # type: ignore[attr-defined]
            values.append(objective.detach().cpu())
    return torch.cat(values).view(grid_x.shape).numpy()


def _contour_levels(values: np.ndarray) -> np.ndarray | None:
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or abs(maximum - minimum) < 1.0e-8:
        return None
    lower = np.percentile(values, 10.0)
    upper = np.percentile(values, 92.0)
    if upper <= lower:
        lower = minimum
        upper = maximum
    return np.linspace(lower, upper, num=9)


def _evaluate_plane_vector_field(
    *,
    solver: object,
    target: torch.Tensor,
    inverse_point: torch.Tensor,
    inverse_projected: torch.Tensor,
    basis: torch.Tensor,
    plane_radius: float,
    grid_size: int = 19,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(-plane_radius, plane_radius, num=grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coords, coords)
    alpha = torch.from_numpy(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)).to(
        device=target.device,
        dtype=target.dtype,
    )
    alpha.requires_grad_(True)
    candidate = inverse_point.unsqueeze(0) + alpha[:, :1] * basis[:, 0].unsqueeze(0) + alpha[:, 1:] * basis[:, 1].unsqueeze(0)
    objective = solver.forward_potential(candidate).view(-1) - (candidate * target.unsqueeze(0)).sum(dim=1)  # type: ignore[attr-defined]
    gradients = torch.autograd.grad(objective.sum(), alpha)[0].detach().cpu().numpy()
    directions = -gradients
    magnitudes = np.linalg.norm(directions, axis=1, keepdims=True)
    scale = max(float(np.percentile(magnitudes, 82.0)), 1.0e-6)
    normalized = directions / np.maximum(magnitudes, 1.0e-6)
    strengths = np.clip(magnitudes / scale, 0.22, 1.0)
    vectors = 0.15 * plane_radius * normalized * strengths
    return (
        inverse_projected[0].item() + grid_x,
        inverse_projected[1].item() + grid_y,
        vectors[:, 0].reshape(grid_x.shape),
        vectors[:, 1].reshape(grid_y.shape),
    )
