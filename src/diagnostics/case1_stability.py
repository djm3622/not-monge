"""Case-study-1 stability diagnostics for learned OT solvers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.benchmarking import load_solver_checkpoint
from src.diagnostics.flatness import make_noisy_potential_copy, summarize_flatness
from src.evaluation.ot_metrics import map_l2_error
from src.evaluation.plotting import apply_publication_axes, save_png_and_pdf
from src.solvers.registry import build_solver
from src.training.trainer import move_to_device


def supports_case1_stability_diagnostic(solver: object, dataset_bundle: object) -> bool:
    """Return whether the solver and dataset expose the pieces needed by the diagnostic."""
    return (
        hasattr(solver, "forward_potential")
        and hasattr(solver, "compute_inverse_map")
        and hasattr(dataset_bundle, "ground_truth_potential")
    )


def empirical_forward_potential_objective(
    inverse_map_fn: Any,
    potential: nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Evaluate the source-side fixed-inverse objective E[f(x)] - E[f(S(y))]."""
    inverse = inverse_map_fn(target)
    source_values = potential(source).view(-1)
    inverse_values = potential(inverse).view(-1)
    return float((source_values.mean() - inverse_values.mean()).detach())


def centered_potential_rmse(
    potential: nn.Module,
    reference_potential: nn.Module,
    source: torch.Tensor,
) -> float:
    """Gauge-invariant potential error after removing the empirical mean."""
    predicted = potential(source).view(-1)
    reference = reference_potential(source).view(-1)
    predicted = predicted - predicted.mean()
    reference = reference - reference.mean()
    return float((predicted - reference).pow(2).mean().sqrt().detach())


def _collect_eval_batch(loader: Any, *, max_items: int) -> dict[str, torch.Tensor]:
    batches = []
    collected = 0
    for batch in loader:
        remaining = max_items - collected
        if remaining <= 0:
            break
        keep = min(int(batch["source"].shape[0]), remaining)
        batches.append(
            {
                "source": batch["source"][:keep].detach().clone(),
                "target": batch["target"][:keep].detach().clone(),
                "ground_truth_map": batch["ground_truth_map"][:keep].detach().clone(),
            }
        )
        collected += keep
    if not batches:
        raise ValueError("Evaluation loader yielded no items for the stability diagnostic")
    return {
        "source": torch.cat([batch["source"] for batch in batches], dim=0),
        "target": torch.cat([batch["target"] for batch in batches], dim=0),
        "ground_truth_map": torch.cat([batch["ground_truth_map"] for batch in batches], dim=0),
    }


def _discover_checkpoints(checkpoint_dir: Path) -> list[tuple[str, Path]]:
    checkpoint_refs: list[tuple[str, Path]] = []
    for checkpoint_path in sorted(checkpoint_dir.glob("epoch_*.pt")):
        checkpoint_refs.append(("epoch", checkpoint_path))
    last_path = checkpoint_dir / "last.pt"
    if last_path.exists():
        checkpoint_refs.append(("last", last_path))
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        checkpoint_refs.append(("best", best_path))
    if not checkpoint_refs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoint_refs


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, int]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    return {
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
    }


def _load_solver_for_checkpoint(
    config: Mapping[str, Any],
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> Any:
    solver = build_solver(config["model"], config["solver"], config["training"])
    total_steps = int(config["training"].get("max_steps") or 1)
    load_solver_checkpoint(solver, checkpoint_path=checkpoint_path, total_steps=total_steps)
    solver.to(device)
    solver.eval()
    return solver


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No stability rows to write")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _trajectory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (int(row["global_step"]), 0 if row["role"] != "best" else 1))
    by_step: dict[int, dict[str, Any]] = {}
    for row in ordered:
        step = int(row["global_step"])
        if step not in by_step or row["role"] == "last":
            by_step[step] = row
    return [by_step[step] for step in sorted(by_step)]


def _plot_stability_figure(
    rows: list[dict[str, Any]],
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    trajectory = _trajectory_rows(rows)
    steps = [float(row["global_step"]) for row in trajectory]
    map_error = [float(row["map_l2_sq"]) for row in trajectory]
    potential_error = [float(row["potential_centered_mse"]) for row in trajectory]
    flatness = [float(row["forward_flatness_std_F"]) for row in trajectory]

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), dpi=200, constrained_layout=True)
    plots = [
        (map_error, "Transport Error ||T - T*||^2"),
        (potential_error, "Centered Potential Error ||f - f*||^2"),
        (flatness, "Forward Flatness Std"),
    ]
    for axis, (values, title) in zip(axes, plots):
        axis.plot(steps, values, marker="o", linewidth=1.5, color="#355c9a")
        axis.set_title(title, fontsize=10)
        apply_publication_axes(axis, xlabel="Global Step", ylabel=None)

    best_row = next((row for row in rows if row["role"] == "best"), None)
    if best_row is not None:
        best_step = float(best_row["global_step"])
        best_values = [
            float(best_row["map_l2_sq"]),
            float(best_row["potential_centered_mse"]),
            float(best_row["forward_flatness_std_F"]),
        ]
        for axis, best_value in zip(axes, best_values):
            axis.scatter(
                [best_step],
                [best_value],
                marker="*",
                s=80,
                color="#c66a33",
                zorder=3,
            )

    return save_png_and_pdf(figure, output_stem)


def _summary_metrics(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    best_row = next((row for row in rows if row["role"] == "best"), None)
    if best_row is None:
        best_row = min(rows, key=lambda row: float(row["map_l2"]))
    return {
        "stability_best_map_l2_sq": float(best_row["map_l2_sq"]),
        "stability_best_potential_centered_rmse": float(best_row["potential_centered_rmse"]),
        "stability_best_forward_flatness_std_F": float(best_row["forward_flatness_std_F"]),
        "stability_best_forward_flatness_range_F": float(best_row["forward_flatness_range_F"]),
        "stability_best_forward_flatness_mean_abs_gap": float(best_row["forward_flatness_mean_abs_gap_to_current"]),
        "stability_best_checkpoint": str(best_row["checkpoint"]),
        "stability_best_global_step": int(best_row["global_step"]),
        "stability_results_path": str(output_dir / "stability_results.csv"),
        "stability_plot_path": str(output_dir / "stability_curve.png"),
    }


def run_case1_stability_diagnostic(
    *,
    run_dir: str | Path,
    config: Mapping[str, Any],
    dataset_bundle: Any,
    device: torch.device,
    max_items: int = 512,
    noise_scale: float = 1.0e-2,
    seed: int = 1234,
) -> dict[str, Any]:
    """Run the case-study-1 stability diagnostic over the saved checkpoint sweep."""
    run_path = Path(run_dir)
    checkpoint_dir = run_path / str(config["training"]["checkpointing"]["dirpath"])
    checkpoint_refs = _discover_checkpoints(checkpoint_dir)
    _, _, test_loader = dataset_bundle.make_dataloaders()
    fixed_batch = move_to_device(_collect_eval_batch(test_loader, max_items=max_items), device)

    ground_truth_potential = dataset_bundle.ground_truth_potential.to(device)
    ground_truth_potential.eval()

    torch.manual_seed(int(seed))
    random_solver = build_solver(config["model"], config["solver"], config["training"]).to(device)
    random_solver.eval()
    if not hasattr(random_solver, "forward_potential"):
        raise ValueError("Solver does not expose forward_potential for the stability diagnostic")
    random_potential = random_solver.forward_potential

    last_path = checkpoint_dir / "last.pt"
    last_solver = _load_solver_for_checkpoint(config, last_path, device=device) if last_path.exists() else None
    last_potential = last_solver.forward_potential if last_solver is not None else None  # type: ignore[attr-defined]

    rows: list[dict[str, Any]] = []
    for index, (role, checkpoint_path) in enumerate(checkpoint_refs):
        solver = _load_solver_for_checkpoint(config, checkpoint_path, device=device)
        if not supports_case1_stability_diagnostic(solver, dataset_bundle):
            raise ValueError("Checkpoint solver does not support the case-study-1 stability diagnostic")

        current_potential = solver.forward_potential  # type: ignore[attr-defined]
        noisy_potential = make_noisy_potential_copy(
            current_potential,
            noise_scale=float(noise_scale),
            seed=int(seed) + index,
        ).to(device)
        noisy_potential.eval()

        values = {
            "current": empirical_forward_potential_objective(
                solver.compute_inverse_map,
                current_potential,
                fixed_batch["source"],
                fixed_batch["target"],
            ),
            "noisy_current": empirical_forward_potential_objective(
                solver.compute_inverse_map,
                noisy_potential,
                fixed_batch["source"],
                fixed_batch["target"],
            ),
            "random_init": empirical_forward_potential_objective(
                solver.compute_inverse_map,
                random_potential,
                fixed_batch["source"],
                fixed_batch["target"],
            ),
            "ground_truth": empirical_forward_potential_objective(
                solver.compute_inverse_map,
                ground_truth_potential,
                fixed_batch["source"],
                fixed_batch["target"],
            ),
        }
        if last_potential is not None and checkpoint_path != last_path:
            values["last"] = empirical_forward_potential_objective(
                solver.compute_inverse_map,
                last_potential,
                fixed_batch["source"],
                fixed_batch["target"],
            )
        flatness = summarize_flatness(values, final_key="current")

        prediction = solver.compute_map(fixed_batch["source"]).detach()
        map_l2 = float(map_l2_error(prediction, fixed_batch["ground_truth_map"]))
        potential_rmse = centered_potential_rmse(current_potential, ground_truth_potential, fixed_batch["source"])
        metadata = _checkpoint_metadata(checkpoint_path)
        row = {
            "role": role,
            "checkpoint": checkpoint_path.name,
            "epoch": metadata["epoch"],
            "global_step": metadata["global_step"],
            "map_l2": map_l2,
            "map_l2_sq": map_l2 * map_l2,
            "potential_centered_rmse": potential_rmse,
            "potential_centered_mse": potential_rmse * potential_rmse,
            "forward_objective_current": values["current"],
            "forward_objective_noisy_current": values["noisy_current"],
            "forward_objective_random_init": values["random_init"],
            "forward_objective_ground_truth": values["ground_truth"],
            "forward_objective_last": values.get("last", math.nan),
            "forward_flatness_std_F": float(flatness["flatness_std_F"]),
            "forward_flatness_range_F": float(flatness["flatness_range_F"]),
            "forward_flatness_mean_abs_gap_to_current": float(flatness["flatness_mean_abs_gap_to_final"]),
        }
        rows.append(row)

    output_dir = run_path / "stability_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stability_results.csv"
    _write_rows_csv(csv_path, rows)
    (output_dir / "stability_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    png_path, pdf_path = _plot_stability_figure(rows, output_stem=output_dir / "stability_curve")

    metrics = _summary_metrics(rows, output_dir=output_dir)
    metrics["stability_plot_pdf_path"] = str(pdf_path)
    metrics["stability_plot_path"] = str(png_path)
    return metrics
