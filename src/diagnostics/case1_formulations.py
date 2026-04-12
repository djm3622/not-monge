"""Case-study-1 diagnostics for the three direct OT formulation families."""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.benchmarking import load_solver_checkpoint
from src.diagnostics.flatness import empirical_semidual_objective, make_noisy_potential_copy, summarize_flatness
from src.evaluation.ot_metrics import map_l2_error
from src.evaluation.plotting import apply_publication_axes, save_png_and_pdf
from src.solvers.registry import build_solver
from src.training.trainer import move_to_device


class PotentialAdapter(nn.Module):
    """Wrap a backbone network into the potential used by a direct solver."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        quadratic_scale: float = 0.0,
        use_c_concave_parameterization: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.quadratic_scale = float(quadratic_scale)
        self.use_c_concave_parameterization = bool(use_c_concave_parameterization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.backbone(x)
        if not self.use_c_concave_parameterization:
            return value
        quadratic = self.quadratic_scale * x.pow(2).sum(dim=-1, keepdim=True)
        return quadratic - value

    def gradient(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        value = self.forward(x)
        return torch.autograd.grad(value.sum(), x, create_graph=create_graph)[0]


def empirical_maxcorr_objective(
    map_fn: Any,
    potential: nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Evaluate the fixed-map expanded-cost / maximum-correlation objective."""
    transported = map_fn(source)
    dot_term = (source * transported).sum(dim=-1).mean()
    potential_target = potential(target).view(-1).mean()
    potential_transported = potential(transported).view(-1).mean()
    return float((dot_term + potential_target - potential_transported).detach())


def centered_potential_rmse(
    potential: nn.Module,
    reference_potential: nn.Module,
    source: torch.Tensor,
) -> float:
    """Gauge-invariant source-potential error after removing empirical means."""
    predicted = potential(source).view(-1)
    reference = reference_potential(source).view(-1)
    predicted = predicted - predicted.mean()
    reference = reference - reference.mean()
    return float((predicted - reference).pow(2).mean().sqrt().detach())


def centered_value_rmse(
    predicted: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    """Gauge-invariant scalar error after removing empirical means."""
    predicted = predicted.view(-1) - predicted.view(-1).mean()
    reference = reference.view(-1) - reference.view(-1).mean()
    return float((predicted - reference).pow(2).mean().sqrt().detach())


def gradient_rmse(
    predicted: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    """Root-mean-square vector error."""
    return float((predicted - reference).pow(2).mean().sqrt().detach())


def potential_gradient(
    potential: nn.Module,
    inputs: torch.Tensor,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    """Differentiate a scalar potential with respect to its inputs."""
    with torch.enable_grad():
        if hasattr(potential, "gradient"):
            return potential.gradient(inputs, create_graph=create_graph)  # type: ignore[return-value]
        eval_inputs = inputs.clone().requires_grad_(True)
        values = potential(eval_inputs)
        return torch.autograd.grad(values.sum(), eval_inputs, create_graph=create_graph)[0]


def formulation_case_name(solver: object) -> str:
    solver_name = str(getattr(solver, "solver_name", "")).lower()
    if solver_name in {"otp", "monge_map"}:
        return "case1_minimax"
    if solver_name in {"maxcorr", "otm"}:
        return "case2_maxcorr"
    if solver_name.startswith("makkuva"):
        return "case3_structural_gradient"
    raise ValueError(f"Unsupported solver for case-study-1 formulations diagnostic: '{solver_name}'")


def potential_input_domain(solver: object) -> str:
    case_name = formulation_case_name(solver)
    if case_name in {"case1_minimax", "case2_maxcorr"}:
        return "target"
    if case_name == "case3_structural_gradient":
        return "source"
    raise ValueError(f"Unsupported formulation case '{case_name}'")


def extract_potential_module(solver: object) -> nn.Module:
    """Return a module whose forward matches the solver's scalar potential."""
    solver_name = str(getattr(solver, "solver_name", "")).lower()
    if solver_name in {"otp", "monge_map", "maxcorr", "otm"}:
        backbone = copy.deepcopy(getattr(solver, "potential_backbone"))
        return PotentialAdapter(
            backbone,
            quadratic_scale=float(getattr(solver, "quadratic_scale", 0.0)),
            use_c_concave_parameterization=bool(
                getattr(solver, "use_c_concave_parameterization", False) and solver_name == "otp"
            ),
        )
    if solver_name.startswith("makkuva"):
        return copy.deepcopy(getattr(solver, "g_potential"))
    raise ValueError(f"Unsupported solver for potential extraction: '{solver_name}'")


def _target_potential_symbol(solver: object) -> str:
    solver_name = str(getattr(solver, "solver_name", "")).lower()
    if solver_name == "otp" and bool(getattr(solver, "use_c_concave_parameterization", False)):
        return "psi"
    if solver_name in {"monge_map", "maxcorr", "otm"}:
        return "g"
    raise ValueError(f"Unsupported solver for target-side potential symbol: '{solver_name}'")


def target_reference_potential(
    solver: object,
    source_potential: nn.Module,
    *,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Construct the canonical target-side potential and its gradient on paired points."""
    source_values = source_potential(source).view(-1)
    conjugate = (source * target).sum(dim=-1) - source_values
    solver_name = str(getattr(solver, "solver_name", "")).lower()
    if solver_name == "otp" and bool(getattr(solver, "use_c_concave_parameterization", False)):
        quadratic_scale = float(getattr(solver, "quadratic_scale", 0.5))
        values = quadratic_scale * target.pow(2).sum(dim=-1) - conjugate
        gradients = 2.0 * quadratic_scale * target - source
        return values, gradients, "psi"
    values = conjugate
    gradients = source
    return values, gradients, _target_potential_symbol(solver)


def _make_zero_potential_copy(potential: nn.Module) -> nn.Module:
    clone = copy.deepcopy(potential)
    with torch.no_grad():
        for parameter in clone.parameters():
            parameter.zero_()
    return clone


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
        raise ValueError("Evaluation loader yielded no items for the formulation diagnostic")
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


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No formulation rows to write")
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


def _plot_diagnostic_figure(
    rows: list[dict[str, Any]],
    *,
    output_stem: Path,
    case_name: str,
) -> tuple[Path, Path]:
    trajectory = _trajectory_rows(rows)
    steps = [float(row["global_step"]) for row in trajectory]
    map_error = [float(row["map_l2_sq"]) for row in trajectory]

    if case_name in {"case1_minimax", "case2_maxcorr"}:
        potential_symbol = str(trajectory[0].get("target_potential_symbol", "psi"))
        target_potential_error = [float(row["target_potential_centered_mse"]) for row in trajectory]
        target_gradient_error = [float(row["target_potential_gradient_mse"]) for row in trajectory]
        flatness = [float(row["flatness_std_F"]) for row in trajectory]
        figure, axes = plt.subplots(1, 4, figsize=(16.8, 3.6), dpi=200, constrained_layout=True)
        plots = [
            (map_error, "Transport Error ||T - T*||^2"),
            (target_potential_error, f"Centered Target Error ||{potential_symbol} - {potential_symbol}*||^2"),
            (target_gradient_error, f"Target Gradient Error ||∇{potential_symbol} - ∇{potential_symbol}*||^2"),
            (flatness, "Potential Flatness Std"),
        ]
    else:
        potential_error = [float(row["potential_centered_mse"]) for row in trajectory]
        figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), dpi=200, constrained_layout=True)
        plots = [
            (map_error, "Transport Error ||T - T*||^2"),
            (potential_error, "Centered Potential Error ||g - g*||^2"),
        ]

    if not isinstance(axes, (list, tuple, np.ndarray)):  # type: ignore[name-defined]
        axes = [axes]

    for axis, (values, title) in zip(axes, plots):
        axis.plot(steps, values, marker="o", linewidth=1.5, color="#355c9a")
        axis.set_title(title, fontsize=10)
        apply_publication_axes(axis, xlabel="Global Step", ylabel=None)

    best_row = next((row for row in rows if row["role"] == "best"), None)
    if best_row is not None:
        highlight_values = [float(best_row["map_l2_sq"])]
        if case_name in {"case1_minimax", "case2_maxcorr"}:
            highlight_values.extend(
                [
                    float(best_row["target_potential_centered_mse"]),
                    float(best_row["target_potential_gradient_mse"]),
                    float(best_row["flatness_std_F"]),
                ]
            )
        else:
            highlight_values.append(float(best_row["potential_centered_mse"]))
        for axis, best_value in zip(axes, highlight_values):
            axis.scatter(
                [float(best_row["global_step"])],
                [best_value],
                marker="*",
                s=80,
                color="#c66a33",
                zorder=3,
            )

    return save_png_and_pdf(figure, output_stem)


def run_case1_formulation_diagnostic(
    *,
    run_dir: str | Path,
    config: Mapping[str, Any],
    dataset_bundle: Any,
    device: torch.device,
    max_items: int = 512,
    noise_scale: float = 1.0e-2,
    seed: int = 1234,
) -> dict[str, Any]:
    """Run the case-study-1 direct-formulation diagnostic over saved checkpoints."""
    run_path = Path(run_dir)
    checkpoint_dir = run_path / str(config["training"]["checkpointing"]["dirpath"])
    checkpoint_refs = _discover_checkpoints(checkpoint_dir)
    _, _, test_loader = dataset_bundle.make_dataloaders()
    fixed_batch = move_to_device(_collect_eval_batch(test_loader, max_items=max_items), device)

    probe_solver = _load_solver_for_checkpoint(config, checkpoint_refs[-1][1], device=device)
    case_name = formulation_case_name(probe_solver)
    ground_truth_potential = getattr(dataset_bundle, "ground_truth_potential", None)
    if ground_truth_potential is not None:
        ground_truth_potential = ground_truth_potential.to(device)
        ground_truth_potential.eval()

    torch.manual_seed(int(seed))
    random_solver = build_solver(config["model"], config["solver"], config["training"]).to(device)
    random_solver.eval()
    random_potential = extract_potential_module(random_solver).to(device)
    random_potential.eval()

    rows: list[dict[str, Any]] = []
    for index, (role, checkpoint_path) in enumerate(checkpoint_refs):
        solver = _load_solver_for_checkpoint(config, checkpoint_path, device=device)
        prediction = solver.compute_map(fixed_batch["source"]).detach()
        map_l2 = float(map_l2_error(prediction, fixed_batch["ground_truth_map"]))
        metadata = _checkpoint_metadata(checkpoint_path)
        row = {
            "case_name": case_name,
            "role": role,
            "checkpoint": checkpoint_path.name,
            "epoch": metadata["epoch"],
            "global_step": metadata["global_step"],
            "map_l2": map_l2,
            "map_l2_sq": map_l2 * map_l2,
        }

        if case_name in {"case1_minimax", "case2_maxcorr"}:
            if ground_truth_potential is None:
                raise ValueError("Case-1/2 exact diagnostic requires dataset_bundle.ground_truth_potential")
            current_potential = extract_potential_module(solver).to(device)
            current_potential.eval()
            noisy_potential = make_noisy_potential_copy(
                current_potential,
                noise_scale=float(noise_scale),
                seed=int(seed) + index,
            ).to(device)
            noisy_potential.eval()
            zero_potential = _make_zero_potential_copy(current_potential).to(device)
            zero_potential.eval()
            objective_fn = empirical_semidual_objective if case_name == "case1_minimax" else empirical_maxcorr_objective
            values = {
                "current": objective_fn(solver.compute_map, current_potential, fixed_batch["source"], fixed_batch["target"]),
                "noisy_current": objective_fn(solver.compute_map, noisy_potential, fixed_batch["source"], fixed_batch["target"]),
                "random_init": objective_fn(solver.compute_map, random_potential, fixed_batch["source"], fixed_batch["target"]),
                "zero": objective_fn(solver.compute_map, zero_potential, fixed_batch["source"], fixed_batch["target"]),
            }
            paired_target = fixed_batch["ground_truth_map"]
            reference_values, reference_gradients, target_symbol = target_reference_potential(
                solver,
                ground_truth_potential,
                source=fixed_batch["source"],
                target=paired_target,
            )
            current_values = current_potential(paired_target).view(-1)
            current_gradients = potential_gradient(current_potential, paired_target, create_graph=False).detach()
            potential_rmse = centered_value_rmse(current_values, reference_values)
            potential_grad_rmse = gradient_rmse(current_gradients, reference_gradients)
            flatness = summarize_flatness(values, final_key="current")
            row.update(
                {
                    "target_potential_symbol": target_symbol,
                    "target_potential_centered_rmse": potential_rmse,
                    "target_potential_centered_mse": potential_rmse * potential_rmse,
                    "target_potential_gradient_rmse": potential_grad_rmse,
                    "target_potential_gradient_mse": potential_grad_rmse * potential_grad_rmse,
                    "objective_current": values["current"],
                    "objective_noisy_current": values["noisy_current"],
                    "objective_random_init": values["random_init"],
                    "objective_zero": values["zero"],
                    "flatness_std_F": float(flatness["flatness_std_F"]),
                    "flatness_range_F": float(flatness["flatness_range_F"]),
                    "flatness_mean_abs_gap_to_current": float(flatness["flatness_mean_abs_gap_to_final"]),
                }
            )
        else:
            if ground_truth_potential is None:
                raise ValueError("Case-3 exact diagnostic requires dataset_bundle.ground_truth_potential")
            current_potential = extract_potential_module(solver).to(device)
            current_potential.eval()
            potential_rmse = centered_potential_rmse(current_potential, ground_truth_potential, fixed_batch["source"])
            row.update(
                {
                    "potential_centered_rmse": potential_rmse,
                    "potential_centered_mse": potential_rmse * potential_rmse,
                }
            )
        rows.append(row)

    output_dir = run_path / "formulation_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(output_dir / "diagnostic_results.csv", rows)
    (output_dir / "diagnostic_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    png_path, pdf_path = _plot_diagnostic_figure(rows, output_stem=output_dir / "diagnostic_curve", case_name=case_name)

    best_row = next((row for row in rows if row["role"] == "best"), None)
    if best_row is None:
        best_row = min(rows, key=lambda row: float(row["map_l2"]))
    summary = {
        "formulation_case_name": case_name,
        "diagnostic_best_map_l2_sq": float(best_row["map_l2_sq"]),
        "diagnostic_results_path": str(output_dir / "diagnostic_results.csv"),
        "diagnostic_plot_path": str(png_path),
        "diagnostic_plot_pdf_path": str(pdf_path),
    }
    if case_name in {"case1_minimax", "case2_maxcorr"}:
        summary.update(
            {
                "diagnostic_target_potential_symbol": str(best_row["target_potential_symbol"]),
                "diagnostic_best_target_potential_centered_rmse": float(best_row["target_potential_centered_rmse"]),
                "diagnostic_best_target_potential_gradient_rmse": float(best_row["target_potential_gradient_rmse"]),
                "diagnostic_best_flatness_std_F": float(best_row["flatness_std_F"]),
                "diagnostic_best_flatness_range_F": float(best_row["flatness_range_F"]),
                "diagnostic_best_flatness_mean_abs_gap": float(best_row["flatness_mean_abs_gap_to_current"]),
            }
        )
    else:
        summary.update(
            {
                "diagnostic_best_potential_centered_rmse": float(best_row["potential_centered_rmse"]),
            }
        )
    return summary


def potential_seed_dispersion(
    *,
    run_dirs: Sequence[str | Path],
    config: Mapping[str, Any],
    dataset_bundle: Any,
    device: torch.device,
    max_items: int = 512,
) -> dict[str, Any]:
    """Compute across-seed dispersion of the learned potential on a fixed validation batch."""
    if not run_dirs:
        return {"seed_potential_dispersion_rmse": math.nan, "potential_domain": "unknown"}

    _, _, test_loader = dataset_bundle.make_dataloaders()
    fixed_batch = move_to_device(_collect_eval_batch(test_loader, max_items=max_items), device)
    potentials: list[torch.Tensor] = []
    domain = "unknown"
    case_name = "unknown"
    for run_dir in run_dirs:
        checkpoint_path = Path(run_dir) / str(config["training"]["checkpointing"]["dirpath"]) / "best.pt"
        solver = _load_solver_for_checkpoint(config, checkpoint_path, device=device)
        module = extract_potential_module(solver).to(device)
        module.eval()
        domain = potential_input_domain(solver)
        case_name = formulation_case_name(solver)
        inputs = fixed_batch["target"] if domain == "target" else fixed_batch["source"]
        values = module(inputs).view(-1).detach()
        values = values - values.mean()
        potentials.append(values)

    if len(potentials) <= 1:
        return {
            "seed_potential_dispersion_rmse": math.nan,
            "potential_domain": domain,
            "formulation_case_name": case_name,
        }

    pairwise = []
    for left in range(len(potentials)):
        for right in range(left + 1, len(potentials)):
            pairwise.append(float((potentials[left] - potentials[right]).pow(2).mean().sqrt().detach()))
    return {
        "seed_potential_dispersion_rmse": float(sum(pairwise) / len(pairwise)),
        "potential_domain": domain,
        "formulation_case_name": case_name,
    }
