"""Benchmark runners and evaluation helpers for OT baselines."""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from omegaconf import OmegaConf

from src.datasets.celeba import build_image_dataset_bundle
from src.datasets.diffusion_latent import build_diffusion_latent_bundle
from src.datasets.makkuva_2d import build_makkuva_2d_benchmark
from src.datasets.paper_mix3to10 import build_paper_mix3to10_benchmark
from src.datasets.synthetic_ot import build_synthetic_ot_benchmark
from src.evaluation.concavity_metrics import convexity_violation, envelope_gap, hessian_spectrum
from src.evaluation.generative_metrics import frechet_inception_distance, precision_recall_from_features
from src.evaluation.ot_metrics import (
    empirical_kr_distance,
    empirical_w2_distance,
    gradient_error,
    l2_unexplained_variance_percentage,
    map_l2_error,
    maximum_mean_discrepancy,
    saddle_residual,
    transport_cosine_similarity,
)
from src.evaluation.visualization import save_ot_visualizations

# once src.solvers are imported, the registry will be populated with all available solvers
from src.solvers.base import BaseOTSolver
from src.solvers.registry import build_solver

from src.training.trainer import Trainer, mean_metrics, move_to_device
from src.utils.checkpointing import save_checkpoint
from src.utils.device import infer_device
from src.utils.data import maybe_override_batch_size
from src.utils.seed import seed_all


class _DirectPotentialAdapter(nn.Module):
    """Wrap a direct-solver potential backbone with its solver parameterization."""

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


def resolve_ot_dataset(config: Mapping[str, Any]) -> Any:
    """Instantiate the requested OT dataset bundle."""
    training_cfg = dict(config["training"])
    dataset_cfg = dict(config["dataset"])
    dataset_name = str(dataset_cfg["name"])
    if dataset_name != "paper_mix3to10":
        fairness_batch_size = training_cfg.get("fairness", {}).get("batch_size")
        dataset_cfg = maybe_override_batch_size(dataset_cfg, fairness_batch_size)
    if dataset_name.startswith("synthetic_ot"):
        return build_synthetic_ot_benchmark(dataset_cfg)
    if dataset_name == "makkuva_2d":
        return build_makkuva_2d_benchmark(dataset_cfg)
    if dataset_name == "paper_mix3to10":
        return build_paper_mix3to10_benchmark(dataset_cfg)
    if dataset_name == "diffusion_latent":
        return build_diffusion_latent_bundle(dataset_cfg)
    if dataset_name in {"celeba", "cifar10", "fake_data"}:
        return build_image_dataset_bundle(dataset_cfg)
    raise ValueError(f"Unsupported OT dataset: {dataset_name}")


def load_solver_checkpoint(
    solver: BaseOTSolver,
    checkpoint_path: str | Path,
    total_steps: int = 1,
) -> dict[str, Any]:
    """Load a saved solver checkpoint into a solver instance."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if solver.supports_training:
        solver.configure_optimizers(total_steps=total_steps)
    solver.load_state_dict(checkpoint["task"])
    return checkpoint


def collect_ot_predictions(
    solver: BaseOTSolver,
    loader: Any,
    device: torch.device,
    max_items: int | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate predictions across a loader."""
    sources = []
    predictions = []
    targets = []
    ground_truth = []
    collected = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        current = batch["source"].shape[0]
        if max_items is not None:
            remaining = max_items - collected
            if remaining <= 0:
                break
            current = min(current, remaining)
        source = batch["source"][:current]
        target = batch["target"][:current]
        with torch.no_grad():
            prediction = solver.compute_map(source)
        sources.append(source.detach().cpu())
        predictions.append(prediction.detach().cpu())
        targets.append(target.detach().cpu())
        if "ground_truth_map" in batch:
            ground_truth.append(batch["ground_truth_map"][:current].detach().cpu())
        collected += current
        if max_items is not None and collected >= max_items:
            break
    result = {
        "source": torch.cat(sources, dim=0),
        "prediction": torch.cat(predictions, dim=0),
        "target": torch.cat(targets, dim=0),
    }
    if ground_truth:
        result["ground_truth_map"] = torch.cat(ground_truth, dim=0)
    return result


def _quadratic_transport_cost(source: torch.Tensor, transported: torch.Tensor) -> float:
    return float((0.5 * (source - transported).pow(2).sum(dim=-1).mean()).detach())


def _raw_dot_reward(source: torch.Tensor, transported: torch.Tensor) -> float:
    return float(((source * transported).sum(dim=-1).mean()).detach())


def _cost_equivalent_theorem_objective(
    solver: BaseOTSolver,
    source: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float | None, float | None, float | None]:
    """Return F, its cost-equivalent value, and raw dot reward on held-out samples."""
    potential_prediction = solver.compute_potential(prediction)
    potential_target = solver.compute_potential(target)
    if potential_prediction is None or potential_target is None:
        return None, None, _raw_dot_reward(source, prediction)

    potential_gap = potential_target.view(-1).mean() - potential_prediction.view(-1).mean()
    raw_dot = _raw_dot_reward(source, prediction)
    solver_name = str(getattr(solver, "solver_name", "")).lower()
    if solver_name in {"maxcorr", "otm"}:
        dot_objective = raw_dot + float(potential_gap.detach())
        constant = 0.5 * (
            source.pow(2).sum(dim=-1).mean() + target.pow(2).sum(dim=-1).mean()
        )
        return dot_objective, float((constant - dot_objective).detach()), raw_dot

    cost = 0.5 * (source - prediction).pow(2).sum(dim=-1).mean()
    objective = cost + potential_gap
    value = float(objective.detach())
    return value, value, raw_dot


def _target_reference_potential_values_and_gradients(
    solver: BaseOTSolver,
    source_potential: torch.nn.Module,
    *,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_values = source_potential(source).view(-1)
    conjugate = (source * target).sum(dim=-1) - source_values
    solver_name = str(getattr(solver, "solver_name", "")).lower()
    if solver_name == "otp" and bool(getattr(solver, "use_c_concave_parameterization", False)):
        quadratic_scale = float(getattr(solver, "quadratic_scale", 0.5))
        values = quadratic_scale * target.pow(2).sum(dim=-1) - conjugate
        gradients = 2.0 * quadratic_scale * target - source
        return values, gradients
    if solver_name in {"monge_map", "maxcorr", "otm"}:
        return conjugate, source
    raise ValueError(f"Unsupported solver for target potential metrics: '{solver_name}'")


def _potential_values_and_gradients(
    solver: BaseOTSolver,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    with torch.enable_grad():
        eval_inputs = inputs.detach().clone().requires_grad_(True)
        values = solver.compute_potential(eval_inputs)
        if values is None:
            return None
        gradients = torch.autograd.grad(values.view(-1).sum(), eval_inputs)[0]
    return values.view(-1), gradients


def _centered_value_mse(predicted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    predicted = predicted.view(-1) - predicted.view(-1).mean()
    reference = reference.view(-1) - reference.view(-1).mean()
    return (predicted - reference).pow(2).mean()


def _extract_direct_potential_module(solver: BaseOTSolver) -> nn.Module:
    solver_name = str(getattr(solver, "solver_name", "")).lower()
    if solver_name not in {"otp", "monge_map", "maxcorr", "otm"}:
        raise ValueError(f"Unsupported solver for target potential metrics: '{solver_name}'")
    return _DirectPotentialAdapter(
        getattr(solver, "potential_backbone"),
        quadratic_scale=float(getattr(solver, "quadratic_scale", 0.0)),
        use_c_concave_parameterization=bool(
            getattr(solver, "use_c_concave_parameterization", False) and solver_name == "otp"
        ),
    )


def _zero_potential_copy(potential: nn.Module) -> nn.Module:
    clone = copy.deepcopy(potential)
    with torch.no_grad():
        for parameter in clone.parameters():
            parameter.zero_()
    return clone


def _noisy_potential_copy(
    potential: nn.Module,
    *,
    noise_scale: float,
    seed: int,
) -> nn.Module:
    clone = copy.deepcopy(potential)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in clone.parameters():
            if parameter.numel() == 0:
                continue
            noise = torch.randn(
                parameter.shape,
                generator=generator,
                device="cpu",
                dtype=parameter.dtype,
            ).to(device=parameter.device)
            parameter.add_(noise_scale * noise)
    return clone


def _empirical_semidual_objective(
    map_fn: Any,
    potential: nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
) -> float:
    transported = map_fn(source)
    cost = 0.5 * (source - transported).pow(2).sum(dim=-1).mean()
    potential_target = potential(target).view(-1).mean()
    potential_transported = potential(transported).view(-1).mean()
    return float((cost + potential_target - potential_transported).detach())


def _empirical_maxcorr_objective(
    map_fn: Any,
    potential: nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
) -> float:
    transported = map_fn(source)
    dot_term = (source * transported).sum(dim=-1).mean()
    potential_target = potential(target).view(-1).mean()
    potential_transported = potential(transported).view(-1).mean()
    return float((dot_term + potential_target - potential_transported).detach())


def _summarize_flatness(values: Mapping[str, float], *, final_key: str) -> dict[str, float]:
    if final_key not in values:
        raise KeyError(f"final_key '{final_key}' missing from flatness values")
    tensor = torch.tensor(list(values.values()), dtype=torch.float32)
    final_value = float(values[final_key])
    gaps = torch.tensor(
        [abs(value - final_value) for key, value in values.items() if key != final_key],
        dtype=torch.float32,
    )
    return {
        "flatness_std_F": float(tensor.std(unbiased=False)),
        "flatness_range_F": float(tensor.max() - tensor.min()),
        "flatness_mean_abs_gap_to_final": 0.0 if gaps.numel() == 0 else float(gaps.mean()),
    }


def _collect_target_potential_metric_batch(
    loader: Any,
    *,
    device: torch.device,
    max_items: int,
) -> dict[str, torch.Tensor]:
    batches: list[dict[str, torch.Tensor]] = []
    collected = 0
    for batch in loader:
        remaining = max_items - collected
        if remaining <= 0:
            break
        if "ground_truth_map" not in batch:
            return {}
        keep = min(int(batch["source"].shape[0]), remaining)
        batches.append(
            {
                "source": batch["source"][:keep].detach(),
                "target": batch["ground_truth_map"][:keep].detach(),
            }
        )
        collected += keep
    if not batches:
        return {}
    return {
        "source": torch.cat([batch["source"] for batch in batches], dim=0).to(device),
        "target": torch.cat([batch["target"] for batch in batches], dim=0).to(device),
    }


def _build_target_potential_validation_metric_fn(
    dataset_bundle: Any,
    config: Mapping[str, Any],
) -> Any | None:
    metric_cfg = dict(config.get("training", {}).get("target_potential_metrics", {}))
    if not bool(metric_cfg.get("enabled", False)):
        return None
    ground_truth_potential = getattr(dataset_bundle, "ground_truth_potential", None)
    if ground_truth_potential is None:
        return None
    reference_potential = copy.deepcopy(ground_truth_potential).eval()
    max_items = int(metric_cfg.get("max_items", 512))
    if max_items <= 0:
        return None
    flatness_enabled = bool(metric_cfg.get("flatness_enabled", True))
    flatness_noise_scale = float(metric_cfg.get("flatness_noise_scale", 1.0e-2))
    metric_seed = int(config.get("training", {}).get("seed", 1234))
    random_potential: nn.Module | None = None

    def metric_fn(
        solver: BaseOTSolver,
        val_loader: Any,
        device: torch.device,
    ) -> Mapping[str, float]:
        nonlocal random_potential
        solver_name = str(getattr(solver, "solver_name", "")).lower()
        if solver_name not in {"otp", "monge_map", "maxcorr", "otm"}:
            return {}
        batch = _collect_target_potential_metric_batch(
            val_loader,
            device=device,
            max_items=max_items,
        )
        if not batch:
            return {}
        reference_potential.to(device)
        reference_potential.eval()
        current = _potential_values_and_gradients(solver, batch["target"])
        if current is None:
            return {}
        current_values, current_gradients = current
        reference_values, reference_gradients = _target_reference_potential_values_and_gradients(
            solver,
            reference_potential,
            source=batch["source"],
            target=batch["target"],
        )
        value_mse = _centered_value_mse(current_values, reference_values)
        gradient_mse = (current_gradients - reference_gradients).pow(2).mean()
        metrics = {
            "val/target_potential_centered_mse": float(value_mse.detach()),
            "val/target_potential_gradient_mse": float(gradient_mse.detach()),
        }
        if flatness_enabled:
            current_potential = _extract_direct_potential_module(solver).to(device)
            current_potential.eval()
            noisy_potential = _noisy_potential_copy(
                current_potential,
                noise_scale=flatness_noise_scale,
                seed=metric_seed + int(getattr(solver, "train_step_index", 0)),
            ).to(device)
            noisy_potential.eval()
            zero_potential = _zero_potential_copy(current_potential).to(device)
            zero_potential.eval()
            if random_potential is None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(metric_seed + 991)
                    random_solver = build_solver(config["model"], config["solver"], config["training"]).to(device)
                random_solver.eval()
                random_potential = _extract_direct_potential_module(random_solver).to(device)
                random_potential.eval()

            objective_fn = (
                _empirical_semidual_objective
                if solver_name in {"otp", "monge_map"}
                else _empirical_maxcorr_objective
            )
            values = {
                "current": objective_fn(solver.compute_map, current_potential, batch["source"], batch["target"]),
                "noisy_current": objective_fn(solver.compute_map, noisy_potential, batch["source"], batch["target"]),
                "random_init": objective_fn(solver.compute_map, random_potential, batch["source"], batch["target"]),
                "zero": objective_fn(solver.compute_map, zero_potential, batch["source"], batch["target"]),
            }
            flatness = _summarize_flatness(values, final_key="current")
            metrics.update(
                {
                    "val/flatness_std_F": float(flatness["flatness_std_F"]),
                    "val/flatness_range_F": float(flatness["flatness_range_F"]),
                    "val/flatness_mean_abs_gap_to_current": float(
                        flatness["flatness_mean_abs_gap_to_final"]
                    ),
                }
            )
        return metrics

    return metric_fn


def evaluate_ot_solver(
    solver: BaseOTSolver,
    dataset_bundle: Any,
    config: Mapping[str, Any],
    device: torch.device,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a solver according to the configured experiment."""
    _, _, test_loader = dataset_bundle.make_dataloaders()
    max_items = int(config.get("evaluation", {}).get("max_items", 2048))
    aggregated = collect_ot_predictions(solver, test_loader, device=device, max_items=max_items)
    transport_cost = _quadratic_transport_cost(aggregated["source"], aggregated["prediction"])
    optimal_cost = (
        _quadratic_transport_cost(aggregated["source"], aggregated["ground_truth_map"])
        if "ground_truth_map" in aggregated
        else None
    )
    theorem_objective, theorem_cost_equivalent, dot_reward = _cost_equivalent_theorem_objective(
        solver,
        aggregated["source"].to(device),
        aggregated["prediction"].to(device),
        aggregated["target"].to(device),
    )
    dot_reward_scale = float(getattr(solver, "dot_reward_scale", 1.0))
    metrics: dict[str, Any] = {
        "map_l2": float(map_l2_error(aggregated["prediction"], aggregated["ground_truth_map"]).detach())
        if "ground_truth_map" in aggregated
        else None,
        "pushforward_w2": empirical_w2_distance(aggregated["prediction"], aggregated["target"]),
        "d_kr": empirical_kr_distance(aggregated["prediction"], aggregated["target"]),
        "mmd": maximum_mean_discrepancy(aggregated["prediction"], aggregated["target"]),
        "transport_cost": transport_cost,
        "optimal_transport_cost": optimal_cost,
        "transport_cost_gap": abs(transport_cost - optimal_cost) if optimal_cost is not None else None,
        "theorem_objective": theorem_objective,
        "theorem_cost_equivalent": theorem_cost_equivalent,
        "theorem_gap": abs(theorem_cost_equivalent - optimal_cost)
        if theorem_cost_equivalent is not None and optimal_cost is not None
        else None,
        "dot_reward": dot_reward,
        "scaled_dot_reward": dot_reward_scale * dot_reward,
        "gradient_error": None,
        "l2_uvp": None,
        "transport_cos": None,
        "saddle_residual": None,
        "visualization_path": None,
        "transport_figure_path": None,
        "saddle_figure_path": None,
        "saddle_sample_paths": [],
    }
    with torch.no_grad():
        validation_metrics = mean_metrics(
            [solver.validation_step(move_to_device(batch, device)) for batch in test_loader]
        )
    if "val/w2_estimate" in validation_metrics:
        metrics["w2_estimate"] = float(validation_metrics["val/w2_estimate"])
    ground_truth_potential = getattr(dataset_bundle, "ground_truth_potential", None)
    if "ground_truth_map" in aggregated and ground_truth_potential is not None:
        metrics["gradient_error"] = gradient_error(
            solver.compute_map,
            lambda x: ground_truth_potential.gradient(x, create_graph=True),
            aggregated["source"].to(device),
        )
        metrics["l2_uvp"] = l2_unexplained_variance_percentage(
            aggregated["prediction"],
            aggregated["ground_truth_map"],
            aggregated["ground_truth_map"],
        )
        metrics["transport_cos"] = transport_cosine_similarity(
            aggregated["prediction"],
            aggregated["ground_truth_map"],
            aggregated["source"],
        )
    elif "ground_truth_map" in aggregated:
        metrics["l2_uvp"] = l2_unexplained_variance_percentage(
            aggregated["prediction"],
            aggregated["ground_truth_map"],
            aggregated["ground_truth_map"],
        )
        metrics["transport_cos"] = transport_cosine_similarity(
            aggregated["prediction"],
            aggregated["ground_truth_map"],
            aggregated["source"],
        )
    if hasattr(solver, "compute_inverse_map"):
        metrics["saddle_residual"] = saddle_residual(
            solver.compute_map,
            solver.compute_inverse_map,  # type: ignore[arg-type]
            aggregated["target"].to(device),
        )

    experiment_id = str(config["experiment"]["id"])
    if experiment_id == "c_concavity" and solver.supports_potential:
        source = aggregated["source"].to(device)
        potential_values = solver.compute_potential(source)
        if potential_values is not None:
            detached = potential_values.detach()
            metrics.update(envelope_gap(source, detached))
            metrics.update(convexity_violation(solver.compute_potential, source))
            metrics.update(hessian_spectrum(solver.compute_potential, source))

    if experiment_id == "diffusion_latent":
        fake_features = aggregated["prediction"].float()
        real_features = aggregated["target"].float()
        metrics["fid"] = frechet_inception_distance(real_features, fake_features)
        metrics.update(precision_recall_from_features(real_features, fake_features))
        metrics["metric_space"] = "latent"

    visualization_cfg = dict(config.get("visualization", {}))
    if (
        output_dir is not None
        and bool(visualization_cfg.get("enabled", False))
    ):
        visualization_dir = Path(output_dir) / str(visualization_cfg.get("dirpath", "visualizations"))
        image_path = save_ot_visualizations(
            aggregated,
            visualization_dir,
            max_items=int(visualization_cfg.get("max_items", 512)),
            solver=solver,
            saddle_examples=int(visualization_cfg.get("saddle_examples", 3)),
        )
        metrics["visualization_path"] = str(image_path)
        metrics["transport_figure_path"] = str(image_path)
        saddle_path = visualization_dir / "saddle_geometry.png"
        metrics["saddle_figure_path"] = str(saddle_path) if saddle_path.exists() else None
        sample_dir = visualization_dir / "saddle_samples"
        if sample_dir.exists():
            metrics["saddle_sample_paths"] = [str(path) for path in sorted(sample_dir.glob("saddle_point_*.png"))]

    return metrics


def train_baseline_run(config: Mapping[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Run training or reference fitting for one OT baseline."""
    seed_all(
        int(config["training"]["seed"]),
        deterministic=bool(config["training"].get("deterministic", False)),
    )
    dataset_bundle = resolve_ot_dataset(config)
    train_loader, val_loader, _ = dataset_bundle.make_dataloaders()

    # 
    solver = build_solver(config["model"], config["solver"], config["training"])

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if solver.supports_training:
        trainer = Trainer(
            config=config["training"],
            experiment_name=str(config["experiment"]["name"]),
            output_dir=output_dir,
            full_config=config,
        )
        final_val_metrics = trainer.fit(
            solver,
            train_loader,
            val_loader,
            extra_validation_metrics=_build_target_potential_validation_metric_fn(dataset_bundle, config),
        )
        device = trainer.device
        checkpoint_dir = output_dir / str(config["training"]["checkpointing"]["dirpath"])
        last_epoch = math.ceil(trainer.global_step / max(len(train_loader), 1))
        save_checkpoint(
            {
                "epoch": int(last_epoch),
                "global_step": int(trainer.global_step),
                "task": solver.state_dict(),
                "val_metrics": dict(final_val_metrics),
            },
            checkpoint_dir / "last.pt",
        )
        best_checkpoint = checkpoint_dir / "best.pt"
        if best_checkpoint.exists():
            load_solver_checkpoint(solver, best_checkpoint)
        solver.eval()
    else:
        solver.fit_reference(train_loader, val_loader)
        checkpoint = {
            "epoch": 0,
            "global_step": 0,
            "task": solver.state_dict(),
            "val_metrics": {},
        }
        checkpoint_dir = output_dir / str(config["training"]["checkpointing"]["dirpath"])
        save_checkpoint(checkpoint, checkpoint_dir / "best.pt")
        save_every_n_epochs = int(config["training"]["checkpointing"].get("save_every_n_epochs", 1) or 0)
        if save_every_n_epochs > 0:
            save_checkpoint(checkpoint, checkpoint_dir / "epoch_0000.pt")
        device = torch.device("cpu")
        solver.to(device)
        solver.eval()

    ground_truth_potential = getattr(dataset_bundle, "ground_truth_potential", None)
    if ground_truth_potential is not None:
        ground_truth_potential.to(device)
    metrics = evaluate_ot_solver(
        solver,
        dataset_bundle,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    result = build_result_record(config, solver, metrics)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def eval_baseline_run(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Evaluate an existing solver checkpoint or reference artifact."""
    dataset_bundle = resolve_ot_dataset(config)
    solver = build_solver(config["model"], config["solver"], config["training"])
    load_solver_checkpoint(solver, checkpoint_path)
    solver.eval()
    device = infer_device(str(config["training"].get("device", "auto")))
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    solver.to(device)
    ground_truth_potential = getattr(dataset_bundle, "ground_truth_potential", None)
    if ground_truth_potential is not None:
        ground_truth_potential.to(device)
    metrics = evaluate_ot_solver(
        solver,
        dataset_bundle,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    result = build_result_record(config, solver, metrics)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def build_result_record(
    config: Mapping[str, Any],
    solver: BaseOTSolver,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a standardized result record for downstream tables."""
    fairness = dict(config["training"].get("fairness", {}))
    if str(config["dataset"].get("name", "")) == "paper_mix3to10":
        batch_size = int(config["dataset"].get("batch_size", 0))
    else:
        batch_size = int(fairness.get("batch_size", config["dataset"].get("batch_size", 0)))
    optimizer_name = str(fairness.get("optimizer", "adamw"))
    scheduler_name = str(fairness.get("scheduler", "onecycle"))
    if solver.supports_training and solver.optimizers:
        optimizer_name = solver.optimizers[0].__class__.__name__.lower()
        scheduler_name = solver.schedulers[0].__class__.__name__.lower() if solver.schedulers else "none"
    metric_record = dict(metrics)
    for metric_name, attribute_name in {
        "solver_transport_steps": "transport_steps",
        "solver_potential_steps": "potential_steps",
        "solver_transport_lr": "transport_lr",
        "solver_potential_lr": "potential_lr",
    }.items():
        value = getattr(solver, attribute_name, None)
        if value is not None:
            metric_record[metric_name] = int(value) if attribute_name.endswith("_steps") else float(value)
    return {
        "solver_id": solver.solver_name,
        "solver_group": solver.solver_group,
        "dataset_id": str(config["dataset"]["name"]),
        "experiment_id": str(config["experiment"]["id"]),
        "seed": int(config["training"]["seed"]),
        "batch_size": batch_size,
        "max_steps": int(config["training"].get("max_steps") or fairness.get("max_steps", 0)),
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "metrics": metric_record,
    }


def generate_tables_from_results(
    search_root: str | Path,
    table_root: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate results.json files into per-experiment CSV/LaTeX tables."""
    results = []
    for path in Path(search_root).rglob("results.json"):
        with path.open("r", encoding="utf-8") as handle:
            results.append(json.load(handle))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in results:
        grouped.setdefault(record["experiment_id"], []).append(record)

    table_dir = Path(table_root)
    table_dir.mkdir(parents=True, exist_ok=True)
    for experiment_id, rows in grouped.items():
        flat_rows = []
        metric_keys = sorted({key for row in rows for key in row["metrics"]})
        for row in rows:
            flat_row = {key: row[key] for key in row if key != "metrics"}
            flat_row.update(row["metrics"])
            flat_rows.append(flat_row)
        csv_path = table_dir / f"{experiment_id}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = sorted({key for row in flat_rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

        tex_path = table_dir / f"{experiment_id}.tex"
        headers = sorted({key for row in flat_rows for key in row})
        with tex_path.open("w", encoding="utf-8") as handle:
            handle.write("\\begin{tabular}{" + "l" * len(headers) + "}\n")
            handle.write(" & ".join(headers) + " \\\\\n")
            handle.write("\\hline\n")
            for row in flat_rows:
                handle.write(" & ".join(str(row.get(header, "")) for header in headers) + " \\\\\n")
            handle.write("\\end{tabular}\n")
    return grouped
