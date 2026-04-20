"""Benchmark runners and evaluation helpers for OT baselines."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from src.datasets.celeba import build_image_dataset_bundle
from src.datasets.diffusion_latent import build_diffusion_latent_bundle
from src.datasets.makkuva_2d import build_makkuva_2d_benchmark
from src.datasets.paper_mix3to10 import build_paper_mix3to10_benchmark
from src.datasets.synthetic_ot import build_synthetic_ot_benchmark
from src.evaluation.concavity_metrics import convexity_violation, envelope_gap, hessian_spectrum
from src.evaluation.generative_metrics import frechet_inception_distance, precision_recall_from_features
from src.evaluation.ot_metrics import (
    empirical_w2_distance,
    gradient_error,
    l2_unexplained_variance_percentage,
    map_l2_error,
    maximum_mean_discrepancy,
    saddle_residual,
    transport_cosine_similarity,
)
from src.evaluation.visualization import save_ot_visualizations
from src.solvers.base import BaseOTSolver
from src.solvers.registry import build_solver
from src.training.trainer import Trainer, mean_metrics, move_to_device
from src.utils.checkpointing import save_checkpoint
from src.utils.device import infer_device
from src.utils.data import maybe_override_batch_size
from src.utils.seed import seed_all


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
    metrics: dict[str, Any] = {
        "map_l2": float(map_l2_error(aggregated["prediction"], aggregated["ground_truth_map"]).detach())
        if "ground_truth_map" in aggregated
        else None,
        "pushforward_w2": empirical_w2_distance(aggregated["prediction"], aggregated["target"]),
        "mmd": maximum_mean_discrepancy(aggregated["prediction"], aggregated["target"]),
        "gradient_error": None,
        "l2_uvp": None,
        "transport_cos": None,
        "saddle_residual": None,
        "visualization_path": None,
        "transport_figure_path": None,
        "saddle_figure_path": None,
        "saddle_sample_paths": [],
    }
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
        final_val_metrics = trainer.fit(solver, train_loader, val_loader)
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
        "metrics": dict(metrics),
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
