"""Evaluate semi-dual flatness across checkpoints for an existing run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import load_solver_checkpoint, resolve_ot_dataset
from src.diagnostics.flatness import empirical_semidual_objective, make_noisy_potential_copy, summarize_flatness
from src.evaluation.plotting import apply_publication_axes
from src.solvers.registry import build_solver
from src.training.trainer import mean_metrics, move_to_device
from src.utils.device import infer_device


def _load_yaml(path: Path) -> dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(data, dict)
    return data


def _infer_config(run_dir: Path, *, max_items: int, overrides: list[str]) -> dict[str, Any]:
    """Reconstruct a runnable config from an existing run directory."""
    result_path = run_dir / "results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Run directory does not contain results.json: {run_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    solver_id = str(result["solver_id"])
    dataset_id = str(result["dataset_id"])
    experiment_id = str(result["experiment_id"])

    dataset_config_name = "makkuva_checkerboard" if dataset_id == "makkuva_2d" else dataset_id
    training_config_name = "makkuva_case1" if solver_id.startswith("makkuva_") else "base"

    config = {
        "model": _load_yaml(ROOT / "configs" / "model" / "ot_map.yaml"),
        "solver": _load_yaml(ROOT / "configs" / "solver" / f"{solver_id}.yaml"),
        "dataset": _load_yaml(ROOT / "configs" / "dataset" / f"{dataset_config_name}.yaml"),
        "training": _load_yaml(ROOT / "configs" / "training" / f"{training_config_name}.yaml"),
        "experiment": {
            "id": experiment_id,
            "name": f"{solver_id}_flatness",
            "output_dir": str(run_dir),
        },
        "evaluation": {
            "checkpoint_path": None,
            "output_dir": str(run_dir),
            "max_items": max_items,
        },
        "visualization": {
            "enabled": False,
            "dirpath": "visualizations",
            "max_items": max_items,
        },
    }
    if "input_dim" in config["dataset"]:
        input_dim = int(config["dataset"]["input_dim"])
        config["model"]["input_dim"] = input_dim
        config["model"]["output_dim"] = input_dim
    config["training"]["seed"] = int(result["seed"])
    config["training"]["max_steps"] = int(result["max_steps"])
    config["dataset"]["batch_size"] = int(result["batch_size"])

    if overrides:
        merged = OmegaConf.merge(OmegaConf.create(config), OmegaConf.from_dotlist(overrides))
        merged_config = OmegaConf.to_container(merged, resolve=True)
        assert isinstance(merged_config, dict)
        return merged_config
    return config


def _collect_eval_batch(loader: Any, *, max_items: int) -> dict[str, torch.Tensor]:
    """Collect one fixed validation or test batch for all potential variants."""
    batches = []
    collected = 0
    for batch in loader:
        current = batch["source"].shape[0]
        remaining = max_items - collected
        if remaining <= 0:
            break
        keep = min(current, remaining)
        batches.append(
            {
                "source": batch["source"][:keep].detach().clone(),
                "target": batch["target"][:keep].detach().clone(),
            }
        )
        collected += keep
    if not batches:
        raise ValueError("Evaluation loader yielded no batches")
    return {
        "source": torch.cat([batch["source"] for batch in batches], dim=0),
        "target": torch.cat([batch["target"] for batch in batches], dim=0),
    }


def _discover_checkpoints(checkpoint_dir: Path, *, num_checkpoints: int) -> list[Path]:
    """Select a small sweep of checkpoints from early to late training."""
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        best = checkpoint_dir / "best.pt"
        if best.exists():
            return [best]
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    if len(checkpoints) <= num_checkpoints:
        return checkpoints
    indices = torch.linspace(0, len(checkpoints) - 1, steps=num_checkpoints).round().to(torch.int64)
    selected = []
    seen: set[int] = set()
    for index in indices.tolist():
        if index in seen:
            continue
        seen.add(index)
        selected.append(checkpoints[index])
    if selected[-1] != checkpoints[-1]:
        selected[-1] = checkpoints[-1]
    return selected


def _infer_hidden_dims_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    kind: str,
) -> list[int]:
    """Infer hidden widths for supported Makkuva potential families."""
    if kind == "mlp":
        linear_weights = [
            tensor
            for key, tensor in sorted(state_dict.items())
            if key.endswith(".weight") and tensor.ndim == 2
        ]
        if len(linear_weights) < 2:
            raise ValueError("Unable to infer MLP hidden_dims from checkpoint")
        return [int(tensor.shape[0]) for tensor in linear_weights[:-1]]
    if kind == "makkuva_icnn":
        indexed_weights = []
        for key, tensor in state_dict.items():
            if key.startswith("input_layers.") and key.endswith(".weight"):
                index = int(key.split(".")[1])
                indexed_weights.append((index, tensor))
        if not indexed_weights:
            raise ValueError("Unable to infer Makkuva ICNN hidden_dims from checkpoint")
        indexed_weights.sort(key=lambda item: item[0])
        return [int(tensor.shape[0]) for _, tensor in indexed_weights]
    raise ValueError(f"Unsupported potential kind for hidden dim inference: {kind}")


def _apply_checkpoint_architecture_overrides(
    config: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Align Makkuva potential widths with the saved checkpoint architecture."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    task_state = payload.get("task", {})
    if not isinstance(task_state, dict):
        return
    for potential_name in ["f_potential", "g_potential"]:
        potential_cfg = config["solver"].get(potential_name)
        state_dict = task_state.get(potential_name)
        if not isinstance(potential_cfg, dict) or not isinstance(state_dict, dict):
            continue
        kind = str(potential_cfg.get("kind", ""))
        if kind not in {"mlp", "makkuva_icnn"}:
            continue
        potential_cfg["hidden_dims"] = _infer_hidden_dims_from_state_dict(
            state_dict,
            kind=kind,
        )


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, int]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    return {
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
    }


def _load_solver_with_checkpoint(
    config: dict[str, Any],
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> Any:
    """Build a solver and load a saved checkpoint into it."""
    solver = build_solver(config["model"], config["solver"], config["training"])
    total_steps = int(config["training"].get("max_steps") or 1)
    load_solver_checkpoint(solver, checkpoint_path=checkpoint_path, total_steps=total_steps)
    solver.to(device)
    solver.eval()
    return solver


def _mean_validation_metrics(solver: Any, loader: Any, device: torch.device) -> dict[str, float]:
    """Average the solver's existing validation metrics over a loader."""
    metrics = []
    for batch in loader:
        metrics.append(solver.validation_step(move_to_device(batch, device)))
    return mean_metrics(metrics)


def _plot_metric_vs_x(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    output_path: Path,
    ylabel: str,
) -> None:
    x_values = [float(row[x_key]) for row in rows]
    y_values = [float(row["flatness_std_F"]) for row in rows]
    figure, axis = plt.subplots(1, 1, figsize=(6.0, 4.0), dpi=200, constrained_layout=True)
    axis.plot(x_values, y_values, marker="o", linewidth=1.5, color="#355c9a")
    apply_publication_axes(axis, xlabel=x_key.replace("_", " "), ylabel=ylabel)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No diagnostic rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semi-dual flatness on an existing run")
    parser.add_argument("--run-dir", required=True, help="Completed run directory containing results.json and checkpoints/")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--num-checkpoints", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=512)
    parser.add_argument("--noise-scale", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--override", action="append", default=[], help="Optional Hydra-style overrides")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    config = _infer_config(run_dir, max_items=int(args.max_items), overrides=list(args.override))

    checkpoint_dir = run_dir / str(config["training"]["checkpointing"]["dirpath"])
    checkpoint_paths = _discover_checkpoints(checkpoint_dir, num_checkpoints=int(args.num_checkpoints))
    early_checkpoint = checkpoint_paths[0]
    _apply_checkpoint_architecture_overrides(config, early_checkpoint)

    dataset_bundle = resolve_ot_dataset(config)
    _, val_loader, test_loader = dataset_bundle.make_dataloaders()
    eval_loader = val_loader if args.split == "val" else test_loader

    device = infer_device(str(config["training"].get("device", "auto")))
    fixed_batch = _collect_eval_batch(eval_loader, max_items=int(args.max_items))
    fixed_batch = move_to_device(fixed_batch, device)

    early_solver = _load_solver_with_checkpoint(config, early_checkpoint, device=device)
    early_potential = early_solver.f_potential.to(device)  # type: ignore[attr-defined]

    torch.manual_seed(int(args.seed))
    random_solver = build_solver(config["model"], config["solver"], config["training"]).to(device)
    random_solver.eval()
    random_potential = random_solver.f_potential  # type: ignore[attr-defined]

    rows = []
    for index, checkpoint_path in enumerate(checkpoint_paths):
        solver = _load_solver_with_checkpoint(config, checkpoint_path, device=device)
        validation_metrics = _mean_validation_metrics(solver, eval_loader, device)

        final_potential = solver.f_potential.to(device)  # type: ignore[attr-defined]
        noisy_potential = make_noisy_potential_copy(
            final_potential,
            noise_scale=float(args.noise_scale),
            seed=int(args.seed) + index,
        ).to(device)

        values = {
            "final": empirical_semidual_objective(solver.compute_map, final_potential, fixed_batch["source"], fixed_batch["target"]),
            "early": empirical_semidual_objective(solver.compute_map, early_potential, fixed_batch["source"], fixed_batch["target"]),
            "random_init": empirical_semidual_objective(solver.compute_map, random_potential, fixed_batch["source"], fixed_batch["target"]),
            "noisy_final": empirical_semidual_objective(solver.compute_map, noisy_potential, fixed_batch["source"], fixed_batch["target"]),
        }
        flatness = summarize_flatness(values, final_key="final")
        metadata = _checkpoint_metadata(checkpoint_path)
        row = {
            "checkpoint": checkpoint_path.name,
            "epoch": metadata["epoch"],
            "global_step": metadata["global_step"],
            "pushforward_w2": float(validation_metrics["val/pushforward_w2"]),
            "mmd": float(validation_metrics["val/mmd"]) if "val/mmd" in validation_metrics else math.nan,
            "w2_estimate": float(validation_metrics["val/w2_estimate"]) if "val/w2_estimate" in validation_metrics else math.nan,
            **flatness,
        }
        rows.append(row)

    output_dir = run_dir / "flatness_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "flatness_results.csv", rows)
    _plot_metric_vs_x(rows, x_key="global_step", output_path=output_dir / "flatness_vs_step.png", ylabel="flatness_std_F")
    _plot_metric_vs_x(rows, x_key="pushforward_w2", output_path=output_dir / "flatness_vs_pushforward_w2.png", ylabel="flatness_std_F")


if __name__ == "__main__":
    main()
