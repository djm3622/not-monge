"""Post-hoc saddle analysis for minimax OT checkpoints."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import hydra
import matplotlib
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from src.benchmarking import evaluate_ot_solver, load_solver_checkpoint, resolve_ot_dataset
from src.solvers.registry import build_solver
from src.training.trainer import move_to_device


def _saddle_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    saddle = dict(config.get("saddle", {}))
    return {
        "br_steps": int(saddle.get("br_steps", 50)),
        "br_lr": float(saddle.get("br_lr", 1.0e-4)),
        "br_clip_norm": float(saddle.get("br_clip_norm", 1.0)),
        "calib_batches": int(saddle.get("calib_batches", 8)),
        "eval_batches": int(saddle.get("eval_batches", 8)),
        "curve_log_every": int(saddle.get("curve_log_every", 1)),
        "training_metrics_path": saddle.get("training_metrics_path"),
        "output_name": str(saddle.get("output_name", "saddle_fit_summary.png")),
        "summary_name": str(saddle.get("summary_name", "saddle_summary.json")),
    }


def _collect_batches(loader: Any, device: torch.device, limit: int) -> list[dict[str, torch.Tensor]]:
    batches = []
    for idx, batch in enumerate(loader):
        batches.append(move_to_device(batch, device))
        if idx + 1 >= limit:
            break
    return batches


def _objective_value(solver: Any, batches: list[dict[str, torch.Tensor]]) -> float:
    values = []
    with torch.no_grad():
        for batch in batches:
            inverse = solver.compute_inverse_map(batch["target"])
            values.append(float(solver._objective(batch["source"], batch["target"], inverse)))
    return sum(values) / max(len(values), 1)


def _best_response_curve(
    solver: Any,
    calib_batches: list[dict[str, torch.Tensor]],
    eval_batches: list[dict[str, torch.Tensor]],
    player: str,
    steps: int,
    lr: float,
    clip_norm: float,
    log_every: int,
    device: torch.device,
) -> list[tuple[int, float]]:
    work = copy.deepcopy(solver).to(device)
    work.eval()
    if player == "inverse":
        frozen_modules = [work.forward_potential, work.forward_correction]
        parameters = list(work.inverse_potential.parameters())
        if work.inverse_correction is not None:
            parameters.extend(work.inverse_correction.parameters())
        sign = 1.0
    elif player == "forward":
        frozen_modules = [work.inverse_potential, work.inverse_correction]
        parameters = list(work.forward_potential.parameters())
        if work.forward_correction is not None:
            parameters.extend(work.forward_correction.parameters())
        sign = -1.0
    else:
        raise ValueError(f"Unsupported player: {player}")

    for module in frozen_modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(parameters, lr=lr)
    curve = [(0, _objective_value(work, eval_batches))]

    for step in range(1, steps + 1):
        batch = calib_batches[(step - 1) % len(calib_batches)]
        optimizer.zero_grad(set_to_none=True)
        inverse = work.compute_inverse_map(batch["target"])
        objective = work._objective(batch["source"], batch["target"], inverse)
        loss = sign * objective
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        optimizer.step()
        if step % log_every == 0 or step == steps:
            curve.append((step, _objective_value(work, eval_batches)))

    return curve


def _load_training_trace(metrics_path: str | Path | None) -> dict[str, list[tuple[int, float]]]:
    if metrics_path is None:
        return {}
    path = Path(metrics_path)
    if not path.exists():
        return {}
    trace: dict[str, list[tuple[int, float]]] = {
        "val/objective": [],
        "val/map_l2": [],
        "train/objective": [],
        "train/map_l2": [],
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            step = int(record["step"])
            for key in trace:
                if key in record:
                    trace[key].append((step, float(record[key])))
    return trace


def _render_summary_figure(
    fit_image_path: Path,
    inverse_curve: list[tuple[int, float]],
    forward_curve: list[tuple[int, float]],
    trace: dict[str, list[tuple[int, float]]],
    summary: Mapping[str, Any],
    output_path: Path,
) -> None:
    fit_image = Image.open(fit_image_path).convert("RGB")
    fig = plt.figure(figsize=(18, 10), dpi=180)
    layout = gridspec.GridSpec(2, 2, width_ratios=[1.8, 1.0], height_ratios=[1.0, 1.0], figure=fig)

    ax_fit = fig.add_subplot(layout[:, 0])
    ax_fit.imshow(fit_image)
    ax_fit.axis("off")
    ax_fit.set_title("Transport Fit", fontsize=14)

    ax_gap = fig.add_subplot(layout[0, 1])
    inv_steps = [step for step, _ in inverse_curve]
    inv_vals = [value for _, value in inverse_curve]
    fwd_steps = [step for step, _ in forward_curve]
    fwd_vals = [value for _, value in forward_curve]
    ax_gap.plot(inv_steps, inv_vals, label="inverse best response", color="#2b6cb0", linewidth=2)
    ax_gap.plot(fwd_steps, fwd_vals, label="forward best response", color="#c05621", linewidth=2)
    ax_gap.axhline(float(summary["current_objective"]), color="#4a5568", linestyle="--", linewidth=1.5, label="current objective")
    ax_gap.set_title("Held-out Saddle Response", fontsize=14)
    ax_gap.set_xlabel("best-response steps")
    ax_gap.set_ylabel("objective value")
    ax_gap.legend(frameon=False, loc="best")
    gap_text = (
        f"current={summary['current_objective']:.4f}\n"
        f"lower={summary['approx_lower_value']:.4f}\n"
        f"upper={summary['approx_upper_value']:.4f}\n"
        f"gap={summary['approx_saddle_gap']:.4f}"
    )
    ax_gap.text(
        0.02,
        0.98,
        gap_text,
        transform=ax_gap.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7fafc", "edgecolor": "#cbd5e0"},
    )

    ax_trace = fig.add_subplot(layout[1, 1])
    if trace.get("val/objective"):
        x = [step for step, _ in trace["val/objective"]]
        y = [value for _, value in trace["val/objective"]]
        ax_trace.plot(x, y, label="val/objective", color="#805ad5", linewidth=2)
    if trace.get("val/map_l2"):
        ax_map = ax_trace.twinx()
        x = [step for step, _ in trace["val/map_l2"]]
        y = [value for _, value in trace["val/map_l2"]]
        ax_map.plot(x, y, label="val/map_l2", color="#2f855a", linewidth=2)
        ax_map.set_ylabel("val/map_l2")
        lines = ax_trace.get_lines() + ax_map.get_lines()
        labels = [line.get_label() for line in lines]
        ax_trace.legend(lines, labels, frameon=False, loc="best")
    else:
        ax_trace.legend(frameon=False, loc="best")
    ax_trace.set_title("Training Trace", fontsize=14)
    ax_trace.set_xlabel("training step")
    ax_trace.set_ylabel("val/objective")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)

    checkpoint_path = config["evaluation"]["checkpoint_path"]
    if checkpoint_path is None:
        raise ValueError("evaluation.checkpoint_path must be provided")

    analysis_cfg = _saddle_defaults(config)
    output_dir = ROOT / str(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    dataset_bundle = resolve_ot_dataset(config)
    solver = build_solver(config["model"], config["solver"], config["training"]).to(device)
    load_solver_checkpoint(solver, checkpoint_path)
    solver.eval()

    metrics = evaluate_ot_solver(
        solver,
        dataset_bundle,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    _, val_loader, test_loader = dataset_bundle.make_dataloaders()
    calib_batches = _collect_batches(val_loader, device=device, limit=analysis_cfg["calib_batches"])
    eval_batches = _collect_batches(test_loader, device=device, limit=analysis_cfg["eval_batches"])

    inverse_curve = _best_response_curve(
        solver,
        calib_batches=calib_batches,
        eval_batches=eval_batches,
        player="inverse",
        steps=analysis_cfg["br_steps"],
        lr=analysis_cfg["br_lr"],
        clip_norm=analysis_cfg["br_clip_norm"],
        log_every=analysis_cfg["curve_log_every"],
        device=device,
    )
    forward_curve = _best_response_curve(
        solver,
        calib_batches=calib_batches,
        eval_batches=eval_batches,
        player="forward",
        steps=analysis_cfg["br_steps"],
        lr=analysis_cfg["br_lr"],
        clip_norm=analysis_cfg["br_clip_norm"],
        log_every=analysis_cfg["curve_log_every"],
        device=device,
    )

    lower = min(value for _, value in inverse_curve)
    upper = max(value for _, value in forward_curve)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "current_objective": inverse_curve[0][1],
        "approx_lower_value": lower,
        "approx_upper_value": upper,
        "approx_saddle_gap": upper - lower,
        "inverse_curve": inverse_curve,
        "forward_curve": forward_curve,
    }

    summary_path = output_dir / analysis_cfg["summary_name"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fit_image_path = Path(metrics["visualization_path"])
    training_metrics_path = analysis_cfg["training_metrics_path"]
    if training_metrics_path is None:
        candidate = Path(checkpoint_path).resolve().parents[1] / "metrics.jsonl"
        training_metrics_path = str(candidate) if candidate.exists() else None
    trace = _load_training_trace(training_metrics_path)
    figure_path = output_dir / analysis_cfg["output_name"]
    _render_summary_figure(
        fit_image_path=fit_image_path,
        inverse_curve=inverse_curve,
        forward_curve=forward_curve,
        trace=trace,
        summary=summary,
        output_path=figure_path,
    )

    print(json.dumps({"summary_path": str(summary_path), "figure_path": str(figure_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
