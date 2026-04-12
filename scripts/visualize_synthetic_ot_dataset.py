"""Visualize the exact synthetic OT benchmark geometry and scale."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.synthetic_ot import SyntheticOTBenchmark, build_synthetic_ot_benchmark
from src.evaluation.plotting import apply_publication_axes, save_png_and_pdf


def _collect_split(bundle: SyntheticOTBenchmark, split: str, max_items: int) -> tuple[torch.Tensor, torch.Tensor]:
    train_loader, val_loader, test_loader = bundle.make_dataloaders()
    loader = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }[split]
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    collected = 0
    for batch in loader:
        keep = min(int(batch["source"].shape[0]), max_items - collected)
        if keep <= 0:
            break
        sources.append(batch["source"][:keep].detach().cpu())
        targets.append(batch["target"][:keep].detach().cpu())
        collected += keep
    return torch.cat(sources, dim=0), torch.cat(targets, dim=0)


def _project_pair(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if source.shape[1] == 1:
        zeros = torch.zeros(source.shape[0], 1, dtype=source.dtype)
        return torch.cat([source, zeros], dim=1), torch.cat([target, zeros], dim=1)
    if source.shape[1] == 2:
        return source, target
    combined = torch.cat([source, target], dim=0)
    center = combined.mean(dim=0)
    _, _, basis = torch.pca_lowrank(combined - center, q=2)
    return (source - center) @ basis[:, :2], (target - center) @ basis[:, :2]


def _vector_rms(values: torch.Tensor) -> float:
    return float(values.pow(2).sum(dim=-1).mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the synthetic OT dataset.")
    parser.add_argument("--config", default="configs/dataset/synthetic_ot.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-items", type=int, default=1024)
    parser.add_argument("--output-dir", default="outputs/synthetic_ot_dataset_viz")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(cfg, dict)

    bundle = build_synthetic_ot_benchmark(cfg)
    source, target = _collect_split(bundle, split=str(args.split), max_items=int(args.max_items))
    projected_source, projected_target = _project_pair(source, target)

    source_rms = _vector_rms(source)
    target_rms = _vector_rms(target)
    ratio = target_rms / max(source_rms, 1.0e-8)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=220, constrained_layout=True)

    line_indices = torch.linspace(0, max(len(source) - 1, 0), steps=min(len(source), 96), dtype=torch.int64)
    for index in line_indices.tolist():
        axes[0].plot(
            [float(projected_source[index, 0]), float(projected_target[index, 0])],
            [float(projected_source[index, 1]), float(projected_target[index, 1])],
            color="#c66a33",
            alpha=0.16,
            linewidth=0.8,
        )
    axes[0].scatter(
        projected_source[:, 0].numpy(),
        projected_source[:, 1].numpy(),
        s=10,
        alpha=0.55,
        color="#355c9a",
        label="Source",
    )
    axes[0].scatter(
        projected_target[:, 0].numpy(),
        projected_target[:, 1].numpy(),
        s=10,
        alpha=0.55,
        color="#3a8f5b",
        label="Target",
    )
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    axes[0].set_title("Projected Source and Target", fontsize=11)
    apply_publication_axes(axes[0], xlabel="PCA Component 1", ylabel="PCA Component 2")

    source_norm = torch.linalg.norm(source, dim=1).numpy()
    target_norm = torch.linalg.norm(target, dim=1).numpy()
    axes[1].hist(source_norm, bins=36, alpha=0.65, color="#355c9a", label="||source||")
    axes[1].hist(target_norm, bins=36, alpha=0.6, color="#3a8f5b", label="||target||")
    axes[1].legend(frameon=False, fontsize=9, loc="best")
    axes[1].set_title("Norm Distribution", fontsize=11)
    apply_publication_axes(axes[1], xlabel="Euclidean Norm", ylabel="Count")
    axes[1].text(
        0.03,
        0.97,
        "\n".join(
            [
                f"source RMS: {source_rms:.3f}",
                f"target RMS: {target_rms:.3f}",
                f"target/source: {ratio:.3f}",
                f"potential scale: {bundle.potential_scale:.6f}",
            ]
        ),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d9dde3"},
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    png_path, pdf_path = save_png_and_pdf(figure, output_dir / "source_target_geometry")
    plt.close(figure)

    stats = {
        "config_path": str(config_path),
        "split": str(args.split),
        "max_items": int(args.max_items),
        "source_rms": source_rms,
        "target_rms": target_rms,
        "target_to_source_rms_ratio": ratio,
        "potential_scale": float(bundle.potential_scale),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
    }
    (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
