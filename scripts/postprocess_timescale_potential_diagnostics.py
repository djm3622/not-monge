"""Add formulation potential diagnostics to existing timescale sweep runs.

This script is intentionally checkpoint-only: it reconstructs the run config,
loads the saved checkpoints in each run directory, runs the case-study-1
formulation diagnostic, and writes the diagnostic fields back into that run's
``results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "notmonge_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_case1_formulation_suite import (  # noqa: E402
    DEFAULT_CACHE_VERSION,
    _run_formulation_diagnostic,
    _solver_specs_for_dataset,
)
from scripts.timescale_grid_sweep import _build_grid_config  # noqa: E402


DEFAULT_OUTPUT_ROOTS = (
    "outputs/timescale_grid_synthetic_harder_mps",
    "outputs/timescale_grid_synthetic_harder_mps_otm",
    "outputs/timescale_grid_synthetic_harder_mps_monge_map_02",
    "outputs/timescale_grid_synthetic_harder_maxcorr",
)


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _diagnostic_complete(result: dict[str, Any]) -> bool:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return False
    results_path = metrics.get("diagnostic_results_path")
    plot_path = metrics.get("diagnostic_plot_path")
    return (
        metrics.get("formulation_case_name") is not None
        and bool(results_path)
        and bool(plot_path)
        and Path(str(results_path)).exists()
        and Path(str(plot_path)).exists()
    )


def _drop_existing_diagnostic_fields(result: dict[str, Any]) -> dict[str, Any]:
    updated = dict(result)
    metrics = updated.get("metrics", {})
    if not isinstance(metrics, dict):
        return updated
    updated_metrics = {
        key: value
        for key, value in metrics.items()
        if not (key.startswith("diagnostic_") or key == "formulation_case_name")
    }
    updated["metrics"] = updated_metrics
    return updated


def _close(left: float, right: float, *, atol: float = 1.0e-12) -> bool:
    return abs(float(left) - float(right)) <= atol


def _infer_timescale_config_version(
    *,
    result_path: Path,
    metrics: dict[str, Any],
    configured_k: int,
    configured_ratio: float,
    transport_lr: float,
    potential_lr: float,
    potential_steps: int,
) -> str:
    """Infer whether a result came from the legacy or current sweep config.

    The legacy sweep used ``potential_lr = ratio * K * transport_lr / potential_steps``
    and did not set ``dataset.seed`` from the run seed. The current sweep uses
    ``potential_lr = ratio * transport_lr / potential_steps`` and does seed the
    synthetic dataset from the run seed. These changes landed together, so the
    persisted configured LR is the most reliable marker available in old JSONs.
    """
    runtime_keys = {
        "solver_transport_steps",
        "solver_potential_steps",
        "solver_transport_lr",
        "solver_potential_lr",
    }
    missing_runtime_keys = sorted(runtime_keys.difference(metrics))
    if missing_runtime_keys:
        return "legacy"

    current_lr = configured_ratio * transport_lr / max(potential_steps, 1)
    legacy_lr = configured_ratio * configured_k * transport_lr / max(potential_steps, 1)
    if _close(potential_lr, current_lr):
        return "current"
    if _close(potential_lr, legacy_lr):
        return "legacy"
    raise ValueError(
        f"Cannot infer timescale config version for {result_path}: "
        f"configured_potential_lr={potential_lr}, current_expected={current_lr}, "
        f"legacy_expected={legacy_lr}"
    )


def _discover_result_paths(output_roots: list[Path], *, solvers: set[str] | None) -> list[Path]:
    paths: list[Path] = []
    for output_root in output_roots:
        for path in output_root.glob("*/*/*/results.json"):
            if solvers is not None and path.parts[-4] not in solvers:
                continue
            paths.append(path)
    return sorted(paths)


def _build_config_for_existing_result(
    *,
    result: dict[str, Any],
    result_path: Path,
    device: str,
    cache_version: str,
    eval_items: int,
    diagnostic_items: int,
    keep_config_noise: bool,
    dataset_seed_mode: str,
    overrides: list[str],
) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"{result_path} does not contain a metrics object")

    solver_name = str(result["solver_id"])
    dataset_name = str(result["dataset_id"])
    seed = int(result["seed"])
    run_dir = result_path.parent
    base_spec = dict(_solver_specs_for_dataset(dataset_name)[solver_name])

    configured_k = int(metrics.get("configured_k", metrics.get("configured_transport_steps")))
    configured_ratio = float(metrics["configured_ratio"])
    transport_lr = float(metrics["configured_transport_lr"])
    potential_lr = float(metrics["configured_potential_lr"])
    potential_steps = int(metrics.get("configured_potential_steps", 1))
    config_version = _infer_timescale_config_version(
        result_path=result_path,
        metrics=metrics,
        configured_k=configured_k,
        configured_ratio=configured_ratio,
        transport_lr=transport_lr,
        potential_lr=potential_lr,
        potential_steps=potential_steps,
    )

    batch_size = int(result.get("batch_size", base_spec.get("batch_size", 0)))
    steps_per_epoch = int(base_spec.get("steps_per_epoch", 0))
    max_steps = int(result["max_steps"])
    build_ratio = configured_ratio if config_version == "current" else configured_ratio * configured_k

    config = _build_grid_config(
        solver_name=solver_name,
        dataset_name=dataset_name,
        seed=seed,
        output_dir=run_dir,
        base_spec=base_spec,
        cache_version=cache_version,
        device=device,
        eval_items=eval_items,
        k_value=configured_k,
        ratio_value=build_ratio,
        transport_lr=transport_lr,
        potential_steps=potential_steps,
        max_steps=max_steps,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        disable_noise=not keep_config_noise,
        visualize=False,
        save_epoch_checkpoints=False,
        extra_overrides=overrides,
    )
    config["evaluation"]["max_items"] = int(eval_items)
    config["visualization"]["max_items"] = int(diagnostic_items)
    if dataset_name.startswith("synthetic_ot"):
        if dataset_seed_mode == "auto":
            use_config_seed = config_version == "legacy"
        else:
            use_config_seed = dataset_seed_mode == "config"
        if use_config_seed:
            configured_seed = int(base_spec.get("dataset_seed", config["dataset"].get("seed", 1234)))
            config["dataset"]["seed"] = configured_seed
    config.setdefault("_postprocess", {})["timescale_config_version"] = config_version
    config["_postprocess"]["original_configured_ratio"] = configured_ratio
    config["_postprocess"]["build_ratio"] = build_ratio
    return config


def _process_one_result(
    *,
    result_path: Path,
    device: str,
    cache_version: str,
    eval_items: int,
    diagnostic_items: int,
    diagnostic_noise_scale: float,
    keep_config_noise: bool,
    dataset_seed_mode: str,
    overrides: list[str],
    rerun: bool,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not rerun and _diagnostic_complete(result):
        return {"result_path": str(result_path), "status": "skipped_complete"}
    if rerun:
        result = _drop_existing_diagnostic_fields(result)

    checkpoint_dir = result_path.parent / "checkpoints"
    if not (checkpoint_dir / "best.pt").exists() and not (checkpoint_dir / "last.pt").exists():
        raise FileNotFoundError(f"No best.pt or last.pt checkpoint found in {checkpoint_dir}")

    config = _build_config_for_existing_result(
        result=result,
        result_path=result_path,
        device=device,
        cache_version=cache_version,
        eval_items=eval_items,
        diagnostic_items=diagnostic_items,
        keep_config_noise=keep_config_noise,
        dataset_seed_mode=dataset_seed_mode,
        overrides=overrides,
    )
    updated = _run_formulation_diagnostic(
        result=result,
        config=config,
        output_dir=result_path.parent,
        max_items=diagnostic_items,
        noise_scale=diagnostic_noise_scale,
    )
    metrics = updated.setdefault("metrics", {})
    if isinstance(metrics, dict):
        metrics["postprocess_timescale_config_version"] = str(
            config.get("_postprocess", {}).get("timescale_config_version", "unknown")
        )
        metrics["postprocess_dataset_seed_mode"] = str(dataset_seed_mode)
        metrics["postprocess_dataset_seed"] = int(config["dataset"]["seed"])
    result_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return {"result_path": str(result_path), "status": "updated"}


def _run_child(args: argparse.Namespace) -> None:
    outcome = _process_one_result(
        result_path=_resolve_path(str(args.run_result_json)),
        device=str(args.device),
        cache_version=str(args.cache_version),
        eval_items=int(args.eval_items),
        diagnostic_items=int(args.diagnostic_items),
        diagnostic_noise_scale=float(args.diagnostic_noise_scale),
        keep_config_noise=bool(args.keep_config_noise),
        dataset_seed_mode=str(args.dataset_seed_mode),
        overrides=list(args.override),
        rerun=bool(args.rerun),
    )
    print(json.dumps(outcome), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process existing timescale runs with potential diagnostics."
    )
    parser.add_argument("--output-roots", default=",".join(DEFAULT_OUTPUT_ROOTS))
    parser.add_argument("--solvers", default="", help="Optional comma-separated solver filter.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-version", default=DEFAULT_CACHE_VERSION)
    parser.add_argument("--eval-items", type=int, default=4096)
    parser.add_argument("--diagnostic-items", type=int, default=512)
    parser.add_argument("--diagnostic-noise-scale", type=float, default=1.0e-2)
    parser.add_argument(
        "--dataset-seed-mode",
        choices=["auto", "run", "config"],
        default="auto",
        help=(
            "For synthetic datasets, use 'auto' to infer legacy/current behavior "
            "from results.json, 'run' for dataset.seed = training seed, or "
            "'config' for the dataset YAML seed, usually 1234."
        ),
    )
    parser.add_argument("--keep-config-noise", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-result-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_result_json is not None:
        _run_child(args)
        return

    output_roots = [_resolve_path(value) for value in _parse_csv_strings(str(args.output_roots))]
    solvers = set(_parse_csv_strings(str(args.solvers))) if str(args.solvers).strip() else None
    result_paths = _discover_result_paths(output_roots, solvers=solvers)
    print(f"Discovered {len(result_paths)} result files.", flush=True)
    if args.dry_run:
        for path in result_paths:
            print(path)
        return

    failures: list[tuple[str, int]] = []
    for index, result_path in enumerate(result_paths, start=1):
        print(f"[{index}/{len(result_paths)}] {result_path}", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-result-json",
                str(result_path),
                "--device",
                str(args.device),
                "--cache-version",
                str(args.cache_version),
                "--eval-items",
                str(int(args.eval_items)),
                "--diagnostic-items",
                str(int(args.diagnostic_items)),
                "--diagnostic-noise-scale",
                str(float(args.diagnostic_noise_scale)),
                "--dataset-seed-mode",
                str(args.dataset_seed_mode),
                *(["--keep-config-noise"] if bool(args.keep_config_noise) else []),
                *(["--rerun"] if bool(args.rerun) else []),
                *[item for override in list(args.override) for item in ("--override", str(override))],
            ],
            cwd=str(ROOT),
            check=False,
        )
        if completed.returncode != 0:
            failures.append((str(result_path), completed.returncode))

    if failures:
        print("Failures:", flush=True)
        for path, returncode in failures:
            print(f"  returncode={returncode}: {path}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
