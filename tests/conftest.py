from __future__ import annotations

import contextlib
import copy
import math
import sys
import types
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.datasets.celeba import build_image_dataset_bundle
from src.datasets.synthetic_ot import SyntheticOTBenchmark, build_synthetic_ot_benchmark
from src.utils.seed import seed_all


@pytest.fixture(autouse=True)
def deterministic_test_seed() -> None:
    seed_all(1234, deterministic=True)


@pytest.fixture(autouse=True)
def cpu_only_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


@pytest.fixture
def ot_model_config() -> dict[str, Any]:
    return {
        "input_dim": 2,
        "output_dim": 2,
        "hidden_dims": [8, 8],
        "activation": "silu",
        "dropout": 0.0,
        "residual": True,
        "layer_norm": False,
    }


@pytest.fixture
def tiny_diffusion_model_config() -> dict[str, Any]:
    return {
        "name": "diffusion_unet",
        "backend": "custom",
        "sample_size": 16,
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 8,
        "channel_multipliers": [1, 2],
        "num_res_blocks": 1,
        "attention_resolutions": [8],
        "dropout": 0.0,
        "num_heads": 1,
    }


@pytest.fixture
def tiny_training_config() -> dict[str, Any]:
    return {
        "seed": 123,
        "max_epochs": 1,
        "max_steps": 2,
        "log_every_n_steps": 1,
        "validate_every_n_epochs": 1,
        "gradient_clip_norm": 1.0,
        "precision": "fp32",
        "compile": False,
        "deterministic": True,
        "device": "cpu",
        "optimizer": {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "pct_start": 0.1,
            "div_factor": 25.0,
            "final_div_factor": 1000.0,
        },
        "fairness": {
            "batch_size": 8,
            "max_steps": 2,
            "optimizer": "adamw",
            "scheduler": "onecycle",
        },
        "logging": {
            "backend": "console",
            "project": "tests",
            "wandb_mode": "offline",
        },
        "checkpointing": {
            "dirpath": "checkpoints",
            "save_every_n_epochs": 1,
            "monitor": "val/map_l2",
            "mode": "min",
        },
    }


@pytest.fixture
def tiny_synthetic_ot_config() -> dict[str, Any]:
    return {
        "name": "synthetic_ot",
        "input_dim": 2,
        "n_train": 32,
        "n_val": 16,
        "n_test": 16,
        "batch_size": 8,
        "num_workers": 0,
        "source_distribution": "gaussian",
        "source_scale": 1.0,
        "mixture_components": 2,
        "generator_hidden_dims": [8, 8],
        "strong_convexity": 0.1,
        "seed": 7,
    }


@pytest.fixture
def tiny_fake_image_dataset_config() -> dict[str, Any]:
    return {
        "name": "fake_data",
        "root": "data",
        "image_size": 16,
        "batch_size": 4,
        "num_workers": 0,
        "download": False,
        "train_size": 16,
        "val_size": 8,
        "test_size": 8,
    }


@pytest.fixture
def tiny_diffusion_latent_dataset_config() -> dict[str, Any]:
    return {
        "name": "diffusion_latent",
        "source_dataset": "fake_data",
        "root": "data",
        "image_size": 16,
        "batch_size": 4,
        "num_workers": 0,
        "download": False,
        "latent_dim": 8,
        "timesteps": [10, 30],
        "encoder_name": "fake_encoder",
        "pretrained_encoder": False,
        "num_train_timesteps": 100,
        "train_size": 12,
        "val_size": 8,
        "test_size": 8,
        "cache_dir": "artifacts/test_latents",
        "save_intermediates": False,
    }


@pytest.fixture
def solver_config_factory() -> Callable[[str], dict[str, Any]]:
    def build(name: str) -> dict[str, Any]:
        configs = {
            "minimax": {
                "name": "minimax",
                "critic_steps": 1,
                "map_lr": 1.0e-3,
                "potential_lr": 1.0e-3,
                "potential": {
                    "kind": "mlp",
                    "hidden_dims": [8, 8],
                    "activation": "silu",
                    "layer_norm": False,
                },
            },
            "icnn": {
                "name": "icnn",
                "critic_steps": 1,
                "map_lr": 1.0e-3,
                "potential_lr": 1.0e-3,
                "potential": {
                    "hidden_dims": [8, 8],
                    "activation": "softplus",
                    "strong_convexity": 0.1,
                },
            },
            "tw2": {
                "name": "tw2",
                "forward_potential": {
                    "hidden_dims": [8, 8],
                    "activation": "softplus",
                    "strong_convexity": 0.1,
                },
                "inverse_potential": {
                    "hidden_dims": [8, 8],
                    "activation": "softplus",
                    "strong_convexity": 0.1,
                },
                "lr": 1.0e-3,
                "transport_weight": 1.0,
                "cycle_weight": 1.0,
                "mmd_weight": 0.1,
            },
            "mmv2": {
                "name": "mmv2",
                "forward_potential": {
                    "hidden_dims": [8, 8],
                    "activation": "softplus",
                    "strong_convexity": 0.1,
                },
                "inverse_potential": {
                    "hidden_dims": [8, 8],
                    "activation": "softplus",
                    "strong_convexity": 0.1,
                },
                "critic_steps": 1,
                "forward_lr": 1.0e-3,
                "inverse_lr": 1.0e-3,
            },
            "mm": {
                "name": "mm",
                "potential": {
                    "hidden_dims": [8, 8],
                    "activation": "silu",
                    "layer_norm": False,
                },
                "inner_map": {
                    "hidden_dims": [8, 8],
                    "activation": "silu",
                    "dropout": 0.0,
                    "residual": True,
                    "layer_norm": False,
                },
                "critic_steps": 1,
                "potential_lr": 1.0e-3,
                "inner_lr": 1.0e-3,
            },
            "mm_b": {
                "name": "mm_b",
                "potential": {
                    "hidden_dims": [8, 8],
                    "activation": "silu",
                    "layer_norm": False,
                },
                "potential_lr": 1.0e-3,
            },
            "qc": {
                "name": "qc",
                "potential": {
                    "hidden_dims": [8, 8],
                    "activation": "silu",
                    "layer_norm": False,
                },
                "potential_lr": 1.0e-3,
            },
            "sinkhorn": {
                "name": "sinkhorn",
                "reg": 1.0,
                "fit_samples": 16,
                "kernel_bandwidth": 1.0,
            },
            "gaussian": {
                "name": "gaussian",
                "fit_samples": 16,
            },
            "entropic": {
                "name": "entropic",
                "lr": 1.0e-3,
                "reg": 1.0,
                "cost_weight": 0.1,
            },
            "otp": {
                "name": "otp",
                "critic_steps": 1,
                "map_lr": 1.0e-3,
                "potential_lr": 1.0e-3,
                "potential": {
                    "kind": "mlp",
                    "hidden_dims": [8, 8],
                    "activation": "silu",
                    "layer_norm": False,
                },
                "smoothing": {
                    "sigma_start": 0.05,
                    "sigma_end": 0.0,
                    "anneal_steps": 16,
                },
                "plan": {
                    "enabled": True,
                    "reg": 1.0,
                    "supervision_weight": 0.5,
                    "entropy_weight": 0.0,
                },
                "regularization": {
                    "potential_gp_weight": 1.0,
                    "potential_l2_weight": 1.0e-3,
                },
            },
            "flow": {
                "name": "flow",
                "velocity": {
                    "hidden_dims": [8, 8],
                    "activation": "silu",
                    "layer_norm": False,
                },
                "lr": 1.0e-3,
                "integration": {
                    "backend": "rk4",
                    "method": "rk4",
                    "steps": 4,
                    "atol": 1.0e-5,
                    "rtol": 1.0e-5,
                    "use_adjoint": False,
                },
                "plan": {"reg": 1.0},
                "loss": {
                    "endpoint_weight": 1.0,
                    "energy_weight": 0.1,
                    "mmd_weight": 0.05,
                },
            },
            "w1": {
                "name": "w1",
                "potential": {"hidden_dims": [8, 8], "group_size": 2},
                "step_map": {
                    "hidden_dims": [8, 8],
                    "activation": "silu",
                    "dropout": 0.0,
                    "residual": True,
                    "layer_norm": False,
                },
                "critic_lr": 1.0e-3,
                "step_lr": 1.0e-3,
                "phase1_ratio": 0.5,
                "gradient_penalty_weight": 1.0,
            },
        }
        return copy.deepcopy(configs[name])

    return build


@pytest.fixture
def synthetic_bundle(tiny_synthetic_ot_config: dict[str, Any]) -> SyntheticOTBenchmark:
    return build_synthetic_ot_benchmark(tiny_synthetic_ot_config)


@pytest.fixture
def fake_image_bundle(tiny_fake_image_dataset_config: dict[str, Any]) -> Any:
    return build_image_dataset_bundle(tiny_fake_image_dataset_config)


@pytest.fixture
def ot_batch(synthetic_bundle: SyntheticOTBenchmark) -> dict[str, torch.Tensor]:
    train_loader, _, _ = synthetic_bundle.make_dataloaders()
    return next(iter(train_loader))


@pytest.fixture
def easy_ot_batch() -> dict[str, torch.Tensor]:
    values = torch.linspace(-1.0, 1.0, 16)
    source = torch.stack([values, values.flip(0)], dim=-1)
    target = 0.5 * source
    return {
        "source": source,
        "target": target,
        "ground_truth_map": target,
    }


@pytest.fixture
def diffusion_batch(fake_image_bundle: Any) -> dict[str, torch.Tensor]:
    train_loader, _, _ = fake_image_bundle.make_dataloaders()
    return next(iter(train_loader))


@pytest.fixture
def disabled_grad_scaler() -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device="cpu", enabled=False)


@pytest.fixture
def null_autocast() -> Callable[[], contextlib.AbstractContextManager[None]]:
    return lambda: contextlib.nullcontext()


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def baseline_config_factory(
    ot_model_config: dict[str, Any],
    tiny_training_config: dict[str, Any],
    tiny_synthetic_ot_config: dict[str, Any],
    solver_config_factory: Callable[[str], dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def build(
        solver_name: str,
        output_dir: str,
        experiment_id: str = "ot_recovery",
        dataset_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dataset_cfg = copy.deepcopy(dataset_config or tiny_synthetic_ot_config)
        model_cfg = copy.deepcopy(model_config or ot_model_config)
        if dataset_cfg["name"] == "diffusion_latent":
            latent_dim = int(dataset_cfg["latent_dim"])
            model_cfg["input_dim"] = latent_dim
            model_cfg["output_dim"] = latent_dim
        elif "input_dim" in dataset_cfg:
            model_cfg["input_dim"] = int(dataset_cfg["input_dim"])
            model_cfg["output_dim"] = int(dataset_cfg["input_dim"])
        training_cfg = copy.deepcopy(tiny_training_config)
        if "batch_size" in dataset_cfg:
            training_cfg["fairness"]["batch_size"] = int(dataset_cfg["batch_size"])
        return {
            "model": model_cfg,
            "solver": solver_config_factory(solver_name),
            "dataset": dataset_cfg,
            "training": training_cfg,
            "experiment": {
                "id": experiment_id,
                "name": f"{solver_name}_{experiment_id}",
                "output_dir": str(output_dir),
            },
            "evaluation": {
                "checkpoint_path": None,
                "output_dir": str(output_dir),
                "max_items": 64,
            },
        }

    return build


@pytest.fixture
def fake_wandb_module(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    class FakeRun:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.logged: list[tuple[int | None, dict[str, Any]]] = []
            self.finished = False

        def log(self, payload: dict[str, Any], step: int | None = None) -> None:
            self.logged.append((step, dict(payload)))

        def finish(self) -> None:
            self.finished = True

    runs: list[FakeRun] = []

    def init(**kwargs: Any) -> FakeRun:
        run = FakeRun(**kwargs)
        runs.append(run)
        return run

    fake_module = types.SimpleNamespace(init=init, runs=runs)
    monkeypatch.setitem(sys.modules, "wandb", fake_module)
    return fake_module


@pytest.fixture
def fake_timm(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    import src.datasets.diffusion_latent as diffusion_latent_module
    import src.models.swin_backbone as swin_module

    class FakeFeatureInfo:
        def __init__(self, channels: list[int]) -> None:
            self._channels = channels

        def channels(self) -> list[int]:
            return list(self._channels)

    class FakeTimmModel(nn.Module):
        def __init__(self, features_only: bool, out_indices: tuple[int, ...]) -> None:
            super().__init__()
            self.features_only = features_only
            self.out_indices = out_indices
            self.num_features = 16
            self._channels = [32, 64, 128, 256]
            if self.features_only:
                self.feature_info = FakeFeatureInfo([self._channels[index] for index in out_indices])

        def forward(self, images: torch.Tensor) -> Any:
            if not self.features_only:
                pooled = images.mean(dim=(-2, -1))
                repeats = math.ceil(self.num_features / pooled.shape[1])
                return pooled.repeat(1, repeats)[:, : self.num_features]
            base = images.mean(dim=1, keepdim=True)
            resolutions = [56, 28, 14, 7]
            features = []
            for index in self.out_indices:
                resized = F.interpolate(
                    base,
                    size=(resolutions[index], resolutions[index]),
                    mode="bilinear",
                    align_corners=False,
                )
                features.append(resized.repeat(1, self._channels[index], 1, 1))
            return features

    def create_model(
        model_name: str,
        pretrained: bool = False,
        num_classes: int = 0,
        features_only: bool = False,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
    ) -> FakeTimmModel:
        del model_name, pretrained, num_classes
        return FakeTimmModel(features_only=features_only, out_indices=tuple(out_indices))

    fake_module = types.SimpleNamespace(create_model=create_model)
    monkeypatch.setitem(sys.modules, "timm", fake_module)
    monkeypatch.setattr(diffusion_latent_module, "timm", fake_module)
    monkeypatch.setattr(swin_module, "timm", fake_module)
    return fake_module


@pytest.fixture
def fake_inception_extractor(monkeypatch: pytest.MonkeyPatch) -> type[nn.Module]:
    import src.evaluation.generative_metrics as generative_metrics_module

    class FakeInceptionFeatureExtractor(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            pooled = images.mean(dim=(-2, -1))
            repeats = math.ceil(8 / pooled.shape[1])
            return pooled.repeat(1, repeats)[:, :8]

    monkeypatch.setattr(
        generative_metrics_module,
        "InceptionFeatureExtractor",
        FakeInceptionFeatureExtractor,
    )
    return FakeInceptionFeatureExtractor
