"""Training task wrapper for DDPM experiments."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from diffusers import DDPMScheduler
from torch import nn

from src.models.unet_diffusion import build_diffusion_model
from src.training.losses import ddpm_noise_prediction_loss
from src.training.schedulers import build_one_cycle_schedulers
from src.utils.device import maybe_compile_module


class DiffusionTrainingTask(nn.Module):
    """DDPM training wrapper compatible with the custom trainer."""

    def __init__(
        self,
        model_config: Mapping[str, Any],
        training_config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.model = build_diffusion_model(model_config)
        self.model_config = dict(model_config)
        self.training_config = dict(training_config)
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        optimizer_cfg = training_config["optimizer"]
        self.learning_rate = float(optimizer_cfg["lr"])
        self.weight_decay = float(optimizer_cfg["weight_decay"])
        self.compile_enabled = bool(training_config.get("compile", False) or model_config.get("compile", False))
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None

    def configure_optimizers(self, total_steps: int) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        optimizer_cfg = self.training_config["optimizer"]
        self.lr_scheduler = build_one_cycle_schedulers(
            optimizers=[self.optimizer],
            total_steps=total_steps,
            max_lrs=[self.learning_rate],
            pct_start=float(optimizer_cfg["pct_start"]),
            div_factor=float(optimizer_cfg["div_factor"]),
            final_div_factor=float(optimizer_cfg["final_div_factor"]),
        )[0]

    def compile_modules(self) -> None:
        self.model = maybe_compile_module(self.model, self.compile_enabled)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "model": self.model.state_dict(*args, **kwargs),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.model.load_state_dict(state_dict["model"], strict=strict)
        if self.optimizer is not None and state_dict.get("optimizer") is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.lr_scheduler is not None and state_dict.get("scheduler") is not None:
            self.lr_scheduler.load_state_dict(state_dict["scheduler"])

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(x, timesteps)

    def _shared_loss(
        self,
        images: torch.Tensor,
        autocast_context: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=images.device,
        )
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        with autocast_context():
            prediction = self.model(noisy_images, timesteps)
            loss = ddpm_noise_prediction_loss(prediction, noise)
        return loss, prediction

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        if self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError("configure_optimizers must be called before training_step")
        self.optimizer.zero_grad(set_to_none=True)
        loss, _ = self._shared_loss(batch["image"], autocast_context)
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
        scaler.step(self.optimizer)
        scaler.update()
        self.lr_scheduler.step()
        return {
            "train/ddpm_loss": float(loss.detach()),
            "train/lr": self.lr_scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        loss, _ = self._shared_loss(batch["image"], lambda: torch.no_grad())
        return {"val/ddpm_loss": float(loss.detach())}

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        num_inference_steps: int = 100,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Generate samples for evaluation."""
        self.scheduler.set_timesteps(num_inference_steps)
        generated: list[torch.Tensor] = []
        channels = int(self.model_config["in_channels"])
        size = int(self.model_config["sample_size"])
        while len(generated) * batch_size < num_samples:
            current_batch = min(batch_size, num_samples - len(generated) * batch_size)
            latents = torch.randn(current_batch, channels, size, size, device=device)
            for timestep in self.scheduler.timesteps.to(device):
                model_output = self.model(
                    latents,
                    torch.full((current_batch,), int(timestep), device=device, dtype=torch.long),
                )
                latents = self.scheduler.step(model_output, timestep, latents).prev_sample
            generated.append(latents.cpu())
        return torch.cat(generated, dim=0)[:num_samples]
