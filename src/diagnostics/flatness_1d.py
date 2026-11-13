"""One-dimensional Gaussian flatness experiment for the semi-dual objective.

mu = N(0, 1), nu = N(m, s^2), c(x, y) = |x - y|^2 / 2. The optimal map is
T*(x) = m + s x and the optimal target potential has gradient
psi*'(y) = y - (y - m) / s. The semi-dual objective

    F(T, psi) = E_mu c(X, T(X)) + E_nu psi(Y) - E_mu psi(T(X))

is minimized over T and maximized over psi, with optimal value
F* = (m^2 + (s - 1)^2) / 2 (= 2.125 for m = 2, s = 1.5).

Each run trains a plain MLP map and a plain MLP potential from scratch with
K map steps per potential step and eta_psi = ratio * eta_T, perturbs the
potential parameters once with seeded Gaussian noise, and logs validation
metrics on a fixed sample shared across all runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from src.diagnostics.flatness import empirical_semidual_objective, make_noisy_potential_copy
from src.utils.seed import seed_all

METRIC_KEYS = (
    "step",
    "T_updates",
    "map_err",
    "grad_err",
    "F",
    "F_gap",
    "dkr",
    "lip_emp",
    "lip_dev",
    "ratio",
)

# Seed offsets for the independent random streams of one run.
DATA_SEED_OFFSET = 1000
PERTURB_SEED_OFFSET = 2000


@dataclass(frozen=True)
class Gaussian1DProblem:
    """Closed-form 1-D Gaussian OT problem mu = N(0, 1) -> nu = N(mean, std^2)."""

    mean: float = 2.0
    std: float = 1.5

    @property
    def optimal_value(self) -> float:
        """F* = W_2^2 under the cost |x - y|^2 / 2."""
        return 0.5 * (self.mean**2 + (self.std - 1.0) ** 2)

    def optimal_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean + self.std * x

    def optimal_potential_gradient(self, y: torch.Tensor) -> torch.Tensor:
        return y - (y - self.mean) / self.std

    def sample(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a source batch and then a target batch from the same stream."""
        source = torch.randn(n, 1, generator=generator)
        target = self.mean + self.std * torch.randn(n, 1, generator=generator)
        return source, target

    def grid(self, num_points: int) -> torch.Tensor:
        """Grid over mean +/- 3 std of nu, used for the empirical Lipschitz estimate."""
        half_width = 3.0 * self.std
        return torch.linspace(self.mean - half_width, self.mean + half_width, num_points).unsqueeze(-1)


def build_mlp(width: int) -> nn.Sequential:
    """Plain 1-width-width-1 ReLU MLP with no convexity or Lipschitz constraint."""
    return nn.Sequential(
        nn.Linear(1, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, 1),
    )


def _semidual_value(map_net: nn.Module, potential: nn.Module, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    transported = map_net(source)
    cost = 0.5 * (source - transported).pow(2).mean()
    return cost + potential(target).mean() - potential(transported).mean()


def _potential_gradient(potential: nn.Module, y: torch.Tensor) -> torch.Tensor:
    y = y.clone().requires_grad_(True)
    return torch.autograd.grad(potential(y).sum(), y)[0]


def evaluate(
    problem: Gaussian1DProblem,
    map_net: nn.Module,
    potential: nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
    grid: torch.Tensor,
) -> dict[str, float]:
    """Validation metrics on a fixed sample and grid.

    ``map_err`` is ||T - T*||^2_{L2(mu)}, ``grad_err`` is
    ||psi' - psi*'||^2_{L2(nu)}, ``dkr`` is the 1-D W1 distance between
    T#mu and nu from sorted samples, and ``lip_emp`` is max_grid |psi'|.
    """
    with torch.no_grad():
        transported = map_net(source)
        map_err = (transported - problem.optimal_map(source)).pow(2).mean().item()
        objective = empirical_semidual_objective(map_net, potential, source, target)
        dkr = (torch.sort(transported.squeeze())[0] - torch.sort(target.squeeze())[0]).abs().mean().item()
    grad_err = (_potential_gradient(potential, target) - problem.optimal_potential_gradient(target)).pow(2).mean().item()
    grid_gradient = _potential_gradient(potential, grid)
    gap = abs(objective - problem.optimal_value)
    return {
        "map_err": map_err,
        "grad_err": grad_err,
        "F": objective,
        "F_gap": gap,
        "dkr": dkr,
        "lip_emp": grid_gradient.abs().max().item(),
        "lip_dev": (grid_gradient - problem.optimal_potential_gradient(grid)).abs().max().item(),
        "ratio": map_err / max(gap + dkr, 1.0e-12),
    }


def run_flatness_1d(config: Mapping[str, Any], *, inner_steps: int, lr_ratio: float, seed: int) -> dict[str, list[float]]:
    """Train one (K, eta_psi / eta_T, seed) run from scratch and return its validation log.

    ``config`` holds the ``problem``, ``model``, ``training`` and ``validation``
    sections of ``configs/flatness_1d.yaml``.
    """
    problem = Gaussian1DProblem(**config["problem"])
    training = config["training"]
    validation = config["validation"]

    seed_all(seed)
    data_generator = torch.Generator().manual_seed(DATA_SEED_OFFSET + seed)
    validation_generator = torch.Generator().manual_seed(int(validation["seed"]))

    map_net = build_mlp(int(config["model"]["width"]))
    potential = build_mlp(int(config["model"]["width"]))
    map_lr = float(training["map_lr"])
    map_optimizer = torch.optim.Adam(map_net.parameters(), lr=map_lr)
    potential_optimizer = torch.optim.Adam(potential.parameters(), lr=map_lr * lr_ratio)

    val_source, val_target = problem.sample(int(validation["num_samples"]), validation_generator)
    grid = problem.grid(int(validation["grid_points"]))

    steps = int(training["potential_steps"])
    perturb_at = int(training["perturb_at"])
    perturb_std = float(training["perturb_std"])
    batch_size = int(training["batch_size"])
    log_every = int(validation["log_every"])

    log: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    map_updates = 0
    for step in range(steps + 1):
        if step == perturb_at and perturb_std > 0:
            noisy = make_noisy_potential_copy(potential, noise_scale=perturb_std, seed=PERTURB_SEED_OFFSET + seed)
            potential.load_state_dict(noisy.state_dict())
        if step % log_every == 0:
            log["step"].append(step)
            log["T_updates"].append(map_updates)
            for key, value in evaluate(problem, map_net, potential, val_source, val_target, grid).items():
                log[key].append(value)
        if step == steps:
            break
        if not (training["freeze_map_after_perturb"] and step >= perturb_at):
            for _ in range(inner_steps):
                source, target = problem.sample(batch_size, data_generator)
                map_optimizer.zero_grad()
                _semidual_value(map_net, potential, source, target).backward()
                map_optimizer.step()
                map_updates += 1
        source, target = problem.sample(batch_size, data_generator)
        potential_optimizer.zero_grad()
        with torch.no_grad():
            transported = map_net(source)
        potential_loss = -(
            0.5 * (source - transported).pow(2).mean() + potential(target).mean() - potential(transported).mean()
        )
        potential_loss.backward()
        potential_optimizer.step()
    return log
