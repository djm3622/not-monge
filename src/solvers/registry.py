"""Registry and factory for OT solver baselines."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from src.solvers.base import BaseOTSolver

SolverBuilder = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], BaseOTSolver]

SOLVER_REGISTRY: dict[str, SolverBuilder] = {}


def register_solver(name: str) -> Callable[[SolverBuilder], SolverBuilder]:
    """Decorator for registering a solver builder."""
    def decorator(builder: SolverBuilder) -> SolverBuilder:
        SOLVER_REGISTRY[name] = builder
        return builder

    return decorator


def build_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> BaseOTSolver:
    """Instantiate a solver from the registry."""
    name = str(solver_config["name"]).lower()
    if name not in SOLVER_REGISTRY:
        raise KeyError(f"Unknown solver '{name}'. Registered solvers: {sorted(SOLVER_REGISTRY)}")
    return SOLVER_REGISTRY[name](model_config, solver_config, training_config)


def registered_solver_ids() -> list[str]:
    """Return sorted registered solver ids."""
    return sorted(SOLVER_REGISTRY)
