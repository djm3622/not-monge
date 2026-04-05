"""Optimization solvers."""

import src.solvers.advanced_baselines  # noqa: F401
import src.solvers.benchmark_baselines  # noqa: F401
import src.solvers.icnn_baseline  # noqa: F401
import src.solvers.minimax_ot  # noqa: F401
import src.solvers.modern_baselines  # noqa: F401
import src.solvers.reference_baselines  # noqa: F401

from src.solvers.base import BaseOTSolver, OTSolver, ReferenceOTSolver
from src.solvers.registry import build_solver, registered_solver_ids

__all__ = [
    "BaseOTSolver",
    "OTSolver",
    "ReferenceOTSolver",
    "build_solver",
    "registered_solver_ids",
]
