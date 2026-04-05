"""ODE integration helpers for flow-based OT baselines."""

from __future__ import annotations

from collections.abc import Callable

import torch

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
    from torchdiffeq import odeint_adjoint as torchdiffeq_odeint_adjoint
except ImportError:  # pragma: no cover - optional dependency
    torchdiffeq_odeint = None
    torchdiffeq_odeint_adjoint = None


def rk4_integrate(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Integrate an ODE with fixed-step RK4 over the provided times."""
    states = [initial_state]
    state = initial_state
    for start, end in zip(times[:-1], times[1:]):
        dt = end - start
        half_dt = 0.5 * dt
        k1 = func(state, start)
        k2 = func(state + half_dt * k1, start + half_dt)
        k3 = func(state + half_dt * k2, start + half_dt)
        k4 = func(state + dt * k3, end)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states.append(state)
    return torch.stack(states, dim=0)


def integrate_ode(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    steps: int,
    backend: str = "auto",
    method: str = "rk4",
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
    use_adjoint: bool = False,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Integrate from t=0 to t=1 using torchdiffeq when available, else RK4."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    times = torch.linspace(0.0, 1.0, steps + 1, device=initial_state.device, dtype=initial_state.dtype)
    selected_backend = backend.lower()
    can_use_torchdiffeq = selected_backend in {"auto", "torchdiffeq"} and torchdiffeq_odeint is not None
    if can_use_torchdiffeq:
        odeint_fn = (
            torchdiffeq_odeint_adjoint
            if use_adjoint and torchdiffeq_odeint_adjoint is not None
            else torchdiffeq_odeint
        )
        assert odeint_fn is not None
        trajectory = odeint_fn(
            lambda t, state: func(state, t),
            initial_state,
            times,
            method=method,
            atol=atol,
            rtol=rtol,
        )
    else:
        trajectory = rk4_integrate(func, initial_state, times)
    if return_trajectory:
        return trajectory[-1], trajectory
    return trajectory[-1]
