"""Shared physical model: diffusivity profile and parameter bounds.

PDE:   u_t = d/dx [ a(x) u_x ],   a(x) = a0 * exp(beta * (2x - 1)),   x, t in [0, 1]
IC:    u(x, 0) = f(x)
Left:  u(0, t) = g(t)                                  (Dirichlet)
Right: -a(1) u_x(1, t) = h * (u(1, t) - Ta(t))         (Robin / convective)
Obs:   y = u(x_s, t) + b_s + noise,   b_ref = 0 for the reference sensor
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Admissible ranges given in the brief.
BOUNDS = {
    "a0": (0.03, 0.12),
    "beta": (-0.7, 0.7),
    "h": (0.03, 0.5),
}
OFFSET_BOUND = 0.03


@dataclass(frozen=True)
class PhysParams:
    a0: float
    beta: float
    h: float

    def to_dict(self) -> dict:
        return asdict(self)

def diffusivity(x, a0: float, beta: float):
    """a(x) = a0 exp(beta (2x - 1)). Works on numpy arrays and torch tensors."""
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return a0 * np.exp(beta * (2.0 * np.asarray(x) - 1.0))
    import torch  # local import keeps the numerical core torch-free

    return a0 * torch.exp(beta * (2.0 * x - 1.0))


def inverse_diffusivity_integral(x, a0: float, beta: float):
    """I(x) = int_0^x ds / a(s), closed form (used by the steady-state check)."""
    x = np.asarray(x, dtype=float)
    if abs(beta) < 1e-12:
        return x / a0
    return np.exp(beta) / a0 * (1.0 - np.exp(-2.0 * beta * x)) / (2.0 * beta)
