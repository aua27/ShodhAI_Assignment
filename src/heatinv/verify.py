"""Independent verification of the forward solver.

1. Method of manufactured solutions (MMS): pick an exact u*, derive by hand the source
   s = u*_t - d/dx(a u*_x) and the boundary data that make u* an exact solution, then
   check the solver reproduces u* and that the error falls at the expected 2nd order.
2. Steady state: with constant g, Ta and no source the solution tends to the closed form
   u(x) = g - q I(x),  I(x) = int_0^x ds/a,  q = h (g - Ta) / (1 + h I(1)).
3. Discrete energy balance: dE/dt must equal flux_in - flux_out (+ source) to round-off.

None of these use the inverse code, so they test the forward model on its own.
"""

from __future__ import annotations

import numpy as np

from .physics import PhysParams, diffusivity, inverse_diffusivity_integral
from .solver import Forcing, solve

# --- 1. manufactured solution:  u* = exp(-t) cos(pi x) + 0.5 x -----------------------------


def mms_exact(x, t):
    return np.exp(-t) * np.cos(np.pi * x) + 0.5 * x


def mms_problem(p: PhysParams, n_hist: int = 2001):
    def source(x, t):
        a = diffusivity(x, p.a0, p.beta)
        u_t = -np.exp(-t) * np.cos(np.pi * x)
        u_x = -np.pi * np.exp(-t) * np.sin(np.pi * x) + 0.5
        u_xx = -(np.pi**2) * np.exp(-t) * np.cos(np.pi * x)
        return u_t - a * (2.0 * p.beta * u_x + u_xx)  # a' = 2 beta a

    tb = np.linspace(0.0, 1.0, n_hist)
    a1 = diffusivity(1.0, p.a0, p.beta)
    g = mms_exact(0.0, tb)
    ux1 = 0.5  # u*_x(1, t): sin(pi) = 0
    Ta = mms_exact(1.0, tb) + a1 * ux1 / p.h  # from -a(1) u_x = h (u - Ta)
    x0 = np.linspace(0.0, 1.0, 4001)
    forcing = Forcing(x0=x0, f0=mms_exact(x0, 0.0), tb=tb, g=g, Ta=Ta)
    return forcing, source


def mms_error(p: PhysParams, nx: int, nt: int) -> float:
    forcing, source = mms_problem(p)
    sol = solve(p, forcing, nx=nx, nt=nt, source=source)
    X, T = np.meshgrid(sol.x[1:-1], sol.t)
    err = sol.U[:, 1:-1] - mms_exact(X, T)
    return float(np.sqrt(np.mean(err**2)))


def refinement_study(p: PhysParams, levels=(25, 50, 100, 200)) -> dict:
    """Refine space and time together (nt = 4 nx); report errors and observed orders."""
    errs = [mms_error(p, nx, 4 * nx) for nx in levels]
    orders = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
    return {
        "nx": list(levels),
        "nt": [4 * n for n in levels],
        "rmse": errs,
        "observed_order": orders,
    }


# --- 2. steady state -----------------------------------------------------------------------


def steady_state_error(p: PhysParams, g0=1.0, Ta0=0.2, nx=200) -> float:
    I1 = inverse_diffusivity_integral(1.0, p.a0, p.beta)
    q = p.h * (g0 - Ta0) / (1.0 + p.h * I1)

    def exact(x):
        return g0 - q * inverse_diffusivity_integral(x, p.a0, p.beta)

    # Start from the exact steady profile: a correct solver must leave it unchanged.
    x0 = np.linspace(0.0, 1.0, 4001)
    tb = np.array([0.0, 1.0])
    forcing = Forcing(
        x0=x0, f0=exact(x0), tb=tb, g=np.array([g0, g0]), Ta=np.array([Ta0, Ta0])
    )
    sol = solve(p, forcing, nx=nx, nt=200)
    return float(np.max(np.abs(sol.U[-1] - exact(sol.x))))


# --- 3. energy balance ---------------------------------------------------------------------


def energy_balance_max(p: PhysParams, nx=100, nt=400) -> float:
    forcing, source = mms_problem(p)
    sol = solve(p, forcing, nx=nx, nt=nt, source=source)
    # Step 0->1 is the backward-Euler start-up, whose fluxes are not trapezoidal; skip it.
    return float(np.max(np.abs(sol.energy_balance_residual()[1:])))


def run_all(p: PhysParams | None = None) -> dict:
    p = p or PhysParams(a0=0.07, beta=0.5, h=0.2)
    return {
        "params": p.to_dict(),
        "mms_refinement": refinement_study(p),
        "steady_state_max_abs_error": steady_state_error(p),
        "energy_balance_max_abs_defect": energy_balance_max(p),
    }
