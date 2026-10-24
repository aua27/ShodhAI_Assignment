"""Cell-centred finite-volume reference solver (Crank-Nicolson in time).

Spatial discretisation (N cells of width dx, centres x_i = (i + 1/2) dx):
    dx * du_i/dt = k_{i-1/2} (u_{i-1} - u_i) + k_{i+1/2} (u_{i+1} - u_i) + dx * s(x_i, t)
with face conductances k_{i+1/2} = a(x_{i+1/2}) / dx, using the exact a(x) at each face.

Boundaries:
    left  face: Dirichlet g(t) reached through a half cell -> k_L = a(0) / (dx/2)
    right face: half-cell conduction in series with the Robin film coefficient h
                -> k_R = 1 / (dx / (2 a(1)) + 1 / h), driving u towards Ta(t)

This is conservative by construction: what leaves one cell enters its neighbour, so the
discrete energy  E = sum(u_i dx)  changes only through the two boundary fluxes (+ source).

Time: Crank-Nicolson (2nd order) with Rannacher start-up (first step replaced by four
backward-Euler quarter steps) to damp the oscillations CN produces when f(0) != g(0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import splu

from .physics import PhysParams, diffusivity

SourceFn = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class Forcing:
    """Known inputs of one experiment, as sampled histories (piecewise-linear between samples)."""

    x0: np.ndarray  # initial-profile sample locations
    f0: np.ndarray  # initial-profile values
    tb: np.ndarray  # boundary-history sample times
    g: np.ndarray  # left boundary temperature u(0, t)
    Ta: np.ndarray  # ambient temperature at the right end

    def f_at(self, x):
        return np.interp(x, self.x0, self.f0)

    def g_at(self, t):
        return np.interp(t, self.tb, self.g)

    def Ta_at(self, t):
        return np.interp(t, self.tb, self.Ta)


@dataclass
class Solution:
    t: np.ndarray  # (nt+1,)
    x: np.ndarray  # (N+2,) nodes: 0, cell centres, 1
    U: np.ndarray  # (nt+1, N+2) temperatures at those nodes
    flux_in: np.ndarray  # (nt+1,) heat flux entering at x=0  (= -a u_x at x=0)
    flux_out: np.ndarray  # (nt+1,) heat flux leaving at x=1 (= h (u(1)-Ta))
    energy: np.ndarray  # (nt+1,) sum(u_i dx) over cells
    source_int: np.ndarray  # (nt+1,) integral of the source over the rod
    dt: float

    def sample(self, xq, tq) -> np.ndarray:
        """Bilinear (piecewise-linear in x and t) interpolation at paired points (xq[k], tq[k])."""
        interp = RegularGridInterpolator((self.t, self.x), self.U, method="linear")
        pts = np.column_stack([np.clip(tq, 0.0, 1.0), np.clip(xq, 0.0, 1.0)])
        return interp(pts)

    def energy_balance_residual(self) -> np.ndarray:
        """Per-step defect of  dE = dt * (avg flux_in - avg flux_out + avg source).

        For the solver itself this should sit at round-off; it is reused as a physical
        diagnostic for any other field (e.g. the PINN) evaluated on the same grid.
        """
        dE = np.diff(self.energy)
        net = 0.5 * (self.flux_in[1:] + self.flux_in[:-1]) - 0.5 * (
            self.flux_out[1:] + self.flux_out[:-1]
        )
        net += 0.5 * (self.source_int[1:] + self.source_int[:-1])
        return dE - self.dt * net


def _operator(params: PhysParams, nx: int):
    dx = 1.0 / nx
    xc = (np.arange(nx) + 0.5) * dx
    xf = np.arange(nx + 1) * dx  # faces
    a_f = diffusivity(xf, params.a0, params.beta)
    k = a_f[1:-1] / dx  # interior faces, length nx-1
    kL = a_f[0] / (0.5 * dx)
    kR = 1.0 / (0.5 * dx / a_f[-1] + 1.0 / params.h)

    main = np.zeros(nx)
    main[:-1] -= k
    main[1:] -= k
    main[0] -= kL
    main[-1] -= kR
    A = sp.diags([k / dx, main / dx, k / dx], [-1, 0, 1], format="csc")
    return A, xc, dx, kL, kR, a_f


def solve(
    params: PhysParams,
    forcing: Forcing,
    nx: int = 100,
    nt: int = 400,
    source: Optional[SourceFn] = None,
) -> Solution:
    A, xc, dx, kL, kR, a_f = _operator(params, nx)
    dt = 1.0 / nt
    t = np.linspace(0.0, 1.0, nt + 1)
    eye = sp.identity(nx, format="csc")

    def bvec(tt):
        c = np.zeros(nx)
        c[0] = kL * forcing.g_at(tt) / dx
        c[-1] = kR * forcing.Ta_at(tt) / dx
        if source is not None:
            c += source(xc, tt)
        return c

    u = forcing.f_at(xc).astype(float)
    Ucells = np.empty((nt + 1, nx))
    Ucells[0] = u

    # Rannacher start-up: four backward-Euler quarter steps.
    sub = dt / 4.0
    be = splu((eye - sub * A).tocsc())
    for j in range(1, 5):
        u = be.solve(u + sub * bvec(j * sub))
    Ucells[1] = u

    lhs = splu((eye - 0.5 * dt * A).tocsc())
    rhs_op = (eye + 0.5 * dt * A).tocsr()
    c_prev = bvec(t[1])
    for n in range(1, nt):
        c_next = bvec(t[n + 1])
        u = lhs.solve(rhs_op @ u + 0.5 * dt * (c_prev + c_next))
        Ucells[n + 1] = u
        c_prev = c_next

    g = forcing.g_at(t)
    Ta = forcing.Ta_at(t)
    # Right boundary value from the Robin condition (continuity of flux through the half cell).
    two_a_dx = a_f[-1] / (0.5 * dx)
    u_right = (two_a_dx * Ucells[:, -1] + params.h * Ta) / (two_a_dx + params.h)
    U = np.column_stack([g, Ucells, u_right])
    x = np.concatenate([[0.0], xc, [1.0]])

    flux_in = kL * (g - Ucells[:, 0])
    flux_out = kR * (Ucells[:, -1] - Ta)
    energy = Ucells.sum(axis=1) * dx
    if source is not None:
        source_int = np.array([source(xc, tt).sum() * dx for tt in t])
    else:
        source_int = np.zeros_like(t)
    return Solution(t, x, U, flux_in, flux_out, energy, source_int, dt)
