"""Numerical inverse problem: weighted, bounded nonlinear least squares on the FV solver.

Unknowns  theta = [a0, beta, h, b_s (all non-reference sensors seen in training)]
Residual  r_k   = (u(x_k, t_k; a0, beta, h) + b_{s(k)} - y_k) / sigma_k
Objective 0.5 * sum r_k^2   == Gaussian negative log-likelihood with the supplied noise sd,
                               assuming independent errors (checked afterwards, see
                               residual autocorrelation in diagnostics).

Uncertainty: Gauss-Newton / Laplace approximation  Cov = (J^T J)^{-1}  at the optimum (J of the
whitened residuals). This is a local, linearised interval conditional on the model being right.
Parameters whose interval spans most of their admissible range are reported but withheld.

Baseline ("constant"): same machinery with beta fixed at 0 (spatially uniform diffusivity).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm, qmc

from .physics import BOUNDS, OFFSET_BOUND, PhysParams
from .solver import solve

FD_REL = (
    1e-4  # finite-difference step as a fraction of each parameter's admissible range
)


@dataclass
class FitConfig:
    model: str = "variable"  # "variable" (a0, beta, h) | "constant" (beta = 0)
    fit_offsets: bool = True  # False -> all offsets assumed zero (ablation)
    nx: int = 100
    nt: int = 400
    n_starts: int = 4
    seed: int = 0
    conf: float = 0.95
    withhold_frac: float = (
        0.5  # withhold an estimate whose interval covers > this frac of its range
    )
    chi2_warn: float = (
        1.5  # reduced chi^2 above this -> noise model / physics questionable
    )


@dataclass
class FitResult:
    params: dict
    offsets: dict
    stderr: dict
    intervals: dict
    withheld: list
    at_bound: list
    corr: pd.DataFrame
    chi2_red: float
    n_obs: int
    n_par: int
    status: str
    notes: list
    starts: list
    runtime_s: float
    cov: np.ndarray = field(repr=False)
    names: list = field(repr=False)
    theta: np.ndarray = field(repr=False)
    cfg: FitConfig = field(repr=False)
    reference_sensor: str = ""
    noise_sd: dict = field(default_factory=dict)  # median training noise sd per sensor

    @property
    def phys(self) -> PhysParams:
        return PhysParams(self.params["a0"], self.params["beta"], self.params["h"])

    def summary(self) -> dict:
        return {
            "model": self.cfg.model,
            "fit_offsets": self.cfg.fit_offsets,
            "status": self.status,
            "params": self.params,
            "offsets": self.offsets,
            "stderr": self.stderr,
            "intervals": {k: list(v) for k, v in self.intervals.items()},
            "interval_type": f"Gauss-Newton/Laplace (linearised) {int(self.cfg.conf * 100)}% Wald interval",
            "withheld": self.withheld,
            "at_bound": self.at_bound,
            "chi2_red": self.chi2_red,
            "n_obs": self.n_obs,
            "n_par": self.n_par,
            "notes": self.notes,
            "multistart": self.starts,
            "runtime_s": self.runtime_s,
            "corr_phys": self.corr.round(3).to_dict(),
        }


class Problem:
    """Packs/unpacks theta and evaluates whitened residuals over a set of experiments."""

    def __init__(self, experiments, reference_sensor, cfg: FitConfig):
        self.exps = experiments
        self.cfg = cfg
        self.phys_names = (
            ["a0", "h"] if cfg.model == "constant" else ["a0", "beta", "h"]
        )
        sensors = sorted({s for e in experiments for s in e.obs["sensor_id"]})
        if reference_sensor not in sensors:
            raise ValueError(
                f"reference sensor {reference_sensor} has no training observations"
            )
        self.offset_sensors = (
            [s for s in sensors if s != reference_sensor] if cfg.fit_offsets else []
        )
        self.names = self.phys_names + [f"b_{s}" for s in self.offset_sensors]
        self.lo = np.array(
            [BOUNDS[n][0] for n in self.phys_names]
            + [-OFFSET_BOUND] * len(self.offset_sensors)
        )
        self.hi = np.array(
            [BOUNDS[n][1] for n in self.phys_names]
            + [OFFSET_BOUND] * len(self.offset_sensors)
        )
        obs = pd.concat(
            [e.obs.assign(_e=i) for i, e in enumerate(experiments)], ignore_index=True
        )
        self.obs = obs
        self.y = obs["y"].to_numpy(float)
        self.w = 1.0 / obs["noise_sd"].to_numpy(float)
        col = {s: j for j, s in enumerate(self.offset_sensors)}
        self.B = np.zeros((len(obs), len(self.offset_sensors)))  # offset design matrix
        for k, s in enumerate(obs["sensor_id"]):
            if s in col:
                self.B[k, col[s]] = 1.0

    def phys_of(self, theta) -> PhysParams:
        d = dict(zip(self.phys_names, theta[: len(self.phys_names)]))
        return PhysParams(d["a0"], d.get("beta", 0.0), d["h"])

    def model_u(self, phys: PhysParams) -> np.ndarray:
        u = np.empty(len(self.obs))
        for i, e in enumerate(self.exps):
            m = (self.obs["_e"] == i).to_numpy()
            sol = solve(phys, e.forcing, nx=self.cfg.nx, nt=self.cfg.nt)
            u[m] = sol.sample(
                self.obs.loc[m, "x"].to_numpy(), self.obs.loc[m, "t"].to_numpy()
            )
        return u

    def residual(self, theta):
        b = self.B @ theta[len(self.phys_names) :]
        return (self.model_u(self.phys_of(theta)) + b - self.y) * self.w

    def jacobian(self, theta):
        """Central differences for the physical parameters; offsets enter linearly (exact)."""
        n_p = len(self.phys_names)
        J = np.empty((len(self.y), len(theta)))
        for j in range(n_p):
            step = FD_REL * (self.hi[j] - self.lo[j])
            tp, tm = theta.copy(), theta.copy()
            tp[j] = min(theta[j] + step, self.hi[j])
            tm[j] = max(theta[j] - step, self.lo[j])
            up = self.model_u(self.phys_of(tp))
            um = self.model_u(self.phys_of(tm))
            J[:, j] = (up - um) / (tp[j] - tm[j]) * self.w
        J[:, n_p:] = self.B * self.w[:, None]
        return J


def fit(experiments, reference_sensor: str, cfg: FitConfig | None = None) -> FitResult:
    cfg = cfg or FitConfig()
    t0 = time.perf_counter()
    prob = Problem(experiments, reference_sensor, cfg)
    n_p = len(prob.phys_names)
    sampler = qmc.LatinHypercube(d=n_p, seed=cfg.seed)
    starts_unit = sampler.random(cfg.n_starts)
    starts_unit[0] = 0.5  # always include the centre of the box
    runs = []
    for su in starts_unit:
        x0 = np.concatenate(
            [
                prob.lo[:n_p] + su * (prob.hi[:n_p] - prob.lo[:n_p]),
                np.zeros(len(prob.offset_sensors)),
            ]
        )
        r = least_squares(
            prob.residual,
            x0,
            jac=prob.jacobian,
            bounds=(prob.lo, prob.hi),
            x_scale=prob.hi - prob.lo,
            method="trf",
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=200,
        )
        runs.append(r)
    best = min(runs, key=lambda r: r.cost)
    starts = [
        {
            "start": dict(
                zip(
                    prob.phys_names,
                    (prob.lo[:n_p] + su * (prob.hi[:n_p] - prob.lo[:n_p]))
                    .round(4)
                    .tolist(),
                )
            ),
            "end": dict(zip(prob.phys_names, r.x[:n_p].round(5).tolist())),
            "cost": float(r.cost),
            "converged": bool(r.success),
        }
        for su, r in zip(starts_unit, runs)
    ]

    theta = best.x
    J = prob.jacobian(theta)
    n, k = J.shape
    chi2_red = float(2 * best.cost / max(n - k, 1))
    notes = []
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J)
        notes.append("J^T J singular: at least one direction is not identifiable")
    if chi2_red > cfg.chi2_warn:
        notes.append(
            f"reduced chi^2 = {chi2_red:.2f} > {cfg.chi2_warn}: misfit beyond stated noise; "
            "intervals inflated by chi^2 and physics/noise model is suspect"
        )
        cov = cov * chi2_red
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    zq = norm.ppf(0.5 + cfg.conf / 2)

    params, stderr, intervals, withheld, at_bound = {}, {}, {}, [], []
    for j, name in enumerate(prob.names):
        lo_b, hi_b = prob.lo[j], prob.hi[j]
        params[name] = float(theta[j])
        stderr[name] = float(se[j])
        intervals[name] = (float(theta[j] - zq * se[j]), float(theta[j] + zq * se[j]))
        if min(theta[j] - lo_b, hi_b - theta[j]) < 1e-3 * (hi_b - lo_b):
            at_bound.append(name)
        if j < n_p and 2 * zq * se[j] > cfg.withhold_frac * (hi_b - lo_b):
            withheld.append(name)
    if cfg.model == "constant":
        params["beta"], stderr["beta"], intervals["beta"] = 0.0, 0.0, (0.0, 0.0)
    offsets = {s: params.pop(f"b_{s}") for s in prob.offset_sensors}
    offsets[reference_sensor] = 0.0
    noise_sd = prob.obs.groupby("sensor_id")["noise_sd"].median().to_dict()

    # Distinct local minima with near-identical cost signal a flat/degenerate valley.
    good = [r for r in runs if r.cost <= best.cost * 1.01 + 1e-9]
    spread = np.ptp(np.array([r.x[:n_p] for r in good]), axis=0) / (
        prob.hi[:n_p] - prob.lo[:n_p]
    )
    if len(good) > 1 and spread.max() > 0.05:
        notes.append(
            f"multistart: {len(good)} starts reach the same cost at different parameters "
            f"(max spread {spread.max():.0%} of range) -> weak identifiability"
        )

    d = np.sqrt(np.diag(cov)[:n_p])
    corr = pd.DataFrame(
        cov[:n_p, :n_p] / np.outer(d, d), index=prob.phys_names, columns=prob.phys_names
    )

    if not best.success:
        status = "failed"
    elif withheld or at_bound or chi2_red > cfg.chi2_warn or len(notes) > 0:
        status = "weak_evidence"
    else:
        status = "ok"
    return FitResult(
        params,
        offsets,
        stderr,
        intervals,
        withheld,
        at_bound,
        corr,
        chi2_red,
        n,
        k,
        status,
        notes,
        starts,
        time.perf_counter() - t0,
        cov,
        prob.names,
        theta,
        cfg,
        reference_sensor,
        noise_sd,
    )


def predict(fitres: FitResult, experiment, conf: float | None = None) -> pd.DataFrame:
    """Predict every query of an experiment from its known inputs only (no labels needed).

    Returns latent temperature u and the noisy reading y = u + b_s with delta-method intervals:
      latent interval : parameter uncertainty only
      observation int.: + offset uncertainty + sensor noise (noise sd of that sensor if known)
    """
    cfg = fitres.cfg
    conf = conf or cfg.conf
    zq = norm.ppf(0.5 + conf / 2)
    q = experiment.queries
    if q.empty:
        return pd.DataFrame()
    xq, tq = q["x"].to_numpy(), q["t"].to_numpy()
    names_p = [n for n in fitres.names if not n.startswith("b_")]
    n_p = len(names_p)

    def u_at(vals):
        d = dict(zip(names_p, vals))
        p = PhysParams(d["a0"], d.get("beta", 0.0), d["h"])
        return solve(p, experiment.forcing, nx=cfg.nx, nt=cfg.nt).sample(xq, tq)

    th = fitres.theta[:n_p]
    u = u_at(th)
    G = np.empty((len(q), n_p))
    for j, name in enumerate(names_p):
        lo, hi = BOUNDS[name]
        step = FD_REL * (hi - lo)
        tp, tm = th.copy(), th.copy()
        tp[j] += step
        tm[j] -= step
        G[:, j] = (u_at(tp) - u_at(tm)) / (2 * step)
    cov_p = fitres.cov[:n_p, :n_p]
    var_u = np.einsum("ij,jk,ik->i", G, cov_p, G)

    b = q["sensor_id"].map(fitres.offsets).astype(float)
    var_b = np.zeros(len(q))
    for i, s in enumerate(q["sensor_id"]):
        name = f"b_{s}"
        if name in fitres.names:
            jb = fitres.names.index(name)
            g_full = np.zeros(len(fitres.names))
            g_full[:n_p] = G[i]
            g_full[jb] = 1.0
            var_b[i] = g_full @ fitres.cov @ g_full - var_u[i]
        elif s != fitres.reference_sensor:
            b.iloc[i] = (
                0.0  # offset never seen in training: unknown, assume 0 with full-range uncertainty
            )
            var_b[i] = (OFFSET_BOUND / np.sqrt(3)) ** 2
    # Unknown noise for a sensor never seen in training -> largest training sd (conservative).
    worst = max(fitres.noise_sd.values()) if fitres.noise_sd else 0.0
    noise = q["sensor_id"].map(fitres.noise_sd).fillna(worst).to_numpy(float)
    sd_u = np.sqrt(np.clip(var_u, 0, None))
    sd_y = np.sqrt(np.clip(var_u + var_b, 0, None) + noise**2)
    y = u + b.to_numpy()
    return pd.DataFrame(
        {
            "query_id": q["query_id"],
            "experiment_id": experiment.id,
            "sensor_id": q["sensor_id"],
            "x": xq,
            "t": tq,
            "u_latent": u,
            "u_lower": u - zq * sd_u,
            "u_upper": u + zq * sd_u,
            "y_pred": y,
            "y_lower": y - zq * sd_y,
            "y_upper": y + zq * sd_y,
        }
    )
