"""Stage 4 model diagnosis with rules frozen in configs/diagnosis_rules.yaml.

Observations (facts about the data/fit): reduced chi^2, per-sensor residual trends in time,
parameter instability between early and late time windows.
Causal hypotheses (explanations): each is a NESTED model = base model + ONE extra parameter
    sensor_drift(s)       y gets + rate * t on sensor s           (tested for every sensor)
    heat_source(xc)       PDE gets + Q exp(-(x-xc)^2 / 2w^2)        (centre scanned)
    boundary_input_error  true left input = recorded g(t) + delta
The chi^2 drop of each nested fit (likelihood-ratio statistic, 1 extra dof) is the evidence;
the extra parameter's estimate +/- sd is the sensitivity / uncertainty of that explanation.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from .data import load_bundle
from .inverse import FD_REL, FitConfig, Problem, fit
from .runlog import provenance, write_json
from .solver import Forcing, solve


class AugProblem(Problem):
    """Base problem + one extra fault parameter appended to theta."""

    def __init__(self, exps, ref, cfg, kind, arg, bound):
        super().__init__(exps, ref, cfg)
        self.kind, self.arg = kind, arg
        self.names = self.names + [f"{kind}"]
        self.lo = np.append(self.lo, -bound)
        self.hi = np.append(self.hi, bound)
        if kind == "sensor_drift":
            self.drift_col = (self.obs["sensor_id"] == arg).to_numpy() * self.obs[
                "t"
            ].to_numpy()

    def _u(self, theta):
        phys = self.phys_of(theta)
        extra = theta[-1]
        u = np.empty(len(self.obs))
        for i, e in enumerate(self.exps):
            m = (self.obs["_e"] == i).to_numpy()
            f, src = e.forcing, None
            if self.kind == "boundary_input_error":
                f = Forcing(f.x0, f.f0, f.tb, f.g + extra, f.Ta)
            elif self.kind == "heat_source":
                xc, w = self.arg

                def src(x, t, Q=extra, xc=xc, w=w):
                    return Q * np.exp(-((x - xc) ** 2) / (2 * w * w))

            sol = solve(phys, f, nx=self.cfg.nx, nt=self.cfg.nt, source=src)
            u[m] = sol.sample(
                self.obs.loc[m, "x"].to_numpy(), self.obs.loc[m, "t"].to_numpy()
            )
        if self.kind == "sensor_drift":
            u = u + extra * self.drift_col
        return u

    def residual(self, theta):
        n_p = len(self.phys_names)
        b = self.B @ theta[n_p:-1]
        return (self._u(theta) + b - self.y) * self.w

    def jacobian(self, theta):
        n_p = len(self.phys_names)
        J = np.empty((len(self.y), len(theta)))
        for j in list(range(n_p)) + [len(theta) - 1]:
            step = FD_REL * (self.hi[j] - self.lo[j])
            tp, tm = theta.copy(), theta.copy()
            tp[j] = min(theta[j] + step, self.hi[j])
            tm[j] = max(theta[j] - step, self.lo[j])
            J[:, j] = (self._u(tp) - self._u(tm)) / (tp[j] - tm[j]) * self.w
        J[:, n_p:-1] = self.B * self.w[:, None]
        return J


def _fit_aug(exps, ref, cfg, base, kind, arg, bound):
    prob = AugProblem(exps, ref, cfg, kind, arg, bound)
    n_p = len(prob.phys_names)
    x0 = np.concatenate(
        [
            base.theta[:n_p],
            [base.offsets.get(s, 0.0) for s in prob.offset_sensors],
            [0.0],
        ]
    )
    x0 = np.clip(x0, prob.lo + 1e-9, prob.hi - 1e-9)
    r = least_squares(
        prob.residual,
        x0,
        jac=prob.jacobian,
        bounds=(prob.lo, prob.hi),
        x_scale=prob.hi - prob.lo,
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
        max_nfev=100,
    )
    J = prob.jacobian(r.x)
    cov = np.linalg.pinv(J.T @ J)
    return {
        "chi2": float(2 * r.cost),
        "estimate": float(r.x[-1]),
        "sd": float(np.sqrt(max(cov[-1, -1], 0))),
        "phys": dict(zip(prob.phys_names, map(float, r.x[:n_p]))),
    }


def decide(ranked, misfit: bool, rules) -> str:
    """Frozen decision rule (configs/diagnosis_rules.yaml). `ranked`: best test per fault kind,
    sorted by delta_chi2 descending."""
    lrt, margin = rules["lrt_threshold"], rules["min_margin"]
    top = ranked[0]
    second = ranked[1]["delta_chi2"] if len(ranked) > 1 else 0.0
    if top["delta_chi2"] < lrt:
        return "unexplained_misfit" if misfit else "no_fault_detected"
    if top["delta_chi2"] - second >= margin:
        return f"suspected:{top['kind']}"
    return "ambiguous:" + "|".join(
        t["kind"] for t in ranked if top["delta_chi2"] - t["delta_chi2"] < margin
    )


def _residual_table(res, exps, ref, cfg):
    prob = Problem(exps, ref, cfg)
    n_p = len(prob.phys_names)
    theta = np.concatenate(
        [res.theta[:n_p], [res.offsets.get(s, 0.0) for s in prob.offset_sensors]]
    )
    df = prob.obs[["experiment_id", "sensor_id", "x", "t"]].copy()
    df["z"] = prob.residual(theta)  # standardised residual (model - data) / sigma
    return df


def diagnose(bundle, rules: dict, quick=False) -> dict:
    cfg = FitConfig(n_starts=1 if quick else 2)
    exps = bundle.subset(
        [
            e
            for e in rules["fit_experiments"]
            if e in bundle.experiments and len(bundle.experiments[e].obs) > 0
        ]
    )
    ref = bundle.reference_sensor
    base = fit(exps, ref, cfg)
    chi2_base = base.chi2_red * (base.n_obs - base.n_par)

    # ---- observations ----
    resid = _residual_table(base, exps, ref, cfg)
    trends = {}
    for s, g in resid.groupby("sensor_id"):
        tt = g["t"].to_numpy() - g["t"].mean()
        slope = float(np.sum(tt * g["z"]) / np.sum(tt * tt))
        se = float(1.0 / np.sqrt(np.sum(tt * tt)))
        trends[s] = {
            "slope": slope,
            "z": slope / se,
            "mean_abs_z": float(g["z"].abs().mean()),
            "x": float(g["x"].iloc[0]),
        }
    split = rules["supporting_checks"]["window_split_t"]
    windows = {}
    try:
        early = fit(
            [_mask(e, e.obs["t"] <= split) for e in exps], ref, replace(cfg, n_starts=1)
        )
        late = fit(
            [_mask(e, e.obs["t"] > split) for e in exps], ref, replace(cfg, n_starts=1)
        )
        for k in ("a0", "beta", "h"):
            d = early.params[k] - late.params[k]
            windows[k] = {
                "early": early.params[k],
                "late": late.params[k],
                "z": float(d / np.hypot(early.stderr[k], late.stderr[k])),
            }
    except Exception as ex:  # a window can be too thin to fit; that is itself reported
        windows["error"] = str(ex)
    observations = {
        "chi2_red_base": base.chi2_red,
        "misfit_beyond_noise": base.chi2_red > rules["chi2_red_misfit"],
        "base_params": base.params,
        "base_status": base.status,
        "residual_trend_by_sensor": trends,
        "sensors_with_time_trend": [
            s
            for s, v in trends.items()
            if abs(v["z"]) > rules["supporting_checks"]["residual_trend_z"]
        ],
        "window_parameter_shift": windows,
    }

    # ---- causal hypotheses (nested 1-parameter models) ----
    H = rules["hypotheses"]
    tests = []
    for s in sorted(set(resid["sensor_id"])):
        r = _fit_aug(
            exps, ref, cfg, base, "sensor_drift", s, H["sensor_drift"]["rate_bound"]
        )
        tests.append({"kind": "sensor_drift", "where": s, **r})
    for xc in H["heat_source"]["centres"]:
        r = _fit_aug(
            exps,
            ref,
            cfg,
            base,
            "heat_source",
            (xc, H["heat_source"]["width"]),
            H["heat_source"]["q_bound"],
        )
        tests.append({"kind": "heat_source", "where": f"xc={xc}", **r})
    r = _fit_aug(
        exps,
        ref,
        cfg,
        base,
        "boundary_input_error",
        None,
        H["boundary_input_error"]["bias_bound"],
    )
    tests.append({"kind": "boundary_input_error", "where": "x=0", **r})
    for t in tests:
        t["delta_chi2"] = chi2_base - t["chi2"]

    # best per kind, then rank kinds
    best_by_kind = {}
    for t in tests:
        if (
            t["kind"] not in best_by_kind
            or t["delta_chi2"] > best_by_kind[t["kind"]]["delta_chi2"]
        ):
            best_by_kind[t["kind"]] = t
    ranked = sorted(best_by_kind.values(), key=lambda t: -t["delta_chi2"])
    top = ranked[0]
    verdict = decide(ranked, observations["misfit_beyond_noise"], rules)
    status = "ok" if verdict == "no_fault_detected" else "weak_evidence"
    return {
        "verdict": verdict,
        "status": status,
        "verdict_is": "a causal hypothesis consistent with the data, not an observed fact",
        "top_hypothesis": {
            k: top[k] for k in ("kind", "where", "delta_chi2", "estimate", "sd")
        },
        "observations": observations,
        "hypothesis_tests": tests,
        "physics_estimates_trustworthy": status == "ok",
    }


def _mask(e, m):
    from .data import Experiment

    return Experiment(e.id, e.forcing, e.obs[m].reset_index(drop=True), e.queries)


def diagnose_bundle(bundle_path, rules_path, out: Path, train_ids=None, quick=False):
    t0 = time.perf_counter()
    rules = yaml.safe_load(Path(rules_path).read_text())
    if train_ids:
        rules["fit_experiments"] = list(train_ids)
    b = load_bundle(bundle_path)
    r = diagnose(b, rules, quick=quick)
    r["provenance"] = provenance("diagnose", rules, b.sha256, t_start=t0)
    out = Path(out)
    write_json(out / "diagnosis.json", r)
    print(
        "verdict:",
        r["verdict"],
        "| top:",
        {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in r["top_hypothesis"].items()
        },
    )
    print(
        "chi2_red base:",
        round(r["observations"]["chi2_red_base"], 3),
        "| sensors with time trend:",
        r["observations"]["sensors_with_time_trend"],
    )
    return r


def validate_rules(
    out: Path, rules_path="configs/diagnosis_rules.yaml", n_per_kind=3, quick=False
):
    """Run the frozen rules on synthetic bundles with KNOWN faults -> confusion table."""
    from .synth import SynthSpec, generate, write_bundle

    rules = yaml.safe_load(Path(rules_path).read_text())
    faults = {
        "none": {"kind": "none"},
        "sensor_drift": {"kind": "drift", "sensor": "S4", "rate": 0.06},
        "heat_source": {"kind": "heat_source", "Q": 0.4, "xc": 0.6, "w": 0.1},
        "boundary_input_error": {"kind": "boundary_error", "bias": 0.06},
    }
    rows = []
    for kind, fault in faults.items():
        for i in range(1 if quick else n_per_kind):
            tables, man, truth = generate(
                SynthSpec(
                    seed=7000 + 10 * i + len(rows), fault=fault, unlabeled=("E5",)
                )
            )
            d = Path(out) / "data" / f"{kind}_{i}"
            write_bundle(
                tables, man, truth, d, Path(out) / "data" / f"{kind}_{i}_truth.json"
            )
            r = diagnose(load_bundle(d), rules, quick=quick)
            rows.append(
                {
                    "true_fault": kind,
                    "replicate": i,
                    "verdict": r["verdict"],
                    "top_delta_chi2": r["top_hypothesis"]["delta_chi2"],
                    "chi2_red_base": r["observations"]["chi2_red_base"],
                }
            )
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / "rule_validation.csv", index=False)
    return df


SWEEP = {
    "sensor_drift": [
        {"kind": "drift", "sensor": "S4", "rate": r} for r in (0.01, 0.02, 0.04)
    ]
    + [{"kind": "drift", "sensor": "S1", "rate": 0.03}],  # S1 = reference sensor
    "heat_source": [
        {"kind": "heat_source", "Q": q, "xc": 0.6, "w": 0.1} for q in (0.03, 0.08, 0.2)
    ],
    "boundary_input_error": [
        {"kind": "boundary_error", "bias": b} for b in (0.005, 0.01, 0.02, 0.04)
    ],
}


def sensitivity_sweep(
    out: Path, rules_path="configs/diagnosis_rules.yaml", replicates=2
):
    """Where do the frozen rules stop detecting? Smaller faults than validate_rules, known truth."""
    from .synth import SynthSpec, generate, write_bundle

    rules = yaml.safe_load(Path(rules_path).read_text())
    rows = []
    for kind, faults in SWEEP.items():
        for f in faults:
            for i in range(replicates):
                seed = 8000 + len(rows)
                tables, man, truth = generate(SynthSpec(seed=seed, fault=f))
                d = Path(out) / "data" / f"sweep_{seed}"
                write_bundle(
                    tables,
                    man,
                    truth,
                    d,
                    Path(out) / "data" / f"sweep_{seed}_truth.json",
                )
                r = diagnose(load_bundle(d), rules, quick=True)
                size = {k: v for k, v in f.items() if k not in ("kind", "xc", "w")}
                rows.append(
                    {
                        "true_fault": kind,
                        **size,
                        "replicate": i,
                        "verdict": r["verdict"],
                        "correct": r["verdict"] == f"suspected:{kind}",
                        "top_delta_chi2": r["top_hypothesis"]["delta_chi2"],
                        "chi2_red_base": r["observations"]["chi2_red_base"],
                        "base_params": r["observations"]["base_params"],
                    }
                )
                print({k: v for k, v in rows[-1].items() if k != "base_params"})
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / "sensitivity_sweep.csv", index=False)
    return df


def null_runs(out: Path, rules_path="configs/diagnosis_rules.yaml", n=10):
    """Fault-free bundles: the largest delta chi^2 under the null sets the false-alarm scale."""
    from .synth import SynthSpec, generate, write_bundle

    rules = yaml.safe_load(Path(rules_path).read_text())
    rows = []
    for i in range(n):
        tables, man, truth = generate(SynthSpec(seed=9000 + i, fault={"kind": "none"}))
        d = Path(out) / "data" / f"null_{i}"
        write_bundle(tables, man, truth, d, Path(out) / "data" / f"null_{i}_truth.json")
        r = diagnose(load_bundle(d), rules, quick=True)
        t = r["top_hypothesis"]
        rows.append({"i": i, "verdict": r["verdict"], "top_kind": t["kind"], "where": t["where"],
                     "top_delta_chi2": t["delta_chi2"], "chi2_red": r["observations"]["chi2_red_base"]})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / "null_tops.csv", index=False)
    return df
