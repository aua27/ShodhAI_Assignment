"""Stage 3 scientific investigation. Every study runs on SEPARATE synthetic data (own seeds,
written under <out>/data/) with known truth, and each states its hypothesis up front.

  coverage : do the 95% Laplace intervals cover the truth? (>= 5 independent datasets; we use 20)
             + data uncertainty (noise redraws) vs Laplace sd; + AR(1)-dependent noise variant
  ablation : controlled experiments on ONE fixed dataset and ONE fixed evaluation:
             reduced temporal / sensor coverage, and separately changed modelling assumptions
  bias     : can sensor bias mask incorrect physics while temperature error stays low?
  design   : extra sensor vs equal-budget denser sampling; selection rule fixed before truth
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from .data import Experiment, load_bundle
from .inverse import FitConfig, Problem, fit, predict
from .pipeline import _as_query_experiment
from .runlog import provenance, write_json
from .synth import SynthSpec, generate, write_bundle

PHYS = ("a0", "beta", "h")


def _bundle(out: Path, name: str, spec: SynthSpec):
    tables, manifest, truth = generate(spec)
    d = out / "data" / name
    write_bundle(tables, manifest, truth, d, out / "data" / f"{name}_truth.json")
    return load_bundle(d), truth


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(c - hw), float(c + hw))


def _with_obs(exp: Experiment, mask) -> Experiment:
    return Experiment(
        exp.id, exp.forcing, exp.obs[mask].reset_index(drop=True), exp.queries
    )


def _latent_rmse(res, exps, truth):
    """Held-out latent error: model prediction vs noise-free truth at the experiment's sensor points."""
    from .physics import PhysParams
    from .solver import solve

    p = truth["params"]
    tp = PhysParams(p["a0"], p["beta"], p["h"])
    errs = []
    for e in exps:
        pr = predict(res, _as_query_experiment(e))
        u_true = solve(tp, e.forcing, nx=400, nt=1600).sample(
            pr["x"].to_numpy(), pr["t"].to_numpy()
        )
        errs.append(pr["u_latent"].to_numpy() - u_true)
    d = np.concatenate(errs)
    return float(np.sqrt(np.mean(d**2)))


def _noisy_rmse(res, exps):
    d = []
    for e in exps:
        pr = predict(res, _as_query_experiment(e))
        d.append(pr["y_pred"].to_numpy() - e.obs["y"].to_numpy())
    d = np.concatenate(d)
    return float(np.sqrt(np.mean(d**2)))


def _row(label, res, truth, val_exps):
    tp = truth["params"]
    r = {
        "variant": label,
        "status": res.status,
        "chi2_red": res.chi2_red,
        "n_obs": res.n_obs,
    }
    for k in PHYS:
        v = res.params[k]
        lo, hi = res.intervals[k]
        r[f"{k}_est"] = v
        r[f"{k}_err"] = v - tp[k]
        r[f"{k}_ci_width"] = hi - lo
        r[f"{k}_covers"] = bool(lo <= tp[k] <= hi) if hi > lo else None
    r["withheld"] = ",".join(res.withheld)
    r["offset_mae"] = float(
        np.mean([abs(res.offsets[s] - truth["offsets"][s]) for s in res.offsets])
    )
    r["val_rmse_noisy"] = _noisy_rmse(res, val_exps)
    r["val_rmse_latent"] = _latent_rmse(res, val_exps, truth)
    r["corr_a0_h"] = float(res.corr.loc["a0", "h"]) if "h" in res.corr else None
    return r


# ---------------------------------------------------------------------------------------------
def coverage(out: Path, quick=False):
    """H: if noise is independent Gaussian with the stated sd and the model is right, ~95% of the
    95% intervals contain the truth. With AR(1) noise (rho=0.6) intervals should under-cover."""
    out = Path(out) / "coverage"
    t0 = time.perf_counter()
    n_ds = 6 if quick else 20
    cfg = FitConfig(n_starts=2)
    rows = []
    for rho in (0.0, 0.6):
        for i in range(n_ds):
            b, truth = _bundle(
                out, f"cov_rho{rho}_s{i}", SynthSpec(seed=1000 + i, noise_rho=rho)
            )
            res = fit(b.subset(["E1", "E2", "E3"]), b.reference_sensor, cfg)
            r = {
                "dataset": i,
                "rho": rho,
                "status": res.status,
                "chi2_red": res.chi2_red,
            }
            for k in PHYS:
                lo, hi = res.intervals[k]
                r[f"{k}_covers"] = bool(lo <= truth["params"][k] <= hi)
                r[f"{k}_z"] = (res.params[k] - truth["params"][k]) / max(
                    res.stderr[k], 1e-12
                )
                # same interval WITHOUT the guards' widening (what plain Laplace would report)
                r[f"{k}_covers_raw"] = bool(abs(r[f"{k}_z"]) * np.sqrt(res.inflation) <= 1.959964)
            rows.append(r)
    df = pd.DataFrame(rows)
    summary = {}
    for rho, g in df.groupby("rho"):
        s = {}
        for k in PHYS:
            kk = int(g[f"{k}_covers"].sum())
            s[k] = {
                "covered_without_guard": int(g[f"{k}_covers_raw"].sum()),
                "covered": kk,
                "n": len(g),
                "rate": kk / len(g),
                "wilson95": _wilson(kk, len(g)),
                "z_sd": float(g[f"{k}_z"].std()),
            }
        summary[f"rho={rho}"] = s

    # Data uncertainty vs Laplace: same truth + inputs, 20 independent noise redraws.
    n_rep = 6 if quick else 20
    base, truth = _bundle(out, "redraw_base", SynthSpec(seed=2000))
    ests, sds = [], []
    for j in range(n_rep):
        # Fresh noise only; latent field, offsets and inputs identical across redraws.
        b = _redraw_noise(base, truth, seed=3000 + j)
        res = fit(b, base.reference_sensor, cfg)
        ests.append([res.params[k] for k in PHYS])
        sds.append([res.stderr[k] for k in PHYS])
    ests, sds = np.array(ests), np.array(sds)
    summary["noise_redraw"] = {
        k: {
            "empirical_sd_over_redraws": float(ests[:, i].std(ddof=1)),
            "mean_laplace_sd": float(sds[:, i].mean()),
        }
        for i, k in enumerate(PHYS)
    }
    summary["notes"] = (
        f"{n_ds} datasets per noise model; Wilson 95% bands on the coverage rate are wide "
        "at this sample size -- treat as a sanity check, not a calibration certificate."
    )
    df.to_csv(out / "coverage_rows.csv", index=False)
    write_json(
        out / "coverage.json",
        {
            "summary": summary,
            "provenance": provenance(
                "study:coverage", {"n": n_ds, "quick": quick}, None, t_start=t0
            ),
        },
    )
    _plot_coverage(df, out)
    print(
        pd.DataFrame(
            {
                k: {r: v[k]["rate"] for r, v in summary.items() if r.startswith("rho")}
                for k in PHYS
            }
        )
    )
    print(summary["noise_redraw"])
    return summary



def _redraw_noise(bundle, truth, seed):
    """Same latent field and offsets, new independent noise draw (data uncertainty in isolation)."""
    from .physics import PhysParams
    from .solver import solve

    rng = np.random.default_rng(seed)
    p = PhysParams(**truth["params"])
    exps = []
    for e in bundle.subset(["E1", "E2", "E3"]):
        u = solve(p, e.forcing, nx=400, nt=1600).sample(
            e.obs["x"].to_numpy(), e.obs["t"].to_numpy()
        )
        o = e.obs.copy()
        o["y"] = (
            u
            + o["sensor_id"].map(truth["offsets"]).to_numpy()
            + rng.normal(0, o["noise_sd"].to_numpy())
        )
        exps.append(Experiment(e.id, e.forcing, o, e.queries))
    return exps


def _plot_coverage(df, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("independent", df[df["rho"] == 0.0], "_covers"),
              ("AR(1), no guard", df[df["rho"] == 0.6], "_covers_raw"),
              ("AR(1), guard", df[df["rho"] == 0.6], "_covers")]
    fig, ax = plt.subplots(figsize=(5, 3))
    for j, (label, g, col) in enumerate(groups):
        ax.bar(np.arange(3) + 0.27 * j, [g[f"{k}{col}"].mean() for k in PHYS], 0.25, label=label)
    ax.axhline(0.95, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(3) + 0.27, PHYS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("95% interval coverage")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "coverage.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
ABLATION_HYPOTHESES = {
    "baseline": "reference: all sensors, all times, variable a(x), offsets fitted",
    "time_early_only": "only t <= 0.3: heat has not reached the right end -> h weakly identifiable",
    "time_sparse": "6 readings per sensor instead of 25 -> all intervals widen ~2x, estimates stay unbiased",
    "drop_right_sensors": "no sensors at x >= 0.7 -> h (and beta) lose information; a0 mostly unaffected",
    "drop_left_sensors": "no sensors at x <= 0.25 -> beta weaker (a(x) slope seen over a shorter span)",
    "assume_constant_a": "beta forced to 0 -> fit compensates via a0/h/offsets; low error can hide wrong physics",
    "assume_zero_offsets": "offsets forced to 0 -> offsets leak into a0, beta, h; chi^2 rises",
}


def ablation(out: Path, quick=False):
    out = Path(out) / "ablation"
    t0 = time.perf_counter()
    b, truth = _bundle(
        out,
        "ablation_ds",
        SynthSpec(
            seed=4242,
            truth={
                "a0": 0.07,
                "beta": 0.45,
                "h": 0.25,
                "offsets": [0.02, 0, -0.015, 0.01, -0.02, 0.025],
            },
        ),
    )
    train, val = b.subset(["E1", "E2", "E3"]), b.subset(["E4"])
    xs = b.sensors.set_index("sensor_id")["x"]
    cfg = FitConfig(n_starts=2 if quick else 4)
    variants = {
        "baseline": (train, cfg),
        "time_early_only": ([_with_obs(e, e.obs["t"] <= 0.3) for e in train], cfg),
        "time_sparse": (
            [
                _with_obs(e, e.obs.groupby("sensor_id").cumcount() % 4 == 0)
                for e in train
            ],
            cfg,
        ),
        "drop_right_sensors": ([_with_obs(e, e.obs["x"] < 0.7) for e in train], cfg),
        "drop_left_sensors": (
            [
                _with_obs(
                    e, (e.obs["x"] > 0.25) | (e.obs["sensor_id"] == b.reference_sensor)
                )
                for e in train
            ],
            cfg,
        ),
        "assume_constant_a": (train, replace(cfg, model="constant")),
        "assume_zero_offsets": (train, replace(cfg, fit_offsets=False)),
    }
    rows = []
    for name, (exps, c) in variants.items():
        res = fit(exps, b.reference_sensor, c)
        r = _row(name, res, truth, val)
        r["hypothesis"] = ABLATION_HYPOTHESES[name]
        r["notes"] = " | ".join(res.notes)
        rows.append(r)
        print(
            f"{name:22s} status={res.status:13s} "
            + " ".join(f"{k}={res.params[k]:.4f}±{res.stderr[k]:.4f}" for k in PHYS)
            + f"  val_latent_rmse={r['val_rmse_latent']:.4f}"
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation.csv", index=False)
    write_json(
        out / "ablation.json",
        {
            "truth": truth["params"],
            "rows": rows,
            "eval": "fixed: E4 held-out, latent + noisy RMSE",
            "provenance": provenance(
                "study:ablation", {"quick": quick}, b.sha256, t_start=t0
            ),
        },
    )
    _plot_ablation(df, truth, out)
    return df


def _plot_ablation(df, truth, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    y = np.arange(len(df))
    for ax, k in zip(axes, PHYS):
        est = df[f"{k}_est"].to_numpy()
        hw = df[f"{k}_ci_width"].to_numpy() / 2
        ax.errorbar(est, y, xerr=hw, fmt="o", ms=4, capsize=3)
        ax.axvline(truth["params"][k], color="k", ls="--", lw=1)
        ax.set_title(k)
    axes[0].set_yticks(y, df["variant"])
    axes[0].invert_yaxis()
    fig.suptitle("Ablation: estimate ± 95% interval (dashed = truth)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def bias(out: Path, quick=False):
    """Q: can sensor bias mask incorrect physics despite low temperature error?
    Setup: strongly graded truth (beta=0.6) with sizeable offsets. Compare the correct model with
    a wrong-physics model (constant a) that is free to use offsets, and the correct physics with
    offsets wrongly pinned to zero."""
    out = Path(out) / "bias"
    t0 = time.perf_counter()
    rows = []
    n = 3 if quick else 5
    for i in range(n):
        b, truth = _bundle(
            out,
            f"bias_{i}",
            SynthSpec(
                seed=5000 + i,
                truth={
                    "a0": 0.06,
                    "beta": 0.6,
                    "h": 0.2,
                    "offsets": [0.025, 0, -0.025, 0.02, -0.02, 0.025],
                },
            ),
        )
        train, val = b.subset(["E1", "E2", "E3"]), b.subset(["E4"])
        for label, c in (
            ("correct_model", FitConfig(n_starts=2)),
            ("wrong_physics_offsets_free", FitConfig(n_starts=2, model="constant")),
            ("correct_physics_offsets_zero", FitConfig(n_starts=2, fit_offsets=False)),
        ):
            res = fit(train, b.reference_sensor, c)
            r = _row(label, res, truth, val)
            r["dataset"] = i
            r["train_rmse_noisy"] = _noisy_rmse(res, train)
            rows.append(r)
    df = pd.DataFrame(rows)
    agg = df.groupby("variant")[
        [
            "train_rmse_noisy",
            "val_rmse_noisy",
            "val_rmse_latent",
            "a0_err",
            "beta_err",
            "h_err",
            "offset_mae",
            "chi2_red",
        ]
    ].mean()
    print(agg.round(4))
    df.to_csv(out / "bias_rows.csv", index=False)
    write_json(
        out / "bias.json",
        {
            "mean_by_variant": agg.reset_index().to_dict(orient="records"),
            "provenance": provenance("study:bias", {"n": n}, None, t_start=t0),
        },
    )
    return agg


# ---------------------------------------------------------------------------------------------
DESIGN_RULE = (
    "Frozen before any truth is consulted: from a PILOT fit on the base design (no truth), compute "
    "the linearised posterior covariance each candidate design would give at the pilot estimate "
    "(Fisher information J^T J incl. any new offset parameter). Choose the design with the smallest "
    "predicted sd of h (the parameter with the widest relative interval in the base design, and the "
    "one traded off against a0). Tie-break: smaller log-det of the (a0, beta, h) covariance."
)


def design(out: Path, quick=False):
    """Candidate A: one new sensor at x=0.95 (24 readings/experiment, unknown offset).
    Candidate B: the same 24 readings/experiment spread over existing sensors (+4 each).
    Ambiguity addressed: the a0 <-> h trade-off (heat leaving at the right end can be explained by
    stronger conduction or a stronger film coefficient) -- readings near x=1 see the Robin end directly."""
    out = Path(out) / "design"
    t0 = time.perf_counter()
    n_ds = 4 if quick else 10
    rows, decisions = [], []
    for i in range(n_ds):
        spec = SynthSpec(
            seed=6000 + i, extra_sensors=(0.95,), n_times=29, sensor_times={6: 24}
        )
        b, truth = _bundle(out, f"design_{i}", spec)
        rng = np.random.default_rng(i)
        train = b.subset(["E1", "E2", "E3"])

        def keep_base(e):
            m = e.obs["sensor_id"] != "S6"
            drop = set()
            for s, g in e.obs[m].groupby("sensor_id"):
                drop |= set(rng.choice(g.index, 4, replace=False))
            return m & ~e.obs.index.isin(drop)

        base_masks = [keep_base(e) for e in train]
        designs = {
            "base": [_with_obs(e, m) for e, m in zip(train, base_masks)],
            "A_new_sensor_x0.95": [
                _with_obs(e, m | (e.obs["sensor_id"] == "S6"))
                for e, m in zip(train, base_masks)
            ],
            "B_denser_existing": [
                _with_obs(e, e.obs["sensor_id"] != "S6") for e in train
            ],
        }
        cfg = FitConfig(n_starts=2)
        pilot = fit(designs["base"], b.reference_sensor, cfg)  # no truth used
        pred_sd = {
            name: _predicted_sd(pilot, exps, b.reference_sensor, cfg)
            for name, exps in designs.items()
        }
        choice = min(
            ("A_new_sensor_x0.95", "B_denser_existing"),
            key=lambda d: (pred_sd[d]["h"], pred_sd[d]["logdet"]),
        )
        decisions.append({"dataset": i, "chosen": choice, "predicted": pred_sd})
        for name, exps in designs.items():  # truth consulted only from here on
            res = fit(exps, b.reference_sensor, cfg)
            r = _row(name, res, truth, b.subset(["E4"]))
            r.update(
                {
                    "dataset": i,
                    "n_readings": int(sum(len(e.obs) for e in exps)),
                    "pred_sd_h": pred_sd[name]["h"],
                    "chosen_by_rule": name == choice,
                }
            )
            rows.append(r)
    df = pd.DataFrame(rows)
    agg = df.groupby("variant")[
        ["n_readings", "a0_ci_width", "beta_ci_width", "h_ci_width", "pred_sd_h"]
    ].mean()
    agg["h_abs_err"] = df.assign(e=df["h_err"].abs()).groupby("variant")["e"].mean()
    agg["a0_abs_err"] = df.assign(e=df["a0_err"].abs()).groupby("variant")["e"].mean()
    print(agg.round(5))
    print(
        "rule chose:",
        pd.Series([d["chosen"] for d in decisions]).value_counts().to_dict(),
    )
    df.to_csv(out / "design_rows.csv", index=False)
    write_json(
        out / "design.json",
        {
            "rule": DESIGN_RULE,
            "decisions": decisions,
            "mean_by_design": agg.reset_index().to_dict(orient="records"),
            "provenance": provenance("study:design", {"n": n_ds}, None, t_start=t0),
        },
    )
    return agg


def _predicted_sd(pilot, exps, reference, cfg):
    """Linearised sd each design WOULD give, evaluated at the pilot estimate (no new data, no truth)."""
    prob = Problem(exps, reference, cfg)
    n_p = len(prob.phys_names)
    theta = np.concatenate(
        [pilot.theta[:n_p], [pilot.offsets.get(s, 0.0) for s in prob.offset_sensors]]
    )
    J = prob.jacobian(theta)
    cov = np.linalg.inv(J.T @ J)
    sd = dict(zip(prob.names, np.sqrt(np.diag(cov))))
    sign, logdet = np.linalg.slogdet(cov[:n_p, :n_p])
    return {
        "a0": float(sd["a0"]),
        "beta": float(sd["beta"]),
        "h": float(sd["h"]),
        "logdet": float(logdet),
    }


def diagnosis_validation(out: Path, quick=False):
    """Frozen diagnosis rules on synthetic bundles with KNOWN, clearly-sized faults."""
    from .diagnose import validate_rules

    return validate_rules(Path(out) / "diagnosis_validation", quick=quick)


def diagnosis_sweep(out: Path, quick=False):
    """Frozen diagnosis rules on smaller faults (detection limits) + fault-free nulls."""
    from .diagnose import null_runs, sensitivity_sweep

    null_runs(Path(out) / "diagnosis_null", n=3 if quick else 10)
    return sensitivity_sweep(Path(out) / "diagnosis_sweep", replicates=1 if quick else 2)
