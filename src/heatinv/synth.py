"""Synthetic data bundles (clearly labelled synthetic).

A bundle follows the schema documented in README.md / data.py. Ground truth is written to a
SEPARATE truth file that the fitting/prediction code never reads; only `evaluate` and the
synthetic studies use it, and only to score results.

To avoid the "inverse crime" (generating and fitting with the same discretisation, which
hides discretisation error), data are generated on a 4x finer grid than inference uses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .physics import BOUNDS, OFFSET_BOUND, PhysParams
from .solver import Forcing, solve

GEN_NX, GEN_NT = 400, 1600
SCHEMA_VERSION = "1.0"


@dataclass
class SynthSpec:
    seed: int = 0
    n_experiments: int = 5
    sensor_x: tuple = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)
    reference_index: int = 1  # sensor with known zero offset
    n_times: int = 25  # observation times per sensor per experiment
    noise_sd: tuple = (
        0.008,
        0.015,
    )  # per-sensor noise sd drawn uniformly from this range
    unlabeled: tuple = ("E5",)  # experiments released without temperature labels
    defects: bool = False  # inject missing/duplicate/conflicting/flagged rows
    truth: Optional[dict] = None  # {"a0","beta","h","offsets":[...]} or None -> random
    fault: Optional[dict] = None  # Stage 4 faults, see make_fault()
    extra_sensors: tuple = ()  # (x, ...) appended after sensor_x (design studies)
    noise_rho: float = 0.0  # AR(1) correlation between consecutive readings of a sensor
    sensor_times: Optional[dict] = (
        None  # {sensor_index: n_times} overrides n_times (design studies)
    )
    meta: dict = field(default_factory=dict)


def random_truth(rng, n_sensors: int, reference_index: int) -> dict:
    """Draw parameters from the inner 80% of each admissible range."""

    def inner(lo, hi):
        pad = 0.1 * (hi - lo)
        return float(rng.uniform(lo + pad, hi - pad))

    offsets = rng.uniform(-0.8 * OFFSET_BOUND, 0.8 * OFFSET_BOUND, n_sensors)
    offsets[reference_index] = 0.0
    return {
        "a0": inner(*BOUNDS["a0"]),
        "beta": inner(*BOUNDS["beta"]),
        "h": inner(*BOUNDS["h"]),
        "offsets": [float(o) for o in offsets],
    }


def random_forcing(rng) -> Forcing:
    """Smooth, varied inputs: oscillating/ramping left input, drifting ambient, curved initial profile."""
    tb = np.linspace(0.0, 1.0, 101)
    g_base = rng.uniform(0.6, 1.0)
    g = g_base + rng.uniform(0.05, 0.25) * np.sin(
        2 * np.pi * rng.integers(1, 3) * tb + rng.uniform(0, 2 * np.pi)
    )
    g += rng.uniform(-0.2, 0.2) * tb
    Ta0 = rng.uniform(0.0, 0.3)
    Ta = Ta0 + rng.uniform(-0.1, 0.2) * tb
    x0 = np.linspace(0.0, 1.0, 51)
    u_far = rng.uniform(0.1, 0.5)
    shape = x0 ** rng.uniform(0.5, 2.0)
    f0 = g[0] + (u_far - g[0]) * shape + rng.uniform(-0.1, 0.1) * np.sin(np.pi * x0)
    return Forcing(x0=x0, f0=f0, tb=tb, g=g, Ta=Ta)


def make_fault(fault: dict, sensor_ids: list):
    """Return (source_fn, obs_transform, recorded_g_transform) for a Stage-4 fault.

    kinds:
      drift            : one sensor's offset grows linearly in time  (rate per unit t)
      heat_source      : un-modelled Gaussian source Q exp(-(x-xc)^2 / (2 w^2))
      boundary_error   : the RECORDED left input differs from the true one by a bias
      none             : clean control
    """
    kind = fault.get("kind", "none")
    source = None
    obs_tf = None
    g_tf = None
    if kind == "drift":
        sid = fault.get("sensor", sensor_ids[-1])
        rate = fault.get("rate", 0.08)

        def obs_tf(df):
            m = df["sensor_id"] == sid
            df.loc[m, "y"] += rate * df.loc[m, "t"]
            return df
    elif kind == "heat_source":
        Q, xc, w = fault.get("Q", 0.5), fault.get("xc", 0.6), fault.get("w", 0.08)

        def source(x, t):
            return Q * np.exp(-((x - xc) ** 2) / (2 * w**2)) * np.ones_like(x)
    elif kind == "boundary_error":
        bias = fault.get("bias", 0.08)

        def g_tf(g):
            return g + bias
    elif kind != "none":
        raise ValueError(f"unknown fault kind {kind!r}")
    return source, obs_tf, g_tf


def _noise(rng, sd, t, rho):
    """Gaussian noise; rho > 0 gives AR(1) dependence along each sensor's time-ordered series."""
    z = rng.normal(0.0, sd, t.size)
    if rho <= 0:
        return z
    order = np.argsort(t)
    e = np.empty(t.size)
    prev = 0.0
    for i, k in enumerate(order):
        prev = z[k] if i == 0 else rho * prev + np.sqrt(1 - rho**2) * z[k]
        e[k] = prev
    return e


def _times(rng, n):
    base = (np.arange(n) + 0.5) / n
    return np.clip(base + rng.uniform(-0.4, 0.4, n) / n, 1e-3, 1.0)


def generate(spec: SynthSpec):
    """Build a bundle in memory. Returns (tables: dict[str, DataFrame], manifest, truth)."""
    rng = np.random.default_rng(spec.seed)
    xs = list(spec.sensor_x) + list(spec.extra_sensors)
    sensor_ids = [f"S{i}" for i in range(len(xs))]
    truth = spec.truth or random_truth(rng, len(xs), spec.reference_index)
    truth = dict(truth)
    truth["offsets"] = list(truth["offsets"])[: len(xs)]
    while len(truth["offsets"]) < len(xs):
        truth["offsets"].append(float(rng.uniform(-0.8, 0.8) * OFFSET_BOUND))
    truth["offsets"][spec.reference_index] = 0.0
    p = PhysParams(truth["a0"], truth["beta"], truth["h"])
    sds = rng.uniform(*spec.noise_sd, len(xs))

    source, obs_tf, g_tf = make_fault(spec.fault or {"kind": "none"}, sensor_ids)

    sensors = pd.DataFrame(
        {
            "sensor_id": sensor_ids,
            "x": xs,
            "is_reference": [i == spec.reference_index for i in range(len(xs))],
        }
    )
    obs_rows, q_rows, label_rows, ic_rows, bh_rows = [], [], [], [], []
    for e in range(spec.n_experiments):
        eid = f"E{e + 1}"
        fr = random_forcing(rng)
        sol = solve(p, fr, nx=GEN_NX, nt=GEN_NT, source=source)
        g_recorded = g_tf(fr.g) if g_tf else fr.g
        ic_rows += [
            {"experiment_id": eid, "x": x, "u0": u} for x, u in zip(fr.x0, fr.f0)
        ]
        bh_rows += [
            {"experiment_id": eid, "t": t, "g": g, "Ta": ta}
            for t, g, ta in zip(fr.tb, g_recorded, fr.Ta)
        ]
        for sid, x, b, sd in zip(sensor_ids, xs, truth["offsets"], sds):
            n_t = (spec.sensor_times or {}).get(sensor_ids.index(sid), spec.n_times)
            t = _times(rng, n_t)
            u = sol.sample(np.full_like(t, x), t)
            y = u + b + _noise(rng, sd, t, spec.noise_rho)
            for k in range(t.size):
                row = {
                    "experiment_id": eid,
                    "sensor_id": sid,
                    "t": float(t[k]),
                    "y": float(y[k]),
                    "noise_sd": float(sd),
                    "quality_flag": "ok",
                    "u_true": float(u[k]),
                }
                if eid in spec.unlabeled:
                    qid = f"Q-{eid}-{sid}-{k:03d}"
                    q_rows.append(
                        {
                            "query_id": qid,
                            "experiment_id": eid,
                            "sensor_id": sid,
                            "t": row["t"],
                        }
                    )
                    label_rows.append(
                        {
                            "query_id": qid,
                            "y_noisy": row["y"],
                            "u_latent": row["u_true"],
                        }
                    )
                else:
                    obs_rows.append(row)

    obs = pd.DataFrame(obs_rows)
    if obs_tf is not None:
        obs = obs_tf(obs)
    obs = obs.drop(columns=["u_true"])
    obs.insert(0, "obs_id", [f"O{i:05d}" for i in range(len(obs))])
    if spec.defects:
        obs = inject_defects(obs, rng)

    tables = {
        "sensors": sensors,
        "observations": obs,
        "queries": pd.DataFrame(
            q_rows, columns=["query_id", "experiment_id", "sensor_id", "t"]
        ),
        "initial_profiles": pd.DataFrame(ic_rows),
        "boundary_histories": pd.DataFrame(bh_rows),
    }
    truth_doc = {
        "synthetic": True,
        "params": {"a0": p.a0, "beta": p.beta, "h": p.h},
        "offsets": dict(zip(sensor_ids, truth["offsets"])),
        "noise_sd": dict(zip(sensor_ids, map(float, sds))),
        "fault": spec.fault or {"kind": "none"},
        "labels": label_rows,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "generator": {
            "seed": spec.seed,
            "grid": [GEN_NX, GEN_NT],
            "defects": spec.defects,
            **spec.meta,
        },
        "experiments": sorted(set(obs["experiment_id"]) | set(spec.unlabeled)),
        "reference_sensor": sensor_ids[spec.reference_index],
    }
    return tables, manifest, truth_doc


def inject_defects(obs: pd.DataFrame, rng) -> pd.DataFrame:
    """Plant the record problems the brief asks us to handle, then shuffle row order."""
    obs = obs.copy()
    n = len(obs)
    idx = rng.permutation(n)
    i = 0

    def take(k):
        nonlocal i
        out = idx[i : i + k]
        i += k
        return out

    obs = obs.drop(index=take(int(0.03 * n)))  # missing records (rows simply absent)
    obs.loc[take(5), "y"] = np.nan  # missing values
    obs.loc[take(4), "quality_flag"] = "bad"
    bad_gross = take(3)  # gross outliers, flagged bad by the "instrument"
    obs.loc[bad_gross, "y"] += 0.5
    obs.loc[bad_gross, "quality_flag"] = "bad"
    obs.loc[take(6), "quality_flag"] = "suspect"
    obs.loc[take(2), "t"] = 1.2  # out-of-domain time stamp
    exact = obs.loc[take(6)].copy()  # repeated IDs, identical content
    conflict = obs.loc[take(4)].copy()  # repeated IDs, conflicting values
    conflict["y"] += rng.choice([-1, 1], len(conflict)) * 0.2
    flagged_conflict = obs.loc[take(3)].copy()  # conflict where the copy is flagged bad
    flagged_conflict["y"] += 0.3
    flagged_conflict["quality_flag"] = "bad"
    obs = pd.concat([obs, exact, conflict, flagged_conflict])
    return obs.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(
        drop=True
    )


def write_bundle(
    tables, manifest, truth, out_dir: Path, truth_path: Optional[Path] = None
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False, float_format="%.8g")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(manifest, files=hashes)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if truth_path is not None:
        truth_path = Path(truth_path)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(json.dumps(truth, indent=2))
    return out_dir
