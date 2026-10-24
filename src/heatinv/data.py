"""Bundle loading, validation and cleaning with row-level provenance.

Bundle layout (one directory, all CSV):
  manifest.json            schema_version, synthetic flag, reference_sensor, file hashes
  sensors.csv              sensor_id, x, is_reference
  observations.csv         obs_id, experiment_id, sensor_id, t, y, noise_sd, quality_flag
  queries.csv              query_id, experiment_id, sensor_id, t        (prediction targets)
  initial_profiles.csv     experiment_id, x, u0                         (piecewise-linear in x)
  boundary_histories.csv   experiment_id, t, g, Ta                      (piecewise-linear in t)

Cleaning rules (applied in this order; every dropped row is logged with its reason):
  1. unknown experiment_id / sensor_id              -> drop  "unknown_reference"
  2. t not finite or outside [0, 1]                  -> drop  "invalid_time"
  3. y missing / not finite                          -> drop  "missing_value"
  4. noise_sd missing or <= 0                        -> drop  "invalid_noise_sd"
  5. quality_flag (case/whitespace-insensitive) == "bad" -> drop "flag_bad";
     any flag other than ok/suspect/bad               -> drop  "unknown_flag"
     quality_flag == "suspect"                       -> keep (policy configurable), logged
  6. identity key is (experiment_id, obs_id) -- IDs may restart in each experiment
     repeated key with identical content          -> keep first, drop rest "exact_duplicate"
     repeated key with conflicting content        -> drop ALL copies "conflict"
     (after rule 5, so a bad-flagged copy cannot create a conflict with a good one)
Rows are then sorted, so results do not depend on input row order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .solver import Forcing

REQUIRED = {
    "sensors": ["sensor_id", "x"],
    "observations": [
        "obs_id",
        "experiment_id",
        "sensor_id",
        "t",
        "y",
        "noise_sd",
        "quality_flag",
    ],
    "queries": ["query_id", "experiment_id", "sensor_id", "t"],
    "initial_profiles": ["experiment_id", "x", "u0"],
    "boundary_histories": ["experiment_id", "t", "g", "Ta"],
}


KNOWN_FLAGS = {"ok", "suspect", "bad"}
ID_COLS = ("experiment_id", "sensor_id", "query_id")
NUMERIC_COLS = {
    "sensors": ("x",),
    "queries": ("t",),
    "initial_profiles": ("x", "u0"),
    "boundary_histories": ("t", "g", "Ta"),
}


class BundleError(ValueError):
    """Raised for inputs that cannot be used safely; message says exactly what is wrong."""


@dataclass
class Experiment:
    id: str
    forcing: Forcing
    obs: (
        pd.DataFrame
    )  # cleaned: obs_id, sensor_id, x, t, y, noise_sd, quality_flag, source_row
    queries: pd.DataFrame  # query_id, sensor_id, x, t


@dataclass
class Bundle:
    path: Path
    manifest: dict
    sha256: str  # hash over all bundle files -> "data version" in every log
    sensors: pd.DataFrame
    reference_sensor: str
    experiments: dict
    cleaning_log: pd.DataFrame  # one row per dropped / flagged record

    def subset(self, ids) -> list:
        missing = [i for i in ids if i not in self.experiments]
        if missing:
            raise BundleError(
                f"experiments {missing} not in bundle (have {sorted(self.experiments)})"
            )
        return [self.experiments[i] for i in ids]

    def cleaning_summary(self) -> dict:
        if self.cleaning_log.empty:
            return {}
        return self.cleaning_log["reason"].value_counts().to_dict()


def _read(path: Path, name: str) -> pd.DataFrame:
    f = path / f"{name}.csv"
    if not f.exists():
        if name == "queries":
            return pd.DataFrame(columns=REQUIRED[name])
        raise BundleError(f"missing required file {f.name} in {path}")
    df = pd.read_csv(
        f,
        dtype={"obs_id": str, "query_id": str, "experiment_id": str, "sensor_id": str},
    )
    absent = [c for c in REQUIRED[name] if c not in df.columns]
    if absent:
        raise BundleError(f"{f.name}: missing columns {absent}")
    df["source_row"] = (
        np.arange(len(df)) + 2
    )  # 1-based line number in the CSV (header = line 1)
    if (
        name != "observations"
    ):  # observations are cleaned row by row instead (logged, not fatal)
        for c in ID_COLS:
            if c in df and df[c].isna().any():
                raise BundleError(
                    f"{f.name}: missing {c} on line(s) {df.loc[df[c].isna(), 'source_row'].tolist()[:5]}"
                )
        for c in NUMERIC_COLS.get(name, ()):
            v = pd.to_numeric(df[c], errors="coerce")
            if not np.isfinite(v.to_numpy(float)).all():
                rows = df.loc[~np.isfinite(v.to_numpy(float)), "source_row"].tolist()[
                    :5
                ]
                raise BundleError(
                    f"{f.name}: non-numeric / missing {c} on line(s) {rows}"
                )
            df[c] = v
    return df


def bundle_hash(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(path).glob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def clean_observations(obs, sensors, experiment_ids, keep_suspect=True):
    log = []

    def drop(mask, reason):
        nonlocal obs
        for _, r in obs[mask].iterrows():
            log.append(
                {
                    "source_row": int(r.source_row),
                    "obs_id": r.obs_id,
                    "reason": reason,
                    "action": "dropped",
                }
            )
        obs = obs[~mask]

    drop(
        ~obs["experiment_id"].isin(experiment_ids)
        | ~obs["sensor_id"].isin(sensors["sensor_id"]),
        "unknown_reference",
    )
    for c in ("t", "y", "noise_sd"):
        obs[c] = pd.to_numeric(obs[c], errors="coerce")
    drop(~np.isfinite(obs["t"]) | (obs["t"] < 0) | (obs["t"] > 1), "invalid_time")
    drop(~np.isfinite(obs["y"]), "missing_value")
    drop(~np.isfinite(obs["noise_sd"]) | (obs["noise_sd"] <= 0), "invalid_noise_sd")
    obs["quality_flag"] = (
        obs["quality_flag"].fillna("ok").astype(str).str.strip().str.lower()
    )
    flag = obs["quality_flag"]
    drop(flag == "bad", "flag_bad")
    # An unrecognised flag is not silently trusted: dropped and logged with its value.
    drop(~obs["quality_flag"].isin(KNOWN_FLAGS), "unknown_flag")
    if keep_suspect:
        for _, r in obs[obs["quality_flag"] == "suspect"].iterrows():
            log.append(
                {
                    "source_row": int(r.source_row),
                    "obs_id": r.obs_id,
                    "reason": "flag_suspect",
                    "action": "kept",
                }
            )
    else:
        drop(obs["quality_flag"] == "suspect", "flag_suspect")

    # Identity key = (experiment_id, obs_id): loggers often restart IDs in each experiment, so an
    # obs_id is only guaranteed unique within its experiment.
    key = ["experiment_id", "obs_id"]
    content = ["sensor_id", "t", "y", "noise_sd"]
    dup = obs.duplicated(key, keep=False)
    if dup.any():
        groups = obs[dup].groupby(key)
        conflict_keys = {
            k for k, g in groups if len(g[content].round(10).drop_duplicates()) > 1
        }
        drop(obs.set_index(key).index.isin(conflict_keys), "conflict")
        drop(obs.duplicated(key, keep="first"), "exact_duplicate")

    obs = obs.sort_values(["experiment_id", "sensor_id", "t", "obs_id"]).reset_index(
        drop=True
    )
    return obs, pd.DataFrame(log, columns=["source_row", "obs_id", "reason", "action"])


def load_bundle(path, keep_suspect: bool = True) -> Bundle:
    path = Path(path)
    if not path.is_dir():
        raise BundleError(f"bundle directory not found: {path}")
    manifest = (
        json.loads((path / "manifest.json").read_text())
        if (path / "manifest.json").exists()
        else {}
    )
    t = {name: _read(path, name) for name in REQUIRED}

    sensors = t["sensors"]
    if sensors["sensor_id"].duplicated().any():
        raise BundleError("sensors.csv: repeated sensor_id")
    if not sensors["x"].between(0, 1).all():
        raise BundleError("sensors.csv: sensor x outside [0, 1]")
    ref = manifest.get("reference_sensor")
    if ref is None and "is_reference" in sensors:
        refs = sensors.loc[
            sensors["is_reference"].astype(str).str.lower() == "true", "sensor_id"
        ].tolist()
        ref = refs[0] if len(refs) == 1 else None
    if ref is None or ref not in set(sensors["sensor_id"]):
        raise BundleError(
            "exactly one reference sensor (known zero offset) must be declared"
        )

    ic, bh = t["initial_profiles"], t["boundary_histories"]
    exp_ids = sorted(set(ic["experiment_id"]) & set(bh["experiment_id"]))
    for name, df in (("initial_profiles", ic), ("boundary_histories", bh)):
        extra = set(df["experiment_id"]) - set(exp_ids)
        if extra:
            raise BundleError(
                f"{name}.csv: experiments {sorted(extra)} lack the other input file"
            )

    obs, log = clean_observations(
        t["observations"].copy(), sensors, exp_ids, keep_suspect
    )
    xs = sensors.set_index("sensor_id")["x"]
    obs["x"] = obs["sensor_id"].map(xs)
    q = t["queries"].copy()
    bad_q = ~q["sensor_id"].isin(xs.index) | ~q["experiment_id"].isin(exp_ids)
    if bad_q.any():
        raise BundleError(
            f"queries.csv: {int(bad_q.sum())} queries reference unknown sensors/experiments"
        )
    q["t"] = pd.to_numeric(q["t"], errors="coerce")
    if (~np.isfinite(q["t"]) | ~q["t"].between(0, 1)).any():
        raise BundleError("queries.csv: query times must lie in [0, 1]")
    if q.duplicated(["experiment_id", "query_id"]).any():
        raise BundleError("queries.csv: repeated query_id within an experiment")
    q["x"] = q["sensor_id"].map(xs)

    experiments = {}
    for eid in exp_ids:
        ice = ic[ic["experiment_id"] == eid].sort_values("x")
        bhe = bh[bh["experiment_id"] == eid].sort_values("t")
        if len(ice) < 2 or len(bhe) < 2:
            raise BundleError(
                f"{eid}: need >= 2 samples of the initial profile and boundary history"
            )
        for name, df, cols in (
            ("initial_profiles", ice, ("x", "u0")),
            ("boundary_histories", bhe, ("t", "g", "Ta")),
        ):
            vals = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            if not np.isfinite(vals).all():
                raise BundleError(
                    f"{eid}: {name}.csv has missing / non-finite values (a NaN input would poison the solve)"
                )
        if ice["x"].duplicated().any() or bhe["t"].duplicated().any():
            raise BundleError(
                f"{eid}: repeated x in initial profile or repeated t in boundary history"
            )
        if (
            ice["x"].min() > 0
            or ice["x"].max() < 1
            or bhe["t"].min() > 0
            or bhe["t"].max() < 1
        ):
            raise BundleError(
                f"{eid}: inputs must cover x in [0,1] and t in [0,1] (no extrapolation)"
            )
        forcing = Forcing(
            x0=ice["x"].to_numpy(float),
            f0=ice["u0"].to_numpy(float),
            tb=bhe["t"].to_numpy(float),
            g=bhe["g"].to_numpy(float),
            Ta=bhe["Ta"].to_numpy(float),
        )
        experiments[eid] = Experiment(
            eid,
            forcing,
            obs[obs["experiment_id"] == eid].reset_index(drop=True),
            q[q["experiment_id"] == eid]
            .sort_values(["sensor_id", "t", "query_id"])
            .reset_index(drop=True),
        )
    return Bundle(path, manifest, bundle_hash(path), sensors, ref, experiments, log)
