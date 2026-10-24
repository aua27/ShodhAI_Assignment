"""Second testing round: hidden-evaluation simulation, end-to-end determinism, loader fuzzing."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from heatinv.data import BundleError, load_bundle
from heatinv.synth import SynthSpec, generate, write_bundle


@pytest.fixture(scope="module")
def odd_bundle(tmp_path_factory):
    """Different shape from anything used in development: 7 experiments, 8 sensors at new
    positions, reference not first, uneven reading counts, IDs that restart per experiment."""
    d = tmp_path_factory.mktemp("odd")
    spec = SynthSpec(
        seed=77,
        n_experiments=7,
        sensor_x=(0.05, 0.2, 0.33, 0.47, 0.6, 0.72, 0.88, 0.97),
        reference_index=5,
        n_times=17,
        unlabeled=("E7",),
        defects=True,
    )
    tables, man, truth = generate(spec)
    obs = tables["observations"]
    obs["obs_id"] = (
        obs.groupby("experiment_id").cumcount().astype(str)
    )  # restart per experiment
    obs["experiment_id"] = obs["experiment_id"].map(lambda e: f"run_{e}")
    for k in ("queries", "initial_profiles", "boundary_histories"):
        tables[k]["experiment_id"] = tables[k]["experiment_id"].map(
            lambda e: f"run_{e}"
        )
    write_bundle(tables, man, truth, d / "b", d / "truth.json")
    return d / "b", truth


def _evaluate(bundle, tmp_path, name):
    from heatinv.pipeline import evaluate

    cfg = {
        "bundle": str(bundle),
        "output": str(tmp_path / name),
        "split": {
            "train": ["run_E1", "run_E2", "run_E3", "run_E4", "run_E5"],
            "val": ["run_E6"],
            "test": ["run_E7"],
        },
        "fv": {"n_starts": 2},
        "pinn": {"enabled": False},
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return evaluate(path, verbose=False)


def test_hidden_eval_shape_and_determinism(odd_bundle, tmp_path):
    bundle, truth = odd_bundle
    r1 = _evaluate(bundle, tmp_path, "a")
    r2 = _evaluate(bundle, tmp_path, "b")
    # 1. every query predicted exactly once, by its ID, with finite values
    q = pd.read_csv(bundle / "queries.csv")
    p = pd.read_csv(tmp_path / "a" / "predictions_fv_variable.csv")
    assert sorted(p["query_id"]) == sorted(q["query_id"])
    assert np.isfinite(
        p[["u_latent", "u_lower", "u_upper", "y_pred", "y_lower", "y_upper"]].to_numpy()
    ).all()
    assert (p["y_lower"] <= p["y_pred"]).all() and (p["y_pred"] <= p["y_upper"]).all()
    # 2. machine-readable estimates with status and provenance
    est = json.loads((tmp_path / "a" / "estimates.json").read_text())
    assert est["status"] in ("ok", "weak_evidence", "failed")
    assert est["provenance"]["data_sha256"] == load_bundle(bundle).sha256
    # 3. deterministic end to end
    assert r1["estimates"]["raw_params"] == r2["estimates"]["raw_params"]
    # 4. sensible recovery on an unfamiliar layout
    for k in ("a0", "beta", "h"):
        lo, hi = r1["estimates"]["intervals"][k]
        assert lo - 2 * (hi - lo) <= truth["params"][k] <= hi + 2 * (hi - lo), k


def test_loader_fuzz(small_bundle, tmp_path):
    """Random corruptions: the loader must either load clean data or raise BundleError --
    never crash with another exception, never return NaN / out-of-domain readings."""
    src, _ = small_bundle
    rng = np.random.default_rng(5)
    ops = [
        "drop_rows",
        "dup_rows",
        "nan_cell",
        "garbage_cell",
        "shuffle",
        "neg_sd",
        "drop_col",
        "huge_t",
    ]
    for it in range(40):
        dst = tmp_path / f"f{it}"
        shutil.copytree(src, dst)
        name = rng.choice(
            [
                "observations",
                "queries",
                "initial_profiles",
                "boundary_histories",
                "sensors",
            ]
        )
        df = pd.read_csv(dst / f"{name}.csv", dtype=str)
        op = rng.choice(ops)
        i = int(rng.integers(len(df)))
        col = df.columns[int(rng.integers(len(df.columns)))]
        if op == "drop_rows":
            df = df.drop(
                index=rng.choice(len(df), max(1, len(df) // 10), replace=False)
            )
        elif op == "dup_rows":
            df = pd.concat([df, df.iloc[[i]]])
        elif op == "nan_cell":
            df.loc[i, col] = np.nan
        elif op == "garbage_cell":
            df.loc[i, col] = "x!@"
        elif op == "shuffle":
            df = df.sample(frac=1, random_state=it)
        elif op == "neg_sd" and "noise_sd" in df:
            df.loc[i, "noise_sd"] = "-1"
        elif op == "drop_col":
            df = df.drop(columns=[col])
        elif op == "huge_t" and "t" in df:
            df.loc[i, "t"] = "1e9"
        df.to_csv(dst / f"{name}.csv", index=False)
        try:
            b = load_bundle(dst)
        except BundleError:
            continue
        for e in b.experiments.values():
            o = e.obs
            assert np.isfinite(o[["t", "y", "noise_sd", "x"]].to_numpy(float)).all(), (
                it,
                op,
                name,
                col,
            )
            assert o["t"].between(0, 1).all() and (o["noise_sd"] > 0).all(), (
                it,
                op,
                name,
                col,
            )
            f = e.forcing
            assert np.isfinite(np.concatenate([f.x0, f.f0, f.tb, f.g, f.Ta])).all(), (
                it,
                op,
                name,
                col,
            )
