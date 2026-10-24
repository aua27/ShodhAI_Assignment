"""Input validation, record cleaning, provenance, row-order and ID/layout invariance."""

import json
import shutil

import pandas as pd
import pytest

from heatinv.data import BundleError, load_bundle
from heatinv.inverse import FitConfig, fit


def test_defects_cleaned_and_logged(defect_bundle):
    b = load_bundle(defect_bundle)
    s = b.cleaning_summary()
    for reason in (
        "flag_bad",
        "conflict",
        "exact_duplicate",
        "missing_value",
        "invalid_time",
    ):
        assert s.get(reason, 0) > 0, reason
    obs = pd.concat([e.obs for e in b.experiments.values()])
    assert not obs["obs_id"].duplicated().any()
    assert obs["y"].notna().all() and obs["t"].between(0, 1).all()
    assert (
        b.cleaning_log["source_row"] >= 2
    ).all()  # provenance: points at the raw CSV line


def test_row_order_invariance(small_bundle, tmp_path):
    src, _ = small_bundle
    dst = tmp_path / "shuffled"
    shutil.copytree(src, dst)
    for name in ("observations", "initial_profiles", "boundary_histories", "sensors"):
        df = pd.read_csv(dst / f"{name}.csv")
        df.sample(frac=1, random_state=3).to_csv(dst / f"{name}.csv", index=False)
    a, b = load_bundle(src), load_bundle(dst)
    cfg = FitConfig(n_starts=1)
    ra = fit(a.subset(["E1", "E2", "E3"]), a.reference_sensor, cfg)
    rb = fit(b.subset(["E1", "E2", "E3"]), b.reference_sensor, cfg)
    for k in ("a0", "beta", "h"):
        assert ra.params[k] == pytest.approx(rb.params[k], rel=1e-9)


def test_unseen_ids_and_layout(small_bundle, tmp_path):
    """Renamed experiments/sensors and moved sensors must still load and fit."""
    src, _ = small_bundle
    dst = tmp_path / "renamed"
    shutil.copytree(src, dst)
    ren_e = {"E1": "run-17", "E2": "run-42", "E3": "X", "E4": "val_A", "E5": "blind"}
    ren_s = {f"S{i}": f"TC{i:02d}" for i in range(6)}
    for name in (
        "observations",
        "initial_profiles",
        "boundary_histories",
        "queries",
        "sensors",
    ):
        df = pd.read_csv(dst / f"{name}.csv")
        if "experiment_id" in df:
            df["experiment_id"] = df["experiment_id"].map(ren_e)
        if "sensor_id" in df:
            df["sensor_id"] = df["sensor_id"].map(ren_s)
        df.to_csv(dst / f"{name}.csv", index=False)
    man = json.loads((dst / "manifest.json").read_text())
    man["reference_sensor"] = ren_s[man["reference_sensor"]]
    (dst / "manifest.json").write_text(json.dumps(man))
    b = load_bundle(dst)
    assert set(b.experiments) == set(ren_e.values())
    assert len(b.experiments["blind"].queries) > 0
    r = fit(
        b.subset(["run-17", "run-42", "X"]), b.reference_sensor, FitConfig(n_starts=1)
    )
    assert r.status != "failed"


def _break(src, tmp_path, fn):
    dst = tmp_path / "broken"
    shutil.copytree(src, dst)
    fn(dst)
    return dst


def test_missing_column_rejected(small_bundle, tmp_path):
    def f(d):
        pd.read_csv(d / "observations.csv").drop(columns=["noise_sd"]).to_csv(
            d / "observations.csv", index=False
        )

    with pytest.raises(BundleError, match="noise_sd"):
        load_bundle(_break(small_bundle[0], tmp_path, f))


def test_inputs_must_cover_domain(small_bundle, tmp_path):
    def f(d):
        bh = pd.read_csv(d / "boundary_histories.csv")
        bh[bh["t"] <= 0.5].to_csv(d / "boundary_histories.csv", index=False)

    with pytest.raises(BundleError, match="cover"):
        load_bundle(_break(small_bundle[0], tmp_path, f))


def test_query_with_unknown_sensor_rejected(small_bundle, tmp_path):
    def f(d):
        q = pd.read_csv(d / "queries.csv")
        q.loc[0, "sensor_id"] = "S99"
        q.to_csv(d / "queries.csv", index=False)

    with pytest.raises(BundleError, match="unknown"):
        load_bundle(_break(small_bundle[0], tmp_path, f))


def test_reference_sensor_required(small_bundle, tmp_path):
    def f(d):
        m = json.loads((d / "manifest.json").read_text())
        m.pop("reference_sensor")
        (d / "manifest.json").write_text(json.dumps(m))
        s = pd.read_csv(d / "sensors.csv")
        s["is_reference"] = False
        s.to_csv(d / "sensors.csv", index=False)

    with pytest.raises(BundleError, match="reference"):
        load_bundle(_break(small_bundle[0], tmp_path, f))
