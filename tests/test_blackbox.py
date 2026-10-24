"""Black-box tests: behaviour specified by the brief, checked only through public entry points
(load_bundle / fit / predict / evaluate / CLI). Includes metamorphic relations: transformations of
the input whose effect on the output is known from the physics without knowing the answer."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from heatinv.data import BundleError, load_bundle
from heatinv.inverse import FitConfig, fit, predict

CFG = FitConfig(n_starts=1)
TRAIN = ["E1", "E2", "E3"]


def _copy(src, tmp_path, name="b"):
    dst = tmp_path / name
    shutil.copytree(src, dst)
    return dst


def _edit(path, name, fn):
    df = pd.read_csv(path / f"{name}.csv", dtype={"obs_id": str, "query_id": str})
    df = fn(df)
    df.to_csv(path / f"{name}.csv", index=False)


def _fit(path):
    b = load_bundle(path)
    return fit(b.subset(TRAIN), b.reference_sensor, CFG)


# ---- metamorphic relations -----------------------------------------------------------------


def test_shift_invariance(small_bundle, tmp_path):
    """PDE and Robin BC involve only differences/derivatives of u: adding c to every temperature
    (f, g, Ta, y) must leave a0, beta, h and offsets unchanged."""
    c = 3.7
    dst = _copy(small_bundle[0], tmp_path)
    _edit(dst, "observations", lambda d: d.assign(y=d["y"] + c))
    _edit(dst, "initial_profiles", lambda d: d.assign(u0=d["u0"] + c))
    _edit(dst, "boundary_histories", lambda d: d.assign(g=d["g"] + c, Ta=d["Ta"] + c))
    a, b = _fit(small_bundle[0]), _fit(dst)
    for k in ("a0", "beta", "h"):
        assert b.params[k] == pytest.approx(a.params[k], rel=1e-4, abs=1e-6)
    for s in a.offsets:
        assert b.offsets[s] == pytest.approx(a.offsets[s], abs=1e-5)


def test_scale_invariance(small_bundle, tmp_path):
    """The model is linear in u: scaling all temperatures AND noise sd by k leaves the physics and
    the whitened residuals unchanged (offsets scale by k; with k<1 they stay inside the bound)."""
    k = 0.5
    dst = _copy(small_bundle[0], tmp_path)
    _edit(
        dst,
        "observations",
        lambda d: d.assign(y=d["y"] * k, noise_sd=d["noise_sd"] * k),
    )
    _edit(dst, "initial_profiles", lambda d: d.assign(u0=d["u0"] * k))
    _edit(dst, "boundary_histories", lambda d: d.assign(g=d["g"] * k, Ta=d["Ta"] * k))
    a, b = _fit(small_bundle[0]), _fit(dst)
    for p in ("a0", "beta", "h"):
        assert b.params[p] == pytest.approx(a.params[p], rel=1e-3)
    assert b.chi2_red == pytest.approx(a.chi2_red, rel=1e-3)


def test_duplicated_readings_give_no_false_precision(small_bundle, tmp_path):
    """Every reading supplied twice under new IDs (identical noise = NO new information): the
    estimate must not move, the dependence must be detected, and the reported sd must not shrink
    by the sqrt(2) that naive independence would claim. (First version of this test asserted the
    sqrt(2) shrink -- that premise was wrong; the autocorrelation guard caught it.)"""
    dst = _copy(small_bundle[0], tmp_path)
    _edit(
        dst,
        "observations",
        lambda d: pd.concat([d, d.assign(obs_id=d["obs_id"] + "_dup")]),
    )
    a, b = _fit(small_bundle[0]), _fit(dst)
    assert any("serially correlated" in n for n in b.notes)
    for p in ("a0", "beta", "h"):
        assert b.params[p] == pytest.approx(a.params[p], rel=1e-3)
        assert b.stderr[p] >= a.stderr[p] / 1.1  # no spurious gain in precision


# ---- split discipline / no private-label access ---------------------------------------------


def test_validation_labels_never_reach_fitting(small_bundle, tmp_path):
    """Corrupting every validation reading must not change the training estimate."""
    dst = _copy(small_bundle[0], tmp_path)
    _edit(
        dst,
        "observations",
        lambda d: d.assign(
            y=np.where(d["experiment_id"] == "E4", d["y"] + 10.0, d["y"])
        ),
    )
    a, b = _fit(small_bundle[0]), _fit(dst)
    for p in ("a0", "beta", "h"):
        assert b.params[p] == a.params[p]


def test_truth_file_never_read_by_fit_or_predict(small_bundle, tmp_path, monkeypatch):
    """Fitting and predicting with the truth file removed must work identically."""
    path, _ = small_bundle
    b = load_bundle(path)
    real_open = open

    def guarded(f, *a, **k):
        assert "truth" not in str(f), f"truth accessed: {f}"
        return real_open(f, *a, **k)

    monkeypatch.setattr("builtins.open", guarded)
    r = fit(b.subset(TRAIN), b.reference_sensor, CFG)
    p = predict(r, b.experiments["E5"])
    assert len(p) == len(b.experiments["E5"].queries)


# ---- input-space partitions the hidden evaluation may use ---------------------------------


def test_obs_ids_restarting_per_experiment(small_bundle, tmp_path):
    """Many real loggers number readings per experiment (1, 2, 3 ... in each run). Such IDs are
    unique only within an experiment and must not be treated as repeated/conflicting IDs."""
    dst = _copy(small_bundle[0], tmp_path)
    _edit(
        dst,
        "observations",
        lambda d: d.assign(obs_id=d.groupby("experiment_id").cumcount().astype(str)),
    )
    b = load_bundle(dst)
    n_raw = len(pd.read_csv(dst / "observations.csv"))
    assert sum(len(e.obs) for e in b.experiments.values()) == n_raw
    assert b.cleaning_summary().get("conflict", 0) == 0


def test_sensor_layout_differs_between_experiments(small_bundle, tmp_path):
    """Some sensors absent in some experiments; uneven reading counts per sensor."""
    dst = _copy(small_bundle[0], tmp_path)
    rng = np.random.default_rng(0)
    _edit(
        dst,
        "observations",
        lambda d: d[
            ~((d["experiment_id"] == "E1") & d["sensor_id"].isin(["S4", "S5"]))
            & ~((d["experiment_id"] == "E2") & (d["sensor_id"] == "S0"))
            & (rng.random(len(d)) > 0.3)
        ],
    )
    r = _fit(dst)
    assert r.status in ("ok", "weak_evidence")
    assert set(r.offsets) == {f"S{i}" for i in range(6)}


def test_prediction_for_sensor_unseen_in_training(small_bundle, tmp_path):
    """A query on a sensor with no training data: its offset is unknown -> wider interval, not a crash."""
    dst = _copy(small_bundle[0], tmp_path)
    _edit(
        dst,
        "sensors",
        lambda d: pd.concat(
            [
                d,
                pd.DataFrame(
                    {"sensor_id": ["NEW"], "x": [0.95], "is_reference": [False]}
                ),
            ]
        ),
    )
    _edit(
        dst,
        "queries",
        lambda d: pd.concat(
            [
                d,
                pd.DataFrame(
                    {
                        "query_id": ["QN1"],
                        "experiment_id": ["E5"],
                        "sensor_id": ["NEW"],
                        "t": [0.5],
                    }
                ),
            ]
        ),
    )
    b = load_bundle(dst)
    r = fit(b.subset(TRAIN), b.reference_sensor, CFG)
    p = predict(r, b.experiments["E5"]).set_index("query_id")
    known = p.drop(index="QN1")
    w_new = p.loc["QN1", "y_upper"] - p.loc["QN1", "y_lower"]
    assert np.isfinite(w_new) and w_new > (known["y_upper"] - known["y_lower"]).median()


@pytest.mark.parametrize("flag", ["OK", " ok ", "Bad", "SUSPECT"])
def test_quality_flag_case_and_whitespace(small_bundle, tmp_path, flag):
    dst = _copy(small_bundle[0], tmp_path)
    _edit(
        dst,
        "observations",
        lambda d: d.assign(quality_flag=np.where(d.index < 5, flag, d["quality_flag"])),
    )
    b = load_bundle(dst)
    dropped = set(
        b.cleaning_log.loc[b.cleaning_log["action"] == "dropped", "source_row"]
    )
    first5_dropped = {r for r in dropped if r < 7}
    assert (len(first5_dropped) == 5) == (flag.strip().lower() == "bad")


def test_unknown_quality_flag_dropped_and_logged(small_bundle, tmp_path):
    """An unrecognised flag is not trusted as 'ok' and does not crash the run: dropped + logged."""
    dst = _copy(small_bundle[0], tmp_path)
    _edit(
        dst,
        "observations",
        lambda d: d.assign(
            quality_flag=np.where(d.index == 0, "maybe", d["quality_flag"])
        ),
    )
    b = load_bundle(dst)
    log = b.cleaning_log
    assert (log["reason"] == "unknown_flag").sum() == 1 and log.loc[log["reason"] == "unknown_flag", "source_row"].item() == 2


def test_nan_in_boundary_history_rejected(small_bundle, tmp_path):
    dst = _copy(small_bundle[0], tmp_path)
    _edit(
        dst,
        "boundary_histories",
        lambda d: d.assign(g=np.where(d.index == 10, np.nan, d["g"])),
    )
    with pytest.raises(BundleError, match="non-numeric|finite"):
        load_bundle(dst)


def test_sensor_outside_domain_rejected(small_bundle, tmp_path):
    dst = _copy(small_bundle[0], tmp_path)
    _edit(dst, "sensors", lambda d: d.assign(x=np.where(d.index == 0, 1.2, d["x"])))
    with pytest.raises(BundleError):
        load_bundle(dst)


# ---- boundary values of the parameter box -----------------------------------------------


@pytest.mark.parametrize(
    "truth",
    [
        {"a0": 0.0305, "beta": 0.0, "h": 0.25},
        {"a0": 0.119, "beta": 0.0, "h": 0.25},
        {"a0": 0.07, "beta": 0.69, "h": 0.25},
        {"a0": 0.07, "beta": -0.69, "h": 0.25},
        {"a0": 0.07, "beta": 0.0, "h": 0.031},
        {"a0": 0.07, "beta": 0.0, "h": 0.49},
    ],
)
def test_truth_near_box_edges(tmp_path, truth):
    """Truth near an edge: either recovered within 4 sd or explicitly flagged (at_bound / withheld)."""
    from heatinv.synth import SynthSpec, generate, write_bundle

    tables, man, t = generate(
        SynthSpec(
            seed=31, truth={**truth, "offsets": [0.01, 0, -0.01, 0.005, -0.005, 0.01]}
        )
    )
    write_bundle(tables, man, t, tmp_path / "b")
    r = _fit(tmp_path / "b")
    for k, v in truth.items():
        ok = abs(r.params[k] - v) < 4 * max(r.stderr[k], 1e-9)
        assert ok or k in r.at_bound or k in r.withheld or r.status != "ok", (
            k,
            r.params[k],
            v,
        )


# ---- CLI end to end -----------------------------------------------------------------------


def test_cli_predict_on_renamed_bundle(small_bundle, tmp_path):
    import yaml

    cfg = {
        "bundle": str(small_bundle[0]),
        "split": {"train": TRAIN, "val": ["E4"], "test": ["E5"]},
        "fv": {"n_starts": 1},
        "pinn": {"enabled": False},
    }
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "heatinv",
            "predict",
            "--config",
            str(tmp_path / "c.yaml"),
            "--method",
            "fv",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    p = pd.read_csv(out / "predictions_fv.csv")
    assert {"query_id", "y_pred", "y_lower", "y_upper", "u_latent"} <= set(p.columns)
    j = json.loads((out / "fit_fv.json").read_text())
    assert j["provenance"]["data_sha256"] and j["provenance"]["config_hash"]


def test_cli_verify_exit_code():
    r = subprocess.run(
        [sys.executable, "-m", "heatinv", "verify", "--out", "outputs/verify_test"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0 and "PASS" in r.stdout
