"""Known-truth recovery, held-out prediction, model mismatch, split discipline."""

import numpy as np

from heatinv.data import load_bundle
from heatinv.inverse import FitConfig, fit, predict
from heatinv.physics import PhysParams
from heatinv.pipeline import _as_query_experiment
from heatinv.solver import solve


def test_known_truth_recovery(small_bundle):
    path, truth = small_bundle
    b = load_bundle(path)
    r = fit(b.subset(["E1", "E2", "E3"]), b.reference_sensor, FitConfig(n_starts=2))
    assert r.status == "ok"
    assert 0.7 < r.chi2_red < 1.3
    for k in ("a0", "beta", "h"):
        assert abs(r.params[k] - truth["params"][k]) < 4 * r.stderr[k], k
    errs = [abs(r.offsets[s] - truth["offsets"][s]) for s in r.offsets]
    assert max(errs) < 0.01


def test_held_out_prediction(small_bundle):
    path, truth = small_bundle
    b = load_bundle(path)
    r = fit(b.subset(["E1", "E2", "E3"]), b.reference_sensor, FitConfig(n_starts=1))
    e4 = b.experiments["E4"]
    pred = predict(
        r, _as_query_experiment(e4)
    )  # labels withheld from the prediction call
    u_true = solve(PhysParams(**truth["params"]), e4.forcing, nx=400, nt=1600).sample(
        pred["x"].to_numpy(), pred["t"].to_numpy()
    )
    assert np.sqrt(np.mean((pred["u_latent"] - u_true) ** 2)) < 0.004
    cov = np.mean((pred["y_lower"] <= e4.obs["y"]) & (e4.obs["y"] <= pred["y_upper"]))
    assert cov > 0.85


def test_prediction_needs_no_labels(small_bundle):
    b = load_bundle(small_bundle[0])
    e5 = b.experiments["E5"]
    assert len(e5.obs) == 0 and len(e5.queries) > 0
    r = fit(b.subset(["E1", "E2", "E3"]), b.reference_sensor, FitConfig(n_starts=1))
    p = predict(r, e5)
    assert len(p) == len(e5.queries) and p["y_pred"].notna().all()


def test_model_mismatch_flagged():
    """Wrong physics (constant a) on strongly graded truth must raise chi^2 / weak status."""
    from heatinv.synth import SynthSpec, generate, write_bundle
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as d:
        tables, man, truth = generate(
            SynthSpec(
                seed=21,
                truth={"a0": 0.06, "beta": 0.65, "h": 0.2, "offsets": [0.0] * 6},
            )
        )
        write_bundle(tables, man, truth, pathlib.Path(d) / "b")
        b = load_bundle(pathlib.Path(d) / "b")
        r = fit(
            b.subset(["E1", "E2", "E3"]),
            b.reference_sensor,
            FitConfig(n_starts=1, model="constant"),
        )
        assert r.chi2_red > 1.5 and r.status == "weak_evidence"


def test_withholding_when_unidentifiable(small_bundle):
    """With only the reference sensor and early times, h must not be reported as supported."""
    from heatinv.data import Experiment

    b = load_bundle(small_bundle[0])
    exps = [
        Experiment(
            e.id,
            e.forcing,
            e.obs[
                (e.obs["sensor_id"] == b.reference_sensor) & (e.obs["t"] < 0.3)
            ].reset_index(drop=True),
            e.queries,
        )
        for e in b.subset(["E1", "E2", "E3"])
    ]
    r = fit(exps, b.reference_sensor, FitConfig(n_starts=1))
    assert "h" in r.withheld and r.status == "weak_evidence"
