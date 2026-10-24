"""White-box tests: internals checked against independent derivations."""

import numpy as np
import pytest
import torch

from heatinv.physics import PhysParams, diffusivity
from heatinv.solver import Forcing, solve
from heatinv.verify import mms_exact, mms_problem


class AnalyticNet(torch.nn.Module):
    """Stands in for a PINN field: returns the manufactured solution exactly."""

    def forward(self, x, t):
        return torch.exp(-t) * torch.cos(np.pi * x) + 0.5 * x


def test_pinn_operator_equals_hand_derived_source():
    """Autograd conservative operator incl. a'(x): u_t - d/dx(a u_x) on the manufactured u must
    equal the hand-derived MMS source (derived in verify.py independently of torch)."""
    from heatinv.pinn import pde_residual

    p = PhysParams(0.07, 0.55, 0.2)
    _, source = mms_problem(p)
    x = torch.rand(500, 1, dtype=torch.float64)
    t = torch.rand(500, 1, dtype=torch.float64)
    r = pde_residual(
        AnalyticNet(),
        x.clone(),
        t.clone(),
        torch.tensor(p.a0, dtype=torch.float64),
        torch.tensor(p.beta, dtype=torch.float64),
    )
    expected = np.array(
        [source(xi, ti) for xi, ti in zip(x.numpy().ravel(), t.numpy().ravel())]
    )
    assert np.allclose(r.detach().numpy().ravel(), expected, atol=1e-10)


def test_pinn_operator_without_aprime_would_fail():
    """Guard against a silently non-conservative operator: a*u_xx alone differs by a'(x) u_x."""
    p = PhysParams(0.07, 0.55, 0.2)
    x = np.linspace(0.05, 0.95, 50)
    t = 0.3
    a = diffusivity(x, p.a0, p.beta)
    u_x = -np.pi * np.exp(-t) * np.sin(np.pi * x) + 0.5
    assert (
        np.max(np.abs(2 * p.beta * a * u_x)) > 1e-2
    )  # the a' term is not negligible here


def test_pinn_robin_residual_zero_on_exact_solution():
    from heatinv.pinn import robin_residual

    p = PhysParams(0.07, 0.55, 0.2)
    forcing, _ = mms_problem(p)
    t = torch.linspace(0, 1, 50, dtype=torch.float64).reshape(-1, 1)
    Ta = torch.tensor(forcing.Ta_at(t.numpy().ravel()), dtype=torch.float64).reshape(
        -1, 1
    )
    r = robin_residual(
        AnalyticNet(),
        t.clone(),
        torch.tensor(p.a0, dtype=torch.float64),
        torch.tensor(p.beta, dtype=torch.float64),
        torch.tensor(p.h, dtype=torch.float64),
        Ta,
    )
    assert float(r.abs().max()) < 1e-6  # Ta itself is piecewise-linear interpolated


def test_pinn_energy_balance_zero_on_exact_solution_without_source():
    """energy_balance() on an exact source-free solution must be ~0. Use the steady closed form."""
    from heatinv.pinn import energy_balance
    from heatinv.physics import inverse_diffusivity_integral

    p = PhysParams(0.06, -0.4, 0.3)
    I1 = inverse_diffusivity_integral(1.0, p.a0, p.beta)
    q = p.h * (1.0 - 0.2) / (1 + p.h * I1)

    class Steady(torch.nn.Module):
        def forward(self, x, t):
            I = np.exp(p.beta) / p.a0 * (1 - torch.exp(-2 * p.beta * x)) / (2 * p.beta)
            return 1.0 - q * I + 0 * t

    d = energy_balance(Steady(), None, p.a0, p.beta, p.h, dtype=torch.float64)
    assert d < 1e-6


def test_maximum_principle_with_incompatible_corner():
    """No source: u must stay within [min, max] of f, g, Ta even when f(0) != g(0) (CN can
    oscillate here; Rannacher start-up must prevent it)."""
    p = PhysParams(0.12, 0.7, 0.5)
    x0 = np.linspace(0, 1, 51)
    tb = np.linspace(0, 1, 11)
    forcing = Forcing(
        x0, np.zeros_like(x0), tb, np.ones_like(tb), np.zeros_like(tb)
    )  # step at x=0
    sol = solve(p, forcing, nx=100, nt=100)
    assert sol.U.min() >= -1e-9 and sol.U.max() <= 1 + 1e-9


def test_jacobian_matches_independent_finite_difference(small_bundle):
    from heatinv.data import load_bundle
    from heatinv.inverse import FitConfig, Problem

    b = load_bundle(small_bundle[0])
    prob = Problem(b.subset(["E1"]), b.reference_sensor, FitConfig())
    theta = np.concatenate(
        [[0.06, 0.3, 0.25], np.full(len(prob.offset_sensors), 0.005)]
    )
    J = prob.jacobian(theta)
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = 1e-6 if j < 3 else 1e-4
        fd = (prob.residual(theta + e) - prob.residual(theta - e)) / (2 * e[j])
        assert np.allclose(J[:, j], fd, rtol=1e-3, atol=1e-3 * np.abs(fd).max()), j


def test_sd_from_interval_roundtrip(small_bundle):
    """Probe 2 (unit consistency): the predictive sd recovered from (upper-lower)/(2*1.96) must
    equal the sd used to build the interval -- the held-out chi^2 guard relies on it."""
    from heatinv.data import load_bundle
    from heatinv.inverse import FitConfig, fit, predict
    from heatinv.pipeline import _as_query_experiment

    b = load_bundle(small_bundle[0])
    r = fit(b.subset(["E1", "E2", "E3"]), b.reference_sensor, FitConfig(n_starts=1))
    p = predict(r, _as_query_experiment(b.experiments["E4"]))
    sd_y = (p["y_upper"] - p["y_lower"]) / (2 * 1.959964)
    assert (
        sd_y >= p["sensor_id"].map(r.noise_sd) - 1e-12
    ).all()  # >= sensor noise alone
    sd_u = (p["u_upper"] - p["u_lower"]) / (2 * 1.959964)
    assert (
        sd_u <= sd_y + 1e-12
    ).all()  # latent interval never wider than reading interval


@pytest.mark.parametrize("case", ["none", "single", "ambiguous", "misfit"])
def test_diagnosis_decision_branches(case):
    """Every verdict branch of the frozen decision rule is reachable and maps correctly."""
    from heatinv.diagnose import decide

    rules = {"lrt_threshold": 15.1, "min_margin": 10.0}
    ranked = {
        "none": [("sensor_drift", 3.0), ("heat_source", 1.0)],
        "single": [("heat_source", 40.0), ("sensor_drift", 12.0)],
        "ambiguous": [("heat_source", 40.0), ("boundary_input_error", 35.0)],
        "misfit": [("heat_source", 5.0), ("sensor_drift", 4.0)],
    }[case]
    v = decide(
        [{"kind": k, "delta_chi2": d} for k, d in ranked],
        misfit=(case == "misfit"),
        rules=rules,
    )
    assert (
        v
        == {
            "none": "no_fault_detected",
            "single": "suspected:heat_source",
            "ambiguous": "ambiguous:heat_source|boundary_input_error",
            "misfit": "unexplained_misfit",
        }[case]
    )


def test_exact_physics_recheck_accepts_optimum_rejects_public_seed2():
    """The PINN recheck must pass the FV optimum itself and reject the public-run seed-2 PINN
    parameters (h 43% low) that the v1 absolute-chi^2 rule let through by 1e-4."""
    from heatinv.data import load_bundle
    from heatinv.inverse import FitConfig, fit
    from heatinv.pipeline import exact_physics_recheck

    b = load_bundle("data/public_v1")
    train = b.subset(["E1", "E2", "E3"])
    ref = fit(train, b.reference_sensor, FitConfig(n_starts=2))
    own = exact_physics_recheck(train, b.reference_sensor, ref.params, ref.offsets, ref)
    assert not own["reject"] and abs(own["delta_chi2"]) < 1e-3
    seed2 = {"a0": 0.09576744586229324, "beta": -0.42195963859558105, "h": 0.1845558136701584}
    import json
    runs = json.loads(open("results/public_eval_v1_pinn_runs.json").read())
    offs = next(r["offsets"] for r in runs if r["seed"] == 2)
    bad = exact_physics_recheck(train, b.reference_sensor, seed2, offs, ref)
    assert bad["reject"], bad
