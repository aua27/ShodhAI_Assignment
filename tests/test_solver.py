"""Solver correctness: independent of the inverse code."""

import numpy as np
import pytest

from heatinv.physics import PhysParams
from heatinv.verify import energy_balance_max, refinement_study, steady_state_error

PARAMS = [
    PhysParams(0.07, 0.5, 0.2),
    PhysParams(0.03, -0.7, 0.5),
    PhysParams(0.12, 0.7, 0.03),
]


@pytest.mark.parametrize("p", PARAMS)
def test_mms_second_order(p):
    r = refinement_study(p, levels=(25, 50, 100))
    assert min(r["observed_order"]) > 1.9
    assert r["rmse"][-1] < 1e-4


@pytest.mark.parametrize("p", PARAMS)
def test_steady_state_closed_form(p):
    assert steady_state_error(p) < 1e-5


def test_energy_conservation_to_roundoff():
    assert energy_balance_max(PARAMS[0]) < 1e-12


def test_constant_diffusivity_limit_matches_beta_zero_integral():
    from heatinv.physics import inverse_diffusivity_integral

    x = np.linspace(0, 1, 11)
    assert np.allclose(inverse_diffusivity_integral(x, 0.05, 1e-14), x / 0.05)
    assert np.allclose(inverse_diffusivity_integral(x, 0.05, 1e-6), x / 0.05, rtol=1e-5)
