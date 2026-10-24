import json

import pytest

from heatinv.synth import SynthSpec, generate, write_bundle


@pytest.fixture(scope="session")
def small_bundle(tmp_path_factory):
    """Clean synthetic bundle with known truth (separate from the public bundle)."""
    d = tmp_path_factory.mktemp("bundle")
    tables, man, truth = generate(
        SynthSpec(
            seed=11,
            truth={
                "a0": 0.06,
                "beta": 0.3,
                "h": 0.25,
                "offsets": [0.01, 0, -0.02, 0.015, -0.01, 0.02],
            },
        )
    )
    write_bundle(tables, man, truth, d / "b", d / "truth.json")
    return d / "b", json.loads((d / "truth.json").read_text())


@pytest.fixture(scope="session")
def defect_bundle(tmp_path_factory):
    d = tmp_path_factory.mktemp("defects")
    tables, man, truth = generate(SynthSpec(seed=12, defects=True))
    write_bundle(tables, man, truth, d / "b", d / "truth.json")
    return d / "b"
