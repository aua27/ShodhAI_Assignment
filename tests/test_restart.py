"""Restart behaviour: deterministic resume, refusal of incompatible / corrupt checkpoints, divergence."""

import shutil
from pathlib import Path

from heatinv.recovery import restart_check


def test_restart_checks(small_bundle, tmp_path):
    assert restart_check(tmp_path / "restart", bundle=str(small_bundle[0]))
