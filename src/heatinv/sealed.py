"""Blind Stage-4 test: build a diagnosis bundle whose fault is chosen by OS randomness and sealed.

    python -m heatinv.sealed make     # writes data/sealed_v1/ + data/sealed_v1_truth.json (do NOT open)
    python -m heatinv diagnose --bundle data/sealed_v1 --out outputs/diagnosis
    python -m heatinv.sealed reveal   # only after the diagnosis is committed

The truth file's sha256 is printed at creation and stored in the bundle manifest, so it cannot be
edited after the diagnosis without detection. The order is proven by the git history:
rules commit -> sealed-bundle commit -> diagnosis-output commit -> reveal.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from .synth import SynthSpec, generate, write_bundle

BUNDLE = Path("data/sealed_v1")
TRUTH = Path("data/sealed_v1_truth.json")


def make():
    if TRUTH.exists():
        sys.exit("sealed bundle already exists; refusing to overwrite")
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    kind = rng.choice(["none", "drift", "heat_source", "boundary_error"])
    fault = {"kind": str(kind)}
    if kind == "drift":
        fault.update(
            sensor=f"S{int(rng.integers(0, 6))}", rate=float(rng.uniform(0.02, 0.06))
        )
    elif kind == "heat_source":
        fault.update(
            Q=float(rng.uniform(0.05, 0.2)), xc=float(rng.uniform(0.3, 0.8)), w=0.1
        )
    elif kind == "boundary_error":
        fault.update(bias=float(rng.choice([-1, 1]) * rng.uniform(0.02, 0.06)))
    tables, man, truth = generate(
        SynthSpec(
            seed=int(rng.integers(1 << 30)),
            fault=fault,
            meta={"purpose": "sealed blind diagnosis (synthetic)"},
        )
    )
    truth_bytes = json.dumps(truth, indent=2).encode()
    digest = hashlib.sha256(truth_bytes).hexdigest()
    man["sealed_truth_sha256"] = digest
    man["generator"]["fault"] = "SEALED"  # the manifest must not leak the fault
    write_bundle(tables, man, truth, BUNDLE, None)
    TRUTH.write_bytes(truth_bytes)
    print(f"sealed bundle written to {BUNDLE}; truth sha256 {digest}")


def reveal():
    truth_bytes = TRUTH.read_bytes()
    man = json.loads((BUNDLE / "manifest.json").read_text())
    ok = hashlib.sha256(truth_bytes).hexdigest() == man["sealed_truth_sha256"]
    t = json.loads(truth_bytes)
    print("truth file intact:", ok)
    print("planted fault:", t["fault"])
    print("true params:", t["params"])


if __name__ == "__main__":
    {"make": make, "reveal": reveal}[sys.argv[1]]()
