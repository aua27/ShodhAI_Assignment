"""Stage 4 recovery checks (run: python -m heatinv restart).

1. deterministic restart : run A uninterrupted; run B is killed mid-training, then resumed from its
                           last checkpoint -> final parameters must match A exactly (CPU, same threads)
2. incompatible ckpt     : a checkpoint written under a different config is refused
3. corrupt ckpt          : a truncated newest file is skipped; the previous valid one is used
4. divergence            : an injected non-finite loss is recorded, rolled back, lr halved, run completes
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import json
import numpy as np

from .data import load_bundle
from .pinn import PINNConfig, Trainer
from .runlog import CheckpointMismatch, Checkpointer, provenance, write_json

SMALL = PINNConfig(
    width=24,
    depth=3,
    n_colloc=400,
    n_ic=64,
    n_bc=64,
    adam_steps=300,
    lbfgs_iters=20,
    resample_every=100,
    ckpt_every=100,
    freeze_phys_steps=100,
    threads=2,
)


def _fp(cfg, b):
    return {"config": json.dumps(asdict(cfg), sort_keys=True), "data": b.sha256}


def restart_check(out: Path, bundle="data/public_v1", cfg: PINNConfig = SMALL) -> bool:
    t0 = time.perf_counter()
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    b = load_bundle(bundle)
    train = b.subset(["E1", "E2", "E3"])
    checks = {}

    ra = Trainer(
        train, b.reference_sensor, cfg, Checkpointer(out / "A", _fp(cfg, b))
    ).train()

    ck_b = Checkpointer(out / "B", _fp(cfg, b))
    try:
        Trainer(train, b.reference_sensor, cfg, ck_b).train(interrupt_at=250)
    except KeyboardInterrupt as e:
        interrupted = str(e)
    rb = Trainer(
        train, b.reference_sensor, cfg, Checkpointer(out / "B", _fp(cfg, b))
    ).train(resume=True)
    diff = max(abs(ra["params"][k] - rb["params"][k]) for k in ra["params"])
    checks["deterministic_restart"] = {
        "pass": diff == 0.0,
        "max_param_diff": diff,
        "interrupted": interrupted,
        "resumed_events": rb["events"],
        "params_A": ra["params"],
        "params_B": rb["params"],
    }

    other = replace(cfg, lr=cfg.lr * 2)
    ck_c = Checkpointer(
        out / "B", _fp(other, b)
    )  # same files, different config fingerprint
    try:
        Trainer(train, b.reference_sensor, other, ck_c).train(resume=True)
        refused = False
    except CheckpointMismatch:
        refused = True
    checks["incompatible_checkpoint_refused"] = {
        "pass": refused,
        "rejections": ck_c.rejections,
    }

    newest = sorted((out / "B").glob("ckpt_*.pt"))[-1]
    corrupt = newest.with_name("ckpt_9999999.pt")
    corrupt.write_bytes(newest.read_bytes()[:100])  # truncated write
    ck_d = Checkpointer(out / "B", _fp(cfg, b))
    state = ck_d.load_latest()
    checks["corrupt_checkpoint_skipped"] = {
        "pass": state is not None
        and any("unreadable" in r["reason"] for r in ck_d.rejections),
        "rejections": ck_d.rejections,
    }
    corrupt.unlink()

    tr = Trainer(train, b.reference_sensor, cfg, None)
    tr.inject_nan_at = 150
    rd = tr.train()
    ev = [e["event"] for e in rd["events"]]
    checks["divergence_recorded_and_recovered"] = {
        "pass": "non_finite_loss" in ev and np.isfinite(rd["final_losses"]["total"]),
        "events": rd["events"],
        "final_status": rd["status"],
    }

    ok = all(c["pass"] for c in checks.values())
    write_json(
        out / "restart_check.json",
        {
            "all_pass": ok,
            "checks": checks,
            "provenance": provenance(
                "restart_check", asdict(cfg), b.sha256, cfg.seed, t0
            ),
        },
    )
    for k, v in checks.items():
        print(f"{k:36s} {'PASS' if v['pass'] else 'FAIL'}")
    return ok
