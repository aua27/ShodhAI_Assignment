"""CPU smoke run (python -m heatinv smoke): every stage end-to-end at reduced size, timed.

Budget 10 min; ~2 min on the reference laptop (i7-1165G7) when idle, leaving headroom for slower or
busy machines. Wall time, hardware and threads are recorded in smoke.json.
Exit code 1 if any stage fails or the budget is exceeded.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from .runlog import provenance, write_json

BUDGET_S = 600


def smoke(out: Path) -> bool:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    stages = {}

    def stage(name, fn):
        t0 = time.perf_counter()
        try:
            info = fn() or {}
            stages[name] = {
                "ok": bool(info.pop("ok", True)),
                "seconds": round(time.perf_counter() - t0, 1),
                **info,
            }
        except Exception as e:
            stages[name] = {
                "ok": False,
                "seconds": round(time.perf_counter() - t0, 1),
                "error": repr(e),
                "trace": traceback.format_exc(limit=3),
            }
        s = stages[name]
        print(f"[smoke] {name:14s} {'ok ' if s['ok'] else 'FAIL'} {s['seconds']:7.1f}s")

    def verify():
        from .verify import refinement_study, steady_state_error
        from .physics import PhysParams

        p = PhysParams(0.07, 0.5, 0.2)
        r = refinement_study(p, levels=(25, 50, 100))
        return {
            "ok": min(r["observed_order"]) > 1.8 and steady_state_error(p) < 1e-4,
            "orders": r["observed_order"],
        }

    def fv():
        from .data import load_bundle
        from .pipeline import run_fv

        b = load_bundle("data/public_v1")
        split = {"train": ["E1", "E2", "E3"], "val": ["E4"], "test": ["E5"]}
        res, o, vp, tp = run_fv(b, split, {"n_starts": 2})
        tp.to_csv(out / "predictions_fv.csv", index=False)
        return {
            "ok": res.status != "failed"
            and len(tp) == len(b.experiments["E5"].queries),
            "status": res.status,
            "params": res.params,
            "val_rmse": o["val"]["pooled"]["rmse"],
        }

    def pinn():
        from .data import load_bundle
        from .pipeline import run_pinn

        b = load_bundle("data/public_v1")
        split = {"train": ["E1", "E2", "E3"], "val": ["E4"], "test": ["E5"]}
        cfg = dict(
            width=32,
            depth=3,
            n_colloc=800,
            adam_steps=800,
            freeze_phys_steps=300,
            lbfgs_iters=50,
            adapt_steps=400,
            adapt_lbfgs=30,
        )
        summ, vp, tp = run_pinn(b, split, cfg, seeds=[0])
        r = summ["runs"][0]
        # A 1-seed, short PINN is NOT expected to recover h; the smoke run checks the plumbing only.
        return {
            "ok": len(tp) > 0,
            "run_status": r["status"],
            "params": r["params"],
            "train_s": r["runtime_s"],
        }

    def restart():
        from .recovery import restart_check

        return {"ok": restart_check(out / "restart")}

    def diagnosis():
        import yaml

        from .data import load_bundle
        from .diagnose import diagnose
        from .synth import SynthSpec, generate, write_bundle

        rules = yaml.safe_load(Path("configs/diagnosis_rules.yaml").read_text())
        tables, man, truth = generate(
            SynthSpec(seed=7777, fault={"kind": "boundary_error", "bias": 0.06})
        )
        write_bundle(tables, man, truth, out / "diag_bundle", out / "diag_truth.json")
        r = diagnose(load_bundle(out / "diag_bundle"), rules, quick=True)
        return {"ok": True, "verdict": r["verdict"], "planted": "boundary_input_error"}

    def studies_quick():
        from . import studies

        studies.ablation(out / "studies", quick=True)
        cov = studies.coverage(out / "studies", quick=True)
        studies.design(out / "studies", quick=True)
        return {"coverage_rate_independent": {k: v["rate"] for k, v in cov["rho=0.0"].items()}}

    stage("verify", verify)
    stage("fv_fit_predict", fv)
    stage("pinn_1seed", pinn)
    stage("restart", restart)
    stage("diagnosis", diagnosis)
    stage("studies_quick", studies_quick)
    total = time.perf_counter() - t_all
    ok = all(s["ok"] for s in stages.values()) and total < BUDGET_S
    write_json(
        out / "smoke.json",
        {
            "ok": ok,
            "total_seconds": round(total, 1),
            "budget_seconds": BUDGET_S,
            "stages": stages,
            "provenance": provenance("smoke", {"budget": BUDGET_S}, None),
        },
    )
    print(
        f"[smoke] total {total:.0f}s / budget {BUDGET_S}s -> {'PASS' if ok else 'FAIL'}"
    )
    return ok
