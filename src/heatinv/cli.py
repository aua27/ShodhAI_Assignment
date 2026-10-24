"""Command-line interface:  python -m heatinv <command> [options]

generate   build a synthetic bundle (+ separate truth file) from a spec YAML
verify     forward-solver verification (MMS refinement, steady state, energy balance)
fit        fit parameters on the training split  (--method fv | constant | pinn)
predict    fit, then predict every query of the prediction experiment(s)
evaluate   THE public command: fit -> validate -> predict -> (synthetic) score, all methods
study      Stage 3 studies: coverage | ablation | design | bias
diagnose   Stage 4 model diagnosis with frozen rules
restart    Stage 4 checkpoint / deterministic-restart check
smoke      reduced end-to-end CPU run (target < 10 min)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def _cmd_generate(a):
    import yaml

    from .synth import SynthSpec, generate, write_bundle

    spec = yaml.safe_load(Path(a.spec).read_text())
    out, truth = spec.pop("out"), spec.pop("truth_out")
    tables, manifest, tr = generate(SynthSpec(**spec))
    write_bundle(tables, manifest, tr, Path(out), Path(truth))
    print(f"wrote bundle {out} (truth -> {truth})")


def _cmd_verify(a):
    from .runlog import provenance, write_json
    from .verify import run_all

    t0 = time.perf_counter()
    r = run_all()
    r["provenance"] = provenance("verify", {}, None, t_start=t0)
    write_json(Path(a.out) / "verify.json", r)
    ref = r["mms_refinement"]
    print(
        "MMS refinement  nx:",
        ref["nx"],
        " rmse:",
        [f"{e:.2e}" for e in ref["rmse"]],
        " order:",
        [round(o, 3) for o in ref["observed_order"]],
    )
    print(
        f"steady-state max error {r['steady_state_max_abs_error']:.2e}   "
        f"energy-balance max defect {r['energy_balance_max_abs_defect']:.2e}"
    )
    ok = min(ref["observed_order"]) > 1.8 and r["steady_state_max_abs_error"] < 1e-4
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _cmd_fit(a, predict_too=False):
    from .data import load_bundle
    from .pipeline import load_config, run_fv, run_pinn
    from .runlog import provenance, write_json

    t0 = time.perf_counter()
    cfg = load_config(a.config)
    b = load_bundle(cfg["bundle"])
    out = Path(a.out or cfg.get("output", "outputs/public"))
    if a.method in ("fv", "constant"):
        res, o, vp, tp = run_fv(
            b,
            cfg["split"],
            cfg.get("fv", {}),
            model="variable" if a.method == "fv" else "constant",
        )
        result = o
    else:
        pc = {
            k: v
            for k, v in cfg.get("pinn", {}).items()
            if k not in ("enabled", "seeds")
        }
        result, vp, tp = run_pinn(
            b,
            cfg["split"],
            pc,
            cfg["pinn"].get("seeds", [0, 1, 2]),
            out / "checkpoints",
            True,
        )
    result["provenance"] = provenance(
        f"fit:{a.method}",
        {k: v for k, v in cfg.items() if not k.startswith("_")},
        b.sha256,
        t_start=t0,
    )
    write_json(out / f"fit_{a.method}.json", result)
    if predict_too and tp is not None and not tp.empty:
        tp.to_csv(out / f"predictions_{a.method}.csv", index=False)
        print(f"predictions -> {out / f'predictions_{a.method}.csv'}")
    print(f"fit -> {out / f'fit_{a.method}.json'}")


def _cmd_evaluate(a):
    from .pipeline import evaluate

    r = evaluate(a.config, a.out, with_pinn=not a.no_pinn)
    print("overall status:", r["status"])
    print(
        "estimates:", r["estimates"]["params"], "withheld:", r["estimates"]["withheld"]
    )


def _cmd_study(a):
    from . import studies

    getattr(studies, a.name)(Path(a.out), quick=a.quick)


def _cmd_diagnose(a):
    from .diagnose import diagnose_bundle

    diagnose_bundle(a.bundle, a.rules, Path(a.out), train_ids=a.train)


def _cmd_restart(a):
    from .recovery import restart_check

    ok = restart_check(Path(a.out))
    return 0 if ok else 1


def _cmd_smoke(a):
    from .smoke import smoke

    return 0 if smoke(Path(a.out)) else 1


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="heatinv",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("generate")
    s.add_argument("--spec", required=True)
    s.set_defaults(f=_cmd_generate)
    s = sub.add_parser("verify")
    s.add_argument("--out", default="outputs/verify")
    s.set_defaults(f=_cmd_verify)
    for name, fn in (("fit", _cmd_fit), ("predict", lambda a: _cmd_fit(a, True))):
        s = sub.add_parser(name)
        s.add_argument("--config", default="configs/public.yaml")
        s.add_argument("--method", choices=["fv", "constant", "pinn"], default="fv")
        s.add_argument("--out", default=None)
        s.set_defaults(f=fn)
    s = sub.add_parser("evaluate")
    s.add_argument("--config", default="configs/public.yaml")
    s.add_argument("--out", default=None)
    s.add_argument("--no-pinn", action="store_true")
    s.set_defaults(f=_cmd_evaluate)
    s = sub.add_parser("study")
    s.add_argument("name", choices=["coverage", "ablation", "design", "bias", "diagnosis_validation", "diagnosis_sweep"])
    s.add_argument("--out", default="outputs/studies")
    s.add_argument("--quick", action="store_true")
    s.set_defaults(f=_cmd_study)
    s = sub.add_parser("diagnose")
    s.add_argument("--bundle", required=True)
    s.add_argument("--rules", default="configs/diagnosis_rules.yaml")
    s.add_argument("--train", nargs="+", default=["E1", "E2", "E3", "E4"])
    s.add_argument("--out", default="outputs/diagnosis")
    s.set_defaults(f=_cmd_diagnose)
    s = sub.add_parser("restart")
    s.add_argument("--out", default="outputs/restart")
    s.set_defaults(f=_cmd_restart)
    s = sub.add_parser("smoke")
    s.add_argument("--out", default="outputs/smoke")
    s.set_defaults(f=_cmd_smoke)

    a = p.parse_args(argv)
    pd.set_option("display.width", 160)
    rc = a.f(a)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
