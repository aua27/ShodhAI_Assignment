"""End-to-end protocol: fit on train -> score on validation -> predict the unlabeled experiment.

Split discipline: only `split.train` observations ever reach a fitting routine. Validation labels
are used only for scoring after fitting. The prediction experiment has no labels in the bundle.
Ground truth (synthetic bundles only) is read ONLY by `score_against_truth`, after predictions are
written, and every such number is labelled synthetic.

Aggregation: MAE/RMSE are pooled over all scored observations of a split (each reading weight 1),
and also reported per experiment. "noisy" = against readings y (includes offset + noise);
"latent" = against the noise-free temperature u (synthetic truth only).
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Experiment, load_bundle
from .inverse import FitConfig, fit, predict
from .physics import PhysParams
from .runlog import provenance, write_json
from .solver import solve


def load_config(path) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_config_path"] = str(path)
    return cfg


def _as_query_experiment(exp: Experiment) -> Experiment:
    """Turn a labelled experiment into a query set (labels withheld) for out-of-sample scoring."""
    q = exp.obs[["obs_id", "sensor_id", "x", "t"]].rename(
        columns={"obs_id": "query_id"}
    )
    q.insert(1, "experiment_id", exp.id)
    return Experiment(exp.id, exp.forcing, exp.obs.iloc[0:0], q)


def _errors(pred: np.ndarray, target: np.ndarray) -> dict:
    d = pred - target
    return {
        "n": int(d.size),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d**2))),
    }


def _score_noisy(preds: pd.DataFrame, exps) -> dict:
    obs = pd.concat([e.obs[["obs_id", "y"]] for e in exps]).rename(
        columns={"obs_id": "query_id"}
    )
    m = preds.merge(obs, on="query_id")
    out = {"pooled": _errors(m["y_pred"].to_numpy(), m["y"].to_numpy())}
    for eid, g in m.groupby("experiment_id"):
        out[eid] = _errors(g["y_pred"].to_numpy(), g["y"].to_numpy())
    return out


def _coverage(preds: pd.DataFrame, target: np.ndarray, lo: str, hi: str) -> float:
    return float(
        np.mean((preds[lo].to_numpy() <= target) & (target <= preds[hi].to_numpy()))
    )


# ---- numerical (FV) route ----------------------------------------------------------------------


def run_fv(bundle, split, fv_cfg: dict, model="variable", fit_offsets=True):
    cfg = FitConfig(model=model, fit_offsets=fit_offsets, **fv_cfg)
    train = bundle.subset(split["train"])
    t0 = time.perf_counter()
    res = fit(train, bundle.reference_sensor, cfg)
    out = {"fit": res.summary(), "fit_runtime_s": time.perf_counter() - t0}

    train_pred = pd.concat([predict(res, _as_query_experiment(e)) for e in train])
    out["train_fit"] = _score_noisy(train_pred, train)
    t1 = time.perf_counter()
    val = bundle.subset(split["val"])
    val_pred = pd.concat([predict(res, _as_query_experiment(e)) for e in val])
    out["val"] = _score_noisy(val_pred, val)
    y_val = val_pred.merge(
        pd.concat([e.obs[["obs_id", "y"]] for e in val]).rename(
            columns={"obs_id": "query_id"}
        ),
        on="query_id",
    )["y"].to_numpy()
    out["val"]["obs_interval_coverage"] = _coverage(
        val_pred, y_val, "y_lower", "y_upper"
    )
    test_pred = pd.concat([predict(res, e) for e in bundle.subset(split["test"])])
    out["predict_runtime_s"] = time.perf_counter() - t1
    out["energy_balance_max_defect"] = max(
        float(
            np.max(
                np.abs(
                    solve(
                        res.phys, e.forcing, cfg.nx, cfg.nt
                    ).energy_balance_residual()[1:]
                )
            )
        )
        for e in train
    )
    return res, out, val_pred, test_pred


# ---- PINN route --------------------------------------------------------------------------------


def run_pinn(
    bundle, split, pinn_cfg: dict, seeds, ckpt_root: Path | None = None, verbose=False
):
    import torch

    from .pinn import PINNConfig, Trainer, adapt, energy_balance
    from .runlog import Checkpointer

    train = bundle.subset(split["train"])
    runs, val_preds, test_preds = [], [], []
    for seed in seeds:
        cfg = PINNConfig(**{**pinn_cfg, "seed": seed})
        ck = None
        if ckpt_root is not None:
            fp = {
                "config": json.dumps(asdict(cfg), sort_keys=True),
                "data": bundle.sha256,
                "split": json.dumps(split),
            }
            ck = Checkpointer(Path(ckpt_root) / f"seed{seed}", fp)
        tr = Trainer(train, bundle.reference_sensor, cfg, ck)
        r = tr.train(verbose=verbose)
        m = tr.model
        a0, beta, h = r["params"]["a0"], r["params"]["beta"], r["params"]["h"]
        offs = r["offsets"]
        dtype = getattr(torch, cfg.dtype)

        def u_net(net, x, t):
            with torch.no_grad():
                return (
                    net(
                        torch.tensor(x, dtype=dtype).reshape(-1, 1),
                        torch.tensor(t, dtype=dtype).reshape(-1, 1),
                    )
                    .numpy()
                    .ravel()
                )

        def as_pred(net, e, q):
            u = u_net(net, q["x"].to_numpy(float), q["t"].to_numpy(float))
            b = q["sensor_id"].map(offs).fillna(0.0).to_numpy()
            return pd.DataFrame(
                {
                    "query_id": q["query_id"],
                    "experiment_id": e.id,
                    "sensor_id": q["sensor_id"],
                    "x": q["x"],
                    "t": q["t"],
                    "u_latent": u,
                    "y_pred": u + b,
                    "seed": seed,
                }
            )

        train_pred = pd.concat(
            [as_pred(m.nets[e.id], e, _as_query_experiment(e).queries) for e in train]
        )
        r["train_fit"] = _score_noisy(train_pred, train)
        r["energy_balance_train"] = {
            e.id: energy_balance(m.nets[e.id], e.forcing, a0, beta, h, dtype)
            for e in train
        }
        # Consistency (NOT accuracy): PINN field vs FV field computed with the PINN's own parameters.
        cons = []
        for e in train:
            sol = solve(PhysParams(a0, beta, h), e.forcing)
            X, T = np.meshgrid(sol.x, sol.t)
            cons.append(
                np.sqrt(
                    np.mean(
                        (u_net(m.nets[e.id], X.ravel(), T.ravel()) - sol.U.ravel()) ** 2
                    )
                )
            )
        r["consistency_rmse_vs_fv_same_params"] = float(np.mean(cons))

        adapt_cost = 0.0
        for e in bundle.subset(split["val"]):
            net, info = adapt(m, e, cfg, tr.sigma_ref)
            adapt_cost += info["adapt_runtime_s"]
            vp = as_pred(net, e, _as_query_experiment(e).queries)
            val_preds.append(vp)
            r.setdefault("val", {})[e.id] = _errors(
                vp.merge(
                    e.obs[["obs_id", "y"]].rename(columns={"obs_id": "query_id"}),
                    on="query_id",
                )
                .eval("y_pred - y")
                .to_numpy(),
                0.0,
            )
        for e in bundle.subset(split["test"]):
            net, info = adapt(m, e, cfg, tr.sigma_ref)
            adapt_cost += info["adapt_runtime_s"]
            test_preds.append(as_pred(net, e, e.queries))
        r["physics_only_adaptation_runtime_s"] = adapt_cost
        if verbose:
            print(
                f"seed {seed}: {r['status']} params={r['params']} train={r['runtime_s']:.0f}s adapt={adapt_cost:.0f}s"
            )
        runs.append(r)

    ok = [r for r in runs if r["status"] == "ok"]
    spread = {}
    for k in ("a0", "beta", "h"):
        vals = np.array([r["params"][k] for r in ok]) if ok else np.array([np.nan])
        spread[k] = {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    summary = {
        "n_seeds": len(runs),
        "n_failed": len(runs) - len(ok),
        "failed_seeds": [
            {"seed": r["seed"], "reasons": r["fail_reasons"]}
            for r in runs
            if r["status"] != "ok"
        ],
        "seed_spread_ok_runs": spread,
        "runs": runs,
    }
    val_pred = pd.concat(val_preds) if val_preds else pd.DataFrame()
    test_pred = pd.concat(test_preds) if test_preds else pd.DataFrame()
    ok_seeds = {r["seed"] for r in ok}
    if not val_pred.empty:
        summary["val_ensemble"] = _score_noisy(
            _ens(val_pred, ok_seeds), bundle.subset(split["val"])
        )
    return summary, _ens(val_pred, ok_seeds), _ens(test_pred, ok_seeds)


def _ens(pred: pd.DataFrame, ok_seeds) -> pd.DataFrame:
    """Mean over successful seeds; lower/upper = min/max across seeds (spread, NOT a calibrated interval)."""
    if pred.empty:
        return pred
    p = pred[pred["seed"].isin(ok_seeds)] if ok_seeds else pred
    g = p.groupby(["query_id", "experiment_id", "sensor_id", "x", "t"], as_index=False)
    return g.agg(
        u_latent=("u_latent", "mean"),
        y_pred=("y_pred", "mean"),
        y_lower=("y_pred", "min"),
        y_upper=("y_pred", "max"),
        n_seeds=("seed", "nunique"),
    )


# ---- truth scoring (synthetic only) -----------------------------------------------------------


def score_against_truth(truth_path, est: dict, preds: dict) -> dict:
    """Parameter recovery and latent-field errors. Uses withheld synthetic truth; never feeds back."""
    truth = json.loads(Path(truth_path).read_text())
    tp = truth["params"]
    out = {
        "synthetic": True,
        "truth_params": tp,
        "parameter_recovery": {},
        "test_predictions": {},
    }
    for name, e in est.items():
        rec = {}
        for k in ("a0", "beta", "h"):
            v = e["params"].get(k)
            rec[k] = {
                "estimate": v,
                "true": tp[k],
                "abs_err": None if v is None else abs(v - tp[k]),
            }
            iv = e.get("intervals", {}).get(k)
            if iv is not None:
                rec[k]["interval_covers_truth"] = bool(iv[0] <= tp[k] <= iv[1])
        offs = e.get("offsets", {})
        if offs:
            errs = [
                abs(offs[s] - truth["offsets"][s])
                for s in offs
                if s in truth["offsets"]
            ]
            rec["offsets_mae"] = float(np.mean(errs))
        out["parameter_recovery"][name] = rec
    labels = pd.DataFrame(truth["labels"]).rename(
        columns={"y_noisy": "y_true_noisy", "u_latent": "u_true"}
    )
    for name, p in preds.items():
        if p is None or p.empty:
            continue
        m = p.merge(labels, on="query_id")
        y_true, u_true = m["y_true_noisy"].to_numpy(), m["u_true"].to_numpy()
        r = {
            "vs_noisy_readings": _errors(m["y_pred"].to_numpy(), y_true),
            "vs_latent_truth": _errors(m["u_latent"].to_numpy(), u_true),
        }
        if "y_lower" in m:
            r["obs_interval_coverage"] = _coverage(m, y_true, "y_lower", "y_upper")
        if "u_lower" in m:
            r["latent_interval_coverage"] = _coverage(m, u_true, "u_lower", "u_upper")
        out["test_predictions"][name] = r
    return out


# ---- the one public command -------------------------------------------------------------------


def evaluate(config_path, out_dir=None, with_pinn=True, verbose=True) -> dict:
    t_start = time.perf_counter()
    cfg = load_config(config_path)
    out = Path(out_dir or cfg.get("output", "outputs/public"))
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(cfg["bundle"], keep_suspect=cfg.get("keep_suspect", True))
    split = cfg["split"]
    bundle.cleaning_log.to_csv(out / "cleaning_log.csv", index=False)

    results = {
        "bundle": {
            "path": str(bundle.path),
            "sha256": bundle.sha256,
            "synthetic": bundle.manifest.get("synthetic"),
            "cleaning": bundle.cleaning_summary(),
            "reference_sensor": bundle.reference_sensor,
        },
        "split": split,
    }
    est, preds = {}, {}
    for name, model in (
        ("fv_variable", "variable"),
        ("fv_constant_baseline", "constant"),
    ):
        res, o, vp, tp = run_fv(bundle, split, cfg.get("fv", {}), model=model)
        results[name] = o
        est[name] = o["fit"]
        preds[name] = tp
        tp.to_csv(out / f"predictions_{name}.csv", index=False)
        if verbose:
            print(
                f"{name}: status={o['fit']['status']} params={ {k: round(v, 4) for k, v in o['fit']['params'].items()} } "
                f"val_rmse={o['val']['pooled']['rmse']:.4f}"
            )

    if with_pinn and cfg.get("pinn", {}).get("enabled", True):
        pc = {
            k: v
            for k, v in cfg.get("pinn", {}).items()
            if k not in ("enabled", "seeds")
        }
        summ, vp, tp = run_pinn(
            bundle,
            split,
            pc,
            cfg["pinn"].get("seeds", [0, 1, 2]),
            ckpt_root=out / "checkpoints",
            verbose=verbose,
        )
        results["pinn"] = summ
        sp = summ["seed_spread_ok_runs"]
        est["pinn_seed_mean"] = {
            "params": {k: sp[k]["mean"] for k in sp},
            "intervals": {k: (sp[k]["min"], sp[k]["max"]) for k in sp},
        }
        preds["pinn_seed_mean"] = tp
        if not tp.empty:
            tp.to_csv(out / "predictions_pinn.csv", index=False)

    # Primary estimate = variable-diffusivity FV fit (method frozen before evaluation; see README).
    primary = results["fv_variable"]["fit"]
    results["estimates"] = {
        "method": "fv_variable (weighted least squares, Laplace intervals)",
        "params": {
            k: (None if k in primary["withheld"] else v)
            for k, v in primary["params"].items()
        },
        "raw_params": primary["params"],
        "intervals": primary["intervals"],
        "interval_type": primary["interval_type"],
        "withheld": primary["withheld"],
        "offsets": primary["offsets"],
        "status": primary["status"],
    }
    statuses = [results["fv_variable"]["fit"]["status"]]
    if "pinn" in results and results["pinn"]["n_failed"] == results["pinn"]["n_seeds"]:
        statuses.append("failed")
    results["status"] = (
        "failed"
        if "failed" in statuses[:1]
        else ("weak_evidence" if "weak_evidence" in statuses else "ok")
    )

    if cfg.get("truth"):  # synthetic only -- scored after everything above is fixed
        results["synthetic_truth_scoring"] = score_against_truth(
            cfg["truth"], est, preds
        )
    results["provenance"] = provenance(
        "evaluate",
        {k: v for k, v in cfg.items() if not k.startswith("_")},
        bundle.sha256,
        seed=cfg.get("fv", {}).get("seed"),
        t_start=t_start,
    )
    write_json(out / "results.json", results)
    write_json(
        out / "estimates.json",
        {
            "estimates": results["estimates"],
            "status": results["status"],
            "provenance": results["provenance"],
        },
    )
    return results
