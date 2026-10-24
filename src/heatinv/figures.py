"""Report figures from saved outputs:  python -m heatinv.figures"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("report/figures")


def refinement(path="outputs/verify/verify.json"):
    r = json.loads(Path(path).read_text())["mms_refinement"]
    dx = 1 / np.array(r["nx"])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.loglog(dx, r["rmse"], "o-", label="FV + CN, nt = 4 nx")
    ax.loglog(dx, r["rmse"][0] * (dx / dx[0]) ** 2, "k--", lw=1, label="slope 2")
    ax.set_xlabel("dx")
    ax.set_ylabel("RMSE vs manufactured u*")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "refinement.png", dpi=200)
    plt.close(fig)


def val_predictions(out="outputs/public", bundle="data/public_v1"):
    """Held-out experiment E4: readings vs the predictions `evaluate` wrote (no refitting here)."""
    from .data import load_bundle

    e4 = load_bundle(bundle).experiments["E4"]
    preds = {m: pd.read_csv(Path(out) / f"val_predictions_fv_{m}.csv") for m in ("variable", "constant_baseline")}
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
    for ax, s in zip(axes, ["S0", "S3", "S5"]):
        o = e4.obs[e4.obs["sensor_id"] == s]
        ax.errorbar(o["t"], o["y"], yerr=o["noise_sd"], fmt=".", ms=3, color="0.5", label="readings (held out)")
        for (m, p), c in zip(preds.items(), ("C0", "C3")):
            p = p[p["sensor_id"] == s].sort_values("t")
            ax.plot(p["t"], p["y_pred"], color=c, label=f"FV {m}")
            ax.fill_between(p["t"], p["y_lower"], p["y_upper"], color=c, alpha=0.15)
        ax.set_title(f"{s} (x={o['x'].iloc[0]:.2f})", fontsize=9)
        ax.set_xlabel("t")
    axes[0].set_ylabel("reading y")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "val_predictions.png", dpi=200)
    plt.close(fig)


def pinn_traces(
    path="outputs/public/results.json", truth="data/truth/public_v1_truth.json"
):
    r = json.loads(Path(path).read_text())
    if "pinn" not in r:
        return
    tp = json.loads(Path(truth).read_text())["params"]
    fv = r["fv_variable"]["fit"]["params"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.4))
    for run in r["pinn"]["runs"]:
        h = pd.DataFrame(run.get("history", []))
        if h.empty:
            continue
        for ax, k in zip(axes, ("a0", "beta", "h")):
            ax.plot(
                h["step"], h[k], lw=1, label=f"seed {run['seed']} ({run['status']})"
            )
    for ax, k in zip(axes, ("a0", "beta", "h")):
        ax.axhline(tp[k], color="k", ls="--", lw=1, label="truth (synthetic)")
        ax.axhline(fv[k], color="C3", ls=":", lw=1, label="FV estimate")
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("step")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "pinn_traces.png", dpi=200)
    plt.close(fig)


def diagnosis(path="outputs/diagnosis/diagnosis.json"):
    if not Path(path).exists():
        return
    r = json.loads(Path(path).read_text())
    t = pd.DataFrame(r["hypothesis_tests"])
    t["label"] = t["kind"].str.replace("_", " ") + "\n" + t["where"].astype(str)
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.bar(range(len(t)), t["delta_chi2"].clip(lower=0.1))
    ax.axhline(25, color="k", ls="--", lw=1, label="frozen LRT threshold")
    ax.set_yscale("log")
    ax.set_xticks(range(len(t)), t["label"], fontsize=6, rotation=60)
    ax.set_ylabel("chi^2 drop")
    ax.set_title(f"verdict: {r['verdict']}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "diagnosis.png", dpi=200)
    plt.close(fig)


def study_plots():
    """The Stage-3 studies draw their own plots; copy them next to the others for the report."""
    import shutil

    for src in ("outputs/studies/ablation/ablation.png", "outputs/studies/coverage/coverage.png"):
        shutil.copy(src, OUT / Path(src).name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (refinement, val_predictions, pinn_traces, diagnosis, study_plots):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # a missing output should not block the other figures
            print("skip", f.__name__, repr(e))


if __name__ == "__main__":
    main()
