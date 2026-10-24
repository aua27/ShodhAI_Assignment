"""Generate report/numbers.tex from result files, so no number in the report is typed by hand.

Run from the repo root after the studies:  python report/make_numbers.py
Every macro is traceable to the JSON/CSV it was read from (see the SOURCES comment in the output).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(".")
out = []
sources = set()


def J(p):
    sources.add(p)
    return json.loads((R / p).read_text())


def C(p):
    sources.add(p)
    return pd.read_csv(R / p)


def m(name, val, fmt="{:.3g}"):
    s = val if isinstance(val, str) else fmt.format(val)
    out.append(f"\\newcommand{{\\{name}}}{{{s}}}")


# --- verification --------------------------------------------------------------------------
v = J("outputs/verify/verify.json")
o = v["mms_refinement"]["observed_order"]
m("OrdMin", min(o), "{:.3f}")
m("OrdMax", max(o), "{:.3f}")
m("MmsErrHundred", v["mms_refinement"]["rmse"][2], "{:.1e}")
m("SteadyErr", v["steady_state_max_abs_error"], "{:.1e}")
m("EnergyDefect", v["energy_balance_max_abs_defect"], "{:.0e}")

# --- public evaluation ---------------------------------------------------------------------
r = J("outputs/public/results.json")
cl = r["bundle"]["cleaning"]
m("ClnBad", cl.get("flag_bad", 0), "{:d}")
m("ClnConflict", cl.get("conflict", 0), "{:d}")
m("ClnDup", cl.get("exact_duplicate", 0), "{:d}")
m("ClnMissing", cl.get("missing_value", 0), "{:d}")
m("ClnTime", cl.get("invalid_time", 0), "{:d}")
m("ClnSuspect", cl.get("flag_suspect", 0), "{:d}")
tr = r["synthetic_truth_scoring"]["truth_params"]
for k, n in (("a0", "A"), ("beta", "B"), ("h", "H")):
    m(f"True{n}", tr[k], "{:.3f}")
for meth, tag in (("fv_variable", "Fv"), ("fv_constant_baseline", "Cst")):
    f = r[meth]["fit"]
    for k, n in (("a0", "A"), ("beta", "B"), ("h", "H")):
        m(f"{tag}{n}", f["params"][k], "{:.3f}")
        lo, hi = f["intervals"][k]
        m(f"{tag}{n}Lo", lo, "{:.3f}")
        m(f"{tag}{n}Hi", hi, "{:.3f}")
    m(f"{tag}Chi", f["chi2_red"], "{:.2f}")
    m(f"{tag}Status", f["status"].replace("_", "\\_"))
    m(f"{tag}TrainRmse", r[meth]["train_fit"]["pooled"]["rmse"], "{:.4f}")
    m(f"{tag}ValRmse", r[meth]["val"]["pooled"]["rmse"], "{:.4f}")
    m(f"{tag}ValCov", r[meth]["val"]["obs_interval_coverage"], "{:.2f}")
    m(f"{tag}ValChi", r[meth]["val"]["predictive_chi2"], "{:.2f}")
    m(f"{tag}FitSec", r[meth]["fit_runtime_s"], "{:.1f}")
    tp = r["synthetic_truth_scoring"]["test_predictions"][meth]
    m(f"{tag}TestNoisy", tp["vs_noisy_readings"]["rmse"], "{:.4f}")
    m(f"{tag}TestLatent", tp["vs_latent_truth"]["rmse"], "{:.4f}")
    m(f"{tag}TestCovY", tp["obs_interval_coverage"], "{:.2f}")
    m(f"{tag}TestCovU", tp["latent_interval_coverage"], "{:.2f}")
m("FvEnergy", r["fv_variable"]["energy_balance_max_defect"], "{:.0e}")
m("LrtBeta", r["model_comparison"]["delta_chi2"], "{:.0f}")
m("ScanVerdict", r["fault_scan"]["verdict"].replace("_", " "))
m("ScanTop", r["fault_scan"]["top_hypothesis"]["delta_chi2"], "{:.1f}")
p = r["pinn"]
m("PinnNSeeds", p["n_seeds"], "{:d}")
m("PinnNFailed", p["n_failed"], "{:d}")
m("PinnNWeak", p["n_weak_evidence"], "{:d}")
sp = p["seed_spread_ok_runs"]
for k, n in (("a0", "A"), ("beta", "B"), ("h", "H")):
    m(f"Pinn{n}", sp[k]["mean"], "{:.3f}")
    m(f"Pinn{n}Min", sp[k]["min"], "{:.3f}")
    m(f"Pinn{n}Max", sp[k]["max"], "{:.3f}")
runs = p["runs"]
m("PinnTrainSec", np.mean([x["runtime_s"] for x in runs]), "{:.0f}")
m(
    "PinnAdaptSec",
    np.mean([x["physics_only_adaptation_runtime_s"] for x in runs]),
    "{:.0f}",
)
m("PinnObsLoss", np.mean([x["final_losses"]["obs"] for x in runs]), "{:.2f}")
m(
    "PinnEB",
    np.mean([np.mean(list(x["energy_balance_train"].values())) for x in runs]),
    "{:.3f}",
)
m(
    "PinnCons",
    np.mean([x["consistency_rmse_vs_fv_same_params"] for x in runs]),
    "{:.4f}",
)
m("PinnDchiMin", min(x["fv_recheck_delta_chi2"] for x in runs), "{:.0f}")
m("PinnDchiMax", max(x["fv_recheck_delta_chi2"] for x in runs), "{:.0f}")
m("PinnDchiLim", runs[0]["fv_recheck_limit"], "{:.1f}")
m("PinnTrainRmse", np.mean([x["train_fit"]["pooled"]["rmse"] for x in runs]), "{:.4f}")
m("PinnValRmse", p["val_ensemble"]["pooled"]["rmse"], "{:.4f}")
tp = r["synthetic_truth_scoring"]["test_predictions"]["pinn_seed_mean"]
m("PinnTestNoisy", tp["vs_noisy_readings"]["rmse"], "{:.4f}")
m("PinnTestLatent", tp["vs_latent_truth"]["rmse"], "{:.4f}")
m("PinnTestCovY", tp["obs_interval_coverage"], "{:.2f}")
m("EvalWall", r["provenance"]["wall_time_s"] / 60, "{:.0f}")
m("EvalCommit", r["provenance"]["git"]["commit"])
m("EvalDirty", str(r["provenance"]["git"]["dirty"]))
v1 = J("results/public_eval_v1_results.json")
s2 = [x for x in v1["pinn"]["runs"] if x["seed"] == 2][0]
m("VOneSeedTwoChi", s2["fv_recheck_chi2_red"], "{:.4f}")
m("VOneSeedTwoLim", s2["fv_recheck_limit"], "{:.4f}")
m("VOneSeedTwoH", s2["params"]["h"], "{:.3f}")
m("VOneChiGap", s2["fv_recheck_chi2_red"] - v1["fv_variable"]["fit"]["chi2_red"], "{:.2f}")

# --- studies -------------------------------------------------------------------------------
cv = J("outputs/studies/coverage/coverage.json")["summary"]
for rho, tag in (("rho=0.0", "Ind"), ("rho=0.6", "Ar")):
    for k, n in (("a0", "A"), ("beta", "B"), ("h", "H")):
        m(f"Cov{tag}{n}", cv[rho][k]["covered"], "{:d}")
    m(f"Cov{tag}N", cv[rho]["a0"]["n"], "{:d}")
    for k, n in (("a0", "A"), ("beta", "B"), ("h", "H")):
        m(f"Cov{tag}Raw{n}", cv[rho][k]["covered_without_guard"], "{:d}")
    lo = min(cv[rho][k]["wilson95"][0] for k in ("a0", "beta", "h"))
    m(f"Cov{tag}WilsonLo", lo, "{:.2f}")
rows = C("outputs/studies/coverage/coverage_rows.csv")
m("CovIndFlagged", int((rows[rows.rho == 0.0].status != "ok").sum()), "{:d}")
m("CovArFlagged", int((rows[rows.rho == 0.6].status != "ok").sum()), "{:d}")
nr = cv["noise_redraw"]
m(
    "RedrawRatioH",
    nr["h"]["empirical_sd_over_redraws"] / nr["h"]["mean_laplace_sd"],
    "{:.2f}",
)
m(
    "RedrawRatioA",
    nr["a0"]["empirical_sd_over_redraws"] / nr["a0"]["mean_laplace_sd"],
    "{:.2f}",
)

ab = J("outputs/studies/ablation/ablation.json")
abd = {x["variant"]: x for x in ab["rows"]}
for v_, tag in (
    ("baseline", "Base"),
    ("time_early_only", "Early"),
    ("time_sparse", "Sparse"),
    ("drop_right_sensors", "NoRight"),
    ("drop_left_sensors", "NoLeft"),
    ("assume_constant_a", "ConstA"),
    ("assume_zero_offsets", "ZeroOff"),
):
    x = abd[v_]
    for k, n in (("a0", "A"), ("beta", "B"), ("h", "H")):
        m(f"Ab{tag}{n}", x[f"{k}_est"], "{:.3f}")
        m(f"Ab{tag}{n}W", x[f"{k}_ci_width"], "{:.3f}")
    m(f"Ab{tag}Lat", x["val_rmse_latent"], "{:.4f}")
    m(f"Ab{tag}Status", x["status"].replace("_", "\\_"))
for k, n in (("a0", "A"), ("beta", "B"), ("h", "H")):
    m(f"AbTrue{n}", ab["truth"][k], "{:.2f}")
m(
    "AbHWidthRatioRight",
    abd["drop_right_sensors"]["h_ci_width"] / abd["baseline"]["h_ci_width"],
    "{:.1f}",
)
m(
    "AbHWidthRatioEarly",
    abd["time_early_only"]["h_ci_width"] / abd["baseline"]["h_ci_width"],
    "{:.1f}",
)
m(
    "AbLatRatioConst",
    abd["assume_constant_a"]["val_rmse_latent"] / abd["baseline"]["val_rmse_latent"],
    "{:.0f}",
)

bi = pd.DataFrame(J("outputs/studies/bias/bias.json")["mean_by_variant"]).set_index(
    "variant"
)
for v_, tag in (
    ("correct_model", "Ok"),
    ("wrong_physics_offsets_free", "Wrong"),
    ("correct_physics_offsets_zero", "ZeroOff"),
):
    x = bi.loc[v_]
    m(f"Bias{tag}Train", x["train_rmse_noisy"], "{:.4f}")
    m(f"Bias{tag}Lat", x["val_rmse_latent"], "{:.4f}")
    m(f"Bias{tag}BErr", abs(x["beta_err"]), "{:.3f}")
    m(f"Bias{tag}Chi", x["chi2_red"], "{:.2f}")

de = C("outputs/studies/design/design_rows.csv")
g = de.groupby("variant")
w = g["h_ci_width"].mean()
m("DesBaseHW", w["base"], "{:.4f}")
m("DesAHW", w["A_new_sensor_x0.95"], "{:.4f}")
m("DesBHW", w["B_denser_existing"], "{:.4f}")
piv = de.pivot(index="dataset", columns="variant", values="h_ci_width")
m("DesAWins", int((piv["A_new_sensor_x0.95"] < piv["B_denser_existing"]).sum()), "{:d}")
m("DesN", len(piv), "{:d}")
dj = J("outputs/studies/design/design.json")
m("DesRuleA", sum(d["chosen"] == "A_new_sensor_x0.95" for d in dj["decisions"]), "{:d}")
e = de.assign(ae=de["h_err"].abs()).groupby("variant")["ae"].mean()
m("DesAErr", e["A_new_sensor_x0.95"], "{:.4f}")
m("DesBErr", e["B_denser_existing"], "{:.4f}")

# --- diagnosis -----------------------------------------------------------------------------
dv = C("outputs/diagnosis_validation/rule_validation.csv")
dv["ok"] = dv.apply(
    lambda x: (
        x.verdict
        == (
            "no_fault_detected"
            if x.true_fault == "none"
            else f"suspected:{x.true_fault}"
        )
    ),
    axis=1,
)
m("DiagValOk", int(dv.ok.sum()), "{:d}")
m("DiagValN", len(dv), "{:d}")
sw = C("outputs/diagnosis_sweep/sensitivity_sweep.csv")
named = sw[sw.verdict.str.startswith("suspected")]
m("SweepNamed", len(named), "{:d}")
m("SweepNamedWrong", int((~named.correct).sum()), "{:d}")
m("SweepN", len(sw), "{:d}")
m("SweepDetected", int(sw.correct.sum()), "{:d}")
m("SweepAmbig", int(sw.verdict.str.startswith("ambiguous").sum()), "{:d}")
m("SweepVOneMiss", int(((sw.top_delta_chi2 >= 15.1) & (sw.top_delta_chi2 < 25) & sw.correct).sum()), "{:d}")
import ast
berr = []
for i, x in sw.iterrows():
    t = J(f"outputs/diagnosis_sweep/data/sweep_{8000 + i}_truth.json")["params"]
    berr.append(abs(ast.literal_eval(x.base_params)["beta"] - t["beta"]))
sw["beta_abs_err"] = berr
miss = sw[sw.verdict == "no_fault_detected"]  # strictly undetected (ambiguous verdicts still flag a fault)
m("SweepMissBetaErr", miss.beta_abs_err.max(), "{:.2f}")
m("SweepMissDrift", miss[miss.true_fault == "sensor_drift"].rate.max(), "{:.2f}")
m("SweepMissQ", miss[miss.true_fault == "heat_source"].Q.max(), "{:.2f}")
m("SweepMissBias", miss[miss.true_fault == "boundary_input_error"].bias.max(), "{:.2f}")
nu = C("outputs/diagnosis_null/null_tops.csv")
nulls = list(nu.top_delta_chi2) + list(dv[dv.true_fault == "none"].top_delta_chi2)
m("NullMax", max(nulls), "{:.1f}")
m("NullN", len(nulls), "{:d}")
m("NullFalse", int((nu.verdict != "no_fault_detected").sum() + (~dv[dv.true_fault == "none"].ok).sum()), "{:d}")
sd = J("results/sealed_diagnosis.json")
m("SealVerdict", sd["verdict"].replace("_", " "))
m("SealDchi", sd["top_hypothesis"]["delta_chi2"], "{:.0f}")
m("SealQ", sd["top_hypothesis"]["estimate"], "{:.3f}")
m("SealQsd", sd["top_hypothesis"]["sd"], "{:.3f}")
m("SealBaseH", sd["observations"]["base_params"]["h"], "{:.3f}")
st = J("data/sealed_v1_truth.json")
m("SealTrueQ", st["fault"]["Q"], "{:.3f}")
m("SealTrueXc", st["fault"]["xc"], "{:.2f}")
m("SealTrueH", st["params"]["h"], "{:.3f}")
m("SealHeatRatio", 100 * abs(sd["top_hypothesis"]["estimate"] * 0.15 / (st["fault"]["Q"] * st["fault"]["w"]) - 1), "{:.0f}")
best_hs = max(
    (t for t in sd["hypothesis_tests"] if t["kind"] == "heat_source"),
    key=lambda t: t["delta_chi2"],
)
m("SealFixedH", best_hs["phys"]["h"], "{:.3f}")

# --- recovery / smoke ----------------------------------------------------------------------
rc = J("outputs/restart/restart_check.json")
m("RestartDiff", rc["checks"]["deterministic_restart"]["max_param_diff"], "{:.1g}")
m("RestartAll", "all pass" if rc["all_pass"] else "FAIL")
sm = J("outputs/smoke/smoke.json")
m("SmokeSec", sm["total_seconds"], "{:.0f}")
m("SmokeMin", sm["total_seconds"] / 60, "{:.1f}")
m("SmokeOk", "PASS" if sm["ok"] else "FAIL")

# --- audit additions (claims checked in the final analysis) -------------------------------
for k, n in (("a0", "A"), ("beta", "B"), ("h", "H")):
    m(f"PinnErr{n}", 100 * abs(sp[k]["mean"] - tr[k]) / abs(tr[k]), "{:.0f}")
    gaps = [abs(x["fv_recheck_param_gap_in_fv_sd"][k]) for x in runs]
    m(f"PinnGap{n}Min", min(gaps), "{:.1f}")
    m(f"PinnGap{n}Max", max(gaps), "{:.1f}")
fvp = r["fv_variable"]["fit"]["params"]
m("FvErrB", 100 * abs(fvp["beta"] - tr["beta"]) / abs(tr["beta"]), "{:.0f}")
zs = [cv["rho=0.0"][k]["z_sd"] for k in ("a0", "beta", "h")]
m("CovZsdMin", min(zs), "{:.2f}")
m("CovZsdMax", max(zs), "{:.2f}")
ex = [(nr[k]["empirical_sd_over_redraws"] / nr[k]["mean_laplace_sd"] - 1) / 0.16 for k in ("a0", "beta", "h")]
m("RedrawSeMin", min(ex), "{:.1f}")
m("RedrawSeMax", max(ex), "{:.1f}")
m("RedrawRatioB", nr["beta"]["empirical_sd_over_redraws"] / nr["beta"]["mean_laplace_sd"], "{:.2f}")
c = abd["assume_constant_a"]
m("AbConstAChi", c["chi2_red"], "{:.2f}")
m("AbConstALim", 1 + 3 * np.sqrt(2 / (c["n_obs"] - 7)), "{:.2f}")
m("AbBWidthRatioLeft", abd["drop_left_sensors"]["beta_ci_width"] / abd["baseline"]["beta_ci_width"], "{:.1f}")
import re as _re
import subprocess
import sys

col = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                     capture_output=True, text=True).stdout
m("NTests", int(_re.search(r"(\d+) tests? collected", col).group(1)), "{:d}")

text = (
    "% AUTO-GENERATED by report/make_numbers.py -- do not edit by hand.\n% SOURCES: "
    + ", ".join(sorted(sources))
    + "\n"
    + "\n".join(out)
    + "\n"
)
(R / "report" / "numbers.tex").write_text(text)
print(f"wrote {len(out)} macros from {len(sources)} files")
