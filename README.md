# Inverse Heat Modelling & Scientific Diagnostics

Estimate `a0`, `beta`, `h` and per-sensor offsets `b_s` for

```
u_t = d/dx[a(x) u_x],  a(x) = a0 exp(beta (2x-1)),  u(x,0)=f(x),  u(0,t)=g(t),  -a(1) u_x(1,t) = h (u(1,t)-Ta(t))
y   = u(x_s,t) + b_s + noise,  b_ref = 0
```

with a verified finite-volume (FV) reference solver + weighted least squares (primary method),
a constant-diffusivity baseline, and an inverse PINN (PyTorch), plus uncertainty, robustness,
diagnosis and recovery studies. Report: [`report/report.pdf`](report/report.pdf).

> **Data status.** Only the assignment PDF was received; the versioned data bundle and the Stage-4
> diagnosis dataset were not attached (requested by email). Everything here runs on **synthetic**
> bundles generated to the schema the brief describes, with ground truth stored separately and
> used only for scoring. Every synthetic result is labelled as such. A real bundle in the documented
> format (below) drops in with a one-line config change.

## Quick start (CPU only)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements-lock.txt   # exact versions of every package (tested on a fresh clone)
# (requirements.txt pins only the direct dependencies; transitive ones may then drift)
pip install -e . --no-deps

python -m heatinv smoke                                   # every stage, reduced size (97 s on a 2-vCPU cloud runner; budget 10 min)
python -m heatinv evaluate --config configs/public.yaml   # THE public evaluation command
pytest -q                                                 # 58 focused checks (~3 min)
```

| Command | What it does | Output |
|---|---|---|
| `python -m heatinv verify` | MMS refinement, steady state, energy balance | `outputs/verify/verify.json` |
| `python -m heatinv fit --method fv\|constant\|pinn` | fit on the training split only | `outputs/public/fit_<m>.json` |
| `python -m heatinv predict --method fv\|constant\|pinn` | fit + predict every query of the prediction experiment | `predictions_<m>.csv` |
| `python -m heatinv evaluate [--no-pinn]` | fit → validate → predict → (synthetic) score, all methods | `results.json`, `estimates.json`, `predictions_*.csv`, `cleaning_log.csv` |
| `python -m heatinv study coverage\|ablation\|bias\|design` | Stage 3 studies on separate synthetic data | `outputs/studies/<name>/` |
| `python -m heatinv study diagnosis_validation\|diagnosis_sweep --out outputs` | frozen diagnosis rules on known faults / fault sizes / fault-free nulls | `outputs/diagnosis_*/` |
| `python -m heatinv diagnose --bundle DIR` | Stage 4 diagnosis, frozen rules `configs/diagnosis_rules.yaml` | `outputs/diagnosis/diagnosis.json` |
| `python -m heatinv restart` | checkpoint resume / refusal / divergence checks | `outputs/restart/restart_check.json` |
| `python -m heatinv generate --spec configs/synth_public.yaml` | rebuild the synthetic public bundle | `data/public_v1/` |

**Resources.** Reference machine: Lenovo laptop, Intel i7-1165G7 (4 cores / 8 threads), 16 GB RAM,
no GPU used, Windows 11, Python 3.11.9, torch 2.12.0+cpu, 4 torch threads.
Measured (see `results/public/results.json` provenance and `outputs/smoke/smoke.json`): FV fit ~6 s;
one PINN seed (5000 Adam + 300 L-BFGS) ~4 min + ~1 min physics-only adaptation per seed;
full `evaluate` incl. 3 PINN seeds ~16 min; smoke 97 s on a clean 2-vCPU GitHub Actions runner
(`results/smoke_cloud_github_actions.json`; re-run via the `cpu-smoke` workflow) and load-dependent on the
laptop (measured 205 s idle for an earlier, larger version; 469 s for the current one with other apps running); all studies ~27 min; `pytest` ~3 min.
Memory < 1 GB. GPU not needed.

**Reproduce everything** (about 50 min):
```bash
python -m heatinv verify && python -m heatinv evaluate
for s in ablation bias coverage design; do python -m heatinv study $s; done
python -m heatinv study diagnosis_validation --out outputs && python -m heatinv study diagnosis_sweep --out outputs
python -m heatinv restart && python -m heatinv smoke
python -m heatinv.figures && python report/make_numbers.py && (cd report && pdflatex report.tex)
```
`report/make_numbers.py` writes every number in the report from these result files (no hand-typed numbers).

**Agreed machine.** The brief's "agreed machine" was not specified; the reference laptop above is
assumed (confirmation requested). The smoke run reports its own wall time and hardware.

**Focused checks required by the brief** (all in `pytest`):

| Check | Tests |
|---|---|
| solver correctness | `tests/test_solver.py` (MMS order, steady state, energy), `test_whitebox.py::test_maximum_principle_*` |
| known-truth recovery | `tests/test_inverse.py::test_known_truth_recovery`, `test_blackbox.py::test_truth_near_box_edges` |
| held-out prediction | `tests/test_inverse.py::test_held_out_prediction`, `test_prediction_needs_no_labels` |
| input validation | `tests/test_data.py`, `test_blackbox.py` (malformed inputs, IDs, layouts), `test_round2.py::test_loader_fuzz` |
| restart behaviour | `tests/test_restart.py` (resume, incompatible / corrupt checkpoint, divergence) |
| model mismatch | `tests/test_inverse.py::test_model_mismatch_flagged`, `test_whitebox.py::test_exact_physics_recheck_*` |
| withholding | `tests/test_inverse.py::test_withholding_when_unidentifiable` |
| no private-label access | `test_blackbox.py::test_truth_file_never_read_*`, `test_validation_labels_never_reach_fitting` |

## Data format (input contract)

A bundle is a directory of CSVs (any row order; IDs are arbitrary strings; any number of
sensors/experiments/readings; sensor x anywhere in [0,1]):

| File | Columns |
|---|---|
| `manifest.json` | `schema_version`, `reference_sensor`, `synthetic`, file hashes (recorded, not enforced; the logged data version is the sha256 over all files) |
| `sensors.csv` | `sensor_id, x[, is_reference]` |
| `observations.csv` | `obs_id, experiment_id, sensor_id, t, y, noise_sd, quality_flag` (`ok`/`suspect`/`bad`) |
| `queries.csv` | `query_id, experiment_id, sensor_id, t` |
| `initial_profiles.csv` | `experiment_id, x, u0` (piecewise-linear in x, must span [0,1]) |
| `boundary_histories.csv` | `experiment_id, t, g, Ta` (piecewise-linear in t, must span [0,1]) |

Cleaning is explicit and logged row-by-row with the raw CSV line (`cleaning_log.csv`):
unknown IDs, invalid times, missing values and non-positive noise are dropped; flags are read
case/whitespace-insensitively: `bad` dropped, `suspect` kept and logged, anything else dropped as
`unknown_flag`. Readings are identified by **(experiment_id, obs_id)** (IDs may restart per
experiment): a repeated key with identical content keeps one copy, with conflicting content drops
**all** copies (no silent tie-breaking). Missing/non-finite initial profiles or boundary histories
are refused.
Unusable inputs raise `BundleError` with the exact reason (missing column, input not covering
[0,1], query on an unknown sensor, no reference sensor, ...).

The split lives in the config (`split: train/val/test`). Only `train` observations reach any
fitting routine; validation labels are used only for scoring; the test experiment has no labels.

## Outputs

* `estimates.json` — primary estimates (FV, variable diffusivity), 95% intervals
  (**Gauss–Newton/Laplace, linearised Wald**), offsets, withheld parameters, `status` ∈
  {`ok`, `weak_evidence`, `failed`}, provenance.
* `predictions_<method>.csv` — by `query_id`: `u_latent` (noise-free temperature) with
  `u_lower/u_upper` (parameter uncertainty only) and `y_pred` (expected **reading** = u + b_s) with
  `y_lower/y_upper` (parameter + offset uncertainty + sensor noise). 95% level. For the PINN,
  lower/upper are the **min/max across seeds** — a spread, not a calibrated interval.
* `results.json` — everything: per-method fits, train/val MAE/RMSE (pooled over readings, weight 1
  each, and per experiment), interval coverage, energy-balance and consistency diagnostics, PINN
  per-seed runs incl. failures, cleaning summary, synthetic truth scoring (labelled synthetic).
* Every JSON has `provenance`: data sha256, config + hash, seed, git commit/dirty flag, hardware,
  threads, library versions, wall time.

## Repository layout

```
src/heatinv/   physics.py solver.py verify.py data.py synth.py inverse.py pinn.py pipeline.py
               studies.py diagnose.py sealed.py recovery.py runlog.py smoke.py figures.py cli.py
configs/       public.yaml  synth_public.yaml  diagnosis_rules.yaml (frozen)
data/          public_v1/ (synthetic bundle)  truth/ (synthetic truth, scoring only)
               sealed_v1/ + sealed_v1_truth.json (Stage-4 blind test; truth hash in manifest)
results/       final public evaluation outputs; sealed-diagnosis verdict; archived v1/v2 runs
tests/         58 tests: solver, data validation, inverse, restart, black-box, white-box
docs/          evidence/ (PINN development log, failed runs)
report/        report.pdf (5 pages incl. 1-page AI appendix), report.tex, make_numbers.py, figures/
```

## Results at a glance (synthetic)

See `report/report.pdf` (Table 1) and `results/public/estimates.json`. In short: the FV fit recovers
`a0, beta, h` inside 95% intervals; the constant-diffusivity baseline matches the noisy readings
almost as well but is rejected by the nested test; the PINN fits the readings but its parameters are off
(`h` badly, `a0` moderately; only `beta` agrees with FV) and every seed is reported `weak_evidence`
automatically; the frozen diagnosis rules identified the sealed blind fault.
