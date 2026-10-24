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
pip install -r requirements.txt   # pinned; full lock in requirements-lock.txt
pip install -e . --no-deps

python -m heatinv smoke                                   # all stages, reduced size, < 10 min CPU
python -m heatinv evaluate --config configs/public.yaml   # THE public evaluation command
pytest -q                                                 # focused checks (~1.5 min)
```

| Command | What it does | Output |
|---|---|---|
| `python -m heatinv verify` | MMS refinement, steady state, energy balance | `outputs/verify/verify.json` |
| `python -m heatinv fit --method fv\|constant\|pinn` | fit on the training split only | `outputs/public/fit_<m>.json` |
| `python -m heatinv predict --method fv\|constant\|pinn` | fit + predict every query of the prediction experiment | `predictions_<m>.csv` |
| `python -m heatinv evaluate [--no-pinn]` | fit → validate → predict → (synthetic) score, all methods | `results.json`, `estimates.json`, `predictions_*.csv`, `cleaning_log.csv` |
| `python -m heatinv study coverage\|ablation\|bias\|design` | Stage 3 studies on separate synthetic data | `outputs/studies/<name>/` |
| `python -m heatinv diagnose --bundle DIR` | Stage 4 diagnosis, frozen rules `configs/diagnosis_rules.yaml` | `outputs/diagnosis/diagnosis.json` |
| `python -m heatinv restart` | checkpoint resume / refusal / divergence checks | `outputs/restart/restart_check.json` |
| `python -m heatinv generate --spec configs/synth_public.yaml` | rebuild the synthetic public bundle | `data/public_v1/` |

**Resources.** Reference machine: Lenovo laptop, Intel i7-1165G7 (4 cores / 8 threads), 16 GB RAM,
no GPU used, Windows 11, Python 3.11.9, torch 2.12.0+cpu, 4 torch threads.
FV fit: ~5 s. One PINN seed (5000 Adam + 300 L-BFGS): ~5 min. Full `evaluate` incl. 3 PINN seeds: ~20 min.
Memory < 1 GB.

## Data format (input contract)

A bundle is a directory of CSVs (any row order; IDs are arbitrary strings; any number of
sensors/experiments/readings; sensor x anywhere in [0,1]):

| File | Columns |
|---|---|
| `manifest.json` | `schema_version`, `reference_sensor`, `synthetic`, file hashes |
| `sensors.csv` | `sensor_id, x[, is_reference]` |
| `observations.csv` | `obs_id, experiment_id, sensor_id, t, y, noise_sd, quality_flag` (`ok`/`suspect`/`bad`) |
| `queries.csv` | `query_id, experiment_id, sensor_id, t` |
| `initial_profiles.csv` | `experiment_id, x, u0` (piecewise-linear in x, must span [0,1]) |
| `boundary_histories.csv` | `experiment_id, t, g, Ta` (piecewise-linear in t, must span [0,1]) |

Cleaning is explicit and logged row-by-row with the raw CSV line (`cleaning_log.csv`):
unknown IDs, invalid times, missing values and non-positive noise are dropped; `bad` flags are
dropped, `suspect` kept and logged; a repeated `obs_id` with identical content keeps one copy, a
repeated `obs_id` with conflicting content drops **all** copies (no silent tie-breaking).
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
               studies.py diagnose.py recovery.py runlog.py smoke.py cli.py
configs/       public.yaml  synth_public.yaml  diagnosis_rules.yaml (frozen)
data/          public_v1/ (synthetic bundle)  truth/ (synthetic truth, scoring only)
tests/         solver, data validation, recovery/prediction/mismatch, restart
docs/evidence/ logs of failed / misleading runs discussed in the report
report/        report (<= 6 pages + 1 page AI use)
```
