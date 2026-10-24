# PINN development log (dev bundle, synthetic; truth a0=0.0394, beta=0.360, h=0.377)

All trials: same synthetic DEVELOPMENT bundle (seed 7, with record defects), train E1-E3, seed 0,
4x48 tanh MLP per experiment, 2000 collocation points/experiment. The public evaluation bundle was
not used for any of these choices. FV reference fit on the same data: a0=0.0399, beta=0.382, h=0.367.

| Trial | Change (one at a time) | a0 | beta | h | L_obs | Outcome |
|---|---|---|---|---|---|---|
| 1 | baseline: lr_phys=2e-2, no warm-up | 0.0374 | 0.321 | **0.031** | 1.78 | **failed**: h on lower bound (sigmoid saturated) |
| 2 | warm-up: physics frozen 1500 steps, lr_phys=5e-3 | 0.0394 | 0.380 | 0.133 | 1.16 | a0, beta right; h 3x low; S5 offset absorbs it |
| 3 | trial 2 + w_bc=10 | 0.0380 | 0.243 | 0.040 | 1.89 | hypothesis "BC weight pins h" **refuted** |
| 4 | trial 2 + lr decay, 10k Adam + 500 L-BFGS | 0.0395 | 0.373 | 0.155 | 1.11 | marginal h gain for 2x runtime |
| 5 | trial 4 settings (5k steps) + w_pde=10 | 0.0394 | 0.109 | 0.095 | 2.27 | worse: stiffer optimisation |

**Mechanism.** Losses are in noise units, so the PDE/BC are enforced only to ~1 noise level. Between
the last sensor (x=0.85) and the Robin end no data constrain the field, and a residual of that size
is enough for the field to bend so that a wrong h satisfies the Robin condition, while the offset of
the sensor nearest the end (S5: fitted -0.006 vs true +0.018) absorbs the remaining misfit. The FV
fit has no such freedom (PDE and BC exact), so it recovers h.

**Detecting check (truth-free, now automatic).** Solve the PDE exactly (FV) with the PINN's
parameters and offsets and compute chi^2_red on the TRAINING readings: truth 1.00, trial 2 1.37,
trial 3 2.29 (limit 1 + 3 sqrt(2/dof) = 1.21). A PINN whose field fits but whose parameters do
not explain the data under exact physics is reported as `weak_evidence`.

**Frozen PINN configuration** (configs/public.yaml): trial 2 + exponential lr decay, 5000 Adam +
300 L-BFGS. Decision: stop tuning for h; the PINN is reported for a0/beta, with h flagged.
