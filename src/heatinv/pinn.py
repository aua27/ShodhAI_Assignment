"""Inverse PINN: one field network per experiment, shared physical parameters and sensor offsets.

    u_e(x, t) = MLP_e(x, t)                       one small tanh MLP per training experiment
    a0, beta, h  = lo + (hi - lo) * sigmoid(z)    bounded by construction
    b_s          = 0.03 * tanh(z_s)               bounded; reference sensor fixed at 0

Losses, all expressed in "noise units" (divided by sigma_ref^2, sigma_ref = median sensor sd), so
a constraint violated by about one noise level costs about as much as one misfitted reading:
    L_obs = mean( ((u + b_s - y) / sigma_k)^2 )
    L_pde = mean( (u_t - d/dx[a(x) u_x])^2 ) / sigma_ref^2     conservative form, autograd twice:
            flux = a(x) * u_x;  d(flux)/dx via autograd -> includes a'(x) u_x automatically
    L_ic  = mean( (u(x,0) - f(x))^2 ) / sigma_ref^2
    L_bc  = mean( (u(0,t) - g(t))^2 ) / sigma_ref^2 + mean( (a(1) u_x(1,t) + h (u(1,t) - Ta(t)))^2 ) / sigma_ref^2
Total = w_obs L_obs + w_pde L_pde + w_ic L_ic + w_bc L_bc (default weights 1; w_bc and w_pde ablations in docs/evidence/README.md).

Optimisation: Adam with collocation points resampled every `resample_every` steps, then L-BFGS
on a FIXED point set (resampling inside L-BFGS's line search breaks its curvature model).

Unseen experiments (validation / prediction): "physics-only adaptation" -- a fresh field network is
trained on PDE + IC + BC losses only with the physical parameters frozen. It uses no labels and is
timed and reported separately.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from .physics import BOUNDS, OFFSET_BOUND
from .runlog import Checkpointer, CheckpointMismatch


@dataclass
class PINNConfig:
    width: int = 48
    depth: int = 4
    n_colloc: int = 2000  # per experiment
    n_ic: int = 200
    n_bc: int = 200
    adam_steps: int = 5000
    lr: float = 2e-3
    lr_phys: float = 5e-3
    lr_final_frac: float = (
        0.1  # Adam lr decays exponentially to this fraction of its start
    )
    freeze_phys_steps: int = (
        1500  # physics params + offsets held at mid-range while fields warm up
    )
    lbfgs_iters: int = 300
    resample_every: int = 250
    w_obs: float = 1.0
    w_pde: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 1.0
    sigma_ref: Optional[float] = None  # default: median training noise sd
    dtype: str = "float32"
    seed: int = 0
    adapt_steps: int = 2000
    adapt_lbfgs: int = 150
    ckpt_every: int = 500
    threads: int = 4
    fail_obs_loss: float = (
        4.0  # final L_obs above this (i.e. misfit > 2 sigma rms) -> failed run
    )


class FieldNet(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(
            torch.cat([2 * x - 1, 2 * t - 1], dim=1)
        )  # inputs scaled to [-1, 1]


def _bounded(z, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(z)


class InversePINN(nn.Module):
    def __init__(self, exp_ids, offset_sensors, cfg: PINNConfig):
        super().__init__()
        self.exp_ids = list(exp_ids)
        self.offset_sensors = list(offset_sensors)
        self.nets = nn.ModuleDict(
            {e: FieldNet(cfg.width, cfg.depth) for e in self.exp_ids}
        )
        self.z_phys = nn.Parameter(torch.zeros(3))  # a0, beta, h -> start at mid-range
        self.z_off = nn.Parameter(torch.zeros(len(self.offset_sensors)))

    def phys(self):
        a0 = _bounded(self.z_phys[0], *BOUNDS["a0"])
        beta = _bounded(self.z_phys[1], *BOUNDS["beta"])
        h = _bounded(self.z_phys[2], *BOUNDS["h"])
        return a0, beta, h

    def offsets(self):
        return OFFSET_BOUND * torch.tanh(self.z_off)

    def phys_dict(self):
        a0, beta, h = self.phys()
        return {"a0": a0.item(), "beta": beta.item(), "h": h.item()}

    def offsets_dict(self, reference):
        d = dict(
            zip(self.offset_sensors, self.offsets().detach().cpu().numpy().tolist())
        )
        d[reference] = 0.0
        return d


def pde_residual(net, x, t, a0, beta):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = net(x, t)
    u_x, u_t = torch.autograd.grad(u.sum(), (x, t), create_graph=True)
    flux = a0 * torch.exp(beta * (2 * x - 1)) * u_x  # a(x) u_x
    flux_x = torch.autograd.grad(flux.sum(), x, create_graph=True)[0]
    return u_t - flux_x


def robin_residual(net, t, a0, beta, h, Ta):
    x1 = torch.ones_like(t).requires_grad_(True)
    u = net(x1, t)
    u_x = torch.autograd.grad(u.sum(), x1, create_graph=True)[0]
    a1 = a0 * torch.exp(beta)
    return a1 * u_x + h * (u - Ta)


class _ExpTensors:
    """Per-experiment tensors: observations, plus callables for known inputs."""

    def __init__(self, exp, offset_index, dtype):
        self.id = exp.id
        f = exp.forcing
        self.forcing = f
        o = exp.obs
        T = lambda a: torch.tensor(np.asarray(a, float), dtype=dtype).reshape(-1, 1)  # noqa: E731
        self.x, self.t, self.y, self.sd = (
            T(o["x"]),
            T(o["t"]),
            T(o["y"]),
            T(o["noise_sd"]),
        )
        self.oidx = torch.tensor(
            [offset_index.get(s, -1) for s in o["sensor_id"]], dtype=torch.long
        )
        self.T = T


def _sample_points(gen, n, dtype):
    return torch.rand(n, 1, generator=gen, dtype=dtype), torch.rand(
        n, 1, generator=gen, dtype=dtype
    )


class Trainer:
    """Joint inverse training with checkpoint/resume, divergence handling and provenance."""

    def __init__(
        self,
        experiments,
        reference,
        cfg: PINNConfig,
        ckpt: Optional[Checkpointer] = None,
    ):
        self.cfg = cfg
        self.dtype = getattr(torch, cfg.dtype)
        torch.set_num_threads(cfg.threads)
        self.reference = reference
        sensors = sorted({s for e in experiments for s in e.obs["sensor_id"]})
        offset_sensors = [s for s in sensors if s != reference]
        self.offset_index = {s: i for i, s in enumerate(offset_sensors)}
        torch.manual_seed(cfg.seed)  # network init; collocation uses self.gen
        self.model = InversePINN([e.id for e in experiments], offset_sensors, cfg).to(
            self.dtype
        )
        self.data = [_ExpTensors(e, self.offset_index, self.dtype) for e in experiments]
        all_sd = np.concatenate(
            [e.obs["noise_sd"].to_numpy(float) for e in experiments]
        )
        self.sigma_ref = cfg.sigma_ref or float(np.median(all_sd))
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.ckpt = ckpt
        self.step = 0
        self.history = []
        self.events = []
        self._make_optim(cfg.lr, cfg.lr_phys)
        self._resample()

    def _make_optim(self, lr, lr_phys):
        net_params = [
            p for n, p in self.model.named_parameters() if n.startswith("nets.")
        ]
        self.opt = torch.optim.Adam(
            [
                {"params": net_params, "lr": lr},
                {"params": [self.model.z_phys, self.model.z_off], "lr": lr_phys},
            ]
        )

    def _resample(self):
        c, d = self.cfg, self.dtype
        self.pts = []
        for ex in self.data:
            xc, tc = _sample_points(self.gen, c.n_colloc, d)
            xi = torch.rand(c.n_ic, 1, generator=self.gen, dtype=d)
            tb = torch.rand(c.n_bc, 1, generator=self.gen, dtype=d)
            f = ex.forcing
            self.pts.append(
                {
                    "xc": xc,
                    "tc": tc,
                    "xi": xi,
                    "fi": ex.T(f.f_at(xi.numpy().ravel())),
                    "tb": tb,
                    "gb": ex.T(f.g_at(tb.numpy().ravel())),
                    "Tab": ex.T(f.Ta_at(tb.numpy().ravel())),
                }
            )

    def losses(self):
        m, s2 = self.model, self.sigma_ref**2
        a0, beta, h = m.phys()
        off = torch.cat(
            [m.offsets(), torch.zeros(1, dtype=self.dtype)]
        )  # index -1 -> reference (0)
        L = {"obs": 0.0, "pde": 0.0, "ic": 0.0, "bc": 0.0}
        for ex, p in zip(self.data, self.pts):
            net = m.nets[ex.id]
            u = net(ex.x, ex.t)
            L["obs"] = (
                L["obs"]
                + (((u + off[ex.oidx].unsqueeze(1) - ex.y) / ex.sd) ** 2).mean()
            )
            L["pde"] = (
                L["pde"]
                + (
                    pde_residual(net, p["xc"].clone(), p["tc"].clone(), a0, beta) ** 2
                ).mean()
                / s2
            )
            L["ic"] = (
                L["ic"]
                + ((net(p["xi"], torch.zeros_like(p["xi"])) - p["fi"]) ** 2).mean() / s2
            )
            bl = ((net(torch.zeros_like(p["tb"]), p["tb"]) - p["gb"]) ** 2).mean()
            br = (
                robin_residual(net, p["tb"].clone(), a0, beta, h, p["Tab"]) ** 2
            ).mean()
            L["bc"] = L["bc"] + (bl + br) / s2
        n = len(self.data)
        L = {k: v / n for k, v in L.items()}
        c = self.cfg
        L["total"] = (
            c.w_obs * L["obs"]
            + c.w_pde * L["pde"]
            + c.w_ic * L["ic"]
            + c.w_bc * L["bc"]
        )
        return L

    # ---- checkpointing -----------------------------------------------------------------------
    def state(self, phase):
        return {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step,
            "phase": phase,
            "history": self.history,
            "events": self.events,
            "pts": self.pts,
            "rng": {"torch": torch.get_rng_state(), "gen": self.gen.get_state()},
        }

    def load_state(self, s):
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["opt"])
        self.step, self.history, self.events, self.pts = (
            s["step"],
            s["history"],
            s["events"],
            s["pts"],
        )
        torch.set_rng_state(s["rng"]["torch"])
        self.gen.set_state(s["rng"]["gen"])
        return s["phase"]

    def _record(self, L, phase):
        rec = {
            "step": self.step,
            "phase": phase,
            **{k: float(v.detach()) for k, v in L.items()},
            **self.model.phys_dict(),
        }
        self.history.append(rec)

    def train(self, resume=False, interrupt_at: Optional[int] = None, verbose=False):
        """Adam then L-BFGS. `interrupt_at` simulates a crash (raises) for the recovery test."""
        c = self.cfg
        t0 = time.perf_counter()
        phase = "adam"
        if resume:
            if self.ckpt is None:
                raise CheckpointMismatch(
                    "resume requested without a checkpoint directory"
                )
            st = self.ckpt.load_latest()
            if st is None:
                raise CheckpointMismatch("no compatible checkpoint found")
            phase = self.load_state(st)
            self.events.append({"step": self.step, "event": "resumed_from_checkpoint"})
        last_good = copy.deepcopy(self.state(phase)) if phase == "adam" else None
        lr_scale = 1.0
        while phase == "adam" and self.step < c.adam_steps:
            if interrupt_at is not None and self.step == interrupt_at:
                raise KeyboardInterrupt(f"simulated interruption at step {self.step}")
            if self.step % c.resample_every == 0 and self.step > 0:
                self._resample()
            L = self.losses()
            if not torch.isfinite(L["total"]):
                # Numerical divergence: record it, roll back to the last good state, halve the lr.
                self.events.append(
                    {
                        "step": self.step,
                        "event": "non_finite_loss",
                        "action": "rollback+halve_lr",
                    }
                )
                lr_scale *= 0.5
                if lr_scale < 1 / 16:
                    self.events.append(
                        {"step": self.step, "event": "diverged", "action": "abort"}
                    )
                    return self._result(t0, failed="diverged")
                events = (
                    self.events
                )  # the rollback must not erase the record of the divergence
                self.load_state(copy.deepcopy(last_good))
                self.events = events
                self._make_optim(c.lr * lr_scale, c.lr_phys * lr_scale)
                continue
            self.opt.zero_grad()
            L["total"].backward()
            if self.step < c.freeze_phys_steps:
                # Warm-up: fields learn data + constraints before the physics may move. Without this,
                # an untrained field makes the Robin term cheapest to satisfy by driving h -> 0,
                # where the sigmoid saturates and h never recovers (docs/evidence/pinn_trial1_*).
                self.model.z_phys.grad = None
                self.model.z_off.grad = None
            # Exponential decay to lr_final_frac of the base rate; a function of step only, so a
            # resumed run reproduces it exactly.
            decay = c.lr_final_frac ** (self.step / c.adam_steps)
            for grp, base in zip(self.opt.param_groups, (c.lr, c.lr_phys)):
                grp["lr"] = base * lr_scale * decay
            self.opt.step()
            self.step += 1
            if self.step % 50 == 0 or self.step == 1:
                self._record(L, phase)
                if verbose and self.step % 500 == 0:
                    print(
                        {
                            k: round(v, 4)
                            for k, v in self.history[-1].items()
                            if k != "phase"
                        }
                    )
            if self.step % c.ckpt_every == 0:
                last_good = copy.deepcopy(self.state(phase))
                if self.ckpt is not None:
                    self.ckpt.save(self.state(phase), self.step)
        if phase == "adam":
            phase = "lbfgs"
            self.opt = torch.optim.LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=c.lbfgs_iters,
                history_size=50,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
            )
            if self.ckpt is not None:
                self.ckpt.save(self.state(phase), self.step)

        def closure():
            self.opt.zero_grad()
            L = self.losses()
            L["total"].backward()
            return L["total"]

        if c.lbfgs_iters > 0 and phase == "lbfgs":
            before = copy.deepcopy(self.model.state_dict())
            self.opt.step(closure)
            L = self.losses()
            if not torch.isfinite(L["total"]):
                self.events.append(
                    {
                        "step": self.step,
                        "event": "lbfgs_non_finite",
                        "action": "revert_to_adam_result",
                    }
                )
                self.model.load_state_dict(before)
            phase = "done"
        L = self.losses()
        self._record(L, "final")
        return self._result(t0)

    def _result(self, t0, failed: Optional[str] = None):
        L = self.losses()
        fin = {k: float(v.detach()) for k, v in L.items()}
        reasons = []
        if failed:
            reasons.append(failed)
        if not np.isfinite(fin["total"]):
            reasons.append("non_finite_loss")
        if fin["obs"] > self.cfg.fail_obs_loss:
            reasons.append(f"obs_loss {fin['obs']:.2f} > {self.cfg.fail_obs_loss}")
        p = self.model.phys_dict()
        for k, (lo, hi) in BOUNDS.items():
            if min(p[k] - lo, hi - p[k]) < 0.01 * (hi - lo):
                reasons.append(f"{k} at bound")
        return {
            "seed": self.cfg.seed,
            "params": p,
            "offsets": self.model.offsets_dict(self.reference),
            "final_losses": fin,
            "status": "failed" if reasons else "ok",
            "fail_reasons": reasons,
            "runtime_s": time.perf_counter() - t0,
            "steps": self.step,
            "events": self.events,
            "history": self.history,
            "sigma_ref": self.sigma_ref,
            "config": asdict(self.cfg),
        }


def adapt(model: InversePINN, experiment, cfg: PINNConfig, sigma_ref: float):
    """Physics-only adaptation to an unseen experiment: PDE + IC + BC with frozen a0, beta, h."""
    dtype = getattr(torch, cfg.dtype)
    torch.manual_seed(cfg.seed + 1000)
    gen = torch.Generator().manual_seed(cfg.seed + 1000)
    net = FieldNet(cfg.width, cfg.depth).to(dtype)
    with torch.no_grad():
        a0, beta, h = [v.detach() for v in model.phys()]
    f = experiment.forcing
    T = lambda a: torch.tensor(np.asarray(a, float), dtype=dtype).reshape(-1, 1)  # noqa: E731
    s2 = sigma_ref**2

    def make_pts():
        xc, tc = _sample_points(gen, cfg.n_colloc, dtype)
        xi = torch.rand(cfg.n_ic, 1, generator=gen, dtype=dtype)
        tb = torch.rand(cfg.n_bc, 1, generator=gen, dtype=dtype)
        return (
            xc,
            tc,
            xi,
            T(f.f_at(xi.numpy().ravel())),
            tb,
            T(f.g_at(tb.numpy().ravel())),
            T(f.Ta_at(tb.numpy().ravel())),
        )

    def loss(P):
        xc, tc, xi, fi, tb, gb, Tab = P
        lp = (pde_residual(net, xc.clone(), tc.clone(), a0, beta) ** 2).mean() / s2
        li = ((net(xi, torch.zeros_like(xi)) - fi) ** 2).mean() / s2
        lb = (
            ((net(torch.zeros_like(tb), tb) - gb) ** 2).mean()
            + (robin_residual(net, tb.clone(), a0, beta, h, Tab) ** 2).mean()
        ) / s2
        return cfg.w_pde * lp + cfg.w_ic * li + cfg.w_bc * lb

    t0 = time.perf_counter()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    P = make_pts()
    for k in range(cfg.adapt_steps):
        if k % cfg.resample_every == 0 and k > 0:
            P = make_pts()
        opt.zero_grad()
        L = loss(P)
        L.backward()
        opt.step()
    if cfg.adapt_lbfgs > 0:
        lb = torch.optim.LBFGS(
            net.parameters(), max_iter=cfg.adapt_lbfgs, line_search_fn="strong_wolfe"
        )

        def closure():
            lb.zero_grad()
            v = loss(P)
            v.backward()
            return v

        lb.step(closure)
    final = float(loss(P))
    return net, {"adapt_runtime_s": time.perf_counter() - t0, "adapt_final_loss": final}




def energy_balance(net, forcing, a0, beta, h, dtype=torch.float32, nx=201, nt=101):
    """Integrated conservation check for a PINN field:  d/dt int u dx  vs  flux_in(0) - flux_out(1).

    Returns the RMS defect normalised by the RMS boundary flux (dimensionless; 0 = conserving).
    """
    xg = torch.linspace(0, 1, nx, dtype=dtype)
    tg = torch.linspace(0, 1, nt, dtype=dtype)
    X, Tt = torch.meshgrid(xg, tg, indexing="xy")
    x = X.reshape(-1, 1).clone().requires_grad_(True)
    t = Tt.reshape(-1, 1).clone().requires_grad_(True)
    u = net(x, t)
    u_x, u_t = torch.autograd.grad(u.sum(), (x, t))
    u_t = u_t.reshape(nt, nx).detach().numpy()
    u_x = u_x.reshape(nt, nx).detach().numpy()
    xs = xg.numpy()
    dEdt = np.trapezoid(u_t, xs, axis=1)
    a = a0 * np.exp(beta * (2 * xs - 1))
    flux_in = -a[0] * u_x[:, 0]
    flux_out = -a[-1] * u_x[:, -1]
    defect = dEdt - (flux_in - flux_out)
    scale = np.sqrt(np.mean(flux_in**2 + flux_out**2)) + 1e-12
    return float(np.sqrt(np.mean(defect**2)) / scale)

