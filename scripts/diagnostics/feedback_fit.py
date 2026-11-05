"""Feedback fit: does how the policy is fit decide whether its own feedback stabilizes or destabilizes?

policy_drift (round 8 of the coupled toward-policy run) showed the policy's own
feedback makes deviations from a walk grow about 10x every 0.2 s, faster than
no feedback at all (10x per 0.7 s), while its on-walk actions with the walk's
LQR feedback finish 4 of 6 walks. Training longer on the same 6 walks lowers
the gap but not takeover, suggesting the network bends to fit them and its
slopes are what destabilize. This fits fresh policies on one fixed set of
trajectories and measures their feedback. Writes only to --out.

Design, fixed before the first run:

- Trajectories: the 6 saved at --round by a similarity_loop run. Each policy
  starts fresh (S1 network, S1 normalization of the bank's walking states,
  seed 0) and is fit with the S1 recipe on those trajectories only.
- Variants: clean (observation, control) pairs for 2,000, 10,000 and 40,000
  updates; noise targets (8 per control, sd 0.3, LQR gains capped at 3) for
  10,000 and 40,000; controller rollouts (16 per walk, action noise 0.2,
  capped gains) for 10,000.
- Per variant and walk: on-path action MSE; "walk + policy" (u = u_t + pi(x) -
  pi(x_bar_t)) and "policy" (u = pi(x)) driven from the walk's start, controls
  survived of 1,000; the feedback growth rate: the slope of log10 of the
  walk + policy state gap per second, fit from the first control until the gap
  passes 0.01 sd or it falls; the same for open loop (u = u_t) as reference.
- Readings: "training length decides the feedback" if the clean variants'
  median growth rates differ by >= 2 decades per second between 2,000 and
  40,000 updates; "a fit makes the feedback stabilizing" if any variant's
  median growth rate is below open loop's and walk + policy survives all 1,000
  controls on >= 4 of 6 walks.

Added after the first two runs (round 8 of the fixed-weight run; round 4 of the
x4-weight run, where plain fits got worse feedback as the walks got closer,
4.1-4.6 decades/s, and 40,000-update noise targets the best, 1.9): the 2015
paper's hidden-unit noise, Gaussian noise of sd h added to each hidden tanh
output during fitting only (its "no noise" ablation removed this and the
sensor noise together). Variants: noise targets 40,000 updates with h = 0.01
(the paper's) and 0.1, and clean 40,000 with h = 0.1. Same measures and
readings.

Added with the hidden-noise variants: "slope agreement" per walk, 1 -
sum_t |J_t + K_t|^2 / sum_t |K_t|^2, where J_t is the policy's derivative of
its action with respect to the sd-scaled state at x_bar_t (autograd through
the observation, zero where velocities are clipped) and K_t the capped LQR
gains, so 1 means the policy's own feedback equals the walk's and 0 means no
better than no feedback. Each fitted policy is saved (policy.pt).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.diagnostics.feedback_rollouts import linearize, lqr
from scripts.diagnostics.feedback_tube import capped
from scripts.diagnostics.loop_task import WALKER
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.policy_decomposition import controller_rollouts
from scripts.diagnostics.policy_drift import trace
from scripts.diagnostics.policy_picks import load_bank
from scripts.diagnostics.similarity_loop import SOURCES, R, draws, tube
from scripts.diagnostics.smooth_teacher_students import RECIPE
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps import policy as policy_module
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

CAP = 3.0
VARIANTS = (
    ("clean 2k", "clean", 2_000, 0.0),
    ("clean 10k", "clean", 10_000, 0.0),
    ("clean 40k", "clean", 40_000, 0.0),
    ("noise targets 10k", "noise", 10_000, 0.0),
    ("noise targets 40k", "noise", 40_000, 0.0),
    ("controller rollouts 10k", "rollouts", 10_000, 0.0),
    ("noise targets 40k hidden 0.01", "noise", 40_000, 0.01),
    ("noise targets 40k hidden 0.1", "noise", 40_000, 0.1),
    ("clean 40k hidden 0.1", "clean", 40_000, 0.1),
)


def hidden_noise(size):
    """Adds N(0, size^2) to every hidden tanh output while a module is training."""
    generator = torch.Generator().manual_seed(0)

    def hook(module, inputs, output):
        if isinstance(module, torch.nn.Tanh) and module.training and output.shape[-1] != 6:
            return output + size * torch.randn(output.shape, generator=generator)
        return None

    return torch.nn.modules.module.register_module_forward_hook(hook)


def growth(gap):
    """Slope of log10 gap per second from the first control to 0.01 sd (or the fall)."""
    end = np.flatnonzero(gap > 0.01)
    end = end[0] + 1 if len(end) else len(gap)
    g = np.log10(np.maximum(gap[:end], 1e-16))
    if end < 3:
        return float("nan")
    t = np.arange(end) / 125
    return float(np.polyfit(t, g, 1)[0])


def slopes(p, xbar, scale, spec=WALKER):
    """d pi / d (sd-scaled state) at each x_bar_t: (T, nu, n)."""
    obs = torch.tensor(spec.observation(xbar).astype(np.float32), requires_grad=True)
    out = p(obs)
    J = np.stack(
        [
            torch.autograd.grad(out[:, i].sum(), obs, retain_graph=True)[0].numpy()
            for i in range(out.shape[1])
        ],
        axis=1,
    )
    full = np.zeros((len(xbar), out.shape[1], xbar.shape[1] - 1))
    full[:, :, spec.observed - 1] = J * spec.slope_factor(xbar, scale)[:, None, :]
    return full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--variant", required=True, choices=[v[0] for v in VARIANTS] + ["open"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK, nthread=args.threads)
    bank = load_bank(args.bank)
    pooled = np.concatenate([observation(e["states"]) for e in bank])
    normalization = dict(mean=pooled.mean(0), std=np.maximum(pooled.std(0), 1e-6))
    walks = []
    with np.load(args.run / f"round_{args.round}.npz") as z:
        for s in SOURCES:
            with np.load(args.bank / f"episode_{s}.npz") as d:
                start, scale = d["states"][0], d["states"][:, 1:].std(0) + 1e-6
            st, u = z[f"states_{s}"], z[f"controls_{s}"]
            xbar = np.concatenate([start[None], st[:-1]])
            K = capped(lqr(*linearize(env, xbar, u, scale), R), CAP)
            walks.append(dict(source=s, start=start, scale=scale, st=st, u=u, xbar=xbar, K=K))

    rows = []
    if args.variant == "open":
        for w in walks:
            seen = trace(env, None, w["start"], w["xbar"], w["u"], w["K"], w["scale"], "open loop")
            gap = np.sqrt(
                (((seen[:, 2:] - w["st"][: len(seen), 2:]) / w["scale"][1:]) ** 2).mean(1)
            )
            rows.append(dict(source=w["source"], survived=len(seen), growth=growth(gap)))
    else:
        kind, updates, hidden = {v[0]: v[1:] for v in VARIANTS}[args.variant]
        obs, acts = [], []
        for w in walks:
            x, u = w["xbar"], w["u"]
            if kind == "noise":
                xs, us = tube(
                    x[None], u[None], w["K"], draws(0.3, len(u), 1, w["source"]), w["scale"]
                )
                x, u = xs.reshape(-1, xs.shape[-1]), us.reshape(-1, 6)
            elif kind == "rollouts":
                x, u = controller_rollouts(
                    env, w["start"], x, u, w["K"], w["scale"], 0.2, [1, w["source"]]
                )
            obs.append(observation(x))
            acts.append(u)
        handle = hidden_noise(hidden) if hidden > 0 else None
        p, _, _ = policy_module.fit_policy(
            policy_module.Policy(0, "cpu", normalization, hidden_dim=128),
            np.concatenate(obs).astype(np.float32),
            np.concatenate(acts).astype(np.float32),
            updates=updates,
            seed=0,
            **RECIPE,
        )
        if handle is not None:
            handle.remove()
        p.eval()
        p.checkpoint(args.out / "policy.pt")
        for w in walks:
            on_path = p.predict(observation(w["xbar"]).astype(np.float32)).astype(np.float64)
            row = dict(source=w["source"], on_path_mse=float(((on_path - w["u"]) ** 2).mean()))
            J = slopes(p, w["xbar"], w["scale"])
            row["slope_agreement"] = float(
                1.0 - ((J + w["K"]) ** 2).sum() / max((w["K"] ** 2).sum(), 1e-12)
            )
            for arm in ("walk + policy", "policy"):
                seen = trace(env, p, w["start"], w["xbar"], w["u"], w["K"], w["scale"], arm)
                gap = np.sqrt(
                    (((seen[:, 2:] - w["st"][: len(seen), 2:]) / w["scale"][1:]) ** 2).mean(1)
                )
                row[arm] = dict(survived=len(seen), growth=growth(gap))
            rows.append(row)
    summary = dict(variant=args.variant, rows=rows, seconds=time.perf_counter() - started)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
