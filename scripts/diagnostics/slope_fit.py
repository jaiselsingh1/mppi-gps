"""Slope fit: can the policy carry a walk's feedback if its slopes are trained directly (Sobolev training)?

Noise targets teach the walk's feedback only through samples at perturbed
states: on the x4-weight run's round-4 walks the fitted policy recovers about
30% of the capped LQR gains (slope agreement 0.3, single walk 0.33, smoothed
gains <= 0.44), and its own feedback never stabilizes. Noise targets are a
finite-difference version of matching the policy's derivative to the gains;
Sobolev training (Czarnecki et al. 2017) matches the derivative directly. This
fits policies with loss |pi(x_bar_t) - u_t|^2 + lam |J_t + K_t|^2, J_t the
policy's derivative with respect to the sd-scaled state, and measures what
that buys. Writes only to --out.

Design, fixed before the first run:

- Trajectories: --round of a similarity_loop run, walks 0-6 (or only walk 0
  with --single). Gains: R = 1 LQR along each walk, singular values capped at 3.
- Network and inputs as S1 (17 -> 128 tanh -> 128 tanh -> 6 tanh, S1
  normalization), seed 0; Adam lr 3e-4, batch 256 controls, 20,000 updates.
  lam in {0, 0.1, 1, 10}.
- Measures per walk: on-path action MSE; slope agreement 1 - sum |J + K|^2 /
  sum |K|^2; controls survived (of 1,000) for walk + policy (u_t + pi(x) -
  pi(x_bar_t)) and policy (pi(x)).
- Readings: "slopes are learnable" if some lam reaches agreement >= 0.8 with
  on-path MSE <= 0.02; "learned slopes make the feedback work" if then walk +
  policy survives 1,000 on >= 4 of 6 walks; "the copy walks" if policy does.

Added after the first runs (single walk: lam 1 agreement 0.65 and walk + policy
survives 1,000; all 6 walks: lam 10 agreement 0.57, walk + policy survives
1,000 on 6/6 but on-path MSE 0.10-0.13 and the policy alone falls at about
200): --hidden, --layers and --updates, to test whether a larger network (256
units; 3 layers of 256 as in the 2015 paper) fit longer (60,000 updates) holds
both the actions and the slopes. lam in {3, 10}.

Added after the size runs started (round-8 walks behave like round 4: lam 10
holds walk + policy on 6/6 but on-path MSE about 0.11): --transverse matches
slopes only across the walk. Time-indexed LQR gains include a component along
the direction of motion that corrects timing (catch up with the schedule); a
state-feedback policy has no clock, and along the path its slope is already
fixed by how u_t changes along the walk, so asking for both conflicts. Orbital
(transverse) stabilization of walking gaits (Shiriaev; Manchester 2011) keeps
only the gains across the orbit. Loss lam |(J + K) P_t|^2 with P_t = I - v v^T,
v the unit direction of motion of x_bar in sd-scaled observed coordinates.
Also measured: transverse slope agreement 1 - sum |(J + K) P|^2 / sum |K P|^2.
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
from torch.func import jacrev, vmap

from scripts.diagnostics.feedback_fit import CAP, slopes
from scripts.diagnostics.feedback_rollouts import linearize, lqr
from scripts.diagnostics.feedback_tube import capped
from scripts.diagnostics.loop_task import WALKER
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.policy_drift import trace
from scripts.diagnostics.policy_picks import load_bank
from scripts.diagnostics.similarity_loop import SOURCES, R
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.gps.mppi_gps.policy import Policy

UPDATES = 20_000
BATCH = 256
LR = 3e-4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--single", action="store_true", help="fit and test walk 0 only")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2, choices=(2, 3))
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument(
        "--transverse", action="store_true", help="match slopes across the walk only"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK, nthread=args.threads)
    bank = load_bank(args.bank)
    pooled = np.concatenate([observation(e["states"]) for e in bank])
    normalization = dict(mean=pooled.mean(0), std=np.maximum(pooled.std(0), 1e-6))
    walks = []
    with np.load(args.run / f"round_{args.round}.npz") as z:
        for s in SOURCES[:1] if args.single else SOURCES:
            with np.load(args.bank / f"episode_{s}.npz") as d:
                start, scale = d["states"][0], d["states"][:, 1:].std(0) + 1e-6
            st, u = z[f"states_{s}"], z[f"controls_{s}"]
            xbar = np.concatenate([start[None], st[:-1]])
            K = capped(lqr(*linearize(env, xbar, u, scale), R), CAP)
            v = (
                np.diff(xbar[:, 2:], axis=0, append=xbar[-1:, 2:] + (xbar[-1, 2:] - xbar[-2, 2:]))
                / scale[1:]
            )
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            P = np.eye(17)[None] - v[:, :, None] * v[:, None, :]
            walks.append(dict(source=s, start=start, scale=scale, u=u, xbar=xbar, K=K, P=P))
    # one policy for all walks: stack rows, each with its own walk's scale
    xbar = np.concatenate([w["xbar"] for w in walks])
    u = np.concatenate([w["u"] for w in walks])
    K = np.concatenate([w["K"] for w in walks])
    scale = np.concatenate([np.broadcast_to(w["scale"], (len(w["u"]), 18)) for w in walks])
    p = Policy(0, "cpu", normalization, hidden_dim=args.hidden)
    if args.layers == 3:
        torch.manual_seed(0)
        h = args.hidden
        p.network = torch.nn.Sequential(
            torch.nn.Linear(17, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(),
            torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, 6), torch.nn.Tanh(),
        )  # fmt: skip
    P = np.concatenate([w["P"] for w in walks]) if args.transverse else None
    p = fit_rows(p, xbar, u, K, scale, args.lam, args.updates, P)
    if args.layers == 2:
        p.checkpoint(args.out / "policy.pt")
    rows = []
    for w in walks:
        on_path = p.predict(observation(w["xbar"]).astype(np.float32)).astype(np.float64)
        J = slopes(p, w["xbar"], w["scale"])
        row = dict(
            source=w["source"],
            on_path_mse=float(((on_path - w["u"]) ** 2).mean()),
            slope_agreement=float(1.0 - ((J + w["K"]) ** 2).sum() / (w["K"] ** 2).sum()),
            transverse_agreement=float(
                1.0
                - (((J + w["K"])[:, :, 1:] @ w["P"]) ** 2).sum()
                / ((w["K"][:, :, 1:] @ w["P"]) ** 2).sum()
            ),
        )
        for arm in ("walk + policy", "policy"):
            row[arm] = len(trace(env, p, w["start"], w["xbar"], w["u"], w["K"], w["scale"], arm))
        rows.append(row)
        print(row, round(time.perf_counter() - started), "s", flush=True)
    summary = dict(
        lam=args.lam,
        hidden=args.hidden,
        layers=args.layers,
        updates=args.updates,
        single=args.single,
        rows=rows,
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


def fit_rows(
    policy, xbar, u, K, scale, lam, updates=UPDATES, P=None, seed=0, spec=WALKER, lr_decay=False
):
    """Adam on |pi - u|^2 + lam |J + K|^2 over (x_bar_t, u_t, K_t, scale) rows (GPU if present).

    lr_decay: the learning rate falls linearly from LR to 0 over the updates, so the fit ends
    at the minimum it is heading for rather than anywhere in minibatch noise."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    obs = torch.tensor(spec.observation(xbar).astype(np.float32), device=device)
    target = torch.tensor(u.astype(np.float32), device=device)
    gains = torch.tensor(K[:, :, spec.observed - 1].astype(np.float32), device=device)
    P = None if P is None else torch.tensor(P.astype(np.float32), device=device)
    factor = torch.tensor(spec.slope_factor(xbar, scale).astype(np.float32), device=device)
    mean, std = policy.observation_mean, policy.observation_std

    def act(o):
        return policy.network((o - mean) / std)

    jac = vmap(jacrev(act))
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(seed)
    policy.train()
    for n in range(updates):
        if lr_decay:
            for group in optimizer.param_groups:
                group["lr"] = LR * (1.0 - n / updates)
        i = torch.randint(len(obs), (BATCH,), generator=generator).to(device)
        loss = ((act(obs[i]) - target[i]) ** 2).mean()
        if lam > 0:
            J = jac(obs[i]) * factor[i][:, None, :]
            miss = J + gains[i]
            if P is not None:
                miss = miss @ P[i]
            loss = loss + lam * (miss**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    policy.eval()
    policy.to("cpu")
    return policy


if __name__ == "__main__":
    raise SystemExit(main())
