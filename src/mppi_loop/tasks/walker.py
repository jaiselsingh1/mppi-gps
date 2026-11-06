"""Walker2d: walk at 1.5 m/s from standing.

Starting data (from --bank, runs/2026-11-03_sample_bank_001, made by an earlier SAC teacher):
the 6 training starts are the first states of bank episodes 0-4 and 6; the starting walks are
those episodes re-executed with LQR feedback (they set the cost normalizer c0); the starting
policy is fit 500 updates (actions only) on the bank's walking episodes. Mid-walk tests take
over at controls 100-900 (every 50) of bank episodes 0-2, held-out mid-walk tests of episodes
7-9 (never trained on).

Judge: the policy alone for 1,000 controls (8 s) from 32 standing draws: how many walk (stay up
and, after the first 100 controls, mean |forward speed - 1.5| <= 0.3) and the median time up;
the same from the 51 mid-walk and 51 held-out states; stride regularity at the thigh section.
Task score: c0's mean over the policy's own mean 8 s cost from 32 fixed standing starts (the
6 training starts and 26 draws, seeds 700,000 + j), at most 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.envs.walker2d import Walker2d
from src.mppi_loop.dynamics import R, closed_loop, linearize, lqr, step, step_costs
from src.mppi_loop.fit import fit_rows
from src.mppi_loop.measures import Section
from src.mppi_loop.policy import Policy, tanh_network
from src.mppi_loop.spec import WALK_MAE, WALKER
from src.mppi_loop.tasks import TaskSetup

WALKER_TASK = {  # the Walker2d task (cost weights) every run uses
    "frame_skip": 4,
    "cost_style": "target_velocity",
    "target_velocity": 1.5,
    "vel_cost_weight": 2.0,
    "vertical_velocity_cost_weight": 0.5,
    "angle_cost_weight": 0.1,
    "unhealthy_cost_weight": 5.0,
    "ctrl_cost_weight": 0.001,
    "terminal_unhealthy_weight": 20.0,
    "apply_terminal_cost_on_done": True,
}
SOURCES = (0, 1, 2, 3, 4, 6)  # bank episodes whose first states are the training starts
TAKEOVERS = tuple(range(100, 901, 50))
EVAL_SEEDS = tuple(range(1_250_000_000, 1_250_000_016)) + tuple(range(1_260_000_000, 1_260_000_016))


def load_bank(folder: Path, every: bool = False) -> list[dict]:
    walks = []
    for path in sorted(folder.glob("episode_*.npz")):
        with np.load(path) as data:
            e = {k: data[k] for k in data.files}
        if every or bool(e["walks"]):
            walks.append(e)
    return walks


def draw(env, seed):
    """A standing start: env.reset() under a fixed numpy seed (the global RNG is restored)."""
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        env.reset()
    finally:
        np.random.set_state(saved)
    return env.get_state()


def setup(args) -> TaskSetup:
    observation = WALKER.observation
    env = Walker2d(**WALKER_TASK, nthread=args.threads)
    length = 120 if args.quick else 1000
    sources = SOURCES[:2] if args.quick else SOURCES
    starts, current, c0 = [], [], []
    for s in sources:
        with np.load(args.bank / f"episode_{s}.npz") as d:
            S, U = d["states"][:length], d["actions"][:length].astype(np.float64)
            scale = d["states"][:, 1:].std(0) + 1e-6
        K = lqr(*linearize(env, S, U, scale), R)
        st, ctl = closed_loop(env, S[0], S, U, K, scale, U[None])
        starts.append(S[0])
        current.append((ctl[0], st[0]))
        c0.append(float(step_costs(env, st, ctl).sum()))
    c0 = np.array(c0)
    bank = load_bank(args.bank)
    states0 = np.concatenate([e["states"] for e in bank])
    obs0 = observation(states0)
    torch.manual_seed(getattr(args, "seed", 0))
    policy = Policy(dict(mean=obs0.mean(0), std=np.maximum(obs0.std(0), 1e-6)), tanh_network(17, 6))
    policy = fit_rows(
        policy,
        states0,
        np.concatenate([e["actions"] for e in bank]),
        np.zeros((len(states0), 6, 18)),
        np.ones((len(states0), 18)),
        0.0,
        updates=50 if args.quick else 500,
        seed=getattr(args, "seed", 0),
    )
    stand = np.array([draw(env, seed) for seed in EVAL_SEEDS])
    mid = np.array([bank[w]["states"][t] for w in range(3) for t in TAKEOVERS])
    held_mid = []
    for e in (7, 8, 9):  # walks the loop never trains on (SOURCES are 0-4 and 6; 5 fell)
        with np.load(args.bank / f"episode_{e}.npz") as d:
            held_mid += [d["states"][t] for t in TAKEOVERS]
    held_mid = np.array(held_mid)
    section = Section([e["states"] for e in bank])

    def travel(p, x):
        """The policy alone for 1,000 controls: states before each control, controls survived,
        whether it walks (survives and, after the first 100 controls, mean |forward speed - 1.5|
        <= 0.3), forward speeds."""
        alive, lasted, speed, seen = np.ones(len(x), bool), np.full(len(x), 1000), [], []
        for t in range(1000):
            seen.append(x)
            u = p.predict(observation(x).astype(np.float32)).astype(np.float64)
            x = step(env, x, np.clip(u, -1.0, 1.0))
            fell = alive & ~env._is_healthy(x[:, 2], x[:, 3])
            lasted[fell], alive = t + 1, alive & ~fell
            speed.append(x[:, 10])
        speed = np.array(speed)
        error = np.abs(speed[100:] - 1.5).mean(0)
        return np.stack(seen, 1), lasted, (lasted == 1000) & (error <= WALK_MAE), speed

    def judge(p):
        if args.quick:
            return {}
        seen, a, wa, _ = travel(p, stand.copy())
        _, b, wb, _ = travel(p, mid.copy())
        _, c, wc, _ = travel(p, held_mid.copy())
        cycle = section.regularity([seen[i] for i in np.flatnonzero(a == 1000)]) or {}
        return dict(
            standing_median_s=float(np.median(a)) * 0.008,
            standing_walks=int(wa.sum()),
            midwalk_median_s=float(np.median(b)) * 0.008,
            midwalk_walks=int(wb.sum()),
            heldout_midwalk_median_s=float(np.median(c)) * 0.008,
            heldout_midwalk_walks=int(wc.sum()),
            stride_runs=cycle.get("runs", 0),
            stride_change=cycle.get("change"),
            stride_spread=cycle.get("spread"),
            stride_period_sd_s=cycle.get("period_sd_s"),
        )

    def own_cost(p, x, length):
        """Task cost of the policy alone from states x for `length` controls (falling keeps costing)."""
        seen, used = [], []
        for _ in range(length):
            u = np.clip(p.predict(observation(x).astype(np.float32)).astype(np.float64), -1, 1)
            x = step(env, x, u)
            seen.append(x)
            used.append(u)
        return step_costs(env, np.stack(seen, 1), np.stack(used, 1)).sum(1)

    accept_starts = np.array(starts + [draw(env, 700_000 + j) for j in range(32 - len(starts))])

    def task_score(p):
        return float(min(1.0, np.mean(c0) / own_cost(p, accept_starts.copy(), 1000).mean()))

    return TaskSetup(env, WALKER, starts, current, policy, c0, judge, task_score, legs={})
