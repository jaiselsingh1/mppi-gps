"""Hopper (Gymnasium's Hopper-v5 model): hop forward at 1.0 m/s from standing.

Not yet usable: the teacher gate (MPPI with Walker's settings hopping from 10 standing draws,
scripts in sac-continuation's history) failed 0 of 10, so the starting walks below do not exist
yet. Kept as the template for a second task.

- observation: qpos[1:] and clipped qvel (11 numbers); "hops": healthy throughout and mean
  |forward speed - 1.0| after the first 100 controls at most 0.3;
- training starts: standing draws (seeds 9,000,000 + j, j 0-5), the gate's MPPI walks from them
  as the starting walks (runs/2026-11-06_hopper_teacher_start{j}_001), j 6-8 held out;
- starting policy: 3 x 256 tanh, fit 500 updates on the training walks;
- judge: 32 standing draws (9,500,000 + j), 51 mid-hop states of the training walks and 51 of
  the held-out walks, 1,000 controls each; task score: c0's mean over the policy's own mean
  cost from the 6 training starts and 26 draws (9,700,000 + j), at most 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.envs.hopper import HOPPER_TASK, Hopper
from src.mppi_loop.dynamics import step, step_costs
from src.mppi_loop.fit import fit_rows
from src.mppi_loop.measures import Section
from src.mppi_loop.policy import Policy, tanh_network
from src.mppi_loop.spec import Task
from src.mppi_loop.tasks import TaskSetup
from src.mppi_loop.tasks.walker import draw

SPEED, MAE, LENGTH = 1.0, 0.3, 1000
TRAIN, HELD = (0, 1, 2, 3, 4, 5), (6, 7, 8)
TAKEOVERS = tuple(range(100, 901, 50))
THIGH, THIGH_VEL = 4, 10
GATE = "runs/2026-11-06_hopper_teacher_start{j}_001"


def observation(states):
    return np.concatenate([states[:, 2:7], np.clip(states[:, 7:13], -10.0, 10.0)], axis=1)


def hops(states):
    healthy = (states[..., 2] > 0.7) & (np.abs(states[..., 3]) < 0.2)
    late = states[..., 100:, 7] if states.shape[-2] > 100 else states[..., 7]
    return healthy.all(-1) & (np.mean(np.abs(late - SPEED), -1) <= MAE)


def setup(args) -> TaskSetup:
    spec = Task(
        observation=observation, observed=np.arange(2, 13), clipped=np.arange(11) >= 5, clip=10.0,
        admissible=hops,
    )  # fmt: skip
    env = Hopper(**HOPPER_TASK, nthread=args.threads)

    def gate_walk(j):
        with np.load(Path(GATE.format(j=j)) / "walk.npz") as d:
            return d["controls"].astype(np.float64), d["states"]

    starts = [draw(env, 9_000_000 + j) for j in TRAIN]
    current = [gate_walk(j) for j in TRAIN]
    if any(len(s) < LENGTH or not hops(s[None])[0] for _, s in current):
        raise SystemExit("a training teacher walk does not hop: the Hopper teacher gate failed")
    held = [gate_walk(j)[1] for j in HELD]
    c0 = np.array([float(step_costs(env, s[None], u[None]).sum()) for u, s in current])
    states0 = np.concatenate([s for _, s in current])
    xbar0 = np.concatenate(
        [np.concatenate([x[None], s[:-1]]) for x, (_, s) in zip(starts, current)]
    )
    actions0 = np.concatenate([u for u, _ in current])
    obs0 = observation(states0)
    torch.manual_seed(getattr(args, "seed", 0))
    policy = Policy(dict(mean=obs0.mean(0), std=np.maximum(obs0.std(0), 1e-6)), tanh_network(11, 3))
    policy = fit_rows(
        policy, xbar0, actions0, np.zeros((len(xbar0), 3, 12)), np.ones((len(xbar0), 12)), 0.0,
        updates=50 if args.quick else 500, seed=getattr(args, "seed", 0), spec=spec,
    )  # fmt: skip
    stand = np.array([draw(env, 9_500_000 + j) for j in range(32)])
    mid = np.array([s[t] for _, s in current[:3] for t in TAKEOVERS])
    held_mid = np.array([s[t] for s in held if len(s) > max(TAKEOVERS) for t in TAKEOVERS])
    section = Section([s for _, s in current], thigh=THIGH, thigh_vel=THIGH_VEL)
    accept = np.array(starts + [draw(env, 9_700_000 + j) for j in range(32 - len(starts))])

    def travel(p, x):
        alive, lasted, speed, seen = np.ones(len(x), bool), np.full(len(x), LENGTH), [], []
        for t in range(LENGTH):
            seen.append(x)
            u = p.predict(observation(x).astype(np.float32)).astype(np.float64)
            x = step(env, x, np.clip(u, -1.0, 1.0))
            fell = alive & ~env._is_healthy(x[:, 2], x[:, 3])
            lasted[fell], alive = t + 1, alive & ~fell
            speed.append(x[:, 7])
        error = np.abs(np.array(speed)[100:] - SPEED).mean(0)
        return np.stack(seen, 1), lasted, (lasted == LENGTH) & (error <= MAE)

    def judge(p):
        if args.quick:
            return {}
        seen, a, wa = travel(p, stand.copy())
        _, b, wb = travel(p, mid.copy())
        _, c, wc = travel(p, held_mid.copy()) if len(held_mid) else (None, np.zeros(1), np.zeros(1))
        cycle = section.regularity([seen[i] for i in np.flatnonzero(a == LENGTH)]) or {}
        return dict(
            standing_median_s=float(np.median(a)) * 0.008,
            standing_walks=int(wa.sum()),
            midwalk_median_s=float(np.median(b)) * 0.008,
            midwalk_walks=int(wb.sum()),
            heldout_midwalk_median_s=float(np.median(c)) * 0.008,
            heldout_midwalk_walks=int(np.sum(wc)),
            stride_runs=cycle.get("runs", 0),
            stride_change=cycle.get("change"),
            stride_spread=cycle.get("spread"),
        )

    def own_cost(p, x, length):
        seen, used = [], []
        for _ in range(length):
            u = np.clip(p.predict(observation(x).astype(np.float32)).astype(np.float64), -1, 1)
            x = step(env, x, u)
            seen.append(x)
            used.append(u)
        return step_costs(env, np.stack(seen, 1), np.stack(used, 1)).sum(1)

    def task_score(p):
        return float(min(1.0, np.mean(c0) / own_cost(p, accept.copy(), LENGTH).mean()))

    return TaskSetup(
        env, spec, starts, current, policy, c0, judge, task_score,
        legs=dict(thigh=THIGH, thigh_vel=THIGH_VEL),
    )  # fmt: skip
