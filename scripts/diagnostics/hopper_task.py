"""Hopper task for the MPPI loop: the same pieces walker_task provides, for src.envs.hopper.

Mirrors walker_task (controller_loop) so mppi_loop runs on it unchanged:
- spec: observation qpos[1:] and clipped qvel (11 numbers), "hops" = healthy throughout and mean
  |forward speed - 1.0| after the first 100 controls at most 0.3 (Walker's walking rule);
- teacher walks: the Hopper teacher gate's MPPI walks (hopper_teacher_probe, draws j, seeds
  9,000,000 + j): starts j 0-5 train, the walks from j 6-8 are held out for mid-hop tests;
- starting policy: Walker's network shape (3 x 256 tanh) for 11 inputs and 3 actions, fit 500
  updates on the training walks (as walker_task fits on its bank);
- judge: from 32 fresh standing draws (seeds 9,500,000 + j), 51 mid-hop states of the training
  walks and 51 of the held-out walks (controls 100-900 every 50), the policy alone for 1,000
  controls; stride regularity on the thigh (state indices 4 and 10);
- task score: the teacher walks' mean cost over the policy's own mean 8 s cost from 32 fixed
  standing starts (the 6 training starts and 26 draws, seeds 9,700,000 + j), at most 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

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


def hopper_task(args, out=None):
    from scripts.diagnostics.feedback_rollouts import step
    from scripts.diagnostics.loop_task import Task
    from scripts.diagnostics.slope_fit import fit_rows
    from scripts.diagnostics.stride_cycle import Section
    from scripts.diagnostics.trajectory_pi import step_costs
    from src.envs.hopper import HOPPER_TASK, Hopper
    from src.gps.mppi_gps import policy as policy_module

    spec = Task(
        observation=observation,
        observed=np.arange(2, 13),
        clipped=np.arange(11) >= 5,
        clip=10.0,
        admissible=hops,
    )
    env = Hopper(**HOPPER_TASK, nthread=args.threads)

    def draw(seed):
        saved = np.random.get_state()
        try:
            np.random.seed(seed)
            env.reset()
        finally:
            np.random.set_state(saved)
        return env.get_state()

    def gate_walk(j):
        with np.load(Path(GATE.format(j=j)) / "walk.npz") as d:
            return d["controls"].astype(np.float64), d["states"]

    starts = [draw(9_000_000 + j) for j in TRAIN]
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
    scale = states0[:, 1:].std(0) + 1e-6
    policy = policy_module.Policy(
        0, "cpu", dict(mean=obs0.mean(0), std=np.maximum(obs0.std(0), 1e-6)), hidden_dim=256
    )
    torch.manual_seed(getattr(args, "seed", 0))
    policy.network = torch.nn.Sequential(
        torch.nn.Linear(11, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(),
        torch.nn.Linear(256, 256), torch.nn.Tanh(), torch.nn.Linear(256, 3), torch.nn.Tanh(),
    )  # fmt: skip
    policy = fit_rows(
        policy, xbar0, actions0, np.zeros((len(xbar0), 3, 12)), np.ones((len(xbar0), 12)), 0.0,
        updates=50 if args.quick else 500, seed=getattr(args, "seed", 0), spec=spec,
    )  # fmt: skip
    stand = np.array([draw(9_500_000 + j) for j in range(32)])
    mid = np.array([s[t] for _, s in current[:3] for t in TAKEOVERS])
    held_mid = np.array([s[t] for s in held if len(s) > max(TAKEOVERS) for t in TAKEOVERS])
    section = Section([s for _, s in current], thigh=THIGH, thigh_vel=THIGH_VEL)
    accept = np.array(starts + [draw(9_700_000 + j) for j in range(32 - len(starts))])

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
            heldout_midwalk_states=int(len(held_mid)),
            stride_runs=cycle.get("runs", 0),
            stride_change=cycle.get("change"),
            stride_spread=cycle.get("spread"),
        )

    def own_cost(p, x, length, noise=0.0, rng=None):
        seen, used = [], []
        for _ in range(length):
            u = p.predict(observation(x).astype(np.float32)).astype(np.float64)
            if noise:
                u = u + noise * rng.standard_normal(u.shape)
            u = np.clip(u, -1, 1)
            x = step(env, x, u)
            seen.append(x)
            used.append(u)
        return step_costs(env, np.stack(seen, 1), np.stack(used, 1)).sum(1)

    def task_score(p, noise=0.0, draws=2):
        if noise:  # the same fixed draws for every policy, so two policies compare paired
            cost = np.mean([
                own_cost(p, accept.copy(), LENGTH, noise, np.random.default_rng(1000 + d)).mean()
                for d in range(draws)
            ])  # fmt: skip
            return float(min(1.0, np.mean(c0) / cost))
        return float(min(1.0, np.mean(c0) / own_cost(p, accept.copy(), LENGTH).mean()))

    scales = [scale] * len(starts)
    return (
        env, spec, starts, current, scales, policy, 1.5 * c0, c0, judge, None, None, None, None,
        task_score,
    )  # fmt: skip


if __name__ == "__main__":  # plumbing check: build the task and judge the starting policy
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    ns = argparse.Namespace(threads=2, quick=False, seed=0)
    out = hopper_task(ns)
    print("starts", len(out[2]), "c0", np.round(out[7], 1), "score", round(out[13](out[5]), 3))
    print(out[8](out[5]))
