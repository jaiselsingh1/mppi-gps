"""Walker get-up task for the MPPI loop: from lying on the ground, get up and stand.

The first test of the loop on a behaviour with no repeating cycle (a transient into a standing
fixed point). Same pieces walker_task and hopper_task provide, so mppi_loop runs on it unchanged
(--task getup --length 500):
- env: src.envs.walker_getup.WalkerGetUp (Walker2d model; getup_probe's dense cost: torso at
  standing height, upright, still, little effort; no fall rule);
- lying states: a standing draw (np.random seed), a random constant torque (uniform in [-1, 1]
  per joint, drawn from the same seed) for 0.3 s, then zero torque until 2.5 s, so the walker
  lands in varied poses. Seeds 20,000,000 + j: j 0-5 are the training starts, 6-11 the gate's
  held-out starts; 32 judge starts 20,500,000 + j; the score's 32 starts are the 6 training
  starts and 26 draws 20,700,000 + j;
- teacher get-ups: the gate's MPPI runs (python getup_task.py --start j --out ...;
  configs/walker2d_best.json as in the walks, planner seed 21,000,000 + j), 4 s each;
- starting policy: Walker's network shape (3 x 256 tanh, 17 observations, 6 actions), fit 500
  updates on the training get-ups (as walker_task fits on its bank);
- judge: the policy alone for 4 s from the 32 judge starts and the 6 training starts: how many
  it gets up from (torso above 1.0 m and |pitch| below 0.5 rad over the final 1 s), reported as
  "success" (the loop's kill rule and readings count it), and the median time to first stand;
- task score: the teacher get-ups' mean cost over the policy's own mean 4 s cost from the score's
  32 starts, at most 1.

PARKED 2026-11-06 ~17:40 (user: a task per failure is hand-made and does not carry to other envs);
kept for --cost task, the general check: can MPPI recover from the ground with the Walker task's
own running cost when the fall rule is removed (src.envs.walker_getup.WalkerNoFall)? Reading fixed
before it ran: "MPPI recovers with the plain task cost" if it stands up (final 1 s upright, torso
above 1.0 m) from at least 4 of the 6 training lying starts at the configured horizon (H 48), or
from at least 2 of starts 0-2 at H 125.
Result (2026-11-06 17:30): by that reading's stand-still rule, 0 of 6 at H 48 and 0 of 3 at H 125, so
it is not met; but the rule was the wrong one for the plain cost, which asks to walk at 1.5 m/s
(walking tilts the torso past 0.5 rad). By Walker's own health rule (torso 0.8-2.0 m, |pitch| < 1),
read after the fact: H 125 got back up in 3 of 3 (by 0.37-0.70 s, then healthy 98-99% of the time,
walking 0.86-1.15 m/s); H 48 in 2 of 6 (by 1.2-1.8 s, then walking 1.5 m/s), the other 4 stayed on
the ground and slid forward at the target speed (pitch 1.6-1.9 or -4.6: the fallen penalty is flat,
so scooting meets the speed term). So without the fall rule the task's own cost carries enough
signal to get up, with a ~1 s horizon; the short horizon finds the scoot.

Checks fixed before the loop runs (2026-11-06 17:20):
- gate: MPPI gets up from at least 5 of the 6 training starts (else fix the teacher first);
- cloning: the starting policy (and a 2,000-update fit) on the 6 teacher get-ups gets up from
  fewer than 16 of the 32 judge starts (else the task is too easy to say anything about the
  loop; make it harder first);
- strategies: the net pitch rotation of each teacher get-up (whole turns), to see whether
  MPPI's get-ups share one way of getting up or several.
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

LENGTH, SETTLE, PUSH = 500, 312, 38
LYING_SEED, JUDGE_SEED, ACCEPT_SEED, PLAN_SEED = 20_000_000, 20_500_000, 20_700_000, 21_000_000
TRAIN, HELD = (0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)
GATE = "runs/2026-11-06_getup_teacher_start{j}_001"


def lying(env, seed):
    """A standing draw, a random constant torque for PUSH controls, then limp until SETTLE."""
    from scripts.diagnostics.feedback_rollouts import step

    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        env.reset()
    finally:
        np.random.set_state(saved)
    x = env.get_state()
    push = np.random.default_rng(seed).uniform(-1.0, 1.0, env.model.nu)
    for t in range(SETTLE):
        u = push if t < PUSH else np.zeros(env.model.nu)
        x = step(env, x[None], u[None])[0]
    return x


def getup_task(args, out=None):
    import dataclasses

    from scripts.diagnostics.feedback_rollouts import step
    from scripts.diagnostics.loop_task import WALKER
    from scripts.diagnostics.slope_fit import fit_rows
    from scripts.diagnostics.trajectory_pi import step_costs
    from src.envs.walker_getup import WalkerGetUp, stood_up
    from src.gps.mppi_gps import policy as policy_module
    from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

    spec = dataclasses.replace(WALKER, admissible=stood_up)
    env = WalkerGetUp(**NATIVE_TASK, nthread=args.threads)
    starts = [lying(env, LYING_SEED + j) for j in TRAIN]

    def gate_walk(j):
        with np.load(Path(GATE.format(j=j)) / "walk.npz") as d:
            return d["controls"].astype(np.float64), d["states"]

    current = [gate_walk(j) for j in TRAIN]
    for j, (u, s) in zip(TRAIN, current):
        if len(s) < LENGTH:
            raise SystemExit(f"teacher get-up {j} is shorter than {LENGTH} controls")
    c0 = np.array([float(step_costs(env, s[None], u[None]).sum()) for u, s in current])
    states0 = np.concatenate([s for _, s in current])
    xbar0 = np.concatenate([np.concatenate([x[None], s[:-1]]) for x, (_, s) in zip(starts, current)])
    actions0 = np.concatenate([u for u, _ in current])
    obs0 = spec.observation(states0)
    scale = states0[:, 1:].std(0) + 1e-6
    policy = policy_module.Policy(
        0, "cpu", dict(mean=obs0.mean(0), std=np.maximum(obs0.std(0), 1e-6)), hidden_dim=256
    )
    torch.manual_seed(getattr(args, "seed", 0))
    policy.network = torch.nn.Sequential(
        torch.nn.Linear(17, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(),
        torch.nn.Linear(256, 256), torch.nn.Tanh(), torch.nn.Linear(256, 6), torch.nn.Tanh(),
    )  # fmt: skip
    policy = fit_rows(
        policy, xbar0, actions0, np.zeros((len(xbar0), 6, 18)), np.ones((len(xbar0), 18)), 0.0,
        updates=50 if args.quick else getattr(args, "start_updates", 500),
        seed=getattr(args, "seed", 0), spec=spec,
    )  # fmt: skip
    judge_starts = np.array([lying(env, JUDGE_SEED + j) for j in range(32)])
    accept = np.array(starts + [lying(env, ACCEPT_SEED + j) for j in range(32 - len(starts))])
    train_starts = np.array(starts)

    def travel(p, x):
        seen = []
        for _ in range(LENGTH):
            u = np.clip(p.predict(spec.observation(x).astype(np.float32)).astype(np.float64), -1, 1)
            x = step(env, x, u)
            seen.append(x)
        seen = np.stack(seen, 1)
        up = (seen[..., 2] > 1.0) & (np.abs(seen[..., 3]) < 0.5)
        first = np.where(up.any(1), np.argmax(up, 1) * 0.008, np.nan)
        return stood_up(seen), first

    def judge(p):
        if args.quick:
            return {}
        ok, first = travel(p, judge_starts.copy())
        ok_train, _ = travel(p, train_starts.copy())
        return dict(
            success=int(ok.sum()),
            success_of=int(len(ok)),
            first_stand_median_s=float(np.nanmedian(first)) if np.isfinite(first).any() else None,
            train_success=int(ok_train.sum()),
        )

    def own_cost(p, x, length, noise=0.0, rng=None):
        seen, used = [], []
        for _ in range(length):
            u = p.predict(spec.observation(x).astype(np.float32)).astype(np.float64)
            if noise:
                u = u + noise * rng.standard_normal(u.shape)
            u = np.clip(u, -1, 1)
            x = step(env, x, u)
            seen.append(x)
            used.append(u)
        return step_costs(env, np.stack(seen, 1), np.stack(used, 1)).sum(1)

    def task_score(p, noise=0.0, draws=2):
        if noise:
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


def gate(args):
    """One teacher get-up: MPPI (as in the walks) from lying start j for LENGTH controls."""
    from scripts.diagnostics.coupled_mppi import CONFIG
    from scripts.diagnostics.feedback_rollouts import step
    from scripts.diagnostics.trajectory_pi import step_costs
    from src.envs.walker_getup import WalkerGetUp, stood_up
    from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    from src.envs.walker_getup import WalkerNoFall

    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    env = (WalkerNoFall if args.cost == "task" else WalkerGetUp)(**NATIVE_TASK, nthread=1)
    x0 = lying(env, LYING_SEED + args.start)
    cfg = json.loads(CONFIG.read_text())
    if args.horizon:
        cfg["H"] = args.horizon
    planner = MPPI(env, MPPIConfig(**cfg))
    planner.reset(seed=PLAN_SEED + args.start)
    x, xs, us = x0.copy(), [], []
    for _ in range(LENGTH):
        env.reset(state=x)
        u, _ = planner.plan_step(x, initial_warmstart=env.get_warmstart())
        u = np.asarray(u, dtype=float)
        x = step(env, x[None], u[None])[0]
        xs.append(x.copy())
        us.append(u)
    states, controls = np.array(xs), np.array(us)
    up = (states[:, 2] > 1.0) & (np.abs(states[:, 3]) < 0.5)
    result = dict(
        start=args.start,
        lying_z=float(x0[2]),
        lying_pitch=float(x0[3]),
        stood_up=bool(stood_up(states)),
        first_stand_s=float(np.argmax(up) * 0.008) if up.any() else None,
        turns=float((states[-1, 3] - x0[3]) / (2 * np.pi)),
        min_pitch=float(states[:, 3].min()),
        max_pitch=float(states[:, 3].max()),
        final_z=float(states[-1, 2]),
        cost=float(step_costs(env, states[None], controls[None]).sum()),
        cost_kind=args.cost,
        horizon=cfg["H"],
        max_z=float(states[:, 2].max()),
        seconds=time.perf_counter() - started,
    )
    np.savez_compressed(args.out / "walk.npz", states=states, controls=controls)
    (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))
    return 0


def check(args):
    """The cloning check: the starting policy (and a longer fit) on the teacher get-ups."""
    for updates in (500, 2000):
        ns = argparse.Namespace(threads=args.threads, quick=False, seed=args.seed, start_updates=updates)
        out = getup_task(ns)
        policy, judge, score, c0 = out[5], out[8], out[13], out[7]
        print(json.dumps(dict(updates=updates, c0=np.round(c0, 1).tolist(), score=score, **judge(policy))),
              flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, help="gate: lying start j")
    parser.add_argument("--out", type=Path, help="gate: output folder")
    parser.add_argument("--check", action="store_true", help="the cloning check")
    parser.add_argument("--cost", choices=("getup", "task"), default="getup",
                        help="gate: the hand-made get-up cost, or the Walker task's own cost without the fall rule")
    parser.add_argument("--horizon", type=int, default=None, help="gate: MPPI horizon override")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    a = parser.parse_args()
    raise SystemExit(check(a) if a.check else gate(a))
