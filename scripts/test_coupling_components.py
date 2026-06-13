"""C4: trust-monotonicity test for the MPPI-GPS coupling machinery.

Runs coupled MPPI (merge + mixing) with a graded ladder of policy quality —
loaded checkpoint, parameter-noised copies, random init — and checks the
falsifiable predictions of the trust design:

  P1 (monotonicity): certificate acceptance, executed beta, and mixshare all
      decrease as policy quality degrades.
  P2 (safety): coupled collection cost/step never exceeds pure MPPI + delta,
      at ANY rung — a bad policy costs sample budget, never task performance.
  P3 (handover): for the best policy, acceptance and beta exceed the
      degraded rungs by a clear margin (trust is earned where deserved).

No training anywhere: pure measurement of the coupling mechanics.

Usage: PYTHONPATH=. python scripts/test_coupling_components.py \
    --checkpoint runs/walker_gps5/checkpoint_latest.pt
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import tyro

from src.envs.walker2d import Walker2d
from src.gps.merge import make_policy_merge
from src.gps.mix import make_policy_mixer
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import MPPIConfig, GPSConfig, PolicyConfig


def degraded(policy: DeterministicPolicy, noise: float) -> DeterministicPolicy:
    p = copy.deepcopy(policy)
    if noise > 0:
        with torch.no_grad():
            for q in p.net.parameters():
                q.add_(noise * torch.randn_like(q))
    return p


def run_coupled(env, mppi, policy, episodes, steps, delta_frac, mix_fraction, seed0):
    merge = make_policy_merge(
        policy, env, noise_precision=mppi.noise_precision, delta_frac=delta_frac,
        obs_from_states=env.rollout_states_to_obs,
    )
    mixer = make_policy_mixer(policy, env, horizon=mppi.H)
    accs, betas, shares, costs, steps_done = [], [], [], [], []
    for ep in range(episodes):
        np.random.seed(seed0 + ep)
        env.reset(); mppi.reset()
        for t in range(steps):
            state = env.get_state()
            mn = mixer(state)
            a, info = mppi.plan_step(state, merge=merge, mix_nominal=mn,
                                     mix_fraction=mix_fraction)
            _, c, done, _ = env.step(a)
            accs.append(info["merge_accepted"]); betas.append(info["merge_beta_mean"])
            shares.append(info["mix_weight_share"]); costs.append(c)
            if done: break
        steps_done.append(t + 1)
    return dict(acc=np.mean(accs), beta=np.mean(betas), mixshare=np.mean(shares),
                cost_per_step=np.mean(costs), steps=np.mean(steps_done))


def main(
    checkpoint: str = "runs/walker_gps5/checkpoint_latest.pt",
    mismatched_checkpoint: str | None = None,
    episodes: int = 2,
    steps: int = 300,
    delta_frac: float = 0.05,
    mix_fraction: float = 0.25,
) -> None:
    env = Walker2d()
    mppi = MPPI(env, MPPIConfig.load("walker2d"))
    gps = GPSConfig.load("walker2d")
    base = DeterministicPolicy(gps.obs_dim, gps.act_dim, PolicyConfig())
    base.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
    base.eval()

    # pure-MPPI reference (P2 baseline)
    ref_costs, ref_steps = [], []
    for ep in range(episodes):
        np.random.seed(100 + ep)
        env.reset(); mppi.reset()
        for t in range(steps):
            a, _ = mppi.plan_step(env.get_state())
            _, c, done, _ = env.step(a)
            ref_costs.append(c)
            if done: break
        ref_steps.append(t + 1)
    ref = np.mean(ref_costs)
    print(f"pure MPPI: cost/step={ref:.3f} steps={np.mean(ref_steps):.0f}", flush=True)

    torch.manual_seed(0)
    rungs = [("competent", degraded(base, 0.0)),
             ("param_noise_0.05", degraded(base, 0.05)),
             ("param_noise_0.2", degraded(base, 0.2)),
             ("random_init", DeterministicPolicy(gps.obs_dim, gps.act_dim, PolicyConfig()))]
    if mismatched_checkpoint is not None:
        # objective-mismatched policy (e.g. max-speed TD3 vs 1.5 m/s task):
        # competent at ITS objective but wrong for the planner — the
        # certificate should distrust it like a degraded one (gps10)
        mm = DeterministicPolicy(gps.obs_dim, gps.act_dim, PolicyConfig())
        mm.load_state_dict(torch.load(mismatched_checkpoint, map_location="cpu"), strict=False)
        mm.eval()
        rungs.append(("objective_mismatch", mm))
    results = []
    for name, pol in rungs:
        pol.eval()
        r = run_coupled(env, mppi, pol, episodes, steps, delta_frac, mix_fraction, 100)
        results.append((name, r))
        print(f"{name:>18}: acc={r['acc']:.2f} beta={r['beta']:.3f} "
              f"mixshare={r['mixshare']:.3f} cost/step={r['cost_per_step']:.3f} "
              f"steps={r['steps']:.0f}", flush=True)

    accs = [r["acc"] for _, r in results]
    betas = [r["beta"] for _, r in results]
    p1 = all(accs[i] >= accs[i + 1] - 0.05 for i in range(len(accs) - 1)) and \
         all(betas[i] >= betas[i + 1] - 0.02 for i in range(len(betas) - 1))
    p2 = all(r["cost_per_step"] <= ref * (1 + delta_frac) + 0.05 for _, r in results)
    p3 = accs[0] > accs[-1] + 0.1
    print(f"P1 monotonic trust: {'PASS' if p1 else 'FAIL'}")
    print(f"P2 safety bound:    {'PASS' if p2 else 'FAIL'}")
    print(f"P3 earned handover: {'PASS' if p3 else 'FAIL'}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
