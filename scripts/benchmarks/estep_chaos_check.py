"""How chaotic is the E-step label with a closed-loop policy tail?

On the same training-policy states as gpu_estep_check.py, with identical
noise draws, measures label differences from:
  - a 1e-6 nudge to one joint velocity, on CPU (the physics' own sensitivity);
  - two identical GPU runs (GPU nondeterminism);
  - GPU vs CPU.
If all three are similar, GPU-vs-CPU disagreement is the label's physical
sensitivity, not a GPU error.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import tyro

from scripts.benchmarks.gpu_estep_check import _collect
from src.envs.walker2d import Walker2d
from src.gps.gpu_estep import WalkerGPUEStep
from src.gps.mdgps import closed_loop_nominal, e_step
from src.gps.policy_tail import WalkerPolicyTail
from src.mppi.mppi import MPPI
from src.rl.sac import load_policy
from src.utils.config import MPPIConfig

_ROOT = Path(__file__).resolve().parents[2]


def main(out_dir: Path, policy: Path, states: int = 24, tail_steps: int = 100, nudge: float = 1e-6) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = Walker2d(**json.loads((_ROOT / "configs/walker2d_task.json").read_text()), nthread=1)
    payload = torch.load(policy, map_location="cpu", weights_only=False)
    actor = load_policy(payload.get("policy", payload), device="cpu")
    actor_gpu = load_policy(payload.get("policy", payload), device="cuda")

    @torch.no_grad()
    def act_batch(obs):
        mean, _ = actor(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32))
        out = (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)
        return out if np.ndim(obs) > 1 else out[0]

    base = MPPIConfig(**json.loads((_ROOT / "configs/walker2d_best.json").read_text()))
    cfg = replace(base, K=256, H=32, initial_iterations=1, lam_target_neff_fraction=0.2)
    xs, ws = _collect(env, act_batch, 10, 400, 2, 0.1, 123_000)
    xs, ws = xs[:states], ws[:states]
    nudged = xs.copy()
    nudged[:, 1 + env.model.nq] += nudge  # root x-velocity
    seeds = [10_000 + i for i in range(states)]

    tail = WalkerPolicyTail(env, act_batch, tail_steps)
    act = lambda s: act_batch(env.rollout_states_to_obs(s))

    def cpu_labels(batch):
        out = []
        for i, (x, w) in enumerate(zip(batch, ws)):
            nominal, _ = closed_loop_nominal(env, x, act, cfg.H, w)
            out.append(e_step(MPPI(env, cfg, seed=seeds[i]), env, x, nominal, w, terminal=tail).label)
        return np.asarray(out)

    cpu, cpu_nudged = cpu_labels(xs), cpu_labels(nudged)
    gpu_step = WalkerGPUEStep(env, actor_gpu, cfg, tail_steps, states_per_batch=64)
    gpu_a, gpu_b = gpu_step(xs, ws, seeds).labels, gpu_step(xs, ws, seeds).labels
    seed_noise = gpu_step(xs, ws, [50_000 + i for i in range(states)]).labels

    diff = lambda a, b: np.linalg.norm(a - b, axis=1)
    summary = lambda d: {"median": float(np.median(d)), "max": float(np.max(d))}
    report = {
        "states": states, "tail_steps": tail_steps, "nudge": nudge,
        "cpu_nudged_vs_cpu": summary(diff(cpu_nudged, cpu)),
        "gpu_repeat": summary(diff(gpu_a, gpu_b)),
        "gpu_vs_cpu": summary(diff(gpu_a, cpu)),
        "sampling_noise_two_seeds": summary(diff(seed_noise, gpu_a)),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
