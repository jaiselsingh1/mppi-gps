"""Batched GPU E-step vs CPU E-step on a training policy's states: labels and time.

States are collected the way training collects them (policy + N(0, 0.1^2)
on executed actions, every 2nd state labelled). The GPU labels all of them in
batches; the CPU labels the first `cpu_states` with identical noise draws.
Label noise is the GPU label difference between two independent noise seeds.
"""
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import tyro

from src.envs.walker2d import Walker2d
from src.gps.gpu_estep import WalkerGPUEStep
from src.gps.mdgps import closed_loop_nominal, e_step
from src.gps.policy_tail import WalkerPolicyTail
from src.mppi.mppi import MPPI
from src.rl.sac import load_policy
from src.utils.config import MPPIConfig

_ROOT = Path(__file__).resolve().parents[2]


def _collect(env, act, episodes: int, steps: int, every: int, noise: float, seed: int):
    rng = np.random.default_rng(900_000 + seed)
    low, high = env.action_bounds
    states, warmstarts = [], []
    for episode in range(episodes):
        np.random.seed(seed + episode)
        obs = env.reset()
        for t in range(steps):
            if t % every == 0:
                states.append(env.get_state())
                warmstarts.append(env.get_warmstart())
            obs, _, done, _ = env.step(np.clip(act(obs) + noise * rng.standard_normal(env.action_dim), low, high))
            if done:
                break
    return np.asarray(states), np.asarray(warmstarts)


def main(out_dir: Path, policy: Path, K: int = 256, H: int = 32, tail_steps: int = 100,
         neff_fraction: float = 0.2, episodes: int = 10, steps: int = 400, cpu_states: int = 48,
         batch_sizes: tuple[int, ...] = (64, 128)) -> None:
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
    cfg = replace(base, K=K, H=H, initial_iterations=1, lam_target_neff_fraction=neff_fraction)
    states, warmstarts = _collect(env, act_batch, episodes, steps, 2, 0.1, 123_000)
    seeds = [10_000 + i for i in range(len(states))]
    report = {"policy": str(policy), "states": int(len(states)), "K": K, "H": H, "tail_steps": tail_steps,
              "gpu": {}, "cpu": {}}

    gpu_results = {}
    for batch in batch_sizes:
        gpu = WalkerGPUEStep(env, actor_gpu, cfg, tail_steps, states_per_batch=batch)
        started = time.perf_counter()
        gpu_results[batch] = gpu(states, warmstarts, seeds)
        seconds = time.perf_counter() - started
        report["gpu"][f"batch_{batch}"] = {"worlds": batch * K, "seconds": seconds,
                                           "seconds_per_label": seconds / len(states)}
        del gpu
    reference = gpu_results[batch_sizes[0]]
    noise_run = WalkerGPUEStep(env, actor_gpu, cfg, tail_steps, states_per_batch=batch_sizes[0])(
        states, warmstarts, [50_000 + i for i in range(len(states))])

    tail = WalkerPolicyTail(env, act_batch, tail_steps)
    cpu_rows, cpu_seconds = [], []
    for i in range(min(cpu_states, len(states))):
        started = time.perf_counter()
        nominal, _ = closed_loop_nominal(env, states[i], lambda s: act_batch(env.rollout_states_to_obs(s)), H, warmstarts[i])
        cpu_rows.append(e_step(MPPI(env, cfg, seed=seeds[i]), env, states[i], nominal, warmstarts[i], terminal=tail))
        cpu_seconds.append(time.perf_counter() - started)
    n = len(cpu_rows)
    label_diff = np.linalg.norm(reference.labels[:n] - np.asarray([r.label for r in cpu_rows]), axis=1)
    noise = np.linalg.norm(reference.labels - noise_run.labels, axis=1)
    rel = lambda a, b: np.abs(a - b) / np.maximum(np.abs(b), 1e-9)
    report["cpu"] = {"states": n, "median_seconds_per_label": float(np.median(cpu_seconds))}
    report["agreement"] = {
        "label_diff_gpu_vs_cpu": {"median": float(np.median(label_diff)), "p95": float(np.quantile(label_diff, 0.95)),
                                  "max": float(np.max(label_diff))},
        "label_noise_two_seeds_gpu": {"median": float(np.median(noise)), "p95": float(np.quantile(noise, 0.95))},
        "lam_rel_diff_median": float(np.median(rel(reference.lam[:n], np.asarray([r.lam for r in cpu_rows])))),
        "nominal_cost_rel_diff_median": float(np.median(rel(reference.nominal_cost[:n], np.asarray([r.nominal_cost for r in cpu_rows])))),
        "expected_cost_rel_diff_median": float(np.median(rel(reference.expected_cost[:n], np.asarray([r.expected_cost for r in cpu_rows])))),
        "plan_cost_rel_diff_median": float(np.median(rel(reference.plan_cost[:n], np.asarray([r.plan_cost for r in cpu_rows])))),
        "gpu_batch_sizes_label_diff_max": float(np.max(np.linalg.norm(
            gpu_results[batch_sizes[0]].labels - gpu_results[batch_sizes[-1]].labels, axis=1))),
    }
    best = min(v["seconds_per_label"] for v in report["gpu"].values())
    report["speedup_per_label_vs_one_cpu_thread"] = report["cpu"]["median_seconds_per_label"] / best
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
