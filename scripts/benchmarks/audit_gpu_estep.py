"""Small, recorded CPU/CUDA differential and operation profile for an audit."""

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from src.envs.walker2d import Walker2d
from src.gps.gpu_estep import WalkerGPUEStep
from src.gps.mdgps import closed_loop_nominal, e_step
from src.gps.policy_tail import WalkerPolicyTail
from src.mppi.mppi import MPPI
from src.rl.sac import load_policy
from src.utils.config import MPPIConfig


def main(checkpoint: str, output: str):
    root = Path(__file__).resolve().parents[2]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = load_policy(payload.get("policy", payload), device="cpu")
    actor_gpu = load_policy(payload.get("policy", payload), device="cuda")
    env = Walker2d(**json.loads((root / "configs/walker2d_task.json").read_text()), nthread=1)

    @torch.no_grad()
    def act_batch(obs):
        mean, _ = actor(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32))
        actions = (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)
        return actions if np.ndim(obs) > 1 else actions[0]

    np.random.seed(123456)
    obs = env.reset()
    states, warmstarts = [], []
    for step in range(61):
        if step in [0, 20, 60]:
            states.append(env.get_state())
            warmstarts.append(env.get_warmstart())
        obs, _, done, _ = env.step(act_batch(obs))
        if done:
            break
    states, warmstarts = np.asarray(states), np.asarray(warmstarts)
    seeds = [811000 + i for i in range(len(states))]
    cfg = replace(MPPIConfig(**json.loads((root / "configs/walker2d_best.json").read_text())),
                  K=256, H=32, initial_iterations=1, lam_target_neff_fraction=.2)
    gpu = WalkerGPUEStep(env, actor_gpu, cfg, tail_steps=100, states_per_batch=2)
    warm = gpu(states, warmstarts, seeds)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        result = gpu(states, warmstarts, seeds)
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    prof.export_chrome_trace(str(out / "gpu_profile_trace.json"))
    (out / "gpu_profile_table.txt").write_text(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=30))
    cpu = []
    tail = WalkerPolicyTail(env, act_batch, 100)
    for i, state in enumerate(states):
        nominal, _ = closed_loop_nominal(env, state,
                                         lambda s: act_batch(env.rollout_states_to_obs(s)),
                                         cfg.H, warmstarts[i])
        cpu.append(asdict(e_step(MPPI(env, cfg, seed=seeds[i]), env, state,
                                nominal, warmstarts[i], terminal=tail)))
    rows = [{name: np.asarray(value).tolist() for name, value in row.items()} for row in cpu]
    report = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "config": asdict(cfg), "tail_steps": 100, "reset_seed": 123456,
        "noise_seeds": seeds, "states": len(states), "cpu": rows,
        "gpu": {name: np.asarray(value).tolist() for name, value in asdict(result).items()},
        "gpu_repeat_label_l2": np.linalg.norm(warm.labels - result.labels, axis=1).tolist(),
        "cpu_gpu_label_l2": np.linalg.norm(result.labels - np.asarray([r["label"] for r in cpu]), axis=1).tolist(),
        "profile_seconds_under_contention": seconds,
        "peak_cuda_allocated_bytes_torch_only": torch.cuda.max_memory_allocated(),
        "contention_caveat": "Five concurrent training GPU processes were active. No isolated speed or throughput claim.",
    }
    np.savez_compressed(out / "gpu_audit_inputs.npz", states=states, warmstarts=warmstarts)
    (out / "gpu_differential_profile.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
