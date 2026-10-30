"""MuJoCo Warp vs CPU parity and speed for Walker E-step rollouts (audit item A1).

Parity, at states from a SAC 800k episode, with 256 policy-centred samples
of 32 controls (the MPPI-GPS E-step):
  - state divergence and cost agreement, Warp (float32, no warm start) vs
    CPU float64 with and without the solver warm start;
  - whether two identical Warp calls agree (state carried between calls);
  - the decision that matters: the E-step label (n_eff/K = 0.2 weighted first
    action) from Warp costs vs CPU costs, against the label noise between two
    independent CPU sample draws.
Speed: open-loop batched rollouts (CPU single thread vs Warp at several
world counts) and a closed-loop policy tail on the GPU.
"""
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import tyro
from scipy.stats import spearmanr

from src.envs.walker2d import Walker2d
from src.gps.mdgps import closed_loop_nominal
from src.mppi.mppi import MPPI
from src.rl.sac import load_policy
from src.utils.config import MPPIConfig
from src.utils.math import effective_sample_size, softmin_weights, temperature_for_neff_fraction

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAC = _REPO_ROOT / "runs/2026-10-11_s2_sac_ladder_seed0_001/checkpoints/step_0800000.pt"


def _task() -> dict:
    return json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())


def _label(costs: np.ndarray, actions: np.ndarray, rho: float = 0.2) -> tuple[np.ndarray, float]:
    weights = softmin_weights(costs, temperature_for_neff_fraction(costs, rho))
    return weights @ actions[:, 0, :], effective_sample_size(weights)


def _sac_states(env: Walker2d, actor, count: int, reset_seed: int) -> list:
    np.random.seed(reset_seed)
    obs = env.reset()
    keep = set(np.linspace(0, 999, count).astype(int))
    states = []
    for t in range(1000):
        if t in keep:
            states.append((env.get_state(), env.get_warmstart()))
        obs, _, done, _ = env.step(actor.act(obs))
        if done:
            break
    return states


def _timed(fn, repeats: int) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def parity(states: list, actor, samples: int, horizon: int) -> dict:
    env_cpu = Walker2d(**_task(), nthread=1)
    env_gpu = Walker2d(**_task(), use_warp=True, nworld=samples)
    base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
    cfg = replace(base, K=samples, H=horizon)
    act = lambda s: actor.act(env_cpu.rollout_states_to_obs(s))
    rows = []
    for index, (state, warmstart) in enumerate(states):
        nominal, _ = closed_loop_nominal(env_cpu, state, act, horizon, warmstart)
        low, high = env_cpu.action_bounds
        draws = [np.clip(nominal[None] + MPPI(env_cpu, cfg, seed=7_000 + 2 * index + d)._sample_noise(), low, high)
                 for d in (0, 1)]
        u = draws[0]
        s_warm, c_warm, _ = env_cpu.batch_rollout(state, u, initial_warmstart=warmstart)
        s_cold, c_cold, _ = env_cpu.batch_rollout(state, u)
        _, c_other, _ = env_cpu.batch_rollout(state, draws[1], initial_warmstart=warmstart)
        s_gpu, c_gpu, _ = env_gpu.batch_rollout(state, u)
        s_gpu2, c_gpu2, _ = env_gpu.batch_rollout(state, u)
        qv = slice(1, 1 + env_cpu.model.nq + env_cpu.model.nv)
        fell = lambda s: ~np.all(env_cpu._is_healthy(s[..., 2], s[..., 3]), axis=1)
        label_warm, neff_warm = _label(c_warm, u)
        label_cold, _ = _label(c_cold, u)
        label_gpu, neff_gpu = _label(c_gpu, u)
        label_other, _ = _label(c_other, draws[1])
        rows.append({
            "state": index,
            "max_state_diff_gpu_vs_cpu_cold_h1": float(np.max(np.abs(s_gpu[:, 0, qv] - s_cold[:, 0, qv]))),
            "median_state_diff_gpu_vs_cpu_cold_hlast": float(np.median(np.abs(s_gpu[:, -1, qv] - s_cold[:, -1, qv]))),
            "median_state_diff_cpu_cold_vs_warm_hlast": float(np.median(np.abs(s_cold[:, -1, qv] - s_warm[:, -1, qv]))),
            "cost_rel_diff_gpu_vs_cpu_warm": float(np.mean(np.abs(c_gpu - c_warm)) / np.mean(np.abs(c_warm))),
            "cost_rel_diff_cpu_cold_vs_warm": float(np.mean(np.abs(c_cold - c_warm)) / np.mean(np.abs(c_warm))),
            "spearman_gpu_vs_cpu_warm": float(spearmanr(c_gpu, c_warm).statistic),
            "fall_agreement_gpu_vs_cpu_warm": float(np.mean(fell(s_gpu) == fell(s_warm))),
            "repeat_gpu_call_identical": bool(np.array_equal(c_gpu, c_gpu2) and np.array_equal(s_gpu, s_gpu2)),
            "repeat_gpu_call_max_cost_diff": float(np.max(np.abs(c_gpu - c_gpu2))),
            "label_diff_gpu_vs_cpu_warm": float(np.linalg.norm(label_gpu - label_warm)),
            "label_diff_cpu_cold_vs_warm": float(np.linalg.norm(label_cold - label_warm)),
            "label_noise_two_cpu_draws": float(np.linalg.norm(label_other - label_warm)),
            "neff_cpu_gpu": [neff_warm, neff_gpu],
        })
    keys = [k for k in rows[0] if k not in {"state", "neff_cpu_gpu", "repeat_gpu_call_identical"}]
    summary = {k: float(np.median([r[k] for r in rows])) for k in keys}
    summary["repeat_gpu_calls_identical"] = int(sum(r["repeat_gpu_call_identical"] for r in rows))
    return {"states": len(rows), "median": summary, "rows": rows}


def speed(horizons: tuple[int, ...], worlds: tuple[int, ...], samples: int, repeats: int) -> dict:
    rng = np.random.default_rng(0)
    env_cpu = Walker2d(**_task(), nthread=1)
    np.random.seed(0)
    env_cpu.reset()
    state = env_cpu.get_state()
    out = {"cpu_single_thread": {}, "gpu": {}}
    for h in horizons:
        u = rng.uniform(-1, 1, size=(samples, h, env_cpu.action_dim))
        seconds = _timed(lambda: env_cpu.batch_rollout(state, u), repeats)
        out["cpu_single_thread"][f"K{samples}_H{h}"] = {
            "seconds": seconds, "physics_steps_per_s": samples * h * env_cpu._frame_skip / seconds}
    for n in worlds:
        env_gpu = Walker2d(**_task(), use_warp=True, nworld=n)
        for h in horizons:
            u = rng.uniform(-1, 1, size=(n, h, env_cpu.action_dim))
            seconds = _timed(lambda: env_gpu.batch_rollout(state, u), repeats)
            out["gpu"][f"N{n}_H{h}"] = {
                "seconds": seconds, "physics_steps_per_s": n * h * env_cpu._frame_skip / seconds}
        del env_gpu
    return out


def gpu_policy_tail(worlds: tuple[int, ...], tail_steps: int, repeats: int) -> dict:
    """Closed loop on the GPU: per control step, policy forward then 4 physics steps."""
    import mujoco_warp as mjw
    import warp as wp

    actor = load_policy(torch.load(_SAC, map_location="cpu", weights_only=False)["policy"], device="cuda")
    env = Walker2d(**_task(), nthread=1)
    np.random.seed(0)
    env.reset()
    nq = env.model.nq
    out = {}
    for n in worlds:
        model = mjw.put_model(env.model)
        data = mjw.make_data(env.model, nworld=n)
        qpos, qvel, ctrl = wp.to_torch(data.qpos), wp.to_torch(data.qvel), wp.to_torch(data.ctrl)
        q0 = torch.as_tensor(env.data.qpos, dtype=torch.float32, device="cuda")
        v0 = torch.as_tensor(env.data.qvel, dtype=torch.float32, device="cuda")

        @torch.no_grad()
        def run() -> None:
            qpos.copy_(q0.expand(n, -1))
            qvel.copy_(v0.expand(n, -1))
            for _ in range(tail_steps):
                obs = torch.cat([qpos[:, 1:], qvel.clamp(-10.0, 10.0)], dim=1)
                mean, _ = actor(obs)
                ctrl.copy_(torch.tanh(mean) * actor.action_scale + actor.action_bias)
                for _ in range(env._frame_skip):
                    mjw.step(model, data)
            wp.synchronize()
            torch.cuda.synchronize()

        seconds = _timed(run, repeats)
        out[f"N{n}_T{tail_steps}"] = {"seconds": seconds,
                                      "physics_steps_per_s": n * tail_steps * env._frame_skip / seconds}
        assert nq == qpos.shape[1]
    return out


def main(out_dir: Path, states: int = 12, samples: int = 256, horizon: int = 32,
         worlds: tuple[int, ...] = (256, 1024, 4096, 16384), repeats: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    actor = load_policy(torch.load(_SAC, map_location="cpu", weights_only=False)["policy"], device="cpu")
    test_states = _sac_states(Walker2d(**_task(), nthread=1), actor, states, 80_000)
    report = {
        "parity": parity(test_states, actor, samples, horizon),
        "speed_open_loop": speed((horizon, horizon + 100), worlds, samples, repeats),
        "speed_gpu_closed_loop_tail": gpu_policy_tail((256, 4096, 16384), 100, repeats),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"parity_median": report["parity"]["median"],
                      "speed_open_loop": report["speed_open_loop"],
                      "speed_gpu_closed_loop_tail": report["speed_gpu_closed_loop_tail"]}, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
