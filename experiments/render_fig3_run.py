"""Render rollouts from a fig3 GPS run for visual inspection.

Loads a checkpoint and renders three side-by-side video comparisons:
  - vanilla MPPI (baseline, no prior)
  - MPPI + π prior (the deployed GPS-trained system)
  - π only (policy alone, no MPPI)

All three use the same seeded initial states so episodes are directly comparable
across modes.

Usage:
    .venv/bin/python -m experiments.render_fig3_run --run-name fig3_lambda_3
    .venv/bin/python -m experiments.render_fig3_run --run-name fig3_lambda_3 --ckpt-iter 7
"""
from __future__ import annotations

import json
from pathlib import Path

import mediapy
import mujoco
import numpy as np
import torch
import tyro

from src.envs.acrobot import Acrobot
from src.gps.prior import make_policy_tracking_prior
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig

EXPERIMENTS_RESULTS = Path(__file__).resolve().parent / "results"


def _rollout_with_render(
    env: Acrobot,
    renderer: mujoco.Renderer,
    step_fn,
    n_episodes: int,
    episode_len: int,
    seed: int,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Generic rollout-with-render: step_fn(state) -> action."""
    frames: list[np.ndarray] = []
    ep_costs: list[float] = []
    max_zs: list[float] = []
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()
        ep_cost = 0.0
        ep_max_z = -np.inf
        for _ in range(episode_len):
            state = env.get_state()
            action = step_fn(state)
            _, cost, done, _ = env.step(action)
            ep_cost += cost
            tip_z = float(env.data.sensordata[2])
            if tip_z > ep_max_z:
                ep_max_z = tip_z
            renderer.update_scene(env.data)
            frames.append(renderer.render().copy())
            if done:
                break
        ep_costs.append(ep_cost)
        max_zs.append(ep_max_z)
    return frames, ep_costs, max_zs


def main(
    run_name: str = "fig3_lambda_3",
    ckpt_iter: int | None = None,
    n_episodes: int = 3,
    episode_len: int = 1000,
    seed: int = 7,
    fps: int = 60,
    skip_vanilla: bool = False,
    skip_prior: bool = False,
    skip_policy: bool = False,
) -> None:
    """Render three rollout videos for a fig3 run.

    Args:
        run_name: subdir of experiments/results/ containing checkpoints.
        ckpt_iter: iter index to render (None = checkpoint_latest.pt).
        n_episodes: number of episodes per video.
        episode_len: max steps per episode.
        seed: base seed for env init (matched across all three modes).
        fps: video framerate.
    """
    run_dir = EXPERIMENTS_RESULTS / run_name
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    # pick checkpoint
    if ckpt_iter is None:
        ckpt_path = run_dir / "checkpoint_latest.pt"
        ckpt_tag = "latest"
    else:
        ckpt_path = run_dir / f"checkpoint_iter_{ckpt_iter:03d}.pt"
        ckpt_tag = f"iter{ckpt_iter:03d}"
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    # load configs and read the run's frozen settings
    cfg_meta = json.loads((run_dir / "config.json").read_text())
    lambda_track = float(cfg_meta["cli"]["lambda_track"])
    print(f"run_dir:    {run_dir}")
    print(f"checkpoint: {ckpt_path}")
    print(f"λ_track:    {lambda_track}")
    print(f"n_eps={n_episodes}  episode_len={episode_len}  seed={seed}  fps={fps}")

    mppi_cfg = MPPIConfig.load("acrobot")
    gps_cfg = GPSConfig.load("acrobot")
    policy_cfg = PolicyConfig()

    env = Acrobot(use_warp=False, nworld=mppi_cfg.K)
    mppi = MPPI(env, mppi_cfg)
    renderer = mujoco.Renderer(env.model, height=480, width=640)

    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, policy_cfg)
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)
    policy.eval()

    results: dict[str, dict] = {}

    # 1. vanilla MPPI (no prior)
    if not skip_vanilla:
        print(f"\n[1/3] rendering vanilla MPPI...")
        def vanilla_step(state):
            mppi.reset() if state[0] == 0 else None  # reset already happens before episode
            action, _ = mppi.plan_step(state, prior_cost=None)
            return action

        # need to reset mppi before each episode; do it inside the rollout loop
        frames_v, costs_v, zs_v = [], [], []
        for ep in range(n_episodes):
            np.random.seed(seed + ep)
            env.reset()
            mppi.reset()
            ep_cost = 0.0
            ep_max_z = -np.inf
            for _ in range(episode_len):
                state = env.get_state()
                action, _ = mppi.plan_step(state, prior_cost=None)
                _, cost, done, _ = env.step(action)
                ep_cost += cost
                tip_z = float(env.data.sensordata[2])
                ep_max_z = max(ep_max_z, tip_z)
                renderer.update_scene(env.data)
                frames_v.append(renderer.render().copy())
                if done:
                    break
            costs_v.append(ep_cost)
            zs_v.append(ep_max_z)
            print(f"   ep {ep}: cost={ep_cost:.1f}  max_tip_z={ep_max_z:.2f}  success={ep_max_z>=3.5}")

        out_v = run_dir / f"render_{ckpt_tag}_vanilla_mppi.mp4"
        mediapy.write_video(str(out_v), frames_v, fps=fps)
        results["vanilla"] = {"path": str(out_v), "costs": costs_v, "max_z": zs_v}
        print(f"   wrote {out_v}")

    # 2. MPPI + π prior
    if not skip_prior:
        print(f"\n[2/3] rendering MPPI + π prior...")
        prior = make_policy_tracking_prior(policy, lambda_track)
        frames_p, costs_p, zs_p = [], [], []
        for ep in range(n_episodes):
            np.random.seed(seed + ep)
            env.reset()
            mppi.reset()
            ep_cost = 0.0
            ep_max_z = -np.inf
            for _ in range(episode_len):
                state = env.get_state()
                action, _ = mppi.plan_step(state, prior_cost=prior)
                _, cost, done, _ = env.step(action)
                ep_cost += cost
                tip_z = float(env.data.sensordata[2])
                ep_max_z = max(ep_max_z, tip_z)
                renderer.update_scene(env.data)
                frames_p.append(renderer.render().copy())
                if done:
                    break
            costs_p.append(ep_cost)
            zs_p.append(ep_max_z)
            print(f"   ep {ep}: cost={ep_cost:.1f}  max_tip_z={ep_max_z:.2f}  success={ep_max_z>=3.5}")

        out_p = run_dir / f"render_{ckpt_tag}_mppi_with_prior.mp4"
        mediapy.write_video(str(out_p), frames_p, fps=fps)
        results["mppi_prior"] = {"path": str(out_p), "costs": costs_p, "max_z": zs_p}
        print(f"   wrote {out_p}")

    # 3. π only
    if not skip_policy:
        print(f"\n[3/3] rendering π only...")
        frames_pi, costs_pi, zs_pi = [], [], []
        for ep in range(n_episodes):
            np.random.seed(seed + ep)
            env.reset()
            ep_cost = 0.0
            ep_max_z = -np.inf
            for _ in range(episode_len):
                obs = env._get_obs()
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    mu = policy.forward(obs_t)
                action = mu.squeeze(0).cpu().numpy()
                _, cost, done, _ = env.step(action)
                ep_cost += cost
                tip_z = float(env.data.sensordata[2])
                ep_max_z = max(ep_max_z, tip_z)
                renderer.update_scene(env.data)
                frames_pi.append(renderer.render().copy())
                if done:
                    break
            costs_pi.append(ep_cost)
            zs_pi.append(ep_max_z)
            print(f"   ep {ep}: cost={ep_cost:.1f}  max_tip_z={ep_max_z:.2f}  success={ep_max_z>=3.5}")

        out_pi = run_dir / f"render_{ckpt_tag}_policy_only.mp4"
        mediapy.write_video(str(out_pi), frames_pi, fps=fps)
        results["policy"] = {"path": str(out_pi), "costs": costs_pi, "max_z": zs_pi}
        print(f"   wrote {out_pi}")

    # summary
    print("\n--- summary ---")
    for k, v in results.items():
        n_succ = sum(1 for z in v["max_z"] if z >= 3.5)
        mean_c = sum(v["costs"]) / len(v["costs"])
        print(f"{k:12s}  cost_mean={mean_c:7.1f}  success={n_succ}/{len(v['costs'])}  -> {v['path']}")

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
