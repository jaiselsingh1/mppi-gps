"""Render policy vs MPPI walker gaits: side-by-side video + trace plots.

Mirrors Mordatch'15 Fig. 3: joint trajectories of the MPC vs the distilled
policy, plus actions and forward velocity. Both controllers run from the
same reset seed.

Usage:
  MUJOCO_GL=osmesa python scripts/visualisation/render_walker_compare.py \
      --checkpoint runs/walker_gps1/checkpoint_latest.pt --seed 0
"""
from __future__ import annotations

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import tyro

from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import MPPIConfig, GPSConfig, PolicyConfig

_JOINTS = ["thigh", "leg", "foot", "thigh_l", "leg_l", "foot_l"]


def run_and_record(env, act_fn, episode_len, seed, renderer, render_every=2):
    np.random.seed(seed)
    env.reset()
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 4.0, -10.0, 90.0
    frames, qpos_hist, act_hist, vx_hist = [], [], [], []
    for t in range(episode_len):
        a = act_fn(env)
        _, _, done, _ = env.step(a)
        qpos_hist.append(env.data.qpos.copy())
        act_hist.append(np.asarray(a, dtype=float).copy())
        vx_hist.append(float(env.data.qvel[0]))
        if t % render_every == 0:
            cam.lookat[:] = [env.data.qpos[0], 0.0, 1.2]
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
        if done:
            break
    return frames, np.array(qpos_hist), np.array(act_hist), np.array(vx_hist)


def main(
    checkpoint: str = "runs/walker_gps1/checkpoint_latest.pt",
    seed: int = 0,
    episode_len: int = 400,
    out_prefix: str = "runs/walker_gps1/gait_compare",
) -> None:
    env = Walker2d()
    mppi = MPPI(env, MPPIConfig.load("walker2d"))
    gps_cfg = GPSConfig.load("walker2d")
    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, PolicyConfig())
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    policy.eval()
    renderer = mujoco.Renderer(env.model, height=320, width=320)

    def policy_act(env):
        obs = torch.as_tensor(env._get_obs(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return policy.forward(obs).squeeze(0).numpy()

    def mppi_act(env):
        return mppi.plan_step(env.get_state())[0]

    print("rolling policy ...")
    p_frames, p_qpos, p_act, p_vx = run_and_record(env, policy_act, episode_len, seed, renderer)
    print(f"policy: {len(p_qpos)} steps, vx={p_vx.mean():.2f}")
    print("rolling mppi ...")
    mppi.reset()
    m_frames, m_qpos, m_act, m_vx = run_and_record(env, mppi_act, episode_len, seed, renderer)
    print(f"mppi:   {len(m_qpos)} steps, vx={m_vx.mean():.2f}")

    # side-by-side video, shorter run frozen on its last frame (a fall)
    n = max(len(p_frames), len(m_frames))
    pad = lambda f: f + [f[-1]] * (n - len(f))
    video = [np.hstack([l, r]) for l, r in zip(pad(p_frames), pad(m_frames))]
    video_path = f"{out_prefix}.mp4"
    imageio.mimsave(video_path, video, fps=31, macro_block_size=1)  # ~realtime at render_every=2
    print(f"saved {video_path}  (left: policy, right: MPPI)")

    # Fig.3-style traces: joint angles, actions, forward velocity
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex="col")
    for col, (name, qpos, act, vx) in enumerate(
        (("Neural Network Policy", p_qpos, p_act, p_vx),
         ("MPPI", m_qpos, m_act, m_vx))
    ):
        t = np.arange(len(qpos)) * env._dt
        for j in range(6):
            axes[0, col].plot(t, qpos[:, 3 + j], lw=0.9, label=_JOINTS[j])
            axes[1, col].plot(t, act[:, j], lw=0.7)
        axes[2, col].plot(t, vx, lw=1.2, color="k")
        axes[2, col].axhline(env._v_target, ls="--", color="r", label="target")
        axes[0, col].set_title(f"{name} — joint angles")
        axes[1, col].set_title("actions")
        axes[2, col].set_title("forward velocity (m/s)")
        axes[2, col].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, ncol=3)
    axes[2, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = f"{out_prefix}_traces.png"
    fig.savefig(plot_path, dpi=120)
    print(f"saved {plot_path}")

    # headline numbers
    smooth = lambda a: float(np.sum(np.diff(a, axis=0) ** 2) / max(len(a) - 1, 1))
    print(f"policy: steps={len(p_qpos)} vx={p_vx.mean():.2f} action_smoothness={smooth(p_act):.4f}")
    print(f"mppi:   steps={len(m_qpos)} vx={m_vx.mean():.2f} action_smoothness={smooth(m_act):.4f}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
