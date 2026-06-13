"""Side-by-side before/after render of two policies on the same seed.

For the inverted experiment: the jittery RL policy (before the GPS loop) vs
the regularized policy (after). Both walk; the difference is gait smoothness.
Produces a side-by-side video (annotated) and a joint-angle / action trace
comparison (Mordatch'15 Fig.3 style).

Usage:
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/visualisation/render_before_after.py \
      --before runs/walker_td3_matched2/actor_matched_final.pt \
      --after runs/walker_gps11_invert_noanchor/checkpoint_iter_002.pt
"""
from __future__ import annotations

import cv2
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import tyro

from src.envs.walker2d import Walker2d
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import PolicyConfig

_JOINTS = ["thigh", "leg", "foot", "thigh_L", "leg_L", "foot_L"]


def rollout(env, ckpt, seed, steps, renderer, render_every=2):
    p = DeterministicPolicy(17, 6, PolicyConfig())
    p.load_state_dict(torch.load(ckpt, map_location="cpu"))
    p.eval()
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 4.0, -10.0, 90.0
    np.random.seed(seed)
    env.reset()
    frames, qpos, acts = [], [], []
    for t in range(steps):
        obs = torch.as_tensor(env._get_obs(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a = p.forward(obs).squeeze(0).numpy()
        env.step(a)
        qpos.append(env.data.qpos.copy())
        acts.append(a)
        if t % render_every == 0:
            cam.lookat[:] = [env.data.qpos[0], 0.0, 1.2]
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
    return frames, np.array(qpos), np.array(acts)


def main(
    before: str = "runs/walker_td3_matched2/actor_matched_final.pt",
    after: str = "runs/walker_gps11_invert_noanchor/checkpoint_iter_002.pt",
    seed: int = 2,
    steps: int = 400,
    out_prefix: str = "runs/inverted_before_after",
) -> None:
    env = Walker2d()
    renderer = mujoco.Renderer(env.model, height=320, width=320)
    bf, bq, ba = rollout(env, before, seed, steps, renderer)
    af, aq, aa = rollout(env, after, seed, steps, renderer)
    sm_b = float(np.sum(np.diff(ba, axis=0) ** 2) / (len(ba) - 1))
    sm_a = float(np.sum(np.diff(aa, axis=0) ** 2) / (len(aa) - 1))

    def annotate(frame, label, sub):
        f = frame.copy()
        cv2.putText(f, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(f, sub, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1)
        return f

    n = min(len(bf), len(af))
    video = [
        np.hstack([
            annotate(bf[i], "RL policy (before)", f"smoothness {sm_b:.2f}"),
            annotate(af[i], "after GPS loop", f"smoothness {sm_a:.2f}"),
        ])
        for i in range(n)
    ]
    vpath = f"{out_prefix}.mp4"
    imageio.mimsave(vpath, video, fps=31, macro_block_size=1)
    print(f"saved {vpath}  (left: before, right: after)")

    # joint-angle + action traces
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for col, (name, q, a, sm) in enumerate(
        (("RL policy (before)", bq, ba, sm_b), ("after GPS loop", aq, aa, sm_a))
    ):
        t = np.arange(len(q)) * env._dt
        for j in range(6):
            axes[0, col].plot(t, q[:, 3 + j], lw=0.9, label=_JOINTS[j] if col == 0 else None)
            axes[1, col].plot(t, a[:, j], lw=0.7)
        axes[0, col].set_title(f"{name} — joint angles")
        axes[1, col].set_title(f"actions (smoothness={sm:.3f})")
        axes[1, col].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, ncol=3)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Inverted experiment: the GPS loop regularizes a competent RL policy (both walk 1000/1000)")
    fig.tight_layout()
    ppath = f"{out_prefix}_traces.png"
    fig.savefig(ppath, dpi=130)
    print(f"saved {ppath}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
