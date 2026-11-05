"""Render a loop policy next to the walk it learned from, as an mp4.

Three panels from the same start state of one saved trajectory: the walk
itself (its saved states), the policy alone (u = pi(x), kept running 1 s past a
fall), and the walk's actions with the policy's own corrections (u = u_t +
pi(x) - pi(x_bar_t)). Track camera, every 4th control (31.25 frames per
second, real time). Writes only --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from scripts.diagnostics.feedback_rollouts import step
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.policy_drift import load
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

EVERY = 4
AFTER_FALL = 125
WIDTH, HEIGHT = 400, 300


def drive(env, p, start, xbar, ubar, arm):
    x, seen, fell = start.copy()[None], [], None
    on_path = p.predict(observation(xbar).astype(np.float32)).astype(np.float64)
    for t in range(len(ubar)):
        here = p.predict(observation(x).astype(np.float32)).astype(np.float64)
        u = here if arm == "policy" else ubar[t] + here - on_path[t]
        x = step(env, x, np.clip(u, -1.0, 1.0))
        seen.append(x[0])
        if fell is None and not env._is_healthy(x[0, 2], x[0, 3]):
            fell = t + 1
        if fell is not None and t + 1 >= fell + AFTER_FALL:
            break
    return np.array(seen), fell


def frame(env, renderer, state, label, color):
    mujoco.mj_setState(env.model, env.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, camera="track")
    image = Image.fromarray(renderer.render().copy())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 22), fill=(252, 252, 251))
    draw.text((6, 5), label, fill=color)
    return np.asarray(image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    args = parser.parse_args()
    env = Walker2d(**NATIVE_TASK)
    p = load(args.run / f"policy_{args.round}.pt")
    p.eval()
    with np.load(args.bank / f"episode_{args.source}.npz") as d:
        start = d["states"][0]
    with np.load(args.run / f"round_{args.round}.npz") as z:
        walk, u = z[f"states_{args.source}"], z[f"controls_{args.source}"]
    xbar = np.concatenate([start[None], walk[:-1]])
    alone, fell = drive(env, p, start, xbar, u, "policy")
    helped, helped_fell = drive(env, p, start, xbar, u, "walk + policy")
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    frames = []
    try:
        for t in range(0, len(u), EVERY):
            s = t * 0.008
            a = alone[min(t, len(alone) - 1)]
            note = f"policy alone  {s:4.2f} s" + (
                f"  (fell at {fell * 0.008:.2f} s)" if fell and t >= fell else ""
            )
            h = helped[min(t, len(helped) - 1)]
            frames.append(
                np.concatenate(
                    [
                        frame(env, renderer, walk[t], f"walk it learned from  {s:4.2f} s", (42, 120, 214)),
                        frame(env, renderer, a, note, (235, 104, 52)),
                        frame(env, renderer, h, f"walk's actions + policy's corrections  {s:4.2f} s"
                              + (f" (fell {helped_fell * 0.008:.2f} s)" if helped_fell and t >= helped_fell else ""),
                              (27, 175, 122)),
                    ],
                    axis=1,
                )
            )  # fmt: skip
    finally:
        renderer.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=1.0 / (0.008 * EVERY), macro_block_size=1)
    print(
        args.out,
        "policy alone fell at",
        fell and fell * 0.008,
        "s; with corrections fell at",
        helped_fell and helped_fell * 0.008,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
