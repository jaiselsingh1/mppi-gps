"""Render Walker controllers side by side from the same reset.

Each panel is one controller (raw MPPI or a policy acting deterministically),
filmed by the tracking camera every 4th control (real time at 31.25 fps). A
panel that falls freezes on its last frame, marked with the time of the fall.
Writes an MP4 and a half-rate GIF of the grid.

controllers: "label=mppi:K:H" or "label=policy:path", as in
scripts/diagnostics/joint_plots.py.
"""
import json
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _film(job: dict) -> dict:
    import mujoco
    import torch

    from src.envs.walker2d import Walker2d
    from src.mppi.mppi import MPPI
    from src.rl.sac import load_policy
    from src.utils.config import MPPIConfig

    torch.set_num_threads(1)
    task = json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())
    env = Walker2d(**task, nthread=1)
    np.random.seed(job["reset_seed"])
    obs = env.reset()
    kind, _, rest = job["spec"].partition(":")
    if kind == "mppi":
        K, H = (int(n) for n in rest.split(":"))
        base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
        planner = MPPI(env, replace(base, K=K, H=H), seed=job["planner_seed"])
        act = lambda _: planner.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())[0]
    else:
        payload = torch.load(_REPO_ROOT / rest, map_location="cpu", weights_only=False)
        actor = load_policy(payload.get("policy", payload), device="cpu")
        act = lambda o: actor.act(o, deterministic=True)
    renderer = mujoco.Renderer(env.model, height=job["height"], width=job["width"])
    frames, fell_at, velocities = [], None, []
    for t in range(job["steps"]):
        obs, _, done, info = env.step(act(obs))
        velocities.append(info["x_velocity"])
        if t % 4 == 0:
            renderer.update_scene(env.data, camera="track")
            frames.append(renderer.render().copy())
        if done:
            fell_at = (t + 1) * env._dt
            break
    renderer.close()
    return {"label": job["label"], "frames": np.asarray(frames), "fell_at": fell_at,
            "mean_vx": float(np.mean(velocities))}


def _caption(frame: np.ndarray, lines: list[str], alert: bool) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 14 * len(lines) + 6], fill=(255, 255, 255))
    for i, line in enumerate(lines):
        draw.text((5, 3 + 14 * i), line, fill=(200, 0, 0) if alert and i == len(lines) - 1 else (0, 0, 0))
    return np.asarray(image)


def main(
    out_dir: Path,
    controllers: tuple[str, ...],
    reset_seed: int = 81_000,
    planner_seed: int = 95_000,
    steps: int = 1000,
    columns: int = 3,
    width: int = 320,
    height: int = 240,
) -> None:
    import mediapy
    from PIL import Image

    jobs = [{"label": label, "spec": spec, "reset_seed": reset_seed, "planner_seed": planner_seed,
             "steps": steps, "width": width, "height": height}
            for label, spec in (entry.rsplit("=", 1) for entry in controllers)]
    out_dir.mkdir(parents=True, exist_ok=False)
    with Pool(min(len(jobs), 8)) as pool:
        panels = pool.map(_film, jobs)
    count = (steps + 3) // 4
    rows = -(-len(panels) // columns)
    grid = []
    for i in range(count):
        tiles = []
        for panel in panels:
            frames = panel["frames"]
            ended = i >= len(frames)
            status = (f"fell at {panel['fell_at']:.1f} s" if panel["fell_at"] is not None and ended
                      else f"t = {4 * i * 0.008:.1f} s")
            tiles.append(_caption(frames[min(i, len(frames) - 1)], [panel["label"], status], ended))
        tiles += [np.full_like(tiles[0], 255)] * (rows * columns - len(tiles))
        grid.append(np.concatenate([np.concatenate(tiles[r * columns:(r + 1) * columns], axis=1)
                                    for r in range(rows)], axis=0))
    mediapy.write_video(str(out_dir / "walkers.mp4"), grid, fps=31.25)
    images = [Image.fromarray(frame) for frame in grid[::2]]
    images[0].save(out_dir / "walkers.gif", save_all=True, append_images=images[1:], duration=64, loop=0)
    summary = [{"label": p["label"], "fell_at": p["fell_at"], "mean_vx": p["mean_vx"]} for p in panels]
    (out_dir / "summary.json").write_text(json.dumps({"reset_seed": reset_seed, "controllers": summary}, indent=2))
    for row in summary:
        print(json.dumps(row))


if __name__ == "__main__":
    tyro.cli(main)
