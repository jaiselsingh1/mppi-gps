from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import mediapy
import mujoco
import numpy as np
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ddpg_mppi_gps.train_walker2d_drqv2 import (
    StateActor,
    Walker2dGaitStats,
    ensure_offscreen_size,
    resolve_device,
    resolve_run_dir,
)
from src.envs.walker2d import Walker2d


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> tuple[Path, str]:
    if checkpoint == "best":
        return run_dir / "checkpoint_best.pt", "best"
    if checkpoint == "latest":
        return run_dir / "checkpoint_latest.pt", "latest"
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path, path.stem


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def load_actor(checkpoint: dict[str, Any], device: torch.device) -> StateActor:
    config = checkpoint.get("config", {})
    actor = StateActor(
        obs_dim=int(config.get("obs_dim", 17)),
        act_dim=int(config.get("act_dim", 6)),
        feature_dim=int(config.get("feature_dim", 50)),
        hidden_dim=int(config.get("hidden_dim", 1024)),
    ).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


@torch.no_grad()
def policy_action(actor: StateActor, obs: np.ndarray, device: torch.device) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    return actor.mean_action(obs_t).squeeze(0).cpu().numpy().astype(np.float32)


def render_frame(renderer: mujoco.Renderer, env: Walker2d, camera: str | None) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera) if camera else renderer.update_scene(env.data)
    return renderer.render().copy()


def episode_video_path(base_path: Path, episode: int, episodes: int) -> Path:
    if episodes == 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}_ep{episode:02d}{base_path.suffix}")


def rollout_and_render(
    *,
    actor: StateActor,
    checkpoint_config: dict[str, Any],
    device: torch.device,
    seed: int,
    steps: int,
    height: int,
    width: int,
    camera: str | None,
    fps: int | None,
    render_every: int,
    stop_on_done: bool,
    output_path: Path,
) -> dict[str, Any]:
    env = Walker2d(
        cost_style=str(checkpoint_config.get("walker_cost_style", "gymnasium")),
        target_velocity=float(checkpoint_config.get("walker_target_velocity", 1.5)),
    )
    ensure_offscreen_size(env.model, height, width)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    frames: list[np.ndarray] = []
    gait = Walker2dGaitStats()
    np.random.seed(seed)
    obs = env.reset()
    total_cost = 0.0
    done = False

    try:
        frames.append(render_frame(renderer, env, camera))
        for step in range(steps):
            action = np.clip(policy_action(actor, obs, device), *env.action_bounds)
            obs, cost, done, info = env.step(action)
            total_cost += float(cost)
            gait.update(
                env.model,
                env.data,
                action,
                reward=-float(cost),
                cost=float(cost),
                x_velocity=float(info.get("x_velocity", env.data.qvel[0])),
                healthy=bool(info.get("healthy", False)),
            )
            if (step + 1) % render_every == 0:
                frames.append(render_frame(renderer, env, camera))
            if done and stop_on_done:
                break
    finally:
        renderer.close()
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_fps = fps if fps is not None else max(1, int(round(1.0 / (env._dt * render_every))))
    mediapy.write_video(str(output_path), frames, fps=video_fps)

    record = {
        "video_path": str(output_path),
        "seed": seed,
        "steps": gait.steps,
        "requested_steps": steps,
        "done": bool(done),
        "cost": total_cost,
        "reward": -total_cost,
        "frames": len(frames),
        "fps": video_fps,
        **gait.summary(prefix="gait_"),
    }
    return record


def main(
    run_name: str = "walker2d_drqv2_state",
    runs_root: str = "runs",
    checkpoint: str = "best",
    out: str | None = None,
    episodes: int = 1,
    steps: int = 1000,
    seed: int = 0,
    device: str = "auto",
    height: int = 540,
    width: int = 960,
    camera: str | None = "track",
    fps: int | None = None,
    render_every: int = 1,
    stop_on_done: bool = True,
) -> None:
    run_dir = resolve_run_dir(runs_root, run_name)
    checkpoint_path, checkpoint_tag = resolve_checkpoint(run_dir, checkpoint)
    torch_device = resolve_device(device)
    checkpoint_data = load_checkpoint(checkpoint_path, torch_device)
    actor = load_actor(checkpoint_data, torch_device)
    if camera in {"", "none", "None"}:
        camera = None

    output_base = Path(out).expanduser() if out is not None else run_dir / f"policy_{checkpoint_tag}_seed{seed}.mp4"
    if not output_base.is_absolute():
        output_base = (PROJECT_ROOT / output_base).resolve()

    metrics_path = run_dir / f"render_policy_{checkpoint_tag}_metrics.jsonl"
    print(f"checkpoint: {checkpoint_path}")
    for episode in range(episodes):
        video_path = episode_video_path(output_base, episode, episodes)
        record = rollout_and_render(
            actor=actor,
            checkpoint_config=checkpoint_data.get("config", {}),
            device=torch_device,
            seed=seed + episode,
            steps=steps,
            height=height,
            width=width,
            camera=camera,
            fps=fps,
            render_every=render_every,
            stop_on_done=stop_on_done,
            output_path=video_path,
        )
        record.update({"episode": episode, "checkpoint": str(checkpoint_path), "checkpoint_tag": checkpoint_tag})
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"episode={episode} reward={record['reward']:.1f} "
            f"len={record['steps']} -> {video_path}"
        )


if __name__ == "__main__":
    tyro.cli(main)
