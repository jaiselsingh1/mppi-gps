"""Run raw MPPI on Ant Maze and optionally render the rollout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.gps_train_warp import TorchWarpMPPI
from src.envs.ant_maze import AntMaze
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def _apply_mppi_overrides(
    cfg: MPPIConfig,
    *,
    k: int | None,
    h: int | None,
    lam: float | None,
    noise_sigma: float | None,
    noise_temporal_alpha: float | None,
    clip_actions: bool | None,
) -> MPPIConfig:
    if k is not None:
        cfg.K = k
    if h is not None:
        cfg.H = h
    if lam is not None:
        cfg.lam = lam
    if noise_sigma is not None:
        cfg.noise_sigma = noise_sigma
    if noise_temporal_alpha is not None:
        cfg.noise_temporal_alpha = noise_temporal_alpha
    if clip_actions is not None:
        cfg.clip_actions = clip_actions
    return cfg


def _env_video_fps(env: AntMaze, render_every: int) -> int:
    return max(1, int(round(1.0 / (env._dt * render_every))))


def _render_frame(renderer: mujoco.Renderer, env: AntMaze, camera: str | None) -> np.ndarray:
    if camera:
        renderer.update_scene(env.data, camera=camera)
    else:
        renderer.update_scene(env.data)
    return renderer.render().copy()


def _plot_path(env: AntMaze, traces: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    for geom_id in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name or not name.startswith("maze_wall_"):
            continue
        pos = env.model.geom_pos[geom_id]
        size = env.model.geom_size[geom_id]
        rect = plt.Rectangle(
            (pos[0] - size[0], pos[1] - size[1]),
            2.0 * size[0],
            2.0 * size[1],
            color="0.2",
            alpha=0.35,
        )
        ax.add_patch(rect)

    for trace in traces:
        xy = np.asarray(trace["xy"], dtype=float)
        goal = np.asarray(trace["goal"], dtype=float)
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.6, label=f"episode {trace['episode']}")
        ax.scatter(xy[0, 0], xy[0, 1], marker="o", s=35)
        ax.scatter(xy[-1, 0], xy[-1, 1], marker="x", s=45)
        ax.scatter(goal[0], goal[1], marker="*", s=120, color="C3")

    ax.set_title("Ant Maze MPPI Path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-10.5, 10.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(
    episodes: int = 3,
    steps: int = 200,
    seed: int = 0,
    backend: Literal["warp", "numpy"] = "warp",
    device: str = "cuda",
    render: bool = True,
    out_dir: str = "runs/ant_maze_mppi_check",
    video_name: str = "ant_maze_mppi.mp4",
    path_name: str = "ant_maze_mppi_path.png",
    summary_name: str = "summary.json",
    width: int = 960,
    height: int = 720,
    camera: str | None = None,
    render_every: int = 1,
    k: int | None = None,
    h: int | None = None,
    lam: float | None = None,
    noise_sigma: float | None = None,
    noise_temporal_alpha: float | None = None,
    clip_actions: bool | None = None,
) -> None:
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}.")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if render_every <= 0:
        raise ValueError(f"render_every must be positive, got {render_every}.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = _apply_mppi_overrides(
        MPPIConfig.load("ant_maze"),
        k=k,
        h=h,
        lam=lam,
        noise_sigma=noise_sigma,
        noise_temporal_alpha=noise_temporal_alpha,
        clip_actions=clip_actions,
    )

    env = AntMaze()
    planner_env: AntMaze | None = None
    if backend == "warp":
        planner_env = AntMaze(use_warp=True, nworld=cfg.K)
        controller = TorchWarpMPPI(planner_env, cfg, n_batches=1, device=device)
    else:
        env.close()
        env = AntMaze(use_warp=False)
        controller = MPPI(env, cfg)

    renderer = mujoco.Renderer(env.model, height=height, width=width) if render else None
    frames: list[np.ndarray] = []
    traces: list[dict] = []
    episode_summaries: list[dict] = []

    try:
        for episode in range(episodes):
            np.random.seed(seed + episode)
            env.reset()
            controller.reset()

            total_cost = 0.0
            first_success_t: int | None = None
            hold_count = 0
            max_hold_count = 0
            xy_trace = [env.data.qpos[:2].copy()]
            metrics = env.task_metrics()

            for t in range(steps):
                state = env.get_state()
                if backend == "warp":
                    action, info = controller.plan_step(state, goals=env.goal)
                else:
                    action, info = controller.plan_step(state)
                _, cost, done, _ = env.step(action)
                total_cost += float(cost)

                metrics = env.task_metrics()
                if metrics["success"]:
                    if first_success_t is None:
                        first_success_t = t
                    hold_count += 1
                else:
                    hold_count = 0
                max_hold_count = max(max_hold_count, hold_count)
                xy_trace.append(env.data.qpos[:2].copy())

                if renderer is not None and t % render_every == 0:
                    frames.append(_render_frame(renderer, env, camera))

                if t % 20 == 0:
                    print(
                        f"episode={episode} step={t:4d} "
                        f"dist={metrics['xy_dist']:7.3f} z={metrics['z_pos']:5.2f} "
                        f"cost={cost:9.3f} S_min={info['cost_min']:9.2f} "
                        f"S_mean={info['cost_mean']:9.2f} n_eff={info['n_eff']:7.2f} "
                        f"success={metrics['success']}"
                    )
                if done:
                    break

            final_metrics = env.task_metrics()
            trace = {
                "episode": episode,
                "xy": np.asarray(xy_trace).tolist(),
                "goal": env.goal.copy().tolist(),
            }
            traces.append(trace)
            summary = {
                "episode": episode,
                "total_cost": total_cost,
                "cost_per_step": total_cost / max(len(xy_trace) - 1, 1),
                "hit_success": first_success_t is not None,
                "final_success": bool(final_metrics["success"]),
                "final_hold_success": hold_count >= 25,
                "max_hold_steps": max_hold_count,
                "time_to_hit": first_success_t if first_success_t is not None else steps,
                "final_xy_dist": final_metrics["xy_dist"],
                "final_z": final_metrics["z_pos"],
                "goal": env.goal.copy().tolist(),
                "final_xy": env.data.qpos[:2].copy().tolist(),
            }
            episode_summaries.append(summary)
            print(
                f"episode={episode} done total_cost={total_cost:.2f} "
                f"final_dist={summary['final_xy_dist']:.3f} "
                f"hit={summary['hit_success']} final={summary['final_success']} "
                f"hold25={summary['final_hold_success']}"
            )
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
        if planner_env is not None:
            planner_env.close()

    if render and frames:
        import mediapy

        video_path = out_path / video_name
        mediapy.write_video(video_path, frames, fps=_env_video_fps(env, render_every))
        print(f"saved video: {video_path}")

    path_plot = out_path / path_name
    plot_env = AntMaze()
    try:
        _plot_path(plot_env, traces, path_plot)
    finally:
        plot_env.close()
    print(f"saved path plot: {path_plot}")

    summary = {
        "backend": backend,
        "seed": seed,
        "episodes": episodes,
        "steps": steps,
        "mppi": {
            "K": cfg.K,
            "H": cfg.H,
            "lam": cfg.lam,
            "noise_sigma": cfg.noise_sigma,
            "noise_temporal_alpha": cfg.noise_temporal_alpha,
            "clip_actions": cfg.clip_actions,
        },
        "success_rate": float(np.mean([ep["final_success"] for ep in episode_summaries])),
        "hit_success_rate": float(np.mean([ep["hit_success"] for ep in episode_summaries])),
        "final_hold_success_rate": float(np.mean([ep["final_hold_success"] for ep in episode_summaries])),
        "mean_final_xy_dist": float(np.mean([ep["final_xy_dist"] for ep in episode_summaries])),
        "episodes_detail": episode_summaries,
    }
    summary_path = out_path / summary_name
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved summary: {summary_path}")
    print(
        "summary "
        f"hit={summary['hit_success_rate']:.3f} "
        f"final={summary['success_rate']:.3f} "
        f"hold25={summary['final_hold_success_rate']:.3f} "
        f"mean_final_dist={summary['mean_final_xy_dist']:.3f}"
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
