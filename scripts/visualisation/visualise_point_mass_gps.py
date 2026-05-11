"""Visualise a completed point-mass GPS/BC run."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy
import mujoco
import numpy as np
import torch
import tyro

from src.envs.point_mass import PointMass
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig


def _env_video_fps(env: PointMass) -> int:
    return max(1, int(round(1.0 / env._dt)))


def _checkpoint_iter_key(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def _select_policy_checkpoint(run_dir: Path) -> Path | None:
    latest = run_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest
    milestones = sorted(run_dir.glob("checkpoint_iter_*.pt"), key=_checkpoint_iter_key)
    return milestones[-1] if milestones else None


def _policy_action(policy: DeterministicPolicy, obs: np.ndarray) -> np.ndarray:
    device = next(policy.parameters()).device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return policy.forward(obs_t).squeeze(0).cpu().numpy()


def _render_frame(renderer: mujoco.Renderer, env: PointMass) -> np.ndarray:
    renderer.update_scene(env.data, camera="fixed")
    return renderer.render().copy()


def _rollout_policy(
    policy: DeterministicPolicy,
    episode_len: int,
    seed: int,
    render: bool,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    env = PointMass()
    renderer = mujoco.Renderer(env.model, height=480, width=640) if render else None
    frames: list[np.ndarray] = []
    policy.eval()
    try:
        np.random.seed(seed)
        env.reset()
        qpos = [env.data.qpos.copy()]
        qvel = [env.data.qvel.copy()]
        actions: list[np.ndarray] = []
        costs: list[float] = []
        dist = [env.task_metrics()["tip_dist"]]
        success: list[bool] = []
        if renderer is not None:
            frames.append(_render_frame(renderer, env))

        for _ in range(episode_len):
            action = _policy_action(policy, env._get_obs())
            _, cost, done, _ = env.step(action)
            metrics = env.task_metrics()
            qpos.append(env.data.qpos.copy())
            qvel.append(env.data.qvel.copy())
            actions.append(action.copy())
            costs.append(float(cost))
            dist.append(metrics["tip_dist"])
            success.append(metrics["success"])
            if renderer is not None:
                frames.append(_render_frame(renderer, env))
            if done:
                break
    finally:
        if renderer is not None:
            renderer.close()
        env.close()

    return {
        "qpos": np.asarray(qpos, dtype=float),
        "qvel": np.asarray(qvel, dtype=float),
        "goal": env.goal.copy(),
        "actions": np.asarray(actions, dtype=float),
        "costs": np.asarray(costs, dtype=float),
        "dist": np.asarray(dist, dtype=float),
        "success": np.asarray(success, dtype=bool),
        "frames": frames,
    }


def _rollout_mppi(
    mppi_cfg: MPPIConfig,
    episode_len: int,
    seed: int,
    render: bool,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    env = PointMass()
    mppi = MPPI(env, mppi_cfg)
    renderer = mujoco.Renderer(env.model, height=480, width=640) if render else None
    frames: list[np.ndarray] = []
    try:
        np.random.seed(seed)
        env.reset()
        mppi.reset()
        qpos = [env.data.qpos.copy()]
        qvel = [env.data.qvel.copy()]
        actions: list[np.ndarray] = []
        costs: list[float] = []
        dist = [env.task_metrics()["tip_dist"]]
        success: list[bool] = []
        if renderer is not None:
            frames.append(_render_frame(renderer, env))

        for _ in range(episode_len):
            action, _ = mppi.plan_step(env.get_state())
            _, cost, done, _ = env.step(action)
            metrics = env.task_metrics()
            qpos.append(env.data.qpos.copy())
            qvel.append(env.data.qvel.copy())
            actions.append(action.copy())
            costs.append(float(cost))
            dist.append(metrics["tip_dist"])
            success.append(metrics["success"])
            if renderer is not None:
                frames.append(_render_frame(renderer, env))
            if done:
                break
    finally:
        if renderer is not None:
            renderer.close()
        env.close()

    return {
        "qpos": np.asarray(qpos, dtype=float),
        "qvel": np.asarray(qvel, dtype=float),
        "goal": env.goal.copy(),
        "actions": np.asarray(actions, dtype=float),
        "costs": np.asarray(costs, dtype=float),
        "dist": np.asarray(dist, dtype=float),
        "success": np.asarray(success, dtype=bool),
        "frames": frames,
    }


def _max_true_run(values: np.ndarray) -> int:
    best = 0
    run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _trace_label(name: str, trace: dict[str, np.ndarray | list[np.ndarray]]) -> str:
    costs = trace["costs"]
    success = trace["success"]
    assert isinstance(costs, np.ndarray)
    assert isinstance(success, np.ndarray)
    cost_per_step = float(np.sum(costs) / max(len(costs), 1))
    hits = np.flatnonzero(success)
    hit = f"hit t={int(hits[0])}" if len(hits) else "no hit"
    hold = _max_true_run(success)
    dist = trace["dist"]
    assert isinstance(dist, np.ndarray)
    return f"{name}: cost/step={cost_per_step:.4f}, {hit}, hold={hold}, final_dist={float(dist[-1]):.4f}"


def plot_metrics(run_dir: Path) -> Path:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    iters = [r["iter"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(iters, [r["mppi_cost_per_step"] for r in rows], marker="o", label="MPPI collect")
    eval_rows = [r for r in rows if r.get("policy_eval_cost_per_step_mean") is not None]
    if eval_rows:
        ax[0].errorbar(
            [r["iter"] for r in eval_rows],
            [r["policy_eval_cost_per_step_mean"] for r in eval_rows],
            yerr=[r["policy_eval_cost_per_step_std"] for r in eval_rows],
            marker="s",
            capsize=3,
            label="Policy eval",
        )
    raw_rows = [r for r in rows if r.get("raw_mppi_eval_cost_per_step_mean") is not None]
    if raw_rows:
        ax[0].errorbar(
            [r["iter"] for r in raw_rows],
            [r["raw_mppi_eval_cost_per_step_mean"] for r in raw_rows],
            yerr=[r["raw_mppi_eval_cost_per_step_std"] for r in raw_rows],
            marker="^",
            linestyle="--",
            capsize=3,
            label="Raw MPPI eval",
        )
    ax[0].set_title("Cost Per Step")
    ax[0].set_xlabel("iter")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(iters, [r["policy_eval_hit_success_rate"] for r in rows], marker="o", label="policy hit")
    ax[1].plot(iters, [r["policy_eval_hold_success_rate"] for r in rows], marker="s", label="policy hold")
    ax[1].plot(iters, [r["mppi_hold_success_rate"] for r in rows], marker="^", label="MPPI hold")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title("Success")
    ax[1].set_xlabel("iter")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    ax[2].plot(iters, [r["bc_loss_final"] for r in rows], marker="o", color="C2")
    ax[2].set_title("BC Loss")
    ax[2].set_xlabel("iter")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out = run_dir / "metrics.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_trace_compare(
    run_dir: Path,
    policy_trace: dict[str, np.ndarray | list[np.ndarray]],
    mppi_trace: dict[str, np.ndarray | list[np.ndarray]],
) -> Path:
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for trace, color, name in [(mppi_trace, "C0", "Raw MPPI"), (policy_trace, "C3", "BC policy")]:
        qpos = trace["qpos"]
        dist = trace["dist"]
        actions = trace["actions"]
        qvel = trace["qvel"]
        assert isinstance(qpos, np.ndarray)
        assert isinstance(dist, np.ndarray)
        assert isinstance(actions, np.ndarray)
        assert isinstance(qvel, np.ndarray)
        ax[0, 0].plot(qpos[:, 0], qpos[:, 1], color=color, label=name)
        ax[0, 0].scatter(qpos[0, 0], qpos[0, 1], color=color, marker="o")
        ax[0, 0].scatter(qpos[-1, 0], qpos[-1, 1], color=color, marker="x")
        ax[0, 1].plot(dist, color=color, label=name)
        if len(actions):
            ax[1, 0].plot(actions[:, 0], color=color, linestyle="-", label=f"{name} ax")
            ax[1, 0].plot(actions[:, 1], color=color, linestyle="--", label=f"{name} ay")
        ax[1, 1].plot(np.linalg.norm(qvel, axis=1), color=color, label=name)

    goal = policy_trace.get("goal", np.zeros(2))
    assert isinstance(goal, np.ndarray)
    ax[0, 0].scatter([goal[0]], [goal[1]], color="k", marker="*", s=120, label="goal")
    ax[0, 0].set_xlim(-0.31, 0.31)
    ax[0, 0].set_ylim(-0.31, 0.31)
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_title("XY Path")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend()
    ax[0, 1].set_title("Distance To Goal")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].legend()
    ax[1, 0].set_title("Actions")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].set_title("Speed")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend()

    fig.suptitle(f"{_trace_label('Raw MPPI', mppi_trace)}\n{_trace_label('BC policy', policy_trace)}")
    fig.tight_layout()
    out = run_dir / "point_mass_trace_compare_latest.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _load_policy(path: Path, gps_cfg: GPSConfig) -> DeterministicPolicy:
    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, PolicyConfig())
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy


def render_checkpoints(run_dir: Path, gps_cfg: GPSConfig, episode_len: int, seed: int) -> list[Path]:
    paths: list[Path] = []
    fps_env = PointMass()
    fps = _env_video_fps(fps_env)
    fps_env.close()
    for ckpt_path in sorted(run_dir.glob("checkpoint_iter_*.pt"), key=_checkpoint_iter_key):
        policy = _load_policy(ckpt_path, gps_cfg)
        trace = _rollout_policy(policy, episode_len=episode_len, seed=seed, render=True)
        iter_tag = ckpt_path.stem.split("_")[-1]
        video_path = run_dir / f"policy_iter_{iter_tag}.mp4"
        mediapy.write_video(str(video_path), trace["frames"], fps=fps)
        paths.append(video_path)
        print(f"{iter_tag}: {_trace_label('policy', trace)} -> {video_path}")
    return paths


def main(
    run_name: str = "point_mass_bc_warp_eps8",
    /,
    episode_len: int = 250,
    seed: int = 0,
    no_mppi: bool = False,
) -> None:
    run_dir = Path("runs") / run_name
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    gps_cfg = GPSConfig.load("point_mass")
    mppi_cfg = MPPIConfig.load("point_mass")
    fps_env = PointMass()
    fps = _env_video_fps(fps_env)
    fps_env.close()
    metrics_path = plot_metrics(run_dir)
    print(f"saved {metrics_path}")

    videos = render_checkpoints(run_dir, gps_cfg, episode_len=episode_len, seed=seed)
    latest_ckpt = _select_policy_checkpoint(run_dir)
    if latest_ckpt is None:
        raise SystemExit(f"no checkpoints found in {run_dir}")

    latest_policy = _load_policy(latest_ckpt, gps_cfg)
    policy_trace = _rollout_policy(latest_policy, episode_len=episode_len, seed=seed, render=True)
    latest_video = run_dir / "policy_latest_seed0.mp4"
    mediapy.write_video(str(latest_video), policy_trace["frames"], fps=fps)
    videos.append(latest_video)
    print(f"latest: {_trace_label('policy', policy_trace)} -> {latest_video}")

    if no_mppi:
        return

    mppi_trace = _rollout_mppi(mppi_cfg, episode_len=episode_len, seed=seed, render=True)
    mppi_video = run_dir / "raw_mppi_seed0.mp4"
    mediapy.write_video(str(mppi_video), mppi_trace["frames"], fps=fps)
    print(f"raw: {_trace_label('raw MPPI', mppi_trace)} -> {mppi_video}")

    policy_frames = policy_trace["frames"]
    mppi_frames = mppi_trace["frames"]
    assert isinstance(policy_frames, list)
    assert isinstance(mppi_frames, list)
    n = min(len(policy_frames), len(mppi_frames))
    side_by_side = [
        np.concatenate([mppi_frames[i], policy_frames[i]], axis=1)
        for i in range(n)
    ]
    compare_video = run_dir / "raw_mppi_vs_policy_latest_seed0.mp4"
    mediapy.write_video(str(compare_video), side_by_side, fps=fps)
    print(f"saved {compare_video}")

    trace_path = plot_trace_compare(run_dir, policy_trace, mppi_trace)
    print(f"saved {trace_path}")


if __name__ == "__main__":
    tyro.cli(main)
