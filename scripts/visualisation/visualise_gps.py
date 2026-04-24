"""Visualise a completed GPS run: metrics plot + per-checkpoint rollout videos.

Usage:
    uv run scripts/visualise_gps.py gps_r1_lambda0
    uv run scripts/visualise_gps.py gps_r2_lambda0.01 --n_eps 5 --ep_len 500
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy
import torch
import tyro

from src.envs.acrobot import Acrobot
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import GPSConfig, PolicyConfig
from src.utils.eval import evaluate_policy


def plot_metrics(run_dir: Path, gps_cfg: GPSConfig) -> None:
    """Plot per-step cost curves. Step counts come from the metrics record
    when present; otherwise we fall back to values from ``gps_cfg``."""
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    iters = [r["iter"] for r in rows]
    bc_loss = [r["bc_loss_final"] for r in rows]

    def mppi_ps(r: dict) -> float:
        if "mppi_cost_per_step" in r:
            return r["mppi_cost_per_step"]
        L = r.get("mppi_ep_len", gps_cfg.steps_per_episode)
        return r["mppi_rollout_mean_cost"] / L

    def eval_ps(r: dict) -> tuple[float, float]:
        if "policy_eval_cost_per_step_mean" in r and r["policy_eval_cost_per_step_mean"] is not None:
            return r["policy_eval_cost_per_step_mean"], r["policy_eval_cost_per_step_std"]
        L = r.get("eval_ep_len", gps_cfg.eval_episode_len)
        return r["policy_eval_mean_cost"] / L, r["policy_eval_std_cost"] / L

    mppi_cost = [mppi_ps(r) for r in rows]
    eval_rows = [r for r in rows if r["policy_eval_mean_cost"] is not None]
    eval_iters = [r["iter"] for r in eval_rows]
    eval_pairs = [eval_ps(r) for r in eval_rows]
    eval_mean = [m for m, _ in eval_pairs]
    eval_std = [s for _, s in eval_pairs]

    has_breakdown = "mppi_S_env_mean" in rows[0]
    n_panels = 3 if has_breakdown else 2
    fig, ax = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4))

    ax[0].plot(
        iters, mppi_cost, marker="o",
        label=f"MPPI collection (per step, {gps_cfg.steps_per_episode}-step eps)",
    )
    if eval_mean:
        ax[0].errorbar(
            eval_iters, eval_mean, yerr=eval_std,
            marker="s",
            label=f"Policy eval (per step, {gps_cfg.eval_n_episodes}-ep mean ± std)",
            capsize=3,
        )
    ax[0].set_xlabel("GPS iter")
    ax[0].set_ylabel("env cost / step")
    ax[0].set_title("Cost curves (per step)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    if has_breakdown:
        s_env   = [r["mppi_S_env_mean"]   for r in rows]
        s_is    = [abs(r["mppi_S_is_mean"])   for r in rows]   # |mean|, IS is 0-mean by symmetry
        s_isstd = [r["mppi_S_is_std"]     for r in rows]
        s_track = [r["mppi_S_track_mean"] for r in rows]
        s_total = [r["mppi_S_total_mean"] for r in rows]
        ax[1].plot(iters, s_env,   marker="o", label="S_env (running+terminal)")
        ax[1].plot(iters, s_isstd, marker="^", label="|S_is| std (IS correction magnitude)")
        ax[1].plot(iters, s_track, marker="s", label="S_track (λ_track·‖a−π‖²)")
        ax[1].plot(iters, s_total, marker="x", linestyle="--", color="k", alpha=0.5, label="S_total (mean)")
        ax[1].set_xlabel("GPS iter")
        ax[1].set_ylabel("cost units per sample (horizon sum)")
        ax[1].set_title("MPPI S-components (mean over K samples, averaged over plan_step calls)")
        ax[1].set_yscale("symlog", linthresh=1e-3)
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)

    ax[-1].plot(iters, bc_loss, marker="o", color="C2")
    ax[-1].set_xlabel("GPS iter")
    ax[-1].set_ylabel("BC MSE (trailing mean)")
    ax[-1].set_title("BC loss")
    ax[-1].grid(alpha=0.3)

    plt.tight_layout()
    out = run_dir / "metrics.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


def render_checkpoints(
    run_dir: Path,
    gps_cfg: GPSConfig,
    n_eps: int,
    episode_len: int,
    seed: int,
) -> None:
    policy_cfg = PolicyConfig()

    env = Acrobot()
    milestones = sorted(run_dir.glob("checkpoint_iter_*.pt"))
    if not milestones:
        print(f"no milestone checkpoints in {run_dir}")
        return

    for ckpt_path in milestones:
        policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, policy_cfg)
        policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        policy.eval()

        stats = evaluate_policy(
            policy, env,
            n_episodes=n_eps, episode_len=episode_len,
            seed=seed, render=True,
        )
        iter_tag = ckpt_path.stem.split("_")[-1]
        video_path = run_dir / f"policy_iter_{iter_tag}.mp4"
        mediapy.write_video(str(video_path), stats["frames"], fps=30)
        print(
            f"iter {iter_tag}  mean_cost={stats['mean_cost']:7.1f}"
            f"  std={stats['std_cost']:.1f}  -> {video_path}"
        )

    env.close()


def main(
    run_name: str,
    /,
    n_eps: int = 3,
    ep_len: int | None = None,
    seed: int = 0,
    no_videos: bool = False,
) -> None:
    """Visualise a completed GPS run.

    Args:
        run_name: directory name under runs/
        n_eps: number of rollout episodes per checkpoint video
        ep_len: episode length for rendered videos (defaults to gps_cfg.eval_episode_len)
        seed: eval seed
        no_videos: only plot metrics, skip video rendering
    """
    run_dir = Path("runs") / run_name
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    gps_cfg = GPSConfig.load("acrobot")
    plot_metrics(run_dir, gps_cfg)
    if not no_videos:
        render_checkpoints(
            run_dir,
            gps_cfg,
            n_eps=n_eps,
            episode_len=ep_len if ep_len is not None else gps_cfg.eval_episode_len,
            seed=seed,
        )


if __name__ == "__main__":
    tyro.cli(main)
