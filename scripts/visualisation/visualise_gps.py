"""Visualise a completed GPS run: metrics plots, trace comparison, and videos.

Usage:
    uv run python scripts/visualisation/visualise_gps.py gps_r1_lambda0
    uv run python scripts/visualisation/visualise_gps.py gps_r2_lambda0.01 --n_eps 5 --ep_len 500
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy
import numpy as np
import torch
import tyro

from src.envs.acrobot import Acrobot
from src.gps.coupling import make_policy_filter_coupling
from src.gps.prior import make_policy_tracking_prior
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig
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
    raw_mppi_eval_rows = [
        r for r in rows
        if r.get("raw_mppi_eval_mean_cost") is not None
    ]
    raw_mppi_eval_iters = [r["iter"] for r in raw_mppi_eval_rows]
    raw_mppi_eval_mean = [
        r["raw_mppi_eval_cost_per_step_mean"] for r in raw_mppi_eval_rows
    ]
    raw_mppi_eval_std = [
        r["raw_mppi_eval_cost_per_step_std"] for r in raw_mppi_eval_rows
    ]

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
    if raw_mppi_eval_mean:
        ax[0].errorbar(
            raw_mppi_eval_iters,
            raw_mppi_eval_mean,
            yerr=raw_mppi_eval_std,
            marker="^",
            linestyle="--",
            label="Raw MPPI eval on policy seeds",
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
        ax[1].plot(iters, s_track, marker="s", label="S_track (λ_track·Σ‖a−π‖²)")
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
    plt.close(fig)
    print(f"saved {out}")


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


def _wrap_angles(qpos: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(qpos), np.cos(qpos))


def _policy_action(policy: DeterministicPolicy, obs: np.ndarray) -> np.ndarray:
    device = next(policy.parameters()).device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = policy.forward(obs_t).squeeze(0).cpu().numpy()
    return action


def _rollout_policy_trace(
    policy: DeterministicPolicy,
    episode_len: int,
    seed: int,
) -> dict[str, np.ndarray]:
    env = Acrobot()
    policy.eval()
    try:
        np.random.seed(seed)
        env.reset()

        qpos = [env.data.qpos.copy()]
        actions: list[np.ndarray] = []
        costs: list[float] = []
        tip_dist = [env.task_metrics()["tip_dist"]]
        success: list[bool] = []

        for _ in range(episode_len):
            action = _policy_action(policy, env._get_obs())
            _, cost, done, _ = env.step(action)
            metrics = env.task_metrics()

            qpos.append(env.data.qpos.copy())
            actions.append(action.copy())
            costs.append(float(cost))
            tip_dist.append(metrics["tip_dist"])
            success.append(metrics["success"])

            if done:
                break
    finally:
        env.close()

    return {
        "qpos": np.asarray(qpos, dtype=float),
        "actions": np.asarray(actions, dtype=float),
        "costs": np.asarray(costs, dtype=float),
        "tip_dist": np.asarray(tip_dist, dtype=float),
        "success": np.asarray(success, dtype=bool),
    }


def _rollout_mppi_trace(
    mppi_cfg: MPPIConfig,
    episode_len: int,
    seed: int,
    prior=None,
    coupling=None,
) -> dict[str, np.ndarray]:
    env = Acrobot()
    mppi = MPPI(env, mppi_cfg)
    try:
        np.random.seed(seed)
        env.reset()
        mppi.reset()

        qpos = [env.data.qpos.copy()]
        actions: list[np.ndarray] = []
        costs: list[float] = []
        tip_dist = [env.task_metrics()["tip_dist"]]
        success: list[bool] = []

        for _ in range(episode_len):
            action, _ = mppi.plan_step(
                env.get_state(),
                prior_cost=prior,
                coupling=coupling,
            )
            _, cost, done, _ = env.step(action)
            metrics = env.task_metrics()

            qpos.append(env.data.qpos.copy())
            actions.append(action.copy())
            costs.append(float(cost))
            tip_dist.append(metrics["tip_dist"])
            success.append(metrics["success"])

            if done:
                break
    finally:
        env.close()

    return {
        "qpos": np.asarray(qpos, dtype=float),
        "actions": np.asarray(actions, dtype=float),
        "costs": np.asarray(costs, dtype=float),
        "tip_dist": np.asarray(tip_dist, dtype=float),
        "success": np.asarray(success, dtype=bool),
    }


def _plot_policy_trust(run_dir: Path) -> float:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return 1.0
    rows = [json.loads(l) for l in open(metrics_path)]
    if not rows:
        return 1.0
    last = rows[-1]
    return float(last.get("policy_trust_next", last.get("policy_trust", 1.0)))


def _make_collection_bias_for_plot(
    policy: DeterministicPolicy,
    gps_cfg: GPSConfig,
    policy_trust: float,
):
    if gps_cfg.coupling_mode == "raw":
        return None, None

    lambda_track = gps_cfg.lambda_policy_track * policy_trust
    coupling_beta = gps_cfg.policy_coupling_beta * policy_trust
    keep_fraction = 1.0 - policy_trust * (1.0 - gps_cfg.policy_coupling_keep_fraction)

    prior = None
    if gps_cfg.coupling_mode in {"cost", "hybrid"}:
        prior = make_policy_tracking_prior(
            policy,
            lambda_track=lambda_track,
        )

    coupling = None
    if gps_cfg.coupling_mode in {"filter", "hard_filter", "hybrid"}:
        coupling = make_policy_filter_coupling(
            policy,
            beta=coupling_beta,
            cost_slack_rel=gps_cfg.policy_coupling_cost_slack_rel,
            cost_slack_abs=gps_cfg.policy_coupling_cost_slack_abs,
            min_fraction=gps_cfg.policy_coupling_min_fraction,
            keep_fraction=keep_fraction,
            min_n_eff=gps_cfg.policy_coupling_min_n_eff,
            max_weight=gps_cfg.policy_coupling_max_weight,
            hard_filter=gps_cfg.coupling_mode == "hard_filter",
        )

    return prior, coupling


def _max_true_run(values: np.ndarray) -> int:
    best = 0
    run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _trace_title(name: str, trace: dict[str, np.ndarray]) -> str:
    costs = trace["costs"]
    cost_per_step = float(np.sum(costs) / max(len(costs), 1))
    success = trace["success"]
    hit_steps = np.flatnonzero(success)
    hit = f"hit t={int(hit_steps[0])}" if len(hit_steps) else "no hit"
    hold = _max_true_run(success)
    final_dist = float(trace["tip_dist"][-1])
    return (
        f"{name}\n"
        f"cost/step={cost_per_step:.2f}, {hit}, max hold={hold}, final dist={final_dist:.2f}"
    )


def _plot_trace_column(
    axes: np.ndarray,
    trace: dict[str, np.ndarray],
    title: str,
) -> None:
    q_steps = np.arange(trace["qpos"].shape[0])
    q_deg = np.rad2deg(_wrap_angles(trace["qpos"]))
    axes[0].plot(q_steps, q_deg[:, 0], label="joint 0")
    axes[0].plot(q_steps, q_deg[:, 1], label="joint 1")
    axes[0].axhline(0.0, color="k", linewidth=0.8, alpha=0.35)
    axes[0].set_title(title)
    axes[0].set_ylabel("joint angle (deg)")
    axes[0].grid(alpha=0.3)

    a_steps = np.arange(trace["actions"].shape[0])
    if len(a_steps):
        axes[1].plot(a_steps, trace["actions"][:, 0], color="C3", label="torque")
    axes[1].set_xlabel("control step")
    axes[1].set_ylabel("action")
    axes[1].grid(alpha=0.3)


def plot_trace_comparison(
    run_dir: Path,
    gps_cfg: GPSConfig,
    episode_len: int,
    seed: int,
) -> None:
    """Figure-3-style rollout trace: raw MPPI vs biased MPPI vs policy."""
    ckpt_path = _select_policy_checkpoint(run_dir)
    if ckpt_path is None:
        print(f"no policy checkpoint in {run_dir}; skipping trajectory comparison")
        return

    policy_cfg = PolicyConfig()
    mppi_cfg = MPPIConfig.load("acrobot")
    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, policy_cfg)
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    policy.eval()

    print(
        f"rolling out trace comparison for {episode_len} steps "
        f"from seed {seed}: raw MPPI vs policy-biased MPPI vs {ckpt_path.name}"
    )
    mppi_trace = _rollout_mppi_trace(mppi_cfg, episode_len=episode_len, seed=seed)
    policy_trust = _plot_policy_trust(run_dir)
    prior, coupling = _make_collection_bias_for_plot(policy, gps_cfg, policy_trust)
    biased_trace = _rollout_mppi_trace(
        mppi_cfg,
        episode_len=episode_len,
        seed=seed,
        prior=prior,
        coupling=coupling,
    )
    policy_trace = _rollout_policy_trace(policy, episode_len=episode_len, seed=seed)

    fig, ax = plt.subplots(2, 3, figsize=(17, 7), sharex="col")
    _plot_trace_column(ax[:, 0], mppi_trace, _trace_title("Raw MPPI", mppi_trace))
    _plot_trace_column(
        ax[:, 1],
        biased_trace,
        _trace_title(f"Policy-Biased MPPI (trust={policy_trust:.2f})", biased_trace),
    )
    _plot_trace_column(ax[:, 2], policy_trace, _trace_title("Policy", policy_trace))
    ax[0, 0].legend(loc="upper right")
    ax[1, 0].legend(loc="upper right")
    fig.suptitle("Acrobot rollout trace comparison (same initial state)", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    suffix = "latest" if ckpt_path.name == "checkpoint_latest.pt" else ckpt_path.stem.split("_")[-1]
    out = run_dir / f"trajectory_compare_{suffix}.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def render_checkpoints(
    run_dir: Path,
    gps_cfg: GPSConfig,
    n_eps: int,
    episode_len: int,
    seed: int,
) -> None:
    policy_cfg = PolicyConfig()

    milestones = sorted(run_dir.glob("checkpoint_iter_*.pt"))
    if not milestones:
        print(f"no milestone checkpoints in {run_dir}")
        return

    env = Acrobot()
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
    trace_len: int | None = None,
    seed: int = 0,
    no_traces: bool = False,
    no_videos: bool = False,
) -> None:
    """Visualise a completed GPS run.

    Args:
        run_name: directory name under runs/
        n_eps: number of rollout episodes per checkpoint video
        ep_len: episode length for rendered videos (defaults to gps_cfg.eval_episode_len)
        trace_len: trajectory comparison length (defaults to min(eval_episode_len, 300))
        seed: eval seed
        no_traces: skip raw-MPPI-vs-policy trajectory comparison
        no_videos: skip checkpoint video rendering
    """
    run_dir = Path("runs") / run_name
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    gps_cfg = GPSConfig.load("acrobot")
    plot_metrics(run_dir, gps_cfg)
    if not no_traces:
        plot_trace_comparison(
            run_dir,
            gps_cfg,
            episode_len=trace_len if trace_len is not None else min(gps_cfg.eval_episode_len, 300),
            seed=seed,
        )
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
