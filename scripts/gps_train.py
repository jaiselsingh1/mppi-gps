"""GPS outer loop: alternate MPPI data collection with BC policy training.

Iter 0:    collect with vanilla MPPI (prior=None)      -> train π from random init
Iter t>=1: collect with policy-biased MPPI (prior set) -> continue training π
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import tyro

import numpy as np
import torch
import torch.nn.functional as F

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import MPPIConfig, PolicyConfig, GPSConfig
from src.utils.eval import evaluate_policy
from src.gps.coupling import make_policy_filter_coupling
from src.gps.prior import make_policy_tracking_prior


def smooth_actions_ema(actions: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA for noisy MPPI action targets within one episode."""
    if alpha <= 0.0 or len(actions) == 0:
        return actions
    out = np.empty_like(actions)
    out[0] = actions[0]
    for t in range(1, len(actions)):
        out[t] = alpha * out[t - 1] + (1.0 - alpha) * actions[t]
    return out


def collect_episodes(
    env: Acrobot,
    mppi: MPPI,
    n_episodes: int,
    steps_per_episode: int,
    prior=None,
    coupling=None,
    seed_base: int = 0,
    hold_steps: int = 25,
    action_ema_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Run MPPI in closed loop.

    Returns (obs, actions, mean_ep_cost, mppi_stats) where mppi_stats averages
    the S-component diagnostics (env / IS / track / total S, plus n_eff, lam)
    over every plan_step call in this iter.
    """
    obs_chunks, act_chunks, ep_costs = [], [], []
    hit_successes: list[bool] = []
    hold_successes: list[bool] = []
    times_to_hit: list[int] = []
    final_tip_dists: list[float] = []
    final_qvel_norms: list[float] = []
    stat_keys = ('cost_env_mean', 'cost_is_mean', 'cost_is_std',
                 'cost_track_mean', 'cost_s_mean', 'n_eff', 'lam',
                 'coupling_active', 'coupling_used_fallback',
                 'coupling_feasible_fraction', 'coupling_policy_cost_mean',
                 'coupling_policy_cost_std', 'coupling_score_mean')
    stat_sums = {k: 0.0 for k in stat_keys}
    n_calls = 0
    action_low, action_high = env.action_bounds

    for ep in range(n_episodes):
        np.random.seed(seed_base + ep)
        env.reset()
        mppi.reset()

        ep_cost = 0.0
        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        first_success_t: int | None = None
        hold_count = 0
        max_hold_count = 0
        for t in range(steps_per_episode):
            state = env.get_state()
            obs = env._get_obs()
            action, info = mppi.plan_step(state, prior_cost=prior, coupling=coupling)
            for k in stat_keys:
                stat_sums[k] += info[k]
            n_calls += 1
            ep_obs.append(obs)
            # MPPI and the simulator see the raw action. BC learns the actuator-
            # bounded command that the policy can actually represent.
            ep_actions.append(np.clip(action, action_low, action_high))
            _, cost, done, _ = env.step(action)
            ep_cost += cost

            metrics = env.task_metrics()
            if metrics["success"]:
                if first_success_t is None:
                    first_success_t = t
                hold_count += 1
            else:
                hold_count = 0
            max_hold_count = max(max_hold_count, hold_count)

            if done:
                break
        ep_costs.append(ep_cost)
        if ep_obs:
            obs_chunks.append(np.asarray(ep_obs, dtype=np.float32))
            ep_act_arr = np.asarray(ep_actions, dtype=np.float32)
            act_chunks.append(smooth_actions_ema(ep_act_arr, action_ema_alpha))
        final_metrics = env.task_metrics()
        hit_successes.append(first_success_t is not None)
        hold_successes.append(max_hold_count >= hold_steps)
        times_to_hit.append(first_success_t if first_success_t is not None else steps_per_episode)
        final_tip_dists.append(final_metrics["tip_dist"])
        final_qvel_norms.append(final_metrics["qvel_norm"])

    obs_arr = np.concatenate(obs_chunks, axis=0) if obs_chunks else np.empty((0, 4), dtype=np.float32)
    act_arr = np.concatenate(act_chunks, axis=0) if act_chunks else np.empty((0, env.action_dim), dtype=np.float32)
    mppi_stats = {k: stat_sums[k] / max(n_calls, 1) for k in stat_keys}
    mppi_stats.update({
        "hit_success_rate": float(np.mean(hit_successes)),
        "hold_success_rate": float(np.mean(hold_successes)),
        "mean_time_to_hit": float(np.mean(times_to_hit)),
        "mean_final_tip_dist": float(np.mean(final_tip_dists)),
        "mean_final_qvel_norm": float(np.mean(final_qvel_norms)),
    })
    return obs_arr, act_arr, float(np.mean(ep_costs)), mppi_stats


def train_policy(
    policy: DeterministicPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    epochs: int = 1,
) -> float:
    """Adam updates on MSE. Returns trailing-50-step mean loss."""
    policy.train()
    device = next(policy.parameters()).device
    obs_b = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_b = torch.as_tensor(actions, dtype=torch.float32, device=device)
    N = len(obs)
    recent: list[float] = []

    for _ in range(max(1, epochs)):
        perm = rng.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            mu = policy.forward(obs_b[idx])
            loss = F.mse_loss(mu, act_b[idx])
            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            recent.append(loss.item())
            if len(recent) > 50:
                recent.pop(0)
    return float(np.mean(recent))


def evaluate_mppi(
    env: Acrobot,
    mppi: MPPI,
    n_episodes: int,
    episode_len: int,
    seed: int,
    hold_steps: int = 25,
) -> dict:
    """Evaluate raw MPPI on the same seed protocol used for policy eval."""
    returns: list[float] = []
    hit_successes: list[bool] = []
    hold_successes: list[bool] = []
    times_to_hit: list[int] = []
    final_tip_dists: list[float] = []
    final_qvel_norms: list[float] = []

    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()
        mppi.reset()

        ep_cost = 0.0
        first_success_t: int | None = None
        hold_count = 0
        max_hold_count = 0
        for t in range(episode_len):
            state = env.get_state()
            action, _ = mppi.plan_step(state)
            _, cost, done, _ = env.step(action)
            ep_cost += cost

            metrics = env.task_metrics()
            if metrics["success"]:
                if first_success_t is None:
                    first_success_t = t
                hold_count += 1
            else:
                hold_count = 0
            max_hold_count = max(max_hold_count, hold_count)

            if done:
                break

        final_metrics = env.task_metrics()
        returns.append(ep_cost)
        hit_successes.append(first_success_t is not None)
        hold_successes.append(max_hold_count >= hold_steps)
        times_to_hit.append(first_success_t if first_success_t is not None else episode_len)
        final_tip_dists.append(final_metrics["tip_dist"])
        final_qvel_norms.append(final_metrics["qvel_norm"])

    arr = np.asarray(returns, dtype=float)
    return {
        "mean_cost": float(arr.mean()),
        "std_cost": float(arr.std()),
        "hit_success_rate": float(np.mean(hit_successes)),
        "hold_success_rate": float(np.mean(hold_successes)),
        "mean_time_to_hit": float(np.mean(times_to_hit)),
        "mean_final_tip_dist": float(np.mean(final_tip_dists)),
        "mean_final_qvel_norm": float(np.mean(final_qvel_norms)),
    }


def make_collection_bias(
    policy: DeterministicPolicy,
    gps_cfg: GPSConfig,
    it: int,
    policy_trust: float = 1.0,
):
    """Return (prior_cost, coupling) for the current GPS iteration."""
    if it < gps_cfg.coupling_warmup_iters or gps_cfg.coupling_mode == "raw":
        return None, None

    lambda_track = gps_cfg.lambda_policy_track * policy_trust
    coupling_beta = gps_cfg.policy_coupling_beta * policy_trust
    keep_fraction = 1.0 - policy_trust * (1.0 - gps_cfg.policy_coupling_keep_fraction)

    if gps_cfg.coupling_mode == "cost":
        prior = make_policy_tracking_prior(
            policy,
            lambda_track=lambda_track,
        )
        return prior, None

    if gps_cfg.coupling_mode in {"filter", "hard_filter", "hybrid"}:
        prior = None
        if gps_cfg.coupling_mode == "hybrid":
            prior = make_policy_tracking_prior(
                policy,
                lambda_track=lambda_track,
            )
        coupling = make_policy_filter_coupling(
            policy,
            beta=coupling_beta,
            min_fraction=gps_cfg.policy_coupling_min_fraction,
            keep_fraction=keep_fraction,
            min_n_eff=gps_cfg.policy_coupling_min_n_eff,
            max_weight=gps_cfg.policy_coupling_max_weight,
            hard_filter=gps_cfg.coupling_mode == "hard_filter",
        )
        return prior, coupling

    raise ValueError(f"Unknown GPS coupling_mode: {gps_cfg.coupling_mode!r}")


def compute_policy_trust(
    *,
    policy_cost: float | None,
    raw_mppi_cost: float | None,
    eval_episode_len: int,
    gps_cfg: GPSConfig,
) -> float:
    if not gps_cfg.adaptive_policy_trust:
        return gps_cfg.policy_trust_max
    if policy_cost is None or raw_mppi_cost is None:
        return gps_cfg.policy_trust_min

    j_policy = policy_cost / eval_episode_len
    j_mppi = raw_mppi_cost / eval_episode_len
    j_bad = gps_cfg.policy_trust_bad_cost_per_step
    denom = max(j_bad - j_mppi, 1e-8)
    quality = np.clip((j_bad - j_policy) / denom, 0.0, 1.0)
    return float(
        gps_cfg.policy_trust_min
        + (gps_cfg.policy_trust_max - gps_cfg.policy_trust_min) * quality
    )


def _apply_gps_overrides(gps_cfg: GPSConfig, **overrides) -> None:
    for key, value in overrides.items():
        if value is not None:
            setattr(gps_cfg, key, value)


def main(
    run_name: str | None = None,
    use_warp: bool = False,
    n_gps_iters: int | None = None,
    episodes_per_iter: int | None = None,
    steps_per_episode: int | None = None,
    eval_every: int | None = None,
    eval_n_episodes: int | None = None,
    eval_episode_len: int | None = None,
    eval_mppi_baseline_episodes: int | None = None,
    coupling_warmup_iters: int | None = None,
    coupling_mode: str | None = None,
    lambda_policy_track: float | None = None,
    adaptive_policy_trust: bool | None = None,
    policy_trust_bad_cost_per_step: float | None = None,
    policy_trust_min: float | None = None,
    policy_trust_max: float | None = None,
    policy_coupling_beta: float | None = None,
    policy_coupling_cost_slack_rel: float | None = None,
    policy_coupling_keep_fraction: float | None = None,
) -> None:
    gps_cfg = GPSConfig.load("acrobot")
    _apply_gps_overrides(
        gps_cfg,
        n_gps_iters=n_gps_iters,
        episodes_per_iter=episodes_per_iter,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_n_episodes=eval_n_episodes,
        eval_episode_len=eval_episode_len,
        eval_mppi_baseline_episodes=eval_mppi_baseline_episodes,
        coupling_warmup_iters=coupling_warmup_iters,
        coupling_mode=coupling_mode,
        lambda_policy_track=lambda_policy_track,
        adaptive_policy_trust=adaptive_policy_trust,
        policy_trust_bad_cost_per_step=policy_trust_bad_cost_per_step,
        policy_trust_min=policy_trust_min,
        policy_trust_max=policy_trust_max,
        policy_coupling_beta=policy_coupling_beta,
        policy_coupling_cost_slack_rel=policy_coupling_cost_slack_rel,
        policy_coupling_keep_fraction=policy_coupling_keep_fraction,
    )
    mppi_cfg = MPPIConfig.load("acrobot")
    policy_cfg = PolicyConfig()

    if run_name is None:
        suffix = "_warp" if use_warp else ""
        if gps_cfg.coupling_mode == "hybrid":
            run_name = (
                f"gps_hybrid_track_{gps_cfg.lambda_policy_track:g}"
                f"_filter_{gps_cfg.policy_coupling_beta:g}{suffix}"
            )
        elif gps_cfg.coupling_mode == "filter":
            run_name = f"gps_filter_beta_{gps_cfg.policy_coupling_beta:g}{suffix}"
        elif gps_cfg.coupling_mode == "raw":
            run_name = f"gps_raw{suffix}"
        else:
            run_name = f"gps_lambda_{gps_cfg.lambda_policy_track:g}{suffix}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    print(f"run_dir: {run_dir}")
    print(f"gps_cfg: {gps_cfg}")
    print(f"use_warp: {use_warp}  nworld: {mppi_cfg.K}")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    env = Acrobot(use_warp=use_warp, nworld=mppi_cfg.K)
    mppi = MPPI(env, mppi_cfg)
    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, policy_cfg).to(device="cuda")
    replay_obs: np.ndarray | None = None
    replay_acts: np.ndarray | None = None
    policy_trust = (
        gps_cfg.policy_trust_max
        if not gps_cfg.adaptive_policy_trust
        else gps_cfg.policy_trust_min
    )

    for it in range(gps_cfg.n_gps_iters):
        t_start = time.time()

        prior, coupling = make_collection_bias(policy, gps_cfg, it, policy_trust=policy_trust)
        seed_base = 10_000 + it * gps_cfg.episodes_per_iter
        print("collecting demos")
        obs, acts, mppi_cost, mppi_stats = collect_episodes(
            env, mppi,
            n_episodes=gps_cfg.episodes_per_iter,
            steps_per_episode=gps_cfg.steps_per_episode,
            prior=prior,
            coupling=coupling,
            seed_base=seed_base,
            action_ema_alpha=gps_cfg.action_ema_alpha,
        )

        if gps_cfg.replay_max_pairs > 0:
            replay_obs = obs if replay_obs is None else np.concatenate([replay_obs, obs], axis=0)
            replay_acts = acts if replay_acts is None else np.concatenate([replay_acts, acts], axis=0)
            if len(replay_obs) > gps_cfg.replay_max_pairs:
                replay_obs = replay_obs[-gps_cfg.replay_max_pairs:]
                replay_acts = replay_acts[-gps_cfg.replay_max_pairs:]
            train_obs, train_acts = replay_obs, replay_acts
        else:
            train_obs, train_acts = obs, acts

        print("training policy")
        bc_loss = train_policy(
            policy, train_obs, train_acts,
            batch_size=gps_cfg.batch_size,
            rng=rng,
            epochs=gps_cfg.bc_epochs_per_iter,
        )

        do_eval = (it % gps_cfg.eval_every == 0) or (it == gps_cfg.n_gps_iters - 1)
        if do_eval:
            print("evaluating policy")
            stats = evaluate_policy(
                policy, env,
                n_episodes=gps_cfg.eval_n_episodes,
                episode_len=gps_cfg.eval_episode_len,
                seed=0,
            )
            eval_mean, eval_std = stats["mean_cost"], stats["std_cost"]
            eval_mean_ps = eval_mean / gps_cfg.eval_episode_len
            eval_std_ps = eval_std / gps_cfg.eval_episode_len
            eval_hit_success = stats.get("hit_success_rate")
            eval_hold_success = stats.get("hold_success_rate")
            eval_time_to_hit = stats.get("mean_time_to_hit")
            eval_final_tip_dist = stats.get("mean_final_tip_dist")
            eval_final_qvel_norm = stats.get("mean_final_qvel_norm")
            if gps_cfg.eval_mppi_baseline_episodes > 0:
                print("evaluating raw MPPI baseline")
                mppi_eval_stats = evaluate_mppi(
                    env,
                    mppi,
                    n_episodes=gps_cfg.eval_mppi_baseline_episodes,
                    episode_len=gps_cfg.eval_episode_len,
                    seed=0,
                )
                mppi_eval_mean = mppi_eval_stats["mean_cost"]
                mppi_eval_std = mppi_eval_stats["std_cost"]
                mppi_eval_mean_ps = mppi_eval_mean / gps_cfg.eval_episode_len
                mppi_eval_std_ps = mppi_eval_std / gps_cfg.eval_episode_len
                mppi_eval_hit_success = mppi_eval_stats["hit_success_rate"]
                mppi_eval_hold_success = mppi_eval_stats["hold_success_rate"]
            else:
                mppi_eval_mean = mppi_eval_std = mppi_eval_mean_ps = mppi_eval_std_ps = None
                mppi_eval_hit_success = mppi_eval_hold_success = None
            policy_trust_next = compute_policy_trust(
                policy_cost=eval_mean,
                raw_mppi_cost=mppi_eval_mean,
                eval_episode_len=gps_cfg.eval_episode_len,
                gps_cfg=gps_cfg,
            )
        else:
            eval_mean = eval_std = eval_mean_ps = eval_std_ps = None
            eval_hit_success = eval_hold_success = eval_time_to_hit = None
            eval_final_tip_dist = eval_final_qvel_norm = None
            mppi_eval_mean = mppi_eval_std = mppi_eval_mean_ps = mppi_eval_std_ps = None
            mppi_eval_hit_success = mppi_eval_hold_success = None
            policy_trust_next = policy_trust

        record = {
            "iter": it,
            "mppi_rollout_mean_cost": mppi_cost,
            "mppi_cost_per_step": mppi_cost / gps_cfg.steps_per_episode,
            "bc_loss_final": bc_loss,
            "policy_eval_mean_cost": eval_mean,
            "policy_eval_std_cost": eval_std,
            "policy_eval_cost_per_step_mean": eval_mean_ps,
            "policy_eval_cost_per_step_std": eval_std_ps,
            "mppi_ep_len": gps_cfg.steps_per_episode,
            "eval_ep_len": gps_cfg.eval_episode_len,
            "n_pairs_this_iter": len(obs),
            "n_pairs_train": len(train_obs),
            "bc_epochs_per_iter": gps_cfg.bc_epochs_per_iter,
            "replay_max_pairs": gps_cfg.replay_max_pairs,
            "action_ema_alpha": gps_cfg.action_ema_alpha,
            "wall_time_s": time.time() - t_start,
            "coupling_mode": gps_cfg.coupling_mode if (prior is not None or coupling is not None) else "raw",
            "policy_trust": policy_trust,
            "policy_trust_next": policy_trust_next,
            "lambda_track": gps_cfg.lambda_policy_track * policy_trust if (it > 0 and prior is not None) else 0.0,
            "policy_coupling_beta": gps_cfg.policy_coupling_beta * policy_trust if (it > 0 and coupling is not None) else 0.0,
            "policy_coupling_keep_fraction_effective": (
                1.0 - policy_trust * (1.0 - gps_cfg.policy_coupling_keep_fraction)
                if (it > 0 and coupling is not None) else 1.0
            ),
            "lambda_track_base": gps_cfg.lambda_policy_track,
            "policy_coupling_beta_base": gps_cfg.policy_coupling_beta,
            "raw_mppi_eval_mean_cost": mppi_eval_mean,
            "raw_mppi_eval_std_cost": mppi_eval_std,
            "raw_mppi_eval_cost_per_step_mean": mppi_eval_mean_ps,
            "raw_mppi_eval_cost_per_step_std": mppi_eval_std_ps,
            "raw_mppi_eval_hit_success_rate": mppi_eval_hit_success,
            "raw_mppi_eval_hold_success_rate": mppi_eval_hold_success,
            "mppi_hit_success_rate": mppi_stats["hit_success_rate"],
            "mppi_hold_success_rate": mppi_stats["hold_success_rate"],
            "mppi_mean_time_to_hit": mppi_stats["mean_time_to_hit"],
            "mppi_mean_final_tip_dist": mppi_stats["mean_final_tip_dist"],
            "mppi_mean_final_qvel_norm": mppi_stats["mean_final_qvel_norm"],
            "policy_eval_hit_success_rate": eval_hit_success,
            "policy_eval_hold_success_rate": eval_hold_success,
            "policy_eval_mean_time_to_hit": eval_time_to_hit,
            "policy_eval_mean_final_tip_dist": eval_final_tip_dist,
            "policy_eval_mean_final_qvel_norm": eval_final_qvel_norm,
            # per-iter means over every MPPI plan_step call in this iter
            "mppi_S_env_mean":    mppi_stats["cost_env_mean"],
            "mppi_S_is_mean":     mppi_stats["cost_is_mean"],
            "mppi_S_is_std":      mppi_stats["cost_is_std"],
            "mppi_S_track_mean":  mppi_stats["cost_track_mean"],
            "mppi_S_total_mean":  mppi_stats["cost_s_mean"],
            "mppi_n_eff":         mppi_stats["n_eff"],
            "mppi_lam_effective": mppi_stats["lam"],
            "coupling_active": mppi_stats["coupling_active"],
            "coupling_used_fallback": mppi_stats["coupling_used_fallback"],
            "coupling_feasible_fraction": mppi_stats["coupling_feasible_fraction"],
            "coupling_policy_cost_mean": mppi_stats["coupling_policy_cost_mean"],
            "coupling_policy_cost_std": mppi_stats["coupling_policy_cost_std"],
            "coupling_score_mean": mppi_stats["coupling_score_mean"],
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        eval_str = f"  eval={eval_mean:7.1f}±{eval_std:.1f}" if eval_mean is not None else ""
        mppi_eval_str = (
            f"  raw_mppi_eval={mppi_eval_mean:7.1f}±{mppi_eval_std:.1f}"
            if mppi_eval_mean is not None else ""
        )
        print(
            f"iter {it:3d}  mppi_cost={mppi_cost:7.1f}  "
            f"bc_loss={bc_loss:.5f}  trust={policy_trust:.3f}->{policy_trust_next:.3f}"
            f"{eval_str}{mppi_eval_str}  "
            f"wall={record['wall_time_s']:.1f}s"
        )
        policy_trust = policy_trust_next

        torch.save(policy.state_dict(), run_dir / "checkpoint_latest.pt")
        if it % 5 == 0 or it == gps_cfg.n_gps_iters - 1:
            torch.save(policy.state_dict(), run_dir / f"checkpoint_iter_{it:03d}.pt")

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
