"""MPPI+GPS — Mordatch Fig 3 reproduction (Acrobot swing-up).

Per-iter loop:
  1. (iter 0 only) eval vanilla MPPI on a fixed seeded eval set → flat baseline.
  2. Collect MPPI rollouts (prior=None on iter 0, else policy-tracking prior).
  3. BC-train π on the collected (obs, act) pairs.
  4. Eval MPPI-with-prior and π-only on the same fixed seeded eval set.
  5. Append metrics; checkpoint π.

At the end, plot fig3.png with three curves: vanilla MPPI baseline (axhline),
MPPI+π prior, and π only.

Run (sweep):
    python -m experiments.exp_gps_mordatch_fig3 --lambda-track 1.0 --run-name fig3_lambda_1
    python -m experiments.exp_gps_mordatch_fig3 --lambda-track 3.0 --run-name fig3_lambda_3
    python -m experiments.exp_gps_mordatch_fig3 --lambda-track 7.0 --run-name fig3_lambda_7
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tyro

from src.envs.acrobot import Acrobot
from src.gps.prior import make_policy_tracking_prior
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig
from src.utils.eval import evaluate_policy

SUCCESS_TIP_Z = 3.5
EXPERIMENTS_RESULTS = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# lifted from scripts/gps_train.py (verbatim) so the experiment is self-contained
# ---------------------------------------------------------------------------
def collect_episodes(
    env: Acrobot,
    mppi: MPPI,
    n_episodes: int,
    steps_per_episode: int,
    prior=None,
    seed_base: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    obs_list, act_list, ep_costs = [], [], []
    stat_keys = ('cost_env_mean', 'cost_is_mean', 'cost_is_std',
                 'cost_track_mean', 'cost_s_mean', 'n_eff', 'lam')
    stat_sums = {k: 0.0 for k in stat_keys}
    n_calls = 0

    for ep in range(n_episodes):
        np.random.seed(seed_base + ep)
        env.reset()
        mppi.reset()

        ep_cost = 0.0
        for _ in range(steps_per_episode):
            state = env.get_state()
            obs = env._get_obs()
            action, info = mppi.plan_step(state, prior_cost=prior)
            for k in stat_keys:
                stat_sums[k] += info[k]
            n_calls += 1
            obs_list.append(obs)
            act_list.append(action)
            _, cost, done, _ = env.step(action)
            ep_cost += cost
            if done:
                break
        ep_costs.append(ep_cost)

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.float32)
    mppi_stats = {k: stat_sums[k] / max(n_calls, 1) for k in stat_keys}
    return obs_arr, act_arr, float(np.mean(ep_costs)), mppi_stats


def train_policy(
    policy: DeterministicPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    policy.train()
    obs_b = torch.as_tensor(obs, dtype=torch.float32, device="cuda")
    act_b = torch.as_tensor(actions, dtype=torch.float32, device="cuda")
    N = len(obs)
    recent: list[float] = []

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


# ---------------------------------------------------------------------------
# new helpers
# ---------------------------------------------------------------------------
def evaluate_mppi(
    env: Acrobot,
    mppi: MPPI,
    prior,
    n_episodes: int,
    episode_len: int,
    seed: int,
) -> dict:
    """π-free analog of evaluate_policy: drive the env with mppi.plan_step.

    Seeding np.random before env.reset() pins both the env's initial state
    sampling AND MPPI's ε noise sampling, so eval is reproducible across iters
    at the same seed. The only thing that varies iter-to-iter is `prior`.
    """
    returns: list[float] = []
    max_tip_z: list[float] = []
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()
        mppi.reset()
        ep_cost = 0.0
        ep_max_z = -np.inf
        for _ in range(episode_len):
            state = env.get_state()
            action, _ = mppi.plan_step(state, prior_cost=prior)
            _, cost, done, _ = env.step(action)
            ep_cost += cost
            tip_z = float(env.data.sensordata[2])
            if tip_z > ep_max_z:
                ep_max_z = tip_z
            if done:
                break
        returns.append(ep_cost)
        max_tip_z.append(ep_max_z)
    arr = np.array(returns)
    z_arr = np.array(max_tip_z)
    return {
        "mean_cost": float(arr.mean()),
        "std_cost": float(arr.std()),
        "per_ep": arr.tolist(),
        "max_tip_z_per_ep": z_arr.tolist(),
        "success_rate": float((z_arr >= SUCCESS_TIP_Z).mean()),
    }


def mppi_sanity_check(
    env: Acrobot,
    mppi: MPPI,
    n_episodes: int,
    episode_len: int,
    seed: int,
    abort_threshold: float = 0.5,
) -> dict:
    """Vanilla MPPI on n seeded episodes. Abort the run if success_rate < threshold.

    Without a working MPPI baseline the GPS loop has nothing to converge to, so
    cheap upfront verification is worth ~10 min.
    """
    print(f"[sanity] running vanilla MPPI on {n_episodes} eps × {episode_len} steps...")
    t0 = time.time()
    result = evaluate_mppi(
        env, mppi, prior=None,
        n_episodes=n_episodes, episode_len=episode_len, seed=seed,
    )
    wall = time.time() - t0
    print(
        f"[sanity] success_rate={result['success_rate']:.2f}  "
        f"mean_cost={result['mean_cost']:.1f}  "
        f"cost_per_step={result['mean_cost']/episode_len:.3f}  "
        f"max_tip_z per ep: {[f'{z:.2f}' for z in result['max_tip_z_per_ep']]}  "
        f"wall={wall:.1f}s"
    )
    if result["success_rate"] < abort_threshold:
        raise RuntimeError(
            f"vanilla MPPI success_rate {result['success_rate']:.2f} < "
            f"{abort_threshold}; the GPS loop's ceiling is MPPI itself, so "
            f"running the full experiment is meaningless. Tune MPPI first."
        )
    return result


def plot_fig3(
    metrics_path: Path,
    out_png: Path,
    vanilla_baseline_per_step: float,
    eval_episode_len: int,
    n_eval_episodes: int,
    lambda_track: float,
    H: int,
    K: int,
) -> None:
    """Mordatch Fig 3 style: cost-per-step vs GPS iter, three curves."""
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    iters = np.array([r["iter"] for r in records])

    def col(key: str) -> np.ndarray:
        return np.array([r[key] if r[key] is not None else np.nan for r in records], dtype=float)

    prior_mean = col("eval_mppi_prior_mean") / eval_episode_len
    prior_std = col("eval_mppi_prior_std") / eval_episode_len
    pi_mean = col("eval_policy_only_mean") / eval_episode_len
    pi_std = col("eval_policy_only_std") / eval_episode_len
    sem_scale = 1.0 / np.sqrt(n_eval_episodes)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(
        vanilla_baseline_per_step,
        linestyle="--", color="black", linewidth=1.5,
        label="MPPI (vanilla, no prior)",
    )

    valid_prior = ~np.isnan(prior_mean)
    if valid_prior.any():
        ax.plot(iters[valid_prior], prior_mean[valid_prior],
                color="tab:blue", linewidth=1.8,
                label="MPPI + π prior (trajectory cost)")
        ax.fill_between(
            iters[valid_prior],
            prior_mean[valid_prior] - prior_std[valid_prior] * sem_scale,
            prior_mean[valid_prior] + prior_std[valid_prior] * sem_scale,
            color="tab:blue", alpha=0.18,
        )

    valid_pi = ~np.isnan(pi_mean)
    if valid_pi.any():
        ax.plot(iters[valid_pi], pi_mean[valid_pi],
                color="tab:orange", linewidth=1.8,
                label="π only (policy cost)")
        ax.fill_between(
            iters[valid_pi],
            pi_mean[valid_pi] - pi_std[valid_pi] * sem_scale,
            pi_mean[valid_pi] + pi_std[valid_pi] * sem_scale,
            color="tab:orange", alpha=0.18,
        )

    ax.set_xlabel("GPS iteration")
    ax.set_ylabel("mean cost per step")
    ax.set_title("GPS on Acrobot swing-up — Mordatch Fig 3 style")
    ax.text(
        0.98, 0.97,
        f"λ_track={lambda_track:g}, H={H}, K={K}, N_eval={n_eval_episodes}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"),
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(
    run_name: str | None = None,
    lambda_track: float = 3.0,
    n_iters: int = 25,
    n_eval_episodes: int = 16,
    eval_episode_len: int = 1000,
    eval_seed: int = 7,
    skip_sanity: bool = False,
    seed: int = 0,
) -> None:
    assert torch.cuda.is_available(), "CUDA required (policy + prior live on GPU)"

    if run_name is None:
        run_name = f"fig3_lambda_{lambda_track:g}"
    run_dir = EXPERIMENTS_RESULTS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    fig_path = run_dir / "fig3.png"

    mppi_cfg = MPPIConfig.load("acrobot")
    gps_cfg = GPSConfig.load("acrobot")
    policy_cfg = PolicyConfig()

    assert mppi_cfg.adaptive_lam is False, (
        "adaptive_lam=True drifts λ_mppi iter-to-iter; β=λ_track/λ_mppi becomes "
        "uninterpretable. Disable it before running this experiment."
    )

    print(f"run_dir: {run_dir}")
    print(f"mppi_cfg: {mppi_cfg}")
    print(f"gps_cfg: {gps_cfg}  [overrides: n_iters={n_iters}, λ_track={lambda_track}]")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = Acrobot(use_warp=False, nworld=mppi_cfg.K)
    mppi = MPPI(env, mppi_cfg)
    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, policy_cfg).to("cuda")

    # ---- 1. sanity check vanilla MPPI ----
    sanity = None
    if not skip_sanity:
        sanity = mppi_sanity_check(
            env, mppi,
            n_episodes=n_eval_episodes,
            episode_len=eval_episode_len,
            seed=eval_seed,
        )

    # ---- 2. one-shot vanilla baseline (cached for the dashed reference line) ----
    if sanity is not None:
        # sanity already ran on (eval_seed, n_eval_episodes, eval_episode_len) with prior=None
        vanilla_eval = sanity
    else:
        print("[baseline] computing vanilla MPPI eval...")
        vanilla_eval = evaluate_mppi(
            env, mppi, prior=None,
            n_episodes=n_eval_episodes,
            episode_len=eval_episode_len,
            seed=eval_seed,
        )
    vanilla_baseline_per_step = vanilla_eval["mean_cost"] / eval_episode_len
    print(
        f"[baseline] vanilla MPPI cost/step={vanilla_baseline_per_step:.4f}  "
        f"success_rate={vanilla_eval['success_rate']:.2f}"
    )

    # ---- 3. dump config (now, so a crashed run still has reproducibility info) ----
    config_dump = {
        "run_name": run_name,
        "git_sha": _git_sha(),
        "cli": {
            "lambda_track": lambda_track,
            "n_iters": n_iters,
            "n_eval_episodes": n_eval_episodes,
            "eval_episode_len": eval_episode_len,
            "eval_seed": eval_seed,
            "skip_sanity": skip_sanity,
            "seed": seed,
        },
        "mppi_cfg": asdict(mppi_cfg),
        "gps_cfg": asdict(gps_cfg),
        "policy_cfg": asdict(policy_cfg),
        "vanilla_baseline_per_step": vanilla_baseline_per_step,
        "vanilla_baseline_success_rate": vanilla_eval["success_rate"],
        "vanilla_baseline_mean_cost": vanilla_eval["mean_cost"],
        "vanilla_baseline_std_cost": vanilla_eval["std_cost"],
    }
    config_path.write_text(json.dumps(config_dump, indent=2))

    # ---- 4. GPS loop ----
    for it in range(n_iters):
        t_start = time.time()

        # iter 0: vanilla MPPI to bootstrap π. iter ≥1: policy-tracking prior.
        prior = None if it == 0 else make_policy_tracking_prior(policy, lambda_track)

        seed_base = 10_000 + it * gps_cfg.episodes_per_iter
        print(f"[iter {it:3d}] collecting...")
        obs, acts, mppi_train_cost, mppi_stats = collect_episodes(
            env, mppi,
            n_episodes=gps_cfg.episodes_per_iter,
            steps_per_episode=gps_cfg.steps_per_episode,
            prior=prior,
            seed_base=seed_base,
        )

        print(f"[iter {it:3d}] training policy on {len(obs)} pairs...")
        bc_loss = train_policy(
            policy, obs, acts,
            batch_size=gps_cfg.batch_size,
            rng=rng,
        )

        # MPPI-with-prior eval. Iter 0 has no trained π; record None on that iter.
        if it == 0:
            eval_prior = None
        else:
            print(f"[iter {it:3d}] evaluating MPPI+prior...")
            eval_prior = evaluate_mppi(
                env, mppi,
                prior=make_policy_tracking_prior(policy, lambda_track),
                n_episodes=n_eval_episodes,
                episode_len=eval_episode_len,
                seed=eval_seed,
            )

        print(f"[iter {it:3d}] evaluating π-only...")
        eval_pi = evaluate_policy(
            policy, env,
            n_episodes=n_eval_episodes,
            episode_len=eval_episode_len,
            seed=eval_seed,
        )

        record = {
            "iter": it,
            "wall_time_s": time.time() - t_start,
            "lambda_track": lambda_track if it > 0 else 0.0,
            "n_pairs_this_iter": len(obs),
            "bc_loss_final": bc_loss,
            "mppi_train_mean_cost": mppi_train_cost,
            "mppi_train_cost_per_step": mppi_train_cost / gps_cfg.steps_per_episode,
            "eval_mppi_prior_mean": eval_prior["mean_cost"] if eval_prior else None,
            "eval_mppi_prior_std":  eval_prior["std_cost"]  if eval_prior else None,
            "eval_mppi_prior_per_ep": eval_prior["per_ep"] if eval_prior else None,
            "eval_mppi_prior_success_rate": eval_prior["success_rate"] if eval_prior else None,
            "eval_policy_only_mean": eval_pi["mean_cost"],
            "eval_policy_only_std":  eval_pi["std_cost"],
            "eval_policy_only_per_ep": eval_pi["per_ep"],
            "mppi_S_env_mean":   mppi_stats["cost_env_mean"],
            "mppi_S_track_mean": mppi_stats["cost_track_mean"],
            "mppi_S_total_mean": mppi_stats["cost_s_mean"],
            "mppi_n_eff":        mppi_stats["n_eff"],
            "mppi_lam_effective": mppi_stats["lam"],
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        prior_str = "—" if eval_prior is None else f"{eval_prior['mean_cost']/eval_episode_len:.4f}"
        pi_str = f"{eval_pi['mean_cost']/eval_episode_len:.4f}"
        print(
            f"[iter {it:3d}] done  bc_loss={bc_loss:.5f}  "
            f"vanilla_cps={vanilla_baseline_per_step:.4f}  "
            f"prior_cps={prior_str}  pi_cps={pi_str}  "
            f"wall={record['wall_time_s']:.1f}s"
        )

        torch.save(policy.state_dict(), run_dir / f"checkpoint_iter_{it:03d}.pt")
        torch.save(policy.state_dict(), run_dir / "checkpoint_latest.pt")

        # incremental plot — useful for monitoring long sweeps
        try:
            plot_fig3(
                metrics_path, fig_path, vanilla_baseline_per_step,
                eval_episode_len, n_eval_episodes,
                lambda_track, mppi_cfg.H, mppi_cfg.K,
            )
        except Exception as e:
            print(f"[iter {it:3d}] plot failed (will retry next iter): {e}")

    env.close()
    print(f"done. artifacts in {run_dir}")


if __name__ == "__main__":
    tyro.cli(main)
