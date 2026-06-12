"""GPS outer loop: alternate MPPI data collection with BC policy training.

Iter 0:    collect with vanilla MPPI (prior=None)      -> train π from random init
Iter t>=1: collect with policy-biased MPPI (prior set) -> continue training π
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from collections.abc import Callable
import tyro

import numpy as np
import torch
import torch.nn.functional as F

from src.envs.base import BaseEnv
from src.envs.acrobot import Acrobot
from src.envs.ant_maze import AntMaze
from src.envs.point_mass import PointMass
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import MPPIConfig, PolicyConfig, GPSConfig
from src.utils.eval import evaluate_policy
from src.gps.coupling import make_policy_filter_coupling
from src.gps.merge import make_policy_merge
from src.gps.mix import make_policy_mixer
from src.gps.prior import make_policy_tracking_prior

_ENV_FACTORIES = {
    "acrobot": Acrobot,
    "ant_maze": AntMaze,
    "point_mass": PointMass,
    "walker2d": Walker2d,
}
_COLLECTION_MODES = {"bc", "gps"}
_COUPLING_MODES = {"track", "filter", "merge"}


def _make_env(env_name: str, **kwargs) -> BaseEnv:
    try:
        env_cls = _ENV_FACTORIES[env_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_ENV_FACTORIES))
        raise ValueError(f"Unknown env_name={env_name!r}; supported: {supported}.") from exc
    return env_cls(**kwargs)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _task_metrics(env: BaseEnv) -> dict:
    if not hasattr(env, "task_metrics"):
        return {"success": False, "tip_dist": float("inf"), "qvel_norm": float("inf")}
    metrics = env.task_metrics()
    return {
        **metrics,
        "success": bool(metrics.get("success", False)),
        "tip_dist": float(metrics.get("tip_dist", float("inf"))),
        "qvel_norm": float(metrics.get("qvel_norm", float("inf"))),
    }


def _normalize_env_name(env_name: str) -> str:
    env_name = env_name.strip().lower().replace("-", "_")
    if env_name not in _ENV_FACTORIES:
        supported = ", ".join(sorted(_ENV_FACTORIES))
        raise ValueError(f"Unknown env_name={env_name!r}; supported: {supported}.")
    return env_name


def _normalize_collection_mode(collection_mode: str) -> str:
    collection_mode = collection_mode.strip().lower().replace("-", "_")
    if collection_mode in {"plain_bc", "vanilla_bc"}:
        collection_mode = "bc"
    if collection_mode not in _COLLECTION_MODES:
        supported = ", ".join(sorted(_COLLECTION_MODES))
        raise ValueError(f"Unknown collection_mode={collection_mode!r}; supported: {supported}.")
    return collection_mode



def collect_episodes(
    env: BaseEnv,
    mppi: MPPI,
    n_episodes: int,
    steps_per_episode: int,
    prior=None,
    coupling=None,
    merge=None,
    mixer=None,
    mix_fraction: float = 0.0,
    policy_act=None,
    dagger_fraction: float = 0.0,
    exec_noise_std: float = 0.0,
    exec_noise_ou_tau: float = 0.0,
    seed_base: int = 0,
    hold_steps: int = 25,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Run MPPI in closed loop.

    With dagger_fraction > 0, that share of episodes is PLATO-style: the
    *policy* drives the env while every visited state is labeled with the
    certified planner action. The policy's compounding drift then generates
    exactly the recovery data plain planner-driven collection never visits
    (observed: the distilled walker fell at ~150-280 steps because MPPI demos
    contain no near-fall states). Labels stay task-certified either way.

    Returns (obs, actions, mean_ep_cost, mppi_stats) where mppi_stats averages
    the S-component diagnostics (env / IS / track / total S, plus n_eff, lam)
    over every plan_step call in this iter.
    """
    obs_chunks, act_chunks, ep_costs = [], [], []
    # per-step diagnostics aligned with (obs, acts): chosen merge beta,
    # whether the policy drove, episode index, certificate cost gap —
    # enables label-achievability bins and the cumulative-budget "ratchet"
    # plot offline
    diag_beta, diag_dagger, diag_ep, diag_gap = [], [], [], []
    hit_successes: list[bool] = []
    hold_successes: list[bool] = []
    times_to_hit: list[int] = []
    final_tip_dists: list[float] = []
    final_qvel_norms: list[float] = []
    stat_keys = ('cost_env_mean', 'cost_is_mean', 'cost_is_std',
                 'cost_track_mean', 'cost_s_mean', 'n_eff', 'lam',
                 'coupling_active', 'coupling_used_fallback',
                 'coupling_feasible_fraction', 'coupling_policy_cost_mean',
                 'coupling_policy_cost_std', 'coupling_score_mean',
                 'merge_beta_mean', 'merge_beta_head', 'merge_kl_mean',
                 'merge_accepted', 'merge_cost_gap',
                 'mix_weight_share', 'mix_fraction_effective')
    stat_sums = {k: 0.0 for k in stat_keys}
    n_calls = 0
    action_low, action_high = env.action_bounds

    n_dagger = int(round(dagger_fraction * n_episodes)) if policy_act is not None else 0

    for ep in range(n_episodes):
        dagger_ep = ep < n_dagger
        np.random.seed(seed_base + ep)
        env.reset()
        mppi.reset()
        if merge is not None and hasattr(merge, "reset_episode"):
            merge.reset_episode()
        ou_state = np.zeros(env.action_dim)

        ep_cost = 0.0
        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        first_success_t: int | None = None
        hold_count = 0
        max_hold_count = 0
        for t in range(steps_per_episode):
            state = env.get_state()
            obs = env._get_obs()
            mix_nominal = mixer(state) if mixer is not None else None
            action, info = mppi.plan_step(
                state, prior_cost=prior, coupling=coupling, merge=merge,
                mix_nominal=mix_nominal, mix_fraction=mix_fraction,
            )
            for k in stat_keys:
                stat_sums[k] += info[k]
            n_calls += 1
            ep_obs.append(obs)
            # The label is always the certified planner action. In DAgger
            # episodes the *policy* drives so its drift states get labeled.
            ep_actions.append(np.clip(action, action_low, action_high))
            diag_beta.append(info['merge_beta_head'])
            diag_dagger.append(dagger_ep)
            diag_ep.append(ep)
            diag_gap.append(info['merge_cost_gap'])
            exec_action = policy_act(obs) if dagger_ep else action
            if exec_noise_std > 0.0:
                # GPS-style stochastic collection: the noise drifts the state
                # off the nominal ribbon; the next step's certified label is
                # the demonstrated correction. Labels stay noise-free. OU
                # correlation makes the drift persistent like real learner
                # bias instead of plant-filtered white jitter.
                if exec_noise_ou_tau > 0.0:
                    decay = np.exp(-1.0 / exec_noise_ou_tau)
                    ou_state = decay * ou_state + exec_noise_std * np.sqrt(
                        1.0 - decay**2
                    ) * np.random.randn(len(action))
                    noise = ou_state
                else:
                    noise = exec_noise_std * np.random.randn(len(action))
                exec_action = np.clip(exec_action + noise, action_low, action_high)
            _, cost, done, _ = env.step(exec_action)
            ep_cost += cost

            metrics = _task_metrics(env)
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
            act_chunks.append(np.asarray(ep_actions, dtype=np.float32))
        final_metrics = _task_metrics(env)
        hit_successes.append(first_success_t is not None)
        hold_successes.append(max_hold_count >= hold_steps)
        times_to_hit.append(first_success_t if first_success_t is not None else steps_per_episode)
        final_tip_dists.append(final_metrics["tip_dist"])
        final_qvel_norms.append(final_metrics["qvel_norm"])

    obs_dim = int(np.asarray(env._get_obs()).shape[-1])
    obs_arr = np.concatenate(obs_chunks, axis=0) if obs_chunks else np.empty((0, obs_dim), dtype=np.float32)
    act_arr = np.concatenate(act_chunks, axis=0) if act_chunks else np.empty((0, env.action_dim), dtype=np.float32)
    step_diag = {
        "beta": np.asarray(diag_beta, dtype=np.float32),
        "dagger": np.asarray(diag_dagger, dtype=bool),
        "episode": np.asarray(diag_ep, dtype=np.int32),
        "cost_gap": np.asarray(diag_gap, dtype=np.float32),
    }
    mppi_stats = {k: stat_sums[k] / max(n_calls, 1) for k in stat_keys}
    mppi_stats["step_diag"] = step_diag
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
    max_epochs: int = 0,
    target_loss: float = 0.0,
    sample_weights: np.ndarray | None = None,
) -> float:
    """Adam updates on (optionally weighted) MSE. Returns final epoch-mean loss.

    With max_epochs > 0, trains until the epoch-mean loss reaches target_loss
    or stops improving (patience 10), up to max_epochs — the policy must
    actually fit its dataset each iteration or no coupling scheme downstream
    can matter. Otherwise runs exactly `epochs` epochs.

    sample_weights down-weight stale replay data: labels from early GPS iters
    predate the merge feedback and carry conflicting modes; recent labels are
    consistent-by-construction. The reported loss stays unweighted so it is
    comparable across runs.
    """
    policy.train()
    device = next(policy.parameters()).device
    obs_b = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_b = torch.as_tensor(actions, dtype=torch.float32, device=device)
    w_b = None
    if sample_weights is not None:
        w_b = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    N = len(obs)

    n_epochs = max_epochs if max_epochs > 0 else max(1, epochs)
    best, patience, epoch_loss = float("inf"), 10, float("inf")
    for _ in range(n_epochs):
        perm = rng.permutation(N)
        losses = []
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            mu = policy.forward(obs_b[idx])
            se = (mu - act_b[idx]) ** 2
            if w_b is None:
                loss = se.mean()
            else:
                w = w_b[idx]
                loss = (w[:, None] * se).sum() / (w.sum() * se.shape[1] + 1e-8)
            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            losses.append(se.mean().item())
        epoch_loss = float(np.mean(losses))
        if max_epochs > 0:
            if epoch_loss <= target_loss:
                break
            if epoch_loss < best - 1e-5:
                best, patience = epoch_loss, 10
            else:
                patience -= 1
                if patience == 0:
                    break
    return epoch_loss


def evaluate_mppi(
    env: BaseEnv,
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

            metrics = _task_metrics(env)
            if metrics["success"]:
                if first_success_t is None:
                    first_success_t = t
                hold_count += 1
            else:
                hold_count = 0
            max_hold_count = max(max_hold_count, hold_count)

            if done:
                break

        final_metrics = _task_metrics(env)
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
    obs_from_states: Callable[[np.ndarray], np.ndarray] | None = None,
    env: BaseEnv | None = None,
    mppi: MPPI | None = None,
    lambda_track_dual: float = 0.0,
):
    """Return (prior_cost, coupling, merge, mixer) for the current GPS iteration."""
    collection_mode = getattr(gps_cfg, "collection_mode", "gps")
    if collection_mode == "bc":
        return None, None, None, None
    if collection_mode != "gps":
        raise ValueError(
            f"Unknown GPS collection_mode: {collection_mode!r}; expected 'bc' or 'gps'."
        )
    if gps_cfg.coupling_mode not in _COUPLING_MODES:
        raise ValueError(
            f"Unknown GPS coupling_mode: {gps_cfg.coupling_mode!r}; "
            f"expected one of {sorted(_COUPLING_MODES)}."
        )
    if it < gps_cfg.coupling_warmup_iters:
        return None, None, None, None

    mixer = (
        make_policy_mixer(policy, env, horizon=mppi.H)
        if gps_cfg.mix_fraction > 0.0
        else None
    )

    if gps_cfg.coupling_mode == "merge":
        # No policy_trust schedule and no agreement gate: the rollout
        # certificate alone decides how far each plan bends toward the policy.
        # With track_dual_alpha > 0 the Mordatch-style penalty additionally
        # pulls the *whole planned horizon* toward pi inside the score, at
        # the dual-ascended weight.
        prior = None
        if gps_cfg.track_dual_alpha > 0.0 and lambda_track_dual > 0.0:
            prior = make_policy_tracking_prior(
                policy,
                lambda_track=lambda_track_dual,
                obs_from_states=obs_from_states,
            )
        merge = make_policy_merge(
            policy,
            env,
            noise_precision=mppi.noise_precision,
            betas=tuple(gps_cfg.merge_betas),
            delta_frac=gps_cfg.merge_delta_frac,
            delta_floor=gps_cfg.merge_delta_floor,
            episode_budget_frac=gps_cfg.merge_episode_budget_frac,
            obs_from_states=obs_from_states,
        )
        return prior, None, merge, mixer

    lambda_track = gps_cfg.lambda_policy_track * policy_trust
    prior = make_policy_tracking_prior(
        policy,
        lambda_track=lambda_track,
        obs_from_states=obs_from_states,
    )
    if gps_cfg.coupling_mode == "track":
        return prior, None, None, mixer

    keep_fraction = 1.0 - policy_trust * (1.0 - gps_cfg.policy_coupling_keep_fraction)
    coupling = make_policy_filter_coupling(
        policy,
        min_fraction=gps_cfg.policy_coupling_min_fraction,
        keep_fraction=keep_fraction,
        min_n_eff=gps_cfg.policy_coupling_min_n_eff,
        max_weight=gps_cfg.policy_coupling_max_weight,
        obs_from_states=obs_from_states,
    )
    return prior, coupling, None, mixer


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
    env_name: str = "acrobot",
    run_name: str | None = None,
    device: str = "auto",
    use_warp: bool = False,
    collection_mode: str | None = None,
    n_gps_iters: int | None = None,
    episodes_per_iter: int | None = None,
    steps_per_episode: int | None = None,
    eval_every: int | None = None,
    eval_n_episodes: int | None = None,
    eval_episode_len: int | None = None,
    eval_mppi_baseline_episodes: int | None = None,
    bc_max_epochs: int | None = None,
    bc_target_loss: float | None = None,
    coupling_warmup_iters: int | None = None,
    coupling_mode: str | None = None,
    lambda_policy_track: float | None = None,
    adaptive_policy_trust: bool | None = None,
    policy_trust_bad_cost_per_step: float | None = None,
    policy_trust_min: float | None = None,
    policy_trust_max: float | None = None,
    policy_coupling_keep_fraction: float | None = None,
    policy_lr: float | None = None,
    mppi_lam: float | None = None,
    merge_delta_frac: float | None = None,
    mix_fraction: float | None = None,
    bc_recency_halflife: float | None = None,
    dagger_fraction: float | None = None,
    track_dual_alpha: float | None = None,
    track_dual_lambda_max: float | None = None,
    exec_noise_std: float | None = None,
    exec_noise_ou_tau: float | None = None,
    normalize_obs: bool | None = None,
    env_frame_skip: int = 1,
    env_energy_cost_weight: float | None = None,
) -> None:
    env_name = _normalize_env_name(env_name)
    gps_cfg = GPSConfig.load(env_name)
    _apply_gps_overrides(
        gps_cfg,
        collection_mode=collection_mode,
        n_gps_iters=n_gps_iters,
        episodes_per_iter=episodes_per_iter,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_n_episodes=eval_n_episodes,
        eval_episode_len=eval_episode_len,
        eval_mppi_baseline_episodes=eval_mppi_baseline_episodes,
        bc_max_epochs=bc_max_epochs,
        bc_target_loss=bc_target_loss,
        coupling_warmup_iters=coupling_warmup_iters,
        coupling_mode=coupling_mode,
        lambda_policy_track=lambda_policy_track,
        adaptive_policy_trust=adaptive_policy_trust,
        policy_trust_bad_cost_per_step=policy_trust_bad_cost_per_step,
        policy_trust_min=policy_trust_min,
        policy_trust_max=policy_trust_max,
        policy_coupling_keep_fraction=policy_coupling_keep_fraction,
        merge_delta_frac=merge_delta_frac,
        mix_fraction=mix_fraction,
        bc_recency_halflife=bc_recency_halflife,
        dagger_fraction=dagger_fraction,
        track_dual_alpha=track_dual_alpha,
        track_dual_lambda_max=track_dual_lambda_max,
        exec_noise_std=exec_noise_std,
        exec_noise_ou_tau=exec_noise_ou_tau,
        normalize_obs=normalize_obs,
    )
    gps_cfg.collection_mode = _normalize_collection_mode(gps_cfg.collection_mode)
    mppi_cfg = MPPIConfig.load(env_name)
    if mppi_lam is not None:
        # Higher temperature averages more samples per update: same task cost,
        # substantially more consistent (clonable) action labels.
        mppi_cfg.lam = mppi_lam
    policy_cfg = PolicyConfig()
    if policy_lr is not None:
        policy_cfg.lr = policy_lr

    if run_name is None:
        env_prefix = "" if env_name == "acrobot" else f"{env_name}_"
        suffix = "_warp" if use_warp else ""
        if gps_cfg.collection_mode == "bc":
            run_name = f"{env_prefix}bc{suffix}"
        elif gps_cfg.coupling_mode == "track":
            run_name = f"{env_prefix}gps_track_lambda_{gps_cfg.lambda_policy_track:g}{suffix}"
        elif gps_cfg.coupling_mode == "filter":
            run_name = (
                f"{env_prefix}gps_filter_lambda_{gps_cfg.lambda_policy_track:g}"
                f"_keep_{gps_cfg.policy_coupling_keep_fraction:g}{suffix}"
            )
        elif gps_cfg.coupling_mode == "merge":
            run_name = f"{env_prefix}gps_merge_delta_{gps_cfg.merge_delta_frac:g}{suffix}"
        else:
            raise ValueError(
                f"Unknown GPS coupling_mode: {gps_cfg.coupling_mode!r}; "
                f"expected one of {sorted(_COUPLING_MODES)}."
            )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    print(f"run_dir: {run_dir}")
    print(f"env_name: {env_name}")
    print(f"gps_cfg: {gps_cfg}")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    torch_device = _resolve_device(device)
    print(f"device: {torch_device}")
    print(f"use_warp: {use_warp}  nworld: {mppi_cfg.K}")

    env_kwargs: dict = {"use_warp": use_warp, "nworld": mppi_cfg.K}
    if env_frame_skip != 1:
        env_kwargs["frame_skip"] = env_frame_skip
    if env_energy_cost_weight is not None:
        env_kwargs["energy_cost_weight"] = env_energy_cost_weight
    env = _make_env(env_name, **env_kwargs)
    mppi = MPPI(env, mppi_cfg)
    policy = DeterministicPolicy(gps_cfg.obs_dim, gps_cfg.act_dim, policy_cfg).to(device=torch_device)
    obs_from_states = getattr(env, "rollout_states_to_obs", None)
    replay_obs: np.ndarray | None = None
    replay_acts: np.ndarray | None = None
    replay_tags: np.ndarray | None = None
    policy_trust = (
        gps_cfg.policy_trust_max
        if not gps_cfg.adaptive_policy_trust
        else gps_cfg.policy_trust_min
    )
    lambda_track_dual = 0.0
    track_violation = 0.0

    for it in range(gps_cfg.n_gps_iters):
        t_start = time.time()

        prior, coupling, merge, mixer = make_collection_bias(
            policy,
            gps_cfg,
            it,
            policy_trust=policy_trust,
            obs_from_states=obs_from_states,
            env=env,
            mppi=mppi,
            lambda_track_dual=lambda_track_dual,
        )
        seed_base = 10_000 + it * gps_cfg.episodes_per_iter

        def policy_act(obs: np.ndarray) -> np.ndarray:
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=torch_device
            ).unsqueeze(0)
            with torch.no_grad():
                return policy.forward(obs_t).squeeze(0).cpu().numpy()

        dagger_now = gps_cfg.dagger_fraction if it >= gps_cfg.coupling_warmup_iters else 0.0
        print("collecting demos")
        obs, acts, mppi_cost, mppi_stats = collect_episodes(
            env, mppi,
            n_episodes=gps_cfg.episodes_per_iter,
            steps_per_episode=gps_cfg.steps_per_episode,
            prior=prior,
            coupling=coupling,
            merge=merge,
            mixer=mixer,
            mix_fraction=gps_cfg.mix_fraction,
            policy_act=policy_act,
            dagger_fraction=dagger_now,
            exec_noise_std=gps_cfg.exec_noise_std,
            exec_noise_ou_tau=gps_cfg.exec_noise_ou_tau,
            seed_base=seed_base,
        )

        if gps_cfg.track_dual_alpha > 0.0 and len(obs) > 0:
            # dual ascent on the consistency constraint u = pi(s): grow the
            # in-score coupling while planner and policy still disagree
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=torch_device)
                mu = policy.forward(obs_t).cpu().numpy()
            track_violation = float(np.mean(np.sum((acts - mu) ** 2, axis=-1)))
            lambda_track_dual = min(
                lambda_track_dual + gps_cfg.track_dual_alpha * track_violation,
                gps_cfg.track_dual_lambda_max,
            )

        step_diag = mppi_stats.pop("step_diag")
        np.savez_compressed(run_dir / f"step_diag_iter_{it:03d}.npz", **step_diag)

        iter_tags = np.full(len(obs), it, dtype=np.int32)
        if gps_cfg.replay_max_pairs > 0:
            replay_obs = obs if replay_obs is None else np.concatenate([replay_obs, obs], axis=0)
            replay_acts = acts if replay_acts is None else np.concatenate([replay_acts, acts], axis=0)
            replay_tags = iter_tags if replay_tags is None else np.concatenate([replay_tags, iter_tags])
            if len(replay_obs) > gps_cfg.replay_max_pairs:
                replay_obs = replay_obs[-gps_cfg.replay_max_pairs:]
                replay_acts = replay_acts[-gps_cfg.replay_max_pairs:]
                replay_tags = replay_tags[-gps_cfg.replay_max_pairs:]
            train_obs, train_acts, train_tags = replay_obs, replay_acts, replay_tags
        else:
            train_obs, train_acts, train_tags = obs, acts, iter_tags

        sample_weights = None
        if gps_cfg.bc_recency_halflife > 0.0:
            age = (it - train_tags).astype(np.float64)
            sample_weights = 0.5 ** (age / gps_cfg.bc_recency_halflife)

        # adaptive stats with PopArt input-layer compensation: recomputed
        # each iter (the adaptivity walker_gps5's climb relied on; frozen
        # stats in gps6 flatlined) but output-preserving, so the update
        # cannot invalidate prior learning (gps5's late-dip suspect)
        if gps_cfg.normalize_obs and len(train_obs) > 0:
            policy.set_obs_stats(train_obs.mean(axis=0), train_obs.std(axis=0))

        np.savez_compressed(
            run_dir / "replay_latest.npz",
            obs=train_obs, acts=train_acts, tags=train_tags,
        )

        print("training policy")
        bc_loss = train_policy(
            policy, train_obs, train_acts,
            batch_size=gps_cfg.batch_size,
            rng=rng,
            epochs=gps_cfg.bc_epochs_per_iter,
            max_epochs=gps_cfg.bc_max_epochs,
            target_loss=gps_cfg.bc_target_loss,
            sample_weights=sample_weights,
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
            "wall_time_s": time.time() - t_start,
            "collection_mode": gps_cfg.collection_mode,
            "coupling_mode": gps_cfg.coupling_mode,
            "coupling_active_mode": (
                gps_cfg.coupling_mode
                if (prior is not None or coupling is not None or merge is not None)
                else ("bc" if gps_cfg.collection_mode == "bc" else "warmup")
            ),
            "policy_trust": policy_trust,
            "policy_trust_next": policy_trust_next,
            "lambda_track": gps_cfg.lambda_policy_track * policy_trust if prior is not None else 0.0,
            "policy_coupling_keep_fraction_effective": (
                1.0 - policy_trust * (1.0 - gps_cfg.policy_coupling_keep_fraction)
                if coupling is not None else 1.0
            ),
            "lambda_track_base": gps_cfg.lambda_policy_track,
            "policy_coupling_keep_fraction_base": gps_cfg.policy_coupling_keep_fraction,
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
            "merge_beta_mean": mppi_stats["merge_beta_mean"],
            "merge_beta_head": mppi_stats["merge_beta_head"],
            "merge_kl_mean": mppi_stats["merge_kl_mean"],
            "merge_accept_rate": mppi_stats["merge_accepted"],
            "merge_cost_gap_mean": mppi_stats["merge_cost_gap"],
            "mix_weight_share": mppi_stats["mix_weight_share"],
            "mix_fraction_effective": mppi_stats["mix_fraction_effective"],
            "mix_fraction_cfg": gps_cfg.mix_fraction,
            "bc_recency_halflife": gps_cfg.bc_recency_halflife,
            "dagger_fraction_effective": dagger_now,
            "policy_eval_mean_steps": (stats.get("mean_steps") if do_eval else None),
            "lambda_track_dual": lambda_track_dual,
            "track_violation": track_violation,
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
        # every iter: survival is noisy and the best policy is often not the
        # last one (gps5's peak checkpoint was lost to the every-5 schedule)
        torch.save(policy.state_dict(), run_dir / f"checkpoint_iter_{it:03d}.pt")

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
