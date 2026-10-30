"""Train a standalone Walker policy with MPPI-GPS, DAgger, or teacher labels.

labeler="mdgps" (method B, docs/objective.md): at states visited by the
current policy, one policy-centred MPPI step with an n_eff-targeted
temperature gives the label; the policy regresses onto labels. Optional
--tail-steps adds the current policy's simulated cost-to-go as MPPI's terminal
cost.

labeler="dagger": a raw MPPI expert (frozen temperature, its own warm start)
plans alongside every episode and labels every visited state; iteration 0
executes the expert (beta = 1), later iterations execute the policy.

labeler="teacher" (Step 15): the same raw MPPI expert runs every episode, with
N(0, exec_noise^2) added to the action it executes, and labels every state it
visits with its own action. The policy plays no part in planning or collection.
With label_neff_fraction set, each label instead re-weights MPPI's last samples
to that n_eff/K, a smoother estimate of its intent (Step 16). With
teacher_runs > 1, MPPI runs that many times from each start state, and the
policy learns only from the teacher_keep runs that do not fall and that it
already explains best (lowest squared error on their labels): the policy picks
its learnable set from MPPI's trajectories, and MPPI itself is unchanged
(Step 20). With teacher_selector set, a fixed policy ranks the runs instead
of the policy being trained (Step 21).

Legacy policies use SacActor means; the policy-selective preset uses a
deterministic tanh MLP. Both are saved in the load_policy format and evaluated
without a planner. Explicit selection modes use frozen weighted action-norm loss.
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from multiprocessing import Pool
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
import tyro

from src.envs.walker2d import Walker2d
from src.gps.health_monitor import EarlyAbortException, TrainingHealthMonitor
from src.gps.mdgps import mdgps_step_multiplier
from src.gps.policy import TanhMLPPolicy
from src.gps.research import (IterationBaseline, ResearchConfig, ResearchSession,
                              deterministic_probe_mask)
from src.gps.selection import SelectionDataset, build_selection, recent_selection, weighted_mse
from src.gps.telemetry import TelemetryMetrics
from src.rl.sac import SacActor
from src.utils.math import softmin_weights, temperature_for_neff_fraction

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OBS_DIM = 17
Policy = SacActor | TanhMLPPolicy


@dataclass
class TrainConfig:
    labeler: str = "mdgps"
    K: int = 256
    H: int = 48
    neff_fraction: float = 0.2
    tail_steps: int = 0
    # MDGPS adaptive step size: eps = log(1/rho) is rescaled after each
    # iteration by mdgps_step_multiplier and kept within the rho bounds.
    adapt_step: bool = False
    neff_fraction_min: float = 0.05
    neff_fraction_max: float = 0.5
    probe_probability: float = 0.02
    # Keep a label only if its improved plan costs no more than the policy's own
    # plan on the E-step objective; otherwise label the policy's own action
    # (docs/theory.md, Proposition 4).
    predictive_check: bool = False
    # Explicit modes run AFTER MPPI verification; legacy preserves prior runs.
    selection_mode: Literal["legacy", "anchor", "selective"] = "legacy"
    tau_s: float = 1.0
    policy_architecture: Literal["sac", "tanh"] = "sac"
    # If set, the E-step's noise std at iteration k is
    # sigma_0 * (1 + k / sigma_decay_iterations)^(-1/2), with sigma_0 from
    # configs/walker2d_best.json (docs/theory.md, Consequence 2).
    sigma_decay_iterations: float | None = None
    # Continue the run in out_dir from its last saved policy (see _resume).
    resume: bool = False
    # labeler="teacher": if set, each label is the first action of the
    # teacher's last samples re-weighted to n_eff/K = label_neff_fraction,
    # instead of the action it executes (Step 16).
    label_neff_fraction: float | None = None
    # labeler="teacher": MPPI runs per start state, and how many of the runs
    # that do not fall the policy keeps (those it already explains best).
    teacher_runs: int = 1
    teacher_keep: int | None = None
    # Optional checkpoint (load_policy format) of a fixed policy that ranks the
    # runs in place of the policy being trained.
    teacher_selector: str | None = None
    # Label with the batched GPU E-step (src/gps/gpu_estep.py): workers only
    # collect, then all of an iteration's states are labelled on the GPU with
    # the same per-episode noise streams the CPU path draws.
    gpu_estep: bool = False
    gpu_states_per_batch: int = 128
    save_labels: bool = False
    # Optional checkpoint (load_policy format) to start from instead of a random policy.
    init_policy: str | None = None
    iterations: int = 30
    episodes_per_iter: int = 20
    steps: int = 1000
    label_every: int = 2
    exec_noise: float = 0.1
    replay_iterations: int = 2
    grad_steps: int = 2000
    # Canonical name for new runs; overrides grad_steps when supplied.
    m_step_updates: int | None = None
    weight_decay: float = 0.0
    batch_size: int = 256
    lr: float = 3e-4
    eval_every: int = 5
    eval_episodes: int = 10
    seed: int = 0
    workers: int = max(1, (os.cpu_count() or 2) - 4)
    # Per-iteration research metrics reuse existing arrays and at most 2% probes.
    research_telemetry: bool = True
    enable_early_abort: bool = True
    wandb_project: str | None = None
    tensorboard: bool = False
    # Dedicated evaluation rollouts, separate from collection/diagnostics.
    research_ladder: bool = False
    ladder_every: int = 5
    ladder_episodes: int = 10
    ladder_steps: int = 400
    ladder_seed: int = 8_300_000


def _walker() -> Walker2d:
    task = json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())
    return Walker2d(**task, nthread=1)


def _new_actor(seed: int, architecture: Literal["sac", "tanh"] = "sac") -> Policy:
    env = _walker()
    low, high = env.action_bounds
    torch.manual_seed(seed)
    if architecture == "tanh":
        return TanhMLPPolicy(_OBS_DIM, low, high)
    return SacActor(_OBS_DIM, low, high)


def policy_state(actor: Policy) -> dict[str, Any]:
    return copy.deepcopy({
        "format_version": 1, "actor": actor.state_dict(), "obs_dim": actor.obs_dim,
        "hidden_dim": actor.hidden_dim, "observation_transform": "identity",
        "policy_type": getattr(actor, "policy_type", "sac"),
        "action_low": actor.action_low.cpu().tolist(),
        "action_high": actor.action_high.cpu().tolist(),
    })


def _batch_act(actor: Policy) -> Callable[[np.ndarray], np.ndarray]:
    """Return deterministic actions (..., 6) for observations (..., 17)."""
    @torch.no_grad()
    def act(obs: np.ndarray) -> np.ndarray:
        batch = np.atleast_2d(obs)
        if not batch.flags.writeable:
            batch = batch.copy()
        mean, _ = actor(torch.as_tensor(batch, dtype=torch.float32))
        out = (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)
        if not np.isfinite(out).all():
            raise FloatingPointError("Policy inference produced a nonfinite action.")
        return out if np.ndim(obs) > 1 else out[0]
    return act


def _noise_sigma(cfg: TrainConfig, base_sigma: float, iteration: int) -> float:
    if cfg.sigma_decay_iterations is None:
        return base_sigma
    return base_sigma / np.sqrt(1.0 + iteration / cfg.sigma_decay_iterations)


def _reweighted_label(rollout: dict[str, Any], neff_fraction: float) -> np.ndarray:
    """First action of MPPI's last samples, re-weighted to n_eff/K = neff_fraction."""
    costs = rollout["costs"]
    weights = softmin_weights(costs, temperature_for_neff_fraction(costs, neff_fraction))
    return np.einsum("k,ka->a", weights, rollout["actions"][:, 0])


def _collect(job: dict[str, Any]) -> dict[str, Any]:
    """Collect CPU states/observations; explicit selection is deferred to main."""
    from src.gps.mdgps import closed_loop_nominal, e_step
    from src.gps.policy_tail import WalkerPolicyTail
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    torch.set_num_threads(1)
    cfg: TrainConfig = job["cfg"]
    env = _walker()
    actor = _new_actor(0, cfg.policy_architecture)
    actor.load_state_dict(job["actor"])
    act = _batch_act(actor)
    base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
    if cfg.labeler == "mdgps":
        planner_cfg = replace(base, K=cfg.K, H=cfg.H, initial_iterations=1,
                              lam_target_neff_fraction=job["neff_fraction"],
                              noise_sigma=job.get("noise_sigma", base.noise_sigma))
    else:
        planner_cfg = replace(base, K=cfg.K, H=cfg.H)
    planner = MPPI(env, planner_cfg, seed=job["planner_seed"])
    tail = WalkerPolicyTail(env, act, cfg.tail_steps) if cfg.tail_steps else None
    rng = np.random.default_rng(job["noise_seed"])
    low, high = env.action_bounds
    np.random.seed(job["reset_seed"])
    obs = env.reset()
    observations, labels, diagnostics, velocities, costs, probes = [], [], [], [], [], []
    proposals = []
    pending, probe_flags, all_probes, label_steps = [], [], [], []
    expected_costs, tail_ratios, saturated = [], [], []
    numerical_valid = True
    for t in range(cfg.steps):
        state, warmstart = env.get_state(), env.get_warmstart()
        expert_action = None
        if cfg.labeler == "mdgps" and cfg.gpu_estep:
            if t % cfg.label_every == 0:
                # Same random draws as the CPU branch, so the same states are visited.
                pending.append((state, warmstart, (job["planner_seed"], len(pending))))
                probe_flags.append(bool(rng.random() < job["probe_probability"]))
                observations.append(obs.copy())
                label_steps.append(t)
        elif cfg.labeler == "mdgps":
            if t % cfg.label_every == 0:
                full_act = lambda s: act(env.rollout_states_to_obs(s))
                nominal, _ = closed_loop_nominal(env, state, full_act, cfg.H, warmstart)
                result = e_step(planner, env, state, nominal, warmstart, terminal=tail)
                rejected = (cfg.selection_mode == "legacy" and cfg.predictive_check
                            and result.plan_cost > result.nominal_cost)
                labels.append(nominal[0].copy() if rejected else result.label)
                proposals.append(result.label.copy())
                diagnostics.append([result.n_eff / cfg.K, result.lam, result.nominal_cost,
                                    result.plan_cost, result.sample_improvement])
                expected_costs.append(result.expected_cost)
                tail_ratios.append(result.tail_var_ratio)
                saturated.append(result.lambda_saturated)
                numerical_valid = numerical_valid and result.numerical_valid
                all_probes.append((state, warmstart, result.nominal_cost, result.plan_cost))
                if rng.random() < job["probe_probability"]:
                    probes.append((state, warmstart, result.nominal_cost, result.plan_cost))
                observations.append(obs.copy())
                label_steps.append(t)
        else:
            expert_action, info = planner.plan_step(state, initial_warmstart=warmstart)
            label = expert_action
            if cfg.label_neff_fraction is not None:
                label = _reweighted_label(planner.get_rollout_data(), cfg.label_neff_fraction)
            labels.append(label)
            observations.append(obs.copy())
            label_steps.append(t)
            diagnostics.append([info["n_eff"] / cfg.K, info["lam"], np.nan, np.nan,
                                float(np.linalg.norm(label - expert_action))])
        if job["execute_expert"]:
            action = expert_action
            if cfg.labeler == "teacher":
                action = np.clip(action + cfg.exec_noise * rng.standard_normal(env.action_dim), low, high)
        else:
            action = np.clip(act(obs) + cfg.exec_noise * rng.standard_normal(env.action_dim), low, high)
        if not np.isfinite(action).all():
            raise FloatingPointError("Collection produced a nonfinite action.")
        obs, cost, done, step_info = env.step(action)
        velocities.append(step_info["x_velocity"])
        costs.append(cost)
        if done:
            break
    return {
        "obs": np.asarray(observations), "labels": np.asarray(labels),
        "diagnostics": np.asarray(diagnostics), "steps": len(velocities),
        "survived": len(velocities) == cfg.steps and not done,
        "velocity_mae": float(np.mean(np.abs(env._v_target - np.asarray(velocities)))),
        "cost_per_step": float(np.mean(costs)), "probes": probes,
        "pending": pending, "probe_flags": probe_flags,
        "all_probes": all_probes, "label_steps": np.asarray(label_steps, dtype=int),
        "fell": bool(done), "expected_costs": np.asarray(expected_costs),
        "tail_var_ratio": np.asarray(tail_ratios, dtype=float),
        "lambda_saturated": np.asarray(saturated, dtype=bool),
        "numerical_valid": numerical_valid,
        "proposals": np.asarray(proposals),
    }


def _select_learnable(episodes: list[dict[str, Any]], actor: Policy, cfg: TrainConfig
                      ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """From each start state's cfg.teacher_runs consecutive runs, keep the
    cfg.teacher_keep that did not fall and that the policy fits best."""
    act = _batch_act(actor)
    errors = [float(np.mean((act(ep["obs"]) - ep["labels"]) ** 2)) for ep in episodes]
    kept, kept_errors, rejected_errors, empty = [], [], [], 0
    for start in range(0, len(episodes), cfg.teacher_runs):
        group = [i for i in range(start, start + cfg.teacher_runs) if episodes[i]["survived"]]
        ranked = sorted(group, key=lambda i: errors[i])
        kept += [episodes[i] for i in ranked[:cfg.teacher_keep]]
        kept_errors += [errors[i] for i in ranked[:cfg.teacher_keep]]
        rejected_errors += [errors[i] for i in ranked[cfg.teacher_keep:]]
        empty += not group
    def median(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None
    return kept, {"runs_kept": len(kept), "start_states_without_a_kept_run": empty,
                  "kept_fit_error_median": median(kept_errors), "rejected_fit_error_median": median(rejected_errors)}


def _improvement_check(actor_new: Policy, probes: list[tuple[np.ndarray, np.ndarray, float, float]],
                       cfg: TrainConfig, new_costs: np.ndarray | None = None) -> dict[str, Any]:
    """MDGPS step signal at probe states visited this iteration.

    predicted = S(old policy plan) - S(improved MPPI plan); actual = S(old
    policy plan) - S(new policy plan), same state and warm start. With a tail,
    each plan's cost includes its policy's simulated cost-to-go (the improved
    plan is followed by the old policy), so actual is the policy-iteration
    improvement of the policy's own H + tail cost.
    """
    if new_costs is None:
        from src.gps.mdgps import closed_loop_nominal
        from src.gps.policy_tail import WalkerPolicyTail

        env = _walker()
        act = _batch_act(actor_new)
        tail = WalkerPolicyTail(env, act, cfg.tail_steps) if cfg.tail_steps else None
        evaluated: list[float] = []
        for state, warmstart, _, _ in probes:
            plan, _ = closed_loop_nominal(env, state, lambda s: act(env.rollout_states_to_obs(s)), cfg.H, warmstart)
            states, cost, _ = env.batch_rollout(state, plan[None], initial_warmstart=warmstart)
            evaluated.append(float(cost[0] + (tail(states, plan[None])[0] if tail is not None else 0.0)))
            if tail is not None and not tail.last_actions_finite:
                raise FloatingPointError("Probe continuation produced nonfinite policy actions.")
        new_costs = np.asarray(evaluated)
    if new_costs.shape != (len(probes),):
        raise ValueError("Probe policy costs must have shape (number of probes,).")
    if not np.isfinite(new_costs).all():
        raise FloatingPointError("Probe policy costs must be finite.")
    old_costs = np.asarray([probe[2] for probe in probes])
    predicted = old_costs - np.asarray([probe[3] for probe in probes])
    actual = old_costs - new_costs
    return {"probes": int(len(probes)), "predicted_improvement": float(np.mean(predicted)),
            "actual_improvement": float(np.mean(actual)),
            "actual_positive_fraction": float(np.mean(actual > 0))}


def _evaluate(actor: Policy, episodes: int, seed: int) -> dict[str, float]:
    from scripts.runners.eval_policy_walker2d import evaluate_policy

    policy = copy.deepcopy(actor).cpu()
    policy.load_state_dict(actor.state_dict())
    return evaluate_policy(policy.eval(), episodes=episodes, seed=seed)["summary"]


def _fit(actor: Policy, obs: np.ndarray, labels: np.ndarray, cfg: TrainConfig,
         rng: np.random.Generator, weights: np.ndarray | None = None) -> float:
    """Fit (M, 17) observations to (M, 6) actions, with frozen (M,) weights.

    Explicit weights use sum of squared action errors / weight mass. Legacy
    unweighted calls retain the historical mean over actions as well as rows.
    Zero-weight rows are excluded before sampling; an empty selection skips Adam.
    """
    if weights is not None:
        if weights.shape != (len(obs),) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("weights must be finite, nonnegative and have shape (M,).")
        valid = weights > 0
        obs, labels, weights = obs[valid], labels[valid], weights[valid]
    if not len(obs):
        actor.cpu().eval()
        return 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = actor.to(device).train()
    params = [p for name, p in model.named_parameters() if not name.startswith("fc_logstd")]
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    w = None if weights is None else torch.as_tensor(weights, dtype=torch.float64, device=device)
    losses = []
    for _ in range(cfg.grad_steps if cfg.m_step_updates is None else cfg.m_step_updates):
        index = torch.as_tensor(rng.integers(0, len(x), size=min(cfg.batch_size, len(x))), device=device)
        mean, _ = model(x[index])
        prediction = torch.tanh(mean) * model.action_scale + model.action_bias
        try:
            loss = (torch.mean((prediction - y[index]) ** 2) if w is None
                    else weighted_mse(prediction, y[index], w[index]))
        except ValueError as exc:
            if "finite" in str(exc):
                raise FloatingPointError(f"M-step numerical failure: {exc}") from exc
            raise
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("M-step produced a nonfinite loss.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    actor.to("cpu").eval()
    return float(np.mean(losses[-100:]))


def _resume(out_dir: Path, cfg: TrainConfig, actor: Policy) -> tuple[int, float]:
    """Load the last saved policy of the run in out_dir; return the next iteration.

    The next iteration index, and so every per-iteration seed, is the one an
    unbroken run would use. Replay restarts empty and the fit's minibatches
    differ, so a resumed run is a statistical, not exact, continuation.
    """
    from src.rl.sac import load_policy

    stored = json.loads((out_dir / "config.json").read_text())
    defaults = asdict(TrainConfig())
    changed = sorted(k for k, v in asdict(cfg).items() if k != "resume" and v != stored.get(k, defaults[k]))
    if changed:
        raise ValueError(f"resume needs the stored config; these differ: {changed}")
    rows = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    payload = torch.load(out_dir / f"policy_{rows[-1]['iteration']:03d}.pt", map_location="cpu", weights_only=False)
    actor.load_state_dict(load_policy(payload).state_dict())
    return rows[-1]["iteration"] + 1, rows[-1].get("neff_fraction_next", cfg.neff_fraction)


def _train(out_dir: Path, cfg: TrainConfig) -> None:
    """Train from noisy on-policy states, saving a planner-free policy each iteration."""
    if cfg.labeler not in {"mdgps", "dagger", "teacher"}:
        raise ValueError("labeler must be 'mdgps', 'dagger' or 'teacher'.")
    if cfg.selection_mode not in {"legacy", "anchor", "selective"}:
        raise ValueError("selection_mode must be 'legacy', 'anchor' or 'selective'.")
    if cfg.policy_architecture not in {"sac", "tanh"}:
        raise ValueError("policy_architecture must be 'sac' or 'tanh'.")
    if cfg.selection_mode != "legacy" and cfg.labeler != "mdgps":
        raise ValueError("Policy selection requires labeler='mdgps'.")
    if cfg.selection_mode != "legacy" and cfg.predictive_check:
        raise ValueError("Use selection_mode='anchor' instead of combining it with predictive_check.")
    for name in ("K", "H", "iterations", "episodes_per_iter", "steps", "label_every", "workers",
                 "replay_iterations", "grad_steps", "batch_size", "eval_every", "eval_episodes",
                 "gpu_states_per_batch"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("ladder_every", "ladder_episodes", "ladder_steps"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if cfg.research_ladder and cfg.labeler != "mdgps":
        raise ValueError("The research ladder requires labeler='mdgps'.")
    if cfg.ladder_seed < 0:
        raise ValueError("ladder_seed must be nonnegative.")
    if cfg.m_step_updates is not None and (isinstance(cfg.m_step_updates, bool)
                                          or not isinstance(cfg.m_step_updates, int) or cfg.m_step_updates < 1):
        raise ValueError("m_step_updates must be a positive integer.")
    for name in ("lr", "tau_s"):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not np.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    if cfg.sigma_decay_iterations is not None and (not np.isfinite(cfg.sigma_decay_iterations)
                                                   or cfg.sigma_decay_iterations <= 0):
        raise ValueError("sigma_decay_iterations must be finite and positive.")
    if not 0 < cfg.neff_fraction < 1 or not 0 <= cfg.probe_probability <= 1:
        raise ValueError("Require 0 < neff_fraction < 1 and 0 <= probe_probability <= 1.")
    if cfg.tail_steps < 0 or not np.isfinite(cfg.exec_noise) or cfg.exec_noise < 0 or cfg.seed < 0:
        raise ValueError("tail_steps, exec_noise and seed must be nonnegative.")
    if cfg.label_neff_fraction is not None and cfg.labeler != "teacher":
        raise ValueError("label_neff_fraction needs labeler='teacher'.")
    if cfg.teacher_runs > 1 and (cfg.labeler != "teacher" or cfg.teacher_keep is None
                                 or not 1 <= cfg.teacher_keep <= cfg.teacher_runs):
        raise ValueError("teacher_runs > 1 needs labeler='teacher' and 1 <= teacher_keep <= teacher_runs.")
    if cfg.teacher_runs < 1 or (cfg.teacher_runs == 1 and (cfg.teacher_keep is not None
                                                           or cfg.teacher_selector is not None)):
        raise ValueError("teacher_keep and teacher_selector select among several runs; set teacher_runs > 1.")
    selector = None
    if cfg.teacher_selector is not None:
        from src.rl.sac import load_policy

        payload = torch.load(cfg.teacher_selector, map_location="cpu", weights_only=False)
        selector = load_policy(payload.get("policy", payload))
    # The seed formula below reserves 10,000 consecutive resets per training
    # seed. Longer runs would reuse the next training seed's reset/planner/
    # execution-noise streams and invalidate independent-seed comparisons.
    if cfg.iterations * cfg.episodes_per_iter > 10_000:
        raise ValueError("Training requires at most 10,000 episodes per seed to keep reset blocks disjoint.")
    rng = np.random.default_rng(cfg.seed)
    actor = _new_actor(cfg.seed, cfg.policy_architecture)
    neff_fraction, start = cfg.neff_fraction, 0
    if cfg.resume:
        start, neff_fraction = _resume(out_dir, cfg, actor)
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        if cfg.init_policy is not None:
            from src.rl.sac import load_policy

            payload = torch.load(cfg.init_policy, map_location="cpu", weights_only=False)
            actor.load_state_dict(load_policy(payload.get("policy", payload)).state_dict())
    base_sigma = json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text())["noise_sigma"]
    replay: list[tuple[np.ndarray, np.ndarray]] = []
    selected_replay: list[SelectionDataset] = []
    started = time.time()
    research_config = ResearchConfig(enabled=cfg.research_telemetry, enable_early_abort=cfg.enable_early_abort,
                                     wandb_project=cfg.wandb_project, wandb_run_name=out_dir.name,
                                     tensorboard=cfg.tensorboard)
    # Teacher/DAgger compatibility paths have no policy-centered E-step metrics.
    if cfg.labeler != "mdgps":
        research_config = replace(research_config, enabled=False, enable_early_abort=False)
    if not cfg.resume:
        torch.save(policy_state(actor), out_dir / "policy_initial.pt")
    with Pool(cfg.workers) as pool, ResearchSession(out_dir, research_config, resume_at=start) as research:
        gpu = None
        if cfg.gpu_estep:
            if cfg.labeler != "mdgps":
                raise ValueError("gpu_estep needs labeler='mdgps'.")
            # Imported and built after the pool forks, so workers never touch CUDA.
            from src.gps.gpu_estep import WalkerGPUEStep
            from src.utils.config import MPPIConfig

            base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
            gpu_cfg = replace(base, K=cfg.K, H=cfg.H, initial_iterations=1, lam_target_neff_fraction=neff_fraction)
            gpu = WalkerGPUEStep(_walker(), copy.deepcopy(actor).cuda().eval(), gpu_cfg, cfg.tail_steps,
                                 states_per_batch=cfg.gpu_states_per_batch)
        if cfg.research_ladder:
            from src.gps.eval_ladder import LadderConfig, evaluate_ladder_and_regularity
            from src.utils.config import MPPIConfig

            ladder_config = LadderConfig(episodes=cfg.ladder_episodes, steps=cfg.ladder_steps,
                                         reset_seed_base=cfg.ladder_seed,
                                         guided_neff_fraction=cfg.neff_fraction)
            # The cold teacher is policy-independent, so its fixed controller
            # configuration (including noise) is constant throughout the cache.
            ladder_planner = replace(MPPIConfig.load("walker2d"), K=cfg.K, H=cfg.H)
        for iteration in range(start, cfg.iterations):
            t0 = time.time()
            sigma = _noise_sigma(cfg, base_sigma, iteration)
            execute_expert = cfg.labeler == "teacher" or (cfg.labeler == "dagger" and iteration == 0)
            base_seed = 100_000 + 10_000 * cfg.seed + cfg.episodes_per_iter * iteration
            # Run r of a start state shares its reset; its planner and execution
            # noise come from a block 10^7 away, so run 0 is the single-run case.
            jobs = [{"cfg": cfg, "actor": {k: v.cpu() for k, v in actor.state_dict().items()},
                     "reset_seed": base_seed + e, "planner_seed": 500_000 + base_seed + e + 10_000_000 * r,
                     "noise_seed": 900_000 + base_seed + e + 10_000_000 * r, "execute_expert": execute_expert,
                     "probe_probability": cfg.probe_probability, "neff_fraction": neff_fraction,
                     "noise_sigma": sigma}
                    for e in range(cfg.episodes_per_iter) for r in range(cfg.teacher_runs)]
            runs = pool.map(_collect, jobs)
            episodes, selection = runs, {}
            if cfg.teacher_runs > 1:
                episodes, selection = _select_learnable(runs, actor if selector is None else selector, cfg)
                if not episodes:
                    raise RuntimeError(f"No MPPI run stayed up at iteration {iteration}; nothing to learn from.")
            seconds_collect = time.time() - t0
            obs = np.concatenate([ep["obs"] for ep in episodes])
            own = _batch_act(actor)(obs) if cfg.labeler == "mdgps" else None
            trajectory_ids = np.concatenate([np.full(len(ep["obs"]), index, dtype=int)
                                              for index, ep in enumerate(episodes)])
            label_steps = np.concatenate([ep["label_steps"] for ep in episodes])
            research_active = cfg.labeler == "mdgps" and (cfg.research_telemetry or cfg.enable_early_abort)
            if gpu is None:
                labels = np.concatenate([ep["labels"] for ep in episodes])
                diagnostics = np.concatenate([ep["diagnostics"] for ep in episodes])
                probes = [probe for ep in episodes for probe in ep["probes"]]
                if cfg.labeler == "mdgps":
                    proposed_actions = np.concatenate([ep["proposals"] for ep in episodes])
                    expected_costs = np.concatenate([ep["expected_costs"] for ep in episodes])
                    tail_ratios = np.concatenate([ep["tail_var_ratio"] for ep in episodes])
                    lambda_saturated = np.concatenate([ep["lambda_saturated"] for ep in episodes])
                    numerical_valid = all(ep["numerical_valid"] for ep in episodes)
                    if research_active:
                        all_probes = [probe for ep in episodes for probe in ep["all_probes"]]
                        mask = deterministic_probe_mask(len(obs), cfg.probe_probability, cfg.seed, iteration)
                        probes = [probe for probe, keep in zip(all_probes, mask) if keep]
            else:
                pending = [item for ep in episodes for item in ep["pending"]]
                flags = [flag for ep in episodes for flag in ep["probe_flags"]]
                gpu.actor.load_state_dict(actor.state_dict())
                gpu.cfg = replace(gpu.cfg, lam_target_neff_fraction=neff_fraction, noise_sigma=sigma)
                result = gpu(np.asarray([p[0] for p in pending]), np.asarray([p[1] for p in pending]),
                             noise_streams=[p[2] for p in pending])
                labels = result.labels
                proposed_actions = result.labels.copy()
                expected_costs = result.expected_cost
                tail_ratios = result.tail_var_ratio
                lambda_saturated = result.lambda_saturated
                numerical_valid = result.numerical_valid
                if cfg.selection_mode == "legacy" and cfg.predictive_check:
                    labels = np.where((result.plan_cost > result.nominal_cost)[:, None], own, labels)
                diagnostics = np.stack([result.n_eff / cfg.K, result.lam, result.nominal_cost, result.plan_cost,
                                        result.sample_mean_cost - result.expected_cost], axis=1)
                probes = [(p[0], p[1], nominal, plan) for p, flag, nominal, plan
                          in zip(pending, flags, result.nominal_cost, result.plan_cost) if flag]
                if research_active:
                    mask = deterministic_probe_mask(len(obs), cfg.probe_probability, cfg.seed, iteration)
                    probes = [(p[0], p[1], nominal, plan) for p, keep, nominal, plan
                              in zip(pending, mask, result.nominal_cost, result.plan_cost) if keep]
            seconds_label = time.time() - t0 - seconds_collect
            if research_active:
                research.check_numerics(iteration, labels, own, proposed_actions,
                                        diagnostics[:, 2:4], valid=numerical_valid)
            selected = None
            if cfg.selection_mode != "legacy":
                selected = build_selection(obs, labels, own, diagnostics[:, 2], diagnostics[:, 3],
                                           sigma, mode=cfg.selection_mode, tau_s=cfg.tau_s)
                labels = selected.actions
                selected_replay = (selected_replay + [selected])[-cfg.replay_iterations:]
            pre_diagnostics = None
            if research_active:
                weights = selected.weights if selected is not None else np.ones(len(obs))
                research.check_numerics(iteration, labels, weights)
                median_temperature = float(np.median(diagnostics[:, 1]))
                baseline = IterationBaseline(
                    mean_ep_length=float(np.mean([ep["steps"] for ep in runs])),
                    fall_rate=float(np.mean([ep["fell"] for ep in runs])),
                    nominal_costs=diagnostics[:, 2], sigma=float(sigma),
                    mean_n_eff=float(np.mean(diagnostics[:, 0]) * cfg.K),
                    median_lambda=median_temperature if np.isfinite(median_temperature) else None)
                pre_diagnostics = research.begin(iteration, observations=obs, targets=labels, old_actions=own,
                                                 weights=weights, trajectory_ids=trajectory_ids,
                                                 plan_cost=diagnostics[:, 3], expected_cost=expected_costs,
                                                 tail_var_ratio=tail_ratios, lambda_saturated=lambda_saturated,
                                                 action_delta=np.linalg.norm(proposed_actions - own, axis=1),
                                                 baseline=baseline, act=_batch_act(actor))
            if cfg.save_labels:
                extra = {} if selected is None else {
                    "weights": selected.weights, "proposals": selected.proposed_actions,
                    "old_actions": selected.old_actions, "sigma_k": np.asarray(sigma),
                }
                if cfg.labeler == "mdgps":
                    extra.update(expected_costs=expected_costs, tail_var_ratio=tail_ratios,
                                 lambda_saturated=lambda_saturated, trajectory_ids=trajectory_ids,
                                 label_steps=label_steps)
                np.savez_compressed(out_dir / f"labels_{iteration:03d}.npz", obs=obs, labels=labels,
                                    diagnostics=diagnostics, **extra)
            fit_started = time.time()
            if selected is None:
                replay = (replay + [(obs, labels)])[-cfg.replay_iterations:]
                loss = _fit(actor, np.concatenate([o for o, _ in replay]),
                            np.concatenate([l for _, l in replay]), cfg, rng)
                replay_size = sum(len(o) for o, _ in replay)
            else:
                dataset = recent_selection(selected_replay, cfg.replay_iterations)
                loss = _fit(actor, dataset.observations, dataset.actions, cfg, rng, dataset.weights)
                replay_size = len(dataset.weights)
            seconds_fit = time.time() - fit_started
            if research_active:
                parameters_valid = all(bool(torch.isfinite(tensor).all()) for tensor in actor.state_dict().values())
                research.check_numerics(iteration, np.asarray([loss]), valid=parameters_valid)
            median_temperature = float(np.nanmedian(diagnostics[:, 1]))
            record = {
                "iteration": iteration, "execute_expert": execute_expert,
                "collect_mean_steps": float(np.mean([ep["steps"] for ep in runs])),
                "collect_completed": int(sum(ep["survived"] for ep in runs)),
                "collect_fall_rate": float(np.mean([ep["fell"] for ep in runs])),
                "collect_velocity_mae": float(np.mean([ep["velocity_mae"] for ep in runs])),
                "collect_cost_per_step": float(np.mean([ep["cost_per_step"] for ep in runs])),
                **selection,
                "resumed": bool(cfg.resume and iteration == start),
                "labels": int(len(labels)), "fit_loss": loss, "neff_fraction_target": neff_fraction,
                "selection_mode": cfg.selection_mode, "replay_size": replay_size,
                "labeling": "gpu" if gpu is not None else "cpu", "seconds_collect": seconds_collect,
                "seconds_label": seconds_label, "seconds_fit": seconds_fit,
                "median_neff_fraction": float(np.nanmedian(diagnostics[:, 0])),
                "median_lam": median_temperature if np.isfinite(median_temperature) else None,
            }
            if cfg.labeler == "mdgps":
                record["predicted_improvement_mean"] = float(np.nanmean(diagnostics[:, 2] - diagnostics[:, 3]))
                # Share of states whose improved plan costs more than the policy's
                # own plan: the labels the predictive check replaces.
                record["plan_worse_fraction"] = float(np.mean(diagnostics[:, 3] > diagnostics[:, 2]))
                record["noise_sigma"] = sigma
                record["label_shift_median"] = float(np.median(np.linalg.norm(labels - own, axis=1)))
                record["share_plans_worse"] = record["plan_worse_fraction"]
                # This diagnostic measures raw MPPI disruption, before replacement/drop.
                delta = np.linalg.norm(labels - own, axis=1) if selected is None else selected.action_delta
                record["median_action_delta"] = float(np.median(delta))
                record["sigma_k"] = sigma
                record["mean_n_eff"] = float(np.mean(diagnostics[:, 0]) * cfg.K)
                record["median_lambda"] = record["median_lam"]
                record["ess_target_unreachable_states"] = int(np.count_nonzero(
                    np.abs(diagnostics[:, 0] - neff_fraction) > 1e-5))
                if selected is not None:
                    record["positive_weight_samples"] = int(np.count_nonzero(selected.weights))
                    record["weight_mass"] = float(np.sum(selected.weights))
                    record["replay_positive_weight_samples"] = int(np.count_nonzero(dataset.weights))
                    record["replay_weight_mass"] = float(np.sum(dataset.weights))
                    record["m_step_skipped"] = not bool(np.any(dataset.weights > 0))
                record["sample_improvement_min"] = float(np.nanmin(diagnostics[:, 4]))
                record["probe_improvement_ratio"] = None
                if probes:
                    new_costs = None
                    if gpu is not None:
                        gpu.actor.load_state_dict(actor.state_dict())
                        new_costs = gpu.policy_cost(np.asarray([p[0] for p in probes]),
                                                   np.asarray([p[1] for p in probes]))
                        if research_active:
                            research.check_numerics(iteration, valid=gpu.last_policy_cost_numerical_valid)
                    record["improvement_check"] = _improvement_check(actor, probes, cfg, new_costs)
                    record["improvement_check"]["cost_backend"] = "gpu" if gpu is not None else "cpu"
                    predicted = record["improvement_check"]["predicted_improvement"]
                    actual = record["improvement_check"]["actual_improvement"]
                    if abs(predicted) > 1e-12:
                        record["probe_improvement_ratio"] = float(actual / predicted)
                    if cfg.adapt_step:
                        check = record["improvement_check"]
                        multiplier = mdgps_step_multiplier(check["predicted_improvement"],
                                                           check["actual_improvement"])
                        eps = np.clip(np.log(1.0 / neff_fraction) * multiplier,
                                      np.log(1.0 / cfg.neff_fraction_max), np.log(1.0 / cfg.neff_fraction_min))
                        neff_fraction = float(np.exp(-eps))
                        record["step_multiplier"] = multiplier
                record["neff_fraction_next"] = neff_fraction
            if cfg.labeler == "teacher":
                record["label_vs_executed_median"] = float(np.median(diagnostics[:, 4]))
            if (iteration + 1) % cfg.eval_every == 0 or iteration == cfg.iterations - 1:
                record["eval_dev"] = _evaluate(actor, cfg.eval_episodes, 81_000)
                record["test_return_standalone"] = record["eval_dev"]["training_return"]
            torch.save(policy_state(actor), out_dir / f"policy_{iteration:03d}.pt")
            telemetry = None
            if pre_diagnostics is not None:
                # Abort broken learning before spending time on the controller ladder.
                telemetry = research.finish(iteration, pre_diagnostics, act=_batch_act(actor),
                                            probe=record.get("improvement_check"), fit_loss=loss,
                                            defer_log=True)
            ladder_values = None
            if cfg.research_ladder and (iteration == 0 or (iteration + 1) % cfg.ladder_every == 0
                                        or iteration == cfg.iterations - 1):
                ladder = evaluate_ladder_and_regularity(act_batch=_batch_act(actor), out_dir=out_dir,
                                                        iteration=iteration, planner_cfg=ladder_planner,
                                                        tail_steps=cfg.tail_steps, config=ladder_config)
                record["ladder"] = ladder
                record["seconds_ladder"] = ladder["elapsed_seconds"]
                ladder_values = ladder["metrics"]
            record["seconds"] = time.time() - t0
            with (out_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if telemetry is not None:
                research.log(telemetry, ladder_values)
            dev = record.get("eval_dev", {})
            check = record.get("improvement_check", {})
            print(f"it {iteration:2d} steps {record['collect_mean_steps']:7.1f} "
                  f"done {record['collect_completed']:2d}/{len(runs)} labels {record['labels']:6d} "
                  f"loss {loss:.4f} neff/K {record['median_neff_fraction']:.3f} "
                  f"pred/act {check.get('predicted_improvement', float('nan')):.2f}/"
                  f"{check.get('actual_improvement', float('nan')):.2f} "
                  f"dev_surv {dev.get('survival_rate', float('nan')):.2f} "
                  f"dev_mae {dev.get('velocity_mae', float('nan')):.3f} {record['seconds']:.0f}s", flush=True)
    status = {"complete": True, "wall_time_s": time.time() - started}
    if cfg.resume:
        status["resumed_at"] = start
    (out_dir / "status.json").write_text(json.dumps(status))


def _save_early_abort(out_dir: Path, exc: EarlyAbortException, started: float) -> None:
    """Mark a deliberate clean stop, retaining its last available policy and report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(out_dir.glob("policy_[0-9][0-9][0-9].pt"))
    checkpoint = checkpoints[-1] if checkpoints else out_dir / "policy_initial.pt"
    report = dict(exc.report)
    report["checkpoint"] = str(checkpoint) if checkpoint.exists() else None
    (out_dir / "early_abort_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    status = {"complete": False, "early_aborted": True, "iteration": exc.iteration,
              "reason_code": exc.reason_code, "diagnosis": str(exc),
              "wall_time_s": time.time() - started, "checkpoint": report["checkpoint"],
              "report": str(out_dir / "early_abort_report.json")}
    (out_dir / "status.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    print(f"Early abort at k={exc.iteration}: {exc}. Report: {status['report']}", flush=True)


def main(out_dir: Path, cfg: TrainConfig) -> None:
    """Train with a clean, checkpoint-preserving exit for diagnosed health failures."""
    started = time.time()
    try:
        _train(out_dir, cfg)
    except EarlyAbortException as exc:
        _save_early_abort(out_dir, exc, started)
    except FloatingPointError as cause:
        if not cfg.enable_early_abort:
            raise
        path = out_dir / "metrics.jsonl"
        rows = [json.loads(row) for row in path.read_text().splitlines() if row.strip()] if path.exists() else []
        iteration = rows[-1]["iteration"] + 1 if rows else 0
        monitor = TrainingHealthMonitor(report_path=out_dir / "early_abort_report.json")
        try:
            monitor.observe(TelemetryMetrics(iteration, {"numerical/actions_weights_finite": False}),
                            numerical_valid=False)
        except EarlyAbortException as exc:
            exc.report["original_error"] = str(cause)
            _save_early_abort(out_dir, exc, started)


if __name__ == "__main__":
    tyro.cli(main)
