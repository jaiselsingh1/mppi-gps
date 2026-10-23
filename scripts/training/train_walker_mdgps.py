"""Step 5: train a Walker policy from scratch with MPPI-GPS (B) or DAgger.

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

Policies are SacActor means trained by MSE on the squashed action, saved in
the load_policy format so scripts/runners/eval_policy_walker2d.py evaluates
them unchanged.
"""
import copy
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import tyro

from src.envs.walker2d import Walker2d
from src.gps.mdgps import mdgps_step_multiplier
from src.rl.sac import SacActor

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OBS_DIM = 17


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
    # If set, the E-step's noise std at iteration k is
    # sigma_0 * (1 + k / sigma_decay_iterations)^(-1/2), with sigma_0 from
    # configs/walker2d_best.json (docs/theory.md, Consequence 2).
    sigma_decay_iterations: float | None = None
    # Continue the run in out_dir from its last saved policy (see _resume).
    resume: bool = False
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
    batch_size: int = 256
    lr: float = 3e-4
    eval_every: int = 5
    eval_episodes: int = 10
    seed: int = 0
    workers: int = max(1, (os.cpu_count() or 2) - 4)


def _walker() -> Walker2d:
    task = json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())
    return Walker2d(**task, nthread=1)


def _new_actor(seed: int) -> SacActor:
    env = _walker()
    low, high = env.action_bounds
    torch.manual_seed(seed)
    return SacActor(_OBS_DIM, low, high)


def policy_state(actor: SacActor) -> dict:
    return copy.deepcopy({
        "format_version": 1, "actor": actor.state_dict(), "obs_dim": actor.obs_dim,
        "hidden_dim": actor.hidden_dim, "observation_transform": "identity",
        "action_low": actor.action_low.cpu().tolist(),
        "action_high": actor.action_high.cpu().tolist(),
    })


def _batch_act(actor: SacActor):
    @torch.no_grad()
    def act(obs: np.ndarray) -> np.ndarray:
        mean, _ = actor(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32))
        out = (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)
        return out if np.ndim(obs) > 1 else out[0]
    return act


def _noise_sigma(cfg: TrainConfig, base_sigma: float, iteration: int) -> float:
    if cfg.sigma_decay_iterations is None:
        return base_sigma
    return base_sigma / np.sqrt(1.0 + iteration / cfg.sigma_decay_iterations)


def _collect(job: dict) -> dict:
    from src.gps.mdgps import closed_loop_nominal, e_step
    from src.gps.policy_tail import WalkerPolicyTail
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    torch.set_num_threads(1)
    cfg: TrainConfig = job["cfg"]
    env = _walker()
    actor = _new_actor(0)
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
    pending, probe_flags = [], []
    for t in range(cfg.steps):
        state, warmstart = env.get_state(), env.get_warmstart()
        expert_action = None
        if cfg.labeler == "mdgps" and cfg.gpu_estep:
            if t % cfg.label_every == 0:
                # Same random draws as the CPU branch, so the same states are visited.
                pending.append((state, warmstart, (job["planner_seed"], len(pending))))
                probe_flags.append(bool(rng.random() < job["probe_probability"]))
                observations.append(obs.copy())
        elif cfg.labeler == "mdgps":
            if t % cfg.label_every == 0:
                full_act = lambda s: act(env.rollout_states_to_obs(s))
                nominal, _ = closed_loop_nominal(env, state, full_act, cfg.H, warmstart)
                result = e_step(planner, env, state, nominal, warmstart, terminal=tail)
                rejected = cfg.predictive_check and result.plan_cost > result.nominal_cost
                labels.append(nominal[0].copy() if rejected else result.label)
                diagnostics.append([result.n_eff / cfg.K, result.lam, result.nominal_cost,
                                    result.plan_cost, result.sample_improvement])
                if rng.random() < job["probe_probability"]:
                    probes.append((state, warmstart, result.nominal_cost, result.plan_cost))
                observations.append(obs.copy())
        else:
            expert_action, info = planner.plan_step(state, initial_warmstart=warmstart)
            labels.append(expert_action)
            observations.append(obs.copy())
            diagnostics.append([info["n_eff"] / cfg.K, info["lam"], np.nan, np.nan, np.nan])
        if job["execute_expert"]:
            action = expert_action
            if cfg.labeler == "teacher":
                action = np.clip(action + cfg.exec_noise * rng.standard_normal(env.action_dim), low, high)
        else:
            action = np.clip(act(obs) + cfg.exec_noise * rng.standard_normal(env.action_dim), low, high)
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
    }


def _improvement_check(actor_new: SacActor, probes: list, cfg: TrainConfig) -> dict:
    """MDGPS step signal at probe states visited this iteration.

    predicted = S(old policy plan) - S(improved MPPI plan); actual = S(old
    policy plan) - S(new policy plan), same state and warm start. With a tail,
    each plan's cost includes its policy's simulated cost-to-go (the improved
    plan is followed by the old policy), so actual is the policy-iteration
    improvement of the policy's own H + tail cost.
    """
    from src.gps.mdgps import closed_loop_nominal
    from src.gps.policy_tail import WalkerPolicyTail

    env = _walker()
    act = _batch_act(actor_new)
    tail = WalkerPolicyTail(env, act, cfg.tail_steps) if cfg.tail_steps else None
    predicted, actual = [], []
    for state, warmstart, old_cost, plan_cost in probes:
        plan, _ = closed_loop_nominal(env, state, lambda s: act(env.rollout_states_to_obs(s)), cfg.H, warmstart)
        states, cost, _ = env.batch_rollout(state, plan[None], initial_warmstart=warmstart)
        new_cost = cost[0] + (tail(states, plan[None])[0] if tail is not None else 0.0)
        predicted.append(old_cost - plan_cost)
        actual.append(old_cost - new_cost)
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    return {"probes": int(len(probes)), "predicted_improvement": float(np.mean(predicted)),
            "actual_improvement": float(np.mean(actual)),
            "actual_positive_fraction": float(np.mean(actual > 0))}


def _evaluate(actor: SacActor, episodes: int, seed: int) -> dict:
    from scripts.runners.eval_policy_walker2d import evaluate_policy

    policy = _new_actor(0)
    policy.load_state_dict(actor.state_dict())
    return evaluate_policy(policy.eval(), episodes=episodes, seed=seed)["summary"]


def _fit(actor: SacActor, obs: np.ndarray, labels: np.ndarray, cfg: TrainConfig, rng: np.random.Generator) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = actor.to(device).train()
    params = [p for name, p in model.named_parameters() if not name.startswith("fc_logstd")]
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    losses = []
    for _ in range(cfg.grad_steps):
        index = torch.as_tensor(rng.integers(0, len(x), size=min(cfg.batch_size, len(x))), device=device)
        mean, _ = model(x[index])
        prediction = torch.tanh(mean) * model.action_scale + model.action_bias
        loss = torch.mean((prediction - y[index]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    actor.to("cpu").eval()
    return float(np.mean(losses[-100:]))


def _resume(out_dir: Path, cfg: TrainConfig, actor: SacActor) -> tuple[int, float]:
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


def main(out_dir: Path, cfg: TrainConfig) -> None:
    if cfg.labeler not in {"mdgps", "dagger", "teacher"}:
        raise ValueError("labeler must be 'mdgps', 'dagger' or 'teacher'.")
    rng = np.random.default_rng(cfg.seed)
    actor = _new_actor(cfg.seed)
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
    started = time.time()
    with Pool(cfg.workers) as pool:
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
        for iteration in range(start, cfg.iterations):
            t0 = time.time()
            sigma = _noise_sigma(cfg, base_sigma, iteration)
            execute_expert = cfg.labeler == "teacher" or (cfg.labeler == "dagger" and iteration == 0)
            base_seed = 100_000 + 10_000 * cfg.seed + cfg.episodes_per_iter * iteration
            jobs = [{"cfg": cfg, "actor": {k: v.cpu() for k, v in actor.state_dict().items()},
                     "reset_seed": base_seed + e, "planner_seed": 500_000 + base_seed + e,
                     "noise_seed": 900_000 + base_seed + e, "execute_expert": execute_expert,
                     "probe_probability": cfg.probe_probability, "neff_fraction": neff_fraction,
                     "noise_sigma": sigma}
                    for e in range(cfg.episodes_per_iter)]
            episodes = pool.map(_collect, jobs)
            seconds_collect = time.time() - t0
            obs = np.concatenate([ep["obs"] for ep in episodes])
            own = _batch_act(actor)(obs) if cfg.labeler == "mdgps" else None
            if gpu is None:
                labels = np.concatenate([ep["labels"] for ep in episodes])
                diagnostics = np.concatenate([ep["diagnostics"] for ep in episodes])
                probes = [probe for ep in episodes for probe in ep["probes"]]
            else:
                pending = [item for ep in episodes for item in ep["pending"]]
                flags = [flag for ep in episodes for flag in ep["probe_flags"]]
                gpu.actor.load_state_dict(actor.state_dict())
                gpu.cfg = replace(gpu.cfg, lam_target_neff_fraction=neff_fraction, noise_sigma=sigma)
                result = gpu(np.asarray([p[0] for p in pending]), np.asarray([p[1] for p in pending]),
                             noise_streams=[p[2] for p in pending])
                labels = result.labels
                if cfg.predictive_check:
                    labels = np.where((result.plan_cost > result.nominal_cost)[:, None], own, labels)
                diagnostics = np.stack([result.n_eff / cfg.K, result.lam, result.nominal_cost, result.plan_cost,
                                        result.sample_mean_cost - result.expected_cost], axis=1)
                probes = [(p[0], p[1], nominal, plan) for p, flag, nominal, plan
                          in zip(pending, flags, result.nominal_cost, result.plan_cost) if flag]
            seconds_label = time.time() - t0 - seconds_collect
            if cfg.save_labels:
                np.savez_compressed(out_dir / f"labels_{iteration:03d}.npz", obs=obs, labels=labels,
                                    diagnostics=diagnostics)
            replay = (replay + [(obs, labels)])[-cfg.replay_iterations:]
            fit_started = time.time()
            loss = _fit(actor, np.concatenate([o for o, _ in replay]),
                        np.concatenate([l for _, l in replay]), cfg, rng)
            seconds_fit = time.time() - fit_started
            record = {
                "iteration": iteration, "execute_expert": execute_expert,
                "collect_mean_steps": float(np.mean([ep["steps"] for ep in episodes])),
                "collect_completed": int(sum(ep["survived"] for ep in episodes)),
                "collect_velocity_mae": float(np.mean([ep["velocity_mae"] for ep in episodes])),
                "collect_cost_per_step": float(np.mean([ep["cost_per_step"] for ep in episodes])),
                "resumed": bool(cfg.resume and iteration == start),
                "labels": int(len(labels)), "fit_loss": loss, "neff_fraction_target": neff_fraction,
                "labeling": "gpu" if gpu is not None else "cpu", "seconds_collect": seconds_collect,
                "seconds_label": seconds_label, "seconds_fit": seconds_fit,
                "median_neff_fraction": float(np.nanmedian(diagnostics[:, 0])),
                "median_lam": float(np.nanmedian(diagnostics[:, 1])),
            }
            if cfg.labeler == "mdgps":
                record["predicted_improvement_mean"] = float(np.nanmean(diagnostics[:, 2] - diagnostics[:, 3]))
                # Share of states whose improved plan costs more than the policy's
                # own plan: the labels the predictive check replaces.
                record["plan_worse_fraction"] = float(np.mean(diagnostics[:, 3] > diagnostics[:, 2]))
                record["noise_sigma"] = sigma
                record["label_shift_median"] = float(np.median(np.linalg.norm(labels - own, axis=1)))
                record["sample_improvement_min"] = float(np.nanmin(diagnostics[:, 4]))
                if probes:
                    record["improvement_check"] = _improvement_check(actor, probes, cfg)
                    if cfg.adapt_step:
                        check = record["improvement_check"]
                        multiplier = mdgps_step_multiplier(check["predicted_improvement"],
                                                           check["actual_improvement"])
                        eps = np.clip(np.log(1.0 / neff_fraction) * multiplier,
                                      np.log(1.0 / cfg.neff_fraction_max), np.log(1.0 / cfg.neff_fraction_min))
                        neff_fraction = float(np.exp(-eps))
                        record["step_multiplier"] = multiplier
                record["neff_fraction_next"] = neff_fraction
            if (iteration + 1) % cfg.eval_every == 0 or iteration == cfg.iterations - 1:
                record["eval_dev"] = _evaluate(actor, cfg.eval_episodes, 81_000)
            record["seconds"] = time.time() - t0
            torch.save(policy_state(actor), out_dir / f"policy_{iteration:03d}.pt")
            with (out_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            dev = record.get("eval_dev", {})
            check = record.get("improvement_check", {})
            print(f"it {iteration:2d} steps {record['collect_mean_steps']:7.1f} "
                  f"done {record['collect_completed']:2d}/{cfg.episodes_per_iter} labels {record['labels']:6d} "
                  f"loss {loss:.4f} neff/K {record['median_neff_fraction']:.3f} "
                  f"pred/act {check.get('predicted_improvement', float('nan')):.2f}/"
                  f"{check.get('actual_improvement', float('nan')):.2f} "
                  f"dev_surv {dev.get('survival_rate', float('nan')):.2f} "
                  f"dev_mae {dev.get('velocity_mae', float('nan')):.3f} {record['seconds']:.0f}s", flush=True)
    status = {"complete": True, "wall_time_s": time.time() - started}
    if cfg.resume:
        status["resumed_at"] = start
    (out_dir / "status.json").write_text(json.dumps(status))


if __name__ == "__main__":
    tyro.cli(main)
