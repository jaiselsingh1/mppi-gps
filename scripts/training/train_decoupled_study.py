"""Paired four-arm falsification study; old training entry points stay unchanged."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
import tyro

from scripts.runners.eval_policy_walker2d import evaluate_policy, provenance, write_json
from scripts.training.train_walker_mdgps import TrainConfig, _fit
from src.gps.policy import TanhMLPPolicy
from src.gps.selection import build_selection, recent_selection
from src.gps.value_credit import PolicyEpisode, ValueCreditTrainer, endpoint_value_credit, value_data_from_episodes
from src.rl.sac import load_policy
from src.rl.walker import WalkerRL
from src.utils.config import MPPIConfig

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("Baseline-Anchor", "Baseline-Selective", "Split-Medoid", "Split-Medoid-Critic")


@dataclass
class StudyConfig:
    iterations: int = 15
    seed: int = 0
    K: int = 256
    H: int = 32
    tail_steps: int = 100
    states_per_batch: int = 64
    max_label_states: int = 2000
    episodes: int = 10
    collection_steps: int = 400
    label_every: int = 2
    exec_noise: float = .1
    sigma_0: float = .3
    sigma_decay: float = 25.
    m_step_updates: int = 2000
    batch_size: int = 256
    lr: float = .0003
    workers: int = 4
    backend: str = "gpu"
    eval_every: int = 5
    eval_episodes: int = 100
    eval_steps: int = 1000
    eval_seed: int = 8_600_000
    value_gamma: float = .995
    value_episodes: int = 10
    value_holdout: int = 2
    value_steps: int = 1000
    value_updates: int = 2000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(actor):
    h = hashlib.sha256()
    for name, value in actor.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def validate(cfg):
    for key in ("iterations", "K", "H", "states_per_batch", "max_label_states", "episodes",
                "collection_steps", "label_every", "m_step_updates", "batch_size", "workers",
                "eval_every", "eval_episodes", "eval_steps", "value_episodes", "value_steps", "value_updates"):
        if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg.K % 2 or cfg.max_label_states > 2000 or cfg.backend not in {"cpu", "gpu"}:
        raise ValueError("Require even K, at most2000 roots, and cpu/gpu backend")
    if cfg.max_label_states < cfg.states_per_batch or cfg.seed < 0 or not 0 <= cfg.value_gamma < 1:
        raise ValueError("Invalid root cap, seed, or gamma")
    if type(cfg.value_holdout) is not int or not 0 <= cfg.value_holdout < cfg.value_episodes:
        raise ValueError("Value holdout must leave at least one training episode")
    if type(cfg.tail_steps) is not int or cfg.tail_steps < 0:
        raise ValueError("Tail controls must be nonnegative")
    if type(cfg.seed) is not int or type(cfg.eval_seed) is not int:
        raise ValueError("Seeds must be integers")
    if cfg.value_episodes > 100 or cfg.episodes * cfg.iterations > 10_000 or cfg.iterations > 100:
        raise ValueError("Acquisition must remain within its disjoint seed blocks")
    schedules = [(100_000 + 10_000 * cfg.seed, cfg.iterations * cfg.episodes),
                 (10_100_000 + 10_000 * cfg.seed, 100 * (cfg.iterations - 1) + cfg.value_episodes),
                 (cfg.eval_seed, cfg.eval_episodes)]
    for start, length in schedules:
        if start < 0 or start + length > 2 ** 32:
            raise ValueError("Reset seeds must fit uint32")
        if start <= 5_000_199 and start + length - 1 >= 5_000_000:
            raise ValueError("Reserved final reset panel must remain untouched")
    for i, (start, length) in enumerate(schedules):
        if any(start <= other + count - 1 and other <= start + length - 1
               for other, count in schedules[i + 1:]):
            raise ValueError("Training, value and evaluation seed blocks must be disjoint")
    for key in ("sigma_0", "sigma_decay", "lr"):
        if not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f"Invalid {key}")
    if not np.isfinite(cfg.exec_noise) or cfg.exec_noise < 0:
        raise ValueError("Invalid execution noise")


def collect_episode(job):
    """Private process-local reset/action randomness; no planner or critic."""
    torch.set_num_threads(1)
    actor = load_policy(job["policy"]).eval()
    env = WalkerRL(max_episode_steps=job["steps"])
    rng = np.random.default_rng(job["noise_seed"])
    observations, states, warmstarts, costs, velocities, actions = [], [], [], [], [], []
    try:
        obs, _ = env.reset(seed=job["reset_seed"])
        observations.append(obs); states.append(env.walker.get_state()); warmstarts.append(env.walker.get_warmstart())
        low, high = env.walker.action_bounds
        terminated = False
        for _ in range(job["steps"]):
            action = np.clip(actor.act(obs) + job["noise"] * rng.standard_normal(6), low, high)
            obs, _, terminated, truncated, info = env.step(action)
            observations.append(obs); states.append(env.walker.get_state()); warmstarts.append(env.walker.get_warmstart())
            costs.append(info["task_cost"]); velocities.append(info["x_velocity"]); actions.append(action)
            if terminated or truncated:
                break
        return {"obs": np.asarray(observations), "states": np.asarray(states), "warmstarts": np.asarray(warmstarts),
                "costs": np.asarray(costs), "actions": np.asarray(actions), "steps": len(costs),
                "terminated": bool(terminated), "reset_seed": job["reset_seed"],
                "velocity_mae": float(np.mean(np.abs(1.5 - np.asarray(velocities))))}
    finally:
        env.close()


def collection_jobs(actor, cfg, iteration, *, value=False):
    # Historical training reset schedule is reused as training data, never a holdout.
    base = (10_100_000 + 100 * iteration if value else 100_000 + cfg.episodes * iteration) + 10_000 * cfg.seed
    return [{"policy": actor.policy_state(), "steps": cfg.value_steps if value else cfg.collection_steps,
             "reset_seed": base + e, "noise_seed": 900_000 + base + e,
             "noise": 0. if value else cfg.exec_noise}
            for e in range(cfg.value_episodes if value else cfg.episodes)]


def roots(episodes, every):
    return {key: np.concatenate([e[key][:-1:every] for e in episodes])
            for key in ("obs", "states", "warmstarts")}


def matched_root_count(available, cap, batch):
    return min(cap, min(available)) // batch * batch


def metric_summary(rows):
    return {"episodes": len(rows), "executed_controls": sum(r["steps"] for r in rows),
            "collection_survival": sum(not r["terminated"] for r in rows) / len(rows),
            "velocity_mae": float(np.mean([r["velocity_mae"] for r in rows]))}


def append_json(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def run(out_dir: Path, cfg: StudyConfig):
    validate(cfg)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base = MPPIConfig(**json.loads((ROOT / "configs/walker2d_best.json").read_text()))
    planner = replace(base, K=cfg.K, H=cfg.H, initial_iterations=1, lam_target_neff_fraction=.2)
    env = WalkerRL()
    torch.manual_seed(cfg.seed)
    initial = TanhMLPPolicy(17, *env.walker.action_bounds)
    env.close()
    actors = {arm: copy.deepcopy(initial).eval() for arm in ARMS}
    replay = {arm: [] for arm in ARMS}
    rngs = {arm: np.random.default_rng(cfg.seed + 70) for arm in ARMS}
    critic = ValueCreditTrainer(gamma=cfg.value_gamma, lr=cfg.lr)
    source_paths = sorted(set(str(p.relative_to(ROOT)) for directory in (ROOT / "src", ROOT / "scripts")
                              for p in directory.rglob("*.py")))
    source_paths += ["configs/walker2d_best.json", "configs/walker2d_task.json", "assets/walker2d.xml",
                     "docs/decoupled_value_study.md"]
    fingerprints = {name: digest(ROOT / name) for name in source_paths}
    for name in fingerprints:
        target = out_dir / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    torch.save(initial.policy_state(), out_dir / "policy_initial.pt")
    initial_checkpoint_hash = digest(out_dir / "policy_initial.pt")
    write_json(out_dir / "protocol.json", {"config": asdict(cfg), "arms": ARMS,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": fingerprints, "environment": provenance(), "initial_parameter_hash": state_digest(initial),
        "initial_checkpoint_sha256": initial_checkpoint_hash,
        "scope": "One paired training-seed early falsification; no training-seed robustness or seeding-necessity claim.",
        "critic_objective": "Discounted continuing cost of deterministic learner only; candidate prefixes are followed by estimated Vpi.",
        "verification": "Full H-step synthesized plan is resimulated. Explicit-tail arms measure H+T cost; critic arm uses actual discounted prefix plus frozen learned value, not an independently observed future return.",
        "fallback": "Already scored single winning candidate, beta0, stable earliest-index tie break; no unverified cluster mean.",
        "critic_bootstrap_gate": "Iteration0 has no previous grounded value and explicitly bypasses the value-change gate; all later iterations use the strict frozen-snapshot gate without relaxation.",
        "budget": "Candidate-only cap; all overhead itemized. Common logical root count, own-policy states, not identical datasets."})
    for arm in ARMS:
        (out_dir / arm / "evaluations").mkdir(parents=True)
    status = {"phase": "initial_evaluation", "completed_iterations": 0, "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(out_dir / "status.json", status)
    initial_report = evaluate_policy(initial, episodes=cfg.eval_episodes, steps=cfg.eval_steps, seed=cfg.eval_seed)
    initial_report.update(checkpoint_sha256=initial_checkpoint_hash, completed_iterations=0)
    write_json(out_dir / "initial.json", initial_report)
    fit_cfg = TrainConfig(policy_architecture="tanh", grad_steps=cfg.m_step_updates,
                         batch_size=cfg.batch_size, lr=cfg.lr)
    context = mp.get_context("spawn")
    started = time.perf_counter()
    with context.Pool(cfg.workers) as pool:
        if cfg.backend == "gpu":
            from src.gps.decoupled_gpu import DecoupledGPUEStep
            simulator = DecoupledGPUEStep(WalkerRL().walker, planner, cfg.tail_steps, cfg.states_per_batch)
        else:
            from src.gps.decoupled_cpu import DecoupledCPUEStep
            simulator = DecoupledCPUEStep(WalkerRL().walker, planner, cfg.tail_steps, cfg.states_per_batch)
        try:
            for iteration in range(cfg.iterations):
                status.update(phase="collecting", iteration=iteration)
                write_json(out_dir / "status.json", status)
                collected, all_roots = {}, {}
                for arm in ARMS:
                    collected[arm] = pool.map(collect_episode, collection_jobs(actors[arm], cfg, iteration))
                    all_roots[arm] = roots(collected[arm], cfg.label_every)
                count = matched_root_count([len(all_roots[a]["obs"]) for a in ARMS], cfg.max_label_states, cfg.states_per_batch)
                if not count:
                    raise RuntimeError("Insufficient collected roots for one shared GPU batch; retain failed pilot.")
                previous_value = critic.snapshot()
                value_rows = pool.map(collect_episode, collection_jobs(actors[ARMS[-1]], cfg, iteration, value=True))
                policy_id = state_digest(actors[ARMS[-1]])
                episodes = [PolicyEpisode(r["obs"], r["costs"], r["terminated"], policy_id) for r in value_rows]
                training_episodes = episodes[:-cfg.value_holdout] if cfg.value_holdout else episodes
                transitions, anchors = value_data_from_episodes(training_episodes, gamma=cfg.value_gamma,
                    horizon=cfg.H, stride=max(1, cfg.H // 2), previous=previous_value)
                value_fit = critic.fit(transitions, anchors, updates=cfg.value_updates,
                                       batch_size=cfg.batch_size, seed=cfg.seed * 1000 + iteration)
                current_value = critic.snapshot()
                if cfg.value_holdout:
                    _, calibration = value_data_from_episodes(episodes[-cfg.value_holdout:], gamma=cfg.value_gamma,
                        horizon=cfg.H, previous=previous_value)
                    mask = calibration.exact
                    value_fit["heldout_exact_anchor_count"] = int(mask.sum())
                    value_fit["heldout_absorbing_rmse"] = (float(np.sqrt(np.mean((current_value.predict(
                        calibration.observations[mask]) - calibration.cost_targets[mask]) ** 2))) if mask.any() else None)
                np.savez_compressed(out_dir / ARMS[-1] / f"policy_value_data_{iteration:03d}.npz",
                    **{key: np.concatenate([row[key] for row in value_rows])
                       for key in ("obs", "states", "warmstarts", "costs", "actions")},
                    state_offsets=np.cumsum([0, *[r["steps"] + 1 for r in value_rows]]),
                    step_offsets=np.cumsum([0, *[r["steps"] for r in value_rows]]),
                    terminated=np.asarray([r["terminated"] for r in value_rows]),
                    reset_seeds=np.asarray([r["reset_seed"] for r in value_rows]),
                    training_episode_count=len(training_episodes), policy_id=policy_id)
                sigma = cfg.sigma_0 / np.sqrt(1 + iteration / cfg.sigma_decay)
                for arm_index, arm in enumerate(ARMS):
                    status.update(phase="labeling", arm=arm, common_label_states=count)
                    write_json(out_dir / "status.json", status)
                    data = all_roots[arm]
                    chosen = np.linspace(0, len(data["obs"]) - 1, count, dtype=int)
                    selected = {key: value[chosen] for key, value in data.items()}
                    actor = actors[arm]
                    before_hash = state_digest(actor)
                    credit_model = None
                    device = "cuda" if cfg.backend == "gpu" else "cpu"
                    frozen_actor = copy.deepcopy(actor).to(device).eval()
                    if arm == ARMS[-1]:
                        model = copy.deepcopy(current_value.model).to(device).eval()
                        credit_model = lambda obs, model=model: model(obs.float()).double()
                    labels_started = time.perf_counter()
                    result = simulator(selected["states"], selected["warmstarts"], frozen_actor,
                        [600_000 + cfg.seed * 1_000_000 + iteration * 2000 + i for i in range(count)],
                        proposal_mode="seeded" if arm_index < 2 else "split_proposal",
                        synthesis_mode="mean_verified" if arm_index < 2 else "medoid_fallback",
                        policy_sigma=float(sigma), credit_value=credit_model, credit_gamma=cfg.value_gamma)
                    labels_seconds = time.perf_counter() - labels_started
                    # The established Anchor control retains equal-cost mean actions.
                    # New fallback arms use the kernel's strict verified acceptance.
                    target_actions = result["mean_first"] if arm_index < 2 else result["labels"]
                    target_costs = result["mean_cost"] if arm_index < 2 else result["plan_cost"]
                    dataset = build_selection(selected["obs"], target_actions, result["nominal_first"],
                        result["nominal_cost"], target_costs, float(sigma),
                        mode="anchor" if arm_index == 0 else "selective")
                    gate = np.ones(count)
                    gate_log = {"bootstrap_bypass": False, "positive": count, "evaluated": 0}
                    if arm_index == 3:
                        valid = result["endpoint_healthy"].astype(bool)
                        if previous_value.grounded:
                            gate[:] = 0.
                            credit = endpoint_value_credit(previous_value, current_value,
                                result["endpoint_obs"][valid], tau=1.) if valid.any() else None
                            if credit is not None:
                                gate[valid] = credit["weights"]
                            gate_log = {"bootstrap_bypass": False, "positive": int((gate > 0).sum()),
                                        "evaluated": int(valid.sum())}
                        else:
                            gate_log["bootstrap_bypass"] = True
                    if arm_index == 3:
                        frozen_weights = np.array(dataset.weights * gate)
                        frozen_weights.setflags(write=False)
                        dataset = replace(dataset, weights=frozen_weights)
                    replay[arm].append(dataset)
                    frozen = recent_selection(replay[arm], window=2)
                    positive = int((frozen.weights > 0).sum())
                    loss = _fit(actor, frozen.observations, frozen.actions, fit_cfg, rngs[arm], frozen.weights)
                    after_hash = state_digest(actor)
                    if not positive and after_hash != before_hash:
                        raise RuntimeError("Empty weighted replay changed policy parameters")
                    checkpoint = out_dir / arm / f"policy_{iteration:03d}.pt"
                    torch.save(actor.policy_state(), checkpoint)
                    ledger = result["ledger"]
                    ledger["collection_controls"] = sum(row["steps"] for row in collected[arm])
                    ledger["value_acquisition_controls"] = sum(row["steps"] for row in value_rows) if arm_index == 3 else 0
                    ledger["evaluation_controls"] = 0
                    diag = result["diagnostics"]
                    if arm_index == 0:
                        diag["anchor_equal_cost_mean_retained"] = int(np.count_nonzero(
                            result["mean_cost"] == result["nominal_cost"]))
                    diag["bimodal_collapse_rate"] = diag["bimodal_collapse_count"] / count
                    diag["medoid_rescue_rate"] = (diag["medoid_rescued_collapse_count"] / diag["bimodal_collapse_count"]
                                                 if diag["bimodal_collapse_count"] else None)
                    row = {"iteration": iteration, "completed_iterations": iteration + 1,
                        "arm": arm, "label_states": count, "sigma_k": float(sigma),
                        "collection": metric_summary(collected[arm]), "diagnostics": diag,
                        "ledger": ledger, "label_seconds": labels_seconds, "mean_n_eff": float(np.mean(result["n_eff"])),
                        "current_weight_mass": float(dataset.weights.sum()), "replay_weight_mass": float(frozen.weights.sum()),
                        "replay_positive_rows": positive, "m_step_loss": loss,
                        "m_step_requested_updates": cfg.m_step_updates, "m_step_applied_updates": cfg.m_step_updates if positive else 0,
                        "parameters_before": before_hash, "parameters_after": after_hash,
                        "checkpoint_sha256": digest(checkpoint), "value_gate": gate_log}
                    if arm_index == 3:
                        row["value_fit"] = value_fit
                        torch.save(critic.state_dict(), out_dir / arm / f"value_{iteration:03d}.pt")
                    np.savez_compressed(out_dir / arm / f"labels_{iteration:03d}.npz",
                        observations=selected["obs"], states=selected["states"], warmstarts=selected["warmstarts"],
                        actions=dataset.actions, weights=dataset.weights, mean_cost=result["mean_cost"],
                        nominal_cost=result["nominal_cost"], selected_cost=target_costs,
                        sources=result["source_codes"], endpoints=result["endpoint_obs"], value_gate=gate)
                    if (iteration + 1) % cfg.eval_every == 0 or iteration + 1 == cfg.iterations:
                        status.update(phase="standalone_evaluation", arm=arm)
                        write_json(out_dir / "status.json", status)
                        report = evaluate_policy(actor, episodes=cfg.eval_episodes, steps=cfg.eval_steps, seed=cfg.eval_seed)
                        report.update(checkpoint_sha256=digest(checkpoint), completed_iterations=iteration + 1)
                        write_json(out_dir / arm / "evaluations" / f"iteration_{iteration + 1:03d}.json", report)
                        row["eval_dev"] = report["summary"]
                        ledger["evaluation_controls"] = sum(e["steps"] for e in report["episodes"])
                    append_json(out_dir / arm / "metrics.jsonl", row)
                    print(json.dumps({"arm": arm, "iteration": iteration + 1, "roots": count,
                        "collapse": diag["bimodal_collapse_rate"], "rescue": diag["medoid_rescue_rate"],
                        "weight_mass": row["current_weight_mass"], "evaluation": row.get("eval_dev")}), flush=True)
                    del frozen_actor, credit_model
                status.update(phase="training", completed_iterations=iteration + 1,
                              elapsed_seconds=time.perf_counter() - started)
                write_json(out_dir / "status.json", status)
            unchanged = all(digest(ROOT / name) == value for name, value in fingerprints.items())
            if not unchanged:
                raise RuntimeError("Training source changed during the study")
            status.update(phase="complete", source_unchanged=True)
            write_json(out_dir / "status.json", status)
        except BaseException as exc:
            status.update(phase="failed", error=repr(exc))
            write_json(out_dir / "status.json", status)
            raise
        finally:
            simulator.env.close()


if __name__ == "__main__":
    tyro.cli(run)
