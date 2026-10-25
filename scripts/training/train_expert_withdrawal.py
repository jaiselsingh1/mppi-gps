"""Outcome-based policy search with an independent expert and verified withdrawal.

The expert value is explicitly V_MPPI(s, remaining), never V^pi. Policy
parameters are searched with complete closed-loop prefixes, without action
regression. Actual paired takeovers verify proposed updates. At L=N neither
the value model nor the MPPI planner participates in policy optimization.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable

import numpy as np
import torch
import tyro

from scripts.runners.eval_policy_walker2d import evaluate_policy, provenance, write_json
from src.gps.expert_continuation import (Snapshot, finite_absorbing_cost,
    independent_teacher_suffix, raw_teacher_config, rollout_policy_prefix, takeover_objective)
from src.gps.expert_value import EXPERT_SOURCE, ExpertValueData, ExpertValueTrainer
from src.gps.policy import TanhMLPPolicy
from src.gps.prefix_search import antithetic_candidates, propose_es_update, verify_parameter_proposal
from src.rl.sac import load_policy
from src.rl.walker import WalkerRL

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class WithdrawalConfig:
    total_steps: int = 1000
    prefix_schedule: tuple[int, ...] = (1, 8, 32, 128, 400, 1000)
    updates_per_stage: int = 10
    pairs: int = 8
    train_starts: int = 4
    calibration_starts: int = 2
    verification_starts: int = 2
    value_labels: int = 16
    value_updates: int = 1000
    value_batch_size: int = 128
    hidden_dim: int = 256
    parameter_sigma: float = .01
    parameter_lr: float = .01
    max_step_norm: float = .05
    normalize_costs: bool = True
    teacher_tolerance: float = .05
    gamma: float = .999
    teacher_K: int = 32
    teacher_H: int = 48
    teacher_initial_iterations: int = 2
    seed: int = 0
    workers: int = 4
    eval_episodes: int = 16


def validate(cfg: WithdrawalConfig) -> None:
    """Reject invalid horizons or overlapping/reserved acquisition seed blocks."""
    positive = ("total_steps", "updates_per_stage", "pairs", "train_starts", "calibration_starts",
                "verification_starts", "value_labels", "value_updates", "value_batch_size",
                "hidden_dim", "teacher_K", "teacher_H", "teacher_initial_iterations", "workers", "eval_episodes")
    for name in positive:
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (not cfg.prefix_schedule or tuple(sorted(set(cfg.prefix_schedule))) != cfg.prefix_schedule
            or any(type(v) is not int or not 1 <= v <= cfg.total_steps for v in cfg.prefix_schedule)
            or cfg.prefix_schedule[-1] != cfg.total_steps):
        raise ValueError("Prefixes must strictly increase and end at the full episode horizon")
    if cfg.value_labels < 2 or cfg.calibration_starts < 1:
        raise ValueError("Expert fitting needs at least two training targets and held-out starts")
    if cfg.teacher_H < 2:
        raise ValueError("Shifted-plan teacher requires a horizon of at least two")
    for name in ("parameter_sigma", "parameter_lr", "max_step_norm"):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"Invalid {name}")
    if not np.isfinite(cfg.gamma) or not 0 <= cfg.gamma < 1:
        raise ValueError("gamma must be finite in [0,1)")
    if not np.isfinite(cfg.teacher_tolerance) or cfg.teacher_tolerance < 0:
        raise ValueError("teacher_tolerance must be nonnegative")
    if type(cfg.seed) is not int or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    iterations = len(cfg.prefix_schedule) * cfg.updates_per_stage
    sizes = (iterations * cfg.train_starts, iterations * cfg.calibration_starts,
             iterations * cfg.verification_starts, cfg.eval_episodes)
    if max(sizes) > 20_000 or 11_060_000 + cfg.seed * 100_000 + cfg.eval_episodes > 2 ** 32:
        raise ValueError("Acquisition must fit disjoint uint32 seed blocks")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger_add(target: dict[str, int], addition: dict[str, int]) -> None:
    for key, value in addition.items():
        target[key] = target.get(key, 0) + int(value)


def _job(job: dict[str, Any]) -> Any:
    """Independent process job: expert jobs do not even deserialize an actor."""
    torch.set_num_threads(1)
    if job["kind"] == "teacher":
        return independent_teacher_suffix(job["snapshot"], steps=job["steps"],
            planner_seed=job["planner_seed"], cfg=job["planner"], gamma=job["gamma"])
    actor = load_policy(job["policy"]).eval()
    if job["kind"] == "prefix":
        return rollout_policy_prefix(job["snapshot"], actor, steps=job["steps"], gamma=job["gamma"])
    if job["kind"] != "takeover":
        raise ValueError("Unknown rollout role")
    return takeover_objective(job["snapshot"], actor, prefix_steps=job["steps"],
        total_steps=job["total_steps"], planner_seed=job["planner_seed"],
        cfg=job["planner"], gamma=job["gamma"])


def _starts(seed: int, count: int, horizon: int) -> list[Snapshot]:
    env = WalkerRL(max_episode_steps=horizon)
    try:
        output = []
        for i in range(count):
            env.reset(seed=seed + i)
            output.append(Snapshot(env.walker.get_state(), env.walker.get_warmstart()))
        return output
    finally:
        env.close()


def prefix_objective(prefix: Any, *, total_steps: int, value: Any | None) -> float:
    """Own executed costs plus an explicit expert suffix, or finite fall future."""
    remaining = total_steps - prefix.steps
    if remaining < 0:
        raise ValueError("Prefix exceeds total horizon")
    if prefix.terminated:
        tail = finite_absorbing_cost(prefix.gamma, remaining, prefix.unhealthy_cost)
    elif remaining == 0:
        tail = 0.
    else:
        if value is None:
            raise RuntimeError("A healthy unfinished prefix requires a grounded expert value")
        tail = float(value.predict(prefix.endpoint_obs[None], remaining)[0])
    cost = float(prefix.cost + prefix.gamma ** prefix.steps * tail)
    if not np.isfinite(cost):
        raise FloatingPointError("Nonfinite prefix objective")
    return cost


def candidate_objectives(prefixes: list[Any], *, total_steps: int, value: Any | None,
                         exact_suffixes: dict[int, Any] | None = None) -> np.ndarray:
    """Score the same takeover objective with a fitted or actually executed tail.

    Failed prefixes remain informative even when no healthy endpoints exist to
    fit a value model. Exact suffixes are used only when that model cannot be
    grounded; they never turn an MPPI action into a policy training target.
    """
    costs = []
    for i, prefix in enumerate(prefixes):
        remaining = total_steps - prefix.steps
        if not prefix.terminated and remaining > 0 and exact_suffixes is not None:
            suffix = exact_suffixes[i]
            if suffix.requested_steps != remaining or suffix.gamma != prefix.gamma:
                raise ValueError("Exact suffix must match the remaining horizon and discount")
            cost = float(prefix.cost + prefix.gamma ** prefix.steps * suffix.cost)
        else:
            cost = prefix_objective(prefix, total_steps=total_steps, value=value)
        if not np.isfinite(cost):
            raise FloatingPointError("Nonfinite candidate objective")
        costs.append(cost)
    return np.asarray(costs)


def _standalone(actor: TanhMLPPolicy, directory: Path, name: str,
                cfg: WithdrawalConfig) -> dict[str, Any]:
    trace = directory / f"{name}.npz"
    report = evaluate_policy(actor, gamma=cfg.gamma, episodes=cfg.eval_episodes,
        steps=cfg.total_steps, seed=11_060_000 + 100_000 * cfg.seed, traces_path=trace)
    with np.load(trace, allow_pickle=False) as arrays:
        offsets, costs = arrays["episode_step_offsets"], arrays["costs"]
        finite = []
        for row, begin, end in zip(report["episodes"], offsets[:-1], offsets[1:]):
            raw = costs[begin:end]
            value = float(np.dot(cfg.gamma ** np.arange(len(raw)), raw))
            if row["terminated"]:
                value += cfg.gamma ** len(raw) * finite_absorbing_cost(cfg.gamma, cfg.total_steps - len(raw))
            finite.append(value)
    report["summary"]["finite_discounted_cost"] = float(np.mean(finite))
    report["summary"]["mean_duration_seconds"] = report["summary"]["mean_steps"] * .008
    report["standalone"] = True
    report["walking_gate"] = (report["summary"]["survival_rate"] >= .9 and
        report["summary"]["velocity_mae"] <= .3) if cfg.total_steps == 1000 else None
    write_json(directory / f"{name}.json", report)
    return report


def run(out_dir: Path, cfg: WithdrawalConfig) -> None:
    """Train the finite-horizon expert-withdrawal formulation in a fresh directory."""
    validate(cfg)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = WalkerRL()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(cfg.seed).get_state())
            actor = TanhMLPPolicy(17, *env.walker.action_bounds, hidden_dim=cfg.hidden_dim)
    finally:
        env.close()
    planner = replace(raw_teacher_config(), K=cfg.teacher_K, H=cfg.teacher_H,
                      initial_iterations=cfg.teacher_initial_iterations)
    expert_identity = hashlib.sha256(json.dumps({"planner": asdict(planner), "gamma": cfg.gamma,
        "law": "fresh independent MPPI, own shifted suffix plan, finite horizon"}, sort_keys=True).encode()).hexdigest()
    value_trainer = ExpertValueTrainer(horizon=cfg.total_steps, hidden_dim=cfg.hidden_dim, seed=cfg.seed)
    sources = ["scripts/training/train_expert_withdrawal.py", "src/gps/expert_continuation.py",
        "src/gps/expert_value.py", "src/gps/prefix_search.py", "src/gps/policy.py", "src/mppi/mppi.py",
        "src/rl/walker.py", "src/rl/sac.py", "src/envs/base.py", "src/envs/walker2d.py", "src/envs/mujoco_env.py",
        "src/utils/config.py", "src/utils/math.py", "configs/walker2d_best.json",
        "configs/walker2d_task.json", "assets/walker2d.xml", "scripts/runners/eval_policy_walker2d.py",
        "scripts/runners/run_walker2d.py"]
    hashes = {name: _hash(ROOT / name) for name in sources}
    for name in sources:
        destination = out_dir / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write_json(out_dir / "protocol.json", {"config": asdict(cfg), "planner": asdict(planner),
        "source_sha256": hashes, "environment": provenance(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "objective": "Own policy prefix then independent fresh-plan MPPI for exactly N-L controls; finite discounted raw task cost. Expert cost is not Vpi or standalone return.",
        "policy_update": "Antithetic parameter-space ES; frozen expert value, or exact takeover scoring if healthy fit/calibration data are unavailable; actual paired takeovers must improve on both training and separate verification starts. No action-cloning loss.",
        "curriculum": "Advance short stages only when actual verification cost is within declared tolerance of pure MPPI. Exhaustion stalls honestly. The final full-N stage uses no expert/value and runs its configured updates.",
        "interpretation": "Implementation/feasibility run, not an exploration or sample-efficiency benchmark."})
    torch.save(actor.policy_state(), out_dir / "policy_initial.pt")
    initial = _standalone(actor, out_dir, "initial", cfg)
    ledger = {"standalone_evaluation_controls": sum(e["steps"] for e in initial["episodes"])}
    status = {"phase": "training", "completed_updates": 0, "curriculum_complete": False}
    write_json(out_dir / "status.json", status)
    started = time.perf_counter()
    executor = ProcessPoolExecutor(cfg.workers, mp_context=mp.get_context("spawn")) if cfg.workers > 1 else None
    mapper: Callable = (lambda jobs: list(executor.map(_job, jobs))) if executor else (lambda jobs: list(map(_job, jobs)))
    iteration, stalled = 0, False

    def evaluate(jobs: list[dict[str, Any]]) -> list[Any]:
        results = mapper(jobs)
        for result in results:
            _ledger_add(ledger, result.ledger)
        return results

    try:
        for stage, prefix_steps in enumerate(cfg.prefix_schedule):
            advanced = False
            for stage_update in range(cfg.updates_per_stage):
                status.update(stage=stage, prefix_steps=prefix_steps, stage_update=stage_update)
                write_json(out_dir / "status.json", status)
                base_seed = 11_000_000 + cfg.seed * 100_000
                train = _starts(base_seed + iteration * cfg.train_starts, cfg.train_starts, cfg.total_steps)
                calibration = _starts(base_seed + 20_000 + iteration * cfg.calibration_starts,
                                      cfg.calibration_starts, cfg.total_steps)
                verification = _starts(base_seed + 40_000 + iteration * cfg.verification_starts,
                                       cfg.verification_starts, cfg.total_steps)
                candidates = antithetic_candidates(actor, pairs=cfg.pairs, sigma=cfg.parameter_sigma,
                                                   seed=cfg.seed * 100_000 + iteration)
                policies = [copy.deepcopy(actor)] + [candidates.actor_copy(actor, i) for i in range(2 * cfg.pairs)]
                common = {"gamma": cfg.gamma, "planner": planner, "total_steps": cfg.total_steps, "steps": prefix_steps}
                jobs = [{**common, "kind": "prefix", "snapshot": snapshot, "policy": policy.policy_state()}
                        for policy in policies for snapshot in train]
                prefixes = evaluate(jobs)
                fit_log = None
                snapshot_value = None
                exact_suffixes = None
                scoring_mode = "standalone_exact" if prefix_steps == cfg.total_steps else "expert_value"
                unavailable_reason = None
                if prefix_steps < cfg.total_steps:
                    eligible = [i for i, prefix in enumerate(prefixes) if prefix.healthy]
                    selected = [eligible[i] for i in np.linspace(0, len(eligible) - 1,
                        min(cfg.value_labels, len(eligible)), dtype=int)] if eligible else []
                    held_prefixes = evaluate([{**common, "kind": "prefix", "snapshot": s,
                        "policy": actor.policy_state()} for s in calibration]) if len(selected) >= 2 else []
                    held_eligible = [i for i, p in enumerate(held_prefixes) if p.healthy]
                    if len(selected) >= 2 and held_eligible:
                        suffix_jobs = [{**common, "kind": "teacher", "snapshot": prefixes[i].snapshot,
                            "steps": cfg.total_steps - prefix_steps,
                            "planner_seed": 20_000_000 + cfg.seed * 100_000 + iteration * 1000 + i % cfg.train_starts}
                            for i in selected]
                        suffixes = evaluate(suffix_jobs)
                        held_suffixes = evaluate([{**common, "kind": "teacher", "snapshot": held_prefixes[i].snapshot,
                            "steps": cfg.total_steps - prefix_steps,
                            "planner_seed": 30_000_000 + cfg.seed * 100_000 + iteration * 1000 + i}
                            for i in held_eligible])
                        training_ids = [base_seed + iteration * cfg.train_starts + i % cfg.train_starts for i in selected]
                        held_ids = [base_seed + 20_000 + iteration * cfg.calibration_starts + i for i in held_eligible]
                        def dataset(rows, targets, ids):
                            return ExpertValueData(np.asarray([p.endpoint_obs for p in rows]),
                                np.asarray([p.cost for p in targets]), np.full(len(rows), cfg.total_steps - prefix_steps),
                                np.asarray(ids), source=EXPERT_SOURCE,
                                provenance={"planner": json.dumps(asdict(planner)), "expert_id": expert_identity})
                        selected_prefixes = [prefixes[i] for i in selected]
                        training_data = dataset(selected_prefixes, suffixes, training_ids)
                        held_data = dataset([held_prefixes[i] for i in held_eligible], held_suffixes, held_ids)
                        fit_log = value_trainer.fit(training_data, held_data, updates=cfg.value_updates,
                            batch_size=cfg.value_batch_size, seed=cfg.seed * 1000 + iteration)
                        snapshot_value = value_trainer.snapshot()
                        np.savez_compressed(out_dir / f"expert_data_{iteration:04d}.npz",
                            observations=training_data.observations, targets=training_data.cost_targets,
                            remaining=training_data.remaining, reset_ids=training_data.reset_ids,
                            states=np.asarray([p.endpoint_state for p in selected_prefixes]),
                            warmstarts=np.asarray([p.endpoint_warmstart for p in selected_prefixes]),
                            planner_seeds=np.asarray([j["planner_seed"] for j in suffix_jobs]),
                            holdout_observations=held_data.observations, holdout_targets=held_data.cost_targets,
                            holdout_remaining=held_data.remaining, holdout_reset_ids=held_data.reset_ids,
                            holdout_states=np.asarray([held_prefixes[i].endpoint_state for i in held_eligible]),
                            holdout_warmstarts=np.asarray([held_prefixes[i].endpoint_warmstart for i in held_eligible]),
                            holdout_planner_seeds=np.asarray([30_000_000 + cfg.seed * 100_000 + iteration * 1000 + i
                                                            for i in held_eligible]),
                            source=EXPERT_SOURCE, expert_id=expert_identity, gamma=cfg.gamma,
                            total_steps=cfg.total_steps)
                        torch.save(value_trainer.state_dict(), out_dir / f"expert_value_{iteration:04d}.pt")
                    else:
                        # A fall supplies an exact finite return, not grounds for
                        # aborting ES. Healthy survivors get an actual expert tail
                        # when a held-out value fit cannot yet be grounded.
                        unavailable_reason = ("insufficient healthy expert-value training endpoints"
                            if len(selected) < 2 else "no healthy expert-value calibration endpoints")
                        scoring_mode = "exact_takeover"
                        exact_jobs = [{**common, "kind": "teacher", "snapshot": prefixes[i].snapshot,
                            "steps": cfg.total_steps - prefixes[i].steps,
                            "planner_seed": 20_000_000 + cfg.seed * 100_000 + iteration * 1000 + i % cfg.train_starts}
                            for i in eligible]
                        exact_suffixes = dict(zip(eligible, evaluate(exact_jobs)))
                estimates = candidate_objectives(prefixes, total_steps=cfg.total_steps,
                    value=snapshot_value, exact_suffixes=exact_suffixes).reshape(
                        len(policies), cfg.train_starts).mean(axis=1)
                proposal = propose_es_update(candidates, estimates[1:], learning_rate=cfg.parameter_lr,
                    max_step_norm=cfg.max_step_norm, normalize_costs=cfg.normalize_costs)
                proposed = candidates.space.actor_copy(actor, proposal.proposed_parameters)

                def measured(policy, starts, offset):
                    return evaluate([{**common, "kind": "takeover", "snapshot": s,
                        "policy": policy.policy_state(),
                        "planner_seed": 40_000_000 + cfg.seed * 100_000 + iteration * 1000 + offset + i}
                        for i, s in enumerate(starts)])
                # Verification uses fresh roots, matching reset/teacher RNG in both branches.
                old_train, new_train = measured(actor, train, 0), measured(proposed, train, 0)
                old_verify, new_verify = measured(actor, verification, 100), measured(proposed, verification, 100)
                means = [float(np.mean([r.cost for r in rows])) for rows in (old_train, new_train, old_verify, new_verify)]
                train_check = verify_parameter_proposal(proposal, base_cost=means[0], proposed_cost=means[1])
                verify_check = verify_parameter_proposal(proposal, base_cost=means[2], proposed_cost=means[3])
                accepted = train_check.accepted and verify_check.accepted
                if accepted:
                    actor = proposed
                current_verification = means[3] if accepted else means[2]
                teacher_reference = None
                if prefix_steps < cfg.total_steps:
                    reference = evaluate([{**common, "kind": "teacher", "snapshot": s,
                        "steps": cfg.total_steps,
                        "planner_seed": 40_000_000 + cfg.seed * 100_000 + iteration * 1000 + 100 + i}
                        for i, s in enumerate(verification)])
                    teacher_reference = float(np.mean([r.cost for r in reference]))
                    advanced = current_verification <= teacher_reference + cfg.teacher_tolerance * max(1., teacher_reference)
                else:
                    advanced = stage_update + 1 == cfg.updates_per_stage
                checkpoint = out_dir / f"policy_{iteration:04d}.pt"
                torch.save(actor.policy_state(), checkpoint)
                row = {"iteration": iteration, "stage": stage, "prefix_steps": prefix_steps,
                    "accepted": bool(accepted), "train_base_cost": means[0], "train_proposed_cost": means[1],
                    "verification_base_cost": means[2], "verification_proposed_cost": means[3],
                    "teacher_reference_cost": teacher_reference, "stage_advanced": bool(advanced),
                    "estimated_base_cost": float(estimates[0]), "candidate_costs": estimates[1:].tolist(),
                    "parameter_step_norm": proposal.step_norm, "expert_value_fit": fit_log,
                    "scoring_mode": scoring_mode, "expert_value_unavailable_reason": unavailable_reason,
                    "expert_assistance": prefix_steps < cfg.total_steps,
                    "checkpoint_sha256": _hash(checkpoint), "ledger_cumulative": dict(ledger)}
                with (out_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
                iteration += 1
                status.update(completed_updates=iteration, ledger=ledger, elapsed_seconds=time.perf_counter() - started)
                write_json(out_dir / "status.json", status)
                if advanced:
                    break
            if stalled or not advanced:
                status.setdefault("stall_reason", "stage exhausted without verified expert-withdrawal criterion")
                stalled = True
                break
        torch.save(actor.policy_state(), out_dir / "policy_final.pt")
        final = _standalone(actor, out_dir, "final", cfg)
        ledger["standalone_evaluation_controls"] += sum(e["steps"] for e in final["episodes"])
        unchanged = all(_hash(ROOT / name) == value for name, value in hashes.items())
        if not unchanged:
            raise RuntimeError("Executed source changed during the run")
        status.update(phase="complete", curriculum_complete=not stalled, source_unchanged=True,
            standalone_walking_gate=final["walking_gate"], final_summary=final["summary"],
            ledger=ledger, elapsed_seconds=time.perf_counter() - started)
        write_json(out_dir / "status.json", status)
    except BaseException as exc:
        status.update(phase="failed", error=repr(exc))
        write_json(out_dir / "status.json", status)
        raise
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    tyro.cli(run)
