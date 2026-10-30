"""Option 1: independent MPPI experience, learner Q evaluation, verified policy search.

MPPI supplies transitions and reachable starts, never the learner's continuation.
The critic evaluates the frozen base policy. It proposes parameter changes;
complete candidate-policy returns decide acceptance. No action cloning or SAC
actor gradient is used. Workers exchange bytes/NumPy, never shared Torch tensors.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
import tyro

from scripts.runners.eval_policy_walker2d import evaluate_policy, provenance, write_json
from src.gps.expert_continuation import Snapshot, finite_absorbing_cost, raw_teacher_config, rollout_policy_prefix
from src.gps.learner_value import LearnerValueTrainer
from src.gps.mppi_experience import collect_teacher_episode, combine_transitions, finite_policy_rollout
from src.gps.policy import TanhMLPPolicy
from src.gps.prefix_search import antithetic_candidates, propose_es_update
from src.rl.sac import load_policy
from src.rl.walker import WalkerRL

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Option1Config:
    total_steps: int = 1000
    handover_schedule: tuple[int, ...] = (900, 500, 0)
    updates_per_stage: int = 40
    root_updates: int = 120
    pairs: int = 32
    teacher_train_episodes: int = 6
    teacher_holdout_episodes: int = 2
    root_train_starts: int = 4
    root_verify_starts: int = 4
    critic_prefix: int = 32
    critic_updates: int = 1000
    critic_batch_size: int = 128
    hidden_dim: int = 64
    critic_hidden_dim: int = 128
    parameter_sigma: float = .025
    parameter_lr: float = .03
    max_step_norm: float = .25
    root_weight: float = .5
    gamma: float = .999
    teacher_K: int = 32
    teacher_H: int = 48
    teacher_initial_iterations: int = 2
    seed: int = 0
    workers: int = 8
    eval_episodes: int = 100
    eval_every: int = 20


def validate(cfg: Option1Config) -> None:
    for name, value in asdict(cfg).items():
        if name in ("handover_schedule", "parameter_sigma", "parameter_lr", "max_step_norm", "root_weight", "gamma", "seed"):
            continue
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (not cfg.handover_schedule or cfg.handover_schedule[-1] != 0
            or tuple(sorted(set(cfg.handover_schedule), reverse=True)) != cfg.handover_schedule
            or any(type(b) is not int or b < 0 or b >= cfg.total_steps for b in cfg.handover_schedule)):
        raise ValueError("Handover times must strictly decrease to zero within the episode")
    if cfg.teacher_H < 2 or type(cfg.seed) is not int or not 0 <= cfg.seed < 100:
        raise ValueError("Invalid teacher horizon or seed")
    if not 0 < cfg.root_weight <= 1 or not np.isfinite(cfg.gamma) or not 0 < cfg.gamma < 1:
        raise ValueError("Invalid mixture weight or finite discount")
    for name in ("parameter_sigma", "parameter_lr", "max_step_norm"):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"Invalid {name}")
    updates = (len(cfg.handover_schedule) - 1) * cfg.updates_per_stage + cfg.root_updates
    if max(updates * cfg.root_train_starts, updates * cfg.root_verify_starts,
           cfg.teacher_train_episodes + cfg.teacher_holdout_episodes, cfg.eval_episodes) > 90_000:
        raise ValueError("Acquisition must fit disjoint reset blocks")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack_policy(actor) -> bytes:
    stream = io.BytesIO(); torch.save(actor.policy_state(), stream)
    return stream.getvalue()


def _unpack_policy(packet: bytes):
    if not isinstance(packet, bytes):
        raise TypeError("Worker policy payload must be bytes, not shared tensors")
    return load_policy(torch.load(io.BytesIO(packet), map_location="cpu", weights_only=True)).eval()


def _job(job: dict):
    torch.set_num_threads(1)
    if job["kind"] == "teacher":
        if "policy" in job:
            raise ValueError("The independent teacher must not receive a policy")
        return collect_teacher_episode(**{k: v for k, v in job.items() if k != "kind"})
    actor = _unpack_policy(job["policy"])
    if job["snapshot"] is None:
        # A guide episode that already fell contributes its absorbing cost,
        # without inventing a valid learner start or dropping the episode.
        cost = finite_absorbing_cost(job["gamma"], job["remaining"])
        prefix = SimpleNamespace(cost=0., steps=0, terminated=True, gamma=job["gamma"],
                                 unhealthy_cost=5., healthy=False, endpoint_obs=np.zeros(17))
        return SimpleNamespace(cost=cost, prefix=prefix, transitions=None, ledger={})
    if job["kind"] == "prefix":
        return rollout_policy_prefix(job["snapshot"], actor,
            steps=min(job["steps"], job["remaining"]), gamma=job["gamma"])
    if job["kind"] != "policy":
        raise ValueError("Unknown rollout kind")
    return finite_policy_rollout(job["snapshot"], actor, remaining=job["remaining"],
                                 gamma=job["gamma"], reset_id=job["reset_id"])


def candidate_score(prefix, *, total_steps: int, value, base_actor) -> float:
    remaining = total_steps - prefix.steps
    if remaining < 0:
        raise ValueError("Candidate prefix exceeds its remaining horizon")
    if prefix.terminated:
        tail = finite_absorbing_cost(prefix.gamma, remaining, prefix.unhealthy_cost)
    elif remaining == 0:
        tail = 0.
    else:
        if value is None:
            raise RuntimeError("Healthy unfinished candidates require a grounded learner critic")
        action = base_actor.act(prefix.endpoint_obs, deterministic=True)
        tail = float(value.predict(prefix.endpoint_obs[None], action[None], remaining)[0])
    cost = float(prefix.cost + prefix.gamma ** prefix.steps * tail)
    if not np.isfinite(cost):
        raise FloatingPointError("Nonfinite candidate score")
    return cost


def _root_starts(seed: int, count: int, horizon: int) -> list[dict]:
    env = WalkerRL(max_episode_steps=horizon)
    try:
        output = []
        for i in range(count):
            env.reset(seed=seed + i)
            output.append(dict(snapshot=Snapshot(env.walker.get_state(), env.walker.get_warmstart()),
                               remaining=horizon, reset_id=seed + i, role="root"))
        return output
    finally:
        env.close()


def _guide_starts(episodes, handover: int, horizon: int) -> list[dict]:
    output = []
    for episode in episodes:
        alive = handover < len(episode.states) and (handover < episode.prefix.steps or not episode.prefix.terminated)
        snapshot = Snapshot(episode.states[handover], episode.warmstarts[handover]) if alive else None
        output.append(dict(snapshot=snapshot, remaining=horizon - handover,
                           reset_id=episode.reset_id, role="guide"))
    return output


def _mixture(costs: list[float], starts: list[dict], cfg: Option1Config) -> float:
    groups = {}
    for cost, start in zip(costs, starts):
        scale = max(1., finite_absorbing_cost(cfg.gamma, start["remaining"]))
        groups.setdefault(start["role"], []).append(cost / scale)
    root = float(np.mean(groups["root"]))
    return root if "guide" not in groups else cfg.root_weight * root + (1 - cfg.root_weight) * float(np.mean(groups["guide"]))


def _summary(rows, starts) -> dict:
    return {role: {"mean_cost": float(np.mean([r.cost for r, s in zip(rows, starts) if s["role"] == role])),
                   "episodes": sum(s["role"] == role for s in starts)} for role in {s["role"] for s in starts}}


def _standalone(actor, out: Path, name: str, cfg: Option1Config) -> dict:
    trace_path = out / f"{name}.npz"
    if trace_path.exists() and not (out / f"{name}.json").exists():
        # Keep a trace left by an interrupted evaluation instead of overwriting.
        trace_path.rename(out / f"{name}.partial_{time.time_ns()}.npz")
    report = evaluate_policy(actor, gamma=cfg.gamma, episodes=cfg.eval_episodes,
        steps=cfg.total_steps, seed=12_900_000 + cfg.seed * 1_000_000, traces_path=out / f"{name}.npz")
    with np.load(out / f"{name}.npz") as data:
        values = []
        for row, lo, hi in zip(report["episodes"], data["episode_step_offsets"][:-1], data["episode_step_offsets"][1:]):
            raw = data["costs"][lo:hi]
            cost = float(np.dot(cfg.gamma ** np.arange(len(raw)), raw))
            if row["terminated"]:
                cost += cfg.gamma ** len(raw) * finite_absorbing_cost(cfg.gamma, cfg.total_steps - len(raw))
            values.append(cost)
    report["summary"].update(finite_discounted_cost=float(np.mean(values)),
                              mean_duration_seconds=.008 * report["summary"]["mean_steps"])
    report.update(standalone=True, walking_gate=(report["summary"]["survival_rate"] >= .9 and
        report["summary"]["velocity_mae"] <= .3) if cfg.total_steps == 1000 else None)
    write_json(out / f"{name}.json", report)
    return report


def run(out_dir: Path, cfg: Option1Config, resume: bool = False) -> None:
    validate(cfg); out_dir = out_dir.resolve(); torch.set_num_threads(1)
    if resume:
        protocol = json.loads((out_dir / "protocol.json").read_text())
        if json.loads(json.dumps(asdict(cfg))) != protocol["config"]:
            raise ValueError("Resume configuration differs from the frozen run")
        hashes = protocol["source_sha256"]
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        sources = ["scripts/training/train_mppi_experience.py", "src/gps/mppi_experience.py",
            "src/gps/learner_value.py", "src/gps/expert_continuation.py", "src/gps/prefix_search.py",
            "src/gps/policy.py", "src/mppi/mppi.py", "src/rl/walker.py", "src/rl/sac.py",
            "src/envs/base.py", "src/envs/walker2d.py", "src/envs/mujoco_env.py", "src/utils/config.py",
            "src/utils/math.py", "configs/walker2d_best.json", "configs/walker2d_task.json",
            "assets/walker2d.xml", "scripts/runners/eval_policy_walker2d.py", "scripts/runners/run_walker2d.py"]
        hashes = {name: _sha(ROOT / name) for name in sources}
        for name in sources:
            dest = out_dir / "source" / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        write_json(out_dir / "protocol.json", dict(config=asdict(cfg), source_sha256=hashes,
            environment=provenance(), git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            objective="Learner executes entire suffix from ordinary and independent-teacher starts; finite discounted raw costs. Early fixed stages use horizon-normalized root/guide mixture; final stage uses roots only.",
            critic="Mixed executed one-step transitions; all bootstraps follow frozen current learner, never recorded next teacher actions or full teacher returns. Twin max cost; approximation only.",
            search="Own candidate prefix then frozen BASE-policy critic tail proposes ES and best sampled candidate. Actual complete CANDIDATE-policy returns on train/heldout panels decide acceptance. No cloning or actor backpropagation.",
            schedule="Fixed stage budgets, not mastery claims; ordinary-root responsibility at the final stage. Guide/root interference is measured and may occur early.",
            interpretation="One capability trial; not a controlled exploration/sample-efficiency benchmark. Saved guide resets are simulator interventions."))
    if not all(_sha(ROOT / n) == h for n, h in hashes.items()):
        raise ValueError("Executed source differs from the frozen protocol")
    env = WalkerRL()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            actor = TanhMLPPolicy(17, *env.walker.action_bounds, hidden_dim=cfg.hidden_dim)
    finally:
        env.close()
    critic = LearnerValueTrainer(horizon=cfg.total_steps, gamma=cfg.gamma,
                                 hidden_dim=cfg.critic_hidden_dim, seed=cfg.seed)
    rows = [json.loads(s) for s in (out_dir / "metrics.jsonl").read_text().splitlines()] if resume and (out_dir / "metrics.jsonl").exists() else []
    ledger = {}; iteration = len(rows); accumulated = 0.
    if rows:
        state = torch.load(out_dir / f"state_{iteration - 1:04d}.pt", map_location="cpu", weights_only=False)
        actor = load_policy(state["policy"]).requires_grad_(True)
        critic.load_state_dict(state["critic"]); ledger = state["ledger"]; accumulated = state["elapsed_seconds"]
    if resume:
        previous = json.loads((out_dir / "status.json").read_text()) if (out_dir / "status.json").exists() else None
        with (out_dir / "recovery_events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(previous_status=previous, resume_completed_updates=len(rows),
                note="Resuming last committed update. Work in an interrupted in-flight batch is unknown and may repeat; committed control ledger is a lower bound across interruptions.")) + "\n")
        reports = [out_dir / "initial.json", *out_dir.glob("standalone_*.json"), out_dir / "final.json"]
        ledger["standalone_evaluation_controls"] = sum(
            sum(e["steps"] for e in json.loads(p.read_text())["episodes"]) for p in reports if p.exists())
    started = time.perf_counter()
    status = dict(phase="starting", completed_updates=iteration, resumed=resume)
    write_json(out_dir / "status.json", status)
    executor = ProcessPoolExecutor(cfg.workers, mp_context=mp.get_context("spawn")) if cfg.workers > 1 else None

    def evaluate(jobs):
        results = list(executor.map(_job, jobs, chunksize=1)) if executor else list(map(_job, jobs))
        for result in results:
            for key, value in result.ledger.items():
                ledger[key] = ledger.get(key, 0) + int(value)
        return results

    def measured(policy, starts, kind="policy"):
        packet = _pack_policy(policy)
        return evaluate([dict(kind=kind, policy=packet, gamma=cfg.gamma, steps=cfg.critic_prefix,
            **{k: v for k, v in start.items() if k != "role"}) for start in starts])

    try:
        if not (out_dir / "initial.json").exists():
            torch.save(actor.policy_state(), out_dir / "policy_initial.pt")
            initial = _standalone(actor, out_dir, "initial", cfg)
            ledger["standalone_evaluation_controls"] = sum(r["steps"] for r in initial["episodes"])
        archive = out_dir / "teacher_episodes.pkl"
        if archive.exists():
            # Only this run's own trusted generated archive, never an external pickle.
            with archive.open("rb") as stream:
                episodes = pickle.load(stream)
            if resume and not rows:
                for episode in episodes:
                    for key, value in episode.ledger.items():
                        ledger[key] = ledger.get(key, 0) + int(value)
        else:
            status["phase"] = "collecting_independent_teacher"; write_json(out_dir / "status.json", status)
            planner = replace(raw_teacher_config(), K=cfg.teacher_K, H=cfg.teacher_H,
                              initial_iterations=cfg.teacher_initial_iterations)
            episodes = evaluate([dict(kind="teacher", reset_seed=12_000_000 + cfg.seed * 1_000_000 + i,
                total_steps=cfg.total_steps, planner_seed=31_000_000 + cfg.seed * 1000 + i,
                planner=planner, gamma=cfg.gamma) for i in range(cfg.teacher_train_episodes + cfg.teacher_holdout_episodes)])
            temporary_archive = archive.with_suffix(".tmp")
            with temporary_archive.open("wb") as stream:
                pickle.dump(episodes, stream, protocol=5)
            temporary_archive.replace(archive)
        teacher_train, teacher_hold = episodes[:cfg.teacher_train_episodes], episodes[cfg.teacher_train_episodes:]
        schedule = [b for b in cfg.handover_schedule for _ in range(cfg.root_updates if b == 0 else cfg.updates_per_stage)]
        for iteration in range(iteration, len(schedule)):
            handover = schedule[iteration]
            status.update(phase="training", iteration=iteration, handover=handover)
            write_json(out_dir / "status.json", status)
            train = _root_starts(12_100_000 + cfg.seed * 1_000_000 + iteration * cfg.root_train_starts,
                                 cfg.root_train_starts, cfg.total_steps)
            hold = _root_starts(12_200_000 + cfg.seed * 1_000_000 + iteration * cfg.root_verify_starts,
                                cfg.root_verify_starts, cfg.total_steps)
            if handover:
                train += _guide_starts(teacher_train, handover, cfg.total_steps)
                hold += _guide_starts(teacher_hold, handover, cfg.total_steps)
            old_train, old_hold = measured(actor, train), measured(actor, hold)
            training_data = combine_transitions([e.transitions for e in teacher_train] + [r.transitions for r in old_train])
            hold_data = combine_transitions([e.transitions for e in teacher_hold] + [r.transitions for r in old_hold])
            # Preserve the actual mixed replay and reset split for target and
            # support audits, rather than retaining only loss summaries.
            replay_arrays = {}
            for split, data in (("train", training_data), ("heldout", hold_data)):
                for name in ("observations", "actions", "costs", "next_observations", "remaining",
                             "terminated", "source", "reset_ids"):
                    replay_arrays[f"{split}_{name}"] = getattr(data, name)
            np.savez_compressed(out_dir / f"replay_{iteration:04d}.npz", **replay_arrays)
            fit = critic.fit(training_data, hold_data, actor, updates=cfg.critic_updates,
                batch_size=cfg.critic_batch_size, seed=cfg.seed * 100_000 + iteration)
            value = critic.snapshot()
            calibration = []
            for result, start in zip(old_hold, hold):
                if start["snapshot"] is not None:
                    obs = result.observations[0]
                    prediction = value.predict(obs[None], actor.act(obs)[None], start["remaining"])[0]
                    calibration.append(float(prediction - result.cost))
            mc = dict(rmse=float(np.sqrt(np.mean(np.square(calibration)))),
                      bias=float(np.mean(calibration)), episodes=len(calibration),
                      errors=calibration,
                      meaning="Current-learner full-rollout calibration, not teacher returns")
            candidates = antithetic_candidates(actor, pairs=cfg.pairs, sigma=cfg.parameter_sigma,
                                               seed=cfg.seed * 100_000 + iteration)
            estimates = []
            for i in range(2 * cfg.pairs):
                candidate = candidates.actor_copy(actor, i)
                prefixes = measured(candidate, train, kind="prefix")
                costs = [candidate_score(getattr(p, "prefix", p), total_steps=s["remaining"],
                    value=value, base_actor=actor) for p, s in zip(prefixes, train)]
                estimates.append(_mixture(costs, train, cfg))
            proposal = propose_es_update(candidates, estimates, learning_rate=cfg.parameter_lr,
                max_step_norm=cfg.max_step_norm, normalize_costs=True)
            choices = [("es", candidates.space.actor_copy(actor, proposal.proposed_parameters)),
                       ("sample", candidates.actor_copy(actor, int(np.argmin(estimates))))]
            base_train = _mixture([r.cost for r in old_train], train, cfg)
            base_hold = _mixture([r.cost for r in old_hold], hold, cfg)
            accepted, accepted_kind, candidate_records = False, None, []
            best_hold = base_hold
            for kind, candidate in choices:
                new_train, new_hold = measured(candidate, train), measured(candidate, hold)
                train_cost = _mixture([r.cost for r in new_train], train, cfg)
                hold_cost = _mixture([r.cost for r in new_hold], hold, cfg)
                valid = train_cost < base_train and hold_cost < base_hold
                candidate_records.append(dict(kind=kind, train_cost=train_cost, heldout_cost=hold_cost,
                    passes=bool(valid), train_components=_summary(new_train, train),
                    heldout_components=_summary(new_hold, hold)))
                if valid and hold_cost < best_hold:
                    best_actor, best_hold, accepted_kind = candidate, hold_cost, kind
                    accepted = True
            if accepted:
                actor = best_actor
            checkpoint = out_dir / f"policy_{iteration:04d}.pt"
            torch.save(actor.policy_state(), checkpoint)
            row = dict(iteration=iteration, handover=handover, accepted=accepted, accepted_kind=accepted_kind,
                train_base_cost=base_train, heldout_base_cost=base_hold, heldout_current_cost=best_hold,
                train_components=_summary(old_train, train), heldout_components=_summary(old_hold, hold),
                proposals=candidate_records, candidate_scores=estimates, critic_fit=fit, learner_mc_calibration=mc,
                planner_takeovers=0, teacher_return_targets=False, checkpoint_sha256=_sha(checkpoint),
                active_training_guides=sum(s["role"] == "guide" and s["snapshot"] is not None for s in train),
                active_heldout_guides=sum(s["role"] == "guide" and s["snapshot"] is not None for s in hold),
                ledger_cumulative=dict(ledger), elapsed_seconds=accumulated + time.perf_counter() - started)
            # State is committed before its metrics row. A reboot before the row
            # simply repeats that unfinished update from the last committed state.
            state_path = out_dir / f"state_{iteration:04d}.pt"
            temporary = state_path.with_suffix(".tmp")
            torch.save(dict(policy=actor.policy_state(), critic=critic.state_dict(), ledger=dict(ledger),
                            elapsed_seconds=row["elapsed_seconds"]), temporary)
            temporary.replace(state_path)
            with (out_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
            print(json.dumps({k: row[k] for k in ("iteration", "handover", "accepted", "accepted_kind", "train_base_cost", "heldout_base_cost", "heldout_current_cost", "elapsed_seconds")}), flush=True)
            status.update(completed_updates=iteration + 1, accepted_updates=sum(r["accepted"] for r in rows) + accepted,
                          ledger=dict(ledger), elapsed_seconds=row["elapsed_seconds"])
            rows.append(row)
            write_json(out_dir / "status.json", status)
            if (iteration + 1) % cfg.eval_every == 0:
                name = f"standalone_{iteration + 1:04d}"
                if not (out_dir / f"{name}.json").exists():
                    report = _standalone(actor, out_dir, name, cfg)
                    ledger["standalone_evaluation_controls"] += sum(r["steps"] for r in report["episodes"])
        torch.save(actor.policy_state(), out_dir / "policy_final.pt")
        if not (out_dir / "final.json").exists():
            final = _standalone(actor, out_dir, "final", cfg)
            ledger["standalone_evaluation_controls"] += sum(r["steps"] for r in final["episodes"])
        else:
            final = json.loads((out_dir / "final.json").read_text())
        unchanged = all(_sha(ROOT / n) == h for n, h in hashes.items())
        if not unchanged:
            raise RuntimeError("Frozen source changed during training")
        status.update(phase="complete", source_unchanged=True, completed_updates=len(schedule),
            walking_gate=final["walking_gate"], final_summary=final["summary"], ledger=ledger)
        write_json(out_dir / "status.json", status)
    except BaseException as exc:
        status.update(phase="failed", error=repr(exc), ledger=ledger)
        write_json(out_dir / "status.json", status)
        raise
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    tyro.cli(run)
