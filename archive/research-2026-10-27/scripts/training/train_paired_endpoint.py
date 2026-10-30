"""Independent MPPI with training-only paired endpoint acquisition.

Both deployed controllers share the exact same evolving action archive. Only
training collection and training reference queries enter that archive. A small
ridge value learns complete canonical-reference returns and paired contrasts;
fresh decision-pair calibration and full-controller qualification never fit it.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import torch
import tyro

from scripts.runners.eval_policy_walker2d import provenance
from scripts.training.train_coupled_value import (
    _json, _save, _sha, _revision, _reset, _mppi, _summary, _public, _promote,
)
from src.gps.coupled_planner import MemoryArchive
from src.gps.expert_continuation import Snapshot, _restore, finite_absorbing_cost
from src.gps.paired_planner import PairedEndpointPlanner
from src.gps.paired_value import PairedRidgeValue
from src.mppi.mppi import MPPI
from src.rl.walker import WalkerRL

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = dict(bootstrap=30_000_000, collection=31_000_000,
              calibration=33_000_000, qualification=34_000_000,
              evaluation=35_000_000, final=36_000_000)
SOURCES = [
    "scripts/training/train_paired_endpoint.py", "src/gps/paired_value.py",
    "src/gps/paired_planner.py", "scripts/training/train_coupled_value.py",
    "src/gps/coupled_planner.py", "src/gps/global_value.py",
    "src/gps/expert_continuation.py", "src/mppi/mppi.py", "src/rl/walker.py",
    "src/envs/base.py", "src/envs/walker2d.py", "src/envs/mujoco_env.py",
    "src/utils/config.py", "src/utils/math.py", "configs/walker2d_best.json",
    "configs/walker2d_task.json", "assets/walker2d.xml",
    "scripts/runners/eval_policy_walker2d.py",
    "scripts/diagnostics/compare_paired_endpoint.py", "docs/paired_endpoint_study.md",
]


@dataclass
class PairedConfig:
    iterations: int = 24
    total_steps: int = 1000
    seed: int = 0
    workers: int = 6
    bootstrap_episodes: int = 4
    collection_episodes: int = 2
    train_pairs: int = 8
    continuation_repeats: int = 2
    calibration_episodes: int = 2
    calibration_pairs: int = 4
    qualification_episodes: int = 6
    evaluation_episodes: int = 16
    final_episodes: int = 64
    eval_every: int = 4
    K: int = 32
    H: int = 48
    memory_candidates: int = 8
    memory_stride: int = 16
    memory_entries: int = 2048
    query_stride: int = 64
    ridge: float = 10.
    alpha_max: float = .5
    alpha_step: float = .1
    support_radius: float = 1.5
    max_tail_shift_temperatures: float = 2.
    pair_rmse_max: float = .2
    promotion_margin: float = 1.
    gamma: float = .999
    angle_noise: float = .03
    velocity_noise: float = .10


def validate(cfg: PairedConfig) -> None:
    floats = {"ridge", "alpha_max", "alpha_step", "support_radius",
              "max_tail_shift_temperatures", "pair_rmse_max", "promotion_margin",
              "gamma", "angle_noise", "velocity_noise"}
    for name, value in asdict(cfg).items():
        if name not in floats and (type(value) is not int or value < (0 if name == "seed" else 1)):
            raise ValueError(f"{name} must be an integer in its allowed range")
    if cfg.seed >= 10 or not 2 <= cfg.H <= cfg.total_steps:
        raise ValueError("Require seed<10 and 2<=H<=total_steps")
    if cfg.continuation_repeats < 2 or cfg.calibration_episodes < 2 or cfg.calibration_pairs < 2:
        raise ValueError("Need repeated continuations and at least two fresh calibration roots/pairs")
    if cfg.train_pairs < cfg.collection_episodes:
        raise ValueError("Training pair budget must cover collection roots")
    if not 0 < cfg.gamma < 1 or not 0 < cfg.alpha_step <= cfg.alpha_max <= 1:
        raise ValueError("Invalid finite discount or influence schedule")
    if any(not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0 for key in
           ("ridge", "support_radius", "max_tail_shift_temperatures", "pair_rmse_max")):
        raise ValueError("Invalid regression, support or correction bound")
    if any(not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0 for key in
           ("promotion_margin", "angle_noise", "velocity_noise")):
        raise ValueError("Invalid nonnegative task settings")
    if cfg.iterations * max(cfg.collection_episodes, cfg.calibration_episodes,
                            cfg.qualification_episodes) >= 900_000:
        raise ValueError("Reset blocks would overlap")


def _controller(archive=None, value=None, alpha=0.) -> dict:
    return dict(archive=archive, value=value, alpha=float(alpha))


def _serial(value):
    if isinstance(value, np.ndarray):
        return _serial(value.tolist())
    if isinstance(value, np.generic):
        return _serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(item) for item in value]
    return value


def _charge(ledger: dict, values: dict, category: str = "") -> None:
    for key, count in values.items():
        name = f"{category}_{key}" if category else key
        ledger[name] = ledger.get(name, 0) + int(count)


def archive_digest(archive: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(archive.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _point(planner, snapshot, group, step, rng, shadow, mode, probe_alpha) -> dict | None:
    """Training may extrapolate for acquisition; calibration uses actual choice."""
    proposal = planner.acquisition_pair(rng, shadow_value=shadow,
        probe_alpha=probe_alpha, mode="guided" if mode == "deployed" else mode)
    if proposal is None:
        return None
    if mode == "deployed":
        bank = planner.last_decision
        index = bank["selected_index"]
        predictions, support = bank["predicted_values"], bank["supported"]
        proposal["candidate"] = dict(index=index, plan=bank["plans"][index].copy(),
            endpoint=bank["endpoints"][index].copy(), prefix_score=float(bank["local_costs"][index]),
            source=bank["sources"][index],
            predicted_tail=(float(predictions[index]) if predictions is not None
                            and np.isfinite(predictions[index]) else None),
            supported=bool(support[index]) if support is not None else None)
        baseline = bank["baseline_index"]
        proposal["baseline"] = dict(index=baseline, plan=bank["plans"][baseline].copy(),
            endpoint=bank["endpoints"][baseline].copy(), prefix_score=float(bank["local_costs"][baseline]),
            source=bank["sources"][baseline],
            predicted_tail=(float(predictions[baseline]) if predictions is not None
                            and np.isfinite(predictions[baseline]) else None),
            supported=bool(support[baseline]) if support is not None else None)
    pair = {key: proposal[key] for key in ("candidate", "baseline", "remaining", "H", "gamma_h")}
    pair.update(snapshot=snapshot, group=int(group), step=int(step),
        mode=mode, acquisition_mode=proposal["acquisition_mode"],
        probe_alpha=float(probe_alpha), candidate_index=pair["candidate"]["index"],
        baseline_index=pair["baseline"]["index"], bank_local_costs=proposal["local_costs"].copy(),
        bank_selection_scores=(planner.last_decision["scores"].copy() if mode == "deployed"
                               else proposal.get("acquisition_scores")),
        guidance_used=bool(planner.last_decision["guidance_used"]),
        action_changed=bool(planner.last_decision["action_changed"]))
    digest = hashlib.sha256()
    for array in (snapshot.state, snapshot.warmstart,
                  pair["candidate"]["plan"], pair["baseline"]["plan"]):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    digest.update(f"{group}:{pair['remaining']}".encode())
    pair["pair_id"] = digest.hexdigest()
    pair["equal_plans"] = bool(np.array_equal(pair["candidate"]["plan"], pair["baseline"]["plan"]))
    return pair


def episode(cfg: PairedConfig, controller: dict, *, reset_id: int, planner_seed: int,
            snapshot=None, remaining=None, collect_pairs=False,
            shadow_packet=None, pair_mode="train", acquisition_alpha=None, raw=False) -> dict:
    torch.set_num_threads(1)
    n = cfg.total_steps if remaining is None else int(remaining)
    env = WalkerRL(gamma=cfg.gamma, max_episode_steps=n)
    ledger = dict(executed_controls=0, candidate_controls=0, memory_controls=0,
                  verification_controls=0, planner_calls=0)
    try:
        profile = _reset(env, reset_id, cfg) if snapshot is None else "restored_query"
        if snapshot is not None:
            _restore(env, snapshot)
        archive = MemoryArchive.from_state_dict(controller["archive"]) if controller.get("archive") is not None else None
        value = PairedRidgeValue.from_packet(controller["value"]) if controller.get("value") is not None else None
        shadow = PairedRidgeValue.from_packet(shadow_packet) if shadow_packet is not None else value
        planner = MPPI(env.walker, _mppi(cfg), seed=planner_seed) if raw else PairedEndpointPlanner(
            env.walker, _mppi(cfg), archive=archive, value=value, alpha=controller["alpha"],
            seed=planner_seed, remaining=n, total_steps=cfg.total_steps, gamma=cfg.gamma,
            memory_candidates=cfg.memory_candidates, support_radius=cfg.support_radius,
            max_tail_shift_temperatures=cfg.max_tail_shift_temperatures)
        states, warms = [env.walker.get_state()], [env.walker.get_warmstart()]
        actions, costs, errors, points, sources = [], [], [], [], []
        guided = changed = skipped = 0
        rng = np.random.default_rng(planner_seed ^ 0x7A79)
        terminated = False
        for step in range(n):
            state, warm = env.walker.get_state(), env.walker.get_warmstart()
            action, info = planner.plan_step(state, initial_warmstart=warm)
            if raw:
                ledger["candidate_controls"] += cfg.K * cfg.H * int(info["optimization_iterations"])
            else:
                for key in ("candidate_controls", "memory_controls", "verification_controls"):
                    ledger[key] += int(info[key])
                guided += int(info["guidance_used"])
                changed += int(info["action_changed"])
                sources.append(info["selected_source"])
                if collect_pairs and step % cfg.query_stride == 0 and n - step > cfg.H:
                    mode = "deployed" if pair_mode == "calibration" else (
                        "guided" if (step // cfg.query_stride) % 2 == 0 else "uniform")
                    point = _point(planner, Snapshot(state, warm), reset_id, step,
                                   rng, shadow, mode,
                                   controller["alpha"] if mode == "deployed" else
                                   (min(cfg.alpha_max, controller["alpha"]+cfg.alpha_step)
                                    if acquisition_alpha is None else acquisition_alpha))
                    if point is None:
                        skipped += 1  # known fallen baseline / exhausted tail; no value claim
                    else:
                        points.append(point)
            ledger["planner_calls"] += 1
            _, _, terminated, truncated, transition = env.step(action)
            actions.append(action.copy()); costs.append(float(transition["task_cost"]))
            errors.append(abs(float(transition["x_velocity"]) - env.walker._v_target))
            states.append(env.walker.get_state()); warms.append(env.walker.get_warmstart())
            if terminated or truncated:
                break
        cost = float(np.dot(cfg.gamma ** np.arange(len(costs)), costs))
        if terminated:
            cost += cfg.gamma ** len(costs) * finite_absorbing_cost(cfg.gamma, n-len(costs))
        ledger["executed_controls"] = len(costs)
        return dict(reset_id=int(reset_id), planner_seed=int(planner_seed), reset_profile=profile,
            states=np.asarray(states), warmstarts=np.asarray(warms), actions=np.asarray(actions).reshape(-1, 6),
            costs=np.asarray(costs), gamma=cfg.gamma, remaining_start=n, terminated=bool(terminated),
            steps=len(costs), survived=bool(not terminated and len(costs) == n),
            duration_seconds=.008*len(costs), velocity_mae=float(np.mean(errors)), cost=cost,
            guidance_fraction=guided/max(1,len(costs)), action_change_fraction=changed/max(1,len(costs)),
            points=points, skipped_known_baseline_falls=skipped, ledger=ledger,
            selected_sources={source:sources.count(source) for source in set(sources)})
    finally:
        env.close()


def prefix(cfg: PairedConfig, snapshot: Snapshot, plan: np.ndarray, remaining: int) -> dict:
    """Execute one branch once; a fall has a complete known finite return."""
    env = WalkerRL(gamma=cfg.gamma, max_episode_steps=remaining)
    try:
        _restore(env, snapshot)
        costs = []
        fell = False
        for action in plan[:min(cfg.H, remaining)]:
            _, _, fell, _, info = env.step(action)
            costs.append(float(info["task_cost"]))
            if fell:
                break
        h = len(costs)
        cost = float(np.dot(cfg.gamma ** np.arange(h), costs))
        endpoint = Snapshot(env.walker.get_state(), env.walker.get_warmstart())
        return dict(snapshot=endpoint, cost=cost, controls=h, fell=bool(fell), remaining=remaining-h,
                    known_full_cost=(cost + cfg.gamma**h * finite_absorbing_cost(cfg.gamma, remaining-h)
                                     if fell else None))
    finally:
        env.close()


def measure_pair(cfg: PairedConfig, reference: dict, point: dict, *, oracle_seed: int,
                 source: str) -> dict:
    if source not in {"train", "calibration"}:
        raise ValueError("Unknown pair measurement role")
    measured, ledger, records, archive_episodes = {}, {}, [], []
    for side in ("candidate", "baseline"):
        branch = point[side]
        executed = prefix(cfg, point["snapshot"], branch["plan"], point["remaining"])
        _charge(ledger, {"prefix_controls":executed["controls"]})
        endpoint = executed["snapshot"]
        delta = endpoint.state - branch["endpoint"]
        public = dict(fell=executed["fell"], prefix_controls=executed["controls"],
            measured_prefix_cost=executed["cost"], original_prefix_score=branch["prefix_score"],
            original_predicted_tail=branch["predicted_tail"], original_supported=branch["supported"],
            endpoint_max_abs_discrepancy=float(np.max(np.abs(delta))),
            replayed_state=endpoint.state.copy(), replayed_warmstart=endpoint.warmstart.copy(),
            original_batch_state=branch["endpoint"].copy(), remaining_tail=executed["remaining"],
            full_costs=[], suffix_costs=[], oracle_seeds=[])
        for replicate in range(cfg.continuation_repeats):
            seed = int(oracle_seed + replicate)
            public["oracle_seeds"].append(seed)
            if executed["fell"] or not executed["remaining"]:
                full = executed["known_full_cost"] if executed["fell"] else executed["cost"]
                public["full_costs"].append(float(full)); public["suffix_costs"].append(None)
                continue
            result = episode(cfg, reference, reset_id=point["group"], planner_seed=seed,
                             snapshot=endpoint, remaining=executed["remaining"])
            _charge(ledger, result["ledger"], "continuation")
            public["suffix_costs"].append(float(result["cost"]))
            public["full_costs"].append(float(executed["cost"] + cfg.gamma**executed["controls"] * result["cost"]))
            if source == "train":
                records.append(dict(source="train", features=endpoint.state[2:].copy(),
                    remaining=executed["remaining"], cost=result["cost"], group=point["group"],
                    pair_id=point["pair_id"], side=side, replicate=replicate, oracle_seed=seed))
                archive_episodes.append(result)
        measured[side] = public
    observed = np.asarray(measured["candidate"]["full_costs"]) - measured["baseline"]["full_costs"]
    predicted = None
    if all(point[side]["predicted_tail"] is not None for side in ("candidate", "baseline")):
        predicted = (point["candidate"]["prefix_score"] - point["baseline"]["prefix_score"]
            + point["gamma_h"] * (point["candidate"]["predicted_tail"] - point["baseline"]["predicted_tail"]))
    noise_se = float(np.std(observed, ddof=1) / np.sqrt(len(observed)))
    return dict(source=source, group=point["group"], pair_id=point["pair_id"], step=point["step"],
        remaining=point["remaining"], equal_plans=point["equal_plans"], mode=point["mode"],
        acquisition_mode=point["acquisition_mode"], guidance_used=point["guidance_used"],
        action_changed=point["action_changed"], branches=measured,
        probe_alpha=point.get("probe_alpha"), candidate_index=point.get("candidate_index"),
        baseline_index=point.get("baseline_index"), bank_local_costs=point.get("bank_local_costs"),
        bank_selection_scores=point.get("bank_selection_scores"),
        measured_differences=observed.tolist(), measured_mean_difference=float(observed.mean()),
        measured_difference_se=noise_se, predicted_full_difference=predicted,
        ledger=ledger, records=records, archive_episodes=archive_episodes)


def select_pairs(points: list[dict], count: int, seed: int, changed_only=False) -> list[dict]:
    """Distinct decisions, sampled without replacement and balanced across roots."""
    unique = {}
    for point in points:
        if not changed_only or (not point["equal_plans"] and point["guidance_used"]):
            unique.setdefault(point["pair_id"], point)
    roots = {}
    for point in unique.values():
        roots.setdefault(point["group"], []).append(point)
    rng = np.random.default_rng(seed)
    groups = sorted(roots)
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(roots[group])
    selected = []
    while len(selected) < count and any(roots.values()):
        for group in groups:
            if roots[group] and len(selected) < count:
                selected.append(roots[group].pop())
    return selected


def pair_calibration(rows: list[dict], cfg: PairedConfig) -> dict:
    """Fresh measured pair means, not a statistical confidence certificate."""
    useful = [r for r in rows if not r["equal_plans"] and r["guidance_used"]
              and r["predicted_full_difference"] is not None
              and all(r["branches"][side]["original_supported"] for side in ("candidate", "baseline"))]
    errors, resolved, correct = [], [], []
    for row in useful:
        scale = finite_absorbing_cost(cfg.gamma, row["remaining"])
        errors.append((row["predicted_full_difference"] - row["measured_mean_difference"]) / scale)
        if abs(row["measured_mean_difference"]) > max(row["measured_difference_se"], 1e-9):
            resolved.append(row)
            correct.append(row["predicted_full_difference"] * row["measured_mean_difference"] > 0)
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    accuracy = float(np.mean(correct)) if correct else None
    roots = len({r["group"] for r in useful})
    resolved_roots = len({r["group"] for r in resolved})
    mean_delta = float(np.mean([r["measured_mean_difference"] for r in useful])) if useful else None
    accepted = (len(useful) >= 2 and roots >= 2 and len(resolved) >= 2 and resolved_roots >= 2 and rmse is not None
                and rmse <= cfg.pair_rmse_max and accuracy >= .5 and mean_delta < 0)
    return dict(eligible=bool(accepted), attempted_pairs=len(rows), supported_nontrivial_pairs=len(useful),
        distinct_roots=roots, resolved_distinct_roots=resolved_roots,
        noise_resolved_pairs=len(resolved), normalized_difference_rmse=rmse,
        sign_accuracy=accuracy, mean_measured_difference=mean_delta,
        repeat_mean_se=(float(np.mean([r["measured_difference_se"] for r in useful])) if useful else None),
        interpretation="Empirical paired continuation check; repeats=2 does not establish a population noise floor or confidence certificate.")


def public_pair(row: dict) -> dict:
    return _serial({key:value for key,value in row.items() if key not in {"records", "archive_episodes"}})


def _job(job):
    cfg = PairedConfig(**job["config"])
    args = {key:value for key,value in job.items() if key not in {"config", "kind"}}
    return episode(cfg, **args) if job["kind"] == "episode" else measure_pair(cfg, **args)


def _aug_summary(rows, horizon):
    result = _summary(rows, horizon)
    result["action_change_fraction"] = float(np.mean([r["action_change_fraction"] for r in rows]))
    return result


def run(out_dir: Path, cfg: PairedConfig, resume=False) -> None:
    validate(cfg)
    torch.set_num_threads(1)
    out_dir = out_dir.resolve()
    start, elapsed_offset = time.monotonic(), 0.
    if not resume:
        out_dir.mkdir(parents=True, exist_ok=False)
        protocol = dict(config=asdict(cfg), source_sha256={name:_sha(ROOT/name) for name in SOURCES},
            environment=provenance(), git_head=_revision(), reset_blocks=BLOCKS,
            reference_controller="Fixed bootstrap-memory plus raw-independent MPPI; canonical zero-plan restart once per suffix, then retain its own plan. No learned guidance.",
            objective="Measured complete finite discounted raw task cost; known failure remainder always charged in full. No actor.",
            archive_rule="Identical common unconditional archive for value-on and value-off; only training collection/reference-query traces enter it.",
            acquisition="8 distinct same-state/warmstart decision pairs, balanced across collection roots; 2 matched suffix-seed replicas/side; rejected shadow proposals still train. At most32 suffix queries/cycle before falls.",
            qualification="Separate fresh roots: actual deployed selected plan versus exact local winner; >=2 nontrivial supported pairs from2roots, >=2 noise-resolved pairs, difference RMSE<=.2, sign accuracy>=.5, mean realized difference<0; then fresh full-controller improvement/survival gate.",
            interpretation="One paired training seed. Measures reuse of discovered experience; not proof of exploration, safety or end-to-end sample efficiency.")
        for name in SOURCES:
            destination = out_dir/"source"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, destination)
        _json(out_dir/"protocol.json", protocol)
    else:
        protocol = json.loads((out_dir/"protocol.json").read_text())
        if protocol["config"] != asdict(cfg):
            raise ValueError("Resume must preserve the full frozen configuration")
    for name, expected in protocol["source_sha256"].items():
        if _sha(ROOT/name) != expected or _sha(out_dir/"source"/name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    if resume and (out_dir/"study_status.json").exists():
        previous = json.loads((out_dir/"study_status.json").read_text())
        if previous["status"] == "complete":
            return
    ledger, histories, records = {}, [], []
    shadow, accepted_value, alpha, begin = None, None, 0., 0
    work_is_lower_bound = False
    offset = cfg.seed * 100_000_000
    def status(phase, iteration=-1, **extra):
        _json(out_dir/"study_status.json", dict(status="running", phase=phase, iteration=iteration,
            requested_iterations=cfg.iterations, elapsed_seconds=elapsed_offset+time.monotonic()-start,
            work=ledger.copy(), driver_pid=os.getpid(), work_is_lower_bound=work_is_lower_bound, **extra))
    def jobs(pool, items):
        return list(pool.map(_job, [dict(config=asdict(cfg), **item) for item in items]))
    def rolls(controller, ids, seed, **extra):
        return [dict(kind="episode", controller=controller, reset_id=rid,
                     planner_seed=seed+index, **extra) for index,rid in enumerate(ids)]
    def archive():
        return MemoryArchive.from_episodes(histories, cfg.H, cfg.memory_stride, cfg.memory_entries).state_dict()
    def charge_rows(rows, category):
        for row in rows:
            _charge(ledger, row["ledger"], category)
    def checkpoint(next_iteration, shared_archive, reference):
        _save(out_dir/"checkpoint.pkl", dict(next_iteration=next_iteration, histories=histories,
            records=records, shadow=shadow, accepted_value=accepted_value, alpha=alpha,
            archive=shared_archive, reference=reference, ledger=ledger.copy(), driver_pid=os.getpid(),
            work_is_lower_bound=work_is_lower_bound,
            elapsed_seconds=elapsed_offset+time.monotonic()-start))
    try:
        if resume:
            state = pickle.loads((out_dir/"checkpoint.pkl").read_bytes())
            old_pid = state.get("driver_pid")
            if old_pid and old_pid != os.getpid():
                try:
                    os.kill(old_pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("Refuse resume while prior driver PID is alive")
            begin, elapsed_offset = state["next_iteration"], state["elapsed_seconds"]
            histories, records, shadow, accepted_value, alpha = (state[k] for k in
                ("histories", "records", "shadow", "accepted_value", "alpha"))
            ledger.update(state["ledger"])
            work_is_lower_bound = True
            shared_archive, reference = state["archive"], state["reference"]
            preserved = out_dir/f"interrupted_{time.time_ns()}"
            # Preserve uncommitted evidence before deterministic iteration replay.
            for folder in ("metric_rows", "collection", "training_pairs", "calibration", "qualification", "controllers", "models", "evaluations"):
                for path in (out_dir/folder).glob("*"):
                    if any(f"{i:04d}" in path.name for i in range(begin, cfg.iterations)):
                        destination = preserved/folder/path.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, destination)
            if (out_dir/"metrics.jsonl").exists():
                preserved.mkdir(exist_ok=True)
                shutil.copy2(out_dir/"metrics.jsonl", preserved/"metrics.jsonl")
            _json(out_dir/f"resume_{time.time_ns()}.json", dict(next_iteration=begin,
                unknown_interrupted_work="Not reconstructed; committed work is a lower bound.",
                source_unchanged=True))
            with (out_dir/"metrics.jsonl").open("w") as stream:
                for iteration in range(begin):
                    row = json.loads((out_dir/"metric_rows"/f"iteration_{iteration:04d}.json").read_text())
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
        with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=mp.get_context("spawn")) as pool:
            if not resume:
                status("bootstrap")
                bootstrap = jobs(pool, rolls(_controller(), range(BLOCKS["bootstrap"]+offset,
                    BLOCKS["bootstrap"]+offset+cfg.bootstrap_episodes), 110_000_000+offset, raw=True))
                charge_rows(bootstrap, "shared_bootstrap")
                histories.extend(bootstrap)
                shared_archive = archive()
                reference = _controller(shared_archive)
                _json(out_dir/"reference_version.json", dict(archive_sha256=archive_digest(shared_archive),
                    source_sha256=protocol["source_sha256"], controller=protocol["reference_controller"]))
                _save(out_dir/"bootstrap.pkl", bootstrap)
                _save(out_dir/"reference_controller.pkl", reference)
                checkpoint(0, shared_archive, reference)
            def evaluate(name, count, block, iteration=-1):
                status(f"evaluation_{name}", iteration)
                panel = {}
                for arm in ("guided", "memory"):
                    controller = _controller(shared_archive, accepted_value if arm == "guided" else None,
                                             alpha if arm == "guided" else 0.)
                    episodes = jobs(pool, rolls(controller, range(block+offset, block+offset+count),
                                                150_000_000+offset+block))
                    charge_rows(episodes, f"evaluation_{arm}")
                    panel[arm] = dict(summary=_aug_summary(episodes, cfg.total_steps),
                        archive_sha256=archive_digest(controller["archive"]),
                        episodes=[dict(_public(e), action_change_fraction=e["action_change_fraction"]) for e in episodes])
                    _save(out_dir/"evaluations"/f"{name}_{arm}_first_trace.pkl", episodes[0])
                panel.update(name=name, controller_alpha=alpha, work=ledger.copy(), archive_identical=True,
                    shared_archive_sha256=archive_digest(shared_archive), work_is_lower_bound=work_is_lower_bound)
                _json(out_dir/"evaluations"/f"{name}.json", panel)
            if begin == 0 and not (out_dir/"evaluations/initial.json").exists():
                evaluate("initial", cfg.evaluation_episodes, BLOCKS["evaluation"])
                checkpoint(0, shared_archive, reference)
            elif begin == 0:
                # Evaluation is durable before the checkpoint; recover its known
                # work if interruption happened between those atomic writes.
                initial_work = json.loads((out_dir/"evaluations/initial.json").read_text())["work"]
                if any(initial_work.get(key, 0) < value for key,value in ledger.items()):
                    raise RuntimeError("Initial evaluation work precedes committed checkpoint")
                ledger.update(initial_work)
                checkpoint(0, shared_archive, reference)
            if archive_digest(reference["archive"]) != json.loads((out_dir/"reference_version.json").read_text())["archive_sha256"]:
                raise RuntimeError("Fixed reference archive version changed")
            for iteration in range(begin, cfg.iterations):
                row = dict(iteration=iteration, deployed_alpha_before=alpha)
                all_points = []
                ids = list(range(BLOCKS["collection"]+offset+iteration*cfg.collection_episodes,
                                 BLOCKS["collection"]+offset+(iteration+1)*cfg.collection_episodes))
                for arm in ("guided", "memory"):
                    status(f"collection_{arm}", iteration)
                    controller = _controller(shared_archive, accepted_value if arm == "guided" else None,
                                             alpha if arm == "guided" else 0.)
                    collected = jobs(pool, rolls(controller, ids, 120_000_000+offset+iteration*1000,
                        collect_pairs=True, shadow_packet=shadow,
                        acquisition_alpha=min(cfg.alpha_max, alpha+cfg.alpha_step)))
                    charge_rows(collected, f"collection_{arm}")
                    row[f"{arm}_collection"] = _aug_summary(collected, cfg.total_steps)
                    all_points.extend(p for e in collected for p in e["points"])
                    _save(out_dir/"collection"/f"{arm}_{iteration:04d}.pkl", collected)
                    for result in collected:
                        result["points"] = []
                    histories.extend(collected)
                selected = select_pairs(all_points, cfg.train_pairs, 130_000_000+offset+iteration)
                status("training_endpoint_pairs", iteration, selected_pairs=len(selected))
                measured = jobs(pool, [dict(kind="pair", reference=reference, point=point,
                    oracle_seed=140_000_000+offset+iteration*1000+index*cfg.continuation_repeats,
                    source="train") for index,point in enumerate(selected)])
                charge_rows(measured, "training_pairs")
                for result in measured:
                    records.extend(result["records"])
                    histories.extend(result["archive_episodes"])
                _save(out_dir/"training_pairs"/f"iteration_{iteration:04d}.pkl", measured)
                _json(out_dir/"training_pairs"/f"iteration_{iteration:04d}.json",
                      dict(pairs=[public_pair(r) for r in measured],
                           records=_serial([r for result in measured for r in result["records"]]),
                           pool_decisions=len(all_points), selected_distinct_pairs=len(selected)))
                shared_archive = archive()  # exactly the same unconditional refresh in both arms
                model = None
                if len({r["group"] for r in records}) >= 4:
                    status("paired_value_fit", iteration)
                    model = PairedRidgeValue.fit(records, total_steps=cfg.total_steps,
                        gamma=cfg.gamma, ridge=cfg.ridge, support_threshold=cfg.support_radius)
                    shadow = model.packet()  # even a rejected value steers future TRAIN queries
                    _save(out_dir/"models"/f"value_{iteration:04d}.pkl", shadow)
                row["value_metrics"] = model.metrics if model is not None else None
                proposed_alpha = min(cfg.alpha_max, alpha+cfg.alpha_step) if model is not None else 0.
                candidate = _controller(shared_archive, shadow if model is not None else None, proposed_alpha)
                memory = _controller(shared_archive)
                incumbent = _controller(shared_archive, accepted_value, alpha)
                calibration = dict(eligible=False, attempted_pairs=0, reason="no_model")
                audit_rows = []
                if model is not None:
                    status("fresh_pair_calibration", iteration)
                    calibration_ids = range(BLOCKS["calibration"]+offset+iteration*cfg.calibration_episodes,
                        BLOCKS["calibration"]+offset+(iteration+1)*cfg.calibration_episodes)
                    audit_sources = jobs(pool, rolls(candidate, calibration_ids,
                        160_000_000+offset+iteration*1000, collect_pairs=True,
                        shadow_packet=shadow, pair_mode="calibration"))
                    charge_rows(audit_sources, "calibration_source")
                    audit_points = select_pairs([p for e in audit_sources for p in e["points"]],
                        cfg.calibration_pairs, 170_000_000+offset+iteration, changed_only=True)
                    audit_rows = jobs(pool, [dict(kind="pair", reference=reference, point=point,
                        oracle_seed=180_000_000+offset+iteration*1000+index*cfg.continuation_repeats,
                        source="calibration") for index,point in enumerate(audit_points)])
                    charge_rows(audit_rows, "calibration_pairs")
                    calibration = pair_calibration(audit_rows, cfg)
                    _save(out_dir/"calibration"/f"source_{iteration:04d}.pkl", audit_sources)
                    _json(out_dir/"calibration"/f"iteration_{iteration:04d}.json",
                          dict(summary=calibration, pairs=[public_pair(r) for r in audit_rows],
                               fitting_records_added=0, archive_episodes_added=0))
                status("full_controller_qualification", iteration)
                qualification_ids = range(BLOCKS["qualification"]+offset+iteration*cfg.qualification_episodes,
                    BLOCKS["qualification"]+offset+(iteration+1)*cfg.qualification_episodes)
                panels = {}
                for label, controller in (("memory", memory), ("incumbent", incumbent), ("candidate", candidate)):
                    if label == "incumbent" and alpha == 0:
                        panels[label] = panels["memory"]
                    elif label == "candidate" and model is None:
                        panels[label] = panels["memory"]
                    else:
                        panels[label] = jobs(pool, rolls(controller, qualification_ids,
                            190_000_000+offset+iteration*1000))
                        charge_rows(panels[label], f"qualification_{label}")
                qualifies = bool(calibration["eligible"] and _promote(panels["candidate"], panels["memory"],
                                                                       panels["incumbent"], cfg.promotion_margin))
                incumbent_ok = bool(alpha > 0 and _promote(panels["incumbent"], panels["memory"],
                                                          panels["memory"], 0.))
                if qualifies:
                    accepted_value, alpha = shadow, proposed_alpha
                    decision = "promote_guidance"
                elif incumbent_ok:
                    decision = "retain_qualified_incumbent"
                else:
                    accepted_value, alpha = None, 0.
                    decision = "memory_only"  # same archive, no archive-selection asymmetry
                row["promotion"] = dict(accepted=qualifies, type=decision, eligible=model is not None,
                    proposed_alpha=proposed_alpha, pair_calibration=calibration,
                    **{name:_aug_summary(panel, cfg.total_steps) for name,panel in panels.items()})
                _json(out_dir/"qualification"/f"iteration_{iteration:04d}.json",
                    {name:[dict(_public(e), action_change_fraction=e["action_change_fraction"]) for e in panel]
                     for name,panel in panels.items()})
                _save(out_dir/"controllers"/f"iteration_{iteration:04d}.pkl",
                      dict(guided=_controller(shared_archive, accepted_value, alpha), memory=memory))
                row.update(measured_query_count=len(records), training_pair_count=len(measured),
                    equal_training_pairs=sum(r["equal_plans"] for r in measured), deployed_alpha=alpha,
                    archive_identical=True, archive_entries=len(shared_archive["states"]),
                    shared_archive_sha256=archive_digest(shared_archive),
                    guided_archive_sha256=archive_digest(shared_archive),
                    memory_archive_sha256=archive_digest(memory["archive"]),
                    qualification_records_added=0, qualification_archive_episodes_added=0)
                if (iteration+1) % cfg.eval_every == 0:
                    evaluate(f"iteration_{iteration:04d}", cfg.evaluation_episodes, BLOCKS["evaluation"], iteration)
                row.update(work=ledger.copy(), work_is_lower_bound=work_is_lower_bound,
                           elapsed_seconds=elapsed_offset+time.monotonic()-start)
                _json(out_dir/"metric_rows"/f"iteration_{iteration:04d}.json", row)
                with (out_dir/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                checkpoint(iteration+1, shared_archive, reference)
                print(json.dumps(dict(iteration=iteration, deployed_alpha=alpha, decision=decision,
                    measured_query_count=len(records), pair_calibration=calibration,
                    elapsed_seconds=row["elapsed_seconds"])), flush=True)
            if not (out_dir/"evaluations/final.json").exists():
                evaluate("final", cfg.final_episodes, BLOCKS["final"], cfg.iterations-1)
            else:
                final_work = json.loads((out_dir/"evaluations/final.json").read_text())["work"]
                if any(final_work.get(key, 0) < value for key,value in ledger.items()):
                    raise RuntimeError("Final evaluation work precedes committed checkpoint")
                ledger.update(final_work)
            checkpoint(cfg.iterations, shared_archive, reference)
            unchanged = all(_sha(ROOT/name) == digest and _sha(out_dir/"source"/name) == digest
                            for name,digest in protocol["source_sha256"].items())
            _json(out_dir/"validation.json", dict(source_unchanged=unchanged, source_files=len(SOURCES)))
            _json(out_dir/"study_status.json", dict(status="complete", phase="complete",
                completed_iterations=cfg.iterations, requested_iterations=cfg.iterations,
                elapsed_seconds=elapsed_offset+time.monotonic()-start, work=ledger,
                source_unchanged=unchanged, deployed_alpha=alpha, archive_identical=True,
                shared_archive_sha256=archive_digest(shared_archive), work_is_lower_bound=work_is_lower_bound))
    except BaseException as error:
        _json(out_dir/"study_status.json", dict(status="failed", phase="failed", error=repr(error),
            completed_iterations=begin if not (out_dir/"metric_rows").exists() else
                len(list((out_dir/"metric_rows").glob("*.json"))), work=ledger,
            elapsed_seconds=elapsed_offset+time.monotonic()-start))
        raise


def main(out_dir: Path, config: PairedConfig = PairedConfig(), resume: bool = False):
    run(out_dir, config, resume)


if __name__ == "__main__":
    tyro.cli(main)
