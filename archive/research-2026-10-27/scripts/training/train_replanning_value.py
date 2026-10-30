"""Frozen-epoch evaluation of actual retained-state replanning controllers.

Independent MPPI supplies candidates. Complete measured C_k trajectories train
V^C_k; one action followed by C_k supplies counterfactual action comparisons.
No canonical-reference restart, Bellman target or standalone actor is used.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
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
from scripts.training.train_coupled_value import _json, _save, _sha, _revision, _summary, _public, _promote
from scripts.training.train_paired_endpoint import archive_digest, _serial
from src.gps.coupled_planner import MemoryArchive
from src.gps.expert_continuation import finite_absorbing_cost
from src.gps.replanning_rollout import rollout_episode, paired_first_action, rerank_point
from src.gps.replanning_value import ReplanningValue

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = dict(bootstrap=46_000_000, collection=47_000_000, retention=48_000_000,
              calibration=49_000_000, qualification=50_000_000,
              evaluation=52_000_000, final=53_000_000)
SOURCES = [
    "scripts/training/train_replanning_value.py", "src/gps/replanning_value.py",
    "src/gps/replanning_planner.py", "src/gps/replanning_rollout.py",
    "scripts/diagnostics/compare_replanning_value.py", "docs/replanning_value_study.md",
    "scripts/training/train_coupled_value.py", "scripts/training/train_paired_endpoint.py",
    "src/gps/paired_planner.py", "src/gps/paired_value.py", "src/gps/coupled_planner.py",
    "src/gps/global_value.py", "src/gps/expert_continuation.py", "src/mppi/mppi.py",
    "src/rl/walker.py", "src/envs/base.py", "src/envs/walker2d.py",
    "src/envs/mujoco_env.py", "src/utils/config.py", "src/utils/math.py",
    "configs/walker2d_best.json", "configs/walker2d_task.json", "assets/walker2d.xml",
    "scripts/runners/eval_policy_walker2d.py",
]
# Freeze the full copied Python/config/asset tree, including transitive helper
# imports used by provenance and reporting, rather than an incomplete hand list.
SOURCES = sorted(set(SOURCES + [str(p.relative_to(ROOT)) for folder, suffixes in
    (("src", {".py"}), ("scripts", {".py"}), ("configs", {".json"}), ("assets", {".xml"}))
    for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in suffixes] + ["pyproject.toml"]))


@dataclass
class ReplanningConfig:
    iterations: int = 24
    total_steps: int = 1000
    seed: int = 0
    workers: int = 6
    bootstrap_episodes: int = 4
    collection_episodes: int = 8
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
    dense_stride: int = 8
    fit_updates: int = 300
    hidden: int = 64
    ensemble_size: int = 3
    pair_weight: float = 1.
    alpha_max: float = 1.
    alpha_step: float = .25
    support_radius: float = 1.5
    pair_rmse_max: float = .2
    promotion_margin: float = 1.
    gamma: float = .999
    angle_noise: float = .03
    velocity_noise: float = .10


def validate(cfg: ReplanningConfig) -> None:
    floats = {"pair_weight", "alpha_max", "alpha_step", "support_radius", "pair_rmse_max",
              "promotion_margin", "gamma", "angle_noise", "velocity_noise"}
    for name, value in asdict(cfg).items():
        if name in floats:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        elif type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a valid integer")
    if cfg.seed >= 10 or not 2 <= cfg.H <= cfg.total_steps:
        raise ValueError("Require seed<10 and2<=H<=N")
    if cfg.collection_episodes < 4 or cfg.continuation_repeats < 2 or cfg.calibration_episodes < 2:
        raise ValueError("Need >=4 fitting roots and repeated comparisons on >=2 fresh roots")
    if not 0 < cfg.gamma < 1 or not 0 < cfg.alpha_step <= cfg.alpha_max <= 1:
        raise ValueError("Invalid discount or influence schedule")
    if min(cfg.support_radius, cfg.pair_rmse_max) <= 0 or min(cfg.pair_weight, cfg.promotion_margin,
                                                           cfg.angle_noise, cfg.velocity_noise) < 0:
        raise ValueError("Invalid model, gate or reset settings")
    if cfg.iterations * max(cfg.collection_episodes, cfg.calibration_episodes,
                            cfg.qualification_episodes) >= 900_000:
        raise ValueError("Reset-role blocks would overlap")


def controller(archive=None, value=None, alpha=0.) -> dict:
    """Flat runtime controller; scorer target version deliberately differs."""
    packet = dict(archive=archive, value=value, alpha=float(alpha))
    # Hash content, not Python object identities or an opaque numeric feature.
    digest = hashlib.sha256()
    def content(item):
        if isinstance(item, np.ndarray):
            a = np.ascontiguousarray(item)
            digest.update(b"array"+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item):
                content(key); content(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"list"+str(len(item)).encode())
            for x in item:
                content(x)
        elif isinstance(item, np.generic):
            content(item.item())
        else:
            digest.update(json.dumps(item,allow_nan=False,sort_keys=True).encode()+b";")
    content(packet)
    packet["version"] = digest.hexdigest()
    packet["value_target_version"] = value["controller_version"] if value is not None else packet["version"]
    return packet


def charge(ledger: dict, rows: list[dict], category: str) -> None:
    for row in rows:
        for key, count in row["ledger"].items():
            if type(count) is not int or count < 0:
                raise ValueError("Invalid measured work ledger")
            name = f"{category}_{key}"
            ledger[name] = ledger.get(name, 0) + count


def summary(rows, horizon):
    result = _summary(rows, horizon)
    result["action_change_fraction"] = float(np.mean([r["action_change_fraction"] for r in rows]))
    result["learned_guidance_fraction"] = float(np.mean([r.get("learned_guidance_fraction", r["guidance_fraction"]) for r in rows]))
    return result


def public_episode(row):
    return dict(_public(row), action_change_fraction=row["action_change_fraction"],
                learned_guidance_fraction=row.get("learned_guidance_fraction", row["guidance_fraction"]))


def archive_episode(row):
    if row.get("source") != "train":
        raise ValueError("Only training traces may refresh common memory")
    keys = ("states", "warmstarts", "actions", "costs", "gamma", "remaining_start", "terminated", "cost")
    return {key: row[key] for key in keys}


def training_records(collection, measured, target_version):
    rows = [r for e in collection for r in e["records"]]
    rows += [r for pair in measured for r in pair["records"]]
    if any(r["source"] != "train" or r["controller_version"] != target_version for r in rows):
        raise ValueError("Fitting cannot mix measurement roles or target controllers")
    return rows


def select_points(points, count, seed, *, value=None, alpha=.25, cfg=None, calibration=False,
                  work_ledger=None, work_category="shadow"):
    """Balance ordinary-root ancestry before selecting another decision."""
    rng = np.random.default_rng(seed)
    by_root = {}
    for point in points:
        p = copy.copy(point)
        if calibration or value is not None:
            ranked = rerank_point(point, value, alpha, cfg)
            p["shadow"] = ranked
            if work_ledger is not None:
                key = f"{work_category}_value_prediction_rows"
                work_ledger[key] = work_ledger.get(key,0) + int(ranked["value_prediction_rows"])
                calls = f"{work_category}_rerank_calls"
                work_ledger[calls] = work_ledger.get(calls,0) + 1
        base = int(point["bank"]["baseline_index"])
        if calibration:
            rank = p["shadow"]
            index = int(rank["selected_index"])
            if not rank["guidance_used"] or not rank["action_changed"]:
                continue
            if not (rank["supported"][base] and rank["supported"][index]):
                continue
            p["candidate_index"] = index
        by_root.setdefault(int(point["group"]), []).append(p)
    roots = sorted(by_root)
    rng.shuffle(roots)
    for group in roots:
        rng.shuffle(by_root[group])
    selected = []
    while len(selected) < count and any(by_root.values()):
        for group in roots:
            if len(selected) >= count or not by_root[group]:
                continue
            point = by_root[group].pop()
            if not calibration:
                base = int(point["bank"]["baseline_index"])
                distinct = [i for i, a in enumerate(point["bank"]["actions"])
                            if not np.array_equal(a, point["bank"]["actions"][base])]
                # Half uniform, half current fitted scorer. Equal shadow pairs
                # remain valid zero outcomes; never silently replace by a win.
                if len(selected) % 2 == 0 and "shadow" in point:
                    point["candidate_index"] = int(point["shadow"]["selected_index"])
                    point["acquisition_mode"] = "shadow_actual_score"
                else:
                    point["candidate_index"] = int(rng.choice(distinct)) if distinct else base
                    point["acquisition_mode"] = "uniform_distinct_first_action"
            selected.append(point)
    return selected


def calibration_summary(rows, cfg):
    useful = [r for r in rows if not r["equal_first_actions"] and r.get("prediction") is not None]
    errors, resolved, signs = [], [], []
    for r in useful:
        d = np.asarray(r["differences"], dtype=float)
        mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        errors.append((r["prediction"] - mean) / finite_absorbing_cost(cfg.gamma, r["remaining"]))
        if abs(mean) > max(se, 1e-9):
            resolved.append(r)
            signs.append(r["prediction"] * mean > 0)
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    accuracy = float(np.mean(signs)) if signs else None
    mean = float(np.mean([np.mean(r["differences"]) for r in useful])) if useful else None
    roots = len({r["group"] for r in useful})
    resolved_roots = len({r["group"] for r in resolved})
    eligible = (len(useful) >= 2 and roots >= 2 and len(resolved) >= 2 and resolved_roots >= 2
                and rmse <= cfg.pair_rmse_max and accuracy >= .5 and mean < 0)
    return dict(eligible=bool(eligible), attempted_pairs=len(rows), supported_nontrivial_pairs=len(useful),
        distinct_roots=roots, resolved_distinct_roots=resolved_roots, noise_resolved_pairs=len(resolved),
        normalized_difference_rmse=rmse, sign_accuracy=accuracy, mean_measured_difference=mean,
        interpretation="Measured FIRST action then frozen actual C_k; two-repeat noise check and empirical gate, not a confidence certificate")


def public_pair(row):
    return _serial({k:v for k,v in row.items() if k not in
                    {"records", "targets", "candidate_traces", "baseline_traces"}})


def _job(job):
    cfg = ReplanningConfig(**job["config"])
    args = {k:v for k,v in job.items() if k not in {"kind", "config"}}
    if job["kind"] == "episode":
        return rollout_episode(cfg, **args)
    return paired_first_action(cfg, **args)


def verify_sources(out_dir, protocol):
    for name, digest in protocol["source_sha256"].items():
        if _sha(ROOT/name) != digest or _sha(out_dir/"source"/name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")


def run(out_dir: Path, cfg: ReplanningConfig, resume=False):
    validate(cfg)
    torch.set_num_threads(1)
    out_dir = out_dir.resolve()
    start, elapsed_offset = time.monotonic(), 0.
    if not resume:
        out_dir.mkdir(parents=True, exist_ok=False)
        protocol = dict(config=asdict(cfg), source_sha256={n:_sha(ROOT/n) for n in SOURCES},
            environment=provenance(), git_head=_revision(), reset_blocks=BLOCKS,
            target="Complete measured return of each frozen actual retained-state C_k; full physical/solver/nominal context; no canonical restart or Bellman targets",
            candidate_score="(1-alpha)*(B_i-B_b)+alpha*(Q_i-Q_b), Q_i=c_i+gamma*V^Ck(z_next_i); empirical local/global interpolation until alpha1; no lambda correction cap",
            archive_rule="One identical training-only archive per epoch; refresh unconditionally at NEXT epoch start; final evaluation uses last qualified epoch archive",
            fitting="Fresh current-controller-version records only. Eight ordinary roots plus8root-balanced first-action pairs,2matchedfuture repeats; dense stride8, whole-root holdout and episode balancing",
            qualification="Fresh C_k source roots with candidate shadow choices, matched actual-C_k first-action continuations, then independent6root full-controller panel versus memory and incumbent on SAME archive",
            interpretation="Actor-free approximate policy iteration; one paired training seed, empirical trust, no exploration or sample-efficiency claim")
        for name in SOURCES:
            dest = out_dir/"source"/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dest)
        _json(out_dir/"protocol.json", protocol)
    else:
        protocol = json.loads((out_dir/"protocol.json").read_text())
        if protocol["config"] != asdict(cfg):
            raise ValueError("Resume requires identical frozen configuration")
    verify_sources(out_dir, protocol)
    if resume and json.loads((out_dir/"study_status.json").read_text()).get("status") == "complete":
        return
    histories, ledger, accepted, alpha, begin = [], {}, None, 0., 0
    shared_archive, lower_bound = None, False
    offset = cfg.seed * 100_000_000

    def status(phase, iteration=-1, **extra):
        _json(out_dir/"study_status.json", dict(status="running", phase=phase, iteration=iteration,
            requested_iterations=cfg.iterations, completed_iterations=begin,
            elapsed_seconds=elapsed_offset+time.monotonic()-start, driver_pid=os.getpid(),
            work=ledger.copy(), work_is_lower_bound=lower_bound, **extra))

    def jobs(pool, items):
        return list(pool.map(_job, [dict(config=asdict(cfg), **item) for item in items]))

    def rolls(c, ids, seed, source, **extra):
        return [dict(kind="episode", controller=c, reset_id=rid, planner_seed=seed+i,
                     source=source, dense_stride=cfg.dense_stride, **extra) for i,rid in enumerate(ids)]

    def checkpoint(next_iteration):
        _save(out_dir/"checkpoint.pkl", dict(next_iteration=next_iteration, histories=histories,
            accepted_value=accepted, alpha=alpha, archive=shared_archive, ledger=ledger.copy(),
            elapsed_seconds=elapsed_offset+time.monotonic()-start, driver_pid=os.getpid(),
            work_is_lower_bound=lower_bound))

    try:
        if resume:
            state = pickle.loads((out_dir/"checkpoint.pkl").read_bytes())
            if state.get("driver_pid") != os.getpid():
                try:
                    os.kill(state["driver_pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("Prior driver is alive; refuse duplicate resume")
            begin, elapsed_offset = state["next_iteration"], state["elapsed_seconds"]
            histories, accepted, alpha, shared_archive = (state[k] for k in
                ("histories", "accepted_value", "alpha", "archive"))
            ledger.update(state["ledger"])
            lower_bound = True  # unknown interrupted work is never invented
            preserved = out_dir/f"interrupted_{time.time_ns()}"
            for folder in ("metric_rows", "collection", "training_pairs", "calibration", "retention",
                           "qualification", "models", "controllers", "evaluations"):
                for path in (out_dir/folder).glob("*"):
                    # A final panel written after all committed cycles is
                    # durable evidence: recover its known work, never rerun it.
                    final_recovery = begin == cfg.iterations and path.name.startswith("final")
                    if not final_recovery and (path.stem == "final" or any(f"{i:04d}" in path.name for i in range(begin, cfg.iterations))):
                        dest = preserved/folder/path.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, dest)
                        path.unlink()
            if (out_dir/"metrics.jsonl").exists():
                preserved.mkdir(exist_ok=True)
                shutil.copy2(out_dir/"metrics.jsonl", preserved/"metrics.jsonl")
            with (out_dir/"metrics.jsonl").open("w") as f:
                for i in range(begin):
                    f.write(json.dumps(json.loads((out_dir/"metric_rows"/f"iteration_{i:04d}.json").read_text()))+"\n")
            _json(out_dir/f"resume_{time.time_ns()}.json", dict(next_iteration=begin,
                unknown_interrupted_work="Uncounted; committed ledger is a lower bound", source_unchanged=True))

        with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=mp.get_context("spawn")) as pool:
            if not resume:
                status("bootstrap")
                bootstrap = jobs(pool, rolls(controller(), range(BLOCKS["bootstrap"]+offset,
                    BLOCKS["bootstrap"]+offset+cfg.bootstrap_episodes), 210_000_000+offset, "train"))
                charge(ledger, bootstrap, "bootstrap")
                histories.extend(archive_episode(e) for e in bootstrap)
                _save(out_dir/"bootstrap.pkl", bootstrap)
                shared_archive = MemoryArchive.from_episodes(histories, cfg.H, cfg.memory_stride,
                                                            cfg.memory_entries).state_dict()
                checkpoint(0)

            def evaluate(name, count, block, iteration=-1):
                status(f"evaluation_{name}", iteration)
                panel = dict(name=name, controller_alpha=alpha, archive_identical=True,
                    shared_archive_sha256=archive_digest(shared_archive), work_is_lower_bound=lower_bound)
                for arm in ("guided", "memory"):
                    c = controller(shared_archive, accepted if arm == "guided" else None,
                                   alpha if arm == "guided" else 0.)
                    episodes = jobs(pool, rolls(c, range(block+offset, block+offset+count),
                        250_000_000+offset+block, "final" if name == "final" else "evaluation"))
                    charge(ledger, episodes, f"evaluation_{arm}")
                    panel[arm] = dict(summary=summary(episodes, cfg.total_steps),
                        archive_sha256=archive_digest(c["archive"]), controller_version=c["version"],
                        value_target_version=c["value_target_version"], episodes=[public_episode(e) for e in episodes])
                    _save(out_dir/"evaluations"/f"{name}_{arm}_first_trace.pkl", episodes[0])
                panel["work"] = ledger.copy()
                _json(out_dir/"evaluations"/f"{name}.json", panel)

            if begin == 0 and not (out_dir/"evaluations/initial.json").exists():
                evaluate("initial", cfg.evaluation_episodes, BLOCKS["evaluation"])
                checkpoint(0)
            elif begin == 0:
                ledger.update(json.loads((out_dir/"evaluations/initial.json").read_text())["work"])
                checkpoint(0)

            for iteration in range(begin, cfg.iterations):
                # Freeze the shared archive BEFORE defining C_k or any targets.
                shared_archive = MemoryArchive.from_episodes(histories, cfg.H, cfg.memory_stride,
                                                            cfg.memory_entries).state_dict()
                digest = archive_digest(shared_archive)
                memory, current = controller(shared_archive), controller(shared_archive, accepted, alpha)
                row = dict(iteration=iteration, deployed_alpha_before=alpha)
                retention = dict(checked=False, revoked=False, reason="alpha_zero")
                if alpha > 0:
                    status("incumbent_retention_on_refreshed_archive", iteration)
                    ids = range(BLOCKS["retention"]+offset+iteration*cfg.qualification_episodes,
                                BLOCKS["retention"]+offset+(iteration+1)*cfg.qualification_episodes)
                    old = jobs(pool, rolls(current, ids, 220_000_000+offset+iteration*1000, "qualification"))
                    mem = jobs(pool, rolls(memory, ids, 220_000_000+offset+iteration*1000, "qualification"))
                    charge(ledger, old, "retention_incumbent"); charge(ledger, mem, "retention_memory")
                    retained = _promote(old, mem, mem, 0.)
                    retention = dict(checked=True, revoked=not retained, incumbent=summary(old,cfg.total_steps),
                                     memory=summary(mem,cfg.total_steps), reason="fresh_same_archive_check")
                    _json(out_dir/"retention"/f"iteration_{iteration:04d}.json", retention)
                    if not retained:
                        accepted, alpha, current = None, 0., memory
                row.update(retention=retention, target_controller_version=current["version"],
                    target_archive_sha256=digest, target_scorer_version=current["value_target_version"])
                _save(out_dir/"controllers"/f"target_{iteration:04d}.pkl", current)

                status("collection_actual_controller", iteration, target_controller_version=current["version"])
                ids = range(BLOCKS["collection"]+offset+iteration*cfg.collection_episodes,
                            BLOCKS["collection"]+offset+(iteration+1)*cfg.collection_episodes)
                collection = jobs(pool, rolls(current, ids, 230_000_000+offset+iteration*1000,
                    "train", collect_points=True, query_stride=cfg.query_stride))
                charge(ledger, collection, "collection")
                _save(out_dir/"collection"/f"iteration_{iteration:04d}.pkl", collection)
                points = [p for e in collection for p in e["points"]]
                selected = select_points(points, cfg.train_pairs, 231_000_000+offset+iteration,
                    value=accepted, alpha=min(cfg.alpha_max, alpha+cfg.alpha_step), cfg=cfg,
                    work_ledger=ledger, work_category="training_shadow")
                status("training_first_action_pairs", iteration, selected_pairs=len(selected))
                measured = jobs(pool, [dict(kind="pair", controller=current, point=p,
                    candidate_index=p["candidate_index"],
                    seeds=[232_000_000+offset+iteration*1000+j*cfg.continuation_repeats+r
                           for r in range(cfg.continuation_repeats)], source="train", dense_stride=cfg.dense_stride)
                    for j,p in enumerate(selected)])
                charge(ledger, measured, "training_pairs")
                _save(out_dir/"training_pairs"/f"iteration_{iteration:04d}.pkl", measured)
                _json(out_dir/"training_pairs"/f"iteration_{iteration:04d}.json",
                      dict(pairs=[public_pair(e) for e in measured], pool_decisions=len(points),
                           selected_distinct_pairs=len(selected), target_controller_version=current["version"]))
                records = training_records(collection, measured, current["version"])
                status("actual_controller_value_fit", iteration, fitting_rows=len(records))
                model = ReplanningValue.fit(records, total_steps=cfg.total_steps, gamma=cfg.gamma,
                    controller_version=current["version"], seed=cfg.seed*1000+iteration,
                    updates=cfg.fit_updates, hidden=cfg.hidden, ensemble_size=cfg.ensemble_size,
                    pair_weight=cfg.pair_weight, support_threshold=cfg.support_radius)
                packet = model.packet()
                _save(out_dir/"models"/f"value_{iteration:04d}.pkl", packet)
                proposed = min(cfg.alpha_max, alpha+cfg.alpha_step)
                candidate = controller(shared_archive, packet, proposed)

                status("fresh_actual_controller_pair_calibration", iteration)
                cal_ids = range(BLOCKS["calibration"]+offset+iteration*cfg.calibration_episodes,
                                BLOCKS["calibration"]+offset+(iteration+1)*cfg.calibration_episodes)
                cal_sources = jobs(pool, rolls(current, cal_ids, 240_000_000+offset+iteration*1000,
                    "calibration", collect_points=True, query_stride=cfg.query_stride))
                charge(ledger, cal_sources, "calibration_source")
                cal_points = select_points([p for e in cal_sources for p in e["points"]],
                    cfg.calibration_pairs, 241_000_000+offset+iteration, value=packet,
                    alpha=proposed, cfg=cfg, calibration=True,
                    work_ledger=ledger, work_category="calibration_shadow")
                audits = jobs(pool, [dict(kind="pair", controller=current, point=p,
                    candidate_index=p["candidate_index"],
                    seeds=[242_000_000+offset+iteration*1000+j*cfg.continuation_repeats+r
                           for r in range(cfg.continuation_repeats)], source="calibration", dense_stride=cfg.dense_stride)
                    for j,p in enumerate(cal_points)])
                charge(ledger, audits, "calibration_pairs")
                for a,p in zip(audits,cal_points):
                    rank = p["shadow"]
                    a["prediction"] = float(rank["q_values"][p["candidate_index"]]-rank["q_values"][rank["baseline_index"]])
                cal = calibration_summary(audits, cfg)
                _save(out_dir/"calibration"/f"source_{iteration:04d}.pkl", cal_sources)
                _save(out_dir/"calibration"/f"pairs_{iteration:04d}.pkl", audits)
                _json(out_dir/"calibration"/f"iteration_{iteration:04d}.json", dict(summary=cal,
                    pairs=[public_pair(a) for a in audits], fitting_records_added=0, archive_episodes_added=0,
                    capture_controller_version=current["version"], candidate_controller_version=candidate["version"]))

                panels = {}
                if cal["eligible"]:
                    status("fresh_full_controller_qualification", iteration)
                    qids = range(BLOCKS["qualification"]+offset+iteration*cfg.qualification_episodes,
                                 BLOCKS["qualification"]+offset+(iteration+1)*cfg.qualification_episodes)
                    for label,c in (("memory",memory),("incumbent",current),("candidate",candidate)):
                        if label == "incumbent" and alpha == 0:
                            panels[label] = panels["memory"]
                        else:
                            panels[label] = jobs(pool, rolls(c,qids,243_000_000+offset+iteration*1000,"qualification"))
                            charge(ledger, panels[label], f"qualification_{label}")
                    qualifies = _promote(panels["candidate"], panels["memory"], panels["incumbent"], cfg.promotion_margin)
                    _json(out_dir/"qualification"/f"iteration_{iteration:04d}.json",
                          {k:[public_episode(e) for e in v] for k,v in panels.items()})
                else:
                    qualifies = False
                if qualifies:
                    accepted, alpha = packet, proposed
                decision = "promote_guidance" if qualifies else ("retain_incumbent" if alpha else "memory_only")
                row.update(value_metrics=model.metrics, collection_summary=summary(collection,cfg.total_steps),
                    training_record_count=len(records), training_pair_count=len(measured),
                    equal_training_pairs=sum(e["equal_first_actions"] for e in measured),
                    training_root_count=len({r["group"] for r in records}),
                    promotion=dict(accepted=bool(qualifies), type=decision, proposed_alpha=proposed,
                        pair_calibration=cal, full_controller_checked=bool(panels),
                        **{k:summary(v,cfg.total_steps) for k,v in panels.items()}),
                    deployed_alpha=alpha, archive_identical=True, shared_archive_sha256=digest,
                    guided_archive_sha256=digest, memory_archive_sha256=digest,
                    archive_entries=len(shared_archive["states"]), qualification_records_added=0,
                    qualification_archive_episodes_added=0)
                _save(out_dir/"controllers"/f"iteration_{iteration:04d}.pkl", dict(
                    guided=controller(shared_archive,accepted,alpha), memory=memory))
                # Only TRAIN traces feed NEXT epoch's identical archive.
                histories.extend(archive_episode(e) for e in collection)
                for pair in measured:
                    histories.extend(archive_episode(e) for e in pair["baseline_traces"])
                    if not pair["equal_first_actions"]:
                        histories.extend(archive_episode(e) for e in pair["candidate_traces"])
                if (iteration+1) % cfg.eval_every == 0:
                    evaluate(f"iteration_{iteration:04d}", cfg.evaluation_episodes, BLOCKS["evaluation"], iteration)
                row.update(work=ledger.copy(), work_is_lower_bound=lower_bound,
                           elapsed_seconds=elapsed_offset+time.monotonic()-start)
                _json(out_dir/"metric_rows"/f"iteration_{iteration:04d}.json", row)
                with (out_dir/"metrics.jsonl").open("a") as f:
                    f.write(json.dumps(row,allow_nan=False)+"\n")
                begin = iteration+1
                checkpoint(begin)
                print(json.dumps(dict(iteration=iteration,deployed_alpha=alpha,decision=decision,
                    target_controller_version=current["version"],records=len(records),calibration=cal)),flush=True)

            if not (out_dir/"evaluations/final.json").exists():
                evaluate("final",cfg.final_episodes,BLOCKS["final"],cfg.iterations-1)
            else:
                ledger.update(json.loads((out_dir/"evaluations/final.json").read_text())["work"])
            verify_sources(out_dir,protocol)
            checkpoint(cfg.iterations)
            _json(out_dir/"validation.json",dict(source_unchanged=True,source_files=len(SOURCES)))
            _json(out_dir/"study_status.json",dict(status="complete",phase="complete",
                completed_iterations=cfg.iterations,requested_iterations=cfg.iterations,source_unchanged=True,
                elapsed_seconds=elapsed_offset+time.monotonic()-start,work=ledger,deployed_alpha=alpha,
                archive_identical=True,shared_archive_sha256=archive_digest(shared_archive),work_is_lower_bound=lower_bound))
    except BaseException as error:
        _json(out_dir/"study_status.json",dict(status="failed",phase="failed",error=repr(error),
            completed_iterations=begin,requested_iterations=cfg.iterations,work=ledger,
            work_is_lower_bound=True,elapsed_seconds=elapsed_offset+time.monotonic()-start))
        raise


def main(out_dir: Path, config: ReplanningConfig = ReplanningConfig(), resume: bool = False):
    run(out_dir,config,resume)


if __name__ == "__main__":
    tyro.cli(main)
