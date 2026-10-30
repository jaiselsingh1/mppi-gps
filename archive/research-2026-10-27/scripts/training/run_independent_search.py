"""Run the frozen, bounded independent-search/global-learning mechanism pilot.

Usage: python -m scripts.training.run_independent_search --out-dir runs/NEW
All actual model/execution queries are written before outcomes are summarized.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy

from src.gps.independent_core import Task, Memory, Search
from src.gps.independent_learning import (
    Value, fit_value, measured_returns, score_guidance, promotion, paired_summary,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ["src/gps/independent_core.py", "src/gps/independent_learning.py",
           "src/gps/independent_preflight.py", "scripts/training/run_independent_search.py",
           "scripts/diagnostics/report_independent_search.py",
           "configs/independent_search_pilot.json", "docs/independent_search_protocol.md",
           "pyproject.toml", "uv.lock", "tests/test_independent_core.py",
           "tests/test_independent_learning.py", "tests/test_independent_preflight.py",
           "tests/test_independent_driver_review.py", "tests/test_independent_report.py",
           "tests/test_independent_archival.py"]
WORK = ("independent_controls", "memory_controls", "verification_controls",
        "scratch_controls", "executed_controls", "planner_calls", "failed_queries",
        "prediction_rows", "fit_updates")


def serial(x):
    if isinstance(x, np.ndarray):
        return serial(x.tolist())
    if isinstance(x, np.generic):
        return serial(x.item())
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save_json(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_id(seed, cycle, role, index=0):
    # Disjoint predeclared roles, three training seeds, deterministic independent streams.
    roles = dict(bootstrap=1, acquisition=2, collection=3, branch=4, calibration=5,
                 qualification=6, retention=7, evaluation=8, final=9)
    return 60_000_000 + seed * 1_000_000 + (cycle + 1) * 100_000 + roles[role] * 10_000 + index


@dataclass
class Controller:
    memory: Memory
    kind: str = "memory"
    value: Value | None = None
    horizon: int = 16

    @property
    def version(self):
        body = self.memory.digest + self.kind + str(self.horizon) + (self.value.digest if self.value else "off")
        return hashlib.sha256(body.encode()).hexdigest()


class Evidence:
    def __init__(self, out, config):
        self.out = Path(out)
        self.cfg = config
        self.task = Task(config["steps"], config["goal"])
        self.events = []
        self.episodes = []
        self.gates = []
        self.curves = []
        self.archives = set()
        self.models = set()
        self.arm_starts = {}
        self.preflight = json.loads((self.out / "preflight.json").read_text()) if (self.out / "preflight.json").exists() else {"counts": {"total_controls": 0}, "wall_seconds": 0.}
        self.start = time.perf_counter()
        self.handle = (self.out / "queries.jsonl").open("x", buffering=1)
        self.out.joinpath("episodes").mkdir()
        self.out.joinpath("controllers").mkdir()

    def log(self, **row):
        row = dict(query_id=len(self.events), elapsed_seconds=time.perf_counter()-self.start, **row)
        self.handle.write(json.dumps(serial(row), separators=(",", ":")) + "\n")
        # Retain only counters/filters in RAM; raw query arrays live durably above.
        keep = set(WORK) | {"query_id", "seed", "arm", "cycle", "phase", "role", "kind", "wall_seconds"}
        self.events.append({k: v for k, v in row.items() if k in keep})
        return row

    def totals(self, *, seed=None, arm=None, cycle=None, phase=None):
        rows = [x for x in self.events if all(v is None or x.get(k) == v for k, v in
                (("seed", seed), ("arm", arm), ("cycle", cycle), ("phase", phase)))]
        out = {key: sum(int(x.get(key, 0)) for x in rows) for key in WORK}
        out["simulated_controls"] = sum(out[k] for k in
            ("independent_controls", "memory_controls", "verification_controls", "scratch_controls"))
        out["total_controls"] = out["simulated_controls"] + out["executed_controls"]
        out["wall_seconds"] = sum(float(x.get("wall_seconds", 0.)) for x in rows)
        return out

    def save_controller(self, controller):
        if controller.memory.digest not in self.archives:
            # Complete raw training episodes are saved separately; archive ids select them exactly.
            save_json(self.out / "controllers" / f"archive_{controller.memory.digest}.json",
                      dict(digest=controller.memory.digest, row_ids=controller.memory.row_ids,
                           episode_ids=[e["id"] for e in controller.memory.episodes]))
            self.archives.add(controller.memory.digest)
        if controller.value is not None and controller.value.digest not in self.models:
            value = controller.value
            np.savez_compressed(self.out / "controllers" / f"value_{value.digest}.npz", **value.packet())
            self.models.add(value.digest)
        save_json(self.out / "controllers" / f"controller_{controller.version}.json",
                  dict(version=controller.version, archive=controller.memory.digest,
                       kind=controller.kind, value=controller.value.digest if controller.value else None,
                       horizon=controller.horizon,
                       target_version=controller.value.target_version if controller.value else None,
                       warm_start="none; independent restart each decision", influence=int(controller.value is not None)))

    def episode(self, controller, *, seed, arm, cycle, role, index, phase="training",
                H=None, state=None, remaining=None, forced=None, root=None,
                stream=None, save_banks=False):
        start = time.perf_counter()
        reset_id = seed_id(seed, cycle, role, index)
        stream = int(reset_id + 100_000_000 if stream is None else stream)
        root = int(reset_id if root is None else root)
        remaining = int(self.cfg["steps"] if remaining is None else remaining)
        state = self.task.reset(reset_id) if state is None else np.array(state, dtype=float)
        if H is not None and H != controller.horizon:
            controller = Controller(controller.memory, controller.kind, controller.value, H)
        identifier = f"s{seed}_{arm}_c{cycle}_{role}_{index}_{len(self.episodes):06d}"
        self.save_controller(controller)
        search = Search(self.task, self.cfg["K"], controller.horizon,
                        self.cfg["temperature"], controller.memory)
        # Local query ledger is drained each call; the durable outer ledger owns all accounting.
        states, actions, costs, changed, support = [state.copy()], [], [], [], []
        points, decisions = [], []
        start_event = len(self.events)
        for t in range(remaining):
            r = remaining-t
            query_seed = int(np.random.SeedSequence([stream, t]).generate_state(1)[0])
            context = dict(seed=seed, arm=arm, cycle=cycle, phase=phase, role=role,
                           episode=identifier, root=root, step=t, remaining=r,
                           controller=controller.version, archive=controller.memory.digest)
            if t == 0 and forced is not None:
                action = np.asarray(forced, dtype=float).reshape(1)
                changed.append(False); support.append(False)
                decisions.append(None)
            else:
                try:
                    decision = search.plan(state, r, query_seed)
                except Exception as error:
                    failed = dict(getattr(error, "query", {"failed_queries": 1}))
                    for key in ("query_id", "seed", "remaining"):
                        failed.pop(key, None)
                    self.log(**context, kind="failed_plan", **failed)
                    raise
                query = dict(decision["query"])
                for key in ("query_id", "seed", "remaining", "selection", "baseline_index"):
                    query.pop(key, None)
                bank = dict(decision["bank"])
                decision["bank"] = bank
                baseline = decision["baseline_index"]
                self.log(**context, kind="plan", **query, proposal_seed=query_seed,
                         independent_digest=hashlib.sha256(bank["plans"][:self.cfg["K"]].tobytes()).hexdigest(),
                         mean_plan=bank["plans"][self.cfg["K"]],
                         memory_row_ids=bank["memory_row_ids"], baseline_index=baseline)
                g = None
                selected = decision["baseline_index"]
                if controller.value is not None:
                    scratch_start = time.perf_counter()
                    try:
                        g = score_guidance(self.task, decision, controller.value, controller.kind)
                    except Exception as error:
                        self.log(**context, kind="failed_guidance", wall_seconds=time.perf_counter()-scratch_start,
                                 error=f"{type(error).__name__}: {error}",
                                 **getattr(error, "guidance_work", {"failed_queries": 1}))
                        raise
                    selected = g["selected"]
                    self.log(**context, kind="guidance", scratch_controls=g["scratch_controls"],
                             prediction_rows=g["prediction_rows"], wall_seconds=time.perf_counter()-scratch_start,
                             supported_baseline=g["supported_baseline"], selected=selected,
                             predicted_scores=g["scores"], supported=g["supported"],
                             support_distances=g["support_distances"])
                action = bank["actions"][selected].copy()
                did_change = not np.array_equal(action, bank["actions"][baseline])
                changed.append(did_change)
                support.append(bool(g and g["supported_baseline"]))
                self.log(**context, kind="selection", selected_index=selected, action=action, changed=did_change)
                if save_banks and t % self.cfg["bank_stride"] == 0:
                    points.append(dict(state=state.copy(), remaining=r, step=t, root=root,
                                       decision=decision, episode=identifier))
                decisions.append(dict(first_actions=bank["actions"], endpoints=bank["endpoints"],
                                      stage_costs=bank["stage_costs"], heuristic_scores=bank["heuristic_scores"],
                                      local_scores=bank["local_scores"], baseline=baseline, selected=selected,
                                      proposal_seed=query_seed))
                search.ledger.events.clear()
            execute_start = time.perf_counter()
            try:
                state, cost, _ = self.task.transition(state, action)
            except Exception as error:
                self.log(**context, kind="failed_execute", executed_controls=1, failed_queries=1,
                         forced=bool(t == 0 and forced is not None), state=states[-1], action=action,
                         error=f"{type(error).__name__}: {error}", wall_seconds=time.perf_counter()-execute_start)
                raise
            self.log(**context, kind="execute", executed_controls=1,
                     forced=bool(t == 0 and forced is not None), state=states[-1], action=action,
                     next_state=state, cost=float(cost), wall_seconds=time.perf_counter()-execute_start)
            states.append(state.copy()); actions.append(action.copy()); costs.append(float(cost))
            # Continue free absorbing controls to finite cutoff, still charge every call.
        returns = measured_returns(costs, remaining)
        result = dict(id=identifier, root=root, seed=seed, arm=arm, cycle=cycle, role=role, phase=phase,
                      states=np.array(states), actions=np.array(actions), costs=np.array(costs), returns=returns,
                      remaining_start=remaining, controller=controller.version, stream=stream,
                      horizon=controller.horizon,
                      success=bool(np.any(np.array(states)[:, 0] >= self.task.goal)),
                      cost=float(returns[0]), changed=int(sum(changed)), supported=int(sum(support)),
                      points=points, forced=forced is not None, elapsed_seconds=time.perf_counter()-start)
        max_bank = self.cfg["K"]+9
        def packed(key, shape=()):
            out = np.full((remaining, max_bank)+shape, np.nan)
            for t, d in enumerate(decisions):
                if d is not None:
                    value = np.asarray(d[key]); out[t, :len(value)] = value
            return out
        arrays = {key: result[key] for key in ("states", "actions", "costs", "returns")}
        arrays.update(bank_first_actions=packed("first_actions", (1,)), bank_endpoints=packed("endpoints", (2,)),
                      bank_stage_costs=packed("stage_costs"), bank_heuristic_scores=packed("heuristic_scores"),
                      bank_local_scores=packed("local_scores"),
                      baseline_indices=np.array([-1 if d is None else d["baseline"] for d in decisions]),
                      selected_indices=np.array([-1 if d is None else d["selected"] for d in decisions]),
                      proposal_seeds=np.array([-1 if d is None else d["proposal_seed"] for d in decisions], dtype=np.int64),
                      remaining_start=remaining)
        np.savez_compressed(self.out / "episodes" / f"{identifier}.npz", **arrays)
        meta = {key: value for key, value in result.items()
                if key not in ("states", "actions", "costs", "returns", "points")}
        meta["ledger_event_range"] = [start_event, len(self.events)]
        save_json(self.out / "episodes" / f"{identifier}.json", meta)
        self.episodes.append(meta)
        return result

    def curve(self, controller, *, seed, arm, cycle, final=False):
        role = "final" if final else "evaluation"
        # Same reset/planner streams for all arms; repeated development roots across cycles.
        n = self.cfg["final_roots"] if final else self.cfg["evaluation_roots"]
        traces = [self.episode(controller, seed=seed, arm=arm, cycle=(-1 if not final else 9),
                               role=role, index=i, phase=role) for i in range(n)]
        total = self.totals(seed=seed, arm=arm)
        shared = self.totals(seed=seed, arm="shared")
        total["shared_bootstrap_controls"] = shared["total_controls"]
        total["preflight_controls"] = self.preflight["counts"]["total_controls"]
        total["charged_controls"] = total["total_controls"] + shared["total_controls"] + total["preflight_controls"]
        total["standalone_wall_seconds"] = (time.perf_counter()-self.arm_starts.get((seed, arm), self.start)
            + sum(x["elapsed_seconds"] for x in self.episodes if x["seed"] == seed and x["arm"] == "shared")
            + self.preflight["wall_seconds"])
        packet = dict(seed=seed, arm=arm, cycle=cycle, final=final,
                      controller=controller.version, archive=controller.memory.digest,
                      model=controller.value.digest if controller.value else None,
                      mean_cost=float(np.mean([e["cost"] for e in traces])),
                      success_rate=float(np.mean([e["success"] for e in traces])),
                      changed_actions=sum(e["changed"] for e in traces),
                      evaluated_executed_controls=sum(len(e["actions"]) for e in traces),
                      costs=[e["cost"] for e in traces], successes=[e["success"] for e in traces],
                      **total)
        self.curves.append(packet)
        save_json(self.out / "learning_curves.json", self.curves)
        return packet


def episode_upper(cfg, H, active=False, remaining=None):
    r = cfg["steps"] if remaining is None else remaining
    bank = cfg["K"] + 9
    return bank * sum(min(H, n) for n in range(1, r+1)) + r * (1 + (bank if active else 0))


def training_used(ev, seed, arm, cycle):
    return ev.totals(seed=seed, arm=arm, cycle=cycle, phase="training")["total_controls"]


def fits_budget(ev, seed, arm, cycle, upper):
    return training_used(ev, seed, arm, cycle)+upper <= ev.cfg["cycle_training_controls"]


def dataset(episodes):
    states, remaining, returns, groups, ids = [], [], [], [], []
    for episode in episodes:
        start = 1 if episode["forced"] else 0
        for t in range(start, len(episode["costs"])):
            states.append(episode["states"][t]); remaining.append(episode["remaining_start"]-t)
            returns.append(episode["returns"][t]); groups.append(episode["root"])
            ids.append((episode["id"], t))
    return np.array(states), np.array(remaining), np.array(returns), np.array(groups), ids


def fit(ev, reference, arm, traces, branch_traces, seed, cycle):
    cfg = ev.cfg
    xs, rs, ys, groups, ids = dataset(traces + branch_traces)
    start = time.perf_counter()
    kwargs = {key: cfg[key] for key in ("centers", "width", "ridge", "support_radius")}
    if arm == "mc":
        value = fit_value(xs, rs, ys, groups, total_steps=cfg["steps"],
                          version=reference.version, kind=arm, **kwargs)
        target_record = dict(states=xs, remaining=rs, targets=ys, groups=groups,
                             ids=np.array([f"{e}:{t}" for e, t in ids]),
                             target_kind="complete MC of actual frozen controller")
        updates = 1
    else:
        points = [p for trace in traces for p in trace["points"]]
        px = np.array([p["state"] for p in points])
        pr = np.array([p["remaining"] for p in points])
        pg = np.array([p["root"] for p in points])
        value, history = None, []
        prediction_rows = 0
        for sweep in range(cfg["polo_sweeps"]):
            targets = []
            for p in points:
                bank = p["decision"]["bank"]
                tail_r = bank["endpoint_remaining"]
                if value is None:
                    tail = ev.task.heuristic(bank["endpoints"]) if tail_r else np.zeros(len(bank["actions"]))
                else:
                    tail = value.predict(bank["endpoints"], tail_r)["values"]
                    prediction_rows += len(tail)
                targets.append(float(np.min(bank["stage_costs"] + tail)))
            targets = np.minimum(targets, pr*1.02)
            history.append(np.array(targets))
            value = fit_value(px, pr, targets, pg, total_steps=cfg["steps"],
                              version=reference.version, kind=arm, **kwargs)
        ev.log(seed=seed, arm=arm, cycle=cycle, phase="training", role="fit", kind="backup_predictions",
               prediction_rows=prediction_rows)
        target_record = dict(states=px, remaining=pr, targets=np.array(history), groups=pg,
                             ids=np.array([f"{p['episode']}:{p['step']}" for p in points]),
                             target_kind="POLO-style optimized model H-step bootstrap; not measured V^C")
        updates = cfg["polo_sweeps"]
    np.savez_compressed(ev.out / "controllers" / f"targets_s{seed}_{arm}_c{cycle}.npz", **target_record)
    # Preserve rejected scorers too; no qualification is needed to archive a fit.
    ev.save_controller(Controller(reference.memory, arm, value))
    ev.log(seed=seed, arm=arm, cycle=cycle, phase="training", role="fit", kind="fit",
           fit_updates=updates, wall_seconds=time.perf_counter()-start,
           train_roots=np.unique(groups), train_rows=len(xs), target_version=reference.version,
           value_digest=value.digest)
    return value


def branches(ev, reference, point, candidate, *, seed, arm, cycle, role, index, repeats):
    bank = point["decision"]["bank"]
    baseline = point["decision"]["baseline_index"]
    equal = np.array_equal(bank["actions"][baseline], bank["actions"][candidate])
    outcomes, traces = {"baseline": [], "candidate": []}, []
    for rep in range(repeats):
        stream = seed_id(seed, cycle, role, index*10+rep)+200_000_000
        for side, candidate_id in (("baseline", baseline), ("candidate", candidate)):
            if side == "candidate" and equal:
                outcomes[side].append(outcomes["baseline"][-1])
                continue
            ep = ev.episode(reference, seed=seed, arm=arm, cycle=cycle, role=role,
                            index=index*100+rep*2+(side == "candidate"),
                            state=point["state"], remaining=point["remaining"],
                            forced=bank["actions"][candidate_id], root=point["root"], stream=stream)
            outcomes[side].append(ep["cost"]); traces.append(ep)
    packet = dict(role=role, seed=seed, arm=arm, cycle=cycle, root=point["root"],
                  point_episode=point["episode"], step=point["step"], candidate=candidate,
                  baseline=baseline, equal=equal, outcomes=outcomes,
                  difference=float(np.mean(outcomes["candidate"])-np.mean(outcomes["baseline"])),
                  target_controller=reference.version)
    ev.log(seed=seed, arm=arm, cycle=cycle, phase="training", role=role, kind="paired_result", result=packet)
    return packet, traces


def heldout_check(ev, reference, candidate, holdout, *, seed, arm, cycle):
    cfg = ev.cfg
    xs, rs, ys, groups, _ = dataset(holdout)
    value = candidate.value
    start = time.perf_counter()
    pred = value.predict(xs, rs)
    nearest = reference.memory.tail(xs, rs)
    rmse = float(np.sqrt(np.mean((pred["values"]-ys)**2)))
    nn_rmse = float(np.sqrt(np.mean((nearest["values"]-ys)**2)))
    horizon_rmse = float(np.sqrt(np.mean((rs-ys)**2)))
    ev.log(seed=seed, arm=arm, cycle=cycle, phase="training", role="calibration", kind="heldout_prediction",
           prediction_rows=len(xs), wall_seconds=time.perf_counter()-start,
           roots=np.unique(groups), rmse=rmse, nn_rmse=nn_rmse, horizon_rmse=horizon_rmse)
    np.savez_compressed(ev.out / "controllers" / f"heldout_s{seed}_{arm}_c{cycle}.npz",
                        states=xs, remaining=rs, targets=ys, predictions=pred["values"],
                        supported=pred["supported"], distances=pred["distances"], groups=groups,
                        nn_predictions=nearest["values"])
    pairs = []
    for root_index, trace in enumerate(holdout):
        choices = []
        for point in trace["points"]:
            shadow_start = time.perf_counter()
            try:
                rank = score_guidance(ev.task, point["decision"], value, arm)
            except Exception as error:
                ev.log(seed=seed, arm=arm, cycle=cycle, phase="training", role="calibration", kind="failed_shadow",
                       wall_seconds=time.perf_counter()-shadow_start,
                       error=f"{type(error).__name__}: {error}",
                       **getattr(error, "guidance_work", {"failed_queries": 1}))
                raise
            ev.log(seed=seed, arm=arm, cycle=cycle, phase="training", role="calibration", kind="shadow",
                   episode=trace["id"], step=point["step"], scratch_controls=rank["scratch_controls"],
                   prediction_rows=rank["prediction_rows"], wall_seconds=time.perf_counter()-shadow_start,
                   changed=rank["changed"], supported_baseline=rank["supported_baseline"])
            if rank["changed"] and rank["supported_baseline"]:
                b = point["decision"]["baseline_index"]
                gap = float(rank["scores"][rank["selected"]] - rank["scores"][b])
                choices.append((gap, point["step"], point, rank))
        if not choices:
            continue
        _, _, point, rank = min(choices, key=lambda x: (x[0], x[1]))
        upper = 2*cfg["branch_repeats"]*episode_upper(cfg, cfg["H"], reference.value is not None,
                                                       point["remaining"])
        if not fits_budget(ev, seed, arm, cycle, upper):
            break
        pair, _ = branches(ev, reference, point, rank["selected"], seed=seed, arm=arm, cycle=cycle,
                           role="calibration", index=root_index, repeats=cfg["branch_repeats"])
        b = point["decision"]["baseline_index"]
        pair["predicted_difference"] = float(rank["scores"][rank["selected"]]-rank["scores"][b])
        pairs.append(pair)
    error = (float(np.sqrt(np.mean([(p["predicted_difference"]-p["difference"])**2 for p in pairs])))
             if pairs else None)
    signs = (float(np.mean([p["difference"] < 0 for p in pairs])) if pairs else None)
    mc_valid = bool(rmse/cfg["steps"] <= cfg["prediction_nrmse_max"] and len(pairs) == 2
                    and error/cfg["steps"] <= cfg["paired_nrmse_max"]
                    and signs >= cfg["final_sign_accuracy_min"]
                    and np.mean([p["difference"] for p in pairs]) < -1.)
    # The approximate-optimality POLO-style target is not declared to be V^C;
    # its empirical qualification remains mandatory, its V^C error is diagnostic.
    result = dict(rmse=rmse, nn_rmse=nn_rmse, horizon_rmse=horizon_rmse,
                  nrmse=rmse/cfg["steps"], supported_fraction=float(pred["supported"].mean()),
                  pairs=pairs, paired_rmse=error, paired_sign_accuracy=signs,
                  passed=mc_valid if arm == "mc" else True,
                  semantics="MC V^C calibration" if arm == "mc" else "V^C discrepancy is diagnostic for optimized-backup control")
    return result


def qualification(ev, candidate, controls, *, seed, arm, cycle, role):
    cfg = ev.cfg
    controllers = {"candidate": candidate, **controls}
    # Reuse exactly identical controllers without claiming an independent rollout.
    unique = {c.version: c for c in controllers.values()}
    n = cfg["retention_roots"] if role == "retention" else cfg["qualification_roots"]
    upper = n * sum(episode_upper(cfg, cfg["H"], c.value is not None) for c in unique.values())
    if not fits_budget(ev, seed, arm, cycle, upper):
        return dict(accepted=False, reason="frozen training budget; qualification not launched")
    measured = {}
    for version, controller in unique.items():
        measured[version] = [ev.episode(controller, seed=seed, arm=arm, cycle=cycle, role=role,
                                       index=i, phase="training") for i in range(n)]
    ce = measured[candidate.version]
    results = {}
    for name, controller in controls.items():
        be = measured[controller.version]
        results[name] = promotion([e["cost"] for e in ce], [e["cost"] for e in be],
                                  [e["success"] for e in ce], [e["success"] for e in be],
                                  margin=cfg["promotion_margin"])
    return dict(accepted=all(x["accepted"] for x in results.values()), comparisons=results,
                candidate_version=candidate.version, control_versions={k: c.version for k, c in controls.items()})


def run_arm(ev, *, seed, arm, bootstrap):
    cfg = ev.cfg
    ev.arm_starts[seed, arm] = time.perf_counter()
    pool = list(bootstrap)
    memory = Memory(pool, cfg["memory_rows"])
    active = Controller(memory)
    ev.curve(active, seed=seed, arm=arm, cycle=-1)
    for cycle in range(cfg["cycles"]):
        gate = dict(seed=seed, arm=arm, cycle=cycle, promoted=False, revoked=False)
        if arm == "memory":
            # Reinvest everything that learning/branching/qualification would cost.
            i = 0
            while fits_budget(ev, seed, arm, cycle, episode_upper(cfg, cfg["acquisition_H"])):
                search_controller = Controller(Memory(pool, cfg["memory_rows"]))
                ep = ev.episode(search_controller, seed=seed, arm=arm, cycle=cycle,
                                role="acquisition", index=i, H=cfg["acquisition_H"])
                pool.append(ep); i += 1
            active = Controller(Memory(pool, cfg["memory_rows"]))
            gate.update(acquisition_episodes=i, reason="no learner; full allowance reinvested")
        else:
            # Discover new evidence with unrestricted long-horizon proposals first.
            independent = Controller(Memory(pool, cfg["memory_rows"]))
            ep = ev.episode(independent, seed=seed, arm=arm, cycle=cycle,
                            role="acquisition", index=0, H=cfg["acquisition_H"])
            pool.append(ep)
            memory = Memory(pool, cfg["memory_rows"])
            baseline = Controller(memory)
            incumbent = Controller(memory, active.kind, active.value)
            if incumbent.value is not None:
                retention = qualification(ev, incumbent, {"memory": baseline},
                                           seed=seed, arm=arm, cycle=cycle, role="retention")
                gate["retention"] = retention
                if not retention["accepted"]:
                    gate["revoked"] = True
                    incumbent = baseline
            reference = incumbent
            traces = [ev.episode(reference, seed=seed, arm=arm, cycle=cycle, role="collection", index=i,
                                 save_banks=True) for i in range(cfg["collection_roots"])]
            train = traces[:-cfg["heldout_roots"]]
            holdout = traces[-cfg["heldout_roots"]:]
            branch_traces = []
            for i, trace in enumerate(train[:cfg["training_branches"]] if arm == "mc" else []):
                # Uniform independent-bank action at the root's middle cached decision.
                point = trace["points"][len(trace["points"])//2]
                rng = np.random.default_rng(seed_id(seed, cycle, "branch", i))
                index = int(rng.integers(cfg["K"]))
                upper = 2*episode_upper(cfg, cfg["H"], reference.value is not None, point["remaining"])
                if fits_budget(ev, seed, arm, cycle, upper):
                    _, new = branches(ev, reference, point, index, seed=seed, arm=arm, cycle=cycle,
                                      role="branch", index=i, repeats=1)
                    branch_traces.extend(new)
            value = fit(ev, reference, arm, train, branch_traces, seed, cycle)
            proposal = Controller(memory, arm, value)
            calibration = heldout_check(ev, reference, proposal, holdout, seed=seed, arm=arm, cycle=cycle)
            gate["calibration"] = calibration
            active = incumbent
            if calibration["passed"]:
                qual = qualification(ev, proposal, {"memory": baseline, "incumbent": incumbent},
                                     seed=seed, arm=arm, cycle=cycle, role="qualification")
                gate["qualification"] = qual
                if qual["accepted"]:
                    active = proposal; gate["promoted"] = True
            # Only training-root episodes can enter the NEXT immutable archive.
            pool.extend(train + branch_traces)
            gate.update(target_controller=reference.version, model=value.digest,
                        acquisition_episodes=1, collection_successes=sum(e["success"] for e in train),
                        train_roots=[e["root"] for e in train], heldout_roots=[e["root"] for e in holdout])
        gate["deployed"] = active.version
        gate["training_work"] = ev.totals(seed=seed, arm=arm, cycle=cycle, phase="training")
        if gate["training_work"]["total_controls"] > cfg["cycle_training_controls"]:
            raise RuntimeError("Frozen training budget exceeded")
        ev.gates.append(gate)
        save_json(ev.out / "gates.json", ev.gates)
        curve = ev.curve(active, seed=seed, arm=arm, cycle=cycle)
        print(json.dumps(serial(dict(seed=seed, arm=arm, cycle=cycle, cost=curve["mean_cost"],
                                     success=curve["success_rate"], promoted=gate["promoted"],
                                     revoked=gate["revoked"], controls=curve["charged_controls"]))), flush=True)
    ev.curve(active, seed=seed, arm=arm, cycle=cfg["cycles"], final=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/independent_search_pilot.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if (cfg["steps"], cfg["goal"], cfg["H"]) != (180, .45, 16):
        raise ValueError("This frozen pilot's features, exact goal handling and controller defaults require steps=180, goal=.45, H=16")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Config is intentionally frozen: other studies should use new config + unique outputs.
    save_json(out / "config.json", cfg)
    hashes = {p: sha(ROOT / p) for p in SOURCES}
    for p in SOURCES:
        dest = out / "source" / p
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / p, dest)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    save_json(out / "manifest.json", dict(source_hashes=hashes, git_revision=revision,
              python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
              command=sys.argv, config_sha256=sha(args.config), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              scope="new files only; no historical runs are inputs or outputs"))
    from src.gps.independent_preflight import run_preflight
    preflight = run_preflight(out / "preflight.json")
    if preflight["status"] != "passed":
        raise RuntimeError("Minimal causal mechanism test failed; main trial not launched")
    if args.preflight_only:
        return
    ev = Evidence(out, cfg)
    try:
        for seed in cfg["seeds"]:
            empty = Controller(Memory([], cfg["memory_rows"]))
            bootstrap = [ev.episode(empty, seed=seed, arm="shared", cycle=-1,
                                     role="bootstrap", index=i, H=cfg["acquisition_H"])
                         for i in range(cfg["bootstrap_episodes"])]
            for arm in cfg["arms"]:
                run_arm(ev, seed=seed, arm=arm, bootstrap=bootstrap)
        if any(sha(ROOT/p) != digest for p, digest in hashes.items()):
            raise RuntimeError("Source changed during frozen trial")
        save_json(out / "episodes.json", ev.episodes)
        save_json(out / "work.json", dict(actual=ev.totals(),
                   per_arm={arm: ev.totals(arm=arm) for arm in ["shared"]+cfg["arms"]},
                   preflight=preflight["counts"], wall_seconds=time.perf_counter()-ev.start,
                   source_unchanged=True, completed=True))
        save_json(out / "status.json", dict(completed=True, elapsed_seconds=time.perf_counter()-ev.start))
    except BaseException:
        save_json(out / "episodes.json", ev.episodes)
        save_json(out / "work.json", dict(actual=ev.totals(), preflight=preflight["counts"],
                   wall_seconds=time.perf_counter()-ev.start, completed=False))
        save_json(out / "status.json", dict(completed=False, error=traceback.format_exc()))
        raise
    finally:
        ev.handle.close()
    from scripts.diagnostics.report_independent_search import report
    report(out)


if __name__ == "__main__":
    main()
