"""Bounded Walker study: independent MPPI, trajectory memory and measured values.

The value names ONE frozen canonical-restart reference controller. Complete
reference suffixes, never bootstrapped predictions, supply its training labels.
Accepted guidance changes later executed collection and therefore later query
locations. The second arm acquires its own experience with the same acquisition
schedule and a memory-aware controller, without fitting/using a value.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import shutil
import subprocess
import time

import numpy as np
import torch
import tyro

from scripts.runners.eval_policy_walker2d import provenance
from src.gps.coupled_planner import CoupledPlanner, MemoryArchive
from src.gps.expert_continuation import Snapshot, _restore, finite_absorbing_cost
from src.gps.global_value import MonteCarloValue
from src.mppi.mppi import MPPI
from src.rl.walker import WalkerRL
from src.utils.config import MPPIConfig

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CoupledConfig:
    iterations: int = 24
    total_steps: int = 1000
    seed: int = 0
    workers: int = 8
    bootstrap_episodes: int = 4
    collection_episodes: int = 2
    queries_per_iteration: int = 16
    qualification_episodes: int = 6
    endpoint_checks: int = 4
    evaluation_episodes: int = 16
    final_episodes: int = 64
    eval_every: int = 4
    K: int = 32
    H: int = 48
    memory_candidates: int = 8
    memory_stride: int = 16
    memory_entries: int = 2048
    value_updates: int = 600
    value_hidden: int = 128
    alpha: float = .5
    alpha_step: float = .1
    endpoint_rmse_max: float = .2
    support_radius: float = 1.5
    gamma: float = .999
    promotion_margin: float = 1.
    query_stride: int = 64
    # Alternate ordinary and modestly perturbed starts, identically in each arm.
    angle_noise: float = .03
    velocity_noise: float = .10


def validate(cfg: CoupledConfig) -> None:
    for key, value in asdict(cfg).items():
        if key in {"alpha", "alpha_step", "endpoint_rmse_max", "support_radius", "gamma", "promotion_margin", "angle_noise", "velocity_noise", "seed"}:
            continue
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg.seed) is not int or not 0 <= cfg.seed < 100 or cfg.H < 2 or cfg.H > cfg.total_steps:
        raise ValueError("Require seed<100 and 2<=H<=total_steps")
    if not 0 < cfg.alpha <= 1 or not 0 < cfg.gamma < 1 or not np.isfinite(cfg.support_radius) or cfg.support_radius <= 0:
        raise ValueError("Invalid guidance, discount or support radius")
    if not 0 < cfg.alpha_step <= cfg.alpha or not np.isfinite(cfg.endpoint_rmse_max) or cfg.endpoint_rmse_max <= 0:
        raise ValueError("Invalid earned-influence increment or endpoint calibration tolerance")
    if any(not np.isfinite(v) or v < 0 for v in (cfg.promotion_margin, cfg.angle_noise, cfg.velocity_noise)):
        raise ValueError("Invalid nonnegative margins/reset scales")
    if cfg.iterations * max(cfg.collection_episodes, cfg.qualification_episodes) >= 90_000:
        raise ValueError("Run exceeds disjoint reset allocation")


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision() -> str:
    revision=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True)
    if revision.returncode == 0:
        return revision.stdout.strip()
    recorded=ROOT/"SOURCE_REVISION"
    return recorded.read_text().strip() if recorded.exists() else "source-manifest-only"


def _mppi(cfg: CoupledConfig) -> MPPIConfig:
    return replace(MPPIConfig.load("walker2d"), K=cfg.K, H=cfg.H,
                   initial_iterations=2, lam_target_neff_fraction=None)


def _controller(archive=None, value=None, alpha=0.) -> dict:
    return dict(archive=archive, value=value, alpha=float(alpha))


def _reset(env, reset_id: int, cfg: CoupledConfig) -> str:
    env.reset(seed=reset_id)
    if reset_id % 2:
        rng = np.random.default_rng(reset_id ^ 0x79AC)
        state = env.walker.get_state()
        state[3] += rng.normal(0., cfg.angle_noise)
        state[10:19] += rng.normal(0., cfg.velocity_noise, 9)
        _restore(env, Snapshot(state, np.zeros(9)))
        return "angle_velocity_perturbed"
    return "standard"


def _episode(cfg: CoupledConfig, controller: dict, *, reset_id: int, planner_seed: int,
             snapshot: Snapshot | None = None, remaining: int | None = None,
             collect_points: bool = False, raw: bool = False) -> dict:
    torch.set_num_threads(1)
    n = cfg.total_steps if remaining is None else int(remaining)
    env = WalkerRL(gamma=cfg.gamma, max_episode_steps=n)
    ledger = dict(executed_controls=0, candidate_controls=0, memory_controls=0,
                  verification_controls=0, planner_calls=0)
    try:
        profile = _reset(env, reset_id, cfg) if snapshot is None else "restored_query"
        if snapshot is not None:
            _restore(env, snapshot)
        archive = (MemoryArchive.from_state_dict(controller["archive"])
                   if controller.get("archive") is not None else None)
        value = (MonteCarloValue.from_packet(controller["value"])
                 if controller.get("value") is not None else None)
        planner = (MPPI(env.walker, _mppi(cfg), seed=planner_seed) if raw else
            CoupledPlanner(env.walker, _mppi(cfg), archive=archive, value=value,
                alpha=controller["alpha"], seed=planner_seed, remaining=n,
                total_steps=cfg.total_steps, gamma=cfg.gamma,
                memory_candidates=cfg.memory_candidates, support_radius=cfg.support_radius))
        states, warmstarts = [env.walker.get_state()], [env.walker.get_warmstart()]
        costs, actions, velocities, points, sources = [], [], [], [], []
        guided, terminated = 0, False
        rng = np.random.default_rng(planner_seed ^ 0xDA71)
        for step in range(n):
            state, warm = env.walker.get_state(), env.walker.get_warmstart()
            action, info = planner.plan_step(state, initial_warmstart=warm)
            if raw:
                ledger["candidate_controls"] += cfg.K * cfg.H * int(info["optimization_iterations"])
            else:
                for key in ("candidate_controls", "memory_controls", "verification_controls"):
                    ledger[key] += int(info[key])
                guided += int(info["guidance_used"])
                sources.append(info["selected_source"])
                if collect_points and step % cfg.query_stride == 0 and n - step > cfg.H:
                    # Alternate the executed chosen plan and an independent bank
                    # draw. No privileged critical-state or successful prefix.
                    chosen = (planner.last_plan if len(points) % 2 == 0 else
                              planner.last_plans[int(rng.integers(len(planner.last_plans)))])
                    points.append(dict(snapshot=Snapshot(state, warm), remaining=n-step,
                        plan=chosen.copy(), step=step, selected=bool(len(points)%2 == 0)))
            ledger["planner_calls"] += 1
            _, _, terminated, truncated, info = env.step(action)
            actions.append(np.array(action, copy=True)); costs.append(float(info["task_cost"]))
            velocities.append(abs(float(info["x_velocity"]) - env.walker._v_target))
            states.append(env.walker.get_state()); warmstarts.append(env.walker.get_warmstart())
            if terminated or truncated:
                break
        raw_costs = np.asarray(costs)
        cost = float(np.dot(cfg.gamma ** np.arange(len(raw_costs)), raw_costs))
        if terminated:
            cost += cfg.gamma ** len(raw_costs) * finite_absorbing_cost(cfg.gamma, n-len(raw_costs))
        ledger["executed_controls"] = len(costs)
        return dict(reset_id=reset_id, planner_seed=planner_seed, reset_profile=profile,
            states=np.asarray(states), warmstarts=np.asarray(warmstarts),
            actions=np.asarray(actions).reshape(-1, 6), costs=raw_costs, gamma=cfg.gamma,
            remaining_start=n, terminated=bool(terminated), steps=len(costs),
            survived=bool(not terminated and len(costs) == n), duration_seconds=.008*len(costs),
            velocity_mae=float(np.mean(velocities)) if velocities else 0., cost=cost,
            guidance_fraction=guided/max(1,len(costs)), points=points, ledger=ledger,
            selected_sources={source: sources.count(source) for source in set(sources)})
    finally:
        env.close()


def _query(cfg: CoupledConfig, reference: dict, point: dict, *, reset_id: int,
           planner_seed: int, endpoint: bool) -> dict:
    snapshot, n = point["snapshot"], int(point["remaining"])
    prefix_controls = 0
    if endpoint:
        env = WalkerRL(gamma=cfg.gamma, max_episode_steps=n)
        try:
            _restore(env, snapshot)
            for action in point["plan"][:min(cfg.H, n-1)]:
                _, _, done, _, _ = env.step(action)
                prefix_controls += 1
                if done:
                    return dict(valid=False, reason="acquisition_prefix_fell", prefix_controls=prefix_controls)
            snapshot = Snapshot(env.walker.get_state(), env.walker.get_warmstart())
            n -= prefix_controls
        finally:
            env.close()
    result = _episode(cfg, reference, reset_id=reset_id, planner_seed=planner_seed,
                      snapshot=snapshot, remaining=n)
    return dict(valid=True, features=snapshot.state[2:].copy(), remaining=n,
                cost=result["cost"], prefix_controls=prefix_controls, episode=result)


def _job(job):
    kind, cfg = job["kind"], CoupledConfig(**job["config"])
    args = {key: value for key,value in job.items() if key not in {"kind", "config"}}
    return _query(cfg, **args) if kind == "query" else _episode(cfg, **args)


def _summary(episodes: list[dict], horizon: int) -> dict:
    if not episodes:
        raise ValueError("Cannot summarize an empty controller panel")
    result = dict(episodes=len(episodes), survival_count=sum(e["survived"] for e in episodes),
        survival_rate=float(np.mean([e["survived"] for e in episodes])),
        mean_duration_seconds=float(np.mean([e["duration_seconds"] for e in episodes])),
        velocity_mae=float(np.mean([e["velocity_mae"] for e in episodes])),
        finite_discounted_cost=float(np.mean([e["cost"] for e in episodes])),
        guidance_fraction=float(np.mean([e["guidance_fraction"] for e in episodes])))
    result["walking_gate"] = (result["survival_rate"] >= .9 and result["velocity_mae"] <= .3) if horizon == 1000 else None
    return result


def _public(episode: dict) -> dict:
    return {key: episode[key] for key in ("reset_id", "planner_seed", "reset_profile", "steps",
        "survived", "duration_seconds", "velocity_mae", "cost", "guidance_fraction", "selected_sources", "ledger")}


def _promote(candidate: list[dict], memory: list[dict], incumbent: list[dict], margin: float) -> bool:
    """Empirical development gate; no confidence or pointwise safety claim."""
    if (not candidate or len(candidate) != len(memory) or len(candidate) != len(incumbent)
            or any([a["reset_id"] for a in candidate] != [b["reset_id"] for b in panel]
                   for panel in (memory,incumbent))):
        raise ValueError("Promotion requires nonempty matched fresh reset panels")
    for reference in (memory, incumbent):
        differences = np.asarray([a["cost"]-b["cost"] for a,b in zip(candidate,reference)])
        if (float(differences.mean()) >= -margin or np.mean(differences < 0) < .5
                or sum(a["survived"] for a in candidate) < sum(a["survived"] for a in reference)):
            return False
    return True


def run(out_dir: Path, cfg: CoupledConfig, resume: bool = False) -> None:
    validate(cfg); torch.set_num_threads(1); start = time.monotonic(); elapsed_offset = 0.
    out_dir = out_dir.resolve()
    sources = ["scripts/training/train_coupled_value.py", "src/gps/coupled_planner.py",
        "src/gps/global_value.py", "src/gps/expert_continuation.py", "src/mppi/mppi.py",
        "src/rl/walker.py", "src/envs/base.py", "src/envs/walker2d.py", "src/envs/mujoco_env.py",
        "src/utils/config.py", "src/utils/math.py", "configs/walker2d_best.json",
        "configs/walker2d_task.json", "assets/walker2d.xml", "scripts/runners/eval_policy_walker2d.py"]
    if resume:
        protocol = json.loads((out_dir/"protocol.json").read_text())
        if protocol["config"] != asdict(cfg):
            raise ValueError("Resume must preserve the frozen configuration")
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        protocol = dict(config=asdict(cfg), source_sha256={name:_sha(ROOT/name) for name in sources},
            environment=provenance(), git_head=_revision(),
            reference_controller="Fixed bootstrap-memory + independent raw MPPI; fresh zero plan/private RNG at each suffix query. No learned value. Measured root return ONLY.",
            objective="Finite discounted raw task cost; charge fall once plus finite absorbing remainder. No standalone actor.",
            qualification="Fresh paired empirical full-episode gate; not confidence certification. Qualification traces never fit value or archive.",
            reset_distribution="Alternating standard resets and Gaussian angle(.03 default)/qvel(.10 default) perturbations. Same distribution and reset IDs across compared controllers.",
            reset_blocks=dict(bootstrap=20_000_000,collection=21_000_000,qualification=22_000_000,evaluation=23_000_000,final=24_000_000),
            interpretation="One paired training seed; matched acquisition schedules, actual planning/execution/qualification counts recorded; no efficiency or novelty claim.")
        for name in sources:
            dest=out_dir/"source"/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
        _json(out_dir/"protocol.json",protocol)
    for name,digest in protocol["source_sha256"].items():
        if _sha(ROOT/name) != digest or _sha(out_dir/"source"/name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    ledger: dict[str,int] = {}
    offset = cfg.seed*10_000_000
    def status(phase, iteration=-1, **extra):
        _json(out_dir/"study_status.json",dict(status="running",phase=phase,iteration=iteration,
            requested_iterations=cfg.iterations,elapsed_seconds=elapsed_offset+time.monotonic()-start,work=ledger,**extra))
    def charge(results, category):
        for row in results:
            for key,value in row.get("ledger",{}).items():
                name=f"{category}_{key}"; ledger[name]=ledger.get(name,0)+int(value)
    def jobs(pool, items):
        return list(pool.map(_job,[dict(config=asdict(cfg),**item) for item in items]))
    def rollout_jobs(controller, reset_ids, phase_seed, **extra):
        return [dict(kind="episode",controller=controller,reset_id=rid,
            planner_seed=phase_seed+i,**extra) for i,rid in enumerate(reset_ids)]
    def archive(episodes):
        return MemoryArchive.from_episodes(episodes,cfg.H,cfg.memory_stride,cfg.memory_entries).state_dict()
    checkpoint=out_dir/"checkpoint.pkl"
    try:
        with ProcessPoolExecutor(max_workers=cfg.workers,mp_context=mp.get_context("spawn")) as pool:
            if resume and checkpoint.exists():
                state=pickle.loads(checkpoint.read_bytes())
                ledger.update(state["ledger"]); begin=state["next_iteration"]
                elapsed_offset=float(state.get("elapsed_seconds",0.))
                arms, reference, records, histories = (state[k] for k in ("arms","reference","records","histories"))
                # Preserve uncommitted output after an interrupted append, then
                # reconstruct only iterations committed by the atomic checkpoint.
                if (out_dir/"metrics.jsonl").exists():
                    shutil.copy2(out_dir/"metrics.jsonl",out_dir/f"metrics_before_resume_{time.time_ns()}.jsonl")
                with (out_dir/"metrics.jsonl").open("w") as stream:
                    for i in range(begin):
                        committed=json.loads((out_dir/"metric_rows"/f"iteration_{i:04d}.json").read_text())
                        stream.write(json.dumps(committed,allow_nan=False)+"\n")
            else:
                status("bootstrap")
                bootstrap=jobs(pool,rollout_jobs(_controller(),range(20_000_000+offset,20_000_000+offset+cfg.bootstrap_episodes),1000+offset,raw=True))
                charge(bootstrap,"shared_bootstrap"); _save(out_dir/"bootstrap.pkl",bootstrap)
                initial_archive=archive(bootstrap)
                reference=_controller(initial_archive)
                _save(out_dir/"reference_controller.pkl",reference)
                arms={name:_controller(initial_archive) for name in ("guided","memory")}
                histories={name:list(bootstrap) for name in arms}; records={name:[] for name in arms}
                begin=0
            def evaluate(name, count, block):
                status(f"evaluation_{name}")
                ids=list(range(block+offset,block+offset+count)); panel={}
                for arm in ("guided","memory"):
                    rows=jobs(pool,rollout_jobs(arms[arm],ids,40_000_000+offset+block))
                    charge(rows,f"evaluation_{arm}")
                    panel[arm]=dict(summary=_summary(rows,cfg.total_steps),episodes=[_public(e) for e in rows])
                    # Preserve the preselected first reset's actual traces.
                    _save(out_dir/"evaluations"/f"{name}_{arm}_first_trace.pkl",rows[0])
                panel.update(name=name,controller_alpha=arms["guided"]["alpha"],work=ledger.copy())
                _json(out_dir/"evaluations"/f"{name}.json",panel)
                return panel
            if begin == 0 and not (out_dir/"evaluations"/"initial.json").exists():
                evaluate("initial",cfg.evaluation_episodes,23_000_000)
            def save_checkpoint(next_iteration):
                _save(checkpoint,dict(next_iteration=next_iteration,arms=arms,reference=reference,
                    records=records,histories=histories,ledger=ledger,
                    elapsed_seconds=elapsed_offset+time.monotonic()-start))
            save_checkpoint(begin)
            for iteration in range(begin,cfg.iterations):
                row=dict(iteration=iteration)
                proposed_archives={}
                for arm_index,arm in enumerate(("guided","memory")):
                    status(f"collection_{arm}",iteration)
                    reset_base=21_000_000+offset+iteration*cfg.collection_episodes
                    collected=jobs(pool,rollout_jobs(arms[arm],range(reset_base,reset_base+cfg.collection_episodes),
                        50_000_000+offset+iteration*1000,collect_points=True))
                    charge(collected,f"collection_{arm}")
                    row[f"{arm}_collection"]=_summary(collected,cfg.total_steps)
                    _save(out_dir/arm/f"collection_{iteration:04d}.pkl",collected)
                    histories[arm].extend(collected)
                    all_points=[(index,p) for index,e in enumerate(collected) for p in e["points"]]
                    if not all_points:
                        raise RuntimeError("No healthy executable collection query points")
                    rng=np.random.default_rng(cfg.seed*1_000_000+iteration)
                    queries=[]; groups=[]
                    for q in range(cfg.queries_per_iteration):
                        index,point=all_points[int(rng.integers(len(all_points)))]
                        queries.append(dict(kind="query",reference=reference,point=point,
                            reset_id=reset_base+index,planner_seed=60_000_000+offset+iteration*1000+q,
                            endpoint=bool(q%2)))
                        groups.append(iteration*cfg.collection_episodes+index)
                    status(f"continuation_queries_{arm}",iteration)
                    queried=jobs(pool,queries)
                    for q,group in zip(queried,groups):
                        name=f"acquisition_{arm}_prefix_controls"
                        ledger[name]=ledger.get(name,0)+q["prefix_controls"]
                        if q["valid"]:
                            records[arm].append(dict(features=q["features"],remaining=q["remaining"],cost=q["cost"],group=group))
                            histories[arm].append(q["episode"])
                            charge([q["episode"]],f"continuation_{arm}")
                    _save(out_dir/arm/f"queries_{iteration:04d}.pkl",queried)
                    proposed_archives[arm]=archive(histories[arm])
                    # Full traces live on disk; keep only arrays needed to rebuild
                    # bounded action memory, rather than duplicate candidate points.
                    for e in histories[arm]:
                        e["points"]=[]
                memory_new=_controller(proposed_archives["memory"])
                same_memory=_controller(proposed_archives["guided"])
                model=None; row["value_metrics"]=None
                data=records["guided"]
                if len({r["group"] for r in data}) >= 4:
                    status("value_fit",iteration)
                    model=MonteCarloValue.fit(np.asarray([r["features"] for r in data]),
                        np.asarray([r["remaining"] for r in data],dtype=np.int64),
                        np.asarray([r["cost"] for r in data]),np.asarray([r["group"] for r in data]),
                        total_steps=cfg.total_steps,gamma=cfg.gamma,seed=cfg.seed+iteration,
                        updates=cfg.value_updates,hidden=cfg.value_hidden,support_threshold=cfg.support_radius)
                    row["value_metrics"]=model.metrics
                    _save(out_dir/"guided"/f"value_{iteration:04d}.pkl",model.packet())
                eligible=model is not None and model.metrics["calibration_eligible"]
                proposed_alpha=min(cfg.alpha,arms["guided"]["alpha"]+cfg.alpha_step) if eligible else 0.
                candidate=_controller(proposed_archives["guided"],model.packet() if eligible else None,proposed_alpha)
                status("qualification",iteration)
                ids=list(range(22_000_000+offset+iteration*cfg.qualification_episodes,
                               22_000_000+offset+(iteration+1)*cfg.qualification_episodes))
                panels={}
                for label,controller in (("memory",same_memory),("incumbent",arms["guided"]),("candidate",candidate)):
                    if label == "candidate" and not eligible:
                        panels[label]=panels["memory"]
                        continue
                    panels[label]=jobs(pool,rollout_jobs(controller,ids,70_000_000+offset+iteration*1000,
                        collect_points=bool(label == "candidate" and eligible)))
                    charge(panels[label],f"qualification_{label}")
                endpoint_audit=dict(eligible=False,checked=0,normalized_rmse=None)
                if eligible:
                    selected=[p for e in panels["candidate"] for p in e["points"] if p["selected"]]
                    if selected:
                        # Independent measurements at endpoints selected by the
                        # newly fitted planner, held outside ALL future fitting.
                        indices=np.linspace(0,len(selected)-1,min(cfg.endpoint_checks,len(selected)),dtype=int)
                        audits=jobs(pool,[dict(kind="query",reference=reference,point=selected[index],
                            reset_id=ids[0],planner_seed=80_000_000+offset+iteration*1000+i,endpoint=True)
                            for i,index in enumerate(indices)])
                        errors=[]
                        for audit in audits:
                            key="qualification_endpoint_prefix_controls"
                            ledger[key]=ledger.get(key,0)+audit["prefix_controls"]
                            if audit["valid"]:
                                charge([audit["episode"]],"qualification_endpoint")
                                predicted,support,_=model.predict(audit["features"][None],np.asarray([audit["remaining"]]))
                                if support[0]:
                                    scale=finite_absorbing_cost(cfg.gamma,audit["remaining"])
                                    errors.append(float((predicted[0]-audit["cost"])/scale))
                        error=float(np.sqrt(np.mean(np.square(errors)))) if errors else None
                        endpoint_audit=dict(eligible=bool(len(errors)>=min(2,cfg.endpoint_checks)
                            and error is not None and error<=cfg.endpoint_rmse_max),checked=len(errors),normalized_rmse=error)
                        _save(out_dir/"qualification"/f"endpoint_audits_{iteration:04d}.pkl",audits)
                promote=bool(eligible and endpoint_audit["eligible"] and _promote(
                    panels["candidate"],panels["memory"],panels["incumbent"],cfg.promotion_margin))
                if promote:
                    arms["guided"]=candidate
                    decision="promote_guidance"
                elif np.mean([e["cost"] for e in panels["memory"]]) <= np.mean([e["cost"] for e in panels["incumbent"]]):
                    arms["guided"]=same_memory
                    decision="memory_refresh_guidance_revoked"
                else:
                    decision="retain_incumbent"
                arms["memory"]=memory_new
                row["promotion"]=dict(accepted=promote,type=decision,eligible=bool(eligible),
                    proposed_alpha=proposed_alpha,endpoint_audit=endpoint_audit,
                    guidance_used_fraction=_summary(panels["candidate"],cfg.total_steps)["guidance_fraction"],
                    **{name:_summary(panel,cfg.total_steps) for name,panel in panels.items()})
                _json(out_dir/"qualification"/f"iteration_{iteration:04d}.json",
                    {name:[_public(e) for e in panel] for name,panel in panels.items()})
                row.update(measured_query_count={arm:len(data) for arm,data in records.items()},
                    deployed_alpha=arms["guided"]["alpha"],work=ledger.copy(),elapsed_seconds=elapsed_offset+time.monotonic()-start)
                _save(out_dir/"controllers"/f"iteration_{iteration:04d}.pkl",arms)
                _json(out_dir/"metric_rows"/f"iteration_{iteration:04d}.json",row)
                with (out_dir/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row,allow_nan=False)+"\n")
                save_checkpoint(iteration+1)
                if (iteration+1)%cfg.eval_every == 0:
                    evaluate(f"iteration_{iteration:04d}",cfg.evaluation_episodes,23_000_000)
                print(json.dumps({key:row[key] for key in ("iteration","deployed_alpha","measured_query_count","elapsed_seconds")}),flush=True)
                save_checkpoint(iteration+1)
            final=evaluate("final",cfg.final_episodes,24_000_000)
            unchanged=all(_sha(ROOT/name)==digest and _sha(out_dir/"source"/name)==digest
                          for name,digest in protocol["source_sha256"].items())
            _json(out_dir/"validation.json",dict(source_unchanged=unchanged,
                completed_iterations=cfg.iterations,reference_controller_sha256=_sha(out_dir/"reference_controller.pkl"),
                no_standalone_actor=True,targets="complete measured fixed-reference restart suffix returns only"))
            _json(out_dir/"study_status.json",dict(status="complete",phase="complete",
                completed_iterations=cfg.iterations,requested_iterations=cfg.iterations,source_unchanged=unchanged,
                elapsed_seconds=elapsed_offset+time.monotonic()-start,work=ledger,final={a:final[a]["summary"] for a in arms}))
    except BaseException as error:
        _json(out_dir/"study_status.json",dict(status="failed",error=repr(error),
            elapsed_seconds=elapsed_offset+time.monotonic()-start,work=ledger))
        raise


def main(out_dir: Path, config: CoupledConfig = CoupledConfig(), resume: bool = False):
    run(out_dir,config,resume)


if __name__ == "__main__":
    tyro.cli(main)
