"""Finite, replayable stages of the bidirectional MPPI/action-policy study."""
from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import time

import numpy as np
import torch

from .task import AcrobotCaptureTask, TaskConfig, Snapshot
from .evidence import Archive, QueryExecutor, EpisodeTrace, file_sha256, ledger_totals
from .search import IndependentSearch, SearchConfig, TrajectoryMemory, GuidedSelector
from .policy import FamilyGaussianActor, policy_hash, controller_hash
from .learning import PolicyLearner, LearningConfig, BehaviorEpisode, AuxiliaryBatch
from .trust import TrustManager, CandidateTrustManager, Outcome
from .meter import attach_meter


DEFAULT = {
    "schema": "bidirectional-acrobot-v0", "task": asdict(TaskConfig()),
    "search": asdict(SearchConfig()), "learning": asdict(LearningConfig()),
    "backend": "cpu", "device": "cpu", "seeds": [0, 1, 2],
    "qualification": {"horizons": [48, 96, 144, 288], "development_roots": 8,
        "confirmation_roots": 32, "local_min": .9, "short_capture_max": .5,
        "long_capture_min": .9, "development_only": False},
    "acquisition": {"episodes": 16, "minimum_successes": 4},
    "development": {"rounds": 2, "updates_per_round": 4, "behavior_contexts": 16,
        "replicates": 4, "validation_roots": 16, "teacher_episodes": 4,
        "handoff_anchors": 8, "qualification_roots": 8,
        "support_radius": .15, "agreement_ladder": [.1, 1., 10.],
        "retain_unpromoted_training_updates": False,
        "agreement_qualification_mode": "sequential"},
    "evaluation": {"roots": 64, "capture_min": .9},
    "roots": {"qualification_dev": 110000, "qualification_confirm": 120000,
        "acquisition": 210000, "actor_training": 310000, "actor_validation": 410000,
        "handoff": 510000, "controller_qualification": 610000,
        "feedback_acquisition": 710000, "feedback_validation": 810000, "evaluation": 910000},
}


def canonical(value):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,default=convert)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    text=json.dumps(json.loads(canonical(value)),indent=2,sort_keys=True,allow_nan=False)+"\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush(); os.fsync(stream.fileno())


def load_config(path=None):
    config = deepcopy(DEFAULT)
    if path:
        supplied = json.loads(Path(path).read_text())
        for key, value in supplied.items():
            if key not in config:
                raise ValueError(f"Unknown configuration key {key}")
            if isinstance(config[key], dict):
                unknown = set(value)-set(config[key])
                if unknown:
                    raise ValueError(f"Unknown {key} fields: {unknown}")
                config[key].update(value)
            else:
                config[key] = value
    TaskConfig.from_dict(config["task"]); SearchConfig(**config["search"])
    LearningConfig(**config["learning"])
    if (not config["seeds"] or len(set(config["seeds"]))!=len(config["seeds"])
            or any(type(s) is not int or s<0 for s in config["seeds"])):
        raise ValueError("Training seeds must be distinct nonnegative integers")
    for group,fields in {"qualification":("development_roots","confirmation_roots"),
            "acquisition":("episodes","minimum_successes"),
            "development":("rounds","updates_per_round","behavior_contexts","replicates",
                "validation_roots","teacher_episodes","handoff_anchors","qualification_roots"),
            "evaluation":("roots",)}.items():
        if any(type(config[group][field]) is not int or config[group][field]<1 for field in fields):
            raise ValueError(f"{group} rosters must be finite positive integer counts")
    if config["acquisition"]["minimum_successes"]>config["acquisition"]["episodes"]:
        raise ValueError("Acquisition success requirement exceeds its declared episode roster")
    horizons=config["qualification"]["horizons"]
    if not horizons or len(set(horizons))!=len(horizons) or any(type(h) is not int or h<1 for h in horizons):
        raise ValueError("Qualification horizons must be distinct positive integers")
    if config["development"]["agreement_ladder"]!=[.1,1.,10.]:
        raise ValueError("This trust protocol uses the declared .1,1,10 agreement ladder")
    if type(config["development"]["retain_unpromoted_training_updates"]) is not bool:
        raise ValueError("Retaining unpromoted training updates must be an explicit boolean")
    if config["development"]["agreement_qualification_mode"] not in ("sequential","independent_candidates"):
        raise ValueError("Unknown agreement qualification mode")
    if type(config["qualification"]["development_only"]) is not bool:
        raise ValueError("Development-only qualification must be an explicit boolean")
    for value in [config["evaluation"]["capture_min"]]+[config["qualification"][key]
            for key in ("local_min","short_capture_max","long_capture_min")]:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=1:
            raise ValueError("Outcome thresholds must be finite fractions")
    if config["development"]["replicates"] < 2:
        raise ValueError("LOO requires independent repeated trajectories")
    if config["development"]["behavior_contexts"] % 2:
        raise ValueError("Half ordinary, half acquired contexts require an even count")
    if len(set(config["roots"].values())) != len(config["roots"]):
        raise ValueError("Root namespace bases must be disjoint")
    return config


def freeze_source(destination, config, stage):
    """Bind the actual checkout, including new/untracked source, before work."""
    root = Path(__file__).resolve().parents[3]
    destination.mkdir(parents=True, exist_ok=False)
    source = destination/"source"
    sources = list((root/"src").rglob("*.py"))
    sources += list((root/"assets").rglob("*.xml"))
    sources += list((root/"tests").glob("test_bidirectional_*.py"))
    sources += [root/"scripts/training/train_bidirectional_gps.py",
        root/"scripts/diagnostics/run_bidirectional_tests.py",
        root/"scripts/diagnostics/run_independent_temporal_assay.py",
        root/"scripts/diagnostics/run_ending_precision_assay.py",
        root/"scripts/diagnostics/run_ending_interpolation_assay.py",
        root/"scripts/diagnostics/run_ending_qfilter_assay.py",
        root/"scripts/diagnostics/run_ending_own_occupancy_assay.py",
        root/"scripts/diagnostics/run_independent_phase_assay.py",
        root/"scripts/diagnostics/run_native_shooting_reference.py",
        root/"scripts/diagnostics/run_affine_native_reference.py",
        root/"scripts/diagnostics/run_constrained_native_reference.py",
        root/"scripts/diagnostics/run_native_rollout_reference.py",
        root/"scripts/diagnostics/run_collocated_pfl_reference.py",
        root/"scripts/diagnostics/run_ending_early_context_loop.py",
        root/"scripts/diagnostics/run_ending_coherent_assay.py",
        root/"scripts/diagnostics/run_ending_geometry_assay.py",
        root/"scripts/diagnostics/run_ending_conditional_feedback.py",
        root/"scripts/diagnostics/run_return_curriculum_loop.py",
        root/"scripts/diagnostics/run_deployment_adjoint_loop.py",
        root/"scripts/diagnostics/run_learned_ending_pilot.py",root/"pyproject.toml"]
    sources += list((root/"docs").glob("bidirectional_*.md"))
    sources += list((root/"configs").glob("bidirectional_*.json"))
    inventory = []
    for path in sources:
        if not path.is_file():
            continue
        target = source/path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        inventory.append(dict(file=str(target.relative_to(destination)), sha256=file_sha256(target)))
    write_json(destination/"config.json", config)
    import importlib.metadata
    versions={}
    for name in ("torch","numpy","scipy","mujoco","mujoco-warp","warp-lang"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]="unavailable"
    write_json(destination/"manifest.json", dict(schema="bidirectional-source-v1", stage=stage,
        config_sha256=file_sha256(destination/"config.json"), files=inventory,
        hardware=platform.platform(), python=platform.python_version(), torch=torch.__version__,
        dependency_versions=versions,cuda_available=torch.cuda.is_available(), cuda_device=(torch.cuda.get_device_name()
            if torch.cuda.is_available() else None)))


def family_number(trace):
    return 1 if trace.family == "negative" else 0


def execute_episode(executor, initial, episode_id, *, actor=None, family=0,
                    deterministic=True, generator=None, planner=None, selector=None,
                    trial_weight=None, source=None):
    """Exactly one controller executes all remaining controls; no rescue."""
    if (actor is None) == (planner is None):
        raise ValueError("Select exactly one controller")
    executor.archive.work_scope=executor.scope
    executor.task.restore(initial)
    if actor is not None:attach_meter(actor,executor.archive,"executed_actor")
    actor_id = policy_hash(actor) if actor is not None else (
        policy_hash(selector.actor) if selector is not None else None)
    controller_id = controller_hash(actor, deterministic=deterministic, family=family) if actor is not None else identity({
        "planner": asdict(planner.config), "guided_actor": actor_id,
        "family": family, "trial_weight": trial_weight,
        "trust": [asdict(r) for r in selector.trust.records] if selector is not None else None})
    steps, qids, latent, likelihood = [], [], [], []
    influenced = changed = 0
    owner = dict(actor_hash=actor_id, controller_hash=controller_id,
        semantics="actor" if actor is not None else "executed_replanning",
        family=family, deterministic=deterministic)
    while executor.task.remaining:
        snapshot = executor.task.snapshot()
        if planner is not None:
            bank = planner.propose(snapshot)
            selection = selector.select(bank, trial_weight=trial_weight) if selector is not None else None
            plan = bank.independent_plan if selection is None else selection.plan
            action = plan[0]
            influenced += int(selection is not None and selection.influenced)
            changed += int(selection is not None and selection.changed)
            saved = executor.archive.save_npz(f"decisions/{episode_id}_{snapshot.elapsed:05d}.npz",
                nominal_before=bank.proposal_state_before["nominal"],
                nominal_after=bank.proposal_state_after["nominal"],
                rng_before=np.array(canonical(bank.proposal_state_before["rng"])),
                rng_after=np.array(canonical(bank.proposal_state_after["rng"])),
                initial_plan_before=np.array(bank.proposal_state_before["initial"]),
                independent_plan=bank.independent_plan, executed_plan=plan,
                independent_score=np.array(bank.independent_score),
                bank_digest=np.array(bank.digest), sources=np.asarray(bank.source))
            executor.archive.append("decision", episode_id=episode_id, elapsed=snapshot.elapsed,
                controller_hash=controller_id, actor_hash=actor_id, family=family,
                bank_query_ids=bank.query_ids, guided_query_ids=(() if selection is None else selection.query_ids),
                influenced=bool(selection is not None and selection.influenced),
                changed=bool(selection is not None and selection.changed), raw=saved)
        elif deterministic:
            action = actor.act(executor.task.observation(), family=family)
        else:
            with torch.no_grad():
                result = actor.sample(executor.task.observation(), family, generator=generator)
            action = result["actions"].cpu().numpy()
            latent.append(result["pre_tanh"].cpu().numpy())
            likelihood.append(float(result["log_prob"].cpu()))
        step = executor.step(np.asarray(action, dtype=np.float64), metadata={"owner": owner,
            "episode_id": episode_id, "ancestry": initial.to_dict()})
        steps.append(step); qids.append(executor.latest_query_id)
    trace = EpisodeTrace.from_steps(initial, steps, config=executor.task.config,
        actor_owner=owner, execution_semantics=owner["semantics"], source={} if source is None else source,
        query_ids=qids)
    raw = trace.archive(executor.archive, episode_id)
    behavior = None
    if actor is not None and not deterministic:
        raw_costs = trace.stage_costs+trace.terminal_costs+trace.capture_failures
        behavior = BehaviorEpisode(trace.observations[:-1], trace.actions, np.asarray(latent),
            np.asarray(likelihood), raw_costs, initial.context_id,
            executor.task.config.T-initial.elapsed, actor_id, controller_id,
            family=family, query_ref=str(executor.archive.path/raw["file"]))
        saved_behavior=executor.archive.save_npz(f"behavior/{episode_id}.npz", pre_tanh=np.asarray(latent),
            old_log_probs=np.asarray(likelihood), raw_costs=raw_costs,
            episode_file=np.array(raw["file"]), behavior_actor_hash=np.array(actor_id),
            behavior_controller_hash=np.array(controller_id))
        executor.archive.append("behavior_saved",episode_id=episode_id,raw=saved_behavior,
            behavior_actor_hash=actor_id,behavior_controller_hash=controller_id)
    summary = dict(episode_id=episode_id, context_id=initial.context_id,
        cost=trace.total_cost, capture=bool(trace.captured), complete=bool(trace.elapsed[-1]==executor.task.config.T),
        ordinary_start=initial.elapsed==0 and initial.context_id.split(":")[0] in ("ordinary","actor_eval","actor_train"),
        root_start=initial.elapsed==0, remaining_steps=executor.task.config.T-initial.elapsed,
        local=bool(np.any((trace.elapsed<=executor.task.config.local_by)&
            (trace.tip_distances<=executor.task.config.local_distance))),
        actor_hash=actor_id, controller_hash=controller_id, family_id=family,
        task_hash=initial.task_hash, query_ids=qids, influenced=influenced, changed=changed,
        raw=str(executor.archive.path/raw["file"]), deterministic=deterministic)
    executor.archive.append("episode_outcome", **summary)
    return trace, behavior, summary


def summary(rows):
    return dict(episodes=len(rows), mean_cost=float(np.mean([r["cost"] for r in rows])),
        capture_rate=float(np.mean([r["capture"] for r in rows])),
        local_rate=float(np.mean([r["local"] for r in rows])),
        changed_controls=sum(r["changed"] for r in rows),
        influenced_controls=sum(r["influenced"] for r in rows))


def make_executor(config, out, scope):
    archive = Archive(Path(out)/"evidence"/scope)
    task = AcrobotCaptureTask(TaskConfig.from_dict(config["task"]), backend=config["backend"])
    return QueryExecutor(task, archive, scope=scope)


def root(task, seed, namespace):
    return task.reset(seed=int(seed), context_id=f"{namespace}:{seed}")


def teacher_panel(executor, cfg, root_seeds, tag, *, actor=None, trust=None,
                  family=0, trial_weight=None, memory=None, planner_seed=0,update_memory=True):
    rows, traces = [], []
    for i, seed in enumerate(root_seeds):
        initial = root(executor.task, seed, "ordinary")
        planner = IndependentSearch(executor, cfg, seed=planner_seed+i,
            memory=TrajectoryMemory() if memory is None else memory)
        selector = GuidedSelector(executor, actor, trust, family, cfg.temperature) if actor is not None else None
        trace, _, outcome = execute_episode(executor, initial, f"{tag}_{i:04d}",
            planner=planner, selector=selector, family=family, trial_weight=trial_weight,
            source={"reset_seed": int(seed), "planner_seed": planner_seed+i})
        rows.append(outcome); traces.append(trace)
        if memory is not None and update_memory and actor is None:
            memory.add(trace.observations[:-1], trace.actions, executor.task.config.T,
                outcome["episode_id"], family_number(trace))
    return rows, traces


def qualify(config, out, previous=None):
    executor = make_executor(config, out, "qualification")
    q = config["qualification"]; roster = []
    base = SearchConfig(**config["search"])
    try:
        binding=identity(dict(task=executor.task.task_hash,search=asdict(base),
            horizons=q["horizons"],development_roots=q["development_roots"],
            root=config["roots"]["qualification_dev"]))
        if previous is not None:
            if previous.get("roster_binding")!=binding:
                raise ValueError("Prior development roster does not bind this exact task/search/root panel")
            roster=previous["roster"]
        else:
            for h in q["horizons"]:
                rows, _ = teacher_panel(executor, replace(base, H=h),
                    range(config["roots"]["qualification_dev"], config["roots"]["qualification_dev"]+q["development_roots"]), f"dev_h{h}")
                roster.append(dict(H=h, K=base.K, **summary(rows), outcomes=rows))
                write_json(Path(out)/f"horizon_{h}.json", roster[-1])
        pairs = [(s,l) for s in roster for l in roster if s["H"]<l["H"]
            and s["local_rate"]>=q["local_min"] and s["capture_rate"]<=q["short_capture_max"]
            and l["capture_rate"]>=q["long_capture_min"]]
        result = dict(qualified=False, failure="no_development_horizon_pair", roster=roster,
            roster_binding=binding,development_only=q["development_only"],task_hash=executor.task.task_hash,
            independent_search_config=asdict(base))
        if pairs and q["development_only"]:
            short,long=max(pairs,key=lambda p:(p[0]["H"],p[1]["H"]))
            result.update(failure="fresh_confirmation_pending",short_H=short["H"],long_H=long["H"])
        if pairs and not q["development_only"]:
            short, long = max(pairs, key=lambda p:(p[0]["H"],p[1]["H"]))
            seeds = list(range(config["roots"]["qualification_confirm"],config["roots"]["qualification_confirm"]+q["confirmation_roots"]))
            conditions = []
            matched_k = int(np.ceil((base.K+base.smooth_candidates+base.memory_candidates)*long["H"]/short["H"]))-base.smooth_candidates-base.memory_candidates
            for label, cfg in (("short",replace(base,H=short["H"])), ("long",replace(base,H=long["H"])),
                ("short_matched_work",replace(base,H=short["H"],K=max(base.K,matched_k)))):
                rows,_=teacher_panel(executor,cfg,seeds,"confirm_"+label,planner_seed=10000)
                conditions.append(dict(condition=label,H=cfg.H,K=cfg.K,**summary(rows),outcomes=rows))
            s,l,m=conditions
            passed=(s["local_rate"]>=q["local_min"] and s["capture_rate"]<=q["short_capture_max"]
                and l["capture_rate"]>=q["long_capture_min"] and m["capture_rate"]<=q["short_capture_max"])
            result.update(qualified=passed,failure=None if passed else "fresh_confirmation_or_matched_work_failed",
                short_H=short["H"],long_H=long["H"],confirmation=conditions,
                horizon_interpretation="supports these controllers only; endpoint heuristics also vary with H")
        result["accounting"]=executor.archive.totals()
        write_json(Path(out)/"qualification.json",result)
        return result
    finally:
        executor.task.close();executor.archive.close()


def acquire(config,out,qualification=None):
    if qualification is not None and not qualification.get("qualified"):
        raise ValueError("Failed task qualification: acquisition for the claimed horizon study is blocked")
    executor=make_executor(config,out,"acquisition")
    cfg=SearchConfig(**config["search"])
    if qualification is not None:cfg=replace(cfg,H=qualification["long_H"])
    memory=TrajectoryMemory()
    reference_archive=None
    try:
        count=config["acquisition"]["episodes"];base=config["roots"]["acquisition"]
        seeds=list(range(base,base+count))
        rows,traces=teacher_panel(executor,cfg,seeds,"acquisition",memory=memory)
        from .dataset import FamilyDataset
        dataset=FamilyDataset(Path(out)/"families",executor.task.config)
        for i,trace in enumerate(traces):dataset.append(trace,f"independent_{i:04d}")
        captures=sum(t.successful for t in traces)
        independent_accounting=executor.archive.totals()
        # A fixed analytic control is a separately paid benchmark. Its actions
        # and successful outcomes never enter the independent discovery gate,
        # trajectory memory, learner corpus or teacher auxiliary labels.
        from .reference import EnergyLQRReference, execute_reference
        reference_archive=Archive(Path(out)/"evidence"/"analytic_reference")
        reference_executor=QueryExecutor(executor.task,reference_archive,
            scope="analytic_reference",owner={"semantics":"analytic_reference",
                "excluded_from_teacher_aux":True})
        reference=EnergyLQRReference(executor.task)
        reference_archive.append("reference_panel_requested",root_seeds=seeds,
            matched_independent_root_seeds=True,controller_hash=reference.controller_identity,
            excluded_from_teacher_aux=True,reference_config=reference.identity_payload)
        reference_rows=[]
        for i,seed in enumerate(seeds):
            initial=root(executor.task,seed,"acquisition")
            _,_,row=execute_reference(reference_executor,initial,f"analytic_reference_{i:04d}",
                reference=reference,source={"reset_seed":seed,
                    "matched_independent_episode_id":rows[i]["episode_id"],
                    "role":"reference_only","excluded_from_teacher_aux":True})
            reference_rows.append(row)
        reference_archive.close()
        reference_accounting=reference_archive.totals()
        accounting={key:(bool(independent_accounting[key] or reference_accounting[key])
            if isinstance(independent_accounting[key],bool) else
            independent_accounting[key]+reference_accounting[key]) for key in independent_accounting}
        result=dict(qualified=captures>=config["acquisition"]["minimum_successes"],
            failure=None if captures>=config["acquisition"]["minimum_successes"] else "insufficient_successful_discovery",
            summary=summary(rows),outcomes=rows,independent_accounting=independent_accounting,
            reference_only=dict(summary=summary(reference_rows),outcomes=reference_rows,
                accounting=reference_accounting,controller_hash=reference.controller_identity,
                excluded_from_teacher_aux=True,
                ledger=str(reference_archive.ledger_path.relative_to(Path(out))),
                ledger_sha256=file_sha256(reference_archive.ledger_path)),
            accounting=accounting,
            accounting_scope="Sum of distinct independent acquisition and analytic-reference ledgers; components are not extra charges")
        write_json(Path(out)/"acquisition.json",result)
        return result
    finally:
        if reference_archive is not None:reference_archive.close()
        executor.task.close();executor.archive.close()


def load_traces(path,successful_only=True):
    # FamilyDataset contains immutable indexed copies of the same executions.
    # Discovery counts come from the actual query archives exactly once.
    root=Path(path)
    if (root/"evidence").is_dir():root=root/"evidence"
    paths=sorted(root.rglob("episodes/**/*.npz"))
    bindings={}
    for p in paths:
        archive=next(parent.parent for parent in p.parents if parent.name=="episodes")
        if archive not in bindings:
            bindings[archive]={r["raw"]["file"]:r["raw"] for r in
                (json.loads(s) for s in (archive/"ledger.jsonl").read_text().splitlines())
                if r["kind"]=="episode_saved"}
        ref=bindings[archive].get(str(p.relative_to(archive)))
        if ref is None or ref["sha256"]!=file_sha256(p) or ref["bytes"]!=p.stat().st_size:
            raise ValueError("Episode is not bound by its original durable archive: "+str(p))
    traces=[EpisodeTrace.from_npz(p) for p in paths]
    return [t for t in traces if t.successful and t.execution_semantics=="executed_replanning"] if successful_only else traces


def auxiliary(traces,family=None):
    traces=[t for t in traces if t.successful and t.execution_semantics=="executed_replanning"
        and (family is None or family_number(t)==family)]
    if not traces:return None
    return AuxiliaryBatch(np.concatenate([t.observations[:-1] for t in traces]),
        np.concatenate([t.actions for t in traces]),
        np.concatenate([np.full(len(t.actions),family_number(t),dtype=np.int64) for t in traces]))


def evaluate_actor(executor,actor,family,seeds,tag,*,deterministic=True,generator=None):
    rows=[]
    for i,seed in enumerate(seeds):
        _,_,row=execute_episode(executor,root(executor.task,seed,"actor_eval"),f"{tag}_{i:04d}",
            actor=actor,family=family,deterministic=deterministic,generator=generator)
        rows.append(row)
    return rows


def promote_actor(base,proposed):
    if [r["context_id"] for r in base]!=[r["context_id"] for r in proposed]:
        raise ValueError("Actor qualification contexts differ")
    return bool(sum(r["capture"] for r in proposed)>=sum(r["capture"] for r in base)
        and np.mean([r["cost"] for r in proposed])<np.mean([r["cost"] for r in base])-1e-6)


def behavior_data(executor,learner,traces,family,seed,tag,config):
    d=config["development"];contexts=[]
    half=d["behavior_contexts"]//2
    for i in range(half):
        contexts.append(root(executor.task,seed+i,"actor_train"))
    for i in range(half):
        trace=traces[i%len(traces)]
        # End-first contexts still execute the full ACTOR remainder. Ordinary
        # root qualification remains separate from this training distribution.
        t=int([.875,.75,.5,.25][i%4]*executor.task.config.T)
        t=min(t,len(trace.actions)-1)
        contexts.append(Snapshot(trace.states[t],trace.warmstarts[t],int(trace.elapsed[t]),
            int(trace.hold_counts[t]),trace.initial_snapshot.task_hash,f"{tag}:acquired:{i}"))
    episodes=[]
    for i,context in enumerate(contexts):
        for j in range(d["replicates"]):
            ancestry={"reset_ancestry":context.to_dict()}
            if i>=half:
                tr=traces[(i-half)%len(traces)]
                ancestry.update(root_context_id=tr.initial_snapshot.context_id,teacher_source=tr.source,
                    teacher_initial=tr.initial_snapshot.to_dict(),teacher_time_index=context.elapsed)
            _,ep,_=execute_episode(executor,context,f"{tag}_c{i:03d}_r{j}",actor=learner.actor,
                family=family,deterministic=False,generator=learner.torch_generator,
                source=ancestry)
            episodes.append(ep)
    return episodes


def checkpoint(out,label,state,archive):
    path=Path(out)/"checkpoints"/(label+".pt")
    path.parent.mkdir(exist_ok=True)
    state["source_fingerprint"]=source_fingerprint(out)
    if path.exists():raise FileExistsError(path)
    with path.open("xb") as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    archive.append("checkpoint",label=label,file=str(path),sha256=file_sha256(path),
        commit_boundary="between complete episodes/updates; interrupted queries remain charged")
    return path


def source_fingerprint(out):
    manifest=json.loads((Path(out)/"manifest.json").read_text())
    return identity(dict(files=sorted((r["file"],r["sha256"]) for r in manifest["files"]),
        dependencies=manifest.get("dependency_versions",{})))


def charged_arm_work(archive,seed,branch):
    """Actual own-arm paid repeats and full shared ancestry, deduplicated."""
    from .accounting import arm_lineage_work
    work=arm_lineage_work(archive.path.parents[1],seed,branch)
    return dict(controls=work["requested_controls"],neural_rows=work["neural_rows"],
        gradient_steps=work["gradient_steps"],shared_assessment_controls=work["shared_assessment_controls"],
        unallocated_events=len(work["unallocated"]),efficiency_claim_blocked=work["efficiency_claim_blocked"],
        accounting_source_sha256=file_sha256(Path(__file__).with_name("accounting.py")),
        query_references=len(work["queryrefs"]),
        scope="Recompute exact reference roster with arm_lineage_work; no neural-to-control conversion")


def outcome(row,semantics):
    return Outcome(context_id=row["context_id"],cost=row["cost"],capture=row["capture"],
        complete=row["complete"],controller_hash=row["controller_hash"],actor_hash=row["actor_hash"],
        family_id=row["family_id"],remaining_steps=row["remaining_steps"],task_hash=row["task_hash"],
        query_ids=tuple(row["query_ids"]),semantics=semantics,deterministic=row["deterministic"])


def qualify_handoffs(executor,actor,traces,family,tag,config):
    manager=TrustManager();ah=policy_hash(actor)
    d=config["development"];anchors=[];rows=[]
    eligible=[t for t in traces if t.successful and family_number(t)==family]
    if not eligible:return manager
    for i in range(d["handoff_anchors"]):
        trace=eligible[i%len(eligible)]
        phase=(i//2)%4
        lower=phase*executor.task.config.T//4
        minimum_tail=min(max(1,executor.task.config.T//4),max(1,int(.5/executor.task.config.dt)))
        upper=min((phase+1)*executor.task.config.T//4-1,executor.task.config.T-minimum_tail)
        t=lower if i%2==0 else max(lower,upper)
        state=trace.states[t].copy();warm=trace.warmstarts[t].copy()
        # Alternate exact restoration and fixed small physical perturbations.
        if i%2:state[1:3]+=.01;state[3:5]+=.025
        snapshot=Snapshot(state,warm,t,int(trace.hold_counts[t]) if i%2==0 else 0,
            trace.initial_snapshot.task_hash,f"{tag}:handoff:{i}")
        anchors.append(snapshot)
        _,_,row=execute_episode(executor,snapshot,f"{tag}_handoff_{i}",actor=actor,family=family,
            source={"root_context_id":trace.initial_snapshot.context_id,"teacher_source":trace.source,
                "teacher_initial":trace.initial_snapshot.to_dict(),"time_index":t,
                "intervention":"exact" if i%2==0 else "angle+.01_velocity+.025_hold_reset"})
        rows.append(outcome(row,"actor_suffix"))
    for phase in range(4):
        ids=[i for i,a in enumerate(anchors) if int(min(3,a.elapsed*4//executor.task.config.T))==phase]
        if not ids:continue
        selected=[anchors[i] for i in ids]
        elapsed=[a.elapsed for a in selected];remaining=[executor.task.config.T-t for t in elapsed]
        rec=manager.register(actor_hash=ah,actor_controller_hash=controller_hash(actor,deterministic=True,family=family),
            task_hash=executor.task.task_hash,family_id=family,total_steps=executor.task.config.T,
            time_interval=(min(elapsed),max(elapsed)),remaining_interval=(min(remaining),max(remaining)),
            support_anchors=np.stack([executor.task.observation(a) for a in selected]),
            observation_scale=np.array([1,1,1,1,2.5,5,1,1]),support_radius=d["support_radius"],
            anchor_context_ids=tuple(a.context_id for a in selected))
        manager.qualify(rec.record_id,[rows[i] for i in ids])
    executor.archive.append("handoff_qualification",actor_hash=ah,records=[asdict(r) for r in manager.records])
    return manager


def qualify_guidance(executor,actor,traces,family,cfg,tag,config,panel_offset=0):
    manager=qualify_handoffs(executor,actor,traces,family,tag,config)
    common_memory=TrajectoryMemory()
    for i,t in enumerate(traces):
        if t.actor_owner.get("actor_hash") is not None:continue
        common_memory.add(t.observations[:-1],t.actions,executor.task.config.T,f"qualification_shared_{i}",family_number(t))
    d=config["development"];base_seed=config["roots"]["controller_qualification"]+panel_offset
    if d.get("agreement_qualification_mode","sequential")=="independent_candidates":
        return _qualify_independent_agreement(executor,actor,family,cfg,tag,d,base_seed,manager,common_memory)
    for weight in d["agreement_ladder"]:
        active=[r for r in manager.records if r.state in ("eligible","promoted")]
        if not active:break
        # Each attempt consumes fresh reset clusters. Provisional guidance is
        # explicitly labelled; it is never confused with promoted deployment.
        offset=d["agreement_ladder"].index(weight)*d["qualification_roots"]
        seeds=list(range(base_seed+offset,base_seed+offset+d["qualification_roots"]))
        baseline,_=teacher_panel(executor,cfg,seeds,f"{tag}_qbase_{weight}",family=family,planner_seed=seed_offset(tag),
            memory=deepcopy(common_memory),update_memory=False)
        guided,_=teacher_panel(executor,cfg,seeds,f"{tag}_qguided_{weight}",actor=actor,trust=manager,
            family=family,trial_weight=weight,planner_seed=seed_offset(tag),memory=deepcopy(common_memory),update_memory=False)
        # Execution identities include the exact pending weight and support.
        for rec in list(active):
            manager.promote(rec.record_id,weight,[outcome(r,"executed_replanning") for r in baseline],
                [outcome(r,"executed_replanning") for r in guided],
                baseline_controller_hash=baseline[0]["controller_hash"],
                guided_controller_hash=guided[0]["controller_hash"])
        executor.archive.append("controller_qualification",weight=weight,
            baseline=summary(baseline),guided=summary(guided),records=[asdict(r) for r in manager.records])
        if not any(r.state=="promoted" for r in manager.records):break
    return manager


def _qualify_independent_agreement(executor,actor,family,cfg,tag,development,base_seed,
        eligible_manager,common_memory):
    """Three isolated candidate tests followed by one fresh selected test.

    The original actor-suffix evidence is reused explicitly as a fixed support
    packet. Candidate outcome panels are disjoint; no candidate manager is
    returned for deployment before a separate complete-controller confirmation.
    """
    eligible=[r for r in eligible_manager.records if r.state=="eligible"]
    if not eligible:
        executor.archive.append("agreement_candidate_ranking",tag=tag,candidates=[],
            selected_weight=None,confirmation_executed=False,reason="no_eligible_actor_suffix")
        return eligible_manager
    count=development["qualification_roots"];candidates=[]
    def panel(weight,index,purpose):
        manager=CandidateTrustManager.from_eligible(eligible_manager,weight)
        seeds=list(range(base_seed+index*count,base_seed+(index+1)*count))
        baseline,_=teacher_panel(executor,cfg,seeds,f"{tag}_{purpose}_base_{weight}",
            family=family,planner_seed=seed_offset(tag),memory=deepcopy(common_memory),update_memory=False)
        guided,_=teacher_panel(executor,cfg,seeds,f"{tag}_{purpose}_guided_{weight}",actor=actor,
            trust=manager,family=family,trial_weight=weight,planner_seed=seed_offset(tag),
            memory=deepcopy(common_memory),update_memory=False)
        for record in tuple(manager.records):
            manager.promote(record.record_id,weight,[outcome(r,"executed_replanning") for r in baseline],
                [outcome(r,"executed_replanning") for r in guided],
                baseline_controller_hash=baseline[0]["controller_hash"],
                guided_controller_hash=guided[0]["controller_hash"])
        passed=any(r.state=="promoted" for r in manager.records)
        executor.archive.append("controller_qualification",tag=tag,weight=weight,
            qualification_mode="independent_candidates",purpose=purpose,seeds=seeds,
            baseline=summary(baseline),guided=summary(guided),
            baseline_outcomes=baseline,guided_outcomes=guided,
            support_handoff_query_ids=sorted({q for r in eligible for q in r.handoff_query_ids}),
            support_packet_reused=True,passed=passed,
            deployment_authorized=bool(purpose=="selected_confirmation" and passed),
            records=[asdict(r) for r in manager.records])
        return manager,dict(weight=float(weight),passed=passed,baseline=summary(baseline),
            guided=summary(guided),seeds=seeds)
    for index,weight in enumerate(development["agreement_ladder"]):
        _,record=panel(weight,index,"candidate")
        candidates.append(record)
    passed=[row for row in candidates if row["passed"]]
    chosen=min(passed,key=lambda row:(-row["guided"]["capture_rate"],row["guided"]["mean_cost"],row["weight"])) if passed else None
    executor.archive.append("agreement_candidate_ranking",tag=tag,candidates=candidates,
        ranking_rule="passed candidates: descending guided capture, ascending guided cost, ascending weight",
        selected_weight=None if chosen is None else chosen["weight"],
        confirmation_executed=chosen is not None,
        confirmation_seeds=list(range(base_seed+3*count,base_seed+4*count)) if chosen is not None else [])
    if chosen is None:return eligible_manager
    confirmed,_=panel(chosen["weight"],3,"selected_confirmation")
    return confirmed


def seed_offset(tag):
    return int(hashlib.sha256(tag.encode()).hexdigest()[:7],16)


def _resume_config_matches(saved_hash,config):
    if saved_hash==identity(config):return True
    development=config["development"]
    if (development.get("retain_unpromoted_training_updates",False) or
            development.get("agreement_qualification_mode","sequential")!="sequential"):
        return False
    # Old checkpoints did not serialize these two opt-in defaults. Their
    # frozen source/dependency fingerprint is still checked independently.
    legacy=deepcopy(config)
    for key in ("retain_unpromoted_training_updates","agreement_qualification_mode"):
        legacy["development"].pop(key,None)
    return saved_hash==identity(legacy)


def _validate_retained_resume(saved):
    names={"independent_data","bidirectional"}
    if (saved.get("training_deployment_state_schema")!="separate-training-deployment-v1" or
            set(saved.get("deployed_learners",{}))!=names or
            set(saved.get("deployed_families",{}))!=names or
            set(saved.get("learners",{}))!=names or set(saved.get("families",{}))!=names):
        raise ValueError("Retained training resume requires both complete training and deployment states")
    if any(type(saved[key][name]) is not int or saved[key][name] not in (0,1)
            for key in ("families","deployed_families") for name in names):
        raise ValueError("Retained training resume requires explicit training and deployment families")
    required={"schema","policy","config","optimizer","numpy_rng","torch_rng","updates"}
    if any(not isinstance(saved[key][name],dict) or not required.issubset(saved[key][name])
            for key in ("learners","deployed_learners") for name in names):
        raise ValueError("Retained training resume requires exact optimizer and private RNG state for both actors")


def _next_actor_states(training,deployed,candidate,accepted,retain,*,deployment_candidate=None):
    """A failed deployment check does not erase an opt-in training update."""
    if retain:
        deployed_candidate=candidate if deployment_candidate is None else deployment_candidate
        return candidate,deployed_candidate.clone_for_trial() if accepted else deployed
    return (candidate,candidate) if accepted else (training,deployed)


def develop(config,out,corpus,qualification=None,resume=None):
    if qualification is not None and not qualification.get("qualified"):
        raise ValueError("Task qualification failed; do not silently run a horizon-gap study")
    traces=load_traces(corpus)
    if len(traces)<config["acquisition"]["minimum_successes"]:
        raise ValueError("Useful complete independent experience must precede learning")
    if any(t.initial_snapshot.task_hash!=TaskConfig.from_dict(config["task"]).task_hash
           or t.actor_owner.get("actor_hash") is not None for t in traces):
        raise ValueError("Initial corpus must be independent and match this exact task/model")
    if config["schema"]=="bidirectional-acrobot-v0":
        for t in traces:
            delta=np.arctan2(np.sin(t.states[0,1:3]-[np.pi,0]),np.cos(t.states[0,1:3]-[np.pi,0]))
            if np.any(np.abs(delta)>config["task"]["reset_angle_radius"]+1e-10) or np.any(
                    np.abs(t.states[0,3:5])>config["task"]["reset_velocity_radius"]+1e-10):
                raise ValueError("Ordinary Acrobot study cannot bootstrap from a mechanism-only upright witness")
    parent=Path(corpus).resolve()
    corpus_accounting=report(parent)
    write_json(Path(out)/"corpus_provenance.json",dict(path=str(parent),
        manifest_sha256=file_sha256(parent/"manifest.json") if (parent/"manifest.json").exists() else None,
        full_shared_cost=corpus_accounting["lineage_unique_requested_controls"],
        charge_to_each_standalone_arm="full shared cost, not half",successful_episodes=len(traces)))
    retain=config["development"].get("retain_unpromoted_training_updates",False)
    saved_resume=None
    if resume:
        saved_resume=torch.load(resume,map_location=config["device"],weights_only=False)
        if saved_resume.get("source_fingerprint")!=source_fingerprint(out):
            raise ValueError("Exact resume requires identical frozen source and dependency versions; use the parent's source tree")
        if not _resume_config_matches(saved_resume["config_hash"],config):raise ValueError("Resume config mismatch")
        if retain:_validate_retained_resume(saved_resume)
    all_results=[] if saved_resume is None else saved_resume.get("completed_results",[])
    cfg=SearchConfig(**config["search"])
    if qualification is not None:cfg=replace(cfg,H=qualification["short_H"])
    for seed in config["seeds"]:
        resume_this_seed=saved_resume is not None and saved_resume["seed"]==seed
        if saved_resume is not None and config["seeds"].index(seed)<config["seeds"].index(saved_resume["seed"]):continue
        executor=make_executor(config,out,f"development_seed{seed}")
        try:
            from .dataset import FamilyDataset
            dataset=FamilyDataset(Path(out)/f"families_seed{seed}",executor.task.config)
            for i,t in enumerate(traces):dataset.append(t,f"shared_{i:04d}")
            actor=FamilyGaussianActor(seed=seed,initial_std=.3).to(config["device"])
            learner=PolicyLearner(actor,LearningConfig(**config["learning"]),seed=seed)
            if not resume_this_seed:
                attach_meter(learner.actor,executor.archive,"auxiliary_bootstrap")
                boot_start=time.perf_counter()
                executor.archive.work_scope=f"seed{seed}/shared_bootstrap"
                boot=learner.bootstrap(auxiliary(traces))
                executor.archive.append("fit_work",purpose="shared_bootstrap",seed=seed,
                    fit_wall_seconds=time.perf_counter()-boot_start,**boot)
            branches={name:learner.clone_for_trial() for name in ("independent_data","bidirectional")}
            family=0 if any(family_number(t)==0 for t in traces) else 1
            family_by_branch={name:family for name in branches}
            deployed_branches={name:value.clone_for_trial() for name,value in branches.items()} if retain else branches
            deployed_families=dict(family_by_branch) if retain else family_by_branch
            memories={name:TrajectoryMemory() for name in branches}
            corpora={name:list(traces) for name in branches}
            for name in memories:
                for i,t in enumerate(traces):
                    memories[name].add(t.observations[:-1],t.actions,config["task"]["T"],
                        f"shared_{i:04d}",family_number(t))
            round_start=0
            if resume_this_seed:
                saved=saved_resume
                if saved["seed"]!=seed or not _resume_config_matches(saved["config_hash"],config):raise ValueError("Resume seed/config mismatch")
                for name in branches:branches[name].load_state_dict(saved["learners"][name])
                family_by_branch=saved["families"];memories=saved["memories"];corpora=saved["corpora"]
                if retain:
                    for name in deployed_branches:deployed_branches[name].load_state_dict(saved["deployed_learners"][name])
                    deployed_families=saved["deployed_families"]
                else:deployed_families=family_by_branch
                round_start=saved["next_round"]
            for round_index in range(round_start,config["development"]["rounds"]):
                for branch,current in branches.items():
                    executor.scope=f"seed{seed}/{branch}"
                    family=family_by_branch[branch];tag=f"s{seed}_r{round_index}_{branch}"
                    deployed=deployed_branches[branch]
                    guiding_family=deployed_families[branch]
                    trust=qualify_guidance(executor,deployed.actor,corpora[branch],guiding_family,cfg,tag,config,
                        panel_offset=seed*10000+round_index*1000) if branch=="bidirectional" else None
                    count=config["development"]["teacher_episodes"]
                    b=config["roots"]["feedback_acquisition"]+seed*10000+round_index*100
                    probe_memory=deepcopy(memories[branch])
                    rows,new=teacher_panel(executor,cfg,range(b,b+count),tag+"_teacher",
                        actor=deployed.actor if trust is not None else None,trust=trust,
                        family=guiding_family if retain else family,
                        memory=memories[branch],planner_seed=seed+round_index*1000,
                        update_memory=branch!="bidirectional")
                    for i,t in enumerate(new):dataset.append(t,tag+f"_teacher_{i:04d}")
                    probe_raw=None
                    if branch=="bidirectional":
                        _,probe_raw=teacher_panel(executor,cfg,range(b,b+count),tag+"_paired_raw_teacher",
                            family=family,memory=probe_memory,planner_seed=seed+round_index*1000,update_memory=False)
                        for i,t in enumerate(probe_raw):
                            dataset.append(t,tag+f"_paired_raw_{i:04d}")
                            memories[branch].add(t.observations[:-1],t.actions,executor.task.config.T,
                                tag+f"_paired_raw_{i:04d}",family_number(t))
                    corpora[branch].extend(t for t in new if t.successful)
                    for update in range(config["development"]["updates_per_round"]):
                        deployed=deployed_branches[branch]
                        update_tag=tag+f"_u{update}"
                        contexts_seed=config["roots"]["actor_training"]+seed*10000+round_index*1000+update*32
                        data=behavior_data(executor,current,corpora[branch],family,contexts_seed,update_tag,config)
                        if update==0 and probe_raw is not None:
                            from .feedback import matched_child_probe
                            probe_base=config["roots"]["feedback_validation"]+seed*10000+round_index*100
                            probe_options=dict(guiding_actor_hash=policy_hash(deployed.actor),
                                shared_teacher_traces=traces) if retain else {}
                            probe=matched_child_probe(executor,current,data,probe_raw,new,family,
                                range(probe_base,probe_base+config["development"]["validation_roots"]),tag+"_feedback_probe",
                                **probe_options)
                            executor.archive.append("feedback_probe",seed=seed,round=round_index,branch=branch,**probe)
                        root_start=config["roots"]["actor_validation"]+seed*10000+round_index*1000+update*32
                        panel=list(range(root_start,root_start+config["development"]["validation_roots"]))
                        base=evaluate_actor(executor,deployed.actor,deployed_families[branch],panel,update_tag+"_base")
                        stochastic_seed=seed_offset(update_tag)
                        base_stochastic=evaluate_actor(executor,deployed.actor,deployed_families[branch],panel,update_tag+"_base_stochastic",
                            deterministic=False,generator=torch.Generator().manual_seed(stochastic_seed))
                        # Same behavior corpus and optimizer starting point for
                        # prospective teaching-family choices; teacher outcomes
                        # never become actor-owned return targets.
                        trials=[]
                        target_families=sorted({family_number(t) for t in corpora[branch]})
                        behavior_owner=policy_hash(current.actor)
                        deployed_before=policy_hash(deployed.actor)
                        for target_family in target_families:
                            candidate=current.clone_for_trial()
                            attach_meter(candidate.actor,executor.archive,"mc_and_auxiliary_fit")
                            fit_start=time.perf_counter()
                            learning=candidate.update(data,behavior_owner,aux=auxiliary(corpora[branch],target_family))
                            executor.archive.append("fit_work",purpose="candidate_update",seed=seed,
                                round=round_index,branch=branch,fit_wall_seconds=time.perf_counter()-fit_start,**learning)
                            proposed=evaluate_actor(executor,candidate.actor,target_family,panel,update_tag+f"_family{target_family}")
                            proposed_stochastic=evaluate_actor(executor,candidate.actor,target_family,panel,
                                update_tag+f"_family{target_family}_stochastic",deterministic=False,
                                generator=torch.Generator().manual_seed(stochastic_seed))
                            accepted=promote_actor(base,proposed) and promote_actor(base_stochastic,proposed_stochastic)
                            checkpoint_state=dict(learner=candidate.state_dict(),qualification=proposed,accepted=accepted)
                            if retain:checkpoint_state.update(stochastic_qualification=proposed_stochastic,
                                training_deployment_state_schema="separate-training-deployment-v1",
                                behavior_actor_hash=behavior_owner,deployed_actor_hash=deployed_before,
                                training_family=family,deployed_family=deployed_families[branch],
                                deployment_baseline=base,deployment_stochastic_baseline=base_stochastic,
                                training_candidate=True)
                            checkpoint(out,update_tag+f"_family{target_family}",checkpoint_state,executor.archive)
                            trials.append((accepted,summary(proposed)["capture_rate"],summary(proposed)["mean_cost"],target_family,candidate,learning,proposed))
                        eligible=[t for t in trials if t[0]]
                        selected=min(eligible,key=lambda t:(-t[1],t[2],t[3])) if eligible else None
                        if retain:
                            # Family is fixed within every complete episode.
                            # Only these declared development panels choose the
                            # next family; final panels never affect learning.
                            training_selected=min(trials,key=lambda t:(-t[1],t[2],t[3]))
                            current,deployed=_next_actor_states(current,deployed,training_selected[4],
                                selected is not None,True,
                                deployment_candidate=None if selected is None else selected[4])
                            branches[branch]=current;deployed_branches[branch]=deployed
                            family=training_selected[3];family_by_branch[branch]=family
                            if selected is not None:deployed_families[branch]=selected[3]
                        elif selected is not None:
                            current=selected[4];branches[branch]=current
                            family=selected[3];family_by_branch[branch]=family
                        state_metadata={} if not retain else dict(training_update_retained=True,
                            behavior_actor_hash=behavior_owner,training_actor_hash=policy_hash(current.actor),
                            deployed_actor_hash_before=deployed_before,deployed_actor_hash=policy_hash(deployed.actor),
                            training_family=family,deployed_family=deployed_families[branch],
                            training_selected_family=training_selected[3],
                            training_selection_rule="all same-parent family trials: complete development capture descending, cost ascending, family ascending",
                            deployment_selection_rule="accepted deterministic AND stochastic complete-root trials: capture descending, cost ascending, family ascending",
                            selection_panels="fresh development panels; final heldout panels excluded",
                            deployment_baseline=base,deployment_stochastic_baseline=base_stochastic)
                        executor.archive.append("actor_update",seed=seed,round=round_index,branch=branch,
                            update=update,base=summary(base),accepted=selected is not None,
                            selected_teacher_family=None if selected is None else selected[3],
                            cumulative_controls=executor.archive.totals()["requested_controls"],
                            charged_arm_work=charged_arm_work(executor.archive,seed,branch),
                            trials=[dict(accepted=t[0],capture=t[1],cost=t[2],family=t[3],learning=t[5]) for t in trials],
                            **state_metadata)
                committed=dict(config_hash=identity(config),seed=seed,
                    next_round=round_index+1,learners={n:l.state_dict() for n,l in branches.items()},
                    families=family_by_branch,memories=memories,corpora=corpora,completed_results=all_results)
                if retain:committed.update(training_deployment_state_schema="separate-training-deployment-v1",
                    deployed_learners={n:l.state_dict() for n,l in deployed_branches.items()},
                    deployed_families=deployed_families)
                checkpoint(out,f"seed{seed}_round{round_index}_committed",committed,executor.archive)
            start=config["roots"]["evaluation"]
            for branch,current in deployed_branches.items():
                executor.scope=f"seed{seed}/{branch}/final"
                rows=evaluate_actor(executor,current.actor,deployed_families[branch],
                    range(start,start+config["evaluation"]["roots"]),f"s{seed}_{branch}_final")
                all_results.append(dict(seed=seed,branch=branch,**summary(rows),outcomes=rows,
                    passed=summary(rows)["capture_rate"]>=config["evaluation"]["capture_min"]))
                final_state=dict(learner=current.state_dict(),family=deployed_families[branch],
                    evaluation=rows,config_hash=identity(config))
                if retain:final_state.update(training_deployment_state_schema="separate-training-deployment-v1",
                    training_learner=branches[branch].state_dict(),training_family=family_by_branch[branch])
                checkpoint(out,f"seed{seed}_{branch}_final",final_state,executor.archive)
            # Blind controller panel: both selectors receive the SAME frozen
            # trajectory memory. No final outcomes update the actor or trust.
            current=deployed_branches["bidirectional"];family=deployed_families["bidirectional"]
            executor.scope=f"seed{seed}/controller_final_qualification"
            trust=qualify_guidance(executor,current.actor,corpora["bidirectional"],family,cfg,
                f"s{seed}_final_controller",config,panel_offset=seed*10000+(config["development"]["rounds"]+1)*1000)
            frozen_memory=deepcopy(memories["bidirectional"])
            eval_seeds=range(config["roots"]["evaluation"],config["roots"]["evaluation"]+config["evaluation"]["roots"])
            executor.scope=f"seed{seed}/independent_search/final"
            baseline,_=teacher_panel(executor,cfg,eval_seeds,f"s{seed}_final_raw",family=family,
                memory=deepcopy(frozen_memory),planner_seed=seed+90000,update_memory=False)
            executor.scope=f"seed{seed}/guided_search/final"
            guided,_=teacher_panel(executor,cfg,eval_seeds,f"s{seed}_final_guided",actor=current.actor,
                trust=trust,family=family,memory=deepcopy(frozen_memory),planner_seed=seed+90000,update_memory=False)
            write_json(Path(out)/f"seed{seed}_controllers.json",dict(baseline=summary(baseline),guided=summary(guided),
                paired_outcomes=list(zip(baseline,guided)),
                passed=promote_actor(baseline,guided) and sum(r["changed"] for r in guided)>0,
                efficiency_claim=False))
            write_json(Path(out)/f"seed{seed}_accounting.json",executor.archive.totals())
        finally:executor.task.close();executor.archive.close()
    per_condition={name:bool(len([r for r in all_results if r["branch"]==name])==len(config["seeds"])
        and all(r["passed"] for r in all_results if r["branch"]==name))
        for name in ("independent_data","bidirectional")}
    controller_results=[json.loads(p.read_text()) for p in sorted(Path(out).glob("seed*_controllers.json"))]
    evidence=report(out)
    probes=[r for r in evidence["events"] if r["kind"]=="feedback_probe"]
    result=dict(results=all_results,standalone_pass=all(per_condition.values()),
        standalone_pass_by_condition=per_condition,
        task_qualification_passed=bool(qualification is not None and qualification.get("qualified")),
        decision_influence_observed=any(r["guided"]["changed_controls"]>0 for r in controller_results),
        executed_control_pass=bool(len(controller_results)==len(config["seeds"]) and all(r["passed"] for r in controller_results)),
        matched_feedback_probes=probes,
        bidirectional_learning_pass=bool(len(probes)==len(config["seeds"])*config["development"]["rounds"]
            and all(p["guided_data_benefit_observed"] for p in probes)),
        efficiency_claim=False,
        mechanism_only=config["schema"]!="bidirectional-acrobot-v0",
        verdict="Evaluate each causal link separately; standalone capture is not an efficiency claim")
    if retain:result.update(training_deployment_state_schema="separate-training-deployment-v1",
        final_actor_evaluation_owner="deployed_actor",unpromoted_training_actor_is_not_deployed=True,
        agreement_qualification_mode=config["development"].get("agreement_qualification_mode","sequential"))
    write_json(Path(out)/"development.json",result)
    return result


def report(path,_visited=None):
    path=Path(path).resolve();totals=[];events=[];work=[];arm_controls={};fit_counts=[]
    visited=set() if _visited is None else _visited
    if str(path) in visited:return dict(unique_requested_controls=0,lineage_unique_requested_controls=0)
    visited.add(str(path))
    for ledger in path.rglob("ledger.jsonl"):
        totals.append(dict(file=str(ledger.relative_to(path)),**ledger_totals(ledger)))
        ledger_work=[]
        for line in ledger.read_text().splitlines():
            row=json.loads(line)
            if row["kind"]=="query_requested":
                arm_controls[row["scope"]]=arm_controls.get(row["scope"],0)+row["requested_controls"]
            if row["kind"] in ("neural_requested","neural_complete","fit_work","reference_arithmetic",
                    "optimizer_requested","optimizer_complete","optimizer_error",
                    "gradient_backward_requested","gradient_backward_complete","gradient_backward_error",
                    "diagnostic_backward_requested","diagnostic_backward_complete","diagnostic_backward_error"):
                work.append(row);ledger_work.append(row)
            if row["kind"] in ("actor_update","episode_outcome","controller_qualification","agreement_candidate_ranking","feedback_probe"):
                events.append(row)
        requested=[r for r in ledger_work if r["kind"]=="optimizer_requested"]
        diagnostic=[r for r in ledger_work if r["kind"]=="diagnostic_backward_requested"]
        legacy=sum(r.get("updates",r.get("gradient_steps",0)) for r in ledger_work if r["kind"]=="fit_work")
        fit_counts.append(dict(requested=sum(r["updates"] for r in requested) if requested else legacy,
            completed=sum(r["updates"] for r in ledger_work if r["kind"]=="optimizer_complete") if requested else legacy,
            legacy=legacy if not requested else 0,
            diagnostic=sum(r["passes"] for r in diagnostic) if diagnostic else
                sum(r.get("diagnostic_backward_passes",0) for r in ledger_work if r["kind"]=="fit_work")))
    result=dict(ledgers=totals,unique_requested_controls=sum(r["requested_controls"] for r in totals),
        completed_controls=sum(r["completed_controls"] for r in totals),
        unknown_controls=sum(r["unknown_controls"] for r in totals),events=events,
        requested_physics_frames=sum(r["requested_physics_frames"] for r in totals),
        completed_physics_frames=sum(r["completed_physics_frames"] for r in totals),
        cutoff_skipped_controls=sum(r["cutoff_skipped_controls"] for r in totals),
        rejected_queries=sum(r["rejected_queries"] for r in totals),
        error_queries=sum(r["error_queries"] for r in totals),
        neural_rows_requested=sum(r["rows"] for r in work if r["kind"]=="neural_requested"),
        neural_rows_returned=sum(r["rows"] for r in work if r["kind"]=="neural_complete" and r["returned"]),
        gradient_updates=sum(r["requested"] for r in fit_counts),
        optimizer_steps_completed=sum(r["completed"] for r in fit_counts),
        legacy_postfit_summary_steps=sum(r["legacy"] for r in fit_counts),
        optimizer_steps_failed=sum(r["updates"] for r in work if r["kind"]=="optimizer_error"),
        gradient_backward_passes=sum(r["passes"] for r in work if r["kind"]=="gradient_backward_requested"),
        fit_wall_seconds=sum(r.get("fit_wall_seconds",0) for r in work if r["kind"]=="fit_work"),
        diagnostic_backward_passes=sum(r["diagnostic"] for r in fit_counts),
        diagnostic_backward_wall_seconds=sum(r.get("diagnostic_wall_seconds",0) for r in work if r["kind"]=="fit_work"),
        arm_query_controls=arm_controls)
    parents=[];unresolved=[]
    provenance=path/"input_provenance.json"
    if provenance.exists():
        for ref in json.loads(provenance.read_text()).values():
            if "parent_run" in ref:
                parent=Path(ref["parent_run"])
                if parent.resolve()!=path and parent.exists():
                    parent_result=report(parent,visited)
                    parents.append(dict(path=str(parent),charged_controls=parent_result["lineage_unique_requested_controls"]))
                elif not parent.exists():unresolved.append(dict(path=str(parent),reference=ref))
    result["parent_costs"]=parents
    result["unresolved_parent_costs"]=unresolved
    result["lineage_unique_requested_controls"]=result["unique_requested_controls"]+sum(p["charged_controls"] for p in parents)
    return result
