"""A matched-data child assay; this function never promotes either child.

Unlike drifting development branches, both children start from the SAME full
parent learner state and consume the SAME complete behavior-owned MC targets.
Only the equal-sized actual teacher auxiliary dataset differs. Complete fresh
actor-only rollouts supply the comparison; teacher plans are never its target.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np


def _digest(value):
    """Content hash of raw arrays/state, independent of pickle/storage IDs."""
    digest=sha256()
    def feed(item):
        if hasattr(item,"detach"):
            item=item.detach().cpu().numpy()
        if isinstance(item,np.ndarray):
            array=np.ascontiguousarray(item)
            digest.update(json.dumps([array.dtype.str,list(array.shape)]).encode());digest.update(array.tobytes())
        elif isinstance(item,np.generic):feed(item.item())
        elif is_dataclass(item):feed(asdict(item))
        elif isinstance(item,dict):
            digest.update(b"{")
            for key in sorted(item,key=lambda key:(type(key).__name__,str(key))):feed(key);feed(item[key])
            digest.update(b"}")
        elif isinstance(item,(list,tuple)):
            digest.update(b"[")
            for part in item:feed(part)
            digest.update(b"]")
        else:digest.update(json.dumps(item,sort_keys=True,allow_nan=False,separators=(",",":")).encode()+b";")
    feed(value)
    return digest.hexdigest()


def _checkpoint_root(archive_path):
    path=Path(archive_path).resolve()
    for candidate in (path,*path.parents):
        if (candidate/"manifest.json").is_file():
            relative=path.relative_to(candidate)
            if len(relative.parts)<2 or relative.parts[0]!="evidence":
                raise ValueError("Matched feedback archive must be under its frozen run/evidence/<scope> before any evaluation")
            return candidate
    raise ValueError("Matched feedback requires an ancestor frozen run manifest before any evaluation")


def _callbacks(supplied,archive_path):
    if supplied is not None:
        # Explicit injection exists only to test wiring without engines or
        # network forwards. Production uses these lazy real implementations.
        return supplied
    from .workflow import auxiliary,evaluate_actor,summary,checkpoint,source_fingerprint
    from .policy import policy_hash,controller_hash
    from .meter import attach_meter
    root=_checkpoint_root(archive_path)
    # Fail before all paid evaluations/fits if the run-root checkpoint contract
    # cannot be satisfied. Archive paths themselves are nested under evidence.
    source_fingerprint(root)
    def bound_checkpoint(_archive_path,label,state,archive):
        return checkpoint(root,label,state,archive)
    return dict(auxiliary=auxiliary,evaluate_actor=evaluate_actor,summary=summary,
        checkpoint=bound_checkpoint,checkpoint_root=str(root),policy_hash=policy_hash,
        controller_hash=controller_hash,attach_meter=attach_meter)


def _validate_actor_panel(rows,*,actor_hash,controller_hash,family,task_hash,count):
    if len(rows)!=count or len({row.get("context_id") for row in rows})!=count:
        raise ValueError("Matched actor evaluation must cover distinct complete fresh contexts")
    refs=[]
    for row in rows:
        if (not row.get("complete") or not row.get("deterministic") or not row.get("root_start")
                or not np.isfinite(row.get("cost",np.nan)) or row.get("actor_hash")!=actor_hash
                or row.get("controller_hash")!=controller_hash or row.get("family_id")!=family
                or row.get("task_hash")!=task_hash or not row.get("query_ids")):
            raise ValueError("Matched scores must be the exact deterministic actor's own complete root continuation")
        refs.extend(row["query_ids"])
    if len(refs)!=len(set(refs)) or any(not isinstance(ref,str) or not ref for ref in refs):
        raise ValueError("Matched actor panels cannot reuse physical control queries")
    return refs


def _comparison(base,proposed):
    if [row["context_id"] for row in base] != [row["context_id"] for row in proposed]:
        raise ValueError("Matched child evaluation contexts differ")
    if not base or any(not row.get("complete") or not row.get("deterministic") or
            not row.get("root_start") or not np.isfinite(row.get("cost",np.nan)) for row in base+proposed):
        raise ValueError("Matched child scores require complete deterministic actor-only root outcomes")
    gain=float(np.mean([a["cost"]-b["cost"] for a,b in zip(base,proposed)]))
    nonregression=sum(row["capture"] for row in proposed)>=sum(row["capture"] for row in base)
    return dict(mean_cost_gain=gain,capture_nonregression=bool(nonregression),
        improved=bool(nonregression and gain>1e-6),
        paired=[dict(context_id=a["context_id"],cost_gain=a["cost"]-b["cost"],
            baseline_capture=a["capture"],proposed_capture=b["capture"],
            baseline_query_ids=a["query_ids"],proposed_query_ids=b["query_ids"]) for a,b in zip(base,proposed)])


def matched_child_probe(executor,parent_learner,behavior_episodes,raw_teacher_traces,
        guided_teacher_traces,family,seeds,tag,*,callbacks=None,
        guiding_actor_hash=None,shared_teacher_traces=()):
    """Run actual paired children, retaining optimizer/RNG/raw-data evidence.

    Caller supplies fresh seed clusters (the workflow reserves 810000 onward)
    and matching raw/guided teacher panels obtained with frozen shared memory.
    Freshness and model equivalence remain externally auditable obligations;
    this helper cannot infer them from numeric seed values. Failures propagate
    after writing a probe error; query/inference ledgers retain all paid work.
    """
    from .dataset import FAMILY_IDS,validate_teacher
    from .learning import AuxiliaryBatch,BehaviorEpisode,build_mc_batch
    archive=executor.archive;started=time.perf_counter();cb=_callbacks(callbacks,archive.path)
    if not isinstance(tag,str) or not tag or any(c not in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
        raise ValueError("Matched probe tag must be a safe unique path component")
    if type(family) is not int or family not in FAMILY_IDS.values():
        raise ValueError("Matched probe requires one persistent positive/negative family")
    seeds=list(seeds)
    if len(seeds)<2 or len(set(seeds))!=len(seeds) or any(type(seed) is not int or seed<0 for seed in seeds):
        raise ValueError("At least two distinct fresh evaluation reset seeds required")
    episodes=[BehaviorEpisode(**value) if isinstance(value,dict) else value for value in behavior_episodes]
    raw=list(raw_teacher_traces);guided=list(guided_teacher_traces)
    shared=list(shared_teacher_traces)
    parent_hash=cb["policy_hash"](parent_learner.actor)
    guiding_hash=parent_hash if guiding_actor_hash is None else guiding_actor_hash
    if not isinstance(guiding_hash,str) or not guiding_hash:
        raise ValueError("Guiding actor identity must be explicit")
    parent_state=parent_learner.state_dict();parent_state_digest=_digest(parent_state)
    behavior_digest=_digest(episodes)
    # Zero-inference validation of complete original-behavior ownership/LOO.
    build_mc_batch(episodes,parent_hash,minimum_replicates=parent_learner.cfg.minimum_replicates)
    if any(ep.family!=family for ep in episodes):raise ValueError("Matched behavior targets must belong to the chosen fixed family")
    archive.append("matched_feedback_probe_started",tag=tag,parent_actor_hash=parent_hash,
        guiding_actor_hash=guiding_hash,
        parent_state_sha256=parent_state_digest,behavior_sha256=behavior_digest,family=family,seeds=seeds,
        behavior_query_refs=[ep.query_ref for ep in episodes],promotion_claim=False)
    def finish(result):
        result.update(tag=tag,parent_actor_hash=parent_hash,parent_state_sha256=parent_state_digest,
            guiding_actor_hash=guiding_hash,
            behavior_sha256=behavior_digest,behavior_query_refs=[ep.query_ref for ep in episodes],
            behavior_target_owner="behavior_actor",behavior_controller_hashes=sorted({ep.behavior_controller_hash for ep in episodes}),
            checkpoint_root=cb.get("checkpoint_root"),
            family=family,seeds=seeds,promotion_claim=False,probe_wall_seconds=time.perf_counter()-started,
            causal_scope="Matched parent/MC/optimizer/RNG/work; only recorded teacher auxiliary data differs; finite fresh panel only")
        archive.append("matched_feedback_probe_result",**result)
        return result
    try:
        for trace in shared:
            validate_teacher(trace,executor.task.config)
            if not trace.successful or trace.actor_owner.get("actor_hash") is not None:
                raise ValueError("Common feedback corpus must be successful executed independent discovery")
        shared_hash=_digest([t.arrays() for t in shared])
        if not raw or len(raw)!=len(guided):
            raise ValueError("Raw/guided teacher quality needs equally sized paired complete panels")
        for left,right in zip(raw,guided):
            validate_teacher(left,executor.task.config);validate_teacher(right,executor.task.config)
            a,b=left.initial_snapshot,right.initial_snapshot
            if a.context_id!=b.context_id or not np.array_equal(a.state,b.state) or not np.array_equal(a.warmstart,b.warmstart):
                raise ValueError("Raw/guided teachers do not share exact reset boundaries")
            if left.actor_owner.get("actor_hash") is not None or right.actor_owner.get("actor_hash")!=guiding_hash:
                raise ValueError("Teacher panel is not independent versus the declared frozen guiding actor")
            for key in ("reset_seed","planner_seed"):
                if left.source.get(key)!=right.source.get(key):
                    raise ValueError("Raw/guided teacher planner/reset seed pairing differs")
        teacher_quality=dict(raw_captures=sum(t.successful for t in raw),guided_captures=sum(t.successful for t in guided),
            raw_mean_cost=float(np.mean([t.total_cost for t in raw])),
            guided_mean_cost=float(np.mean([t.total_cost for t in guided])))
        teacher_quality["passed"]=bool(teacher_quality["guided_captures"]>=teacher_quality["raw_captures"]
            and teacher_quality["guided_mean_cost"]<teacher_quality["raw_mean_cost"]-1e-6)
        teacher_hashes=dict(raw=_digest([t.arrays() for t in raw]),guided=_digest([t.arrays() for t in guided]))
        # Each child retains the exact common discovery corpus. New successful
        # teachers augment it; a raw panel with no new success still has its
        # original discovery data. Equal-size fixed row selection below makes
        # this a matched pooled-corpus comparison, never fabricated new labels.
        aux={name:cb["auxiliary"](shared+traces,family) for name,traces in (("raw_data",raw),("guided_data",guided))}
        if any(batch is None or not len(batch.actions) for batch in aux.values()):
            return finish(dict(status="missing_successful_current_family",teacher_quality=teacher_quality,
                teacher_sha256=teacher_hashes,shared_teacher_sha256=shared_hash,
                comparison_executed=False,guided_data_benefit_observed=False))
        if not teacher_quality["passed"]:
            return finish(dict(status="teacher_quality_not_met",teacher_quality=teacher_quality,
                teacher_sha256=teacher_hashes,shared_teacher_sha256=shared_hash,
                comparison_executed=False,guided_data_benefit_observed=False))
        common=min(len(batch.actions) for batch in aux.values());indices={};equalized={}
        for name,batch in aux.items():
            if batch.source!="executed_teacher" or np.any(np.asarray(batch.families)!=family):
                raise ValueError("Matched child auxiliary data must be actual successful teachers in the fixed family")
            indices[name]=np.linspace(0,len(batch.actions)-1,common,dtype=np.int64)
            chosen=indices[name]
            equalized[name]=AuxiliaryBatch(np.asarray(batch.observations)[chosen].copy(),
                np.asarray(batch.actions)[chosen].copy(),np.asarray(batch.families)[chosen].copy(),
                source="executed_teacher")
        aux_saved=archive.save_npz(f"feedback/{tag}_aligned_aux.npz",
            **{f"{name}_{key}":getattr(batch,key) for name,batch in equalized.items()
                for key in ("observations","actions","families")},
            **{f"{name}_source_row_indices":value for name,value in indices.items()})
        children={name:parent_learner.clone_for_trial() for name in equalized}
        if any(_digest(child.state_dict())!=parent_state_digest for child in children.values()):
            raise ValueError("Matched child must clone the full exact parent optimizer and private RNG state")
        cb["attach_meter"](parent_learner.actor,archive,"matched_feedback_parent_evaluation")
        base=cb["evaluate_actor"](executor,parent_learner.actor,family,seeds,tag+"_parent",deterministic=True)
        evaluation_refs=set(_validate_actor_panel(base,actor_hash=parent_hash,
            controller_hash=cb["controller_hash"](parent_learner.actor,deterministic=True,family=family),
            family=family,task_hash=executor.task.config.task_hash,count=len(seeds)))
        if _digest(parent_learner.state_dict())!=parent_state_digest:
            raise ValueError("Parent evaluation changed the frozen parent learner")
        rows={};learning={};checkpoints={};aux_digests={}
        for name,child in children.items():
            cb["attach_meter"](child.actor,archive,"matched_feedback_"+name+"_fit")
            start=time.perf_counter()
            learning[name]=child.update(episodes,parent_hash,aux=equalized[name])
            archive.append("fit_work",purpose="matched_feedback_"+name,tag=tag,
                scope=getattr(archive,"work_scope",None),
                fit_wall_seconds=time.perf_counter()-start,**learning[name])
            if _digest(episodes)!=behavior_digest:raise ValueError("Matched complete behavior data mutated between child updates")
            cb["attach_meter"](child.actor,archive,"matched_feedback_"+name+"_evaluation")
            rows[name]=cb["evaluate_actor"](executor,child.actor,family,seeds,tag+"_"+name,deterministic=True)
            refs=_validate_actor_panel(rows[name],actor_hash=cb["policy_hash"](child.actor),
                controller_hash=cb["controller_hash"](child.actor,deterministic=True,family=family),
                family=family,task_hash=executor.task.config.task_hash,count=len(seeds))
            if evaluation_refs.intersection(refs):raise ValueError("Matched child panels reused a physical evaluation query")
            evaluation_refs.update(refs)
            aux_digests[name]=_digest(equalized[name])
            checkpoint=cb["checkpoint"](archive.path,tag+"_"+name,
                dict(learner=child.state_dict(),parent_actor_hash=parent_hash,parent_state_sha256=parent_state_digest,
                    behavior_sha256=behavior_digest,behavior_query_refs=[ep.query_ref for ep in episodes],
                    teacher_sha256=teacher_hashes[name.removesuffix("_data")],auxiliary_sha256=aux_digests[name],
                    shared_teacher_sha256=shared_hash,guiding_actor_hash=guiding_hash,
                    auxiliary_raw=aux_saved,auxiliary_rows=common,family=family,evaluation=rows[name],
                    promotion_claim=False),archive)
            checkpoints[name]=dict(file=str(checkpoint),sha256=sha256(Path(checkpoint).read_bytes()).hexdigest())
        if learning["raw_data"]["gradient_steps"]!=learning["guided_data"]["gradient_steps"]:
            raise ValueError("Matched children used different optimizer work")
        if _digest(parent_learner.state_dict())!=parent_state_digest:
            raise ValueError("Matched probe changed the frozen parent learner")
        raw_comparison=_comparison(base,rows["raw_data"])
        guided_comparison=_comparison(base,rows["guided_data"])
        data_comparison=_comparison(rows["raw_data"],rows["guided_data"])
        return finish(dict(status="complete_unpromoted_comparison",comparison_executed=True,
            teacher_quality=teacher_quality,teacher_sha256=teacher_hashes,shared_teacher_sha256=shared_hash,
            shared_teacher_episodes=len(shared),auxiliary_rows_per_child=common,
            auxiliary_row_rule="evenly spaced fixed indices over common discovery plus each arm's successful new pre-action teachers; equal total rows; no fit-based row selection",
            auxiliary_raw=aux_saved,auxiliary_sha256=aux_digests,learning=learning,checkpoints=checkpoints,
            parent=cb["summary"](base),children={name:dict(actor_hash=cb["policy_hash"](children[name].actor),
                summary=cb["summary"](outcomes),outcomes=outcomes) for name,outcomes in rows.items()},
            parent_outcomes=base,raw_vs_parent=raw_comparison,guided_vs_parent=guided_comparison,
            guided_vs_raw=data_comparison,
            guided_data_benefit_observed=bool(data_comparison["improved"] and guided_comparison["improved"])))
    except BaseException as failure:
        archive.append("matched_feedback_probe_error",tag=tag,type=type(failure).__name__,
            message=str(failure),probe_wall_seconds=time.perf_counter()-started,promotion_claim=False)
        raise

