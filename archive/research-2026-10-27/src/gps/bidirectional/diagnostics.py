"""Offline physical metrics from actual recorded Acrobot trajectories.

Only NumPy, raw files and optional plotting are used. No simulator, producer
task, policy or learning module is imported. Planner query budgets are reported
separately from executed outcomes; hypothetical endpoint scores cannot establish
capture, actor competence, or a control improvement.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def _write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())


def _read_json(path,inputs):
    path=Path(path).resolve();value=json.loads(path.read_text())
    inputs[str(path)]=_hash(path)
    return value


def _rows(path):
    result=[];partial=False
    lines=Path(path).read_bytes().splitlines(keepends=True)
    for index,line in enumerate(lines):
        try:result.append(json.loads(line))
        except (ValueError,UnicodeError):
            if index!=len(lines)-1 or line.endswith(b"\n"):raise ValueError("Corrupt middle ledger line")
            partial=True
    return result,partial


def _task_for_episode(path,raw,provided,inputs):
    if provided is not None:
        return dict(provided),"explicit argument"
    # The validation sentinel uses T=12 even when its enclosing run config is
    # T=800. Its first actual query's saved task_json is the authoritative task.
    references=[str(x) for x in raw.get("query_ids",[])]
    for parent in path.parents:
        ledger=parent/"ledger.jsonl"
        if ledger.is_file() and references:
            rows,partial=_rows(ledger);inputs[str(ledger.resolve())]=_hash(ledger)
            request=next((r for r in rows if r.get("kind")=="query_requested"
                          and r.get("query_id")==references[0]),None)
            if request is not None:
                target=(parent/request["inputs"]["file"]).resolve()
                if not target.is_relative_to(parent.resolve()):raise ValueError("Unsafe query task reference")
                digest=_hash(target);inputs[str(target)]=digest
                if digest!=request["inputs"]["sha256"]:raise ValueError("First query input digest mismatch")
                with np.load(target,allow_pickle=False) as values:
                    return json.loads(str(values["task_json"])),str(target)
        config=parent/"config.json"
        if config.is_file():
            value=_read_json(config,inputs);task=value.get("task",value)
            if task.get("T")==int(raw["T"]):return dict(task),str(config)
    raise ValueError("Actual task definition unavailable; pass task_config explicitly")


def analyze_episode(npz,*,task_config=None):
    """Measure one saved actual episode, with no integration or action inference."""
    started=time.perf_counter();path=Path(npz).resolve();before=path.stat()
    digest=_hash(path);inputs={str(path):digest}
    with np.load(path,allow_pickle=False) as loaded:
        names=("states","actions","elapsed","hold_counts","stage_costs","terminal_costs",
            "capture_failures","component_costs","tip_distances","initial_snapshot_json",
            "actor_owner_json","source_json","execution_semantics","family","complete","captured","T","query_ids")
        if not set(names)<=set(loaded.files):raise ValueError("Expected an actual EpisodeTrace archive")
        raw={name:loaded[name].copy() for name in names}
    task,task_source=_task_for_episode(path,raw,task_config,inputs)
    required=("T","dt","capture_steps","capture_distance","capture_qvel_norm","local_distance","local_by")
    if not all(name in task for name in required):raise ValueError("Incomplete actual task configuration")
    T=int(task["T"]);dt=float(task["dt"]);capture_steps=int(task["capture_steps"])
    if T!=raw["T"] or T<1 or dt<=0 or not 1<=capture_steps<=T:raise ValueError("Invalid task horizon/window")
    states=np.asarray(raw["states"],dtype=float);actions=np.asarray(raw["actions"],dtype=float)
    n=len(actions)
    if n<1 or states.shape!=(n+1,5) or actions.shape!=(n,1):raise ValueError("Not an aligned actual trajectory")
    elapsed=np.asarray(raw["elapsed"])
    if (elapsed.shape!=(n+1,) or not np.array_equal(elapsed,np.round(elapsed))
            or elapsed[0]<0 or elapsed[-1]>T or not np.array_equal(np.diff(elapsed),np.ones(n))):
        raise ValueError("Actual clock must advance by one control")
    if not np.allclose(states[:,0],elapsed*dt,rtol=0,atol=1e-9):raise ValueError("Physics/explicit clock mismatch")
    for name,shape in (("hold_counts",(n+1,)),("stage_costs",(n,)),("terminal_costs",(n,)),
                       ("capture_failures",(n,)),("component_costs",(n,4)),("tip_distances",(n+1,))):
        if raw[name].shape!=shape or not np.isfinite(raw[name]).all():raise ValueError("Invalid aligned "+name)
    if not np.isfinite(states).all() or not np.isfinite(actions).all() or np.any(abs(actions)>1):
        raise ValueError("Invalid actual states/actions")
    initial=json.loads(str(raw["initial_snapshot_json"]));owner=json.loads(str(raw["actor_owner_json"]))
    source=json.loads(str(raw["source_json"]));semantics=str(raw["execution_semantics"])
    if semantics not in ("actor","executed_replanning","analytic_reference"):
        raise ValueError("Hypothetical/model-only trajectories are not executed evidence")
    if not np.array_equal(initial["state"],states[0]) or initial["elapsed"]!=elapsed[0]:
        raise ValueError("Initial snapshot does not match actual trajectory")
    if (type(initial["hold_count"]) is not int or not 0<=initial["hold_count"]<=min(capture_steps,elapsed[0])):
        raise ValueError("Invalid carried capture history")
    if raw["query_ids"].shape!=(n,) or len(set(str(q) for q in raw["query_ids"]))!=n:
        raise ValueError("Actual controls need distinct query references")
    q=states[:,1:3];tip_x=np.sin(q[:,0])+np.sin(q[:,0]+q[:,1])
    tip_z=2+np.cos(q[:,0])+np.cos(q[:,0]+q[:,1]);distance=np.sqrt(tip_x**2+(tip_z-4)**2)
    speed=np.linalg.norm(states[:,3:5],axis=1)
    inside=(distance<=task["capture_distance"])&(speed<=task["capture_qvel_norm"])
    hold=int(initial["hold_count"]);holds=[hold];observed=longest=0
    for value in inside[1:]:
        hold=min(hold+1,capture_steps) if value else 0;holds.append(hold)
        observed=observed+1 if value else 0;longest=max(longest,observed)
    complete=int(elapsed[-1])==T;root_complete=int(elapsed[0])==0 and complete
    capture=bool(complete and hold>=capture_steps)
    by6=states[:,0]<=6.+1e-12
    local=(elapsed<=task["local_by"])&(distance<=task["local_distance"])
    hits=np.flatnonzero(distance<=task["local_distance"])
    first=float(states[hits[0],0]) if len(hits) else None
    final=(elapsed>max(0,T-int(round(1./dt))))&(elapsed<=T)
    final_distance=float(distance[final].max()) if final.any() else None
    final_speed=float(speed[final].max()) if final.any() else None
    if len(hits):
        shoulder=np.unwrap(q[:,0]);delta=float(shoulder[hits[0]]-shoulder[0])
        family="positive" if delta>1e-9 else ("negative" if delta < -1e-9 else "stationary")
    else:family="unreached"
    consistency=[]
    if not np.array_equal(np.asarray(holds),raw["hold_counts"]):consistency.append("recorded capture history differs from post-state criterion")
    if bool(raw["captured"])!=capture:consistency.append("recorded capture flag differs from full ending criterion")
    if bool(raw["complete"])!=root_complete:consistency.append("recorded root completion differs from actual clock")
    if not np.allclose(raw["tip_distances"],distance,rtol=0,atol=1e-9):consistency.append("recorded tip distances differ from post-state FK")
    if not np.allclose(raw["component_costs"].sum(axis=1),raw["stage_costs"],rtol=1e-10,atol=1e-10):
        consistency.append("recorded stage components do not sum to stage cost")
    if np.count_nonzero(raw["terminal_costs"])>1 or np.count_nonzero(raw["capture_failures"])>1:
        consistency.append("multiple true ending/failure charges")
    if (np.any(raw["terminal_costs"][:-1]) or np.any(raw["capture_failures"][:-1])
            or (not complete and (np.any(raw["terminal_costs"]) or np.any(raw["capture_failures"])))):
        consistency.append("true ending cost charged before episode termination")
    stat=path.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(before.st_size,before.st_mtime_ns) or _hash(path)!=digest:
        raise RuntimeError("Source episode changed during analysis")
    components={key:float(value) for key,value in zip(("target","qvel","qvel_excess","control"),raw["component_costs"].sum(axis=0))}
    stage=float(raw["stage_costs"].sum());terminal=float(raw["terminal_costs"].sum());failure=float(raw["capture_failures"].sum())
    full_task=(T==800 and np.isclose(dt,.01) and capture_steps==100
        and np.isclose(task["capture_distance"],.2) and np.isclose(task["capture_qvel_norm"],.5)
        and np.isclose(task["local_distance"],.5) and task["local_by"]==600
        and task.get("failure_penalty")==1000.)
    result=dict(file=str(path),sha256=digest,input_sha256=inputs,task_config=task,task_config_source=task_source,
        task_hash=initial["task_hash"],context_id=initial["context_id"],semantics=semantics,
        actor_owner=owner,source=source,initial_elapsed=int(elapsed[0]),final_elapsed=int(elapsed[-1]),
        T=T,duration_seconds=float((elapsed[-1]-elapsed[0])*dt),recorded_controls=n,
        root_complete=root_complete,complete=complete,capture=capture,recorded_capture=bool(raw["captured"]),
        full_eight_second_task=bool(full_task),scope="full eight-second Acrobot task" if full_task else
            f"short/operational task T={T}; no full Acrobot competence claim",
        initial_tip_distance=float(distance[0]),nontrivial_swingup_start=bool(distance[0]>task["local_distance"]),
        min_tip_distance_by6s=float(distance[by6].min()) if by6.any() else None,
        by6s_window_fully_observed=bool(states[0,0]<=0 and states[-1,0]>=6),
        first_approach_time=first,local_success=bool(local.any()),
        final_second_max_tip_distance=final_distance,final_second_max_speed=final_speed,
        final_second_fully_observed=bool(complete and T*dt>=1. and states[0,0]<=(T*dt-1.)),
        final_continuous_hold_controls=int(hold),longest_observed_continuous_hold_controls=int(longest),
        longest_observed_continuous_hold_seconds=float(longest*dt),family=family,recorded_family=str(raw["family"]),
        stage_cost=stage,true_end_cost=terminal,failure_cost=failure,total_cost=stage+terminal+failure,
        stage_components=components,query_ids=[str(qid) for qid in raw["query_ids"]],
        consistency_errors=consistency,failure_class=("incomplete_record" if not complete else
            "captured" if capture else "no_useful_approach" if not local.any() else "approach_without_capture"),
        analytic_boundary_rows=n+1,analytic_cost_sum_rows=n,new_simulator_controls=0,
        new_actor_inference_rows=0,new_fits=0,wall_seconds=time.perf_counter()-started,
        curve=dict(elapsed=elapsed.astype(int).tolist(),time=states[:,0].tolist(),
            tip_distance=distance.tolist(),qvel_norm=speed.tolist(),hold_counts=holds))
    return result


def _query_horizon(row):
    metadata=row.get("metadata",{});owner=row.get("owner",{})
    config=owner.get("planner_config",{}) if isinstance(owner,dict) else {}
    value=config.get("H",metadata.get("horizon"))
    return int(value) if type(value) is int and value>0 else None


def _plot_trajectories(episodes,out):
    if not episodes:return []
    temporary_root="/private/tmp" if Path("/private/tmp").is_dir() else "/tmp"
    with tempfile.TemporaryDirectory(prefix="mppi-gps-diag-font-",dir=temporary_root) as cache:
        previous=os.environ.get("MPLCONFIGDIR");os.environ["MPLCONFIGDIR"]=cache
        try:
            import matplotlib
            matplotlib.use("Agg",force=True)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            outputs=[]
            for name,by_capture in (("actual_trajectories_by_horizon.png",False),
                                    ("actual_capture_qualified_vs_not.png",True)):
                fig=Figure(figsize=(11,4));FigureCanvasAgg(fig);axes=fig.subplots(1,2)
                labels=set()
                palette=("#2474b5","#e98128","#7d55a4","#298c8c","#ab5d77","#756d37")
                keys=sorted({(str(e.get("planner_horizon")),e["semantics"]) for e in episodes})
                colors={key:palette[i%len(palette)] for i,key in enumerate(keys)}
                for episode in episodes:
                    label=("capture criterion met" if episode["capture"] else "capture criterion not met") if by_capture else (
                        f"H={episode.get('planner_horizon','unknown')} · {episode['semantics']}")
                    color=("#218c56" if episode["capture"] else "#b44b45") if by_capture else colors[
                        (str(episode.get("planner_horizon")),episode["semantics"])]
                    shown=label if label not in labels else None;labels.add(label)
                    curve=episode["curve"]
                    for ax,key in zip(axes,("tip_distance","qvel_norm")):
                        ax.plot(curve["time"],curve[key],color=color,alpha=.65,
                                linewidth=1,label=shown,linestyle="-" if episode["capture"] else "--")
                for ax,key in zip(axes,("tip distance (m)","joint-speed norm (rad/s)")):
                    ax.set(xlabel="Actual episode time (s)",ylabel=key);ax.grid(alpha=.2)
                axes[0].axhline(.2,color=".4",linewidth=.7);axes[1].axhline(.5,color=".4",linewidth=.7)
                axes[0].legend(fontsize=7)
                smoke=any(not e["full_eight_second_task"] for e in episodes)
                fig.suptitle("Recorded post-state geometry; no controller/model rerun"+
                    (" · short operational task, no full-task claim" if smoke else ""),fontsize=10)
                fig.tight_layout();target=out/name;fig.savefig(target);fig.clear();outputs.append(str(target.name))
            return outputs
        finally:
            if previous is None:os.environ.pop("MPLCONFIGDIR",None)
            else:os.environ["MPLCONFIGDIR"]=previous


def diagnose_run(run,newout):
    """Create an additive report; never change run files or select new actions."""
    started=time.perf_counter();run=Path(run).resolve();out=Path(newout).resolve()
    out.mkdir(parents=True,exist_ok=False)
    _write(out/"started.json",dict(schema="bidirectional-offline-diagnosis-v1",run=str(run),
        source_sha256=_hash(__file__),utc=datetime.now(timezone.utc).isoformat(),
        new_simulator_controls=0,new_actor_inference_rows=0,new_fits=0))
    inputs={};errors=[];pending=[];episodes=[];all_rows=[];query_groups={};source_checks=[]
    try:
        config=_read_json(run/"config.json",inputs)
        manifest_path=run/"manifest.json"
        if manifest_path.is_file():
            manifest=_read_json(manifest_path,inputs)
            for item in manifest.get("files",[]):
                path=(run/item["file"]).resolve()
                if not path.is_relative_to(run):raise ValueError("Source manifest escapes run")
                actual=_hash(path);inputs[str(path)]=actual
                source_checks.append(dict(file=item["file"],matches=actual==item["sha256"]))
            if any(not item["matches"] for item in source_checks):errors.append("Archived source manifest mismatch")
        else:pending.append("No source manifest available")
        for ledger in sorted(run.rglob("ledger.jsonl")):
            if ledger.is_relative_to(out):continue
            inputs[str(ledger)]=_hash(ledger);rows,partial=_rows(ledger);all_rows.extend(rows)
            if partial:pending.append("Incomplete final ledger line: "+str(ledger))
            requests={r["query_id"]:r for r in rows if r.get("kind")=="query_requested"}
            done={r["query_id"]:r for r in rows if r.get("kind")=="query_complete"}
            failed={r["query_id"]:r for r in rows if r.get("kind")=="query_error"}
            decisions={}
            for row in rows:
                if row.get("kind")=="decision":decisions.setdefault(row["episode_id"],[]).append(row)
            for qid,row in requests.items():
                horizon=_query_horizon(row)
                owner=row.get("owner",{});planner=owner.get("planner_config",{}) if isinstance(owner,dict) else {}
                configured_samples=planner.get("K")
                key=(row.get("mode"),horizon,configured_samples,row.get("query_kind"))
                group=query_groups.setdefault(key,dict(mode=key[0],configured_horizon=horizon,
                    configured_samples=configured_samples,query_kind=key[3],queries=0,requested_controls=0,requested_physics_controls=0,
                    completed_controls=0,unknown_controls=0,failed_queries=0,query_wall_seconds=0.))
                group["queries"]+=1;group["requested_controls"]+=row["requested_controls"]
                group["requested_physics_controls"]+=row["requested_physics_controls"]
                resolution=done.get(qid,failed.get(qid))
                if resolution:
                    group["completed_controls"]+=resolution.get("completed_controls",0)
                    group["query_wall_seconds"]+=resolution.get("query_wall_seconds",0.)
                if qid in failed:group["failed_queries"]+=1
                if qid not in done and not failed.get(qid,{}).get("call_returned",False):
                    group["unknown_controls"]+=row["requested_physics_controls"]
                if qid not in done and qid not in failed:pending.append("Unresolved query "+str(ledger)+":"+qid)
            seen=set()
            outcomes={r["episode_id"]:r for r in rows if r.get("kind")=="episode_outcome"}
            for row in (r for r in rows if r.get("kind")=="episode_saved"):
                path=(ledger.parent/row["raw"]["file"]).resolve()
                if not path.is_relative_to(ledger.parent.resolve()):raise ValueError("Episode path escapes archive")
                if path in seen:errors.append("Repeated physical episode reference: "+str(path));continue
                seen.add(path)
                try:
                    result=analyze_episode(path)
                    if result["sha256"]!=row["raw"]["sha256"]:raise ValueError("Episode digest differs from ledger")
                    inputs.update(result["input_sha256"])
                    hs={_query_horizon(requests[q]) for d in decisions.get(row["episode_id"],[])
                        for q in d["bank_query_ids"] if q in requests}
                    hs.discard(None)
                    result.update(episode_id=row["episode_id"],archive=str(ledger.parent),
                        planner_horizon=next(iter(hs)) if len(hs)==1 else None,
                        horizon_identity="query-owner planner configuration" if len(hs)==1 else "unknown/mixed",
                        outcome_scope=outcomes.get(row["episode_id"],{}).get("scope"),
                        declared_outcome=outcomes.get(row["episode_id"]))
                    errors.extend(str(path)+": "+error for error in result["consistency_errors"])
                    episodes.append(result)
                except Exception as error:errors.append(str(path)+": "+str(error))
        stage_files={}
        for name in ("qualification.json","acquisition.json","development.json","validate.json","stage_complete.json","stage_failed.json"):
            if (run/name).is_file():stage_files[name]=_read_json(run/name,inputs)
        terminal="stage_complete.json" in stage_files or "stage_failed.json" in stage_files
        if not terminal:pending.append("No terminal stage record; observed subsets cannot qualify the controller")
        full=[e for e in episodes if e["full_eight_second_task"] and e["root_complete"]]
        independent=[e for e in full if e["semantics"]=="executed_replanning"
            and isinstance(e["actor_owner"],dict) and e["actor_owner"].get("actor_hash") is None
            and e["nontrivial_swingup_start"]]
        groups=[]
        for horizon in sorted({e["planner_horizon"] for e in episodes},key=lambda h:-1 if h is None else h):
            selected=[e for e in episodes if e["planner_horizon"]==horizon]
            groups.append(dict(horizon=horizon,episodes=len(selected),
                complete_episodes=sum(e["complete"] for e in selected),captures=sum(e["capture"] for e in selected),
                local_successes=sum(e["local_success"] for e in selected),
                mean_executed_total_cost=float(np.mean([e["total_cost"] for e in selected])),
                full_task_episodes=sum(e["full_eight_second_task"] for e in selected),
                episode_ids=[e["episode_id"] for e in selected]))
        q=config.get("qualification",{});minimum=q.get("confirmation_roots",32)
        competent=[]
        for horizon in {e["planner_horizon"] for e in independent if e["planner_horizon"] is not None}:
            selected=[e for e in independent if e["planner_horizon"]==horizon]
            if (len({e["context_id"] for e in selected})>=minimum and
                    np.mean([e["local_success"] for e in selected])>=q.get("local_min",.9)):
                competent.append(horizon)
        captures=sum(e["capture"] for e in independent)
        actor_updates=[r for r in all_rows if r.get("kind")=="actor_update"]
        actual_actor=[e for e in full if e["semantics"]=="actor"]
        reported_qualification=stage_files.get("qualification.json",{}).get("qualified")
        valid=not errors and not pending
        links=dict(
            discovery=dict(status="OBSERVED_SUCCESSFUL_INDEPENDENT_EXPERIENCE" if valid and captures else "UNSUPPORTED",
                nontrivial_full_task_independent_successes=captures,
                protocol_acquisition_gate=stage_files.get("acquisition.json",{}).get("qualified")),
            local_competence=dict(status="OBSERVED_ON_RECORDED_PANELS" if valid and competent else "UNSUPPORTED",
                horizons=competent,minimum_unique_root_contexts=minimum),
            ending=dict(status="REPORTED_QUALIFICATION_REQUIRES_CAUSAL_AUDIT" if valid and reported_qualification else "UNSUPPORTED",
                reported_task_qualification=reported_qualification,
                limitation="Actual trajectories describe outcomes; horizon changes also move terminal heuristics, and are not a proof about every short-horizon controller."),
            actor_learning=dict(status="NOT_TESTED" if not actor_updates else "RAW_LEARNING_WORK_PRESENT_CAUSAL_RESULT_UNESTABLISHED",
                actor_update_events=len(actor_updates),full_task_actor_episodes=len(actual_actor),
                limitation="This diagnostic does not verify likelihoods, gradients, target ownership or generalization; fitting work alone cannot establish learned long-term information."),
            improving_search=dict(status="UNSUPPORTED",limitation="Requires paired actual independent/guided complete outcomes, valid promotion and changes; hypothetical plan scores are insufficient."),
            efficiency=dict(status="NOT_ESTABLISHED",limitation="Requires strong controls after all discovery, learning, qualification and evaluation costs; this diagnostic assigns no efficiency claim."))
        plots=_plot_trajectories(episodes,out)
        with (out/"trajectory_curves.csv").open("x",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=("episode_id","horizon","semantics","elapsed","time","tip_distance","qvel_norm","hold_count","capture"));writer.writeheader()
            for e in episodes:
                c=e["curve"]
                for i in range(len(c["time"])):writer.writerow(dict(episode_id=e["episode_id"],horizon=e["planner_horizon"],
                    semantics=e["semantics"],elapsed=c["elapsed"][i],time=c["time"][i],tip_distance=c["tip_distance"][i],
                    qvel_norm=c["qvel_norm"][i],hold_count=c["hold_counts"][i],capture=e["capture"]))
        compact=[{key:value for key,value in e.items() if key!="curve"} for e in episodes]
        result=dict(schema="bidirectional-offline-diagnosis-v1",run=str(run),input_sha256=inputs,
            diagnostic_source_sha256=_hash(__file__),
            source_checks=source_checks,errors=errors,pending=pending,terminal_stage_record=terminal,
            episodes=compact,per_horizon_outcomes=groups,per_horizon_query_cost=list(query_groups.values()),
            causal_links=links,plots=plots,scope="Executed trajectory description and budget crosswalk; no new scientific qualification",
            short_operational_task_present=any(not e["full_eight_second_task"] for e in episodes),
            analytic_boundary_rows=sum(e["analytic_boundary_rows"] for e in episodes),
            analytic_cost_sum_rows=sum(e["analytic_cost_sum_rows"] for e in episodes),
            new_simulator_controls=0,new_actor_inference_rows=0,new_fits=0,
            hypothetical_plan_outcomes_used=False,
            model_query_scope="Query accounting from hash-bound ledgers only; model-bank endpoints are hypothetical and are not executed successes",
            query_wall_scope="Sum of returned query wall intervals; may overlap and is not calendar wall time",
            wall_seconds=time.perf_counter()-started)
        _write(out/"diagnosis.json",result)
        _write(out/"output_manifest.json",dict(files=[dict(file=p.name,sha256=_hash(p),bytes=p.stat().st_size)
            for p in sorted(out.iterdir()) if p.is_file()]))
        return result
    except BaseException as error:
        _write(out/"failure.json",dict(error_type=type(error).__name__,error=str(error),
            wall_seconds=time.perf_counter()-started,input_sha256=inputs,
            new_simulator_controls=0,new_actor_inference_rows=0,new_fits=0))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path);parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args();result=diagnose_run(args.run,args.out)
    print(json.dumps({k:v for k,v in result.items() if k not in ("episodes","input_sha256")},indent=2))


if __name__=="__main__":main()
