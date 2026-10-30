"""Independent arithmetic audit of bidirectional Acrobot raw evidence.

Only NumPy and the standard library are used.  No environment, planner, policy,
learning code or producer scorer is imported; no dynamics or actions are
re-executed.  PASS means consistent, closed arithmetic evidence, not scientific
success or an independently validated dynamics model.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np


TASK_FIELDS = frozenset(("T", "dt", "capture_steps", "capture_distance", "capture_qvel_norm",
    "local_distance", "local_by", "failure_penalty", "reset_angle_radius", "reset_velocity_radius"))
BASE_FILES = ("assets/acrobot.xml", "src/envs/acrobot.py", "src/envs/mujoco_env.py")
STEP_FIELDS={"optimizer":"updates","gradient_backward":"passes","diagnostic_backward":"passes"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def finite(value, shape=None):
    a = np.asarray(value)
    require(a.dtype.kind in "biuf" and np.isfinite(a).all()
            and (shape is None or a.shape == shape), "Invalid numeric array shape or nonfinite value")
    return a


def same(actual, expected, message, *, atol=2e-8, rtol=2e-8):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype.kind in "biuf" and np.isfinite(a).all()
            and np.allclose(a,b,atol=atol,rtol=rtol), message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name+" must be a nonnegative integer")
    return value


def verify_block_proposals(actions,config):
    """Check the declared independent block channel from raw queried actions."""
    lengths=config.get('independent_block_lengths',()) if isinstance(config,dict) else ()
    if not lengths:return ()
    require(isinstance(lengths,(list,tuple)) and all(type(n) is int
        and n>=config['knot_spacing'] for n in lengths),'Invalid independent block specification')
    K=integer(config['K'],'Gaussian count',1);smooth=integer(config['smooth_candidates'],'smooth count')
    require(actions.shape[0]==K+smooth+config['memory_candidates'],'Block bank count differs')
    labels=[]
    for i in range(smooth):
        length=lengths[i%len(lengths)];row=actions[K+i]
        starts=(np.arange(len(row))//length)*length
        same(row,row[starts],'Independent block proposal changes inside a block',atol=0,rtol=0)
        labels.append('independent_blocks:'+str(length))
    return tuple(labels)


def packed(raw, key):
    value = np.asarray(raw[key])
    require(value.shape == () and value.dtype.kind in "US", "Invalid packed JSON: "+key)
    return json.loads(str(value))


def scalar(raw, key):
    value = np.asarray(raw[key])
    require(value.shape == (), "Expected scalar: "+key)
    return value.item()


def validate_config(config):
    require(isinstance(config,dict) and set(config) == TASK_FIELDS, "Task config keys differ")
    for key in ("T","capture_steps","local_by"):
        integer(config[key],key,1)
    require(config["capture_steps"] <= config["T"] and config["local_by"] <= config["T"], "Task windows exceed T")
    for key,value in config.items():
        require(type(value) in (int,float) and math.isfinite(value) and value >= 0, "Invalid task field: "+key)
    require(config["dt"] == .01 and config["failure_penalty"] == 1000., "Frozen clock/failure cost differs")
    require(min(config[k] for k in ("capture_distance","capture_qvel_norm","local_distance")) > 0,
            "Capture thresholds must be positive")
    return config


def task_identity(config, source_root):
    source_root=Path(source_root)
    wrapper="src/gps/bidirectional/task.py"
    # Parse the archived schema declaration; never import/execute its producer.
    tree=ast.parse((source_root/wrapper).read_bytes())
    declarations=[]
    for cls in tree.body:
        if not isinstance(cls,ast.ClassDef) or cls.name!="TaskConfig":continue
        for method in cls.body:
            if not isinstance(method,ast.FunctionDef) or method.name!="task_hash":continue
            for assignment in method.body:
                if isinstance(assignment,ast.Assign) and isinstance(assignment.value,ast.Dict):
                    for key,value in zip(assignment.value.keys,assignment.value.values):
                        if isinstance(key,ast.Constant) and key.value=="schema":
                            declarations.append(ast.literal_eval(value))
    require(len(declarations)==1 and declarations[0] in ("acrobot-capture-v1","acrobot-capture-v2-post-action-fk"),
        "Archived task schema declaration is unsupported or ambiguous")
    post_action=declarations[0]=="acrobot-capture-v2-post-action-fk"
    sources = {name:digest((source_root/name).read_bytes())
               for name in BASE_FILES+((wrapper,) if post_action else ())}
    payload = dict(schema="acrobot-capture-v2-post-action-fk" if post_action else "acrobot-capture-v1", config=config,
        base_cost="unchanged-src.envs.acrobot.Acrobot", base_source_sha256=sources)
    if post_action:payload["cost_boundary_source"]="post-action-full-state-analytic-fk"
    return digest(canonical(payload).encode()), sources


def kinematics(states):
    s = finite(states)
    require(s.shape[-1] == 5, "Acrobot full states must have five coordinates")
    q0, q1 = s[...,1], s[...,2]
    x = np.sin(q0)+np.sin(q0+q1)
    z = 2.+np.cos(q0)+np.cos(q0+q1)
    distance = np.hypot(x,z-4.)
    target = 1.-np.clip(z/4.,0.,1.)+.05*x*x
    reward = np.exp(math.log(.1)*(np.maximum(distance-.2,0.)/2.)**2)
    velocity = s[...,3:5]
    squared = (velocity[...,0]/2.5)**2+(velocity[...,1]/5.)**2
    gated = reward**2*squared
    excess = np.sum((np.maximum(np.abs(velocity)-8.,0.)/8.)**2,axis=-1)
    return dict(distance=distance, speed=np.linalg.norm(velocity,axis=-1), target=target,
        components=np.stack((target,.15*gated,.05*excess,np.zeros_like(target)),axis=-1),
        terminal=100.*target+25.*gated)


def observations(states, elapsed, holds, config):
    s = finite(states)
    shape = s.shape[:-1]
    elapsed, holds = np.broadcast_to(elapsed,shape), np.broadcast_to(holds,shape)
    return np.stack((np.sin(s[...,1]),np.cos(s[...,1]),np.sin(s[...,2]),np.cos(s[...,2]),
        s[...,3],s[...,4],(config["T"]-elapsed)/config["T"],
        np.minimum(holds,config["capture_steps"])/config["capture_steps"]),axis=-1)


def hold_sequence(states, initial_hold, config, distances=None):
    metrics=kinematics(states)
    if distances is not None:
        metrics["distance"]=finite(distances,states.shape[:2])
    count=np.full(states.shape[0],initial_hold,dtype=np.int64)
    pre, post=[],[]
    for t in range(states.shape[1]):
        pre.append(count.copy())
        inside=(metrics["distance"][:,t] <= config["capture_distance"]) & (
            metrics["speed"][:,t] <= config["capture_qvel_norm"])
        count=np.where(inside,np.minimum(count+1,config["capture_steps"]),0)
        post.append(count.copy())
    return np.stack(pre,axis=1),np.stack(post,axis=1),metrics


def validate_snapshot(snapshot, config, expected_hash):
    require(isinstance(snapshot,dict), "Initial snapshot missing")
    integer(snapshot["elapsed"],"elapsed");integer(snapshot["hold_count"],"hold_count")
    require(snapshot["elapsed"] <= config["T"] and snapshot["hold_count"] <= min(
        snapshot["elapsed"],config["capture_steps"]), "Invalid snapshot clock/hold count")
    require(snapshot["task_hash"] == expected_hash and bool(snapshot["context_id"]), "Snapshot task/context mismatch")
    require(type(snapshot["warmstart_exact"]) is bool, "Missing warmstart provenance")
    state=finite(snapshot["state"],(5,));finite(snapshot["warmstart"],(2,))
    same(state[0],snapshot["elapsed"]*.01,"Snapshot physics clock differs",atol=1e-9,rtol=0)


def audit_query_arrays(inputs, raw, request, config, expected_hash, temperature=None, *, post_action_boundary=False):
    """Reconstruct one completed model bank or returned live control."""
    snapshot=packed(inputs,"initial_snapshot_json")
    validate_snapshot(snapshot,config,expected_hash)
    require(canonical(packed(inputs,"task_json")) == canonical(config), "Query task changed")
    require(canonical(packed(raw,"initial_snapshot_json")) == canonical(snapshot), "Result snapshot differs from request")
    for key in ("owner_json","metadata_json"):
        require(canonical(packed(inputs,key)) == canonical(packed(raw,key)), "Result input metadata changed")
    require(canonical(packed(inputs,"owner_json")) == canonical(request["owner"])
            and canonical(packed(inputs,"metadata_json")) == canonical(request["metadata"]), "Ledger/input owner binding differs")
    require(snapshot["context_id"] == request["context_id"] and snapshot["elapsed"] == request["initial_elapsed"]
            and request["task_hash"] == expected_hash == str(raw["task_hash"]), "Ledger snapshot identity differs")
    for key,snapshot_key in (("initial_state","state"),("initial_warmstart","warmstart")):
        same(inputs[key],snapshot[snapshot_key],"Input snapshot array differs",atol=0,rtol=0)
        same(raw[key],inputs[key],"Result initial array differs",atol=0,rtol=0)
    require(tuple(str(x) for x in raw["component_names"]) == ("target","qvel","qvel_excess","control"),
            "Cost component layout differs")
    mode=request["mode"];require(mode in ("model","live"),"Unknown query mode")
    backend=packed(raw,"backend_json")
    warp=mode=="model" and backend.get("backend")=="warp"
    require((mode=="model" and backend.get("backend") in ("cpu","warp"))
            or (mode=="live" and backend.get("backend")=="mujoco_cpu" and backend.get("frame_skip")==1),
            "Raw backend identity differs from model/live semantics")
    geometry_atol=2e-5 if warp else 2e-8
    cost_atol,cost_rtol=(2e-4,2e-5) if warp else (2e-8,2e-8)
    actions=finite(inputs["actions"])
    require(np.all(np.abs(actions)<=1.), "Input action exceeds native bounds")
    if mode == "model":
        require(actions.ndim == 3 and actions.shape[0]>0 and actions.shape[1]>0 and actions.shape[2]==1,"Invalid model action bank")
        K,H=actions.shape[:2]
    else:
        require(actions.shape == (1,),"Live query must request exactly one action")
        K,H=1,1
    remaining=config["T"]-snapshot["elapsed"]
    require(remaining>0,"Physical query requested after episode termination")
    h=min(H,remaining)
    if mode=="model":
        require(backend.get("worlds")==K and backend.get("horizon")==h
                and backend.get("time_column_reconstructed") is warp,"Model backend dimensions/clock provenance differs")
    require(request["requested_controls"] == K*H and request["requested_physics_controls"] == K*h
            and request["requested_physics_frames"] == K*h and request["cutoff_skipped_controls"] == K*(H-h),
            "Requested/cutoff/frame accounting differs from actual finite horizon")
    s=finite(raw["states"],(K,h,5) if mode=="model" else (1,5))
    p=finite(raw["prestates"],s.shape)
    used=finite(raw["actions"],(K,h,1) if mode=="model" else (1,1))
    expected_actions=actions[:,:h] if mode=="model" else actions[None]
    same(used,expected_actions,"Returned actions differ from requested executable prefix",atol=0,rtol=0)
    sb=s if mode=="model" else s[None]
    pb=p if mode=="model" else p[None]
    expected_pre=np.concatenate((np.broadcast_to(snapshot["state"],(K,1,5)),sb[:,:-1]),axis=1)
    same(pb,expected_pre,"Pre/post state alignment differs",atol=0,rtol=0)
    clocks=snapshot["state"][0]+.01*np.arange(1,h+1)
    same(sb[...,0],np.broadcast_to(clocks,(K,h)),"Physical clock skips/repeats or pads controls",atol=1e-9,rtol=0)
    analytical=kinematics(sb)
    if post_action_boundary:
        require(str(raw["cost_boundary_source"])=="post-action-full-state-analytic-fk",
            "Task v2 primary boundary provenance differs")
        tip=np.stack((np.sin(sb[...,1])+np.sin(sb[...,1]+sb[...,2]),np.zeros(sb.shape[:2]),
            2.+np.cos(sb[...,1])+np.cos(sb[...,1]+sb[...,2])),axis=-1)
        same(raw["boundary_sensors"],tip if mode=="model" else tip[0],
            "Post-action boundary sensors differ from analytical FK",atol=2e-8,rtol=2e-8)
        legacy_shape=(K,h,3) if mode=="model" else (1,3)
        finite(raw["legacy_sensors"],legacy_shape)
        finite(raw["legacy_stage_costs"],(K,h) if mode=="model" else (1,))
        if mode=="model":
            finite(raw["legacy_base_scores"],(K,));finite(raw["legacy_terminal_heuristic"],(K,))
    reported_distances=finite(raw["tip_distances"],(K,h) if mode=="model" else (1,))
    same(reported_distances,(analytical["distance"] if mode=="model" else analytical["distance"][0]),
         "Analytical tip distances differ",atol=geometry_atol,rtol=2e-5 if warp else 2e-8)
    # Capture uses the recorded sensor distance after independently bounding its
    # FK error. This preserves float32 threshold rounding without dynamics replay.
    pre_holds,holds,metrics=hold_sequence(sb,snapshot["hold_count"],config,
        reported_distances if mode=="model" else reported_distances[None])
    elapsed=snapshot["elapsed"]+np.arange(h)
    obs=observations(pb,elapsed[None,:],pre_holds,config)
    flat=lambda value:value if mode=="model" else value[0]
    same(raw["observations"],flat(obs),"Observations are not pre-action state/hold aligned")
    same(raw["component_costs"],flat(metrics["components"]),"Independent raw cost components differ",atol=cost_atol,rtol=cost_rtol)
    stages=metrics["components"].sum(axis=-1)
    same(raw["stage_costs"],flat(stages),"Independent stage cost differs",atol=cost_atol,rtol=cost_rtol)
    same(raw["stage_costs"],np.asarray(raw["component_costs"]).sum(axis=-1),"Stage cost does not sum its raw components")
    same(raw["qvel_norms"],flat(metrics["speed"]),"Velocity norms differ")
    same(raw["hold_counts"],flat(holds),"Capture hold sequence differs",atol=0,rtol=0)
    terminal=metrics["terminal"][:,-1]
    reaches_end=snapshot["elapsed"]+h == config["T"]
    actual_terminal=terminal if reaches_end else np.zeros(K)
    failure=np.where(holds[:,-1] < config["capture_steps"],1000.,0.) if reaches_end else np.zeros(K)
    same(raw["terminal_costs"],actual_terminal,"Actual terminal cost must occur exactly at T",atol=cost_atol,rtol=cost_rtol)
    same(raw["capture_failures"],failure,"Capture failure must occur only at T, exactly once")
    if mode=="model":
        same(raw["terminal_heuristic"],terminal,"Model endpoint heuristic differs",atol=cost_atol,rtol=cost_rtol)
        scores=stages.sum(axis=1)+terminal+failure
        same(raw["actual_partial_costs"],stages.sum(axis=1)+actual_terminal+failure,"Actual partial return includes hypothetical endpoint cost",atol=cost_atol*(h+1),rtol=cost_rtol)
        same(raw["endpoint_warmstarts"],np.zeros((K,2)),"Model endpoint intervention changed")
        require(np.array_equal(raw["endpoint_warmstart_exact"],np.zeros(K,dtype=bool))
                and str(raw["endpoint_restart_semantics"]) == "explicit-zero-solver-warmstart-intervention",
                "Model solver warmstart is incorrectly claimed exact")
    else:
        scores=stages.sum(axis=1)+actual_terminal+failure
        same(raw["elapsed"],np.array([snapshot["elapsed"]+1]),"Live elapsed clock differs",atol=0,rtol=0)
        require(np.array_equal(raw["done"],np.array([reaches_end])) and np.array_equal(raw["captured"],
            np.array([reaches_end and holds[0,-1]>=config["capture_steps"]])),"Live terminal/capture flags differ")
        same(raw["pre_warmstarts"],np.asarray(snapshot["warmstart"])[None],"Live warmstart input differs",atol=0,rtol=0)
        finite(raw["warmstarts"],(1,2))
    same(raw["total_costs"],scores,"Raw total/score differs from independent reconstruction",atol=cost_atol*(h+1),rtol=cost_rtol)
    recorded_scores=(np.asarray(raw["stage_costs"]).sum(axis=1)+raw["terminal_heuristic"]+failure
                     if mode=="model" else raw["stage_costs"]+raw["terminal_costs"]+failure)
    same(raw["total_costs"],recorded_scores,"Raw score does not sum recorded costs")
    scores=np.asarray(raw["total_costs"],dtype=np.float64).copy()
    prefix=hashlib.sha256()
    if mode=="model":
        for value in (used,raw["observations"],raw["total_costs"]):prefix.update(np.ascontiguousarray(value).tobytes())
    weighted_mean=None
    planner_config=request["owner"].get("planner_config") if isinstance(request["owner"],dict) else None
    block_labels=verify_block_proposals(used,planner_config) if request['query_kind']=='planner_bank' else ()
    phase_labels=()
    if request['query_kind']=='planner_bank' and isinstance(planner_config,dict):
        from .phase_audit import verify_phase_proposals
        phase_labels=verify_phase_proposals(used,planner_config,snapshot['elapsed'])
    effective_temperature=planner_config.get("temperature") if isinstance(planner_config,dict) else temperature
    if effective_temperature is not None and mode=="model":
        require(type(effective_temperature) in (int,float) and np.isfinite(effective_temperature)
                and effective_temperature>0,"Invalid bound effective search temperature")
        weights=np.exp(-(scores-scores.min())/effective_temperature);weights/=weights.sum()
        weighted_mean=np.einsum("k,kha->ha",weights,used)
    return dict(mode=mode,K=K,h=h,snapshot=snapshot,owner=request["owner"],metadata=request["metadata"],
        query_kind=request["query_kind"],scores=scores,score_sum=float(scores.sum()),reaches_end=reaches_end,
        captured=int(np.sum((holds[:,-1]>=config["capture_steps"]) & reaches_end)),backend=backend,
        effective_temperature=effective_temperature,planner_config=planner_config,
        block_labels=block_labels,
        phase_labels=phase_labels,
        cost_boundary_source="post-action-full-state-analytic-fk" if post_action_boundary else "legacy-unbound-sensor-timing",
        arithmetic_tolerance=dict(geometry_atol=geometry_atol,cost_atol=cost_atol,cost_rtol=cost_rtol),
        first_plan=used[0].copy() if mode=="model" else used.copy(),
        min_plan=used[int(np.argmin(scores))].copy() if mode=="model" else used.copy(),
        plan_hashes={digest(np.ascontiguousarray(plan).tobytes()) for plan in used} if mode=="model" else set(),
        prefix=prefix,weighted_mean=weighted_mean,live_arrays=raw if mode=="live" else None)


class Reader:
    def __init__(self, root, allowed_root=None, artifact_bindings=None):
        self.root=Path(root).resolve();self.allowed_root=Path(allowed_root or root).resolve()
        self.hashes={};self.bound_npz=set()
        # Optional verifier-owned aliases bind a COMPLETE reference identity
        # to one explicit namespace. They never broaden ordinary path search.
        self.artifact_bindings={};self.artifact_binding_names=set();self.applied_artifact_bindings=[]
        for binding in artifact_bindings or ():
            require(isinstance(binding,dict) and set(binding)=={"reference","archive_root"},"Invalid artifact binding declaration")
            ref=binding["reference"];namespace=binding["archive_root"]
            require(isinstance(ref,dict) and set(ref)=={"file","sha256","bytes"}
                and isinstance(ref["file"],str) and isinstance(ref["sha256"],str)
                and len(ref["sha256"])==64 and all(c in "0123456789abcdef" for c in ref["sha256"])
                and type(ref["bytes"]) is int and ref["bytes"]>=0,"Artifact binding must include exact file/SHA256/byte identity")
            part=PurePosixPath(ref["file"])
            require(not part.is_absolute() and ".." not in part.parts,"Unsafe bound artifact name")
            require(isinstance(namespace,str),"Artifact binding namespace must be relative")
            relative=PurePosixPath(namespace)
            require(not relative.is_absolute() and ".." not in relative.parts,"Unsafe artifact binding namespace")
            archive=(self.allowed_root/namespace).resolve()
            require(archive.is_relative_to(self.allowed_root),"Artifact binding namespace escapes allowed run")
            key=(ref["file"],ref["sha256"],ref["bytes"])
            require(key not in self.artifact_bindings,"Duplicate or conflicting artifact binding")
            self.artifact_bindings[key]=archive
            self.artifact_binding_names.add(ref["file"])

    def bound(self, reference, archive_root=None):
        require(isinstance(reference,dict) and isinstance(reference.get("file"),str),"Invalid artifact reference")
        name=reference["file"];part=PurePosixPath(name)
        require(".." not in part.parts,"Artifact traversal")
        if "bytes" in reference:require(type(reference["bytes"]) is int and reference["bytes"]>=0,"Artifact byte count must be an integer")
        key=(name,reference.get("sha256"),reference.get("bytes"))
        if archive_root is None and name in self.artifact_binding_names:
            require(key in self.artifact_bindings,"Bound artifact identity differs from explicit file/SHA256/byte binding")
        bound_archive=self.artifact_bindings.get(key) if archive_root is None else None
        if bound_archive is not None:
            candidates={(bound_archive/name).resolve()}
        elif part.is_absolute():candidates={Path(name).resolve()}
        elif archive_root is not None:candidates={(Path(archive_root)/name).resolve()}
        else:
            # Producer checkpoints historically store project-relative paths;
            # raw arrays store archive-relative paths. Resolve either only if
            # it names one existing file inside this explicitly allowed run.
            bases=(self.root,self.allowed_root,*self.allowed_root.parents)
            candidates={(base/name).resolve() for base in bases}
        safe={p for p in candidates if p.is_relative_to(self.allowed_root)}
        require(safe,"Artifact escapes allowed run directory")
        existing={p for p in safe if p.is_file()}
        require(len(existing)==1,"Artifact missing or ambiguous inside allowed run: "+name)
        path=existing.pop()
        raw=path.read_bytes();actual=digest(raw)
        require(actual == reference["sha256"],"Artifact SHA256 mismatch: "+name)
        if "bytes" in reference:require(len(raw)==reference["bytes"],"Artifact byte count mismatch: "+name)
        self.hashes[str(path)]=actual
        if bound_archive is not None:
            binding=dict(reference=dict(file=name,sha256=reference["sha256"],bytes=reference["bytes"]),
                archive_root=str(bound_archive.relative_to(self.allowed_root)),resolved_path=str(path))
            if binding not in self.applied_artifact_bindings:self.applied_artifact_bindings.append(binding)
        if path.suffix==".npz":self.bound_npz.add(path)
        return path,raw

    def arrays(self, reference):
        _,raw=self.bound(reference)
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
            arrays={key:archive[key].copy() for key in archive.files}
        require(not any(a.dtype.hasobject for a in arrays.values()),"Pickled evidence prohibited")
        return arrays

    def harvest(self,value,archive_root=None):
        if isinstance(value,dict):
            # Paired backend records carry references to BOTH sibling
            # archives. Their explicit archive namespace owns nested NPZs.
            if ("archive" in value or "query_archive" in value) and "inputs" in value and "result" in value:
                if "archive" in value and "query_archive" in value:
                    require(value["archive"]==value["query_archive"],"Conflicting archive/query_archive namespaces")
                namespace=value["archive"] if "archive" in value else value["query_archive"]
                require(isinstance(namespace,str),"Referenced archive must be a relative path")
                part=PurePosixPath(namespace)
                require(not part.is_absolute() and ".." not in part.parts,"Unsafe referenced archive namespace")
                archive_root=(self.allowed_root/namespace).resolve()
                require(archive_root.is_relative_to(self.allowed_root),"Referenced archive escapes allowed run")
            if isinstance(value.get("file"),str) and "sha256" in value:self.bound(value,archive_root)
            for child in value.values():self.harvest(child,archive_root)
        elif isinstance(value,list):
            for child in value:self.harvest(child,archive_root)


def load_ledger(path):
    rows=[];partial=False
    lines=Path(path).read_bytes().splitlines(keepends=True)
    for i,line in enumerate(lines):
        try:
            row=json.loads(line)
            require(isinstance(row,dict),"Ledger record is not an object")
            rows.append(row)
        except (ValueError,UnicodeError):
            if i==len(lines)-1 and not line.endswith(b"\n"):partial=True
            else:raise ValueError("Malformed complete ledger line")
    return rows,partial


def ledger_counts(rows, partial=False):
    """Observed intent is retained even when later evidence validation fails."""
    requests=[r for r in rows if r.get("kind")=="query_requested"]
    completed=[r for r in rows if r.get("kind")=="query_complete"]
    errors=[r for r in rows if r.get("kind")=="query_error"]
    valid_id=lambda r:isinstance(r.get("query_id"),str) and bool(r["query_id"])
    begins={r["query_id"]:r for r in requests if valid_id(r)}
    done={r["query_id"]:r for r in completed if valid_id(r)}
    failed={r["query_id"]:r for r in errors if valid_id(r)}
    returned={k:r for k,r in failed.items() if r.get("call_returned") is True}
    known={**done,**returned};unknown=set(begins)-set(done)-set(failed)
    number=lambda r,k:int(r.get(k,0)) if type(r.get(k,0)) is int and r.get(k,0)>=0 else 0
    requested=sum(number(r,"requested_controls") for r in requests)
    physical=sum(number(r,"requested_physics_controls") for r in requests)
    counts=dict(requested_controls=requested,charged_requested_controls=requested,
        requested_physics_controls=physical,requested_physics_frames=sum(number(r,"requested_physics_frames") for r in requests),
        completed_controls=sum(number(r,"completed_controls") for r in known.values()),
        completed_physics_frames=sum(number(r,"completed_physics_frames") for r in known.values()),
        cutoff_skipped_controls=requested-physical,error_requested_controls=sum(number(begins.get(q,{}),"requested_controls") for q in failed),
        unknown_error_controls=sum(number(begins.get(q,{}),"requested_physics_controls") for q in failed if q not in returned),
        inflight_or_interrupted_controls=sum(number(begins[q],"requested_physics_controls") for q in unknown),
        unknown_controls=sum(number(begins.get(q,{}),"requested_physics_controls") for q in unknown | (set(failed)-set(returned))),
        queries=len(requests),completed_queries=len(completed),error_queries=len(errors),
        inflight_or_interrupted_queries=len(unknown),rejected_queries=sum(r.get("kind")=="query_rejected" for r in rows),
        incomplete_final_line=partial)
    return counts,begins,done,failed


def neural_counts(rows):
    """Keep durable forward-call intent separate from successful returns.

    Legacy post-call inference summaries are descriptive and may overlap the
    newer hook meter; they are therefore never added to hook-meter charges.
    """
    requests=[r for r in rows if r.get("kind")=="neural_requested"]
    completed=[r for r in rows if r.get("kind")=="neural_complete"]
    number=lambda r:int(r.get("rows",0)) if type(r.get("rows",0)) is int and r.get("rows",0)>=0 else 0
    requested={r["id"]:r for r in requests if isinstance(r.get("id"),str)}
    resolutions={r["request_id"]:r for r in completed if isinstance(r.get("request_id"),str)}
    unresolved=set(requested)-set(resolutions)
    legacy=sum(number(r) for r in rows if r.get("kind")=="inference")
    returned=sum(number(r) for r in completed if r.get("returned") is True)
    return dict(neural_requested_rows=sum(number(r) for r in requests),
        charged_neural_rows=sum(number(r) for r in requests),
        neural_complete_rows=sum(number(r) for r in completed),
        neural_returned_rows=returned,
        neural_failed_rows=sum(number(r) for r in completed if r.get("returned") is False),
        neural_unresolved_rows=sum(number(requested[k]) for k in unresolved),
        neural_requested_calls=len(requests),neural_complete_calls=len(completed),
        neural_unresolved_calls=len(unresolved),legacy_inference_rows=legacy,
        recorded_inference_rows=returned if requests else legacy)


def validate_neural_events(rows):
    requested={};resolved=set()
    for row in rows:
        if row.get("kind")=="inference":integer(row.get("rows"),"Legacy inference rows")
        elif row.get("kind")=="neural_requested":
            integer(row.get("rows"),"Neural requested rows",1)
            require(row["id"] not in requested,"Repeated neural intent")
            require(isinstance(row.get("purpose"),str) and bool(row["purpose"])
                and isinstance(row.get("policy_type"),str) and bool(row["policy_type"]),
                "Neural intent purpose/type missing")
            requested[row["id"]]=row
        elif row.get("kind")=="neural_complete":
            ref=row.get("request_id")
            require(isinstance(ref,str) and ref in requested and ref not in resolved,
                "Neural resolution lacks unique earlier durable intent")
            integer(row.get("rows"),"Neural completion rows",1)
            require(row["rows"]==requested[ref]["rows"] and type(row.get("returned")) is bool,
                "Neural completion rows/return status differ")
            wall=row.get("compute_wall_seconds")
            require(type(wall) in (int,float) and math.isfinite(wall) and wall>=0,
                "Neural compute wall time invalid")
            resolved.add(ref)
    return set(requested)-resolved


def optimizer_work_counts(rows):
    """Charge requests before validation, retaining failure/interruption work."""
    number=lambda r,k:int(r.get(k,0)) if type(r.get(k,0)) is int and r.get(k,0)>=0 else 0
    fit=[r for r in rows if r.get("kind")=="fit_work"]
    candidate=sum(number(r,"gradient_steps") for r in fit)
    bootstrap=sum(number(r,"updates") for r in fit)
    diagnostic_summary=sum(number(r,"diagnostic_backward_passes") for r in fit)
    new=any(str(r.get("kind","")).startswith(tuple(k+"_" for k in STEP_FIELDS)) for r in rows)
    counts=dict(fit_summary_optimizer_steps=candidate+bootstrap,
        fit_summary_diagnostic_backward_passes=diagnostic_summary,fit_work_records=len(fit),
        optimizer_count_basis="durable_requested" if new else "legacy_completed_fit_reports" if fit else "no_fitting_recorded")
    for prefix,unit in STEP_FIELDS.items():
        requested=[r for r in rows if r.get("kind")==prefix+"_requested"]
        complete=[r for r in rows if r.get("kind")==prefix+"_complete"]
        failed=[r for r in rows if r.get("kind")==prefix+"_error"]
        resolved={r["request_id"] for r in complete+failed if isinstance(r.get("request_id"),str)}
        suffix="steps" if prefix=="optimizer" else "passes"
        counts[prefix+"_requested_"+suffix]=len(requested)
        counts[prefix+"_complete_"+suffix]=len(complete)
        counts[prefix+"_error_"+suffix]=len(failed)
        counts[prefix+"_unresolved_"+suffix]=sum(not isinstance(r.get("id"),str) or r["id"] not in resolved for r in requested)
    opt=[r for r in rows if r.get("kind")=="optimizer_requested"]
    boot_requests=sum(r.get("purpose") in ("shared_bootstrap","auxiliary_bootstrap") for r in opt)
    counts.update(recorded_optimizer_gradient_steps=len(opt) if new else candidate+bootstrap,
        recorded_candidate_optimizer_steps=len(opt)-boot_requests if new else candidate,
        recorded_bootstrap_optimizer_steps=boot_requests if new else bootstrap,
        recorded_gradient_backward_passes=counts["gradient_backward_requested_passes"],
        recorded_diagnostic_backward_passes=counts["diagnostic_backward_requested_passes"] if new else diagnostic_summary)
    return counts


def validate_optimizer_events(rows):
    requested={};resolved=set();completed={p:0 for p in STEP_FIELDS};provisional=[]
    new=any(str(r.get("kind","")).startswith(tuple(k+"_" for k in STEP_FIELDS)) for r in rows)
    for row in rows:
        kind=row.get("kind")
        for prefix,unit in STEP_FIELDS.items():
            if kind==prefix+"_requested":
                require(integer(row.get(unit),prefix+" units")==1,"Tensor work request must contain exactly one unit")
                require(isinstance(row.get("purpose"),str) and bool(row["purpose"]),"Tensor-work purpose missing")
                require(row.get("scope") is None or isinstance(row["scope"],str),"Tensor-work scope malformed")
                if prefix=="diagnostic_backward":
                    require(row.get("diagnostic")=="first_minibatch_pg_only" and row.get("retain_graph") is True,
                        "Diagnostic backward identity/graph semantics differ")
                requested[row["id"]]=(prefix,row)
            elif kind in (prefix+"_complete",prefix+"_error"):
                ref=row.get("request_id")
                require(isinstance(ref,str) and ref in requested and requested[ref][0]==prefix and ref not in resolved,
                    "Tensor-work resolution lacks unique earlier intent")
                original=requested[ref][1]
                require(integer(row.get(unit),prefix+" resolution units")==1 and row.get("scope")==original.get("scope"),
                    "Tensor-work resolution changed scope/units")
                wall=row.get("compute_wall_seconds")
                require(type(wall) in (int,float) and math.isfinite(wall) and wall>=0,"Tensor-work wall time invalid")
                resolved.add(ref)
                if kind.endswith("_complete"):completed[prefix]+=1
                else:
                    require(row.get("partial_work_unknown") is True and isinstance(row.get("error_type"),str)
                        and isinstance(row.get("error"),str),"Tensor-work error erased partial-work uncertainty")
                    provisional.append("failed "+prefix+" request: "+ref)
    provisional.extend("unresolved "+prefix+" request: "+ref
        for ref,(prefix,_) in requested.items() if ref not in resolved)
    summary=diagnostic=0
    for row in rows:
        if row.get("kind")!="fit_work":continue
        require(("gradient_steps" in row)!=("updates" in row),"Fit work has ambiguous/missing optimizer-step count")
        summary+=integer(row.get("gradient_steps",row.get("updates")),"Fit optimizer-step summary")
        diagnostic+=integer(row.get("diagnostic_backward_passes",0),"Fit diagnostic-backward summary")
        wall=row.get("fit_wall_seconds")
        require(type(wall) in (int,float) and math.isfinite(wall) and wall>=0,"Fit work wall time invalid")
    if new:
        require(summary<=completed["optimizer"],"Fit optimizer summaries exceed completed metered steps")
        require(diagnostic<=completed["diagnostic_backward"],"Fit diagnostic summaries exceed completed metered backwards")
    return provisional


def audit_episode_arrays(raw, event, queries, config):
    initial=packed(raw,"initial_snapshot_json");T=config["T"]
    n=len(raw["actions"]);require(n>0 and n<=T-initial["elapsed"],"Episode has fake post-termination padding")
    s=finite(raw["states"],(n+1,5));w=finite(raw["warmstarts"],(n+1,2))
    actions=finite(raw["actions"],(n,1));require(np.all(abs(actions)<=1.),"Episode action bounds violated")
    elapsed=initial["elapsed"]+np.arange(n+1)
    same(raw["elapsed"],elapsed,"Episode elapsed controls not consecutive",atol=0,rtol=0)
    same(s[:,0],elapsed*.01,"Episode clock has repeated/padded states",atol=1e-9,rtol=0)
    same(s[0],initial["state"],"Episode initial state differs",atol=0,rtol=0)
    same(w[0],initial["warmstart"],"Episode initial warmstart differs",atol=0,rtol=0)
    pre_h,post_h,metrics=hold_sequence(s[None,1:],initial["hold_count"],config)
    holds=np.r_[initial["hold_count"],post_h[0]]
    same(raw["hold_counts"],holds,"Episode capture window differs",atol=0,rtol=0)
    same(raw["observations"],observations(s,elapsed,holds,config),"Episode observations not boundary aligned")
    same(raw["component_costs"],metrics["components"][0],"Episode raw components differ")
    stages=metrics["components"][0].sum(axis=-1)
    terminal=np.zeros(n);failure=np.zeros(n)
    reaches_end=bool(elapsed[-1]==T)
    captured=bool(reaches_end and holds[-1]>=config["capture_steps"])
    if reaches_end:
        terminal[-1]=metrics["terminal"][0,-1]
        failure[-1]=0. if captured else 1000.
    for key,value in (("stage_costs",stages),("terminal_costs",terminal),("capture_failures",failure)):
        same(raw[key],value,"Episode "+key+" differs / terminal charged outside T")
    allmetrics=kinematics(s)
    same(raw["tip_distances"],allmetrics["distance"],"Episode tip distances differ")
    require(scalar(raw,"T")==T and scalar(raw,"complete") is (initial["elapsed"]==0 and reaches_end)
            and scalar(raw,"captured") is captured,"Episode completeness/capture metadata differs")
    owner=packed(raw,"actor_owner_json");semantics=str(raw["execution_semantics"])
    reference=semantics in ("analytic_reference","native_shooting_reference")
    require(isinstance(owner,dict) and semantics in ("actor","executed_replanning","analytic_reference","native_shooting_reference")
            and owner.get("semantics")==semantics and bool(owner.get("controller_hash")),
            "Episode controller ownership is incomplete")
    if reference:
        require(owner.get("actor_hash") is None and owner.get("excluded_from_teacher_aux") is True,
            "Analytic reference must have no actor owner and be excluded from teacher supervision")
        source=packed(raw,"source_json")
        require(source.get("controller")==semantics and source.get("excluded_from_teacher_aux") is True,
            "Analytic reference provenance exclusion missing")
    else:
        require(type(owner.get("deterministic")) is bool and type(owner.get("family")) is int,
            "Actor/replanning deterministic family identity is incomplete")
    require(semantics!="actor" or isinstance(owner.get("actor_hash"),str),"Actor episode has no exact owner")
    refs=tuple(str(x) for x in raw["query_ids"])
    require(len(refs)==n and len(set(refs))==n,"Episode query references duplicated/missing")
    for t,ref in enumerate(refs):
        require(ref in queries and queries[ref]["mode"]=="live","Episode references non-live or unverified query")
        query=queries[ref];qraw=query["live_arrays"]
        require(canonical(query["owner"])==canonical(owner)
                and query["snapshot"]["context_id"]==initial["context_id"],"Episode switches controller/context")
        same(qraw["prestates"][0],s[t],"Episode/query prestate differs",atol=0,rtol=0)
        same(qraw["states"][0],s[t+1],"Episode/query poststate differs",atol=0,rtol=0)
        same(qraw["pre_warmstarts"][0],w[t],"Episode/query prewarmstart differs",atol=0,rtol=0)
        same(qraw["warmstarts"][0],w[t+1],"Episode/query warmstart differs",atol=0,rtol=0)
        same(qraw["actions"][0],actions[t],"Episode/query action differs",atol=0,rtol=0)
        require(query["metadata"].get("episode_id")==event["episode_id"],"Episode query ancestry differs")
        if reference:require(query["metadata"].get("excluded_from_teacher_aux") is True,
            "Analytic reference live query lost teacher exclusion")
    total=float(stages.sum()+terminal.sum()+failure.sum())
    same(scalar(raw,"total_cost"),total,"Episode complete return differs")
    same(event["total_cost"],total,"Saved episode event return differs")
    require(canonical(event["actor_owner"])==canonical(owner) and event["execution_semantics"]==semantics,
            "Saved episode owner event differs")
    require(event["context_id"]==initial["context_id"] and event["task_hash"]==initial["task_hash"],"Saved episode identity differs")
    hit=np.flatnonzero(allmetrics["distance"]<=config["local_distance"])
    if len(hit):
        d=float(np.unwrap(s[:,1])[hit[0]]-np.unwrap(s[:,1])[0])
        family="positive" if d>1e-9 else "negative" if d< -1e-9 else "stationary"
    else:family="unreached"
    require(str(raw["family"])==event["family"]==family,"Episode family metadata differs")
    require(event["complete"] is (initial["elapsed"]==0 and reaches_end)
            and event["successful"] is (initial["elapsed"]==0 and reaches_end and captured),"Saved success partition is incorrect")
    return dict(episode_id=event["episode_id"],context_id=initial["context_id"],cost=total,capture=captured,
        complete=reaches_end,ordinary_start=initial["elapsed"]==0 and (reference or initial["context_id"].split(":")[0]
            in ("ordinary","actor_eval","actor_train")),root_start=initial["elapsed"]==0,
        remaining_steps=T-initial["elapsed"],
        local=bool(np.any((elapsed<=config["local_by"]) & (allmetrics["distance"]<=config["local_distance"]))),
        actor_hash=owner.get("actor_hash"),controller_hash=owner["controller_hash"],family_id=None if reference else owner["family"],
        task_hash=initial["task_hash"],query_ids=refs,deterministic=True if reference else owner["deterministic"],
        excluded_from_teacher_aux=reference,
        execution_semantics=semantics,stages=stages,terminals=terminal,failures=failure)


def audit_archive(path, *, source_root=None, allowed_root=None, require_closed=True, temperature=None,
                  artifact_bindings=None):
    started=time.perf_counter();root=Path(path).resolve();errors=[];pending=[];queries={};episodes={};decisions=[]
    reader=Reader(root,allowed_root,artifact_bindings);counts={};rows=[];partial=False;configs={};request_seq={};completion_seq={};decision_keys=set()
    def guard(label,call):
        try:return call()
        except (ValueError,KeyError,TypeError,IndexError,OSError,OverflowError,FloatingPointError,
                SyntaxError,EOFError,zipfile.BadZipFile) as error:
            errors.append(label+": "+str(error));return None
    loaded=guard("ledger",lambda:load_ledger(root/"ledger.jsonl"))
    if loaded is None:
        return dict(schema="bidirectional-independent-arithmetic-audit-v1",status="FAIL",arithmetic_valid=False,
                    errors=errors,pending=pending,work=counts,archive=str(root))
    rows,partial=loaded
    counts,begins,done,failed=ledger_counts(rows,partial)
    reader.hashes[str(root/"ledger.jsonl")]=digest((root/"ledger.jsonl").read_bytes())
    if partial:pending.append("unfinished ledger tail")
    def ledger_checks():
        require(rows and rows[0]["kind"]=="archive_begin" and rows[0].get("schema")=="bidirectional-evidence-v1","Archive schema/begin missing")
        require([r["id"] for r in rows]==[f"event-{i:09d}" for i in range(len(rows))],"Ledger IDs repeat or are reordered")
        clocks=[r["wall_seconds"] for r in rows];require(all(math.isfinite(v) and v>=0 for v in clocks)
            and all(a<=b for a,b in zip(clocks,clocks[1:])),"Ledger wall clock regressed")
        require(len(begins)==counts["queries"] and len(done)==counts["completed_queries"]
            and len(failed)==counts["error_queries"],"Query ID duplicated")
        require(not set(done).intersection(failed) and (set(done)|set(failed))<=set(begins),"Query resolutions lack unique requests")
        for i,row in enumerate(rows):
            if row["kind"]=="query_requested":
                for key in ("requested_controls","requested_physics_controls","requested_physics_frames","cutoff_skipped_controls"):
                    integer(row.get(key),key,0 if key=="cutoff_skipped_controls" else 1)
                request_seq[row["query_id"]]=i
            if row["kind"] in ("query_complete","query_error"):
                for key in ("completed_controls","completed_physics_frames"):
                    integer(row.get(key),key,1 if row["kind"]=="query_complete" else 0)
                require(row["query_id"] in request_seq,"Query resolution precedes durable intent")
                completion_seq[row["query_id"]]=i
            if row["kind"]=="archive_close":
                require(i==len(rows)-1,"Events appended after archive close")
                for key,value in counts.items():require(row["accounting"].get(key)==value,"Archive close accounting differs: "+key)
        if require_closed and rows[-1]["kind"]!="archive_close":pending.append("archive still open")
    guard("ledger invariants",ledger_checks)
    for row in rows:guard("artifact "+row.get("id","?"),lambda row=row:reader.harvest(row))
    source_root=Path(source_root or Path(__file__).resolve().parents[3])
    for ref,request in begins.items():
        def one_query(ref=ref,request=request):
            inputs=reader.arrays(request["inputs"]);config=validate_config(packed(inputs,"task_json"))
            expected_hash,sources=task_identity(config,source_root)
            configs[expected_hash]=config
            for name,value in sources.items():reader.hashes[str(source_root/name)]=value
            snap=packed(inputs,"initial_snapshot_json");validate_snapshot(snap,config,expected_hash)
            action=finite(inputs["actions"])
            require(np.all(abs(action)<=1.),"Requested actions violate native bounds")
            mode=request["mode"];require(mode in ("model","live"),"Unknown request mode")
            if mode=="model":
                require(action.ndim==3 and action.shape[0]>0 and action.shape[1]>0 and action.shape[2]==1,"Invalid requested bank")
                K,H=action.shape[:2]
            else:
                require(action.shape==(1,),"Invalid requested live action")
                K,H=1,1
            remaining=config["T"]-snap["elapsed"];require(remaining>0,"Requested control after finite episode")
            require(request["requested_controls"]==K*H and request["requested_physics_controls"]==K*min(H,remaining)
                and request["requested_physics_frames"]==K*min(H,remaining)
                and request["cutoff_skipped_controls"]==K*(H-min(H,remaining)),"Request invents or hides finite-horizon padding")
            require(canonical(packed(inputs,"owner_json"))==canonical(request["owner"])
                and canonical(packed(inputs,"metadata_json"))==canonical(request["metadata"]),"Failed/request input metadata differs")
            if ref not in done:
                pending.append("failed/unresolved query has no complete raw result: "+ref)
                return
            completion=done[ref]
            require(completion["completed_controls"]==request["requested_physics_controls"]
                    and completion["completed_physics_frames"]==request["requested_physics_frames"],"Query completed count mismatch")
            require(canonical(completion["owner"])==canonical(request["owner"]),"Query completion owner changed")
            require(completion["scope"]==request["scope"],"Query completion scope changed")
            raw=reader.arrays(completion["result"])
            info=audit_query_arrays(inputs,raw,request,config,expected_hash,temperature,
                post_action_boundary="src/gps/bidirectional/task.py" in sources)
            if info["mode"]=="model":require(canonical(completion["backend"])==canonical(info["backend"]),"Model backend result/ledger differs")
            else:require(completion["backend"]=="mujoco_cpu","Live completion backend differs")
            info["request_index"]=request_seq[ref];info["completion_index"]=completion_seq[ref]
            info["result_file"]=completion["result"]["file"]
            queries[ref]=info
        guard("query "+ref,one_query)
    for ref,error in failed.items():
        def one_error(ref=ref,error=error):
            request=begins[ref];require(type(error["call_returned"]) is bool,"Error return status unknown")
            expected=request["requested_physics_controls"] if error["call_returned"] else 0
            require(error["completed_controls"]==expected and error["completed_physics_frames"]==expected
                    and error["partial_physics_unknown"] is (not error["call_returned"]),"Failed query erased or invented physical work")
        guard("error "+ref,one_error)
    episode_events={};outcomes={}
    for index,row in enumerate(rows):
        if row["kind"]=="episode_saved":
            def one_episode(row=row,index=index):
                require(row["episode_id"] not in episode_events,"Repeated episode artifact")
                raw=reader.arrays(row["raw"]);initial=packed(raw,"initial_snapshot_json")
                require(initial["task_hash"] in configs,"Episode refers to unverified task")
                validate_snapshot(initial,configs[initial["task_hash"]],initial["task_hash"])
                result=audit_episode_arrays(raw,row,queries,configs[initial["task_hash"]])
                require(all(queries[q]["completion_index"]<index for q in result["query_ids"]),"Episode claims future unreturned controls")
                episode_events[row["episode_id"]]=row;episodes[row["episode_id"]]=result
            guard("episode "+row["episode_id"],one_episode)
        elif row["kind"]=="episode_outcome":
            def one_outcome(row=row):
                name=row["episode_id"];require(name in episodes and name not in outcomes,"Outcome lacks unique verified episode")
                e=episodes[name]
                for key in ("context_id","capture","complete","ordinary_start","remaining_steps","local",
                            "actor_hash","controller_hash","family_id","task_hash","deterministic"):
                    require(row[key]==e[key],"Episode outcome field mismatch: "+key)
                if "root_start" in row:require(row["root_start"] is e["root_start"],"Episode root-start partition differs")
                if e["execution_semantics"] in ("analytic_reference","native_shooting_reference"):
                    require(row.get("excluded_from_teacher_aux") is True and row.get("semantics")==e["execution_semantics"],
                        "Analytic reference outcome lost separate benchmark identity")
                same(row["cost"],e["cost"],"Episode outcome return mismatch")
                require(tuple(row["query_ids"])==e["query_ids"],"Episode outcome query ownership differs")
                outcomes[name]=row
            guard("outcome "+row["episode_id"],one_outcome)
        elif row["kind"]=="decision":
            def one_decision(row=row,index=index):
                key=(row["episode_id"],row["elapsed"]);require(key not in decision_keys,"Repeated decision at one control")
                decision_keys.add(key);raw=reader.arrays(row["raw"])
                bankrefs=tuple(row["bank_query_ids"]);guidedrefs=tuple(row["guided_query_ids"])
                require(bankrefs and len(bankrefs)%2==0 and len(set(bankrefs+guidedrefs))==len(bankrefs+guidedrefs),"Decision model references missing/duplicated")
                for ref in bankrefs+guidedrefs:
                    require(ref in queries and queries[ref]["mode"]=="model"
                            and queries[ref]["completion_index"]<index,"Decision references future/unverified/live model")
                    require(queries[ref]["snapshot"]["elapsed"]==row["elapsed"],"Decision model clock differs")
                nominal=finite(raw["nominal_before"]);require(nominal.ndim==2 and nominal.shape[1]==1,"Invalid private nominal")
                champion=score=None
                for bankref,meanref in zip(bankrefs[::2],bankrefs[1::2]):
                    bank,mean=queries[bankref],queries[meanref]
                    require(bank["query_kind"]=="planner_bank" and mean["query_kind"]=="independent_mean_verification"
                            and mean["K"]==1 and canonical(bank["snapshot"])==canonical(mean["snapshot"]),"Independent mean verification identity differs")
                    require(canonical(bank["planner_config"])==canonical(mean["planner_config"]),
                        "Mean verification effective planner configuration differs")
                    expected_nominal=nominal[:bank["h"]] if champion is None else champion
                    same(bank["first_plan"],expected_nominal,"Independent proposal lost its private nominal",atol=0,rtol=0)
                    if bank["weighted_mean"] is not None:
                        same(mean["first_plan"],bank["weighted_mean"],"Independent mean was not computed from raw task scores",atol=1e-12,rtol=1e-12)
                    else:pending.append("Frozen search temperature absent; cannot verify independent mean weights")
                    if mean["scores"][0]<=bank["scores"].min():champion,score=mean["first_plan"],float(mean["scores"][0])
                    else:champion,score=bank["min_plan"],float(bank["scores"].min())
                same(raw["independent_plan"],champion,"Saved independent champion differs",atol=0,rtol=0)
                same(raw["independent_score"],score,"Saved independent score differs")
                finalbank=queries[bankrefs[-2]];hashed=finalbank["prefix"].copy();hashed.update(np.ascontiguousarray(champion).tobytes())
                require(str(raw["bank_digest"])==hashed.hexdigest(),"Saved bank digest is not raw model bank")
                allowed={"own_gaussian","independent_knots","own_memory_fallback"}|set(finalbank['block_labels'])|set(finalbank['phase_labels'])
                require(len(raw["sources"])==finalbank["K"] and all(str(s) in allowed
                    or str(s).startswith("memory:") for s in raw["sources"]),"Unsupported/actor-seeded candidate source")
                if finalbank['block_labels']:
                    start=finalbank['planner_config']['K']
                    require(tuple(str(s) for s in raw['sources'][start:start+len(finalbank['block_labels'])])
                        ==finalbank['block_labels'],'Independent block provenance order differs')
                if finalbank['phase_labels']:
                    start=finalbank['planner_config']['K']
                    require(tuple(str(s) for s in raw['sources'][start:start+32])
                        ==finalbank['phase_labels'],'Independent phase provenance order differs')
                    checker=Path(__file__).with_name('phase_audit.py')
                    reader.hashes[str(checker)]=digest(checker.read_bytes())
                after=nominal.copy();after[:len(champion)]=champion;after[:-1]=after[1:]
                after[-1]=after[-2] if len(after)>1 else after[-1]
                same(raw["nominal_after"],after,"Guidance changed private nominal update",atol=0,rtol=0)
                chosen=finite(raw["executed_plan"],champion.shape)
                permissible=finalbank["plan_hashes"]|{digest(champion.tobytes())}
                for ref in guidedrefs:
                    q=queries[ref];require(q["query_kind"]=="guided_mean_verification" and q["K"]==1,"Unverified guided mean")
                    require(canonical(q["snapshot"])==canonical(finalbank["snapshot"]),"Guided mean source state differs")
                    permissible.add(digest(q["first_plan"].tobytes()))
                require(digest(chosen.tobytes()) in permissible,"Executed hypothetical plan was never independently modeled")
                require(type(row["influenced"]) is bool and type(row["changed"]) is bool
                    and row["changed"] is (not np.array_equal(chosen[0],champion[0])),"Decision change flag differs")
                if not row["influenced"]:
                    same(chosen,champion,"Guidance-off execution differs from independent plan",atol=0,rtol=0)
                    require(not guidedrefs,"Guidance-off decision paid unlabelled guided verification")
                decisions.append(dict(episode_id=row["episode_id"],elapsed=row["elapsed"],controller_hash=row["controller_hash"],
                    actor_hash=row["actor_hash"],family_id=row["family"],influenced=row["influenced"],changed=row["changed"],
                    bank_query_ids=bankrefs,guided_query_ids=guidedrefs,action=chosen[0].tolist(),
                    source_snapshot=finalbank["snapshot"]))
            guard("decision "+str(row.get("episode_id")),one_decision)
    claimed={q for e in episodes.values() for q in e["query_ids"]}
    for ref,q in queries.items():
        if q["mode"]=="live" and q["metadata"].get("episode_id") and ref not in claimed:
            pending.append("live episode query lacks closed trace: "+ref)
    for d in decisions:
        d["execution_verified"]=False
        def align(d=d):
            # A failed episode can retain valid decisions/returned controls
            # without ever reaching an episode_saved event. Bind these raw
            # controls directly, but keep the missing complete trace pending.
            matches=[q for q in queries.values() if q["mode"]=="live"
                and q["metadata"].get("episode_id")==d["episode_id"]
                and q["snapshot"]["elapsed"]==d["elapsed"]]
            if not matches:
                pending.append("modeled decision has no returned actual control: "+d["episode_id"]+":"+str(d["elapsed"]))
                return
            require(len(matches)==1,"Decision has ambiguous duplicate actual returned controls")
            q=matches[0]
            require(canonical(q["snapshot"])==canonical(d["source_snapshot"]),
                "Decision models and actual control start from different full snapshots")
            require(q["owner"]["controller_hash"]==d["controller_hash"]
                and q["owner"].get("actor_hash")==d["actor_hash"] and q["owner"]["family"]==d["family_id"],"Decision/executed owner differs")
            same(q["live_arrays"]["actions"][0],d["action"],"Selected action was not actually executed",atol=0,rtol=0)
            d["execution_verified"]=True
        guard("decision-to-control",align)
    for name,row in outcomes.items():
        ds=[d for d in decisions if d["episode_id"]==name]
        def verify_counts(row=row,ds=ds,name=name):
            require(row["influenced"]==sum(d["influenced"] for d in ds)
                and row["changed"]==sum(d["changed"] for d in ds),"Outcome decision change/influence counts differ")
            e=episodes[name]
            if e["execution_semantics"]=="executed_replanning":
                require(len(ds)==len(e["query_ids"]),"Replanning trace omits controller decisions")
            else:require(not ds,"Actor-only trace contains planner decisions")
        guard("decision counts "+name,verify_counts)
    for path in (root/"behavior").glob("*.npz") if (root/"behavior").exists() else ():
        def behavior(path=path):
            with np.load(path,allow_pickle=False) as f:raw={k:f[k] for k in f.files}
            episode_path=str(raw["episode_file"])
            matches=[name for name,row in episode_events.items() if row["raw"]["file"]==episode_path]
            require(len(matches)==1,"Behavior targets lack bound actual episode")
            e=episodes[matches[0]];n=len(e["stages"])
            same(raw["raw_costs"],e["stages"]+e["terminals"]+e["failures"],"Behavior full-return targets have foreign continuation")
            require(str(raw["behavior_actor_hash"])==e["actor_hash"] and str(raw["behavior_controller_hash"])==e["controller_hash"],"Behavior target owner differs")
            finite(raw["pre_tanh"],(n,1));finite(raw["old_log_probs"],(n,))
            reader.hashes[str(path.resolve())]=digest(path.read_bytes())
            if path.resolve() not in reader.bound_npz:pending.append("behavior file lacks execution-time digest binding: "+str(path))
        guard("behavior "+path.name,behavior)
    counts.update(neural_counts(rows))
    counts['phase_proposal_controls_verified']=sum(32*q['h'] for q in queries.values() if q.get('phase_labels'))
    unresolved=guard("neural ledger",lambda:validate_neural_events(rows))
    if unresolved:pending.append("unresolved durable neural forward intents: "+str(len(unresolved)))
    counts["recorded_fit_updates"]=sum(r["kind"]=="actor_update" for r in rows)
    counts["actor_update_proposals"]=counts["recorded_fit_updates"]
    counts["accepted_actor_update_proposals"]=sum(r["kind"]=="actor_update" and r.get("accepted") is True for r in rows)
    counts.update(optimizer_work_counts(rows))
    tensor_pending=guard("optimizer/backward ledger",lambda:validate_optimizer_events(rows))
    if tensor_pending:pending.extend(tensor_pending)
    status="FAIL" if errors else "PENDING" if pending else "PASS"
    public_queries=[dict(query_id=ref,mode=q["mode"],query_kind=q["query_kind"],context_id=q["snapshot"]["context_id"],
        initial_elapsed=q["snapshot"]["elapsed"],candidates=q["K"],executed_horizon=q["h"],
        actual_controls=q["K"]*q["h"],score_sum=q["score_sum"],reaches_end=q["reaches_end"],captured=q["captured"],
        backend=q["backend"],effective_temperature=q["effective_temperature"],
        cost_boundary_source=q["cost_boundary_source"],arithmetic_tolerance=q["arithmetic_tolerance"])
        for ref,q in queries.items()]
    public_episodes=[{k:v for k,v in e.items() if k not in ("stages","terminals","failures")} for e in episodes.values()]
    return dict(schema="bidirectional-independent-arithmetic-audit-v1",status=status,arithmetic_valid=not errors,
        closed=bool(rows and rows[-1]["kind"]=="archive_close" and not partial),errors=errors,pending=pending,
        archive=str(root),work=counts,queries=public_queries,episodes=public_episodes,decisions=decisions,
        input_sha256=reader.hashes,source_sha256=digest(Path(__file__).read_bytes()),
        applied_artifact_bindings=reader.applied_artifact_bindings,
        scientific_verdict="NOT_ASSIGNED",additional_physics_controls=0,additional_actor_inference_rows=0,
        additional_fits=0,wall_seconds=time.perf_counter()-started,
        limitations=["Arithmetic and saved-state consistency only; no independent dynamics reintegration or action inference.",
            "Neural hook requests are charged even on failure/interruption; legacy post-call summaries may overlap and are reported separately.",
            "No ledger attests to calls that bypassed its instrumentation.",
            "Per-step optimizer/backward requests are charged before execution; fit_work summaries overlap and are not extra work.",
            "Legacy optimizer counts use completed fit reports and cannot attest to interrupted unreported fitting or ordinary backward counts.",
            "Raw action ownership metadata is checked; neural outputs and likelihoods are not independently recomputed.",
            "Query endpoints use explicit zero warmstart interventions, not recovered exact model solver states.",
            "This audit does not establish learning, guidance benefit, capture competence or sample efficiency."])


def audit(path, *, source_root=None, require_closed=True):
    """Audit a single archive or all archive ledgers beneath a run directory."""
    root=Path(path).resolve()
    if (root/"ledger.jsonl").is_file():
        return audit_archive(root,source_root=source_root,require_closed=require_closed)
    started=time.perf_counter();errors=[];inputs={};temperature=None;run_pending=[]
    if (root/"config.json").exists():
        data=(root/"config.json").read_bytes();inputs[str(root/"config.json")]=digest(data)
        try:
            config=json.loads(data);require(isinstance(config,dict),"Frozen config must be an object")
            require(isinstance(config.get("search",{}),dict),"Frozen search config must be an object")
            temperature=config.get("search",{}).get("temperature")
        except (ValueError,TypeError) as error:errors.append("Frozen config: "+str(error))
    else:run_pending.append("Run frozen config missing")
    if (root/"manifest.json").exists():
        data=(root/"manifest.json").read_bytes();inputs[str(root/"manifest.json")]=digest(data)
        try:
            manifest=json.loads(data);require(isinstance(manifest,dict),"Frozen manifest must be an object")
            require(isinstance(manifest.get("files",[]),list),"Frozen source inventory must be a list")
        except (ValueError,TypeError) as error:
            errors.append("Frozen manifest: "+str(error));manifest={}
        if "config_sha256" in manifest and manifest["config_sha256"]!=inputs.get(str(root/"config.json")):
            errors.append("Frozen config digest differs")
        for item in manifest.get("files",[]):
            try:
                name=PurePosixPath(item["file"]);require(not name.is_absolute() and ".." not in name.parts,"Unsafe source path")
                p=root/str(name);require(digest(p.read_bytes())==item["sha256"],"Frozen source digest differs: "+str(name))
                inputs[str(p)]=item["sha256"]
            except (ValueError,KeyError,OSError) as error:errors.append(str(error))
    else:run_pending.append("Run source manifest missing")
    source_root=Path(source_root) if source_root is not None else (root/"source" if (root/"source").is_dir() else None)
    # Backend qualification deliberately has two top-level model-only archives.
    # Do not recursively traverse unrelated parent runs or imported corpora.
    ledgers=sorted(set(root.glob("evidence/**/ledger.jsonl"))|
                   set(root.glob("*_qualification/ledger.jsonl")))
    results=[audit_archive(p.parent,source_root=source_root,allowed_root=root,require_closed=require_closed,
                           temperature=temperature) for p in ledgers]
    errors.extend(error for r in results for error in r["errors"])
    pending=run_pending+[error for r in results for error in r["pending"]]
    if not results:pending.append("No raw archive ledgers found")
    totals=Counter()
    for r in results:
        for key,value in r["work"].items():
            if type(value) is int:totals[key]+=value
    return dict(schema="bidirectional-run-arithmetic-audit-v1",status="FAIL" if errors else "PENDING" if pending else "PASS",
        arithmetic_valid=not errors,errors=errors,pending=pending,archives=results,work=dict(totals),
        input_sha256=inputs,scientific_verdict="NOT_ASSIGNED",additional_physics_controls=0,
        additional_actor_inference_rows=0,additional_fits=0,wall_seconds=time.perf_counter()-started)


audit_run=audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path",type=Path);parser.add_argument("--source-root",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();require(not args.out.exists(),"Audit output already exists")
    result=audit(args.path,source_root=args.source_root)
    with args.out.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"],output=str(args.out),wall_seconds=result["wall_seconds"])))


if __name__=="__main__":main()
