"""One prospective, finite variance intervention after the closed Ending v3.

No work runs on import. This is a learning-mechanism assay, with no planning,
deployment promotion, new discovery or sample-efficiency claim. The original
negative run is an immutable, explicitly paid ancestor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time


PARENT_CHECKPOINT_SHA256 = "9bd1d4d4b3cc46c7f2f169dd2f7dea7e48a96d1ceb672e417f58b731e402e158"
PARENT_SCHEMA = "bidirectional-learned-ending-v3-first-action-completion"
DEFAULT = dict(schema="bidirectional-ending-precision-assay-v0", task=dict(
    gate=12, delay=12, move=.2, ending_y=-.1, effort=.02, greed=.2,
    detour=.2, detour_steps=5, penalty=20.),
    arms=["original", "precision"], precision_initial_std=.01,
    variance_intervention="initial_log_std_only_once_then_trainable",
    learning=dict(lr=.0003, max_grad_norm=.5, epochs=4, clip_ratio=.2,
        aux_weight=.1, batch_size=256, bootstrap_updates=2000,
        minimum_replicates=2, return_scale=25.),
    parent_label="completion_r0_parent", qualification_roots=16,
    qualification_seed=109000, qualification_capture_min=1.,
    ordinary_contexts=8, ordinary_seed=110000, acquired_contexts=8,
    acquired_steps=[6, 12, 18, 21], replicates=4,
    evaluation_roots=64, evaluation_seed=111000, evaluation_noise_seed=112000,
    child_auxiliary_rows=200, minimum_complete_gain=.001, capture_min=.9,
    max_controls=22800, max_actor_rows=77200, max_adam_steps=112,
    max_diagnostic_backward_passes=4)


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):digest.update(block)
    return digest.hexdigest()


def _write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())


def load_config(path=None):
    config=deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    if config!=DEFAULT:raise ValueError("Only the single predeclared precision assay is accepted")
    return config


def work_bound(config=None):
    """No paid call: all acquired starts conservatively counted as T25 roots."""
    config=DEFAULT if config is None else config
    if config!=DEFAULT:raise ValueError("Undeclared precision assay")
    T=25;arms=2;children=2;own=16*4*T;steps=4*math.ceil(own/256)
    qualification=16*T;evaluation=arms*3*64*T*2
    return dict(executed_controls=arms*own+evaluation+qualification,
        model_controls=0,total_controls=arms*own+evaluation+qualification,
        actor_rows=qualification+arms*(own+children*(own+4*own+steps*200)+3*64*T*2),
        adam_steps=arms*children*steps,diagnostic_backward_passes=arms*children,
        scope="Conservative full T25 reservations; actual acquired continuations are shorter; no retry or automatic expansion")


def _safe_source(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or ".." in relative.parts:raise ValueError("Unsafe ancestor reference")
    path=root/relative
    if not path.is_file() or path.is_symlink():raise ValueError("Missing/nonregular ancestor input: "+str(path))
    return path


def input_provenance(parent_run):
    """Read/hash closed evidence only. No actor load, forward or transition."""
    root=Path(parent_run).resolve()
    files={name:_safe_source(root,name) for name in (
        "manifest.json","config.json","pilot_result.json","evidence/pilot/ledger.jsonl",
        "work_ledger.jsonl","checkpoints/completion_r0_parent.pt")}
    result=json.loads(files["pilot_result.json"].read_text())
    config=json.loads(files["config.json"].read_text())
    if result.get("schema")!=PARENT_SCHEMA or config.get("schema")!=PARENT_SCHEMA:
        raise ValueError("The exact closed v3 ancestor is required")
    if result.get("stage")!="failed_matched_child_improvement" or result.get("passed") is not False:
        raise ValueError("Retain the closed negative ancestor; do not substitute another parent")
    if config["task"]!=DEFAULT["task"] or config["learning"]!=DEFAULT["learning"]:
        raise ValueError("Ancestor task/update contract differs")
    manifest=json.loads(files["manifest.json"].read_text())
    if _sha(files["config.json"])!=manifest["config_sha256"]:
        raise ValueError("Ancestor frozen configuration changed")
    if _sha(files["manifest.json"])!=result["manifest_sha256"] or _sha(files["evidence/pilot/ledger.jsonl"])!=result["query_ledger_sha256"] or _sha(files["work_ledger.jsonl"])!=result["work_ledger_sha256"]:
        raise ValueError("Ancestor result/manifest/ledger binding changed")
    critical={}
    for relative in ("src/ending/core.py","src/gps/bidirectional/ending_pilot.py",
                     "src/gps/bidirectional/policy.py","src/gps/bidirectional/learning.py"):
        name="source/"+relative
        rows=[row for row in manifest["files"] if row["file"]==name]
        if len(rows)!=1 or _sha(_safe_source(root,name))!=rows[0]["sha256"]:
            raise ValueError("Missing/changed ancestor critical source: "+name)
        current=Path(__file__).resolve().parents[3]/relative
        if _sha(current)!=rows[0]["sha256"]:
            raise ValueError("Current actor/dynamics/observation/update contract differs from ancestor: "+relative)
        critical[relative]=rows[0]["sha256"]
    expected={f"teacher_{i:03d}":"shared" for i in range(8)}
    expected.update({f"completion_r0_acquisition_{arm}_{i:03d}":arm
                     for arm in ("raw","guided") for i in range(8)})
    outcomes={};checkpoint=None;last=None
    with files["evidence/pilot/ledger.jsonl"].open() as stream:
        for line in stream:
            row=json.loads(line);last=row
            if row.get("kind")=="episode_outcome" and row.get("episode") in expected:
                if row["episode"] in outcomes:raise ValueError("Duplicate reused episode")
                outcomes[row["episode"]]=row
            if row.get("kind")=="checkpoint" and row.get("label")==DEFAULT["parent_label"]:
                if checkpoint is not None:raise ValueError("Duplicate parent checkpoint")
                checkpoint=row
    if last is None or last.get("kind")!="archive_close" or set(outcomes)!=set(expected):
        raise ValueError("Ancestor not closed or required discovery/acquisition records missing")
    parent_sha=_sha(files["checkpoints/completion_r0_parent.pt"])
    if checkpoint is None or parent_sha!=PARENT_CHECKPOINT_SHA256 or checkpoint["sha256"]!=parent_sha:
        raise ValueError("This assay requires the exact original completion_r0_parent bytes")
    refs=[]
    for name,role in expected.items():
        row=outcomes[name];ref=row["raw"];relative="evidence/pilot/"+ref["file"]
        source=_safe_source(root,relative)
        if _sha(source)!=ref["sha256"] or source.stat().st_size!=ref["bytes"]:
            raise ValueError("Reused raw episode bytes differ: "+name)
        if not row["outcome"]["complete"] or row["owner"]["semantics"]!="executed_replanning":
            raise ValueError("Only complete actually executed teacher trajectories may be reused")
        refs.append(dict(episode=name,role=role,file=relative,sha256=ref["sha256"],
            bytes=ref["bytes"],original_event_id=row["id"],capture=bool(row["outcome"]["capture"]),
            owner=row["owner"],outcome=row["outcome"]))
    if not any(r["capture"] for r in refs if r["role"]=="shared"):
        raise ValueError("No successful independent teacher to form the fixed common stratum")
    return dict(schema="ending-precision-input-provenance-v1",parent_run=str(root),
        parent_checkpoint_sha256=parent_sha,parent_checkpoint_event=checkpoint,
        files=[dict(file=name,sha256=_sha(path),bytes=path.stat().st_size) for name,path in files.items()],
        episodes=refs,critical_source_sha256=critical,
        reused_cost_ancestry=result["accounting"],reused_wall_seconds=result["pilot_wall_seconds"],
        arithmetic_audit="Not required for this source-binding step; independent prior arithmetic audit is separate",
        cost_relationship="One shared paid ancestor, fully attributed once to each arm's standalone cost; never sum the common ancestor twice as unique work",
        efficiency_claim=False,old_heldout_teacher_labels_used=False)


def snapshot_inputs(parent_run,out):
    """Create-only portable copies of the exact selected parent/teacher inputs."""
    provenance=input_provenance(parent_run);root=Path(provenance["parent_run"]);out=Path(out)
    target=out/"inputs";target.mkdir(exist_ok=False)
    for row in provenance["files"]+provenance["episodes"]:
        relative=row["file"];destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        source=_safe_source(root,relative)
        with source.open("rb") as src,destination.open("xb") as dst:
            for block in iter(lambda:src.read(1024*1024),b""):dst.write(block)
            dst.flush();os.fsync(dst.fileno())
        if _sha(destination)!=row["sha256"] or destination.stat().st_size!=row["bytes"]:
            raise ValueError("Ancestor changed while copying; preserve failed setup")
    _write(out/"input_provenance.json",provenance)
    return provenance


def verify_inputs(out):
    out=Path(out);provenance=json.loads((out/"input_provenance.json").read_text())
    if provenance["parent_checkpoint_sha256"]!=PARENT_CHECKPOINT_SHA256:
        raise ValueError("Wrong fixed parent input")
    for row in provenance["files"]+provenance["episodes"]:
        path=_safe_source(out/"inputs",row["file"])
        if _sha(path)!=row["sha256"] or path.stat().st_size!=row["bytes"]:
            raise ValueError("Frozen ancestor input changed")
    return provenance


def _load_teachers(out,provenance):
    import numpy as np
    groups={name:[] for name in ("shared","raw","guided")}
    for row in provenance["episodes"]:
        with np.load(Path(out)/"inputs"/row["file"],allow_pickle=False) as raw:
            arrays={name:raw[name].copy() for name in ("states","observations","actions","raw_costs")}
            owner=json.loads(str(raw["owner_json"]));captured=bool(raw["captured"])
            if not bool(raw["complete"]) or captured!=row["capture"] or owner!=row["owner"]:
                raise ValueError("Reused raw teacher and source outcome disagree")
        if arrays["states"].shape!=(26,2) or arrays["observations"].shape!=(26,8) or arrays["actions"].shape!=(25,1) or arrays["raw_costs"].shape!=(25,) or not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Reused teacher must be an aligned complete finite T25 trajectory")
        if not np.array_equal(arrays["states"][:,0],np.arange(26)):
            raise ValueError("Reused teacher clock is incomplete")
        groups[row["role"]].append(dict(**arrays,successful=captured,
            raw=dict(file=str(Path(out)/"inputs"/row["file"]),sha256=row["sha256"],bytes=row["bytes"]),
            ancestry=dict(parent_run=provenance["parent_run"],episode=row["episode"],event=row["original_event_id"])))
    return groups


def _comparison(base,proposed,minimum):
    import numpy as np
    if [r.context_id for r in base]!=[r.context_id for r in proposed]:raise ValueError("Unpaired complete outcomes")
    gains=[a.cost-b.cost for a,b in zip(base,proposed)]
    nonregression=sum(r.capture for r in proposed)>=sum(r.capture for r in base)
    return dict(mean_cost_gain=float(np.mean(gains)),paired_cost_gains=gains,
        capture_nonregression=bool(nonregression),improved=bool(nonregression and np.mean(gains)>=minimum))


def _summary(rows):
    import numpy as np
    return dict(capture_rate=float(np.mean([r.capture for r in rows])),
        mean_cost=float(np.mean([r.cost for r in rows])),outcomes=[asdict(r) for r in rows])


def configure_arm(parent,arm):
    """Clone all learner state, then perform only the declared initial variance change."""
    import torch
    from .feedback import _digest
    if arm not in DEFAULT["arms"]:raise ValueError("Undeclared variance arm")
    learner=parent.clone_for_trial();original=learner.state_dict()
    initial_std=learner.actor.log_std.detach().exp().cpu().numpy().copy()
    if arm=="precision":
        with torch.no_grad():learner.actor.log_std.fill_(math.log(DEFAULT["precision_initial_std"]))
    state=learner.state_dict()
    changed=[name for name in original["policy"]["actor"]
        if not torch.equal(original["policy"]["actor"][name],state["policy"]["actor"][name])]
    if changed!=([] if arm=="original" else ["log_std"]) or any(
            _digest(original[k])!=_digest(state[k]) for k in ("optimizer","numpy_rng","torch_rng","updates")):
        raise RuntimeError("Variance intervention changed an undeclared parameter/state")
    if not learner.actor.log_std.requires_grad:raise RuntimeError("The initial precision variance must remain trainable")
    return learner,dict(original_log_std_exp=initial_std.tolist(),
        initial_log_std_exp=learner.actor.log_std.detach().exp().cpu().numpy().tolist(),
        changed_parameters=changed,log_std_trainable=True,
        intervention=DEFAULT["variance_intervention"] if arm=="precision" else "none")


def comparison_passes(deterministic,stochastic,minimum=.001,capture_min=.9):
    """Both modes must improve against BOTH baselines, with capture nonregression."""
    benefit={name:_comparison(deterministic[name],deterministic["guided"],minimum) for name in ("parent","raw")}
    stochastic_benefit={name:_comparison(stochastic[name],stochastic["guided"],minimum) for name in ("parent","raw")}
    capture=sum(r.capture for r in deterministic["guided"])/len(deterministic["guided"])
    return bool(all(c["improved"] for c in list(benefit.values())+list(stochastic_benefit.values())) and capture>=capture_min),benefit,stochastic_benefit


def run_assay(config,out):
    """Run only frozen source; all actual scalar/forward/backward/update work metered."""
    config=deepcopy(config)
    if config!=DEFAULT:raise ValueError("Undeclared precision assay")
    bound=work_bound(config)
    if (bound["total_controls"],bound["actor_rows"],bound["adam_steps"],bound["diagnostic_backward_passes"])!=(config["max_controls"],config["max_actor_rows"],config["max_adam_steps"],config["max_diagnostic_backward_passes"]):
        raise ValueError("Fixed schedule and declared work caps disagree")
    out=Path(out).resolve();manifest=json.loads((out/"manifest.json").read_text())
    if not Path(__file__).resolve().is_relative_to(out/"source") or _sha(out/"config.json")!=manifest["config_sha256"] or any(_sha(out/r["file"])!=r["sha256"] for r in manifest["files"]):
        raise ValueError("Run the immutable frozen assay source/config")
    provenance=verify_inputs(out)
    import numpy as np
    import torch
    from scripts.diagnostics.run_bidirectional_tests import DurableWork,WorkPlugin
    from .evidence import Archive
    from .policy import FamilyGaussianActor,policy_hash
    from .learning import PolicyLearner,LearningConfig
    from .meter import attach_meter
    from .feedback import _digest
    from .ending_pilot import CountedEnding,EndingContext,EndingExecutor,EndingSnapshot,_episode,_actor_panel
    from .completion_pilot import teaching_auxiliary
    from .workflow import checkpoint
    started=time.perf_counter();archive=Archive(out/"evidence"/"assay")
    archive.work_scope="setup";archive.budget_failed=False;archive.query_failed=False
    work=DurableWork(out/"work_ledger.jsonl");plugin=WorkPlugin(work)
    result=dict(schema=config["schema"],passed=False,stage="setup",arms={},negative_retained=True,
        no_planner_guidance_deployed=True,efficiency_claim=False,full_feedback_established=False)
    actor_rows=0;adam_attempts=0;diagnostic_attempts=0;dynamics=None;installed=False
    def phase(name):archive.work_scope=name;work.current_test=name;archive.append("assay_phase",phase=name)
    original_before=plugin._before
    def bounded_before(module,args):
        nonlocal actor_rows
        if isinstance(module,FamilyGaussianActor):
            shape=np.shape(args[0]);rows=int(np.prod(shape[:-1])) if len(shape)>1 else 1
            if actor_rows+rows>config["max_actor_rows"]:
                archive.budget_failed=True;work.append("neural_rejected",rows=rows,reason="fixed_actor_row_cap",before_forward=True)
                raise RuntimeError("Fixed actor-row cap exhausted before forward")
            actor_rows+=rows
        original_before(module,args)
    plugin._before=bounded_before
    try:
        plugin.install();installed=True;global_adam=torch.optim.Adam.step
        def bounded_adam(optimizer,*args,**kwargs):
            nonlocal adam_attempts
            if adam_attempts>=config["max_adam_steps"]:
                archive.budget_failed=True;work.append("optimizer_rejected",updates=1,reason="fixed_Adam_step_cap",before_optimizer=True)
                raise RuntimeError("Fixed Adam-step cap exhausted before optimizer")
            adam_attempts+=1;return global_adam(optimizer,*args,**kwargs)
        torch.optim.Adam.step=bounded_adam
        global_grad=torch.autograd.grad
        def bounded_grad(*args,**kwargs):
            nonlocal diagnostic_attempts
            if diagnostic_attempts>=config["max_diagnostic_backward_passes"]:
                archive.budget_failed=True;work.append("autograd_grad_rejected",calls=1,
                    reason="fixed_diagnostic_backward_cap",before_backward=True)
                raise RuntimeError("Fixed diagnostic backward cap exhausted before differentiation")
            diagnostic_attempts+=1;return global_grad(*args,**kwargs)
        torch.autograd.grad=bounded_grad
        dynamics=CountedEnding(archive,work,config["max_controls"],**config["task"])
        executor=EndingExecutor(EndingContext(dynamics),archive)
        groups=_load_teachers(out,provenance);shared=[t for t in groups["shared"] if t["successful"]]
        # Trusted generated checkpoint, verified by its fixed original SHA before unpickling.
        state=torch.load(out/"inputs/checkpoints/completion_r0_parent.pt",map_location="cpu",weights_only=False)
        parent_actor=FamilyGaussianActor.from_checkpoint(state["learner"]["policy"])
        parent=PolicyLearner(parent_actor,LearningConfig(**config["learning"]))
        parent.load_state_dict(state["learner"]);parent_digest=_digest(parent.state_dict())
        archive.append("input_reuse",provenance_sha256=_sha(out/"input_provenance.json"),
            exact_parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,parent_state_sha256=parent_digest,
            paid_ancestry=provenance["reused_cost_ancestry"],efficiency_claim=False)
        checkpoint(out,"precision_source_parent",dict(learner=parent.state_dict(),
            original_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,promotion_claim=False),archive)
        attach_meter(parent.actor,archive,"fresh_parent_mean_qualification")
        phase("fresh_parent_mean_qualification")
        qualified=_actor_panel(executor,parent.actor,range(config["qualification_seed"],config["qualification_seed"]+16),"precision_parent_qualification")
        result["parent_qualification"]=_summary(qualified)
        if result["parent_qualification"]["capture_rate"]<config["qualification_capture_min"]:
            result["stage"]="failed_fresh_parent_competence"
        else:
            # Both arms start with exactly the same saved sampling stream.
            stream_start=parent.torch_generator.get_state().clone()
            arm_generators=[]
            for arm in config["arms"]:
                prefix="precision_"+arm;learner,intervention=configure_arm(parent,arm)
                arm_state=learner.state_dict();arm_digest=_digest(arm_state);ah=policy_hash(learner.actor)
                entry=dict(parent_actor_hash=ah,parent_state_sha256=arm_digest,**intervention,
                    shared_ancestry_attributed_once=provenance["reused_cost_ancestry"])
                result["arms"][arm]=entry
                checkpoint(out,prefix+"_parent",dict(learner=arm_state,source_parent_state_sha256=parent_digest,
                    intervention=entry["intervention"],promotion_claim=False),archive)
                attach_meter(learner.actor,archive,prefix+"_owned_mc")
                phase(prefix+"_owned_mc");contexts=[executor.task.reset(config["ordinary_seed"]+i) for i in range(8)]
                for i in range(8):
                    trace=shared[i%len(shared)];t=config["acquired_steps"][i%4]
                    contexts.append(EndingSnapshot(trace["states"][t].copy(),t,f"precision_acquired:{i}:{t}"))
                context_raw=archive.save_npz(f"assay/{prefix}_contexts.npz",states=np.stack([c.state for c in contexts]),
                    context_ids=np.array([c.context_id for c in contexts]),sampling_rng_start=stream_start.numpy())
                sample_generator=deepcopy(learner.torch_generator)
                if not torch.equal(sample_generator.get_state(),stream_start):raise RuntimeError("Different arm sampling stream")
                episodes=[]
                for i,initial in enumerate(contexts):
                    for j in range(4):
                        _,_,ep,_=_episode(executor,initial,f"{prefix}_own_c{i:02d}_r{j}",actor=learner.actor,
                            deterministic=False,generator=sample_generator);episodes.append(ep)
                if _digest(learner.state_dict())!=arm_digest:raise RuntimeError("Own collection changed immutable arm parent")
                stream_end=sample_generator.get_state().clone();arm_generators.append(stream_end)
                stream_raw=archive.save_npz(f"assay/{prefix}_noise_stream.npz",start=stream_start.numpy(),end=stream_end.numpy())
                behavior_digest=_digest(episodes);children={name:learner.clone_for_trial() for name in ("raw","guided")}
                if any(_digest(child.state_dict())!=arm_digest for child in children.values()):raise RuntimeError("Unmatched children")
                aux={};indices={}
                for name in children:aux[name],indices[name]=teaching_auxiliary(shared,groups[name],200)
                if not np.array_equal(aux["raw"].observations[:100],aux["guided"].observations[:100]) or not np.array_equal(aux["raw"].actions[:100],aux["guided"].actions[:100]):raise RuntimeError("Different shared teacher stratum")
                aux_raw=archive.save_npz(f"assay/{prefix}_matched_aux.npz",**{f"{name}_{field}":getattr(batch,field)
                    for name,batch in aux.items() for field in ("observations","actions","families")},
                    **{f"{name}_{stratum}":value for name,details in indices.items() for stratum,value in details.items()})
                phase(prefix+"_child_fit");learning={}
                for name,child in children.items():
                    attach_meter(child.actor,archive,prefix+"_child_"+name)
                    learning[name]=child.update(episodes,ah,aux=aux[name])
                    archive.append("fit_work",purpose=prefix+"_matched_child_"+name,**learning[name])
                    if _digest(episodes)!=behavior_digest:raise RuntimeError("Own behavior changed between children")
                    checkpoint(out,prefix+"_child_"+name,dict(learner=child.state_dict(),parent_actor_hash=ah,
                        parent_state_sha256=arm_digest,behavior_sha256=behavior_digest,learning=learning[name],
                        auxiliary_raw=aux_raw,shared_teacher_refs=[t["raw"] for t in shared],
                        new_teacher_refs=[t["raw"] for t in groups[name]],promotion_claim=False),archive)
                if learning["raw"]["gradient_steps"]!=learning["guided"]["gradient_steps"]:raise RuntimeError("Unequal child optimizer schedule")
                phase(prefix+"_fresh_evaluation");seeds=range(config["evaluation_seed"],config["evaluation_seed"]+64)
                actors={"parent":learner.actor,**{n:c.actor for n,c in children.items()}}
                panels={name:_actor_panel(executor,actor,seeds,prefix+"_eval_"+name) for name,actor in actors.items()}
                stochastic={name:_actor_panel(executor,actor,seeds,prefix+"_stochastic_"+name,
                    deterministic=False,stochastic_seed=config["evaluation_noise_seed"]) for name,actor in actors.items()}
                passed,benefit,stochastic_benefit=comparison_passes(panels,stochastic,
                    config["minimum_complete_gain"],config["capture_min"])
                entry.update(passed=bool(passed),context_raw=context_raw,noise_stream_raw=stream_raw,
                    behavior_sha256=behavior_digest,own_episodes=len(episodes),own_actual_controls=sum(len(ep.raw_costs) for ep in episodes),
                    auxiliary_raw=aux_raw,auxiliary_strata={k:{f:(v.tolist() if hasattr(v,"tolist") else v) for f,v in d.items()} for k,d in indices.items()},
                    learning=learning,panels={n:_summary(r) for n,r in panels.items()},
                    stochastic_panels={n:_summary(r) for n,r in stochastic.items()},
                    guided_vs=benefit,stochastic_guided_vs=stochastic_benefit,
                    children_final_std={n:c.actor.log_std.detach().exp().cpu().numpy().tolist() for n,c in children.items()},
                    evaluation_role="Fresh assay end outcomes; no actor/controller selection, no further update or deployment")
                archive.append("precision_arm_result",arm=arm,result=entry)
            if len(arm_generators)!=2 or not torch.equal(*arm_generators):raise RuntimeError("Arm noise streams consumed differently")
            result.update(passed=result["arms"]["precision"]["passed"],
                stage="complete" if result["arms"]["precision"]["passed"] else "failed_precision_child_improvement",
                matched_underlying_noise_stream=True,variance_specific_evidence=bool(result["arms"]["precision"]["passed"] and not result["arms"]["original"]["passed"]))
            result["negative_retained"]=not result["passed"]
        if _digest(parent.state_dict())!=parent_digest:raise RuntimeError("Assay changed the original frozen parent")
    except BaseException as error:
        result.update(passed=False,stage="execution_error",error_type=type(error).__name__,error=str(error),negative_retained=True)
        archive.append("assay_error",error_type=type(error).__name__,error=str(error))
    finally:
        if installed:plugin.uninstall()
        work.close();archive.close()
    events=[json.loads(line) for line in work.path.read_text().splitlines()]
    local=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
    result["accounting"]=dict(requested_scalar_controls=sum(e["controls"] for e in events if e["kind"]=="scalar_transition_requested"),
        completed_scalar_controls=sum(e["controls"] for e in events if e["kind"]=="scalar_transition_complete"),
        reserved_query_controls=0 if dynamics is None else dynamics.reserved,
        actor_rows=sum(e["rows"] for e in events if e["kind"]=="neural_requested"),
        actor_calls=sum(e["kind"]=="neural_requested" for e in events),
        adam_attempts=sum(e["kind"]=="optimizer_update_requested" for e in events),
        adam_completed=sum(e["kind"]=="optimizer_update_complete" for e in events),
        diagnostic_backward_attempts=sum(e.get("passes",0) for e in local if e["kind"]=="diagnostic_backward_requested"),
        gradient_backward_attempts=sum(e.get("passes",0) for e in local if e["kind"]=="gradient_backward_requested"),
        global_autograd_grad_calls=sum(e["kind"]=="autograd_grad_requested" for e in events),
        global_autograd_backward_calls=sum(e["kind"]=="autograd_backward_requested" for e in events),
        new_mujoco_controls=0,new_warp_controls=0,model_controls=0,query_totals=archive.totals(),
        overlap="Global hooks and local learner events observe the same work; never add their totals")
    def scoped_work(prefix):
        subset=[e for e in events if str(e.get("test","")).startswith(prefix)]
        return dict(requested_scalar_controls=sum(e["controls"] for e in subset if e["kind"]=="scalar_transition_requested"),
            actor_rows=sum(e["rows"] for e in subset if e["kind"]=="neural_requested"),
            adam_attempts=sum(e["kind"]=="optimizer_update_requested" for e in subset),
            diagnostic_backward_attempts=sum(e["kind"]=="autograd_grad_requested" for e in subset))
    shared_work=scoped_work("fresh_parent_mean_qualification")
    arm_work={arm:scoped_work("precision_"+arm) for arm in config["arms"]}
    ancestor=provenance["reused_cost_ancestry"]
    for arm,paid in arm_work.items():
        if arm in result["arms"]:result["arms"][arm]["costs"]=dict(new_own_arm=paid,
            new_shared_parent_qualification=shared_work,standalone_with_shared_ancestry=dict(
                requested_scalar_controls=ancestor["requested_scalar_controls"]+shared_work["requested_scalar_controls"]+paid["requested_scalar_controls"],
                actor_rows=ancestor["actor_rows"]+shared_work["actor_rows"]+paid["actor_rows"],
                adam_attempts=ancestor["adam_attempts"]+paid["adam_attempts"],
                diagnostic_backward_attempts=ancestor["diagnostic_backward_attempts"]+paid["diagnostic_backward_attempts"]))
    result["cost_crosswalk"]=dict(shared_parent_qualification=shared_work,new_by_arm=arm_work,
        unique_requested_scalar_controls_including_reused_ancestor_once=ancestor["requested_scalar_controls"]+result["accounting"]["requested_scalar_controls"],
        unique_actor_rows_including_reused_ancestor_once=ancestor["actor_rows"]+result["accounting"]["actor_rows"],
        unique_adam_attempts_including_reused_ancestor_once=ancestor["adam_attempts"]+result["accounting"]["adam_attempts"],
        relationship="Shared discovery/old qualification/acquisition remain paid once in unique lineage, once per standalone arm attribution; arm attributions overlap and must not be summed")
    unchanged=all(_sha(out/r["file"])==r["sha256"] for r in manifest["files"])
    if not unchanged:result.update(passed=False,stage="source_changed",negative_retained=True)
    result.update(config_sha256=_sha(out/"config.json"),manifest_sha256=_sha(out/"manifest.json"),
        input_provenance_sha256=_sha(out/"input_provenance.json"),source_unchanged=unchanged,
        query_ledger_sha256=_sha(archive.ledger_path),work_ledger_sha256=_sha(work.path),
        assay_wall_seconds=time.perf_counter()-started,declared_work_bound=bound,
        reused_ancestry=provenance["reused_cost_ancestry"],reused_ancestry_wall_seconds=provenance["reused_wall_seconds"],
        scope="One cheap Ending precision-learning assay; no new exploration, efficiency, physical-task, robust variance remedy or full-feedback success claim",
        no_retuning=True,old_negative_unchanged=True)
    _write(out/"assay_result.json",result)
    return result
