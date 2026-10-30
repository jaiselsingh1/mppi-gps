"""Prospective finite source-only development of coherent head-return learning.

No work on import. Exact old states and weak bootstrap are paid ancestry.
Every new parameter proposal owns every newly measured whole continuation.
This one learner assay is not two-round feedback or a PPO comparison.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import os
import time

from .interpolation_assay import _sha,_write,_safe_source,_file_ref,_comparison
from .early_context_loop import BOOTSTRAP_SHA,BOOTSTRAP_ACTOR

CHAINS=("ordinary","raw025","raw1","long025","long1")
SOURCES=dict(ordinary=("ordinary",0.),raw025=("raw",.25),raw1=("raw",1.),long025=("long",.25),long1=("long",1.))
DEFAULT=dict(schema="bidirectional-ending-coherent-head-assay-v1",
    task=dict(gate=12,delay=12,move=.2,ending_y=-.1,effort=.02,greed=.2,detour=.2,detour_steps=5,penalty=20.),
    chains=list(CHAINS),search_steps=3,candidates_per_step=13,basis_seed=1701,
    head_weight_directions=4,head_step=.1,log_std_step=.5,coefficient_l1_bound=3.,
    data_weights=[0.,.25,1.],controller_mode_weights=dict(mean=.5,stochastic=.5),
    ordinary_training_roots=8,ordinary_training_seed=601000,training_noise_seed=603000,
    acquired_elapsed=6,source_contexts_per_role=8,all_paid_roles=["ordinary","raw","long"],
    ordinary_step_constraint="complete_mean_and_stochastic_cost_and_capture_nonregression_vs_current_center",
    qualification_roots=16,qualification_seed=611000,qualification_noise_seed=613000,
    evaluation_roots=64,evaluation_seed=612000,evaluation_noise_seed=614000,
    minimum_complete_gain=.001,capture_min=.9,max_controls=210960,max_executed_controls=210960,
    max_model_controls=0,max_actor_rows=210960,max_adam_steps=0,max_diagnostic_backward_passes=0)


def load_config(path=None):
    config=deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    if config!=DEFAULT:raise ValueError("Only the declared coherent-head v1 schedule is accepted")
    return config


def work_bound(config=None):
    config=DEFAULT if config is None else config
    if config!=DEFAULT:raise ValueError("Undeclared coherent-head schedule")
    each=2*(8*25+16*19);train=5*3*13*each;qual=6*16*25*2;final=3*64*25*2
    return dict(training_controls=train,qualification_controls=qual,final_controls=final,
        total_controls=train+qual+final,executed_controls=train+qual+final,actor_rows=train+qual+final,
        model_controls=0,adam_steps=0,diagnostic_backward_passes=0,
        training_actor_versions=195,training_episodes=195*24*2,qualification_episodes=6*16*2,final_episodes=3*64*2,
        scope="Every candidate pays all24contextsbothmodes; identical versions are queried anew, correlated repeats not independent evidence; no retry/extra radius")


def input_provenance(early_run):
    root=Path(early_run).resolve();result=json.loads((root/"loop_result.json").read_text());audit=json.loads((root/"audit_001.json").read_text())
    if (audit.get("status")!="PASS" or not audit.get("evidence_arithmetic_valid") or audit.get("scientific_success") is not False
        or result["schema"]!="bidirectional-ending-early-context-loop-v5"
        or result["stage"]!="failed_initial_data_all_actor_qualification" or result["passed"] is not False):
        raise ValueError("Require the closed independently audited v5 negative")
    names=["manifest.json","config.json","loop_result.json","audit_001.json","input_provenance.json",
        "evidence/loop/ledger.jsonl","work_ledger.jsonl","inputs/checkpoints/bootstrap.pt"]
    names += [f"evidence/loop/episodes/early_initial_{role}_{i:03d}.npz" for role in ("short","long") for i in range(8)]
    refs=[_file_ref(root,n) for n in names];by={r["file"]:r for r in refs}
    for name in names:
        if name!="audit_001.json" and by[name]["sha256"]!=audit["input_sha256"].get(name):
            raise ValueError("Old audit binding changed: "+name)
    if by["inputs/checkpoints/bootstrap.pt"]["sha256"]!=BOOTSTRAP_SHA:raise ValueError("Wrong weak bootstrap")
    for name,key in (("manifest.json","manifest_sha256"),("config.json","config_sha256"),
        ("evidence/loop/ledger.jsonl","query_ledger_sha256"),("work_ledger.jsonl","work_ledger_sha256"),
        ("input_provenance.json","input_provenance_sha256")):
        if result[key]!=by[name]["sha256"]:raise ValueError("Closed ancestor result binding changed")
    manifest=json.loads((root/"manifest.json").read_text());critical={}
    for relative in ("src/ending/core.py","src/gps/bidirectional/ending_pilot.py","src/gps/bidirectional/policy.py"):
        old=next(r["sha256"] for r in manifest["files"] if r["file"]=="source/"+relative)
        if _sha(Path(__file__).resolve().parents[3]/relative)!=old:raise ValueError("Core actor/task/observation source changed")
        critical[relative]=old
    selected={f"early_initial_{role}_{i:03d}" for role in ("short","long") for i in range(8)};events={};last=None
    with (root/"evidence/loop/ledger.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);last=row
            if row["kind"]=="episode_outcome" and row.get("episode") in selected:
                if row["episode"] in events:raise ValueError("Duplicated source episode")
                events[row["episode"]]=row
    if last["kind"]!="archive_close" or set(events)!=selected:raise ValueError("Unclosed/incomplete source archive")
    return dict(schema="ending-coherent-head-input-provenance-v1",original_early_run=str(root),files=refs,
        bootstrap_sha256=BOOTSTRAP_SHA,bootstrap_actor_hash=BOOTSTRAP_ACTOR,
        source_events={role:[events[f"early_initial_{tag}_{i:03d}"] for i in range(8)] for role,tag in (("raw","short"),("long","long"))},
        critical_source_sha256=critical,reused_early_ancestry=result["accounting"],
        reused_v3_ancestry=result["cost_crosswalk"]["v3_ancestor"],
        reused_wall_seconds=result["loop_wall_seconds"]+result["reused_wall_seconds"],
        source_role="Only recorded e6 states; no teacher/actor action labels or prior successful actor initialization")


def snapshot_inputs(early_run,out):
    provenance=input_provenance(early_run);target=Path(out)/"inputs";target.mkdir(exist_ok=False)
    for ref in provenance["files"]:
        source=_safe_source(provenance["original_early_run"],ref["file"]);dest=target/ref["file"]
        dest.parent.mkdir(parents=True,exist_ok=True)
        with source.open("rb") as src,dest.open("xb") as dst:
            for block in iter(lambda:src.read(1024*1024),b""):dst.write(block)
            dst.flush();os.fsync(dst.fileno())
        if _sha(dest)!=ref["sha256"]:raise ValueError("Input changed during copy")
    _write(Path(out)/"input_provenance.json",provenance);return provenance


def verify_inputs(out):
    out=Path(out);p=json.loads((out/"input_provenance.json").read_text())
    if p["bootstrap_sha256"]!=BOOTSTRAP_SHA or p["bootstrap_actor_hash"]!=BOOTSTRAP_ACTOR:raise ValueError("Wrong source actor")
    for ref in p["files"]:
        path=_safe_source(out/"inputs",ref["file"])
        if _sha(path)!=ref["sha256"] or path.stat().st_size!=ref["bytes"]:raise ValueError("Frozen ancestor changed")
    return p


def assess(mean,noisy,proposed,references):
    criteria={mode:{name:_comparison(values[name],values[proposed],.001,det) for name in references}
        for mode,values,det in (("mean",mean,True),("stochastic",noisy,False))}
    rate=sum(r.capture for r in mean[proposed])/len(mean[proposed])
    return dict(comparisons=criteria,deterministic_capture_rate=rate,
        passed=bool(rate>=.9 and all(v["improved"] for row in criteria.values() for v in row.values())))


def choose_controls(mean,noisy):
    """Controls and data weights selected only from their actual qualification."""
    controls=("ordinary","raw025","raw1")
    eligible=[name for name in controls if assess(mean,noisy,name,("parent",))["passed"]]
    best="parent" if not eligible else min(eligible,key=lambda n:(sum(o.cost for o in mean[n])/len(mean[n]),controls.index(n)))
    long=[name for name in ("long025","long1") if assess(mean,noisy,name,("parent",best))["passed"]]
    selected=None if not long else min(long,key=lambda n:(sum(o.cost for o in mean[n])/len(mean[n]),("long025","long1").index(n)))
    return dict(best_control=best,selected_long=selected,control_parent_fallback=best=="parent",
        control_criteria={n:assess(mean,noisy,n,("parent",)) for n in controls},
        long_criteria={n:assess(mean,noisy,n,("parent",best)) for n in ("long025","long1")})


def run_assay(config,out):
    config=deepcopy(config)
    if config!=DEFAULT:raise ValueError("Undeclared coherent-head assay")
    bound=work_bound(config)
    if any(config[k]!=bound[v] for k,v in (("max_controls","total_controls"),("max_executed_controls","executed_controls"),
        ("max_actor_rows","actor_rows"),("max_model_controls","model_controls"),("max_adam_steps","adam_steps"))):
        raise ValueError("Before-call caps differ from the fixed schedule")
    out=Path(out).resolve();manifest=json.loads((out/"manifest.json").read_text())
    if (not Path(__file__).resolve().is_relative_to(out/"source") or _sha(out/"config.json")!=manifest["config_sha256"]
        or any(_sha(out/r["file"])!=r["sha256"] for r in manifest["files"])):
        raise ValueError("Run only immutable frozen source/config")
    provenance=verify_inputs(out)
    import numpy as np
    import torch
    from scripts.diagnostics.run_bidirectional_tests import DurableWork,WorkPlugin
    from .evidence import Archive
    from .ending_pilot import CountedEnding,EndingContext,EndingExecutor,EndingSnapshot,_episode,_actor_panel
    from .policy import FamilyGaussianActor,policy_hash
    from .meter import attach_meter
    from .workflow import checkpoint
    from .feedback import _digest
    from .completion_pilot import _summary
    from .coherent_return import make_subspace,coordinate_bank,actor_from_coordinates,score_candidate,selector_view,select_candidate
    started=time.perf_counter();archive=Archive(out/"evidence"/"assay");archive.work_scope="setup"
    archive.budget_failed=False;archive.query_failed=False
    work=DurableWork(out/"work_ledger.jsonl");plugin=WorkPlugin(work);installed=False;actor_rows=0;dynamics=None
    result=dict(schema=config["schema"],passed=False,stage="setup",negative_retained=True,chains=[],
        no_new_action_labels=True,critic_used=False,gradient_optimizer_used=False,parameter_return_search_used=True,
        full_feedback_established=False,efficiency_claim=False,ppo_comparison=False,
        prior_bootstrap_and_PPO_and_MPPI_ancestry_paid=True)
    def phase(name):archive.work_scope=name;work.current_test=name;archive.append("coherent_phase",phase=name)
    original_before=plugin._before
    def before(module,args):
        nonlocal actor_rows
        if isinstance(module,FamilyGaussianActor):
            shape=np.shape(args[0]);n=int(np.prod(shape[:-1])) if len(shape)>1 else 1
            if actor_rows+n>config["max_actor_rows"]:
                archive.budget_failed=True;work.append("neural_rejected",rows=n,before_forward=True);raise RuntimeError("Actor-row cap")
            actor_rows+=n
        return original_before(module,args)
    plugin._before=before
    class LiveEnding(CountedEnding):
        def _begin(self,state,actions,kind,metadata):
            if np.ndim(actions)==3:
                archive.budget_failed=True;archive.append("query_rejected",query_kind=kind,reason="no_model_calls_in_coherent_assay",before_physics=True)
                raise RuntimeError("Model query forbidden")
            return super()._begin(state,actions,kind,metadata)
    def evaluate(actor,seeds,tag,det,noise):
        attach_meter(actor,archive,tag)
        return _actor_panel(executor,actor,seeds,tag,deterministic=det,stochastic_seed=noise)
    def source_panel(actor,contexts,tag,det,noise):
        generator=torch.Generator().manual_seed(noise);rng_before=generator.get_state().clone();owner=policy_hash(actor)
        attach_meter(actor,archive,tag);outcomes=[];names=[];first_actions=[];raw_refs=[]
        for i,snapshot in enumerate(contexts):
            name=f"{tag}_{i:03d}";value,trace,_,_=_episode(executor,snapshot,name,actor=actor,deterministic=det,generator=generator)
            outcomes.append(value);names.append(name);first_actions.append(float(trace["actions"][0,0]));raw_refs.append(trace["raw"])
        ref=archive.save_npz("sampling/"+tag+".npz",before=rng_before.numpy(),after=generator.get_state().numpy(),
            seed=np.array(noise),actor_hash=np.array(owner),deterministic=np.array(det))
        archive.append("coherent_panel",panel_id=tag,actor_hash=owner,deterministic=det,episode_names=names,
            outcomes=[asdict(r) for r in outcomes],sampling_rng=ref,raw_refs=raw_refs,
            first_actions=first_actions,initial_states=[s.state.tolist() for s in contexts])
        return outcomes,np.asarray(first_actions)
    try:
        plugin.install();installed=True
        def no_adam(*args,**kwargs):
            archive.budget_failed=True;work.append("optimizer_rejected",updates=1,before_optimizer=True);raise RuntimeError("No optimizer in coherent assay")
        def no_grad(*args,**kwargs):
            archive.budget_failed=True;work.append("autograd_grad_rejected",calls=1,before_backward=True);raise RuntimeError("No backward in coherent assay")
        torch.optim.Adam.step=no_adam;torch.autograd.grad=no_grad
        dynamics=LiveEnding(archive,work,config["max_controls"],**config["task"])
        executor=EndingExecutor(EndingContext(dynamics),archive)
        saved=torch.load(out/"inputs/inputs/checkpoints/bootstrap.pt",map_location="cpu",weights_only=False)
        base=FamilyGaussianActor.from_checkpoint(saved["learner"]["policy"])
        if policy_hash(base)!=BOOTSTRAP_ACTOR:raise ValueError("Wrong exact starting weak actor")
        base_digest=_digest(base.policy_state())
        checkpoint(out,"coherent_source_parent",dict(actor=base.policy_state(),source_checkpoint_sha256=BOOTSTRAP_SHA,
            source_full_learner_sha256=_digest(saved["learner"]),promotion_claim=False),archive)
        subspace=make_subspace(1701)
        basis=archive.save_npz("coherent/basis.npz",weight_directions=subspace.weight_directions,seed=np.array(1701),
            head_step=np.array(.1),log_std_step=np.array(.5),base_actor_hash=np.array(BOOTSTRAP_ACTOR))
        result["subspace"]=dict(raw=basis,dimensions=6,frozen_trunk=True,coordinates=["bias","W0","W1","W2","W3","log_std"],
            parameter_construction="original_float32_parent_plus_absolute_float64_coordinate_change_then_cast_original_dtype")
        acquired={};source_refs={}
        for role in ("raw","long"):
            acquired[role]=[];source_refs[role]=[]
            for i,event in enumerate(provenance["source_events"][role]):
                path=out/"inputs/evidence/loop"/event["raw"]["file"]
                with np.load(path,allow_pickle=False) as z:state=z["states"][6].copy();qids=z["query_ids"].tolist()
                context=f"coherent_source_{role}:e6:{i}";acquired[role].append(EndingSnapshot(state,6,context))
                source_refs[role].append(dict(context_id=context,state=state.tolist(),elapsed=6,source_state_index=6,
                    source_episode=event["episode"],source_outcome=event["outcome"],source_query_id=qids[5],
                    source_raw=dict(file=str(path.relative_to(out)),sha256=_sha(path),bytes=path.stat().st_size)))
        result["source_contexts"]=source_refs;archive.append("coherent_source_contexts",sources=source_refs,labels_are_actions=False)
        finals={}
        for chain in CHAINS:
            source,beta=SOURCES[chain];coordinates=np.zeros(6);entry=dict(chain=chain,source=source,beta=beta,steps=[])
            result["chains"].append(entry)
            for step in range(3):
                prefix=f"coherent_{chain}_s{step}";phase(prefix+"_proposals")
                iteration_request=archive.append("coherent_search_iteration_requested",chain=chain,step=step,
                    initial_coordinates=coordinates.tolist(),source=source,beta=beta,proposal_versions=13,
                    candidate_scoring_calls=13,selector_calls=1)
                proposals=coordinate_bank(coordinates,step);actors=[];labels=[]
                raw=archive.save_npz(f"coherent/{prefix}_proposals.npz",coordinates=proposals,center=coordinates,
                    basis_sha256=np.array(basis["sha256"]),source=np.array(source),beta=np.array(beta),step=np.array(step))
                for i,c in enumerate(proposals):
                    construction_request=archive.append("coherent_proposal_version_requested",iteration_request_id=iteration_request,
                        chain=chain,step=step,candidate_index=i,coordinates=c.tolist(),parameter_constructions=1)
                    actor=actor_from_coordinates(base,subspace,c);actors.append(actor);label=f"{prefix}_c{i:02d}";labels.append(label)
                    checkpoint(out,label,dict(actor=actor.policy_state(),coordinates=c,proposal_index=i,step=step,chain=chain,
                        base_actor_hash=BOOTSTRAP_ACTOR,subspace_raw=basis,proposals_raw=raw,promotion_claim=False),archive)
                    archive.append("coherent_proposal_version_complete",request_id=construction_request,
                        checkpoint_label=label,actor_hash=policy_hash(actor))
                bank_digest=_digest(dict(coordinates=proposals,owners=[policy_hash(a) for a in actors],basis_sha256=basis["sha256"]))
                archive.append("coherent_bank_ready",chain=chain,step=step,proposals_raw=raw,checkpoint_labels=labels,
                    actor_hashes=[policy_hash(a) for a in actors],proposal_digest=bank_digest,
                    maximum_new_controls=13*1008,maximum_actor_rows=13*1008,
                    unused_paid_panels_hidden_from_selector=True)
                phase(prefix+"_owned_returns");scores=[];center_first=None;bank_ids=set()
                ordinary=[executor.task.reset(config["ordinary_training_seed"]+100*step+i) for i in range(8)]
                contexts=dict(ordinary=ordinary,**acquired)
                for i,actor in enumerate(actors):
                    evaluation_request=archive.append("coherent_candidate_evaluation_requested",iteration_request_id=iteration_request,
                        chain=chain,step=step,candidate_index=i,actor_hash=policy_hash(actor),
                        complete_episode_queries=48,maximum_controls=1008,maximum_actor_rows=1008)
                    panels={};mean_first=[]
                    for role,initials in contexts.items():
                        panels[role]={}
                        for mode,det in (("mean",True),("stochastic",False)):
                            tag=f"{prefix}_c{i:02d}_{role}_{mode}"
                            panels[role][mode],first=source_panel(actor,initials,tag,det,config["training_noise_seed"]+100*step)
                            if det:mean_first.append(first)
                    scoring_request=archive.append("coherent_candidate_score_requested",evaluation_request_id=evaluation_request,
                        chain=chain,step=step,candidate_index=i,actor_hash=policy_hash(actor),scoring_calls=1,owned_panels=6)
                    score=score_candidate(panels,policy_hash(actor));scores.append(score)
                    query_ids=[q for data in score["roles"].values() for mode in ("mean","stochastic") for q in data[mode]["query_ids"]]
                    if bank_ids.intersection(query_ids):raise RuntimeError("A policy candidate reused another candidate's physical query evidence")
                    bank_ids.update(query_ids);first=np.concatenate(mean_first)
                    if center_first is None:center_first=first.copy()
                    effects=dict(saved_initial_context_rows=24,mean_action_delta_rms=float(np.sqrt(np.mean((first-center_first)**2))),
                        mean_action_delta_max=float(np.max(np.abs(first-center_first))),saturation_fraction=float(np.mean(np.abs(first)>=.98)),
                        scope="Diagnostics fromalreadyqueriedfirstactions only; no additionalforward,norescaling,nogate")
                    archive.append("coherent_candidate_return",chain=chain,step=step,candidate_index=i,
                        checkpoint_label=labels[i],coordinates=proposals[i].tolist(),score=score,action_effects=effects,
                        evaluation_request_id=evaluation_request,scoring_request_id=scoring_request)
                views=[selector_view(s,source) for s in scores]
                selector_request=archive.append("coherent_selector_requested",iteration_request_id=iteration_request,
                    chain=chain,step=step,selector_input_sha256=_digest(views),source=source,beta=beta,
                    selector_calls=1,candidate_objective_evaluations=13)
                decision=select_candidate(views,source,beta)
                selection=dict(chain=chain,step=step,source=source,beta=beta,decision=decision,
                    proposal_digest=bank_digest,selector_input_sha256=_digest(views),visible_roles=list(views[0]["roles"]),
                    all_complete_scores=scores,proposals_raw=raw,selected_coordinates=proposals[decision["selected_index"]].tolist(),
                    selector_request_id=selector_request,iteration_request_id=iteration_request,
                    center_changed=not np.array_equal(coordinates,proposals[decision["selected_index"]]))
                entry["steps"].append(selection);archive.append("coherent_selection",result=selection)
                coordinates=proposals[decision["selected_index"]].copy();finals[chain]=actors[decision["selected_index"]]
                archive.append("coherent_search_iteration_complete",request_id=iteration_request,
                    chain=chain,step=step,center_changed=selection["center_changed"],final_coordinates=coordinates.tolist())
            entry["final_actor_hash"]=policy_hash(finals[chain]);entry["final_coordinates"]=coordinates.tolist()
        phase("coherent_ordinary_qualification");seeds=range(611000,611016)
        roster=dict(parent=base,**finals);mean={};noisy={}
        for name,actor in roster.items():
            mean[name]=evaluate(actor,seeds,"coherent_qual_"+name+"_mean",True,613000)
            noisy[name]=evaluate(actor,seeds,"coherent_qual_"+name+"_stochastic",False,613000)
        qualification_request=archive.append("coherent_qualification_selection_requested",selection_calls=1,
            complete_roster=["parent",*CHAINS],fresh_final_evidence_not_seen=True)
        selection=choose_controls(mean,noisy)
        result["qualification"]=dict(mean_panels={n:_summary(p) for n,p in mean.items()},
            stochastic_panels={n:_summary(p) for n,p in noisy.items()},selection=selection)
        archive.append("coherent_qualification",request_id=qualification_request,result=result["qualification"],fresh_final_evidence_not_seen=True)
        if selection["selected_long"] is None:result["stage"]="failed_all_coherent_data_qualification"
        else:
            phase("coherent_fresh_ordinary_evaluation");seeds=range(612000,612064)
            final_actors=dict(parent=base,guided=finals[selection["selected_long"]],
                best_control=base if selection["best_control"]=="parent" else finals[selection["best_control"]])
            m={n:evaluate(a,seeds,"coherent_final_"+n+"_mean",True,614000) for n,a in final_actors.items()}
            s={n:evaluate(a,seeds,"coherent_final_"+n+"_stochastic",False,614000) for n,a in final_actors.items()}
            final_request=archive.append("coherent_final_assessment_requested",assessment_calls=1)
            criterion=assess(m,s,"guided",("parent","best_control"))
            result["final"]=dict(criteria=criterion,mean_panels={n:_summary(v) for n,v in m.items()},
                stochastic_panels={n:_summary(v) for n,v in s.items()},actor_hashes={n:policy_hash(a) for n,a in final_actors.items()})
            result.update(passed=criterion["passed"],negative_retained=not criterion["passed"],
                stage="complete" if criterion["passed"] else "failed_fresh_coherent_data_improvement")
            archive.append("coherent_final_assessment_complete",request_id=final_request,criteria=criterion)
        if _digest(base.policy_state())!=base_digest:raise RuntimeError("Source parent or private RNG changed")
    except BaseException as error:
        result.update(passed=False,stage="execution_error",negative_retained=True,error_type=type(error).__name__,error=str(error))
        archive.append("coherent_error",error_type=type(error).__name__,error=str(error))
    finally:
        if installed:plugin.uninstall()
        work.close();archive.close()
    events=[json.loads(line) for line in work.path.read_text().splitlines()]
    local=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
    kinds=lambda name:[r for r in local if r["kind"]==name]
    returned=kinds("coherent_candidate_return");selections=kinds("coherent_selection")
    result["numerical_search_work"]=dict(
        declared_maximum=dict(search_iterations=15,proposal_versions=195,candidate_scoring_calls=195,
            selector_calls=15,candidate_objective_evaluations=195,qualification_selections=1,final_assessments=1),
        search_iterations_requested=len(kinds("coherent_search_iteration_requested")),
        search_iterations_completed=len(kinds("coherent_search_iteration_complete")),
        proposal_versions_requested=len(kinds("coherent_proposal_version_requested")),
        proposal_versions_constructed=len(kinds("coherent_proposal_version_complete")),
        candidate_evaluations_requested=len(kinds("coherent_candidate_evaluation_requested")),
        candidate_evaluations_completed=len(returned),candidate_scoring_calls_requested=len(kinds("coherent_candidate_score_requested")),
        candidate_scoring_calls_completed=len(returned),
        owned_panel_score_evaluations_requested=sum(r["owned_panels"] for r in kinds("coherent_candidate_score_requested")),
        owned_panel_score_evaluations_completed=6*len(returned),
        candidate_objective_evaluations_requested=sum(r["candidate_objective_evaluations"] for r in kinds("coherent_selector_requested")),
        candidate_objective_evaluations_completed=13*len(selections),selector_calls_requested=len(kinds("coherent_selector_requested")),
        selector_calls_completed=len(selections),center_changes=sum(r["result"]["center_changed"] for r in selections),
        center_stays=sum(not r["result"]["center_changed"] for r in selections),
        qualification_selections_requested=len(kinds("coherent_qualification_selection_requested")),
        qualification_selections_completed=len(kinds("coherent_qualification")),
        final_assessments_requested=len(kinds("coherent_final_assessment_requested")),
        final_assessments_completed=len(kinds("coherent_final_assessment_complete")),
        scope="Durable before-call numerical-search intents include failed/unresolved attempts; completed summaries overlap those intents,neveradd")
    result["accounting"]=dict(requested_scalar_controls=sum(e["controls"] for e in events if e["kind"]=="scalar_transition_requested"),
        completed_scalar_controls=sum(e["controls"] for e in events if e["kind"]=="scalar_transition_complete"),
        reserved_query_controls=0 if dynamics is None else dynamics.reserved,model_controls=0,
        executed_controls=0 if dynamics is None else dynamics.reserved,
        actor_rows=sum(e["rows"] for e in events if e["kind"]=="neural_requested"),actor_calls=sum(e["kind"]=="neural_requested" for e in events),
        adam_attempts=sum(e["kind"]=="optimizer_update_requested" for e in events),adam_completed=sum(e["kind"]=="optimizer_update_complete" for e in events),
        diagnostic_backward_attempts=sum(e.get("passes",0) for e in local if e["kind"]=="diagnostic_backward_requested"),
        gradient_backward_attempts=sum(e.get("passes",0) for e in local if e["kind"]=="gradient_backward_requested"),
        global_autograd_grad_calls=sum(e["kind"]=="autograd_grad_requested" for e in events),
        global_autograd_backward_calls=sum(e["kind"]=="autograd_backward_requested" for e in events),
        new_mujoco_controls=0,new_warp_controls=0,query_totals=archive.totals(),
        overlap="Attached local and global neural meters overlap; never sum")
    fresh=result["accounting"];early=provenance["reused_early_ancestry"];v3=provenance["reused_v3_ancestry"]
    result["cost_crosswalk"]=dict(new_assay=fresh,early_ancestor=early,v3_ancestor=v3,
        **{"unique_lineage_"+k:sum(row[k] for row in (fresh,early,v3)) for k in ("requested_scalar_controls","actor_rows","adam_attempts")},
        scope="Shared paid v3 and v5 lineage each once; every other historical negative remains paid separately in global work index")
    unchanged=all(_sha(out/r["file"])==r["sha256"] for r in manifest["files"])
    if not unchanged:result.update(passed=False,stage="source_changed",negative_retained=True)
    storage={"evidence_bytes":sum(p.stat().st_size for p in archive.path.rglob("*") if p.is_file()),
        "checkpoint_bytes":sum(p.stat().st_size for p in (out/"checkpoints").glob("*.pt")),"work_ledger_bytes":work.path.stat().st_size}
    result.update(config_sha256=_sha(out/"config.json"),manifest_sha256=_sha(out/"manifest.json"),
        input_provenance_sha256=_sha(out/"input_provenance.json"),query_ledger_sha256=_sha(archive.ledger_path),work_ledger_sha256=_sha(work.path),
        source_unchanged=unchanged,no_retuning=True,old_negatives_unchanged=True,declared_work_bound=bound,
        assay_wall_seconds=time.perf_counter()-started,reused_wall_seconds=provenance["reused_wall_seconds"],storage=storage,
        scope="One coherent whole-policy return/data-weight learner assay; no claimed superiority to PPO, no exploration efficiency, no two-round feedback result")
    _write(out/"assay_result.json",result);return result
