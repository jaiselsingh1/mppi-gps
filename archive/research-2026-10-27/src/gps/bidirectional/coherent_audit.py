"""Independent coherent-head assay arithmetic; NumPy/stdlib, no producers.

The generic Ending auditor validates every transition, finite complete return,
checkpoint owner and work event. This adapter validates the parameter search,
state-only ancestry, all measured panels and selection/confirmation gates.
No forward, simulator, optimizer, backward or test is invoked by this module.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np


BOOTSTRAP_SHA="e415ae406f8d1db5cd829f64b042da9abbfa1fd7710bfe46dfd7e64832322c83"
BOOTSTRAP_ACTOR="cb4f427fbd45c611e1ab256003319100fca744fc842f0333bea8aff500948158"
CHAINS=("ordinary","raw025","raw1","long025","long1")
SOURCES=dict(ordinary=("ordinary",0.),raw025=("raw",.25),raw1=("raw",1.),long025=("long",.25),long1=("long",1.))


def require(condition,message):
    if not condition:raise ValueError(message)


def same(actual,expected,message):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and a.dtype.kind in "biuf" and b.dtype.kind in "biuf"
        and np.isfinite(a).all() and np.isfinite(b).all()
        and np.allclose(a,b,atol=2e-12,rtol=2e-12),message)


def declared_bound():
    each=2*(8*25+16*19)
    return dict(training_controls=5*3*13*each,qualification_controls=6*16*25*2,
        final_controls=3*64*25*2,total_controls=210960,executed_controls=210960,
        actor_rows=210960,model_controls=0,adam_steps=0,diagnostic_backward_passes=0,
        training_actor_versions=195,training_episodes=9360,qualification_episodes=192,final_episodes=384)


def make_basis():
    draws=np.random.Generator(np.random.PCG64(1701)).standard_normal((128,4))
    q,r=np.linalg.qr(draws,mode="reduced")
    return (q*np.where(np.diag(r)<0,-1.,1.)[None,:]).T


def candidate_coordinates(center,step):
    c=np.asarray(center,dtype=np.float64)
    require(c.shape==(6,) and np.isfinite(c).all() and type(step) is int and 0<=step<3
        and np.abs(c).sum()<=step+1e-12,"Invalid coherent center/reachability")
    directions=np.eye(6);bank=[c.copy()]
    for vector in directions:
        bank.extend((c-vector,c+vector))
    return np.stack(bank)


def verify_head_actor(parent,candidate,basis,coordinates):
    """Rebuild absolute float64 changes, followed by original-dtype casting."""
    c,b=np.asarray(coordinates,dtype=np.float64),np.asarray(basis,dtype=np.float64)
    require(c.shape==(6,) and np.isfinite(c).all() and np.abs(c).sum()<=3+1e-12
        and b.shape==(4,128) and np.isfinite(b).all(),"Invalid coherent basis/coordinates")
    require(np.allclose(b@b.T,np.eye(4),atol=1e-12,rtol=0),"Coherent basis not orthonormal")
    for field in ("format_version","policy_type","obs_dim","hidden_dim","num_families","action_low","action_high"):
        require(parent[field]==candidate[field],"Coherent actor architecture/bounds changed")
    require(parent["obs_dim"]==8 and parent["hidden_dim"]==128
        and type(parent["num_families"]) is int and parent["num_families"]>=1
        and parent["action_low"]==[-1.] and parent["action_high"]==[1.],"Wrong coherent architecture")
    require(set(parent["actor"])==set(candidate["actor"]),"Coherent actor tensor roster changed")
    changes={"fc_mean.bias":.1*c[0],"fc_mean.weight":.1*(c[1:5]@b)[None,:],"log_std":.5*c[5]}
    for name,a in parent["actor"].items():
        a,z=np.asarray(a),np.asarray(candidate["actor"][name])
        require(a.shape==z.shape and a.dtype==z.dtype and np.isfinite(a).all() and np.isfinite(z).all(),"Coherent tensor shape/dtype/finite differs")
        expected=(a.astype(np.float64)+changes[name]).astype(a.dtype) if name in changes else a
        require(np.ascontiguousarray(z).tobytes()==np.ascontiguousarray(expected).tobytes(),
                "Coherent head/trunk/buffer bytes differ: "+name)
    require(np.asarray(parent["actor"]["fc_mean.weight"]).shape==(1,128)
        and np.asarray(parent["actor"]["fc_mean.bias"]).shape==(1,)
        and np.asarray(parent["actor"]["log_std"]).shape==(1,),"Wrong scalar coherent head shape")
    original_rng,current_rng=np.asarray(parent["sampling_rng_state"]),np.asarray(candidate["sampling_rng_state"])
    require(current_rng.shape==original_rng.shape and current_rng.dtype==original_rng.dtype
        and np.ascontiguousarray(current_rng).tobytes()==np.ascontiguousarray(original_rng).tobytes(),
        "Coherent private RNG changed")
    require(np.all(np.asarray(candidate["actor"]["log_std"])>=-5.)
        and np.all(np.asarray(candidate["actor"]["log_std"])<=1.),"Coherent log-standard-deviation bound crossed")
    return True


def select_from_scores(scores,source,beta):
    require(source in ("ordinary","raw","long") and beta in (0.,.25,1.)
        and ((source=="ordinary")==(beta==0.)) and len(scores)==13,"Undeclared coherent selection inputs")
    baseline=scores[0]["roles"]["ordinary"];feasible=[];objective=[]
    for score in scores:
        ordinary=score["roles"]["ordinary"]
        require(all(math.isfinite(ordinary[m]["mean_cost"]) for m in ("mean","stochastic")),"Nonfinite coherent score")
        feasible.append(all(ordinary[m]["mean_cost"]<=baseline[m]["mean_cost"]
            and ordinary[m]["captures"]>=baseline[m]["captures"] for m in ("mean","stochastic")))
        a=math.fsum(ordinary[m]["mean_cost"] for m in ("mean","stochastic"))/50.
        b=0. if source=="ordinary" else math.fsum(score["roles"][source][m]["mean_cost"] for m in ("mean","stochastic"))/38.
        require(math.isfinite(b),"Nonfinite coherent source score")
        objective.append((a+beta*b)/(1+beta))
    require(feasible[0],"Coherent current center infeasible")
    index=min((i for i,ok in enumerate(feasible) if ok),key=lambda i:(objective[i],i))
    return dict(selected_index=index,feasible_flags=feasible,objectives=objective)


def _assessment(mean,noisy,proposed,references):
    checks={};all_pass=[]
    for mode,data in (("mean",mean),("stochastic",noisy)):
        checks[mode]={}
        for name in references:
            a,b=data[name],data[proposed]
            require(len(a)==len(b)>0 and [x["context_id"] for x in a]==[x["context_id"] for x in b],"Coherent unpaired qualification")
            gains=[x["cost"]-y["cost"] for x,y in zip(a,b)];gain=math.fsum(gains)/len(gains)
            nonreg=sum(x["capture"] for x in b)>=sum(x["capture"] for x in a)
            check=dict(mean_cost_gain=gain,paired_cost_gains=gains,capture_nonregression=bool(nonreg),
                improved=bool(nonreg and gain>=.001))
            checks[mode][name]=check;all_pass.append(check["improved"])
    capture=sum(x["capture"] for x in mean[proposed])/len(mean[proposed])
    return dict(comparisons=checks,deterministic_capture_rate=capture,passed=bool(capture>=.9 and all(all_pass)))


def _same_assessment(actual,expected):
    same(actual["deterministic_capture_rate"],expected["deterministic_capture_rate"],"Coherent capture criterion differs")
    require(actual["passed"]==expected["passed"] and set(actual["comparisons"])==set(expected["comparisons"]),"Coherent criterion verdict/modes differ")
    for mode,data in expected["comparisons"].items():
        require(set(actual["comparisons"][mode])==set(data),"Coherent comparison roster differs")
        for name,row in data.items():
            a=actual["comparisons"][mode][name]
            for key in ("mean_cost_gain","paired_cost_gains"):same(a[key],row[key],"Coherent complete paired gain differs")
            for key in ("capture_nonregression","improved"):require(a[key]==row[key],"Coherent capture/gain decision differs")


def validate_coherent_assay(e,base,rows,work,result,config,checkpoints,episodes,qreq,qdone,task_hash):
    from .ending_audit import checkpoint_numpy,policy_identity,controller_identity,utc,transition_arithmetic,COMPONENTS
    from .ending_completion_audit import numeric_digest
    from .ending_precision_audit import _raw,_summary_check
    from .early_context_audit import actual_state_context
    task=dict(gate=12,delay=12,move=.2,ending_y=-.1,effort=.02,greed=.2,detour=.2,detour_steps=5,penalty=20.)
    required=dict(schema="bidirectional-ending-coherent-head-assay-v1",task=task,chains=list(CHAINS),
        search_steps=3,candidates_per_step=13,basis_seed=1701,head_weight_directions=4,head_step=.1,log_std_step=.5,
        coefficient_l1_bound=3.,data_weights=[0.,.25,1.],controller_mode_weights=dict(mean=.5,stochastic=.5),
        ordinary_training_roots=8,ordinary_training_seed=601000,training_noise_seed=603000,acquired_elapsed=6,
        source_contexts_per_role=8,all_paid_roles=["ordinary","raw","long"],
        ordinary_step_constraint="complete_mean_and_stochastic_cost_and_capture_nonregression_vs_current_center",
        qualification_roots=16,qualification_seed=611000,qualification_noise_seed=613000,
        evaluation_roots=64,evaluation_seed=612000,evaluation_noise_seed=614000,minimum_complete_gain=.001,
        capture_min=.9,max_controls=210960,max_executed_controls=210960,max_model_controls=0,max_actor_rows=210960,
        max_adam_steps=0,max_diagnostic_backward_passes=0)
    require(config==required,"Undeclared coherent-head assay configuration")
    bound=declared_bound()
    for k,v in bound.items():require(result["declared_work_bound"][k]==v,"Coherent prospective work bound differs")
    require(result["no_new_action_labels"] is True and result["critic_used"] is False
        and result["gradient_optimizer_used"] is False and result["parameter_return_search_used"] is True
        and result["full_feedback_established"] is False and result["efficiency_claim"] is False
        and result["ppo_comparison"] is False and result["no_retuning"] is True and result["old_negatives_unchanged"] is True,
        "Coherent assay claimed a different mechanism/scope")
    require(all(r["mode"]=="live" for r in qreq.values()) and result["accounting"]["model_controls"]==0,
        "Coherent assay made an unallowed model query")
    require(not any(r["kind"] in ("fit_work","gradient_backward_requested","diagnostic_backward_requested","optimizer_requested") for r in rows)
        and not any(r["kind"] in ("optimizer_update_requested","autograd_grad_requested","autograd_backward_requested") for r in work),
        "Coherent assay performed a gradient/optimizer update")
    provenance=e.json("input_provenance.json")
    require(e.hashes["input_provenance.json"]==result["input_provenance_sha256"]
        and provenance["schema"]=="ending-coherent-head-input-provenance-v1"
        and provenance["bootstrap_sha256"]==BOOTSTRAP_SHA and provenance["bootstrap_actor_hash"]==BOOTSTRAP_ACTOR,
        "Coherent exact weak-parent provenance differs")
    refs={r["file"]:r for r in provenance["files"]}
    expected_inputs={"manifest.json","config.json","loop_result.json","audit_001.json","input_provenance.json",
        "evidence/loop/ledger.jsonl","work_ledger.jsonl","inputs/checkpoints/bootstrap.pt"}
    expected_inputs|={f"evidence/loop/episodes/early_initial_{role}_{i:03d}.npz" for role in ("short","long") for i in range(8)}
    require(set(refs)==expected_inputs and len(refs)==len(provenance["files"]),"Coherent ancestor input roster differs")
    for name,ref in refs.items():
        path=e.file(Path("inputs")/name,ref["sha256"]);require(path.stat().st_size==ref["bytes"],"Coherent ancestor bytes differ")
    old=e.json("inputs/loop_result.json");old_audit=e.json("inputs/audit_001.json");old_manifest=e.json("inputs/manifest.json")
    require(old_audit["status"]=="PASS" and old_audit["evidence_arithmetic_valid"] is True and old_audit["scientific_success"] is False
        and not old_audit["errors"] and not old_audit["pending"] and old["passed"] is False
        and old["schema"]=="bidirectional-ending-early-context-loop-v5" and old["stage"]=="failed_initial_data_all_actor_qualification",
        "Coherent ancestor negative/audit substituted")
    for name,ref in refs.items():
        if name!="audit_001.json":require(old_audit["input_sha256"].get(name)==ref["sha256"],"Coherent ancestor audit binding differs: "+name)
    for name,key in (("manifest.json","manifest_sha256"),("config.json","config_sha256"),
        ("evidence/loop/ledger.jsonl","query_ledger_sha256"),("work_ledger.jsonl","work_ledger_sha256"),("input_provenance.json","input_provenance_sha256")):
        require(refs[name]["sha256"]==old[key],"Coherent ancestor result binding differs")
    require(provenance["reused_early_ancestry"]==old["accounting"] and provenance["reused_v3_ancestry"]==old["cost_crosswalk"]["v3_ancestor"],"Coherent reused costs differ")
    same(provenance["reused_wall_seconds"],old["loop_wall_seconds"]+old["reused_wall_seconds"],"Coherent reused wall differs")
    old_sources={r["file"]:r["sha256"] for r in old_manifest["files"]}
    critical=("src/ending/core.py","src/gps/bidirectional/ending_pilot.py","src/gps/bidirectional/policy.py")
    require(set(provenance["critical_source_sha256"])==set(critical),"Coherent critical source roster differs")
    for name,sha in provenance["critical_source_sha256"].items():
        require(sha==old_sources["source/"+name]==e.hashes["source/"+name],"Coherent reused actor/task source changed")
    source=checkpoint_numpy(e.file("inputs/inputs/checkpoints/bootstrap.pt",BOOTSTRAP_SHA))
    parent=source["learner"]["policy"];require(policy_identity(parent)==BOOTSTRAP_ACTOR,"Wrong coherent bootstrap actor")
    loaded=checkpoints["coherent_source_parent"]
    require(numeric_digest(loaded["actor"])==numeric_digest(parent)
        and loaded["source_checkpoint_sha256"]==BOOTSTRAP_SHA
        and loaded["source_full_learner_sha256"]==numeric_digest(source["learner"]),"Coherent loaded full parent/RNG differs")
    require(np.all(np.asarray(parent["actor"]["log_std"])-1.5>=-5.)
        and np.all(np.asarray(parent["actor"]["log_std"])+1.5<=1.),"Fixed coherent reachable logstd domain crosses actor bounds")
    basis_raw=e.npz(base,result["subspace"]["raw"]);basis=basis_raw["weight_directions"]
    same(basis,make_basis(),"Coherent source-independent QR basis differs")
    require(int(basis_raw["seed"])==1701 and float(basis_raw["head_step"])==.1 and float(basis_raw["log_std_step"])==.5
        and str(basis_raw["base_actor_hash"])==BOOTSTRAP_ACTOR and result["subspace"]["dimensions"]==6
        and result["subspace"]["frozen_trunk"] is True,"Coherent subspace metadata differs")
    old_rows=e.rows("inputs/evidence/loop/ledger.jsonl")
    old_events={r["episode"]:r for r in old_rows if r["kind"]=="episode_outcome"}
    sources={};reused_checks=0
    for role,tag in (("raw","short"),("long","long")):
        records=result["source_contexts"][role];require(len(records)==8,"Coherent source panel incomplete")
        for i,record in enumerate(records):
            name=f"early_initial_{tag}_{i:03d}";event=old_events[name]
            require(provenance["source_events"][role][i]==event and record["source_episode"]==name
                and record["source_outcome"]==event["outcome"] and record["context_id"]==f"coherent_source_{role}:e6:{i}","Coherent source event/context differs")
            ref=record["source_raw"];path=Path("inputs/evidence/loop")/event["raw"]["file"]
            require(ref==dict(file=str(path),sha256=event["raw"]["sha256"],bytes=event["raw"]["bytes"]),"Coherent state source raw reference differs")
            raw=_raw(e,path,ref["sha256"],ref["bytes"]);state=actual_state_context(raw,record)
            post,parts,cost,flags=transition_arithmetic(raw["states"][:-1],raw["actions"],task)
            same(raw["states"][1:],post,"Coherent reused acquisition transition differs");same(raw["raw_costs"],cost,"Coherent reused acquisition cost differs")
            same(raw["cost_components"],np.stack([parts[k] for k in COMPONENTS],axis=-1),"Coherent reused acquisition components differ")
            sources[record["context_id"]]=state;reused_checks+=25
    source_events=[r for r in rows if r["kind"]=="coherent_source_contexts"]
    require(len(source_events)==1 and source_events[0]["sources"]==result["source_contexts"] and source_events[0]["labels_are_actions"] is False,"Coherent source declaration differs")
    checkpoint_events={r["label"]:r for r in rows if r["kind"]=="checkpoint"}
    require(len(checkpoint_events)==sum(r["kind"]=="checkpoint" for r in rows),"Repeated coherent checkpoint label")
    by_name={ep["episode"]:ep for ep in episodes};require(len(by_name)==len(episodes),"Duplicated coherent episode name")
    rowgroups=defaultdict(list)
    for row in rows:rowgroups[row["kind"]].append(row)
    used=set();all_ids=set();reset_checks=0;panel_checks=0
    def panel(prefix,n,det,owner,*,seed=None,role=None,step=None):
        nonlocal reset_checks,panel_checks
        values=[by_name[f"{prefix}_{i:03d}"] for i in range(n)]
        for i,ep in enumerate(values):
            expected_context=f"ordinary:{seed+i}" if seed is not None else f"coherent_source_{role}:e6:{i}"
            require(ep["episode"] not in used and ep["context_id"]==expected_context and ep["controls"]==(25 if seed is not None else 19)
                and ep["actor_hash"]==owner and ep["deterministic"] is det and ep["semantics"]=="actor_suffix" and ep["closed_outcome"],"Coherent panel ownership/context/completion differs")
            require(not all_ids.intersection(ep["query_ids"]),"Coherent panels alias physical queries")
            raw=e.npz(base,ep["raw_ref"])
            initial=[0.,np.random.default_rng(seed+i).uniform(.2,.4)] if seed is not None else sources[expected_context]
            same(raw["initial_state"],initial,"Coherent actual reset/source state differs")
            require(all(qreq[q]["mode"]=="live" for q in ep["query_ids"]),"Coherent panel contains model control")
            reset_checks+=seed is not None;used.add(ep["episode"]);all_ids.update(ep["query_ids"])
        panel_checks+=1;return values
    def score_panels(panels,owner):
        score=dict(actor_hash=owner,roles={})
        for role in ("ordinary","raw","long"):
            modes={}
            for mode,det in (("mean",True),("stochastic",False)):
                p=panels[role][mode];modes[mode]=dict(mean_cost=math.fsum(x["cost"] for x in p)/8,
                    captures=sum(x["capture"] for x in p),controller_hash=controller_identity(dict(actor_hash=owner,family=0,deterministic=det)),
                    context_ids=[x["context_id"] for x in p],query_ids=[q for x in p for q in x["query_ids"]])
            modes["joint_normalized_cost"]=(modes["mean"]["mean_cost"]+modes["stochastic"]["mean_cost"])/(50. if role=="ordinary" else 38.)
            score["roles"][role]=modes
        return score
    def same_score(actual,expected):
        require(actual["actor_hash"]==expected["actor_hash"] and set(actual["roles"])==set(expected["roles"]),"Coherent score role/owner differs")
        for role,value in expected["roles"].items():
            same(actual["roles"][role]["joint_normalized_cost"],value["joint_normalized_cost"],"Coherent normalized score differs")
            for mode in ("mean","stochastic"):
                a,b=actual["roles"][role][mode],value[mode];same(a["mean_cost"],b["mean_cost"],"Coherent full mean return differs")
                for key in ("captures","controller_hash","context_ids","query_ids"):require(a[key]==b[key],"Coherent score evidence/owner differs: "+key)
    def unique(kind,**keys):
        matched=[r for r in rowgroups[kind] if all(r.get(k)==v for k,v in keys.items())]
        require(len(matched)==1,"Coherent event missing/duplicate: "+kind+str(keys));return matched[0]
    if result["stage"]=="execution_error":
        return dict(complete=False,scientific_success=False,partial_stage=True,
            paid_query_requests=len(qreq),paid_controls=sum(r["requested_controls"] for r in qreq.values()),
            numerical_search_intents={kind:len(events) for kind,events in rowgroups.items()
                if kind.startswith("coherent_") and kind.endswith("_requested")},
            analytic_reused_episode_transition_row_checks=reused_checks,
            limits="Operationally incomplete; no full candidate/selection/qualification assertion")
    require([c["chain"] for c in result["chains"]]==list(CHAINS),"Coherent chain roster differs")
    final_owners={};step_reports=[]
    for chain_entry in result["chains"]:
        chain=chain_entry["chain"];source_name,beta=SOURCES[chain];center=np.zeros(6)
        require(chain_entry["source"]==source_name and chain_entry["beta"]==beta and len(chain_entry["steps"])==3,"Coherent chain source/steps differ")
        for step,selection in enumerate(chain_entry["steps"]):
            prefix=f"coherent_{chain}_s{step}";expected_coords=candidate_coordinates(center,step)
            proposals=e.npz(base,selection["proposals_raw"])
            same(proposals["coordinates"],expected_coords,"Coherent proposal bank/order differs");same(proposals["center"],center,"Coherent proposal center differs")
            require(str(proposals["source"])==source_name and float(proposals["beta"])==beta and int(proposals["step"])==step
                and str(proposals["basis_sha256"])==result["subspace"]["raw"]["sha256"],"Coherent proposal metadata differs")
            request=unique("coherent_search_iteration_requested",chain=chain,step=step)
            bank=unique("coherent_bank_ready",chain=chain,step=step)
            same(request["initial_coordinates"],center,"Coherent iteration initial center differs")
            require(request["id"]==selection["iteration_request_id"] and request["source"]==source_name and request["beta"]==beta
                and request["proposal_versions"]==request["candidate_scoring_calls"]==13 and request["selector_calls"]==1
                and bank["unused_paid_panels_hidden_from_selector"] is True
                and bank["proposals_raw"]==selection["proposals_raw"] and bank["maximum_new_controls"]==bank["maximum_actor_rows"]==13104,
                "Coherent iteration/bank request differs")
            scores=[];owners=[];all_first=[];bank_last=[];bank_first=[]
            for i,coordinates in enumerate(expected_coords):
                label=f"{prefix}_c{i:02d}";saved=checkpoints[label];actor=saved["actor"];owner=policy_identity(actor);owners.append(owner)
                verify_head_actor(parent,actor,basis,coordinates)
                require(saved["proposal_index"]==i and saved["step"]==step and saved["chain"]==chain
                    and saved["base_actor_hash"]==BOOTSTRAP_ACTOR and saved["subspace_raw"]==result["subspace"]["raw"]
                    and saved["proposals_raw"]==selection["proposals_raw"] and saved["promotion_claim"] is False,"Coherent candidate checkpoint source differs")
                same(saved["coordinates"],coordinates,"Coherent saved coordinates differ")
                version_request=unique("coherent_proposal_version_requested",chain=chain,step=step,candidate_index=i)
                version_complete=unique("coherent_proposal_version_complete",request_id=version_request["id"])
                require(version_request["iteration_request_id"]==request["id"] and version_request["parameter_constructions"]==1
                    and version_complete["checkpoint_label"]==label and version_complete["actor_hash"]==owner
                    and utc(request)<utc(version_request)<=utc(checkpoint_events[label])<=utc(version_complete)<=utc(bank),"Coherent candidate bank was not frozen before returns")
                same(version_request["coordinates"],coordinates,"Coherent proposal request coordinates differ")
                evaluation=unique("coherent_candidate_evaluation_requested",chain=chain,step=step,candidate_index=i)
                returned=unique("coherent_candidate_return",chain=chain,step=step,candidate_index=i)
                scoring=unique("coherent_candidate_score_requested",chain=chain,step=step,candidate_index=i)
                require(evaluation["iteration_request_id"]==request["id"] and evaluation["actor_hash"]==owner
                    and evaluation["complete_episode_queries"]==48 and evaluation["maximum_controls"]==evaluation["maximum_actor_rows"]==1008
                    and returned["evaluation_request_id"]==evaluation["id"] and returned["scoring_request_id"]==scoring["id"]
                    and scoring["evaluation_request_id"]==evaluation["id"] and scoring["scoring_calls"]==1 and scoring["owned_panels"]==6,
                    "Coherent candidate accounting linkage differs")
                require(scoring["actor_hash"]==owner,"Coherent score request names another actor")
                panels={};first=[]
                for role in ("ordinary","raw","long"):
                    panels[role]={}
                    for mode,det in (("mean",True),("stochastic",False)):
                        tag=f"{prefix}_c{i:02d}_{role}_{mode}"
                        p=panel(tag,8,det,owner,seed=601000+100*step if role=="ordinary" else None,role=role,step=step)
                        panels[role][mode]=p;event=unique("coherent_panel",panel_id=tag)
                        require(event["episode_names"]==[v["episode"] for v in p] and event["actor_hash"]==owner
                            and event["deterministic"] is det and event["raw_refs"]==[v["raw_ref"] for v in p],"Coherent panel event identity differs")
                        _summary_check(dict(outcomes=event["outcomes"],mean_cost=math.fsum(x["cost"] for x in p)/8,capture_rate=sum(x["capture"] for x in p)/8),p)
                        raws=[e.npz(base,v["raw_ref"]) for v in p]
                        same(event["initial_states"],[z["initial_state"] for z in raws],"Coherent panel initial states differ")
                        same(event["first_actions"],[z["actions"][0,0] for z in raws],"Coherent queried first action diagnostics differ")
                        rng=e.npz(base,event["sampling_rng"])
                        require(int(rng["seed"])==603000+100*step and str(rng["actor_hash"])==owner and bool(rng["deterministic"]) is det
                            and rng["before"].dtype==rng["after"].dtype==np.dtype("uint8") and rng["before"].shape==rng["after"].shape,"Coherent sampling packet differs")
                        if det:require(np.array_equal(rng["before"],rng["after"]),"Deterministic coherent controller consumed sampling noise")
                        if det:first.extend(float(z["actions"][0,0]) for z in raws)
                        first_time=min(utc(qreq[q]) for v in p for q in v["query_ids"])
                        last_time=max(utc(qdone[q]) for v in p for q in v["query_ids"])
                        require(utc(bank)<utc(evaluation)<=first_time<=last_time<=utc(event)<=utc(scoring)<=utc(returned),"Coherent candidate score used future/unfrozen behavior")
                        bank_first.append(first_time);bank_last.append(last_time)
                expected=score_panels(panels,owner);same_score(returned["score"],expected)
                require(returned["checkpoint_label"]==label and returned["score"]["score_semantics"]=="equal_mean_of_two_SEPARATELY_EXECUTED_controller_costs_normalized_by_remaining","Coherent score semantics differ")
                same(returned["coordinates"],coordinates,"Coherent measured actor coordinates differ")
                first=np.asarray(first);all_first.append(first)
                effects=returned["action_effects"];same(effects["mean_action_delta_rms"],np.sqrt(np.mean((first-all_first[0])**2)),"Coherent action-effect RMS differs")
                same(effects["mean_action_delta_max"],np.max(abs(first-all_first[0])),"Coherent action-effect max differs")
                same(effects["saturation_fraction"],np.mean(abs(first)>=.98),"Coherent first-action saturation differs")
                require(effects["saved_initial_context_rows"]==24,"Coherent action effect used another state roster")
                scores.append(returned["score"])
            require(bank["checkpoint_labels"]==[f"{prefix}_c{i:02d}" for i in range(13)] and bank["actor_hashes"]==owners
                and utc(bank)<min(bank_first),"Coherent full bank not ready before any query")
            bank_digest=numeric_digest(dict(coordinates=expected_coords,owners=owners,basis_sha256=result["subspace"]["raw"]["sha256"]))
            require(bank["proposal_digest"]==selection["proposal_digest"]==bank_digest,"Coherent bank digest differs")
            require(selection["all_complete_scores"]==scores,"Coherent selection changed measured full scores")
            decision=select_from_scores(scores,source_name,beta)
            actual=selection["decision"]
            require(actual["selected_index"]==decision["selected_index"] and actual["feasible_flags"]==decision["feasible_flags"]
                and actual["source"]==source_name and actual["beta"]==beta and actual["actor_hash"]==owners[decision["selected_index"]],"Coherent feasible selection differs")
            same(actual["objectives"],decision["objectives"],"Coherent full-return objective differs")
            visible=["ordinary"] if source_name=="ordinary" else ["ordinary",source_name]
            views=[dict(actor_hash=s["actor_hash"],roles={r:deepcopy(s["roles"][r]) for r in visible},score_semantics=s["score_semantics"]) for s in scores]
            selector=unique("coherent_selector_requested",chain=chain,step=step)
            require(selection["visible_roles"]==visible and selector["selector_input_sha256"]==selection["selector_input_sha256"]==numeric_digest(views)
                and selector["id"]==selection["selector_request_id"] and selector["iteration_request_id"]==request["id"]
                and selector["source"]==source_name and selector["beta"]==beta and selector["selector_calls"]==1
                and selector["candidate_objective_evaluations"]==13 and max(bank_last)<=utc(selector),"Coherent selector accessed different/unfinished panels")
            selected=expected_coords[decision["selected_index"]];changed=not np.array_equal(center,selected)
            require(selection["center_changed"]==changed,"Coherent center-change label differs")
            same(selection["selected_coordinates"],selected,"Coherent chosen center differs")
            event=unique("coherent_selection",result=selection)
            complete=unique("coherent_search_iteration_complete",chain=chain,step=step)
            require(complete["request_id"]==request["id"] and complete["center_changed"]==changed
                and utc(selector)<=utc(event)<=utc(complete),"Coherent selector/iteration chronology differs")
            same(complete["final_coordinates"],selected,"Coherent completed center differs")
            center=selected.copy();final_owners[chain]=owners[decision["selected_index"]]
            step_reports.append(dict(chain=chain,step=step,selected_index=decision["selected_index"]
                ,center_changed=changed,selected_actor_hash=final_owners[chain],objectives=decision["objectives"],feasible_flags=decision["feasible_flags"]))
        require(chain_entry["final_actor_hash"]==final_owners[chain],"Coherent final-chain owner differs")
        same(chain_entry["final_coordinates"],center,"Coherent final-chain coordinates differ")
    require(set(checkpoints)=={"coherent_source_parent"}|{f"coherent_{chain}_s{step}_c{i:02d}" for chain in CHAINS for step in range(3) for i in range(13)},"Coherent checkpoint roster has extra/missing policies")
    roster=dict(parent=BOOTSTRAP_ACTOR,**final_owners);qual={mode:{} for mode in ("mean","stochastic")}
    for mode,det in (("mean",True),("stochastic",False)):
        for name,owner in roster.items():
            values=panel(f"coherent_qual_{name}_{mode}",16,det,owner,seed=611000)
            qual[mode][name]=values;_summary_check(result["qualification"]["mean_panels" if det else "stochastic_panels"][name],values)
    controls=("ordinary","raw025","raw1")
    assessments={name:_assessment(qual["mean"],qual["stochastic"],name,("parent",)) for name in controls}
    eligible=[n for n in controls if assessments[n]["passed"]]
    best="parent" if not eligible else min(eligible,key=lambda n:(math.fsum(x["cost"] for x in qual["mean"][n])/16,controls.index(n)))
    long_names=("long025","long1")
    long_assess={name:_assessment(qual["mean"],qual["stochastic"],name,("parent",best)) for name in long_names}
    eligible_long=[n for n in long_names if long_assess[n]["passed"]]
    chosen=None if not eligible_long else min(eligible_long,key=lambda n:(math.fsum(x["cost"] for x in qual["mean"][n])/16,long_names.index(n)))
    selection=result["qualification"]["selection"]
    require(selection["best_control"]==best and selection["selected_long"]==chosen
        and selection["control_parent_fallback"]==(best=="parent"),"Coherent qualification-only actor/weight selection differs")
    for name,value in assessments.items():_same_assessment(selection["control_criteria"][name],value)
    for name,value in long_assess.items():_same_assessment(selection["long_criteria"][name],value)
    qselect=unique("coherent_qualification_selection_requested")
    qevent=unique("coherent_qualification")
    require(qselect["complete_roster"]==["parent",*CHAINS] and qselect["fresh_final_evidence_not_seen"] is True
        and qselect["selection_calls"]==1 and qevent["request_id"]==qselect["id"]
        and qevent["result"]==result["qualification"] and qevent["fresh_final_evidence_not_seen"] is True
        and utc(qselect)<=utc(qevent),"Coherent qualification request/event differs")
    qual_ids=[q for mode in qual.values() for p in mode.values() for ep in p for q in ep["query_ids"]]
    require(max(utc(qdone[q]) for q in qual_ids)<=utc(qselect),"Coherent strongest-control selection predates qualification")
    final_pass=False
    if chosen is None:
        require(result["passed"] is False and result["stage"]=="failed_all_coherent_data_qualification" and "final" not in result
            and not rowgroups["coherent_final_assessment_requested"] and not rowgroups["coherent_final_assessment_complete"],"Failed coherent qualification did not stop")
    else:
        final_roster=dict(parent=BOOTSTRAP_ACTOR,guided=final_owners[chosen],best_control=roster[best]);final={m:{} for m in ("mean","stochastic")}
        require(result["final"]["actor_hashes"]==final_roster,"Coherent fresh-final owner changed")
        for mode,det in (("mean",True),("stochastic",False)):
            for name,owner in final_roster.items():
                values=panel(f"coherent_final_{name}_{mode}",64,det,owner,seed=612000);final[mode][name]=values
                _summary_check(result["final"]["mean_panels" if det else "stochastic_panels"][name],values)
                require(utc(qevent)<min(utc(qreq[q]) for ep in values for q in ep["query_ids"]),"Coherent untouched final panel preceded actor selection")
        criterion=_assessment(final["mean"],final["stochastic"],"guided",("parent","best_control"));_same_assessment(result["final"]["criteria"],criterion)
        final_request=unique("coherent_final_assessment_requested");final_event=unique("coherent_final_assessment_complete")
        require(final_event["request_id"]==final_request["id"] and final_request["assessment_calls"]==1
            and final_event["criteria"]==result["final"]["criteria"],"Coherent final assessment request differs")
        final_ids=[q for mode in final.values() for p in mode.values() for ep in p for q in ep["query_ids"]]
        require(max(utc(qdone[q]) for q in final_ids)<=utc(final_request)<=utc(final_event),"Coherent final criterion preceded completed returns")
        final_pass=criterion["passed"]
        require(result["passed"]==final_pass and result["stage"]==("complete" if final_pass else "failed_fresh_coherent_data_improvement"),"Coherent scientific verdict differs")
    require(set(by_name)==used and set(qreq)==all_ids,"Coherent unaccounted episode/query outside fixed panels")
    training_panels=rowgroups["coherent_panel"]
    require(len(training_panels)==1170,"Coherent paid training panel count differs")
    expected_work=dict(search_iterations_requested=15,search_iterations_completed=15,proposal_versions_requested=195,
        proposal_versions_constructed=195,candidate_evaluations_requested=195,candidate_evaluations_completed=195,
        candidate_scoring_calls_requested=195,candidate_scoring_calls_completed=195,owned_panel_score_evaluations_requested=1170,
        owned_panel_score_evaluations_completed=1170,candidate_objective_evaluations_requested=195,candidate_objective_evaluations_completed=195,
        selector_calls_requested=15,selector_calls_completed=15,center_changes=sum(s["center_changed"] for s in step_reports),
        center_stays=sum(not s["center_changed"] for s in step_reports),qualification_selections_requested=1,qualification_selections_completed=1,
        final_assessments_requested=int(chosen is not None),final_assessments_completed=int(chosen is not None))
    require(result["numerical_search_work"]["declared_maximum"]==dict(search_iterations=15,proposal_versions=195,
        candidate_scoring_calls=195,selector_calls=15,candidate_objective_evaluations=195,
        qualification_selections=1,final_assessments=1),"Coherent numerical-work prospective maximum differs")
    event_counts={"coherent_search_iteration_requested":15,"coherent_search_iteration_complete":15,
        "coherent_proposal_version_requested":195,"coherent_proposal_version_complete":195,
        "coherent_candidate_evaluation_requested":195,"coherent_candidate_score_requested":195,
        "coherent_candidate_return":195,"coherent_selector_requested":15,"coherent_selection":15}
    for kind,n in event_counts.items():require(len(rowgroups[kind])==n,"Extra/missing coherent work event: "+kind)
    for key,value in expected_work.items():require(result["numerical_search_work"][key]==value,"Coherent numerical parameter-work summary differs: "+key)
    actual_controls=196560+4800+(9600 if chosen is not None else 0)
    require(sum(q["requested_controls"] for q in qreq.values())==actual_controls
        and result["accounting"]["actor_rows"]==actual_controls and len(episodes)==9360+192+(384 if chosen is not None else 0),"Coherent completed schedule work differs")
    cross=result["cost_crosswalk"]
    require(cross["new_assay"]==result["accounting"] and cross["early_ancestor"]==provenance["reused_early_ancestry"]
        and cross["v3_ancestor"]==provenance["reused_v3_ancestry"],"Coherent cost ancestry rows differ")
    for key in ("requested_scalar_controls","actor_rows","adam_attempts"):
        require(cross["unique_lineage_"+key]==sum(v[key] for v in (result["accounting"],provenance["reused_early_ancestry"],provenance["reused_v3_ancestry"])),"Coherent unique cost crosswalk differs")
    return dict(complete=True,scientific_success=final_pass,selected_long=chosen,best_control=best,
        exact_parameter_versions_verified=195,search_steps=step_reports,numerical_search_work=expected_work,
        complete_owned_training_episodes=9360,ordinary_qualification_episodes=192,fresh_final_episodes=384 if chosen is not None else 0,
        actual_scalar_controls=actual_controls,analytic_reused_episode_transition_row_checks=reused_checks,
        reset_RNG_checks=int(reset_checks),owned_panels_verified=panel_checks,
        frozen_trunk_and_private_rng=True,unused_source_panels_hidden_by_bound_digest=True,
        correlated_identical_candidate_repeats_are_not_independent_evidence=True,
        audit_physics_controls=0,audit_actor_forward_rows=0,audit_optimizer_steps=0,
        limits=["Actor outputs and stochastic RNG draws are source/checkpoint/chronology bound, not independently forwarded or regenerated",
            "Arithmetic PASS does not establish data-transfer benefit, MPPI discovery, two-round feedback or efficiency"])
