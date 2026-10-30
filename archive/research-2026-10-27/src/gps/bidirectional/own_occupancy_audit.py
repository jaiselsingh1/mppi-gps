"""Independent own-occupancy action-gradient evidence checks; NumPy/stdlib only.

Saved complete frozen-parent branches supply action derivatives, not a learned
critic or returns owned by an updated policy. No simulator or actor is imported.
"""
from __future__ import annotations

import json
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

SEMANTICS="frozen_parent_own_occupancy_action_gradient"


def require(condition,message):
    if not condition:raise ValueError(message)


def same(actual,expected,message):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and a.dtype.kind in "biuf" and b.dtype.kind in "biuf"
        and np.isfinite(a).all() and np.isfinite(b).all()
        and np.allclose(a,b,atol=2e-12,rtol=2e-12),message)


def difference_targets(mean_action,first_actions,full_costs,prefix_costs):
    """Clipped minus/plus finite differences in bounded action units."""
    mean=np.asarray(mean_action,dtype=np.float64)
    first=np.asarray(first_actions,dtype=np.float64)
    full=np.asarray(full_costs,dtype=np.float64);prefix=np.asarray(prefix_costs,dtype=np.float64)
    require(mean.shape==(1,) and first.shape==(2,1) and full.shape==prefix.shape==(2,)
        and all(np.isfinite(value).all() for value in (mean,first,full,prefix))
        and np.all(np.abs(mean)<=1),"Invalid own-occupancy derivative arrays")
    expected=np.clip(mean[None,:]+np.array([[-.05],[.05]]),-1.,1.)
    same(first,expected,"Own-occupancy perturbations differ from declared clipped ±.05")
    denominator=float(first[1,0]-first[0,0])
    require(denominator>0,"Degenerate finite-difference denominator")
    return dict(denominator=denominator,full_gradient=float((full[1]-full[0])/denominator),
        local_gradient=float((prefix[1]-prefix[0])/denominator),null_gradient=0.)


def validate_probe(raw,request,event,center_raw,elapsed,query,task_hash,controller,ending_y=-.1):
    """Bind both complete probes and the actual center suffix, without a forward."""
    state=np.asarray(center_raw["states"])[elapsed];mean=np.asarray(center_raw["actions"])[elapsed].astype(np.float64)
    center_costs=np.asarray(center_raw["raw_costs"])[elapsed:];R=25-elapsed;n=2*R
    center_ids=np.asarray(center_raw["query_ids"])[elapsed:].tolist();owner=request["actor_hash"]
    center_owner=json.loads(str(center_raw["owner_json"]))
    require(center_owner["actor_hash"]==owner and center_owner["controller_hash"]==controller
        and center_owner["deterministic"] is True and center_owner["semantics"]=="actor_suffix",
        "Own probe continuation does not belong to the actual center parent")
    require(request["semantics"]==str(raw["semantics"])==SEMANTICS and str(raw["status"])=="complete"
        and request["elapsed"]==event["elapsed"]==elapsed and request["family"]==0
        and request["actor_hash"]==event["actor_hash"]==str(raw["actor_hash"])
        and request["actor_controller_hash"]==str(raw["actor_controller_hash"])==controller
        and request["task_hash"]==event["task_hash"]==str(raw["task_hash"])==task_hash,"Own probe frozen owner/semantics differs")
    require(request["source_episode"]==event["source_episode"] and request["center_query_ids"]==event["center_query_ids"]
        ==np.asarray(raw["center_query_ids"]).tolist()==center_ids,"Own probe center suffix lineage differs")
    require(request["center_raw"]==event["center_raw"] and request["delta"]==float(raw["delta"])==.05
        and request["local_horizon"]==int(raw["local_horizon"])==4
        and request["maximum_additional_work"]==dict(model_controls=2*R,actor_rows=2*(R-1)),"Own probe finite-difference/work declaration differs")
    for actual in (request["initial_state"],event["initial_state"],raw["initial_state"]):same(actual,state,"Own probe source state differs")
    for actual in (request["saved_mean"],event["saved_mean"],raw["saved_mean"]):same(actual,mean,"Own probe mean was not the actually executed parent action")
    same(raw["center_raw_costs"],center_costs,"Own probe center costs are not the actual suffix")
    center_full=float(center_costs.sum());center_local=float(center_costs[:min(4,R)].sum())
    for source in (request,raw):
        same(source["center_full_cost"],center_full,"Own actual center complete return differs")
        same(source["center_local_cost"],center_local,"Own actual center prefix return differs")
    pre,post,actions,costs=(np.asarray(raw[key]) for key in ("prestates","states","actions","raw_costs"))
    require(pre.shape==post.shape==(n,2) and actions.shape==(n,1) and costs.shape==(n,),"Own probe trace dimensions differ")
    first=np.asarray(raw["first_actions"]);same(request["first_actions"],first,"Own perturbation request/raw differs")
    same(raw["branch_indices"],np.repeat([0,1],R),"Own branch-major ordering differs")
    same(raw["steps"],np.tile(np.arange(R),2),"Own branch step ordering differs")
    obs=np.zeros((n,8));obs[:,0]=pre[:,1];obs[:,1]=pre[:,0]/25;obs[:,2]=1-obs[:,1]
    same(raw["observations"],obs,"Own probe observation/state alignment differs")
    for branch in (0,1):
        lo=branch*R;hi=lo+R
        same(pre[lo],state,"Own branches did not restart actual state")
        same(pre[lo+1:hi],post[lo:hi-1],"Own frozen-actor tail is discontinuous")
        same(pre[lo:hi,0],np.arange(elapsed,25),"Own probe pre-clock differs")
        same(post[lo:hi,0],np.arange(elapsed+1,26),"Own branch fails true ending")
        same(actions[lo],first[branch],"Own probe first action differs")
    ids=np.asarray(raw["query_ids"]).tolist()
    require(len(ids)==n and len(set(ids))==n and event["minus_query_ids"]==ids[:R]
        and event["plus_query_ids"]==ids[R:] and not set(ids).intersection(center_ids),"Own center/probe controls alias or are incomplete")
    for j,qid in enumerate(ids):
        branch,step=divmod(j,R);saved=query(qid);req,inp,res=saved["request"],saved["inputs"],saved["result"]
        expected=dict(semantics=SEMANTICS,actor_hash=owner,actor_controller_hash=controller,task_hash=task_hash,
            family=0,probe_id=request["probe_id"],source_episode=request["source_episode"],branch=branch,
            branch_role="minus" if branch==0 else "plus",step=step,remaining=R)
        require(req["mode"]=="model" and req["query_kind"]=="own_gradient_completion_step"
            and req["owner"]==expected,"Own probe query continuation owner differs")
        same(inp["initial_state"],pre[j],"Own probe queried state differs")
        same(inp["actions"],actions[j].reshape(1,1,1),"Own probe queried action differs")
        same(res["states"],post[j].reshape(1,1,2),"Own probe queried poststate differs")
        same(res["totals"],[costs[j]],"Own probe queried stage cost differs")
    stages=costs.reshape(2,R);full=stages.sum(axis=1);local=stages[:,:min(4,R)].sum(axis=1)
    gradients=difference_targets(mean,first,full,local)
    targets=dict(full_costs=full.tolist(),local_costs=local.tolist(),slope_full=gradients["full_gradient"],
        slope_local=gradients["local_gradient"],slope_null=0.,width=gradients["denominator"],
        curvature_full=float(full.sum()-2*center_full),curvature_local=float(local.sum()-2*center_local),
        slope_sign_disagrees=bool(np.sign(gradients["full_gradient"])!=np.sign(gradients["local_gradient"])),
        endpoint_capture=[bool(post[(branch+1)*R-1,1]<=ending_y) for branch in (0,1)],
        probes_clipped=bool(np.any((mean[None,:]+np.array([[-.05],[.05]]))<-1.) or np.any((mean[None,:]+np.array([[-.05],[.05]]))>1.)),
        postgate_invariant=bool(elapsed>=12 and np.allclose(stages[0],stages[1],atol=1e-12,rtol=0)))
    encoded=json.loads(str(raw["targets_json"]))
    for key,value in targets.items():
        if isinstance(value,(bool,list)) and (isinstance(value,bool) or (value and isinstance(value[0],bool))):
            require(encoded[key]==event["targets"][key]==value,"Own probe diagnostic flag differs: "+key)
        else:
            same(encoded[key],value,"Own raw finite-difference target differs: "+key)
            same(event["targets"][key],value,"Own event finite-difference target differs: "+key)
    if elapsed>=12:
        same([gradients["full_gradient"],gradients["local_gradient"]],[0.,0.],"Postgate action-invariant slope is nonzero")
    return dict(probe_id=request["probe_id"],source_episode=request["source_episode"],elapsed=elapsed,
        model_controls=n,actor_rows=2*(R-1),query_ids=ids,targets=targets)


def validate_own_occupancy(e,base,rows,work,result,config,checkpoints,episodes,qreq,qdone,task_hash):
    from .ending_audit import checkpoint_numpy,policy_identity,controller_identity,utc,transition_arithmetic,observations,COMPONENTS
    from .ending_completion_audit import numeric_digest
    from .ending_interpolation_audit import CHECKPOINTS,ALPHAS,tensor_interpolation
    from .ending_precision_audit import _raw,_paired,_summary_check
    required=dict(schema="bidirectional-ending-own-occupancy-assay-v0",occupancy_roots=16,occupancy_seed=181000,
        probe_delta=.05,local_horizon=4,gradient_rows=200,gradient_sampling="linspace_all_400_rows_including_postgate_zero",
        gradient_loss="mean((bounded_mean-saved_actual_parent_mean)*detached_measured_slope/25)",
        arms=["full","local","null"],alphas=ALPHAS,qualification_roots=16,qualification_seed=182000,
        qualification_noise_seed=184000,evaluation_roots=64,evaluation_seed=183000,evaluation_noise_seed=185000,
        minimum_complete_gain=.001,capture_min=.9,max_controls=39600,max_model_controls=10400,
        max_executed_controls=29200,max_actor_rows=80000,max_adam_steps=84,max_diagnostic_backward_passes=3)
    require(all(config.get(key)==value for key,value in required.items()),"Undeclared own-occupancy schedule/criterion")
    provenance=e.json("input_provenance.json")
    require(provenance["schema"]=="ending-own-occupancy-input-provenance-v1"
        and e.hashes["input_provenance.json"]==result["input_provenance_sha256"],"Own input provenance binding differs")
    refs={row["input_file"]:row for row in provenance["files"]}
    for path,row in refs.items():
        p=e.file(Path("inputs")/path,row["sha256"])
        require(p.stat().st_size==row["bytes"],"Own ancestor bytes differ")
    prior=e.json("inputs/precision/assay_result.json");pmanifest=e.json("inputs/precision/manifest.json")
    pcfg=e.json("inputs/precision/config.json");pprov=e.json("inputs/precision/input_provenance.json")
    uniform=e.json("inputs/uniform/assay_result.json");umanifest=e.json("inputs/uniform/manifest.json")
    require(prior["schema"]=="bidirectional-ending-precision-assay-v0" and prior["stage"]=="failed_precision_child_improvement"
        and prior["passed"] is False and config["task"]==pcfg["task"] and config["learning"]==pcfg["learning"],"Frozen precision ancestor changed")
    require(uniform["schema"]=="bidirectional-ending-interpolation-assay-v0"
        and uniform["stage"]=="failed_fresh_interpolation_improvement" and uniform["passed"] is False
        and uniform["raw_selected_alpha"]==.5 and uniform["best_raw_parent_fallback"] is False,"Uniform reference ancestor changed")
    for scope,saved,manifest in (("precision",prior,pmanifest),("uniform",uniform,umanifest)):
        for path,field in (("manifest.json","manifest_sha256"),("input_provenance.json","input_provenance_sha256"),
                           ("evidence/assay/ledger.jsonl","query_ledger_sha256"),("work_ledger.jsonl","work_ledger_sha256")):
            require(refs[scope+"/"+path]["sha256"]==saved[field],"Closed own ancestor ledger/result differs")
        require(refs[scope+"/config.json"]["sha256"]==manifest["config_sha256"]==saved["config_sha256"],"Own ancestor config binding differs")
    old_sources={row["file"]:row["sha256"] for row in pmanifest["files"]}
    for path,digest in provenance["precision"]["critical_source_sha256"].items():
        require(digest==old_sources["source/"+path]==e.hashes["source/"+path],"Own dynamics/observation/policy source changed")
    require(provenance["learning_source_sha256"]==old_sources["source/src/gps/bidirectional/learning.py"]
        ==e.hashes["source/src/gps/bidirectional/learning.py"],"Own base MC learner changed")
    parent=checkpoint_numpy(e.file("inputs/precision/checkpoints/precision_precision_parent.pt",CHECKPOINTS["parent"][1]))["learner"]
    ah=policy_identity(parent["policy"]);digest=numeric_digest(parent)
    require(ah==provenance["parent_actor_hash"] and digest==provenance["parent_state_sha256"]
        and numeric_digest(checkpoints["own_gradient_parent"]["learner"])==digest,"Own full parent optimizer/RNG differs")
    uniform_sha="a9ed48b00c9995d8fdae1efbb7d46ebdafcfa4f87ac2f19d86ce48f43b4f0422"
    uniform_source=checkpoint_numpy(e.file("inputs/uniform/checkpoints/interpolation_a0p5_raw.pt",uniform_sha))["actor"]
    require(policy_identity(uniform_source)==provenance["uniform_actor_hash"]
        and numeric_digest(checkpoints["own_gradient_uniform_best_raw"]["actor"])==numeric_digest(uniform_source),"Own uniform reference actor differs")
    names=[f"precision_precision_own_c{i:02d}_r{j}" for i in range(16) for j in range(4)]
    require([row["episode"] for row in provenance["own_episodes"]]==names,"Own old MC roster differs")
    old_events={};last=None
    with e.file("inputs/precision/evidence/assay/ledger.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);last=row
            if row["kind"]=="episode_outcome" and row["episode"] in names:
                require(row["episode"] not in old_events,"Duplicate reused MC episode")
                old_events[row["episode"]]=row
    require(last["kind"]=="archive_close" and set(old_events)==set(names),"Own original stochastic archive not closed")
    behavior=[];return_values=[];groups=defaultdict(list);full=[];reused_checks=0
    for row in provenance["own_episodes"]:
        event=row["original_event"];require(event==old_events[row["episode"]],"Own old MC Outcome changed")
        raw=_raw(e,Path("inputs")/row["input_file"],event["raw"]["sha256"],event["raw"]["bytes"])
        owner=json.loads(str(raw["owner_json"]));n=len(raw["raw_costs"])
        require(owner==event["owner"] and owner["actor_hash"]==ah and owner["controller_hash"]==controller_identity(owner)
            and owner["semantics"]=="actor_suffix" and owner["deterministic"] is False and bool(raw["complete"])
            and raw["states"].shape==(n+1,2) and n==25-int(raw["states"][0,0]) and raw["states"][-1,0]==25,"Own old MC continuation is incomplete/mixed")
        require(raw["pre_tanh"].shape==raw["actions"].shape==(n,1) and raw["old_log_probs"].shape==(n,)
            and np.isfinite(raw["pre_tanh"]).all() and np.isfinite(raw["old_log_probs"]).all()
            and np.allclose(raw["actions"],np.tanh(raw["pre_tanh"]),atol=2e-7,rtol=2e-12),"Own old stochastic action/latent labels differ")
        post,parts,costs,flags=transition_arithmetic(raw["states"][:-1],raw["actions"],config["task"])
        same(raw["states"][1:],post,"Own old MC state transition differs");same(raw["raw_costs"],costs,"Own old MC costs differ")
        same(raw["observations"],observations(raw["states"],25),"Own old MC observation differs")
        same(raw["cost_components"],np.stack([parts[k] for k in COMPONENTS],axis=-1),"Own old MC component cost differs")
        require(raw["query_ids"].tolist()==event["outcome"]["query_ids"] and bool(raw["captured"])==event["outcome"]["capture"]==bool(flags["ending_good"][-1]),"Own old MC ending/query lineage differs")
        same(event["outcome"]["cost"],costs.sum(),"Own old MC actual complete return differs")
        context=str(raw["context_id"]);values=np.cumsum(costs[::-1],dtype=np.float64)[::-1];full.append(values[0])
        for t,value in enumerate(values):groups[(context,0,n,n-t)].append(len(return_values));return_values.append(value)
        behavior.append(dict(observations=raw["observations"][:-1],actions=raw["actions"],pre_tanh=raw["pre_tanh"],
            log_probs=raw["old_log_probs"],raw_costs=raw["raw_costs"],context_id=context,remaining_start=n,
            behavior_policy_hash=ah,behavior_controller_hash=owner["controller_hash"],family=0,gamma=1.,complete=True,
            terminated=False,absorbing_cost=0.,continuation_owner="behavior_actor",deterministic=False,
            query_ref=str(Path(provenance["original_precision_run"])/"evidence/assay"/event["raw"]["file"])))
        reused_checks+=n
    behavior_digest=numeric_digest(behavior)
    require(behavior_digest==provenance["behavior_sha256"]==prior["arms"]["precision"]["behavior_sha256"],"Own old stochastic data digest differs")
    return_values=np.asarray(return_values);baseline=np.empty(len(return_values))
    for indices in groups.values():
        require(len(indices)==4,"Own old LOO replicate group differs")
        baseline[indices]=(return_values[indices].sum()-return_values[indices])/(len(indices)-1)
    advantages=return_values-baseline
    by_name={ep["episode"]:ep for ep in episodes};used=set();reset_checks=0
    def panel(prefix,n,seed,det,owner):
        nonlocal reset_checks
        values=[by_name[f"{prefix}_{i:03d}"] for i in range(n)]
        for i,ep in enumerate(values):
            require(ep["context_id"]==f"ordinary:{seed+i}" and ep["controls"]==25 and ep["actor_hash"]==owner
                and ep["deterministic"] is det and ep["closed_outcome"],"Own ordinary actor panel identity differs")
            raw=e.npz(base,ep["raw_ref"])
            same(raw["initial_state"],[0.,np.random.default_rng(seed+i).uniform(.2,.4)],"Own ordinary reset seed/RNG differs")
            reset_checks+=1;used.add(ep["episode"])
        return values
    occupancy=panel("own_gradient_occupancy_parent",16,181000,True,ah)
    _summary_check(result["occupancy"]["outcomes"],occupancy)
    require([row["episode"] for row in result["occupancy"]["episodes"]]==[ep["episode"] for ep in occupancy]
        and result["occupancy"]["states"]==400,"Own actual occupancy roster differs")
    center_raw={}
    for ep,record in zip(occupancy,result["occupancy"]["episodes"]):
        require(record["raw"]==ep["raw_ref"] and record["query_ids"]==ep["query_ids"],"Own occupancy raw/actual control refs differ")
        center_raw[ep["episode"]]=e.npz(base,ep["raw_ref"])
    occupancy_events=[row for row in rows if row["kind"]=="own_gradient_occupancy_complete"]
    require(len(occupancy_events)==1 and occupancy_events[0]["result"]==result["occupancy"],"Own occupancy event/result differs")
    requests={row["probe_id"]:row for row in rows if row["kind"]=="own_gradient_probe_requested"}
    done={row["probe_id"]:row for row in rows if row["kind"]=="own_gradient_probe_complete"}
    require(len(requests)==len(done)==400 and len(requests)==sum(row["kind"]=="own_gradient_probe_requested" for row in rows)
        and len(done)==sum(row["kind"]=="own_gradient_probe_complete" for row in rows)
        and not any(row["kind"]=="own_gradient_probe_error" for row in rows),"Own probes duplicated/incomplete/failed")
    measured=[];model_ids=set();states=[];observed=[];means=[];row_refs=[]
    controller=controller_identity(dict(actor_hash=ah,family=0,deterministic=True))
    for i,ep in enumerate(occupancy):
        raw=center_raw[ep["episode"]]
        for t in range(25):
            probe_id=f"probe-{25*i+t:04d}";request,event=requests[probe_id],done[probe_id]
            require(request["source_episode"]==ep["episode"] and request["elapsed"]==t
                and request["center_raw"]==ep["raw_ref"] and event["request_id"]==request["id"]
                and utc(occupancy_events[0])<utc(request)<utc(event),"Own probe actual source/chronology differs")
            def query(qid):
                require(qid in qreq and qid in qdone and qid not in model_ids
                    and utc(request)<=utc(qreq[qid])<utc(qdone[qid])<=utc(event),"Own probe model query is reused/out of order")
                model_ids.add(qid)
                return dict(request=qreq[qid],inputs=e.npz(base,qreq[qid]["inputs"]),result=e.npz(base,qdone[qid]["result"]))
            checked=validate_probe(e.npz(base,event["raw"]),request,event,raw,t,query,task_hash,controller,config["task"]["ending_y"])
            nn=[row for row in rows if row["kind"]=="neural_requested" and utc(request)<=utc(row)<=utc(event)]
            require(sum(row["rows"] for row in nn)==checked["actor_rows"]
                and all(row["scope"]=="own_gradient_measured_slopes" for row in nn),"Own probe actor work is outside registered complete branches")
            measured.append(checked);states.append(raw["states"][t]);observed.append(raw["observations"][t]);means.append(raw["actions"][t])
            row_refs.append(dict(semantics=SEMANTICS,actor_hash=ah,task_hash=task_hash,elapsed=t,state=raw["states"][t].tolist(),
                source_episode=ep["episode"],center_query_ids=ep["query_ids"][t:],minus_query_ids=event["minus_query_ids"],
                plus_query_ids=event["plus_query_ids"],center_raw=ep["raw_ref"],slope_raw=event["raw"]))
    require(model_ids=={qid for qid,row in qreq.items() if row["mode"]=="model"},"Unbound own-occupancy model controls")
    gradients=result["gradients"];batch=e.npz(base,gradients["raw"]);indices=np.linspace(0,399,200,dtype=np.int64)
    require(gradients["full_rows"]==400 and gradients["selected_rows"]==200 and gradients["weights"]=="all1_including_null"
        and gradients["actor_hash"]==str(batch["actor_hash"])==ah and gradients["task_hash"]==str(batch["task_hash"])==task_hash
        and gradients["objective"]==str(batch["semantics"])==SEMANTICS and gradients["fixed_delta"]==.05
        and gradients["local_horizon"]==4 and gradients["targets_are_updated_actor_returns"] is False
        and gradients["teacher_actions_used"] is False,"Own gradient object has wrong owner/objective")
    same(batch["all_states"],np.stack(states),"Own all-state gradient coverage differs")
    same(batch["all_observations"],np.stack(observed),"Own all-observation gradient alignment differs")
    same(batch["all_saved_means"],np.stack(means),"Own all-saved-mean action differs")
    same(batch["all_weights"],np.ones(400),"Own gradients dropped/reweighted original rows")
    same(batch["training_indices"],indices,"Own gradient fixed subsample differs")
    require(json.loads(str(batch["all_row_references_json"]))==row_refs,"Own actual/probe training ancestry differs")
    for field,values in (("states",states),("observations",observed),("saved_means",means)):
        same(batch["selected_"+field],np.stack(values)[indices],"Own selected gradient data differs")
    same(batch["selected_families"],np.zeros(200,dtype=int),"Own gradient family differs")
    same(batch["selected_weights"],np.ones(200),"Own/null unit weighting differs")
    for arm in ("full","local","null"):
        slope=np.array([row["targets"]["slope_"+arm] for row in measured])[:,None]
        same(batch["all_slope_"+arm],slope,"Own full/local/null target slope differs")
    fields=("probe_id","actor_hash","task_hash","source_episode","elapsed","initial_state","saved_mean", "center_query_ids","minus_query_ids","plus_query_ids","center_raw","raw","targets")
    require(gradients["probes"]==[{key:done[f"probe-{i:04d}"][key] for key in fields} for i in range(400)],"Own probe result summaries differ")
    credit_events=[row for row in rows if row["kind"]=="own_gradient_credit_complete"]
    require(len(credit_events)==1 and credit_events[0]["result"]==gradients,"Own credit event/result differs")
    fits=[row for row in rows if row["kind"]=="fit_work"]
    require({row["purpose"] for row in fits}=={"own_gradient_fit_"+arm for arm in ("full","local","null")} and len(fits)==3,"Own fit roster differs")
    steps=4*math.ceil(len(return_values)/256)
    for fit in fits:
        arm=fit["gradient_kind"];child=checkpoints["own_gradient_child_"+arm]
        require(fit["complete_episodes"]==64 and fit["controls"]==len(return_values) and fit["behavior_policy_hash"]==ah
            and fit["target_owner"]=="behavior_actor" and fit["gradient_steps"]==steps
            and fit["auxiliary_loss_semantics"]==SEMANTICS and fit["auxiliary_is_action_target_regression"] is False
            and fit["gradient_batch_raw"]==gradients["raw"] and fit["unit_weight_rows"]==200
            and child["parent_actor_hash"]==ah and child["parent_state_sha256"]==digest
            and child["behavior_sha256"]==behavior_digest and child["gradient_batch_raw"]==gradients["raw"]
            and child["gradient_kind"]==arm and child["own_gradient_task_hash"]==task_hash
            and utc(fit)>utc(credit_events[0]),"Own fits changed parent/MC ownership/gradient object")
        for key,wanted in dict(mc_return_mean=return_values.mean(),raw_mc_advantage_std=advantages.std(),
            raw_mc_advantage_range=[advantages.min(),advantages.max()],episode_full_cost_range=[min(full),max(full)]).items():
            same(fit[key],wanted,"Own original MC/LOO summary differs")
    require(len({numeric_digest(checkpoints["own_gradient_child_"+arm]["learner"]["numpy_rng"]) for arm in ("full","local","null")})==1
        and result["matched_realized_sampler_schedule"] is True,"Own realized sampler streams diverged")
    cp={row["label"]:row for row in rows if row["kind"]=="checkpoint"}
    first_qual=min(utc(row) for row in rows if row["kind"]=="neural_requested" and str(row.get("scope","")).startswith("own_gradient_qualification"))
    for alpha in ALPHAS:
        label="own_gradient_a"+str(alpha).replace(".","p")
        for arm in ("full","local","null"):
            saved=checkpoints[label+"_"+arm];child=checkpoints["own_gradient_child_"+arm]["learner"]["policy"]
            require(saved["alpha"]==alpha and saved["parent_actor_hash"]==ah and saved["source_child_actor_hash"]==child["policy_hash"]
                and utc(cp[label+"_"+arm])<first_qual,"Own alpha construction/qualification chronology differs")
            tensor_interpolation(parent["policy"],child,saved["actor"],alpha,saved["parameter_names"])
    parents={mode:panel("own_gradient_qual_parent_"+mode,16,182000,mode=="mean",ah) for mode in ("mean","stochastic")}
    for mode in parents:_summary_check(result["qualification_parent"][mode],parents[mode])
    def comparison(actual,baseline,proposed):
        wanted=_paired(baseline,proposed,.001)
        for key in ("mean_cost_gain","paired_cost_gains"):same(actual[key],wanted[key],"Own evaluated gain differs")
        require(actual["capture_nonregression"]==wanted["capture_nonregression"] and actual["improved"]==wanted["improved"],"Own evaluated capture/gain criterion differs")
        return wanted["improved"]
    def assess_full(saved,mean,noisy,final=False):
        decisions=[];roles=("parent","best_local","best_null","uniform_best_raw") if final else ("parent","local","null")
        for values,mode in ((mean,"mean"),(noisy,"stochastic")):
            for role in roles:decisions.append(comparison(saved["comparisons"][mode][role],values[role],values["full"]))
        capture=float(np.mean([ep["capture"] for ep in mean["full"]]));passed=bool(all(decisions) and capture>=.9)
        same(saved["deterministic_capture_rate"],capture,"Own full capture result differs")
        require(saved["passed"]==passed,"Own full scientific criterion differs")
        return passed
    def assess_control(saved,mean,noisy,arm):
        a=comparison(saved["mean_vs_parent"],mean["parent"],mean[arm]);b=comparison(saved["stochastic_vs_parent"],noisy["parent"],noisy[arm])
        capture=float(np.mean([ep["capture"] for ep in mean[arm]]));passed=bool(a and b and capture>=.9)
        same(saved["deterministic_capture_rate"],capture,"Own control capture result differs")
        require(saved["passed"]==passed,"Own independent control eligibility differs")
        return passed
    eligible={arm:[] for arm in ("full","local","null")};qual_events=[row for row in rows if row["kind"]=="own_gradient_qualification"]
    require([row["alpha"] for row in result["qualification"]]==ALPHAS and len(qual_events)==5,"Own alpha qualification roster differs")
    for i,row in enumerate(result["qualification"]):
        alpha=row["alpha"];label="own_gradient_a"+str(alpha).replace(".","p")
        mean={"parent":parents["mean"]};noisy={"parent":parents["stochastic"]}
        for arm in ("full","local","null"):
            owner=checkpoints[label+"_"+arm]["actor"]["policy_hash"]
            require(row["actor_hashes"][arm]==owner,"Own alpha result actor hash differs")
            tag="own_gradient_qual_a"+str(alpha).replace(".","p")+"_"+arm
            mean[arm]=panel(tag+"_mean",16,182000,True,owner);noisy[arm]=panel(tag+"_stochastic",16,182000,False,owner)
            _summary_check(row["mean_panels"][arm],mean[arm]);_summary_check(row["stochastic_panels"][arm],noisy[arm])
            same(row["mean_costs"][arm],np.mean([ep["cost"] for ep in mean[arm]]),"Own alpha ranking score differs")
        require(qual_events[i]["result"]==row,"Own alpha qualification event/result differs")
        if assess_full(row["criteria"]["full"],mean,noisy):eligible["full"].append(row)
        for arm in ("local","null"):
            if assess_control(row["criteria"][arm],mean,noisy,arm):eligible[arm].append(row)
    selected={arm:None if not eligible[arm] else min(eligible[arm],key=lambda row:(row["mean_costs"][arm],row["alpha"]))["alpha"] for arm in eligible}
    decision=[row for row in rows if row["kind"]=="own_gradient_selection"]
    require(len(decision)==1 and decision[0]["selected_steps"]==result["selected_steps"]==selected
        and decision[0]["final_evidence_not_seen"] is True and all(utc(row)<utc(decision[0]) for row in qual_events),"Own qualification-only step selection differs")
    if selected["full"] is None:
        require(result["passed"] is False and result["stage"]=="failed_all_own_gradient_qualification" and "final" not in result,"Failed own step ladder did not stop")
    else:
        def owner(arm):
            return ah if selected[arm] is None else checkpoints["own_gradient_a"+str(selected[arm]).replace(".","p")+"_"+arm]["actor"]["policy_hash"]
        owners=dict(parent=ah,full=owner("full"),best_local=owner("local"),best_null=owner("null"),uniform_best_raw=uniform_source["policy_hash"])
        require(result["final_actor_hashes"]==owners,"Own fresh final five-role controller identity differs")
        final={mode:{role:panel("own_gradient_final_"+role+"_"+mode,64,183000,mode=="mean",ahash) for role,ahash in owners.items()} for mode in ("mean","stochastic")}
        for mode,values in final.items():
            for role,value in values.items():_summary_check(result["final"]["mean_panels" if mode=="mean" else "stochastic_panels"][role],value)
        first_final=min(utc(row) for row in rows if row["kind"] in ("neural_requested","query_requested") and row.get("scope")=="own_gradient_final_fresh_evaluation")
        require(utc(decision[0])<first_final,"Own final evidence preceded step selection")
        passed=assess_full(result["final"]["criteria"],final["mean"],final["stochastic"],True)
        require(result["passed"]==passed and result["stage"]==("complete" if passed else "failed_fresh_own_gradient_improvement"),"Own fresh final verdict differs")
    require(set(by_name)==used and len(episodes)==(528 if selected["full"] is None else 1168),"Unexpected/missing own actual episode roster")
    live=13200 if selected["full"] is None else 29200
    expected_nn=400+9600+3*(len(return_values)+4*len(return_values)+steps*200)+live-400
    accounting=result["accounting"]
    require(accounting["model_controls"]==sum(row["model_controls"] for row in measured)==10400
        and accounting["executed_controls"]==live and accounting["requested_scalar_controls"]==10400+live
        and accounting["actor_rows"]==expected_nn and accounting["adam_attempts"]==3*steps
        and accounting["diagnostic_backward_attempts"]==3,"Own paid model/live/neural/update schedule differs")
    cross=result["cost_crosswalk"];lineage=(accounting,prior["accounting"],pprov["reused_cost_ancestry"],uniform["accounting"])
    require(cross["new_assay"]==lineage[0] and cross["precision_ancestor"]==lineage[1]
        and cross["v3_ancestor"]==lineage[2] and cross["uniform_step_ancestor"]==lineage[3],"Own shared paid ancestry differs")
    for key in ("requested_scalar_controls","actor_rows","adam_attempts"):
        require(cross["unique_lineage_"+key]==sum(row[key] for row in lineage),"Own unique paid ancestry double counts/omits work")
    require(result["no_teacher_action_targets"] is True and result["no_new_discovery"] is True
        and result["full_feedback_established"] is False and result["bidirectional_data_benefit_tested"] is False
        and result["efficiency_claim"] is False and result["no_deployment_promotion"] is True,"Unsupported own-occupancy scope claim")
    return dict(probes=measured,actual_occupancy_episodes=16,actual_occupancy_states=400,training_rows=200,
        old_mc_episodes=64,old_mc_controls=len(return_values),behavior_sha256=behavior_digest,selected_steps=selected,
        model_controls=10400,live_controls=live,analytic_reused_episode_transition_row_checks=reused_checks,
        analytic_reset_draw_checks=reset_checks,new_physics_controls=0,new_actor_rows=0,new_optimizer_steps=0,
        limitations=["Saved mean/tail actions and MC likelihoods are source/checkpoint bound; no neural forward was rerun.",
            "Full and H4 targets evaluate one perturbed action then frozen deterministic parent, not the changed actor or replanning controller.",
            "A finite difference across a discontinuity is not certified as a derivative; actual complete changed-actor outcomes decide success."])
