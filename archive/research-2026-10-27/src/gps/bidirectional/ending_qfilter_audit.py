"""Independent saved-Q/owned-MC/filtered-label audit; NumPy and stdlib only."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .ending_completion_audit import numeric_digest
from .ending_interpolation_audit import CHECKPOINTS,ALPHAS,tensor_interpolation
from .ending_precision_audit import require,same,_paired,_summary_check,_raw


UNIFORM_SHA="a9ed48b00c9995d8fdae1efbb7d46ebdafcfa4f87ac2f19d86ce48f43b4f0422"
SEMANTICS="one_actual_teacher_action_then_complete_deterministic_frozen_parent"


def exact_pair_key(state,action,owner,task):
    digest=hashlib.sha256(json.dumps(dict(owner=owner,task=task,family=0),sort_keys=True).encode())
    digest.update(np.ascontiguousarray(state,dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(action,dtype=np.float64).tobytes())
    return digest.hexdigest()


def validate_pair(raw,request,event,source_state,teacher_action,query,task_hash,controller):
    """Link two complete saved branches to already-arithmetic-audited H1 queries."""
    owner=request["actor_hash"];pair=request["pair_id"];key=exact_pair_key(source_state,teacher_action,owner,task_hash)
    R=25-int(source_state[0]);n=2*R
    require(request["pair_key"]==event["pair_key"]==str(raw["pair_key"])==key
        and request["remaining"]==event["remaining"]==int(raw["remaining"])==R
        and request["semantics"]==SEMANTICS and str(raw["status"])=="complete","Q pair semantics/key/horizon differs")
    require(request["family"]==0 and request["task_hash"]==event["task_hash"]==str(raw["task_hash"])==task_hash
        and event["actor_hash"]==str(raw["actor_hash"])==owner
        and request["actor_controller_hash"]==str(raw["actor_controller_hash"])==controller,"Q pair continuation owner differs")
    require(request["maximum_additional_work"]==dict(model_controls=2*R,actor_rows=1+2*(R-1)),"Q pair work ceiling differs")
    same(request["initial_state"],source_state,"Q pair source prestate differs")
    same(raw["initial_state"],source_state,"Q raw source prestate differs")
    same(request["teacher_action"],teacher_action,"Q request relabeled teacher action")
    same(raw["branch_indices"],np.repeat([0,1],R),"Q branch-major indices differ")
    same(raw["steps"],np.tile(np.arange(R),2),"Q branch step indices differ")
    pre,post,actions,costs=(np.asarray(raw[key]) for key in ("prestates","states","actions","raw_costs"))
    first=np.asarray(raw["first_actions"])
    require(pre.shape==post.shape==(n,2) and actions.shape==(n,1) and costs.shape==(n,)
        and first.shape==(2,1) and np.isfinite(first).all() and np.all(np.abs(first)<=1),"Q trace array shape/bounds differ")
    obs=np.zeros((n,8));obs[:,0]=pre[:,1];obs[:,1]=pre[:,0]/25;obs[:,2]=1-obs[:,1]
    same(raw["observations"],obs,"Q observations differ from saved prestate")
    same(first[1],teacher_action,"Q teacher-first action was not the actual label")
    for branch in range(2):
        lo=branch*R;hi=lo+R
        same(pre[lo],source_state,"Q branches do not share true initial state")
        same(pre[lo+1:hi],post[lo:hi-1],"Q tail state continuity differs")
        same(pre[lo:hi,0],np.arange(int(source_state[0]),25),"Q pre-clock differs")
        same(post[lo:hi,0],np.arange(int(source_state[0])+1,26),"Q branch did not reach true ending")
        same(actions[lo],first[branch],"Q branch first action differs")
    qids=np.asarray(raw["query_ids"]).tolist()
    require(len(qids)==n and len(set(qids))==n and event["query_ids"]==qids,"Q controls reused/missing")
    for j,qid in enumerate(qids):
        branch,step=divmod(j,R);saved=query(qid);req,inp,res=saved["request"],saved["inputs"],saved["result"]
        expected_owner=dict(semantics=SEMANTICS,actor_hash=owner,actor_controller_hash=controller,
            task_hash=task_hash,family=0,pair_id=pair,pair_key=key,branch=branch,
            branch_role="actor_mean" if branch==0 else "actual_teacher",step=step,remaining=R)
        require(req["mode"]=="model" and req["query_kind"]=="qfilter_actor_completion_step"
            and req["owner"]==expected_owner,"Q query first-action/tail owner differs")
        same(inp["initial_state"],pre[j],"Q query prestate differs")
        same(inp["actions"],actions[j].reshape(1,1,1),"Q query action differs")
        same(res["states"],post[j].reshape(1,1,2),"Q query poststate differs")
        same(res["totals"],[costs[j]],"Q query raw cost differs")
    totals=costs.reshape(2,R).sum(axis=1)
    same(raw["complete_branch_costs"],totals,"Complete Q branch cost differs")
    same([event["q_actor"],event["q_teacher"]],totals,"Q event costs differ")
    same(event["weight"],max(float(totals[0]-totals[1]),0.),"Q positive weight differs")
    return dict(pair_id=pair,pair_key=key,remaining=R,model_controls=n,actor_rows=1+2*(R-1),
        weight=float(event["weight"]),q_actor=float(totals[0]),q_teacher=float(totals[1]),query_ids=qids)


def positive_map(weights):
    weights=np.asarray(weights)
    require(weights.shape==(200,) and np.isfinite(weights).all() and np.all(weights>=0),"Invalid Q weights")
    positive=np.flatnonzero(weights>0)
    indices=np.arange(200,dtype=np.int64) if not len(positive) else positive[np.linspace(0,len(positive)-1,200,dtype=np.int64)]
    return positive,indices


def validate_qfilter_assay(e,base,rows,work,result,config,checkpoints,episodes,qreq,qdone,task_hash):
    from .ending_audit import checkpoint_numpy,policy_identity,controller_identity,utc,transition_arithmetic,observations,COMPONENTS,canonical
    expected=dict(schema="bidirectional-ending-qfilter-assay-v0",source_arm="precision",auxiliary_rows=200,
        positive_resampling="linspace_indices_over_original_positive_rows_to_200",q_semantics=SEMANTICS,
        weight_rule="max(Q_parent_mean-Q_actual_teacher,0)",cache_rule="exact_float64_state_action_bytes+parent_actor_hash+task_hash+family",
        alphas=ALPHAS,qualification_roots=16,qualification_seed=171000,qualification_noise_seed=173000,
        evaluation_roots=64,evaluation_seed=172000,evaluation_noise_seed=174000,
        minimum_complete_gain=.001,capture_min=.9,max_controls=34800,max_model_controls=10000,
        max_executed_controls=24800,max_actor_rows=61800,max_adam_steps=56,max_diagnostic_backward_passes=2)
    require(all(config.get(k)==v for k,v in expected.items()),"Undeclared Q-filter protocol")
    provenance=e.json("input_provenance.json")
    require(e.hashes["input_provenance.json"]==result["input_provenance_sha256"],"Q provenance result binding differs")
    refs={row["input_file"]:row for row in provenance["files"]}
    for path,row in refs.items():
        file=e.file(Path("inputs")/path,row["sha256"])
        require(file.stat().st_size==row["bytes"],"Q ancestor input bytes differ")
    prior=e.json("inputs/precision/assay_result.json");pmanifest=e.json("inputs/precision/manifest.json")
    pcfg=e.json("inputs/precision/config.json");pprov=e.json("inputs/precision/input_provenance.json")
    uniform=e.json("inputs/uniform/assay_result.json");umanifest=e.json("inputs/uniform/manifest.json")
    ucfg=e.json("inputs/uniform/config.json");uprov=e.json("inputs/uniform/input_provenance.json")
    require(prior["schema"]=="bidirectional-ending-precision-assay-v0" and prior["passed"] is False
        and prior["stage"]=="failed_precision_child_improvement" and pcfg["task"]==config["task"]
        and pcfg["learning"]==config["learning"],"Precision ancestor changed")
    require(uniform["schema"]=="bidirectional-ending-interpolation-assay-v0" and uniform["passed"] is False
        and uniform["stage"]=="failed_fresh_interpolation_improvement" and uniform["raw_selected_alpha"]==.5
        and uniform["best_raw_parent_fallback"] is False and ucfg["task"]==config["task"],"Uniform-data reference was substituted")
    for scope,saved,manifest in (("precision",prior,pmanifest),("uniform",uniform,umanifest)):
        for path,field in (("manifest.json","manifest_sha256"),("input_provenance.json","input_provenance_sha256"),
                           ("evidence/assay/ledger.jsonl","query_ledger_sha256"),("work_ledger.jsonl","work_ledger_sha256")):
            require(refs[scope+"/"+path]["sha256"]==saved[field],"Closed ancestor binding differs")
        require(refs[scope+"/config.json"]["sha256"]==manifest["config_sha256"]==saved["config_sha256"],"Ancestor config binding differs")
    old_sources={row["file"]:row["sha256"] for row in pmanifest["files"]}
    for path,digest in provenance["precision"]["critical_source_sha256"].items():
        require(digest==old_sources["source/"+path]==e.hashes["source/"+path],"Q task/observation/policy source changed")
    require(provenance["learning_source_sha256"]==old_sources["source/src/gps/bidirectional/learning.py"]
        ==e.hashes["source/src/gps/bidirectional/learning.py"],"Q learner implementation changed")
    parent_file="inputs/precision/checkpoints/precision_precision_parent.pt"
    parent=checkpoint_numpy(e.file(parent_file,CHECKPOINTS["parent"][1]))["learner"]
    ah=policy_identity(parent["policy"]);parent_digest=numeric_digest(parent)
    require(ah==provenance["parent_actor_hash"] and parent_digest==provenance["parent_state_sha256"]
        and numeric_digest(checkpoints["qfilter_parent"]["learner"])==parent_digest,"Full frozen Q parent/optimizer/RNG differs")
    require(provenance["precision"]["checkpoints"]["parent"]["sha256"]==CHECKPOINTS["parent"][1],"Wrong original precision parent")
    uniform_source=checkpoint_numpy(e.file("inputs/uniform/checkpoints/interpolation_a0p5_raw.pt",UNIFORM_SHA))["actor"]
    require(policy_identity(uniform_source)==provenance["uniform_actor_hash"]
        and numeric_digest(checkpoints["qfilter_uniform_best_raw"]["actor"])==numeric_digest(uniform_source),"Uniform selected actor/RNG differs")
    names=[f"precision_precision_own_c{i:02d}_r{j}" for i in range(16) for j in range(4)]
    require([row["episode"] for row in provenance["own_episodes"]]==names,"Reused MC roster differs or includes held outcomes")
    required_teacher={f"teacher_{i:03d}" for i in range(8)}|{f"completion_r0_acquisition_{arm}_{i:03d}" for arm in ("raw","guided") for i in range(8)}
    require({row["episode"] for row in provenance["teacher_episodes"]}==required_teacher
        and len(provenance["teacher_episodes"])==24,"Teacher roster includes held outcomes or omitted sources")
    old_outcomes={};old_arm=None;last=None
    with e.file("inputs/precision/evidence/assay/ledger.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);last=row
            if row["kind"]=="episode_outcome" and row["episode"] in names:old_outcomes[row["episode"]]=row
            if row["kind"]=="precision_arm_result" and row["arm"]=="precision":old_arm=row
    require(last["kind"]=="archive_close" and set(old_outcomes)==set(names)
        and old_arm["result"]=={k:v for k,v in prior["arms"]["precision"].items() if k!="costs"},"Closed original MC source result differs")
    own=[];behavior=[];returns=[];return_groups=defaultdict(list);full=[];reused_checks=0
    for row in provenance["own_episodes"]:
        event=row["original_event"];require(event==old_outcomes[row["episode"]],"Reused own outcome differs")
        raw=_raw(e,Path("inputs")/row["input_file"],event["raw"]["sha256"],event["raw"]["bytes"])
        owner=json.loads(str(raw["owner_json"]));n=len(raw["raw_costs"])
        require(owner==event["owner"] and owner["actor_hash"]==ah and owner["controller_hash"]==controller_identity(owner)
            and owner["semantics"]=="actor_suffix" and owner["deterministic"] is False and bool(raw["complete"])
            and n==25-int(raw["states"][0,0]) and raw["states"][-1,0]==25,"Reused MC continuation is incomplete or not exact parent-owned")
        post,parts,cost,flags=transition_arithmetic(raw["states"][:-1],raw["actions"],config["task"])
        same(raw["states"][1:],post,"Reused own MC transition differs");same(raw["raw_costs"],cost,"Reused own MC cost differs")
        same(raw["cost_components"],np.stack([parts[k] for k in COMPONENTS],axis=-1),"Reused MC components differ")
        same(raw["observations"],observations(raw["states"],25),"Reused MC observations differ")
        require(raw["query_ids"].tolist()==event["outcome"]["query_ids"] and bool(raw["captured"])==event["outcome"]["capture"]==bool(flags["ending_good"][-1]),"Reused MC source control/ending differs")
        same(event["outcome"]["cost"],cost.sum(),"Reused MC complete Outcome return differs")
        context=str(raw["context_id"]);values=np.cumsum(cost[::-1],dtype=np.float64)[::-1];full.append(values[0])
        for t,value in enumerate(values):return_groups[(context,0,n,n-t)].append(len(returns));returns.append(value)
        behavior.append(dict(observations=raw["observations"][:-1],actions=raw["actions"],pre_tanh=raw["pre_tanh"],
            log_probs=raw["old_log_probs"],raw_costs=raw["raw_costs"],context_id=context,remaining_start=n,
            behavior_policy_hash=ah,behavior_controller_hash=owner["controller_hash"],family=0,gamma=1.,complete=True,
            terminated=False,absorbing_cost=0.,continuation_owner="behavior_actor",deterministic=False,
            query_ref=str(Path(provenance["original_precision_run"])/"evidence/assay"/event["raw"]["file"])))
        own.append(raw);reused_checks+=n
    behavior_digest=numeric_digest(behavior)
    require(behavior_digest==provenance["behavior_sha256"]==prior["arms"]["precision"]["behavior_sha256"],"Reused raw MC digest differs")
    values=np.asarray(returns);baseline=np.empty(len(values))
    for indices in return_groups.values():
        require(len(indices)==4,"Reused MC LOO replicate group differs")
        baseline[indices]=(values[indices].sum()-values[indices])/(len(indices)-1)
    advantages=values-baseline
    teachers={role:[] for role in ("shared","raw","guided")}
    v3_episodes={row["episode"]:row for row in pprov["episodes"]}
    for ref in provenance["teacher_episodes"]:
        original=v3_episodes[ref["episode"]]
        require(all(ref[key]==original[key] for key in original),"Teacher provenance differs from v3 copy")
        raw=_raw(e,Path("inputs")/ref["input_file"],ref["sha256"],ref["bytes"])
        post,parts,cost,flags=transition_arithmetic(raw["states"][:-1],raw["actions"],config["task"])
        require(raw["states"].shape==(26,2) and bool(raw["complete"])
            and json.loads(str(raw["owner_json"]))["semantics"]=="executed_replanning","Q teacher was not complete actual execution")
        same(raw["states"][1:],post,"Reused teacher transition differs");same(raw["raw_costs"],cost,"Reused teacher cost differs")
        same(raw["observations"],observations(raw["states"],25),"Reused teacher observations differ")
        require(bool(raw["captured"])==ref["capture"]==bool(flags["ending_good"][-1]),"Reused teacher success differs")
        teachers[ref["role"]].append((ref,raw));reused_checks+=25
    aux_ref=refs[provenance["auxiliary_input_file"]]
    original_aux=_raw(e,Path("inputs")/provenance["auxiliary_input_file"],aux_ref["sha256"],aux_ref["bytes"])
    credit=e.npz(base,result["qfilter"]["raw"]);shared=[pair for pair in teachers["shared"] if bool(pair[1]["captured"])]
    require(bool(shared),"Common actual teacher stratum missing")
    source_rows={};keys={};pairs={};expected_aliases={}
    for arm in ("raw","guided"):
        specific=[pair for pair in teachers[arm] if bool(pair[1]["captured"])];selected=[]
        require(bool(original_aux[arm+"_specific_is_shared_fallback"])==(not bool(specific)),"Original teacher fallback differs")
        for stratum,traces in (("common",shared),("specific",specific or shared)):
            indices=np.linspace(0,25*len(traces)-1,100,dtype=np.int64)
            same(original_aux[arm+"_"+stratum],indices,"Original auxiliary index strata differ")
            for flat in indices:
                ref,raw=traces[int(flat)//25];t=int(flat)%25
                selected.append(dict(state=raw["states"][t],observation=raw["observations"][t],action=raw["actions"][t],
                    elapsed=t,stratum=stratum,source_episode=ref["episode"],source_input_file=ref["input_file"],source_sha256=ref["sha256"]))
        source_rows[arm]=selected;keys[arm]=[]
        for field in ("observations","actions","families"):
            expected_array=np.zeros(200,dtype=int) if field=="families" else np.stack([row["observation" if field=="observations" else "action"] for row in selected])
            same(original_aux[arm+"_"+field],expected_array,"Original auxiliary pre-action alignment differs")
            same(credit[arm+"_original_"+field],expected_array,"Q filtering changed original teacher labels")
        same(credit[arm+"_source_states"],np.stack([row["state"] for row in selected]),"Q teacher source prestates differ")
        descriptors=[{k:v for k,v in row.items() if k not in ("state","observation","action")} for row in selected]
        require(json.loads(str(credit[arm+"_source_rows_json"]))==descriptors,"Q source row aliases differ")
        for i,row in enumerate(selected):
            key=exact_pair_key(row["state"],row["action"],ah,task_hash);keys[arm].append(key)
            if key not in pairs:pairs[key]=dict(pair_id=f"pair-{len(pairs):04d}",row=row)
            expected_aliases.setdefault(key,[]).append(dict(arm=arm,row=i,source_episode=row["source_episode"],elapsed=row["elapsed"],source_sha256=row["source_sha256"],stratum=row["stratum"]))
    require(len(pairs)<=200,"Unique paid Q pair cap exceeded")
    source_events=[row for row in rows if row["kind"]=="qfilter_pair_sources"]
    require(len(source_events)==1 and source_events[0]["pairs"]==[dict(pair_key=key,pair_id=value["pair_id"],aliases=expected_aliases[key]) for key,value in pairs.items()]
        and source_events[0]["logical_rows"]==400 and source_events[0]["unique_pairs"]==len(pairs),"Original400 row cache/alias roster differs")
    requests={row["pair_id"]:row for row in rows if row["kind"]=="qfilter_pair_requested"}
    done={row["pair_id"]:row for row in rows if row["kind"]=="qfilter_pair_complete"}
    require(len(requests)==sum(row["kind"]=="qfilter_pair_requested" for row in rows)
        and len(done)==sum(row["kind"]=="qfilter_pair_complete" for row in rows)
        and len(requests)==len(done)==len(pairs) and not any(row["kind"]=="qfilter_pair_error" for row in rows),"Duplicate/incomplete/error/missing paid Q pairs")
    measured={};used_model=set();controller=controller_identity(dict(actor_hash=ah,family=0,deterministic=True))
    for key,pair in pairs.items():
        request=requests[pair["pair_id"]];event=done[pair["pair_id"]]
        require(event["request_id"]==request["id"] and utc(source_events[0])<utc(request)<utc(event),"Q request/source chronology differs")
        def query(qid):
            require(qid not in used_model and qid in qreq and qid in qdone
                and utc(request)<=utc(qreq[qid])<utc(qdone[qid])<=utc(event),"Q query ownership/time/reuse differs")
            used_model.add(qid)
            return dict(request=qreq[qid],inputs=e.npz(base,qreq[qid]["inputs"]),result=e.npz(base,qdone[qid]["result"]))
        measured[key]=validate_pair(e.npz(base,event["raw"]),request,event,pair["row"]["state"],pair["row"]["action"],query,task_hash,controller)
        neural=[row for row in rows if row["kind"]=="neural_requested" and utc(request)<=utc(row)<=utc(event)]
        require(sum(row["rows"] for row in neural)==measured[key]["actor_rows"]
            and all(row["scope"]=="qfilter_measured_credit" for row in neural),"Q actor work falls outside before-call pair request or differs from full tails")
    require(used_model=={qid for qid,row in qreq.items() if row["mode"]=="model"},"Unbound extra Q/model control")
    record_fields=("pair_id","pair_key","actor_hash","task_hash","q_actor","q_teacher","weight","raw","query_ids","remaining")
    require(result["qfilter"]["pairs"]==[{key:done[pair["pair_id"]][key] for key in record_fields} for pair in pairs.values()],"Q pair result summaries differ")
    credit_events=[row for row in rows if row["kind"]=="qfilter_credit_complete"]
    require(len(credit_events)==1 and credit_events[0]["result"]==result["qfilter"],"Q credit event/result differs")
    for arm in ("raw","guided"):
        weights=np.array([measured[key]["weight"] for key in keys[arm]]);positive,indices=positive_map(weights)
        same(credit[arm+"_original_weights"],weights,"Q per-label weights differ")
        require(credit[arm+"_pair_ids"].tolist()==[pairs[key]["pair_id"] for key in keys[arm]],"Q original-label cache references differ")
        same(credit[arm+"_filtered_positive_original_indices"],positive,"Q positive row mask differs")
        same(credit[arm+"_filtered_original_indices"],indices,"Q fixed resampling index map differs")
        for field in ("observations","actions","families"):
            same(credit[arm+"_filtered_"+field],credit[arm+"_original_"+field][indices],"Q filtered labels do not equal actual selected rows")
        same(credit[arm+"_filtered_weights"],weights[indices],"Q resampled κ differs")
        reported=result["qfilter"]["arms"][arm]
        require(reported["positive_original_rows"]==len(positive) and reported["no_auxiliary_signal"]==(not len(positive))
            and reported["original_rows"]==reported["filtered_rows"]==200,"Q filtered row summary differs")
        same(reported["weight_sum"],weights.sum(),"Q weight sum summary differs")
    require(result["qfilter"]["unique_pairs"]==len(pairs) and result["qfilter"]["logical_rows"]==400,"Q credit result roster differs")
    fits=[row for row in rows if row["kind"]=="fit_work"]
    require({row["purpose"] for row in fits}=={"qfilter_child_raw","qfilter_child_guided"} and len(fits)==2,"Unexpected extra/missing Q-filter fits")
    expected_steps=4*math.ceil(len(values)/256)
    for fit in fits:
        arm=fit["purpose"].rsplit("_",1)[1];child=checkpoints["qfilter_child_"+arm]
        require(fit["complete_episodes"]==64 and fit["controls"]==len(values) and fit["behavior_policy_hash"]==ah
            and fit["target_owner"]=="behavior_actor" and fit["gradient_steps"]==expected_steps
            and child["parent_actor_hash"]==ah and child["parent_state_sha256"]==parent_digest
            and child["behavior_sha256"]==behavior_digest and child["auxiliary_raw"]==result["qfilter"]["raw"],"Q-filter child did not use exact matched old own MC/full parent")
        for key,wanted in dict(mc_return_mean=values.mean(),raw_mc_advantage_std=advantages.std(),
            raw_mc_advantage_range=[advantages.min(),advantages.max()],episode_full_cost_range=[min(full),max(full)]).items():
            same(fit[key],wanted,"Reused own MC/LOO fit summary differs")
        require(utc(fit)>max(utc(event) for event in done.values()),"Fitting preceded complete paid Q credit")
    cp_events={row["label"]:row for row in rows if row["kind"]=="checkpoint"}
    require(utc(cp_events["qfilter_parent"])<min(utc(row) for row in requests.values()),"Measured Q predates frozen parent checkpoint")
    first_qual=min(utc(row) for row in rows if row["kind"]=="neural_requested" and str(row.get("scope","")).startswith("qfilter_qualification"))
    for alpha in ALPHAS:
        label="qfilter_a"+str(alpha).replace(".","p")
        for arm in ("raw","guided"):
            saved=checkpoints[label+"_"+arm];child=checkpoints["qfilter_child_"+arm]["learner"]["policy"]
            require(utc(cp_events[label+"_"+arm])<first_qual and saved["alpha"]==alpha
                and saved["parent_actor_hash"]==ah and saved["source_child_actor_hash"]==child["policy_hash"],"Q alpha proposal identity/qualification chronology differs")
            tensor_interpolation(parent["policy"],child,saved["actor"],alpha,saved["parameter_names"])
    by_name={ep["episode"]:ep for ep in episodes};used=set();reset_checks=0
    def panel(prefix,n,seed,det,owner):
        nonlocal reset_checks
        values=[by_name[f"{prefix}_{i:03d}"] for i in range(n)]
        for i,ep in enumerate(values):
            require(ep["context_id"]==f"ordinary:{seed+i}" and ep["controls"]==25 and ep["actor_hash"]==owner
                and ep["deterministic"] is det and ep["closed_outcome"],"Q fresh executed panel owner/root differs")
            raw=e.npz(base,ep["raw_ref"])
            same(raw["initial_state"],[0.,np.random.default_rng(seed+i).uniform(.2,.4)],"Q fresh reset RNG differs")
            reset_checks+=1;used.add(ep["episode"])
        return values
    parents={mode:panel("qfilter_qual_parent_"+mode,16,171000,mode=="mean",ah) for mode in ("mean","stochastic")}
    for mode in parents:_summary_check(result["qualification_parent"][mode],parents[mode])
    def assess(saved,mean,noisy):
        decisions=[]
        for panels,field in ((mean,"deterministic_guided_vs"),(noisy,"stochastic_guided_vs")):
            for role in ("parent","raw","best_raw","uniform_best_raw"):
                if role not in panels:continue
                wanted=_paired(panels[role],panels["guided"],.001);actual=saved[field][role]
                for key in ("mean_cost_gain","paired_cost_gains"):same(actual[key],wanted[key],"Q evaluated gain differs")
                require(actual["capture_nonregression"]==wanted["capture_nonregression"] and actual["improved"]==wanted["improved"],"Q evaluated improvement decision differs")
                decisions.append(wanted["improved"])
        capture=float(np.mean([ep["capture"] for ep in mean["guided"]]));passed=bool(all(decisions) and capture>=.9)
        same(saved["deterministic_capture_rate"],capture,"Q evaluated capture differs")
        require(saved["passed"]==passed,"Q scientific criterion differs")
        return passed
    qualified=[];raw_qualified=[];qual_events=[row for row in rows if row["kind"]=="qfilter_qualification"]
    require([row["alpha"] for row in result["qualification"]]==ALPHAS and len(qual_events)==5,"Q alpha qualification roster differs")
    for i,row in enumerate(result["qualification"]):
        alpha=row["alpha"];label="qfilter_a"+str(alpha).replace(".","p")
        mean={"parent":parents["mean"]};noisy={"parent":parents["stochastic"]}
        for arm in ("raw","guided"):
            owner=checkpoints[label+"_"+arm]["actor"]["policy_hash"]
            require(row["actor_hashes"][arm]==owner,"Q candidate result hash differs")
            tag="qfilter_qual_a"+str(alpha).replace(".","p")+"_"+arm
            mean[arm]=panel(tag+"_mean",16,171000,True,owner);noisy[arm]=panel(tag+"_stochastic",16,171000,False,owner)
            _summary_check(row["mean_panels"][arm],mean[arm]);_summary_check(row["stochastic_panels"][arm],noisy[arm])
        require(qual_events[i]["result"]==row,"Q qualification event/result differs")
        same(row["guided_mean_complete_cost"],np.mean([ep["cost"] for ep in mean["guided"]]),"Q guide ranking score differs")
        same(row["raw_mean_complete_cost"],np.mean([ep["cost"] for ep in mean["raw"]]),"Q raw ranking score differs")
        if assess(row["criteria"],mean,noisy):qualified.append(row)
        raw_pass=[]
        for data,field in ((mean,"deterministic_raw_vs_parent"),(noisy,"stochastic_raw_vs_parent")):
            wanted=_paired(data["parent"],data["raw"],.001);actual=row["raw_criteria"][field]
            for key in ("mean_cost_gain","paired_cost_gains"):same(actual[key],wanted[key],"Q independent raw gain differs")
            require(actual["capture_nonregression"]==wanted["capture_nonregression"] and actual["improved"]==wanted["improved"],"Q independent raw criterion differs")
            raw_pass.append(wanted["improved"])
        capture=float(np.mean([ep["capture"] for ep in mean["raw"]]));raw_ok=bool(all(raw_pass) and capture>=.9)
        same(row["raw_criteria"]["deterministic_capture_rate"],capture,"Q raw capture summary differs")
        require(row["raw_criteria"]["passed"]==raw_ok,"Q raw selection eligibility differs")
        if raw_ok:raw_qualified.append(row)
    selected=None if not qualified else min(qualified,key=lambda row:(row["guided_mean_complete_cost"],row["alpha"]))["alpha"]
    raw_selected=None if not raw_qualified else min(raw_qualified,key=lambda row:(row["raw_mean_complete_cost"],row["alpha"]))["alpha"]
    decisions=[row for row in rows if row["kind"]=="qfilter_selection"]
    require(len(decisions)==1 and result["selected_alpha"]==decisions[0]["selected_alpha"]==selected
        and result["raw_selected_alpha"]==decisions[0]["raw_selected_alpha"]==raw_selected
        and result["best_raw_parent_fallback"]==decisions[0]["best_raw_parent_fallback"]==(raw_selected is None)
        and decisions[0]["final_evidence_not_seen"] is True and all(utc(event)<utc(decisions[0]) for event in qual_events),"Q qualification-only guide/raw selection differs")
    if selected is None:
        require(not result["passed"] and result["stage"]=="failed_all_qfilter_qualification" and "final" not in result,"Failed Q alpha ladder did not stop")
    else:
        label="qfilter_a"+str(selected).replace(".","p")
        owners={"parent":ah,**{arm:checkpoints[label+"_"+arm]["actor"]["policy_hash"] for arm in ("raw","guided")},
            "best_raw":ah if raw_selected is None else checkpoints["qfilter_a"+str(raw_selected).replace(".","p")+"_raw"]["actor"]["policy_hash"],
            "uniform_best_raw":uniform_source["policy_hash"]}
        require(result["final_actor_hashes"]==owners,"Q final five-role identity differs")
        final={mode:{role:panel("qfilter_final_"+role+"_"+mode,64,172000,mode=="mean",owner) for role,owner in owners.items()} for mode in ("mean","stochastic")}
        for mode,values in final.items():
            for role,value in values.items():_summary_check(result["final"]["mean_panels" if mode=="mean" else "stochastic_panels"][role],value)
        first_final=min(utc(row) for row in rows if row["kind"] in ("neural_requested","query_requested") and row.get("scope")=="qfilter_final_fresh_evaluation")
        require(utc(decisions[0])<first_final,"Q fresh final evidence preceded selection")
        passed=assess(result["final"]["criteria"],final["mean"],final["stochastic"])
        require(result["passed"]==passed and result["stage"]==("complete" if passed else "failed_fresh_qfilter_improvement"),"Q fresh final verdict differs")
    require(set(by_name)==used and len(episodes)==(352 if selected is None else 992),"Unexpected/missing Q fresh actor episode roster")
    model_controls=sum(pair["model_controls"] for pair in measured.values());q_actor_rows=sum(pair["actor_rows"] for pair in measured.values())
    live_controls=8800 if selected is None else 24800
    expected_fit_rows=2*(len(returns)+4*len(returns))+sum(expected_steps*200 for arm in ("raw","guided") if result["qfilter"]["arms"][arm]["positive_original_rows"])
    require(result["accounting"]["model_controls"]==model_controls<=10000
        and result["accounting"]["executed_controls"]==live_controls
        and result["accounting"]["requested_scalar_controls"]==model_controls+live_controls
        and result["accounting"]["actor_rows"]==q_actor_rows+expected_fit_rows+live_controls
        and result["accounting"]["adam_attempts"]==2*expected_steps and result["accounting"]["diagnostic_backward_attempts"]==2,"Q paid query/neural/update schedule differs")
    cross=result["cost_crosswalk"];lineage=(result["accounting"],prior["accounting"],pprov["reused_cost_ancestry"],uniform["accounting"])
    require(cross["new_assay"]==lineage[0] and cross["precision_ancestor"]==lineage[1]
        and cross["v3_ancestor"]==lineage[2] and cross["uniform_step_ancestor"]==lineage[3],"Q paid ancestry attribution differs")
    for key in ("requested_scalar_controls","actor_rows","adam_attempts"):
        require(cross["unique_lineage_"+key]==sum(row[key] for row in lineage),"Q unique ancestry costs double counted/omitted")
    require(result["full_feedback_established"] is False and result["efficiency_claim"] is False
        and result["q_queries_are_behavior_targets"] is False and result["no_new_discovery"] is True
        and result["no_deployment_promotion"] is True,"Unsupported Q-filter scope or ownership claim")
    return dict(exact_parent_state_sha256=parent_digest,behavior_sha256=behavior_digest,reused_mc_episodes=64,
        reused_mc_controls=len(returns),monte_carlo_targets_audited=True,measured_pairs=list(measured.values()),
        logical_auxiliary_rows=400,selected_alpha=selected,raw_selected_alpha=raw_selected,
        analytic_reused_episode_transition_row_checks=reused_checks,analytic_reset_draw_checks=reset_checks,
        model_controls=model_controls,live_controls=live_controls,new_physics_controls=0,new_actor_rows=0,new_optimizer_steps=0,
        limitations=["Actor outputs and recorded likelihoods remain frozen-source/checkpoint bound; no neural forward was rerun.",
            "Deterministic parent Q weights actual labels; it is not a measured return of a child, stochastic parent, or replanning controller."])
