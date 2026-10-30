"""Independent completion/child evidence checks; no producer or engine imports."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np


SEMANTICS="one_candidate_action_then_deterministic_actor"


def require(value,message):
    if not value:raise ValueError(message)


def same(a,b,message):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and a.dtype.kind in "biuf" and b.dtype.kind in "biuf"
        and np.isfinite(a).all() and np.isfinite(b).all()
        and np.allclose(a,b,atol=2e-12,rtol=2e-12),message)


def numeric_digest(value):
    """Reconstruct the recorded numeric-state digest without loading Torch."""
    digest=hashlib.sha256()
    def feed(item):
        if isinstance(item,np.ndarray):
            array=np.ascontiguousarray(item)
            digest.update(json.dumps([array.dtype.str,list(array.shape)]).encode());digest.update(array.tobytes())
        elif isinstance(item,np.generic):feed(item.item())
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


def behavior_stem(purpose,phase):
    if purpose=="candidate_update":
        match=re.fullmatch(r"update(\d+)_fit",phase)
        require(match is not None,"MC fit not phase-bound")
        return "u"+match.group(1)+"_own_"
    precision=re.fullmatch(r"precision_(original|precision)_matched_child_(raw|guided)",purpose)
    if precision:
        require(phase=="precision_"+precision.group(1)+"_child_fit","Precision MC fit not phase-bound")
        return "precision_"+precision.group(1)+"_own_"
    match=re.fullmatch(r"completion_r(\d+)_matched_child_(raw|guided)",purpose)
    require(match is not None and phase=="completion_r"+match.group(1)+"_child_fit",
        "Completion MC fit not phase-bound")
    return "completion_r"+match.group(1)+"_own_"


def is_mc_fit(purpose):
    return purpose=="candidate_update" or re.fullmatch(r"completion_r\d+_matched_child_(raw|guided)",purpose) is not None or re.fullmatch(r"precision_(original|precision)_matched_child_(raw|guided)",purpose) is not None


def validate_completion(raw,event,request,decision,initial_state,actual_action,bank_actions,independent_plan,query,T,task_hash):
    """Check one actual selection against every already-audited K1/H1 query.

    `query` returns saved input/result/request dictionaries. Actor actions are
    checkpoint/source bound; this function never evaluates a neural network.
    """
    first=np.asarray(raw["candidate_first_actions"]);m=len(first);remaining=T-int(initial_state[0])
    require(1<=m<=5 and first.shape==(m,1) and np.isfinite(first).all() and np.all(np.abs(first)<=1),"Completion candidate bounds differ")
    require(str(raw["status"])=="complete" and str(raw["semantics"])==event["semantics"]==request["semantics"]==SEMANTICS,"Completion semantics/status differs")
    require(event["request_id"]==request["id"] and request["elapsed"]==int(initial_state[0])
        and request["remaining"]==remaining and event["guidance_weight"]==request["guidance_weight"]==1.,"Completion request/support differs")
    require(request["maximum_additional_work"]==dict(model_controls=5*remaining,model_queries=5*remaining,
        actor_rows=1+5*(remaining-1),candidates=5),"Completion declared work ceiling differs")
    same(raw["initial_state"],initial_state,"Completion actual initial state differs")
    require(str(raw["context_id"])==request["context_id"],"Completion context differs")
    actor_hash=str(raw["actor_hash"]);actor_controller=str(raw["actor_controller_hash"])
    require(actor_hash==event["actor_hash"]==request["actor_hash"] and actor_controller==request["actor_controller_hash"],"Completion actor owner differs")
    protocol=dict(version="ending-first-action-actor-completion-v1",semantics=SEMANTICS,actor_hash=actor_hash,
        actor_controller_hash=actor_controller,task_hash=task_hash,family=0,total_steps=T,empirical_quantiles=[0.,.5,1.],
        quantile_rule="floor(q*(K-1)) of stable-sorted native bank first actions",
        order=["actor_first","independent_champion","bank_min","bank_median","bank_max"],
        deduplication="exact scalar equality; first role retains tie priority",
        cost="sum one-control Ending totals through true T; no endpoint heuristic",
        selection="stable minimum complete cost; actor-first ties",
        deployment="current promoted support; eligible weight1 trial only")
    protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    require(str(raw["protocol_identity"])==event["protocol_identity"]==request["protocol_identity"]==protocol_hash,"Completion selector protocol hash differs")
    bank=np.asarray(bank_actions);order=np.argsort(bank[:,0,0],kind="stable")
    proposals=[(float(first[0,0]),"actor_first",None),(float(independent_plan[0,0]),"independent_champion",None)]
    proposals += [(float(bank[order[rank],0,0]),role,int(order[rank])) for rank,role in
        zip((0,(len(order)-1)//2,len(order)-1),("bank_min","bank_median","bank_max"))]
    expected=[]
    for action,role,row in proposals:
        prior=next((item for item in expected if item["action"]==action),None)
        if prior is None:expected.append(dict(candidate_id=f"candidate-{len(expected):03d}",action=action,roles=[dict(role=role,bank_row=row)]))
        else:prior["roles"].append(dict(role=role,bank_row=row))
    roles=json.loads(str(raw["candidate_roles_json"]))
    require(roles==expected and np.array_equal(first,np.array([r["action"] for r in expected])[:,None]),"Completion candidate order/aliases differ")
    n=m*remaining;qids=np.asarray(raw["query_ids"]).tolist()
    require(len(qids)==n and len(set(qids))==n and qids==event["query_ids"]==decision["guided_query_ids"],"Completion raw query linkage differs")
    same(raw["trace_candidate_indices"],np.repeat(np.arange(m),remaining),"Completion candidate-major trace order differs")
    same(raw["trace_steps"],np.tile(np.arange(remaining),m),"Completion trace step order differs")
    same(raw["completed_candidate_indices"],np.arange(m),"Completion candidate coverage differs")
    pre=np.asarray(raw["prestates"]);post=np.asarray(raw["states"]);actions=np.asarray(raw["actions"]);costs=np.asarray(raw["raw_costs"])
    require(pre.shape==post.shape==(n,2) and actions.shape==(n,1) and costs.shape==(n,),"Completion trace dimensions differ")
    expected_obs=np.zeros((n,8));expected_obs[:,0]=pre[:,1];expected_obs[:,1]=pre[:,0]/T;expected_obs[:,2]=1-expected_obs[:,1]
    same(raw["observations"],expected_obs,"Completion pre-action observations differ")
    for i in range(m):
        lo=i*remaining;hi=lo+remaining
        same(pre[lo],initial_state,"Completion candidate restart differs")
        same(pre[lo+1:hi],post[lo:hi-1],"Completion actor tail is discontinuous")
        same(pre[lo:hi,0],np.arange(int(initial_state[0]),T),"Completion pre-clock differs")
        same(post[lo:hi,0],np.arange(int(initial_state[0])+1,T+1),"Completion does not reach true ending")
        same(actions[lo],first[i],"Completion first action differs")
    for j,qid in enumerate(qids):
        saved=query(qid);req=saved["request"];inp=saved["inputs"];res=saved["result"];owner=req["owner"]
        i,step=divmod(j,remaining)
        require(req["mode"]=="model" and req["query_kind"]=="actor_completion_step"
            and owner["semantics"]==SEMANTICS and owner["completion_request_id"]==request["id"]
            and owner["protocol_identity"]==protocol_hash and owner["actor_hash"]==actor_hash
            and owner["actor_controller_hash"]==actor_controller and owner["candidate_index"]==i
            and owner["candidate_id"]==roles[i]["candidate_id"] and owner["step"]==step
            and owner["first_action_roles"]==roles[i]["roles"],"Completion query continuation owner differs")
        require(owner["owner"]==dict(semantics=SEMANTICS,actor_hash=actor_hash,actor_controller_hash=actor_controller,
            family=0,deterministic=True,candidate_id=roles[i]["candidate_id"]),"Completion nested actor owner differs")
        same(inp["initial_state"],pre[j],"Completion query prestate differs")
        same(inp["actions"],actions[j].reshape(1,1,1),"Completion queried action differs")
        same(res["states"],post[j].reshape(1,1,2),"Completion query poststate differs")
        same(res["totals"],np.array([costs[j]]),"Completion one-control cost differs")
    totals=costs.reshape(m,remaining).sum(axis=1)
    same(raw["complete_candidate_costs"],totals,"Completion full return totals differ")
    same(event["candidate_costs"],totals,"Completion event totals differ")
    winner=int(np.argmin(totals))
    require(int(raw["selected_index"])==event["selected_index"]==winner
        and event["candidate_ids"]==[row["candidate_id"] for row in roles]
        and event["selected_candidate_id"]==roles[winner]["candidate_id"],"Completion stable actor-first minimum differs")
    same(event["selected_first_action"],first[winner,0],"Completion selected event action differs")
    same(actual_action,first[winner],"Completion selected action was not executed")
    return dict(candidates=m,model_controls=n,query_ids=qids,selected_index=winner,
        actor_hash=actor_hash,protocol_identity=protocol_hash,actor_outputs_independently_forwarded=False)


def validate_child_pair(raw_child,guided_child,parent,aux,shared,raw_new,guided_new,rows):
    """Reconstruct both fixed auxiliary strata and bind exact parent metadata."""
    require(raw_child["parent_actor_hash"]==guided_child["parent_actor_hash"]==parent["policy"]["policy_hash"],"Matched children have different parent actor")
    parent_digest=numeric_digest(parent)
    require(raw_child["parent_state_sha256"]==guided_child["parent_state_sha256"]==parent_digest,"Matched full parent optimizer/RNG digest differs")
    require(raw_child["behavior_sha256"]==guided_child["behavior_sha256"] and len(raw_child["behavior_sha256"])==64,"Matched behavior digest differs")
    require(raw_child["shared_teacher_refs"]==guided_child["shared_teacher_refs"]
        and raw_child["auxiliary_raw"]==guided_child["auxiliary_raw"],"Matched common teacher/auxiliary evidence differs")
    require(rows>0 and rows%2==0,"Matched auxiliary rows must form two equal positive strata")
    def dataset(traces):
        eligible=[r for r in traces if bool(r["captured"])]
        require(bool(eligible),"Successful auxiliary source absent")
        return np.concatenate([r["observations"][:-1] for r in eligible]),np.concatenate([r["actions"] for r in eligible])
    common_obs,common_actions=dataset(shared);half=rows//2
    for name,new in (("raw",raw_new),("guided",guided_new)):
        successful=[r for r in new if bool(r["captured"])]
        specific_obs,specific_actions=dataset(successful or shared)
        ci=np.linspace(0,len(common_actions)-1,half,dtype=np.int64)
        si=np.linspace(0,len(specific_actions)-1,half,dtype=np.int64)
        same(aux[name+"_common"],ci,"Matched common indices differ")
        same(aux[name+"_specific"],si,"Matched new-teacher indices differ")
        require(bool(aux[name+"_specific_is_shared_fallback"])==(not bool(successful)),"Matched fallback source differs")
        same(aux[name+"_observations"],np.concatenate((common_obs[ci],specific_obs[si])),"Matched observations differ from actual teacher strata")
        same(aux[name+"_actions"],np.concatenate((common_actions[ci],specific_actions[si])),"Matched actions differ from actual teacher strata")
        same(aux[name+"_families"],np.zeros(rows,dtype=int),"Matched family differs")
    same(aux["raw_observations"][:half],aux["guided_observations"][:half],"Matched shared observation half differs")
    same(aux["raw_actions"][:half],aux["guided_actions"][:half],"Matched shared action half differs")
    return dict(parent_state_sha256=parent_digest,behavior_sha256=raw_child["behavior_sha256"],
        auxiliary_rows_per_child=rows,shared_rows_per_child=half,behavior_digest_scope="equal checkpoint metadata; raw own-MC panels audited separately")
