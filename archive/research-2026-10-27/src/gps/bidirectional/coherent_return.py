"""Small coherent whole-policy proposals, selected by actual owned returns.

This is bounded parameter-space search, not an unbiased gradient estimator or
a new primitive. A policy version owns its complete trajectories. Discovery
states affect the search objective; discovered actions never become labels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class HeadSubspace:
    weight_directions:np.ndarray
    seed:int=1701
    head_step:float=.1
    log_std_step:float=.5

    def __post_init__(self):
        basis=np.asarray(self.weight_directions,dtype=np.float64)
        if (basis.shape!=(4,128) or not np.isfinite(basis).all()
            or not np.allclose(basis@basis.T,np.eye(4),atol=1e-12,rtol=0)):
            raise ValueError("Four finite orthonormal128-dimensional head directions required")
        if type(self.seed) is not int or self.seed<0 or self.head_step!=.1 or self.log_std_step!=.5:
            raise ValueError("Undeclared coherent-head geometry")
        immutable=np.frombuffer(np.ascontiguousarray(basis).tobytes(),dtype=np.float64).reshape(4,128)
        object.__setattr__(self,"weight_directions",immutable)


def make_subspace(seed=1701):
    if type(seed) is not int or seed!=1701:raise ValueError("The fixed source-independent seed is1701")
    values=np.random.Generator(np.random.PCG64(seed)).standard_normal((128,4))
    q,r=np.linalg.qr(values,mode="reduced")
    q*=np.where(np.diag(r)<0,-1.,1.)[None,:]
    return HeadSubspace(q.T,seed)


def coordinate_bank(center,step):
    """Center-first stable ties; ±bias, ±four W coordinates, ±logσ."""
    center=np.asarray(center,dtype=np.float64)
    if center.shape!=(6,) or not np.isfinite(center).all() or type(step) is not int or not 0<=step<3:
        raise ValueError("A finite six-coordinate center and one of three steps required")
    if np.abs(center).sum()>step+1e-12:raise ValueError("Center is outside the paid prior-step reachability bound")
    result=[center.copy()]
    for i in range(6):
        for sign in (-1.,1.):
            value=center.copy();value[i]+=sign;result.append(value)
    return np.stack(result)


def actor_from_coordinates(base,subspace,coordinates):
    """No forward/optimizer: copy original trunk/buffers/RNG, alter only head."""
    import torch
    from .policy import FamilyGaussianActor
    c=np.asarray(coordinates,dtype=np.float64)
    if c.shape!=(6,) or not np.isfinite(c).all() or np.abs(c).sum()>3+1e-12:
        raise ValueError("Finite in-budget absolute coordinates required")
    if base.hidden_dim!=128 or base.act_dim!=1 or base.obs_dim!=8:
        raise ValueError("The declared scalar128-head observation8 actor is required")
    before=base.policy_state();actor=FamilyGaussianActor.from_checkpoint(before)
    arrays={n:v.detach().cpu().numpy() for n,v in base.named_parameters()}
    if set(arrays)!={"fc1.weight","fc1.bias","fc2.weight","fc2.bias","fc_mean.weight","fc_mean.bias","log_std"}:
        raise ValueError("Unknown actor parameter layout")
    updates={"fc_mean.bias":arrays["fc_mean.bias"].astype(np.float64)+subspace.head_step*c[0],
        "fc_mean.weight":arrays["fc_mean.weight"].astype(np.float64)+subspace.head_step*(c[1:5]@subspace.weight_directions)[None,:],
        "log_std":arrays["log_std"].astype(np.float64)+subspace.log_std_step*c[5]}
    if np.any(updates["log_std"]<-5) or np.any(updates["log_std"]>1):
        raise ValueError("Proposal crosses declared active log-standard-deviation bounds")
    with torch.no_grad():
        for name,value in actor.named_parameters():
            if name in updates:value.copy_(torch.as_tensor(updates[name],dtype=value.dtype,device=value.device))
            value.requires_grad_(False)
    after=actor.policy_state()
    for name,value in before["actor"].items():
        if name not in updates and not torch.equal(value,after["actor"][name]):raise RuntimeError("Frozen trunk/buffer changed")
    if not torch.equal(before["sampling_rng_state"],after["sampling_rng_state"]):raise RuntimeError("Private actor RNG changed")
    return actor


def _checked_panel(panel):
    if not panel:raise ValueError("Nonempty actual complete owned panel required")
    owners=set();contexts=set();mode=None;remaining=None;task=None;family=None
    for outcome in panel:
        if (not outcome.complete or not math.isfinite(outcome.cost) or outcome.semantics!="actor_suffix"
            or not isinstance(outcome.actor_hash,str) or not outcome.actor_hash or not outcome.query_ids):
            raise ValueError("Only finite complete actor-owned actual continuations may score policy proposals")
        if outcome.context_id in contexts:raise ValueError("A candidate panel cannot duplicate one reset context")
        contexts.add(outcome.context_id);owners.add((outcome.actor_hash,outcome.controller_hash))
        if mode is None:
            mode=outcome.deterministic;remaining=outcome.remaining_steps;task=outcome.task_hash;family=outcome.family_id
        if (outcome.deterministic is not mode or outcome.remaining_steps!=remaining
            or outcome.task_hash!=task or outcome.family_id!=family or len(outcome.query_ids)!=remaining):
            raise ValueError("Mixed task/family/controller mode/horizon or incomplete query trace")
    if len(owners)!=1:raise ValueError("One frozen actor/controller must own every complete outcome in a panel")
    return math.fsum(o.cost for o in panel)/len(panel),sum(o.capture for o in panel),mode,remaining,next(iter(owners)),task,family


def score_candidate(panels,expected_actor):
    """Both modes measured separately; joint score explicitly weights each .5."""
    result={};all_ids=set();mode_owners={};task_family=None
    for role,remaining in (("ordinary",25),("raw",19),("long",19)):
        result[role]={}
        mode_contexts=[]
        for mode,det in (("mean",True),("stochastic",False)):
            panel=panels[role][mode];cost,captures,actual_det,h,owner,task,family=_checked_panel(panel)
            ids=[q for o in panel for q in o.query_ids]
            if owner[0]!=expected_actor or actual_det is not det or h!=remaining or len(panel)!=8:
                raise ValueError("Candidate owner/mode/eight-context/horizon differs from the fixed assay")
            if task_family is None:task_family=(task,family)
            if task_family!=(task,family) or family!=0:raise ValueError("Candidate task/family changed between source panels")
            if mode in mode_owners and mode_owners[mode]!=owner:raise ValueError("Candidate continuation controller changed across source panels")
            mode_owners[mode]=owner
            if len(set(ids))!=len(ids) or all_ids.intersection(ids):raise ValueError("Candidate physical query evidence is reused")
            all_ids.update(ids);mode_contexts.append([o.context_id for o in panel])
            result[role][mode]=dict(mean_cost=cost,captures=captures,controller_hash=owner[1],
                context_ids=mode_contexts[-1],query_ids=ids)
        if mode_contexts[0]!=mode_contexts[1]:raise ValueError("Candidate modes do not share reset contexts")
        result[role]["joint_normalized_cost"]=(result[role]["mean"]["mean_cost"]+result[role]["stochastic"]["mean_cost"])/(2*remaining)
    return dict(actor_hash=expected_actor,roles=result,score_semantics="equal_mean_of_two_SEPARATELY_EXECUTED_controller_costs_normalized_by_remaining")


def select_candidate(scores,source,beta):
    """Root nonregression first, then declared measured teaching objective."""
    if source not in ("ordinary","raw","long") or beta not in (0.,.25,1.) or ((source=="ordinary")!=(beta==0.)):
        raise ValueError("Undeclared source/data weight")
    if len(scores)!=13:raise ValueError("Every center-plus-twelve complete candidate must be scored")
    allowed={"ordinary"} if source=="ordinary" else {"ordinary",source}
    if any(set(score["roles"])!=allowed for score in scores):
        raise ValueError("Selector may only see its declared ordinary/source return panels")
    baseline=scores[0]["roles"]["ordinary"];feasible=[];objectives=[]
    for score in scores:
        ordinary=score["roles"]["ordinary"]
        valid=all(ordinary[mode]["mean_cost"]<=baseline[mode]["mean_cost"]
            and ordinary[mode]["captures"]>=baseline[mode]["captures"] for mode in ("mean","stochastic"))
        feasible.append(valid)
        a=ordinary["joint_normalized_cost"]
        b=0. if source=="ordinary" else score["roles"][source]["joint_normalized_cost"]
        objectives.append((a+beta*b)/(1+beta))
    if not feasible[0]:raise RuntimeError("Current center lost feasibility")
    selected=min((i for i,ok in enumerate(feasible) if ok),key=lambda i:(objectives[i],i))
    return dict(selected_index=selected,feasible_flags=feasible,objectives=objectives,
        source=source,beta=beta,primary_constraint="ordinary_complete_cost_and_capture_nonregression_in_EACH_mode",
        joint_objective="(ordinary_equal_mode_cost/25+beta*source_equal_mode_cost/19)/(1+beta)",
        actor_hash=scores[selected]["actor_hash"])


def selector_view(score,source):
    """Unused paid panels are retained in evidence, excluded from selection."""
    if source not in ("ordinary","raw","long"):raise ValueError("Unknown source")
    roles=("ordinary",) if source=="ordinary" else ("ordinary",source)
    return dict(actor_hash=score["actor_hash"],roles={r:deepcopy(score["roles"][r]) for r in roles},
        score_semantics=score["score_semantics"])
