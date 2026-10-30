"""State-informed head geometry, selected only by complete ordinary returns.

No work runs on import. Geometry uses frozen parent pre-squash hidden features;
it is not global action whitening, a feature-capacity claim or action imitation.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def require(ok,message):
    if not ok:raise ValueError(message)


def random_basis(seed=1701):
    require(type(seed) is int and seed==1701,'The prospective basis seed is fixed')
    q,r=np.linalg.qr(np.random.Generator(np.random.PCG64(seed)).standard_normal((128,4)),mode='reduced')
    return (q*np.where(np.diag(r)<0,-1.,1.)[None,:]).T


def prefix_rows(observations,states,query_ids):
    o,s=np.asarray(observations),np.asarray(states);q=list(query_ids)
    require(o.shape==(26,8) and s.shape==(26,2) and len(q)==25 and np.isfinite(o[:12]).all()
        and np.isfinite(s[:12]).all() and np.array_equal(s[:12,0],np.arange(12)),
        'Only the twelve pre-gate rows of a complete original episode are eligible')
    return o[:12].copy(),s[:12].copy(),q[:12]


def calibration_indices(mode):
    require(mode in ('fixed','ordinary','raw','long'),'Undeclared geometry calibration source')
    return np.r_[np.arange(96),np.arange(96) if mode in ('fixed','ordinary') else np.arange(96,192) if mode=='raw' else np.arange(192,288)]


@dataclass(frozen=True)
class Geometry:
    weight_axes:np.ndarray
    bias_axes:np.ndarray
    diagnostics:dict

    def __post_init__(self):
        w,b=np.asarray(self.weight_axes,dtype=np.float64),np.asarray(self.bias_axes,dtype=np.float64)
        require(w.shape==(4,128) and b.shape==(4,) and np.isfinite(w).all() and np.isfinite(b).all(),
            'Four finite128-weight and scalar-bias axes required')
        require(np.max(np.linalg.norm(w,axis=1))<=1+1e-12 and np.max(abs(b))<=.5+1e-12,'Geometry norm/bias cap exceeded')
        object.__setattr__(self,'weight_axes',np.frombuffer(np.ascontiguousarray(w).tobytes(),dtype=np.float64).reshape(4,128))
        object.__setattr__(self,'bias_axes',np.frombuffer(np.ascontiguousarray(b).tobytes(),dtype=np.float64))


def fit_geometry(hidden,basis,mode):
    H,B=np.asarray(hidden,dtype=np.float64),np.asarray(basis,dtype=np.float64)
    require(H.shape==(192,128) and B.shape==(4,128) and np.isfinite(H).all() and np.isfinite(B).all()
        and np.allclose(B@B.T,np.eye(4),atol=1e-12,rtol=0),'Declared192-row frozen feature calibration required')
    require(mode in ('fixed','ordinary','raw','long'),'Undeclared geometry source')
    A=H@B.T;mu=A.mean(axis=0)
    _,s,V=np.linalg.svd(A-mu,full_matrices=False)
    # ± candidates are sign-symmetric, but canonically orient stored axes.
    for i in range(4):
        if V[i,np.argmax(np.abs(V[i]))]<0:V[i]*=-1
    if mode=='fixed':
        weights=.1*B;bias=np.zeros(4);scales=np.full(4,.1);active=np.ones(4,dtype=bool)
        caps=['fixed_old_random_basis']*4
    else:
        rms=s/math.sqrt(len(H));active=rms>=1e-3;scales=np.zeros(4);caps=[]
        for i in range(4):
            if not active[i]:caps.append('inactive_singular_rms_below_.001');continue
            projection=float(mu@V[i]);values=[.1/rms[i],1.,math.inf if projection==0 else .5/abs(projection)]
            scales[i]=min(values);caps.append(('target_rms_.1','weight_norm1','bias_abs_.5')[int(np.argmin(values))])
        weights=(V@B)*scales[:,None];bias=-(mu@V.T)*scales
    response=H@weights.T+bias;centered=response-response.mean(axis=0)
    gram=centered.T@centered/len(H)
    diagnostics=dict(mode=mode,calibration_rows=192,pre_squash_feature_dim=128,projected_feature_dim=4,
        projected_mean=mu.tolist(),right_singular_vectors=V.tolist(),singular_values=s.tolist(),
        singular_rms=(s/math.sqrt(len(H))).tolist(),active_axes=active.tolist(),axis_scales=scales.tolist(),limiting_caps=caps,
        weight_norms=np.linalg.norm(weights,axis=1).tolist(),bias_offsets=bias.tolist(),
        achieved_pre_squash_rms=np.sqrt(np.mean(response*response,axis=0)).tolist(),
        achieved_centered_pre_squash_rms=np.sqrt(np.mean(centered*centered,axis=0)).tolist(),
        response_means=response.mean(axis=0).tolist(),centered_response_gram=gram.tolist(),
        empirical_active_rank=int(np.sum(s/math.sqrt(len(H))>=1e-3)),active_parameter_axes=int(np.sum(active)),
        singular_vectors_used_for_axes=mode!='fixed',
        scope='Empirical parent pre-squash feature geometry only; caps and inactive axes may prevent target RMS; no rescaling or retry')
    return Geometry(weights,bias,diagnostics)


def coordinate_bank(center,step):
    c=np.asarray(center,dtype=np.float64)
    require(c.shape==(6,) and np.isfinite(c).all() and type(step) is int and 0<=step<3
        and np.abs(c).sum()<=step+1e-12,'Finite in-budget absolute coordinates required')
    result=[c.copy()]
    for vector in np.eye(6):result.extend((c-vector,c+vector))
    return np.stack(result)


def actor_from_geometry(base,geometry,coordinates):
    import torch
    from .policy import FamilyGaussianActor
    c=np.asarray(coordinates,dtype=np.float64)
    require(c.shape==(6,) and np.isfinite(c).all() and np.abs(c).sum()<=3+1e-12,'Coordinate L1 cap exceeded')
    require(base.obs_dim==8 and base.hidden_dim==128 and base.act_dim==1,'Declared128-head scalar actor required')
    before=base.policy_state();actor=FamilyGaussianActor.from_checkpoint(before)
    params={k:v.detach().cpu().numpy() for k,v in base.named_parameters()}
    require(set(params)=={'fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc_mean.weight','fc_mean.bias','log_std'},'Unknown actor roster')
    changes={'fc_mean.weight':c[1:5]@geometry.weight_axes,
        'fc_mean.bias':.1*c[0]+c[1:5]@geometry.bias_axes,'log_std':.5*c[5]}
    with torch.no_grad():
        for name,value in actor.named_parameters():
            if name in changes:
                updated=params[name].astype(np.float64)+changes[name]
                require(np.isfinite(updated).all() and (name!='log_std' or np.all((updated>=-5)&(updated<=1))),'Invalid active logstd proposal')
                value.copy_(torch.as_tensor(updated,dtype=value.dtype,device=value.device))
            value.requires_grad_(False)
    after=actor.policy_state()
    require(all(torch.equal(v,after['actor'][k]) for k,v in before['actor'].items() if k not in changes),'Frozen trunk/buffer changed')
    require(torch.equal(before['sampling_rng_state'],after['sampling_rng_state']),'Private actor RNG changed')
    return actor


def score_candidate(panels,actor_hash):
    result={};ids=set();contexts=None;task=None
    for mode,det in (('mean',True),('stochastic',False)):
        values=panels[mode];require(len(values)==8,'Exactly eight ordinary complete outcomes required')
        require(all(v.complete and v.semantics=='actor_suffix' and v.deterministic is det and v.actor_hash==actor_hash
            and v.remaining_steps==25 and len(v.query_ids)==25 and v.family_id==0 and math.isfinite(v.cost) for v in values),'Invalid complete own ordinary return')
        roots=[v.context_id for v in values];require(len(set(roots))==8 and (contexts is None or contexts==roots),'Reset panel differs between modes')
        contexts=roots;task=values[0].task_hash if task is None else task
        require(all(v.task_hash==task and v.controller_hash==values[0].controller_hash for v in values),'Task/controller changed within panel')
        q=[qid for v in values for qid in v.query_ids];require(len(set(q))==200 and not ids.intersection(q),'Complete control evidence reused')
        ids.update(q);result[mode]=dict(mean_cost=math.fsum(v.cost for v in values)/8,captures=sum(v.capture for v in values),
            capture_flags=[bool(v.capture) for v in values],context_ids=roots,query_ids=q,controller_hash=values[0].controller_hash)
    return dict(actor_hash=actor_hash,ordinary=result,joint_normalized_cost=(result['mean']['mean_cost']+result['stochastic']['mean_cost'])/50,
        score_semantics='ordinary_only_equal_mean_of_two_SEPARATELY_EXECUTED_complete_controller_returns')


def select_candidate(scores):
    require(len(scores)==13,'Center plus twelve fully measured proposals required')
    base=scores[0]['ordinary'];feasible=[];changes=[]
    for row in scores:
        require(all(row['ordinary'][m]['context_ids']==base[m]['context_ids'] for m in ('mean','stochastic')),'Candidate ordinary reset panel differs')
        feasible.append(all(row['ordinary'][mode]['mean_cost']<=base[mode]['mean_cost']
            and row['ordinary'][mode]['captures']>=base[mode]['captures'] for mode in ('mean','stochastic')))
        changes.append({m:dict(lost=sum(a and not b for a,b in zip(base[m]['capture_flags'],row['ordinary'][m]['capture_flags'])),
            gained=sum(b and not a for a,b in zip(base[m]['capture_flags'],row['ordinary'][m]['capture_flags']))) for m in ('mean','stochastic')})
    require(feasible[0],'Center must remain eligible')
    selected=min((i for i,ok in enumerate(feasible) if ok),key=lambda i:(scores[i]['joint_normalized_cost'],i))
    return dict(selected_index=selected,feasible_flags=feasible,objectives=[r['joint_normalized_cost'] for r in scores],
        actor_hash=scores[selected]['actor_hash'],source_return_weight=0.,paired_capture_changes=changes,
        capture_nonregression_definition='Aggregate capture count in each separately executed matched mode; per-root losses/gains retained',
        scope='Only actual complete ordinary-root returns; cost and capture nonregression in each mode')
