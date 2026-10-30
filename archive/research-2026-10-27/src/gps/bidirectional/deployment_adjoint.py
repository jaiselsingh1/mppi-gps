"""Deployment-aligned actor gradients from measured own trajectories.

The Ending Jacobian is analytic. It is not a learned critic, a teacher action
target, or a reinterpreted return of an updated actor. Task costs are unchanged;
the relaxed barrier is explicitly a training constraint surrogate.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
import torch
from .policy import policy_hash


@dataclass(frozen=True)
class AdjointConfig:
    margin:float=.001
    barrier_mu:float=.001
    relaxation:float=.01
    acquired_weight:float=.25
    lr:float=.0003
    gradient_clip:float=.5
    fractions:tuple=(1.,.5,.25,.125)


def relaxed_barrier(slack,mu=.001,delta=.01):
    # Clamp the unused log branch as torch.where still evaluates both branches.
    log=-mu*torch.log(slack.clamp_min(delta))
    quadratic=mu*(.5*((slack-2*delta)/delta).square()-.5-np.log(delta))
    return torch.where(slack>=delta,log,quadratic)


def barrier_numpy(slack,mu=.001,delta=.01):
    s=np.asarray(slack,dtype=float)
    return np.where(s>=delta,-mu*np.log(np.maximum(s,delta)),mu*(.5*((s-2*delta)/delta)**2-.5-np.log(delta)))


def validate_record(record,owner,task):
    if record['actor_hash']!=owner or record['deterministic'] is not True or record['continuation']!='uninterrupted_actor':
        raise ValueError('Adjoint targets require this exact deterministic actor, not teacher/planner trajectories')
    s,a=np.asarray(record['states']),np.asarray(record['actions']);t=int(s[0,0]);T=task['gate']+task['delay']+1
    if s.shape!=(T-t+1,2) or a.shape!=(T-t,1) or not np.array_equal(s[:,0],np.arange(t,T+1)):
        raise ValueError('Complete pre/post aligned actor trajectory required')
    if T!=25 or not np.isfinite(s).all() or not np.isfinite(a).all() or np.any(abs(a)>1) or len(record['query_ids'])==0:
        raise ValueError('Invalid actor trajectory/evidence')
    if len(record['query_ids'])!=T-t or len(set(record['query_ids']))!=T-t:
        raise ValueError('Each actual control needs distinct query evidence')
    if not np.array_equal(s[max(0,task['gate']-t):,1],np.full(len(s[max(0,task['gate']-t):]),s[-1,1])):
        raise ValueError('Ending adapter/Jacobian assumes position fixed after the gate')
    return t


def trajectory_tangent(actor,record,task):
    """Exact first-order closed-loop tangent at the measured actor/version.

    Neural forwards use measured pre-action states with a state-gradient bridge.
    Numerical graph values remain on the measured trajectory. No transition is
    simulated by this function. Derivatives of the binary terminal cost are not
    invented: the separately declared goal constraint provides that feedback.
    """
    owner=policy_hash(actor);start=validate_record(record,owner,task);s=np.asarray(record['states']);a=np.asarray(record['actions'])
    dtype=actor.action_scale.dtype;device=actor.action_scale.device
    y=torch.tensor(float(s[0,1]),dtype=dtype,device=device);running=y*0.;rows=0
    for i,t in enumerate(range(start,task['gate'])):
        obs=np.zeros(8);obs[0]=s[i,1];obs[1]=t/25;obs[2]=1-t/25
        observed=torch.as_tensor(obs,dtype=dtype,device=device)
        observed=observed+torch.cat(((y-y.detach()).reshape(1),torch.zeros(7,dtype=dtype,device=device)))
        mean,_=actor(observed,0);action=actor._bounded(mean)[0];rows+=1
        if abs(float(action.detach().cpu())-a[i,0])>2e-7:raise ValueError('Saved action is not the current deterministic mean')
        unclipped=s[i,1]+task['move']*a[i,0];jac=float(-1.<unclipped<1.)
        dy=(y-y.detach())+task['move']*(action-action.detach())
        next_y=torch.tensor(float(s[i+1,1]),dtype=dtype,device=device)+jac*dy
        # Tangent to effort + post-state greed. Fixed detour/ending values are
        # verified in actual executions; their discontinuities have no fake slope.
        value=task['effort']*a[i,0]**2+.5*task['greed']*(1-s[i+1,1])
        running=running+value+2*task['effort']*a[i,0]*(action-action.detach())-.5*task['greed']*(next_y-next_y.detach())
        y=next_y
    return running,y,rows


def trajectory_surrogate(actor,record,task,cfg=AdjointConfig()):
    running,y,rows=trajectory_tangent(actor,record,task)
    owner=policy_hash(actor);s=np.asarray(record['states'])
    slack=task['ending_y']-cfg.margin-y
    barrier=relaxed_barrier(slack,cfg.barrier_mu,cfg.relaxation)
    loss=(running if record['ordinary'] else running*0.)+barrier*(1. if record['ordinary'] else cfg.acquired_weight)
    return loss,dict(actor_hash=owner,active_jacobian_rows=rows,ordinary=record['ordinary'],
        smooth_running_value=float(running.detach().cpu()),barrier_value=float(barrier.detach().cpu()),
        ending_y=float(s[-1,1]),slack=float(slack.detach().cpu()),
        semantics='own_measured_closed_loop_tangent;binary_terminal_replaced_only_in_training_surrogate')


def feasible_descent(gradient,constraint_gradients,upper,step_norm=.03,cycles=64):
    """Alternating halfspace projection; validate feasibility and descent.

    This is a finite feasibility calculation, not a claim of exact QP solution.
    Actual changed-controller executions still decide update acceptance.
    """
    g=np.asarray(gradient,dtype=float);A=np.asarray(constraint_gradients,dtype=float);b=np.asarray(upper,dtype=float)
    if g.ndim!=1 or A.ndim!=2 or A.shape[1]!=len(g) or b.shape!=(len(A),) or not all(np.isfinite(x).all() for x in (g,A,b)):
        raise ValueError('Finite aligned policy and ending gradients required')
    d=-step_norm*g/max(np.linalg.norm(g),1e-12);denom=np.einsum('ij,ij->i',A,A)
    for _ in range(cycles):
        for row,rhs,norm in zip(A,b,denom):
            excess=float(row@d-rhs)
            if excess>0 and norm>1e-20:d-=excess/norm*row
    violations=A@d-b;descent=float(g@d)
    valid=bool(max(violations,default=0.)<=1e-7 and descent< -1e-9 and np.linalg.norm(d)<=step_norm+1e-7)
    return d if valid else np.zeros_like(d),dict(valid=valid,objective_directional_derivative=descent,
        maximum_linear_violation=float(max(violations,default=0.)),direction_norm=float(np.linalg.norm(d)),
        projection_cycles=cycles,halfspace_checks=cycles*len(A))


def projected_tangent_direction(actor,records,task,margin=.001,step_norm=.03,cycles=64,restoration=.01):
    """Separate ordinary cost descent from explicit measured ending constraints."""
    parameters=tuple(p for name,p in actor.named_parameters() if name!='log_std');names=[name for name,p in actor.named_parameters() if name!='log_std']
    pieces=[trajectory_tangent(actor,r,task) for r in records]
    ordinary=[v[0] for v,r in zip(pieces,records) if r['ordinary']]
    cost=sum(ordinary)/len(ordinary)
    def flatten(loss,retain=True):
        gs=torch.autograd.grad(loss,parameters,retain_graph=retain,allow_unused=True)
        return np.concatenate([(torch.zeros_like(p) if g is None else g).detach().cpu().numpy().ravel() for p,g in zip(parameters,gs)]).astype(float)
    gradient=flatten(cost);rows=[];upper=[]
    for i,(piece,record) in enumerate(zip(pieces,records)):
        rows.append(flatten(piece[1],retain=i<len(pieces)-1))
        slack=task['ending_y']-margin-float(record['states'][-1,1])
        upper.append(max(slack,0.) if record['ordinary'] else (slack if slack>=0 else -min(-slack,restoration)))
    A=np.stack(rows);b=np.asarray(upper);direction,info=feasible_descent(gradient,A,b,step_norm,cycles)
    offset=0;updates={}
    for name,p in zip(names,parameters):
        updates[name]=torch.as_tensor(direction[offset:offset+p.numel()].reshape(p.shape),dtype=p.dtype,device=p.device);offset+=p.numel()
    return updates,dict(**info,active_jacobian_rows=sum(v[2] for v in pieces),gradient_calls=1+len(records),
        ordinary_running_value=float(cost.detach().cpu()),upper_bounds=b.tolist()),dict(cost_gradient=gradient,ending_gradients=A,upper_bounds=b,direction=direction)


def head_tangent_geometry(actor,records,task,*,serial=False):
    """Measured mean-head closed-loop Jacobians, including state feedback.

    One paid neural batch supplies hidden values. Analytic NN and physical
    Jacobians then differentiate the frozen backbone and trainable mean head.
    No planner action label or new simulator transition is used.
    """
    owner=policy_hash(actor);observations=[];expected=[];starts=[]
    for r in records:
        t=validate_record(r,owner,task);starts.append(t)
        for i,clock in enumerate(range(t,12)):
            obs=np.zeros(8);obs[0]=r['states'][i,1];obs[1]=clock/25;obs[2]=1-clock/25
            observations.append(obs);expected.append(r['actions'][i,0])
    boxes={}
    def capture(name):
        def hook(module,args):boxes[name]=args[0].detach().cpu().numpy().copy()
        return hook
    hooks=[actor.fc2.register_forward_pre_hook(capture('h1')),actor.fc_mean.register_forward_pre_hook(capture('h2'))]
    try:
        with torch.no_grad():
            if serial:
                first=[];second=[];actions=[]
                for observation in observations:
                    mean,_=actor(observation,0);actions.append(float(actor._bounded(mean)[0].cpu()))
                    first.append(boxes['h1'].copy());second.append(boxes['h2'].copy())
                actual=np.asarray(actions);boxes['h1']=np.stack(first);boxes['h2']=np.stack(second)
            else:mean,_=actor(np.asarray(observations),0);actual=actor._bounded(mean).cpu().numpy()[:,0]
    finally:
        for hook in hooks:hook.remove()
    if not np.allclose(actual,expected,rtol=0,atol=2e-7):raise ValueError('Feature geometry is not this measured mean actor')
    W1=actor.fc1.weight.detach().cpu().numpy().astype(float);W2=actor.fc2.weight.detach().cpu().numpy().astype(float)
    w=actor.fc_mean.weight.detach().cpu().numpy()[0].astype(float)
    h1,h2=boxes['h1'].astype(float),boxes['h2'].astype(float);offset=0;costs=[];ending=[];J=[];running=[]
    for r,t0 in zip(records,starts):
        state_grad=np.zeros(len(w)+1);cost_grad=state_grad.copy();cost=0.
        for i,t in enumerate(range(t0,12)):
            u=float(expected[offset]);one=1-u*u;feature=np.r_[h2[offset],1.]
            dh1=(1-h1[offset]**2)*W1[:,0];dh2=(1-h2[offset]**2)*(W2@dh1);u_y=one*float(w@dh2)
            action_grad=one*feature+u_y*state_grad;J.append(action_grad)
            unclipped=float(r['states'][i,1])+task['move']*u
            state_grad=float(-1.<unclipped<1.)*(state_grad+task['move']*action_grad)
            cost_grad+=2*task['effort']*u*action_grad-.5*task['greed']*state_grad
            cost+=task['effort']*u*u+.5*task['greed']*(1-r['states'][i+1,1]);offset+=1
        if r['ordinary']:costs.append(cost_grad);running.append(cost)
        ending.append(state_grad)
    return dict(cost_gradient=np.mean(costs,axis=0),ending_gradients=np.stack(ending),control_jacobian=np.stack(J),
        observations=np.asarray(observations),h1=boxes['h1'],h2=boxes['h2'],mean_actions=np.asarray(expected),
        ordinary_running_value=np.array(np.mean(running)))


def metric_feasible_descent(g,A,b,J,rms=.05,ridge=1e-5,cycles=64):
    """Finite halfspace projection in a measured closed-loop action metric."""
    F=J.T@J/len(J)+ridge*np.eye(len(g));solutions=np.linalg.solve(F,np.column_stack((g,A.T)))
    z=solutions[:,0];inverse_rows=solutions[:,1:].T
    d=-rms*z/max(float(np.sqrt(z@F@z)),1e-12)
    for _ in range(cycles):
        for row,inverse,rhs in zip(A,inverse_rows,b):
            excess=float(row@d-rhs);denom=float(row@inverse)
            if excess>0 and denom>1e-20:d-=excess/denom*inverse
        norm=float(np.sqrt(max(0.,d@F@d)))
        if norm>rms:d*=rms/norm
    norm=float(np.sqrt(max(0.,d@F@d)));descent=float(g@d);violation=float(max(A@d-b))
    valid=bool(norm<=rms+1e-7 and violation<=1e-7 and descent< -1e-9)
    info=dict(valid=valid,metric_norm=norm,parameter_norm=float(np.linalg.norm(d)),objective_directional_derivative=descent,
        maximum_linear_violation=violation,halfspace_checks=cycles*len(A),metric_ball_checks=cycles,linear_solves=1,ridge=ridge,action_rms_bound=rms)
    return d if valid else np.zeros_like(d),F,info


def records_from(executor,actor,outcomes,traces,ordinary):
    owner=policy_hash(actor);records=[]
    for row,trace,is_ordinary in zip(outcomes,traces,ordinary):
        if row.actor_hash!=owner or not row.complete or not row.deterministic:raise ValueError('Actual own complete mean outcomes required')
        path=executor.archive.path/trace['raw']['file']
        with np.load(path,allow_pickle=False) as z:
            saved={k:z[k].copy() for k in ('states','actions','raw_costs','query_ids','owner_json')}
        o=json.loads(str(saved.pop('owner_json')))
        if o['actor_hash']!=owner or o['semantics']!='actor_suffix':raise ValueError('Foreign controller target')
        records.append(dict(**saved,actor_hash=owner,deterministic=True,continuation='uninterrupted_actor',
            ordinary=bool(is_ordinary),raw=trace['raw']))
    return records


def admissible(base,proposed,ordinary,margin=.001):
    """Accept actual ordinary improvement and no acquired feasibility regression."""
    bo=[r for r,flag in zip(base,ordinary) if flag];po=[r for r,flag in zip(proposed,ordinary) if flag]
    if not bo or len(base)!=len(proposed) or [r.context_id for r in base]!=[r.context_id for r in proposed]:raise ValueError('Paired complete contexts required')
    if not all(r.complete and r.deterministic for r in list(base)+list(proposed)):raise ValueError('Complete mean controller outcomes required')
    gain=np.mean([a.cost-b.cost for a,b in zip(bo,po)])
    return bool(gain>=1e-6 and sum(r.capture for r in po)>=sum(r.capture for r in bo) and all(r.capture for r in po))
