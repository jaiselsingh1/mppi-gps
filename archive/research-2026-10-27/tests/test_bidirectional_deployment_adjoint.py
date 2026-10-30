"""Finite executable falsifier of own-trajectory adjoint before training.

The five complete mean episodes below cost125 real scalar controls. No native
engine, planner, optimization step or pretrained actor is involved.
"""
import json
from dataclasses import asdict
import numpy as np
import pytest
import torch
from src.gps.bidirectional.ending_pilot import CountedEnding as _Ending
from src.gps.bidirectional.deployment_adjoint import (AdjointConfig,relaxed_barrier,barrier_numpy,
    trajectory_tangent,validate_record,records_from)

class RawLedger:
    def __init__(self,path):self.path=path;self.stream=path.open('x');self.scope='adjoint-gradient-contract';self.n=0
    def append(self,kind,**data):
        key=f'r{self.n}';self.n+=1
        self.stream.write(json.dumps(dict(id=key,kind=kind,**data),default=lambda a:np.asarray(a).tolist())+'\n');self.stream.flush();return key

class CountedEnding(_Ending):
    def transition(self,states,actions):
        n=int(np.prod(np.shape(states)[:-1])) if np.ndim(states)>1 else 1
        req=self.ledger.append('transition_attempt',controls=n,prestates=states,actions=actions)
        result=super().transition(states,actions)
        self.ledger.append('transition_complete',request_id=req,states=result[0],components=result[1],costs=result[2]);return result

def test_relaxed_barrier_matches_value_first_and_second_derivative_at_join():
    cfg=AdjointConfig();delta=cfg.relaxation
    for value in (delta-1e-7,delta,delta+1e-7):
        s=torch.tensor(value,dtype=torch.float64,requires_grad=True);b=relaxed_barrier(s)
        g,=torch.autograd.grad(b,s,create_graph=True);gg,=torch.autograd.grad(g,s)
        assert float(b.detach())==pytest.approx(float(barrier_numpy(value)),abs=1e-12)
        assert float(g.detach())==pytest.approx(-cfg.barrier_mu/delta,rel=2e-5)
        assert float(gg.detach())==pytest.approx(cfg.barrier_mu/delta**2,rel=3e-5)

def test_adjoint_rejects_teacher_or_another_actor_before_learning():
    with pytest.raises(ValueError):validate_record(dict(actor_hash=None,deterministic=True,continuation='uninterrupted_actor'),'own',{})
    with pytest.raises(ValueError):validate_record(dict(actor_hash='own',deterministic=False,continuation='uninterrupted_actor'),'own',{})

def test_own_closed_loop_adjoint_matches_actual_parameter_perturbations(tmp_path):
    from src.gps.bidirectional.ending_pilot import EndingContext,EndingExecutor,_episode
    from src.gps.bidirectional.evidence import Archive
    from src.gps.bidirectional.policy import FamilyGaussianActor
    from scripts.diagnostics.run_bidirectional_tests import DurableWork
    archive=Archive(tmp_path/'evidence');archive.work_scope='gradient-qualification'
    work=DurableWork(tmp_path/'local_work.jsonl');dynamics=CountedEnding(archive,work,125)
    ledger=RawLedger(tmp_path/'raw_ending_ledger.jsonl');object.__setattr__(dynamics,'ledger',ledger)
    executor=EndingExecutor(EndingContext(dynamics),archive)
    archive.append('test_actor_construction_requested',actor_constructions=1,orthogonal_initializers=3)
    actor=FamilyGaussianActor(seed=1001)
    with torch.no_grad():actor.fc_mean.bias.fill_(-.2)
    initial=executor.task.reset(513001)
    row,trace,_,_=_episode(executor,initial,'own_center',actor=actor,deterministic=True)
    record=records_from(executor,actor,[row],[trace],[True])[0];task=asdict(dynamics)
    cfg=AdjointConfig()
    running,ending,rows=trajectory_tangent(actor,record,task)
    loss=running+relaxed_barrier(task['ending_y']-cfg.margin-ending)
    gb,gw=torch.autograd.grad(loss,(actor.fc_mean.bias,actor.fc_mean.weight),retain_graph=True)
    eb,ew=torch.autograd.grad(ending,(actor.fc_mean.bias,actor.fc_mean.weight))
    def score():
        r,t,_,_=_episode(executor,initial,f'own_perturb_{dynamics.reserved}',actor=actor,deterministic=True)
        with np.load(archive.path/t['raw']['file'],allow_pickle=False) as z:
            return np.array([z['cost_components'][:,:2].sum()+barrier_numpy(task['ending_y']-cfg.margin-z['states'][-1,1]),z['states'][-1,1]])
    for parameter,index,gradient,end_gradient in [(actor.fc_mean.bias,0,float(gb[0]),float(eb[0])),(actor.fc_mean.weight,(0,0),float(gw[0,0]),float(ew[0,0]))]:
        original=float(parameter[index].detach());epsilon=.001
        with torch.no_grad():parameter[index]=original-epsilon
        minus=score()
        with torch.no_grad():parameter[index]=original+epsilon
        plus=score()
        with torch.no_grad():parameter[index]=original
        finite=(plus-minus)/(2*epsilon)
        assert abs(gradient)>1e-4
        assert gradient==pytest.approx(finite[0],rel=.004,abs=.003)
        assert end_gradient==pytest.approx(finite[1],rel=.004,abs=.003)
    assert rows==12 and dynamics.reserved==125
    work.close();archive.close();ledger.stream.close()

