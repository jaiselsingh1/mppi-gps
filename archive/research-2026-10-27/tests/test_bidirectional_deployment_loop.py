"""Pure roster, acceptance and data-boundary checks before the new full study."""
from copy import deepcopy
from dataclasses import dataclass
import json
import numpy as np
import pytest
from src.gps.bidirectional.deployment_loop import work_bound,load_config,DEFAULT,PROJECTED_DEFAULT,violation
from src.gps.bidirectional.deployment_adjoint import admissible,feasible_descent

@dataclass
class Outcome:
    context_id:str
    cost:float
    capture:bool=True
    complete:bool=True
    deterministic:bool=True

def test_full_bound_charges_every_rejected_fraction_both_children_and_final():
    b=work_bound();assert b['adjoint_steps']==192 and b['MC_training_controls']==22656
    assert b['model_controls']==3000816 and b['executed_controls']==221968
    assert b['total_controls']==3222784 and b['actor_rows']==464960
    assert b['adam_steps']==b['backward_calls']==320 and b['diagnostic_backward_passes']==32
    assert b['active_jacobian_rows']==15360 and b['teacher_pairs']==112
    projected=work_bound(PROJECTED_DEFAULT)
    assert projected['total_controls']==b['total_controls'] and projected['actor_rows']==b['actor_rows']
    assert projected['adam_steps']==projected['backward_calls']==128 and projected['diagnostic_backward_passes']==1760

def test_only_frozen_schedule_allowed(tmp_path):
    assert load_config()==DEFAULT
    c=deepcopy(DEFAULT);c['initial_adjoint_steps']+=1;p=tmp_path/'changed.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError):load_config(p)

def test_ordinary_deployment_cost_cannot_be_hidden_by_easy_acquired_costs():
    flags=[True,False];base=[Outcome('ordinary',2.),Outcome('acquired',20.,False)]
    assert not admissible(base,[Outcome('ordinary',2.01),Outcome('acquired',1.)],flags)
    assert not admissible(base,[Outcome('ordinary',1.,False),Outcome('acquired',1.)],flags)
    assert admissible(base,[Outcome('ordinary',1.99),Outcome('acquired',20.,False)],flags)
    traces=[dict(states=np.array([[0.,.3],[25.,-.2]])),dict(states=np.array([[4.,.6],[25.,.1]]))]
    assert violation(traces,flags,.001)==pytest.approx(.201)

def test_projected_direction_can_redistribute_controls_while_preserving_endpoint():
    gradient=np.array([-1.,1.]);A=np.array([[1.,1.]])
    direction,info=feasible_descent(gradient,A,np.array([0.]))
    assert info['valid'] and gradient@direction<0 and (A@direction)[0]<=1e-7
    assert info['halfspace_checks']==64

def test_no_descent_or_impossible_constraint_returns_zero_influence():
    for A,b in [(np.array([[1.,0.]]),np.array([0.])),(np.zeros((1,2)),np.array([-.01]))]:
        direction,info=feasible_descent(np.array([-1.,0.]),A,b)
        assert not info['valid'];np.testing.assert_array_equal(direction,np.zeros(2))
