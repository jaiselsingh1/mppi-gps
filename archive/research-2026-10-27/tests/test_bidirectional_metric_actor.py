"""Pure constrained action-metric checks; no simulator or neural execution."""
import numpy as np
from src.gps.bidirectional.deployment_adjoint import metric_feasible_descent
from src.gps.bidirectional.deployment_loop import METRIC_DEFAULT,SERIAL_DEFAULT,work_bound

def test_action_metric_enforces_ball_and_ending_halfspace_together():
    g=np.array([1.,1.]);A=np.array([[1.,0.]]);b=np.array([-.01]);J=np.array([[1.,0.],[0.,.01]])
    d,F,info=metric_feasible_descent(g,A,b,J)
    assert info['valid'] and g@d<0 and (A@d-b).max()<=1e-7
    assert np.sqrt(d@F@d)<=.05+1e-7
    assert info['halfspace_checks']==64 and info['metric_ball_checks']==64 and info['linear_solves']==1

def test_impossible_ending_constraint_does_not_generate_a_false_improvement():
    d,F,info=metric_feasible_descent(np.ones(2),np.zeros((1,2)),np.array([-.01]),np.eye(2))
    assert not info['valid'];np.testing.assert_array_equal(d,np.zeros(2))

def test_metric_work_counts_manual_updates_and_numeric_work_separately():
    b=work_bound(METRIC_DEFAULT)
    assert b['total_controls']==3222784 and b['actor_rows']==464960
    assert b['adam_steps']==b['backward_calls']==128 and b['diagnostic_backward_passes']==32
    assert b['projected_parameter_updates']==b['metric_linear_solves']==192
    assert b['metric_ball_checks']==12288 and b['projection_halfspace_checks']==98304
    assert b['active_jacobian_rows']==b['analytic_policy_jacobian_rows']==15360

def test_scalar_layout_repair_changes_no_learning_or_budget_setting():
    unchanged={k:v for k,v in SERIAL_DEFAULT.items() if k not in ('schema','root_base','feature_forward_layout')}
    previous={k:v for k,v in METRIC_DEFAULT.items() if k not in ('schema','root_base')}
    assert unchanged==previous and work_bound(SERIAL_DEFAULT)==work_bound(METRIC_DEFAULT)
