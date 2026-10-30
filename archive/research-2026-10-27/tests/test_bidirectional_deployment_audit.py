"""Pure verifier fixtures: measured outcomes and feasibility outrank fit loss."""
import numpy as np
from src.gps.bidirectional.deployment_audit_helpers import actual_acceptance,metric_replay,euclidean_replay,barrier


def outcomes(cost=2.,capture=True):
    return [dict(context_id=str(i),cost=cost,capture=capture) for i in range(8)]


def traces(ending=-.2):
    return [dict(states=np.array([[0.,.3],[25.,ending]])) for _ in range(8)]


def test_actual_cost_descent_cannot_override_acquired_ending_regression():
    base=outcomes();changed=outcomes(1.9)
    assert actual_acceptance(base,changed,traces(),traces())[0]
    assert not actual_acceptance(base,changed,traces(.2),traces(.21))[0]
    assert actual_acceptance(base,changed,traces(.2),traces(.19))[0]


def test_capture_and_cost_both_required_for_changed_controller():
    assert not actual_acceptance(outcomes(),outcomes(1.,False),traces(),traces())[0]
    assert not actual_acceptance(outcomes(),outcomes(2.),traces(),traces())[0]


def test_metric_has_feasible_time_redistribution_descent():
    # Ending depends on the sum; cost prefers moving effort to the second step.
    g=np.array([1.,0.]);A=np.array([[1.,1.]]);b=np.array([0.]);J=np.eye(2)
    d,F,ok=metric_replay(g,A,b,J)
    assert ok and g@d<0 and A[0]@d<=1e-7 and np.sqrt(d@F@d)<=.05+1e-7
    z,ok=euclidean_replay(g,A,np.array([-1.]))
    assert not ok and np.array_equal(z,np.zeros(2))


def test_relaxed_constraint_is_finite_for_failed_measured_ending():
    assert np.isfinite(barrier(-.5)) and barrier(-.5)>barrier(.1)>barrier(.2)
