"""Independent arithmetic fixtures; no actor, optimizer or simulator call."""
import numpy as np
import pytest
import json
from copy import deepcopy

from src.gps.bidirectional.own_occupancy_audit import difference_targets,validate_probe,SEMANTICS


def test_full_and_prefix_credit_use_the_same_paid_action_denominator():
    result=difference_targets(np.array([.2]),np.array([[.15],[.25]]),np.array([2.,5.]),np.array([.5,.4]))
    assert result["full_gradient"]==pytest.approx(30.)
    assert result["local_gradient"]==pytest.approx(-1.)
    assert result["null_gradient"]==0.


def test_clipped_bound_uses_actual_separation_not_fixed_point_one():
    result=difference_targets(np.array([.99]),np.array([[.94],[1.]]),np.array([2.,5.]),np.array([1.,1.]))
    assert result["denominator"]==pytest.approx(.06)
    assert result["full_gradient"]==pytest.approx(50.)
    assert result["local_gradient"]==0.


def test_wrong_perturbation_or_nonfinite_target_rejected():
    with pytest.raises(ValueError,match="perturbations"):
        difference_targets(np.array([.2]),np.array([[.1],[.3]]),np.array([2.,5.]),np.array([1.,1.]))
    with pytest.raises(ValueError,match="Invalid"):
        difference_targets(np.array([.2]),np.array([[.15],[.25]]),np.array([2.,np.nan]),np.array([1.,1.]))


def _probe_fixture():
    t=23;R=2;mean=np.array([.2]);state=np.array([23.,-.2])
    center=dict(states=np.column_stack((np.arange(26),np.full(26,-.2))),actions=np.full((25,1),.2),
        raw_costs=np.zeros(25),query_ids=np.array([f"live{i}" for i in range(25)]),
        owner_json=np.array(json.dumps(dict(actor_hash="actor",controller_hash="controller",deterministic=True,semantics="actor_suffix"))))
    pre=np.array([[23.,-.2],[24.,-.2],[23.,-.2],[24.,-.2]]);post=pre.copy();post[:,0]+=1
    first=np.array([[.15],[.25]]);actions=np.array([[.15],[.2],[.25],[.2]])
    obs=np.zeros((4,8));obs[:,0]=pre[:,1];obs[:,1]=pre[:,0]/25;obs[:,2]=1-obs[:,1]
    targets=dict(full_costs=[0.,0.],local_costs=[0.,0.],slope_full=0.,slope_local=0.,slope_null=0.,
        width=float(first[1,0]-first[0,0]),curvature_full=0.,curvature_local=0.,slope_sign_disagrees=False,
        endpoint_capture=[True,True],probes_clipped=False,postgate_invariant=True)
    ref=dict(file="center.npz",sha256="saved",bytes=1);ids=[f"model{i}" for i in range(4)]
    request=dict(probe_id="probe-0000",actor_hash="actor",actor_controller_hash="controller",task_hash="task",family=0,
        semantics=SEMANTICS,source_episode="occupancy",elapsed=t,initial_state=state,saved_mean=mean,first_actions=first,
        delta=.05,local_horizon=4,center_raw=ref,center_query_ids=["live23","live24"],center_full_cost=0.,center_local_cost=0.,
        maximum_additional_work=dict(model_controls=4,actor_rows=2))
    raw=dict(status=np.array("complete"),semantics=np.array(SEMANTICS),actor_hash=np.array("actor"),
        actor_controller_hash=np.array("controller"),task_hash=np.array("task"),initial_state=state,saved_mean=mean,
        first_actions=first,delta=np.array(.05),local_horizon=np.array(4),branch_indices=np.array([0,0,1,1]),steps=np.array([0,1,0,1]),
        prestates=pre,states=post,observations=obs,actions=actions,raw_costs=np.zeros(4),query_ids=np.array(ids),
        center_query_ids=np.array(["live23","live24"]),center_raw_costs=np.zeros(2),center_full_cost=np.array(0.),center_local_cost=np.array(0.),
        targets_json=np.array(json.dumps(targets)))
    event=dict(actor_hash="actor",task_hash="task",source_episode="occupancy",elapsed=t,initial_state=state.tolist(),
        saved_mean=mean.tolist(),center_raw=ref,center_query_ids=["live23","live24"],minus_query_ids=ids[:2],plus_query_ids=ids[2:],targets=deepcopy(targets))
    queries={}
    for j,qid in enumerate(ids):
        branch,step=divmod(j,2)
        owner=dict(semantics=SEMANTICS,actor_hash="actor",actor_controller_hash="controller",task_hash="task",family=0,
            probe_id="probe-0000",source_episode="occupancy",branch=branch,branch_role="minus" if branch==0 else "plus",step=step,remaining=2)
        queries[qid]=dict(request=dict(mode="model",query_kind="own_gradient_completion_step",owner=owner),
            inputs=dict(initial_state=pre[j],actions=actions[j].reshape(1,1,1)),
            result=dict(states=post[j].reshape(1,1,2),totals=np.zeros(1)))
    return raw,request,event,center,t,queries


def _check_probe(value):
    raw,request,event,center,t,queries=value
    return validate_probe(raw,request,event,center,t,queries.__getitem__,"task","controller")


def test_actual_center_owned_complete_probes_have_postgate_zero_signal():
    result=_check_probe(_probe_fixture())
    assert result["actor_rows"]==2 and result["model_controls"]==4
    assert result["targets"]["slope_full"]==result["targets"]["slope_local"]==0.


def test_own_probe_rejects_foreign_continuation_or_not_true_ending():
    value=_probe_fixture();value[5]["model2"]["request"]["owner"]["actor_hash"]="foreign"
    with pytest.raises(ValueError,match="continuation owner"):_check_probe(value)
    value=_probe_fixture();value[0]["states"][-1,0]=24
    with pytest.raises(ValueError,match="true ending"):_check_probe(value)
    value=_probe_fixture();value[1]["actor_hash"]="foreign"
    with pytest.raises(ValueError,match="actual center parent"):_check_probe(value)


def test_own_probe_rejects_invented_full_target_and_model_center_alias():
    value=_probe_fixture();value[2]["targets"]["slope_full"]=3.
    with pytest.raises(ValueError,match="target differs"):_check_probe(value)
    value=_probe_fixture();value[0]["query_ids"][0]="live23"
    with pytest.raises(ValueError,match="alias"):_check_probe(value)
