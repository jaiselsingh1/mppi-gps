"""Independent handwritten Q-query records; no simulator or neural calls."""
from copy import deepcopy
import numpy as np
import pytest

from src.gps.bidirectional.ending_qfilter_audit import (
    exact_pair_key,validate_pair,positive_map,SEMANTICS)


def fixture():
    initial=np.array([23.,.2]);teacher=np.array([-.1]);actor="actor";task="task";cid="controller"
    key=exact_pair_key(initial,teacher,actor,task);R=2
    pre=np.array([[23.,.2],[24.,.2],[23.,.2],[24.,.2]])
    post=pre.copy();post[:,0]+=1
    actions=np.array([[.5],[.2],[-.1],[.2]]);costs=np.array([1.,2.,.5,.5])
    obs=np.zeros((4,8));obs[:,0]=pre[:,1];obs[:,1]=pre[:,0]/25;obs[:,2]=1-obs[:,1]
    qids=[f"q{i}" for i in range(4)]
    raw=dict(pair_key=np.array(key),status=np.array("complete"),initial_state=initial,
        remaining=np.array(R),actor_hash=np.array(actor),actor_controller_hash=np.array(cid),task_hash=np.array(task),
        branch_indices=np.array([0,0,1,1]),steps=np.array([0,1,0,1]),prestates=pre,states=post,
        observations=obs,actions=actions,raw_costs=costs,query_ids=np.array(qids),
        first_actions=np.array([[.5],[-.1]]),complete_branch_costs=np.array([3.,1.]))
    request=dict(id="request",pair_id="pair-0000",pair_key=key,actor_hash=actor,actor_controller_hash=cid,
        task_hash=task,family=0,initial_state=initial,remaining=R,teacher_action=teacher,
        semantics=SEMANTICS,maximum_additional_work=dict(model_controls=4,actor_rows=3))
    event=dict(pair_id="pair-0000",pair_key=key,actor_hash=actor,task_hash=task,remaining=R,
        query_ids=qids,q_actor=3.,q_teacher=1.,weight=2.)
    queries={}
    for j,qid in enumerate(qids):
        branch,step=divmod(j,R)
        owner=dict(semantics=SEMANTICS,actor_hash=actor,actor_controller_hash=cid,task_hash=task,
            family=0,pair_id="pair-0000",pair_key=key,branch=branch,
            branch_role="actor_mean" if branch==0 else "actual_teacher",step=step,remaining=R)
        queries[qid]=dict(request=dict(mode="model",query_kind="qfilter_actor_completion_step",owner=owner),
            inputs=dict(initial_state=pre[j],actions=actions[j].reshape(1,1,1)),
            result=dict(states=post[j].reshape(1,1,2),totals=np.array([costs[j]])))
    return raw,request,event,initial,teacher,queries,task,cid


def check(value):
    raw,request,event,initial,teacher,queries,task,cid=value
    return validate_pair(raw,request,event,initial,teacher,queries.__getitem__,task,cid)


def test_independent_complete_q_uses_full_costs_and_exact_teacher_first_action():
    value=fixture();result=check(value)
    assert result["model_controls"]==4 and result["actor_rows"]==3 and result["weight"]==2.
    changed=deepcopy(value);changed[0]["first_actions"][1,0]=.1
    with pytest.raises(ValueError,match="actual label"):check(changed)


@pytest.mark.parametrize("field",["actor_hash","branch_role","step"])
def test_independent_q_rejects_foreign_tail_or_wrong_branch_owner(field):
    value=fixture();owner=value[5]["q3"]["request"]["owner"]
    owner[field]="foreign" if field!="step" else 7
    with pytest.raises(ValueError,match="owner differs"):check(value)


def test_independent_q_rejects_prefix_only_and_invented_positive_weight():
    value=fixture();value[0]["states"][-1,0]=24
    with pytest.raises(ValueError,match="true ending"):check(value)
    value=fixture();value[2]["weight"]=3.
    with pytest.raises(ValueError,match="positive weight"):check(value)


def test_exact_q_cache_and_positive_resampling_are_byte_sensitive():
    a=np.array([.1]);s=np.array([3.,.2]);key=exact_pair_key(s,a,"actor","task")
    assert exact_pair_key(s,np.nextafter(a,1.),"actor","task")!=key
    weights=np.zeros(200);weights[[3,11,198]]=[.1,.2,.3]
    positive,indices=positive_map(weights)
    np.testing.assert_array_equal(positive,[3,11,198])
    np.testing.assert_array_equal(indices,np.array([3,11,198])[np.linspace(0,2,200,dtype=np.int64)])
    positive,indices=positive_map(np.zeros(200));assert not len(positive)
    np.testing.assert_array_equal(indices,np.arange(200))
