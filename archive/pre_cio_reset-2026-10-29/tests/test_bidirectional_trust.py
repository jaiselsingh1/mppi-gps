"""Synthetic trust/evidence arithmetic only: no simulator or actor inference."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.gps.bidirectional.trust import Outcome, TrustManager, compatibility_scores


def outcome(context="root", *, query="q", actor="actor", controller="actor-controller",
            cost=10., capture=True, complete=True, remaining=80, semantics="actor_suffix",
            task="task", family=0, deterministic=True):
    return Outcome(context, cost, capture, complete, controller, actor, family,
                   remaining, task, (query,), semantics, deterministic)


def fixture(*, successes=10, qualify=True, actor="actor", task="task"):
    manager = TrustManager()
    remaining = [70,71,72,73,74,75,76,77,78,80]
    anchors = np.zeros((10,8)); anchors[:,6] = np.asarray(remaining)/100.
    record = manager.register(actor_hash=actor, actor_controller_hash="actor-controller",
        task_hash=task, family_id=0, total_steps=100, time_interval=(20,30),
        remaining_interval=(70,80), support_anchors=anchors, observation_scale=np.ones(8),
        support_radius=.06, anchor_context_ids=tuple(f"anchor{i}" for i in range(10)))
    rows = [outcome(f"anchor{i}", query=f"handoff{i}", actor=actor, task=task,
                    remaining=n, capture=i < successes) for i,n in enumerate(remaining)]
    if qualify:
        record = manager.qualify(record.record_id, rows)
    return manager, record, rows


def panel(weight=.1, *, cost_gain=1., base_captures=2, guided_captures=2, remaining=100,
          prefix="", guided_actor="actor"):
    base = [outcome(f"root{i}", query=f"{prefix}base{i}", actor=None,
                    controller="baseline", cost=10.+i, capture=i<base_captures,
                    remaining=remaining, semantics="executed_replanning") for i in range(3)]
    guided = [outcome(f"root{i}", query=f"{prefix}guided{i}", actor=guided_actor,
                      controller=f"guided-{weight}", cost=10.+i-cost_gain,
                      capture=i<guided_captures, remaining=remaining,
                      semantics="executed_replanning") for i in range(3)]
    return base,guided


def promote(manager, record, weight=.1, **kwargs):
    base,guided = panel(weight, **kwargs)
    return manager.promote(record.record_id, weight, base, guided,
        baseline_controller_hash="baseline", guided_controller_hash=f"guided-{weight}")


def observation(remaining=80):
    result=np.zeros(8); result[6]=remaining/100.
    return result


def test_immutable_identity_and_support_arrays():
    manager, record, _ = fixture(qualify=False)
    with pytest.raises(FrozenInstanceError):
        record.actor_hash="another"
    with pytest.raises(ValueError):
        record.support_anchors[0,0]=100
    with pytest.raises(ValueError):
        record.support_anchors.flags.writeable=True
    assert manager.records == (record,)


def test_invalid_elapsed_remaining_intervals_rejected_before_registration():
    manager, record, _=fixture(qualify=False)
    with pytest.raises(ValueError, match="intervals disagree"):
        manager.register(actor_hash="actor",actor_controller_hash="actor-controller",task_hash="task",
            family_id=1,total_steps=100,time_interval=(0,10),remaining_interval=(70,80),
            support_anchors=np.zeros((1,8)),observation_scale=np.ones(8),support_radius=.1,
            anchor_context_ids=("a",))
    assert manager.get(record.record_id).state == "absent"


def test_ninety_percent_handoffs_are_eligible_but_never_deployed():
    manager,record,_=fixture(successes=9)
    assert record.state == "eligible" and record.handoff_success_rate == .9
    assert record.lambda_weight == 0.
    assert manager.weight("actor","task",0,observation(),80) == 0.
    assert manager.qualification_weight("actor","task",0,observation(),80,.1) == .1
    assert manager.qualification_weight("actor","task",0,observation(),80,1.) == 0.
    assert manager.get(record.record_id).state == "eligible"


def test_failed_handoffs_have_zero_trial_and_deployment_weight():
    manager,record,_=fixture(successes=8)
    assert record.state == "revoked"
    assert manager.weight("actor","task",0,observation(),80) == 0.
    assert manager.qualification_weight("actor","task",0,observation(),80,.1) == 0.


@pytest.mark.parametrize("change",[
    {"actor_hash":"wrong"},{"controller_hash":"actor"},{"task_hash":"wrong"},
    {"complete":False},{"semantics":"executed_replanning"},{"deterministic":False},
])
def test_handoff_ownership_and_completion_are_enforced(change):
    manager,record,rows=fixture(qualify=False)
    rows[0]=replace(rows[0],**change)
    with pytest.raises(ValueError):
        manager.qualify(record.record_id,rows)
    assert manager.get(record.record_id).state == "revoked"


def test_all_support_anchors_and_both_horizon_endpoints_need_outcomes():
    manager,record,rows=fixture(qualify=False)
    with pytest.raises(ValueError,match="every declared support anchor"):
        manager.qualify(record.record_id,rows[:-1])
    manager,record,rows=fixture(qualify=False)
    rows[-1]=replace(rows[-1],remaining_steps=79)
    with pytest.raises(ValueError,match="endpoints"):
        manager.qualify(record.record_id,rows)


def test_handoff_refresh_cannot_duplicate_old_physical_evidence():
    manager,record,rows=fixture()
    with pytest.raises(ValueError,match="not fresh"):
        manager.qualify(record.record_id,rows)
    assert manager.get(record.record_id).state == "revoked"
    fresh=[replace(r,query_ids=(f"new-{i}",)) for i,r in enumerate(rows)]
    assert manager.qualify(record.record_id,fresh).state == "eligible"


def test_paired_complete_replanning_promotes_and_ladder_cannot_jump():
    manager,record,_=fixture()
    first=promote(manager,record)
    assert first.state == "promoted" and first.lambda_weight == .1
    assert first.promotion_mean_gain == 1.
    assert len(first.promotion_query_ids) == 6
    assert manager.weight("actor","task",0,observation(),80) == .1
    second=promote(manager,first,1.,prefix="second-")
    assert second.lambda_weight == 1.
    third=promote(manager,second,10.,prefix="third-")
    assert third.lambda_weight == 10.
    manager,record,_=fixture()
    with pytest.raises(ValueError,match="next"):
        promote(manager,record,1.)
    assert manager.get(record.record_id).lambda_weight == 0.


@pytest.mark.parametrize("kwargs",[
    {"cost_gain":0.},{"cost_gain":1e-7},
    {"cost_gain":1.,"base_captures":3,"guided_captures":2},
])
def test_worse_controller_or_capture_regression_revokes(kwargs):
    manager,record,_=fixture()
    result=promote(manager,record,**kwargs)
    assert result.state == "revoked" and result.lambda_weight == 0.
    assert len(result.promotion_query_ids) == 6


@pytest.mark.parametrize("corruption",["wrong_actor","short","contexts","wrong_controller","shared_query","wrong_semantics"])
def test_fake_controller_promotions_are_rejected(corruption):
    manager,record,_=fixture()
    base,guided=panel()
    if corruption == "wrong_actor": guided[0]=replace(guided[0],actor_hash="other")
    if corruption == "short":
        base=[replace(r,remaining_steps=80) for r in base]
        guided=[replace(r,remaining_steps=80) for r in guided]
    if corruption == "contexts": guided[0]=replace(guided[0],context_id="different")
    if corruption == "wrong_controller": guided[0]=replace(guided[0],controller_hash="other")
    if corruption == "shared_query": guided[0]=replace(guided[0],query_ids=base[0].query_ids)
    if corruption == "wrong_semantics": guided[0]=replace(guided[0],semantics="actor_suffix")
    with pytest.raises(ValueError):
        manager.promote(record.record_id,.1,base,guided,
                        baseline_controller_hash="baseline",guided_controller_hash="guided-0.1")
    assert manager.get(record.record_id).state == "revoked"


def test_unsupported_versions_modes_clock_and_states_have_zero_weight():
    manager,record,_=fixture(); promote(manager,record)
    obs=observation()
    assert manager.weight("bad","task",0,obs,80) == 0.
    assert manager.weight("actor","bad",0,obs,80) == 0.
    assert manager.weight("actor","task",1,obs,80) == 0.
    assert manager.weight("actor","task",0,obs+10.,80) == 0.
    assert manager.weight("actor","task",0,np.full(8,np.nan),80) == 0.
    assert manager.weight("actor","task",0,obs,81) == 0.
    assert manager.weight_for("actor","task",0,21,80,obs) == 0.
    assert manager.weight_for("actor","task",0,20,80,obs) == .1
    manager.set_version("new-actor","task")
    assert manager.get(record.record_id).state == "revoked"
    assert manager.weight("actor","task",0,obs,80) == 0.


def test_revocation_keeps_old_snapshot_but_manager_returns_zero():
    manager,record,_=fixture(); old=promote(manager,record)
    manager.revoke_actor("actor","new failed qualification")
    assert old.state == "promoted"  # retained immutable history, not the manager's current decision
    assert manager.get(record.record_id).state == "revoked"
    assert manager.weight("actor","task",0,observation(),80) == 0.
    assert manager.qualification_weight("actor","task",0,observation(),80,.1) == 0.


def test_vectorized_support_and_horizon_matches_scalar_checks():
    manager,record,_=fixture(); promote(manager,record)
    obs=np.stack([[observation(80-h) for h in range(12)] for _ in range(2)])
    obs[0,2,0]=1.
    obs[1,3,1]=np.nan
    batch=manager.weights_for_batch("actor","task",0,obs,80)
    expected=np.array([[manager.weight("actor","task",0,row,80-h)
                        for h,row in enumerate(candidate)] for candidate in obs])
    np.testing.assert_array_equal(batch,expected)
    assert batch[0,0] == .1 and batch[0,-1] == 0.


def test_zero_influence_never_calls_bad_policy():
    manager,record,_=fixture()  # eligibility alone cannot influence search
    def bad_policy(*args):
        raise AssertionError("Unqualified policy was queried")
    obs=np.tile(observation(),(2,4,1)); actions=np.ones((2,4,2))
    actual=compatibility_scores(actions,obs,bad_policy,manager,
        actor_hash="actor",task_hash="task",family=0,remaining_start=80)
    np.testing.assert_array_equal(actual,np.zeros(2))
    assert manager.get(record.record_id).state == "eligible"


def test_compatibility_is_batched_supported_and_does_not_mutate_independent_bank():
    manager,record,_=fixture(); promote(manager,record)
    obs=np.stack([[observation(80-h) for h in range(4)] for _ in range(2)])
    obs[0,1,0]=10.
    actions=np.ones((2,4,2)); saved_actions=actions.copy(); saved_obs=obs.copy()
    calls=[]
    def mean(values,family):
        calls.append((values.copy(),family))
        return np.zeros((len(values),2))
    score=compatibility_scores(actions,obs,mean,manager,
        actor_hash="actor",task_hash="task",family=0,remaining_start=80)
    np.testing.assert_allclose(score,[.6,.8],atol=1e-15)
    assert len(calls) == 1 and calls[0][0].shape == (7,8) and calls[0][1] == 0
    np.testing.assert_array_equal(actions,saved_actions)
    np.testing.assert_array_equal(obs,saved_obs)


def test_nonfinite_actor_output_cannot_supply_compatibility_score():
    manager,record,_=fixture(); promote(manager,record)
    with pytest.raises(ValueError,match="output invalid"):
        compatibility_scores(np.ones((1,1,2)),observation()[None,None,:],
            lambda values,family:np.full((len(values),2),np.nan),manager,
            actor_hash="actor",task_hash="task",family=0,remaining_start=80)


def test_outcome_rejects_nonfinite_costs_and_implicit_flags():
    with pytest.raises(ValueError,match="finite"):
        outcome(cost=np.nan)
    with pytest.raises(ValueError,match="bools"):
        outcome(capture=1)
    with pytest.raises(ValueError,match="unique"):
        replace(outcome(),query_ids=("q","q"))
    with pytest.raises(ValueError,match="explicit sequence"):
        replace(outcome(),query_ids="query-string")


def test_records_cannot_fabricate_qualification_or_change_execution_mode():
    _,record,_=fixture(qualify=False)
    with pytest.raises(ValueError,match="actor-suffix evidence"):
        replace(record,state="eligible")
    with pytest.raises(ValueError,match="deterministic"):
        replace(record,deterministic=False)


def test_promotion_retains_matched_raw_outcomes_and_costs():
    manager,record,_=fixture()
    result=promote(manager,record)
    assert len(result.promotion_pairs) == 3
    for base,guided in result.promotion_pairs:
        assert base.context_id == guided.context_id
        assert base.cost-guided.cost == 1.
        assert base.actor_hash is None and guided.actor_hash == record.actor_hash
        assert set(base.query_ids).isdisjoint(guided.query_ids)


def test_numerical_overflow_in_paired_mean_revokes_instead_of_promoting():
    manager,record,_=fixture();record=promote(manager,record)
    base,guided=panel(1.,prefix="fresh-")
    base=[replace(row,cost=1e308) for row in base]
    guided=[replace(row,cost=-1e308) for row in guided]
    with pytest.raises(FloatingPointError):
        manager.promote(record.record_id,1.,base,guided,
            baseline_controller_hash="baseline",guided_controller_hash="guided-1.0")
    assert manager.get(record.record_id).state == "revoked"
    assert manager.weight("actor","task",0,observation(),80) == 0.


def test_boolean_weight_cannot_stand_in_for_ladder_one():
    manager,record,_=fixture();record=promote(manager,record)
    assert manager.qualification_weight("actor","task",0,observation(),80,True) == 0.
    with pytest.raises(ValueError,match="explicit finite"):
        promote(manager,record,True,prefix="fresh-")


@pytest.mark.parametrize('weight',[.1,1.,10.])
def test_separate_candidate_trial_needs_actor_evidence_and_executed_promotion(weight):
    from src.gps.bidirectional.trust import CandidateTrustManager
    _,template,handoffs=fixture()
    manager=CandidateTrustManager(weight)
    record=manager.register(**{key:getattr(template,key) for key in (
        'actor_hash','actor_controller_hash','task_hash','family_id','total_steps','time_interval',
        'remaining_interval','support_anchors','observation_scale','support_radius','anchor_context_ids')})
    assert manager.qualification_weight('actor','task',0,observation(),80,weight)==0.
    manager.qualify(record.record_id,handoffs)
    assert manager.weight('actor','task',0,observation(),80)==0.
    assert manager.qualification_weight('actor','task',0,observation(),80,weight)==weight
    promoted=promote(manager,record,weight)
    assert promoted.state=='promoted' and manager.weight('actor','task',0,observation(),80)==weight
    assert manager.weight('changed-version','task',0,observation(),80)==0.


def test_failed_candidate_is_revoked_instead_of_deployed():
    from src.gps.bidirectional.trust import CandidateTrustManager
    _,template,handoffs=fixture()
    manager=CandidateTrustManager(10.)
    record=manager.register(**{key:getattr(template,key) for key in (
        'actor_hash','actor_controller_hash','task_hash','family_id','total_steps','time_interval',
        'remaining_interval','support_anchors','observation_scale','support_radius','anchor_context_ids')})
    manager.qualify(record.record_id,handoffs)
    assert promote(manager,record,10.,cost_gain=-1.).state=='revoked'
    assert manager.qualification_weight('actor','task',0,observation(),80,10.)==0.
    assert manager.weight('actor','task',0,observation(),80)==0.


def test_candidate_from_fixed_eligible_packet_is_isolated_and_reuses_only_handoffs():
    from src.gps.bidirectional.trust import CandidateTrustManager
    source,record,_=fixture()
    first=CandidateTrustManager.from_eligible(source,.1)
    second=CandidateTrustManager.from_eligible(source,10.)
    assert first.records==second.records==source.records
    assert first.get(record.record_id).lambda_weight==0.
    assert first._used_queries[record.record_id] is not source._used_queries[record.record_id]
    assert first._used_queries[record.record_id]==set(record.handoff_query_ids)
    assert promote(first,record,.1,prefix="candidate-only-").state=="promoted"
    assert source.get(record.record_id).state==second.get(record.record_id).state=="eligible"
    assert not second._used_queries[record.record_id].intersection(
        first.get(record.record_id).promotion_query_ids)
    assert second.qualification_weight('actor','task',0,observation(),80,10.)==10.
    assert CandidateTrustManager.from_eligible(first,1.).records==()


def test_candidate_reconstruction_rejects_contaminated_eligibility_packet():
    from src.gps.bidirectional.trust import CandidateTrustManager
    source,record,_=fixture()
    source._used_queries[record.record_id].add("unrelated-controller-query")
    with pytest.raises(ValueError,match="only its original handoff"):
        CandidateTrustManager.from_eligible(source,1.)
