"""Pure saved-evidence fixtures: no task transitions or actor evaluations."""
import copy
import json

import numpy as np
import pytest

from src.gps.bidirectional.early_context_audit import (
    actual_state_context, complete_mc_statistics, no_action_auxiliary, paired_improvement,
)


def mc_episode(index, *, owner="parent", context="ctx", costs=(1., 2.)):
    n = len(costs)
    return dict(episode=f"mc-{index}", raw=dict(
        owner_json=np.array(json.dumps(dict(actor_hash=owner, semantics="actor_suffix", deterministic=False, family=0))),
        states=np.column_stack((np.arange(25-n, 26), np.zeros(n+1))),
        raw_costs=np.asarray(costs), complete=np.array(True), context_id=np.array(context),
        query_ids=np.asarray([f"q-{index}-{j}" for j in range(n)])))


def test_own_mc_targets_reconstruct_true_complete_returns_and_loo():
    episodes = [mc_episode(i, costs=(float(i+1), 2.)) for i in range(4)]
    result = complete_mc_statistics(episodes, actor_hash="parent")
    returns = np.array([3., 2., 4., 2., 5., 2., 6., 2.])
    baseline = np.array([5., 2., 14/3, 2., 13/3, 2., 4., 2.])
    np.testing.assert_array_equal(result["returns"], returns)
    np.testing.assert_allclose(result["baseline"], baseline)
    np.testing.assert_allclose(result["advantages"], returns-baseline)
    assert result["controls"] == 8 and result["complete_episodes"] == 4


def test_old_owner_returns_cannot_label_updated_actor():
    with pytest.raises(ValueError, match="different actor"):
        complete_mc_statistics([mc_episode(i) for i in range(4)], actor_hash="updated")


def test_mc_mixed_actor_owner_rejected():
    episodes = [mc_episode(i) for i in range(4)]
    episodes[-1]["raw"]["owner_json"] = np.array(json.dumps(dict(
        actor_hash="other", semantics="actor_suffix", deterministic=False, family=0)))
    with pytest.raises(ValueError, match="different actor"):
        complete_mc_statistics(episodes, actor_hash="parent")


def test_mc_requires_true_ending_and_replication():
    with pytest.raises(ValueError, match="replicate"):
        complete_mc_statistics([mc_episode(i) for i in range(3)], actor_hash="parent")
    episodes = [mc_episode(i) for i in range(4)]
    episodes[0]["raw"]["states"][-1, 0] = 24
    with pytest.raises(ValueError, match="true ending"):
        complete_mc_statistics(episodes, actor_hash="parent")


def test_mc_physical_queries_cannot_alias_between_episodes():
    episodes = [mc_episode(i) for i in range(4)]
    episodes[1]["raw"]["query_ids"] = episodes[0]["raw"]["query_ids"].copy()
    with pytest.raises(ValueError, match="alias"):
        complete_mc_statistics(episodes, actor_hash="parent")


def source_fixture():
    raw = dict(owner_json=np.array(json.dumps(dict(semantics="executed_replanning", actor_hash=None))),
        complete=np.array(True), states=np.column_stack((np.arange(26), np.linspace(.3, -.2, 26))),
        query_ids=np.asarray([f"actual-{i}" for i in range(25)]))
    context = dict(elapsed=6, source_state_index=6, source_query_id="actual-5", state=raw["states"][6].tolist())
    return raw, context


def test_context_uses_actual_early_boundary_without_action_labels():
    raw, context = source_fixture()
    np.testing.assert_array_equal(actual_state_context(raw, context), raw["states"][6])


@pytest.mark.parametrize("key", ["actions", "action", "target_actions", "teacher_actions", "returns", "raw_costs"])
def test_context_rejects_action_or_return_labels(key):
    raw, context = source_fixture(); context[key] = [0.]
    with pytest.raises(ValueError, match="exports action or return"):
        actual_state_context(raw, context)


def test_context_forecast_or_wrong_boundary_rejected():
    raw, context = source_fixture(); context["source_query_id"] = "actual-6"
    with pytest.raises(ValueError, match="boundary"):
        actual_state_context(raw, context)
    raw, context = source_fixture()
    raw["owner_json"] = np.array(json.dumps(dict(semantics="hypothetical_model_plan")))
    with pytest.raises(ValueError, match="actual complete"):
        actual_state_context(raw, context)


def test_no_auxiliary_even_if_aux_weight_zero():
    fit = dict(target_owner="behavior_actor", auxiliary_minibatches=0,
        auxiliary_loss_mean=0., weighted_auxiliary_loss_mean=0.)
    no_action_auxiliary(fit)
    for field in ("auxiliary_minibatches", "auxiliary_loss_mean", "weighted_auxiliary_loss_mean"):
        broken = copy.deepcopy(fit); broken[field] = 1
        with pytest.raises(ValueError): no_action_auxiliary(broken)


def test_complete_gain_cannot_hide_capture_regression():
    base = [dict(context_id=str(i), cost=2., capture=True) for i in range(16)]
    candidate = [dict(context_id=str(i), cost=1., capture=(i != 0)) for i in range(16)]
    result = paired_improvement(base, candidate)
    assert result["mean_cost_gain"] == 1. and result["capture_rate"] > .9
    assert not result["capture_nonregression"] and not result["improved"]


def test_paired_complete_gain_threshold_and_matching_resets():
    base = [dict(context_id=str(i), cost=2., capture=True) for i in range(16)]
    candidate = [dict(context_id=str(i), cost=1.998, capture=True) for i in range(16)]
    assert paired_improvement(base, candidate)["improved"]
    candidate[0]["context_id"] = "foreign"
    with pytest.raises(ValueError, match="reset clusters"):
        paired_improvement(base, candidate)


def test_declared_full_loop_budget_is_typed_and_finite():
    from src.gps.bidirectional.early_context_audit import declared_bound
    b = declared_bound()
    assert (b["total_controls"], b["model_controls"], b["executed_controls"]) == (1246436, 1166624, 79812)
    assert b["actor_rows"] == 307792 and b["adam_steps"] == 288
    assert b["rows_per_update"] == 1408 and b["adam_per_update"] == 24
    assert b["diagnostic_backward_passes"] == 12


def guard_fixture(passed=True):
    from src.gps.bidirectional.early_context_audit import base_completion_protocol
    from src.gps.bidirectional.ending_audit import controller_identity
    actor, task = "parent", "task"
    cid = controller_identity(dict(actor_hash=actor, family=0, deterministic=True))
    inner, outer = base_completion_protocol(actor, cid, task)
    initial = np.array([24., -.2 if passed else .2]);post = initial.copy();post[0] = 25.
    request = dict(id="guard", protocol_identity=outer, base_protocol_identity=inner,
        actor_hash=actor, actor_controller_hash=cid, context_id="ordinary:1", initial_state=initial.tolist(),
        elapsed=24, remaining=1, trial_weight=1., maximum_additional_work=dict(model_controls=1, actor_rows=1),
        utc="2026-10-27T00:00:01+00:00")
    trace = dict(prestate=initial.tolist(), action=[0.], state=post.tolist(), cost=0., query_id="guard-q")
    event = dict(request_id="guard", actor_hash=actor, protocol_identity=outer, passed=passed,
        query_ids=["guard-q"], traces=[trace], complete_cost=0., not_actor_training_target=True,
        utc="2026-10-27T00:00:03+00:00")
    selection = dict(request_id="guard", protocol_identity=outer, base_protocol_identity=inner,
        completion_query_ids=["completion-q"], eligibility_query_ids=["guard-q"], selected_first_action=[0.],
        changed=False, guidance_weight=1., selected_candidate_id="candidate-0") if passed else None
    decision = dict(guided_query_ids=["guard-q", "completion-q"] if passed else ["guard-q"], changed=False,
        utc="2026-10-27T00:00:04+00:00")
    q = dict(request=dict(mode="model", query_kind="early_current_actor_eligibility_step",
        owner=dict(semantics="deterministic_current_actor_eligibility_only", request_id="guard",
            protocol_identity=outer, actor_hash=actor, actor_controller_hash=cid, step=0, remaining=1,
            context_id="ordinary:1", training_target=False), utc="2026-10-27T00:00:02+00:00"),
        inputs=dict(initial_state=initial.copy(), actions=np.zeros((1,1,1))),
        result=dict(states=post.reshape(1,1,2), totals=np.array([0.])))
    return request,event,selection,decision,initial,np.array([0.]),lambda _:q,task


def test_paid_guard_completes_true_ending_and_links_actual_completion():
    from src.gps.bidirectional.early_context_audit import validate_guard
    r = validate_guard(*guard_fixture(True))
    assert r["passed"] and r["model_controls"] == 1


def test_paid_guard_failure_requires_zero_influence_fallback():
    from src.gps.bidirectional.early_context_audit import validate_guard
    fixture = list(guard_fixture(False))
    assert not validate_guard(*fixture)["passed"]
    fixture[3]["changed"] = True
    with pytest.raises(ValueError, match="retained guidance"):
        validate_guard(*fixture)


def test_paid_guard_rejects_old_owner_or_hypothetical_training_credit():
    from src.gps.bidirectional.early_context_audit import validate_guard
    fixture = list(guard_fixture(True)); fixture[1]["not_actor_training_target"] = False
    with pytest.raises(ValueError, match="target semantics"):
        validate_guard(*fixture)
    fixture = list(guard_fixture(True)); fixture[2]["eligibility_query_ids"] = ["wrong"]
    with pytest.raises(ValueError, match="not bound"):
        validate_guard(*fixture)
