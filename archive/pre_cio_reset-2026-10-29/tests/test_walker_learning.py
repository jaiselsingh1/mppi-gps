"""Pure return-estimator and evidence-binding tests; no environment or actor forward."""
from copy import deepcopy

import numpy as np
import pytest

from src.gps.bidirectional.walker_learning import es_gradient, snapshot_identity, _contexts, _owned_trace


def test_antithetic_quadratic_gradient_has_known_scale_and_equal_context_weights():
    e = np.array([[1.], [-1.], [1.], [-1.]])
    theta, sigma = 2., .1
    plus = np.repeat(np.square(theta + sigma * e), 2, axis=1) + [0., 8.]
    minus = np.repeat(np.square(theta - sigma * e), 2, axis=1) + [0., 8.]
    result = es_gradient(e, plus, minus, sigma=sigma, cost_scale=2., beta=0.)
    np.testing.assert_allclose(result["gradient"], [2.], rtol=0, atol=1e-14)


def test_teacher_outcome_budget_changes_signal_without_any_action_labels():
    e = [[1.]]
    tight = es_gradient(e, [[2., 4.]], [[1., 3.]], sigma=1., cost_scale=1., budgets=[None, 2.])
    loose = es_gradient(e, [[2., 4.]], [[1., 3.]], sigma=1., cost_scale=1., budgets=[None, 10.])
    np.testing.assert_array_equal(tight["plus_objectives"], [[2., 8.]])
    np.testing.assert_array_equal(tight["minus_objectives"], [[1., 4.]])
    np.testing.assert_array_equal(tight["gradient"], [1.25])
    np.testing.assert_array_equal(loose["gradient"], [.5])


def test_equal_paired_outcomes_produce_zero_signal_and_frozen_coordinates_stay_zero():
    e = [[0., 1.], [0., -2.]]
    result = es_gradient(e, [[5.], [3.]], [[5.], [3.]], sigma=.02, cost_scale=2800.)
    np.testing.assert_array_equal(result["gradient"], [0., 0.])
    changed = es_gradient(e, [[6.], [4.]], [[5.], [3.]], sigma=.02, cost_scale=2800.)
    assert changed["gradient"][0] == 0. and changed["gradient"][1] != 0.


def test_malformed_estimator_data_rejected_without_rollouts():
    for kwargs in (dict(sigma=0., cost_scale=1.), dict(sigma=.02, cost_scale=float("nan"))):
        with pytest.raises(ValueError):
            es_gradient([[1.]], [[1.]], [[0.]], **kwargs)
    for plus, minus in (([[np.inf]], [[0.]]), ([[1., 2.]], [[0.]]), ([], [])):
        with pytest.raises(ValueError):
            es_gradient([[1.]], plus, minus, sigma=.02, cost_scale=1.)


def _roster():
    contexts = []
    for i in range(4):
        snapshot = dict(state=[float(i)], warmstart=[0.], elapsed=i, task_hash="task")
        contexts.append(dict(snapshot=snapshot, context_id=str(i), snapshot_id=snapshot_identity(snapshot),
                             source="ordinary" if i < 2 else "mppi_acquired"))
    budgets = [None, None] + [dict(cost=2., snapshot_id=contexts[i]["snapshot_id"], teacher_hash="teacher", query_ids=["teacher" + str(i)]) for i in range(2, 4)]
    return contexts, budgets


def test_teacher_budget_requires_exact_full_snapshot_and_no_ordinary_fabrication():
    contexts, budgets = _roster()
    assert len(_contexts(contexts, budgets)[2]) == 4
    wrong = deepcopy(budgets); wrong[2]["snapshot_id"] = "different"
    with pytest.raises(ValueError, match="same full snapshot"):
        _contexts(contexts, wrong)
    wrong = deepcopy(budgets); wrong[0] = dict(cost=0.)
    with pytest.raises(ValueError, match="fabricated"):
        _contexts(contexts, wrong)
    changed = deepcopy(contexts); changed[2]["snapshot"]["warmstart"] = [1.]
    with pytest.raises(ValueError, match="actual full snapshot"):
        _contexts(changed, budgets)


def test_owned_complete_return_cannot_be_relabeled_to_another_actor_or_context():
    context = dict(context_id="c", snapshot_id="s")
    trace = dict(total_cost=12., actor_hash="owner", context_id="c", snapshot_id="s", complete=True, query_ids=["q1"])
    assert _owned_trace(trace, "owner", context) == 12.
    for key, value in (("actor_hash", "other"), ("snapshot_id", "other"), ("complete", False), ("query_ids", ["q1", "q1"])):
        bad = dict(trace); bad[key] = value
        with pytest.raises(ValueError, match="this actor"):
            _owned_trace(bad, "owner", context)
