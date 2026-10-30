"""Evidence and execution semantics rather than network-shape tests."""
import numpy as np
import pytest

from src.gps.independent_core import Task, Search, Memory
from src.gps.independent_learning import (
    fit_value, measured_returns, score_guidance, promotion,
)
from scripts.training.run_independent_search import dataset, episode_upper


def test_complete_returns_reject_incomplete_and_forced_predecision_labels():
    with pytest.raises(ValueError, match="Incomplete"):
        measured_returns([1., 2.], 3)
    np.testing.assert_array_equal(measured_returns([1., 2., 0.], 3), [3., 2., 0.])
    ep = dict(states=np.zeros((4, 2)), costs=np.array([1., 2., 3.]),
              remaining_start=3, returns=np.array([6., 5., 3.]), forced=True, root=4, id="branch")
    _, r, y, groups, ids = dataset([ep])
    np.testing.assert_array_equal(r, [2, 1])
    np.testing.assert_array_equal(y, [5., 3.])
    assert ids == [("branch", 1), ("branch", 2)]
    assert groups.tolist() == [4, 4]


def test_global_fit_interpolates_complete_costs_but_does_not_claim_unknown_support():
    states = np.column_stack((np.linspace(-.7, -.3, 24), np.zeros(24)))
    r = np.full(24, 100)
    # Return evidence for a deliberately simple test continuation.
    y = 100*(.5 + .1*states[:, 0])
    v = fit_value(states, r, y, np.repeat(np.arange(4), 6), total_steps=180,
                  version="declared-C", kind="mc", centers=12)
    pred = v.predict([[-.51, 0.], [.1, -.07], [.5, .07]], [100, 100, 100])
    assert abs(pred["values"][0]-44.9) < .1
    assert pred["supported"].tolist() == [True, False, True]
    assert pred["values"][2] == 0  # Exact known absorption, not model optimism.
    assert v.target_version == "declared-C"


def test_guidance_evaluates_one_action_and_keeps_independent_proposals():
    task = Task()
    decision = Search(task, K=8, H=4, memory=Memory([])).plan(np.array([-.5, 0.]), 10, 10)
    original = decision["bank_plans"].copy()

    class ZeroTail:
        def predict(self, states, remaining):
            assert remaining == 9  # NOT the H=4 hypothetical endpoint's six.
            expected, _, _ = task.transition(np.repeat([[-.5, 0.]], len(states), axis=0),
                                             np.unique(decision["bank_actions"], axis=0))
            np.testing.assert_array_equal(states, expected)
            return dict(values=np.zeros(len(states)), supported=np.ones(len(states), bool),
                        distances=np.zeros(len(states)))

    rank = score_guidance(task, decision, ZeroTail(), "mc")
    assert rank["scratch_controls"] == len(np.unique(decision["bank_actions"], axis=0))
    np.testing.assert_array_equal(decision["bank_plans"], original)


def test_unsupported_baseline_and_regressed_outcome_revoke_preference():
    task = Task()
    decision = Search(task, K=8, H=4).plan(np.array([-.5, 0.]), 10, 10)

    class Unknown:
        def predict(self, states, remaining):
            return dict(values=np.arange(len(states), dtype=float),
                        supported=np.zeros(len(states), bool), distances=np.ones(len(states)))

    rank = score_guidance(task, decision, Unknown(), "mc")
    assert rank["selected"] == decision["baseline_index"]
    assert not rank["changed"]
    assert not promotion([11]*4, [10]*4, [True]*4, [True]*4)["accepted"]
    assert promotion([7]*4, [10]*4, [True]*4, [True]*4)["accepted"]


def test_budget_bound_covers_mean_memory_scratch_and_execution():
    cfg = dict(steps=3, K=8)
    assert episode_upper(cfg, 2, True) == 17*(2+2+1)+3*18
