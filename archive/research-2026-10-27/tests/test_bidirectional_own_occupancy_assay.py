"""Pure budget/finite-difference/acceptance checks; no simulator or NN calls."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from src.gps.bidirectional.own_occupancy_assay import DEFAULT, work_bound, difference_targets, assess_full, assess_control, select_steps


def test_exact_full_schedule_and_conservative_neural_headroom():
    bound = work_bound()
    assert bound["model_controls"] == 10400
    assert bound["occupancy_controls"] == 400
    assert bound["qualification_controls"] == 12800
    assert bound["final_controls"] == 16000
    assert bound["total_controls"] == 39600 and bound["executed_controls"] == 29200
    assert bound["actor_rows"] == 79600 and bound["actor_row_cap_headroom"] == 400
    assert bound["adam_steps"] == 84 and bound["diagnostic_backward_passes"] == 3
    assert 400+9600+3*(1144+4*1144+20*200)+12800+16000 == 67960
    changed = deepcopy(DEFAULT); changed["probe_delta"] = .1
    with pytest.raises(ValueError): work_bound(changed)


def test_full_credit_distinguishes_delayed_cost_while_local_uses_same_first_four_rows():
    actions = np.array([[-.05], [.05]])
    raw = np.zeros((2, 25)); raw[:, :4] = [[.1]*4, [.2]*4]; raw[0, -1] = 20.
    result = difference_targets(actions, raw, 1., .6)
    assert result["slope_local"] > 0 and result["slope_full"] < 0
    assert result["slope_null"] == 0 and result["slope_sign_disagrees"]
    np.testing.assert_allclose(result["full_costs"], [20.4, .8])
    np.testing.assert_allclose(result["local_costs"], [.4, .8])


def test_postgate_action_invariance_yields_zero_full_local_and_null_slopes():
    raw = np.tile([.2, .2, 0., 0.], (2, 1))
    result = difference_targets([[-.05], [.05]], raw, .4, .4)
    assert result["slope_full"] == result["slope_local"] == result["slope_null"] == 0.
    with pytest.raises(ValueError): difference_targets([[.05], [.05]], raw, .4, .4)


def panels(deterministic=True):
    def rows(cost):
        return [SimpleNamespace(context_id=f"root:{i}", cost=cost, capture=True,
            complete=True, deterministic=deterministic) for i in range(10)]
    return {arm: rows(cost) for arm, cost in (("parent", 1.02), ("full", 1.), ("local", 1.01), ("null", 1.005),
        ("best_local", 1.004), ("best_null", 1.003), ("uniform_best_raw", 1.002))}


def test_full_must_beat_same_alpha_local_null_and_all_fresh_final_controls():
    mean, noisy = panels(), panels(False)
    assert assess_full(mean, noisy)["passed"]
    assert assess_full(mean, noisy, final=True)["passed"]
    for row in noisy["uniform_best_raw"]: row.cost = .999
    assert not assess_full(mean, noisy, final=True)["passed"]
    for row in mean["local"]: row.cost = .999
    assert not assess_full(mean, noisy)["passed"]
    for row in noisy["local"]: row.cost = 1.03
    assert not assess_control(mean, noisy, "local")["passed"]


def test_each_control_selects_its_own_qualified_alpha_before_final_and_can_fallback():
    rows = [dict(alpha=a, criteria={arm: dict(passed=False) for arm in ("full", "local", "null")},
        mean_costs=dict(full=1., local=1., null=1.)) for a in DEFAULT["alphas"]]
    assert select_steps(rows) == dict(full=None, local=None, null=None)
    rows[0]["criteria"]["full"]["passed"] = True
    rows[1]["criteria"]["local"]["passed"] = rows[3]["criteria"]["local"]["passed"] = True
    rows[3]["mean_costs"]["local"] = .99
    assert select_steps(rows) == dict(full=.0625, local=.5, null=None)
    with pytest.raises(ValueError): select_steps(rows[:-1])
