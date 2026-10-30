"""Pure tests for admission, fixed budgets and complete-outcome decisions."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.diagnostics.guided_data_comparison import default_config, planned_work, task_gate, validate_episode


def score(cost=100., success=0, survived=0, steps=300.):
    return dict(episodes=1, mean_cost=cost, costs=[cost], successes=success, survived=survived, mean_steps=steps)


def test_incremental_gain_does_not_require_action_agreement_or_full_qualification():
    cfg = default_config('/saved/feedback', '/saved/experience')
    assert task_gate(score(), score(), score(cost=94., steps=320.), cfg['gate'])['passed']


@pytest.mark.parametrize('change', [dict(mean_cost=96.), dict(successes=0), dict(survived=0), dict(mean_steps=299.)])
def test_task_gate_requires_cost_and_every_nonregression_against_both_controls(change):
    cfg = default_config('/saved/feedback', '/saved/experience')
    base = score(success=1, survived=1)
    child = score(cost=90., success=1, survived=1)
    child.update(change)
    child['costs'] = [child['mean_cost']]
    if child['successes'] > child['survived']:
        child['successes'] = child['survived']
    assert not task_gate(base, base, child, cfg['gate'])['passed']


def test_budget_counts_all_curves_primary_and_conditional_confirmation():
    cfg = default_config('/saved/feedback', '/saved/experience')
    work = planned_work(cfg)
    assert work == dict(requested_controls=124000, neural_rows=1233344, backward_rows=1024000, optimizer_steps=4000)
    assert all(work[k] <= cfg['budgets'][k] for k in work)


def test_successful_teacher_needs_actual_preaction_alignment_and_full_snapshots():
    states = np.zeros((1001, 19))
    values = dict(states=states, observations=np.zeros((1000, 17)), actions=np.zeros((1000, 6)),
        integrations=np.zeros((1001, 91)), warmstarts=np.zeros((1001, 9)), costs=np.ones(1000),
        terminal_adjustment=np.array(0.), absorbing_cost=np.array(0.))
    event = dict(steps=1000, complete=True, success=True, survived=True, cost=1000.)
    validate_episode(values, event)
    corrupt = deepcopy(values); corrupt['observations'][20, 0] = 1.
    with pytest.raises(ValueError, match='pre-action'):
        validate_episode(corrupt, event)
    corrupt = deepcopy(values); corrupt['integrations'] = corrupt['integrations'][:-1]
    with pytest.raises(ValueError, match='snapshots'):
        validate_episode(corrupt, event)
