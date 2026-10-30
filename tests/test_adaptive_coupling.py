"""Outcome gates, finite budgets and real manipulation ownership checks."""
from copy import deepcopy
from types import SimpleNamespace
import json

import h5py
import numpy as np
import pytest

from scripts.diagnostics.adaptive_coupling import (
    admission, changed_trajectory, check_teacher_decisions, default_config,
    paired_resets, planned_work, task_gate, export, seed_audit, check_environment,
)


def config():
    return default_config('/comparison', dict(task={}, mppi=dict(K=256, H=48, initial_iterations=2)))


def score(cost=100., successes=0, survived=0, steps=200., episodes=16):
    return dict(episodes=episodes, mean_cost=cost, costs=[cost]*episodes,
                successes=successes, survived=survived, mean_steps=steps, steps=[steps]*episodes)


def beneficial():
    return dict(parent=score(cost=120, survived=1), independent=score(cost=110, survived=1),
                fixed_high=score(cost=115, survived=1), backoff=score(cost=100, successes=2, survived=3, steps=300))


def test_complete_fixed_schedule_within_budgets_including_confirmation_and_gaps():
    cfg = config(); work = planned_work(cfg)
    assert work['model_controls'] == 74569728
    assert work['live_controls'] == 142000
    assert work['requested_controls'] == 74711728
    assert work['optimizer_steps'] == 6000
    assert work['backward_rows'] == 1536000
    assert work['full_mse_rows'] == 97152
    assert work['neural_rows'] == 51311152
    assert all(work[name] <= cap for name, cap in cfg['budgets'].items())


def test_reserved_second_round_avoids_student_953_seeds():
    cfg = config()
    assert cfg['reserved_second_round_roots']['acquisition'] == 954000000
    assert default_config('/comparison', dict(task={}, mppi={}), round_index=2)['roots']['primary'] == 954200000


def test_worker_rejects_changed_platform_before_any_query():
    cfg = config(); check_environment(cfg['execution_platform'])
    wrong = deepcopy(cfg['execution_platform']); wrong['machine'] = 'another-machine'
    with pytest.raises(ValueError, match='prelaunch seal'):
        check_environment(wrong)


def test_teacher_admission_uses_full_task_not_agreement_and_is_aggregate():
    scores = {arm: score(cost=100, successes=2, survived=2, steps=1000, episodes=2)
              for arm in ('independent', 'fixed_high', 'backoff')}
    scores['backoff'].update(mean_cost=104, action_disagreement=1000)
    assert admission(scores)['passed']
    scores['backoff']['successes'] = 1
    assert not admission(scores)['passed']
    scores['backoff']['successes'] = 2; scores['fixed_high']['mean_cost'] = 106
    assert not admission(scores)['passed']


def test_standing_or_cost_advantage_alone_cannot_pass():
    scores = beneficial(); rule = config()['child_guard']
    assert task_gate(scores, rule)['passed']
    scores['backoff']['successes'] = 0
    assert not task_gate(scores, rule)['passed']


@pytest.mark.parametrize('change', [dict(survived=0), dict(mean_steps=100),
    dict(successes=0), dict(mean_cost=107)])
def test_child_success_cost_and_stability_requirements_are_joint(change):
    scores = beneficial(); scores['backoff'].update(change)
    assert not task_gate(scores, config()['child_guard'])['passed']


def test_first_divergence_preserves_exact_and_threshold_differences():
    actions = np.zeros((5, 6)); other = actions.copy(); other[1, 0] = 1e-9; other[3, 2] = .1
    make = lambda a: SimpleNamespace(steps=5, actions=a, observations=np.zeros((5, 17)))
    result = changed_trajectory(make(actions), make(other))
    assert result['first_byte_differing_action_control'] == 1
    assert result['first_changed_control'] == 3 and result['changed_action_rows'] == 1


def decision(rho=.1):
    return dict(actor_hash='new', alpha=rho, enabled=rho > 0,
                influence_used=rho > 0, numerical_revocation=False,
                raw_diagnostics=dict(coupling_active=False, cost_track_mean=0.))


def test_actual_acquisition_requires_updated_actor_and_declared_weight():
    good = decision(); make = lambda d: SimpleNamespace(infos=[dict(controller=d)])
    check_teacher_decisions(make(good), .1, 'new')
    for change in (dict(actor_hash='old'), dict(alpha=.2), dict(enabled=False), dict(numerical_revocation=True)):
        value = deepcopy(good); value.update(change)
        with pytest.raises(ValueError):
            check_teacher_decisions(make(value), .1, 'new')
    check_teacher_decisions(make(decision(0)), 0., 'new')
    bad = decision(0); bad['influence_used'] = True
    with pytest.raises(ValueError):
        check_teacher_decisions(make(bad), 0., 'new')
    bad = decision(); bad['raw_diagnostics']['coupling_active'] = True
    with pytest.raises(ValueError, match='Private MPPI'):
        check_teacher_decisions(make(bad), .1, 'new')


def test_full_reset_pairing_checks_integration_not_only_seed():
    def episode():
        return SimpleNamespace(seed=1, states=np.zeros((2, 19)), warmstarts=np.zeros((2, 9)),
            integrations=np.zeros((2, 91)), observations=np.zeros((1, 17)))
    first, second = episode(), episode()
    paired_resets(dict(a=[first], b=[second]))
    second.integrations[0, 80] = 1
    with pytest.raises(ValueError):
        paired_resets(dict(a=[first], b=[second]))


def test_seed_audit_blocks_both_current_and_reserved_collisions(tmp_path):
    cfg = config(); old = tmp_path / 'old' / 'evidence'; old.mkdir(parents=True)
    ledger = old / 'ledger.jsonl'
    ledger.write_text(json.dumps(dict(event='reset_requested', seed=951000000)) + '\n')
    assert seed_audit(tmp_path, cfg)['conflicts'] == []
    ledger.write_text(json.dumps(dict(event='reset_requested', seed=954200000)) + '\n')
    with pytest.raises(ValueError, match='collision'):
        seed_audit(tmp_path, cfg)
    ledger.write_text('{}\n')
    (old.parent / 'config.json').write_text(json.dumps(dict(roots=dict(planner=952100000))))
    with pytest.raises(ValueError, match='collision'):
        seed_audit(tmp_path, cfg)


def test_saved_export_preserves_complete_and_first_divergent_bank_groups(tmp_path):
    out = tmp_path / 'run'; evidence = out / 'evidence'; evidence.mkdir(parents=True)
    refs = ['arrays/a' + str(i) for i in range(7)]
    with h5py.File(evidence / 'raw.h5', 'x') as stream:
        for i, name in enumerate(refs):
            stream.create_group(name).create_dataset('actual', data=np.array([i]))
    rows = [dict(event='episode_complete', raw=dict(group=refs[0])),
            dict(event='decision', episode='h4_r1_backoff_00', step=3, evidence=dict(group=refs[1])),
            dict(event='decision', episode='h4_r1_fixed_high_00', step=3, evidence=dict(group=refs[2])),
            dict(event='decision', episode='h4_r1_backoff_00', step=0, evidence=dict(group=refs[3]))]
    (evidence / 'ledger.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = dict(round=1, datasets=dict(backoff=dict(group=refs[4])),
        matching=dict(sampled_indices=dict(group=refs[5])),
        manipulation=dict(backoff_vs_fixed_high=[dict(first_changed_control=3, first_byte_differing_action_control=3)]))
    (out / 'result.json').write_text(json.dumps(result))
    destination = tmp_path / 'export'
    export(out, destination)
    locator = json.loads((destination / 'remote_evidence.json').read_text())
    assert locator['compact']['groups'] == refs[:6]
    assert locator['controls_added'] == locator['neural_rows_added'] == 0
    with h5py.File(destination / 'complete_executed_evidence.h5', 'r') as stream:
        assert stream[refs[2] + '/actual'][0] == 2
    with pytest.raises(FileExistsError):
        export(out, destination)
