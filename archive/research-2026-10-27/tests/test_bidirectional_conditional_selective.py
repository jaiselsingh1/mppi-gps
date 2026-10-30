"""Synthetic selective-teacher gate fixtures; no actor or dynamics calls.

These tests are authored for a separately accounted future test run. They do not
invoke run_assay, an actual controller, a neural network, or a simulator.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.gps.bidirectional import conditional_feedback as module
from src.gps.bidirectional.conditional_feedback import (
    DEFAULT, LEGACY_TEACHER_PAIR_RULE, SELECTIVE_DEFAULT,
    SELECTIVE_TEACHER_PAIR_RULE, compare_panels, load_config, work_bound,
)


@dataclass
class Row:
    context_id: str
    cost: float
    capture: bool = True
    complete: bool = True
    deterministic: bool = True
    family_id: int = 0
    remaining_steps: int = 25
    query_ids: tuple = ()
    task_hash: str = 'synthetic-task'
    controller_hash: str = 'synthetic-controller'
    actor_hash: str | None = 'synthetic-actor'
    semantics: str = 'executed_replanning'


def rows(tag, costs, captures=None, seeds=None):
    costs = list(costs)
    captures = [True] * len(costs) if captures is None else list(captures)
    seeds = list(range(len(costs))) if seeds is None else list(seeds)
    return [Row(f'ordinary:{seed}', float(cost), capture,
                query_ids=tuple(f'{tag}:{i}:{t}' for t in range(25)),
                controller_hash=f'controller:{tag}')
            for i, (seed, cost, capture) in enumerate(zip(seeds, costs, captures))]


def panels(gains, *, base_captures=None, guide_captures=None):
    # Zero baseline avoids subtractive rounding at the exact gain thresholds.
    gains = list(gains)
    return dict(RAW=rows('RAW', [0.] * len(gains), base_captures),
                GUIDE=rows('GUIDE', [-g for g in gains], guide_captures))


def selective(panel, references=('RAW',), *, teacher=True):
    return compare_panels(panel, 'GUIDE', references, teacher=teacher,
                          teacher_pair_rule=SELECTIVE_TEACHER_PAIR_RULE)


def test_legacy_recipe_and_new_variant_are_exact_separate_configurations(tmp_path):
    root = Path(__file__).resolve().parents[1]
    legacy_path = root / 'configs/bidirectional_ending_conditional_mean_v0.json'
    selective_path = root / 'configs/bidirectional_ending_conditional_mean_selective_v1.json'
    assert load_config() == load_config(legacy_path) == DEFAULT
    assert 'teacher_pair_rule' not in DEFAULT
    expected = deepcopy(DEFAULT)
    expected.update(schema='bidirectional-ending-conditional-mean-feedback-v2-selective',
                    roots={k: v + 100000 for k, v in DEFAULT['roots'].items()},
                    teacher_pair_rule='zero_observed_paired_loss')
    assert SELECTIVE_DEFAULT == load_config(selective_path) == expected
    assert SELECTIVE_DEFAULT['roots'] is not DEFAULT['roots']
    assert SELECTIVE_DEFAULT['task'] is not DEFAULT['task']
    assert SELECTIVE_DEFAULT['search'] is not DEFAULT['search']
    assert set(DEFAULT['roots'].values()).isdisjoint(SELECTIVE_DEFAULT['roots'].values())
    copied = load_config(selective_path)
    copied['roots']['final'] += 1
    copied['search']['H'] = 3
    assert SELECTIVE_DEFAULT['roots']['final'] == 831000
    assert SELECTIVE_DEFAULT['search']['H'] == 4


def test_selective_rule_preserves_exact_full_schedule_and_all_caps():
    bound = work_bound()
    assert bound == work_bound(DEFAULT) == work_bound(SELECTIVE_DEFAULT)
    assert (bound['model_controls'], bound['executed_controls'],
            bound['total_controls'], bound['actor_rows']) == (4675680, 49424, 4725104, 600400)
    assert bound['training_policy_versions'] == 132
    assert bound['feature_forward_rows'] == 576
    assert bound['adam_steps'] == bound['diagnostic_backward_passes'] == 0
    assert bound['forced_revocation_new_controls'] == 0


@pytest.mark.parametrize('mutation', [
    'v1_rule', 'v1_schema_only', 'v1_roots_only', 'v2_old_roots',
    'v2_old_rule', 'v2_missing_rule', 'cap', 'stochastic', 'horizon',
    'extra', 'capture_floor',
])
def test_only_two_exact_recipes_are_allowed_no_hybrids(tmp_path, mutation):
    c = deepcopy(DEFAULT if mutation.startswith('v1_') else SELECTIVE_DEFAULT)
    if mutation == 'v1_rule':
        c['teacher_pair_rule'] = SELECTIVE_TEACHER_PAIR_RULE
    elif mutation == 'v1_schema_only':
        c['schema'] = SELECTIVE_DEFAULT['schema']
    elif mutation == 'v1_roots_only':
        c['roots'] = deepcopy(SELECTIVE_DEFAULT['roots'])
    elif mutation == 'v2_old_roots':
        c['roots'] = deepcopy(DEFAULT['roots'])
    elif mutation == 'v2_old_rule':
        c['teacher_pair_rule'] = LEGACY_TEACHER_PAIR_RULE
    elif mutation == 'v2_missing_rule':
        del c['teacher_pair_rule']
    elif mutation == 'cap':
        c['max_controls'] += 1
    elif mutation == 'stochastic':
        c['controller_mode'] = 'stochastic'
    elif mutation == 'horizon':
        c['search']['H'] = 3
    elif mutation == 'extra':
        c['undeclared_retry'] = True
    else:
        c['capture_min'] = .8
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_config(path)
    with pytest.raises(ValueError):
        work_bound(c)


def test_five_gains_eleven_ties_remain_v1_negative_and_selective_positive():
    p = panels([.04] * 5 + [0.] * 11)
    legacy = compare_panels(p, 'GUIDE', ('RAW',), teacher=True)
    explicit_legacy = compare_panels(p, 'GUIDE', ('RAW',), teacher=True,
                                    teacher_pair_rule=LEGACY_TEACHER_PAIR_RULE)
    chosen = selective(p)
    assert legacy == explicit_legacy
    assert not legacy['passed'] and chosen['passed']
    assert legacy['minimum_pair_fraction'] == .75
    assert chosen['minimum_pair_fraction'] is None
    assert chosen['minimum_paired_cost_gain'] == -1e-9
    assert chosen['teacher_pair_rule'] == SELECTIVE_TEACHER_PAIR_RULE
    assert chosen['population_noninterference_guarantee'] is False
    for report in (legacy, chosen):
        comparison = report['comparisons']['RAW']
        assert comparison['mean_cost_gain'] == pytest.approx(.0125)
        assert comparison['positive_pairs'] == comparison['gain_pairs'] == 5
        assert comparison['tie_pairs'] == 11 and comparison['loss_pairs'] == 0
        assert comparison['improvement_prevalence'] == 5 / 16
        assert comparison['threshold_improvement_prevalence'] == 5 / 16
        assert comparison['zero_observed_paired_loss'] is True


def test_compensated_half_unit_loss_cannot_pass_selective_even_with_large_mean():
    p = panels([1.] * 15 + [-.5])
    assert compare_panels(p, 'GUIDE', ('RAW',), teacher=True)['passed']
    result = selective(p)
    assert not result['passed']
    comparison = result['comparisons']['RAW']
    assert comparison['mean_cost_gain'] > .01
    assert comparison['loss_pairs'] == 1
    assert comparison['zero_observed_paired_loss'] is False


@pytest.mark.parametrize(('small_gain', 'passed', 'losses'), [
    (-1e-9, True, 0),
    (math.nextafter(-1e-9, 0.), True, 0),
    (math.nextafter(-1e-9, -math.inf), False, 1),
])
def test_selective_loss_tolerance_is_inclusive_and_not_mean_compensated(small_gain, passed, losses):
    result = selective(panels([1., small_gain] + [0.] * 14))
    assert result['passed'] is passed
    c = result['comparisons']['RAW']
    assert c['loss_pairs'] == losses
    assert c['gain_pairs'] + c['tie_pairs'] + c['loss_pairs'] == 16
    assert c['paired_cost_tolerance'] == 1e-9


@pytest.mark.parametrize(('total_gain', 'passed'), [
    (.16, True), (math.nextafter(.16, 0.), False),
])
def test_selective_mean_gain_still_requires_point01(total_gain, passed):
    result = selective(panels([total_gain] + [0.] * 15))
    assert result['minimum_mean_gain'] == .01
    assert result['passed'] is passed


@pytest.mark.parametrize(('before', 'after', 'passed'), [(16, 15, False), (14, 14, False), (14, 15, True)])
def test_selective_capture_nonregression_and_point9_floor_are_independent(before, after, passed):
    result = selective(panels([.02] * 16,
                              base_captures=[i < before for i in range(16)],
                              guide_captures=[i < after for i in range(16)]))
    assert result['passed'] is passed
    assert result['comparisons']['RAW']['capture_nonregression'] is (after >= before)
    assert result['capture_rate'] == after / 16


def test_capture_rule_remains_aggregate_with_paired_losses_and_gains_disclosed():
    before = [False] + [True] * 15
    after = [True, False] + [True] * 14
    result = selective(panels([.02] * 16, base_captures=before, guide_captures=after))
    assert result['passed']
    c = result['comparisons']['RAW']
    assert c['paired_capture_losses'] == c['paired_capture_gains'] == 1
    assert c['capture_nonregression']


def test_each_raw_actor_and_oldguide_reference_must_independently_pass():
    p = panels([.04] * 5 + [0.] * 11)
    p['actor'] = rows('actor', [0.] * 16)
    p['OLDGUIDE'] = rows('OLDGUIDE', [1.] * 16)
    assert selective(p, ('RAW', 'actor', 'OLDGUIDE'))['passed']
    # This comparison has a large positive mean and one individual loss.
    p['OLDGUIDE'][0].cost = -1.
    result = selective(p, ('RAW', 'actor', 'OLDGUIDE'))
    assert result['comparisons']['RAW']['passed']
    assert result['comparisons']['actor']['passed']
    assert not result['comparisons']['OLDGUIDE']['passed']
    assert not result['passed']


@pytest.mark.parametrize('rule', [LEGACY_TEACHER_PAIR_RULE, SELECTIVE_TEACHER_PAIR_RULE])
def test_actor_gate_keeps_point001_mean_and_allows_paired_cost_loss(rule):
    result = compare_panels(panels([.1, -.01] + [0.] * 14), 'GUIDE', ('RAW',),
                            teacher=False, teacher_pair_rule=rule)
    assert result['passed']
    assert result['minimum_mean_gain'] == .001
    assert result['minimum_pair_fraction'] is None
    assert result['minimum_paired_cost_gain'] is None
    assert result['teacher_pair_rule'] is None
    assert result['comparisons']['RAW']['loss_pairs'] == 1


@pytest.mark.parametrize('rule', ['unknown', None, True])
def test_undeclared_pair_rule_is_rejected(rule):
    with pytest.raises(ValueError):
        compare_panels(panels([.02] * 16), 'GUIDE', ('RAW',), teacher=True,
                       teacher_pair_rule=rule)


@pytest.mark.parametrize(('field', 'value'), [
    ('complete', False), ('deterministic', False), ('family_id', 1),
    ('remaining_steps', 24), ('cost', math.nan), ('task_hash', 'different-task'),
])
def test_selective_preserves_complete_mean_task_ownership_guards(field, value):
    p = panels([.02] * 16)
    setattr(p['GUIDE'][0], field, value)
    with pytest.raises(ValueError):
        selective(p)


@pytest.mark.parametrize('corruption', ['unpaired', 'duplicate', 'shared_query'])
def test_selective_rejects_unpaired_duplicate_or_reused_actual_evidence(corruption):
    p = panels([.02] * 16)
    if corruption == 'unpaired':
        p['GUIDE'][0].context_id = 'foreign-root'
    elif corruption == 'duplicate':
        for name in ('RAW', 'GUIDE'):
            p[name][1].context_id = p[name][0].context_id
    else:
        p['GUIDE'][0].query_ids = p['RAW'][0].query_ids
    with pytest.raises(ValueError):
        selective(p)


def test_finite_cost_inputs_cannot_hide_overflowed_paired_gain():
    p = panels([.02] * 16)
    p['RAW'][0].cost = 1e308
    p['GUIDE'][0].cost = -1e308
    with pytest.raises(ValueError, match='finite'):
        selective(p)


@dataclass
class GateRecord:
    record_id: str = 'synthetic-record'
    state: str = 'eligible'
    lambda_weight: float = 0.


class GateManager:
    def __init__(self):
        self.records = [GateRecord()]
        self.promotions = 0
        self.revocations = []

    def promote(self, record_id, weight, baseline, guided, **kwargs):
        assert record_id == 'synthetic-record' and weight == 1.
        self.promotions += 1
        self.records = [GateRecord(state='promoted', lambda_weight=1.)]
        return self.records[0]

    def revoke(self, record_id, reason):
        self.revocations.append(reason)
        self.records = [GateRecord(state='revoked')]


class FakeArchive:
    def __init__(self):
        self.rows = []

    def append(self, kind, **fields):
        identifier = f'e{len(self.rows):04d}'
        self.rows.append(dict(id=identifier, kind=kind, **fields))
        return identifier


@pytest.mark.parametrize(('rule', 'confirmation_loss', 'expected_phase'), [
    (LEGACY_TEACHER_PAIR_RULE, False, 'qualification'),
    (SELECTIVE_TEACHER_PAIR_RULE, False, 'complete'),
    (SELECTIVE_TEACHER_PAIR_RULE, True, 'confirmation'),
])
def test_controller_threads_rule_through_trial_promoted_confirmation_and_oldguide(
        monkeypatch, rule, confirmation_loss, expected_phase):
    from src.gps.bidirectional.trust import CandidateTrustManager

    actor, old_actor, eligible, old_eligible = object(), object(), object(), object()
    manager, old_manager = GateManager(), GateManager()
    monkeypatch.setattr(CandidateTrustManager, 'from_eligible',
                        classmethod(lambda cls, packet, weight:
                                    manager if packet is eligible else old_manager))
    archive = FakeArchive()
    executor = SimpleNamespace(archive=archive)
    seen = []

    def search(executor, current_actor, actual_manager, memory, seeds, tag, *, trial=False):
        phase = 'confirmation' if 'confirmation' in tag else 'qualification'
        costs = [0.] * 16
        if current_actor is actor:
            assert actual_manager is manager
            assert trial is (phase == 'qualification')
            assert manager.records[0].state == ('eligible' if trial else 'promoted')
            costs = [-.04] * 5 + [0.] * 11
            if phase == 'confirmation' and confirmation_loss:
                costs[-1] = .5
        if current_actor is old_actor:
            assert actual_manager is old_manager and trial is True
        seen.append((phase, current_actor, trial, tuple(seeds)))
        return rows(tag, costs, seeds=seeds), []

    def mean(executor, current_actor, seeds, tag):
        return rows(tag, [0.] * 16, seeds=seeds), []

    monkeypatch.setattr(module, '_search_panel', search)
    monkeypatch.setattr(module, '_mean_panel', mean)
    result, returned, _ = module.controller_gate(
        executor, actor, eligible, SimpleNamespace(identity='synthetic-memory'),
        'selective_fixture', 802000, 803000, old_actor=old_actor,
        old_eligible=old_eligible, teacher_pair_rule=rule)
    assert returned is manager and result['phase'] == expected_phase
    assert result['passed'] is (expected_phase == 'complete')
    assert result['qualification']['teacher_pair_rule'] == rule
    assert set(result['qualification']['comparisons']) == {'RAW', 'actor', 'OLDGUIDE'}
    assert manager.promotions == (0 if expected_phase == 'qualification' else 1)
    assert old_manager.promotions == 0
    if expected_phase == 'qualification':
        assert all(phase == 'qualification' for phase, _, _, _ in seen)
        assert 'confirmation' not in result
    else:
        assert result['confirmation']['teacher_pair_rule'] == rule
        assert set(result['confirmation']['comparisons']) == {'RAW', 'actor', 'OLDGUIDE'}
        assert archive.rows[-1]['second_promotion'] is False
        assert {seeds for phase, _, _, seeds in seen if phase == 'qualification'} == {tuple(range(802000, 802016))}
        assert {seeds for phase, _, _, seeds in seen if phase == 'confirmation'} == {tuple(range(803000, 803016))}
    assert bool(manager.revocations) is (expected_phase != 'complete')
