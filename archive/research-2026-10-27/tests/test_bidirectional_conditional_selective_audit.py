"""Source-independent selective criterion fixtures; no simulator/NN calls."""
from copy import deepcopy
import math
import numpy as np
import pytest
from src.gps.bidirectional.conditional_selective_audit import ROOTS,criteria,validate_selective_initial


def panels(gains):
    raw=[];guide=[]
    for i,g in enumerate(gains):
        common=dict(context_id=f'ordinary:{802000+i}',capture=True,complete=True,deterministic=True,
            family_id=0,remaining_steps=25,task_hash='task')
        raw.append(dict(common,cost=0.,query_ids=[f'raw:{i}:{t}' for t in range(25)]))
        guide.append(dict(common,cost=-float(g),query_ids=[f'guide:{i}:{t}' for t in range(25)]))
    return dict(RAW=raw,GUIDE=guide)


def test_selective_criterion_does_not_relabel_first75percent_failure():
    r=criteria(panels([.13]*5+[0.]*11),'GUIDE',('RAW',))
    assert r['passed'] and r['minimum_pair_fraction'] is None and r['minimum_paired_cost_gain']==-1e-9
    assert r['comparisons']['RAW']['positive_pairs']==5 and r['comparisons']['RAW']['tie_pairs']==11
    assert r['population_noninterference_guarantee'] is False


def test_selective_counterexample_mean_gain_and_allcapture_cannot_compensate_one_loss():
    r=criteria(panels([.087]*8+[0.]*7+[-.000903338119]),'GUIDE',('RAW',))
    c=r['comparisons']['RAW'];assert c['mean_cost_gain']>.01 and r['capture_rate']==1 and not r['passed']
    assert c['gain_pairs']==8 and c['tie_pairs']==7 and c['loss_pairs']==1 and not c['zero_observed_paired_loss']


@pytest.mark.parametrize('loss,passed',[
    (-1e-9,True),(math.nextafter(-1e-9,0.),True),(math.nextafter(-1e-9,-math.inf),False)])
def test_declared_loss_tolerance_is_inclusive_and_not_relaxed(loss,passed):
    r=criteria(panels([1.,loss]+[0.]*14),'GUIDE',('RAW',))
    assert r['passed'] is passed and r['minimum_paired_cost_gain']==-1e-9


@pytest.mark.parametrize('gain,passed',[(.16,True),(math.nextafter(.16,0.),False)])
def test_mean_gain_threshold_is_still_required(gain,passed):
    assert criteria(panels([gain]+[0.]*15),'GUIDE',('RAW',))['passed'] is passed


def test_actor_gate_is_owned_mean_gain_without_teacher_loss_requirement():
    p=panels([1.]*15+[-.5]);r=criteria(p,'GUIDE',('RAW',),teacher=False)
    assert r['passed'] and r['minimum_mean_gain']==.001 and r['teacher_pair_rule'] is None
    assert r['minimum_paired_cost_gain'] is None and r['comparisons']['RAW']['loss_pairs']==1


@pytest.mark.parametrize('bad',['query_alias','task','root','capture','short'])
def test_selective_validation_rejects_wrong_evidence_or_capture_loss(bad):
    p=panels([.1]*16)
    if bad=='query_alias':p['GUIDE'][0]['query_ids']=p['RAW'][0]['query_ids']
    elif bad=='task':p['GUIDE'][0]['task_hash']='another'
    elif bad=='root':p['GUIDE'][0]['context_id']='another'
    elif bad=='capture':
        p['GUIDE'][0]['capture']=False;assert not criteria(p,'GUIDE',('RAW',))['passed'];return
    else:p['GUIDE'][0]['remaining_steps']=24
    with pytest.raises(ValueError):criteria(p,'GUIDE',('RAW',))


def test_declared_fresh_namespace_does_not_overlap_prior_initial_or_final_roots():
    assert ROOTS['competence']==801000 and ROOTS['initial_qualification']==802000 and ROOTS['final']==831000
    for key,seed in ROOTS.items():assert seed-100000<800000
    assert len(set(ROOTS.values()))==len(ROOTS)


def test_initial_adapter_refuses_legacy_rule_before_reading_any_raw_data():
    cfg=dict(schema='bidirectional-ending-conditional-mean-feedback-v2-selective',roots=ROOTS,teacher_pair_rule='legacy')
    with pytest.raises(ValueError):validate_selective_initial(None,None,[],[],{},cfg,{},[],{}, {},'task')


def test_initial_adapter_refuses_claimed_future_feedback_before_query_work():
    cfg=dict(schema='bidirectional-ending-conditional-mean-feedback-v2-selective',roots=ROOTS,teacher_pair_rule='zero_observed_paired_loss')
    result=dict(teacher_pair_rule='zero_observed_paired_loss',selective_noninterference_variant=True,
        criterion_disclosure=dict(outcome_informed=True,motivating_run='2026-10-27_bidirectional_ending_conditional_001',
            motivating_result='failed_predeclared_75_percent_improvement_prevalence',original_v1_negative_unchanged=True,
            original_v1_rows_are_not_fresh_confirmation=True,population_noninterference_guarantee=False,efficiency_claim=False),
        stage='complete_two_round_conditional_mean_feedback',rounds=[{}],passed=True,full_feedback_established=True)
    with pytest.raises(ValueError):validate_selective_initial(None,None,[],[],result,cfg,{},[],{}, {},'task')
