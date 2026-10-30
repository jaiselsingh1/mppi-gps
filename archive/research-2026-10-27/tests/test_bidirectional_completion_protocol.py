"""Completion-pilot schedule and data contracts: zero physics/actor forwards."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional.completion_pilot import work_bound,teaching_auxiliary,_comparison
from src.gps.bidirectional.ending_pilot import COMPLETION,load_pilot_config,pilot_work_bound
from src.gps.bidirectional.evidence import Archive


def teacher(marker,length,*,successful=True):
    """Handwritten labels only; no simulated or neural evidence is implied."""
    observations=np.full((length+1,8),float(marker))
    observations[:,1]=np.arange(length+1)
    actions=np.full((length,1),marker/1000.)
    return dict(observations=observations,actions=actions,successful=successful)


def test_finite_completion_schedule_includes_suffixes_both_child_modes_and_end_only_panel():
    root=Path(__file__).resolve().parents[1]
    config=load_pilot_config(root/"configs/bidirectional_ending_pilot_v3.json")
    assert config==COMPLETION and config['feedback_rounds']==2 and config['completion_candidates']==5
    bounds=work_bound(config)
    assert bounds==pilot_work_bound(config)
    assert bounds['model_controls']==2319970
    assert bounds['executed_controls']==50716
    assert bounds['total_controls']==config['max_controls']==2370686
    assert bounds['actor_rows']==config['max_actor_rows']==841866
    assert bounds['adam_steps']==config['max_adam_steps']==2168
    assert bounds['diagnostic_backward_passes']==6
    # The earlier cap omitted the separate final panel. These 3*64 actual
    # deterministic actor episodes do not feed either of the two learning rounds.
    assert bounds['executed_controls']-3*64*25==45916


def test_completion_config_rejects_silent_candidate_or_budget_change(tmp_path):
    changed=deepcopy(COMPLETION);changed['completion_candidates']=6
    path=tmp_path/'unfrozen_candidate_revision.json';path.write_text(__import__('json').dumps(changed))
    with pytest.raises(ValueError,match='no automatic retuning'):
        load_pilot_config(path)


def test_auxiliary_common_half_identical_and_empty_new_arm_fallback_explicit():
    common=[teacher(1,2)]
    raw,raw_indices=teaching_auxiliary(common,[teacher(99,7,successful=False)],200)
    guided,guided_indices=teaching_auxiliary(common,[teacher(2,3)],200)
    np.testing.assert_array_equal(raw.observations[:100],guided.observations[:100])
    np.testing.assert_array_equal(raw.actions[:100],guided.actions[:100])
    np.testing.assert_array_equal(raw_indices['common'],guided_indices['common'])
    assert raw_indices['specific_is_shared_fallback'] is True
    assert guided_indices['specific_is_shared_fallback'] is False
    assert raw.observations.shape==guided.observations.shape==(200,8)
    assert raw.actions.shape==guided.actions.shape==(200,1)
    assert set(raw_indices['common'])=={0,1}  # Declared short-stratum resampling.
    assert set(guided_indices['specific'])=={0,1,2}
    np.testing.assert_array_equal(raw.actions[100:],raw.actions[:100])
    assert np.all(guided.actions[100:]==.002)
    assert np.all(raw.families==0) and raw.source==guided.source=='executed_teacher'


def test_auxiliary_new_success_count_cannot_change_shared_exposure_or_include_failures(tmp_path):
    shared=[teacher(1,3),teacher(2,2)]
    raw,raw_indices=teaching_auxiliary(shared,[teacher(3,2)],200)
    guided,guided_indices=teaching_auxiliary(shared,
        [teacher(9,8,successful=False),teacher(4,3),teacher(5,4)],200)
    np.testing.assert_array_equal(raw.observations[:100],guided.observations[:100])
    np.testing.assert_array_equal(raw.actions[:100],guided.actions[:100])
    assert np.all(raw.actions[100:]==.003)
    assert set(guided.actions[100:,0])=={.004,.005}
    with Archive(tmp_path/'arrays_only') as archive:
        reference=archive.save_npz('feedback/stratified.npz',raw_actions=raw.actions,guided_actions=guided.actions,
            **{f'{arm}_{key}':value for arm,details in (('raw',raw_indices),('guided',guided_indices))
                for key,value in details.items()})
        with np.load(archive.path/reference['file'],allow_pickle=False) as saved:
            assert not any(saved[key].dtype.hasobject for key in saved.files)
            assert not bool(saved['raw_specific_is_shared_fallback'])
        assert archive.totals()['requested_controls']==0


def test_auxiliary_rejects_missing_common_success_and_odd_strata():
    with pytest.raises(ValueError,match='even rows'):
        teaching_auxiliary([teacher(1,2)],[],199)
    with pytest.raises(ValueError,match='No successful actual teacher'):
        teaching_auxiliary([teacher(1,2,successful=False)],[teacher(2,2)],200)


def outcomes(costs,*,captures=None):
    captures=[True]*len(costs) if captures is None else captures
    return [SimpleNamespace(context_id=f'ordinary:{i}',cost=float(cost),capture=bool(capture))
        for i,(cost,capture) in enumerate(zip(costs,captures))]


def test_complete_cost_gain_does_not_override_capture_regression_or_pairing():
    base=outcomes([10.]*8)
    proposed=outcomes([9.]*8,captures=[True]*7+[False])
    comparison=_comparison(base,proposed,.01)
    assert comparison['mean_cost_gain']==1. and comparison['positive_pairs']==8
    assert comparison['capture_nonregression'] is False and comparison['improved'] is False
    proposed[-1].context_id='different-root'
    with pytest.raises(ValueError,match='contexts are not paired'):
        _comparison(base,proposed,.01)


def test_strict_teacher_gain_counts_each_pair_and_exposes_outlier_only_mean_gain():
    comparison=_comparison(outcomes([10.]*8),outcomes([9.,10.,10.,10.,10.,10.,10.,10.]),.01)
    assert comparison['improved'] and comparison['mean_cost_gain']==.125
    assert comparison['positive_pairs']==1 and comparison['total_pairs']==8
    assert comparison['positive_pairs']/comparison['total_pairs']<COMPLETION['minimum_positive_fraction']
    small=_comparison(outcomes([10.]*8),outcomes([9.9995]*8),.001)
    assert not small['improved'] and small['positive_pairs']==0
