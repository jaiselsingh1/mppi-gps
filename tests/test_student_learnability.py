import numpy as np
import pytest

from scripts.diagnostics.student_learnability import features,normalization,default_config,planned_work,feature_benefit


def test_full_features_preserve_velocity_and_clipped_features_bind_native_slices():
    state=np.arange(19,dtype=float);state[10:19]=np.arange(-40,50,10)
    clipped=features(state,'clipped');full=features(state,'full')
    assert clipped.shape==full.shape==(17,)
    np.testing.assert_array_equal(clipped[:8],state[2:10])
    np.testing.assert_array_equal(full[8:],state[10:19])
    np.testing.assert_array_equal(clipped[8:],np.clip(state[10:19],-10,10))


def test_normalization_is_training_only_float64_statistics_with_same_floor():
    data=np.tile(np.arange(17,dtype=float),(4,1));norm=normalization(data)
    assert norm['mean'].dtype==norm['std'].dtype==np.float32
    np.testing.assert_array_equal(norm['mean'],np.arange(17,dtype=np.float32))
    np.testing.assert_array_equal(norm['std'],np.full(17,.05,dtype=np.float32))


@pytest.mark.parametrize('bad',[np.zeros(18),np.full(19,np.nan),np.zeros((2,18))])
def test_invalid_physical_features_reject(bad):
    with pytest.raises(ValueError):features(bad,'full')


def test_roster_budget_includes_parent_curves_and_conditional_confirmation():
    cfg=default_config('/comparison','/experience','/feedback');work=planned_work(cfg)
    assert work['requested_controls']==124000
    assert work['neural_rows']==5298864
    assert work['backward_rows']==5120000 and work['optimizer_steps']==20000
    assert all(work[k]<=cfg['budgets'][k] for k in cfg['budgets'])


def test_standing_survival_cannot_pass_feature_benefit():
    cfg=default_config('/comparison','/experience','/feedback')
    base=dict(successes=0,survived=0,mean_steps=200,mean_cost=4000)
    standing=dict(successes=0,survived=16,mean_steps=1000,mean_cost=3000)
    assert not feature_benefit(base,standing,cfg)['passed']


def test_feature_benefit_needs_competence_cost_and_nonregression():
    cfg=default_config('/comparison','/experience','/feedback')
    base=dict(successes=14,survived=16,mean_steps=1000,mean_cost=500)
    full=dict(successes=15,survived=16,mean_steps=1000,mean_cost=470)
    assert feature_benefit(base,full,cfg)['passed']
    full['mean_cost']=490
    assert not feature_benefit(base,full,cfg)['passed']
