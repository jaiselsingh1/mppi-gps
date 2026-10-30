"""Synthetic geometry/return fixtures; no simulator calls or actor forwards."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pytest

from src.gps.bidirectional.geometry_return import (random_basis,prefix_rows,calibration_indices,fit_geometry,
    coordinate_bank,actor_from_geometry,score_candidate,select_candidate)
from src.gps.bidirectional.geometry_assay import DEFAULT,work_bound,choose_controls,assess


def features(scales=(2.,1.,.5,.25),means=(0.,0.,0.,0.)):
    draws=np.random.default_rng(817).normal(size=(192,4));draws-=draws.mean(axis=0)
    q,_=np.linalg.qr(draws);H=np.zeros((192,128));H[:,:4]=q*np.sqrt(192)*scales+means
    B=np.zeros((4,128));B[:,:4]=np.eye(4)
    return H,B


def test_basis_is_reproducible_orthonormal_and_does_not_borrow_global_rng():
    old=np.random.get_state();B=random_basis()
    np.testing.assert_array_equal(B,random_basis());np.testing.assert_allclose(B@B.T,np.eye(4),atol=1e-12,rtol=0)
    np.testing.assert_array_equal(old[1],np.random.get_state()[1])
    with pytest.raises(ValueError):random_basis(2)


def test_source_rows_exclude_gate_and_all_later_observations():
    states=np.column_stack((np.arange(26),np.linspace(.2,.8,26)));obs=np.arange(208).reshape(26,8)
    oo,ss,qq=prefix_rows(obs,states,[f'q{i}' for i in range(25)])
    assert oo.shape==(12,8) and ss[-1,0]==11 and qq[-1]=='q11'
    np.testing.assert_array_equal(oo,obs[:12]);assert not np.shares_memory(oo,obs)
    bad=states.copy();bad[11,0]=12
    with pytest.raises(ValueError):prefix_rows(obs,bad,list(range(25)))


@pytest.mark.parametrize('mode',['fixed','ordinary','raw','long'])
def test_equal_192row_support_and_ordinary_duplicate_control(mode):
    indices=calibration_indices(mode)
    np.testing.assert_array_equal(indices[:96],np.arange(96));assert len(indices)==192
    offset=0 if mode in ('fixed','ordinary') else 96 if mode=='raw' else 192
    np.testing.assert_array_equal(indices[96:],np.arange(offset,offset+96))


def test_uncapped_empirical_pre_squash_responses_are_centered_orthogonal_point1rms():
    H,B=features();g=fit_geometry(H,B,'long');response=H@g.weight_axes.T+g.bias_axes
    np.testing.assert_allclose(response.mean(0),np.zeros(4),atol=1e-14,rtol=0)
    np.testing.assert_allclose(response.T@response/192,.01*np.eye(4),atol=1e-14,rtol=0)
    assert g.diagnostics['active_axes']==[True]*4 and g.diagnostics['empirical_active_rank']==4
    for v in g.diagnostics['right_singular_vectors']:assert v[np.argmax(np.abs(v))]>=0
    with pytest.raises(ValueError):g.weight_axes.flags.writeable=True


def test_inactive_low_singular_axes_stay_zero_and_duplicates_are_not_replaced():
    H,B=features((1.,.1,.0005,0.));g=fit_geometry(H,B,'raw')
    assert g.diagnostics['active_axes']==[True,True,False,False]
    np.testing.assert_array_equal(g.weight_axes[2:],np.zeros((2,128)));np.testing.assert_array_equal(g.bias_axes[2:],np.zeros(2))
    assert coordinate_bank(np.zeros(6),0).shape==(13,6)


def test_bias_and_weight_caps_report_unachieved_target_without_rescaling():
    H,B=features(means=(100.,0.,0.,0.));g=fit_geometry(H,B,'long')
    assert g.diagnostics['limiting_caps'][0]=='bias_abs_.5' and abs(g.bias_axes[0])==pytest.approx(.5)
    assert g.diagnostics['achieved_pre_squash_rms'][0]==pytest.approx(.01)
    H,B=features((.05,.025,.01,.002));g=fit_geometry(H,B,'ordinary')
    np.testing.assert_allclose(np.linalg.norm(g.weight_axes,axis=1),np.ones(4),atol=1e-14)
    assert all(c=='weight_norm1' for c in g.diagnostics['limiting_caps'])
    np.testing.assert_allclose(g.diagnostics['achieved_pre_squash_rms'],[.05,.025,.01,.002],atol=1e-14)


def test_fixed_control_remains_exact_old_random_weights_independent_of_calibration():
    H,B=features();a=fit_geometry(H,B,'fixed');b=fit_geometry(H+100,B,'fixed')
    np.testing.assert_array_equal(a.weight_axes,.1*B);np.testing.assert_array_equal(a.weight_axes,b.weight_axes)
    np.testing.assert_array_equal(a.bias_axes,np.zeros(4));assert a.diagnostics['singular_vectors_used_for_axes'] is False


def test_absolute_actor_changes_only_head_and_private_rng_no_forward(monkeypatch):
    import torch
    from src.gps.bidirectional.policy import FamilyGaussianActor,policy_hash
    base=FamilyGaussianActor(seed=64);H,B=features();g=fit_geometry(H,B,'long');old=base.policy_state()
    def forbidden(*a,**k):raise AssertionError('Construction must never call the actor')
    monkeypatch.setattr(FamilyGaussianActor,'forward',forbidden)
    zero=actor_from_geometry(base,g,np.zeros(6));assert policy_hash(zero)==policy_hash(base)
    c=np.array([1.,1.,0.,0.,0.,-1.]);actor=actor_from_geometry(base,g,c)
    for name,value in actor.state_dict().items():
        if name not in ('fc_mean.weight','fc_mean.bias','log_std'):assert torch.equal(value,old['actor'][name])
    np.testing.assert_allclose(actor.fc_mean.bias.detach().numpy(),old['actor']['fc_mean.bias'].numpy()+.1+g.bias_axes[0],atol=1e-7)
    np.testing.assert_allclose(actor.fc_mean.weight.detach().numpy(),old['actor']['fc_mean.weight'].numpy()+g.weight_axes[0],atol=1e-7)
    assert torch.equal(actor._generator.get_state(),base._generator.get_state())
    with pytest.raises(ValueError):coordinate_bank(np.array([1.,0.,0.,0.,0.,0.]),0)


@dataclass(frozen=True)
class O:
    context_id:str
    cost:float
    capture:bool
    complete:bool
    actor_hash:str
    controller_hash:str
    deterministic:bool
    remaining_steps:int
    query_ids:tuple
    semantics:str='actor_suffix'
    task_hash:str='task'
    family_id:int=0


def panels(name='a',cost=2.,n=8):
    return {m:[O(f'ordinary:{i}',cost,True,True,name,name+m,det,25,
        tuple(f'{name}{m}{i}q{t}' for t in range(25))) for i in range(n)] for m,det in (('mean',True),('stochastic',False))}


def test_complete_ordinary_own_returns_only_and_nonregression_before_stable_argmin():
    base=score_candidate(panels(),'a');scores=[deepcopy(base) for _ in range(13)]
    scores[1]['ordinary']['mean']['mean_cost']=2.1;scores[1]['joint_normalized_cost']=0.
    scores[2]['ordinary']['mean']['mean_cost']=1.9;scores[2]['joint_normalized_cost']=.078
    decision=select_candidate(scores);assert decision['selected_index']==2 and not decision['feasible_flags'][1]
    assert decision['source_return_weight']==0.
    p=panels();p['mean'][0]=O('ordinary:0',2.,True,True,'foreign','foreignmean',True,25,tuple(f'x{t}' for t in range(25)))
    with pytest.raises(ValueError):score_candidate(p,'a')


def test_exact_caps_include_calibration_and_single_feature_forward():
    b=work_bound();assert b['calibration_controls']+b['training_controls']+b['qualification_controls']+b['final_controls']==76200
    assert b['actor_rows']==76488==DEFAULT['max_actor_rows'] and b['feature_forward_rows']==288
    assert b['training_actor_versions']==156 and b['search_iterations']==12 and b['geometry_svd_calls']==4
    assert b['model_controls']==b['adam_steps']==b['diagnostic_backward_passes']==0


def test_strong_control_and_aggregate_capture_definition_with_paired_losses_retained():
    costs=dict(parent=2.,fixed=1.8,ordinary=1.7,raw=1.75,long=1.72)
    m={name:panels(name,cost,16)['mean'] for name,cost in costs.items()}
    s={name:panels(name,cost,16)['stochastic'] for name,cost in costs.items()}
    assert choose_controls(m,s)['best_control']=='ordinary' and choose_controls(m,s)['selected_long'] is None
    m['long']=panels('long',1.6,16)['mean'];s['long']=panels('long',1.6,16)['stochastic']
    assert choose_controls(m,s)['selected_long']=='long'
    criterion=assess(m,s,'long',('parent','ordinary'))
    assert criterion['comparisons']['mean']['parent']['paired_capture_losses']==0
    assert 'Aggregate' in criterion['comparisons']['mean']['parent']['capture_nonregression_definition']
