"""Synthetic neural/arithmetic tests only: zero simulator controls or imports."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from src.gps.bidirectional.policy import FamilyGaussianActor,policy_hash,controller_hash
from src.gps.bidirectional.learning import (AuxiliaryBatch,BehaviorEpisode,LearningConfig,
    PolicyLearner,build_mc_batch,clipped_cost_loss,monte_carlo_cost_to_go)


@pytest.fixture(scope='module')
def work():
    torch.set_num_threads(1)
    counts={}
    yield counts
    print('DEVELOPMENT_SYNTHETIC_FORWARD_WORK='+json.dumps(counts,sort_keys=True))


@pytest.fixture(autouse=True)
def counted_forwards(monkeypatch,request,work):
    # Engine imports would fail; these tests need no environment construction.
    monkeypatch.setitem(__import__('sys').modules,'mujoco',None)
    original=FamilyGaussianActor.forward
    record=work.setdefault(request.node.name,dict(calls=0,rows=0,simulator_controls=0,diagnostic_backwards=0,gradient_backwards=0,optimizer_steps=0))
    def forward(self,observations,families=0):
        shape=np.shape(observations)
        record['calls']+=1;record['rows']+=int(np.prod(shape[:-1])) if len(shape)>1 else 1
        return original(self,observations,families)
    monkeypatch.setattr(FamilyGaussianActor,'forward',forward)
    original_grad=torch.autograd.grad;original_backward=torch.autograd.backward;original_step=torch.optim.Adam.step
    def grad(*args,**kwargs):
        record['diagnostic_backwards']+=1;return original_grad(*args,**kwargs)
    def step(*args,**kwargs):
        record['optimizer_steps']+=1;return original_step(*args,**kwargs)
    def backward(*args,**kwargs):
        record['gradient_backwards']+=1;return original_backward(*args,**kwargs)
    monkeypatch.setattr(torch.autograd,'grad',grad)
    monkeypatch.setattr(torch.optim.Adam,'step',step)
    monkeypatch.setattr(torch.autograd,'backward',backward)


def episode(actor, costs, *, context='c',family=0,ref='q',latent=None,remaining=None):
    n=len(costs);obs=np.zeros((n,actor.obs_dim),dtype=np.float32)
    obs[:,0]=1;obs[:,-2]=np.arange(n,0,-1)/max(n,1)
    with torch.no_grad():
        if latent is None:
            sample=actor.sample(obs,family)
            latent=sample['pre_tanh'].numpy();actions=sample['actions'].numpy();logp=sample['log_prob'].numpy()
        else:
            latent=np.full((n,actor.act_dim),latent,dtype=np.float32)
            logp,_,_=actor.evaluate_actions(obs,family,latent);logp=logp.numpy();actions=np.tanh(latent)
    return BehaviorEpisode(obs,actions,latent,logp,np.asarray(costs,dtype=float),context,n if remaining is None else remaining,
        policy_hash(actor),controller_hash(actor,deterministic=False,family=family),family=family,query_ref=ref)


def test_modes_bounds_log_probability_and_private_rng():
    before=torch.get_rng_state().clone()
    actor=FamilyGaussianActor(action_low=(-2.,),action_high=(3.,),seed=7)
    assert torch.equal(before,torch.get_rng_state())
    obs=np.zeros((4,8),dtype=np.float32)
    sample=actor.sample(obs,np.array([0,1,0,1]))
    assert ((sample['actions']>=-2)&(sample['actions']<=3)).all()
    logp,_,means=actor.evaluate_actions(obs,np.array([0,1,0,1]),sample['pre_tanh'])
    torch.testing.assert_close(logp,sample['log_prob'])
    assert not torch.equal(means[0],means[1])
    private=actor._generator.get_state().clone()
    actor.act(obs[0],family=0,deterministic=True)
    assert torch.equal(private,actor._generator.get_state())
    assert torch.equal(before,torch.get_rng_state())
    # Stable density for saturated actions, without an inverse-tanh clamp.
    logp,_,_=actor.evaluate_actions(obs,0,np.full((4,1),30.))
    assert torch.isfinite(logp).all()


def test_auxiliary_family_separation_uses_actual_bounded_means():
    actor=FamilyGaussianActor(hidden_dim=16,seed=2)
    learner=PolicyLearner(actor,LearningConfig(lr=.02,batch_size=32,bootstrap_updates=120),seed=3)
    obs=np.zeros((32,8),dtype=np.float32);families=np.tile([0,1],16)
    targets=np.where(families[:,None]==0,.7,-.7).astype(np.float32)
    result=learner.bootstrap(AuxiliaryBatch(obs,targets,families))
    assert result['updates']==120 and result['promotion_claim'] is False
    assert actor.act(obs[0],0)[0]>.6 and actor.act(obs[0],1)[0]<-.6


def test_complete_mc_terminal_and_absorbing_tail_count_once():
    np.testing.assert_allclose(monte_carlo_cost_to_go([0,0,10],remaining=3,gamma=.9),[8.1,9,10])
    result=monte_carlo_cost_to_go([1,20],remaining=5,gamma=.9,terminated=True,absorbing_cost=2)
    tail=2*(1+.9+.9**2)
    np.testing.assert_allclose(result,[1+.9*(20+.9*tail),20+.9*tail])
    with pytest.raises(ValueError,match='Incomplete'):
        monte_carlo_cost_to_go([1],remaining=2)


def test_loo_separates_context_start_horizon_and_remaining():
    actor=FamilyGaussianActor(hidden_dim=8)
    eps=[episode(actor,[1,1],ref='a'),episode(actor,[3,3],ref='b'),
         episode(actor,[100,100,100],ref='c'),episode(actor,[200,200,200],ref='d'),
         episode(actor,[10,10],context='other',ref='e'),episode(actor,[30,30],context='other',ref='f')]
    batch=build_mc_batch(eps,policy_hash(actor))
    np.testing.assert_allclose(batch['baselines'][:4],[6,3,2,1])
    np.testing.assert_allclose(batch['baselines'][4:7],[600,400,200])
    np.testing.assert_allclose(batch['baselines'][-4:],[60,30,20,10])
    assert np.all(batch['replicate_counts']==2)


def test_full_delayed_ending_cost_drives_gradient_without_auxiliary():
    actor=FamilyGaussianActor(hidden_dim=16,seed=4)
    learner=PolicyLearner(actor,LearningConfig(lr=.01,epochs=4,batch_size=32,aux_weight=0),seed=5)
    # Identical local stage costs. Only the distant terminal distinguishes
    # negative versus positive sampled actions, and supplies earlier credit.
    episodes=[episode(actor,[0,0,0,0,0,0,0,0],ref='good',latent=-.5),
              episode(actor,[0,0,0,0,0,0,0,10],ref='bad',latent=.5)]
    before=actor.act(episodes[0].observations[0])[0]
    result=learner.update(episodes,policy_hash(actor))
    after=actor.act(episodes[0].observations[0])[0]
    assert after<before-.02
    assert result['target_owner']=='behavior_actor' and result['promotion_claim'] is False
    assert result['gradient_steps']==4
    assert result['return_scale']==800.
    assert result['mc_return_mean']==pytest.approx(5.)


def test_cost_ppo_uses_pessimistic_maximum_with_correct_sign():
    logp=torch.log(torch.tensor([1.5,.5],requires_grad=True))
    loss,_=clipped_cost_loss(logp,torch.zeros(2),torch.tensor([2.,-2.]),torch.ones(2),.2)
    assert loss.item()==pytest.approx(1.)  # max(3,2.4), max(-1,-1.6)


@pytest.mark.parametrize('varied',[False,True])
def test_pg_gradient_diagnostic_separates_zero_and_varied_own_returns_from_auxiliary(varied):
    actor=FamilyGaussianActor(hidden_dim=8,seed=13)
    learner=PolicyLearner(actor,LearningConfig(epochs=1,batch_size=32,aux_weight=.1),seed=14)
    episodes=[episode(actor,[0,0],ref='left',latent=-.5),
        episode(actor,[0,10 if varied else 0],ref='right',latent=.5)]
    aux=AuxiliaryBatch(np.zeros((4,8),dtype=np.float32),np.full((4,1),.8,dtype=np.float32),np.zeros(4,dtype=int))
    result=learner.update(episodes,aux=aux)
    assert result['diagnostic_backward_passes']==1 and result['diagnostic_wall_seconds']>=0
    assert result['gradient_steps']==1 and result['auxiliary_minibatches']==1
    assert result['auxiliary_loss_mean']>0
    assert result['weighted_auxiliary_loss_mean']==pytest.approx(.1*result['auxiliary_loss_mean'])
    assert result['loss']==pytest.approx(result['pg_loss_mean']+result['weighted_auxiliary_loss_mean'])
    if varied:
        assert result['raw_mc_advantage_std']==pytest.approx(10.)
        assert result['raw_mc_advantage_range']==[-10.,10.]
        assert result['episode_full_cost_range']==[0.,10.]
        assert result['first_minibatch_pg_gradient_norm']>0
    else:
        assert result['raw_mc_advantage_std']==0 and result['raw_mc_advantage_range']==[0.,0.]
        assert result['episode_full_cost_range']==[0.,0.]
        assert result['pg_loss_mean']==0 and result['first_minibatch_pg_gradient_norm']==0


def test_backward_and_failed_optimizer_intents_survive_without_fit_summary(tmp_path,monkeypatch):
    from src.gps.bidirectional.evidence import Archive
    actor=FamilyGaussianActor(hidden_dim=8);learner=PolicyLearner(actor)
    archive=Archive(tmp_path/'work');archive.work_scope='synthetic_failed_fit'
    actor._work_meter=dict(archive=archive,purpose='failure_injection',handles=[])
    def fail():raise RuntimeError('injected optimizer failure before Adam kernel')
    monkeypatch.setattr(learner.optimizer,'step',fail)
    with pytest.raises(RuntimeError,match='injected optimizer'):learner._step(actor.fc_mean.bias.sum())
    assert learner.updates==0
    with pytest.raises(RuntimeError):learner._step(torch.tensor(1.))
    archive.close()
    events=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
    requests={row['id']:row for row in events if row['kind'].endswith('_requested')}
    assert [r['kind'] for r in events if r['kind'].startswith(('gradient_','optimizer_'))]==[
        'gradient_backward_requested','gradient_backward_complete','optimizer_requested','optimizer_error',
        'gradient_backward_requested','gradient_backward_error']
    for row in events:
        if row['kind'].endswith(('_complete','_error')) and 'request_id' in row:
            assert row['request_id'] in requests
    assert not any(row['kind']=='fit_work' for row in events)


def test_mixed_owner_incomplete_and_reused_queries_rejected_before_update():
    actor=FamilyGaussianActor(hidden_dim=8);learner=PolicyLearner(actor)
    a=episode(actor,[0,1],ref='a');b=episode(actor,[0,2],ref='b')
    owner=policy_hash(actor)
    for bad in [replace(b,behavior_policy_hash='candidate'),replace(b,continuation_owner='teacher'),
                replace(b,complete=False),replace(b,query_ref='a'),replace(b,query_ref='')]:
        with pytest.raises(ValueError):learner.update([a,bad],owner)
        assert policy_hash(actor)==owner and learner.updates==0
    with pytest.raises(ValueError,match='replicates'):build_mc_batch([a],owner)


def test_deterministic_controller_is_not_stochastic_credit():
    actor=FamilyGaussianActor(hidden_dim=8)
    assert controller_hash(actor,deterministic=True)!=controller_hash(actor,deterministic=False)
    assert controller_hash(actor,deterministic=True,family=0)!=controller_hash(actor,deterministic=True,family=1)
    a=episode(actor,[0,1],ref='a');b=episode(actor,[0,2],ref='b')
    wrong=replace(b,behavior_controller_hash=controller_hash(actor,deterministic=True))
    with pytest.raises(ValueError,match='distinct controller'):PolicyLearner(actor).update([a,wrong])


def test_trial_clone_and_checkpoint_preserve_optimizer_and_private_streams():
    actor=FamilyGaussianActor(hidden_dim=8,seed=8)
    learner=PolicyLearner(actor,LearningConfig(epochs=1,batch_size=16),seed=9)
    initial=policy_hash(actor)
    learner.update([episode(actor,[0,0],ref='a',latent=-.3),episode(actor,[0,2],ref='b',latent=.3)])
    assert policy_hash(actor)!=initial and learner.optimizer.state
    clone=learner.clone_for_trial()
    assert clone is not learner and clone.actor is not actor and policy_hash(clone.actor)==policy_hash(actor)
    assert clone.updates==learner.updates
    np.testing.assert_array_equal(clone.rng.integers(100,size=10),learner.rng.integers(100,size=10))
    torch.testing.assert_close(clone.actor.sample(np.zeros((3,8),dtype=np.float32))['actions'],actor.sample(np.zeros((3,8),dtype=np.float32))['actions'])
    saved=learner.state_dict();cloned=clone.state_dict()
    key=next(iter(saved['optimizer']['state']))
    torch.testing.assert_close(saved['optimizer']['state'][key]['exp_avg'],cloned['optimizer']['state'][key]['exp_avg'])
    with torch.no_grad():next(clone.actor.parameters()).add_(1.)
    assert policy_hash(actor)!=policy_hash(clone.actor)
    checkpoint=actor.policy_state();checkpoint['policy_hash']='corrupt'
    with pytest.raises(ValueError,match='hash mismatch'):FamilyGaussianActor.from_checkpoint(checkpoint)
