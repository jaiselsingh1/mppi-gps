"""Policy/data/optimizer contracts; no environment or physics is constructed."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from src.gps.mppi_gps.policy import Policy, fit_policy


class CountingMeter:
    def __init__(self):
        self.counts = dict(neural_rows=0, backward_rows=0, optimizer_steps=0)
    def measure(self, kind, count, function, *, purpose):
        assert purpose
        self.counts[kind] += count
        return function()


def equal_tree(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert set(first) == set(second)
        for key in first:equal_tree(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):equal_tree(a, b)
    else:
        assert first == second


def data():
    rng = np.random.default_rng(13)
    observations = rng.normal(0, .2, (9, 17)).astype(np.float32)
    actions = np.tanh(observations[:, :6] + .1).astype(np.float32)
    return observations, actions


def test_bounded_shapes_meter_and_private_initialization():
    torch.manual_seed(91);np.random.seed(92)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = deepcopy(np.random.get_state())
    policy = Policy(seed=3)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    after = np.random.get_state()
    assert numpy_before[0] == after[0] and np.array_equal(numpy_before[1], after[1]) and numpy_before[2:] == after[2:]
    torch.manual_seed(9);np.random.seed(10)
    assert Policy(seed=3).hash() == policy.hash()
    assert Policy(seed=4).hash() != policy.hash()
    meter = CountingMeter()
    values = policy.predict(np.full((2, 3, 17), 100., dtype=np.float64), meter=meter)
    assert values.shape == (2, 3, 6) and np.isfinite(values).all() and np.all(np.abs(values) <= 1)
    assert policy.predict(np.zeros(17)).shape == (6,)
    assert policy.predict(np.empty((0, 17))).shape == (0, 6)
    assert meter.counts == dict(neural_rows=6, backward_rows=0, optimizer_steps=0)
    assert sum(p.numel() for p in policy.parameters()) == 19590


@pytest.mark.parametrize("observations,actions", [
    (np.zeros((2, 18)), np.zeros((2, 6))),
    (np.zeros((2, 17)), np.zeros((3, 6))),
    (np.zeros((1, 2, 17)), np.zeros((1, 2, 6))),
    (np.empty((0, 17)), np.empty((0, 6))),
    (np.full((2, 17), np.nan), np.zeros((2, 6))),
    (np.zeros((2, 17)), np.full((2, 6), 1.01)),
])
def test_pre_action_matrix_validation_precedes_any_metered_forward(observations, actions):
    policy = Policy();before = policy.hash();meter = CountingMeter()
    with pytest.raises(ValueError):fit_policy(policy, observations, actions, updates=1, meter=meter)
    assert policy.hash() == before and not any(meter.counts.values())


def test_one_update_matches_explicit_loss_adam_and_private_minibatch():
    parent = Policy(seed=8);obs, actions = data();before = parent.hash();meter = CountingMeter()
    reference = Policy(seed=8);optimizer = torch.optim.Adam(reference.parameters(), lr=3e-4, foreach=False, fused=False)
    rows = np.random.default_rng(44).integers(0, len(obs), size=4)
    prediction = reference(torch.as_tensor(obs[rows]));mse = torch.mean((prediction-torch.as_tensor(actions[rows]))**2)
    mse.backward();gradient_norm = float(torch.nn.utils.clip_grad_norm_(reference.parameters(), .5));optimizer.step()
    child, state, evidence = fit_policy(parent, obs, actions, updates=1, batch_size=4, eta_theta=.1, seed=44, meter=meter)
    equal_tree(child.state_dict(), reference.state_dict());equal_tree(state['adam'], optimizer.state_dict())
    assert parent.hash() == before and evidence['learning_curve'][0]['proximal_loss'] == 0.
    assert evidence['learning_curve'][0]['action_mse'] == float(mse.detach())
    assert evidence['learning_curve'][0]['unclipped_gradient_norm'] == gradient_norm
    assert meter.counts == dict(neural_rows=4, backward_rows=4, optimizer_steps=1)


def test_positive_proximal_block_resume_and_checkpoint_are_exact(tmp_path):
    normalization = dict(mean=np.arange(17)/20, std=np.linspace(.5, 2, 17))
    parent = Policy(seed=2, normalization=normalization);obs, actions = data();before = parent.hash()
    whole, whole_state, whole_evidence = fit_policy(parent, obs, actions, updates=4, batch_size=5, eta_theta=.07, seed=41)
    first, first_state, first_evidence = fit_policy(parent, obs, actions, updates=2, batch_size=5, eta_theta=.07, seed=41)
    path = tmp_path/'policy.pt';reference = first.checkpoint(path, first_state, {'role':'fixed unit checkpoint'})
    loaded, loaded_state, metadata = Policy.from_checkpoint(path)
    assert loaded.hash() == first.hash() == reference['policy_hash'] and metadata == {'role':'fixed unit checkpoint'}
    equal_tree(loaded_state, first_state)
    with pytest.raises(FileExistsError):first.checkpoint(path, first_state)
    resumed, resumed_state, resumed_evidence = fit_policy(loaded, obs, actions, loaded_state, updates=2,
        batch_size=5, eta_theta=.07, seed=999, resume_fit=True)
    equal_tree(resumed.state_dict(), whole.state_dict());equal_tree(resumed_state['adam'], whole_state['adam'])
    assert resumed_state['rng'] == whole_state['rng'] and resumed_state['updates'] == resumed_state['fit_updates'] == 4
    assert first_evidence['learning_curve']+resumed_evidence['learning_curve'] == whole_evidence['learning_curve']
    assert parent.hash() == before and resumed_state['anchor_policy_hash'] == before
    assert resumed.normalization['mean'].tobytes() == parent.normalization['mean'].tobytes()
    assert resumed.normalization['std'].tobytes() == parent.normalization['std'].tobytes()
    assert whole_evidence['learning_curve'][-1]['proximal_loss'] > 0


def test_new_block_keeps_adam_rng_but_resets_anchor_and_rejects_wrong_owner():
    obs, actions = data();parent = Policy(7)
    child, state, _ = fit_policy(parent, obs, actions, updates=2, batch_size=3, seed=20)
    new, new_state, evidence = fit_policy(child, obs, actions, state, updates=1, batch_size=3)
    assert new_state['updates'] == 3 and new_state['fit_updates'] == 1
    assert new_state['anchor_policy_hash'] == child.hash() and evidence['learning_curve'][0]['proximal_loss'] == 0
    with pytest.raises(ValueError, match='different parent'):fit_policy(parent, obs, actions, state, updates=1)
    with pytest.raises(ValueError, match='unchanged data'):fit_policy(child, obs+.01, actions, state,
        updates=1, batch_size=3, seed=20, resume_fit=True)
    with pytest.raises(ValueError, match='resume_fit'):fit_policy(new, obs, actions, updates=1, resume_fit=True)
