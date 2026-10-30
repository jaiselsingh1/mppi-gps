"""Own-gradient metadata and tensor algebra; no model/actor forward calls."""
from copy import deepcopy
import numpy as np
import pytest

from src.gps.bidirectional.own_gradient import OwnGradientBatch, OwnGradientLearner, validate_batch, clone_gradient_learner, SEMANTICS


def batch(kind="full", gradient=2.):
    states = np.tile([24., .1], (200, 1)); obs = np.zeros((200, 8))
    obs[:, 0] = .1; obs[:, 1] = 24/25; obs[:, 2] = 1-24/25
    refs = tuple(dict(semantics=SEMANTICS, actor_hash="parent", task_hash="task", elapsed=24,
        state=[24., .1], source_episode=f"actual:{i}", center_query_ids=[f"live:{i}"],
        minus_query_ids=[f"minus:{i}"], plus_query_ids=[f"plus:{i}"],
        center_raw=dict(file="actual.npz", sha256="actual_sha", bytes=1),
        slope_raw=dict(file="probe.npz", sha256="probe_sha", bytes=1)) for i in range(200))
    return OwnGradientBatch(obs, np.zeros(200, dtype=int), states, np.zeros((200, 1)),
        np.full((200, 1), 0. if kind == "null" else gradient), "parent", "task", refs, kind)


def test_slopes_are_not_bounded_actions_and_null_still_has_all_unit_weight_rows():
    values = validate_batch(batch(gradient=300.), "parent", "task")
    assert np.max(values["gradients"]) == 300.
    np.testing.assert_array_equal(values["weights"], np.ones(200))
    values = validate_batch(batch("null"), "parent", "task")
    assert not np.any(values["gradients"])
    assert np.all(values["weights"] == 1)


def test_owner_task_state_and_complete_reference_corruption_fail_before_forward():
    with pytest.raises(ValueError): validate_batch(batch(), "updated_actor", "task")
    with pytest.raises(ValueError): validate_batch(batch(), "parent", "other_task")
    broken = batch(); broken.row_references[0]["plus_query_ids"] = ["live:0"]
    with pytest.raises(ValueError): validate_batch(broken, "parent", "task")
    broken = batch(); broken.observations[0, 0] += .1
    with pytest.raises(ValueError): validate_batch(broken, "parent", "task")


def test_linearized_cost_gradient_has_correct_sign_and_null_zero_without_network():
    import torch
    class FakeActor:
        def __init__(self): self.mean = torch.tensor(.2, requires_grad=True)
        def __call__(self, obs, families): return self.mean.expand(len(obs), 1), torch.zeros((len(obs), 1))
        def _bounded(self, latent): return latent.tanh()
    learner = object.__new__(OwnGradientLearner); learner.actor = FakeActor()
    aux = dict(observations=torch.zeros((3, 8)), families=torch.zeros(3, dtype=torch.long),
        weights=torch.ones(3), saved_means=torch.zeros((3, 1)), gradients=torch.full((3, 1), 2.), scale=25.)
    idx = torch.arange(3)
    loss = learner._aux_loss(aux, idx)
    derivative = torch.autograd.grad(loss, learner.actor.mean)[0]
    assert float(derivative) > 0  # Gradient descent moves action down when increasing action costs more.
    expected = 2*(1-torch.tanh(learner.actor.mean).square())/25
    assert torch.allclose(derivative, expected)
    aux["gradients"] = torch.zeros((3, 1))
    null_loss = learner._aux_loss(aux, idx)
    assert float(torch.autograd.grad(null_loss, learner.actor.mean)[0]) == 0.


def test_subclass_reuses_mc_update_and_exact_parent_optimizer_rng_without_forward():
    import torch
    from src.gps.bidirectional.policy import FamilyGaussianActor
    from src.gps.bidirectional.learning import PolicyLearner, LearningConfig
    from src.gps.bidirectional.feedback import _digest
    parent = PolicyLearner(FamilyGaussianActor(hidden_dim=4), LearningConfig(return_scale=25.), seed=9)
    for parameter in parent.actor.parameters():
        parent.optimizer.state[parameter] = dict(step=torch.tensor(3.), exp_avg=torch.full_like(parameter, .01), exp_avg_sq=torch.full_like(parameter, .02))
    before = _digest(parent.state_dict()); children = [clone_gradient_learner(parent, "task") for _ in range(3)]
    assert OwnGradientLearner.update is PolicyLearner.update
    assert OwnGradientLearner._step is PolicyLearner._step
    assert all(_digest(c.state_dict()) == before and c.own_gradient_task_hash == "task" for c in children)
    assert _digest(parent.state_dict()) == before
