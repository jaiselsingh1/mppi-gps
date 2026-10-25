"""Mixed-transition learner values preserve continuation and finite-cost meaning."""
import copy

import numpy as np
import pytest
import torch
from torch import nn

from src.gps.learner_value import LearnerValueTrainer, TransitionBatch, learner_cost_targets


class ConstantActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.full((6,), .2))
        self.register_buffer("action_scale", torch.ones(6))
        self.register_buffer("action_bias", torch.zeros(6))

    def forward(self, obs):
        mean = self.mean.expand(len(obs), -1)
        return mean, torch.zeros_like(mean)


def data(*, reset=1, count=8, source=None, remaining=None, terminated=None):
    return TransitionBatch(np.zeros((count, 17)), np.full((count, 6), .9), np.ones(count),
        np.ones((count, 17)), np.full(count, 4, dtype=int) if remaining is None else remaining,
        np.zeros(count, dtype=bool) if terminated is None else terminated,
        np.array(["teacher"] * (count - 1) + ["policy"]) if source is None else source,
        np.full(count, reset, dtype=int))


def test_finite_targets_follow_current_learner_and_absorb_fall_once():
    raw = torch.tensor([2., 25., 7., 25.], dtype=torch.float64, requires_grad=True)
    n = torch.tensor([4, 4, 1, 1])
    fall = torch.tensor([False, True, False, True])
    first = torch.tensor([4., float("nan"), float("nan"), float("nan")], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([6., float("nan"), float("nan"), float("nan")], dtype=torch.float64, requires_grad=True)
    result = learner_cost_targets(raw, n, fall, first, second, gamma=.5)
    # No second terminal20 charge; three finite future controls follow the fall.
    np.testing.assert_allclose(result.numpy(), [2. + .5 * 6., 25. + .5 * (5. + 2.5 + 1.25), 7., 25.])
    assert not result.requires_grad and result.grad_fn is None


@pytest.mark.parametrize("gamma", [0., 1.])
def test_finite_discount_boundaries(gamma):
    result = learner_cost_targets(torch.tensor([25.]), torch.tensor([4]), torch.tensor([True]),
                                  torch.tensor([float("nan")]), torch.tensor([float("nan")]), gamma=gamma)
    assert result.item() == pytest.approx(25. if gamma == 0 else 40.)


def test_bootstrap_queries_actor_action_not_recorded_teacher_action(monkeypatch):
    trainer, actor = LearnerValueTrainer(horizon=5, hidden_dim=8, seed=0), ConstantActor()
    queries = []

    class ActionCost(nn.Module):
        def forward(self, obs, actions, remaining):
            queries.append(actions.clone())
            return actions.square().sum(dim=1)

    trainer.q1_target = ActionCost()
    trainer.q2_target = ActionCost()
    batch = data(source="teacher")
    targets = trainer._targets(batch, np.arange(8), copy.deepcopy(actor).requires_grad_(False))
    expected = torch.tanh(torch.tensor(.2)).square() * 6
    np.testing.assert_allclose(targets.numpy(), 1. + trainer.gamma * expected.item(), rtol=1e-6)
    assert queries and all(torch.allclose(query, torch.full((8, 6), np.tanh(.2))) for query in queries)
    assert not np.allclose(queries[0].numpy(), batch.actions)


def test_balanced_source_sampling_actor_immutable_and_global_rng_preserved():
    actor = ConstantActor()
    actor.train()
    actor_before = copy.deepcopy(actor.state_dict())
    numpy_before, torch_before = copy.deepcopy(np.random.get_state()), torch.random.get_rng_state().clone()
    trainer = LearnerValueTrainer(horizon=5, hidden_dim=8, seed=9)
    metrics = trainer.fit(data(), data(reset=2), actor, updates=4, batch_size=6, seed=7)
    assert metrics["balanced_source_sampling"]
    assert metrics["sampled_sources"] == {"teacher": 12, "policy": 12}
    assert metrics["training_teacher_rows"] == 7 and metrics["training_policy_rows"] == 1
    assert metrics["target_continuation"].startswith("frozen current learner")
    assert actor.training and actor.mean.requires_grad
    for key in actor_before:
        torch.testing.assert_close(actor.state_dict()[key], actor_before[key], rtol=0, atol=0)
    assert actor.mean.grad is None
    assert all(np.array_equal(a, b) for a, b in zip(numpy_before, np.random.get_state()))
    torch.testing.assert_close(torch_before, torch.random.get_rng_state(), rtol=0, atol=0)
    assert not any(p.requires_grad for net in (trainer.q1_target, trainer.q2_target) for p in net.parameters())


def test_zero_terminal_ungrounded_guard_and_snapshot_immutability():
    trainer = LearnerValueTrainer(horizon=5, hidden_dim=8)
    obs, actions = np.zeros((2, 17)), np.zeros((2, 6))
    np.testing.assert_array_equal(trainer.snapshot().predict(obs, actions, 0), [0., 0.])
    with pytest.raises(RuntimeError, match="fitted"):
        trainer.snapshot().predict(obs, actions, 1)
    trainer.fit(data(), data(reset=2), ConstantActor(), updates=2, batch_size=4)
    frozen = trainer.snapshot()
    old = frozen.predict(obs, actions, np.array([0, 4]))
    assert old[0] == 0 and old[1] > 0
    trainer.fit(data(), data(reset=2), ConstantActor(), updates=2, batch_size=4)
    np.testing.assert_array_equal(frozen.predict(obs, actions, np.array([0, 4])), old)
    assert np.isfinite(trainer.predict(obs, actions, 4)).all()


def test_checkpoint_roundtrip_has_same_predictions_targets_and_optimizer():
    first = LearnerValueTrainer(horizon=5, hidden_dim=8, seed=3)
    first.fit(data(), data(reset=2), ConstantActor(), updates=3, batch_size=4)
    state = first.state_dict()
    second = LearnerValueTrainer(horizon=5, hidden_dim=8, seed=99)
    second.load_state_dict(state)
    obs, actions = np.ones((3, 17)), np.zeros((3, 6))
    np.testing.assert_array_equal(first.predict(obs, actions, 3), second.predict(obs, actions, 3))
    assert first.policy_id == second.policy_id and second.total_updates == 3
    assert len(second.optimizer.state) == len(first.optimizer.state) > 0
    state["q1"]["head.bias"].fill_(100.)
    np.testing.assert_array_equal(first.predict(obs, actions, 3), second.predict(obs, actions, 3))


def test_holdout_reset_overlap_and_invalid_sources_or_costs_rejected():
    trainer = LearnerValueTrainer(horizon=5, hidden_dim=8)
    with pytest.raises(ValueError, match="disjoint"):
        trainer.fit(data(), data(), ConstantActor(), updates=1)
    with pytest.raises(ValueError, match="Source"):
        data(source="accepted_plan")
    with pytest.raises(ValueError, match="positive|integer|counts"):
        data(remaining=np.zeros(8, dtype=int))
    with pytest.raises(ValueError, match="boolean"):
        data(terminated=np.zeros(8, dtype=int))
    with pytest.raises(ValueError, match="exceeds"):
        trainer.fit(data(remaining=np.full(8, 6)), data(reset=2), ConstantActor(), updates=1)


def test_terminal_data_learns_observed_raw_cost_and_does_not_bootstrap(monkeypatch):
    trainer = LearnerValueTrainer(horizon=1, hidden_dim=8, lr=.02)
    monkeypatch.setattr(trainer, "_actor_actions", lambda *a: pytest.fail("Terminal data queried learner continuation"))
    training = data(remaining=np.ones(8, dtype=int))
    holdout = data(reset=2, remaining=np.ones(8, dtype=int))
    metrics = trainer.fit(training, holdout, ConstantActor(), updates=150, batch_size=8)
    assert metrics["holdout_bellman_rmse"] < .05
    assert metrics["holdout_bellman_rmse"] < metrics["holdout_bellman_rmse_before"] / 20


def test_delayed_policy_cost_propagates_through_mixed_teacher_transitions():
    actor = ConstantActor()
    observations = np.zeros((8, 17)); observations[4:, 0] = 1.
    following = observations.copy(); following[:4, 0] = 1.
    actions = np.full((8, 6), np.tanh(.2))  # replay supports the current actor's next query

    def transitions(reset):
        return TransitionBatch(observations, actions, np.array([1.] * 4 + [9.] * 4), following,
            np.array([2] * 4 + [1] * 4), np.zeros(8, dtype=bool),
            np.array(["teacher", "policy"] * 4), np.full(8, reset))

    trainer = LearnerValueTrainer(horizon=2, gamma=.5, hidden_dim=16, lr=.01)
    trainer.fit(transitions(1), transitions(2), actor, updates=650, batch_size=16)
    predicted = trainer.predict(observations[[0, 4]], actions[[0, 4]], np.array([2, 1]))
    # State0's immediate cost is1, but its learner-owned next decision costs9.
    np.testing.assert_allclose(predicted, [1. + .5 * 9., 9.], atol=.15, rtol=0)


def test_bootstrap_and_raw_data_nonfinite_values_are_rejected():
    with pytest.raises(ValueError, match="bootstrap"):
        learner_cost_targets(torch.tensor([1.]), torch.tensor([2]), torch.tensor([False]),
            torch.tensor([float("nan")]), torch.tensor([1.]), gamma=.5)
    original = data()
    values = dict(vars(original)); values["costs"] = np.full(8, np.nan)
    with pytest.raises(ValueError, match="costs"):
        TransitionBatch(**values)
    values["costs"] = -np.ones(8)
    with pytest.raises(ValueError, match="RAW"):
        TransitionBatch(**values)
