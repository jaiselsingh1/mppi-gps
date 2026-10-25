"""Policy-value credit checks against explicit delayed-cost and failure returns."""

import numpy as np
import pytest
import torch

from src.gps.value_credit import (
    PolicyCostValue, PolicyEpisode, PolicyReturnAnchors, ValueCreditTrainer,
    ValueTransitionBatch, build_nstep_targets, endpoint_value_credit,
    value_data_from_episodes,
)


def _observations(length):
    obs = np.zeros((length, 17))
    obs[:, 0] = np.linspace(-1, 1, length)
    return obs


def test_cost_network_is_17_256_256_1_and_grounded_in_cost_units():
    model = PolicyCostValue()
    assert model.fc1.in_features == 17 and model.fc1.out_features == 256
    assert model.fc2.in_features == model.fc2.out_features == 256
    assert model.head.in_features == 256 and model.head.out_features == 1
    model.initialize_cost_scale(1250)
    values = model(torch.randn((4, 17)))
    torch.testing.assert_close(values, torch.full((4,), 1250.0))
    assert (values >= 0).all()


def test_targets_have_correct_discount_detach_and_ignore_failed_future_nan():
    prefix = torch.tensor([1.0, 7.0, 3.0], dtype=torch.float64, requires_grad=True)
    future = torch.tensor([10.0, float("nan"), float("inf")], dtype=torch.float64, requires_grad=True)
    target = build_nstep_targets(prefix, torch.tensor([2, 1, 3]), torch.tensor([False, True, True]),
                                future, gamma=.5, absorbing_cost=5)
    torch.testing.assert_close(target, torch.tensor([3.5, 12.0, 4.25], dtype=torch.float64))
    assert not target.requires_grad and target.grad_fn is None
    assert prefix.grad is None and future.grad is None


def test_raw_fall_charge_and_absorbing_future_are_each_counted_once():
    # 25 is an observed unhealthy cost5 plus terminal charge20. Its future
    # costs5 forever at gamma=.995: 25 + .995*5/.005 = 1020, not 2015.
    target = build_nstep_targets(torch.tensor([25.0], dtype=torch.float64), torch.tensor([1]),
                                torch.tensor([True]), torch.tensor([float("nan")]), gamma=.995)
    assert target.item() == pytest.approx(1020.0)


def test_healthy_cutoffs_bootstrap_instead_of_becoming_terminal():
    target = build_nstep_targets(torch.tensor([2.0]), torch.tensor([32]), torch.tensor([False]),
                                torch.tensor([1000.0]), gamma=.995)
    assert target.item() == pytest.approx(2 + .995 ** 32 * 1000, rel=1e-6)
    with pytest.raises(ValueError, match="endpoint"):
        build_nstep_targets(torch.tensor([2.0]), torch.tensor([32]), torch.tensor([False]),
                            torch.tensor([float("nan")]), gamma=.995)


def test_delayed_bad_outcome_propagates_backward_through_bellman_target():
    # Deterministic A --two zero-cost controls--> B --cost100--> terminal.
    # The local prefix at A contains no information about the cost at B.
    before = build_nstep_targets(torch.tensor([0.0]), torch.tensor([2]), torch.tensor([False]),
                                torch.tensor([0.0]), gamma=.9, absorbing_cost=0)
    measured_b = build_nstep_targets(torch.tensor([100.0]), torch.tensor([1]), torch.tensor([True]),
                                    torch.tensor([float("nan")]), gamma=.9, absorbing_cost=0)
    after = build_nstep_targets(torch.tensor([0.0]), torch.tensor([2]), torch.tensor([False]),
                               measured_b, gamma=.9, absorbing_cost=0)
    assert before.item() == 0
    assert measured_b.item() == 100
    assert after.item() == pytest.approx(81)


def test_episode_windows_and_long_return_anchors_use_raw_measured_outcomes():
    episode = PolicyEpisode(_observations(5), np.array([0, 0, 10, 25]), True, "pi0")
    batch, anchors = value_data_from_episodes([episode], gamma=.8, horizon=2)
    np.testing.assert_allclose(batch.discounted_costs, [0, 30])
    np.testing.assert_array_equal(batch.controls, [2, 2])
    np.testing.assert_array_equal(batch.terminated, [False, True])
    # A: .8²*10+.8³*25+.8⁴*5/(1-.8); B:10+.8*25+.8²*25.
    np.testing.assert_allclose(anchors.cost_targets, [29.44, 46])
    assert anchors.exact.all() and anchors.target_kind == "realized_absorbing"
    assert not batch.observations.flags.writeable


def test_policy_evaluation_rejects_plan_hybrid_and_mixed_policy_versions():
    accepted = PolicyEpisode(_observations(3), np.ones(2), True, "pi0", "accepted_plan")
    with pytest.raises(ValueError, match="accepted-plan"):
        value_data_from_episodes([accepted], horizon=2)
    a = PolicyEpisode(_observations(3), np.ones(2), True, "pi0")
    b = PolicyEpisode(_observations(3), np.ones(2), True, "pi1")
    with pytest.raises(ValueError, match="versions"):
        value_data_from_episodes([a, b], horizon=2)
    batch, anchors = value_data_from_episodes([a], horizon=2)
    bad_batch = ValueTransitionBatch(batch.observations, batch.next_observations, batch.discounted_costs,
                                    batch.controls, batch.terminated, batch.gamma, batch.policy_id, "accepted_plan")
    with pytest.raises(ValueError, match="accepted-plan"):
        ValueCreditTrainer(hidden_dim=8).fit(bad_batch, anchors, updates=1)


def test_cold_start_cannot_fit_healthy_lower_bounds_or_negative_sac_rewards():
    with pytest.raises(ValueError, match="nonnegative"):
        PolicyEpisode(_observations(3), -np.ones(2), True, "pi0")
    episode = PolicyEpisode(_observations(3), np.ones(2), False, "pi0")
    batch, anchors = value_data_from_episodes([episode], horizon=2)
    assert not anchors.exact.any() and anchors.target_kind == "healthy_prefix_lower_bound"
    trainer = ValueCreditTrainer(hidden_dim=8)
    with pytest.raises(RuntimeError, match="Cold-start"):
        trainer.fit(batch, anchors, updates=1)
    with pytest.raises(RuntimeError, match="grounded"):
        trainer.predict(np.zeros((1, 17)))


def test_estimated_healthy_anchors_use_frozen_prior_not_zero_or_false_mc_truth():
    trainer = ValueCreditTrainer(gamma=.5, hidden_dim=8)
    trainer.value.initialize_cost_scale(40)
    trainer.grounded, trainer.policy_id = True, "old_pi"
    prior = trainer.snapshot()
    episode = PolicyEpisode(_observations(3), np.array([2, 4]), False, "new_pi")
    batch, anchors = value_data_from_episodes([episode], gamma=.5, horizon=2, previous=prior)
    assert anchors.target_kind == "frozen_bootstrap" and not anchors.exact.any()
    np.testing.assert_allclose(anchors.cost_targets, [2 + .5 * 4 + .5 ** 2 * 40], atol=1e-5)
    assert not batch.terminated.any()
    trainer.hard_sync_target()
    log = trainer.fit(batch, anchors, updates=2, seed=11)
    assert log["exact_anchor_count"] == 0 and log["fitted_anchor_count"] == 1
    assert log["anchor_target_kind"] == "frozen_bootstrap" and log["healthy_bootstrap_count"] == 1


def test_joint_fit_uses_frozen_detached_target_and_hard_syncs_after_updates():
    torch.set_num_threads(1)
    torch.manual_seed(10)
    episode = PolicyEpisode(_observations(5), np.array([0, 0, 10, 25]), True, "pi0")
    batch, anchors = value_data_from_episodes([episode], gamma=.8, horizon=2)
    trainer = ValueCreditTrainer(gamma=.8, hidden_dim=16, lr=.01)
    log = trainer.fit(batch, anchors, updates=150, batch_size=2, seed=2)
    assert log["anchor_rmse_after"] < log["anchor_rmse_before"]
    assert log["target_frozen_during_fit"] and log["cost_scale"] > 1
    assert trainer.grounded and trainer.policy_id == "pi0" and trainer.total_updates == 150
    for name, parameter in trainer.target.named_parameters():
        assert not parameter.requires_grad and parameter.grad is None
        torch.testing.assert_close(parameter, dict(trainer.value.named_parameters())[name])
    assert np.isfinite(trainer.predict(anchors.observations)).all()


def test_credit_compares_same_endpoints_and_freezes_previous_parameters():
    trainer = ValueCreditTrainer(hidden_dim=8)
    trainer.value.initialize_cost_scale(100)
    trainer.grounded, trainer.policy_id = True, "pi0"
    old = trainer.snapshot()
    trainer.value.initialize_cost_scale(60)
    trainer.policy_id = "pi1"
    new = trainer.snapshot()
    endpoints = np.zeros((3, 17))
    credit = endpoint_value_credit(old, new, endpoints, tau=20)
    assert credit["ready"]
    np.testing.assert_allclose(credit["predicted_decrease"], 40, atol=1e-5)
    np.testing.assert_allclose(credit["weights"], np.tanh(2), atol=1e-6)
    np.testing.assert_allclose(old.predict(endpoints), 100, atol=1e-5)
    assert "estimation error" in credit["interpretation"]
    reverse = endpoint_value_credit(new, old, endpoints)
    np.testing.assert_array_equal(reverse["weights"], 0)


def test_uninitialized_snapshot_has_no_credit_and_unfrozen_model_is_rejected():
    trainer = ValueCreditTrainer(hidden_dim=8)
    previous = trainer.snapshot()
    credit = endpoint_value_credit(previous, trainer.snapshot(), np.zeros((2, 17)))
    assert not credit["ready"] and not credit["weights"].any()
    previous.model.head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        endpoint_value_credit(previous, trainer.snapshot(), np.zeros((2, 17)))


def test_value_checkpoint_roundtrips_target_scale_grounding_and_optimizer():
    episode = PolicyEpisode(_observations(3), np.array([2, 25]), True, "pi0")
    batch, anchors = value_data_from_episodes([episode], gamma=.8, horizon=2)
    trainer = ValueCreditTrainer(gamma=.8, hidden_dim=8)
    trainer.fit(batch, anchors, updates=2)
    saved = trainer.state_dict()
    restored = ValueCreditTrainer(gamma=.8, hidden_dim=8)
    restored.load_state_dict(saved)
    np.testing.assert_allclose(restored.predict(anchors.observations), trainer.predict(anchors.observations))
    assert restored.grounded and restored.policy_id == "pi0" and restored.total_updates == 2
    for key, value in restored.target.state_dict().items():
        torch.testing.assert_close(value, trainer.target.state_dict()[key])
    trainer.value.initialize_cost_scale(1)
    assert saved["value"]["cost_scale"].item() != trainer.value.cost_scale.item()
    with pytest.raises(ValueError, match="configuration"):
        ValueCreditTrainer(gamma=.9, hidden_dim=8).load_state_dict(saved)


@pytest.mark.parametrize("gamma", [1.0, -.1, float("nan")])
def test_discount_validation(gamma):
    with pytest.raises(ValueError):
        ValueCreditTrainer(gamma=gamma)
