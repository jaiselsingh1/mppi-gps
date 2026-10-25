"""Independent selection and weighted-regression checks for policy-only GPS."""

import math

import numpy as np
import pytest
import torch

from src.gps.selection import build_selection, recent_selection, weighted_mse


def _selection(mode="selective", **overrides):
    args = dict(observations=np.arange(12).reshape(3, 4),
                proposals=np.array([[0.3, -0.4], [0.6, 0.8], [-0.2, 0.1]]),
                old_actions=np.zeros((3, 2)), nominal_cost=np.array([5., 5., 5.]),
                plan_cost=np.array([4., 5., 6.]), sigma=0.5, mode=mode, tau_s=2.)
    args.update(overrides)
    return build_selection(**args)


def test_anchor_rejects_only_strictly_worse_plans_and_keeps_ties():
    data = _selection("anchor")
    np.testing.assert_array_equal(data.actions[:2], [[0.3, -0.4], [0.6, 0.8]])
    np.testing.assert_array_equal(data.actions[2], [0., 0.])
    np.testing.assert_array_equal(data.weights, [1., 1., 1.])
    np.testing.assert_array_equal(data.improvement, [1., 0., -1.])


def test_selective_weights_match_independent_tanh_gaussian_and_drop_ties():
    data = _selection()
    expected = math.tanh(1. / 2.) * math.exp(-(.3**2 + .4**2) / (2 * .5**2))
    np.testing.assert_allclose(data.weights, [expected, 0., 0.], rtol=1e-15)
    np.testing.assert_array_equal(data.actions, data.proposed_actions)
    np.testing.assert_allclose(data.action_delta, [.5, 1., math.sqrt(.05)])


def test_snapshots_copy_inputs_and_replay_keeps_old_policy_weights():
    old = np.zeros((3, 2))
    proposals = np.array([[.3, -.4], [.6, .8], [-.2, .1]])
    first = _selection(old_actions=old, proposals=proposals)
    frozen_weight = first.weights.copy()
    old[:] = .5
    proposals[:] = 0.
    np.testing.assert_array_equal(first.old_actions, np.zeros((3, 2)))
    np.testing.assert_array_equal(first.proposed_actions[0], [.3, -.4])
    with pytest.raises(ValueError, match="read-only"):
        first.weights[0] = 1.
    second = _selection(sigma=.05)
    third = _selection("anchor")
    replay = recent_selection([first, second], window=2)
    np.testing.assert_array_equal(replay.weights[:3], frozen_weight)
    np.testing.assert_array_equal(replay.weights[3:], second.weights)
    latest = recent_selection([first, second, third], window=2)
    np.testing.assert_array_equal(latest.weights, np.r_[second.weights, third.weights])
    assert latest.observations.shape == (6, 4)
    assert latest.actions.shape == (6, 2)
    assert not np.shares_memory(replay.weights, first.weights)


def test_selective_underflow_drops_distant_improvements_without_nan():
    data = _selection(sigma=1e-300, tau_s=1e-300)
    np.testing.assert_array_equal(data.weights, np.zeros(3))
    same = _selection(old_actions=np.array([[.3, -.4], [.6, .8], [-.2, .1]]),
                      sigma=1e-300, tau_s=1e-300)
    np.testing.assert_array_equal(same.weights, [1., 0., 0.])


@pytest.mark.parametrize("field", ["observations", "proposals", "old_actions", "nominal_cost", "plan_cost"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_corrupt_estep_data_is_rejected(field, invalid):
    shapes = {"observations": (3, 4), "proposals": (3, 2), "old_actions": (3, 2),
              "nominal_cost": (3,), "plan_cost": (3,)}
    value = np.zeros(shapes[field])
    value.flat[0] = invalid
    with pytest.raises(ValueError, match="finite"):
        _selection(**{field: value})


@pytest.mark.parametrize("overrides, message", [
    ({"mode": "other"}, "mode"),
    ({"sigma": 0.}, "sigma"), ({"sigma": float("inf")}, "sigma"),
    ({"tau_s": -1.}, "tau_s"), ({"tau_s": float("nan")}, "tau_s"),
    ({"observations": np.zeros(3)}, "observations"),
    ({"proposals": np.zeros((2, 2))}, "proposals"),
    ({"old_actions": np.zeros((3, 1))}, "old_actions"),
    ({"nominal_cost": np.zeros((3, 1))}, "cost"),
])
def test_selection_validates_scalars_and_shapes(overrides, message):
    with pytest.raises(ValueError, match=message):
        _selection(**overrides)


def test_empty_dataset_and_replay_are_shape_stable():
    empty = build_selection(np.empty((0, 4)), np.empty((0, 2)), np.empty((0, 2)),
                            np.empty(0), np.empty(0), .3, "selective")
    assert recent_selection([empty]).weights.shape == (0,)
    with pytest.raises(ValueError, match="at least one"):
        recent_selection([])
    with pytest.raises(ValueError, match="positive integer"):
        recent_selection([empty], window=0)


def test_weighted_loss_matches_squared_norm_oracle_and_gradient():
    prediction = torch.tensor([[1., 2.], [3., 1.], [100., 100.]], dtype=torch.float64,
                              requires_grad=True)
    target = torch.tensor([[0., 0.], [1., 1.], [0., 0.]], dtype=torch.float64)
    weights = torch.tensor([1., 3., 0.], dtype=torch.float64)
    loss = weighted_mse(prediction, target, weights)
    assert loss.item() == (1 * (1**2 + 2**2) + 3 * (2**2 + 0**2)) / 4
    loss.backward()
    np.testing.assert_array_equal(prediction.grad.numpy(), [[.5, 1.], [3., 0.], [0., 0.]])
    assert weighted_mse(prediction, target, weights * 7).item() == loss.item()


@pytest.mark.parametrize("batch_size", [0, 3])
def test_all_zero_weight_loss_is_finite_differentiable_zero(batch_size):
    prediction = torch.ones((batch_size, 2), requires_grad=True)
    loss = weighted_mse(prediction, torch.zeros_like(prediction), torch.zeros(batch_size))
    assert loss.item() == 0.
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_zero_weight_rows_do_not_overflow_loss():
    prediction = torch.tensor([[1., 2.], [1e30, 1e30]], requires_grad=True)
    loss = weighted_mse(prediction, torch.zeros_like(prediction), torch.tensor([1., 0.]))
    assert loss.item() == 5.
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[2., 4.], [0., 0.]]))


def test_all_zero_weights_do_not_overflow_even_the_prediction_sum():
    prediction = torch.full((3, 2), 1e38, requires_grad=True)
    loss = weighted_mse(prediction, torch.zeros_like(prediction), torch.zeros(3))
    assert loss.item() == 0.
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


@pytest.mark.parametrize("weights", [torch.tensor([-1., 1.]), torch.tensor([float("nan"), 1.]),
                                     torch.ones(2, 1), torch.ones(2, dtype=torch.int64)])
def test_weighted_loss_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        weighted_mse(torch.zeros(2, 3), torch.zeros(2, 3), weights)
