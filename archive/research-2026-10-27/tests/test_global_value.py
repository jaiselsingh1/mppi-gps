"""Direct complete-return fitting must not manufacture Bellman targets or support."""
import copy
import json
import pickle

import numpy as np
import pytest
import torch

from src.gps.global_value import MonteCarloValue


@pytest.fixture(scope="module", autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def rows():
    groups = np.repeat(np.arange(6), 8)
    features = np.tile(np.linspace(-1., 1., 8), 6)[:, None]
    remaining = np.full(48, 20, dtype=np.int64)
    costs = np.full(48, 40.)
    return features, remaining, costs, groups


def fitted(**kwargs):
    return MonteCarloValue.fit(*rows(), total_steps=100, gamma=1., seed=17,
                              hidden=16, updates=100, **kwargs)


def test_measured_costs_define_direct_labels_not_physical_bounds():
    model = fitted()
    x, n, _, _ = rows()
    predicted, supported, distances = model.predict(x[:8], n[:8])
    np.testing.assert_allclose(predicted, 40., atol=2.)
    assert supported.all() and np.all(distances == 0)
    assert model.metrics["calibration_eligible"]
    assert "no bootstrap" in model.metrics["target_source"]
    assert "not a return bound" in model.metrics["normalizer"]
    assert all(not p.requires_grad and p.grad is None for p in model.network.parameters())


def test_whole_query_holdout_does_not_leak_into_fit_scale_or_cap():
    data = rows()
    original = MonteCarloValue.fit(*data, total_steps=100, gamma=1., seed=5, updates=20, hidden=8)
    changed = [a.copy() for a in data]
    changed[2][changed[3] >= 4] = 100_000.
    second = MonteCarloValue.fit(*changed, total_steps=100, gamma=1., seed=5, updates=20, hidden=8)
    assert original.metrics["training_groups"] == [0, 1, 2, 3]
    assert original.metrics["holdout_groups"] == [4, 5]
    assert not set(original.metrics["training_groups"]) & set(original.metrics["holdout_groups"])
    assert original.output_cap == second.output_cap == 1.
    for key, value in original.packet()["network"].items():
        np.testing.assert_array_equal(value, second.packet()["network"][key])
    assert second.metrics["holdout_raw_rmse"] > 90_000.
    assert not second.metrics["calibration_eligible"]


def test_prediction_diagnostics_compare_against_train_only_baseline():
    x, n, costs, groups = rows()
    costs[groups >= 4] = 100.
    model = MonteCarloValue.fit(x, n, costs, groups, total_steps=100, gamma=1.,
                                seed=5, updates=100, hidden=16)
    metrics = model.metrics
    # Training cost40 / finite scale100 = .4; held cost100 /100 =1.
    # A baseline that used the held labels would produce a different error.
    assert metrics["normalized_training_mean"] == pytest.approx(.4)
    assert metrics["holdout_training_mean_normalized_rmse"] == pytest.approx(.6)
    assert metrics["training_normalized_rmse"] < .03
    assert metrics["holdout_normalized_rmse"] > .5
    assert metrics["holdout_normalized_bias"] < -.5
    json.dumps(metrics, allow_nan=False)


def test_checkpoint_reproduces_predictions_and_has_no_torch_ipc_tensors():
    model = fitted()
    packet = model.packet()
    assert all(isinstance(value, np.ndarray) for value in packet["network"].values())
    restored = MonteCarloValue.from_packet(pickle.loads(pickle.dumps(packet)))
    x, n, _, _ = rows()
    for left, right in zip(model.predict(x, n), restored.predict(x, n)):
        np.testing.assert_array_equal(left, right)
    packet["mean"][:] = 42.
    assert not np.all(model.mean == 42.) and not np.all(restored.mean == 42.)


def test_support_abstains_for_new_features_or_different_remaining_horizon():
    model = fitted(remaining_tolerance=2)
    values, supported, distances = model.predict(np.array([[0.], [100.], [0.], [100.]]),
                                                np.array([20, 20, 80, 0]))
    assert supported.tolist() == [True, False, False, True]
    assert distances[1] > model.support_threshold and np.isinf(distances[2])
    assert values[-1] == distances[-1] == 0.
    assert np.isfinite(values).all()


def test_unsupported_holdout_metrics_are_strict_json_and_cannot_qualify():
    x, n, costs, groups = rows()
    n[groups >= 4] = 80
    model = MonteCarloValue.fit(x, n, costs, groups, total_steps=100, gamma=1.,
                                seed=0, updates=5, hidden=8, remaining_tolerance=2)
    json.dumps(model.metrics, allow_nan=False)
    assert model.metrics["holdout_distance_mean"] is None
    assert model.metrics["holdout_unsupported_count"] == 16
    assert model.metrics["holdout_supported_fraction"] == 0
    assert not model.metrics["calibration_eligible"]


def test_extreme_finite_holdout_cost_does_not_overflow_metric_serialization():
    x, n, costs, groups = rows()
    costs[groups >= 4] = 1e300
    model = MonteCarloValue.fit(x, n, costs, groups, total_steps=100, gamma=1.,
                                seed=0, updates=2, hidden=8)
    json.dumps(model.metrics, allow_nan=False)
    assert model.metrics["holdout_raw_rmse"] == pytest.approx(1e300)
    assert not model.metrics["calibration_eligible"]


def test_data_cap_contains_large_training_returns_without_unbounded_predictions():
    x, n, costs, groups = rows()
    costs[0] = 1e8
    model = MonteCarloValue.fit(x, n, costs, groups, total_steps=100, gamma=1.,
                                seed=0, updates=20, hidden=8)
    assert model.output_cap == pytest.approx(1.5e6)
    prediction, support, _ = model.predict(np.array([[1e12]]), np.array([20]))
    assert np.isfinite(prediction).all() and not support.any()
    assert 0 <= prediction[0] <= model.output_cap * 100.


def test_fitting_and_loading_preserve_global_random_streams():
    numpy_before, torch_before = copy.deepcopy(np.random.get_state()), torch.random.get_rng_state().clone()
    model = fitted()
    MonteCarloValue.from_packet(model.packet())
    assert all(np.array_equal(a, b) for a, b in zip(numpy_before, np.random.get_state()))
    torch.testing.assert_close(torch_before, torch.random.get_rng_state(), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["few_groups", "negative", "nonfinite", "fractional_remaining"])
def test_invalid_or_ungrounded_data_are_rejected(kind):
    x, n, costs, groups = rows()
    if kind == "few_groups":
        groups[:] %= 3
    elif kind == "negative":
        costs[0] = -1.
    elif kind == "nonfinite":
        costs[0] = np.nan
    else:
        n = n.astype(float) + .5
    with pytest.raises(ValueError):
        MonteCarloValue.fit(x, n, costs, groups, total_steps=100, gamma=1., seed=0,
                            updates=2, hidden=8)


@pytest.mark.parametrize("gamma", [0., .999, 1.])
def test_finite_normalization_and_exact_zero_horizon(gamma):
    x, n, costs, groups = rows()
    model = MonteCarloValue.fit(x, n, costs, groups, total_steps=100, gamma=gamma,
                                seed=0, updates=5, hidden=8)
    prediction, supported, distance = model.predict(np.array([[0.]]), np.array([0]))
    assert prediction.item() == 0 and supported.item() and distance.item() == 0
