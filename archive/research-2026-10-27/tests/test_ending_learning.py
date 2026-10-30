"""Pure-array learning checks: zero simulated/executed physics controls."""
import io

import numpy as np
import pytest

from src.ending.learning import (
    MCValue, compare_scores, complete_returns, es_gradient, fit_value,
    generate_candidates, gradient_alignment, propose_update,
)


def test_antithetic_candidates_are_common_private_rng_and_interleaved():
    base = np.array([.1, -.2, .3])
    first = generate_candidates(base, 17)
    np.random.seed(123)
    second = generate_candidates(base, 17)
    np.testing.assert_array_equal(first["eps"], second["eps"])
    assert first["candidates"].shape == (16, 3)
    np.testing.assert_allclose(first["candidates"][0::2]+first["candidates"][1::2], np.broadcast_to(2*base, (8, 3)))
    np.testing.assert_allclose(first["candidates"][0::2]-base, .3*first["eps"])
    np.testing.assert_array_equal(base, [.1, -.2, .3])


def test_es_gradient_matches_quadratic_finite_difference_and_descends():
    base = np.array([1., -2., .5])
    eps = np.eye(3)
    sigma = .3
    candidates = np.empty((6, 3))
    candidates[0::2], candidates[1::2] = base+sigma*eps, base-sigma*eps
    costs = np.sum(candidates**2, axis=1)
    gradient = es_gradient(eps, costs, sigma)
    np.testing.assert_allclose(gradient, 2*base/(3*20))
    update = propose_update(base, eps, costs, sigma)
    assert np.sum(update["parameters"]**2) < np.sum(base**2)
    np.testing.assert_allclose(update["step"], -.4*gradient)
    # Root replication averages observations; it never creates directions.
    np.testing.assert_allclose(es_gradient(eps, np.repeat(costs[:, None], 4, axis=1), sigma), gradient)


def test_es_step_is_bounded_and_invalid_inputs_fail():
    eps = np.eye(3)
    update = propose_update(np.zeros(3), eps, [100., 0., 100., 0., 100., 0.], .3)
    assert update["clipped"]
    assert update["step_norm"] == pytest.approx(.35)
    with pytest.raises(ValueError):
        es_gradient(eps, [1., 2.], .3)
    with pytest.raises(ValueError):
        generate_candidates(np.zeros(3), 0, sigma=0.)


def test_complete_returns_do_not_bootstrap_or_accept_declared_truncation():
    np.testing.assert_allclose(complete_returns([1., 2., 3.], remaining=3), [6., 5., 3.])
    assert complete_returns([], remaining=0).shape == (0,)
    with pytest.raises(ValueError, match="Truncated"):
        complete_returns([1.], complete=False)
    with pytest.raises(ValueError, match="exhaust"):
        complete_returns([1.], remaining=2)
    with pytest.raises(ValueError):
        complete_returns([np.nan])


def _training():
    t, y = np.meshgrid([0., 6., 12., 18., 24.], [-.8, -.4, 0., .4, .8], indexing="ij")
    states = np.column_stack((t.ravel(), y.ravel()))
    targets = 3. + 2*states[:, 0]/25 + states[:, 1]
    groups = np.where(states[:, 1] < 0, "negative_root", "other_root")
    return states, targets, groups


def _fit(states, targets, groups, **kwargs):
    return fit_value(states, targets, groups, total_steps=25, gate=12,
                     ending_y=-.1, target_actor_hash="frozen-actor-a", **kwargs)


def test_measured_value_fit_prediction_metadata_and_exact_terminal_boundary():
    states, targets, groups = _training()
    value = _fit(states, targets, groups)
    prediction = value.score_predict(states)
    assert np.sqrt(np.mean((prediction["values"]-targets)**2)) < .15
    assert prediction["prediction_rows"] == len(states)
    assert prediction["simulated_controls"] == 0
    assert value.fit_updates == 1
    assert value.packet()["simulated_controls"] == 0
    terminal = value.predict([[25., -1.], [25., 1.]])
    np.testing.assert_array_equal(terminal["values"], [0., 0.])
    assert terminal["supported"].all()
    assert prediction["target_actor_hash"] == "frozen-actor-a"


def test_root_weights_do_not_change_when_a_root_is_replicated():
    states, targets, groups = _training()
    value = _fit(states, targets, groups)
    extra = np.flatnonzero(groups == "negative_root")
    indices = np.concatenate((np.arange(len(states)), np.tile(extra, 7)))
    replicated = _fit(states[indices], targets[indices], groups[indices])
    np.testing.assert_allclose(value.predict(states)["values"], replicated.predict(states)["values"], atol=1e-10)


def test_value_support_is_geometric_and_actor_version_is_identity():
    states = np.array([[0., 0.], [6., 0.], [12., 0.], [18., 0.], [24., 0.]])
    value = _fit(states, np.ones(5), np.arange(5), support_radius=.05)
    prediction = value.predict([[12., 0.], [12., 1.]])
    np.testing.assert_array_equal(prediction["supported"], [True, False])
    packet = value.packet()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **packet)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        restored = MCValue.from_packet(saved)
    assert restored.digest == value.digest
    restored.target_actor_hash = "different-actor"
    assert restored.digest != value.digest
    np.testing.assert_array_equal(restored.train_features, value.train_features)


def test_value_rejects_impossible_targets_and_missing_provenance():
    states, targets, groups = _training()
    bad = targets.copy(); bad[0] = 26.
    with pytest.raises(ValueError, match="bound"):
        _fit(states, bad, groups)
    with pytest.raises(ValueError, match="exactly zero"):
        _fit([[25., 0.]], [1.], [0])
    with pytest.raises(ValueError, match="hash"):
        fit_value(states, targets, groups, total_steps=25, gate=12, ending_y=-.1, target_actor_hash="")


def test_three_scores_separate_value_error_from_continuation_mismatch():
    a = np.array([2., 4., 3., 6.])
    b = np.array([5., 4., 6., 4.])
    c = b + np.array([.5, -.5, -.5, .5])
    result = compare_scores(a, b, c, eps=np.array([[1., 0., 0.], [0., 1., 0.]]))
    np.testing.assert_allclose(result["continuation_error"]+result["critic_error"], result["combined_error"])
    assert result["critic_rmse"] == pytest.approx(.5)
    assert result["continuation_pair_error"]["sign_accuracy"] == 0.
    assert result["critic_pair_error"]["sign_accuracy"] == 1.
    assert result["gradient_alignments"]["continuation"]["cosine"] < 0
    assert result["simulated_controls"] == 0
    assert gradient_alignment(np.zeros(3), np.ones(3))["cosine"] is None
