"""Synthetic independent coherent-head audit fixtures, authored without execution.

The audit module is pure NumPy arithmetic. These fixtures construct checkpoint
dictionaries directly; they import no actor, producer, task, engine, or Torch.
No actor forward rows or physical controls are exercised by these tests.
"""
from copy import deepcopy

import numpy as np
import pytest

from src.gps.bidirectional.coherent_audit import (
    candidate_coordinates,
    declared_bound,
    make_basis,
    select_from_scores,
    verify_head_actor,
)


def synthetic_actor(dtype=np.float32):
    """A complete, nonexecuting policy-state fixture, including a private RNG."""
    actor = {
        "fc1.weight": np.linspace(-.2, .2, 128*9, dtype=dtype).reshape(128, 9),
        "fc1.bias": np.linspace(-.1, .1, 128, dtype=dtype),
        "fc2.weight": np.linspace(-.05, .05, 128*128, dtype=dtype).reshape(128, 128),
        "fc2.bias": np.linspace(.1, -.1, 128, dtype=dtype),
        "fc_mean.weight": np.linspace(-.02, .02, 128, dtype=dtype)[None, :],
        "fc_mean.bias": np.array([.03], dtype=dtype),
        "log_std": np.array([-2.], dtype=dtype),
        "action_low": np.array([-1.], dtype=dtype),
        "action_high": np.array([1.], dtype=dtype),
        "action_scale": np.array([1.], dtype=dtype),
        "action_bias": np.array([0.], dtype=dtype),
        "synthetic_numeric_buffer": np.array([7, 11], dtype=np.int64),
    }
    return {
        "format_version": 1,
        "policy_type": "bidirectional-family-gaussian-v1",
        "obs_dim": 8,
        "hidden_dim": 128,
        "num_families": 1,
        "action_low": [-1.],
        "action_high": [1.],
        "actor": actor,
        "sampling_rng_state": np.arange(32, dtype=np.uint8),
        "policy_hash": "synthetic-parent-hash-placeholder",
    }


def expected_head(parent, basis, coordinates):
    """Apply only the three declared changes, with one final original-dtype cast."""
    result = deepcopy(parent)
    result["policy_hash"] = "synthetic-candidate-hash-placeholder"
    c = np.asarray(coordinates, dtype=np.float64)
    changes = {
        "fc_mean.bias": .1*c[0],
        "fc_mean.weight": .1*(c[1:5]@basis)[None, :],
        "log_std": .5*c[5],
    }
    for key, delta in changes.items():
        original = parent["actor"][key]
        result["actor"][key] = (original.astype(np.float64)+delta).astype(original.dtype)
    return result


def score(index, *, ordinary=(250., 250.), raw=(190., 190.),
          long=(190., 190.), captures=(6, 6)):
    roles = {}
    for name, values in (("ordinary", ordinary), ("raw", raw), ("long", long)):
        roles[name] = {
            "mean": {"mean_cost": float(values[0]), "captures": captures[0]},
            "stochastic": {"mean_cost": float(values[1]), "captures": captures[1]},
        }
    return {"actor_hash": f"synthetic-candidate-{index}", "roles": roles}


def scores():
    return [score(i) for i in range(13)]


def test_declared_bound_counts_every_paid_role_mode_and_stage():
    bound = declared_bound()
    # Each training actor pays all three panels in both modes, even when the
    # selector may inspect only one source panel.
    actors = 5*3*13
    per_actor_controls = 2*(8*25+8*19+8*19)
    training = actors*per_actor_controls
    qualification = 6*16*25*2
    final = 3*64*25*2
    expected = {
        "total_controls": training+qualification+final,
        "training_controls": training,
        "qualification_controls": qualification,
        "final_controls": final,
        "actor_rows": training+qualification+final,
        "training_actor_versions": actors,
        "training_episodes": actors*3*8*2,
        "qualification_episodes": 6*16*2,
        "final_episodes": 3*64*2,
        "model_controls": 0,
        "adam_steps": 0,
    }
    assert {key: bound[key] for key in expected} == expected
    assert bound["total_controls"] == 210960
    assert bound["training_controls"] == 196560
    assert bound["actor_rows"] == bound["total_controls"]


def test_basis_is_the_declared_pcg64_draw_with_canonical_qr_signs():
    actual = make_basis()
    draw = np.random.Generator(np.random.PCG64(1701)).standard_normal((128, 4))
    q, r = np.linalg.qr(draw, mode="reduced")
    expected = (q*np.where(np.diag(r) < 0., -1., 1.)[None, :]).T
    assert actual.shape == (4, 128)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual@actual.T, np.eye(4), rtol=0, atol=1e-14)
    assert np.all(np.diag(actual@draw) > 0.)
    np.testing.assert_array_equal(make_basis(), actual)


@pytest.mark.parametrize("center,step", [
    ([0., 0., 0., 0., 0., 0.], 0),
    ([.25, -.5, 0., .25, 0., 0.], 1),
    ([1., -1., 0., 0., 0., 0.], 2),
])
def test_coordinate_bank_is_center_first_then_minus_plus_each_axis(center, step):
    center = np.asarray(center, dtype=np.float64)
    before = center.copy()
    actual = candidate_coordinates(center, step)
    expected = [center.copy()]
    for axis in range(6):
        for direction in (-1., 1.):
            candidate = center.copy()
            candidate[axis] += direction
            expected.append(candidate)
    assert actual.shape == (13, 6)
    np.testing.assert_array_equal(actual, np.stack(expected))
    np.testing.assert_array_equal(center, before)
    assert np.all(np.abs(actual).sum(axis=1) <= step+1.)


@pytest.mark.parametrize("center,step", [
    ([0.]*5, 0),
    ([float("nan"), 0., 0., 0., 0., 0.], 0),
    ([.001, 0., 0., 0., 0., 0.], 0),
    ([0.]*6, -1),
    ([0.]*6, 3),
    ([0.]*6, .5),
])
def test_coordinate_bank_rejects_bad_geometry_or_undeclared_step(center, step):
    with pytest.raises(ValueError):
        candidate_coordinates(np.asarray(center, dtype=float), step)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_head_verification_accepts_only_declared_changes_cast_to_original_dtype(dtype):
    parent = synthetic_actor(dtype)
    basis = make_basis()
    coordinates = np.array([.25, -.5, .125, .25, -.125, .5])
    candidate = expected_head(parent, basis, coordinates)
    assert verify_head_actor(parent, candidate, basis, coordinates) is True
    for key, original in parent["actor"].items():
        assert candidate["actor"][key].dtype == original.dtype
        if key not in {"fc_mean.bias", "fc_mean.weight", "log_std"}:
            np.testing.assert_array_equal(candidate["actor"][key], original)
    np.testing.assert_array_equal(candidate["sampling_rng_state"], parent["sampling_rng_state"])


def test_zero_coordinates_preserve_all_actor_arrays_despite_new_hash_label():
    parent = synthetic_actor()
    basis, coordinates = make_basis(), np.zeros(6)
    candidate = expected_head(parent, basis, coordinates)
    assert candidate["policy_hash"] != parent["policy_hash"]
    assert verify_head_actor(parent, candidate, basis, coordinates) is True
    for key in parent["actor"]:
        np.testing.assert_array_equal(candidate["actor"][key], parent["actor"][key])


@pytest.mark.parametrize("key", ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"])
def test_head_verification_rejects_any_changed_trunk_parameter(key):
    parent = synthetic_actor()
    basis, coordinates = make_basis(), np.array([1., 0., 0., 0., 0., 0.])
    candidate = expected_head(parent, basis, coordinates)
    candidate["actor"][key].flat[0] += .001
    with pytest.raises(ValueError):
        verify_head_actor(parent, candidate, basis, coordinates)


@pytest.mark.parametrize("key", ["fc_mean.bias", "fc_mean.weight", "log_std"])
def test_even_one_wrong_head_ulp_is_not_a_permitted_proposal(key):
    parent = synthetic_actor()
    basis, coordinates = make_basis(), np.array([.25, -.5, 0., .25, 0., .5])
    candidate = expected_head(parent, basis, coordinates)
    value = candidate["actor"][key].flat[0]
    candidate["actor"][key].flat[0] = np.nextafter(value, np.float32(np.inf))
    with pytest.raises(ValueError):
        verify_head_actor(parent, candidate, basis, coordinates)


def test_head_verification_rejects_private_rng_change_without_any_forward():
    parent = synthetic_actor()
    basis, coordinates = make_basis(), np.zeros(6)
    candidate = expected_head(parent, basis, coordinates)
    candidate["sampling_rng_state"][0] ^= np.uint8(1)
    with pytest.raises(ValueError):
        verify_head_actor(parent, candidate, basis, coordinates)


def test_head_verification_rejects_private_rng_dtype_change_with_identical_values():
    parent = synthetic_actor()
    basis, coordinates = make_basis(), np.zeros(6)
    candidate = expected_head(parent, basis, coordinates)
    candidate["sampling_rng_state"] = parent["sampling_rng_state"].astype(np.int64)
    assert np.array_equal(candidate["sampling_rng_state"], parent["sampling_rng_state"])
    assert candidate["sampling_rng_state"].dtype != parent["sampling_rng_state"].dtype
    with pytest.raises(ValueError):
        verify_head_actor(parent, candidate, basis, coordinates)


def test_head_verification_rejects_negative_zero_in_an_unchanged_numeric_buffer():
    parent = synthetic_actor()
    basis, coordinates = make_basis(), np.zeros(6)
    candidate = expected_head(parent, basis, coordinates)
    candidate["actor"]["action_bias"][0] = -0.
    original, altered = parent["actor"]["action_bias"], candidate["actor"]["action_bias"]
    assert np.array_equal(original, altered)
    assert original.dtype == altered.dtype and original.shape == altered.shape
    assert original.tobytes() != altered.tobytes()
    with pytest.raises(ValueError):
        verify_head_actor(parent, candidate, basis, coordinates)


@pytest.mark.parametrize("key", ["action_scale", "synthetic_numeric_buffer"])
def test_head_verification_preserves_numeric_buffers_exactly(key):
    parent = synthetic_actor()
    basis, coordinates = make_basis(), np.zeros(6)
    candidate = expected_head(parent, basis, coordinates)
    candidate["actor"][key].flat[0] += 1
    with pytest.raises(ValueError):
        verify_head_actor(parent, candidate, basis, coordinates)


def test_head_verification_rejects_dtype_change_even_with_same_numeric_values():
    parent = synthetic_actor()
    basis, coordinates = make_basis(), np.zeros(6)
    candidate = expected_head(parent, basis, coordinates)
    candidate["actor"]["fc_mean.weight"] = candidate["actor"]["fc_mean.weight"].astype(np.float64)
    with pytest.raises(ValueError):
        verify_head_actor(parent, candidate, basis, coordinates)


@pytest.mark.parametrize("field,value", [("obs_dim", 9), ("hidden_dim", 64), ("num_families", 2)])
def test_head_verification_requires_the_declared_architecture(field, value):
    parent = synthetic_actor()
    basis, coordinates = make_basis(), np.zeros(6)
    candidate = expected_head(parent, basis, coordinates)
    candidate[field] = value
    with pytest.raises(ValueError):
        verify_head_actor(parent, candidate, basis, coordinates)


def test_score_objective_uses_separate_mode_means_and_role_horizons():
    values = scores()
    values[2] = score(2, ordinary=(200., 225.), raw=(95., 114.))
    result = select_from_scores(values, "raw", .25)
    assert result["selected_index"] == 2
    assert all(result["feasible_flags"])
    # ordinary=(200+225)/50=8.5, raw=(95+114)/38=5.5.
    assert result["objectives"][2] == pytest.approx((8.5+.25*5.5)/1.25)
    assert result["objectives"][0] == pytest.approx(10.)


@pytest.mark.parametrize("source,beta,hidden", [
    ("ordinary", 0., ("raw", "long")),
    ("raw", .25, ("long",)),
    ("long", 1., ("raw",)),
])
def test_unused_paid_panels_do_not_influence_selection(source, beta, hidden):
    values = scores()
    values[4] = score(4, ordinary=(200., 200.), raw=(95., 95.), long=(95., 95.))
    expected = select_from_scores(values, source, beta)
    before = deepcopy(values)
    for index, entry in enumerate(values):
        for role in hidden:
            entry["roles"][role]["mean"]["mean_cost"] = float((13-index)*10000)
            entry["roles"][role]["stochastic"]["mean_cost"] = float(index*100000)
            entry["roles"][role]["mean"]["captures"] = index % 9
            entry["roles"][role]["stochastic"]["captures"] = (index+1) % 9
    actual = select_from_scores(values, source, beta)
    assert actual["selected_index"] == expected["selected_index"] == 4
    assert list(actual["feasible_flags"]) == list(expected["feasible_flags"])
    np.testing.assert_array_equal(actual["objectives"], expected["objectives"])
    # The meaningful mutation is confined to panels excluded by the protocol.
    for index in range(13):
        assert values[index]["roles"]["ordinary"] == before[index]["roles"]["ordinary"]


@pytest.mark.parametrize("mode", ["mean", "stochastic"])
def test_ordinary_cost_regression_in_either_mode_blocks_a_better_source(mode):
    values = scores()
    values[1] = score(1, ordinary=(249., 249.), raw=(0., 0.))
    values[1]["roles"]["ordinary"][mode]["mean_cost"] = 251.
    result = select_from_scores(values, "raw", 1.)
    # The sum of the two ordinary costs still equals the center's sum. The
    # protocol requires each independently executed controller to avoid loss.
    assert sum(values[1]["roles"]["ordinary"][m]["mean_cost"] for m in ("mean", "stochastic")) == 500.
    assert not result["feasible_flags"][1]
    assert result["selected_index"] == 0


@pytest.mark.parametrize("mode", ["mean", "stochastic"])
def test_ordinary_capture_regression_in_either_mode_blocks_lower_cost(mode):
    values = scores()
    values[1] = score(1, ordinary=(100., 100.), long=(0., 0.), captures=(7, 7))
    values[1]["roles"]["ordinary"][mode]["captures"] = 5
    result = select_from_scores(values, "long", 1.)
    assert not result["feasible_flags"][1]
    assert result["selected_index"] == 0


def test_stable_ties_keep_the_center_then_the_first_feasible_minimizer():
    values = scores()
    assert select_from_scores(values, "ordinary", 0.)["selected_index"] == 0
    values[3] = score(3, ordinary=(200., 200.))
    values[7] = score(7, ordinary=(200., 200.))
    result = select_from_scores(values, "ordinary", 0.)
    assert result["selected_index"] == 3
    assert result["objectives"][3] == result["objectives"][7]


@pytest.mark.parametrize("source,beta", [("ordinary", .25), ("raw", 0.), ("long", .5), ("other", 1.)])
def test_selector_rejects_undeclared_source_beta_pairs(source, beta):
    with pytest.raises(ValueError):
        select_from_scores(scores(), source, beta)


def test_selector_requires_exactly_thirteen_candidates():
    with pytest.raises(ValueError):
        select_from_scores(scores()[:-1], "raw", .25)
