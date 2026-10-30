"""Measured-return ownership, ancestry splits and paired contrast integrity."""
import copy
import json

import numpy as np
import pytest

from src.gps.paired_value import PairedRidgeValue


def records(*, constant=False, replicates=2):
    rows = []
    for group in range(6):
        for pair in range(4):
            for replicate in range(replicates):
                for side, offset in (("baseline", -.2), ("candidate", .2)):
                    features = np.zeros(17)
                    if not constant:
                        features[0] = -.6 + .4 * pair + offset
                        features[1] = group * .02
                    normalized = .4 if constant else .4 + .08 * features[0] - .03 * features[1]
                    rows.append(dict(features=features.tolist(), remaining=20, cost=100. * normalized,
                                     group=group, pair_id=f"pair-{pair}", side=side,
                                     replicate=replicate, source="train"))
    return rows


def fit(rows=None, **kwargs):
    return PairedRidgeValue.fit(records() if rows is None else rows,
                                total_steps=100, gamma=1., **kwargs)


def test_rejects_validation_qualification_and_other_target_sources():
    for source in ("validation", "qualification", "evaluation", None):
        rows = records()
        rows[0]["source"] = source
        with pytest.raises(ValueError, match="source='train'"):
            fit(rows)


def test_whole_root_holdout_never_sets_weights_scaler_cap_or_support():
    rows = records()
    original = fit(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["group"] >= 4:
            row["features"] = (np.asarray(row["features"]) + 100.).tolist()
            row["cost"] = 100_000.
    second = fit(changed)
    assert original.metrics["training_groups"] == [0, 1, 2, 3]
    assert original.metrics["holdout_groups"] == [4, 5]
    assert not set(original.metrics["training_groups"]) & set(original.metrics["holdout_groups"])
    for key in ("mean", "scale", "weights", "support_features", "support_remaining"):
        np.testing.assert_array_equal(original.packet()[key], second.packet()[key])
    assert original.output_cap == second.output_cap == 1.
    assert second.metrics["holdout_raw_rmse"] > 99_000
    assert not second.metrics["calibration_eligible"]
    assert second.metrics["holdout_supported_fraction"] == 0.


def test_fixed_linear_cost_and_pair_contrasts_are_learned_without_action_labels():
    rows = records()
    model = fit(rows, ridge=1e-8)
    values, supported, _ = model.predict(np.asarray([row["features"] for row in rows]),
                                         np.full(len(rows), 20, dtype=np.int64))
    np.testing.assert_allclose(values, [row["cost"] for row in rows], atol=2e-6)
    assert supported.all()
    assert model.metrics["training_pair_difference_rows"] == 32
    assert model.metrics["training_pair_count"] == 32
    assert model.metrics["holdout_pair_count"] == 16
    assert model.metrics["holdout_pair_rank_accuracy"] == 1.
    assert model.metrics["holdout_pair_normalized_difference_rmse"] < 1e-8
    assert model.metrics["fit_parameter_count"] == 19
    assert model.metrics["pair_difference_weight"] == 1.
    assert model.metrics["calibration_eligible"]


def test_pair_join_requires_same_root_pair_replica_and_remaining():
    rows = records(replicates=1)
    # Candidates/baselines with different replica, pair or remaining are not contrasts.
    for row in rows:
        if row["side"] == "candidate":
            if row["pair_id"] == "pair-0":
                row["replicate"] = 1
            elif row["pair_id"] == "pair-1":
                row["pair_id"] = "other-pair"
            elif row["pair_id"] == "pair-2":
                row["remaining"] = 21
    model = fit(rows)
    assert model.metrics["training_pair_count"] == 4
    assert model.metrics["holdout_pair_count"] == 2
    # Sharing an ID does not pair endpoints across collection roots.
    rows = records(replicates=1)
    rows = [r for r in rows if (r["side"] == "baseline") == (r["group"] % 2 == 0)]
    model = fit(rows)
    assert model.metrics["training_pair_count"] == model.metrics["holdout_pair_count"] == 0


def test_exact_duplicates_do_not_reweight_fit_but_new_replicas_remain():
    rows = records()
    original = fit(rows)
    duplicate = fit(rows + copy.deepcopy(rows[:7]))
    np.testing.assert_array_equal(original.weights, duplicate.weights)
    assert duplicate.metrics["exact_duplicate_count"] == 7
    assert duplicate.metrics["deduplicated_record_count"] == len(rows)
    assert duplicate.metrics["training_count"] == 64  # Both real replicas retained.
    assert duplicate.metrics["training_pair_count"] == 32


def test_identical_endpoint_contrast_does_not_add_zero_design_rows():
    model = fit(records(constant=True))
    assert model.metrics["training_pair_count"] == 32
    assert model.metrics["training_pair_difference_rows"] == 0
    prediction, supported, distances = model.predict(np.zeros((2, 17)), np.array([20, 0]))
    np.testing.assert_allclose(prediction, [40., 0.], atol=1e-10)
    assert supported.all()
    np.testing.assert_array_equal(distances, [0., 0.])
    assert model.metrics["holdout_pair_rank_accuracy"] is None


def test_ambiguous_same_replica_side_rejected_instead_of_accidentally_joined():
    rows = records()
    conflicting = copy.deepcopy(rows[0])
    conflicting["cost"] += 1.
    with pytest.raises(ValueError, match="Ambiguous"):
        fit(rows + [conflicting])


def test_mean_baseline_and_cap_are_training_only_with_strict_json_metrics():
    rows = records(constant=True)
    for row in rows:
        if row["group"] >= 4:
            row["cost"] = 100.
    model = fit(rows)
    assert model.metrics["normalized_training_mean"] == pytest.approx(.4)
    assert model.metrics["holdout_training_mean_normalized_rmse"] == pytest.approx(.6)
    assert model.metrics["holdout_knn_normalized_rmse"] == pytest.approx(.6)
    assert model.metrics["holdout_normalized_bias"] == pytest.approx(-.6)
    assert model.output_cap == 1.
    assert not model.metrics["calibration_eligible"]
    json.dumps(model.metrics, allow_nan=False)


def test_support_and_output_cap_do_not_turn_extrapolation_into_evidence():
    model = fit(remaining_tolerance=2)
    features = np.zeros((4, 17))
    features[1, 0] = features[3, 0] = 1e12
    values, supported, distances = model.predict(features, np.array([20, 20, 80, 0]))
    assert supported.tolist() == [True, False, False, True]
    assert distances[1] > model.support_threshold and np.isinf(distances[2])
    assert values[3] == distances[3] == 0.
    assert np.isfinite(values).all()
    assert np.all((values >= 0) & (values <= model.output_cap * 5 * np.array([20, 20, 80, 0])))


def test_unsupported_holdout_reports_null_distance_and_finite_json():
    rows = records()
    for row in rows:
        if row["group"] >= 4:
            row["remaining"] = 80
    model = fit(rows, remaining_tolerance=2)
    assert model.metrics["holdout_distance_mean"] is None
    assert model.metrics["holdout_unsupported_count"] == 32
    assert not model.metrics["calibration_eligible"]
    json.dumps(model.metrics, allow_nan=False)


def test_packet_roundtrip_owns_numpy_arrays_and_preserves_random_stream():
    before = copy.deepcopy(np.random.get_state())
    model = fit()
    packet = model.packet()
    restored = PairedRidgeValue.from_packet(packet)
    features, remaining = np.zeros((2, 17)), np.array([20, 80])
    for left, right in zip(model.predict(features, remaining), restored.predict(features, remaining)):
        np.testing.assert_array_equal(left, right)
    packet["mean"][:] = 42.
    packet["metrics"]["training_groups"].append(99)
    assert not np.all(model.mean == 42.) and not np.all(restored.mean == 42.)
    assert 99 not in model.metrics["training_groups"] and 99 not in restored.metrics["training_groups"]
    assert all(np.array_equal(left, right) for left, right in zip(before, np.random.get_state()))


@pytest.mark.parametrize("gamma", [0., .999, 1.])
def test_normalization_and_exact_finished_episode(gamma):
    rows = records(constant=True)
    model = PairedRidgeValue.fit(rows, total_steps=100, gamma=gamma)
    values, supported, distances = model.predict(np.zeros((1, 17)), np.array([0]))
    assert values.item() == distances.item() == 0. and supported.item()
    json.dumps(model.metrics, allow_nan=False)


@pytest.mark.parametrize("kind", ["few_groups", "negative", "nan", "fractional_remaining", "wrong_features"])
def test_invalid_or_ungrounded_records_rejected(kind):
    rows = records()
    if kind == "few_groups":
        rows = [row for row in rows if row["group"] < 3]
    elif kind == "negative":
        rows[0]["cost"] = -1.
    elif kind == "nan":
        rows[0]["features"][0] = np.nan
    elif kind == "fractional_remaining":
        rows[0]["remaining"] = 20.5
    else:
        rows[0]["features"] = [0.] * 16
    with pytest.raises(ValueError):
        fit(rows)


def test_extreme_finite_held_costs_cannot_leak_and_metrics_remain_serializable():
    rows = records(constant=True)
    for row in rows:
        if row["group"] >= 4:
            row["cost"] = 1e300
    model = fit(rows)
    assert model.metrics["holdout_raw_rmse"] == pytest.approx(1e300)
    assert not model.metrics["calibration_eligible"]
    json.dumps(model.metrics, allow_nan=False)
