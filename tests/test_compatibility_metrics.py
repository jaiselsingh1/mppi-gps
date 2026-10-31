"""Known saved-array cases only: no roster predictions, fitting or physics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.gps.mppi_gps.compatibility_metrics import (
    aggregate_trajectory_scores,
    agreement_metrics,
    neighborhood_summary,
    prepare_neighborhoods,
)


def test_primary_error_is_per_scalar_torque_and_retains_short_bins():
    actions = np.ones((3, 6))
    prediction = np.zeros_like(actions)
    prediction[2] = 1
    result = agreement_metrics(prediction, actions, early_controls=2, bin_controls=2)
    assert result["mse"] == pytest.approx(2 / 3)
    assert result["per_actuator_mse"] == [pytest.approx(2 / 3)] * 6
    assert result["early"]["mse"] == 1
    assert [part["rows"] for part in result["bins"]] == [2, 1]
    assert [part["mse"] for part in result["bins"]] == [1, 0]
    assert result["actions"]["saturation_fraction"] == 1
    assert result["predictions"]["saturation_fraction"] == pytest.approx(1 / 3)
    assert agreement_metrics(actions, actions)["mse"] == 0
    assert agreement_metrics(-actions, actions)["mse"] == 4


def test_metrics_reject_unbounded_or_misaligned_rows_without_clipping():
    with pytest.raises(ValueError, match="bounded torques"):
        agreement_metrics(np.full((2, 6), 1.1), np.ones((2, 6)))
    with pytest.raises(ValueError, match="same rows"):
        agreement_metrics(np.zeros((2, 6)), np.zeros((3, 6)))
    with pytest.raises(ValueError, match="finite"):
        agreement_metrics(np.full((2, 6), np.nan), np.zeros((2, 6)))
    with pytest.raises(ValueError, match="positive integer"):
        agreement_metrics(np.zeros((2, 6)), np.zeros((2, 6)), early_controls=0)
    actions = np.linspace(-1, 1, 18).reshape(3, 6)
    assert agreement_metrics(actions, actions)["mse"] == 0
    assert agreement_metrics(actions, np.roll(actions, 1, axis=0))["mse"] > 0


def test_group_means_and_bootstrap_preserve_ancestry_and_policy_pairing():
    scores = np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 4.0]])
    counts = np.array([1, 1, 8])
    result = aggregate_trajectory_scores(
        scores,
        counts,
        ["shared", "shared", "other"],
        rng=np.random.default_rng(12),
        bootstrap_draws=20,
    )
    assert result["equal_trajectory_mean"] == pytest.approx([4 / 3, 7 / 3])
    assert result["sample_weighted_mean"] == pytest.approx([2.5, 3.5])
    draws = result["replicates_equal_trajectory_mean"]
    assert set(np.round(draws[:, 0], 12)) <= {0.5, round(4 / 3, 12), 3.0}
    np.testing.assert_allclose(draws[:, 1] - draws[:, 0], 1)
    repeat = aggregate_trajectory_scores(
        scores,
        counts,
        ["shared", "shared", "other"],
        rng=np.random.default_rng(12),
        bootstrap_draws=20,
    )
    np.testing.assert_array_equal(draws, repeat["replicates_equal_trajectory_mean"])
    with pytest.raises(ValueError, match="Generator"):
        aggregate_trajectory_scores(scores, counts, ["a", "a", "b"], rng=None)


def geometry_fixture():
    # Each row changes all 17 coordinates equally, making RMS distances known.
    coordinates = np.array([0.0, 0.1, 0.2, 0.11, 0.21, 0.31, 0.09, 0.19, 0.29])
    obs = np.repeat(coordinates[:, None], 17, axis=1)
    trajectories = np.repeat(["a", "b", "c"], 3)
    ancestry = np.repeat(["shared", "shared", "other"], 3)
    steps = np.tile([0, 50, 100], 3)
    return prepare_neighborhoods(
        obs,
        trajectories,
        steps,
        ancestry,
        np.zeros(17),
        np.ones(17),
        {"anchors_per_trajectory": 3, "minimum_step_separation": 50, "distance_caliper": 0.02},
    )


def test_observation_geometry_separates_ancestry_and_matches_distance():
    geometry = geometry_fixture()
    np.testing.assert_array_equal(geometry["anchor_index"], np.arange(9))
    assert geometry["within_index"][0] == 1
    assert geometry["same_ancestry_nearest_index"][0] == 3
    assert geometry["different_ancestry_nearest_index"][0] == 6
    assert geometry["same_ancestry_matched_index"][0] == 3
    assert geometry["same_ancestry_matched_distance_residual"][0] == pytest.approx(0.01)
    assert np.all(
        geometry["same_ancestry_matched_distance_residual"][
            geometry["same_ancestry_matched_index"] >= 0
        ]
        <= 0.02
    )
    frozen = {name: value.copy() for name, value in geometry.items()}
    actions = np.repeat(np.arange(9)[:, None] / 8, 6, axis=1)
    summary = neighborhood_summary(geometry, actions)
    for name in geometry:
        np.testing.assert_array_equal(geometry[name], frozen[name])
    assert summary["comparisons"]["same_ancestry"]["matched_pairs"] > 0
    assert summary["comparisons"]["different_ancestry"]["matched_pairs"] > 0
    json.dumps(summary, allow_nan=False)


def test_no_caliper_overlap_is_missing_evidence_not_a_matched_contrast():
    obs = np.repeat(np.array([0, 0.1, 10, 10.1])[:, None], 17, axis=1)
    geometry = prepare_neighborhoods(
        obs,
        ["a", "a", "b", "b"],
        [0, 50, 0, 50],
        ["a", "a", "b", "b"],
        np.zeros(17),
        np.ones(17),
        {"distance_caliper": 0.02},
    )
    assert np.all(geometry["different_ancestry_matched_index"] == -1)
    assert np.all(geometry["different_ancestry_matched_distance"] == -1)
    summary = neighborhood_summary(geometry, np.zeros((4, 6)))
    different = summary["comparisons"]["different_ancestry"]
    assert different["match_coverage_of_within_eligible"] == 0
    assert different["paired_between_minus_within_action_mse"]["mean"] is None
    json.dumps(summary, allow_nan=False)


def test_effective_float32_collision_floor_does_not_certify_task_conflict():
    obs = np.ones((2, 17))
    obs[1, 0] = np.nextafter(1.0, 2.0)
    geometry = prepare_neighborhoods(
        obs,
        ["a", "b"],
        [0, 0],
        ["same", "same"],
        np.zeros(17),
        np.ones(17),
    )
    assert geometry["collision_group"][0] == geometry["collision_group"][1]
    action = np.array([[0] * 6, [1] * 6])
    physical = np.zeros((2, 19))
    physical[1, 10] = 0.5
    summary = neighborhood_summary(
        geometry,
        action,
        physical,
        {"rho": np.array([0.0, 0.1]), "nominal": np.zeros((2, 48, 6))},
    )
    collision = summary["exact_incoming_h4_effective_input_collisions"]
    assert collision["groups"] == 1
    assert collision["weighted_empirical_deterministic_mse_floor"] == 0.25
    item = collision["details"][0]
    assert item["exact_distinct_action_vectors"] == 2
    assert item["exact_distinct_physical_states"] == 2
    assert item["planner_context_constant"] == {"rho": False, "nominal": True}
    assert "do not establish" in collision["scope"]
    json.dumps(summary, allow_nan=False)


def test_geometry_rejects_inconsistent_or_unfrozen_inputs():
    obs = np.zeros((2, 17))
    with pytest.raises(ValueError, match="unique control indices"):
        prepare_neighborhoods(obs, ["a", "a"], [0, 0], ["s", "s"], np.zeros(17), np.ones(17))
    with pytest.raises(ValueError, match="one reset ancestry"):
        prepare_neighborhoods(obs, ["a", "a"], [0, 1], ["s", "other"], np.zeros(17), np.ones(17))
    with pytest.raises(ValueError, match="Unknown"):
        prepare_neighborhoods(
            obs,
            ["a", "b"],
            [0, 0],
            ["s", "s"],
            np.zeros(17),
            np.ones(17),
            {"choose_after_actions": True},
        )


def test_missing_planner_context_is_unknown_instead_of_evidence_of_difference():
    geometry = prepare_neighborhoods(
        np.zeros((2, 17)),
        ["a", "b"],
        [0, 0],
        ["same", "same"],
        np.zeros(17),
        np.ones(17),
    )
    summary = neighborhood_summary(
        geometry,
        np.zeros((2, 6)),
        planner_context={
            "planner_context_available": np.zeros(2, dtype=bool),
            "planner_nominal": np.full((2, 48, 6), np.nan),
            "rho": np.array([0, 0.1]),
        },
    )
    item = summary["exact_incoming_h4_effective_input_collisions"]["details"][0]
    assert item["planner_context_available_rows"] == 0
    assert item["planner_context_comparable"] is False
    assert item["planner_context_constant"]["planner_nominal"] is None
    assert item["planner_context_constant"]["rho"] is False
    json.dumps(summary, allow_nan=False)
