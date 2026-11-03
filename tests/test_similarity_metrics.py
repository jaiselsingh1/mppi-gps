"""Synthetic calibration and leakage guards, never Walker research executions."""

import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from src.gps.mppi_gps import similarity_metrics as metrics


def trace(controls=1000, *, fallen=False):
    states = np.zeros((controls + 1, 19))
    states[:, 0] = np.arange(controls + 1) * 0.008
    states[:, 1] = states[:, 0] * 1.5
    states[:, 2] = 1.1
    states[:, 10] = 1.5
    done = np.zeros(controls, dtype=bool)
    done[-1] = fallen
    return dict(
        states=states,
        done=done,
        actions=np.zeros((controls, 6)),
        observations=np.zeros((controls, 17)),
    )


def cases(*, all_failure=False):
    records = []
    for split, prefix in [("development", "dev"), ("protected", "test")]:
        for group in range(8):
            for learner in range(2):
                motion = 0.1 + 0.3 * group + 0.04 * learner
                signal = float(np.log1p(motion))
                records.append(
                    dict(
                        case_id=f"{prefix}{group}/{learner}/mppi",
                        split=split,
                        group=f"{prefix}{group}",
                        learner=learner,
                        source="mppi",
                        scores=dict(A=1.0, B=1.0, C=signal, D=1.0, E=1.0),
                        post_motion=motion,
                        walking_successes=0 if all_failure or group < 4 else 16,
                        walking_trials=16,
                        reproduced=group < 4,
                    )
                )
    return records


def test_import_and_calibration_do_not_import_policy_or_simulator():
    script = """import sys
from src.gps.mppi_gps.similarity_metrics import calibration_suite
assert calibration_suite()['passed']
assert not any(x in sys.modules for x in ['torch','mujoco','src.gps.mppi_gps.policy'])
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_calibration_suite_and_rejected_score_identities():
    result = metrics.calibration_suite()
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert set(metrics.score_definitions()["rejected_candidates"]) == {
        "orthonormal_fourier_mse",
        "action_law_to_point_w2",
        "explained_only",
        "conditional_mean_lambda1",
    }
    assert result["contradictory_observation_components"]["score"] == 2
    assert result["nonlinear_exact_copy_action_mse"] == 0
    assert result["nonlinear_exact_copy_C"]["score"] > 0
    assert result["nonlinear_exact_copy_C"]["score"] == pytest.approx(
        3 * result["nonlinear_exact_copy_C"]["residual_mse"]
    )
    assert result["shuffled_label_prediction_mse"] > 10 * result["synthetic_prediction_mse"]


def test_smoothing_uses_only_valid_causal_windows():
    a = np.zeros((7, 6))
    a[0] = 5
    expected = (1 + 0 + 0) / 3
    assert metrics.smoothed_action_mse(a, np.zeros_like(a)) == pytest.approx(expected)
    assert metrics.smoothed_action_mse(a, np.zeros_like(a), window=1) == metrics.action_mse(
        a, np.zeros_like(a)
    )
    with pytest.raises(ValueError, match="window"):
        metrics.smoothed_action_mse(a[:4], a[:4])


def test_motion_copy_progress_origin_and_early_fall_accounting():
    teacher = trace()
    same = copy.deepcopy(teacher)
    same["states"][:, 1] += 200
    assert metrics.motion_metrics(teacher, same)["error"] == pytest.approx(0, abs=1e-12)
    assert metrics.motion_metrics(teacher, same)["reproduced"]
    short = trace(250, fallen=True)
    result = metrics.motion_metrics(teacher, short)
    assert result["common_controls"] == 250
    assert result["error"] == pytest.approx(np.sqrt(4 * 0.75))
    assert result["missing_penalty_inside_sqrt"] == 3
    assert result["final_relative_progress_error"] is None
    assert not result["reproduced"]


def test_motion_cap_binary_pitch_threshold_and_no_time_warp():
    teacher, changed = trace(), trace()
    changed["states"][1:, 3] = 0.8
    result = metrics.motion_metrics(teacher, changed)
    assert result["joint_angle_rms"] == pytest.approx(0.8 / np.sqrt(7))
    assert result["actuated_angle_rms"] == 0
    assert not result["reproduced"]
    changed["states"][:, 2] += 1000
    assert metrics.motion_metrics(teacher, changed)["error"] <= 3
    changed["states"][:, 0] *= 2
    with pytest.raises(ValueError, match="synchronized"):
        metrics.motion_metrics(teacher, changed)
    changed = trace()
    changed["done"][100] = True
    with pytest.raises(ValueError, match="after a fall"):
        metrics.motion_metrics(teacher, changed)


def test_motion_averages_components_over_fixed_horizon_not_prefix():
    teacher, short = trace(), trace(500, fallen=True)
    short["states"][1:, 2] += 0.25
    # One of four components is one on 500 observed controls, then explicit missing mass.
    result = metrics.motion_metrics(teacher, short)
    assert result["error"] == pytest.approx(np.sqrt(0.5 / 4 + 4 * 0.5))


def test_explainer_uses_actual_float32_normalization_and_protected_fit_rejected():
    rng = np.random.default_rng(42)
    observations = rng.normal(size=(100, 17))
    normalization = dict(mean=np.arange(17) / 10, std=np.arange(17) / 10 + 1)
    z = metrics.normalized_student_features(observations, normalization)
    expected = (
        (observations.astype(np.float32) - np.array(normalization["mean"], np.float32))
        / np.array(normalization["std"], np.float32)
    ).astype(np.float64)
    np.testing.assert_array_equal(z, expected)
    actions = z[:, :6] * 0.2
    fitted = metrics.fit_state_explainer(observations, actions, normalization, group_ids=["dev0"])
    assert fitted["rank"] == 18
    assert fitted["residual_mse"] < 1e-8
    result = metrics.state_explained_score(observations, actions, actions, fitted)
    assert result["score"] == pytest.approx(3 * result["residual_mse"])
    assert result["outside_development_box_fraction"] == 0
    with pytest.raises(ValueError, match="development-only"):
        metrics.fit_state_explainer(observations, actions, normalization, split="protected")


def test_occupancy_preserves_pair_order_invariance_and_missing_mass():
    rng = np.random.default_rng(5)
    observations, actions = rng.normal(size=(1000, 17)), rng.normal(size=(1000, 6))
    normalization = metrics.fit_occupancy_normalizer(np.column_stack((observations, actions)))
    order = rng.permutation(1000)
    result = metrics.sliced_occupancy_distance(
        observations, actions, observations[order], actions[order], normalization
    )
    assert result["score"] == 0
    truncated = metrics.sliced_occupancy_distance(
        observations, actions, observations[:250], actions[:250], normalization
    )
    assert truncated["cemetery_mass_penalty"] == 3
    assert truncated["score"] >= 3
    with pytest.raises(ValueError, match="development-only"):
        metrics.fit_occupancy_normalizer(
            np.column_stack((observations, actions)), split="protected"
        )


def test_gain_fit_recovers_affine_feedback_and_flags_rank_deficiency():
    rng = np.random.default_rng(10)
    x = np.vstack((np.eye(17), -np.eye(17)))
    gain = rng.normal(size=(6, 17))
    held = rng.normal(size=(8, 17))
    fitted = metrics.fit_local_feedback(
        x,
        x @ gain.T,
        held_observations=held,
        held_actions=held @ gain.T,
        contact_changed=np.array([False, True] * 17),
    )
    np.testing.assert_allclose(fitted["gain"], gain, rtol=0, atol=1e-12)
    assert fitted["held_rmse"] < 1e-12
    assert fitted["rank"] == 17 and fitted["uniquely_identified"]
    assert fitted["contact_regime_changes"] == 17
    deficient = x.copy()
    deficient[:, -1] = 0
    result = metrics.fit_local_feedback(deficient, deficient @ gain.T)
    assert result["rank"] == 16
    assert result["condition_number"] is None
    assert not result["uniquely_identified"]
    assert not result["optimal_gain_claim"]


def test_contact_missing_differs_from_absence_and_raw_flicker_is_retained():
    with pytest.raises(ValueError, match="missing sensors"):
        metrics.stride_segments(np.empty((10, 0), dtype=bool))
    assert metrics.stride_segments(np.zeros(10, dtype=bool))["onsets"] == []
    result = metrics.stride_segments(np.array([True, True, False, True, False, True]))
    assert result["onsets"] == [3, 5]
    assert not result["intervals"][0]["complete"]
    assert result["intervals"][1]["complete"]
    assert not result["intervals"][-1]["complete"]
    assert metrics.stride_segments(np.ones(2, dtype=bool), initial_contact=False)["onsets"] == [0]


def test_all_failure_walking_is_finite_constant_and_unidentified():
    records = cases(all_failure=True)
    models = metrics.fit_prediction_models(records)
    expected = 0.5 / (16 * 16 + 1)
    assert models["walking_degenerate"]
    for model in models["models"].values():
        assert model["walking_coefficients"][1] == 0
        assert model["walking_constant_probability"] == pytest.approx(expected)
    result = metrics.evaluate_predictions(records, models)
    assert not result["walking_identified"]
    assert not result["objective_passed"]
    assert all(not r["passed"] for r in result["score_table"])
    json.dumps(result, allow_nan=False)


def test_predictive_models_have_equal_capacity_and_bootstrap_preserves_groups():
    records = cases()
    models = metrics.fit_prediction_models(records)
    assert all(
        len(m["motion_coefficients"]) == len(m["walking_coefficients"]) == 2
        for m in models["models"].values()
    )
    result = metrics.evaluate_predictions(records, models)
    assert result["motion_identified"] and result["walking_identified"]
    assert result["objective_passed"]
    table = {r["candidate"]: r for r in result["score_table"]}
    assert table["C"]["passed"]
    assert not table["B"]["passed"]  # Exactly the same information as baseline A.
    indices = np.array(result["bootstrap"]["group_indices"], dtype=np.int64)
    assert indices.shape == (9999, 8)
    assert indices.min() == 0 and indices.max() == 7
    assert (
        hashlib.sha256(indices.tobytes()).hexdigest() == result["bootstrap"]["group_indices_sha256"]
    )
    again = metrics.evaluate_predictions(records, models)
    assert again["bootstrap"]["group_indices_sha256"] == result["bootstrap"]["group_indices_sha256"]


def test_proper_brier_includes_bernoulli_variance_and_missing_binary_stays_missing():
    records = cases()
    for r in records:
        r["walking_successes"] = 8
        r.pop("reproduced")
    result = metrics.evaluate_predictions(records, metrics.fit_prediction_models(records))
    assert all(row["walking_brier"] == pytest.approx(0.25) for row in result["score_table"])
    assert not result["binary_reproduction_identified"]
    assert all(row["reproduction_auc"] is None for row in result["score_table"])


def test_protected_group_leakage_duplicates_incomplete_roster_and_variance_gates():
    records = cases()
    models = metrics.fit_prediction_models(records)
    overlap = copy.deepcopy(records)
    for r in overlap:
        if r["split"] == "protected":
            r["group"] = r["group"].replace("test", "dev")
    with pytest.raises(ValueError, match="disjoint"):
        metrics.evaluate_predictions(overlap, models)
    with pytest.raises(ValueError, match="Duplicated"):
        metrics.fit_prediction_models(records + [records[0]])
    incomplete = [r for r in records if r["group"] != "test7"]
    result = metrics.evaluate_predictions(incomplete, models)
    assert not result["motion_identified"] and not result["walking_identified"]
    low_variation = copy.deepcopy(records)
    for r in low_variation:
        r["post_motion"] = 0.2
    result = metrics.evaluate_predictions(
        low_variation, metrics.fit_prediction_models(low_variation)
    )
    assert not result["motion_identified"]
    assert not result["objective_passed"]


def test_protected_outcomes_never_fit_prediction_coefficients():
    records = cases()
    expected = metrics.fit_prediction_models(records)
    changed = copy.deepcopy(records)
    for row in changed:
        if row["split"] == "protected":
            row["post_motion"] = 2.99
            row["walking_successes"] = 3
    assert metrics.fit_prediction_models(changed) == expected


def test_constant_development_rates_cannot_establish_walking_discrimination():
    records = cases()
    for row in records:
        if row["split"] == "development":
            row["walking_successes"] = 4
    models = metrics.fit_prediction_models(records)
    assert models["walking_degenerate"]
    assert all(m["walking_constant_probability"] == 0.25 for m in models["models"].values())
    assert not metrics.evaluate_predictions(records, models)["walking_identified"]


@pytest.mark.parametrize("option", [dict(seed=1), dict(draws=20)])
def test_protected_bootstrap_cannot_silently_change_protocol(option):
    records = cases()
    with pytest.raises(ValueError, match="frozen"):
        metrics.evaluate_predictions(records, metrics.fit_prediction_models(records), **option)
