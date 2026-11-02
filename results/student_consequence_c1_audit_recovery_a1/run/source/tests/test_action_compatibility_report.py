"""K1 saved-report tests use known arrays, never roster policies or physics."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.gps.mppi_gps.compatibility_metrics import agreement_metrics
from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.reporting.action_compatibility import (
    build_tables,
    collision_training_scope,
    comparison_matrix,
    render,
)


def synthetic_result():
    names = ["untrained_0", "h4_parent", "reference"]
    config = dict(
        policy_order=names,
        bootstrap_seed=3,
        bootstrap_draws=20,
        metrics=dict(bin_controls=2),
        policies={
            name: dict(feature="clipped", architecture="synthetic fixture") for name in names
        },
        missing_evidence=["Synthetic test fixture has no scientific evidence."],
    )
    trajectories = []
    for index, (n, target_value) in enumerate(((2, 0.2), (4, 0.6))):
        labels = np.full((n, 6), target_value)
        policies = {}
        for name in names:
            # These are known synthetic prediction arrays, not neural forwards.
            prediction = np.zeros_like(labels)
            value = agreement_metrics(prediction, labels, early_controls=2, bin_controls=2)
            value["synthetic_random_labels"] = agreement_metrics(
                prediction, np.zeros_like(labels), early_controls=2, bin_controls=2
            )
            value["permuted_recorded_labels"] = agreement_metrics(
                prediction, labels[::-1], early_controls=2, bin_controls=2
            )
            policies[name] = value
        trajectories.append(
            dict(
                trajectory=dict(
                    id=f"teacher_{index}",
                    group="source",
                    steps=n,
                    seed=10,
                    ancestry=dict(reset_cluster="shared"),
                    membership={
                        "h4_parent": "fitted_original_discovery"
                        if index == 0
                        else "trajectory_held_out_same_resets_as_fitted_guided"
                    },
                ),
                policies=policies,
            )
        )
    trace = dict(
        trajectory=dict(
            id="old_parent",
            policy="h4_parent",
            steps=5,
            seed=2,
            ancestry=dict(reset_cluster="old"),
            outcome=dict(complete=True, success=False, survived=False, cost=4999, mae=1.0),
        ),
        diagnostics=dict(
            rows=5,
            actions_mean_abs=0.1,
            saturation_fraction=0,
            clipped_velocity_rows=0,
            scope="Existing trace only",
        ),
    )
    outcomes = {}
    for name in names:
        outcomes[name] = dict(
            episodes=2,
            successes=0,
            survived=0,
            mean_cost=4500,
            mean_steps=12,
            costs=[4400, 4600],
            steps=[10, 14],
        )
    result = dict(
        execution_complete=True,
        error=None,
        work={},
        wall_seconds=1,
        stages=dict(
            A=dict(trajectories=trajectories, retained_datasets={}, historical_standalone=[trace]),
            B=dict(
                primary=outcomes,
                reference_qualified=False,
                reference_status="unqualified_candidate",
                confirmation=None,
                random_collection_triggered=False,
                episodes=[],
                trajectories=[],
            ),
        ),
    )
    return result, config


def test_tables_keep_label_controls_weights_exposure_and_old_failure_distinct():
    result, config = synthetic_result()
    tables = build_tables(result, config)
    assert len(tables["matrix"]) == 2 * 3 * 3
    group = next(
        row
        for row in tables["groups"]
        if row["policy"] == "h4_parent" and row["label_condition"] == "recorded"
    )
    assert group["equal_trajectory_mean"] == pytest.approx(0.2)
    assert group["sample_weighted_mean"] == pytest.approx((2 * 0.04 + 4 * 0.36) / 6)
    assert group["ancestries"] == 1
    assert group["equal_low"] == pytest.approx(group["equal_high"])
    random = next(
        row
        for row in tables["groups"]
        if row["policy"] == "h4_parent" and row["label_condition"] == "synthetic_random"
    )
    assert random["equal_trajectory_mean"] == 0  # No preferred metric ordering imposed.
    later_bin = next(row for row in tables["curves"] if row["start_control"] == 2)
    assert later_bin["trajectories"] == 1
    assert later_bin["rows"] == 2
    assert {row["exposure"] for row in tables["exposure"]} == {
        "fitted",
        "trajectory_held_out_shared_reset",
    }
    parent = next(row for row in tables["reconstruction"] if row["policy"] == "h4_parent")
    assert parent["fitted_teacher_mse"] == pytest.approx(0.04)
    assert parent["shared_reset_held_trajectory_mse"] == pytest.approx(0.36)
    assert parent["historical_walking"] == 0
    assert "not expert labels" in tables["historical"][0]["ground_truth_action_error"]
    json.dumps(tables, allow_nan=False)


def test_saved_render_is_create_only_and_binds_outputs(tmp_path):
    result, config = synthetic_result()
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(config))
    result["protocol_sha256"] = sha(run / "config.json")
    (run / "result.json").write_text(json.dumps(result))
    (run / "inputs/inventory.json").write_text("{}")
    out = tmp_path / "report"
    receipt = render(run, out)
    assert receipt["new_forward_rows"] == receipt["new_controls"] == 0
    assert receipt["schema"] == "walker-action-compatibility-report-v2"
    assert receipt["scientific_metric_changed"] is False
    assert receipt["result_sha256"] == sha(run / "result.json")
    for name in ("agreement_heatmaps", "source_control_curves", "standalone_outcomes"):
        for suffix in ("png", "pdf"):
            assert (out / f"{name}.{suffix}").stat().st_size > 0
    text = (out / "report.md").read_text()
    assert "unqualified or unevaluated candidate" in text
    assert "does not prospectively test the third" in text
    assert "not expert labels off the teacher path" in text
    with (out / "trajectory_policy_matrix.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 18
    assert all(sha(out / name) == binding["sha256"] for name, binding in receipt["outputs"].items())
    with pytest.raises(FileExistsError):
        render(run, out)
    (run / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration binding"):
        render(run, tmp_path / "tampered")


def test_editorial_matrix_preserves_equal_trajectory_means_and_fixed_clone():
    def row(group, values, count):
        names = ["untrained_0", "untrained_1", "untrained_2", "h4_independent", "reference"]
        return dict(
            trajectory=dict(group=group, steps=count),
            policies={name: {"mse": value} for name, value in zip(names, values, strict=True)},
        )

    result = dict(
        stages={
            "A": {
                "trajectories": [
                    row("a", [0, 1, 2, 0.4, 0.7], 1000),
                    row("b", [1, 2, 3, 0.6, 0.8], 100),
                ]
            },
            "B": {
                "reference_qualified": True,
                "trajectories": [
                    row("qualified_reference", [0.2, 0.4, 0.6, 0.8, 0], 1000),
                    row("executed_random_actions", [0.1, 0.3, 0.5, 0.7, 0.9], 8),
                ],
            },
        }
    )
    matrix = comparison_matrix(result)
    assert matrix[0]["mean_three_untrained"] == 1.5
    assert matrix[0]["h4_independent_clone"] == 0.5
    assert matrix[0]["independent_reference"] == 0.75
    assert matrix[0]["executed_rows"] == 1100
    assert matrix[1]["mean_three_untrained"] == pytest.approx(0.4)
    assert matrix[1]["independent_reference"] == 0
    assert matrix[2]["mean_three_untrained"] == pytest.approx(0.3)
    assert matrix[2]["executed_rows"] == 8
    result["stages"]["B"]["reference_qualified"] = False
    assert comparison_matrix(result)[1]["trajectories"] == 0


def test_editorial_collision_scope_keeps_actual_fitted_corpora_separate():
    result = dict(
        stages={
            "A": {
                "trajectories": [
                    {
                        "trajectory": {
                            "steps": 2,
                            "group": "independent",
                            "membership": {"clone": "fitted_new_acquisition"},
                        }
                    },
                    {
                        "trajectory": {
                            "steps": 2,
                            "group": "guided",
                            "membership": {
                                "clone": "trajectory_held_out_same_resets_as_fitted_h4_arm"
                            },
                        }
                    },
                ],
                "neighborhoods": {
                    "exact_incoming_h4_effective_input_collisions": {
                        "details": [{"row_indices": [0, 2]}]
                    }
                },
                "retained_datasets": {"auxiliary": {"trajectory": {"steps": 96}}},
            }
        }
    )
    scope = collision_training_scope(result)
    assert scope["pooled_teacher_trajectories"] == 2
    assert scope["cross_acquisition_source_groups"] == 1
    assert scope["maximum_fitted_teacher_rows_per_collision_per_actual_policy"] == 1
    assert scope["auxiliary_rows_outside_geometry"] == 96
