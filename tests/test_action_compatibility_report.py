"""K1 saved-report tests use known arrays, never roster policies or physics."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.gps.mppi_gps.compatibility_metrics import agreement_metrics
from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.reporting.action_compatibility import build_tables, render


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
