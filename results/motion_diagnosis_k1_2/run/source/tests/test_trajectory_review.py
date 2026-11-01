"""Synthetic-only checks for the saved K1.1 descriptive reductions and joins."""

import builtins
import csv
import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from src.gps.mppi_gps.trajectory_review import (
    CAPSULE_ROLES,
    analyze,
    group_summary,
    trajectory_summary,
)


def metric(profile, rows=1000):
    marginal = dict(
        mean_abs=0.5,
        rms=0.6,
        saturation_fraction=0.1,
        mean_per_actuator=[0.2] * 6,
        mean_abs_per_actuator=[0.5] * 6,
        saturation_fraction_per_actuator=[0.1] * 6,
    )
    bins = [
        dict(
            start_control=100 * i,
            stop_control_exclusive=min(100 * (i + 1), rows),
            rows=min(100, rows - 100 * i),
            mse=value,
            per_actuator_mse=[value] * 6,
        )
        for i, value in enumerate(profile)
    ]
    error = sum(part["rows"] * part["mse"] for part in bins) / rows
    return dict(
        rows=rows,
        mse=error,
        per_actuator_mse=[error] * 6,
        early=dict(
            requested_controls=100,
            rows=min(100, rows),
            mse=profile[0],
            per_actuator_mse=[profile[0]] * 6,
        ),
        bins=bins,
        saturation_threshold=0.95,
        actions=deepcopy(marginal),
        predictions=deepcopy(marginal),
    )


def summary(profile, identity="first", rows=1000, stage="A", source="teacher"):
    return trajectory_summary(
        stage,
        dict(id=identity, group=source, steps=rows, ancestry="reset-1"),
        metric(profile, rows),
        0,
        dict(complete=True, success=rows == 1000),
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reseal(paths):
    files = {}
    for role, name in CAPSULE_ROLES.items():
        raw = paths[role].read_bytes()
        files[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    paths["capsule"].write_text(json.dumps(dict(files=files)))


@pytest.fixture
def saved_inputs(tmp_path):
    """Four tiny metadata trajectories; never model, raw-array or physics files."""
    paths = {role: tmp_path / (role + ".json") for role in ("capsule", "spread", *CAPSULE_ROLES)}
    profiles = [([0.5, 1.5] * 5, 1000), ([1.5, 0.5] * 5, 1000), ([0.0] * 10, 1000), ([2.0], 8)]
    original = dict(
        policies={"reference": dict(policy_hash="actor", file_sha256="checkpoint")},
        policy_order=["reference"],
        metrics=dict(early_controls=100, bin_controls=100, saturation_threshold=0.95),
    )
    result = dict(
        execution_complete=True,
        policy_hashes={"reference": "actor"},
        stages={
            "A": dict(trajectories=[]),
            "B": dict(trajectories=[], episodes=[], reference_qualified=True),
        },
    )
    spread = dict(
        fixed_reference=dict(
            qualified=True,
            policy_hash="actor",
            checkpoint_sha256="checkpoint",
            action_scale=[1.0] * 6,
            action_bias=[0.0] * 6,
        ),
        trajectories_in_original_order=[],
    )
    tables = {role: [] for role in ("matrix", "actuators", "saturation", "curves", "groups")}
    members = {}
    for index, (profile, count) in enumerate(profiles):
        stage, source = (
            ("A", "teacher") if index < 2 else ("B", "reference" if index == 2 else "random")
        )
        identity, ancestry = "trajectory-" + str(index), "reset-" + str(index)
        trajectory = dict(id=identity, group=source, steps=count, ancestry=ancestry)
        outcome = dict(complete=True, success=count == 1000, survived=count == 1000, cost=1.0)
        if stage == "A":
            trajectory["outcome"] = outcome
        else:
            result["stages"]["B"]["episodes"].append(dict(tag=identity, **outcome))
        value = metric(profile, count)
        raw = dict(arrays=dict(predictions=dict(shape=[count, 1, 6])))
        result["stages"][stage]["trajectories"].append(
            dict(trajectory=trajectory, policies={"reference": value}, raw=raw)
        )
        spread["trajectories_in_original_order"].append(
            dict(
                stage=stage,
                source=source,
                trajectory=identity,
                ancestry=ancestry,
                saved_control_bins=value["bins"],
                first100_saved=value["early"],
                prediction_raw=raw,
                E_saved_mse=value["mse"],
                executed_rows_N=count,
                D_algebraic_6NE=6 * count * value["mse"],
                outcome=outcome,
            )
        )
        base = dict(
            stage=stage,
            group=source,
            trajectory=identity,
            ancestry=ancestry,
            policy="reference",
            label_condition="recorded",
            rows=count,
        )
        tables["matrix"].append(
            dict(**base, mse=value["mse"], early_mse=profile[0], early_rows=min(100, count))
        )
        for actuator in range(6):
            tables["actuators"].append(
                dict(**base, actuator=actuator, mse=value["mse"], early_mse=profile[0])
            )
        for owner in ("actions", "predictions"):
            tables["saturation"].append(
                dict(
                    **base,
                    owner=owner,
                    mean_abs=0.5,
                    rms=0.6,
                    saturation_fraction=0.1,
                    saturation_threshold=0.95,
                )
            )
        members.setdefault((stage, source), []).append((ancestry, value))
    for (stage, source), values in members.items():
        base = dict(
            stage=stage,
            group=source,
            policy="reference",
            label_condition="recorded",
            trajectories=len(values),
            ancestries=len(values),
        )
        means = [value["mse"] for _, value in values]
        lengths = [value["rows"] for _, value in values]
        tables["groups"].append(
            dict(
                **base,
                rows=sum(lengths),
                equal_trajectory_mean=np.mean(means),
                sample_weighted_mean=np.average(means, weights=lengths),
            )
        )
        for b in range(len(values[0][1]["bins"])):
            bins = [value["bins"][b] for _, value in values]
            tables["curves"].append(
                dict(
                    **base,
                    start_control=100 * b,
                    stop_control_exclusive=100 * (b + 1),
                    rows=sum(part["rows"] for part in bins),
                    equal_trajectory_mean=np.mean([part["mse"] for part in bins]),
                    sample_weighted_mean=np.average(
                        [part["mse"] for part in bins], weights=[part["rows"] for part in bins]
                    ),
                )
            )
    for role, value in (("config", original), ("result", result), ("spread", spread)):
        paths[role].write_text(json.dumps(value))
    for role, rows in tables.items():
        write_csv(paths[role], rows)
    reseal(paths)
    config = dict(
        reference_policy="reference",
        reference_policy_hash="actor",
        reference_checkpoint_sha256="checkpoint",
        expected=dict(trajectories=4, rows=3008, bins=31, actuator_entries=24),
    )
    return paths, config


def test_equal_full_means_can_have_different_temporal_profiles():
    first = summary([0.5, 1.5] * 5)
    second = summary([1.5, 0.5] * 5, "second")
    group = group_summary([first, second], "teacher")
    decomposition = group["decomposition"]
    assert first["E"] == second["E"] == 1
    assert decomposition["var_between_full"] == 0
    assert decomposition["mean_between_bin_var"] == pytest.approx(0.25)
    assert decomposition["covariance_diag_contribution"] == pytest.approx(0.025)
    assert decomposition["covariance_offdiag_contribution"] == pytest.approx(-0.025)
    assert [b["delta_E"] for b in group["two_trajectory_difference"]["bin_deltas"]] == [-1, 1] * 5
    assert first["bins"][0]["cumulative_E"] == 0.5
    assert first["bins"][-1]["cumulative_E"] == 1


def test_common_temporal_reversal_preserves_decomposition_but_changes_order():
    profiles = [[0.1 * b for b in range(10)], [0.9 - 0.03 * b for b in range(10)], [0.4] * 10]
    forward = [summary(p, str(i)) for i, p in enumerate(profiles)]
    reverse = [summary(p[::-1], str(i)) for i, p in enumerate(profiles)]
    left, right = [group_summary(rows, "teacher")["decomposition"] for rows in (forward, reverse)]
    for name in (
        "var_between_full",
        "mean_between_bin_var",
        "covariance_diag_contribution",
        "covariance_offdiag_contribution",
    ):
        assert left[name] == pytest.approx(right[name], abs=1e-12)
    np.testing.assert_allclose(
        np.array(left["covariance_matrix"])[::-1, ::-1], right["covariance_matrix"]
    )
    assert forward[0]["bins"][0]["E"] != reverse[0]["bins"][0]["E"]
    assert forward[0]["bin_sd"] == pytest.approx(reverse[0]["bin_sd"])


@pytest.mark.parametrize("mutation", ["missing", "reordered", "gapped", "overlap", "extra"])
def test_rejects_invalid_ordered_bins(mutation):
    value = metric([0.5] * 10)
    if mutation == "missing":
        value["bins"].pop()
    elif mutation == "reordered":
        value["bins"][0], value["bins"][1] = value["bins"][1], value["bins"][0]
    elif mutation == "gapped":
        value["bins"][1]["start_control"] += 1
    elif mutation == "overlap":
        value["bins"][0]["stop_control_exclusive"] += 1
    else:
        value["bins"].append(deepcopy(value["bins"][-1]))
    with pytest.raises(ValueError, match="bin"):
        trajectory_summary(
            "A", dict(id="x", group="teacher", steps=1000, ancestry="reset"), value, 0, {}
        )


def test_zero_self_and_partial_random_are_not_padded():
    self_control = summary([0.0] * 10)
    partial = summary([2.0], rows=8)
    assert self_control["actuator_error_shares"] is None
    assert self_control["D"] == 0
    assert group_summary([self_control], "self")["decomposition"]["var_between_full"] == 0
    assert partial["D"] == 96
    assert partial["bins"][0]["stop_control_exclusive"] == 8
    assert len(partial["bins"]) == 1
    assert group_summary([partial], "random")["decomposition"] is None


def test_actuator_shares_describe_errors_not_marginal_saturation():
    value = metric([1.0] * 10)
    value["per_actuator_mse"] = [6.0, 0, 0, 0, 0, 0]
    value["early"]["per_actuator_mse"] = value["per_actuator_mse"]
    for part in value["bins"]:
        part["per_actuator_mse"] = value["per_actuator_mse"]
    row = trajectory_summary(
        "A", dict(id="x", group="teacher", steps=1000, ancestry="reset"), value, 0, {}
    )
    assert row["actuator_error_shares"] == [1.0, 0, 0, 0, 0, 0]
    assert row["marginals"]["actions"]["saturation_fraction_per_actuator"] == [0.1] * 6


def test_saved_inputs_complete_lifecycle_without_forbidden_imports(saved_inputs, monkeypatch):
    paths, config = saved_inputs
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"h5py", "torch", "mujoco"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = analyze(paths, config)
    assert result["schema"] == "walker-trajectory-review-result-v1"
    assert result["totals"] == config["expected"]
    assert [row["id"] for row in result["trajectories"]] == [
        "trajectory-" + str(i) for i in range(4)
    ]
    assert result["groups"][-1]["source"] == "pooled_MPPI_2"
    assert result["groups"][-1]["descriptive_mixture"]
    assert result["analysis_counts"] == dict(
        **config["expected"],
        matrix_joins=4,
        actuator_joins=24,
        saturation_joins=8,
        curve_joins=21,
        group_joins=3,
    )
    assert result["trajectories"][-1]["N"] == 8
    assert result["trajectories"][-1]["bins"][0]["stop_control_exclusive"] == 8
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "role,field",
    [
        ("matrix", "trajectory"),
        ("actuators", "actuator"),
        ("saturation", "owner"),
        ("curves", "group"),
        ("groups", "group"),
    ],
)
def test_rejects_source_and_csv_join_drift(saved_inputs, role, field):
    paths, config = saved_inputs
    with paths[role].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][field] = "wrong-owner"
    write_csv(paths[role], rows)
    reseal(paths)
    with pytest.raises(ValueError, match="join|roster|counts"):
        analyze(paths, config)


def test_rejects_spread_reordering_and_source_hash_drift(saved_inputs):
    paths, config = saved_inputs
    spread = json.loads(paths["spread"].read_text())
    spread["trajectories_in_original_order"].reverse()
    paths["spread"].write_text(json.dumps(spread))
    with pytest.raises(ValueError, match="order"):
        analyze(paths, config)
    paths["config"].write_text(paths["config"].read_text() + " ")
    with pytest.raises(ValueError, match="capsule binding"):
        analyze(paths, config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_policy_hash", "other"),
        ("reference_checkpoint_sha256", "other"),
        ("bin_controls", 50),
        ("expected", {}),
    ],
)
def test_rejects_changed_frozen_contract(saved_inputs, field, value):
    paths, config = saved_inputs
    with pytest.raises(ValueError):
        analyze(paths, dict(config, **{field: value}))
