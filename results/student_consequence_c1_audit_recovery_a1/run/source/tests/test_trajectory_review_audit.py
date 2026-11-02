"""Independent K1.1 arithmetic fixtures; no raw arrays, inference or physics."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.gps.mppi_gps.audits.trajectory_review import (
    ReviewAudit,
    contained,
    derive_group,
    derive_trajectory,
    verify_metric,
    verify_reference_qualification,
    verify_worker_close,
)


def metric_fixture(values: list[float], n: int | None = None) -> dict:
    n = n if n is not None else 100 * len(values)
    bins = [
        dict(
            start_control=100 * index,
            stop_control_exclusive=min(100 * (index + 1), n),
            rows=min(100, n - 100 * index),
            mse=value,
            per_actuator_mse=[value] * 6,
        )
        for index, value in enumerate(values)
    ]
    score = sum(row["rows"] * row["mse"] for row in bins) / n
    marginal = dict(
        mean_abs=0.0,
        rms=0.0,
        mean_per_actuator=[0.0] * 6,
        mean_abs_per_actuator=[0.0] * 6,
        saturation_fraction=0.0,
        saturation_fraction_per_actuator=[0.0] * 6,
    )
    return dict(
        rows=n,
        mse=score,
        per_actuator_mse=[score] * 6,
        early=dict(
            requested_controls=100,
            rows=min(100, n),
            mse=values[0],
            per_actuator_mse=[values[0]] * 6,
        ),
        bins=bins,
        saturation_threshold=0.95,
        actions=deepcopy(marginal),
        predictions=deepcopy(marginal),
    )


def trajectory_fixture(
    values: list[float], *, identifier: str = "t", n: int | None = None, source: str = "teacher"
) -> tuple[dict, dict]:
    metric = metric_fixture(values, n)
    count = metric["rows"]
    record = dict(
        trajectory=dict(id=identifier, group=source, steps=count, ancestry=dict(reset_cluster="r")),
        policies=dict(reference=metric),
        raw=dict(
            arrays={
                key: dict(shape=shape, encoding="array", sha256="a" * 64)
                for key, shape in (
                    ("predictions", [count, 9, 6]),
                    ("executed_actions", [count, 6]),
                    ("observations", [count, 17]),
                )
            }
        ),
    )
    outcome = dict(complete=True, success=count == 1000, survived=count == 1000, cost=1.0)
    return record, outcome


def derive(values: list[float], **kwargs) -> dict:
    record, outcome = trajectory_fixture(values, **kwargs)
    return derive_trajectory(ReviewAudit(), record, "A", 0, outcome)


def test_saved_weighted_bins_actuators_cumulative_and_algebraic_sum():
    # A final partial bin contributes only its 25 executed controls.
    metric = metric_fixture([0.5, 2.0], n=125)
    bins = verify_metric(ReviewAudit(), metric, 125)
    assert bins[0]["cumulative_E"] == 0.5
    assert bins[1]["cumulative_E"] == pytest.approx(0.8)
    assert bins[1]["stop_control_exclusive"] == 125
    record, outcome = trajectory_fixture([0.5, 2.0], n=125)
    row = derive_trajectory(ReviewAudit(), record, "A", 0, outcome)
    assert row["D"] == 600.0 and row["E"] == pytest.approx(0.8)
    assert sum(row["actuator_error_shares"]) == pytest.approx(1.0)
    wrong = deepcopy(row)
    wrong["bins"][1]["cumulative_E"] += 0.01
    with pytest.raises(AssertionError, match="cumulative_E"):
        ReviewAudit().prove(wrong, row, "saved output")


@pytest.mark.parametrize("tamper", ["missing", "reordered", "gap", "actuator", "full", "early"])
def test_missing_reordered_bins_and_reduction_tampering_are_rejected(tamper):
    metric = metric_fixture([0.5, 2.0])
    if tamper == "missing":
        metric["bins"].pop()
    elif tamper == "reordered":
        metric["bins"].reverse()
    elif tamper == "gap":
        metric["bins"][1]["start_control"] += 1
    elif tamper == "actuator":
        metric["bins"][1]["per_actuator_mse"][0] += 0.1
    elif tamper == "full":
        metric["mse"] += 0.1
        metric["per_actuator_mse"] = [metric["mse"]] * 6
    else:
        metric["early"]["mse"] += 0.1
    with pytest.raises(AssertionError):
        verify_metric(ReviewAudit(), metric, 200)


def test_temporal_cancellation_and_population_covariance_are_distinct():
    first = derive([0.0] * 5 + [2.0] * 5, identifier="first")
    second = derive([2.0] * 5 + [0.0] * 5, identifier="second")
    group = derive_group([first, second], "pair", "A")
    decomposition = group["decomposition"]
    assert group["E_sd"] == decomposition["var_between_full"] == 0
    assert decomposition["mean_between_bin_var"] == 1
    assert decomposition["covariance_diag_contribution"] == pytest.approx(0.1)
    assert decomposition["covariance_offdiag_contribution"] == pytest.approx(-0.1)
    assert decomposition["identity_residual"] == pytest.approx(0.0)
    difference = group["two_trajectory_difference"]
    assert difference["first"] == "first" and difference["second"] == "second"
    assert difference["full_delta"] == 0
    assert [row["delta_E"] for row in difference["bin_deltas"]] == [-2] * 5 + [2] * 5
    # ddof=0 is essential: two constant profiles 1 and 3 have variance 1.
    constant = derive_group([derive([1.0] * 10), derive([3.0] * 10)], "pair", "A")
    assert constant["decomposition"]["var_between_full"] == 1
    assert constant["decomposition"]["covariance_matrix"][0][0] == 1
    wrong = deepcopy(group)
    wrong["decomposition"]["covariance_offdiag_contribution"] = 0.1
    with pytest.raises(AssertionError, match="offdiag"):
        ReviewAudit().prove(wrong, group, "covariance")


def test_self_zero_and_actual_short_random_lengths_are_not_padded():
    self_row = derive([0.0] * 10, source="qualified_reference")
    assert self_row["actuator_error_shares"] is None
    assert self_row["bin_sd"] == 0
    self_group = derive_group([self_row] * 16, "qualified_reference", "B")
    assert self_group["decomposition"]["covariance_matrix"] == [[0.0] * 10 for _ in range(10)]
    random_rows = [derive([0.8], n=n, source="executed_random_actions") for n in (8, 26, 22, 14)]
    group = derive_group(random_rows, "executed_random_actions", "B")
    assert group["N_values"] == [8, 26, 22, 14]
    assert not group["balanced"] and group["decomposition"] is None
    assert all(len(row["bins"]) == 1 and row["bin_sd"] == 0 for row in random_rows)
    record, outcome = trajectory_fixture([1.0] * 10, source="qualified_reference")
    with pytest.raises(AssertionError, match="self-match"):
        derive_trajectory(ReviewAudit(), record, "B", 0, outcome)


def test_regular_file_hashes_detect_changes_and_unsafe_sealed_paths(tmp_path):
    audit = ReviewAudit()
    path = tmp_path / "input.json"
    path.write_text("{}")
    original = audit.bind(path)
    assert audit.bind(path, original["sha256"]) == original
    path.write_text("[]")
    with pytest.raises(AssertionError, match="SHA-256"):
        audit.bind(path, original["sha256"])
    with pytest.raises(AssertionError, match="Contained"):
        contained(ReviewAudit(), tmp_path, "../escape.json")
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(AssertionError, match="symlinks"):
        contained(ReviewAudit(), tmp_path, "link.json")


def close_fixture() -> dict:
    return dict(
        schema="walker-trajectory-review-exit-v1",
        exit_code=0,
        timeout=False,
        error=None,
        execution_complete=True,
        result_sha256="a" * 64,
        child_started=True,
        child_reaped=True,
        kill_attempted=False,
        cleanup_errors=[],
        result_error=None,
    )


def test_saved_worker_close_requires_clean_child_and_bound_complete_result():
    receipt = close_fixture()
    verify_worker_close(ReviewAudit(), receipt, "a" * 64)
    with pytest.raises(AssertionError, match="result_sha256"):
        verify_worker_close(ReviewAudit(), receipt, "b" * 64)


def reference_summary() -> dict:
    return dict(episodes=16, successes=16, survived=16, steps=[1000] * 16)


def test_k1_primary_has_reference_wrapper_but_confirmation_is_direct_summary():
    stage = dict(
        primary=dict(reference=reference_summary(), another_policy=dict(episodes=16)),
        confirmation=reference_summary(),
    )
    verify_reference_qualification(ReviewAudit(), stage)
    wrapped_confirmation = deepcopy(stage)
    wrapped_confirmation["confirmation"] = dict(reference=reference_summary())
    with pytest.raises(AssertionError, match="confirmation.*directly"):
        verify_reference_qualification(ReviewAudit(), wrapped_confirmation)
    unwrapped_primary = deepcopy(stage)
    unwrapped_primary["primary"] = reference_summary()
    with pytest.raises(AssertionError, match="primary.*mapping"):
        verify_reference_qualification(ReviewAudit(), unwrapped_primary)


@pytest.mark.parametrize("cohort", ("primary", "confirmation"))
@pytest.mark.parametrize("field", ("episodes", "successes", "survived", "steps"))
def test_asymmetric_layout_correction_preserves_every_reference_competence_gate(cohort, field):
    stage = dict(primary=dict(reference=reference_summary()), confirmation=reference_summary())
    summary = stage["primary"]["reference"] if cohort == "primary" else stage["confirmation"]
    summary[field] = [1000] * 15 + [999] if field == "steps" else 15
    with pytest.raises(AssertionError, match=field):
        verify_reference_qualification(ReviewAudit(), stage)


@pytest.mark.parametrize(
    "field,value",
    [
        ("child_started", False),
        ("child_reaped", False),
        ("kill_attempted", True),
        ("cleanup_errors", [dict(phase="kill", type="OSError", message="unresolved")]),
        ("result_error", dict(type="ValueError", message="stale result", traceback="fixture")),
        ("execution_complete", False),
        ("timeout", True),
        ("exit_code", 1),
    ],
)
def test_incomplete_or_unclean_worker_closures_cannot_pass(field, value):
    receipt = close_fixture()
    receipt[field] = value
    with pytest.raises(AssertionError, match=field):
        verify_worker_close(ReviewAudit(), receipt, "a" * 64)
    missing = close_fixture()
    del missing[field]
    with pytest.raises(AssertionError, match=field):
        verify_worker_close(ReviewAudit(), missing, "a" * 64)


def test_auditor_imports_only_numpy_and_preserves_create_only_failure_receipt(tmp_path):
    root = Path(__file__).resolve().parents[1]
    program = """
from src.gps.mppi_gps.audits.trajectory_review import ReviewAudit, assert_independent_imports
assert_independent_imports(ReviewAudit())
"""
    imported = subprocess.run(
        [sys.executable, "-B", "-c", program], cwd=root, text=True, capture_output=True, timeout=30
    )
    assert imported.returncode == 0, imported.stderr
    out = tmp_path / "audit"
    command = [
        sys.executable,
        "-B",
        "scripts/audits/trajectory_review.py",
        "--run",
        str(tmp_path / "missing"),
        "--out",
        str(out),
        "--config-sha",
        "a" * 64,
        "--seal-sha",
        "b" * 64,
    ]
    failed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=30)
    assert failed.returncode == 1, failed.stderr
    receipt = json.loads((out / "receipt.json").read_text())
    assert not receipt["passed"] and receipt["error"]
    assert receipt["new_controls"] == receipt["new_neural_rows"] == receipt["raw_array_reads"] == 0
    assert (out / "findings.json").exists() and (out / "file_bindings.json").exists()
    again = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=30)
    assert again.returncode != 0 and "FileExistsError" in again.stderr
