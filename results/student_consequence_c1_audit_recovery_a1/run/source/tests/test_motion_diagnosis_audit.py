"""Independent synthetic raw-motion proofs; no real motion inputs are opened."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.audits.motion_diagnosis import (
    MotionAudit,
    array_binding,
    audit_journal,
    distance,
    load_originals,
    physical_coordinates,
    query_batches,
    recompute_probe,
    segment_starts,
    source_descriptors,
    verify_array,
    verify_compact,
)


def descriptor(value: np.ndarray) -> dict:
    return dict(
        shape=list(value.shape),
        dtype=value.dtype.str,
        sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    )


def states(n: int) -> np.ndarray:
    value = np.zeros((n, 19), dtype=np.float64)
    value[:, 0] = np.arange(n) * 0.008
    value[:, 1] = np.arange(n) * 0.01
    value[:, 2] = 1.25
    value[:, 10] = 1.25
    return value


def series(n: int, identifier: str = "a", index: int = 0) -> dict:
    return dict(
        id=identifier,
        source="synthetic",
        index=index,
        N=n,
        states=states(n),
        actions=np.zeros((n, 6)),
        reference_predictions=np.zeros((n, 6), dtype=np.float32),
        terminal_state=None,
    )


def test_translation_invariance_wrapped_angles_and_velocity_are_separate():
    first, second = states(25), states(25)
    second[:, 0] += 2.0
    second[:, 1] += 50.0
    second[:, 3:10] += 2 * np.pi
    r, parts = distance(physical_coordinates(first), physical_coordinates(second))
    assert r == pytest.approx(0, abs=1e-12)
    second[:, 10] += 17
    r, parts = distance(physical_coordinates(first), physical_coordinates(second))
    assert parts["velocities"] == pytest.approx(17)
    assert r * r == pytest.approx(sum(parts.values()))
    second[:, 2] += 1
    assert distance(physical_coordinates(first), physical_coordinates(second))[1][
        "height"
    ] == pytest.approx(1 / 17)


def test_minimum_lag_self_exclusion_and_exact_lexical_tie_control():
    # Static relative coordinates are recurrent, even though recurrence alone
    # says nothing about walking. The first eligible pair must win exact ties.
    result = recompute_probe(MotionAudit(), [series(150)])
    within = result["trajectories"][0]["within"]
    assert within["starts"] == [0, 25, 50, 75, 100, 125]
    assert within["eligible_pairs"] == 3
    assert within["best"]["first_start_control"] == 0
    assert within["best"]["second_start_control"] == 100
    assert within["R_matrix"][0][0] is None
    assert within["R_matrix"][0][3] is None
    assert within["R_matrix"][0][4] == within["R_matrix"][4][0] == 0
    assert result["counts"]["aligned_state_pairs"] == 75


def test_all_cross_pairs_partial_horizons_and_unprobed_tail_are_preserved():
    first, second, third = series(26), series(8, "b", 1), series(22, "c", 2)
    first["actions"][25] = 1
    second["actions"][:] = 0.5
    result = recompute_probe(MotionAudit(), [first, second, third])
    assert result["counts"] == dict(
        trajectories=3,
        rows=56,
        curve_bins=4,
        segments=3,
        cross_trajectory_pairs=3,
        within_queries=0,
        cross_queries=3,
        pair_queries=3,
        aligned_state_pairs=38,
    )
    assert [(row["H"], row["kind"]) for row in result["cross_pairs"]] == [
        (8, "partial"),
        (22, "partial"),
        (8, "partial"),
    ]
    assert all(row["within"]["best"] is None for row in result["trajectories"])
    assert result["trajectories"][0]["curves"][-1]["rows"] == 1
    assert result["trajectories"][0]["curves"][-1]["reference_error"]["E"] == 1
    assert result["cross_pairs"][0]["best"]["action_mse"] == 0.25
    assert result["cross_pairs"][0]["best"]["first_reference_E"] == 0
    assert result["cross_pairs"][0]["best"]["second_reference_E"] == 0.25
    assert segment_starts(26) == [0] and segment_starts(8) == [0]


def test_array_hash_shape_dtype_and_nonfinite_contracts_are_distinct():
    value = np.arange(6, dtype=np.float32)
    verify_array(MotionAudit(), value, descriptor(value), "fixture")
    for changed in (value + 1, value.astype(np.float64), value.reshape(2, 3)):
        with pytest.raises(AssertionError):
            verify_array(MotionAudit(), changed, descriptor(value), "fixture")
    invalid = np.array([np.nan])
    with pytest.raises(AssertionError, match="finite"):
        verify_array(MotionAudit(), invalid, descriptor(invalid), "fixture")


def test_failed_whole_file_hash_retains_paid_read_bytes(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"saved evidence")
    audit = MotionAudit()
    with pytest.raises(AssertionError, match="SHA-256"):
        audit.bind_source(path, "a" * 64, path.stat().st_size)
    assert audit.work["source_hash_bytes"] == audit.work["total_hash_bytes"] == path.stat().st_size
    assert audit.work["dataset_attempts"] == 0


def test_rigid_order_differs_even_when_marginal_states_match():
    first = physical_coordinates(states(25))
    first[:, 0] = np.arange(25)
    reverse = first[::-1].copy()
    assert np.array_equal(np.sort(first, axis=0), np.sort(reverse, axis=0))
    assert distance(first, reverse)[0] > 0


def test_full_roster_query_census_includes_every_partial_comparison():
    catalog = dict(roster=[dict(N=n) for n in [1000] * 30 + [8, 26, 22, 14]])
    batches = query_batches(catalog)
    assert sum(pairs for pairs, _ in batches) == 720786
    assert sum(aligned for _, aligned in batches) == 17982374
    assert max(pairs for pairs, _ in batches) == 1536
    assert len(batches) == 1026


def put_hdf(stream, name, arrays):
    group = stream.create_group(name)
    result = {}
    for key, value in arrays.items():
        group.create_dataset(key, data=value).attrs["sha256"] = descriptor(value)["sha256"]
        result[key] = descriptor(value)
    return result


@pytest.fixture
def raw_fixture(tmp_path):
    teacher, native = tmp_path / "teacher.npz", tmp_path / "raw.h5"
    source_states, native_states = states(26), states(9)
    source_states[:, 11] = 12  # Projection must clip observed qvel, not physical states.
    terminal = source_states[-1].copy()
    terminal[0] += 0.008
    np.savez(teacher, states=source_states, row_steps=np.arange(26), terminal_state=terminal)
    catalog, recorded = (
        dict(files=[], roster=[]),
        dict(stages=dict(A=dict(trajectories=[]), B=dict(trajectories=[]))),
    )
    with h5py.File(native, "w") as stream:
        physical_b = put_hdf(stream, "episode", dict(states=native_states))["states"]
        for index, (name, count, physical_states) in enumerate(
            (("teacher", 26, source_states), ("random", 8, native_states[:-1]))
        ):
            actions = np.zeros((count, 6))
            observation = np.concatenate(
                [physical_states[:, 2:10], np.clip(physical_states[:, 10:19], -10, 10)], axis=1
            )
            prediction = put_hdf(
                stream,
                name,
                dict(
                    predictions=np.zeros((count, 9, 6), dtype=np.float32),
                    observations=observation,
                    executed_actions=actions,
                ),
            )
            physical = (
                dict(
                    file=str(teacher),
                    kind="sealed_teacher_npz",
                    group=None,
                    states=descriptor(source_states),
                    row_steps=descriptor(np.arange(26)),
                    terminal=descriptor(terminal),
                )
                if index == 0
                else dict(
                    file=str(native),
                    kind="k1_stage_b_episode_hdf",
                    group="episode",
                    states=physical_b,
                    row_steps=None,
                    terminal=dict(row=8),
                )
            )
            catalog["roster"].append(
                dict(
                    index=index,
                    id=name,
                    source=name,
                    N=count,
                    physical_source=physical,
                    prediction_source=dict(
                        file=str(native), group=name, arrays=prediction, reference_column=8
                    ),
                    terminal_missing=False,
                )
            )
            recorded["stages"]["A" if index == 0 else "B"]["trajectories"].append(
                dict(
                    trajectory=dict(id=name, group=name, steps=count),
                    policies=dict(
                        reference=dict(
                            mse=0.0,
                            bins=[dict(start_control=0, stop_control_exclusive=count, mse=0.0)],
                        )
                    ),
                )
            )
    for path in (teacher, native):
        catalog["files"].append(
            dict(
                path=str(path),
                recorded_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                recorded_bytes=path.stat().st_size,
            )
        )
    return catalog, recorded


def test_original_npz_hdf_payloads_exact_projection_terminals_and_read_census(raw_fixture):
    catalog, recorded = raw_fixture
    audit = MotionAudit()
    result = load_originals(audit, catalog, recorded)
    assert [row["N"] for row in result] == [26, 8]
    assert result[0]["states"][0, 11] == 12
    assert all(row["terminal_state"].shape == (19,) for row in result)
    assert audit.work["dataset_attempts"] == audit.work["completed_datasets"] == 10
    assert audit.work["failed_datasets"] == 0
    assert audit.work["source_hash_bytes"] == sum(row["recorded_bytes"] for row in catalog["files"])
    assert audit.work["source_io_bytes"] > audit.work["decoded_bytes"] > 0
    expected = deepcopy(catalog)
    expected["roster"][0]["prediction_source"]["arrays"]["predictions"]["sha256"] = "a" * 64
    with pytest.raises(AssertionError, match="SHA"):
        load_originals(MotionAudit(), expected, recorded)


def journal_fixture(catalog, bindings):
    zero = (
        "new_neural_rows",
        "new_physics_controls",
        "new_optimizer_steps",
        "new_mppi_controls",
        "checkpoint_loads",
        "gpu_operations",
    )
    work = dict.fromkeys(
        (
            "source_hash_bytes",
            "source_io_bytes",
            "dataset_attempts",
            "completed_datasets",
            "decoded_bytes",
            "failed_datasets",
            "physical_rows",
            "terminal_rows",
            "queries_requested",
            "queries_completed",
            "aligned_requested",
            "aligned_completed",
            *zero,
        ),
        0,
    )
    events = []

    def emit(event, **fields):
        events.append(
            dict(event=event, work=dict(work), wall_seconds=len(events) * 0.001, **fields)
        )

    for entry in catalog["files"]:
        emit("source_hash_requested", file=entry["path"], bytes=entry["recorded_bytes"])
        work["source_hash_bytes"] += entry["recorded_bytes"]
        emit("source_hash_completed", file=entry["path"], sha256=entry["recorded_sha256"])
    descriptors = source_descriptors(catalog)
    for row in catalog["roster"]:
        for key, value in descriptors.items():
            if not key.startswith(row["id"] + "/"):
                continue
            work["dataset_attempts"] += 1
            emit("dataset_requested", label=key, expected=value)
            work["source_io_bytes"] += 500
            work["decoded_bytes"] += bindings[key]["bytes"]
            work["completed_datasets"] += 1
            emit("dataset_completed", label=key, actual=bindings[key])
        work["physical_rows"] += row["N"]
        work["terminal_rows"] += int(not row["terminal_missing"])
        emit(
            "row_completed",
            id=row["id"],
            physical_rows=row["N"],
            terminal_rows=int(not row["terminal_missing"]),
        )
    for pairs, aligned in query_batches(catalog):
        for suffix in ("requested", "completed"):
            work["queries_" + suffix] += pairs
            work["aligned_" + suffix] += aligned
            emit("queries_" + suffix, pairs=pairs, aligned_state_pairs=aligned)
    return events, dict(work=work, wall_seconds=1.0)


@pytest.mark.parametrize(
    "tamper", (None, "missing", "pair_count", "array_label", "io_regression", "failed")
)
def test_independent_journal_replays_exact_requests_rows_batches_and_paid_work(
    tmp_path, raw_fixture, tamper
):
    catalog, recorded = raw_fixture
    audit = MotionAudit()
    load_originals(audit, catalog, recorded)
    events, result = journal_fixture(catalog, audit.arrays)
    if tamper == "missing":
        events.pop()
    elif tamper == "pair_count":
        events[-1]["pairs"] += 1
    elif tamper == "array_label":
        next(row for row in events if row["event"] == "dataset_completed")["label"] = "wrong-owner"
    elif tamper == "io_regression":
        events[-1]["work"]["source_io_bytes"] = 0
    elif tamper == "failed":
        events[-1]["event"] = "worker_failed"
    path = tmp_path / "work.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    config = dict(budgets=dict(source_io_bytes=2**20))
    if tamper is None:
        assert audit_journal(audit, path, catalog, result, config)["work"] == result["work"]
    else:
        with pytest.raises(AssertionError):
            audit_journal(audit, path, catalog, result, config)


@pytest.mark.parametrize("tamper", (None, "value", "missing_terminal", "binding"))
def test_compact_motion_archive_is_checked_against_reopened_original_arrays(
    tmp_path, raw_fixture, tamper
):
    catalog, recorded = raw_fixture
    audit = MotionAudit()
    data = load_originals(audit, catalog, recorded)
    arrays = {}
    for index, row in enumerate(data):
        for prefix, key in (
            ("states", "states"),
            ("actions", "actions"),
            ("reference", "reference_predictions"),
            ("terminal", "terminal_state"),
        ):
            arrays[f"{prefix}_{index:03d}"] = row[key].copy()
    bindings = {key: array_binding(value) for key, value in arrays.items()}
    if tamper == "value":
        arrays["states_000"][0, 2] += 0.01
    elif tamper == "missing_terminal":
        del arrays["terminal_000"]
    elif tamper == "binding":
        bindings["reference_000"]["sha256"] = "b" * 64
    path = tmp_path / "motion.npz"
    np.savez_compressed(path, **arrays)
    result = dict(
        motion_archive=dict(
            file=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,
        ),
        motion_array_bindings=bindings,
    )
    if tamper is None:
        verify_compact(audit, tmp_path, result, data)
        assert audit.work["compact_arrays"] == 8
        assert audit.work["compact_io_bytes"] > 0
    else:
        with pytest.raises(AssertionError):
            verify_compact(audit, tmp_path, result, data)


def test_audit_imports_no_execution_or_metric_implementation_and_fails_create_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    imported = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from src.gps.mppi_gps.audits.motion_diagnosis import MotionAudit, assert_independent_imports; assert_independent_imports(MotionAudit())",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert imported.returncode == 0, imported.stderr
    out = tmp_path / "audit"
    command = [
        sys.executable,
        "-B",
        "scripts/audits/motion_diagnosis.py",
        "--run",
        str(tmp_path / "missing"),
        "--out",
        str(out),
        "--config-sha",
        "a" * 64,
        "--seal-sha",
        "b" * 64,
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    receipt = json.loads((out / "receipt.json").read_text())
    assert not receipt["passed"] and receipt["new_controls"] == receipt["new_neural_rows"] == 0
    again = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert again.returncode != 0 and "FileExistsError" in again.stderr
