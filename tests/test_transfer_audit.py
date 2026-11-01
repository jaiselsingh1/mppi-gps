"""Independent transfer-audit fixtures contain no simulator or network work."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.audits.transfer import (
    TransferAudit,
    audit_ledger,
    expected_forks,
    leave_trajectory_out,
    native_trace,
    normalized_distances,
    prefix_statistics,
    support_arrays,
    support_statistics,
    verify_baseline_order,
    verify_trace,
)


def trace(steps=10, *, fall=False):
    states = np.zeros((steps + 1, 19))
    states[:, 0] = np.arange(steps + 1) * 0.008
    states[:, 2] = 1.25
    states[:, 10] = 1.5
    if fall:
        states[-1, 2] = 0.7
    arrays = dict(
        states=states,
        actions=np.zeros((steps, 6)),
        warmstarts=np.zeros((steps + 1, 9)),
        integrations=np.zeros((steps + 1, 91)),
    )
    saved = native_trace(arrays)
    arrays.update(
        {
            key: saved[key]
            for key in ("costs", "observations", "velocities", "done_flags", "fall_costs")
        }
    )
    return arrays


def test_loto_uses_other_trajectories_and_fixed_rms_normalization():
    observations = np.zeros((4, 17))
    observations[:, 0] = [0, 2, 10, 12]
    owners = np.repeat([0, 1], 2)
    mean, std = np.zeros(17), np.ones(17)
    distance = leave_trajectory_out(observations, owners, mean, std)
    assert np.allclose(distance, np.array([10, 8, 8, 10]) / np.sqrt(17))
    assert np.array_equal(normalized_distances(observations, observations, mean, std), np.zeros(4))
    std[0] = 2
    assert np.allclose(leave_trajectory_out(observations, owners, mean, std), distance / 2)
    with pytest.raises(ValueError, match="two"):
        leave_trajectory_out(observations, np.zeros(4), mean, std)


def test_support_threshold_is_strict_and_physical_axes_are_not_observation_axes():
    distance = np.array([0.0, 1.0, 2.0])
    states = np.zeros((3, 19))
    states[:, 2] = [1.1, 1.2, 1.3]
    states[:, 3] = [0.1, 0.2, 0.3]
    states[:, 10] = [1.5, 11, -12]
    raw = support_arrays(distance, states, 1.0)
    assert raw["outside"].tolist() == [False, False, True]
    assert raw["time_to_end"].tolist() == [2, 1, 0]
    assert raw["height"].tolist() == [1.1, 1.2, 1.3]
    assert raw["forward_velocity"].tolist() == [1.5, 11, -12]
    assert raw["clipped_count"].tolist() == [0, 1, 1]
    summary = support_statistics(distance, 1.0)
    assert summary["median_distance"] == 1
    assert summary["outside_fraction"] == 1 / 3
    assert summary["first100_outside_fraction"] == 1 / 3
    assert summary["last25_outside_fraction"] == 1 / 3


def test_native_prefix_cost_terminal_once_and_fall_tail():
    baseline = trace()
    branch = trace(2, fall=True)
    result = prefix_statistics(branch, baseline, [1, 2, 10])
    assert result[0]["complete_cost"] == 0
    assert result[0]["done"] is False
    assert result[1]["raw_cost"] == 5
    assert result[1]["terminal_cost"] == 20
    assert result[1]["absorbing_cost"] == 0
    assert result[1]["complete_cost"] == 25
    assert result[2]["executed"] == 2
    assert result[2]["complete_cost"] == 65
    assert result[2]["absorbing_cost"] == 40
    assert result[2]["delta_cost"] == 65
    assert result[2]["state_rms"] is None
    assert result[2]["warmstart_rms"] is None
    assert result[2]["integration_rms"] is None
    # A failed unmodified replay is retained but never licenses substitutions.
    invalid_baseline = prefix_statistics(branch, branch, [1, 2, 10])
    assert invalid_baseline[2]["complete_cost"] == 65
    assert invalid_baseline[2]["delta_cost"] == 40
    assert invalid_baseline[2]["state_rms"] is None


def test_native_trace_audit_detects_cost_double_terminal_and_post_fall_steps(tmp_path):
    arrays = trace(2, fall=True)
    audit = TransferAudit(tmp_path, tmp_path)
    verify_trace(audit, arrays, dict(steps=2, done=True))
    changed = deepcopy(arrays)
    changed["costs"][-1] += 20
    with pytest.raises(AssertionError, match="arithmetic costs"):
        verify_trace(audit, changed, dict(steps=2, done=True))
    changed = trace(3)
    changed["states"][1, 2] = 0.7
    with pytest.raises(ValueError, match="first fall"):
        native_trace(changed)


def test_raw_native_actions_and_observation_tampering_is_detected(tmp_path):
    arrays = trace(2)
    arrays["actions"][0, 0] = 0.5
    with pytest.raises(AssertionError, match="arithmetic costs"):
        verify_trace(TransferAudit(tmp_path, tmp_path), arrays, dict(steps=2, done=False))
    arrays = trace(2)
    arrays["observations"][0, 0] += 0.01
    with pytest.raises(AssertionError, match="arithmetic observations"):
        verify_trace(TransferAudit(tmp_path, tmp_path), arrays, dict(steps=2, done=False))


def test_every_baseline_must_precede_any_perturbation(tmp_path):
    events = [
        dict(event="transfer_baseline_complete", teacher_id="t", anchor=0, replay_passed=True),
        dict(event="transfer_baseline_complete", teacher_id="t", anchor=1, replay_passed=True),
        dict(event="transfer_baseline_gate", passed=True, baselines=2),
        dict(event="transfer_branch_start", baseline=False),
    ]
    verify_baseline_order(TransferAudit(tmp_path, tmp_path), events, 2)
    wrong = [events[-1], *events[:-1]]
    with pytest.raises(AssertionError, match="before baseline gate"):
        verify_baseline_order(TransferAudit(tmp_path, tmp_path), wrong, 2)
    wrong = deepcopy(events)
    wrong[0]["replay_passed"] = False
    with pytest.raises(AssertionError, match="replay gate"):
        verify_baseline_order(TransferAudit(tmp_path, tmp_path), wrong, 2)
    with pytest.raises(AssertionError, match="Complete baseline"):
        verify_baseline_order(TransferAudit(tmp_path, tmp_path), events, 3)


def test_saved_support_summary_and_arrays_are_independently_proved(tmp_path):
    distance = np.array([0.1, 0.4, 0.2])
    expected = support_statistics(distance, 0.2)
    audit = TransferAudit(tmp_path, tmp_path)
    audit.prove(deepcopy(expected), expected, "Support")
    changed = deepcopy(expected)
    changed["outside_fraction"] = 0.0
    with pytest.raises(AssertionError, match="outside_fraction"):
        audit.prove(changed, expected, "Support")


def test_fork_roster_joins_saved_owner_rows_and_draws_random_only_twenty_times():
    names = ["reference", "untrained_0", "untrained_1", "untrained_2"] + [
        f"clone{i}" for i in range(5)
    ]
    teachers, points, files, data = [], [], {}, {}
    data["validation.npz"] = dict(
        row_episode=np.repeat(["a", "b"], 1000), observations=np.zeros((2000, 17))
    )
    for index, name in enumerate(("a", "b")):
        values = trace(1000)
        relative, prediction_file = name + ".npz", name + "_predictions.npz"
        data[relative] = dict(
            states=values["states"][:-1],
            terminal_state=values["states"][-1],
            warmstarts=values["warmstarts"][:-1],
            terminal_warmstarts=values["warmstarts"][-1],
            integrations=values["integrations"][:-1],
            terminal_integrations=values["integrations"][-1],
            observations=np.zeros((1000, 17)),
            actions=values["actions"],
            costs=values["costs"],
        )
        data[prediction_file] = dict(predictions=np.full((1000, 9, 6), index / 10))
        teachers.append(
            dict(
                id=name, file=relative, k1_predictions_file=prediction_file, validation_episode=name
            )
        )
        files[relative] = dict(sha256="a" * 64, arrays=dict(actions=dict(sha256="b" * 64)))
        files[prediction_file] = dict(
            sha256="c" * 64, arrays=dict(predictions=dict(sha256="d" * 64))
        )
    for pair in range(3):
        for update in (2000, 10000):
            name = f"s{pair}_u{update}"
            relative = name + ".npz"
            points.append(
                dict(
                    id=name,
                    pair=pair,
                    update=update,
                    predictions_file=relative,
                    policy_hash=name,
                    checkpoint_file=name + ".pt",
                )
            )
            predictions = np.zeros((2000, 6))
            predictions[:1000] = 0.2
            predictions[1000:] = 0.3
            data[relative] = dict(validation_predictions=predictions)
            files[relative] = dict(
                sha256="e" * 64, arrays=dict(validation_predictions=dict(sha256="f" * 64))
            )
    inventory = dict(
        teachers=teachers,
        w2_points=list(reversed(points)),
        k1_policy_names=names,
        files=files,
        policies={name: dict(policy_hash=name, checkpoint_file=name + ".pt") for name in names},
    )
    forks = expected_forks(
        inventory, data, dict(anchors=list(range(0, 1000, 100)), random_seed=2026111001)
    )
    assert len(forks) == 20 and all(len(row["branches"]) == 12 for row in forks)
    assert [row["name"] for row in forks[0]["branches"]][5:11] == [row["id"] for row in points]
    assert np.array_equal(forks[0]["branches"][5]["action"], np.full(6, 0.2))
    assert np.array_equal(forks[10]["branches"][5]["action"], np.full(6, 0.3))
    expected = np.random.Generator(np.random.PCG64(2026111001)).uniform(-1, 1, (20, 6))
    assert np.array_equal(np.asarray([row["branches"][-1]["action"] for row in forks]), expected)
    assert forks[10]["branches"][5]["provenance"]["row"] == 1000


def ledger_fixture(tmp_path, *, failed=False):
    counts = dict(
        requested_controls=1,
        requested_live_controls=1,
        requested_physics_substeps=4,
        live_queries=1,
        completed_controls=0 if failed else 1,
        completed_live_controls=0 if failed else 1,
        completed_physics_substeps=0 if failed else 4,
        queries=1,
        physics_frames=0 if failed else 4,
        snapshots=0,
        wall_seconds=0.5,
    )
    if failed:
        counts.update(
            failed_queries=1, unknown_partial_controls=1, unknown_partial_physics_substeps=4
        )
    owner = dict(
        teacher_id="t",
        anchor=0,
        name="teacher",
        baseline=True,
        step=0,
        stage="d2",
        mode="live",
        query_id="q0",
    )
    events = [
        dict(event="runtime_open"),
        dict(
            event="query_requested",
            **owner,
            requested_controls=1,
            requested_physics_substeps=4,
            frame_skip=4,
        ),
        dict(
            event="query_error" if failed else "query_complete",
            **owner,
            request_id="event-000000001",
            completed_controls=0 if failed else 1,
            completed_physics_substeps=0 if failed else 4,
        ),
        dict(event="runtime_close", totals=counts),
    ]
    if failed:
        events[2]["unknown_partial_controls"] = 1
    folder = tmp_path / "evidence"
    folder.mkdir()

    def save():
        for index, row in enumerate(events):
            row.update(id=f"event-{index:09d}", kind=row["event"], wall_seconds=index * 0.1)
        (folder / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))

    save()
    config = dict(
        d1_budgets=dict(support_queries=75000, wall_seconds=1200),
        d2_budgets=dict(
            branches=240, live_controls=24000, physics_substeps=96000, wall_seconds=2400
        ),
    )
    return events, counts, config, save


def test_ledger_independently_recounts_controls_and_rejects_open_queries(tmp_path):
    events, counts, cfg, save = ledger_fixture(tmp_path)
    checked = audit_ledger(TransferAudit(tmp_path, tmp_path), tmp_path, cfg, dict(work=counts))
    assert checked["counts"]["completed_controls"] == 1
    changed = dict(counts, completed_controls=2)
    with pytest.raises(AssertionError, match="work census"):
        audit_ledger(TransferAudit(tmp_path, tmp_path), tmp_path, cfg, dict(work=changed))
    events.pop(2)
    save()
    with pytest.raises(AssertionError, match="Closed durable"):
        audit_ledger(TransferAudit(tmp_path, tmp_path), tmp_path, cfg, dict(work=counts))


def test_failed_query_keeps_unknown_work_instead_of_imputing_zero(tmp_path):
    _, counts, cfg, _ = ledger_fixture(tmp_path, failed=True)
    checked = audit_ledger(TransferAudit(tmp_path, tmp_path), tmp_path, cfg, dict(work=counts))
    assert checked["counts"]["completed_controls"] == 0
    assert checked["counts"]["unknown_partial_controls"] == 1
    changed = dict(counts, unknown_partial_controls=0)
    with pytest.raises(AssertionError, match="work census"):
        audit_ledger(TransferAudit(tmp_path, tmp_path), tmp_path, cfg, dict(work=changed))


def test_cli_failure_receipt_and_no_production_imports(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from src.gps.mppi_gps.audits.transfer import prefix_statistics
for name in sys.modules:
    assert not name.startswith(('mujoco', 'src.envs', 'src.gps.mppi_gps.policy', 'src.gps.mppi_gps.experiments', 'src.gps.mppi_gps.transfer_inputs', 'src.gps.mppi_gps.transfer_metrics'))
"""
    checked = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=root, text=True, capture_output=True
    )
    assert checked.returncode == 0, checked.stderr
    out = tmp_path / "audit"
    checked = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/audits/transfer_diagnosis.py",
            "--run",
            str(tmp_path / "missing"),
            "--out",
            str(out),
            "--config-sha",
            "0" * 64,
            "--seal-sha",
            "0" * 64,
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert checked.returncode == 1
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["passed"] is False
    assert (
        receipt["new_controls"] == receipt["new_neural_rows"] == receipt["new_optimizer_steps"] == 0
    )
    assert (out / "findings.json").is_file()
