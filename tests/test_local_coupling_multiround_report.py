"""Synthetic saved-output tests; no physics, policy inference or fitting."""

import csv
import subprocess
import sys

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.provenance import sha, write
from src.gps.mppi_gps.reporting.local_coupling_multiround import (
    activity_summary,
    answers,
    endpoint_summary,
    render,
    student_gap_endpoints,
)


def binding(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def trace(path, controls):
    states = np.zeros((controls + 1, 19), dtype=np.float64)
    states[:, 0] = np.arange(controls + 1) * 0.008
    states[:, 2], states[:, 10] = 1.2, 1.5
    states[:, 4:10] = np.arange(controls + 1)[:, None] * 0.001
    with h5py.File(path, "x") as raw:
        raw["states"] = states
        raw["actions"] = np.zeros((controls, 6), dtype=np.float64)
    return binding(path)


def test_empty_report_is_create_only_and_has_no_complete_loop_claim(tmp_path):
    value = render(tmp_path, tmp_path / "report")
    assert value["status"] == "RENDERED_WITH_MISSING_EVIDENCE"
    assert not value["answers"]["complete_trajectory"]
    assert not value["answers"]["audit_passed"]
    assert value["rendering_work"]["neural_rows"] == 0
    assert value["rendering_work"]["physics_substeps"] == 0
    assert value["rendering_work"]["optimizer_steps"] == 0
    with pytest.raises(FileExistsError):
        render(tmp_path, tmp_path / "report")


def test_complete_loop_status_requires_all_rounds_and_final_audit():
    result = dict(status="COMPLETED_ALL_ROUNDS", planned_rounds=8, criteria={"walking": False})
    streams = [
        dict(
            status="COMPLETED_ALL_ROUNDS",
            initial_evaluations=[{}] * 32,
            initial_probes=[{}] * 2,
            final_evaluations=[{}] * 64,
            rounds=[
                dict(
                    round=i,
                    status="COMPLETED",
                    acquisitions=[{}] * 16,
                    admissions=[{}] * 16,
                    fits=[{}] * 2,
                    evaluations=[{}] * 32,
                    probes=[{}] * 2,
                )
                for i in range(1, 9)
            ],
        )
        for _ in range(3)
    ]
    value = answers(result, streams, dict(passed=True))
    assert value["complete_trajectory"]
    assert value["completed_stream_rounds"] == value["planned_stream_rounds"] == 24
    assert "independently audited" in value["certification"]
    assert "not proof" in value["convergence"]
    assert not answers(result, streams, None)["audit_passed"]
    streams[2]["rounds"].pop()
    value = answers(result, streams, dict(passed=True))
    assert not value["complete_trajectory"]
    assert "no complete-loop certification" in value["certification"]


def test_activity_distinguishes_accepted_stationary_data_and_fallback():
    rows = [
        dict(
            round=2,
            stream=0,
            arm="positive",
            slot=0,
            admitted=True,
            new_experience=False,
            fallback=False,
            age=0,
            new_rows=0,
        ),
        dict(
            round=2,
            stream=0,
            arm="positive",
            slot=1,
            admitted=False,
            new_experience=False,
            fallback=True,
            age=2,
            new_rows=0,
        ),
        dict(
            round=2,
            stream=0,
            arm="positive",
            slot=2,
            admitted=True,
            new_experience=True,
            fallback=False,
            age=0,
            new_rows=300,
        ),
    ]
    value = activity_summary(rows)[0]
    assert value["newly_admitted"] == 2
    assert value["new_experience_slots"] == 1
    assert value["fallback_slots"] == 1
    assert value["new_rows"] == 300
    assert value["max_age"] == 2


def test_endpoint_summary_keeps_survival_and_walking_separate():
    row = dict(
        round=8,
        stream=0,
        arm="positive",
        kind="child",
        panel="primary",
        controls=1000,
        complete_cost=12,
        velocity_mae=1.5,
        full_survival=True,
        walking=False,
    )
    result = endpoint_summary([row])[0]
    assert result["episodes"] == 1
    assert result["walking"] == 0
    assert result["full_survival"] == 1


def test_missing_student_anchors_are_not_zero_gap_and_lost_coverage_is_flagged():
    rows = []
    for round_index, controls in ((0, (0, 50, 100)), (8, (0,))):
        for control in (0, 50, 100):
            observed = control in controls
            rows.append(
                dict(
                    round=round_index,
                    stream=0,
                    arm="positive",
                    control=control,
                    status="OBSERVED" if observed else "MISSING_AFTER_EXECUTED_PREFIX",
                    **({"gap_action_mse": 0.2 if round_index == 0 else 0.1} if observed else {}),
                )
            )
    value = student_gap_endpoints(rows, 8)[0]
    assert value["initial_observed"] == 3 and value["final_observed"] == 1
    assert value["common_controls"] == [0]
    assert value["ratio"] == 0.5
    assert not value["no_lost_anchor_coverage"]
    assert value["final_common_action_mse"] == 0.1
    zero = [
        dict(round=0, stream=0, arm="positive", control=0, status="OBSERVED", gap_action_mse=0.0)
    ]
    assert student_gap_endpoints(zero, 8)[0]["ratio"] is None


def saved_fixture(tmp_path):
    worker = tmp_path / "worker"
    worker.mkdir()
    attempted = worker / "attempt.json"
    retained = worker / "retained.json"
    write(
        attempted,
        dict(
            trace=trace(worker / "attempt.h5", 3),
            alpha=0.12,
            outcome=dict(
                controls=3,
                local_walking=False,
                full_survival=False,
                complete_cost=1500.0,
                velocity_mae=1.0,
            ),
        ),
    )
    write(
        retained,
        dict(
            trace=trace(worker / "retained.h5", 300),
            alpha=0.0,
            outcome=dict(
                controls=300,
                local_walking=True,
                full_survival=True,
                complete_cost=100.0,
                velocity_mae=0.1,
            ),
        ),
    )
    evaluations = []
    for seed, controls in ((10, 1), (11, 1000)):
        path = worker / f"eval{seed}.json"
        write(
            path,
            dict(
                trace=trace(worker / f"eval{seed}.h5", controls),
                outcome=dict(
                    controls=controls,
                    full_survival=controls == 1000,
                    walking=False,
                    complete_cost=500.0,
                    velocity_mae=1.1,
                ),
            ),
        )
        evaluations.append(
            dict(
                arm="positive", kind="child", panel="primary", reset_seed=seed, result=binding(path)
            )
        )
    curve = worker / "curve.json"
    write(
        curve,
        dict(
            learning_curve=[
                dict(
                    update=4001 + i,
                    fit_update=i + 1,
                    action_mse=0.3 - i * 0.01,
                    total_loss=0.3 - i * 0.01,
                    unclipped_gradient_norm=0.2,
                )
                for i in range(2)
            ]
        ),
    )
    fitted = worker / "fit.json"
    diagnostics = {
        name: dict(rows=3, action_mse=0.3, rms_l2=0.4) for name in ("own", "fixed_anchor", "held")
    }
    write(
        fitted,
        dict(fit=binding(curve), diagnostics_before=diagnostics, diagnostics_after=diagnostics),
    )
    probe = worker / "probe.json"
    write(
        probe,
        dict(
            anchors=[
                dict(
                    control=0, status="OBSERVED", seed=100, executed_controls=1, gap_action_mse=0.2
                ),
                dict(
                    control=50,
                    status="MISSING_AFTER_EXECUTED_PREFIX",
                    seed=101,
                    executed_controls=1,
                ),
            ]
        ),
    )
    stream = worker / "stream.json"
    write(
        stream,
        dict(
            status="FAILED_STOPPED",
            stream=0,
            initial_evaluations=[],
            initial_probes=[],
            rounds=[
                dict(
                    round=1,
                    acquisitions=[dict(arm="positive", slot=0, manifest=binding(attempted))],
                    admissions=[
                        dict(
                            arm="positive",
                            slot=0,
                            admitted=False,
                            new_experience=False,
                            fallback=True,
                            chosen=binding(retained),
                            attempt=binding(attempted),
                            last_accepted_round=0,
                            age=1,
                            new_rows=0,
                        )
                    ],
                    fits=[dict(arm="positive", result=binding(fitted))],
                    evaluations=evaluations,
                    probes=[dict(arm="positive", result=binding(probe))],
                )
            ],
            final_evaluations=[],
        ),
    )
    write(
        worker / "result.json",
        dict(
            status="FAILED_STOPPED",
            planned_rounds=8,
            streams=[dict(stream=0, result=binding(stream))],
            accounting=dict(counts={}),
        ),
    )
    return worker


def test_render_preserves_attempt_fallback_fitting_and_failed_first_reset(tmp_path):
    saved_fixture(tmp_path)
    value = render(tmp_path, tmp_path / "report")
    assert value["measurements"] == dict(
        streams=1,
        teacher_records=2,
        admission_records=1,
        standalone_episodes=2,
        fitting_updates=2,
        reconstruction_records=6,
        student_probe_anchors=2,
    )
    assert {
        "teacher_trajectory.png",
        "standalone_trajectory.png",
        "acquisition_activity.png",
        "fitting_trajectory.png",
        "reconstruction_trajectory.png",
        "student_state_gap_trajectory.png",
        "joint_angles_r1_s0.png",
    } <= set(value["figures"])
    with (tmp_path / "report/teacher_outcomes.csv").open() as f:
        teachers = list(csv.DictReader(f))
    assert {row["role"] for row in teachers} == {"attempt", "retained"}
    assert next(row for row in teachers if row["role"] == "attempt")["controls"] == "3"
    text = (tmp_path / "report/report.md").read_text()
    assert "fallback-only block is inactive" in text
    assert "UNPAIRED" in text and "including failures" in text
    assert "no complete-loop certification" in text
    assert value["rendering_work"]["neural_rows"] == 0


def test_missing_first_primary_is_not_replaced_by_later_surviving_episode(tmp_path):
    worker = saved_fixture(tmp_path)
    (worker / "eval10.json").unlink()
    value = render(tmp_path, tmp_path / "report")
    assert any("no favorable substitute" in row["reason"] for row in value["missing_evidence"])
    assert value["measurements"]["standalone_episodes"] == 1


def test_empty_render_does_not_import_simulator_or_neural_library(tmp_path):
    script = f"from pathlib import Path;import sys;from src.gps.mppi_gps.reporting.local_coupling_multiround import render;render(Path({str(tmp_path)!r}),Path({str(tmp_path / 'report')!r}));assert 'torch' not in sys.modules;assert 'mujoco' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, check=True)
