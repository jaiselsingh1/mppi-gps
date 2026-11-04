"""Pure saved-data rendering: no checkpoint, optimizer or simulator execution."""

import subprocess
import sys

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.provenance import sha, write
from src.gps.mppi_gps.reporting.local_coupling_learning import answers, render


def binding(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def trace(path, n):
    states = np.zeros((n + 1, 19))
    states[:, 0], states[:, 2], states[:, 10] = np.arange(n + 1) * 0.008, 1.2, 1.5
    states[:, 4:10] = np.arange(n + 1)[:, None] * 0.001
    with h5py.File(path, "x") as raw:
        raw["states"] = states
        raw["actions"] = np.zeros((n, 6))
    return binding(path)


def test_missing_report_is_explicit_and_create_only(tmp_path):
    worker = tmp_path / "worker"
    worker.mkdir()
    result = render(tmp_path, tmp_path / "report")
    assert result["worker_status"] is None
    assert result["status"] == "RENDERED_WITH_MISSING_EVIDENCE"
    assert result["missing_evidence"]
    assert "Not tested or incomplete" in result["answers"]["walking"]
    assert result["rendering_work"]["neural_rows"] == 0
    assert result["rendering_work"]["native_controls"] == 0
    with pytest.raises(FileExistsError):
        render(tmp_path, tmp_path / "report")


def test_worker_walking_and_feedback_declarations_need_independent_audit():
    value = answers(
        dict(
            rounds=[dict(round=1, criterion=dict(passed=True, winning_streams=[0, 1]))],
            useful_feedback=True,
        ),
        dict(selected_condition="weak", alpha=0.12),
        None,
    )
    assert not value["audit_passed"]
    assert "Worker declarations only" in value["certification"]
    assert "does not establish convergence" in value["feedback"]


def fixture_run(tmp_path):
    worker = tmp_path / "worker"
    worker.mkdir()
    acquisitions = []
    for slot in range(8):
        arrays = trace(worker / f"teacher{slot}.h5", 300)
        path = worker / f"teacher{slot}.json"
        write(
            path,
            dict(
                alpha=0.0,
                trace=arrays,
                outcome=dict(
                    controls=300,
                    local_walking=True,
                    full_survival=True,
                    complete_cost=100.0 + slot,
                    velocity_mae=0.0,
                ),
            ),
        )
        acquisitions.append(
            dict(round=1, stream=-1, arm="independent", slot=slot, manifest=binding(path))
        )
    records = []
    for seed, n in ((10, 1), (11, 1000)):
        arrays = trace(worker / f"eval{seed}.h5", n)
        path = worker / f"eval{seed}.json"
        write(
            path,
            dict(
                trace=arrays,
                outcome=dict(
                    controls=n,
                    walking=n == 1000,
                    full_survival=n == 1000,
                    complete_cost=100.0,
                    velocity_mae=0.0,
                ),
            ),
        )
        records.append(
            dict(
                stream=0,
                arm="positive",
                kind="child",
                panel="primary",
                reset_seed=seed,
                result=binding(path),
            )
        )
    fitted = worker / "fit.json"
    write(
        fitted,
        dict(
            learning_curve=[
                dict(
                    update=2001 + i,
                    fit_update=i + 1,
                    action_mse=0.4 - 0.1 * i,
                    total_loss=0.4 - 0.1 * i,
                    unclipped_gradient_norm=0.2,
                )
                for i in range(2)
            ]
        ),
    )
    fit_result = worker / "fit_result.json"
    write(
        fit_result,
        dict(
            fit=binding(fitted),
            **{
                "diagnostics_" + stage: {
                    name: dict(rows=3, action_mse=0.4 - 0.1 * index, rms_l2=0.2)
                    for name in ("own", "cross", "held")
                }
                for index, stage in enumerate(("before", "after"))
            },
        ),
    )
    write(
        worker / "result.json",
        dict(
            status="FAILED_STOPPED",
            error="Synthetic partial fixture",
            acquisitions=acquisitions,
            rounds=[
                dict(
                    round=1,
                    evaluations=records,
                    fits=[dict(stream=0, arm="positive", result=binding(fit_result))],
                )
            ],
            accounting=dict(counts=dict(completed_optimizer_steps=2)),
        ),
    )
    return worker


def test_saved_curves_all_teachers_and_first_reset_including_failure_are_rendered(tmp_path):
    fixture_run(tmp_path)
    result = render(tmp_path, tmp_path / "report")
    assert result["measurements"] == dict(
        teacher_episodes=8, standalone_episodes=2, fitting_updates=2, reconstruction_records=6
    )
    assert result["policy_illustration_selection"][0]["reset_seed"] == 10
    assert {
        "teacher_outcomes.png",
        "standalone_outcomes.png",
        "fitting_curves.png",
        "reconstruction.png",
        "teacher_angles_r1_s-1_independent.png",
        "policy_angles_r1.png",
    } <= set(result["figures"])
    assert all((tmp_path / "report" / name).stat().st_size > 0 for name in result["figures"])
    text = (tmp_path / "report/report.md").read_text()
    assert "UNPAIRED" in text and "including failures" in text
    assert result["rendering_work"]["optimizer_steps"] == 0
    assert result["rendering_work"]["logical_io"]["decoded_array_bytes"] > 0


def test_missing_first_reset_has_no_favorable_replacement(tmp_path):
    worker = fixture_run(tmp_path)
    (worker / "eval10.json").unlink()
    result = render(tmp_path, tmp_path / "report")
    assert result["policy_illustration_selection"][0]["reset_seed"] == 10
    assert any("no favorable substitute" in row["reason"] for row in result["missing_evidence"])


def test_empty_render_does_not_import_torch_or_mujoco(tmp_path):
    code = f"from pathlib import Path; import sys; from src.gps.mppi_gps.reporting.local_coupling_learning import render; render(Path({str(tmp_path)!r}),Path({str(tmp_path / 'report')!r})); assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True)
