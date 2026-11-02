"""Synthetic C1 audit coordinator failures retain receipts without reruns."""

from __future__ import annotations

import json

import pytest

from src.gps.mppi_gps import lifecycle, proposal_launch
from src.gps.mppi_gps.experiments import student_consequence as module


@pytest.mark.parametrize("has_process_receipt", (True, False))
def test_audit_coordinator_persists_exception_receipt_and_restores_affinity(
    tmp_path, monkeypatch, has_process_receipt
):
    run, output = tmp_path / "run", tmp_path / "audit"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            dict(
                runtime=dict(
                    cpu_affinity=[0],
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MUJOCO_GL="disable",
                    CUDA_VISIBLE_DEVICES="",
                ),
                limits=dict(audit_wall_seconds=60, audit_peak_rss_bytes=1024**3),
            )
        )
    )
    (run / "manifest.json").write_text("{}")
    monkeypatch.setattr(lifecycle, "verify", lambda path: None)
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    affinity = []
    monkeypatch.setattr(
        module.os, "sched_setaffinity", lambda pid, cpus: affinity.append(cpus), raising=False
    )
    failure = RuntimeError("synthetic coordinator RSS failure")
    paid_receipt = dict(
        exit_code=-9,
        stopping_reason="coordinator_exception",
        leader_reaped=True,
        process_group_kill_requested=True,
        unknown_partial_work_possible=True,
        wall_seconds=0.1,
        retries=0,
    )
    if has_process_receipt:
        failure.process_receipt = paid_receipt
    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise failure

    monkeypatch.setattr(proposal_launch, "bounded_process", fail)
    with pytest.raises(RuntimeError, match="synthetic coordinator") as caught:
        module.launch_audit(run, output)
    assert caught.value is failure and len(calls) == 1
    assert affinity == [[0], {0, 1}]
    saved = json.loads((output / "exit.json").read_text())
    if has_process_receipt:
        assert saved == paid_receipt and saved["unknown_partial_work_possible"]
    else:
        assert saved["process_started"] is None and saved["unknown_partial_work_possible"]
        assert saved["error"]["type"] == "RuntimeError"
        assert "exit_code" not in saved and "leader_reaped" not in saved
    with pytest.raises(FileExistsError):
        module.launch_audit(run, output)
    assert len(calls) == 1
