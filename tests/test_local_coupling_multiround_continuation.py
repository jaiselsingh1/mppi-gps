"""Synthetic L3 continuation: no network, simulator or optimizer is constructed."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from test_local_coupling_multiround_lifecycle import fixture_launch, mock_jobs, saved_trace

from src.gps.mppi_gps import local_coupling_controller as controller
from src.gps.mppi_gps import local_coupling_multiround as original
from src.gps.mppi_gps import local_coupling_multiround_continuation as continuation
from src.gps.mppi_gps.provenance import write
from src.gps.mppi_gps.trajectory_pi_native import binding


def fixture_resume(tmp_path, monkeypatch, *, stream=0):
    """Stop only mocked scientific bodies after three completed round blocks."""
    old_base = tmp_path / "old"
    old_base.mkdir()
    launch, config = fixture_launch(old_base)
    if stream:
        prepared = original.read(original.bound(launch["prepared_manifest"]))
        for row in prepared["initial_streams"]:
            row["stream"] = stream
        write(old_base / "prepared_stream.json", prepared)
        launch["prepared_manifest"] = binding(old_base / "prepared_stream.json")
    calls = mock_jobs(monkeypatch, config)
    acquire = controller.run_episode
    cut = {0: 2, 1: 0, 2: 1}[stream]
    interrupted_job = {}

    def stop_at_four(job):
        if job["round"] == 4 and job["arm"] == "independent" and job["slot"] == cut:
            if stream:
                interrupted_job.update(copy.deepcopy(job))
                raise RuntimeError("synthetic interrupted forecast")
            result = acquire(job)
            result["result"]["status"] = "FAILED_STOPPED"
            return result
        return acquire(job)

    monkeypatch.setattr(controller, "run_episode", stop_at_four)
    old_study = original.StreamStudy(launch, stream)
    old_result = old_study.run()
    assert old_result["status"] == "FAILED_STOPPED"
    snapshot = binding((old_study.root / "result.json").resolve())
    retained = {
        f"{arm}:{row['slot']}": dict(chosen=row["manifest"], last_accepted_round=0)
        for arm in original.ARMS
        for row in old_study.prepared["anchors"]
    }
    for block in old_result["rounds"]:
        for row in block["admissions"]:
            retained[f"{row['arm']}:{row['slot']}"] = dict(
                chosen=row["chosen"], last_accepted_round=row["last_accepted_round"]
            )
    new_base = tmp_path / "new"
    new_base.mkdir()
    extra = []
    interrupted = []
    if stream:
        out = Path(interrupted_job["out"])
        out.mkdir(parents=True)
        write(out / "job.json", interrupted_job)
        prefix = saved_trace(out / "trace.h5", controls=3)
        write(out / "banks.h5", dict(synthetic=True))
        with (out / "full_t000_p0.npz").open("xb") as target:
            np.savez(target, synthetic=np.asarray(True))
        pending = dict(
            event="requested",
            id=7,
            operation="native_forecast",
            charges=dict(model_controls=12288, model_queries=1, physics_substeps=49152),
        )
        (out / "ledger.jsonl").write_text(json.dumps(pending) + "\n")
        write(new_base / "resume_context.npz", dict(synthetic=True))
        interrupted.append(
            dict(
                stream=stream,
                round=4,
                arm="independent",
                slot=cut,
                directory=str(out),
                job=binding(out / "job.json"),
                trace=prefix,
                banks=binding(out / "banks.h5"),
                ledger=binding(out / "ledger.jsonl"),
                resume_context=binding(new_base / "resume_context.npz"),
                resume_rng_state=dict(synthetic=True),
                ledger_counts=dict(
                    completed=dict(native_controls=3, neural_rows=7),
                    open_requests=[pending],
                    unknown_partial_upper=pending["charges"],
                ),
            )
        )
        extra = [continuation.pending_overhead(interrupted[0])]
    cert_dir = new_base / "certificate"
    cert_dir.mkdir()
    write(
        cert_dir / "receipt.json",
        dict(
            status="COMPLETED",
            counts=dict(requested_neural_rows=4, completed_neural_rows=4),
            cpu_seconds=0.0,
        ),
    )
    if stream == 0:
        source = old_study.root / "acquisition/r4/independent/slot2/manifest.json"
        write(cert_dir / "adopted_manifest.json", original.read(source))
    entry = dict(
        stream=stream,
        snapshot=snapshot,
        ledger=binding(old_study.root / "study_ledger.jsonl"),
        incoming_for_round4=[
            dict(arm=arm, envelope=old_study.incoming[arm]) for arm in original.ARMS
        ],
        retained_after_latest_saved_admissions=retained,
        interrupted_round4=interrupted,
        infrastructure_overhead=extra,
        closed_round4=[],
    )
    engineering = dict(
        parent_launch=binding(old_base / "config.json"),
        parent_worker_result=snapshot,
        inventory=snapshot,
        amendment=snapshot,
        source_manifest=snapshot,
        budget_antecedents=dict(
            stream_cpu_seconds_upper=[0.0] * 3,
            worker_cpu_seconds=0.0,
            global_cpu_seconds=0.0,
            worker_wall_seconds=0.0,
            global_wall_seconds=0.0,
        ),
    )
    resumed_launch = dict(launch, base_directory=str(new_base), engineering=engineering)
    resume = dict(
        schema=continuation.RESUME_SCHEMA,
        config=launch["config"],
        streams=[entry],
        numerical_completion=dict(adopted_manifest=binding(cert_dir / "adopted_manifest.json"))
        if stream == 0
        else {},
        numerical_completion_receipt=binding(cert_dir / "receipt.json"),
        budget_antecedents=engineering["budget_antecedents"],
    )
    write(new_base / "resume_manifest.json", resume)
    new_calls = mock_jobs(monkeypatch, config)
    return resumed_launch, resume, old_study, old_result, calls, new_calls


def test_three_completed_rounds_and_partial_four_are_reused_then_all_eight_finish(
    tmp_path, monkeypatch
):
    launch, resume, old, old_result, _, calls = fixture_resume(tmp_path, monkeypatch)
    old_bytes = {path: path.read_bytes() for path in old.root.rglob("*") if path.is_file()}
    study = continuation.ResumeStudy(launch, 0, resume)
    result = study.run()
    assert result["status"] == "COMPLETED_ALL_ROUNDS", result.get("error")
    assert result["rounds"][:3] == old_result["rounds"][:3]
    assert all(row["status"] == "COMPLETED" for row in result["rounds"])
    assert min(row["round"] for row in calls["acquisition"]) == 4
    assert len(calls["acquisition"]) == 5 * 16 - 3
    assert len(calls["fits"]) == 10
    assert len(calls["evaluations"]) == 14
    assert len(calls["probes"]) == 10
    assert all(
        not original.read(original.bound(row["result"]))["outcome"]["walking"]
        for block in result["rounds"]
        for row in block["evaluations"]
    )
    assert all(path.read_bytes() == data for path, data in old_bytes.items())
    old_ledger = (old.root / "study_ledger.jsonl").read_bytes()
    assert (study.root / "study_ledger.jsonl").read_bytes().startswith(old_ledger)
    accounts = result["accounting"]
    assert len(accounts["extra_verification_receipts"]) == 1
    assert accounts["logical_counts"]["completed_neural_rows"] == 4
    assert accounts["counts"] == accounts["logical_counts"]
    assert len(accounts["child_receipts"]) == len(
        {row["path"] for row in accounts["child_receipts"]}
    )
    # The final baseline is the original 4k actor; the retained actor reaches20k.
    for arm in original.ARMS:
        assert original.read(original.bound(study.initial[arm]))["adam_updates"] == 4000
        assert original.read(original.bound(study.incoming[arm]))["adam_updates"] == 20000
        assert (
            next(row for row in calls["fits"] if row["arm"] == arm)["before"]["adam_updates"]
            == 10000
        )


@pytest.mark.parametrize("stream", [1, 2])
def test_interrupted_prefix_is_reserved_then_imported_once_without_prefix_reexecution(
    tmp_path, monkeypatch, stream
):
    launch, resume, old, _, _, calls = fixture_resume(tmp_path, monkeypatch, stream=stream)
    from src.gps.mppi_gps import local_coupling_prefix_resume as prefix

    acquire = controller.run_episode
    seen = []
    study = continuation.ResumeStudy(launch, stream, resume)
    before = study.reserved_counts()
    assert before["requested_native_controls"] == 3
    assert before["requested_model_controls"] == 12288

    def resume_episode(job):
        assert job["checkpoint"] == original.read(original.bound(job["prefix_job"]))["checkpoint"]
        assert job["config"] == launch["config"]
        assert len(job["prefix_full_banks"]) == 1
        assert Path(job["reference_R1_trace"]["path"]).is_file()
        # The fixture does no actual inference/physics. Its raw suffix job is
        # joined to explicit known-prefix counts exactly as the real helper is.
        seen.append(copy.deepcopy(job))
        value = acquire(job)
        prefix_counts = continuation.prefix_reservation(
            resume["streams"][0]["interrupted_round4"][0]
        )
        receipt = dict(
            value["receipt"],
            counts=continuation.add_counts(value["receipt"]["counts"], prefix_counts),
        )
        receipt_path = Path(value["path"]) / "receipt.json"
        raw = receipt_path.read_bytes()
        receipt_path.unlink()
        receipt_path.with_name("suffix_receipt.json").write_bytes(raw)
        write(receipt_path, receipt)
        value["receipt"] = receipt
        return value

    monkeypatch.setattr(prefix, "resume_episode", resume_episode)
    result = study.run()
    assert result["status"] == "COMPLETED_ALL_ROUNDS", result.get("error")
    assert len(seen) == 1 and not study.pending_prefixes
    assert len(calls["fits"]) == 10
    accounts = result["accounting"]
    assert accounts["logical_counts"]["completed_native_controls"] == 3
    assert accounts["logical_counts"]["requested_neural_rows"] == 7
    assert accounts["infrastructure_counts"] == dict(
        requested_model_controls=12288,
        unknown_partial_model_controls=12288,
        requested_model_queries=1,
        unknown_partial_model_queries=1,
        requested_physics_substeps=49152,
        unknown_partial_physics_substeps=49152,
    )
    assert accounts["counts"] == continuation.add_counts(
        accounts["logical_counts"], accounts["infrastructure_counts"]
    )
    assert not accounts["pending_known_prefix_reservations"]
    assert not accounts["extra_verification_receipts"]
    assert old.result["status"] == "FAILED_STOPPED"


def test_unknown_new_work_and_exhausted_continuation_cpu_stop(tmp_path, monkeypatch):
    launch, resume, _, _, _, _ = fixture_resume(tmp_path, monkeypatch)
    study = continuation.ResumeStudy(launch, 0, resume)
    study.counts["unknown_partial_native_controls"] = 1
    with pytest.raises(RuntimeError, match="unresolved execution"):
        study.check()
    study.counts.pop("unknown_partial_native_controls")
    study.previous_cpu = study.config["budgets"]["worker"]["stream_cpu_seconds"] + 1
    with pytest.raises(RuntimeError, match="CPU budget"):
        study.check()
    study.ledger.close()


def test_failed_certificate_cancels_before_resume_or_worker_launch(tmp_path, monkeypatch):
    write(tmp_path / "config.json", dict(streams=3, rounds=8))
    launch = dict(
        base_directory=str(tmp_path),
        config=binding(tmp_path / "config.json"),
        prepared_manifest={},
        source_manifest={},
        engineering=dict(
            parent_launch={},
            parent_worker_result={},
            amendment={},
            inventory={},
            source_manifest={},
            budget_antecedents=dict(
                stream_cpu_seconds_upper=[0] * 3,
                worker_cpu_seconds=0,
                global_cpu_seconds=0,
                worker_wall_seconds=0,
                global_wall_seconds=0,
            ),
        ),
    )
    called = []

    def failed(_):
        called.append("certificate")
        raise RuntimeError("saved numeric prerequisite failed")

    monkeypatch.setattr(continuation, "_certify", failed)
    monkeypatch.setattr(continuation, "freeze_resume", lambda *_: pytest.fail("must not freeze"))
    monkeypatch.setattr(
        continuation.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("must not launch")
    )
    result = continuation.coordinator(launch)
    assert result["status"] == "FAILED_STOPPED"
    assert "numeric prerequisite failed" in result["error"]
    assert called == ["certificate"] and not (tmp_path / "resume_manifest.json").exists()


def test_pending_overhead_requires_exact_saved_unresolved_forecast():
    record = dict(
        job={},
        ledger={},
        ledger_counts=dict(
            open_requests=[dict(id=5, charges=dict(model_controls=7))],
            unknown_partial_upper=dict(model_controls=8),
        ),
    )
    with pytest.raises(ValueError, match="uncertainty"):
        continuation.pending_overhead(record)


def test_source_and_cli_import_remain_lazy():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from src.gps.mppi_gps import local_coupling_multiround_continuation; "
                "assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
