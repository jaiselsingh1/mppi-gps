"""Independent segmented-evaluation fixtures; no simulator or model arithmetic."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from copy import deepcopy

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.audits import native
from src.gps.mppi_gps.audits.source_evaluation import (
    EvaluationAudit,
    counter_sum,
    ordered_panel,
    prove_episode_owner,
    verify_closed_segment,
    verify_coordinator,
    verify_cumulative,
    verify_exchange,
    verify_owner_journal,
)


def test_panel_object_order_is_separate_from_frozen_execution_order(tmp_path):
    order = ["reference", "incoming_s0", "mppi_s0"]
    records = json.loads(json.dumps({name: dict(point=name) for name in order}, sort_keys=True))
    assert list(records) != order
    audit = EvaluationAudit(tmp_path)
    assert [name for name, _ in ordered_panel(audit, records, order)] == order
    for invalid in ({"reference": {}}, records | {"unexpected": {}}):
        with pytest.raises(AssertionError, match="Every actor"):
            ordered_panel(audit, invalid, order)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return native.sha(path)


def binding(path):
    return dict(sha256=native.sha(path), bytes=path.stat().st_size)


def descriptor(value):
    value = np.asarray(value)
    return dict(
        shape=list(value.shape),
        dtype=value.dtype.str,
        encoding="array",
        sha256=hashlib.sha256(value.tobytes()).hexdigest(),
    )


def closed_episode(path):
    """A single synthetic fall tests terminal-once and unexecuted absorbing cost."""
    path.mkdir(parents=True)
    states = np.zeros((2, 19))
    states[:, 2] = [1, 0.7]
    states[1, 0], states[:, 10] = 0.008, 1.5
    actions = np.zeros((1, 6), dtype=np.float32)
    warm, integration = np.zeros((2, 9)), np.zeros((2, 91))
    values = dict(
        states=states,
        actions=actions,
        warmstarts=warm,
        integrations=integration,
        observations=native.obs(states[:-1]),
        velocities=states[1:, 10],
        costs=np.array([25.0]),
        terminal_cost=np.array(20.0),
        terminal_adjustment=np.array(0.0),
        absorbing_cost=np.array(4995.0),
        infos=np.array(
            json.dumps(
                dict(
                    rows=[
                        dict(controller=dict(policy_hash="policy", controller="standalone_torque"))
                    ]
                )
            )
        ),
    )
    with h5py.File(path / "raw.h5", "w") as stream:
        group = stream.create_group("arrays/episode")
        group.attrs["tag"] = "episode"
        arrays = {}
        for key, value in values.items():
            meta = descriptor(value)
            if key == "infos":
                meta["encoding"] = "json"
            dataset = group.create_dataset(
                key, data=np.char.encode(value, "utf-8") if value.dtype.kind == "U" else value
            )
            dataset.attrs.update(
                numpy_dtype=meta["dtype"], encoding=meta["encoding"], sha256=meta["sha256"]
            )
            arrays[key] = meta
    raw = dict(file="raw.h5", group="arrays/episode", tag="episode", arrays=arrays)
    episode = dict(
        tag="primary_mppi_s0_00",
        seed=1270000000,
        steps=1,
        cost=5020.0,
        mae=0.0,
        survived=False,
        success=False,
        complete=True,
        terminal_cost=20.0,
        terminal_adjustment=0.0,
        absorbing_cost=4995.0,
        raw=raw,
    )
    inputs = {
        key: descriptor(value)
        for key, value in dict(
            action=actions[0].astype(float),
            state=states[0],
            warmstart=warm[0],
            integration=integration[0],
        ).items()
    }
    outputs = {
        key: descriptor(value)
        for key, value in dict(
            state=states[1],
            warmstart=warm[1],
            integration=integration[1],
            observation=native.obs(states[1]),
            cost=25.0,
        ).items()
    }
    context = dict(point="mppi_s0", cohort="primary", role="source_standalone")
    work = dict(
        requested_controls=1,
        completed_controls=1,
        requested_live_controls=1,
        completed_live_controls=1,
        requested_physics_substeps=4,
        completed_physics_substeps=4,
        live_queries=1,
        measurement_requests=1,
        requested_neural_rows=1,
        completed_neural_rows=1,
        resets=1,
        episodes=1,
        snapshots=0,
        neural_rows=1,
        backward_rows=0,
        optimizer_steps=0,
        queries=1,
        physics_frames=4,
        wall_seconds=1.0,
    )
    rows = [
        dict(event="runtime_open"),
        dict(event="reset_complete", tag=episode["tag"], seed=episode["seed"]),
        dict(event="measurement_requested", measure_kind="neural_rows", amount=1, context=context),
        dict(
            event="measurement_complete",
            measure_kind="neural_rows",
            amount=1,
            context=context,
            request_id="event-000000002",
        ),
        dict(
            event="query_requested",
            mode="live",
            requested_controls=1,
            requested_physics_substeps=4,
            query_id="q0",
            episode=episode["tag"],
            step=0,
            context=context,
            inputs=dict(arrays=inputs),
        ),
        dict(
            event="query_complete",
            mode="live",
            completed_controls=1,
            completed_physics_substeps=4,
            query_id="q0",
            request_id="event-000000004",
            result=dict(arrays=outputs),
        ),
        dict(event="episode_complete", **episode),
        dict(event="runtime_close", totals=work),
    ]
    for index, row in enumerate(rows):
        row.update(id=f"event-{index:09d}", kind=row["event"], wall_seconds=index * 0.1)
    (path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = dict(
        schema="walker-durable-episode-v1",
        ownership=dict(cumulative_before={}, boot_id="boot", boot_id_after="boot"),
        episode=episode,
        work=work,
        cumulative=work,
        files={name: binding(path / name) for name in ("raw.h5", "ledger.jsonl")},
    )
    sha = write_json(path / "receipt.json", receipt)
    return receipt, sha


def close_audit(a):
    for handle, _, _ in a.hdfs.values():
        handle.close()


def test_closed_segment_recomputes_fall_cost_and_only_owned_raw_alias(tmp_path):
    segment = tmp_path / "run/evidence/episodes/000_primary_mppi_s0_00"
    receipt, sha = closed_episode(segment)
    a = EvaluationAudit(tmp_path / "run")
    actual, stage = verify_closed_segment(a, segment, sha, {})
    event, arrays = a.episode(stage, receipt["episode"]["tag"])
    assert event["cost"] == 5020 and not event["success"] and not event["survived"]
    assert arrays["states"].shape == (2, 19)
    assert actual["cumulative"]["completed_live_controls"] == 1
    assert set(a.hdfs) == {(segment / "raw.h5").resolve()}
    close_audit(a)


@pytest.mark.parametrize("change", ["raw", "unclosed", "extra", "cost"])
def test_closed_segment_tampering_cannot_become_a_complete_outcome(tmp_path, change):
    segment = tmp_path / "segment"
    receipt, sha = closed_episode(segment)
    if change == "raw":
        with h5py.File(segment / "raw.h5", "a") as stream:
            stream["arrays/episode/actions"][0, 0] = 0.5
    elif change == "extra":
        (segment / "extra").write_text("unsafe")
    elif change == "unclosed":
        path = segment / "ledger.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        receipt["files"]["ledger.jsonl"] = binding(path)
        sha = write_json(segment / "receipt.json", receipt)
    else:
        receipt["episode"]["cost"] -= 20
        sha = write_json(segment / "receipt.json", receipt)
    with pytest.raises(AssertionError):
        verify_closed_segment(EvaluationAudit(tmp_path), segment, sha, {})


def test_rotation_cannot_replenish_counts_or_change_boot(tmp_path):
    a = EvaluationAudit(tmp_path)
    before = dict(requested_live_controls=319999, wall_seconds=2.0)
    work = dict(requested_live_controls=1, wall_seconds=0.5)
    receipt = dict(
        ownership=dict(cumulative_before=before), work=work, cumulative=counter_sum(before, work)
    )
    assert verify_cumulative(a, receipt, before)["requested_live_controls"] == 320000
    wrong = deepcopy(receipt)
    wrong["cumulative"]["requested_live_controls"] = 1
    with pytest.raises(AssertionError):
        verify_cumulative(a, wrong, before)
    with pytest.raises(AssertionError, match="boot"):
        prove_episode_owner(a, dict(boot_id="before", boot_id_after="after"), {})


def exchange_fixture(tmp_path):
    run, mirror = tmp_path / "run", tmp_path / "mirror"
    identifier = "000_primary_mppi_s0_00"
    relative = "evidence/episodes/" + identifier
    receipt, sha = closed_episode(run / relative)
    (mirror / relative).parent.mkdir(parents=True)
    shutil.copytree(run / relative, mirror / relative)
    files = receipt["files"] | {"receipt.json": binding(run / relative / "receipt.json")}
    request = dict(
        schema="walker-s1.1-publication-v1",
        identifier=identifier,
        ordinal=0,
        segment=relative,
        worker_boot_id="boot",
        receipt_sha256=sha,
        files=files,
    )
    request_file = "publications/" + identifier + ".json"
    request_sha = write_json(run / request_file, request)
    commit = dict(
        schema="walker-s1.1-mirror-commit-v1",
        identifier=identifier,
        ordinal=0,
        segment=relative,
        worker_boot_id="boot",
        request_sha256=request_sha,
        receipt_sha256=sha,
        files=files,
        coordinator_instance="instance",
        coordinator_host="Mac",
        copied_bytes=sum(row["bytes"] for row in files.values()),
        copy_wall_seconds=0.1,
        local_directory=str((mirror / relative).resolve()),
    )
    commit_sha = write_json(mirror / "local_coordinator/commits" / (identifier + ".json"), commit)
    ack = dict(
        schema="walker-s1.1-ack-v1",
        identifier=identifier,
        request_sha256=request_sha,
        receipt_sha256=sha,
        worker_boot_id="boot",
        mirror_commit_sha256=commit_sha,
        coordinator_instance="instance",
        coordinator_host="Mac",
    )
    ack_file = "acks/" + identifier + ".json"
    ack_sha = write_json(run / ack_file, ack)
    write_json(mirror / "local_coordinator/acks" / (identifier + ".json"), ack)
    descriptor = dict(
        segment=relative,
        receipt_sha256=sha,
        request=dict(file=request_file, sha256=request_sha),
        ack=dict(file=ack_file, sha256=ack_sha),
        mirror_commit_sha256=commit_sha,
        wait_seconds=0.2,
    )
    return run, mirror, identifier, receipt, descriptor


def test_independent_mirror_joins_all_three_files_and_refuses_mutated_copy(tmp_path):
    run, mirror, identifier, receipt, saved = exchange_fixture(tmp_path)
    verify_exchange(EvaluationAudit(run), run, mirror, saved, receipt, identifier, 0)
    with (mirror / saved["segment"] / "raw.h5").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(AssertionError, match="Pinned hash"):
        verify_exchange(EvaluationAudit(run), run, mirror, saved, receipt, identifier, 0)


def test_self_mirror_and_rebound_ack_are_rejected(tmp_path):
    run, mirror, identifier, receipt, saved = exchange_fixture(tmp_path)
    with pytest.raises(AssertionError, match="distinct"):
        verify_exchange(EvaluationAudit(run), run, run, saved, receipt, identifier, 0)
    path = run / saved["ack"]["file"]
    ack = json.loads(path.read_text())
    ack["request_sha256"] = "other publication"
    saved["ack"]["sha256"] = write_json(path, ack)
    with pytest.raises(AssertionError, match="acknowledgement"):
        verify_exchange(EvaluationAudit(run), run, mirror, saved, receipt, identifier, 0)


def test_coordinator_accounting_requires_off_host_identity_and_complete_mirror_metadata(tmp_path):
    run, mirror, identifier, receipt, saved = exchange_fixture(tmp_path)
    for name in (
        "launch_receipt.json",
        "worker_started.json",
        "worker_process.json",
        "result.json",
        "worker_exit.json",
        "owner_exit.json",
        "owner_ledger.jsonl",
    ):
        write_json(run / name, {"file": name})
        (mirror / "global").mkdir(exist_ok=True)
        shutil.copyfile(run / name, mirror / "global" / name)
    row = dict(
        identifier=identifier,
        request_sha256=saved["request"]["sha256"],
        receipt_sha256=saved["receipt_sha256"],
        mirror_commit_sha256=saved["mirror_commit_sha256"],
        ack_sha256=saved["ack"]["sha256"],
        segment=saved["segment"],
        work=receipt["work"],
    )
    coordinator = dict(
        schema="walker-s1.1-coordinator-exit-v1",
        execution_complete=True,
        error=None,
        worker_boot_id="boot",
        new_physics=0,
        new_neural_rows=0,
        new_optimizer_steps=0,
        wall_seconds=1.0,
        records=[row],
        calls=[dict(exit_code=0, wall_seconds=0.1)],
    )
    write_json(mirror / "coordinator_exit.json", coordinator)
    launch = dict(boot_id="boot", host="robopil5090")
    accounting = verify_coordinator(EvaluationAudit(run), run, mirror, [saved], launch)
    assert accounting["copied_closed_packet_bytes"] > 0
    assert accounting["copy_wall_seconds"] == 0.1
    with pytest.raises(AssertionError, match="Off-host"):
        verify_coordinator(EvaluationAudit(run), run, mirror, [saved], launch | dict(host="Mac"))
    coordinator["records"] = []
    write_json(mirror / "coordinator_exit.json", coordinator)
    with pytest.raises(AssertionError, match="transport"):
        verify_coordinator(EvaluationAudit(run), run, mirror, [saved], launch)


def test_owner_state_machine_forbids_next_start_before_previous_ack(tmp_path):
    run, mirror, identifier, receipt, saved = exchange_fixture(tmp_path)
    receipt_path = run / saved["segment"] / "receipt.json"
    records = []
    previous = None
    descriptors = []
    for index in range(2):
        identifier = "loads" if index == 0 else "episode"
        saved = deepcopy(saved)
        saved["segment"] = "evidence/" + identifier
        saved["ack"]["sha256"] = str(index)
        write_json(run / saved["segment"] / "receipt.json", receipt)
        descriptors.append(saved)
        records.extend(
            [
                dict(
                    event="segment_started",
                    ordinal=index - 1,
                    identifier=identifier,
                    previous_ack_sha256=previous,
                ),
                dict(
                    event="segment_closed",
                    ordinal=index - 1,
                    identifier=identifier,
                    receipt_sha256=saved["receipt_sha256"],
                    cumulative=receipt["cumulative"],
                ),
                dict(
                    event="publication_requested",
                    ordinal=index - 1,
                    identifier=identifier,
                    request_sha256=saved["request"]["sha256"],
                ),
                dict(event="ack_accepted", ordinal=index - 1, identifier=identifier, **saved),
            ]
        )
        if index:
            # Result-only labels are added after the immutable ACK event.
            saved.update(ordinal=0, point="reference", panel="primary", tag="primary_reference_00")
        previous = str(index)
    records.append(dict(event="worker_finished", execution_complete=True, work=receipt["work"]))
    for index, row in enumerate(records):
        row.update(id=index, monotonic=index * 0.1)
    path = run / "owner_ledger.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    verify_owner_journal(EvaluationAudit(run), run, descriptors, receipt["work"])
    records[4]["previous_ack_sha256"] = None
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(AssertionError, match="waits"):
        verify_owner_journal(EvaluationAudit(run), run, descriptors, receipt["work"])
    assert receipt_path.exists()


def test_cli_failed_audit_is_create_only_and_saved_only(tmp_path):
    run, mirror, out = (tmp_path / name for name in ("run", "mirror", "audit"))
    run.mkdir()
    mirror.mkdir()
    command = [
        sys.executable,
        "scripts/audits/student_source_evaluation.py",
        "--run",
        str(run),
        "--mirror",
        str(mirror),
        "--out",
        str(out),
        "--config-sha",
        "missing",
        "--seal-sha",
        "missing",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["passed"] is False and receipt["new_controls"] == receipt["new_neural_rows"] == 0
    original = (out / "receipt.json").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (out / "receipt.json").read_bytes() == original
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.gps.mppi_gps.audits.source_evaluation; assert not any(n == 'mujoco' or n.startswith('src.envs') or n.startswith('src.gps.mppi_gps.experiments') or n == 'src.gps.mppi_gps.durability' for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
