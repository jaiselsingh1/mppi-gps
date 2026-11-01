"""Synthetic filesystem/crash tests; no model, fitting or simulator work."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps import durability as durable

BOOT = "00000000-0000-0000-0000-000000000001"


def segment_inputs(path, *, before=None, steps=3):
    path.mkdir()
    values = np.arange(steps, dtype=np.float64)
    with h5py.File(path / "raw.h5", "x") as stream:
        stream.create_dataset("episode/costs", data=values)
    raw = {
        "file": "raw.h5",
        "group": "episode",
        "arrays": {
            "costs": {
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "shape": [steps],
                "dtype": "<f8",
                "encoding": "array",
            }
        },
    }
    episode = {
        "tag": "primary_sac_reference_s0_00",
        "seed": 1270000000,
        "steps": steps,
        "complete": True,
        "cost": 1.25,
        "success": False,
        "survived": True,
        "mae": 1.0,
        "raw": raw,
    }
    work = {
        "episodes": 1,
        "completed_live_controls": steps,
        "completed_neural_rows": steps,
        "completed_physics_substeps": 4 * steps,
        "wall_seconds": 0.02,
    }
    before = dict(
        before
        or {
            "episodes": 0,
            "completed_live_controls": 0,
            "completed_neural_rows": 0,
            "completed_physics_substeps": 0,
            "completed_checkpoint_loads": 10,
            "wall_seconds": 0.01,
        }
    )
    after = {key: before.get(key, 0) + work.get(key, 0) for key in set(before) | set(work)}
    owner = {key: "a" * 64 for key in durable.HASH_FIELDS}
    owner.update(
        ordinal=0,
        point="sac_reference_s0",
        panel="primary",
        seed=episode["seed"],
        tag=episode["tag"],
        boot_id=BOOT,
        boot_id_after=BOOT,
        owner_pid=1234,
        cumulative_before=before,
    )
    rows = [
        {"event": "runtime_open"},
        {"event": "episode_complete", **episode},
        {"event": "runtime_close", "totals": work},
    ]
    (path / "ledger.jsonl").write_text(
        "".join(
            json.dumps(dict(row, id=f"event-{index:09d}")) + "\n" for index, row in enumerate(rows)
        )
    )
    return episode, owner, work, after


def snapshots(path):
    return {entry.name: entry.read_bytes() for entry in path.iterdir()}


def test_import_boundary_is_standard_library_only():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.gps.mppi_gps.durability; assert not any(x in sys.modules for x in ('torch','mujoco','h5py','numpy'))",
        ],
        check=True,
    )


def test_atomic_json_never_overwrites_existing_bytes(tmp_path):
    path = tmp_path / "receipt.json"
    durable.durable_json(path, {"committed": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        durable.durable_json(path, {"committed": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_json_fsync_failure_cannot_create_receipt(tmp_path, monkeypatch):
    def fail(_):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(durable.os, "fsync", fail)
    with pytest.raises(OSError, match="fsync"):
        durable.durable_json(tmp_path / "receipt.json", {"value": 1})
    assert list(tmp_path.iterdir()) == []


def test_atomic_json_link_failure_preserves_racing_target(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"

    def race(source, target):
        target.write_bytes(b"a different owner committed first")
        raise FileExistsError(target)

    monkeypatch.setattr(durable.os, "link", race)
    with pytest.raises(FileExistsError):
        durable.durable_json(path, {"value": 1})
    assert path.read_bytes() == b"a different owner committed first"
    assert list(tmp_path.iterdir()) == [path]


def test_directory_commit_failure_leaves_no_acknowledged_target(tmp_path, monkeypatch):
    def fail(_):
        raise OSError("synthetic directory commit failure")

    monkeypatch.setattr(durable, "fsync_directory", fail)
    with pytest.raises(OSError, match="directory commit"):
        durable.durable_json(tmp_path / "receipt.json", {"value": 1})
    assert list(tmp_path.iterdir()) == []


def test_failed_commit_never_removes_another_owners_replacement(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"

    def replace_then_fail(_):
        path.unlink()
        path.write_bytes(b"replacement belongs to another owner")
        raise OSError("directory commit failed")

    monkeypatch.setattr(durable, "fsync_directory", replace_then_fail)
    with pytest.raises(OSError, match="directory commit"):
        durable.durable_json(path, {"value": 1})
    assert path.read_bytes() == b"replacement belongs to another owner"


def test_rollback_pins_original_inode_until_ownership_decision_finishes(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    streams = []
    original_fdopen = durable.os.fdopen

    def capture_stream(*args, **kwargs):
        stream = original_fdopen(*args, **kwargs)
        streams.append(stream)
        return stream

    def replace_then_fail(_):
        stream = streams[0]
        assert not stream.closed
        owned = durable.os.fstat(stream.fileno())
        path.unlink()
        assert durable.os.fstat(stream.fileno()).st_nlink == 0
        path.write_bytes(b"a replacement whose inode cannot alias the open original")
        replacement = path.stat()
        assert (replacement.st_dev, replacement.st_ino) != (owned.st_dev, owned.st_ino)
        raise OSError("directory commit failed with pinned inode")

    monkeypatch.setattr(durable.os, "fdopen", capture_stream)
    monkeypatch.setattr(durable, "fsync_directory", replace_then_fail)
    with pytest.raises(OSError, match="pinned inode"):
        durable.durable_json(path, {"value": 1})
    assert streams[0].closed
    assert path.read_bytes() == b"a replacement whose inode cannot alias the open original"
    assert list(tmp_path.iterdir()) == [path]


def test_nonfinite_json_is_rejected_before_creating_files(tmp_path):
    with pytest.raises(ValueError):
        durable.durable_json(tmp_path / "receipt.json", {"value": float("nan")})
    assert list(tmp_path.iterdir()) == []


def test_seal_and_mirror_preserve_closed_file_bytes_and_ownership(tmp_path):
    source, target = tmp_path / "source", tmp_path / "mirror"
    episode, owner, work, total = segment_inputs(source)
    old = snapshots(source)
    receipt = durable.seal_segment(source, episode, owner, work, total)
    for name, data in old.items():
        assert (source / name).read_bytes() == data
    digest = durable.file_binding(source / "receipt.json")["sha256"]
    assert durable.verify_segment(source, digest) == receipt
    assert durable.mirror_segment(source, target, digest) == receipt
    assert snapshots(target) == snapshots(source)
    owner["point"] = "caller-mutated-after-commit"
    assert durable.verify_segment(source, digest)["ownership"]["point"] == "sac_reference_s0"
    with pytest.raises(FileExistsError):
        durable.mirror_segment(source, target, digest)
    with pytest.raises(durable.DurabilityError, match="immutable source"):
        durable.mirror_segment(source, source / "nested", digest)


def test_segment_fsync_failure_preserves_previous_committed_segment(tmp_path, monkeypatch):
    prior, active = tmp_path / "prior", tmp_path / "active"
    durable.seal_segment(prior, *segment_inputs(prior))
    before = snapshots(prior)
    arguments = segment_inputs(active)

    def fail(_):
        raise OSError("synthetic closed-file fsync failure")

    monkeypatch.setattr(durable, "_fsync_file", fail)
    with pytest.raises(OSError, match="closed-file fsync"):
        durable.seal_segment(active, *arguments)
    assert snapshots(prior) == before
    assert not (active / "receipt.json").exists()
    durable.verify_segment(prior)


def test_truncated_active_hdf_cannot_damage_previous_segment(tmp_path):
    prior, active = tmp_path / "prior", tmp_path / "active"
    durable.seal_segment(prior, *segment_inputs(prior))
    before = snapshots(prior)
    arguments = segment_inputs(active)
    data = (active / "raw.h5").read_bytes()
    (active / "raw.h5").write_bytes(data[:-16])
    with pytest.raises(durable.DurabilityError, match="truncated"):
        durable.seal_segment(active, *arguments)
    assert snapshots(prior) == before
    assert not (active / "receipt.json").exists()
    durable.verify_segment(prior)


def test_unclosed_latest_hdf_writer_is_not_sealed(tmp_path):
    active = tmp_path / "active"
    arguments = segment_inputs(active)
    (active / "raw.h5").unlink()
    with h5py.File(active / "raw.h5", "x", libver="latest") as writer:
        writer.create_dataset("episode/costs", data=np.arange(3, dtype=np.float64))
        writer.flush()
        with pytest.raises(durable.DurabilityError, match="open-writer"):
            durable.seal_segment(active, *arguments)
    assert not (active / "receipt.json").exists()


def test_parent_directory_failure_prevents_seal_and_preserves_prior(tmp_path, monkeypatch):
    prior, active = tmp_path / "prior", tmp_path / "active"
    durable.seal_segment(prior, *segment_inputs(prior))
    previous = snapshots(prior)
    arguments = segment_inputs(active)
    original = durable.fsync_directory

    def fail_parent(path):
        if path == active.parent:
            raise OSError("segment parent fsync failed")
        original(path)

    monkeypatch.setattr(durable, "fsync_directory", fail_parent)
    with pytest.raises(OSError, match="parent fsync"):
        durable.seal_segment(active, *arguments)
    assert not (active / "receipt.json").exists()
    assert snapshots(prior) == previous
    durable.verify_segment(prior)


def test_postverification_parent_failure_returns_no_acknowledgement(tmp_path, monkeypatch):
    active = tmp_path / "active"
    arguments = segment_inputs(active)
    original, parent_calls, acknowledged = durable.fsync_directory, [], []

    def fail_second_parent(path):
        if path == active.parent:
            parent_calls.append(path)
            if len(parent_calls) == 2:
                raise OSError("final parent fsync failed")
        original(path)

    monkeypatch.setattr(durable, "fsync_directory", fail_second_parent)
    with pytest.raises(OSError, match="final parent"):
        receipt = durable.seal_segment(active, *arguments)
        acknowledged.append(receipt)
    assert acknowledged == []
    assert (active / "receipt.json").exists()  # Retain unacknowledged evidence.
    durable.verify_segment(active)


def test_mirror_parent_failure_never_acknowledges_copy(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    durable.seal_segment(source, *segment_inputs(source))
    previous = snapshots(source)
    digest = durable.file_binding(source / "receipt.json")["sha256"]
    original, parent_calls, acknowledged = durable.fsync_directory, [], []

    def fail_second_parent(path):
        if path == target.parent:
            parent_calls.append(path)
            if len(parent_calls) == 2:
                raise OSError("mirror parent fsync failed")
        original(path)

    monkeypatch.setattr(durable, "fsync_directory", fail_second_parent)
    with pytest.raises(OSError, match="mirror parent"):
        receipt = durable.mirror_segment(source, target, digest)
        acknowledged.append(receipt)
    assert acknowledged == []
    assert snapshots(source) == previous
    assert snapshots(target) == previous  # Verified bytes are retained, not retried.


@pytest.mark.parametrize("change", ["extra", "missing", "symlink"])
def test_segment_roster_rejects_unsafe_files(tmp_path, change):
    path = tmp_path / "segment"
    arguments = segment_inputs(path)
    durable.seal_segment(path, *arguments)
    if change == "extra":
        (path / "unexpected.json").write_text("{}")
    elif change == "missing":
        (path / "ledger.jsonl").unlink()
    else:
        original = path / "raw.h5"
        destination = tmp_path / "elsewhere.h5"
        original.rename(destination)
        original.symlink_to(destination)
    with pytest.raises(durable.DurabilityError):
        durable.verify_segment(path)


def test_receipt_or_raw_hash_mismatch_is_not_mirrored(tmp_path):
    path = tmp_path / "segment"
    durable.seal_segment(path, *segment_inputs(path))
    digest = durable.file_binding(path / "receipt.json")["sha256"]
    with pytest.raises(durable.DurabilityError, match="Receipt SHA"):
        durable.mirror_segment(path, tmp_path / "bad-mirror", "0" * 64)
    assert not (tmp_path / "bad-mirror").exists()
    with (path / "raw.h5").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(durable.DurabilityError, match="file bytes"):
        durable.verify_segment(path, digest)


def test_unclosed_or_multiple_episode_ledger_cannot_be_sealed(tmp_path):
    path = tmp_path / "segment"
    arguments = segment_inputs(path)
    ledger = path / "ledger.jsonl"
    lines = ledger.read_text().splitlines(keepends=True)
    ledger.write_text("".join(lines[:-1]))
    with pytest.raises(durable.DurabilityError, match="closed ledger"):
        durable.seal_segment(path, *arguments)
    duplicate = json.loads(lines[1])
    duplicate["id"] = "event-000000002"
    close = json.loads(lines[2])
    close["id"] = "event-000000003"
    ledger.write_text("".join(lines[:2]) + json.dumps(duplicate) + "\n" + json.dumps(close) + "\n")
    with pytest.raises(durable.DurabilityError, match="exactly one"):
        durable.seal_segment(path, *arguments)


def test_setup_loads_are_not_a_fake_episode(tmp_path):
    path = tmp_path / "segment"
    episode, owner, work, total = segment_inputs(path)
    episode["steps"] = 0
    work["episodes"] = 0
    with pytest.raises(durable.DurabilityError, match="executed controls"):
        durable.seal_segment(path, episode, owner, work, total)


def test_cumulative_work_cannot_reset_when_meter_segment_rotates(tmp_path):
    path = tmp_path / "segment"
    episode, owner, work, total = segment_inputs(path)
    total["completed_checkpoint_loads"] = 0
    with pytest.raises(durable.DurabilityError, match="does not add"):
        durable.seal_segment(path, episode, owner, work, total)


def test_receipt_must_match_native_episode_and_closed_work(tmp_path):
    path = tmp_path / "segment"
    episode, owner, work, total = segment_inputs(path)
    episode["cost"] = 99
    with pytest.raises(durable.DurabilityError, match="closed ledger: cost"):
        durable.seal_segment(path, episode, owner, work, total)


def test_boot_reader_and_guard_are_injectable(tmp_path):
    value = durable.read_boot_id(tmp_path / "synthetic-boot", reader=lambda _: BOOT + "\n")
    assert value == BOOT
    assert durable.require_boot_id(BOOT, reader=lambda: BOOT) == BOOT
    with pytest.raises(durable.DurabilityError, match="changed"):
        durable.require_boot_id(BOOT, reader=lambda: "00000000-0000-0000-0000-000000000002")
    with pytest.raises(durable.DurabilityError, match="Invalid"):
        durable.read_boot_id(reader=lambda _: "not-a-uuid")


def test_boot_change_during_segment_prevents_receipt(tmp_path):
    path = tmp_path / "segment"
    episode, owner, work, total = segment_inputs(path)
    owner["boot_id_after"] = "00000000-0000-0000-0000-000000000002"
    with pytest.raises(durable.DurabilityError, match="boot identity changed"):
        durable.seal_segment(path, episode, owner, work, total)
    assert not (path / "receipt.json").exists()


def test_worker_started_hash_is_preserved_as_additional_ownership(tmp_path):
    path = tmp_path / "segment"
    episode, owner, work, total = segment_inputs(path)
    owner["worker_started_sha256"] = "b" * 64
    receipt = durable.seal_segment(path, episode, owner, work, total)
    assert receipt["ownership"]["worker_started_sha256"] == "b" * 64


def test_global_budget_rejects_overrun_atomically_across_segments():
    budget = durable.CumulativeBudget({"episodes": 2, "live_controls": 5})
    assert budget.add({"episodes": 1, "live_controls": 3}) == {"episodes": 1, "live_controls": 3}
    before = budget.totals()
    with pytest.raises(durable.DurabilityError, match="across segments"):
        budget.add({"episodes": 1, "live_controls": 3})
    assert budget.totals() == before
    assert budget.remaining() == {"episodes": 1, "live_controls": 2}
    with pytest.raises(durable.DurabilityError, match="unknown"):
        budget.add({"unaccounted": 1})
    with pytest.raises(durable.DurabilityError):
        budget.add({"episodes": True})


def test_receipt_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "segment"
    durable.seal_segment(path, *segment_inputs(path))
    receipt = path / "receipt.json"
    receipt.write_bytes(b'{"schema":"one","schema":"two"}')
    with pytest.raises(durable.DurabilityError, match="Duplicate JSON"):
        durable.verify_segment(path)
