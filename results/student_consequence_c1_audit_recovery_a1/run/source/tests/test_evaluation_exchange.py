"""Synthetic publication and acknowledgement faults without simulator/model work."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.gps.mppi_gps import evaluation_exchange as exchange
from src.gps.mppi_gps.durability import durable_json, file_binding


class Clock:
    def __init__(self):
        self.now = 10.0
        self.on_sleep = lambda: None

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.on_sleep()


@pytest.fixture
def packet(tmp_path, monkeypatch):
    segment = tmp_path / "evidence/loads"
    segment.mkdir(parents=True)
    for name in ("publications", "acks"):
        (tmp_path / name).mkdir()
    (segment / "raw.h5").write_bytes(b"closed synthetic payload")
    (segment / "ledger.jsonl").write_bytes(b'{"closed":true}\n')
    receipt = dict(
        schema="walker-durable-loads-v1",
        ownership=dict(boot_id="boot-one", boot_id_after="boot-one", ordinal=-1),
        files={name: file_binding(segment / name) for name in ("raw.h5", "ledger.jsonl")},
    )
    durable_json(segment / "receipt.json", receipt)
    clock = Clock()
    monkeypatch.setattr(
        exchange, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    )
    monkeypatch.setattr(exchange, "read_boot_id", lambda: "boot-one")
    journal = exchange.OwnerJournal(tmp_path / "owner.jsonl")
    yield tmp_path, segment, receipt, clock, journal
    journal.close()


def request_for(segment):
    return dict(
        schema="walker-s1.1-publication-v1",
        identifier="loads",
        ordinal=-1,
        segment="evidence/loads",
        worker_boot_id="boot-one",
        receipt_sha256=file_binding(segment / "receipt.json")["sha256"],
        files={
            name: file_binding(segment / name)
            for name in ("raw.h5", "ledger.jsonl", "receipt.json")
        },
    )


def acknowledge(out, **changes):
    request_path = out / "publications/loads.json"
    request = json.loads(request_path.read_text())
    ack = dict(
        schema="walker-s1.1-ack-v1",
        identifier="loads",
        request_sha256=file_binding(request_path)["sha256"],
        receipt_sha256=request["receipt_sha256"],
        worker_boot_id="boot-one",
        mirror_commit_sha256="a" * 64,
        coordinator_instance="synthetic-coordinator",
        coordinator_host="off-host",
    )
    ack.update(changes)
    durable_json(out / "acks/loads.json", ack)


def run_publication(packet, **kwargs):
    out, segment, _, _, journal = packet
    return exchange.publish_and_wait(out, segment, "loads", -1, "boot-one", 20, journal, **kwargs)


def test_closed_packet_waits_for_bound_off_host_ack(packet):
    out, _, _, clock, _ = packet
    clock.on_sleep = lambda: acknowledge(out)
    descriptor = run_publication(packet)
    assert descriptor["wait_seconds"] == pytest.approx(0.1)
    assert descriptor["mirror_commit_sha256"] == "a" * 64
    rows = [json.loads(line) for line in (out / "owner.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["publication_requested", "ack_accepted"]
    assert [row["id"] for row in rows] == [0, 1]
    with pytest.raises(FileExistsError):
        run_publication(packet)


def test_missing_ack_stops_at_finite_deadline_and_retains_request(packet):
    out, _, _, clock, _ = packet
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        run_publication(packet, timeout_seconds=0.2)
    assert clock.now < 10.4
    assert (out / "publications/loads.json").is_file()
    assert not (out / "acks/loads.json").exists()
    assert len((out / "owner.jsonl").read_text().splitlines()) == 1


def test_expired_global_budget_never_publishes(packet):
    out, segment, _, clock, journal = packet
    clock.now = 20
    with pytest.raises(TimeoutError, match="one-shot deadline"):
        exchange.publish_and_wait(out, segment, "loads", -1, "boot-one", 20, journal)
    assert not (out / "publications/loads.json").exists()


def test_ack_after_deadline_is_not_accepted(packet):
    out, _, _, clock, _ = packet

    def late():
        clock.now = 21
        acknowledge(out)

    clock.on_sleep = late
    with pytest.raises(TimeoutError, match="after a stop boundary"):
        run_publication(packet)
    assert len((out / "owner.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"request_sha256": "b" * 64},
        {"receipt_sha256": "b" * 64},
        {"worker_boot_id": "other-boot"},
        {"mirror_commit_sha256": "not-a-hash"},
        {"coordinator_host": ""},
        {"identifier": "other"},
    ],
)
def test_wrong_ack_is_preserved_but_not_accepted(packet, changes):
    out, _, _, clock, _ = packet
    clock.on_sleep = lambda: acknowledge(out, **changes)
    with pytest.raises(ValueError):
        run_publication(packet)
    assert (out / "acks/loads.json").is_file()
    assert len((out / "owner.jsonl").read_text().splitlines()) == 1


def test_boot_change_while_waiting_fails_closed(packet, monkeypatch):
    _, _, _, clock, _ = packet
    clock.on_sleep = lambda: monkeypatch.setattr(exchange, "read_boot_id", lambda: "new-boot")
    with pytest.raises(RuntimeError, match="boot identity changed"):
        run_publication(packet)


def test_boot_change_with_arriving_ack_is_not_accepted(packet, monkeypatch):
    out, _, _, clock, _ = packet

    def changed():
        acknowledge(out)
        monkeypatch.setattr(exchange, "read_boot_id", lambda: "new-boot")

    clock.on_sleep = changed
    with pytest.raises(RuntimeError, match="boot identity changed"):
        run_publication(packet)


@pytest.mark.parametrize("damage", ["payload", "extra", "symlink", "cross-binding", "receipt-boot"])
def test_exact_closed_packet_rejects_damage(packet, tmp_path, damage):
    _, segment, receipt, _, _ = packet
    request = request_for(segment)
    if damage == "payload":
        (segment / "raw.h5").write_bytes(b"changed")
    elif damage == "extra":
        (segment / "extra.txt").write_text("unexpected")
    elif damage == "symlink":
        original = segment / "raw.h5"
        external = tmp_path / "external"
        original.rename(external)
        original.symlink_to(external)
    else:
        (segment / "receipt.json").unlink()
        if damage == "cross-binding":
            receipt["files"]["raw.h5"]["sha256"] = "f" * 64
        else:
            receipt["ownership"]["boot_id_after"] = "new-boot"
        durable_json(segment / "receipt.json", receipt)
        request = request_for(segment)
    with pytest.raises(ValueError):
        exchange.verify_publication(segment, request)


@pytest.mark.parametrize("identifier", ["../outside", "", "/absolute", "x/y"])
def test_unsafe_identifier_is_rejected_before_writing(packet, identifier):
    out, segment, _, _, journal = packet
    with pytest.raises(ValueError, match="Unsafe"):
        exchange.publish_and_wait(out, segment, identifier, -1, "boot-one", 20, journal)
    assert not list((out / "publications").iterdir())


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_nonfinite_wait_is_rejected(packet, interval):
    with pytest.raises(ValueError, match="Finite positive"):
        run_publication(packet, timeout_seconds=interval)


def test_journal_fsyncs_parent_and_is_create_only(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(exchange, "fsync_directory", lambda path: calls.append(path))
    path = tmp_path / "journal.jsonl"
    journal = exchange.OwnerJournal(path)
    assert calls == [tmp_path]
    journal.close()
    with pytest.raises(FileExistsError):
        exchange.OwnerJournal(path)


def test_journal_parent_sync_failure_retains_failed_create(tmp_path, monkeypatch):
    def fail(path):
        raise OSError("synthetic directory sync failure")

    monkeypatch.setattr(exchange, "fsync_directory", fail)
    path = tmp_path / "journal.jsonl"
    with pytest.raises(OSError, match="directory sync failure"):
        exchange.OwnerJournal(path)
    assert path.is_file()
    with pytest.raises(FileExistsError):
        exchange.OwnerJournal(path)


def test_exchange_and_transport_import_without_model_or_array_libraries():
    code = (
        "import sys; from src.gps.mppi_gps import evaluation_exchange, "
        "source_evaluation_coordinator; "
        "assert not any(n in sys.modules for n in ('torch', 'mujoco', 'h5py', 'numpy'))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
