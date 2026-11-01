"""Create-only owner journal and off-host publication/acknowledgement contract."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

from .durability import durable_json, file_binding, fsync_directory, read_boot_id


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Unsafe publication identifier")


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_request(request: dict) -> None:
    """Check existing wire fields before using them as paths or byte bindings."""
    _identifier(request.get("identifier"))
    if request.get("schema") != "walker-s1.1-publication-v1":
        raise ValueError("Unexpected publication schema")
    ordinal = request.get("ordinal")
    if type(ordinal) is not int or ordinal < -1:
        raise ValueError("Invalid publication ordinal")
    identifier = request["identifier"]
    expected_path = "evidence/loads" if identifier == "loads" else "evidence/episodes/" + identifier
    if request.get("segment") != expected_path or (identifier == "loads") != (ordinal == -1):
        raise ValueError("Publication path/ordinal does not identify its fixed segment")
    if not isinstance(request.get("worker_boot_id"), str) or not request["worker_boot_id"]:
        raise ValueError("Missing publication boot owner")
    files = request.get("files")
    if not isinstance(files, dict) or set(files) != {"raw.h5", "ledger.jsonl", "receipt.json"}:
        raise ValueError("Unexpected publication file roster")
    for binding in files.values():
        if (
            not isinstance(binding, dict)
            or set(binding) != {"sha256", "bytes"}
            or not _hash(binding["sha256"])
            or type(binding["bytes"]) is not int
            or binding["bytes"] < 1
        ):
            raise ValueError("Invalid publication file binding")
    if (
        not _hash(request.get("receipt_sha256"))
        or request["receipt_sha256"] != files["receipt.json"]["sha256"]
    ):
        raise ValueError("Publication receipt hash differs from its file binding")


def validate_ack(ack: dict, request: dict, request_sha: str) -> None:
    expected = dict(
        schema="walker-s1.1-ack-v1",
        identifier=request["identifier"],
        request_sha256=request_sha,
        receipt_sha256=request["receipt_sha256"],
        worker_boot_id=request["worker_boot_id"],
    )
    if any(ack.get(key) != value for key, value in expected.items()) or not _hash(
        ack.get("mirror_commit_sha256")
    ):
        raise ValueError("Off-host acknowledgement does not bind this publication")
    if any(
        not isinstance(ack.get(key), str) or not ack[key]
        for key in ("coordinator_instance", "coordinator_host")
    ):
        raise ValueError("Off-host acknowledgement has no coordinator identity")


def sha(path: Path) -> str:
    return file_binding(path)["sha256"]


class OwnerJournal:
    def __init__(self, path: Path):
        self.path, self.sequence = path, 0
        self.stream = path.open("x")
        try:
            fsync_directory(path.parent)
        except BaseException:
            self.stream.close()
            raise

    def record(self, event: str, **fields) -> None:
        row = dict(id=self.sequence, event=event, monotonic=time.monotonic(), **fields)
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.sequence += 1

    def close(self) -> None:
        self.stream.close()


def verify_publication(directory: Path, request: dict) -> dict:
    """Verify the exact closed three-file packet; no HDF/model import or replay."""
    validate_request(request)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Publication directory must be a real directory")
    if {p.name for p in directory.iterdir()} != set(request["files"]):
        raise ValueError("Missing or extra mirrored segment file")
    for name, binding in request["files"].items():
        path = directory / name
        if path.is_symlink() or file_binding(path) != binding:
            raise ValueError("Closed publication bytes differ: " + name)
    if sha(directory / "receipt.json") != request["receipt_sha256"]:
        raise ValueError("Publication receipt owner differs")
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt.get("schema") not in ("walker-durable-episode-v1", "walker-durable-loads-v1"):
        raise ValueError("Unexpected closed publication receipt schema")
    owner = receipt["ownership"]
    if (
        owner["boot_id"] != request["worker_boot_id"]
        or owner.get("boot_id_after") != request["worker_boot_id"]
    ):
        raise ValueError("Publication boot owner differs")
    if owner.get("ordinal") != request["ordinal"]:
        raise ValueError("Publication ordinal owner differs")
    expected_schema = (
        "walker-durable-loads-v1"
        if request["identifier"] == "loads"
        else "walker-durable-episode-v1"
    )
    if receipt["schema"] != expected_schema or receipt.get("files") != {
        name: request["files"][name] for name in ("raw.h5", "ledger.jsonl")
    }:
        raise ValueError("Publication receipt does not bind its exact data files")
    return receipt


def publish_and_wait(
    out: Path,
    segment: Path,
    identifier: str,
    ordinal: int,
    boot_id: str,
    deadline: float,
    journal: OwnerJournal,
    *,
    timeout_seconds: float = 60,
    poll_seconds: float = 0.1,
) -> dict:
    _identifier(identifier)
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in (deadline, timeout_seconds, poll_seconds)
    ):
        raise ValueError("Finite positive acknowledgement deadlines/poll intervals required")
    if read_boot_id() != boot_id:
        raise RuntimeError("Worker boot identity changed")
    if time.monotonic() >= deadline:
        raise TimeoutError("Publication attempted after the one-shot deadline")
    request_path = out / "publications" / (identifier + ".json")
    request = dict(
        schema="walker-s1.1-publication-v1",
        identifier=identifier,
        ordinal=ordinal,
        segment=str(segment.relative_to(out)),
        worker_boot_id=boot_id,
        receipt_sha256=sha(segment / "receipt.json"),
        files={
            name: file_binding(segment / name)
            for name in ("raw.h5", "ledger.jsonl", "receipt.json")
        },
    )
    verify_publication(segment, request)
    durable_json(request_path, request)
    request_sha = sha(request_path)
    journal.record(
        "publication_requested", identifier=identifier, ordinal=ordinal, request_sha256=request_sha
    )
    started = time.monotonic()
    ack_path = out / "acks" / (identifier + ".json")
    while not ack_path.exists():
        if read_boot_id() != boot_id:
            raise RuntimeError("Worker boot identity changed")
        if time.monotonic() >= min(deadline, started + timeout_seconds):
            raise TimeoutError("Closed episode off-host acknowledgement deadline exhausted")
        time.sleep(poll_seconds)
    if ack_path.is_symlink():
        raise ValueError("Acknowledgement must not be a symlink")
    ack = json.loads(ack_path.read_text())
    validate_ack(ack, request, request_sha)
    if read_boot_id() != boot_id:
        raise RuntimeError("Worker boot identity changed")
    if time.monotonic() >= min(deadline, started + timeout_seconds):
        raise TimeoutError("Off-host acknowledgement arrived after a stop boundary")
    descriptor = dict(
        segment=request["segment"],
        receipt_sha256=request["receipt_sha256"],
        request=dict(file=str(request_path.relative_to(out)), sha256=request_sha),
        ack=dict(file=str(ack_path.relative_to(out)), sha256=sha(ack_path)),
        mirror_commit_sha256=ack["mirror_commit_sha256"],
        wait_seconds=time.monotonic() - started,
    )
    journal.record("ack_accepted", identifier=identifier, ordinal=ordinal, **descriptor)
    return descriptor
