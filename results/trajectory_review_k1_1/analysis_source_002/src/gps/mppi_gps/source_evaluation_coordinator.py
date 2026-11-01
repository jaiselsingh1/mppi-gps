"""Off-host backup coordinator for S1.1; no policy or simulator imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .durability import durable_json, file_binding, fsync_directory, read_boot_id, verify_segment
from .evaluation_exchange import validate_ack, validate_request, verify_publication


def sha(path: Path) -> str:
    return file_binding(path)["sha256"]


def process_alive(identity: dict) -> bool:
    if (
        type(identity.get("pid")) is not int
        or identity["pid"] <= 0
        or not isinstance(identity.get("start_ticks"), str)
    ):
        return False
    try:
        data = Path(f"/proc/{identity['pid']}/stat").read_text()
        fields = data[data.rfind(")") + 2 :].split()
        return fields[19] == identity["start_ticks"] and fields[0] != "Z"
    except (OSError, KeyError, IndexError):
        return False


def remote_probe(out: Path, identifier: str, wait_seconds: float = 2) -> dict:
    started = time.monotonic()
    while True:
        boot = read_boot_id()
        launch_path = out / "launch_receipt.json"
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else None
        owner_path = out / "owner_exit.json"
        owner_exit = json.loads(owner_path.read_text()) if owner_path.exists() else None
        request_path = out / "publications" / (identifier + ".json")
        pending = request_path.is_file() and not (out / "acks" / (identifier + ".json")).exists()
        alive = bool(launch and boot == launch["boot_id"] and process_alive(launch["owner"]))
        if (
            pending
            or owner_exit is not None
            or (launch and not alive)
            or time.monotonic() - started >= wait_seconds
        ):
            result = dict(
                boot_id=boot,
                launch=launch,
                owner_alive=alive,
                owner_exit=owner_exit,
                request=None,
                metadata={},
            )
            if pending:
                result["request"] = dict(
                    sha256=sha(request_path), value=json.loads(request_path.read_text())
                )
            for name in ("launch_receipt.json", "worker_started.json", "worker_process.json"):
                path = out / name
                if path.is_file():
                    result["metadata"][name] = dict(
                        binding=file_binding(path), value=json.loads(path.read_text())
                    )
            return result
        time.sleep(0.05)


def write_ack(out: Path, identifier: str, value: dict) -> None:
    if not identifier or Path(identifier).name != identifier or identifier in (".", ".."):
        raise ValueError("Unsafe acknowledgement identifier")
    path = out / "publications" / (identifier + ".json")
    request = json.loads(path.read_text())
    validate_request(request)
    validate_ack(value, request, sha(path))
    launch = json.loads((out / "launch_receipt.json").read_text())
    if (
        identifier != request["identifier"]
        or read_boot_id() != value["worker_boot_id"]
        or launch["boot_id"] != value["worker_boot_id"]
        or value["request_sha256"] != sha(path)
        or value["receipt_sha256"] != request["receipt_sha256"]
    ):
        raise ValueError("Coordinator ACK owner differs from the waiting publication")
    if time.monotonic() >= launch["deadline_monotonic"]:
        raise TimeoutError("Acknowledgement arrived after the one-shot deadline")
    worker = json.loads((out / "worker_process.json").read_text())
    if (
        any(
            (out / name).exists() for name in ("owner_exit.json", "worker_exit.json", "result.json")
        )
        or not process_alive(launch["owner"])
        or not process_alive(worker)
    ):
        raise RuntimeError("Acknowledgement owner/worker is no longer waiting")
    verify_publication(out / request["segment"], request)
    durable_json(out / "acks" / (identifier + ".json"), value)


def publication_roster(config: dict) -> list[dict]:
    values = [dict(identifier="loads", point="loads", panel=None, seed=None, tag="loads")]
    ordinal = 0
    for panel in ("primary", "confirmation"):
        for point in config["evaluation_order"]:
            for index, seed in enumerate(config[panel + "_seeds"]):
                tag = f"{panel}_{point}_{index:02d}"
                values.append(
                    dict(
                        identifier=f"{ordinal:03d}_{tag}",
                        point=point,
                        panel=panel,
                        seed=seed,
                        tag=tag,
                    )
                )
                ordinal += 1
    return values


def expected_identifiers(config: dict) -> list[str]:
    return [row["identifier"] for row in publication_roster(config)]


def _verify_owner(receipt: dict, status: dict, local_pins: dict) -> None:
    heads = status["metadata"]
    required = {"launch_receipt.json", "worker_started.json", "worker_process.json"}
    if not required.issubset(heads):
        raise ValueError("Publication lacks immutable worker/launch metadata")
    launch = heads["launch_receipt.json"]["value"]
    worker = heads["worker_started.json"]["value"]
    process = heads["worker_process.json"]["value"]
    owner = receipt["ownership"]
    if launch != status["launch"] or any(
        row.get("boot_id") != status["boot_id"] for row in (launch, worker, process, owner)
    ):
        raise ValueError("Publication launch/worker boot ownership differs")
    if (
        owner.get("boot_id_after") != status["boot_id"]
        or owner.get("owner_pid") != worker.get("pid")
        or worker.get("pid") != process.get("pid")
    ):
        raise ValueError("Publication worker process ownership differs")
    if (
        owner.get("launch_receipt_sha256") != heads["launch_receipt.json"]["binding"]["sha256"]
        or worker.get("launch_receipt_sha256") != owner["launch_receipt_sha256"]
        or owner.get("worker_started_sha256") != heads["worker_started.json"]["binding"]["sha256"]
    ):
        raise ValueError("Publication immutable launch/worker hashes differ")
    for key, expected in local_pins.items():
        if (
            owner.get(key) != expected
            or launch.get(key) != expected
            or (key in worker and worker[key] != expected)
        ):
            raise ValueError("Publication differs from the prepared local source/input owner")


def coordinate(local_run: Path, mirror: Path, host: str, remote_out: Path) -> dict:
    config = json.loads((local_run / "config.json").read_text())
    if host != config["platform"]["host"] or not remote_out.is_absolute():
        raise ValueError("Coordinator host/run does not match the frozen protocol")
    local_pins = {
        "config_sha256": sha(local_run / "config.json"),
        "input_manifest_sha256": sha(local_run / "input_manifest.json"),
        "source_manifest_sha256": sha(local_run / "manifest.json"),
    }
    mirror.mkdir(parents=True, exist_ok=False)
    for relative in (
        "evidence/episodes",
        "local_coordinator/commits",
        "local_coordinator/acks",
        "global",
    ):
        (mirror / relative).mkdir(parents=True, exist_ok=True)
    for path in sorted(
        (p for p in mirror.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True
    ):
        fsync_directory(path)
    fsync_directory(mirror)
    fsync_directory(mirror.parent)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "UpdateHostKeys=no"]
    interpreter = config["platform"]["native_interpreter"]
    script = remote_out / "source/scripts/diagnostics/coordinator_student_source_evaluation.py"
    roster = publication_roster(config)
    identifier_list = [row["identifier"] for row in roster]
    started = time.monotonic()
    instance = str(uuid.uuid4())
    records, calls, boot, error, cumulative = [], [], None, None, {}

    def invoke(command, *, data=None, timeout=30):
        tick = time.monotonic()
        try:
            process = subprocess.run(
                command, input=data, capture_output=True, text=True, timeout=timeout
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            output = getattr(exc, "stdout", None) or b""
            stderr = getattr(exc, "stderr", None) or ""
            if isinstance(output, str):
                output = output.encode()
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            calls.append(
                dict(
                    command=command,
                    exit_code=None,
                    timeout=isinstance(exc, subprocess.TimeoutExpired),
                    error_type=type(exc).__name__,
                    wall_seconds=time.monotonic() - tick,
                    stdout_bytes=len(output),
                    stderr=stderr,
                )
            )
            raise
        calls.append(
            dict(
                command=command,
                exit_code=process.returncode,
                wall_seconds=time.monotonic() - tick,
                stdout_bytes=len(process.stdout.encode()),
                stderr=process.stderr,
            )
        )
        if process.returncode:
            raise RuntimeError("Coordinator transport failed: " + process.stderr[-2000:])
        return process.stdout

    try:
        index = 0
        while True:
            if time.monotonic() - started > config["budgets"]["wall_seconds"] + 30:
                raise TimeoutError("Coordinator finite wall ceiling exhausted")
            identifier = identifier_list[index] if index < len(identifier_list) else "finished"
            command = ssh + [
                host,
                shlex.join(
                    [
                        interpreter,
                        "-B",
                        str(script),
                        "--probe",
                        "--remote-out",
                        str(remote_out),
                        "--identifier",
                        identifier,
                    ]
                ),
            ]
            status = json.loads(invoke(command, timeout=15))
            if boot is None:
                boot = status["boot_id"]
            if status["boot_id"] != boot or (
                status["launch"] and status["launch"]["boot_id"] != boot
            ):
                raise RuntimeError("Worker host rebooted; coordinator will not restart it")
            for name, meta in status["metadata"].items():
                if name not in {
                    "launch_receipt.json",
                    "worker_started.json",
                    "worker_process.json",
                }:
                    raise ValueError("Unexpected immutable worker metadata name")
                path = mirror / "global" / name
                if not path.exists():
                    durable_json(path, meta["value"])
                if (
                    json.loads(path.read_text()) != meta["value"]
                    or file_binding(path) != meta["binding"]
                ):
                    raise ValueError("Immutable worker metadata value/bytes changed")
            if status["request"]:
                if not status["owner_alive"] or status["owner_exit"] is not None:
                    raise RuntimeError("Owner stopped with an unacknowledged publication")
                item = status["request"]
                request = item["value"]
                validate_request(request)
                canonical = (
                    json.dumps(
                        request, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
                    )
                    + "\n"
                ).encode()
                if hashlib.sha256(canonical).hexdigest() != item["sha256"]:
                    raise ValueError("Transported publication body differs from its request SHA")
                expected_ordinal = index - 1
                relative = Path(request["segment"])
                if (
                    request["identifier"] != identifier
                    or request["ordinal"] != expected_ordinal
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or request["worker_boot_id"] != boot
                ):
                    raise ValueError("Publication does not match the fixed next roster entry")
                target = mirror / relative
                target.mkdir(parents=True, exist_ok=False)
                fsync_directory(target.parent)
                copied = time.monotonic()
                invoke(
                    [
                        "rsync",
                        "-a",
                        "-e",
                        shlex.join(ssh),
                        f"{host}:{remote_out / relative}/",
                        str(target) + "/",
                    ],
                    timeout=45,
                )
                receipt = verify_publication(target, request)
                _verify_owner(receipt, status, local_pins)
                if index >= len(roster) or any(
                    receipt["ownership"].get(key) != value
                    for key, value in roster[index].items()
                    if key != "identifier"
                ):
                    raise ValueError("Publication episode/reset role differs from the fixed roster")
                if index:
                    point = roster[index]["point"]
                    expected_actor = (
                        config["reference"]
                        if point == "reference"
                        else config["checkpoints"][point]
                    )
                    if (
                        receipt["ownership"].get("checkpoint_sha256") != expected_actor["sha256"]
                        or receipt["ownership"].get("policy_hash") != expected_actor["policy_hash"]
                    ):
                        raise ValueError("Publication actor differs from the frozen checkpoint")
                if receipt["ownership"].get("cumulative_before") != cumulative:
                    raise ValueError("Cumulative work chain changed between closed segments")
                for name in request["files"]:
                    with (target / name).open("rb") as stream:
                        os.fsync(stream.fileno())
                fsync_directory(target)
                fsync_directory(target.parent)
                verify_publication(target, request)
                if index:
                    verify_segment(target, request["receipt_sha256"])
                commit = dict(
                    schema="walker-s1.1-mirror-commit-v1",
                    identifier=identifier,
                    ordinal=expected_ordinal,
                    segment=str(relative),
                    request_sha256=item["sha256"],
                    receipt_sha256=request["receipt_sha256"],
                    worker_boot_id=boot,
                    files=request["files"],
                    coordinator_instance=instance,
                    coordinator_host=socket.gethostname(),
                    local_directory=str(target.resolve()),
                    copied_bytes=sum(b["bytes"] for b in request["files"].values()),
                    copy_wall_seconds=time.monotonic() - copied,
                    committed_utc=datetime.now(timezone.utc).isoformat(),
                )
                commit_path = mirror / "local_coordinator/commits" / (identifier + ".json")
                durable_json(commit_path, commit)
                ack = dict(
                    schema="walker-s1.1-ack-v1",
                    identifier=identifier,
                    request_sha256=item["sha256"],
                    receipt_sha256=request["receipt_sha256"],
                    worker_boot_id=boot,
                    mirror_commit_sha256=sha(commit_path),
                    coordinator_instance=instance,
                    coordinator_host=socket.gethostname(),
                )
                ack_path = mirror / "local_coordinator/acks" / (identifier + ".json")
                durable_json(ack_path, ack)
                invoke(
                    ssh
                    + [
                        host,
                        shlex.join(
                            [
                                interpreter,
                                "-B",
                                str(script),
                                "--write-ack",
                                "--remote-out",
                                str(remote_out),
                                "--identifier",
                                identifier,
                            ]
                        ),
                    ],
                    data=json.dumps(ack),
                    timeout=10,
                )
                records.append(
                    dict(
                        identifier=identifier,
                        request_sha256=item["sha256"],
                        receipt_sha256=request["receipt_sha256"],
                        mirror_commit_sha256=sha(commit_path),
                        ack_sha256=sha(ack_path),
                        segment=str(relative),
                        work=receipt["work"],
                    )
                )
                cumulative = dict(receipt["cumulative"])
                print(
                    json.dumps(
                        dict(
                            event="mirrored_closed_segment",
                            identifier=identifier,
                            closed_segments=len(records),
                            work=receipt["work"],
                        )
                    ),
                    flush=True,
                )
                index += 1
            if status["owner_exit"] is not None and not status["owner_alive"]:
                if not status["owner_exit"]["execution_complete"] or index != len(identifier_list):
                    raise RuntimeError(
                        "S1.1 owner stopped before the fixed committed roster completed"
                    )
                break
            if status["launch"] and not status["owner_alive"] and status["owner_exit"] is None:
                raise RuntimeError("S1.1 owner vanished without an exit; no restart permitted")
        invoke(
            [
                "rsync",
                "-a",
                "--ignore-existing",
                "-e",
                shlex.join(ssh),
                "--exclude=source/",
                "--exclude=inputs/",
                "--exclude=evidence/",
                f"{host}:{remote_out}/",
                str(mirror / "global") + "/",
            ],
            timeout=30,
        )
        final = json.loads((mirror / "global/owner_exit.json").read_text())
        worker_exit = json.loads((mirror / "global/worker_exit.json").read_text())
        if (
            final != status["owner_exit"]
            or sha(mirror / "global/worker_exit.json") != final["worker_exit_sha256"]
            or sha(mirror / "global/result.json") != worker_exit["result_sha256"]
        ):
            raise ValueError("Final mirrored owner/worker/result bindings differ")
        if any(
            row.get(key) != boot
            for row in (final, worker_exit)
            for key in ("boot_id", "boot_id_after")
        ):
            raise ValueError("Final worker/owner boot identity differs")
        for path in (mirror / "global").rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        for path in sorted(
            (p for p in (mirror / "global").rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            fsync_directory(path)
        fsync_directory(mirror / "global")
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    result = dict(
        schema="walker-s1.1-coordinator-exit-v1",
        execution_complete=error is None,
        error=error,
        worker_boot_id=boot,
        records=records,
        calls=calls,
        worker_run=str(remote_out),
        local_run=str(local_run),
        mirror=str(mirror),
        wall_seconds=time.monotonic() - started,
        new_physics=0,
        new_neural_rows=0,
        new_optimizer_steps=0,
    )
    durable_json(mirror / "coordinator_exit.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--write-ack", action="store_true")
    parser.add_argument("--remote-out", type=Path, required=True)
    parser.add_argument("--identifier", default="loads")
    parser.add_argument("--local-run", type=Path)
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--host", default="robopil5090")
    args = parser.parse_args(argv)
    if not all(c.isalnum() or c in "_-" for c in args.identifier):
        parser.error("Invalid publication identifier")
    if args.probe:
        print(json.dumps(remote_probe(args.remote_out, args.identifier)), flush=True)
        return 0
    if args.write_ack:
        write_ack(args.remote_out, args.identifier, json.load(sys.stdin))
        return 0
    if args.local_run is None or args.mirror is None:
        parser.error("Coordinator requires --local-run and --mirror")
    result = coordinate(args.local_run.resolve(), args.mirror.resolve(), args.host, args.remote_out)
    return 0 if result["execution_complete"] else 1
