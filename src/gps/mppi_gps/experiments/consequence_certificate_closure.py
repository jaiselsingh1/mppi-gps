"""Own the tiny, certificate-only A2 bundle; execute no scientific work."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import math
import os
import secrets
import stat
import sys
import time
from pathlib import Path
from typing import Any

SOURCE_DIR = Path(__file__).resolve().parent
SOURCE_FILES = {"observer.py", "closure_checks.py", "watchdog.py"}
SCIENCE_KEYS = (
    "neural_rows",
    "prediction_calls",
    "checkpoint_tensor_loads",
    "native_controls",
    "physics_substeps",
    "backward_rows",
    "optimizer_steps",
    "resets",
    "restores",
    "mppi_queries",
)
ROOT_KEYS = ("c1_run", "c1_artifacts", "a1_run", "a1_artifacts")
RESULT_FLAGS = dict(
    schema="walker-c1-certificate-closure-result-v1",
    certificate_checks_passed=True,
    retained_A1_integrity_certification=True,
    retained_original_numerical_certification=True,
    original_A1_admission_failed=True,
    original_C1_coordinator_failed=True,
    original_C1_child_exit_code=None,
    original_A1_failure_flags_preserved=True,
)


def _components():
    if __package__:
        from ..audits.consequence_closure import close_certificates
        from ..proposal_launch import bounded_process
    else:
        sys.dont_write_bytecode = True
        from closure_checks import close_certificates
        from watchdog import bounded_process
    return close_certificates, bounded_process


def _require(value: dict, expected: dict, label: str) -> None:
    if not isinstance(value, dict) or any(
        key not in value or type(value[key]) is not type(item) or value[key] != item
        for key, item in expected.items()
    ):
        raise ValueError(label + " does not satisfy the frozen closure contract")


def validate_protocol(protocol: dict) -> None:
    _require(
        protocol,
        dict(
            schema="walker-c1-certificate-closure-protocol-v1",
            study="C1.0-A2",
            user_authorized=True,
            automatic_followup=False,
        ),
        "Protocol",
    )
    science = protocol.get("new_scientific_work")
    if science != dict.fromkeys(SCIENCE_KEYS, 0) or any(
        type(v) is not int for v in science.values()
    ):
        raise ValueError("A2 cannot authorize scientific work")
    limits = protocol["limits"]
    caps = dict(
        wall_seconds=30,
        coordinator_wall_seconds=45,
        finalization_wall_seconds=15,
        metadata_read_bytes=32 * 1024**2,
        source_read_bytes=4 * 1024**2,
        read_bytes=32 * 1024**2,
        rss_bytes=512 * 1024**2,
        output_bytes=4 * 1024**2,
        max_file_bytes=2 * 1024**2,
        stdout_bytes=4 * 1024**2,
    )
    for key, cap in caps.items():
        value = limits[key]
        if type(value) is not int or not 0 < value <= cap:
            raise ValueError("Invalid A2 reservation: " + key)
    if not (
        limits["wall_seconds"] <= limits["coordinator_wall_seconds"]
        and limits["coordinator_wall_seconds"] + limits["finalization_wall_seconds"] <= 60
        and limits["total_stage_wall_seconds"] == 60
        and limits["total_stage_read_bytes"] == 128 * 1024**2
    ):
        raise ValueError("A2 stage reservations differ")
    cumulative = protocol["cumulative"]
    if not (
        0 <= cumulative["prior_wall_upper_seconds"]
        and cumulative["prior_wall_upper_seconds"] + 60
        <= cumulative["audit_wall_limit_seconds"]
        <= 7200
        and 0 <= cumulative["prior_read_upper_bytes"]
        and cumulative["prior_read_upper_bytes"] + 128 * 1024**2
        <= cumulative["audit_read_limit_bytes"]
        <= 64 * 1024**3
    ):
        raise ValueError("A2 would reset the original audit budget")


def _guard_output(run: Path, protocol: dict) -> None:
    resolved = run.resolve()
    if run.is_symlink():
        raise ValueError("A2 run cannot be a symlink")
    for key in ROOT_KEYS:
        original = Path(protocol["locations"][key])
        if not original.is_absolute() or resolved.is_relative_to(original.resolve()):
            raise ValueError("A2 output would modify a preserved input tree")


def _path(run: Path, relative: str) -> Path:
    path = run / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("A2 path escapes its bundle")
    current = path
    while current != run:
        if current.is_symlink():
            raise ValueError("A2 symlinks are forbidden")
        current = current.parent
    return path


class _Reader:
    def __init__(self, limit: int, seconds: float):
        self.started = time.monotonic()
        self.limit, self.deadline = limit, self.started + seconds
        self.source_limit = 4 * 1024**2
        self.bytes = self.source_bytes = 0
        self.identities: dict[Path, tuple] = {}

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise ValueError("A2 metadata deadline exhausted")

    @staticmethod
    def identity(path: Path) -> tuple:
        value = path.lstat()
        if not stat.S_ISREG(value.st_mode):
            raise ValueError("A2 requires ordinary input files")
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def reserve(self, count: int, *, source: bool = False) -> None:
        self.check()
        if self.bytes + count > self.limit or (
            source and self.source_bytes + count > self.source_limit
        ):
            raise ValueError("A2 metadata read reservation exhausted")

    def read(self, path: Path, *, source: bool = False) -> tuple[bytes, dict]:
        self.check()
        identity = self.identity(path)
        if identity[2] > 2 * 1024**2 or self.identities.setdefault(path, identity) != identity:
            raise ValueError("A2 input is oversized or changed")
        self.reserve(identity[2], source=source)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        parts, remaining = [], identity[2]
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ) != identity:
                raise ValueError("A2 input changed while opening")
            while remaining:
                amount = min(1024**2, remaining)
                self.reserve(amount, source=source)
                payload = stream.read(amount)
                self.bytes += len(payload)
                if source:
                    self.source_bytes += len(payload)
                if not payload:
                    raise ValueError("A2 input shrank")
                remaining -= len(payload)
                parts.append(payload)
        if self.identity(path) != identity:
            raise ValueError("A2 input changed during read")
        self.check()
        payload = b"".join(parts)
        return payload, dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

    def json(self, path: Path, *, source: bool = False) -> tuple[dict, dict]:
        payload, binding = self.read(path, source=source)

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate A2 metadata key")
                result[key] = value
            return result

        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite A2 JSON")),
        )
        if not isinstance(value, dict):
            raise ValueError("A2 metadata must be an object")
        return value, binding

    def unchanged(self) -> None:
        for path, identity in self.identities.items():
            self.check()
            if self.identity(path) != identity:
                raise ValueError("A2 input changed before closure")


def _bundle(run: Path, reader: _Reader) -> tuple[dict, dict]:
    if SOURCE_DIR != run / "source":
        raise ValueError("Execute A2 from its frozen tiny source bundle")
    protocol, protocol_binding = reader.json(_path(run, "protocol.json"), source=True)
    validate_protocol(protocol)
    reader.source_limit = protocol["limits"]["source_read_bytes"]
    reader.reserve(0, source=True)
    _guard_output(run, protocol)
    manifest, manifest_binding = reader.json(_path(run, "source_manifest.json"), source=True)
    if (
        manifest.get("schema") != "walker-c1-certificate-source-v1"
        or manifest.get("protocol_sha256") != protocol_binding["sha256"]
        or set(manifest.get("files", {})) != SOURCE_FILES
        or not isinstance(manifest.get("source_commit"), str)
        or len(manifest["source_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in manifest["source_commit"])
    ):
        raise ValueError("A2 frozen manifest differs")
    if {path.name for path in (run / "source").iterdir()} != SOURCE_FILES:
        raise ValueError("A2 source bundle contains unbound files")
    total = sum(reader.identity(_path(run, "source/" + name))[2] for name in SOURCE_FILES)
    reader.reserve(total, source=True)
    for name, expected in manifest["files"].items():
        if (
            not isinstance(expected, dict)
            or set(expected) != {"bytes", "sha256"}
            or type(expected["bytes"]) is not int
            or expected["bytes"] < 0
            or not isinstance(expected["sha256"], str)
            or len(expected["sha256"]) != 64
        ):
            raise ValueError("Malformed A2 source binding")
        _, actual = reader.read(_path(run, "source/" + name), source=True)
        if actual != expected:
            raise ValueError("A2 frozen source differs: " + name)
    reader.unchanged()
    return protocol, dict(protocol=protocol_binding, source_manifest=manifest_binding)


def _write(run: Path, name: str, value: Any, protocol: dict) -> None:
    _guard_output(run, protocol)
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    paths = list(run.rglob("*"))
    if any(p.is_symlink() for p in paths) or len(paths) > 512:
        raise ValueError("A2 output roster is unsafe or excessive")
    if (
        sum(p.stat().st_size for p in paths if p.is_file()) + len(payload)
        > protocol["limits"]["output_bytes"]
    ):
        raise ValueError("A2 whole-output reservation exhausted")
    with _path(run, name).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _clean(value: dict, label: str, maximum: int) -> None:
    _require(
        value,
        dict(
            exit_code=0,
            leader_reaped=True,
            stopping_reason=None,
            process_group_kill_requested=False,
            unknown_partial_work_possible=False,
        ),
        label,
    )
    elapsed = value.get("wall_seconds")
    if (
        type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or not 0 <= elapsed <= maximum
    ):
        raise ValueError(label + " exceeded its reservation")


def _candidate(run: Path, reader: _Reader, protocol: dict) -> dict:
    names = dict(
        result="result/certificate_result.json",
        protocol="protocol.json",
        source_manifest="source_manifest.json",
        worker_exit="worker_exit.json",
    )
    documents, bindings = {}, {}
    for key, relative in names.items():
        documents[key], actual = reader.json(_path(run, relative))
        bindings[key] = dict(file=relative, **actual)
    _require(documents["result"], RESULT_FLAGS, "A2 result")
    if documents["result"].get("new_scientific_work") != protocol["new_scientific_work"] or any(
        type(v) is not int for v in documents["result"]["new_scientific_work"].values()
    ):
        raise ValueError("A2 result contains scientific work")
    canonical = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        documents["result"].get("protocol_sha256") != canonical
        or documents["protocol"] != protocol
        or documents["source_manifest"].get("protocol_sha256") != bindings["protocol"]["sha256"]
    ):
        raise ValueError("A2 candidate input association differs")
    _clean(documents["worker_exit"], "A2 worker", protocol["limits"]["wall_seconds"])
    return dict(
        schema="walker-c1-certificate-candidate-v1",
        audit_recovery_resolved=False,
        **{key: documents["result"][key] for key in RESULT_FLAGS if key != "schema"},
        new_scientific_work=protocol["new_scientific_work"],
        bindings=bindings,
    )


class _NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {
            "torch",
            "mujoco",
            "gymnasium",
            "dm_control",
            "stable_baselines3",
            "jax",
        }:
            raise ValueError("A2 forbids scientific execution imports")
        return None


def worker(run: Path) -> int:
    reader = _Reader(4 * 1024**2, 30)
    protocol, proof = _bundle(run, reader)
    reader.deadline = min(reader.deadline, reader.started + protocol["limits"]["wall_seconds"])
    owner, _ = reader.json(_path(run, "ownership.json"))
    if (
        os.environ.get("C1_A2_NONCE") != owner.get("nonce")
        or os.getppid() != owner.get("coordinator_pid")
        or owner.get("protocol_sha256") != proof["protocol"]["sha256"]
        or owner.get("source_manifest_sha256") != proof["source_manifest"]["sha256"]
    ):
        raise ValueError("A2 worker lacks its frozen owner")
    guard = _NoScience()
    for name in tuple(sys.modules):
        guard.find_spec(name)
    sys.meta_path.insert(0, guard)
    reader.check()
    _write(
        run,
        "worker_prelude.json",
        dict(
            read_bytes=reader.bytes,
            source_read_bytes=reader.source_bytes,
            prelude_bytes_also_charged_by_core=reader.bytes,
            scope="All worker bundle/ownership prefix reads are included in the core prepaid-read debit, not additive.",
        ),
        protocol,
    )
    reader.check()
    close_certificates, _ = _components()
    result = close_certificates(protocol, run / "result", protocol_read_bytes=reader.bytes)
    return int(result.get("certificate_checks_passed") is not True)


def coordinate(run: Path) -> int:
    reader = _Reader(32 * 1024**2, 45)
    protocol, proof = _bundle(run, reader)
    reader.limit = protocol["limits"]["metadata_read_bytes"]
    reader.deadline = min(
        reader.deadline, reader.started + protocol["limits"]["coordinator_wall_seconds"]
    )
    reader.reserve(0)
    for name in (
        "ownership.json",
        "worker_exit.json",
        "admission_candidate.json",
        "admission.json",
    ):
        if (run / name).exists():
            raise FileExistsError("A2 is create-only; no automatic retry")
    nonce = secrets.token_hex(32)
    _write(
        run,
        "ownership.json",
        dict(
            coordinator_pid=os.getpid(),
            nonce=nonce,
            protocol_sha256=proof["protocol"]["sha256"],
            source_manifest_sha256=proof["source_manifest"]["sha256"],
        ),
        protocol,
    )
    environment = dict(os.environ, C1_A2_NONCE=nonce, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "MUJOCO_GL"):
        environment[name] = protocol["runtime"][name]
    _, bounded = _components()
    previous = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, protocol["runtime"]["cpu_affinity"])
        try:
            outer = bounded(
                [sys.executable, str(run / "source/observer.py"), "worker", "--run", str(run)],
                cwd=run / "source",
                environment=environment,
                output=run / "stdout.txt",
                wall_seconds=min(
                    protocol["limits"]["wall_seconds"], reader.deadline - time.monotonic()
                ),
                stdout_bytes=protocol["limits"]["stdout_bytes"],
                peak_rss_bytes=protocol["limits"]["rss_bytes"],
            )
        except BaseException as error:
            _write(
                run,
                "worker_exit.json",
                getattr(
                    error,
                    "process_receipt",
                    dict(
                        process_started=None,
                        unknown_partial_work_possible=True,
                        error=dict(type=type(error).__name__, message=str(error)),
                    ),
                ),
                protocol,
            )
            raise
    finally:
        os.sched_setaffinity(0, previous)
    _write(run, "worker_exit.json", outer, protocol)
    _clean(outer, "A2 worker", protocol["limits"]["wall_seconds"])
    _bundle(run, reader)
    candidate = _candidate(run, reader, protocol)
    reader.unchanged()
    _write(run, "admission_candidate.json", candidate, protocol)
    _write(
        run,
        "coordinate_accounting.json",
        dict(
            read_bytes=reader.bytes,
            source_read_bytes=reader.source_bytes,
            scope="Coordinator metadata only; worker certificate reads separate; final write follows.",
        ),
        protocol,
    )
    reader.check()
    return 0


def finalize(run: Path) -> dict:
    reader = _Reader(32 * 1024**2, 15)
    protocol, _ = _bundle(run, reader)
    reader.limit = protocol["limits"]["metadata_read_bytes"]
    reader.deadline = min(
        reader.deadline, reader.started + protocol["limits"]["finalization_wall_seconds"]
    )
    reader.reserve(0)
    if (run / "admission.json").exists():
        raise FileExistsError("A2 final admission is create-only")
    coordinator, coordinator_binding = reader.json(_path(run, "coordinator_exit.json"))
    owner, _ = reader.json(_path(run, "ownership.json"))
    _clean(coordinator, "A2 coordinator", protocol["limits"]["coordinator_wall_seconds"])
    if (
        type(coordinator.get("pid")) is not int
        or coordinator["pid"] < 1
        or coordinator["pid"] != owner.get("coordinator_pid")
    ):
        raise ValueError("A2 coordinator exit belongs to another process")
    candidate, candidate_binding = reader.json(_path(run, "admission_candidate.json"))
    if candidate != _candidate(run, reader, protocol):
        raise ValueError("A2 candidate differs from closed inputs")
    admission = dict(candidate)
    admission.update(
        schema="walker-c1-certificate-admission-v1",
        audit_recovery_resolved=True,
        coordinator_pid=coordinator["pid"],
        admission_candidate=dict(file="admission_candidate.json", **candidate_binding),
    )
    admission["bindings"] = dict(
        candidate["bindings"],
        coordinator_exit=dict(file="coordinator_exit.json", **coordinator_binding),
    )
    admission["finalization_accounting"] = dict(
        read_bytes=reader.bytes,
        source_read_bytes=reader.source_bytes,
        scope="Metadata checks through pre-write; admission write and external finalizer exit follow.",
    )
    reader.unchanged()
    _write(run, "admission.json", admission, protocol)
    return admission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("coordinate", "worker", "finalize"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if args.command == "finalize":
        finalize(run)
        return 0
    return coordinate(run) if args.command == "coordinate" else worker(run)


if __name__ == "__main__":
    raise SystemExit(main())
