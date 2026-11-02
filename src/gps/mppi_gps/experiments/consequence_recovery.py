"""Freeze and own a saved-output-only compensating C1 audit closure."""

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
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
COMMAND = "student-consequence-audit-recovery-c1.0-a1"
RECIPE = ROOT / "configs/studies/student_consequence_c1_audit_recovery_1.json"
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
EXPECTED_COUNTS = dict(
    input_files=741,
    actors=12,
    teacher_replays=12,
    assays=12000,
    fit_blocks=9,
    curve_rows=18000,
    standalone=432,
    prediction_calls=97896,
    prediction_rows=187806,
    prediction_events=195792,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class _ReadBudget:
    """Bound every orchestration payload read before opening or reading it."""

    def __init__(self, limit: int, deadline: float):
        self.limit, self.deadline = limit, deadline
        self.read_bytes = 0
        self.identities: dict[Path, tuple[int, ...]] = {}

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise ValueError("Recovery orchestration deadline exhausted")

    def reserve(self, amount: int) -> None:
        self.check()
        if type(amount) is not int or amount < 0 or self.read_bytes + amount > self.limit:
            raise ValueError("Recovery orchestration read reservation exhausted")

    @staticmethod
    def identity(path: Path) -> tuple[int, ...]:
        value = path.lstat()
        if not stat.S_ISREG(value.st_mode):
            raise ValueError("Recovery source must be an ordinary file")
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def read(self, path: Path, maximum: int, *, capture: bool = True) -> tuple[bytes, dict]:
        self.check()
        identity = self.identity(path)
        if identity[2] > maximum:
            raise ValueError("Recovery orchestration file exceeds its reservation")
        if self.identities.setdefault(path, identity) != identity:
            raise ValueError("Recovery source changed before a repeated read")
        self.reserve(identity[2])
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        digest, parts, remaining = hashlib.sha256(), [], identity[2]
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ) != identity:
                raise ValueError("Recovery source changed while opening")
            while remaining:
                amount = min(1024**2, remaining)
                self.reserve(amount)
                block = stream.read(amount)
                self.read_bytes += len(block)
                if not block:
                    raise ValueError("Recovery source shrank during a read")
                remaining -= len(block)
                digest.update(block)
                if capture:
                    parts.append(block)
        if self.identity(path) != identity:
            raise ValueError("Recovery source changed during a read")
        self.check()
        return b"".join(parts), dict(bytes=identity[2], sha256=digest.hexdigest())

    def json(self, path: Path, maximum: int = 16 * 1024**2) -> dict:
        payload, _ = self.read(path, maximum)

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate recovery metadata key")
                result[key] = value
            return result

        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite recovery JSON")),
        )
        if not isinstance(value, dict):
            raise ValueError("Recovery metadata must be an object")
        return value

    def unchanged(self) -> None:
        self.check()
        for path, expected in self.identities.items():
            self.check()
            if self.identity(path) != expected:
                raise ValueError("Recovery source changed before admission")
        self.check()


def _source_file(root: Path, relative: str) -> Path:
    if root.is_symlink():
        raise ValueError("Recovery source root cannot be a symlink")
    name = PurePosixPath(relative)
    if not name.parts or name.is_absolute() or ".." in name.parts:
        raise ValueError("Recovery source escapes its root")
    root = root.resolve()
    current = root
    for part in name.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Recovery source symlinks are forbidden")
    if not current.resolve(strict=True).is_relative_to(root):
        raise ValueError("Recovery source escapes its root")
    return current


def _output_guard(out: Path, config: dict) -> None:
    resolved = out.resolve()
    for key in ("original_run", "original_artifacts"):
        original = Path(config["locations"][key]).resolve()
        if resolved.is_relative_to(original):
            raise ValueError("Recovery output would modify an original input tree")


def validate_protocol(config: dict[str, Any]) -> None:
    """The executable recovery has no authority to run new scientific work."""
    if (
        config.get("schema") != "walker-c1-audit-recovery-protocol-v1"
        or config.get("study") != "C1.0-A1"
        or config.get("user_authorized") is not True
        or config.get("automatic_followup") is not False
        or config.get("expected_counts") != EXPECTED_COUNTS
        or config.get("new_scientific_work") != dict.fromkeys(SCIENCE_KEYS, 0)
        or any(type(value) is not int for value in config.get("new_scientific_work", {}).values())
    ):
        raise ValueError("Recovery identity, authority or zero-work contract differs")
    limits, cumulative = config["limits"], config["cumulative"]
    for key in (
        "wall_seconds",
        "coordinator_preflight_wall_seconds",
        "external_coordinator_wall_seconds",
        "finalization_wall_seconds",
        "prior_wall_upper_seconds",
    ):
        value = cumulative[key] if key.startswith("prior_") else limits[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Recovery wall reservations must be finite nonnegative numbers")
    for key in (
        "read_bytes",
        "source_verification_read_bytes",
        "rss_bytes",
        "output_bytes",
        "source_export_bytes",
    ):
        if type(limits[key]) is not int:
            raise ValueError("Recovery byte reservations must be integers")
    if limits["source_export_bytes"] <= 0 or limits["source_export_bytes"] > 64 * 1024**2:
        raise ValueError("Recovery source export reservation differs")
    if not (
        0 < limits["wall_seconds"] <= 800
        and 0 < limits["coordinator_preflight_wall_seconds"] <= 60
        and limits["total_stage_wall_seconds"] == 900
        and limits["wall_seconds"] + limits["coordinator_preflight_wall_seconds"] <= 900
        and 0 < limits["external_coordinator_wall_seconds"] <= 860
        and 0 < limits["finalization_wall_seconds"] <= 40
        and limits["external_coordinator_wall_seconds"] + limits["finalization_wall_seconds"] <= 900
        and limits["wall_seconds"] + limits["coordinator_preflight_wall_seconds"]
        <= limits["external_coordinator_wall_seconds"]
        and 0 < limits["read_bytes"] <= 7 * 1024**3
        and 0 < limits["source_verification_read_bytes"] <= 1024**3
        and limits["total_stage_read_bytes"] == 8 * 1024**3
        and limits["read_bytes"] + limits["source_verification_read_bytes"] <= 8 * 1024**3
        and 0 < limits["rss_bytes"] <= 1024**3
        and 0 < limits["output_bytes"] <= 64 * 1024**2
        and cumulative["prior_wall_upper_seconds"] + 900
        <= cumulative["audit_wall_limit_seconds"]
        <= 7200
        and cumulative["prior_read_upper_bytes"] + 8 * 1024**3
        <= cumulative["audit_read_limit_bytes"]
        <= 64 * 1024**3
    ):
        raise ValueError("Recovery exceeds its new or cumulative frozen envelope")


def freeze_recovery(recipe: Path, out: Path, source_commit: str) -> None:
    from ..lifecycle import freeze

    reader = _ReadBudget(32 * 1024**2, time.monotonic() + 60)
    config = reader.json(recipe)
    validate_protocol(config)
    _output_guard(out, config)
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("Explicit immutable recovery source commit required")
    freeze(out, config, COMMAND)
    source_bytes = sum(p.stat().st_size for p in (out / "source").rglob("*") if p.is_file())
    if source_bytes > config["limits"]["source_export_bytes"]:
        raise ValueError("Recovery frozen source exceeds its reservation; retain and stop")
    _write(
        out / "input_manifest.json",
        dict(
            schema="walker-c1-audit-recovery-inputs-v1",
            protocol_sha256=_sha(out / "config.json"),
            source_commit=source_commit,
            source_bytes=source_bytes,
            anchors=config["anchors"],
            original_inputs_are_read_only=True,
        ),
    )


def _verify_source(
    run: Path, *, budget: _ReadBudget | None = None, source_export_bytes: int = 64 * 1024**2
) -> dict[str, Any]:
    if ROOT != run / "source":
        raise ValueError("Execute recovery from its frozen source tree")
    budget = budget or _ReadBudget(448 * 1024**2, time.monotonic() + 60)
    started, before = time.monotonic(), budget.read_bytes
    manifest_path = _source_file(run, "manifest.json")
    manifest = budget.json(manifest_path)
    config_path = _source_file(run, "config.json")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries or len(entries) > 2000:
        raise ValueError("Recovery frozen source roster is missing or excessive")
    if manifest.get("input_sha256") is not None or manifest.get("command") != COMMAND:
        raise ValueError("Recovery source manifest has an incompatible execution contract")
    paths, names, total = [], set(), 0
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("Recovery source binding must be an object")
        name, digest = row.get("file"), row.get("sha256")
        if not isinstance(name, str) or name in names or not isinstance(digest, str):
            raise ValueError("Duplicate or malformed recovery source binding")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid recovery source SHA256")
        names.add(name)
        path = _source_file(run / "source", name)
        identity = budget.identity(path)
        paths.append((path, identity, digest))
        total += identity[2]
    if total > source_export_bytes or source_export_bytes > 64 * 1024**2:
        raise ValueError("Recovery source exceeds its pre-read reservation")
    config_size = budget.identity(config_path)[2]
    if config_size > 16 * 1024**2:
        raise ValueError("Recovery configuration exceeds16MiB")
    budget.reserve(total + config_size)
    _, config_binding = budget.read(config_path, 16 * 1024**2, capture=False)
    if config_binding["sha256"] != manifest.get("config_sha256"):
        raise ValueError("Recovery frozen configuration changed")
    for path, identity, digest in paths:
        if budget.identity(path) != identity:
            raise ValueError("Recovery source changed after pre-read reservation")
        _, actual = budget.read(path, identity[2], capture=False)
        if actual["sha256"] != digest:
            raise ValueError("Recovery frozen source changed: " + str(path))
    budget.unchanged()
    return dict(
        explicit_source_verification_read_bytes=budget.read_bytes - before,
        orchestration_read_bytes=budget.read_bytes,
        wall_seconds=time.monotonic() - started,
        source_files=len(entries),
        source_payload_bytes=total,
        scope="Bounded source/config/manifest reads, reserved before payload access; module-import IO remains engineering overhead.",
    )


class _NoScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {
            "torch",
            "mujoco",
            "gymnasium",
            "dm_control",
            "stable_baselines3",
            "jax",
        }:
            raise RuntimeError("Audit recovery forbids scientific execution import: " + fullname)
        return None


def worker(run: Path) -> int:
    budget = _ReadBudget(448 * 1024**2, time.monotonic() + 800)
    config = budget.json(_source_file(run, "config.json"))
    validate_protocol(config)
    budget.limit = config["limits"]["source_verification_read_bytes"] * 7 // 16
    budget.reserve(0)
    _output_guard(run, config)
    owner = budget.json(_source_file(run, "ownership.json"))
    hashes = {
        key: budget.read(_source_file(run, name), 16 * 1024**2, capture=False)[1]["sha256"]
        for key, name in (
            ("config_sha256", "config.json"),
            ("manifest_sha256", "manifest.json"),
            ("input_manifest_sha256", "input_manifest.json"),
        )
    }
    if (
        ROOT != run / "source"
        or os.environ.get("C1_RECOVERY_NONCE") != owner.get("nonce")
        or os.getppid() != owner.get("coordinator_pid")
        or any(hashes[key] != owner.get(key) for key in hashes)
    ):
        raise ValueError("Recovery worker requires its bounded frozen owner")
    guard = _NoScientificImports()
    for name in tuple(sys.modules):
        guard.find_spec(name)
    sys.meta_path.insert(0, guard)
    prelude = _verify_source(
        run, budget=budget, source_export_bytes=config["limits"]["source_export_bytes"]
    )
    prelude["all_worker_prelude_read_bytes"] = budget.read_bytes
    budget.check()
    _write(run / "worker_source_verification.json", prelude)
    budget.check()
    from ..audits.consequence_recovery import recover

    result = recover(config, run / "result")
    return int(result.get("recovery_checks_passed") is not True)


def _admission_candidate(run: Path, *, budget: _ReadBudget) -> dict[str, Any]:
    """Validate inner work; outer coordinator closure is still pending."""
    paths = dict(
        result="result/recovery.json",
        config="config.json",
        manifest="manifest.json",
        exit="exit.json",
    )
    documents = {key: budget.json(_source_file(run, value)) for key, value in paths.items()}
    config, result, outer = (documents[key] for key in ("config", "result", "exit"))
    validate_protocol(config)
    _output_guard(run, config)
    if (
        documents["manifest"].get("config_sha256")
        != budget.read(_source_file(run, "config.json"), 16 * 1024**2, capture=False)[1]["sha256"]
    ):
        raise ValueError("Recovery manifest binds a different configuration")
    protocol_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if result.get("protocol_sha256") != protocol_sha:
        raise ValueError("Recovery result binds a different protocol")
    expected = dict(
        exit_code=0,
        leader_reaped=True,
        stopping_reason=None,
        process_group_kill_requested=False,
        unknown_partial_work_possible=False,
    )
    if any(
        k not in outer or type(outer[k]) is not type(v) or outer[k] != v
        for k, v in expected.items()
    ):
        raise ValueError("New recovery process did not close cleanly")
    expected_result = dict(
        schema="walker-c1-audit-recovery-result-v1",
        recovery_checks_passed=True,
        admitted_original_scientific_checks=True,
        retained_numerical_certification=True,
        original_overall_audit_closure_passed=False,
        original_child_exit_code=None,
    )
    if (
        any(
            k not in result or type(result[k]) is not type(v) or result[k] != v
            for k, v in expected_result.items()
        )
        or result.get("new_scientific_work") != config["new_scientific_work"]
    ):
        raise ValueError("Recovery result cannot be admitted")
    if result.get("original_audit_coordinator_failed") is not True or any(
        type(value) is not int for value in result["new_scientific_work"].values()
    ):
        raise ValueError("Recovery must retain the failure and exact integer zeroes")
    receipt = dict(
        schema="walker-c1-audit-recovery-admission-candidate-v1",
        audit_recovery_resolved=False,
        candidate_checks_passed=True,
        retained_numerical_certification=True,
        original_overall_audit_closure_passed=False,
        original_child_exit_code=None,
        original_audit_coordinator_failed=True,
        new_scientific_work=result["new_scientific_work"],
        bindings={
            key: dict(
                file=value, **budget.read(_source_file(run, value), 16 * 1024**2, capture=False)[1]
            )
            for key, value in paths.items()
        },
        scope="Candidate only: worker checks closed, while external coordinator closure and final admission remain pending. Original failure remains historical fact; no missing exit status is inferred.",
    )
    budget.unchanged()
    return receipt


def admit(run: Path, *, budget: _ReadBudget | None = None) -> dict[str, Any]:
    """Save a candidate only; it cannot resolve recovery before outer closure."""
    budget = budget or _ReadBudget(448 * 1024**2, time.monotonic() + 60)
    receipt = _admission_candidate(run, budget=budget)
    budget.unchanged()
    _write(run / "admission_candidate.json", receipt)
    return receipt


def finalize(run: Path) -> dict[str, Any]:
    """Admit only after the external watchdog reaps the coordinator cleanly."""
    if ROOT != run / "source":
        raise ValueError("Finalize recovery from its frozen source tree")
    started = time.monotonic()
    budget = _ReadBudget(128 * 1024**2, started + 40)
    if (run / "admission.json").exists():
        raise FileExistsError("Final recovery admission is create-only")
    config = budget.json(_source_file(run, "config.json"))
    validate_protocol(config)
    _output_guard(run, config)
    budget.limit = config["limits"]["source_verification_read_bytes"] // 8
    budget.deadline = started + config["limits"]["finalization_wall_seconds"]
    budget.reserve(0)
    coordinator_path = _source_file(run, "coordinator_exit.json")
    coordinator = budget.json(coordinator_path)
    owner = budget.json(_source_file(run, "ownership.json"))
    if (
        type(owner.get("coordinator_pid")) is not int
        or owner["coordinator_pid"] < 1
        or type(coordinator.get("pid")) is not int
        or coordinator["pid"] != owner["coordinator_pid"]
    ):
        raise ValueError("External exit belongs to a different recovery coordinator")
    required = dict(
        exit_code=0,
        leader_reaped=True,
        stopping_reason=None,
        process_group_kill_requested=False,
        unknown_partial_work_possible=False,
    )
    if any(
        key not in coordinator
        or type(coordinator[key]) is not type(value)
        or coordinator[key] != value
        for key, value in required.items()
    ):
        raise ValueError("Recovery coordinator did not close cleanly")
    elapsed = coordinator.get("wall_seconds")
    if (
        type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or not 0 <= elapsed <= config["limits"]["external_coordinator_wall_seconds"]
    ):
        raise ValueError("Recovery coordinator exceeded its reservation")
    candidate_path = _source_file(run, "admission_candidate.json")
    candidate = budget.json(candidate_path)
    expected = _admission_candidate(run, budget=budget)
    if candidate != expected:
        raise ValueError("Recovery admission candidate differs from closed inputs")
    coordinator_binding = budget.read(coordinator_path, 16 * 1024**2, capture=False)[1]
    candidate_binding = budget.read(candidate_path, 16 * 1024**2, capture=False)[1]
    receipt = dict(candidate)
    receipt.update(
        schema="walker-c1-audit-recovery-admission-v1",
        audit_recovery_resolved=True,
        coordinator_pid=coordinator["pid"],
        admission_candidate=dict(file="admission_candidate.json", **candidate_binding),
    )
    receipt["bindings"] = dict(
        candidate["bindings"],
        coordinator_exit=dict(file="coordinator_exit.json", **coordinator_binding),
    )
    receipt["finalization_accounting"] = dict(
        read_bytes=budget.read_bytes,
        read_limit=budget.limit,
        wall_seconds=time.monotonic() - started,
        wall_limit=config["limits"]["finalization_wall_seconds"],
        scope="Explicit finalizer metadata reads and checks through pre-write; final admission write follows. No scientific work.",
    )
    receipt["scope"] = (
        "New admission after both worker and external coordinator close cleanly. Original failure and unknown child exit remain unchanged; numerical certification is retained, not recomputed."
    )
    budget.unchanged()
    _write(run / "admission.json", receipt)
    return receipt


def launch(run: Path) -> int:
    from ..proposal_launch import bounded_process

    started = time.monotonic()
    budget = _ReadBudget(448 * 1024**2, started + 60)
    config = budget.json(_source_file(run, "config.json"))
    validate_protocol(config)
    _output_guard(run, config)
    limits = config["limits"]
    stage_deadline = started + limits["external_coordinator_wall_seconds"]
    budget.limit = limits["source_verification_read_bytes"] * 7 // 16
    budget.deadline = min(stage_deadline, started + limits["coordinator_preflight_wall_seconds"])
    budget.reserve(0)
    if (run / "ownership.json").exists() or (run / "exit.json").exists():
        raise FileExistsError("Audit recovery is create-only; paid work cannot be retried")
    preflight = _verify_source(
        run, budget=budget, source_export_bytes=limits["source_export_bytes"]
    )
    budget.check()
    _write(run / "coordinator_source_verification.json", preflight)
    nonce = secrets.token_hex(32)
    _write(
        run / "ownership.json",
        dict(
            coordinator_pid=os.getpid(),
            nonce=nonce,
            **{
                key: budget.read(_source_file(run, name), 16 * 1024**2, capture=False)[1]["sha256"]
                for key, name in (
                    ("config_sha256", "config.json"),
                    ("manifest_sha256", "manifest.json"),
                    ("input_manifest_sha256", "input_manifest.json"),
                )
            },
        ),
    )
    environment = dict(os.environ, C1_RECOVERY_NONCE=nonce, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "MUJOCO_GL"):
        environment[key] = config["runtime"][key]
    old = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, config["runtime"]["cpu_affinity"])
        child_started = time.monotonic()
        budget.check()
        try:
            outer = bounded_process(
                [
                    sys.executable,
                    str(run / "source/scripts/train_mppi_gps.py"),
                    "study",
                    "student-consequence",
                    "audit-recovery",
                    "worker",
                    "--run",
                    str(run),
                ],
                cwd=run / "source",
                environment=environment,
                output=run / "stdout.txt",
                wall_seconds=min(limits["wall_seconds"], stage_deadline - child_started),
                stdout_bytes=limits["stdout_bytes"],
                peak_rss_bytes=limits["rss_bytes"],
            )
            child_elapsed = time.monotonic() - child_started
        except BaseException as error:
            _write(
                run / "exit.json",
                getattr(
                    error,
                    "process_receipt",
                    dict(
                        process_started=None,
                        unknown_partial_work_possible=True,
                        error=dict(type=type(error).__name__, message=str(error)),
                    ),
                ),
            )
            raise
    finally:
        os.sched_setaffinity(0, old)
    remaining_nonchild = limits["coordinator_preflight_wall_seconds"] - (
        time.monotonic() - started - child_elapsed
    )
    budget.deadline = min(stage_deadline, time.monotonic() + remaining_nonchild)
    _write(run / "exit.json", outer)
    budget.check()
    if outer["exit_code"] != 0 or outer["stopping_reason"] is not None:
        return 1
    postflight = _verify_source(
        run, budget=budget, source_export_bytes=limits["source_export_bytes"]
    )
    budget.check()
    _write(run / "postflight_source_verification.json", postflight)
    admit(run, budget=budget)
    _write(
        run / "orchestration_accounting.json",
        dict(
            coordinator_read_bytes=budget.read_bytes,
            coordinator_read_limit=budget.limit,
            worker_metadata_read_limit=limits["source_verification_read_bytes"] * 7 // 16,
            finalizer_metadata_read_limit=limits["source_verification_read_bytes"] // 8,
            total_source_and_metadata_read_limit=limits["source_verification_read_bytes"],
            stage_wall_seconds=time.monotonic() - started,
            nonchild_wall_seconds=time.monotonic() - started - child_elapsed,
            child_wall_seconds=child_elapsed,
            scope="Disjoint coordinator/worker/finalizer metadata-read reservations; child core reads are separately metered. Final accounting-file write follows this snapshot.",
        ),
    )
    budget.check()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("freeze", help="Freeze one authorized saved-output recovery")
    item.add_argument("--recipe", type=Path, default=RECIPE)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--source-commit", required=True)
    for name in ("launch", "worker", "finalize"):
        item = sub.add_parser(name)
        item.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_recovery(args.recipe.resolve(), args.out.resolve(), args.source_commit)
        return 0
    if args.command == "finalize":
        finalize(args.run.resolve())
        return 0
    return launch(args.run.resolve()) if args.command == "launch" else worker(args.run.resolve())
