"""A2 metadata-only compensating closure; no C1/A1 audit or science is rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

ROOT_NAMES = ("c1_run", "c1_artifacts", "a1_run", "a1_artifacts")
RSS_ERROR = "Pinned Linux worker RSS could not be measured"
REQUIRED_ANCHORS = {
    "original_scientific_receipt",
    "original_failure",
    "a1_result",
    "a1_verified_files",
    "a1_config",
    "a1_manifest",
    "a1_input_manifest",
    "a1_ownership",
    "a1_worker_exit",
    "a1_launcher_trace",
}
JSON_ANCHORS = REQUIRED_ANCHORS - {"a1_launcher_trace"}


class ClosureError(ValueError):
    """A failed metadata check retains its prefix; no retry is implied."""


def _need(value: Any, message: str) -> None:
    if not value:
        raise ClosureError(message)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(payload: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        payload,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ClosureError("Nonfinite JSON")),
    )


def _binding(value: dict) -> dict:
    _need(type(value.get("bytes")) is int and value["bytes"] >= 0, "Invalid byte binding")
    _need(
        isinstance(value.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "Invalid SHA256 binding",
    )
    return dict(bytes=value["bytes"], sha256=value["sha256"])


def _zero(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(type(v) is int and v == 0 for v in value.values())
    )


def _rss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _identity(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    _need(path.is_file() and not path.is_symlink(), "Input must be an ordinary non-symlink file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


class _Certificates:
    def __init__(self, protocol: dict):
        self.started = time.monotonic()
        self.limits = protocol["limits"]
        self.roots = {}
        for name in ROOT_NAMES:
            root = Path(protocol["locations"][name])
            _need(
                root.is_absolute() and root.is_dir() and not root.is_symlink(),
                "Invalid/symlink certificate root",
            )
            self.roots[name] = root.resolve(strict=True)
        self.read_bytes = 0
        self.identities: dict[Path, tuple[int, ...]] = {}
        self.bindings: dict[str, dict] = {}
        for root in self.roots.values():
            _need(root.is_dir(), "Certificate input root is not a directory")

    def check(self) -> None:
        _need(
            time.monotonic() - self.started < self.limits["wall_seconds"],
            "Closure wall limit exhausted",
        )
        _need(_rss() <= self.limits["rss_bytes"], "Closure RSS limit exhausted")

    def read(self, name: str, anchor: dict) -> bytes:
        self.check()
        _need(len(self.bindings) < self.limits["max_files"], "Certificate file census exhausted")
        _need(anchor.get("root") in ROOT_NAMES, "Unknown certificate root")
        relative = PurePosixPath(anchor["file"])
        _need(
            relative.parts and not relative.is_absolute() and ".." not in relative.parts,
            "Escaping certificate path",
        )
        root = self.roots[anchor["root"]]
        path = root.joinpath(*relative.parts)
        current = path
        while current != root:
            _need(not current.is_symlink(), "Symlink certificate path")
            current = current.parent
        identity = _identity(path)
        expected = _binding(anchor)
        _need(
            identity[2] == expected["bytes"] <= self.limits["max_file_bytes"],
            "Certificate size differs/exceeds bound",
        )
        _need(
            self.read_bytes + identity[2] <= self.limits["read_bytes"],
            "Closure read limit exhausted",
        )
        old = self.identities.setdefault(path, identity)
        _need(old == identity, "Certificate changed between reads")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            _need(
                (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                == identity,
                "Certificate changed while opening",
            )
            remaining = identity[2]
            parts = []
            while remaining:
                self.check()
                request = min(1024**2, remaining)
                _need(
                    self.read_bytes + request <= self.limits["read_bytes"],
                    "Closure read limit exhausted",
                )
                payload = stream.read(request)
                self.read_bytes += len(payload)
                _need(bool(payload), "Certificate ended during read")
                parts.append(payload)
                remaining -= len(payload)
        payload = b"".join(parts)
        _need(
            _identity(path) == identity and _sha(payload) == expected["sha256"],
            "Certificate changed or SHA256 differs",
        )
        self.bindings[name] = dict(root=anchor["root"], file=anchor["file"], **expected)
        return payload

    def unchanged(self) -> None:
        self.check()
        for path, identity in self.identities.items():
            _need(_identity(path) == identity, "Certificate changed before closure")


def _protocol(protocol: dict) -> None:
    _need(
        protocol.get("schema") == "walker-c1-certificate-closure-protocol-v1"
        and protocol.get("study") == "C1.0-A2",
        "Wrong A2 protocol",
    )
    _need(
        protocol.get("user_authorized") is True and protocol.get("automatic_followup") is False,
        "A2 authority differs",
    )
    _need(_zero(protocol.get("new_scientific_work")), "A2 must declare zero new scientific work")
    _need(REQUIRED_ANCHORS <= set(protocol["anchors"]), "Required certificate is absent")
    caps = dict(
        wall_seconds=30,
        read_bytes=32 * 1024**2,
        rss_bytes=512 * 1024**2,
        output_bytes=4 * 1024**2,
        max_file_bytes=2 * 1024**2,
        max_files=32,
    )
    for name, cap in caps.items():
        value = protocol["limits"][name]
        _need(
            type(value) in (int, float) and math.isfinite(value) and 0 < value <= cap,
            "Increased/invalid A2 bound",
        )
        if name != "wall_seconds":
            _need(type(value) is int, "A2 byte/count limits must be integers")
    _need(
        type(protocol["expected_a1_verified_files"]) is int
        and protocol["expected_a1_verified_files"] > 0,
        "Invalid A1 file census",
    )
    _need(
        re.fullmatch(r"[0-9a-f]{40}", protocol["expected_a1_source_commit"]),
        "Invalid A1 source commit",
    )
    if "cumulative" in protocol:
        cumulative = protocol["cumulative"]
        _need(
            cumulative["audit_wall_limit_seconds"] <= 7200
            and cumulative["audit_read_limit_bytes"] <= 64 * 1024**3,
            "Original audit envelope increased",
        )
        _need(
            0 <= cumulative["prior_wall_upper_seconds"]
            and cumulative["prior_wall_upper_seconds"]
            + protocol["limits"]["total_stage_wall_seconds"]
            <= cumulative["audit_wall_limit_seconds"],
            "Cumulative A2 wall budget exhausted",
        )
        _need(
            0 <= cumulative["prior_read_upper_bytes"]
            and cumulative["prior_read_upper_bytes"] + protocol["limits"]["total_stage_read_bytes"]
            <= cumulative["audit_read_limit_bytes"],
            "Cumulative A2 read budget exhausted",
        )


def _check_documents(protocol: dict, documents: dict, payloads: dict, roots: dict) -> dict:
    original = documents["original_scientific_receipt"]
    failure = documents["original_failure"]
    result = documents["a1_result"]
    config, manifest, inputs = (
        documents[name] for name in ("a1_config", "a1_manifest", "a1_input_manifest")
    )
    owner, exit_value = documents["a1_ownership"], documents["a1_worker_exit"]
    verified = documents["a1_verified_files"]
    _need(
        all(
            original.get(k) is True
            for k in (
                "passed",
                "full_scientific_audit_passed",
                "scientific_contrast_completed",
                "evidence_integrity_passed",
            )
        ),
        "Original numerical certification failed",
    )
    _need(
        original.get("schema") == "walker-student-consequence-c1.0-audit-v1",
        "Wrong original certification",
    )
    original_binding = _binding(protocol["anchors"]["original_scientific_receipt"])
    _need(
        failure.get("schema") == "walker-c1-audit-launcher-failure-v1"
        and failure.get("overall_audit_closure_passed") is False
        and failure.get("audit_coordinator_failed") is True
        and failure.get("original_audit_exit_exists") is False
        and failure.get("original_child_exit_code") is None
        and failure.get("coordinator_exit_code") == 1,
        "Original C1 failure facts differ",
    )
    _need(
        _binding(failure["scientific_receipt"]) == original_binding,
        "Original failure/certificate binding differs",
    )
    _need(
        result.get("schema") == "walker-c1-audit-recovery-result-v1"
        and all(
            result.get(k) is True
            for k in (
                "recovery_checks_passed",
                "admitted_original_scientific_checks",
                "retained_numerical_certification",
                "original_audit_coordinator_failed",
            )
        ),
        "A1 core certification did not pass",
    )
    _need(
        result.get("original_overall_audit_closure_passed") is False
        and result.get("original_child_exit_code") is None
        and result.get("error") is None,
        "A1 erased original failure or reported an error",
    )
    _need(
        _zero(result.get("new_scientific_work"))
        and result["new_scientific_work"] == config["new_scientific_work"],
        "A1 executed or failed to account for science",
    )
    _need(
        _binding(result["original_scientific_receipt"]) == original_binding,
        "A1 certifies a different original receipt",
    )
    _need(
        config.get("schema") == "walker-c1-audit-recovery-protocol-v1"
        and config.get("study") == "C1.0-A1"
        and config.get("user_authorized") is True
        and config.get("automatic_followup") is False,
        "A1 protocol identity differs",
    )
    _need(
        result["protocol_sha256"] == _sha(_canonical(config)),
        "A1 canonical configuration binding differs",
    )
    config_sha = _sha(payloads["a1_config"])
    _need(
        manifest["config_sha256"]
        == inputs["protocol_sha256"]
        == owner["config_sha256"]
        == config_sha,
        "A1 frozen configuration bindings differ",
    )
    _need(
        manifest["command"]
        == protocol.get("expected_a1_source_command", "student-consequence-audit-recovery-c1.0-a1")
        and manifest.get("input_sha256") is None,
        "A1 source execution contract differs",
    )
    _need(
        inputs["source_commit"] == protocol["expected_a1_source_commit"], "A1 source commit differs"
    )
    _need(
        inputs.get("schema") == "walker-c1-audit-recovery-inputs-v1"
        and inputs.get("original_inputs_are_read_only") is True
        and inputs.get("anchors") == config["anchors"],
        "A1 input/source seal differs",
    )
    _need(
        owner["manifest_sha256"] == _sha(payloads["a1_manifest"])
        and owner["input_manifest_sha256"] == _sha(payloads["a1_input_manifest"]),
        "A1 owner/source bindings differ",
    )
    _need(
        _binding(config["anchors"]["scientific_receipt"]) == original_binding
        and _binding(config["anchors"]["failure"])
        == _binding(protocol["anchors"]["original_failure"]),
        "A1 original anchors differ",
    )
    expected_auditor = protocol.get("expected_a1_auditor_sha256")
    if expected_auditor is not None:
        sources = {row["file"]: row["sha256"] for row in manifest["files"]}
        _need(
            sources.get("src/gps/mppi_gps/audits/consequence_recovery.py") == expected_auditor,
            "A1 numerical-free verifier source differs",
        )
    sources = {row["file"]: row["sha256"] for row in manifest["files"]}
    for anchor_name, source_name in (
        ("a1_auditor_source", "src/gps/mppi_gps/audits/consequence_recovery.py"),
        ("a1_coordinator_source", "src/gps/mppi_gps/experiments/consequence_recovery.py"),
        ("a1_watchdog_source", "src/gps/mppi_gps/proposal_launch.py"),
    ):
        if anchor_name in protocol["anchors"]:
            _need(
                sources.get(source_name) == protocol["anchors"][anchor_name]["sha256"],
                "A1 selected source bytes differ from its manifest",
            )
    _need(
        _binding(result["files_catalog"]) == _binding(protocol["anchors"]["a1_verified_files"]),
        "A1 verified-map binding differs",
    )
    _need(
        result["files_catalog"]["file"] == "verified_files.json"
        and verified.get("schema") == "walker-c1-recovery-verified-files-v1",
        "Wrong A1 verified-map contract",
    )
    _need(
        result["verified_file_count"]
        == len(verified["files"])
        == protocol["expected_a1_verified_files"],
        "A1 verified file census differs",
    )
    for name, key in (
        ("run/config.json", "config_sha256"),
        ("run/worker_result.json", "worker_result_sha256"),
    ):
        _need(
            verified["files"][name]["sha256"] == original[key],
            "A1 map/original numerical-result binding differs",
        )
    for value in verified["files"].values():
        _binding(value)
    _need(
        exit_value.get("exit_code") == 0
        and type(exit_value.get("exit_code")) is int
        and exit_value.get("leader_reaped") is True,
        "No genuine reaped A1 worker exit zero",
    )
    _need(
        exit_value.get("stopping_reason") == "coordinator_exception"
        and exit_value.get("process_group_kill_requested") is True
        and exit_value.get("unknown_partial_work_possible") is True,
        "A1 failed-observer flags were changed",
    )
    _need(
        exit_value.get("error", {}).get("type") == "RuntimeError"
        and RSS_ERROR in exit_value.get("error", {}).get("message", ""),
        "A1 observer error differs",
    )
    _need(
        type(owner.get("coordinator_pid")) is int
        and owner["coordinator_pid"] > 0
        and type(exit_value.get("pid")) is int
        and exit_value["pid"] > 0
        and owner["coordinator_pid"] != exit_value["pid"],
        "A1 process ownership differs",
    )
    command = exit_value.get("command")
    _need(
        isinstance(command, list)
        and len(command) == 8
        and command[1] == str(roots["a1_run"] / "source/scripts/train_mppi_gps.py")
        and command[2:6] == ["study", "student-consequence", "audit-recovery", "worker"]
        and command[6:] == ["--run", str(roots["a1_run"])],
        "Exit zero belongs to a different A1 command",
    )
    _need(RSS_ERROR in payloads["a1_launcher_trace"].decode(), "A1 launcher traceback differs")
    if "a1_external_failure" in payloads:
        _need(
            RSS_ERROR in payloads["a1_external_failure"].decode(),
            "A1 external failure evidence differs",
        )
        external = _json(payloads["a1_external_failure"])
        _need(
            external.get("schema") == "walker-c1-a1-external-failure-v1"
            and external.get("external_tool_exit_code") == 1
            and all(
                external.get(k) is False
                for k in (
                    "original_coordinator_exit_available",
                    "original_candidate_exists",
                    "original_admission_exists",
                )
            )
            and external.get("worker_exit_recorded_zero") is True
            and external.get("worker_coordinator_exception_flags_preserved") is True
            and external.get("error_type") == "RuntimeError"
            and RSS_ERROR in external.get("error_message", ""),
            "A1 external failure facts differ",
        )
    if "a1_external_trace" in payloads:
        _need(RSS_ERROR in payloads["a1_external_trace"].decode(), "A1 external traceback differs")
    original_relative = PurePosixPath(protocol["anchors"]["original_scientific_receipt"]["file"])
    _need(
        not (roots["c1_artifacts"] / str(original_relative.parent.parent) / "exit.json").exists(),
        "Original C1 missing exit was replaced",
    )
    for name in ("admission_candidate.json", "admission.json", "coordinator_exit.json"):
        _need(not (roots["a1_run"] / name).exists(), "A1 failed admission state was replaced")
    return dict(
        a1_verified_files=result["verified_file_count"],
        a1_worker_exit_code=0,
        a1_worker_leader_reaped=True,
        a1_failure_flags_preserved=True,
        original_child_exit_code=None,
        original_failure_preserved=True,
    )


def close_certificates(protocol: dict, out: Path, *, protocol_read_bytes: int = 0) -> dict:
    """Check only preserved metadata; a separate clean observer admits closure."""
    out = Path(out)
    for name in ROOT_NAMES:
        root = Path(protocol["locations"][name]).resolve(strict=False)
        _need(
            not out.resolve(strict=False).is_relative_to(root),
            "A2 output is inside preserved inputs",
        )
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    reader = None
    result = dict(
        schema="walker-c1-certificate-closure-result-v1",
        certificate_checks_passed=False,
        retained_A1_integrity_certification=False,
        retained_original_numerical_certification=False,
        original_A1_admission_failed=True,
        original_C1_coordinator_failed=True,
        original_C1_child_exit_code=None,
        new_scientific_work=dict(protocol.get("new_scientific_work", {})),
        scope="New certificate-only compensating closure checks. A1 remains failed under its original admission gate. Its point-in-time integrity/accounting certification and the original numerical certification are retained; no original payload, journal, actor or simulator is re-evaluated. Only a separately observed clean A2 process exit may admit this result.",
    )
    try:
        _protocol(protocol)
        result["protocol_sha256"] = _sha(_canonical(protocol))
        reader = _Certificates(protocol)
        _need(
            type(protocol_read_bytes) is int and protocol_read_bytes >= 0,
            "Invalid protocol-read debit",
        )
        reader.read_bytes = protocol_read_bytes
        _need(
            reader.read_bytes + sum(_binding(a)["bytes"] for a in protocol["anchors"].values())
            <= protocol["limits"]["read_bytes"],
            "A2 complete metadata read reservation exhausted",
        )
        payloads = {name: reader.read(name, value) for name, value in protocol["anchors"].items()}
        documents = {name: _json(value) for name, value in payloads.items() if name in JSON_ANCHORS}
        facts = _check_documents(protocol, documents, payloads, reader.roots)
        reader.unchanged()
        result.update(
            certificate_checks_passed=True,
            retained_A1_integrity_certification=True,
            retained_original_numerical_certification=True,
            original_A1_failure_flags_preserved=True,
            checked_facts=facts,
            verified_certificates=reader.bindings,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    result["work"] = dict(
        read_bytes=protocol_read_bytes if reader is None else reader.read_bytes,
        checked_certificates=0 if reader is None else len(reader.bindings),
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=_rss(),
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    _need(
        len(payload) <= protocol.get("limits", {}).get("output_bytes", 4 * 1024**2),
        "A2 output limit exhausted",
    )
    with (out / "certificate_result.json").open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    identity = _identity(args.protocol)
    size = identity[2]
    _need(size <= 2 * 1024**2, "Protocol exceeds bound")
    descriptor = os.open(args.protocol, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        _need(
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            == identity,
            "Protocol changed while opening",
        )
        payload = stream.read(size)
    _need(
        len(payload) == size and _identity(args.protocol) == identity,
        "Protocol changed while reading",
    )
    result = close_certificates(_json(payload), args.out, protocol_read_bytes=len(payload))
    print(
        json.dumps(
            dict(certificate_checks_passed=result["certificate_checks_passed"], work=result["work"])
        )
    )
    return int(result["certificate_checks_passed"] is not True)


if __name__ == "__main__":
    raise SystemExit(main())
