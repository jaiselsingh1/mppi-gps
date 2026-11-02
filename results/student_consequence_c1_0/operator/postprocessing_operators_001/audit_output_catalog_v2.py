"""Hash closed C1 audit OUTPUTS inside the original audit's remaining envelope.

This standalone stdlib companion never opens the scientific run. It does not
rerun the audit, a network, an optimizer, or physics. Admit its catalog only
alongside a successful, reaped, independently bounded helper-process exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GIB = 1024**3
CHUNK = 1024**2
MAX_FILES = 20000
MAX_RSS = 4 * GIB


class CatalogError(RuntimeError):
    """A failed prerequisite or budget preserves the already paid read census."""

    def __init__(self, message: str, accounting: dict[str, Any] | None = None):
        super().__init__(message)
        self.accounting = dict(accounting or {})


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def _finite(value: Any, label: str) -> float:
    _need(type(value) in (int, float) and math.isfinite(value) and value >= 0,
          "Invalid nonnegative finite " + label)
    return float(value)


def _identity(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    _need(not stat.S_ISLNK(value.st_mode), "Symlink is ineligible: " + str(path))
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _inventory(root: Path) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    files, directories = {}, {}

    def unreadable(error):
        raise CatalogError("Unreadable audit output tree: " + str(error)) from error

    for current, names, filenames in os.walk(root, followlinks=False, onerror=unreadable):
        directory = Path(current)
        identity = _identity(directory)
        _need(stat.S_ISDIR(identity[2]), "Non-directory audit path")
        directories[str(directory.relative_to(root))] = identity
        for name in names:
            value = _identity(directory / name)
            _need(stat.S_ISDIR(value[2]), "Non-directory audit tree entry")
        for name in filenames:
            path = directory / name
            identity = _identity(path)
            _need(stat.S_ISREG(identity[2]), "Only ordinary audit output files are eligible")
            _need(path.resolve(strict=True).is_relative_to(root), "Audit output escapes root")
            relative = str(path.relative_to(root))
            _need(len(files) < MAX_FILES, "Audit output file census exceeds20000")
            files[relative] = identity
    return files, directories


def _usage() -> dict[str, Any]:
    value = resource.getrusage(resource.RUSAGE_SELF)
    return dict(
        user_seconds=float(value.ru_utime), system_seconds=float(value.ru_stime),
        process_seconds=time.process_time(),
        peak_rss_bytes=int(value.ru_maxrss if sys.platform == "darwin" else value.ru_maxrss * 1024),
        kernel_input_blocks=int(value.ru_inblock), kernel_output_blocks=int(value.ru_oublock),
        voluntary_context_switches=int(value.ru_nvcsw),
        involuntary_context_switches=int(value.ru_nivcsw),
    )


def _json(payload: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(CatalogError("Nonfinite JSON")))
    except (ValueError, UnicodeError) as error:
        raise CatalogError("Malformed closed audit JSON") from error


def hash_audit_outputs(
    audit_outer: str | Path,
    out_json: str | Path,
    *,
    max_audit_seconds: float = 7200,
    max_audit_read_bytes: int = 64 * GIB,
    max_audit_output_bytes: int = 4 * GIB,
    admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an exact catalog of already closed audit files; never reset limits.

    A trusted operator token supplies prior debits before any audit payload is
    opened. Header bytes are charged for verification and again during hashing.
    Parent metadata-inspection costs are also debited. Without a token this
    function fails closed. Limits may be reduced but never increased.
    """
    started, initial_usage = time.monotonic(), _usage()
    paid_reads = 0
    prior_read = 0
    prior_wall = 0.0
    parent_read = 0
    parent_wall = 0.0
    reserved_reads = 0
    deadline = started + 7200
    output_created = False

    def accounting() -> dict[str, Any]:
        elapsed = time.monotonic() - started
        return dict(previous_audit_read_bytes=prior_read, new_read_bytes=paid_reads,
                    paid_parent_metadata_read_bytes=parent_read,
                    cumulative_read_bytes=prior_read + parent_read + paid_reads,
                    previous_audit_wall_seconds=prior_wall, new_wall_seconds=elapsed,
                    paid_parent_metadata_wall_seconds=parent_wall,
                    cumulative_wall_seconds=prior_wall + parent_wall + elapsed,
                    reserved_new_read_bytes=reserved_reads,
                    new_neural_rows=0, new_backward_rows=0, new_optimizer_steps=0,
                    new_native_controls=0, new_physics_substeps=0, new_mppi_queries=0)

    def check() -> None:
        _need(time.monotonic() < deadline, "Original cumulative audit wall budget exhausted")
        _need(_usage()["peak_rss_bytes"] <= MAX_RSS, "Audit-output hash RSS exceeds4GiB")

    try:
        seconds = _finite(max_audit_seconds, "audit wall limit")
        _need(0 < seconds <= 7200, "Audit wall limit may not exceed7200s")
        for value, cap, label in ((max_audit_read_bytes, 64 * GIB, "read"),
                                  (max_audit_output_bytes, 4 * GIB, "output")):
            _need(type(value) is int and 0 < value <= cap, "Invalid or increased audit " + label + " limit")
        _need(isinstance(admission, Mapping), "Trusted admission token required before audit payload reads")
        token = dict(admission)
        prior_wall = _finite(token.get("prior_audit_wall_seconds"), "prior_audit_wall_seconds")
        parent_wall = _finite(token.get("paid_parent_metadata_wall_seconds"),
                              "paid_parent_metadata_wall_seconds")
        for key in ("prior_source_read_bytes", "paid_parent_metadata_read_bytes"):
            _need(type(token.get(key)) is int and token[key] >= 0, "Invalid admission " + key)
        prior_read = token["prior_source_read_bytes"]
        parent_read = token["paid_parent_metadata_read_bytes"]
        for key, maximum in (("audit_exit", CHUNK), ("audit_receipt", 64 * CHUNK)):
            binding = token.get(key)
            _need(isinstance(binding, dict) and type(binding.get("bytes")) is int
                  and 0 < binding["bytes"] <= maximum, "Invalid admission " + key + " byte binding")
            value = binding.get("sha256")
            _need(isinstance(value, str) and len(value) == 64
                  and all(character in "0123456789abcdef" for character in value),
                  "Invalid admission " + key + " SHA256")
        _need(parent_read >= token["audit_exit"]["bytes"] + token["audit_receipt"]["bytes"],
              "Parent inspection read debit does not cover both admission headers")
        token_bytes = json.dumps(token, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        _need(len(token_bytes) <= 65536, "Admission token exceeds64KiB")
        token = json.loads(token_bytes)
        deadline = started + seconds - prior_wall - parent_wall
        check()
        _need(prior_read + parent_read <= max_audit_read_bytes,
              "Original remaining read reservation exhausted by prior metadata debits")
        candidate = Path(audit_outer).absolute()
        _need(not candidate.is_symlink() and candidate.is_dir(), "Closed audit directory required")
        root = candidate.resolve(strict=True)
        destination = Path(out_json).absolute()
        _need(not destination.exists() and not destination.is_symlink(), "Catalog output is create-only")
        _need(destination.parent.is_dir() and not destination.parent.is_symlink(), "Existing ordinary output parent required")
        destination = destination.resolve(strict=False)
        _need(not destination.is_relative_to(root), "Catalog output must be outside audit tree")
        files, directories = _inventory(root)
        check()
        _need(not any(name in directories for name in ("worker", "source", "prepared"))
              and not any(Path(name).name in {"arrays.bin", "arrays.index.jsonl", "ledger.jsonl"}
                          or Path(name).suffix == ".pt" for name in files),
              "Scientific run/checkpoint/raw streams are outside audit-output scope")
        _need("exit.json" in files and "result/audit.json" in files, "Closed exit and audit receipts are required")
        _need(files["exit.json"][3] == token["audit_exit"]["bytes"]
              and files["result/audit.json"][3] == token["audit_receipt"]["bytes"],
              "Admission header size differs before reads")
        header_bytes = files["exit.json"][3] + files["result/audit.json"][3]
        file_bytes = sum(identity[3] for identity in files.values())
        reserved_reads = header_bytes + file_bytes
        _need(prior_read + parent_read + reserved_reads <= max_audit_read_bytes,
              "Closed audit outputs exceed original remaining read reservation")
        skeleton = {relative: dict(bytes=identity[3], sha256="0" * 64)
                    for relative, identity in files.items()}
        catalog_reservation = 65536 + len(json.dumps(skeleton, sort_keys=True, indent=2).encode())
        _need(file_bytes + catalog_reservation <= max_audit_output_bytes,
              "Audit outputs plus catalog exceed original4GiB output envelope")
        check()

        def read_exact(relative: str, *, capture: bool = False) -> tuple[str, bytes | None]:
            nonlocal paid_reads
            check()
            path, expected = root / relative, files[relative]
            _need(_identity(path) == expected, "Audit output changed before read: " + relative)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            digest, parts, remaining = hashlib.sha256(), [], expected[3]
            try:
                value = os.fstat(descriptor)
                opened = (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                          value.st_mtime_ns, value.st_ctime_ns)
                _need(opened == expected, "Audit output changed at open: " + relative)
                while remaining:
                    check()
                    amount = min(CHUNK, remaining)
                    _need(prior_read + parent_read + paid_reads + amount <= max_audit_read_bytes,
                          "Original cumulative audit read budget exhausted")
                    data = os.read(descriptor, amount)
                    paid_reads += len(data)
                    _need(bool(data), "Audit output truncated during read: " + relative)
                    remaining -= len(data)
                    digest.update(data)
                    if capture:
                        parts.append(data)
                value = os.fstat(descriptor)
                closed = (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                          value.st_mtime_ns, value.st_ctime_ns)
                _need(closed == expected and _identity(path) == expected,
                      "Audit output changed while reading: " + relative)
            finally:
                os.close(descriptor)
            return digest.hexdigest(), b"".join(parts) if capture else None

        exit_sha, payload = read_exact("exit.json", capture=True)
        _need(exit_sha == token["audit_exit"]["sha256"], "Admission exit SHA256 differs")
        outer_exit = _json(payload)
        receipt_sha, payload = read_exact("result/audit.json", capture=True)
        _need(receipt_sha == token["audit_receipt"]["sha256"], "Admission receipt SHA256 differs")
        receipt = _json(payload)
        _need(type(outer_exit.get("exit_code")) is int and outer_exit["exit_code"] == 0
              and outer_exit.get("leader_reaped") is True
              and outer_exit.get("stopping_reason") is None
              and outer_exit.get("process_group_kill_requested") is False
              and outer_exit.get("unknown_partial_work_possible") is False,
              "Scientific audit did not exit successfully and close/reap normally")
        _need(receipt.get("schema") == "walker-student-consequence-c1.0-audit-v1"
              and receipt.get("passed") is True
              and receipt.get("full_scientific_audit_passed") is True,
              "Scientific audit receipt has not passed")
        observed_wall = max(_finite(outer_exit.get("wall_seconds"), "outer audit wall"),
                            _finite(receipt.get("wall_seconds"), "inner audit wall"))
        previous_read = receipt.get("accounting", {}).get("source_read_bytes")
        _need(type(previous_read) is int and previous_read >= 0,
              "Missing original aggregate audit-read census")
        _need(observed_wall == prior_wall and previous_read == prior_read,
              "Trusted admission prior facts differ from bound audit headers")
        exact_files = {}
        for relative in sorted(files):
            digest, _ = read_exact(relative)
            exact_files[relative] = dict(bytes=files[relative][3], sha256=digest)
        _need(exact_files["exit.json"]["sha256"] == exit_sha
              and exact_files["result/audit.json"]["sha256"] == receipt_sha,
              "Admission receipt bytes changed before hashing")
        final_files, final_directories = _inventory(root)
        _need(final_files == files and final_directories == directories,
              "Closed audit tree changed during catalog construction")
        check()
        final_usage = _usage()
        result = dict(
            schema="walker-c1-closed-audit-output-catalog-v2", passed=True,
            scope="Exact checksums of closed scientific audit OUTPUTS only; no scientific run files opened.",
            audit_outer=str(root), files=exact_files, file_count=len(exact_files),
            file_bytes=file_bytes, source_audit_receipt=exact_files["result/audit.json"],
            source_audit_exit=exact_files["exit.json"], accounting=accounting(),
            admission=token,
            admission_token_canonical_sha256=hashlib.sha256(token_bytes).hexdigest(),
            budgets=dict(max_audit_seconds=seconds, max_audit_read_bytes=max_audit_read_bytes,
                         max_audit_output_bytes=max_audit_output_bytes, peak_rss_bytes=MAX_RSS,
                         reserved_catalog_bytes=catalog_reservation),
            process=dict(
                user_seconds=final_usage["user_seconds"] - initial_usage["user_seconds"],
                system_seconds=final_usage["system_seconds"] - initial_usage["system_seconds"],
                process_seconds=final_usage["process_seconds"] - initial_usage["process_seconds"],
                peak_rss_bytes=final_usage["peak_rss_bytes"],
                kernel_input_blocks=final_usage["kernel_input_blocks"] - initial_usage["kernel_input_blocks"],
                kernel_output_blocks=final_usage["kernel_output_blocks"] - initial_usage["kernel_output_blocks"],
                kernel_block_size_bytes=None,
                kernel_io_scope="OS getrusage block counts; no conversion to bytes is assumed.",
            ),
            timing_scope="Hashing complete, before final catalog construction/serialization. Helper outer exit is authoritative for full wall/CPU including serialization, publication/fsync and process closure.",
            admission_requires_successful_reaped_helper_exit=True,
            cost_currency=None,
        )
        encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        _need(len(encoded) <= catalog_reservation and file_bytes + len(encoded) <= max_audit_output_bytes,
              "Catalog exceeded its reserved output allowance")
        check()
        with destination.open("xb") as stream:
            output_created = True
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        check()
        return result
    except BaseException as error:
        if isinstance(error, CatalogError):
            error.accounting = dict(accounting(), output_created=output_created)
            raise
        raise CatalogError(str(error), dict(accounting(), output_created=output_created)) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-outer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission", help="Trusted admission token as one JSON literal argv value, never a file path")
    parser.add_argument("--max-audit-seconds", type=float, default=7200)
    parser.add_argument("--max-audit-read-bytes", type=int, default=64 * GIB)
    parser.add_argument("--max-audit-output-bytes", type=int, default=4 * GIB)
    args = parser.parse_args()
    try:
        result = hash_audit_outputs(args.audit_outer, args.out,
                                    max_audit_seconds=args.max_audit_seconds,
                                    max_audit_read_bytes=args.max_audit_read_bytes,
                                    max_audit_output_bytes=args.max_audit_output_bytes,
                                    admission=None if args.admission is None else _json(args.admission))
    except CatalogError as error:
        print(json.dumps(dict(passed=False, error=str(error), accounting=error.accounting),
                         sort_keys=True, allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps(dict(passed=True, file_count=result["file_count"],
                          file_bytes=result["file_bytes"], accounting=result["accounting"]),
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
