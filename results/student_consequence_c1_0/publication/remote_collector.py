"""Create a compact C1 evidence packet after closed worker/audit/report stages.

Standalone standard library only. Never opens scientific arrays, checkpoints,
raw indexes or work journals. Their already verified hashes remain declarations
with explicit origin and current filesystem availability.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import tarfile
import time
from pathlib import Path

MIB = 1024**2
MAX_PAYLOAD = 128 * MIB
MAX_OUTPUT = 512 * MIB
FROZEN_SOURCE_COMMIT = "56b737413884c031bb65bcc331b48ed700e59540"
ALLOWED = {".json", ".jsonl", ".md", ".csv", ".svg", ".png", ".txt", ".py", ".time"}
RAW_SUFFIXES = {".npz", ".npy", ".pt", ".bin", ".sqlite", ".h5", ".mp4"}
SOURCE_FILES = (
    "src/gps/mppi_gps/policy.py",
    "src/gps/mppi_gps/consequence_inputs.py",
    "src/gps/mppi_gps/consequence_native.py",
    "src/gps/mppi_gps/consequence_learning.py",
    "src/gps/mppi_gps/consequence_student.py",
    "src/gps/mppi_gps/consequence_storage.py",
    "src/gps/mppi_gps/consequence_execution.py",
    "src/gps/mppi_gps/consequence_preflight.py",
    "src/gps/mppi_gps/experiments/student_consequence.py",
    "src/gps/mppi_gps/audits/student_consequence.py",
    "src/gps/mppi_gps/reporting/student_consequence.py",
    "scripts/diagnostics/report_student_consequence.py",
    "configs/studies/student_consequence_c1_0.json",
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def safe(root, relative):
    part = Path(relative)
    need(not part.is_absolute() and ".." not in part.parts, "Unsafe packet/source path")
    path = (root / part).resolve()
    need(
        path.is_relative_to(root) and path.is_file() and not (root / part).is_symlink(),
        "Missing or escaped source file: " + str(path),
    )
    return path


def identify(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class Packet:
    def __init__(self, out, limit):
        need(65536 <= limit <= MAX_PAYLOAD, "Packet payload limit must be at most128 MiB")
        need(not out.exists(), "Collector output is create-only")
        out.mkdir(parents=True)
        self.out, self.limit = out, limit
        self.reserve = min(8 * MIB, limit // 8)
        self.files, self.missing, self.identities = {}, [], {}
        self.json_bindings = {}
        self.bytes = self.read_bytes = 0
        self.started = time.monotonic()

    def check(self):
        need(time.monotonic() - self.started <= 1800, "Collector 1800-second limit exhausted")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if os.uname().sysname == "Darwin" else 1024
        )
        need(peak <= 2 * 1024**3, "Collector 2 GiB RSS limit exhausted")

    def read_json(self, path):
        self.check()
        need(
            path.suffix == ".json" and path.stat().st_size <= 64 * MIB,
            "Expected bounded saved JSON",
        )
        before = identify(path)
        payload = path.read_bytes()
        self.read_bytes += len(payload)
        need(
            before == identify(path) and len(payload) == before[2], "JSON changed during collection"
        )
        self.identities[path] = before
        self.json_bindings[path] = dict(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
        return json.loads(
            payload, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON"))
        )

    def copy(self, source, relative, *, expected=None):
        self.check()
        need(source.is_file() and not source.is_symlink(), "Missing or symlinked compact source")
        need(source.suffix in ALLOWED, "Raw/binary source is not eligible for compact copying")
        need(
            source.name not in {"ledger.jsonl", "arrays.index.jsonl", "work.jsonl"},
            "Raw journal/index cannot enter compact packet",
        )
        need(
            source.suffix != ".jsonl" or source.name == "evaluations.jsonl",
            "Only scalar evaluations JSONL is compact-copy eligible",
        )
        need(
            relative not in self.files and len(self.files) < 4997,
            "Duplicate path or packet file count exhausted",
        )
        size = source.stat().st_size
        need(
            size <= 64 * MIB and self.bytes + size <= self.limit - self.reserve,
            "Compact packet budget exhausted before copy",
        )
        before = identify(source)
        target = self.out / relative
        need(target.resolve().is_relative_to(self.out.resolve()), "Packet target escapes output")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            remaining = size
            while remaining:
                self.check()
                part = incoming.read(min(MIB, remaining))
                need(bool(part), "Truncated compact source")
                self.read_bytes += len(part)
                outgoing.write(part)
                digest.update(part)
                remaining -= len(part)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        row = dict(bytes=size, sha256=digest.hexdigest(), source=str(source))
        if expected is not None:
            need(row["sha256"] == expected["sha256"], "Compact source SHA differs: " + str(source))
            need(
                "bytes" not in expected or row["bytes"] == expected["bytes"],
                "Compact source size differs",
            )
        need(before == identify(source), "Source changed during copy")
        self.identities[source] = before
        self.files[relative] = row
        self.bytes += size
        return row

    def optional(self, source, relative, reason, *, expected=None):
        if source.is_file():
            return self.copy(source, relative, expected=expected)
        self.missing.append(dict(file=relative, reason=reason))
        return None

    def generated(self, relative, value):
        self.check()
        need(
            relative not in self.files and len(self.files) < 5000,
            "Packet metadata roster exhausted",
        )
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        need(self.bytes + len(payload) <= self.limit, "Packet generated metadata exceeds 128 MiB")
        path = self.out / relative
        with path.open("xb") as stream:
            stream.write(payload)
        self.files[relative] = dict(
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            source="collector_generated_metadata",
        )
        self.bytes += len(payload)


def result_root(path, filename):
    for candidate in (path, path / "result"):
        if (candidate / filename).is_file():
            return candidate
    raise ValueError("Missing closed stage result: " + str(path / filename))


class LimitedArchive:
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, value):
        need(
            self.stream.tell() + len(value) <= self.limit,
            "Compressed archive exceeds 128 MiB before write",
        )
        return self.stream.write(value)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def failed_audit_closure(packet, artifacts, auditroot, auditouter):
    """Preserve this coordinator failure without inventing an outer exit."""
    companion = safe(artifacts, "audit_launcher_failure.json")
    record = packet.read_json(companion)
    need(
        record.get("schema") == "walker-c1-audit-launcher-failure-v1",
        "Unknown audit launcher failure schema",
    )
    need(
        record.get("frozen_source_commit") == FROZEN_SOURCE_COMMIT,
        "Wrong failed-launch frozen source commit",
    )
    need(
        record.get("coordinator_exit_code") == 1
        and type(record.get("coordinator_exit_code")) is int,
        "Failure companion lacks original coordinator exit1",
    )
    need(
        record.get("original_audit_exit_exists") is False
        and not (auditouter / "exit.json").exists(),
        "Original audit exit must remain absent",
    )
    need(
        "original_child_exit_code" in record
        and record["original_child_exit_code"] is None
        and record.get("original_child_exit_status_known") is False,
        "Original child exit status must remain unknown",
    )
    need(
        record.get("overall_audit_closure_passed") is False
        and record.get("audit_coordinator_failed") is True,
        "Failure companion overstates audit closure",
    )
    expected = {
        "scientific_receipt": ("audit_001/result/audit.json", auditroot / "audit.json", None),
        "traceback": (
            "audit_launch_001/stdout.txt",
            None,
            "accounting/audit_launcher_traceback.txt",
        ),
        "gnu_time": ("audit_launch_001/time.txt", None, "accounting/audit_launcher_gnu_time.txt"),
    }
    for name, (relative, actual, destination) in expected.items():
        descriptor = record.get(name, {})
        need(
            descriptor.get("file") == relative
            and type(descriptor.get("bytes")) is int
            and descriptor["bytes"] > 0,
            "Failure companion lacks exact original binding: " + name,
        )
        source = safe(artifacts, relative)
        if actual is not None:
            need(
                source == actual
                and packet.json_bindings.get(actual)
                == {key: descriptor[key] for key in ("bytes", "sha256")},
                "Failure companion/scientific receipt binding differs",
            )
        else:
            packet.copy(source, destination, expected=descriptor)
    packet.copy(companion, "accounting/audit_launcher_failure.json")
    optional = record.get("audit_outputs_catalog")
    if optional is not None:
        packet.copy(
            safe(artifacts, optional["file"]),
            "accounting/audit_outputs_catalog.json",
            expected=optional,
        )
    packet.missing.append(
        dict(
            file="audit/operator_exit.json",
            reason="Original audit coordinator failed before persisting its child exit receipt; no replacement exit was synthesized. See accounting/audit_launcher_failure.json.",
        )
    )
    return record


def collect(
    run: Path,
    artifacts: Path,
    auditdir: Path,
    reportdir: Path,
    out: Path,
    *,
    max_payload_bytes=MAX_PAYLOAD,
):
    """Collect only closed, bounded metadata; no report/audit/research execution."""
    run, artifacts, auditdir, reportdir, out = [
        Path(p).resolve() for p in (run, artifacts, auditdir, reportdir, out)
    ]
    packet = Packet(out, max_payload_bytes)
    auditroot = result_root(auditdir, "audit.json")
    reportroot = result_root(reportdir, "receipt.json")
    config = packet.read_json(safe(run, "config.json"))
    manifest = packet.read_json(safe(run, "manifest.json"))
    worker_exit = packet.read_json(safe(run, "worker_exit.json"))
    need(worker_exit.get("leader_reaped") is True, "Worker process is not recorded closed")
    result = (
        packet.read_json(run / "worker_result.json")
        if (run / "worker_result.json").is_file()
        else {}
    )
    need(
        bool(result) or worker_exit.get("exit_code") != 0,
        "Successful worker lacks its required result",
    )
    audit = packet.read_json(auditroot / "audit.json")
    report = packet.read_json(reportroot / "receipt.json")
    auditouter = auditdir if auditdir != auditroot else auditdir.parent
    failure = None
    if not (auditouter / "exit.json").is_file():
        failure = failed_audit_closure(packet, artifacts, auditroot, auditouter)
    elif (artifacts / "audit_launcher_failure.json").exists():
        raise ValueError("Failure companion contradicts an existing original audit exit")
    overall_audit_closure_passed = False
    need(
        config.get("study") == "C1.0" and config.get("user_authorized") is True,
        "Wrong frozen study/configuration",
    )
    need("c1" in str(audit.get("schema", "")).lower(), "Unknown C1 audit receipt")
    need(report.get("schema") == "walker-c1-saved-report-receipt-v1", "Unknown C1 report receipt")
    for name in ("worker_result.json", "config.json"):
        key = name.removesuffix(".json") + "_sha256"
        if audit.get(key) is not None:
            need(
                packet.json_bindings.get(run / name, {}).get("sha256") == audit[key],
                "Audit/run association differs",
            )
        elif audit.get("full_scientific_audit_passed"):
            raise ValueError("Full C1 audit lacks its required run association")
    for relative in (
        "config.json",
        "manifest.json",
        "input_manifest.json",
        "prerequisites.json",
        "worker_exit.json",
        "prepared/catalog.json",
        "prepared/prepared_input_manifest.json",
        "prepared/preparation_receipt.json",
        "prepared/metadata/source_bindings.json",
    ):
        packet.copy(safe(run, relative), "run/" + relative)
    packet.optional(
        run / "worker_result.json",
        "run/worker_result.json",
        "Worker terminated without a final result; scientific closure is incomplete",
    )
    for relative in ("worker_stdout.txt", "launch_ownership.json", "audit_ownership.json"):
        packet.copy(safe(run, relative), "run/" + relative)
    for source in sorted(run.iterdir()):
        if (
            source.is_file()
            and ("time" in source.name.lower())
            and source.suffix in {".txt", ".time", ".json"}
        ):
            packet.copy(source, "run/" + source.name)
    bindings = packet.read_json(run / "prepared/metadata/source_bindings.json")
    for name in ("config.json", "manifest.json", "input_manifest.json", "worker_result.json"):
        preferred = run / "prepared/metadata/h5_source" / name
        source = (
            preferred if preferred.is_file() else Path(config["locations"]["source_run"]) / name
        )
        packet.copy(
            source,
            "run/prepared/metadata/h5_source/" + name,
            expected=bindings["metadata_files"]["run/" + name],
        )
    measured = bool(result.get("learning_measured"))
    standalone = bool(result.get("standalone_measured"))
    for name in (
        "replay_gate.json",
        "contrast_gate.json",
        "metrics.json",
        "actor_registry.json",
        "fit_records.json",
        "outcomes.json",
        "evaluations.jsonl",
    ):
        required = (
            (name == "replay_gate.json" and result.get("replay_gate", {}).get("passed"))
            or (name == "contrast_gate.json" and result.get("contrast_gate") is not None)
            or (name in {"metrics.json", "actor_registry.json", "fit_records.json"} and measured)
            or (name in {"outcomes.json", "evaluations.jsonl"} and standalone)
        )
        source = run / "worker" / name
        if required:
            packet.copy(safe(run, "worker/" + name), "run/worker/" + name)
        else:
            packet.optional(
                source,
                "run/worker/" + name,
                "Stage unmeasured/unavailable after the saved stop; no zeros substituted",
            )
    for source in sorted((run / "worker/learning").glob("*/*.json")):
        need(
            source.name in {"fit.json", "initial.json", "failure.json"},
            "Unexpected fitting JSON requires explicit collection review",
        )
        packet.copy(source, "run/" + str(source.relative_to(run)))
    packet.copy(auditroot / "audit.json", "audit/audit.json")
    stream_file = next(
        (
            auditroot / "stream" / name
            for name in ("audit.json", "receipt.json")
            if (auditroot / "stream" / name).is_file()
        ),
        None,
    )
    if stream_file is None:
        need(
            not audit.get("full_scientific_audit_passed"),
            "Full audit lacks required stream receipt",
        )
        packet.missing.append(
            dict(
                file="audit/stream/receipt.json",
                reason="Stream audit unavailable on early audit failure",
            )
        )
    else:
        packet.copy(stream_file, "audit/stream/" + stream_file.name)
    if audit.get("full_scientific_audit_passed"):
        need(audit.get("passed") is True, "Full scientific audit does not declare passed")
        stream_receipt = packet.read_json(stream_file)
        need(
            stream_receipt.get("structural_passed") is True
            and stream_receipt.get("complete_journal") is True,
            "Full scientific audit lacks closed stream integrity",
        )
        audited_gain = audit.get("weighting_confirmed")
        worker_gain = result.get("gates", {}).get("confirmed")
        if audit.get("scientific_contrast_completed") is True:
            need(
                type(audited_gain) is bool
                and type(worker_gain) is bool
                and audited_gain == worker_gain,
                "Audit/worker Boolean weighting verdict differs",
            )
        else:
            need(
                (audited_gain is None or audited_gain is False)
                and (worker_gain is None or worker_gain is False),
                "Unmeasured contrast cannot claim a confirmed gain",
            )
    for name, derivative in audit.get("derivatives", {}).items():
        need(
            name == "assay_scalars" and derivative["file"] == "assay_scalars.csv",
            "Unreviewed audit derivative",
        )
        packet.copy(
            safe(auditroot, derivative["file"]), "audit/" + derivative["file"], expected=derivative
        )
    need(
        "assay_scalars" in audit.get("derivatives", {})
        or not audit.get("full_scientific_audit_passed"),
        "Full audit lacks its scalar derivative",
    )
    for stage, outer, resultpath in (
        ("audit", auditdir, auditroot),
        ("report", reportdir, reportroot),
    ):
        outer = outer if outer != resultpath else outer.parent
        names = (
            ("stdout.txt",)
            if stage == "audit" and failure is not None
            else ("exit.json", "stdout.txt")
        )
        for name in names:
            packet.copy(safe(outer, name), stage + "/operator_" + name)
        if stage != "audit" or failure is None:
            closed = packet.read_json(outer / "exit.json")
            need(
                closed.get("leader_reaped") is True, stage + " outer process is not recorded closed"
            )
            if stage == "report":
                need(closed.get("exit_code") == 0, "Report outer process must have a clean exit")
            else:
                overall_audit_closure_passed = (
                    closed.get("exit_code") == 0 and audit.get("passed") is True
                )
        for source in sorted(outer.iterdir()):
            if (
                source.is_file()
                and source.name not in {"exit.json", "stdout.txt"}
                and source.suffix in {".json", ".txt", ".py", ".time"}
            ):
                packet.copy(source, stage + "/operator/" + source.name)
    packet.copy(reportroot / "receipt.json", "report/receipt.json")
    for relative, expected in report["outputs"].items():
        need(
            Path(relative).suffix in {".md", ".csv", ".svg", ".png", ".json"},
            "Unexpected bound report artifact",
        )
        packet.copy(safe(reportroot, relative), "report/" + relative, expected=expected)
    copied_by_source = {row["source"]: row for row in packet.files.values()}
    for source, expected in report.get("inputs", {}).items():
        actual = copied_by_source.get(str(Path(source).resolve()))
        need(
            actual is not None
            and actual["sha256"] == expected["sha256"]
            and actual["bytes"] == expected["bytes"],
            "A consumed report input is absent/different in the compact packet: " + source,
        )
    need(
        {"report.md", "curves.csv", "outcomes.csv", "costs.json"}.issubset(report["outputs"]),
        "Report lacks required readable tables/accounting",
    )
    source_hashes = {row["file"]: row["sha256"] for row in manifest["files"]}
    for relative in SOURCE_FILES:
        need(relative in source_hashes, "Frozen manifest lacks selected original implementation")
        packet.copy(
            safe(run / "source", relative),
            "source/" + relative,
            expected=dict(sha256=source_hashes[relative]),
        )
    processor = Path(report["source_code"]["file"]).resolve()
    need(
        any(processor.is_relative_to(base) for base in (run, artifacts, reportdir)),
        "Report implementation escapes declared collection roots",
    )
    packet.copy(processor, "postprocessor/student_consequence.py", expected=report["source_code"])
    packet.copy(Path(__file__).resolve(), "publication/remote_collector.py")
    skipped = {
        "source",
        "prepared",
        "evidence",
        "active",
        "learning",
        ".git",
        ".venv",
        "__pycache__",
    }
    operator_bytes = 0
    for directory, dirs, names in os.walk(artifacts):
        directory = Path(directory)
        dirs[:] = [
            name
            for name in dirs
            if name not in skipped
            and not (directory / name).resolve().is_relative_to(out)
            and (directory / name).resolve() not in {auditdir, reportdir}
            and not (directory / name).is_symlink()
        ]
        for name in sorted(names):
            source = directory / name
            if source.suffix not in {".json", ".txt", ".py", ".md", ".time"}:
                continue
            if source.resolve().is_relative_to(out):
                continue
            operator_bytes += source.stat().st_size
            need(
                operator_bytes <= 64 * MIB, "Operator metadata exceeds 64 MiB; no silent truncation"
            )
            packet.copy(source, "operator/" + str(source.relative_to(artifacts)))
    external = {}

    def declare(relative, value, origin):
        if relative in external:
            previous = external[relative]
            need(
                not value.get("sha256")
                or not previous.get("sha256")
                or previous["sha256"] == value["sha256"],
                "Conflicting existing external hashes",
            )
            return
        path = run / relative
        external[relative] = dict(
            value,
            external_path=str(path),
            available=path.is_file(),
            actual_bytes=path.stat().st_size if path.is_file() else None,
            verified_by=origin,
            collector_rehashed=False,
        )

    for relative, value in audit.get("input_files", {}).items():
        if Path(relative).suffix in RAW_SUFFIXES or relative == "worker/assays.jsonl":
            declare(relative, value, "closed_scientific_audit_input_files")
    sizes = audit.get("accounting", {}).get("evidence_file_bytes", {})
    hashes = audit.get("stream", {}).get("file_hashes", {})
    for name in ("arrays.bin", "arrays.index.jsonl", "ledger.jsonl"):
        path = run / "worker/evidence" / name
        declare(
            "worker/evidence/" + name,
            dict(
                bytes=sizes.get(name, path.stat().st_size if path.is_file() else None),
                sha256=hashes.get(name),
            ),
            "closed_stream_audit" if name in hashes else "checksum_unavailable_after_partial_audit",
        )
    prepared = packet.read_json(run / "prepared/prepared_input_manifest.json")
    for row in prepared["files"]:
        if Path(row["file"]).suffix in RAW_SUFFIXES:
            declare("prepared/" + row["file"], row, "prepared_manifest_and_closed_audit")
    registry = result.get("checkpoints", {})
    for row in registry.values():
        declare(
            row["file"],
            dict(
                bytes=(run / row["file"]).stat().st_size if (run / row["file"]).is_file() else None,
                sha256=row["sha256"],
            ),
            "saved_registry_checked_by_scientific_audit"
            if audit.get("full_scientific_audit_passed")
            else "saved_registry_partial_audit_scope",
        )
    fit_records = dict(result.get("fit_records", {}))
    for source in sorted((run / "worker/learning").glob("*/fit.json")):
        value = packet.read_json(source)
        alias = source.parent.name
        need(
            alias not in fit_records or fit_records[alias] == value,
            "Saved fit record copies disagree",
        )
        fit_records[alias] = value
    for alias, record in fit_records.items():
        for key in ("actual_indices", "curve", "predictions", "prediction_valid"):
            if key in record:
                declare(
                    f"worker/learning/{alias}/" + record[key]["file"],
                    record[key],
                    "saved_fit_record_checked_by_scientific_audit"
                    if audit.get("full_scientific_audit_passed")
                    else "saved_fit_record_partial_audit_scope",
                )
    for directory in (run / "worker", run / "prepared"):
        for source in sorted(directory.rglob("*")):
            if source.is_file() and (source.suffix in RAW_SUFFIXES or source.suffix == ".jsonl"):
                declare(
                    str(source.relative_to(run)),
                    dict(bytes=source.stat().st_size, sha256=None),
                    "filesystem_inventory_only_checksum_not_available",
                )
    audit_external = []
    for source in sorted(auditroot.rglob("*")):
        if source.is_file() and source.suffix in {".sqlite", ".jsonl", ".npy", ".npz"}:
            audit_external.append(
                dict(
                    file=str(source),
                    bytes=source.stat().st_size,
                    sha256=None,
                    collector_rehashed=False,
                    availability="closed_external_audit_output_checksum_not_declared",
                )
            )
    packet.generated(
        "external_catalog.json",
        dict(
            schema="walker-c1-external-evidence-catalog-v1",
            run_directory=str(run),
            files=external,
            audit_external_outputs=audit_external,
            inherited_H5_run_directory=config["locations"]["source_run"],
            inherited_H5_raw=bindings["external_evidence_files"],
            scope="Large scientific files were not copied or rehashed. Known hashes retain their original verification scope; missing checksums/availability remain explicit.",
        ),
    )
    for path, identity in packet.identities.items():
        need(identify(path) == identity, "Collection source changed after copying")
    if failure is not None:
        need(
            not (auditouter / "exit.json").exists(),
            "Original audit exit appeared during failure collection",
        )
    packet.generated(
        "collection_receipt.json",
        dict(
            schema="walker-c1-metadata-collection-receipt-v1",
            scope="Closed saved metadata only; zero NN/physics/optimizer work",
            source_read_bytes=packet.read_bytes,
            payload_bytes_before_closure=packet.bytes,
            wall_seconds=time.monotonic() - packet.started,
            missing=packet.missing,
            originals_unchanged=True,
        ),
    )
    packet_manifest = dict(
        schema="walker-c1-compact-packet-v1",
        run_id=config["locations"]["run_id"],
        scientific_status=result.get("status", "NO_WORKER_RESULT_INCOMPLETE"),
        audit_passed=audit.get("passed", False),
        full_scientific_audit_passed=audit.get("full_scientific_audit_passed", False),
        scientific_contrast_completed=audit.get("scientific_contrast_completed", False),
        overall_audit_closure_passed=overall_audit_closure_passed,
        audit_coordinator_failed=failure is not None,
        weighting_confirmed=bool(
            audit.get("full_scientific_audit_passed") and result.get("gates", {}).get("confirmed")
        ),
        recovery_or_coupling_tested=False,
        files=dict(packet.files),
        missing=packet.missing,
        manifest_excludes_itself=True,
        original_run_directory=str(run),
    )
    packet.generated("packet_manifest.json", packet_manifest)
    archive = out.with_name(out.name + ".tar.gz")
    need(not archive.exists(), "Archive output is create-only")
    with archive.open("xb") as raw:
        with gzip.GzipFile(
            fileobj=LimitedArchive(raw, max_payload_bytes), mode="wb", mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for relative in sorted(packet.files):
                    path = out / relative
                    header = tarfile.TarInfo("packet/" + relative)
                    header.size, header.mode, header.mtime = path.stat().st_size, 0o644, 0
                    with path.open("rb") as incoming:
                        tar.addfile(header, incoming)
                    packet.check()
                    need(raw.tell() <= max_payload_bytes, "Compressed archive exceeds 128 MiB")
        raw.flush()
        os.fsync(raw.fileno())
    need(
        archive.stat().st_size <= max_payload_bytes
        and packet.bytes + archive.stat().st_size <= MAX_OUTPUT,
        "Collector output/transfer ceiling exceeded",
    )
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for part in iter(lambda: stream.read(MIB), b""):
            digest.update(part)
    receipt = dict(
        schema="walker-c1-archive-receipt-v1",
        archive=str(archive),
        bytes=archive.stat().st_size,
        sha256=digest.hexdigest(),
        payload_bytes=packet.bytes,
        payload_files=len(packet.files),
        scientific_status=packet_manifest["scientific_status"],
        full_scientific_audit_passed=packet_manifest["full_scientific_audit_passed"],
        overall_audit_closure_passed=overall_audit_closure_passed,
        audit_coordinator_failed=failure is not None,
        wall_seconds=time.monotonic() - packet.started,
        new_neural_rows=0,
        new_native_controls=0,
        new_optimizer_steps=0,
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    receipt.update(
        metadata_source_read_bytes=packet.read_bytes,
        archive_payload_read_bytes=packet.bytes,
        archive_hash_read_bytes=archive.stat().st_size,
        tracked_file_read_bytes=packet.read_bytes + packet.bytes + archive.stat().st_size,
        output_file_bytes_before_receipt=packet.bytes + archive.stat().st_size,
        process_lifetime_cpu=dict(user_seconds=usage.ru_utime, system_seconds=usage.ru_stime),
        peak_rss_bytes=usage.ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024),
        limits=dict(
            wall_seconds=1800,
            peak_rss_bytes=2 * 1024**3,
            payload_bytes=MAX_PAYLOAD,
            output_bytes=MAX_OUTPUT,
        ),
        receipt_write_excluded_from_snapshots=True,
    )
    with out.with_name(out.name + "_archive_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "art", "auditdir", "reportdir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-payload-bytes", type=int, default=MAX_PAYLOAD)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            collect(
                args.run,
                args.art,
                args.auditdir,
                args.reportdir,
                args.out,
                max_payload_bytes=args.max_payload_bytes,
            )
        )
    )


if __name__ == "__main__":
    main()
