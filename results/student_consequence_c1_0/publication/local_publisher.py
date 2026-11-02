"""Verify an exact compact C1 archive and create a publication capsule.

Standalone standard library only. Does not run packaged Python, open external
evidence, access Git/network, train a policy, render a report or run an audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import resource
import tarfile
import time
from pathlib import Path, PurePosixPath

MIB = 1024**2
MAX_PAYLOAD = 128 * MIB
FROZEN_SOURCE_COMMIT = "56b737413884c031bb65bcc331b48ed700e59540"
ALLOWED = {".json", ".jsonl", ".md", ".csv", ".svg", ".png", ".txt", ".py", ".time"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(value):
    return json.loads(
        value, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON"))
    )


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(MIB), b""):
            digest.update(part)
    return digest.hexdigest()


def verify_failed_closure(payloads):
    """Authenticate a documented failed coordinator, never manufacture exit0."""
    key = "accounting/audit_launcher_failure.json"
    need(key in payloads, "Failed audit closure lacks required failure companion")
    record = read_json(payloads[key])
    need(
        record.get("schema") == "walker-c1-audit-launcher-failure-v1"
        and record.get("frozen_source_commit") == FROZEN_SOURCE_COMMIT,
        "Failure companion schema/source differs",
    )
    need(
        type(record.get("coordinator_exit_code")) is int and record["coordinator_exit_code"] == 1,
        "Failure companion lacks original coordinator exit1",
    )
    need(
        record.get("original_audit_exit_exists") is False
        and "audit/operator_exit.json" not in payloads,
        "Original audit exit must remain absent",
    )
    need(
        not any(name.endswith("audit_001/exit.json") for name in payloads),
        "Packet fabricates an original audit exit",
    )
    need(
        "original_child_exit_code" in record
        and record["original_child_exit_code"] is None
        and record.get("original_child_exit_status_known") is False,
        "Original child exit must remain unknown",
    )
    need(
        record.get("overall_audit_closure_passed") is False
        and record.get("audit_coordinator_failed") is True,
        "Failure companion overstates closure",
    )
    bindings = {
        "scientific_receipt": ("audit_001/result/audit.json", "audit/audit.json"),
        "traceback": ("audit_launch_001/stdout.txt", "accounting/audit_launcher_traceback.txt"),
        "gnu_time": ("audit_launch_001/time.txt", "accounting/audit_launcher_gnu_time.txt"),
    }
    for name, (original, copied) in bindings.items():
        descriptor = record.get(name, {})
        need(
            descriptor.get("file") == original and copied in payloads,
            "Failure companion lacks original proof: " + name,
        )
        need(
            len(payloads[copied]) == descriptor.get("bytes")
            and hashlib.sha256(payloads[copied]).hexdigest() == descriptor.get("sha256"),
            "Failure proof hash/size differs: " + name,
        )
    need(
        b"Traceback" in payloads["accounting/audit_launcher_traceback.txt"]
        and b"Pinned Linux worker RSS could not be measured"
        in payloads["accounting/audit_launcher_traceback.txt"],
        "Original coordinator traceback does not match this failure",
    )
    need(
        re.search(
            rb"(?m)^\s*Exit status:\s*1\s*$", payloads["accounting/audit_launcher_gnu_time.txt"]
        )
        is not None,
        "Original GNU time does not record coordinator exit1",
    )
    optional = record.get("audit_outputs_catalog")
    if optional is not None:
        data = payloads.get("accounting/audit_outputs_catalog.json", b"")
        need(
            data
            and len(data) == optional.get("bytes")
            and hashlib.sha256(data).hexdigest() == optional.get("sha256"),
            "Optional generated audit-output catalog differs",
        )
    return record


def recorded_outcome_text(result):
    panels = result.get("student_outcomes")
    if not panels:
        return "Standalone outcome totals are unavailable or unmeasured in this saved result; missing episodes are not counted as zeros."
    parts, total, walking, survival = [], 0, 0, 0
    for name in ("primary", "confirmation", "source"):
        need(name in panels, "Recorded standalone outcome panel is missing")
        rows = [value for stream in panels[name].values() for value in stream.values()]
        for row in rows:
            need(
                all(type(row.get(key)) is int for key in ("episodes", "successes", "survived"))
                and 0 <= row["successes"] <= row["episodes"]
                and 0 <= row["survived"] <= row["episodes"],
                "Invalid recorded standalone outcome totals",
            )
        episodes = sum(row["episodes"] for row in rows)
        successes, survived = (
            sum(row["successes"] for row in rows),
            sum(row["survived"] for row in rows),
        )
        total += episodes
        walking += successes
        survival += survived
        parts.append(
            f"{name}: **{successes}/{episodes} walking, {survived}/{episodes} full survival**"
        )
    text = "Recorded standalone outcomes — " + "; ".join(parts) + "."
    if total and walking == survival == 0:
        text += f" **All {total} recorded episodes failed both walking and full survival.** This fixed test produced no competent standalone walker."
    return text


def publish(archive: Path, expected_sha256: str, out: Path):
    """Authenticate every member before creating a never-overwritten capsule."""
    archive, out = Path(archive).resolve(), Path(out).resolve()
    need(not out.exists(), "Publication target is create-only")
    need(
        archive.is_file() and archive.stat().st_size <= MAX_PAYLOAD,
        "Missing/oversized compact archive",
    )
    need(digest_file(archive) == expected_sha256, "Transferred archive SHA256 differs")
    started = time.monotonic()
    payloads, total = {}, 0

    def check():
        need(time.monotonic() - started <= 1800, "Publication 1800-second ceiling exhausted")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if os.uname().sysname == "Darwin" else 1024
        )
        need(peak <= 2 * 1024**3, "Publication 2 GiB RSS ceiling exhausted")

    with tarfile.open(archive, mode="r|gz") as stream:
        for member in stream:
            check()
            path = PurePosixPath(member.name)
            need(
                member.isfile()
                and not path.is_absolute()
                and ".." not in path.parts
                and len(path.parts) >= 2
                and path.parts[0] == "packet",
                "Unsafe archive member",
            )
            relative = str(PurePosixPath(*path.parts[1:]))
            need(
                relative not in payloads and len(payloads) < 5000,
                "Repeated or excessive archive members",
            )
            need(
                Path(relative).suffix in ALLOWED,
                "Scientific binary is forbidden in compact publication",
            )
            need(
                relative not in {"README.md", "capsule.json"},
                "Archive overwrites generated publication metadata",
            )
            need(
                member.size >= 0 and total + member.size <= MAX_PAYLOAD,
                "Expanded packet exceeds 128 MiB",
            )
            handle = stream.extractfile(member)
            need(handle is not None, "Unreadable archive member")
            payload = handle.read(member.size + 1)
            need(len(payload) == member.size, "Truncated archive member")
            payloads[relative] = payload
            total += member.size
    need("packet_manifest.json" in payloads, "Missing exact packet manifest")
    manifest = read_json(payloads["packet_manifest.json"])
    need(manifest.get("schema") == "walker-c1-compact-packet-v1", "Unknown packet schema")
    need(
        set(payloads) == set(manifest["files"]) | {"packet_manifest.json"},
        "Archive file roster differs from exact manifest",
    )
    for relative, expected in manifest["files"].items():
        payload = payloads[relative]
        need(
            len(payload) == expected["bytes"]
            and hashlib.sha256(payload).hexdigest() == expected["sha256"],
            "Packet file hash/size differs: " + relative,
        )
    for required in (
        "run/config.json",
        "run/manifest.json",
        "audit/audit.json",
        "report/receipt.json",
        "report/report.md",
        "report/costs.json",
        "report/curves.csv",
        "external_catalog.json",
    ):
        need(required in payloads, "Required capsule evidence missing: " + required)
    audit = read_json(payloads["audit/audit.json"])
    report = read_json(payloads["report/receipt.json"])
    coordinator_failed = manifest.get("audit_coordinator_failed") is True
    need(
        type(manifest.get("overall_audit_closure_passed")) is bool
        and type(manifest.get("audit_coordinator_failed")) is bool,
        "Missing explicit audit closure flags",
    )
    if coordinator_failed:
        verify_failed_closure(payloads)
        need(
            manifest["overall_audit_closure_passed"] is False,
            "Failed coordinator cannot claim overall audit closure",
        )
    else:
        need(
            "accounting/audit_launcher_failure.json" not in payloads
            and "audit/operator_exit.json" in payloads,
            "Clean audit closure lacks original exit proof",
        )
        closed = read_json(payloads["audit/operator_exit.json"])
        need(closed.get("leader_reaped") is True, "Audit leader is not recorded reaped")
        need(
            manifest["overall_audit_closure_passed"]
            == (closed.get("exit_code") == 0 and audit.get("passed") is True),
            "Overall audit closure claim differs from original exit",
        )
    need("report/operator_exit.json" in payloads, "Report outer exit proof is required")
    report_exit = read_json(payloads["report/operator_exit.json"])
    need(
        report_exit.get("leader_reaped") is True and report_exit.get("exit_code") == 0,
        "Report outer process must have a clean exit",
    )
    for name in ("config", "worker_result"):
        relative = "run/" + name + ".json"
        if audit.get(name + "_sha256") is not None:
            need(
                relative in payloads
                and hashlib.sha256(payloads[relative]).hexdigest() == audit[name + "_sha256"],
                "Audit/capsule association differs",
            )
    for relative, expected in report["outputs"].items():
        need(
            "report/" + relative in payloads
            and hashlib.sha256(payloads["report/" + relative]).hexdigest() == expected["sha256"],
            "Published report output differs",
        )
    for derivative in audit.get("derivatives", {}).values():
        relative = "audit/" + derivative["file"]
        need(
            relative in payloads
            and len(payloads[relative]) == derivative["bytes"]
            and hashlib.sha256(payloads[relative]).hexdigest() == derivative["sha256"],
            "Audited scalar derivative differs",
        )
        count = sum(1 for _ in csv.DictReader(io.StringIO(payloads[relative].decode())))
        need(count == derivative["rows"], "Audited scalar derivative row count differs")
    curves = sum(1 for _ in csv.DictReader(io.StringIO(payloads["report/curves.csv"].decode())))
    need(curves == report["table_rows"]["curves"], "Full saved curve point count differs")
    external = read_json(payloads["external_catalog.json"])
    result = (
        read_json(payloads["run/worker_result.json"])
        if "run/worker_result.json" in payloads
        else {}
    )
    full = audit.get("full_scientific_audit_passed") is True
    confirmed = bool(full and result.get("gates", {}).get("confirmed"))
    need(manifest["full_scientific_audit_passed"] == full, "Packet/audit completion differs")
    need(
        manifest["audit_passed"] == audit.get("passed", False)
        and manifest["scientific_contrast_completed"]
        == audit.get("scientific_contrast_completed", False),
        "Packet changes original scientific receipt flags",
    )
    if full:
        need(audit.get("passed") is True, "Full scientific audit does not declare passed")
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
    need(manifest["weighting_confirmed"] == confirmed, "Packet overstates confirmed weighting")
    status = manifest["scientific_status"]
    closure = manifest["overall_audit_closure_passed"]
    failure_notice = (
        """**The original audit coordinator failed with exit 1. Overall audit closure did not pass.** Its scientific receipt is preserved byte-for-byte; it is a separate recorded result. The original child exit status remains unknown, and no audit exit receipt was synthesized. See [the failure record](accounting/audit_launcher_failure.json), [original traceback](accounting/audit_launcher_traceback.txt), and [GNU time](accounting/audit_launcher_gnu_time.txt).

"""
        if coordinator_failed
        else ""
    )
    outcomes_notice = recorded_outcome_text(result)
    readme = f"""# C1.0: cost-effect-weighted fitting

**Overall audit closure passed: {closure}. Audit coordinator failed: {coordinator_failed}.**

{failure_notice}Saved status: `{status}`. The original scientific receipt declares passed: **{audit.get("passed", False)}**, full scientific audit passed: **{full}**. The declared weighting walking gain is **{"confirmed" if confirmed else "not established"}**.

{outcomes_notice}

This capsule preserves exact original compact evidence and **{curves} recorded learning points**. Missing children, partial prefixes and unmeasured endpoints remain identified by the saved status and packet manifest. A prerequisite or degenerate-contrast stop is not a failed-learning result. Walking, full survival, fitting error and complete native cost are reported separately.

Read [the report](report/report.md), [cost accounting](report/costs.json), [all learning points](report/curves.csv), [audit](audit/audit.json), and [external evidence catalog](external_catalog.json). The packet manifest binds every copied original file by exact bytes and SHA256.

Raw arrays, checkpoints, native ledgers/indexes and training prediction tapes remain outside Git at `{external["run_directory"]}`. Current availability, known checksums and any unavailable checksums are explicit in the external catalog; this publication did not rehash or decode those large files.

The audit receipt records its checked scope and any failure. This publication reran no native physics, gradients or optimizer training. Fixed-tape effects do not establish recovery, intrinsic importance, an adaptive MPPI–policy loop or CIO–GPS reproduction. No follow-up was launched.
"""
    need(
        total + len(readme.encode()) + 2 * MIB <= MAX_PAYLOAD + 8 * MIB,
        "Capsule metadata exceeds bounded output allowance",
    )
    out.mkdir(parents=True)
    for relative, payload in payloads.items():
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
    publisher_source = Path(__file__).read_bytes()
    publisher_relative = "publication/local_publisher.py"
    need(publisher_relative not in payloads, "Packet collides with local publisher provenance")
    (out / "publication").mkdir(exist_ok=True)
    with (out / publisher_relative).open("xb") as stream:
        stream.write(publisher_source)
    with (out / "README.md").open("x") as stream:
        stream.write(readme)
    files = {
        relative: dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        for relative, payload in payloads.items()
    }
    files["README.md"] = dict(
        bytes=len(readme.encode()), sha256=hashlib.sha256(readme.encode()).hexdigest()
    )
    files[publisher_relative] = dict(
        bytes=len(publisher_source), sha256=hashlib.sha256(publisher_source).hexdigest()
    )
    capsule = dict(
        schema="walker-c1-evidence-capsule-v1",
        run_id=manifest["run_id"],
        scientific_status=status,
        full_scientific_audit_passed=full,
        audit_passed=audit.get("passed", False),
        overall_audit_closure_passed=closure,
        audit_coordinator_failed=coordinator_failed,
        scientific_contrast_completed=audit.get("scientific_contrast_completed", False),
        weighting_confirmed=confirmed,
        uniform_parent_confirmed=bool(
            full and result.get("gates", {}).get("uniform_parent_confirmed")
        ),
        recovery_or_coupling_tested=False,
        source_archive=dict(bytes=archive.stat().st_size, sha256=expected_sha256),
        files=files,
        capsule_excludes_itself=True,
        preserved_learning_points=curves,
        raw_evidence_outside_Git=True,
        external_run_directory=external["run_directory"],
    )
    with (out / "capsule.json").open("x") as stream:
        json.dump(capsule, stream, indent=2, sort_keys=True)
        stream.write("\n")
    check()
    need(
        sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= MAX_PAYLOAD + 8 * MIB,
        "Published capsule output ceiling exceeded",
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    receipt = dict(
        schema="walker-c1-publication-receipt-v1",
        destination=str(out),
        archive_sha256=expected_sha256,
        archive_bytes=archive.stat().st_size,
        preserved_payload_bytes=total,
        published_file_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        preserved_learning_points=curves,
        overall_audit_closure_passed=closure,
        audit_coordinator_failed=coordinator_failed,
        wall_seconds=time.monotonic() - started,
        process_cpu_user_seconds=usage.ru_utime,
        process_cpu_system_seconds=usage.ru_stime,
        new_neural_rows=0,
        new_native_controls=0,
        new_optimizer_steps=0,
        git_operations=0,
        network_operations=0,
    )
    receipt.update(
        archive_hash_read_bytes=archive.stat().st_size,
        archive_decompression_input_bytes_upper=archive.stat().st_size,
        expanded_member_bytes_read=total,
        publisher_source_read_bytes=len(publisher_source),
        peak_rss_bytes=usage.ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024),
        limits=dict(
            wall_seconds=1800,
            peak_rss_bytes=2 * 1024**3,
            packet_bytes=MAX_PAYLOAD,
            published_capsule_bytes=MAX_PAYLOAD + 8 * MIB,
        ),
    )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(publish(args.archive, args.sha256, args.out)))


if __name__ == "__main__":
    main()
