"""Bound one saved H5 report after successful collection and scientific audit.

This postprocessor never launches an experiment or audit. It imports the pinned
uploaded report under the frozen package only for provenance helpers. Report
outputs are create-only and live outside RUN and the immutable compact packet.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib
import importlib.util
import json
import os
import resource
import secrets
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/home/jaisel/mppi-gps-workspace")
RUN = ROOT / "runs/2026-11-01_teacher_proposal_h5_0_r1_001"
ART = ROOT / "artifacts/2026-11-01_teacher_proposal_h5_0_r1_001"
OUT = ART / "report_render_001"
REPORT_SOURCE = ART / "report_teacher_proposal_edc0.py"
REPORT_SHA = "de1ad2c0048ed64f5627a955c737dd79f94f46c6b0b929148f8be36643e66318"
FROZEN_MANIFEST_SHA = "c45de63fe2d7f133fa9cbc2f806e3d038792a56d01cecb5bdb10721664d10a80"
MODULE = "src.gps.mppi_gps.reporting.teacher_proposal_saved_edc0"
WALL = 1800
MEMORY = 2 * 1024**3
OUTPUT = 512 * 1024**2
STDOUT = 64 * 1024**2
FILE_BYTES = 16 * 1024**2
WRITE_PATHS = 27


def need(condition, message):
    if not condition:
        raise ValueError(message)


class Reads:
    def __init__(self):
        self.bytes = 0
        self.bindings = {}

    def read(self, path, limit=16 * 1024**2):
        path = Path(path)
        need(
            path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            "Missing, symlinked or oversized report input: " + str(path),
        )
        before = path.stat()
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        self.bytes += len(data)
        after = path.stat()
        need(
            len(data) <= limit
            and (before.st_size, before.st_mtime_ns, before.st_ino)
            == (after.st_size, after.st_mtime_ns, after.st_ino),
            "Report input changed",
        )
        self.bindings[str(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        return data

    def load(self, path):
        return json.loads(self.read(path))

    def hash(self, path, limit=128 * 1024**2):
        path = Path(path)
        need(
            path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            "Unbounded report hash input",
        )
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                self.bytes += len(block)
                digest.update(block)
        after = path.stat()
        need(
            (before.st_size, before.st_mtime_ns, before.st_ino)
            == (after.st_size, after.st_mtime_ns, after.st_ino),
            "Hash input changed",
        )
        self.bindings[str(path)] = dict(bytes=after.st_size, sha256=digest.hexdigest())
        return digest.hexdigest()


def write(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    need(len(data) < 1024**2, "Operator control record exceeds1MiB")
    with path.open("xb") as stream:
        stream.write(data)


def validate_packet(collection, reader, *, report_source=REPORT_SOURCE, report_sha=REPORT_SHA):
    collection = Path(collection).resolve()
    closed = reader.load(collection / "receipt.json")
    need(closed.get("collection_complete") is True, "Collection is not complete")
    need(
        closed.get("scientific_work") and not any(closed["scientific_work"].values()),
        "Collection claims unexpected scientific execution",
    )
    packet = collection / "packet"
    manifest = reader.load(packet / "packet_manifest.json")
    need(
        manifest.get("scientific_round_audit_complete") is True
        and manifest.get("manifest_excludes_itself") is True,
        "Packet lacks completed audit binding",
    )
    actual = {str(path.relative_to(packet)) for path in packet.rglob("*") if path.is_file()}
    need(
        actual == set(manifest["files"]) | {"packet_manifest.json"},
        "Packet has missing/extra files",
    )
    total = 0
    for name, binding in manifest["files"].items():
        path = packet / name
        need(
            not Path(name).is_absolute()
            and path.resolve().is_relative_to(packet)
            and not path.is_symlink(),
            "Packet path escapes its root",
        )
        need(
            not name.startswith(("run/source/", "run/evidence/", "run/checkpoints/"))
            and path.suffix not in (".npz", ".pt", ".sqlite", ".bin", ".jsonl"),
            "Compact packet unexpectedly contains bulk evidence",
        )
        need(
            path.stat().st_size == binding["bytes"] and reader.hash(path) == binding["sha256"],
            "Packet binding differs: " + name,
        )
        total += path.stat().st_size
    total += (packet / "packet_manifest.json").stat().st_size
    need(
        total == closed["packet_bytes"] and total <= 128 * 1024**2,
        "Packet size reservation differs",
    )
    audit = reader.load(packet / "audit/result/receipt.json")
    exit_record = reader.load(packet / "audit/exit.json")
    need(
        audit.get("scientific_round_audit_complete") is True
        and audit.get("prediction_replay_complete") is True,
        "Successful scientific audit is required",
    )
    need(
        exit_record.get("exit_code") == 0
        and exit_record.get("stopping_reason") is None
        and exit_record.get("unknown_partial_work_possible") is False,
        "Audit closure is incomplete",
    )
    need(
        manifest["audit_receipt_sha256"]
        == reader.bindings[str(packet / "audit/result/receipt.json")]["sha256"],
        "Packet audit receipt binding differs",
    )
    owner = reader.load(packet / "audit/ownership.json")
    for name, key in (
        ("config.json", "config_sha256"),
        ("manifest.json", "source_manifest_sha256"),
        ("input_manifest.json", "input_manifest_sha256"),
        ("worker_result.json", "worker_result_sha256"),
    ):
        need(
            owner.get(key) == manifest["files"]["run/" + name]["sha256"],
            "Audited report input ownership differs: " + name,
        )
    worker = reader.load(packet / "run/worker_result.json")
    need(
        worker.get("execution_complete") is True and not worker.get("error"),
        "Incomplete worker cannot become an audited report",
    )
    diagnostics = reader.load(packet / "diagnostics.json")
    need(
        diagnostics.get("audit_receipt", {}).get("sha256") == manifest["audit_receipt_sha256"]
        and diagnostics.get("source_ledger", {}).get("sha256")
        == audit["file_hashes"]["ledger.jsonl"],
        "Diagnostic extract differs from its closed audit",
    )
    need(reader.hash(report_source, 1024**2) == report_sha, "Uploaded report source changed")
    return packet, manifest


def frozen_helpers(reader):
    need(
        reader.hash(RUN / "manifest.json", 1024**2) == FROZEN_MANIFEST_SHA,
        "Frozen source owner changed",
    )
    manifest = reader.load(RUN / "manifest.json")
    rows = {row["file"]: row["sha256"] for row in manifest["files"]}
    for name in ("src/gps/mppi_gps/proposal_launch.py", "src/gps/mppi_gps/provenance.py"):
        need(
            reader.hash(RUN / "source" / name, 1024**2) == rows[name],
            "Frozen report helper changed",
        )
    sys.path.insert(0, str(RUN / "source"))
    return importlib.import_module("src.gps.mppi_gps.proposal_launch").bounded_process


def output_guard(out):
    """Finite report-file roster; no generic filesystem permissions framework.

    27 distinct writable paths times16MiB +64MiB stdout +4MiB controls is500MiB.
    Deleted temporary/font paths do not refund slots. This preserves all saved
    points; an unexpectedly larger renderer stops rather than truncating data.
    """
    out = Path(out).resolve()
    paths = set()
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))

    def hook(event, args):
        if event != "open":
            return
        file, mode, flags = args
        writable = (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or (
            isinstance(flags, int)
            and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if not writable or isinstance(file, int):
            return
        path = Path(os.fsdecode(file)).resolve()
        need(path.is_relative_to(out), "Report attempted a write outside its output directory")
        if path not in paths:
            need(len(paths) < WRITE_PATHS, "Report output file roster exceeds its reserved bound")
            paths.add(path)

    sys.addaudithook(hook)
    return paths


def load_report():
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        need(
            name.split(".", 1)[0] not in ("torch", "mujoco", "mujoco_warp", "warp"),
            "Saved reporting attempted to import an inference/physics stack",
        )
        return original(name, *args, **kwargs)

    builtins.__import__ = guarded
    sys.path.insert(0, str(RUN / "source"))
    importlib.import_module("src.gps.mppi_gps.reporting")
    spec = importlib.util.spec_from_file_location(MODULE, REPORT_SOURCE)
    need(spec is not None and spec.loader is not None, "Cannot bind pinned report source")
    report = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = report
    spec.loader.exec_module(report)
    return report


def render_child(collection, ownership):
    started = time.perf_counter()
    reader = Reads()
    marker = reader.load(ownership)
    need(
        os.getppid() == marker["coordinator_pid"]
        and os.environ.get("H5_REPORT_NONCE") == marker["nonce"],
        "Report child lacks its live outer owner",
    )
    need(
        reader.hash(Path(__file__).resolve(), 1024**2) == marker["helper_sha256"],
        "Report wrapper changed",
    )
    packet = Path(collection).resolve() / "packet"
    need(
        reader.hash(Path(collection) / "receipt.json") == marker["collection_receipt_sha256"]
        and reader.hash(packet / "packet_manifest.json") == marker["packet_manifest_sha256"]
        and reader.hash(REPORT_SOURCE, 1024**2) == REPORT_SHA,
        "Report input/source ownership changed",
    )
    out = Path(ownership).resolve().parent
    need(not (out / "result").exists(), "Report result output is create-only")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib")
    os.environ["TMPDIR"] = str(out / "temp")
    (out / "matplotlib").mkdir(exist_ok=False)
    (out / "temp").mkdir(exist_ok=False)
    tempfile.tempdir = None
    paths = output_guard(out)
    report = load_report()
    receipt = report.render(packet / "run", out / "result", diagnostics=packet / "diagnostics.json")
    manifest = reader.load(packet / "packet_manifest.json")
    for binding in receipt["inputs"].values():
        path = Path(binding["file"]).resolve()
        need(path.is_relative_to(packet), "Report consumed an input outside its verified packet")
        expected = manifest["files"].get(str(path.relative_to(packet)))
        need(
            expected and all(binding[key] == expected[key] for key in ("bytes", "sha256")),
            "Consumed report input differs from collection binding",
        )
    need(
        receipt["source_sha256"] == REPORT_SHA
        and all(
            receipt[key] == 0
            for key in (
                "raw_arrays_decoded",
                "checkpoints_loaded",
                "network_forwards",
                "physics_controls",
                "optimizer_steps",
            )
        ),
        "Saved report claims an unexpected source or scientific call",
    )
    total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    need(total <= OUTPUT, "Report directory exceeds512MiB")
    result = dict(
        schema="walker-h5-bounded-report-child-v1",
        render_complete=True,
        source_sha256=REPORT_SHA,
        ownership_read_bytes=reader.bytes,
        report_receipt=receipt,
        writable_paths_reserved=len(paths),
        report_directory_bytes_before_child_receipt=total,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        wall_seconds=time.perf_counter() - started,
        scientific_calls=0,
        scope="Derived saved JSON/curves only. Renderer counts its consumed payloads separately; do not add overlapping wall timers. Library/font/cache reads are not fully instrumented.",
    )
    write(out / "child_receipt.json", result)
    print(json.dumps(dict(render_complete=True, table_rows=receipt["table_rows"])))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ownership", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.render_only:
        need(args.ownership is not None, "Report ownership marker required")
        return render_child(args.collection, args.ownership)
    need(sys.platform == "linux", "Bounded report coordinator requires Linux RSS accounting")
    started = time.monotonic()
    reader = Reads()
    need(not OUT.exists(), "Report coordinator output is create-only; no retry")
    collection = args.collection.resolve()
    need(
        not OUT.is_relative_to(collection) and not OUT.is_relative_to(RUN),
        "Report output overlaps immutable evidence",
    )
    packet, manifest = validate_packet(collection, reader)
    for key, value in dict(
        PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    ).items():
        os.environ[key] = value
    bounded_process = frozen_helpers(reader)
    OUT.mkdir(parents=True, exist_ok=False)
    nonce = secrets.token_hex(32)
    marker = dict(
        coordinator_pid=os.getpid(),
        nonce=nonce,
        helper_sha256=reader.hash(Path(__file__).resolve(), 1024**2),
        collection_receipt_sha256=reader.bindings[str(collection / "receipt.json")]["sha256"],
        packet_manifest_sha256=reader.bindings[str(packet / "packet_manifest.json")]["sha256"],
        audit_receipt_sha256=manifest["audit_receipt_sha256"],
        report_source_sha256=REPORT_SHA,
    )
    write(OUT / "ownership.json", marker)
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--render-only",
        "--collection",
        str(collection),
        "--ownership",
        str(OUT / "ownership.json"),
    ]
    environment = dict(
        os.environ,
        H5_REPORT_NONCE=nonce,
        MPLBACKEND="Agg",
        CUDA_VISIBLE_DEVICES="",
        MUJOCO_GL="disable",
    )
    environment.pop("PYTHONPATH", None)
    try:
        exit_record = bounded_process(
            command,
            cwd=ART,
            environment=environment,
            output=OUT / "stdout.txt",
            wall_seconds=max(0.001, WALL - (time.monotonic() - started)),
            stdout_bytes=STDOUT,
            peak_rss_bytes=MEMORY,
        )
    except BaseException as error:
        write(
            OUT / "exit.json",
            getattr(error, "process_receipt", dict(error=str(error), process_started=False)),
        )
        raise
    write(OUT / "exit.json", exit_record)
    need(
        exit_record["exit_code"] == 0 and exit_record["stopping_reason"] is None,
        "Saved report did not finish within its envelope",
    )
    child = reader.load(OUT / "child_receipt.json")
    need(child.get("render_complete") is True, "Saved report child did not complete")
    for name, binding in child["report_receipt"]["outputs"].items():
        path = OUT / "result" / name
        need(
            path.resolve().is_relative_to(OUT / "result")
            and path.stat().st_size == binding["bytes"]
            and reader.hash(path) == binding["sha256"],
            "Rendered output binding differs",
        )
    total = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    need(
        total < OUTPUT - 1024**2 and time.monotonic() - started <= WALL,
        "Final report resource envelope exhausted",
    )
    result = dict(
        schema="walker-h5-bounded-report-v1",
        render_complete=True,
        source_sha256=REPORT_SHA,
        audit_receipt_sha256=manifest["audit_receipt_sha256"],
        operator_read_bytes=reader.bytes,
        operator_bindings=reader.bindings,
        outer_exit=exit_record,
        child_receipt=child,
        bytes_before_operator_receipt=total,
        wall_seconds=time.monotonic() - started,
        limits=dict(
            wall_seconds=WALL,
            peak_rss_bytes=MEMORY,
            stdout_bytes=STDOUT,
            total_output_bytes=OUTPUT,
            child_write_paths=WRITE_PATHS,
            per_file_bytes=FILE_BYTES,
            reserved_operator_control_bytes=4 * 1024**2,
        ),
        scientific_calls=0,
        no_retry=True,
        accounting_scope="Operator reads include its packet verification and final output verification. Child ownership reads and renderer payload/hash reads are separately reported passes; library/font/cache IO is not fully metered. Outer and child/report wall timers overlap and must not be added.",
    )
    write(OUT / "receipt.json", result)
    print(json.dumps(dict(render_complete=True, report=str(OUT / "result/report.md"), bytes=total)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
