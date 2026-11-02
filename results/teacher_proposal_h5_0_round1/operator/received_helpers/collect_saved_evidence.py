"""Collect a compact, closed H5 audit packet without inference or raw decoding.

Run this standalone standard-library helper only after the outer scientific
audit has exited successfully. Immutable raw streams and checkpoints remain
external. Their verification is attributed to that audit, never repeated here.
The ledger is streamed exactly once to hash it and extract scalar events.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import re
import resource
import signal
import sys
import tarfile
import time
from pathlib import Path

MIB = 1024**2
GIB = 1024**3
PACKET_LIMIT = 128 * MIB
OUTPUT_LIMIT = 512 * MIB
MEMORY_LIMIT = 2 * GIB
WALL_LIMIT = 1800
MAX_JSON = 16 * MIB
MAX_LOG = 16 * MIB
MAX_LEDGER = 80 * GIB
MAX_LINE = 16384
MAX_RECORDS = 5_000_000
MAX_EVENTS = 1000
ARMS = dict(I="independent", S="selector_only", P="proposal_guided")
SCALARS = {"policy_metric_complete", "teacher_agreement_complete", "episode_complete"}
HEX = re.compile(r"[0-9a-f]{64}\Z")
EPISODE = re.compile(
    r"(gap_teacher|teacher)_s([0-2])_(independent|selector_only|proposal_guided)_([0-9]{2})\Z"
)


class CollectionError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise CollectionError(message)


def identity(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def pairs(values):
    result = {}
    for key, value in values:
        need(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def decode(payload):
    def reject(value):
        raise CollectionError("Nonfinite JSON constant: " + value)

    def finite_float(value):
        result = float(value)
        need(math.isfinite(result), "Overflowing JSON float")
        return result

    return json.loads(
        payload, object_pairs_hook=pairs, parse_constant=reject, parse_float=finite_float
    )


def payload(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


class Collector:
    def __init__(self, run, audit, out, *, packet_limit=PACKET_LIMIT):
        self.run, self.audit, self.out = (Path(p).resolve() for p in (run, audit, out))
        need(
            type(packet_limit) is int and 0 < packet_limit <= PACKET_LIMIT,
            "Packet limit may only be lowered from128MiB",
        )
        need(self.run.is_dir() and self.audit.is_dir(), "Run and outer audit must exist")
        need(
            not self.out.is_relative_to(self.run) and not self.out.is_relative_to(self.audit),
            "Collection output must remain outside immutable run/audit evidence",
        )
        need(not self.out.exists(), "Collection output is create-only")
        self.packet_limit = packet_limit
        self.started = time.monotonic()
        self.out.mkdir(parents=True, exist_ok=False)
        self.packet = self.out / "packet"
        self.packet.mkdir()
        self.reads = collections.Counter()
        self.packet_bytes = self.output_bytes = 0
        self.bindings, self.inputs = {}, {}
        self.events, self.mechanisms, self.external = [], {}, {}
        self.rows = self.selections = 0

    def boundary(self):
        need(time.monotonic() - self.started <= WALL_LIMIT, "Collection wall-time cap exhausted")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = int(rss if sys.platform == "darwin" else rss * 1024)
        need(rss <= MEMORY_LIMIT, "Collection memory cap exhausted")

    def read(self, path, *, role="small_metadata", limit=MAX_JSON):
        self.boundary()
        path = Path(path)
        need(path.is_file() and not path.is_symlink(), "Missing or symlinked input: " + str(path))
        before = identity(path)
        need(before[2] <= limit, "Oversized compact input: " + str(path))
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        self.reads[role] += len(data)
        need(len(data) <= limit and identity(path) == before, "Input changed during compact read")
        self.inputs[str(path)] = dict(bytes=len(data), sha256=sha_bytes(data))
        return data

    def fund(self, amount, *, packet=False):
        self.boundary()
        need(self.output_bytes + amount <= OUTPUT_LIMIT, "Total collection output cap exhausted")
        if packet:
            need(
                self.packet_bytes + amount <= self.packet_limit,
                "Compact packet exceeds its128MiB-or-lower cap before write",
            )
            self.packet_bytes += amount
        self.output_bytes += amount

    def write(self, relative, data, *, original=None):
        destination = self.packet / relative
        need(destination.resolve().is_relative_to(self.packet), "Escaping compact packet path")
        self.fund(len(data), packet=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
        self.bindings[relative] = dict(bytes=len(data), sha256=sha_bytes(data))
        if original is not None:
            self.bindings[relative]["source"] = str(original)

    def copy(self, path, relative, *, limit=MAX_JSON):
        data = self.read(path, limit=limit)
        self.write(relative, data, original=path)
        return decode(data) if path.suffix == ".json" else None

    def safe(self, root, relative):
        need(isinstance(relative, str) and bool(relative), "Missing external relative filename")
        path = root / relative
        need(
            not Path(relative).is_absolute()
            and path.resolve().is_relative_to(root)
            and not path.is_symlink()
            and path.is_file(),
            "Missing or escaping external binding",
        )
        return path

    def external_ref(self, relative, digest, *, expected_bytes=None, verified_by):
        need(
            isinstance(digest, str) and HEX.fullmatch(digest), "Missing verified external checksum"
        )
        path = self.safe(self.run, relative)
        size = path.stat().st_size
        need(expected_bytes is None or expected_bytes == size, "External byte size differs")
        self.external[relative] = dict(
            bytes=size, sha256=digest, verified_by=verified_by, collector_rehashed=False
        )

    def validate_closed(self):
        need(not (self.audit / "result/failure.json").exists(), "Scientific audit failed")
        audit_receipt = self.copy(self.audit / "result/receipt.json", "audit/result/receipt.json")
        need(
            audit_receipt.get("schema") == "walker-h5.0-round-audit-v1"
            and audit_receipt.get("scientific_round_audit_complete") is True
            and audit_receipt.get("prediction_replay_complete") is True,
            "A complete successful scientific audit is required",
        )
        exit_receipt = self.copy(self.audit / "exit.json", "audit/exit.json")
        need(
            exit_receipt.get("exit_code") == 0 and exit_receipt.get("stopping_reason") is None,
            "Outer audit did not close successfully",
        )
        need(
            exit_receipt.get("unknown_partial_work_possible") is False,
            "Outer audit closure has unknown or missing completion evidence",
        )
        ownership = self.copy(self.audit / "ownership.json", "audit/ownership.json")
        self.copy(self.audit / "stdout.txt", "audit/stdout.txt", limit=MAX_LOG)
        config = self.copy(self.run / "config.json", "run/config.json")
        manifest = self.copy(self.run / "manifest.json", "run/manifest.json")
        seal = self.copy(self.run / "input_manifest.json", "run/input_manifest.json")
        worker = self.copy(self.run / "worker_result.json", "run/worker_result.json")
        need(
            worker.get("execution_complete") is True and not worker.get("error"),
            "Incomplete worker cannot produce a successful collection",
        )
        worker_exit = self.copy(self.run / "worker_exit.json", "run/worker_exit.json")
        need(
            worker_exit.get("exit_code") == 0 and worker_exit.get("stopping_reason") is None,
            "Worker outer watchdog did not close successfully",
        )
        need(
            worker_exit.get("unknown_partial_work_possible") is False,
            "Worker closure has unknown or missing completion evidence",
        )
        for filename, key in (
            ("config.json", "config_sha256"),
            ("manifest.json", "source_manifest_sha256"),
            ("input_manifest.json", "input_manifest_sha256"),
            ("worker_result.json", "worker_result_sha256"),
        ):
            need(
                ownership.get(key) == self.bindings["run/" + filename]["sha256"],
                "Audit ownership differs: " + filename,
            )
        need(
            manifest.get("config_sha256") == ownership["config_sha256"]
            and seal.get("config_sha256") == ownership["config_sha256"]
            and seal.get("source_manifest_sha256") == ownership["source_manifest_sha256"],
            "Frozen config/source/input seal differs",
        )
        source_rows = {row["file"]: row for row in manifest["files"]}
        auditor = "src/gps/mppi_gps/audits/teacher_proposal.py"
        need(
            source_rows.get(auditor, {}).get("sha256") == ownership.get("auditor_sha256"),
            "Closed audit source differs from the frozen source manifest",
        )
        preparation = audit_receipt.get("preparation", {}).get("bindings", {})
        for name in ("config.json", "manifest.json", "input_manifest.json"):
            binding = self.bindings["run/" + name]
            need(
                preparation.get(name) == {key: binding[key] for key in ("bytes", "sha256")},
                "Scientific audit preparation binding differs: " + name,
            )
        for name, row in source_rows.items():
            binding = preparation.get("source/" + name, {})
            need(binding.get("sha256") == row["sha256"], "Audited frozen source binding differs")
            self.external_ref(
                "source/" + name,
                row["sha256"],
                expected_bytes=binding.get("bytes"),
                verified_by="closed_scientific_audit_preparation",
            )
        hashes = audit_receipt.get("file_hashes", {})
        for name in ("arrays.bin", "arrays.index.jsonl", "ledger.jsonl"):
            self.external_ref(
                "evidence/" + name, hashes.get(name), verified_by="closed_scientific_audit"
            )
        for record in worker.get("checkpoint_records", []):
            self.external_ref(
                record["file"],
                record.get("sha256"),
                expected_bytes=record.get("bytes"),
                verified_by="closed_scientific_audit_checkpoint_ownership",
            )
        for name, binding in seal["files"].items():
            if name.startswith("inputs/"):
                need(preparation.get(name) == binding, "Audited benchmark binding differs")
                self.external_ref(
                    name,
                    binding.get("sha256"),
                    expected_bytes=binding.get("bytes"),
                    verified_by="closed_scientific_audit_preparation",
                )
        self.copy(self.run / "result.json", "run/result.json")
        for name in ("launch_receipt.json", "launch_ownership.json", "call_census.json"):
            self.copy(self.run / name, "run/" + name)
        for folder, pattern in (("metadata", "*.json"), ("learning", "*.json")):
            paths = sorted((self.run / folder).glob(pattern))
            need(len(paths) <= 128, "Unbounded compact metadata/learning file roster")
            for path in paths:
                self.safe(self.run, str(path.relative_to(self.run)))
                self.copy(path, "run/" + str(path.relative_to(self.run)), limit=8 * MIB)
        prep_receipt = self.audit / "result/preparation/receipt.json"
        if prep_receipt.is_file():
            self.copy(prep_receipt, "audit/result/preparation/receipt.json")
        stdout = self.run / "worker_stdout.txt"
        if stdout.is_file() and stdout.stat().st_size <= MAX_LOG:
            self.copy(stdout, "run/worker_stdout.txt", limit=MAX_LOG)
        elif stdout.exists():
            self.external["worker_stdout.txt"] = dict(
                bytes=stdout.stat().st_size,
                sha256=None,
                verified_by=None,
                collector_rehashed=False,
                reason="Exceeds16MiB compact-log cap",
            )
        return audit_receipt, worker, config

    def selection(self, row, last_model, expected_steps):
        context = row.get("context")
        need(isinstance(context, dict), "Selection lacks native query context")
        episode = context.get("episode")
        match = EPISODE.fullmatch(episode) if isinstance(episode, str) else None
        need(match is not None, "Unknown selection episode identity")
        arm = ARMS.get(row.get("arm"))
        need(
            arm is not None
            and context.get("arm") == arm
            and match[3] == arm
            and context.get("stream") == int(match[2])
            and context.get("phase") == "worker"
            and context.get("role") == "teacher",
            "Selection arm/stream context differs",
        )
        need(
            int(match[4]) < (2 if match[1] == "gap_teacher" else 4),
            "Unknown selection acquisition/gap index",
        )
        step = context.get("step")
        need(
            type(step) is int and step == expected_steps.get(episode, 0) and 0 <= step < 1000,
            "Duplicate, missing or nonsequential selection step",
        )
        need(
            last_model is not None
            and all(
                last_model.get(key) == context.get(key)
                for key in ("episode", "step", "stream", "arm", "phase", "role")
            ),
            "Selection differs from its completed native model-query context",
        )
        expected_steps[episode] = step + 1
        origin = row.get("selected_origin")
        need(isinstance(origin, str) and bool(origin), "Selection origin is missing")
        stream = self.mechanisms.setdefault(str(context["stream"]), {})
        counts = stream.setdefault(
            arm,
            dict(
                decisions=0,
                selected_origin_counts={},
                baseline_verification_status_counts={},
                guided_verification_status_counts={},
            ),
        )
        counts["decisions"] += 1
        for key, value in (
            ("selected_origin_counts", origin),
            ("baseline_verification_status_counts", row.get("baseline_verification_status")),
            ("guided_verification_status_counts", row.get("guided_verification_status")),
        ):
            if key != "selected_origin_counts":
                need(
                    value in ("ok", "guard_fallback", "native_error"),
                    "Unknown or missing selector verification status",
                )
            counts[key][value] = counts[key].get(value, 0) + 1
        self.selections += 1

    def ledger(self, receipt):
        path = self.run / "evidence/ledger.jsonl"
        before = identity(path)
        need(before[2] <= MAX_LEDGER, "Ledger exceeds its frozen category cap")
        digest, last_model, expected_steps = hashlib.sha256(), None, {}
        completed_episodes = {}
        with path.open("rb") as stream:
            while True:
                self.boundary()
                line = stream.readline(MAX_LINE + 1)
                self.reads["ledger_single_pass"] += len(line)
                if not line:
                    break
                need(
                    len(line) <= MAX_LINE and line.endswith(b"\n"),
                    "Oversized/truncated ledger line",
                )
                digest.update(line)
                row = decode(line)
                need(
                    row.get("id") == f"event-{self.rows:09d}",
                    "Duplicate, missing or out-of-order ledger event identity",
                )
                self.rows += 1
                need(
                    self.rows <= MAX_RECORDS
                    and row.get("kind", row.get("event")) == row.get("event"),
                    "Ledger roster or event kind differs",
                )
                event = row.get("event")
                if event == "query_complete" and row.get("mode") == "model":
                    last_model = row
                elif event == "two_stream_selection":
                    self.selection(row, last_model, expected_steps)
                elif event in SCALARS:
                    need(len(self.events) < MAX_EVENTS, "Scalar event packet roster exceeds1000")
                    self.events.append(row)
                    if event == "episode_complete" and EPISODE.fullmatch(row.get("tag", "")):
                        tag = row["tag"]
                        need(
                            tag not in completed_episodes
                            and expected_steps.get(tag, 0) == row.get("steps"),
                            "Teacher completion lacks exactly one selection per control",
                        )
                        completed_episodes[tag] = row["steps"]
        need(identity(path) == before, "Closed ledger changed during extraction")
        need(
            digest.hexdigest() == receipt["file_hashes"]["ledger.jsonl"],
            "Ledger hash differs from the closed scientific audit",
        )
        need(
            set(expected_steps) == set(completed_episodes),
            "Selection episode has no completed outcome",
        )
        recorded = receipt.get("structure", {}).get("ledger_events")
        need(type(recorded) is int and recorded == self.rows, "Audit ledger line census differs")
        need(
            receipt.get("scientific", {}).get("selection_checks") == self.selections,
            "Audited selection census differs",
        )
        self.external["evidence/ledger.jsonl"]["collector_rehashed"] = True
        self.external["evidence/ledger.jsonl"]["collector_hash_scope"] = (
            "single_scalar_extraction_pass"
        )

    def archive(self):
        target = self.out / "packet.tar.gz"
        owner = self

        class BoundedWriter:
            def __init__(self, stream):
                self.stream, self.bytes = stream, 0

            def write(self, data):
                need(self.bytes + len(data) <= PACKET_LIMIT, "Transfer archive exceeds128MiB")
                owner.fund(len(data))
                count = self.stream.write(data)
                self.bytes += count
                return count

            def flush(self):
                self.stream.flush()

        with target.open("xb") as raw:
            writer = BoundedWriter(raw)
            with gzip.GzipFile(fileobj=writer, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as archive:
                    for path in sorted(self.packet.rglob("*")):
                        if not path.is_file():
                            continue
                        self.boundary()
                        relative = str(path.relative_to(self.packet))
                        info = tarfile.TarInfo(relative)
                        info.size = path.stat().st_size
                        info.mode = 0o644
                        with path.open("rb") as source:
                            archive.addfile(info, source)
                        self.reads["packet_archive_source"] += info.size
        data = self.read(target, role="archive_output_hash", limit=PACKET_LIMIT)
        return dict(file=target.name, bytes=len(data), sha256=sha_bytes(data))

    def collect(self, *, operator_bootstrap=None):
        receipt, worker, config = self.validate_closed()
        self.ledger(receipt)
        if operator_bootstrap is not None:
            self.copy(Path(operator_bootstrap), "operator/launch_bootstrap.py", limit=MIB)
        script = Path(__file__).resolve()
        self.copy(script, "publication/collect_saved_evidence.py", limit=MIB)
        diagnostics = dict(
            schema="walker-h5-saved-diagnostics-v1",
            source_ledger=dict(
                file="evidence/ledger.jsonl",
                scope="external_run",
                bytes=self.external["evidence/ledger.jsonl"]["bytes"],
                sha256=receipt["file_hashes"]["ledger.jsonl"],
            ),
            audit_receipt=dict(
                file="audit/result/receipt.json",
                sha256=self.bindings["audit/result/receipt.json"]["sha256"],
            ),
            events=self.events,
            mechanism_summary=self.mechanisms,
        )
        self.write("diagnostics.json", payload(diagnostics))
        self.write(
            "external_catalog.json",
            payload(
                dict(
                    schema="walker-h5-external-evidence-catalog-v1",
                    files=self.external,
                    original_run_directory=str(self.run),
                    verification_scope="Raw/index/checkpoint/input hashes are reused from the closed scientific audit; only ledger is rehashed while extracting scalar events. No raw arrays or checkpoints decoded.",
                )
            ),
        )
        packet_manifest = dict(
            schema="walker-h5-compact-packet-v1",
            run_id=self.run.name,
            scientific_status=worker.get("status"),
            scientific_round_audit_complete=True,
            loop_closed=False,
            config_sha256=self.bindings["run/config.json"]["sha256"],
            audit_receipt_sha256=self.bindings["audit/result/receipt.json"]["sha256"],
            collector_sha256=self.bindings["publication/collect_saved_evidence.py"]["sha256"],
            files=self.bindings.copy(),
            manifest_excludes_itself=True,
            raw_arrays_checkpoints_npz_source_sqlite_and_journal_copied=False,
        )
        self.write("packet_manifest.json", payload(packet_manifest))
        archive = self.archive()
        self.boundary()
        result = dict(
            schema="walker-h5-compact-collection-receipt-v1",
            collection_complete=True,
            run_id=self.run.name,
            packet_bytes=self.packet_bytes,
            packet_file_count=len(self.bindings),
            archive=archive,
            audit_receipt=dict(
                file="packet/audit/result/receipt.json",
                sha256=self.bindings["audit/result/receipt.json"]["sha256"],
            ),
            diagnostics_file="packet/diagnostics.json",
            external_catalog_file="packet/external_catalog.json",
            scalar_event_count=len(self.events),
            selection_metadata_count=self.selections,
            ledger_lines=self.rows,
            mechanism_summary=self.mechanisms,
            source_read_bytes_by_role=dict(self.reads),
            source_read_bytes=sum(self.reads.values()),
            wall_seconds=time.monotonic() - self.started,
            limits=dict(
                packet_bytes=self.packet_limit,
                archive_bytes=PACKET_LIMIT,
                total_output_bytes=OUTPUT_LIMIT,
                ram_bytes=MEMORY_LIMIT,
                wall_seconds=WALL_LIMIT,
            ),
            scientific_work=dict(
                array_decodes=0,
                checkpoint_loads=0,
                network_forwards=0,
                physics_controls=0,
                optimizer_steps=0,
                mppi_queries=0,
            ),
            accounting_scope="Read counters include consumed metadata, the one ledger pass, copied packet bytes read during archiving and archive hashing. Library/import/kernel/cache reads, filesystem stat work, billing and energy are not metered. Receipt write is outside inner timer.",
        )
        final = payload(result)
        self.fund(len(final))
        with (self.out / "receipt.json").open("xb") as stream:
            stream.write(final)
        return result


def enforce_limits():
    """CLI process owns hard OS bounds; callers cannot increase them."""
    for name, maximum in (
        (resource.RLIMIT_AS, MEMORY_LIMIT),
        (resource.RLIMIT_FSIZE, OUTPUT_LIMIT),
    ):
        _, hard = resource.getrlimit(name)
        limit = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
        resource.setrlimit(name, (limit, limit))

    def timeout(signum, frame):
        raise CollectionError("Collection hard wall-time alarm expired")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, WALL_LIMIT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--operator-bootstrap", type=Path)
    parser.add_argument(
        "--packet-limit-bytes",
        type=int,
        default=PACKET_LIMIT,
        help="May only lower the128MiB cap for boundary verification",
    )
    args = parser.parse_args(argv)
    enforce_limits()
    collector = None
    try:
        collector = Collector(args.run, args.audit, args.out, packet_limit=args.packet_limit_bytes)
        result = collector.collect(operator_bootstrap=args.operator_bootstrap)
    except BaseException as error:
        if collector is not None:
            failure = dict(
                schema="walker-h5-compact-collection-failure-v1",
                collection_complete=False,
                error=dict(type=type(error).__name__, message=str(error)),
                source_read_bytes_by_role=dict(collector.reads),
                partial_packet_bytes=collector.packet_bytes,
                partial_output_bytes=collector.output_bytes,
                scalar_event_count=len(collector.events),
                ledger_lines=collector.rows,
                wall_seconds=time.monotonic() - collector.started,
                scientific_calls=0,
            )
            with (collector.out / "failure.json").open("xb") as stream:
                stream.write(payload(failure))
        print(json.dumps(dict(collection_complete=False, error=str(error))), file=sys.stderr)
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
