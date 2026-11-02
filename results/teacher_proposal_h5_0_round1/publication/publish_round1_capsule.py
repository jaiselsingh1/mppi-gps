"""Publish exact compact H5.0r1 evidence only after a successful closed audit.

This standard-library helper never opens raw arrays, checkpoints, NPZ inputs,
SQLite indexes or work journals. It never runs a worker, inference or physics.
All input and output directories are explicit. Existing evidence is immutable.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

LIMIT = 128 * 1024**2
CHUNK = 1024**2
FROZEN_COMMIT = "ff67495dae33ee43952677d0c0165a43f8d469b0"
NEGATIVE = "teacher_competence_negative_no_child_fits"
ALLOWED_SUFFIXES = {".json", ".md", ".txt", ".csv", ".py", ".png", ".pdf"}
FORBIDDEN_PARTS = {"evidence", "checkpoints", "inputs", "source", "__pycache__"}
REQUIRED_RUN = {
    "config.json",
    "manifest.json",
    "input_manifest.json",
    "result.json",
    "worker_result.json",
    "call_census.json",
    "launch_receipt.json",
    "launch_ownership.json",
    "worker_exit.json",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def pairs(rows):
    value = {}
    for key, item in rows:
        need(key not in value, "Duplicate JSON key: " + key)
        value[key] = item
    return value


def relative(name):
    need(isinstance(name, str) and bool(name), "Missing publication-relative filename")
    value = Path(name)
    need(
        not value.is_absolute()
        and ".." not in value.parts
        and str(value) == name
        and value.suffix in ALLOWED_SUFFIXES
        and not (set(value.parts) & FORBIDDEN_PARTS),
        "Forbidden raw, escaping or unsupported publication file: " + name,
    )
    return value


class Publisher:
    def __init__(self, args):
        self.args = args
        self.started = time.perf_counter()
        self.reads = collections.Counter()
        self.plan = {}
        self.bindings = {}
        self.bytes_written = 0
        self.out = args.out.resolve()
        need(not self.out.exists(), "Capsule output must be create-only")
        need(not args.receipt.exists(), "Publication receipt must be create-only")
        need(
            args.implementation_commit == FROZEN_COMMIT,
            "Expected the full frozen ff67495 implementation commit",
        )

    def identity(self, path):
        need(path.is_file() and not path.is_symlink(), "Missing or symlinked file: " + str(path))
        row = path.stat()
        need(row.st_size <= LIMIT, "Oversized compact publication input: " + str(path))
        return row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns

    def hash(self, path, role):
        before = self.identity(path)
        value = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(CHUNK):
                self.reads[role] += len(block)
                value.update(block)
        need(self.identity(path) == before, "Input changed while hashing: " + str(path))
        return dict(bytes=before[2], sha256=value.hexdigest())

    def load(self, path):
        before = self.identity(path)
        data = path.read_bytes()
        self.reads["json_preflight"] += len(data)
        need(len(data) == before[2] and self.identity(path) == before, "JSON input changed")

        def reject(value):
            raise ValueError("Nonfinite JSON constant: " + value)

        return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)

    def add(self, source, name, expected=None, original_source=None):
        name = str(relative(name))
        need(name != "capsule.json", "The capsule manifest is reserved for the publisher")
        need(name not in self.plan, "Duplicate publication destination: " + name)
        source = Path(source)
        binding = self.hash(source, "payload_preflight_hash")
        if expected is not None:
            need(
                binding == {key: expected[key] for key in ("bytes", "sha256")},
                "Input differs from its saved manifest binding: " + name,
            )
        need(
            sum(row[1]["bytes"] for row in self.plan.values()) + binding["bytes"] <= LIMIT,
            "Publication payload cannot fit 128 MiB before copying",
        )
        self.plan[name] = source, binding, original_source

    def bound_file(self, root, name):
        path = root / relative(name)
        need(path.resolve().is_relative_to(root), "Input path escapes its packet/report root")
        self.identity(path)
        return path

    def verify_packet(self):
        packet = self.args.packet.resolve()
        need(packet.is_dir(), "Compact collector packet is missing")
        manifest_path = packet / "packet_manifest.json"
        manifest = self.load(manifest_path)
        need(
            manifest.get("schema") == "walker-h5-compact-packet-v1"
            and manifest.get("scientific_round_audit_complete") is True
            and manifest.get("scientific_status") == NEGATIVE
            and manifest.get("manifest_excludes_itself") is True
            and manifest.get("raw_arrays_checkpoints_npz_source_sqlite_and_journal_copied")
            is False,
            "A closed, compact normal-teacher-negative packet is required",
        )
        files = manifest.get("files")
        need(isinstance(files, dict), "Packet manifest lacks its exact file roster")
        actual = {str(path.relative_to(packet)) for path in packet.rglob("*") if path.is_file()}
        need(actual == set(files) | {"packet_manifest.json"}, "Unbound or missing packet file")
        need({"run/" + name for name in REQUIRED_RUN} <= set(files), "Packet run metadata missing")
        for name, binding in sorted(files.items()):
            self.add(self.bound_file(packet, name), name, binding, binding.get("source"))
        self.add(manifest_path, "packet_manifest.json")
        audit = self.load(packet / "audit/result/receipt.json")
        audit_exit = self.load(packet / "audit/exit.json")
        ownership = self.load(packet / "audit/ownership.json")
        need(
            audit.get("schema") == "walker-h5.0-round-audit-v1"
            and audit.get("scientific_round_audit_complete") is True
            and audit.get("prediction_replay_complete") is True,
            "Scientific prediction audit did not pass completely",
        )
        need(
            audit_exit.get("exit_code") == 0
            and audit_exit.get("stopping_reason") is None
            and audit_exit.get("unknown_partial_work_possible") is False,
            "Outer audit closure is incomplete or failed",
        )
        need(
            manifest["audit_receipt_sha256"] == files["audit/result/receipt.json"]["sha256"],
            "Packet audit binding differs",
        )
        config = self.load(packet / "run/config.json")
        source = self.load(packet / "run/manifest.json")
        seal = self.load(packet / "run/input_manifest.json")
        worker = self.load(packet / "run/worker_result.json")
        worker_exit = self.load(packet / "run/worker_exit.json")
        need(
            config.get("schema") == "walker-teacher-proposal-h5.0-r1"
            and config.get("launch_authorized") is True
            and config.get("eligibility", {}).get("decision") == "approved_single_round_exception"
            and config["eligibility"].get("useful_learning_prerequisite_satisfied") is False
            and config["eligibility"].get("second_round_authorized") is False,
            "Frozen first-round authorization/prerequisite ownership differs",
        )
        need(
            worker.get("execution_complete") is True
            and worker.get("status") == NEGATIVE
            and worker.get("comparison_complete") is False
            and worker.get("confirmed_benefit") is False
            and worker.get("next_round_eligible") is False
            and not worker.get("error")
            and not worker.get("evaluations")
            and not worker.get("paired_teacher_student_gap")
            and len(worker.get("fit_records", [])) == 3
            and len(worker.get("checkpoint_records", [])) == 6,
            "Teacher stop was incomplete or an unmeasured learning contrast was promoted",
        )
        need(
            worker_exit.get("exit_code") == 0
            and worker_exit.get("stopping_reason") is None
            and worker_exit.get("unknown_partial_work_possible") is False,
            "Outer worker closure is incomplete or failed",
        )
        for name, key in (
            ("config.json", "config_sha256"),
            ("manifest.json", "source_manifest_sha256"),
            ("input_manifest.json", "input_manifest_sha256"),
            ("worker_result.json", "worker_result_sha256"),
        ):
            need(
                ownership.get(key) == files["run/" + name]["sha256"], "Audit owner binding differs"
            )
        config_hash = files["run/config.json"]["sha256"]
        need(
            manifest["config_sha256"]
            == source["config_sha256"]
            == seal["config_sha256"]
            == config_hash
            and seal["source_manifest_sha256"] == files["run/manifest.json"]["sha256"],
            "Original frozen configuration/source/input chain differs",
        )
        auditor = "src/gps/mppi_gps/audits/teacher_proposal.py"
        source_rows = {row["file"]: row["sha256"] for row in source["files"]}
        need(source_rows.get(auditor) == ownership.get("auditor_sha256"), "Frozen auditor differs")
        source_review = self.load(self.args.source_review)
        need(
            source_review.get("schema") == "walker-h5-authorized-source-independent-review-v1"
            and source_review.get("passed") is True
            and source_review.get("implementation_commit") == FROZEN_COMMIT
            and source_review.get("source_and_git_blob_match") is True
            and source_review.get("source_files_verified") == len(source_rows)
            and source_review.get("scientific_settings_seeds_census_unchanged") is True,
            "The explicit prelaunch frozen-source/Git review is missing or differs",
        )
        self.add(self.args.source_review, "publication/prelaunch_independent_review.json")
        diagnostics = self.load(packet / "diagnostics.json")
        external = self.load(packet / "external_catalog.json")
        ledger = external["files"]["evidence/ledger.jsonl"]
        need(
            diagnostics["source_ledger"]["sha256"]
            == audit["file_hashes"]["ledger.jsonl"]
            == ledger["sha256"]
            and diagnostics["source_ledger"]["bytes"] == ledger["bytes"]
            and diagnostics["audit_receipt"]["sha256"] == manifest["audit_receipt_sha256"],
            "Diagnostics lost their original ledger/audit binding",
        )
        need(
            external.get("schema") == "walker-h5-external-evidence-catalog-v1"
            and external.get("original_run_directory") == self.args.remote_run,
            "External run ownership differs",
        )
        collection = self.load(self.args.collection_receipt)
        need(
            collection.get("schema") == "walker-h5-compact-collection-receipt-v1"
            and collection.get("collection_complete") is True
            and collection.get("run_id") == manifest["run_id"]
            and collection.get("packet_file_count") == len(actual)
            and collection.get("packet_bytes")
            == sum(self.plan[name][1]["bytes"] for name in actual)
            and collection.get("audit_receipt", {}).get("sha256")
            == manifest["audit_receipt_sha256"]
            and collection.get("scientific_work")
            and all(value == 0 for value in collection["scientific_work"].values()),
            "Compact collection did not close with the exact audited packet",
        )
        self.add(self.args.collection_receipt, "publication/collection_receipt.json")
        return manifest, diagnostics, worker, external

    def verify_report(self, diagnostics):
        report = self.args.report.resolve()
        receipt = self.load(report / "receipt.json")
        need(
            all(
                receipt.get(key) == 0
                for key in (
                    "network_forwards",
                    "physics_controls",
                    "optimizer_steps",
                    "raw_arrays_decoded",
                    "checkpoints_loaded",
                )
            ),
            "Report performed or failed to account for scientific work",
        )
        outputs = receipt.get("outputs", {})
        actual = {str(path.relative_to(report)) for path in report.rglob("*") if path.is_file()}
        need(actual == set(outputs) | {"receipt.json"}, "Unbound or missing report file")
        for name, binding in sorted(outputs.items()):
            self.add(self.bound_file(report, name), "report/" + name, binding)
        self.add(report / "receipt.json", "report/receipt.json")
        for label, binding in receipt["inputs"].items():
            name = "diagnostics.json" if label == "saved_diagnostics" else "run/" + label
            need(
                name in self.plan
                and binding["sha256"] == self.plan[name][1]["sha256"]
                and binding["bytes"] == self.plan[name][1]["bytes"],
                "Report consumed another input",
            )
        need(
            receipt["config_sha256"] == self.plan["run/config.json"][1]["sha256"]
            and receipt["source_result_sha256"] == self.plan["run/worker_result.json"][1]["sha256"]
            and receipt["source_ledger_binding"] == diagnostics["source_ledger"]
            and receipt["audit_receipt_binding"] == diagnostics["audit_receipt"],
            "Report receipt lost its worker/diagnostic ownership",
        )
        self.add(self.args.report_source, "publication/reporting_teacher_proposal.py")
        need(
            receipt["source_sha256"]
            == self.plan["publication/reporting_teacher_proposal.py"][1]["sha256"],
            "Saved report implementation differs from its receipt",
        )

    def copy(self, name, row):
        source, binding, original = row
        destination = self.out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = self.identity(source)
        digest = hashlib.sha256()
        copied = 0
        with source.open("rb") as incoming, destination.open("xb") as target:
            while block := incoming.read(CHUNK):
                self.reads["exact_payload_copy"] += len(block)
                need(
                    self.bytes_written + len(block) <= LIMIT,
                    "Capsule 128 MiB write ceiling exhausted",
                )
                target.write(block)
                self.bytes_written += len(block)
                copied += len(block)
                digest.update(block)
        need(
            self.identity(source) == before
            and copied == binding["bytes"]
            and digest.hexdigest() == binding["sha256"],
            "Payload changed while copying; preserve the partial publication",
        )
        self.bindings[name] = dict(binding, source=str(source))
        if original is not None:
            self.bindings[name]["original_source"] = original

    def publish(self):
        packet, diagnostics, worker, external = self.verify_packet()
        self.verify_report(diagnostics)
        self.load(self.args.accounting)
        self.add(self.args.accounting, "accounting/accounting.json")
        self.add(self.args.readme, "README.md")
        self.add(Path(__file__).resolve(), "publication/publish_round1_capsule.py")
        for source, name in self.args.include:
            self.add(Path(source), name)
        need(not self.out.is_relative_to(self.args.packet.resolve()), "Output is inside packet")
        need(not self.out.is_relative_to(self.args.report.resolve()), "Output is inside report")
        need(
            not self.out.is_relative_to(self.args.local_initial_export.resolve()),
            "Output is inside the immutable local initial export",
        )
        need(
            not self.args.receipt.resolve().is_relative_to(self.out),
            "Final receipt must be external",
        )
        need(
            all(
                not self.args.receipt.resolve().is_relative_to(root.resolve())
                for root in (self.args.packet, self.args.report, self.args.local_initial_export)
            ),
            "Final receipt cannot be written inside immutable input evidence",
        )
        self.out.mkdir(parents=True, exist_ok=False)
        for name, row in sorted(self.plan.items()):
            self.copy(name, row)
        capsule = dict(
            schema="walker-result-capsule-v1",
            study="teacher_proposal_h5_0_round1",
            run_id=packet["run_id"],
            status=worker["status"],
            integrity_status="independent_round_audit_passed",
            implementation_commit=self.args.implementation_commit,
            scientific_round_audit_complete=True,
            prediction_replay_complete=True,
            execution_complete=True,
            comparison_complete=False,
            learning_contrast_measured=False,
            standalone_panels_measured=False,
            loop_closed=False,
            confirmed_benefit=False,
            files=self.bindings,
            copied_file_count=len(self.bindings),
            copied_bytes=self.bytes_written,
            bulk_evidence=dict(
                ssh_host="robopil5090",
                remote_directory=self.args.remote_run,
                remote_copy_verified=False,
                remote_primary_evidence_verified=True,
                remote_raw_evidence_verified_by="closed_independent_scientific_audit",
                local_initial_export=str(self.args.local_initial_export.resolve()),
                local_initial_export_is_complete_run_mirror=False,
                local_complete_run_available=False,
                external_catalog="external_catalog.json",
                omitted=sorted(external["files"]),
                scope="Raw arrays, ledgers, checkpoints, inputs, full frozen source and audit work files remain external on robopil5090. The local initial export predates execution; it is not a complete raw mirror. Publication reuses closed-audit hashes and rehashes compact copies only.",
            ),
            scope="Exact compact evidence for a completed normal teacher-competence stop. Three parents were fitted; child learning and standalone student outcomes were not measured. Different successful motions remain acceptable; no loop-closure claim.",
            publication_receipt_external=str(self.args.receipt.resolve()),
        )
        data = encode(capsule)
        need(self.bytes_written + len(data) <= LIMIT, "Capsule manifest cannot fit 128 MiB")
        with (self.out / "capsule.json").open("xb") as stream:
            stream.write(data)
        self.bytes_written += len(data)
        receipt = dict(
            schema="walker-h5-round1-publication-receipt-v1",
            publication_complete=True,
            capsule_directory=str(self.out),
            capsule_sha256=hashlib.sha256(data).hexdigest(),
            payload_files=len(self.bindings),
            payload_bytes=capsule["copied_bytes"],
            total_files=len(self.bindings) + 1,
            total_bytes=self.bytes_written,
            maximum_bytes=LIMIT,
            input_read_bytes_by_role=dict(self.reads),
            input_read_bytes=sum(self.reads.values()),
            wall_seconds=time.perf_counter() - self.started,
            no_scientific_work=True,
            accounting_scope="Counts include explicit JSON reads, compact-file verification and exact-copy reads. Library/import/kernel/cache/stat work, billing and energy are unmetered; final receipt write is outside the inner timer. Source collector/audit/scientific costs are preserved separately, not added to publication IO.",
        )
        self.args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with self.args.receipt.open("xb") as stream:
            stream.write(encode(receipt))
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "packet",
        "collection-receipt",
        "report",
        "report-source",
        "source-review",
        "readme",
        "accounting",
        "out",
        "receipt",
        "local-initial-export",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--remote-run", required=True)
    parser.add_argument(
        "--include", nargs=2, action="append", default=[], metavar=("SOURCE", "DEST")
    )
    args = parser.parse_args(argv)
    print(json.dumps(Publisher(args).publish(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
