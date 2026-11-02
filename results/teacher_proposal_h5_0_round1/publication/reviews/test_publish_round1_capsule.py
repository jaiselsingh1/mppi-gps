"""Synthetic metadata-only publication checks; never open actual study evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("publish_round1_capsule.py")
SPEC = importlib.util.spec_from_file_location("h5_round1_publication", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def binding(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def fixture(root, *, audit_complete=True):
    packet = root / "packet"
    run = packet / "run"
    run.mkdir(parents=True)
    remote = "/synthetic/remote/run"
    config = dict(
        schema="walker-teacher-proposal-h5.0-r1",
        launch_authorized=True,
        eligibility=dict(
            decision="approved_single_round_exception",
            useful_learning_prerequisite_satisfied=False,
            second_round_authorized=False,
        ),
    )
    save(run / "config.json", config)
    config_sha = binding(run / "config.json")["sha256"]
    auditor = "src/gps/mppi_gps/audits/teacher_proposal.py"
    save(
        run / "manifest.json",
        dict(config_sha256=config_sha, files=[dict(file=auditor, sha256="d" * 64)]),
    )
    source_sha = binding(run / "manifest.json")["sha256"]
    save(
        run / "input_manifest.json",
        dict(config_sha256=config_sha, source_manifest_sha256=source_sha),
    )
    save(
        run / "worker_result.json",
        dict(
            execution_complete=True,
            status=module.NEGATIVE,
            comparison_complete=False,
            confirmed_benefit=False,
            next_round_eligible=False,
            evaluations={},
            paired_teacher_student_gap={},
            fit_records=[{}] * 3,
            checkpoint_records=[{}] * 6,
        ),
    )
    for name in ("result.json", "call_census.json", "launch_receipt.json", "launch_ownership.json"):
        save(run / name, dict(synthetic=name))
    closed = dict(exit_code=0, stopping_reason=None, unknown_partial_work_possible=False)
    save(run / "worker_exit.json", closed)
    save(
        packet / "audit/result/receipt.json",
        dict(
            schema="walker-h5.0-round-audit-v1",
            scientific_round_audit_complete=audit_complete,
            prediction_replay_complete=audit_complete,
            file_hashes={"ledger.jsonl": "a" * 64},
        ),
    )
    save(packet / "audit/exit.json", closed)
    ownership = {
        key: binding(run / name)["sha256"]
        for name, key in (
            ("config.json", "config_sha256"),
            ("manifest.json", "source_manifest_sha256"),
            ("input_manifest.json", "input_manifest_sha256"),
            ("worker_result.json", "worker_result_sha256"),
        )
    }
    ownership["auditor_sha256"] = "d" * 64
    save(packet / "audit/ownership.json", ownership)
    audit_sha = binding(packet / "audit/result/receipt.json")["sha256"]
    diagnostics = dict(
        source_ledger=dict(file="evidence/ledger.jsonl", sha256="a" * 64, bytes=7),
        audit_receipt=dict(file="audit/result/receipt.json", sha256=audit_sha),
        events=[],
    )
    save(packet / "diagnostics.json", diagnostics)
    save(
        packet / "external_catalog.json",
        dict(
            schema="walker-h5-external-evidence-catalog-v1",
            original_run_directory=remote,
            files={"evidence/ledger.jsonl": dict(bytes=7, sha256="a" * 64)},
        ),
    )
    files = {
        str(path.relative_to(packet)): binding(path) for path in packet.rglob("*") if path.is_file()
    }
    save(
        packet / "packet_manifest.json",
        dict(
            schema="walker-h5-compact-packet-v1",
            run_id="synthetic",
            scientific_status=module.NEGATIVE,
            scientific_round_audit_complete=True,
            manifest_excludes_itself=True,
            raw_arrays_checkpoints_npz_source_sqlite_and_journal_copied=False,
            audit_receipt_sha256=audit_sha,
            config_sha256=config_sha,
            files=files,
        ),
    )
    packet_files = [path for path in packet.rglob("*") if path.is_file()]
    collection = root / "collection_receipt.json"
    save(
        collection,
        dict(
            schema="walker-h5-compact-collection-receipt-v1",
            collection_complete=True,
            run_id="synthetic",
            packet_file_count=len(packet_files),
            packet_bytes=sum(path.stat().st_size for path in packet_files),
            audit_receipt=dict(sha256=audit_sha),
            scientific_work=dict(network_forwards=0, physics_controls=0, optimizer_steps=0),
        ),
    )
    report = root / "report"
    report.mkdir()
    (report / "report.md").write_text("# Synthetic saved report\n")
    report_source = root / "report_source.py"
    report_source.write_text("# Synthetic renderer source; never executed.\n")
    report_inputs = {
        name: binding(run / name)
        for name in ("config.json", "worker_result.json", "call_census.json")
    }
    report_inputs["saved_diagnostics"] = binding(packet / "diagnostics.json")
    save(
        report / "receipt.json",
        dict(
            network_forwards=0,
            physics_controls=0,
            optimizer_steps=0,
            raw_arrays_decoded=0,
            checkpoints_loaded=0,
            outputs={"report.md": binding(report / "report.md")},
            inputs=report_inputs,
            config_sha256=config_sha,
            source_result_sha256=binding(run / "worker_result.json")["sha256"],
            source_sha256=binding(report_source)["sha256"],
            source_ledger_binding=diagnostics["source_ledger"],
            audit_receipt_binding=diagnostics["audit_receipt"],
        ),
    )
    readme = root / "README.md"
    readme.write_text("# Synthetic capsule\n")
    accounting = root / "accounting.json"
    save(accounting, dict(synthetic=True))
    source_review = root / "source_review.json"
    save(
        source_review,
        dict(
            schema="walker-h5-authorized-source-independent-review-v1",
            passed=True,
            implementation_commit=module.FROZEN_COMMIT,
            source_and_git_blob_match=True,
            source_files_verified=1,
            scientific_settings_seeds_census_unchanged=True,
        ),
    )
    return argparse.Namespace(
        out=root / "capsule",
        receipt=root / "publication_receipt.json",
        packet=packet,
        collection_receipt=collection,
        report=report,
        report_source=report_source,
        readme=readme,
        accounting=accounting,
        implementation_commit=module.FROZEN_COMMIT,
        source_review=source_review,
        remote_run=remote,
        local_initial_export=root / "initial_export",
        include=[],
    )


class PublicationChecks(unittest.TestCase):
    def test_exact_copy_preserves_raw_availability_and_unmeasured_learning(self):
        with tempfile.TemporaryDirectory(prefix="h5-publication-fixture-") as directory:
            args = fixture(Path(directory))
            receipt = module.Publisher(args).publish()
            capsule = json.loads((args.out / "capsule.json").read_text())
            for name, expected in capsule["files"].items():
                self.assertEqual(
                    binding(args.out / name), {key: expected[key] for key in ("bytes", "sha256")}
                )
            self.assertFalse(capsule["learning_contrast_measured"])
            self.assertFalse(capsule["standalone_panels_measured"])
            self.assertFalse(capsule["bulk_evidence"]["local_complete_run_available"])
            self.assertTrue(capsule["bulk_evidence"]["remote_primary_evidence_verified"])
            self.assertFalse(capsule["bulk_evidence"]["remote_copy_verified"])
            self.assertLess(receipt["total_bytes"], module.LIMIT)

    def test_failed_audit_stops_before_capsule_creation(self):
        with tempfile.TemporaryDirectory(prefix="h5-publication-fixture-") as directory:
            args = fixture(Path(directory), audit_complete=False)
            with self.assertRaisesRegex(ValueError, "audit did not pass"):
                module.Publisher(args).publish()
            self.assertFalse(args.out.exists())

    def test_tampered_packet_stops_before_capsule_creation(self):
        with tempfile.TemporaryDirectory(prefix="h5-publication-fixture-") as directory:
            args = fixture(Path(directory))
            (args.packet / "run/config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest binding"):
                module.Publisher(args).publish()
            self.assertFalse(args.out.exists())

    def test_existing_output_is_never_reused(self):
        with tempfile.TemporaryDirectory(prefix="h5-publication-fixture-") as directory:
            args = fixture(Path(directory))
            args.out.mkdir()
            (args.out / "sentinel").write_text("preserved")
            with self.assertRaisesRegex(ValueError, "create-only"):
                module.Publisher(args)
            self.assertEqual((args.out / "sentinel").read_text(), "preserved")

    def test_raw_array_include_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory(prefix="h5-publication-fixture-") as directory:
            args = fixture(Path(directory))
            source = Path(directory) / "opaque.bin"
            source.write_bytes(b"synthetic opaque bytes")
            args.include = [[str(source), "publication/raw.bin"]]
            with self.assertRaisesRegex(ValueError, "Forbidden raw"):
                module.Publisher(args).publish()
            self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
