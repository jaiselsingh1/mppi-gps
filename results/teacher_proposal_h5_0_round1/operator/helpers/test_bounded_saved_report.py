"""Synthetic packet plus genuine saved-only Matplotlib rendering; no research."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ART = Path(__file__).parent.resolve()
PROJECT = ART.parents[1]
SCRIPT = ART / "render_saved_report_bounded.py"
SPEC = importlib.util.spec_from_file_location("bounded_h5_report", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def packet_fixture(root):
    collection = root / "collection"
    packet = collection / "packet"
    save(
        packet / "run/config.json",
        dict(
            launch_authorized=True, eligibility=dict(useful_learning_prerequisite_satisfied=False)
        ),
    )
    save(packet / "run/manifest.json", dict(synthetic=True))
    save(packet / "run/input_manifest.json", dict(synthetic=True))
    save(
        packet / "run/worker_result.json",
        dict(
            schema="walker-teacher-proposal-round-v1",
            execution_complete=True,
            status="teacher_competence_negative_no_child_fits",
            work={},
            teacher_summaries={},
            evaluations={},
            checkpoint_records=[],
        ),
    )
    save(
        packet / "run/call_census.json",
        dict(
            total=dict(model_controls=1, live_controls=1, physics_substeps=4, worker_neural_rows=1),
            strong_audit_neural_rows=1,
            worker_optimizer_steps=2,
        ),
    )
    save(
        packet / "run/learning/s0_parent.json",
        dict(
            completed_updates=2,
            failed=False,
            learning_curve=[
                dict(
                    update=i,
                    fit_update=i,
                    action_mse=1 / i,
                    proximal_loss=0.0,
                    total_loss=1 / i,
                    unclipped_gradient_norm=2 / i,
                )
                for i in (1, 2)
            ],
        ),
    )
    save(
        packet / "audit/result/receipt.json",
        dict(
            scientific_round_audit_complete=True,
            prediction_replay_complete=True,
            file_hashes={"ledger.jsonl": "a" * 64},
        ),
    )
    save(
        packet / "audit/exit.json",
        dict(exit_code=0, stopping_reason=None, unknown_partial_work_possible=False),
    )
    save(
        packet / "audit/ownership.json",
        {
            key: sha(packet / "run" / name)
            for name, key in (
                ("config.json", "config_sha256"),
                ("manifest.json", "source_manifest_sha256"),
                ("input_manifest.json", "input_manifest_sha256"),
                ("worker_result.json", "worker_result_sha256"),
            )
        },
    )
    audit_sha = sha(packet / "audit/result/receipt.json")
    save(
        packet / "diagnostics.json",
        dict(
            schema="walker-h5-saved-diagnostics-v1",
            source_ledger=dict(file="evidence/ledger.jsonl", sha256="a" * 64, bytes=0),
            audit_receipt=dict(file="audit/result/receipt.json", sha256=audit_sha),
            events=[],
        ),
    )
    files = {
        str(path.relative_to(packet)): binding(path) for path in packet.rglob("*") if path.is_file()
    }
    save(
        packet / "packet_manifest.json",
        dict(
            manifest_excludes_itself=True,
            scientific_round_audit_complete=True,
            audit_receipt_sha256=audit_sha,
            files=files,
        ),
    )
    save(
        collection / "receipt.json",
        dict(
            collection_complete=True,
            scientific_work=dict(array_decodes=0, network_forwards=0, physics_controls=0),
            packet_bytes=sum(path.stat().st_size for path in packet.rglob("*") if path.is_file()),
        ),
    )
    return collection, packet


class BoundedReportChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="walker-h5-bounded-report-synthetic-")
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_packet_hash_and_audit_rejection_before_render(self):
        collection, packet = packet_fixture(self.root)
        source = PROJECT / "src/gps/mppi_gps/reporting/teacher_proposal.py"
        module.validate_packet(collection, module.Reads(), report_source=source)
        (packet / "run/config.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "binding differs"):
            module.validate_packet(collection, module.Reads(), report_source=source)
        self.assertFalse((self.root / "result").exists())

    def test_pinned_source_rejects_another_report(self):
        collection, _ = packet_fixture(self.root)
        source = self.root / "other.py"
        source.write_text("raise AssertionError('never execute')\n")
        with self.assertRaisesRegex(ValueError, "report source changed"):
            module.validate_packet(collection, module.Reads(), report_source=source)

    def test_failed_audit_is_not_promoted_by_a_consistent_packet_hash(self):
        collection, packet = packet_fixture(self.root)
        receipt = json.loads((packet / "audit/result/receipt.json").read_text())
        receipt["scientific_round_audit_complete"] = False
        save(packet / "audit/result/receipt.json", receipt)
        manifest = json.loads((packet / "packet_manifest.json").read_text())
        manifest["files"]["audit/result/receipt.json"] = binding(
            packet / "audit/result/receipt.json"
        )
        save(packet / "packet_manifest.json", manifest)
        closed = json.loads((collection / "receipt.json").read_text())
        closed["packet_bytes"] = sum(
            path.stat().st_size for path in packet.rglob("*") if path.is_file()
        )
        save(collection / "receipt.json", closed)
        source = PROJECT / "src/gps/mppi_gps/reporting/teacher_proposal.py"
        with self.assertRaisesRegex(ValueError, "Successful scientific audit"):
            module.validate_packet(collection, module.Reads(), report_source=source)

    def test_direct_child_without_live_parent_owner_stops_before_render(self):
        collection, _ = packet_fixture(self.root)
        owner = self.root / "owner.json"
        save(owner, dict(coordinator_pid=-1, nonce="not-a-live-owner"))
        with self.assertRaisesRegex(ValueError, "live outer owner"):
            module.render_child(collection, owner)

    def test_real_saved_matplotlib_plot_under_guard_and_frozen_provenance(self):
        collection, packet = packet_fixture(self.root)
        out = self.root / "render"
        out.mkdir()
        source = PROJECT / "src/gps/mppi_gps/reporting/teacher_proposal.py"
        self.assertEqual(sha(source), module.REPORT_SHA)
        frozen = PROJECT / "runs/2026-11-01_teacher_proposal_h5_0_r1_001"
        nonce = "synthetic-test-nonce"
        marker = dict(
            coordinator_pid=os.getpid(),
            nonce=nonce,
            helper_sha256=sha(SCRIPT),
            collection_receipt_sha256=sha(collection / "receipt.json"),
            packet_manifest_sha256=sha(packet / "packet_manifest.json"),
        )
        save(out / "ownership.json", marker)
        code = (
            "import importlib.util;from pathlib import Path;"
            f"s=importlib.util.spec_from_file_location('report_envelope',{str(SCRIPT)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            f"m.RUN=Path({str(frozen)!r});m.REPORT_SOURCE=Path({str(source)!r});"
            f"raise SystemExit(m.render_child(Path({str(collection)!r}),Path({str(out / 'ownership.json')!r})))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True,
            text=True,
            env=dict(
                os.environ,
                H5_REPORT_NONCE=nonce,
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            ),
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads((out / "child_receipt.json").read_text())
        self.assertTrue(receipt["render_complete"])
        self.assertEqual(receipt["report_receipt"]["table_rows"]["learning_curves.csv"], 2)
        self.assertLessEqual(receipt["writable_paths_reserved"], module.WRITE_PATHS)
        self.assertGreater((out / "result/learning_curves.png").stat().st_size, 1000)
        self.assertTrue((out / "result/learning_curves.pdf").read_bytes().startswith(b"%PDF"))
        self.assertEqual(receipt["scientific_calls"], 0)
        self.assertTrue(
            all(
                receipt["report_receipt"][key] == 0
                for key in (
                    "raw_arrays_decoded",
                    "checkpoints_loaded",
                    "network_forwards",
                    "physics_controls",
                    "optimizer_steps",
                )
            )
        )
        self.assertEqual(receipt["report_receipt"]["source_sha256"], module.REPORT_SHA)
        print(
            json.dumps(
                dict(
                    synthetic_real_plot=True,
                    writable_paths_reserved=receipt["writable_paths_reserved"],
                    peak_rss_bytes=receipt["peak_rss_bytes"],
                    report_directory_bytes=receipt["report_directory_bytes_before_child_receipt"],
                    learning_curve_points=receipt["report_receipt"]["table_rows"][
                        "learning_curves.csv"
                    ],
                    scientific_calls=receipt["scientific_calls"],
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
