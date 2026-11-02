"""Synthetic-only checks; importing helpers never starts an audit or SSH."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parent


def imported(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


bootstrap, monitor = imported("audit_bootstrap"), imported("audit_monitor")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker_fixture(root):
    run = root / "run"
    run.mkdir()
    config = dict(
        launch_authorized=True,
        eligibility=dict(
            decision="approved_single_round_exception",
            useful_learning_prerequisite_satisfied=False,
            second_round_authorized=False,
        ),
        budgets=dict(
            audit_wall_seconds=7200,
            audit_peak_rss_bytes=4 * 1024**3,
            audit_output_storage_bytes=4 * 1024**3,
            audit_neural_rows=1999713552,
            audit_neural_queries=564270,
            audit_checkpoint_loads=15,
            audit_model_controls=0,
            audit_optimizer_steps=0,
        ),
        audit=dict(repeated_source_read_ceiling_bytes=3 * 1024**4),
    )
    save(run / "config.json", config)
    auditor = run / "source/src/gps/mppi_gps/audits/teacher_proposal.py"
    auditor.parent.mkdir(parents=True)
    auditor.write_text("# Synthetic source only\n")
    save(
        run / "manifest.json",
        dict(
            config_sha256=sha(run / "config.json"),
            files=[dict(file="src/gps/mppi_gps/audits/teacher_proposal.py", sha256=sha(auditor))],
        ),
    )
    save(
        run / "input_manifest.json",
        dict(
            config_sha256=sha(run / "config.json"),
            source_manifest_sha256=sha(run / "manifest.json"),
        ),
    )
    parents = [f"s{i}_parent" for i in range(3)]
    summaries = {
        str(i): {
            arm: dict(episodes=4, successes=4, complete=True)
            for arm in ("independent", "selector_only", "proposal_guided")
        }
        for i in range(3)
    }
    summaries["1"]["selector_only"]["successes"] = 3
    result = dict(
        execution_complete=True,
        status=bootstrap.EXPECTED_STATUS,
        error=None,
        comparison_complete=False,
        confirmed_benefit=False,
        next_round_eligible=False,
        evaluations={},
        paired_teacher_student_gap={},
        fit_records=[dict(tag=p) for p in parents],
        checkpoint_records=[dict(name=p) for p in parents + [p + "_initial" for p in parents]],
        teacher_summaries=summaries,
        work=dict(requested_optimizer_steps=6000, completed_optimizer_steps=6000, episodes=36),
    )
    save(run / "worker_result.json", result)
    save(
        run / "worker_exit.json",
        dict(
            exit_code=0,
            stopping_reason=None,
            unknown_partial_work_possible=False,
            leader_reaped=True,
            retries=0,
        ),
    )
    pinned = {
        name: sha(run / name) for name in ("config.json", "manifest.json", "input_manifest.json")
    }
    return run, pinned, sha(auditor)


def monitor_fixture(root):
    audit, operator, proc = root / "audit", root / "operator", root / "proc"
    (audit / "result").mkdir(parents=True)
    save(
        operator / "bootstrap_receipt.json",
        dict(
            coordinator_pid=123,
            coordinator_birth_ticks=999,
            boot_id="synthetic-boot",
            created_unix_seconds=time.time() - 1,
            observed_worker_status=bootstrap.EXPECTED_STATUS,
            observed_worker_result_sha256="a" * 64,
        ),
    )
    boot = proc / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("synthetic-boot\n")
    stat = proc / "123/stat"
    stat.parent.mkdir()
    stat.write_text(
        "123 (synthetic process) S " + " ".join(["0"] * 18 + ["999"] + ["0"] * 4) + "\n"
    )
    save(
        audit / "ownership.json",
        dict(
            coordinator_pid=123,
            watchdog_wall_seconds=7200,
            peak_rss_bytes=4 * 1024**3,
            worker_result_sha256="a" * 64,
        ),
    )
    (audit / "result/work.jsonl").write_text(
        json.dumps(dict(event="prediction_complete", id=1, rows=48, completed_rows_after=48)) + "\n"
    )
    (audit / "result/array-index.sqlite").write_bytes(b"never open this SQLite file")
    (audit / "stdout.txt").write_text("")
    (operator / "coordinator_stdout.txt").write_text("")
    return audit, operator, proc


class OperatorChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="walker-h5-audit-operator-synthetic-")
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_normal_negative_metadata_gate_has_no_process_or_scientific_call(self):
        run, pinned, auditor = worker_fixture(self.root)
        with mock.patch.object(
            bootstrap.subprocess, "Popen", side_effect=AssertionError("no launch")
        ):
            config, result = bootstrap.validate_closed(
                run,
                sha(run / "worker_result.json"),
                bootstrap.Metadata(),
                pinned=pinned,
                auditor_sha=auditor,
            )
        self.assertTrue(config["launch_authorized"])
        self.assertEqual(result["status"], bootstrap.EXPECTED_STATUS)

    def test_worker_hash_incomplete_child_and_unknown_closure_fail(self):
        for case in ("hash", "incomplete", "child", "unknown", "missing"):
            with self.subTest(case=case):
                folder = self.root / case
                folder.mkdir()
                run, pinned, auditor = worker_fixture(folder)
                result = json.loads((run / "worker_result.json").read_text())
                if case == "incomplete":
                    result["execution_complete"] = False
                if case == "child":
                    result["fit_records"].append(dict(tag="s0_proposal_guided_child"))
                save(run / "worker_result.json", result)
                if case == "unknown":
                    save(
                        run / "worker_exit.json",
                        dict(
                            exit_code=0,
                            stopping_reason=None,
                            unknown_partial_work_possible=True,
                            leader_reaped=True,
                            retries=0,
                        ),
                    )
                if case == "missing":
                    (run / "worker_exit.json").unlink()
                expected = "0" * 64 if case == "hash" else sha(run / "worker_result.json")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    bootstrap.validate_closed(
                        run, expected, bootstrap.Metadata(), pinned=pinned, auditor_sha=auditor
                    )

    def test_monitor_requires_closed_exit_and_success_receipt(self):
        audit, operator, proc = monitor_fixture(self.root)
        forbidden = audit / "result/array-index.sqlite"
        original = Path.open

        def guarded(path, *args, **kwargs):
            if path == forbidden:
                raise AssertionError("Monitor opened SQLite")
            return original(path, *args, **kwargs)

        reader = monitor.Reader()
        with mock.patch.object(Path, "open", guarded):
            running = monitor.snapshot(reader, audit=audit, operator=operator, proc=proc)
        self.assertTrue(running["same_coordinator_alive"])
        self.assertFalse(running["scientific_audit_complete"])
        save(
            audit / "result/receipt.json",
            dict(scientific_round_audit_complete=True, prediction_replay_complete=True),
        )
        before_exit = monitor.snapshot(reader, audit=audit, operator=operator, proc=proc)
        self.assertFalse(before_exit["scientific_audit_complete"])
        save(
            audit / "exit.json",
            dict(exit_code=0, stopping_reason=None, unknown_partial_work_possible=False),
        )
        complete = monitor.snapshot(reader, audit=audit, operator=operator, proc=proc)
        self.assertTrue(complete["scientific_audit_complete"])
        self.assertLess(reader.bytes, 64 * 1024**2)

    def test_pid_reuse_and_missing_outer_exit_never_authorize_retry(self):
        audit, operator, proc = monitor_fixture(self.root)
        path = proc / "123/stat"
        path.write_text(path.read_text().replace("999", "888"))
        result = monitor.snapshot(monitor.Reader(), audit=audit, operator=operator, proc=proc)
        self.assertFalse(result["same_coordinator_alive"])
        self.assertFalse(result["scientific_audit_complete"])
        self.assertIn("preserve_prefix_no_retry", result["completion_status"])

    def test_monitor_byte_cap_stops_before_reading_missing_as_zero(self):
        audit, operator, proc = monitor_fixture(self.root)
        reader = monitor.Reader(1)
        with self.assertRaisesRegex(ValueError, "reservation exhausted"):
            monitor.snapshot(reader, audit=audit, operator=operator, proc=proc)
        self.assertEqual(reader.bytes, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
