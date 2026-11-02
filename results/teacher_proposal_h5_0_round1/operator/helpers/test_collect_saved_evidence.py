"""Synthetic-only stdlib checks; never read a project run or invoke research."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("collect_saved_evidence.py")
SPEC = importlib.util.spec_from_file_location("h5_compact_collector", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return dict(bytes=path.stat().st_size, sha256=digest(path))


def fixture(root, *, mutate=None, worker_complete=True, audit_complete=True):
    run, audit = root / "run", root / "audit"
    run.mkdir()
    (audit / "result").mkdir(parents=True)
    save(run / "config.json", dict(schema="synthetic", launch_authorized=True))
    source = run / "source/src/gps/mppi_gps/audits/teacher_proposal.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic frozen source, never executed.\n")
    manifest = dict(
        config_sha256=digest(run / "config.json"),
        files=[dict(file=str(source.relative_to(run / "source")), sha256=digest(source))],
    )
    save(run / "manifest.json", manifest)
    benchmark = run / "inputs/train.npz"
    benchmark.parent.mkdir()
    benchmark.write_bytes(b"opaque synthetic input; not an NPZ")
    save(
        run / "input_manifest.json",
        dict(
            config_sha256=digest(run / "config.json"),
            source_manifest_sha256=digest(run / "manifest.json"),
            files={"inputs/train.npz": binding(benchmark)},
        ),
    )
    checkpoint = run / "checkpoints/s0_parent.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"opaque synthetic checkpoint; never loaded")
    save(
        run / "worker_result.json",
        dict(
            execution_complete=worker_complete,
            status="teacher_competence_negative_no_child_fits",
            error=None,
            checkpoint_records=[dict(file="checkpoints/s0_parent.pt", **binding(checkpoint))],
        ),
    )
    for name in ("result.json", "launch_receipt.json", "launch_ownership.json", "call_census.json"):
        save(run / name, dict(synthetic=name))
    save(
        run / "worker_exit.json",
        dict(exit_code=0, stopping_reason=None, unknown_partial_work_possible=False),
    )
    save(run / "metadata/synthetic.json", dict(saved=True))
    save(run / "learning/s0_parent.json", dict(completed_updates=0, learning_curve=[]))
    save(
        run / "learning/s1_parent_partial.json",
        dict(completed_updates=0, learning_curve=[], failed=True),
    )
    (run / "worker_stdout.txt").write_text("synthetic stdout\n")
    evidence = run / "evidence"
    evidence.mkdir()
    (evidence / "arrays.bin").write_bytes(b"opaque canonical raw bytes")
    (evidence / "arrays.index.jsonl").write_bytes(b"opaque canonical index bytes\n")
    context = dict(
        phase="worker",
        role="teacher",
        stream=0,
        arm="independent",
        episode="teacher_s0_independent_00",
        step=0,
    )
    raw = dict(file="arrays.bin", group="arrays/a000000001_teacher")
    selection = dict(
        event="two_stream_selection",
        arm="I",
        context=context,
        selected_origin="bank:0:protected:raw_sample",
        baseline_verification_status="ok",
        guided_verification_status="guard_fallback",
    )
    rows = [
        dict(event="query_complete", mode="model", query_id="q000000001", **context),
        selection,
        dict(event="two_stream_selection_evidence", request_id="event-000000001", evidence={}),
        dict(
            event="policy_metric_complete",
            tag="s0_parent/train",
            actor_hash="a" * 64,
            observations=dict(file="inputs/train.npz", key="observations"),
            targets=dict(file="inputs/train.npz", key="actions"),
            rows=1,
            batch_size=1024,
            action_mse=0.5,
        ),
        dict(
            event="episode_complete",
            tag=context["episode"],
            seed=1,
            steps=1,
            success=False,
            survived=False,
            complete=True,
            cost=5002.0,
            mae=1.0,
            terminal_cost=20.0,
            terminal_adjustment=0.0,
            absorbing_cost=4995.0,
            raw=raw,
        ),
        dict(
            event="teacher_agreement_complete",
            actor_hash="a" * 64,
            episode_raw=raw,
            rows=1,
            action_agreement=0.4,
        ),
    ]
    if mutate is not None:
        mutate(rows)
    for index, row in enumerate(rows):
        row.update(id=f"event-{index:09d}", kind=row["event"])
    ledger = evidence / "ledger.jsonl"
    ledger.write_bytes(b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows))
    prep_bindings = {
        name: binding(run / name)
        for name in ("config.json", "manifest.json", "input_manifest.json", "inputs/train.npz")
    }
    prep_bindings["source/" + str(source.relative_to(run / "source"))] = binding(source)
    receipt = dict(
        schema="walker-h5.0-round-audit-v1",
        scientific_round_audit_complete=audit_complete,
        prediction_replay_complete=audit_complete,
        preparation=dict(bindings=prep_bindings),
        file_hashes={
            name: digest(evidence / name)
            for name in ("arrays.bin", "arrays.index.jsonl", "ledger.jsonl")
        },
        structure=dict(ledger_events=len(rows)),
        scientific=dict(
            selection_checks=sum(row["event"] == "two_stream_selection" for row in rows)
        ),
    )
    save(audit / "result/receipt.json", receipt)
    save(
        audit / "exit.json",
        dict(exit_code=0, stopping_reason=None, unknown_partial_work_possible=False),
    )
    save(
        audit / "ownership.json",
        dict(
            config_sha256=digest(run / "config.json"),
            source_manifest_sha256=digest(run / "manifest.json"),
            input_manifest_sha256=digest(run / "input_manifest.json"),
            worker_result_sha256=digest(run / "worker_result.json"),
            auditor_sha256=digest(source),
        ),
    )
    (audit / "stdout.txt").write_text("synthetic audit complete\n")
    return run, audit, rows


class CollectorChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="walker-h5-collector-synthetic-")
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_hashes_once_excludes_bulk_and_preserves_scalar_ids(self):
        run, audit, source_rows = fixture(self.root)
        before = {
            str(p): digest(p) for folder in (run, audit) for p in folder.rglob("*") if p.is_file()
        }
        protected = {
            run / "evidence/arrays.bin",
            run / "evidence/arrays.index.jsonl",
            run / "checkpoints/s0_parent.pt",
            run / "inputs/train.npz",
        }
        original = Path.open
        opens = {run / "evidence/ledger.jsonl": 0}

        def guard(path, *args, **kwargs):
            if path in protected:
                raise AssertionError("Collector opened excluded bulk evidence")
            if path in opens:
                opens[path] += 1
            return original(path, *args, **kwargs)

        collector = module.Collector(run, audit, self.root / "out")
        with mock.patch.object(Path, "open", guard):
            result = collector.collect()
        self.assertEqual(opens[run / "evidence/ledger.jsonl"], 1)
        self.assertEqual(result["selection_metadata_count"], 1)
        self.assertEqual(result["scalar_event_count"], 3)
        self.assertEqual(
            result["source_read_bytes_by_role"]["ledger_single_pass"],
            (run / "evidence/ledger.jsonl").stat().st_size,
        )
        diagnostics = json.loads((collector.packet / "diagnostics.json").read_text())
        self.assertEqual(
            [r["id"] for r in diagnostics["events"]],
            [r["id"] for r in source_rows if r["event"] in module.SCALARS],
        )
        mechanism = diagnostics["mechanism_summary"]["0"]["independent"]
        self.assertEqual(mechanism["decisions"], 1)
        self.assertEqual(mechanism["guided_verification_status_counts"], dict(guard_fallback=1))
        self.assertTrue((collector.packet / "run/learning/s1_parent_partial.json").is_file())
        external = json.loads((collector.packet / "external_catalog.json").read_text())["files"]
        self.assertFalse(external["evidence/arrays.bin"]["collector_rehashed"])
        self.assertTrue(external["evidence/ledger.jsonl"]["collector_rehashed"])
        with tarfile.open(collector.out / "packet.tar.gz") as archive:
            names = archive.getnames()
        self.assertTrue(all(not name.endswith((".npz", ".pt", ".sqlite")) for name in names))
        self.assertTrue(all(not name.startswith(("source/", "evidence/")) for name in names))
        self.assertFalse(any(name.endswith("work.jsonl") for name in names))
        self.assertEqual(
            before,
            {
                str(p): digest(p)
                for folder in (run, audit)
                for p in folder.rglob("*")
                if p.is_file()
            },
        )
        with self.assertRaisesRegex(module.CollectionError, "create-only"):
            module.Collector(run, audit, collector.out)

    def test_duplicate_selection_cannot_inflate_mechanism_counts(self):
        def change(rows):
            rows.insert(2, deepcopy(rows[1]))

        run, audit, _ = fixture(self.root, mutate=change)
        collector = module.Collector(run, audit, self.root / "out")
        with self.assertRaisesRegex(module.CollectionError, "Duplicate, missing or nonsequential"):
            collector.collect()
        self.assertFalse((collector.out / "packet.tar.gz").exists())

    def test_normal_teacher_stop_keeps_six_checkpoints_and_all36_variable_prefixes(self):
        run, audit, _ = fixture(self.root)
        rows = []
        for stream in range(3):
            for point in ("parent_initial", "parent"):
                for split in ("train", "validation"):
                    rows.append(
                        dict(
                            event="policy_metric_complete",
                            tag=f"s{stream}_{point}/{split}",
                            actor_hash="a" * 64,
                            observations={},
                            targets={},
                            rows=1,
                            action_mse=0.1,
                        )
                    )
        controls = 0
        for stream in range(3):
            for arm_code, arm in module.ARMS.items():
                for index in range(4):
                    tag = f"teacher_s{stream}_{arm}_{index:02d}"
                    steps = (
                        201
                        if (stream, arm, index) == (1, "selector_only", 0)
                        else (199 if (stream, arm, index) == (2, "proposal_guided", 0) else 1)
                    )
                    controls += steps
                    for step in range(steps):
                        context = dict(
                            phase="worker",
                            role="teacher",
                            stream=stream,
                            arm=arm,
                            episode=tag,
                            step=step,
                        )
                        rows.append(dict(event="query_complete", mode="model", **context))
                        rows.append(
                            dict(
                                event="two_stream_selection",
                                arm=arm_code,
                                context=context,
                                selected_origin="protected_guard_fallback",
                                baseline_verification_status="ok",
                                guided_verification_status="guard_fallback",
                            )
                        )
                    raw = dict(file="arrays.bin", group="arrays/" + tag)
                    rows.append(dict(event="episode_complete", tag=tag, steps=steps, raw=raw))
                    rows.append(
                        dict(
                            event="teacher_agreement_complete",
                            episode_raw=raw,
                            rows=steps,
                            actor_hash="a" * 64,
                            action_agreement=0.4,
                        )
                    )
        for index, row in enumerate(rows):
            row.update(id=f"event-{index:09d}", kind=row["event"])
        ledger = run / "evidence/ledger.jsonl"
        ledger.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in rows))
        result = json.loads((run / "worker_result.json").read_text())
        result["checkpoint_records"] = []
        for stream in range(3):
            for suffix in ("parent_initial", "parent"):
                checkpoint = run / f"checkpoints/s{stream}_{suffix}.pt"
                checkpoint.write_bytes(b"same hash aliases are legitimate synthetic payload")
                result["checkpoint_records"].append(
                    dict(file=str(checkpoint.relative_to(run)), **binding(checkpoint))
                )
        save(run / "worker_result.json", result)
        owner = json.loads((audit / "ownership.json").read_text())
        owner["worker_result_sha256"] = digest(run / "worker_result.json")
        save(audit / "ownership.json", owner)
        audit_receipt = json.loads((audit / "result/receipt.json").read_text())
        audit_receipt["file_hashes"]["ledger.jsonl"] = digest(ledger)
        audit_receipt["structure"]["ledger_events"] = len(rows)
        audit_receipt["scientific"]["selection_checks"] = controls
        save(audit / "result/receipt.json", audit_receipt)
        collector = module.Collector(run, audit, self.root / "out")
        recorded = collector.collect()
        self.assertEqual(recorded["scalar_event_count"], 84)
        self.assertEqual(recorded["selection_metadata_count"], 434)
        self.assertEqual(recorded["mechanism_summary"]["1"]["selector_only"]["decisions"], 204)
        self.assertEqual(recorded["mechanism_summary"]["2"]["proposal_guided"]["decisions"], 202)
        external = json.loads((collector.packet / "external_catalog.json").read_text())["files"]
        self.assertEqual(sum(name.startswith("checkpoints/") for name in external), 6)
        self.assertFalse(any("child" in name for name in external))

    def test_selection_context_must_join_completed_query(self):
        def change(rows):
            rows[0]["arm"] = "selector_only"

        run, audit, _ = fixture(self.root, mutate=change)
        collector = module.Collector(run, audit, self.root / "out")
        with self.assertRaisesRegex(module.CollectionError, "completed native model-query context"):
            collector.collect()

    def test_ledger_hash_mismatch_stops_before_archive(self):
        run, audit, _ = fixture(self.root)
        path = run / "evidence/ledger.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"action_mse": 0.5', b'"action_mse": 0.6'))
        collector = module.Collector(run, audit, self.root / "out")
        with self.assertRaisesRegex(module.CollectionError, "Ledger hash differs"):
            collector.collect()
        self.assertFalse((collector.out / "packet.tar.gz").exists())

    def test_packet_cap_funds_before_copy_or_archive(self):
        run, audit, _ = fixture(self.root)
        collector = module.Collector(run, audit, self.root / "out", packet_limit=1)
        with self.assertRaisesRegex(module.CollectionError, "before write"):
            collector.collect()
        self.assertEqual(collector.packet_bytes, 0)
        self.assertFalse(any(collector.packet.rglob("*")))
        self.assertFalse((collector.out / "packet.tar.gz").exists())

    def test_missing_failed_audit_and_incomplete_worker_are_not_success(self):
        for case in ("missing", "failed", "incomplete", "outer_exit"):
            with self.subTest(case=case):
                root = self.root / case
                root.mkdir()
                run, audit, _ = fixture(
                    root, worker_complete=case != "incomplete", audit_complete=case != "failed"
                )
                if case == "missing":
                    (audit / "result/receipt.json").unlink()
                if case == "outer_exit":
                    save(audit / "exit.json", dict(exit_code=1, stopping_reason=None))
                collector = module.Collector(run, audit, root / "out")
                with self.assertRaises(module.CollectionError):
                    collector.collect()
                self.assertFalse((collector.out / "receipt.json").exists())
                self.assertFalse((collector.out / "packet.tar.gz").exists())

    def test_unknown_or_missing_selection_fields_fail_closed(self):
        for case in ("context", "origin", "status"):
            with self.subTest(case=case):
                root = self.root / case
                root.mkdir()

                def change(rows):
                    if case == "context":
                        rows[1].pop("context")
                    elif case == "origin":
                        rows[1].pop("selected_origin")
                    else:
                        rows[1]["guided_verification_status"] = "unknown"

                run, audit, _ = fixture(root, mutate=change)
                collector = module.Collector(run, audit, root / "out")
                with self.assertRaises(module.CollectionError):
                    collector.collect()
                self.assertFalse((collector.out / "packet.tar.gz").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
