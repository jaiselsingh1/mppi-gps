"""Small closed-evidence fixtures for recovery; no simulator or learner calls."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from src.gps.mppi_gps.audits.consequence_recovery import (
    RecoveryError,
    _predictions,
    _Reader,
    _worker_journal,
    recover,
)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def bind(payload):
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else encode(value)
    path.write_bytes(payload)
    return bind(payload)


def fixture(tmp_path):
    run, art = tmp_path / "run", tmp_path / "artifacts"
    run.mkdir()
    art.mkdir()
    config = dict(
        source_episodes_per_stream=1,
        source_controls_per_episode=1,
        seeds=dict(primary=[11], confirmation=[12]),
        task=dict(frame_skip=4, terminal_unhealthy_weight=20),
        learning=dict(updates=2),
    )
    registry = {}
    fits = {}
    for arm in ("parent", "uniform", "cost_effect", "shuffled"):
        alias = "s0_" + arm
        name = "checkpoints/" + alias + ".pt"
        payload = b"not a decodable checkpoint: " + alias.encode()
        identity = put(run, name, payload)
        registry[alias] = dict(
            file=name,
            **identity,
            actor_hash=hashlib.sha256(alias.encode()).hexdigest(),
            stream=0,
            arm=arm,
        )
        if arm != "parent":
            curve = [
                dict(fit_update=i, update=i + 2, action_mse=0.2 / i, total_loss=0.2 / i)
                for i in (1, 2)
            ]
            fits[alias] = dict(
                tag=alias,
                sampled_updates=2,
                completed_updates=2,
                fitting_evidence=dict(learning_curve=curve, failed=False),
            )
    teacher = dict(
        source_tag="teacher_s0_independent_00",
        walking=True,
        survived=True,
        controls=1,
        cost=1.0,
        h=1,
    )
    replay = dict(passed=True, episodes=[teacher])
    evaluations = []
    panels = {}
    gaps = []
    for alias, actor in registry.items():
        arm = actor["arm"]
        for panel in ("primary", "confirmation", "source"):
            row = dict(
                tag=f"eval_{alias}_{panel}_00",
                actor=alias,
                panel=panel,
                case=0,
                stream=0,
                arm=arm,
                seed=None if panel == "source" else config["seeds"][panel][0],
                source_tag=teacher["source_tag"] if panel == "source" else None,
                h=1,
                controls=1,
                done=True,
                survived=False,
                walking=False,
                mae=1.0,
                raw_cost=1.0,
                native_terminal=20.0,
                native_cost=21.0,
                absorbing_cost=0.0,
                cost=21.0,
            )
            evaluations.append(row)
            panels.setdefault(panel, {}).setdefault("0", {})[arm] = dict(
                episodes=1,
                successes=0,
                survived=0,
                controls=1,
                mean_cost=21.0,
                mean_steps=1.0,
                mean_velocity_mae=1.0,
            )
            if panel == "source":
                gaps.append(dict(actor=alias, source_tag=teacher["source_tag"], cost_gap=20.0))
    gates = dict(confirmed=False, uniform_parent_confirmed=False)
    outcomes = dict(panels=panels, gates=gates, source_context_gaps=gaps)
    requested = dict(
        neural_rows=3,
        prediction_calls=1,
        native_controls=14,
        physics_substeps=56,
        external_mj_forward_calls=0,
        snapshot_reads=28,
    )
    ledger = [
        dict(event="runtime_open"),
        dict(event="work_requested", category="neural_rows", amount=3, operation=1),
        dict(event="work_complete", category="neural_rows", amount=3, operation=1),
    ]
    for ordinal in range(14):
        ledger.extend(
            [
                dict(event="trace_intent", alias="example", ordinal=ordinal),
                dict(event="trace_returned", alias="example", ordinal=ordinal),
            ]
        )
    ledger.append(dict(event="runtime_close"))
    for index, row in enumerate(ledger):
        row["id"] = f"event-{index:09d}"
    ledger_bytes = b"".join(encode(row) for row in ledger)
    evidence = {
        "arrays.bin": put(run, "worker/evidence/arrays.bin", b"not a NumPy archive"),
        "arrays.index.jsonl": put(run, "worker/evidence/arrays.index.jsonl", b"opaque index\n"),
        "ledger.jsonl": put(run, "worker/evidence/ledger.jsonl", ledger_bytes),
    }
    worker = dict(
        status="COMPLETED_FIXED_C1_ROSTER",
        error=None,
        parent_refits=0,
        teacher_reacquisitions=0,
        followup_launched=False,
        fit_records=fits,
        student_outcomes=panels,
        gates=gates,
        costs=dict(requested=requested, completed=requested),
    )
    inputs = {}
    for name, value in {
        "config.json": config,
        "worker_result.json": worker,
        "worker/actor_registry.json": registry,
        "worker/fit_records.json": fits,
        "worker/replay_gate.json": replay,
        "worker/outcomes.json": outcomes,
        "worker/evaluations.jsonl": b"".join(encode(row) for row in evaluations),
    }.items():
        inputs[name] = put(run, name, value)
    receipt = dict(
        schema="walker-student-consequence-c1.0-audit-v1",
        passed=True,
        full_scientific_audit_passed=True,
        scientific_contrast_completed=True,
        evidence_integrity_passed=True,
        physics_controls=0,
        optimizer_steps=0,
        backward_rows=0,
        unknown_partial_prediction_rows=0,
        prediction_calls=2,
        prediction_rows=3,
        completed_prediction_rows=3,
        teacher_replays=1,
        assays=1,
        standalone_episodes=12,
        weighting_confirmed=False,
        config_sha256=inputs["config.json"]["sha256"],
        worker_result_sha256=inputs["worker_result.json"]["sha256"],
        input_files=inputs,
        outcomes=dict(panels=panels, gates=gates),
        stream=dict(
            structural_passed=True,
            complete_journal=True,
            unknown_partial_controls=0,
            pending_trace_alias=None,
            file_hashes={k: v["sha256"] for k, v in evidence.items()},
            event_inventory=dict(Counter(r["event"] for r in ledger)),
        ),
        accounting=dict(
            evidence_file_bytes={k: v["bytes"] for k, v in evidence.items()},
            independent_requested=requested,
            independent_completed=requested,
            independent_unknown_partial={},
            independent_returned_unpersisted={},
            complete_reconciliation=True,
        ),
    )
    receipt_binding = put(art, "audit_001/result/audit.json", receipt)
    owner = registry["s0_parent"]["actor_hash"]
    predictions = []
    completed = 0
    for call, amount in enumerate((1, 2)):
        predictions.append(
            dict(
                id=2 * call,
                event="prediction_requested",
                actor_hash=owner,
                scope=None,
                rows=amount,
                call=call,
                requested_rows_after=completed + amount,
            )
        )
        completed += amount
        predictions.append(
            dict(
                id=2 * call + 1,
                event="prediction_complete",
                actor_hash=owner,
                scope=None,
                rows=amount,
                completed_rows_after=completed,
            )
        )
    put(art, "audit_001/result/predictions.jsonl", b"".join(encode(row) for row in predictions))
    csv_out = io.StringIO(newline="")
    fields = (
        "source_tag",
        "t",
        "h",
        "controls",
        "done",
        "native_terminal",
        "absorbing_cost",
        "cost",
        "native_cost",
        "baseline_cost",
        "delta",
    )
    writer = csv.DictWriter(csv_out, fieldnames=fields)
    writer.writeheader()
    writer.writerow(
        dict(
            source_tag=teacher["source_tag"],
            t=0,
            h=1,
            controls=1,
            done=False,
            native_terminal=0,
            absorbing_cost=0,
            cost=1,
            native_cost=1,
            baseline_cost=1,
            delta=0,
        )
    )
    put(art, "audit_001/result/assay_scalars.csv", csv_out.getvalue().encode())
    put(art, "audit_001/result/stream/receipt.json", dict(structural_passed=True))
    put(art, "audit_001/result/stream/stream_index.sqlite", b"opaque generated cache")
    put(art, "audit_001/result/stream/trace_summaries.jsonl", b"{}\n")
    put(art, "audit_001/stdout.txt", b"")
    trace = put(
        art, "audit_launch_001/stdout.txt", b"Pinned Linux worker RSS could not be measured\n"
    )
    timing = put(art, "audit_launch_001/time.txt", b"Exit status: 1\n")
    failure = dict(
        schema="walker-c1-audit-launcher-failure-v1",
        overall_audit_closure_passed=False,
        audit_coordinator_failed=True,
        original_audit_exit_exists=False,
        original_child_exit_code=None,
        coordinator_exit_code=1,
        scientific_receipt=dict(file="audit_001/result/audit.json", **receipt_binding),
        traceback=dict(file="audit_launch_001/stdout.txt", **trace),
        gnu_time=dict(file="audit_launch_001/time.txt", **timing),
    )
    failure_binding = put(art, "audit_launcher_failure.json", failure)
    files = {
        str(p.relative_to(art / "audit_001")): bind(p.read_bytes())
        for p in sorted((art / "audit_001").rglob("*"))
        if p.is_file()
    }
    catalog = dict(
        audit_outer=str(art / "audit_001"),
        file_count=len(files),
        file_bytes=sum(v["bytes"] for v in files.values()),
        files=files,
        failure_companion=failure_binding,
        accounting=dict(
            previous_wall_seconds=0.5,
            helper_wall_seconds_through_hashing=0.01,
            cumulative_read_bytes=1000,
        ),
    )
    catalog_binding = put(art, "audit_output_catalog_failure_001.json", catalog)
    protocol = dict(
        schema="walker-c1-audit-recovery-protocol-v1",
        study="C1.0-A1",
        user_authorized=True,
        automatic_followup=False,
        locations=dict(original_run=str(run), original_artifacts=str(art)),
        anchors=dict(
            scientific_receipt=dict(
                root="artifacts", file="audit_001/result/audit.json", **receipt_binding
            ),
            failure=dict(root="artifacts", file="audit_launcher_failure.json", **failure_binding),
            audit_output_catalog=dict(
                root="artifacts", file="audit_output_catalog_failure_001.json", **catalog_binding
            ),
        ),
        limits=dict(
            wall_seconds=30,
            rss_bytes=1024**3,
            read_bytes=64 * 1024**2,
            output_bytes=1024**2,
            max_files=100,
            max_json_bytes=16 * 1024**2,
            max_journal_line_bytes=512,
            max_lifecycle_line_bytes=2048,
        ),
        cumulative=dict(
            prior_wall_upper_seconds=1,
            prior_read_upper_bytes=1000,
            audit_wall_limit_seconds=7200,
            audit_read_limit_bytes=64 * 1024**3,
        ),
        expected_counts=dict(
            input_files=len(inputs),
            actors=4,
            teacher_replays=1,
            assays=1,
            fit_blocks=3,
            curve_rows=6,
            standalone=12,
            prediction_calls=2,
            prediction_rows=3,
            prediction_events=4,
        ),
        new_scientific_work=dict.fromkeys(
            (
                "neural_rows",
                "prediction_calls",
                "checkpoint_tensor_loads",
                "native_controls",
                "physics_substeps",
                "backward_rows",
                "optimizer_steps",
                "resets",
                "restores",
                "mppi_queries",
            ),
            0,
        ),
    )
    return protocol, run, art


def test_complete_recovery_preserves_original_failure_without_science(tmp_path):
    protocol, run, art = fixture(tmp_path)
    result = recover(protocol, tmp_path / "new")
    assert result["recovery_checks_passed"], result.get("error")
    assert result["admitted_original_scientific_checks"]
    assert result["retained_numerical_certification"]
    assert result["original_overall_audit_closure_passed"] is False
    assert result["original_child_exit_code"] is None
    assert result["new_scientific_work"] == protocol["new_scientific_work"]
    assert not (art / "audit_001/exit.json").exists()
    assert result["saved_outcomes"]["walking"] == 0
    assert result["saved_fitting"]["complete_curve_rows"] == 6
    assert result["prediction_journal"]["events"] == 4
    assert result["work"]["read_bytes"] > 0
    assert (
        result["protocol_sha256"]
        == hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize("target", ["checkpoint", "source", "prediction"])
def test_changed_preserved_bytes_stop_and_record_paid_work(tmp_path, target):
    protocol, run, art = fixture(tmp_path)
    path = {
        "checkpoint": run / "checkpoints/s0_uniform.pt",
        "source": run / "config.json",
        "prediction": art / "audit_001/result/predictions.jsonl",
    }[target]
    payload = path.read_bytes()
    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    result = recover(protocol, tmp_path / "new")
    assert not result["recovery_checks_passed"]
    assert result["accounting"]["read_bytes"] > 0
    assert (tmp_path / "new/recovery.json").is_file()


@pytest.mark.parametrize("path", ["../outside.json", "/absolute.json"])
def test_anchor_path_escape_is_rejected(tmp_path, path):
    protocol, _, _ = fixture(tmp_path)
    protocol["anchors"]["failure"]["file"] = path
    result = recover(protocol, tmp_path / "new")
    assert not result["recovery_checks_passed"]
    assert "escapes" in result["error"]["message"]


def test_symlink_input_is_rejected(tmp_path):
    protocol, run, _ = fixture(tmp_path)
    path = run / "config.json"
    original = path.read_bytes()
    path.unlink()
    other = tmp_path / "config-copy.json"
    other.write_bytes(original)
    path.symlink_to(other)
    result = recover(protocol, tmp_path / "new")
    assert not result["recovery_checks_passed"]
    assert "Symlink" in result["error"]["message"]


def test_original_missing_exit_must_not_be_fabricated(tmp_path):
    protocol, _, art = fixture(tmp_path)
    put(art, "audit_001/exit.json", dict(exit_code=0))
    result = recover(protocol, tmp_path / "new")
    assert not result["recovery_checks_passed"]
    assert "replaced" in result["error"]["message"]


def test_read_budget_stops_before_large_payloads(tmp_path):
    protocol, _, _ = fixture(tmp_path)
    protocol["limits"]["read_bytes"] = 100
    result = recover(protocol, tmp_path / "new")
    assert not result["recovery_checks_passed"]
    assert result["accounting"]["read_bytes"] == 0
    assert "read budget" in result["error"]["message"]


def test_cumulative_budget_cannot_be_reset(tmp_path):
    protocol, _, _ = fixture(tmp_path)
    protocol["cumulative"]["prior_wall_upper_seconds"] = 7199
    result = recover(protocol, tmp_path / "new")
    assert not result["recovery_checks_passed"]
    assert result["accounting"]["read_bytes"] == 0
    assert "Cumulative audit time" in result["error"]["message"]


def test_unknown_science_or_auto_followup_is_rejected(tmp_path):
    protocol, _, _ = fixture(tmp_path)
    protocol["new_scientific_work"]["neural_rows"] = 1
    result = recover(protocol, tmp_path / "new")
    assert not result["recovery_checks_passed"]
    assert result["accounting"]["read_bytes"] == 0


def test_create_only_attempt_cannot_overwrite(tmp_path):
    protocol, _, _ = fixture(tmp_path)
    out = tmp_path / "new"
    out.mkdir()
    (out / "retained").write_text("paid prefix")
    with pytest.raises(FileExistsError):
        recover(protocol, out)
    assert (out / "retained").read_text() == "paid prefix"


def test_module_import_does_not_load_science_libraries():
    root = Path(__file__).resolve().parents[1]
    code = """import builtins,sys
sys.path.insert(0,ROOT)
old=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','numpy','mujoco','gymnasium','stable_baselines3','jax'}:
  raise AssertionError('scientific import: '+name)
 return old(name,*args,**kwargs)
builtins.__import__=guard
from src.gps.mppi_gps.audits.consequence_recovery import recover
""".replace("ROOT", repr(str(root)))
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", ["partial", "pending", "wrong_id", "unknown_actor", "wrong_rows"])
def test_saved_prediction_journal_requires_complete_paired_accounting(tmp_path, fault):
    protocol, run, art = fixture(tmp_path)
    path = art / "audit_001/result/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if fault == "partial":
        payload = path.read_bytes()[:-1]
    elif fault == "pending":
        payload = b"".join(encode(row) for row in rows[:-1])
    else:
        if fault == "wrong_id":
            rows[-1]["id"] = 99
        elif fault == "unknown_actor":
            rows[0]["actor_hash"] = "unknown"
        else:
            rows[-1]["completed_rows_after"] = 4
        payload = b"".join(encode(row) for row in rows)
    path.write_bytes(payload)
    reader = _Reader(protocol)
    registry = json.loads((run / "worker/actor_registry.json").read_text())
    with pytest.raises(RecoveryError):
        _predictions(
            reader,
            "audit_001/result/predictions.jsonl",
            protocol["expected_counts"],
            {row["actor_hash"] for row in registry.values()},
        )
    assert reader.read_bytes > 0


def test_conflicting_hash_contracts_fail_before_hash_pass(tmp_path):
    protocol, run, _ = fixture(tmp_path)
    reader = _Reader(protocol)
    original = bind((run / "config.json").read_bytes())
    reader.add("run", "config.json", original, "first contract")
    other = dict(original, sha256="0" * 64)
    with pytest.raises(RecoveryError, match="Conflicting"):
        reader.add("run", "config.json", other, "second contract")
    assert reader.read_bytes == 0


def test_missing_frozen_input_cannot_be_replaced_by_receipt_claim(tmp_path):
    protocol, run, _ = fixture(tmp_path)
    (run / "checkpoints/s0_uniform.pt").unlink()
    result = recover(protocol, tmp_path / "new")
    assert result["recovery_checks_passed"] is False
    assert result["error"]["type"] == "FileNotFoundError"


def test_failed_attempt_receipt_does_not_claim_outer_admission(tmp_path):
    protocol, _, _ = fixture(tmp_path)
    protocol["automatic_followup"] = True
    result = recover(protocol, tmp_path / "new")
    assert result["recovery_checks_passed"] is False
    assert result["admitted_original_scientific_checks"] is False
    assert "overall_audit_closure_passed" not in result
    assert result["original_overall_audit_closure_passed"] is False
    assert result["accounting"]["read_bytes"] == 0


@pytest.mark.parametrize("fault", ["native_return", "pending_work", "long_frequent_line"])
def test_worker_ledger_rejects_unclosed_or_invalid_paid_boundaries(tmp_path, fault):
    protocol, run, art = fixture(tmp_path)
    path = run / "worker/evidence/ledger.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if fault == "native_return":
        rows[4]["ordinal"] = 1
    elif fault == "pending_work":
        rows = rows[:2]
    else:
        rows[1]["diagnostic"] = "x" * 600
    path.write_bytes(b"".join(encode(row) for row in rows))
    receipt = json.loads((art / "audit_001/result/audit.json").read_text())
    reader = _Reader(protocol)
    with pytest.raises(RecoveryError):
        _worker_journal(reader, receipt, 4)
    assert reader.read_bytes > 0


def test_only_the_two_existing_lifecycle_events_can_use_larger_lines(tmp_path):
    protocol, run, art = fixture(tmp_path)
    path = run / "worker/evidence/ledger.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows[-1]["diagnostic"] = "x" * 600
    path.write_bytes(b"".join(encode(row) for row in rows))
    receipt = json.loads((art / "audit_001/result/audit.json").read_text())
    summary = _worker_journal(_Reader(protocol), receipt, 4)
    assert summary["completed"]["snapshot_reads"] == 28


def test_output_budget_failure_retains_small_error_receipt(tmp_path):
    protocol, _, _ = fixture(tmp_path)
    protocol["limits"]["output_bytes"] = 4096
    result = recover(protocol, tmp_path / "new")
    assert not result["recovery_checks_passed"]
    assert "output budget" in result["error"]["message"]
    assert not (tmp_path / "new/verified_files.json").exists()
    assert (tmp_path / "new/recovery.json").stat().st_size <= 4096


def test_repeated_input_identity_must_stay_unchanged(tmp_path):
    protocol, run, _ = fixture(tmp_path)
    reader = _Reader(protocol)
    reader.json("run", "config.json")
    path = run / "config.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RecoveryError, match="changed"):
        reader.unchanged()


@pytest.mark.parametrize("root_key", ["original_run", "original_artifacts"])
def test_output_inside_original_evidence_is_rejected_before_any_write(tmp_path, root_key):
    protocol, _, _ = fixture(tmp_path)
    out = Path(protocol["locations"][root_key]) / "do-not-create"
    with pytest.raises(RecoveryError, match="inside original evidence"):
        recover(protocol, out)
    assert not out.exists()


def test_output_symlink_parent_cannot_bypass_original_tree_guard(tmp_path):
    protocol, run, _ = fixture(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(run, target_is_directory=True)
    with pytest.raises(RecoveryError, match="inside original evidence"):
        recover(protocol, alias / "do-not-create")
    assert not (run / "do-not-create").exists()


@pytest.mark.parametrize("operation", ["metadata", "hash"])
def test_growing_input_never_reads_beyond_reserved_original_bytes(tmp_path, monkeypatch, operation):
    protocol, run, _ = fixture(tmp_path)
    path = run / "config.json"
    original = path.read_bytes()
    protocol["limits"]["read_bytes"] = len(original)
    reader = _Reader(protocol)
    real_open = reader._open
    requests = []

    class GrowingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, amount):
            requests.append(amount)
            if len(requests) == 1:
                with path.open("ab") as target:
                    target.write(b"x" * 100000)
            return self.stream.read(amount)

    def changed_open(candidate):
        stream, identity = real_open(candidate)
        return GrowingStream(stream), identity

    monkeypatch.setattr(reader, "_open", changed_open)
    with pytest.raises(RecoveryError, match="changed"):
        if operation == "metadata":
            reader.read("run", "config.json")
        else:
            reader.add("run", "config.json", bind(original), "frozen")
            reader.hash_roster()
    assert sum(requests) == len(original)
    assert reader.read_bytes == len(original)
