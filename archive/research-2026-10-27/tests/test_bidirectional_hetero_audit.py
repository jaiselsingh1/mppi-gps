"""Synthetic saved packets only; no producer/task/engine/solver/NN imports.

These tests are authored for later authorized execution. Their fixtures encode
an arbitrary affine output, never a claim to simulated Acrobot dynamics.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.gps.bidirectional.hetero_audit import audit_heterogeneous


TASK = dict(T=800, dt=.01, capture_steps=100, capture_distance=.2, capture_qvel_norm=.5,
    local_distance=.5, local_by=600, failure_penalty=1000., reset_angle_radius=.1, reset_velocity_radius=.05)
BASE = ("assets/acrobot.xml", "src/envs/acrobot.py", "src/envs/mujoco_env.py", "src/gps/bidirectional/task.py")
EXECUTOR = "src/gps/bidirectional/hetero_executor.py"
RESTART = "explicit-zero-solver-warmstart-intervention"
SCHEMA = "bidirectional-heterogeneous-native-h1-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def packed(value):
    return np.array(canonical(value))


class SavedFixture:
    def __init__(self, path):
        self.root = Path(path)/"run"
        self.archive = self.root/"evidence"/"heterogeneous"
        self.archive.mkdir(parents=True)
        self.rows, self.query_number = [], 0
        contents = {name: ("synthetic frozen source: "+name+"\n").encode() for name in (*BASE, EXECUTOR)}
        contents[BASE[0]] = b'''<mujoco><option timestep="0.01" integrator="RK4"><flag constraint="disable"/></option>
<actuator><motor joint="elbow" gear="2" ctrllimited="true" ctrlrange="-1 1"/></actuator></mujoco>'''
        inventory = []
        for name, blob in contents.items():
            target = self.root/"source"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            inventory.append(dict(file="source/"+name, sha256=sha(blob), bytes=len(blob)))
        self.source_hashes = {name: sha(contents[name]) for name in contents}
        self.task_hash = sha(canonical(dict(schema="acrobot-capture-v2-post-action-fk", config=TASK,
            base_cost="unchanged-src.envs.acrobot.Acrobot", cost_boundary_source="post-action-full-state-analytic-fk",
            base_source_sha256={name:self.source_hashes[name] for name in BASE})).encode())
        config_blob, protocol_blob = canonical(dict(max_controls=100_000)).encode(), canonical(dict(schema="synthetic-protocol")).encode()
        (self.root/"config.json").write_bytes(config_blob)
        (self.root/"protocol.json").write_bytes(protocol_blob)
        identity = dict(files=inventory, config_sha256=sha(config_blob), protocol_sha256=sha(protocol_blob),
                        dependency_versions={"numpy":"synthetic-not-executed"})
        self.run_fingerprint = sha(canonical(identity).encode())
        (self.root/"manifest.json").write_text(canonical(dict(**identity, source_fingerprint=self.run_fingerprint)))
        self.append("archive_begin", schema="bidirectional-evidence-v1")

    def append(self, kind, **fields):
        row = dict(id=f"event-{len(self.rows):09d}", kind=kind,
                   utc="2026-10-27T00:00:00+00:00", wall_seconds=len(self.rows)*.01, **fields)
        self.rows.append(row)
        return row

    def arrays(self, name, values):
        target = self.archive/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            np.savez_compressed(stream, **values)
        blob = target.read_bytes()
        return dict(file=name, sha256=sha(blob), bytes=len(blob))

    def read_arrays(self, reference):
        with np.load(self.archive/reference["file"], allow_pickle=False) as packet:
            return {name:packet[name].copy() for name in packet.files}

    def rewrite_arrays(self, reference, mutate):
        values = self.read_arrays(reference)
        mutate(values)
        replacement = self.arrays(reference["file"], values)
        reference.clear(); reference.update(replacement)

    def query(self, elapsed=(0, 799), *, cumulative=None, reservation=None, resolution="complete"):
        index = self.query_number; self.query_number += 1
        qid = f"q{index:09d}"; n = len(elapsed)
        elapsed = np.array(elapsed, dtype=np.int64)
        states = np.zeros((n, 5)); states[:, 0] = elapsed*.01; states[:, 1:5] = [.2, -.3, .4, -.5]
        actions = np.zeros((n, 1, 1)); actions[:, 0, 0] = np.linspace(-1., 1., n)
        fingerprints = dict(task_hash=self.task_hash, task_config_sha256=sha(canonical(TASK).encode()),
            heterogeneous_executor_sha256=self.source_hashes[EXECUTOR], run_source_fingerprint=self.run_fingerprint)
        owner, metadata = dict(controller="synthetic", actor_hash=None), dict(stencil_axis=0)
        common = dict(schema=np.array(SCHEMA), task_hash=np.array(self.task_hash), task_json=packed(TASK),
            source_fingerprints_json=packed(fingerprints), owner_json=packed(owner), metadata_json=packed(metadata),
            purpose=np.array("synthetic_fd"), dynamics_only=np.array(True), returns_semantics=np.array("none"),
            initial_states=states, initial_elapsed=elapsed, initial_warmstarts=np.zeros((n, 2)),
            initial_warmstart_exact=np.zeros(n, dtype=bool), initial_restart_semantics=np.array(RESTART), actions=actions)
        inputs = self.arrays(f"queries/{qid}_heterogeneous_inputs.npz", common)
        previous = sum(r.get("requested_controls", 0) for r in self.rows if r["kind"] == "query_requested")
        request = self.append("query_requested", query_id=qid, schema=SCHEMA, query_kind="heterogeneous_native_h1",
            mode="model_heterogeneous", scope="synthetic", owner=owner, metadata=metadata, purpose="synthetic_fd",
            task_hash=self.task_hash, source_fingerprints=fingerprints, rows=n, horizon=1, initial_elapsed=elapsed.tolist(),
            dynamics_only=True, returns_semantics="none", requested_controls=n, requested_physics_controls=n,
            requested_physics_frames=n, cutoff_skipped_controls=0, budget_reservation_id=reservation or f"b{index:09d}",
            budget_requested_cumulative=previous+n if cumulative is None else cumulative, inputs=inputs)
        finish = None
        if resolution == "complete":
            output = states[:, None, :].copy(); output[:, 0, 0] += .01
            output[:, 0, 1] += actions[:, 0, 0]
            result = dict(**common, prestates=states[:, None, :], states=output, elapsed=elapsed+1,
                input_sha256=np.array(inputs["sha256"]), endpoint_warmstarts=np.zeros((n, 2)),
                endpoint_warmstart_exact=np.zeros(n, dtype=bool), endpoint_restart_semantics=np.array(RESTART),
                backend_json=packed(dict(backend="mujoco_cpu", frame_skip=1, horizon=1,
                    timestep=.01, integrator="RK4", heterogeneous_initial_states=True)))
            raw = self.arrays(f"queries/{qid}_heterogeneous_result.npz", result)
            finish = self.append("query_complete", query_id=qid, schema=SCHEMA, scope="synthetic", owner=owner,
                purpose="synthetic_fd", dynamics_only=True, returns_semantics="none", backend="mujoco_cpu",
                completed_controls=n, completed_physics_frames=n, query_wall_seconds=.001,
                budget_reservation_id=request["budget_reservation_id"], result=raw)
        elif resolution in {"unknown_error", "returned_error"}:
            returned = resolution == "returned_error"
            finish = self.append("query_error", query_id=qid, schema=SCHEMA, scope="synthetic", purpose="synthetic_fd",
                call_started=True, call_returned=returned, completed_controls=n if returned else 0,
                completed_physics_frames=n if returned else 0, partial_physics_unknown=not returned,
                unknown_partial=not returned, error_type="SyntheticError", error="fixture only",
                query_wall_seconds=.001, budget_reservation_id=request["budget_reservation_id"])
        return request, finish

    def close(self):
        requests = {r["query_id"]:r for r in self.rows if r["kind"] == "query_requested"}
        done = {r["query_id"]:r for r in self.rows if r["kind"] == "query_complete"}
        errors = {r["query_id"]:r for r in self.rows if r["kind"] == "query_error"}
        returned = {q:r for q,r in errors.items() if r["call_returned"]}
        unresolved = set(requests)-set(done)-set(errors)
        unknown_errors = set(errors)-set(returned)
        controls = lambda ids: sum(requests[q]["requested_controls"] for q in ids)
        requested = controls(requests)
        counts = dict(requested_controls=requested, charged_requested_controls=requested,
            requested_physics_controls=requested, requested_physics_frames=requested,
            completed_controls=sum(r["completed_controls"] for r in {**done, **returned}.values()),
            completed_physics_frames=sum(r["completed_physics_frames"] for r in {**done, **returned}.values()),
            cutoff_skipped_controls=0, error_requested_controls=controls(errors), unknown_error_controls=controls(unknown_errors),
            inflight_or_interrupted_controls=controls(unresolved), unknown_controls=controls(unresolved|unknown_errors),
            queries=len(requests), query_attempts=len(requests), completed_queries=len(done), error_queries=len(errors),
            inflight_or_interrupted_queries=len(unresolved), rejected_queries=0, incomplete_final_line=False)
        self.append("archive_close", accounting=counts)
        self.write_ledger()

    def write_ledger(self):
        (self.archive/"ledger.jsonl").write_text("".join(canonical(row)+"\n" for row in self.rows))


def test_complete_saved_heterogeneous_rows_bind_sources_and_allow_native_budget_jumps(tmp_path):
    fixture = SavedFixture(tmp_path)
    fixture.query(cumulative=2)
    fixture.query(elapsed=(123,), cumulative=9, reservation="b000000005")
    fixture.close()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "PASS" and result["arithmetic_valid"] and result["closed"]
    assert result["counts"]["requested_controls"] == result["counts"]["completed_controls"] == 3
    assert result["counts"]["completed_rows_checked"] == 3
    assert result["counts"]["budget_requested_cumulative"] == 9
    assert len(result["queries"]) == 2 and not result["errors"] and not result["pending"]
    assert str(fixture.root/"manifest.json") in result["input_sha256"]
    assert str(fixture.root/"source"/BASE[0]) in result["input_sha256"]
    assert result["additional_physics_controls"] == result["additional_actor_inference_rows"] == 0


@pytest.mark.parametrize("corruption", ["output_clock", "input_elapsed", "actions", "warmstart", "exact_flag", "prestate", "return_field", "metadata", "backend"])
def test_rehashed_structural_corruption_rejects_without_erasing_paid_inventory(tmp_path, corruption):
    fixture = SavedFixture(tmp_path); request, complete = fixture.query()
    reference = complete["result"]
    def mutate(raw):
        if corruption == "output_clock": raw["states"][:, :, 0] += .01
        elif corruption == "input_elapsed": raw["initial_elapsed"][0] = 1
        elif corruption == "actions": raw["actions"][0, 0, 0] = 1.01
        elif corruption == "warmstart": raw["endpoint_warmstarts"][0, 0] = .1
        elif corruption == "exact_flag": raw["endpoint_warmstart_exact"][0] = True
        elif corruption == "prestate": raw["prestates"][0, 0, 1] += .1
        elif corruption == "return_field": raw["total_costs"] = np.zeros(2)
        elif corruption == "metadata": raw["metadata_json"] = packed(dict(stencil_axis=1))
        else: raw["backend_json"] = packed(dict(backend="warp"))
    fixture.rewrite_arrays(reference, mutate); fixture.close()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "FAIL" and not result["arithmetic_valid"]
    assert result["counts"]["requested_controls"] == result["counts"]["completed_controls"] == 2


@pytest.mark.parametrize("corruption", ["sha", "bytes", "traversal", "query_alias", "missing_result"])
def test_raw_reference_integrity_and_query_binding_reject(tmp_path, corruption):
    fixture = SavedFixture(tmp_path); _, complete = fixture.query()
    if corruption == "sha": complete["result"]["sha256"] = "0"*64
    elif corruption == "bytes": complete["result"]["bytes"] += 1
    elif corruption == "traversal": complete["result"]["file"] = "../outside.npz"
    elif corruption == "query_alias": complete["result"]["file"] = "queries/q999999999_heterogeneous_result.npz"
    else: (fixture.archive/complete["result"]["file"]).unlink()
    fixture.close()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "FAIL" and result["counts"]["requested_controls"] == 2


@pytest.mark.parametrize("resolution,known,unknown", [("unknown_error", 0, 2), ("returned_error", 2, 0), ("pending", 0, 2)])
def test_closed_failed_or_unresolved_queries_are_pending_with_exact_paid_counts(tmp_path, resolution, known, unknown):
    fixture = SavedFixture(tmp_path); fixture.query(resolution=resolution); fixture.close()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "PENDING" and result["arithmetic_valid"] and result["closed"]
    assert result["counts"]["requested_controls"] == 2 and result["counts"]["completed_controls"] == known
    assert result["counts"]["unknown_controls"] == unknown and result["counts"]["completed_rows_checked"] == 0


def test_false_error_completion_and_unknown_flags_reject(tmp_path):
    fixture = SavedFixture(tmp_path); _, error = fixture.query(resolution="unknown_error")
    error["completed_controls"] = error["completed_physics_frames"] = 2
    error["unknown_partial"] = False
    fixture.close()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "FAIL" and result["counts"]["requested_controls"] == 2


def test_over_capacity_saved_query_cannot_pass_even_with_consistent_paid_counts(tmp_path):
    fixture = SavedFixture(tmp_path); fixture.query(elapsed=(0,)*8001); fixture.close()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "FAIL" and result["counts"]["requested_controls"] == 8001
    assert any("capacity" in error for error in result["errors"])


def test_coherently_rebound_wrong_task_hash_fails_independent_reconstruction(tmp_path):
    fixture = SavedFixture(tmp_path); fixture.task_hash = "0"*64
    fixture.query(); fixture.close()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "FAIL" and result["counts"]["requested_controls"] == 2
    assert any("reconstructed task identity" in error for error in result["errors"])


def test_explicit_larger_cap_cannot_relax_the_frozen_shared_budget(tmp_path):
    fixture = SavedFixture(tmp_path)
    fixture.query(cumulative=100_001); fixture.close()
    result = audit_heterogeneous(fixture.archive, max_requested_controls=200_000)
    assert result["status"] == "FAIL" and not result["arithmetic_valid"]
    assert result["counts"]["requested_controls"] == 2
    assert any("cap exceeded" in error for error in result["errors"])


@pytest.mark.parametrize("corruption", ["cumulative", "token", "cap", "request_count", "close_count", "duplicate_completion", "unexpected_work"])
def test_ledger_intent_budget_and_closure_failures_are_detected(tmp_path, corruption):
    fixture = SavedFixture(tmp_path); request, complete = fixture.query()
    second, _ = fixture.query(elapsed=(1,), cumulative=3)
    if corruption == "cumulative": second["budget_requested_cumulative"] = 2
    elif corruption == "token": second["budget_reservation_id"] = request["budget_reservation_id"]
    elif corruption == "cap": second["budget_requested_cumulative"] = 100_001
    elif corruption == "request_count": request["requested_physics_frames"] = 1
    elif corruption == "duplicate_completion": fixture.append("query_complete", **{k:v for k,v in complete.items() if k not in {"id", "kind", "utc", "wall_seconds"}})
    elif corruption == "unexpected_work": fixture.append("neural_requested", rows=1)
    fixture.close()
    if corruption == "close_count": fixture.rows[-1]["accounting"]["requested_controls"] = 0; fixture.write_ledger()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "FAIL" and not result["arithmetic_valid"]
    assert result["counts"]["requested_controls"] == 3


@pytest.mark.parametrize("missing", ["manifest", "task_source"])
def test_missing_frozen_source_is_pending_while_raw_paid_work_is_still_counted(tmp_path, missing):
    fixture = SavedFixture(tmp_path); fixture.query(); fixture.close()
    target = fixture.root/"manifest.json" if missing == "manifest" else fixture.root/"source"/BASE[-1]
    target.unlink()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "PENDING" and result["arithmetic_valid"]
    assert result["counts"]["requested_controls"] == result["counts"]["completed_controls"] == 2


@pytest.mark.parametrize("corruption", ["source_bytes", "config_bytes", "manifest_fingerprint", "ledger_task_hash", "run_fingerprint"])
def test_source_task_and_run_fingerprints_are_independently_bound(tmp_path, corruption):
    fixture = SavedFixture(tmp_path); request, complete = fixture.query()
    if corruption == "source_bytes": (fixture.root/"source"/BASE[1]).write_text("changed frozen bytes")
    elif corruption == "config_bytes": (fixture.root/"config.json").write_text("{}")
    elif corruption == "manifest_fingerprint":
        path = fixture.root/"manifest.json"; manifest = json.loads(path.read_text()); manifest["source_fingerprint"] = "0"*64; path.write_text(canonical(manifest))
    elif corruption == "ledger_task_hash": request["task_hash"] = "0"*64
    else:
        request["source_fingerprints"]["run_source_fingerprint"] = "0"*64
        fixture.rewrite_arrays(request["inputs"], lambda raw: raw.update(source_fingerprints_json=packed(request["source_fingerprints"])))
    fixture.close()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "FAIL" and result["counts"]["requested_controls"] == 2


def test_open_or_unfinished_tail_never_becomes_closed_pass(tmp_path):
    fixture = SavedFixture(tmp_path); fixture.query(); fixture.write_ledger()
    opened = audit_heterogeneous(fixture.archive)
    assert opened["status"] == "PENDING" and not opened["closed"]
    with (fixture.archive/"ledger.jsonl").open("ab") as stream: stream.write(b'{"kind":"query_requested"')
    interrupted = audit_heterogeneous(fixture.archive)
    assert interrupted["status"] == "PENDING" and interrupted["counts"]["incomplete_final_line"]
    assert interrupted["counts"]["requested_controls"] == 2


def test_complete_close_json_without_newline_is_still_unfinished(tmp_path):
    fixture = SavedFixture(tmp_path); fixture.query(); fixture.close()
    path = fixture.archive/"ledger.jsonl"; path.write_bytes(path.read_bytes().rstrip(b"\n"))
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "PENDING" and not result["closed"]


def test_symbolic_extra_fingerprint_is_pending_instead_of_silently_verified(tmp_path):
    fixture = SavedFixture(tmp_path); request, complete = fixture.query()
    request["source_fingerprints"]["unknown_symbol"] = "2"*64
    for reference in (request["inputs"], complete["result"]):
        fixture.rewrite_arrays(reference, lambda raw: raw.update(source_fingerprints_json=packed(request["source_fingerprints"])))
    fixture.rewrite_arrays(complete["result"], lambda raw: raw.update(input_sha256=np.array(request["inputs"]["sha256"])))
    fixture.close()
    result = audit_heterogeneous(fixture.archive)
    assert result["status"] == "PENDING" and result["arithmetic_valid"]
