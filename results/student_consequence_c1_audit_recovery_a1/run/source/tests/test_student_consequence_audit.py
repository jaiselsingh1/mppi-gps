"""Independent saved-only C1 arithmetic, bindings and incomplete evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.mppi_gps.audits.student_consequence import (
    TRACE_KEYS,
    AuditError,
    _independent_gates,
    _Inputs,
    _reconcile_accounting,
    _Work,
    audit_run,
    audit_saved_stream,
    compare_baseline_arrays,
    recompute_shuffled_weights,
    recompute_weights,
    verify_active_spool,
    verify_assay_arrays,
    verify_native_measurements,
    verify_trace_arrays,
)
from src.gps.mppi_gps.consequence_storage import C1TraceRecorder, CompactEvidenceMeter

TASK = dict(
    cost_style="target_velocity",
    target_velocity=1.5,
    vel_cost_weight=1.0,
    vertical_velocity_cost_weight=0.0,
    angle_cost_weight=0.1,
    unhealthy_cost_weight=5.0,
    ctrl_cost_weight=0.001,
    terminal_unhealthy_weight=20.0,
)


def synthetic_arrays(count=2, horizon=None, *, done=False, failed=False):
    horizon = count if horizon is None else horizon
    states = np.zeros((count + 1, 19), dtype="<f8")
    states[:, 2] = 1.25
    states[:, 10] = 1.5
    if done:
        states[-1, 2] = 0.75
    arrays = dict(
        observations=np.concatenate((states[:, 2:10], states[:, 10:19]), axis=1),
        states=states,
        warmstarts=np.zeros((count + 1, 9), dtype="<f8"),
        integrations=np.zeros((count + 1, 91), dtype="<f8"),
        actions=np.zeros((count, 6), dtype="<f8"),
        native_costs=np.zeros(count, dtype="<f8"),
        x_velocities=np.full(count, 1.5, dtype="<f8"),
        fall_costs=np.zeros(count, dtype="<f8"),
        done=np.zeros(count, dtype="u1"),
        control_ordinals=np.arange(count, dtype="<u4"),
        row_digests=np.empty((count, 32), dtype="u1"),
    )
    if done:
        arrays["done"][-1], arrays["native_costs"][-1], arrays["fall_costs"][-1] = 1, 25, 20
    arrays["metadata"] = np.array(
        json.dumps(
            dict(
                alias="synthetic",
                kind="baseline",
                horizon=horizon,
                completed_controls=count,
                availability="failed_actual_prefix" if failed else "closed_actual_prefix",
                absorbing_cost=5.0 * (horizon - count) if done and not failed else 0.0,
                unknown_partial_controls=0,
                unknown_partial_physics_substeps=0,
            )
        )
    )
    rehash(arrays)
    return arrays


def rehash(arrays):
    digest = hashlib.sha256()
    for key in TRACE_KEYS[:4]:
        digest.update(arrays[key][0].tobytes())
    previous = digest.digest()
    arrays["initial_digest"] = np.frombuffer(previous, dtype="u1").copy()
    for index in range(len(arrays["actions"])):
        digest = hashlib.sha256(previous)
        for key in TRACE_KEYS[:4]:
            digest.update(arrays[key][index + 1].tobytes())
        for key in TRACE_KEYS[4:10]:
            digest.update(np.asarray(arrays[key][index]).tobytes())
        previous = digest.digest()
        arrays["row_digests"][index] = np.frombuffer(previous, dtype="u1")


def source_from(arrays):
    source = {key: np.array(arrays[key], copy=True) for key in TRACE_KEYS[:5]}
    source.update(
        costs=arrays["native_costs"].copy(),
        velocities=arrays["x_velocities"].copy(),
        fall_costs=arrays["fall_costs"].copy(),
        policy_actions=arrays["actions"].astype(np.float32),
    )
    return source


def test_independent_hash_and_native_arithmetic_accept_actual_prefix():
    arrays = synthetic_arrays(2, horizon=100, done=True)
    summary = verify_trace_arrays(arrays, task=TASK)
    assert summary["controls"] == 2 and summary["done"]
    assert summary["raw_cost"] == 5
    assert summary["native_cost"] == 25 and summary["native_terminal"] == 20
    assert summary["absorbing_cost"] == 490
    assert summary["cost"] == 515
    assert arrays["states"].shape == (3, 19)


@pytest.mark.parametrize(
    "field",
    [
        "states",
        "warmstarts",
        "integrations",
        "actions",
        "native_costs",
        "x_velocities",
        "fall_costs",
        "done",
        "control_ordinals",
    ],
)
def test_hash_chain_detects_changed_returned_field(field):
    arrays = synthetic_arrays()
    arrays[field].flat[-1] += 1
    with pytest.raises(AuditError):
        verify_trace_arrays(arrays)


def test_source_replay_gate_does_not_relax_initial_bytes_or_absolute_tolerance():
    arrays = synthetic_arrays()
    source = source_from(arrays)
    assert compare_baseline_arrays(arrays, source)["states"] == 0
    source["warmstarts"][0, 0] = np.nextafter(0.0, 1.0)
    with pytest.raises(AuditError, match="byte-exact"):
        compare_baseline_arrays(arrays, source)
    source["warmstarts"][0, 0] = 0
    source["warmstarts"][1, 0] = 2e-10
    with pytest.raises(AuditError, match="replay differs"):
        compare_baseline_arrays(arrays, source)


def test_assay_saved_action_and_fixed_suffix_are_not_replaced_by_reference_policy():
    teacher = synthetic_arrays(1000)
    source = source_from(teacher)
    arrays = synthetic_arrays(2)
    metadata = json.loads(arrays["metadata"].item())
    metadata.update(kind="assay", t=998)
    arrays["metadata"] = np.array(json.dumps(metadata))
    result = verify_assay_arrays(arrays, source, 998, baseline_costs=teacher["native_costs"])
    assert result["baseline_cost"] == 0 and result["delta"] == 0
    arrays["actions"][1, 0] = 0.1
    rehash(arrays)
    with pytest.raises(AuditError, match="suffix"):
        verify_assay_arrays(arrays, source, 998, baseline_costs=teacher["native_costs"])


def test_nonfinite_and_extra_fields_or_fabricated_postfall_rows_are_rejected():
    arrays = synthetic_arrays()
    with pytest.raises(AuditError, match="thirteen"):
        verify_trace_arrays(arrays | {"extra": np.array(1)})
    arrays["states"][1, 0] = np.nan
    with pytest.raises(AuditError, match="Nonfinite"):
        verify_trace_arrays(arrays)
    arrays = synthetic_arrays()
    arrays["done"][0] = 1
    with pytest.raises(AuditError, match="post-fall"):
        verify_trace_arrays(arrays)


def test_nonfinite_json_duplicate_keys_and_false_partial_accounting_fail():
    arrays = synthetic_arrays()
    arrays["metadata"] = np.array('{"horizon":2,"horizon":3}')
    with pytest.raises(AuditError, match="Duplicate"):
        verify_trace_arrays(arrays)
    arrays = synthetic_arrays()
    metadata = json.loads(arrays["metadata"].item())
    metadata.update(unknown_partial_controls=1, unknown_partial_physics_substeps=0)
    arrays["metadata"] = np.array(json.dumps(metadata))
    with pytest.raises(AuditError, match="partial substep"):
        verify_trace_arrays(arrays)


def test_one_predeclared_weight_rule_preserves_actual_float32_cast():
    deltas = np.tile(np.array([-20.0, 0.0, 20.0, 100.0, 180.0, 200.0, 40.0, 60.0]), 500)
    weights, actual = recompute_weights(deltas)
    expected_raw = np.tile(np.array([1, 1, 2, 6, 10, 10, 3, 4], dtype=np.float64), 500)
    np.testing.assert_array_equal(weights, expected_raw / np.mean(expected_raw))
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, weights.astype(np.float32))
    with pytest.raises(AuditError, match="delta roster"):
        recompute_weights(deltas[:100])


def test_shuffle_preserves_every_declared_stratum_rank_group_multiset():
    weights = np.arange(4000, dtype=np.float32)
    errors = np.arange(4000, dtype=np.float64) % 11
    shuffled, indices = recompute_shuffled_weights(weights, errors, 2800000000)
    repeated, repeated_indices = recompute_shuffled_weights(weights, errors, 2800000000)
    np.testing.assert_array_equal(shuffled, repeated)
    np.testing.assert_array_equal(indices, repeated_indices)
    np.testing.assert_array_equal(np.sort(indices), np.arange(4000))
    bounds = [(100 * i, 100 * (i + 1)) for i in range(9)] + [
        (990, 1000),
        (975, 990),
        (950, 975),
        (900, 950),
    ]
    for episode in range(4):
        for first, last in bounds:
            rows = np.arange(1000 * episode + first, 1000 * episode + last)
            ranked = rows[np.lexsort((rows, errors[rows]))]
            for group in np.array_split(ranked, 4):
                np.testing.assert_array_equal(np.sort(shuffled[group]), np.sort(weights[group]))


def write_stream(tmp_path, *, failed=False, incomplete=False):
    evidence = tmp_path / "evidence"
    meter = CompactEvidenceMeter(evidence, max_controls=10, max_neural_rows=10, max_bytes=2**20)
    trace = C1TraceRecorder(meter, tmp_path / "spool")
    arrays = synthetic_arrays(2)

    def context(i):
        return dict(
            state=arrays["states"][i],
            warmstart=arrays["warmstarts"][i],
            integration=arrays["integrations"][i],
        )

    trace.begin("synthetic", 2, context(0), arrays["observations"][0])
    trace.intent(np.zeros(6), snapshot=context(0), observation=arrays["observations"][0])
    if failed:
        trace.fail(RuntimeError("stopped"))
    elif incomplete:
        trace._release_mapping()
    else:
        trace.completed(
            context(1), arrays["observations"][1], cost=0.0, velocity=1.5, fall=0.0, done=False
        )
        trace.intent(np.zeros(6), snapshot=context(1), observation=arrays["observations"][1])
        trace.completed(
            context(2), arrays["observations"][2], cost=0.0, velocity=1.5, fall=0.0, done=False
        )
        trace.close(meta={"kind": "baseline"})
    meter.close()
    return evidence


def test_structural_audit_joins_every_paid_action_and_has_no_scientific_pass(tmp_path):
    evidence = write_stream(tmp_path)
    receipt = audit_saved_stream(evidence, tmp_path / "audit", task=TASK)
    assert receipt["structural_passed"] and receipt["complete_journal"]
    assert not receipt["full_scientific_audit_passed"]
    assert receipt["returned_controls"] == receipt["saved_controls"] == 2
    assert receipt["array_checks"] == 13
    assert receipt["new_neural_rows"] == receipt["new_physics_controls"] == 0


def test_graceful_failure_preserves_zero_returns_as_missing_and_counts_unknown(tmp_path):
    evidence = write_stream(tmp_path, failed=True)
    receipt = audit_saved_stream(evidence, tmp_path / "audit")
    assert receipt["structural_passed"]
    assert receipt["normally_closed_traces"] == 0
    assert receipt["failed_prefix_trace_groups"] == 1
    assert receipt["saved_controls"] == 0
    assert receipt["unknown_partial_controls"] == 1
    assert receipt["unknown_partial_physics_substeps"] == 4


def test_unclosed_journal_is_structurally_readable_but_explicitly_incomplete(tmp_path):
    evidence = write_stream(tmp_path, incomplete=True)
    receipt = audit_saved_stream(evidence, tmp_path / "audit")
    assert receipt["structural_passed"] and not receipt["complete_journal"]
    assert receipt["pending_trace_alias"] == "synthetic"
    assert receipt["unknown_partial_controls"] == 1
    assert receipt["saved_closed_trace_groups"] == 0
    spool = verify_active_spool(tmp_path / "spool")
    assert spool["committed_controls"] == 0
    assert not spool["normally_closed"]
    assert not spool["journal_ownership_verified"]


def test_independent_active_reader_checks_pending_prediction_without_promoting_native_row(tmp_path):
    meter = CompactEvidenceMeter(
        tmp_path / "evidence", max_controls=10, max_neural_rows=10, max_bytes=2**20
    )
    trace = C1TraceRecorder(meter, tmp_path / "spool")
    arrays = synthetic_arrays()
    initial = dict(
        state=arrays["states"][0],
        warmstart=arrays["warmstarts"][0],
        integration=arrays["integrations"][0],
    )
    trace.begin("pending_prediction", 2, initial, arrays["observations"][0])
    trace.prediction(np.zeros(6), actor_alias="parent_0")
    receipt = verify_active_spool(trace.directory)
    assert receipt["committed_controls"] == 0
    assert receipt["pending_prediction"]["actor"] == "parent_0"
    trace._rows[1]["actions"][0] = 0.2
    trace._sync()
    with pytest.raises(AuditError, match="prediction"):
        verify_active_spool(trace.directory)
    trace._rows[1]["actions"][0] = 0
    trace.fail("stop")
    meter.close()


def test_corrupt_payload_fails_without_repairing_original(tmp_path):
    evidence = write_stream(tmp_path)
    path = evidence / "arrays.bin"
    data = path.read_bytes()
    path.write_bytes(b"x" + data[1:])
    before = path.read_bytes()
    receipt = audit_saved_stream(evidence, tmp_path / "audit")
    assert not receipt["structural_passed"] and receipt["error_type"] == "AuditError"
    assert path.read_bytes() == before


def test_audit_storage_and_read_budget_fail_create_only(tmp_path):
    evidence = write_stream(tmp_path)
    receipt = audit_saved_stream(evidence, tmp_path / "audit", max_read_bytes=1)
    assert not receipt["structural_passed"]
    with pytest.raises(FileExistsError):
        audit_saved_stream(evidence, tmp_path / "audit")
    with pytest.raises(AuditError, match="outside"):
        audit_saved_stream(evidence, evidence / "bad")


def test_native_arithmetic_separates_terminal_and_absorbing_costs():
    arrays = synthetic_arrays(1, horizon=10, done=True)
    verify_native_measurements(arrays, TASK)
    arrays["native_costs"][0] += 20
    with pytest.raises(AuditError, match="native cost"):
        verify_native_measurements(arrays, TASK)


def test_audit_import_has_no_simulator_or_policy_modules():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.gps.mppi_gps.audits.student_consequence; "
            "assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules; "
            "assert 'src.gps.mppi_gps.consequence_storage' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_confirmation_requires_same_two_streams_for_both_contrasts():
    panels = {}
    for panel, qualified in (("primary", {0, 1}), ("confirmation", {1, 2})):
        panels[panel] = {}
        for stream in range(3):
            panels[panel][str(stream)] = {
                arm: dict(
                    episodes=16,
                    successes=(
                        4 if arm in {"uniform", "cost_effect"} and stream in qualified else 2
                    ),
                    survived=16,
                )
                for arm in ("parent", "uniform", "cost_effect", "shuffled")
            }
            # Give the treatment a separate two-success gain over uniform.
            panels[panel][str(stream)]["cost_effect"]["successes"] += (
                2 if stream in qualified else 0
            )
    result = _independent_gates(panels)
    assert result["primary"]["passed"] and result["confirmation"]["passed"]
    assert not result["confirmed"] and result["confirmed_streams"] == ["1"]
    assert not result["uniform_parent_confirmed"]


def test_returned_unpersisted_work_is_not_double_counted_unknown():
    work = _Work()
    work.event(dict(event="work_requested", operation=1, category="neural_rows", amount=1))
    work.event(
        dict(event="work_error", operation=1, category="neural_rows", amount=1, returned=True)
    )
    assert work.completed["neural_rows"] == 1
    assert work.unpersisted["neural_rows"] == 1
    assert work.unknown["neural_rows"] == 0
    work.closed()


def test_source_read_budget_is_shared_before_any_small_file_read(tmp_path, monkeypatch):
    import time

    from src.gps.mppi_gps.audits import student_consequence as audit

    path = tmp_path / "four"
    path.write_bytes(b"1234")
    inputs = _Inputs(tmp_path, time.monotonic() + 10)
    inputs.archive = SimpleNamespace(read_bytes=2, max_read_bytes=5)
    inputs.checkpoint_read_bytes = 1
    monkeypatch.setattr(audit, "MAX_READ_BYTES", 5)
    with pytest.raises(AuditError, match="read reservation"):
        inputs.read("four")
    assert inputs.read_bytes == 0


def test_accounting_rejects_matching_worker_and_journal_that_disagree_with_geometry(tmp_path):
    evidence = tmp_path / "worker/evidence"
    evidence.mkdir(parents=True)
    for name in ("arrays.bin", "arrays.index.jsonl", "ledger.jsonl"):
        (evidence / name).write_bytes(b"")
    # A complete fixed roster can fall early; its realized eval/assay controls
    # determine native work, while the nine fits still require all18k updates.
    native, evaluation = 12000 + 12000 + 432, 432
    expected = dict(
        native_controls=native,
        physics_substeps=4 * native,
        restores=12012,
        resets=432,
        environment_constructions=1,
        external_mj_forward_calls=12876,
        snapshot_reads=2 * native + 12012 + 432,
        shuffle_calls=624,
        shuffle_entries=12000,
        neural_rows=4608000 + 90000 + evaluation,
        prediction_calls=18000 + 90 + evaluation,
        backward_rows=4608000,
        optimizer_steps=18000,
        checkpoint_loads=12,
        policy_constructions=21,
        endpoint_saves=9,
        expected_index_draws=1536000,
        actual_index_draws=4608000,
    )
    work = _Work()
    work.requested, work.completed = Counter(expected), Counter(expected)
    saved = dict(
        requested=dict(expected),
        completed=dict(expected),
        unknown_partial={},
        returned_unpersisted={},
        storage=dict(storage_bytes=0),
    )
    checks = SimpleNamespace(
        work=work,
        result=dict(status="COMPLETED_FIXED_C1_ROSTER", costs=saved, worker_wall_seconds=1),
        baselines={str(i): dict(controls=1000) for i in range(12)},
        assay_rows=[dict(controls=1)] * 12000,
        evaluation_rows=[dict(controls=1)] * 432,
        closed_prerequisite_negative=False,
        run=tmp_path,
        inputs=SimpleNamespace(read_bytes=30, checkpoint_read_bytes=10),
    )
    assert _reconcile_accounting(checks, dict(source_read_bytes=40))["source_read_bytes"] == 80
    work.completed["optimizer_steps"] -= 1
    work.requested["optimizer_steps"] -= 1
    saved["completed"]["optimizer_steps"] -= 1
    saved["requested"]["optimizer_steps"] -= 1
    with pytest.raises(AuditError, match="geometry"):
        _reconcile_accounting(checks, dict(source_read_bytes=40))


def prerequisite_negative_run(root):
    """Twelve fake saved sources plus one real storage mismatch prefix; no NN/physics."""
    root.mkdir()
    prepared = root / "prepared"
    prepared.mkdir()

    def save_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))

    def binding(path, base=prepared):
        return dict(
            file=str(path.relative_to(base)),
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def npz(relative, arrays):
        path = prepared / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        return binding(path)

    def cbyte(value):
        return dict(sha256=hashlib.sha256(value.tobytes()).hexdigest())

    teacher = synthetic_arrays(1000)
    episode = source_from(teacher)
    episode.update(
        squared_disagreement=np.zeros(1000, dtype=np.float64),
        integration_spec=np.array(1, dtype=np.int64),
        dimensions=np.array([9, 9, 6], dtype=np.int64),
    )
    catalog = dict(status="complete", episodes={}, streams={}, parents={}, integration_spec=1)
    sources = dict(
        source_episodes_in_fixed_training_order=[],
        incoming_parents={},
        normalization_and_historical_inputs=dict(config_inputs={}),
        reused_incoming_historical_metric_events=[],
    )
    for stream in range(3):
        parent = prepared / f"parents/s{stream}.pt"
        parent.parent.mkdir(exist_ok=True)
        parent.write_bytes(f"synthetic saved parent {stream}".encode())
        original = dict(
            sha256=binding(parent)["sha256"],
            bytes=parent.stat().st_size,
            actor_hash=f"{stream + 1:064x}",
            optimizer_sha256=f"{stream + 4:064x}",
        )
        catalog["parents"][str(stream)] = binding(parent) | dict(
            actor_hash=original["actor_hash"],
            incoming_optimizer_sha256=original["optimizer_sha256"],
            original_checkpoint_record=original,
        )
        sources["incoming_parents"][str(stream)] = dict(checkpoint_record=original)
        tags = []
        for case in range(4):
            tag = f"teacher_s{stream}_independent_{case:02d}"
            tags.append(tag)
            raw = {
                key: cbyte(episode[key][:-1] if key == "observations" else episode[key])
                for key in (
                    "observations",
                    "states",
                    "warmstarts",
                    "integrations",
                    "actions",
                    "costs",
                    "velocities",
                )
            }
            catalog["episodes"][tag] = npz(f"episodes/{tag}.npz", episode) | dict(
                stream=stream,
                case=case,
                seed=100 + case,
                terminal_binding=dict(
                    last_live_result=dict(
                        arrays=dict(observation=cbyte(episode["observations"][-1]))
                    ),
                    last_live_inputs=dict(
                        arrays=dict(
                            spec=cbyte(episode["integration_spec"]),
                            dimensions=cbyte(episode["dimensions"]),
                        )
                    ),
                ),
            )
            sources["source_episodes_in_fixed_training_order"].append(
                dict(
                    tag=tag,
                    exact_episode_complete_event=dict(raw=dict(arrays=raw)),
                    exact_teacher_agreement_complete_event=dict(
                        actor_hash=original["actor_hash"],
                        predictions=dict(
                            reference=dict(
                                arrays=dict(predictions=cbyte(episode["policy_actions"]))
                            ),
                            key="predictions",
                        ),
                    ),
                )
            )
        corpus = {
            key: np.concatenate([episode[key][:-1] if key == "observations" else episode[key]] * 4)
            for key in ("observations", "actions", "policy_actions", "squared_disagreement")
        }
        corpus["observations"], corpus["actions"] = (
            corpus["observations"].astype(np.float32),
            corpus["actions"].astype(np.float32),
        )
        catalog["streams"][str(stream)] = npz(f"streams/s{stream}_corpus.npz", corpus) | dict(
            episode_tags=tags
        )
    historical = dict(
        observations=np.zeros((4000, 17), dtype=np.float32),
        actions=np.zeros((4000, 6), dtype=np.float32),
    )
    for name, arrays in dict(
        train=historical,
        validation={key: value[:2000] for key, value in historical.items()},
        normalization=dict(
            mean=np.zeros(17, dtype=np.float32), std=np.full(17, 0.05, dtype=np.float32)
        ),
    ).items():
        sources["normalization_and_historical_inputs"]["config_inputs"][name] = npz(
            f"inputs/{name}.npz", arrays
        )
    save_json(prepared / "inputs/task.json", dict(task=TASK))
    sources["normalization_and_historical_inputs"]["config_inputs"]["task"] = binding(
        prepared / "inputs/task.json"
    )
    save_json(prepared / "metadata/source_bindings.json", sources)
    source_sha = binding(prepared / "metadata/source_bindings.json")["sha256"]
    save_json(prepared / "catalog.json", catalog)
    save_json(
        prepared / "prepared_input_manifest.json",
        dict(files=[binding(path) for path in sorted(prepared.rglob("*")) if path.is_file()]),
    )
    save_json(
        root / "prerequisites.json",
        dict(
            passed=True,
            **{
                name: dict(passed=True)
                for name in ("seed", "capacity", "arithmetic", "continuation")
            },
        ),
    )
    recipe = dict(
        study="C1.0",
        user_authorized=True,
        automatic_followup=False,
        source_bindings_sha256=source_sha,
        task=TASK,
    )
    recipe_path = root / "source/configs/studies/student_consequence_c1_0.json"
    save_json(recipe_path, recipe)
    config = recipe | dict(
        recipe_sha256=binding(recipe_path, root)["sha256"],
        prepared_manifest_sha256=binding(prepared / "prepared_input_manifest.json")["sha256"],
        prerequisites_sha256=binding(root / "prerequisites.json", root)["sha256"],
    )
    save_json(root / "config.json", config)
    save_json(
        root / "manifest.json",
        dict(
            config_sha256=binding(root / "config.json", root)["sha256"],
            files=[
                dict(
                    file="configs/studies/student_consequence_c1_0.json",
                    sha256=config["recipe_sha256"],
                )
            ],
        ),
    )
    meter = CompactEvidenceMeter(
        root / "worker/evidence", max_controls=1000, max_neural_rows=0, max_bytes=8 * 1024**2
    )
    meter.record("environment_requested", mjdata_objects=17)
    meter.record("environment_complete", integration_size=91, mjdata_objects=17)
    meter.record("work_requested", category="restores", amount=1, operation=1)
    meter.record("work_complete", category="restores", amount=1, operation=1)
    trace = C1TraceRecorder(meter, root / "worker/active")
    tag = next(iter(catalog["episodes"]))

    def context(index):
        return dict(
            state=episode["states"][index].copy(),
            warmstart=episode["warmstarts"][index].copy(),
            integration=episode["integrations"][index].copy(),
        )

    trace.begin("baseline_" + tag, 1000, context(0), episode["observations"][0])
    trace.intent(np.zeros(6), snapshot=context(0), observation=episode["observations"][0])
    changed = context(1)
    changed["warmstart"][0] = 1e-8
    trace.completed(
        changed, episode["observations"][1], cost=0.0, velocity=1.5, fall=0.0, done=False
    )
    trace.fail(
        ValueError("C1 replay mismatch"),
        meta=dict(
            kind="baseline",
            source=tag,
            t=0,
            actor=None,
            seed=None,
            integration_spec=1,
            dimensions=[9, 9, 6],
        ),
    )
    meter.close()
    work = dict(
        native_controls=1,
        physics_substeps=4,
        restores=1,
        environment_constructions=1,
        external_mj_forward_calls=1,
        snapshot_reads=3,
    )
    result = dict(
        schema="walker-student-consequence-c1.0-result-v1",
        status="STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE",
        error=dict(type="ValueError", message="C1 replay mismatch"),
        costs=dict(
            requested=work,
            completed=work,
            unknown_partial={},
            returned_unpersisted={},
            storage=meter.totals(),
        ),
        worker_wall_seconds=1.0,
        parent_refits=0,
        teacher_reacquisitions=0,
        followup_launched=False,
    )
    save_json(root / "worker_result.json", result)
    return root


def test_complete_prerequisite_negative_is_audited_without_claiming_learning(tmp_path):
    run = prerequisite_negative_run(tmp_path / "run")
    receipt = audit_run(run, tmp_path / "audit")
    assert receipt["passed"], receipt.get("error")
    assert receipt["scientific_status"] == "prerequisite_negative"
    assert receipt["evidence_integrity_passed"] and receipt["full_scientific_audit_passed"]
    assert not receipt["scientific_contrast_completed"] and not receipt["learning_measured"]
    assert receipt["weighting_confirmed"] is None
    assert (
        receipt["prediction_rows"]
        == receipt["prediction_calls"]
        == receipt["checkpoint_loads"]
        == 0
    )
    assert receipt["accounting"]["independent_completed"]["native_controls"] == 1
    assert (
        receipt["worker_result_sha256"]
        == hashlib.sha256((run / "worker_result.json").read_bytes()).hexdigest()
    )
    assert (
        receipt["config_sha256"] == hashlib.sha256((run / "config.json").read_bytes()).hexdigest()
    )
    assert receipt["source_read_bytes_total"] < 64 * 1024**3
