"""Saved-array audit fixtures; no policy evaluation, training or simulator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.audits.compatibility import (
    CompatibilityAudit,
    audit_scores,
    recompute_agreement,
    recompute_groups,
    recompute_neighborhood_summary,
    saved_collision_context,
    saved_native_outcome,
    stage_ledger,
    verify_geometry,
)

SETTINGS = dict(early_controls=2, bin_controls=2, saturation_threshold=0.95)


def test_known_action_denominator_early_bins_and_saturation():
    prediction = np.zeros((3, 6))
    target = np.zeros((3, 6))
    target[0, 0] = 1
    target[2, 1] = -1
    result = recompute_agreement(prediction, target, SETTINGS)
    assert result["mse"] == 2 / 18
    assert result["early"]["mse"] == 1 / 12
    assert [row["rows"] for row in result["bins"]] == [2, 1]
    assert result["per_actuator_mse"][:2] == [1 / 3, 1 / 3]
    assert result["actions"]["saturation_fraction"] == 2 / 18
    with pytest.raises(ValueError):
        recompute_agreement(prediction, target[:-1], SETTINGS)
    with pytest.raises(ValueError):
        recompute_agreement(prediction, target + 2, SETTINGS)


def test_reset_cluster_bootstrap_and_equal_vs_sample_means():
    cfg = dict(policy_order=["u0", "u1", "u2", "clone"], bootstrap_seed=17, bootstrap_draws=30)
    rows = []
    for score, count, ancestry in ((0.0, 1, "same"), (1.0, 3, "same"), (0.25, 2, "other")):
        rows.append(
            dict(
                trajectory=dict(group="g", steps=count, ancestry={"reset_cluster": ancestry}),
                policies={name: dict(mse=score) for name in cfg["policy_order"]},
            )
        )
    result = recompute_groups(rows, cfg)["g"]
    assert result["ancestries"] == 2 and result["trajectories"] == 3
    assert result["equal_trajectory_mean"] == [1.25 / 3] * 4
    assert result["sample_weighted_mean"] == [3.5 / 6] * 4
    assert result["contrasts"]["clone_minus_mean_untrained"]["interval"] == [0.0, 0.0]


def test_geometry_recomputed_and_action_changes_cannot_change_neighbors(tmp_path):
    # Production prepares geometry; the independent checker recomputes every
    # choice. This cross-check also exercises ties and a real exact collision.
    from src.gps.mppi_gps.compatibility_metrics import prepare_neighborhoods

    observations = np.zeros((12, 17))
    observations[:, 0] = [0, 0.1, 0.2, 0.3, 0.005, 0.105, 0.205, 0.305, 0, 0.15, 0.25, 0.35]
    ids = np.repeat(["a", "b", "c"], 4)
    ancestry = np.repeat(["r0", "r0", "r1"], 4)
    steps = np.tile([0, 50, 100, 150], 3)
    normalization = dict(mean=np.zeros(17), std=np.ones(17))
    config = dict(
        anchors_per_trajectory=2,
        minimum_step_separation=50,
        distance_caliper=0.02,
        action_difference_atol=1e-6,
        distance_bins=[0, 0.1, 1, 1e9],
    )
    geometry = prepare_neighborhoods(
        observations, ids, steps, ancestry, **normalization, config=config
    )
    audit = CompatibilityAudit(tmp_path, tmp_path)
    verify_geometry(audit, observations, ids, ancestry, steps, normalization, config, geometry)
    wrong = {key: value.copy() for key, value in geometry.items()}
    wrong["within_index"][0] = 0
    with pytest.raises(AssertionError, match="within-neighbor"):
        verify_geometry(audit, observations, ids, ancestry, steps, normalization, config, wrong)
    actions = np.zeros((12, 6))
    actions[0, 0] = 1
    actions[8, 0] = -1
    states = np.zeros((12, 19))
    result = recompute_neighborhood_summary(
        geometry, actions, states, dict(planner_context_available=np.ones(12, bool))
    )
    collision = result["exact_incoming_h4_effective_input_collisions"]
    assert collision["groups"] == 1 and collision["rows"] == 2
    assert collision["details"][0]["exact_distinct_action_vectors"] == 2
    assert collision["weighted_empirical_deterministic_mse_floor"] == pytest.approx(1 / 36)
    available = np.ones(12, dtype=bool)
    available[0] = False
    rho = np.zeros(12)
    rho[8:] = 0.05
    missing = recompute_neighborhood_summary(
        geometry,
        actions,
        states,
        dict(planner_context_available=available, planner_nominal=np.zeros((12, 48, 6)), rho=rho),
    )
    detail = missing["exact_incoming_h4_effective_input_collisions"]["details"][0]
    assert detail["planner_context_available_rows"] == 1
    assert detail["planner_context_comparable"] is False
    assert detail["planner_context_constant"]["planner_nominal"] is None
    assert detail["planner_context_constant"]["rho"] is False
    rows = [dict(warmstarts=np.zeros((4, 9))), dict(), dict(warmstarts=np.zeros((4, 9)))]
    roster = [dict(steps=4, controller=dict(rho=value)) for value in (0, 0, 0.05)]
    saved_collision_context(missing, rows, roster)
    assert detail["guidance_weights"] == [0, 0.05]
    assert detail["saved_physical_context"]["warmstarts"] == dict(
        available_rows=2, total_rows=2, all_identical=True
    )
    assert detail["saved_physical_context"]["integrations"] == dict(
        available_rows=0, total_rows=2, all_identical=None
    )


def raw_matrix(path, observations, actions, predictions):
    arrays = dict(observations=observations, executed_actions=actions, predictions=predictions)
    ref = dict(file="raw.h5", group="arrays/test", tag="synthetic", arrays={})
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as stream:
        group = stream.create_group(ref["group"])
        group.attrs["tag"] = ref["tag"]
        for key, value in arrays.items():
            dataset = group.create_dataset(key, data=value)
            digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            dataset.attrs.update(numpy_dtype=value.dtype.str, encoding="array", sha256=digest)
            ref["arrays"][key] = dict(
                shape=list(value.shape), dtype=value.dtype.str, encoding="array", sha256=digest
            )
    return ref


def test_prediction_matrix_alignment_reuse_and_numeric_tamper_detection(tmp_path):
    observations = np.zeros((3, 17))
    actions = np.full((3, 6), 0.25)
    predictions = np.zeros((3, 1, 6), dtype=np.float32)
    ref = raw_matrix(tmp_path / "evidence/stage_a/raw.h5", observations, actions, predictions)
    metrics = recompute_agreement(predictions[:, 0], actions, SETTINGS)
    metrics.update(new_forward_rows=1, reused_forward_rows=2)
    record = dict(
        trajectory=dict(id="t", group="g", steps=3, ancestry="r"), policies=dict(p=metrics), raw=ref
    )
    cfg = dict(policy_order=["p"], metrics=SETTINGS)
    data = dict(t=dict(observations=observations, actions=actions))
    audit = CompatibilityAudit(tmp_path, tmp_path)
    cache = dict(p={})
    assert audit_scores(audit, tmp_path, cfg, [record], data, "stage_a", cache) == 1
    second = deepcopy(record)
    second["policies"]["p"].update(new_forward_rows=0, reused_forward_rows=3)
    assert audit_scores(audit, tmp_path, cfg, [second], data, "stage_a", cache) == 0
    tampered = deepcopy(record)
    tampered["policies"]["p"]["per_actuator_mse"][0] += 0.1
    with pytest.raises(AssertionError):
        audit_scores(audit, tmp_path, cfg, [tampered], data, "stage_a", dict(p={}))
    for handle, _, _ in audit.hdfs.values():
        handle.close()


def test_native_helper_stage_raw_alias_is_explicit_and_exact(tmp_path):
    observations, actions = np.zeros((2, 17)), np.zeros((2, 6))
    actual = tmp_path / "evidence/stage_b/raw.h5"
    ref = raw_matrix(actual, observations, actions, np.zeros((2, 1, 6)))
    audit = CompatibilityAudit(tmp_path, tmp_path)
    direct = audit.group(actual, ref)
    alias = audit.group(tmp_path / "evidence/stage_b/evidence/raw.h5", ref)
    assert all(np.array_equal(direct[key], alias[key]) for key in direct)
    assert list(audit.hdfs) == [actual.resolve()]
    for handle, _, _ in audit.hdfs.values():
        handle.close()


def test_saved_outcome_charges_terminal_once_and_absorbing_tail():
    states = np.zeros((2, 19))
    states[:, 2] = 1.25
    states[:, 10] = 1.5
    terminal = states[-1].copy()
    terminal[2] = 0.7
    result = saved_native_outcome(
        dict(states=states, terminal_state=terminal, actions=np.zeros((2, 6)))
    )
    assert result == dict(
        complete=True, success=False, survived=False, cost=5 + 20 + 5 * 998, mae=0.0
    )


def test_closed_ledger_recounts_only_registered_work_and_rejects_open_request(tmp_path):
    work = dict(
        requested_checkpoint_loads=1,
        completed_checkpoint_loads=1,
        requested_neural_rows=2,
        completed_neural_rows=2,
        measurement_requests=2,
        snapshots=0,
        wall_seconds=0.5,
    )
    rows = [
        dict(event="runtime_open"),
        dict(
            event="measurement_requested", measure_kind="checkpoint_loads", amount=1, purpose="load"
        ),
        dict(
            event="measurement_complete",
            request_id="event-000000001",
            measure_kind="checkpoint_loads",
            amount=1,
            purpose="load",
        ),
        dict(
            event="measurement_requested", measure_kind="neural_rows", amount=2, purpose="predict"
        ),
        dict(
            event="measurement_complete",
            request_id="event-000000003",
            measure_kind="neural_rows",
            amount=2,
            purpose="predict",
        ),
        dict(event="runtime_close", totals=work),
    ]
    for index, row in enumerate(rows):
        row.update(id=f"event-{index:09d}", kind=row["event"], wall_seconds=index * 0.01)
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with h5py.File(tmp_path / "raw.h5", "w"):
        pass
    audit = CompatibilityAudit(tmp_path, tmp_path)
    result = stage_ledger(audit, tmp_path, work)
    assert result["counts"]["completed_neural_rows"] == 2
    assert result["counts"]["completed_controls"] == 0
    wrong = deepcopy(work)
    wrong["completed_neural_rows"] = 3
    with pytest.raises(AssertionError, match="census"):
        stage_ledger(audit, tmp_path, wrong)
    rows.pop(4)
    for index, row in enumerate(rows):
        row["id"] = f"event-{index:09d}"
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    fresh = CompatibilityAudit(tmp_path, tmp_path)
    with pytest.raises(AssertionError, match="Closed stage"):
        stage_ledger(fresh, tmp_path, work)


def test_cli_failure_receipt_and_import_independence(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from src.gps.mppi_gps.audits.compatibility import recompute_agreement
for name in sys.modules:
    assert not name.startswith(('mujoco','src.envs','src.gps.mppi_gps.policy','src.gps.mppi_gps.experiments','src.gps.mppi_gps.compatibility_metrics','src.gps.mppi_gps.diagnostic_policies'))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=root, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    out = tmp_path / "audit"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/audits/compatibility.py",
            "--run",
            str(tmp_path / "missing"),
            "--out",
            str(out),
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["passed"] is False and receipt["new_controls"] == receipt["new_neural_rows"] == 0
    assert (out / "findings.json").is_file()
