"""Seed proposals must never leak their new planner memory into continuations."""

import copy
import json

import numpy as np
import pytest

from scripts.diagnostics import walker_action_controls as controls
from scripts.diagnostics import walker_gate_counterfactual as branch
from src.mppi.mppi import MPPI
from src.utils.trace_replay import decision_snapshot
from test_walker_gate_counterfactual import shadow_bundle


@pytest.mark.parametrize("step, iterations", [(0, 2), (3, 1)])
def test_original_pre_call_rng_recovers_saved_action_plan_and_post_snapshot(shadow_bundle, step, iterations):
    start = branch.load_decision(shadow_bundle, "matched", step)
    action, report = controls.generate_alternative(start, None)
    np.testing.assert_array_equal(action, start["baseline_action"])
    np.testing.assert_array_equal(report["optimized_plan"], start["trace"]["plans"][step])
    assert report["discarded_post_call_snapshot"] == branch._json_snapshot(start["snapshot"])
    assert report["optimization_iterations"] == iterations
    assert report["original_rng_preserved"]


def test_new_seed_changes_only_rng_of_original_pre_call_snapshot(shadow_bundle, monkeypatch):
    start = branch.load_decision(shadow_bundle, "matched", 0)
    snapshots = []
    original = MPPI.plan_step

    def observe(self, *args, **kwargs):
        snapshots.append(self.snapshot())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MPPI, "plan_step", observe)
    _, report = controls.generate_alternative(start, 31000)
    pre = decision_snapshot(start["trace"], 0)
    assert len(snapshots) == 1
    np.testing.assert_array_equal(snapshots[0]["U"], pre["U"])
    assert snapshots[0]["initial_plan"] == pre["initial_plan"] is True
    assert snapshots[0]["rng_state"] == np.random.default_rng(31000).bit_generator.state
    assert snapshots[0]["rng_state"] != pre["rng_state"]
    assert report["optimization_iterations"] == 2


def test_proposals_reproduce_without_mutating_input_or_global_rng(shadow_bundle):
    start = branch.load_decision(shadow_bundle, "matched", 3)
    frozen = copy.deepcopy(start)
    global_state = copy.deepcopy(np.random.get_state())
    a, a_report = controls.generate_alternative(start, 31000)
    controls.generate_alternative(start, 31001)
    b, b_report = controls.generate_alternative(start, 31000)
    np.testing.assert_array_equal(a, b)
    assert a_report == b_report
    for key in ("state", "warmstart", "baseline_action", "candidate_action"):
        np.testing.assert_array_equal(start[key], frozen[key])
    assert branch._json_snapshot(start["snapshot"]) == branch._json_snapshot(frozen["snapshot"])
    for key in start["trace"]:
        np.testing.assert_array_equal(start["trace"][key], frozen["trace"][key])
    for left, right in zip(np.random.get_state(), global_state):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "31000"])
def test_invalid_proposal_seed_rejected_before_simulation(seed):
    with pytest.raises(ValueError, match="seed"):
        controls.generate_alternative({}, seed)


@pytest.mark.parametrize("first_fall, expected", [(None, None), (2, 233.0)])
def test_window_cost_only_pads_after_a_fall(first_fall, expected):
    states = np.zeros((3, 19))
    states[1:, 10] = [1.0, 2.0]
    report = controls.window_metrics({"costs": np.array([1., 2.]), "states": states},
                                     {"first_fall_step": first_fall},
                                     {"unhealthy_cost_weight": 5., "target_velocity": 1.5})
    assert report["fixed_horizon_task_cost"] == expected
    assert report["velocity_mae"] == .5 and report["observed_steps"] == 2
    assert report["complete_window_or_absorbing_tail"] == (first_fall is not None)


def test_window_cost_clips_longer_trace_and_later_fall():
    states = np.zeros((61, 19))
    report = controls.window_metrics({"costs": np.ones(60), "states": states},
                                     {"first_fall_step": 60},
                                     {"unhealthy_cost_weight": 5., "target_velocity": 1.5})
    assert report["fixed_horizon_task_cost"] == 48.
    assert not report["fell"] and report["observed_steps"] == 48


def test_archived_controls_share_original_continuation_and_hash_every_artifact(shadow_bundle, tmp_path, monkeypatch):
    start = branch.load_decision(shadow_bundle, "matched", 3)
    snapshots = []
    original = controls.run_branch

    def observe(task, cfg, state, warmstart, snapshot, action, steps):
        snapshots.append(branch._json_snapshot(snapshot))
        return original(task, cfg, state, warmstart, snapshot, action, steps)

    monkeypatch.setattr(controls, "run_branch", observe)
    output = tmp_path / "controls"
    controls.main(shadow_bundle, "matched", 3, output, proposal_seeds=(31000, 31001), steps=6)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["original_rng_recovery_exact"]
    assert report["identity_comparison"]["exact"]
    assert report["raw_reference_comparison"]["exact"]
    assert len(snapshots) == 4 and all(snapshot == branch._json_snapshot(start["snapshot"]) for snapshot in snapshots)
    assert list(report["branches"]) == ["raw", "identity", "seed_31000", "seed_31001"]
    for label, proposal in report["proposals"].items():
        summary = report["branches"][label]
        assert summary["start_snapshot"] == branch._json_snapshot(start["snapshot"])
        assert summary["start_snapshot"] != proposal["discarded_post_call_snapshot"]
        assert summary["first_action"] == proposal["action"]
        assert summary["action_l2_difference"] == proposal["action_l2_difference"]
    with np.load(output / "raw.npz") as raw, np.load(output / "identity.npz") as identity:
        for key in raw.files:
            np.testing.assert_array_equal(raw[key], identity[key])
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert all(branch.sha256(output / relative) == value for relative, value in hashes.items())
    assert {"inputs/decision.npz", "inputs/task.json", "original_rng_recovery.json", "proposals.json",
            "source/scripts/diagnostics/walker_action_controls.py"}.issubset(hashes)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete" and len(status["completed_branches"]) == 4
    with pytest.raises(FileExistsError):
        controls.main(shadow_bundle, "matched", 3, output, proposal_seeds=(31000,), steps=6)


@pytest.mark.parametrize("kwargs", [
    {"steps": 0}, {"steps": True}, {"proposal_seeds": ()},
    {"proposal_seeds": (31000, 31000)}, {"proposal_seeds": (True,)},
    {"proposal_seeds": (None,)},
])
def test_invalid_run_arguments_rejected_before_creating_output(tmp_path, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        controls.main(tmp_path / "absent", "matched", 3, output, **kwargs)
    assert not output.exists()


def test_partial_failure_records_completed_branches_and_hashes(shadow_bundle, tmp_path, monkeypatch):
    original = controls.generate_alternative

    def fail_on_seed(start, seed):
        if seed == 31000:
            raise RuntimeError("injected proposal failure")
        return original(start, seed)

    monkeypatch.setattr(controls, "generate_alternative", fail_on_seed)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected proposal"):
        controls.main(shadow_bundle, "matched", 3, output, proposal_seeds=(31000,), steps=6)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and set(report["branches"]) == {"raw", "identity"}
    assert report["identity_comparison"]["exact"]
    assert "injected proposal failure" in report["error"]
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert all(branch.sha256(output / relative) == value for relative, value in hashes.items())


def test_recovery_mismatch_stops_before_executing_branches(shadow_bundle, tmp_path, monkeypatch):
    original = controls.generate_alternative

    def mismatch(start, seed):
        action, report = original(start, seed)
        action[0] += .001
        return action, report

    monkeypatch.setattr(controls, "generate_alternative", mismatch)
    output = tmp_path / "mismatch"
    with pytest.raises(RuntimeError, match="Original pre-call RNG"):
        controls.main(shadow_bundle, "matched", 3, output, proposal_seeds=(31000,), steps=6)
    report = json.loads((output / "report.json").read_text())
    assert not report["original_rng_recovery_exact"] and not report["branches"]


@pytest.mark.parametrize("location", ["same", "nested", "symlink"])
def test_output_cannot_modify_audited_shadow_bundle(tmp_path, monkeypatch, location):
    source = tmp_path / "shadow"
    source.mkdir()
    if location == "same":
        output = source
    elif location == "nested":
        output = source / "new" / "controls"
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        output = alias / "controls"
    monkeypatch.setattr(controls, "load_decision", lambda *args: pytest.fail("Reject output before loading inputs"))
    with pytest.raises(ValueError, match="outside the audited shadow"):
        controls.main(source, "matched", 3, output, proposal_seeds=(31000,), steps=6)
    assert not list(source.iterdir())
