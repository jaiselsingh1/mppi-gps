"""A single intervention must fork AFTER, not repeat, the original MPPI call."""

import copy
import json
import platform
import shutil
from dataclasses import asdict
from importlib.metadata import version

import numpy as np
import pytest

from scripts.diagnostics import walker_gate_counterfactual as branch
from scripts.runners import run_walker2d as baseline
from scripts.runners import run_walker_shadow as shadow
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import load_episode


def _hash_bundle(directory):
    hashes = {str(path.relative_to(directory)): branch.sha256(path)
              for path in directory.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"}
    branch.write_json(directory / "artifact_hashes.json", hashes)


@pytest.fixture(scope="module")
def shadow_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("counterfactual_shadow")
    inputs = directory / "inputs"
    inputs.mkdir()
    task = json.loads((branch._ROOT / "configs/walker2d_task.json").read_text())
    task["nthread"] = 1
    cfg = MPPIConfig(K=8, H=6, lam=3.0, noise_sigma=0.3, initial_iterations=2)
    branch.write_json(inputs / "task.json", task)
    branch.write_json(inputs / "mppi.json", asdict(cfg))
    baseline.main(config=inputs / "mppi.json", task_config=inputs / "task.json",
                  episodes=1, steps=12, seed=43, planner_seed=71,
                  traces_output=directory / "reference.npz", log_every=0)
    reference = load_episode(directory / "reference.npz", 0)

    class RecordedPlanPolicy:
        calls = 0

        def act(self, obs, deterministic=True):
            step, offset = divmod(self.calls, cfg.H)
            self.calls += 1
            return reference["plans"][step, offset].copy()

    trace, candidates, report = shadow.run_shadow(reference, task, cfg,
                                                  {"matched": RecordedPlanPolicy()}, steps=12)
    assert report["reference_trace_exact"]
    assert all(row["accepted"] for row in report["decisions"]["matched"])
    np.savez_compressed(directory / "traces.npz", **trace)
    (directory / "matched").mkdir()
    np.savez_compressed(directory / "matched/candidates.npz", **candidates["matched"])
    branch.write_json(directory / "matched/decisions.json", {"decisions": report["decisions"]["matched"]})
    branch.write_json(directory / "report.json", {key: value for key, value in report.items() if key != "decisions"})
    sources = branch.FROZEN_SOURCES + ("scripts/runners/run_walker_shadow.py",)
    source_hashes = {relative: branch.sha256(branch._ROOT / relative) for relative in sources}
    for relative in sources:
        destination = directory / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(branch._ROOT / relative, destination)
    manifest = {"selection": {"checkpoints": {"matched": {}}}, "task": task,
                "mppi_config": asdict(cfg), "task_sha256": branch.sha256(inputs / "task.json"),
                "config_sha256": branch.sha256(inputs / "mppi.json"),
                "provenance": {"python": platform.python_version(),
                               "packages": {name: version(name) for name in ("mujoco", "numpy", "scipy")},
                               "source_sha256": source_hashes}}
    branch.write_json(directory / "manifest.json", manifest)
    _hash_bundle(directory)
    return directory


def _run(start, action=None, steps=6):
    return branch.run_branch(start["task"], start["cfg"], start["state"], start["warmstart"],
                             start["snapshot"], start["baseline_action"] if action is None else action, steps)


def _assert_trace_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_identical_initial_actions_produce_bit_identical_continuations(shadow_bundle):
    start = branch.load_decision(shadow_bundle, "matched", 3)
    first, first_report = _run(start)
    second, second_report = _run(start, start["candidate_action"])
    _assert_trace_equal(first, second)
    for key in first_report:
        if key != "elapsed_seconds":
            assert first_report[key] == second_report[key]


@pytest.mark.parametrize("decision_step, steps", [(0, 6), (3, 6), (10, 5)])
def test_raw_branch_matches_available_saved_shadow_prefix(shadow_bundle, decision_step, steps):
    start = branch.load_decision(shadow_bundle, "matched", decision_step)
    trace, _ = _run(start, steps=steps)
    comparison = branch.compare_reference(trace, start["trace"], decision_step, steps)
    assert comparison["exact"]
    assert comparison["checked_steps"] == min(steps, 12 - decision_step)


def test_first_step_uses_post_call_snapshot_without_another_plan_call(shadow_bundle, monkeypatch):
    start = branch.load_decision(shadow_bundle, "matched", 0)
    calls = []
    original = MPPI.plan_step

    def track_call(self, *args, **kwargs):
        calls.append(self.snapshot())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MPPI, "plan_step", track_call)
    trace, report = _run(start, steps=4)
    assert len(calls) == report["planner_calls"] == 3
    np.testing.assert_array_equal(trace["planned_here"], [False, True, True, True])
    np.testing.assert_array_equal(calls[0]["U"], start["snapshot"]["U"])
    assert calls[0]["rng_state"] == start["snapshot"]["rng_state"]
    assert report["start_snapshot"] == branch._json_snapshot(start["snapshot"])
    np.testing.assert_array_equal(trace["post_plan_nominals"][0], start["snapshot"]["U"])
    assert json.loads(str(trace["post_planner_rng_states"][0])) == start["snapshot"]["rng_state"]


def test_branches_are_independent_reproducible_and_do_not_mutate_input(shadow_bundle):
    start = branch.load_decision(shadow_bundle, "matched", 3)
    initial_state, initial_warmstart = start["state"].copy(), start["warmstart"].copy()
    initial_snapshot = copy.deepcopy(start["snapshot"])
    raw_before, _ = _run(start)
    changed, _ = _run(start, np.clip(start["baseline_action"] + .3, -1, 1))
    raw_after, _ = _run(start)
    _assert_trace_equal(raw_before, raw_after)
    assert not np.array_equal(raw_before["states"][1], changed["states"][1])
    np.testing.assert_array_equal(start["state"], initial_state)
    np.testing.assert_array_equal(start["warmstart"], initial_warmstart)
    np.testing.assert_array_equal(start["snapshot"]["U"], initial_snapshot["U"])
    assert start["snapshot"]["rng_state"] == initial_snapshot["rng_state"]


def test_fall_stops_branch_and_fixed_horizon_cost_adds_only_absorbing_suffix(shadow_bundle):
    start = branch.load_decision(shadow_bundle, "matched", 0)
    start["state"][3], start["state"][12] = .999, 2.0
    trace, report = _run(start, np.zeros(6), steps=6)
    assert report["completed_steps"] == report["first_fall_step"] == 1
    assert not report["completed_without_fall"] and report["planner_calls"] == 0
    assert len(trace["states"]) == 2
    assert report["fixed_horizon_task_cost"] == report["task_cost"] + 5 * start["task"]["unhealthy_cost_weight"]
    assert report["start_snapshot"] == report["end_snapshot"]


def test_saves_source_hashes_both_traces_and_exact_reference_report(shadow_bundle, tmp_path):
    output = tmp_path / "branches"
    branch.main(shadow_bundle, "matched", 3, output, steps=6)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete" and report["single_intervention_only"]
    assert report["raw_reference_comparison"]["exact"]
    assert report["branches"]["raw"]["completed_without_fall"]
    assert report["branches"]["raw"]["fixed_horizon_task_cost"] == report["branches"]["raw"]["task_cost"]
    with np.load(output / "raw.npz") as raw, np.load(output / "intervention.npz") as intervention:
        _assert_trace_equal(dict(raw), dict(intervention))
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert all(branch.sha256(output / relative) == expected for relative, expected in hashes.items())
    assert "source/scripts/diagnostics/walker_gate_counterfactual.py" in hashes
    with pytest.raises(FileExistsError):
        branch.main(shadow_bundle, "matched", 3, output, steps=6)


@pytest.mark.parametrize("problem", ["not_accepted", "bad_rule", "hash", "candidate_bounds", "snapshot",
                                     "task_hash", "config_value", "current_source", "source_archive", "status"])
def test_rejects_invalid_decisions_or_bundle_provenance(shadow_bundle, tmp_path, problem):
    directory = tmp_path / "shadow"
    shutil.copytree(shadow_bundle, directory)
    if problem in {"not_accepted", "bad_rule"}:
        path = directory / "matched/decisions.json"
        data = json.loads(path.read_text())
        if problem == "not_accepted":
            data["decisions"][3]["accepted"] = False
        else:
            data["decisions"][3]["policy_cost"] = data["decisions"][3]["baseline_cost"] + 1
        branch.write_json(path, data)
    elif problem in {"hash", "candidate_bounds", "snapshot"}:
        path = directory / ("traces.npz" if problem == "snapshot" else "matched/candidates.npz")
        with np.load(path) as data:
            arrays = {key: data[key].copy() for key in data.files}
        if problem == "snapshot":
            arrays["post_initial_plan_flags"][3] = True
        else:
            arrays["plans"][3, 0, 0] = 2.0
        np.savez_compressed(path, **arrays)
    elif problem == "source_archive":
        (directory / "source/src/mppi/mppi.py").write_text("# not the recorded source\n")
    elif problem == "status":
        path = directory / "report.json"
        data = json.loads(path.read_text())
        data["reference_trace_exact"] = False
        branch.write_json(path, data)
    else:
        path = directory / "manifest.json"
        data = json.loads(path.read_text())
        if problem == "task_hash":
            data["task_sha256"] = "0" * 64
        elif problem == "config_value":
            data["mppi_config"]["H"] = 100
        else:
            data["provenance"]["source_sha256"]["src/mppi/mppi.py"] = "0" * 64
        branch.write_json(path, data)
    if problem != "hash":
        _hash_bundle(directory)
    with pytest.raises((ValueError, RuntimeError)):
        branch.load_decision(directory, "matched", 3)


@pytest.mark.parametrize("step", [-1, 12, True])
def test_rejects_invalid_decision_indices(shadow_bundle, step):
    with pytest.raises(ValueError):
        branch.load_decision(shadow_bundle, "matched", step)


def test_raw_reference_mismatch_is_saved_as_failure(shadow_bundle, tmp_path, monkeypatch):
    original = branch.compare_reference

    def mismatched(*args):
        result = original(*args)
        result["exact"] = False
        return result

    monkeypatch.setattr(branch, "compare_reference", mismatched)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="saved shadow prefix"):
        branch.main(shadow_bundle, "matched", 3, output, steps=6)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert "raw" in report["branches"] and "intervention" not in report["branches"]
    assert (output / "artifact_hashes.json").is_file()
