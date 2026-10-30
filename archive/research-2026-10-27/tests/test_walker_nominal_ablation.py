"""Synthetic trace tests: no new Walker simulation or production runs."""

import copy
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics import walker_nominal_ablation as ablation
from src.utils.config import MPPIConfig


def _hash(directory):
    ablation.write_json(directory / "artifact_hashes.json", {
        str(p.relative_to(directory)): ablation.sha256(p) for p in directory.rglob("*")
        if p.is_file() and p != directory / "artifact_hashes.json"})


def _fake_branch(task, cfg, state, warmstart, snapshot, action, steps):
    states = np.tile(state, (steps + 1, 1))
    states[1:, 0] += np.arange(1, steps + 1) * .008
    states[1:, 1] += float(action[0]) * .001
    states[2:, 1] += float(snapshot["U"].mean()) * .002
    warmstarts = np.tile(warmstart, (steps + 1, 1))
    warmstarts[1:] += float(action[0]) * .1
    warmstarts[2:] += float(snapshot["U"].mean()) * .2
    actions = np.tile(snapshot["U"][0], (steps, 1))
    actions[0] = action
    costs = np.full(steps, .5 + float(snapshot["U"].mean()) * .1)
    costs[0] = .5 + float(action.sum()) ** 2 * .001
    rng = np.array([json.dumps(snapshot["rng_state"])] * steps)
    trace = {"states": states, "warmstarts": warmstarts, "actions": actions, "costs": costs,
             "pre_plan_nominals": np.tile(snapshot["U"], (steps, 1, 1)),
             "post_plan_nominals": np.tile(snapshot["U"], (steps, 1, 1)),
             "planner_rng_states": rng.copy(), "post_planner_rng_states": rng.copy(),
             "initial_plan_flags": np.zeros(steps, bool), "post_initial_plan_flags": np.zeros(steps, bool),
             "planned_here": np.arange(steps) > 0}
    summary = {"first_fall_step": None, "completed_steps": steps, "completed_without_fall": True,
               "requested_steps": steps, "fixed_horizon_task_cost": float(costs.sum()),
               "task_cost": float(costs.sum()), "velocity_mae": 0., "mean_velocity": 1.5,
               "start_snapshot": ablation._json_snapshot(snapshot)}
    summary["window48"] = ablation.window_metrics(trace, summary, task)
    return trace, summary


@pytest.fixture
def controls(tmp_path, monkeypatch):
    directory = tmp_path / "evidence/controls/case"
    directory.mkdir(parents=True)
    shadow = tmp_path / "evidence/shadow/success"
    shadow.mkdir(parents=True)
    task = json.loads((ablation._ROOT / "configs/walker2d_task.json").read_text())
    cfg = MPPIConfig(K=8, H=48, initial_iterations=2)
    state = np.zeros(19)
    state[2], state[10] = 1.25, 1.5
    warmstart, baseline = np.arange(9, dtype=float), np.full(6, .2)
    pre = {"U": np.full((48, 6), .05), "rng_state": np.random.default_rng(11).bit_generator.state, "initial_plan": True}
    post = {"U": np.full((48, 6), .2), "rng_state": np.random.default_rng(13).bit_generator.state, "initial_plan": False}
    start = {"task": task, "cfg": cfg, "state": state, "warmstart": warmstart,
             "snapshot": post, "baseline_action": baseline, "decision_step": 0, "policy_label": "strong",
             "shadow_dir": str(shadow), "artifact_hashes_sha256": "f" * 64,
             "trace": {"pre_plan_nominals": pre["U"][None], "planner_rng_states": np.array([json.dumps(pre["rng_state"])]),
                       "initial_plan_flags": np.array([True]), "plans": np.full((1, 48, 6), .2)}}
    monkeypatch.setattr(ablation, "load_decision", lambda *args: copy.deepcopy(start))
    monkeypatch.setattr(ablation, "run_branch", _fake_branch)
    sources = {p: ablation.sha256(ablation._ROOT / p) for p in ablation.FROZEN_SOURCES}
    for relative in sources:
        dest = directory / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ablation._ROOT / relative, dest)
    manifest = {"shadow_dir": str(shadow), "decision_step": 0, "policy_label": "strong",
                "shadow_artifact_hashes_sha256": "f" * 64, "steps": 250,
                "proposal_seeds": list(range(31000, 31005)), "task": task, "mppi_config": asdict(cfg),
                "pre_call_snapshot": ablation._json_snapshot(pre),
                "original_post_call_snapshot": ablation._json_snapshot(post),
                "provenance": {"source_sha256": sources}}
    ablation.write_json(directory / "manifest.json", manifest)
    report = {"status": "complete", "raw_reference_comparison": {"exact": True},
              "identity_comparison": {"exact": True}, "original_rng_recovery_exact": True, "branches": {}}
    for label in ("raw", "identity"):
        trace, summary = _fake_branch(task, cfg, state, warmstart, post, baseline, 250)
        ablation._save_trace(directory / f"{label}.npz", trace)
        report["branches"][label] = summary
    proposals = {}
    for i, seed in enumerate(range(31000, 31005)):
        name, plan = f"seed_{seed}", np.full((48, 6), .3 + i * .01)
        proposal_pre = copy.deepcopy(pre)
        proposal_pre["rng_state"] = np.random.default_rng(seed).bit_generator.state
        proposal_post = {"U": plan.copy(), "rng_state": np.random.default_rng(seed + 100).bit_generator.state, "initial_plan": False}
        proposals[name] = {"proposal_seed": seed, "action": plan[0].tolist(), "optimized_plan": plan.tolist(),
                           "pre_call_snapshot": ablation._json_snapshot(proposal_pre),
                           "discarded_post_call_snapshot": ablation._json_snapshot(proposal_post), "optimization_iterations": 2}
        trace, summary = _fake_branch(task, cfg, state, warmstart, post, plan[0], 250)
        ablation._save_trace(directory / f"{name}.npz", trace)
        report["branches"][name] = summary
    ablation.write_json(directory / "proposals.json", proposals)
    ablation.write_json(directory / "report.json", report)
    ablation.write_json(directory / "original_rng_recovery.json", {
        "pre_call_snapshot": ablation._json_snapshot(pre), "discarded_post_call_snapshot": ablation._json_snapshot(post),
        "action": baseline.tolist(), "optimized_plan": np.full((48, 6), .2).tolist()})
    _hash(directory)
    prereg = tmp_path / "preregistration.md"
    prereg.write_text("Freeze same actions and future RNG; replace only U.\n")
    return directory, start, prereg


def test_matched_snapshot_changes_only_u_without_mutating_inputs(controls):
    directory, start, _ = controls
    proposal = json.loads((directory / "proposals.json").read_text())["seed_31000"]
    before = copy.deepcopy((start["snapshot"], proposal))
    result = ablation.matched_snapshot(start["snapshot"], proposal)
    np.testing.assert_array_equal(result["U"], proposal["discarded_post_call_snapshot"]["U"])
    assert result["rng_state"] == start["snapshot"]["rng_state"] != proposal["discarded_post_call_snapshot"]["rng_state"]
    assert result["initial_plan"] is False
    result["U"][0, 0] = -.9
    assert ablation._json_snapshot(start["snapshot"]) == ablation._json_snapshot(before[0])
    assert proposal == before[1]


def test_main_archives_prereg_before_runs_and_changes_only_nominal(controls, tmp_path, monkeypatch):
    directory, start, prereg = controls
    output, seen = tmp_path / "ablation", []

    def run(task, cfg, state, warmstart, snapshot, action, steps):
        assert (output / "manifest.json").is_file()
        assert (output / "inputs/preregistration.md").read_bytes() == prereg.read_bytes()
        assert (output / "source/scripts/diagnostics/walker_nominal_ablation.py").is_file()
        np.testing.assert_array_equal(state, start["state"])
        np.testing.assert_array_equal(warmstart, start["warmstart"])
        assert snapshot["rng_state"] == start["snapshot"]["rng_state"] and not snapshot["initial_plan"]
        seen.append(copy.deepcopy(snapshot))
        return _fake_branch(task, cfg, state, warmstart, snapshot, action, steps)

    monkeypatch.setattr(ablation, "run_branch", run)
    ablation.main(directory, output, prereg)
    assert len(seen) == 6
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete" and report["raw_replay_comparison"]["exact"]
    assert len(report["branches"]) == 5
    assert report["summary"]["matched_minus_original"]["250"]["n"] == 5
    assert report["summary"]["matched_minus_original"]["250"]["higher_cost"] == 5
    for branch in report["branches"].values():
        assert branch["checks"]["first_transition_exact"] and branch["checks"]["future_rng_prefix_exact"]
        assert branch["checks"]["future_rng_checked_steps"] == 250
        for horizon in ("48", "250"):
            expected = branch["matched_metrics"][horizon]["fixed_horizon_task_cost"] - branch["original_metrics"][horizon]["fixed_horizon_task_cost"]
            assert branch["matched_minus_original"][horizon]["task_cost"] == expected > 0
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert "inputs/artifact_hashes.json" in hashes
    assert all(ablation.sha256(output / p) == h for p, h in hashes.items())
    assert json.loads((output / "manifest.json").read_text())["preregistration_sha256"] == ablation.sha256(prereg)
    with pytest.raises(FileExistsError):
        ablation.main(directory, output, prereg)


@pytest.mark.parametrize("problem", ["shift", "action", "seed_rng", "iterations", "post_flag", "bounds"])
def test_rejects_inconsistent_hashed_proposals(controls, problem):
    directory, _, _ = controls
    path = directory / "proposals.json"
    proposals = json.loads(path.read_text())
    p = proposals["seed_31000"]
    if problem == "shift":
        p["discarded_post_call_snapshot"]["U"][0][0] += .01
    elif problem == "action":
        p["action"][0] += .01
    elif problem == "seed_rng":
        p["pre_call_snapshot"]["rng_state"] = np.random.default_rng(1).bit_generator.state
    elif problem == "iterations":
        p["optimization_iterations"] = 1
    elif problem == "post_flag":
        p["discarded_post_call_snapshot"]["initial_plan"] = True
    else:
        p["optimized_plan"][5][0] = 1.01
    ablation.write_json(path, proposals)
    _hash(directory)
    with pytest.raises(ValueError):
        ablation.load_controls(directory)


def test_rejects_modified_unhashed_evidence(controls):
    directory, _, _ = controls
    (directory / "proposals.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        ablation.load_controls(directory)


@pytest.mark.parametrize("field", ["states", "warmstarts", "actions", "costs", "planner_rng_states", "post_planner_rng_states", "post_initial_plan_flags"])
def test_pair_guard_rejects_changed_transition_or_future_rng(controls, field):
    directory, _, _ = controls
    original = ablation.load_controls(directory)["traces"]["seed_31000"]
    modified = copy.deepcopy(original)
    if field in {"planner_rng_states", "post_planner_rng_states"}:
        modified[field][3] = "bad"
    elif field == "post_initial_plan_flags":
        modified[field][3] = True
    else:
        modified[field][1 if field in {"states", "warmstarts"} else 0] += .01
    with pytest.raises(RuntimeError):
        ablation.verify_pair(original, modified)


def test_rng_prefix_can_end_at_different_fall_times(controls):
    directory, _, _ = controls
    original = ablation.load_controls(directory)["traces"]["seed_31000"]
    shorter = {k: v[:11 if k in {"states", "warmstarts"} else 10].copy() for k, v in original.items()}
    check = ablation.verify_pair(original, shorter)
    assert check["first_transition_exact"] and check["future_rng_prefix_exact"]
    assert check["future_rng_checked_steps"] == 10


def test_raw_replay_mismatch_stops_and_preserves_failure_report(controls, tmp_path, monkeypatch):
    directory, _, prereg = controls
    output = tmp_path / "failed"

    def fail(*args):
        trace, summary = _fake_branch(*args)
        trace["costs"][0] += .01
        return trace, summary

    monkeypatch.setattr(ablation, "run_branch", fail)
    with pytest.raises(RuntimeError, match="Raw/original-U"):
        ablation.main(directory, output, prereg)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and not report["branches"]
    assert (output / "artifact_hashes.json").is_file()


@pytest.mark.parametrize("where", ["controls", "controls_sibling", "shadow", "shadow_sibling", "symlink"])
def test_outputs_cannot_modify_audited_roots(controls, tmp_path, where):
    directory, start, prereg = controls
    if where == "controls":
        output = directory / "new"
    elif where == "controls_sibling":
        output = directory.parent / "new"
    elif where == "shadow":
        output = Path(start["shadow_dir"]) / "new"
    elif where == "shadow_sibling":
        output = Path(start["shadow_dir"]).parent / "new"
    else:
        link = tmp_path / "alias"
        link.symlink_to(directory, target_is_directory=True)
        output = link / "new"
    with pytest.raises(ValueError, match="outside"):
        ablation.main(directory, output, prereg)
    assert not output.exists()


def test_preregistration_is_required_before_loading_or_simulating(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, "load_controls", lambda *args: pytest.fail("No loading without preregistration"))
    with pytest.raises(ValueError, match="preregistration"):
        ablation.main(tmp_path / "inputroot/case", tmp_path / "output", tmp_path / "missing.md")


def test_descriptive_summary_preserves_signs_falls_and_empty_groups():
    assert ablation.summarize({})["matched_minus_original"]["48"]["median"] is None
    branches = {str(i): {contrast: {h: {"task_cost": value, "additional_fall": i == 2,
                                       "prevented_fall": i == 0} for h in ("48", "250")}
                         for contrast in ("matched_minus_original", "matched_minus_raw")}
                for i, value in enumerate((-2., 0., 5.))}
    result = ablation.summarize(branches)["matched_minus_original"]["48"]
    assert result == {"n": 3, "lower_cost": 1, "higher_cost": 1, "ties": 1,
                      "min": -2., "median": 0., "max": 5., "additional_falls": 1, "prevented_falls": 1}
