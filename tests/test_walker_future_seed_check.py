"""Future randomness changes neither fixed first actions nor nominal mode."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics import walker_action_controls as controls
from scripts.diagnostics import walker_future_seed_check as future
from scripts.diagnostics import walker_gate_counterfactual as branch
from test_walker_gate_counterfactual import shadow_bundle


def _hashes(directory):
    return {str(path.relative_to(directory)): branch.sha256(path)
            for path in directory.rglob("*") if path.is_file()}


def _trace(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


@pytest.fixture(scope="module")
def evidence(shadow_bundle, tmp_path_factory):
    root = tmp_path_factory.mktemp("future_seed_inputs")
    cdir, pdir = root / "controls", root / "policy"
    controls.main(shadow_bundle, "matched", 3, cdir, proposal_seeds=future.PROPOSAL_SEEDS, steps=12)
    branch.main(shadow_bundle, "matched", 3, pdir, steps=12)
    start = branch.load_decision(shadow_bundle, "matched", 3)
    actions, metadata, expected = future.load_actions(start, cdir, pdir)
    return start, actions, metadata, expected, cdir, pdir


def test_future_snapshot_changes_only_rng_and_preserves_inputs_and_global_rng(evidence):
    start, *_ = evidence
    original = copy.deepcopy(start["snapshot"])
    before = copy.deepcopy(np.random.get_state())
    result = future.future_snapshot(start["snapshot"], 41000)
    np.testing.assert_array_equal(result["U"], original["U"])
    assert result["initial_plan"] is original["initial_plan"] is False
    assert result["rng_state"] == np.random.default_rng(41000).bit_generator.state
    assert result["rng_state"] != original["rng_state"]
    result["U"][0, 0] += 1
    assert branch._json_snapshot(start["snapshot"]) == branch._json_snapshot(original)
    for left, right in zip(np.random.get_state(), before):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "41000", None])
def test_invalid_future_seed_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        future.future_snapshot({"initial_plan": False}, seed)


@pytest.mark.parametrize("flag", [True, 0, None])
def test_future_snapshot_requires_original_post_call_flag(flag):
    with pytest.raises(ValueError, match="post-planning"):
        future.future_snapshot({"initial_plan": flag}, 41000)


def test_loaded_actions_are_exact_saved_first_actions_and_all_inputs_hashed(evidence):
    start, actions, metadata, expected, cdir, pdir = evidence
    assert list(actions) == ["raw", "policy"] + [f"seed_{seed}" for seed in range(31000, 31005)]
    assert metadata["artifact_hashes_sha256"] == branch.sha256(cdir / "artifact_hashes.json")
    assert metadata["policy_artifact_hashes_sha256"] == branch.sha256(pdir / "artifact_hashes.json")
    for label in actions:
        trace = _trace((pdir / "intervention.npz") if label == "policy" else cdir / f"{label}.npz")
        np.testing.assert_array_equal(actions[label], trace["actions"][0])
        future.verify_first_transition(trace, expected[label])
    np.testing.assert_array_equal(actions["raw"], start["baseline_action"])


@pytest.mark.parametrize("source", ["controls", "policy"])
def test_hash_corruption_prevents_loading_actions(evidence, tmp_path, source):
    start, _, _, _, cdir, pdir = evidence
    changed = tmp_path / source
    shutil.copytree(cdir if source == "controls" else pdir, changed)
    (changed / "raw.npz").write_bytes((changed / "raw.npz").read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        future.load_actions(start, changed if source == "controls" else cdir,
                            changed if source == "policy" else pdir)


@pytest.mark.parametrize("field", ["actions", "states", "warmstarts", "costs"])
def test_saved_first_transition_guard_catches_every_boundary_field(evidence, field):
    _, _, _, expected, cdir, _ = evidence
    trace = _trace(cdir / "raw.npz")
    index = 1 if field in {"states", "warmstarts"} else 0
    if field == "costs":
        trace[field][index] += .01
    else:
        trace[field][index].flat[0] += .01
    with pytest.raises(RuntimeError, match="fixed first transition"):
        future.verify_first_transition(trace, expected["raw"])


def _run(start, action, seed=41000, steps=8):
    return branch.run_branch(start["task"], start["cfg"], start["state"], start["warmstart"],
                             future.future_snapshot(start["snapshot"], seed), action, steps)


def test_identical_action_repeat_is_bit_identical_and_matches_saved_first_transition(evidence):
    start, actions, _, expected, *_ = evidence
    first, _ = _run(start, actions["raw"])
    changed, _ = _run(start, actions["seed_31000"])
    repeated, _ = _run(start, actions["raw"])
    for field in first:
        np.testing.assert_array_equal(first[field], repeated[field])
    shared = future.future_snapshot(start["snapshot"], 41000)
    assert future.verify_pair(first, changed, start["state"], start["warmstart"], shared)["exact"]
    future.verify_first_transition(first, expected["raw"])
    future.verify_first_transition(changed, expected["seed_31000"])


def test_unequal_branch_lengths_still_check_entire_shared_rng_prefix(evidence):
    start, actions, *_ = evidence
    raw, _ = _run(start, actions["raw"], steps=8)
    short, _ = _run(start, actions["seed_31000"], steps=3)
    shared = future.future_snapshot(start["snapshot"], 41000)
    result = future.verify_pair(raw, short, start["state"], start["warmstart"], shared)
    assert result["checked_controls"] == 3
    short["post_planner_rng_states"][2] = "{}"
    with pytest.raises(RuntimeError, match="different future"):
        future.verify_pair(raw, short, start["state"], start["warmstart"], shared)


@pytest.mark.parametrize("field", ["states", "warmstarts", "pre_plan_nominals", "post_plan_nominals",
                                   "planner_rng_states", "planned_here", "initial_plan_flags"])
def test_pair_guard_rejects_changed_initial_state_or_nominal_mode(evidence, field):
    start, actions, *_ = evidence
    raw, _ = _run(start, actions["raw"])
    altered = copy.deepcopy(raw)
    if field == "planner_rng_states":
        altered[field][0] = "{}"
    elif field in {"planned_here", "initial_plan_flags"}:
        altered[field][0] = True
    else:
        altered[field][0].flat[0] += .01
    with pytest.raises(RuntimeError, match="common state/post-call memory"):
        future.verify_pair(raw, altered, start["state"], start["warmstart"],
                           future.future_snapshot(start["snapshot"], 41000))


def test_completed_case_reuses_one_raw_per_seed_keeps_inputs_and_archives_every_trace(evidence, tmp_path, monkeypatch):
    start, actions, _, expected, cdir, pdir = evidence
    frozen = copy.deepcopy((start, actions, expected))
    before = {path: _hashes(path) for path in (cdir, pdir, Path(start["shadow_dir"]))}
    calls = []
    original = future.run_branch

    def observe(task, cfg, state, warmstart, snapshot, action, steps):
        calls.append((branch._json_snapshot(snapshot), action.copy()))
        return original(task, cfg, state, warmstart, snapshot, action, steps)

    monkeypatch.setattr(future, "run_branch", observe)
    output = tmp_path / "case"
    report = future.run_case(start, actions, output, future_seeds=(41000, 41001),
                              steps=8, score_horizon=6, expected_first_transitions=expected)
    assert report["status"] == "complete" and report["completed_branches"] == 14
    assert len(calls) == 14 and len(report["pairs"]) == 12
    assert len(list(output.glob("future_*/*.npz"))) == 14
    for seed_index, seed in enumerate((41000, 41001)):
        shared = branch._json_snapshot(future.future_snapshot(start["snapshot"], seed))
        assert all(snapshot == shared for snapshot, _ in calls[7 * seed_index:7 * seed_index + 7])
    for pair in report["pairs"]:
        seed, label = pair["future_seed"], pair["intervention"]
        raw = _trace(output / f"future_{seed}/raw.npz")
        candidate = _trace(output / f"future_{seed}/{label}.npz")
        assert pair["score_horizon_delta"] == candidate["costs"][:6].sum() - raw["costs"][:6].sum()
        assert pair["continuation_delta"] == candidate["costs"].sum() - raw["costs"].sum()
        assert pair["shared_future_rng_prefix"]["checked_controls"] == 8
        assert pair["saved_first_transition_exact"]
    for label, summary in report["summary"].items():
        group = [row for row in report["pairs"] if row["intervention"] == label]
        values = [row["continuation_delta"] for row in group]
        assert summary["continuation_delta"]["mean"] == np.mean(values)
        assert summary["continuation_delta"]["sample_sd"] == np.std(values, ddof=1)
        assert summary["future_seeds"] == [41000, 41001]
    raw_summaries = [report["branches"][f"future_{seed}/raw"] for seed in (41000, 41001)]
    values = [row["fixed_horizon_task_cost"] for row in raw_summaries]
    assert report["raw_summary"]["continuation_cost"]["mean"] == np.mean(values)
    assert report["raw_summary"]["continuation_cost"]["sample_sd"] == np.std(values, ddof=1)
    assert report["raw_summary"]["continuation_cost"]["n"] == 2
    assert "improved" not in report["raw_summary"]["continuation_cost"]
    for path in before:
        assert _hashes(path) == before[path]
    assert branch._json_snapshot(start["snapshot"]) == branch._json_snapshot(frozen[0]["snapshot"])
    for label in actions:
        np.testing.assert_array_equal(actions[label], frozen[1][label])
        for key in ("action", "state", "warmstart"):
            np.testing.assert_array_equal(expected[label][key], frozen[2][label][key])
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert all(branch.sha256(output / key) == digest for key, digest in hashes.items())
    with pytest.raises(FileExistsError):
        future.run_case(start, actions, output, future_seeds=(41000,), steps=8,
                         score_horizon=6, expected_first_transitions=expected)


def test_falls_use_one_terminal_cost_and_absorbing_tail(evidence, tmp_path):
    start = copy.deepcopy(evidence[0])
    start["state"][3], start["state"][12] = .999, 2.
    actions = {"raw": start["baseline_action"].copy(), "identity": start["baseline_action"].copy()}
    trace, summary = _run(start, actions["raw"], steps=8)
    assert summary["first_fall_step"] == 1
    expected = {name: future.first_transition(trace) for name in actions}
    report = future.run_case(start, actions, tmp_path / "fall", future_seeds=(41000, 41001),
                              steps=8, score_horizon=6, expected_first_transitions=expected)
    for pair in report["pairs"]:
        assert pair["score_horizon_delta"] == pair["continuation_delta"] == 0.
        assert pair["candidate_fell"] and pair["raw_fell"]
        assert pair["shared_future_rng_prefix"]["checked_controls"] == 1
        assert pair["candidate_window"]["fixed_horizon_task_cost"] == trace["costs"][0] + 5 * 5
    for branch_summary in report["branches"].values():
        assert branch_summary["fixed_horizon_task_cost"] == trace["costs"][0] + 7 * 5


def test_partial_failure_is_reported_with_completed_traces_and_hashes(evidence, tmp_path, monkeypatch):
    start, actions, _, expected, *_ = evidence
    original = future.run_branch
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected branch failure")
        return original(*args)

    monkeypatch.setattr(future, "run_branch", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected branch"):
        future.run_case(start, actions, output, future_seeds=(41000,), steps=8,
                         score_horizon=6, expected_first_transitions=expected)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["completed_branches"] == 2
    assert "injected branch failure" in report["error"]
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert all(branch.sha256(output / key) == digest for key, digest in hashes.items())


def test_descriptive_summaries_keep_each_intervention_separate_and_use_sample_sd():
    template = {"candidate_window": {"fell": False}, "raw_window": {"fell": False},
                "candidate_fell": False, "raw_fell": False}
    rows = [{**template, "intervention": label, "future_seed": seed,
             "score_horizon_delta": value, "continuation_delta": -value}
            for label, seed, value in (("policy", 41000, -2.), ("policy", 41001, 2.),
                                      ("seed_31000", 41000, 1.))]
    summary = future.summarize_pairs(rows)
    assert summary["policy"]["score_horizon_delta"] == {
        "n": 2, "mean": 0., "sample_sd": np.std([-2., 2.], ddof=1),
        "min": -2., "max": 2., "improved": 1, "tied": 0, "worsened": 1}
    assert summary["seed_31000"]["continuation_delta"]["sample_sd"] is None
    assert "pooled" not in summary


def test_fall_summaries_separate_score_horizon_from_continuation():
    rows = [{"intervention": "policy", "future_seed": 41000,
             "score_horizon_delta": 1., "continuation_delta": -2.,
             "candidate_window": {"fell": True}, "raw_window": {"fell": False},
             "candidate_fell": True, "raw_fell": True}]
    summary = future.summarize_pairs(rows)["policy"]
    assert summary["additional_falls_at_score_horizon"] == 1
    assert summary["additional_falls_at_continuation"] == 0
    assert summary["prevented_falls_at_score_horizon"] == summary["prevented_falls_at_continuation"] == 0


@pytest.mark.parametrize("key,value", [("first_fall_step", 8), ("completed_steps", 7),
                                      ("fixed_horizon_task_cost", -1.), ("completed_without_fall", False)])
def test_trace_not_unchecked_report_decides_absorption(evidence, key, value):
    start, actions, *_ = evidence
    trace, summary = _run(start, actions["raw"], steps=8)
    summary[key] = value
    with pytest.raises(RuntimeError, match="summary disagrees"):
        future.validate_branch_summary(trace, summary, start["task"], 8)


def test_healthy_early_truncation_is_not_padded_as_a_fall(evidence):
    start, actions, *_ = evidence
    trace, summary = _run(start, actions["raw"], steps=3)
    with pytest.raises(RuntimeError, match="unexplained truncation"):
        future.validate_branch_summary(trace, summary, start["task"], 8)


def test_main_requires_preregistration_and_preserves_input_roots(tmp_path):
    with pytest.raises(ValueError, match="preregistration"):
        future.main(tmp_path / "output", tmp_path / "missing.md")
    assert not (tmp_path / "output").exists()
    for keyword in ("controls_root", "shadow_root"):
        with pytest.raises(ValueError, match="outside input"):
            future.main(tmp_path / "evidence/output", tmp_path / "missing.md",
                        **{keyword: tmp_path / "evidence"})


def test_main_saves_prereg_and_sources_before_first_run_and_reports_exact_fixed_design(evidence, tmp_path, monkeypatch):
    start, _, _, _, cdir, pdir = evidence
    shadow_root, controls_root = tmp_path / "shadow", tmp_path / "controls"
    name = "success_matched_3"
    shutil.copytree(Path(start["shadow_dir"]), shadow_root / "success")
    shutil.copytree(pdir, shadow_root / "branches" / name)
    shutil.copytree(cdir, controls_root / name)
    monkeypatch.setattr(future, "CASES", (("success", "matched", 3),))
    monkeypatch.setattr(future, "FUTURE_SEEDS", (41000, 41001))
    monkeypatch.setattr(future, "STEPS", 8)
    monkeypatch.setattr(future, "SCORE_HORIZON", 6)
    prereg = tmp_path / "protocol.md"
    prereg.write_text("Fixed original U, paired future RNG, no adaptation.\n")
    output = tmp_path / "outputs"
    original = future.run_branch

    def check_before_execution(*args):
        assert (output / "preregistration.md").read_bytes() == prereg.read_bytes()
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["preregistration_sha256"] == branch.sha256(prereg)
        assert manifest["requested_branches"] == 14
        assert (output / "source/scripts/diagnostics/walker_future_seed_check.py").is_file()
        return original(*args)

    monkeypatch.setattr(future, "run_branch", check_before_execution)
    future.main(output, prereg, controls_root, shadow_root)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete" and status["completed_branches"] == status["requested_branches"] == 14
    report = json.loads((output / "report.json").read_text())
    assert report["original_nominal_only"] and report["no_pooled_confidence_interval"]
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert all(branch.sha256(output / key) == digest for key, digest in hashes.items())


@pytest.mark.parametrize("kwargs", [{"future_seeds": ()}, {"future_seeds": (41000, 41000)},
                                    {"steps": 0}, {"score_horizon": 9}])
def test_bad_case_configuration_rejected_before_outputs(evidence, tmp_path, kwargs):
    start, actions, _, expected, *_ = evidence
    settings = dict(future_seeds=(41000,), steps=8, score_horizon=6, expected_first_transitions=expected)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        future.run_case(start, actions, tmp_path / "bad", **settings)
    assert not (tmp_path / "bad").exists()
