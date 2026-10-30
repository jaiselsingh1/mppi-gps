"""Saved-window analysis: exact indexing and terminal-vs-truncated handling."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics import compare_walker_plan_realized as analysis


def episode(costs, fell=False):
    states = np.zeros((len(costs) + 1, 19))
    states[:, 2] = 1.2
    if fell:
        states[-1, 2] = .7
    return {"states": states, "costs": np.asarray(costs, dtype=float)}


def decisions(count, baseline=10., policy=11.):
    return [{"step": i, "baseline_cost": baseline, "policy_cost": policy,
             "accepted": policy is not None and policy <= baseline,
             "baseline_fell": False, "policy_fell": False if policy is not None else None}
            for i in range(count)]


def test_windows_include_current_action_and_exact_horizon_boundary():
    rows, counts = analysis.analyze_episode(episode([1, 2, 4, 8]), decisions(4), 2, 5, False)
    assert [row["realized_cost"] for row in rows] == [3, 6, 12]
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert counts["fully_observed"] == 3
    assert counts["skipped_incomplete_nonfall"] == 1


def test_fall_absorbs_only_unused_steps_and_terminal_is_not_added_twice():
    # 27 includes the one-time terminal charge already logged by live Walker.
    rows, counts = analysis.analyze_episode(episode([1, 2, 27], True), decisions(3), 2, 5, True)
    assert [row["realized_cost"] for row in rows] == [3, 29, 32]
    assert [row["absorbing_steps"] for row in rows] == [0, 0, 1]
    assert rows[1]["window_type"] == "fully_observed"
    assert counts["fully_observed"] == 2 and counts["absorbing_padded"] == 1


def test_nonfall_short_episode_has_no_absorbing_padding():
    rows, counts = analysis.analyze_episode(episode([1, 2]), decisions(2), 4, 5, False)
    assert rows == []
    assert counts["skipped_incomplete_nonfall"] == 2
    assert counts["absorbing_padded"] == 0
    assert analysis.summarize(rows)["median_ratio"] is None


def test_short_fall_episode_pads_all_windows():
    rows, counts = analysis.analyze_episode(episode([1, 27], True), decisions(2), 4, 5, True)
    assert [row["realized_cost"] for row in rows] == [38, 42]
    assert counts["fully_observed"] == 0 and counts["absorbing_padded"] == 2


def test_acceptance_denominator_excludes_skipped_incomplete_windows():
    records = decisions(4, baseline=10, policy=4)
    records[1].update(policy_cost=11, accepted=False)
    rows, counts = analysis.analyze_episode(episode([1, 2, 4, 8]), records, 2, 5, False)
    summary = analysis.summarize(rows)
    assert counts["accepted_total"] == 3 and counts["accepted_included"] == 2
    assert summary["accepted"] == 2 and summary["accepted_beating_realized"] == 1
    assert summary["fraction_accepted_beating_realized"] == .5


def test_zero_acceptance_and_missing_policy_scores_are_json_null_not_nan():
    rows, _ = analysis.analyze_episode(episode([1, 2]), decisions(2, policy=None), 1, 5, False)
    summary = analysis.summarize(rows)
    assert summary["accepted"] == 0
    assert summary["fraction_accepted_beating_realized"] is None
    json.dumps(summary, allow_nan=False)


def test_zero_baseline_cost_omits_only_undefined_ratios():
    rows, _ = analysis.analyze_episode(episode([1, 2]), decisions(2, baseline=0), 1, 5, False)
    assert analysis.summarize(rows)["ratio_n"] == 0
    assert analysis.summarize(rows)["median_ratio"] is None


@pytest.mark.parametrize("state_fell,reported_fell", [(True, False), (False, True)])
def test_fall_report_must_match_actual_final_health(state_fell, reported_fell):
    with pytest.raises(ValueError, match="Final-state health"):
        analysis.analyze_episode(episode([1, 2], state_fell), decisions(2), 2, 5, reported_fell)


def test_refuse_actions_after_termination():
    data = episode([1, 2], True)
    data["states"][1, 2] = .7
    with pytest.raises(ValueError, match="after its first fall"):
        analysis.analyze_episode(data, decisions(2), 2, 5, True)


@pytest.mark.parametrize("problem", ["index", "score", "accepted", "fall_guard", "length"])
def test_refuse_misaligned_or_invalid_decisions(problem):
    records = decisions(3)
    if problem == "index":
        records[1]["step"] = 2
    elif problem == "score":
        records[0]["baseline_cost"] = float("nan")
    elif problem == "accepted":
        records[0]["accepted"] = True
    elif problem == "fall_guard":
        records[0].update(policy_cost=1, policy_fell=True, accepted=True)
    else:
        records.pop()
    with pytest.raises(ValueError):
        analysis.analyze_episode(episode([1, 2, 3]), records, 2, 5, False)


@pytest.mark.parametrize("horizon", [True, 0, -1, 1.5])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(ValueError, match="horizon"):
        analysis.analyze_episode(episode([1]), decisions(1), horizon, 5, False)


def test_output_is_exclusive_and_cannot_be_inside_inputs(tmp_path, monkeypatch):
    existing = tmp_path / "existing.json"
    existing.write_text("keep")
    with pytest.raises(FileExistsError):
        analysis.main(output=existing)
    assert existing.read_text() == "keep"
    with pytest.raises(ValueError, match="outside the input"):
        analysis.main(success_shadow=tmp_path / "shadow", output=tmp_path / "shadow/new.json")
    monkeypatch.setattr(analysis, "compare", lambda *args: {key: {} for key in (
        "analysis", "policy_label", "horizon", "absorbing_cost", "groups", "episodes", "limits")})
    fresh = tmp_path / "fresh/report.json"
    analysis.main(output=fresh)
    assert fresh.exists()


def test_artifact_paths_cannot_escape_bundle(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        analysis._artifact_path(tmp_path, "../other.json")
    with pytest.raises(ValueError, match="escapes"):
        analysis._artifact_path(tmp_path, str(Path("/absolute.json")))


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False))


def _rehash(directory):
    _write_json(directory / "artifact_hashes.json", {
        str(path.relative_to(directory)): analysis.sha256(path)
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"})


@pytest.fixture
def saved_bundle(tmp_path):
    directory = tmp_path / "shadow"
    directory.mkdir()
    trace = tmp_path / "reference.npz"
    data = episode([1, 2, 3])
    data.update(warmstarts=np.zeros((4, 9)), actions=np.zeros((3, 6)),
                plans=np.zeros((3, 2, 6)), pre_plan_nominals=np.zeros((3, 2, 6)),
                planner_rng_states=np.asarray(["{}"] * 3), initial_plan_flags=np.array([True, False, False]),
                episode_step_offsets=np.array([0, 3]), episode_state_offsets=np.array([0, 4]),
                reset_seeds=np.array([30]), planner_seeds=np.array([20030]))
    np.savez_compressed(trace, **data)
    np.savez_compressed(directory / "traces.npz", **data)
    task = {"cost_style": "target_velocity", "unhealthy_cost_weight": 7.0}
    config = {"H": 2}
    _write_json(directory / "inputs/task.json", task)
    _write_json(directory / "inputs/mppi.json", config)
    _write_json(directory / "strong/decisions.json", {"decisions": decisions(3)})
    _write_json(directory / "report.json", {"status": "complete", "shadow_only": True,
        "reference_trace_exact": True, "tau": 0.0, "fell": False, "requested_steps": 3,
        "completed_steps": 3, "policies": {"strong": {"decisions": 3}}})
    sources = {}
    for relative in ("src/envs/walker2d.py", "src/utils/trace_replay.py"):
        target = directory / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((analysis._ROOT / relative).read_bytes())
        sources[relative] = analysis.sha256(target)
    _write_json(directory / "manifest.json", {"reference_sha256": analysis.sha256(trace),
        "reference_episode": 0, "task": task, "mppi_config": config,
        "task_sha256": analysis.sha256(directory / "inputs/task.json"),
        "config_sha256": analysis.sha256(directory / "inputs/mppi.json"),
        "provenance": {"source_sha256": sources}, "selection": {"checkpoints": {"strong": {}}},
        "reset_seed": 30, "planner_seed": 20030})
    _rehash(directory)
    return trace, directory


def test_loader_infers_cost_and_horizon_from_hashed_inputs(saved_bundle):
    trace, directory = saved_bundle
    loaded = analysis.load_inputs(trace, directory, "strong")
    assert loaded["horizon"] == 2 and loaded["absorbing_cost"] == 7.0
    assert loaded["provenance"]["trace_sha256"] == analysis.sha256(trace)


def test_report_contains_input_source_and_reproducer_hashes(saved_bundle):
    trace, directory = saved_bundle
    report = analysis.compare(trace, trace, directory, directory)
    assert report["groups"]["all"]["n"] == 4
    assert report["source_sha256"]["scripts/diagnostics/compare_walker_plan_realized.py"] == analysis.sha256(Path(analysis.__file__))
    assert report["inputs"]["success"]["artifact_hashes_sha256"] == analysis.sha256(directory / "artifact_hashes.json")
    assert "not the causal benefit" in report["limits"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("problem", ["artifact_hash", "reference_hash", "trace_alignment", "config", "source", "status", "fall_report", "decision_index"])
def test_loader_or_analysis_rejects_corrupted_evidence(saved_bundle, problem):
    trace, directory = saved_bundle
    if problem in {"artifact_hash", "reference_hash"}:
        target = trace if problem == "reference_hash" else directory / "strong/decisions.json"
        target.write_bytes(target.read_bytes() + b" ")
    elif problem == "trace_alignment":
        with np.load(directory / "traces.npz") as arrays:
            data = {key: arrays[key].copy() for key in arrays.files}
        data["states"][0, 1] = .1
        np.savez_compressed(directory / "traces.npz", **data)
    else:
        target = directory / ("report.json" if problem in {"status", "fall_report"}
                              else "strong/decisions.json" if problem == "decision_index" else "manifest.json")
        data = json.loads(target.read_text())
        if problem == "config":
            data["mppi_config"]["H"] = 3
        elif problem == "source":
            data["provenance"]["source_sha256"]["src/envs/walker2d.py"] = "0" * 64
        elif problem == "status":
            data["status"] = "failed"
        elif problem == "fall_report":
            data["fell"] = True
        else:
            data["decisions"][1]["step"] = 2
        _write_json(target, data)
    if problem != "artifact_hash":
        _rehash(directory)
    with pytest.raises(ValueError):
        loaded = analysis.load_inputs(trace, directory, "strong")
        analysis.analyze_episode(loaded["episode"], loaded["decisions"],
            loaded["horizon"], loaded["absorbing_cost"], loaded["reported_fall"])
