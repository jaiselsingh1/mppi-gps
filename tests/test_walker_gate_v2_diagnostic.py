"""V2 calibration must consume audited outcomes without changing their evidence."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics import walker_action_controls as controls
from scripts.diagnostics import walker_gate_counterfactual as branch
from scripts.diagnostics import walker_gate_v2_diagnostic as diagnostic
from src.mppi.gate_v2 import ContinuationGate
from test_walker_gate_counterfactual import shadow_bundle


def _hashes(directory):
    return {str(path.relative_to(directory)): branch.sha256(path)
            for path in directory.rglob("*") if path.is_file()}


def _rehash(directory):
    hashes = {key: value for key, value in _hashes(directory).items() if key != "artifact_hashes.json"}
    branch.write_json(directory / "artifact_hashes.json", hashes)


@pytest.fixture(scope="module")
def continuation_bundles(shadow_bundle, tmp_path_factory):
    root = tmp_path_factory.mktemp("v2_diagnostic_inputs")
    control_dir, policy_dir = root / "controls", root / "policy"
    controls.main(shadow_bundle, "matched", 3, control_dir, proposal_seeds=(31000, 31001), steps=12)
    branch.main(shadow_bundle, "matched", 3, policy_dir, steps=12)
    return branch.load_decision(shadow_bundle, "matched", 3), control_dir, policy_dir


@pytest.mark.parametrize("predicted,actual,expected", [
    (-2., -3., "same_sign"), (2., 3., "same_sign"),
    (-2., 3., "opposite_sign"), (2., -3., "opposite_sign"),
    (0., 2., "tie"), (2., 0., "tie"), (0., 0., "tie"),
    (None, 2., "invalid"), (None, 0., "invalid"),
])
def test_sign_agreement_distinguishes_zero_and_invalid(predicted, actual, expected):
    assert diagnostic.prediction_agreement(predicted, actual) == expected


def test_48_control_prediction_and_250_control_outcome_can_have_opposite_signs():
    raw = {"costs": np.ones(250), "states": np.zeros((251, 19))}
    candidate = {"costs": np.r_[np.full(48, .5), np.full(202, 2.)], "states": raw["states"]}
    task = {"unhealthy_cost_weight": 5., "target_velocity": 1.5}
    summary = {"first_fall_step": None}
    short_delta = (diagnostic.window_metrics(candidate, summary, task, 48)["fixed_horizon_task_cost"]
                   - diagnostic.window_metrics(raw, summary, task, 48)["fixed_horizon_task_cost"])
    long_delta = (diagnostic.window_metrics(candidate, summary, task, 250)["fixed_horizon_task_cost"]
                  - diagnostic.window_metrics(raw, summary, task, 250)["fixed_horizon_task_cost"])
    assert short_delta == -24 and long_delta == 178
    assert diagnostic.prediction_agreement(-1., short_delta) == "same_sign"
    assert diagnostic.prediction_agreement(-1., long_delta) == "opposite_sign"


def test_score_case_preserves_inputs_identity_and_saves_successor_probes(continuation_bundles, tmp_path):
    start, control_dir, policy_dir = continuation_bundles
    before = {str(path): _hashes(path) for path in (control_dir, policy_dir, Path(start["shadow_dir"]))}
    output = tmp_path / "scores"
    report = diagnostic.score_case(start, control_dir, policy_dir, output)
    assert report["status"] == "complete" and report["calibration_only"]
    assert report["successor_replay_exact"]
    assert len(report["rows"]) == 4
    for path in (control_dir, policy_dir, Path(start["shadow_dir"])):
        assert _hashes(path) == before[str(path)]
    with np.load(control_dir / "raw.npz") as data:
        raw = {key: data[key].copy() for key in data.files}
    for row in report["rows"]:
        assert row["score_horizon"] == 6 and row["continuation_controls"] == 12
        label = "intervention" if row["name"] == "policy" else row["name"]
        directory = policy_dir if label == "intervention" else control_dir
        with np.load(directory / f"{label}.npz") as data:
            trace = {key: data[key].copy() for key in data.files}
        assert row["actual_score_horizon_delta"] == float(trace["costs"][:6].sum() - raw["costs"][:6].sum())
        assert row["actual_continuation_delta"] == float(trace["costs"].sum() - raw["costs"].sum())
        for name, actual in (("baseline", raw), ("candidate", trace)):
            with np.load(output / row["name"] / f"{name}.npz") as probe:
                assert probe["optimized_plan"].shape == (6, 6)
                assert probe["scored_plan"].shape == (5, 6)
                np.testing.assert_array_equal(probe["scored_plan"], probe["optimized_plan"][:5])
                np.testing.assert_array_equal(probe["successor_state"], actual["states"][1])
                np.testing.assert_array_equal(probe["successor_warmstart"], actual["warmstarts"][1])
                np.testing.assert_array_equal(probe["optimized_plan"][0], actual["actions"][1])
            post = row[f"{name}_probe"]["planner_snapshot_after"]
            np.testing.assert_array_equal(post["U"], actual["post_plan_nominals"][1])
            assert post["rng_state"] == json.loads(str(actual["post_planner_rng_states"][1]))
    identity = next(row for row in report["rows"] if row["category"] == "identity")
    assert identity["accepted"]
    assert identity["predicted_delta"] == identity["actual_score_horizon_delta"] == identity["actual_continuation_delta"] == 0.
    assert identity["horizon_sign_agreement"] == identity["continuation_sign_agreement"] == "tie"
    assert not identity["false_acceptance_at_score_horizon"] and not identity["false_rejection_at_score_horizon"]
    assert diagnostic.summarize(report["rows"])["identity"]["accepted"] == 1
    assert json.loads((output / "report.json").read_text())["input_hashes"] == {
        "controls": branch.sha256(control_dir / "artifact_hashes.json"),
        "policy_continuations": branch.sha256(policy_dir / "artifact_hashes.json")}
    with pytest.raises(FileExistsError):
        diagnostic.score_case(start, control_dir, policy_dir, output)


def test_score_case_refuses_output_inside_any_evidence_bundle(continuation_bundles):
    start, control_dir, policy_dir = continuation_bundles
    for directory in (control_dir, policy_dir, Path(start["shadow_dir"])):
        output = directory / "never_create_v2_results"
        with pytest.raises(ValueError, match="outside all input"):
            diagnostic.score_case(start, control_dir, policy_dir, output)
        assert not output.exists()


@pytest.mark.parametrize("problem", ["hash", "source_hash", "unfinished", "nominal", "task", "config", "source_archive"])
def test_reject_bad_bundle_before_creating_outputs(continuation_bundles, tmp_path, problem):
    start, control_dir, policy_dir = continuation_bundles
    changed = tmp_path / "controls"
    shutil.copytree(control_dir, changed)
    if problem == "hash":
        (changed / "raw.npz").write_bytes((changed / "raw.npz").read_bytes() + b"tamper")
    elif problem == "source_archive":
        path = changed / "source/src/mppi/mppi.py"
        path.write_text("# different archived source\n")
    else:
        path = changed / ("report.json" if problem == "unfinished" else "manifest.json")
        data = json.loads(path.read_text())
        if problem == "unfinished":
            data["status"] = "running"
        elif problem == "source_hash":
            data["provenance"]["source_sha256"]["src/mppi/mppi.py"] = "0" * 64
        elif problem == "nominal":
            data["original_post_call_snapshot"]["U"][0][0] += .01
        elif problem == "task":
            data["task"]["target_velocity"] += .1
        else:
            data["mppi_config"]["H"] += 1
        branch.write_json(path, data)
    if problem != "hash":
        _rehash(changed)
    output = tmp_path / "scores"
    with pytest.raises(ValueError):
        diagnostic.score_case(start, changed, policy_dir, output)
    assert not output.exists()


@pytest.mark.parametrize("field", ["states", "warmstarts", "pre_plan_nominals", "planner_rng_states", "initial_plan_flags", "planned_here"])
def test_reject_changed_simulator_or_planner_boundary(continuation_bundles, field):
    start, control_dir, _ = continuation_bundles
    with np.load(control_dir / "raw.npz") as data:
        trace = {key: data[key].copy() for key in data.files}
    if field == "planner_rng_states":
        trace[field][0] = "{}"
    elif field in {"initial_plan_flags", "planned_here"}:
        trace[field][0] = True
    else:
        trace[field][0].flat[0] += .01
    with pytest.raises(ValueError, match="different simulator/planner boundary"):
        diagnostic._validate_trace_start(start, trace)


@pytest.mark.parametrize("field", ["states", "warmstarts", "actions", "costs", "post_plan_nominals", "post_planner_rng_states"])
def test_successor_audit_catches_transition_and_next_planner_changes(continuation_bundles, field):
    start, control_dir, _ = continuation_bundles
    with np.load(control_dir / "raw.npz") as data:
        trace = {key: data[key].copy() for key in data.files}
    with ContinuationGate(start["task"], start["cfg"]) as gate:
        score = gate.decide(start["state"], start["warmstart"], start["snapshot"],
                            start["baseline_action"], start["baseline_action"]).baseline_score
    if field == "post_planner_rng_states":
        trace[field][1] = "{}"
    elif field == "costs":
        trace[field][0] += .01
    else:
        trace[field][1].flat[0] += .01
    with pytest.raises(RuntimeError, match="Scoring probe"):
        diagnostic._verify_probe(score, trace)


def test_summaries_keep_identity_separate_from_policy_and_seed_controls():
    rows = [{"category": category, "accepted": True, "false_acceptance_at_score_horizon": False,
             "false_rejection_at_score_horizon": False, "horizon_sign_agreement": "same_sign",
             "continuation_sign_agreement": "opposite_sign"}
            for category in ("policy", "seed_control", "seed_control", "identity")]
    summary = diagnostic.summarize(rows)
    assert summary["policy"]["n"] == 1 and summary["seed_control"]["n"] == 2
    assert summary["identity"]["n"] == 1
    assert summary["seed_control"]["horizon_sign_agreement"]["same_sign"] == 2
    assert summary["seed_control"]["continuation_sign_agreement"]["opposite_sign"] == 2


def test_main_rejects_existing_outputs_and_nested_roots(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        diagnostic.main(existing)
    assert (existing / "keep.txt").read_text() == "keep"
    for kwarg in ("shadow_root", "controls_root"):
        with pytest.raises(ValueError, match="outside the input"):
            diagnostic.main(tmp_path / "evidence/results", **{kwarg: tmp_path / "evidence"})


def test_failed_case_is_logged_and_hashed_by_main(tmp_path, monkeypatch):
    selected = tmp_path / "selection.json"
    branch.write_json(selected, {"branches": [{"episode": "success", "policy_label": "matched", "decision_step": 3}]})
    def fail(*args):
        raise ValueError("injected incomplete evidence")
    monkeypatch.setattr(diagnostic, "load_decision", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected incomplete"):
        diagnostic.main(output, selection=selected)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["cases"] == {}
    assert "injected incomplete" in report["error"]
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert all(branch.sha256(output / relative) == digest for relative, digest in hashes.items())
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
