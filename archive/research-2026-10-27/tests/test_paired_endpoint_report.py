"""Reports must validate measured pairing, isolation, archive equality and work."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics import compare_paired_endpoint as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def equality():
    return dict(archive_identical=True, shared_archive_sha256="a" * 64,
                guided_archive_sha256="a" * 64, memory_archive_sha256="a" * 64)


def panel(*, improved=False, base=35_000_000, horizon=64, count=4):
    episodes = []
    for index in range(count):
        survived = improved or index % 2 == 0
        steps = horizon if survived else horizon // 2
        episodes.append(dict(reset_id=base + index, survived=survived, steps=steps,
            duration_seconds=steps * .008, velocity_mae=.2 if improved else .4,
            cost=100. + index * 10 - (20 if improved else 0),
            guidance_fraction=.5 if improved else 0., action_change_fraction=.25 if improved else 0.))
    summary = dict(episodes=count, survival_count=sum(e["survived"] for e in episodes),
        survival_rate=np.mean([e["survived"] for e in episodes]),
        mean_duration_seconds=np.mean([e["duration_seconds"] for e in episodes]),
        velocity_mae=np.mean([e["velocity_mae"] for e in episodes]),
        finite_discounted_cost=np.mean([e["cost"] for e in episodes]),
        guidance_fraction=np.mean([e["guidance_fraction"] for e in episodes]),
        action_change_fraction=np.mean([e["action_change_fraction"] for e in episodes]),
        walking_gate=improved if horizon == 1000 else None)
    return dict(summary=summary, episodes=episodes)


@pytest.fixture
def study(tmp_path):
    source = tmp_path / "source" / "src" / "frozen.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen paired source\n")
    write(tmp_path / "protocol.json", dict(config=dict(iterations=2, total_steps=64, seed=0, final_episodes=4),
        source_sha256={"src/frozen.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        reset_blocks=dict(evaluation=35_000_000, final=36_000_000),
        reference_controller="Frozen complete-return reference; no actor."))
    rows = []
    for iteration in range(2):
        metrics = None if iteration == 0 else dict(training_normalized_rmse=.1,
            holdout_normalized_rmse=.2, holdout_training_mean_normalized_rmse=.3,
            holdout_knn_normalized_rmse=.25, holdout_pair_normalized_difference_rmse=.1,
            training_groups=[0, 1], holdout_groups=[2, 3])
        rows.append(dict(iteration=iteration, elapsed_seconds=2. + iteration, **equality(),
            deployed_alpha_before=0., deployed_alpha=.1 if iteration else 0.,
            guided_collection=panel(improved=bool(iteration))["summary"],
            memory_collection=panel()["summary"], value_metrics=metrics, measured_query_count=32 * (iteration + 1),
            qualification_records_added=0, qualification_archive_episodes_added=0,
            promotion=dict(accepted=bool(iteration), type="promote_guidance" if iteration else "memory_only",
                pair_calibration=dict(eligible=bool(iteration), normalized_difference_rmse=.1 if iteration else None),
                candidate=panel(improved=True)["summary"], memory=panel()["summary"], incumbent=panel()["summary"]),
            work=dict(shared_bootstrap_candidate_controls=100,
                      training_pairs_prefix_controls=20 * (iteration + 1),
                      training_pairs_continuation_candidate_controls=200 * (iteration + 1),
                      training_pairs_continuation_executed_controls=40 * (iteration + 1))))
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(tmp_path / "evaluations" / "initial.json", dict(guided=panel(), memory=panel(), **equality()))
    write(tmp_path / "evaluations" / "iteration_0001.json", dict(guided=panel(improved=True),
        memory=panel(), work=rows[-1]["work"], **equality()))
    work = dict(rows[-1]["work"], evaluation_guided_candidate_controls=1000,
                evaluation_memory_candidate_controls=1000)
    write(tmp_path / "evaluations" / "final.json", dict(guided=panel(improved=True, base=36_000_000),
        memory=panel(base=36_000_000), work=work, **equality()))
    write(tmp_path / "validation.json", dict(source_unchanged=True))
    write(tmp_path / "study_status.json", dict(status="complete", phase="complete", completed_iterations=2,
        source_unchanged=True, work=work, elapsed_seconds=8., **equality()))
    return tmp_path


def change_rows(study, mutate):
    path = study / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_final_report_uses_fresh_panel_complete_costs_and_no_actor_claim(study):
    result = report.main(study)
    assert result["progress"]["complete"]
    assert result["progress"]["completed_iterations"] == 2
    assert not result["production_protocol"]
    assert result["final"]["guided"]["survival_count"] == 4
    assert result["final"]["guided"]["walking_gate"] is None
    assert result["guidance"]["accepted_promotions"] == 1
    assert result["work"]["known_simulated_controls"] == 2620
    assert result["work"]["simulated_controls_by_kind"]["prefix"] == 40
    assert result["work"]["simulated_controls_by_phase"]["training_pairs"] == 520
    assert result["work"]["simulated_controls_by_phase"]["evaluation_guided"] == 1000
    assert result["paired_guided_minus_memory"]["metrics"]["finite_discounted_cost"]["mean"] == -20.
    assert result["paired_guided_minus_memory"]["reset_ids"] == list(range(36_000_000, 36_000_004))
    assert result["interpretation"]["standalone_policy_evaluated"] is False
    assert result["interpretation"]["sample_efficiency_claim"] is False
    assert result["archive_equality_evidence"] == "Matching recorded numeric archive hashes"
    for filename in ("comparison.json", "comparison.md", "training_curves.png", "final_metrics.png"):
        assert (study / filename).stat().st_size > 100
    assert "not standalone-policy results" in (study / "comparison.md").read_text()


def test_development_curve_excludes_final_panel(study, monkeypatch):
    capture = {}
    def plots(directory, rows, development, initial, latest, intervals, complete):
        capture["development"] = [item["reset_ids"] for item in development]
        capture["final"] = latest["reset_ids"]
    monkeypatch.setattr(report, "_plots", plots)
    report.main(study)
    assert capture["development"] == [list(range(35_000_000, 35_000_004))] * 2
    assert capture["final"] == list(range(36_000_000, 36_000_004))


def test_interim_report_uses_latest_committed_evaluation(study, monkeypatch):
    (study / "evaluations" / "final.json").unlink()
    (study / "validation.json").unlink()
    write(study / "study_status.json", dict(status="running", phase="training_endpoint_pairs"))
    monkeypatch.setattr(report, "_plots", lambda *args: None)
    result = report.main(study)
    assert not result["progress"]["complete"]
    assert result["progress"]["phase"] == "training_endpoint_pairs"
    assert result["final"] is None
    assert result["latest_evaluation"] == "iteration_0001"
    assert "INTERIM" in (study / "comparison.md").read_text()


@pytest.mark.parametrize("kind", ["source", "archive", "final_count", "final_resets", "unpaired", "unordered", "work", "leak", "fit_groups", "promote", "summary"])
def test_rejects_invalid_evidence(study, kind, monkeypatch):
    monkeypatch.setattr(report, "_plots", lambda *args: None)
    if kind == "source":
        (study / "source" / "src" / "frozen.py").write_text("changed")
    elif kind in ("archive", "unordered", "work", "leak", "fit_groups", "promote"):
        def mutate(rows):
            if kind == "archive":
                rows[-1]["guided_archive_sha256"] = "b" * 64
            elif kind == "unordered":
                rows[-1]["iteration"] = 2
            elif kind == "work":
                rows[-1]["work"]["shared_bootstrap_candidate_controls"] = 0
            elif kind == "leak":
                rows[-1]["qualification_records_added"] = 1
            elif kind == "fit_groups":
                rows[-1]["value_metrics"]["holdout_groups"].append(0)
            else:
                rows[-1]["promotion"]["pair_calibration"]["eligible"] = False
        change_rows(study, mutate)
    else:
        path = study / "evaluations" / "final.json"
        value = json.loads(path.read_text())
        if kind == "final_count":
            protocol = json.loads((study / "protocol.json").read_text())
            protocol["config"]["final_episodes"] = 64
            write(study / "protocol.json", protocol)
        elif kind == "final_resets":
            for arm in report.ARMS:
                for episode in value[arm]["episodes"]:
                    episode["reset_id"] += 1
        elif kind == "unpaired":
            value["memory"]["episodes"][0]["reset_id"] += 1000
        elif kind == "summary":
            value["guided"]["summary"]["action_change_fraction"] = 0.
        write(path, value)
    with pytest.raises(ValueError):
        report.main(study)


def test_interrupted_resume_marks_all_work_a_lower_bound(study, monkeypatch):
    write(study / "resume_123.json", dict(unknown_interrupted_work="unrecoverable"))
    monkeypatch.setattr(report, "_plots", lambda *args: None)
    result = report.main(study)
    assert result["work"]["count_is_lower_bound"]
    assert result["work"]["interrupted_work_unknown"]


def test_actual_driver_nested_archive_hashes_and_status_lower_bound(study, monkeypatch):
    for path in (study / "evaluations").glob("*.json"):
        value = json.loads(path.read_text())
        for arm in report.ARMS:
            value[arm]["archive_sha256"] = value.pop(f"{arm}_archive_sha256")
        write(path, value)
    status = json.loads((study / "study_status.json").read_text())
    for arm in report.ARMS:
        status.pop(f"{arm}_archive_sha256")
    status["work_is_lower_bound"] = True
    write(study / "study_status.json", status)
    monkeypatch.setattr(report, "_plots", lambda *args: None)
    result = report.main(study)
    assert result["progress"]["complete"]
    assert result["work"]["count_is_lower_bound"]


def test_production_requires_24_ordered_rows_and_all_64_final_resets(study, monkeypatch):
    protocol = json.loads((study / "protocol.json").read_text())
    protocol["config"].update(iterations=24, total_steps=1000, final_episodes=64)
    write(study / "protocol.json", protocol)
    monkeypatch.setattr(report, "_plots", lambda *args: None)
    with pytest.raises(ValueError):
        report.main(study)


def test_empty_bootstrap_progress_can_be_reported_without_metrics_or_evaluation(study):
    (study / "metrics.jsonl").unlink()
    for path in (study / "evaluations").glob("*.json"):
        path.unlink()
    write(study / "study_status.json", dict(status="running", phase="bootstrap", work={}))
    result = report.main(study)
    assert result["progress"]["completed_iterations"] == 0
    assert result["latest_evaluation"] is None
    assert result["work"]["known_simulated_controls"] == 0
    assert result["latest"] is None


def test_committed_file_and_stream_disagreement_rejected(study):
    rows = [json.loads(line) for line in (study / "metrics.jsonl").read_text().splitlines()]
    altered = copy.deepcopy(rows[0])
    altered["measured_query_count"] += 1
    write(study / "metric_rows" / "iteration_0000.json", altered)
    with pytest.raises(ValueError, match="disagree"):
        report.main(study)
