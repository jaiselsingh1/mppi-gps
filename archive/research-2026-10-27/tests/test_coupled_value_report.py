"""Check controller pairing, source provenance and bounded claims in reports."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics import compare_coupled_value as report


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _panel(*, improved: bool = False, horizon: int = 1000) -> dict:
    episodes = []
    for index in range(4):
        survived = improved or index % 2 == 0
        steps = horizon if survived else horizon // 2
        episodes.append(dict(reset_id=4000 + index, steps=steps, survived=survived,
            duration_seconds=steps * .008, velocity_mae=.2 if improved else .4,
            cost=100 + index * 30 - (20 if improved else 0),
            guidance_fraction=.5 if improved else 0.))
    summary = dict(episodes=len(episodes), survival_count=sum(e["survived"] for e in episodes),
        survival_rate=np.mean([e["survived"] for e in episodes]),
        mean_duration_seconds=np.mean([e["duration_seconds"] for e in episodes]),
        velocity_mae=np.mean([e["velocity_mae"] for e in episodes]),
        finite_discounted_cost=np.mean([e["cost"] for e in episodes]),
        guidance_fraction=np.mean([e["guidance_fraction"] for e in episodes]),
        walking_gate=improved if horizon == 1000 else None)
    return dict(summary=summary, episodes=episodes)


@pytest.fixture
def study(tmp_path: Path) -> Path:
    source = tmp_path / "source" / "src" / "frozen.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen training snapshot\n")
    protocol = dict(config=dict(iterations=2, total_steps=1000, seed=0),
        source_sha256={"src/frozen.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        reference_controller="Fixed canonical-restart reference, not Vpi")
    _write(tmp_path / "protocol.json", protocol)
    rows = [dict(iteration=k, elapsed_seconds=2. + k,
        guided_collection=_panel(improved=True)["summary"],
        memory_collection=_panel()["summary"],
        value_metrics=None if not k else dict(holdout_normalized_rmse=.1),
        promotion=dict(accepted=bool(k), type="value_promotion" if k else "retain_incumbent",
            guidance_used_fraction=.5 if k else 0.),
        work=dict(candidate_controls=100 * (k + 1), executed_controls=40 * (k + 1))) for k in range(2)]
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    _write(tmp_path / "evaluations" / "initial.json", dict(guided=_panel(), memory=_panel()))
    _write(tmp_path / "evaluations" / "iteration_0001.json",
        dict(guided=_panel(improved=True), memory=_panel(), work=rows[-1]["work"]))
    final_work = dict(candidate_controls=400, executed_controls=160, final_evaluation_controls=6000)
    _write(tmp_path / "evaluations" / "final.json",
        dict(guided=_panel(improved=True), memory=_panel(), work=final_work))
    _write(tmp_path / "validation.json", dict(source_unchanged=True, completed_iterations=2,
        no_standalone_actor=True))
    _write(tmp_path / "study_status.json", dict(status="complete", source_unchanged=True,
        completed_iterations=2, elapsed_seconds=8., work=final_work))
    return tmp_path


def test_writes_actual_controller_results_and_paired_intervals(study: Path) -> None:
    result = report.main(study)
    assert result["progress"] == dict(requested_iterations=2, completed_iterations=2, complete=True)
    assert result["initial"]["guided"]["survival_count"] == 2
    assert result["final"]["guided"]["walking_gate"] is True
    assert result["final"]["memory"]["walking_gate"] is False
    interval = result["paired_guided_minus_memory"]["metrics"]["finite_discounted_cost"]
    # Reset-specific offsets must cancel before resampling.
    assert interval["mean"] == interval["lower"] == interval["upper"] == -20.
    assert "one training seed" in result["paired_guided_minus_memory"]["uncertainty_scope"]
    assert result["work"]["final_evaluation_controls"] == 6000
    assert result["elapsed_seconds"] == 8.
    assert result["interpretation"]["standalone_policy_evaluated"] is False
    assert result["interpretation"]["sample_efficiency_claim"] is False
    text = (study / "comparison.md").read_text()
    assert "not standalone-policy results" in text
    assert "empirical" in text and "not a confidence certificate" in text
    for name in ("comparison.json", "comparison.md", "training_curves.png", "final_metrics.png"):
        assert (study / name).stat().st_size > 100


@pytest.mark.parametrize("mutation", ["changed", "missing", "empty_manifest", "unsafe_path"])
def test_source_archive_is_required_and_hash_checked(study: Path, mutation: str) -> None:
    protocol = json.loads((study / "protocol.json").read_text())
    if mutation == "changed":
        (study / "source" / "src" / "frozen.py").write_text("# edited after freeze\n")
    elif mutation == "missing":
        (study / "source" / "src" / "frozen.py").unlink()
    elif mutation == "empty_manifest":
        protocol["source_sha256"] = {}
    else:
        protocol["source_sha256"] = {"../outside.py": "0" * 64}
    with pytest.raises(ValueError, match="source|manifest|SHA256"):
        report._source(study, protocol)


def test_runtime_source_failure_cannot_be_overridden_by_archive(study: Path, monkeypatch) -> None:
    _write(study / "validation.json", dict(source_unchanged=False, completed_iterations=2))
    monkeypatch.setattr(report, "_plots", lambda *args: None)
    result = report.main(study)
    assert result["source_validation"]["archived_source_verified"] is True
    assert result["source_unchanged"] is False


def test_fresh_final_resets_do_not_extend_development_curve(study: Path, monkeypatch) -> None:
    path = study / "evaluations" / "final.json"
    final = json.loads(path.read_text())
    for arm in report.ARMS:
        for episode in final[arm]["episodes"]:
            episode["reset_id"] += 1000
    _write(path, final)
    captured = {}
    def capture_plots(directory, rows, evaluations, initial, latest, intervals, complete):
        captured["development_resets"] = [entry["reset_ids"] for entry in evaluations]
        captured["iterations"] = [entry["completed_iterations"] for entry in evaluations]
        captured["final_resets"] = latest["reset_ids"]
    monkeypatch.setattr(report, "_plots", capture_plots)
    result = report.main(study)
    assert captured["iterations"] == [0, 2]
    assert captured["development_resets"] == [list(range(4000, 4004))] * 2
    assert captured["final_resets"] == list(range(5000, 5004))
    assert result["paired_guided_minus_memory"]["reset_ids"] == list(range(5000, 5004))
    assert result["final"] is not None


def test_interim_report_cannot_claim_final_completion(study: Path, monkeypatch) -> None:
    (study / "evaluations" / "final.json").unlink()
    (study / "validation.json").unlink()
    _write(study / "study_status.json", dict(status="running"))
    protocol = json.loads((study / "protocol.json").read_text())
    protocol["config"]["iterations"] = 4
    _write(study / "protocol.json", protocol)
    monkeypatch.setattr(report, "_plots", lambda *args: None)
    result = report.main(study)
    assert result["progress"] == dict(requested_iterations=4, completed_iterations=2, complete=False)
    assert result["latest_evaluation"] == "iteration_0001"
    assert result["final"] is None
    assert "INTERIM" in (study / "comparison.md").read_text()


def test_refuses_to_join_different_development_reset_panels(study: Path) -> None:
    path = study / "evaluations" / "iteration_0001.json"
    value = json.loads(path.read_text())
    for arm in report.ARMS:
        for episode in value[arm]["episodes"]:
            episode["reset_id"] += 1000
    _write(path, value)
    with pytest.raises(ValueError, match="Development evaluation reset IDs"):
        report.main(study)


def test_refuses_unpaired_resets_even_when_means_match(study: Path) -> None:
    path = study / "evaluations" / "final.json"
    value = json.loads(path.read_text())
    value["memory"]["episodes"][0]["reset_id"] += 100
    _write(path, value)
    with pytest.raises(ValueError, match="paired reset IDs"):
        report._evaluation(path, 1000)


def test_refuses_summary_not_earned_by_episodes(study: Path) -> None:
    path = study / "evaluations" / "final.json"
    value = json.loads(path.read_text())
    value["guided"]["summary"]["finite_discounted_cost"] -= 1.
    _write(path, value)
    with pytest.raises(ValueError, match="differs from raw episodes"):
        report._evaluation(path, 1000)


def test_short_horizon_cannot_pass_walking_gate(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    panel = _panel(improved=True, horizon=64)
    _write(path, dict(guided=panel, memory=panel))
    result, _ = report._evaluation(path, 64)
    assert result["guided"]["summary"]["walking_gate"] is None
    panel["summary"]["walking_gate"] = True
    _write(path, dict(guided=panel, memory=panel))
    with pytest.raises(ValueError, match="walking gate is unavailable"):
        report._evaluation(path, 64)


def test_refuses_nonfinite_or_decreasing_work(study: Path) -> None:
    path = study / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["work"]["candidate_controls"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Cumulative work decreased"):
        report._rows(study, 2, 1000)
