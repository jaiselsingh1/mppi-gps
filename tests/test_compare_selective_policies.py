"""Check the final-policy comparison's pairing and checkpoint safeguards."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics.compare_selective_policies import _bootstrap, _evaluation, _training, main


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def study(tmp_path: Path) -> Path:
    config = {"iterations": 75, "seed": 0, "episodes_per_iter": 10, "teacher_runs": 1,
              "steps": 400, "eval_episodes": 10}
    for mode in ("anchor", "selective"):
        directory = tmp_path / mode
        _write_json(directory / "config.json", {**config, "selection_mode": mode})
        _write_json(directory / "status.json", {"complete": True})
        rows = [{"iteration": iteration, "selection_mode": mode, "fit_loss": .05,
                 "collect_mean_steps": 300, "collect_completed": 5, "mean_n_eff": 51.2,
                 "sigma_k": .2, "share_plans_worse": .3, "positive_weight_samples": 70,
                 "labels": 100, "seconds": 2,
                 **({"eval_dev": {"survival_rate": .5, "velocity_mae": .4, "training_return": 200}}
                    if (iteration + 1) % 5 == 0 else {})} for iteration in range(75)]
        (directory / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for mode, offset in (("initial", 0), ("anchor", 100), ("selective", 200)):
        episodes = [{"reset_seed": seed, "steps": 200 + offset, "survived": False,
                     "terminated": True, "truncated": False, "velocity_mae": .5 - offset / 1000,
                     "training_return": 50 + offset} for seed in range(91_000, 91_010)]
        _write_json(tmp_path / "evaluation" / f"{mode}.json", {
            "episodes": episodes,
            "summary": {"survival_rate": 0., "mean_steps": 200 + offset,
                        "velocity_mae": .5 - offset / 1000, "training_return": 50 + offset, "episodes": 10},
            "run": {"requested_episodes": 10, "complete": True, "steps": 1000, "reset_seed_base": 91_000},
            "checkpoint": f"/remote/{mode}/{'initial.pt' if mode == 'initial' else 'policy_074.pt'}", "task": {}})
    return tmp_path


def test_bootstrap_retains_pairwise_constant_gain() -> None:
    # Large reset offsets disappear only if candidate/reference stay paired.
    reference = np.arange(20) * 50.
    interval = _bootstrap(reference + 3. - reference)
    assert interval["mean"] == interval["lower"] == interval["upper"] == 3.
    assert interval["n_resamples"] == 10_000


def test_refuses_best_checkpoint(study: Path) -> None:
    path = study / "evaluation" / "anchor.json"
    report = json.loads(path.read_text())
    report["checkpoint"] = "/remote/anchor/policy_best.pt"
    _write_json(path, report)
    with pytest.raises(ValueError, match="policy_074.pt"):
        _evaluation(path, "anchor")


def test_refuses_other_arm_checkpoint(study: Path) -> None:
    path = study / "evaluation" / "selective.json"
    report = json.loads(path.read_text())
    report["checkpoint"] = "/remote/anchor/policy_074.pt"
    _write_json(path, report)
    with pytest.raises(ValueError, match="other-arm checkpoint"):
        _evaluation(path, "selective")


def test_refuses_incomplete_training(study: Path) -> None:
    path = study / "anchor" / "metrics.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="complete 75-iteration"):
        _training(study)


def test_refuses_unpaired_reports(study: Path) -> None:
    path = study / "evaluation" / "selective.json"
    report = json.loads(path.read_text())
    for episode in report["episodes"]:
        episode["reset_seed"] += 10
    report["run"]["reset_seed_base"] += 10
    _write_json(path, report)
    with pytest.raises(ValueError, match="seed lists must match exactly"):
        main(study)


def test_writes_complete_comparison_artifacts(study: Path) -> None:
    # Wilson's nominal zero lower bound for 0/100 can round slightly positive.
    # This is the real final-study boundary that previously broke the plot.
    for mode in ("initial", "anchor", "selective"):
        path = study / "evaluation" / f"{mode}.json"
        report = json.loads(path.read_text())
        prototype = report["episodes"][0]
        report["episodes"] = [{**prototype, "reset_seed": seed} for seed in range(91_000, 91_100)]
        report["summary"]["episodes"] = report["run"]["requested_episodes"] = 100
        _write_json(path, report)
    main(study)
    report = json.loads((study / "comparison.json").read_text())
    pair = report["comparisons"]["selective_minus_anchor"]["metrics"]
    assert pair["mean_steps"]["mean"] == pair["mean_steps"]["lower"] == pair["mean_steps"]["upper"] == 100.
    assert pair["mean_time_seconds"]["mean"] == pytest.approx(.8)
    assert pair["survival_rate"]["score_interval"]["upper"] > 0
    assert "single training seed" in report["protocol"]["uncertainty_scope"]
    assert not report["working_gate"]["outcomes"]["selective"]["passed"]
    for name in ("comparison.md", "training_curves.png", "final_metrics.png"):
        assert (study / name).stat().st_size > 100
