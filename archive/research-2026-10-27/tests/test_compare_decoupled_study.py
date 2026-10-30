"""Protocol, provenance, pairing and control-accounting checks for the pilot report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.diagnostics.compare_decoupled_study import ARMS, _ledger, _rates, _sha, main, validate


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _evaluation(steps: int, checkpoint_hash: str, milestone: int) -> dict[str, Any]:
    survived = steps == 1000
    episodes = [{"reset_seed": seed, "steps": steps, "survived": survived, "terminated": not survived,
                 "truncated": survived, "velocity_mae": .3, "training_return": -500.} for seed in range(8_600_000, 8_600_100)]
    return {"episodes": episodes, "run": {"steps": 1000, "requested_episodes": 100, "complete": True},
            "summary": {"episodes": 100, "survival_rate": float(survived), "mean_steps": steps,
                        "velocity_mae": .3, "training_return": -500.}, "checkpoint_sha256": checkpoint_hash,
            "completed_iterations": milestone, "task": {"target_velocity": 1.5}}


@pytest.fixture
def study(tmp_path: Path) -> Path:
    cfg = {"iterations": 15, "seed": 0, "K": 256, "H": 32, "tail_steps": 100,
           "eval_episodes": 100, "eval_steps": 1000, "eval_seed": 8_600_000,
           "eval_every": 5, "states_per_batch": 64, "max_label_states": 2000}
    initial = tmp_path / "policy_initial.pt"
    initial.write_bytes(b"initial network")
    source = tmp_path / "source" / "module.py"
    source.parent.mkdir(); source.write_text("# preserved source\n")
    _write(tmp_path / "protocol.json", {"config": cfg, "arms": ARMS,
                                       "source_sha256": {"module.py": _sha(source)},
                                       "initial_checkpoint_sha256": _sha(initial)})
    _write(tmp_path / "status.json", {"phase": "complete", "source_unchanged": True, "completed_iterations": 15})
    _write(tmp_path / "initial.json", _evaluation(60, _sha(initial), 0))
    for index, arm in enumerate(ARMS):
        directory = tmp_path / arm
        directory.mkdir()
        rows = []
        for k in range(15):
            checkpoint = directory / f"policy_{k:03d}.pt"
            checkpoint.write_bytes(f"{arm} network {k}".encode())
            length = 32 if index == 3 else 132
            logical = {"candidate_controls": 64 * 256 * length, "nominal_generation_controls": 64 * (32 + length),
                       "mean_verification_controls": 64 * length}
            skipped = index == 3 and k >= 2
            row = {"iteration": k, "completed_iterations": k + 1, "arm": arm, "label_states": 64,
                   "checkpoint_sha256": _sha(checkpoint), "parameters_before": "unchanged",
                   "parameters_after": "unchanged" if skipped else "updated", "m_step_applied_updates": 0 if skipped else 2000,
                   "diagnostics": {"bimodal_collapse_count": 0, "medoid_rescued_collapse_count": 0},
                   "current_weight_mass": 0. if skipped else 64.,
                   "value_gate": {"bootstrap_bypass": index == 3 and k == 0, "positive": 0, "evaluated": 64 if index == 3 else 0},
                   "ledger": {"logical": logical, "physical": dict(logical), "total_controls": sum(logical.values()),
                              "actor_evaluations": 999999, "world_capacities": {"single": 128},
                              "collection_controls": 100, "value_acquisition_controls": 200 if index == 3 else 0,
                              "evaluation_controls": 0}}
            if k + 1 in (5, 10, 15):
                evaluation = _evaluation(1000 - 50 * index, _sha(checkpoint), k + 1)
                _write(directory / "evaluations" / f"iteration_{k+1:03d}.json", evaluation)
                row["ledger"]["evaluation_controls"] = sum(e["steps"] for e in evaluation["episodes"])
            rows.append(row)
        (directory / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def test_rejects_incomplete_pilot(study: Path) -> None:
    path = study / ARMS[0] / "metrics.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="Incomplete"):
        validate(study)


def test_rejects_corrupt_checkpoint_and_saved_source(study: Path) -> None:
    checkpoint = study / ARMS[1] / "policy_010.pt"
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate(study)
    (study / "source/module.py").write_text("changed source")
    with pytest.raises(ValueError, match="source hash"):
        validate(study)


def test_rejects_wrong_evaluation_checkpoint(study: Path) -> None:
    path = study / ARMS[2] / "evaluations/iteration_010.json"
    report = json.loads(path.read_text())
    report["checkpoint_sha256"] = "wrong checkpoint"
    _write(path, report)
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate(study)


def test_rejects_unpaired_evaluation_seed_order(study: Path) -> None:
    path = study / ARMS[0] / "evaluations/iteration_005.json"
    report = json.loads(path.read_text())
    report["episodes"][0], report["episodes"][1] = report["episodes"][1], report["episodes"][0]
    _write(path, report)
    with pytest.raises(ValueError, match="pairing"):
        validate(study)


def test_rejects_candidate_budget_violation(study: Path) -> None:
    path = study / ARMS[-1] / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[4]["ledger"]["logical"]["candidate_controls"] = 67_584_001
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Candidate controls"):
        validate(study)


def test_no_eligible_rescues_and_disjoint_ledger_totals(study: Path) -> None:
    row = json.loads((study / ARMS[0] / "metrics.jsonl").read_text().splitlines()[0])
    assert _rates(row)["rescue"] is None
    counts = _ledger(row)
    assert counts["total_logical"] == sum(row["ledger"]["logical"].values()) + 100
    assert counts["total_physical"] == counts["total_logical"]


def test_writes_complete_boundary_safe_report(study: Path) -> None:
    main(study)
    report = json.loads((study / "comparison.json").read_text())
    assert report["training"][ARMS[-1]]["rescue_rate"] is None
    assert report["training"][ARMS[-1]]["bootstrap_bypass_iterations"] == 1
    assert report["training"][ARMS[-1]]["skipped_policy_updates"] == 13
    assert report["shared_initial_evaluation_controls"] == 6000
    assert "one training seed" in report["uncertainty"]
    for name in ("comparison.md", "learning_curves.png", "standalone_metrics.png"):
        assert (study / name).stat().st_size > 100
