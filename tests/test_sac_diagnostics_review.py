"""Saved-summary corrections must not require science runtimes or raw evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.gps.mppi_gps.reporting import sac_diagnostics_review as review


def summaries() -> dict:
    students = [
        dict(walks=walks, **dict.fromkeys(review.SCORES, score))
        for walks, score in [(0, 0.4), (0, 0.3), (16, 0.2), (32, 0.1)]
    ]
    return dict(
        student_positive_control=dict(
            table={"SAC / S1 128 tanh": dict(standing=[30, 32, 32])},
            readings={"pipeline_can_clone_a_walker": True},
        ),
        jitter_tolerance=dict(
            conditions=[
                dict(
                    jitter="white",
                    signal_share=0.9,
                    students=[dict(standing_walks=value) for value in [25, 0, 26]],
                    standing_walk_rate=51 / 96,
                )
            ],
            readings={},
        ),
        on_policy_disagreement=dict(
            students=students,
            spearman_with_walking=dict.fromkeys(review.SCORES, -0.5),
            readings={"on_policy_predicts_walking": False},
        ),
    )


def capsule(tmp_path: Path) -> Path:
    root = tmp_path / "capsule"
    files = {}
    for study, value in summaries().items():
        name = f"runs/2026-11-02_{study}_001/summary.json"
        path = root / name
        path.parent.mkdir(parents=True)
        payload = json.dumps(value).encode()
        path.write_bytes(payload)
        files[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (root / "capsule.json").write_text(json.dumps(dict(files=files)))
    return root


def test_tied_rank_correlation_is_order_invariant() -> None:
    assert review.average_ranks([3.0, 1.0, 1.0, 9.0]) == [2.0, 0.5, 0.5, 3.0]
    x, y = [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 2.0]
    expected = 0.9486832980505138
    assert review.spearman(x, y) == pytest.approx(expected)
    assert review.spearman(x[::-1], y[::-1]) == pytest.approx(expected)
    assert review.spearman([1.0, 1.0], [0.0, 1.0]) is None
    assert review.spearman([], []) is None
    with pytest.raises(ValueError, match="finite"):
        review.average_ranks([float("nan")])


def test_pairwise_auc_half_credit_and_missing_class() -> None:
    assert review.auc_lower_is_better([0.0, 1.0, 1.0, 2.0], [True, True, False, False]) == 0.875
    assert review.auc_lower_is_better([1.0, 1.0], [True, False]) == 0.5
    assert review.auc_lower_is_better([0.0, 1.0], [False, False]) is None
    with pytest.raises(ValueError, match="equal length"):
        review.auc_lower_is_better([0.0], [])


def test_outcomes_and_original_gate_are_distinct() -> None:
    result = review.analyze(summaries())
    assert result["positive_control_outcomes"]["SAC / S1 128 tanh"]["walks"] == 94
    assert result["jitter_outcomes"][0]["walks"] == 51
    assert result["auc_population"]["positive_students"] == 2
    assert result["on_policy_statistics"]["on_policy_standing"]["auc_any_walking"] == 1.0
    assert result["corrected_original_on_policy_gate"]["passes"] is False
    assert result["original_summary_statistics"]["on_policy_spearman"]["on_policy_standing"] == -0.5
    damaged = summaries()
    damaged["jitter_tolerance"]["conditions"][0]["standing_walk_rate"] = 0
    with pytest.raises(ValueError, match="aggregate"):
        review.analyze(damaged)


def test_hash_binding_and_create_only_outputs(tmp_path: Path) -> None:
    root = capsule(tmp_path)
    output = tmp_path / "review"
    receipt = review.run_review(root, output)
    assert all(value == 0 for value in receipt["new_scientific_work"].values())
    assert len(receipt["inputs"]) == 3
    with pytest.raises(FileExistsError):
        review.run_review(root, output)
    path = root / "runs/2026-11-02_jitter_tolerance_001/summary.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash/size"):
        review.run_review(root, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_cli_uses_only_saved_json_without_simulator_or_torch(tmp_path: Path) -> None:
    root = capsule(tmp_path)
    output = tmp_path / "cli"
    program = (
        "import sys; from src.gps.mppi_gps.reporting.sac_diagnostics_review import main; "
        f"main(['--capsule',{str(root)!r},'--output',{str(output)!r}]); "
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert json.loads(completed.stdout)["status"] == "PASSED_SAVED_SUMMARY_REANALYSIS_ONLY"
