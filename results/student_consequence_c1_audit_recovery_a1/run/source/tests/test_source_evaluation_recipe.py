"""S1.1 executes the reviewed science and reuses the exact closed S1 fits."""

import json
import subprocess
import sys
from pathlib import Path

from src.gps.mppi_gps import lifecycle
from src.gps.mppi_gps.recipes import load_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_source_manifest_does_not_claim_new_fitting(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "README.md").write_text("Synthetic frozen source.\n")
    monkeypatch.setattr(lifecycle, "ROOT", checkout)
    out = lifecycle.freeze(tmp_path / "evaluation", {}, "student-source-evaluation-s1.1")
    assert json.loads((out / "manifest.json").read_text())["policy_action_fitting"] is False
    old = lifecycle.freeze(tmp_path / "existing_fit", {}, "student-source-s1")
    assert json.loads((old / "manifest.json").read_text())["policy_action_fitting"] is True


def test_execution_preserves_all_proposed_scientific_values():
    proposal = json.loads((ROOT / "configs/studies/student_source_s1_1_proposal.json").read_text())
    actual = load_recipe("student_source_evaluation")
    for key in (
        "checkpoints",
        "reference",
        "platform",
        "task",
        "episode_steps",
        "evaluation_order",
        "primary_seeds",
        "confirmation_seeds",
        "walking_screen",
        "competence",
        "bootstrap",
        "budgets",
        "reconstruction",
        "historical_episode_role",
        "closed_fit_reuse",
    ):
        assert actual[key] == proposal[key], key
    original = load_recipe("student_source")
    for key in (
        "learning",
        "seed_pairs",
        "fit_arms",
        "training_rows_per_arm",
        "held_rows_per_arm",
        "feature_map",
        "normalization",
        "policy",
    ):
        assert actual[key] == original[key], key
    assert set(actual["checkpoints"]) == set(actual["evaluation_order"]) - {"reference"}
    assert len(set(actual["primary_seeds"] + actual["confirmation_seeds"])) == 32


def test_public_evaluation_cli_help_is_model_and_simulator_free():
    code = (
        "import runpy,sys;sys.argv=['scripts/train_mppi_gps.py','study','source-evaluation','--help'];"
        "\ntry:runpy.run_path(sys.argv[0],run_name='__main__')"
        "\nexcept SystemExit as e:assert e.code==0"
        "\nassert not any(n=='torch' or n=='mujoco' or n.startswith('src.envs') for n in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "prepare" in result.stdout and "worker" in result.stdout
