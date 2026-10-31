"""W2 audit tests use known saved dictionaries; no neural or simulator work."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.gps.mppi_gps.audits import native
from src.gps.mppi_gps.audits.update_budget import (
    UpdateBudgetAudit,
    audit_reload,
    recompute_contrast,
)


def contrast_fixture():
    config = dict(seed_pairs=[{}, {}, {}])
    points, evaluations = {}, {}
    for pair in range(3):
        for update in (2000, 10000):
            name = f"s{pair}_u{update}"
            early = update == 2000
            points[name] = dict(
                metrics=dict(
                    train=dict(mse=1.2 if early else 1), validation=dict(mse=0.8 if early else 1)
                )
            )
            evaluations[name] = dict(
                summary=dict(
                    mean_cost=110 if early else 100,
                    mean_steps=120 if early else 100,
                    successes=0,
                    survived=0,
                )
            )
    return points, evaluations, config


def test_validation_retention_cost_and_walking_have_separate_claims():
    points, evaluations, config = contrast_fixture()
    result = recompute_contrast(points, evaluations, config)
    assert result["validation_screen"] and result["retention_screen"]
    assert not result["cost_screen"] and not result["walking_screen"]
    assert result["pairs"][0]["relative_train_mse_gain"] == pytest.approx(-0.2)
    assert result["pairs"][0]["relative_validation_mse_gain"] == pytest.approx(0.2)
    assert result["pairs"][0]["relative_controls_gain"] == pytest.approx(0.2)
    assert result["pairs"][0]["relative_cost_gain"] == pytest.approx(-0.1)
    assert result["pairs"][0]["controls_delta"] == 20 and result["pairs"][0]["cost_delta"] == 10
    assert "task_screen" not in result
    for pair in range(3):
        evaluations[f"s{pair}_u2000"]["summary"].update(mean_cost=90, successes=2, survived=2)
    result = recompute_contrast(points, evaluations, config)
    assert all(result[key + "_screen"] for key in ("validation", "retention", "cost", "walking"))
    evaluations["s0_u10000"]["summary"]["survived"] = 3
    result = recompute_contrast(points, evaluations, config)
    assert result["validation_screen"]
    assert not any(result[key + "_screen"] for key in ("retention", "cost", "walking"))


def test_retention_does_not_require_cost_but_cost_requires_control_nonregression():
    points, evaluations, config = contrast_fixture()
    for pair in range(3):
        evaluations[f"s{pair}_u2000"]["summary"].update(mean_cost=90, mean_steps=150)
    evaluations["s0_u2000"]["summary"]["mean_steps"] = 99
    result = recompute_contrast(points, evaluations, config)
    assert result["retention_screen"]
    assert not result["cost_screen"]
    assert not result["cost_checks"]["all_pair_controls_nonregression"]


@pytest.mark.parametrize(
    "field", ["policy_hash", "optimizer_fingerprint", "private_rng_fingerprint"]
)
def test_saved_reload_requires_exact_policy_adam_and_private_rng_identity(tmp_path, field):
    payload = dict(
        policy_hash="a" * 64,
        optimizer_state={"updates": 2000},
        private_rng_state=torch.tensor([1, 2, 3], dtype=torch.uint8),
    )
    expected = dict(
        policy_hash=payload["policy_hash"],
        optimizer_fingerprint=native.fingerprint(payload["optimizer_state"]),
        private_rng_fingerprint=native.fingerprint(payload["private_rng_state"]),
    )
    binding = dict(file="checkpoints/s0_u2000.pt", sha256="b" * 64)
    receipt = dict(
        pair=0,
        point="s0_u2000",
        file=binding["file"],
        checkpoint_sha256=binding["sha256"],
        before=expected,
        after=deepcopy(expected),
    )
    audit_reload(UpdateBudgetAudit(tmp_path, tmp_path), receipt, payload, binding, 0)
    changed = deepcopy(receipt)
    changed["after"][field] = "c" * 64
    with pytest.raises(AssertionError, match="checkpoint load"):
        audit_reload(UpdateBudgetAudit(tmp_path, tmp_path), changed, payload, binding, 0)


def test_audit_cli_import_independence_failure_receipt_and_create_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
from src.gps.mppi_gps.audits.update_budget import recompute_contrast
assert not any(name.startswith(('mujoco', 'src.envs', 'src.gps.mppi_gps.policy',
 'src.gps.mppi_gps.experiments', 'src.gps.mppi_gps.compatibility_metrics',
 'src.gps.mppi_gps.update_inputs')) for name in sys.modules)
"""
    imported = subprocess.run(
        [sys.executable, "-B", "-c", program], cwd=root, capture_output=True, text=True, timeout=30
    )
    assert imported.returncode == 0, imported.stderr
    out = tmp_path / "audit"
    command = [
        sys.executable,
        "-B",
        "scripts/audits/update_budget.py",
        "--run",
        str(tmp_path / "missing"),
        "--out",
        str(out),
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 1
    receipt = json.loads((out / "receipt.json").read_text())
    assert not receipt["passed"] and receipt["new_controls"] == receipt["new_neural_rows"] == 0
    again = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert again.returncode != 0 and "FileExistsError" in again.stderr
