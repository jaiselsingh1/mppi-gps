"""Known synthetic arrays/checkpoints exercise W1 proofs without neural forwards."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.gps.mppi_gps.audits import native
from src.gps.mppi_gps.audits.compatibility import recompute_agreement
from src.gps.mppi_gps.audits.width import (
    WidthAudit,
    audit_prediction_metrics,
    audit_sampler,
    decode_checkpoint,
    recompute_contrast,
)


@pytest.mark.parametrize("width", [128, 256])
def test_width_specific_checkpoint_tensor_hash_and_shape_proofs(tmp_path, width):
    norm = dict(mean=np.zeros(17, dtype=np.float32), std=np.ones(17, dtype=np.float32))
    shapes = {
        "observation_mean": (17,),
        "observation_std": (17,),
        "network.0.weight": (width, 17),
        "network.0.bias": (width,),
        "network.2.weight": (width, width),
        "network.2.bias": (width,),
        "network.4.weight": (6, width),
        "network.4.bias": (6,),
    }
    state = {key: torch.zeros(shape, dtype=torch.float32) for key, shape in shapes.items()}
    state["observation_std"].fill_(1)
    payload = dict(
        schema="walker-tanh-policy-v1",
        obs_dim=17,
        policy_hash=native.tensor_hash(state),
        state_dict=state,
    )
    if width == 256:
        payload["hidden_dim"] = width
    path = tmp_path / "synthetic.pt"
    torch.save(payload, path)
    a = WidthAudit(tmp_path, tmp_path)
    assert decode_checkpoint(a, path, width, norm)["policy_hash"] == native.tensor_hash(state)
    bad = deepcopy(payload)
    bad["state_dict"]["network.2.weight"] = torch.zeros(width, 17)
    bad["policy_hash"] = native.tensor_hash(bad["state_dict"])
    changed = tmp_path / "malformed.pt"
    torch.save(bad, changed)
    with pytest.raises(AssertionError, match="width-specific tensor"):
        decode_checkpoint(a, changed, width, norm)
    bad = deepcopy(payload)
    bad["state_dict"]["network.4.bias"][0] = 0.5
    changed_hash = tmp_path / "wrong_owner.pt"
    torch.save(bad, changed_hash)
    with pytest.raises(AssertionError, match="named-tensor"):
        decode_checkpoint(a, changed_hash, width, norm)


def test_all_shared_sampler_indices_and_endpoint_states_are_checked(tmp_path):
    rng = np.random.Generator(np.random.PCG64(14))
    states = {"0": deepcopy(rng.bit_generator.state)}
    rows = []
    for update in range(1, 13):
        rows.append(rng.integers(0, 7, size=3))
        if update in (4, 12):
            states[str(update)] = deepcopy(rng.bit_generator.state)
    indices = np.stack(rows)
    settings = dict(seed=14, rows=7, batch_size=3, updates=12, checkpoints=[0, 4, 12])
    audit_sampler(WidthAudit(tmp_path, tmp_path), indices, states, **settings)
    wrong = indices.copy()
    wrong[8, 2] = (wrong[8, 2] + 1) % 7
    with pytest.raises(AssertionError, match="PCG64 minibatch"):
        audit_sampler(WidthAudit(tmp_path, tmp_path), wrong, states, **settings)
    changed_state = deepcopy(states)
    changed_state["4"]["state"]["state"] += 1
    with pytest.raises(AssertionError, match="endpoint"):
        audit_sampler(WidthAudit(tmp_path, tmp_path), indices, changed_state, **settings)


def test_saved_prediction_metrics_use_every_actuator_and_trajectory(tmp_path):
    data, predictions, metrics = {}, {}, {}
    for split in ("train", "validation"):
        targets = np.full((6, 6), 0.5)
        prediction = np.zeros((6, 6), dtype=np.float32)
        prediction[:, 0] = 0.5
        tags = np.array(["a"] * 2 + ["b"] * 4)
        data[split] = dict(actions=targets, row_episode=tags)
        predictions[split + "_predictions"] = prediction
        metrics[split] = dict(
            mse=5 / 24,
            per_trajectory={
                tag: recompute_agreement(
                    prediction[tags == tag],
                    targets[tags == tag],
                    dict(early_controls=100, bin_controls=100, saturation_threshold=0.95),
                )
                for tag in ("a", "b")
            },
        )
    a = WidthAudit(tmp_path, tmp_path)
    assert audit_prediction_metrics(a, dict(metrics=metrics), data, predictions) == 12
    changed = deepcopy(metrics)
    changed["train"]["per_trajectory"]["b"]["per_actuator_mse"][0] = 0.25
    with pytest.raises(AssertionError, match="trajectory agreement"):
        audit_prediction_metrics(a, dict(metrics=changed), data, predictions)


def contrast_fixture():
    config = dict(
        seed_pairs=[{}, {}, {}],
        updates=10000,
        gates=dict(
            minimum_train_mse_gain=0.1,
            minimum_fitting_pairs=2,
            minimum_cost_gain=0.05,
            minimum_cost_pairs=2,
            minimum_walking_gain=2,
            minimum_walking_pairs=2,
        ),
    )
    points, evaluations = {}, {}
    for pair in range(3):
        for width in (128, 256):
            name = f"s{pair}_w{width}_u10000"
            points[name] = dict(
                metrics={
                    split: dict(mse=1.0 if width == 128 else 0.8)
                    for split in ("train", "validation")
                }
            )
            evaluations[name] = dict(
                summary=dict(
                    mean_cost=100 if width == 128 else 90,
                    successes=2 if width == 128 else 4,
                    survived=6,
                    mean_steps=700,
                )
            )
    return points, evaluations, config


def test_reconstruction_validation_cost_and_walking_are_distinct_frozen_screens():
    points, evaluations, cfg = contrast_fixture()
    output = recompute_contrast(points, evaluations, cfg)
    assert output["reconstruction_screen"] and output["validation_guard"]
    assert output["fitting_screen"] and output["task_screen"]
    for point in points.values():
        if point["metrics"]["train"]["mse"] == 0.8:
            point["metrics"]["validation"]["mse"] = 1.2
    output = recompute_contrast(points, evaluations, cfg)
    assert output["reconstruction_screen"] and not output["validation_guard"]
    assert not output["fitting_screen"] and output["task_screen"]
    for row in evaluations.values():
        row["summary"]["successes"] = 0
    output = recompute_contrast(points, evaluations, cfg)
    assert output["task_checks"]["mean_cost_gain"]
    assert not output["task_checks"]["walking_pairs"] and not output["task_screen"]
    assert not recompute_contrast(points, {}, cfg)["task_screen"]


def test_cli_failure_receipt_is_create_only_and_imports_are_independent(tmp_path):
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
from src.gps.mppi_gps.audits.width import recompute_contrast
assert not any(name.startswith(('mujoco', 'src.envs', 'src.gps.mppi_gps.policy',
 'src.gps.mppi_gps.experiments', 'src.gps.mppi_gps.compatibility_metrics',
 'src.gps.mppi_gps.width_inputs')) for name in sys.modules)
"""
    imported = subprocess.run(
        [sys.executable, "-B", "-c", program], cwd=root, text=True, capture_output=True, timeout=30
    )
    assert imported.returncode == 0, imported.stderr
    out = tmp_path / "audit"
    command = [
        sys.executable,
        "-B",
        "scripts/audits/width.py",
        "--run",
        str(tmp_path / "missing"),
        "--out",
        str(out),
    ]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 1
    receipt = json.loads((out / "receipt.json").read_text())
    assert not receipt["passed"] and receipt["new_neural_rows"] == receipt["new_controls"] == 0
    assert (out / "findings.json").is_file()
    again = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=30)
    assert again.returncode != 0 and "FileExistsError" in again.stderr
