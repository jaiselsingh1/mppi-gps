"""Offline design/data checks; no fitted network or simulator is executed."""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import diagnostics


def example():
    observed = np.arange(68, dtype=float).reshape(4, 17) / 100.0
    previous = np.array([[0.0] * 6, [0.2] * 6, [-0.1] * 6, [0.4] * 6])
    experience = dict(
        observations=observed, actions=np.full((4, 6), 0.3), previous_actions=previous
    )
    normal = dict(
        mean=np.arange(17, dtype=np.float32) / 10.0, std=np.full(17, 0.7, dtype=np.float32)
    )
    return experience, normal


def test_capacity_control_and_history_share_original_features_targets_and_normalization():
    experience, normal = example()
    data = diagnostics._datasets(experience, normal)
    np.testing.assert_array_equal(data["state"], data["padded"][:, :17])
    np.testing.assert_array_equal(data["state"], data["history"][:, :17])
    np.testing.assert_array_equal(data["padded"][:, 17:], np.zeros((4, 6)))
    np.testing.assert_array_equal(data["history"][:, 17:], experience["previous_actions"])
    np.testing.assert_array_equal(data["actions"], experience["actions"])
    np.testing.assert_array_equal(data["normalization23"]["mean"][:17], normal["mean"])
    np.testing.assert_array_equal(data["normalization23"]["std"][:17], normal["std"])
    np.testing.assert_array_equal(data["normalization23"]["mean"][17:], np.zeros(6))
    np.testing.assert_allclose(
        data["normalization23"]["std"][17:], experience["previous_actions"].std(axis=0)
    )


def test_constant_history_uses_declared_floor_and_construction_does_not_mutate_inputs():
    experience, normal = example()
    experience["previous_actions"][:] = 0.0
    original = experience["observations"].copy()
    data = diagnostics._datasets(experience, normal)
    np.testing.assert_array_equal(data["normalization23"]["std"][17:], np.full(6, 0.05))
    data["state"][:] = 100.0
    data["history"][:] = -100.0
    data["actions"][:] = -1.0
    np.testing.assert_array_equal(experience["observations"], original)
    np.testing.assert_array_equal(experience["actions"], np.full((4, 6), 0.3))


@pytest.mark.parametrize(
    "key,value",
    [
        ("observations", np.zeros((4, 18))),
        ("actions", np.zeros((3, 6))),
        ("previous_actions", np.full((4, 6), np.nan)),
        ("actions", np.full((4, 6), 1.01)),
        ("previous_actions", np.full((4, 6), -1.01)),
    ],
)
def test_invalid_or_misaligned_teacher_features_are_rejected(key, value):
    experience, normal = example()
    experience[key] = value
    with pytest.raises(ValueError):
        diagnostics._datasets(experience, normal)


def test_work_roster_charges_all_optimizer_and_full_dataset_measurement_rows():
    work = diagnostics.expected_work(4064)
    assert work["training_forward_rows"] == work["completed_backward_rows"] == 15360000
    assert work["full_dataset_mse_rows"] == 73152
    assert work["completed_neural_rows"] == 15433152
    assert work["completed_optimizer_steps"] == 60000
    assert work["requested_controls"] == work["completed_controls"] == 0
    with pytest.raises(ValueError):
        diagnostics.expected_work(True)


def test_offline_module_has_no_simulator_dependency_and_rejects_a_loaded_engine(monkeypatch):
    tree = ast.parse(Path(diagnostics.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [
        item.name for node in ast.walk(tree) if isinstance(node, ast.Import) for item in node.names
    ]
    assert not any(
        name.startswith(("mujoco", "warp", "src.envs", "envs", "loop")) for name in imports
    )
    monkeypatch.setitem(sys.modules, "mujoco.offline_guard_sentinel", object())
    with pytest.raises(RuntimeError, match="before any MuJoCo"):
        diagnostics._offline_only()
