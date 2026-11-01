"""Synthetic D1 checks: no inference, optimization or native simulation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.gps.mppi_gps.transfer_inputs import SCHEMA, _npz
from src.gps.mppi_gps.transfer_metrics import analyze_inputs, distance_summary, physical_trace


def test_physical_trace_respects_time_column_and_clipping_boundary():
    states = np.zeros((3, 19))
    states[:, 0] = [0, 100, 200]
    states[:, 2] = [1, 1.1, 1.2]
    states[:, 3] = [-0.2, 0, 0.3]
    states[:, 10] = [10, 10.1, -11]
    result = physical_trace(states, 3)
    np.testing.assert_array_equal(result["height"], states[:, 2])
    np.testing.assert_array_equal(result["forward_velocity"], states[:, 10])
    np.testing.assert_array_equal(result["clipped_count"], [0, 1, 1])


def test_support_threshold_uses_strict_greater_and_actual_short_trace():
    result = distance_summary(np.array([0.0, 1.0, 2.0]), 1.0)
    assert result["outside_fraction"] == pytest.approx(1 / 3)
    assert result["first100_outside_fraction"] == pytest.approx(1 / 3)
    assert result["last25_outside_fraction"] == pytest.approx(1 / 3)
    assert result["median_distance"] == 1
    with pytest.raises(ValueError, match="nonempty"):
        distance_summary(np.empty(0), 1)


class SavedMeter:
    def __init__(self):
        self.requests = []
        self.arrays = {}

    def support_queries(self, rows):
        self.requests.append(rows)

    def save_arrays(self, tag, **arrays):
        assert tag not in self.arrays
        self.arrays[tag] = arrays
        return {"tag": tag}


def test_small_saved_lifecycle_charges_queries_once_and_keeps_trajectory_means(tmp_path):
    files = {}

    def put(name, values):
        files[name] = _npz(tmp_path / name, values)

    def pairs(heights, names):
        states = np.zeros((len(heights), 19))
        states[:, 2] = heights
        return dict(
            states=states,
            observations=np.concatenate((states[:, 2:10], states[:, 10:19]), axis=1),
            actions=np.zeros((len(states), 6)),
            row_episode=np.asarray(names),
            row_step=np.tile([0, 1], len(states) // 2),
        )

    train = pairs([1, 1.1, 1.5, 1.6], ["discovery_00"] * 2 + ["discovery_01"] * 2)
    validation = pairs([1.2, 1.3, 1.7, 1.8], ["held_0"] * 2 + ["held_1"] * 2)
    put("train.npz", train)
    put("validation.npz", validation)
    put("normalization.npz", {"mean": np.zeros(17), "std": np.ones(17)})
    teachers = []
    for index in range(2):
        mask = validation["row_episode"] == f"held_{index}"
        name = f"teachers/held_{index}.npz"
        put(name, {key: value[mask] for key, value in validation.items()})
        teachers.append(
            {
                "id": f"held_{index}",
                "file": name,
                "rows": 2,
                "validation_episode": f"held_{index}",
                "outcome": {},
            }
        )
    trajectories = []
    for index in range(2):
        identifier = f"discovery_{index:02d}"
        mask = train["row_episode"] == identifier
        name, predictions = f"k1/{identifier}.npz", f"k1/{identifier}_predictions.npz"
        values = {key: value[mask] for key, value in train.items()}
        put(name, values)
        put(
            predictions,
            {"predictions": np.full((2, 1, 6), index), "policy_names": np.array(["reference"])},
        )
        trajectories.append(
            {
                "id": identifier,
                "group": "discovery",
                "file": name,
                "predictions_file": predictions,
                "rows": 2,
                "outcome": {},
            }
        )
    put("episode.npz", pairs([1, 1.2], ["episode"] * 2))
    put(
        "point.npz",
        {"train_predictions": np.ones((4, 6)), "validation_predictions": np.ones((4, 6))},
    )
    inventory = {
        "schema": SCHEMA,
        "files": files,
        "teachers": teachers,
        "w2_episodes": [
            {
                "tag": "one",
                "point": "s0_u2000",
                "pair": 0,
                "update": 2000,
                "panel": "primary",
                "steps": 2,
                "file": "episode.npz",
                "outcome": {},
            }
        ],
        "k1_trajectories": trajectories,
        "w2_points": [{"id": "s0_u2000", "predictions_file": "point.npz"}],
    }
    (tmp_path / "inventory.json").write_text(json.dumps(inventory))
    meter = SavedMeter()
    result = analyze_inputs(tmp_path, meter, {"expected": {"support_queries": 14}})
    assert result["query_count"] == sum(meter.requests) == 14
    assert meter.requests == [4, 2, 2, 2, 2, 2]
    assert (
        result["new_forward_rows"]
        == result["new_physics_controls"]
        == result["new_optimizer_steps"]
        == 0
    )
    assert result["k1_group_matrix"][0]["mse"] == 0.5
    assert len(result["teacher_reconstruction"]) == 4
    episode = next(row for row in result["coverage_records"] if row["kind"] == "w2_episode")
    np.testing.assert_array_equal(meter.arrays[episode["raw"]["tag"]]["time_to_end"], [1, 0])
    originals = [row for row in result["coverage_records"] if row["kind"] == "k1_trajectory"]
    assert all(
        row["distance_reference"] == "in_sample_training_self_matches_not_generalization"
        for row in originals
    )
