"""Step 25: the raw arm is Step 18's runner, and memoryless MPPI forgets its plan."""
import numpy as np

from scripts.diagnostics.context_rescore_check import _episode
from scripts.diagnostics.memoryless_check import run_episode, single_start_components


def test_raw_arm_matches_the_step18_runner():
    job = {"K": 8, "H": 6, "context": "none", "rule": None, "context_path": None, "tail_steps": 0, "steps": 30,
           "reset_seed": 70_003, "planner_seed": 95_003}
    step18 = _episode(job)
    raw = run_episode(8, 2, False, 70_003, 95_003, 30, horizon=6)
    for key in ("steps", "survived", "mean_vx", "velocity_mae"):
        assert raw[key] == step18[key]


def test_memoryless_matches_raw_only_at_the_first_decision():
    first_raw = run_episode(8, 2, False, 70_003, 95_003, 1, horizon=6)
    first_memoryless = run_episode(8, 2, True, 70_003, 95_003, 1, horizon=6)
    assert first_memoryless["mean_vx"] == first_raw["mean_vx"]
    raw = run_episode(8, 2, False, 70_003, 95_003, 20, horizon=6)
    memoryless = run_episode(8, 2, True, 70_003, 95_003, 20, horizon=6)
    assert memoryless["mean_vx"] != raw["mean_vx"]


def test_single_start_components_recover_known_parts():
    rng = np.random.default_rng(0)
    labels = rng.normal(0.0, 0.5, (2000, 1, 3)) + rng.normal(0.0, 0.2, (2000, 16, 3))
    parts = single_start_components(labels)
    np.testing.assert_allclose(parts["floor"], 0.04, rtol=0.05)
    np.testing.assert_allclose(parts["state"], 0.25, rtol=0.08)
