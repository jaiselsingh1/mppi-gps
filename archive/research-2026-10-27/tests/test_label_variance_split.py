"""Label-variance split: the decomposition, and the replay that regenerates warm starts."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.diagnostics.label_variance_split import (SplitConfig, _planner_cfg, _seeds, query, record_episode,
                                                       regenerate_warm_starts, split_variance)
from scripts.training.train_walker_mdgps import _new_actor, _walker, policy_state

_SMALL = SplitConfig(K=4, H=6, steps=10, episodes=1, anchors=3, warm_starts=2, queries=2, workers=1)


def test_split_recovers_known_components():
    rng = np.random.default_rng(0)
    N, W, R, nu = 400, 40, 40, 2
    labels = (rng.normal(0.0, 1.0, (N, 1, 1, nu)) + rng.normal(0.0, 0.5, (N, W, 1, nu))
              + rng.normal(0.0, 0.2, (N, W, R, nu)))
    parts = split_variance(labels)
    np.testing.assert_allclose(parts["jitter"].mean(), 0.04, rtol=0.05)
    np.testing.assert_allclose(parts["warm_start"].mean(), 0.25, rtol=0.05)
    np.testing.assert_allclose(parts["floor"].mean(), 0.29, rtol=0.05)


def test_replay_with_the_recorded_seed_reproduces_plans_and_labels():
    from src.mppi.mppi import MPPI

    data = record_episode(_SMALL, "teacher", 0)
    times = list(range(data["steps"]))
    snapshots = regenerate_warm_starts(_SMALL, data, times, _seeds(_SMALL, "teacher", 0)["planner"])
    planner = MPPI(_walker(), _planner_cfg(_SMALL), seed=0)
    for t, snapshot in zip(times, snapshots):
        np.testing.assert_array_equal(snapshot["U"], data["plans"][t])
        label = query(_SMALL, planner, data["states"][t], data["warmstarts"][t], snapshot)["executed"]
        np.testing.assert_array_equal(label, data["labels"][t])


@pytest.mark.parametrize("population", ["teacher", "student"])
def test_other_seeds_change_the_plans_not_the_states(population, tmp_path):
    cfg = _SMALL
    if population == "student":
        torch.save(policy_state(_new_actor(0)), tmp_path / "policy.pt")
        cfg = replace(_SMALL, student=str(tmp_path / "policy.pt"))
    data = record_episode(cfg, population, 0)
    last = data["steps"] - 1
    # regenerate_warm_starts raises if any replayed state differs from the recording.
    first, final = regenerate_warm_starts(cfg, data, [0, last], seed=123)
    np.testing.assert_array_equal(first["U"], data["plans"][0])
    assert not np.array_equal(final["U"], data["plans"][last])


def test_teacher_executes_its_labels_and_student_its_policy(tmp_path):
    from src.rl.sac import load_policy

    quiet = replace(_SMALL, exec_noise=0.0)
    teacher = record_episode(quiet, "teacher", 0)
    np.testing.assert_array_equal(teacher["executed"], teacher["labels"])
    actor = _new_actor(0)
    torch.save(policy_state(actor), tmp_path / "policy.pt")
    student = record_episode(replace(quiet, student=str(tmp_path / "policy.pt")), "student", 0)
    policy = load_policy(policy_state(actor))
    np.testing.assert_allclose(student["executed"], [policy.act(o) for o in student["obs"]])
    assert not np.allclose(student["executed"], student["labels"])
