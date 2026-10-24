"""Step 29: the matching term stays on through recording, replays and queries."""
import numpy as np
import torch

from scripts.diagnostics.matched_label_floor import labels, walk
from scripts.diagnostics.walker_headroom import _episode as raw_mppi_episode
from scripts.training.train_walker_mdgps import _new_actor, policy_state


def _job(tmp_path, alpha: float) -> dict:
    path = tmp_path / "student.pt"
    if not path.exists():
        torch.save(policy_state(_new_actor(0)), path)
    return {"student": str(path), "K": 8, "H": 6, "alpha": alpha, "episode": 0, "steps": 12, "exec_noise": 0.1,
            "anchors": 3, "warm_starts": 2, "queries": 2, "reset_seed": 8_000_000, "planner_seed": 8_100_000,
            "noise_seed": 8_200_000, "regen_seed": 8_300_000, "anchor_seed": 8_400_000, "query_seed": 9_000_000}


def test_labels_replay_the_recorded_states_and_change_with_the_term(tmp_path):
    raw = labels(_job(tmp_path, 0.0))
    pulled = labels(_job(tmp_path, 10.0))
    assert raw["labels"].shape == (3, 2, 2, 6) and pulled["labels"].shape == (3, 2, 2, 6)
    assert not np.allclose(raw["labels"], pulled["labels"])


def test_walking_without_the_term_is_raw_mppi(tmp_path):
    job = {**_job(tmp_path, 0.0), "steps": 15, "reset_seed": 70_000, "planner_seed": 95_000}
    ours = walk(job)
    theirs = raw_mppi_episode({"K": 8, "H": 6, "neff_fraction": None, "tail_policy": None, "tail_steps": 0,
                               "steps": 15, "reset_seed": 70_000, "planner_seed": 95_000})
    assert (ours["steps"], ours["survived"], ours["velocity_mae"]) == (theirs["steps"], theirs["survived"],
                                                                       theirs["velocity_mae"])
