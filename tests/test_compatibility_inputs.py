"""Saved-evidence interface fixtures; no predictions, simulator or training."""

from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.compatibility_inputs import (
    _from_hdf,
    _save,
    _scan,
    membership,
    native_observations,
    validate_pairs,
)


def physical_rows(n=4):
    states = np.zeros((n, 19), dtype=np.float64)
    states[:, 2] = 1.1 + np.arange(n) * 0.01
    states[:, 10] = np.arange(n) * 12.0
    return states


def test_pre_action_alignment_is_not_post_action_alignment():
    states = physical_rows()
    observations = native_observations(states)
    actions = np.zeros((len(states), 6))
    validate_pairs(states, observations, actions)
    np.testing.assert_array_equal(observations[:, 8], [0, 10, 10, 10])
    with pytest.raises(ValueError, match="pre-action"):
        validate_pairs(states, np.roll(observations, -1, axis=0), actions)


def test_action_targets_are_rejected_instead_of_silently_clipped():
    states = physical_rows()
    actions = np.full((len(states), 6), -1.0)
    validate_pairs(states, native_observations(states), actions)
    actions[0, 0] = 1.001
    with pytest.raises(ValueError, match="bounded"):
        validate_pairs(states, native_observations(states), actions)


def test_membership_distinguishes_trajectory_holdout_from_reset_holdout():
    later = membership("v2_r0_trial_raw_0")
    assert later["l1_clipped"] == "fitted_added_independent"
    assert "same_resets" in later["h4_parent"]
    new = membership("h4_r1_backoff_00")
    assert new["h4_backoff"] == "fitted_new_acquisition"
    assert "same_resets" in new["h4_independent"]
    assert "conditioned_on_this_guide" in new["h4_parent"]
    assert new["l1_clipped"] == "not_fitted_new_reset_context"
    with pytest.raises(ValueError, match="Unknown"):
        membership("unknown_episode")


def test_hdf_extraction_preserves_endpoint_and_detects_tampering(tmp_path):
    path = tmp_path / "saved.h5"
    states = physical_rows(5)
    data = dict(
        states=states,
        observations=native_observations(states[:-1]),
        actions=np.zeros((4, 6)),
        warmstarts=np.zeros((5, 9)),
        integrations=np.zeros((5, 91)),
        costs=np.ones(4),
        velocities=np.ones(4),
    )
    with h5py.File(path, "x") as stream:
        group = stream.create_group("episode")
        for key, value in data.items():
            dataset = group.create_dataset(key, data=value)
            dataset.attrs["sha256"] = hashlib.sha256(value.tobytes()).hexdigest()
    result = _from_hdf(path, "episode")
    validate_pairs(result["states"], result["observations"], result["actions"])
    assert result["states"].shape == (4, 19)
    np.testing.assert_array_equal(result["terminal_state"], states[-1])
    assert result["terminal_integrations"].shape == (91,)
    with h5py.File(path, "r+") as stream:
        stream["episode/actions"][0, 0] = 0.1
    with pytest.raises(ValueError, match="bytes changed"):
        _from_hdf(path, "episode")


def test_curated_inputs_are_create_only(tmp_path):
    path = tmp_path / "new/episode.npz"
    value = np.arange(8)
    binding = _save(path, {"value": value})
    before = path.read_bytes()
    assert binding["file_sha256"] == hashlib.sha256(before).hexdigest()
    with np.load(path, allow_pickle=False) as stream:
        np.testing.assert_array_equal(stream["value"], value)
    with pytest.raises(FileExistsError):
        _save(path, {"value": np.zeros(2)})
    assert path.read_bytes() == before


def test_context_selection_does_not_reconstruct_missing_planner_state(tmp_path):
    run = tmp_path / "saved"
    evidence = run / "evidence"
    evidence.mkdir(parents=True)
    records = [
        dict(event="reset_requested", seed=17, tag="teacher"),
        dict(event="reset_requested", seed=99, tag="other_unselected_episode"),
        dict(
            event="independent_planner_before",
            episode="teacher",
            step=0,
            private=dict(
                U=np.zeros((48, 6)).tolist(),
                initial_plan=True,
                rng_state=dict(bit_generator="PCG64", state={"state": 17, "inc": 3}),
            ),
        ),
        dict(
            event="decision",
            episode="teacher",
            step=0,
            bank_digest="a" * 64,
            alpha=0.0,
            enabled=False,
            eta_u=0.01,
            actor_hash=None,
        ),
        dict(event="episode_complete", tag="teacher", steps=2),
        dict(event="runtime_close"),
    ]
    ledger = evidence / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in records))
    selected = _scan(run, {"teacher"}, {"teacher"}, context=True)
    assert selected["reset_seeds"] == [17, 99]
    assert list(selected["private"]["teacher"]) == [0]
    assert 1 not in selected["private"]["teacher"]
    assert selected["decisions"]["teacher"][0] == "a" * 64
    assert len(selected["private"]["teacher"][0]["rng"]) == 64
    ledger.write_text("".join(json.dumps(row) + "\n" for row in records[:-1]))
    with pytest.raises(ValueError, match="closed"):
        _scan(run, {"teacher"}, {"teacher"}, context=False)
