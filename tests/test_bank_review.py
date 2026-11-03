"""Byte joins use complete saved episodes and retain prototype precision limits."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import bank_review


def fixture_files(tmp_path: Path, controls: int = 3):
    bank = tmp_path / "bank"
    bank.mkdir()
    rng = np.random.default_rng(4)
    episodes = []
    records = []
    summary = []
    for seed in range(2):
        state = np.zeros((controls + 1, 19))
        state[:, 0] = np.arange(controls + 1) * 0.008
        state[:, 2] = 1.25
        state[:, 10] = 1.5
        warm = rng.normal(size=(controls + 1, 9))
        plans = rng.uniform(-1, 1, size=(controls, 4, 6))
        nominal = rng.uniform(-1, 1, size=(controls, 4, 6))
        actions = plans[:, 0].copy()
        costs = np.ones(controls)
        if controls == 8:
            costs[0] = 1e16
        total = 0.0
        for value in costs:
            total += float(value)
        flags = np.zeros(controls, dtype=bool)
        flags[0] = True
        e = dict(
            states=state,
            warmstarts=warm,
            actions=actions,
            plans=plans,
            pre_plan_nominals=nominal,
            costs=costs,
            initial_plan_flags=flags,
        )
        episodes.append(e)
        stored = dict(
            states=state[:-1],
            warmstarts=warm[:-1],
            actions=actions,
            plan=plans.astype(np.float32),
            nominal_first=nominal[:, 0].copy(),
            prefix=np.zeros((controls, 3, 2, 6), dtype=np.float16),
            cost=np.asarray(total),
            survived=np.asarray(True),
            walks=np.asarray(True),
        )
        stored["nominal_first"][0] += 10  # final refinement pass differs from entering decision
        np.savez_compressed(bank / f"episode_{seed}.npz", **stored)
        records.append(
            dict(
                reset_seed=seed,
                planner_seed=10000 + seed,
                steps=controls,
                survived=True,
                velocity_mae=0.0,
                total_cost=total,
            )
        )
        summary.append(dict(seed=seed, controls=controls))
    reference = tmp_path / "reference.npz"
    source = {key: np.concatenate([e[key] for e in episodes]) for key in episodes[0]}
    source.update(
        episode_step_offsets=np.array([0, controls, 2 * controls]),
        episode_state_offsets=np.array([0, controls + 1, 2 * (controls + 1)]),
        reset_seeds=np.array([0, 1]),
        planner_seeds=np.array([10000, 10001]),
    )
    np.savez_compressed(reference, **source)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            dict(
                episodes=records,
                run=dict(requested_episodes=2, complete=True),
                task=dict(max_steps=controls, target_velocity=1.5),
            )
        )
    )
    (bank / "summary.json").write_text(json.dumps(dict(episodes=summary)))
    return bank, reference, metrics


def modify(path: Path, function):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    function(arrays)
    np.savez_compressed(path, **arrays)


def test_full_byte_join_preserves_quantized_plan_and_final_pass_exclusion(tmp_path):
    paths = fixture_files(tmp_path)
    result = bank_review.verify_bank_reference(*paths)
    assert result["passed"] is True
    assert result["checked_episodes"] == result["expected_episodes"] == 2
    assert result["scientific_audit"] is False
    assert result["survival_independently_verified"] is False
    assert all(row["nominal_first_initial_rows_excluded"] == 1 for row in result["episodes"])
    assert all(row["stored_candidate_dtype"] == "<f2" for row in result["episodes"])
    assert all("integrations" in row["missing_full_replay_context"] for row in result["episodes"])
    assert len(result["inputs"]["bank"]) == 2
    assert result["work"]["new_native_controls"] == result["work"]["new_neural_rows"] == 0


def test_action_mismatch_and_short_prefix_cannot_pass(tmp_path):
    bank, reference, metrics = fixture_files(tmp_path)
    modify(
        bank / "episode_0.npz",
        lambda arrays: arrays["actions"].__setitem__((1, 0), arrays["actions"][1, 0] + 1e-9),
    )
    result = bank_review.verify_bank_reference(bank, reference, metrics)
    assert (
        result["passed"] is False and result["episodes"][0]["checks"]["executed_actions"] is False
    )

    def shorten(arrays):
        for key, value in arrays.copy().items():
            if value.ndim and len(value) == 3:
                arrays[key] = value[:-1]

    modify(bank / "episode_1.npz", shorten)
    result = bank_review.verify_bank_reference(bank, reference, metrics)
    assert result["episodes"][1]["checks"]["pre_action_states"] is False
    assert result["episodes"][1]["checks"]["nominal_first_noninitial"] is False


def test_noninitial_nominal_and_precision_mismatch_are_detected(tmp_path):
    bank, reference, metrics = fixture_files(tmp_path)
    modify(bank / "episode_0.npz", lambda arrays: arrays["nominal_first"].__setitem__((1, 0), 0.0))
    modify(
        bank / "episode_1.npz",
        lambda arrays: arrays.__setitem__("plan", arrays["plan"].astype(np.float64)),
    )
    result = bank_review.verify_bank_reference(bank, reference, metrics)
    assert result["episodes"][0]["checks"]["nominal_first_noninitial"] is False
    assert result["episodes"][1]["checks"]["stored_plan_precision"] is False


def test_missing_episode_and_memory_cap_stop_complete_claims(tmp_path, monkeypatch):
    bank, reference, metrics = fixture_files(tmp_path)
    (bank / "episode_1.npz").unlink()
    result = bank_review.verify_bank_reference(bank, reference, metrics)
    assert result["passed"] is False and result["missing_bank_seeds"] == [1]
    monkeypatch.setattr(bank_review, "MAX_UNCOMPRESSED_BYTES", 128)
    with pytest.raises(ValueError, match="exceeds"):
        bank_review.verify_bank_reference(bank, reference, metrics)


def test_cost_uses_original_python_left_fold(tmp_path):
    bank, reference, metrics = fixture_files(tmp_path, controls=8)
    result = bank_review.verify_bank_reference(bank, reference, metrics)
    assert result["passed"] is True
    assert result["episodes"][0]["left_fold_cost"] == 1e16
    assert np.sum(np.array([1e16] + [1.0] * 7)) != result["episodes"][0]["left_fold_cost"]


def test_bank_join_import_and_execution_need_no_simulator_or_torch(tmp_path):
    paths = fixture_files(tmp_path)
    program = """
import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'mujoco','torch'} or name.startswith(('src.envs','src.mppi')):
        raise AssertionError('Unexpected science runtime '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from pathlib import Path
from src.gps.mppi_gps.bank_review import verify_bank_reference
assert 'numpy' not in sys.modules
assert verify_bank_reference(*map(Path,sys.argv[1:]))['passed']
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program, *map(str, paths)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
