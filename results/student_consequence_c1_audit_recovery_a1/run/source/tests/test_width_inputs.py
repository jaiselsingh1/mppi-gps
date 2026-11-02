"""Synthetic saved-lineage checks; no actors, simulator or fitting are executed."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.width_inputs import (
    TRAIN_TAGS,
    VALIDATION_TAGS,
    _array_binding,
    _check_overlap,
    audit_reset_seeds,
    extract_benchmark,
)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_npz(path):
    with np.load(path, allow_pickle=False) as stream:
        return {key: np.array(stream[key]) for key in stream.files}


def _binding(path):
    return dict(
        bytes=path.stat().st_size,
        file_sha256=_hash(path),
        arrays={key: _array_binding(value) for key, value in _read_npz(path).items()},
    )


def _reseal(k1):
    inventory_path = k1 / "inputs/inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["dataset_bindings"]["l1_training"] = _binding(k1 / "inputs/datasets/l1_training.npz")
    for teacher in inventory["trajectories"]:
        teacher["extracted"] = _binding(k1 / "inputs" / teacher["file"])
    _write_json(inventory_path, inventory)
    seal = dict(
        files={
            str(path.relative_to(k1)): _hash(path)
            for path in sorted((k1 / "inputs").rglob("*"))
            if path.is_file()
        },
        config_sha256=_hash(k1 / "config.json"),
        source_manifest_sha256=_hash(k1 / "manifest.json"),
    )
    _write_json(k1 / "input_manifest.json", seal)
    _write_json(
        k1 / "launch_receipt.json",
        dict(
            input_manifest_sha256=_hash(k1 / "input_manifest.json"),
            source_manifest_sha256=_hash(k1 / "manifest.json"),
            config_sha256=_hash(k1 / "config.json"),
        ),
    )


@pytest.fixture
def saved_k1(tmp_path):
    k1 = tmp_path / "synthetic_k1"
    (k1 / "inputs/datasets").mkdir(parents=True)
    (k1 / "inputs/teachers").mkdir()
    _write_json(k1 / "config.json", {"fixture": "no worker"})
    _write_json(k1 / "manifest.json", {"fixture": "no frozen executable required"})
    row = np.arange(6096, dtype=np.float64)
    states = np.zeros((6096, 19), dtype=np.float64)
    states[:, 0] = row * 0.008
    states[:, 2] = 1.0 + row * 1e-5
    states[:, 10] = 1.5 + row * 1e-6
    observations = np.concatenate((states[:, 2:10], np.clip(states[:, 10:19], -10, 10)), axis=1)
    actions = row[:, None] * 1e-5 + np.arange(6, dtype=np.float64)[None, :] * 0.01
    episodes = np.full(6096, "discovery_00", dtype="U29")
    kinds = np.full(6096, "executed_discovery", dtype="U30")
    steps = np.zeros(6096, dtype=np.int64)
    kinds[4000:4064] = "hypothetical_perturbed_resolve"
    kinds[4064:4096] = "executed_recovery"
    teachers = []
    for index, tag in enumerate(TRAIN_TAGS + VALIDATION_TAGS):
        discovery = index < 4
        start = index * 1000 if discovery else 4096 + (index - 4) * 1000
        selected = slice(start, start + 1000)
        seed = 811000000 + index if discovery else 943000000 + index - 4
        episodes[selected] = tag
        steps[selected] = np.arange(1000)
        kinds[selected] = "executed_discovery" if discovery else "executed_added_independent"
        values = dict(
            states=states[selected],
            observations=observations[selected],
            actions=actions[selected],
            row_steps=np.arange(1000),
        )
        if not discovery:
            values["terminal_state"] = states[start + 999].copy()
        source_name = "teachers/" + tag + ".npz"
        np.savez(k1 / "inputs" / source_name, **values)
        teachers.append(
            dict(
                id=tag,
                file=source_name,
                group="original_discovery" if discovery else "later_independent",
                steps=1000,
                seed=seed,
                ancestry=dict(reset_seed=seed, reset_cluster=f"native-reset-{seed}"),
                outcome=dict(
                    complete=True, success=True, survived=True, cost=300.0 + index, mae=0.1
                ),
                controller=dict(H=48, K=256, eta_u=0.01, rho=0.0, effective_guidance_enabled=False),
                availability=dict(
                    terminal_state=not discovery, warmstarts=False, integrations=False
                ),
                source=dict(
                    raw=dict(
                        arrays={
                            key: _array_binding(values[key]) for key in ("observations", "actions")
                        }
                    )
                ),
            )
        )
    np.savez(
        k1 / "inputs/datasets/l1_training.npz",
        states=states,
        observations=observations,
        actions=actions,
        row_episode=episodes,
        row_steps=steps,
        row_kind=kinds,
    )
    _write_json(
        k1 / "inputs/inventory.json",
        dict(
            trajectories=teachers,
            datasets=dict(l1_training="datasets/l1_training.npz"),
            dataset_bindings={},
        ),
    )
    _reseal(k1)
    return k1


def test_extraction_preserves_executed_lineage_and_training_only_statistics(saved_k1, tmp_path):
    destination = tmp_path / "benchmark"
    result = extract_benchmark(saved_k1, destination)
    source = _read_npz(saved_k1 / "inputs/datasets/l1_training.npz")
    assert set(path.name for path in destination.iterdir()) == {
        "train.npz",
        "validation.npz",
        "normalization.npz",
        "provenance.json",
        "k1_inventory.json",
        "k1_input_manifest.json",
    }
    for name, selected in (("train", slice(0, 4000)), ("validation", slice(4096, 6096))):
        actual = _read_npz(destination / f"{name}.npz")
        assert set(actual) == {
            "states",
            "observations",
            "actions",
            "row_episode",
            "row_step",
            "row_kind",
        }
        for key in actual:
            source_key = "row_steps" if key == "row_step" else key
            np.testing.assert_array_equal(actual[key], source[source_key][selected])
        assert actual["actions"].dtype == np.float64
        assert not np.any(
            np.isin(actual["row_kind"], ["executed_recovery", "hypothetical_perturbed_resolve"])
        )
    normalization = _read_npz(destination / "normalization.npz")
    train_observations = source["observations"][:4000]
    np.testing.assert_array_equal(
        normalization["mean"], train_observations.mean(axis=0).astype(np.float32)
    )
    np.testing.assert_array_equal(
        normalization["std"], np.maximum(train_observations.std(axis=0), 0.05).astype(np.float32)
    )
    assert normalization["std"].dtype == np.float32
    assert np.any(normalization["std"] == np.float32(0.05))
    assert not np.array_equal(
        normalization["mean"], source["observations"].mean(axis=0).astype(np.float32)
    )
    assert result["normalization"]["training_only"] is True
    assert result["splits"]["train"]["teacher_mean_cost"] == 301.5
    assert result["splits"]["validation"]["teacher_mean_cost"] == 304.5
    assert result["work"] == dict(new_forward_rows=0, new_controls=0, new_training_updates=0)
    assert all(record["cross_split_overlap"] == 0 for record in result["overlap_checks"].values())
    assert (destination / "k1_inventory.json").read_bytes() == (
        saved_k1 / "inputs/inventory.json"
    ).read_bytes()
    assert (destination / "k1_input_manifest.json").read_bytes() == (
        saved_k1 / "input_manifest.json"
    ).read_bytes()
    assert json.loads((destination / "provenance.json").read_text()) == result


def test_destination_is_create_only_and_cannot_extend_frozen_k1(saved_k1, tmp_path):
    destination = tmp_path / "benchmark"
    extract_benchmark(saved_k1, destination)
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(FileExistsError):
        extract_benchmark(saved_k1, destination)
    assert before == {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(ValueError, match="outside the frozen"):
        extract_benchmark(saved_k1, saved_k1 / "new_benchmark")


@pytest.mark.parametrize(
    "name",
    [
        "input_manifest.json",
        "manifest.json",
        "config.json",
        "inputs/inventory.json",
        "inputs/datasets/l1_training.npz",
        "inputs/teachers/discovery_00.npz",
    ],
)
def test_source_hashes_fail_closed_without_writing_output(saved_k1, tmp_path, name):
    with (saved_k1 / name).open("ab") as stream:
        stream.write(b" ")
    destination = tmp_path / "rejected"
    with pytest.raises(ValueError, match="hash differs"):
        extract_benchmark(saved_k1, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("row_episode", "discovery_01"),
        ("row_kind", "executed_recovery"),
        ("row_steps", 17),
    ],
)
def test_resealed_source_still_requires_exact_episode_kind_and_step(saved_k1, tmp_path, key, value):
    path = saved_k1 / "inputs/datasets/l1_training.npz"
    values = _read_npz(path)
    values[key][0] = value
    np.savez(path, **values)
    _reseal(saved_k1)
    with pytest.raises(ValueError, match="lineage changed"):
        extract_benchmark(saved_k1, tmp_path / "rejected")


def test_resealed_dataset_cannot_change_executed_teacher_target(saved_k1, tmp_path):
    path = saved_k1 / "inputs/datasets/l1_training.npz"
    values = _read_npz(path)
    values["actions"][0, 0] += 0.1
    np.savez(path, **values)
    _reseal(saved_k1)
    with pytest.raises(ValueError, match="target lineage differs"):
        extract_benchmark(saved_k1, tmp_path / "rejected")


def test_teacher_arrays_remain_bound_to_original_executed_descriptor(saved_k1, tmp_path):
    teacher_path = saved_k1 / "inputs/teachers/discovery_00.npz"
    teacher = _read_npz(teacher_path)
    teacher["actions"][0, 0] += 0.1
    np.savez(teacher_path, **teacher)
    dataset_path = saved_k1 / "inputs/datasets/l1_training.npz"
    dataset = _read_npz(dataset_path)
    dataset["actions"][0, 0] += 0.1
    np.savez(dataset_path, **dataset)
    _reseal(saved_k1)
    with pytest.raises(ValueError, match="original executed descriptor"):
        extract_benchmark(saved_k1, tmp_path / "rejected")


@pytest.mark.parametrize("change", ["ancestry", "success"])
def test_reset_ancestry_and_success_are_admission_conditions(saved_k1, tmp_path, change):
    path = saved_k1 / "inputs/inventory.json"
    inventory = json.loads(path.read_text())
    if change == "ancestry":
        inventory["trajectories"][-1]["ancestry"]["reset_seed"] = 811000000
    else:
        inventory["trajectories"][-1]["outcome"]["success"] = False
    _write_json(path, inventory)
    _reseal(saved_k1)
    with pytest.raises(ValueError, match="ancestry|successful independent"):
        extract_benchmark(saved_k1, tmp_path / "rejected")


def test_overlap_guards_cover_float32_and_effective_normalized_inputs():
    train = dict(
        states=np.array([[1.0, 2.0]]), observations=np.array([[1.0]]), actions=np.array([[0.0]])
    )
    validation = dict(
        states=np.array([[3.0, 4.0]]),
        observations=np.array([[1.0 + 1e-9]]),
        actions=np.array([[0.5]]),
    )
    normalization = dict(
        mean=np.array([0.0], dtype=np.float32), std=np.array([1.0], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="observations:float32"):
        _check_overlap(train, validation, normalization)
    validation["observations"] = np.array(
        [[np.nextafter(np.float32(1), np.float32(2))]], dtype=np.float64
    )
    normalization["mean"] = np.array([-1e8], dtype=np.float32)
    with pytest.raises(ValueError, match="normalized_inputs:float32"):
        _check_overlap(train, validation, normalization)
    validation["states"] = train["states"].copy()
    with pytest.raises(ValueError, match="states:float64"):
        _check_overlap(train, validation, normalization)


@pytest.mark.parametrize("change", ["features", "actions"])
def test_native_features_and_bounded_targets_are_validated_before_extraction(
    saved_k1, tmp_path, change
):
    path = saved_k1 / "inputs/datasets/l1_training.npz"
    values = _read_npz(path)
    if change == "features":
        values["observations"][0, 0] += 0.01
    else:
        values["actions"][0, 0] = 1.001
    np.savez(path, **values)
    _reseal(saved_k1)
    with pytest.raises(ValueError, match="pre-action features|bounded"):
        extract_benchmark(saved_k1, tmp_path / "rejected")


def test_importing_extractor_does_not_import_policy_optimizer_or_simulator():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('torch', 'mujoco', 'src.envs', 'src.gps.mppi_gps.policy')):
        raise AssertionError('forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from src.gps.mppi_gps.width_inputs import extract_benchmark
assert callable(extract_benchmark)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_seed_census_includes_prior_k1_and_root_offset_ranges(tmp_path):
    runs = tmp_path / "runs"
    prior = runs / "prior_k1"
    other = runs / "older"
    prior.mkdir(parents=True)
    other.mkdir()
    _write_json(prior / "config.json", dict(primary_seeds=[2_026_103_100], unrelated_count=72))
    _write_json(other / "config.json", dict(roots=dict(evaluation=800_000_000)))
    report = audit_reset_seeds(tmp_path, [2_026_103_100, 2_026_103_101, 800_999_999, 72])
    assert report["passed"] is False
    assert len(report["historical_configurations"]) == 2
    assert {row["seed"] for row in report["collisions"]} == {
        2_026_103_100,
        2_026_103_101,
        800_999_999,
        72,
    }
    assert any(row.get("conservative_root") == 800_000_000 for row in report["collisions"])
    for binding in report["historical_configurations"]:
        assert binding["sha256"] == _hash(Path(binding["file"]))
    assert "No ledger" in report["scope"]
    fresh = audit_reset_seeds(tmp_path, [801_000_000, 2_027_103_100])
    assert fresh["passed"] is True
    assert fresh["collisions"] == []


def test_seed_census_rejects_duplicate_or_invalid_proposals_and_missing_census(tmp_path):
    run = tmp_path / "runs/old"
    run.mkdir(parents=True)
    _write_json(run / "config.json", dict(seed=3))
    duplicate = audit_reset_seeds(tmp_path, [9_000_000, 9_000_000])
    assert duplicate["passed"] is False
    assert duplicate["collisions"][0]["reason"] == "duplicate proposal"
    for invalid in ([], [True], [-1], [2**32], [1.0]):
        with pytest.raises(ValueError, match="uint32"):
            audit_reset_seeds(tmp_path, invalid)
    with pytest.raises(ValueError, match="No local historical"):
        audit_reset_seeds(tmp_path / "missing", [9_000_000])
