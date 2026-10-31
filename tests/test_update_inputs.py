"""Synthetic sealed-benchmark reuse checks; no learning or simulator operations."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import width_inputs
from src.gps.mppi_gps.update_inputs import EXACT_COPIES, copy_benchmark


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as stream:
        return {key: np.array(stream[key]) for key in stream.files}


def binding(path):
    return dict(
        file=path.name,
        file_sha256=file_hash(path),
        bytes=path.stat().st_size,
        arrays={key: width_inputs._array_binding(value) for key, value in read_npz(path).items()},
    )


def reseal(w1):
    inputs = w1 / "inputs"
    provenance_path = inputs / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    for split in ("train", "validation"):
        provenance["splits"][split].update(binding(inputs / f"{split}.npz"))
    provenance["normalization"].update(binding(inputs / "normalization.npz"))
    write_json(provenance_path, provenance)
    seal = dict(
        files={"inputs/" + path.name: file_hash(path) for path in inputs.iterdir()},
        config_sha256=file_hash(w1 / "config.json"),
        source_manifest_sha256=file_hash(w1 / "manifest.json"),
    )
    write_json(w1 / "input_manifest.json", seal)
    write_json(
        w1 / "launch_receipt.json",
        dict(
            input_manifest_sha256=file_hash(w1 / "input_manifest.json"),
            source_manifest_sha256=file_hash(w1 / "manifest.json"),
            config_sha256=file_hash(w1 / "config.json"),
        ),
    )


@pytest.fixture
def saved_w1(tmp_path):
    w1 = tmp_path / "saved_w1"
    inputs = w1 / "inputs"
    inputs.mkdir(parents=True)
    task = dict(target_velocity=1.5, frame_skip=4)
    config = dict(
        schema="walker-student-width-w1-v1",
        rows=4000,
        validation_rows=2000,
        normalization_floor=0.05,
        task=task,
        xml_sha256="a" * 64,
    )
    write_json(w1 / "config.json", config)
    write_json(w1 / "manifest.json", dict(config_sha256=file_hash(w1 / "config.json")))
    write_json(inputs / "task.json", dict(task=task, xml_sha256=config["xml_sha256"]))
    splits, teachers = {}, {}
    for split, start, rows, tags, kind in (
        ("train", 0, 4000, width_inputs.TRAIN_TAGS, "executed_discovery"),
        ("validation", 4096, 2000, width_inputs.VALIDATION_TAGS, "executed_added_independent"),
    ):
        row = np.arange(start, start + rows, dtype=np.float64)
        states = np.zeros((rows, 19), dtype=np.float64)
        states[:, 0] = row * 0.008
        states[:, 2] = 1 + row * 1e-5
        states[:, 10] = 1.5
        observations = np.concatenate((states[:, 2:10], states[:, 10:19]), axis=1)
        actions = row[:, None] * 1e-5 + np.arange(6)[None, :] * 0.01
        values = dict(
            states=states,
            observations=observations,
            actions=actions,
            row_episode=np.repeat(tags, 1000),
            row_kind=np.full(rows, kind),
            row_step=np.tile(np.arange(1000), len(tags)),
        )
        np.savez(inputs / f"{split}.npz", **values)
        outcomes = {}
        for index, tag in enumerate(tags):
            selected = slice(index * 1000, (index + 1) * 1000)
            seed = 811000000 + index if split == "train" else 943000000 + index
            outcome = dict(complete=True, success=True, survived=True, cost=300.0 + index, mae=0.1)
            teachers[tag] = dict(
                id=tag,
                steps=1000,
                seed=seed,
                outcome=outcome,
                ancestry=dict(reset_seed=seed, reset_cluster=f"native-reset-{seed}"),
                extracted=dict(
                    arrays={
                        key: width_inputs._array_binding(values[key][selected])
                        for key in ("states", "observations", "actions")
                    }
                ),
                availability=dict(
                    terminal_state=split == "validation", warmstarts=False, integrations=False
                ),
            )
            outcomes[tag] = outcome
        splits[split] = dict(
            **binding(inputs / f"{split}.npz"),
            rows=rows,
            teachers=list(tags),
            indices=dict(start=start, stop=start + rows),
            outcomes=outcomes,
            teacher_mean_cost=float(np.mean([value["cost"] for value in outcomes.values()])),
            teacher_mean_mae=float(np.mean([value["mae"] for value in outcomes.values()])),
            fit_data=dict(
                observations_sha256=width_inputs._fit_hash(observations),
                actions_sha256=width_inputs._fit_hash(actions),
                rows=rows,
            ),
        )
    observations = read_npz(inputs / "train.npz")["observations"]
    np.savez(
        inputs / "normalization.npz",
        mean=observations.mean(axis=0).astype(np.float32),
        std=np.maximum(observations.std(axis=0), 0.05).astype(np.float32),
    )
    normalization = dict(**binding(inputs / "normalization.npz"), training_only=True, rows=4000)
    write_json(inputs / "k1_inventory.json", dict(trajectories=list(teachers.values())))
    write_json(
        inputs / "k1_input_manifest.json",
        dict(files={"inputs/inventory.json": file_hash(inputs / "k1_inventory.json")}),
    )
    provenance = dict(
        schema="walker-width-w1-executed-benchmark-v1",
        splits=splits,
        normalization=normalization,
        teachers=teachers,
        excluded_rows=dict(start=4000, stop=4096, hypothetical=64, recovery=32),
        source=dict(
            run="unchanged_original_k1",
            k1_inventory_sha256=file_hash(inputs / "k1_inventory.json"),
            k1_input_manifest_sha256=file_hash(inputs / "k1_input_manifest.json"),
        ),
        missing_evidence=["Original discovery final endpoint absent locally"],
    )
    write_json(inputs / "provenance.json", provenance)
    reseal(w1)
    return w1


def test_copies_exact_containers_and_preserves_original_provenance(saved_w1, tmp_path):
    original = json.loads((saved_w1 / "inputs/provenance.json").read_text())
    before = {
        str(path.relative_to(saved_w1)): path.read_bytes()
        for path in saved_w1.rglob("*")
        if path.is_file()
    }
    destination = tmp_path / "w2_inputs"
    result = copy_benchmark(saved_w1, destination)
    assert {path.name for path in destination.iterdir()} == set(EXACT_COPIES) | {"provenance.json"}
    for name, source in EXACT_COPIES.items():
        assert (destination / name).read_bytes() == (saved_w1 / source).read_bytes()
        assert result["w1_source"]["exact_copies"][name]["sha256"] == file_hash(destination / name)
    assert result["splits"] == original["splits"]
    assert result["normalization"] == original["normalization"]
    assert result["teachers"] == original["teachers"]
    assert result["missing_evidence"] == original["missing_evidence"]
    assert result["source"] == original["source"]
    assert all(row["cross_split_overlap"] == 0 for row in result["reuse_overlap_checks"].values())
    assert result["work"] == dict(new_forward_rows=0, new_controls=0, new_training_updates=0)
    assert before == {
        str(path.relative_to(saved_w1)): path.read_bytes()
        for path in saved_w1.rglob("*")
        if path.is_file()
    }
    assert json.loads((destination / "provenance.json").read_text()) == result


def test_reuse_is_create_only_and_never_adds_to_frozen_parent(saved_w1, tmp_path):
    destination = tmp_path / "w2_inputs"
    copy_benchmark(saved_w1, destination)
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(FileExistsError):
        copy_benchmark(saved_w1, destination)
    assert before == {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(ValueError, match="outside the frozen"):
        copy_benchmark(saved_w1, saved_w1 / "w2_inputs")


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "manifest.json",
        "input_manifest.json",
        "inputs/provenance.json",
        "inputs/train.npz",
        "inputs/validation.npz",
        "inputs/normalization.npz",
        "inputs/task.json",
    ],
)
def test_launch_and_input_pins_reject_changed_bytes(saved_w1, tmp_path, name):
    with (saved_w1 / name).open("ab") as stream:
        stream.write(b" ")
    destination = tmp_path / "rejected"
    with pytest.raises(ValueError, match="source hash differs"):
        copy_benchmark(saved_w1, destination)
    assert not destination.exists()


@pytest.mark.parametrize("change", ["features", "actions", "row_episode", "row_step", "row_kind"])
def test_resealed_arrays_still_require_native_executed_lineage(saved_w1, tmp_path, change):
    path = saved_w1 / "inputs/train.npz"
    values = read_npz(path)
    if change == "features":
        values["observations"][0, 0] += 0.1
    elif change == "actions":
        values["actions"][0, 0] = 1.001
    elif change == "row_episode":
        values[change][0] = "discovery_01"
    elif change == "row_step":
        values[change][0] = 17
    else:
        values[change][0] = "executed_recovery"
    np.savez(path, **values)
    reseal(saved_w1)
    with pytest.raises(ValueError, match="pre-action features|bounded|row lineage"):
        copy_benchmark(saved_w1, tmp_path / "rejected")


def test_even_resealed_target_and_fit_hash_cannot_replace_teacher_label(saved_w1, tmp_path):
    path = saved_w1 / "inputs/train.npz"
    values = read_npz(path)
    values["actions"][0, 0] += 0.1
    np.savez(path, **values)
    provenance_path = saved_w1 / "inputs/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["splits"]["train"]["fit_data"]["actions_sha256"] = width_inputs._fit_hash(
        values["actions"]
    )
    write_json(provenance_path, provenance)
    reseal(saved_w1)
    with pytest.raises(ValueError, match="original teacher array binding"):
        copy_benchmark(saved_w1, tmp_path / "rejected")


@pytest.mark.parametrize("change", ["validation_statistics", "floor"])
def test_resealed_normalization_cannot_use_validation_or_change_floor(saved_w1, tmp_path, change):
    path = saved_w1 / "inputs/normalization.npz"
    normalization = read_npz(path)
    if change == "validation_statistics":
        observations = np.concatenate(
            [
                read_npz(saved_w1 / f"inputs/{split}.npz")["observations"]
                for split in ("train", "validation")
            ]
        )
        normalization["mean"] = observations.mean(axis=0).astype(np.float32)
    else:
        normalization["std"][normalization["std"] == np.float32(0.05)] = np.float32(0.01)
    np.savez(path, **normalization)
    reseal(saved_w1)
    with pytest.raises(ValueError, match="training-only normalization"):
        copy_benchmark(saved_w1, tmp_path / "rejected")


def test_teacher_outcomes_cannot_be_changed_in_reuse_provenance(saved_w1, tmp_path):
    path = saved_w1 / "inputs/provenance.json"
    provenance = json.loads(path.read_text())
    provenance["teachers"]["discovery_00"]["outcome"]["success"] = False
    write_json(path, provenance)
    reseal(saved_w1)
    with pytest.raises(ValueError, match="teacher/outcome ancestry"):
        copy_benchmark(saved_w1, tmp_path / "rejected")


def test_helper_import_does_not_load_experiments_policies_or_simulator():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('torch', 'mujoco', 'src.envs', 'src.gps.mppi_gps.policy', 'src.gps.mppi_gps.experiments')):
        raise AssertionError('forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from src.gps.mppi_gps.update_inputs import copy_benchmark
assert callable(copy_benchmark)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
