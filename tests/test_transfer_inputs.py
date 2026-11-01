"""Portable saved-evidence preparation and first-action ownership checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.transfer_inputs import (
    SCHEMA,
    _npz,
    _physical,
    _read_raw,
    array_binding,
    load_arrays,
    load_fork_inputs,
    prepare_inputs,
)


def physical(rows, offset=0):
    states = np.zeros((rows, 19), dtype=np.float64)
    states[:, 0] = np.arange(rows) * 0.008
    states[:, 1] = np.arange(rows) * 0.01 + offset
    states[:, 2] = 1.1
    observations = np.concatenate((states[:, 2:10], states[:, 10:19]), axis=1)
    return dict(states=states, observations=observations, actions=np.zeros((rows, 6)))


def test_transfer_imports_do_not_load_simulator_or_learning_libraries():
    code = """
import sys
import src.gps.mppi_gps.transfer_inputs
import src.gps.mppi_gps.transfer_metrics
assert not any(k in sys.modules for k in ('torch', 'mujoco', 'src.envs.walker2d'))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_raw_hash_contract_is_separate_from_file_and_fitting_contracts():
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    binding = array_binding(value)
    assert binding["sha256"] == hashlib.sha256(value.tobytes()).hexdigest()
    typed = hashlib.sha256(json.dumps([list(value.shape), str(value.dtype)]).encode())
    typed.update(value.tobytes())
    assert binding["sha256"] != typed.hexdigest()
    assert binding["dtype"] == "<f4"


def test_portable_npz_detects_file_and_array_tampering(tmp_path):
    path = tmp_path / "value.npz"
    data = {"value": np.arange(6)}
    binding = _npz(path, data)
    inventory = {"files": {"value.npz": binding}}
    np.testing.assert_array_equal(
        load_arrays(tmp_path, "value.npz", inventory)["value"], data["value"]
    )
    binding["arrays"]["value"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw-array"):
        load_arrays(tmp_path, "value.npz", inventory)
    binding["arrays"]["value"] = array_binding(data["value"])
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="file differs"):
        load_arrays(tmp_path, "value.npz", inventory)


def test_input_creation_and_paths_are_create_only(tmp_path):
    path = tmp_path / "one.npz"
    _npz(path, {"value": np.zeros(1)})
    with pytest.raises(FileExistsError):
        _npz(path, {"value": np.ones(1)})
    with pytest.raises(FileExistsError):
        prepare_inputs(tmp_path / "absent", tmp_path / "absent", tmp_path)
    with pytest.raises(ValueError, match="under their owner"):
        load_arrays(tmp_path, "../outside.npz", {"files": {"../outside.npz": {}}})


def test_historical_hdf_uses_reference_hash_and_original_unicode(tmp_path):
    path = tmp_path / "raw.h5"
    value = np.array("saved α", dtype="<U7")
    reference = {"group": "rows", "arrays": {"text": {**array_binding(value), "encoding": "json"}}}
    with h5py.File(path, "x") as stream:
        stream.create_group("rows").create_dataset("text", data=np.char.encode(value, "utf-8"))
    with h5py.File(path, "r") as stream:
        actual = _read_raw(stream, reference)
    np.testing.assert_array_equal(actual["text"], value)
    with h5py.File(path, "r+") as stream:
        stream["rows/text"][()] = b"changed"
    with h5py.File(path, "r") as stream, pytest.raises(ValueError, match="Historical raw"):
        _read_raw(stream, reference)


def test_saved_physical_pairs_cannot_be_shifted_or_clipped_silently():
    values = physical(4)
    values["states"][:, 10] = [0, 11, -12, 3]
    values["observations"][:, 8] = [0, 10, -10, 3]
    _physical(values, 4)
    values["observations"] = np.roll(values["observations"], 1, axis=0)
    with pytest.raises(ValueError, match="pre-action"):
        _physical(values, 4)
    values = physical(4)
    values["actions"][0, 0] = 1.01
    with pytest.raises(ValueError, match="bounds"):
        _physical(values, 4)


@pytest.fixture
def fork_inputs(tmp_path):
    files, teachers, validation, points = {}, [], [], []
    names = [
        "untrained_0",
        "untrained_1",
        "untrained_2",
        "h4_parent",
        "h4_independent",
        "h4_fixed_high",
        "h4_backoff",
        "l1_clipped",
        "reference",
    ]
    policies = {
        name: {"policy_hash": str(index) * 64, "checkpoint_file": f"checkpoints/{name}.pt"}
        for index, name in enumerate(names)
    }

    def put(name, values):
        files[name] = _npz(tmp_path / name, values)

    for index in range(2):
        identifier = f"v2_r0_trial_raw_{index}"
        values = physical(1000, index)
        values.update(
            terminal_state=physical(1001, index)["states"][-1],
            warmstarts=np.zeros((1000, 9)),
            terminal_warmstarts=np.ones(9),
            integrations=np.zeros((1000, 91)),
            terminal_integrations=np.ones(91),
            costs=np.ones(1000),
        )
        teacher_file = f"teachers/{identifier}.npz"
        put(teacher_file, values)
        prediction_file = f"k1/predictions/{identifier}.npz"
        prediction = np.broadcast_to(np.arange(9)[None, :, None] * 0.01, (1000, 9, 6)).copy()
        put(prediction_file, {"predictions": prediction, "policy_names": np.asarray(names)})
        teachers.append(
            {
                "id": identifier,
                "rows": 1000,
                "validation_episode": identifier,
                "file": teacher_file,
                "k1_predictions_file": prediction_file,
            }
        )
        validation.append(
            dict(
                states=values["states"],
                observations=values["observations"],
                actions=values["actions"],
                row_episode=np.repeat(identifier, 1000),
                row_step=np.arange(1000),
            )
        )
    put(
        "validation.npz",
        {key: np.concatenate([row[key] for row in validation]) for key in validation[0]},
    )
    for pair in range(3):
        for update in (2000, 10000):
            identifier = f"s{pair}_u{update}"
            prediction_file = f"w2/predictions/{identifier}.npz"
            put(
                prediction_file,
                {"validation_predictions": np.full((2000, 6), pair * 0.1 + update / 100000)},
            )
            points.append(
                {
                    "id": identifier,
                    "pair": pair,
                    "update": update,
                    "policy_hash": "a" * 64,
                    "predictions_file": prediction_file,
                    "checkpoint_file": f"checkpoints/{identifier}.pt",
                }
            )
    inventory = {
        "schema": SCHEMA,
        "files": files,
        "teachers": teachers,
        "w2_points": points,
        "policies": policies,
        "k1_policy_names": names,
    }
    (tmp_path / "inventory.json").write_text(json.dumps(inventory))
    return tmp_path, inventory


def test_twenty_forks_preserve_terminal_context_and_all_action_owners(fork_inputs):
    path, inventory = fork_inputs
    anchors = load_fork_inputs(path)
    assert len(anchors) == 20
    assert [(a["teacher_id"], a["step"]) for a in anchors] == [
        (f"v2_r0_trial_raw_{teacher}", step) for teacher in range(2) for step in range(0, 1000, 100)
    ]
    assert [b["name"] for b in anchors[0]["branches"]] == [
        "teacher",
        "reference",
        "untrained_0",
        "untrained_1",
        "untrained_2",
        "s0_u2000",
        "s0_u10000",
        "s1_u2000",
        "s1_u10000",
        "s2_u2000",
        "s2_u10000",
        "uniform_random",
    ]
    last = anchors[-1]
    assert last["expected"]["states"].shape == (101, 19)
    np.testing.assert_array_equal(last["expected"]["warmstarts"][-1], np.ones(9))
    np.testing.assert_array_equal(last["expected"]["integrations"][-1], np.ones(91))
    np.testing.assert_array_equal(
        last["observation"],
        last["initial"]["state"][2:10].tolist() + last["initial"]["state"][10:19].tolist(),
    )
    first = anchors[0]["branches"][1]
    assert (
        first["provenance"]["array_sha256"]
        == inventory["files"]["k1/predictions/v2_r0_trial_raw_0.npz"]["arrays"]["predictions"][
            "sha256"
        ]
    )
    np.testing.assert_array_equal(first["action"], np.full(6, 0.08))
    other = load_fork_inputs(path)
    np.testing.assert_array_equal(
        anchors[0]["branches"][-1]["action"], other[0]["branches"][-1]["action"]
    )
    assert not np.array_equal(
        anchors[0]["branches"][-1]["action"], anchors[1]["branches"][-1]["action"]
    )


def test_fork_join_rejects_wrong_control_order(fork_inputs):
    path, inventory = fork_inputs
    values = load_arrays(path, "validation.npz", inventory)
    values["row_step"][[0, 1]] = values["row_step"][[1, 0]]
    (path / "validation.npz").unlink()
    inventory["files"]["validation.npz"] = _npz(path / "validation.npz", values)
    (path / "inventory.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="observation join"):
        load_fork_inputs(path)
