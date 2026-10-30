"""Independent checker contracts using tiny saved-only synthetic evidence."""

import hashlib
import json
from copy import deepcopy

import h5py
import numpy as np
import pytest
import torch

from src.gps.mppi_gps.audits.native import Audit, fingerprint, fit_hash, obs, tensor_hash
from src.gps.mppi_gps.audits.outcome_detail import primary_details


def descriptor(value):
    array = np.ascontiguousarray(value)
    return dict(
        shape=list(np.shape(value)),
        dtype=str(array.dtype),
        sha256=hashlib.sha256(array.tobytes()).hexdigest(),
        encoding="numpy",
    )


def write_group(path, arrays, tag="synthetic"):
    reference = dict(file="raw.h5", group="arrays/example", tag=tag, arrays={})
    with h5py.File(path, "w") as stream:
        group = stream.create_group(reference["group"])
        group.attrs["tag"] = tag
        for key, value in arrays.items():
            array = np.asarray(value)
            meta = descriptor(array)
            reference["arrays"][key] = meta
            dataset = group.create_dataset(key, data=array)
            dataset.attrs["numpy_dtype"] = meta["dtype"]
            dataset.attrs["encoding"] = meta["encoding"]
            dataset.attrs["sha256"] = meta["sha256"]
    return reference


def close_hdfs(audit):
    for handle, _, _ in audit.hdfs.values():
        handle.close()


def synthetic_fall(tmp_path):
    """Two paid controls, one terminal penalty and 998 absorbing controls."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    states = np.zeros((3, 19), dtype=np.float64)
    states[:, 0] = np.arange(3) * 0.008
    states[:, 2] = [1.2, 1.2, 0.7]
    states[:, 10] = 1.5
    arrays = dict(
        states=states,
        actions=np.zeros((2, 6)),
        warmstarts=np.zeros((3, 9)),
        integrations=np.zeros((3, 91)),
        observations=obs(states[:-1]),
        velocities=states[1:, 10],
        costs=np.array([0.0, 25.0]),
        terminal_cost=np.array(20.0),
        terminal_adjustment=np.array(0.0),
        absorbing_cost=np.array(4990.0),
    )
    reference = write_group(evidence / "raw.h5", arrays)
    event = dict(
        steps=2,
        raw=reference,
        seed=9,
        cost=5015.0,
        mae=0.0,
        survived=False,
        success=False,
        complete=True,
    )
    live = []
    for step in range(2):
        before = dict(
            step=step,
            inputs=dict(
                arrays={
                    key: descriptor(value)
                    for key, value in dict(
                        action=arrays["actions"][step],
                        state=states[step],
                        warmstart=arrays["warmstarts"][step],
                        integration=arrays["integrations"][step],
                    ).items()
                }
            ),
        )
        after = dict(
            result=dict(
                arrays={
                    key: descriptor(value)
                    for key, value in dict(
                        state=states[step + 1],
                        warmstart=arrays["warmstarts"][step + 1],
                        integration=arrays["integrations"][step + 1],
                        observation=obs(states[step + 1]),
                        cost=arrays["costs"][step],
                    ).items()
                }
            )
        )
        live.append((before, after))
    return dict(
        path=tmp_path,
        episodes={"synthetic": event},
        resets={"synthetic": 9},
        live={"synthetic": live},
    )


def test_distinct_hash_contracts_match_frozen_checker_vectors():
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    assert fit_hash(values) == "b403aa4f690823002590c077ab0e5c30a44e037376e777e86e5fe3d646b4bb3f"
    assert (
        hashlib.sha256(values.tobytes()).hexdigest()
        == "6bab56d2f81d4b5a2dbf102bf6a6ff7d5211a475fc5f97813f977e8ba714b07d"
    )
    tree = {"adam": {2: torch.tensor([1.0, 2.0]), "2": [1, 2]}, "rng": (3, 4)}
    assert fingerprint(tree) == "08619c8e96d080ae32634b711eb161371812c872de34858d59408639cc5e4a31"
    assert fingerprint({"rng": (3, 4), "adam": tree["adam"]}) == fingerprint(tree)
    assert fingerprint([3, 4]) != fingerprint((3, 4))
    assert fit_hash(values) == fit_hash(values.astype(np.float32))
    assert fit_hash(values) != fit_hash(values.reshape(4))
    state = {"network.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    assert tensor_hash(state) == "efc785ec1819017bd57f8a67eb6db612bd6be997a0405ef8fce8c6fe93bbc8f9"


def test_cpu_checkpoint_decoder_binds_tensor_owner_without_policy(tmp_path):
    state = {"network.0.weight": torch.zeros((128, 17))}
    package = dict(
        schema="walker-tanh-policy-v1",
        state_dict=state,
        policy_hash=tensor_hash(state),
        optimizer_state=None,
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(package, path)
    audit = Audit(tmp_path, tmp_path / "source")
    decoded = audit.checkpoint(path)
    assert decoded["policy_hash"] == package["policy_hash"]
    assert decoded["state_dict"]["network.0.weight"].device.type == "cpu"
    package["state_dict"]["network.0.weight"][0, 0] = 1
    torch.save(package, tmp_path / "tampered.pt")
    with pytest.raises(AssertionError, match="tensor owner"):
        audit.checkpoint(tmp_path / "tampered.pt")


def test_native_complete_cost_terminal_once_and_absorbing_tail(tmp_path):
    stage = synthetic_fall(tmp_path)
    audit = Audit(tmp_path, tmp_path / "source")
    try:
        event, arrays = audit.episode(stage, "synthetic")
        assert event["cost"] == 5015
        assert arrays["costs"].sum() == 25
        assert len(audit.arrays) == 10
        bad = deepcopy(stage)
        bad["episodes"]["synthetic"]["cost"] += 20
        with pytest.raises(AssertionError, match="Complete cost/absorbing tail"):
            audit.episode(bad, "synthetic")
    finally:
        close_hdfs(audit)


def test_primary_detail_uses_result_roster_rather_than_episode_name_prefix(tmp_path):
    stage = synthetic_fall(tmp_path)
    event = dict(stage["episodes"]["synthetic"], event="episode_complete", tag="synthetic")
    (tmp_path / "result.json").write_text(
        json.dumps(dict(cases=dict(primary=dict(control=dict(tags=["synthetic"])))))
    )
    (tmp_path / "evidence/ledger.jsonl").write_text(json.dumps(event) + "\n")
    result = primary_details(tmp_path)
    row = result["cases"]["control"][0]
    assert row["complete_cost"] == 5015
    assert row["termination"] == "height"
    assert sum(row["cost_components"].values()) == 5015


def test_hdf_descriptor_rejects_mutated_bytes(tmp_path):
    path = tmp_path / "raw.h5"
    reference = write_group(path, {"states": np.array([1.0, 2.0])})
    with h5py.File(path, "r+") as stream:
        stream["arrays/example/states"][0] = 3
    audit = Audit(tmp_path, tmp_path / "source")
    try:
        with pytest.raises(AssertionError, match="HDF actual bytes states"):
            audit.group(path, reference)
    finally:
        close_hdfs(audit)


def test_native_observation_clips_velocities_but_omits_clock_and_translation():
    state = np.arange(19, dtype=np.float64)
    actual = obs(state)
    assert np.array_equal(actual[:8], state[2:10])
    assert np.array_equal(actual[8:], np.full(9, 10.0))
    state[:2] = [900.0, -100.0]
    assert np.array_equal(obs(state), actual)


def test_hdf_unicode_json_decoding_preserves_original_array_hash(tmp_path):
    value = np.asarray(json.dumps({"rows": [{"controller": "saved-only"}]}))
    meta = descriptor(value)
    meta["encoding"] = "json"
    reference = dict(file="raw.h5", group="arrays/json", tag="proof", arrays={"infos": meta})
    path = tmp_path / "raw.h5"
    with h5py.File(path, "w") as stream:
        group = stream.create_group(reference["group"])
        group.attrs["tag"] = "proof"
        dataset = group.create_dataset("infos", data=np.char.encode(value, "utf-8"))
        dataset.attrs["numpy_dtype"] = meta["dtype"]
        dataset.attrs["encoding"] = "json"
        dataset.attrs["sha256"] = meta["sha256"]
    audit = Audit(tmp_path, tmp_path / "source")
    try:
        assert audit.group(path, reference)["infos"] == {"rows": [{"controller": "saved-only"}]}
    finally:
        close_hdfs(audit)


def write_ledger(tmp_path, *, incorrect_work=False, close=True):
    """One live query and one completed neural-row measurement, no physics."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    rows = []

    def add(event, **fields):
        row = dict(
            id=f"event-{len(rows):09d}",
            event=event,
            kind=event,
            wall_seconds=len(rows) * 0.01,
            **fields,
        )
        rows.append(row)
        return row

    request = add(
        "query_requested",
        mode="live",
        requested_controls=1,
        requested_physics_substeps=4,
        query_id="query-0",
        episode="one",
    )
    add(
        "query_complete",
        request_id=request["id"],
        query_id="query-0",
        mode="live",
        completed_controls=1,
        completed_physics_substeps=4,
    )
    measurement = add("measurement_requested", measure_kind="neural_rows", amount=3)
    add("measurement_complete", request_id=measurement["id"], measure_kind="neural_rows", amount=3)
    work = dict(
        requested_controls=1,
        completed_controls=1,
        requested_live_controls=1,
        completed_live_controls=1,
        requested_physics_substeps=4,
        completed_physics_substeps=4,
        live_queries=1,
        queries=1,
        physics_frames=4,
        measurement_requests=1,
        requested_neural_rows=3,
        completed_neural_rows=3,
        neural_rows=3,
        snapshots=0,
    )
    if close:
        add("runtime_close", totals=deepcopy(work))
    if incorrect_work:
        work["completed_controls"] += 1
    (tmp_path / "result.json").write_text(json.dumps(dict(work=work, wall_seconds=0.05)))
    (evidence / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_closed_ledger_recounts_queries_measurements_and_work(tmp_path):
    write_ledger(tmp_path)
    audit = Audit(tmp_path, tmp_path / "source")
    stage = audit.ledger(tmp_path)
    assert stage["counts"]["completed_controls"] == 1
    assert stage["counts"]["completed_neural_rows"] == 3
    assert stage["logged_query_requests"] == 1
    assert len(stage["live"]["one"]) == 1


@pytest.mark.parametrize(
    "incorrect_work,close,message",
    [
        (True, True, "Work census completed_controls"),
        (False, False, "Closed stage only"),
    ],
)
def test_ledger_rejects_unclosed_or_unreconciled_work(tmp_path, incorrect_work, close, message):
    write_ledger(tmp_path, incorrect_work=incorrect_work, close=close)
    with pytest.raises(AssertionError, match=message):
        Audit(tmp_path, tmp_path / "source").ledger(tmp_path)
