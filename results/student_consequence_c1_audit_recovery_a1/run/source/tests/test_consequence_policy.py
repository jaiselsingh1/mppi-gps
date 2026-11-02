"""Synthetic-only weighted fitting and exact legacy continuation contracts."""

from __future__ import annotations

import builtins
import struct
from copy import deepcopy

import numpy as np
import pytest
import torch

from src.gps.mppi_gps.policy import FIT_SCHEMA, POLICY_SCHEMA, Policy, fit_policy


class RecordingMeter:
    def __init__(self, *, capture_predictions=False):
        self.calls = []
        self.predictions = []
        self.capture_predictions = capture_predictions

    def measure(self, kind, amount, operation, **metadata):
        self.calls.append((kind, amount, metadata))
        value = operation()
        if kind == "neural_rows" and self.capture_predictions:
            self.predictions.append(value.detach().cpu().clone())
        return value


def exact_tree(first, second):
    """Compare tensor/scalar bytes, not an approximate optimization endpoint."""
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        assert first.dtype == second.dtype and first.shape == second.shape
        assert torch.equal(
            first.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            second.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
    elif isinstance(first, np.ndarray):
        assert first.dtype == second.dtype and first.shape == second.shape
        assert first.tobytes() == second.tobytes()
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            exact_tree(first[name], second[name])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            exact_tree(left, right)
    elif isinstance(first, float):
        assert struct.pack(">d", first) == struct.pack(">d", second)
    else:
        assert first == second


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def toy_data(rows=37, seed=817):
    rng = np.random.Generator(np.random.PCG64(seed))
    observations = rng.normal(0.0, 0.3, size=(rows, 17)).astype(np.float32)
    actions = rng.uniform(-0.3, 0.3, size=(rows, 6)).astype(np.float32)
    return observations, actions


def incoming_parent():
    initial = Policy(
        seed=971,
        normalization=dict(
            mean=np.linspace(-0.2, 0.2, 17, dtype=np.float32),
            std=np.linspace(0.7, 1.3, 17, dtype=np.float32),
        ),
    )
    old_x, old_y = toy_data(rows=23, seed=818)
    parent, state, _ = fit_policy(
        initial,
        old_x,
        old_y,
        updates=7,
        batch_size=256,
        lr=0.0003,
        eta_theta=0.0,
        prox_reduction="mean",
        seed=819,
    )
    return parent, state


def test_index_request_persistence_failure_stops_before_draw_and_paid_forward():
    parent, incoming = incoming_parent()
    x, y = toy_data()
    meter, sampled, requested = RecordingMeter(), [], []

    def stop(value):
        requested.append(value)
        raise OSError("synthetic pre-draw persistence failure")

    with pytest.raises(OSError, match="pre-draw") as caught:
        fit_policy(
            parent,
            x,
            y,
            optimizer_state=incoming,
            updates=1,
            meter=meter,
            index_request_observer=stop,
            index_observer=sampled.append,
        )
    assert meter.calls == sampled == []
    assert len(requested) == 1
    exact_tree(requested[0]["rng_before"], incoming["rng"])
    assert caught.value.fit_evidence["completed_updates"] == 0


def test_full_2000_all_ones_continues_exact_legacy_tensors_adam_anchor_and_rng(tmp_path):
    global_before = torch.get_rng_state().clone()
    parent, incoming = incoming_parent()
    path = tmp_path / "incoming.pt"
    parent.checkpoint(path, optimizer_state=incoming, metadata=dict(scope="synthetic only"))
    parent, incoming, _ = Policy.from_checkpoint(path)
    parent_before = deepcopy(parent.state_dict())
    private_before = parent._generator.get_state().clone()
    incoming_before = deepcopy(incoming)
    x, y = toy_data()
    tapes, before_forward, requests, initials, results, meters = [], [], [], [], [], []
    for weights in (None, np.ones(len(x), dtype=np.float64)):
        tape, sampled, requested, initial = [], [], [], []
        meter = RecordingMeter(capture_predictions=weights is not None)
        result = fit_policy(
            parent,
            x,
            y,
            optimizer_state=incoming,
            updates=2000,
            batch_size=256,
            lr=0.0003,
            eta_theta=0.0,
            prox_reduction="mean",
            anchor_refresh=0,
            seed=820,
            resume_fit=False,
            initial_observer=initial.append,
            update_observer=lambda row: tape.append(row["indices"]),
            index_observer=None if weights is None else sampled.append,
            index_request_observer=None if weights is None else requested.append,
            row_weights=weights,
            meter=meter,
        )
        tapes.append(np.stack(tape))
        before_forward.append(sampled)
        requests.append(requested)
        initials.append(initial)
        results.append(result)
        meters.append(meter)
    (legacy, legacy_state, legacy_evidence), (ones, ones_state, ones_evidence) = results
    exact_tree(initials[0], initials[1])
    exact_tree(legacy.state_dict(), ones.state_dict())
    exact_tree(legacy_state, ones_state)
    exact_tree(legacy._generator.get_state(), ones._generator.get_state())
    exact_tree(tapes[0], tapes[1])
    assert before_forward[0] == []
    assert requests[0] == []
    for index, event in enumerate(before_forward[1]):
        exact_tree(event["indices"], tapes[0][index])
        exact_tree(requests[1][index]["rng_before"], event["rng_before"])
        assert requests[1][index]["fit_update"] == index + 1
        assert event["fit_update"] == index + 1 and event["update"] == index + 8
        exact_tree(
            event["rng_before"],
            incoming["rng"] if index == 0 else before_forward[1][index - 1]["rng_after"],
        )
    exact_tree(before_forward[1][-1]["rng_after"], legacy_state["rng"])
    assert tapes[0].shape == (2000, 256)
    assert legacy_state["updates"] == 2007 and legacy_state["fit_updates"] == 2000
    assert legacy_state["anchor_policy_hash"] == parent.hash()
    assert legacy_state["fit_data"] != incoming["fit_data"]
    assert legacy_state["schema"] == ones_state["schema"] == FIT_SCHEMA
    assert set(legacy_state) == {
        "schema",
        "adam",
        "rng",
        "updates",
        "fit_updates",
        "policy_hash",
        "parent_policy_hash",
        "anchor_policy_hash",
        "anchor",
        "parameter_names",
        "fit_config",
        "fit_data",
    }
    for before, weighted in zip(
        legacy_evidence["learning_curve"], ones_evidence["learning_curve"], strict=True
    ):
        exact_tree(before, {key: weighted[key] for key in before})
        assert set(weighted) == set(before) | {"weighted_action_mse"}
        exact_tree(weighted["weighted_action_mse"], before["action_mse"])
    exact_tree(parent.state_dict(), parent_before)
    exact_tree(parent._generator.get_state(), private_before)
    exact_tree(incoming, incoming_before)
    exact_tree(torch.get_rng_state(), global_before)
    for meter in meters:
        assert len(meter.calls) == 6000
        for kind, amount in (("neural_rows", 256), ("backward_rows", 256), ("optimizer_steps", 1)):
            calls = [row for row in meter.calls if row[0] == kind]
            assert len(calls) == 2000 and all(row[1] == amount for row in calls)
    legacy_path, ones_path = tmp_path / "legacy.pt", tmp_path / "ones.pt"
    legacy.checkpoint(legacy_path, optimizer_state=legacy_state)
    ones.checkpoint(ones_path, optimizer_state=ones_state)
    legacy_payload = torch.load(legacy_path, map_location="cpu", weights_only=True)
    ones_payload = torch.load(ones_path, map_location="cpu", weights_only=True)
    assert set(legacy_payload) == {
        "schema",
        "seed",
        "obs_dim",
        "policy_hash",
        "state_dict",
        "private_rng_state",
        "training",
        "optimizer_state",
        "metadata",
    }
    assert legacy_payload["schema"] == ones_payload["schema"] == POLICY_SCHEMA
    exact_tree(legacy_payload, ones_payload)
    assert legacy_path.read_bytes() == ones_path.read_bytes()


def test_weighted_and_ordinary_losses_share_the_single_paid_forward():
    x, y = toy_data()
    parent, state = incoming_parent()
    weights = np.linspace(0.25, 2.5, len(x), dtype=np.float64)
    meter = RecordingMeter(capture_predictions=True)
    updates = []
    _, _, evidence = fit_policy(
        parent,
        x,
        y,
        optimizer_state=state,
        updates=1,
        batch_size=19,
        eta_theta=0.0,
        prox_reduction="mean",
        row_weights=weights,
        meter=meter,
        update_observer=updates.append,
    )
    rows = updates[0]["indices"]
    squared = (meter.predictions[0] - torch.as_tensor(y[rows])) ** 2
    ordinary = float(torch.mean(squared))
    weighted = float(torch.mean(squared * torch.as_tensor(weights.astype(np.float32)[rows, None])))
    point = evidence["learning_curve"][0]
    exact_tree(point["action_mse"], ordinary)
    exact_tree(point["weighted_action_mse"], weighted)
    exact_tree(point["total_loss"], weighted)
    assert weighted != ordinary
    assert [(kind, amount) for kind, amount, _ in meter.calls] == [
        ("neural_rows", 19),
        ("backward_rows", 19),
        ("optimizer_steps", 1),
    ]
    assert len(meter.predictions) == 1
    assert evidence["work"]["neural_rows"] == dict(requested=19, completed=19, unknown_partial=0)


def test_uniform_and_shuffled_weighted_blocks_share_incoming_state_and_minibatches():
    x, y = toy_data()
    parent, incoming = incoming_parent()
    before = deepcopy(incoming)
    global_before = torch.get_rng_state().clone()
    weights = np.linspace(0.25, 2.5, len(x), dtype=np.float32)
    shuffled = weights[np.random.Generator(np.random.PCG64(821)).permutation(len(weights))]
    results, tapes, initials = [], [], []
    for values in (np.ones(len(x), dtype=np.float32), weights, shuffled):
        tape, initial = [], []
        results.append(
            fit_policy(
                parent,
                x,
                y,
                optimizer_state=incoming,
                updates=3,
                batch_size=19,
                eta_theta=0.0,
                prox_reduction="mean",
                row_weights=values,
                initial_observer=initial.append,
                update_observer=lambda row: tape.append(row["indices"]),
            )
        )
        tapes.append(np.stack(tape))
        initials.append(initial)
    exact_tree(initials[0], initials[1])
    exact_tree(initials[0], initials[2])
    exact_tree(tapes[0], tapes[1])
    exact_tree(tapes[0], tapes[2])
    assert results[1][0].hash() != results[2][0].hash()
    for _, state, _ in results:
        exact_tree(state["rng"], results[0][1]["rng"])
        assert state["updates"] == 10 and state["fit_updates"] == 3
    exact_tree(incoming, before)
    exact_tree(torch.get_rng_state(), global_before)


@pytest.mark.parametrize(
    "weights",
    (
        np.ones(36),
        np.ones((37, 1)),
        1.0,
        np.array([]),
        np.zeros(37),
        -np.ones(37),
        np.full(37, np.nan),
        np.full(37, np.inf),
        np.full(37, 1e100),
        np.full(37, 1e-100),
        np.ones(37, dtype=bool),
        np.ones(37, dtype=complex),
        np.array(["1"] * 37),
    ),
)
def test_invalid_weights_stop_before_any_paid_call(weights):
    x, y = toy_data()
    parent = Policy(822)
    before = parent.hash()
    meter = RecordingMeter()
    with pytest.raises(ValueError, match="row_weights"):
        fit_policy(parent, x, y, updates=1, row_weights=weights, meter=meter)
    assert meter.calls == [] and parent.hash() == before


def test_weighted_resume_cannot_silently_change_unserialized_weight_binding():
    x, y = toy_data()
    parent, state = incoming_parent()
    meter = RecordingMeter()
    with pytest.raises(ValueError, match="externally bound block"):
        fit_policy(
            parent,
            x,
            y,
            optimizer_state=state,
            updates=1,
            resume_fit=True,
            row_weights=np.ones(len(x)),
            meter=meter,
        )
    assert meter.calls == []


def test_weighted_fitting_does_not_import_a_simulator(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in ("mujoco", "gymnasium", "mujoco_warp"):
            raise AssertionError("Unexpected simulator import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    x, y = toy_data()
    fit_policy(Policy(823), x, y, updates=1, row_weights=np.ones(len(x)))


def test_index_persistence_failure_stops_before_any_paid_forward():
    x, y = toy_data()
    parent, state = incoming_parent()
    meter = RecordingMeter()
    sampled = []

    def fail(row):
        sampled.append(row)
        raise OSError("Synthetic pre-forward persistence failure")

    with pytest.raises(OSError, match="pre-forward persistence") as failure:
        fit_policy(
            parent,
            x,
            y,
            optimizer_state=state,
            updates=1,
            row_weights=np.ones(len(x)),
            index_observer=fail,
            meter=meter,
        )
    assert meter.calls == []
    assert sampled[0]["fit_update"] == 1 and sampled[0]["update"] == 8
    exact_tree(sampled[0]["rng_before"], state["rng"])
    assert failure.value.fit_evidence["completed_updates"] == 0
    assert all(
        row == dict(requested=0, completed=0, unknown_partial=0)
        for row in failure.value.fit_evidence["work"].values()
    )


def test_index_observer_mutations_cannot_change_actual_indices_or_private_rng():
    x, y = toy_data()
    parent, state = incoming_parent()

    def mutate(row):
        row["indices"].fill(0)
        row["rng_before"]["state"]["state"] = 0
        row["rng_after"]["state"]["state"] = 0

    def mutate_request(row):
        row["rng_before"]["state"]["state"] = 0

    baseline = fit_policy(
        parent, x, y, optimizer_state=state, updates=2, batch_size=19, row_weights=np.ones(len(x))
    )
    observed = fit_policy(
        parent,
        x,
        y,
        optimizer_state=state,
        updates=2,
        batch_size=19,
        row_weights=np.ones(len(x)),
        index_observer=mutate,
        index_request_observer=mutate_request,
    )
    exact_tree(baseline[0].state_dict(), observed[0].state_dict())
    exact_tree(baseline[1], observed[1])
    exact_tree(baseline[2]["learning_curve"], observed[2]["learning_curve"])
