"""Synthetic C1 math, admission and durable inherited-stream fitting checks."""

from __future__ import annotations

import builtins
import hashlib
from copy import deepcopy

import numpy as np
import pytest
import torch

from src.gps.mppi_gps import consequence_learning as c1
from src.gps.mppi_gps.policy import Policy, fit_policy
from src.gps.mppi_gps.provenance import fingerprint, sha


class Meter:
    def __init__(self):
        self.work, self.completed, self.events = {}, {}, []
        self.before_forward = None
        self.after_prediction = None

    def reserve_external(self, amount):
        assert amount <= 24 * 1024**2

    def reserve_work(self, kind, amount=1):
        self.work[kind] = self.work.get(kind, 0) + amount

    def returned(self, kind, amount):
        self.completed[kind] = self.completed.get(kind, 0) + amount

    def record(self, event, **fields):
        self.events.append(dict(event=event, **fields))

    def measure(self, kind, amount, operation, **metadata):
        self.reserve_work(kind, amount)
        if kind == "neural_rows" and self.before_forward is not None:
            self.before_forward()
        value = operation()
        if kind == "neural_rows" and self.after_prediction is not None:
            self.after_prediction(value)
        self.returned(kind, amount)
        return value


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_weight_rule_exact_native_units_and_float32_conversion():
    result = c1.cost_effect_weights(np.array([-100, 0, 20, 100, 180, 1000.0]))
    np.testing.assert_array_equal(result["raw"], [1, 1, 2, 6, 10, 10])
    assert result["mean_raw"] == 5
    np.testing.assert_array_equal(result["normalized"], result["raw"] / 5)
    np.testing.assert_array_equal(result["weights"], (result["raw"] / 5).astype(np.float32))


@pytest.mark.parametrize("bad", [[], [[1]], [np.nan], [np.inf], ["wrong"]])
def test_weight_rule_rejects_missing_nonfinite_or_nonnumeric_deltas(bad):
    with pytest.raises(ValueError):
        c1.cost_effect_weights(bad)


def test_shuffle_has_fixed_groups_and_exact_within_group_multisets_and_rng():
    weights = np.linspace(0.1, 1.9, 4000, dtype=np.float32)
    # Ties force the control-index tie rule, including horizon groups.
    errors = np.zeros(4000, dtype=np.float64)
    observed = []
    shuffled = c1.shuffle_stratified(weights, errors, 2800000000, observed.append)
    assert shuffled["permutation_calls"] == 208 and shuffled["returned_entries"] == 4000
    assert len(observed) == 416
    reference = np.random.Generator(np.random.PCG64(2800000000))
    for index, group in enumerate(shuffled["groups"]):
        destination = group["destination"]
        assert np.all(np.diff(destination) > 0)
        np.testing.assert_array_equal(group["source"], reference.permutation(destination))
        np.testing.assert_array_equal(
            np.sort(weights[destination]), np.sort(shuffled["shuffled"][destination])
        )
        assert observed[2 * index]["event"] == "before"
        assert observed[2 * index + 1]["event"] == "after"
        assert fingerprint(observed[2 * index]["rng"]) == fingerprint(group["rng_before"])
    np.testing.assert_array_equal(np.sort(shuffled["source_indices"]), np.arange(4000))
    # Listed remaining-horizon order follows nine full temporal blocks.
    assert shuffled["groups"][36]["destination"].tolist() == [990, 991, 992]
    assert shuffled["groups"][40]["destination"].tolist() == [975, 976, 977, 978]
    assert shuffled["groups"][44]["destination"].tolist() == list(range(950, 957))
    assert shuffled["groups"][48]["destination"].tolist() == list(range(900, 913))
    assert fingerprint(shuffled["rng_terminal"]) == fingerprint(reference.bit_generator.state)


def test_shuffle_observer_failure_stops_before_draw_without_retry():
    events = []

    def stop(value):
        events.append(value)
        raise OSError("synthetic persistence failure")

    with pytest.raises(OSError, match="persistence"):
        c1.shuffle_stratified(np.ones(4000, np.float32), np.zeros(4000), 7, stop)
    assert len(events) == 1 and events[0]["event"] == "before"


def test_global_contrast_gate_uses_actual_float32_degeneracy_and_never_redraws():
    weight = np.linspace(0.5, 1.5, 4000, dtype=np.float32)
    streams = {s: dict(cost_effect=weight.copy(), shuffled=weight[::-1].copy()) for s in range(3)}
    assert c1.contrast_gate(streams)["passed"]
    streams[2] = dict(cost_effect=np.ones(4000, np.float32), shuffled=np.ones(4000, np.float32))
    result = c1.contrast_gate(streams)
    assert not result["passed"] and not result["redraw_allowed"]
    assert len(result["reasons"]) == 2
    # A nonconstant float64 rule can become constant in the actual loss dtype.
    narrow = (1 + np.arange(4000) * 1e-15).astype(np.float32)
    streams[2] = dict(cost_effect=narrow, shuffled=narrow[::-1].copy())
    assert not c1.contrast_gate(streams)["passed"]


def panel():
    return {
        s: {arm: dict(successes=0, survived=5, episodes=16) for arm in c1.ARMS} for s in range(3)
    }


def test_learning_requires_walking_all_comparators_and_unconditional_confirmation():
    primary, confirmation = panel(), panel()
    for values in (primary, confirmation):
        for stream in (0, 1):
            values[stream]["cost_effect"]["successes"] = 2
    assert c1.learning_gates(primary, confirmation)["confirmed"]
    confirmation[2]["cost_effect"]["survived"] = 4
    assert not c1.learning_gates(primary, confirmation)["confirmed"]
    confirmation[2]["cost_effect"]["survived"] = 16
    confirmation[0]["shuffled"]["successes"] = 1
    assert not c1.learning_gates(primary, confirmation)["confirmed"]


def test_longer_survival_without_walking_is_not_success():
    values = panel()
    for actors in values.values():
        actors["cost_effect"]["survived"] = 16
        actors["uniform"]["survived"] = 16
    assert not c1.panel_gate(values)["passed"]
    assert not c1.uniform_parent_gate(values)["passed"]
    values[0]["cost_effect"]["successes"] = 16
    assert not c1.panel_gate(values)["passed"]
    values[0]["parent"]["episodes"] = 15
    with pytest.raises(ValueError, match="sixteen"):
        c1.panel_gate(values)


def test_confirmation_requires_the_same_two_qualifying_streams_on_both_panels():
    primary, confirmation = panel(), panel()
    for values, streams in ((primary, (0, 1)), (confirmation, (1, 2))):
        for stream in streams:
            values[stream]["cost_effect"]["successes"] = 4
            values[stream]["uniform"]["successes"] = 2
    result = c1.learning_gates(primary, confirmation)
    assert result["primary"]["passed"] and result["confirmation"]["passed"]
    assert result["uniform_parent_primary"]["passed"]
    assert result["uniform_parent_confirmation"]["passed"]
    assert result["confirmed_streams"] == result["uniform_parent_confirmed_streams"] == ["1"]
    assert not result["confirmed"] and not result["uniform_parent_confirmed"]


def test_weighted_gain_must_beat_all_controls_in_same_streams():
    values = panel()
    for stream in range(3):
        values[stream]["cost_effect"]["successes"] = 3
        values[stream][("parent", "uniform", "shuffled")[stream]]["successes"] = 2
    assert not c1.panel_gate(values)["passed"]


def test_native_sum_six_and_fitting_mse_are_different_targets_not_just_units():
    prediction = np.ones((2, 6), dtype=np.float32)
    target = np.ones((2, 6), dtype=np.float64) + 1e-9
    native = c1.native_action_error(prediction, target)
    fit = c1.fitting_mse(prediction, target, np.array([0.5, 1.5], np.float32))
    assert np.all(native > 0)
    assert fit == dict(action_mse=0.0, weighted_action_mse=0.0)
    prediction[:] = 0
    assert c1.fitting_mse(prediction, target)["action_mse"] == 1.0


def make_parent(updates=2):
    rng = np.random.Generator(np.random.PCG64(901))
    x = rng.normal(0, 0.1, (4000, 17)).astype(np.float32)
    y = rng.uniform(-0.2, 0.2, (4000, 6)).astype(np.float64)
    policy = Policy(seed=902)
    parent, state, _ = fit_policy(
        policy, x, y, updates=updates, batch_size=256, eta_theta=0, prox_reduction="mean", seed=903
    )
    return parent, state, dict(observations=x, actions=y)


def checkpoint_fixture(tmp_path):
    parent, state, data = make_parent()
    path = tmp_path / "parent.pt"
    record = parent.checkpoint(path, optimizer_state=state)
    global_rng = torch.Generator(device="cpu").manual_seed(904).get_state().numpy()
    descriptor = dict(
        bytes=record["bytes"],
        sha256=record["sha256"],
        actor_hash=parent.hash(),
        incoming_optimizer_sha256=fingerprint(state),
        original_checkpoint_record=dict(
            global_rng=dict(
                key="state",
                reference=dict(
                    arrays=dict(
                        state=dict(
                            shape=list(global_rng.shape),
                            dtype=global_rng.dtype.str,
                            sha256=hashlib.sha256(global_rng.tobytes()).hexdigest(),
                        )
                    )
                ),
            )
        ),
    )
    return parent, state, data, path, descriptor, global_rng


def test_preparation_validates_saved_continuation_without_policy_construction(
    tmp_path, monkeypatch
):
    parent, _, _, path, descriptor, global_rng = checkpoint_fixture(tmp_path)
    global_before = torch.get_rng_state().clone()
    monkeypatch.setattr(Policy, "__init__", lambda *a, **kw: pytest.fail("Policy construction"))
    result = c1.validate_parent_checkpoint(
        path,
        descriptor,
        parent.normalization,
        global_rng,
        expected_updates=2,
        expected_global_seed=904,
    )
    assert result["passed"] and result["policy_constructions"] == result["network_forwards"] == 0
    assert torch.equal(global_before, torch.get_rng_state())


@pytest.mark.parametrize(
    "change", ["file", "optimizer", "normalization", "global", "adam", "private"]
)
def test_parent_validation_stops_changed_continuation_or_sidecar(tmp_path, change):
    parent, _, _, path, descriptor, global_rng = checkpoint_fixture(tmp_path)
    normalization = parent.normalization
    if change == "file":
        descriptor["sha256"] = "0" * 64
    elif change == "optimizer":
        descriptor["incoming_optimizer_sha256"] = "0" * 64
    elif change == "normalization":
        normalization["std"][0] *= 2
    elif change == "global":
        global_rng[0] ^= 1
    else:
        payload = torch.load(path, weights_only=True)
        if change == "adam":
            payload["optimizer_state"]["adam"]["state"][0]["step"].fill_(1)
        else:
            payload["private_rng_state"] = torch.zeros(4, dtype=torch.uint8)
        torch.save(payload, path)
        descriptor.update(bytes=path.stat().st_size, sha256=sha(path))
    with pytest.raises((ValueError, RuntimeError)):
        c1.validate_parent_checkpoint(
            path,
            descriptor,
            normalization,
            global_rng,
            expected_updates=2,
            expected_global_seed=904,
        )


def test_full_durable_fit_uses_one_shared_expected_tape_and_exact_continuation(tmp_path):
    parent, state, data = make_parent(updates=2000)
    meter = Meter()
    before = fingerprint(state)
    expected = c1.generate_expected_tape(state, tmp_path / "expected.npy", meter, "s0")
    assert meter.work["expected_index_draws"] == 512000
    assert meter.completed["expected_index_draws"] == 512000
    assert len(meter.events) == 4000
    directory = tmp_path / "learning" / "s0_uniform_child"

    def before_forward():
        sampled = meter.work.get("actual_index_draws", 0) // 256
        actual = np.load(directory / "actual_indices.npy", mmap_mode="r")
        np.testing.assert_array_equal(actual[sampled - 1], expected["tape"][sampled - 1])
        assert meter.events[-1]["event"] == "index_after"

    meter.before_forward = before_forward
    result = c1.fit_arm(
        parent,
        state,
        data,
        None,
        905,
        "s0_uniform_child",
        tmp_path / "learning",
        expected["tape"],
        meter,
    )
    record, continuation = result["record"], result["optimizer_state"]
    assert fingerprint(state) == before
    assert continuation["updates"] == 4000 and continuation["fit_updates"] == 2000
    assert fingerprint(continuation["rng"]) == fingerprint(expected["rng_terminal"])
    assert record["checkpoint_loads"] == 0 and record["policy_constructions"] == 1
    assert meter.completed["policy_constructions"] == meter.completed["endpoint_saves"] == 1
    assert meter.completed["actual_index_draws"] == meter.completed["neural_rows"] == 512000
    assert meter.completed["backward_rows"] == 512000 and meter.completed["optimizer_steps"] == 2000
    np.testing.assert_array_equal(np.load(directory / "actual_indices.npy"), expected["tape"])
    curve = np.load(directory / "curve.npy")
    assert np.all(curve["valid"] == 1)
    assert curve["update"].tolist() == list(range(2001, 4001))
    assert curve["action_mse"].tolist() == [
        row["action_mse"] for row in result["evidence"]["learning_curve"]
    ]
    predictions = np.load(directory / "predictions.npy", mmap_mode="r")
    assert predictions.shape == (2000, 256, 6) and predictions.dtype == np.float32
    assert np.isfinite(predictions).all()
    assert np.all(np.load(directory / "prediction_valid.npy") == 1)
    first_targets = data["actions"].astype(np.float32)[expected["tape"][0]]
    np.testing.assert_allclose(
        np.mean((predictions[0] - first_targets) ** 2), curve["action_mse"][0], rtol=1e-6
    )
    assert meter.after_prediction is None
    checkpoint = torch.load(directory / "endpoint.pt", weights_only=True)
    assert "row_weights" not in checkpoint and set(checkpoint["optimizer_state"]) == set(state)
    # Create-only and externally supplied tape contracts hold before paid work.
    paid_before = deepcopy(meter.work)
    with pytest.raises(ValueError, match="already exists"):
        c1.fit_arm(
            parent,
            state,
            data,
            None,
            905,
            "s0_uniform_child",
            tmp_path / "learning",
            expected["tape"],
            meter,
        )
    assert meter.work == paid_before


def test_changed_expected_tape_stops_before_first_forward_and_preserves_failure(tmp_path):
    parent, state, data = make_parent()
    # Synthetic continuation metadata at2k: this fixture never gets to a forward.
    state["updates"] = state["fit_updates"] = 2000
    expected = np.zeros((2000, 256), dtype=np.int64)
    meter = Meter()
    with pytest.raises(ValueError, match="Actual minibatch"):
        c1.fit_arm(
            parent, state, data, np.ones(4000, np.float32), 9, "bad", tmp_path, expected, meter
        )
    assert meter.work.get("neural_rows", 0) == 0
    assert (tmp_path / "bad/failure.json").is_file()
    actual = np.load(tmp_path / "bad/actual_indices.npy")
    assert np.all(actual[0] >= 0) and np.all(actual[1:] == -1)
    assert meter.work["actual_index_draws"] == meter.completed["actual_index_draws"] == 256
    assert np.isnan(np.load(tmp_path / "bad/predictions.npy")).all()
    assert np.all(np.load(tmp_path / "bad/prediction_valid.npy") == 0)
    assert meter.after_prediction is None


def test_fitting_and_metadata_validation_never_import_simulators(tmp_path, monkeypatch):
    parent, _, _, path, descriptor, global_rng = checkpoint_fixture(tmp_path)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("mujoco", "gym", "dm_control")):
            pytest.fail("Simulator import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert c1.validate_parent_checkpoint(
        path,
        descriptor,
        parent.normalization,
        global_rng,
        expected_updates=2,
        expected_global_seed=904,
    )["passed"]
