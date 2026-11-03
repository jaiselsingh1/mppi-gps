"""Synthetic J1 planning/gain calibration; no native simulator or fits."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pytest

from src.gps.mppi_gps import similarity_probe as probe
from src.gps.mppi_gps.search import CoupledSelector, IndependentMPPI
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


class ArrayEnvironment:
    action_dim = 6
    action_bounds = (-1.0, 1.0)

    def __init__(self):
        self.state = np.zeros(19)
        self.state[2] = 1.2
        self.warmstart = np.zeros(9)
        self.calls = 0

    def _get_obs(self):
        return np.r_[self.state[2:10], np.clip(self.state[10:19], -10, 10)]

    def rollout_states_to_obs(self, states):
        return np.concatenate((states[..., 2:10], np.clip(states[..., 10:19], -10, 10)), axis=-1)

    def batch_rollout(self, state, actions, initial_warmstart=None):
        self.calls += 1
        count, horizon, _ = actions.shape
        states = np.broadcast_to(state, (count, horizon, 19)).copy()
        states[..., 4:10] += np.cumsum(actions, axis=1) * 0.01
        costs = np.sum(actions**2, axis=(1, 2))
        return states, costs, np.empty((count, horizon, 0))


def config(initial=1):
    return MPPIConfig(K=4, H=3, noise_sigma=0.3, lam=3.0, initial_iterations=initial)


def test_roster_is_exact_and_nonadaptive():
    rows = probe.decision_roster()
    assert len(rows) == 208
    assert len({r["id"] for r in rows}) == 208
    assert Counter(r["role"] for r in rows) == {
        "original_repeat": 4,
        "factorial": 24,
        "gain_axis": 136,
        "gain_held": 32,
        "hidden_state": 8,
        "warmstart": 4,
    }
    assert [r["context"] for r in rows[:4]] == list(range(4))
    assert all(r["observe"] is False for r in rows[:4])
    assert probe.decision_roster() == rows
    assert probe.probe_recipe()["decisions"] == 208
    assert probe.probe_recipe()["innovation_seeds"] == [2026129701, 2026129702]
    directions = probe.held_directions()
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1)
    assert len(np.unique(directions, axis=0)) == 8


def test_private_innovation_states_do_not_advance_or_alias_source():
    source = np.random.PCG64(77).state
    old = copy.deepcopy(source)
    states = probe.innovation_states(source)
    assert states[0] == source == old
    states[0]["state"]["state"] += 1
    assert source == old
    assert states[1] != states[2]


@pytest.mark.parametrize("initial", [1, 2])
def test_iteration_observer_is_numerically_passive(initial):
    env = ArrayEnvironment()
    plain, observed = MPPI(env, config(initial), seed=3), MPPI(env, config(initial), seed=3)
    plain.U[:] = observed.U[:] = np.linspace(-0.1, 0.1, 18).reshape(3, 6)
    original = observed.snapshot()
    action, info = plain.plan_step(env.state, initial_warmstart=env.warmstart)
    with probe.IterationObserver(observed) as recorder:
        copied_action, copied_info = observed.plan_step(env.state, initial_warmstart=env.warmstart)
    np.testing.assert_array_equal(action, copied_action)
    assert info == copied_info
    np.testing.assert_array_equal(plain.U, observed.U)
    assert plain.snapshot()["rng_state"] == observed.snapshot()["rng_state"]
    assert len(recorder.passes) == initial
    assert "_sample_noise" not in observed.__dict__
    assert "_plan_iteration" not in observed.__dict__
    assert "_j1_observer_active" not in observed.__dict__
    for index, row in enumerate(recorder.passes):
        np.testing.assert_array_equal(
            row["actions"],
            np.clip(row["nominal_before"][None] + row["correlated_innovation"], -1, 1),
        )
        np.testing.assert_array_equal(
            row["effective_perturbations"], row["actions"] - row["nominal_before"][None]
        )
        np.testing.assert_allclose(row["weights"].sum(), 1.0)
        assert row["states"].shape == (4, 3, 19)
        assert row["sensordata"].shape == (4, 3, 0)
        if index:
            assert row["rng_before"] == recorder.passes[index - 1]["rng_after"]
            np.testing.assert_array_equal(
                row["nominal_before"], recorder.passes[index - 1]["nominal_after"]
            )
    np.testing.assert_array_equal(recorder.passes[0]["nominal_before"], original["U"])


def test_observer_restores_instance_methods_even_after_failure():
    native = MPPI(ArrayEnvironment(), config(), seed=3)
    native._sample_noise = lambda: (_ for _ in ()).throw(RuntimeError("synthetic failure"))
    original = native.__dict__["_sample_noise"]
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with probe.IterationObserver(native):
            native.plan_step(np.zeros(19))
    assert native.__dict__["_sample_noise"] is original
    assert "_plan_iteration" not in native.__dict__


def test_nested_observers_are_rejected_without_removing_outer_observer():
    native = MPPI(ArrayEnvironment(), config(), seed=3)
    with probe.IterationObserver(native):
        with pytest.raises(ValueError, match="Nested"):
            with probe.IterationObserver(native):
                pass
        assert native._j1_observer_active
    assert "_j1_observer_active" not in native.__dict__


def affine_design():
    axes = np.zeros((34, 17))
    for coordinate in range(17):
        axes[2 * coordinate, coordinate] = -probe.OBSERVATION_SCALES[coordinate]
        axes[2 * coordinate + 1, coordinate] = probe.OBSERVATION_SCALES[coordinate]
    held = 1.5 * probe.held_directions() * probe.OBSERVATION_SCALES
    gain = np.arange(102, dtype=float).reshape(6, 17) * 0.002 - 0.1
    return axes, held, gain


def test_known_affine_gain_and_held_directions_are_recovered():
    axes, held, gain = affine_design()
    value = probe.fit_local_gain(
        np.zeros(17),
        np.zeros(6),
        axes,
        axes @ gain.T,
        held,
        held @ gain.T,
        normalization_std=np.full(17, 2.0),
    )
    assert value["rank"] == 17 and value["full_column_rank"]
    np.testing.assert_allclose(value["gain_per_physical_observation"], gain, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        value["gain_per_normalized_observation"], 2 * gain, rtol=0, atol=1e-14
    )
    assert value["held_action_rmse"] < 1e-16
    assert value["optimal_gain_claim"] is False


def test_identical_observations_do_not_identify_hidden_action_response():
    axes, held, gain = affine_design()
    targets = axes @ gain.T
    axes[:, 16] = 0
    held_targets = held @ gain.T
    held[:, 16] = 0
    value = probe.fit_local_gain(np.zeros(17), np.zeros(6), axes, targets, held, held_targets)
    assert value["rank"] == 16
    assert value["unidentified_dimensions"] == 1
    assert value["condition_number"] is None
    assert value["training_action_rmse"] > 0
    assert value["held_action_rmse"] > 0


def test_requested_state_uses_actual_observation_coordinates():
    source = dict(state=np.arange(19, dtype=float), warmstart=np.zeros(9))
    request = next(
        r for r in probe.decision_roster() if r["role"] == "gain_axis" and r["coordinate"] == 0
    )
    state, warm = probe.requested_state(source, request)
    assert state[2] == source["state"][2] - 0.001
    np.testing.assert_array_equal(state[:2], source["state"][:2])
    np.testing.assert_array_equal(warm, source["warmstart"])
    hidden = next(r for r in probe.decision_roster() if r["time_delta"] < 0)
    state, _ = probe.requested_state(source, hidden)
    assert state[0] == -0.008
    np.testing.assert_array_equal(state[1:], source["state"][1:])


def test_unperturbed_request_preserves_signed_zero_bytes():
    source = dict(state=np.full(19, -0.0), warmstart=np.full(9, -0.0))
    state, warm = probe.requested_state(source, probe.decision_roster()[0])
    assert state.tobytes() == source["state"].tobytes()
    assert warm.tobytes() == source["warmstart"].tobytes()


def test_single_producer_original_replay_and_observer_share_exact_result():
    env = ArrayEnvironment()
    original = IndependentMPPI(env, config(), seed=9)
    private = original.private_state()
    private["initial_plan"] = False
    original.restore({"native": private})
    source = dict(state=env.state.copy(), warmstart=env.warmstart.copy())
    bank = original.propose(source["state"], source["warmstart"])
    plan, _ = CoupledSelector(alpha=0, eta_u=0.01).select(bank, actor=None, meter=None)
    context = dict(
        snapshot=source,
        planner=private,
        original_action=plan[0],
        original_selected_plan=plan,
        original_planner_after=original.private_state(),
    )

    def restore(value):
        env.state = value["state"].copy()
        env.warmstart = value["warmstart"].copy()
        return copy.deepcopy(value)

    chosen = [
        r
        for r in probe.decision_roster()
        if r["context"] == 0
        and (
            r["role"] == "original_repeat"
            or (r["role"] == "factorial" and r["nominal"] == "carried" and r["innovation"] == 0)
        )
    ]
    values = [
        probe.run_decision(
            env,
            IndependentMPPI(env, config(), seed=9),
            CoupledSelector(alpha=0, eta_u=0.01),
            context,
            row,
            restore_context=restore,
        )
        for row in chosen
    ]
    assert len(values) == 2 and all(v["original_replay_verified"] for v in values)
    np.testing.assert_array_equal(values[0]["selected_plan"], values[1]["selected_plan"])
    assert not values[0]["passes"] and len(values[1]["passes"]) == 1
    assert values[1]["contact_measurement"] == "unavailable"


def test_selector_logging_has_identical_arithmetic_and_compact_details(tmp_path):
    from src.gps.mppi_gps.similarity_execution import JMeter

    env = ArrayEnvironment()
    planner = IndependentMPPI(env, config(), seed=9)
    bank = planner.propose(env.state, env.warmstart)
    original, original_info = CoupledSelector(alpha=0, eta_u=0.01).select(
        bank, actor=None, meter=None
    )
    meter = JMeter(tmp_path / "run/worker/evidence", {})
    try:
        observed, info = CoupledSelector(alpha=0, eta_u=0.01).select(
            bank, actor=None, meter=probe._SelectorEvidence(meter, "calibration")
        )
        np.testing.assert_array_equal(original, observed)
        assert {k: v for k, v in info.items() if k != "evidence"} == {
            k: v for k, v in original_info.items() if k != "evidence"
        }
        assert info["evidence"]["arrays"]["bank_actions"]["sha256"]
        lines = meter.ledger_path.read_bytes().splitlines()
        assert all(len(line) + 1 <= 512 for line in lines[1:])
        assert b"details_group" in lines[-1]
    finally:
        meter.close()


def synthetic_results():
    gain = affine_design()[2]
    results = []
    for request in probe.decision_roster():
        observation = np.asarray(request["observation_delta"])
        action = gain @ observation
        original = request["role"] == "original_repeat" or (
            request["role"] == "factorial"
            and request["nominal"] == "carried"
            and request["innovation"] == 0
        )
        contact = np.array([True, False])
        if request["role"] == "gain_axis" and request["coordinate"] == 0:
            contact[0] = False
        results.append(
            dict(
                request=request,
                observation=observation,
                action=action,
                selected_plan=np.tile(action, (3, 1)),
                planner_after=dict(U=np.zeros((3, 6)), rng_state={"source": 1}, initial_plan=False),
                original_replay_verified=original,
                contacts={"presence": contact},
            )
        )
    return results


def test_summary_reconciles_contact_changes_without_claiming_optimality():
    results = synthetic_results()
    value = probe.summarize_probe(results)
    assert value["decisions"] == 208 and value["original_replay_checks"] == 8
    assert len(value["local_fits"]) == 4
    assert all(v["axis_contact_regime_changes"] == 2 for v in value["local_fits"])
    assert all(v["held_contact_regime_changes"] == 0 for v in value["local_fits"])
    assert len(value["hidden_input_contrasts"]) == 12
    assert value["unexplained_variation_is_irrelevant"] is False


@pytest.mark.parametrize("change", ["missing", "duplicate", "request", "gate", "observer"])
def test_summary_rejects_missing_work_mutated_protocol_or_failed_gate(change):
    rows = synthetic_results()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "request":
        rows[-1]["request"]["warmstart_delta"] = 9
    elif change == "gate":
        rows[0]["original_replay_verified"] = False
    else:
        rows[4]["action"][0] += 0.1
    with pytest.raises(ValueError):
        probe.summarize_probe(rows)
