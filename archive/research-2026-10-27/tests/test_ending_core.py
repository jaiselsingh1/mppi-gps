"""Core checks with raw model-call and arithmetic-verification accounting."""
import inspect
import json

import numpy as np
import pytest

from src.ending.core import (
    Actor, IndependentMPPI, Task, actor_features, capacity_witness_theta,
    local_lower_bound, pure_actor_actions,
)


@pytest.fixture(autouse=True)
def controls(request, monkeypatch, tmp_path):
    """Keep per-test raw model calls in pytest's temporary artifact directory."""
    transition = Task.transition
    lower_bound = Task.local_lower_bound
    ledger = {"test": request.node.nodeid, "simulator_controls": 0,
              "arithmetic_verification_controls": 0, "model_calls": []}

    def counted_transition(self, states, actions):
        result = transition(self, states, actions)
        count = int(np.prod(np.asarray(states).shape[:-1], dtype=int))
        ledger["simulator_controls"] += count
        next_states, components, total, flags = result
        ledger["model_calls"].append({"states": np.asarray(states).tolist(),
            "actions": np.asarray(actions).tolist(), "next_states": next_states.tolist(),
            "components": {name: np.asarray(values).tolist() for name, values in components.items()},
            "total": np.asarray(total).tolist(),
            "flags": {name: np.asarray(values).tolist() for name, values in flags.items()},
            "controls": count})
        return result

    def counted_bound(self, state, h):
        result = lower_bound(self, state, h)
        ledger["arithmetic_verification_controls"] += result["verification_controls"]
        return result

    monkeypatch.setattr(Task, "transition", counted_transition)
    monkeypatch.setattr(Task, "local_lower_bound", counted_bound)
    yield ledger
    destination = tmp_path / "ending-core-controls.json"
    destination.write_text(json.dumps(ledger, indent=2) + "\n")
    print("ENDING_CORE_TEST_ACCOUNTING " + json.dumps({
        key: value for key, value in ledger.items() if key != "model_calls"
    }) + " artifact=" + str(destination))


def test_exact_component_boundaries_and_absorbing_terminal(controls):
    task = Task()
    assert task.T == 25
    states = np.array([[0., .3], [11., -.2], [12., -.1], [16., -.2],
                       [17., -.2], [24., .2], [25., .5]])
    next_states, components, costs, flags = task.transition(states, np.array([[1.], [-1.], [1.],
        [1.], [1.], [1.], [1.]]))
    np.testing.assert_allclose(next_states[:, 1], [.5, -.4, -.1, -.2, -.2, .2, .5])
    np.testing.assert_array_equal(next_states[:, 0], [1., 12., 13., 17., 18., 25., 25.])
    np.testing.assert_allclose(components["effort"], [.02, .02, 0., 0., 0., 0., 0.])
    np.testing.assert_allclose(components["greed"], [.05, .14, 0., 0., 0., 0., 0.])
    np.testing.assert_allclose(components["detour"], [0., 0., .2, .2, 0., 0., 0.])
    np.testing.assert_allclose(components["ending"], [0., 0., 0., 0., 0., 20., 0.])
    np.testing.assert_allclose(costs, [.07, .16, .2, .2, 0., 20., 0.])
    np.testing.assert_array_equal(flags["ending_event"], [False] * 5 + [True, False])
    assert flags["terminated"][-2:].all()
    _, _, cost, good = task.transition(np.array([24., -.1]), np.array([0.]))
    assert cost == 0. and good["ending_event"] and good["ending_good"]
    clipped, _, _, _ = task.transition(np.array([[0., .95], [0., -.95]]), [[1.], [-1.]])
    np.testing.assert_array_equal(clipped[:, 1], [1., -1.])
    assert controls["simulator_controls"] == 10


def test_rollout_scalar_parity_and_charged_terminal_padding(controls):
    task = Task(gate=2, delay=1, detour_steps=1)
    state = np.array([0., .3])
    plans = np.repeat(np.array([[[-1.]], [[1.]], [[0.]]]), 6, axis=1)
    result = task.rollout(state, plans)
    for row, plan in enumerate(plans):
        current = state.copy()
        for index, action in enumerate(plan):
            current, components, cost, flags = task.transition(current, action)
            np.testing.assert_array_equal(result["states"][row, index], current)
            assert result["stage_costs"][row, index] == cost
            for name, value in components.items():
                assert result["stage_components"][name][row, index] == value
            for name, value in flags.items():
                assert result["flags"][name][row, index] == value
    assert result["charged_controls"] == 18
    assert result["padding_controls"] == 6
    np.testing.assert_array_equal(result["stage_costs"][:, 4:], 0.)
    assert result["flags"]["terminated"][:, 3:].all()
    np.testing.assert_allclose(result["totals"], result["stage_costs"].sum(axis=1))
    assert controls["simulator_controls"] == 36


def test_actor_features_are_pure_and_witness_is_only_representational(controls):
    task = Task()
    states = np.array([[0., .3], [12., -.2], [25., .5]])
    np.testing.assert_array_equal(actor_features(states, task), [[1., .3, 0.], [1., -.2, 12 / 25],
                                                               [1., .5, 1.]])
    parameters = np.array([.1, .2, -.3])
    actor = Actor(parameters)
    expected = np.tanh(np.array([.16, .1 - .04 - .3 * 12 / 25, .1 + .1 - .3]))[:, None]
    np.testing.assert_allclose(actor.actions(states, task), expected)
    np.testing.assert_array_equal(pure_actor_actions(parameters, states, task), actor(states, task))
    parameters[:] = 0.
    np.testing.assert_allclose(actor.actions(states, task), expected)
    np.testing.assert_allclose(pure_actor_actions(capacity_witness_theta(), states, task), -.9)
    with pytest.raises(ValueError):
        actor.theta.setflags(write=True)
    assert "actor" not in inspect.signature(IndependentMPPI).parameters
    assert "critic" not in inspect.signature(IndependentMPPI.plan).parameters
    assert controls["simulator_controls"] == 0


def test_lower_bound_uses_reachability_and_charges_arithmetic_checks(controls):
    task = Task()
    state = np.array([0., .2])
    bound = local_lower_bound(state, 3, task)
    np.testing.assert_allclose(bound["stage_bounds"], [.06, .04, .02])
    np.testing.assert_allclose(bound["max_y_by_step"], [.4, .6, .8])
    assert bound["bound"] == pytest.approx(.12)
    assert bound["verification_controls"] == 3
    result = task.rollout(state, np.ones((1, 3, 1)))
    assert bound["bound"] <= result["totals"][0]
    final = task.local_lower_bound(np.array([11., .9]), 4)
    assert final["bound"] == 0.
    np.testing.assert_array_equal(final["max_y_by_step"], 1.)
    assert controls["arithmetic_verification_controls"] == 7
    assert controls["simulator_controls"] == 3


def test_mppi_proposal_overrides_verified_mean_and_own_nominal_retention(controls):
    task = Task()
    planner = IndependentMPPI(task, K=8, H=4, seed=3)
    result = planner.plan(task.reset(4))
    assert result["plans"].shape == (9, 4, 1)
    np.testing.assert_array_equal(result["plans"][0], result["nominal_pre"])
    for row, constant in enumerate((0., 1., -1., .5, -.5), start=1):
        np.testing.assert_array_equal(result["plans"][row], constant)
    weights = np.exp(-(result["totals"][:8] - result["totals"][:8].min()) / .2)
    weights /= weights.sum()
    np.testing.assert_array_equal(result["weights"], weights)
    np.testing.assert_allclose(result["meanplan"],
                               np.sum(result["plans"][:8] * weights[:, None, None], axis=0))
    np.testing.assert_array_equal(result["plans"][-1], result["meanplan"])
    assert result["selected"] == result["totals"].argmin()
    np.testing.assert_array_equal(result["nominal_post"][:-1], result["plans"][result["selected"], 1:])
    np.testing.assert_array_equal(result["nominal_post"][-1], 0.)
    np.testing.assert_array_equal(planner.nominal, result["nominal_post"])
    assert result["workcounts"] == {"candidate_controls": 32, "verification_controls": 4,
                                    "total_controls": 36, "padding_controls": 0}
    for name in ("plans", "states", "totals", "nominal_pre", "nominal_post"):
        with pytest.raises(ValueError):
            result[name].setflags(write=True)
    assert controls["simulator_controls"] == 36


def test_snapshot_replay_is_bitwise_for_rng_proposals_and_selected_prefix(controls):
    task = Task()
    planner = IndependentMPPI(task, K=8, H=3, seed=9)
    snapshot = planner.snapshot()
    # JSON round trip exercises the representation written into raw evidence.
    snapshot = json.loads(json.dumps(snapshot))
    state = task.reset(6)
    first = planner.plan(state)
    next_state = first["states"][first["selected"], 0]
    second = planner.plan(next_state)
    planner.restore(snapshot)
    repeated_first = planner.plan(state)
    repeated_second = planner.plan(next_state)
    for original, repeated in ((first, repeated_first), (second, repeated_second)):
        for key in ("action", "plans", "states", "totals", "weights", "meanplan",
                    "nominal_pre", "nominal_post"):
            np.testing.assert_array_equal(original[key], repeated[key])
        assert original["rng_pre"] == repeated["rng_pre"]
        assert original["rng_post"] == repeated["rng_post"]
        assert original["selected"] == repeated["selected"]
    assert controls["simulator_controls"] == 108


def test_private_rng_final_window_and_exact_ties_prefer_own_nominal(controls):
    task = Task(effort=0., greed=0., detour=0., penalty=0.)
    global_before = np.random.get_state()
    first = IndependentMPPI(task, K=8, H=4, seed=7).plan(np.array([24., .3]))
    second = IndependentMPPI(task, K=8, H=4, seed=7).plan(np.array([24., .3]))
    assert first["h"] == 1 and first["plans"].shape == (9, 1, 1)
    assert first["selected"] == 0
    np.testing.assert_array_equal(first["plans"], second["plans"])
    np.testing.assert_array_equal(first["nominal_post"], np.zeros((4, 1)))
    global_after = np.random.get_state()
    assert all(np.array_equal(before, after) for before, after in zip(global_before, global_after))
    assert controls["simulator_controls"] == 18


def test_invalid_configuration_and_terminal_planning_fail_before_model_queries(controls):
    with pytest.raises(ValueError, match="gate"):
        Task(gate=0)
    with pytest.raises(ValueError, match="K"):
        IndependentMPPI(Task(), K=5)
    with pytest.raises(ValueError, match="absorbing"):
        IndependentMPPI(Task(), K=8).plan(np.array([25., .2]))
    with pytest.raises(ValueError, match="configuration"):
        IndependentMPPI(Task(), K=8).restore(IndependentMPPI(Task(), K=9).snapshot())
    assert controls["simulator_controls"] == 0
