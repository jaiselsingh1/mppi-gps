"""Falsification checks for independent proposals and measured memory semantics."""
import numpy as np
import pytest

from src.gps.independent_core import Ledger, Memory, Search, Task, independent_bank


def _episode(identifier="e", root="r", *, seed=1, length=20, remaining_start=None,
             action=.5):
    task = Task()
    remaining_start = length if remaining_start is None else remaining_start
    states = [task.reset(seed)]
    costs = []
    actions = []
    for _ in range(length):
        next_state, cost, done = task.transition(states[-1], np.array([action]))
        states.append(next_state)
        actions.append([action])
        costs.append(float(cost))
        if done:
            break
    return {"states": np.asarray(states), "actions": np.asarray(actions),
            "returns": np.cumsum(costs[::-1])[::-1], "id": identifier, "root": root,
            "remaining_start": remaining_start}


def test_transition_matches_equations_and_bounds():
    task = Task()
    states = np.array([[-.5, 0.], [-1.2, -.03], [.4, .07]])
    next_states, costs, done = task.transition(states, np.array([1., -1., 1.]))
    expected_velocity = .0015 - .0025 * np.cos(-1.5)
    np.testing.assert_allclose(next_states[0], [-.5 + expected_velocity, expected_velocity])
    np.testing.assert_array_equal(next_states[1], [-1.2, 0.])
    assert next_states[2, 1] == .07 and next_states[2, 0] == pytest.approx(.47)
    np.testing.assert_allclose(costs, 1.02)
    np.testing.assert_array_equal(done, [False, False, True])
    np.testing.assert_array_equal(task.transition(states, [[1], [-1], [1]])[0], next_states)


def test_successful_transition_charged_once_then_absorbing():
    task = Task()
    reached, cost, done = task.transition(np.array([.44, .03]), [0.])
    assert done and cost == 1.
    for action in [-1., 0., 1.]:
        after, cost, done = task.transition(reached, [action])
        np.testing.assert_array_equal(after, reached)
        assert done and cost == 0.
    assert task.heuristic(reached) == 0.


def test_rollout_equals_actual_steps_and_charges_absorbing_and_padding_slots():
    task = Task()
    state = np.array([.43, .02])
    plans = np.array([[[0.], [1.], [-1.], [1.]],
                      [[-1.], [-1.], [1.], [1.]]])
    result = task.rollout(state, plans, remaining=2)
    for index, plan in enumerate(plans):
        current = state.copy()
        total = 0.
        for step, action in enumerate(plan):
            if step < 2:
                current, cost, _ = task.transition(current, action)
                total += cost
            np.testing.assert_array_equal(result["states"][index, step], current)
        np.testing.assert_array_equal(result["endpoints"][index], current)
        assert result["costs"][index] == total
    assert result["charged_controls"] == 8
    np.testing.assert_array_equal(result["lengths"], [4, 4])
    np.testing.assert_array_equal(result["stage"][:, 2:], 0.)
    empty = task.rollout(state, plans, remaining=0)
    assert empty["charged_controls"] == 8
    np.testing.assert_array_equal(empty["costs"], [0., 0.])


def test_private_seeded_proposals_and_reset_do_not_consume_global_rng():
    global_before = np.random.get_state()
    first = independent_bank(42, 64, 23)
    task = Task()
    initial = task.reset(42)
    np.testing.assert_array_equal(initial, task.reset(42))
    assert -.6 <= initial[0] < -.4 and initial[1] == 0.
    np.testing.assert_array_equal(first, independent_bank(42, 64, 23))
    assert not np.array_equal(first, independent_bank(43, 64, 23))
    np.testing.assert_array_equal(first[0], 0.)
    assert first.min() >= -1. and first.max() <= 1.
    np.testing.assert_array_equal(first[32:48, :16],
                                  np.repeat(first[32:48, :1], 16, axis=1))
    np.testing.assert_array_equal(first[48:], np.repeat(first[48:, :1], 23, axis=1))
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(global_before, after))


def test_memory_uses_explicit_remaining_and_rejects_incomplete_failure():
    episode = _episode(length=20)
    memory = Memory([episode])
    expected = np.column_stack((episode["states"][:-1, 0] / 1.8,
                                episode["states"][:-1, 1] / .07,
                                np.arange(20, 0, -1) / 180.))
    np.testing.assert_array_equal(memory.features, expected)
    target = memory.tail(episode["states"][3], remaining=17)
    assert target["distances"] == 0.
    assert target["values"] == episode["returns"][3]
    assert target["observed_remaining"] == 17
    assert target["supported"]
    assert memory.tail(episode["states"][3], remaining=180)["distances"] > .25
    assert not memory.tail(episode["states"][3], remaining=180)["supported"]
    assert memory.tail(episode["states"][3], remaining=0)["values"] == 0.
    truncated = _episode(length=20, remaining_start=30)
    with pytest.raises(ValueError, match="truncated"):
        Memory([truncated])


def test_memory_balances_roots_is_immutable_and_preserves_recorded_suffixes():
    long = _episode(identifier="long", root="a", length=80, action=.75)
    short = _episode(identifier="short", root="b", length=4, action=-.25)
    memory = Memory([long, short], max_rows=10)
    assert len(memory.rows) == 10
    assert sum(row.startswith("short:") for row in memory.row_ids) == 4
    assert sum(row.startswith("long:") for row in memory.row_ids) == 6
    digest = memory.digest
    long["actions"][:] = -1.
    short["states"][:] = 0.
    assert memory.digest == digest
    assert Memory([_episode(identifier="long", root="a", length=80, action=.75),
                   _episode(identifier="short", root="b", length=4, action=-.25)],
                  max_rows=10).digest == digest
    for array in [memory.states, memory.features, memory.returns,
                  memory.episodes[0]["actions"]]:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    plans, rows = memory.proposals(memory.episodes[1]["states"][0], H=7, count=8,
                                   remaining=4)
    index = rows.index("short:0")
    np.testing.assert_array_equal(plans[index], np.full((7, 1), -.25))
    assert len(rows) == len(set(rows)) == 8


def test_empty_memory_has_no_fake_support_or_proposals():
    memory = Memory([])
    assert memory.features.shape == (0, 3)
    plans, rows = memory.proposals(np.array([-.5, 0.]), H=5)
    assert plans.shape == (0, 5, 1) and rows == ()
    tail = memory.tail(np.array([[-.5, 0.], [.5, 0.]]), remaining=10)
    np.testing.assert_array_equal(tail["supported"], [False, True])
    assert np.isinf(tail["distances"]).all()


def test_indexed_memory_matches_exhaustive_lookup_and_resolves_duplicate_rows():
    episodes = [_episode(identifier=f"e{index}", root=f"r{index}", seed=index,
                         length=80) for index in range(6)]
    # Identical traces create exact nearest-neighbor ties at every row.
    episodes.append(_episode(identifier="duplicate", root="z", seed=0, length=80))
    memory = Memory(episodes)
    states = np.array([[-.55, .001], [-.3, -.04], [-.7, .02], episodes[0]["states"][5]])
    remaining = np.array([70, 30, 10, 75])
    features = memory._features(states, remaining)
    exhaustive = np.linalg.norm(features[:, None] - memory.features[None], axis=-1)
    expected = exhaustive.argmin(axis=1)
    tail = memory.tail(states, remaining)
    np.testing.assert_array_equal(tail["indices"], expected)
    np.testing.assert_allclose(tail["distances"], exhaustive[np.arange(len(states)), expected])
    np.testing.assert_array_equal(tail["values"], memory.returns[expected])
    distances, indices = memory._nearest(features, 64)
    expected_order = np.argsort(exhaustive, axis=1, kind="stable")[:, :64]
    np.testing.assert_array_equal(indices, expected_order)
    np.testing.assert_allclose(distances, np.take_along_axis(exhaustive, expected_order, axis=1))


def test_guidance_and_memory_cannot_seed_or_change_independent_proposals():
    task = Task()
    state = task.reset(2)
    plain = Search(task, K=16, H=9).plan(state, 100, seed=42)
    memory = Memory([_episode(seed=2, length=100)])
    strong = Search(task, K=16, H=9, memory=memory).plan(state, 100, seed=42)

    def adversary(bank):
        np.testing.assert_array_equal(bank["state"], state)
        with pytest.raises(ValueError):
            bank["state"].setflags(write=True)
        with pytest.raises(TypeError):
            bank["plans"] = np.zeros_like(bank["plans"])
        with pytest.raises(ValueError):
            bank["plans"].setflags(write=True)
        with pytest.raises(ValueError):
            bank["plans"][0, 0, 0] = -1.
        scores = np.zeros(len(bank["plans"]))
        scores[-1] = -1.
        return {"scores": scores, "supported_count": len(scores)}

    guided = Search(task, K=16, H=9, memory=memory).plan(state, 100, seed=42,
                                                       guidance=adversary)
    for key in ["bank_plans", "bank_endpoints", "bank_states", "bank_stage_costs"]:
        np.testing.assert_array_equal(plain[key][:16], strong[key][:16])
        np.testing.assert_array_equal(strong[key], guided[key])
    np.testing.assert_array_equal(plain["bank_plans"][16], strong["bank_plans"][16])
    np.testing.assert_array_equal(strong["local_scores"], guided["local_scores"])
    assert guided["selected_index"] == len(guided["bank_plans"]) - 1
    np.testing.assert_array_equal(guided["action"], guided["bank_actions"][-1])
    assert guided["selection"]["supported_count"] == len(guided["bank_plans"])
    query = guided["query"]
    assert query["independent_controls"] == 16 * 9
    assert query["verification_controls"] == 9
    assert query["memory_controls"] == 8 * 9
    assert query["simulated_controls"] == 25 * 9


def test_baseline_uses_measured_memory_but_mppi_mean_uses_independent_heuristic():
    task = Task()
    state = task.reset(3)
    memory = Memory([_episode(seed=3, length=100)])
    result = Search(task, K=16, H=5, memory=memory).plan(state, 100, seed=7)
    bank = result["bank"]
    tail = memory.tail(bank["endpoints"], remaining=95)
    expected = bank["stage_costs"] + np.where(tail["supported"], tail["values"],
                                                task.heuristic(bank["endpoints"]))
    np.testing.assert_array_equal(result["local_scores"], expected)
    assert result["baseline_index"] == expected.argmin()
    independent_scores = bank["heuristic_scores"][:16]
    weights = np.exp(-(independent_scores - independent_scores.min()) / 3.)
    weights /= weights.sum()
    np.testing.assert_allclose(bank["plans"][16],
                               np.sum(bank["plans"][:16] * weights[:, None, None], axis=0))


def test_last_window_uses_actual_horizon_and_no_unexecuted_tail():
    task = Task()
    result = Search(task, K=8, H=16).plan(task.reset(0), remaining=3, seed=4)
    assert result["bank_plans"].shape == (9, 3, 1)
    assert result["bank"]["endpoint_remaining"] == 0
    np.testing.assert_array_equal(result["local_scores"], result["bank_stage_costs"])
    assert result["query"]["simulated_controls"] == 27


def test_failed_guidance_preserves_actual_query_cost_in_ledger():
    task = Task()
    ledger = Ledger()
    search = Search(task, K=4, H=3, ledger=ledger)
    with pytest.raises(ValueError, match="guidance scores") as failure:
        search.plan(task.reset(1), 10, seed=1, guidance=lambda bank: np.array([np.nan]))
    assert failure.value.query["simulated_controls"] == 15
    assert failure.value.query["failed_queries"] == 1
    assert ledger.totals()["simulated_controls"] == 15
    assert ledger.totals()["planner_calls"] == 1
    assert ledger.totals()["failed_queries"] == 1
