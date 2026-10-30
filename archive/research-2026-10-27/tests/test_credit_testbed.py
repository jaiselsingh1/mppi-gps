"""Ground-truth tasks, exact values, recorded-randomness branching and estimators."""
import gymnasium as gym
import numpy as np
import pytest

from src.credit import estimators as est
from src.credit.control import Controller, TerminalValueEnv, branch, most_distinct, replay_decision, run_episode
from src.credit.exact import solve_car, solve_fork
from src.envs.delayed_fork import DelayedFork
from src.envs.mountain_car import MountainCar
from src.mppi import mppi as repo_mppi
from src.utils.config import MPPIConfig

CFG = MPPIConfig(K=16, H=8, lam=0.3, noise_sigma=1.0)


def _fork(delay=12, noise_std=0.0):
    return DelayedFork(gate=10, delay=delay, noise_std=noise_std)


def test_fork_batch_rollout_matches_step():
    env = _fork()
    rng = np.random.default_rng(0)
    U = rng.uniform(-1.5, 1.5, (5, env.episode_steps + 3, 1))
    for k in range(5):
        start = np.array([0.0, env.y_value(rng.integers(env.n_y))])
        states, costs, _ = env.batch_rollout(start, U[k : k + 1])
        env.reset(start)
        total = 0.0
        for h in range(U.shape[1]):
            s, c, _, _ = env.step(U[k, h])
            np.testing.assert_array_equal(s, states[0, h])
            total += c
        assert total == pytest.approx(costs[0], abs=1e-12)


def test_fork_lane_locks_at_gate_and_penalty_is_delayed():
    env = _fork(delay=12)
    states, _, _ = env.batch_rollout(np.array([0.0, 0.5]), np.full((1, env.episode_steps, 1), -1.0))
    assert np.all(states[0, env.gate - 1 :, 1] == states[0, env.gate - 1, 1])
    lane_a = env.batch_rollout(np.array([0.0, 0.5]), np.zeros((1, env.episode_steps, 1)))
    running = env.running_cost(lane_a[0], np.zeros((1, env.episode_steps, 1)))[0]
    assert running[env.gate + env.delay] == pytest.approx(env.penalty)
    assert np.all(running[env.gate : env.gate + env.delay] == 0.0)


def test_fork_exact_value_is_attained_and_never_beaten():
    env = _fork(delay=15)
    V = solve_fork(env)
    rng = np.random.default_rng(1)
    for _ in range(20):
        start = np.array([float(rng.integers(0, env.gate + 3)), env.y_value(rng.integers(env.n_y))])
        s = env.reset(start)
        total, done = 0.0, start[0] >= env.episode_steps
        while not done:
            s, c, done, _ = env.step(np.array([V.action(s)]))
            total += c
        assert total == pytest.approx(V(start), abs=1e-9)
        steps = env.episode_steps - int(start[0])
        U = rng.uniform(-1, 1, (256, steps, 1))
        _, costs, _ = env.batch_rollout(start, U)
        assert np.all(costs >= V(start) - 1e-9)


def test_mountain_car_matches_gymnasium_dynamics():
    gym_env = gym.make("MountainCarContinuous-v0").unwrapped
    env = MountainCar(power=0.0015)
    rng = np.random.default_rng(2)
    gym_env.reset(seed=0)
    # Gymnasium keeps its state in float32, so compare one step at a time.
    for _ in range(1000):
        p, v = rng.uniform(-1.2, 0.6), rng.uniform(-0.07, 0.07)
        u = rng.uniform(-1.5, 1.5, 1)
        gym_env.state = np.array([p, v])
        env.reset(np.array([p, v, 0.0, 0.0]))
        obs, _, terminated, _, _ = gym_env.step(u)
        s, _, _, _ = env.step(u)
        np.testing.assert_allclose(s[:2], obs, atol=1e-6)
        assert bool(s[2]) == terminated


def test_car_value_matches_its_greedy_policy():
    env = MountainCar(power=0.0015, max_steps=400)
    V = solve_car(env, n_position=151, n_velocity=151)
    for p in (-0.55, -0.45):
        s = env.reset(np.array([p, 0.0, 0.0, 0.0]))
        total, done = 0.0, False
        while not done:
            s, c, done, _ = env.step(np.array([V.action(s)]))
            total += c
        assert s[2] == 1.0
        assert total == pytest.approx(V(np.array([p, 0.0, 0.0, 0.0])), rel=0.03)


def test_controllers_use_the_repository_mppi_unchanged():
    env = _fork()
    ctrl = Controller(env, CFG)
    assert isinstance(ctrl.planner(0), repo_mppi.MPPI)
    a = ctrl.planner(3)
    b = repo_mppi.MPPI(env, CFG, seed=3)
    state = np.array([0.0, 0.3])
    np.testing.assert_array_equal(a.plan_step(state)[0], b.plan_step(state)[0])


def test_terminal_value_env_adds_value_at_horizon_end_only():
    env = _fork()
    value = lambda s: 100.0 * s[..., 1]
    wrapped = TerminalValueEnv(env, value)
    U = np.random.default_rng(4).uniform(-1, 1, (6, 8, 1))
    states, costs, _ = env.batch_rollout(np.array([0.0, 0.2]), U)
    w_states, w_costs, _ = wrapped.batch_rollout(np.array([0.0, 0.2]), U)
    np.testing.assert_array_equal(states, w_states)
    np.testing.assert_allclose(w_costs, costs + 100.0 * states[:, -1, 1])


def test_selection_keeps_mppi_plan_and_random_stream():
    env = _fork()
    plain, chooser = Controller(env, CFG), Controller(env, CFG, scorer=lambda s, o: -o.model_cost)
    a, b = plain.planner(9), chooser.planner(9)
    state = np.array([0.0, 0.3])
    _, opts, chosen = chooser.decide(b, state, record=True)
    _, opts_plain, _ = plain.decide(a, state, record=True)
    np.testing.assert_array_equal(opts.actions, opts_plain.actions)
    np.testing.assert_array_equal(a.U, b.U)
    assert a.snapshot()["rng_state"] == b.snapshot()["rng_state"]
    assert chosen == int(np.argmax(opts.model_cost))


def test_branch_of_executed_option_reproduces_recorded_suffix():
    env = _fork(delay=12, noise_std=2.0)
    ctrl = Controller(env, CFG)
    noise = env.sample_noise(np.random.default_rng(5))
    ep = run_episode(env, ctrl, np.array([0.0, 0.3]), 11, noise, record="snapshots")
    for t in (0, 4, 9, 15):
        br = branch(env, ctrl, ep, t, lambda o, c: o.mean_index, commit=1)
        np.testing.assert_array_equal(br.trajectory.states, ep.trajectory.states[t:])
        np.testing.assert_array_equal(br.trajectory.costs, ep.trajectory.costs[t:])
    br = branch(env, ctrl, ep, 3, most_distinct(np.array([1.0, 1.0])))
    np.testing.assert_array_equal(br.trajectory.actions[: CFG.H], br.options.actions[br.option])
    np.testing.assert_array_equal(br.trajectory.states[1 : CFG.H + 1], br.options.states[br.option])
    # Shared randomness: the same exogenous noise is charged on both paths.
    assert br.trajectory.costs.sum() - ep.trajectory.costs[3:].sum() == pytest.approx(
        env.running_cost(br.trajectory.states[1:][None], br.trajectory.actions[None]).sum()
        - env.running_cost(ep.trajectory.states[4:][None], ep.trajectory.actions[3:][None]).sum()
    )


def test_replay_detects_a_changed_controller():
    env = _fork()
    ep = run_episode(env, Controller(env, CFG), np.array([0.0, 0.3]), 12, np.zeros(env.episode_steps), record="snapshots")
    other = Controller(env, MPPIConfig(K=16, H=8, lam=0.31, noise_sigma=1.0))
    with pytest.raises(RuntimeError):
        for t in range(len(ep.trajectory.actions)):
            replay_decision(env, other, ep, t)


def _chain_view(T):
    coder = est.TileCoder([0.0, -1.0], [float(T), 1.0], [T, 2], 1)
    return est.TaskView(coder, terminal=lambda s: np.asarray(s)[..., 0] >= T, active_step=lambda s: np.asarray(s)[..., 0] <= T)


def _chain_data(T, n, rng, noise):
    data = []
    for _ in range(n):
        lane = rng.integers(2)
        y = 0.5 if lane else -0.5
        states = np.array([[float(t), y] for t in range(T + 1)])
        costs = np.where(np.arange(T) == T - 1, 10.0 * lane, 1.0) + noise * rng.standard_normal(T)
        data.append(est.TrajectoryData(states, costs, False))
    return data


def test_lstd_one_equals_mc_and_lstd_zero_is_exact_on_a_chain():
    T = 6
    rng = np.random.default_rng(6)
    view = _chain_view(T)
    data = _chain_data(T, 40, rng, 0.0)
    probe = np.array([[float(t), y] for t in range(T) for y in (0.5, -0.5)])
    # The last step costs 10 in lane A instead of 1, and 0 in lane B.
    truth = np.array([(T - t - 1) + (10.0 if y > 0 else 0.0) for t in range(T) for y in (0.5, -0.5)])
    mc = est.fit_mc(view, data, ridge=1e-8)(probe)
    td1 = est.fit_lstd(view, data, 1.0, ridge=1e-8)(probe)
    td0 = est.fit_lstd(view, data, 0.0, ridge=1e-8)(probe)
    np.testing.assert_allclose(mc, truth, atol=1e-5)
    np.testing.assert_allclose(td1, truth, atol=1e-5)
    np.testing.assert_allclose(td0, truth, atol=1e-5)


def test_paired_scorer_recovers_value_differences_without_the_common_noise():
    T = 6
    view = _chain_view(T)
    rng = np.random.default_rng(7)
    pairs = []
    for _ in range(30):
        common = 50.0 * rng.standard_normal()
        a, b = np.array([3.0, 0.5]), np.array([3.0, -0.5])
        pairs.append(est.Pair(a, b, 12.0 + common, 3.0 + common))
    v = est.fit_paired(view, pairs, ridge=1e-8)
    diff = v(np.array([3.0, 0.5])) - v(np.array([3.0, -0.5]))
    assert diff == pytest.approx(9.0, abs=1e-5)


def test_return_decomposition_spreads_an_unlocalized_penalty():
    T = 8
    view = _chain_view(T)
    data = _chain_data(T, 60, np.random.default_rng(8), 0.0)
    credit = est.fit_return_decomposition(view, data, ridge=1e-6)
    lane_a = credit(np.array([[float(t), 0.5] for t in range(1, T + 1)]))
    lane_b = credit(np.array([[float(t), -0.5] for t in range(1, T + 1)]))
    # Totals identify only the lane's summed credit, not when it is due: the
    # minimum-norm fit spreads the final-step penalty over the lane's steps.
    diff = lane_a - lane_b
    assert diff.sum() == pytest.approx(10.0, abs=1e-3)
    assert np.all(diff > 0.5)
    assert diff[-1] < 0.3 * diff.sum()
