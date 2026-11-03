"""Synthetic optimizer and reconstruction contracts; no task physics is run."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import trajectory_pi_optimizer as module
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/prototype_sources/full_pi_2026_11_03.py"


class FakeEnv:
    action_dim = 6
    action_bounds = (np.full(6, -1.0), np.ones(6))
    _dt = 0.008

    def __init__(self):
        self.calls = []
        self.observation_rows = []

    @staticmethod
    def initial():
        state = np.zeros(19, dtype=np.float64)
        state[2] = 1.25
        state[10] = 1.5
        return state

    @staticmethod
    def _is_healthy(z, pitch):
        return (z > 0.8) & (z < 2) & (np.abs(pitch) < 1)

    def rollout_states_to_obs(self, states):
        states = np.asarray(states)
        assert np.isfinite(states).all(), "Post-fall states reached the adapter"
        self.observation_rows.append(states.copy())
        return states[..., 2:19]

    def batch_rollout(self, start, actions, initial_warmstart=None):
        self.calls.append(
            dict(start=start.copy(), actions=actions.copy(), warm=initial_warmstart.copy())
        )
        k, h, _ = actions.shape
        state = np.broadcast_to(start, (k, len(start))).copy()
        states = []
        for step in range(h):
            state[:, 0] += self._dt
            state[:, 1] += 0.01 * actions[:, step, 0]
            state[:, 10] = 1.5 + 0.1 * actions[:, step, 0]
            states.append(state.copy())
        post = np.stack(states, axis=1)
        costs = np.sum(actions**2, axis=(1, 2))
        return post, costs, np.zeros((k, h, 1))

    def step(self, _action):
        raise AssertionError("Full sequence optimizer executed a physical step")


class ConstantActor:
    def __init__(self, value=0.25):
        self.value, self.rows, self.observations = value, 0, []

    def predict(self, observation):
        assert observation.dtype == np.float32
        self.rows += len(observation)
        self.observations.append(observation.copy())
        return np.full((len(observation), 6), self.value, dtype=np.float32)


def config(**changes):
    return MPPIConfig(
        **(
            dict(
                K=16,
                H=6,
                lam=0.3,
                noise_sigma=0.3,
                initial_iterations=7,
                lam_target_neff_fraction=0.5,
            )
            | changes
        )
    )


def archived_optimizer():
    tree = ast.parse(ARCHIVE.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("FullPI", "optimize")
    ]
    namespace = dict(MPPI=MPPI, np=np, FINAL_SCALE=0.1)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ARCHIVE), "exec"), namespace)
    return namespace


def test_promoted_fullpi_class_ast_is_identical_to_archived_prototype():
    def node(path):
        return next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "FullPI"
        )

    assert ast.dump(node(ARCHIVE)) == ast.dump(node(Path(module.__file__)))


def test_zero_incumbent_noise_consumes_same_private_random_stream():
    env, cfg = FakeEnv(), config()
    plain, promoted = MPPI(env, cfg), module.FullPI(env, cfg)
    plain.reset(seed=17)
    promoted.reset(seed=17)
    promoted.scale = 0.37
    expected = plain._sample_noise() * promoted.scale
    expected[0] = 0
    np.testing.assert_array_equal(promoted._sample_noise(), expected)
    assert plain.snapshot()["rng_state"] == promoted.snapshot()["rng_state"]


@pytest.mark.parametrize("shaped", [False, True])
def test_alpha_zero_driver_exactly_reuses_archived_loop_and_rng(monkeypatch, shaped):
    cfg = config()
    if shaped:
        cfg.noise_lowpass_cutoff_hz = 23.4375
        cfg.noise_lowpass_sample_rate_hz = 125.0
        cfg.noise_lowpass_order = 2
    namespace = archived_optimizer()
    captured = []
    archived_class = namespace["FullPI"]

    class Capture(archived_class):
        def __init__(self, *args):
            super().__init__(*args)
            captured.append(self)

    namespace["FullPI"] = Capture
    monkeypatch.setattr(MPPI, "plan_step", lambda *a, **k: pytest.fail("Horizon shift invoked"))
    start, warm = FakeEnv.initial(), np.arange(9, dtype=float)
    initial = np.linspace(-0.2, 0.2, cfg.H * 6).reshape(cfg.H, 6)
    expected, expected_curve = namespace["optimize"](FakeEnv(), cfg, start, warm, initial, 23, 5)
    events = []
    actual = module.optimize_full(
        FakeEnv(), cfg, start, warm, initial, 23, 5, observer=events.append
    )
    np.testing.assert_array_equal(actual["best_actions"], expected)
    np.testing.assert_array_equal(actual["final_actions"], captured[0].U)
    np.testing.assert_array_equal([r["C"] for r in actual["curve"]], expected_curve)
    assert actual["planner_snapshot"]["rng_state"] == captured[0].snapshot()["rng_state"]
    assert actual["planner_snapshot"]["initial_plan"] is True
    assert actual["planner_seed"] == 20_023
    assert actual["counts"] == dict(
        requested_passes=5,
        completed_passes=5,
        requested_model_controls=5 * cfg.K * cfg.H,
        completed_model_controls=5 * cfg.K * cfg.H,
        unknown_partial_model_controls=0,
    )
    assert len(events) == 10
    assert [e["phase"] for e in events] == ["before", "after"] * 5
    np.testing.assert_array_equal(events[0]["U"], initial)
    assert events[-1]["scale"] == 0.1
    assert events[-1]["C"].shape == events[-1]["J"].shape == (cfg.K,)
    assert events[-1]["D"] is None
    assert actual["best_D"] is None


def test_frozen_prefix_score_uses_pre_action_states_and_float64_control_residuals():
    env, actor = FakeEnv(), ConstantActor(0.1)
    start = env.initial()
    states = np.broadcast_to(start, (2, 3, 19)).copy()
    states[:, :, 2] = [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]]
    actions = np.full((2, 3, 6), 0.2, dtype=np.float64)
    prior = module.FrozenReconstructionCost(env, actor, start, 0.12)
    score = prior(states, actions)
    expected = 0.5 * 18 * (0.2 - float(np.float32(0.1))) ** 2
    np.testing.assert_allclose(prior.last_D, expected, rtol=0, atol=1e-16)
    np.testing.assert_allclose(score, 0.12 * expected, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(
        actor.observations[0][:, 0], np.array([1.25, 1.1, 1.2, 1.25, 1.4, 1.5], np.float32)
    )
    np.testing.assert_array_equal(prior.last_prefix_lengths, [3, 3])
    assert prior.last_complete.all()
    assert prior.counts["completed_actor_rows"] == 6


def test_invalid_candidate_is_excluded_without_aborting_valid_bank():
    env, actor = FakeEnv(), ConstantActor(0)
    start = env.initial()
    states = np.broadcast_to(start, (3, 4, 19)).copy()
    actions = np.zeros((3, 4, 6))
    states[1, 0, 10] = np.nan
    actions[2, 1, 0] = np.nan
    prior = module.FrozenReconstructionCost(env, actor, start, 0.12)
    np.testing.assert_array_equal(prior(states, actions), [0, np.inf, np.inf])
    np.testing.assert_array_equal(prior.last_valid_candidates, [True, False, False])
    assert actor.rows == 4
    assert prior.counts["requested_actor_rows"] == 4
    assert prior.counts["completed_actor_rows"] == 4


def test_all_invalid_reconstruction_has_no_policy_evaluations():
    env, actor = FakeEnv(), ConstantActor()
    states = np.broadcast_to(env.initial(), (2, 3, 19)).copy()
    states[:, 0, 0] = np.nan
    prior = module.FrozenReconstructionCost(env, actor, env.initial(), 0.12)
    assert np.isinf(prior(states, np.zeros((2, 3, 6)))).all()
    assert not prior.last_valid_candidates.any()
    assert actor.rows == 0


def test_score_never_evaluates_or_fabricates_post_fall_states():
    env, actor = FakeEnv(), ConstantActor(0.25)
    start = env.initial()
    states = np.broadcast_to(start, (1, 4, 19)).copy()
    states[0, 1, 2] = 0.5
    states[0, 2:] = np.nan
    actions = np.zeros((1, 4, 6))
    actions[0, 2:] = np.nan
    prior = module.FrozenReconstructionCost(env, actor, start, 0.12)
    np.testing.assert_array_equal(prior(states, actions), [0.12 * 0.375])
    np.testing.assert_array_equal(prior.last_active_mask, [[True, True, False, False]])
    np.testing.assert_array_equal(prior.last_prefix_lengths, [2])
    assert not prior.last_complete[0]
    assert actor.rows == 2
    assert prior.counts["requested_actor_rows"] == prior.counts["completed_actor_rows"] == 2


def test_alpha_zero_does_not_query_actor_or_observation_adapter():
    class NoPrediction:
        def predict(self, _observation):
            pytest.fail("Alpha-zero bank evaluated the policy")

    env, cfg = FakeEnv(), config()
    env.rollout_states_to_obs = lambda *_: pytest.fail("Alpha-zero adapter queried")
    prior = module.FrozenReconstructionCost(env, NoPrediction(), env.initial(), 0.0)
    result = module.optimize_full(
        env, cfg, env.initial(), np.zeros(9), np.zeros((cfg.H, 6)), 7, 3, prior_cost=prior
    )
    assert prior.counts == dict(
        calls=3, requested_actor_rows=0, completed_actor_rows=0, unknown_partial_actor_rows=0
    )
    assert prior.last_D is result["best_D"] is None
    assert result["best_J"] == result["best_cost"]


def test_retention_selects_total_objective_not_lowest_native_cost():
    env, cfg = FakeEnv(), config(K=64)
    actor = ConstantActor(0.5)
    prior = module.FrozenReconstructionCost(env, actor, env.initial(), 5.0)
    result = module.optimize_full(
        env, cfg, env.initial(), np.zeros(9), np.zeros((cfg.H, 6)), 47, 5, prior_cost=prior
    )
    assert result["curve"][0]["C"] == 0
    assert result["best_cost"] > 0  # native-cost-only retention would keep zeros
    assert result["best_J"] < result["curve"][0]["J"]
    assert result["best_J"] == result["best_cost"] + 5 * result["best_D"]
    assert result["best_iteration"] > 0
    assert result["best_states"].shape == (cfg.H, 19)


def test_bank_weights_use_the_saved_total_score_and_temperature():
    env, cfg = FakeEnv(), config()
    prior = module.FrozenReconstructionCost(env, ConstantActor(0.3), env.initial(), 0.12)
    events = []
    result = module.optimize_full(
        env,
        cfg,
        env.initial(),
        np.zeros(9),
        np.zeros((cfg.H, 6)),
        11,
        3,
        prior_cost=prior,
        observer=events.append,
    )
    for event in events:
        if event["phase"] != "after":
            continue
        np.testing.assert_array_equal(event["J"], event["C"] + 0.12 * event["D"])
        weight = np.exp(-(event["J"] - np.min(event["J"])) / event["lambda"])
        weight /= weight.sum()
        np.testing.assert_array_equal(event["weights"], weight)
    assert result["reconstruction_work"]["completed_actor_rows"] == 3 * cfg.K * cfg.H


def test_final_weighted_mean_is_not_silently_selected_without_scoring():
    env, cfg = FakeEnv(), config()
    initial = np.full((cfg.H, 6), 0.4)
    result = module.optimize_full(env, cfg, env.initial(), np.zeros(9), initial, 2, 1)
    np.testing.assert_array_equal(result["best_actions"], initial)
    assert not np.array_equal(result["final_actions"], initial)
    assert result["final_update_scored"] is False
    assert result["best_iteration"] == 0


def test_observer_copies_cannot_mutate_the_search_or_private_rng():
    env, cfg = FakeEnv(), config()
    initial = np.full((cfg.H, 6), 0.3)
    expected = module.optimize_full(env, cfg, env.initial(), np.zeros(9), initial, 4, 4)

    def observe(event):
        if event["phase"] == "before":
            event["U"][:] = 1
            event["planner_snapshot"]["U"][:] = -1
            event["initial_state"][:] = 100
        else:
            event["post_actions"][:] = -1
            event["states"][:] = -123

    actual = module.optimize_full(
        FakeEnv(), cfg, env.initial(), np.zeros(9), initial, 4, 4, observer=observe
    )
    np.testing.assert_array_equal(actual["final_actions"], expected["final_actions"])
    assert actual["planner_snapshot"]["rng_state"] == expected["planner_snapshot"]["rng_state"]


def test_zero_reconstruction_residual_reproduces_deterministic_fake_execution():
    env, actor = FakeEnv(), ConstantActor(0.25)
    start, warm = env.initial(), np.zeros(9)
    actions = np.full((1, 5, 6), 0.25)
    teacher, _, _ = env.batch_rollout(start, actions, initial_warmstart=warm)
    prior = module.FrozenReconstructionCost(env, actor, start, 0.12)
    np.testing.assert_array_equal(prior(teacher, actions), [0.0])
    state, policy_states = start.copy(), []
    for _ in range(5):
        action = actor.predict(env.rollout_states_to_obs(state[None]).astype(np.float32))
        post, _, _ = env.batch_rollout(
            state, action[None].astype(np.float64), initial_warmstart=warm
        )
        state = post[0, 0]
        policy_states.append(state.copy())
    np.testing.assert_array_equal(teacher[0], np.stack(policy_states))


@pytest.mark.parametrize("iterations", [0, -1, True, 2.5])
def test_invalid_refinement_budget_stops_before_any_bank(iterations):
    env, cfg = FakeEnv(), config()
    with pytest.raises(ValueError):
        module.optimize_full(
            env, cfg, env.initial(), np.zeros(9), np.zeros((cfg.H, 6)), 0, iterations
        )
    assert not env.calls


def test_bad_context_and_sequence_shape_are_rejected_before_bank():
    env, cfg = FakeEnv(), config()
    with pytest.raises(ValueError, match="shape"):
        module.optimize_full(env, cfg, env.initial(), np.zeros(9), np.zeros((2, 6)), 0, 2)
    prior = module.FrozenReconstructionCost(env, ConstantActor(), env.initial(), 0.1)
    other = env.initial()
    other[1] += 1
    with pytest.raises(ValueError, match="contexts differ"):
        module.optimize_full(
            env, cfg, other, np.zeros(9), np.zeros((cfg.H, 6)), 0, 2, prior_cost=prior
        )
    assert not env.calls


@pytest.mark.parametrize("bad", ["horizon", "action_dim", "nonfinite_initial"])
def test_malformed_reconstruction_inputs_stop_before_policy(bad):
    env, actor = FakeEnv(), ConstantActor()
    start = env.initial()
    states = np.broadcast_to(start, (2, 3, 19)).copy()
    actions = np.zeros((2, 3, 6))
    if bad == "nonfinite_initial":
        start[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            module.FrozenReconstructionCost(env, actor, start, 0.1)
    else:
        prior = module.FrozenReconstructionCost(env, actor, start, 0.1)
        if bad == "horizon":
            states = states[:, :2]
        else:
            actions = np.zeros((2, 3, 7))
        with pytest.raises(ValueError, match="dimensions"):
            prior(states, actions)
    assert actor.rows == 0


def test_weight_is_read_only_publicly_and_initial_context_is_a_private_copy():
    env, start = FakeEnv(), FakeEnv.initial()
    prior = module.FrozenReconstructionCost(env, ConstantActor(), start, 0.1)
    start[:] = 99
    assert prior.initial_state[2] == 1.25
    assert not prior.initial_state.flags.writeable
    with pytest.raises(AttributeError):
        prior.alpha = 0.2


def test_actor_errors_are_not_clipped_and_preserve_unknown_forward_rows():
    env = FakeEnv()
    prior = module.FrozenReconstructionCost(env, ConstantActor(2.0), env.initial(), 0.1)
    states = np.broadcast_to(env.initial(), (2, 3, 19)).copy()
    with pytest.raises(ValueError, match="do not clip"):
        prior(states, np.zeros((2, 3, 6)))
    assert prior.last_D is None
    assert prior.counts["requested_actor_rows"] == prior.counts["unknown_partial_actor_rows"] == 6


def test_failed_native_call_retains_requested_geometry_without_claiming_completion():
    env, cfg = FakeEnv(), config()
    env.batch_rollout = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("native partial"))
    with pytest.raises(RuntimeError, match="native partial") as info:
        module.optimize_full(env, cfg, env.initial(), np.zeros(9), np.zeros((cfg.H, 6)), 9, 2)
    assert info.value.trajectory_pi_work == dict(
        requested_passes=1,
        completed_passes=0,
        requested_model_controls=cfg.K * cfg.H,
        completed_model_controls=0,
        unknown_partial_model_controls=cfg.K * cfg.H,
    )
