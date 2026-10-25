"""Exact Walker objective/termination parity independent of CUDA physics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.envs.walker2d import Walker2d
from src.gps.gpu_estep import WalkerGPUEStep, walker_rollout_cost
from src.utils.config import MPPIConfig


@pytest.mark.parametrize("cost_style", ["target_velocity", "gymnasium"])
@pytest.mark.parametrize("terminal", [True, False])
def test_device_cost_matches_numpy_for_falls_recovery_and_corrupt_suffix(
    cost_style: str, terminal: bool,
) -> None:
    env = Walker2d(nthread=1, cost_style=cost_style, apply_terminal_cost_on_done=terminal,
                   vertical_velocity_cost_weight=0.5)
    try:
        rng = np.random.default_rng(421)
        states = np.tile(env.get_state(), (7, 6, 1))
        states[..., 10:] = rng.normal(size=(7, 6, 9))
        actions = rng.uniform(-1, 1, (7, 6, 6))
        states[0, 0, 2] = 0.5  # Fall on first control; later recovery stays absorbed.
        states[1, 2:4, 3] = 1.1  # Fall in the middle, then recover.
        states[2, -1, 2] = 0.5  # Fall on the final control, no absorbing suffix.
        states[3, 1, 2] = 0.5
        states[3, 2:] = np.nan  # Corrupt unused suffix is ignored.
        actions[3, 2:] = np.nan
        states[4, 0, -1] = np.nan  # Corrupt active prefix must be rejected.
        actions[5, 1, 0] = np.nan
        expected = env.rollout_cost(states, actions, np.empty((7, 6, 0)))
        actual = walker_rollout_cost(env, torch.from_numpy(states[..., 1:10]),
                                    torch.from_numpy(states[..., 10:]), torch.from_numpy(actions))
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        assert torch.isinf(actual[4:6]).all()
    finally:
        env.close()


def test_device_tail_orchestration_absorbs_horizon_falls_and_scores_tail_falls() -> None:
    env = Walker2d(nthread=1, cost_style="target_velocity")
    try:
        horizon, tail_steps, batch = 3, 4, 4
        initial = env.get_state()
        horizon_states = np.tile(initial, (batch, horizon, 1))
        tail_states = np.tile(initial, (batch, tail_steps, 1))
        horizon_states[0, 1, 2] = 0.5
        tail_states[0] = np.nan  # A horizon fall must never use its tail dynamics.
        tail_states[1, 1, 2] = 0.5
        tail_states[2, -1, 2] = 0.5
        actions = np.zeros((batch, horizon, 6))
        tail_actions = np.zeros((batch, tail_steps, 6))

        class ScriptedWorlds:
            ctrl = torch.zeros(batch, 6, dtype=torch.float64)
            calls = 0

            def rollout(self, steps: int, n: int, actions: torch.Tensor | None = None,
                        policy: object = None, reset_warmstart_each_step: bool = False,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                self.calls += 1
                if self.calls == 1:
                    assert steps == horizon and not reset_warmstart_each_step
                    states, controls = horizon_states, np.zeros((batch, horizon, 6))
                else:
                    assert steps == tail_steps and reset_warmstart_each_step
                    states, controls = tail_states, tail_actions
                return (torch.from_numpy(states[..., 1:10]), torch.from_numpy(states[..., 10:]),
                        torch.from_numpy(controls))

        planner = WalkerGPUEStep.__new__(WalkerGPUEStep)
        planner.env, planner.cfg, planner.tail_steps = env, SimpleNamespace(H=horizon), tail_steps
        expected_tail = env.rollout_cost(tail_states, tail_actions, np.empty((batch, tail_steps, 0)))
        expected_tail[0] = env._w_unhealthy * tail_steps
        expected = env.rollout_cost(horizon_states, actions, np.empty((batch, horizon, 0))) + expected_tail
        actual = planner._cost_tensor(ScriptedWorlds(), batch, torch.from_numpy(actions))
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
    finally:
        env.close()


def _unallocated_estep() -> WalkerGPUEStep:
    """Input validation and empty batches must work without creating CUDA worlds."""
    planner = WalkerGPUEStep.__new__(WalkerGPUEStep)
    planner.env = SimpleNamespace(model=SimpleNamespace(nq=9, nv=9), _nstate=19, action_dim=6)
    planner.cfg = MPPIConfig(H=3, K=8, lam_target_neff_fraction=0.2)
    return planner


def test_empty_batch_has_correct_output_shapes_without_cuda() -> None:
    planner = _unallocated_estep()
    result = planner(np.empty((0, 19)), np.empty((0, 9)))
    assert result.labels.shape == (0, 6)
    for name in ("lam", "n_eff", "sample_mean_cost", "expected_cost", "nominal_cost", "plan_cost"):
        assert getattr(result, name).shape == (0,)
    assert planner.policy_cost(np.empty((0, 19)), np.empty((0, 9))).shape == (0,)


def test_input_shapes_nonfinite_states_and_seed_lengths_are_rejected() -> None:
    planner = _unallocated_estep()
    states, warmstarts = np.zeros((1, 19)), np.zeros((1, 9))
    with pytest.raises(ValueError, match="states must have shape"):
        planner(states[:, :18], warmstarts)
    with pytest.raises(ValueError, match="warmstarts must have shape"):
        planner(states, warmstarts[:, :8])
    with pytest.raises(ValueError, match="must be finite"):
        planner(states + np.nan, warmstarts)
    with pytest.raises(ValueError, match="not both"):
        planner(states, warmstarts, noise_seeds=[1], noise_streams=[(1, 0)])
    for seeds in ([], [-1], [True]):
        with pytest.raises(ValueError, match="noise_seeds"):
            planner(states, warmstarts, noise_seeds=seeds)
    for streams in ([], [(1, -1)], [(1, 0, 1)], [None]):
        with pytest.raises(ValueError, match="noise_streams"):
            planner(states, warmstarts, noise_streams=streams)


def test_policy_probes_recompute_current_policy_and_preserve_chunk_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = Walker2d(nthread=1, cost_style="target_velocity")
    try:
        original_as_tensor = torch.as_tensor

        def cpu_tensor(value: object, **kwargs: object) -> torch.Tensor:
            return original_as_tensor(value, **{**kwargs, "device": "cpu"})

        monkeypatch.setattr(torch, "as_tensor", cpu_tensor)

        class StationaryWorlds:
            ctrl = torch.zeros(4, 6, dtype=torch.float32)

            def load(self, qpos: torch.Tensor, qvel: torch.Tensor, warmstart: torch.Tensor) -> None:
                self.qpos, self.qvel = qpos, qvel

            def rollout(self, steps: int, n: int, actions: torch.Tensor | None = None,
                        policy: object = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if actions is None:
                    obs = torch.cat([self.qpos[:n, 1:], self.qvel[:n]], dim=1)
                    actions = policy(obs)[:, None].repeat(1, steps, 1)
                return (self.qpos[:n, None].repeat(1, steps, 1),
                        self.qvel[:n, None].repeat(1, steps, 1), actions)

        planner = WalkerGPUEStep.__new__(WalkerGPUEStep)
        planner.env, planner.cfg, planner.tail_steps = env, MPPIConfig(H=3), 0
        planner.singles, planner.batch = StationaryWorlds(), 2
        planner.policy = lambda obs: (0.1 * obs[:, :1]).repeat(1, 6)
        states = np.tile(env.get_state(), (5, 1))
        states[:, 2] = [1.0, 1.125, 1.25, 1.375, 1.5]
        warmstarts = np.zeros((5, 9))
        before = planner.policy_cost(states, warmstarts)
        order = np.array([3, 0, 4, 1, 2])
        np.testing.assert_array_equal(planner.policy_cost(states[order], warmstarts[order]), before[order])
        planner.policy = lambda obs: (0.2 * obs[:, :1]).repeat(1, 6)
        after = planner.policy_cost(states, warmstarts)
        assert (after > before).all()
    finally:
        env.close()


@pytest.mark.parametrize("invalid_source", ["nominal", "tail"])
def test_policy_probe_validity_retains_absorbed_action_failures_across_all_chunks(
    monkeypatch: pytest.MonkeyPatch, invalid_source: str,
) -> None:
    env = Walker2d(nthread=1, cost_style="target_velocity")
    try:
        original_as_tensor = torch.as_tensor

        def cpu_tensor(value: object, **kwargs: object) -> torch.Tensor:
            return original_as_tensor(value, **{**kwargs, "device": "cpu"})

        monkeypatch.setattr(torch, "as_tensor", cpu_tensor)

        class AbsorbingWorlds:
            ctrl = torch.zeros(1, 6, dtype=torch.float32)

            def load(self, qpos: torch.Tensor, qvel: torch.Tensor, warmstart: torch.Tensor) -> None:
                self.qpos, self.qvel = qpos, qvel

            def rollout(
                self, steps: int, n: int, actions: torch.Tensor | None = None,
                policy: object = None, reset_warmstart_each_step: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                positions = self.qpos[:n, None].repeat(1, steps, 1)
                positions[..., 1] = 0.5  # Fall immediately; later controls are unused.
                velocities = self.qvel[:n, None].repeat(1, steps, 1)
                if actions is None:
                    actions = torch.zeros(n, steps, 6)
                    corrupt = self.qpos[:n, 0] < 0
                    if invalid_source == "tail" and reset_warmstart_each_step:
                        actions[corrupt] = torch.nan
                    elif invalid_source == "nominal" and not reset_warmstart_each_step:
                        actions[corrupt, 1:] = torch.nan
                return positions, velocities, actions

        planner = WalkerGPUEStep.__new__(WalkerGPUEStep)
        planner.env, planner.cfg, planner.tail_steps = env, MPPIConfig(H=2), 3
        planner.singles, planner.batch = AbsorbingWorlds(), 1
        states = np.tile(env.get_state(), (2, 1))
        states[:, 1] = [-1.0, 1.0]  # First chunk is corrupt, final chunk is clean.
        warmstarts = np.zeros((2, 9))
        costs = planner.policy_cost(states, warmstarts)
        assert np.isfinite(costs).all()
        assert planner._last_cost_actions_valid.item() is True
        assert planner.last_policy_cost_numerical_valid is False
        assert np.isfinite(planner.policy_cost(states[1:], warmstarts[1:])).all()
        assert planner.last_policy_cost_numerical_valid is True
        planner.policy_cost(states[:1], warmstarts[:1])
        assert planner.last_policy_cost_numerical_valid is False
        assert planner.policy_cost(states[:0], warmstarts[:0]).shape == (0,)
        assert planner.last_policy_cost_numerical_valid is True
    finally:
        env.close()


@pytest.mark.parametrize("kwargs, match", [
    ({"tail_steps": -1}, "tail_steps"),
    ({"tail_steps": 0, "states_per_batch": 0}, "states_per_batch"),
])
def test_invalid_gpu_allocation_config_is_rejected_before_allocating(
    kwargs: dict[str, int], match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        WalkerGPUEStep(None, None, MPPIConfig(lam_target_neff_fraction=0.2), **kwargs)
