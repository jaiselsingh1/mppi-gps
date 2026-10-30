"""Research summaries reuse the existing sampled and verification rollouts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.gps.gpu_estep import GPULabels, WalkerGPUEStep, tail_variance_ratio
from src.gps.mdgps import EStepResult, PolicyTail, _raw_weights_finite, _temperature_saturated, e_step
from src.utils.math import temperature_for_neff_fraction


class _Environment:
    action_bounds = np.array([-1.0]), np.array([1.0])

    def __init__(self) -> None:
        self.calls: int = 0

    def batch_rollout(
        self, state: np.ndarray, actions: np.ndarray, **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.calls += 1
        batch, horizon, _ = actions.shape
        return (np.zeros((batch, horizon, 1)), (actions ** 2).sum(axis=(1, 2)),
                np.zeros((batch, horizon, 0)))


class _Tail:
    def __init__(self, invalid_sample_actions: bool = False) -> None:
        self.last_tail: np.ndarray = np.empty(0)
        self.last_actions_finite: bool = True
        self.calls: int = 0
        self.invalid_sample_actions: bool = invalid_sample_actions

    def __call__(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.last_tail = np.array([4.0, 1.0, 0.0, 7.0]) if len(actions) == 4 else np.array([2.0])
        self.last_actions_finite = not (self.invalid_sample_actions and len(actions) == 4)
        return self.last_tail


class _Planner:
    initial_iterations = 1
    lam_target_neff_fraction = 0.5

    def __init__(self) -> None:
        self.last_plan: np.ndarray = np.full((2, 1), 0.25)
        self.rollout: dict[str, np.ndarray] = {
            "costs": np.array([2.0, 4.0, 8.0, 12.0]),
            "actions": np.arange(8, dtype=float).reshape(4, 2, 1) / 10,
            "weights": np.array([0.4, 0.3, 0.2, 0.1]),
        }

    def plan_step(
        self, state: np.ndarray, nominal: np.ndarray, prior_cost: PolicyTail | None = None,
        **kwargs: object,
    ) -> tuple[np.ndarray, dict[str, float]]:
        tails = prior_cost(np.zeros((4, 2, 1)), self.rollout["actions"]) if prior_cost is not None else 0.0
        lam = temperature_for_neff_fraction(self.rollout["costs"] + tails, 0.5)
        return self.last_plan[0], {"lam": lam, "n_eff": 2.0}

    def get_rollout_data(self) -> dict[str, np.ndarray]:
        return self.rollout


def test_cpu_estep_snapshots_sample_tail_variance_without_more_rollouts() -> None:
    env, planner, tail = _Environment(), _Planner(), _Tail()
    result = e_step(planner, env, np.zeros(1), np.zeros((2, 1)), terminal=tail)
    expected = np.var([4.0, 1.0, 0.0, 7.0]) / (np.var([6.0, 5.0, 8.0, 19.0]) + 1e-6)
    assert result.tail_var_ratio == pytest.approx(expected)
    assert result.lambda_saturated is False and result.numerical_valid is True
    assert env.calls == 2 and tail.calls == 3
    np.testing.assert_array_equal(tail.last_tail, [2.0])  # Verification has overwritten it.


@pytest.mark.parametrize("invalid", ["weights", "actions", "tail"])
def test_cpu_estep_preserves_numerical_failure_status_before_discarding_data(invalid: str) -> None:
    env, planner, tail = _Environment(), _Planner(), _Tail(invalid_sample_actions=invalid == "tail")
    if invalid != "tail":
        planner.rollout[invalid].flat[0] = np.nan
    result = e_step(planner, env, np.zeros(1), np.zeros((2, 1)), terminal=tail)
    assert result.numerical_valid is False
    assert env.calls == 2 and tail.calls == 3


def test_scalar_saturation_checks_the_actual_bracket_and_degenerate_limits() -> None:
    scores = np.array([0.0, 1e-12, 1e8])
    lam = temperature_for_neff_fraction(scores, 0.4)
    assert _temperature_saturated(scores, lam, 0.4) is True
    regular = np.array([0.0, 1.0, 2.0, 3.0])
    assert _temperature_saturated(regular, temperature_for_neff_fraction(regular, 0.5), 0.5) is False
    assert _temperature_saturated(np.ones(4), 1.0, 0.5) is True
    assert _temperature_saturated(regular, 0.0, 0.5) is True
    assert _temperature_saturated(regular, np.inf, 0.5) is True
    assert _temperature_saturated(regular, 1.0, None) is None


def test_cpu_pre_fallback_normalization_failures_remain_visible() -> None:
    assert _raw_weights_finite(np.array([0.0, 1.0, np.inf]), 0.5) is True
    assert _raw_weights_finite(np.array([0.0, 1.0]), 0.0) is False
    assert _raw_weights_finite(np.array([0.0, 1.0]), np.nan) is False
    assert _raw_weights_finite(np.array([-1e308, 1e308]), np.inf) is False


def test_device_tail_variance_uses_population_variance_and_finite_support() -> None:
    tails = torch.tensor([[4.0, 1.0, 0.0, 7.0], [500.0] * 4, [1.0, 3.0, torch.inf, torch.nan]],
                         dtype=torch.float64)
    totals = torch.tensor([[6.0, 5.0, 8.0, 19.0], [1.0, 2.0, 3.0, 4.0], [2.0, 8.0, torch.inf, torch.nan]],
                          dtype=torch.float64)
    result = tail_variance_ratio(tails, totals)
    np.testing.assert_allclose(result, [np.var(tails[0].numpy()) / (np.var(totals[0].numpy()) + 1e-6),
                                       0.0, 1.0 / (9.0 + 1e-6)])


def test_gpu_cost_retains_action_failures_even_when_absorbing_tail_overrides_them() -> None:
    class Worlds:
        ctrl = torch.zeros((1, 1), dtype=torch.float64)

        def rollout(self, steps: int, n: int, actions: torch.Tensor | None = None,
                    policy: object = None, reset_warmstart_each_step: bool = False,
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            qpos = torch.zeros((n, steps, 9), dtype=torch.float64)  # Already fallen.
            qvel = torch.zeros((n, steps, 9), dtype=torch.float64)
            controls = actions if actions is not None else torch.full((n, steps, 1), torch.nan)
            return qpos, qvel, controls

    planner = WalkerGPUEStep.__new__(WalkerGPUEStep)
    planner.env = SimpleNamespace(
        _is_healthy=lambda z, angle: (z > 0.8) & (z < 2.0) & (angle > -1) & (angle < 1),
        _w_ctrl=0.001, _cost_style="gymnasium", _w_forward=1.0, _healthy_reward=1.0,
        _w_unhealthy=5.0, _w_term=20.0, _apply_terminal_cost_on_done=True,
    )
    planner.cfg, planner.tail_steps = SimpleNamespace(H=2), 3
    total, tail = planner._cost_components_tensor(Worlds(), 1, torch.zeros((1, 2, 1)))
    assert torch.isfinite(total).all() and tail.tolist() == [15.0]
    assert planner._last_cost_actions_valid.item() is False


def test_result_telemetry_defaults_preserve_old_constructors() -> None:
    cpu = EStepResult(np.zeros(1), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0)
    gpu = GPULabels(np.zeros((1, 1)), *[np.zeros(1) for _ in range(6)])
    assert cpu.tail_var_ratio is None and cpu.lambda_saturated is None and cpu.numerical_valid is True
    assert gpu.tail_var_ratio is None and gpu.lambda_saturated is None and gpu.numerical_valid is True
