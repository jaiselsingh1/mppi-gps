"""Batched MPPI-GPS E-step on the GPU (MuJoCo Warp), for Walker.

Computes what `e_step(planner, env, x, nominal, warmstart,
terminal=WalkerPolicyTail(env, policy, T))` computes, for many states at once:
the policy's closed-loop nominal, K noisy open-loop rollouts of H controls,
the policy's closed-loop tail after each, the n_eff-targeted softmin and the
label. Every rollout is a GPU world. Each control step's physics runs as one
captured CUDA graph, with the policy evaluated in torch between launches.
Task costs, policy tails, temperature solves and plan synthesis stay on the
GPU. Walker's scoring semantics are reproduced with float64 arithmetic; the
physics remains float32 and is not bitwise repeatable on GPU.
"""
from collections.abc import Callable
from dataclasses import dataclass

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from jaxtyping import Bool, Float

from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.gps.torch_mppi import (
    softmin_weights,
    synthesize_plan,
    temperature_for_neff_fraction_with_details,
)


class _Worlds:
    """`nworld` copies of the model; one captured graph per control step."""

    def __init__(self, env: Walker2d, nworld: int) -> None:
        self.nworld = nworld
        self.model = mjw.put_model(env.model)
        self.data = mjw.make_data(env.model, nworld=nworld)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.warmstart = wp.to_torch(self.data.qacc_warmstart)
        self.time = wp.to_torch(self.data.time)
        with wp.ScopedCapture() as capture:
            for _ in range(env._frame_skip):
                mjw.step(self.model, self.data)
        self.graph = capture.graph

    def load(self, qpos: Float[torch.Tensor, "batch nq"],
             qvel: Float[torch.Tensor, "batch nv"],
             warmstart: Float[torch.Tensor, "batch nv"]) -> None:
        """Set the first len(qpos) worlds; pad the rest by repetition."""
        index = torch.arange(self.nworld, device=qpos.device) % len(qpos)
        self.qpos.copy_(qpos[index])
        self.qvel.copy_(qvel[index])
        self.warmstart.copy_(warmstart[index])
        self.time.zero_()

    def obs(self, n: int) -> Float[torch.Tensor, "batch 17"]:
        """Walker2d.rollout_states_to_obs for the first n worlds."""
        return torch.cat([self.qpos[:n, 1:], self.qvel[:n].clamp(-10.0, 10.0)], dim=1)

    def rollout(
        self,
        steps: int,
        n: int,
        actions: Float[torch.Tensor, "batch steps action"] | None = None,
        policy: Callable[[torch.Tensor], torch.Tensor] | None = None,
        reset_warmstart_each_step: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Open loop (actions (n, steps, nu)) or closed loop (policy(obs) -> ctrl).

        Returns qpos (n, steps, nq), qvel (n, steps, nv) after each control
        step, and the controls applied (n, steps, nu).
        """
        if steps < 1 or not 0 < n <= len(self.qpos):
            raise ValueError("rollout needs positive steps and a batch within world capacity.")
        if (actions is None) == (policy is None):
            raise ValueError("Provide exactly one of actions or policy.")
        qpos: list[torch.Tensor] = []
        qvel: list[torch.Tensor] = []
        ctrl: list[torch.Tensor] = []
        for t in range(steps):
            if actions is not None:
                u = actions[:, t]
            else:
                assert policy is not None
                u = policy(self.obs(n))
            self.ctrl[:n].copy_(u)
            if reset_warmstart_each_step:
                # WalkerPolicyTail starts a separate CPU rollout at each
                # control boundary; mjSTATE_FULLPHYSICS excludes warmstart.
                self.warmstart.zero_()
            torch.cuda.synchronize()
            wp.capture_launch(self.graph)
            wp.synchronize()
            qpos.append(self.qpos[:n].clone())
            qvel.append(self.qvel[:n].clone())
            ctrl.append(u)
        return torch.stack(qpos, 1), torch.stack(qvel, 1), torch.stack(ctrl, 1)


def walker_rollout_cost(
    env: Walker2d,
    qpos: Float[torch.Tensor, "batch horizon nq"],
    qvel: Float[torch.Tensor, "batch horizon nv"],
    actions: Float[torch.Tensor, "batch horizon action"],
) -> Float[torch.Tensor, "batch"]:
    """Device equivalent of Walker2d.rollout_cost, including absorbing falls.

    All arithmetic uses float64, matching NumPy scoring of the same physical
    trajectories. Score through the first fall, add its configured terminal
    charge once, and charge only the absorbing penalty for the unused suffix.
    Corrupt states/actions on the executable prefix invalidate the rollout;
    a corrupt suffix after termination is ignored.
    """
    if qpos.ndim != 3 or qvel.ndim != 3 or actions.ndim != 3 or qpos.shape[1] < 1:
        raise ValueError("rollout tensors must have shape (batch, positive horizon, feature).")
    if qpos.shape[:2] != qvel.shape[:2] or qpos.shape[:2] != actions.shape[:2]:
        raise ValueError("rollout tensors must share their batch and horizon dimensions.")
    qpos, qvel, actions = qpos.double(), qvel.double(), actions.double()
    z, angle, vx = qpos[..., 1], qpos[..., 2], qvel[..., 0]
    healthy = env._is_healthy(z, angle)
    active = torch.ones_like(healthy)
    active[:, 1:] = healthy[:, :-1].to(torch.int64).cumprod(dim=1).bool()
    ctrl_cost = env._w_ctrl * actions.square().sum(dim=-1)
    if env._cost_style == "gymnasium":
        running = -env._w_forward * vx - env._healthy_reward * healthy.double() + ctrl_cost
    else:
        running = (env._w_vel * (env._v_target - vx).abs()
                   + env._w_vertical_velocity * qvel[..., 1].square()
                   + env._w_angle * angle.square()
                   + env._w_unhealthy * (~healthy).double() + ctrl_cost)
    cost = torch.where(active, running, 0.0).sum(dim=1)
    cost += env._w_unhealthy * (~active).sum(dim=1).double()
    if env._apply_terminal_cost_on_done:
        cost += env._w_term * (~healthy).any(dim=1).double()
    valid = (torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
             & torch.isfinite(actions).all(dim=-1))
    return torch.where((valid | ~active).all(dim=1), cost, torch.inf)


@dataclass
class GPULabels:
    """Per-state actions (M,A) and scalar diagnostics (M,) copied to the CPU."""

    labels: Float[np.ndarray, "states action"]
    lam: Float[np.ndarray, "states"]
    n_eff: Float[np.ndarray, "states"]
    sample_mean_cost: Float[np.ndarray, "states"]
    expected_cost: Float[np.ndarray, "states"]
    nominal_cost: Float[np.ndarray, "states"]
    plan_cost: Float[np.ndarray, "states"]
    tail_var_ratio: Float[np.ndarray, "states"] | None = None
    lambda_saturated: Bool[np.ndarray, "states"] | None = None
    numerical_valid: bool = True


class WalkerGPUEStep:
    def __init__(
        self,
        env: Walker2d,
        actor: torch.nn.Module,
        cfg: MPPIConfig,
        tail_steps: int,
        states_per_batch: int = 64,
    ) -> None:
        if cfg.lam_target_neff_fraction is None:
            raise ValueError("The E-step needs an n_eff-targeted temperature.")
        if cfg.initial_iterations != 1:
            raise ValueError("The E-step takes exactly one MPPI update per state.")
        if type(tail_steps) is not int or tail_steps < 0:
            raise ValueError("tail_steps must be a nonnegative integer.")
        if type(states_per_batch) is not int or states_per_batch < 1:
            raise ValueError("states_per_batch must be a positive integer.")
        # Validate the existing noise/config contract before allocating worlds.
        MPPI(env, cfg)
        self.env, self.actor, self.cfg = env, actor, cfg
        self.tail_steps, self.batch = tail_steps, states_per_batch
        self.samples = _Worlds(env, states_per_batch * cfg.K)
        self.singles = _Worlds(env, 2 * states_per_batch)
        low, high = env.action_bounds
        self.low_np, self.high_np = low, high
        self.low = torch.as_tensor(low, dtype=torch.float32, device="cuda")
        self.high = torch.as_tensor(high, dtype=torch.float32, device="cuda")
        self.last_policy_cost_numerical_valid: bool = True

    @torch.no_grad()
    def policy(self, obs: Float[torch.Tensor, "batch 17"]) -> Float[torch.Tensor, "batch 6"]:
        mean, _ = self.actor(obs)
        action = torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
        return torch.clamp(action, self.low, self.high)

    def _cost_tensor(
        self,
        worlds: _Worlds,
        n: int,
        actions: Float[torch.Tensor, "batch horizon action"],
    ) -> Float[torch.Tensor, "batch"]:
        """Pure H-step task cost plus frozen policy tail, all on the device."""
        total, _ = self._cost_components_tensor(worlds, n, actions)
        return total

    def _cost_components_tensor(
        self,
        worlds: _Worlds,
        n: int,
        actions: Float[torch.Tensor, "batch horizon action"],
    ) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
        """Score one existing rollout and retain its tail and action validity.

        ``_last_cost_actions_valid`` records all applied horizon/tail actions,
        including data discarded after a fall, before those tensors expire.
        """
        env = self.env
        u = actions.to(dtype=worlds.ctrl.dtype)
        hq, hv, _ = worlds.rollout(self.cfg.H, n, actions=u)
        cost = walker_rollout_cost(env, hq, hv, actions)
        self._last_cost_actions_valid = torch.isfinite(actions).all() & torch.isfinite(u).all()
        if not self.tail_steps:
            return cost, torch.zeros_like(cost)
        fell = ~env._is_healthy(hq[..., 1], hq[..., 2]).all(dim=1)
        tq, tv, tu = worlds.rollout(self.tail_steps, n, policy=self.policy,
                                    reset_warmstart_each_step=True)
        tail = walker_rollout_cost(env, tq, tv, tu)
        self._last_cost_actions_valid &= torch.isfinite(tu).all()
        tail = torch.where(fell, env._w_unhealthy * self.tail_steps, tail)
        return cost + tail, tail

    def _cost(self, worlds: _Worlds, n: int,
              actions: Float[np.ndarray, "batch horizon action"]) -> Float[np.ndarray, "batch"]:
        """NumPy compatibility wrapper for cost-orchestration diagnostics."""
        tensor = torch.as_tensor(actions, dtype=torch.float64, device="cuda")
        return self._cost_tensor(worlds, n, tensor).cpu().numpy()

    @torch.no_grad()
    def policy_cost(
        self,
        states: Float[np.ndarray, "states nstate"],
        warmstarts: Float[np.ndarray, "states nv"],
    ) -> Float[np.ndarray, "states"]:
        """Score this actor's independent closed-loop H-step plans plus tails.

        Uses the same GPU physics and cost backend as E-step nominal costs,
        permitting before/after policy probes without CPU/GPU backend drift.
        This method draws no MPPI noise and does not reuse previous plans.
        ``last_policy_cost_numerical_valid`` reports action finiteness across
        every chunk, including actions in unused absorbing suffixes.
        """
        env = self.env
        _validate_initial_states(env, states, warmstarts)
        self.last_policy_cost_numerical_valid = True
        if not len(states):
            return np.empty(0)
        costs: list[np.ndarray] = []
        numerical_checks: list[torch.Tensor] = []
        nq, nv = env.model.nq, env.model.nv
        for start in range(0, len(states), self.batch):
            stop = min(start + self.batch, len(states))
            n = stop - start
            q0 = torch.as_tensor(states[start:stop, 1:1 + nq], dtype=torch.float32, device="cuda")
            v0 = torch.as_tensor(states[start:stop, 1 + nq:1 + nq + nv], dtype=torch.float32, device="cuda")
            w0 = torch.as_tensor(warmstarts[start:stop], dtype=torch.float32, device="cuda")
            self.singles.load(q0, v0, w0)
            _, _, nominal = self.singles.rollout(self.cfg.H, n, policy=self.policy)
            numerical_checks.append(torch.isfinite(nominal).all())
            self.singles.load(q0, v0, w0)
            cost = self._cost_tensor(self.singles, n, nominal.double())
            numerical_checks.append(self._last_cost_actions_valid)
            costs.append(cost.cpu().numpy())
        self.last_policy_cost_numerical_valid = bool(torch.stack(numerical_checks).all().item())
        return np.concatenate(costs)

    @torch.no_grad()
    def __call__(
        self,
        states: Float[np.ndarray, "states nstate"],
        warmstarts: Float[np.ndarray, "states nv"],
        noise_seeds: list[int] | None = None,
        noise_streams: list[tuple[int, int]] | None = None,
    ) -> GPULabels:
        """Label `states`. Noise for state i is the first draw of an MPPI
        planner seeded noise_seeds[i], or, with noise_streams[i] = (seed, j),
        the j-th draw of one planner per seed: the stream the CPU training path
        uses for the j-th labelled state of an episode."""
        env, cfg = self.env, self.cfg
        sample_count, horizon = cfg.K, cfg.H
        nq, nv = env.model.nq, env.model.nv
        _validate_initial_states(env, states, warmstarts)
        if noise_seeds is not None and noise_streams is not None:
            raise ValueError("Provide noise_seeds or noise_streams, not both.")
        if noise_seeds is not None:
            if len(noise_seeds) != len(states) or any(not _nonnegative_int(s) for s in noise_seeds):
                raise ValueError("noise_seeds needs one nonnegative integer per state.")
        if noise_streams is not None:
            if len(noise_streams) != len(states) or any(
                not isinstance(stream, (tuple, list)) or len(stream) != 2
                or not all(_nonnegative_int(value) for value in stream)
                for stream in noise_streams
            ):
                raise ValueError("noise_streams needs one (nonnegative seed, draw) pair per state.")
        if not len(states):
            return GPULabels(labels=np.empty((0, env.action_dim)), **{
                name: np.empty(0, dtype=bool if name == "lambda_saturated" else float)
                for name in GPULabels.__dataclass_fields__ if name not in {"labels", "numerical_valid"}
            })
        rows: dict[str, list[np.ndarray]] = {
            name: [] for name in GPULabels.__dataclass_fields__ if name != "numerical_valid"
        }
        numerical_checks: list[torch.Tensor] = []
        planners: dict[int, tuple[MPPI, int]] = {}

        def noise(i: int) -> np.ndarray:
            if noise_streams is None:
                seed = None if noise_seeds is None else noise_seeds[i]
                return MPPI(env, cfg, seed=seed)._sample_noise()
            seed, draw = noise_streams[i]
            if seed not in planners:
                planners[seed] = MPPI(env, cfg, seed=seed), 0
            planner, expected_draw = planners[seed]
            if draw != expected_draw:
                raise ValueError(f"noise stream {seed} expected draw {expected_draw}, got {draw}.")
            planners[seed] = planner, expected_draw + 1
            return planner._sample_noise()

        for start in range(0, len(states), self.batch):
            stop = min(start + self.batch, len(states))
            n = stop - start
            def gpu(a: np.ndarray) -> torch.Tensor:
                return torch.as_tensor(a, dtype=torch.float32, device="cuda")

            q0, v0 = gpu(states[start:stop, 1:1 + nq]), gpu(states[start:stop, 1 + nq:1 + nq + nv])
            w0 = gpu(warmstarts[start:stop])
            self.singles.load(q0, v0, w0)
            _, _, nominal = self.singles.rollout(horizon, n, policy=self.policy)
            nominal = nominal.double()
            draws = torch.as_tensor(np.stack([noise(i) for i in range(start, stop)]),
                                    dtype=torch.float64, device=q0.device)
            sampled = torch.maximum(torch.minimum(nominal[:, None] + draws, self.high), self.low)
            self.samples.load(q0.repeat_interleave(sample_count, 0), v0.repeat_interleave(sample_count, 0),
                              w0.repeat_interleave(sample_count, 0))
            total, tail = self._cost_components_tensor(
                self.samples, n * sample_count, sampled.reshape(n * sample_count, horizon, -1)
            )
            numerical_checks.append(self._last_cost_actions_valid)
            scores, tails = total.reshape(n, sample_count), tail.reshape(n, sample_count)
            assert cfg.lam_target_neff_fraction is not None
            lam, saturated = temperature_for_neff_fraction_with_details(scores, cfg.lam_target_neff_fraction)
            weights = softmin_weights(scores, lam)
            plans = synthesize_plan(nominal, sampled, weights, self.low, self.high)
            numerical_checks.append(torch.isfinite(nominal).all() & torch.isfinite(sampled).all()
                                    & torch.isfinite(weights).all() & torch.isfinite(plans).all())
            finite = torch.isfinite(scores)
            finite_scores = torch.where(finite, scores, 0.0)
            chunk = {
                "labels": plans[:, 0], "lam": lam,
                "n_eff": weights.square().sum(dim=1).reciprocal(),
                "sample_mean_cost": finite_scores.sum(dim=1) / finite.sum(dim=1),
                "expected_cost": (weights * finite_scores).sum(dim=1),
                "tail_var_ratio": tail_variance_ratio(tails, scores),
                "lambda_saturated": saturated,
            }
            self.singles.load(torch.cat([q0, q0]), torch.cat([v0, v0]), torch.cat([w0, w0]))
            both = self._cost_tensor(self.singles, 2 * n, torch.cat([nominal, plans]))
            numerical_checks.append(self._last_cost_actions_valid)
            chunk.update(nominal_cost=both[:n], plan_cost=both[n:])
            for name, tensor in chunk.items():
                rows[name].append(tensor.cpu().numpy())
        return GPULabels(**{name: np.concatenate(values) for name, values in rows.items()},
                         numerical_valid=bool(torch.stack(numerical_checks).all().item()))


def tail_variance_ratio(
    tails: Float[torch.Tensor, "batch samples"],
    totals: Float[torch.Tensor, "batch samples"],
) -> Float[torch.Tensor, "batch"]:
    """Population tail variance / (total variance + 1e-6), on finite pairs.

    Invalid trajectory scores are already excluded by MPPI's weights. The
    same finite support is used here; a row without valid pairs returns NaN
    explicitly rather than silently fabricating a diagnostic.
    """
    if tails.shape != totals.shape or tails.ndim != 2:
        raise ValueError("tails and totals must have identical shape (batch, samples).")
    finite = torch.isfinite(tails) & torch.isfinite(totals)
    count = finite.sum(dim=1).clamp_min(1)
    tail_mean = torch.where(finite, tails, 0.0).sum(dim=1) / count
    total_mean = torch.where(finite, totals, 0.0).sum(dim=1) / count
    tail_var = torch.where(finite, (tails - tail_mean[:, None]).square(), 0.0).sum(dim=1) / count
    total_var = torch.where(finite, (totals - total_mean[:, None]).square(), 0.0).sum(dim=1) / count
    ratio = tail_var / (total_var + 1e-6)
    return torch.where(finite.any(dim=1), ratio, torch.nan)


def _nonnegative_int(value: object) -> bool:
    """Whether a seed/draw value is an integer accepted by NumPy's RNG."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0


def _validate_initial_states(
    env: Walker2d,
    states: Float[np.ndarray, "states nstate"],
    warmstarts: Float[np.ndarray, "states nv"],
) -> None:
    """Reject malformed initial batches before any CUDA or physics work."""
    if states.ndim != 2 or states.shape[1] != env._nstate:
        raise ValueError(f"states must have shape (M, {env._nstate}).")
    if warmstarts.shape != (len(states), env.model.nv):
        raise ValueError(f"warmstarts must have shape (M, {env.model.nv}).")
    if not np.isfinite(states).all() or not np.isfinite(warmstarts).all():
        raise ValueError("states and warmstarts must be finite.")
