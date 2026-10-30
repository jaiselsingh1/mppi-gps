"""Finite Acrobot swing-up/capture task; old environments remain unchanged.

The endpoint cost used by a planner is a heuristic until the actual episode
ends. Complete episode credit contains that cost exactly once, plus an explicit
capture failure charge. No policy or planner is called by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


_ROOT = Path(__file__).resolve().parents[3]
_BASE_SOURCE_HASHES = {
    relative: hashlib.sha256((_ROOT/relative).read_bytes()).hexdigest()
    for relative in ("assets/acrobot.xml", "src/envs/acrobot.py", "src/envs/mujoco_env.py",
                     "src/gps/bidirectional/task.py")
}


def _array(value, shape=None):
    result = np.array(value, dtype=np.float64, copy=True)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError(f"Expected finite array of shape {shape}, got {result.shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TaskConfig:
    T: int = 800
    dt: float = .01
    capture_steps: int = 100
    capture_distance: float = .2
    capture_qvel_norm: float = .5
    local_distance: float = .5
    local_by: int = 600
    failure_penalty: float = 1000.
    reset_angle_radius: float = .1
    reset_velocity_radius: float = .05

    def __post_init__(self):
        for name in ("T", "capture_steps", "local_by"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.capture_steps > self.T or self.local_by > self.T:
            raise ValueError("Capture/local windows must fit inside the episode")
        for field in fields(self):
            if field.name in ("T", "capture_steps", "local_by"):
                continue
            value = getattr(self, field.name)
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid task field {field.name}")
        if self.dt != .01:
            raise ValueError("This task preserves the Acrobot .01-second control interval")
        if min(self.capture_distance, self.capture_qvel_norm, self.local_distance) <= 0:
            raise ValueError("Distance and velocity thresholds must be positive")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise TypeError("Task configuration must be a dictionary")
        return cls(**value)

    def to_dict(self):
        return asdict(self)

    @property
    def task_hash(self):
        payload = {"schema": "acrobot-capture-v2-post-action-fk", "config": self.to_dict(),
                   "base_cost": "unchanged-src.envs.acrobot.Acrobot",
                   "cost_boundary_source": "post-action-full-state-analytic-fk",
                   "base_source_sha256": _BASE_SOURCE_HASHES}
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    state: np.ndarray
    warmstart: np.ndarray
    elapsed: int
    hold_count: int
    task_hash: str
    context_id: str
    warmstart_exact: bool = True

    def __post_init__(self):
        object.__setattr__(self, "state", _array(self.state, (5,)))
        object.__setattr__(self, "warmstart", _array(self.warmstart, (2,)))
        if type(self.elapsed) is not int or self.elapsed < 0:
            raise ValueError("Snapshot elapsed must be a nonnegative integer")
        if type(self.hold_count) is not int or not 0 <= self.hold_count <= self.elapsed:
            raise ValueError("Snapshot hold count cannot exceed elapsed controls")
        for name in ("task_hash", "context_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Snapshot requires {name}")
        if type(self.warmstart_exact) is not bool:
            raise ValueError("Explicit warmstart provenance is required")

    def to_dict(self):
        return dict(state=self.state.tolist(), warmstart=self.warmstart.tolist(),
                    elapsed=self.elapsed, hold_count=self.hold_count,
                    task_hash=self.task_hash, context_id=self.context_id,
                    warmstart_exact=self.warmstart_exact)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class StepResult:
    initial_snapshot: Snapshot
    snapshot: Snapshot
    action: np.ndarray
    observation: np.ndarray
    stage_cost: float
    terminal_cost: float
    capture_failure: float
    total_cost: float
    done: bool
    components: np.ndarray
    tip_distance: float
    qvel_norm: float
    captured: bool
    legacy_stage_cost: float
    legacy_sensors: np.ndarray
    boundary_sensors: np.ndarray
    cost_boundary_source: str = "post-action-full-state-analytic-fk"

    @property
    def cost(self):
        return self.total_cost


@dataclass(frozen=True)
class ModelResult:
    initial_snapshot: Snapshot
    prestates: np.ndarray
    states: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    stage_costs: np.ndarray
    component_costs: np.ndarray
    terminal_heuristic: np.ndarray
    terminal_costs: np.ndarray
    capture_failures: np.ndarray
    total_costs: np.ndarray
    actual_partial_costs: np.ndarray
    hold_counts: np.ndarray
    tip_distances: np.ndarray
    qvel_norms: np.ndarray
    end_snapshots: tuple[Snapshot, ...]
    requested_horizon: int
    legacy_sensors: np.ndarray
    boundary_sensors: np.ndarray
    legacy_base_scores: np.ndarray
    legacy_stage_costs: np.ndarray
    legacy_terminal_heuristic: np.ndarray
    cost_boundary_source: str = "post-action-full-state-analytic-fk"

    @property
    def scores(self):
        return self.total_costs

    @property
    def horizon(self):
        return self.actions.shape[1]


def observation(snapshot: Snapshot, config: TaskConfig) -> np.ndarray:
    q, v = snapshot.state[1:3], snapshot.state[3:5]
    return np.array([np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1]),
                     v[0], v[1], (config.T-snapshot.elapsed)/config.T,
                     min(snapshot.hold_count, config.capture_steps)/config.capture_steps])


def post_action_tip(states):
    """Exact tip geometry for the unchanged two-link XML, without engine calls.

    RK4 sensor timing differs between the original CPU and Warp adapters. The
    task therefore uses this explicit post-action state boundary for BOTH costs
    and capture. Original sensor outputs remain archived as diagnostics.
    """
    states = np.asarray(states)
    if states.shape[-1] != 5 or not np.isfinite(states).all():
        raise ValueError("Tip geometry requires finite full Acrobot states")
    q0, q1 = states[..., 1], states[..., 2]
    return np.stack([np.sin(q0)+np.sin(q0+q1), np.zeros_like(q0),
                     2.+np.cos(q0)+np.cos(q0+q1)], axis=-1)


class AcrobotCaptureTask:
    """MuJoCo Acrobot task wrapper with explicit physics and solver snapshots."""
    component_names = ("target", "qvel", "qvel_excess", "control")

    def __init__(self, config: TaskConfig | dict | None = None, *, seed=0, env=None,
                 backend="cpu", nthread=1):
        self.config = TaskConfig() if config is None else (
            TaskConfig.from_dict(config) if isinstance(config, dict) else config)
        if not isinstance(self.config, TaskConfig):
            raise TypeError("Expected TaskConfig")
        if backend not in ("cpu", "warp"):
            raise ValueError("Backend must be cpu or warp")
        if env is None:
            from .backend import make_acrobot
            env = make_acrobot(backend=backend, nworld=1, frame_skip=1, nthread=nthread)
        from src.envs.mujoco_env import MuJoCoEnv
        if not isinstance(env, MuJoCoEnv):
            raise TypeError('The active Acrobot task requires a MuJoCo environment')
        self.env = env
        self.backend = "warp" if getattr(env, "_use_warp", False) else "cpu"
        self._model_envs = {}
        self.model_construction_events = []
        self.last_model_backend = None
        if self.backend == "warp":
            self._model_envs[env._warp_nworld] = env
        if env.state_dim != 5 or env.action_dim != 1:
            raise ValueError("Task requires the existing Acrobot 5-state/1-control model")
        if not np.isclose(env._dt, self.config.dt, rtol=0, atol=1e-15):
            raise ValueError("Environment control interval differs from task")
        if getattr(env, "_frame_skip", 1) != 1:
            raise ValueError("Acrobot task requires frame_skip=1")
        low, high = env.action_bounds
        if not np.array_equal(low, [-1.]) or not np.array_equal(high, [1.]):
            raise ValueError("Acrobot native action bounds must remain [-1,1]")
        original_parameters = {"_qvel_cost_weight": .15, "_qvel_excess_cost_weight": .05,
            "_qvel_target_gate_power": 2., "_qvel_scale": [2.5, 5.],
            "_qvel_excess_threshold": 8., "_terminal_qvel_cost_weight": 25.}
        for name, expected in original_parameters.items():
            if hasattr(env, name) and not np.array_equal(np.asarray(getattr(env, name)), expected):
                raise ValueError(f"Acrobot original cost parameter changed: {name}")
        self._rng = np.random.default_rng(seed)
        self._reset_count = 0
        self.elapsed = self.hold_count = 0
        self.context_id = "uninitialized"

    @property
    def task_hash(self):
        return self.config.task_hash

    @property
    def remaining(self):
        return self.config.T-self.elapsed

    def reset(self, *, seed=None, context_id=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        state = np.zeros(5)
        state[1:3] = [np.pi, 0.] + self._rng.uniform(
            -self.config.reset_angle_radius, self.config.reset_angle_radius, 2)
        state[3:5] = self._rng.uniform(-self.config.reset_velocity_radius,
                                      self.config.reset_velocity_radius, 2)
        self.env.reset(state=state)
        self.env.set_warmstart(np.zeros(2))
        self.elapsed = self.hold_count = 0
        self.context_id = context_id or f"reset-{self._reset_count:06d}"
        self._reset_count += 1
        return self.snapshot()

    def snapshot(self):
        return Snapshot(self.env.get_state(), self.env.get_warmstart(), self.elapsed,
                        self.hold_count, self.task_hash, self.context_id)

    def validate_snapshot(self, value):
        if not isinstance(value, Snapshot) or value.task_hash != self.task_hash:
            raise ValueError("Snapshot belongs to a different task")
        if value.elapsed > self.config.T:
            raise ValueError("Snapshot is beyond episode termination")
        if value.hold_count > self.config.capture_steps:
            raise ValueError("Snapshot hold count exceeds the saturated capture window")
        if not np.isclose(value.state[0], value.elapsed*self.config.dt, rtol=0, atol=1e-9):
            raise ValueError("Physics time differs from explicit episode clock")
        return value

    def restore(self, value: Snapshot):
        self.validate_snapshot(value)
        self.env.reset(state=value.state.copy())
        self.env.set_warmstart(value.warmstart.copy())
        self.elapsed, self.hold_count = value.elapsed, value.hold_count
        self.context_id = value.context_id
        return self.snapshot()

    def observation(self, value=None):
        value = self.snapshot() if value is None else self.validate_snapshot(value)
        return observation(value, self.config)

    def validate_actions(self, actions, *, batch=False):
        result = np.array(actions, dtype=np.float64, copy=True)
        shape_ok = (result.ndim == 3 and result.shape[0] > 0 and result.shape[1] > 0
                    and result.shape[2] == 1) if batch else result.shape == (1,)
        if not shape_ok or not np.isfinite(result).all() or np.any(abs(result) > 1.):
            raise ValueError("Actions must be finite native controls within [-1,1]")
        return result

    def _parts(self, states, actions, sensors):
        raw = self.env.running_cost_components(states, actions, sensors)
        weighted = self.env.weighted_cost_components(raw)
        components = np.stack([weighted.target_cost, weighted.qvel_cost,
                               weighted.qvel_excess_cost, weighted.control_cost], axis=-1)
        distance = np.linalg.norm(sensors[..., :3]-[0., 0., 4.], axis=-1)
        speed = np.linalg.norm(states[..., 3:5], axis=-1)
        return components, distance, speed

    def step(self, action):
        action = self.validate_actions(action)
        if not self.remaining:
            raise RuntimeError("No physical control is allowed after finite termination")
        initial = self.snapshot()
        _, returned_stage, env_done, _ = self.env.step(action.copy())
        if env_done:
            raise RuntimeError("Unexpected base-environment termination")
        self.elapsed += 1
        state = self.env.get_state()
        legacy_sensors = np.asarray(self.env.data.sensordata).copy()
        sensors = post_action_tip(state)
        parts, distance, speed = self._parts(state, action, sensors)
        inside = distance <= self.config.capture_distance and speed <= self.config.capture_qvel_norm
        self.hold_count = min(self.hold_count+1, self.config.capture_steps) if inside else 0
        stage = float(np.sum(parts))
        if not np.isfinite(stage) or not np.isfinite(returned_stage):
            raise ValueError("Nonfinite running cost")
        done = self.elapsed == self.config.T
        terminal = float(np.asarray(self.env.terminal_cost(state, sensors))) if done else 0.
        captured = done and self.hold_count >= self.config.capture_steps
        failure = self.config.failure_penalty if done and not captured else 0.
        final = self.snapshot()
        self.validate_snapshot(final)
        return StepResult(initial, final, _array(action), self.observation(final), stage,
                          terminal, failure, stage+terminal+failure, done, _array(parts),
                          float(distance), float(speed), captured, float(returned_stage),
                          _array(legacy_sensors), _array(sensors))

    def model_rollout(self, initial: Snapshot, actions):
        self.validate_snapshot(initial)
        actions = self.validate_actions(actions, batch=True)
        requested_horizon = actions.shape[1]
        horizon = min(requested_horizon, self.config.T-initial.elapsed)
        if not horizon:
            raise RuntimeError("Cannot model controls beyond finite termination")
        used = actions[:, :horizon].copy()
        model_env = self._model_environment(len(used))
        # The base implementation preserves live CPU state; Warp may change it.
        # Preserve/restore host bookkeeping even on a failed query, without any step.
        live = self.snapshot()
        try:
            states, base_scores, sensors = model_env.batch_rollout(
                initial.state.copy(), used, initial_warmstart=initial.warmstart.copy())
        finally:
            self.restore(live)
        states, legacy_sensors = np.array(states, copy=True), np.asarray(sensors)
        if self.backend == "warp":
            # The shared Warp adapter omits time from its state buffer. The
            # time-invariant Acrobot clock is known from the requested controls.
            states[..., 0] = initial.state[0]+self.config.dt*np.arange(1, horizon+1)
        self.last_model_backend = dict(backend=self.backend, worlds=len(used),
            horizon=horizon, time_column_reconstructed=self.backend == "warp",
            endpoint_warmstart="explicit-zero-intervention-not-exact-model-solver-state",
            implementation=getattr(model_env, "fast_backend_metadata", None))
        if states.shape != (len(used), horizon, 5) or not np.isfinite(states).all():
            raise ValueError("Model returned invalid full physics states")
        sensors = post_action_tip(states)
        parts, distance, speed = self._parts(states, used, sensors)
        stages = parts.sum(axis=-1)
        heuristic = np.asarray(self.env.terminal_cost(states[:, -1], sensors[:, -1]), dtype=float)
        legacy_parts, _, _ = self._parts(states, used, legacy_sensors)
        legacy_stages = legacy_parts.sum(axis=-1)
        legacy_terminal = self.env.terminal_cost(states[:, -1], legacy_sensors[:, -1])
        pre = np.concatenate([np.broadcast_to(initial.state, (len(used), 1, 5)),
                              states[:, :-1]], axis=1)
        counts = np.empty((len(used), horizon), dtype=np.int64)
        obs = np.empty((len(used), horizon, 8))
        count = np.full(len(used), initial.hold_count, dtype=np.int64)
        for t in range(horizon):
            # Observations belong to the pre-action state and pre-action hold count.
            q, v = pre[:, t, 1:3], pre[:, t, 3:5]
            obs[:, t] = np.column_stack([np.sin(q[:, 0]), np.cos(q[:, 0]),
                np.sin(q[:, 1]), np.cos(q[:, 1]), v,
                np.full(len(used), (self.config.T-initial.elapsed-t)/self.config.T),
                count/self.config.capture_steps])
            inside = (distance[:, t] <= self.config.capture_distance) & (
                speed[:, t] <= self.config.capture_qvel_norm)
            count = np.where(inside, np.minimum(count+1, self.config.capture_steps), 0)
            counts[:, t] = count
        reaches_end = initial.elapsed+horizon == self.config.T
        actual_terminal = heuristic.copy() if reaches_end else np.zeros(len(used))
        failure = np.where(count < self.config.capture_steps, self.config.failure_penalty, 0.) if reaches_end else np.zeros(len(used))
        ends = tuple(Snapshot(states[k, -1], np.zeros(2), initial.elapsed+horizon,
                              int(count[k]), self.task_hash, initial.context_id, False)
                     for k in range(len(used)))
        return ModelResult(initial, _array(pre), _array(states), _array(obs), _array(used),
            _array(stages), _array(parts), _array(heuristic), _array(actual_terminal),
            _array(failure), _array(stages.sum(axis=1)+heuristic+failure),
            _array(stages.sum(axis=1)+actual_terminal+failure), counts, _array(distance),
            _array(speed), ends, requested_horizon, _array(legacy_sensors), _array(sensors),
            _array(base_scores), _array(legacy_stages), _array(legacy_terminal))

    def _model_environment(self, worlds):
        if self.backend == "cpu":
            return self.env
        if worlds not in self._model_envs:
            from .backend import make_acrobot
            started = time.perf_counter()
            original = self.env
            model_env = make_acrobot(backend="warp", frame_skip=1, nthread=1, nworld=worlds,
                qvel_cost_weight=original._qvel_cost_weight,
                qvel_excess_cost_weight=original._qvel_excess_cost_weight,
                qvel_target_gate_power=original._qvel_target_gate_power,
                qvel_scale=tuple(original._qvel_scale),
                qvel_excess_threshold=original._qvel_excess_threshold,
                terminal_qvel_cost_weight=original._terminal_qvel_cost_weight)
            self._model_envs[worlds] = model_env
            self.model_construction_events.append(dict(backend="warp", worlds=worlds,
                construction_wall_seconds=time.perf_counter()-started, controls=0))
        return self._model_envs[worlds]

    def close(self):
        for env in {id(env): env for env in [self.env, *self._model_envs.values()]}.values():
            env.close()


# Explicit, mode-coherent family assignment; never used as an action target.
def trajectory_family(states, tip_distances, *, approach_distance=.5):
    states = np.asarray(states)
    distance = np.asarray(tip_distances)
    if states.ndim != 2 or states.shape[1] != 5 or distance.shape != (len(states),):
        raise ValueError("Family classification requires aligned full-state boundaries")
    hit = np.flatnonzero(distance <= approach_distance)
    if not len(hit):
        return "unreached"
    shoulder = np.unwrap(states[:, 1])
    displacement = float(shoulder[hit[0]]-shoulder[0])
    return "positive" if displacement > 1e-9 else (
        "negative" if displacement < -1e-9 else "stationary")
