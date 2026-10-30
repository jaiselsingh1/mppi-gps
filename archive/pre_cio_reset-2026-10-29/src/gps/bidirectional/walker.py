"""MuJoCo Walker walk--stop--settle task and metered physics boundary.

All transitions use the existing Walker2d/MuJoCo implementation. The task adds
an announced target-speed schedule and finite complete-outcome criterion. It
contains no policy, planner, learned dynamics or teacher action targets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .evidence import Archive, _json, _packed


ROOT = Path(__file__).resolve().parents[3]
COMPONENTS = ('velocity', 'vertical_velocity', 'torso_angle', 'unhealthy', 'control')


def _array(value, shape=None):
    value = np.asarray(value, dtype=np.float64)
    if (shape is not None and value.shape != shape) or not np.isfinite(value).all():
        raise ValueError('Finite matching array required')
    return np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class WalkerConfig:
    T: int = 600
    walk_steps: int = 400
    burnin: int = 125
    settle_steps: int = 125
    target_velocity: float = 1.5
    walk_distance: float = 1.0
    minimum_walk_mean_velocity: float = .75
    settle_speed: float = .2
    settle_joint_speed: float = 1.0
    failure_penalty: float = 1000.
    frame_skip: int = 4
    nthread: int = 1
    velocity_weight: float = 2.
    vertical_velocity_weight: float = .5
    angle_weight: float = .1
    unhealthy_weight: float = 5.
    control_weight: float = .001
    terminal_unhealthy_weight: float = 20.

    def __post_init__(self):
        for name in ('T', 'walk_steps', 'burnin', 'settle_steps', 'frame_skip', 'nthread'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Positive integer ' + name + ' required')
        if not self.burnin < self.walk_steps <= self.T-self.settle_steps:
            raise ValueError('Walk, burn-in and final settling windows must fit the episode')
        if self.frame_skip != 4:
            raise ValueError('This task preserves Walker frame_skip=4 and .008-second controls')
        for name, value in asdict(self).items():
            if name in ('T', 'walk_steps', 'burnin', 'settle_steps', 'frame_skip', 'nthread'):
                continue
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0.:
                raise ValueError('Finite nonnegative task field ' + name + ' required')
        if min(self.target_velocity, self.walk_distance, self.settle_speed, self.settle_joint_speed) <= 0.:
            raise ValueError('Positive task thresholds required')

    def to_dict(self):
        return asdict(self)

    @property
    def env_kwargs(self):
        return dict(frame_skip=self.frame_skip, cost_style='target_velocity',
            target_velocity=self.target_velocity, vel_cost_weight=self.velocity_weight,
            vertical_velocity_cost_weight=self.vertical_velocity_weight,
            angle_cost_weight=self.angle_weight, unhealthy_cost_weight=self.unhealthy_weight,
            ctrl_cost_weight=self.control_weight, terminal_unhealthy_weight=self.terminal_unhealthy_weight,
            apply_terminal_cost_on_done=False, use_warp=False, nthread=self.nthread)


@dataclass(frozen=True)
class WalkerSnapshot:
    state: np.ndarray
    warmstart: np.ndarray
    integration_state: np.ndarray
    elapsed: int
    hold_count: int
    initial_x: float
    walk_progress: float
    walk_velocity_sum: float
    walk_samples: int
    local_competence: bool
    terminated: bool
    task_hash: str
    context_id: str

    def __post_init__(self):
        for name, shape in (('state', (19,)), ('warmstart', (9,)), ('integration_state', None)):
            object.__setattr__(self, name, _array(getattr(self, name), shape))
        for name in ('elapsed', 'hold_count', 'walk_samples'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError('Nonnegative integer snapshot counters required')
        for name in ('initial_x', 'walk_progress', 'walk_velocity_sum'):
            if not np.isfinite(getattr(self, name)):
                raise ValueError('Finite snapshot history required')
        if type(self.local_competence) is not bool or type(self.terminated) is not bool:
            raise ValueError('Explicit snapshot history flags required')
        if not self.task_hash or not self.context_id:
            raise ValueError('Task and context identity required')

    def to_dict(self):
        return _json(asdict(self))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class WalkerStep:
    pre_snapshot: WalkerSnapshot
    snapshot: WalkerSnapshot
    action: np.ndarray
    observation: np.ndarray
    components: np.ndarray
    stage_cost: float
    terminal_cost: float
    failure_cost: float
    absorbing_cost: float
    total_cost: float
    done: bool
    captured: bool
    info: dict

    @property
    def cost(self):
        return self.total_cost

    @property
    def initial_snapshot(self):
        return self.pre_snapshot


@dataclass(frozen=True)
class WalkerModel:
    initial_snapshot: WalkerSnapshot
    actions: np.ndarray
    prestates: np.ndarray
    states: np.ndarray
    observations: np.ndarray
    component_costs: np.ndarray
    stage_costs: np.ndarray
    terminal_costs: np.ndarray
    failure_costs: np.ndarray
    absorbing_costs: np.ndarray
    terminal_heuristic: np.ndarray
    total_costs: np.ndarray
    actual_partial_costs: np.ndarray
    healthy: np.ndarray
    active: np.ndarray
    hold_counts: np.ndarray
    local_competence: np.ndarray
    captured: np.ndarray
    native_base_scores: np.ndarray
    requested_horizon: int

    @property
    def scores(self):
        return self.total_costs


@dataclass(frozen=True)
class WalkerEpisode:
    states: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    components: np.ndarray
    stage_costs: np.ndarray
    terminal_costs: np.ndarray
    failure_costs: np.ndarray
    absorbing_costs: np.ndarray
    total_costs: np.ndarray
    snapshots: tuple[WalkerSnapshot, ...]
    query_id: str
    metadata: dict
    complete: bool
    captured: bool

    @property
    def prestates(self):
        return self.states[:-1]

    @property
    def poststates(self):
        return self.states[1:]

    @property
    def steps(self):
        return len(self.actions)

    @property
    def cost(self):
        return float(self.total_costs.sum())


class WalkerTask:
    """Actual MuJoCo Walker physics with observable finite task history."""
    obs_dim, action_dim, state_dim = 23, 6, 19
    component_names = COMPONENTS

    def __init__(self, config=None):
        self.config = WalkerConfig() if config is None else (
            WalkerConfig(**config) if isinstance(config, dict) else config)
        if not isinstance(self.config, WalkerConfig):
            raise TypeError('WalkerConfig required')
        from src.envs.walker2d import Walker2d
        import mujoco
        self._mj = mujoco
        self.env = Walker2d(**self.config.env_kwargs)
        if (self.env.state_dim, self.env.action_dim) != (19, 6) or not np.isclose(self.env._dt, .008):
            self.env.close()
            raise ValueError('The unchanged 19-state/6-control Walker model is required')
        files = ('assets/walker2d.xml', 'src/envs/walker2d.py', 'src/envs/mujoco_env.py',
                 'src/gps/bidirectional/walker.py')
        payload = dict(schema='mujoco-walker-walk-stop-settle-v1', config=self.config.to_dict(),
            sources={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
        self.task_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self._integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._integration_size = mujoco.mj_stateSize(self.env.model, self._integration_spec)
        self._boundary_completed = 0
        self._boundary_started = False
        self.elapsed = self.hold_count = self.walk_samples = 0
        self.initial_x = self.walk_progress = self.walk_velocity_sum = 0.
        self.local_competence = self.terminated = False
        self.context_id = 'uninitialized'

    @property
    def remaining(self):
        return 0 if self.terminated else self.config.T-self.elapsed

    def reset(self, *, seed, context_id=None):
        if type(seed) is not int or seed < 0:
            raise ValueError('Explicit nonnegative reset seed required')
        # Preserve the caller's NumPy stream while using the native reset exactly.
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            self.env.reset()
        finally:
            np.random.set_state(state)
        self.elapsed = self.hold_count = self.walk_samples = 0
        self.walk_progress = self.walk_velocity_sum = 0.
        self.initial_x = float(self.env.data.qpos[0])
        self.local_competence = self.terminated = False
        self.context_id = context_id or 'walker-reset-' + str(seed)
        return self.snapshot()

    def snapshot(self):
        integration = np.empty(self._integration_size)
        self._mj.mj_getState(self.env.model, self.env.data, integration, self._integration_spec)
        return WalkerSnapshot(self.env.get_state(), self.env.get_warmstart(), integration,
            self.elapsed, self.hold_count, self.initial_x, self.walk_progress,
            self.walk_velocity_sum, self.walk_samples, self.local_competence,
            self.terminated, self.task_hash, self.context_id)

    def validate_snapshot(self, snapshot):
        if not isinstance(snapshot, WalkerSnapshot) or snapshot.task_hash != self.task_hash:
            raise ValueError('Snapshot belongs to another Walker task')
        if (snapshot.elapsed > self.config.T or snapshot.hold_count > self.config.settle_steps
                or snapshot.integration_state.shape != (self._integration_size,)
                or not np.isclose(snapshot.state[0], snapshot.elapsed*.008, rtol=0., atol=1e-9)):
            raise ValueError('Invalid finite Walker snapshot history/physics time')
        maximum_samples = min(max(snapshot.elapsed-self.config.burnin, 0),
                              self.config.walk_steps-self.config.burnin)
        if snapshot.walk_samples != maximum_samples:
            raise ValueError('Snapshot walk statistics do not match its observed elapsed history')
        if snapshot.local_competence and snapshot.elapsed < self.config.walk_steps:
            raise ValueError('Local competence cannot precede the complete walking window')
        return snapshot

    def restore(self, snapshot):
        self.validate_snapshot(snapshot)
        self._mj.mj_setState(self.env.model, self.env.data, snapshot.integration_state, self._integration_spec)
        self._mj.mj_forward(self.env.model, self.env.data)
        # Forward refreshes derived quantities; preserve integration inputs and
        # warmstart exactly as recorded for the next actual engine step.
        self._mj.mj_setState(self.env.model, self.env.data, snapshot.integration_state, self._integration_spec)
        self.env.set_warmstart(snapshot.warmstart)
        for name in ('elapsed', 'hold_count', 'initial_x', 'walk_progress', 'walk_velocity_sum',
                     'walk_samples', 'local_competence', 'terminated', 'context_id'):
            setattr(self, name, getattr(snapshot, name))
        actual = self.snapshot()
        if any(not np.array_equal(getattr(actual, key), getattr(snapshot, key))
               for key in ('state', 'warmstart', 'integration_state')):
            raise ValueError('Walker integration-state restoration differs')
        return actual

    def target_velocity(self, elapsed):
        return np.where(np.asarray(elapsed) < self.config.walk_steps, self.config.target_velocity, 0.)

    def observation(self, snapshot=None):
        value = self.snapshot() if snapshot is None else self.validate_snapshot(snapshot)
        return self._observation(value.state, value.elapsed, value.hold_count,
            value.walk_progress, value.walk_velocity_sum, value.walk_samples)

    def _observation(self, states, elapsed, hold, progress, velocity_sum, samples):
        states = np.asarray(states)
        native = self.env.rollout_states_to_obs(states)
        shape = states.shape[:-1]
        samples = np.asarray(samples)
        mean = np.divide(velocity_sum, samples, out=np.zeros(shape), where=samples > 0)
        context = np.stack(np.broadcast_arrays(self.target_velocity(elapsed),
            (self.config.T-np.asarray(elapsed))/self.config.T,
            np.maximum(self.config.walk_steps-np.asarray(elapsed), 0)/self.config.T,
            progress, mean, np.asarray(hold)/self.config.settle_steps), axis=-1)
        return np.concatenate((native, context), axis=-1)

    def validate_actions(self, actions, *, batch=False):
        value = np.asarray(actions, dtype=np.float64)
        good = (value.ndim == 3 and min(value.shape[:2]) > 0 and value.shape[-1] == 6) if batch else value.shape == (6,)
        if not good or not np.isfinite(value).all() or np.any(np.abs(value) > 1.):
            raise ValueError('Finite native Walker actions in [-1,1] required')
        return value.copy()

    def _components(self, states, actions, elapsed):
        c = self.config
        states, actions = np.asarray(states), np.asarray(actions)
        healthy = self.env._is_healthy(states[..., 2], states[..., 3])
        vx = states[..., 10]
        parts = np.stack((c.velocity_weight*np.abs(self.target_velocity(elapsed)-vx),
            c.vertical_velocity_weight*states[..., 11]**2,
            c.angle_weight*states[..., 3]**2, c.unhealthy_weight*(~healthy),
            c.control_weight*np.sum(actions**2, axis=-1)), axis=-1)
        return parts, healthy

    def _history(self, state, clock, hold, progress, velocity_sum, samples, local, active):
        cfg = self.config
        health = self.env._is_healthy(state[..., 2], state[..., 3])
        in_walk = active & (clock < cfg.walk_steps)
        # Current displacement is observable task progress. Freeze its boundary
        # value after walking; a historical maximum would hide backtracking
        # position from the policy's otherwise translation-invariant inputs.
        progress = np.where(in_walk, state[..., 1]-self.initial_x, progress)
        measured = active & (clock >= cfg.burnin) & (clock < cfg.walk_steps)
        velocity_sum = velocity_sum + np.where(measured, state[..., 10], 0.)
        samples = samples + measured.astype(int)
        mean = np.divide(velocity_sum, samples, out=np.zeros_like(velocity_sum, dtype=float), where=samples > 0)
        local = np.where(active & (clock+1 == cfg.walk_steps),
            health & (progress >= cfg.walk_distance) & (mean >= cfg.minimum_walk_mean_velocity), local)
        settling = (health & (np.abs(state[..., 10]) <= cfg.settle_speed)
            & (np.linalg.norm(state[..., 11:19], axis=-1) <= cfg.settle_joint_speed))
        hold = np.where(active & (clock >= cfg.T-cfg.settle_steps),
            np.where(settling, np.minimum(hold+1, cfg.settle_steps), 0), hold)
        return hold, progress, velocity_sum, samples, local, health

    def step(self, action):
        action = self.validate_actions(action)
        if not self.remaining:
            raise RuntimeError('No Walker control after true episode termination')
        initial = self.snapshot()
        self._boundary_completed = 0
        self._boundary_started = False
        self._known_model_inputs()
        self._boundary_started = True
        _, native_stage, native_done, native_info = self.env.step(action)
        self._boundary_completed = 1
        state = self.env.get_state()
        parts, healthy = self._components(state, action, self.elapsed)
        history = self._history(state, self.elapsed, np.array(self.hold_count), np.array(self.walk_progress),
            np.array(self.walk_velocity_sum), np.array(self.walk_samples), np.array(self.local_competence), np.array(True))
        self.hold_count, self.walk_progress, self.walk_velocity_sum, self.walk_samples, self.local_competence = (
            int(history[0]), float(history[1]), float(history[2]), int(history[3]), bool(history[4]))
        self.elapsed += 1
        self.terminated = bool(native_done or self.elapsed == self.config.T)
        captured = bool(self.terminated and self.elapsed == self.config.T and healthy
            and self.local_competence and self.hold_count == self.config.settle_steps)
        terminal = float(self.env.terminal_cost(state[None])[0]) if self.terminated else 0.
        failure = self.config.failure_penalty if self.terminated and not captured else 0.
        absorbing = self.config.unhealthy_weight*(self.config.T-self.elapsed) if native_done else 0.
        total = float(parts.sum()+terminal+failure+absorbing)
        final = self.snapshot()
        info = dict(native_info, native_raw_info=dict(native_info),
            native_stage_cost=float(native_stage), cost=total, reward=-total, task_total_cost=total,
            target_velocity=float(self.target_velocity(initial.elapsed)),
            velocity_error=float(abs(self.target_velocity(initial.elapsed)-state[10])),
            local_competence=self.local_competence, walked_distance=self.walk_progress,
            measured_walk_mean_velocity=self.walk_velocity_sum/max(1, self.walk_samples),
            settling_hold=self.hold_count, capture=captured)
        return WalkerStep(initial, final, _array(action), _array(self.observation(final)),
            _array(parts), float(parts.sum()), terminal, float(failure), float(absorbing),
            total, self.terminated, captured, info)

    def model_rollout(self, initial, actions):
        self.validate_snapshot(initial)
        actions = self.validate_actions(actions, batch=True)
        if initial.terminated or initial.elapsed == self.config.T:
            raise RuntimeError('No model rollout from a terminated Walker context')
        requested = actions.shape[1]
        used = actions[:, :min(requested, self.config.T-initial.elapsed)]
        live = self.snapshot()
        self._boundary_completed = 0
        self._boundary_started = False
        self._known_model_inputs()
        self._boundary_started = True
        states, native_scores, _ = self.env.batch_rollout(initial.state, used, initial_warmstart=initial.warmstart)
        self._boundary_completed = int(np.prod(used.shape[:2]))
        current = self.snapshot()
        if any(not np.array_equal(getattr(current, k), getattr(live, k)) for k in ('state', 'warmstart', 'integration_state')):
            self.restore(live)
            raise ValueError('Model call mutated live Walker integration state')
        K, H = used.shape[:2]
        pre = np.concatenate((np.broadcast_to(initial.state, (K, 1, 19)), states[:, :-1]), axis=1)
        active = np.ones((K, H), bool)
        healthy = self.env._is_healthy(states[..., 2], states[..., 3])
        if H > 1:
            active[:, 1:] = np.logical_and.accumulate(healthy[:, :-1], axis=1)
        clocks = initial.elapsed+np.arange(H)
        raw_parts, _ = self._components(states, used, clocks)
        parts = np.where(active[..., None], raw_parts, 0.)
        hold = np.full(K, initial.hold_count, int)
        progress = np.full(K, initial.walk_progress)
        velocity_sum = np.full(K, initial.walk_velocity_sum)
        samples = np.full(K, initial.walk_samples, int)
        local = np.full(K, initial.local_competence, bool)
        obs = np.empty((K, H, self.obs_dim)); holds = np.empty((K, H), int)
        for t, clock in enumerate(clocks):
            obs[:, t] = self._observation(pre[:, t], clock, hold, progress, velocity_sum, samples)
            # History depends on the original context's starting x, not live state.
            original_x = self.initial_x
            self.initial_x = initial.initial_x
            try:
                hold, progress, velocity_sum, samples, local, _ = self._history(
                    states[:, t], clock, hold, progress, velocity_sum, samples, local, active[:, t])
            finally:
                self.initial_x = original_x
            holds[:, t] = hold
        fell = np.any(~healthy, axis=1)
        first_fall = np.argmax(~healthy, axis=1)
        terminal = np.zeros(K); failure = np.zeros(K); absorbing = np.zeros(K)
        if fell.any():
            rows = np.flatnonzero(fell)
            terminal[rows] = self.env.terminal_cost(states[rows, first_fall[rows]])
            failure[rows] = self.config.failure_penalty
            absorbing[rows] = self.config.unhealthy_weight*(self.config.T-initial.elapsed-first_fall[rows]-1)
        reaches_end = initial.elapsed+H == self.config.T
        captured = (~fell) & local & (hold == self.config.settle_steps) if reaches_end else np.zeros(K, bool)
        if reaches_end:
            failure[~fell & ~captured] = self.config.failure_penalty
        heuristic = np.zeros(K) if reaches_end else self.env.terminal_cost(states[:, -1])
        heuristic = np.where(fell, 0., heuristic)
        partial = parts.sum(axis=(1, 2))+terminal+failure+absorbing
        scores = partial+heuristic
        expected_clock = initial.state[0]+.008*np.arange(1, H+1)
        valid = np.isfinite(states).all(axis=-1) & np.isclose(states[..., 0], expected_clock, atol=1e-9, rtol=0.)
        valid_prefix = np.all(valid | ~active, axis=1)
        scores = np.where(valid_prefix, scores, np.inf)
        return WalkerModel(initial, used.copy(), pre, states.copy(), obs, parts, parts.sum(axis=-1),
            terminal, failure, absorbing, heuristic, scores, partial, healthy, active, holds,
            local, captured, np.asarray(native_scores).copy(), requested)

    def close(self):
        self.env.close()

    def _known_model_inputs(self):
        # The native batch API carries FULLPHYSICS and solver warmstart. This
        # unchanged Walker has no activation/mocap state and no external-force
        # intervention. Do not silently model a different input contract.
        if self.env.model.na or self.env.model.nmocap or np.any(self.env.data.qfrc_applied) or np.any(self.env.data.xfrc_applied):
            raise ValueError('The qualified Walker batch input contract excludes external-force or activation interventions')


class WalkerExecutor:
    """Durable model/live query accounting for the real Walker task only."""
    def __init__(self, task, archive: Archive, *, max_requested_controls=None, scope='walker'):
        if not isinstance(task, WalkerTask):
            raise TypeError('An actual MuJoCo WalkerTask is required')
        if max_requested_controls is not None and (type(max_requested_controls) is not int or max_requested_controls < 0):
            raise ValueError('Nonnegative finite control cap required')
        self.task, self.archive, self.scope = task, archive, str(scope)
        self.max_requested_controls = max_requested_controls
        self._requested_controls = 0
        self.latest_query_id = None

    @property
    def ledger_totals(self):
        return self.totals()

    def totals(self):
        """Walker accounting with known partial policy episodes kept separate."""
        rows = [json.loads(line) for line in self.archive.ledger_path.read_text().splitlines()]
        requests = {r['query_id']: r for r in rows if r['kind'] == 'query_requested'}
        completed = {r['query_id']: r for r in rows if r['kind'] == 'query_complete'}
        errors = {r['query_id']: r for r in rows if r['kind'] == 'query_error'}
        if set(completed) & set(errors):
            raise ValueError('Walker query has multiple terminal resolutions')
        unknown = set(requests)-set(completed)-set(errors)
        known = list(completed.values())+list(errors.values())
        return dict(requested_controls=sum(r['requested_controls'] for r in requests.values()),
            requested_physics_controls=sum(r['requested_physics_controls'] for r in requests.values()),
            requested_physics_frames=sum(r['requested_physics_frames'] for r in requests.values()),
            completed_controls=sum(r.get('completed_controls', 0) for r in known),
            completed_physics_frames=sum(r.get('completed_physics_frames', 0) for r in known),
            unknown_partial_controls=sum(r.get('unknown_boundary_controls',
                0 if r.get('call_returned') else requests[r['query_id']]['requested_physics_controls']) for r in errors.values()),
            inflight_controls=sum(requests[q]['requested_physics_controls'] for q in unknown),
            unused_reserved_controls=sum(r.get('unused_reserved_controls', 0) for r in known),
            cutoff_skipped_controls=sum(r.get('cutoff_skipped_controls', 0) for r in requests.values()),
            queries=len(requests), completed_queries=len(completed), error_queries=len(errors),
            inflight_queries=len(unknown), rejected_queries=sum(r['kind'] == 'query_rejected' for r in rows))

    def _query(self, initial, actions, *, mode, kind, metadata):
        metadata = _json({} if metadata is None else metadata)
        query = self.archive.next_query_id(); self.latest_query_id = query
        try:
            self.task.validate_snapshot(initial)
            actions = self.task.validate_actions(actions, batch=mode == 'model')
            if initial.terminated or initial.elapsed == self.task.config.T:
                raise RuntimeError('No control after Walker termination')
            K, H = actions.shape[:2] if mode == 'model' else (1, 1)
            requested = int(K*H)
            physical = int(K*min(H, self.task.config.T-initial.elapsed))
            if self.max_requested_controls is not None and self._requested_controls+requested > self.max_requested_controls:
                raise RuntimeError('Fixed Walker query control budget exhausted before physics')
        except BaseException as error:
            self.archive.append('query_rejected', query_id=query, scope=self.scope, query_kind=kind,
                mode=mode, requested_controls=0, requested_physics_controls=0,
                requested_physics_frames=0, error_type=type(error).__name__, error=str(error))
            raise
        raw = self.archive.save_npz('queries/'+query+'_inputs.npz', actions=actions,
            initial_state=initial.state, initial_warmstart=initial.warmstart,
            integration_state=initial.integration_state, initial_snapshot_json=_packed(initial.to_dict()),
            task_json=_packed(self.task.config.to_dict()), metadata_json=_packed(metadata))
        self._requested_controls += requested
        frames = self.task.config.frame_skip
        self.archive.append('query_requested', query_id=query, scope=self.scope, query_kind=kind,
            mode=mode, owner=metadata.get('owner', {}), metadata=metadata, inputs=raw,
            context_id=initial.context_id, task_hash=initial.task_hash, initial_elapsed=initial.elapsed,
            requested_controls=requested, requested_physics_controls=physical,
            requested_physics_frames=physical*frames, cutoff_skipped_controls=requested-physical)
        started = time.perf_counter(); self.task._boundary_completed = 0; self.task._boundary_started = False
        try:
            result = self.task.model_rollout(initial, actions) if mode == 'model' else self.task.step(actions)
            if mode == 'model':
                arrays = {name: getattr(result, name) for name in ('actions', 'prestates', 'states',
                    'observations', 'component_costs', 'stage_costs', 'terminal_costs', 'failure_costs',
                    'absorbing_costs', 'terminal_heuristic', 'total_costs', 'actual_partial_costs',
                    'healthy', 'active', 'hold_counts', 'local_competence', 'captured', 'native_base_scores')}
                arrays['endpoint_warmstart_semantics'] = np.array('not returned; model endpoints are not acquired-state snapshots')
            else:
                arrays = dict(prestates=initial.state[None], states=result.snapshot.state[None],
                    pre_warmstarts=initial.warmstart[None], warmstarts=result.snapshot.warmstart[None],
                    integration_state=result.snapshot.integration_state, actions=result.action[None],
                    observations=self.task.observation(initial)[None], component_costs=result.components[None],
                    stage_costs=np.array([result.stage_cost]), terminal_costs=np.array([result.terminal_cost]),
                    failure_costs=np.array([result.failure_cost]), absorbing_costs=np.array([result.absorbing_cost]),
                    total_costs=np.array([result.total_cost]), done=np.array([result.done]),
                    captured=np.array([result.captured]), final_snapshot_json=_packed(result.snapshot.to_dict()),
                    metrics_json=_packed(result.info))
            ref = self.archive.save_npz('queries/'+query+'_result.npz', **arrays,
                task_hash=np.array(self.task.task_hash), component_names=np.array(COMPONENTS),
                metadata_json=_packed(metadata), mode=np.array(mode))
            self.archive.append('query_complete', query_id=query, scope=self.scope,
                completed_controls=physical, completed_physics_frames=physical*frames,
                result=ref, query_wall_seconds=time.perf_counter()-started,
                backend='mujoco_cpu', frame_skip=frames)
            return result
        except BaseException as error:
            completed = self.task._boundary_completed
            unknown = physical if self.task._boundary_started and completed == 0 else 0
            self.archive.append('query_error', query_id=query, scope=self.scope,
                call_returned=completed == physical, completed_controls=completed,
                completed_physics_frames=completed*frames, partial_physics_unknown=completed != physical,
                unknown_boundary_controls=unknown, unused_reserved_controls=physical-completed-unknown,
                error_type=type(error).__name__, error=str(error), query_wall_seconds=time.perf_counter()-started)
            raise

    def model_rollout(self, initial, actions, *, kind='planner_bank', metadata=None):
        return self._query(initial, actions, mode='model', kind=kind, metadata=metadata)

    def step(self, action, *, metadata=None):
        return self._query(self.task.snapshot(), action, mode='live', kind='executed_control', metadata=metadata)

    def execute_policy(self, action_callback, snapshot=None, *, kind='actor_rollout', metadata=None):
        """One complete callback-owned episode, with every native step charged.

        Callback inference work is metered by its owner. This function records
        its exact observed pre-action inputs and actually executed outputs;
        no planner action or hypothetical continuation is substituted.
        """
        query = self.archive.next_query_id(); self.latest_query_id = query
        initial = self.task.snapshot() if snapshot is None else snapshot
        try:
            metadata = _json({} if metadata is None else metadata)
            self.task.validate_snapshot(initial)
            if not callable(action_callback) or initial.terminated:
                raise ValueError('A callback and unterminated actual Walker context are required')
            reserved = self.task.config.T-initial.elapsed
            if reserved <= 0 or (self.max_requested_controls is not None
                    and self._requested_controls+reserved > self.max_requested_controls):
                raise RuntimeError('Fixed Walker policy-query reservation exhausted before physics')
        except BaseException as error:
            self.archive.append('query_rejected', query_id=query, scope=self.scope,
                query_kind=kind, mode='policy', requested_controls=0, requested_physics_controls=0,
                requested_physics_frames=0, error_type=type(error).__name__, error=str(error))
            raise
        inputs = self.archive.save_npz('queries/'+query+'_inputs.npz',
            initial_state=initial.state, initial_warmstart=initial.warmstart,
            integration_state=initial.integration_state, initial_snapshot_json=_packed(initial.to_dict()),
            task_json=_packed(self.task.config.to_dict()), metadata_json=_packed(metadata))
        self._requested_controls += reserved
        frames = self.task.config.frame_skip
        self.archive.append('query_requested', query_id=query, scope=self.scope, query_kind=kind,
            mode='policy', owner=metadata.get('owner', {}), metadata=metadata, inputs=inputs,
            context_id=initial.context_id, task_hash=initial.task_hash, initial_elapsed=initial.elapsed,
            requested_controls=reserved, requested_physics_controls=reserved,
            requested_physics_frames=reserved*frames, cutoff_skipped_controls=0,
            reservation_semantics='Complete remaining episode; early termination releases unstarted suffix')
        started = time.perf_counter()
        states = [initial.state.copy()]; snapshots = [initial]
        observations, actions, components, stages, terminals, failures, absorbing, costs = ([] for _ in range(8))
        completed = 0; native_pending = False
        try:
            self.task.restore(initial)
            while self.task.remaining:
                obs = self.task.observation()
                observations.append(obs.copy())
                action = self.task.validate_actions(action_callback(obs.copy()))
                native_pending = True; self.task._boundary_completed = 0; self.task._boundary_started = False
                step = self.task.step(action)
                completed += 1; native_pending = False
                snapshots.append(step.snapshot); states.append(step.snapshot.state.copy())
                actions.append(step.action); components.append(step.components)
                stages.append(step.stage_cost); terminals.append(step.terminal_cost)
                failures.append(step.failure_cost); absorbing.append(step.absorbing_cost); costs.append(step.total_cost)
                if completed % 16 == 0 or step.done:
                    self.archive.append('policy_query_progress', query_id=query,
                        completed_controls=completed, completed_physics_frames=completed*frames,
                        last_snapshot=step.snapshot.to_dict())
            episode = WalkerEpisode(np.asarray(states), np.asarray(observations), np.asarray(actions),
                np.asarray(components), np.asarray(stages), np.asarray(terminals), np.asarray(failures),
                np.asarray(absorbing), np.asarray(costs), tuple(snapshots), query, metadata,
                self.task.terminated, bool(step.captured))
            raw = self.archive.save_npz('queries/'+query+'_result.npz',
                episode_states=episode.states, prestates=episode.prestates, states=episode.poststates,
                observations=episode.observations, actions=episode.actions, component_costs=episode.components,
                stage_costs=episode.stage_costs, terminal_costs=episode.terminal_costs,
                failure_costs=episode.failure_costs, absorbing_costs=episode.absorbing_costs,
                total_costs=episode.total_costs, snapshots_json=_packed([s.to_dict() for s in snapshots]),
                warmstarts=np.stack([s.warmstart for s in snapshots]),
                integration_states=np.stack([s.integration_state for s in snapshots]),
                metadata_json=_packed(metadata), task_hash=np.array(self.task.task_hash),
                component_names=np.array(COMPONENTS), complete=np.array(episode.complete), captured=np.array(episode.captured))
            self.archive.append('query_complete', query_id=query, scope=self.scope,
                result=raw, completed_controls=completed, completed_physics_frames=completed*frames,
                unused_reserved_controls=reserved-completed, query_wall_seconds=time.perf_counter()-started,
                backend='mujoco_cpu', frame_skip=frames, mode='policy', captured=episode.captured)
            return episode
        except BaseException as error:
            known = completed+(self.task._boundary_completed if native_pending else 0)
            unknown = int(native_pending and self.task._boundary_started and self.task._boundary_completed == 0)
            partial = self.archive.save_npz('queries/'+query+'_partial.npz',
                episode_states=np.asarray(states), observations=np.asarray(observations),
                actions=np.asarray(actions), total_costs=np.asarray(costs),
                snapshots_json=_packed([s.to_dict() for s in snapshots]))
            self.archive.append('query_error', query_id=query, scope=self.scope,
                call_returned=unknown == 0, completed_controls=known, completed_physics_frames=known*frames,
                unknown_boundary_controls=unknown, unused_reserved_controls=max(0, reserved-known-unknown),
                partial_physics_unknown=unknown > 0, partial=partial,
                error_type=type(error).__name__, error=str(error), query_wall_seconds=time.perf_counter()-started)
            raise
