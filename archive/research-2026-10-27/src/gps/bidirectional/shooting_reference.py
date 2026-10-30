"""Bounded native-RK4 feasibility reference, separate from MPPI and learning.

Importing this module performs no engine import, integration, solver execution,
or actor work. Guessed shooting states are optimizer variables, never experience.
Only run_reference crosses the explicitly metered simulator boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


class SolveLimit(RuntimeError):
    """A prospective callback limit, not solver success or a physics failure."""


@dataclass(frozen=True)
class ShootingConfig:
    T: int = 800
    dt: float = .01
    max_controls: int = 1_600_000
    reset_seeds: tuple[int, ...] = (191000, 191001)
    qualification_seeds: tuple[int, ...] = tuple(range(192000, 192008))
    final_seeds: tuple[int, ...] = tuple(range(193000, 193032))
    max_residual_calls: int = 80
    max_jacobian_calls: int = 40
    state_fd: tuple[float, ...] = (1e-5, 1e-5, 1e-5, 1e-5)
    action_fd: float = 1e-5
    q_bound: float = 4*np.pi
    velocity_bound: float = 30.
    dynamics_scale: tuple[float, ...] = (1., 1., 5., 5.)
    capture_steps: int = 100
    capture_distance: float = .2
    capture_speed: float = .5
    local_by: int = 600
    local_distance: float = .5
    require_local_at_600_residual: bool = True
    qualification_atol: float = 2e-10
    defect_tolerance: float = 1e-6
    Q: tuple[float, ...] = (10., 10., 1., 1.)
    R: float = .1
    final_min_capture_and_local: int = 29

    def __post_init__(self):
        for name in ("reset_seeds", "qualification_seeds", "final_seeds", "state_fd",
                     "dynamics_scale", "Q"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (self.T != 800 or self.dt != .01 or self.max_controls != 1_600_000
                or self.reset_seeds != (191000, 191001)
                or self.qualification_seeds != tuple(range(192000, 192008))
                or self.final_seeds != tuple(range(193000, 193032))
                or self.max_residual_calls != 80 or self.max_jacobian_calls != 40
                or self.capture_steps != 100 or self.capture_distance != .2
                or self.capture_speed != .5 or self.local_by != 600
                or self.local_distance != .5 or self.require_local_at_600_residual is not True
                or self.final_min_capture_and_local != 29):
            raise ValueError("Only the prospective native feasibility reference is declared")
        for name in ("state_fd", "dynamics_scale", "Q"):
            value = np.asarray(getattr(self, name))
            if value.shape != (4,) or not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError("Four positive finite values required: "+name)
        for name in ("action_fd", "q_bound", "velocity_bound", "qualification_atol",
                     "defect_tolerance", "R"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("Positive finite reference setting required: "+name)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def protocol(config=None):
    c = ShootingConfig() if config is None else config
    return dict(schema="native-rk4-shooting-reference-v0", config=asdict(c),
        work_bound=dict(residual_controls=4*80*800, jacobian_controls=4*40*8000,
            candidate_live_replays=8*800, tracker_jacobian_controls=2*8000,
            nominal_and_selection_live_controls=10*800, final_live_controls=32*800,
            qualification_allowance=8000, total=1_600_000,
            maximum_heterogeneous_solver_batches=4*(80+40)),
        objective="Feasibility residual only; original task return is measured on actual full episodes",
        capture_window="POST boundaries701..800 inclusive; no post-episode padding",
        local_residual="distance at boundary600 <=.5 is a stronger sufficient optimization target; acceptance uses original any-by600 predicate",
        state_search_box="q in[-4pi,4pi], v in[-30,30]; a finite optimizer search restriction, not modified physics",
        initialization="Two opposite shoulder windings per root, zero native controls; guessed trajectories are not experience",
        tracker_selection="At most one successful candidate per root. Fresh seeds allocated round-robin across retained trackers; each must pass own nominal and all allocated fresh episodes. Lowest allocated fresh mean cost, stable tracker-index ties",
        final_gate="Fixed selected tracker; >=29/32 actual complete captures AND original local criterion; no controller modification during panel",
        stopping_rule="No retries, source changes, extra starts, actor training, data promotion or budget replenishment",
        interpretation="Independent trajectory-optimization feasibility reference, not MPPI, GPS or an efficiency result",
        actor_rows=0, adam_steps=0,
        numerical_optimization_work="SciPy solve/residual/Jacobian counts and wall are reported separately, not zero")


def tip_squared(physical):
    x = np.asarray(physical, dtype=np.float64)
    q0, q1 = x[..., 0], x[..., 1]
    a, b = np.sin(q0)+np.sin(q0+q1), np.cos(q0)+np.cos(q0+q1)-2.
    return a*a+b*b


def tip_squared_gradient(physical):
    x = np.asarray(physical, dtype=np.float64)
    q0, q1 = x[..., 0], x[..., 1]
    a, b = np.sin(q0)+np.sin(q0+q1), np.cos(q0)+np.cos(q0+q1)-2.
    out = np.zeros_like(x)
    out[..., 0] = 2*a*(np.cos(q0)+np.cos(q0+q1))-2*b*(np.sin(q0)+np.sin(q0+q1))
    out[..., 1] = 2*a*np.cos(q0+q1)-2*b*np.sin(q0+q1)
    return out


def finite_difference_inputs(states, actions, *, state_fd=(1e-5,)*4, action_fd=1e-5):
    """Ten native H1 rows per node; actions remain bounded at either endpoint."""
    s = np.asarray(states, dtype=np.float64)
    u = np.asarray(actions, dtype=np.float64)
    eps = np.asarray(state_fd, dtype=np.float64)
    if (s.ndim != 2 or s.shape[1] != 5 or u.shape != (len(s), 1)
            or eps.shape != (4,) or np.any(eps <= 0) or action_fd <= 0
            or not all(np.isfinite(a).all() for a in (s, u, eps)) or np.any(abs(u) > 1)):
        raise ValueError("Finite aligned native states/actions and positive stencils required")
    ss = np.repeat(s[:, None, :], 10, axis=1)
    uu = np.repeat(u[:, None, :], 10, axis=1)
    for j in range(4):
        ss[:, 2*j, j+1] += eps[j]
        ss[:, 2*j+1, j+1] -= eps[j]
    plus = np.minimum(u[:, 0]+action_fd, 1.)
    minus = np.maximum(u[:, 0]-action_fd, -1.)
    uu[:, 8, 0], uu[:, 9, 0] = plus, minus
    return ss.reshape(-1, 5), uu.reshape(-1, 1), plus-minus


def derivative_blocks(outputs, action_denominator, state_fd=(1e-5,)*4):
    value = np.asarray(outputs, dtype=np.float64).reshape(-1, 10, 5)[..., 1:5]
    denominator = np.asarray(action_denominator, dtype=np.float64)
    if denominator.shape != (len(value),) or np.any(denominator <= 0) or not np.isfinite(value).all():
        raise ValueError("Invalid finite-difference result")
    A = np.stack([(value[:, 2*j]-value[:, 2*j+1])/(2*state_fd[j]) for j in range(4)], axis=-1)
    B = ((value[:, 8]-value[:, 9])/denominator[:, None])[..., None]
    return A, B


class ShootingProblem:
    """Sparse native multiple shooting; callback work is separately hard capped."""
    def __init__(self, initial, executor, archive, solve_id, config=None):
        self.config = ShootingConfig() if config is None else config
        self.initial = np.asarray(initial, dtype=np.float64)
        if self.initial.shape != (5,) or not np.isfinite(self.initial).all() or self.initial[0] != 0:
            raise ValueError("Ordinary native root required")
        self.executor, self.archive, self.solve_id = executor, archive, str(solve_id)
        self.residual_calls = self.jacobian_calls = 0
        self.best_variables = None
        self.best_residual_norm = float("inf")
        self.best_scaled_defect_max = self.best_constraint_max = None

    def unpack(self, variables):
        z = np.asarray(variables, dtype=np.float64)
        if z.shape != (5*self.config.T,) or not np.isfinite(z).all():
            raise ValueError("Shooting variables must be finite native states and controls")
        states = np.vstack((self.initial[1:], z[:4*self.config.T].reshape(self.config.T, 4)))
        return states, z[4*self.config.T:, None]

    def full_pre_states(self, physical):
        return np.column_stack((np.arange(self.config.T)*self.config.dt, physical[:-1]))

    def initial_guess(self, winding):
        if winding not in (-1, 1):
            raise ValueError("One of two fixed winding guesses required")
        c = self.config
        t = np.arange(1, c.T+1)*c.dt
        p = np.minimum(t/7., 1.)
        blend, rate = p*p*(3-2*p), 6*p*(1-p)/7.
        goal = np.array([0. if winding < 0 else 2*np.pi, 0.])
        delta = goal-self.initial[1:3]
        states = np.column_stack((self.initial[1:3]+blend[:, None]*delta,
                                  rate[:, None]*delta))
        return np.concatenate((states.ravel(), np.zeros(c.T)))

    def bounds(self):
        c = self.config
        state = np.array([c.q_bound, c.q_bound, c.velocity_bound, c.velocity_bound])
        high = np.concatenate((np.tile(state, c.T), np.ones(c.T)))
        return -high, high

    def _capture(self, states):
        c = self.config
        tail = states[-c.capture_steps:]
        dist = tip_squared(tail)/c.capture_distance**2-1.
        speed = np.square(tail[:, 2:]).sum(axis=1)/c.capture_speed**2-1.
        result = [np.maximum(dist, 0.), np.maximum(speed, 0.)]
        if c.require_local_at_600_residual:
            result.append(np.maximum([tip_squared(states[c.local_by])/c.local_distance**2-1.], 0.))
        return np.concatenate(result)

    def residual(self, variables):
        c = self.config
        if self.residual_calls >= c.max_residual_calls:
            raise SolveLimit("Residual callback limit reached before model work")
        states, actions = self.unpack(variables)
        index = self.residual_calls; self.residual_calls += 1
        self.archive.append("shooting_residual_requested", solve_id=self.solve_id, callback=index,
            physical_controls=0, model_controls_in_following_query=c.T)
        result = self.executor.evaluate(self.full_pre_states(states), actions,
            purpose="shooting_residual", metadata=dict(solve_id=self.solve_id, callback=index,
                optimizer_states_are_experience=False))
        dynamics = (states[1:]-result.states[:, 1:5])/np.asarray(c.dynamics_scale)
        residual = np.concatenate((dynamics.ravel(), self._capture(states)))
        norm = float(np.linalg.norm(residual))
        raw = self.archive.save_npz(f"shooting/{self.solve_id}/residual{index:03d}.npz",
            variables=np.asarray(variables), residual=residual, dynamics_defects=dynamics,
            guessed_states=states, actions=actions, guessed_states_are_experience=np.array(False))
        self.archive.append("shooting_residual_complete", solve_id=self.solve_id, callback=index,
            query_id=result.query_id, residual_norm=norm, raw=raw,
            query_archive="evidence/heterogeneous",
            analytic_constraint_rows=len(residual), new_simulator_controls=0)
        if norm < self.best_residual_norm:
            self.best_residual_norm, self.best_variables = norm, np.array(variables, copy=True)
            self.best_scaled_defect_max = float(np.max(abs(dynamics)))
            self.best_constraint_max = float(np.max(self._capture(states)))
        return residual

    def jacobian(self, variables):
        from scipy.sparse import coo_matrix
        c = self.config
        if self.jacobian_calls >= c.max_jacobian_calls:
            raise SolveLimit("Jacobian callback limit reached before model work")
        states, actions = self.unpack(variables)
        index = self.jacobian_calls; self.jacobian_calls += 1
        ss, uu, denominator = finite_difference_inputs(self.full_pre_states(states), actions,
            state_fd=c.state_fd, action_fd=c.action_fd)
        self.archive.append("shooting_jacobian_requested", solve_id=self.solve_id, callback=index,
            physical_controls=0, model_controls_in_following_query=10*c.T)
        result = self.executor.evaluate(ss, uu, purpose="shooting_jacobian",
            metadata=dict(solve_id=self.solve_id, callback=index, state_fd=c.state_fd,
                action_fd=c.action_fd, stencil="4 central state pairs plus bound-aware action pair"))
        A, B = derivative_blocks(result.states, denominator, c.state_fd)
        rows, cols, values = [], [], []
        scales = np.asarray(c.dynamics_scale)
        for t in range(c.T):
            for j in range(4):
                r = 4*t+j
                rows.append(r); cols.append(4*t+j); values.append(1./scales[j])
                if t:
                    for k in range(4):
                        rows.append(r); cols.append(4*(t-1)+k); values.append(-A[t, j, k]/scales[j])
                rows.append(r); cols.append(4*c.T+t); values.append(-B[t, j, 0]/scales[j])
        tail = states[-c.capture_steps:]
        dist = tip_squared(tail)/c.capture_distance**2-1.
        speeds = np.square(tail[:, 2:]).sum(axis=1)/c.capture_speed**2-1.
        dg = tip_squared_gradient(tail)/c.capture_distance**2
        for k in range(c.capture_steps):
            node = c.T-c.capture_steps+k  # variable x_(node+1)
            for j in range(4):
                rows.append(4*c.T+k); cols.append(4*node+j)
                values.append(float(dg[k, j]) if dist[k] > 0 else 0.)
            for j in range(2):
                rows.append(4*c.T+c.capture_steps+k); cols.append(4*node+2+j)
                values.append(float(2*tail[k, 2+j]/c.capture_speed**2) if speeds[k] > 0 else 0.)
        nrows = 4*c.T+2*c.capture_steps
        if c.require_local_at_600_residual:
            active = tip_squared(states[c.local_by]) > c.local_distance**2
            gradient = tip_squared_gradient(states[c.local_by])/c.local_distance**2
            for j in range(4):
                rows.append(nrows); cols.append(4*(c.local_by-1)+j)
                values.append(float(gradient[j]) if active else 0.)
            nrows += 1
        matrix = coo_matrix((values, (rows, cols)), shape=(nrows, 5*c.T)).tocsr()
        raw = self.archive.save_npz(f"shooting/{self.solve_id}/jacobian{index:03d}.npz",
            variables=np.asarray(variables), A=A, B=B, action_denominator=denominator,
            csr_data=matrix.data, csr_indices=matrix.indices, csr_indptr=matrix.indptr,
            csr_shape=np.array(matrix.shape))
        self.archive.append("shooting_jacobian_complete", solve_id=self.solve_id, callback=index,
            query_id=result.query_id, query_archive="evidence/heterogeneous", raw=raw, new_simulator_controls=0)
        return matrix


def solve(problem, winding):
    """Explicit call only; custom callback caps also bound SciPy's hidden work."""
    from scipy.optimize import least_squares
    initial = problem.initial_guess(winding)
    started = time.perf_counter(); limit = None
    problem.archive.append("shooting_solve_begin", solve_id=problem.solve_id, winding=winding,
        initial_guess_is_experience=False, solver="scipy-least_squares-trf-lsmr",
        max_nfev=problem.config.max_residual_calls, max_jacobian_calls=problem.config.max_jacobian_calls)
    result = None
    try:
        result = least_squares(problem.residual, initial, jac=problem.jacobian,
            bounds=problem.bounds(), method="trf", tr_solver="lsmr", x_scale="jac",
            loss="linear", max_nfev=problem.config.max_residual_calls,
            ftol=1e-8, xtol=1e-8, gtol=1e-8)
    except SolveLimit as error:
        limit = str(error)
    chosen = initial if problem.best_variables is None else problem.best_variables
    physical, actions = problem.unpack(chosen)
    record = dict(solve_id=problem.solve_id, winding=winding, solver_success=bool(result.success) if result is not None else False,
        solver_status=int(result.status) if result is not None else None,
        solver_message=str(result.message) if result is not None else limit,
        solver_nfev=int(result.nfev) if result is not None else None,
        solver_njev=int(result.njev) if result is not None and result.njev is not None else None,
        residual_calls=problem.residual_calls, jacobian_calls=problem.jacobian_calls,
        best_residual_norm=problem.best_residual_norm if np.isfinite(problem.best_residual_norm) else None,
        best_scaled_dynamics_defect_max=problem.best_scaled_defect_max,
        best_capture_local_violation_max=problem.best_constraint_max,
        guessed_constraints_closed=bool(problem.best_scaled_defect_max is not None and
            max(problem.best_scaled_defect_max, problem.best_constraint_max) <= problem.config.defect_tolerance),
        accepted_as_experience=False, solve_wall_seconds=time.perf_counter()-started)
    raw = problem.archive.save_npz(f"shooting/{problem.solve_id}/candidate.npz",
        guessed_states=physical, actions=actions, initial_state=problem.initial,
        guessed_states_are_experience=np.array(False))
    record["raw"] = raw
    problem.archive.append("shooting_solve_end", **record)
    return actions, record


class BudgetedQueryExecutor:
    """Existing query ledger remains authoritative; reservations overlap it."""
    def __init__(self, executor, budget):
        self.executor, self.budget = executor, budget
        self.task, self.archive = executor.task, executor.archive

    @property
    def latest_query_id(self):
        return self.executor.latest_query_id

    def _call(self, method, n, *args, **kwargs):
        token = self.budget.reserve(n, purpose="ordinary_"+method)
        previous_query_id = self.latest_query_id
        self.archive.append("reference_budget_reserved", requested=n,
            overlap_with="The following ordinary query_requested; not additional physics",
            purpose=method, budget_reservation_id=token.reservation_id,
            budget_requested_cumulative=token.cumulative_requested_controls,
            task_model_restore_forward_calls_planned=1 if method == "model_rollout" else 0)
        try:
            value = getattr(self.executor, method)(*args, **kwargs)
        except BaseException:
            # Returned-invalid/archive failures still consumed known native work.
            returned, attempted = False, False
            if self.latest_query_id != previous_query_id:
                for line in self.archive.ledger_path.open():
                    row = json.loads(line)
                    if row.get("query_id") == self.latest_query_id:
                        if row["kind"] == "query_requested": attempted = True
                        elif row["kind"] == "query_error": returned = bool(row["call_returned"])
            self.budget.error(token, call_returned=returned, attempted=attempted)
            raise
        self.budget.complete(token)
        return value

    def step(self, action, *, metadata=None):
        return self._call("step", 1, action, metadata=metadata)

    def model_rollout(self, initial, actions, *, kind="native_reference", metadata=None):
        actions = np.asarray(actions)
        return self._call("model_rollout", int(actions.shape[0]*actions.shape[1]), initial, actions,
                          kind=kind, metadata=metadata)


def _record_reset(archive, task, seed, context):
    archive.append("zero_time_reset_requested", seed=seed, context_id=context,
        direct_mj_forward_calls=1, new_simulator_controls=0)
    snapshot = task.reset(seed=seed, context_id=context)
    archive.append("zero_time_reset_complete", seed=seed, context_id=context,
        snapshot=snapshot.to_dict(), direct_mj_forward_calls=1, overlap_with_request=True,
        new_simulator_controls=0)
    return snapshot


def execute_native(executor, initial, controller, episode_id, *, owner_hash, source):
    """Full live continuation only; no imagined states or endpoint padding."""
    from .evidence import EpisodeTrace
    source = dict(source, controller="native_shooting_reference", excluded_from_teacher_aux=True)
    executor.archive.append("zero_time_restore_requested", context_id=initial.context_id,
        direct_mj_forward_calls=1, new_simulator_controls=0)
    executor.task.restore(initial)
    steps, ids = [], []
    owner = dict(actor_hash=None, controller_hash=owner_hash, semantics="native_shooting_reference",
                 excluded_from_teacher_aux=True)
    while executor.task.remaining:
        before = executor.task.snapshot()
        action = np.asarray(controller(before), dtype=np.float64)
        steps.append(executor.step(action, metadata=dict(owner=owner, episode_id=episode_id,
            source=source, excluded_from_teacher_aux=True)))
        ids.append(executor.latest_query_id)
    trace = EpisodeTrace.from_steps(initial, steps, config=executor.task.config,
        actor_owner=owner, execution_semantics="native_shooting_reference", source=source, query_ids=ids)
    raw = trace.archive(executor.archive, episode_id)
    local = bool(np.any((trace.elapsed <= executor.task.config.local_by)&
                        (trace.tip_distances <= executor.task.config.local_distance)))
    outcome = dict(episode_id=episode_id, context_id=initial.context_id, cost=trace.total_cost,
        capture=trace.captured, local=local, complete=trace.complete,
        admissible=bool(trace.complete and trace.captured and local), raw=raw,
        controller_hash=owner_hash, actor_hash=None, task_hash=initial.task_hash,
        query_ids=ids, excluded_from_teacher_aux=True)
    executor.archive.append("reference_episode_outcome", **outcome)
    return trace, outcome


def tracking_gains(A, B, Q=(10., 10., 1., 1.), R=.1):
    A, B = np.asarray(A), np.asarray(B)
    if A.ndim != 3 or A.shape[1:] != (4, 4) or B.shape != (len(A), 4, 1):
        raise ValueError("Aligned native discrete Jacobians required")
    P = np.diag(Q); gains = np.empty((len(A), 1, 4))
    for t in range(len(A)-1, -1, -1):
        S = R+B[t].T@P@B[t]
        K = np.linalg.solve(S, B[t].T@P@A[t])
        gains[t] = K
        P = np.diag(Q)+A[t].T@P@A[t]-A[t].T@P@B[t]@K
        P = (P+P.T)/2
    if not np.isfinite(gains).all():
        raise ValueError("Nonfinite native tracking gains")
    return gains


def tracking_action(snapshot, states, actions, gains):
    t = snapshot.elapsed
    error = np.asarray(snapshot.state[1:])-states[t, 1:]
    error[:2] = (error[:2]+np.pi)%(2*np.pi)-np.pi
    return np.clip(actions[t]-gains[t]@error, -1., 1.)


def qualify_heterogeneous(executor, heterogeneous, config):
    """Paid fixed native comparisons before any optimization callback."""
    from .task import Snapshot
    states, actions = [], []
    qcases = ((np.pi, 0.), (0., 0.), (.4, -.7), (2.4, 1.2))
    vcases = ((0., 0.), (.2, -.3), (3., -2.), (-4., 1.))
    for i in range(16):
        elapsed = (53*i)%800
        states.append([elapsed*.01, *qcases[i//4], *vcases[i%4]])
        actions.append([(-1., -.3, .4, 1.)[i%4]])
    states, actions = np.array(states), np.array(actions)
    batched = heterogeneous.evaluate(states, actions, purpose="heterogeneous_backend_qualification",
        metadata=dict(fixed_states=True, before_any_solver=True, zero_warmstart=True))
    rows = []
    for i in range(len(states)):
        outputs, refs = [], []
        for warm in (np.zeros(2), np.array([.125, -.25])):
            snapshot = Snapshot(states[i], warm, int(round(states[i, 0]/.01)), 0,
                executor.task.task_hash, f"native-backend:{i}", False)
            result = executor.model_rollout(snapshot, actions[i:i+1, None, :],
                kind="native_heterogeneous_qualification", metadata=dict(case=i,
                    warmstart_intervention=warm.tolist(), dynamics_only_comparison=True))
            outputs.append(result.states[0, 0]); refs.append(executor.latest_query_id)
        errors = [float(np.max(abs(outputs[0]-batched.states[i]))),
                  float(np.max(abs(outputs[0]-outputs[1])))]
        rows.append(dict(case=i, scalar_query_ids=refs, heterogeneous_query_id=batched.query_id,
            scalar_query_archive="evidence/native", heterogeneous_query_archive="evidence/heterogeneous",
            heterogeneous_error=errors[0], warmstart_intervention_error=errors[1],
            passed=max(errors) <= config.qualification_atol))
    result = dict(passed=all(r["passed"] for r in rows), rows=rows,
        controls=48, criteria_atol=config.qualification_atol,
        scope="Fixed16 native H1 states; heterogeneous vs scalar AND nonzero vs zero solver warmstart; not long-horizon solver/controller qualification")
    executor.archive.append("native_heterogeneous_qualification", **result)
    return result


def run_reference(out, *, source_fingerprint, config=None):
    """Explicit runtime entry. No retry or silent extension after any error."""
    from .evidence import Archive, QueryExecutor
    from .hetero_executor import HeterogeneousH1Executor, SharedControlBudget
    from .task import AcrobotCaptureTask, TaskConfig
    c = ShootingConfig() if config is None else config
    out = Path(out); started = time.perf_counter()
    archive = Archive(out/"evidence"/"native")
    heterogeneous_archive = Archive(out/"evidence"/"heterogeneous")
    budget = SharedControlBudget(c.max_controls)
    task = None; error = None; close_errors = []; candidates = []; trackers = []; qualification = None; final = []
    try:
        archive.append("backend_construction_requested", backend="cpu", new_simulator_controls=0)
        task = AcrobotCaptureTask(TaskConfig(), backend="cpu", nthread=1)
        import mujoco
        if (int(task.env.model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_RK4)
                or not int(task.env.model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
                or task.env._frame_skip != 1):
            raise ValueError("Native unchanged RK4/constraint-disabled/frame-skip1 model required")
        archive.append("backend_constructed", backend="cpu", task_hash=task.task_hash,
            source_fingerprint=source_fingerprint, native_control_dt=.01, actor_rows=0, adam_steps=0)
        native = QueryExecutor(task, archive, scope="native_reference", max_requested_controls=c.max_controls)
        executor = BudgetedQueryExecutor(native, budget)
        heterogeneous = HeterogeneousH1Executor(task, heterogeneous_archive, budget=budget,
            source_fingerprints=dict(run_source_fingerprint=source_fingerprint),
            owner=dict(controller="native_shooting_reference", actor_hash=None,
                       source_fingerprint=source_fingerprint))
        qualification = qualify_heterogeneous(executor, heterogeneous, c)
        write_json(out/"heterogeneous_qualification.json", qualification)
        if not qualification["passed"]:
            raise RuntimeError("Native heterogeneous qualification failed before solver")
        for root_index, seed in enumerate(c.reset_seeds):
            initial = _record_reset(archive, task, seed, f"ordinary:{seed}")
            root_candidates = []
            for winding in (-1, 1):
                solve_id = f"root{root_index}_winding{'minus' if winding < 0 else 'plus'}"
                problem = ShootingProblem(initial.state, heterogeneous, archive, solve_id, c)
                actions, record = solve(problem, winding)
                controller_hash = identity(dict(algorithm="native-shooting-open-loop", solve_id=solve_id,
                    actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(), task_hash=task.task_hash,
                    source_fingerprint=source_fingerprint))
                repeats = []
                for repeat in range(2):
                    trace, outcome = execute_native(executor, initial,
                        lambda s, u=actions: u[s.elapsed], f"{solve_id}_replay{repeat}",
                        owner_hash=controller_hash, source=dict(solve=record, replication_is_independent_evidence=False))
                    repeats.append(outcome)
                candidate = dict(root_index=root_index, seed=seed, solve=record, replays=repeats,
                    accepted=all(r["admissible"] for r in repeats))
                candidates.append(candidate)
                if candidate["accepted"]:
                    root_candidates.append((float(np.mean([r["cost"] for r in repeats])), len(candidates)-1, trace))
            if root_candidates:
                _, candidate_index, trace = min(root_candidates, key=lambda x:(x[0], x[1]))
                inputs, uu, denominator = finite_difference_inputs(trace.states[:-1], trace.actions,
                    state_fd=c.state_fd, action_fd=c.action_fd)
                diff = heterogeneous.evaluate(inputs, uu, purpose="native_tracker_linearization",
                    metadata=dict(root_index=root_index, candidate_index=candidate_index, state_fd=c.state_fd, action_fd=c.action_fd))
                A, B = derivative_blocks(diff.states, denominator, c.state_fd)
                K = tracking_gains(A, B, c.Q, c.R)
                raw = archive.save_npz(f"trackers/root{root_index}.npz", states=trace.states,
                    actions=trace.actions, A=A, B=B, gains=K, Q=np.array(c.Q), R=np.array(c.R))
                tracker_hash = identity(dict(algorithm="native-shooting-tvlqr-clipped", raw=raw,
                    source_fingerprint=source_fingerprint, task_hash=task.task_hash))
                trackers.append(dict(index=len(trackers), root_index=root_index, candidate_index=candidate_index,
                    raw=raw, controller_hash=tracker_hash, trace=trace, gains=K, qualification=[],
                    linearization_query_id=diff.query_id, linearization_query_archive="evidence/heterogeneous",
                    reference_episode_raw=candidates[candidate_index]["replays"][-1]["raw"],
                    tracking_rule="clip(u_ref[t]-K[t]@[wrapped(q-q_ref),v-v_ref],-1,1)",
                    source_fingerprint=source_fingerprint))
        if trackers:
            for tracker in trackers:
                trace = tracker["trace"]
                _, nominal = execute_native(executor, trace.initial_snapshot,
                    lambda s, t=tracker: tracking_action(s, t["trace"].states, t["trace"].actions, t["gains"]),
                    f"tracker{tracker['index']}_nominal", owner_hash=tracker["controller_hash"],
                    source=dict(tracker=tracker["raw"], panel="nominal_tracker"))
                tracker["nominal"] = nominal
            for i, seed in enumerate(c.qualification_seeds):
                tracker = trackers[i%len(trackers)]
                initial = _record_reset(archive, task, seed, f"ordinary:{seed}")
                _, outcome = execute_native(executor, initial,
                    lambda s, t=tracker: tracking_action(s, t["trace"].states, t["trace"].actions, t["gains"]),
                    f"tracker{tracker['index']}_selection{seed}", owner_hash=tracker["controller_hash"],
                    source=dict(tracker=tracker["raw"], panel="tracker_selection", fixed_seed=seed))
                tracker["qualification"].append(outcome)
            eligible = [t for t in trackers if t["nominal"]["admissible"] and t["qualification"]
                        and all(r["admissible"] for r in t["qualification"])]
            if eligible:
                chosen = min(eligible, key=lambda t:(float(np.mean([r["cost"] for r in t["qualification"]])), t["index"]))
                archive.append("native_tracker_frozen", tracker_index=chosen["index"],
                    controller_hash=chosen["controller_hash"], raw=chosen["raw"],
                    before_fresh_final_seeds=list(c.final_seeds))
                for seed in c.final_seeds:
                    initial = _record_reset(archive, task, seed, f"ordinary:{seed}")
                    _, outcome = execute_native(executor, initial,
                        lambda s, t=chosen: tracking_action(s, t["trace"].states, t["trace"].actions, t["gains"]),
                        f"final{seed}", owner_hash=chosen["controller_hash"],
                        source=dict(tracker=chosen["raw"], panel="fresh_final", fixed_seed=seed))
                    final.append(outcome)
    except BaseException as failure:
        error = dict(type=type(failure).__name__, message=str(failure))
        archive.append("native_reference_error", error=error, retry=False)
    finally:
        if task is not None:
            try:
                task.close()
            except BaseException as failure:
                close_errors.append(dict(type=type(failure).__name__, message=str(failure)))
        for to_close in (archive, heterogeneous_archive):
            try:
                to_close.close()
            except BaseException as failure:
                close_errors.append(dict(type=type(failure).__name__, message=str(failure)))
    native_accounting, heterogeneous_accounting = archive.totals(), heterogeneous_archive.totals()
    accounting = {key:native_accounting[key]+heterogeneous_accounting[key]
                  for key in native_accounting if type(native_accounting[key]) is int}
    accounting["incomplete_final_line"] = bool(native_accounting["incomplete_final_line"] or heterogeneous_accounting["incomplete_final_line"])
    # Budget requests overlap exactly the physical requests in this one archive.
    budget_totals = budget.totals()
    callback_counts = {kind:0 for kind in ("shooting_solve_begin", "shooting_solve_end",
        "shooting_residual_requested", "shooting_residual_complete",
        "shooting_jacobian_requested", "shooting_jacobian_complete")}
    for line in archive.ledger_path.open():
        event = json.loads(line)
        if event["kind"] in callback_counts:
            callback_counts[event["kind"]] += 1
    if error or close_errors:
        stopped_at = "execution_or_evidence_error"
    elif not any(candidate["accepted"] for candidate in candidates):
        stopped_at = "no_admissible_native_candidate"
    elif not final:
        stopped_at = "no_admissible_feedback_tracker"
    elif sum(r["admissible"] for r in final) < c.final_min_capture_and_local:
        stopped_at = "fresh_final_feedback_qualification_failed"
    else:
        stopped_at = "tested_native_reference_qualified"
    result = dict(schema="native-rk4-shooting-reference-result-v0", source_fingerprint=source_fingerprint,
        status="error" if error or close_errors else "complete", stopped_at=stopped_at,
        task_hash=None if task is None else task.task_hash, error=error, close_errors=close_errors,
        qualification=qualification, candidates=candidates,
        trackers=[{k:v for k,v in t.items() if k not in ("trace", "gains")} for t in trackers],
        final_outcomes=final, final_admissible=sum(r["admissible"] for r in final),
        native_reference_pass=bool(error is None and not close_errors and len(final)==32 and
            sum(r["admissible"] for r in final)>=c.final_min_capture_and_local and
            accounting["requested_controls"] <= c.max_controls and accounting["unknown_controls"] == 0 and
            budget_totals["requested_controls"] == accounting["requested_controls"]),
        accounting=accounting, archive_accounting=dict(native=native_accounting, heterogeneous=heterogeneous_accounting),
        shared_budget=budget_totals, budget_matches_paid_ledger=bool(
            budget_totals["requested_controls"] == accounting["requested_controls"]),
        actor_inference_rows=0, adam_steps=0,
        scipy_work_counts=callback_counts,
        native_controls_are_physics_frames=True, rk4_internal_stages_are_not_extra_control_slots=True,
        wall_seconds=time.perf_counter()-started,
        scientific_scope="Finite tested native trajectory-reference feasibility; not MPPI/GPS/standalone actor or an efficiency claim",
        teacher_promotion_performed=False)
    write_json(out/"reference_result.json", result)
    return result
