"""Native Walker execution, exact snapshots and durable, compact work evidence.

No model or policy is constructed here. MuJoCo is imported only for snapshots;
all integration is delegated to the instrumented native environment methods.
One HDF5 file holds arrays, while an append-only JSONL ledger records intent,
completion and unknown partial work before and after every external boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class BudgetExhausted(RuntimeError):
    """A predeclared reservation would exceed the run's finite budget."""


class EvidenceError(RuntimeError):
    """Evidence cannot be persisted; continuing would lose ownership/accounting."""


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _dumps(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json)


def _amount(value):
    if type(value) is not int or value < 0:
        raise ValueError("Work amount must be an explicit nonnegative integer")
    return value


class Meter:
    def __init__(self, out: Path, max_controls=350000000, max_neural_rows=150000000):
        import h5py

        self.out = self.path = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.max_controls, self.max_neural_rows = _amount(max_controls), _amount(max_neural_rows)
        self.ledger_path, self.raw_path = self.out / "ledger.jsonl", self.out / "raw.h5"
        if self.ledger_path.exists() or self.raw_path.exists():
            raise FileExistsError("Runtime evidence is create-only")
        self._raw = h5py.File(self.raw_path, "x")
        self._ledger = self.ledger_path.open("x", encoding="utf-8")
        self._lock = threading.RLock()
        self._sequence = self._array_sequence = self._query_sequence = 0
        self._started = time.perf_counter()
        self.closed, self.context = False, {}
        self._counts = dict(
            requested_controls=0,
            completed_controls=0,
            requested_model_controls=0,
            completed_model_controls=0,
            requested_live_controls=0,
            completed_live_controls=0,
            requested_physics_substeps=0,
            completed_physics_substeps=0,
            unknown_partial_controls=0,
            unknown_partial_physics_substeps=0,
            model_queries=0,
            live_queries=0,
            failed_queries=0,
            rejected_queries=0,
            measurement_requests=0,
            failed_measurements=0,
            rejected_measurements=0,
            requested_neural_rows=0,
            completed_neural_rows=0,
            unknown_partial_neural_rows=0,
            requested_backward_rows=0,
            completed_backward_rows=0,
            unknown_partial_backward_rows=0,
            requested_optimizer_steps=0,
            completed_optimizer_steps=0,
            unknown_partial_optimizer_steps=0,
            resets=0,
            reset_errors=0,
            restores=0,
            snapshots=0,
            episodes=0,
        )
        self.record(
            "runtime_open",
            schema="mppi-gps-runtime-v1",
            max_controls=max_controls,
            max_neural_rows=max_neural_rows,
            raw_file="raw.h5",
        )

    def _open(self):
        if self.closed:
            raise EvidenceError("Meter is closed; no further external call is allowed")

    def _fields(self, metadata):
        fields = dict(self.context)
        fields.update(metadata)
        fields["context"] = dict(self.context)
        return fields

    def record(self, event, **fields):
        with self._lock:
            self._open()
            if (
                not isinstance(event, str)
                or not event
                or set(fields) & {"id", "event", "kind", "wall_seconds"}
            ):
                raise ValueError("Explicit event and nonreserved metadata keys required")
            identifier = f"event-{self._sequence:09d}"
            row = dict(
                fields,
                id=identifier,
                event=event,
                kind=event,
                wall_seconds=time.perf_counter() - self._started,
            )
            try:
                self._ledger.write(_dumps(row) + "\n")
                self._ledger.flush()
                os.fsync(self._ledger.fileno())
            except (OSError, TypeError, ValueError) as error:
                raise EvidenceError("Cannot persist runtime ledger") from error
            self._sequence += 1
            return identifier

    def save_arrays(self, tag, **arrays):
        """Create a unique HDF5 group; repeated human tags never overwrite data."""
        with self._lock:
            self._open()
            if not isinstance(tag, str) or not tag:
                raise ValueError("An array-group tag is required")
            label = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:100]
            name = f"arrays/a{self._array_sequence:09d}_{label}"
            self._array_sequence += 1
            group = self._raw.create_group(name)
            group.attrs["tag"] = tag
            refs = {}
            try:
                for key, value in arrays.items():
                    if not isinstance(key, str) or not key or "/" in key:
                        raise ValueError("Array names must be simple nonempty strings")
                    encoding = "array"
                    if isinstance(value, dict):
                        value = np.array(_dumps(value))
                        encoding = "json"
                    array = np.asarray(value)
                    if array.dtype.hasobject:
                        raise ValueError("Object/pickle arrays are prohibited")
                    packed = np.char.encode(array, "utf-8") if array.dtype.kind == "U" else array
                    options = (
                        dict(chunks=True, compression="gzip", compression_opts=1, shuffle=True)
                        if array.ndim and array.size
                        else {}
                    )
                    dataset = group.create_dataset(key, data=packed, **options)
                    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                    dataset.attrs["numpy_dtype"] = array.dtype.str
                    dataset.attrs["encoding"] = encoding
                    dataset.attrs["sha256"] = digest
                    refs[key] = dict(
                        sha256=digest,
                        shape=list(array.shape),
                        dtype=array.dtype.str,
                        encoding=encoding,
                    )
                self._raw.flush()
                handle = self._raw.id.get_vfd_handle()
                if isinstance(handle, int):
                    os.fsync(handle)
                else:
                    raise EvidenceError("HDF5 driver does not expose a durable file handle")
            except (OSError, TypeError, ValueError) as error:
                raise EvidenceError("Cannot persist HDF5 arrays; partial group retained") from error
            return dict(file="raw.h5", group=name, tag=tag, arrays=refs)

    def measure(self, kind, amount, fn, **metadata):
        """Charge a callable; GPU callers synchronize inside fn for kernel timing."""
        self._open()
        amount = _amount(amount)
        if (
            not isinstance(kind, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", kind)
            or not callable(fn)
        ):
            raise ValueError("Named work kind and callable required")
        requested, completed, unknown = (
            prefix + kind for prefix in ("requested_", "completed_", "unknown_partial_")
        )
        for key in (requested, completed, unknown):
            self._counts.setdefault(key, 0)
        fields = self._fields(metadata)
        if kind == "neural_rows" and self._counts[requested] + amount > self.max_neural_rows:
            self._counts["rejected_measurements"] += 1
            self.record("measurement_rejected", measure_kind=kind, amount=amount, **fields)
            raise BudgetExhausted("Neural-row reservation exhausted before callable")
        self._counts[requested] += amount
        self._counts["measurement_requests"] += 1
        request = self.record("measurement_requested", measure_kind=kind, amount=amount, **fields)
        started = time.perf_counter()
        try:
            result = fn()
        except BaseException as error:
            self._counts[unknown] += amount
            self._counts["failed_measurements"] += 1
            self.record(
                "measurement_error",
                request_id=request,
                measure_kind=kind,
                amount=amount,
                completed_amount=0,
                partial_work_unknown=True,
                error_type=type(error).__name__,
                error=str(error),
                call_wall_seconds=time.perf_counter() - started,
                **fields,
            )
            raise
        self._counts[completed] += amount
        self.record(
            "measurement_complete",
            request_id=request,
            measure_kind=kind,
            amount=amount,
            call_wall_seconds=time.perf_counter() - started,
            **fields,
        )
        return result

    def _query(self, mode, controls, frame_skip, inputs, fn, pack):
        self._open()
        controls = _amount(controls)
        fields = self._fields({})
        if self._counts["requested_controls"] + controls > self.max_controls:
            self._counts["rejected_queries"] += 1
            self.record(
                "query_rejected",
                mode=mode,
                attempted_controls=controls,
                requested_controls=0,
                requested_physics_substeps=0,
                **fields,
            )
            raise BudgetExhausted("Physical control reservation exhausted before native call")
        query = f"q{self._query_sequence:09d}"
        self._query_sequence += 1
        raw = self.save_arrays(query + "_inputs", **inputs)
        substeps = controls * frame_skip
        self._counts["requested_controls"] += controls
        self._counts["requested_" + mode + "_controls"] += controls
        self._counts["requested_physics_substeps"] += substeps
        self._counts[mode + "_queries"] += 1
        request = self.record(
            "query_requested",
            query_id=query,
            mode=mode,
            inputs=raw,
            requested_controls=controls,
            requested_physics_substeps=substeps,
            frame_skip=frame_skip,
            **fields,
        )
        started, returned = time.perf_counter(), False
        try:
            result = fn()
            returned = True
            output = self.save_arrays(query + "_result", **pack(result))
        except BaseException as error:
            self._counts["failed_queries"] += 1
            if returned:
                self._complete_controls(mode, controls, substeps)
            else:
                self._counts["unknown_partial_controls"] += controls
                self._counts["unknown_partial_physics_substeps"] += substeps
            self.record(
                "query_error",
                request_id=request,
                query_id=query,
                mode=mode,
                completed_controls=controls if returned else 0,
                completed_physics_substeps=substeps if returned else 0,
                partial_work_unknown=not returned,
                unknown_partial_controls=0 if returned else controls,
                error_type=type(error).__name__,
                error=str(error),
                call_wall_seconds=time.perf_counter() - started,
                **fields,
            )
            raise
        self._complete_controls(mode, controls, substeps)
        self.record(
            "query_complete",
            request_id=request,
            query_id=query,
            mode=mode,
            result=output,
            completed_controls=controls,
            completed_physics_substeps=substeps,
            call_wall_seconds=time.perf_counter() - started,
            **fields,
        )
        return result

    def _complete_controls(self, mode, controls, substeps):
        self._counts["completed_controls"] += controls
        self._counts["completed_" + mode + "_controls"] += controls
        self._counts["completed_physics_substeps"] += substeps

    def _reject_query(self, mode, reason, error_type=ValueError):
        """Persist a rejected native boundary before any physical integration."""
        self._open()
        self._counts["rejected_queries"] += 1
        self.record(
            "query_rejected",
            mode=mode,
            reason=reason,
            error_type=error_type.__name__,
            requested_controls=0,
            requested_physics_substeps=0,
            **self._fields({}),
        )
        raise error_type(reason)

    def instrument(self, env):
        self._open()
        previous = getattr(env, "_mppi_gps_meter", None)
        if previous is self:
            return env
        if previous is not None and not previous.closed:
            raise EvidenceError("A live environment cannot have two work owners")
        native_batch = getattr(env, "_mppi_gps_native_batch", env.batch_rollout)
        native_step = getattr(env, "_mppi_gps_native_step", env.step)
        env._mppi_gps_native_batch, env._mppi_gps_native_step = native_batch, native_step
        frame_skip = int(env._frame_skip)
        if frame_skip < 1:
            raise ValueError("Positive native frame_skip required")

        def batch(initial_state, actions, initial_warmstart=None):
            try:
                actions = np.asarray(actions, dtype=float)
            except (TypeError, ValueError):
                self._reject_query("model", "Native action tape must be a numeric array")
            if (
                actions.ndim != 3
                or min(actions.shape[:2]) < 1
                or actions.shape[2] != env.action_dim
            ):
                self._reject_query("model", "Native action tape must have shape(K,H,action_dim)")
            if not np.isfinite(actions).all():
                self._reject_query("model", "Finite native action tapes required")
            before = snapshot(env)
            if np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied):
                self._reject_query(
                    "model",
                    "Native batch state contract excludes external-force interventions",
                    EvidenceError,
                )
            inputs = dict(
                initial_state=np.asarray(initial_state),
                actions=actions,
                initial_warmstart=np.zeros(env.model.nv)
                if initial_warmstart is None
                else np.asarray(initial_warmstart),
                live_integration_before=before["integration"],
            )

            def pack(value):
                actual = snapshot(env)
                if not all(
                    np.array_equal(actual[k], before[k])
                    for k in ("state", "warmstart", "integration")
                ):
                    raise EvidenceError("Native model query mutated the live integration state")
                states, scores, sensors = value
                return dict(
                    states=states,
                    raw_scores=scores,
                    sensordata=sensors,
                    terminal_role=np.array("native_planning_horizon_score"),
                )

            return self._query(
                "model",
                int(np.prod(actions.shape[:2])),
                frame_skip,
                inputs,
                lambda: native_batch(initial_state, actions, initial_warmstart=initial_warmstart),
                pack,
            )

        def step(action):
            try:
                action = np.asarray(action, dtype=float)
            except (TypeError, ValueError):
                self._reject_query("live", "Native live action must be a numeric array")
            if action.shape != (env.action_dim,) or not np.isfinite(action).all():
                self._reject_query(
                    "live", "Native live action must be finite with shape(action_dim)"
                )
            before = snapshot(env)

            def pack(value):
                obs, cost, done, info = value
                after = snapshot(env)
                return dict(
                    observation=obs,
                    cost=np.array(cost),
                    done=np.array(done),
                    state=after["state"],
                    warmstart=after["warmstart"],
                    integration=after["integration"],
                    info=info,
                )

            return self._query(
                "live",
                1,
                frame_skip,
                dict(action=action, **before),
                lambda: native_step(action),
                pack,
            )

        env.batch_rollout, env.step, env._mppi_gps_meter = batch, step, self
        self.record(
            "environment_instrumented",
            frame_skip=frame_skip,
            action_dim=env.action_dim,
            state_dim=env.state_dim,
            native_physics_only=True,
        )
        return env

    def totals(self):
        result = dict(self._counts)
        result.update(
            neural_rows=result["completed_neural_rows"],
            backward_rows=result["completed_backward_rows"],
            optimizer_steps=result["completed_optimizer_steps"],
            queries=result["model_queries"] + result["live_queries"],
            physics_frames=result["completed_physics_substeps"],
            wall_seconds=time.perf_counter() - self._started,
        )
        return result

    def close(self):
        if not self.closed:
            self.record("runtime_close", totals=self.totals())
            self._raw.flush()
            self._raw.close()
            self._ledger.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def instrument(env, meter):
    return meter.instrument(env)


def snapshot(env):
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, integration, spec)
    meter = getattr(env, "_mppi_gps_meter", None)
    if meter is not None and not meter.closed:
        meter._counts["snapshots"] += 1
    return dict(
        state=env.get_state().copy(),
        warmstart=env.get_warmstart().copy(),
        integration=integration,
        spec=np.array(int(spec)),
        dimensions=np.array([env.model.nq, env.model.nv, env.model.nu]),
    )


def restore(env, value):
    import mujoco

    meter = getattr(env, "_mppi_gps_meter", None)
    if meter is not None:
        meter._open()
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    dimensions = [env.model.nq, env.model.nv, env.model.nu]
    if (
        int(value["spec"]) != int(spec)
        or not np.array_equal(value["dimensions"], dimensions)
        or np.shape(value["integration"]) != (mujoco.mj_stateSize(env.model, spec),)
        or not all(np.isfinite(value[k]).all() for k in ("state", "warmstart", "integration"))
    ):
        raise ValueError("Snapshot does not match this finite native model")
    mujoco.mj_setState(env.model, env.data, value["integration"], spec)
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_setState(env.model, env.data, value["integration"], spec)
    env.set_warmstart(value["warmstart"])
    actual = snapshot(env)
    if not all(np.array_equal(actual[k], value[k]) for k in ("state", "warmstart", "integration")):
        raise ValueError("Native integration state/warmstart did not restore exactly")
    if meter is not None:
        meter._counts["restores"] += 1
        meter.record("snapshot_restored", integration_steps=0, forward_calls=1, **meter._fields({}))
    return actual


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    warmstarts: np.ndarray
    integrations: np.ndarray
    costs: np.ndarray
    velocities: np.ndarray
    cost: float
    survived: bool
    success: bool
    mae: float
    steps: int
    tag: str
    seed: int
    complete: bool
    terminal_cost: float
    terminal_adjustment: float
    absorbing_cost: float
    raw: dict
    infos: tuple


def _complete_cost(costs, terminal_already_charged, native_terminal, requested_steps, done):
    """Native step costs already include fall terminal when that option is on."""
    terminal_adjustment = 0.0 if terminal_already_charged else float(native_terminal)
    absorbing = 5.0 * (requested_steps - len(costs)) if done else 0.0
    return float(np.sum(costs) + terminal_adjustment + absorbing), terminal_adjustment, absorbing


def execute_episode(env, controller, seed, steps, meter, tag):
    if (
        type(seed) is not int
        or seed < 0
        or type(steps) is not int
        or steps < 1
        or not callable(controller)
    ):
        raise ValueError("Explicit reset seed, positive horizon and controller callback required")
    meter.instrument(env)
    meter.context = dict(meter.context, episode=tag, step=0)
    meter.record("reset_requested", seed=seed, tag=tag, integration_steps=0, **meter._fields({}))
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        obs = env.reset()
    except BaseException as error:
        meter._counts["reset_errors"] += 1
        meter.record("reset_error", seed=seed, tag=tag, error=str(error), **meter._fields({}))
        raise
    finally:
        np.random.set_state(state)
    meter._counts["resets"] += 1
    initial = snapshot(env)
    states, warms, integrations = (
        [initial["state"]],
        [initial["warmstart"]],
        [initial["integration"]],
    )
    observations, actions, costs, velocities, infos = [], [], [], [], []
    meter.record("reset_complete", seed=seed, tag=tag, integration_steps=0, **meter._fields({}))
    done = False
    try:
        for t in range(steps):
            meter.context = dict(meter.context, episode=tag, step=t)
            observations.append(np.asarray(obs).copy())
            before = snapshot(env)
            action, controller_info = controller(
                before["state"].copy(), before["warmstart"].copy(), np.asarray(obs).copy(), t
            )
            actual = snapshot(env)
            if not all(
                np.array_equal(actual[k], before[k]) for k in ("state", "warmstart", "integration")
            ):
                raise EvidenceError(
                    "Controller did not restore live state before executing its action"
                )
            obs, cost, done, info = env.step(np.asarray(action))
            current = snapshot(env)
            actions.append(np.asarray(action).copy())
            costs.append(float(cost))
            states.append(current["state"])
            warms.append(current["warmstart"])
            integrations.append(current["integration"])
            velocities.append(float(info["x_velocity"]))
            infos.append(dict(controller=controller_info, native=info))
            if len(actions) % 100 == 0 or done:
                progress = dict(
                    tag=str(tag), steps=len(actions), done=bool(done), work=meter.totals()
                )
                meter.record("episode_progress", **progress)
                print(_dumps(dict(event="episode_progress", **progress)), flush=True)
            if done:
                break
        terminal_charged = bool(done and getattr(env, "_apply_terminal_cost_on_done", False))
        terminal = (
            float(infos[-1]["native"].get("fall_cost", 0.0))
            if terminal_charged
            else float(env.terminal_cost(env.get_state()[None])[0])
        )
        total, adjustment, absorbing = _complete_cost(
            costs, terminal_charged, terminal, steps, done
        )
        survived = len(actions) == steps and not done
        mae = float(np.mean(np.abs(np.asarray(velocities) - env._v_target)))
        raw = meter.save_arrays(
            "episode_" + tag,
            observations=np.asarray(observations),
            actions=np.asarray(actions),
            states=np.asarray(states),
            warmstarts=np.asarray(warms),
            integrations=np.asarray(integrations),
            costs=np.asarray(costs),
            velocities=np.asarray(velocities),
            infos={"rows": infos},
            terminal_cost=np.array(terminal),
            terminal_adjustment=np.array(adjustment),
            absorbing_cost=np.array(absorbing),
        )
        episode = Episode(
            np.asarray(observations),
            np.asarray(actions),
            np.asarray(states),
            np.asarray(warms),
            np.asarray(integrations),
            np.asarray(costs),
            np.asarray(velocities),
            total,
            bool(survived),
            bool(survived and mae <= 0.3),
            mae,
            len(actions),
            str(tag),
            seed,
            bool(done or len(actions) == steps),
            terminal,
            adjustment,
            absorbing,
            raw,
            tuple(infos),
        )
        meter._counts["episodes"] += 1
        summary = {
            k: getattr(episode, k)
            for k in (
                "cost",
                "survived",
                "success",
                "mae",
                "steps",
                "tag",
                "seed",
                "complete",
                "terminal_cost",
                "terminal_adjustment",
                "absorbing_cost",
                "raw",
            )
        }
        meter.record("episode_complete", **summary)
        return episode
    except BaseException as error:
        raw = meter.save_arrays(
            "episode_" + tag + "_partial",
            observations=np.asarray(observations),
            actions=np.asarray(actions),
            states=np.asarray(states),
            warmstarts=np.asarray(warms),
            integrations=np.asarray(integrations),
            costs=np.asarray(costs),
        )
        meter.record(
            "episode_error",
            tag=tag,
            seed=seed,
            completed_steps=len(actions),
            partial=raw,
            complete=False,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
