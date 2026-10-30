"""Durable accounting for future actor-only Walker queries; no physics imports.

One journal is one query, created exclusively and never resumed. Every live
control is charged and fsynced before calling the wrapped environment. A failed
or interrupted MuJoCo call has unknown partial execution and retains its one
requested control slot. This wrapper does not establish actor ownership or
account for calls made directly through an exposed underlying environment.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np


class ControlBudgetExceeded(RuntimeError):
    pass


def _json(value):
    """Copy JSON-compatible metadata and arrays, rejecting nonfinite values."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("JSON metadata keys must be strings")
            return {key: convert(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(val) for val in item]
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        raise ValueError("Journal metadata must contain finite JSON values")
    return json.loads(json.dumps(convert(value), allow_nan=False))


def _vector(value, size, name):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return array


def _snapshot(env):
    return {"state": _vector(env.walker.get_state(), 19, "state").tolist(),
            "warmstart": _vector(env.walker.get_warmstart(), 9, "warmstart").tolist()}


def _task(task):
    task = _json(task)
    if not isinstance(task, dict) or task.get("cost_style") != "target_velocity":
        raise ValueError("Journal requires the explicit target_velocity task")
    for key in ("target_velocity", "vel_cost_weight", "vertical_velocity_cost_weight",
                "angle_cost_weight", "unhealthy_cost_weight", "ctrl_cost_weight",
                "terminal_unhealthy_weight"):
        value = task.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid task field {key}")
    if type(task.get("apply_terminal_cost_on_done")) is not bool:
        raise ValueError("Explicit apply_terminal_cost_on_done required")
    return task


def _components(state, action, info, terminated, task):
    """Saved-state arithmetic only; do not integrate or alter task/reward data."""
    healthy = .8 < state[2] < 2. and -1. < state[3] < 1.
    if terminated != (not healthy):
        raise ValueError("Termination differs from saved-state Walker health")
    fall = task["terminal_unhealthy_weight"] if not healthy and task["apply_terminal_cost_on_done"] else 0.
    parts = dict(velocity=task["vel_cost_weight"] * abs(task["target_velocity"]-state[10]),
        vertical_velocity=task["vertical_velocity_cost_weight"] * state[11]**2,
        angle=task["angle_cost_weight"] * state[3]**2,
        unhealthy=task["unhealthy_cost_weight"] * (not healthy),
        control=task["ctrl_cost_weight"] * float(np.square(action).sum()), fall=fall)
    if "fall_cost" in info and not math.isclose(info["fall_cost"], fall, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Recorded fall cost differs from the one falling-transition charge")
    raw = info.get("task_cost")
    if type(raw) not in (int, float) or not math.isfinite(raw):
        raise ValueError("info.task_cost must preserve the finite raw task cost")
    if not math.isclose(raw, sum(parts.values()), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Raw task cost differs from saved-state cost components")
    return parts


def journal_stats(path):
    """Read complete durable lines; an incomplete final line is reported.

    This is inspection only, never resume. Requested slots are charged even
    without a completion/error record. A corrupt middle line fails inspection.
    """
    rows, incomplete = [], False
    lines = Path(path).read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines)-1 or line.endswith(b"\n"):
                raise ValueError("Corrupt durable journal line")
            incomplete = True
    attempts = {row["id"] for row in rows if row["kind"] == "control_attempt"}
    completed = {row["attempt_id"] for row in rows if row["kind"] == "control_complete"}
    errors = {row["attempt_id"] for row in rows if row["kind"] == "control_error"}
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Journal record IDs must be unique")
    if any(row.get("controls") != int(row["kind"] == "control_attempt") for row in rows):
        raise ValueError("Only a control attempt charges one control slot")
    if completed & errors or not (completed | errors) <= attempts:
        raise ValueError("Journal has inconsistent attempt references")
    return dict(charged_requested_controls=len(attempts), completed_controls=len(completed),
        error_controls=len(errors), inflight_or_interrupted_controls=len(attempts-completed-errors),
        confirmed_returned_controls=len(completed)+sum(row["kind"] == "control_error" and row["call_returned"] for row in rows),
        unknown_partial_error_controls=sum(row["kind"] == "control_error" and not row["call_returned"] for row in rows),
        rejected_before_call=sum(row["kind"] == "control_rejected" for row in rows),
        query_ended=any(row["kind"] == "query_end" for row in rows),
        incomplete_final_line=incomplete, records=len(rows),
        controls=sum(row.get("controls", 0) for row in rows))


class QueryJournal:
    def __init__(self, path, *, query_id, owner_hash, context_id, ancestry,
                 budget_controls, task_config, metadata=None):
        for name, value in (("query_id", query_id), ("owner_hash", owner_hash), ("context_id", context_id)):
            if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
                raise ValueError(f"{name} must be a nonempty printable string")
        if type(budget_controls) is not int or budget_controls < 0:
            raise ValueError("budget_controls must be a nonnegative integer")
        if not isinstance(ancestry, Mapping):
            raise ValueError("ancestry must be an explicit mapping")
        payload = _json(dict(owner_hash=owner_hash, context_id=context_id,
                             ancestry=ancestry, metadata={} if metadata is None else metadata))
        self._task = _task(task_config)
        self.path, self.query_id = Path(path), query_id
        self.budget_controls = budget_controls
        self._sequence = self._charged = 0
        self._active, self._failed, self._ended = None, False, False
        self._started = time.perf_counter()
        self._file = self.path.open("x", encoding="utf-8")
        self._emit("query_begin", budget_controls=budget_controls,
                   task_config=self._task, **payload)

    @property
    def task_config(self):
        return deepcopy(self._task)

    @property
    def stats(self):
        return journal_stats(self.path)

    def _emit(self, kind, *, controls=0, **payload):
        if self._file.closed or self._ended:
            raise RuntimeError("Journal is closed or ended")
        identifier = f"{self.query_id}:{self._sequence}"
        row = _json(dict(id=identifier, query_id=self.query_id, kind=kind, controls=controls,
            utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-self._started,
            **payload))
        self._file.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._sequence += 1
        return identifier

    def before_control(self, pre, action):
        if self._active is not None or self._failed:
            raise RuntimeError("Previous attempt is unresolved or failed")
        pre = {"state": _vector(pre["state"], 19, "state"),
               "warmstart": _vector(pre["warmstart"], 9, "warmstart")}
        action = _vector(action, 6, "action")
        if np.any(abs(action) > 1.):
            raise ValueError("Action must be within native Walker bounds")
        if self._charged >= self.budget_controls:
            self._emit("control_rejected", reason="control_budget_exhausted")
            raise ControlBudgetExceeded("Control budget exhausted before environment call")
        identifier = self._emit("control_attempt", controls=1, pre=pre, action=action)
        self._charged += 1
        self._active = identifier
        return identifier

    def complete_control(self, attempt_id, **payload):
        if attempt_id != self._active:
            raise RuntimeError("Completion must name the current attempt")
        self._emit("control_complete", attempt_id=attempt_id, **payload)
        self._active = None

    def error_control(self, attempt_id, error, *, call_returned, evidence=None):
        if attempt_id != self._active:
            raise RuntimeError("Error must name the current attempt")
        self._emit("control_error", attempt_id=attempt_id, error_type=type(error).__name__,
                   error=str(error), call_returned=call_returned, evidence=evidence,
                   partial_physics_execution_unknown=not call_returned)
        self._active, self._failed = None, True

    def finish(self, *, status="complete", metadata=None):
        if not isinstance(status, str) or not status.strip():
            raise ValueError("A query-end status is required")
        if status == "complete" and (self._failed or self._active is not None):
            raise ValueError("A failed or unresolved query cannot finish as complete")
        self._emit("query_end", status=status, metadata={} if metadata is None else metadata,
                   accounting=self.stats)
        self._ended = True

    def close(self):
        if not self._file.closed:
            if not self._ended:
                self.finish(status="closed_without_result")
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if not self._ended:
            self.finish(status="error" if error else "closed_without_result",
                        metadata={} if error is None else {"error": str(error)})
        self.close()


class JournaledEnv:
    """Proxy injected WalkerRL; only ``step`` performs charged interception.

    ``close`` closes the environment, not the journal: the caller can then append
    the finite-return result and analytic absorbing-slot metadata with finish().
    No actor, planner, objective conversion, automatic resume or reset is added.
    """
    def __init__(self, env, journal):
        if not isinstance(journal, QueryJournal):
            raise TypeError("An explicit QueryJournal is required")
        if _task(env.task) != journal.task_config:
            raise ValueError("Environment task differs from frozen journal task")
        self._env, self.journal = env, journal

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        try:
            action = _vector(action, 6, "action")
            if np.any(abs(action) > 1.):
                raise ValueError("Action must be within native Walker bounds")
            pre = _snapshot(self._env)
        except Exception as error:
            self.journal._emit("control_rejected", reason=str(error))
            raise
        attempt = self.journal.before_control(pre, action)
        started, returned, evidence = time.perf_counter(), False, None
        try:
            result = self._env.step(action.copy())
            returned = True
            obs, reward, terminated, truncated, info = result
            evidence = {"post": _snapshot(self._env)}
            evidence.update(_json(dict(observation=_vector(obs, 17, "observation"), reward=reward,
                terminated=terminated, cutoff=truncated, info=info)))
            if type(evidence["terminated"]) is not bool or type(evidence["cutoff"]) is not bool:
                raise ValueError("Environment termination/cutoff must be booleans")
            parts = _components(evidence["post"]["state"], action, evidence["info"], terminated,
                                self.journal._task)
            self.journal.complete_control(attempt, **evidence, raw_task_cost=info["task_cost"],
                cost_components=parts, call_wall_seconds=time.perf_counter()-started)
            return result
        except BaseException as error:
            self.journal.error_control(attempt, error, call_returned=returned, evidence=evidence)
            raise

    def close(self):
        return self._env.close()
