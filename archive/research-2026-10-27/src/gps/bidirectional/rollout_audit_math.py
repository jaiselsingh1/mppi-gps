"""Independent control-rollout evidence arithmetic, NumPy/stdlib only.

No producer, actor, task, engine, SciPy or optimizer is imported. The recurrence
below differentiates supplied saved arrays; it never integrates dynamics.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


PHASE_PERIOD = 2.356877965498118
CALLBACKS = ("objective", "gradient", "inequality", "jacobian", "chain", "iteration")


def require(condition, message):
    if not condition:
        raise ValueError(message)


class SensitivityFailure(ValueError):
    """Known finite-difference/chain qualification failure, not native work."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fixed_config():
    return dict(T=800, dt=.01, max_controls=4281648, original_seed=191000,
        max_unique_points=200, max_jacobian_calls=100, maxiter=100, phase_period=PHASE_PERIOD,
        direction_epsilon=1e-5, direction_bound_margin=2e-5, sensitivity_boundaries=[1, 64, 144, 288, 800],
        sensitivity_scales=[1., 1., 5., 5.], sensitivity_relative_tolerance=.01,
        qualification_atol=2e-10, effort_coefficient=1e-4, Q=[10., 10., 1., 1.], R=.1,
        selection_seeds=list(range(208000, 208008)), final_seeds=list(range(209000, 209032)),
        final_min_admissible=29)


def fixed_bound():
    return dict(native_parity=48, unique_model_points=4*200*800, fresh_jacobians=4*100*8800,
        sensitivity_central_rollouts=4*4*2*800, actual_openloop_replays=4*2*800,
        actual_tracker_derivatives=4*8800, tracker_nominals=4*800, selection_live=4*8*800,
        final_live=32*800, total=4281648)


def controls(value):
    u = np.asarray(value, dtype=np.float64)
    require(u.shape == (800,) and np.isfinite(u).all() and np.all(abs(u) <= 1.),
        "Finite bounded native 800-control vector required")
    return np.array(u, copy=True, order="C")


def phase_controls(sign):
    require(type(sign) is int and sign in (-1, 1), "One fixed phase sign required")
    return sign*np.sin(2*np.pi*(.01*np.arange(800, dtype=np.float64))/PHASE_PERIOD)


def control_key(snapshot, source_fingerprint, value):
    """Bind exact float64 action bytes and the complete original snapshot."""
    fields = {"state", "warmstart", "elapsed", "hold_count", "task_hash", "context_id", "warmstart_exact"}
    require(isinstance(snapshot, dict) and set(snapshot) == fields, "Full native cache snapshot fields differ")
    state, warm = np.asarray(snapshot["state"]), np.asarray(snapshot["warmstart"])
    require(state.shape == (5,) and warm.shape == (2,) and np.isfinite(state).all() and np.isfinite(warm).all()
        and type(snapshot["elapsed"]) is int and 0 <= snapshot["elapsed"] <= 800
        and abs(float(state[0])-.01*snapshot["elapsed"]) <= 1e-12
        and type(snapshot["hold_count"]) is int and 0 <= snapshot["hold_count"] <= snapshot["elapsed"]
        and type(snapshot["warmstart_exact"]) is bool
        and all(isinstance(snapshot[name], str) and snapshot[name] for name in ("task_hash", "context_id"))
        and isinstance(source_fingerprint, str) and source_fingerprint, "Invalid native cache snapshot/source")
    header = canonical(dict(snapshot=snapshot, source_fingerprint=source_fingerprint,
        controller_kind="native-control-rollout-reference-v1", dtype="float64", shape=[800])).encode()
    return hashlib.sha256(header+b"\0"+controls(value).tobytes(order="C")).hexdigest()


def fixed_directions(value):
    u = controls(value)
    t = np.arange(800, dtype=float)
    mask = abs(u) <= 1.-2e-5
    directions = np.stack((np.ones(800), np.where((t.astype(int)//50) % 2 == 0, 1., -1.),
        np.sin(2*np.pi*t/800), np.cos(2*np.pi*t/200)))*mask[None, :]
    if not np.all(np.any(directions != 0., axis=1)):
        raise SensitivityFailure("All four masked tangent directions must be nonzero")
    require(np.all(abs(u[None, :]+1e-5*directions) <= 1.)
        and np.all(abs(u[None, :]-1e-5*directions) <= 1.), "Central sensitivity controls would require clipping")
    return directions, mask


def distance_squared(states):
    x = np.asarray(states, dtype=np.float64)
    require(x.ndim >= 1 and x.shape[-1] == 4 and np.isfinite(x).all(), "Finite physical four-state array required")
    shoulder, elbow = x[..., 0], x[..., 1]
    horizontal = np.sin(shoulder)+np.sin(shoulder+elbow)
    vertical = np.cos(shoulder)+np.cos(shoulder+elbow)-2.
    return horizontal*horizontal+vertical*vertical


def distance_gradient(states):
    x = np.asarray(states, dtype=np.float64)
    distance_squared(x)
    shoulder, elbow = x[..., 0], x[..., 1]
    horizontal = np.sin(shoulder)+np.sin(shoulder+elbow)
    vertical = np.cos(shoulder)+np.cos(shoulder+elbow)-2.
    result = np.zeros_like(x)
    result[..., 0] = 2*horizontal*(np.cos(shoulder)+np.cos(shoulder+elbow))-2*vertical*(np.sin(shoulder)+np.sin(shoulder+elbow))
    result[..., 1] = 2*horizontal*np.cos(shoulder+elbow)-2*vertical*np.sin(shoulder+elbow)
    return result


def hard_inequalities(states):
    x = np.asarray(states, dtype=np.float64)
    require(x.shape == (801, 4) and np.isfinite(x).all(), "Full 801-boundary native physical trajectory required")
    return np.r_[distance_squared(x[701:])/.2**2-1.,
        np.sum(x[701:, 2:]**2, axis=1)/.5**2-1., distance_squared(x[600])/.5**2-1.]


def chained_constraints(states, A, B, boundaries=(1, 64, 144, 288, 800)):
    """Differentiate saved native maps; initial-state sensitivity is zero."""
    x, a, b = np.asarray(states, dtype=float), np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    require(x.shape == (801, 4) and a.shape == (800, 4, 4) and b.shape == (800, 4, 1)
        and all(np.isfinite(value).all() for value in (x, a, b)), "Full finite native trace/Jacobian arrays required")
    boundaries = tuple(boundaries)
    require(boundaries and all(type(t) is int and 1 <= t <= 800 for t in boundaries)
        and tuple(sorted(set(boundaries))) == boundaries, "Unique chronological sensitivity boundaries required")
    sensitivity = np.zeros((4, 800))
    jacobian = np.zeros((201, 800))
    saved, maximum = [], 0.
    for t in range(800):
        with np.errstate(over="ignore", invalid="ignore"):
            sensitivity = a[t]@sensitivity
            sensitivity[:, t] += b[t, :, 0]
        if not np.isfinite(sensitivity).all():
            raise SensitivityFailure("Nonfinite chained sensitivities at boundary "+str(t+1))
        maximum = max(maximum, float(abs(sensitivity).max()))
        if t+1 in boundaries:
            saved.append(sensitivity.copy())
        if t+1 >= 701:
            row = t-700
            jacobian[row] = (distance_gradient(x[t+1])/.2**2)@sensitivity
            velocity_gradient = np.r_[np.zeros(2), 2*x[t+1, 2:]/.5**2]
            jacobian[100+row] = velocity_gradient@sensitivity
        if t+1 == 600:
            jacobian[200] = (distance_gradient(x[t+1])/.5**2)@sensitivity
    if not np.isfinite(jacobian).all():
        raise SensitivityFailure("Nonfinite analytic endpoint constraint Jacobian")
    return jacobian, np.stack(saved), maximum


def scaled_relative_error(measured, predicted, scales=1.):
    scales = np.asarray(scales, dtype=float)
    require(np.isfinite(scales).all() and np.all(scales > 0), "Positive finite sensitivity scales required")
    measured, predicted = np.asarray(measured, dtype=float), np.asarray(predicted, dtype=float)
    require(measured.shape == predicted.shape and np.isfinite(measured).all() and np.isfinite(predicted).all(),
        "Finite matching measured/predicted sensitivity arrays required")
    a, b = measured/scales, predicted/scales
    absolute = abs(a-b)
    return absolute, absolute/(1.+np.maximum(abs(a), abs(b)))


def objective_terms(value):
    u = controls(value)
    return float(.5e-4*(u@u)), 1e-4*u


def callback_closure(rows):
    """Strict accepted-callback linkage; cap refusals have no accepted intent."""
    allowed = {"objective": ("complete",), "gradient": ("complete",),
        "inequality": ("complete", "limit"), "jacobian": ("complete", "numeric_failure", "limit"),
        "chain": ("complete", "numeric_failure"), "iteration": ("complete",)}
    requests, resolved, ids = {}, set(), set()
    requested = {name: 0 for name in CALLBACKS}
    completed = {name: 0 for name in CALLBACKS}
    failures = {name: 0 for name in CALLBACKS}
    limits = {name: 0 for name in CALLBACKS}
    rejected = 0
    for row in rows:
        identifier, kind = row["id"], row["kind"]
        require(isinstance(identifier, str) and identifier and identifier not in ids, "Duplicate/invalid callback ledger ID")
        ids.add(identifier)
        if kind == "rollout_jacobian_rejected":
            require("request_id" not in row and type(row["attempted_callback"]) is int
                and row["attempted_callback"] == row["maximum_requests"] == 100
                and row["reason"] == "declared_cap_before_native_work" and row["new_native_controls"] == 0,
                "Jacobian refusal did not precede native work at its declared cap")
            rejected += 1
            continue
        family = next((name for name in CALLBACKS if kind.startswith("rollout_"+name+"_")), None)
        if family is None:
            continue
        suffix = kind[len("rollout_"+family+"_"):]
        if suffix == "requested":
            requests[identifier] = family
            requested[family] += 1
            continue
        require(suffix in allowed[family], "Undeclared callback resolution kind: "+kind)
        request = row.get("request_id")
        require(request in requests and requests[request] == family and request not in resolved,
            "Callback resolution is unknown, reversed, wrong-family or duplicated")
        resolved.add(request)
        target = completed if suffix == "complete" else failures if suffix == "numeric_failure" else limits
        target[family] += 1
    return dict(pending=[key for key in requests if key not in resolved], requested=requested,
        completed=completed, numeric_failures=failures, limits=limits, rejected_jacobians=rejected)


def pending_callbacks(rows):
    return callback_closure(rows)["pending"]
