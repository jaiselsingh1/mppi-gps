"""Independent saved-only review of Acrobot's locked upright and basin joins.

This reviewer imports no simulator, planner or worker mathematics. It derives
the local native-map Jacobian, Riccati checks, deployed torques, height/capture
outcomes and adaptive MPPI weights from recorded arrays. Finite basin probes
remain empirical evidence, not a nonlinear invariant-set certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

ATOL = 1e-10
ROUNDING = 64 * np.finfo(np.float64).eps
MODEL_SHA = "324c8a6bf3bfea0b7faf72395dd3ac7a75cc9c2047159d1f1cd212aeee203cbe"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def errors(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    require(states.ndim >= 1 and states.shape[-1] == 5, "Full Acrobot state required")
    require(np.isfinite(states).all(), "Nonfinite state")
    value = states[..., 1:].copy()
    value[..., :2] = (value[..., :2] + np.pi) % (2 * np.pi) - np.pi
    return value


def values(states: np.ndarray, p: np.ndarray) -> np.ndarray:
    e = errors(states)
    return np.einsum("...i,ij,...j->...", e, p, e)


def costs(states: np.ndarray, actions: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """POST-state dense/energy running costs and one endpoint terminal value."""
    states, actions = np.asarray(states), np.asarray(actions)
    q0, q1, v0, v1 = (states[..., j] for j in range(1, 5))
    x, z = np.sin(q0) + np.sin(q0 + q1), 2 + np.cos(q0) + np.cos(q0 + q1)
    d2 = x * x + (z - 4) ** 2
    if kind == "dense":
        return d2 + 0.01 * (v0 * v0 + v1 * v1 + (actions * actions).sum(-1)), np.zeros_like(z)
    require(kind == "energy", "Unknown frozen objective")
    potential = 9.81 * (4 + 1.5 * np.cos(q0) + 0.5 * np.cos(q0 + q1))
    lower, cross = 0.3453036571964956, 0.5 * np.cos(q1)
    kinetic = 0.5 * (
        (1.6908739696964956 + 2 * cross) * v0 * v0 + 2 * (lower + cross) * v0 * v1 + lower * v1 * v1
    )
    gate = np.exp(2 * np.log(0.1) * (np.maximum(np.sqrt(d2) - 0.2, 0) / 2) ** 2)
    pose = 1 - np.clip(z / 4, 0, 1) + 0.05 * x * x
    speed = (v0 / 2.5) ** 2 + (v1 / 5) ** 2
    excess = ((np.maximum(np.abs(states[..., 3:5]) - 8, 0) / 8) ** 2).sum(-1)
    running = pose + (160 / 9) * (1 - gate) * ((potential + kinetic - 58.86) / 39.24) ** 2
    return running + 0.15 * gate * speed + 0.05 * excess, 100 * pose + 25 * gate * speed


def trace_measurements(states: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    states, actions = np.asarray(states), np.asarray(actions)
    n = len(actions)
    require(states.shape == (n + 1, 5) and actions.shape == (n, 1), "Trace needs N+1 states")
    require(np.isfinite(states).all() and np.isfinite(actions).all(), "Invalid trace arrays")
    require(np.max(np.abs(actions), initial=0) <= 1, "Unbounded native actuator control")
    require(np.allclose(np.diff(states[:, 0]), 0.02, rtol=0, atol=ATOL), "Native control timing")
    post = states[1:]
    height = 2 + np.cos(post[:, 1]) + np.cos(post[:, 1] + post[:, 2])
    e = errors(post)
    enough = n >= 100
    final = e[-100:]
    high = bool(enough and np.all(height[-100:] >= 3.8))
    angle = float(np.max(np.abs(final[:, :2]), initial=0))
    speed = float(np.max(np.abs(final[:, 2:]), initial=0))
    return dict(
        controls=n,
        height_success=high,
        capture=bool(high and angle <= 0.02 and speed <= 0.05),
        final_maximum_wrapped_angle=angle,
        final_maximum_velocity=speed,
        maximum_height=float(np.max(height, initial=0)),
        final_minimum_height=None if not n else float(np.min(height[-100:])),
    )


def native_jacobians(states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> dict:
    """Independently decode the frozen equilibrium/FD/held-out row ordering."""
    states, actions, next_states = map(np.asarray, (states, actions, next_states))
    require(states.shape == next_states.shape == (37, 5), "All37 native linearization rows")
    require(actions.shape == (37, 1), "All37 native probe controls")
    require(np.isfinite(next_states).all(), "Finite native probe outputs")
    require(np.allclose(next_states[:, 0] - states[:, 0], 0.02, rtol=0, atol=ATOL), "Probe time")
    require(np.array_equal(states[0], np.zeros(5)) and actions[0, 0] == 0, "Upright reference")
    require(
        np.max(np.abs(next_states[0, 1:]), initial=0) <= 1e-12, "Upright is a native fixed point"
    )
    result = []
    for start, h in ((1, 1e-6), (11, 5e-7)):
        jac = np.empty((4, 5))
        for j in range(5):
            for sign, offset in ((1, 0), (-1, 1)):
                row = start + 2 * j + offset
                state, action = np.zeros(5), np.zeros(1)
                if j < 4:
                    state[j + 1] = sign * h
                else:
                    action[0] = sign * h
                require(np.array_equal(states[row], state), "FD perturbation row/state binding")
                require(np.array_equal(actions[row], action), "FD perturbation row/action binding")
            jac[:, j] = (next_states[start + 2 * j, 1:] - next_states[start + 2 * j + 1, 1:]) / (
                2 * h
            )
        result.append(jac)
    require(np.max(np.abs(result[0] - result[1])) <= 1e-7, "Two-step FD agreement")
    a, b = result[0][:, :4], result[0][:, 4:]
    prediction = states[21:, 1:] @ a.T + actions[21:] @ b.T
    prediction_error = float(np.max(np.abs(prediction - next_states[21:, 1:])))
    require(prediction_error <= 1e-7, "Held-out local native-map prediction")
    return dict(
        A=a,
        B=b,
        finite_difference_error=float(np.max(np.abs(result[0] - result[1]))),
        held_out_error=prediction_error,
    )


def analytic_map() -> tuple[np.ndarray, np.ndarray]:
    """Independent mechanical linearization and two held-input native RK4 steps."""
    m = np.array(
        [[2.6908739696964956, 0.8453036571964956], [0.8453036571964956, 0.3453036571964956]]
    )
    gravity = 9.81 * np.array([[2.0, 0.5], [0.5, 0.5]])
    a = np.zeros((4, 4))
    a[:2, 2:] = np.eye(2)
    a[2:, :2], a[2:, 2:] = np.linalg.solve(m, gravity), np.linalg.solve(m, -0.05 * np.eye(2))
    b = np.r_[np.zeros((2, 1)), np.linalg.solve(m, [[0.0], [2.0]])]
    augmented = np.zeros((5, 5))
    augmented[:4, :4], augmented[:4, 4:] = a, b
    z = 0.01 * augmented
    step = (
        np.eye(5)
        + z
        + np.linalg.matrix_power(z, 2) / 2
        + np.linalg.matrix_power(z, 3) / 6
        + np.linalg.matrix_power(z, 4) / 24
    )
    two = step @ step
    return two[:4, :4], two[:4, 4:]


def riccati_measurements(packet: dict, native: dict) -> dict:
    a, b, q, r, p, k = (
        np.asarray(packet[name], dtype=float) for name in ("A", "B", "Q", "R", "P", "K")
    )
    require(
        a.shape == p.shape == q.shape == (4, 4)
        and b.shape == (4, 1)
        and k.shape == (1, 4)
        and r.shape == (1, 1),
        "LQR packet shapes",
    )
    require(all(np.isfinite(x).all() for x in (a, b, q, r, p, k)), "Finite LQR packet")
    require(
        np.max(np.abs(a - native["A"])) <= ATOL and np.max(np.abs(b - native["B"])) <= ATOL,
        "DARE uses first native FD map",
    )
    expected_q = np.diag([0.0, 0.0, 0.01, 0.01])
    expected_q[:2, :2] = [[4, 2], [2, 1]]
    require(
        np.array_equal(q, expected_q) and np.array_equal(r, [[0.01]]),
        "Unchanged dense-cost quadratic",
    )
    require(
        np.allclose(p, p.T, rtol=0, atol=ATOL) and np.linalg.eigvalsh(p).min() > 0,
        "Positive symmetric Riccati value",
    )
    expected_k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    require(np.allclose(k, expected_k, rtol=0, atol=ATOL), "Riccati feedback gain")
    residual = p - (q + a.T @ p @ (a - b @ k))
    relative = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(p)))
    eigenvalues = np.linalg.eigvals(a - b @ k)
    radius = float(np.max(np.abs(eigenvalues)))
    require(relative <= 1e-8 and radius < 1, "Local discrete LQR stability")
    aa, bb = analytic_map()
    analytic_error = float(max(np.max(np.abs(a - aa)), np.max(np.abs(b - bb))))
    require(analytic_error <= 1e-7, "Independent mechanical RK4 Jacobian")
    unsaturated = float(1 / (k @ np.linalg.solve(p, k.T))[0, 0])
    require(
        abs(float(packet["unsaturated_rho"]) - unsaturated) <= ATOL + ROUNDING * abs(unsaturated),
        "Exact input-bound ellipsoid",
    )
    return dict(
        spectral_radius=radius,
        dare_relative_residual=relative,
        analytic_native_error=analytic_error,
        unsaturated_rho=unsaturated,
    )


def selected_shell(
    rows: list[dict], unsaturated_rho: float, factors: Any = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0)
) -> float | None:
    selected = None
    for factor in factors:
        found = [row for row in rows if row.get("kind") == "shell" and row.get("factor") == factor]
        require(len(found) == 24, "Every declared shell needs24 preserved directions")
        if not all(row["capture"] for row in found):
            break
        selected = factor * unsaturated_rho
    return selected


def weight_measurements(
    scores: np.ndarray, weights: np.ndarray, temperature: float, target: float = 32
) -> dict:
    scores, weights = np.asarray(scores), np.asarray(weights)
    require(
        scores.ndim == 1 and weights.shape == scores.shape and len(scores) > 1,
        "Candidate bank shape",
    )
    require(np.isfinite(scores).all() and np.isfinite(weights).all(), "Finite candidate bank")
    require(np.isfinite(temperature) and temperature > 0, "Finite positive adaptive temperature")
    expected = np.exp(-(scores - np.min(scores)) / temperature)
    expected /= expected.sum()
    require(
        np.allclose(weights, expected, rtol=0, atol=ATOL), "Gibbs weights use saved native costs"
    )
    ess = float(1 / np.sum(weights * weights))
    spread = float(np.ptp(scores))
    if spread > 0 and np.sum(scores == scores.min()) < target:
        require(abs(ess - target) <= 1e-6, "Frozen adaptive ESS target")
    return dict(
        ess=ess,
        temperature=temperature,
        cost_std=float(np.std(scores)),
        cost_range=spread,
        within_aim=bool(10 <= ess <= 50),
    )


class Review:
    def __init__(self, base: Path, cpu_seconds: float, previous: Path | None = None):
        self.base, self.limit, self.started = base, cpu_seconds, time.process_time()
        self.previous = previous
        self.checks = 0
        self.inputs: dict[str, dict] = {}

    def check(self, value: Any, label: str) -> None:
        require(time.process_time() - self.started <= self.limit, "Saved-review CPU cap")
        require(value, label)
        self.checks += 1

    def same(self, left: Any, right: Any, label: str, *, exact: bool = False, atol: float = ATOL):
        a, b = np.asarray(left), np.asarray(right)
        self.check(a.shape == b.shape, label + " shape")
        self.check(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.check(
            np.array_equal(a, b)
            if exact
            else np.all(np.abs(a - b) <= atol + ROUNDING * np.maximum(1, np.abs(b))),
            label,
        )

    def consume(self, path: Path) -> Path:
        if str(path) not in self.inputs:
            self.inputs[str(path)] = dict(bytes=path.stat().st_size, sha256=digest(path))
        return path

    def load(self, path: Path) -> dict:
        return json.loads(self.consume(path).read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def check_trace(
    review: Review, group: Any, kind: str, gain: np.ndarray | None = None
) -> tuple[dict, np.ndarray, np.ndarray]:
    states, actions = group["states"][:], group["actions"][:]
    measurement = trace_measurements(states, actions)
    n = len(actions)
    for key in ("integration", "warmstarts"):
        x = group[key][:]
        review.check(len(x) == n + 1 and np.isfinite(x).all(), "Complete boundary context: " + key)
    review.same(
        group["pre_action_observations"][:], states[:-1, 1:], "PRE-action observations", exact=True
    )
    running, terminal = costs(states[1:], actions, kind)
    review.same(group["costs"][:], running, "Independent POST-state running costs")
    if gain is not None:
        raw = -errors(states[:-1]) @ gain.T
        raw_key = "raw_actions" if "raw_actions" in group else "unclipped_actions"
        review.same(group[raw_key][:], raw, "Deployed LQR raw torque")
        review.same(actions, np.clip(raw, -1, 1), "Deployed clipped LQR torque")
    measurement.update(
        running_cost=float(running.sum()), terminal_cost=float(terminal[-1]) if n else 0.0
    )
    return measurement, states, actions


def check_sources(review: Review, run: Path, config_path: Path, cfg: dict) -> dict:
    result = review.load(run / "result.json")
    review.check(review.load(run / "config.json") == cfg, "Exact frozen stage configuration")
    inputs, environment = review.load(run / "inputs.json"), review.load(run / "environment.json")
    review.check(inputs["config_sha256"] == digest(config_path), "Configuration content binding")
    manifest_path = review.base / "source/source_manifest.json"
    manifest = review.load(manifest_path)
    seal = inputs["frozen_source_manifest"]
    review.check(seal["sha256"] == digest(manifest_path), "Immutable worker source seal")
    records = manifest.get("files", manifest.get("payloads", []))
    if isinstance(records, dict):
        records = [dict(path=n, **v) for n, v in records.items()]
    review.check(bool(records), "Nonempty frozen worker source")
    for record in records:
        name = record.get("path", record.get("relative_path"))
        path = review.base / "source" / name
        review.check(
            path.resolve().is_relative_to((review.base / "source").resolve()),
            "Contained source path",
        )
        review.check(digest(path) == record["sha256"], "Frozen source payload: " + name)
    native_root = PurePosixPath(seal["path"]).parent
    external = {
        b["path"]: b["sha256"] for b in cfg["bindings"] if PurePosixPath(b["path"]).is_absolute()
    }
    for entry in inputs["files"]:
        path_native = PurePosixPath(entry["path"])
        if entry["path"] in external:
            require(
                review.previous is not None, "Pass --previous to verify frozen historical H5 inputs"
            )
            require(
                path_native.parent.name in {"near-upright", "near-upright-completion", "energy"}
                and path_native.name == "evidence.h5",
                "Declared historical input layout",
            )
            path = review.previous / path_native.parent.name / path_native.name
            review.check(
                entry["sha256"] == external[entry["path"]],
                "External input agrees with frozen binding",
            )
        else:
            name = path_native.relative_to(native_root)
            path = review.base / "source" / str(name)
        review.check(
            digest(review.consume(path)) == entry["sha256"],
            "Native worker input file: " + entry["path"],
        )
    review.check(environment["versions"] == cfg["expected_environment"], "Pinned native runtime")
    review.check(all(x == "1" for x in environment["math_threads"].values()), "Pinned math threads")
    for name, binding in list(result.get("evidence", {}).items()) + list(
        result.get("additional_evidence", {}).items()
    ):
        path = run / name
        review.check(
            path.stat().st_size == binding["bytes"] and digest(path) == binding["sha256"],
            "Closed evidence: " + name,
        )
    review.check(
        not any(result.get("unknown_partial_work", {"missing": 1}).values()),
        "No unknown partial physics",
    )
    for key in ("network_forwards", "backward_calls", "optimizer_updates"):
        review.check(result["work"][key] == 0, "No unauthorized learning: " + key)
    return result


def check_work(review: Review, result: dict, expected: dict, limits: dict) -> None:
    for name, count in expected.items():
        for ending in ("attempted", "completed"):
            review.check(
                result["work"][name + "_" + ending] == count, "Independent work count: " + name
            )
    for name in ("cpu_seconds", "wall_seconds", "output_bytes", "rss_bytes"):
        review.check(result["resources"][name] <= limits[name], "Frozen resource cap: " + name)
    review.check(expected["physics_substeps"] <= limits["physics_substeps"], "Frozen physics cap")


def lqr_roster(cfg: dict, packet: dict, rho: float | None = None) -> list[dict]:
    """Regenerate finite inputs using independent elementary coordinate transforms."""
    p = np.asarray(packet["P"])
    unsaturated = packet["unsaturated_rho"]
    directions = np.r_[
        np.eye(4), -np.eye(4), np.asarray(list(itertools.product((-1.0, 1.0), repeat=4))) / 2
    ]
    upper = np.linalg.cholesky(p).T

    def states_at(level, directions):
        return np.c_[
            np.zeros(len(directions)), np.linalg.solve(upper, (np.sqrt(level) * directions).T).T
        ]

    rows = [
        dict(id=f"historical_{i}", kind="historical", state=x["initial_state"])
        for i, x in enumerate(cfg["historical_starts"])
    ]
    rows.append(dict(id="equilibrium", kind="equilibrium", state=[0.0] * 5))
    for factor in cfg["shell_factors"]:
        for i, state in enumerate(states_at(factor * unsaturated, directions)):
            rows.append(
                dict(id=f"shell_{factor:g}_{i}", kind="shell", factor=factor, state=state.tolist())
            )
    for j in range(4):
        for magnitude in cfg["axis_angles"] if j < 2 else cfg["axis_velocities"]:
            for sign in (-1, 1):
                state = np.zeros(5)
                state[j + 1] = sign * magnitude
                rows.append(
                    dict(id=f"axis_{j}_{sign * magnitude:g}", kind="marginal", state=state.tolist())
                )
    for i, state in enumerate(cfg["grid_states"]):
        rows.append(dict(id=f"joint_grid_{i}", kind="joint_grid", state=state))
    if rho is not None:
        for kind, seed in (
            ("confirm_boundary", cfg["confirmation_boundary_seed"]),
            ("confirm_interior", cfg["confirmation_interior_seed"]),
        ):
            direction = np.random.Generator(np.random.PCG64(seed)).normal(size=(64, 4))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            generated = states_at(rho, direction)
            if kind == "confirm_interior":
                radii = (
                    np.random.Generator(np.random.PCG64(cfg["confirmation_radius_seed"])).uniform(
                        size=64
                    )
                    ** 0.25
                )
                generated[:, 1:] *= radii[:, None]
            rows += [
                dict(id=f"{kind}_{i}", kind=kind, state=x.tolist()) for i, x in enumerate(generated)
            ]
    return rows


def check_lqr(review: Review, cfg: dict, config_path: Path, rows: list[dict]) -> tuple[dict, dict]:
    import h5py

    run = review.base / "lqr"
    result = check_sources(review, run, config_path, cfg)
    packet = review.load(run / "lqr.json")
    review.check(
        result["status"]
        in {
            "VERIFIED_EMPIRICAL_JOIN_REGION",
            "FAILED_BASIN_CONFIRMATION",
            "NO_VERIFIED_LOCAL_BASIN",
        },
        "Closed valid LQR stage",
    )
    require(packet["model_sha256"] == MODEL_SHA == cfg["model_sha256"], "Unchanged native model")
    native_controls = 0
    with h5py.File(review.consume(run / "evidence.h5"), "r") as raw:
        review.check(raw.attrs["status"] == result["status"], "LQR H5 status")
        probes = raw["linearization"]
        native = native_jacobians(
            probes["probe_states"][:], probes["probe_actions"][:], probes["next_states"][:]
        )
        held = np.random.Generator(
            np.random.PCG64(cfg["lqr"]["heldout_linearization_seed"])
        ).normal(0, 1e-5, (16, 5))
        review.same(
            probes["probe_states"][21:, 1:],
            held[:, :4],
            "Private held-out probe states",
            exact=True,
        )
        review.same(
            probes["probe_actions"][21:], held[:, 4:], "Private held-out probe controls", exact=True
        )
        math = riccati_measurements(packet, native)
        rho = packet.get("selected_rho")
        expected = lqr_roster(cfg["lqr"], packet, rho)
        review.check(
            set(raw["trials"]) == {x["id"] for x in expected}, "Complete finite LQR roster"
        )
        recorded = {x["id"]: x for x in result["trials"]}
        for case in expected:
            group = raw["trials"][case["id"]]
            declared = json.loads(group.attrs["case_json"])
            review.check(
                declared["id"] == case["id"] and declared["kind"] == case["kind"],
                "Preserved finite-case identity",
            )
            review.same(declared["state"], case["state"], "Preserved finite-case input", exact=True)
            measurement, states, _ = check_trace(review, group, "dense", np.asarray(packet["K"]))
            review.check(
                group.attrs["status"] == "COMPLETED" and len(states) == 501,
                "Full500-control LQR probe",
            )
            review.same(states[0], case["state"], "Frozen basin input", exact=True)
            if case["kind"] == "historical":
                original = cfg["lqr"]["historical_starts"][int(case["id"].split("_")[-1])][
                    "initial_context"
                ]
                for name, array in original.items():
                    review.same(
                        group[name][0], array, "Historical full initial context", exact=True
                    )
            review.check(
                bool(group.attrs["captured"])
                == measurement["capture"]
                == recorded[case["id"]]["captured"],
                "Independent empirical capture",
            )
            review.check(
                bool(group.attrs["height_success"])
                == measurement["height_success"]
                == recorded[case["id"]]["height_success"],
                "Independent height success",
            )
            rows.append(dict(stage="lqr", **case, **measurement))
            native_controls += measurement["controls"]
        chosen = selected_shell(rows, math["unsaturated_rho"], cfg["lqr"]["shell_factors"])
        review.check(
            (rho is None and chosen is None) or (rho is not None and abs(rho - chosen) <= ATOL),
            "Largest contiguous successful shell",
        )
        confirmed = bool(
            rho is not None and all(x["capture"] for x in rows if x["kind"].startswith("confirm_"))
        )
        review.check(result["join_validation_passed"] == confirmed, "Fresh confirmation gate")
        if rho is not None:
            review.same(
                packet["coordinate_projections"],
                np.sqrt(rho * np.diag(np.linalg.inv(np.asarray(packet["P"])))),
                "Marginal ellipsoid projections, not a joint box",
            )
        replay = raw["replay"]
        original = raw["trials/historical_0"]
        for key in ("states", "warmstarts", "integration"):
            review.same(replay[key][:], original[key][:13], "Native prefix replay: " + key)
            review.check(
                replay[key][0].tobytes() == original[key][0].tobytes(),
                "Byte-exact initial restoration: " + key,
            )
        for key in ("actions", "costs"):
            review.same(replay[key][:], original[key][:12], "Native replay: " + key)
    work = dict(
        native_controls=native_controls,
        verification_controls=49,
        forecast_controls=0,
        model_queries=0,
        physics_substeps=2 * (native_controls + 49),
        restores=1,
    )
    check_work(review, result, work, cfg["lqr"]["limits"])
    return dict(
        status=result["status"],
        math=math,
        join_validation_passed=result["join_validation_passed"],
        selected_rho=packet.get("selected_rho"),
        work=work,
    ), packet


def check_banks(
    review: Review, group: Any, cfg: dict, seed: int, bank_rows: list[dict], identity: dict
) -> tuple[int, int]:
    """Regenerate proposals and weighted updates without a simulator or planner."""
    states, actions = group["states"][:], group["actions"][:]
    k, h = cfg["K"], cfg["H"]
    rng = np.random.Generator(np.random.PCG64(seed))
    previous = np.zeros((h, 1))
    queries = 0
    review.check(not cfg.get("use_is_correction", False), "No importance-cost correction")
    review.check(cfg.get("noise_lowpass_cutoff_hz") is None, "Declared white innovations")
    review.check(cfg["lam_target_neff_fraction"] * k == 32, "Same ESS32 mechanism")
    review.check(len(group["planning"]) == len(actions), "One plan per executed MPPI control")
    chol = np.linalg.cholesky([[cfg["noise_sigma"] ** 2]])
    for t, action in enumerate(actions):
        plan = group[f"planning/control_{t:04d}"]
        review.same(
            plan["pre_action_observation"][:], states[t, 1:], "Planner PRE state", exact=True
        )
        passes = cfg["initial_iterations"] if t == 0 else 1
        review.check(
            len([n for n in plan if n.startswith("pass_")]) == passes, "Complete refinement passes"
        )
        for iteration in range(passes):
            bank = plan[f"pass_{iteration:03d}"]
            before, after = bank["nominal_before"][:], bank["nominal_after"][:]
            review.same(before, previous, "Unshifted refinement/shifted continuation", exact=True)
            scores, weights = bank["candidate_costs"][:], bank["weights"][:]
            require(scores.shape == (k,), "Frozen number of candidates")
            measured = weight_measurements(scores, weights, float(bank.attrs["lam"]))
            review.same(bank.attrs["neff"], measured["ess"], "Independent ESS")
            review.same(
                bank.attrs["candidate_cost_std"], measured["cost_std"], "Native cost spread"
            )
            sampled = np.clip(
                before[None] + np.einsum("khi,ji->khj", rng.standard_normal((k, h, 1)), chol), -1, 1
            )
            action_hash = hashlib.sha256(np.ascontiguousarray(sampled).tobytes()).hexdigest()
            review.check(
                action_hash == str(bank.attrs["action_bank_sha256"]),
                "Exact sampled-action bank binding",
            )
            review.check(
                json.loads(bank.attrs["rng_state_after_json"]) == rng.bit_generator.state,
                "Exact carried private RNG",
            )
            expected = np.clip(
                before + np.einsum("k,kha->ha", weights, sampled - before[None]), -1, 1
            )
            review.same(after, expected, "Independent weighted full-plan update")
            previous = after
            queries += 1
            bank_rows.append(dict(**identity, control=t, refinement=iteration, **measured))
        review.same(action, previous[0], "Executed first optimized control", exact=True)
        previous = np.concatenate([previous[1:], previous[-1:]])
    return queries, queries * k * h


def check_initial_context(review: Review, group: Any, start: dict) -> None:
    review.same(group["states"][0], start["initial_state"], "Frozen restart state", exact=True)
    for key, array in start.get("initial_context", {}).items():
        actual, expected = np.asarray(group[key][0]), np.asarray(array, dtype=float)
        review.check(
            actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
            "Byte-exact restart context: " + key,
        )


def check_handoff(review: Review, prefix: Any, suffix: Any, p: np.ndarray, rho: float) -> dict:
    v = values(prefix["states"][:], p)
    reached = np.flatnonzero(v <= rho)
    review.check(
        len(reached) > 0 and int(reached[0]) == len(prefix["actions"]),
        "Earliest join at final MPPI boundary",
    )
    for key in ("states", "warmstarts", "integration"):
        review.check(
            prefix[key][-1].tobytes() == suffix[key][0].tobytes(),
            "Unchanged handoff context: " + key,
        )
    review.check(not suffix.attrs["reset_at_join"], "No reset at the controller switch")
    return dict(join_control=int(reached[0]), join_V=float(v[-1]), join_rho=rho)


def check_mppi(
    review: Review,
    stage: str,
    cfg: dict,
    config_path: Path,
    packet: dict,
    rows: list[dict],
    banks: list[dict],
) -> dict:
    import h5py

    run = review.base / stage
    spec = cfg["stages"][stage]
    result = check_sources(review, run, config_path, cfg)
    swing = stage != "near"
    p, k = np.asarray(packet["P"]), np.asarray(packet["K"])
    rho = float(packet["rho"])
    valid = (
        {"PASSED_JOIN_GATE", "FAILED_JOIN_GATE"}
        if swing
        else {"COMPLETED_ADAPTIVE_TEMPERATURE_PROBE"}
    )
    review.check(result["status"] in valid, "Closed controller stage: " + stage)
    review.check(
        result["lqr_binding"]["sha256"] == digest(review.base / "lqr/lqr.json"),
        "Bound retained upright controller",
    )
    native = forecast = queries = successes = 0
    with h5py.File(review.consume(run / "evidence.h5"), "r") as raw:
        review.check(raw.attrs["status"] == result["status"], "Controller H5 status")
        review.check(
            len(raw["trials/variant_0"]) == len(spec["starts"]), "Full controller start roster"
        )
        for i, start in enumerate(spec["starts"]):
            group = raw[f"trials/variant_0/trial_{i}"]
            identity = dict(stage=stage, trial_index=i)
            review.check(group.attrs["status"] == "COMPLETED", "Complete MPPI episode/prefix")
            check_initial_context(review, group, start)
            for key in ("reset_seed", "planner_seed"):
                review.check(group.attrs[key] == start[key], "Paired private seed: " + key)
            measured, states, actions = check_trace(review, group, spec["profile"]["kind"])
            n = len(actions)
            native += n
            count, paid = check_banks(
                review, group, spec["profile"]["planner"], start["planner_seed"], banks, identity
            )
            queries += count
            forecast += paid
            row = dict(**identity, **measured, joined=False)
            receipt = result["trials"][i]
            if not swing:
                review.check(n == 500, "Complete near-upright MPPI hold")
                review.check(
                    bool(receipt["success"]) == measured["height_success"],
                    "Near height success separate from LQR",
                )
                baseline = start["baseline"]
                require(review.previous is not None, "Historical bank mirror required")
                oldpath = (
                    review.previous / PurePosixPath(baseline["h5_path"]).parent.name / "evidence.h5"
                )
                review.check(
                    digest(review.consume(oldpath)) == baseline["sha256"], "Frozen old-bank input"
                )
                with h5py.File(oldpath, "r") as old:
                    newbank = group["planning/control_0000/pass_000"]
                    oldbank = old[baseline["group"] + "/planning/control_0000/pass_000"]
                    for key in ("candidate_costs", "nominal_before"):
                        review.same(
                            newbank[key][:],
                            oldbank[key][:],
                            "Paired first candidate bank: " + key,
                            atol=1e-10,
                        )
                row["success"] = measured["height_success"]
            else:
                v = values(states, p)
                joined = bool(np.any(v <= rho))
                row["joined"] = joined
                review.check(v[0] > rho, "Hanging start outside join set")
                review.check(
                    bool(group.attrs["success"]) == joined == receipt["joined"],
                    "MPPI basin-entry outcome",
                )
                if joined:
                    suffix = group["lqr_suffix"]
                    join = check_handoff(review, group, suffix, p, rho)
                    locked, sx, su = check_trace(review, suffix, "energy", k)
                    review.check(
                        locked["controls"] == 200 and suffix.attrs["status"] == "COMPLETED",
                        "Full200-control locked suffix",
                    )
                    review.same(
                        suffix["lyapunov_values"][:], values(sx, p), "Locked-suffix value trace"
                    )
                    native += 200
                    row.update(
                        **join,
                        lqr_height_success=locked["height_success"],
                        success=locked["height_success"],
                        lqr_controls=200,
                    )
                    combined = (
                        measured["running_cost"] + locked["running_cost"] + locked["terminal_cost"]
                    )
                    review.same(receipt["join_V"], join["join_V"], "Recorded join value")
                    review.check(receipt["join_control"] == n, "Recorded first join time")
                else:
                    review.check(
                        n == 600 and "lqr_suffix" not in group,
                        "All failed swing-up attempts preserved",
                    )
                    combined = measured["running_cost"] + measured["terminal_cost"]
                    row.update(success=False, lqr_controls=0)
                review.same(
                    receipt["task_cost"], combined, "One final terminal charge across handoff"
                )
                review.check(
                    bool(group.attrs["teacher_success"]) == row["success"] == receipt["success"],
                    "Hybrid teacher outcome",
                )
                row["hybrid_cost"] = combined
            review.same(
                group.attrs["task_cost"],
                measured["running_cost"] + measured["terminal_cost"],
                "Prefix cost scope",
            )
            successes += int(row["success"])
            rows.append(row)
    work = dict(
        native_controls=native,
        verification_controls=0,
        forecast_controls=forecast,
        model_queries=queries,
        physics_substeps=2 * (native + forecast),
        restores=0,
    )
    check_work(review, result, work, spec["limits"])
    review.check(result["successes"] == successes, "Recomputed full-stage success count")
    if swing:
        required = 2 if stage == "swing-development" else 9
        review.check(
            (result["status"] == "PASSED_JOIN_GATE") == (successes >= required),
            "Frozen teacher competence gate",
        )
    return dict(
        status=result["status"], successes=successes, episodes=len(spec["starts"]), work=work
    )


def render(
    base: Path, config: Path, out: Path, *, previous: Path | None = None, cpu_seconds: float = 120
) -> dict:
    base, config, out = base.resolve(), config.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    review = Review(base, cpu_seconds, previous.resolve() if previous else None)
    result = dict(
        status="STARTING",
        new_physics_controls=0,
        new_network_forwards=0,
        new_fitting_updates=0,
        limitations=[
            "Finite captured inputs and coordinate projections are not a certified nonlinear basin or a captured Cartesian box.",
            "Saved numeric review does not repeat native candidate physics.",
        ],
    )
    rows, banks = [], []
    try:
        require(0 < cpu_seconds <= 120, "Frozen120CPU-second saved-review allocation")
        cfg = review.load(config)
        review.check(
            cfg["protocol"] == "acrobot-transient-attractor" and cfg["version"] == "1.0",
            "Frozen protocol version",
        )
        lqr, packet = check_lqr(review, cfg, config, rows)
        parts = {"lqr": lqr}
        if not lqr["join_validation_passed"]:
            review.check(
                not any((base / x).exists() for x in cfg["stages"]),
                "No controller launches after failed local-attractor gate",
            )
        else:
            for stage in ("near", "swing-development", "swing-qualification"):
                review.check(
                    (base / stage / "result.json").is_file(),
                    "Required authorized stage completed: " + stage,
                )
                parts[stage] = check_mppi(review, stage, cfg, config, packet, rows, banks)
                if stage == "swing-development" and parts[stage]["status"] != "PASSED_JOIN_GATE":
                    review.check(
                        not (base / "swing-qualification").exists(),
                        "No qualification after failed development gate",
                    )
                    break
        totals = {
            key: sum(part["work"][key] for part in parts.values())
            for key in (
                "native_controls",
                "verification_controls",
                "forecast_controls",
                "model_queries",
                "physics_substeps",
            )
        }
        review.check(
            totals["physics_substeps"] <= cfg["shared_limits"]["physics_substeps"],
            "Shared paid physics ceiling",
        )
        write_csv(out / "episodes.csv", rows)
        write_csv(out / "planner_banks.csv", banks)
        result.update(
            status="PASSED",
            parts=parts,
            recomputed_work=totals,
            teacher_qualified=parts.get("swing-qualification", {}).get("status")
            == "PASSED_JOIN_GATE",
            learning_loop_tested=False,
        )
    except Exception as error:
        result.update(status="FAILED_OR_INCOMPLETE", error=f"{type(error).__name__}: {error}")
    result.update(
        check_count=review.checks,
        inputs=review.inputs,
        review_cpu_seconds=time.process_time() - review.started,
    )
    write_json(out / "independent_audit.json", result)
    (out / "report.md").write_text(
        "# Acrobot attractor saved review\n\n"
        + json.dumps(
            {k: v for k, v in result.items() if k not in {"inputs", "limitations"}}, indent=2
        )
        + "\n"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--cpu-seconds", type=float, default=120)
    args = parser.parse_args(argv)
    result = render(
        args.base, args.config, args.out, previous=args.previous, cpu_seconds=args.cpu_seconds
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "error", "check_count", "review_cpu_seconds")
                if k in result
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
