"""Independent saved-evidence audit of the fixed-alpha full-trajectory PI loop.

No producer metric, optimizer, policy, simulator or study module is imported.
The audit decodes checkpoints on CPU without constructing a policy or restoring
RNGs, and evaluates their frozen arrays with the declared NumPy action map.
Saved states are checked algebraically; no dynamics or gradients are repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


class AuditFailure(ValueError):
    """An evidence, identity, numerical or finite-budget contract failed."""


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def typed_fingerprint(value: Any) -> str:
    """Independent implementation of the pre-existing typed tree encoding."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if hasattr(item, "detach"):
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(json.dumps(["array", list(array.shape), str(array.dtype)]).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit((type(key).__name__, key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(
                json.dumps([type(item).__name__, item], default=_plain, allow_nan=False).encode()
            )

    visit(value)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(json.dumps([list(value.shape), str(value.dtype)]).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def tensor_arrays(state: dict) -> dict[str, np.ndarray]:
    return {
        key: (value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value))
        for key, value in state.items()
    }


def policy_hash(state: dict) -> str:
    digest = hashlib.sha256(b"walker-tanh-policy-v1")
    for name, value in sorted(tensor_arrays(state).items()):
        array = np.ascontiguousarray(value)
        digest.update(
            json.dumps([name, list(array.shape), str(array.dtype)], separators=(",", ":")).encode()
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


class Audit:
    """Bound every file and reserve independent head evaluations before work."""

    def __init__(
        self,
        *,
        seconds: float = 7200,
        neural_rows: int = 140_000_000,
        output_bytes: int = 8 * 1024**3,
        memory_bytes: int = 32 * 1024**3,
    ):
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.seconds, self.neural_limit = seconds, neural_rows
        self.output_limit, self.memory_limit = output_bytes, memory_bytes
        self.files: dict[str, dict] = {}
        self.checks = 0
        self.requested_rows = self.completed_rows = self.unknown_rows = 0
        self.checkpoints: dict[str, dict] = {}
        self.native: dict[str, dict] = {}

    def check(self) -> None:
        if time.monotonic() - self.started > self.seconds:
            raise AuditFailure("Independent audit wall budget exhausted")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == "darwin" else 1024
        if peak > self.memory_limit:
            raise AuditFailure("Independent audit process peak RSS exceeds limit")

    def need(self, condition: Any, message: str) -> None:
        self.check()
        self.checks += 1
        if not condition:
            raise AuditFailure(message)

    def close(
        self, actual: Any, expected: Any, message: str, *, atol: float = 1e-9, rtol: float = 0
    ) -> None:
        self.need(
            np.shape(actual) == np.shape(expected)
            and np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True),
            message,
        )

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        self.need(typed_fingerprint(actual) == typed_fingerprint(expected), message)

    def bind(self, path: Path, expected: dict | None = None) -> dict:
        path = path.resolve()
        self.need(path.is_file(), "Missing evidence: " + str(path))
        stat = path.stat()
        previous = self.files.get(str(path))
        if previous is None:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    self.check()
                    digest.update(block)
            value = dict(bytes=stat.st_size, sha256=digest.hexdigest(), mtime_ns=stat.st_mtime_ns)
            self.files[str(path)] = value
        else:
            self.need(
                stat.st_size == previous["bytes"] and stat.st_mtime_ns == previous["mtime_ns"],
                "Evidence changed during audit",
            )
            value = previous
        if expected is not None:
            self.need(
                value["bytes"] == expected["bytes"] and value["sha256"] == expected["sha256"],
                "Binding differs: " + str(path),
            )
        return value

    def bound(self, value: dict) -> Path:
        self.need(
            isinstance(value, dict) and {"path", "bytes", "sha256"} <= value.keys(),
            "Complete external file binding required",
        )
        path = Path(value["path"])
        self.need(path.is_absolute(), "External evidence locations must be explicit")
        self.bind(path, value)
        return path

    def json(self, path: Path) -> dict:
        self.bind(path)
        return json.loads(path.read_text())

    def package(self, path: Path) -> dict:
        self.bind(path)
        import torch

        return torch.load(path, map_location="cpu", weights_only=True)

    def checkpoint(self, path: Path) -> dict:
        key = str(path.resolve())
        if key not in self.checkpoints:
            payload = self.package(path)
            self.need(
                payload.get("schema") == "walker-tanh-policy-v1"
                and payload.get("obs_dim", 17) == 17
                and payload.get("hidden_dim", 128) == 128,
                "Fixed W2 checkpoint schema/architecture",
            )
            shapes = {"observation_mean": (17,), "observation_std": (17,)}
            for layer, outputs, inputs in ((0, 128, 17), (2, 128, 128), (4, 6, 128)):
                shapes[f"network.{layer}.weight"] = (outputs, inputs)
                shapes[f"network.{layer}.bias"] = (outputs,)
            arrays = tensor_arrays(payload["state_dict"])
            self.need(set(arrays) == set(shapes), "Exact W2 tensor roster")
            for name, shape in shapes.items():
                self.need(
                    arrays[name].shape == shape
                    and arrays[name].dtype == np.float32
                    and np.isfinite(arrays[name]).all(),
                    "Finite FP32 fixed tensor: " + name,
                )
            self.need(np.all(arrays["observation_std"] > 0), "Positive inherited normalization")
            self.need(policy_hash(arrays) == payload["policy_hash"], "Checkpoint policy hash")
            self.checkpoints[key] = payload
        self.bind(path)
        return self.checkpoints[key]

    def predict(self, payload: dict, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float32)
        self.need(
            values.ndim == 2 and values.shape[1] == 17 and np.isfinite(values).all(),
            "Finite pre-action 17-dimensional audit inputs",
        )
        count = len(values)
        self.need(
            self.requested_rows + count <= self.neural_limit,
            "Independent NumPy-head row budget exhausted",
        )
        self.requested_rows += count
        state = tensor_arrays(payload["state_dict"])
        pieces = []
        try:
            for begin in range(0, count, 8192):
                self.check()
                value = (values[begin : begin + 8192] - state["observation_mean"]) / state[
                    "observation_std"
                ]
                for layer in (0, 2, 4):
                    value = np.tanh(
                        np.einsum(
                            "bi,oi->bo",
                            value,
                            state[f"network.{layer}.weight"],
                            optimize=False,
                            dtype=np.float32,
                        )
                        + state[f"network.{layer}.bias"]
                    )
                self.need(
                    np.isfinite(value).all() and np.all(np.abs(value) <= 1),
                    "Audit action function is finite and bounded",
                )
                pieces.append(value)
        except BaseException:
            self.unknown_rows += count
            raise
        self.completed_rows += count
        return np.concatenate(pieces) if pieces else np.empty((0, 6), dtype=np.float32)


def observations(states: np.ndarray) -> np.ndarray:
    value = np.asarray(states)
    if value.shape[-1:] != (19,) or not np.isfinite(value).all():
        raise AuditFailure("Finite physical Walker states required for audit observations")
    return np.concatenate((value[..., 2:10], np.clip(value[..., 10:19], -10, 10)), axis=-1)


def healthy(states: np.ndarray) -> np.ndarray:
    return (
        (states[..., 2] > 0.8) & (states[..., 2] < 2) & (states[..., 3] > -1) & (states[..., 3] < 1)
    )


def active_prefix(states: np.ndarray) -> np.ndarray:
    alive = healthy(states)
    mask = np.ones(alive.shape, dtype=bool)
    mask[..., 1:] = np.logical_and.accumulate(alive[..., :-1], axis=-1)
    return mask


def running_cost(post: np.ndarray, actions: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        return (
            2 * np.abs(1.5 - post[..., 10])
            + 0.5 * post[..., 11] ** 2
            + 0.1 * post[..., 3] ** 2
            + 5 * ~healthy(post)
            + 0.001 * np.sum(actions**2, axis=-1)
        )


def batch_cost(post: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if post.ndim != 3 or post.shape[-1] != 19 or actions.shape != (*post.shape[:2], 6):
        raise AuditFailure("Batch evidence requires K,H physical states and controls")
    mask = active_prefix(post)
    valid = np.all((np.isfinite(post).all(-1) & np.isfinite(actions).all(-1)) | ~mask, axis=1)
    cost = (
        np.where(mask, running_cost(post, actions), 0).sum(1)
        + 5 * (~mask).sum(1)
        + 20 * np.any(~healthy(post), axis=1)
    )
    return np.where(valid, cost, np.inf), mask, valid


def reconstruction(
    a: Audit, payload: dict, initial: np.ndarray, post: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, int]:
    _, mask, valid = batch_cost(post, actions)
    use = mask & valid[:, None]
    pre = np.concatenate((np.broadcast_to(initial, (len(post), 1, 19)), post[:, :-1]), axis=1)
    prediction = a.predict(payload, observations(pre[use])).astype(np.float64)
    terms = np.zeros(mask.shape, dtype=np.float64)
    terms[use] = 0.5 * np.sum((actions[use].astype(np.float64) - prediction) ** 2, axis=-1)
    total = terms.sum(axis=1)
    total[~valid] = np.inf
    return total, int(use.sum())


def outcome(a: Audit, raw: dict[str, np.ndarray], horizon: int) -> dict:
    states, actions, costs = raw["states"], raw["actions"], raw["costs"]
    n = len(actions)
    a.need(
        states.shape == (n + 1, 19)
        and actions.shape == (n, 6)
        and costs.shape == (n,)
        and 0 < n <= horizon,
        "Native N+1 trace dimensions",
    )
    a.need(
        np.isfinite(states).all()
        and np.isfinite(actions).all()
        and np.isfinite(costs).all()
        and np.all(np.abs(actions) <= 1),
        "Finite bounded native trace",
    )
    alive = healthy(states[1:])
    a.need(np.all(alive[:-1]), "Trace stops at its first actual fall")
    fell = not bool(alive[-1])
    a.need(n == horizon or fell, "Unexplained missing native controls")
    native_terminal = 20.0 * ~alive
    a.close(
        costs,
        running_cost(states[1:], actions) + native_terminal,
        "Native running cost and one fall terminal",
        atol=1e-10,
    )
    if "native_fall_cost" in raw:
        a.close(raw["native_fall_cost"], native_terminal, "One separately stored native terminal")
    for name, width in (("warmstarts", 9), ("integrations", 91)):
        a.need(
            raw[name].shape == (n + 1, width) and np.isfinite(raw[name]).all(),
            "Complete native boundary contexts: " + name,
        )
    a.close(
        raw["observations"], observations(states), "Saved observations are pre-action map", atol=0
    )
    mae = float(np.mean(np.abs(states[1:, 10] - 1.5)))
    survived = n == horizon and not fell
    tail = float(5 * (horizon - n)) if fell else 0.0
    return dict(
        controls=n,
        requested_controls=horizon,
        fell=fell,
        full_survival=survived,
        walking=bool(horizon == 1000 and survived and mae <= 0.3),
        velocity_mae=mae,
        mean_velocity=float(np.mean(states[1:, 10])),
        raw_native_cost=float(costs.sum()),
        absorbing_tail_cost=tail,
        complete_cost=float(costs.sum()) + tail,
    )


def compare_outcome(a: Audit, saved: dict, recomputed: dict) -> None:
    for key, value in recomputed.items():
        if isinstance(value, (bool, int)):
            a.need(saved.get(key) == value, "Outcome identity: " + key)
        else:
            a.close(saved.get(key), value, "Outcome arithmetic: " + key)


def load_arrays(a: Audit, path: Path) -> dict[str, np.ndarray]:
    a.bind(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as source:
            return {key: source[key].copy() for key in source.files}
    import h5py

    with h5py.File(path, "r") as source:
        return {key: source[key][:] for key in source}


def gibbs(scores: np.ndarray, temperature: float) -> tuple[np.ndarray, float]:
    finite = np.isfinite(scores)
    if not finite.any() or temperature <= 0 or not np.isfinite(temperature):
        raise AuditFailure("A finite bank and positive finite temperature are required")
    unnormalized = np.zeros(len(scores), dtype=float)
    unnormalized[finite] = np.exp(-(scores[finite] - np.min(scores[finite])) / temperature)
    weights = unnormalized / unnormalized.sum()
    return weights, float(1 / np.sum(weights**2))


def temperature(scores: np.ndarray, fraction: float) -> float:
    finite = scores[np.isfinite(scores)]
    if not len(finite) or not 0 < fraction < 1:
        raise AuditFailure("Invalid finite-bank ESS target")
    spread = float(finite.max() - finite.min())
    if spread == 0:
        return 1.0
    lo, hi = np.log(spread * 1e-8), np.log(spread * 1e8)
    for _ in range(60):
        middle = 0.5 * (lo + hi)
        if gibbs(scores, float(np.exp(middle)))[1] < fraction * len(scores):
            lo = middle
        else:
            hi = middle
    return float(np.exp(0.5 * (lo + hi)))


def audit_ledger(
    a: Audit, directory: Path, *, coordinator: bool = False, allow_open: bool = False
) -> dict:
    path = directory / "ledger.jsonl"
    a.bind(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    counts: dict[str, int] = {}
    pending: dict[int, dict] = {}
    imported_workers: set[str] = set()
    antecedents: dict[str, dict] = {}
    prerequisite_counts: dict | None = None
    rejected = failed = 0

    def add(values: dict, prefix: str = "") -> None:
        for key, value in values.items():
            a.need(type(value) is int and value >= 0, "Integer nonnegative ledger charges")
            name = prefix + key
            counts[name] = counts.get(name, 0) + value

    for index, row in enumerate(rows):
        a.need(row["id"] == index, "Unbroken work ledger sequence")
        event = row["event"]
        if event == "requested":
            pending[index] = row
            add(row["charges"], "requested_")
        elif event in ("completed", "failed"):
            request = pending.pop(row["request_id"], None)
            a.need(
                request is not None
                and request["operation"] == row["operation"]
                and request["charges"] == row["charges"],
                "Exactly one matching request resolution",
            )
            add(row["charges"], "completed_" if event == "completed" else "unknown_partial_")
            failed += event == "failed"
        elif event == "rejected":
            rejected += 1
        elif event == "derived_cost_components":
            add({"derived_cost_component_rows": row["rows"]})
        elif coordinator and event == "prerequisite_work_included":
            a.need(prerequisite_counts is None, "P0 prerequisite work is imported once")
            prerequisite_counts = row["counts"]
            add(row["counts"])
        elif coordinator and event == "antecedent_work_included":
            phase = row["phase"]
            a.need(
                phase in ("r0", "rv") and phase not in antecedents,
                "Each previously paid origin/native-validation phase imported exactly once",
            )
            paid = a.json(a.bound(row["receipt"]))
            if phase == "r0":
                a.need(
                    paid["schema"] == "trajectory-pi-replay-result-1"
                    and paid["status"] == "ORIGIN_REPLAY_PASS",
                    "Paid origin result is the strict passing gate",
                )
                actual_counts = paid["work"]
            else:
                a.need(
                    paid["status"] == "COMPLETED",
                    "Paid worker-native validation closed successfully",
                )
                actual_counts = paid["counts"]
            a.equal(
                row["counts"],
                actual_counts,
                "Antecedent counts match their original durable source",
            )
            antecedents[phase] = dict(phase=phase, receipt=row["receipt"], counts=actual_counts)
            add(actual_counts)
        elif coordinator and event in ("worker_returned", "failed_worker_receipt_recovered"):
            a.need(row["path"] not in imported_workers, "No duplicate imported worker accounting")
            imported_workers.add(row["path"])
            actual = a.json(Path(row["path"]) / "receipt.json")
            a.equal(row["receipt"], actual, "Imported worker receipt is the preserved receipt")
            add(actual["counts"])
        elif coordinator and event == "unknown_pending_worker":
            add(row["reserved"], "requested_")
            add(row["reserved"], "unknown_partial_")
    closed = bool(rows) and rows[-1]["event"] == "closed"
    if allow_open and not closed:
        unresolved = {}
        for row in pending.values():
            for unit, amount in row["charges"].items():
                unresolved[unit] = unresolved.get(unit, 0) + amount
        return dict(
            counts=counts,
            receipt=None,
            failed=failed,
            rejected=rejected,
            imported_workers=sorted(imported_workers),
            antecedents=list(antecedents.values()),
            prerequisite_counts=prerequisite_counts,
            status="OPEN_UNCONFIRMED",
            unresolved_request_upper_charges=unresolved,
            unresolved_requests=sorted(pending),
            original_closure_present=False,
        )
    a.need(
        bool(rows) and rows[-1]["event"] == "closed" and not pending,
        "Closed ledger with every entered request resolved",
    )
    receipt = a.json(directory / "receipt.json")
    a.equal(rows[-1]["receipt"], receipt, "Closed receipt identity")
    a.equal(counts, receipt["counts"], "Independent requested/completed/unknown census")
    if receipt.get("result_binding"):
        a.bound(receipt["result_binding"])
    return dict(
        counts=counts,
        receipt=receipt,
        failed=failed,
        rejected=rejected,
        imported_workers=sorted(imported_workers),
        antecedents=list(antecedents.values()),
        prerequisite_counts=prerequisite_counts,
    )


def audit_stream(a: Audit, directory: Path) -> tuple[dict, dict]:
    envelope = a.json(directory / "stream.json")
    a.need(
        envelope.get("schema") == "trajectory-pi-w2-stream-1" and envelope["device"] == "cpu",
        "Complete CPU continuation envelope",
    )
    a.need(set(envelope["files"]) == {"policy.pt", "global_rng.pt"}, "Both stream payloads")
    for name, binding in envelope["files"].items():
        a.bind(directory / name, binding)
    payload = a.checkpoint(directory / "policy.pt")
    globals_saved = a.package(directory / "global_rng.pt")
    a.need(
        payload["policy_hash"] == envelope["policy_hash"]
        and typed_fingerprint(payload["optimizer_state"]) == envelope["optimizer_fingerprint"]
        and typed_fingerprint(globals_saved) == envelope["global_rng_fingerprint"],
        "Actor/Adam/private/global stream fingerprints",
    )
    return payload, envelope


def _context(a: Audit, path: Path, index: int) -> dict[str, np.ndarray]:
    raw = load_arrays(a, path)
    return {name: raw[name][index] for name in ("states", "warmstarts", "integrations")}


def audit_native(a: Audit, directory: Path) -> dict:
    """Recompute a complete paid native job, using saved physics outputs only."""
    key = str(directory.resolve())
    if key in a.native:
        return a.native[key]
    job, result = a.json(directory / "job.json"), a.json(directory / "result.json")
    ledger = audit_ledger(a, directory)
    a.need(
        ledger["receipt"]["status"] == "COMPLETED"
        and not ledger["failed"]
        and not ledger["rejected"],
        "Native job closed successfully without omitted paid attempts",
    )
    checkpoint = a.bound(job["checkpoint_binding"])
    a.need(checkpoint.resolve() == Path(job["checkpoint"]).resolve(), "Job checkpoint binding")
    payload = a.checkpoint(checkpoint)
    a.need(
        payload["policy_hash"] == result["policy_hash"]
        and result["policy_checkpoint"] == job["checkpoint_binding"],
        "Frozen native policy owner",
    )
    a.bind(directory / "trace.h5", result["trace"])
    raw = load_arrays(a, directory / "trace.h5")
    measured = outcome(a, raw, job["horizon"])
    compare_outcome(a, result["outcome"], measured)
    n = measured["controls"]
    initial = {name: raw[name][0] for name in ("states", "warmstarts", "integrations")}
    expected = dict(
        environment_constructions=1, checkpoint_loads=1, numpy_constructions=1, native_controls=n
    )
    entry = dict(
        path=key,
        kind=job["kind"],
        policy_hash=payload["policy_hash"],
        outcome=measured,
        first_context=typed_fingerprint(initial),
        counts=ledger["counts"],
    )
    entry["reset_context_fingerprint"] = typed_fingerprint(
        {name: raw[name][0] for name in ("states", "warmstarts", "integrations", "observations")}
    )
    if job["kind"] == "actor":
        prediction = a.predict(payload, raw["observations"][:-1]).astype(np.float64)
        a.close(
            raw["actions"], prediction, "Every native actor control belongs to frozen actor", atol=0
        )
        expected.update(
            neural_rows=n, resets=1, external_mj_forward_calls=2, physics_substeps=4 * n
        )
        entry["reset_seed"] = job["reset_seed"]
    else:
        a.need(job["kind"] == "solve", "Known native worker kind")
        context_path = a.bound(job["context_binding"])
        a.need(context_path.resolve() == Path(job["context_file"]).resolve(), "Context file owner")
        context = _context(a, context_path, job.get("context_step", 0))
        for name in initial:
            a.equal(initial[name], context[name], "Selected replay starts at exact bound context")
        initial_plan_path = a.bound(job["initial_plan_binding"])
        a.need(
            initial_plan_path.resolve() == Path(job["initial_plan_file"]).resolve(),
            "Initial full-control sequence owner",
        )
        incoming = load_arrays(a, initial_plan_path)["actions"]
        solution = load_arrays(a, directory / "solution.npz")
        last = load_arrays(a, directory / "last_mean.npz")
        iterations, k, h = job["iterations"], job["planner"]["K"], job["horizon"]
        a.need(
            job["planner"]["H"] == h
            and incoming.shape == (h, 6)
            and job["planner"]["use_is_correction"] is False,
            "Fixed full-horizon dimensions and no extra score terms",
        )
        a.need(len(result["curve"]) == iterations, "Complete within-solve curve")
        import h5py

        a.bind(directory / "iterations.h5")
        best, best_score, previous_after = None, math.inf, None
        actor_rows, full_banks, rng_states = 0, 0, []
        with h5py.File(directory / "iterations.h5", "r") as groups:
            a.need(
                list(groups) == [f"iteration_{i:04d}" for i in range(iterations)],
                "Every scheduled refinement retained exactly once",
            )
            for index in range(iterations):
                group = groups[f"iteration_{index:04d}"]
                U, after, post = (
                    group[name][:]
                    for name in ("nominal_before", "nominal_after", "nominal_post_states")
                )
                a.need(
                    U.shape == after.shape == (h, 6)
                    and post.shape == (h, 19)
                    and np.isfinite(U).all()
                    and np.isfinite(after).all()
                    and np.all(np.abs(U) <= 1)
                    and np.all(np.abs(after) <= 1),
                    "Bounded complete control sequences, never shifted",
                )
                a.close(
                    U,
                    np.clip(incoming, -1, 1) if index == 0 else previous_after,
                    "Each refinement starts from the preceding whole-sequence update",
                    atol=0,
                )
                previous_after = after
                C, J, weights = (
                    group[name][:] for name in ("native_costs", "total_objectives", "weights")
                )
                a.need(C.shape == J.shape == weights.shape == (k,), "Complete bank score vectors")
                own_C, nominal_mask, _ = batch_cost(post[None], U[None])
                a.close(C[0], own_C[0], "Every nominal task cost from its saved post states")
                own_D, _ = reconstruction(a, payload, context["states"], post[None], U[None])
                available = bool(group.attrs["nominal_D_available"])
                a.need(available == bool(np.isfinite(own_C[0])), "Nominal D availability")
                if available:
                    a.close(group.attrs["nominal_D"], own_D[0], "Every nominal reconstruction")
                else:
                    a.need(not np.isfinite(group.attrs["nominal_D"]), "Unavailable D not zero")
                packed = np.unpackbits(group["active_prefix_packed"][:], count=k * h).reshape(k, h)
                a.close(
                    group["prefix_lengths"][:],
                    packed.sum(axis=1),
                    "Every bank prefix length matches retained fall mask",
                    atol=0,
                )
                a.close(
                    group["absorbing_tail_cost"][:],
                    5 * (h - packed.sum(axis=1)),
                    "Every bank absorbing-tail charge",
                    atol=0,
                )
                components = (
                    group["prefix_running_cost"][:]
                    + group["fall_terminal_cost"][:]
                    + group["absorbing_tail_cost"][:]
                )
                finite_cost = np.isfinite(C)
                a.close(
                    components[finite_cost],
                    C[finite_cost],
                    "Saved running/terminal/tail decomposition joins all finite native scores",
                )
                a.equal(packed[0].astype(bool), nominal_mask[0], "Nominal executable prefix")
                if job["alpha"]:
                    a.need(
                        bool(group.attrs["population_D_measured"]) and "reconstruction" in group,
                        "Coupled population D retained",
                    )
                    D = group["reconstruction"][:]
                    a.need(D.shape == (k,), "Coupled D vector shape")
                    a.close(D[0], own_D[0], "Coupled nominal D joins population")
                    a.close(J, C + job["alpha"] * D, "Exact declared total objective")
                    actor_rows += int(np.sum(packed.astype(bool) & np.isfinite(D)[:, None]))
                else:
                    a.need(
                        not bool(group.attrs["population_D_measured"])
                        and "reconstruction" not in group,
                        "Independent population D stays unmeasured",
                    )
                    a.close(J, C, "Independent objective excludes reconstruction", atol=0)
                    actor_rows += int(nominal_mask.sum()) if available else 0
                lam = temperature(J, job["planner"]["lam_target_neff_fraction"])
                rho, ess = gibbs(J, lam)
                a.close(
                    group.attrs["lambda"],
                    lam,
                    "Independent adaptive ESS temperature",
                    rtol=1e-12,
                    atol=1e-12,
                )
                a.close(weights, rho, "Per-bank Gibbs weights", atol=2e-13)
                a.close(group.attrs["n_eff"], ess, "Per-bank effective sample size", atol=1e-9)
                a.close(group.attrs["maximum_weight"], rho.max(), "Maximum Gibbs weight")
                a.close(
                    group.attrs["scale"],
                    0.1 ** (index / max(iterations - 1, 1)),
                    "Fixed geometric proposal scale",
                    atol=0,
                )
                a.close(
                    group.attrs["update_rms"],
                    np.sqrt(np.mean((after - U) ** 2)),
                    "Whole-sequence update progress",
                )
                rng_states.append(json.loads(group.attrs["rng_before"]))
                if index == 0:
                    first_rng = np.random.default_rng(
                        20000 + job["planner_seed"]
                    ).bit_generator.state
                    a.equal(rng_states[0], first_rng, "Declared fresh private PI seed")
                if J[0] < best_score:
                    best, best_score = index, float(J[0])
                curve = result["curve"][index]
                for name, value in (
                    ("C", C[0]),
                    ("D", own_D[0] if job["alpha"] else None),
                    ("J", J[0]),
                    ("best_J", best_score),
                ):
                    recorded = curve[name]
                    if value is None or not np.isfinite(value):
                        a.need(recorded is None, "Curve missing/nonfinite score is explicit")
                    else:
                        a.close(recorded, value, "Within-solve curve joins raw scores: " + name)
                a.need(curve["best_iteration"] == best, "Strict incumbent nominal retention")
                if index in job["full_banks"]:
                    bank = load_arrays(a, directory / f"bank_{index:04d}.npz")
                    costs, mask, _ = batch_cost(bank["states"], bank["actions"])
                    a.close(costs, C, "All saved full-bank native task scores")
                    a.equal(mask, packed.astype(bool), "All saved full-bank fall prefixes")
                    a.close(
                        group["prefix_running_cost"][:],
                        np.where(mask, running_cost(bank["states"], bank["actions"]), 0).sum(1),
                        "All saved full-bank running components",
                    )
                    a.close(
                        group["fall_terminal_cost"][:],
                        20 * np.any(~healthy(bank["states"]), axis=1),
                        "All saved full-bank single terminal components",
                        atol=0,
                    )
                    complete = healthy(bank["states"]).all(1) & np.isfinite(bank["states"]).all(
                        (1, 2)
                    )
                    walking = complete & (
                        np.mean(np.abs(bank["states"][..., 10] - 1.5), axis=1) <= 0.3
                    )
                    a.equal(
                        group["complete_candidates"][:],
                        complete,
                        "Saved full-bank complete-survival flags",
                    )
                    a.equal(
                        group["walking_candidates"][:],
                        walking,
                        "Saved full-bank walking flags separately from survival",
                    )
                    best_walking = float(J[walking].min()) if walking.any() else np.nan
                    a.close(
                        group.attrs["best_walking_population_J"],
                        best_walking,
                        "Saved best walking population diagnostic",
                    )
                    a.close(bank["actions"][0], U, "Sample zero is the current nominal", atol=0)
                    a.close(bank["states"][0], post, "Full-bank nominal state indexing", atol=0)
                    update = np.clip(
                        U + np.einsum("k,kha->ha", rho, bank["actions"] - U[None]), -1, 1
                    )
                    a.close(after, update, "Weighted bounded whole-sequence update", atol=2e-12)
                    if job["alpha"]:
                        scores, _ = reconstruction(
                            a, payload, context["states"], bank["states"], bank["actions"]
                        )
                        a.close(D, scores, "All coupled saved full-bank reconstruction scores")
                    full_banks += 1
        a.need(
            best is not None and result["selected_iteration"] == best,
            "Selected teacher is earliest best scored nominal, not final mean",
        )
        with h5py.File(directory / "iterations.h5", "r") as groups:
            chosen = groups[f"iteration_{best:04d}"]
            a.close(
                solution["actions"],
                chosen["nominal_before"][:],
                "Selected nominal controls",
                atol=0,
            )
            a.close(
                solution["batch_post_states"],
                chosen["nominal_post_states"][:],
                "Selected nominal batch states",
                atol=0,
            )
            a.close(
                solution["final_unselected_actions"],
                previous_after,
                "Final mean explicitly unselected",
                atol=0,
            )
        a.close(
            raw["actions"], solution["actions"][:n], "Native selected control execution", atol=0
        )
        a.close(
            raw["states"][1:],
            solution["batch_post_states"][:n],
            "Batch versus selected native state trace",
            atol=1e-10,
        )
        selected_prediction = a.predict(payload, raw["observations"][:-1]).astype(np.float64)
        saved_prediction = load_arrays(a, directory / "selected_predictions.npz")["predictions"]
        a.close(saved_prediction, selected_prediction, "Independent selected predictions", atol=0)
        selected_D = float(0.5 * np.sum((raw["actions"] - selected_prediction) ** 2))
        a.close(result["selected_D"], selected_D, "Selected native reconstruction")
        a.close(
            result["selected_J"],
            measured["complete_cost"] + job["alpha"] * selected_D,
            "Selected native total objective",
        )
        a.close(
            result["selected_task_batch"],
            measured["complete_cost"],
            "Selected batch/native task accounting",
        )
        a.close(result["selected_total_batch"], best_score, "Selected total and retention score")
        a.close(result["selected_J"], best_score, "Batch/native total objective equality")
        final_C, _, _ = batch_cost(last["states"][None], last["actions"][None])
        a.close(
            last["actions"],
            solution["final_unselected_actions"],
            "Final diagnostic control owner",
            atol=0,
        )
        final_D = 0.0
        if job["alpha"]:
            final, rows = reconstruction(
                a, payload, context["states"], last["states"][None], last["actions"][None]
            )
            final_D = float(final[0])
            actor_rows += rows
        final_total = final_C[0] + job["alpha"] * final_D
        final_available = bool(np.isfinite(final_C[0]) and np.isfinite(final_total))
        a.need(
            result["final_mean_score_available"] is final_available,
            "Unselected final-mean diagnostic availability",
        )
        if final_available:
            a.close(
                result["final_mean_task"], final_C[0], "Final unselected native task diagnostic"
            )
            a.close(result["final_mean_total"], final_total, "Final unselected total diagnostic")
        else:
            a.need(
                result["final_mean_task"] is None and result["final_mean_total"] is None,
                "Unavailable final mean stays missing; selected teacher remains eligible",
            )
        model = iterations * k * h + h
        expected.update(
            neural_rows=actor_rows + n,
            restores=2,
            external_mj_forward_calls=2,
            model_queries=iterations + 1,
            pi_passes=iterations,
            model_controls=model,
            physics_substeps=4 * (model + n),
        )
        entry.update(
            selected_D=selected_D,
            selected_J=result["selected_J"],
            selected_first_action=solution["actions"][0].tolist(),
            selected_iteration=best,
            nominal_paths=iterations,
            full_banks=full_banks,
            rng_states=rng_states,
            context_binding=job["context_binding"],
            context_step=job.get("context_step", 0),
            planner_seed=job["planner_seed"],
            alpha=job["alpha"],
        )
        a.equal(
            result["optimizer_counts"],
            dict(
                requested_passes=iterations,
                completed_passes=iterations,
                requested_model_controls=iterations * k * h,
                completed_model_controls=iterations * k * h,
                unknown_partial_model_controls=0,
            ),
            "Optimizer geometry census subordinate to actual native ledger",
        )
    expected_counts = {
        prefix + unit: value
        for unit, value in expected.items()
        for prefix in ("requested_", "completed_")
    }
    if job["kind"] == "solve":
        expected_counts["derived_cost_component_rows"] = iterations * k * h
    a.equal(ledger["counts"], expected_counts, "Exact actual native/PI/actor-row work census")
    for unit, limit in job["limits"].items():
        if unit not in ("wall_seconds", "output_bytes", "memory_bytes"):
            a.need(
                ledger["counts"].get("requested_" + unit, 0) <= limit, "Worker work limit " + unit
            )
    a.need(
        ledger["receipt"]["wall_seconds"] <= job["limits"]["wall_seconds"],
        "Worker finite wall time",
    )
    a.native[key] = entry
    return entry


def training_rows(a: Audit, bank: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inputs, targets = [], []
    a.need(
        len(bank) == 2 and [item["slot"] for item in bank] == [0, 1],
        "Two source-ordered retained training trajectories",
    )
    for item in bank:
        path = a.bound(item["trace_binding"])
        a.need(path.resolve() == Path(item["trace"]).resolve(), "Training trace binding")
        a.bound(item["plan_binding"])
        raw = load_arrays(a, path)
        a.need(
            raw["states"].shape == (1001, 19) and raw["actions"].shape == (1000, 6),
            "Complete walking trajectories only in fitting corpus",
        )
        a.need(
            healthy(raw["states"][1:]).all()
            and np.mean(np.abs(raw["states"][1:, 10] - 1.5)) <= 0.3,
            "Retained teacher still walks",
        )
        obs = observations(raw["states"][:-1])
        recorded = raw["observations"]
        a.close(
            obs,
            recorded[:-1] if len(recorded) == 1001 else recorded,
            "Training labels joined to exact pre-action rows",
            atol=0,
        )
        inputs.append(obs)
        targets.append(raw["actions"])
    return np.concatenate(inputs), np.concatenate(targets)


def teacher_gap(a: Audit, bank: list[dict], payload: dict) -> dict:
    paths = []
    for item in bank:
        raw = load_arrays(a, a.bound(item["trace_binding"]))
        prediction = a.predict(payload, observations(raw["states"][:-1])).astype(np.float64)
        D = float(0.5 * np.sum((raw["actions"].astype(np.float64) - prediction) ** 2))
        paths.append(
            dict(
                slot=item["slot"],
                controls=len(prediction),
                D=D,
                rms=math.sqrt(2 * D / len(prediction)),
                trace_binding=item["trace_binding"],
            )
        )
    return dict(
        policy_hash=payload["policy_hash"],
        paths=paths,
        mean_D=float(np.mean([row["D"] for row in paths])),
        rms=float(np.mean([row["rms"] for row in paths])),
    )


def compare_gap(a: Audit, saved: dict, measured: dict) -> None:
    a.need(
        saved["policy_hash"] == measured["policy_hash"]
        and len(saved["paths"]) == len(measured["paths"]),
        "Gap policy and trajectory roster",
    )
    for actual, expected in zip(saved["paths"], measured["paths"], strict=True):
        for name in ("slot", "controls", "trace_binding"):
            a.need(actual[name] == expected[name], "Trajectory-equal gap owner: " + name)
        for name in ("D", "rms"):
            a.close(actual[name], expected[name], "Teacher-state reconstruction: " + name)
    for name in ("mean_D", "rms"):
        a.close(saved[name], measured[name], "Equal-trajectory aggregation: " + name)


def audit_fit(
    a: Audit, directory: Path, expected_parent: Path, expected_bank: list[dict], updates: int = 100
) -> dict:
    fit = a.json(directory / "fit.json")
    metadata = fit["study"]
    for field, name in (
        ("initial_state", "initial_state.pt"),
        ("data_bindings", "data_bindings.json"),
        ("index_tape", "index_tape.npz"),
        ("index_rng", "index_rng.jsonl"),
    ):
        a.need(
            a.bound(metadata[field]).resolve() == (directory / name).resolve(),
            "Fit proof location and binding: " + field,
        )
    data = a.json(directory / "data_bindings.json")
    a.need(
        Path(data["parent_stream_directory"]).resolve() == expected_parent.resolve()
        and data["bank"] == expected_bank,
        "Fit uses current bank and exact incoming stream",
    )
    parent, parent_envelope = audit_stream(a, expected_parent)
    child_directory = Path(metadata["child_stream_directory"])
    a.need(
        child_directory.resolve() == (directory.parent / "stream").resolve(),
        "Actual child continuation directory",
    )
    child, child_envelope = audit_stream(a, child_directory)
    for name, filename in (("parent_policy", "policy.pt"), ("parent_envelope", "stream.json")):
        a.need(
            a.bound(data[name]).resolve() == (expected_parent / filename).resolve(),
            "Incoming stream proof binding",
        )
    state, continued = parent["optimizer_state"], child["optimizer_state"]
    a.need(
        data["incoming_optimizer_fingerprint"] == typed_fingerprint(state),
        "Incoming Adam/private-PCG identity",
    )
    initial = a.package(directory / "initial_state.pt")
    a.equal(initial["state_dict"], parent["state_dict"], "Actual fit initial tensors match parent")
    a.equal(initial["adam"], state["adam"], "Actual fit initial Adam moments and groups")
    a.equal(initial["rng"], state["rng"], "Actual fit initial private PCG")
    a.equal(
        initial["private_rng_state"],
        parent["private_rng_state"],
        "Actual fit initial policy-private RNG",
    )
    a.equal(
        child["private_rng_state"],
        parent["private_rng_state"],
        "Policy-private RNG unchanged by deterministic fitting",
    )
    a.need(
        child_envelope["global_rng_fingerprint"] == parent_envelope["global_rng_fingerprint"],
        "Actual global RNG continuation retained across fit/save/reload",
    )
    for name in ("observation_mean", "observation_std"):
        a.equal(
            child["state_dict"][name],
            parent["state_dict"][name],
            "Inherited normalization unchanged: " + name,
        )
    a.need(
        fit["schema"] == state["schema"] == continued["schema"] == "walker-adam-proximal-fit-v1"
        and fit["optimizer_continued"] is True
        and fit["resume_fit"] is False
        and fit["failed"] is False,
        "Existing continuation fitter and new acquisition block",
    )
    settings = dict(
        batch_size=256,
        lr=0.0003,
        eta_theta=0.0,
        gradient_clip=0.5,
        prox_reduction="mean",
        anchor_refresh=0,
    )
    a.need(
        fit["fit_config"] == continued["fit_config"] == state["fit_config"] == settings,
        "No architecture/optimizer/loss/noise intervention",
    )
    a.need(
        fit["completed_updates"] == fit["fit_updates"] == continued["fit_updates"] == updates
        and continued["updates"] == state["updates"] + updates == fit["cumulative_updates"],
        "Fixed update budget and cumulative cursor",
    )
    for field, expected in (
        ("parent_policy_hash", parent["policy_hash"]),
        ("child_policy_hash", child["policy_hash"]),
        ("anchor_policy_hash", parent["policy_hash"]),
    ):
        a.need(fit[field] == expected, "Fit hash identity: " + field)
    a.need(
        continued["parent_policy_hash"] == parent["policy_hash"]
        and continued["policy_hash"] == child["policy_hash"]
        and continued["anchor_policy_hash"] == parent["policy_hash"]
        and policy_hash(continued["anchor"]) == parent["policy_hash"],
        "New-corpus anchor and parent-child hash chain",
    )
    a.need(
        metadata["saved_optimizer_fingerprint"]
        == metadata["reloaded_optimizer_fingerprint"]
        == typed_fingerprint(continued),
        "Actual child checkpoint reload proof",
    )
    inputs, targets = training_rows(a, expected_bank)
    fit_data = dict(
        observations_sha256=array_hash(inputs.astype(np.float32)),
        actions_sha256=array_hash(targets.astype(np.float32)),
        rows=len(inputs),
    )
    a.need(
        fit["fit_data"] == continued["fit_data"] == fit_data and len(inputs) == 2000,
        "Exact source-ordered equal-sized FP32 fitting corpus",
    )
    tape = load_arrays(a, directory / "index_tape.npz")
    a.need(
        tape["indices"].dtype == np.int64 and tape["indices"].shape == (updates, 256),
        "Complete minibatch index tape",
    )
    rows = [json.loads(line) for line in (directory / "index_rng.jsonl").read_text().splitlines()]
    a.need(len(rows) == updates, "Every actual minibatch has before/after PCG proof")
    sampler = np.random.Generator(np.random.PCG64(0))
    sampler.bit_generator.state = state["rng"]
    for i, row in enumerate(rows):
        a.equal(row["rng_before"], sampler.bit_generator.state, "Minibatch incoming private PCG")
        expected_indices = sampler.integers(0, len(inputs), size=256)
        a.equal(tape["indices"][i], expected_indices, "Independent actual minibatch draw")
        a.need(
            row["indices_fingerprint"] == typed_fingerprint(expected_indices),
            "Minibatch row fingerprint",
        )
        a.equal(row["rng_after"], sampler.bit_generator.state, "Minibatch outgoing private PCG")
        a.need(
            row["update"] == int(tape["updates"][i]) == state["updates"] + i + 1
            and row["fit_update"] == int(tape["fit_updates"][i]) == i + 1,
            "Cumulative versus new-block minibatch indexing",
        )
    a.equal(continued["rng"], sampler.bit_generator.state, "Child private-PCG continuation")
    a.equal(
        continued["adam"]["param_groups"],
        state["adam"]["param_groups"],
        "Adam hyperparameters and parameter order unchanged",
    )
    a.need(
        set(continued["adam"]["state"]) == set(state["adam"]["state"]),
        "Adam state roster unchanged",
    )
    for key, actual in continued["adam"]["state"].items():
        before = state["adam"]["state"][key]
        a.need(set(actual) == {"step", "exp_avg", "exp_avg_sq"}, "Existing Adam slots only")
        a.close(
            float(actual["step"]),
            float(before["step"]) + updates,
            "Every Adam parameter advances by the fixed budget",
            atol=0,
        )
        for slot in ("exp_avg", "exp_avg_sq"):
            value = actual[slot].detach().cpu().numpy()
            a.need(
                value.shape == before[slot].shape
                and value.dtype == np.float32
                and np.isfinite(value).all(),
                "Finite continued Adam moments",
            )
        a.need(
            np.all(actual["exp_avg_sq"].detach().cpu().numpy() >= 0),
            "Nonnegative Adam second moments",
        )
    curve = fit["learning_curve"]
    a.need(
        len(curve) == updates
        and not fit["anchor_refreshes"]
        and fit["parameter_count"] == fit["proximal_divisor"] == 19590,
        "Existing learner and complete fitting curve",
    )
    for i, row in enumerate(curve):
        a.need(
            row["update"] == state["updates"] + i + 1
            and row["fit_update"] == i + 1
            and row["proximal_loss"] == 0
            and row["total_loss"] == row["action_mse"]
            and all(
                np.isfinite(row[name]) and row[name] >= 0
                for name in ("action_mse", "total_loss", "unclipped_gradient_norm")
            ),
            "Finite ordinary-MSE progress without extra fit operations",
        )
    expected_work = {
        unit: dict(requested=amount, completed=amount, unknown_partial=0)
        for unit, amount in (
            ("neural_rows", updates * 256),
            ("backward_rows", updates * 256),
            ("optimizer_steps", updates),
        )
    }
    a.equal(fit["work"], expected_work, "Exact forward/backward/Adam fit census")
    return dict(
        parent=parent["policy_hash"],
        child=child["policy_hash"],
        updates=updates,
        index_tape_sha256=a.bind(directory / "index_tape.npz")["sha256"],
        minibatch_indices_fingerprint=typed_fingerprint(tape["indices"]),
        child_directory=str(child_directory),
    )


def _roster(a: Audit, directory: Path, rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        name = row["path"]
        path = directory / name
        a.need(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and path.resolve().is_relative_to(directory.resolve())
            and name not in seen,
            "Contained unique source/input roster path",
        )
        seen.add(name)
        a.bind(path, row)
    a.need([row["path"] for row in rows] == sorted(seen), "Canonical sorted file roster")


def audit_native_input_lineage(a: Audit, cfg: dict, prepared: dict, data_root: Path) -> dict:
    """P1.1 changes input provenance only; P1.0 remains an immutable record."""
    a.need(
        cfg.get("version") == prepared.get("version") == "P1.1"
        and cfg.get("input_mode") == prepared.get("input_mode") == "worker-native"
        and prepared.get("new_global_envelopes_created") == 0,
        "Versioned worker-native inputs retain all incoming stochastic streams",
    )
    previous_config_path = a.bound(prepared["previous_config"])
    previous_prepared_path = a.bound(prepared["previous_prepared_manifest"])
    previous_config, previous = a.json(previous_config_path), a.json(previous_prepared_path)
    a.need(
        previous_config.get("version") == "P1.0"
        and previous["schema"] == "trajectory-pi-prepared-1"
        and previous["config_sha256"] == prepared["previous_config"]["sha256"],
        "Exact previously failed P1.0 configuration and prepared lineage",
    )
    for name, binding_name in (
        ("previous_config", "previous_config"),
        ("previous_prepared_manifest", "previous_prepared_manifest"),
    ):
        declared = cfg["lineage"][name]
        a.need(
            all(prepared[binding_name][key] == declared[key] for key in ("bytes", "sha256")),
            "Committed P1.0 lineage hash: " + name,
        )
    scientific = (
        "hypothesis",
        "streams",
        "arms",
        "rounds",
        "horizon",
        "parents",
        "teachers",
        "planner",
        "task",
        "learning",
        "optimizer",
        "score",
        "admission",
        "queries",
        "evaluation",
        "seeds",
        "criteria",
        "resources",
        "limits",
        "prerequisites",
    )
    for name in scientific:
        a.equal(
            cfg[name], previous_config[name], "Unchanged P1.0 scientific recipe and gate: " + name
        )
    a.need(
        len(prepared["parents"]) == len(previous["parents"]) == 3,
        "Exactly three incoming streams, aliased by both arms",
    )
    previous_files = {row["path"]: row for row in previous["files"]}
    declared_files = {
        (row["stream"], row["filename"]): row
        for row in cfg["lineage"]["parent_stream_files"]["files"]
    }
    copied = []
    for parent, old in zip(prepared["parents"], previous["parents"], strict=True):
        for key in old:
            a.equal(parent[key], old[key], "Original parent metadata remains unchanged: " + key)
        for name in ("policy.pt", "global_rng.pt", "stream.json"):
            relative = str(Path(old["directory"]) / name)
            source_binding = previous_files[relative]
            a.bind(data_root / relative, source_binding)
            declared = declared_files[(old["stream"], name)]
            a.need(
                all(declared[key] == source_binding[key] for key in ("bytes", "sha256")),
                "Committed byte identity of all nine incoming stream files",
            )
            if "inherited_stream_files" in parent:
                a.bind(data_root / relative, parent["inherited_stream_files"][name])
            copied.append(
                dict(
                    stream=old["stream"],
                    filename=name,
                    bytes=source_binding["bytes"],
                    sha256=source_binding["sha256"],
                )
            )
    a.need(
        len(copied) == 9, "No incoming policy, Adam/private/global RNG or envelope reserialization"
    )
    manifest_path = a.bound(prepared["validation_manifest"])
    validation = a.json(manifest_path)
    a.need(
        validation["schema"] == "trajectory-pi-native-validation-1"
        and validation["status"] == "PASS"
        and validation["p1_config_sha256"] == prepared["config_sha256"]
        and validation["previous_prepared_manifest"] == prepared["previous_prepared_manifest"],
        "Exact audited worker-native validation is bound to this P1.1 recipe",
    )
    native_receipt_path = a.bound(validation["native_audit"]["receipt"])
    native_findings_path = a.bound(validation["native_audit"]["findings"])
    native_receipt, native_findings = a.json(native_receipt_path), a.json(native_findings_path)
    a.need(
        native_receipt["passed"] is True
        and native_receipt["native_validation_passed"] is True
        and native_findings["native_validation_passed"] is True
        and native_findings["result_binding"] == validation["result"]
        and native_findings["source_manifest_sha256"] == validation["source_manifest_sha256"]
        and native_findings["config_sha256"] == validation["config_sha256"]
        and native_findings["p1_config_sha256"] == validation["p1_config_sha256"],
        "Audit pass and scientific baseline pass belong to the same raw result/source/config",
    )
    a.bind(native_findings_path, native_receipt["findings_binding"])
    result_path, receipt_path = a.bound(validation["result"]), a.bound(validation["receipt"])
    result, receipt = a.json(result_path), a.json(receipt_path)
    a.need(
        result["status"] == "PASS"
        and receipt["status"] == "COMPLETED"
        and result["work"] == receipt["counts"] == native_findings["work"],
        "Worker-native paid result and audit census agree",
    )
    a.need(
        len(validation["episodes"]) == len(native_findings["episodes"]) == 2,
        "Both original slots have native capture, repeat and K=1 proof",
    )
    for teacher, episode, proof in zip(
        prepared["teachers"], validation["episodes"], native_findings["episodes"], strict=True
    ):
        historical = next(row for row in cfg["teachers"] if row["slot"] == teacher["slot"])
        a.need(
            teacher["slot"] == episode["slot"] == proof["slot"]
            and teacher["source"] == historical["episode"]
            and episode["original_episode"] == proof["original_episode"]
            and all(
                teacher["source"][key] == episode["original_episode"][key]
                for key in ("bytes", "sha256")
            )
            and teacher["native_capture"] == episode["capture"] == proof["capture"],
            "Worker-native teacher retains its original episode and exact audited capture owner",
        )
        for key in ("original_prepared", "replay", "candidate"):
            a.need(episode[key] == proof[key], "Same audited native proof array: " + key)
            a.bound(episode[key])
        for key in (
            "native_replay_passed",
            "single_candidate_passed",
            "initial_context_exact",
            "actions_exact",
        ):
            a.need(
                episode[key] is True and proof[key] is True,
                "Complete native validation gate: " + key,
            )
        raw = load_arrays(a, a.bound(episode["capture"]))
        canonical = load_arrays(a, data_root / teacher["file_relative"])
        for key in ("states", "warmstarts", "integrations", "actions", "costs"):
            a.equal(
                canonical[key],
                raw[key],
                "Canonical native teaching array copies capture bytes: " + key,
            )
        a.equal(
            canonical["observations"],
            raw["observations"][:-1],
            "Native canonical observations retain exact pre-action indexing",
        )
        a.close(
            teacher["complete_cost"],
            np.sum(raw["costs"]),
            "New immutable native initial-cost reference",
        )
        a.need(
            teacher["walking"] is True and episode["outcome"]["walking"] is True,
            "New baseline remains competent walking",
        )
        original_copy = data_root / "originals" / f"teacher_{teacher['slot']}.npz"
        a.bind(original_copy, teacher["source"])
    origin = validation["origin_replay"]
    r0_result, r0_receipt, r0_findings = (
        a.json(a.bound(origin[name])) for name in ("result", "audit_receipt", "audit_findings")
    )
    a.bound(origin["external_summary"])
    a.need(
        origin["origin_replay_passed"] is True
        and r0_result["status"] == "ORIGIN_REPLAY_PASS"
        and r0_receipt["passed"] is True
        and r0_receipt["origin_replay_passed"] is True
        and r0_findings["origin_replay_passed"] is True
        and all(row["passed"] for row in r0_result["source_comparisons"].values()),
        "Preserved origin-host replay passed before worker-native validation; no origin rerun",
    )
    origin_names = ("result", "audit_receipt", "audit_findings", "external_summary")
    a.need(
        set(cfg["lineage"]["origin_replay"]["bindings"]) == set(origin_names),
        "All four canonical origin proof bindings are retained",
    )
    for name in origin_names:
        frozen = cfg["lineage"]["origin_replay"]["bindings"][name]
        a.need(
            all(origin[name][key] == frozen[key] for key in ("bytes", "sha256")),
            "Committed original replay/audit proof bytes",
        )
    expected_imports = [
        dict(phase="r0", receipt=origin["result"], counts=r0_result["work"]),
        dict(phase="rv", receipt=validation["receipt"], counts=receipt["counts"]),
    ]
    a.equal(
        prepared["imported_work"],
        expected_imports,
        "Origin and worker-native paid physics are imported once, separately from new preparation/P0",
    )
    return dict(
        input_mode="worker-native",
        previous_config=prepared["previous_config"],
        previous_prepared_manifest=prepared["previous_prepared_manifest"],
        validation_manifest=prepared["validation_manifest"],
        copied_incoming_files=copied,
        r0_work=r0_result["work"],
        worker_native_validation_work=receipt["counts"],
        r0_receipt_binding=origin["result"],
        worker_native_validation_receipt_binding=validation["receipt"],
        native_audit=validation["native_audit"],
        new_parent_refits=0,
        imported_work=expected_imports,
        original_p1_failure="Preserved historical cost, separately reported; never relabeled or rerun",
    )


def audit_launch(a: Audit, run: Path) -> tuple[dict, dict, dict, dict]:
    launch = a.json(run / "launch.json")
    a.need(launch.get("schema") == "trajectory-pi-p1-launch-1", "Frozen launch schema")
    bound = {
        name: a.bound(launch[name])
        for name in (
            "config",
            "source_manifest",
            "prepared_manifest",
            "preflight",
            "resource_observation",
        )
    }
    cfg, source, prepared, preflight = (
        a.json(bound[name])
        for name in ("config", "source_manifest", "prepared_manifest", "preflight")
    )
    a.need(
        cfg["schema"] == "trajectory-pi-p1-1" and cfg["launch_ready"] is True,
        "Committed launch-ready P1 configuration",
    )
    a.need(
        cfg["rounds"] == 20
        and cfg["horizon"] == 1000
        and cfg["streams"] == [0, 1, 2]
        and cfg["arms"] == [dict(name="independent", alpha=0.0), dict(name="coupled", alpha=0.12)]
        and cfg["learning"]["updates"] == 100,
        "Exactly the frozen finite single-mechanism contrast",
    )
    task = dict(
        frame_skip=4,
        cost_style="target_velocity",
        target_velocity=1.5,
        vel_cost_weight=2.0,
        vertical_velocity_cost_weight=0.5,
        angle_cost_weight=0.1,
        unhealthy_cost_weight=5.0,
        ctrl_cost_weight=0.001,
        terminal_unhealthy_weight=20.0,
        apply_terminal_cost_on_done=True,
    )
    a.need(cfg["task"] == task, "Audit cost formula matches unchanged declared task")
    if (run / "config.json").is_file():
        a.equal(a.json(run / "config.json"), cfg, "Output records exact frozen configuration")
    a.need(
        source["schema"] == "trajectory-pi-source-1"
        and prepared["schema"] == "trajectory-pi-prepared-1",
        "Source/prepared schemas",
    )
    source_root, data_root = Path(launch["source_directory"]), Path(launch["prepared_directory"])
    _roster(a, source_root, source["files"])
    canonical = json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()
    a.need(hashlib.sha256(canonical).hexdigest() == source["roster_sha256"], "Source roster hash")
    _roster(a, data_root, prepared["files"])
    a.need(
        prepared["config_sha256"] == launch["config"]["sha256"]
        and prepared["normalization_inherited"] is True
        and prepared["parent_refits"] == 0
        and prepared["historical_global_rng_available"] is False
        and prepared["new_study_globals_frozen"] is True,
        "Preserved legacy/new-global distinction",
    )
    for name in ("config", "source_manifest", "prepared_manifest"):
        a.need(preflight[name + "_sha256"] == launch[name]["sha256"], "P0 input/source seal")
    normals = []
    for spec, row in zip(cfg["parents"], prepared["parents"], strict=True):
        a.need(
            spec["stream"] == row["stream"]
            and spec["prepared_directory"] == row["directory"]
            and spec["checkpoint"] == row["source"],
            "Parent roster and original byte binding",
        )
        payload, envelope = audit_stream(a, data_root / row["directory"])
        a.bind(data_root / row["directory"] / "policy.pt", row["source"])
        a.need(
            Path(row["source"]["path"]).is_absolute()
            and payload["optimizer_state"]["updates"] == 2000
            and payload["optimizer_state"]["fit_updates"] == 2000
            and payload["policy_hash"] == row["policy_hash"]
            and envelope["optimizer_fingerprint"] == row["optimizer_fingerprint"]
            and envelope["global_rng_fingerprint"] == row["global_rng_fingerprint"]
            and envelope["new_study_global_seed"] == spec["new_global_envelope_seed"],
            "Original 2k actor/Adam/private-PCG retained without refit",
        )
        normals.append(
            {key: payload["state_dict"][key] for key in ("observation_mean", "observation_std")}
        )
    for normal in normals[1:]:
        a.equal(normal, normals[0], "Common inherited W2 normalization across streams")
    native_input = prepared.get("input_mode") == "worker-native"
    if native_input:
        a.input_lineage = audit_native_input_lineage(a, cfg, prepared, data_root)
    for spec, row in zip(cfg["teachers"], prepared["teachers"], strict=True):
        a.need(
            spec["slot"] == row["slot"] and spec["episode"] == row["source"],
            "Original teacher roster",
        )
        copied_source = data_root / "originals" / f"teacher_{spec['slot']}.npz"
        a.need(Path(row["source"]["path"]).is_absolute(), "Explicit original evidence location")
        a.bind(copied_source, row["source"])
        original = load_arrays(a, copied_source)
        converted = load_arrays(a, data_root / row["file_relative"])
        if not native_input:
            for key, terminal in (
                ("states", "terminal_state"),
                ("warmstarts", "terminal_warmstarts"),
                ("integrations", "terminal_integrations"),
            ):
                a.equal(
                    converted[key],
                    np.concatenate((original[key], original[terminal][None])),
                    "Terminal-inclusive source context conversion: " + key,
                )
            for key in ("observations", "actions", "costs"):
                a.equal(converted[key], original[key], "Unchanged original teacher array: " + key)
        else:
            a.equal(
                converted["actions"],
                original["actions"],
                "Worker-native inputs retain exact original U",
            )
        a.close(
            converted["observations"],
            observations(converted["states"][:-1]),
            "Original pre-action indexing",
            atol=0,
        )
        cost, _, valid = batch_cost(converted["states"][None, 1:], converted["actions"][None])
        a.need(
            valid[0]
            and healthy(converted["states"][1:]).all()
            and np.mean(np.abs(converted["states"][1:, 10] - 1.5)) <= 0.3,
            "Bound initial teacher competence",
        )
        a.close(cost[0], converted["costs"].sum(), "Original task-cost array accounting")
        a.close(row["complete_cost"], cost[0], "Prepared teacher cost summary")
    seed = cfg["seeds"]
    planner_roster = [
        dict(
            stream=s,
            round=r,
            slot=t,
            argument=seed["planner_argument_base"]
            + s * seed["stream_stride"]
            + r * seed["round_stride"]
            + t * seed["slot_stride"],
            actual_pcg_seed=seed["planner_argument_base"]
            + s * seed["stream_stride"]
            + r * seed["round_stride"]
            + t * seed["slot_stride"]
            + 20000,
        )
        for s in cfg["streams"]
        for r in range(1, cfg["rounds"] + 1)
        for t in (0, 1)
    ]
    a.need(
        seed["planner_roster"] == planner_roster and seed["prototype_seed_offset"] == 20000,
        "Explicit paired planner seed roster",
    )
    query_roster = [
        dict(
            stream=s,
            round=r,
            anchor_index=i,
            anchor_step=t,
            argument=seed["query_argument_base"]
            + s * seed["stream_stride"]
            + r * seed["round_stride"]
            + i * seed["slot_stride"],
            actual_pcg_seed=seed["query_argument_base"]
            + s * seed["stream_stride"]
            + r * seed["round_stride"]
            + i * seed["slot_stride"]
            + 20000,
        )
        for s in cfg["streams"]
        for r in range(cfg["rounds"] + 1)
        for i, t in enumerate(cfg["queries"]["anchors"])
    ]
    a.need(seed["query_roster"] == query_roster, "Explicit paired query seed roster")
    panels = cfg["evaluation"]["panels"]
    a.need(
        [row["round"] for row in panels] == list(range(cfg["rounds"] + 1)),
        "Every round has a predeclared fresh reset panel",
    )
    resets = [value for panel in panels for value in panel["reset_seeds"]]
    resets += cfg["evaluation"]["confirmation_seeds"]
    a.need(len(set(resets)) == len(resets), "Evaluation starts and confirmation panels disjoint")
    return cfg, launch, prepared, preflight


def panel_summary(rows: list[dict]) -> dict:
    return dict(
        episodes=len(rows),
        walking=sum(row["walking"] for row in rows),
        full_survival=sum(row["full_survival"] for row in rows),
        mean_cost=float(np.mean([row["complete_cost"] for row in rows])),
        mean_controls=float(np.mean([row["controls"] for row in rows])),
    )


def audit_panel(
    a: Audit, panel: dict, seeds: list[int], checkpoint: Path, paired_contexts: dict[int, str]
) -> dict:
    a.need(len(panel["jobs"]) == len(panel["outcomes"]) == len(seeds), "Complete evaluation panel")
    outcomes = []
    policy = a.checkpoint(checkpoint)["policy_hash"]
    for index, (saved, saved_outcome, seed) in enumerate(
        zip(panel["jobs"], panel["outcomes"], seeds, strict=True)
    ):
        directory = Path(saved["path"])
        entry = audit_native(a, directory)
        a.need(
            entry["kind"] == "actor"
            and entry["reset_seed"] == seed
            and entry["policy_hash"] == policy,
            "Evaluation reset/checkpoint owner",
        )
        a.equal(saved["result"], a.json(directory / "result.json"), "Panel native result binding")
        a.equal(
            saved["receipt"], a.json(directory / "receipt.json"), "Panel native receipt binding"
        )
        compare_outcome(a, saved_outcome, entry["outcome"])
        a.need(
            paired_contexts.setdefault(seed, entry["first_context"]) == entry["first_context"],
            "Paired full simulator reset across policies and streams",
        )
        a.need(
            len(panel["paired_reset_context_fingerprints"]) == len(seeds)
            and panel["paired_reset_context_fingerprints"][index]
            == entry["reset_context_fingerprint"],
            "Panel saved reset fingerprint includes exact physical context and observation",
        )
        outcomes.append(entry["outcome"])
    measured = panel_summary(outcomes)
    for name in measured:
        a.close(panel["summary"][name], measured[name], "Independent panel summary: " + name)
    return dict(outcomes=outcomes, summary=measured)


def _screen(early: list[float], late: list[float], independent: list[float], gate: dict) -> dict:
    before, after, control = (float(np.mean(values)) for values in (early, late, independent))
    resolved = before > 1e-12 and control > 1e-12
    return dict(
        resolved=resolved,
        early=before,
        late=after,
        independent_late=control,
        contraction=(1 - after / before) if resolved else None,
        late_ratio=after / control if resolved else None,
        passed=bool(
            resolved
            and after <= (1 - gate["minimum_relative_contraction"]) * before
            and after <= gate["maximum_late_ratio_to_independent"] * control
        ),
    )


def _walking(child: dict, independent: dict, parent: dict, threshold: int) -> dict:
    wd = [child["walking"] - row["walking"] for row in (independent, parent)]
    sd = [child["full_survival"] - row["full_survival"] for row in (independent, parent)]
    return dict(
        passed=all(value >= threshold for value in wd),
        no_regression=all(value >= 0 for value in [*wd, *sd]),
        walking_vs_independent=wd[0],
        walking_vs_parent=wd[1],
        survival_vs_independent=sd[0],
        survival_vs_parent=sd[1],
    )


def independent_criteria(history: list[dict], confirmation: dict, config: dict) -> dict:
    """Apply the frozen scientific gates to independently rebuilt round records."""
    g = config["criteria"]
    rounds = {record["round"]: record for record in history}
    streams, joint, no_regression = {}, [], True
    for stream in config["streams"]:
        name = str(stream)

        def rows(indices: list[int], arm: str) -> list[dict]:
            return [rounds[index]["streams"][name][arm] for index in indices]

        early = rows(g["early_rounds"], "coupled")
        late = rows(g["late_rounds"], "coupled")
        control = rows(g["late_rounds"], "independent")
        teacher = _screen(
            [r["teacher_after_fit"]["rms"] for r in early],
            [r["teacher_after_fit"]["rms"] for r in late],
            [r["teacher_after_fit"]["rms"] for r in control],
            g["teacher_gap"],
        )
        student = {}
        for anchor in g["student_gap"]["anchors"]:
            key = str(anchor)
            all_rows = rows(g["early_rounds"], "independent") + early + late + control
            qualified = all(
                row["queries"][key]["present"] and row["queries"][key]["qualified"]
                for row in all_rows
            )
            student[key] = (
                _screen(
                    [r["queries"][key]["action_gap"] for r in early],
                    [r["queries"][key]["action_gap"] for r in late],
                    [r["queries"][key]["action_gap"] for r in control],
                    g["student_gap"],
                )
                if qualified
                else dict(
                    passed=False,
                    resolved=False,
                    reason="Fixed query missing or not walking-qualified",
                )
            )
        original = rounds[0]["streams"][name]["coupled"]["bank"]
        initial_cost = float(np.mean([r["outcome"]["complete_cost"] for r in original]))
        late_cost = float(
            np.mean([item["outcome"]["complete_cost"] for row in late for item in row["bank"]])
        )
        retained = all(
            item["outcome"]["walking"] and item["outcome"]["full_survival"]
            for record in history
            for arm in ("independent", "coupled")
            for item in record["streams"][name][arm]["bank"]
        )
        counts = [
            sum(row["raw_candidates"][slot]["result"]["outcome"]["walking"] for row in late)
            for slot in range(2)
        ]
        competent = bool(
            retained
            and late_cost
            <= initial_cost
            * g["teacher_competence"]["maximum_late_aggregate_cost_ratio_to_initial"]
            and min(counts)
            >= g["teacher_competence"]["raw_best_minimum_walking_rounds_per_path_in_late_window"]
        )
        final = rounds[config["rounds"]]
        primary = _walking(
            final["streams"][name]["coupled"]["student"]["summary"],
            final["streams"][name]["independent"]["student"]["summary"],
            final["parents"][name]["summary"],
            g["walking"]["minimum_additional_successes"],
        )
        confirm = _walking(
            confirmation[name]["coupled"]["summary"],
            confirmation[name]["independent"]["summary"],
            confirmation[name]["parent"]["summary"],
            g["walking"]["minimum_additional_successes"],
        )
        no_regression = bool(
            no_regression and retained and primary["no_regression"] and confirm["no_regression"]
        )
        passed = bool(
            teacher["passed"]
            and all(row["passed"] for row in student.values())
            and competent
            and primary["passed"]
            and confirm["passed"]
        )
        if passed:
            joint.append(stream)
        streams[name] = dict(
            teacher_gap=teacher,
            student_gap=student,
            teacher_competence=dict(
                passed=competent,
                retained_nonregression=retained,
                late_cost_ratio=late_cost / initial_cost,
                raw_walking_rounds_per_path=counts,
            ),
            primary=primary,
            confirmation=confirm,
            joint_passed=passed,
            individual_competence=bool(
                final["streams"][name]["coupled"]["student"]["summary"]["walking"] >= 15
                and confirmation[name]["coupled"]["summary"]["walking"] >= 15
            ),
        )
    passed = len(joint) >= g["minimum_joint_streams"] and no_regression
    walking = sum(
        row["primary"]["passed"]
        and row["confirmation"]["passed"]
        and row["teacher_competence"]["passed"]
        for row in streams.values()
    )
    gap = sum(
        row["teacher_gap"]["passed"]
        and all(item["passed"] for item in row["student_gap"].values())
        and row["teacher_competence"]["passed"]
        for row in streams.values()
    )
    unresolved = any(
        not row["teacher_gap"]["resolved"]
        or any(not item["resolved"] for item in row["student_gap"].values())
        for row in streams.values()
    )
    verdict = (
        "HELPED_FINITE_ALTERNATING_LOOP"
        if passed
        else "PARTIAL_WALKING_GAIN_WITH_UNRESOLVED_GAP"
        if walking >= 2 and no_regression and unresolved
        else "PARTIAL_WALKING_GAIN_GAP_NOT_CLOSED"
        if walking >= 2 and no_regression
        else "GAP_ONLY_IMPROVEMENT"
        if gap >= 2 and no_regression
        else "NO_CONFIRMED_HELP"
    )
    return dict(
        verdict=verdict,
        passed=bool(passed),
        joint_streams=joint,
        per_stream=streams,
        all_stream_nonregression=no_regression,
        full_project_success=False,
        broader_requirements_not_tested=g["broader_requirements_not_tested"],
    )


def audit_query(
    a: Audit,
    saved: dict,
    actor_panel: dict,
    checkpoint: Path,
    config: dict,
    stream: int,
    arm: dict,
    round_index: int,
    anchor: int,
    initial_bank: list[dict],
    directory: Path,
) -> dict:
    episode = actor_panel["jobs"][config["queries"]["evaluation_episode"]]
    actor_directory = Path(episode["path"])
    actor_raw = load_arrays(a, actor_directory / "trace.h5")
    present = len(actor_raw["actions"]) > anchor
    a.need(saved["present"] is present, "Missing query anchor respects actual student termination")
    if not present:
        a.need(
            saved["qualified"] is False
            and saved["action_gap"] is None
            and not (directory / "job.json").exists(),
            "Missing query stays missing without paid replacement",
        )
        return dict(saved)
    a.need(Path(saved["path"]).resolve() == directory.resolve(), "Declared fixed query path")
    job = a.json(directory / "job.json")
    index = config["queries"]["anchors"].index(anchor)
    seed = config["seeds"]
    argument = (
        seed["query_argument_base"]
        + stream * seed["stream_stride"]
        + round_index * seed["round_stride"]
        + index * seed["slot_stride"]
    )
    a.need(
        job["kind"] == "solve"
        and job["alpha"] == arm["alpha"]
        and job["iterations"] == config["queries"]["iterations"]
        and job["planner_seed"] == argument
        and job["planner"] == config["planner"]
        and job["context_step"] == anchor
        and Path(job["context_file"]).resolve() == (actor_directory / "trace.h5").resolve()
        and Path(job["checkpoint"]).resolve() == checkpoint.resolve()
        and job["initial_plan_binding"]
        == initial_bank[config["queries"]["initial_template_slot"]]["plan_binding"]
        and job["full_banks"] == config["queries"]["full_bank_iterations"],
        "Teacher-query operator has frozen context, initial plan, policy and seed",
    )
    measured = audit_native(a, directory)
    gap = float(
        np.linalg.norm(
            np.asarray(measured["selected_first_action"], dtype=np.float64)
            - actor_raw["actions"][anchor].astype(np.float64)
        )
    )
    a.close(saved["action_gap"], gap, "Student-state action gap at exact saved pre-action anchor")
    a.need(
        saved["qualified"] is measured["outcome"]["walking"],
        "Query qualification requires its own complete walking",
    )
    compare_outcome(a, saved["query_outcome"], measured["outcome"])
    a.bind(actor_directory / "trace.h5", saved["actor_trace"])
    if anchor == 0:
        cost_gap = (
            a.native[str(actor_directory.resolve())]["outcome"]["complete_cost"]
            - measured["outcome"]["complete_cost"]
        )
        a.close(
            saved["policy_cost_gap"], cost_gap, "Same-context full-horizon cost gap at anchor zero"
        )
    else:
        a.need(saved["policy_cost_gap"] is None, "No unmatched suffix cost gap at later anchor")
    return dict(
        saved,
        action_gap=gap,
        qualified=measured["outcome"]["walking"],
        query_outcome=measured["outcome"],
    )


def audit_rounds(
    a: Audit, run: Path, config: dict, launch: dict, prepared: dict, saved_history: list[dict]
) -> tuple[list[dict], dict, list[dict]]:
    data_root = Path(launch["prepared_directory"])
    parents = {row["stream"]: data_root / row["directory"] for row in prepared["parents"]}
    current = {(s, arm["name"]): parents[s] for s in config["streams"] for arm in config["arms"]}
    banks: dict[tuple[int, str], list[dict]] = {}
    history, fit_proofs = [], []
    paired_contexts: dict[int, str] = {}
    initial_bank = None
    for round_index in range(config["rounds"] + 1):
        directory = run / "rounds" / f"round_{round_index:03d}"
        record = a.json(directory / "complete.json")
        a.equal(record, saved_history[round_index], "Root history equals durable complete round")
        a.need(
            record["schema"] == "trajectory-pi-round-1"
            and record["round"] == round_index
            and set(record["streams"]) == {str(s) for s in config["streams"]},
            "Exact chronological stream/round roster",
        )
        rebuilt = json.loads(json.dumps(record))
        seeds = config["evaluation"]["panels"][round_index]["reset_seeds"]
        acquisition_pairs: dict[int, dict] = {}
        fit_pairs: dict[int, str] = {}
        for stream in config["streams"]:
            parent_panel = record["parents"][str(stream)]
            rebuilt["parents"][str(stream)] = audit_panel(
                a, parent_panel, seeds, parents[stream] / "policy.pt", paired_contexts
            )
            for arm in config["arms"]:
                name, key = arm["name"], (stream, arm["name"])
                row = record["streams"][str(stream)][name]
                if round_index == 0:
                    bank = row["bank"]
                    if initial_bank is None:
                        initial_bank = bank
                    a.need(
                        bank == initial_bank, "Identical incoming teacher paths across arms/streams"
                    )
                    for item, spec in zip(bank, prepared["teachers"], strict=True):
                        source = data_root / spec["file_relative"]
                        a.need(
                            Path(item["trace"]).resolve() == source.resolve()
                            and Path(item["plan"]).resolve() == source.resolve()
                            and Path(item["context"]).resolve() == source.resolve()
                            and item["origin_round"] == 0,
                            "Original complete-context initial bank",
                        )
                        teacher = load_arrays(a, source)
                        a.close(
                            item["outcome"]["complete_cost"],
                            teacher["costs"].sum(),
                            "Round-zero initial cost binding used by admission and competence",
                        )
                        a.close(
                            item["outcome"]["velocity_mae"],
                            np.mean(np.abs(teacher["states"][1:, 10] - 1.5)),
                            "Round-zero original walking velocity error",
                        )
                        a.need(
                            item["outcome"]["walking"] is True
                            and item["outcome"]["full_survival"] is True
                            and item["outcome"]["controls"] == 1000,
                            "Round-zero complete competent source outcome",
                        )
                    a.need(
                        row["fit"] is None and not row["admission"] and not row["raw_candidates"],
                        "Round zero has no fitting/acquisition",
                    )
                    gap = teacher_gap(a, bank, a.checkpoint(current[key] / "policy.pt"))
                    for phase in (
                        "teacher_before",
                        "teacher_after_acquisition",
                        "teacher_after_fit",
                    ):
                        compare_gap(a, row[phase], gap)
                        rebuilt["streams"][str(stream)][name][phase] = gap
                else:
                    incoming = banks[key]
                    payload = a.checkpoint(current[key] / "policy.pt")
                    old_gap = teacher_gap(a, incoming, payload)
                    compare_gap(a, row["teacher_before"], old_gap)
                    bank = list(incoming)
                    a.need(
                        len(row["admission"]) == len(row["raw_candidates"]) == 2,
                        "Two paid acquisitions preserved before admission",
                    )
                    for slot in range(2):
                        target = directory / f"s{stream}" / name / "acquisition" / f"slot{slot}"
                        returned, decision = row["raw_candidates"][slot], row["admission"][slot]
                        a.need(
                            Path(returned["path"]).resolve() == target.resolve()
                            and Path(decision["path"]).resolve() == target.resolve(),
                            "Acquisition and admission owner",
                        )
                        job = a.json(target / "job.json")
                        seed = config["seeds"]
                        expected_seed = (
                            seed["planner_argument_base"]
                            + stream * seed["stream_stride"]
                            + round_index * seed["round_stride"]
                            + slot * seed["slot_stride"]
                        )
                        a.need(
                            job["alpha"] == arm["alpha"]
                            and job["iterations"] == config["optimizer"]["iterations"]
                            and job["planner_seed"] == expected_seed
                            and job["planner"] == config["planner"]
                            and job["initial_plan_binding"] == incoming[slot]["plan_binding"]
                            and job["context_binding"] == incoming[slot]["context_binding"]
                            and job["context_step"] == 0
                            and Path(job["checkpoint"]).resolve()
                            == (current[key] / "policy.pt").resolve()
                            and job["full_banks"] == config["optimizer"]["full_bank_iterations"],
                            "Matched fixed-alpha full-sequence acquisition recipe",
                        )
                        candidate = audit_native(a, target)
                        a.equal(
                            returned["result"],
                            a.json(target / "result.json"),
                            "Preserved raw acquisition result",
                        )
                        a.equal(
                            returned["receipt"],
                            a.json(target / "receipt.json"),
                            "Preserved raw acquisition paid accounting",
                        )
                        compare_outcome(a, decision["outcome"], candidate["outcome"])
                        accepted = bool(
                            candidate["outcome"]["walking"]
                            and candidate["outcome"]["full_survival"]
                            and candidate["outcome"]["complete_cost"]
                            <= config["admission"]["maximum_cost_ratio_to_initial"]
                            * initial_bank[slot]["outcome"]["complete_cost"]
                        )
                        a.need(
                            decision["slot"] == slot and decision["accepted"] is accepted,
                            "Parent-relative teacher competence gate, without agreement rejection",
                        )
                        if accepted:
                            bank[slot] = dict(
                                slot=slot,
                                trace=str(target / "trace.h5"),
                                trace_binding=dict(
                                    path=str((target / "trace.h5").resolve()),
                                    **{
                                        k: a.bind(target / "trace.h5")[k]
                                        for k in ("bytes", "sha256")
                                    },
                                ),
                                plan=str(target / "solution.npz"),
                                plan_binding=dict(
                                    path=str((target / "solution.npz").resolve()),
                                    **{
                                        k: a.bind(target / "solution.npz")[k]
                                        for k in ("bytes", "sha256")
                                    },
                                ),
                                context=incoming[slot]["context"],
                                context_binding=incoming[slot]["context_binding"],
                                origin_round=round_index,
                                outcome=candidate["outcome"],
                            )
                        pair = acquisition_pairs.setdefault(slot, {})
                        rng = candidate["rng_states"]
                        a.equal(
                            pair.setdefault(stream, rng),
                            rng,
                            "Paired PI random streams across agreement arms",
                        )
                    a.need(
                        row["bank"] == bank,
                        "Accepted candidates or unchanged incumbent, exact provenance",
                    )
                    acquired_gap = teacher_gap(a, bank, payload)
                    compare_gap(a, row["teacher_after_acquisition"], acquired_gap)
                    fit_directory = directory / f"s{stream}" / name / "fit"
                    a.need(
                        Path(row["fit"]["path"]).resolve() == fit_directory.resolve(),
                        "Fit directory owner",
                    )
                    a.bind(fit_directory / "fit.json", row["fit"]["binding"])
                    proof = audit_fit(
                        a, fit_directory, current[key], bank, config["learning"]["updates"]
                    )
                    fit_proofs.append(proof)
                    a.need(
                        fit_pairs.setdefault(stream, proof["minibatch_indices_fingerprint"])
                        == proof["minibatch_indices_fingerprint"],
                        "Paired inherited minibatch stream across arms",
                    )
                    current[key] = Path(proof["child_directory"])
                    fitted_gap = teacher_gap(a, bank, a.checkpoint(current[key] / "policy.pt"))
                    compare_gap(a, row["teacher_after_fit"], fitted_gap)
                    for phase, gap in (
                        ("teacher_before", old_gap),
                        ("teacher_after_acquisition", acquired_gap),
                        ("teacher_after_fit", fitted_gap),
                    ):
                        rebuilt["streams"][str(stream)][name][phase] = gap
                banks[key] = bank
                measured_panel = audit_panel(
                    a, row["student"], seeds, current[key] / "policy.pt", paired_contexts
                )
                rebuilt["streams"][str(stream)][name]["student"] = measured_panel
                for anchor in config["queries"]["anchors"]:
                    query_directory = directory / "queries" / f"s{stream}" / name / f"a{anchor:03d}"
                    measured_query = audit_query(
                        a,
                        row["queries"][str(anchor)],
                        row["student"],
                        current[key] / "policy.pt",
                        config,
                        stream,
                        arm,
                        round_index,
                        anchor,
                        initial_bank,
                        query_directory,
                    )
                    rebuilt["streams"][str(stream)][name]["queries"][str(anchor)] = measured_query
                a.equal(
                    a.json(directory / f"s{stream}" / name / "measurements.json"),
                    row,
                    "Per-arm report equals durable round source",
                )
        state = a.json(directory / "state.json")
        a.need(
            state["round"] == round_index and state["schema"] == "trajectory-pi-state-1",
            "Closed continuation boundary for every round",
        )
        for row in state["streams"]:
            key = (row["stream"], row["arm"])
            a.need(
                Path(row["directory"]).resolve() == current[key].resolve()
                and row["bank"] == banks[key],
                "Closed state binds exact current stream and retained evidence",
            )
        history.append(rebuilt)
    confirmation = a.json(run / "final_panels.json")
    measured_confirmation = {}
    for stream in config["streams"]:
        name = str(stream)
        measured_confirmation[name] = {}
        for arm in ("independent", "coupled", "parent"):
            checkpoint = (
                parents[stream] if arm == "parent" else current[(stream, arm)]
            ) / "policy.pt"
            measured_confirmation[name][arm] = audit_panel(
                a,
                confirmation[name][arm],
                config["evaluation"]["confirmation_seeds"],
                checkpoint,
                paired_contexts,
            )
    return history, measured_confirmation, fit_proofs


def audit_preflight(a: Audit, launch: dict, config: dict, prepared: dict, preflight: dict) -> dict:
    directory = Path(launch["preflight"]["path"]).parent
    for path in directory.rglob("*"):
        if path.is_file():
            a.bind(path)
    ledger_path = directory / "ledger.jsonl"
    external = preflight["status"] in ("NOT_ATTEMPTED", "EXTERNAL_STOPPED")
    if preflight["status"] == "NOT_ATTEMPTED":
        a.need(
            not ledger_path.exists()
            and not (directory / "receipt.json").exists()
            and preflight.get("accounting_complete") is False
            and not preflight.get("work"),
            "P0 was never entered; no internal receipt or paid calls are fabricated",
        )
        return dict(
            status="NOT_ATTEMPTED",
            work={},
            accounting_complete=True,
            original_closure_present=False,
            original_failure=preflight.get("external_stop"),
            scope="No P0 ledger exists; external supervisor records no prerequisite launch",
        )
    if external and not ledger_path.exists():
        return dict(
            status=preflight["status"],
            work={},
            accounting_complete=False,
            original_closure_present=False,
            original_failure=preflight.get("external_stop"),
            scope="Externally stopped before a readable internal ledger; paid work remains unknown",
        )
    ledger = audit_ledger(a, directory, allow_open=external)
    if ledger["receipt"] is None:
        a.need(
            preflight.get("accounting_complete") is False,
            "External stop explicitly preserves missing internal closure",
        )
        return dict(
            status=preflight["status"],
            work=ledger["counts"],
            accounting_complete=False,
            unresolved_request_upper_charges=ledger["unresolved_request_upper_charges"],
            original_closure_present=False,
            original_failure=preflight.get("external_stop"),
            scope="Bound actual durable requests and completions; unresolved work is not inferred complete",
        )
    if external:
        return dict(
            status=preflight["status"],
            work=ledger["counts"],
            accounting_complete=True,
            original_closure_present=True,
            original_failure=preflight.get("external_stop"),
            scope="Preserved internal work closure reconciled; externally interrupted prerequisite remains unconfirmed",
        )
    saved_counts = {key: value for key, value in preflight["work"].items() if value}
    own_counts = {key: value for key, value in ledger["counts"].items() if value}
    a.equal(saved_counts, own_counts, "P0 paid work and closed ledger, including original failure")
    a.need(
        len({row["name"] for row in preflight["checks"]}) == len(preflight["checks"]),
        "Unique prerequisite checks",
    )
    if preflight["status"] != "PASS":
        a.need(
            ledger["receipt"]["status"] == "FAILED", "Original prerequisite failure remains failed"
        )
        return dict(
            status=preflight["status"],
            original_failure=preflight.get("error"),
            checks=preflight["checks"],
            work=ledger["counts"],
            accounting_complete=True,
            scope="Bound preserved failed P0 outputs/accounting; no replay or relaxed gate",
        )
    a.need(
        all(row["passed"] is True for row in preflight["checks"])
        and ledger["receipt"]["status"] == "COMPLETED",
        "All frozen prerequisites passed before launch",
    )
    required = {
        "full_context_replay",
        "pre_action_index",
        "full_sequence_no_shift",
        "score_formula",
        "checkpoint_continuation",
        "fall_accounting",
        "budget_enforcement",
        "zero_residual_reproduction",
    }
    a.need(
        required <= {row["name"] for row in preflight["checks"]},
        "Every required prerequisite present; absence is not a pass",
    )
    data = Path(launch["prepared_directory"])
    for row in prepared["teachers"]:
        source = load_arrays(a, data / row["file_relative"])
        replay = load_arrays(a, directory / f"teacher_{row['slot']}_replay.h5")
        measured = outcome(a, replay, 1000)
        a.need(measured["walking"], "Saved source prerequisite replay walks")
        for key in ("states", "warmstarts", "integrations", "actions", "costs"):
            a.close(replay[key], source[key], "P0 native source replay: " + key, atol=1e-10)
        a.close(
            replay["observations"][:-1],
            source["observations"],
            "P0 source replay pre-action observations",
            atol=1e-10,
        )
    parent = a.checkpoint(data / prepared["parents"][0]["directory"] / "policy.pt")
    raw = load_arrays(a, directory / "zero_residual_native.h5")
    measured = outcome(a, raw, 16)
    a.need(measured["controls"] == 16 and not measured["fell"], "P0 zero-residual live prefix")
    predicted = a.predict(parent, raw["observations"][:-1]).astype(np.float64)
    a.close(raw["actions"], predicted, "P0 zero-residual action identity", atol=0)
    forecast = load_arrays(a, directory / "zero_residual_forecast.npz")
    a.close(forecast["states"][0], raw["states"][1:], "P0 zero-residual replay states", atol=1e-10)
    a.close(forecast["costs"], [raw["costs"].sum()], "P0 zero-residual cost", atol=1e-10)
    D, _ = reconstruction(a, parent, raw["states"][0], forecast["states"], raw["actions"][None])
    a.close(D, np.zeros(1), "P0 exact zero reconstruction", atol=0)
    a.close(forecast["D"], D, "P0 saved zero reconstruction", atol=0)
    import h5py

    initial = load_arrays(a, data / prepared["teachers"][0]["file_relative"])["states"][0]
    for arm in config["arms"]:
        path = directory / ("solve_" + arm["name"])
        with h5py.File(path / "iterations.h5", "r") as groups:
            for index in range(2):
                group = groups[f"iteration_{index:04d}"]
                bank = load_arrays(a, path / f"bank_{index:04d}.npz")
                C, _, _ = batch_cost(bank["states"], bank["actions"])
                a.close(group["native_costs"][:], C, "P0 saved full-bank task cost")
                J = C
                if arm["alpha"]:
                    D, _ = reconstruction(a, parent, initial, bank["states"], bank["actions"])
                    a.close(group["reconstruction"][:], D, "P0 saved full-bank reconstruction")
                    J = C + arm["alpha"] * D
                a.close(group["total_objectives"][:], J, "P0 saved full-bank objective")
                weights, _ = gibbs(J, temperature(J, config["planner"]["lam_target_neff_fraction"]))
                a.close(group["weights"][:], weights, "P0 saved full-bank weights", atol=2e-13)
    full, full_envelope = audit_stream(a, directory / "continuation_full")
    split, split_envelope = audit_stream(a, directory / "continuation_resumed")
    for key in ("state_dict", "private_rng_state"):
        a.equal(full[key], split[key], "P0 actual 4 versus 2+reload+2 continuation: " + key)
    a.equal(
        {k: v for k, v in full["optimizer_state"].items() if k != "parent_policy_hash"},
        {k: v for k, v in split["optimizer_state"].items() if k != "parent_policy_hash"},
        "P0 exact operative Adam/private-PCG continuation; lineage remains distinct",
    )
    a.need(
        full_envelope["global_rng_fingerprint"] == split_envelope["global_rng_fingerprint"],
        "P0 global RNG continuation",
    )
    if config["prerequisites"].get("action_function_parity_atol") == 1e-6:
        parity = load_arrays(a, directory / "action_function_parity.npz")
        a.close(
            parity["numpy_actions"],
            a.predict(parent, parity["observations"]),
            "P0 canonical NumPy action function",
            atol=0,
        )
        a.close(
            parity["numpy_actions"],
            parity["torch_actions"],
            "Saved actual Torch/NumPy action parity",
            atol=1e-6,
        )
    return dict(
        status="PASS",
        work=ledger["counts"],
        accounting_complete=True,
        checks=preflight["checks"],
        saved_replay_algebra_verified=True,
        repeated_physics=0,
    )


def audit_run(a: Audit, run: Path, allow_stopped: bool = False) -> dict:
    """Audit a closed run; stopped evidence remains explicitly scientifically incomplete."""
    config, launch, prepared, preflight = audit_launch(a, run)
    input_lineage = getattr(a, "input_lineage", None)
    result = a.json(run / "result.json")
    a.need(result.get("full_project_success") is False, "No broader project-success claim")
    prerequisites = audit_preflight(a, launch, config, prepared, preflight)
    if result["status"] == "PRELAUNCH_STOPPED":
        a.need(
            allow_stopped
            and not result.get("history")
            and not list(run.rglob("job.json"))
            and not (run / "coordinator" / "ledger.jsonl").exists(),
            "Failed prerequisite stops all alternating research calls",
        )
        external = None
        if (run / "external_supervisor.json").exists():
            external = a.json(run / "external_supervisor.json")
            a.need(
                external["schema"] == "trajectory-pi-external-stop-1"
                and external["new_research_calls_after_stop"] == 0
                and external["reason"],
                "Explicit external stop reason and no follow-on launch",
            )
        a.need(
            preflight["status"] != "PASS" or external is not None,
            "Passed P0 followed by another failed launch gate must have supervisor evidence",
        )
        return dict(
            status="PASS_STOPPED"
            if prerequisites["accounting_complete"]
            else "INCOMPLETE_STOPPED_AUDIT",
            original_status=result["status"],
            original_prerequisite_status=preflight["status"],
            prerequisites=prerequisites,
            scientific_result=(
                "NOT_LAUNCHED_CAPACITY_STOP"
                if preflight["status"] == "PASS"
                else "NOT_LAUNCHED_PREREQUISITE_FAILED"
            ),
            criteria=None,
            new_research_calls_after_failed_prerequisite=0,
            external_supervisor=external,
            input_lineage=input_lineage,
            phase_work=prelaunch_phase_accounting(a, prerequisites, input_lineage, config),
        )
    a.need(preflight["status"] == "PASS", "Research execution required passed prerequisites")
    if result["status"] != "COMPLETED_20_ROUNDS":
        a.need(
            allow_stopped and result["status"] == "FAILED_STOPPED", "Closed complete run required"
        )
        coordinator = audit_ledger(a, run / "coordinator", coordinator=True)
        jobs = []
        for path in sorted(run.rglob("job.json")):
            directory = path.parent
            if (directory / "receipt.json").is_file():
                paid = audit_ledger(a, directory)
                if paid["receipt"]["status"] == "COMPLETED":
                    jobs.append(audit_native(a, directory))
                else:
                    jobs.append(
                        dict(
                            path=str(directory),
                            status="ORIGINAL_FAILED_PRESERVED",
                            counts=paid["counts"],
                        )
                    )
            else:
                jobs.append(dict(path=str(directory), status="UNKNOWN_PARTIAL_PRESERVED"))
        for path in run.rglob("*"):
            if path.is_file():
                a.bind(path)
        return dict(
            status="PASS_STOPPED",
            original_status=result["status"],
            scientific_result="INCOMPLETE_STOPPED",
            criteria=None,
            prerequisites=prerequisites,
            coordinator=coordinator,
            phase_work=phase_accounting(a, coordinator, preflight, input_lineage),
            input_lineage=input_lineage,
            jobs=jobs,
            scope="Closed work accounting and completed native outputs; incomplete fits/rounds remain unconfirmed",
        )
    a.need(
        len(result["history"]) == config["rounds"] + 1
        and [row["round"] for row in result["history"]] == list(range(config["rounds"] + 1)),
        "All alternating rounds completed without quality-dependent early stopping",
    )
    history, confirmation, fits = audit_rounds(a, run, config, launch, prepared, result["history"])
    a.equal(
        result["confirmation"],
        a.json(run / "final_panels.json"),
        "Final result confirmation binding",
    )
    criteria = independent_criteria(history, confirmation, config)
    compare_tree(a, result["criteria"], criteria, "Independent finite-loop criteria")
    coordinator = audit_ledger(a, run / "coordinator", coordinator=True)
    counts = coordinator["counts"]
    phase_work = phase_accounting(a, coordinator, preflight, input_lineage)
    a.equal(result["accounting"]["counts"], counts, "Root result complete actual work census")
    a.need(
        coordinator["receipt"]["status"] == "COMPLETED"
        and not coordinator["failed"]
        and not coordinator["rejected"],
        "Successful coordinator has no hidden failed calls",
    )
    jobs = {str(path.parent.resolve()) for path in run.rglob("job.json")}
    a.need(
        jobs == set(a.native) == set(coordinator["imported_workers"]),
        "Every actual worker is reported, independently audited and charged once",
    )
    a.need(
        len(fits) == 120 and sum(row["updates"] for row in fits) == 12000,
        "Three paired streams, twenty rounds and exactly 12,000 new Adam updates",
    )
    for unit, amount in (("optimizer_steps", 12000), ("backward_rows", 3072000)):
        a.need(
            counts["completed_" + unit] == amount + preflight["work"].get("completed_" + unit, 0),
            "Exact study/prerequisite work separation: " + unit,
        )
    for key, value in counts.items():
        if key.startswith("unknown_partial_"):
            a.need(value == 0, "No unknown partial work in a complete run")
    for unit in (
        "model_controls",
        "native_controls",
        "physics_substeps",
        "neural_rows",
        "backward_rows",
        "optimizer_steps",
    ):
        a.need(
            counts.get("requested_" + unit, 0) <= config["limits"][unit],
            "Global finite requested-work limit: " + unit,
        )
    a.need(
        coordinator["receipt"]["wall_seconds"] <= config["limits"]["execution_wall_seconds"],
        "Finite execution wall budget",
    )
    size = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    a.need(size <= config["limits"]["execution_output_bytes"], "Closed study output storage limit")
    for path in run.rglob("*"):
        if path.is_file():
            a.bind(path)
    return dict(
        status="PASS",
        scientific_result=criteria["verdict"],
        criteria=criteria,
        prerequisites=prerequisites,
        actual_work=counts,
        phase_work=phase_work,
        input_lineage=input_lineage,
        fit_blocks=len(fits),
        actor_episodes=sum(row["kind"] == "actor" for row in a.native.values()),
        solve_jobs=sum(row["kind"] == "solve" for row in a.native.values()),
        saved_full_banks=sum(row.get("full_banks", 0) for row in a.native.values()),
        saved_nominal_paths=sum(row.get("nominal_paths", 0) for row in a.native.values()),
        study_output_bytes=size,
        fit_proofs=fits,
        limitations=[
            "Saved-state algebra and source/optimizer/RNG bindings are verified; dynamics and gradient arithmetic are not replayed.",
            "All nominal scores and scheduled full acquisition banks are checked. Unsaved candidate physics and query populations cannot be independently replayed from these outputs.",
            "Independent-arm population D was deliberately unmeasured; it remains missing.",
            "Private PI seed identities and paired RNG snapshots are checked; the audit does not regenerate every sampled perturbation.",
            "Equal optimizer budgets are not equal total compute: agreement requires additional frozen-policy evaluations.",
            "Frozen incoming parents are not equal-compute cloning or DAgger controls; no paper reproduction or second-body success is claimed.",
        ],
    )


def phase_accounting(a: Audit, coordinator: dict, preflight: dict, lineage: dict | None) -> dict:
    """Partition previously paid prerequisites and actual study calls exactly once."""
    expected = [] if lineage is None else lineage["imported_work"]
    a.equal(
        coordinator["antecedents"],
        expected,
        "Exactly the frozen origin and worker-native prerequisite imports, never historical reruns",
    )
    a.equal(
        coordinator["prerequisite_counts"],
        preflight["work"],
        "P0 ledger contribution is its own paid work, not cumulative prior work",
    )
    known: dict[str, int] = {}
    phases = {row["phase"]: row["counts"] for row in expected}
    phases["p0"] = preflight["work"]
    for counts in phases.values():
        for key, value in counts.items():
            known[key] = known.get(key, 0) + value
    study = {key: value - known.get(key, 0) for key, value in coordinator["counts"].items()}
    a.need(
        all(value >= 0 for value in study.values())
        and all(coordinator["counts"].get(key, 0) >= value for key, value in known.items()),
        "No prerequisite overwritten or subtracted twice from complete paid work",
    )
    phases["study"] = study
    return dict(
        phases=phases,
        combined=coordinator["counts"],
        imports_counted_once=True,
        previous_p1_failure="Historical separately retained cost; excluded from this new R0/RV/P0/study budget scope",
        audit_scope="Independent audit performs only declared saved-array CPU/NumPy work; no repeated native/model physics, gradients or Adam updates",
    )


def prelaunch_phase_accounting(
    a: Audit, prerequisite: dict, lineage: dict | None, cfg: dict
) -> dict:
    """Retain R0/RV/P0 paid counts even if the alternating study never starts."""
    imports = [] if lineage is None else lineage["imported_work"]
    phases = {row["phase"]: row["counts"] for row in imports}
    phases["p0"] = prerequisite["work"]
    phases["study"] = {}
    combined: dict[str, int] = {}
    for values in phases.values():
        for key, value in values.items():
            combined[key] = combined.get(key, 0) + value
    for unit in (
        "model_controls",
        "native_controls",
        "physics_substeps",
        "neural_rows",
        "backward_rows",
        "optimizer_steps",
    ):
        a.need(
            combined.get("requested_" + unit, 0) <= cfg["limits"][unit],
            "Known stopped phase reservations remain within unchanged global ceiling: " + unit,
        )
    return dict(
        phases=phases,
        combined_known_ledger_counts=combined,
        accounting_complete=prerequisite["accounting_complete"],
        imports_counted_once=True,
        unknown_request_upper_charges=prerequisite.get("unresolved_request_upper_charges"),
        scope="No alternating study was entered. A missing external-stop closure remains incomplete; known prior paid phases are never erased or simulated again.",
    )


def compare_tree(a: Audit, actual: Any, expected: Any, message: str) -> None:
    if isinstance(expected, dict):
        a.need(isinstance(actual, dict) and set(actual) == set(expected), message + " keys")
        for key, value in expected.items():
            compare_tree(a, actual[key], value, message + "/" + str(key))
    elif isinstance(expected, list):
        a.need(isinstance(actual, list) and len(actual) == len(expected), message + " length")
        for i, value in enumerate(expected):
            compare_tree(a, actual[i], value, message + "/" + str(i))
    elif isinstance(expected, float):
        a.close(actual, expected, message, atol=1e-9)
    else:
        a.need(actual == expected, message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--stopped",
        action="store_true",
        help="Audit a preserved stop; never promote it to convergence evidence",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    a, findings, error = Audit(), {}, None
    try:
        a.bind(Path(__file__))
        findings = audit_run(a, args.run.resolve(), allow_stopped=args.stopped)
        forbidden = (
            "mujoco",
            "src.envs.",
            "src.mppi.",
            "src.gps.mppi_gps.policy",
            "src.gps.mppi_gps.trajectory_pi_study",
            "src.gps.mppi_gps.trajectory_pi_native",
            "src.gps.mppi_gps.trajectory_pi_student",
            "src.gps.mppi_gps.trajectory_pi_optimizer",
        )
        a.need(
            not any(name.startswith(forbidden) for name in sys.modules),
            "Independent audit imported no simulator, producer, optimizer or trainable policy",
        )
        for path in list(a.files):
            a.bind(Path(path))
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    receipt = dict(
        schema="trajectory-pi-p1-independent-audit-1",
        passed=error is None and findings.get("status") in ("PASS", "PASS_STOPPED"),
        status=findings.get("status", "FAIL"),
        error=error,
        checks=a.checks,
        bound_files=len(a.files),
        hashed_file_bytes=sum(r["bytes"] for r in a.files.values()),
        requested_numpy_head_rows=a.requested_rows,
        completed_numpy_head_rows=a.completed_rows,
        unknown_partial_numpy_head_rows=a.unknown_rows,
        new_model_controls=0,
        new_native_controls=0,
        new_physics_substeps=0,
        new_gradients=0,
        new_optimizer_updates=0,
        wall_seconds=time.monotonic() - a.started,
        cpu_seconds=time.process_time() - a.cpu_started,
        original_failure_preserved=bool(findings.get("status") == "PASS_STOPPED"),
    )
    for name, value in (
        ("findings.json", findings),
        ("file_bindings.json", a.files),
        ("receipt.json", receipt),
    ):
        encoded = json.dumps(value, indent=2, default=_plain, allow_nan=False) + "\n"
        if (
            sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
            + len(encoded.encode())
            > a.output_limit
        ):
            raise AuditFailure("Audit output budget exhausted before receipt")
        with (args.out / name).open("x") as stream:
            stream.write(encoded)
    print(json.dumps(receipt, allow_nan=False))
    return int(not receipt["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
