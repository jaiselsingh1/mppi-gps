"""Independent saved-evidence audit for J1; never repeats fitting or physics.

The common independent stream index/digest decoder is reused. Native outcome,
motion, action and occupancy measurements below are separate numerical oracles.
Only the declared functional checkpoint verification imports Torch, lazily.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .student_consequence import TRACE_KEYS, verify_trace_arrays
from .teacher_proposal import AuditError, StreamArchive, optimizer_fingerprint

MAX_READ = 32 * 1024**3
MAX_OUTPUT = 2 * 1024**3
MAX_WALL = 3 * 3600
MAX_NEURAL = 3_000_000
MAX_RSS = 4 * 1024**3
# The coordinator hashes source files before this saved-evidence worker starts.
# Reserve a separate, explicit share of the global 32 GiB audit-read ceiling.
COORDINATOR_READ_RESERVE = 128 * 1024**2


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(data: bytes | str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "Duplicate J1 JSON key")
            result[key] = value
        return result

    return json.loads(
        data,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(AuditError("Nonfinite J1 JSON")),
    )


def _identity(stat: os.stat_result) -> tuple:
    return stat.st_dev, stat.st_ino, stat.st_mode, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _same(left: Any, right: Any, label: str, tolerance: float = 1e-9) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        _need(set(left) == set(right), "J1 dictionary roster differs: " + label)
        for key in left:
            _same(left[key], right[key], label + "/" + str(key), tolerance)
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        _need(len(left) == len(right), "J1 list length differs: " + label)
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            _same(a, b, label + "/" + str(i), tolerance)
    elif type(left) is bool or left is None or isinstance(left, str):
        _need(type(left) is type(right) and left == right, "J1 value differs: " + label)
    elif type(left) is int:
        _need(type(right) is int and left == right, "J1 integer differs: " + label)
    else:
        _need(
            type(right) in (int, float)
            and math.isfinite(float(right))
            and math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance),
            "J1 numeric differs: " + label,
        )


class Reader:
    """Contain every input and charge metadata/checkpoint/stream reads together."""

    def __init__(
        self, root: Path, out: Path, *, read_limit: int = MAX_READ, wall_seconds: float = MAX_WALL
    ):
        _need(
            0 < read_limit <= MAX_READ and 0 < wall_seconds <= MAX_WALL,
            "J1 audit allocation exceeds limits",
        )
        self.root, self.out = root.resolve(), out.resolve()
        self.started = time.monotonic()
        self.deadline = self.started + wall_seconds
        self.read_limit, self.read_bytes = read_limit, 0
        self.archive: StreamArchive | None = None
        self.files: dict[str, dict] = {}
        self.identities: dict[Path, tuple] = {}

    def check(self) -> None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if sys.platform == "darwin" else rss * 1024
        _need(
            time.monotonic() <= self.deadline and rss <= MAX_RSS,
            "J1 audit wall/RSS allocation exhausted",
        )
        _need(
            self.total_reads <= self.read_limit, "J1 combined metadata/stream read budget exhausted"
        )

    @property
    def total_reads(self) -> int:
        return self.read_bytes + (self.archive.read_bytes if self.archive else 0)

    def path(self, value: str | Path) -> Path:
        source = Path(value)
        path = source if source.is_absolute() else self.root / source
        _need(
            path.resolve().is_relative_to(self.root) and not path.is_symlink() and path.is_file(),
            "Missing/escaped J1 input: " + str(value),
        )
        return path

    def read(self, value: str | Path | dict, *, maximum: int = 64 * 1024**2) -> bytes:
        self.check()
        binding = value if isinstance(value, dict) else None
        path = self.path(binding["file"] if binding else value)
        before = path.stat()
        _need(
            path not in self.identities or self.identities[path] == _identity(before),
            "J1 input changed between audited reads",
        )
        _need(
            before.st_size <= maximum and self.total_reads + before.st_size <= self.read_limit,
            "J1 read limit before input",
        )
        payload = path.read_bytes()
        self.read_bytes += len(payload)
        if self.archive is not None:
            self.archive.max_read_bytes = self.read_limit - self.read_bytes
        _need(
            len(payload) == before.st_size and _identity(path.stat()) == _identity(before),
            "J1 input changed during read",
        )
        actual = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        if binding:
            _need(
                binding.get("bytes", actual["bytes"]) == actual["bytes"]
                and binding.get("sha256") == actual["sha256"],
                "J1 input byte binding differs",
            )
        self.files[str(path.relative_to(self.root))] = actual
        self.identities[path] = _identity(before)
        return payload

    def verify_unchanged(self) -> None:
        for path, expected in self.identities.items():
            _need(
                _identity(path.stat()) == expected, "J1 audited metadata/checkpoint input changed"
            )

    def json(self, value: str | Path | dict) -> Any:
        return _json(self.read(value))

    def array(self, reference: dict, key: str) -> np.ndarray:
        self.check()
        _need(self.archive is not None, "J1 stream index must be open")
        size = reference["arrays"][key]["byte_length"]
        _need(self.total_reads + size <= self.read_limit, "J1 array read budget exhausted")
        return self.archive.array(reference, key)

    def arrays(self, reference: dict) -> dict[str, np.ndarray]:
        return {key: self.array(reference, key) for key in reference["arrays"]}

    def npy(self, binding: str | dict) -> np.ndarray:
        value = np.load(io.BytesIO(self.read(binding)), allow_pickle=False)
        _need(isinstance(value, np.ndarray) and not value.dtype.hasobject, "J1 plain NPY required")
        if isinstance(binding, dict):
            _need(
                binding.get("array_sha256")
                == hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                "J1 array C-byte binding differs",
            )
            _need(
                binding.get("shape") == list(value.shape)
                and binding.get("dtype") == str(value.dtype),
                "J1 array dtype/shape binding differs",
            )
        return value


def native_outcome(arrays: Mapping[str, np.ndarray], *, task: dict, horizon: int = 1000) -> dict:
    """Independent state/integration/observation/cost equations, no dynamics call."""
    metadata = _json(np.asarray(arrays["metadata"]).item())
    _need(
        type(metadata.get("integration_spec")) is int
        and metadata["integration_spec"] == 16383
        and metadata.get("dimensions") == [9, 9, 6],
        "J1 trace lacks the pinned full MuJoCo 3.6 integration-state contract",
    )
    states, actions = np.asarray(arrays["states"]), np.asarray(arrays["actions"])
    n = len(actions)
    _need(states.shape == (n + 1, 19) and 0 < n <= horizon, "J1 actual state/action census")
    integration, warm = arrays["integrations"], arrays["warmstarts"]
    _need(
        integration.shape == (n + 1, 91) and warm.shape == (n + 1, 9),
        "J1 full integration geometry differs",
    )
    _need(
        np.array_equal(integration[:, :19], states) and np.array_equal(integration[:, 19:28], warm),
        "J1 integration state/warmstart mismatch",
    )
    _need(
        np.array_equal(integration[1:, 28:34], actions),
        "J1 post-control integration torque differs",
    )
    _need(
        np.allclose(np.diff(states[:, 0]), 0.008, rtol=0, atol=1e-10),
        "J1 actual control timing differs",
    )
    observations = np.hstack((states[:, 2:10], np.clip(states[:, 10:19], -10, 10)))
    _need(
        np.array_equal(observations, arrays["observations"]),
        "J1 student observations differ from physical state",
    )
    post = states[1:]
    healthy = (post[:, 2] > 0.8) & (post[:, 2] < 2) & (post[:, 3] > -1) & (post[:, 3] < 1)
    done = np.asarray(arrays["done"], dtype=bool)
    _need(
        np.array_equal(done, ~healthy) and not np.any(done[:-1]),
        "J1 termination/health/post-fall rows differ",
    )
    velocity = post[:, 10]
    _need(np.array_equal(velocity, arrays["x_velocities"]), "J1 native velocity differs")
    running = (
        task["vel_cost_weight"] * np.abs(task["target_velocity"] - velocity)
        + task["vertical_velocity_cost_weight"] * post[:, 11] ** 2
        + task["angle_cost_weight"] * post[:, 3] ** 2
        + task["unhealthy_cost_weight"] * (~healthy)
        + task["ctrl_cost_weight"] * np.sum(actions**2, axis=1)
    )
    terminal = task["terminal_unhealthy_weight"] * (~healthy).astype(float)
    _need(
        task["cost_style"] == "target_velocity"
        and task["terminal_unhealthy_weight"] == 20
        and task["unhealthy_cost_weight"] == 5,
        "J1 task changed",
    )
    _need(
        np.array_equal(terminal, arrays["fall_costs"]) and np.count_nonzero(terminal) <= 1,
        "J1 native terminal must be charged once",
    )
    _need(
        np.allclose(running + terminal, arrays["native_costs"], rtol=0, atol=1e-10),
        "J1 native task costs differ",
    )
    fell = bool(done[-1])
    _need(n == horizon or fell, "J1 normally closed short trace has no terminal fall")
    survived = n == horizon and not fell
    total = float(np.sum(arrays["native_costs"], dtype=np.float64))
    terminals = float(np.sum(terminal, dtype=np.float64))
    mae = float(np.mean(np.abs(velocity - 1.5)))
    tail = float(5 * (horizon - n)) if fell else 0.0
    return dict(
        controls=n,
        done=fell,
        survived=bool(survived),
        walking=bool(horizon == 1000 and survived and mae <= 0.3),
        mae=mae,
        native_cost=total,
        raw_cost=total - terminals,
        native_terminal=terminals,
        absorbing_cost=tail,
        cost=total + tail,
    )


def motion_oracle(
    teacher: Mapping[str, np.ndarray], student: Mapping[str, np.ndarray], horizon: int = 1000
) -> dict:
    t, s = teacher["states"], student["states"]
    n = min(len(t), len(s)) - 1
    _need(n > 0 and n <= horizon, "Missing actual motion prefix")
    _need(
        np.allclose(t[: n + 1, 0] - t[0, 0], s[: n + 1, 0] - s[0, 0], rtol=0, atol=1e-10),
        "J1 motion timing differs",
    )
    progress = t[1 : n + 1, 1] - t[0, 1] - (s[1 : n + 1, 1] - s[0, 1])
    height = t[1 : n + 1, 2] - s[1 : n + 1, 2]
    angle = np.mod(t[1 : n + 1, 3:10] - s[1 : n + 1, 3:10] + np.pi, 2 * np.pi) - np.pi
    velocity = t[1 : n + 1, 10:19] - s[1 : n + 1, 10:19]
    raw = np.stack(
        [
            progress**2,
            height**2 / 0.25**2,
            np.mean(angle**2, axis=1) / 0.5**2,
            np.mean(velocity**2, axis=1) / 3**2,
        ],
        axis=1,
    )
    error = float(
        np.sqrt(
            np.minimum(raw, 9).sum(dtype=np.float64) / (4 * horizon) + 4 * (horizon - n) / horizon
        )
    )
    full = (
        len(t) == len(s) == horizon + 1
        and not np.any(teacher["done"])
        and not np.any(student["done"])
    )
    joint = float(np.sqrt(np.mean(angle**2)))
    vx = float(np.sqrt(np.mean(velocity[:, 0] ** 2)))
    h = float(np.sqrt(np.mean(height**2)))
    final = float(abs(progress[-1])) if full else None
    return dict(
        error=error,
        reproduced=bool(full and joint <= 0.25 and vx <= 0.3 and h <= 0.1 and final <= 0.5),
        joint_angle_rms=joint,
        forward_velocity_rms=vx,
        height_rms=h,
        final_relative_progress_error=final,
        common_controls=n,
        missing_fraction=(horizon - n) / horizon,
    )


def score_oracle(
    teacher: Mapping[str, np.ndarray],
    incoming: Mapping[str, np.ndarray],
    predictions: np.ndarray,
    models: dict,
) -> dict:
    """A/B/D/E are separate oracles; C reuses the declared pure explanatory formula."""
    from ..similarity_metrics import state_explained_score

    actions = np.asarray(teacher["actions"])
    observations = teacher["observations"][:-1]
    _need(actions.shape == predictions.shape == (1000, 6), "J1 preprediction row ownership differs")
    a = float(np.square(actions - predictions).sum(dtype=np.float64) / 6000)
    kernel = np.ones(5) / 5
    teacher_filtered = np.column_stack(
        [np.convolve(actions[:, j], kernel, mode="valid") for j in range(6)]
    )
    policy_filtered = np.column_stack(
        [np.convolve(predictions[:, j], kernel, mode="valid") for j in range(6)]
    )
    b = float(np.mean((teacher_filtered - policy_filtered) ** 2))
    c = state_explained_score(observations, actions, predictions, models["explainer"])["score"]
    d = motion_oracle(teacher, incoming)["error"]
    occupancy = models["occupancy_normalizer"]
    generator = np.random.Generator(np.random.PCG64(2026129801))
    directions = generator.normal(size=(32, 23))
    directions /= np.sqrt((directions**2).sum(axis=1, keepdims=True))
    _need(
        np.allclose(directions, occupancy["directions"], rtol=0, atol=1e-14),
        "J1 occupancy innovations/directions changed",
    )
    mean, std = np.array(occupancy["mean"]), np.array(occupancy["std"])
    teacher_cloud = np.column_stack((observations, actions))
    student_cloud = np.column_stack((incoming["observations"][:-1], incoming["actions"]))
    _need(
        len(student_cloud) > 0 and mean.shape == std.shape == (23,) and np.all(std >= 0.05),
        "J1 occupancy scale/rows differ",
    )
    left = np.sort(((teacher_cloud - mean) / std) @ directions.T, axis=0)
    right = np.sort(((student_cloud - mean) / std) @ directions.T, axis=0)
    quantiles = (np.arange(1000) + 0.5) / 1000

    def interpolate(values):
        position = quantiles * (len(values) - 1)
        lower = np.floor(position).astype(int)
        upper = np.ceil(position).astype(int)
        fraction = position - lower
        return values[lower] * (1 - fraction[:, None]) + values[upper] * fraction[:, None]

    e = float(
        np.mean((interpolate(left) - interpolate(right)) ** 2)
        + 4 * (1000 - len(student_cloud)) / 1000
    )
    return dict(A=a, B=b, C=c, D=d, E=e)


def metric_trace(arrays: Mapping[str, np.ndarray]) -> dict:
    return dict(
        states=arrays["states"],
        done=np.asarray(arrays["done"], dtype=bool),
        observations=arrays["observations"][:-1],
        actions=arrays["actions"],
    )


class Actors:
    """Bound CPU functional actor replicas; no network/optimizer constructors."""

    def __init__(self, reader: Reader, maximum: int = MAX_NEURAL):
        _need(0 <= maximum <= MAX_NEURAL, "J1 audit neural allocation exceeds ceiling")
        self.reader, self.maximum = reader, maximum
        self.payloads: dict[str, dict] = {}
        self.requested = self.completed = self.calls = self.loads = self.unknown = 0

    def load(self, binding: dict) -> dict:
        path = str(self.reader.path(binding["file"]).relative_to(self.reader.root))
        if path in self.payloads:
            payload = self.payloads[path]
            _need(
                binding["sha256"] == self.reader.files[path]["sha256"],
                "J1 checkpoint has conflicting bindings",
            )
            return payload
        _need(self.loads < 97, "J1 audit checkpoint census exceeds 32 parents/64children/reference")
        data = self.reader.read(binding, maximum=16 * 1024**2)
        import torch

        torch.set_num_threads(1)
        self.loads += 1
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        reference = "policy" in payload and "actor" in payload["policy"]
        state = payload["policy"]["actor"] if reference else payload["state_dict"]
        if not reference:
            _need(payload["schema"] == "walker-tanh-policy-v1", "J1 checkpoint schema changed")
            expected = {
                "observation_mean": (17,),
                "observation_std": (17,),
                "network.0.weight": (128, 17),
                "network.0.bias": (128,),
                "network.2.weight": (128, 128),
                "network.2.bias": (128,),
                "network.4.weight": (6, 128),
                "network.4.bias": (6,),
            }
            _need(set(state) == set(expected), "J1 actor tensor/architecture roster changed")
            digest = hashlib.sha256(b"walker-tanh-policy-v1")
            for name, tensor in sorted(state.items()):
                _need(
                    tuple(tensor.shape) == expected[name]
                    and tensor.dtype == torch.float32
                    and bool(torch.isfinite(tensor).all()),
                    "J1 actor tensor type differs",
                )
                array = tensor.cpu().contiguous().numpy()
                digest.update(
                    json.dumps(
                        [name, list(array.shape), str(array.dtype)], separators=(",", ":")
                    ).encode()
                )
                digest.update(array.tobytes())
            _need(
                digest.hexdigest()
                == payload["policy_hash"]
                == binding.get("policy_hash", payload["policy_hash"]),
                "J1 named actor tensor hash differs",
            )
            _need(bool((state["observation_std"] > 0).all()), "J1 normalization is not positive")
        self.payloads[path] = payload
        return payload

    def verify(
        self, binding: dict, observations: np.ndarray, expected: np.ndarray, *, batch_one: bool
    ) -> None:
        payload = self.load(binding)
        count = len(observations)
        _need(
            self.requested + count <= self.maximum,
            "J1 audit neural budget exhausted before forward",
        )
        import torch
        import torch.nn.functional as functional

        reference = "policy" in payload and "actor" in payload["policy"]
        state = payload["policy"]["actor"] if reference else payload["state_dict"]
        observations = np.asarray(observations, dtype=np.float32)
        _need(
            expected.shape == (count, 6)
            and expected.dtype == np.dtype(np.float64 if batch_one else np.float32),
            "J1 native/pre-score prediction dtype and batch shape differ",
        )
        values = []
        with torch.no_grad():
            batches = (
                (observations[j : j + 1] if reference else observations[j] for j in range(count))
                if batch_one
                else [observations]
            )
            for batch in batches:
                rows = 1 if batch_one else count
                self.requested += rows
                try:
                    self.reader.check()
                    self.calls += 1
                    value = torch.as_tensor(np.array(batch, copy=True))
                    if reference:
                        value = (value - state["obs_mean"]) / state["obs_std"]
                        value = functional.relu(
                            functional.linear(value, state["fc1.weight"], state["fc1.bias"])
                        )
                        value = functional.relu(
                            functional.linear(value, state["fc2.weight"], state["fc2.bias"])
                        )
                        mean = functional.linear(
                            value, state["fc_mean.weight"], state["fc_mean.bias"]
                        )
                        _ = -5 + 3.5 * (
                            torch.tanh(
                                functional.linear(
                                    value, state["fc_logstd.weight"], state["fc_logstd.bias"]
                                )
                            )
                            + 1
                        )
                        value = torch.tanh(mean) * state["action_scale"] + state["action_bias"]
                    else:
                        value = (value - state["observation_mean"]) / state["observation_std"]
                        for layer in (0, 2, 4):
                            value = torch.tanh(
                                functional.linear(
                                    value,
                                    state[f"network.{layer}.weight"],
                                    state[f"network.{layer}.bias"],
                                )
                            )
                    values.append(value.cpu().numpy().reshape(-1, 6))
                    self.completed += rows
                except BaseException:
                    self.unknown += rows
                    raise
        actual = np.concatenate(values)
        _need(
            actual.astype(expected.dtype).tobytes() == np.ascontiguousarray(expected).tobytes(),
            "J1 saved actor actions differ at their declared batching",
        )


def _policy_array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.float32)
    digest = hashlib.sha256(json.dumps([list(value.shape), str(value.dtype)]).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def verify_learning_case(
    reader: Reader, actors: Actors, case: dict, teacher: Mapping[str, np.ndarray]
) -> dict:
    """Authenticate fitted endpoints/tapes/Adam/RNG without applying an update."""
    bindings = case["incoming_bindings"]
    parent_binding = bindings["checkpoint"]
    child_binding = case["child_checkpoint"]
    _same(case["parent_checkpoint"], parent_binding, "case incoming checkpoint binding", 0)
    tape_file = case["tape_file"]
    tape_file = tape_file["file"] if isinstance(tape_file, dict) else tape_file
    _need(
        reader.path(tape_file) == reader.path(bindings["tape"]["file"]),
        "J1 declared minibatch tape is not the incoming package tape",
    )
    parent, child = actors.load(parent_binding), actors.load(child_binding)
    with np.load(
        io.BytesIO(reader.read("prepared/normalization.npz")), allow_pickle=False
    ) as saved_norm:
        for key in ("mean", "std"):
            _need(
                parent["state_dict"]["observation_" + key].numpy().tobytes()
                == np.asarray(saved_norm[key], dtype=np.float32).tobytes(),
                "J1 parent changed the inherited W1 normalization",
            )
    _need(
        parent["seed"] == bindings["seed"]
        and parent["optimizer_state"]["updates"] == 0
        and not parent["optimizer_state"]["adam"]["state"],
        "J1 parent is not the declared fresh empty-Adam package",
    )
    for key, value in (
        ("optimizer_fingerprint", parent["optimizer_state"]),
        ("private_rng_fingerprint", parent["private_rng_state"]),
    ):
        _need(
            optimizer_fingerprint(value) == bindings[key],
            "J1 incoming state fingerprint differs: " + key,
        )
    global_state = reader.npy(bindings["global_cpu_rng"])
    _need(
        optimizer_fingerprint(global_state) == bindings["global_rng_fingerprint"],
        "J1 global CPU RNG binding differs",
    )
    tape = reader.npy(bindings["tape"])
    _need(
        tape.shape == (2000, 256)
        and tape.dtype == np.int64
        and tape.min() >= 0
        and tape.max() < 1000,
        "J1 unique-row/minibatch tape shape/range differs",
    )
    rng = np.random.Generator(np.random.PCG64(bindings["fitseed"]))
    _need(
        rng.bit_generator.state == bindings["initial_pcg64"] == parent["optimizer_state"]["rng"],
        "J1 initial private PCG differs",
    )
    for row in tape:
        _need(
            np.array_equal(row, rng.integers(0, 1000, size=256)),
            "J1 presealed minibatch stream differs",
        )
    _need(rng.bit_generator.state == bindings["terminal_pcg64"], "J1 terminal tape PCG differs")
    receipt = reader.json(case["fit_receipt_file"])
    _need(
        receipt["schema"] == "walker-j1-learning-case-fit-v1"
        and receipt["updates"] == 2000
        and receipt["all2000minibatches_and_rng_matched"] is True
        and receipt["parent_unchanged"] is True
        and receipt["global_rng_unchanged"] is True,
        "J1 fixed-update fit certification differs",
    )
    _need(
        receipt["incoming_actor_hash"] == parent["policy_hash"]
        and receipt["child_actor_hash"] == child["policy_hash"]
        and receipt["incoming_optimizer_fingerprint"] == bindings["optimizer_fingerprint"]
        and receipt["child_optimizer_fingerprint"]
        == optimizer_fingerprint(child["optimizer_state"]),
        "J1 endpoint/optimizer ownership differs",
    )
    _same(receipt["checkpoint"], child_binding, "child checkpoint binding", 0)
    _need(
        receipt["tape_sha256"] == bindings["tape"]["sha256"]
        and receipt["observed_minibatch_hashes"]
        == [hashlib.sha256(row.tobytes()).hexdigest() for row in tape],
        "J1 observed fit minibatches differ from frozen tape",
    )
    state = child["optimizer_state"]
    _need(
        state["anchor_policy_hash"] == parent["policy_hash"]
        and optimizer_fingerprint(state["anchor"]) == optimizer_fingerprint(parent["state_dict"]),
        "J1 fixed proximal anchor differs from incoming actor",
    )
    _need(
        state["updates"] == state["fit_updates"] == 2000
        and state["rng"] == bindings["terminal_pcg64"]
        and state["parent_policy_hash"] == parent["policy_hash"]
        and state["policy_hash"] == child["policy_hash"],
        "J1 Adam/RNG continuation counters differ",
    )
    expected_config = dict(
        batch_size=256,
        lr=0.0003,
        eta_theta=0.0,
        gradient_clip=0.5,
        prox_reduction="sum",
        anchor_refresh=0,
    )
    _same(state["fit_config"], expected_config, "fit settings", 0)
    _need(
        state["fit_data"]
        == dict(
            observations_sha256=_policy_array_hash(teacher["observations"][:-1]),
            actions_sha256=_policy_array_hash(teacher["actions"]),
            rows=1000,
        ),
        "J1 fit rows/targets differ from complete executed corpus",
    )
    _need(
        optimizer_fingerprint(parent["private_rng_state"])
        == optimizer_fingerprint(child["private_rng_state"]),
        "J1 private policy RNG changed during deterministic fitting",
    )
    for name in ("observation_mean", "observation_std"):
        _need(
            parent["state_dict"][name].numpy().tobytes()
            == child["state_dict"][name].numpy().tobytes(),
            "J1 fitting changed normalization",
        )
    for value in state["adam"]["state"].values():
        _need(
            float(value["step"]) == 2000
            and bool(np.isfinite(value["exp_avg"].numpy()).all())
            and bool(np.isfinite(value["exp_avg_sq"].numpy()).all()),
            "J1 Adam moments/step are invalid",
        )
    _need(
        len(state["adam"]["state"]) == 6 and len(state["adam"]["param_groups"]) == 1,
        "J1 Adam parameter roster differs from width128 actor",
    )
    for parameter_group in (
        parent["optimizer_state"]["adam"]["param_groups"][0],
        state["adam"]["param_groups"][0],
    ):
        for key, expected in dict(
            lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=False, fused=False
        ).items():
            _need(parameter_group[key] == expected, "J1 Adam setting differs: " + key)
    curve_path = reader.path(case["fit_receipt_file"]).with_name("learning_curve.json")
    curve = reader.json(curve_path)
    _need(
        curve["completed_updates"] == 2000 and len(curve["learning_curve"]) == 2000,
        "J1 curve does not cover every update",
    )
    for index, row in enumerate(curve["learning_curve"]):
        _need(
            row["update"] == row["fit_update"] == index + 1
            and all(
                math.isfinite(row[k])
                for k in ("action_mse", "proximal_loss", "total_loss", "unclipped_gradient_norm")
            ),
            "J1 learning curve update/scalar differs",
        )
    return dict(
        case_id=case["case_id"],
        updates=2000,
        backward_rows=512000,
        checkpoint_binding_verified=True,
        optimizer_rng_tape_verified=True,
        learning_curve_points=2000,
        fitting_repeated=False,
    )


class Ledger:
    """Independent ownership/cost census, keeping only one active native prefix."""

    def __init__(
        self,
        reader: Reader,
        task: dict,
        traces: list[dict],
        actors: Actors | None,
        actor_by_tag: dict,
    ):
        self.reader, self.task, self.actors, self.actor_by_tag = reader, task, actors, actor_by_tag
        self.traces = {row["tag"]: row for row in traces}
        _need(len(self.traces) == len(traces), "Duplicated J1 trace tags")
        self.pending: dict[int, dict] = {}
        self.requested = Counter()
        self.completed = Counter()
        self.events: dict[str, dict] = {}
        self.trace_outcomes: dict[str, dict] = {}
        self.active: dict | None = None
        self.opened = self.closed = 0
        self.serial = 0
        self.rows = 0

    def event(self, row: dict) -> None:
        self.reader.check()
        self.rows += 1
        _need(self.rows <= 16_000_000, "J1 journal row ceiling exceeded")
        event = row["event"]
        _need(not self.closed, "J1 records after runtime closure")
        if event == "runtime_open":
            self.opened += 1
        elif event in {"paid_requested", "work_requested"}:
            operation = row["operation"]
            _need(
                operation == self.serial + 1 and operation not in self.pending,
                "J1 paid request order/owner differs",
            )
            self.serial = operation
            counts = (
                row["counts"] if event == "paid_requested" else {row["category"]: row["amount"]}
            )
            if event == "work_requested" and row["category"] == "neural_rows":
                counts["prediction_calls"] = 1
            _need(
                all(type(v) is int and v >= 0 for v in counts.values()),
                "J1 negative/untyped work count",
            )
            self.pending[operation] = counts
            self.requested.update(counts)
        elif event in {"paid_complete", "work_complete"}:
            operation = row["operation"]
            _need(operation in self.pending, "J1 orphan paid completion")
            counts = row["counts"] if event == "paid_complete" else {row["category"]: row["amount"]}
            if event == "work_complete" and row["category"] == "neural_rows":
                counts["prediction_calls"] = 1
            _need(counts == self.pending.pop(operation), "J1 requested/completed work differs")
            self.completed.update(counts)
        elif event in {"paid_error", "work_error", "trace_error"}:
            raise AuditError("J1 failed/uncertain paid work prevents full certification")
        elif event == "trace_begin":
            _need(self.active is None, "Overlapping J1 native traces")
            self.active = dict(
                alias=row["alias"], horizon=row["horizon"], actions=[], digests=[], intent=None
            )
        elif event == "trace_intent":
            _need(
                self.active is not None
                and self.active["alias"] == row["alias"]
                and self.active["intent"] is None
                and row["ordinal"] == len(self.active["actions"]),
                "J1 native intent ownership differs",
            )
            self.active["intent"] = np.asarray(row["action"], dtype=np.float64)
        elif event == "trace_returned":
            _need(
                self.active is not None
                and self.active["alias"] == row["alias"]
                and self.active["intent"] is not None
                and row["ordinal"] == len(self.active["actions"]),
                "J1 native return ownership differs",
            )
            self.active["actions"].append(self.active.pop("intent"))
            self.active["intent"] = None
            self.active["digests"].append(row["digest"])
        elif event == "trace_complete":
            _need(
                self.active is not None
                and self.active["alias"] == row["alias"]
                and self.active["intent"] is None,
                "J1 trace closure ownership differs",
            )
            saved = self.traces[row["alias"]]
            reference = saved["trace"]
            _need(
                reference["group"] == row["group"]
                and hashlib.sha256(_canonical(reference)).hexdigest() == row["reference_sha256"],
                "J1 trace reference/index differs",
            )
            arrays = self.reader.arrays(reference)
            summary = verify_trace_arrays(arrays, task=self.task)
            _need(
                summary["normally_closed"] and not summary["unknown_partial_controls"],
                "J1 uncertain native prefix",
            )
            _need(
                arrays["actions"].tobytes()
                == np.asarray(self.active["actions"], dtype=np.float64).tobytes()
                and [x.tobytes().hex() for x in arrays["row_digests"]] == self.active["digests"],
                "J1 native intent/return digest chain differs",
            )
            outcome = native_outcome(arrays, task=self.task, horizon=self.active["horizon"])
            for name, value in outcome.items():
                _same(value, saved[name], "native outcome/" + name, 1e-10)
            contacts = self.reader.arrays(saved.get("contact_trace", saved.get("contact_ref")))
            n = outcome["controls"]
            presence = contacts["contacts"]
            forces = contacts.get("contact_normal_forces", contacts.get("forces"))
            _need(
                presence.shape == (n, 2)
                and presence.dtype.kind == "b"
                and forces.shape == (n, 2)
                and np.isfinite(forces).all()
                and np.all(forces >= 0),
                "J1 contact measurement dtype/census differs",
            )
            _need(
                np.array_equal(contacts["control_ordinals"], np.arange(n))
                and np.all(forces[~presence] == 0),
                "J1 contact/force row binding differs",
            )
            if self.actors is not None and saved["tag"] in self.actor_by_tag:
                self.actors.verify(
                    self.actor_by_tag[saved["tag"]],
                    arrays["observations"][:-1],
                    arrays["actions"],
                    batch_one=True,
                )
            self.trace_outcomes[saved["tag"]] = dict(
                outcome,
                initial_context_sha256={
                    key: hashlib.sha256(np.asarray(arrays[key][0]).tobytes()).hexdigest()
                    for key in ("states", "warmstarts", "integrations", "observations")
                },
            )
            self.active = None
        elif event in {"j1_scores_sealed", "j1_fit_begin", "j1_prediction_models_sealed"}:
            self.events[row["id"]] = row
        elif event == "runtime_close":
            self.closed += 1

    def finish(self, worker: dict, limits: dict) -> dict:
        _need(
            self.opened == self.closed == 1 and not self.pending and self.active is None,
            "J1 journal has unclosed work/native prefix",
        )
        _need(set(self.trace_outcomes) == set(self.traces), "J1 listed/closed native traces differ")
        costs = worker["costs"]
        _need(
            set(costs["requested"]) == set(self.requested)
            and set(costs["completed"]) == set(self.completed),
            "J1 worker has unaccounted work categories",
        )
        _need(
            not costs["unknown_partial"] and not costs["returned_unpersisted"],
            "J1 worker has unknown/unpersisted work",
        )
        for key, value in self.requested.items():
            _need(
                value
                == self.completed[key]
                == costs["requested"].get(key)
                == costs["completed"].get(key),
                "J1 complete work census differs: " + key,
            )
            _need(value <= limits[key], "J1 work limit exceeded: " + key)
        _need(
            self.completed["native_controls"]
            == sum(v["controls"] for v in self.trace_outcomes.values()),
            "J1 native control/outcome census differs",
        )
        _need(
            self.completed["physics_substeps"]
            == 4 * (self.completed["native_controls"] + self.completed["model_controls"]),
            "J1 combined physics substep census differs",
        )
        return dict(
            requested=dict(self.requested),
            completed=dict(self.completed),
            ledger_rows=self.rows,
            closed_traces=len(self.trace_outcomes),
            unknown_partial={},
            returned_unpersisted={},
        )


def audit_run(run: Path, out: Path) -> dict:
    """Create-only bounded J1 audit; failure preserves observations without retry."""
    run, out = Path(run).resolve(), Path(out).resolve()
    _need(
        run.is_dir() and not out.is_relative_to(run),
        "J1 audit must not write inside its preserved run",
    )
    out.mkdir(parents=True, exist_ok=False)
    reader = Reader(run, out)
    actors: Actors | None = None
    archive = None
    ledger = None
    result = dict(
        schema="walker-similarity-j1-audit-v1",
        passed=False,
        full_scientific_audit_passed=False,
        new_physics_controls=0,
        new_optimizer_steps=0,
        automatic_guidance_authorized=False,
    )
    try:
        config = reader.json("config.json")
        manifest = reader.json("manifest.json")
        _need(
            config["schema"] == "walker-similarity-j1-execution-v1"
            and config["automatic_followup"] is False
            and config["guidance_launch_authorized"] is False,
            "J1 scope/config schema changed",
        )
        limits = config["limits"]
        _need(
            limits["audit_read_bytes"] <= MAX_READ
            and limits["audit_stage_wall_seconds"] <= MAX_WALL
            and limits["audit_output_bytes"] <= MAX_OUTPUT
            and limits["audit_neural_rows"] <= MAX_NEURAL,
            "J1 audit allocation exceeds frozen ceiling",
        )
        _need(
            limits["audit_read_bytes"] > COORDINATOR_READ_RESERVE,
            "J1 audit read allocation lacks its coordinator reserve",
        )
        reader.read_limit = limits["audit_read_bytes"] - COORDINATOR_READ_RESERVE
        reader.deadline = reader.started + limits["audit_wall_seconds"]
        _need(
            manifest["config_sha256"] == reader.files["config.json"]["sha256"],
            "J1 source/config binding differs",
        )
        for item in manifest["files"]:
            reader.read(dict(file="source/" + item["file"], sha256=item["sha256"]))
        protocol_path = "source/" + config["protocol"]
        _need(
            reader.files[protocol_path]["sha256"] == config["protocol_sha256"],
            "J1 declared protocol is not the preserved source protocol",
        )
        xml_path = "source/" + config["inputs"]["xml"]["file"]
        _need(
            reader.files[xml_path]["sha256"] == config["inputs"]["xml"]["sha256"],
            "J1 native Walker source differs from the pinned XML",
        )
        input_manifest = reader.json("input_manifest.json")
        for name, binding in input_manifest["files"].items():
            reader.read(dict(binding, file=name))
        for role, filename in (
            ("reference", "reference.pt"),
            ("normalization", "normalization.npz"),
        ):
            binding = reader.files["prepared/" + filename]
            _need(
                all(binding[key] == config["inputs"][role][key] for key in ("bytes", "sha256")),
                "J1 prepared input differs from the declared pin: " + role,
            )
        worker = reader.json("worker_result.json")
        _need(
            worker["config_sha256"] == reader.files["config.json"]["sha256"]
            and worker["source_manifest_sha256"] == reader.files["manifest.json"]["sha256"]
            and worker["input_manifest_sha256"] == reader.files["input_manifest.json"]["sha256"],
            "J1 worker/config/source/input joins differ",
        )
        closed = worker["status"] in {
            "COMPLETED_J1_FIXED_VERSION",
            "CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE",
        }
        result["worker_status"] = worker["status"]
        traces = reader.json("worker/traces.json")
        cases = reader.json("worker/cases.json")
        _need(
            isinstance(traces, list) and isinstance(cases, list),
            "J1 trace/case catalog types differ",
        )
        actor_by_tag = {}
        for case in cases:
            binding = case["incoming_bindings"]["checkpoint"]
            actor_by_tag[case["incoming"]["tag"]] = binding
            if case.get("eligible", False) and "child_checkpoint" in case:
                for outcome in [case["reproduction"], *case["evaluations"]]:
                    actor_by_tag[outcome["tag"]] = case["child_checkpoint"]
        reference_binding = dict(
            input_manifest["files"]["prepared/reference.pt"], file="prepared/reference.pt"
        )
        for trace in traces:
            if trace["tag"].endswith("_reference"):
                actor_by_tag[trace["tag"]] = reference_binding
        actors = Actors(reader, maximum=limits["audit_neural_rows"]) if closed else None
        archive = StreamArchive(
            run / "worker/evidence",
            out / "stream_index.sqlite",
            max_read_bytes=reader.read_limit - reader.read_bytes,
            max_database_bytes=512 * 1024**2,
            max_metadata_bytes=8192,
            max_arrays_per_group=32,
            deadline=reader.deadline,
        )
        reader.archive = archive
        archive.verify_index()
        ledger = Ledger(reader, config["task"], traces, actors, actor_by_tag)
        for row in archive.ledger():
            ledger.event(row)
        census = ledger.finish(worker, config["work_limits"])
        _need(
            closed,
            "J1 worker stopped at an infrastructure/prerequisite gate; only paid-prefix checks are available",
        )
        models = reader.json("worker/score_models.json")
        checks = _audit_cases(reader, actors, ledger, cases, models, config)
        probes = audit_probe(reader, config)
        worker_checks = verify_worker_summary(
            worker, cases, reader.json("worker/teachers.json"), ledger.trace_outcomes, checks
        )
        result.update(
            passed=True,
            full_scientific_audit_passed=True,
            case_checks=checks,
            probe_checks=probes,
            worker_summary_checks=worker_checks,
            trace_outcomes=ledger.trace_outcomes,
            complete_work_census=census,
            stream_checks=dict(
                hashes=dict(archive.file_hashes),
                groups=archive.group_checks,
                arrays=archive.array_checks,
            ),
            objective_passed=checks["objective_passed"],
            claim_scope="Saved complete bounded J1 version; scientific audit does not imply a successful similarity objective",
        )
        archive.verify_unchanged()
        reader.verify_unchanged()
        reader.check()
    except Exception as error:
        result.update(
            error_type=type(error).__name__,
            error=str(error),
            unavailable_outcomes="Missing/unexecuted evidence stays missing, not failure zeros",
        )
    finally:
        if ledger is not None and not result["passed"]:
            result["available_paid_prefix"] = dict(
                requested=dict(ledger.requested),
                completed=dict(ledger.completed),
                pending_owners={str(k): v for k, v in ledger.pending.items()},
                ledger_rows_checked=ledger.rows,
                native_outcomes=ledger.trace_outcomes,
                active_native_alias=ledger.active["alias"] if ledger.active else None,
                scope="Only this verified prefix is available; later work is not inferred absent or free",
            )
        if archive is not None:
            archive.close()
    result.update(
        config_sha256=reader.files.get("config.json", {}).get("sha256"),
        source_manifest_sha256=reader.files.get("manifest.json", {}).get("sha256"),
        input_manifest_sha256=reader.files.get("input_manifest.json", {}).get("sha256"),
        worker_result_sha256=reader.files.get("worker_result.json", {}).get("sha256"),
        input_files=reader.files,
        work=dict(
            source_read_bytes=reader.total_reads,
            source_read_allocation_bytes=reader.read_limit,
            coordinator_read_reserve_bytes=COORDINATOR_READ_RESERVE,
            wall_seconds=time.monotonic() - reader.started,
            checkpoint_loads=actors.loads if actors else 0,
            neural_rows=actors.requested if actors else 0,
            completed_neural_rows=actors.completed if actors else 0,
            prediction_calls=actors.calls if actors else 0,
            unknown_partial_neural_rows=actors.unknown if actors else 0,
            new_physics_controls=0,
            new_optimizer_steps=0,
        ),
    )
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _need(
        sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) + len(payload) <= MAX_OUTPUT,
        "J1 audit output budget exhausted",
    )
    with (out / "audit.json").open("xb") as stream:
        stream.write(payload)
    return result


def verify_worker_summary(
    worker: dict, cases: list[dict], teachers: dict, outcomes: dict, checks: dict
) -> dict:
    """A worker's published success/counts must equal the independently derived result."""
    protected = any(case["split"] == "protected" for case in cases)
    expected_status = (
        "COMPLETED_J1_FIXED_VERSION" if protected else "CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE"
    )
    expected = dict(
        status=expected_status,
        objective_passed=checks["objective_passed"],
        protected_test_revealed=protected,
        cases=len(cases),
        eligible_cases=sum(bool(case["eligible"]) for case in cases),
        declared_source_groups=16,
        independent_groups=len({case["group"] for case in cases}),
        acquired_source_groups=len({key.rsplit("_", 1)[0] for key in teachers}),
        no_guidance_experiment=True,
        guidance_launched=False,
        automatic_followup=False,
    )
    for key, value in expected.items():
        _same(value, worker[key], "worker independent summary/" + key, 0)
    expected_teachers = {}
    for key, teacher in teachers.items():
        _need(
            key == teacher["tag"] and key in outcomes,
            "J1 teacher summary names an unavailable source",
        )
        expected_teachers[key] = {
            name: outcomes[key][name] for name in ("controls", "walking", "survived", "cost")
        }
    _same(expected_teachers, worker["teacher_outcomes"], "worker complete teacher outcomes", 1e-10)
    return dict(verdict_bound=True, count_fields_bound=True, complete_teacher_outcomes_bound=True)


def verify_planner_trajectory(
    reader: Reader, outcome: dict, trace: dict, config: dict, planner_seed: int
) -> dict:
    """Authenticate every carried private context, without inventing missing banks."""
    private = reader.arrays(outcome["planner_trace"])
    n, horizon = len(trace["actions"]), config["mppi"]["H"]
    for name in ("nominals_before", "nominals_after", "selected_plans"):
        value = private[name]
        _need(
            value.shape == (n, horizon, 6)
            and np.isfinite(value).all()
            and np.max(np.abs(value)) <= 1,
            "J1 planner trajectory geometry/bounds differ: " + name,
        )
    _need(
        np.array_equal(private["control_ordinals"], np.arange(n))
        and np.array_equal(private["selected_actions"], trace["actions"])
        and np.array_equal(private["selected_plans"][:, 0], trace["actions"]),
        "J1 selected planner controls are not the executed corpus",
    )
    _need(
        private["initial_before"].dtype.kind == private["initial_after"].dtype.kind == "b"
        and private["initial_before"].shape == private["initial_after"].shape == (n,)
        and private["initial_before"][0]
        and not private["initial_before"][1:].any()
        and not private["initial_after"].any()
        and np.all(private["nominals_before"][0] == 0),
        "J1 initial/noninitial optimization lifecycle changed",
    )
    _need(
        np.array_equal(private["nominals_before"][1:], private["nominals_after"][:-1])
        and np.array_equal(private["nominals_after"][:, -1], private["nominals_after"][:, -2]),
        "J1 carried horizon nominal was restarted or shifted differently",
    )
    before = [_json(str(value)) for value in private["rng_before"]]
    after = [_json(str(value)) for value in private["rng_after"]]
    _need(
        len(before) == len(after) == n
        and before[0] == np.random.PCG64(planner_seed).state
        and before[1:] == after[:-1],
        "J1 planner private RNG seed/continuation differs",
    )
    for name in ("raw_scores", "selected_task_costs", "selected_scores"):
        _need(
            private[name].shape == (n,) and np.isfinite(private[name]).all(),
            "J1 preserved decision cost summary is missing/nonfinite",
        )
    _need(
        np.all(private["selected_scores"] + 1e-10 >= private["selected_task_costs"])
        and private["bank_digests"].shape == (n,)
        and all(
            len(str(value)) == 64 and set(str(value)) <= set("0123456789abcdef")
            for value in private["bank_digests"]
        ),
        "J1 planner decision digest/selector cost summary differs",
    )
    return dict(
        controls=n,
        executed_action_joins=n,
        private_context_continuations=n - 1,
        full_raw_cost_banks_available=False,
    )


def _audit_cases(
    reader: Reader, actors: Actors, ledger: Ledger, cases: list[dict], models: dict, config: dict
) -> dict:
    """Case/prospective joins and prediction recomputation; never refit actors."""
    from ..similarity_metrics import (
        evaluate_predictions,
        fit_occupancy_normalizer,
        fit_prediction_models,
        fit_state_explainer,
        predict_case_scores,
    )

    expected = {
        (g["id"], learner_row["id"], s): (g, learner_row)
        for g in config["groups"]
        for learner_row in g["learners"]
        for s in ("mppi", "reference")
    }
    seen = set()
    records = []
    fit_checks = []
    teacher_by_key = {}
    incoming_by_learner = {}
    seal_cache = {}
    paired = {}
    private_checked = set()

    def bind_trace(outcome):
        recorded = ledger.traces.get(outcome["tag"])
        _need(recorded is not None, "J1 case names an unclosed/unlisted native trace")
        for key in ("trace", "contact_trace"):
            _same(outcome[key], recorded[key], "case/global trace reference/" + key, 0)
        for name in ("controls", "walking", "survived", "cost"):
            _same(
                outcome[name],
                ledger.trace_outcomes[outcome["tag"]][name],
                "case native outcome/" + name,
                1e-10,
            )

    for case in cases:
        key = (case["group"], case["learner"], case["source"])
        _need(key in expected and key not in seen, "J1 case roster duplicated or undeclared")
        seen.add(key)
        group, learner = expected[key]
        _need(
            case["split"] == group["split"]
            and case["evaluation_seeds"] == learner["evaluation_seeds"],
            "J1 case split/evaluation resets differ",
        )
        bind_trace(case["teacher"])
        bind_trace(case["incoming"])
        _need(
            case["teacher"]["seed"] == group["acquisition_seed"],
            "J1 teacher reset seed differs from its source-group binding",
        )
        teacher = reader.arrays(case["teacher"]["trace"])
        incoming = reader.arrays(case["incoming"]["trace"])
        teacher_by_key[(case["group"], case["source"])] = teacher
        incoming_by_learner[case["learner"]] = incoming
        _need(
            case["eligible"] == (len(teacher["actions"]) == 1000),
            "J1 source eligibility depends on outcome or padding",
        )
        for field in ("states", "warmstarts", "integrations", "observations"):
            _need(
                teacher[field][0].tobytes() == incoming[field][0].tobytes(),
                "J1 incoming execution did not start at exact teacher context",
            )
        replay = case["teacher"].get("replay")
        _need(replay is not None, "J1 teacher replay certificate is missing")
        bind_trace(replay)
        replay_arrays = reader.arrays(replay["trace"])
        for field in TRACE_KEYS:
            if field in {"metadata", "initial_digest", "row_digests"}:
                continue
            if field in {"states", "warmstarts", "integrations", "observations"}:
                _need(
                    teacher[field][0].tobytes() == replay_arrays[field][0].tobytes(),
                    "J1 teacher restoration is not byte-exact",
                )
            tolerance = 0 if field in {"actions", "done", "control_ordinals"} else 1e-10
            _need(
                teacher[field].shape == replay_arrays[field].shape
                and np.allclose(teacher[field], replay_arrays[field], rtol=0, atol=tolerance),
                "J1 teacher replay differs: " + field,
            )
        original_contacts = reader.arrays(case["teacher"]["contact_trace"])
        replay_contacts = reader.arrays(replay["contact_trace"])
        for field in original_contacts:
            _need(
                np.allclose(original_contacts[field], replay_contacts[field], rtol=0, atol=1e-10),
                "J1 teacher replay contact/force differs: " + field,
            )
        if case["source"] == "mppi" and case["teacher"]["tag"] not in private_checked:
            verify_planner_trajectory(
                reader, case["teacher"], teacher, config, group["planner_seed"]
            )
            private_checked.add(case["teacher"]["tag"])
        if not case["eligible"]:
            _need(
                "child_checkpoint" not in case and not case.get("evaluations"),
                "J1 short source was fitted/replaced",
            )
            continue
        bindings = case["incoming_bindings"]
        pairkey = (case["group"], case["learner"])
        if pairkey in paired:
            _same(bindings, paired[pairkey], "paired incoming tensor/Adam/RNG/tape", 0)
        paired[pairkey] = bindings
        _need(
            bindings["seed"] == learner["policy_seed"]
            and bindings["fitseed"] == learner["fit_seed"],
            "J1 fresh learner seeds changed",
        )
        prediction = reader.array(case["pre_prediction_ref"], "predictions")
        _need(
            case.get("pre_prediction_batch_shape") == [1000, 17],
            "J1 preprediction batching is not bound",
        )
        actors.verify(
            bindings["checkpoint"], teacher["observations"][:-1], prediction, batch_one=False
        )
        scores = score_oracle(teacher, incoming, prediction, models)
        _same(scores, case["scores"], "pre-update scores", 1e-9)
        seal_file = case["score_seal_file"]
        if seal_file not in seal_cache:
            seal_cache[seal_file] = reader.json(seal_file)
        seal = seal_cache[seal_file]
        models_file = (
            "worker/score_models_prelearning.json"
            if case["split"] == "development"
            else "worker/score_models.json"
        )
        if models_file not in reader.files:
            reader.json(models_file)
        _need(
            seal["models_sha256"] == reader.files[models_file]["sha256"]
            and seal["no_learning_in_split_yet"] is True
            and seal["split"] == case["split"],
            "J1 score seal does not bind its pre-update scoring model",
        )
        event = ledger.events[case["score_seal_event"]]
        begin = ledger.events[case["fit_begin_event"]]
        _need(
            event["event"] == "j1_scores_sealed"
            and event["split"] == case["split"]
            and event["sha256"]
            == reader.files[str(reader.path(seal_file).relative_to(reader.root))]["sha256"]
            and begin["event"] == "j1_fit_begin"
            and begin["case"] == case["case_id"]
            and begin.get("seal", begin.get("seal_event")) == case["score_seal_event"]
            and int(event["id"].split("-")[-1]) < int(begin["id"].split("-")[-1]),
            "J1 score was not sealed before the learning update",
        )
        sealed = next((r for r in seal["cases"] if r["case_id"] == case["case_id"]), None)
        _need(
            sealed is not None and "post_motion" not in sealed and "child_checkpoint" not in sealed,
            "J1 pre-update seal contains/misses outcome information",
        )
        _same(sealed["scores"], case["scores"], "sealed score values", 0)
        fit_checks.append(verify_learning_case(reader, actors, case, teacher))
        bind_trace(case["reproduction"])
        reproduction = reader.arrays(case["reproduction"]["trace"])
        for field in ("states", "warmstarts", "integrations", "observations"):
            _need(
                teacher[field][0].tobytes() == reproduction[field][0].tobytes(),
                "J1 child reproduction context differs",
            )
        actual = motion_oracle(teacher, reproduction)
        _same(actual["error"], case["post_motion"], "post-update independent motion", 1e-9)
        _need(actual["reproduced"] is case["reproduced"], "J1 binary reproduction differs")
        evaluations = case["evaluations"]
        for outcome in evaluations:
            bind_trace(outcome)
        _need(
            len(evaluations) == 16
            and [r["seed"] for r in evaluations] == learner["evaluation_seeds"],
            "J1 fresh evaluation roster differs",
        )
        walking = sum(ledger.trace_outcomes[r["tag"]]["walking"] for r in evaluations)
        _need(
            walking == case["walking_successes"] and case["walking_trials"] == 16,
            "J1 post-update standalone walking differs",
        )
        records.append(
            dict(
                case,
                post_motion=actual["error"],
                reproduced=actual["reproduced"],
                walking_successes=walking,
            )
        )
    devgroups = {g["id"] for g in config["groups"] if g["split"] == "development"}
    if any(c["split"] == "protected" for c in cases):
        _need(seen == set(expected), "J1 final case roster is incomplete")
    else:
        _need(
            seen == {key for key in expected if key[0] in devgroups},
            "J1 development-only roster is incomplete",
        )
    for group in {key[0] for key in seen}:
        mppi, reference = teacher_by_key[group, "mppi"], teacher_by_key[group, "reference"]
        for field in ("states", "warmstarts", "integrations", "observations"):
            _need(
                mppi[field][0].tobytes() == reference[field][0].tobytes(),
                "J1 MPPI/reference source initial contexts differ",
            )
    teacher_rows = [
        teacher_by_key[(g["id"], source)]
        for g in config["groups"]
        if g["split"] == "development"
        for source in ("mppi", "reference")
        if len(teacher_by_key[(g["id"], source)]["actions"]) == 1000
    ]
    normalization = models["explainer"]["normalization"]
    explanatory = fit_state_explainer(
        np.concatenate([t["observations"][:-1] for t in teacher_rows]),
        np.concatenate([t["actions"] for t in teacher_rows]),
        normalization,
        group_ids=sorted(devgroups),
    )
    _same(explanatory, models["explainer"], "development-only explanatory map", 1e-8)
    clouds = [np.column_stack((t["observations"][:-1], t["actions"])) for t in teacher_rows]
    clouds.extend(
        np.column_stack(
            (
                incoming_by_learner[learner_row["id"]]["observations"][:-1],
                incoming_by_learner[learner_row["id"]]["actions"],
            )
        )
        for g in config["groups"]
        if g["split"] == "development"
        for learner_row in g["learners"]
    )
    occupancy = fit_occupancy_normalizer(np.concatenate(clouds), group_ids=sorted(devgroups))
    _same(occupancy, models["occupancy_normalizer"], "development-only occupancy scales", 1e-9)
    prelearning = reader.json("worker/score_models_prelearning.json")
    _same(
        prelearning,
        {key: models[key] for key in ("explainer", "occupancy_normalizer")},
        "immutable prelearning scoring models",
        0,
    )
    predicted = fit_prediction_models(records, source="mppi")
    _same(predicted, models["prediction_models"], "development-only prediction coefficients", 1e-6)
    for candidate in predicted["models"]:
        _same(
            predicted["models"][candidate]["motion_coefficients"],
            models["prediction_models"]["models"][candidate]["motion_coefficients"],
            "development linear coefficients/" + candidate,
            1e-8,
        )
    final = [r for r in records if r["split"] == "protected"]
    if final:
        comparison = evaluate_predictions(records, predicted, source="mppi")
        saved = reader.json("worker/predictive_comparison.json")
        _same(comparison, saved, "protected prediction/paired bootstrap", 1e-9)
        objective = comparison["objective_passed"]
        for record in final:
            _same(
                predict_case_scores(record["scores"], predicted),
                record["predictions"],
                "prospective prediction",
                1e-9,
            )
        prediction_events = [
            e for e in ledger.events.values() if e["event"] == "j1_prediction_models_sealed"
        ]
        _need(
            len(prediction_events) == 1,
            "J1 prediction model must be sealed once before protected fitting",
        )
        event = prediction_events[0]
        _need(
            event["sha256"] == reader.files["worker/score_models.json"]["sha256"]
            and all(
                int(event["id"].split("-")[-1]) < int(c["score_seal_event"].split("-")[-1])
                for c in final
            ),
            "J1 final scores/models were not prospectively frozen",
        )
    else:
        _need(
            not predicted["development_roster_complete"] or predicted["motion_sd"] < 0.1,
            "J1 protected test skipped without its declared variation gate",
        )
        comparison = None
        objective = False
    _need(
        ledger.completed["fits"] == len(records)
        and ledger.completed["optimizer_steps"] == 2000 * len(records)
        and ledger.completed["backward_rows"] == 512000 * len(records),
        "J1 case/learning work census differs",
    )
    return dict(
        cases=len(cases),
        eligible_cases=len(records),
        learning_checks=fit_checks,
        preupdate_scores_recomputed=True,
        development_models_recomputed=True,
        protected_comparison=comparison,
        objective_passed=objective,
        independent_score_oracles=["A", "B", "D", "E"],
        shared_pure_formulas=["C", "development prediction fitting", "paired bootstrap"],
        limitations="No actor refitting or physics replay; independent execution measurements are computed from immutable raw simulator evidence.",
    )


def planning_cost_oracle(states: np.ndarray, actions: np.ndarray, task: dict) -> np.ndarray:
    """Hypothetical batch costs include the first fall, then an absorbing tail."""
    _need(
        states.ndim == 3 and states.shape[2] == 19 and actions.shape == (*states.shape[:2], 6),
        "J1 within-decision sample shapes differ",
    )
    healthy = (
        (states[..., 2] > 0.8) & (states[..., 2] < 2) & (states[..., 3] > -1) & (states[..., 3] < 1)
    )
    active = np.ones_like(healthy)
    if states.shape[1] > 1:
        active[:, 1:] = np.cumprod(healthy[:, :-1], axis=1).astype(bool)
    running = (
        task["vel_cost_weight"] * np.abs(task["target_velocity"] - states[..., 10])
        + task["vertical_velocity_cost_weight"] * states[..., 11] ** 2
        + task["angle_cost_weight"] * states[..., 3] ** 2
        + task["unhealthy_cost_weight"] * (~healthy)
        + task["ctrl_cost_weight"] * np.sum(actions**2, axis=2)
    )
    total = np.where(active, running, 0).sum(axis=1)
    total += task["unhealthy_cost_weight"] * (~active).sum(axis=1)
    total += task["terminal_unhealthy_weight"] * np.any(~healthy, axis=1)
    valid = np.isfinite(states).all(axis=2) & np.isfinite(actions).all(axis=2)
    return np.where(np.all(valid | ~active, axis=1), total, np.inf)


def probe_roster_oracle() -> list[dict]:
    """Independent enumeration of the prospective 208-decision factorial design."""
    rows = []
    scales = np.array([0.001] * 8 + [0.01] * 9)
    directions = np.random.Generator(np.random.PCG64(2026129901)).choice(
        [-1.0, 1.0], size=(8, 17)
    ) / np.sqrt(17)

    def add(context, role, innovation=0, **fields):
        row = dict(
            id=f"probe_{len(rows):03d}",
            context=context,
            role=role,
            nominal="carried",
            innovation=innovation,
            observation_delta=[0.0] * 17,
            time_delta=0.0,
            rootx_delta=0.0,
            warmstart_delta=0.0,
        )
        row.update(fields)
        rows.append(row)

    for context in range(4):
        add(context, "original_repeat", observe=False)
    for context in range(4):
        for nominal in ("carried", "zero"):
            for innovation in range(3):
                row = dict(
                    id=f"probe_{len(rows):03d}",
                    context=context,
                    role="factorial",
                    nominal=nominal,
                    innovation=innovation,
                    observation_delta=[0.0] * 17,
                    time_delta=0.0,
                    rootx_delta=0.0,
                    warmstart_delta=0.0,
                    observe=True,
                )
                rows.append(row)
    for context in range(2):
        for innovation in range(2):
            for coordinate in range(17):
                for sign in (-1, 1):
                    delta = np.zeros(17)
                    delta[coordinate] = sign * scales[coordinate]
                    add(
                        context,
                        "gain_axis",
                        innovation,
                        coordinate=coordinate,
                        sign=sign,
                        observation_delta=delta.tolist(),
                        observe=True,
                    )
            for index, direction in enumerate(directions):
                add(
                    context,
                    "gain_held",
                    innovation,
                    held_index=index,
                    observation_delta=(1.5 * direction * scales).tolist(),
                    observe=True,
                )
    for context in range(2):
        for coordinate, amount in (("time", 0.008), ("rootx", 0.01)):
            for sign in (-1, 1):
                add(context, "hidden_state", **{coordinate + "_delta": sign * amount}, observe=True)
    for context in range(2):
        for sign in (-1, 1):
            add(context, "warmstart", warmstart_delta=sign * 0.01, observe=True)
    return rows


def _temporal_cholesky(config: dict) -> np.ndarray:
    from scipy.linalg import solve_discrete_lyapunov, toeplitz
    from scipy.signal import butter, tf2ss

    numerator, denominator = butter(
        2,
        config["noise_lowpass_cutoff_hz"],
        btype="lowpass",
        fs=config["noise_lowpass_sample_rate_hz"],
        output="ba",
    )
    a, b, c, d = tf2ss(numerator, denominator)
    stationary = solve_discrete_lyapunov(a, b @ b.T)
    covariance = np.empty(config["H"])
    covariance[0] = (c @ stationary @ c.T + d @ d.T).item()
    previous = np.eye(a.shape[0])
    for lag in range(1, config["H"]):
        current = a @ previous
        covariance[lag] = (c @ current @ stationary @ c.T + c @ previous @ b @ d.T).item()
        previous = current
    correlation = toeplitz(covariance / covariance[0])
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return np.linalg.cholesky(correlation)


def _reference_value(reader: Reader, reference: dict, name: str) -> Any:
    value = reader.array(reference, name)
    if reference["arrays"][name]["encoding"] == "json":
        _need(value.shape == (), "J1 JSON evidence is not scalar")
        return _json(value.item())
    return value


def audit_probe_selection(
    reader: Reader, row: dict, context: dict, last_pass: dict | None, config: dict
) -> dict:
    """Independently score the archived selector bank and its verified mean."""
    selection = row["selection"]
    bank = reader.arrays(selection["evidence"])
    actions, posts, costs = bank["bank_actions"], bank["bank_post_states"], bank["bank_raw_costs"]
    _need(
        actions.shape == (config["mppi"]["K"] + 1, config["mppi"]["H"], 6)
        and np.isfinite(actions).all()
        and np.max(np.abs(actions)) <= 1,
        "J1 selector bank sample geometry/bounds differ",
    )
    _need(
        np.array_equal(bank["initial_state"], context["state"])
        and np.array_equal(bank["initial_warmstart"], context["warmstart"])
        and np.array_equal(bank["reference_u"], np.asarray(row["planner_before"]["U"])),
        "J1 bank is bound to a different physical/private context",
    )
    digest = hashlib.sha256()
    for name in (
        "initial_state",
        "initial_warmstart",
        "bank_actions",
        "bank_post_states",
        "bank_raw_costs",
        "reference_u",
    ):
        array = np.ascontiguousarray(bank[name])
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    _need(digest.hexdigest() == selection["bank_digest"], "J1 immutable search-bank digest differs")
    _need(
        np.allclose(planning_cost_oracle(posts, actions, config["task"]), costs, rtol=0, atol=1e-9),
        "J1 selector bank task costs differ",
    )
    pre = np.concatenate(
        (np.broadcast_to(context["state"], (len(actions), 1, 19)), posts[:, :-1]), axis=1
    )
    observations = np.concatenate((pre[..., 2:10], np.clip(pre[..., 10:19], -10, 10)), axis=2)
    healthy = (
        (posts[..., 2] > 0.8) & (posts[..., 2] < 2) & (posts[..., 3] > -1) & (posts[..., 3] < 1)
    )
    active = np.ones(healthy.shape, dtype=bool)
    active[:, 1:] = np.cumprod(healthy[:, :-1], axis=1).astype(bool)
    _need(
        np.array_equal(observations, bank["bank_observations"])
        and np.array_equal(active, bank["bank_active_mask"]),
        "J1 hypothetical pre-action observations/active fall prefix differ",
    )
    if last_pass is not None:
        _need(
            np.array_equal(actions[:-1], last_pass["actions"])
            and np.array_equal(posts[:-1], last_pass["states"])
            and np.array_equal(costs[:-1], last_pass["native_costs"])
            and np.array_equal(actions[-1], last_pass["nominal_after"]),
            "J1 final iteration/selector bank join differs",
        )
    raw_index = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))
    _need(
        np.array_equal(bank["raw_plan"], actions[raw_index])
        and math.isclose(selection["raw_score"], costs[raw_index], rel_tol=0, abs_tol=1e-9),
        "J1 raw best native candidate differs",
    )
    proximity = (
        0.5
        * config["selector"]["eta_u"]
        * np.sum((actions - bank["reference_u"][None]) ** 2, axis=(1, 2))
    )
    scores = costs + proximity
    _need(
        np.allclose(proximity, bank["proximity_costs"], rtol=0, atol=1e-12)
        and np.allclose(scores, bank["baseline_scores"], rtol=0, atol=1e-9)
        and np.array_equal(bank["guided_scores"], bank["baseline_scores"])
        and not np.any(bank["support"])
        and not np.any(bank["agreement_squared"]),
        "J1 independent selector gained policy influence or changed costs",
    )
    index = int(np.argmin(np.where(np.isfinite(scores), scores, np.inf)))
    selected, selected_score, selected_cost, origin = (
        actions[index],
        scores[index],
        costs[index],
        f"bank:{index}",
    )
    finite = np.isfinite(scores)
    weights = np.zeros(scores.shape)
    weights[finite] = np.exp(-(scores[finite] - np.min(scores[finite])) / config["mppi"]["lam"])
    weights /= weights.sum()
    if "baseline_mean_plan" in bank:
        mean = np.einsum("k,kha->ha", weights, actions)
        _need(np.array_equal(mean, bank["baseline_mean_plan"]), "J1 baseline selector mean differs")
        checked_cost = planning_cost_oracle(
            bank["baseline_mean_post_states"], mean[None], config["task"]
        )[0]
        _need(
            np.allclose([checked_cost], bank["baseline_mean_raw_costs"], rtol=0, atol=1e-9),
            "J1 verified selector mean task cost differs",
        )
        # Preserve the selector's exact tie arithmetic after independently
        # checking the native cost; summation roundoff cannot change its origin.
        mean_cost = float(bank["baseline_mean_raw_costs"][0])
        mean_score = mean_cost + 0.5 * config["selector"]["eta_u"] * np.sum(
            (mean - bank["reference_u"]) ** 2
        )
        if mean_score < selected_score:
            selected, selected_score, selected_cost, origin = (
                mean,
                mean_score,
                mean_cost,
                "verified_baseline_mean",
            )
    else:
        _need(
            selection["baseline_verification_error"] is not None,
            "J1 selector mean verification evidence is missing",
        )
    _need(
        np.array_equal(selected, bank["selected_plan"])
        and np.array_equal(selected, np.asarray(row["selected_plan"]))
        and np.array_equal(selected[0], np.asarray(row["action"]))
        and selection["selected_origin"] == origin
        and math.isclose(selected_score, selection["selected_score"], rel_tol=0, abs_tol=1e-9)
        and math.isclose(selected_cost, selection["selected_task_cost"], rel_tol=0, abs_tol=1e-9)
        and selection["enabled"] is False
        and selection["influence_used"] is False
        and selection["actor_hash"] is None
        and selection["alpha"] == 0,
        "J1 independent selector output/tie rule differs",
    )
    return dict(
        bank_candidates=len(actions),
        native_cost_rows=len(actions) * actions.shape[1],
        digest=digest.hexdigest(),
    )


def audit_probe(reader: Reader, config: dict) -> dict:
    rows = reader.json("worker/probe_results.json")
    saved = reader.json("worker/probe_summary.json")
    expected = probe_roster_oracle()
    _need(
        len(rows) == len(expected) == config["probe"]["decisions"] == 208,
        "J1 probe decision roster differs",
    )
    teachers = reader.json("worker/teachers.json")
    covariance = _temporal_cholesky(config["mppi"])
    enriched = []
    passes = 0
    noise_values = 0
    bank_checks = []
    for row, request in zip(rows, expected, strict=True):
        _same(request, row["request"], "probe intervention roster", 0)
        reference = row["context_ref"]
        actual = {name: _reference_value(reader, reference, name) for name in reference["arrays"]}
        for field in ("action", "selected_plan", "planner_before", "planner_after", "observation"):
            _same(
                np.asarray(actual[field]).tolist()
                if isinstance(actual[field], np.ndarray)
                else actual[field],
                row[field],
                "probe persisted/summary/" + field,
                0,
            )
        anchor = config["probe"]["contexts"][request["context"]]
        teacher = teachers[anchor["group"] + "_mppi"]
        source = reader.arrays(teacher["trace"])
        private = reader.arrays(teacher["planner_trace"])
        t = anchor["control"]
        before = dict(
            U=private["nominals_before"][t].tolist(),
            rng_state=_json(str(private["rng_before"][t])),
            initial_plan=bool(private["initial_before"][t]),
        )
        if request["innovation"]:
            before["rng_state"] = np.random.PCG64(
                (2026129701, 2026129702)[request["innovation"] - 1]
            ).state
        if request["nominal"] == "zero":
            before["U"] = np.zeros((config["mppi"]["H"], 6)).tolist()
        _same(before, row["planner_before"], "probe private/RNG/nominal input", 0)
        expected_state = source["states"][t].copy()
        expected_warm = source["warmstarts"][t].copy()
        delta = np.asarray(request["observation_delta"])
        active = np.flatnonzero(delta)
        expected_state[active + 2] += delta[active]
        for index, field in ((0, "time_delta"), (1, "rootx_delta")):
            if request[field] != 0:
                expected_state[index] += request[field]
        if request["warmstart_delta"] != 0:
            expected_warm[0] += request["warmstart_delta"]
        _need(
            expected_state.tobytes() == actual["state"].tobytes()
            and expected_warm.tobytes() == actual["warmstart"].tobytes()
            and actual["integration"].shape == (91,)
            and np.array_equal(actual["integration"][:19], expected_state)
            and np.array_equal(actual["integration"][19:28], expected_warm)
            and np.array_equal(actual["integration"][28:], source["integrations"][t, 28:]),
            "J1 requested physical/hidden perturbation differs",
        )
        observation = np.hstack((expected_state[2:10], np.clip(expected_state[10:19], -10, 10)))
        _need(
            np.array_equal(observation, actual["observation"]),
            "J1 probe observation interface changed",
        )
        _need(
            len(row["pass_refs"]) == int(request["observe"]),
            "J1 observer changed noninitial pass count",
        )
        last_pass = None
        for pass_reference in row["pass_refs"]:
            data = {
                name: _reference_value(reader, pass_reference, name)
                for name in pass_reference["arrays"]
            }
            sampled = data["actions"]
            nominal = data["nominal_before"]
            costs = data["native_costs"]
            _need(
                sampled.shape == (config["mppi"]["K"], config["mppi"]["H"], 6),
                "J1 pass sample count/horizon changed",
            )
            raw = planning_cost_oracle(data["states"], sampled, config["task"])
            _need(
                np.allclose(raw, costs, rtol=0, atol=1e-9), "J1 within-decision native costs differ"
            )
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = data["rng_before"]
            standard = rng.standard_normal(sampled.shape)
            noise_values += standard.size
            temporal = np.einsum("ht,kta->kha", covariance, standard, optimize=True)
            spatial = np.linalg.cholesky(config["mppi"]["noise_sigma"] ** 2 * np.eye(6))
            noise = np.einsum("khi,ji->khj", temporal, spatial)
            _need(
                np.array_equal(noise, data["correlated_innovation"])
                and rng.bit_generator.state == data["rng_after"],
                "J1 private innovations/RNG continuation differ",
            )
            _need(
                np.array_equal(np.clip(nominal[None] + noise, -1, 1), sampled),
                "J1 bounded samples differ from innovations/nominal",
            )
            perturbation = sampled - nominal[None]
            _need(
                np.array_equal(perturbation, data["effective_perturbations"]),
                "J1 effective clipping perturbation differs",
            )
            finite = np.isfinite(costs)
            _need(finite.any(), "No finite independently scored probe sample")
            weights = np.exp(
                -np.where(finite, costs - np.min(costs[finite]), np.inf) / config["mppi"]["lam"]
            )
            weights /= weights.sum()
            _need(
                np.allclose(weights, data["weights"], rtol=0, atol=1e-12),
                "J1 within-decision weights differ",
            )
            update = np.clip(nominal + np.einsum("k,kha->ha", weights, perturbation), -1, 1)
            _need(
                np.allclose(update, data["nominal_after"], rtol=0, atol=1e-10),
                "J1 weighted nominal update differs",
            )
            _need(
                np.array_equal(data["returned_action"], update[0]),
                "J1 iteration returned action differs",
            )
            passes += 1
            last_pass = data
        bank_checks.append(audit_probe_selection(reader, row, actual, last_pass, config))
        if request["role"] == "original_repeat" or (
            request["role"] == "factorial"
            and request["nominal"] == "carried"
            and request["innovation"] == 0
        ):
            _need(
                row["original_replay_verified"] is True
                and np.array_equal(actual["action"], source["actions"][t])
                and np.array_equal(actual["selected_plan"], private["selected_plans"][t]),
                "J1 original decision did not reproduce",
            )
            after = dict(
                U=private["nominals_after"][t].tolist(),
                rng_state=_json(str(private["rng_after"][t])),
                initial_plan=bool(private["initial_after"][t]),
            )
            _same(after, row["planner_after"], "original private continuation", 0)
            _need(
                row["selection"]["bank_digest"] == str(private["bank_digests"][t])
                and math.isclose(
                    row["selection"]["selected_task_cost"],
                    float(private["selected_task_costs"][t]),
                    rel_tol=0,
                    abs_tol=1e-10,
                ),
                "J1 original replay bank/cost differs from acquisition",
            )
        enriched.append(
            dict(
                row,
                **{
                    name: actual[name]
                    for name in (
                        "action",
                        "selected_plan",
                        "planner_before",
                        "planner_after",
                        "observation",
                    )
                },
            )
        )
    with np.load(
        io.BytesIO(reader.read("prepared/normalization.npz")), allow_pickle=False
    ) as nfile:
        normal_std = nfile["std"]
    descriptive = probe_descriptive_oracle(enriched, normal_std)
    _same(descriptive, saved, "descriptive gains, contact regimes and factorial contrasts", 1e-8)
    fits = descriptive["local_fits"]
    _need(
        saved["decisions"] == 208
        and saved["optimal_gain_claim"] is False
        and saved["unexplained_variation_is_irrelevant"] is False,
        "J1 probe summary claim scope differs",
    )
    return dict(
        decisions=208,
        observed_passes=passes,
        original_decision_replays=4,
        observer_equivalence_checks=4,
        private_standard_normal_values_recomputed=noise_values,
        local_gain_checks=fits,
        selector_banks_checked=len(bank_checks),
        selector_cost_rows_checked=sum(value["native_cost_rows"] for value in bank_checks),
        native_dynamics_repeated=False,
        optimal_gain_claim=False,
        scope="Every preserved probe sample/cost/weight/private innovation and gain is checked; full raw banks were not retained for all acquisition decisions.",
    )


def probe_descriptive_oracle(enriched: list[dict], normal_std: np.ndarray) -> dict:
    """Recompute gain, contact-regime and nuisance contrasts from saved rows."""
    fits = []
    scales = np.array([0.001] * 8 + [0.01] * 9)

    def presence(row):
        value = row.get("contacts")
        if value is None:
            return None
        value = value.get("presence", value.get("contacts")) if isinstance(value, dict) else value
        value = np.asarray(value)
        _need(
            value.shape == (2,) and value.dtype.kind == "b",
            "J1 restored-boundary contact diagnostic is invalid",
        )
        return value

    for context in range(2):
        for innovation in range(2):
            subset = [
                r
                for r in enriched
                if r["request"]["context"] == context and r["request"]["innovation"] == innovation
            ]
            base = next(
                r
                for r in subset
                if r["request"]["role"] == "factorial" and r["request"]["nominal"] == "carried"
            )
            axes = [r for r in subset if r["request"]["role"] == "gain_axis"]
            held = [r for r in subset if r["request"]["role"] == "gain_held"]
            x = (np.array([r["observation"] for r in axes]) - base["observation"]) / scales
            y = np.array([r["action"] for r in axes]) - base["action"]
            singular = np.linalg.svd(x, compute_uv=False)
            tol = np.finfo(float).eps * max(x.shape) * singular[0]
            coef, _, rank, _ = np.linalg.lstsq(
                x, y, rcond=tol / singular[0] if singular[0] else None
            )
            residual = (
                np.array([r["observation"] for r in held]) - base["observation"]
            ) / scales @ coef - (np.array([r["action"] for r in held]) - base["action"])
            fit = dict(
                schema="walker-j1-local-empirical-gain-v1",
                context=context,
                innovation=innovation,
                rank=int(rank),
                full_column_rank=bool(rank == 17),
                unidentified_dimensions=17 - int(rank),
                singular_values=singular.tolist(),
                rank_tolerance=float(tol),
                condition_number=float(singular[0] / singular[-1]) if rank == 17 else None,
                gain_standardized=coef.T.tolist(),
                gain_per_physical_observation=(coef.T / scales).tolist(),
                gain_per_normalized_observation=(coef.T / scales * normal_std).tolist(),
                training_action_rmse=float(np.sqrt(np.mean((x @ coef - y) ** 2))),
                held_action_rmse=float(np.sqrt(np.mean(residual**2))),
                held_action_max_abs_error=float(np.max(np.abs(residual))),
                held_residuals=residual.tolist(),
                minimum_norm_solution=True,
                optimal_gain_claim=False,
                normalization_is_algebraic_transform_only=True,
                baseline_clipped_velocity_coordinates=np.asarray(
                    base.get("observation_clipped_velocities", np.zeros(9, dtype=bool))
                ).tolist(),
            )
            base_contacts = presence(base)
            for name, rows in (("axis", axes), ("held", held)):
                values = [presence(row) for row in rows]
                fit[name + "_contact_regime_changes"] = (
                    None
                    if base_contacts is None or any(value is None for value in values)
                    else sum(not np.array_equal(value, base_contacts) for value in values)
                )
                fit[name + "_contact_measurements_available"] = sum(
                    value is not None for value in values
                )
                fit[name + "_action_boundary_rows"] = sum(
                    bool(np.any(np.abs(row["action"]) >= 1 - 1e-12)) for row in rows
                )
            fits.append(fit)
    contrasts = []
    for context in range(4):
        cells = {
            (row["request"]["nominal"], row["request"]["innovation"]): row
            for row in enriched
            if row["request"]["context"] == context and row["request"]["role"] == "factorial"
        }
        contrasts.append(
            dict(
                context=context,
                nominal_action_squared_changes=[
                    float(
                        np.mean(
                            (
                                np.asarray(cells["carried", i]["action"])
                                - np.asarray(cells["zero", i]["action"])
                            )
                            ** 2
                        )
                    )
                    for i in range(3)
                ],
                carried_rng_action_variance=np.var(
                    [cells["carried", i]["action"] for i in range(3)], axis=0
                ).tolist(),
                zero_rng_action_variance=np.var(
                    [cells["zero", i]["action"] for i in range(3)], axis=0
                ).tolist(),
            )
        )
    hidden = []
    for row in enriched:
        request = row["request"]
        if request["role"] not in {"hidden_state", "warmstart"}:
            continue
        base = next(
            r
            for r in enriched
            if r["request"]["context"] == request["context"]
            and r["request"]["role"] == "factorial"
            and r["request"]["nominal"] == "carried"
            and r["request"]["innovation"] == 0
        )
        _need(
            np.array_equal(row["observation"], base["observation"]),
            "J1 hidden-state intervention changed the policy input",
        )
        hidden.append(
            dict(
                id=request["id"],
                context=request["context"],
                role=request["role"],
                observed_input_unchanged=True,
                action_squared_change=float(
                    np.mean((np.asarray(row["action"]) - np.asarray(base["action"])) ** 2)
                ),
            )
        )
    return dict(
        schema="walker-j1-probe-summary-v1",
        decisions=208,
        original_replay_checks=sum(row.get("original_replay_verified") is True for row in enriched),
        local_fits=fits,
        factorial_contrasts=contrasts,
        hidden_input_contrasts=hidden,
        optimal_gain_claim=False,
        unexplained_variation_is_irrelevant=False,
    )
