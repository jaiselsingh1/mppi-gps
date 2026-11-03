"""Independent J2 saved-evidence audit; never fits actors or repeats physics.

The primary comparison groups all label conditions by their fresh learner seed.
Its exact sign-flip test has six units, not 78 independent fitted students.
Checkpoint verification uses lazy functional forwards, with a separate census.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import resource
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .teacher_proposal import AuditError, optimizer_fingerprint

FRESH_SEEDS = tuple(range(3_600_003_000, 3_600_003_006))
MAX_READ_BYTES = 64 * 1024**3
MAX_WALL_SECONDS = 7200
MAX_NEURAL_ROWS = 5_000_000
MAX_RSS_BYTES = 4 * 1024**3


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _need(array.ndim == 1 and np.isfinite(array).all(), "Nonfinite J2 rank observations")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def rank_correlation(x: list[float], y: list[float]) -> float | None:
    _need(len(x) == len(y), "J2 correlation observations are not paired")
    if len(x) < 2:
        return None
    a, b = _ranks(x), _ranks(y)
    a -= a.mean()
    b -= b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if norm == 0 else float(a @ b / norm)


def any_walking_auc(error: list[float], walks: list[int]) -> float | None:
    _need(len(error) == len(walks), "J2 AUC observations are not paired")
    _need(all(math.isfinite(value) for value in error), "Nonfinite J2 AUC error")
    positives = [value for value, count in zip(error, walks, strict=True) if count > 0]
    negatives = [value for value, count in zip(error, walks, strict=True) if count == 0]
    if not positives or not negatives:
        return None
    credit = math.fsum(
        1.0 if positive < negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return credit / (len(positives) * len(negatives))


def sign_flip_oracle(differences: list[float]) -> dict[str, Any]:
    """One-sided exhaustive paired test, with six whole-seed units and ties included."""
    _need(len(differences) == 6, "J2 sign flips require six whole-seed differences")
    _need(all(math.isfinite(value) for value in differences), "Nonfinite J2 differences")
    observed = math.fsum(differences) / 6
    distribution = [
        math.fsum(sign * value for sign, value in zip(signs, differences, strict=True)) / 6
        for signs in itertools.product((-1, 1), repeat=6)
    ]
    tail = sum(value >= observed for value in distribution)
    return dict(
        alternative="mean paired improvement greater than zero",
        null_assumption="independent seed-wise differences are sign-symmetric; not a randomized causal test",
        observed_mean_improvement=observed,
        patterns=64,
        upper_tail_patterns=tail,
        p_value=tail / 64,
        null_means=distribution,
    )


def primary_oracle(
    records: list[dict[str, Any]], seeds: list[int], conditions: list[str]
) -> dict[str, Any]:
    """Recompute the frozen seed-level comparison from observed case measurements."""
    _need(len(seeds) == len(set(seeds)) == 6, "J2 requires six distinct learner streams")
    _need(len(conditions) == len(set(conditions)) == 13, "J2 requires thirteen fixed conditions")
    indexed = {(record["seed"], record["condition"]): record for record in records}
    _need(
        len(indexed) == len(records) and set(indexed) == set(itertools.product(seeds, conditions)),
        "J2 seed/condition case roster differs",
    )
    per_seed = []
    for seed in seeds:
        cases = [indexed[seed, condition] for condition in conditions]
        walking = [case["walking"] for case in cases]
        _need(
            all(type(value) is int and 0 <= value <= 32 for value in walking),
            "J2 walking counts must be integers out of 32",
        )
        errors = {
            key: [float(case[key]) for case in cases]
            for key in ("diagnostic_error", "offline_error")
        }
        _need(
            all(
                math.isfinite(value) and value >= 0
                for values in errors.values()
                for value in values
            ),
            "Invalid J2 diagnostic error",
        )
        correlations = {
            key: rank_correlation([-value for value in values], walking)
            for key, values in errors.items()
        }
        diagnostic, offline = correlations["diagnostic_error"], correlations["offline_error"]
        per_seed.append(
            dict(
                seed=seed,
                walking_counts=walking,
                diagnostic_spearman=diagnostic,
                offline_spearman=offline,
                improvement=None if diagnostic is None or offline is None else diagnostic - offline,
                diagnostic_auc_any_walking=any_walking_auc(errors["diagnostic_error"], walking),
                offline_auc_any_walking=any_walking_auc(errors["offline_error"], walking),
                secondary_duration_controls=dict(
                    mean_probe_controls_spearman=rank_correlation(
                        [float(case["mean_probe_controls"]) for case in cases], walking
                    )
                    if all("mean_probe_controls" in case for case in cases)
                    else None,
                    negative_probe_fall_count_spearman=rank_correlation(
                        [-float(case["probe_falls"]) for case in cases], walking
                    )
                    if all("probe_falls" in case for case in cases)
                    else None,
                    primary_gate_uses_these_controls=False,
                ),
            )
        )
    eligible = all(row["improvement"] is not None for row in per_seed)
    result: dict[str, Any] = dict(
        eligible=eligible,
        independent_test_units="six whole fresh learner seeds; thirteen related conditions each",
        score_timing="post-fit execution diagnostic, not pre-update teachability",
        prefix_scope="probe disagreement pools actual unequal-length pre-fall prefixes; duration/fall controls are secondary",
        generalization_scope="fresh learner initialization/minibatch streams with fixed corpora and noise profiles",
        conditions=conditions,
        per_seed=per_seed,
        confirmed=False,
        automatic_followup_authorized=False,
        secondary_pooled_auc=dict(
            diagnostic=any_walking_auc(
                [record["diagnostic_error"] for record in records],
                [record["walking"] for record in records],
            ),
            offline=any_walking_auc(
                [record["offline_error"] for record in records],
                [record["walking"] for record in records],
            ),
            exploratory=True,
            positive_definition="at least one walking success out of 32; not competence",
            related_conditions_not_independent=True,
        ),
    )
    if eligible:
        differences = [row["improvement"] for row in per_seed]
        test = sign_flip_oracle(differences)
        utility = math.fsum(row["diagnostic_spearman"] for row in per_seed) / 6
        noninferior = sum(row["improvement"] >= 0 for row in per_seed)
        result.update(
            exact_sign_flip=test,
            mean_diagnostic_spearman=utility,
            streams_at_least_offline=noninferior,
            confirmed=(
                test["observed_mean_improvement"] >= 0.15
                and test["p_value"] <= 0.05
                and utility >= 0.7
                and noninferior >= 5
            ),
        )
    return result


def episode_oracle(
    states: np.ndarray,
    actions: np.ndarray,
    native_costs: np.ndarray,
    terminal_costs: np.ndarray,
    velocities: np.ndarray,
    done: np.ndarray,
    horizon: int,
    task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute finite native outcomes without fabricating post-fall observations."""
    controls = len(actions)
    _need(
        type(horizon) is int
        and 0 < controls <= horizon
        and states.shape == (controls + 1, 19)
        and actions.shape == (controls, 6)
        and all(
            value.shape == (controls,) for value in (native_costs, terminal_costs, velocities, done)
        ),
        "J2 episode arrays have inconsistent control/state lengths",
    )
    _need(
        all(
            np.isfinite(value).all()
            for value in (states, actions, native_costs, terminal_costs, velocities)
        )
        and np.all(np.abs(actions) <= 1)
        and done.dtype.kind in "bu"
        and np.all((done == 0) | (done == 1))
        and not np.any(done[:-1]),
        "J2 episode has nonfinite, unbounded or post-fall controls",
    )
    fall = bool(done[-1])
    survived = controls == horizon and not fall
    _need(fall or survived, "J2 nonterminal episode stopped before its horizon")
    _need(np.all(terminal_costs[:-1] == 0), "J2 terminal cost charged before terminal control")
    _need((terminal_costs[-1] > 0) == fall, "J2 native fall charge differs from termination")
    if task is not None:
        post = states[1:]
        healthy = (post[:, 2] > 0.8) & (post[:, 2] < 2) & (post[:, 3] > -1) & (post[:, 3] < 1)
        _need(np.array_equal(velocities, post[:, 10]), "J2 saved velocity differs from state")
        _need(np.array_equal(done.astype(bool), ~healthy), "J2 terminal/health states differ")
        _need(
            task["cost_style"] == "target_velocity"
            and task["target_velocity"] == 1.5
            and task["terminal_unhealthy_weight"] == 20
            and task["unhealthy_cost_weight"] == 5,
            "J2 native task contract changed",
        )
        running = (
            task["vel_cost_weight"] * np.abs(task["target_velocity"] - post[:, 10])
            + task["vertical_velocity_cost_weight"] * post[:, 11] ** 2
            + task["angle_cost_weight"] * post[:, 3] ** 2
            + task["unhealthy_cost_weight"] * (~healthy)
            + task["ctrl_cost_weight"] * np.sum(actions**2, axis=1)
        )
        terminal = task["terminal_unhealthy_weight"] * (~healthy).astype(float)
        _need(np.array_equal(terminal, terminal_costs), "J2 native terminal charged incorrectly")
        _need(
            np.allclose(running + terminal, native_costs, rtol=0, atol=1e-10),
            "J2 native costs differ from state/action task equation",
        )
        _need(
            np.allclose(np.diff(states[:, 0]), 0.008, rtol=0, atol=1e-10),
            "J2 actual control timing differs",
        )
    mae = float(np.abs(velocities - 1.5).mean())
    native = float(native_costs.sum(dtype=np.float64))
    terminal = float(terminal_costs.sum(dtype=np.float64))
    tail = 5.0 * (horizon - controls) if fall else 0.0
    return dict(
        controls=controls,
        horizon=horizon,
        done=fall,
        survived=survived,
        walking=bool(horizon == 1000 and survived and mae <= 0.3),
        velocity_mae=mae,
        raw_cost=native - terminal,
        native_terminal=terminal,
        native_cost=native,
        absorbing_cost=tail,
        complete_cost=native + tail,
    )


def array_fingerprint(value: np.ndarray) -> str:
    """Existing policy fit-data hash contract, independent of the fitter."""
    array = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256(json.dumps([list(array.shape), str(array.dtype)]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


class SavedReader:
    """Bounded ordinary-file reads, exact seals and independent NPZ decoding."""

    def __init__(self, root: Path, limits: dict[str, Any]):
        self.root = root.resolve()
        self.started = time.monotonic()
        self.deadline = self.started + limits.get("audit_wall_seconds", MAX_WALL_SECONDS)
        self.maximum = limits.get("audit_read_bytes", MAX_READ_BYTES)
        self.max_rss = limits.get("audit_peak_rss_bytes", MAX_RSS_BYTES)
        _need(
            0 < self.maximum <= MAX_READ_BYTES
            and 0 < self.deadline - self.started <= MAX_WALL_SECONDS
            and 0 < self.max_rss <= MAX_RSS_BYTES,
            "J2 audit allocation exceeds hard ceilings",
        )
        self.read_bytes = 0
        self.decoded_array_bytes = 0
        self.files: dict[str, dict[str, Any]] = {}
        self.identities: dict[Path, tuple[int, ...]] = {}

    def check(self) -> None:
        if sys.platform.startswith("linux"):
            fields = {}
            for line in Path("/proc/self/status").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in {"VmHWM", "VmRSS"}:
                    parts = value.split()
                    _need(len(parts) == 2 and parts[1] == "kB", "Malformed J2 Linux RSS metadata")
                    fields[key] = int(parts[0]) * 1024
            _need(set(fields) == {"VmHWM", "VmRSS"}, "Missing J2 Linux RSS metadata")
            rss = max(fields.values())
        else:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss = int(rss if sys.platform == "darwin" else rss * 1024)
        _need(
            time.monotonic() <= self.deadline
            and rss <= self.max_rss
            and self.read_bytes + self.decoded_array_bytes <= self.maximum,
            "J2 audit wall/RSS ceiling exhausted",
        )

    def path(self, name: str | Path) -> Path:
        value = Path(name)
        path = value if value.is_absolute() else self.root / value
        _need(path.resolve().is_relative_to(self.root), "J2 evidence path escapes the run")
        relative = path.relative_to(self.root)
        _need(
            ".." not in relative.parts
            and all(
                not (self.root / Path(*relative.parts[:index])).is_symlink()
                for index in range(1, len(relative.parts) + 1)
            )
            and path.is_file(),
            "Missing or symlinked J2 saved input: " + str(name),
        )
        return path

    def read(self, name: str | Path | dict[str, Any], maximum: int = 128 * 1024**2) -> bytes:
        self.check()
        binding = name if isinstance(name, dict) else None
        path = self.path(binding["file"] if binding else name)
        before = path.stat()

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

        _need(
            before.st_size <= maximum
            and self.read_bytes + self.decoded_array_bytes + before.st_size <= self.maximum,
            "J2 audit read ceiling exhausted before input",
        )
        _need(
            path not in self.identities or self.identities[path] == identity(before),
            "J2 input changed between reads",
        )
        payload = path.read_bytes()
        self.read_bytes += len(payload)
        _need(
            len(payload) == before.st_size and identity(path.stat()) == identity(before),
            "J2 input changed during read",
        )
        actual = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        if binding is not None:
            _need(
                all(binding.get(key) == value for key, value in actual.items()),
                "J2 input byte binding differs",
            )
        self.files[str(path.relative_to(self.root))] = actual
        self.identities[path] = identity(before)
        return payload

    def json(self, name: str | Path | dict[str, Any]) -> Any:
        def pairs(items):
            result = {}
            for key, value in items:
                _need(key not in result, "Duplicate J2 JSON key")
                result[key] = value
            return result

        return json.loads(
            self.read(name),
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(AuditError("Nonfinite J2 JSON")),
        )

    def npz(self, name: str | Path | dict[str, Any]) -> dict[str, np.ndarray]:
        payload = self.read(name)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            _need(
                0 < len(members) <= 64
                and len({value.filename for value in members}) == len(members)
                and sum(value.file_size for value in members) <= 512 * 1024**2
                and self.read_bytes
                + self.decoded_array_bytes
                + sum(value.file_size for value in members)
                <= self.maximum,
                "J2 NPZ decoded allocation or member roster exceeded",
            )
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            result = {key: archive[key] for key in archive.files}
        _need(
            all(not value.dtype.hasobject for value in result.values()), "J2 object array forbidden"
        )
        self.decoded_array_bytes += sum(value.file_size for value in members)
        self.check()
        return result

    def external_binding(self, binding: dict[str, Any]) -> Path:
        """Read only a declared frozen external source, streaming its entire hash."""
        self.check()
        path = Path(binding.get("path", binding.get("file", ""))).resolve()
        _need(
            path.is_file() and 0 <= binding["bytes"] <= 2 * 1024**3,
            "Missing or oversized J2 declared external source",
        )
        before = path.stat()
        _need(
            before.st_size == binding["bytes"]
            and self.read_bytes + self.decoded_array_bytes + before.st_size <= self.maximum,
            "J2 external input size/read allocation differs",
        )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
                self.read_bytes += len(block)
                self.check()
        after = path.stat()
        _need(
            (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            and digest.hexdigest() == binding["sha256"],
            "J2 declared external source hash/identity differs",
        )
        self.identities[path] = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        self.files[str(path)] = dict(bytes=after.st_size, sha256=digest.hexdigest())
        return path

    def verify_unchanged(self) -> None:
        for path, expected in self.identities.items():
            stat = path.stat()
            actual = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
            _need(actual == expected, "J2 audited input changed before closure")


class FunctionalActors:
    """Independent named-tensor actor equations; no network or optimizer construction."""

    def __init__(self, reader: SavedReader, maximum: int):
        _need(0 < maximum <= MAX_NEURAL_ROWS, "J2 audit forward-row allocation exceeds ceiling")
        self.reader, self.maximum = reader, maximum
        self.payloads: dict[str, Any] = {}
        self.requested = self.completed = self.unknown = self.calls = self.loads = 0

    def load(self, binding: str | dict[str, Any]) -> dict[str, Any]:
        name = binding["file"] if isinstance(binding, dict) else binding
        key = str(self.reader.path(name))
        if key in self.payloads:
            if isinstance(binding, dict):
                actual = self.reader.files[str(Path(key).relative_to(self.reader.root))]
                _need(
                    all(binding.get(field) == value for field, value in actual.items()),
                    "J2 cached checkpoint has conflicting byte bindings",
                )
            return self.payloads[key]
        import torch

        payload = torch.load(
            io.BytesIO(self.reader.read(binding)), map_location="cpu", weights_only=True
        )
        self.loads += 1
        if "state_dict" in payload:
            state = payload["state_dict"]
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
            _need(
                payload["schema"] == "walker-tanh-policy-v1" and set(state) == set(expected),
                "J2 policy schema/architecture differs",
            )
            digest = hashlib.sha256(b"walker-tanh-policy-v1")
            for name, tensor in sorted(state.items()):
                _need(
                    tuple(tensor.shape) == expected[name]
                    and tensor.dtype == torch.float32
                    and bool(torch.isfinite(tensor).all()),
                    "Invalid J2 actor tensor",
                )
                array = tensor.contiguous().numpy()
                digest.update(
                    json.dumps(
                        [name, list(array.shape), str(array.dtype)], separators=(",", ":")
                    ).encode()
                )
                digest.update(array.tobytes())
            _need(
                digest.hexdigest() == payload["policy_hash"], "J2 named actor tensor hash differs"
            )
            _need(bool((state["observation_std"] > 0).all()), "J2 normalization is not positive")
            if isinstance(binding, dict) and "policy_hash" in binding:
                _need(
                    binding["policy_hash"] == payload["policy_hash"],
                    "J2 endpoint actor binding differs",
                )
        else:
            _need(
                "policy" in payload and "actor" in payload["policy"],
                "J2 reference checkpoint schema differs",
            )
        self.payloads[key] = payload
        return payload

    def predict(
        self, binding: str | dict[str, Any], observations: np.ndarray, *, batch_one: bool
    ) -> np.ndarray:
        payload = self.load(binding)
        observations = np.asarray(observations, dtype=np.float32)
        _need(
            observations.ndim == 2
            and observations.shape[1] == 17
            and np.isfinite(observations).all(),
            "J2 actor observation shape differs",
        )
        _need(
            self.requested + len(observations) <= self.maximum,
            "J2 audit forward ceiling exhausted before operation",
        )
        import torch
        import torch.nn.functional as functional

        reference = "state_dict" not in payload
        state = payload["policy"]["actor"] if reference else payload["state_dict"]
        batches = (
            (row[None] if reference else row for row in observations)
            if batch_one
            else [observations]
        )
        predictions = []
        with torch.no_grad():
            for batch in batches:
                rows = 1 if batch_one else len(observations)
                self.requested += rows
                try:
                    if self.calls % 256 == 0:
                        self.reader.check()
                    self.calls += 1
                    value = torch.as_tensor(np.array(batch, copy=True))
                    if reference:
                        value = (value - state["obs_mean"]) / state["obs_std"]
                        for number in (1, 2):
                            value = functional.relu(
                                functional.linear(
                                    value, state[f"fc{number}.weight"], state[f"fc{number}.bias"]
                                )
                            )
                        mean = functional.linear(
                            value, state["fc_mean.weight"], state["fc_mean.bias"]
                        )
                        # The original actor also computes this unused head.
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
                    predictions.append(value.numpy().reshape(-1, 6))
                    self.completed += rows
                except BaseException:
                    self.unknown += rows
                    raise
        return np.concatenate(predictions)

    def verify(
        self,
        binding: str | dict[str, Any],
        observations: np.ndarray,
        expected: np.ndarray,
        *,
        batch_one: bool,
    ) -> None:
        actual = self.predict(binding, observations, batch_one=batch_one)
        _need(
            actual.shape == expected.shape and np.isfinite(expected).all(),
            "J2 saved prediction shape differs",
        )
        _need(
            actual.astype(expected.dtype).tobytes() == np.ascontiguousarray(expected).tobytes(),
            "J2 saved actor prediction differs at declared batching",
        )


def _same(actual: Any, expected: Any, label: str, tolerance: float = 1e-10) -> None:
    if isinstance(expected, dict):
        _need(
            isinstance(actual, dict) and set(actual) == set(expected),
            "J2 dictionary roster differs: " + label,
        )
        for key, value in expected.items():
            _same(actual[key], value, label + "/" + str(key), tolerance)
    elif isinstance(expected, list):
        _need(
            isinstance(actual, list) and len(actual) == len(expected),
            "J2 list roster differs: " + label,
        )
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _same(left, right, label + f"/{index}", tolerance)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        _need(
            type(actual) is type(expected) and actual == expected,
            "J2 saved value differs: " + label,
        )
    elif isinstance(expected, int):
        _need(type(actual) is int and actual == expected, "J2 saved integer differs: " + label)
    else:
        _need(
            isinstance(actual, (float, int))
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
            "J2 saved number differs: " + label,
        )


def verify_fit(
    reader: SavedReader,
    actors: FunctionalActors,
    directory: str,
    seed: int,
    condition: str,
    corpus: dict[str, np.ndarray],
    stream: dict[str, np.ndarray],
    parent: dict[str, Any],
    paired_initial: dict[int, str],
) -> dict[str, Any]:
    """Bind every update/tape/scalar and check first-batch loss, without updates."""
    import torch

    fit = reader.json(directory + "/fit.json")
    arrays = reader.npz(directory + "/fit_arrays.npz")
    _need(
        fit["schema"] == "walker-adam-proximal-fit-v1"
        and fit["initial_file"] == "initial.pt"
        and fit["tape_file"] == f"streams/{seed}.npz",
        "J2 fitting schema or source ownership differs",
    )
    initial_bytes = reader.read(directory + "/initial.pt", maximum=4 * 1024**2)
    initial = torch.load(io.BytesIO(initial_bytes), map_location="cpu", weights_only=True)
    actors.loads += 1
    _need(
        hashlib.sha256(initial_bytes).hexdigest() == fit["initial_package_sha256"],
        "J2 initial package byte binding differs",
    )
    _need(
        set(initial)
        == {
            "state_dict",
            "adam",
            "rng",
            "private_rng_state",
            "global_torch_rng",
            "global_numpy_state",
        },
        "J2 initial package roster differs",
    )
    _need(
        optimizer_fingerprint(initial["state_dict"]) == optimizer_fingerprint(parent["state_dict"]),
        "J2 fit did not start from paired parent tensors",
    )
    _need(
        optimizer_fingerprint(initial["private_rng_state"])
        == optimizer_fingerprint(parent["private_rng_state"]),
        "J2 initial private Torch stream differs",
    )
    fingerprint = optimizer_fingerprint(initial)
    _need(
        seed not in paired_initial or paired_initial[seed] == fingerprint,
        "J2 label contrasts changed incoming tensors/Adam/global/private RNG",
    )
    paired_initial[seed] = fingerprint
    _need(
        not initial["adam"]["state"] and len(initial["adam"]["param_groups"]) == 1,
        "J2 initial Adam is not fresh and empty",
    )
    tape = stream["indices"]
    _need(
        tape.dtype == np.int64 and tape.shape == (10000, 256),
        "J2 minibatch tape dtype/shape differs",
    )
    _need(
        np.array_equal(initial["global_torch_rng"].numpy(), stream["global_torch_rng"]),
        "J2 actual initial global Torch state differs from frozen stream",
    )
    _same(
        initial["global_numpy_state"],
        json.loads(stream["global_numpy_state_json"].item()),
        "initial global NumPy state",
        0,
    )
    _same(
        initial["global_numpy_state"],
        parent["metadata"]["global_numpy_state"],
        "parent/global NumPy state",
        0,
    )
    _same(
        initial["global_torch_rng"].tolist(),
        parent["metadata"]["global_torch_rng_state"],
        "parent/global Torch state",
        0,
    )
    _need(
        hashlib.sha256(stream["global_torch_rng"].tobytes()).hexdigest()
        == fit["initial_global_rng_sha256"],
        "J2 global Torch byte binding differs",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    _same(initial["rng"], rng.bit_generator.state, "initial PCG")
    hashes = arrays["observed_tape_update_hashes"]
    _need(
        hashes.dtype == np.uint8 and hashes.shape == (10000, 32),
        "J2 observed index digest roster differs",
    )
    for index, row in enumerate(tape):
        _need(
            np.array_equal(row, rng.integers(0, 9000, size=256)),
            "J2 minibatches differ from declared fresh PCG stream",
        )
        _need(
            hashlib.sha256(row.tobytes()).digest() == hashes[index].tobytes(),
            "J2 actual observed minibatch digest differs",
        )
    tape_hash = hashlib.sha256(np.ascontiguousarray(tape).tobytes()).hexdigest()
    _need(
        fit["tape_sha256"] == fit["observed_tape_sha256"] == tape_hash,
        "J2 full observed/frozen tape binding differs",
    )
    child_binding = fit["checkpoint"]
    _need(
        reader.path(child_binding["file"]) == reader.path(directory + "/child.pt"),
        "J2 child checkpoint escapes its learning case",
    )
    child = actors.load(child_binding)
    state = child["optimizer_state"]
    _need(
        state["schema"] == "walker-adam-proximal-fit-v1"
        and fit["parameter_count"] == fit["proximal_divisor"] == 19590,
        "J2 fit architecture/divisor metadata differs",
    )
    _same(
        initial["global_numpy_state"],
        child["metadata"]["global_numpy_state"],
        "terminal global NumPy state",
        0,
    )
    _same(
        initial["global_torch_rng"].tolist(),
        child["metadata"]["global_torch_rng_state"],
        "terminal global Torch state",
        0,
    )
    expected_config = dict(
        batch_size=256,
        lr=0.0003,
        eta_theta=0.0,
        gradient_clip=0.5,
        prox_reduction="mean",
        anchor_refresh=0,
    )
    _same(fit["fit_config"], expected_config, "fit config", 0)
    _same(state["fit_config"], expected_config, "endpoint config", 0)
    expected_data = dict(
        observations_sha256=array_fingerprint(corpus["observations"]),
        actions_sha256=array_fingerprint(corpus["labels__" + condition]),
        rows=9000,
    )
    _same(fit["fit_data"], expected_data, "fit corpus", 0)
    _same(state["fit_data"], expected_data, "endpoint corpus", 0)
    _need(
        fit["completed_updates"]
        == fit["cumulative_updates"]
        == fit["fit_updates"]
        == state["updates"]
        == state["fit_updates"]
        == 10000,
        "J2 fixed update endpoint differs",
    )
    _need(
        fit["failed"] is False
        and fit["resume_fit"] is False
        and fit["optimizer_continued"] is False
        and fit["anchor_refreshes"] == [],
        "J2 update block changed its fresh fixed recipe",
    )
    _need(
        fit["parent_policy_hash"]
        == fit["anchor_policy_hash"]
        == state["parent_policy_hash"]
        == state["anchor_policy_hash"]
        == parent["policy_hash"],
        "J2 incoming actor/anchor hash differs",
    )
    _need(
        fit["child_policy_hash"] == state["policy_hash"] == child["policy_hash"],
        "J2 fitted actor owner differs",
    )
    _same(state["rng"], rng.bit_generator.state, "terminal PCG", 0)
    _need(
        optimizer_fingerprint(state["anchor"]) == optimizer_fingerprint(parent["state_dict"]),
        "J2 endpoint anchor tensors differ",
    )
    _need(
        optimizer_fingerprint(child["private_rng_state"])
        == optimizer_fingerprint(parent["private_rng_state"]),
        "J2 private actor RNG changed during deterministic fit",
    )
    _need(
        len(state["adam"]["state"]) == 6 and len(state["adam"]["param_groups"]) == 1,
        "J2 Adam endpoint parameter roster differs",
    )
    _need(
        state["parameter_names"]
        == [f"network.{layer}.{kind}" for layer in (0, 2, 4) for kind in ("weight", "bias")],
        "J2 optimizer parameter order differs",
    )
    for group in [*initial["adam"]["param_groups"], *state["adam"]["param_groups"]]:
        _need(group["params"] == list(range(6)), "J2 Adam parameter identifiers differ")
        for key, expected in dict(
            lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=False, fused=False
        ).items():
            _need(group[key] == expected, "J2 Adam setting differs: " + key)
    for parameter, value in state["adam"]["state"].items():
        tensor = parent["state_dict"][state["parameter_names"][parameter]]
        _need(
            float(value["step"]) == 10000
            and value["exp_avg"].shape == value["exp_avg_sq"].shape == tensor.shape
            and value["exp_avg"].dtype == value["exp_avg_sq"].dtype == tensor.dtype
            and bool(torch.isfinite(value["exp_avg"]).all())
            and bool(torch.isfinite(value["exp_avg_sq"]).all())
            and bool((value["exp_avg_sq"] >= 0).all()),
            "J2 Adam moments/steps invalid",
        )
    for name in ("observation_mean", "observation_std"):
        _need(
            torch.equal(child["state_dict"][name], parent["state_dict"][name]),
            "J2 fitting changed corpus normalization",
        )
    curve = fit["learning_curve"]
    _need(len(curve) == 10000, "J2 curve omits updates")
    for index, row in enumerate(curve):
        _need(row["update"] == row["fit_update"] == index + 1, "J2 curve update sequence differs")
        _need(
            all(
                math.isfinite(row[key]) and row[key] >= 0
                for key in ("action_mse", "proximal_loss", "total_loss", "unclipped_gradient_norm")
            ),
            "J2 curve has invalid scalars",
        )
        _need(
            row["proximal_loss"] == 0 and row["total_loss"] == row["action_mse"],
            "J2 action-only loss identity differs",
        )
    first_predictions = actors.predict(
        f"parents/{seed}.pt", corpus["observations"][tape[0]], batch_one=False
    )
    first_targets = corpus["labels__" + condition][tape[0]]
    with torch.no_grad():
        first_loss = float(
            torch.mean((torch.as_tensor(first_predictions) - torch.as_tensor(first_targets)) ** 2)
        )
    _need(
        math.isclose(first_loss, curve[0]["action_mse"], rel_tol=0, abs_tol=1e-7),
        "J2 first minibatch loss differs from incoming actor",
    )
    for category, amount in dict(
        neural_rows=2560000, backward_rows=2560000, optimizer_steps=10000
    ).items():
        _same(
            fit["work"][category],
            dict(requested=amount, completed=amount, unknown_partial=0),
            "fit work/" + category,
            0,
        )
    for key, observations, binding in (
        ("held_predictions", corpus["held_observations"], child_binding),
        ("incoming_predictions", corpus["observations"], f"parents/{seed}.pt"),
        ("train_predictions", corpus["observations"], child_binding),
    ):
        _need(arrays[key].dtype == np.float32, "J2 saved batched prediction dtype differs")
        actors.verify(binding, observations, arrays[key], batch_one=False)
    offline = float(np.mean((arrays["held_predictions"] - corpus["held_clean"]) ** 2))
    _need(
        math.isclose(offline, fit["held_clean_mse"], rel_tol=0, abs_tol=1e-12),
        "J2 held reconstruction error differs",
    )
    return dict(
        checkpoint=child_binding,
        offline_error=offline,
        train_error=float(
            np.mean((arrays["train_predictions"] - corpus["labels__" + condition]) ** 2)
        ),
        curve_points=10000,
        first_batch_loss=first_loss,
        gradients_repeated=False,
        all_curve_losses_recomputed=False,
    )


def verify_traces(
    reader: SavedReader,
    actors: FunctionalActors,
    filename: str,
    task: dict[str, Any],
    expected: list[tuple[str, int, int]],
    actor: str | dict[str, Any] | None,
    reference: str | dict[str, Any] | None,
    contexts: dict[tuple[str, int], dict[str, str]],
    saved_outcomes: list[dict[str, Any]] | None = None,
    replay_sources: dict[int, dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Verify every executed state, action, outcome and probe-reference prediction."""
    arrays = reader.npz(filename)
    state_widths = dict(states=19, observations=17, warmstarts=9, integrations=91)
    step_widths = dict(actions=6, reference_actions=6)
    metadata_keys = {
        "episode_step_offsets",
        "episode_state_offsets",
        "episode_seeds",
        "episode_kinds",
        "episode_horizons",
    }
    scalar_steps = {"has_reference", "costs", "fall_costs", "velocities", "done"}
    _need(
        set(arrays) == set(state_widths) | set(step_widths) | metadata_keys | scalar_steps,
        "J2 trace array roster differs",
    )
    step_offsets, state_offsets = arrays["episode_step_offsets"], arrays["episode_state_offsets"]
    count = len(expected)
    _need(
        step_offsets.dtype == state_offsets.dtype == np.int64
        and step_offsets.shape == state_offsets.shape == (count + 1,)
        and step_offsets[0] == state_offsets[0] == 0
        and np.all(np.diff(step_offsets) > 0)
        and np.array_equal(np.diff(state_offsets), np.diff(step_offsets) + 1),
        "J2 episode offsets omit or fabricate states/controls",
    )
    steps, states = int(step_offsets[-1]), int(state_offsets[-1])
    for key, width in state_widths.items():
        value = arrays[key]
        _need(
            value.dtype == np.float64
            and value.shape == (states, width)
            and np.isfinite(value).all(),
            "J2 trace state array invalid: " + key,
        )
    for key, width in step_widths.items():
        _need(
            arrays[key].dtype == np.float64 and arrays[key].shape == (steps, width),
            "J2 trace action array invalid: " + key,
        )
    for key in scalar_steps:
        dtype = np.dtype(bool if key in {"has_reference", "done"} else np.float64)
        _need(
            arrays[key].dtype == dtype and arrays[key].shape == (steps,),
            "J2 scalar control array invalid: " + key,
        )
    _need(
        arrays["episode_seeds"].dtype == arrays["episode_horizons"].dtype == np.int64
        and arrays["episode_kinds"].dtype.kind == "U"
        and all(
            arrays[key].shape == (count,)
            for key in ("episode_seeds", "episode_horizons", "episode_kinds")
        ),
        "J2 episode metadata roster differs",
    )
    if saved_outcomes is not None:
        _need(len(saved_outcomes) == count, "J2 saved episode outcome roster differs")
    outcomes, gaps, short_gaps = [], [], []
    controls_by_kind: dict[str, int] = {}
    snapshot_reads = 0
    for index, (kind, seed, horizon) in enumerate(expected):
        _need(
            arrays["episode_kinds"][index] == kind
            and int(arrays["episode_seeds"][index]) == seed
            and int(arrays["episode_horizons"][index]) == horizon,
            "J2 declared episode seed/kind/horizon differs",
        )
        step = slice(int(step_offsets[index]), int(step_offsets[index + 1]))
        state = slice(int(state_offsets[index]), int(state_offsets[index + 1]))
        observations = arrays["observations"][state]
        physical = arrays["states"][state]
        integration = arrays["integrations"][state]
        warmstart = arrays["warmstarts"][state]
        actions = arrays["actions"][step]
        _need(
            np.array_equal(integration[:, :19], physical)
            and np.array_equal(integration[:, 19:28], warmstart)
            and np.array_equal(integration[1:, 28:34], actions),
            "J2 integration/state/warmstart/control join differs",
        )
        expected_observations = np.hstack((physical[:, 2:10], np.clip(physical[:, 10:19], -10, 10)))
        _need(
            np.array_equal(expected_observations, observations),
            "J2 native observation interface differs",
        )
        outcome = episode_oracle(
            physical,
            actions,
            arrays["costs"][step],
            arrays["fall_costs"][step],
            arrays["velocities"][step],
            arrays["done"][step],
            horizon,
            task,
        )
        native_fields = dict(
            kind=kind,
            seed=seed,
            controls=outcome["controls"],
            horizon=horizon,
            done=outcome["done"],
            survived=outcome["survived"],
            walking=outcome["walking"],
            mae=outcome["velocity_mae"],
            native_cost=outcome["native_cost"],
            raw_cost=outcome["raw_cost"],
            terminal_cost=outcome["native_terminal"],
            absorbing_cost=outcome["absorbing_cost"],
            cost=outcome["complete_cost"],
            observed_seconds=0.008 * outcome["controls"],
            right_censored=not outcome["done"],
        )
        if saved_outcomes is not None:
            _same(saved_outcomes[index], native_fields, "episode outcome")
        initial = {
            key: hashlib.sha256(arrays[key][int(state_offsets[index])].tobytes()).hexdigest()
            for key in state_widths
        }
        context_key = ("evaluation" if kind == "reference" else kind, seed)
        if kind != "replay":
            _need(
                context_key not in contexts or contexts[context_key] == initial,
                "J2 paired reset initial simulator contexts differ",
            )
            contexts[context_key] = initial
        has_reference = arrays["has_reference"][step]
        reference_actions = arrays["reference_actions"][step]
        if kind == "probe":
            _need(
                reference is not None
                and has_reference.all()
                and np.isfinite(reference_actions).all(),
                "J2 probe reference supervision is missing",
            )
            actors.verify(reference, observations[:-1], reference_actions, batch_one=True)
            errors = np.mean((actions - reference_actions) ** 2, axis=1)
            gaps.append(errors)
            short_gaps.append(errors[:25])
        else:
            _need(
                not has_reference.any() and np.isnan(reference_actions).all(),
                "J2 missing reference predictions were fabricated",
            )
        if actor is not None:
            actors.verify(actor, observations[:-1], actions, batch_one=True)
        if replay_sources is not None:
            source = replay_sources[seed]
            for key in ("states", "warmstarts", "integrations"):
                _need(
                    np.allclose(arrays[key][state], source[key], rtol=0, atol=1e-10),
                    "J2 unchanged-action replay differs: " + key,
                )
            _need(
                np.array_equal(actions, source["actions"].astype(np.float64))
                and np.allclose(arrays["costs"][step], source["costs"], rtol=0, atol=1e-10),
                "J2 unchanged-action replay actions/costs differ",
            )
        controls_by_kind[kind] = controls_by_kind.get(kind, 0) + outcome["controls"]
        snapshot_reads += outcome["controls"] + 1
        outcomes.append(native_fields)
    return dict(
        episodes=outcomes,
        controls_by_kind=controls_by_kind,
        snapshot_reads=snapshot_reads,
        diagnostic_error=float(np.mean(np.concatenate(gaps))) if gaps else None,
        diagnostic_short_error=float(np.mean(np.concatenate(short_gaps))) if short_gaps else None,
    )


def verify_corpus(reader: SavedReader, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Rejoin all fixed training/held data to original saved sources and jitter recipe."""
    import h5py
    from scipy import signal

    manifest = reader.json("inputs/manifest.json")
    _same(
        reader.files["inputs/manifest.json"],
        config["data"]["prepared_manifest"],
        "launch prepared-manifest pin",
        0,
    )
    for name, binding in manifest["files"].items():
        reader.read(dict(binding, file="inputs/" + name))
    for name, pin in config["inputs"].items():
        _need(
            all(manifest["source_inputs"][name][key] == pin[key] for key in ("bytes", "sha256")),
            "J2 preparation source pin differs: " + name,
        )
    corpus = reader.npz("inputs/corpus.npz")
    source = reader.external_binding(config["inputs"]["reference_h5"])
    reference: dict[str, list[np.ndarray]] = {}
    with h5py.File(source, "r") as handle:
        names = sorted(name for name in handle["arrays"] if "episode_k1_primary_reference_" in name)
        names.sort(key=lambda name: int(name.rsplit("_", 1)[-1]))
        _need(len(names) == 16, "J2 reference source lacks sixteen episodes")
        for field in ("states", "observations", "actions", "integrations", "warmstarts", "costs"):
            values = []
            for name in names:
                dataset = handle["arrays"][name][field]
                _need(
                    reader.read_bytes
                    + reader.decoded_array_bytes
                    + dataset.size * dataset.dtype.itemsize
                    <= reader.maximum,
                    "J2 selected H5 dataset read exceeds reservation",
                )
                value = dataset[:]
                reader.read_bytes += value.nbytes
                values.append(value)
            reference[field] = values
            _need(
                np.array_equal(np.stack(values), corpus["sac_" + field]),
                "J2 original reference/corpus join differs: " + field,
            )
    observations = np.concatenate(reference["observations"][:9]).astype(np.float32)
    clean = np.concatenate(reference["actions"][:9]).astype(np.float64)
    held_observations = np.concatenate(reference["observations"][9:]).astype(np.float32)
    held_clean = np.concatenate(reference["actions"][9:]).astype(np.float64)
    for key, value in dict(
        observations=observations,
        clean=clean,
        held_observations=held_observations,
        held_clean=held_clean,
    ).items():
        _need(
            np.array_equal(corpus[key], value), "J2 complete train/held row binding differs: " + key
        )
    _need(
        observations.shape == (9000, 17) and held_observations.shape == (7000, 17),
        "J2 unique row/trajectory budget changed",
    )
    _need(
        np.array_equal(corpus["normalization_mean"], observations.mean(0))
        and np.array_equal(corpus["normalization_std"], np.maximum(observations.std(0), 1e-6)),
        "J2 corpus-only normalization differs",
    )
    original = reader.external_binding(config["inputs"]["mppi_traces"])
    with np.load(original, allow_pickle=False) as archive:
        offsets, actions = archive["episode_step_offsets"], archive["actions"]
    reader.read_bytes += offsets.nbytes + actions.nbytes
    reader.check()
    walks = [
        actions[int(start) : int(end)].astype(np.float64)
        for start, end in zip(offsets[:-1], offsets[1:], strict=True)
        if int(end - start) == 1000
    ][:9]
    _need(len(walks) == 9, "J2 original MPPI jitter source lacks nine complete sequences")
    pooled = np.concatenate(walks)
    rng = np.random.default_rng(20261102)
    noises = {"mppi": [(value - pooled.mean(0)) / pooled.std(0) for value in walks]}
    noises["white"] = [rng.standard_normal((1000, 6)) for _ in range(9)]
    low_pass = signal.butter(4, 2, fs=125, output="sos")
    slow = [signal.sosfiltfilt(low_pass, rng.standard_normal((1000, 6)), axis=0) for _ in range(9)]
    slow_std = np.concatenate(slow).std(0)
    noises["slow"] = [value / slow_std for value in slow]
    expected_keys = ["clean"]
    _need(
        np.array_equal(corpus["labels__clean"], np.clip(clean, -1, 1).astype(np.float32)),
        "J2 clean labels differ from original reference",
    )
    for kind in ("mppi", "white", "slow"):
        for share in (0.9, 0.75, 0.5, 0.25):
            key = kind + "_" + str(share).replace(".", "p")
            expected_keys.append(key)
            scale = np.sqrt(clean.var(0) * (1 - share) / share)
            expected = np.clip(clean + scale * np.concatenate(noises[kind]), -1, 1).astype(
                np.float32
            )
            _need(
                np.array_equal(corpus["labels__" + key], expected),
                "J2 fixed label/noise recipe differs: " + key,
            )
    _need(
        set(manifest["conditions"]) == set(expected_keys),
        "J2 fixed thirteen-condition roster differs",
    )
    reader.external_binding(config["inputs"]["reference_checkpoint"])
    _need(
        reader.files["inputs/reference.pt"]["sha256"]
        == config["inputs"]["reference_checkpoint"]["sha256"],
        "J2 reference actor changed between original and copied input",
    )
    reader.verify_unchanged()
    return corpus


def primitive_oracle(traces: list[dict[str, Any]], fits: int, parents: int) -> dict[str, int]:
    """Derive every scientific worker primitive from persisted roster dimensions."""
    controls: dict[str, int] = {}
    episodes: dict[str, int] = {}
    for trace in traces:
        for kind, value in trace["controls_by_kind"].items():
            controls[kind] = controls.get(kind, 0) + value
        for episode in trace["episodes"]:
            kind = episode["kind"]
            episodes[kind] = episodes.get(kind, 0) + 1
    native = sum(controls.values())
    replay_controls = controls.get("replay", 0)
    replay_episodes = episodes.get("replay", 0)
    resets = sum(episodes.values()) - replay_episodes
    actor_rows = native - replay_controls
    probe_reference_rows = controls.get("probe", 0)
    return dict(
        environment_constructions=1,
        checkpoint_loads=1,
        reference_actor_constructions=1,
        policy_constructions=parents + fits,
        optimizer_constructions=fits,
        fits=fits,
        native_controls=native,
        physics_substeps=4 * native,
        neural_rows=2560000 * fits
        + 9000 * parents
        + 16000 * fits
        + actor_rows
        + probe_reference_rows,
        network_calls=10000 * fits + parents + 2 * fits + actor_rows + probe_reference_rows,
        backward_rows=2560000 * fits,
        optimizer_steps=10000 * fits,
        preallocated_index_values=2560000 * parents,
        sampled_index_values=2560000 * fits,
        resets=resets,
        restores=replay_episodes,
        external_forwards=2 * resets + replay_episodes,
        snapshot_reads=native + sum(episodes.values()) + replay_episodes,
    )


def audit_run(run: Path, out: Path) -> dict[str, Any]:
    """Create a bounded independent saved-data verdict, retaining failures as failures."""
    run, out = run.resolve(), out.resolve()
    _need(
        run.is_dir() and not out.is_relative_to(run),
        "J2 audit output must be outside its immutable run",
    )
    out.mkdir(parents=True, exist_ok=False)
    reader = SavedReader(run, {})
    actors: FunctionalActors | None = None
    result: dict[str, Any] = dict(
        schema="walker-jitter-diagnosis-j2-audit-v1",
        passed=False,
        primary_comparison_executed=False,
        new_physics_controls=0,
        new_gradient_rows=0,
        new_optimizer_steps=0,
        automatic_followup_authorized=False,
    )
    max_output = 16 * 1024**2
    try:
        config = reader.json("config.json")
        audit_limits = config["audit"]
        reader.deadline = reader.started + audit_limits["wall_seconds"]
        reader.maximum = audit_limits.get("read_bytes", MAX_READ_BYTES)
        reader.max_rss = audit_limits.get("peak_rss_bytes", MAX_RSS_BYTES)
        max_output = audit_limits.get("output_bytes", max_output)
        _need(
            0 < audit_limits["wall_seconds"] <= MAX_WALL_SECONDS
            and 0 < reader.maximum <= MAX_READ_BYTES
            and 0 < reader.max_rss <= MAX_RSS_BYTES
            and 0 < max_output <= 16 * 1024**2,
            "J2 frozen audit allocation exceeds ceiling",
        )
        _need(
            config["schema"] == "walker-jitter-diagnosis-j2-0"
            and config["launch_ready"] is True
            and config["automatic_followup"] is False,
            "J2 scope or launch status differs",
        )
        seeds = config["seeds"]["training"]
        evaluations, probes = config["seeds"]["evaluation"], config["seeds"]["probe"]
        _need(
            seeds == list(FRESH_SEEDS)
            and len(evaluations) == len(set(evaluations)) == 32
            and len(probes) == len(set(probes)) == 8
            and not set(evaluations) & set(probes),
            "J2 fresh seed/reset namespace differs",
        )
        conditions = [value["id"] for value in config["conditions"]]
        _need(
            conditions
            == ["clean"]
            + [
                kind + "_" + str(share).replace(".", "p")
                for kind in ("mppi", "white", "slow")
                for share in (0.9, 0.75, 0.5, 0.25)
            ],
            "J2 thirteen-condition roster or order differs",
        )
        source_root = Path(config["locations"]["source_export"]).resolve()
        for name, binding in config["source_bindings"].items():
            path = (source_root / name).resolve()
            _need(path.is_relative_to(source_root), "J2 frozen source path escapes export")
            reader.external_binding(dict(binding, path=str(path)))
        own_name = "src/gps/mppi_gps/audits/jitter_diagnosis.py"
        reader.external_binding(
            dict(config["source_bindings"][own_name], path=str(Path(__file__).resolve()))
        )
        corpus = verify_corpus(reader, config)
        worker = reader.json("result.json")
        work = reader.json("work.json")
        result.update(
            worker_status=worker["status"], observed_worker_work=work, worker_work_verified=False
        )
        _same(worker["work"], work, "worker work sidecar", 0)
        _need(not work["unknown_partial"], "J2 worker has uncertain paid work")
        _need(
            worker["status"]
            in {
                "COMPLETED_FIXED_J2_ROSTER",
                "STOPPED_REFERENCE_QUALIFICATION",
                "STOPPED_CLEAN_POSITIVE_CONTROL",
            },
            "J2 worker failed before a scientifically closed stopping point",
        )
        actors = FunctionalActors(reader, audit_limits["neural_rows"])
        import torch

        torch.set_num_threads(1)
        task = config["task"]
        contexts: dict[tuple[str, int], dict[str, str]] = {}
        reference_binding = dict(reader.files["inputs/reference.pt"], file="inputs/reference.pt")
        replay_sources = {}
        expected_replays = []
        for episode in range(9, 16):
            for anchor in range(100, 1000, 100):
                seed = episode * 1000 + anchor
                expected_replays.append(("replay", seed, 25))
                source = {
                    key: corpus["sac_" + key][episode, anchor : anchor + 26]
                    for key in ("states", "warmstarts", "integrations")
                }
                source.update(
                    actions=corpus["sac_actions"][episode, anchor : anchor + 25],
                    costs=corpus["sac_costs"][episode, anchor : anchor + 25],
                )
                replay_sources[seed] = source
        replay = verify_traces(
            reader,
            actors,
            "prerequisites/replay.npz",
            task,
            expected_replays,
            None,
            None,
            contexts,
            replay_sources=replay_sources,
        )
        _same(
            reader.files["prerequisites/replay.npz"],
            worker["prerequisites"]["held_replays"]["traces"],
            "replay container binding",
            0,
        )
        _need(
            worker["prerequisites"]["held_replays"]["episodes"] == 63,
            "J2 replay prerequisite roster differs",
        )
        reference = verify_traces(
            reader,
            actors,
            "prerequisites/reference.npz",
            task,
            [("reference", seed, 1000) for seed in evaluations],
            reference_binding,
            None,
            contexts,
            worker["prerequisites"]["reference"]["episodes"],
        )
        _same(
            reader.files["prerequisites/reference.npz"],
            worker["prerequisites"]["reference"]["traces"],
            "reference container binding",
            0,
        )
        panels = [
            sum(row["walking"] for row in reference["episodes"][start : start + 16])
            for start in (0, 16)
        ]
        _same(
            worker["prerequisites"]["reference"]["panel_walking"],
            panels,
            "reference qualification",
            0,
        )
        reference_passes = min(panels) >= 15
        _need(
            reference_passes == (worker["status"] != "STOPPED_REFERENCE_QUALIFICATION"),
            "J2 reference stopping gate differs",
        )
        traces = [replay, reference]
        records, learning_checks = [], []
        parent_count = 6 if reference_passes else 0
        streams: dict[int, dict[str, np.ndarray]] = {}
        parents: dict[int, dict[str, Any]] = {}
        if reference_passes:
            for seed in seeds:
                parent = actors.load(f"parents/{seed}.pt")
                _need(
                    parent["seed"] == seed and parent["optimizer_state"] is None,
                    "J2 incoming parent is not the declared fresh actor",
                )
                for key in ("mean", "std"):
                    _need(
                        np.array_equal(
                            parent["state_dict"]["observation_" + key].numpy(),
                            corpus["normalization_" + key],
                        ),
                        "J2 parent normalization differs from fixed clean corpus",
                    )
                parents[seed] = parent
                streams[seed] = reader.npz(f"streams/{seed}.npz")
            _need(
                len({parent["policy_hash"] for parent in parents.values()}) == 6,
                "J2 learner streams reuse actor tensors",
            )
        roster = worker["cases"]
        required = list(
            itertools.product(
                conditions
                if worker["status"] == "COMPLETED_FIXED_J2_ROSTER"
                else ["clean"]
                if reference_passes
                else [],
                seeds,
            )
        )
        _need(
            [(case["condition"], case["seed"]) for case in roster] == required
            and worker["actual_fits"] == len(roster),
            "J2 completed or stopped case roster differs",
        )
        paired_initial: dict[int, str] = {}
        for case in roster:
            seed, condition = case["seed"], case["condition"]
            directory = f"cases/{seed}_{condition}"
            _need(
                case["case_id"] == f"{seed}_{condition}"
                and case["summary"] == directory + "/summary.json",
                "J2 learning-case path ownership differs",
            )
            summary = reader.json(case["summary"])
            _need(
                summary["case_id"] == case["case_id"]
                and summary["seed"] == seed
                and summary["condition"]["id"] == condition,
                "J2 saved case identity differs",
            )
            fit = verify_fit(
                reader,
                actors,
                directory,
                seed,
                condition,
                corpus,
                streams[seed],
                parents[seed],
                paired_initial,
            )
            _same(
                summary["checkpoint"],
                fit["checkpoint"],
                "same actor for fitting/evaluation/probes",
                0,
            )
            trace = verify_traces(
                reader,
                actors,
                directory + "/traces.npz",
                task,
                [("probe", value, 100) for value in probes]
                + [("evaluation", value, 1000) for value in evaluations],
                fit["checkpoint"],
                reference_binding,
                contexts,
                summary["episodes"],
            )
            _same(
                reader.files[directory + "/traces.npz"],
                summary["traces"],
                "case trace-container binding",
                0,
            )
            for key, value in dict(
                standing100=trace["diagnostic_error"],
                standing25=trace["diagnostic_short_error"],
                offline_action_mse=fit["offline_error"],
            ).items():
                _need(
                    math.isclose(summary["postfit"][key], value, rel_tol=0, abs_tol=1e-12),
                    "J2 saved post-fit score differs: " + key,
                )
            evaluation_rows = [row for row in trace["episodes"] if row["kind"] == "evaluation"]
            probe_rows = [row for row in trace["episodes"] if row["kind"] == "probe"]
            for key, value in dict(
                probe_mean_controls=float(np.mean([row["controls"] for row in probe_rows])),
                probe_falls=sum(row["done"] for row in probe_rows),
            ).items():
                _same(summary["postfit"][key], value, "secondary probe duration/" + key)
            _same(
                summary["fitting"],
                dict(train_mse=fit["train_error"], held_clean_mse=fit["offline_error"]),
                "fit reconstruction report",
                1e-10,
            )
            walking_panels = [
                sum(row["walking"] for row in evaluation_rows[start : start + 16])
                for start in (0, 16)
            ]
            records.append(
                dict(
                    seed=seed,
                    condition=condition,
                    case_id=case["case_id"],
                    walking=sum(walking_panels),
                    walking_panels=walking_panels,
                    two_panel_competence=min(walking_panels) >= 15,
                    full_survival=sum(row["survived"] for row in evaluation_rows),
                    actual_controls=sum(row["controls"] for row in evaluation_rows),
                    complete_cost=math.fsum(row["cost"] for row in evaluation_rows),
                    native_cost=math.fsum(row["native_cost"] for row in evaluation_rows),
                    absorbing_cost=math.fsum(row["absorbing_cost"] for row in evaluation_rows),
                    diagnostic_error=trace["diagnostic_error"],
                    diagnostic_short_error=trace["diagnostic_short_error"],
                    offline_error=fit["offline_error"],
                    mean_probe_controls=float(np.mean([row["controls"] for row in probe_rows])),
                    probe_falls=sum(row["done"] for row in probe_rows),
                )
            )
            traces.append(trace)
            learning_checks.append(dict(case_id=case["case_id"], **fit))
        if reference_passes:
            clean_counts = [row["walking"] for row in records if row["condition"] == "clean"]
            clean_passes = (
                len(clean_counts) == 6 and sum(clean_counts) >= 173 and min(clean_counts) >= 24
            )
            _same(
                worker["prerequisites"]["clean"],
                dict(walking=clean_counts, qualified=clean_passes),
                "clean-child qualification",
                0,
            )
            _need(
                clean_passes == (worker["status"] != "STOPPED_CLEAN_POSITIVE_CONTROL"),
                "J2 clean qualification stopping rule differs",
            )
        census = primitive_oracle(traces, len(roster), parent_count)
        census = {key: value for key, value in census.items() if value}
        _same(work["requested"], census, "requested primitive census", 0)
        _same(work["completed"], census, "completed primitive census", 0)
        _need(
            all(value <= config["limits"][key] for key, value in census.items()),
            "J2 primitive work ceiling exceeded",
        )
        reader.verify_unchanged()
        reader.check()
        primary = (
            primary_oracle(records, seeds, conditions)
            if worker["status"] == "COMPLETED_FIXED_J2_ROSTER"
            else None
        )
        result.update(
            passed=True,
            worker_status=worker["status"],
            primary_comparison_executed=primary is not None,
            primary=primary,
            records=records,
            learning_checks=learning_checks,
            reference_qualification_walking=panels,
            census=census,
            worker_work_verified=True,
            observed_native_episodes=sum(len(trace["episodes"]) for trace in traces),
            scope="independent saved measurements and declared stopping result; no gradients or physics repeated",
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    finally:
        result.update(
            read_bytes=reader.read_bytes,
            decoded_array_bytes=reader.decoded_array_bytes,
            logical_read_bytes=reader.read_bytes + reader.decoded_array_bytes,
            input_files=reader.files,
            wall_seconds=time.monotonic() - reader.started,
            new_neural_rows=actors.completed if actors else 0,
            requested_neural_rows=actors.requested if actors else 0,
            unknown_neural_rows=actors.unknown if actors else 0,
            functional_network_calls=actors.calls if actors else 0,
            checkpoint_loads=actors.loads if actors else 0,
        )
        payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        _need(len(payload) <= max_output, "J2 audit output ceiling exhausted")
        (out / "audit_result.json").write_bytes(payload)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_run(args.run, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
