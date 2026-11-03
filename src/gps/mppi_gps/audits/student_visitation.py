"""Independent V-only evidence audit; no simulator, worker or fitting imports.

The old independent trace decoder verifies the shared binary evidence contract.
This module derives the visitation outcomes, restored-source comparisons and
reference-usefulness decision itself. Explicit optional prediction verification
replays recorded batch-one actor calls; it never performs native execution.
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
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .student_consequence import TRACE_KEYS, audit_saved_stream, verify_trace_arrays
from .teacher_proposal import AuditError, AuditJournal, optimizer_fingerprint

GIB = 1024**3
MAX_READ = 16 * GIB
MAX_OUTPUT = 2 * GIB
MAX_RSS = 4 * GIB
MAX_WALL = 3600
MAX_FORWARDS = 224000
BOUNDARIES = ("observations", "states", "warmstarts", "integrations")
SCIENCE_ZERO = {
    "native_controls": 0,
    "physics_substeps": 0,
    "backward_rows": 0,
    "optimizer_steps": 0,
    "resets": 0,
    "restores": 0,
    "mppi_queries": 0,
}


def _need(value: Any, message: str) -> None:
    if not value:
        raise AuditError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(payload: bytes | str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "Duplicate visitation JSON key")
            result[key] = value
        return result

    return json.loads(
        payload,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(AuditError("Nonfinite visitation JSON")),
    )


def _rss() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _same_bytes(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def trace_outcome(arrays: Mapping[str, np.ndarray], *, horizon: int = 1000) -> dict[str, Any]:
    """Derive complete-window walking and terminal-once/tail costs from rows.

    Prefix MAE describes only returned controls. It is never extrapolated into
    missing post-fall observations or used to label an incomplete walk.
    """
    n = len(arrays["actions"])
    velocity = np.asarray(arrays["x_velocities"], dtype=np.float64)
    done = np.asarray(arrays["done"])
    native = np.asarray(arrays["native_costs"], dtype=np.float64)
    terminals = np.asarray(arrays["fall_costs"], dtype=np.float64)
    _need(0 <= n <= horizon, "Visitation returned prefix exceeds horizon")
    _need(
        velocity.shape == native.shape == terminals.shape == done.shape == (n,),
        "Visitation outcome row census differs",
    )
    _need(
        np.isfinite(velocity).all() and np.isfinite(native).all() and np.isfinite(terminals).all(),
        "Nonfinite visitation outcome",
    )
    _need(np.all((done == 0) | (done == 1)) and not np.any(done[:-1]), "Post-fall visitation rows")
    fell = bool(n and done[-1])
    _need(np.count_nonzero(terminals) <= 1, "Native fall terminal charged more than once")
    _need(not np.any(terminals[done == 0]), "Native terminal charged on healthy control")
    native_cost = float(np.sum(native, dtype=np.float64))
    terminal = float(np.sum(terminals, dtype=np.float64))
    mae = float(np.mean(np.abs(velocity - 1.5))) if n else None
    survived = bool(n == horizon and not fell)
    absorbing = float(5 * (horizon - n)) if fell else 0.0
    return dict(
        controls=n,
        done=fell,
        survived=survived,
        walking=bool(horizon == 1000 and survived and mae is not None and mae <= 0.3),
        mae=mae,
        native_cost=native_cost,
        native_terminal=terminal,
        raw_cost=native_cost - terminal,
        absorbing_cost=absorbing,
        cost=native_cost + absorbing,
    )


def compare_replay(
    actual: Mapping[str, np.ndarray], source: Mapping[str, np.ndarray], control: int
) -> dict[str, float]:
    """Compare a continued parent to its exact saved suffix, including its fall.

    Initial contexts and executed torques require byte identity. Returned
    boundaries, terminal and cost use the frozen rtol=0/atol=1e-10 contract.
    """
    n = len(actual["actions"])
    expected_n = min(1000, len(source["actions"]) - control)
    _need(n == expected_n, "Unchanged-parent replay length differs from preserved suffix")
    differences = {}
    for key in (*BOUNDARIES, "actions", "native_costs", "x_velocities", "fall_costs", "done"):
        left = np.asarray(actual[key])
        right = np.asarray(source[key][control : control + n + int(key in BOUNDARIES)])
        _need(
            left.shape == right.shape and left.dtype == right.dtype,
            "Parent replay dtype/shape " + key,
        )
        if key in BOUNDARIES:
            _need(
                _same_bytes(left[0], right[0]), "Parent initial restore is not byte exact: " + key
            )
        if key == "actions":
            _need(_same_bytes(left, right), "Parent replay action bytes differ")
        _need(np.allclose(left, right, rtol=0, atol=1e-10), "Parent replay mismatch: " + key)
        differences[key] = float(
            np.max(np.abs(left.astype(float) - right.astype(float)), initial=0)
        )
    return differences


def reference_screen(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use only the 72 noninitial anchors and the same two improving streams."""
    roster: dict[tuple[int, str, str, int, str], dict] = {}
    for row in rows:
        if row["kind"] not in {"parent_replay", "reference_takeover"}:
            continue
        ordinal = row["anchor_ordinal"]
        _need(type(ordinal) is int and ordinal in (0, 1, 2, 3), "Unknown anchor ordinal")
        _need((ordinal == 0) == (row["control"] == 0), "Noninitial anchor control is not positive")
        key = row["stream"], row["panel"], row["source_tag"], ordinal, row["kind"]
        _need(key not in roster, "Duplicated visitation anchor/controller")
        roster[key] = row
    _need(len(roster) == 192, "Complete screen requires exactly 96 paired anchors")
    parent_keys = {key[:-1] for key in roster if key[-1] == "parent_replay"}
    reference_keys = {key[:-1] for key in roster if key[-1] == "reference_takeover"}
    _need(
        parent_keys == reference_keys and len(parent_keys) == 96,
        "Reference and parent anchors are not paired",
    )
    panel_rows = {}
    qualified = {}
    nonregression = True
    for panel in ("primary", "confirmation"):
        qualified[panel] = []
        panel_rows[panel] = {}
        for stream in range(3):
            counts = {}
            for kind in ("parent_replay", "reference_takeover"):
                selected = [
                    row
                    for (s, p, _, ordinal, k), row in roster.items()
                    if s == stream and p == panel and ordinal > 0 and k == kind
                ]
                _need(len(selected) == 12, "Noninitial per-stream/panel roster differs")
                _need(
                    len({row["source_tag"] for row in selected}) == 4,
                    "Anchor source roster differs",
                )
                counts[kind] = dict(
                    walking=sum(row["walking"] is True for row in selected),
                    survival=sum(row["survived"] is True for row in selected),
                )
            parent, reference = counts["parent_replay"], counts["reference_takeover"]
            gain = reference["walking"] - parent["walking"]
            survival_gain = reference["survival"] - parent["survival"]
            nonregression = nonregression and gain >= 0 and survival_gain >= 0
            if gain >= 2:
                qualified[panel].append(stream)
            panel_rows[panel][str(stream)] = dict(
                parent=parent, reference=reference, walking_gain=gain, survival_gain=survival_gain
            )
    common = sorted(set(qualified["primary"]) & set(qualified["confirmation"]))
    return dict(
        passed=bool(nonregression and len(common) >= 2),
        primary_noninitial_anchors=72,
        secondary_initial_anchors=24,
        no_walking_or_survival_regression=bool(nonregression),
        qualifying_streams=qualified,
        same_qualifying_streams=common,
        panels=panel_rows,
        automatic_learning_authorized=False,
    )


def positive_control_screen(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["kind"] == "reference_reset"]
    _need(len(selected) == 32, "Reference positive controls require 32 complete episodes")
    _need(len({row["seed"] for row in selected}) == 32, "Reference positive-control seed repeated")
    panels = {}
    for panel in ("primary", "confirmation"):
        group = [row for row in selected if row["panel"] == panel]
        _need(len(group) == 16, "Reference positive-control panel differs")
        panels[panel] = dict(
            episodes=16,
            walking=sum(row["walking"] is True for row in group),
            survival=sum(row["survived"] is True for row in group),
        )
    return dict(passed=all(v["walking"] >= 15 for v in panels.values()), panels=panels)


def verify_outcome_roster(rows: list[dict], config: dict, catalog: dict) -> None:
    """Bind controller, source, ordinal, panel and seed before prediction replay."""
    anchors = catalog["anchors"]
    _need(
        anchors == config["validity"]["anchors"] and len(anchors) == 96,
        "Frozen V anchor roster differs",
    )
    expected = {
        (
            a["stream"],
            a["panel"],
            a["source_tag"],
            a["source_case"],
            a["anchor_ordinal"],
            a["control"],
        )
        for a in anchors
    }
    _need(len(expected) == 96, "Frozen V anchor identities are not unique")
    for kind in ("parent_replay", "reference_takeover"):
        selected = [r for r in rows if r["kind"] == kind]
        actual = {
            (
                r["stream"],
                r["panel"],
                r["source_tag"],
                r["source_case"],
                r["anchor_ordinal"],
                r["control"],
            )
            for r in selected
        }
        _need(
            len(actual) == len(selected) and actual <= expected,
            "V outcome is not an exact frozen anchor/controller",
        )
        for row in selected:
            actor = f"s{row['stream']}_parent" if kind == "parent_replay" else "reference"
            _need(
                row["actor"] == actor and row["seed"] is None, "V restored controller/seed differs"
            )
    seeds = config["validity"]["positive_control_reference_reset_seeds"]
    _need(len(seeds) == len(set(seeds)) == 32, "V positive-control seeds differ")
    for row in rows:
        if row["kind"] == "reference_reset":
            _need(
                row["seed"] in seeds and row["actor"] == "reference",
                "Unknown reference positive-control seed/controller",
            )
            index = seeds.index(row["seed"])
            _need(
                row["panel"] == ("primary" if index < 16 else "confirmation")
                and row["source_case"] == index % 16,
                "Positive-control seed/panel pairing changed",
            )


class Reader:
    """Stable bounded file reads, shared with selected raw-source verification."""

    def __init__(self, *, max_bytes: int = MAX_READ, seconds: float = MAX_WALL):
        self.started = time.monotonic()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu_started = usage.ru_utime + usage.ru_stime
        self.deadline = self.started + seconds
        self.max_bytes = max_bytes
        self.read_bytes = 0
        self.files: dict[str, dict] = {}
        self.identities: dict[Path, tuple[int, int, int, int]] = {}
        self.external_read_bytes = 0
        self.archive = None
        self.selected_source_arrays = 0

    def total_reads(self) -> int:
        external = self.archive.read_bytes if self.archive is not None else self.external_read_bytes
        return self.read_bytes + external

    def check(self) -> None:
        _need(time.monotonic() <= self.deadline, "Visitation audit wall budget exhausted")
        _need(_rss() <= MAX_RSS, "Visitation audit RSS budget exhausted")
        _need(self.total_reads() <= self.max_bytes, "Audit read budget exhausted")

    def path(self, root: Path, relative: str) -> Path:
        _need(
            isinstance(relative, str)
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "Escaped visitation input",
        )
        root = root.resolve()
        candidate = root / relative
        for part in candidate.relative_to(root).parents:
            _need(not (root / part).is_symlink(), "Symlinked visitation input parent")
        _need(not candidate.is_symlink(), "Symlinked visitation input")
        path = candidate.resolve()
        _need(path.is_relative_to(root) and path.is_file(), "Missing visitation evidence")
        value = path.stat()
        identity = value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
        _need(self.identities.setdefault(path, identity) == identity, "Visitation evidence changed")
        return path

    def read(
        self, root: Path, descriptor: Mapping[str, Any] | str, *, maximum: int = 32 * 1024**2
    ) -> bytes:
        self.check()
        binding = {"file": descriptor} if isinstance(descriptor, str) else descriptor
        path = self.path(root, binding["file"])
        size = path.stat().st_size
        _need(
            size <= maximum and self.total_reads() + size <= self.max_bytes,
            "Visitation file/read reservation exhausted",
        )
        _need(
            "bytes" not in binding or size == binding["bytes"],
            "Visitation file size binding differs",
        )
        digest = hashlib.sha256()
        data = bytearray()
        with path.open("rb", buffering=0) as stream:
            remaining = size
            while remaining:
                self.check()
                part = stream.read(min(1024**2, remaining))
                self.read_bytes += len(part)
                _need(bool(part), "Truncated visitation evidence")
                remaining -= len(part)
                data.extend(part)
                digest.update(part)
        self.path(root, binding["file"])
        sha = digest.hexdigest()
        _need(
            "sha256" not in binding or sha == binding["sha256"],
            "Visitation file hash binding differs",
        )
        self.files[str(path)] = dict(bytes=size, sha256=sha)
        return bytes(data)

    def json(self, root: Path, descriptor: Mapping[str, Any] | str) -> Any:
        return _json(self.read(root, descriptor))

    def npz(self, root: Path, descriptor: Mapping[str, Any]) -> dict[str, np.ndarray]:
        payload = self.read(root, descriptor, maximum=128 * 1024**2)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            _need(
                len(members) <= 16 and sum(v.file_size for v in members) <= 128 * 1024**2,
                "Expanded NPZ exceeds audit reservation",
            )
            _need(len({v.filename for v in members}) == len(members), "Repeated NPZ array name")
            for member in members:
                _need(
                    member.filename.endswith(".npy") and "/" not in member.filename,
                    "Unexpected NPZ member",
                )
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    _need(version in {(1, 0), (2, 0)}, "Unsupported prepared NPY header version")
                    read_header = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    shape, _, dtype = read_header(stream, max_header_size=4096)
                    _need(
                        not dtype.hasobject
                        and dtype.fields is None
                        and len(shape) <= 4
                        and math.prod(shape) * dtype.itemsize == member.file_size - stream.tell(),
                        "NPZ declared array exceeds its bounded member",
                    )
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            result = {key: data[key].copy() for key in data.files}
        _need(
            sum(a.nbytes for a in result.values()) <= 128 * 1024**2,
            "Expanded NPZ exceeds audit reservation",
        )
        _need(all(not a.dtype.hasobject for a in result.values()), "Object visitation evidence")
        return result

    def unchanged(self) -> None:
        for path, original in self.identities.items():
            value = path.stat()
            _need(
                (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns) == original,
                "Visitation input changed after audit",
            )

    def selected_array(self, root: Path, relative: str, binding: Mapping[str, Any]) -> np.ndarray:
        self.check()
        path = self.path(root, relative)
        offset, length = binding["offset"], binding["byte_length"]
        _need(
            type(offset) is int
            and type(length) is int
            and offset >= 0
            and 0 <= length <= 16 * 1024**2,
            "Invalid selected source byte range",
        )
        _need(
            offset + length <= path.stat().st_size
            and self.total_reads() + length <= self.max_bytes,
            "Selected source read exceeds reservation",
        )
        with path.open("rb", buffering=0) as stream:
            stream.seek(offset)
            payload = stream.read(length)
        self.read_bytes += len(payload)
        _need(
            len(payload) == length and hashlib.sha256(payload).hexdigest() == binding["sha256"],
            "Selected original C1 array bytes differ",
        )
        dtype = np.dtype(binding["dtype"])
        _need(
            not dtype.hasobject
            and dtype.fields is None
            and math.prod(binding["shape"]) * dtype.itemsize == length,
            "Invalid selected source shape/dtype",
        )
        self.path(root, relative)
        self.selected_source_arrays += 1
        return np.frombuffer(payload, dtype=dtype).reshape(binding["shape"]).copy()

    def compact_ledger(self, root: Path) -> dict[str, Any]:
        """Check actual encoded line lengths, including whitespace and newline."""
        path = self.path(root, "ledger.jsonl")
        size = path.stat().st_size
        _need(
            self.total_reads() + size <= self.max_bytes, "V raw ledger cannot fit read reservation"
        )
        digest = hashlib.sha256()
        count = 0
        with path.open("rb", buffering=0) as stream:
            while stream.tell() < size:
                self.check()
                raw = stream.readline(min(2049, size - stream.tell()))
                self.read_bytes += len(raw)
                _need(raw.endswith(b"\n"), "V raw ledger has an oversized/incomplete line")
                row = _json(raw)
                maximum = 2048 if row["event"] in {"runtime_open", "runtime_close"} else 512
                _need(
                    len(raw) <= maximum,
                    "V encoded lifecycle/frequent-event line exceeds frozen bound",
                )
                count += 1
                _need(count <= 1_200_000, "V raw journal exceeds record census")
                digest.update(raw)
        self.path(root, "ledger.jsonl")
        return dict(
            lines=count,
            bytes=size,
            sha256=digest.hexdigest(),
            frequent_line_bytes=512,
            lifecycle_line_bytes=2048,
        )


class ActorVerifier:
    """Independent functional replicas of the two frozen actor interfaces."""

    def __init__(
        self,
        reader: Reader,
        root: Path,
        registry: Mapping[str, dict],
        journal: AuditJournal,
        catalog: dict | None = None,
    ):
        _need(len(registry) == 4, "V audit requires precisely three parents and one reference")
        self.reader, self.root, self.registry, self.journal = reader, root, dict(registry), journal
        self.actors = {}
        self.requested = self.completed = self.calls = self.loads = self.unknown = 0
        self.catalog = catalog

    def load_all(self) -> None:
        import torch

        torch.set_num_threads(1)
        for alias, record in self.registry.items():
            self.reader.check()
            _need(self.loads < 4, "V checkpoint load budget exhausted")
            payload_bytes = self.reader.read(self.root, record)
            self.journal.record(dict(event="checkpoint_requested", actor=alias))
            self.loads += 1
            payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
            reference = alias == "reference"
            state = payload["policy"]["actor"] if reference else payload["state_dict"]
            schema = (
                "walker-legacy-sac-deterministic-actor-v1" if reference else "walker-tanh-policy-v1"
            )
            digest = hashlib.sha256(schema.encode())
            expected = (
                {
                    "fc1.weight": (256, 17),
                    "fc1.bias": (256,),
                    "fc2.weight": (256, 256),
                    "fc2.bias": (256,),
                    "fc_mean.weight": (6, 256),
                    "fc_mean.bias": (6,),
                    "fc_logstd.weight": (6, 256),
                    "fc_logstd.bias": (6,),
                    "action_low": (6,),
                    "action_high": (6,),
                    "action_scale": (6,),
                    "action_bias": (6,),
                    "obs_mean": (17,),
                    "obs_std": (17,),
                }
                if reference
                else {
                    "observation_mean": (17,),
                    "observation_std": (17,),
                    "network.0.weight": (128, 17),
                    "network.0.bias": (128,),
                    "network.2.weight": (128, 128),
                    "network.2.bias": (128,),
                    "network.4.weight": (6, 128),
                    "network.4.bias": (6,),
                }
            )
            _need(set(state) == set(expected), "V actor tensor roster differs")
            for name, tensor in sorted(state.items()):
                _need(
                    tuple(tensor.shape) == expected[name]
                    and tensor.dtype == torch.float32
                    and bool(torch.isfinite(tensor).all()),
                    "V actor tensor interface differs",
                )
                array = tensor.detach().cpu().contiguous().numpy()
                digest.update(_canonical([name, list(array.shape), str(array.dtype)]))
                digest.update(array.tobytes())
            _need(digest.hexdigest() == record["actor_hash"], "V actor named-tensor hash differs")
            if reference:
                for name, value in (
                    ("action_low", -1),
                    ("action_high", 1),
                    ("action_scale", 1),
                    ("action_bias", 0),
                    ("obs_mean", 0),
                    ("obs_std", 1),
                ):
                    _need(bool((state[name] == value).all()), "Frozen reference interface changed")
            else:
                _need(
                    payload["policy_hash"] == record["actor_hash"]
                    and bool((state["observation_std"] > 0).all()),
                    "Parent hash/normalization differs",
                )
                if self.catalog is not None:
                    descriptor = self.catalog["parents"][alias[1]]
                    continuation = descriptor["continuation"]
                    incoming = payload["optimizer_state"]
                    _need(
                        optimizer_fingerprint(incoming)
                        == continuation["optimizer_sha256"]
                        == record["optimizer_sha256"],
                        "Complete incoming Adam/RNG differs",
                    )
                    _need(
                        incoming["updates"] == incoming["fit_updates"] == 2000,
                        "Incoming parent update counters changed",
                    )
                    _need(
                        optimizer_fingerprint(payload["private_rng_state"])
                        == continuation["private_rng_sha256"]
                        == record["private_rng_sha256"],
                        "Incoming private RNG differs",
                    )
                    _need(
                        optimizer_fingerprint(incoming["rng"])
                        == continuation["pcg_sha256"]
                        == record["pcg_sha256"],
                        "Incoming minibatch PCG state differs",
                    )
                    sidecar = self.reader.read(
                        self.root / "prepared", descriptor["global_cpu_sidecar"]
                    )
                    global_rng = np.load(io.BytesIO(sidecar), allow_pickle=False)
                    _need(
                        optimizer_fingerprint(global_rng) == continuation["global_rng_sha256"],
                        "Incoming global CPU RNG sidecar differs",
                    )
                    normalisation = self.reader.npz(
                        self.root / "prepared", self.catalog["normalization"]
                    )
                    for key in ("mean", "std"):
                        _need(
                            _same_bytes(
                                state["observation_" + key].numpy(),
                                np.asarray(normalisation[key], dtype=np.float32),
                            ),
                            "Frozen parent normalization differs",
                        )
            self.actors[alias] = state
            self.journal.record(dict(event="checkpoint_complete", actor=alias))

    def verify(self, alias: str, observation: np.ndarray, action: np.ndarray) -> None:
        self.reader.check()
        _need(
            alias in self.actors and self.requested < MAX_FORWARDS and self.calls < MAX_FORWARDS,
            "V audit neural reservation exhausted",
        )
        _need(
            observation.shape == (17,) and np.isfinite(observation).all(),
            "V prediction input shape differs",
        )
        self.journal.record(
            dict(
                event="prediction_requested",
                actor=alias,
                call=self.calls,
                rows=1,
                observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest(),
                expected_action_sha256=hashlib.sha256(action.tobytes()).hexdigest(),
            )
        )
        self.requested += 1
        self.calls += 1
        try:
            import torch
            import torch.nn.functional as f

            state = self.actors[alias]
            with torch.no_grad():
                value = torch.as_tensor(observation.astype(np.float32, copy=True))
                if alias == "reference":
                    value = value.unsqueeze(0)
                    value = (value - state["obs_mean"]) / state["obs_std"]
                    value = f.relu(f.linear(value, state["fc1.weight"], state["fc1.bias"]))
                    value = f.relu(f.linear(value, state["fc2.weight"], state["fc2.bias"]))
                    mean = f.linear(value, state["fc_mean.weight"], state["fc_mean.bias"])
                    # The deployed frozen reference also evaluates this head.
                    _ = -5.0 + 3.5 * (
                        torch.tanh(
                            f.linear(value, state["fc_logstd.weight"], state["fc_logstd.bias"])
                        )
                        + 1.0
                    )
                    value = (
                        torch.tanh(mean) * state["action_scale"] + state["action_bias"]
                    ).squeeze(0)
                else:
                    value = (value - state["observation_mean"]) / state["observation_std"]
                    for layer in (0, 2, 4):
                        value = torch.tanh(
                            f.linear(
                                value,
                                state[f"network.{layer}.weight"],
                                state[f"network.{layer}.bias"],
                            )
                        )
                actual = value.cpu().numpy().astype(np.float64)
        except BaseException:
            self.unknown += 1
            self.journal.record(
                dict(event="prediction_error", actor=alias, call=self.calls - 1, rows=1)
            )
            raise
        self.completed += 1
        self.journal.record(
            dict(
                event="prediction_complete",
                actor=alias,
                call=self.calls - 1,
                rows=1,
                action_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            )
        )
        _need(
            _same_bytes(actual, action),
            "Saved batch-one policy action differs from independent replay",
        )


class WorkChecks:
    """Join every saved request/return without trusting worker totals."""

    def __init__(self):
        self.requested: Counter[str] = Counter()
        self.completed: Counter[str] = Counter()
        self.unknown: Counter[str] = Counter()
        self.unpersisted: Counter[str] = Counter()
        self.pending: dict[int, tuple[str, int]] = {}
        self.serial = 0
        self.opened = self.closed = 0
        self.phase: str | None = None
        self.trace_order: list[tuple[str, int]] = []
        self.event_count = 0
        self.environment_pending = False

    def event(self, row: dict[str, Any]) -> None:
        self.event_count += 1
        event = row["event"]
        _need(
            event not in {"query_requested", "query_complete", "query_error"},
            "V must not perform planner/model queries",
        )
        _need(self.event_count <= 1_200_000, "V ledger row ceiling exhausted")
        _need(
            len(_canonical(row)) + 1
            <= (2048 if event in {"runtime_open", "runtime_close"} else 512),
            "V lifecycle/frequent-event line bound differs",
        )
        if event == "runtime_open":
            self.opened += 1
        elif event == "runtime_close":
            self.closed += 1
        elif event == "work_requested":
            operation, category, amount = row["operation"], row["category"], row["amount"]
            _need(
                category
                in {
                    "neural_rows",
                    "checkpoint_loads",
                    "policy_constructions",
                    "restores",
                    "resets",
                    "snapshot_reads",
                    "external_mj_forward_calls",
                    "backward_rows",
                    "optimizer_steps",
                },
                "Undeclared V measured work category",
            )
            _need(
                type(operation) is int and operation == self.serial + 1,
                "V work operation order differs",
            )
            _need(type(amount) is int and amount >= 0, "V work amount differs")
            _need(
                category not in {"backward_rows", "optimizer_steps", "mppi_queries"} or amount == 0,
                "Forbidden V fitting/planning work",
            )
            _need(
                category != "neural_rows" or amount == 1, "V execution must retain batch-one calls"
            )
            self.serial = operation
            self.pending[operation] = category, amount
            self.requested[category] += amount
            if category == "neural_rows":
                self.requested["prediction_calls"] += 1
        elif event in {"work_complete", "work_error"}:
            operation = row["operation"]
            _need(operation in self.pending, "V work return lacks request")
            category, amount = self.pending.pop(operation)
            _need(
                (row["category"], row["amount"]) == (category, amount),
                "V work return binding differs",
            )
            counts = (
                self.completed
                if event == "work_complete" or row.get("returned") is True
                else self.unknown
            )
            counts[category] += amount
            if category == "neural_rows":
                counts["prediction_calls"] += 1
            if event == "work_error" and row.get("returned") is True:
                self.unpersisted[category] += amount
        elif event == "trace_begin":
            self.trace_order.append((row["alias"], self.requested["neural_rows"]))
        elif event == "environment_requested":
            _need(
                not self.environment_pending
                and self.requested["environment_constructions"] == 0
                and row["mjdata_objects"] == 17,
                "V environment construction roster differs",
            )
            self.environment_pending = True
            self.requested["environment_constructions"] += 1
        elif event == "environment_complete":
            _need(
                self.environment_pending
                and row["integration_size"] == 91
                and row["mjdata_objects"] == 17,
                "V environment return/interface differs",
            )
            self.environment_pending = False
            self.completed["environment_constructions"] += 1

    def verify_complete(self) -> None:
        _need(self.opened == self.closed == 1, "V runtime lifecycle is incomplete")
        _need(
            not self.pending
            and not self.environment_pending
            and not any(self.unknown.values())
            and not any(self.unpersisted.values()),
            "V journal retains uncertain measured work",
        )


def _source_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {key: descriptor[key] for key in ("file", "bytes", "sha256") if key in descriptor}


def _normalise_row(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row["kind"] = row.get("stage", row.get("kind"))
    row["control"] = row.get("t", row.get("control", 0))
    _need(
        row["kind"] in {"parent_replay", "reference_reset", "reference_takeover"},
        "Unknown V outcome kind",
    )
    return row


def _check_summary(saved: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    for key, value in actual.items():
        _need(key in saved, "Missing V saved scalar: " + key)
        target = saved[key]
        if type(value) is float:
            _need(
                type(target) in (float, int)
                and math.isclose(value, target, rel_tol=0, abs_tol=1e-10),
                "V saved scalar differs: " + key,
            )
        else:
            _need(
                value == target and (type(value) is not bool or type(target) is bool),
                "V saved scalar differs: " + key,
            )


def reconcile_costs(worker: dict, rows: list[dict], stream: dict, work: WorkChecks) -> dict:
    """Count native work from returned prefixes and helper calls from their owners."""
    controls = sum(row["controls"] for row in rows)
    restores = sum(row["kind"] != "reference_reset" for row in rows)
    resets = sum(row["kind"] == "reference_reset" for row in rows)
    expected = dict(
        native_controls=controls,
        physics_substeps=4 * controls,
        neural_rows=controls,
        prediction_calls=controls,
        restores=restores,
        resets=resets,
        external_mj_forward_calls=restores + 2 * resets,
        snapshot_reads=2 * controls + restores + resets,
        checkpoint_loads=4,
        policy_constructions=4,
        environment_constructions=1,
        backward_rows=0,
        optimizer_steps=0,
    )
    costs = worker["costs"]
    for counter in ("requested", "completed"):
        for key, value in expected.items():
            _need(
                costs[counter].get(key, 0) == value,
                "V complete cost census differs: " + counter + "/" + key,
            )
    _need(
        not any(costs["unknown_partial"].values())
        and not any(costs["returned_unpersisted"].values()),
        "V worker cost record contains unknown/unpersisted work",
    )
    for key in (
        "neural_rows",
        "prediction_calls",
        "restores",
        "resets",
        "checkpoint_loads",
        "policy_constructions",
        "environment_constructions",
    ):
        _need(
            work.requested[key] == work.completed[key] == expected[key],
            "V measured journal cost differs: " + key,
        )
    _need(
        stream["saved_controls"] == controls and stream["unknown_partial_controls"] == 0,
        "Native stream cost differs",
    )
    return dict(
        independently_derived=expected,
        worker=costs,
        physics_repeated_by_audit=0,
        fitting_repeated_by_audit=0,
    )


def _write_json(out: Path, name: str, value: Any) -> dict[str, Any]:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    _need(len(payload) <= 8 * 1024**2, "V compact audit output exceeds bound")
    used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    _need(used + len(payload) <= MAX_OUTPUT, "V audit output ceiling exhausted")
    with (out / name).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(file=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def _prepared_inputs(
    reader: Reader, run: Path, catalog: dict, config: dict
) -> tuple[dict[str, dict], dict]:
    prepared = run / "prepared"
    manifest = reader.json(prepared, "prepared_input_manifest.json")
    files = manifest.get("files", manifest.get("output_files"))
    _need(isinstance(files, (dict, list)), "Missing prepared file census")
    bindings = list(files.values()) if isinstance(files, dict) else files
    for record in bindings:
        reader.read(prepared, record, maximum=128 * 1024**2)
    static = reader.json(prepared, "static_receipt.json")
    _need(static.get("passed") is True, "Prepared static-context gate did not pass")
    settings = config["input_preparation"]
    review = reader.json(
        prepared,
        dict(
            file="metadata/review_recipe.json",
            bytes=settings["review_recipe"]["bytes"],
            sha256=settings["review_recipe"]["sha256"],
        ),
    )
    bound_metadata = {}
    for name in ("a1_verified_files", "recovery_admission"):
        binding = settings[name]
        bound_metadata[name] = reader.json(
            prepared,
            dict(
                file="metadata/" + name + ".json", bytes=binding["bytes"], sha256=binding["sha256"]
            ),
        )
    admission = bound_metadata["recovery_admission"]
    _need(
        admission["audit_recovery_resolved"] is True
        and admission["retained_A1_integrity_certification"] is True
        and admission["retained_original_numerical_certification"] is True,
        "Original C1 compensating admission is not established",
    )
    _need(
        catalog["anchors"] == config["validity"]["anchors"] == review["validity"]["anchors"],
        "Exact frozen V anchors changed",
    )
    for stream in range(3):
        expected_parent = review["parents"][str(stream)]
        frozen_parent = config["parents"][str(stream)]
        actual_parent = catalog["parents"][str(stream)]
        for key in ("sha256", "actor_hash", "continuation"):
            _need(
                actual_parent[key] == frozen_parent[key] == expected_parent[key],
                "Complete frozen parent/source binding differs: " + key,
            )
        _need(
            actual_parent["global_cpu_sidecar"]["sha256"]
            == expected_parent["global_cpu_sidecar"]["sha256"],
            "Frozen global RNG sidecar binding differs",
        )
    _need(
        catalog["reference"]["sha256"]
        == config["reference"]["checkpoint_sha256"]
        == review["reference"]["checkpoint_sha256"]
        == settings["reference"]["sha256"],
        "Fixed reference checkpoint binding differs",
    )
    _need(
        catalog["reference"]["actor_hash"]
        == config["reference"]["actor_hash"]
        == review["reference"]["actor_hash"],
        "Fixed reference actor binding differs",
    )
    _need(
        catalog["normalization"]["sha256"]
        == config["normalization"]["sha256"]
        == review["normalization"]["sha256"],
        "Frozen W1 normalization binding differs",
    )
    _need(
        catalog["xml"]["sha256"] == config["xml_sha256"] == settings["xml"]["sha256"],
        "Frozen XML binding differs",
    )
    old_root = Path(catalog["source"]["c1_run"]).resolve()
    _need(old_root == Path(settings["c1_run"]).resolve(), "Original C1 source location changed")
    certified = bound_metadata["a1_verified_files"]["files"]
    _need(
        catalog["source"]["raw"] == certified["run/worker/evidence/arrays.bin"]
        and catalog["source"]["index"] == certified["run/worker/evidence/arrays.index.jsonl"],
        "Retained original stream certification differs",
    )
    raw = reader.path(old_root, "worker/evidence/arrays.bin")
    _need(
        raw.stat().st_size == certified["run/worker/evidence/arrays.bin"]["bytes"],
        "Original C1 raw stream size changed",
    )
    reader.read(
        old_root,
        dict(
            file="worker/evidence/arrays.index.jsonl",
            **certified["run/worker/evidence/arrays.index.jsonl"],
        ),
        maximum=128 * 1024**2,
    )
    evaluations = reader.read(
        prepared,
        dict(
            file="metadata/c1/worker/evaluations.jsonl",
            **{key: review["input_bindings"]["c1_evaluations"][key] for key in ("bytes", "sha256")},
        ),
    )
    original_rows = {_json(line)["tag"]: _json(line) for line in evaluations.splitlines()}
    initial_compared = 0
    for seed, descriptors in catalog["parent_initial_contexts"].items():
        _need(set(descriptors) == {"0", "1", "2"}, "Missing parent initial context")
        canonical = reader.npz(prepared, catalog["canonical_contexts"][seed])
        _need(
            set(canonical)
            == {"state", "warmstart", "integration", "observation", "spec", "dimensions"},
            "Canonical context keys changed",
        )
        for stream, descriptor in descriptors.items():
            context = reader.npz(prepared, descriptor)
            _need(set(context) == set(canonical), "Initial context keys changed")
            row = original_rows[descriptor["source_tag"]]
            _need(
                str(row["stream"]) == stream and str(row["seed"]) == seed,
                "Initial context row/seed binding changed",
            )
            for key, source_key in (
                ("state", "states"),
                ("warmstart", "warmstarts"),
                ("integration", "integrations"),
                ("observation", "observations"),
            ):
                binding = dict(row["trace"]["arrays"][source_key])
                # Read the complete bound array so its original C-byte hash is
                # independently checked before selecting the initial boundary.
                original = reader.selected_array(old_root, "worker/evidence/arrays.bin", binding)
                _need(
                    _same_bytes(context[key], original[0]),
                    "Prepared initial context differs from original C1 bytes",
                )
                _need(
                    context[key].shape == canonical[key].shape
                    and np.allclose(context[key], canonical[key], rtol=0, atol=1e-10),
                    "Static all-parent reset contexts differ",
                )
            for key in ("spec", "dimensions"):
                _need(
                    _same_bytes(context[key], canonical[key]), "Static integration layout differs"
                )
            _need(
                np.array_equal(context["spec"], np.array(16383, dtype="<i8"))
                and np.array_equal(context["dimensions"], np.array([9, 9, 6], dtype="<i8")),
                "Original integration specification changed",
            )
            initial_compared += 1
    _need(
        initial_compared == 96 and len(catalog["canonical_contexts"]) == 32,
        "Static initial-context census differs",
    )
    task_document = reader.json(prepared, catalog["task"])
    task = task_document.get("task", task_document)
    _need(
        task == config["task"] and task_document["xml_sha256"] == config["xml_sha256"],
        "Frozen native task binding differs",
    )
    source = {}
    for tag, record in catalog["episodes"].items():
        arrays = reader.npz(prepared, record)
        _need(
            len(arrays["actions"]) == record["controls"], "Prepared source control census differs"
        )
        _need(
            record["source_record"] == original_rows[tag]
            and record["source_trace"] == original_rows[tag]["trace"],
            "Prepared episode/source outcome binding differs",
        )
        for key, descriptor in record.get("array_hashes", {}).items():
            expected = descriptor["sha256"] if isinstance(descriptor, dict) else descriptor
            _need(
                hashlib.sha256(np.asarray(arrays[key]).tobytes()).hexdigest() == expected,
                "Prepared source C-byte binding differs: " + key,
            )
        original = record["source_trace"]["arrays"]
        for key, binding in original.items():
            _need(key in arrays, "Prepared source omits an original trace array")
            value = np.asarray(arrays[key])
            _need(
                list(value.shape) == binding["shape"]
                and value.dtype == np.dtype(binding["dtype"])
                and hashlib.sha256(value.tobytes()).hexdigest() == binding["sha256"],
                "Prepared source differs from original C1 trace binding",
            )
            original_array = reader.selected_array(old_root, "worker/evidence/arrays.bin", binding)
            _need(
                _same_bytes(value, original_array),
                "Selected source array differs from original C1 payload",
            )
        independently_checked = verify_trace_arrays(
            {key: arrays[key] for key in TRACE_KEYS}, task=task
        )
        _need(
            independently_checked["normally_closed"]
            and independently_checked["controls"] == record["controls"]
            and independently_checked["unknown_partial_controls"] == 0,
            "Original C1 source is not a complete actual trace prefix",
        )
        source[tag] = arrays
    _need(len(source) == 24, "V requires 24 complete preserved source episodes")
    expected_replay_controls = sum(
        len(source[a["source_tag"]]["actions"]) - a["control"] for a in catalog["anchors"]
    )
    _need(
        expected_replay_controls
        == config["expected_parent_replay_controls"]
        == review["finite_census"]["validity"][
            "expected_unchanged_parent_replay_controls_from_bound_metadata"
        ],
        "Frozen source-derived parent replay census differs",
    )
    return source, dict(
        passed=True,
        independently_compared_initial_contexts=initial_compared,
        canonical_contexts=32,
        noninitial_anchors=72,
        source_episodes=24,
        retained_preparation_receipt=static,
        full_original_raw_rehashed=False,
        retained_original_raw_sha256=certified["run/worker/evidence/arrays.bin"]["sha256"],
        selected_original_array_checks=reader.selected_source_arrays,
        current_original_index_sha256=reader.files[
            str(old_root / "worker/evidence/arrays.index.jsonl")
        ]["sha256"],
    )


def audit_run(run: Path, out: Path) -> dict[str, Any]:
    """Audit the frozen V block, preserving negative screens and paid failures.

    Output is create-only and separate from all preserved source directories.
    A passed audit can certify a negative positive-control/usefulness screen;
    it never authorizes conditional learning or changes either scientific gate.
    """
    run, out = Path(run).resolve(), Path(out).resolve()
    _need(not out.is_relative_to(run), "V audit output must be outside preserved run")
    reader = Reader()
    config = reader.json(run, "config.json")
    c1_root = Path(config["input_preparation"]["c1_run"]).resolve()
    _need(not out.is_relative_to(c1_root), "V audit output must be outside original C1 evidence")
    out.mkdir(parents=True, exist_ok=False)
    verifier = None
    work = WorkChecks()
    rows: list[dict] = []
    prefixes: list[dict] = []
    checked_tags: set[str] = set()
    result: dict[str, Any] = dict(
        schema="walker-student-visitation-s1.2-v-audit-v1",
        passed=False,
        full_scientific_audit_passed=False,
        automatic_learning_authorized=False,
        new_scientific_work=dict(SCIENCE_ZERO),
        reference_screen_passed=None,
    )
    journal = None
    try:
        limits = config["limits"]
        _need(
            config["schema"] == "walker-student-visitation-v-execution-v1",
            "Unknown V execution schema",
        )
        _need(
            config["user_authorized"] is True
            and config["authorized_stage"] == "V"
            and config["conditional_learning_authorized"] is False
            and config["automatic_followup"] is False,
            "V-only launch authorization/scope differs",
        )
        _need(
            0 < limits["audit_wall_seconds"] <= MAX_WALL
            and limits["audit_read_bytes"] <= MAX_READ
            and limits["audit_output_bytes"] <= MAX_OUTPUT
            and limits["audit_peak_rss_bytes"] <= MAX_RSS,
            "V audit declared limits exceed allocation",
        )
        _need(
            limits["audit_neural_rows"] == limits["audit_prediction_calls"] == MAX_FORWARDS
            and limits["audit_checkpoint_loads"] == 4,
            "V audit actor work census changed",
        )
        reader.deadline = reader.started + limits["audit_wall_seconds"]
        prelude_reservation = limits["source_and_metadata_read_reservation_bytes"]
        _need(prelude_reservation == 128 * 1024**2, "V audit prelude reservation changed")
        reader.max_bytes = limits["audit_read_bytes"] - prelude_reservation
        result["read_allocation"] = dict(
            core_limit_bytes=reader.max_bytes,
            source_and_metadata_prelude_reserved_bytes=prelude_reservation,
            whole_stage_limit_bytes=limits["audit_read_bytes"],
            core_counts_exclude_coordinator_source_verification=True,
        )
        result["config_sha256"] = reader.files[str(run / "config.json")]["sha256"]
        catalog = reader.json(run / "prepared", "catalog.json")
        sources, static = _prepared_inputs(reader, run, catalog, config)
        result["prepared_manifest_sha256"] = reader.files[
            str(run / "prepared/prepared_input_manifest.json")
        ]["sha256"]
        worker_result = reader.json(run, "worker_result.json")
        result["worker_result_sha256"] = reader.files[str(run / "worker_result.json")]["sha256"]
        _need(
            worker_result["config_sha256"] == result["config_sha256"]
            and worker_result["prepared_manifest_sha256"] == result["prepared_manifest_sha256"],
            "Worker/config/preparation association differs",
        )
        _need(
            worker_result.get("learning_launched") is False
            and worker_result.get("followup_launched") is False,
            "V launched an unauthorized learning/followup block",
        )
        task_doc = reader.json(run / "prepared", catalog["task"])
        task = task_doc.get("task", task_doc)
        result["encoded_ledger"] = reader.compact_ledger(run / "worker/evidence")
        if worker_result["status"] not in {
            "COMPLETED_V_FIXED_ROSTER",
            "STOPPED_REFERENCE_POSITIVE_CONTROL_FAILED",
        }:
            stream = audit_saved_stream(
                run / "worker/evidence",
                out / "incomplete_stream",
                task=task,
                max_read_bytes=reader.max_bytes - reader.read_bytes,
                max_output_bytes=MAX_OUTPUT // 2,
                max_wall_seconds=max(0.001, reader.deadline - time.monotonic()),
            )
            reader.external_read_bytes = stream.get("source_read_bytes", 0)
            result.update(
                scope="saved_structural_prefix_only_after_worker_failure",
                incomplete_stream=stream,
                worker_status=worker_result["status"],
                preserved_worker_costs=worker_result.get("costs"),
                unavailable_outcomes="unexecuted and unresolved observations remain missing",
            )
            raise AuditError(
                "Worker stopped at a prerequisite/infrastructure gate; complete V certification unavailable"
            )
        registry = reader.json(run / "worker", "actor_registry.json")
        _need(
            isinstance(registry, dict)
            and set(registry) == {"s0_parent", "s1_parent", "s2_parent", "reference"},
            "Frozen V actor registry differs",
        )
        for alias, binding in registry.items():
            expected = (
                catalog["reference"] if alias == "reference" else catalog["parents"][alias[1]]
            )
            for key in ("actor_hash", "sha256"):
                _need(binding[key] == expected[key], "V actor/source checkpoint join differs")
        outcome_payload = reader.read(run / "worker", "outcomes.jsonl", maximum=8 * 1024**2)
        result["outcomes_sha256"] = hashlib.sha256(outcome_payload).hexdigest()
        _need(
            not outcome_payload or outcome_payload.endswith(b"\n"), "Incomplete saved V outcome row"
        )
        rows = [_normalise_row(_json(line)) for line in outcome_payload.splitlines() if line]
        _need(
            len(rows) <= 224 and len({row["tag"] for row in rows}) == len(rows),
            "V outcome roster duplicated/oversized",
        )
        verify_outcome_roster(rows, config, catalog)
        by_tag = {row["tag"]: row for row in rows}
        journal = AuditJournal(
            out / "prediction_journal.jsonl", max_bytes=256 * 1024**2, max_line_bytes=512
        )
        verifier = ActorVerifier(reader, run, registry, journal, catalog)
        verifier.load_all()
        replay_differences = {}
        phase_order = {"parent_replay": 0, "reference_reset": 1, "reference_takeover": 2}
        last_phase = -1

        def check_trace(summary, arrays, reference):
            nonlocal last_phase
            reader.check()
            tag = summary["alias"]
            _need(tag in by_tag and tag not in checked_tags, "Trace lacks unique scalar outcome")
            saved = by_tag[tag]
            kind = saved["kind"]
            _need(phase_order[kind] >= last_phase, "V phases executed out of order")
            last_phase = phase_order[kind]
            _need(
                summary["normally_closed"] and summary["unknown_partial_controls"] == 0,
                "V trace is an incomplete/uncertain prefix",
            )
            metadata = summary["metadata"]
            _need(
                metadata["kind"] == kind and metadata["actor"] == saved["actor"],
                "V trace actor/kind binding differs",
            )
            _need(
                metadata["integration_spec"] == 16383 and metadata["dimensions"] == [9, 9, 6],
                "V integration specification changed",
            )
            _need(reference == saved["trace"], "Saved V trace reference differs")
            _need(summary["horizon"] == 1000 and saved["h"] == 1000, "V horizon changed")
            _need(
                saved["actor_hash"] == registry[saved["actor"]]["actor_hash"]
                and saved["checkpoint_sha256"] == registry[saved["actor"]]["sha256"],
                "V row checkpoint ownership differs",
            )
            measured = trace_outcome(arrays)
            _check_summary(saved, measured)
            _need(
                metadata.get("prediction_actor") == registry[saved["actor"]]["sha256"],
                "V durable prediction checkpoint differs",
            )
            if kind != "reference_reset":
                source = sources[saved["source_tag"]]
                control = saved["control"]
                for key in BOUNDARIES:
                    _need(
                        _same_bytes(arrays[key][0], source[key][control]),
                        "V restored full context differs: " + key,
                    )
                if kind == "parent_replay":
                    replay_differences[tag] = compare_replay(arrays, source, control)
                else:
                    _need(
                        sum(row["kind"] == "parent_replay" for row in prefixes) == 96,
                        "Reference takeover preceded complete replay gate",
                    )
                    _need(
                        positive_control_screen(prefixes)["passed"],
                        "Reference takeover followed failed positive control",
                    )
            else:
                _need(
                    sum(row["kind"] == "parent_replay" for row in prefixes) == 96,
                    "Reference reset preceded complete parent replay gate",
                )
                context = reader.npz(
                    run / "prepared", catalog["canonical_contexts"][str(saved["seed"])]
                )
                for key, other in (
                    ("states", "state"),
                    ("warmstarts", "warmstart"),
                    ("integrations", "integration"),
                    ("observations", "observation"),
                ):
                    _need(
                        arrays[key][0].shape == context[other].shape
                        and np.allclose(arrays[key][0], context[other], rtol=0, atol=1e-10),
                        "Reference reset differs from canonical saved context",
                    )
            for observation, action in zip(
                arrays["observations"][:-1], arrays["actions"], strict=True
            ):
                verifier.verify(saved["actor"], observation, action)
            actual = dict(saved, **measured)
            actual.pop("trace", None)
            prefixes.append(actual)
            checked_tags.add(tag)
            with (out / "prefix_curves.jsonl").open("ab") as curve:
                for index in range(measured["controls"]):
                    point = dict(
                        tag=tag,
                        kind=kind,
                        control=index,
                        elapsed_seconds=(index + 1) * 0.008,
                        velocity=float(arrays["x_velocities"][index]),
                        height=float(arrays["states"][index + 1, 2]),
                        pitch=float(arrays["states"][index + 1, 3]),
                        native_cost=float(arrays["native_costs"][index]),
                        terminal_cost=float(arrays["fall_costs"][index]),
                        done=bool(arrays["done"][index]),
                    )
                    payload = _canonical(point) + b"\n"
                    _need(len(payload) <= 512, "V audit prefix curve row exceeds reservation")
                    curve.write(payload)

        stream = audit_saved_stream(
            run / "worker/evidence",
            out / "stream",
            task=task,
            max_read_bytes=reader.max_bytes - reader.read_bytes - 64 * 1024**2,
            max_output_bytes=MAX_OUTPUT // 2,
            max_wall_seconds=max(0.001, reader.deadline - time.monotonic()),
            _trace_check=check_trace,
            _event_check=work.event,
            _archive_open=lambda archive: setattr(reader, "archive", archive),
        )
        reader.external_read_bytes = stream.get("source_read_bytes", 0)
        reader.archive = None
        _need(
            stream["structural_passed"] and stream["complete_journal"],
            "V raw evidence structural audit failed: " + stream.get("error", "incomplete journal"),
        )
        work.verify_complete()
        _need(checked_tags == set(by_tag), "V scalar/raw trace roster differs")
        _need(stream["array_groups"] == len(rows) <= 300, "V has unowned or excess array groups")
        _need(
            stream["unknown_partial_controls"] == 0
            and stream["saved_controls"]
            == stream["requested_controls"]
            == stream["returned_controls"],
            "V native request/return census differs",
        )
        _need(
            verifier.requested
            == verifier.completed
            == verifier.calls
            == stream["saved_controls"]
            == work.completed["neural_rows"]
            == work.completed["prediction_calls"],
            "V audit/worker neural/native census differs",
        )
        _need(len(replay_differences) == 96, "Parent replay gate is incomplete")
        positive = positive_control_screen(prefixes)
        takeover_rows = [row for row in prefixes if row["kind"] == "reference_takeover"]
        if positive["passed"]:
            _need(
                len(takeover_rows) == 96,
                "Positive reference controls require full declared takeover roster",
            )
            screen = reference_screen(prefixes)
            closure = "complete_reference_usefulness_screen"
        else:
            _need(
                not takeover_rows,
                "Reference takeovers were executed after a negative own-reset gate",
            )
            screen = None
            closure = "negative_reference_positive_controls"
        expected_status = (
            "COMPLETED_V_FIXED_ROSTER"
            if positive["passed"]
            else "STOPPED_REFERENCE_POSITIVE_CONTROL_FAILED"
        )
        _need(
            worker_result["status"] == expected_status,
            "Worker stop/status differs from independently derived gate",
        )
        replay_saved = reader.json(run / "worker", "parent_replay_gate.json")
        expected_replay_controls = config["expected_parent_replay_controls"]
        _need(
            replay_saved["passed"] is True
            and replay_saved["episodes"] == 96
            and replay_saved["controls"]
            == expected_replay_controls
            == sum(r["controls"] for r in prefixes if r["kind"] == "parent_replay"),
            "Saved unchanged-parent gate/census differs",
        )
        positive_saved = reader.json(run / "worker", "reference_positive_control_gate.json")
        _need(
            positive_saved["passed"] == positive["passed"],
            "Saved positive-control admission differs",
        )
        for panel, values in positive["panels"].items():
            _need(
                positive_saved["panels"][panel]["walking"] == values["walking"]
                and positive_saved["panels"][panel]["survived"] == values["survival"],
                "Saved positive-control counts differ",
            )
        if screen is not None:
            saved_screen = reader.json(run / "worker", "reference_usefulness_screen.json")
            _need(
                saved_screen["passed"] == screen["passed"]
                and saved_screen["same_qualifying_streams"] == screen["same_qualifying_streams"],
                "Saved noninitial usefulness decision differs",
            )
            for panel, streams in screen["panels"].items():
                for stream_number, values in streams.items():
                    saved_values = saved_screen["panels"][panel]["streams"][stream_number]
                    for actor in ("parent", "reference"):
                        _need(
                            saved_values[actor]["walking"] == values[actor]["walking"]
                            and saved_values[actor]["survived"] == values[actor]["survival"],
                            "Saved noninitial walking/survival counts differ",
                        )
        accounting = reconcile_costs(worker_result, prefixes, stream, work)
        storage = worker_result["costs"]["storage"]
        for prefix in ("requested", "completed"):
            for key, value in (
                ("controls", stream["saved_controls"]),
                ("physics_substeps", 4 * stream["saved_controls"]),
                ("neural_rows", stream["saved_controls"]),
                ("backward_rows", 0),
                ("optimizer_steps", 0),
            ):
                _need(storage[prefix + "_" + key] == value, "V stored runtime cost counters differ")
        _need(storage["evidence_poisoned"] is False, "V evidence stream reports a partial write")
        stream_bytes = sum(
            reader.path(run / "worker/evidence", name).stat().st_size
            for name in ("arrays.bin", "arrays.index.jsonl", "ledger.jsonl")
        )
        _need(
            storage["storage_bytes"] == stream_bytes,
            "V recorded evidence storage bytes differ from actual files",
        )
        reader.read(out, "prefix_curves.jsonl", maximum=MAX_FORWARDS * 512)
        result["prefix_curve_file"] = dict(
            file="prefix_curves.jsonl", **reader.files[str(out / "prefix_curves.jsonl")]
        )
        reader.unchanged()
        reader.check()
        result.update(
            passed=True,
            full_scientific_audit_passed=True,
            closure_reason=closure,
            static_context_gate=static,
            unchanged_parent_replay_gate=dict(
                passed=True, traces=96, max_errors=replay_differences
            ),
            reference_positive_control_gate=positive,
            reference_usefulness_screen=screen,
            reference_screen_passed=None if screen is None else screen["passed"],
            outcomes=prefixes,
            complete_cost_accounting=accounting,
            raw_stream=stream,
            worker_journal=dict(
                requested=dict(work.requested),
                completed=dict(work.completed),
                unknown_partial=dict(work.unknown),
                returned_unpersisted=dict(work.unpersisted),
                events=work.event_count,
            ),
        )
    except Exception as error:
        result.update(
            error_type=type(error).__name__,
            error=str(error),
            available_outcomes=prefixes,
            unavailable_outcomes="not reconstructed or treated as failures/zeros",
        )
    finally:
        if journal is not None:
            journal.close()
    result["work"] = dict(
        source_read_bytes=reader.total_reads(),
        wall_seconds=time.monotonic() - reader.started,
        peak_rss_bytes=_rss(),
        checkpoint_loads=0 if verifier is None else verifier.loads,
        completed_checkpoint_loads=0 if verifier is None else len(verifier.actors),
        unknown_partial_checkpoint_loads=0
        if verifier is None
        else verifier.loads - len(verifier.actors),
        policy_constructions=0,
        functional_actor_replicas=0 if verifier is None else len(verifier.actors),
        neural_rows=0 if verifier is None else verifier.requested,
        completed_neural_rows=0 if verifier is None else verifier.completed,
        prediction_calls=0 if verifier is None else verifier.calls,
        unknown_partial_neural_rows=0 if verifier is None else verifier.unknown,
        process_cpu_seconds=(
            resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime
            - reader.cpu_started
        ),
        selected_original_array_checks=reader.selected_source_arrays,
    )
    result["new_scientific_work"]["neural_rows"] = result["work"]["neural_rows"]
    result["new_scientific_work"]["prediction_calls"] = result["work"]["prediction_calls"]
    result["new_scientific_work"]["checkpoint_tensor_loads"] = result["work"]["checkpoint_loads"]
    result["input_files"] = reader.files
    _write_json(out, "audit.json", result)
    return result
