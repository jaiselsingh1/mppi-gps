"""Independent saved-evidence primitives for the native Walker task.

The serialization contracts intentionally differ: policy tensors include names,
shape and dtype; fitting matrices first become float32; raw HDF descriptors hash
array bytes only; optimizer trees preserve container and scalar types. These
contracts reproduce the immutable historical checker without importing project
policy, planner, fitting or simulator code. Torch only decodes CPU checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch


def plain(v: Any) -> Any:
    """Convert NumPy values for JSON receipts without changing hash inputs."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [plain(x) for x in v]
    return v


def sha(p: Path) -> str:
    """Hash exact file bytes in bounded blocks."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def tensor_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Reproduce the named-tensor walker-tanh-policy-v1 identity contract."""
    h = hashlib.sha256(b"walker-tanh-policy-v1")
    for name, t in sorted(state.items()):
        a = t.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name, list(a.shape), str(a.dtype)], separators=(",", ":")).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fingerprint(value: Any) -> str:
    """Hash the typed optimizer/RNG tree, including list versus tuple identity."""
    h = hashlib.sha256()

    def visit(v):
        if isinstance(v, torch.Tensor):
            visit(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            h.update(json.dumps(["array", list(a.shape), str(a.dtype)]).encode())
            h.update(a.tobytes())
        elif isinstance(v, dict):
            h.update(b"dict")
            for k in sorted(v, key=lambda k: (type(k).__name__, repr(k))):
                visit((type(k).__name__, k))
                visit(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for x in v:
                visit(x)
        else:
            h.update(json.dumps([type(v).__name__, v], allow_nan=False).encode())

    visit(value)
    return h.hexdigest()


def fit_hash(v: np.ndarray) -> str:
    """Hash fitting matrices after float32 conversion, shape and dtype prefix."""
    a = np.ascontiguousarray(v, dtype=np.float32)
    h = hashlib.sha256(json.dumps([list(a.shape), str(a.dtype)]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def obs(s: np.ndarray) -> np.ndarray:
    """Derive the task's native clipped pre-action observation from full state."""
    return np.concatenate((s[..., 2:10], np.clip(s[..., 10:19], -10.0, 10.0)), axis=-1)


class Audit:
    """Bind immutable files and independently reconcile saved native execution."""

    def __init__(self, base: Path, source: Path):
        self.base, self.source = (base, source)
        self.files = {}
        self.arrays = {}
        self.hdfs = {}
        self.checks = 0
        self.stages = {}
        self.report = {}

    def need(self, condition: Any, message: str) -> None:
        self.checks += 1
        if not condition:
            raise AssertionError(message)

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        self.need(
            np.shape(actual) == np.shape(expected)
            and np.allclose(actual, expected, atol=1e-10, rtol=1e-12),
            message,
        )

    def bind(self, path: Path, expected: str | None = None) -> dict[str, Any]:
        path = Path(path).resolve()
        record = dict(bytes=path.stat().st_size, sha256=sha(path))
        self.need(self.files.setdefault(str(path), record) == record, "Changed input " + str(path))
        if expected is not None:
            self.need(record["sha256"] == expected, "Pinned hash " + str(path))
        return record

    def load(self, path: Path) -> Any:
        self.bind(path)
        return json.loads(path.read_text())

    def group(
        self, path: Path, ref: dict[str, Any], keys: list[str] | None = None
    ) -> dict[str, Any]:
        """Decode retained HDF arrays and verify dtype/shape/exact byte digests."""
        self.need(
            ref["file"] == "raw.h5" and ref["group"].startswith("arrays/"), "Local HDF reference"
        )
        path = Path(path).resolve()
        if path not in self.hdfs:
            st = path.stat()
            self.hdfs[path] = (h5py.File(path, "r"), st.st_size, st.st_mtime_ns)
        group = self.hdfs[path][0][ref["group"]]
        self.need(str(group.attrs["tag"]) == ref["tag"], "HDF group owner")
        result = {}
        for key in ref["arrays"] if keys is None else keys:
            meta = ref["arrays"][key]
            dataset = group[key]
            dtype = np.dtype(meta["dtype"])
            array = np.asarray(dataset[()])
            array = (
                np.char.decode(array, "utf-8").astype(dtype)
                if dtype.kind == "U"
                else array.astype(dtype, copy=False)
            )
            digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            self.need(
                list(array.shape) == meta["shape"]
                and str(dataset.attrs["numpy_dtype"]) == meta["dtype"]
                and (str(dataset.attrs["encoding"]) == meta["encoding"])
                and (digest == meta["sha256"] == str(dataset.attrs["sha256"])),
                "HDF actual bytes " + key,
            )
            self.arrays[str(path) + ":" + ref["group"] + "/" + key] = meta
            result[key] = json.loads(str(array.item())) if meta["encoding"] == "json" else array
        return result

    def desc(self, meta: dict[str, Any], value: Any) -> None:
        array = np.asarray(value, dtype=np.dtype(meta["dtype"]))
        self.need(
            list(array.shape) == meta["shape"]
            and hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == meta["sha256"],
            "Query/trace descriptor binding",
        )

    def checkpoint(self, path: Path) -> dict[str, Any]:
        """Decode CPU tensor packages; do not instantiate or evaluate a policy."""
        self.bind(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.need(
            checkpoint["schema"] == "walker-tanh-policy-v1"
            and tensor_hash(checkpoint["state_dict"]) == checkpoint["policy_hash"],
            "Checkpoint tensor owner",
        )
        self.need(
            checkpoint.get("obs_dim", 17) == 17
            and tuple(checkpoint["state_dict"]["network.0.weight"].shape) == (128, 17),
            "17-input actor",
        )
        return checkpoint

    def seal(self, folder: Path) -> dict[str, Any]:
        m = self.load(folder / "manifest.json")
        rows = (
            m["files"].items()
            if isinstance(m["files"], dict)
            else ((r["file"], r["sha256"]) for r in m["files"])
        )
        for p, h in rows:
            self.bind(folder / p, h)
        return m

    def ledger(self, path: Path, failed_reservation: bool = False) -> dict[str, Any]:
        """Recount every saved boundary, terminal receipt and work category."""
        result = self.load(path / "result.json")
        ledger_path = path / "evidence/ledger.jsonl"
        self.bind(ledger_path)
        counts = Counter()
        pending = {}
        measures = {}
        resets = {}
        episodes = {}
        live = defaultdict(list)
        special = defaultdict(list)
        last = None
        query_rows = 0
        for index, line in enumerate(ledger_path.open()):
            event = json.loads(line)
            kind = event["event"]
            self.need(
                event["id"] == f"event-{index:09d}" and event["kind"] == kind, "Ledger identity"
            )
            self.need(
                last is None or event["wall_seconds"] >= last["wall_seconds"], "Ledger chronology"
            )
            last = event
            if kind == "query_requested":
                query_rows += 1
                pending[event["id"]] = event
                mode, requested_controls = (event["mode"], event["requested_controls"])
                self.need(
                    mode in ("live", "model")
                    and type(requested_controls) is int
                    and (requested_controls > 0)
                    and (event["requested_physics_substeps"] == 4 * requested_controls),
                    "Native request",
                )
                counts["requested_controls"] += requested_controls
                counts["requested_" + mode + "_controls"] += requested_controls
                counts["requested_physics_substeps"] += 4 * requested_controls
                counts[mode + "_queries"] += 1
            elif kind in ("query_complete", "query_error"):
                request = pending.pop(event["request_id"])
                mode = request["mode"]
                requested_controls = request["requested_controls"]
                completed_controls = event["completed_controls"]
                self.need(
                    event["query_id"] == request["query_id"]
                    and event["mode"] == mode
                    and (completed_controls in (0, requested_controls))
                    and (event["completed_physics_substeps"] == 4 * completed_controls),
                    "Native terminal ownership",
                )
                counts["completed_controls"] += completed_controls
                counts["completed_" + mode + "_controls"] += completed_controls
                counts["completed_physics_substeps"] += 4 * completed_controls
                if kind == "query_error":
                    counts["failed_queries"] += 1
                    unknown = event["unknown_partial_controls"]
                    counts["unknown_partial_controls"] += unknown
                    counts["unknown_partial_physics_substeps"] += 4 * unknown
                elif mode == "live" and "episode" in request:
                    live[request["episode"]].append((request, event))
            elif kind == "measurement_requested":
                measures[event["id"]] = event
                counts["measurement_requests"] += 1
                counts["requested_" + event["measure_kind"]] += event["amount"]
            elif kind in ("measurement_complete", "measurement_error"):
                request = measures.pop(event["request_id"])
                self.need(
                    all(
                        (
                            event.get(k) == request.get(k)
                            for k in (
                                "measure_kind",
                                "amount",
                                "context",
                                "purpose",
                                "arm",
                                "cohort",
                                "episode",
                                "step",
                            )
                        )
                    ),
                    "Measurement terminal owner",
                )
                prefix = "completed_" if kind == "measurement_complete" else "unknown_partial_"
                counts[prefix + event["measure_kind"]] += event["amount"]
                if kind == "measurement_error":
                    counts["failed_measurements"] += 1
            elif kind == "query_rejected":
                counts["rejected_queries"] += 1
            elif kind == "measurement_rejected":
                counts["rejected_measurements"] += 1
            elif kind == "reset_complete":
                counts["resets"] += 1
                resets[event["tag"]] = event["seed"]
            elif kind == "reset_error":
                counts["reset_errors"] += 1
            elif kind == "snapshot_restored":
                counts["restores"] += 1
            elif kind == "episode_complete":
                self.need(event["tag"] not in episodes, "Unique complete episode")
                episodes[event["tag"]] = event
                counts["episodes"] += 1
            special[kind].append(event)
        self.need(
            last is not None
            and last["event"] == "runtime_close"
            and (not pending)
            and (not measures),
            "Closed stage only",
        )
        if failed_reservation:
            self.need(
                result["error"]["type"] == "TypeError"
                and "mode" in result["error"]["message"]
                and (query_rows == 0),
                "Known pre-boundary harness failure",
            )
            counts.update(
                requested_controls=1,
                requested_live_controls=1,
                requested_physics_substeps=4,
                live_queries=1,
            )
        counts["queries"] = counts["model_queries"] + counts["live_queries"]
        counts["physics_frames"] = counts["completed_physics_substeps"]
        for alias in ("neural_rows", "backward_rows", "optimizer_steps"):
            counts[alias] = counts["completed_" + alias]
        for key, value in result["work"].items():
            if key in ("wall_seconds", "snapshots"):
                continue
            self.need(
                value == counts[key] and last["totals"].get(key, 0) == value, "Work census " + key
            )
        self.need(
            result["work"]["snapshots"] == last["totals"]["snapshots"],
            "Saved snapshot count consistency only",
        )
        entry = dict(
            path=path,
            result=result,
            counts=counts,
            episodes=episodes,
            live=live,
            resets=resets,
            events=special,
            logged_query_requests=query_rows,
        )
        self.stages[str(path.relative_to(self.base))] = dict(
            work=dict(counts),
            wall_seconds=result["wall_seconds"],
            known_preboundary_failed_attempt=int(failed_reservation),
            logged_query_requests=query_rows,
        )
        return entry

    def episode(self, stage: dict[str, Any], tag: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Recompute native complete cost and flags from actual executed arrays."""
        event = stage["episodes"][tag]
        steps = event["steps"]
        arrays = self.group(stage["path"] / "evidence/raw.h5", event["raw"])
        states, actions = (arrays["states"], arrays["actions"])
        self.need(
            states.shape == (steps + 1, 19)
            and actions.shape == (steps, 6)
            and (arrays["warmstarts"].shape == (steps + 1, 9))
            and (arrays["integrations"].shape == (steps + 1, 91)),
            "Native trace shapes",
        )
        self.need(
            0 < steps <= 1000
            and all((np.isfinite(v).all() for v in arrays.values() if isinstance(v, np.ndarray))),
            "Finite executed trace",
        )
        self.need(
            np.array_equal(arrays["observations"], obs(states[:-1]))
            and np.array_equal(arrays["velocities"], states[1:, 10]),
            "Pre-action/velocity alignment",
        )
        self.need(
            np.all(np.abs(actions) <= 1) and stage["resets"][tag] == event["seed"],
            "Native bounds/reset owner",
        )
        healthy = (
            (states[:, 2] > 0.8) & (states[:, 2] < 2) & (states[:, 3] > -1) & (states[:, 3] < 1)
        )
        self.need(np.all(healthy[:-1]), "Stop at first fall")
        done = not healthy[-1]
        survived = steps == 1000 and (not done)
        mae = float(np.abs(states[1:, 10] - 1.5).mean())
        success = survived and mae <= 0.3
        actions_float = actions.astype(float)
        costs = (
            2 * np.abs(1.5 - states[1:, 10])
            + 0.5 * states[1:, 11] ** 2
            + 0.1 * states[1:, 3] ** 2
            + 5 * ~healthy[1:]
            + 0.001 * np.sum(actions_float * actions_float, axis=1)
        )
        if done:
            costs[-1] += 20
        absorbing = 5 * (1000 - steps) if done else 0
        self.equal(arrays["costs"], costs, "Native post-state cost/terminal once")
        self.equal(event["cost"], costs.sum() + absorbing, "Complete cost/absorbing tail")
        self.equal(arrays["terminal_cost"], 20 * done, "Terminal")
        self.equal(arrays["terminal_adjustment"], 0, "No double terminal")
        self.equal(arrays["absorbing_cost"], absorbing, "Tail")
        self.equal(event["mae"], mae, "Actual MAE")
        self.need(
            event["survived"] == survived
            and event["success"] == success
            and (event["complete"] == bool(done or steps == 1000))
            and event["complete"],
            "Full-task flags",
        )
        self.equal(np.diff(states[:, 0]), np.full(steps, 0.008), "Native control time spacing")
        pairs = stage["live"][tag]
        self.need(len(pairs) == steps, "One live boundary per action")
        for step, (before, after) in enumerate(pairs):
            self.need(before["step"] == step, "Native step order")
            for k, v in dict(
                action=actions_float[step],
                state=states[step],
                warmstart=arrays["warmstarts"][step],
                integration=arrays["integrations"][step],
            ).items():
                self.desc(before["inputs"]["arrays"][k], v)
            for k, v in dict(
                state=states[step + 1],
                warmstart=arrays["warmstarts"][step + 1],
                integration=arrays["integrations"][step + 1],
                observation=obs(states[step + 1]),
                cost=costs[step],
            ).items():
                self.desc(after["result"]["arrays"][k], v)
        return (event, arrays)

    def summary(self, stage: dict[str, Any], value: dict[str, Any]) -> None:
        rows = [stage["episodes"][tag] for tag in value["tags"]]
        self.need(len(rows) == value["episodes"], "Cohort count")
        for k, v in dict(
            costs=[r["cost"] for r in rows],
            mean_cost=np.mean([r["cost"] for r in rows]),
            mean_steps=np.mean([r["steps"] for r in rows]),
            mean_velocity_mae=np.mean([r["mae"] for r in rows]),
        ).items():
            if k in value:
                self.equal(value[k], v, "Cohort " + k)
        self.need(
            value["successes"] == sum((r["success"] for r in rows))
            and value["survived"] == sum((r["survived"] for r in rows)),
            "Cohort outcomes",
        )
