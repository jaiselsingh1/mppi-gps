"""Bounded saved-only CIO history plots; no actors, simulator, or source imports.

Historical ``state.pkl`` files are decoded only by the explicit restricted loader:
NumPy arrays and inert Slot/Trajectory records are allowed; optimizer tensor
storage is discarded without importing Torch or evaluating its embedded pickle.
This is evidence navigation, not a replay or scientific audit.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "walker-cio-history-review-1"
ZERO_WORK = dict(
    native_controls=0,
    physics_substeps=0,
    neural_rows=0,
    backward_rows=0,
    optimizer_steps=0,
    mppi_queries=0,
    checkpoint_loads=0,
)
PICKLE_GLOBALS = [
    "src.ciogps.loop.Slot",
    "src.ciogps.trajectory.Trajectory",
    "numpy.ndarray",
    "numpy.dtype",
    "numpy._core.multiarray._reconstruct",
    "numpy.core.multiarray._reconstruct",
    "numpy._core.multiarray.scalar",
    "numpy.core.multiarray.scalar",
    "collections.OrderedDict",
    "torch.storage._load_from_bytes",
    "torch._utils._rebuild_tensor_v2",
]
JOINTS = ("right hip", "right knee", "right ankle", "left hip", "left knee", "left ankle")


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(payload: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        out = {}
        for key, value in items:
            _need(key not in out, "Duplicate JSON key: " + key)
            out[key] = value
        return out

    def reject(value: str) -> None:
        raise ValueError("Nonfinite JSON: " + value)

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=reject)


class Budget:
    """Bound analysis work, distinguishing logical reads from unknown physical IO."""

    def __init__(self, limits: dict):
        self.limits = limits
        self.started = time.monotonic()
        self.cpu_started = time.process_time()
        self.hash_bytes = 0
        self.logical_read_bytes = 0
        self.decoded_array_bytes = 0
        self.output_bytes = 0
        self.config_read_bytes = 0

    def check(self) -> None:
        _need(time.monotonic() - self.started <= self.limits["wall_seconds"], "Wall budget")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        _need(peak_bytes <= self.limits["memory_bytes"], "Memory budget")

    def read(self, binding: dict) -> bytes:
        self.check()
        path = Path(binding["path"])
        _need(path.is_absolute() and path.is_file(), "Explicit absolute source file required")
        before = path.stat()
        _need(before.st_size == binding["bytes"], "Source size changed: " + str(path))
        _need(before.st_size <= self.limits["max_file_bytes"], "Source file budget")
        _need(self.logical_read_bytes + before.st_size <= 64 * 1024**2, "Total source-read budget")
        with path.open("rb") as stream:
            payload = stream.read(before.st_size + 1)
        self.logical_read_bytes += len(payload)
        self.hash_bytes += len(payload)
        _need(len(payload) == before.st_size, "Source size changed during read")
        _need(hashlib.sha256(payload).hexdigest() == binding["sha256"], "Source hash changed")
        _need(before.st_mtime_ns == path.stat().st_mtime_ns, "Source changed during read")
        self.check()
        return payload

    def write(self, path: Path, payload: bytes) -> None:
        self.check()
        _need(self.output_bytes + len(payload) <= self.limits["output_bytes"], "Output budget")
        with path.open("xb") as stream:
            stream.write(payload)
        self.output_bytes += len(payload)

    def receipt(self) -> dict:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(
            **ZERO_WORK,
            wall_seconds=time.monotonic() - self.started,
            cpu_seconds=time.process_time() - self.cpu_started,
            source_hash_bytes=self.hash_bytes,
            logical_source_read_bytes=self.logical_read_bytes,
            decoded_array_bytes=self.decoded_array_bytes,
            output_bytes=self.output_bytes,
            peak_process_rss_bytes=peak if sys.platform == "darwin" else peak * 1024,
            physical_disk_read_bytes=None,
            config_read_and_hash_bytes=self.config_read_bytes,
            peak_rss_scope=(
                "RUSAGE_SELF process-lifetime high-water, including startup and any "
                "prior work in this process; not current RSS or incremental analysis RSS"
            ),
        )


class _Record:
    """Inert historical dataclass replacement; never calls external methods."""


def _discard_storage(_payload: bytes) -> None:
    # Deliberately do not decode Torch's embedded serialization stream.
    return None


def _discard_tensor(*_args: Any) -> None:
    return None


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        import numpy as np

        allowed = {
            ("src.ciogps.loop", "Slot"): _Record,
            ("src.ciogps.trajectory", "Trajectory"): _Record,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch.storage", "_load_from_bytes"): _discard_storage,
            ("torch._utils", "_rebuild_tensor_v2"): _discard_tensor,
        }
        _need((module, name) in allowed, f"Forbidden historical pickle global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, _pid: Any) -> Any:
        raise ValueError("Persistent historical pickle objects are forbidden")


def read_teacher_traces(payload: bytes, max_traces: int = 100) -> tuple[int, list[dict], int]:
    """Decode final retained teachers only; never restores model/optimizer state."""
    import numpy as np

    state = _ArrayUnpickler(io.BytesIO(payload)).load()
    _need(type(state) is dict and type(state.get("slots")) is list, "Historical state schema")
    _need(len(state["slots"]) <= max_traces, "Trace count budget")
    seen = set()

    def array_bytes(value: Any) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            _need(value.dtype.kind != "O", "Object array forbidden")
            return value.nbytes
        if type(value) in (dict, collections.OrderedDict):
            return sum(array_bytes(item) for item in value.values())
        if type(value) in (list, tuple):
            return sum(array_bytes(item) for item in value)
        if type(value) is _Record:
            return array_bytes(value.__dict__)
        return 0

    decoded_bytes = array_bytes(state)
    traces = []
    for slot in state["slots"]:
        _need(type(slot) is _Record, "Historical slot schema")
        trajectory = slot.traj
        if trajectory is None:
            continue
        _need(type(trajectory) is _Record, "Historical trajectory schema")
        states, actions = trajectory.states, trajectory.actions
        _need(isinstance(states, np.ndarray) and states.dtype.kind == "f", "State array schema")
        _need(isinstance(actions, np.ndarray) and actions.dtype.kind == "f", "Action array schema")
        _need(states.shape == (len(actions) + 1, 19), "Walker N+1 fullphysics state shape")
        _need(actions.shape == (len(actions), 6), "Walker action shape")
        _need(0 < len(actions) <= trajectory.length <= 1000, "Historical duration schema")
        _need(np.isfinite(states).all() and np.isfinite(actions).all(), "Nonfinite trace")
        elapsed = states[:, 0] - states[0, 0]
        _need(np.all(np.diff(elapsed) > 0), "Nonincreasing saved simulator time")
        traces.append(
            dict(
                id=slot.name,
                round=int(state["round"]),
                controls=len(actions),
                intended_controls=int(trajectory.length),
                fell=bool(trajectory.fell),
                start_simulator_time=float(states[0, 0]),
                duration_seconds=float(elapsed[-1]),
                reset_seed=None,
                planner_seed=int(trajectory.planner_seed),
                start_state=states[0].tolist(),
                time=elapsed.copy(),
                angles=states[:, 4:10].copy(),
            )
        )
    return int(state["round"]), traces, decoded_bytes


def flatten_metrics(rows: list[dict]) -> list[dict]:
    """Keep every reported round and numeric field; absent values remain absent."""
    result = []
    for row in rows:
        flat = {}

        def visit(value: dict, prefix: str = "") -> None:
            for key, item in value.items():
                name = prefix + key
                if type(item) is dict:
                    visit(item, name + ".")
                elif type(item) in (int, float) and math.isfinite(item):
                    flat[name] = item

        visit(row)
        _need(type(flat.get("round")) is int, "Reported round required")
        result.append(flat)
    rounds = [row["round"] for row in result]
    _need(
        len(set(rounds)) == len(rounds) and rounds == sorted(rounds), "Duplicate/unordered rounds"
    )
    return result


def inventory(run: Path) -> list[dict]:
    """Names/sizes before decoding; checkpoint files are never loaded."""
    _need(run.is_absolute() and run.is_dir(), "Explicit absolute run directory required")
    return [
        dict(path=str(path), bytes=path.stat().st_size, suffix=path.suffix)
        for path in sorted(run.iterdir())
        if path.is_file()
    ]


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _csv(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _figure(fig: Any, out: Path, budget: Budget) -> None:
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=140)
    budget.write(out, stream.getvalue())


def render(config_path: Path, out: Path) -> dict:
    """Create a fresh report from frozen explicit inputs; never regenerates traces."""
    _need(config_path.stat().st_size <= 64 * 1024, "Configuration byte ceiling")
    config_bytes = config_path.read_bytes()
    config = _json(config_bytes)
    _need(config["schema"] == SCHEMA and config["launch_ready"] is True, "Reviewed config required")
    limits = config["limits"]
    _need(0 < limits["wall_seconds"] <= 1200, "Wall ceiling")
    _need(0 < limits["memory_bytes"] <= 1024**3, "Memory ceiling")
    _need(0 < limits["output_bytes"] <= 100 * 1024**2, "Output ceiling")
    _need(limits["max_file_bytes"] <= 64 * 1024**2, "File ceiling")
    _need(config["body"] == "MuJoCo Walker2d, 9 qpos / 9 qvel / 6 torques", "Body binding")
    _need(config["trace_loader"] == "restricted_numpy_pickle_v1", "Explicit loader consent")
    _need(config["allowed_pickle_globals"] == PICKLE_GLOBALS, "Restricted loader binding")
    budget = Budget(limits)
    budget.config_read_bytes = len(config_bytes)
    budget.check()
    out.mkdir(parents=True, exist_ok=False)
    budget.write(out / "config.json", config_bytes)
    all_rows, traces_by_run, report_runs = [], {}, []
    # Inventory is persisted before any historical array package is decoded.
    inventories = {run["id"]: inventory(Path(run["directory"])) for run in config["runs"]}
    budget.write(out / "inventory.json", _encoded(inventories))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for run in config["runs"]:
        budget.check()
        bindings = run["inputs"]
        metadata = _json(budget.read(bindings["run"]))
        rows = [
            _json(line) for line in budget.read(bindings["metrics"]).splitlines() if line.strip()
        ]
        flat = flatten_metrics(rows)
        all_rows.extend(dict(run=run["id"], **row) for row in flat)
        round_number, traces, decoded_bytes = read_teacher_traces(budget.read(bindings["teachers"]))
        _need(round_number == flat[-1]["round"], "Final teacher/metrics round mismatch")
        traces_by_run[run["id"]] = traces
        budget.decoded_array_bytes += decoded_bytes
        budget.check()
        trace_metadata = [
            {k: v for k, v in trace.items() if k not in ("time", "angles")} for trace in traces
        ]
        fresh = _json(budget.read(bindings["fresh"])) if "fresh" in bindings else None
        report_runs.append(
            dict(
                id=run["id"],
                reported_rounds=[row["round"] for row in flat],
                metrics=rows,
                provenance=metadata,
                final_teacher_round=round_number,
                final_teacher_traces=trace_metadata,
                teacher_trace_role="Final retained incumbents, not last rejected candidates",
                student_noise_joint_traces=dict(
                    available=False, reason="No saved student/noise-conditioned joint-state arrays"
                ),
                standalone_fresh=fresh,
                student_joint_traces=dict(
                    available=False, reason="No saved student state arrays in source inventory"
                ),
                earlier_teacher_joint_traces=dict(
                    available=False, reason="Only final overwritten state.pkl retained"
                ),
                alignment="Each saved trace uses its own elapsed simulator time; no phase/path alignment",
                reset_binding="Exact saved initial state; reset seed not encoded per retained slot",
                source_inputs=bindings,
                omitted_saved_fields={
                    "optimizer": "Tensor constructors discarded; embedded Torch storage not decoded",
                    "warmstarts_and_planner_snapshots": "Loaded as saved arrays, not plotted",
                    "sensor_state": "No separately saved sensor-state trace in this package",
                },
            )
        )
        lengths = [len(t["time"]) for t in traces]
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            offsets=np.cumsum([0, *lengths], dtype=np.int64),
            elapsed_seconds=np.concatenate([t["time"] for t in traces]),
            joint_angles_radians=np.concatenate([t["angles"] for t in traces]),
        )
        budget.write(out / (run["id"] + "_teacher_angles.npz"), buffer.getvalue())
        for prefix in (None, 100):
            fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
            for trace in traces:
                stop = len(trace["time"]) if prefix is None else min(prefix + 1, len(trace["time"]))
                for j, ax in enumerate(axes.flat):
                    ax.plot(trace["time"][:stop], trace["angles"][:stop, j], lw=0.7, alpha=0.55)
            for joint, ax in zip(JOINTS, axes.flat, strict=True):
                ax.set(title=joint, xlabel="elapsed simulator seconds", ylabel="angle (radians)")
            view = (
                "all saved controls" if prefix is None else "first 100 controls (up to 101 states)"
            )
            fig.suptitle(
                f"Walker2d — {run['id']}, retained round {round_number}; {view}\n"
                "All retained trials/branches, dependent traces; no student traces saved"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            _figure(fig, out / f"{run['id']}_teacher_{prefix or 'full'}.png", budget)
            plt.close(fig)
    budget.write(out / "round_metrics.csv", _csv(all_rows))
    columns = [
        key
        for key in dict.fromkeys(k for row in all_rows for k in row)
        if key not in ("run", "round", "seed")
    ]
    for page in range(0, len(columns), 6):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        for ax, column in zip(axes.flat, columns[page : page + 6]):
            for run in config["runs"]:
                rows = [row for row in all_rows if row["run"] == run["id"]]
                ax.plot(
                    [row["round"] for row in rows],
                    [row.get(column, np.nan) for row in rows],
                    "o-",
                    label=run["id"],
                    ms=3,
                )
            ax.set(title=column, xlabel="reported round (all saved rounds)")
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(columns[page : page + 6]) :]:
            ax.set_visible(False)
        fig.suptitle("Walker2d historical CIO-inspired loops — saved metrics, no rerun")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        _figure(fig, out / f"round_metrics_{page // 6 + 1:02}.png", budget)
        plt.close(fig)
    result = dict(
        schema=SCHEMA,
        status="SAVED_ONLY_REVIEW",
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        body=config["body"],
        runs=report_runs,
        limitations=[
            "No independent replay or original-source equivalence audit",
            "No physical, inference, fitting or checkpoint work",
            "Teacher joint curves show final retained arrays only; no regenerated earlier rounds",
            "Post-fall observations stay missing; no trace padding",
            "Optimizer tensor pickle payloads discarded, never loaded",
        ],
        accounting=budget.receipt()
        | {"scope": "Before result and final receipt writes; full output bytes in accounting.json"},
    )
    budget.write(out / "result.json", _encoded(result))
    receipt = budget.receipt()
    receipt["timing_scope"] = "Measured immediately before final receipt serialization/write"
    # The self-inclusive output byte count reaches a fixed point in its digit width.
    prior_bytes = budget.output_bytes
    while True:
        payload = _encoded(receipt)
        total_bytes = prior_bytes + len(payload)
        if receipt["output_bytes"] == total_bytes:
            break
        receipt["output_bytes"] = total_bytes
    budget.write(out / "accounting.json", payload)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    render(args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
