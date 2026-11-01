"""Independent saved-array K1.2 motion audit; no policy or simulator code.

The checker reopens the exact original sources, verifies their array bytes,
and independently enumerates every declared ordered-window comparison. It
does not import production loading, metric, reporting or execution code.
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

import h5py
import numpy as np

from .trajectory_review import REFERENCE_HASH, ReviewAudit, contained

WINDOW = STRIDE = 25
MINIMUM_LAG = 100
ATOL = RTOL = 1e-10


class MotionAudit(ReviewAudit):
    """Hash proofs plus explicit independent audit read/work accounting."""

    def __init__(self) -> None:
        super().__init__()
        self.started = time.perf_counter()
        self.source_stats: dict[str, tuple[int, int]] = {}
        self.arrays: dict[str, dict] = {}
        self.work = dict(
            source_hash_bytes=0,
            source_io_bytes=0,
            dataset_attempts=0,
            completed_datasets=0,
            failed_datasets=0,
            decoded_bytes=0,
            within_queries=0,
            cross_queries=0,
            aligned_state_pairs=0,
        )

    def need(self, condition: Any, message: str) -> None:
        if time.perf_counter() - self.started > 600:
            raise TimeoutError("Independent saved-motion audit deadline exhausted")
        super().need(condition, message)

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        first, second = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.need(
            first.shape == second.shape
            and np.isfinite(first).all()
            and np.isfinite(second).all()
            and np.allclose(first, second, atol=ATOL, rtol=RTOL),
            message,
        )

    def bind_source(self, path: Path, sha: str, size: int) -> None:
        """Hash each full original once; later checks compare immutable stats."""
        key = str(path.resolve())
        self.need(key not in self.source_stats, "Original source is hashed exactly once")
        actual = self.bind(path, sha)
        self.work["source_hash_bytes"] += actual["bytes"]
        self.need(actual["bytes"] == size, "Exact original whole-file byte count")
        self.source_stats[key] = (path.stat().st_size, path.stat().st_mtime_ns)
        self.need(self.work["source_hash_bytes"] <= 2 * 1024**3, "Original hash-byte ceiling")

    def check_source_stats(self) -> None:
        for name, expected in self.source_stats.items():
            path = Path(name)
            self.need(
                (path.stat().st_size, path.stat().st_mtime_ns) == expected,
                "Original source unchanged after saved-array reads",
            )


class CountedSource:
    """Read-only file adapter that counts actual HDF/NPZ decoder reads."""

    def __init__(self, path: Path, audit: MotionAudit) -> None:
        self.stream = path.open("rb")
        self.audit = audit

    def __enter__(self) -> CountedSource:
        return self

    def __exit__(self, *args) -> None:
        self.stream.close()

    def read(self, size: int = -1) -> bytes:
        value = self.stream.read(size)
        self._charge(len(value))
        return value

    def readinto(self, buffer) -> int:
        count = self.stream.readinto(buffer)
        self._charge(count)
        return count

    def _charge(self, count: int) -> None:
        self.audit.work["source_io_bytes"] += count
        self.audit.need(
            self.audit.work["source_io_bytes"] <= 256 * 1024**2,
            "Original source decoder-read ceiling",
        )

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.stream.seek(offset, whence)

    def tell(self) -> int:
        return self.stream.tell()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def flush(self) -> None:
        pass


def verify_array(a: MotionAudit, value: np.ndarray, descriptor: dict, message: str) -> None:
    a.need(list(value.shape) == descriptor["shape"], message + " exact shape")
    a.need(value.dtype == np.dtype(descriptor["dtype"]), message + " exact dtype")
    a.need(
        value.dtype.kind in "fiu" and np.isfinite(value).all(), message + " finite native numbers"
    )
    a.need(
        hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == descriptor["sha256"],
        message + " exact C-contiguous array-byte SHA-256",
    )


def read_array(
    a: MotionAudit,
    path: Path,
    key: str,
    descriptor: dict,
    group: str | None = None,
    label: str | None = None,
) -> np.ndarray:
    a.need(str(path.resolve()) in a.source_stats, "Hash whole original before decoding a member")
    a.work["dataset_attempts"] += 1
    try:
        with CountedSource(path, a) as source:
            if group is None:
                with np.load(source, allow_pickle=False) as archive:
                    value = np.array(archive[key], copy=True)
            else:
                with h5py.File(source, "r") as archive:
                    dataset = archive[group][key]
                    a.need(
                        dataset.shape == tuple(descriptor["shape"]), "Original HDF shape metadata"
                    )
                    a.need(
                        dataset.dtype == np.dtype(descriptor["dtype"]),
                        "Original HDF dtype metadata",
                    )
                    a.need(
                        dataset.attrs["sha256"] == descriptor["sha256"], "Original HDF SHA metadata"
                    )
                    value = np.array(dataset[()], copy=True)
        a.work["decoded_bytes"] += value.nbytes
        verify_array(a, value, descriptor, "Original " + key)
        if label is not None:
            a.need(label not in a.arrays, "No duplicate decoded source-array owner")
            a.arrays[label] = array_binding(value)
        a.work["completed_datasets"] += 1
        a.need(a.work["decoded_bytes"] <= 32 * 1024**2, "Selected decoded-byte ceiling")
        return value
    except BaseException:
        a.work["failed_datasets"] += 1
        raise


def array_binding(value: np.ndarray) -> dict:
    return descriptor_binding(value) | dict(bytes=value.nbytes)


def descriptor_binding(value: np.ndarray) -> dict:
    return dict(
        shape=list(value.shape),
        dtype=value.dtype.str,
        sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    )


def physical_coordinates(states: np.ndarray) -> np.ndarray:
    """Drop only absolute time and horizontal translation; preserve raw qvel."""
    return np.asarray(states[:, 2:19], dtype=np.float64)


def segment_starts(n: int) -> list[int]:
    return list(range(0, n - WINDOW + 1, STRIDE)) if n >= WINDOW else [0]


def distance(first: np.ndarray, second: np.ndarray) -> tuple[float, dict[str, float]]:
    """Mean of 17 fixed SI-scaled coordinate differences, then RMS in time."""
    count = min(len(first), len(second), WINDOW)
    delta = np.asarray(first[:count], dtype=np.float64) - np.asarray(
        second[:count], dtype=np.float64
    )
    angles = np.arctan2(np.sin(delta[:, 1:8]), np.cos(delta[:, 1:8]))
    denominator = 17 * count
    contributions = dict(
        height=float(np.sum(delta[:, 0] ** 2) / denominator),
        angles=float(np.sum(angles**2) / denominator),
        velocities=float(np.sum(delta[:, 8:] ** 2) / denominator),
    )
    return math.sqrt(sum(contributions.values())), contributions


def match_record(first: dict, second: dict, start: int, other_start: int) -> dict:
    h = min(WINDOW, first["N"] - start, second["N"] - other_start)
    first_slice, second_slice = slice(start, start + h), slice(other_start, other_start + h)
    r, contributions = distance(
        first["coordinates"][first_slice], second["coordinates"][second_slice]
    )
    first_actions = np.asarray(first["actions"][first_slice], dtype=np.float64)
    second_actions = np.asarray(second["actions"][second_slice], dtype=np.float64)
    first_predictions = np.asarray(first["reference_predictions"][first_slice], dtype=np.float64)
    second_predictions = np.asarray(second["reference_predictions"][second_slice], dtype=np.float64)
    return dict(
        first_start_control=start,
        second_start_control=other_start,
        H=h,
        R=r,
        squared_contributions=contributions,
        action_mse=float(np.mean((first_actions - second_actions) ** 2)),
        first_reference_E=float(np.mean((first_actions - first_predictions) ** 2)),
        second_reference_E=float(np.mean((second_actions - second_predictions) ** 2)),
    )


def minimum_key(value: dict) -> tuple[float, int, int]:
    return value["R"], value["first_start_control"], value["second_start_control"]


def curves(row: dict) -> list[dict]:
    output = []
    state = np.asarray(row["states"], dtype=np.float64)
    physical = physical_coordinates(state)
    actions = np.asarray(row["actions"], dtype=np.float64)
    predictions = np.asarray(row["reference_predictions"], dtype=np.float64)
    displacement = state[:, 1] - state[0, 1]
    for start in range(0, row["N"], WINDOW):
        stop = min(start + WINDOW, row["N"])
        span = slice(start, stop)
        residual = (actions[span] - predictions[span]) ** 2
        output.append(
            dict(
                start_control=start,
                stop_control_exclusive=stop,
                rows=stop - start,
                physical={
                    name: getattr(np, name)(physical[span], axis=0).tolist()
                    for name in ("mean", "min", "max")
                },
                displacement={
                    name: float(getattr(np, name)(displacement[span]))
                    for name in ("mean", "min", "max")
                },
                actions={
                    name: getattr(np, name)(actions[span], axis=0).tolist()
                    for name in ("mean", "min", "max")
                },
                reference_error=dict(
                    E=float(residual.mean()),
                    per_actuator_mse=residual.mean(axis=0).tolist(),
                    min=residual.min(axis=0).tolist(),
                    max=residual.max(axis=0).tolist(),
                ),
            )
        )
    return output


def recompute_probe(a: MotionAudit, data: list[dict]) -> dict:
    """Enumerate every eligible unordered pair, never sample the saved motions."""
    prepared, trajectories, cross = [], [], []
    for row in data:
        a.need(True, "Within-trajectory probe deadline")
        item = dict(row, coordinates=physical_coordinates(row["states"]))
        prepared.append(item)
        starts = segment_starts(row["N"])
        matrix = [[None for _ in starts] for _ in starts]
        candidates = []
        for i, first in enumerate(starts):
            for j in range(i + 1, len(starts)):
                second = starts[j]
                if second - first < MINIMUM_LAG:
                    continue
                record = match_record(item, item, first, second)
                matrix[i][j] = matrix[j][i] = record["R"]
                candidates.append(record)
                a.work["within_queries"] += 1
                a.work["aligned_state_pairs"] += record["H"]
        errors = (
            np.asarray(row["actions"], dtype=float)
            - np.asarray(row["reference_predictions"], dtype=float)
        ) ** 2
        trajectories.append(
            dict(
                id=row["id"],
                source=row["source"],
                index=row["index"],
                N=row["N"],
                E=float(errors.mean()),
                per_actuator_mse=errors.mean(axis=0).tolist(),
                first100_E=float(errors[:100].mean()),
                terminal_state_available=row.get("terminal_state") is not None,
                curves=curves(row),
                within=dict(
                    starts=starts,
                    eligible_pairs=len(candidates),
                    R_matrix=matrix,
                    best=min(candidates, key=minimum_key) if candidates else None,
                ),
            )
        )
    for i, first in enumerate(prepared):
        for second in prepared[i + 1 :]:
            a.need(True, "Cross-trajectory probe deadline")
            candidates = []
            for start in segment_starts(first["N"]):
                for other in segment_starts(second["N"]):
                    record = match_record(first, second, start, other)
                    candidates.append(record)
                    a.work["cross_queries"] += 1
                    a.work["aligned_state_pairs"] += record["H"]
            h = min(WINDOW, first["N"], second["N"])
            cross.append(
                dict(
                    first=first["id"],
                    second=second["id"],
                    first_index=first["index"],
                    second_index=second["index"],
                    kind="full" if h == WINDOW else "partial",
                    H=h,
                    eligible_pairs=len(candidates),
                    best=min(candidates, key=minimum_key),
                )
            )
    counts = dict(
        trajectories=len(data),
        rows=sum(row["N"] for row in data),
        curve_bins=sum(len(row["curves"]) for row in trajectories),
        segments=sum(len(segment_starts(row["N"])) for row in data),
        cross_trajectory_pairs=len(cross),
        within_queries=a.work["within_queries"],
        cross_queries=a.work["cross_queries"],
        pair_queries=a.work["within_queries"] + a.work["cross_queries"],
        aligned_state_pairs=a.work["aligned_state_pairs"],
    )
    return dict(trajectories=trajectories, cross_pairs=cross, counts=counts)


def load_originals(a: MotionAudit, catalog: dict, original: dict) -> list[dict]:
    """Read only predeclared original members; never infer physical qvel from observations."""
    for entry in catalog["files"]:
        a.bind_source(Path(entry["path"]), entry["recorded_sha256"], entry["recorded_bytes"])
    records = [
        record for stage in ("A", "B") for record in original["stages"][stage]["trajectories"]
    ]
    a.need(len(records) == len(catalog["roster"]), "Original complete motion roster length")
    output = []
    for index, (entry, recorded) in enumerate(zip(catalog["roster"], records, strict=True)):
        n, physical, prediction = entry["N"], entry["physical_source"], entry["prediction_source"]
        a.prove(
            entry,
            dict(
                index=index,
                id=recorded["trajectory"]["id"],
                source=recorded["trajectory"]["group"],
                N=recorded["trajectory"]["steps"],
            ),
            "Original motion identity/order",
        )
        full = read_array(
            a,
            Path(physical["file"]),
            "states",
            physical["states"],
            physical["group"],
            entry["id"] + "/states",
        )
        terminal = None
        if physical["kind"] == "sealed_teacher_npz":
            a.need(full.shape == (n, 19), "Teacher NPZ retains N actual pre-action states")
            row_steps = read_array(
                a,
                Path(physical["file"]),
                "row_steps",
                physical["row_steps"],
                label=entry["id"] + "/row_steps",
            )
            a.need(np.array_equal(row_steps, np.arange(n)), "Exact teacher source row/control join")
            if physical["terminal"] is not None:
                terminal = read_array(
                    a,
                    Path(physical["file"]),
                    "terminal_state",
                    physical["terminal"],
                    label=entry["id"] + "/terminal_state",
                )
            states = full
        else:
            a.need(
                physical["kind"] == "k1_stage_b_episode_hdf" and full.shape == (n + 1, 19),
                "Native episode retains true N+1 states",
            )
            states, terminal = full[:-1], full[-1].copy()
        a.need(
            (terminal is None) == entry["terminal_missing"], "No fabricated or discarded terminal"
        )
        times = (
            full[:, 0] if terminal is None or len(full) == n + 1 else np.r_[full[:, 0], terminal[0]]
        )
        a.need(
            np.allclose(np.diff(times), 0.008, atol=1e-10, rtol=0),
            "Exact saved control-time increment",
        )
        arrays = {
            key: read_array(
                a,
                Path(prediction["file"]),
                key,
                descriptor,
                prediction["group"],
                entry["id"] + "/" + key,
            )
            for key, descriptor in prediction["arrays"].items()
        }
        a.need(
            set(arrays) == {"predictions", "observations", "executed_actions"},
            "Only declared prediction-matrix members",
        )
        a.need(
            arrays["predictions"].shape == (n, 9, 6) and prediction["reference_column"] == 8,
            "Hash full nine-policy matrix before selecting fixed reference",
        )
        observations = np.concatenate([states[:, 2:10], np.clip(states[:, 10:19], -10, 10)], axis=1)
        a.need(
            np.array_equal(observations, arrays["observations"]),
            "Bit-exact physical-state/native-observation join",
        )
        actions = arrays["executed_actions"]
        a.need(
            actions.shape == (n, 6) and np.all(np.abs(actions) <= 1),
            "Exact finite bounded executed controls",
        )
        a.need(np.all(np.abs(arrays["predictions"]) <= 1), "Saved bounded policy prediction matrix")
        item = dict(
            id=entry["id"],
            source=entry["source"],
            index=index,
            N=n,
            states=states,
            actions=actions,
            reference_predictions=arrays["predictions"][:, 8, :].copy(),
            terminal_state=terminal,
        )
        errors = (
            actions.astype(np.float64) - item["reference_predictions"].astype(np.float64)
        ) ** 2
        a.equal(
            float(errors.mean()),
            recorded["policies"]["reference"]["mse"],
            "Raw reference errors match original saved E",
        )
        for part in recorded["policies"]["reference"]["bins"]:
            span = slice(part["start_control"], part["stop_control_exclusive"])
            a.equal(
                float(errors[span].mean()),
                part["mse"],
                "Raw reference residuals match every original bin",
            )
        output.append(item)
    a.check_source_stats()
    return output


def assert_independent_imports(a: MotionAudit) -> None:
    a.need(
        not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in ("torch", "mujoco", "src.envs", "src.mppi")
        ),
        "No policy, physics or planner imports",
    )
    allowed = {
        "src.gps",
        "src.gps.mppi_gps",
        "src.gps.mppi_gps.audits",
        "src.gps.mppi_gps.audits.trajectory_review",
        "src.gps.mppi_gps.audits.motion_diagnosis",
    }
    a.need(
        not any(name.startswith("src.") and name not in allowed for name in sys.modules),
        "No production core, loader, report or execution imports",
    )


def prove_descriptor(a: MotionAudit, actual: dict, expected: dict, message: str) -> None:
    a.need(actual["shape"] == expected["shape"], message + " shape")
    a.need(np.dtype(actual["dtype"]) == np.dtype(expected["dtype"]), message + " dtype")
    a.need(actual["sha256"] == expected["sha256"], message + " byte SHA")


def source_descriptors(catalog: dict) -> dict[str, dict]:
    result = {}
    for row in catalog["roster"]:
        prefix, physical = row["id"] + "/", row["physical_source"]
        result[prefix + "states"] = physical["states"]
        if physical["row_steps"] is not None:
            result[prefix + "row_steps"] = physical["row_steps"]
        if physical["kind"] == "sealed_teacher_npz" and physical["terminal"] is not None:
            result[prefix + "terminal_state"] = physical["terminal"]
        result.update(
            {prefix + key: value for key, value in row["prediction_source"]["arrays"].items()}
        )
    return result


def verify_catalog(a: MotionAudit, metadata: dict) -> None:
    """Prove every locator/descriptor using original K1 metadata, not preflight assertions."""
    catalog, config, result = [metadata[key] for key in ("motion_roster", "k1_config", "k1_result")]
    capsule, files, inputs = [metadata[key] for key in ("k1_capsule", "k1_files", "k1_inputs")]
    a.need(
        config["policy_order"][8] == "reference" and len(config["policy_order"]) == 9,
        "Original fixed reference prediction column",
    )
    a.need(
        config["policies"]["reference"]["policy_hash"]
        == result["policy_hashes"]["reference"]
        == REFERENCE_HASH,
        "Exact inherited fixed reference actor",
    )
    a.need(capsule["study"] == "action_compatibility_k1", "Original immutable K1 capsule")
    a.need(
        catalog["source_run"] == files["local_directory"], "Original full-file catalog source root"
    )
    root = Path(catalog["source_run"])
    file_map = {row["path"]: row for row in catalog["files"]}
    a.need(len(file_map) == len(catalog["files"]), "Unique original source file roster")
    for path, entry in file_map.items():
        a.need(Path(path).is_relative_to(root), "Original file remains in bound K1 run")
        binding = files["files"][str(Path(path).relative_to(root))]
        a.prove(
            entry,
            dict(
                recorded_sha256=binding["sha256"],
                recorded_bytes=binding["bytes"],
                bytes=binding["bytes"],
            ),
            "Original full-file manifest locator",
        )
    originals = [
        (stage, row) for stage in ("A", "B") for row in result["stages"][stage]["trajectories"]
    ]
    a.need(len(originals) == len(catalog["roster"]), "Complete original motion roster")
    a.need(
        config["trajectory_roster"]
        == [row["trajectory"] for row in result["stages"]["A"]["trajectories"]],
        "Unchanged fourteen teacher descriptors",
    )
    episodes = {row["tag"]: row for row in result["stages"]["B"]["episodes"]}
    used_files = set()
    for index, ((stage, original), row) in enumerate(
        zip(originals, catalog["roster"], strict=True)
    ):
        trajectory = original["trajectory"]
        physical, prediction = row["physical_source"], row["prediction_source"]
        a.prove(
            row,
            dict(
                index=index,
                id=trajectory["id"],
                source=trajectory["group"],
                N=trajectory["steps"],
                ancestry=trajectory["ancestry"],
            ),
            "Original raw-motion source identity",
        )
        a.prove(
            prediction,
            dict(
                file=str(root / f"evidence/stage_{stage.lower()}/raw.h5"),
                group=original["raw"]["group"],
                arrays=original["raw"]["arrays"],
                reference_column=8,
                reference_slice_sha256=None,
            ),
            "Original full prediction-matrix descriptor",
        )
        if stage == "A":
            extracted = trajectory["extracted"]
            expected_path = str(root / "inputs" / trajectory["file"])
            a.prove(
                physical,
                dict(file=expected_path, kind="sealed_teacher_npz", group=None),
                "Original curated teacher source",
            )
            a.need(
                file_map[expected_path]["recorded_sha256"]
                == extracted["file_sha256"]
                == inputs["files"]["inputs/" + trajectory["file"]],
                "NPZ file hash has three historical owners",
            )
            for key in ("states", "row_steps"):
                prove_descriptor(
                    a, physical[key], extracted["arrays"][key], "Original teacher " + key
                )
            if "terminal_state" in extracted["arrays"]:
                a.prove(
                    physical["terminal"],
                    dict(storage="separate_npz_member", key="terminal_state"),
                    "Original separate endpoint representation",
                )
                prove_descriptor(
                    a,
                    physical["terminal"],
                    extracted["arrays"]["terminal_state"],
                    "Original separate endpoint",
                )
            else:
                a.need(physical["terminal"] is None, "Missing discovery endpoint stays missing")
            for source_key, prediction_key in (
                ("actions", "executed_actions"),
                ("observations", "observations"),
            ):
                prove_descriptor(
                    a,
                    prediction["arrays"][prediction_key],
                    extracted["arrays"][source_key],
                    "Original teacher label/observation hash join",
                )
        else:
            episode = episodes[trajectory["id"]]
            a.prove(
                physical,
                dict(
                    file=str(root / "evidence/stage_b/raw.h5"),
                    kind="k1_stage_b_episode_hdf",
                    group=episode["raw"]["group"],
                    states=episode["raw"]["arrays"]["states"],
                    row_steps=None,
                ),
                "Original native episode physical source",
            )
            a.prove(
                physical["terminal"],
                dict(
                    storage="last_row_of_full_state_array",
                    row=row["N"],
                    parent_key="states",
                    parent_sha256=physical["states"]["sha256"],
                    shape=[19],
                    independent_terminal_sha256=None,
                ),
                "Original N+1 endpoint provenance",
            )
            for source_key, prediction_key in (
                ("actions", "executed_actions"),
                ("observations", "observations"),
            ):
                prove_descriptor(
                    a,
                    prediction["arrays"][prediction_key],
                    episode["raw"]["arrays"][source_key],
                    "Native episode label/observation hash join",
                )
        used_files.update((physical["file"], prediction["file"]))
    a.need(used_files == set(file_map), "No unneeded or omitted original source files")


def expected_representation() -> dict:
    joints = [
        "rootx",
        "rootz",
        "rooty",
        "thigh",
        "leg",
        "foot",
        "thigh_left",
        "leg_left",
        "foot_left",
    ]
    return dict(
        physical_names=joints[1:] + [name + "_velocity" for name in joints],
        physical_columns=list(range(2, 19)),
        units=["m"] + ["rad"] * 7 + ["m/s"] * 2 + ["rad/s"] * 7,
        scales=[1.0] * 17,
        wrapped_angle_positions=list(range(1, 8)),
        rootx_column=1,
        state_dim=19,
        action_dim=6,
        window_controls=25,
        stride_controls=25,
        min_lag_controls=100,
        control_seconds=0.008,
        batch_pairs=1536,
        distance="SI-scaled root-x-free state RMS with wrapped hinge differences",
        partial_policy="Separate actual-H prefixes; never pooled with full25 matches",
    )


def verify_compact(a: MotionAudit, run: Path, result: dict, data: list[dict]) -> None:
    binding = result["motion_archive"]
    path = contained(a, run, binding["file"])
    a.prove(
        a.bind(path, binding["sha256"]),
        {key: binding[key] for key in ("sha256", "bytes")},
        "Exact compact saved-motion archive",
    )
    expected = {}
    for index, row in enumerate(data):
        for prefix, key in (
            ("states", "states"),
            ("actions", "actions"),
            ("reference", "reference_predictions"),
            ("terminal", "terminal_state"),
        ):
            if row[key] is not None:
                expected[f"{prefix}_{index:03d}"] = row[key]
    a.need(
        set(result["motion_array_bindings"]) == set(expected), "Exact compact array binding roster"
    )
    compact_bytes = 0
    with np.load(path, allow_pickle=False) as archive:
        a.need(
            set(archive.files) == set(expected),
            "Compact archive has no omitted, extra or fabricated endpoints",
        )
        for key, original in expected.items():
            value = archive[key]
            compact_bytes += value.nbytes
            a.need(
                value.dtype == original.dtype
                and value.shape == original.shape
                and np.array_equal(value, original),
                "Compact arrays exactly reproduce independently reopened originals " + key,
            )
            a.prove(
                result["motion_array_bindings"][key],
                array_binding(value),
                "Exact compact native array binding " + key,
            )
    a.work.update(compact_arrays=len(expected), compact_decoded_bytes=compact_bytes)
    a.need(
        a.work["decoded_bytes"] + compact_bytes <= 64 * 1024**2,
        "Combined independent decoded-byte ceiling",
    )


def query_batches(catalog: dict) -> list[tuple[int, int]]:
    lengths = [row["N"] for row in catalog["roster"]]
    output = []

    def append(pairs, horizon):
        for start in range(0, pairs, 1536):
            count = min(1536, pairs - start)
            output.append((count, count * horizon))

    for n in lengths:
        starts = segment_starts(n)
        pairs = sum(
            second - first >= 100 for i, first in enumerate(starts) for second in starts[i + 1 :]
        )
        append(pairs, min(25, n))
    for i, first in enumerate(lengths):
        for second in lengths[i + 1 :]:
            append(len(segment_starts(first)) * len(segment_starts(second)), min(25, first, second))
    return output


def audit_journal(a: MotionAudit, path: Path, catalog: dict, result: dict, config: dict) -> dict:
    """Recount every source/hash/decode/query request and completion in order."""
    a.bind(path)
    source_files = iter(catalog["files"])
    descriptors = source_descriptors(catalog)
    decode_order, batches = iter(descriptors), iter(query_batches(catalog))
    source_pending = dataset_pending = query_pending = None
    source_closed = datasets_closed = False
    expected = {key: 0 for key in result["work"]}
    previous_time, previous_io, events = -1.0, 0, 0
    with path.open() as stream:
        for line in stream:
            a.need(line.endswith("\n"), "Complete durable work-journal line")
            row = json.loads(line)
            event = row["event"]
            if event == "source_hash_requested":
                a.need(source_pending is None, "One source hash at a time")
                source_pending = next(source_files)
                a.prove(
                    row,
                    dict(file=source_pending["path"], bytes=source_pending["recorded_bytes"]),
                    "Original full-file hash request",
                )
            elif event == "source_hash_completed":
                a.need(source_pending is not None, "Source completion has a request")
                a.prove(
                    row,
                    dict(file=source_pending["path"], sha256=source_pending["recorded_sha256"]),
                    "Original full-file hash completion",
                )
                expected["source_hash_bytes"] += source_pending["recorded_bytes"]
                source_pending = None
            elif event == "dataset_requested":
                if not source_closed:
                    a.need(
                        next(source_files, None) is None,
                        "All source hashes precede numeric decoding",
                    )
                    source_closed = True
                a.need(
                    source_pending is None and dataset_pending is None,
                    "One declared dataset request at a time",
                )
                dataset_pending = next(decode_order)
                a.prove(
                    row,
                    dict(label=dataset_pending, expected=descriptors[dataset_pending]),
                    "Declared numeric member request",
                )
                expected["dataset_attempts"] += 1
            elif event == "dataset_completed":
                a.need(dataset_pending is not None, "Numeric completion has a request")
                a.prove(
                    row,
                    dict(label=dataset_pending, actual=a.arrays[dataset_pending]),
                    "Original member decode completion",
                )
                expected["completed_datasets"] += 1
                expected["decoded_bytes"] += a.arrays[dataset_pending]["bytes"]
                dataset_pending = None
            elif event in ("queries_requested", "queries_completed"):
                if not datasets_closed:
                    a.need(
                        next(decode_order, None) is None,
                        "All source members precede probe arithmetic",
                    )
                    datasets_closed = True
                a.need(dataset_pending is None, "Numeric reads complete before the probe")
                expected["physical_rows"] = sum(item["N"] for item in catalog["roster"])
                expected["terminal_rows"] = sum(
                    not item["terminal_missing"] for item in catalog["roster"]
                )
                if event == "queries_requested":
                    a.need(query_pending is None, "One declared query batch at a time")
                    query_pending = next(batches)
                    suffix = "requested"
                else:
                    a.need(query_pending is not None, "Query completion has a request")
                    suffix = "completed"
                pairs, aligned = query_pending
                a.prove(
                    row,
                    dict(pairs=pairs, aligned_state_pairs=aligned),
                    "Exact eligible query batch",
                )
                expected["queries_" + suffix] += pairs
                expected["aligned_" + suffix] += aligned
                if suffix == "completed":
                    query_pending = None
            else:
                raise AssertionError("Unplanned or failed work-journal event " + event)
            elapsed, io_bytes = row["wall_seconds"], row["work"]["source_io_bytes"]
            a.need(
                math.isfinite(elapsed) and previous_time <= elapsed <= result["wall_seconds"],
                "Monotone journal elapsed time",
            )
            a.need(
                type(io_bytes) is int
                and previous_io <= io_bytes <= config["budgets"]["source_io_bytes"],
                "Monotone actual decoder storage reads",
            )
            expected["source_io_bytes"] = io_bytes
            a.prove(row["work"], expected, "Independently reconstructed cumulative worker work")
            previous_time, previous_io, events = elapsed, io_bytes, events + 1
    a.need(source_pending is dataset_pending is query_pending is None, "No unclosed paid requests")
    for remaining, message in (
        (source_files, "All original files hashed"),
        (decode_order, "All selected members decoded"),
        (batches, "Every declared pair batch evaluated"),
    ):
        a.need(next(remaining, None) is None, message)
    a.prove(result["work"], expected, "Exact complete work-journal/result closure")
    return dict(events=events, work=expected)


def audit_run(a: MotionAudit, run: Path, *, config_sha: str, seal_sha: str) -> dict:
    a.bind(run / "config.json", config_sha)
    a.bind(run / "input_manifest.json", seal_sha)
    config, seal, manifest = [
        a.load(run / name) for name in ("config.json", "input_manifest.json", "manifest.json")
    ]
    source_sha = a.bind(run / "manifest.json")["sha256"]
    a.need(
        config["schema"] == "walker-motion-diagnosis-config-v1"
        and seal["schema"] == "walker-motion-diagnosis-inputs-v1",
        "Versioned frozen motion protocol/seal",
    )
    a.need(
        config_sha == seal["config_sha256"] == manifest["config_sha256"]
        and seal["source_manifest_sha256"] == source_sha,
        "Frozen config/input/source ownership",
    )
    a.prove(config["derived_tolerance"], dict(atol=ATOL, rtol=RTOL), "Frozen derived tolerances")
    a.prove(config["time_tolerance"], dict(atol=1e-10, rtol=0.0), "Frozen native clock tolerance")
    for binding in manifest["files"]:
        a.bind(contained(a, run / "source", binding["file"]), binding["sha256"])
    a.need(
        set(seal["roles"])
        == set(config["metadata"])
        == {
            "motion_roster",
            "k1_config",
            "k1_result",
            "k1_inputs",
            "k1_files",
            "k1_capsule",
            "k11_result",
        },
        "Exact seven metadata inputs",
    )
    a.need(set(seal["files"]) == set(seal["roles"].values()), "No unbound metadata aliases")
    metadata, bindings = {}, {}
    for role, name in seal["roles"].items():
        path = contained(a, run, name)
        a.need(path.is_relative_to(run / "input_metadata"), "Contained portable JSON metadata")
        bindings[role] = a.bind(path, seal["files"][name]["sha256"])
        a.prove(
            bindings[role],
            {key: config["metadata"][role][key] for key in ("sha256", "bytes")},
            "Frozen metadata identity " + role,
        )
        a.prove(bindings[role], seal["files"][name], "Complete metadata seal")
        metadata[role] = a.load(path)
    a.need(seal["budgets"] == config["budgets"], "Unchanged finite motion budgets")
    capsule, full_files = metadata["k1_capsule"], metadata["k1_files"]
    for role, name in (
        ("k1_config", "run/config.json"),
        ("k1_result", "run/result.json"),
        ("k1_inputs", "run/input_manifest.json"),
        ("k1_files", "external/full_run_file_manifest.json"),
    ):
        a.prove(
            bindings[role],
            {key: capsule["files"][name][key] for key in ("sha256", "bytes")},
            "Original K1 capsule proof " + role,
        )
    a.prove(
        bindings["k1_inputs"],
        full_files["files"]["input_manifest.json"],
        "Original whole-run input seal proof",
    )
    verify_catalog(a, metadata)
    catalog, original = metadata["motion_roster"], metadata["k1_result"]
    a.need(
        config["reference_policy"] == "reference"
        and config["reference_policy_hash"] == REFERENCE_HASH
        and config["reference_column"] == 8,
        "Fixed competent reference identity",
    )
    a.need(
        config["student_execution_roster"] == [],
        "No additional student/environment execution roster",
    )
    a.prove(
        config["new_work"],
        {
            key: 0
            for key in (
                "new_neural_rows",
                "new_physics_controls",
                "new_optimizer_steps",
                "new_mppi_controls",
                "checkpoint_loads",
                "gpu_operations",
            )
        },
        "Zero model/planner/simulator work",
    )
    a.need(
        len(catalog["files"]) == config["source_files"] == 16
        and sum(row["recorded_bytes"] for row in catalog["files"])
        == config["source_file_bytes"]
        == 1301839980,
        "Exact original full-file census",
    )
    data = load_originals(a, catalog, original)
    a.need(
        config["selected_datasets"] == 160 and config["selected_decode_bytes"] == 16331200,
        "Frozen exact selected array census",
    )
    a.prove(
        a.work,
        dict(
            source_hash_bytes=config["source_file_bytes"],
            dataset_attempts=160,
            completed_datasets=160,
            failed_datasets=0,
            decoded_bytes=16331200,
        ),
        "Independent selected source read census",
    )
    result = a.load(run / "result.json")
    a.prove(
        result,
        dict(
            schema="walker-motion-diagnosis-result-v1",
            execution_complete=True,
            error=None,
            config_sha256=config_sha,
            input_manifest_sha256=seal_sha,
            source_manifest_sha256=source_sha,
        ),
        "Complete owned saved-motion result",
    )
    a.prove(
        result["platform"],
        {key: config["platform"][key] for key in ("python", "numpy", "h5py")},
        "Pinned raw-decoder arithmetic runtime",
    )
    a.prove(
        result["array_bindings"], a.arrays, "Every original decoded member independently rehashed"
    )
    a.need(
        set(result["array_bindings"]) == set(a.arrays), "No extra or omitted source-array bindings"
    )
    verify_compact(a, run, result, data)
    summaries = metadata["k11_result"]["trajectories"]
    a.need(len(summaries) == len(data), "Complete K1.1 row provenance")
    for row, summary in zip(data, summaries, strict=True):
        a.prove(
            summary,
            {key: row[key] for key in ("id", "index", "source", "N")},
            "K1.1 exact motion identity",
        )
        errors = (row["actions"].astype(float) - row["reference_predictions"].astype(float)) ** 2
        a.equal(float(errors.mean()), summary["E"], "Independent raw/K1.1 E join")
        a.equal(
            errors.mean(axis=0), summary["per_actuator_mse"], "Independent raw/K1.1 actuator join"
        )
    expected = recompute_probe(a, data)
    a.prove(
        result,
        expected,
        "Complete independently recomputed motion curves and every eligible minimum",
    )
    a.prove(
        result["representation"],
        expected_representation(),
        "Frozen physical representation and rigid segment probe",
    )
    a.prove(
        config["expected"],
        expected["counts"],
        "All declared eligible queries and actual aligned controls",
    )
    for key in ("window_controls", "stride_controls", "min_lag_controls", "control_seconds"):
        a.prove(config[key], expected_representation()[key], "Unchanged probe parameter " + key)
    a.need(all(value == 1.0 for value in config["scales"].values()), "Common fixed SI scales")
    a.need(
        sum(row["terminal_state"] is not None for row in data) == config["terminal_rows"] == 30,
        "Thirty real endpoints, four missing",
    )
    journal = audit_journal(a, run / "work.jsonl", catalog, result, config)
    launch, exit_receipt = [
        a.load(run / name) for name in ("launch_receipt.json", "worker_exit.json")
    ]
    a.prove(
        launch,
        dict(
            schema="walker-motion-diagnosis-launch-v1",
            config_sha256=config_sha,
            input_manifest_sha256=seal_sha,
            source_manifest_sha256=source_sha,
            timeout_seconds=600,
            retries_permitted=False,
        ),
        "Frozen single-shot motion launcher",
    )
    a.prove(
        launch["command"][1:],
        [
            "-B",
            str(run / "source/scripts/train_mppi_gps.py"),
            "study",
            "motion-diagnosis",
            "worker",
            "--out",
            str(run),
        ],
        "Actual frozen source invocation",
    )
    a.prove(
        exit_receipt,
        dict(
            schema="walker-motion-diagnosis-exit-v1",
            exit_code=0,
            timeout=False,
            error=None,
            execution_complete=True,
            child_started=True,
            child_reaped=True,
            kill_attempted=False,
            cleanup_errors=[],
            result_error=None,
            result_sha256=a.bind(run / "result.json")["sha256"],
        ),
        "Clean native-free child closure",
    )
    for name, expected_lock in (
        ("launch.lock", dict(config_sha256=config_sha)),
        ("worker.lock", dict(launch_receipt_sha256=a.bind(run / "launch_receipt.json")["sha256"])),
    ):
        lock = a.load(run / name)
        a.prove(lock, expected_lock, "Immutable owner/worker lock")
        a.need(type(lock["pid"]) is int and lock["pid"] > 0, "Actual process identity")
    a.need(
        set(exit_receipt["streams"]) == {"worker.stdout", "worker.stderr"},
        "Both worker streams retained",
    )
    for name, binding in exit_receipt["streams"].items():
        a.prove(a.bind(contained(a, run, name), binding["sha256"]), binding, "Exact worker stream")
    for actual, ceiling in (
        (result["wall_seconds"], 600),
        (exit_receipt["wall_seconds"], 600),
        (result["peak_rss_bytes"], config["budgets"]["peak_rss_bytes"]),
        (result["storage_bytes_before_result"], config["budgets"]["storage_bytes"]),
    ):
        a.need(
            isinstance(actual, (float, int)) and math.isfinite(actual) and 0 <= actual <= ceiling,
            "Worker finite runtime/memory/storage bound",
        )
    retained = [path for path in run.rglob("*") if path.is_file()]
    a.need(
        all(str(path.resolve()) in a.files for path in retained), "Every retained run file bound"
    )
    storage = sum(path.stat().st_size for path in retained)
    a.need(
        storage - (run / "worker_exit.json").stat().st_size == exit_receipt["stored_bytes"]
        and storage <= config["budgets"]["storage_bytes"],
        "Exact independent closed storage census",
    )
    a.check_source_stats()
    return dict(
        counts=expected["counts"],
        work=result["work"],
        independent_work=dict(a.work),
        journal_events=journal["events"],
        closed_run_bytes=storage,
        reference_hash=REFERENCE_HASH,
        trajectories=expected["trajectories"],
        cross_pairs=expected["cross_pairs"],
        scope="Rigid fixed-SI state-window distances and conditional action/reference errors over saved trajectories only; not recurrence-period, gait-phase, task importance, teachability or coupling evidence.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--seal-sha", required=True)
    args = parser.parse_args(argv)
    run, out = args.run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        parser.error("Independent audit output must remain outside immutable evidence")
    out.mkdir(parents=True, exist_ok=False)
    audit, error, findings = MotionAudit(), None, {}
    try:
        audit.bind(Path(__file__))
        findings = audit_run(audit, run, config_sha=args.config_sha, seal_sha=args.seal_sha)
        audit.check_source_stats()
        for name, binding in list(audit.files.items()):
            if name not in audit.source_stats:
                audit.need(
                    audit.bind(Path(name)) == binding,
                    "Audited source/metadata/result bytes unchanged",
                )
        assert_independent_imports(audit)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        audit.need(
            int(rss if sys.platform == "darwin" else rss * 1024) <= 512 * 1024**2,
            "Independent audit peak RSS ceiling",
        )
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    receipt = dict(
        schema="walker-motion-diagnosis-independent-audit-v1",
        passed=error is None,
        error=error,
        checks=audit.checks,
        work=audit.work,
        new_controls=0,
        new_neural_rows=0,
        new_optimizer_steps=0,
        checkpoint_loads=0,
        peak_rss_bytes=int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
        wall_seconds=time.perf_counter() - audit.started,
    )
    for name, value in (
        ("receipt.json", receipt),
        ("findings.json", findings),
        ("file_bindings.json", audit.files),
    ):
        with (out / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(passed=error is None, checks=audit.checks, error=error)))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
