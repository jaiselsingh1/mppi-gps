"""Bound read-only motion decoding; never constructs a model or simulator."""

from __future__ import annotations

import contextlib
import hashlib
import io
from pathlib import Path
from typing import Callable

import h5py
import numpy as np


class CountedReader(io.BufferedReader):
    """Count requested storage reads, including compressed/header HDF/ZIP IO."""

    def __init__(self, path: Path, work: dict, ceiling: int):
        super().__init__(path.open("rb", buffering=0))
        self.work, self.ceiling = work, ceiling

    def charge(self, count: int) -> None:
        self.work["source_io_bytes"] += count
        if self.work["source_io_bytes"] > self.ceiling:
            raise RuntimeError("Source read-byte ceiling exhausted")

    def read(self, size=-1):
        value = super().read(size)
        self.charge(len(value))
        return value

    def readinto(self, buffer):
        count = super().readinto(buffer)
        self.charge(count)
        return count


def array_binding(value: np.ndarray) -> dict:
    return dict(
        shape=list(value.shape),
        dtype=value.dtype.str,
        sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        bytes=value.nbytes,
    )


def read_sources(catalog: dict, budgets: dict, work: dict, journal: Callable) -> tuple[list, dict]:
    """Verify sixteen complete files, then decode only the predeclared arrays."""
    before = {}
    for entry in catalog["files"]:
        path = Path(entry["path"])
        stat = path.stat()
        if not path.is_file() or path.is_symlink() or stat.st_size != entry["bytes"]:
            raise ValueError("Original source file availability/size differs: " + str(path))
        before[str(path)] = tuple(
            getattr(stat, field)
            for field in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        )
        journal(dict(event="source_hash_requested", file=str(path), bytes=stat.st_size))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                work["source_hash_bytes"] += len(block)
                if work["source_hash_bytes"] > budgets["source_hash_bytes"]:
                    raise RuntimeError("Whole-source hash-byte ceiling exhausted")
                digest.update(block)
        if digest.hexdigest() != entry["recorded_sha256"]:
            raise ValueError("Original source file SHA differs: " + str(path))
        journal(dict(event="source_hash_completed", file=str(path), sha256=digest.hexdigest()))
    decoded = {}

    def decode(owner, key, descriptor, label):
        if work["dataset_attempts"] >= budgets["dataset_attempts"]:
            journal(dict(event="dataset_rejected", label=label, reason="dataset attempt ceiling"))
            raise RuntimeError("Dataset attempt ceiling exhausted before read")
        work["dataset_attempts"] += 1
        journal(dict(event="dataset_requested", label=label, expected=descriptor))
        try:
            value = np.asarray(owner[key])
            work["decoded_bytes"] += value.nbytes
            if work["decoded_bytes"] > budgets["decoded_bytes"]:
                raise RuntimeError("Decoded-array byte ceiling exhausted")
            actual = array_binding(value)
            if (
                actual["shape"] != descriptor["shape"]
                or np.dtype(actual["dtype"]) != np.dtype(descriptor["dtype"])
                or actual["sha256"] != descriptor["sha256"]
                or not np.isfinite(value).all()
            ):
                raise ValueError("Saved array shape/dtype/SHA/finite contract differs: " + label)
            work["completed_datasets"] += 1
            decoded[label] = actual
            journal(dict(event="dataset_completed", label=label, actual=actual))
            return value
        except BaseException:
            work["failed_datasets"] += 1
            journal(dict(event="dataset_failed", label=label))
            raise

    rows = []
    with contextlib.ExitStack() as stack:
        owners = {}
        for entry in catalog["files"]:
            path = Path(entry["path"])
            stream = stack.enter_context(CountedReader(path, work, budgets["source_io_bytes"]))
            owners[str(path)] = stack.enter_context(
                np.load(stream, allow_pickle=False)
                if path.suffix == ".npz"
                else h5py.File(stream, "r")
            )
        for record in catalog["roster"]:
            name, n = record["id"], record["N"]
            physical, prediction = record["physical_source"], record["prediction_source"]
            owner = owners[physical["file"]]
            if physical["group"]:
                owner = owner[physical["group"]]
            states = decode(owner, "states", physical["states"], name + "/states")
            if physical["row_steps"] is not None:
                steps = decode(owner, "row_steps", physical["row_steps"], name + "/row_steps")
                if not np.array_equal(steps, np.arange(n)):
                    raise ValueError("Physical row ordering differs: " + name)
            terminal = physical["terminal"]
            endpoint = None
            if terminal is not None:
                if terminal["storage"] == "last_row_of_full_state_array":
                    if states.shape != (n + 1, 19) or terminal["row"] != n:
                        raise ValueError("N+1 terminal context differs")
                    endpoint = states[-1].copy()
                elif terminal["storage"] == "separate_npz_member":
                    endpoint = decode(owner, terminal["key"], terminal, name + "/terminal_state")
                else:
                    raise ValueError("Unknown terminal representation")
            source = owners[prediction["file"]][prediction["group"]]
            values = {
                key: decode(source, key, descriptor, name + "/" + key)
                for key, descriptor in prediction["arrays"].items()
            }
            pre = states[:n]
            if pre.shape != (n, 19) or not np.allclose(
                np.diff(states[:, 0]), 0.008, atol=1e-10, rtol=0
            ):
                raise ValueError("Physical state/time contract differs: " + name)
            if endpoint is not None and not np.isclose(
                endpoint[0] - pre[-1, 0], 0.008, atol=1e-10, rtol=0
            ):
                raise ValueError("Terminal state time differs: " + name)
            observations = np.concatenate((pre[:, 2:10], np.clip(pre[:, 10:19], -10, 10)), axis=1)
            actions = values["executed_actions"]
            if not np.array_equal(observations, values["observations"]) or actions.shape != (n, 6):
                raise ValueError("Physical state/conditional action row join differs: " + name)
            if np.any(np.abs(actions) > 1) or values["predictions"].shape != (n, 9, 6):
                raise ValueError("Action or saved prediction interface differs: " + name)
            rows.append(
                dict(
                    id=name,
                    index=record["index"],
                    source=record["source"],
                    ancestry=record["ancestry"],
                    states=pre.copy(),
                    actions=actions,
                    reference_predictions=values["predictions"][
                        :, prediction["reference_column"], :
                    ].copy(),
                    terminal_state=endpoint,
                )
            )
            work["physical_rows"] += len(pre)
            work["terminal_rows"] += int(endpoint is not None)
            journal(
                dict(
                    event="row_completed",
                    id=name,
                    physical_rows=len(pre),
                    terminal_rows=int(endpoint is not None),
                )
            )
    for name, state in before.items():
        stat = Path(name).lstat()
        if (
            tuple(
                getattr(stat, field)
                for field in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            )
            != state
        ):
            raise ValueError("Original file changed while decoding: " + name)
    return rows, decoded
