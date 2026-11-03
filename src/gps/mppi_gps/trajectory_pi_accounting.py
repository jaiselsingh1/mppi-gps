"""Durable finite work reservations for full-trajectory PI workers and fitting."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .provenance import default, write


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class Work:
    """Charge before each external call, retaining uncertain work on failure.

    Nested reservations intentionally charge different units: a model forecast
    owns controls/substeps; policy evaluation owns rows. They are never summed
    into an invented single compute number. Physical disk I/O is unknown.
    """

    def __init__(self, directory: Path, limits: dict[str, int], *, filename="ledger.jsonl"):
        self.directory, self.limits = Path(directory), limits
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / filename).open("x")
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.counts: dict[str, int] = {}
        self.sequence = 0
        self.closed = False

    def record(self, event: str, **fields: Any) -> None:
        if self.closed:
            raise RuntimeError("Work ledger is closed")
        if {"id", "event", "wall_seconds"} & fields.keys():
            raise ValueError("Reserved ledger fields")
        row = dict(
            id=self.sequence, event=event, wall_seconds=time.monotonic() - self.started, **fields
        )
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False, default=default) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.sequence += 1

    def check(self, *, reserve_output: int = 0) -> None:
        if time.monotonic() - self.started > self.limits["wall_seconds"]:
            raise RuntimeError("Wall budget exhausted")
        if directory_bytes(self.directory) + reserve_output > self.limits["output_bytes"]:
            raise RuntimeError("Output budget exhausted")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == "darwin" else 1024
        if peak > self.limits["memory_bytes"]:
            raise RuntimeError("Process-lifetime peak RSS budget exhausted")

    def call(self, name: str, charges: dict[str, int], function: Callable) -> Any:
        self.check()
        for key, value in charges.items():
            if type(value) is not int or value < 0:
                raise ValueError("Explicit nonnegative integer reservations required")
            if self.counts.get("requested_" + key, 0) + value > self.limits[key]:
                self.record("rejected", operation=name, charges=charges)
                raise RuntimeError("Work budget exhausted: " + key)
        for key, value in charges.items():
            field = "requested_" + key
            self.counts[field] = self.counts.get(field, 0) + value
        token = self.sequence
        self.record("requested", operation=name, charges=charges)
        entered = time.monotonic()
        try:
            result = function()
        except BaseException as error:
            for key, value in charges.items():
                field = "unknown_partial_" + key
                self.counts[field] = self.counts.get(field, 0) + value
            self.record(
                "failed",
                request_id=token,
                operation=name,
                charges=charges,
                error=repr(error),
                call_seconds=time.monotonic() - entered,
            )
            raise
        for key, value in charges.items():
            field = "completed_" + key
            self.counts[field] = self.counts.get(field, 0) + value
        self.record(
            "completed",
            request_id=token,
            operation=name,
            charges=charges,
            call_seconds=time.monotonic() - entered,
        )
        self.check()
        return result

    def measure(self, kind: str, amount: int, function: Callable, **metadata: Any) -> Any:
        """Compatibility with existing Policy/fit_policy metering interface."""
        return self.call(kind, {kind: amount}, function)

    def receipt(self) -> dict:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(
            counts=dict(self.counts),
            wall_seconds=time.monotonic() - self.started,
            cpu_seconds=time.process_time() - self.cpu_started,
            peak_process_rss_bytes=peak * (1 if sys.platform == "darwin" else 1024),
            output_bytes_before_receipt=directory_bytes(self.directory),
            physical_disk_io_bytes=None,
        )

    def close(self, status: str, **fields: Any) -> dict:
        if status == "COMPLETED":
            self.check(reserve_output=32 * 1024)
        value = dict(status=status, **self.receipt(), **fields)
        self.record("closed", receipt=value)
        self.stream.close()
        self.closed = True
        write(self.directory / "receipt.json", value)
        return value
