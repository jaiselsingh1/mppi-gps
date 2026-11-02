"""C1 finite work boundaries, independent of the planner and old studies."""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .consequence_storage import CompactEvidenceMeter
from .runtime import BudgetExhausted


class ConsequenceMeter(CompactEvidenceMeter):
    """Reserve each operation before paying it; retain uncertain failures.

    Native intents/completions are written by the durable trace recorder. This
    owner charges their work without duplicating its journal records.
    """

    def __init__(
        self,
        out: Path,
        *,
        limits: dict[str, int],
        wall_seconds: float,
        max_bytes: int,
        rss_bytes: int,
        prior: dict[str, int] | None = None,
    ):
        self.limits = dict(limits)
        self.work = dict(prior or {})
        self.completed_work: dict[str, int] = {}
        self.unknown_work: dict[str, int] = {}
        self.returned_unpersisted: dict[str, int] = {}
        self.deadline = time.monotonic() + wall_seconds
        self.rss_bytes = rss_bytes
        self._serial = 0
        self.after_prediction: Callable[[Any], None] | None = None
        super().__init__(
            out,
            max_controls=limits.get("native_controls", 0),
            max_neural_rows=limits.get("neural_rows", 0),
            max_bytes=max_bytes,
            reserved_bytes=512 * 1024**2,
        )

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise BudgetExhausted("C1 worker deadline reached")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(peak if sys.platform == "darwin" else peak * 1024)
        if peak > self.rss_bytes:
            raise BudgetExhausted("C1 resident-memory limit reached")

    def reserve_work(self, kind: str, amount: int = 1) -> None:
        self.authorize({kind: amount})
        self.charge({kind: amount})

    def authorize(self, work: dict[str, int]) -> None:
        """Atomically check every category; a rejection consumes no paid work."""
        self.check()
        for kind, amount in work.items():
            if type(amount) is not int or amount < 0 or kind not in self.limits:
                raise ValueError("Undeclared C1 work category/amount: " + kind)
            if self.work.get(kind, 0) + amount > self.limits[kind]:
                raise BudgetExhausted("C1 work ceiling reached: " + kind)

    def charge(self, work: dict[str, int]) -> None:
        """Charge an already authorized, durably journaled call immediately before entry."""
        for kind, amount in work.items():
            self.work[kind] = self.work.get(kind, 0) + amount

    def returned(self, kind: str, amount: int) -> None:
        self.completed_work[kind] = self.completed_work.get(kind, 0) + amount

    def uncertain(self, kind: str, amount: int) -> None:
        self.unknown_work[kind] = self.unknown_work.get(kind, 0) + amount

    def measure(
        self, kind: str, amount: int, fn: Callable[[], Any], *, purpose: str = "", **_: Any
    ) -> Any:
        work = {kind: amount}
        if kind == "neural_rows":
            work["prediction_calls"] = 1
        self.authorize(work)
        self._serial += 1
        operation = self._serial
        self.record(
            "work_requested",
            category=kind,
            amount=amount,
            operation=operation,
            purpose=purpose[:80],
        )
        self.charge(work)
        returned = False
        try:
            value = fn()
            returned = True
            self.returned(kind, amount)
            if kind == "neural_rows":
                self.returned("prediction_calls", 1)
            if kind == "neural_rows" and self.after_prediction is not None:
                self.after_prediction(value)
            self.record("work_complete", category=kind, amount=amount, operation=operation)
        except BaseException:
            if returned:
                self.returned_unpersisted[kind] = self.returned_unpersisted.get(kind, 0) + amount
            else:
                self.uncertain(kind, amount)
                if kind == "neural_rows":
                    self.uncertain("prediction_calls", 1)
            try:
                self.record(
                    "work_error",
                    category=kind,
                    amount=amount,
                    operation=operation,
                    returned=returned,
                )
            except BaseException:
                pass
            raise
        return value

    def reserve_external(self, amount: int) -> None:
        self.check()
        actual = sum(
            p.stat().st_size
            for p in self.out.parent.rglob("*")
            if p.is_file() and not p.is_relative_to(self.out)
        )
        if actual + amount > 512 * 1024**2:
            raise BudgetExhausted("C1 external worker files exceed their reservation")

    def _reserve(self, amount: int, *, emergency: bool = False) -> None:
        if not emergency:
            self.check()
        return super()._reserve(amount, emergency=emergency)

    def record(self, event: str, **fields: Any) -> str:
        if getattr(self, "_sequence", 0) >= 5_000_000:
            raise BudgetExhausted("C1 journal record ceiling reached")
        return super().record(event, **fields)

    def save_arrays(self, tag: str, **arrays: Any) -> dict[str, Any]:
        if self._array_sequence >= 13_468:
            raise BudgetExhausted("C1 array group ceiling reached")
        return super().save_arrays(tag, **arrays)

    def accounting(self) -> dict[str, Any]:
        return dict(
            requested=self.work,
            completed=self.completed_work,
            unknown_partial=self.unknown_work,
            returned_unpersisted=self.returned_unpersisted,
            storage=self.totals(),
        )

    def totals(self) -> dict[str, Any]:
        result = super().totals()
        for work, name in (
            ("native_controls", "controls"),
            ("physics_substeps", "physics_substeps"),
            ("neural_rows", "neural_rows"),
            ("backward_rows", "backward_rows"),
            ("optimizer_steps", "optimizer_steps"),
        ):
            for prefix, values in (
                ("requested", self.work),
                ("completed", self.completed_work),
                ("unknown_partial", self.unknown_work),
            ):
                result[f"{prefix}_{name}"] = values.get(work, 0)
        for prefix, values in (("requested", self.work), ("completed", self.completed_work)):
            result[f"{prefix}_live_controls"] = values.get("native_controls", 0)
        result.update(
            resets=self.completed_work.get("resets", 0),
            restores=self.completed_work.get("restores", 0),
            snapshots=self.completed_work.get("snapshot_reads", 0),
            live_queries=self.work.get("native_controls", 0),
        )
        return result
