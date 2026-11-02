"""Finite H5 native boundaries; historical Meter/HDF contracts are untouched."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .proposal_budget import ARMS, ReservationLedger, census
from .proposal_evidence import BinaryEvidenceMeter
from .proposal_storage import (
    CONTRACT,
    QUOTAS,
    bounded_json_payload,
    episode_payload_bound,
    live_query_payload_bound,
    model_query_payload_bound,
    storage_census,
)
from .runtime import BudgetExhausted, _dumps


def verify_runtime(config: dict[str, Any]) -> None:
    """Check metadata and pinning before importing an inference/simulator stack."""
    runtime = config["runtime"]
    actual = dict(host=platform.node(), python=sys.version, platform=platform.platform())
    for key, value in actual.items():
        if value != runtime[key]:
            raise ValueError("Frozen runtime changed: " + key)
    for name, version in runtime["distributions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen distribution changed: " + name)
    if sorted(os.sched_getaffinity(0)) != runtime["cpu_affinity"]:
        raise ValueError("Worker CPU affinity is not pinned to its frozen allocation")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(key) != runtime[key]:
            raise ValueError("Frozen native environment changed: " + key)


class NativeMeter(BinaryEvidenceMeter):
    """Pre-call work and evidence reservations plus per-file byte ceilings.

    Array, index and ledger caps are explicit. A whole episode's final evidence
    allowance stays reserved during its live controls. Native output allowance
    is checked before entering each query, not after it has been paid for.
    """

    def __init__(self, out: Path, config: dict[str, Any]):
        self.config = config
        work = census(config)
        if not storage_census(config)["fits"]:
            raise ValueError("The complete serialized roster exceeds its frozen evidence envelope")
        budgets = config["budgets"]
        self.deadline = time.monotonic() + budgets["worker_wall_seconds"]
        self.held_episode_bytes = 0
        self._operation_serial = 0
        self._episode_serial = 0
        self._initializing = True
        self.native_validation_failed = False
        self.category_limits = {name: QUOTAS[name] for name in ("payload", "index", "ledger")}
        limits = dict(
            requested_controls=budgets["requested_controls"],
            model_controls=budgets["model_controls"],
            live_controls=budgets["live_controls"],
            physics_substeps=budgets["physics_substeps"],
            model_queries=work["total"]["model_queries"],
            live_queries=budgets["live_controls"],
            neural_rows=budgets["neural_rows"],
            prediction_queries=budgets["worker_neural_queries"],
            backward_rows=budgets["backward_rows"],
            optimizer_steps=budgets["optimizer_steps"],
            checkpoint_loads=budgets["checkpoint_loads"],
            episodes=budgets["episodes"],
        )
        arm_census = work["per_arm"]
        arm_limits = dict(
            requested_controls=arm_census["model_controls"] + 120000,
            model_controls=arm_census["model_controls"],
            live_controls=120000,
            physics_substeps=(arm_census["model_controls"] + 120000) * 4,
            model_queries=108036,
            live_queries=120000,
            neural_rows=arm_census["worker_neural_rows"],
            prediction_queries=162078,
            backward_rows=arm_census["child_backward_rows"],
            optimizer_steps=6000,
            checkpoint_loads=3,
            episodes=120,
        )
        self.reservations = ReservationLedger(
            limits,
            per_arm_limits={arm: arm_limits for arm in ARMS},
            wall_seconds=budgets["worker_wall_seconds"],
            retain_events=False,
        )
        super().__init__(
            out,
            max_controls=budgets["requested_controls"],
            max_neural_rows=budgets["neural_rows"],
            max_bytes=budgets["storage_bytes"],
            reserved_bytes=QUOTAS["external"],
            max_event_bytes=CONTRACT["max_event_line_bytes"],
            max_header_bytes=CONTRACT["max_index_line_bytes"],
            max_array_bytes=CONTRACT["max_array_bytes"],
            max_arrays_per_group=CONTRACT["max_arrays_per_group"],
        )
        self._initializing = False

    def _check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExhausted("H5 worker deadline reached; external watchdog owns long calls")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = int(rss if sys.platform == "darwin" else rss * 1024)
        if rss > self.config["budgets"]["peak_rss_bytes"]:
            raise BudgetExhausted("H5 worker resident-memory ceiling exhausted")

    def reserve_external(self, amount):
        """Fund small frozen-source/checkpoint/curve files before creating them."""
        self._check()
        total = sum(
            path.stat().st_size
            for path in self.out.parent.rglob("*")
            if path.is_file() and not path.is_relative_to(self.out)
        )
        if total + amount > QUOTAS["external"]:
            raise BudgetExhausted("Non-evidence files cannot fit their frozen16GiB allowance")

    def _reserve(self, amount, *, emergency=False):
        if not self._initializing and not emergency and not getattr(self, "_recovering", False):
            self._check()
        return super()._reserve(
            amount + (0 if emergency else self.held_episode_bytes), emergency=emergency
        )

    def _append(self, stream, payload, *, sync=True):
        kind = "payload" if stream is self._raw else "index" if stream is self._index else "ledger"
        if stream.tell() + len(payload) > self.category_limits[kind]:
            raise BudgetExhausted("H5 " + kind + " category ceiling exhausted before write")
        return super()._append(stream, payload, sync=sync)

    def record(self, event, **fields):
        if self._sequence >= 5_000_000 and not (
            event.endswith("_error") or event == "runtime_close"
        ):
            raise BudgetExhausted("H5 fixed ledger-record census exhausted")
        return super().record(event, **fields)

    def save_arrays(self, tag, **arrays):
        if tag == "worker_failure_context" or (
            tag.startswith("episode_") and tag.endswith("_partial")
        ):
            self.held_episode_bytes = 0
            self._recovering = True
            try:
                return super().save_arrays(tag, **arrays)
            finally:
                self._recovering = False
        if self._array_sequence >= 2_000_000:
            raise BudgetExhausted("H5 fixed array-group census exhausted")
        if tag.startswith("episode_"):
            self.held_episode_bytes = 0
        for name in ("protected_before", "proposal_before", "protected_after", "proposal_after"):
            if name in arrays:
                bounded_json_payload(
                    _dumps(arrays[name]),
                    text_bytes=CONTRACT["private_planner_text_bytes"],
                    purpose="Private planner JSON",
                )
        if tag.endswith("_result") and "info" in arrays:
            bounded_json_payload(
                _dumps(arrays["info"]),
                text_bytes=CONTRACT["native_info_text_bytes"],
                purpose="Native step info",
            )
        if tag.startswith("episode_") and "infos" in arrays:
            for row in arrays["infos"]["rows"]:
                bounded_json_payload(
                    _dumps(row),
                    text_bytes=CONTRACT["episode_row_text_bytes"],
                    purpose="Episode controller/native info",
                )
        return super().save_arrays(tag, **arrays)

    def _ticket(self, name, work, operation):
        if self.native_validation_failed:
            raise RuntimeError("Native validation failed; no further paid work is permitted")
        self._check()
        self._operation_serial += 1
        arm = self.context.get("arm")
        if arm not in ARMS:
            arm = None
        return self.reservations.run(
            f"{name}_{self._operation_serial}",
            work,
            operation,
            phase="worker",
            arm=arm,
        )

    def _future_metadata(self, groups, events):
        if (
            self._array_sequence + groups > 2_000_000
            or self._sequence + events > 5_000_000
            or self._index.tell() + groups * 2 * 8192 > self.category_limits["index"]
            or self._ledger.tell() + events * 16384 > self.category_limits["ledger"]
        ):
            raise BudgetExhausted("Paid operation cannot reserve its bounded metadata outputs")

    def _query(self, mode, controls, frame_skip, inputs, fn, pack):
        self._check()
        if mode not in ("live", "model") or frame_skip != 4:
            raise ValueError("Only frozen native Walker query modes are permitted")
        input_bytes = sum(np.asarray(value).nbytes for value in inputs.values())
        if mode == "model":
            actions = np.asarray(inputs["actions"])
            if actions.shape[1:] != (48, 6) or len(actions) not in (1, 256):
                raise ValueError("Unbudgeted H5 native model batch shape")
            # 19 float64 post-state entries/control, score per trajectory and
            # bounded terminal-role scalar; zero sensors in the frozen XML.
            payload_bound = model_query_payload_bound(controls, len(actions))
        else:
            if controls != 1:
                raise ValueError("One native live control is required")
            payload_bound = live_query_payload_bound()
        output_bytes = payload_bound - input_bytes
        if output_bytes < 0:
            raise ValueError("Native input payload exceeds its complete query byte reservation")
        reserve = input_bytes + output_bytes + 4 * 8192 + 3 * 16384
        self._future_metadata(2, 3)
        self._reserve(reserve)
        if (
            self._raw.tell() + input_bytes + output_bytes + self.held_episode_bytes
            > self.category_limits["payload"]
        ):
            raise BudgetExhausted("Native output payload does not fit before the paid query")
        result = self._ticket(
            "native_" + mode,
            dict(
                requested_controls=controls,
                **{mode + "_controls": controls},
                physics_substeps=controls * frame_skip,
                **{mode + "_queries": 1},
            ),
            lambda: super(NativeMeter, self)._query(mode, controls, frame_skip, inputs, fn, pack),
        )
        if mode == "model":
            try:
                states, scores, sensors = result
                valid = (
                    np.shape(states) == (len(actions), 48, 19)
                    and np.shape(scores) == (len(actions),)
                    and np.shape(sensors) == (len(actions), 48, 0)
                )
                if len(actions) == 1:
                    valid = valid and np.isfinite(states).all() and np.isfinite(scores).all()
            except (TypeError, ValueError):
                valid = False
            if not valid:
                self.record(
                    "h5_native_validation_error",
                    mode=mode,
                    reason="native model return violates the frozen interface/mean contract",
                    **self._fields({}),
                )
                self.native_validation_failed = True
                raise RuntimeError("Invalid native model output preserved; stop before more work")
        return result

    def measure(self, kind, amount, fn, **metadata):
        self._check()
        if kind not in ("neural_rows", "backward_rows", "optimizer_steps", "checkpoint_loads"):
            raise ValueError("Unbudgeted H5 measurement kind: " + kind)
        work = {kind: amount}
        if kind == "neural_rows":
            work["prediction_queries"] = 1
            # Fund the resulting six-action prediction, support and score data
            # before inference. The actual group bytes remain counted on write.
            self._reserve(amount * 64 + 4 * 8192 + 4 * 16384)
            self._future_metadata(2, 8)
            if (
                self._raw.tell() + amount * 64 + self.held_episode_bytes
                > self.category_limits["payload"]
            ):
                raise BudgetExhausted("Prediction persistence cannot fit before inference")
        return self._ticket(
            kind, work, lambda: super(NativeMeter, self).measure(kind, amount, fn, **metadata)
        )

    def run_episode(self, operation):
        if self.held_episode_bytes:
            raise ValueError("Nested native episodes are forbidden")
        held = max(32 * 1024**2, episode_payload_bound(1000) + 2 * 8192 + 2 * 16384)
        self._reserve(held)
        self.held_episode_bytes = held
        try:
            return self._ticket("episode", dict(episodes=1), operation)
        finally:
            self.held_episode_bytes = 0

    def totals(self):
        result = super().totals()
        reservation = self.reservations.snapshot(include_events=False)
        result["reservation_integrity"] = dict(
            stopped=reservation["stopped"],
            errors=reservation["errors"],
            pending=reservation["pending"],
            native_validation_failed=self.native_validation_failed,
        )
        return result
