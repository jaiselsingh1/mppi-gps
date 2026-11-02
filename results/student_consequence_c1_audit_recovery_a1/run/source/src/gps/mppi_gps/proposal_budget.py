"""Exact H5.0 work census and fail-closed pre-call reservations.

Only the standard library is imported. This is an accounting boundary, not a
simulator, policy runner, wall-time watchdog or proof of HDF5 storage usage.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeVar

ARMS = ("independent", "selector_only", "proposal_guided")
COUNTERS = frozenset(
    {
        "requested_controls",
        "model_controls",
        "live_controls",
        "physics_substeps",
        "neural_rows",
        "backward_rows",
        "optimizer_steps",
        "model_queries",
        "live_queries",
        "prediction_queries",
        "checkpoint_loads",
        "episodes",
        "seconds",
        "storage_bytes",
        "imports",
    }
)
T = TypeVar("T")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an explicit integer >= {minimum}")
    return value


def _fixed(recipe: Mapping[str, Any]) -> None:
    """Reject a changed experiment instead of silently recosting it as H5.0."""
    if recipe.get("schema") != "walker-teacher-proposal-h5.0-r1":
        raise ValueError("Expected the H5.0 round-one recipe")
    settings = {
        ("round",): 1,
        ("mppi", "lam"): 3.0,
        ("mppi", "noise_sigma"): 0.3,
        ("mppi", "noise_lowpass_cutoff_hz"): 23.4375,
        ("mppi", "noise_lowpass_sample_rate_hz"): 125.0,
        ("mppi", "noise_lowpass_order"): 2,
        ("mppi", "use_is_correction"): False,
        ("mppi", "noise_std"): None,
        ("mppi", "noise_cov"): None,
        ("mppi", "lam_target_neff_fraction"): None,
        ("guidance", "alpha"): 0.1199341983507337,
        ("guidance", "eta_u"): 0.01,
        ("guidance", "temperature"): 3.0,
        ("guidance", "support_radius"): None,
        ("policy", "observation_dim"): 17,
        ("policy", "hidden"): 128,
        ("policy", "hidden_layers"): 2,
        ("policy", "action_dim"): 6,
        ("policy", "dtype"): "float32",
        ("policy", "device"): "cpu",
        ("policy", "torch_threads"): 1,
        ("learning", "lr"): 0.0003,
        ("learning", "eta_theta"): 0.0,
        ("learning", "prox_reduction"): "mean",
        ("learning", "anchor_refresh"): 0,
        ("learning", "gradient_clip"): 0.5,
        ("learning", "child_resume_fit"): False,
        ("learning", "normalization_floor"): 0.05,
        ("learning", "global_torch_seed"): 2791000000,
        ("native_guard", "relative_cost_tolerance"): 0.05,
        ("native_guard", "absolute_atol"): 1e-12,
    }
    for path, expected in settings.items():
        value: Any = recipe
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise ValueError("Missing H5.0 fixed setting: " + ".".join(path))
            value = value[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError("Version the changed fixed setting: " + ".".join(path))
    if recipe.get("arms") != list(ARMS) or len(recipe.get("streams", [])) != 3:
        raise ValueError("H5.0 requires its three ordered arms and three paired streams")
    paths = {
        ("episode_steps",): 1000,
        ("teacher_episodes_per_stream_arm",): 4,
        ("gap_episodes_per_stream_arm",): 2,
        ("task", "frame_skip"): 4,
        ("mppi", "K"): 256,
        ("mppi", "H"): 48,
        ("mppi", "initial_iterations"): 2,
        ("learning", "parent_updates"): 2000,
        ("learning", "child_updates"): 2000,
        ("learning", "batch_size"): 256,
        ("learning", "parent_corpus_rows"): 4000,
        ("learning", "child_corpus_rows"): 4000,
        ("learning", "held_rows"): 2000,
        ("learning", "prediction_batch_size"): 1024,
    }
    for path, expected in paths.items():
        value: Any = recipe
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise ValueError("Missing H5.0 census field: " + ".".join(path))
            value = value[key]
        if _integer(value, ".".join(path)) != expected:
            raise ValueError("Version the changed census field: " + ".".join(path))
    if (
        len(recipe.get("primary_seeds", [])) != 16
        or len(recipe.get("confirmation_seeds", [])) != 16
    ):
        raise ValueError("Both fixed sixteen-reset panels are required")
    reset_seeds = list(recipe["primary_seeds"]) + list(recipe["confirmation_seeds"])
    for ordinal, stream in enumerate(recipe["streams"]):
        if stream.get("id") != ordinal:
            raise ValueError("Stream IDs must retain their canonical order")
        for name, count in (
            ("acquisition_seeds", 4),
            ("gap_seeds", 2),
            ("protected_planner_seeds", 4),
            ("second_planner_seeds", 4),
            ("gap_protected_planner_seeds", 2),
            ("gap_second_planner_seeds", 2),
        ):
            if len(stream.get(name, [])) != count:
                raise ValueError("Missing fixed stream seed roster: " + name)
            for seed in stream[name]:
                _integer(seed, name)
        reset_seeds += stream["acquisition_seeds"] + stream["gap_seeds"]
    if len(set(reset_seeds)) != len(reset_seeds):
        raise ValueError("Acquisition, gap, primary and confirmation resets must be disjoint")


def teacher_decision_bounds(recipe: Mapping[str, Any], control_index: int) -> dict[str, int]:
    """Maximum model/actor work for one planning decision, excluding its live step.

    I and S run the inner-guidance observer and I discards guided selection.
    Thus all arms reserve the same two producer and two selector mean queries.
    """
    _fixed(recipe)
    control_index = _integer(control_index, "control_index")
    if control_index >= recipe["episode_steps"]:
        raise ValueError("Control index lies outside the frozen episode")
    k, h = recipe["mppi"]["K"], recipe["mppi"]["H"]
    passes = recipe["mppi"]["initial_iterations"] if control_index == 0 else 1
    model = (2 * passes * k + 4) * h
    return dict(
        model_controls=model,
        requested_controls=model,
        physics_substeps=model * recipe["task"]["frame_skip"],
        neural_rows=(passes * k + 2 * (k + 1) + 1) * h,
        model_queries=2 * passes + 4,
        prediction_queries=passes + 2,
    )


def census(recipe: Mapping[str, Any], *, validate_recorded: bool = True) -> dict[str, Any]:
    """Independently derive complete roster maxima, not realized paid work.

    Strong audit re-evaluates each saved prediction input once and never fits,
    steps physics or repeats training forwards. Its extra rows are explicit.
    """
    _fixed(recipe)
    steps, streams, arms = recipe["episode_steps"], len(recipe["streams"]), len(ARMS)
    k, h = recipe["mppi"]["K"], recipe["mppi"]["H"]
    initial = recipe["mppi"]["initial_iterations"]
    passes = steps - 1 + initial
    per_stream_teacher = (
        recipe["teacher_episodes_per_stream_arm"] + recipe["gap_episodes_per_stream_arm"]
    )
    teacher_per_arm = streams * per_stream_teacher
    teacher_total = teacher_per_arm * arms
    model_episode = (2 * passes * k + 4 * steps) * h
    guidance_episode = (passes * k + (2 * (k + 1) + 1) * steps) * h
    model_queries_episode = 2 * passes + 4 * steps
    panels = len(recipe["primary_seeds"]) + len(recipe["confirmation_seeds"])
    parent_updates = streams * recipe["learning"]["parent_updates"]
    child_updates_arm = streams * recipe["learning"]["child_updates"]
    batch = recipe["learning"]["batch_size"]
    train, held = recipe["learning"]["parent_corpus_rows"], recipe["learning"]["held_rows"]
    child_rows = recipe["learning"]["child_corpus_rows"]
    parent_diagnostic = streams * 2 * (train + held)
    child_diagnostic_arm = streams * (2 * child_rows + train + held)
    parent_direct = streams * panels * steps
    child_direct_arm = streams * (panels + recipe["gap_episodes_per_stream_arm"]) * steps
    shared = dict(
        optimizer_updates=parent_updates,
        backward_rows=parent_updates * batch,
        diagnostic_forward_rows=parent_diagnostic,
        standalone_controls=parent_direct,
        neural_rows=parent_updates * batch + parent_diagnostic + parent_direct,
    )
    arm = dict(
        teacher_episodes=teacher_per_arm,
        model_controls=model_episode * teacher_per_arm,
        guidance_forward_rows=guidance_episode * teacher_per_arm,
        child_optimizer_updates=child_updates_arm,
        child_backward_rows=child_updates_arm * batch,
        child_diagnostic_forward_rows=child_diagnostic_arm,
        child_standalone_controls=child_direct_arm,
        teacher_posthoc_forward_rows=teacher_per_arm * steps,
    )
    arm["worker_neural_rows"] = sum(
        arm[key]
        for key in (
            "guidance_forward_rows",
            "child_backward_rows",
            "child_diagnostic_forward_rows",
            "child_standalone_controls",
            "teacher_posthoc_forward_rows",
        )
    )
    audit_arm_neural = arm["worker_neural_rows"] - arm["child_backward_rows"]
    live = teacher_total * steps + parent_direct + arms * child_direct_arm
    models = arms * arm["model_controls"]
    updates = parent_updates + arms * child_updates_arm
    diagnostics = parent_diagnostic + arms * child_diagnostic_arm
    worker_neural = arms * arm["worker_neural_rows"] + shared["neural_rows"]
    strong_audit_neural = worker_neural - updates * batch
    episode = dict(
        teacher_live_decisions=steps,
        optimization_passes_per_stream=passes,
        sampled_model_calls=2 * passes,
        stream_mean_model_calls=2 * steps,
        selector_mean_model_calls=2 * steps,
        model_queries=model_queries_episode,
        model_controls=model_episode,
        guidance_forward_rows=guidance_episode,
    )
    total = dict(
        teacher_decisions=teacher_total * steps,
        model_queries=teacher_total * model_queries_episode,
        model_controls=models,
        live_controls=live,
        physics_substeps=(models + live) * recipe["task"]["frame_skip"],
        guidance_forward_rows=teacher_total * guidance_episode,
        training_forward_and_backward_rows_each=updates * batch,
        diagnostic_forward_rows=diagnostics,
        worker_neural_rows=worker_neural,
    )
    output = dict(
        per_episode=episode,
        per_arm=arm,
        shared_parent_prefix=shared,
        total=total,
        strong_audit_neural_rows=strong_audit_neural,
        per_arm_strong_audit_neural_rows=audit_arm_neural,
        per_arm_combined_neural_rows=arm["worker_neural_rows"] + audit_arm_neural,
        shared_parent_strong_audit_neural_rows=shared["neural_rows"] - shared["backward_rows"],
        shared_parent_combined_neural_rows=2 * shared["neural_rows"] - shared["backward_rows"],
        combined_neural_rows=worker_neural + strong_audit_neural,
        worker_optimizer_steps=updates,
        worker_backward_rows=updates * batch,
        worker_requested_controls=models + live,
        teacher_live_controls=teacher_total * steps,
        standalone_live_controls=parent_direct + arms * child_direct_arm,
        worker_episodes=teacher_total
        + streams * panels
        + arms * streams * (panels + recipe["gap_episodes_per_stream_arm"]),
        checkpoint_loads=streams * (1 + arms),
        per_decision_first=teacher_decision_bounds(recipe, 0),
        per_decision_later=teacher_decision_bounds(recipe, 1),
        scope="Strict roster reservations, not observations or HDF5 storage proof",
    )
    if validate_recorded:
        recorded = recipe.get("call_census", {})
        for section in ("per_episode", "per_arm", "shared_parent_prefix", "total"):
            for key, expected in output[section].items():
                if recorded.get(section, {}).get(key) != expected:
                    raise ValueError(f"Recorded H5.0 census differs: {section}.{key}")
        checks = {
            "model_controls": models,
            "live_controls": live,
            "requested_controls": models + live,
            "physics_substeps": total["physics_substeps"],
            "neural_rows": worker_neural,
            "backward_rows": updates * batch,
            "optimizer_steps": updates,
            "checkpoint_loads": output["checkpoint_loads"],
            "episodes": output["worker_episodes"],
            "audit_neural_rows": strong_audit_neural,
            "total_worker_and_audit_neural_rows": worker_neural + strong_audit_neural,
        }
        for key, expected in checks.items():
            if recipe.get("budgets", {}).get(key) != expected:
                raise ValueError("Recorded H5.0 budget differs: " + key)
        if recipe["budgets"]["per_arm_model_control_ceiling"] < arm["model_controls"]:
            raise ValueError("Per-arm model ceiling cannot cover the fixed roster")
        if recipe["budgets"]["per_arm_neural_row_ceiling"] < output["per_arm_combined_neural_rows"]:
            raise ValueError("Per-arm neural ceiling cannot cover worker plus strong audit")
        if recorded.get("audit", {}).get("mode") != "full_saved_prediction_recomputation":
            raise ValueError("The full saved-prediction audit mode must be frozen")
        if recorded["audit"].get("neural_rows") != strong_audit_neural:
            raise ValueError("The strong audit's neural work must be explicitly reserved")
    return output


class ReservationError(RuntimeError):
    """No external work may begin after this boundary rejects a request."""


def _work(value: Mapping[str, int]) -> dict[str, int]:
    result = dict(value)
    if set(result) - COUNTERS:
        raise ValueError("Unknown work counters: " + ",".join(sorted(set(result) - COUNTERS)))
    for key, amount in result.items():
        _integer(amount, key)
    return result


@dataclass(frozen=True)
class Reservation:
    identifier: str
    name: str
    arm: str | None
    phase: str


class ReservationLedger:
    """Atomic pre-call ceilings with irreversible reservation ownership.

    A raised external call retains its full reservation as unknown partial work
    and stops the ledger. Rejected calls are never invoked. Scope and counters
    are explicit; no retry/reset/refund API exists. ``seconds`` is a caller's
    integral time reservation, while elapsed wall time is measured separately.
    Boundary polling requires an outer watchdog for interrupting a long call.
    """

    def __init__(
        self,
        limits: Mapping[str, int],
        *,
        per_arm_limits: Mapping[str, Mapping[str, int]] | None = None,
        phase_limits: Mapping[str, Mapping[str, int]] | None = None,
        wall_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
        retain_events: bool = True,
    ) -> None:
        self._limits = {"total": _work(limits)}
        if not self._limits["total"]:
            raise ValueError("A nonempty explicit total work budget is required")
        self._arms = dict(per_arm_limits or {})
        self._phases = dict(phase_limits or {})
        for prefix, scopes in (("arm", self._arms), ("phase", self._phases)):
            for name, values in scopes.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("Scope names must be nonempty strings")
                self._limits[prefix + ":" + name] = _work(values)
        if wall_seconds is not None and (
            type(wall_seconds) not in (int, float)
            or not math.isfinite(wall_seconds)
            or wall_seconds < 0
        ):
            raise ValueError("wall_seconds must be finite and nonnegative")
        self._clock, self._started, self._wall_seconds = clock, clock(), wall_seconds
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        if type(retain_events) is not bool:
            raise TypeError("retain_events must be a boolean")
        self._event_sink, self._retain_events = event_sink, retain_events
        self._requested = {
            scope: dict.fromkeys(values, 0) for scope, values in self._limits.items()
        }
        self._completed = deepcopy(self._requested)
        self._unknown = deepcopy(self._requested)
        self._pending: dict[str, tuple[Reservation, dict[str, int], tuple[str, ...]]] = {}
        self._names: set[str] = set()
        self._events: list[dict[str, Any]] = []
        self._stopped = False
        self._errors = dict(input_errors=0, budget_errors=0, import_errors=0, external_errors=0)

    def _emit(self, event: dict[str, Any]) -> None:
        if self._retain_events:
            self._events.append(deepcopy(event))
        if self._event_sink is not None:
            try:
                self._event_sink(deepcopy(event))
            except BaseException as error:
                self._stopped = True
                self._errors["evidence_errors"] = self._errors.get("evidence_errors", 0) + 1
                raise ReservationError(
                    "Reservation event sink failed; stop before further work"
                ) from error

    def _stop(self, kind: str, message: str) -> None:
        self._errors[kind] += 1
        self._stopped = True
        self._emit(dict(event="rejected", kind=kind, message=message))
        raise ReservationError(message)

    def _check_wall(self) -> None:
        if self._wall_seconds is not None and self._clock() - self._started > self._wall_seconds:
            self._stop("budget_errors", "Wall-time boundary exhausted; outer watchdog required")

    def reserve(
        self, name: str, work: Mapping[str, int], *, phase: str, arm: str | None = None
    ) -> Reservation:
        if self._stopped:
            raise ReservationError("Ledger stopped; no retry or further external work permitted")
        self._check_wall()
        try:
            values = _work(work)
        except (TypeError, ValueError) as error:
            self._stop("input_errors", str(error))
        if not values or not any(values.values()):
            self._stop("input_errors", "An external operation needs a positive work reservation")
        if not isinstance(name, str) or not name or name in self._names:
            self._stop("input_errors", "Unique nonempty reservation name required; no retry")
        if not isinstance(phase, str) or not phase:
            self._stop("input_errors", "An explicit phase is required")
        if self._phases and phase not in self._phases:
            self._stop("input_errors", "Unbudgeted phase: " + phase)
        if arm is not None and (arm not in self._arms or arm not in ARMS):
            self._stop("input_errors", "Unbudgeted arm: " + str(arm))
        scopes = ("total",)
        if arm is not None:
            scopes += ("arm:" + arm,)
        if phase in self._phases:
            scopes += ("phase:" + phase,)
        for scope in scopes:
            for key, amount in values.items():
                if key not in self._limits[scope]:
                    self._stop("input_errors", f"Unbudgeted counter {scope}.{key}")
                if self._requested[scope][key] + amount > self._limits[scope][key]:
                    self._stop("budget_errors", f"Pre-call budget exhausted: {scope}.{key}")
        identifier = f"reservation-{len(self._names):09d}"
        ticket = Reservation(identifier, name, arm, phase)
        for scope in scopes:
            for key, amount in values.items():
                self._requested[scope][key] += amount
        self._pending[identifier] = ticket, values, scopes
        self._names.add(name)
        self._emit(
            dict(
                event="reserved",
                identifier=identifier,
                name=name,
                arm=arm,
                phase=phase,
                work=values,
            )
        )
        return ticket

    def _settle(self, ticket: Reservation, error: BaseException | None = None) -> None:
        pending = self._pending.get(ticket.identifier)
        if pending is None or pending[0] != ticket:
            raise ReservationError("Unknown, altered or already settled reservation")
        _, values, scopes = self._pending.pop(ticket.identifier)
        target = self._completed if error is None else self._unknown
        for scope in scopes:
            for key, amount in values.items():
                target[scope][key] += amount
        self._emit(
            dict(
                event="completed" if error is None else "failed_unknown_partial",
                identifier=ticket.identifier,
                work=values,
                error_type=None if error is None else type(error).__name__,
                error=None if error is None else str(error),
            )
        )
        if error is not None:
            kind = "import_errors" if isinstance(error, ImportError) else "external_errors"
            self._errors[kind] += 1
            self._stopped = True

    def complete(self, ticket: Reservation) -> None:
        self._settle(ticket)
        self._check_wall()

    def fail(self, ticket: Reservation, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("Failure evidence must contain the raised exception")
        self._settle(ticket, error)

    def run(
        self,
        name: str,
        work: Mapping[str, int],
        operation: Callable[[], T],
        *,
        phase: str,
        arm: str | None = None,
    ) -> T:
        if not callable(operation):
            self._stop("input_errors", "External operation must be callable")
        ticket = self.reserve(name, work, phase=phase, arm=arm)
        try:
            result = operation()
        except BaseException as error:
            self.fail(ticket, error)
            raise
        self.complete(ticket)
        return result

    def snapshot(self, *, include_events: bool | None = None) -> dict[str, Any]:
        if include_events is None:
            include_events = self._retain_events
        if type(include_events) is not bool:
            raise TypeError("include_events must be a boolean")
        snapshot = dict(
            schema="walker-h5.0-precall-reservations-v1",
            limits=self._limits,
            requested=self._requested,
            completed=self._completed,
            unknown_partial=self._unknown,
            pending=list(self._pending),
            stopped=self._stopped,
            errors=self._errors,
            elapsed_seconds=self._clock() - self._started,
            storage_proof=False,
        )
        if include_events:
            snapshot["events"] = self._events
        return deepcopy(snapshot)
