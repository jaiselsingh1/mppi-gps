"""J1's finite repeated-planning and local-gain diagnostics.

The native planner and selector are unchanged. The observer copies their
returned arrays after native arithmetic, and restores every patched instance
method on exit. MuJoCo is imported only at the explicit execution boundary.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

OBSERVATION_SCALES = np.array([0.001] * 8 + [0.01] * 9, dtype=np.float64)
INNOVATION_SEEDS = (2026129701, 2026129702)
HELD_SEED = 2026129901


def probe_recipe() -> dict[str, Any]:
    return dict(
        schema="walker-j1-planner-probe-v1",
        contexts=4,
        context_controls=[250, 750, 250, 750],
        initial_plan=False,
        carried_and_zero_nominals=True,
        innovation_seeds=list(INNOVATION_SEEDS),
        original_rng_is_innovation=0,
        held_seed=HELD_SEED,
        held_directions=8,
        held_radius=1.5,
        observation_scales=OBSERVATION_SCALES.tolist(),
        hidden_time_step=0.008,
        hidden_rootx_step=0.01,
        warmstart_coordinate=0,
        warmstart_step=0.01,
        decisions=208,
        unwrapped_original_repeats=4,
        factorial_decisions=24,
        extra_gain_decisions=168,
        hidden_state_decisions=8,
        warmstart_decisions=4,
        local_fit_contexts=[0, 1],
        local_fit_innovations=[0, 1],
        local_gain_rule="SVD minimum-norm least squares; rank deficiency remains unidentified",
        contacts="Derived restored-boundary diagnostics, never original missing measurements",
        selector_event_details="Typed JSON array group; compact ledger binds group and C-byte SHA",
    )


def innovation_states(original: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return private states without drawing from the source planner's stream."""
    return [copy.deepcopy(original)] + [
        copy.deepcopy(np.random.PCG64(seed).state) for seed in INNOVATION_SEEDS
    ]


def held_directions() -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(HELD_SEED))
    directions = generator.choice(np.array([-1.0, 1.0]), size=(8, 17)) / np.sqrt(17.0)
    if len(np.unique(directions, axis=0)) != 8:
        raise ValueError("The frozen held perturbations are not distinct")
    return directions


def decision_roster() -> list[dict[str, Any]]:
    """Enumerate exactly 208 attempts; no outcome-dependent replacement exists."""
    rows: list[dict[str, Any]] = []

    def add(context: int, role: str, innovation: int = 0, **fields: Any) -> None:
        rows.append(
            dict(
                id=f"probe_{len(rows):03d}",
                context=context,
                role=role,
                nominal="carried",
                innovation=innovation,
                observation_delta=[0.0] * 17,
                time_delta=0.0,
                rootx_delta=0.0,
                warmstart_delta=0.0,
            )
            | fields
        )

    # Every original decision closes its unwrapped gate before any variation.
    for context in range(4):
        add(context, "original_repeat", observe=False)
    for context in range(4):
        for nominal in ("carried", "zero"):
            for innovation in range(3):
                add(context, "factorial", innovation, nominal=nominal, observe=True)
    for context in range(2):
        for innovation in range(2):
            for coordinate in range(17):
                for sign in (-1, 1):
                    delta = np.zeros(17)
                    delta[coordinate] = sign * OBSERVATION_SCALES[coordinate]
                    add(
                        context,
                        "gain_axis",
                        innovation,
                        coordinate=coordinate,
                        sign=sign,
                        observation_delta=delta.tolist(),
                        observe=True,
                    )
            for index, direction in enumerate(held_directions()):
                add(
                    context,
                    "gain_held",
                    innovation,
                    held_index=index,
                    observation_delta=(1.5 * direction * OBSERVATION_SCALES).tolist(),
                    observe=True,
                )
    for context in range(2):
        for coordinate, amount in (("time", 0.008), ("rootx", 0.01)):
            for sign in (-1, 1):
                add(context, "hidden_state", **{coordinate + "_delta": sign * amount}, observe=True)
    for context in range(2):
        for sign in (-1, 1):
            add(context, "warmstart", warmstart_delta=sign * 0.01, observe=True)
    if len(rows) != 208 or len({row["id"] for row in rows}) != 208:
        raise AssertionError("Frozen J1 planner census differs")
    return rows


def _copy(value: Any) -> Any:
    return np.array(value, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value)


def _exact(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError("Probe exact restoration differs: " + label)
        for key in expected:
            _exact(actual[key], expected[key], label + "/" + str(key))
    elif isinstance(expected, np.ndarray):
        left = np.asarray(actual)
        if (
            left.dtype != expected.dtype
            or left.shape != expected.shape
            or left.tobytes() != expected.tobytes()
        ):
            raise ValueError("Probe exact restoration differs: " + label)
    elif actual != expected:
        raise ValueError("Probe exact restoration differs: " + label)


class IterationObserver:
    """An instance-only copy observer; no native arithmetic or RNG call changes."""

    def __init__(self, native_mppi: Any):
        self.native = native_mppi
        self.passes: list[dict[str, Any]] = []
        self._noise: dict[str, Any] | None = None
        self._previous: dict[str, Any] = {}

    def __enter__(self) -> IterationObserver:
        native = self.native
        if getattr(native, "_j1_observer_active", False):
            raise ValueError("Nested planner observers are not permitted")
        self._previous = {
            name: native.__dict__.get(name) for name in ("_sample_noise", "_plan_iteration")
        }
        self._present = {name: name in native.__dict__ for name in self._previous}
        original_noise, original_iteration = native._sample_noise, native._plan_iteration

        def sample_noise() -> np.ndarray:
            before = copy.deepcopy(native._rng.bit_generator.state)
            result = original_noise()
            self._noise = dict(
                correlated_innovation=np.array(result, copy=True),
                rng_before=before,
                rng_after=copy.deepcopy(native._rng.bit_generator.state),
            )
            return result

        def iteration(*args: Any, **kwargs: Any) -> Any:
            nominal_before = native.U.copy()
            self._noise = None
            result = original_iteration(*args, **kwargs)
            if self._noise is None:
                raise ValueError("The native iteration supplied no innovation evidence")
            data = native.get_rollout_data()
            sampled = np.array(data["actions"], copy=True)
            self.passes.append(
                dict(
                    ordinal=len(self.passes),
                    nominal_before=nominal_before,
                    nominal_after=native.U.copy(),
                    effective_perturbations=sampled - nominal_before[None],
                    actions=sampled,
                    states=np.array(data["states"], copy=True),
                    native_costs=np.array(data["costs"], copy=True),
                    weights=np.array(data["weights"], copy=True),
                    sensordata=_copy(data["sensordata"]),
                    returned_action=np.array(result[0], copy=True),
                    diagnostics=copy.deepcopy(result[1]),
                    **self._noise,
                )
            )
            return result

        native._sample_noise = sample_noise
        native._plan_iteration = iteration
        native._j1_observer_active = True
        return self

    def __exit__(self, *error: Any) -> None:
        for name, value in self._previous.items():
            if self._present[name]:
                setattr(self.native, name, value)
            else:
                self.native.__dict__.pop(name, None)
        self.native.__dict__.pop("_j1_observer_active", None)


class _SelectorEvidence:
    """Retain large selector details without expanding frequent ledger rows."""

    def __init__(self, meter: Any, tag: str):
        self.meter, self.tag = meter, tag

    def save_arrays(self, tag: str, **arrays: Any) -> Any:
        return self.meter.save_arrays(tag, **arrays)

    def record(self, event: str, **fields: Any) -> str:
        if event != "coupled_selection":
            return self.meter.record(event, **fields)
        reference = self.meter.save_arrays(self.tag + "_selection_details", details=fields)
        return self.meter.record(
            event,
            details_group=reference["group"],
            details_sha256=reference["arrays"]["details"]["sha256"],
        )


def requested_state(
    snapshot: Mapping[str, Any], request: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    state = np.array(snapshot["state"], dtype=np.float64, copy=True)
    warmstart = np.array(snapshot["warmstart"], dtype=np.float64, copy=True)
    delta = np.asarray(request["observation_delta"], dtype=np.float64)
    if state.shape != (19,) or warmstart.shape != (9,) or delta.shape != (17,):
        raise ValueError("Probe state/observation/warmstart dimensions differ")
    active = np.flatnonzero(delta)
    state[active + 2] += delta[active]
    for coordinate, field in ((0, "time_delta"), (1, "rootx_delta")):
        if request[field] != 0:
            state[coordinate] += float(request[field])
    if request["warmstart_delta"] != 0:
        warmstart[0] += float(request["warmstart_delta"])
    if not np.isfinite(state).all() or not np.isfinite(warmstart).all():
        raise ValueError("Nonfinite probe perturbation")
    return state, warmstart


def _measure(meter: Any, category: str, operation: Callable[[], Any]) -> Any:
    return (
        operation() if meter is None else meter.measure(category, 1, operation, purpose="j1_probe")
    )


def prepare_context(
    env: Any,
    context: Mapping[str, Any],
    request: Mapping[str, Any],
    meter: Any = None,
    *,
    restore_context: Callable[[Mapping[str, Any]], Any] | None = None,
    contact_reader: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Restore first; perturb raw state and forward only for an actual intervention.

    A supplied restore callback owns its restore/forward/snapshot work census.
    Its argument is the complete simulator snapshot, without the probe request.
    """
    from .runtime import restore, snapshot

    source = context["snapshot"]
    if restore_context is None:
        if meter is not None:
            raise ValueError("A metered probe requires its owner's exact-restore callback")
        actual = restore(env, source)
    else:
        actual = restore_context(source)
    state, warmstart = requested_state(source, request)
    changed = not np.array_equal(state, source["state"]) or not np.array_equal(
        warmstart, source["warmstart"]
    )
    if changed:
        import mujoco

        env.set_state(state)
        env.set_warmstart(warmstart)
        _measure(meter, "external_mj_forward_calls", lambda: mujoco.mj_forward(env.model, env.data))
        env.set_warmstart(warmstart)
        actual = _measure(meter, "snapshot_reads", lambda: snapshot(env))
    if not np.array_equal(actual["state"], state) or not np.array_equal(
        actual["warmstart"], warmstart
    ):
        raise ValueError("The constructed perturbation context differs")
    observation = np.array(env._get_obs(), dtype=np.float64, copy=True)
    expected = np.concatenate((state[2:10], np.clip(state[10:19], -10.0, 10.0)))
    if observation.shape != (17,) or not np.array_equal(observation, expected):
        raise ValueError("Probe observations differ from the actual Walker interface")
    contacts = None if contact_reader is None else copy.deepcopy(contact_reader(env))
    return dict(
        snapshot=actual,
        observation=observation,
        observation_clipped_velocities=(np.abs(state[10:19]) > 10.0),
        contacts=contacts,
        contact_measurement="unavailable" if contacts is None else "derived_restored_boundary",
        intervened=changed,
    )


def run_decision(
    env: Any,
    planner: Any,
    selector: Any,
    context: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    meter: Any = None,
    restore_context: Callable[[Mapping[str, Any]], Any] | None = None,
    contact_reader: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """One restored single-producer decision; its owner persists the full result."""
    if selector.alpha != 0 or selector.eta_u != 0.01:
        raise ValueError("The probe retains the alpha0/eta0.01 independent selector")
    private = copy.deepcopy(context["planner"])
    if private["initial_plan"] is not False:
        raise ValueError("Every frozen probe context is noninitial")
    private["rng_state"] = innovation_states(private["rng_state"])[request["innovation"]]
    if request["nominal"] == "zero":
        private["U"] = np.zeros_like(private["U"])
    elif request["nominal"] != "carried":
        raise ValueError("Unknown probe nominal intervention")
    planner.restore({"native": private})
    prepared = prepare_context(
        env,
        context,
        request,
        meter,
        restore_context=restore_context,
        contact_reader=contact_reader,
    )

    def decision() -> tuple[Any, Any]:
        bank = planner.propose(prepared["snapshot"]["state"], prepared["snapshot"]["warmstart"])
        return selector.select(
            bank,
            actor=None,
            meter=None if meter is None else _SelectorEvidence(meter, request["id"]),
            enabled=False,
        )

    observed = bool(request["observe"])
    if observed:
        with IterationObserver(planner._planner) as observer:
            plan, diagnostics = decision()
        passes = observer.passes
        if len(passes) != 1:
            raise ValueError("The noninitial probe changed its optimization-pass count")
    else:
        plan, diagnostics = decision()
        passes = []
    after = planner.private_state()
    result = dict(
        request=copy.deepcopy(request),
        **prepared,
        selected_plan=np.array(plan, copy=True),
        action=np.array(plan[0], copy=True),
        action_at_bounds=(np.abs(plan[0]) >= 1.0 - 1e-12),
        planner_before=private,
        planner_after=after,
        passes=passes,
        selection=copy.deepcopy(diagnostics),
        original_replay_verified=False,
    )
    if request["role"] == "original_repeat" or (
        request["role"] == "factorial"
        and request["nominal"] == "carried"
        and request["innovation"] == 0
    ):
        _exact(result["action"], np.asarray(context["original_action"]), "original_action")
        _exact(after, context["original_planner_after"], "original_planner_after")
        if "original_selected_plan" in context:
            _exact(
                result["selected_plan"],
                np.asarray(context["original_selected_plan"]),
                "original_selected_plan",
            )
        if "original_bank_digest" in context:
            _exact(
                diagnostics["bank_digest"], context["original_bank_digest"], "original_bank_digest"
            )
        if "original_selected_task_cost" in context and not np.isclose(
            diagnostics["selected_task_cost"],
            context["original_selected_task_cost"],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Probe selected native task cost differs")
        result["original_replay_verified"] = True
    return result


def fit_local_gain(
    baseline_observation: Any,
    baseline_action: Any,
    axis_observations: Any,
    axis_actions: Any,
    held_observations: Any,
    held_actions: Any,
    *,
    normalization_std: Any = None,
) -> dict[str, Any]:
    """Fit a local first-action derivative, including its unidentified subspace."""
    base_o = np.asarray(baseline_observation, dtype=np.float64)
    base_u = np.asarray(baseline_action, dtype=np.float64)
    train_o = np.asarray(axis_observations, dtype=np.float64)
    train_u = np.asarray(axis_actions, dtype=np.float64)
    held_o = np.asarray(held_observations, dtype=np.float64)
    held_u = np.asarray(held_actions, dtype=np.float64)
    if (base_o.shape, base_u.shape, train_o.shape, train_u.shape, held_o.shape, held_u.shape) != (
        (17,),
        (6,),
        (34, 17),
        (34, 6),
        (8, 17),
        (8, 6),
    ) or not all(np.isfinite(x).all() for x in (base_o, base_u, train_o, train_u, held_o, held_u)):
        raise ValueError("The local gain's frozen finite design differs")
    x = (train_o - base_o) / OBSERVATION_SCALES
    y = train_u - base_u
    left, singular, right = np.linalg.svd(x, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(x.shape) * (singular[0] if len(singular) else 0.0)
    identified = singular > tolerance
    coefficients = right[identified].T @ ((left[:, identified].T @ y) / singular[identified, None])
    held_x = (held_o - base_o) / OBSERVATION_SCALES
    train_error = x @ coefficients - y
    held_error = held_x @ coefficients - (held_u - base_u)
    rank = int(identified.sum())
    physical = coefficients.T / OBSERVATION_SCALES[None]
    result = dict(
        schema="walker-j1-local-empirical-gain-v1",
        rank=rank,
        full_column_rank=rank == 17,
        unidentified_dimensions=17 - rank,
        singular_values=singular.tolist(),
        rank_tolerance=float(tolerance),
        condition_number=float(singular[0] / singular[-1]) if rank == 17 else None,
        gain_standardized=coefficients.T.tolist(),
        gain_per_physical_observation=physical.tolist(),
        training_action_rmse=float(np.sqrt(np.mean(train_error**2))),
        held_action_rmse=float(np.sqrt(np.mean(held_error**2))),
        held_action_max_abs_error=float(np.max(np.abs(held_error))),
        held_residuals=held_error.tolist(),
        minimum_norm_solution=True,
        optimal_gain_claim=False,
    )
    if normalization_std is not None:
        std = np.asarray(normalization_std, dtype=np.float64)
        if std.shape != (17,) or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Invalid original learner normalization")
        result["gain_per_normalized_observation"] = (physical * std[None]).tolist()
        result["normalization_is_algebraic_transform_only"] = True
    return result


def _presence(result: Mapping[str, Any]) -> np.ndarray | None:
    contacts = result.get("contacts")
    if contacts is None:
        return None
    if isinstance(contacts, Mapping):
        contacts = contacts.get("presence", contacts.get("contacts"))
    value = np.asarray(contacts)
    if value.shape != (2,) or value.dtype.kind != "b":
        raise ValueError("Contact diagnostics require two actual boolean presence channels")
    return value


def summarize_probe(results: Sequence[Mapping[str, Any]], *, normalization_std: Any = None) -> dict:
    roster = decision_roster()
    by_id = {row["request"]["id"]: row for row in results}
    if len(results) != 208 or len(by_id) != 208 or set(by_id) != {r["id"] for r in roster}:
        raise ValueError("The completed probe roster differs; missing work remains missing")
    for request in roster:
        if by_id[request["id"]]["request"] != request:
            raise ValueError("A probe request changed after enumeration")
    results = [by_id[request["id"]] for request in roster]
    for context in range(4):
        repeat = next(
            r
            for r in results
            if r["request"]["context"] == context and r["request"]["role"] == "original_repeat"
        )
        baseline = next(
            r
            for r in results
            if r["request"]["context"] == context
            and r["request"]["role"] == "factorial"
            and r["request"]["nominal"] == "carried"
            and r["request"]["innovation"] == 0
        )
        if (
            repeat.get("original_replay_verified") is not True
            or baseline.get("original_replay_verified") is not True
        ):
            raise ValueError("The original/observer decision gate did not pass")
        for field in ("action", "selected_plan", "planner_after"):
            _exact(baseline[field], repeat[field], "observer_equivalence/" + field)
    fits = []
    for context in range(2):
        for innovation in range(2):
            subset = [
                r
                for r in results
                if r["request"]["context"] == context and r["request"]["innovation"] == innovation
            ]
            baseline = next(
                r
                for r in subset
                if r["request"]["role"] == "factorial" and r["request"]["nominal"] == "carried"
            )
            axes = [r for r in subset if r["request"]["role"] == "gain_axis"]
            held = [r for r in subset if r["request"]["role"] == "gain_held"]
            fit = fit_local_gain(
                baseline["observation"],
                baseline["action"],
                [r["observation"] for r in axes],
                [r["action"] for r in axes],
                [r["observation"] for r in held],
                [r["action"] for r in held],
                normalization_std=normalization_std,
            )
            base_contacts = _presence(baseline)
            fit.update(context=context, innovation=innovation)
            for name, group in (("axis", axes), ("held", held)):
                observed = [_presence(r) for r in group]
                fit[name + "_contact_regime_changes"] = (
                    None
                    if base_contacts is None or any(v is None for v in observed)
                    else sum(not np.array_equal(v, base_contacts) for v in observed)
                )
                fit[name + "_contact_measurements_available"] = sum(v is not None for v in observed)
                fit[name + "_action_boundary_rows"] = sum(
                    bool(np.any(np.abs(r["action"]) >= 1.0 - 1e-12)) for r in group
                )
            fit["baseline_clipped_velocity_coordinates"] = np.asarray(
                baseline.get("observation_clipped_velocities", np.zeros(9, dtype=bool))
            ).tolist()
            fits.append(fit)
    # Numeric contrasts are descriptive; no residual is classified irrelevant.
    contrasts = []
    for context in range(4):
        cells = {
            (r["request"]["nominal"], r["request"]["innovation"]): r
            for r in results
            if r["request"]["context"] == context and r["request"]["role"] == "factorial"
        }
        contrasts.append(
            dict(
                context=context,
                nominal_action_squared_changes=[
                    float(
                        np.mean(
                            (
                                np.asarray(cells["carried", i]["action"])
                                - np.asarray(cells["zero", i]["action"])
                            )
                            ** 2
                        )
                    )
                    for i in range(3)
                ],
                carried_rng_action_variance=np.var(
                    [cells["carried", i]["action"] for i in range(3)], axis=0
                ).tolist(),
                zero_rng_action_variance=np.var(
                    [cells["zero", i]["action"] for i in range(3)], axis=0
                ).tolist(),
            )
        )
    hidden = []
    for row in results:
        if row["request"]["role"] not in {"hidden_state", "warmstart"}:
            continue
        baseline = next(
            r
            for r in results
            if r["request"]["role"] == "factorial"
            and r["request"]["context"] == row["request"]["context"]
            and r["request"]["nominal"] == "carried"
            and r["request"]["innovation"] == 0
        )
        _exact(
            np.asarray(row["observation"]),
            np.asarray(baseline["observation"]),
            "hidden_observation_interface",
        )
        hidden.append(
            dict(
                id=row["request"]["id"],
                context=row["request"]["context"],
                role=row["request"]["role"],
                observed_input_unchanged=True,
                action_squared_change=float(
                    np.mean((np.asarray(row["action"]) - np.asarray(baseline["action"])) ** 2)
                ),
            )
        )
    return dict(
        schema="walker-j1-probe-summary-v1",
        decisions=208,
        original_replay_checks=sum(r.get("original_replay_verified") is True for r in results),
        local_fits=fits,
        factorial_contrasts=contrasts,
        hidden_input_contrasts=hidden,
        unexplained_variation_is_irrelevant=False,
        optimal_gain_claim=False,
    )
