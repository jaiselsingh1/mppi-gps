"""Uncompressed H5.0 byte census and executable serialization contracts.

Only fixed array shapes and bounded ASCII JSON are used here. The resulting
upper bound is conditional on the native adapter enforcing every listed
contract; a global file quota alone is not a full-roster storage proof. This
module imports no array, policy, simulator or binary writer implementation.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .proposal_budget import census

GIB = 1024**3
QUOTAS = {
    "payload": 384 * GIB,
    "ledger": 80 * GIB,
    "index": 32 * GIB,
    "external": 16 * GIB,
}
CONTRACT = {
    "max_array_groups": 2_000_000,
    "max_ledger_records": 5_000_000,
    "emergency_records": 8,
    "index_lines_per_group": 2,
    "max_index_line_bytes": 8192,
    "max_event_line_bytes": 16384,
    "max_array_bytes": 16 * 1024**2,
    "max_arrays_per_group": 24,
    "max_integration_components": 128,
    "state_components": 19,
    "action_components": 6,
    "sensor_components": 0,
    "native_info_text_bytes": 1024,
    "episode_row_text_bytes": 2048,
    "private_planner_text_bytes": 8192,
    "child_row_tag_characters": 64,
    "global_rng_bytes": 65536,
}


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an explicit integer >= {minimum}")
    return value


def check_native_dimensions(*, state: int, action: int, integration: int, sensors: int) -> None:
    """Reject uncosted model dimensions before entering a native query."""
    values = {"state": state, "action": action, "integration": integration, "sensors": sensors}
    for name, value in values.items():
        _integer(value, name)
    if state != 19 or action != 6 or sensors != 0 or not 1 <= integration <= 128:
        raise ValueError("Native dimensions exceed the frozen H5.0 storage contract")


def bounded_json_payload(text: str, *, text_bytes: int, purpose: str) -> int:
    """Validate serialized ASCII JSON and return its NumPy Unicode C-byte size.

    Callers serialize with the existing ``runtime._dumps`` first, so NumPy
    arrays/scalars retain that established conversion. ``ensure_ascii=True``
    makes byte and character limits identical. Stored ``<U`` data uses four
    bytes per character; charging UTF8 length alone would undercount it.
    """
    _integer(text_bytes, "text_bytes", 1)
    if not isinstance(text, str) or not text.isascii():
        raise ValueError("The frozen JSON contract requires an ASCII serialized string")
    if len(text) > text_bytes:
        raise ValueError(f"{purpose} exceeds its frozen JSON text bound")

    def reject_constant(value: str) -> None:
        raise ValueError("Nonfinite JSON constants violate the frozen contract: " + value)

    json.loads(text, parse_constant=reject_constant)
    return 4 * len(text)


def model_query_payload_bound(controls: int, trajectories: int) -> int:
    """Canonical actions, post-states, scores and both initial-context payloads.

    The 2KiB context allowance covers initial state19, warmstart9, at most128
    integration values, and the fixed terminal-role Unicode scalar. There are
    no sensor values. Binary index/ledger bytes are separate categories.
    """
    _integer(controls, "controls", 1)
    _integer(trajectories, "trajectories", 1)
    if trajectories not in (1, 256) or controls != trajectories * 48:
        raise ValueError("Uncosted H5.0 model batch shape")
    return controls * (19 + 6) * 8 + trajectories * 8 + 2048


def live_query_payload_bound() -> int:
    """A full before/after live query, including bounded native-info JSON.

    Input: action6,state19,warm9,integration128,spec1,dimensions3. Output:
    observation17,cost1,state19,warm9,integration128,done bool and native info.
    This is 340 eight-byte words, one bool and at most1024 Unicode characters.
    """
    return 340 * 8 + 1 + 1024 * 4


def episode_payload_bound(controls: int) -> int:
    """Complete episode payload, including N+1 physical boundaries and infos.

    Every combined controller/native row must fit2048 serialized characters.
    The JSON list contributes its commas and ``{"rows":[]}`` envelope. An
    unexpected exception stores a smaller partial episode instead, never both
    a complete and a partial group. Three terminal scalars are included.
    """
    _integer(controls, "controls")
    if controls > 1000:
        raise ValueError("Uncosted H5.0 episode duration")
    numeric = ((19 + 9 + 128) * (controls + 1) + (17 + 6 + 1 + 1) * controls) * 8
    text = controls * 2048 + max(controls - 1, 0) + 11
    return numeric + text * 4 + 3 * 8


def storage_census(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the full roster's bounded bytes without allocating its arrays.

    The event count follows current Meter query/measurement request+result
    pairs, guidance callbacks and native operations. ReservationLedger must
    use retain_events=False and sink=None: its counters summarize the same
    paid operations and cannot create a second duplicated event stream.
    Eight emergency lines are included, rather than an unbounded error retry.
    """
    work = census(recipe)
    if recipe["budgets"]["storage_bytes"] != sum(QUOTAS.values()):
        raise ValueError("Version the changed H5.0 storage envelope")
    total = work["total"]
    models = total["model_controls"]
    model_queries = total["model_queries"]
    live = total["live_controls"]
    decisions = total["teacher_decisions"]
    rows = total["guidance_forward_rows"]
    teacher_episodes = sum(work["per_arm"]["teacher_episodes"] for _ in recipe["arms"])
    passes = teacher_episodes * work["per_episode"]["optimization_passes_per_stream"]
    episodes = work["worker_episodes"]
    streams, arms = len(recipe["streams"]), len(recipe["arms"])
    fits = streams * (1 + arms)
    checkpoints = streams * (2 + arms)
    children = streams * arms
    trajectories = models // recipe["mppi"]["H"]
    # Every largest candidate scalar/support field is covered by twelve
    # float64 values per trajectory, even where the actual field is a bool.
    payload = dict(
        native_action_and_post_states=models * 25 * 8,
        native_raw_scores=trajectories * 8,
        native_context=model_queries * 2048,
        guidance_predictions=rows * 6 * 8,
        guidance_support_masks=rows,
        guidance_scalar_vectors=(rows // 48) * 12 * 8,
        private_planner_unicode=decisions * 4 * 8192 * 4,
        predecision_planner_unicode=decisions * 2 * 8192 * 4,
        final_cache_weights=decisions * 2 * recipe["mppi"]["K"] * 8,
        selector_plan_tapes=decisions * 3 * 48 * 6 * 8,
        inner_nominals=passes * 48 * 6 * 8,
        inner_initial_states=passes * 19 * 8,
        live_native=live * live_query_payload_bound(),
        episode_groups=episodes * episode_payload_bound(1000),
        minibatch_tapes=fits * 2000 * 256 * 8,
        child_datasets=children * 4000 * (23 * 8 + 64 * 4 + 8),
        global_torch_rng=checkpoints * 65536,
        teacher_posthoc=decisions * (6 * 8 + 8),
        unexpected_failure_context=2048,
        miscellaneous_payload=GIB,
    )
    groups = dict(
        native_query_groups=2 * (model_queries + live),
        proposal_pass_groups=passes,
        selector_groups=decisions,
        predecision_context_groups=decisions,
        episode_groups=episodes,
        checkpoint_rng_groups=checkpoints,
        minibatch_tape_groups=fits,
        child_dataset_groups=children,
        teacher_posthoc_groups=teacher_episodes,
        unexpected_failure_context_groups=1,
    )
    # Guidance predict calls are passes+two final calls per decision; posthoc
    # is one ordered batch per teacher episode. Metrics use fixed1024 batches.
    guidance_calls = passes + 2 * decisions
    diagnostic_calls = 162
    neural_calls = guidance_calls + teacher_episodes + work["standalone_live_controls"]
    neural_calls += work["worker_optimizer_steps"] + diagnostic_calls
    measurement_calls = neural_calls + 2 * work["worker_optimizer_steps"]
    measurement_calls += work["checkpoint_loads"]
    ledger = dict(
        query_request_and_result=2 * (model_queries + live),
        measurement_request_and_result=2 * measurement_calls,
        fit_update_observer=work["worker_optimizer_steps"],
        fit_initial_state_observer=fits,
        agreement_requested=guidance_calls,
        scoring_events=2 * (passes + decisions),
        decision_started=decisions,
        predecision_context_events=2 * decisions,
        episode_reset_progress_complete=episodes * 13,
        checkpoint_events=checkpoints,
        full_metric_events=48,
        teacher_posthoc_events=teacher_episodes,
        teacher_stream_closed=streams,
        lifecycle_events=5,
        emergency_events=CONTRACT["emergency_records"],
    )
    payload_bytes = sum(payload.values())
    group_count, record_count = sum(groups.values()), sum(ledger.values())
    index_bytes = group_count * 2 * CONTRACT["max_index_line_bytes"]
    ledger_bytes = record_count * CONTRACT["max_event_line_bytes"]
    checks = dict(
        payload=payload_bytes <= QUOTAS["payload"],
        array_groups=group_count <= CONTRACT["max_array_groups"],
        ledger_records=record_count <= CONTRACT["max_ledger_records"],
        index=index_bytes <= QUOTAS["index"],
        ledger=ledger_bytes <= QUOTAS["ledger"],
        category_sum=sum(QUOTAS.values()) == recipe["budgets"]["storage_bytes"],
        # Largest actual known array is the full episode's Unicode infos.
        episode_info_array=(2049 * 1000 + 10) * 4 <= CONTRACT["max_array_bytes"],
    )
    if not all(checks.values()):
        raise ValueError("H5.0 byte census does not fit; version before execution")
    return dict(
        schema="walker-h5-storage-census-v1",
        payload_components=payload,
        payload_upper_bytes=payload_bytes,
        array_group_components=groups,
        array_groups_upper=group_count,
        index_upper_bytes=index_bytes,
        ledger_components=ledger,
        ledger_records_upper=record_count,
        ledger_upper_bytes=ledger_bytes,
        external_reserved_bytes=QUOTAS["external"],
        total_upper_bytes=payload_bytes + index_bytes + ledger_bytes + QUOTAS["external"],
        quotas=dict(QUOTAS),
        contract=dict(CONTRACT),
        fits=True,
        proof_scope=(
            "Conditional full-roster upper bound: NativeMeter must enforce all dimensions, "
            "JSON/schema/group/line/byte limits and pre-call output reservations. "
            "No duplicated native banks, HDF5 metadata, compression credit or second "
            "ReservationLedger event sink. OS errors retain paid intent and stop."
        ),
    )
