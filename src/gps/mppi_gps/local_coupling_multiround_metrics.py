"""Saved-only final L3 screens; no outcome can terminate an alternating round.

The worker uses these summaries for its declared final measurements. Independent
audit recomputes the underlying physical outcomes and bindings separately. No
policy, planner, fitter or simulator is imported here. Missing work stays missing;
an observed zero success count is distinct from an absent evaluation panel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("independent", "positive")
PANELS = ("primary", "confirmation")
PROBE_CONTROLS = (0, 50, 100)


class SavedMeasurements:
    def __init__(self) -> None:
        self.missing: list[dict] = []
        self._json: dict[str, dict] = {}
        self._traces: dict[str, dict] = {}

    def absent(self, role: str, detail: str) -> None:
        self.missing.append(dict(role=role, detail=detail))

    def bound(self, spec: dict) -> Path:
        if not isinstance(spec, dict) or not {"path", "bytes", "sha256"} <= spec.keys():
            raise ValueError("An explicit saved byte binding is required")
        path = Path(spec["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Bound saved evidence is unavailable: " + str(path))
        data = path.read_bytes()
        if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError("Bound saved evidence changed: " + str(path))
        return path

    def json(self, spec: dict | None, role: str) -> dict | None:
        if spec is None:
            self.absent(role, "No closed saved record")
            return None
        path = self.bound(spec)
        key = str(path)
        if key not in self._json:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("Saved measurement JSON must contain an object")
            self._json[key] = value
        return self._json[key]

    def trace(self, spec: dict) -> dict:
        path = self.bound(spec)
        key = str(path)
        if key not in self._traces:
            import h5py

            with h5py.File(path, "r") as raw:
                actions = raw["actions"][:]
                contexts = {name: raw[name][0] for name in ("states", "warmstarts", "integrations")}
                lengths = {name: len(raw[name]) for name in contexts}
            if (
                actions.ndim != 2
                or actions.shape[1] != 6
                or not 0 < len(actions) <= 300
                or actions.dtype != np.float64
                or not np.isfinite(actions).all()
                or np.any(np.abs(actions) > 1)
                or any(n != len(actions) + 1 for n in lengths.values())
                or any(not np.isfinite(value).all() for value in contexts.values())
            ):
                raise ValueError(
                    "Expected actual bounded native acquisition controls and N+1 contexts"
                )
            self._traces[key] = dict(actions=actions, initial_context=contexts)
        return self._traces[key]


def _records(rows: list[dict], key, role: str) -> dict:
    result = {}
    for row in rows:
        if "arm" in row and row["arm"] not in ARMS:
            raise ValueError("Unknown saved acquisition/learning arm")
        identity = key(row)
        if identity in result:
            raise ValueError("Duplicate saved " + role + ": " + str(identity))
        result[identity] = row
    return result


def _panel(saved: SavedMeasurements, rows: list[dict], role: str, size: int) -> dict:
    unique = _records(rows, lambda row: row["reset_seed"], role)
    if len(unique) > size:
        raise ValueError("Saved panel exceeds the frozen reset count")
    observed = []
    for seed, row in unique.items():
        value = saved.json(row.get("result"), role)
        if value is None:
            continue
        if value.get("status") != "COMPLETED" or "outcome" not in value:
            saved.absent(role, "Evaluation did not close successfully: " + str(seed))
            continue
        if value.get("reset_seed", seed) != seed:
            raise ValueError("Saved evaluation reset differs from its roster")
        outcome = value["outcome"]
        if (
            type(outcome["walking"]) is not bool
            or type(outcome["full_survival"]) is not bool
            or type(outcome["controls"]) is not int
            or not 0 < outcome["controls"] <= 1000
            or not np.isfinite(outcome["complete_cost"])
            or outcome["full_survival"]
            and outcome["controls"] != 1000
            or outcome["walking"]
            and not outcome["full_survival"]
        ):
            raise ValueError("Contradictory or nonfinite saved standalone outcome")
        observed.append(outcome)
    complete = len(observed) == size
    if not complete:
        saved.absent(role, f"Only {len(observed)}/{size} actual episodes")
    return dict(
        complete=complete,
        episodes=len(observed),
        walking=sum(row["walking"] for row in observed) if complete else None,
        full_survival=sum(row["full_survival"] for row in observed) if complete else None,
        observed_walking=sum(row["walking"] for row in observed),
        observed_full_survival=sum(row["full_survival"] for row in observed),
        controls=sum(row["controls"] for row in observed),
        complete_cost=float(np.mean([row["complete_cost"] for row in observed]))
        if observed
        else None,
    )


def _ratio(initial: float | None, final: float | None, threshold: float) -> dict:
    positive_initial = initial is not None and initial > 0 and final is not None
    fraction = (initial - final) / initial if positive_initial else None
    return dict(
        initial=initial,
        final=final,
        ratio=final / initial if positive_initial else None,
        fractional_reduction=fraction,
        reduction_threshold=threshold,
        reduction_met=fraction >= threshold if fraction is not None else None,
        stationary_zero_gap=initial == 0 and final == 0,
    )


def _metric(saved: SavedMeasurements, fit: dict | None, stage: str, name: str, role: str):
    metric = fit.get("diagnostics_" + stage, {}).get(name) if fit else None
    if metric is None:
        saved.absent(role, f"No {stage}/{name} measurement")
        return None
    value = float(metric["action_mse"])
    if not np.isfinite(value) or value < 0:
        raise ValueError("A saved action-MSE measurement is invalid")
    return value


def _slope(values: list[tuple[int, float | None]], expected: int) -> dict:
    observed = [(x, y) for x, y in values if y is not None]
    if len(observed) != expected:
        return dict(slope=None, observations=len(observed), expected=expected, points=values)
    x, y = np.asarray(observed, dtype=np.float64).T
    centered = x - x.mean()
    slope = float(np.sum(centered * (y - y.mean())) / np.sum(centered**2)) if expected > 1 else None
    return dict(slope=slope, observations=len(observed), expected=expected, points=values)


def _probe(saved: SavedMeasurements, rows: list[dict], arm: str, role: str) -> dict[int, float]:
    matching = [row for row in rows if row["arm"] == arm]
    if len(matching) > 1:
        raise ValueError("Duplicate saved probe actor/round")
    value = saved.json(matching[0].get("result") if matching else None, role)
    if value is None or value.get("status") != "COMPLETED":
        if value is not None:
            saved.absent(role, "Probe job did not close successfully")
        return {}
    anchors = _records(value.get("anchors", []), lambda row: row["control"], "probe anchor")
    if set(anchors) != set(PROBE_CONTROLS):
        raise ValueError("Saved probe must preserve all three observed/missing anchor declarations")
    result = {}
    for control, row in anchors.items():
        if row["status"] == "MISSING_AFTER_EXECUTED_PREFIX":
            if row.get("gap_action_mse") is not None:
                raise ValueError("Missing post-fall probe cannot have an invented gap")
            continue
        if row["status"] != "OBSERVED":
            raise ValueError("Unknown saved probe anchor status")
        gap = float(row["gap_action_mse"])
        if not np.isfinite(gap) or gap < 0:
            raise ValueError("Invalid observed student-state action gap")
        result[control] = gap
    return result


def _student_gap(initial: dict[int, float], final: dict[int, float], threshold: float) -> dict:
    common = sorted(initial.keys() & final.keys())
    first = float(np.mean([initial[k] for k in common])) if common else None
    last = float(np.mean([final[k] for k in common])) if common else None
    no_fewer = len(final) >= len(initial)
    no_lost = initial.keys() <= final.keys()
    value = _ratio(first, last, threshold)
    return dict(
        **value,
        initial_observed=len(initial),
        final_observed=len(final),
        common_anchors=len(common),
        common_controls=common,
        initial_controls=sorted(initial),
        final_controls=sorted(final),
        no_fewer_anchors=no_fewer,
        no_lost_anchor_coverage=no_lost,
        contraction_with_preserved_coverage=bool(value["reduction_met"] and no_fewer and no_lost),
    )


def _activity(saved: SavedMeasurements, rows: list[dict], stream: int, round_index: int, arm: str):
    matching = _records(
        [row for row in rows if row["arm"] == arm], lambda row: row["slot"], "admission slot"
    )
    if set(matching) != set(range(8)):
        saved.absent(
            "admission roster", f"stream{stream}/r{round_index}/{arm}: {len(matching)}/8 slots"
        )
    for row in matching.values():
        if (
            type(row["admitted"]) is not bool
            or type(row["fallback"]) is not bool
            or row["admitted"] == row["fallback"]
            or type(row["new_experience"]) is not bool
            or row["new_experience"]
            and not row["admitted"]
            or row["new_rows"] != (300 if row["new_experience"] else 0)
        ):
            raise ValueError("Saved admission/new-experience/fallback declarations contradict")
    n = len(matching)
    admitted = sum(row["admitted"] for row in matching.values())
    fresh = sum(row["new_experience"] for row in matching.values())
    fallback = sum(row["fallback"] for row in matching.values())
    return dict(
        stream=stream,
        round=round_index,
        arm=arm,
        complete=n == 8,
        observed_slots=n,
        admitted_slots=admitted,
        accepted_identical_slots=admitted - fresh,
        new_experience_slots=fresh,
        fallback_slots=fallback,
        new_rows=sum(row["new_rows"] for row in matching.values()),
        admitted_fraction=admitted / n if n else None,
        accepted_identical_fraction=(admitted - fresh) / n if n else None,
        new_fraction=fresh / n if n else None,
        fallback_fraction=fallback / n if n else None,
        max_retained_age=max((row["age"] for row in matching.values()), default=None),
    )


def _attempts(saved: SavedMeasurements, block: dict, arm: str, stream: int) -> dict:
    rows = _records(
        [row for row in block.get("acquisitions", []) if row["arm"] == arm],
        lambda row: row["slot"],
        "attempted acquisition slot",
    )
    if set(rows) != set(range(8)):
        saved.absent("acquisition roster", f"stream{stream}/r{block['round']}/{arm}: {len(rows)}/8")
    values = {}
    for slot, row in rows.items():
        value = saved.json(row.get("manifest"), "attempted teacher manifest")
        if value is not None and value.get("status") == "COMPLETED":
            values[slot] = value
        else:
            saved.absent(
                "attempted teacher manifest", f"stream{stream}/r{block['round']}/{arm}/s{slot}"
            )
    return values


def _feedback(
    saved: SavedMeasurements, previous: dict, current: dict, activity: dict, threshold: float
) -> dict:
    value = dict(
        stream=activity["stream"],
        round=activity["round"],
        arm=activity["arm"],
        new_rows=activity["new_rows"],
        complete=False,
        actor_changed=None,
        same_private_seeds=None,
        same_initial_contexts=None,
        pooled_action_rms=None,
        effective_attempt=None,
        genuine_feedback_event=False,
    )
    if set(previous) != set(range(8)) or set(current) != set(range(8)):
        return value
    before = {row["actor_hash"] for row in previous.values()}
    after = {row["actor_hash"] for row in current.values()}
    if len(before) != 1 or len(after) != 1:
        raise ValueError(
            "One actual incoming actor must score every restart in an acquisition block"
        )
    same_seeds, same_contexts, differences, cases = True, True, [], []
    for slot in range(8):
        a, b = previous[slot], current[slot]
        left, right = saved.trace(a["trace"]), saved.trace(b["trace"])
        same_seeds = same_seeds and a["seed"] == b["seed"]
        same = all(
            left["initial_context"][key].tobytes() == right["initial_context"][key].tobytes()
            for key in left["initial_context"]
        )
        same_contexts = same_contexts and same
        n = min(len(left["actions"]), len(right["actions"]))
        delta = (left["actions"][:n] - right["actions"][:n]) ** 2
        differences.append(delta.ravel())
        cases.append(dict(slot=slot, common_controls=n, action_rms=float(np.sqrt(delta.mean()))))
    rms = float(np.sqrt(np.concatenate(differences).mean()))
    changed = before != after
    effective = bool(changed and same_seeds and same_contexts and rms >= threshold)
    value.update(
        complete=True,
        actor_changed=changed,
        same_private_seeds=same_seeds,
        same_initial_contexts=same_contexts,
        pooled_action_rms=rms,
        cases=cases,
        effective_attempt=effective,
        genuine_feedback_event=bool(effective and activity["new_rows"] > 0),
    )
    return value


def summarize(config: dict, streams: list[dict]) -> dict:
    """Read the full planned trajectory; final criteria never control execution."""
    planned, size = config["rounds"], config["seeds"]["panel_size"]
    if planned != 8 or size != 16:
        raise ValueError("The frozen L3 eight-round, sixteen-reset design is required")
    rules = config["criteria"]
    gain, required = rules["walking_gain_min"], rules["winning_streams_min"]
    if gain != 2 or required != 2 or rules["last_round_window"] != 4:
        raise ValueError("Frozen final L3 outcome definitions changed")
    saved = SavedMeasurements()
    indexed = _records(streams, lambda row: row["stream"], "stream identity")
    if not set(indexed) <= set(range(3)):
        raise ValueError("Unknown stream identity")
    panels, gap_details, activities, events, statuses, competence_rows = {}, [], [], [], [], []
    stream_blocks = {}
    for stream_id in range(3):
        stream = indexed.get(stream_id)
        if stream is None:
            saved.absent("stream", f"No closed stream{stream_id} record")
            continue
        blocks = _records(stream.get("rounds", []), lambda row: row["round"], "round identity")
        if not set(blocks) <= set(range(1, planned + 1)):
            raise ValueError("Round identity exceeds the frozen trajectory")
        stream_blocks[stream_id] = blocks
        complete = (
            stream.get("status") == "COMPLETED_ALL_ROUNDS"
            and set(blocks) == set(range(1, planned + 1))
            and all(block.get("status") == "COMPLETED" for block in blocks.values())
        )
        statuses.append(
            dict(
                stream=stream_id,
                status=stream.get("status"),
                rounds=sorted(blocks),
                complete=complete,
            )
        )
        if not complete:
            saved.absent(
                "complete trajectory",
                f"stream{stream_id}: {stream.get('status')}, rounds{sorted(blocks)}",
            )
        fits, attempts = {}, {}
        primary = {}
        for arm in ARMS:
            initial = [
                row
                for row in stream.get("initial_evaluations", [])
                if row["arm"] == arm and row["panel"] == "primary"
            ]
            primary[(0, arm)] = _panel(
                saved, initial, f"stream{stream_id}/{arm}/initial-primary", size
            )
            panels[(stream_id, "primary", arm, "initial")] = primary[(0, arm)]
            for kind in ("initial", "final"):
                rows = [
                    row
                    for row in stream.get("final_evaluations", [])
                    if row["arm"] == arm and row["panel"] == "confirmation" and row["kind"] == kind
                ]
                panels[(stream_id, "confirmation", arm, kind)] = _panel(
                    saved, rows, f"stream{stream_id}/{arm}/{kind}-confirmation", size
                )
        for round_index in range(1, planned + 1):
            block = blocks.get(round_index, dict(round=round_index))
            fit_rows = _records(block.get("fits", []), lambda row: row["arm"], "fit actor")
            for arm in ARMS:
                fits[(round_index, arm)] = saved.json(
                    fit_rows.get(arm, {}).get("result"),
                    f"stream{stream_id}/r{round_index}/{arm}/fit",
                )
                if (
                    fits[(round_index, arm)] is not None
                    and fits[(round_index, arm)].get("status") != "COMPLETED"
                ):
                    saved.absent("completed fit", f"stream{stream_id}/r{round_index}/{arm}")
                    fits[(round_index, arm)] = None
                rows = [
                    row
                    for row in block.get("evaluations", [])
                    if row["arm"] == arm and row["panel"] == "primary"
                ]
                primary[(round_index, arm)] = _panel(
                    saved, rows, f"stream{stream_id}/r{round_index}/{arm}/primary", size
                )
                attempts[(round_index, arm)] = _attempts(saved, block, arm, stream_id)
                activity = _activity(
                    saved, block.get("admissions", []), stream_id, round_index, arm
                )
                activities.append(activity)
                if round_index > 1:
                    events.append(
                        _feedback(
                            saved,
                            attempts[(round_index - 1, arm)],
                            attempts[(round_index, arm)],
                            activity,
                            0.01,
                        )
                    )
        for arm in ARMS:
            panels[(stream_id, "primary", arm, "final")] = primary[(planned, arm)]
            gaps = {}
            for name, key, threshold in (
                ("own", "current_teacher", rules["teacher_gap_reduction_min"]),
                ("anchor", "fixed_anchor", rules["fixed_anchor_gap_reduction_min"]),
                ("held", "held", rules["fixed_anchor_gap_reduction_min"]),
            ):
                initial = _metric(
                    saved, fits[(1, arm)], "before", name, f"stream{stream_id}/{arm}/initial-{name}"
                )
                final = _metric(
                    saved,
                    fits[(planned, arm)],
                    "after",
                    name,
                    f"stream{stream_id}/{arm}/final-{name}",
                )
                points = [
                    (
                        r,
                        _metric(
                            saved,
                            fits[(r, arm)],
                            "after",
                            name,
                            f"stream{stream_id}/r{r}/{arm}/{name}",
                        ),
                    )
                    for r in range(planned - 3, planned + 1)
                ]
                gaps[key] = _ratio(initial, final, threshold) | dict(
                    last_four_postfit_slope=_slope(points, 4),
                    distribution="Changing admitted teacher states"
                    if name == "own"
                    else "Fixed measurement states",
                )
            observed_probes = []
            for r in range(0, planned + 1):
                rows = (
                    stream.get("initial_probes", [])
                    if r == 0
                    else blocks.get(r, {}).get("probes", [])
                )
                observed_probes.append(
                    _probe(saved, rows, arm, f"stream{stream_id}/r{r}/{arm}/probe")
                )
            gaps["student_common_anchor"] = _student_gap(
                observed_probes[0],
                observed_probes[-1],
                rules["student_common_anchor_gap_reduction_min"],
            )
            gaps["student_probe_coverage"] = [
                dict(round=r, observed_controls=sorted(value), observed_anchors=len(value))
                for r, value in enumerate(observed_probes)
            ]
            gap_details.append(dict(stream=stream_id, arm=arm, **gaps))
            last_window = [primary[(r, arm)] for r in range(planned - 2, planned + 1)]
            competence_rows.append(
                dict(
                    stream=stream_id,
                    arm=arm,
                    primary_final=primary[(planned, arm)],
                    confirmation_final=panels[(stream_id, "confirmation", arm, "final")],
                    individually_competent=all(
                        p["complete"] and p["walking"] >= 15
                        for p in (
                            primary[(planned, arm)],
                            panels[(stream_id, "confirmation", arm, "final")],
                        )
                    ),
                    last_three_primary=[
                        dict(
                            round=r,
                            walking=p["walking"],
                            complete=p["complete"],
                            competent=p["walking"] >= 15 if p["complete"] else None,
                        )
                        for r, p in zip(range(planned - 2, planned + 1), last_window, strict=True)
                    ],
                )
            )
    winners, walking_rows = [], []
    for stream_id in range(3):
        wins = []
        for panel in PANELS:
            cell = {
                name: panels.get((stream_id, panel, arm, kind))
                for arm, kind, name in (
                    ("independent", "initial", "incoming_independent"),
                    ("positive", "initial", "incoming_positive"),
                    ("independent", "final", "independent"),
                    ("positive", "final", "positive"),
                )
            }
            complete = all(p is not None and p["complete"] for p in cell.values())
            if complete:
                a, b, zero, positive = (
                    cell[key]
                    for key in (
                        "incoming_independent",
                        "incoming_positive",
                        "independent",
                        "positive",
                    )
                )
                gp, gz = positive["walking"] - b["walking"], zero["walking"] - a["walking"]
                safe = all(
                    positive[key] >= other[key]
                    for key in ("walking", "full_survival")
                    for other in (zero, b)
                )
                win = (
                    gp >= gain and positive["walking"] >= zero["walking"] + gain and gp - gz >= gain
                )
            else:
                gp = gz = safe = win = None
            wins.append(win is True)
            walking_rows.append(
                dict(
                    stream=stream_id,
                    panel=panel,
                    **cell,
                    complete=complete,
                    walking_gain_positive=gp,
                    walking_gain_independent=gz,
                    difference_in_gain=gp - gz if complete else None,
                    nonregression=safe,
                    qualifying_gain=win,
                )
            )
        if all(wins):
            winners.append(stream_id)
    positive_events = [
        row for row in events if row["arm"] == "positive" and row["genuine_feedback_event"]
    ]
    complete = len(statuses) == 3 and all(row["complete"] for row in statuses) and not saved.missing
    nonregression = (
        all(row["nonregression"] for row in walking_rows)
        if all(row["complete"] for row in walking_rows)
        else None
    )
    return dict(
        schema="local-coupling-multiround-final-measurements-1",
        scientific_complete=complete,
        planned_rounds=planned,
        stream_status=statuses,
        walking=dict(
            passed=bool(complete and nonregression and len(winners) >= required),
            winning_streams=winners,
            nonregression_all=nonregression,
            details=walking_rows,
        ),
        competence=dict(required_successes_each_panel=15, details=competence_rows),
        gaps=dict(descriptive_only=True, details=gap_details, execution_gate=False),
        feedback=dict(
            active=bool(positive_events),
            active_streams=sorted({row["stream"] for row in positive_events}),
            events=events,
            activity=activities,
            minimum_new_fraction_required=False,
            interpretation="No .5 fresh-data gate: accepted-identical experience can be stationary; fallback and genuinely changed admitted acquisition are reported separately.",
        ),
        missing=saved.missing,
        convergence_claim=False,
        superiority_claim=False,
        interpretation="All final criteria describe the bounded trajectory; none permits early scientific stopping. Mathematical convergence, equal-compute low-pass-cloning/DAgger superiority and second-body generalization are not established by these measurements.",
    )
