"""Render H5 metadata, saved scalar events and every recorded learning point.

Raw arrays, NPZs and checkpoints are never opened. External ledger/audit hashes
are retained as declarations from the hash-bound diagnostic extract; rendering
does not repeat that audit or certify an incomplete scientific contrast.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from ..provenance import sha, write

CURVE_FIELDS = (
    "update",
    "fit_update",
    "action_mse",
    "proximal_loss",
    "total_loss",
    "unclipped_gradient_norm",
)


def _load(path: Path, label: str, inputs: dict, limit: int = 16 * 1024**2) -> Any:
    """Hash precisely the bounded JSON payload consumed; never open raw arrays."""
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing or oversized saved JSON: " + str(path))
    payload = path.read_bytes()
    if len(payload) > limit:
        raise ValueError("Saved JSON grew beyond its read bound")
    inputs[label] = dict(
        file=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )

    def reject(value: str) -> None:
        raise ValueError("Nonfinite saved JSON constant: " + value)

    return json.loads(payload, parse_constant=reject)


def _number(value: Any, name: str) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite saved number: " + name)
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _point(tag: str) -> dict:
    match = re.fullmatch(
        r"s([0-2])_(parent_initial|parent|independent_child|selector_only_child|proposal_guided_child)",
        tag,
    )
    if match is None:
        raise ValueError("Unrecognized saved fitting point: " + tag)
    point = match[2]
    return dict(
        stream=int(match[1]),
        point=point,
        arm=point.removesuffix("_child") if point.endswith("_child") else "parent",
    )


def _episode_identity(tag: str) -> dict:
    match = re.fullmatch(
        r"(teacher|gap_teacher|primary|confirmation|gap)_s([0-2])_"
        r"(parent|independent|selector_only|proposal_guided)_([0-9]{2})",
        tag,
    )
    if match is None:
        raise ValueError("Unrecognized saved episode tag: " + tag)
    return dict(
        cohort=dict(teacher="acquisition", gap_teacher="gap_teacher", gap="gap_student").get(
            match[1], match[1]
        ),
        stream=int(match[2]),
        arm=match[3],
        role="teacher" if match[1] in ("teacher", "gap_teacher") else "student",
    )


def _saved_diagnostics(saved: dict, result: dict) -> dict[str, list[dict]]:
    """Join exact scalar events without re-predicting or decoding their arrays."""
    if saved.get("schema") != "walker-h5-saved-diagnostics-v1":
        raise ValueError("Unsupported saved-diagnostics schema")
    for name in ("source_ledger", "audit_receipt"):
        value = saved.get(name, {})
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("file"), str)
            or not value["file"]
            or not isinstance(value.get("sha256"), str)
            or re.fullmatch("[0-9a-f]{64}", value["sha256"]) is None
        ):
            raise ValueError("Saved diagnostics lack a source file/hash binding")
    if type(saved["source_ledger"].get("bytes")) is not int or saved["source_ledger"]["bytes"] < 0:
        raise ValueError("Saved ledger byte count is missing")
    events = saved.get("events")
    if not isinstance(events, list) or len(events) > 1000:
        raise ValueError("Saved diagnostics exceed the scalar-event roster bound")
    tables: dict[str, list[dict]] = dict(metrics=[], episodes=[], agreement=[])
    identifiers, metric_tags, episodes, agreements = set(), {}, {}, set()
    owners = {row["name"]: row["actor_hash"] for row in result.get("checkpoint_records", [])}
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("id"), str):
            raise ValueError("Saved event lacks its original ledger identity")
        if event["id"] in identifiers:
            raise ValueError("A saved ledger event was repeated")
        identifiers.add(event["id"])
        kind = event.get("event")
        if event.get("kind", kind) != kind:
            raise ValueError("Saved event kind differs")
        if kind == "policy_metric_complete":
            tag = event["tag"]
            if tag in metric_tags:
                raise ValueError("A saved metric tag was repeated")
            point, split = tag.rsplit("/", 1)
            identity = _point(point)
            if split not in ("train", "validation", "incoming", "endpoint"):
                raise ValueError("Unknown saved metric split")
            owner = f"s{identity['stream']}_parent" if split == "incoming" else point
            if owner in owners and event["actor_hash"] != owners[owner]:
                raise ValueError("Saved metric actor differs from its checkpoint point")
            row = dict(
                event_id=event["id"],
                tag=tag,
                **identity,
                split=split,
                dataset="held_trajectory"
                if split == "validation"
                else "historical_W1"
                if split == "train"
                else "own_acquisition",
                actor_hash=event["actor_hash"],
                rows=event["rows"],
                action_mse=_number(event["action_mse"], "action MSE"),
                reduction="mean_over_rows_and_six_actions",
                observations=_canonical(event["observations"]),
                targets=_canonical(event["targets"]),
            )
            if type(row["rows"]) is not int or row["rows"] < 1:
                raise ValueError("Saved metric rows must be a positive integer")
            metric_tags[tag] = row
            tables["metrics"].append(row)
        elif kind == "episode_complete":
            raw = _canonical(event["raw"])
            if raw in episodes or any(row["tag"] == event["tag"] for row in episodes.values()):
                raise ValueError("A saved episode identity or raw reference was repeated")
            row = dict(event_id=event["id"], tag=event["tag"], **_episode_identity(event["tag"]))
            for name in ("seed", "steps", "success", "survived", "complete"):
                row[name] = event[name]
            if any(type(row[name]) is not int or row[name] < 0 for name in ("seed", "steps")):
                raise ValueError("Saved episode seed and controls must be nonnegative integers")
            if any(type(row[name]) is not bool for name in ("success", "survived", "complete")):
                raise ValueError("Saved episode outcomes must be boolean")
            for name in ("cost", "mae", "terminal_cost", "terminal_adjustment", "absorbing_cost"):
                row[name] = _number(event[name], name)
            row["executed_prefix_cost"] = (
                row["cost"] - row["terminal_adjustment"] - row["absorbing_cost"]
            )
            row["raw_reference"] = raw
            episodes[raw] = row
            tables["episodes"].append(row)
        elif kind != "teacher_agreement_complete":
            raise ValueError("Unsupported scalar event: " + str(kind))
    for row in tables["metrics"]:
        if row["split"] == "incoming":
            endpoint = metric_tags.get(row["tag"].removesuffix("incoming") + "endpoint")
            if endpoint is not None and any(
                row[key] != endpoint[key] for key in ("rows", "observations", "targets")
            ):
                raise ValueError("Incoming and endpoint reconstruction use different saved data")
    for event in events:
        if event["event"] != "teacher_agreement_complete":
            continue
        raw = _canonical(event["episode_raw"])
        if raw not in episodes or raw in agreements:
            raise ValueError("Teacher agreement lacks one unique saved episode join")
        episode = episodes[raw]
        if episode["role"] != "teacher" or event["rows"] != episode["steps"]:
            raise ValueError("Teacher agreement rows differ from its executed episode")
        guide = owners.get(f"s{episode['stream']}_parent")
        if guide is not None and event["actor_hash"] != guide:
            raise ValueError("Teacher agreement actor differs from its common parent")
        agreements.add(raw)
        tables["agreement"].append(
            dict(
                event_id=event["id"],
                episode=episode["tag"],
                stream=episode["stream"],
                arm=episode["arm"],
                cohort=episode["cohort"],
                rows=event["rows"],
                actor_hash=event["actor_hash"],
                action_agreement=_number(event["action_agreement"], "teacher agreement"),
                reduction="mean_over_time_of_sum_six_squared_action_errors",
            )
        )
    return tables


def _outcomes(result: dict, episodes: list[dict]) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = dict(teachers=[], students=[], gaps=[])
    for stream, per_arm in result.get("teacher_summaries", {}).items():
        for arm, value in per_arm.items():
            tables["teachers"].append(dict(cohort="acquisition", stream=stream, arm=arm, **value))
    for cohort, per_stream in result.get("evaluations", {}).items():
        for stream, per_arm in per_stream.items():
            for arm, value in per_arm.items():
                tables["students"].append(dict(cohort=cohort, stream=stream, arm=arm, **value))
    for stream, per_arm in result.get("paired_teacher_student_gap", {}).items():
        for arm, value in per_arm.items():
            for role, key in (("teacher", "teachers"), ("student", "students")):
                tables[key].append(dict(cohort="paired_gap", stream=stream, arm=arm, **value[role]))
            if len(value["seeds"]) != len(value["child_minus_teacher_cost"]):
                raise ValueError("Saved paired-gap reset and difference rosters differ")
            for seed, difference in zip(
                value["seeds"], value["child_minus_teacher_cost"], strict=True
            ):
                tables["gaps"].append(
                    dict(
                        stream=stream,
                        arm=arm,
                        seed=seed,
                        child_minus_teacher_cost=difference,
                        scope=value.get("scope"),
                    )
                )
    for name in ("teachers", "students"):
        for row in tables[name]:
            row["summary_source"] = "worker_result"
            row["roster_status"] = "recorded_worker_summary"
    # An infrastructure error can prevent run_round from returning its summary.
    # Show only cohorts actually present in saved episode events; missing actors
    # or episodes are never filled with failed/zero observations.
    for role, key in (("teacher", "teachers"), ("student", "students")):
        grouped: dict[tuple, list[dict]] = {}
        for row in episodes:
            if row["role"] == role:
                grouped.setdefault((row["cohort"], str(row["stream"]), row["arm"]), []).append(row)
        for (cohort, stream, arm), rows in grouped.items():
            equivalent = "paired_gap" if cohort in ("gap_teacher", "gap_student") else cohort
            if any(
                row["cohort"] == equivalent and str(row["stream"]) == stream and row["arm"] == arm
                for row in tables[key]
            ):
                continue
            tables[key].append(
                dict(
                    cohort=cohort,
                    stream=stream,
                    arm=arm,
                    episodes=len(rows),
                    successes=sum(row["success"] for row in rows),
                    survived=sum(row["survived"] for row in rows),
                    controls=sum(row["steps"] for row in rows),
                    mean_cost=sum(row["cost"] for row in rows) / len(rows),
                    summary_source="saved_episode_events",
                    roster_status="observed_prefix_only",
                )
            )
    return tables


def _curves(root: Path, inputs: dict) -> list[dict]:
    rows = []
    for path in sorted((root / "learning").glob("*.json")):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Learning curve path escapes the recorded run")
        evidence = _load(path, str(path.relative_to(root)), inputs, 8 * 1024**2)
        curve = evidence.get("learning_curve")
        if (
            not isinstance(curve, list)
            or type(evidence.get("completed_updates")) is not int
            or evidence["completed_updates"] != len(curve)
        ):
            raise ValueError("Saved update count differs from its recorded curve")
        tag = path.stem.removesuffix("_partial")
        identity = _point(tag)
        for point in curve:
            row = dict(
                artifact=str(path.relative_to(root)),
                tag=tag,
                **identity,
                partial=path.stem.endswith("_partial") or evidence.get("failed") is True,
            )
            for name in CURVE_FIELDS:
                row[name] = _number(point[name], name)
                if name in ("update", "fit_update") and (
                    type(row[name]) is not int or row[name] < 1
                ):
                    raise ValueError("Saved update indices must be positive integers")
            rows.append(row)
    return rows


def _mechanisms(saved: dict | None) -> list[dict]:
    """Keep optional selection-event counts descriptive, without new gates."""
    if saved is None or "mechanism_summary" not in saved:
        return []
    value = saved["mechanism_summary"]
    if not isinstance(value, dict):
        raise ValueError("Saved mechanism summary must be grouped by stream and arm")
    rows = []
    for stream, per_arm in value.items():
        if not isinstance(per_arm, dict):
            raise ValueError("Saved mechanism arm roster is not a mapping")
        for arm, summary in per_arm.items():
            decisions = summary.get("decisions")
            if type(decisions) is not int or decisions < 0:
                raise ValueError("Saved mechanism decision count must be nonnegative")
            row = dict(stream=stream, arm=arm, decisions=decisions)
            for name in (
                "selected_origin_counts",
                "baseline_verification_status_counts",
                "guided_verification_status_counts",
            ):
                counts = summary.get(name)
                if (
                    not isinstance(counts, dict)
                    or any(
                        not isinstance(key, str) or type(count) is not int or count < 0
                        for key, count in counts.items()
                    )
                    or sum(counts.values()) != decisions
                ):
                    raise ValueError("Saved mechanism count map does not reconcile: " + name)
                row[name] = counts
            row["protected_guard_fallback_count"] = row["selected_origin_counts"].get(
                "protected_guard_fallback", 0
            )
            row["unique_selected_origins"] = len(row["selected_origin_counts"])
            rows.append(row)
    return rows


def _csv(path: Path, rows: list[dict], columns: Sequence[str]) -> None:
    names = list(dict.fromkeys([*columns, *(key for row in rows for key in row)]))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _canonical(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _table(title: str, rows: list[dict], columns: Sequence[str], filename: str) -> list[str]:
    text = ["", "## " + title, "", f"All saved rows: [{filename}]({filename}).", ""]
    if not rows:
        return text + ["Not measured or unavailable in supplied evidence; not zero."]
    text.extend(
        ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    )
    for row in rows:
        values = [str(row.get(key, "missing")).replace("|", "\\|") for key in columns]
        text.append("| " + " | ".join(values) + " |")
    return text


def _figures(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    # No Matplotlib import or image is needed for a metadata-only preparation.
    with tempfile.TemporaryDirectory(prefix="walker-h5-report-fonts-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        if previous is None:
            os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            streams = sorted({row["stream"] for row in rows})
            figure, axes = plt.subplots(
                len(streams),
                2,
                figsize=(11, 3.1 * len(streams)),
                squeeze=False,
                constrained_layout=True,
            )
            colors = dict(
                parent="#555555",
                independent="#335c81",
                selector_only="#cf5c36",
                proposal_guided="#37845a",
            )
            for index, stream in enumerate(streams):
                selected = [row for row in rows if row["stream"] == stream]
                for artifact in sorted({row["artifact"] for row in selected}):
                    curve = [row for row in selected if row["artifact"] == artifact]
                    label = curve[0]["point"] + (" (partial)" if curve[0]["partial"] else "")
                    for column, metric in enumerate(("action_mse", "unclipped_gradient_norm")):
                        axes[index, column].plot(
                            [row["update"] for row in curve],
                            [row[metric] for row in curve],
                            label=label,
                            color=colors[curve[0]["arm"]],
                            linewidth=1,
                            linestyle="--" if curve[0]["partial"] else "-",
                        )
                for column, label in enumerate(("Minibatch action MSE", "Unclipped gradient norm")):
                    axes[index, column].set(
                        title=f"Training stream {stream}",
                        xlabel="Recorded cumulative Adam update",
                        ylabel=label,
                    )
                    axes[index, column].legend(fontsize="small")
                    axes[index, column].grid(alpha=0.2)
            figure.suptitle("All recorded learning-curve points; minibatch loss is not walking")
            for suffix in ("png", "pdf"):
                figure.savefig(out / ("learning_curves." + suffix), dpi=150)
            plt.close(figure)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)


def render(run: Path, out: Path, *, diagnostics: Path | None = None) -> dict:
    started = time.perf_counter()
    run, out = run.resolve(), out.resolve()
    root = run / "run" if (run / "capsule.json").is_file() else run
    if out.is_relative_to(run):
        raise ValueError("Reports must remain outside frozen evidence")
    if out.exists():
        raise FileExistsError(out)
    inputs = {}
    config = _load(root / "config.json", "config.json", inputs)
    result_path = root / "worker_result.json"
    if not result_path.is_file():
        result_path = root / "result.json"
    result = _load(result_path, result_path.name, inputs)
    if result_path.name == "result.json" and any(
        (root / name).exists() for name in ("launch_ownership.json", "worker_exit.json", "evidence")
    ):
        result.update(
            status="launch_attempt_without_worker_result",
            execution_complete=False,
            work=dict(completion_unknown=True, authoritative_prefix="evidence/ledger.jsonl"),
        )
    plan = _load(root / "call_census.json", "call_census.json", inputs)
    saved = None
    measurements = dict(metrics=[], episodes=[], agreement=[])
    if diagnostics is not None:
        saved = _load(diagnostics.resolve(), "saved_diagnostics", inputs, 64 * 1024**2)
        measurements = _saved_diagnostics(saved, result)
    curves = _curves(root, inputs)
    outcomes = _outcomes(result, measurements["episodes"])
    mechanisms = _mechanisms(saved)
    out.mkdir(parents=True)
    text = [
        "# H5.0 preparation and execution status",
        "",
        f"Status: **{result.get('status')}**.",
        "",
        "The descriptive motion distance R, conditional action agreement D, and usefulness "
        "after a fixed2,000-update learning block have different roles. H5.0 holds D fixed "
        "and changes only its placement in the second search stream.",
        "",
        f"Launch authorized: `{config['launch_authorized']}`. "
        f"Useful-learning prerequisite satisfied: "
        f"`{config['eligibility']['useful_learning_prerequisite_satisfied']}`.",
        "",
        "| Full-roster reservation | Amount |",
        "|---|---:|",
        f"| Model controls | {plan['total']['model_controls']:,} |",
        f"| Live controls | {plan['total']['live_controls']:,} |",
        f"| Physics substeps | {plan['total']['physics_substeps']:,} |",
        f"| Worker neural rows | {plan['total']['worker_neural_rows']:,} |",
        f"| Independent audit neural rows | {plan['strong_audit_neural_rows']:,} |",
        f"| Adam updates | {plan['worker_optimizer_steps']:,} |",
        "",
        "These are prospective maxima, not claims of completed work. "
        "Preparation performs no inference, integration, fitting or acquisition.",
        "",
        "## Recorded work",
        "",
        "```json",
        json.dumps(result.get("work", {}), indent=2),
        "```",
        "",
        "## Launch blockers",
        "",
    ]
    text.extend("- " + value for value in result.get("launch_blockers", []))
    if result.get("schema") == "walker-teacher-proposal-round-v1":
        text.extend(
            [
                "",
                "## Recorded outcomes",
                "",
                "Worker claims require the separate full independent audit before admission.",
                "",
                "```json",
                json.dumps(
                    {
                        key: result.get(key)
                        for key in (
                            "teacher_summaries",
                            "teacher_admission",
                            "evaluations",
                            "primary_gate",
                            "confirmation_gate",
                            "paired_teacher_student_gap",
                            "confirmed_benefit",
                        )
                    },
                    indent=2,
                ),
                "```",
            ]
        )
    if result.get("status") == "launch_attempt_without_worker_result" or (
        result_path.name == "worker_result.json"
        and (result.get("execution_complete") is not True or result.get("error"))
    ):
        text.extend(
            [
                "",
                "**Incomplete execution.** Preserve the paid prefix and failure. This is "
                "not an audited scientific negative or evidence that the mechanism cannot help.",
                "",
                "```json",
                json.dumps(result.get("error"), indent=2),
                "```",
            ]
        )
    if result.get("status") == "teacher_competence_negative_no_child_fits":
        text.extend(
            [
                "",
                "The fixed teacher-competence stop prevented child fitting and standalone "
                "student evaluation. Useful learning was not measured.",
            ]
        )
    if saved is not None:
        text.extend(
            [
                "",
                "## Saved diagnostic provenance",
                "",
                "The supplied extract declares ledger and audit-receipt bindings. This renderer "
                "hashes the extract, not those external raw files, and does not repeat the audit.",
                "",
                "```json",
                json.dumps(
                    {key: saved[key] for key in ("source_ledger", "audit_receipt")}, indent=2
                ),
                "```",
            ]
        )
    summary_columns = (
        "cohort",
        "stream",
        "arm",
        "episodes",
        "successes",
        "survived",
        "controls",
        "mean_cost",
        "summary_source",
        "roster_status",
    )
    table_specs = (
        (
            "full_data_metrics.csv",
            measurements["metrics"],
            ("tag", "rows", "action_mse", "dataset"),
        ),
        (
            "teacher_action_agreement.csv",
            measurements["agreement"],
            ("episode", "rows", "action_agreement"),
        ),
        (
            "episode_outcomes.csv",
            measurements["episodes"],
            ("tag", "seed", "steps", "success", "survived", "cost"),
        ),
        ("teacher_outcomes.csv", outcomes["teachers"], summary_columns),
        ("student_outcomes.csv", outcomes["students"], summary_columns),
        ("paired_gap.csv", outcomes["gaps"], ("stream", "arm", "seed", "child_minus_teacher_cost")),
        ("learning_curves.csv", curves, ("artifact", "tag", *CURVE_FIELDS)),
    )
    mechanism_columns = (
        "stream",
        "arm",
        "decisions",
        "protected_guard_fallback_count",
        "unique_selected_origins",
        "baseline_verification_status_counts",
        "guided_verification_status_counts",
    )
    if mechanisms:
        table_specs += (("mechanism_summary.csv", mechanisms, mechanism_columns),)
    for filename, rows, columns in table_specs:
        _csv(out / filename, rows, columns)
    text.extend(
        _table(
            "Full-data reconstruction and held error",
            measurements["metrics"],
            table_specs[0][2],
            "full_data_metrics.csv",
        )
    )
    text.extend(
        [
            "",
            "Full-data action MSE averages over rows and six actions. Incoming/endpoint "
            "rows use the same arm's acquisition data; historical W1 and held-trajectory measurements "
            "remain separate. Minibatch loss is a different measurement.",
        ]
    )
    text.extend(
        _table("Teacher outcomes", outcomes["teachers"], summary_columns, "teacher_outcomes.csv")
    )
    text.extend(
        _table(
            "Standalone student outcomes",
            outcomes["students"],
            summary_columns,
            "student_outcomes.csv",
        )
    )
    text.extend(
        [
            "",
            "Longer retention without walking is not competent walking. Episode cost is "
            "complete cost. The episode CSV retains executed-prefix cost, terminal metadata, terminal "
            "adjustment and absorbing tail separately; the native fall terminal must not be charged "
            "twice. Missing post-fall observations remain missing.",
        ]
    )
    text.extend(
        _table(
            "Exactly paired teacher–student gaps",
            outcomes["gaps"],
            table_specs[5][2],
            "paired_gap.csv",
        )
    )
    text.extend(
        [
            "",
            "Only the saved paired-gap contexts are differenced. Acquisition teachers "
            "and primary students are not paired gaps.",
        ]
    )
    text.extend(
        _table(
            "Teacher-state conditional action agreement",
            measurements["agreement"],
            table_specs[1][2],
            "teacher_action_agreement.csv",
        )
    )
    if mechanisms:
        text.extend(
            _table(
                "Saved proposal-selection mechanism counts",
                mechanisms,
                mechanism_columns,
                "mechanism_summary.csv",
            )
        )
        text.extend(
            [
                "",
                "The CSV preserves every exact selected-origin count. These are counts of "
                "saved selection events; they alone do not prove the following control executed, "
                "that matched training data changed, or that any trajectory became teachable. "
                "The independent source/control audit and fixed-update walking outcomes remain "
                "separate evidence. No admission criterion is added by this summary.",
            ]
        )
    text.extend(
        [
            "",
            "Teacher agreement averages the sum of six squared action errors over "
            "executed time. Its action reduction differs from full-data MSE; neither establishes "
            "teachability.",
            "",
            "## Recorded learning curves",
            "",
            "Every saved update, including partial fitting records, is retained in "
            "[learning_curves.csv](learning_curves.csv). No smoothing, new evaluation or missing-update "
            "interpolation is performed.",
        ]
    )
    if curves:
        text.extend(
            [
                "",
                "![All recorded learning curves](learning_curves.png)",
                "",
                "[Vector figure](learning_curves.pdf).",
            ]
        )
    else:
        text.extend(["", "No recorded curve is available; missing curves are not zero loss."])
    (out / "report.md").write_text("\n".join(text) + "\n")
    _figures(curves, out)
    outputs = {
        str(path.relative_to(out)): dict(bytes=path.stat().st_size, sha256=sha(path))
        for path in sorted(out.iterdir())
        if path.is_file()
    }
    code = Path(__file__)
    receipt = dict(
        schema="walker-h5-readonly-report-v1",
        source_result_sha256=inputs[result_path.name]["sha256"],
        config_sha256=inputs["config.json"]["sha256"],
        source_sha256=sha(code),
        source_file=str(code),
        inputs=inputs,
        source_read_bytes=sum(value["bytes"] for value in inputs.values()),
        source_read_scope="Small source JSON payloads only; plotting/library/font reads are not instrumented.",
        code_hash_read_bytes=code.stat().st_size,
        output_hash_read_bytes=sum(value["bytes"] for value in outputs.values()),
        outputs=outputs,
        table_rows={filename: len(rows) for filename, rows, _ in table_specs},
        source_ledger_binding=None if saved is None else saved["source_ledger"],
        audit_receipt_binding=None if saved is None else saved["audit_receipt"],
        provenance_scope="External ledger/audit hashes are declared by the hash-bound extract; "
        "the renderer does not reopen, rehash or independently audit those raw files.",
        raw_arrays_decoded=0,
        checkpoints_loaded=0,
        network_forwards=0,
        physics_controls=0,
        optimizer_steps=0,
        wall_seconds=time.perf_counter() - started,
        timing_scope="Renderer through output hashing; receipt write is outside this inner timer.",
    )
    write(out / "receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--saved-diagnostics", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render(args.run, args.out, diagnostics=args.saved_diagnostics), indent=2))
    return 0
