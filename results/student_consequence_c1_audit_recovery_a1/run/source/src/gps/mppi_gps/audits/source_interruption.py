"""Versioned saved-only postmortem for the externally interrupted S1 attempt.

The original completed-study auditor is unchanged. This checker accepts only a
durable ledger prefix and hash-verified datasets read through an explicitly
bound recovery copy. It never repairs originals or executes a policy, optimizer
or simulator. An accepted scoped prefix is not a completed S1 comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import native, source, width
from .common import assert_independent_imports


class InterruptionAudit(source.SourceAudit):
    def __init__(self, run: Path, recovered_raw: Path, max_wall_seconds: float = 1800):
        super().__init__(run, run / "source")
        self.recovered_raw = recovered_raw.resolve()
        self.logical_eval = (run / "evidence/eval/raw.h5").resolve()
        self.started = time.perf_counter()
        self.max_wall_seconds = max_wall_seconds

    def need(self, condition, message):
        if time.perf_counter() - self.started > self.max_wall_seconds:
            raise TimeoutError("Postmortem saved-only audit time ceiling exhausted")
        super().need(condition, message)

    def group(self, path, ref, keys=None):
        logical = Path(path).resolve()
        if logical.parent.name == "evidence" and logical.parent.parent.name == "eval":
            logical = logical.parent.parent / "raw.h5"
        if logical == self.logical_eval:
            # No group enumeration: the truncated original's directory tail is
            # incomplete, while earlier directly named groups can remain valid.
            return native.Audit.group(self, self.recovered_raw, ref, keys)
        return super().group(logical, ref, keys)


def verify_recovery(a, run: Path, recovered_raw: Path, receipt: dict) -> dict:
    """Prove an exact original-byte prefix and zero-only appended copy tail."""
    original = run / "evidence/eval/raw.h5"
    recovered_raw = recovered_raw.resolve()
    a.need(not recovered_raw.is_relative_to(run.resolve()), "Recovery is outside immutable run")
    a.need(
        receipt["schema"] == "walker-s1-derived-container-recovery-v1", "Recovery receipt schema"
    )
    original_binding = a.bind(original, receipt["original_sha256"])
    recovered_binding = a.bind(recovered_raw, receipt["recovered_sha256"])
    a.need(
        original_binding["bytes"] == receipt["original_bytes"]
        and recovered_binding["bytes"] == receipt["recovered_bytes"],
        "Exact original/recovered sizes",
    )
    appended = receipt["appended_zero_bytes"]
    a.need(
        type(appended) is int
        and appended > 0
        and recovered_binding["bytes"] == original_binding["bytes"] + appended,
        "Only declared copy extension",
    )
    digest = hashlib.sha256()
    with recovered_raw.open("rb") as stream:
        remaining = original_binding["bytes"]
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            a.need(bool(block), "Complete recovered original-byte prefix")
            digest.update(block)
            remaining -= len(block)
        tail = stream.read()
    a.need(
        digest.hexdigest() == original_binding["sha256"],
        "Every original byte remains unchanged in copy",
    )
    a.need(
        len(tail) == appended and tail == b"\0" * appended,
        "Derived append contains zero bytes only",
    )
    return dict(
        original=original_binding,
        recovered=recovered_binding,
        original_file=str(original.resolve()),
        recovered_file=str(recovered_raw),
        recorded_original_file=receipt["original_file"],
        recorded_recovered_file=receipt["recovered_file"],
        appended_zero_bytes=appended,
        payload_acceptance="Only directly referenced arrays with exact saved shape, dtype and SHA-256 are accepted; padding is not a recovered scientific observation.",
    )


def parse_eval_prefix(a, path: Path) -> dict:
    """Recount the surviving prefix without synthesizing a runtime_close."""
    a.bind(path)
    counts, events, live = Counter(), defaultdict(list), defaultdict(list)
    pending_queries, pending_measures, episodes, resets = {}, {}, {}, {}
    neural_by_episode = Counter()
    parsed, last, trailing = 0, None, None
    with path.open("rb") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                remainder = stream.read()
                a.need(
                    not remainder and not line.endswith(b"\n"),
                    "Only an incomplete final ledger fragment may be unparsed",
                )
                trailing = dict(bytes=len(line), sha256=hashlib.sha256(line).hexdigest())
                break
            kind = row["event"]
            a.need(
                row["id"] == f"event-{parsed:09d}" and row["kind"] == kind,
                "Durable prefix event identity",
            )
            a.need(
                np.isfinite(row["wall_seconds"])
                and (last is None or row["wall_seconds"] >= last["wall_seconds"]),
                "Durable prefix chronology",
            )
            if parsed == 0:
                a.need(kind == "runtime_open", "Prefix begins at original runtime opening")
            a.need(
                kind != "runtime_close",
                "This explicit interruption checker requires an unclosed evaluation prefix",
            )
            parsed += 1
            last = row
            if kind == "query_requested":
                a.need(
                    row["mode"] == "live"
                    and row["requested_controls"] == 1
                    and row["requested_physics_substeps"] == 4,
                    "Only declared native evaluation controls",
                )
                pending_queries[row["id"]] = row
                counts.update(
                    requested_controls=1,
                    requested_live_controls=1,
                    requested_physics_substeps=4,
                    live_queries=1,
                )
            elif kind in ("query_complete", "query_error"):
                a.need(
                    row["request_id"] in pending_queries, "One saved native terminal per request"
                )
                request = pending_queries.pop(row["request_id"])
                a.need(
                    row["query_id"] == request["query_id"] and row["mode"] == request["mode"],
                    "Native prefix terminal owner",
                )
                completed = row["completed_controls"]
                a.need(
                    completed in (0, 1) and row["completed_physics_substeps"] == 4 * completed,
                    "Recorded native completion",
                )
                counts.update(
                    completed_controls=completed,
                    completed_live_controls=completed,
                    completed_physics_substeps=4 * completed,
                )
                if kind == "query_error":
                    unknown = row["unknown_partial_controls"]
                    a.need(
                        unknown in (0, 1) and unknown + completed == 1,
                        "Explicitly unresolved native call",
                    )
                    counts.update(
                        failed_queries=1,
                        unknown_partial_controls=unknown,
                        unknown_partial_physics_substeps=4 * unknown,
                    )
                else:
                    live[request["episode"]].append((request, row))
            elif kind == "measurement_requested":
                a.need(
                    row["measure_kind"] in ("neural_rows", "checkpoint_loads")
                    and type(row["amount"]) is int
                    and row["amount"] > 0,
                    "Only permitted evaluation measurements",
                )
                pending_measures[row["id"]] = row
                counts["measurement_requests"] += 1
                counts["requested_" + row["measure_kind"]] += row["amount"]
                events["measurement_requested"].append(row)
            elif kind in ("measurement_complete", "measurement_error"):
                a.need(
                    row["request_id"] in pending_measures, "One measurement terminal per request"
                )
                request = pending_measures.pop(row["request_id"])
                for key in ("measure_kind", "amount", "context", "purpose", "episode", "step"):
                    a.need(row.get(key) == request.get(key), "Recorded measurement owner " + key)
                prefix = "completed_" if kind == "measurement_complete" else "unknown_partial_"
                counts[prefix + row["measure_kind"]] += row["amount"]
                if kind == "measurement_error":
                    counts["failed_measurements"] += 1
                elif row["measure_kind"] == "neural_rows" and request.get("episode"):
                    neural_by_episode[request["episode"]] += row["amount"]
            elif kind == "reset_complete":
                a.need(row["tag"] not in resets, "No reset retry in durable prefix")
                resets[row["tag"]] = row["seed"]
                counts["resets"] += 1
            elif kind == "reset_error":
                counts["reset_errors"] += 1
            elif kind == "episode_complete":
                a.need(row["tag"] not in episodes, "Unique complete episode in prefix")
                episodes[row["tag"]] = row
                counts["episodes"] += 1
            elif kind in ("query_rejected", "measurement_rejected"):
                counts[
                    "rejected_queries" if kind == "query_rejected" else "rejected_measurements"
                ] += 1
            elif kind == "snapshot_restored":
                counts["restores"] += 1
            else:
                events[kind].append(row)
    a.need(last is not None, "At least one durable evaluation record")
    pending_native = sum(row["requested_controls"] for row in pending_queries.values())
    pending_work = Counter()
    for row in pending_measures.values():
        pending_work[row["measure_kind"]] += row["amount"]
    counts["queries"] = counts["live_queries"]
    counts["physics_frames"] = counts["completed_physics_substeps"]
    for kind in ("neural_rows", "backward_rows", "optimizer_steps"):
        counts[kind] = counts["completed_" + kind]
    return dict(
        path=path.parent,
        counts=counts,
        events=events,
        live=live,
        episodes=episodes,
        resets=resets,
        pending_queries=pending_queries,
        pending_measurements=pending_measures,
        pending_reserved_controls=pending_native,
        pending_reserved_measurements=dict(pending_work),
        neural_by_episode=neural_by_episode,
        parsed_records=parsed,
        last_record=last,
        unparsed_trailing_fragment=trailing,
    )


def observed_summary(episodes: list[dict], expected: int = 16) -> dict:
    """Observed denominators only; an absent attempt never becomes a failure."""
    count = len(episodes)
    return dict(
        episodes=count,
        expected_episodes=expected,
        cohort_complete=count == expected,
        tags=[row["tag"] for row in episodes],
        successes=sum(bool(row["success"]) for row in episodes),
        survived=sum(bool(row["survived"]) for row in episodes),
        mean_cost=float(np.mean([row["cost"] for row in episodes])) if count else None,
        costs=[row["cost"] for row in episodes],
        mean_steps=float(np.mean([row["steps"] for row in episodes])) if count else None,
        steps=[row["steps"] for row in episodes],
        mean_velocity_mae=float(np.mean([row["mae"] for row in episodes])) if count else None,
    )


def partial_attempt(a, stage: dict, tag: str, role: dict) -> dict:
    """Verify surviving native boundaries without inventing an episode terminal."""
    pairs = stage["live"].get(tag, [])
    observed_costs, unavailable = [], []
    previous = None
    last_done = None
    for index, (before, after) in enumerate(pairs):
        try:
            a.need(before["step"] == index, "Partial-attempt native control order")
            inputs = a.group(stage["path"] / "raw.h5", before["inputs"])
            outputs = a.group(stage["path"] / "raw.h5", after["result"])
            if previous is not None:
                for key in ("state", "warmstart", "integration"):
                    a.need(
                        np.array_equal(inputs[key], previous[key]),
                        "Partial-attempt full context continuity",
                    )
            state, action, post = inputs["state"], inputs["action"].astype(float), outputs["state"]
            a.need(
                state.shape == post.shape == (19,)
                and action.shape == (6,)
                and np.isfinite(state).all()
                and np.isfinite(post).all()
                and np.all(np.abs(action) <= 1),
                "Finite bounded partial native trace",
            )
            healthy = bool(0.8 < post[2] < 2 and -1 < post[3] < 1)
            a.need(
                last_done is not True and bool(outputs["done"]) == (not healthy),
                "Partial trace stops at native fall",
            )
            expected = (
                2 * abs(1.5 - post[10])
                + 0.5 * post[11] ** 2
                + 0.1 * post[3] ** 2
                + 5 * (not healthy)
                + 0.001 * np.sum(action * action)
                + 20 * (not healthy)
            )
            a.equal(outputs["cost"], expected, "Partial observed native cost/terminal once")
            a.need(
                np.array_equal(outputs["observation"], native.obs(post)),
                "Partial native post-action observation",
            )
            a.equal(post[0] - state[0], 0.008, "Partial native control spacing")
            previous, last_done = outputs, bool(outputs["done"])
            observed_costs.append(float(outputs["cost"]))
        except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
            unavailable.append(
                dict(
                    step=index,
                    query_id=before["query_id"],
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            # Later states cannot be joined across an unavailable prefix gap.
            break
    return dict(
        **role,
        tag=tag,
        seed=stage["resets"].get(tag),
        verified_steps=len(observed_costs),
        recorded_native_steps=len(pairs),
        raw_unavailable_steps=len(pairs) - len(observed_costs),
        observed_native_cost=float(sum(observed_costs)) if observed_costs else None,
        observed_native_fall=last_done,
        completed_neural_rows=stage["neural_by_episode"].get(tag, 0),
        unavailable_native_boundaries=unavailable,
        complete=False,
        success=None,
        survived=None,
        cost=None,
        complete_cost=None,
        terminal_cost=None,
        absorbing_cost=None,
        administrative_censoring="External host reboot; no episode_complete record. No missing-controls penalty or full-task outcome is imputed.",
    )


EPISODE_FIELDS = (
    "tag",
    "seed",
    "steps",
    "success",
    "survived",
    "mae",
    "cost",
    "complete",
    "terminal_cost",
    "terminal_adjustment",
    "absorbing_cost",
    "raw",
)


def derive_prefix(
    a, run: Path, cfg: dict, fit: dict, inventory: dict, payloads: dict, stage: dict
) -> dict:
    """Verify complete native attempts and distinguish every missing role."""
    order = cfg["evaluation_order"]
    planned = []
    for panel in ("primary", "confirmation"):
        for point in order:
            for index, seed in enumerate(cfg[panel + "_seeds"]):
                planned.append(
                    dict(tag=f"{panel}_{point}_{index:02d}", seed=seed, panel=panel, point=point)
                )
    a.need(
        len(planned) == 320 and len({row["tag"] for row in planned}) == 320,
        "Fixed 320-role interrupted roster",
    )
    a.need(
        list(stage["episodes"]) == [row["tag"] for row in planned[: len(stage["episodes"])]],
        "Durable complete episodes are the prescribed execution prefix",
    )
    roles = {row["tag"]: row for row in planned}
    started = set(stage["resets"]) | set(stage["live"])
    a.need(started <= set(roles), "No undeclared reset or native attempt")
    known, unavailable, reset_contexts = {}, [], {}
    for tag, event in stage["episodes"].items():
        role = roles[tag]
        try:
            a.need(event["seed"] == role["seed"], "Original paired seed role")
            _, arrays = a.episode(stage, tag)
            expected_hash = (
                inventory["reference"]["policy_hash"]
                if role["point"] == "reference"
                else payloads[role["point"]]["policy_hash"]
            )
            for info in arrays["infos"]["rows"]:
                a.prove(
                    info["controller"],
                    dict(policy_hash=expected_hash, controller="standalone_torque"),
                    "Verified prefix torque owner",
                )
            for request, _ in stage["live"][tag]:
                a.need(
                    request["context"]["point"] == role["point"]
                    and request["context"]["cohort"] == role["panel"],
                    "Verified prefix native actor/panel",
                )
            context = native.fingerprint(
                {key: arrays[key][0] for key in ("states", "warmstarts", "integrations")}
            )
            a.need(
                reset_contexts.setdefault(role["seed"], context) == context,
                "Every observed actor shares the complete reset context",
            )
            known[tag] = {key: event[key] for key in EPISODE_FIELDS}
        except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
            unavailable.append(dict(tag=tag, error_type=type(error).__name__, message=str(error)))
    evaluations = {panel: {} for panel in ("primary", "confirmation")}
    for panel in evaluations:
        for point in order:
            episodes = [
                known[row["tag"]]
                for row in planned
                if row["panel"] == panel and row["point"] == point and row["tag"] in known
            ]
            policy_hash = (
                inventory["reference"]["policy_hash"]
                if point == "reference"
                else payloads[point]["policy_hash"]
            )
            optimizer = None if point == "reference" else payloads[point]["optimizer_state"]
            evaluations[panel][point] = dict(
                policy_hash=policy_hash,
                optimizer_fingerprint=native.fingerprint(optimizer),
                episodes=episodes,
                summary=observed_summary(episodes),
            )
    for event in stage["events"]["source_actor_evaluated"]:
        record = evaluations[event["cohort"]][event["point"]]
        if record["summary"]["cohort_complete"]:
            a.prove(record["summary"], event["summary"], "Durable complete actor summary")
        else:
            a.need(
                bool(unavailable),
                "Actor completion is not inferred from incomplete retained episodes",
            )
    partials = [
        partial_attempt(a, stage, tag, {key: roles[tag][key] for key in ("point", "panel")})
        for tag in roles
        if tag in started and tag not in stage["episodes"]
    ]
    missing = []
    for role in planned:
        tag = role["tag"]
        status = (
            "verified_complete"
            if tag in known
            else "unverified_raw"
            if tag in stage["episodes"]
            else "partial"
            if tag in started
            else "not_started"
        )
        missing.append(dict(**role, status=status))
    counts = Counter(row["status"] for row in missing)
    return dict(
        evaluations=evaluations,
        partial_attempts=partials,
        missing_roster=missing,
        roster_counts=dict(planned=320, **counts),
        unavailable_complete_episodes=unavailable,
    )


def audit_interruption(a, args) -> tuple[dict, dict]:
    run = args.run.resolve()
    a.bind(run / "config.json", args.config_sha)
    a.bind(run / "input_manifest.json", args.seal_sha)
    cfg = a.load(run / "config.json")
    seal = a.load(run / "input_manifest.json")
    manifest = a.load(run / "manifest.json")
    a.need(
        cfg["schema"] == "walker-student-source-s1-v1"
        and manifest["config_sha256"] == seal["config_sha256"] == args.config_sha
        and seal["source_manifest_sha256"] == native.sha(run / "manifest.json"),
        "Unchanged original S1 source/configuration seal",
    )
    for row in manifest["files"]:
        a.bind(width.contained(a, run / "source", row["file"]), row["sha256"])
    for relative, binding in seal["files"].items():
        actual = a.bind(width.contained(a, run, relative), binding["sha256"])
        a.need(actual["bytes"] == binding["bytes"], "Unchanged original input seal")
    for module in (source, width, native):
        active = Path(module.__file__)
        original = run / "source/src/gps/mppi_gps/audits" / active.name
        a.need(
            native.sha(active) == native.sha(original),
            "Original completed-study audit helper remains unchanged",
        )
    a.bind(args.original_manifest, args.original_manifest_sha)
    original_manifest = a.load(args.original_manifest)
    actual_paths = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()}
    a.need(
        actual_paths == {row["file"] for row in original_manifest["files"]},
        "No original run file created, removed or replaced",
    )
    for row in original_manifest["files"]:
        record = a.bind(width.contained(a, run, row["file"]), row["sha256"])
        a.need(record["bytes"] == row["bytes"], "Exact interrupted original file size")
    a.need(
        all(
            not (run / name).exists()
            for name in ("result.json", "eval_exit.json", "owner_exit.json")
        ),
        "Original missing final/exit files remain absent",
    )
    proposal = a.load(run / "source/configs/studies/student_source_s1_proposal.json")
    for key, value in proposal.items():
        if key not in ("schema", "status", "stopping"):
            a.prove(cfg[key], value, "Preserved original S1 proposal " + key)
    data, norm, inventory = source.audit_inputs(a, run, cfg)
    fit = a.load(run / "fit_result.json")
    a.need(
        fit["execution_complete"] is True
        and fit["error"] is None
        and fit["config_sha256"] == args.config_sha
        and fit["input_manifest_sha256"] == args.seal_sha,
        "Closed unmodified six-fit phase",
    )
    fit_exit = a.load(run / "fit_exit.json")
    a.need(
        fit_exit["exit_code"] == 0
        and fit_exit["fit_result_sha256"] == native.sha(run / "fit_result.json")
        and fit_exit["timeout"] is False,
        "Original external fit completion",
    )
    fit_stage = source.stage_ledger(a, run / "evidence/fit", fit["work"])
    a.prove(fit["platform"], cfg["expected_runtime"]["fit"], "Original actual CUDA fit runtime")
    a.need(
        fit["platform"]["deterministic_algorithms"] is True
        and fit["platform"]["allow_tf32"] is False,
        "Original deterministic fit settings",
    )
    a.prove(
        fit["teacher_outcomes"], inventory["teachers"], "Exact original selected teacher outcomes"
    )
    fit_details = source.audit_fit(a, run, cfg, fit, data, norm)
    a.need(
        fit_stage["counts"]["completed_neural_rows"] == 3180000
        and fit_stage["counts"]["completed_backward_rows"] == 3072000
        and fit_stage["counts"]["completed_optimizer_steps"] == 12000
        and fit_stage["counts"]["requested_controls"] == 0,
        "Exact closed fit and diagnostic work",
    )
    recovery_receipt = a.load(args.recovery_receipt)
    recovery = verify_recovery(a, run, args.recovered_raw, recovery_receipt)
    evidence = a.load(args.interruption_evidence)
    a.need(
        evidence["launch"]["config_sha256"] == args.config_sha
        and evidence["launch"]["input_manifest_sha256"] == args.seal_sha
        and evidence["monotonic"]
        < evidence["launch"]["deadline_monotonic"] - cfg["budgets"]["wall_seconds"],
        "Recorded boot/monotonic reset evidence",
    )
    a.need(
        set(evidence["missing"]) >= {"owner_exit.json", "eval_exit.json", "result.json"},
        "Interruption is not a fabricated ordinary phase exit",
    )
    stage = parse_eval_prefix(a, run / "evidence/eval/ledger.jsonl")
    a.need(
        len(stage["episodes"]) == args.expected_complete_episodes,
        "Recorded complete-episode prefix census",
    )
    loads = [
        row
        for row in stage["events"]["measurement_requested"]
        if row["measure_kind"] == "checkpoint_loads"
    ]
    a.need(
        len(loads) == 10
        and [row["context"]["point"] for row in loads] == cfg["evaluation_order"]
        and stage["counts"]["completed_checkpoint_loads"] == 10,
        "Original CPU ten-actor load roster",
    )
    recovered = derive_prefix(a, run, cfg, fit, inventory, fit_details["payloads"], stage)
    counts = stage["counts"]
    a.need(
        counts["requested_controls"] <= cfg["budgets"]["live_controls"]
        and counts["requested_physics_substeps"] <= cfg["budgets"]["physics_substeps"]
        and 3180000 + counts["requested_neural_rows"] <= cfg["budgets"]["neural_rows"],
        "Recorded-prefix work remains below original ceilings",
    )
    a.need(
        not counts["requested_model_controls"]
        and not counts["requested_backward_rows"]
        and not counts["requested_optimizer_steps"],
        "No new teacher or learning phase in recorded evaluation",
    )
    fit_counts = fit_stage["counts"]
    combined = {key: fit_counts[key] + counts[key] for key in set(fit_counts) | set(counts)}
    combined.update(
        snapshots=None,
        wall_seconds=None,
        wall_seconds_lower_bound=fit["wall_seconds"] + stage["last_record"]["wall_seconds"],
    )
    eval_work = dict(
        counts,
        snapshots=None,
        wall_seconds=None,
        last_durable_wall_seconds=stage["last_record"]["wall_seconds"],
    )
    unavailable = recovered["unavailable_complete_episodes"]
    summary = dict(
        schema="walker-student-source-s1-interrupted-summary-v1",
        audit_complete=True,
        integrity_status="valid_scoped_prefix"
        if not unavailable
        else "incomplete_retained_evidence",
        execution_complete=False,
        comparison_complete=False,
        status="interrupted_infrastructure",
        error=dict(
            type="ExternalHostReboot",
            message="The original owner and evaluation ended without final exits or result. This derived postmortem does not resume or complete S1.",
        ),
        config_sha256=args.config_sha,
        input_manifest_sha256=args.seal_sha,
        fit_result_sha256=native.sha(run / "fit_result.json"),
        points=fit["points"],
        fits=fit["fits"],
        samplers=fit["samplers"],
        teacher_outcomes=fit["teacher_outcomes"],
        platform=dict(
            fit=fit["platform"],
            eval=None,
            eval_receipt_availability="Final evaluation platform dictionary was not saved; frozen preflight and launch inputs remain available.",
        ),
        gates=dict(
            withheld=True,
            reason="The required unconditional two-panel 320-attempt comparison did not finish. Missing attempts are not failures, and no full S1 source-benefit or child-competence conclusion is assigned.",
        ),
        phase_work=dict(fit=fit["work"], eval_known_durable_prefix=eval_work),
        work=dict(
            known_durable_prefix=combined,
            unknown_unrecorded_tail=dict(
                status="unresolved_not_zero",
                explanation="An abrupt reboot prevented final counters and exits. Pending registered operations are listed separately; even an empty pending set does not certify that the durable prefix captures all pre-reboot work.",
            ),
        ),
        wall_seconds=None,
        observed_wall_seconds_lower_bound=combined["wall_seconds_lower_bound"],
        interruption_evidence=evidence,
        recovery=recovery,
        durable_prefix=dict(
            parsed_records=stage["parsed_records"],
            last_record=stage["last_record"],
            unparsed_trailing_fragment=stage["unparsed_trailing_fragment"],
            pending_registered_controls=stage["pending_reserved_controls"],
            pending_registered_measurements=stage["pending_reserved_measurements"],
        ),
        audit_receipt=dict(
            file="receipt.json", schema="walker-student-source-s1-interruption-audit-v1"
        ),
        **recovered,
    )
    findings = dict(
        comparison_complete=False,
        closed_fits=6,
        fixed_actors=9,
        paired_initial_proofs=6,
        diagnostic_forward_rows=108000,
        verified_complete_episodes=len(stage["episodes"]) - len(unavailable),
        complete_episode_records=len(stage["episodes"]),
        partial_attempts=len(recovered["partial_attempts"]),
        native_controls_recorded=counts["completed_controls"],
        pending_registered_controls=stage["pending_reserved_controls"],
        unknown_unrecorded_tail="unresolved_not_zero",
        original_files=len(original_manifest["files"]),
        recovery_copy_used=True,
        unavailable_complete_episodes=unavailable,
    )
    return summary, findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--seal-sha", required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--original-manifest-sha", required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--recovered-raw", type=Path, required=True)
    parser.add_argument("--interruption-evidence", type=Path, required=True)
    parser.add_argument("--expected-complete-episodes", type=int, default=195)
    args = parser.parse_args(argv)
    run, out = args.run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        parser.error("Postmortem output must remain outside the immutable original run")
    out.mkdir(parents=True, exist_ok=False)
    a = InterruptionAudit(run, args.recovered_raw)
    started = time.perf_counter()
    error, summary, findings = None, {}, {}
    try:
        a.bind(Path(__file__))
        summary, findings = audit_interruption(a, args)
        if summary["unavailable_complete_episodes"]:
            raise AssertionError(
                "At least one declared complete episode is not fully hash/native verified"
            )
        for path, before in list(a.files.items()):
            a.need(
                a.bind(Path(path)) == before,
                "Original or recovery input unchanged during postmortem",
            )
        for path, (_, size, mtime) in a.hdfs.items():
            a.need(
                (path.stat().st_size, path.stat().st_mtime_ns) == (size, mtime),
                "Read-only HDF input unchanged",
            )
        assert_independent_imports(a)
        forbidden = ("src.gps.mppi_gps.source_inputs", "src.gps.mppi_gps.reporting")
        a.need(
            not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules
                for prefix in forbidden
            ),
            "No production source preparation or reporting import",
        )
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if not summary:
            summary = dict(
                schema="walker-student-source-s1-interrupted-summary-v1",
                audit_complete=False,
                integrity_status="failed",
                execution_complete=False,
                comparison_complete=False,
                status="postmortem_failed",
                config_sha256=args.config_sha,
                input_manifest_sha256=args.seal_sha,
                gates=dict(
                    withheld=True,
                    reason="The scoped postmortem could not verify its required inputs/evidence.",
                ),
                error=error,
            )
        else:
            summary["integrity_status"] = "incomplete_retained_evidence"
    finally:
        for handle, _, _ in a.hdfs.values():
            handle.close()
    summary_path = out / "interruption_summary.json"
    with summary_path.open("x") as stream:
        json.dump(native.plain(summary), stream, indent=2, allow_nan=False)
        stream.write("\n")
    receipt = dict(
        schema="walker-student-source-s1-interruption-audit-v1",
        passed=error is None,
        error=error,
        comparison_complete=False,
        summary_sha256=native.sha(summary_path),
        checks=a.checks,
        bound_files=len(a.files),
        hashed_arrays=len(a.arrays),
        wall_seconds=time.perf_counter() - started,
        wall_ceiling_seconds=a.max_wall_seconds,
        new_controls=0,
        new_neural_rows=0,
        new_backward_rows=0,
        new_optimizer_steps=0,
        scope="Separate infrastructure-interruption postmortem: original source/input/fit proofs, durable unclosed evaluation prefix, hash-verified complete native episodes through a separate zero-extended recovery copy, censored partial attempt and unresolved work tail. No full S1 gate/competence result; no network, optimizer or simulator arithmetic; no original evidence repair.",
    )
    for name, value in (
        ("receipt.json", receipt),
        ("findings.json", findings),
        ("file_bindings.json", a.files),
        ("array_bindings.json", a.arrays),
    ):
        with (out / name).open("x") as stream:
            json.dump(native.plain(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(receipt))
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
