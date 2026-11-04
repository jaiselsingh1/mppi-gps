"""Independent L3.1 validation-amendment audit; no producer or simulator imports.

L3.0's original cross-group check remains failed. This amendment certifies the
same float32 batch computation as recorded trajectory scoring, while retaining
cross-group differences as measurements. It does not certify the old parity
bound, change a scientific outcome, or permit unaccounted interrupted work.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from . import local_coupling_multiround as legacy
from .local_coupling_learning import LearningAudit, _closed_job, audit_acquisition
from .trajectory_pi import (
    AuditFailure,
    _plain,
    audit_ledger,
    load_arrays,
    observations,
    typed_fingerprint,
)

COMPARISON_VERSION = "MATCHED_GROUP_L3_1"
ACTION_ATOL = 1e-6
STATE_ATOL = 1e-10
COST_ATOL = 1e-9


def _value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def matching_group_inputs(bank: dict, replay: dict, candidate: int) -> tuple[np.ndarray, int, int]:
    """Independently reconstruct the original complete valid-prefix grouping."""
    posts, actions = np.asarray(bank["posts"]), np.asarray(bank["actions"])
    if posts.ndim != 3 or posts.shape[-1] != 19 or actions.shape != (*posts.shape[:2], 6):
        raise AuditFailure("Complete recorded candidate state/action bank required")
    k, h = posts.shape[:2]
    initial = np.asarray(bank["initial_state"])
    if initial.shape != (19,) or not 0 <= candidate < k:
        raise AuditFailure("Valid candidate and fixed initial state required")
    use = np.asarray(bank["mask"], dtype=bool) & np.isfinite(bank["C"])[:, None]
    if use.shape != (k, h):
        raise AuditFailure("Recorded valid-prefix grouping shape differs")
    pre = np.concatenate((np.broadcast_to(initial, (k, 1, 19)), posts[:, :-1]), axis=1)
    grouped = observations(pre[use])
    n = len(replay["actions"])
    offset = int(use[:candidate].sum())
    if n != int(use[candidate].sum()) or n <= 0:
        raise AuditFailure("Replay must retain the whole executable candidate prefix")
    if replay["observations"].shape != (n + 1, 17):
        raise AuditFailure("Actual N+1 replay observations required")
    native = replay["observations"][:-1]
    if not np.array_equal(native, observations(replay["states"][:-1])):
        raise AuditFailure("Native replay inputs must be exact pre-action observations")
    grouped[offset : offset + n] = native
    return grouped, offset, n


def audit_matched_predictions(
    audit: LearningAudit,
    *,
    payload: dict,
    bank: dict,
    replay: dict,
    prediction: dict,
    row: dict,
    charge_matching_rows: bool = True,
) -> tuple[dict, int]:
    """Check actual small/full grouped outputs with independent deployed math.

    The returned extra count belongs to the acquisition's scientific receipt
    only for new episodes. A certificate's new forwards have their own receipt,
    so its adopted original episode returns zero additional science rows.
    """
    audit.need(row.get("comparison_version") == COMPARISON_VERSION, "Explicit amended comparison")
    audit.need(
        row.get("cross_group_is_diagnostic") is True,
        "Cross-group bound is not retroactively passed",
    )
    candidate = int(row["candidate"])
    grouped, offset, n = matching_group_inputs(bank, replay, candidate)
    native = replay["observations"][:-1]
    audit.equal(prediction["observations"], native, "Actual native pre-action prediction inputs")
    audit.need(
        int(_value(prediction["replacement_offset"])) == offset,
        "Exact candidate replacement offset",
    )
    audit.need(
        int(_value(prediction["matching_group_rows"])) == len(grouped),
        "Original matching group size",
    )
    audit.need(int(_value(prediction["candidate"])) == candidate, "Exact numerical candidate owner")
    audit.need(
        _value(prediction["comparison_version"]) == COMPARISON_VERSION,
        "Prediction comparison version",
    )
    digest = typed_fingerprint(grouped)
    audit.need(
        _value(prediction["matching_group_observation_fingerprint"]) == digest,
        "Whole matched grouping includes the actual native prefix",
    )
    audit.need(
        row["matching_group_observation_fingerprint"] == digest,
        "Manifest matching-input fingerprint",
    )
    audit.need(
        row["native_prefix_rows"] == n and row["matching_group_rows"] == len(grouped),
        "Actual native and matching inference row census",
    )
    audit.need(
        row["added_matching_forward_rows"] == len(grouped),
        "Additional grouped inference is declared",
    )
    source_expected = bank["predictions"][offset : offset + n]
    audit.equal(
        prediction["expected_predictions"], source_expected, "Exact original bank prediction slice"
    )
    prefix_expected = audit.predict(payload, native).astype(np.float64)
    group_expected = audit.predict(payload, grouped).astype(np.float64)
    audit.close(
        prediction["predictions"],
        prefix_expected,
        "Actual native-prefix action computation",
        atol=ACTION_ATOL,
    )
    audit.close(
        prediction["matching_group_predictions"],
        group_expected,
        "Actual original-group action computation",
        atol=ACTION_ATOL,
    )
    matched = group_expected[offset : offset + n]
    audit.close(
        prediction["matched_predictions"],
        matched,
        "Saved matching candidate slice",
        atol=ACTION_ATOL,
    )
    # The amended gate compares the same call shape/order and native inputs.
    audit.close(
        matched,
        source_expected,
        "Matched-group native versus recorded scoring action",
        atol=ACTION_ATOL,
    )
    stored_prefix = prediction["predictions"]
    stored_matched = prediction["matched_predictions"]
    native_d = float(0.5 * np.sum((replay["actions"] - stored_prefix) ** 2))
    matched_d = float(0.5 * np.sum((replay["actions"] - stored_matched) ** 2))
    cross_error = float(np.max(np.abs(stored_prefix - source_expected)))
    matched_error = float(np.max(np.abs(stored_matched - source_expected)))
    tolerance = 1e-10 + 2 * replay["actions"].size * ACTION_ATOL
    for saved in (prediction["D"], row["D"]):
        audit.close(saved, native_d, "Recorded native-prefix D", atol=1e-10)
    for saved in (prediction["matched_D"], row["matched_D"]):
        audit.close(saved, matched_d, "Recorded matched-group D", atol=1e-10)
    audit.close(matched_d, bank["D"][candidate], "Matched native/scoring D", atol=tolerance)
    for saved in (prediction["D_tolerance"], row["D_tolerance"]):
        audit.close(saved, tolerance, "Original propagated D tolerance retained", atol=0)
    for saved in (
        prediction["action_max_error"],
        row["action_max_error"],
        row["cross_group_action_max_error"],
    ):
        audit.close(
            saved, cross_error, "Cross-group error remains a measured discrepancy", atol=1e-15
        )
    for saved in (prediction["matched_action_max_error"], row["matched_action_max_error"]):
        audit.close(saved, matched_error, "Matched-group numerical error", atol=1e-15)
    audit.need(matched_error <= ACTION_ATOL, "Amended matched-group per-action limit")
    return dict(
        comparison_version=COMPARISON_VERSION,
        action_max_error=cross_error,
        cross_group_action_max_error=cross_error,
        matched_action_max_error=matched_error,
        D_tolerance=tolerance,
        D_native=native_d,
        D_matched=matched_d,
        D_matched_error=abs(matched_d - float(bank["D"][candidate])),
        native_prefix_rows=n,
        matching_group_rows=len(grouped),
        original_cross_group_bound_passed=cross_error <= ACTION_ATOL,
    ), len(grouped) if charge_matching_rows else 0


def original_failed_ledger(audit: LearningAudit, directory: Path, original: dict) -> dict:
    """Known completed external calls remain FAILED under the original checker."""
    ledger = audit_ledger(audit, directory)
    audit.need(
        original.get("status") == "FAILED_STOPPED"
        and original.get("error")
        == "ValueError('Candidate/native deployed reconstruction mismatch')",
        "Only the bound numeric-checker failure can receive a completion certificate",
    )
    audit.need(
        ledger["receipt"]["status"] == "FAILED" and ledger["failed"] == ledger["rejected"] == 0,
        "Original failure remains a failed closed phase, not an external-call failure",
    )
    audit.need(
        not any(k.startswith("unknown_partial_") and v for k, v in ledger["counts"].items()),
        "Adopted physical work has no unknown external-call prefix",
    )
    for key, value in ledger["counts"].items():
        if key.startswith("requested_"):
            audit.need(
                ledger["counts"].get("completed_" + key.removeprefix("requested_"), 0) == value,
                "Every adopted external call completed before the predicate failure",
            )
    audit.equal(original["work"], ledger["counts"], "Original failed receipt census is unchanged")
    return ledger


def interrupted_ledger_counts(data: bytes) -> dict:
    """Bound paid interruption work without inventing completed external calls.

    A complete requested record precedes each call. Pending/failed requests are
    charged in full against the ceiling and remain unknown in the completion
    census. A truncated JSON record is not silently accepted as usable evidence.
    """
    if not data or not data.endswith(b"\n"):
        raise AuditFailure("Interrupted ledger must retain a complete durable record prefix")
    rows = []
    for line in data.splitlines():
        try:
            rows.append(json.loads(line))
        except (ValueError, TypeError) as error:
            raise AuditFailure("Corrupt complete interrupted ledger record") from error
    pending = {}
    finished = set()
    requested = {}
    completed = {}
    unknown = {}
    closed = False
    for sequence, row in enumerate(rows):
        if row.get("id") != sequence:
            raise AuditFailure("Interrupted ledger record order is not complete")
        event = row.get("event")
        if closed:
            raise AuditFailure("Interrupted ledger contains work after closure")
        if event == "closed":
            closed = True
            continue
        if event == "rejected":
            raise AuditFailure("Budget rejection is not an externally interrupted call")
        if event == "requested":
            charges = row.get("charges")
            if not isinstance(charges, dict) or any(
                type(v) is not int or v < 0 for v in charges.values()
            ):
                raise AuditFailure("Explicit nonnegative interruption reservations required")
            pending[row["id"]] = dict(charges)
            for unit, value in charges.items():
                requested[unit] = requested.get(unit, 0) + value
        elif event in ("completed", "failed"):
            token = row.get("request_id")
            if token not in pending or token in finished or row.get("charges") != pending[token]:
                raise AuditFailure("Interruption completion lacks its unique actual request")
            finished.add(token)
            target = completed if event == "completed" else unknown
            for unit, value in pending[token].items():
                target[unit] = target.get(unit, 0) + value
    if closed:
        raise AuditFailure("Closed jobs are not charged again as interrupted prefixes")
    outstanding = []
    for token, charges in pending.items():
        if token not in finished:
            outstanding.append(token)
            for unit, value in charges.items():
                unknown[unit] = unknown.get(unit, 0) + value
    counts = {}
    for prefix, values in (
        ("requested_", requested),
        ("completed_", completed),
        ("unknown_partial_", unknown),
    ):
        counts.update({prefix + unit: value for unit, value in values.items()})
    for unit, amount in requested.items():
        if completed.get(unit, 0) + unknown.get(unit, 0) != amount:
            raise AuditFailure("Interrupted upper-bound census does not close")
    return dict(
        counts=counts,
        records=len(rows),
        outstanding_requests=outstanding,
        requested_is_conservative_paid_upper_bound=True,
        closed=False,
    )


def sum_counts(*rows: dict) -> dict:
    """Add integer resource censuses, retaining declared zero fields."""
    result = {}
    for row in rows:
        for key, value in row.items():
            if type(value) is not int or value < 0:
                raise AuditFailure("Resource census requires nonnegative integer amounts")
            result[key] = result.get(key, 0) + value
    return result


def pending_overhead(data: bytes) -> tuple[list[dict], dict]:
    """Only unresolved reservations are overhead; known prefix work is reused."""
    census = interrupted_ledger_counts(data)
    rows = [json.loads(line) for line in data.splitlines()]
    pending = [rows[index] for index in census["outstanding_requests"]]
    if any(row.get("event") == "failed" for row in rows):
        raise AuditFailure("Recovery admits interruption, not a failed external call")
    charges = sum_counts(*(row["charges"] for row in pending))
    counts = {
        prefix + key: value
        for prefix in ("requested_", "unknown_partial_")
        for key, value in charges.items()
    }
    return pending, counts


def renumber_events(rows: list[dict], offset: int = 0, *, source: str | None = None) -> list[dict]:
    """Independently remap durable requests when combining saved ledger phases."""
    mapping = {row["id"]: index + offset for index, row in enumerate(rows)}
    result = []
    for index, row in enumerate(rows):
        value = dict(row, id=index + offset)
        if source is not None:
            value.update(continuation_source=source, source_id=row["id"])
        if "request_id" in row:
            if row["request_id"] not in mapping:
                raise AuditFailure("Combined ledger contains an unbound resolution")
            value["request_id"] = mapping[row["request_id"]]
        result.append(value)
    return result


def derive_combined_events(prefix: bytes, suffix: bytes) -> tuple[list[dict], dict]:
    """Rebuild the logical ledger; retain excluded paid requests separately."""
    pending, overhead = pending_overhead(prefix)
    exclude = {row["id"] for row in pending}
    first = [json.loads(line) for line in prefix.splitlines()]
    first = renumber_events(
        [row for row in first if row["id"] not in exclude], source="immutable_prefix"
    )
    second = [json.loads(line) for line in suffix.splitlines()]
    if not second or second[-1].get("event") != "closed":
        raise AuditFailure("Raw resumed suffix needs its actual closed receipt")
    # The final derived closure has a combined receipt and is checked separately.
    if any(row.get("event") == "closed" for row in second[:-1]):
        raise AuditFailure("Only one actual suffix closure is permitted")
    second = renumber_events(second[:-1], len(first), source="new_suffix")
    return first + second, overhead


class ContinuationReview:
    """Explicit amendment hooks; no old evidence or legacy audit code is rewritten."""

    def __init__(self, audit: LearningAudit):
        self.audit = audit
        self.failed_receipts: dict[Path, dict] = {}
        self.certificate_receipts: set[Path] = set()
        self.resumed: dict[Path, dict] = {}
        self.original_failures: list[dict] = []
        self.native_checks: list[dict] = []

    def certificate(self, manifest_path: Path, adopted: dict) -> Path:
        """Require a separately paid certificate without laundering FAILED status."""
        a = self.audit
        original_path = a.bound(adopted["original_validation_failure"])
        original = a.json(original_path)
        directory = original_path.parent
        ledger = original_failed_ledger(a, directory, original)
        original_receipt = a.bound(adopted["original_failed_receipt"])
        a.need(original_receipt == directory / "receipt.json", "Original failed receipt owner")
        for key, filename in (("original_job", "job.json"), ("original_ledger", "ledger.jsonl")):
            a.need(a.bound(adopted[key]) == directory / filename, "Original failed source: " + key)
        completion_path = a.bound(adopted["validation_completion"])
        proof = a.json(completion_path)
        a.need(
            proof.get("schema") == "local-coupling-numeric-completion-1"
            and proof.get("status") == "COMPLETED"
            and proof.get("comparison_version") == COMPARISON_VERSION
            and proof.get("new_physics") is False
            and proof.get("new_fitting") is False,
            "Explicit completed numerical-only certificate",
        )
        for key in ("new_native_controls", "new_model_controls", "new_optimizer_steps"):
            a.need(proof[key] == 0, "Certificate cannot repeat science: " + key)
        for proof_key, expected in (
            ("original_manifest", adopted["original_validation_failure"]),
            ("original_job", adopted["original_job"]),
            ("original_receipt", adopted["original_failed_receipt"]),
            ("original_ledger", adopted["original_ledger"]),
            ("original_trace", adopted["trace"]),
            ("original_bank_stats", adopted["bank_stats"]),
            ("original_error", original["error"]),
            ("actor_hash", original["actor_hash"]),
            ("episode_outcome", original["outcome"]),
            ("native_candidate_checks", adopted["native_candidate_checks"]),
            ("original_work_counts", original["work"]),
        ):
            a.equal(proof[proof_key], expected, "Completion binds original/result: " + proof_key)
        certificate_job = a.json(completion_path.parent / "job.json")
        a.equal(
            certificate_job["original_manifest"],
            adopted["original_validation_failure"],
            "Certificate requested the exact failed result",
        )
        a.equal(certificate_job["checkpoint"], original["checkpoint"], "No replacement actor")
        a.equal(
            certificate_job["config"],
            a.json(directory / "job.json")["config"],
            "Certificate retains original scientific config",
        )
        changed = {"status", "error", "trace", "bank_stats", "native_candidate_checks"}
        for key, value in original.items():
            if key not in changed:
                a.equal(adopted[key], value, "Adoption preserves original field: " + key)
        a.need(adopted["status"] == "COMPLETED" and "error" not in adopted, "Derived status only")
        a.equal(adopted["work"], ledger["counts"], "Old physical receipt counted unchanged")
        for key, filename in (("trace", "trace.h5"), ("bank_stats", "banks.h5")):
            a.need(a.bound(adopted[key]) == directory / filename, "Reuse original physical " + key)
        receipt = a.bound(proof["verification_receipt"])
        a.need(receipt == completion_path.parent / "receipt.json", "Certificate receipt owner")
        verified = _closed_job(a, receipt.parent)
        rows = sum(
            row["native_prefix_rows"] + row["matching_group_rows"]
            for row in proof["native_candidate_checks"]
        )
        a.need(
            verified["counts"].get("completed_neural_rows", 0) == rows,
            "Certificate pays every native-prefix and matching-group forward",
        )
        for unit in (
            "native_controls",
            "model_controls",
            "physics_substeps",
            "backward_rows",
            "optimizer_steps",
        ):
            a.need(
                verified["counts"].get("requested_" + unit, 0) == 0, "Numerical-only certificate"
            )
        a.need(original_receipt not in self.failed_receipts, "One certificate per failed episode")
        self.failed_receipts[original_receipt] = original
        self.certificate_receipts.add(receipt)
        self.original_failures.append(
            dict(
                original_manifest=adopted["original_validation_failure"],
                original_status=original["status"],
                original_error=original["error"],
                original_receipt_status="FAILED",
                adopted_manifest=str(manifest_path),
                validation_completion=adopted["validation_completion"],
                old_cross_group_contract_passed=False,
            )
        )
        return directory

    def resumed_episode(self, directory: Path, manifest: dict) -> None:
        """Independently check prefix reuse and derived receipt, never repeat physics."""
        a = self.audit
        proof = manifest["continuation"]
        prefix_job = a.json(a.bound(proof["prefix_job"]))
        job = a.json(directory / "job.json")
        for key, value in prefix_job.items():
            if key not in ("out", "limits"):
                a.equal(job[key], value, "Resumed original episode job: " + key)
        a.need(
            job["arm"] == "independent" and job["alpha"] == 0 and job["round"] == 4,
            "Only the two inventoried independent R4 prefixes are resumed",
        )
        prefix_ledger = a.bound(proof["prefix_ledger"])
        raw_suffix_ledger = a.bound(proof["suffix_ledger"])
        suffix_receipt = a.bound(proof["suffix_receipt"])
        a.need(
            raw_suffix_ledger == directory / "suffix_ledger.jsonl"
            and suffix_receipt == directory / "suffix_receipt.json",
            "Preserved distinct raw suffix ledger and receipt",
        )
        expected, overhead = derive_combined_events(
            prefix_ledger.read_bytes(), raw_suffix_ledger.read_bytes()
        )
        actual = [
            json.loads(line) for line in (directory / "ledger.jsonl").read_text().splitlines()
        ]
        a.bind(directory / "ledger.jsonl")
        a.equal(
            actual[:-1], expected, "Logical ledger is exactly prefix completed calls plus suffix"
        )
        a.need(
            actual[-1]["id"] == len(expected) and actual[-1]["event"] == "closed",
            "Single new combined closure",
        )
        suffix_rows = [json.loads(line) for line in raw_suffix_ledger.read_text().splitlines()]
        a.equal(suffix_rows[-1]["receipt"], a.json(suffix_receipt), "Actual raw suffix receipt")
        old = load_arrays(a, a.bound(proof["prefix_trace"]))
        full = load_arrays(a, a.bound(manifest["trace"]))
        n = len(old["actions"])
        a.need(n == proof["prefix_controls"] and n in (198, 151), "Bound exact preserved prefix")
        for key, value in old.items():
            a.equal(
                full[key][: len(value)], value, "Original physical prefix copied exactly: " + key
            )
        reference = load_arrays(a, a.bound(proof["reference_R1_trace"]))
        for key in ("states", "warmstarts", "integrations", "observations", "actions", "costs"):
            a.close(
                full[key],
                reference[key],
                "Independent R1 deterministic counterfactual: " + key,
                atol=STATE_ATOL,
            )
        import h5py

        with (
            h5py.File(a.bound(proof["prefix_banks"]), "r") as source,
            h5py.File(a.bound(manifest["bank_stats"]), "r") as target,
        ):
            a.need(
                len(source) == proof["prefix_banks_count"] == n + 1,
                "All original bank groups retained",
            )
            for name in source:
                a.need(name in target, "Original prefix bank group is not absent")
                a.equal(
                    dict(source[name].attrs), dict(target[name].attrs), "Original bank attributes"
                )
                a.need(set(source[name]) == set(target[name]), "Original bank datasets")
                for field in source[name]:
                    a.equal(
                        source[name][field][:],
                        target[name][field][:],
                        "Original bank array: " + field,
                    )
        for spec in proof["prefix_full_banks"]:
            original_full = a.bound(spec)
            a.bind(directory / original_full.name, spec)
        resume = load_arrays(a, a.bound(proof["resume_context"]))
        for key, plural in (
            ("state", "states"),
            ("warmstart", "warmstarts"),
            ("integration", "integrations"),
        ):
            a.equal(resume[key], old[plural][-1], "Resume uses actual complete boundary: " + key)
        a.need(
            int(resume["control"]) == n
            and int(resume["bank_count"]) == n + 1
            and not bool(resume["initial_plan"])
            and proof["acquisition_replayed_prefix_controls"] == 0
            and proof["native_suffix_controls"] == len(full["actions"]) - n,
            "Prefix is reused once; only the remaining native suffix is acquired",
        )
        with h5py.File(a.bound(proof["prefix_banks"]), "r") as source:
            last = source[f"{n:06d}"]
            carried = np.concatenate((last["U_after"][:][1:], last["U_after"][:][-1:]))
            a.equal(resume["U_carried_after_one_shift"], carried, "Exactly one carried-plan shift")
            a.equal(
                proof["resume_rng_state"],
                json.loads(last.attrs["rng_after"]),
                "Exact sampler continuation",
            )
            a.need(
                proof["carried_plan_fingerprint"] == typed_fingerprint(carried),
                "Carried plan fingerprint",
            )
        self.resumed[directory.resolve()] = dict(
            prefix_ledger=proof["prefix_ledger"], overhead=overhead
        )

    def acquisition_roster(
        self, audit, block, stream, current, prepared, config, config_binding, reviewed
    ):
        entries = {(row["arm"], row["slot"]): row for row in block["acquisitions"]}
        roles = {(arm, slot) for arm in legacy.ARMS for slot in range(8)}
        audit.need(
            len(entries) == len(block["acquisitions"]) == 16 and set(entries) == roles,
            "Complete actual controller roster",
        )
        acquisitions = {}
        for (arm, slot), row in entries.items():
            manifest_path = audit.bound(row["manifest"])
            manifest = audit.json(manifest_path)
            adopted = "original_validation_failure" in manifest
            directory = (
                self.certificate(manifest_path, manifest) if adopted else manifest_path.parent
            )
            if "continuation" in manifest:
                self.resumed_episode(directory, manifest)
            checker = functools.partial(audit_matched_predictions, charge_matching_rows=not adopted)
            closed = (
                functools.partial(
                    original_failed_ledger,
                    original=audit.json(audit.bound(manifest["original_validation_failure"])),
                )
                if adopted
                else None
            )
            expected = dict(
                round=block["round"],
                stream=stream,
                arm=arm,
                slot=slot,
                alpha=0.0 if arm == "independent" else 0.12,
                seed=config["seeds"]["planner_base"]
                + config["seeds"]["planner_stream_stride"] * stream
                + slot,
                checkpoint=current[arm]["checkpoint"],
            )
            measured = audit_acquisition(
                audit,
                directory,
                expected,
                config,
                result_override=manifest if adopted else None,
                native_prediction_checker=checker,
                closed_job_checker=closed,
            )
            job = audit.json(directory / "job.json")
            for key, value in (
                ("config", config_binding),
                ("restarts", prepared["restarts"]),
                ("task", config["controller"]["task"]),
            ):
                audit.equal(job[key], value, "Frozen controller binding: " + key)
            audit.need(
                job["controls"] == 300 and job["physics_threads"] == 4, "Frozen controller runtime"
            )
            measured["manifest_path"] = str(directory / "manifest.json")
            measured["manifest_binding"] = row["manifest"]
            acquisitions[(arm, slot)] = measured
            reviewed.add((directory / "receipt.json").resolve())
            if adopted:
                proof = audit.json(audit.bound(manifest["validation_completion"]))
                reviewed.add(audit.bound(proof["verification_receipt"]))
            self.native_checks.extend(
                dict(round=block["round"], stream=stream, arm=arm, slot=slot, **value)
                for value in measured["native_discrepancy_checks"]
                if value.get("comparison_version") == COMPARISON_VERSION
            )
        legacy._paired_first_banks(audit, acquisitions)
        return acquisitions

    def accounting(self, audit, stream_root, result, audited, out):
        """Close all logical phases plus separately bounded interrupted requests."""
        declaration = result["accounting"]
        roster = declaration["child_receipts"]
        counted, logical = set(), {}
        for spec in roster:
            path = audit.bound(spec)
            audit.need(path not in counted, "Every paid closed receipt is counted once")
            counted.add(path)
            if path in self.failed_receipts:
                ledger = original_failed_ledger(audit, path.parent, self.failed_receipts[path])
            else:
                ledger = _closed_job(audit, path.parent)
            logical = sum_counts(logical, ledger["counts"])
        audit.equal(
            sorted(map(str, counted)), sorted(map(str, audited)), "No omitted or unreviewed phase"
        )
        audit.equal(
            logical,
            declaration["logical_counts"],
            "Closed logical census including verification once",
        )
        extra = {audit.bound(spec) for spec in declaration["extra_verification_receipts"]}
        audit.equal(
            sorted(map(str, extra)),
            sorted(map(str, self.certificate_receipts & counted)),
            "Supplemental numerical receipts are a counted subset, not added twice",
        )
        overhead, seen = {}, set()
        expected_ledgers = {
            str(row["prefix_ledger"]["path"]): row["overhead"]
            for directory, row in self.resumed.items()
            if directory.is_relative_to(stream_root)
        }
        for row in declaration["infrastructure_overhead"]:
            ledger_path = audit.bound(row["ledger"])
            audit.need(
                str(ledger_path) in expected_ledgers and ledger_path not in seen,
                "Only frozen interrupted forecast overhead",
            )
            seen.add(ledger_path)
            pending, counts = pending_overhead(ledger_path.read_bytes())
            audit.equal(
                counts,
                expected_ledgers[str(ledger_path)],
                "Independently derived pending request reservation",
            )
            audit.equal(
                row["counts"],
                counts,
                "Unknown work is charged conservatively, not declared completed",
            )
            declared = row["excluded_pending_requests"]
            audit.equal(
                [(item.get("request_id", item.get("id")), item["charges"]) for item in declared],
                [(item["id"], item["charges"]) for item in pending],
                "Exact excluded pending IDs and charges",
            )
            original_job = (
                audit.bound(row["job"]) if "job" in row else ledger_path.parent / "job.json"
            )
            audit.bind(original_job)
            audit.need(original_job.parent == ledger_path.parent, "Original interrupted job owner")
            overhead = sum_counts(overhead, counts)
        audit.equal(
            sorted(map(str, seen)),
            sorted(expected_ledgers),
            "Every interrupted reservation retained",
        )
        audit.equal(
            overhead,
            declaration["infrastructure_counts"],
            "Separate known infrastructure uncertainty",
        )
        combined = sum_counts(logical, overhead)
        audit.equal(
            combined, declaration["counts"], "Complete original plus continuation work census"
        )
        ledger_path = stream_root / "study_ledger.jsonl"
        audit.bind(ledger_path)
        snapshot = out / f"input_s{result['stream']}_study_ledger.jsonl"
        snapshot.write_bytes(ledger_path.read_bytes())
        events = [json.loads(line) for line in snapshot.read_text().splitlines()]
        audit.need(len(events) == len(roster), "Every actual closed import is present")
        for event, spec in zip(events, roster, strict=True):
            audit.need(
                event["event"] == "closed_child_included" and event["receipt"] == spec,
                "Original/new import order",
            )
            audit.equal(
                event["counts"], audit.json(audit.bound(spec))["counts"], "Imported receipt census"
            )
        return combined


def audit_engineering_bindings(audit: LearningAudit, base: Path) -> dict:
    """Separate original scientific inputs from the amended execution source."""
    launch = audit.json(base / "launch.json")
    engineering = launch["engineering"]
    audit.need(
        engineering["schema"] == "local-coupling-continuation-l3-1",
        "Explicit versioned engineering continuation",
    )
    legacy._input_bindings(audit, engineering)
    parent_path = audit.bound(engineering["parent_launch"])
    parent = audit.json(parent_path)
    for key in (
        "config",
        "prepared_manifest",
        "source_manifest",
        "environment",
        "source_directory",
    ):
        audit.equal(launch[key], parent[key], "Original scientific input is unchanged: " + key)
    audit.need(
        Path(launch["base_directory"]).resolve() == base and parent_path.parent != base,
        "Continuation uses a separate output root",
    )
    original = audit.json(audit.bound(engineering["parent_worker_result"]))
    audit.need(
        original["status"] == "FAILED_STOPPED", "Original stopped result retained as failure"
    )
    original_source = audit.json(audit.bound(launch["source_manifest"]))
    new_source = audit.json(audit.bound(engineering["source_manifest"]))
    original_files = {row["path"]: row for row in original_source["files"]}
    new_files = {row["path"]: row for row in new_source["files"]}
    root = Path(engineering["source_directory"]).resolve()
    for row in new_source["files"]:
        audit.bind(root / row["path"], row)
    essential = {
        "assets/walker2d.xml",
        "src/gps/mppi_gps/local_coupling_controller.py",
        "src/gps/mppi_gps/local_coupling_learning.py",
        "src/gps/mppi_gps/local_coupling_multiround.py",
        "src/gps/mppi_gps/policy.py",
        "src/gps/mppi_gps/runtime.py",
        "src/gps/mppi_gps/provenance.py",
        "src/gps/mppi_gps/trajectory_pi_native.py",
        "src/gps/mppi_gps/trajectory_pi_student.py",
    }
    essential.update(
        name for name in original_files if name.startswith(("src/envs/", "src/mppi/", "src/utils/"))
    )
    for name in sorted(essential):
        audit.need(name in new_files, "Original core source exists in amended export: " + name)
        for key in ("sha256", "bytes"):
            audit.equal(
                new_files[name][key], original_files[name][key], "Unchanged core algorithm: " + name
            )
    amendment = audit.json(audit.bound(engineering["amendment"]))
    inventory = audit.json(audit.bound(engineering["inventory"]))
    audit.need(
        inventory["schema"] == "l3-continuation-recovery-inventory-1"
        and inventory["original_worker_status"] == "FAILED_STOPPED",
        "Complete preserved source/output inventory",
    )
    catalogue = audit.json(audit.bound(inventory["full_original_file_catalogue"]))
    # The catalogue's exact schema is preserved; its per-file binding rows are
    # checked recursively without assigning a scientific meaning to filenames.
    legacy._input_bindings(audit, catalogue)
    audit.need(
        sum(row["native_controls"] for row in inventory["interrupted_round4_episodes"]) == 349
        and len(inventory["interrupted_round4_episodes"]) == 2,
        "Exactly the two preserved interrupted prefixes",
    )
    old_audit_path = parent_path.parent / "audit/final/receipt.json"
    old_audit = audit.json(old_audit_path)
    old_supervisor = audit.json(audit.bound(engineering["parent_supervisor_result"]))
    diagnostic_path = parent_path.parent / "engineering/validation_failure_review_002/result.json"
    diagnostic = audit.json(diagnostic_path)
    legacy._input_bindings(audit, diagnostic["inputs"])
    audit.need(
        old_audit["passed"] is False
        and old_audit["requested_neural_rows"] == 0
        and diagnostic["requested_forward_rows"] == diagnostic["completed_forward_rows"] == 52,
        "All paid numerical review is retained and charged to the original audit allowance",
    )
    for key in (
        "native_controls",
        "model_controls",
        "physics_substeps",
        "optimizer_steps",
        "gradients",
    ):
        audit.need(diagnostic[key] == 0, "Antecedent review did not repeat science: " + key)
    paid = dict(
        requested_neural_rows=old_audit["requested_neural_rows"]
        + diagnostic["requested_forward_rows"],
        completed_neural_rows=old_audit["completed_neural_rows"]
        + diagnostic["completed_forward_rows"],
        unknown_partial_neural_rows=old_audit["unknown_partial_neural_rows"],
        cpu_seconds=old_supervisor["audit"]["process_tree_cpu_seconds"] + diagnostic["cpu_seconds"],
        wall_seconds=old_supervisor["audit"]["wall_seconds"] + diagnostic["wall_seconds"],
    )
    audit.need(paid["unknown_partial_neural_rows"] == 0, "No unbounded antecedent audit forwards")
    audit.neural_limit = min(audit.neural_limit, 40_000_000 - paid["requested_neural_rows"])
    audit.cpu_limit = min(audit.cpu_limit, 7200 - paid["cpu_seconds"])
    audit.seconds = min(audit.seconds, 5400 - paid["wall_seconds"])
    audit.check()
    return dict(
        engineering=engineering,
        amendment=amendment,
        inventory=inventory,
        original_source_commit=original_source["git_commit"],
        engineering_source_commit=new_source["git_commit"],
        antecedent_audit=paid,
        core_algorithm_bytes_identical=True,
        original_cross_group_contract_passed=False,
        original_failure_retained=True,
    )


def _inspect(audit: LearningAudit, base: Path, out: Path) -> dict:
    engineering = audit_engineering_bindings(audit, base)
    review = ContinuationReview(audit)
    # Composition is scoped to this isolated audit invocation. The original
    # functions remain byte-identical, and default audits retain their gate.
    old_roster, old_accounting = legacy._acquisition_roster, legacy._accounting
    try:
        legacy._acquisition_roster, legacy._accounting = (
            review.acquisition_roster,
            review.accounting,
        )
        findings = legacy._inspect(audit, base, out)
    finally:
        legacy._acquisition_roster, legacy._accounting = old_roster, old_accounting
    audit.need(
        len(review.original_failures) == 1, "Exactly one explicitly certified validation failure"
    )
    audit.need(
        len(review.resumed) == 2, "Exactly two independently derived interrupted continuations"
    )
    raw = audit.json(base / "worker/result.json")
    metadata = raw["engineering_continuation"]
    audit.need(metadata["version"] == "L3.1", "Final root declares engineering continuation")
    for key in ("parent_launch", "parent_worker_result", "amendment", "inventory"):
        audit.equal(
            metadata[key], engineering["engineering"][key], "Final amended provenance: " + key
        )
    audit.equal(
        metadata["engineering_source_manifest"],
        engineering["engineering"]["source_manifest"],
        "Final result binds actual amended source",
    )
    total_unknown = {
        key: value
        for key, value in findings["accounting"]["counts"].items()
        if key.startswith("unknown_partial_") and value
    }
    audit.equal(
        total_unknown,
        dict(
            unknown_partial_model_controls=24576,
            unknown_partial_physics_substeps=98304,
            unknown_partial_model_queries=2,
        ),
        "Only the two frozen interrupted model forecasts remain uncertain",
    )
    global_forward = (
        findings["accounting"]["counts"].get("requested_neural_rows", 0)
        + audit.requested_rows
        + engineering["antecedent_audit"]["requested_neural_rows"]
    )
    config = audit.json(audit.bound(raw["config"]))
    audit.need(
        global_forward <= config["budgets"]["global"]["neural_rows"],
        "All antecedent/current forwards share original global ceiling",
    )
    findings.update(
        schema="local-coupling-multiround-l3-1-independent-findings-1",
        version="L3.1",
        engineering=engineering,
        original_validation_failures=review.original_failures,
        amended_native_discrepancy_checks=review.native_checks,
        resumed_episode_count=len(review.resumed),
        original_cross_group_contract_passed=False,
        original_failure_preserved=True,
    )
    findings["accounting"].update(
        clean=False,
        complete_logical_work=True,
        infrastructure_uncertainty_retained=True,
        unknown_partial_upper=total_unknown,
        antecedent_audit=engineering["antecedent_audit"],
        combined_requested_neural_rows=global_forward,
    )
    return findings


def audit_run(
    base: Path,
    out: Path,
    *,
    seconds: float = 5400,
    cpu_seconds: float = 7200,
    neural_rows: int = 40_000_000,
    output_bytes: int = 2 * 1024**3,
    memory_bytes: int = 6 * 1024**3,
) -> dict:
    """Audit the full planned continuation once, charging every retained prefix."""
    base, out = Path(base).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    audit = LearningAudit(
        seconds=seconds,
        cpu_seconds=cpu_seconds,
        neural_rows=neural_rows,
        output_bytes=output_bytes,
        memory_bytes=memory_bytes,
    )
    try:
        findings = _inspect(audit, base, out)
    except BaseException as error:
        findings = dict(
            passed=False,
            error=repr(error),
            traceback=traceback.format_exc(),
            original_failure_preserved=True,
            original_cross_group_contract_passed=False,
            scope="Failed or incomplete L3.1 audit; no absent round is certified and the original L3.0 failure remains unchanged.",
        )
    findings["checks"] = audit.checks
    for filename, value in (("findings.json", findings), ("file_bindings.json", audit.files)):
        (out / filename).write_text(
            json.dumps(value, default=_plain, indent=2, allow_nan=False) + "\n"
        )

    def seal(path):
        data = path.read_bytes()
        return dict(
            path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )

    receipt = dict(
        schema="local-coupling-multiround-l3-1-independent-audit-1",
        stage="final",
        passed=findings["passed"],
        wall_seconds=time.monotonic() - audit.started,
        cpu_seconds=time.process_time() - audit.cpu_started,
        requested_neural_rows=audit.requested_rows,
        completed_neural_rows=audit.completed_rows,
        unknown_partial_neural_rows=audit.unknown_rows,
        model_controls=0,
        native_controls=0,
        physics_substeps=0,
        backward_rows=0,
        optimizer_steps=0,
        findings=seal(out / "findings.json"),
        file_bindings=seal(out / "file_bindings.json"),
        original_failure_preserved=True,
        original_cross_group_contract_passed=False,
        limits=dict(
            wall_seconds=audit.seconds,
            cpu_seconds=audit.cpu_limit,
            neural_rows=audit.neural_limit,
            output_bytes=output_bytes,
            memory_bytes=memory_bytes,
        ),
    )
    size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    if size > output_bytes:
        receipt.update(passed=False, error="Independent audit output ceiling exceeded")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5400)
    parser.add_argument("--cpu-seconds", type=float, default=7200)
    parser.add_argument("--neural-rows", type=int, default=40_000_000)
    args = parser.parse_args(argv)
    receipt = audit_run(
        args.run,
        args.out,
        seconds=args.seconds,
        cpu_seconds=args.cpu_seconds,
        neural_rows=args.neural_rows,
    )
    print(json.dumps(receipt, indent=2))
    return int(not receipt["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
