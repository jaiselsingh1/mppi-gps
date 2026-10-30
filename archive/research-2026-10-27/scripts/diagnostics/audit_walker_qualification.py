"""Audit converted Step32 outcomes without importing a simulator or unpickling.

This checks the committed summary records, not raw acquisition trajectories or
source identity. Existing failed/in-flight work may be absent from the legacy
append stream, so reported work is a lower bound on all historical attempts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
import time


ARMS = ("A", "B1", "B2", "C")
SLOT_KEYS = ("executed_controls", "candidate_controls", "memory_controls",
             "verification_controls", "one_step_verification_controls")
IGNORED_AS_CONTROLS = ("planner_calls", "value_prediction_rows", "prediction_rows")
RECIPES = {
    "A": "Frozen C_23: K32/H48, archive, discrete bank selection, alpha zero",
    "B1": "C_23 selection rule without archive, K32/H48",
    "B2": "Independent raw K32/H48, own weighted mean and retained plan",
    "C": "Independent raw K256/H48, own weighted mean and retained plan",
}


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    return sum(values)/len(values) if values else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, (Integral, bool)) and not math.isfinite(value):
        return None
    return value


def audit_rows(rows, *, finished: bool, expected=64, total_steps=1000):
    """Return PENDING until completion; a completed incomplete panel is FAIL.

    Complete means all four arms have every expected unique job ID AND the
    runner's successful-finish marker is present. Repeated rows are charged
    as observed work, but cannot become extra statistical observations or
    overwrite a less favorable earlier outcome.
    """
    if type(finished) is not bool:
        raise ValueError("finished must be a boolean")
    if not _integer(expected) or expected < 1 or not _integer(total_steps) or total_steps < 1:
        raise ValueError("expected and total_steps must be positive integers")
    rows = list(rows)
    errors = []
    error_count = 0

    def error(category, index, identifier, message):
        nonlocal error_count
        error_count += 1
        if len(errors) < 200:
            errors.append(dict(category=category, row=index, job_id=identifier, message=message))

    seen, occurrences, unique = {}, Counter(), {arm: [] for arm in ARMS}
    work = {arm: dict.fromkeys(SLOT_KEYS, 0) for arm in ARMS}
    unassigned_work = dict.fromkeys(SLOT_KEYS, 0)
    observed = Counter()
    seconds = {arm: [] for arm in ARMS}
    invalid_rows = 0
    unexpected = []
    for index, row in enumerate(rows):
        before = error_count
        if not isinstance(row, dict):
            error("record", index, None, "record must be an object")
            invalid_rows += 1
            continue
        identifier = row.get("job_id")
        match = re.fullmatch(r"(A|B1|B2|C):(\d+)", identifier) if isinstance(identifier, str) else None
        if not match or not 0 <= int(match.group(2)) < expected:
            error("job_id", index, identifier, "unexpected arm or index")
            unexpected.append(identifier)
            # An invalid identifier does not erase its committed query cost.
            target = work[match.group(1)] if match else unassigned_work
            if match:
                observed[match.group(1)] += 1
            ledger = row.get("ledger", {})
            if isinstance(ledger, dict):
                for key in SLOT_KEYS:
                    value = ledger.get(key, 0)
                    if _integer(value) and value >= 0:
                        target[key] += int(value)
            invalid_rows += 1
            continue
        arm, root_index = match.group(1), int(match.group(2))
        observed[arm] += 1
        occurrences[identifier] += 1
        duplicate = identifier in seen
        if duplicate:
            error("duplicate", index, identifier, "duplicate job ID; original outcome retained")
        else:
            seen[identifier] = index
            unique[arm].append(row)

        steps = row.get("steps")
        valid_steps = _integer(steps) and 1 <= steps <= total_steps
        if not valid_steps:
            error("steps", index, identifier, "steps must be an integer in [1,total_steps]")
        survived, terminated = row.get("survived"), row.get("terminated")
        if type(survived) is not bool or type(terminated) is not bool:
            error("termination", index, identifier, "survived and terminated must be booleans")
        elif valid_steps:
            if survived != (steps == total_steps and not terminated):
                error("termination", index, identifier, "survival must mean complete healthy execution")
            if steps < total_steps and not terminated:
                error("termination", index, identifier, "healthy short outcome is incomplete")
        for key in ("cost", "velocity_mae", "seconds"):
            if not _finite(row.get(key)) or row[key] < 0:
                error("finite_outcome", index, identifier, f"{key} must be finite and nonnegative")
        if _finite(row.get("seconds")) and row["seconds"] >= 0:
            seconds[arm].append(float(row["seconds"]))
        for key, expected_value in (("reset_id", 55_000_000+root_index),
                                    ("reset_seed", 55_000_000+root_index),
                                    ("planner_seed", 885_000_000+root_index)):
            if key in row and row[key] != expected_value:
                error("seed_recipe", index, identifier, f"{key} does not match frozen Step32 recipe")
        if "reset_profile" in row:
            profile = "angle_velocity_perturbed" if root_index % 2 else "standard"
            if row["reset_profile"] != profile:
                error("reset_profile", index, identifier, "reset perturbation profile mismatch")
        ledger = row.get("ledger")
        if not isinstance(ledger, dict):
            error("ledger", index, identifier, "missing ledger object")
            ledger = {}
        for key in SLOT_KEYS:
            count = ledger.get(key, 0)
            if not _integer(count) or count < 0:
                error("ledger", index, identifier, f"invalid {key}")
            else:
                work[arm][key] += int(count)
        for key in ("executed_controls", "candidate_controls", "planner_calls"):
            if key not in ledger:
                error("ledger", index, identifier, f"required counter {key} missing")
        if valid_steps:
            for key in ("executed_controls", "planner_calls"):
                if ledger.get(key) != steps:
                    error("ledger", index, identifier, f"{key} must equal executed steps")
        for key in IGNORED_AS_CONTROLS:
            if key in ledger and (not _integer(ledger[key]) or ledger[key] < 0):
                error("ledger", index, identifier, f"invalid non-control counter {key}")
        if arm in ("C", "B2"):
            K = 256 if arm == "C" else 32
            if valid_steps and ledger.get("candidate_controls") != K*48*(steps+1):
                error("raw_recipe", index, identifier, f"candidate count must be {K}*48*(steps+1), including initial extra pass")
            for key in ("memory_controls", "verification_controls", "one_step_verification_controls"):
                if ledger.get(key, 0) != 0:
                    error("raw_recipe", index, identifier, f"raw MPPI must not charge {key}")
            for container in (row, row.get("mppi_config", {}), row.get("planner", {})):
                if isinstance(container, dict):
                    if "K" in container and container["K"] != K:
                        error("raw_recipe", index, identifier, f"declared K must be {K}")
                    if "H" in container and container["H"] != 48:
                        error("raw_recipe", index, identifier, "declared H must be 48")
        invalid_rows += error_count > before

    expected_ids = {f"{arm}:{i}" for arm in ARMS for i in range(expected)}
    missing = sorted(expected_ids-set(seen), key=lambda s:(ARMS.index(s.split(":")[0]), int(s.split(":")[1])))
    duplicates = {key: value for key, value in occurrences.items() if value > 1}
    panel_complete = not missing and not duplicates and not unexpected and len(rows) == 4*expected
    per_arm = {}
    all_falls = []
    for arm in ARMS:
        arm_rows = sorted(unique[arm], key=lambda r:int(r["job_id"].split(":")[1]))
        valid_maes = [float(r["velocity_mae"]) for r in arm_rows if _finite(r.get("velocity_mae")) and r["velocity_mae"] >= 0]
        survivors = sum(r.get("survived") is True for r in arm_rows)
        falls = []
        for r in arm_rows:
            if r.get("terminated") is True:
                ledger = r.get("ledger", {})
                cost = sum(int(ledger.get(k, 0)) for k in SLOT_KEYS if _integer(ledger.get(k, 0)) and ledger.get(k, 0) >= 0)
                fall = {k: r.get(k) for k in ("job_id", "steps", "survived", "terminated", "cost", "velocity_mae", "reset_profile")}
                fall["recorded_controls"] = cost
                falls.append(fall)
                all_falls.append(fall)
        counters = work[arm]
        total_controls = sum(counters.values())
        executed = counters["executed_controls"]
        per_arm[arm] = dict(recipe=RECIPES[arm], observed_rows=observed[arm],
            unique_rows=len(arm_rows), expected_rows=expected, survivors=survivors,
            observed_survival_rate=survivors/len(arm_rows) if arm_rows else None,
            velocity_mae=_mean(valid_maes), mae_rows=len(valid_maes), falls=falls,
            recorded_worker_seconds=sum(seconds[arm]),
            recorded_seconds_per_executed_control=sum(seconds[arm])/executed if executed else None,
            **counters, simulated_controls=total_controls-executed, total_controls=total_controls)
    c = per_arm["C"]
    minimum = math.ceil(.9*expected)
    gates = dict(all_expected_unique_jobs=panel_complete, finished_marker=finished,
        all_record_checks_valid=error_count == 0, repaired_arm_has_all_rows=c["unique_rows"] == expected,
        repaired_survival=c["unique_rows"] == expected and c["survivors"] >= minimum,
        repaired_mae=c["mae_rows"] == expected and c["velocity_mae"] <= .30)
    passed = all(gates.values())
    status = "PENDING" if not finished else "PASS" if passed else "FAIL"
    actual_slots = {key: sum(work[arm][key] for arm in ARMS)+unassigned_work[key] for key in SLOT_KEYS}
    return _json_safe(dict(schema="walker-step32-summary-audit-v1", status=status, qualification_passed=passed,
        finished=finished, expected_per_arm=expected, expected_all_rows=4*expected,
        observed_rows=len(rows), observed_C_rows=observed["C"], unique_C_rows=c["unique_rows"],
        observed_unique_jobs=len(seen), all_256_completion=bool(expected == 64 and panel_complete),
        panel_complete=panel_complete, missing_job_ids=missing, duplicate_job_ids=duplicates,
        unexpected_job_ids=unexpected, invalid_rows=invalid_rows, validation_error_count=error_count,
        validation_errors=errors, gates=gates,
        criteria=dict(total_steps=total_steps, repaired_minimum_survivors=minimum,
                      repaired_maximum_mae=.30, default_recipe="K256/H48, initial_iterations=2"),
        per_arm=per_arm, all_falls=all_falls, recorded_slot_totals=actual_slots,
        unassigned_recorded_slot_totals=unassigned_work,
        recorded_total_controls=sum(actual_slots.values()),
        non_control_counters_excluded=list(IGNORED_AS_CONTROLS),
        source_identity_verified=False, raw_acquisition_states_certified=False,
        all_attempt_work_is_lower_bound=True, additional_physics_controls=0,
        limitations=["Outcome/counter audit only; frozen source/configuration identity requires a separate manifest audit.",
            "Legacy raw-controller rows omit states, actions, warmstarts, raw costs, nominal and RNG; this is not a raw acquisition corpus.",
            "Only committed finished-job costs are available. Failed/in-flight and repeated uncommitted attempts may add work.",
            "One planner configuration and reset panel do not establish training-seed robustness or generalization."]))


def audit_directory(evidence_dir):
    started = time.perf_counter()
    directory = Path(evidence_dir).resolve()
    records_path, status_path = directory/"records.json", directory/"status.json"
    records, process = json.loads(records_path.read_text()), json.loads(status_path.read_text())
    rows = records["rows"] if isinstance(records, dict) else records
    result = audit_rows(rows, finished=process.get("finished") is True)
    result["process_status"] = process
    if process.get("terminated") is True and not result["finished"]:
        result["status"] = "FAIL"
        result["qualification_passed"] = False
        result["process_terminated_without_finish_marker"] = True
    inputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (records_path, status_path)}
    manifest = directory/"manifest.json"
    if manifest.exists():
        inputs[manifest.name] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    result["input_sha256"] = inputs
    result["audit_wall_seconds"] = time.perf_counter()-started
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = directory/f"qualification_audit_{stamp}.json"
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result, destination


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()
    result, path = audit_directory(args.evidence_dir)
    print(json.dumps(dict(status=result["status"], qualification_passed=result["qualification_passed"],
                         observed_C_rows=result["observed_C_rows"], audit=str(path))))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
