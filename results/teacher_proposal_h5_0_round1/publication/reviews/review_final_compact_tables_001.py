"""Append-only CSV uniqueness and copied operator-binding crosschecks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import time
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
BASE = ART / "received_bundle_001/unpacked"
PACKET = BASE / "collection/packet"
REPORT = BASE / "report_render/result"
OUT = ART / "final_independent_review_supplement_001.json"


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(OUT)
    cache, inputs = {}, {}
    checks = 0

    def check(value, message):
        nonlocal checks
        checks += 1
        if not value:
            raise ValueError(message)

    def read(path):
        path = Path(path).resolve()
        check(path.is_relative_to(BASE.resolve()), "Input outside copied bundle")
        if path not in cache:
            check(path.stat().st_size < 16 * 1024**2, "Oversized compact input")
            with path.open("rb") as stream:
                value = stream.read(16 * 1024**2 + 1)
            check(len(value) <= 16 * 1024**2, "Compact input grew")
            cache[path] = value
            inputs[str(path)] = dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
        return cache[path]

    def load(path):
        return json.loads(read(path))

    def table(name):
        return list(csv.DictReader(io.StringIO(read(REPORT / name).decode())))

    diag = load(PACKET / "diagnostics.json")
    worker = load(PACKET / "run/worker_result.json")
    external = load(PACKET / "external_catalog.json")
    episodes = {row["tag"] for row in diag["events"] if row["event"] == "episode_complete"}
    metrics = {row["tag"] for row in diag["events"] if row["event"] == "policy_metric_complete"}
    cohorts = {(sid, arm) for sid, arms in worker["teacher_summaries"].items() for arm in arms}
    for name, key, expected in (
        ("episode_outcomes.csv", "tag", episodes),
        ("full_data_metrics.csv", "tag", metrics),
        ("teacher_action_agreement.csv", "episode", episodes),
    ):
        rows = table(name)
        values = [row[key] for row in rows]
        check(
            len(values) == len(set(values)) and set(values) == expected,
            "Duplicate/missing CSV identity: " + name,
        )
    teacher_rows = table("teacher_outcomes.csv")
    identities = [(row["stream"], row["arm"]) for row in teacher_rows]
    check(
        len(identities) == len(set(identities)) and set(identities) == cohorts,
        "Teacher CSV cohort roster differs",
    )
    for row in teacher_rows:
        source = worker["teacher_summaries"][row["stream"]][row["arm"]]
        for key in ("episodes", "successes", "survived", "controls"):
            check(int(row[key]) == source[key], "Teacher CSV count differs")
        check(
            math.isclose(float(row["mean_cost"]), source["mean_cost"], rel_tol=0.0, abs_tol=1e-12),
            "Teacher CSV cost differs",
        )
    mechanism = table("mechanism_summary.csv")
    keys = [(row["stream"], row["arm"]) for row in mechanism]
    check(len(keys) == len(set(keys)) and set(keys) == cohorts, "Mechanism CSV identities differ")
    expected_curves = {
        ("learning/" + path.name, point["update"])
        for path in sorted((PACKET / "run/learning").glob("*.json"))
        for point in load(path)["learning_curve"]
    }
    curve_rows = table("learning_curves.csv")
    actual_curves = [(row["artifact"], int(row["update"])) for row in curve_rows]
    check(
        len(actual_curves) == len(set(actual_curves)) == 6000
        and set(actual_curves) == expected_curves,
        "Learning points duplicated or missing",
    )
    check(
        all(
            row["partial"] == "False" and row["point"] == row["arm"] == "parent"
            for row in curve_rows
        ),
        "Parent learning point was mislabeled",
    )

    operator = load(BASE / "report_render/receipt.json")
    child = load(BASE / "report_render/child_receipt.json")
    report = load(REPORT / "receipt.json")
    check(
        child["report_receipt"] == report and operator["child_receipt"] == child,
        "Nested report receipts differ",
    )
    check(
        operator["outer_exit"] == load(BASE / "report_render/exit.json"),
        "Report outer exit differs",
    )
    own = load(BASE / "report_render/ownership.json")
    check(
        own["report_source_sha256"]
        == report["source_sha256"]
        == operator["source_sha256"]
        == child["source_sha256"],
        "Reporter source chain differs",
    )
    helper = BASE / "helpers/render_saved_report_bounded.py"
    check(
        hashlib.sha256(read(helper)).hexdigest() == own["helper_sha256"],
        "Report outer helper source differs",
    )
    check(
        own["collection_receipt_sha256"]
        == hashlib.sha256(read(BASE / "collection/receipt.json")).hexdigest(),
        "Collector/report owner differs",
    )
    check(
        own["packet_manifest_sha256"]
        == hashlib.sha256(read(PACKET / "packet_manifest.json")).hexdigest(),
        "Packet/report owner differs",
    )
    remote = "/home/jaisel/mppi-gps-workspace/artifacts/2026-11-01_teacher_proposal_h5_0_r1_001/"
    run = "/home/jaisel/mppi-gps-workspace/runs/2026-11-01_teacher_proposal_h5_0_r1_001/"
    copied_count = external_source_count = 0
    for filename, declared in operator["operator_bindings"].items():
        if filename.startswith(remote + "collection_001/packet/"):
            path = PACKET / filename.removeprefix(remote + "collection_001/packet/")
        elif filename == remote + "collection_001/receipt.json":
            path = BASE / "collection/receipt.json"
        elif filename.startswith(remote + "report_render_001/"):
            path = BASE / "report_render" / filename.removeprefix(remote + "report_render_001/")
        elif filename == run + "manifest.json":
            path = PACKET / "run/manifest.json"
        elif filename.startswith(run + "source/"):
            name = filename.removeprefix(run)
            record = external["files"][name]
            check(
                all(record[key] == declared[key] for key in ("bytes", "sha256")),
                "External frozen helper binding differs",
            )
            external_source_count += 1
            continue
        else:
            check(filename.startswith(remote), "Unexpected remote operator input")
            path = BASE / "helpers" / filename.removeprefix(remote)
        value = read(path)
        check(
            len(value) == declared["bytes"]
            and hashlib.sha256(value).hexdigest() == declared["sha256"],
            "Copied operator input differs: " + filename,
        )
        copied_count += 1
    bootstrap = load(BASE / "operators/audit_operator_001/bootstrap_receipt.json")
    check(
        bootstrap["source_commit"] == "ff67495dae33ee43952677d0c0165a43f8d469b0",
        "Audit bootstrap source-commit declaration differs",
    )
    check(
        bootstrap["observed_worker_result_sha256"]
        == hashlib.sha256(read(PACKET / "run/worker_result.json")).hexdigest(),
        "Audit bootstrap worker owner differs",
    )
    check(
        operator["scientific_calls"] == child["scientific_calls"] == 0,
        "Report operator performed scientific work",
    )
    output = dict(
        schema="walker-h5-final-independent-compact-review-supplement-v1",
        passed=True,
        checks=checks,
        supplements="final_independent_review_001.json",
        csv_identity_sets_and_all6000_learning_points_verified=True,
        teacher_summary_csv_values_verified=True,
        nested_report_receipts_verified=True,
        operator_input_bindings_verified=dict(
            total=len(operator["operator_bindings"]),
            copied_payloads=copied_count,
            frozen_external_source_declarations=external_source_count,
        ),
        bootstrap_declared_source_commit=bootstrap["source_commit"],
        source_commit_scope="The copied audit-operator bootstrap explicitly declares ff67495 and its observed worker checksum matches. This is metadata attribution, not a new Git-blob proof; the original prelaunch independent source/Git review remains the stronger association.",
        report_operator_costs=dict(
            operator_read_bytes=operator["operator_read_bytes"],
            child_ownership_read_bytes=child["ownership_read_bytes"],
            renderer_source_payload_read_bytes=report["source_read_bytes"],
            renderer_code_hash_read_bytes=report["code_hash_read_bytes"],
            renderer_output_hash_read_bytes=report["output_hash_read_bytes"],
            renderer_wall_seconds=report["wall_seconds"],
            child_wall_seconds=child["wall_seconds"],
            outer_wall_seconds=operator["outer_exit"]["wall_seconds"],
            operator_wall_seconds=operator["wall_seconds"],
            scope="These are distinct reported read passes and overlapping nested wall timers; no wall-time sum or billing estimate is inferred.",
        ),
        input_bindings=inputs,
        logical_file_read_bytes=sum(row["bytes"] for row in inputs.values()),
        wall_seconds=time.perf_counter() - started,
        new_scientific_calls=0,
        raw_arrays_decoded=0,
        external_raw_or_checkpoints_opened=0,
        ssh_calls=0,
    )
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in output.items()
                if key not in ("input_bindings", "report_operator_costs")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
