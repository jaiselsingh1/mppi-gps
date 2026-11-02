"""Select closed compact H5 evidence explicitly, validate it and publish once.

No remote access, research execution or Git operations. New publisher logs stay
outside the capsule; an active capture is never copied into its own payload.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
REPO = ART.parents[1]
BASE = ART / "received_bundle_001/unpacked"
PACKET = BASE / "collection/packet"
REPORT = BASE / "report_render/result"
PUBLICATION = ART / "publication"
OUT = REPO / "results/teacher_proposal_h5_0_round1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            value.update(block)
    return value.hexdigest()


def binding(path):
    return dict(bytes=path.stat().st_size, sha256=digest(path))


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError("The capsule must never be reused or amended")
    accounting_path = ART / "publication_inputs/accounting.json"
    accounting = load(accounting_path)
    worker = load(PACKET / "run/worker_result.json")
    audit = load(PACKET / "audit/result/receipt.json")
    external = load(PACKET / "external_catalog.json")
    collection = load(BASE / "collection/receipt.json")
    report = load(REPORT / "receipt.json")
    report_operator = load(BASE / "report_render/receipt.json")
    worker_exit = load(PACKET / "run/worker_exit.json")
    audit_exit = load(PACKET / "audit/exit.json")
    audit_outputs = load(BASE / "external_audit_outputs_catalog.json")
    checks = []

    def check(condition, message):
        checks.append(message)
        if not condition:
            raise ValueError(message)

    check(
        accounting["source_commit"] == "ff67495dae33ee43952677d0c0165a43f8d469b0",
        "Frozen source commit",
    )
    for name, key in (
        ("run/config.json", "config_sha256"),
        ("run/worker_result.json", "worker_result_sha256"),
        ("audit/result/receipt.json", "audit_receipt_sha256"),
    ):
        check(accounting[key] == digest(PACKET / name), "Accounting file ownership: " + name)
    check(
        accounting["scientific_status"]
        == worker["status"]
        == "teacher_competence_negative_no_child_fits",
        "Teacher-negative status",
    )
    check(
        accounting["execution_complete"] is True and accounting["comparison_complete"] is False,
        "Completed acquisition, incomplete learning contrast",
    )
    check(
        accounting["child_fits"] == 0
        and accounting["standalone_panels_measured"] is False
        and accounting["paired_teacher_student_gap_measured"] is False,
        "Unmeasured student outcomes",
    )
    check(accounting["worker"]["reported_work"] == worker["work"], "Unchanged worker work snapshot")
    check(accounting["worker"]["outer_exit"] == worker_exit, "Worker outer closure")
    for key, value in accounting["audit"].items():
        if key in audit:
            check(value == audit[key], "Audit counter: " + key)
    check(accounting["audit"]["outer_exit"] == audit_exit, "Audit outer closure")
    check(
        accounting["audit"]["preparation_source_read_bytes"]
        == audit["preparation"]["source_read_bytes"],
        "Audit preparation reads",
    )
    check(
        accounting["audit"]["independently_recreated_standard_normal_draws"]
        == audit["scientific"]["independently_recreated_standard_normal_draws"],
        "Sampler RNG work",
    )
    check(
        accounting["audit"]["replayed_minibatch_indices"]
        == audit["fitting"]["replayed_minibatch_indices"],
        "Audit index work",
    )
    raw = {name: row for name, row in external["files"].items() if name.startswith("evidence/")}
    check(accounting["evidence"]["closed_raw_files"] == raw, "Closed raw-file declarations")
    check(
        accounting["evidence"]["closed_raw_bytes"]
        == sum(row["bytes"] for row in raw.values())
        == 206888154706,
        "Closed raw byte total",
    )
    check(
        accounting["evidence"]["closing_record_bytes_source_and_metadata_deduction"]
        == accounting["evidence"]["closed_raw_bytes"] - worker["work"]["storage_bytes"]
        == 1624,
        "Preserved pre-close storage distinction",
    )
    check(
        accounting["evidence"]["external_audit_outputs"] == audit_outputs,
        "Audit work-file declarations",
    )
    check(
        accounting["evidence"]["external_catalog_file_count"] == len(external["files"]),
        "External declared file count",
    )
    check(
        accounting["evidence"]["external_catalog_bytes"]
        == sum(row["bytes"] for row in external["files"].values()),
        "External declared byte count",
    )
    for key, value in accounting["collection"].items():
        check(value == collection[key], "Collection counter: " + key)
    for key, value in accounting["reporting"]["renderer"].items():
        check(value == report[key], "Renderer counter: " + key)
    check(accounting["reporting"]["outer"] == report_operator["outer_exit"], "Report outer closure")
    check(
        accounting["reporting"]["operator_read_bytes"] == report_operator["operator_read_bytes"],
        "Report operator read bytes",
    )
    verification = load(ART / "received_bundle_001/verification.json")
    check(
        accounting["transfer"]["local_verification"] == verification,
        "Transfer verification receipt",
    )
    check(
        accounting["transfer"]["bundle_archive"]
        == load(ART / "received_bundle_001/receipt.json")["archive"],
        "Transfer archive declaration",
    )
    check(
        binding(ART / "received_bundle_001/bundle.tar.gz")
        == {key: accounting["transfer"]["bundle_archive"][key] for key in ("bytes", "sha256")},
        "Local compact archive checksum",
    )
    check(
        accounting["transfer"]["downloaded_file_bytes"]
        == (ART / "received_bundle_001/bundle.tar.gz").stat().st_size
        + (ART / "received_bundle_001/receipt.json").stat().st_size
        == 1066760,
        "One-attempt compact download bytes",
    )
    config = load(PACKET / "run/config.json")
    check(accounting["frozen_reservations"] == config["budgets"], "Unchanged frozen reservations")
    initial = REPO / "runs/2026-11-01_teacher_proposal_h5_0_r1_001"
    initial_files = [path for path in initial.rglob("*") if path.is_file()]
    check(
        len(initial_files) == 539
        and sum(path.stat().st_size for path in initial_files) == 10622281,
        "Original local source/input export size",
    )
    check(
        not (initial / "evidence").exists() and not (initial / "worker_result.json").exists(),
        "Local initial export is not a completed-run mirror",
    )
    accounting_review = PUBLICATION / "accounting_input_review_001.json"
    save(
        accounting_review,
        dict(
            schema="walker-h5-publication-accounting-input-review-v1",
            passed=True,
            checks=checks,
            accounting=binding(accounting_path),
            scientific_calls=0,
            raw_external_files_rehashed=False,
            scope="Compact metadata and declared external checksums only; actual full raw verification is the closed independent audit.",
        ),
    )

    original_readme = ART / "publication_inputs/README.md"
    text = original_readme.read_text()
    old = "`diagnostics.json` binds 84 scalar events and 34,400 selection summaries to the complete original ledger hash."
    new = "`diagnostics.json` binds 84 scalar events and nine mechanism summaries covering 34,400 selections to the complete original ledger hash."
    check(text.count(old) == 1, "Expected original README precision correction")
    corrected_readme = ART / "publication_inputs/README_publication_001.md"
    with corrected_readme.open("x") as stream:
        stream.write(text.replace(old, new))
    selected = []

    def include(source, destination):
        source = Path(source)
        check(source.is_file(), "Selected closed payload exists: " + str(source))
        selected.append(dict(source=str(source), destination=destination, **binding(source)))

    include(original_readme, "publication/README_original_input_001.md")
    include(accounting_review, "publication/accounting_input_review_001.json")
    include(
        BASE / "external_audit_outputs_catalog.json",
        "accounting/external_audit_outputs_catalog.json",
    )
    for name in (
        "final_independent_review_001.json",
        "final_independent_review_001.md",
        "final_independent_review_supplement_001.json",
        "final_report_review_001.json",
        "preliminary_closed_worker_review_001.json",
        "preliminary_closed_worker_review_001.md",
    ):
        include(ART / name, "publication/reviews/" + name)
    for name in (
        "review_closed_worker_probe_001.py",
        "review_final_compact_evidence_001.py",
        "review_final_compact_tables_001.py",
        "test_publish_round1_capsule.py",
    ):
        include(PUBLICATION / name, "publication/reviews/" + name)
    for name in (
        "runtime_probe.py",
        "launch_bootstrap.py",
        "audit_bootstrap.py",
        "audit_monitor.py",
        "monitor.py",
        "monitor_v1.py",
        "monitor_v2.py",
        "monitor_v3.py",
        "monitor_series.py",
        "read_progress.py",
        "read_audit_progress.py",
        "closed_worker_probe.py",
        "collect_saved_evidence.py",
        "test_collect_saved_evidence.py",
        "render_saved_report_bounded.py",
        "test_bounded_saved_report.py",
        "test_audit_operator_helpers.py",
        "audit_operator_helpers.md",
        "postprocessor_helpers_manifest.json",
        "build_accounting.py",
        "build_publication_bundle.py",
        "verify_received_bundle.py",
    ):
        include(ART / name, "operator/helpers/" + name)
    for name in (
        "audit_bootstrap.py",
        "audit_monitor.py",
        "collect_saved_evidence.py",
        "render_saved_report_bounded.py",
    ):
        include(BASE / "helpers" / name, "operator/received_helpers/" + name)
    for directory in ("operator_001", "audit_operator_001"):
        for name in ("bootstrap_receipt.json", "coordinator_stdout.txt", "intent.json"):
            include(BASE / "operators" / directory / name, "operator/" + directory + "/" + name)
    for name in ("child_receipt.json", "exit.json", "ownership.json", "receipt.json", "stdout.txt"):
        include(BASE / "report_render" / name, "operator/report_render/" + name)
    for name in ("receipt.json", "verification.json"):
        include(ART / "received_bundle_001" / name, "publication/compact_transfer/" + name)
    validation = (
        ("audit_operator_validation_001", 4),
        ("collector_validation_001", 3),
        ("bounded_report_validation_001", 3),
    )
    for directory, command_count in validation:
        include(ART / directory / "receipt.json", "validation/" + directory + "/receipt.json")
        for index in range(command_count):
            for channel in ("stdout", "stderr"):
                name = f"{index:02d}.{channel}.txt"
                include(ART / directory / name, "validation/" + directory + "/" + name)
    closed_commands = [
        "012_progress_001",
        "013_progress_002",
        "014_progress_003",
        "015_progress_004",
        "016_progress_005",
        "017_progress_006",
        "018_progress_007",
        "019_progress_008",
        "020_live_monitor_series",
        "021_report_integrated_suite",
        "022_running_publication_checks",
        "023_live_source_integrity",
        "024_teacher_termination_check",
        "025_teacher_termination_check",
        "026_postprocessor_helper_transfer",
        "027_bounded_report_helper_transfer",
        "028_closed_worker_probe",
        "029_scientific_audit_launch",
        "030_audit_monitor_series",
        "031_candidate_clean_export_validation",
        "032_active_ruff_check",
        "033_active_format_check",
        "034_active_diff_check",
        "035_compact_evidence_collection",
        "036_saved_report_render",
        "037_publication_bundle",
        "038_compact_transfer",
        "039_local_transfer_verification",
        "040_build_accounting",
    ]
    for directory in closed_commands:
        folder = ART / "engineering" / directory
        receipt = load(folder / "receipt.json")
        for channel in ("stdout", "stderr"):
            name = channel + ".txt"
            check(
                digest(folder / name) == receipt[channel + "_sha256"],
                "Closed command channel hash: " + directory + "/" + name,
            )
            include(folder / name, "engineering/" + directory + "/" + name)
        include(folder / "receipt.json", "engineering/" + directory + "/receipt.json")
    for name in ("operator_report_checks_001.json", "operator_report_checks_002.json"):
        include(ART / "engineering" / name, "validation/" + name)
    export = ART / "clean_export_code_001"
    for name in ("receipt.json", "logs_receipt.json"):
        include(export / name, "validation/clean_export/" + name)
    export_logs = load(export / "logs_receipt.json")
    check(len(export_logs["files"]) == 74, "All74 clean-export channels are preserved")
    for index in range(37):
        for channel in ("stdout", "stderr"):
            name = f"{index:02d}.{channel}.txt"
            check(
                binding(export / "commands" / name) == export_logs["files"][name],
                "Clean-export channel hash: " + name,
            )
            include(export / "commands" / name, "validation/clean_export_commands/" + name)

    validation_folder = PUBLICATION / "publisher_validation_001"
    validation_folder.mkdir(exist_ok=False)
    validation_rows = []
    commands = [
        [sys.executable, "-B", str(PUBLICATION / "test_publish_round1_capsule.py"), "-v"],
        [
            sys.executable,
            "-B",
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "E4,E7,E9,F,I",
            str(PUBLICATION / "publish_round1_capsule.py"),
            str(PUBLICATION / "test_publish_round1_capsule.py"),
        ],
        [
            sys.executable,
            "-B",
            "-m",
            "ruff",
            "format",
            "--isolated",
            "--line-length",
            "100",
            "--check",
            str(PUBLICATION / "publish_round1_capsule.py"),
            str(PUBLICATION / "test_publish_round1_capsule.py"),
        ],
    ]
    for index, command in enumerate(commands):
        before = time.perf_counter()
        with (
            (validation_folder / f"{index:02d}.stdout.txt").open("xb") as out,
            (validation_folder / f"{index:02d}.stderr.txt").open("xb") as err,
        ):
            process = subprocess.run(command, cwd=REPO, stdout=out, stderr=err, timeout=60)
        row = dict(
            command=command,
            exit_code=process.returncode,
            wall_seconds=time.perf_counter() - before,
            stdout=binding(validation_folder / f"{index:02d}.stdout.txt"),
            stderr=binding(validation_folder / f"{index:02d}.stderr.txt"),
        )
        validation_rows.append(row)
        check(process.returncode == 0, "Publisher synthetic/style validation")
    save(
        validation_folder / "receipt.json",
        dict(
            commands=validation_rows,
            no_research_execution=True,
            test_cases=5,
            scope="Temporary synthetic metadata-only publication fixtures and style checks. No actual run was published by these tests.",
        ),
    )
    include(validation_folder / "receipt.json", "validation/publisher_validation_001/receipt.json")
    for index in range(3):
        for channel in ("stdout", "stderr"):
            name = f"{index:02d}.{channel}.txt"
            include(validation_folder / name, "validation/publisher_validation_001/" + name)
    include(Path(__file__).resolve(), "publication/execute_publication_001.py")
    plan = PUBLICATION / "selected_publication_inputs_001.json"
    save(
        plan,
        dict(
            schema="walker-h5-selected-publication-inputs-v1",
            selected=selected,
            selection_count=len(selected),
            selected_bytes=sum(row["bytes"] for row in selected),
            manifest_excludes_itself=True,
            accounting_input=binding(accounting_path),
            original_readme=binding(original_readme),
            corrected_readme=binding(corrected_readme),
            correction="Nine mechanism summaries cover34400 selections; the original README draft remains preserved.",
            closed_command_range="012..040 only; all earlier001..011 remain in the tracked prelaunch/launch validation record.",
            raw_source_arrays_checkpoints_archives_SQLite_journals_fontcache_excluded=True,
        ),
    )
    include(plan, "publication/selected_publication_inputs_001.json")
    command = [
        sys.executable,
        "-B",
        str(PUBLICATION / "publish_round1_capsule.py"),
        "--packet",
        str(PACKET),
        "--collection-receipt",
        str(BASE / "collection/receipt.json"),
        "--report",
        str(REPORT),
        "--report-source",
        str(BASE / "helpers/report_teacher_proposal_edc0.py"),
        "--source-review",
        str(ART / "prelaunch_independent_review.json"),
        "--readme",
        str(corrected_readme),
        "--accounting",
        str(accounting_path),
        "--implementation-commit",
        "ff67495dae33ee43952677d0c0165a43f8d469b0",
        "--remote-run",
        "/home/jaisel/mppi-gps-workspace/runs/2026-11-01_teacher_proposal_h5_0_r1_001",
        "--local-initial-export",
        str(initial),
        "--out",
        str(OUT),
        "--receipt",
        str(PUBLICATION / "publication_receipt_001.json"),
    ]
    for row in selected:
        command += ["--include", row["source"], row["destination"]]
    capture = PUBLICATION / "publisher_command_001"
    capture.mkdir(exist_ok=False)
    before = time.perf_counter()
    with (capture / "stdout.txt").open("xb") as out, (capture / "stderr.txt").open("xb") as err:
        process = subprocess.run(command, cwd=REPO, stdout=out, stderr=err, timeout=600)
    receipt = dict(
        command=command,
        exit_code=process.returncode,
        wall_seconds=time.perf_counter() - before,
        stdout=binding(capture / "stdout.txt"),
        stderr=binding(capture / "stderr.txt"),
        selected_payloads=len(selected),
        source_accounting_bytes_preserved=binding(accounting_path),
        original_readme_bytes_preserved=binding(original_readme),
        corrected_readme=binding(corrected_readme),
        actual_scientific_calls=0,
        enclosing_wrapper_wall_seconds=time.perf_counter() - started,
        scope="Publisher invocation and its explicit compact input verification/copies only. Nested publisher time must not be added; synthetic/style validation is separately recorded. Import/stat/helper selection hashing and other wrapper read counts are not fully instrumented.",
    )
    save(capture / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    print((capture / "stdout.txt").read_text())
    print((capture / "stderr.txt").read_text())
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
