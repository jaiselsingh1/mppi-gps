"""Validate a clean Git export using the caller's locked CPU environment.

The export must contain no .git, runs or artifacts directory. All rendering uses
saved capsule data, and all generated files go to a new temporary directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def verify_report_outputs(report: Path, bindings: dict[str, dict]) -> None:
    """Resolve host path aliases while rejecting files outside the report."""
    report = report.resolve()
    for name, binding in bindings.items():
        path = (report / name).resolve()
        if (
            not path.is_relative_to(report)
            or not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]
        ):
            raise ValueError(f"Rendered trajectory-review binding differs: {path}")


def verify_teacher_proposal_report(
    root: Path, capsule: Path, report: Path, *, diagnostics: bool
) -> None:
    """Verify saved H5 JSON, table and figure bindings without raw evidence."""
    receipt = json.loads((report / "receipt.json").read_text())
    if any(
        receipt[key] != 0
        for key in (
            "network_forwards",
            "physics_controls",
            "optimizer_steps",
            "raw_arrays_decoded",
            "checkpoints_loaded",
        )
    ):
        raise ValueError("Saved H5 report must perform zero scientific calls")
    run = capsule / "run"
    result_name = "worker_result.json" if (run / "worker_result.json").is_file() else "result.json"
    expected_inputs = {
        name: run / name for name in ("config.json", result_name, "call_census.json")
    }
    expected_inputs.update(
        {str(path.relative_to(run)): path for path in sorted((run / "learning").glob("*.json"))}
    )
    if diagnostics:
        expected_inputs["saved_diagnostics"] = capsule / "diagnostics.json"
    if set(receipt["inputs"]) != set(expected_inputs):
        raise ValueError("Rendered H5 input roster differs")
    for name, path in expected_inputs.items():
        binding = receipt["inputs"][name]
        if (
            path.stat().st_size != binding["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]
        ):
            raise ValueError("Rendered H5 input binding differs: " + name)
    for name, key in (("config.json", "config_sha256"), (result_name, "source_result_sha256")):
        if receipt[key] != receipt["inputs"][name]["sha256"]:
            raise ValueError("Rendered H5 receipt input differs: " + name)
    source = root / "src/gps/mppi_gps/reporting/teacher_proposal.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["source_sha256"]:
        raise ValueError("Rendered H5 source differs")
    if receipt["source_read_bytes"] != sum(row["bytes"] for row in receipt["inputs"].values()):
        raise ValueError("Rendered H5 input-read census differs")
    required_tables = {
        "full_data_metrics.csv",
        "teacher_action_agreement.csv",
        "episode_outcomes.csv",
        "teacher_outcomes.csv",
        "student_outcomes.csv",
        "paired_gap.csv",
        "learning_curves.csv",
    }
    tables = set(receipt["table_rows"])
    if not required_tables <= tables or tables - required_tables - {"mechanism_summary.csv"}:
        raise ValueError("Rendered H5 table roster differs")
    if not (tables | {"report.md"}) <= set(receipt["outputs"]):
        raise ValueError("Rendered H5 tables or report lack output bindings")
    verify_report_outputs(report, receipt["outputs"])
    for name, expected_rows in receipt["table_rows"].items():
        with (report / name).open(newline="") as stream:
            actual_rows = sum(1 for _ in csv.DictReader(stream))
        if actual_rows != expected_rows:
            raise ValueError("Rendered H5 table row count differs: " + name)
    expected_curve_rows = sum(
        len(json.loads(path.read_text())["learning_curve"])
        for name, path in expected_inputs.items()
        if name.startswith("learning/")
    )
    if receipt["table_rows"]["learning_curves.csv"] != expected_curve_rows:
        raise ValueError("Rendered H5 learning-point roster differs")
    for suffix in ("png", "pdf"):
        name = "learning_curves." + suffix
        if expected_curve_rows and (
            name not in receipt["outputs"] or (report / name).stat().st_size == 0
        ):
            raise ValueError("Missing bound H5 learning figure: " + name)
    if diagnostics:
        saved = json.loads((capsule / "diagnostics.json").read_text())
        for name, key in (
            ("source_ledger", "source_ledger_binding"),
            ("audit_receipt", "audit_receipt_binding"),
        ):
            if receipt[key] != saved[name]:
                raise ValueError("Rendered H5 declared external binding differs: " + name)


def verify_student_consequence_report(root: Path, capsule: Path, report: Path) -> None:
    """Check the completed C1 capsule's saved-only report and all 18k points."""
    root, capsule, report = root.resolve(), capsule.resolve(), report.resolve()
    run = capsule / "run"
    metadata = json.loads((capsule / "capsule.json").read_text())
    result = json.loads((run / "worker_result.json").read_text())
    audit_path = capsule / "audit/audit.json"
    audit = json.loads(audit_path.read_text())
    receipt = json.loads((report / "receipt.json").read_text())
    if (
        result.get("status") != "COMPLETED_FIXED_C1_ROSTER"
        or result.get("learning_measured") is not True
        or result.get("standalone_measured") is not True
        or metadata.get("scientific_contrast_completed") is not True
        or metadata.get("full_scientific_audit_passed") is not True
        or audit.get("passed") is not True
        or audit.get("full_scientific_audit_passed") is not True
        or audit.get("scientific_contrast_completed") is not True
    ):
        raise ValueError("C1 clean-export validation requires the completed audited roster")
    if (
        receipt.get("C1_audit_bound") is not True
        or receipt.get("C1_audit_association") != "worker_result_sha256_verified"
        or receipt.get("C1_config_association") != "config_sha256_verified"
    ):
        raise ValueError("Rendered C1 audit association differs")
    if any(
        type(receipt.get(key)) is not int or receipt[key] != 0
        for key in ("new_neural_rows", "new_native_controls", "new_optimizer_steps")
    ):
        raise ValueError("Saved C1 report must perform zero scientific calls")
    if receipt.get("incomplete_JSONL_fragments") != []:
        raise ValueError("Rendered C1 report contains incomplete saved evidence")
    closure_path = capsule / "accounting/audit_launcher_failure.json"
    closure = None
    if receipt.get("overall_audit_closure_passed") is False:
        if (
            not closure_path.is_file()
            or receipt.get("scientific_receipt_checks_passed") is not True
        ):
            raise ValueError("Rendered C1 failed closure lacks its explicit accounting companion")
        closure = json.loads(closure_path.read_text())
        expected_closure = dict(
            schema="walker-c1-audit-launcher-failure-v1",
            overall_audit_closure_passed=False,
            audit_coordinator_failed=True,
            original_audit_exit_exists=False,
            original_child_exit_code=None,
            coordinator_exit_code=1,
        )
        if (
            any(
                key not in closure or type(closure[key]) is not type(value) or closure[key] != value
                for key, value in expected_closure.items()
            )
            or receipt.get("audit_closure_status") != "AUDIT_COORDINATOR_FAILED"
        ):
            raise ValueError("Rendered C1 launcher failure facts differ")
        original = closure.get("scientific_receipt", {})
        if (
            original.get("bytes") != audit_path.stat().st_size
            or original.get("sha256") != hashlib.sha256(audit_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Rendered C1 failure companion binds a different scientific receipt")
        commit = closure.get("frozen_source_commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
        ):
            raise ValueError("Rendered C1 failure companion lacks frozen-source provenance")
        for key in ("traceback", "gnu_time"):
            declared = closure.get(key, {})
            matches = [
                name
                for name, value in metadata.get("files", {}).items()
                if value.get("bytes") == declared.get("bytes")
                and value.get("sha256") == declared.get("sha256")
            ]
            if type(declared.get("bytes")) is not int or declared["bytes"] <= 0 or not matches:
                raise ValueError("Rendered C1 failure evidence is not tracked: " + key)
            for name in matches:
                path = (capsule / name).resolve()
                if (
                    not path.is_relative_to(capsule)
                    or not path.is_file()
                    or path.stat().st_size != declared["bytes"]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != declared["sha256"]
                ):
                    raise ValueError("Rendered C1 tracked failure evidence differs: " + key)
    elif closure_path.is_file():
        raise ValueError("Rendered C1 report did not preserve the failed overall audit closure")

    names = (
        "config.json",
        "manifest.json",
        "worker_result.json",
        "worker_exit.json",
        "input_manifest.json",
        "prerequisites.json",
        "prepared/preparation_receipt.json",
        "worker/replay_gate.json",
        "worker/contrast_gate.json",
        "worker/metrics.json",
        "worker/actor_registry.json",
        "worker/fit_records.json",
        "worker/outcomes.json",
        "worker/evaluations.jsonl",
    )
    expected_inputs = {(run / name).resolve() for name in names if (run / name).is_file()}
    expected_inputs.update(path.resolve() for path in (run / "worker/learning").glob("*/*.json"))
    expected_inputs.add(audit_path.resolve())
    if closure is not None:
        expected_inputs.add(closure_path.resolve())
    actual_inputs = {}
    for name, binding in receipt["inputs"].items():
        path = Path(name)
        path = (path if path.is_absolute() else root / path).resolve()
        if path not in expected_inputs or path in actual_inputs:
            raise ValueError("Rendered C1 input roster differs")
        if (
            path.stat().st_size != binding["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]
        ):
            raise ValueError("Rendered C1 input binding differs: " + name)
        actual_inputs[path] = binding
    if set(actual_inputs) != expected_inputs:
        raise ValueError("Rendered C1 input roster differs")
    for name in ("config", "worker_result"):
        binding = actual_inputs[(run / (name + ".json")).resolve()]
        if audit.get(name + "_sha256") != binding["sha256"]:
            raise ValueError("Rendered C1 audit/result binding differs: " + name)
    if receipt.get("saved_JSON_read_bytes") != sum(row["bytes"] for row in actual_inputs.values()):
        raise ValueError("Rendered C1 input-read census differs")
    source = root / "src/gps/mppi_gps/reporting/student_consequence.py"
    if (
        receipt["source_code"]["bytes"] != source.stat().st_size
        or receipt["source_code"]["sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
    ):
        raise ValueError("Rendered C1 source binding differs")

    records = result["fit_records"]
    aliases = {f"s{s}_{arm}" for s in range(3) for arm in ("uniform", "cost_effect", "shuffled")}
    if set(records) != aliases or metadata.get("preserved_learning_points") != 18000:
        raise ValueError("Saved C1 learning roster differs")
    fields = (
        "update",
        "fit_update",
        "action_mse",
        "weighted_action_mse",
        "proximal_loss",
        "total_loss",
        "unclipped_gradient_norm",
    )
    expected_points = {}
    for alias, record in records.items():
        points = record["fitting_evidence"]["learning_curve"]
        if len(points) != 2000:
            raise ValueError("Saved C1 child curve lacks its fixed 2000 updates")
        for index, point in enumerate(points, 1):
            if point["fit_update"] != index or point["update"] != 2000 + index:
                raise ValueError("Saved C1 child update identity differs")
            expected_points[alias, index] = point

    tables = {}
    for name in ("curves", "outcomes", "metrics"):
        with (report / (name + ".csv")).open(newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    seen = set()
    for row in tables["curves"]:
        key = row["actor"], int(row["fit_update"])
        if key not in expected_points or key in seen:
            raise ValueError("Rendered C1 curve point roster differs")
        seen.add(key)
        for field in fields:
            saved = expected_points[key].get(field)
            if (saved is None and row[field] != "") or (
                saved is not None and float(row[field]) != saved
            ):
                raise ValueError("Rendered C1 saved curve value differs: " + field)
    if seen != set(expected_points):
        raise ValueError("Rendered C1 curve point roster differs")
    expected_counts = dict(curves=18000, outcomes=36 + 432 + 12, metrics=12 * 3)
    if (
        receipt.get("table_rows") != expected_counts
        or {name: len(rows) for name, rows in tables.items()} != expected_counts
    ):
        raise ValueError("Rendered C1 table row counts differ")
    required = {
        "curves.csv",
        "outcomes.csv",
        "metrics.csv",
        "report.md",
        "costs.json",
        "learning_curves.png",
        "learning_curves.svg",
        "standalone_outcomes.png",
        "standalone_outcomes.svg",
    }
    if not required <= set(receipt["outputs"]):
        raise ValueError("Rendered C1 table or figure lacks an output binding")
    verify_report_outputs(report, receipt["outputs"])
    if any((report / name).stat().st_size == 0 for name in required):
        raise ValueError("Rendered C1 table or figure is empty")
    if receipt.get("output_hash_read_bytes") != sum(
        row["bytes"] for row in receipt["outputs"].values()
    ):
        raise ValueError("Rendered C1 output-read census differs")
    costs = json.loads((report / "costs.json").read_text())
    if costs.get("worker") != result.get("costs") or costs.get("audit") != audit:
        raise ValueError("Rendered C1 cost/audit copy differs")
    if closure is not None and costs.get("audit_launcher_failure") != closure:
        raise ValueError("Rendered C1 closure accounting copy differs")


CLOSURE_SCIENCE_KEYS = (
    "neural_rows",
    "prediction_calls",
    "checkpoint_tensor_loads",
    "native_controls",
    "physics_substeps",
    "backward_rows",
    "optimizer_steps",
    "resets",
    "restores",
    "mppi_queries",
)


def verify_student_visitation_report(capsule: Path, report: Path) -> None:
    """Compare saved V replay with its audited outcomes and sealed summary."""
    audit = json.loads((capsule / "audit/result/audit.json").read_text())
    receipt = json.loads((report / "report_receipt.json").read_text())
    if (
        audit.get("passed") is not True
        or audit.get("full_scientific_audit_passed") is not True
        or receipt.get("saved_only") is not True
        or receipt.get("audit_passed") is not True
        or receipt.get("no_automatic_learning") is not True
    ):
        raise ValueError("V report replay requires audited saved-only evidence without learning")
    work = receipt.get("new_scientific_work", {})
    if set(work) != set(CLOSURE_SCIENCE_KEYS) or any(
        type(value) is not int or value != 0 for value in work.values()
    ):
        raise ValueError("Saved V report must perform zero scientific calls")
    outcomes = audit.get("outcomes")
    if (
        not isinstance(outcomes, list)
        or not 0 < len(outcomes) <= 224
        or any(
            type(row.get("controls")) is not int or not 0 <= row["controls"] <= 1000
            for row in outcomes
        )
    ):
        raise ValueError("The audited V actual-outcome roster is unavailable or invalid")
    controls = sum(row["controls"] for row in outcomes)
    if receipt.get("outcomes") != len(outcomes) or receipt.get("prefix_points") != controls:
        raise ValueError("Rendered V outcome/prefix census differs from the independent audit")
    for filename, expected in (("outcomes.csv", len(outcomes)), ("actual_prefixes.csv", controls)):
        with (report / filename).open(newline="") as stream:
            if sum(1 for _ in csv.DictReader(stream)) != expected:
                raise ValueError("Rendered V CSV census differs: " + filename)
    summaries = []
    for path in (capsule / "report/outcome_summary.csv", report / "outcome_summary.csv"):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            summaries.append((reader.fieldnames, list(reader)))
    if summaries[0] != summaries[1]:
        raise ValueError("Regenerated V summary differs rowwise from the sealed report")
    names = {"prefixes_full.png", "prefixes_first100.png"}
    if {path.name for path in report.glob("*.png")} != names or any(
        not (report / name).is_file() or (report / name).stat().st_size == 0 for name in names
    ):
        raise ValueError("V saved report must retain both nonempty all-prefix figures")


SIMILARITY_CLOSING_HASHES = {
    "files_manifest.json": "65e0d63120b68e2fec6a88823e27503ce3d61839a032767f88994375b2f079c2",
    "collection_exit.json": "18dab5ead75750458c64c617a038468818b8996db91d16940c89010d0f40ad80",
}
SIMILARITY_SCIENCE_KEYS = (
    "neural_rows",
    "prediction_calls",
    "native_controls",
    "model_controls",
    "physics_substeps",
    "checkpoint_tensor_loads",
    "backward_rows",
    "optimizer_steps",
    "policy_constructions",
)


def similarity_export_commands(root: Path, outputs: Path) -> list[list[str]]:
    """Add only saved-data J1 commands; older exports retain their existing roster."""
    capsule = root / "results/similarity_j1"
    if not (capsule / "capsule.json").is_file():
        return []
    materializer = capsule / "publication_tools/materialize.py"
    if not materializer.is_file():
        raise ValueError("J1 capsule is missing its sealed publication materializer")
    replay = (outputs / "similarity_materialized").resolve()
    report = (outputs / "similarity").resolve()
    return [
        ["scripts/train_mppi_gps.py", "study", "similarity-investigation", "--help"],
        [
            "-m",
            "src.gps.mppi_gps.reporting.similarity_capsule",
            "--capsule",
            str(capsule.resolve()),
            "--out",
            str(replay),
            "--manifest-sha256",
            SIMILARITY_CLOSING_HASHES["files_manifest.json"],
            "--exit-sha256",
            SIMILARITY_CLOSING_HASHES["collection_exit.json"],
        ],
        [
            "-m",
            "src.gps.mppi_gps.reporting.similarity",
            "--run",
            str(replay),
            "--out",
            str(report),
            "--audit",
            str(replay / "audit/result"),
            "--atlas",
            str(replay / "atlas/result"),
        ],
    ]


def verify_similarity_report(
    root: Path,
    capsule: Path,
    replay: Path,
    report: Path,
    *,
    expected_cases: int = 32,
    expected_updates: int = 2000,
    closing_hashes: dict[str, str] | None = None,
) -> None:
    """Check the saved-only J1 replay, every original curve and every rendered row."""
    root, capsule, replay, report = (path.resolve() for path in (root, capsule, replay, report))
    hashes = SIMILARITY_CLOSING_HASHES if closing_hashes is None else closing_hashes
    for name, expected in hashes.items():
        if hashlib.sha256((capsule / name).read_bytes()).hexdigest() != expected:
            raise ValueError("J1 actual compact closing hash differs: " + name)
    manifest = json.loads((capsule / "files_manifest.json").read_text())
    verify_report_outputs(capsule, manifest["files"])
    materialization = json.loads((replay / "materialization_receipt.json").read_text())
    if (
        materialization.get("passed") is not True
        or materialization.get("new_scientific_work") != 0
        or type(materialization.get("new_scientific_work")) is not int
        or materialization.get("immutable_capsule_modified") is not False
        or materialization.get("manifest_sha256") != hashes["files_manifest.json"]
        or materialization.get("collection_exit_sha256") != hashes["collection_exit.json"]
    ):
        raise ValueError("J1 requires a verified fresh saved-only materialization")
    verify_report_outputs(replay, materialization["files"])
    compact = json.loads((capsule / "capsule.json").read_text())
    original_bindings = {}
    for name, binding in compact["files"].items():
        compressed = binding.get("encoding") == "gzip_exact_original_bytes"
        target = name[:-3] if compressed and name.endswith(".gz") else name
        original_bindings[target] = dict(
            bytes=binding["original_bytes"] if compressed else binding["bytes"],
            sha256=binding["original_sha256"] if compressed else binding["sha256"],
        )
    verify_report_outputs(replay, original_bindings)
    run = replay / "run"
    config = json.loads((run / "config.json").read_text())
    result = json.loads((run / "worker_result.json").read_text())
    cases = json.loads((run / "worker/cases.json").read_text())
    audit = json.loads((replay / "audit/result/audit.json").read_text())
    receipt = json.loads((report / "report_receipt.json").read_text())
    if (
        len(cases) != expected_cases
        or config.get("learning", {}).get("updates") != expected_updates
        or len({case["case_id"] for case in cases}) != expected_cases
        or receipt.get("actual_cases") != expected_cases
        or receipt.get("completed_fits") != expected_cases
        or receipt.get("learning_cases_with_curves") != expected_cases
        or receipt.get("learning_points") != expected_cases * expected_updates
    ):
        raise ValueError("J1 completed case/learning-point roster differs")
    if (
        receipt.get("saved_only") is not True
        or receipt.get("no_followup_launched") is not True
        or receipt.get("audit_passed") is not True
        or receipt.get("full_scientific_audit_passed") is not True
        or receipt.get("audit_association_verified") is not True
        or audit.get("passed") is not True
        or audit.get("full_scientific_audit_passed") is not True
        or receipt.get("status") != result.get("status")
        or receipt.get("objective_passed") is not result.get("objective_passed")
    ):
        raise ValueError("J1 replay audit/status association differs")
    work = receipt.get("new_scientific_work", {})
    if set(work) != set(SIMILARITY_SCIENCE_KEYS) or any(
        type(value) is not int or value != 0 for value in work.values()
    ):
        raise ValueError("J1 saved replay must perform zero scientific calls")
    for key, name in (
        ("config_sha256", "config.json"),
        ("source_manifest_sha256", "manifest.json"),
        ("input_manifest_sha256", "input_manifest.json"),
        ("worker_result_sha256", "worker_result.json"),
    ):
        if audit.get(key) != hashlib.sha256((run / name).read_bytes()).hexdigest():
            raise ValueError("J1 independent audit input hash differs: " + name)
    source = root / "src/gps/mppi_gps/reporting/similarity.py"
    if receipt.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("J1 saved reporter source binding differs")
    expected_points = {}
    expected_inputs = {
        (run / name).resolve()
        for name in (
            "config.json",
            "manifest.json",
            "input_manifest.json",
            "worker_result.json",
            "worker_process_receipt.json",
            "worker/teachers.json",
            "worker/cases.json",
            "worker/score_definitions.json",
            "worker/calibration.json",
            "worker/predictive_comparison.json",
            "worker/score_models.json",
            "worker/probe_summary.json",
            "worker/score_seals/development.json",
            "worker/score_seals/protected.json",
        )
        if (run / name).is_file()
    }
    expected_inputs.update(
        (replay / name).resolve()
        for name in ("audit/result/audit.json", "atlas/result/atlas_receipt.json")
    )
    for case in cases:
        fit_path = (run / case["fit_receipt_file"]).resolve()
        if not fit_path.is_relative_to(run):
            raise ValueError("J1 curve path escaped the materialized run")
        fit = json.loads(fit_path.read_text())
        curve_path = fit_path.with_name("learning_curve.json")
        expected_inputs.update((fit_path, curve_path))
        curve = json.loads(curve_path.read_text())["learning_curve"]
        if fit.get("updates") != expected_updates or len(curve) != expected_updates:
            raise ValueError("J1 fixed-update curve length differs: " + case["case_id"])
        for ordinal, point in enumerate(curve, 1):
            if point.get("update") != ordinal or point.get("fit_update") != ordinal:
                raise ValueError("J1 saved curve update identity differs")
            expected_points[case["case_id"], ordinal] = (case, point)
    fields = (
        "update",
        "fit_update",
        "action_mse",
        "proximal_loss",
        "total_loss",
        "unclipped_gradient_norm",
    )
    seen = set()
    with (report / "learning_curves.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["case_id"], int(row["fit_update"])
            if key not in expected_points or key in seen:
                raise ValueError("J1 rendered learning-point roster differs")
            seen.add(key)
            case, point = expected_points[key]
            if row["fit_status"] != "completed_fit" or any(
                row[field] != str(case[field]) for field in ("group", "learner", "split", "source")
            ):
                raise ValueError("J1 rendered curve case join differs")
            for field in fields:
                value = point.get(field)
                if (value is None and row[field] != "") or (
                    value is not None and float(row[field]) != value
                ):
                    raise ValueError("J1 rendered curve value differs: " + field)
    if seen != set(expected_points):
        raise ValueError("J1 rendered learning-point roster differs")
    for name, binding in receipt["input_files"].items():
        path = Path(name).resolve()
        if not path.is_relative_to(replay):
            raise ValueError("J1 replay opened an input outside fresh materialization")
        verify_report_outputs(replay, {str(path.relative_to(replay)): binding})
    if {Path(name).resolve() for name in receipt["input_files"]} != expected_inputs:
        raise ValueError("J1 saved reporter input roster differs")
    if receipt["read_bytes"] != sum(row["bytes"] for row in receipt["input_files"].values()):
        raise ValueError("J1 saved input-read census differs")
    required = {
        "cases.csv",
        "outcome_summary.csv",
        "teachers.csv",
        "score_table.csv",
        "score_outcome_pairs.csv",
        "learning_curves.csv",
        "scores_vs_motion.png",
        "scores_vs_walking.png",
        "learning_curves.png",
        "saved_measurements.json",
        "report.md",
    }
    if not required <= set(receipt["outputs"]):
        raise ValueError("J1 rendered table/figure/report binding is missing")
    verify_report_outputs(report, receipt["outputs"])
    sealed = json.loads((replay / "report/report_receipt.json").read_text())
    verify_report_outputs(replay / "report", sealed["outputs"])
    for name in sorted(required):
        if not (report / name).stat().st_size:
            raise ValueError("J1 rendered output is empty: " + name)
        if (
            name.endswith(".csv")
            and (report / name).read_bytes() != (replay / "report" / name).read_bytes()
        ):
            raise ValueError("J1 regenerated table differs from sealed saved rows: " + name)


def _closure_payload(capsule: Path, relative: str) -> tuple[bytes, dict]:
    path = (capsule / relative).resolve()
    if (
        not path.is_relative_to(capsule.resolve())
        or not path.is_file()
        or (capsule / relative).is_symlink()
        or path.stat().st_size > 4 * 1024**2
    ):
        raise ValueError("Invalid compact closure input: " + relative)
    payload = path.read_bytes()
    return payload, dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def _closure_json(capsule: Path, relative: str) -> dict:
    return json.loads(_closure_payload(capsule, relative)[0])


def _closure_require(document: dict, expected: dict, label: str) -> None:
    if not isinstance(document, dict) or any(
        key not in document or type(document[key]) is not type(value) or document[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("Closure metadata differs: " + label)


def _closure_zero(document: dict) -> None:
    work = document.get("new_scientific_work", {})
    if (
        not isinstance(work, dict)
        or set(work) != set(CLOSURE_SCIENCE_KEYS)
        or any(type(value) is not int or value != 0 for value in work.values())
    ):
        raise ValueError("Closure smoke test cannot admit new scientific work")


def _closure_binding(
    capsule: Path, binding: dict, relative: str, *, declared_file: str | None = None
) -> None:
    actual = _closure_payload(capsule, "run/" + relative)[1]
    if (
        not isinstance(binding, dict)
        or binding.get("file") != (relative if declared_file is None else declared_file)
        or type(binding.get("bytes")) is not int
        or {key: binding.get(key) for key in actual} != actual
    ):
        raise ValueError("Closure receipt binding differs: " + relative)


def verify_consequence_recovery_capsules(root: Path, inspections: dict[str, dict]) -> None:
    """Check portable A1 failure/A2 admission certificates, never predecessor payloads."""
    a1_name = "student_consequence_c1_audit_recovery_a1"
    a2_name = "student_consequence_c1_certificate_closure_a2"
    for name in (a1_name, a2_name):
        capsule = root / "results" / name
        if not (capsule / "capsule.json").is_file():
            continue
        view = inspections.get(name, {})
        if (
            view.get("verification_requested") is not True
            or view.get("errors") != []
            or view.get("hash_mismatches") != []
        ):
            raise ValueError("Closure capsule lacks its actual verified inspector smoke: " + name)
        recovery = view.get("audit_recovery", {})
        if name == a1_name:
            result = _closure_json(capsule, "run/result/recovery.json")
            _closure_require(
                result,
                dict(
                    schema="walker-c1-audit-recovery-result-v1",
                    recovery_checks_passed=True,
                    admitted_original_scientific_checks=True,
                    retained_numerical_certification=True,
                    original_overall_audit_closure_passed=False,
                    original_child_exit_code=None,
                    original_audit_coordinator_failed=True,
                    verified_file_count=765,
                ),
                "preserved A1 checks",
            )
            _closure_zero(result)
            outer = _closure_json(capsule, "run/exit.json")
            _closure_require(
                outer,
                dict(
                    exit_code=0,
                    stopping_reason="coordinator_exception",
                    leader_reaped=True,
                    process_group_kill_requested=True,
                    unknown_partial_work_possible=True,
                ),
                "preserved A1 failure",
            )
            _closure_require(
                recovery,
                dict(
                    audit_recovery_resolved=False,
                    recovery_checks_passed=True,
                    original_overall_audit_closure_passed=False,
                    original_child_exit_code=None,
                ),
                "A1 inspector failure scope",
            )
            for absent in (
                "admission.json",
                "admission_candidate.json",
                "coordinator_exit.json",
                "finalizer_exit.json",
            ):
                if (capsule / "run" / absent).exists():
                    raise ValueError("Historical A1 missing closure was replaced: " + absent)
            _closure_binding(
                capsule,
                result["files_catalog"],
                "result/verified_files.json",
                declared_file="verified_files.json",
            )
            catalog = _closure_json(capsule, "run/result/verified_files.json")
            if len(catalog.get("files", {})) != 765:
                raise ValueError("Preserved A1 verified-file roster differs")
            failure = _closure_json(capsule, "operator/postmortem_001/external_failure.json")
            _closure_require(
                failure,
                dict(
                    schema="walker-c1-a1-external-failure-v1",
                    external_tool_exit_code=1,
                    original_coordinator_exit_available=False,
                    original_candidate_exists=False,
                    original_admission_exists=False,
                    worker_exit_recorded_zero=True,
                    worker_coordinator_exception_flags_preserved=True,
                ),
                "A1 external failure",
            )
        else:
            _closure_require(
                recovery,
                dict(
                    audit_recovery_resolved=True,
                    status="REPORTED_CERTIFICATE_RECOVERY_ACCEPTED_METADATA_VERIFIED",
                    retained_A1_integrity_certification=True,
                    retained_original_numerical_certification=True,
                    original_A1_admission_failed=True,
                    original_C1_coordinator_failed=True,
                    original_C1_child_exit_code=None,
                    original_A1_failure_flags_preserved=True,
                ),
                "A2 inspector admission",
            )
            protocol = _closure_json(capsule, "run/protocol.json")
            result = _closure_json(capsule, "run/result/certificate_result.json")
            admission = _closure_json(capsule, "run/admission.json")
            candidate = _closure_json(capsule, "run/admission_candidate.json")
            retained = dict(
                certificate_checks_passed=True,
                retained_A1_integrity_certification=True,
                retained_original_numerical_certification=True,
                original_A1_admission_failed=True,
                original_C1_coordinator_failed=True,
                original_C1_child_exit_code=None,
                original_A1_failure_flags_preserved=True,
            )
            _closure_require(
                result,
                dict(schema="walker-c1-certificate-closure-result-v1", **retained),
                "A2 result",
            )
            _closure_require(
                admission,
                dict(
                    schema="walker-c1-certificate-admission-v1",
                    audit_recovery_resolved=True,
                    **retained,
                ),
                "A2 admission",
            )
            _closure_require(
                candidate,
                dict(
                    schema="walker-c1-certificate-candidate-v1",
                    audit_recovery_resolved=False,
                    **retained,
                ),
                "A2 candidate",
            )
            names = dict(
                result="result/certificate_result.json",
                protocol="protocol.json",
                source_manifest="source_manifest.json",
                worker_exit="worker_exit.json",
                coordinator_exit="coordinator_exit.json",
            )
            if set(admission.get("bindings", {})) != set(names) or set(
                candidate.get("bindings", {})
            ) != set(names) - {"coordinator_exit"}:
                raise ValueError("A2 admission binding roster differs")
            for key, relative in names.items():
                _closure_binding(capsule, admission["bindings"][key], relative)
                if key != "coordinator_exit":
                    _closure_binding(capsule, candidate["bindings"][key], relative)
            _closure_binding(capsule, admission["admission_candidate"], "admission_candidate.json")
            for document in (protocol, result, candidate, admission):
                _closure_zero(document)
            for name, limit in (
                ("worker_exit.json", 30),
                ("coordinator_exit.json", 45),
                ("finalizer_exit.json", 15),
            ):
                outer = _closure_json(capsule, "run/" + name)
                _closure_require(
                    outer,
                    dict(
                        exit_code=0,
                        leader_reaped=True,
                        stopping_reason=None,
                        process_group_kill_requested=False,
                        unknown_partial_work_possible=False,
                    ),
                    name,
                )
                if (
                    type(outer.get("wall_seconds")) not in (int, float)
                    or not 0 <= outer["wall_seconds"] <= limit
                ):
                    raise ValueError("A2 closing time differs: " + name)
            canonical = hashlib.sha256(
                json.dumps(
                    protocol, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            source_manifest = _closure_json(capsule, "run/source_manifest.json")
            if (
                result.get("protocol_sha256") != canonical
                or source_manifest.get("protocol_sha256")
                != _closure_payload(capsule, "run/protocol.json")[1]["sha256"]
            ):
                raise ValueError("A2 protocol hash contracts differ")
            anchors, verified = protocol.get("anchors", {}), result.get("verified_certificates", {})
            if (
                not anchors
                or anchors != verified
                or result.get("work", {}).get("checked_certificates") != len(anchors)
            ):
                raise ValueError("A2 retained certificate roster differs")
            for key, anchor in anchors.items():
                suffix = Path(anchor["file"]).suffix
                actual = _closure_payload(capsule, "inputs/" + key + suffix)[1]
                if {field: anchor.get(field) for field in actual} != actual:
                    raise ValueError("A2 copied certificate differs: " + key)
            operator = _closure_json(capsule, "operator/launch_001/receipt.json")
            _closure_require(
                operator,
                dict(
                    audit_recovery_resolved=True,
                    original_C1_failure_retained=True,
                    original_A1_admission_remains_failed=True,
                    admission_sha256=_closure_payload(capsule, "run/admission.json")[1]["sha256"],
                    source_commit=source_manifest["source_commit"],
                    certificate_result=result,
                ),
                "A2 external receipt/admission association",
            )
            if json.dumps(
                operator["certificate_result"], sort_keys=True, allow_nan=False
            ) != json.dumps(result, sort_keys=True, allow_nan=False):
                raise ValueError("A2 external receipt copied result differs in JSON value or type")
            _closure_zero(operator)
            if (
                type(operator.get("stage_wall_seconds")) not in (int, float)
                or not 0 <= operator["stage_wall_seconds"] <= 60
            ):
                raise ValueError("A2 external receipt exceeds the whole-stage budget")
            a1 = root / "results" / a1_name
            if (a1 / "capsule.json").is_file():
                for key, relative in (
                    ("a1_result", "run/result/recovery.json"),
                    ("a1_worker_exit", "run/exit.json"),
                    ("a1_verified_files", "run/result/verified_files.json"),
                ):
                    actual = _closure_payload(a1, relative)[1]
                    if {field: anchors[key].get(field) for field in actual} != actual:
                        raise ValueError("A2 changes its preserved A1 certificate: " + key)
