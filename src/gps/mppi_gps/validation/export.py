"""Validate a clean export using saved reports and the existing test suite."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .reports import (
    similarity_export_commands,
    verify_consequence_recovery_capsules,
    verify_report_outputs,
    verify_similarity_report,
    verify_student_consequence_report,
    verify_student_visitation_report,
    verify_teacher_proposal_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.export.resolve()
    for name in (".git", "runs", "artifacts", ".venv"):
        if (root / name).exists():
            parser.error(f"Not a clean export: {root / name} exists")
    if not (root / "uv.lock").is_file():
        parser.error("Export is missing its locked environment")
    outputs = Path(tempfile.mkdtemp(prefix="walker-export-validation-"))
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(outputs / "matplotlib"),
    )
    environment.pop("PYTHONPATH", None)
    commands = [
        ["-m", "pytest", "-q", "--basetemp", str(outputs / "pytest")],
        ["scripts/check_repository.py"],
        ["scripts/train_mppi_gps.py", "review", "--help"],
        ["scripts/train_mppi_gps.py", "review", "inventory", "--root", "."],
        ["scripts/train_mppi_gps.py", "walker-run", "--help"],
        ["scripts/train_mppi_gps.py", "walker-replay", "--help"],
        ["scripts/train_mppi_gps.py", "validate-export", "--help"],
        ["scripts/train_mppi_gps.py", "study", "trajectory-review", "prepare", "--help"],
        ["scripts/train_mppi_gps.py", "study", "trajectory-review", "run", "--help"],
        ["scripts/audits/trajectory_review.py", "--help"],
        ["scripts/diagnostics/report_trajectory_review.py", "--help"],
        ["scripts/train_mppi_gps.py", "study", "motion-diagnosis", "--help"],
        ["scripts/audits/motion_diagnosis.py", "--help"],
        ["scripts/diagnostics/report_motion_diagnosis.py", "--help"],
        ["scripts/train_mppi_gps.py", "study", "teacher-proposal", "--help"],
        ["scripts/train_mppi_gps.py", "study", "teacher-proposal", "census"],
        ["scripts/audits/teacher_proposal.py", "--help"],
        ["scripts/diagnostics/report_teacher_proposal.py", "--help"],
        ["scripts/train_mppi_gps.py", "study", "student-consequence", "--help"],
        ["scripts/train_mppi_gps.py", "study", "student-consequence", "preparation", "--help"],
        ["scripts/train_mppi_gps.py", "study", "student-consequence", "freeze", "--help"],
        ["scripts/train_mppi_gps.py", "study", "student-consequence", "launch", "--help"],
        ["scripts/audits/student_consequence.py", "--help"],
        ["scripts/diagnostics/report_student_consequence.py", "--help"],
    ]
    capsules = (
        "guided_data_v3_1",
        "student_learnability_l1",
        "adaptive_coupling_h4_1",
        "action_compatibility_k1",
        "student_width_w1",
        "student_update_w2",
        "transfer_diagnosis_d1d2",
    )
    if (root / "results/student_source_s1/capsule.json").is_file():
        capsules += ("student_source_s1",)
    if (root / "results/student_source_s1_1/capsule.json").is_file():
        capsules += ("student_source_s1_1",)
    if (root / "results/trajectory_review_k1_1/capsule.json").is_file():
        capsules += ("trajectory_review_k1_1",)
    if (root / "results/motion_diagnosis_k1_2/capsule.json").is_file():
        capsules += ("motion_diagnosis_k1_2",)
    if (root / "results/teacher_proposal_h5_0_preparation/capsule.json").is_file():
        capsules += ("teacher_proposal_h5_0_preparation",)
    if (root / "results/teacher_proposal_h5_0_round1/capsule.json").is_file():
        capsules += ("teacher_proposal_h5_0_round1",)
    if (root / "results/student_consequence_c1_0/capsule.json").is_file():
        capsules += ("student_consequence_c1_0",)
    for name in (
        "student_consequence_c1_audit_recovery_a1",
        "student_consequence_c1_certificate_closure_a2",
        "student_visitation_s1_2_v",
        "similarity_j1",
        "jitter_diagnosis_j2_001",
    ):
        if (root / "results" / name / "capsule.json").is_file():
            capsules += (name,)
    commands += [
        ["scripts/train_mppi_gps.py", "inspect", "--run", f"results/{name}", "--verify", "--json"]
        for name in capsules
    ]
    commands += [
        [
            "scripts/diagnostics/report_guided_data_comparison.py",
            "--run",
            "results/guided_data_v3_1/run",
            "--out",
            str(outputs / "guided"),
        ],
        [
            "scripts/diagnostics/report_walker_experiments.py",
            "--kind",
            "student",
            "--run",
            "results/student_learnability_l1/run",
            "--out",
            str(outputs / "student"),
        ],
        [
            "scripts/diagnostics/report_walker_experiments.py",
            "--kind",
            "coupling",
            "--run",
            "results/adaptive_coupling_h4_1/run",
            "--out",
            str(outputs / "coupling"),
        ],
    ]
    commands += [
        [
            "scripts/diagnostics/report_action_compatibility.py",
            "--run",
            "results/action_compatibility_k1/run",
            "--out",
            str(outputs / "compatibility"),
        ],
        [
            "scripts/diagnostics/report_width.py",
            "--run",
            "results/student_width_w1/run",
            "--out",
            str(outputs / "width"),
        ],
    ]
    commands.append(
        [
            "scripts/diagnostics/report_update_budget.py",
            "--run",
            "results/student_update_w2/run",
            "--out",
            str(outputs / "update_budget"),
        ]
    )
    commands.append(
        [
            "scripts/diagnostics/report_transfer_diagnosis.py",
            "--run",
            "results/transfer_diagnosis_d1d2",
            "--out",
            str(outputs / "transfer_diagnosis"),
        ]
    )
    if "student_source_s1" in capsules:
        source_report = [
            "scripts/diagnostics/report_student_source.py",
            "--run",
            "results/student_source_s1",
            "--out",
            str(outputs / "student_source"),
        ]
        source_capsule = json.loads((root / "results/student_source_s1/capsule.json").read_text())
        if source_capsule.get("report_requires_failed_audit_display"):
            source_report.append("--display-failed-audit")
        commands.append(source_report)
    if "student_source_s1_1" in capsules:
        commands.append(
            [
                "scripts/diagnostics/report_student_source_s1_1.py",
                "--run",
                "results/student_source_s1_1",
                "--out",
                str(outputs / "student_source_s1_1"),
            ]
        )
    if "trajectory_review_k1_1" in capsules:
        commands.append(
            [
                "scripts/diagnostics/report_trajectory_review.py",
                "--run",
                "results/trajectory_review_k1_1",
                "--out",
                str(outputs / "trajectory_review"),
            ]
        )
    if "motion_diagnosis_k1_2" in capsules:
        commands.append(
            [
                "scripts/diagnostics/report_motion_diagnosis.py",
                "--run",
                "results/motion_diagnosis_k1_2",
                "--out",
                str(outputs / "motion_diagnosis"),
            ]
        )
    if "teacher_proposal_h5_0_preparation" in capsules:
        commands.append(
            [
                "scripts/diagnostics/report_teacher_proposal.py",
                "--run",
                "results/teacher_proposal_h5_0_preparation",
                "--out",
                str(outputs / "teacher_proposal"),
            ]
        )
    if "teacher_proposal_h5_0_round1" in capsules:
        commands.append(
            [
                "scripts/diagnostics/report_teacher_proposal.py",
                "--run",
                "results/teacher_proposal_h5_0_round1",
                "--saved-diagnostics",
                "results/teacher_proposal_h5_0_round1/diagnostics.json",
                "--out",
                str(outputs / "teacher_proposal_round1"),
            ]
        )
    if "student_consequence_c1_0" in capsules:
        consequence_report = [
            "scripts/diagnostics/report_student_consequence.py",
            "--run",
            "results/student_consequence_c1_0",
            "--audit",
            "results/student_consequence_c1_0/audit/audit.json",
            "--out",
            str(outputs / "student_consequence"),
        ]
        closure = "results/student_consequence_c1_0/accounting/audit_launcher_failure.json"
        if (root / closure).is_file():
            consequence_report.extend(["--closure", closure])
        commands.append(consequence_report)
    if "student_visitation_s1_2_v" in capsules:
        commands.append(
            [
                "results/student_visitation_s1_2_v/render_report.py",
                "--out",
                str(outputs / "student_visitation"),
            ]
        )
    commands.extend(similarity_export_commands(root, outputs))
    if "jitter_diagnosis_j2_001" in capsules:
        commands.append(
            [
                "-m",
                "src.gps.mppi_gps.reporting.jitter_diagnosis",
                "--worker",
                "results/jitter_diagnosis_j2_001/run",
                "--audit",
                "results/jitter_diagnosis_j2_001/audit/audit_result.json",
                "--out",
                str(outputs / "jitter_diagnosis"),
            ]
        )
    if (root / "results/similarity_j1_review_001/capsule.json").is_file():
        commands.append(
            [
                "-m",
                "src.gps.mppi_gps.reporting.similarity_diagnostic",
                "--capsule",
                "results/similarity_j1",
                "--out",
                str(outputs / "similarity_review"),
            ]
        )
    if (root / "results/sac_diagnostics_review_001/capsule.json").is_file():
        commands.append(
            [
                "-m",
                "src.gps.mppi_gps.reporting.sac_diagnostics_review",
                "--capsule",
                "results/sac_diagnostics_review_001",
                "--output",
                str(outputs / "sac_diagnostics_review"),
            ]
        )
    inspections = {}
    for index, command in enumerate(commands):
        print(f"[{index + 1}/{len(commands)}] {command[0]}", flush=True)
        process = subprocess.run(
            [sys.executable, "-B", *command],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
        )
        (outputs / f"{index:02d}.stdout.txt").write_text(process.stdout)
        (outputs / f"{index:02d}.stderr.txt").write_text(process.stderr)
        if process.returncode:
            print(process.stdout)
            print(process.stderr, file=sys.stderr)
            print(f"Validation logs: {outputs}")
            return process.returncode
        if index < 2:
            print(process.stdout.strip())
        if command[:3] == ["scripts/train_mppi_gps.py", "inspect", "--run"]:
            inspections[Path(command[3]).name] = json.loads(process.stdout)
    for name in ("guided", "student", "coupling"):
        for extension in ("png", "pdf"):
            figure = outputs / name / f"learning_and_outcomes.{extension}"
            if not figure.is_file() or figure.stat().st_size == 0:
                raise ValueError(f"Missing rendered figure: {figure}")
    figures = (
        ("compatibility", "agreement_heatmaps"),
        ("width", "fitting_curves"),
        ("update_budget", "fitting_curves"),
        ("transfer_diagnosis", "support_distribution"),
    )
    if "student_source_s1" in capsules:
        figures += tuple(
            ("student_source", stem)
            for stem in (
                "reconstruction",
                "learning_curves",
                "standalone_outcomes",
                "cost_components",
            )
        )
    if "trajectory_review_k1_1" in capsules:
        figures += tuple(
            ("trajectory_review", stem)
            for stem in (
                "individual_bin_errors",
                "cumulative_agreement",
                "actuator_and_saturation",
                "variance_decomposition",
            )
        )
    for folder, stem in figures:
        for extension in ("png", "pdf"):
            figure = outputs / folder / f"{stem}.{extension}"
            if not figure.is_file() or figure.stat().st_size == 0:
                raise ValueError(f"Missing rendered figure: {figure}")
    if "trajectory_review_k1_1" in capsules:
        report = outputs / "trajectory_review"
        receipt = json.loads((report / "receipt.json").read_text())
        if receipt["scientific_calls"] != 0:
            raise ValueError("Saved trajectory review must perform zero scientific calls")
        for name, expected in receipt["inputs"].items():
            path = root / "results/trajectory_review_k1_1/run" / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Rendered trajectory-review input differs: {path}")
        source = root / "src/gps/mppi_gps/reporting/trajectory_review.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["source_sha256"]:
            raise ValueError("Rendered trajectory-review source differs")
        verify_report_outputs(report, receipt["outputs"])
    if "motion_diagnosis_k1_2" in capsules:
        report = outputs / "motion_diagnosis"
        receipt = json.loads((report / "receipt.json").read_text())
        if (
            not receipt["execution_complete"]
            or receipt["error"] is not None
            or any(receipt["scientific_calls"].values())
            or receipt["raw_read_bytes"] != 0
        ):
            raise ValueError("Clean motion report must use saved derived curves and bound assets")
        verify_report_outputs(report, receipt["outputs"])
        for name in ("full_motion_atlas.pdf", "derived_curve_atlas.pdf"):
            if not (report / name).is_file():
                raise ValueError("Missing motion atlas: " + name)
        for stem in (
            "all_trajectory_errors",
            "within_recurrence",
            "cross_trajectory_minima",
            "near_equal_pair",
        ):
            for suffix in ("png", "pdf"):
                if not (report / (stem + "." + suffix)).is_file():
                    raise ValueError("Missing motion figure: " + stem)
    if "teacher_proposal_h5_0_preparation" in capsules:
        verify_teacher_proposal_report(
            root,
            root / "results/teacher_proposal_h5_0_preparation",
            outputs / "teacher_proposal",
            diagnostics=False,
        )
    if "teacher_proposal_h5_0_round1" in capsules:
        verify_teacher_proposal_report(
            root,
            root / "results/teacher_proposal_h5_0_round1",
            outputs / "teacher_proposal_round1",
            diagnostics=True,
        )
    if "student_consequence_c1_0" in capsules:
        verify_student_consequence_report(
            root, root / "results/student_consequence_c1_0", outputs / "student_consequence"
        )
    if "student_visitation_s1_2_v" in capsules:
        verify_student_visitation_report(
            root / "results/student_visitation_s1_2_v", outputs / "student_visitation"
        )
    if "similarity_j1" in capsules:
        view = inspections.get("similarity_j1", {})
        if (
            view.get("verification_requested") is not True
            or view.get("errors") != []
            or view.get("hash_mismatches") != []
        ):
            raise ValueError("J1 capsule lacks its verified metadata inspector smoke")
        verify_similarity_report(
            root,
            root / "results/similarity_j1",
            outputs / "similarity_materialized",
            outputs / "similarity",
        )
    verify_consequence_recovery_capsules(root, inspections)
    if "jitter_diagnosis_j2_001" in capsules:
        capsule = root / "results/jitter_diagnosis_j2_001"
        report = outputs / "jitter_diagnosis"
        receipt = json.loads((report / "receipt.json").read_text())
        view = inspections["jitter_diagnosis_j2_001"]
        if (
            view.get("errors") != []
            or view.get("hash_mismatches") != []
            or not receipt["passed"]
            or receipt["actual_cases"] != 6
            or receipt["curve_points"] != 60000
            or receipt["primary_comparison_executed"]
            or any(receipt["new_scientific_work"].values())
        ):
            raise ValueError("J2 compact replay or metadata inspection differs")
        verify_report_outputs(report, receipt["output_files"])
        for name in ("cases.csv", "per_seed_primary.csv"):
            if (report / name).read_bytes() != (capsule / "report" / name).read_bytes():
                raise ValueError("J2 compact replay table differs: " + name)
        name = "learning_curves.csv.gz"
        if gzip.decompress((report / name).read_bytes()) != gzip.decompress(
            (capsule / "report" / name).read_bytes()
        ):
            raise ValueError("J2 compact replay changed a recorded learning point")
    for capsule_name, output_name, summary_name in (
        ("similarity_j1_review_001", "similarity_review", "summary.json"),
        ("sac_diagnostics_review_001", "sac_diagnostics_review", "reanalysis/summary.json"),
    ):
        capsule = root / "results" / capsule_name
        if (capsule / "capsule.json").is_file():
            expected = json.loads((capsule / summary_name).read_text())
            actual = json.loads((outputs / output_name / "summary.json").read_text())
            if actual != expected:
                raise ValueError(f"Saved-only review no longer reproduces: {capsule_name}")
    print(
        json.dumps(
            dict(
                export=str(root),
                validation_outputs=str(outputs),
                commands=len(commands),
                new_research_experiments=0,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
