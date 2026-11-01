"""Validate a clean Git export using the caller's locked CPU environment.

The export must contain no .git, runs or artifacts directory. All rendering uses
saved capsule data, and all generated files go to a new temporary directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
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
        ["scripts/train_mppi_gps.py", "study", "trajectory-review", "prepare", "--help"],
        ["scripts/train_mppi_gps.py", "study", "trajectory-review", "run", "--help"],
        ["scripts/audits/trajectory_review.py", "--help"],
        ["scripts/diagnostics/report_trajectory_review.py", "--help"],
        ["scripts/train_mppi_gps.py", "study", "motion-diagnosis", "--help"],
        ["scripts/audits/motion_diagnosis.py", "--help"],
        ["scripts/diagnostics/report_motion_diagnosis.py", "--help"],
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
