"""Validate a clean Git export using the caller's locked CPU environment.

The export must contain no .git, runs or artifacts directory. All rendering uses
saved capsule data, and all generated files go to a new temporary directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


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
    ]
    capsules = ("guided_data_v3_1", "student_learnability_l1", "adaptive_coupling_h4_1")
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
