"""Public Walker commands for execution, frozen studies and read-only review."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUDIES = {
    "coupling-calibration": "src.gps.mppi_gps.coupling_calibration",
    "coupling-controller-check": "src.gps.mppi_gps.coupling_controller_check",
    "local-coupling-continuation": "src.gps.mppi_gps.experiments.local_coupling_multiround_continuation",
    "local-coupling-multiround": "src.gps.mppi_gps.experiments.local_coupling_multiround",
    "local-coupling-loop": "src.gps.mppi_gps.experiments.local_coupling_loop",
    "trajectory-pi": "src.gps.mppi_gps.experiments.trajectory_pi",
    "guided-data": "src.gps.mppi_gps.experiments.guided_data_comparison",
    "learnability": "src.gps.mppi_gps.experiments.student_learnability",
    "backoff": "src.gps.mppi_gps.experiments.adaptive_coupling",
    "compatibility": "src.gps.mppi_gps.experiments.action_compatibility",
    "width": "src.gps.mppi_gps.experiments.student_width",
    "update-budget": "src.gps.mppi_gps.experiments.student_update_budget",
    "transfer-diagnosis": "src.gps.mppi_gps.experiments.transfer_diagnosis",
    "source": "src.gps.mppi_gps.experiments.student_source",
    "source-evaluation": "src.gps.mppi_gps.experiments.student_source_evaluation",
    "trajectory-review": "src.gps.mppi_gps.experiments.trajectory_review",
    "motion-diagnosis": "src.gps.mppi_gps.experiments.motion_diagnosis",
    "teacher-proposal": "src.gps.mppi_gps.experiments.teacher_proposal",
    "student-consequence": "src.gps.mppi_gps.experiments.student_consequence",
    "visitation": "src.gps.mppi_gps.experiments.student_visitation",
    "similarity-investigation": "src.gps.mppi_gps.experiments.similarity_investigation",
    "jitter-diagnosis": "src.gps.mppi_gps.jitter_diagnosis",
}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    utilities = {
        "review": ("src.gps.mppi_gps.research_review", "main"),
        "walker-run": ("src.gps.mppi_gps.native_runner", "cli"),
        "walker-replay": ("src.gps.mppi_gps.native_replay", "cli"),
        "validate-export": ("src.gps.mppi_gps.validation.export", "main"),
    }
    if arguments and arguments[0] in utilities:
        module_name, entry_point = utilities[arguments[0]]
        module = importlib.import_module(module_name)
        return int(getattr(module, entry_point)(arguments[1:]) or 0)
    # Forward the study parser intact, including --help and its worker protocol.
    if len(arguments) >= 2 and arguments[0] == "study" and arguments[1] in STUDIES:
        module = importlib.import_module(STUDIES[arguments[1]])
        return int(module.main(arguments[2:]) or 0)

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("baseline", "train", "evaluate"):
        item = sub.add_parser(name)
        item.add_argument("--out", type=Path, required=True)
        item.add_argument("--config", type=Path, default=ROOT / "configs/mppi_gps_walker.json")
        item.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        if name == "evaluate":
            item.add_argument("--checkpoint", type=Path)
        if name == "train":
            item.add_argument("--stage", choices=("fit", "feedback"))
            item.add_argument("--experience", type=Path)
            item.add_argument("--fit-run", type=Path)
    item = sub.add_parser("render")
    item.add_argument("--run", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--episode")
    item = sub.add_parser("study", help="dispatch a frozen study workflow")
    item.add_argument("name", choices=tuple(STUDIES))
    item.add_argument("arguments", nargs=argparse.REMAINDER)
    item = sub.add_parser("inspect", help="read metadata without physics or inference")
    item.add_argument("--run", type=Path, required=True)
    item.add_argument("--json", action="store_true", dest="as_json")
    item.add_argument("--verify", action="store_true", help="check copied/source and input hashes")
    sub.add_parser("review", help="inventory source or inspect a saved candidate bank")
    sub.add_parser("walker-run", help="native Walker runner (historical runner interface)")
    sub.add_parser("walker-replay", help="verify a native runner trace with exact replay")
    sub.add_parser("validate-export", help="test a clean export and replay saved reports")
    args = parser.parse_args(arguments)
    if args.command == "inspect":
        from src.gps.mppi_gps.inspection import format_inspection, inspect_run

        try:
            report = inspect_run(args.run, verify=args.verify)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2) if args.as_json else format_inspection(report))
        return 1 if report["errors"] or report["hash_mismatches"] else 0

    from src.gps.mppi_gps import loop

    if args.command == "render":
        print(loop.render(args.run, args.out, args.episode))
        return 0
    if args.worker:
        result = loop.run(args.command, args.out)
        keys = (
            "command",
            "error",
            "standalone_qualified",
            "guidance_promoted",
            "feedback_rounds_completed",
            "scientific_loop_pass",
            "work",
            "wall_seconds",
        )
        print(json.dumps({key: result.get(key) for key in keys}, indent=2))
        return 0 if result["error"] is None else 1
    if args.command == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires --checkpoint")
    config = json.loads(args.config.read_text())
    if args.command == "train" and config["schema"] == "walker-cio-mppi-v2":
        if args.stage:
            config["stage"] = args.stage
        for key, value in (("experience_run", args.experience), ("fit_run", args.fit_run)):
            selected = value if value is not None else Path(config[key])
            config[key] = str((selected if selected.is_absolute() else ROOT / selected).resolve())
    out = loop.freeze(args.out, config, args.command, getattr(args, "checkpoint", None))
    command = [
        sys.executable,
        "-B",
        str(out / "source/scripts/train_mppi_gps.py"),
        args.command,
        "--out",
        str(out),
        "--worker",
    ]
    started = time.perf_counter()
    with (out / "stdout.log").open("x") as log:
        process = subprocess.run(
            command,
            cwd=out / "source",
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(
                os.environ,
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
            ),
        )
    loop.write_json(
        out / "launch.json",
        dict(
            command=command,
            exit_code=process.returncode,
            external_wall_seconds=time.perf_counter() - started,
            python=sys.executable,
        ),
    )
    print(json.dumps(dict(run=str(out), exit_code=process.returncode)))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
