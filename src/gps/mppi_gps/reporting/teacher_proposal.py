"""Render the H5 preparation or recorded first-round result without execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..provenance import sha, write


def render(run: Path, out: Path) -> dict:
    run, out = run.resolve(), out.resolve()
    root = run / "run" if (run / "capsule.json").is_file() else run
    if out.is_relative_to(run):
        raise ValueError("Reports must remain outside frozen evidence")
    if out.exists():
        raise FileExistsError(out)
    config = json.loads((root / "config.json").read_text())
    result_path = root / "worker_result.json"
    if not result_path.is_file():
        result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    if result_path.name == "result.json" and any(
        (root / name).exists() for name in ("launch_ownership.json", "worker_exit.json", "evidence")
    ):
        result.update(
            status="launch_attempt_without_worker_result",
            execution_complete=False,
            work=dict(completion_unknown=True, authoritative_prefix="evidence/ledger.jsonl"),
        )
    plan = json.loads((root / "call_census.json").read_text())
    out.mkdir(parents=True)
    text = [
        "# H5.0 preparation and execution status",
        "",
        f"Status: **{result.get('status')}**.",
        "",
        "The descriptive motion distance R, conditional action agreement D, and usefulness "
        "after a fixed2,000-update learning block have different roles. H5.0 holds D fixed "
        "and changes only its placement in the second search stream.",
        "",
        f"Launch authorized: `{config['launch_authorized']}`. "
        f"Useful-learning prerequisite satisfied: "
        f"`{config['eligibility']['useful_learning_prerequisite_satisfied']}`.",
        "",
        "| Full-roster reservation | Amount |",
        "|---|---:|",
        f"| Model controls | {plan['total']['model_controls']:,} |",
        f"| Live controls | {plan['total']['live_controls']:,} |",
        f"| Physics substeps | {plan['total']['physics_substeps']:,} |",
        f"| Worker neural rows | {plan['total']['worker_neural_rows']:,} |",
        f"| Independent audit neural rows | {plan['strong_audit_neural_rows']:,} |",
        f"| Adam updates | {plan['worker_optimizer_steps']:,} |",
        "",
        "These are prospective maxima, not claims of completed work. "
        "Preparation performs no inference, integration, fitting or acquisition.",
        "",
        "## Recorded work",
        "",
        "```json",
        json.dumps(result.get("work", {}), indent=2),
        "```",
        "",
        "## Launch blockers",
        "",
    ]
    text.extend("- " + value for value in result.get("launch_blockers", []))
    if result.get("schema") == "walker-teacher-proposal-round-v1":
        text.extend(
            [
                "",
                "## Recorded outcomes",
                "",
                "Worker claims require the separate full independent audit before admission.",
                "",
                "```json",
                json.dumps(
                    {
                        key: result.get(key)
                        for key in (
                            "teacher_summaries",
                            "teacher_admission",
                            "evaluations",
                            "primary_gate",
                            "confirmation_gate",
                            "paired_teacher_student_gap",
                            "confirmed_benefit",
                        )
                    },
                    indent=2,
                ),
                "```",
            ]
        )
    (out / "report.md").write_text("\n".join(text) + "\n")
    receipt = dict(
        schema="walker-h5-readonly-report-v1",
        source_result_sha256=sha(result_path),
        config_sha256=sha(root / "config.json"),
        outputs={
            "report.md": dict(
                bytes=(out / "report.md").stat().st_size, sha256=sha(out / "report.md")
            )
        },
        network_forwards=0,
        physics_controls=0,
        optimizer_steps=0,
    )
    write(out / "receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(render(args.run, args.out), indent=2))
    return 0
