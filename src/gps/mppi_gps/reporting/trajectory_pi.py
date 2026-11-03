"""Saved-only metadata, curves and physical traces for the P1 comparison.

Rendering imports plotting/array readers lazily. Neither entry point imports a
policy, restores a checkpoint, simulates, fits or queries an optimizer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from ..provenance import sha, write


def _read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def inspect(root: Path, *, verify: bool = False) -> dict:
    root = root.resolve()
    capsule = _read(root / "capsule.json", {})
    run = root / "run" if capsule else root
    config = _read(run / "config.json", {})
    result = _read(run / "result.json", {})
    launch = _read(run / "launch.json", {})
    evidence = root if capsule or root.name != "worker" else root.parent
    audit = {}
    for name in ("findings.json", "audit.json", "audit_result.json"):
        audit = _read(evidence / "audit" / name, {})
        if audit:
            break
    prerequisite = _read(evidence / "preflight" / "preflight.json", {})
    source_roster = _read(root / "frozen/source_manifest.json", {}) if capsule else {}
    files, missing, mismatches, errors = [], [], [], []
    declared = capsule.get("files", {})
    if isinstance(declared, list):
        declared = {row["path"]: row for row in declared}
    names = sorted(declared) if capsule else ["config.json", "launch.json", "result.json"]
    for name in names:
        path = root / name
        if path.resolve().is_relative_to(root) is False:
            errors.append("unsafe capsule path: " + name)
            continue
        exists = path.is_file()
        row = dict(file=name, available=exists, bytes=path.stat().st_size if exists else None)
        if not exists:
            missing.append(name)
        elif verify and name in declared:
            bound = declared[name]
            expected = bound if isinstance(bound, str) else bound.get("sha256")
            row["verified"] = sha(path) == expected
            if isinstance(bound, dict):
                row["verified"] &= row["bytes"] == bound.get("bytes", row["bytes"])
            if not row["verified"]:
                mismatches.append(name)
        files.append(row)
    history = result.get("history", [])
    latest = history[-1] if history else {}
    external = _read(root / "external_evidence.json", {})
    return dict(
        schema="trajectory-pi-inspection-1",
        status=result.get("status", "NO_CLOSED_RESULT"),
        kind="full-trajectory-pi",
        run_id=root.name,
        directory=str(root),
        config=config,
        result_file="result.json",
        outcomes=dict(
            status=result.get("status", "NO_CLOSED_RESULT"),
            error=result.get("error"),
            completed_round=latest.get("round"),
            criteria=result.get("criteria"),
            saved_audit_status=audit.get("status"),
            prerequisite_status=prerequisite.get("status"),
            scientific_result=audit.get("scientific_result"),
            similarity_test_executed=False if result.get("status") == "PRELAUNCH_STOPPED" else None,
            full_project_success=False,
        ),
        provenance=dict(
            source_file_count=len(source_roster["files"]) if "files" in source_roster else None,
            source_commit=source_roster.get("git_commit"),
            source_present=Path(launch.get("source_directory", "/missing")).is_dir(),
            input_manifest=launch.get("prepared_manifest"),
            source_export=launch.get("source_directory"),
        ),
        teacher_outcomes={
            stream: {arm: value.get("bank", []) for arm, value in arms.items()}
            for stream, arms in latest.get("streams", {}).items()
        },
        student_outcomes=latest.get("streams", {}),
        teacher_student_gap=result.get("criteria", {}),
        gates=dict(
            saved_audit=audit.get("status"),
            prerequisite=prerequisite.get("status"),
            audit_scope="Saved stop/accounting certification; failed replay is not validated"
            if audit.get("status") == "PASS_STOPPED"
            else "Saved audit declaration; see its explicit coverage",
            automatic_followup=False,
        ),
        costs=dict(
            recorded=result.get("accounting", {}),
            prerequisite=prerequisite.get("work", {}),
            limits=config.get("limits", {}),
        ),
        files=files,
        missing_evidence=missing,
        omitted_bulk_evidence=[],
        bulk_evidence=external,
        hash_mismatches=mismatches,
        errors=errors,
        verification_requested=verify,
        verification_scope="Compact saved metadata and declared capsule bytes only. No physics, inference, fitting or independent scientific recomputation; external raw evidence is not reopened.",
    )


def curve_rows(history: list[dict]) -> list[dict]:
    rows = []
    for round_record in history:
        for stream, arms in round_record.get("streams", {}).items():
            for arm, value in arms.items():
                bank = value.get("bank", [])
                raw = value.get("raw_candidates", [])
                student = value.get("student", {}).get("summary", {})
                gap = value.get("teacher_after_fit", value.get("teacher_before", {}))
                row = dict(
                    round=round_record["round"],
                    stream=int(stream),
                    arm=arm,
                    teacher_cost=mean([r["outcome"]["complete_cost"] for r in bank])
                    if bank
                    else None,
                    teacher_walking=sum(bool(r["outcome"]["walking"]) for r in bank),
                    teacher_survival=sum(bool(r["outcome"]["full_survival"]) for r in bank),
                    teacher_after_fit_rms=gap.get("rms"),
                    raw_candidate_cost=mean([r["result"]["outcome"]["complete_cost"] for r in raw])
                    if raw
                    else None,
                    raw_candidate_walking=sum(bool(r["result"]["outcome"]["walking"]) for r in raw),
                    student_walking=student.get("walking"),
                    student_survival=student.get("full_survival"),
                    student_cost=student.get("mean_cost"),
                    student_controls=student.get("mean_controls"),
                )
                for anchor in ("0", "100"):
                    query = value.get("queries", {}).get(anchor, {})
                    row.update(
                        {
                            f"query_{anchor}_" + key: query.get(key)
                            for key in ("present", "qualified", "action_gap", "policy_cost_gap")
                        }
                    )
                rows.append(row)
    return rows


def _array_trace(path: Path) -> dict:
    import numpy as np

    if path.suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as raw:
            return {key: raw[key][:] for key in ("states", "actions")}
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key].copy() for key in ("states", "actions")}


def render(run: Path, out: Path) -> dict:
    """Create figures from preserved values; a missing trace stays unavailable."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, out = run.resolve(), out.resolve()
    result = _read(run / "result.json", {})
    config = _read(run / "config.json", {})
    history = result.get("history", [])
    out.mkdir(parents=True, exist_ok=False)
    rows = curve_rows(history)
    with (out / "learning_curves.csv").open("x", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    missing, figures = [], []
    if rows:
        fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
        measures = [
            "teacher_cost",
            "teacher_after_fit_rms",
            "student_walking",
            "student_survival",
            "student_cost",
            "student_controls",
        ]
        for axis, key in zip(axes.flat, measures, strict=True):
            for stream in config["streams"]:
                for arm, color in (("independent", "tab:blue"), ("coupled", "tab:orange")):
                    subset = [r for r in rows if r["stream"] == stream and r["arm"] == arm]
                    axis.plot(
                        [r["round"] for r in subset],
                        [r[key] for r in subset],
                        color=color,
                        alpha=0.65,
                        label=f"s{stream} {arm}",
                    )
            axis.set_title(key.replace("_", " "))
            axis.grid(alpha=0.2)
            axis.set_xlabel("alternating round")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Saved P1 loop outcomes; survival and walking are separate")
        fig.tight_layout()
        fig.savefig(out / "loop_curves.png", dpi=130)
        plt.close(fig)
        figures.append("loop_curves.png")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for column, anchor in enumerate(("0", "100")):
            for stream in config["streams"]:
                for arm, color in (("independent", "tab:blue"), ("coupled", "tab:orange")):
                    subset = [r for r in rows if r["stream"] == stream and r["arm"] == arm]
                    axes[0, column].plot(
                        [r["round"] for r in subset],
                        [r[f"query_{anchor}_action_gap"] for r in subset],
                        color=color,
                        alpha=0.65,
                    )
                    axes[1, column].plot(
                        [r["round"] for r in subset],
                        [int(r[f"query_{anchor}_qualified"] or False) for r in subset],
                        color=color,
                        alpha=0.65,
                    )
            axes[0, column].set_title(f"anchor {anchor}: raw action gap")
            axes[1, column].set_title("query independently walks (missing shown separately in CSV)")
            axes[1, column].set_xlabel("round")
        fig.tight_layout()
        fig.savefig(out / "student_state_queries.png", dpi=130)
        plt.close(fig)
        figures.append("student_state_queries.png")
    # The two controllers use different starts unless a query explicitly shares
    # the actor context. Display both paths without implying a paired task gap.
    for record in history:
        for stream, arms in record["streams"].items():
            for arm, value in arms.items():
                sources = []
                for item in value.get("bank", []):
                    sources.append((f"teacher slot{item['slot']}", Path(item["trace"])))
                jobs = value.get("student", {}).get("jobs", [])
                if jobs:
                    sources.append(("standalone reset0", Path(jobs[0]["path"]) / "trace.h5"))
                query = value.get("queries", {}).get("0", {})
                if query.get("present"):
                    sources.append(
                        (
                            f"same-reset PI query (walks={query.get('qualified')})",
                            Path(query["path"]) / "trace.h5",
                        )
                    )
                fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
                plotted = False
                for label, path in sources:
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    data = _array_trace(path)
                    for joint, axis in enumerate(axes.flat):
                        axis.plot(
                            np.arange(len(data["states"])) * 0.008,
                            data["states"][:, 4 + joint],
                            label=label,
                            linewidth=0.8,
                        )
                        axis.set_title(f"joint {joint}")
                        axis.set_xlabel("elapsed seconds")
                    plotted = True
                if plotted:
                    axes[0, 0].legend(fontsize=8)
                    fig.suptitle(f"round {record['round']}, s{stream} {arm}: actual joint angles")
                    fig.tight_layout()
                    name = f"joints_r{record['round']:02d}_s{stream}_{arm}.png"
                    fig.savefig(out / name, dpi=100)
                    figures.append(name)
                plt.close(fig)
    # Preserve all within-solve/fit progress in compact tables, not only the
    # favorable or final solves. Raw native banks remain in their run locations.
    solve_progress, fit_progress = [], []
    for record in history:
        for stream, arms in record.get("streams", {}).items():
            for arm, value in arms.items():
                for item in value.get("raw_candidates", []):
                    for point in item["result"].get("curve", []):
                        solve_progress.append(
                            dict(
                                round=record["round"],
                                stream=stream,
                                arm=arm,
                                solve=item["path"],
                                **point,
                            )
                        )
                fitting = value.get("fit", {})
                if fitting:
                    saved = _read(Path(fitting["path"]) / "fit.json", {})
                    fit_progress.extend(
                        dict(round=record["round"], stream=stream, arm=arm, **point)
                        for point in saved.get("learning_curve", [])
                    )
    write(out / "optimization_progress.json", solve_progress)
    write(out / "fitting_progress.json", fit_progress)
    if solve_progress or fit_progress:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for arm, color in (("independent", "tab:blue"), ("coupled", "tab:orange")):
            solves = sorted({r["solve"] for r in solve_progress if r["arm"] == arm})
            for name in solves:
                points = [r for r in solve_progress if r["solve"] == name]
                initial = points[0]["J"]
                if initial is not None and initial > 0:
                    axes[0].plot(
                        [r["iteration"] for r in points],
                        [r["J"] / initial if r["J"] is not None else np.nan for r in points],
                        color=color,
                        alpha=0.08,
                        linewidth=0.8,
                    )
                    axes[0].plot(
                        [r["iteration"] for r in points],
                        [r["best_J"] / initial for r in points],
                        color=color,
                        alpha=0.15,
                        linewidth=0.8,
                    )
            for stream in config.get("streams", []):
                points = [r for r in fit_progress if r["arm"] == arm and int(r["stream"]) == stream]
                axes[1].plot(
                    [r["update"] for r in points],
                    [r["action_mse"] for r in points],
                    color=color,
                    alpha=0.55,
                    linewidth=0.6,
                    label=f"s{stream} {arm}",
                )
        axes[0].set(
            title="Every acquisition solve: current and best J / initial J",
            xlabel="whole-sequence refinement",
            ylabel="objective ratio",
        )
        axes[1].set(
            title="Actual minibatch fit loss; round corpora may change",
            xlabel="cumulative Adam updates",
            ylabel="coordinate action MSE",
        )
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "optimization_and_fitting.png", dpi=130)
        figures.append("optimization_and_fitting.png")
        plt.close(fig)
    report = dict(
        schema="trajectory-pi-report-1",
        status=result.get("status", "NO_CLOSED_RESULT"),
        output=str(out),
        curves=len(rows),
        figures=figures,
        missing_traces=missing,
        criteria=result.get("criteria"),
        accounting=result.get("accounting"),
        rendering_scientific_work=dict(physics=0, inference=0, gradients=0),
        full_project_success=False,
    )
    write(out / "report.json", report)
    (out / "report.md").write_text(
        f"# Full-trajectory PI P1 saved report\n\nStatus: **{report['status']}**. "
        f"{len(history)} closed round records, {len(rows)} arm/stream measurements.\n\n"
        "Teacher task outcomes, teacher-state reconstruction, student-state queries, "
        "standalone walking and survival are separate measurements. Query gaps only "
        "support closure when their query teachers remain qualified; missing states "
        "remain missing. A frozen parent is not an equal-compute cloning control.\n\n"
        f"Recorded criteria: `{json.dumps(report['criteria'], sort_keys=True)}`.\n\n"
        "Scientific conclusions require the separately saved independent audit. "
        "This rendering performs no physics, inference or fitting. Plain cloning, "
        "DAgger, CIO ablation reproduction and a second body remain untested here.\n"
    )
    return report
