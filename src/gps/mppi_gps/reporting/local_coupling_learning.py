"""Saved-only L2 answers, curves and explicitly unpaired physical illustrations."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from ..provenance import sha, write
from .trajectory_pi import _array_trace


class Saved:
    def __init__(self):
        self.references, self.missing = {}, []
        self.reads = dict(
            json_bytes=0, hashed_input_bytes=0, array_package_bytes_opened=0, decoded_array_bytes=0
        )

    def refs(self, value):
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"} <= value.keys():
                self.references[value["path"]] = {
                    key: value[key] for key in ("path", "bytes", "sha256")
                }
            for child in value.values():
                self.refs(child)
        elif isinstance(value, list):
            for child in value:
                self.refs(child)

    def failure(self, role, path, reason):
        self.missing.append(dict(role=role, path=str(path), reason=reason))

    def json(self, spec, role):
        path = Path(spec["path"]) if isinstance(spec, dict) else Path(spec)
        try:
            payload = path.read_bytes()
            self.reads["json_bytes"] += len(payload)
            digest = hashlib.sha256(payload).hexdigest()
            if isinstance(spec, dict) and (
                len(payload) != spec["bytes"] or digest != spec["sha256"]
            ):
                raise ValueError("Input byte binding differs")
            self.references[str(path)] = dict(path=str(path), bytes=len(payload), sha256=digest)
            value = json.loads(payload)
            self.refs(value)
            return value
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.failure(role, path, repr(error))
            return None

    def trace(self, spec, role):
        path = Path(spec["path"])
        try:
            if path.stat().st_size != spec["bytes"] or sha(path) != spec["sha256"]:
                raise ValueError("Trace byte binding differs")
            self.reads["hashed_input_bytes"] += path.stat().st_size
            self.reads["array_package_bytes_opened"] += path.stat().st_size
            raw = _array_trace(path)
            self.reads["decoded_array_bytes"] += sum(value.nbytes for value in raw.values())
            if raw["states"].shape != (len(raw["actions"]) + 1, 19):
                raise ValueError("Expected actual N+1 states, without padded tails")
            self.refs(spec)
            return raw
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.failure(role, path, repr(error))
            return None


def answers(result: dict, selection: dict | None, audit: dict | None) -> dict:
    certified = audit is not None and audit.get("passed") is True
    selected = selection.get("selected_condition") if selection else None
    teacher = (
        f"Executed selection qualified {selected}, alpha={selection.get('alpha')}, on the eight fixed local restarts. This is 300-control evidence, not full-task walking."
        if selected
        else "No competent, effective teacher contrast qualified."
        if selection
        else "Not established: selection evidence is absent or unreadable."
    )
    blocks = {row["round"]: row for row in result.get("rounds", [])}
    first = blocks.get(1, {}).get("criterion")
    walking = (
        f"Round 1 confirmed the frozen walking-gain screen in forks {first.get('winning_streams', [])}."
        if first and first.get("passed") is True
        else "Round 1 did not confirm the walking-gain screen; retention or fitting gains are separate."
        if first
        else "Not tested or incomplete: no closed round-one walking screen."
    )
    feedback = (
        "Two useful finite rounds were reported, including the frozen teacher-feedback and walking-gain tests. This does not establish convergence."
        if result.get("useful_feedback") is True
        else "Round 2 did not confirm useful feedback."
        if "useful_feedback" in result
        else "Feedback remains untested or incomplete; no closed round-two verdict."
    )
    return dict(
        teacher=teacher,
        walking=walking,
        feedback=feedback,
        audit_passed=certified,
        certification="Saved worker declarations independently audited."
        if certified
        else "Worker declarations only: final independent audit is missing or has not passed.",
    )


def _csv(path, rows, fields):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _figures(out, teachers, evaluations, curves, errors, saved, first_primary):
    if not any((teachers, evaluations, curves, errors)):
        return []
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=120)
        plt.close(fig)
        figures.append(name)

    def grouped(rows, keys):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in keys)].append(row)
        return sorted(groups.items())

    if teachers:
        groups = grouped(teachers, ("round", "stream", "arm"))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        labels = [f"r{key[0]}/s{key[1]}/{key[2]}\nn={len(rows)}/8" for key, rows in groups]
        for axis, key, title in zip(
            axes,
            ("complete_cost", "local_walking", "full_survival"),
            (
                "Mean native complete local cost",
                "Local 300-control walking count",
                "Full local survival count",
            ),
            strict=True,
        ):
            values = [
                float(np.mean([row[key] for row in rows]))
                if key == "complete_cost"
                else sum(row[key] is True for row in rows)
                for _, rows in groups
            ]
            axis.bar(range(len(values)), values)
            axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=7)
            axis.set_title(title)
        finish(fig, "teacher_outcomes.png")
    if evaluations:
        rounds = sorted({row["round"] for row in evaluations})
        fig, axes = plt.subplots(len(rounds), 2, squeeze=False, figsize=(15, 4 * len(rounds)))
        for index, round_number in enumerate(rounds):
            for column, panel in enumerate(("primary", "confirmation")):
                axis = axes[index, column]
                groups = grouped(
                    [
                        row
                        for row in evaluations
                        if row["round"] == round_number and row["panel"] == panel
                    ],
                    ("stream", "arm", "kind"),
                )
                x = np.arange(len(groups))
                axis.bar(
                    x - 0.18,
                    [sum(row["walking"] is True for row in rows) for _, rows in groups],
                    0.36,
                    label="walking",
                )
                axis.bar(
                    x + 0.18,
                    [sum(row["full_survival"] is True for row in rows) for _, rows in groups],
                    0.36,
                    label="full survival",
                )
                labels = [f"s{key[0]}/{key[1]}/{key[2]}\nn={len(rows)}/16" for key, rows in groups]
                axis.set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
                axis.set_ylim(0, 16)
                axis.set_title(f"Round {round_number}: {panel} standalone outcomes")
                axis.legend(fontsize=7)
        finish(fig, "standalone_outcomes.png")
    if curves:
        fig, axis = plt.subplots(figsize=(11, 4))
        for key, rows in grouped(curves, ("round", "stream", "arm")):
            axis.plot(
                [row["fit_update"] for row in rows],
                [row["action_mse"] for row in rows],
                linewidth=0.6,
                alpha=0.7,
                label=f"r{key[0]}/s{key[1]}/{key[2]}",
            )
        axis.set(xlabel="Within-block update", ylabel="Saved minibatch action MSE")
        axis.legend(fontsize=6, ncol=3)
        finish(fig, "fitting_curves.png")
    if errors:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for axis, dataset in zip(axes, ("own", "cross", "held"), strict=True):
            for key, rows in grouped(
                [row for row in errors if row["dataset"] == dataset], ("round", "stream", "arm")
            ):
                rows.sort(key=lambda row: row["stage"] == "after")
                axis.plot(
                    [0 if row["stage"] == "before" else 1 for row in rows],
                    [row["action_mse"] for row in rows],
                    marker="o",
                    label=f"r{key[0]}/s{key[1]}/{key[2]}",
                )
            axis.set(
                title=dataset,
                ylabel="Saved action MSE",
                xticks=[0, 1],
                xticklabels=["incoming", "fitted"],
            )
        axes[-1].legend(fontsize=6)
        finish(fig, "reconstruction.png")

    def angles(rows, title, filename):
        fig, axes = plt.subplots(
            len(rows), 6, squeeze=False, figsize=(15, max(2, 1.55 * len(rows)))
        )
        for index, row in enumerate(rows):
            raw = saved.trace(row["trace"], "joint-angle illustration") if row["trace"] else None
            if raw is None:
                axes[index, 0].text(0, 0.5, row["label"] + ": missing trace", fontsize=7)
                continue
            states = raw["states"]
            elapsed = states[:, 0] - states[0, 0]
            for joint in range(6):
                axes[index, joint].plot(elapsed, states[:, joint + 4], linewidth=0.75)
                if index == 0:
                    axes[index, joint].set_title(
                        f"Joint {joint} (state column {joint + 4})", fontsize=8
                    )
                if joint == 0:
                    axes[index, joint].set_ylabel(row["label"], fontsize=7)
        fig.supxlabel(
            "Actual elapsed simulator time (s); traces stop at their saved terminal boundary"
        )
        fig.suptitle(title, fontsize=10)
        finish(fig, filename)

    for key, rows in grouped(teachers, ("round", "stream", "arm")):
        ordered = sorted(rows, key=lambda row: row["slot"])
        angles(
            [dict(trace=row["trace"], label=f"slot {row['slot']}") for row in ordered],
            f"UNPAIRED illustrations: all saved teacher restarts, r{key[0]}/s{key[1]}/{key[2]} (n={len(rows)}/8)",
            f"teacher_angles_r{key[0]}_s{key[1]}_{key[2]}.png",
        )
    for round_number in sorted({key[0] for key in first_primary}):
        representatives = []
        for identity, seed in sorted(first_primary.items()):
            if identity[0] != round_number:
                continue
            key = identity[1:]
            row = next(
                (
                    value
                    for value in evaluations
                    if (value["round"], value["stream"], value["arm"], value["kind"]) == identity
                    and value["panel"] == "primary"
                    and value["reset_seed"] == seed
                ),
                None,
            )
            if row is None:
                saved.failure(
                    "first primary reset illustration",
                    str(identity),
                    f"Reset {seed} unavailable; no favorable substitute",
                )
            representatives.append(
                dict(trace=row["trace"] if row else None, label=f"s{key[0]}/{key[1]}/{key[2]}")
            )
        if representatives:
            angles(
                representatives,
                f"UNPAIRED illustrations: first primary reset of every actor, round {round_number}; includes falls",
                f"policy_angles_r{round_number}.png",
            )
    return figures


def render(base: Path, out: Path) -> dict:
    """Create a report from saved declarations; never restore a model or simulator."""
    started, cpu = time.monotonic(), time.process_time()
    base, out = Path(base).resolve(), Path(out).resolve()
    worker = base / "worker" if (base / "worker").is_dir() else base
    evidence = worker.parent if worker.name == "worker" else base
    out.mkdir(parents=True, exist_ok=False)
    saved = Saved()
    result = saved.json(worker / "result.json", "worker result") or {}
    selection = saved.json(result["selection"], "selection") if result.get("selection") else None
    audit_path = evidence / "audit/final/receipt.json"
    audit = saved.json(audit_path, "final independent audit")
    teachers, evaluations, curves, errors = [], [], [], []
    first_primary = {}
    for row in result.get("acquisitions", []):
        value = saved.json(row["manifest"], "teacher manifest")
        if value is None or "outcome" not in value or "trace" not in value:
            if value is not None:
                saved.failure(
                    "teacher outcome/trace", row["manifest"]["path"], "Incomplete teacher manifest"
                )
            continue
        teachers.append(
            {key: row[key] for key in ("round", "stream", "arm", "slot")}
            | dict(alpha=value.get("alpha"), trace=value["trace"])
            | value["outcome"]
        )
    for block in result.get("rounds", []):
        for row in block.get("evaluations", []):
            if row["panel"] == "primary":
                identity = (block["round"], row["stream"], row["arm"], row["kind"])
                first_primary[identity] = min(
                    first_primary.get(identity, row["reset_seed"]), row["reset_seed"]
                )
            value = saved.json(row["result"], "standalone outcome")
            if value is None or "outcome" not in value or "trace" not in value:
                if value is not None:
                    saved.failure(
                        "standalone outcome/trace",
                        row["result"]["path"],
                        "Incomplete standalone result",
                    )
                continue
            evaluations.append(
                dict(
                    round=block["round"],
                    **{key: row[key] for key in ("stream", "arm", "kind", "panel", "reset_seed")},
                    trace=value["trace"],
                )
                | value["outcome"]
            )
        for row in block.get("fits", []):
            value = saved.json(row["result"], "fit result")
            if value is None:
                continue
            identity = dict(round=block["round"], stream=row["stream"], arm=row["arm"])
            if value.get("fit"):
                fitted = saved.json(value["fit"], "fitting curve") or {}
                curves.extend(identity | point for point in fitted.get("learning_curve", []))
            for stage in ("before", "after"):
                for name, metric in value.get("diagnostics_" + stage, {}).items():
                    errors.append(
                        identity
                        | dict(
                            stage=stage,
                            dataset=name,
                            rows=metric["rows"],
                            action_mse=metric["action_mse"],
                            rms_l2=metric["rms_l2"],
                        )
                    )
    _csv(
        out / "teacher_outcomes.csv",
        teachers,
        [
            "round",
            "stream",
            "arm",
            "slot",
            "alpha",
            "controls",
            "local_walking",
            "full_survival",
            "complete_cost",
            "velocity_mae",
        ],
    )
    _csv(
        out / "standalone_outcomes.csv",
        evaluations,
        [
            "round",
            "stream",
            "arm",
            "kind",
            "panel",
            "reset_seed",
            "controls",
            "walking",
            "full_survival",
            "complete_cost",
            "velocity_mae",
        ],
    )
    _csv(
        out / "fitting_curves.csv",
        curves,
        [
            "round",
            "stream",
            "arm",
            "update",
            "fit_update",
            "action_mse",
            "total_loss",
            "unclipped_gradient_norm",
        ],
    )
    _csv(
        out / "reconstruction.csv",
        errors,
        ["round", "stream", "arm", "stage", "dataset", "rows", "action_mse", "rms_l2"],
    )
    figures = _figures(out, teachers, evaluations, curves, errors, saved, first_primary)
    verdict = answers(result, selection, audit)
    lines = [
        "# L2.0 — saved evidence",
        "",
        f"Worker status: **{result.get('status', 'MISSING_RESULT')}**. {verdict['certification']}",
        "",
        f"**Competent, different teacher experience?** {verdict['teacher']}",
        "",
        f"**Did standalone walking improve?** {verdict['walking']}",
        "",
        f"**Did learning feedback help?** {verdict['feedback']}",
        "",
        "Eight fixed restarts share one source student episode. Three minibatch forks share one incoming actor. Local teacher survival, fitting and retention are distinct from full standalone walking; no asymptotic-convergence, equal-compute cloning/DAgger or second-body claim follows.",
        "",
        "[Teacher outcomes](teacher_outcomes.csv) · [Standalone outcomes](standalone_outcomes.csv) · [Fitting curves](fitting_curves.csv) · [Own/cross/held error](reconstruction.csv)",
        "",
        "Figures: "
        + (", ".join(f"[{name}]({name})" for name in figures) or "none: measurements absent"),
        "",
        "Joint-angle figures are UNPAIRED illustrations, not paired task-cost gaps: all available teacher slots and the first primary reset per actor, including failures. No favorable episode selection, reference-path return requirement or fabricated post-fall tail.",
        "",
        f"Missing/unreadable evidence entries: **{len(saved.missing)}**; see [report manifest](report_manifest.json). Raw inputs remain external and byte-bound there. Report rendering performs zero neural forwards, fitting updates or physics calls.",
        "",
        "Saved complete accounting is retained in the report manifest; phase times are nested and must not be summed twice.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    artifacts = [
        dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path))
        for path in sorted(out.iterdir())
        if path.is_file()
    ]
    manifest = dict(
        schema="local-coupling-report-1",
        status="RENDERED_WITH_MISSING_EVIDENCE" if saved.missing else "RENDERED",
        worker=str(worker),
        worker_status=result.get("status"),
        answers=verdict,
        saved_accounting=result.get("accounting"),
        missing_evidence=saved.missing,
        external_inputs=list(saved.references.values()),
        artifacts=artifacts,
        figures=figures,
        policy_illustration_selection=[
            dict(
                round=key[0],
                stream=key[1],
                arm=key[2],
                kind=key[3],
                reset_seed=value,
                rule="First recorded primary reset, no replacement if missing",
            )
            for key, value in sorted(first_primary.items())
        ],
        measurements=dict(
            teacher_episodes=len(teachers),
            standalone_episodes=len(evaluations),
            fitting_updates=len(curves),
            reconstruction_records=len(errors),
        ),
        rendering_work=dict(
            neural_rows=0,
            model_controls=0,
            native_controls=0,
            backward_rows=0,
            optimizer_steps=0,
            logical_io=saved.reads,
            physical_disk_io_bytes=None,
            wall_seconds_before_manifest=time.monotonic() - started,
            cpu_seconds_before_manifest=time.process_time() - cpu,
        ),
        interpretation="Saved declarations and illustrations only; independent audit status is separate.",
    )
    write(out / "report_manifest.json", manifest)
    return manifest
