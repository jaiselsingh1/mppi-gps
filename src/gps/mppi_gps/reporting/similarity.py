"""Saved-JSON J1 reporting: no checkpoint decoding, model inference or physics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

MAX_READ = 512 * 1024**2
MAX_OUTPUT = 256 * 1024**2
MAX_RSS = 2 * 1024**3
MAX_WALL = 1800
CANDIDATES = ("A", "B", "C", "D", "E")
ZERO_WORK = dict(
    neural_rows=0,
    prediction_calls=0,
    native_controls=0,
    model_controls=0,
    physics_substeps=0,
    checkpoint_tensor_loads=0,
    backward_rows=0,
    optimizer_steps=0,
    policy_constructions=0,
)


def _identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject(value):
    raise ValueError("Nonfinite saved J1 JSON: " + value)


class SavedJSON:
    def __init__(self):
        self.started = time.monotonic()
        self.inputs: dict[str, dict] = {}
        self.read_bytes = 0
        self.peak_rss = 0

    def check(self):
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak_rss = max(self.peak_rss, int(usage if sys.platform == "darwin" else usage * 1024))
        if time.monotonic() - self.started > MAX_WALL or self.peak_rss > MAX_RSS:
            raise ValueError("J1 saved report wall/RSS reservation exhausted")

    def load(self, path: Path):
        self.check()
        if path.suffix != ".json" or path.is_symlink():
            raise ValueError("J1 report reads ordinary saved JSON only")
        before = path.stat()
        if before.st_size > 128 * 1024**2 or self.read_bytes + before.st_size > MAX_READ:
            raise ValueError("J1 saved report read reservation exhausted")
        with path.open("rb") as stream:
            payload = stream.read(before.st_size)
        self.read_bytes += len(payload)
        if len(payload) != before.st_size or _identity(path.stat()) != _identity(before):
            raise ValueError("J1 saved report input changed")
        self.inputs[str(path)] = dict(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
        return json.loads(payload, parse_constant=_reject)


def _contained(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("J1 saved metadata path escapes its run")
    path = root / relative
    if any(
        (root / Path(*relative.parts[:index])).is_symlink()
        for index in range(1, len(relative.parts) + 1)
    ):
        raise ValueError("J1 report refuses source symlinks")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("J1 saved metadata path escapes its run")
    return path


class Outputs:
    def __init__(self, root: Path):
        self.root, self.files = root, {}

    def save(self, name: str, payload: bytes):
        if sum(row["bytes"] for row in self.files.values()) + len(payload) > MAX_OUTPUT - 1024**2:
            raise ValueError("J1 saved report output reservation exhausted")
        path = self.root / name
        with path.open("xb") as stream:
            stream.write(payload)
        self.files[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

    def json(self, name: str, value: Any):
        self.save(name, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())

    def csv(self, name: str, rows: list[dict], fields: list[str]):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fields
                }
            )
        self.save(name, stream.getvalue().encode())


def _table(rows: list[dict], fields: list[str]) -> str:
    def text(value):
        if value is None:
            return "missing"
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(fields) + " |",
            "| " + " | ".join("---" for _ in fields) + " |",
            *["| " + " | ".join(text(row.get(key)) for key in fields) + " |" for row in rows],
        ]
    )


def _case_union(documents: dict) -> list[dict]:
    cases = {}
    for split in ("development", "protected"):
        for row in documents.get("worker/score_seals/" + split + ".json", {}).get("cases", []):
            if row["case_id"] in cases:
                raise ValueError("Duplicate J1 pre-update case ID")
            cases[row["case_id"]] = dict(row)
    completed_ids = set()
    for row in documents.get("worker/cases.json", []):
        key = row["case_id"]
        if key in completed_ids:
            raise ValueError("Duplicate J1 completed case ID")
        completed_ids.add(key)
        before = cases.get(key, {})
        for name in ("scores", "incoming_bindings", "group", "learner", "source", "split"):
            if name in before and name in row and before[name] != row[name]:
                raise ValueError("J1 pre-update/completed case join differs: " + name)
        cases[key] = before | row
    if len(cases) > 64:
        raise ValueError("J1 case roster exceeds64")
    return [cases[key] for key in sorted(cases)]


def _case_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        evaluations = case.get("evaluations")
        if evaluations is not None:
            if len(evaluations) > 16:
                raise ValueError("J1 standalone panel exceeds16")
            for key in ("walking", "survived"):
                if any(type(row.get(key)) is not bool for row in evaluations):
                    raise ValueError("J1 measured standalone outcome flags are missing")
        measured = evaluations is not None and bool(evaluations)
        walking = sum(row["walking"] for row in evaluations) if measured else None
        if "walking_successes" in case and walking != case["walking_successes"]:
            raise ValueError("J1 case/panel walking join differs")
        reproduction = case.get("reproduction", {})
        row = {
            key: case.get(key)
            for key in (
                "case_id",
                "group",
                "learner",
                "split",
                "source",
                "eligible",
                "ineligible_reason",
            )
        }

        def mean_cost(key):
            if not measured or any(value.get(key) is None for value in evaluations):
                return None
            return sum(value[key] for value in evaluations) / len(evaluations)

        row.update(
            teacher_controls=case.get("teacher", {}).get("controls"),
            incoming_actor_hash=case.get("incoming_bindings", {}).get("actor_hash"),
            fit_completed="child_checkpoint" in case,
            standalone_attempts=len(evaluations) if evaluations is not None else None,
            standalone_walking=walking,
            standalone_full_survival=sum(v["survived"] for v in evaluations) if measured else None,
            standalone_controls=sum(v["controls"] for v in evaluations) if measured else None,
            standalone_mean_cost=mean_cost("cost"),
            standalone_mean_native_cost=mean_cost("native_cost"),
            standalone_mean_absorbing_cost=mean_cost("absorbing_cost"),
            post_motion=case.get("post_motion"),
            reproduced=case.get("reproduced"),
            reproduction_walking=reproduction.get("walking"),
            reproduction_full_survival=reproduction.get("survived"),
            reproduction_controls=reproduction.get("controls"),
            reproduction_cost=reproduction.get("cost"),
        )
        rows.append(row)
    return rows


def _summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("split"), row.get("source"))].append(row)
    result = []
    for (split, source), cases in groups.items():
        measured = [row for row in cases if row["standalone_attempts"]]
        reproduced = [row for row in cases if type(row["reproduced"]) is bool]
        result.append(
            dict(
                split=split,
                source=source,
                actual_groups=len({r["group"] for r in cases}),
                cases=len(cases),
                eligible_cases=sum(row.get("eligible") is True for row in cases),
                completed_fits=sum(row["fit_completed"] for row in cases),
                standalone_measured_cases=len(measured),
                standalone_episodes=sum(r["standalone_attempts"] for r in measured),
                standalone_walking=sum(r["standalone_walking"] for r in measured)
                if measured
                else None,
                standalone_full_survival=sum(r["standalone_full_survival"] for r in measured)
                if measured
                else None,
                standalone_controls=sum(r["standalone_controls"] for r in measured)
                if measured
                else None,
                reproduction_measured_cases=len(reproduced),
                reproduced=sum(r["reproduced"] for r in reproduced) if reproduced else None,
                reproduction_walking=sum(r["reproduction_walking"] is True for r in reproduced)
                if reproduced and all(type(r["reproduction_walking"]) is bool for r in reproduced)
                else None,
                reproduction_full_survival=sum(r["reproduction_full_survival"] for r in reproduced)
                if reproduced
                and all(type(r["reproduction_full_survival"]) is bool for r in reproduced)
                else None,
            )
        )
    return result


def _curves(root: Path, cases: list[dict], reader: SavedJSON) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for case in cases:
        name = case.get("fit_receipt_file")
        conventional = "worker/fits/" + case["case_id"] + "/fit_receipt.json"
        if name is None:
            if "child_checkpoint" in case:
                missing.append(case["case_id"] + ":fit_receipt_file")
            name = conventional
        receipt_path = _contained(root, name)
        receipt = reader.load(receipt_path) if receipt_path.is_file() else None
        failure_path = receipt_path.with_name("fit_failure.json")
        if receipt is None and failure_path.is_file():
            failure = reader.load(failure_path)
            evidence = failure.get("fit_evidence") or {}
            fit_status = "partial_failed"
        elif receipt is None:
            if case.get("fit_receipt_file") is not None:
                missing.append(name)
            continue
        else:
            curve_name = str(Path(name).with_name("learning_curve.json"))
            path = _contained(root, curve_name)
            if not path.is_file():
                missing.append(curve_name)
                continue
            evidence = reader.load(path)
            fit_status = "completed_fit"
        if not evidence:
            missing.append(str(failure_path.relative_to(root)) + ":fit_evidence")
            continue
        curve = evidence.get("learning_curve")
        if curve is None:
            missing.append(name + ":learning_curve")
            continue
        if len(curve) > 2000 or (
            receipt is not None and receipt.get("updates") == 2000 and len(curve) != 2000
        ):
            raise ValueError("J1 fixed-budget learning curve length differs")
        for ordinal, point in enumerate(curve, start=1):
            if point.get("fit_update") != ordinal:
                raise ValueError("J1 learning curve update order differs")
            rows.append(
                {key: case.get(key) for key in ("case_id", "group", "learner", "split", "source")}
                | point
                | dict(fit_status=fit_status)
            )
    return rows, missing


def _pairs(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        for candidate in CANDIDATES:
            prediction = case.get("predictions", {}).get(candidate, {})
            actual = case.get("post_motion")
            trials, successes = case.get("walking_trials"), case.get("walking_successes")
            probability = prediction.get("walking_probability")
            motion = prediction.get("motion")
            rate = successes / trials if successes is not None and trials else None
            rows.append(
                {
                    key: case.get(key)
                    for key in ("case_id", "group", "learner", "split", "source", "eligible")
                }
                | dict(
                    candidate=candidate,
                    preupdate_score=case.get("scores", {}).get(candidate),
                    post_motion=actual,
                    standalone_walking_rate=rate,
                    reproduced=case.get("reproduced"),
                    predicted_motion=motion,
                    predicted_walking_probability=probability,
                    motion_absolute_error=None
                    if actual is None or motion is None
                    else abs(actual - motion),
                    walking_rate_absolute_error=None
                    if rate is None or probability is None
                    else abs(rate - probability),
                )
            )
    return rows


def _score_rows(
    definitions: dict, calibration: dict, comparison: dict, pairs: list[dict]
) -> list[dict]:
    measured = {row["candidate"]: row for row in comparison.get("score_table", [])}
    rows = []
    for candidate in CANDIDATES:
        actual = measured.get(candidate, {})
        failures = sorted(
            [
                r
                for r in pairs
                if r["candidate"] == candidate
                and r["split"] == "protected"
                and r["source"] == "mppi"
                and r["motion_absolute_error"] is not None
            ],
            key=lambda row: (-row["motion_absolute_error"], row["case_id"]),
        )
        rows.append(
            dict(
                candidate=candidate,
                definition=definitions.get("candidates", {}).get(candidate),
                status="baseline"
                if candidate == "A"
                else "untested_or_unidentified"
                if not actual
                else "passed_frozen_joint_gate"
                if actual.get("passed") is True
                else "did_not_pass_frozen_joint_gate",
                calibration_passed=calibration.get("passed"),
                calibration_checks=calibration.get("checks"),
                motion_identified=comparison.get("motion_identified"),
                walking_identified=comparison.get("walking_identified"),
                **{
                    key: actual.get(key)
                    for key in (
                        "motion_rmse",
                        "walking_brier",
                        "reproduction_auc",
                        "motion_relative_improvement",
                        "walking_absolute_improvement",
                        "walking_relative_improvement",
                        "motion_simultaneous_lower",
                        "walking_simultaneous_lower",
                        "passed",
                    )
                },
                largest_motion_failure_cases=[row["case_id"] for row in failures[:3]],
                score_cost="One shared teacher-state prediction array/incoming execution; candidate diagnostic arithmetic only; see saved worker accounting",
                limitations={
                    "A": "Conditional torque agreement is not competence or learnability",
                    "B": "Smoothing can remove consequential timing differences",
                    "C": calibration.get(
                        "C_identity_limitation",
                        "Affine model misspecification/support shift can penalize a nonlinear exact copy",
                    ),
                    "D": "Incoming execution similarity may not predict a later fixed learning update",
                    "E": "Joint occupancy intentionally discards temporal ordering",
                }[candidate],
            )
        )
    for name, reason in definitions.get("rejected_candidates", {}).items():
        rows.append(
            dict(
                candidate=name,
                definition=reason,
                status="rejected_calibration_or_algebraic_alias",
                calibration_passed=calibration.get("passed"),
                limitations=reason,
            )
        )
    return rows


def _figures(
    outputs: Outputs, pairs: list[dict], curves: list[dict], reader: SavedJSON
) -> list[str]:
    if not pairs and not curves:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    for outcome, name, ylabel in (
        ("post_motion", "scores_vs_motion.png", "Post-update motion error"),
        ("standalone_walking_rate", "scores_vs_walking.png", "Standalone walking fraction"),
    ):
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
        for axis, candidate in zip(axes, CANDIDATES, strict=True):
            for split, marker in (("development", "o"), ("protected", "s")):
                for source, color in (("mppi", "#467599"), ("reference", "#bc6d37")):
                    selected = [
                        r
                        for r in pairs
                        if r["candidate"] == candidate
                        and r.get("split") == split
                        and r.get("source") == source
                        and r.get(outcome) is not None
                        and r["preupdate_score"] is not None
                    ]
                    if selected:
                        axis.scatter(
                            [r["preupdate_score"] for r in selected],
                            [r[outcome] for r in selected],
                            s=22,
                            marker=marker,
                            color=color,
                            alpha=0.65,
                            label=split + "/" + source,
                        )
            axis.set_xlabel(candidate + " pre-update score")
            axis.grid(alpha=0.15)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)
        outputs.save(name, buffer.getvalue())
        names.append(name)
        reader.check()
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        grouped = defaultdict(list)
        for row in curves:
            grouped[row["case_id"]].append(row)
        for points in grouped.values():
            axis = axes[0 if points[0].get("source") == "mppi" else 1]
            axis.plot(
                [p["fit_update"] for p in points],
                [p["action_mse"] for p in points],
                alpha=0.35,
                linewidth=0.6,
            )
        for axis, source in zip(axes, ("MPPI", "Reference"), strict=True):
            axis.set_title(source + ": every recorded update, every fitted case")
            axis.set_xlabel("Fixed-fit update")
            axis.set_ylabel("Training minibatch action MSE")
            axis.grid(alpha=0.15)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)
        outputs.save("learning_curves.png", buffer.getvalue())
        names.append("learning_curves.png")
    return names


def report_run(
    run: str | Path,
    out: str | Path,
    audit: str | Path | None = None,
    atlas: str | Path | None = None,
) -> dict:
    run, out = Path(run).resolve(), Path(out).resolve()
    root = run / "run" if (run / "capsule.json").is_file() else run
    if out.exists() or out.is_relative_to(root):
        raise ValueError("J1 report output must be fresh and outside preserved evidence")
    reader = SavedJSON()
    documents, missing = {}, []
    names = (
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
    for name in names:
        path = _contained(root, name)
        if path.is_file():
            documents[name] = reader.load(path)
        else:
            missing.append(name)
    audit_path = Path(audit).resolve() if audit is not None else run / "audit/audit.json"
    if audit_path.is_dir():
        audit_path = audit_path / "audit.json"
    audit_value = reader.load(audit_path) if audit_path.is_file() else None
    audit_association = None
    if audit_value is not None and audit_value.get("passed") is True:
        for key, name in (
            ("config_sha256", "config.json"),
            ("source_manifest_sha256", "manifest.json"),
            ("input_manifest_sha256", "input_manifest.json"),
            ("worker_result_sha256", "worker_result.json"),
        ):
            binding = reader.inputs.get(str(root / name))
            if binding is None or audit_value.get(key) != binding["sha256"]:
                raise ValueError("J1 report/audit association differs: " + key)
        audit_association = True
    full_audit_passed = (
        None if audit_value is None else audit_value.get("full_scientific_audit_passed")
    )
    cases = _case_union(documents)
    rows = _case_rows(cases)
    pairs = _pairs(cases)
    curves, missing_curves = _curves(root, cases, reader)
    missing.extend(missing_curves)
    comparison = documents.get("worker/predictive_comparison.json", {})
    definitions = documents.get("worker/score_definitions.json", {})
    calibration = documents.get("worker/calibration.json", {})
    scores = _score_rows(definitions, calibration, comparison, pairs)
    teachers = documents.get("worker/teachers.json", {})
    config = documents.get("config.json", {})
    declared_groups = {group["id"]: group for group in config.get("groups", [])}
    teacher_context = {}
    for group, declaration in declared_groups.items():
        for source in ("mppi", "reference"):
            teacher_context[group + "_" + source] = dict(
                group=group, source=source, split=declaration.get("split")
            )
    teacher_rows = [
        dict(
            teacher=tag,
            **teacher_context.get(tag, {}),
            **{
                key: row.get(key)
                for key in (
                    "controls",
                    "walking",
                    "survived",
                    "mae",
                    "raw_cost",
                    "native_terminal",
                    "native_cost",
                    "absorbing_cost",
                    "cost",
                )
            },
        )
        for tag, row in sorted(teachers.items())
    ]
    summaries = _summaries(rows)
    atlas_value, atlas_link = None, None
    if atlas is not None:
        atlas_path = Path(atlas).resolve()
        receipt_path = atlas_path / "atlas_receipt.json" if atlas_path.is_dir() else atlas_path
        if receipt_path.is_file():
            atlas_value = reader.load(receipt_path)
            atlas_link = os.path.relpath(receipt_path.parent / "index.html", out)
        else:
            missing.append(str(receipt_path))
    out.mkdir(parents=True, exist_ok=False)
    outputs = Outputs(out)
    case_fields = (
        list(rows[0])
        if rows
        else [
            "case_id",
            "split",
            "source",
            "eligible",
            "standalone_walking",
            "reproduced",
            "post_motion",
        ]
    )
    outputs.csv("cases.csv", rows, case_fields)
    outputs.csv(
        "outcome_summary.csv",
        summaries,
        list(summaries[0])
        if summaries
        else ["split", "source", "cases", "standalone_walking", "reproduced"],
    )
    outputs.csv(
        "teachers.csv",
        teacher_rows,
        list(teacher_rows[0])
        if teacher_rows
        else ["teacher", "controls", "walking", "survived", "cost"],
    )
    score_fields = [
        "candidate",
        "definition",
        "status",
        "calibration_passed",
        "calibration_checks",
        "motion_identified",
        "walking_identified",
        "motion_rmse",
        "walking_brier",
        "reproduction_auc",
        "motion_relative_improvement",
        "walking_absolute_improvement",
        "walking_relative_improvement",
        "motion_simultaneous_lower",
        "walking_simultaneous_lower",
        "passed",
        "largest_motion_failure_cases",
        "score_cost",
        "limitations",
    ]
    outputs.csv("score_table.csv", scores, score_fields)
    outputs.csv(
        "score_outcome_pairs.csv",
        pairs,
        list(pairs[0])
        if pairs
        else ["case_id", "candidate", "preupdate_score", "post_motion", "standalone_walking_rate"],
    )
    outputs.csv(
        "learning_curves.csv",
        curves,
        [
            "case_id",
            "group",
            "learner",
            "split",
            "source",
            "fit_status",
            "update",
            "fit_update",
            "action_mse",
            "proximal_loss",
            "total_loss",
            "unclipped_gradient_norm",
        ],
    )
    figures = _figures(outputs, pairs, curves, reader)
    worker = documents.get("worker_result.json", {})
    accounting = dict(
        worker=worker.get("costs"),
        worker_process=documents.get("worker_process_receipt.json"),
        audit=None if audit_value is None else audit_value.get("work"),
        rendering=atlas_value,
    )
    saved_context = dict(
        score_definitions=definitions,
        calibrations=calibration,
        frozen_prediction_gates=config.get("prediction"),
        measured_prediction_comparison=comparison,
        within_mppi_probe=documents.get("worker/probe_summary.json"),
        accounting=accounting,
    )
    outputs.json("saved_measurements.json", saved_context)
    objective = worker.get("objective_passed")
    verdict = "The objective remains unmeasured or unclosed; missing outcomes are not zeros."
    if objective is False:
        verdict = "The joint predictive objective remains unmet in J1.0. Unidentified walking discrimination is not evidence that a distance predicts walking success. A motion-only result is partial."
    elif objective is True:
        verdict = (
            "The worker reports the joint frozen predictive gate passed on this finite benchmark."
        )
        verdict += (
            " Its independent audit association passed."
            if audit_association and full_audit_passed is True
            else " Independent certification is missing; no confirmed claim follows."
        )
    text = [
        "# J1.0: similarity before a fixed learning update",
        verdict,
        "This is a saved-only report. Walking, full survival, complete cost and same-start motion reproduction are separate outcomes. Competent walking with a different gait remains valid walking. A copied walking checkpoint is an execution control, not evidence of fresh-student learning.",
        "The five candidates share measured learning outcomes and prediction-model capacity. Scores were intended to be sealed before learning. Only the independent audit can certify ordering, source joins and the full experiment; this reporter does not repeat those checks or fit a model.",
        f"Recorded status: {worker.get('status', 'missing')}. Independent audit: {None if audit_value is None else audit_value.get('passed')}; full scientific audit: {full_audit_passed}; association verified: {audit_association}.",
        "## Actual learning outcomes",
        _table(
            summaries,
            [
                "split",
                "source",
                "actual_groups",
                "cases",
                "completed_fits",
                "standalone_episodes",
                "standalone_walking",
                "standalone_full_survival",
                "reproduction_measured_cases",
                "reproduced",
            ],
        ),
        "Missing/unattempted outcomes remain missing. Full-window walking/reproduction gates cannot be replaced by prefix fit or survival. The report contains every recorded case and update, including ineligible sources and failed controls.",
        "Teacher source outcomes are recorded independently in teachers.csv, including complete cost, walking and full survival. They are not child outcomes and are not multiplied into independent groups by learner replication.",
        "## Five-score and rejected-candidate record",
        _table(scores, ["candidate", "definition", "limitations"]),
        _table(
            scores,
            [
                "candidate",
                "status",
                "motion_identified",
                "walking_identified",
                "motion_rmse",
                "walking_brier",
                "motion_simultaneous_lower",
                "walking_simultaneous_lower",
                "passed",
            ],
        ),
        "Definitions, calibrations, limitations, largest protected prediction errors and costs are preserved together in score_table.csv. score_outcome_pairs.csv keeps all available pre-score/post-outcome joins. Unidentified numeric errors are descriptive, not a predictive certificate. Bootstrap uncertainty is the saved paired-group procedure, not a new fit or resampling here.",
        "Frozen predictive thresholds, observed eligibility/variation gates, bootstrap bounds, calibration results and the within-MPPI probe summary are preserved verbatim in [saved_measurements.json](saved_measurements.json). No score is selected by an expected oracle ranking, and no new calibration or bootstrap is run here.",
        "Frozen prediction gates: "
        + (
            "missing"
            if config.get("prediction") is None
            else "`" + json.dumps(config["prediction"], sort_keys=True) + "`"
        ),
        "## Visual evidence",
        *[f"![{name}]({name})" for name in figures],
        "Learning curves show all stochastic training minibatch measurements; they are not full-corpus reconstruction or standalone competence measurements.",
        "Atlas: "
        + (
            "missing/unattempted"
            if atlas_link is None
            else f"[Synchronized actual motion/contact/control/video atlas]({atlas_link})"
        ),
        "## Costs and limits",
        "The receipt preserves worker, audit and rendering accounting separately. Planning simulations are included in the worker census. Nested wall/CPU measures are not additive; rendering/engineering do not become free planning. Missing cost fields remain missing.",
        "Scientific worker work census: "
        + (
            "missing"
            if worker.get("costs") is None
            else "`" + json.dumps(worker["costs"], sort_keys=True) + "`"
        ),
        "No in-MPPI similarity guidance comparison or follow-up is launched by this report. Only a separately reviewed, frozen protocol may test whether a passing score improves a next standalone student while protecting competent independent discovery.",
    ]
    outputs.save("report.md", ("\n\n".join(text) + "\n").encode())
    receipt = dict(
        schema="walker-similarity-j1-saved-report-v1",
        saved_only=True,
        status=worker.get("status"),
        objective_passed=objective,
        audit_passed=None if audit_value is None else audit_value.get("passed"),
        full_scientific_audit_passed=full_audit_passed,
        audit_association_verified=audit_association,
        no_followup_launched=True,
        actual_cases=len(cases),
        actual_groups=len({row.get("group") for row in cases}),
        completed_fits=sum(row["fit_completed"] for row in rows),
        learning_points=len(curves),
        learning_cases_with_curves=len({row["case_id"] for row in curves}),
        score_rows=len(scores),
        score_outcome_pairs=len(pairs),
        teacher_trajectories=len(teacher_rows),
        protected_groups=len({row["group"] for row in cases if row.get("split") == "protected"}),
        missing_evidence=missing,
        new_scientific_work=dict(ZERO_WORK),
        read_bytes=reader.read_bytes,
        input_files=reader.inputs,
        accounting=accounting,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figures=figures,
        outputs=dict(outputs.files),
        wall_seconds=time.monotonic() - reader.started,
        peak_rss_bytes=reader.peak_rss,
        limits=dict(
            read_bytes=MAX_READ,
            output_bytes=MAX_OUTPUT,
            wall_seconds=MAX_WALL,
            peak_rss_bytes=MAX_RSS,
        ),
    )
    outputs.json("report_receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--atlas", type=Path)
    args = parser.parse_args(argv)
    report_run(args.run, args.out, args.audit, args.atlas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
