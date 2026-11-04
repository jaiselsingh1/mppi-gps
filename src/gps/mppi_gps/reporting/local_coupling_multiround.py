"""Saved-only L3 curves, teacher activity and finite-loop outcome reporting.

The renderer reads declared bound measurements. It never imports a simulator,
constructs an actor, restores RNGs, fits, evaluates or queries a planner.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ..provenance import sha, write
from .local_coupling_learning import Saved, _csv


def _load(saved: Saved, value: Any, role: str) -> dict | None:
    if isinstance(value, dict) and "path" not in value:
        saved.refs(value)
        return value
    if isinstance(value, (str, Path)) or isinstance(value, dict) and "path" in value:
        return saved.json(value, role)
    saved.failure(role, repr(value), "Missing declared JSON binding")
    return None


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def endpoint_summary(evaluations: list[dict]) -> list[dict]:
    """Summarize actual observed episodes without treating missing starts as zero."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in evaluations:
        groups[(row["round"], row["stream"], row["arm"], row["kind"], row["panel"])].append(row)
    return [
        dict(
            round=key[0],
            stream=key[1],
            arm=key[2],
            kind=key[3],
            panel=key[4],
            episodes=len(rows),
            walking=sum(row.get("walking") is True for row in rows),
            full_survival=sum(row.get("full_survival") is True for row in rows),
            mean_controls=_mean([row.get("controls") for row in rows]),
            mean_complete_cost=_mean([row.get("complete_cost") for row in rows]),
            mean_velocity_mae=_mean([row.get("velocity_mae") for row in rows]),
        )
        for key, rows in sorted(groups.items())
    ]


def activity_summary(admissions: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in admissions:
        groups[(row["round"], row["stream"], row["arm"])].append(row)
    return [
        dict(
            round=key[0],
            stream=key[1],
            arm=key[2],
            slots=len(rows),
            newly_admitted=sum(row.get("admitted") is True for row in rows),
            new_experience_slots=sum(row.get("new_experience") is True for row in rows),
            fallback_slots=sum(row.get("fallback") is True for row in rows),
            new_rows=sum(row.get("new_rows", 0) for row in rows),
            max_age=max((row.get("age", 0) for row in rows), default=None),
            mean_age=_mean([row.get("age") for row in rows]),
        )
        for key, rows in sorted(groups.items())
    ]


def student_gap_endpoints(probes: list[dict], final_round: int) -> list[dict]:
    """Compare actually shared endpoint anchors and expose lost coverage."""
    groups: dict[tuple, dict[int, dict[int, float]]] = defaultdict(dict)
    for row in probes:
        if row["round"] not in (0, final_round):
            continue
        key = (row["stream"], row["arm"])
        observed = groups[key].setdefault(row["round"], {})
        if row.get("status") == "OBSERVED" and row.get("gap_action_mse") is not None:
            observed[row["control"]] = float(row["gap_action_mse"])
    result = []
    for key, endpoints in sorted(groups.items()):
        initial, final = endpoints.get(0, {}), endpoints.get(final_round, {})
        controls = sorted(set(initial) & set(final))
        first = _mean([initial[control] for control in controls])
        last = _mean([final[control] for control in controls])
        result.append(
            dict(
                stream=key[0],
                arm=key[1],
                initial_observed=len(initial),
                final_observed=len(final),
                common_anchors=len(controls),
                common_controls=controls,
                initial_common_action_mse=first,
                final_common_action_mse=last,
                ratio=last / first
                if first is not None and first > 0 and last is not None
                else None,
                no_lost_anchor_coverage=set(initial) <= set(final),
            )
        )
    return result


def answers(result: dict, streams: list[dict], audit: dict | None) -> dict:
    planned = result.get("planned_rounds", 8)
    complete = (
        result.get("status") == "COMPLETED_ALL_ROUNDS"
        and len(streams) == 3
        and all(
            row.get("status") == "COMPLETED_ALL_ROUNDS"
            and sorted(block["round"] for block in row.get("rounds", []))
            == list(range(1, planned + 1))
            and len(row.get("initial_evaluations", [])) == 32
            and len(row.get("initial_probes", [])) == 2
            and len(row.get("final_evaluations", [])) == 64
            and all(
                block.get("status") == "COMPLETED"
                and len(block.get("acquisitions", [])) == 16
                and len(block.get("admissions", [])) == 16
                and len(block.get("fits", [])) == 2
                and len(block.get("evaluations", [])) == 32
                and len(block.get("probes", [])) == 2
                for block in row.get("rounds", [])
            )
            for row in streams
        )
    )
    certified = bool(audit and audit.get("passed") is True)
    return dict(
        planned_rounds=planned,
        completed_stream_rounds=sum(len(row.get("rounds", [])) for row in streams),
        planned_stream_rounds=3 * planned,
        complete_trajectory=complete,
        audit_passed=certified,
        certification="Complete saved trajectory independently audited."
        if complete and certified
        else "Incomplete trajectory or missing/failed final audit; no complete-loop certification.",
        continuation="All planned rounds run independently of walking, fit quality and temporary regression.",
        teacher="Attempted acquisition and admitted/fallback experience are reported separately; reused data are not new acquisition.",
        walking="Walking requires full 1,000-control survival and velocity tracking; retention-only gains remain separate.",
        convergence="Finite contraction measurements are not proof of convergence or superiority. Equal-compute low-pass cloning/DAgger and second-body controls are absent.",
        declared_final_criteria=result.get("criteria"),
    )


def _collect(base: Path, saved: Saved) -> dict:
    result = saved.json(base / "worker/result.json", "aggregate worker result") or {}
    streams, teachers, evaluations, curves, errors, admissions, probes = [], [], [], [], [], [], []
    first_primary: dict[tuple, dict] = {}
    teacher_slot_zero: dict[tuple, dict] = {}
    planned = result.get("planned_rounds", 8)
    for spec in result.get("streams", []):
        stream = _load(saved, spec.get("result"), "stream result")
        if stream is None:
            continue
        streams.append(stream)
        stream_id = spec.get("stream", stream.get("stream"))

        def evaluation_rows(rows: list[dict], round_index: int) -> None:
            for row in rows:
                identity = dict(
                    round=round_index,
                    stream=stream_id,
                    arm=row["arm"],
                    kind=row["kind"],
                    panel=row["panel"],
                    reset_seed=row["reset_seed"],
                )
                key = (round_index, stream_id, row["arm"])
                if row["panel"] == "primary":
                    previous = first_primary.get(key)
                    if previous is None or row["reset_seed"] < previous["reset_seed"]:
                        first_primary[key] = identity | dict(trace=None)
                value = _load(saved, row.get("result"), "standalone outcome")
                if value is None or "outcome" not in value or "trace" not in value:
                    saved.failure(
                        "standalone outcome/trace",
                        repr(row.get("result")),
                        "Incomplete saved evaluator result",
                    )
                    continue
                actual = identity | dict(trace=value["trace"]) | value["outcome"]
                evaluations.append(actual)
                if (
                    row["panel"] == "primary"
                    and first_primary[key]["reset_seed"] == row["reset_seed"]
                ):
                    first_primary[key] = actual

        def probe_rows(rows: list[dict], round_index: int) -> None:
            for row in rows:
                value = _load(saved, row.get("result"), "student-state probe")
                if value is None:
                    continue
                for anchor in value.get("anchors", []):
                    probes.append(
                        dict(round=round_index, stream=stream_id, arm=row["arm"], **anchor)
                    )

        evaluation_rows(stream.get("initial_evaluations", []), 0)
        probe_rows(stream.get("initial_probes", []), 0)
        for block in stream.get("rounds", []):
            round_index = block["round"]
            for row in block.get("acquisitions", []):
                value = _load(saved, row.get("manifest"), "attempted teacher manifest")
                if value is None or "outcome" not in value:
                    saved.failure(
                        "attempted teacher outcome",
                        repr(row.get("manifest")),
                        "Incomplete saved controller manifest",
                    )
                    continue
                teachers.append(
                    dict(
                        round=round_index,
                        stream=stream_id,
                        arm=row["arm"],
                        slot=row["slot"],
                        role="attempt",
                        trace=value.get("trace"),
                        alpha=value.get("alpha"),
                    )
                    | value["outcome"]
                )
            for row in block.get("admissions", []):
                identity = dict(
                    round=round_index, stream=stream_id, arm=row["arm"], slot=row["slot"]
                )
                admissions.append(
                    identity
                    | {
                        key: row.get(key)
                        for key in (
                            "admitted",
                            "new_experience",
                            "fallback",
                            "last_accepted_round",
                            "age",
                            "new_rows",
                            "chosen_experience_fingerprint",
                            "previous_experience_fingerprint",
                        )
                    }
                    | dict(chosen=row.get("chosen"), attempt=row.get("attempt"))
                )
                value = _load(saved, row.get("chosen"), "retained teacher manifest")
                if value is None or "outcome" not in value:
                    saved.failure(
                        "retained teacher outcome",
                        repr(row.get("chosen")),
                        "Missing competent retained experience",
                    )
                    continue
                measured = (
                    identity
                    | dict(role="retained", trace=value.get("trace"), alpha=value.get("alpha"))
                    | value["outcome"]
                )
                teachers.append(measured)
                if row["slot"] == 0:
                    teacher_slot_zero[(round_index, stream_id, row["arm"])] = measured
            evaluation_rows(block.get("evaluations", []), round_index)
            probe_rows(block.get("probes", []), round_index)
            for row in block.get("fits", []):
                value = _load(saved, row.get("result"), "fit result")
                if value is None:
                    continue
                identity = dict(round=round_index, stream=stream_id, arm=row["arm"])
                if value.get("fit"):
                    fitted = _load(saved, value["fit"], "fitting curve") or {}
                    curves.extend(identity | point for point in fitted.get("learning_curve", []))
                for stage in ("before", "after"):
                    for name, metric in value.get("diagnostics_" + stage, {}).items():
                        errors.append(
                            identity
                            | dict(
                                stage=stage,
                                dataset=name,
                                rows=metric.get("rows"),
                                action_mse=metric.get("action_mse"),
                                rms_l2=metric.get("rms_l2"),
                            )
                        )
        for row in stream.get("final_evaluations", []):
            evaluation_rows([row], 0 if row["kind"] == "initial" else planned)
    return dict(
        result=result,
        streams=streams,
        teachers=teachers,
        evaluations=evaluations,
        curves=curves,
        errors=errors,
        admissions=admissions,
        probes=probes,
        first_primary=first_primary,
        teacher_slot_zero=teacher_slot_zero,
    )


def _group(rows: list[dict], keys: tuple[str, ...]) -> list[tuple[tuple, list[dict]]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return sorted(groups.items())


def _figures(out: Path, data: dict, saved: Saved) -> list[str]:
    if not any(data[key] for key in ("teachers", "evaluations", "curves", "errors", "probes")):
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []

    def finish(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out / name, dpi=120)
        plt.close(fig)
        figures.append(name)

    summaries = endpoint_summary(data["evaluations"])
    if summaries:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for key, rows in _group(
            [row for row in summaries if row["panel"] == "primary"], ("stream", "arm")
        ):
            rows.sort(key=lambda row: row["round"])
            label = f"s{key[0]}/{key[1]}"
            for axis, field in zip(
                axes, ("walking", "full_survival", "mean_complete_cost"), strict=True
            ):
                axis.plot(
                    [row["round"] for row in rows],
                    [row[field] for row in rows],
                    marker="o",
                    label=label,
                )
        for axis, title in zip(
            axes,
            ("Walking / observed16", "Full survival / observed16", "Mean complete standalone cost"),
            strict=True,
        ):
            axis.set(title=title, xlabel="Completed learning rounds")
        axes[0].set_ylim(0, 16)
        axes[1].set_ylim(0, 16)
        axes[-1].legend(fontsize=7)
        finish(fig, "standalone_trajectory.png")
    if data["teachers"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for key, rows in _group(data["teachers"], ("stream", "arm", "role")):
            rounds = sorted({row["round"] for row in rows})
            label = f"s{key[0]}/{key[1]}/{key[2]}"
            linestyle = "--" if key[2] == "attempt" else "-"
            values = [
                [row for row in rows if row["round"] == round_index] for round_index in rounds
            ]
            for axis, field in zip(
                axes, ("local_walking", "full_survival", "complete_cost"), strict=True
            ):
                ys = [
                    sum(row.get(field) is True for row in group)
                    if field != "complete_cost"
                    else _mean([row.get(field) for row in group])
                    for group in values
                ]
                axis.plot(rounds, ys, linestyle=linestyle, marker=".", label=label)
        for axis, title in zip(
            axes,
            ("Local300-control walking /8", "Local full survival /8", "Mean complete local cost"),
            strict=True,
        ):
            axis.set(title=title, xlabel="Round")
        axes[-1].legend(fontsize=6, ncol=2)
        finish(fig, "teacher_trajectory.png")
    activity = activity_summary(data["admissions"])
    if activity:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for key, rows in _group(activity, ("stream", "arm")):
            axes[0].plot(
                [row["round"] for row in rows],
                [row["new_experience_slots"] / 8 for row in rows],
                marker="o",
                label=f"s{key[0]}/{key[1]}",
            )
            axes[1].plot(
                [row["round"] for row in rows],
                [row["max_age"] for row in rows],
                marker="o",
                label=f"s{key[0]}/{key[1]}",
            )
        axes[0].set(title="Changed competent experience fraction", xlabel="Round", ylim=(0, 1))
        axes[1].set(
            title="Oldest retained episode age", xlabel="Round", ylabel="Rounds since admission"
        )
        axes[1].legend(fontsize=7)
        finish(fig, "acquisition_activity.png")
    if data["curves"]:
        fig, axis = plt.subplots(figsize=(12, 4))
        for key, rows in _group(data["curves"], ("round", "stream", "arm")):
            axis.plot(
                [row.get("update", row["fit_update"]) for row in rows],
                [row["action_mse"] for row in rows],
                linewidth=0.6,
                alpha=0.65,
                label=f"r{key[0]}/s{key[1]}/{key[2]}",
            )
        axis.set(xlabel="Cumulative optimizer update", ylabel="Saved minibatch action MSE")
        finish(fig, "fitting_trajectory.png")
    if data["errors"]:
        datasets = sorted({row["dataset"] for row in data["errors"]})
        fig, axes = plt.subplots(1, len(datasets), squeeze=False, figsize=(4.4 * len(datasets), 4))
        for axis, dataset in zip(axes[0], datasets, strict=True):
            for key, rows in _group(
                [row for row in data["errors"] if row["dataset"] == dataset],
                ("stream", "arm", "stage"),
            ):
                axis.plot(
                    [row["round"] - (0.15 if row["stage"] == "before" else 0) for row in rows],
                    [row["action_mse"] for row in rows],
                    marker=".",
                    linestyle="--" if key[2] == "before" else "-",
                    label=f"s{key[0]}/{key[1]}/{key[2]}",
                )
            axis.set(title=dataset, xlabel="Round (before offset -.15)", ylabel="Saved action MSE")
        axes[0][-1].legend(fontsize=6)
        finish(fig, "reconstruction_trajectory.png")
    if data["probes"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for key, rows in _group(data["probes"], ("stream", "arm")):
            endpoints = sorted({row["round"] for row in rows})
            observed = [
                [row for row in rows if row["round"] == i and row.get("status") == "OBSERVED"]
                for i in endpoints
            ]
            axes[0].plot(
                endpoints,
                [_mean([row.get("gap_action_mse") for row in group]) for group in observed],
                marker="o",
                label=f"s{key[0]}/{key[1]}",
            )
            axes[1].plot(endpoints, [len(group) for group in observed], marker="o")
        axes[0].set(
            title="Observed scratch-plan student-state action gap",
            xlabel="Round",
            ylabel="Mean action MSE on observed anchors",
        )
        axes[1].set(title="Reached probe anchors", xlabel="Round", ylim=(0, 3))
        axes[0].legend(fontsize=7)
        finish(fig, "student_state_gap_trajectory.png")
    # Frozen slot0 and first primary reset only; no favorable replacement.
    identities = sorted(data["teacher_slot_zero"])
    for round_stream in sorted({key[:2] for key in identities}):
        arms = sorted(key[2] for key in identities if key[:2] == round_stream)
        fig, axes = plt.subplots(len(arms), 6, squeeze=False, figsize=(15, 3.4 * len(arms)))
        for index, arm in enumerate(arms):
            key = (*round_stream, arm)
            teacher = data["teacher_slot_zero"][key]
            policy = data["first_primary"].get(key)
            for value, label, style in (
                (teacher, "retained teacher slot0", "-"),
                (policy, "policy first primary", "--"),
            ):
                raw = saved.trace(value["trace"], label) if value and value.get("trace") else None
                if raw is None:
                    saved.failure(
                        "joint-angle illustration",
                        repr(key),
                        label + " missing; no favorable substitute",
                    )
                    continue
                elapsed = raw["states"][:, 0] - raw["states"][0, 0]
                for joint in range(6):
                    axes[index, joint].plot(
                        elapsed, raw["states"][:, joint + 4], style, linewidth=0.8, label=label
                    )
                    if index == 0:
                        axes[index, joint].set_title(
                            f"Joint{joint}, state column{joint + 4}", fontsize=8
                        )
                    if joint == 0:
                        axes[index, joint].set_ylabel(arm)
            axes[index, -1].legend(fontsize=6)
        fig.suptitle(
            f"UNPAIRED joint-angle illustrations r{round_stream[0]}/s{round_stream[1]}: different start contexts; includes failures",
            fontsize=10,
        )
        fig.supxlabel("Actual elapsed simulator seconds; no post-fall tail")
        finish(fig, f"joint_angles_r{round_stream[0]}_s{round_stream[1]}.png")
    return figures


def render(base: Path, out: Path) -> dict:
    """Create a saved-only report in a fresh directory, without scientific calls."""
    start, cpu = time.monotonic(), time.process_time()
    base, out = Path(base).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    saved = Saved()
    data = _collect(base, saved)
    audit = saved.json(base / "audit/final/receipt.json", "final independent audit")
    verdict = answers(data["result"], data["streams"], audit)
    tables = {
        "teacher_outcomes.csv": (
            data["teachers"],
            [
                "round",
                "stream",
                "arm",
                "slot",
                "role",
                "alpha",
                "controls",
                "local_walking",
                "full_survival",
                "complete_cost",
                "velocity_mae",
            ],
        ),
        "admission_fallback.csv": (
            data["admissions"],
            [
                "round",
                "stream",
                "arm",
                "slot",
                "admitted",
                "new_experience",
                "fallback",
                "last_accepted_round",
                "age",
                "new_rows",
                "chosen_experience_fingerprint",
                "previous_experience_fingerprint",
            ],
        ),
        "acquisition_activity.csv": (
            activity_summary(data["admissions"]),
            [
                "round",
                "stream",
                "arm",
                "slots",
                "newly_admitted",
                "new_experience_slots",
                "fallback_slots",
                "new_rows",
                "max_age",
                "mean_age",
            ],
        ),
        "standalone_outcomes.csv": (
            data["evaluations"],
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
        ),
        "standalone_summary.csv": (
            endpoint_summary(data["evaluations"]),
            [
                "round",
                "stream",
                "arm",
                "kind",
                "panel",
                "episodes",
                "walking",
                "full_survival",
                "mean_controls",
                "mean_complete_cost",
                "mean_velocity_mae",
            ],
        ),
        "fitting_curves.csv": (
            data["curves"],
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
        ),
        "reconstruction.csv": (
            data["errors"],
            ["round", "stream", "arm", "stage", "dataset", "rows", "action_mse", "rms_l2"],
        ),
        "student_state_probes.csv": (
            data["probes"],
            [
                "round",
                "stream",
                "arm",
                "control",
                "status",
                "seed",
                "executed_controls",
                "gap_action_mse",
            ],
        ),
    }
    tables["student_gap_endpoints.csv"] = (
        student_gap_endpoints(data["probes"], verdict["planned_rounds"]),
        [
            "stream",
            "arm",
            "initial_observed",
            "final_observed",
            "common_anchors",
            "common_controls",
            "initial_common_action_mse",
            "final_common_action_mse",
            "ratio",
            "no_lost_anchor_coverage",
        ],
    )
    for name, (rows, fields) in tables.items():
        _csv(out / name, rows, fields)
    write(out / "declared_final_criteria.json", data["result"].get("criteria", {}))
    figures = _figures(out, data, saved)
    verdict["render_evidence_complete"] = not saved.missing
    if saved.missing:
        verdict["certification"] = (
            "Missing or unreadable evidence in this render; no complete-loop certification. "
            "The saved audit declaration is reported separately."
        )
    final = [
        row
        for row in endpoint_summary(data["evaluations"])
        if row["round"] == verdict["planned_rounds"]
    ]
    lines = [
        "# L3.0 — complete planned alternation, saved evidence",
        "",
        f"Worker status: **{data['result'].get('status', 'MISSING_RESULT')}**. {verdict['certification']}",
        f"Closed stream-rounds: **{verdict['completed_stream_rounds']}/{verdict['planned_stream_rounds']}**; planned length **{verdict['planned_rounds']}** additional rounds per stream.",
        "",
        "**Did teachers change while remaining competent?** Attempted controller outcomes and admitted experience are separate in [teacher outcomes](teacher_outcomes.csv). [Activity and fallback](acquisition_activity.csv) show every old episode reuse; a fallback-only block is inactive acquisition, even if fitting lowers its reconstruction error.",
        "",
        "**Did learning transfer?** Final observed panels are below. Intermediate zero walking or regression did not alter the planned roster. Full survival without velocity tracking is retention, not walking.",
        "",
        "| Stream/arm | Panel | Observed episodes | Walking | Full survival |",
        "|---|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['stream']}/{row['arm']} | {row['panel']} | {row['episodes']} | {row['walking']} | {row['full_survival']} |"
        for row in final
    )
    lines += [
        "",
        "**Did the gap contract?** See [fixed/current/held reconstruction](reconstruction.csv), [student-state queries](student_state_probes.csv), [common-anchor endpoint comparison](student_gap_endpoints.csv) and the [declared final criteria](declared_final_criteria.json). Current-teacher gap can change with the data distribution; fixed-anchor gap uses fixed states. Student queries use independent MPPI from a zero plan at the restored actual context. The observed-gap curve must be read beside reached-anchor coverage. Missing post-fall anchors are missing, not zero discrepancy. Endpoint ratios use common reached control indices and explicitly flag lost coverage. Criterion declarations need the independent audit and are not a convergence proof.",
        "",
        "[Full standalone outcomes](standalone_outcomes.csv) · [All fitting updates](fitting_curves.csv) · [Every admission/fallback](admission_fallback.csv)",
        "",
        "Three retained L2 minibatch streams continue different arm-specific actors/Adam states; this is not a new matched-actor initialization. Eight restart contexts share one original episode. Local teachers cover300 controls; standalone evaluations cover up to1,000. Joint-angle figures are UNPAIRED frozen slot0/first-primary illustrations, including failures, not paired performance gaps. No favorable substitution or reference-path requirement is used.",
        "",
        verdict["convergence"],
        "",
        f"Missing/unreadable evidence entries: **{len(saved.missing)}**. Report rendering performs **zero physics, planner queries, neural forwards, gradients or Adam updates**. Saved accounting is preserved in the manifest; nested wall/CPU scopes must not be added twice.",
        "",
        "Figures: "
        + (", ".join(f"[{name}]({name})" for name in figures) or "none: saved measurements absent"),
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    artifacts = [
        dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(out.rglob("*"))
        if p.is_file()
    ]
    manifest = dict(
        schema="local-coupling-multiround-report-1",
        status="RENDERED_WITH_MISSING_EVIDENCE" if saved.missing else "RENDERED",
        worker_status=data["result"].get("status"),
        answers=verdict,
        missing_evidence=saved.missing,
        external_inputs=list(saved.references.values()),
        artifacts=artifacts,
        figures=figures,
        saved_accounting=data["result"].get("accounting"),
        measurements=dict(
            streams=len(data["streams"]),
            teacher_records=len(data["teachers"]),
            admission_records=len(data["admissions"]),
            standalone_episodes=len(data["evaluations"]),
            fitting_updates=len(data["curves"]),
            reconstruction_records=len(data["errors"]),
            student_probe_anchors=len(data["probes"]),
        ),
        rendering_work=dict(
            neural_rows=0,
            model_controls=0,
            native_controls=0,
            physics_substeps=0,
            backward_rows=0,
            optimizer_steps=0,
            logical_io=saved.reads,
            physical_disk_io_bytes=None,
            cpu_seconds_before_manifest=time.process_time() - cpu,
            wall_seconds_before_manifest=time.monotonic() - start,
        ),
    )
    write(out / "report_manifest.json", manifest)
    return manifest
