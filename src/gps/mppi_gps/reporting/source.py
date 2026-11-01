"""Create-only S1 source-comparison reports from saved results and learning curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..provenance import sha, write

REPORT_SCHEMA = "walker-student-source-s1-report-v1"
CURVE_SCHEMA = "walker-student-source-s1-learning-bins-v1"
RESULT_SCHEMAS = {"walker-student-source-s1-result-v1", "walker-student-source-s1-fit-v1"}
INTERRUPTED_SCHEMA = "walker-student-source-s1-interrupted-summary-v1"
INTERRUPTION_AUDIT_SCHEMA = "walker-student-source-s1-interruption-audit-v1"
SPLITS = ("mppi_train", "mppi_validation", "sac_reference_train", "sac_reference_validation")
CURVE_FIELDS = ("action_mse", "proximal_loss", "total_loss", "unclipped_gradient_norm")
COST_FIELDS = ("raw_cost", "terminal_cost", "terminal_adjustment", "absorbing_cost", "cost")


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output = {}
    for key, child in value.items():
        name = prefix + key
        if isinstance(child, Mapping):
            output.update(_flatten(child, name + "_"))
        elif isinstance(child, (list, tuple)):
            output[name] = json.dumps(child, allow_nan=False, sort_keys=True)
        else:
            output[name] = child
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] = ()) -> None:
    names = list(dict.fromkeys([*columns, *(key for row in rows for key in row)]))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _point_identity(name: str, points: Mapping[str, Any]) -> dict[str, Any]:
    point = points.get(name, {})
    return dict(
        point=name,
        pair=point.get("pair"),
        arm=point.get("arm", "reference" if name == "reference" else None),
        update=point.get("update"),
    )


def episode_components(episode: Mapping[str, Any]) -> dict[str, float | None]:
    """Separate the native terminal and adjustment from the absorbing tail once."""
    output = {field: episode.get(field) for field in COST_FIELDS}
    if all(
        output.get(field) is not None
        for field in ("cost", "terminal_cost", "terminal_adjustment", "absorbing_cost")
    ):
        output["raw_cost"] = float(
            output["cost"]
            - output["terminal_cost"]
            - output["terminal_adjustment"]
            - output["absorbing_cost"]
        )
    else:
        output["raw_cost"] = None
    return output


def _observed_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in episodes if row.get("complete") is True]
    output = dict(
        recorded_attempts=len(episodes),
        completed_episodes=len(completed),
        incomplete_records=len(episodes) - len(completed),
    )
    for field, label in (("success", "walking"), ("survived", "survival")):
        known = [row[field] for row in completed if row.get(field) is not None]
        output[label + "_known_episodes"] = len(known)
        output[label + "_count"] = sum(bool(value) for value in known) if known else None
        output[label + "_fraction"] = float(np.mean(known)) if known else None
    for field in ("steps", "mae", *COST_FIELDS):
        values = [
            episode_components(row).get(field) if field in COST_FIELDS else row.get(field)
            for row in completed
        ]
        known = [float(value) for value in values if value is not None]
        output["mean_" + field] = float(np.mean(known)) if known else None
        output[field + "_known_episodes"] = len(known)
    return output


def _evaluation_records(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Include explicit incomplete attempts without inventing their final outcomes."""
    output = {
        panel: {
            name: dict(record, episodes=list(record.get("episodes", [])))
            for name, record in records.items()
        }
        for panel, records in result.get("evaluations", {}).items()
    }
    for partial in result.get("partial_attempts", []):
        panel, point = partial["panel"], partial["point"]
        record = output.setdefault(panel, {}).setdefault(point, dict(episodes=[]))
        if any(row.get("tag") == partial.get("tag") for row in record["episodes"]):
            raise ValueError("S1 interrupted summary duplicates a partial native attempt")
        row = dict(partial)
        row.update(
            complete=False,
            success=None,
            survived=None,
            cost=None,
            mae=None,
            terminal_cost=None,
            terminal_adjustment=None,
            absorbing_cost=None,
        )
        record["episodes"].append(row)
    return output


def _teacher_tables(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    teachers = []
    for name, record in result.get("teacher_outcomes", {}).items():
        row = {key: record.get(key) for key in ("id", "source", "split", "seed", "rows")}
        row["id"] = row["id"] or name
        row.update(_flatten(record.get("outcome", {}), "outcome_"))
        teachers.append(row)
    groups = []
    for source in sorted({str(row["source"]) for row in teachers}):
        for split in ("all", "train", "validation"):
            selected = [
                row
                for row in teachers
                if row["source"] == source and (split == "all" or row["split"] == split)
            ]
            if not selected:
                continue
            group = dict(
                source=source,
                split=split,
                episodes=len(selected),
                scope="Historical recorded teachers; equal episode weight",
            )
            for field in ("cost", "mae", "success", "survived"):
                known = [
                    float(row["outcome_" + field])
                    for row in selected
                    if row.get("outcome_" + field) is not None
                ]
                group["mean_" + field] = float(np.mean(known)) if known else None
                group[field + "_known_episodes"] = len(known)
            groups.append(group)
    return teachers, groups


def matched_reference_gaps(
    result: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare each reference-data child with the one shared fresh reference panel."""
    gaps, availability = [], []
    children = {
        name: point
        for name, point in result.get("points", {}).items()
        if point.get("arm") == "sac_reference" and point.get("update") == 2000
    }
    for panel in ("primary", "confirmation"):
        records = result.get("evaluations", {}).get(panel, {})
        reference = records.get("reference", {}).get("episodes", [])
        expected = config.get(panel + "_seeds") if config is not None else None
        for name, point in children.items():
            child = records.get(name, {}).get("episodes", [])
            identity = dict(
                panel=panel,
                child_point=name,
                pair=point.get("pair"),
                teacher_point="reference",
                source="sac_reference",
            )
            reference_seeds = [row.get("seed") for row in reference]
            child_seeds = [row.get("seed") for row in child]
            reason = None
            if len(reference) != 16 or len(child) != 16:
                reason = "Both complete 16-reset episode rosters are required; missing attempts are not imputed"
            elif (
                any(seed is None for seed in reference_seeds)
                or len(set(reference_seeds)) != 16
                or reference_seeds != child_seeds
                or (expected is not None and reference_seeds != list(expected))
            ):
                reason = "Reference and child reset rosters are duplicated, misaligned or differ from the fixed panel"
            elif any(row.get("complete") is not True for row in [*reference, *child]):
                reason = "An incomplete episode record prevents the matched gap"
            elif any(
                row.get(field) is None
                for row in [*reference, *child]
                for field in ("success", "survived", "steps", "mae", *COST_FIELDS[1:])
            ):
                reason = "A required outcome or native cost component is unavailable"
            availability.append(dict(**identity, available=reason is None, reason=reason))
            if reason is not None:
                continue
            row = dict(
                **identity,
                paired_resets=16,
                shared_reference_control=True,
                reference_walking_count=sum(bool(episode["success"]) for episode in reference),
                child_walking_count=sum(bool(episode["success"]) for episode in child),
                reference_survival_count=sum(bool(episode["survived"]) for episode in reference),
                child_survival_count=sum(bool(episode["survived"]) for episode in child),
                direction="Child minus fresh reference; negative count/control gaps and positive cost/velocity-error gaps indicate deficits",
                scope="One shared reference control across the three training pairs; not three independent teacher replications. Native snapshot pairing belongs to the independent audit.",
            )
            row["walking_count_gap"] = row["child_walking_count"] - row["reference_walking_count"]
            row["survival_count_gap"] = (
                row["child_survival_count"] - row["reference_survival_count"]
            )
            for field in ("steps", "mae", *COST_FIELDS):
                pairs = [
                    (episode_components(candidate)[field], episode_components(baseline)[field])
                    if field in COST_FIELDS
                    else (candidate[field], baseline[field])
                    for baseline, candidate in zip(reference, child, strict=True)
                ]
                row["reference_mean_" + field] = float(np.mean([baseline for _, baseline in pairs]))
                row["child_mean_" + field] = float(np.mean([candidate for candidate, _ in pairs]))
                row["mean_" + field + "_gap"] = float(
                    np.mean([candidate - baseline for candidate, baseline in pairs])
                )
            gaps.append(row)
    return gaps, availability


def build_tables(
    result: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> dict[str, list[dict[str, Any]]]:
    points = result.get("points", {})
    fitting, trajectories, actuators, control_bins, standalone, episodes, paired, competence = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for name, point in sorted(points.items()):
        identity = _point_identity(name, points)
        for split, metric in point.get("metrics", {}).items():
            fitting.append(
                dict(
                    **identity,
                    split=split,
                    **_flatten(
                        {key: value for key, value in metric.items() if key != "per_trajectory"}
                    ),
                )
            )
            for teacher, score in metric.get("per_trajectory", {}).items():
                trajectories.append(
                    dict(
                        **identity,
                        split=split,
                        trajectory=teacher,
                        **_flatten(
                            {
                                key: value
                                for key, value in score.items()
                                if key not in {"bins", "per_actuator_mse"}
                            }
                        ),
                    )
                )
                for actuator, mse in enumerate(score.get("per_actuator_mse", [])):
                    actuators.append(
                        dict(
                            **identity, split=split, trajectory=teacher, actuator=actuator, mse=mse
                        )
                    )
                for control_bin in score.get("bins", []):
                    control_bins.append(
                        dict(**identity, split=split, trajectory=teacher, **_flatten(control_bin))
                    )
    for panel, records in _evaluation_records(result).items():
        for name, record in records.items():
            identity = dict(panel=panel, **_point_identity(name, points))
            saved = record.get("episodes", [])
            standalone.append(
                dict(
                    **identity,
                    **_observed_summary(saved),
                    **_flatten(record.get("summary", {}), "declared_"),
                )
            )
            for episode in saved:
                row = {key: value for key, value in episode.items() if key != "raw"}
                saved_row = dict(identity, **_flatten(row))
                saved_row.update(episode_components(episode))
                episodes.append(saved_row)
    gates = result.get("gates", {})
    for panel in ("primary", "confirmation"):
        for row in gates.get(panel, {}).get("pairs", []):
            paired.append(dict(panel=panel, **_flatten(row)))
    for name, value in gates.get("competence", {}).items():
        competence.append(dict(**_point_identity(name, points), provisionally_competent=value))
    teachers, teacher_groups = _teacher_tables(result)
    gaps = []
    for row in standalone:
        if row.get("arm") not in {"mppi", "sac_reference"} or row.get("update") != 2000:
            continue
        for teacher in teacher_groups:
            if teacher["source"] != row["arm"]:
                continue
            cost, baseline = row.get("mean_cost"), teacher.get("mean_cost")
            gaps.append(
                dict(
                    panel=row["panel"],
                    point=row["point"],
                    pair=row["pair"],
                    arm=row["arm"],
                    teacher_split=teacher["split"],
                    historical_teacher_episodes=teacher["episodes"],
                    completed_student_episodes=row["completed_episodes"],
                    teacher_mean_cost=baseline,
                    student_mean_cost=cost,
                    student_minus_teacher_cost=cost - baseline
                    if cost is not None and baseline is not None
                    else None,
                    scope="Descriptive own-source historical teacher versus fresh student; reset contexts are unpaired",
                )
            )
    paired_components = []
    final = {
        (point.get("pair"), point.get("arm")): name
        for name, point in points.items()
        if point.get("update") == 2000
    }
    for panel, records in result.get("evaluations", {}).items():
        for pair in sorted({key[0] for key in final}):
            left = records.get(final.get((pair, "mppi"), ""), {}).get("episodes", [])
            right = records.get(final.get((pair, "sac_reference"), ""), {}).get("episodes", [])
            if (
                not left
                or [row.get("seed") for row in left] != [row.get("seed") for row in right]
                or any(not row.get("complete") for row in [*left, *right])
            ):
                continue
            record = dict(
                panel=panel,
                pair=pair,
                completed_reset_pairs=len(left),
                direction="sac_reference-data child minus MPPI-data child",
            )
            for field in COST_FIELDS:
                values = [
                    (
                        episode_components(candidate).get(field),
                        episode_components(baseline).get(field),
                    )
                    for baseline, candidate in zip(left, right, strict=True)
                ]
                record[field + "_delta"] = (
                    float(np.mean([candidate - baseline for candidate, baseline in values]))
                    if all(
                        candidate is not None and baseline is not None
                        for candidate, baseline in values
                    )
                    else None
                )
            paired_components.append(record)
    components = [
        {
            key: value
            for key, value in row.items()
            if key in {"point", "pair", "arm", "update", "panel", "completed_episodes"}
            or key.startswith("mean_")
        }
        for row in standalone
    ]
    reference_gaps, reference_gap_status = matched_reference_gaps(result, config)
    return dict(
        fitting=fitting,
        trajectories=trajectories,
        actuator_errors=actuators,
        reconstruction_control_bins=control_bins,
        standalone=standalone,
        episodes=episodes,
        paired_sources=paired,
        competence=competence,
        source_teachers=teachers,
        source_teacher_groups=teacher_groups,
        teacher_student_gaps=gaps,
        cost_components=components,
        paired_cost_components=paired_components,
        matched_reference_gaps=reference_gaps,
        matched_reference_gap_status=reference_gap_status,
        partial_attempts=[_flatten(row) for row in result.get("partial_attempts", [])],
        missing_roster=[_flatten(row) for row in result.get("missing_roster", [])],
    )


def paired_bootstrap(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Cross paired training streams with one shared reset sample for both arms."""
    settings = config.get("bootstrap", {})
    draws, seed = int(settings.get("draws", 2000)), int(settings.get("seed", 2026111199))
    rng = np.random.Generator(np.random.PCG64(seed))
    points = result.get("points", {})
    pairs = sorted({point["pair"] for point in points.values() if point.get("update") == 2000})
    lookup = {
        (point["pair"], point.get("arm")): name
        for name, point in points.items()
        if point.get("update") == 2000
    }
    output = dict(
        draws=draws,
        seed=seed,
        scope="Descriptive paired-training-stream/shared-reset bootstrap. Fixed corpora are not resampled; frames, endpoints and resets are not independent training runs. Three streams do not establish population coverage.",
    )
    if result.get("schema") == INTERRUPTED_SCHEMA:
        for panel in ("primary", "confirmation"):
            output[panel] = dict(
                available=False,
                reason="Experiment interrupted; final comparative intervals are withheld. Verified-prefix observations remain descriptive and missing outcomes are not imputed.",
            )
        return output
    for panel in ("primary", "confirmation"):
        records = result.get("evaluations", {}).get(panel, {})
        expected_seeds = config.get(panel + "_seeds", [])
        arrays, reason = {}, None
        if len(pairs) != 3 or not expected_seeds:
            reason = "All three paired final streams and the fixed reset roster are required"
        for arm in ("mppi", "sac_reference"):
            if reason:
                break
            arrays[arm] = {
                field: [] for field in ("success", "survived", "steps", "cost", *COST_FIELDS[:-1])
            }
            for pair in pairs:
                record = records.get(lookup.get((pair, arm), ""))
                saved = record.get("episodes", []) if record else []
                if [row.get("seed") for row in saved] != expected_seeds or any(
                    not row.get("complete") for row in saved
                ):
                    reason = "Incomplete or unpaired final panel; missing outcomes are not imputed"
                    break
                for field in arrays[arm]:
                    values = [
                        episode_components(row).get(field)
                        if field in COST_FIELDS
                        else row.get(field)
                        for row in saved
                    ]
                    if any(value is None for value in values):
                        reason = "A required outcome or cost component is unavailable"
                        break
                    arrays[arm][field].append(values)
        if reason:
            output[panel] = dict(available=False, reason=reason)
            continue
        arrays = {
            arm: {field: np.asarray(values, dtype=np.float64) for field, values in measures.items()}
            for arm, measures in arrays.items()
        }
        replicas: dict[str, list[float]] = defaultdict(list)
        fields = ("success", "survived", "steps", "cost", *COST_FIELDS[:-1])
        for _ in range(draws):
            index = np.ix_(
                rng.integers(0, 3, size=3),
                rng.integers(0, len(expected_seeds), size=len(expected_seeds)),
            )
            for field in fields:
                delta = arrays["sac_reference"][field][index] - arrays["mppi"][field][index]
                replicas[field].append(float(np.mean(delta)))
        output[panel] = dict(
            available=True,
            training_streams=3,
            paired_resets=len(expected_seeds),
            direction="sac_reference-data child minus MPPI-data child; negative cost and positive walking/survival/controls favor reference data",
            intervals={
                field: np.percentile(values, [2.5, 97.5]).tolist()
                for field, values in replicas.items()
            },
        )
    return output


def _fits(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = result.get("fits", {})
    if isinstance(value, Mapping):
        return [dict(record, point=record.get("point", name)) for name, record in value.items()]
    return list(value)


def learning_bins(
    run: Path, result: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize already-computed minibatch losses; perform no extra predictions."""
    bins, bindings = [], []
    for fit in _fits(result):
        path = (run / fit["file"]).resolve()
        if not path.is_relative_to(run) or sha(path) != fit["sha256"]:
            raise ValueError("S1 learning-curve file binding differs")
        evidence = json.loads(path.read_text())
        curve = evidence["learning_curve"]
        if evidence.get("completed_updates", len(curve)) != len(curve):
            raise ValueError("S1 learning-curve completed-update count differs")
        updates = [int(row["fit_update"]) for row in curve]
        if updates != list(range(1, len(curve) + 1)):
            raise ValueError("S1 learning curves require the contiguous fresh fit stream")
        bindings.append(
            dict(
                point=fit["point"],
                file=fit["file"],
                sha256=fit["sha256"],
                bytes=path.stat().st_size,
                updates=len(curve),
            )
        )
        for start in range(0, len(curve), 100):
            rows = curve[start : start + 100]
            record = dict(
                point=fit["point"],
                pair=fit.get("pair"),
                arm=fit.get("arm"),
                start_fit_update=start + 1,
                stop_fit_update=start + len(rows),
                updates=len(rows),
            )
            for field in CURVE_FIELDS:
                values = [float(row[field]) for row in rows if row.get(field) is not None]
                if values:
                    if not np.isfinite(values).all():
                        raise ValueError("S1 learning curve contains a nonfinite measurement")
                    record[field + "_mean"] = float(np.mean(values))
                    record[field + "_min"] = float(np.min(values))
                    record[field + "_max"] = float(np.max(values))
            bins.append(record)
    return bins, bindings


def _load_interruption_summary(
    run: Path, path: Path, *, display_failed_audit: bool = False
) -> tuple[dict[str, Any], str, bool]:
    if (run / "result.json").exists():
        raise ValueError(
            "An interruption summary cannot replace an original closed evaluation result"
        )
    result = json.loads(path.read_text())
    if (
        result.get("schema") != INTERRUPTED_SCHEMA
        or result.get("execution_complete") is not False
        or result.get("comparison_complete") is not False
    ):
        raise ValueError("S1 interruption summary must identify an incomplete comparison")
    if result.get("audit_complete") is not True:
        raise ValueError("S1 interruption summary lacks a completed scoped-prefix audit")
    gates = result.get("gates", {})
    if gates.get("withheld") is not True or any(
        name in gates
        for name in ("confirmed_walking_benefit", "competence", "primary", "confirmation")
    ):
        raise ValueError("S1 interrupted comparison must withhold final gates and competence flags")
    for field, filename in (
        ("fit_result_sha256", "fit_result.json"),
        ("config_sha256", "config.json"),
        ("input_manifest_sha256", "input_manifest.json"),
    ):
        if result.get(field) != sha(run / filename):
            raise ValueError("S1 interruption summary input binding differs: " + filename)
    fit = json.loads((run / "fit_result.json").read_text())
    if (
        fit.get("schema") != "walker-student-source-s1-fit-v1"
        or fit.get("execution_complete") is not True
        or fit.get("error")
    ):
        raise ValueError("S1 interruption summary requires the original complete fit phase")
    for field in ("points", "fits", "samplers", "teacher_outcomes"):
        if result.get(field) != fit.get(field):
            raise ValueError("S1 interruption summary changed original fit ownership: " + field)
    binding = result.get("audit_receipt", {})
    if binding.get("file") != "receipt.json" or binding.get("schema") != INTERRUPTION_AUDIT_SCHEMA:
        raise ValueError("S1 interruption audit receipt location or schema differs")
    receipt_path = path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != INTERRUPTION_AUDIT_SCHEMA or receipt.get("summary_sha256") != sha(
        path
    ):
        raise ValueError("S1 interruption audit receipt binding differs")
    passed = receipt.get("passed") is True
    if passed and result.get("integrity_status") != "valid_scoped_prefix":
        raise ValueError("S1 passed audit and summary integrity status differ")
    if not passed and not display_failed_audit:
        raise ValueError(
            "S1 audit failed; explicit --display-failed-audit is required for a forensic display"
        )
    if not passed and result.get("integrity_status") != "incomplete_retained_evidence":
        raise ValueError("S1 failed-audit display requires retained verified-prefix evidence")
    return result, sha(receipt_path), passed


def _load_ledger_claims(run: Path, path: Path) -> tuple[dict[str, Any], Path, bool]:
    claims = json.loads(path.read_text())
    if claims.get("schema") != "walker-s1-surviving-ledger-claims-v1":
        raise ValueError("S1 surviving ledger-claim schema differs")
    binding = claims["original_file_manifest"]
    if Path(binding["file"]).name != binding["file"]:
        raise ValueError("S1 original file catalog must be adjacent to ledger claims")
    catalog_path = path.parent / binding["file"]
    if sha(catalog_path) != binding["sha256"]:
        raise ValueError("S1 original file catalog binding differs")
    files = json.loads(catalog_path.read_text())["files"]
    if isinstance(files, list):
        files = {row["file"]: row for row in files}
    ledger = claims["source_ledger"]
    relative = Path(ledger["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("S1 source ledger path escapes its run")
    original = files.get(ledger["file"], {})
    if any(original.get(key) != ledger.get(key) for key in ("sha256", "bytes")):
        raise ValueError("S1 ledger source binding differs from the original file catalog")
    source = run / relative
    if source.is_file() and (
        sha(source) != ledger["sha256"] or source.stat().st_size != ledger["bytes"]
    ):
        raise ValueError("S1 surviving source ledger differs from its preservation catalog")
    verified = source.is_file()
    if verified:
        episodes, actors = [], []
        with source.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break
                if record.get("event") == "episode_complete":
                    episodes.append(record)
                elif record.get("event") == "source_actor_evaluated":
                    actors.append(record)
        if episodes != claims.get("episode_complete") or actors != claims.get("actor_summaries"):
            raise ValueError("S1 ledger claim rows differ from the original source ledger")
    return claims, catalog_path, verified


def _forensic_tables(
    tables: dict[str, Any], result: Mapping[str, Any], claims: Mapping[str, Any]
) -> None:
    roster = {row["tag"]: row for row in result["missing_roster"]}
    episodes = claims["episode_complete"]
    expected = {
        tag
        for tag, row in roster.items()
        if row["status"] in {"verified_complete", "unverified_raw"}
    }
    if (
        len({row["tag"] for row in episodes}) != len(episodes)
        or {row["tag"] for row in episodes} != expected
    ):
        raise ValueError("S1 ledger claims do not match the declared complete-record roster")
    annotated = []
    for row in episodes:
        role = roster[row["tag"]]
        if row.get("seed") != role["seed"]:
            raise ValueError("S1 ledger claim reset role differs")
        annotated.append(
            dict(
                **_flatten({key: value for key, value in row.items() if key != "raw"}),
                raw_verification_status=role["status"],
                evidence_scope="Verbatim worker ledger claim; independent native verification is separate",
            )
        )
    tables["ledger_episode_claims"] = annotated
    tables["unverified_ledger_episode_claims"] = [
        row for row in annotated if row["raw_verification_status"] == "unverified_raw"
    ]
    tables["worker_actor_summary_claims"] = [_flatten(row) for row in claims["actor_summaries"]]
    tables["verified_native_subset"] = [
        dict(
            row,
            evidence_scope="Individually verified, nonrepresentative retained subset; no native source ranking",
        )
        for row in tables["episodes"]
        if row.get("complete") is True
    ]
    tables["episodes"] = tables["verified_native_subset"]
    tables["standalone"] = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "panel",
                "point",
                "pair",
                "arm",
                "update",
                "recorded_attempts",
                "completed_episodes",
                "incomplete_records",
            }
        }
        for row in tables["standalone"]
    ]
    for name in (
        "teacher_student_gaps",
        "paired_cost_components",
        "cost_components",
        "matched_reference_gaps",
        "paired_sources",
        "competence",
    ):
        tables[name] = []
    for row in tables["matched_reference_gap_status"]:
        row.update(
            available=False,
            reason="Native comparisons withheld: failed retained-evidence integrity audit; subset is not representative",
        )


def _portable_bins(
    run: Path,
    result_file: str,
    result: Mapping[str, Any],
    cache: Path,
    capsule: Path | None,
    result_path: Path | None = None,
    ledger_claims: Path | None = None,
    catalog_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_path = result_path or run / result_file
    manifest_path, receipt_path = (
        cache.parent / "learning_bins_manifest.json",
        cache.parent / "receipt.json",
    )
    required = [
        run / "config.json",
        run / "input_manifest.json",
        result_path,
        cache,
        manifest_path,
        receipt_path,
    ]
    if result.get("schema") == INTERRUPTED_SCHEMA:
        required.extend([run / "fit_result.json", result_path.parent / "receipt.json"])
    if ledger_claims is not None:
        required.extend([ledger_claims, catalog_path])
    if capsule is not None:
        inventory = json.loads((capsule / "capsule.json").read_text())["files"]
        for path in required:
            relative = str(path.resolve().relative_to(capsule.resolve()))
            binding = inventory.get(relative, {})
            if (
                not path.is_file()
                or sha(path) != binding.get("sha256")
                or path.stat().st_size != binding.get("bytes")
            ):
                raise ValueError("S1 capsule file binding differs: " + relative)
    receipt = json.loads(receipt_path.read_text())
    for path in (cache, manifest_path):
        binding = receipt.get("outputs", {}).get(path.name, {})
        if sha(path) != binding.get("sha256") or path.stat().st_size != binding.get("bytes"):
            raise ValueError("S1 portable learning-bin file binding differs")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != CURVE_SCHEMA
        or manifest.get("bin_updates") != 100
        or manifest.get("result_file") != result_file
    ):
        raise ValueError("S1 portable learning-bin schema differs")
    if manifest.get("learning_bins_sha256") != sha(cache):
        raise ValueError("S1 portable learning-bin summary binding differs")
    for field, source_path in (
        ("result_sha256", result_path),
        ("config_sha256", run / "config.json"),
        ("input_manifest_sha256", run / "input_manifest.json"),
    ):
        if manifest.get(field) != sha(source_path):
            raise ValueError("S1 portable learning-bin source binding differs")
    if result.get("schema") == INTERRUPTED_SCHEMA and (
        manifest.get("interruption_summary_sha256") != sha(result_path)
        or manifest.get("fit_result_sha256") != sha(run / "fit_result.json")
        or manifest.get("interruption_audit_receipt_sha256")
        != sha(result_path.parent / "receipt.json")
    ):
        raise ValueError("S1 portable interruption-summary binding differs")
    if ledger_claims is not None and (
        manifest.get("failed_audit_display") is not True
        or manifest.get("ledger_claims_sha256") != sha(ledger_claims)
        or manifest.get("original_file_catalog_sha256") != sha(catalog_path)
    ):
        raise ValueError("S1 portable forensic claim binding differs")
    if ledger_claims is not None and not (
        receipt.get("verified_against_original_ledger") is True
        or receipt.get("source_report_verified_against_original_ledger") is True
    ):
        raise ValueError(
            "S1 portable claims lack a source report that verified original ledger rows"
        )
    expected = sorted((row["point"], row["file"], row["sha256"]) for row in _fits(result))
    bindings = manifest["source_curves"]
    if sorted((row["point"], row["file"], row["sha256"]) for row in bindings) != expected:
        raise ValueError("S1 portable full-curve roster differs")
    with cache.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key in {"pair", "start_fit_update", "stop_fit_update", "updates"}:
                row[key] = int(value) if value else None
            elif key.endswith(("_mean", "_min", "_max")):
                row[key] = float(value) if value else None
    for binding in bindings:
        selected = [row for row in rows if row["point"] == binding["point"]]
        covered = [
            index
            for row in selected
            for index in range(row["start_fit_update"], row["stop_fit_update"] + 1)
        ]
        if covered != list(range(1, binding["updates"] + 1)) or any(
            row["updates"] != row["stop_fit_update"] - row["start_fit_update"] + 1
            for row in selected
        ):
            raise ValueError("S1 portable bins omit or duplicate recorded updates")
    if set(row["point"] for row in rows) - set(row["point"] for row in bindings):
        raise ValueError("S1 portable bins contain an undeclared fit")
    return rows, bindings


def _save_figure(figure: Any, path: Path, plt: Any) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=160)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _figures(
    tables: Mapping[str, Any], config: Mapping[str, Any], out: Path, *, forensic: bool = False
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "mppi": "#335c81",
        "sac_reference": "#cf5c36",
        "incoming": "#90979c",
        "reference": "#2b8c5a",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, split in zip(axes.flat, SPLITS, strict=True):
        for arm, offset, marker in (
            ("incoming", 0.0, "^"),
            ("mppi", -0.15, "o"),
            ("sac_reference", 0.15, "s"),
        ):
            rows = [
                row
                for row in tables["fitting"]
                if row.get("split") == split
                and row.get("arm") == arm
                and row.get("mse") is not None
            ]
            axis.scatter(
                [row["pair"] + offset for row in rows],
                [row["mse"] for row in rows],
                color=colors[arm],
                marker=marker,
                label=arm,
            )
        axis.set(title=split, xlabel="Paired training seed", ylabel="Original-torque action MSE")
        axis.set_xticks([0, 1, 2])
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("All four fixed reconstruction splits; held data select no endpoint")
    _save_figure(figure, out / "reconstruction", plt)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for pair, axis in enumerate(axes):
        for arm in ("mppi", "sac_reference"):
            rows = sorted(
                (
                    row
                    for row in tables["learning_bins"]
                    if row.get("pair") == pair and row.get("arm") == arm
                ),
                key=lambda row: row["start_fit_update"],
            )
            rows = [row for row in rows if row.get("action_mse_mean") is not None]
            x = [(row["start_fit_update"] + row["stop_fit_update"]) / 2 for row in rows]
            axis.plot(
                x,
                [row["action_mse_mean"] for row in rows],
                color=colors[arm],
                marker=".",
                label=arm,
            )
            axis.fill_between(
                x,
                [row["action_mse_min"] for row in rows],
                [row["action_mse_max"] for row in rows],
                color=colors[arm],
                alpha=0.15,
            )
        axis.set(
            title=f"Training pair {pair}",
            xlabel="Already-computed fit update",
            ylabel="Minibatch action MSE",
        )
    axes[0].legend(fontsize=8)
    figure.suptitle("Fixed 100-update bins; min–max shading is minibatch spread, not uncertainty")
    _save_figure(figure, out / "learning_curves", plt)

    if forensic:
        figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
        order = list(config.get("evaluation_order", []))
        for axis, panel in zip(axes, ("primary", "confirmation"), strict=True):
            bottom = np.zeros(len(order))
            for status, color in (
                ("verified_complete", "#2b8c5a"),
                ("unverified_raw", "#b54040"),
                ("partial", "#cf8e36"),
                ("not_started", "#a8adb1"),
            ):
                values = [
                    sum(
                        row.get("panel") == panel
                        and row.get("point") == name
                        and row.get("status") == status
                        for row in tables["missing_roster"]
                    )
                    for name in order
                ]
                axis.bar(range(len(order)), values, bottom=bottom, color=color, label=status)
                bottom += values
            axis.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
            axis.set(
                title=panel, ylabel="Recorded-role availability, not performance", ylim=(0, 17)
            )
        axes[0].legend(fontsize=7)
        figure.suptitle("Native comparison withheld: incomplete retained evidence")
        _save_figure(figure, out / "standalone_outcomes", plt)
        figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.5,
            0.55,
            "Native comparison withheld: incomplete retained evidence\n\nNo source ranking, competence decision, cost comparison or gap is issued.\nIndividually verified episodes form a nonrepresentative retained subset.\nUnverified worker ledger claims are exported separately.",
            ha="center",
            va="center",
            fontsize=12,
        )
        _save_figure(figure, out / "cost_components", plt)
        return

    order = list(config.get("evaluation_order", []))
    if not order:
        order = sorted({row["point"] for row in tables["standalone"]})
    figure, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    for panel_index, panel in enumerate(("primary", "confirmation")):
        records = {row["point"]: row for row in tables["standalone"] if row["panel"] == panel}
        for axis, field, label in zip(
            axes[panel_index],
            ("walking_fraction", "survival_fraction", "mean_steps", "mean_cost"),
            (
                "Observed walking fraction",
                "Observed full-survival fraction",
                "Mean executed controls",
                "Mean complete cost",
            ),
            strict=True,
        ):
            for index, name in enumerate(order):
                row = records.get(name)
                if row and row.get(field) is not None:
                    axis.bar(index, row[field], color=colors.get(row.get("arm"), "#90979c"))
            axis.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
            axis.set(title=panel, ylabel=label)
            if field.endswith("fraction"):
                axis.set_ylim(0, 1.03)
    figure.suptitle(
        "Recorded complete episode outcomes only; missing actors/attempts are not failures"
    )
    _save_figure(figure, out / "standalone_outcomes", plt)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    component_colors = ("#335c81", "#b54040", "#cf8e36", "#a8adb1")
    for axis, panel in zip(axes, ("primary", "confirmation"), strict=True):
        records = {row["point"]: row for row in tables["standalone"] if row["panel"] == panel}
        for index, name in enumerate(order):
            row, bottom = records.get(name), 0.0
            if row is None:
                continue
            for field, color in zip(COST_FIELDS[:-1], component_colors, strict=True):
                value = row.get("mean_" + field)
                if value is not None:
                    axis.bar(
                        index,
                        value,
                        bottom=bottom,
                        color=color,
                        label=field if index == 0 else None,
                    )
                    bottom += value
        axis.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
        axis.set(title=panel, ylabel="Mean complete cost components")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=7)
    figure.suptitle("Absorbing tail and native terminal charges are reported separately")
    _save_figure(figure, out / "cost_components", plt)


def _display(value: Any) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _reference_gap_markdown(tables: Mapping[str, Any]) -> list[str]:
    lines = [
        "### Fresh matched reference–student gaps",
        "",
        "`matched_reference_gaps.csv` compares each reference-data child with the fresh reference on "
        "the same complete 16-reset panel. Differences are child minus reference. It retains walking "
        "and full-survival count gaps, controls, velocity MAE and every complete-cost component. "
        "`matched_reference_gap_status.csv` explains unavailable or misaligned comparisons; none is imputed.",
        "",
        "The reference panel is one shared control across three training pairs, not three independent "
        "teacher replications or three times its recorded work. Reset-snapshot equality is checked by "
        "the independent native audit. Fresh matched MPPI-teacher gaps are unavailable because S1 "
        "collects no new MPPI teacher episodes; historical own-source gaps above remain unpaired.",
        "",
        "| Panel | Child pair | Walking gap | Survival gap | Mean controls gap | Complete-cost gap | Velocity-MAE gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables["matched_reference_gaps"]:
        lines.append(
            "| "
            + " | ".join(
                _display(row.get(key))
                for key in (
                    "panel",
                    "pair",
                    "walking_count_gap",
                    "survival_count_gap",
                    "mean_steps_gap",
                    "mean_cost_gap",
                    "mean_mae_gap",
                )
            )
            + " |"
        )
    if not tables["matched_reference_gaps"]:
        lines.append("No complete aligned fresh reference–child comparison is available.")
    lines.append("")
    return lines


def markdown_report(
    result: Mapping[str, Any], tables: Mapping[str, Any], *, phase: str, portable: bool
) -> str:
    gates = result.get("gates", {})
    interrupted = result.get("schema") == INTERRUPTED_SCHEMA
    observed = len(tables["episodes"])
    lines = [
        "# S1: MPPI-data versus competent-reference-data cloning",
        "",
        f"Artifact: **{phase}**. Artifact execution complete: **{result.get('execution_complete', False)}**. "
        f"Status: **{result.get('status', 'unavailable')}**. Recorded native attempts: **{observed}/320**.",
        f"Error: `{json.dumps(result.get('error'), allow_nan=False)}`.",
        "**Interrupted experiment; this is a scoped verified-prefix report, not a completed S1 comparison.** "
        "The original evaluation result and owner exit remain absent. This separately audited summary "
        "does not replace them. Final source-benefit, competence gates and comparative bootstrap intervals "
        "are withheld; no automatic retry or new intervention follows."
        if interrupted
        else "",
        "",
        "Portable scope: result summaries and exact cached 100-update learning bins. Full per-update fit "
        "JSONs, tensors, predictions, minibatch tapes and native snapshots remain external. Their absence "
        "does not authorize recomputing a fit or filling missing outcomes."
        if portable
        else "Full saved-result scope: the six original per-update fit JSONs are read and hash-checked; their "
        "100-update summaries are exported for portable rendering. No policy or simulator is called.",
        "",
        "## Intervention and limits",
        "",
        "Frozen design: three paired fresh 128-tanh initializations each fit once on 4,000 original MPPI rows and once on "
        "4,000 executed competent-reference rows, at the fixed 2k endpoint. Both use the exact shared W2 "
        "normalization, paired minibatch index tapes and optimizer settings. The source changes reset "
        "contexts, states, gait, torque statistics and controller history jointly. This is a whole-corpus "
        "source comparison; it does not isolate noise, multimodality or trajectory diversity.",
        "The common MPPI-training normalization preserves the current input map; a failure may still "
        "reflect conditioning or the finite fitting budget.",
        "",
        "The reference-data arm uses the reference's own executed behavior, not reference relabels at MPPI "
        "states. Both source arms are fresh fits on the same pinned CUDA backend; historical CPU W2 policies "
        "are context, not matched controls. Native standalone execution uses the pinned CPU simulator. "
        "No new teacher acquisition, trajectory optimization or acquisition–learning feedback occurs.",
        "",
        "## Reconstruction and learning curves",
        "",
        "`fitting.csv` contains every available incoming/final actor on all four fixed splits; "
        "`trajectories.csv` and `actuator_errors.csv` retain source, teacher and actuator detail. Each "
        "source has 4k training and 2k held rows. Held corpora were previously examined development "
        "evidence, excluded from gradients and normalization; they select no endpoint. Original target "
        "dtypes are preserved in evidence, with float32 fitting and float64 diagnostic residual arithmetic.",
        "",
        "`learning_bins.csv` summarizes already-computed minibatch losses in fixed 100-update bins. "
        "A final partial bin contains only recorded updates. Min–max shading is minibatch spread, not "
        "a confidence interval. Full per-update JSON hashes and update counts remain in "
        "`learning_bins_manifest.json`; no extra diagnostic inference is hidden in these curves.",
        f"Recorded curve files: **{len(_fits(result))}/6**; binned recorded updates: "
        f"**{sum(int(row['updates']) for row in tables['learning_bins'])}/12000**.",
        "",
        "## Standalone outcomes and separate decisions",
        "",
        "Both 16-reset panels are required for all ten actors, regardless of the primary result. "
        "`episodes.csv` retains every supplied attempt, including incomplete records. Summaries below use "
        "recorded complete episodes with known outcomes, and disclose their denominators; absent or failed "
        "work is not imputed as a completed failure. A partial fitting or CPU stage remains incomplete.",
        "For the interrupted run, `partial_attempts.csv` retains verified executed steps and observed native "
        "cost separately from unknown terminal walking, survival, velocity error and complete-cost outcomes. "
        "`missing_roster.csv` records all planned roles. Its unstarted/unverified entries describe the "
        "durable record, not a quantified bound on unknown unrecorded work."
        if interrupted
        else "",
        "",
        "| Panel | Actor | Complete / recorded | Walking | Full survival | Mean controls | Velocity MAE | Complete cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables["standalone"]:
        lines.append(
            f"| {row['panel']} | {row['point']} | {row['completed_episodes']}/{row['recorded_attempts']} | "
            + " | ".join(
                _display(row.get(field))
                for field in (
                    "walking_count",
                    "survival_count",
                    "mean_steps",
                    "mean_mae",
                    "mean_cost",
                )
            )
            + " |"
        )
    lines += [
        "",
        "The source-benefit screen requires ≥2 additional walking successes in ≥2 of 3 pairs, with no "
        "walking or full-survival regression in any pair, separately on both panels. Individual provisional "
        "competence requires ≥15/16 walking on each panel. These are separate decisions: a walking gain "
        "need not be competence, and retention without walking is retention improvement. All actor seeds "
        "and the fresh reference control remain visible; one competent seed is not seed-robust fitting.",
        "",
        f"Declared confirmed source walking benefit: **{_display(gates.get('confirmed_walking_benefit'))}**.",
        "",
        "```json",
        json.dumps(gates, indent=2, allow_nan=False),
        "```",
        "",
        "`descriptive_bootstrap.json` resamples three paired training streams and shared reset columns. "
        "It requires complete paired panels; missing outcomes produce an unavailable result, not an imputed "
        "interval. Corpora are fixed, and frames, actors' resets and endpoints are not extra training "
        "replications. All-zero observed walking does not prove equal population success rates.",
        "For this interrupted comparison both final bootstrap panels are unavailable. Known completed "
        "primary observations remain descriptive; incomplete confirmation is not silently substituted "
        "with a smaller experiment."
        if interrupted
        else "",
        "",
        "## Costs, source teachers and causal verdict",
        "",
        "Complete cost = raw native step cost excluding terminal + native terminal + terminal adjustment "
        "+ absorbing missing-controls tail. The native terminal is not charged twice. `cost_components.csv` "
        "and episode rows retain these components separately. `paired_cost_components.csv` gives matched-reset "
        "reference-data-minus-MPPI-data component differences where both recorded panels align; their sum is "
        "the complete-cost difference. Lower complete cost and longer retention "
        "need not independently corroborate recovered gait.",
        "",
        f"Recorded historical source-teacher outcomes: **{len(tables['source_teachers'])}/12**, retained in "
        "`teacher_outcomes.json` and source tables. "
        "`teacher_student_gaps.csv` compares each child's own-source historical teachers against fresh "
        "student episodes. These are descriptive unpaired gaps, not matched-reset causal differences. "
        "Unavailable historical cost components remain unavailable; none is reconstructed from assumptions.",
        "",
        *_reference_gap_markdown(tables),
        "A reference-data clone meeting the provisional competence criterion would establish a competent "
        "learning control under this exact setup; occasional successful episodes remain successes, not competence. "
        "A source advantage identifies the chosen corpus intervention, not its causal property or unique "
        "correct actions. Failure of both arms does not prove class incapacity. Neither outcome closes the "
        "MPPI–policy loop; a later mechanism and a second useful acquisition–learning round would require "
        "their own versioned protocol. Retain negative results and formulate one unlaunched next hypothesis; "
        "no weighting, transport loss, critic, normalization change or follow-up launches automatically.",
        "",
        "`cost_accounting.json` separates saved fit/evaluation phase counters and imported evidence from "
        "report arithmetic. The planned six fits total 12,000 Adam steps and 3,072,000 backward rows; only "
        "actual recorded counters are reported as completed. Reporting performs zero new forward, backward, "
        "optimizer, simulator or teacher-acquisition work.",
        "The interrupted work census is a known durable prefix plus an **unknown unrecorded tail**, "
        "not an exact all-work total or a zero tail. Recovery copies and accepted array hashes belong "
        "to the scoped audit; the original truncated HDF5 bytes remain preserved."
        if interrupted
        else "",
        "```json\n" + json.dumps(result.get("work", {}), indent=2, allow_nan=False) + "\n```"
        if interrupted
        else "",
        "",
        "![Four-split reconstruction](reconstruction.png)",
        "![Learning curves](learning_curves.png)",
        "![Standalone outcomes](standalone_outcomes.png)",
        "![Complete cost components](cost_components.png)",
        "",
    ]
    return "\n".join(lines)


def _forensic_markdown(result: Mapping[str, Any], tables: Mapping[str, Any], portable: bool) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in tables["missing_roster"]:
        counts[str(row["status"])] += 1
    lines = [
        "# S1 forensic display: interrupted experiment, failed native-evidence audit",
        "",
        "**The scientific integrity audit remains FAILED. This explicit forensic display does not accept "
        "the run, complete its comparison or change the failed summary/receipt.** The original evaluation "
        "result and owner exit remain absent. No retry or later learning intervention is launched.",
        "",
        f"Individually verified native outcomes: **{len(tables['verified_native_subset'])}**, a nonrepresentative "
        f"retained subset. Unverified ledger episode claims: **{len(tables['unverified_ledger_episode_claims'])}**. "
        f"Partial attempts: **{len(tables['partial_attempts'])}**. Worker actor-panel claims: "
        f"**{len(tables['worker_actor_summary_claims'])}**.",
        "",
        "## Closed fitting remains available",
        "",
        "All six original fixed-budget source fits, their ownership receipts, four-split reconstruction "
        "metrics and loss curves remain separate from the native HDF5 loss. Both arms use matched fresh "
        "128-tanh initializations, common W2 normalization, paired indices and exactly 2,000 updates. "
        "Changing the demonstrated corpus jointly changes states, gaits and labels; lower reconstruction "
        "is not evidence of walking or a property such as multimodality being isolated.",
        "",
        "`fitting.csv`, `trajectories.csv`, `actuator_errors.csv`, `reconstruction_control_bins.csv` and "
        "`learning_bins.csv` retain these fitting measurements. CUDA batched reconstruction and CPU "
        "batch-one deployment differ as frozen; no extra CPU reconstruction forwards were performed. "
        "Full per-update curve hashes remain bound in `learning_bins_manifest.json`.",
        "",
        "Portable mode uses exact hash-bound 100-update bins; full curves and tensors remain external."
        if portable
        else "This render reads the original hash-bound fit curves; it performs no new fitting or inference.",
        "",
        "## Native evidence classes remain separate",
        "",
        "| Durable roster status | Planned roles |",
        "|---|---:|",
    ]
    lines.extend(f"| {status} | {count} |" for status, count in sorted(counts.items()))
    lines += [
        "",
        "`verified_native_subset.csv` and `episodes.csv` contain only individually verified complete "
        "episode outcomes. Their missingness is not random or balanced; they are not a smaller substitute "
        "experiment. No actor/source rate, native cost ranking, teacher–student gap, competence gate or "
        "comparative interval is issued from that subset. Native plots display availability and a "
        "withheld-comparison notice, not performance bars.",
        "",
        "`ledger_episode_claims.csv` preserves documentary worker claims with raw-verification status. "
        "`unverified_ledger_episode_claims.csv` isolates claims lacking full native-array verification. "
        "`worker_actor_summary_claims.csv` preserves the original worker panel summaries separately. "
        "Their ledger source SHA and byte size match the independently preserved original file catalog. "
        "A ledger claim is not promoted to an independently verified native outcome.",
        "",
        "`partial_attempts.csv` retains verified steps and observed native cost separately from ledger-"
        "recorded boundaries. Final walking, survival and complete cost remain unknown: no post-interruption "
        "fall, absorbing penalty or completion is invented. `missing_roster.csv` preserves all 320 roles; "
        "unstarted roles describe the durable record while an unknown unrecorded tail remains unresolved.",
        "",
        "Both final gates and bootstrap panels are withheld. Historical source-teacher outcomes remain "
        "available, but fresh native gaps are withheld, including comparisons using the shared reference. "
        "Documentary success counts must not be presented as fully verified panel rates.",
        "",
        "## Preservation and accounting",
        "",
        "Original files, the failed audit and the separately labelled recovery copy remain preserved. "
        "Padding a recovery copy does not recover missing information. This renderer does not read or "
        "repair HDF5; its inputs are the unchanged closed-fit result, failed postmortem summary and "
        "receipt, source-bound ledger claims and preservation catalog. All are hash-bound in the report.",
        "",
        "`cost_accounting.json` retains exact closed-fit counters and the known durable native ledger "
        "prefix separately from the **unknown unrecorded tail**. Final wall and snapshot totals may be "
        "null; a lower bound is not an exact total. Reporting itself performs zero new policy inference, "
        "training, physics or teacher acquisition.",
        "",
        "```json",
        json.dumps(result.get("work", {}), indent=2, allow_nan=False),
        "```",
        "",
        "![Valid fitting reconstruction](reconstruction.png)",
        "![Valid recorded learning curves](learning_curves.png)",
        "![Native data availability only](standalone_outcomes.png)",
        "![Native comparisons withheld](cost_components.png)",
        "",
    ]
    return "\n".join(lines)


def render(
    run: Path,
    out: Path,
    *,
    interruption_summary: Path | None = None,
    display_failed_audit: bool = False,
    ledger_claims: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    run, out = run.resolve(), out.resolve()
    requested, capsule = run, None
    if not (run / "config.json").is_file() and (run / "run/config.json").is_file():
        capsule, run = run, run / "run"
    elif run.name == "run" and (run.parent / "capsule.json").is_file():
        capsule = run.parent
    if interruption_summary is None and (run / "interruption_summary.json").is_file():
        interruption_summary = run / "interruption_summary.json"
    interruption_audit_sha256 = None
    audit_passed, forensic, claims, catalog_path = None, False, None, None
    ledger_verified, source_ledger_verified = False, False
    if interruption_summary is not None:
        result_file, result_path = "interruption_summary.json", interruption_summary.resolve()
        result, interruption_audit_sha256, audit_passed = _load_interruption_summary(
            run, result_path, display_failed_audit=display_failed_audit
        )
        forensic = audit_passed is False
        if forensic:
            if ledger_claims is None:
                ledger_claims = run / "ledger_claims.json"
            ledger_claims = ledger_claims.resolve()
            claims, catalog_path, ledger_verified = _load_ledger_claims(run, ledger_claims)
    else:
        result_file = "result.json" if (run / "result.json").is_file() else "fit_result.json"
        result_path = run / result_file
        result = json.loads(result_path.read_text())
        if display_failed_audit:
            raise ValueError("Failed-audit display requires an interrupted-run summary")
    config = json.loads((run / "config.json").read_text())
    seal = json.loads((run / "input_manifest.json").read_text())
    if result.get("schema") not in RESULT_SCHEMAS | {INTERRUPTED_SCHEMA}:
        raise ValueError("Unexpected S1 result schema")
    if sha(run / "config.json") != seal.get("config_sha256") or result.get(
        "config_sha256", sha(run / "config.json")
    ) != sha(run / "config.json"):
        raise ValueError("S1 configuration binding differs")
    if result.get("input_manifest_sha256", sha(run / "input_manifest.json")) != sha(
        run / "input_manifest.json"
    ):
        raise ValueError("S1 input-manifest binding differs")
    portable = capsule is not None or any(
        not (run / fit["file"]).is_file() for fit in _fits(result)
    )
    if portable:
        candidates = [
            requested / "report/learning_bins.csv",
            run / "report/learning_bins.csv",
            run.parent / "report/learning_bins.csv",
        ]
        cache = next((path for path in candidates if path.is_file()), candidates[0])
        bins, bindings = _portable_bins(
            run,
            result_file,
            result,
            cache,
            capsule,
            result_path,
            ledger_claims if forensic else None,
            catalog_path,
        )
        if forensic:
            source_receipt = json.loads((cache.parent / "receipt.json").read_text())
            source_ledger_verified = bool(
                source_receipt.get("verified_against_original_ledger")
                or source_receipt.get("source_report_verified_against_original_ledger")
            )
    else:
        if forensic and not ledger_verified:
            raise ValueError(
                "Full forensic display requires original ledger row verification or a canonical portable cache"
            )
        bins, bindings = learning_bins(run, result)
    out.mkdir(parents=True, exist_ok=False)
    tables = build_tables(result, config)
    if forensic:
        _forensic_tables(tables, result, claims)
    tables["learning_bins"] = bins
    for name, rows in tables.items():
        columns = (
            ("point", "pair", "arm", "start_fit_update", "stop_fit_update", "updates")
            if name == "learning_bins"
            else ()
        )
        _write_csv(out / f"{name}.csv", rows, columns)
    curve_manifest = dict(
        schema=CURVE_SCHEMA,
        bin_updates=100,
        result_file=result_file,
        result_sha256=sha(result_path),
        config_sha256=sha(run / "config.json"),
        input_manifest_sha256=sha(run / "input_manifest.json"),
        source_curves=bindings,
        binned_updates=sum(int(row["updates"]) for row in bins),
        learning_bins_sha256=sha(out / "learning_bins.csv"),
        scope="Fixed summaries of already-computed losses; full per-update JSONs remain external. This cache cannot replay optimization.",
    )
    if interruption_summary is not None:
        curve_manifest.update(
            interruption_summary_sha256=sha(result_path),
            fit_result_sha256=sha(run / "fit_result.json"),
            interruption_audit_receipt_sha256=interruption_audit_sha256,
        )
    if forensic:
        curve_manifest.update(
            failed_audit_display=True,
            ledger_claims_sha256=sha(ledger_claims),
            original_file_catalog_sha256=sha(catalog_path),
        )
    write(out / "learning_bins_manifest.json", curve_manifest)
    write(out / "teacher_outcomes.json", result.get("teacher_outcomes", {}))
    write(out / "gates.json", result.get("gates", {}))
    write(out / "descriptive_bootstrap.json", paired_bootstrap(result, config))
    (out / "report.md").write_text(
        _forensic_markdown(result, tables, portable)
        if forensic
        else markdown_report(result, tables, phase=result_file, portable=portable)
    )
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="walker-s1-report-"))
    import matplotlib

    matplotlib.use("Agg")
    _figures(tables, config, out, forensic=forensic)
    write(
        out / "cost_accounting.json",
        dict(
            phase_work=result.get("phase_work", {}),
            recorded_work=result.get("work", {}),
            platform=result.get("platform", {}),
            planned_budgets=config.get("budgets", {}),
            counter_scope="Combined recorded work already includes each phase once; phase tables explain it and must not be added again.",
            worker_wall_seconds=result.get("wall_seconds"),
            artifact_phase=result_file,
            failed_audit_display=forensic,
            audit_passed=audit_passed,
            comparison_complete=result.get("comparison_complete", result.get("execution_complete")),
            scoped_interruption_audit=result.get("integrity_status"),
            partial_attempts=len(result.get("partial_attempts", [])),
            unknown_unrecorded_tail=result.get("work", {}).get("unknown_unrecorded_tail"),
            interruption=result.get("interruption_evidence", result.get("interruption")),
            recovery=result.get("recovery"),
            durable_prefix=result.get("durable_prefix"),
            observed_wall_seconds_lower_bound=result.get("observed_wall_seconds_lower_bound"),
            recorded_native_attempts=len(claims["episode_complete"])
            + len(tables["partial_attempts"])
            if forensic
            else len(tables["episodes"]),
            verified_native_outcomes=len(tables["verified_native_subset"]) if forensic else None,
            unverified_ledger_outcome_claims=len(tables["unverified_ledger_episode_claims"])
            if forensic
            else None,
            original_curve_files=len(bindings),
            binned_updates=sum(int(row["updates"]) for row in bins),
            render_scope="portable_learning_bins" if portable else "full_saved_curves",
            report_new_forward_rows=0,
            report_backward_rows=0,
            report_optimizer_steps=0,
            report_native_controls=0,
            report_new_teacher_episodes=0,
            report_wall_seconds=time.perf_counter() - started,
            imported_evidence_scope="Historical teacher experience and reference training are not newly charged S1 work; their unavailable original overhead remains unknown, not zero.",
        ),
    )
    receipt = dict(
        schema=REPORT_SCHEMA,
        run=str(run),
        result_file=result_file,
        result_sha256=sha(result_path),
        interruption_summary_sha256=sha(result_path) if interruption_summary is not None else None,
        fit_result_sha256=sha(run / "fit_result.json")
        if interruption_summary is not None
        else result.get("fit_result_sha256"),
        interruption_audit_receipt_sha256=interruption_audit_sha256,
        failed_audit_display=forensic,
        audit_passed=audit_passed,
        native_comparison_withheld=forensic,
        verified_against_original_ledger=ledger_verified,
        source_report_verified_against_original_ledger=source_ledger_verified,
        ledger_claims_sha256=sha(ledger_claims) if forensic else None,
        original_file_catalog_sha256=sha(catalog_path) if forensic else None,
        config_sha256=sha(run / "config.json"),
        input_manifest_sha256=sha(run / "input_manifest.json"),
        source_sha256=sha(Path(__file__)),
        render_scope="portable_learning_bins" if portable else "full_saved_curves",
        scientific_recipe_changed=False,
        outputs={
            path.name: dict(sha256=sha(path), bytes=path.stat().st_size)
            for path in sorted(out.iterdir())
            if path.is_file()
        },
        new_forward_rows=0,
        backward_rows=0,
        optimizer_steps=0,
        new_controls=0,
        wall_seconds=time.perf_counter() - started,
    )
    write(out / "receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--interruption-summary",
        type=Path,
        help="Use a separately audited interrupted-run summary; never writes an original result or owner exit",
    )
    parser.add_argument(
        "--display-failed-audit",
        action="store_true",
        help="Explicit forensic display of retained verified subsets without accepting a failed integrity audit",
    )
    parser.add_argument(
        "--ledger-claims",
        type=Path,
        help="Source-bound surviving ledger claims for the explicit failed-audit display",
    )
    args = parser.parse_args(argv)
    receipt = render(
        args.run,
        args.out,
        interruption_summary=args.interruption_summary,
        display_failed_audit=args.display_failed_audit,
        ledger_claims=args.ledger_claims,
    )
    print(json.dumps(dict(report=str(args.out.resolve()), outputs=len(receipt["outputs"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
