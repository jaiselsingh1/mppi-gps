"""Independent, saved-JSON/CSV-only audit of the K1.1 temporal review.

No production metric, worker, report, policy, checkpoint or simulator code is
imported. Saved squared-error bin means can establish aggregation identities;
they cannot reconstruct predictions or prove causal action consequences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_HASH = "e1ec300c39d6880f893995012fa7c5a0789f9ef0b931280454310b57e62c9cba"
ATOL = RTOL = 1e-12
CAPSULE_KEYS = {
    "config": "run/config.json",
    "result": "run/result.json",
    "matrix": "report/trajectory_policy_matrix.csv",
    "curves": "report/source_control_curves.csv",
    "actuators": "report/actuator_errors.csv",
    "saturation": "report/saturation.csv",
    "groups": "report/group_means.csv",
}
SOURCE_COUNTS = {
    "original_discovery": 4,
    "later_independent": 2,
    "later_guided": 2,
    "h4_independent": 2,
    "h4_fixed_high": 2,
    "h4_backoff": 2,
    "qualified_reference": 16,
    "executed_random_actions": 4,
}
ZERO_WORK = (
    "new_policy_forward_rows",
    "new_physics_controls",
    "new_optimizer_steps",
    "new_mppi_controls",
    "raw_array_reads",
    "checkpoint_loads",
    "gpu_operations",
)


class ReviewAudit:
    """Small fail-fast proof collector with immutable file bindings."""

    def __init__(self) -> None:
        self.checks = 0
        self.files: dict[str, dict[str, Any]] = {}

    def need(self, condition: Any, message: str) -> None:
        self.checks += 1
        if not condition:
            raise AssertionError(message)

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        first, second = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.need(
            first.shape == second.shape
            and np.isfinite(first).all()
            and np.isfinite(second).all()
            and np.allclose(first, second, atol=ATOL, rtol=RTOL),
            message,
        )

    def prove(self, actual: Any, expected: Any, message: str) -> None:
        """Check every recomputed field, allowing unrelated envelope metadata."""
        if isinstance(expected, dict):
            self.need(isinstance(actual, dict), message + " object")
            for key, value in expected.items():
                self.need(key in actual, message + " missing field " + key)
                self.prove(actual[key], value, message + "/" + key)
        elif isinstance(expected, (list, tuple)):
            self.need(isinstance(actual, (list, tuple)), message + " sequence")
            self.need(len(actual) == len(expected), message + " length")
            for index, (first, second) in enumerate(zip(actual, expected, strict=True)):
                self.prove(first, second, message + f"/{index}")
        elif isinstance(expected, float):
            self.equal(actual, expected, message)
        else:
            self.need(type(actual) is type(expected) and actual == expected, message)

    def bind(self, path: Path, expected_sha: str | None = None) -> dict[str, Any]:
        self.need(path.is_file() and not path.is_symlink(), "Regular immutable file " + str(path))
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        self.need(
            (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "File unchanged while hashing " + str(path),
        )
        binding = dict(sha256=digest.hexdigest(), bytes=before.st_size)
        if expected_sha is not None:
            self.need(binding["sha256"] == expected_sha, "Exact file SHA-256 " + str(path))
        key = str(path.resolve())
        if key in self.files:
            self.need(self.files[key] == binding, "Bound file unchanged " + str(path))
        self.files[key] = binding
        return binding

    def load(self, path: Path) -> Any:
        self.bind(path)
        with path.open() as stream:
            return json.load(stream)


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def variance(values: list[float]) -> float:
    center = mean(values)
    return mean([(value - center) ** 2 for value in values])


def ancestry(value: Any) -> str:
    return str(value["reset_cluster"] if isinstance(value, dict) else value)


def contained(a: ReviewAudit, root: Path, name: str) -> Path:
    candidate = root / name
    a.need(not Path(name).is_absolute(), "Relative sealed path")
    a.need(candidate.resolve().is_relative_to(root.resolve()), "Contained sealed path")
    # Reject symlink parents as well as the final file.
    a.need(not any(part.is_symlink() for part in [candidate, *candidate.parents]), "No symlinks")
    return candidate


def verify_metric(a: ReviewAudit, metric: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """Reconcile saved scalar, actuator and prefix reductions without raw arrays."""
    a.need(type(n) is int and n > 0 and metric["rows"] == n, "Executed metric row count")
    a.equal(metric["saturation_threshold"], 0.95, "Unchanged marginal saturation threshold")
    per_actuator = metric["per_actuator_mse"]
    a.need(len(per_actuator) == 6, "Six actuator error coordinates")
    a.need(all(math.isfinite(x) and 0 <= x <= 4 for x in per_actuator), "Bounded MSE")
    a.equal(mean(per_actuator), metric["mse"], "Full scalar/six-actuator denominator")
    bins, cumulative, seen = [], 0.0, 0
    expected_starts = list(range(0, n, 100))
    a.need(len(metric["bins"]) == len(expected_starts), "Complete bin census without padding")
    for start, source in zip(expected_starts, metric["bins"], strict=True):
        stop = min(start + 100, n)
        a.prove(
            source,
            dict(start_control=start, stop_control_exclusive=stop, rows=stop - start),
            "Ordered contiguous executed bin",
        )
        a.need(len(source["per_actuator_mse"]) == 6, "Six bin coordinates")
        a.need(
            all(math.isfinite(x) and 0 <= x <= 4 for x in source["per_actuator_mse"]),
            "Bounded bin MSE",
        )
        a.equal(mean(source["per_actuator_mse"]), source["mse"], "Bin scalar/actuator identity")
        cumulative += source["rows"] * source["mse"]
        seen += source["rows"]
        bins.append(
            dict(
                start_control=start,
                stop_control_exclusive=stop,
                rows=stop - start,
                E=source["mse"],
                per_actuator_mse=source["per_actuator_mse"],
                cumulative_E=cumulative / seen,
            )
        )
    a.equal(cumulative / n, metric["mse"], "Weighted bins reproduce complete E")
    for coordinate in range(6):
        weighted = math.fsum(row["rows"] * row["per_actuator_mse"][coordinate] for row in bins) / n
        a.equal(weighted, per_actuator[coordinate], "Weighted bins reproduce actuator MSE")
    first = metric["early"]
    a.prove(
        first,
        dict(requested_controls=100, rows=min(n, 100)),
        "Original first-100 convention",
    )
    a.equal(first["mse"], bins[0]["E"], "First bin/early scalar identity")
    a.equal(first["per_actuator_mse"], bins[0]["per_actuator_mse"], "First bin/early vector")
    for owner in ("actions", "predictions"):
        summary = metric[owner]
        for key in ("mean_abs", "rms", "saturation_fraction"):
            a.need(math.isfinite(summary[key]) and 0 <= summary[key] <= 1, "Marginal bound")
        for key in (
            "mean_per_actuator",
            "mean_abs_per_actuator",
            "saturation_fraction_per_actuator",
        ):
            a.need(len(summary[key]) == 6, "Six marginal coordinates")
            lower = -1 if key == "mean_per_actuator" else 0
            a.need(
                all(math.isfinite(x) and lower <= x <= 1 for x in summary[key]),
                "Marginal vector bound",
            )
        a.equal(
            mean(summary["mean_abs_per_actuator"]),
            summary["mean_abs"],
            "Marginal absolute-mean reduction",
        )
        a.equal(
            mean(summary["saturation_fraction_per_actuator"]),
            summary["saturation_fraction"],
            "Marginal saturation reduction",
        )
    return bins


def derive_trajectory(
    a: ReviewAudit, record: dict[str, Any], stage: str, index: int, outcome: dict[str, Any]
) -> dict[str, Any]:
    source = record["trajectory"]
    n, metric = source["steps"], record["policies"]["reference"]
    bins = verify_metric(a, metric, n)
    raw = record["raw"]["arrays"]
    for key, shape in (
        ("predictions", [n, 9, 6]),
        ("executed_actions", [n, 6]),
        ("observations", [n, 17]),
    ):
        a.prove(raw[key], dict(shape=shape, encoding="array"), "Original raw descriptor " + key)
        a.need(len(raw[key]["sha256"]) == 64, "Saved raw array SHA-256 metadata")
    e = metric["mse"]
    if source["group"] == "qualified_reference":
        a.need(
            e == 0 and all(x == 0 for x in metric["per_actuator_mse"]),
            "Exact self-match scalar/vector",
        )
        a.need(all(row["E"] == 0 for row in bins), "Exact self-match bins")
    return dict(
        index=index,
        stage=stage,
        source=source["group"],
        id=source["id"],
        ancestry=ancestry(source["ancestry"]),
        N=n,
        E=e,
        D=6 * n * e,
        outcome=outcome,
        per_actuator_mse=metric["per_actuator_mse"],
        actuator_error_shares=([x / (6 * e) for x in metric["per_actuator_mse"]] if e else None),
        early=metric["early"],
        bins=bins,
        bin_sd=math.sqrt(variance([row["E"] for row in bins])),
        full_horizon=n == 1000 and len(bins) == 10,
        marginals={key: metric[key] for key in ("actions", "predictions")},
        saturation_threshold=metric["saturation_threshold"],
    )


def derive_group(rows: list[dict[str, Any]], source: str, stage: str) -> dict[str, Any]:
    """Use explicit population covariance sums, independent of the core reducer."""
    scores = [row["E"] for row in rows]
    balanced = all(row["full_horizon"] for row in rows)
    decomposition = difference = None
    if balanced:
        b = len(rows[0]["bins"])
        columns = [[row["bins"][j]["E"] for row in rows] for j in range(b)]
        centers = [mean(column) for column in columns]
        covariance = [
            [
                math.fsum(
                    (x - centers[j]) * (y - centers[k])
                    for x, y in zip(columns[j], columns[k], strict=True)
                )
                / len(rows)
                for k in range(b)
            ]
            for j in range(b)
        ]
        diag = math.fsum(covariance[j][j] for j in range(b)) / b**2
        off = math.fsum(covariance[j][k] for j in range(b) for k in range(b) if j != k) / b**2
        full = variance(scores)
        decomposition = dict(
            B=b,
            var_between_full=full,
            mean_between_bin_var=mean([variance(column) for column in columns]),
            covariance_diag_contribution=diag,
            covariance_offdiag_contribution=off,
            covariance_matrix=covariance,
            identity_residual=diag + off - full,
        )
        if len(rows) == 2:
            difference = dict(
                first=rows[0]["id"],
                second=rows[1]["id"],
                full_delta=scores[0] - scores[1],
                bin_deltas=[
                    dict(
                        start_control=first["start_control"],
                        stop_control_exclusive=first["stop_control_exclusive"],
                        delta_E=first["E"] - second["E"],
                    )
                    for first, second in zip(rows[0]["bins"], rows[1]["bins"], strict=True)
                ],
            )
    return dict(
        source=source,
        stage=stage,
        descriptive_mixture=source == "pooled_MPPI_14",
        trajectory_ids=[row["id"] for row in rows],
        N_values=[row["N"] for row in rows],
        count=len(rows),
        E_mean=mean(scores),
        E_sd=math.sqrt(variance(scores)),
        E_min=min(scores),
        E_max=max(scores),
        E_spread=max(scores) - min(scores),
        balanced=balanced,
        decomposition=decomposition,
        two_trajectory_difference=difference,
    )


def selected_csv(a: ReviewAudit, path: Path) -> list[dict[str, str]]:
    a.bind(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row for row in rows if row["policy"] == "reference" and row["label_condition"] == "recorded"
    ]


def audit_csv_joins(
    a: ReviewAudit,
    inputs: dict[str, Path],
    trajectories: list[dict[str, Any]],
    original: dict[str, Any],
) -> None:
    """Prove exact ownership/counts and saved numerical joins across all five CSVs."""
    matrices = selected_csv(a, inputs["matrix"])
    actuators = selected_csv(a, inputs["actuators"])
    saturation = selected_csv(a, inputs["saturation"])
    curves = selected_csv(a, inputs["curves"])
    groups = selected_csv(a, inputs["groups"])
    a.need(len(matrices) == len(trajectories), "Exactly one matrix row per trajectory")
    a.need(len(actuators) == 6 * len(trajectories), "Exactly six actuator CSV rows per trajectory")
    a.need(len(saturation) == 2 * len(trajectories), "Exactly two marginal CSV rows per trajectory")
    for index, (expected, row) in enumerate(zip(trajectories, matrices, strict=True)):
        for key, value in (
            ("stage", expected["stage"]),
            ("group", expected["source"]),
            ("trajectory", expected["id"]),
            ("ancestry", expected["ancestry"]),
        ):
            a.need(row[key] == value, "Matrix ownership/order " + key)
        a.need(int(row["rows"]) == expected["N"], "Matrix actual length")
        a.equal(float(row["mse"]), expected["E"], "Matrix E join")
        a.equal(float(row["early_mse"]), expected["early"]["mse"], "Matrix early E join")
        a.need(int(row["early_rows"]) == expected["early"]["rows"], "Matrix early length")
        for coordinate, detail in enumerate(actuators[6 * index : 6 * index + 6]):
            for key in ("stage", "group", "trajectory", "ancestry", "rows"):
                a.need(detail[key] == row[key], "Actuator CSV exact owner")
            a.need(int(detail["actuator"]) == coordinate, "Actuator coordinate order")
            a.equal(
                float(detail["mse"]),
                expected["per_actuator_mse"][coordinate],
                "Actuator full error join",
            )
            a.equal(
                float(detail["early_mse"]),
                expected["early"]["per_actuator_mse"][coordinate],
                "Actuator early error join",
            )
        owners = set()
        for detail in saturation[2 * index : 2 * index + 2]:
            for key in ("stage", "group", "trajectory", "ancestry", "rows"):
                a.need(detail[key] == row[key], "Marginal CSV exact owner")
            owner = detail["owner"]
            a.need(
                owner in ("actions", "predictions") and owner not in owners,
                "Marginal CSV unique owner",
            )
            owners.add(owner)
            for key in ("mean_abs", "rms", "saturation_fraction"):
                a.equal(
                    float(detail[key]),
                    expected["marginals"][owner][key],
                    "Marginal CSV join " + key,
                )
            a.equal(float(detail["saturation_threshold"]), 0.95, "Marginal CSV threshold")
    group_keys = list(dict.fromkeys((row["stage"], row["source"]) for row in trajectories))
    a.need(len(groups) == len(group_keys), "Complete original source group CSV census")
    expected_curves = []
    for (stage, source), csv_group in zip(group_keys, groups, strict=True):
        rows = [row for row in trajectories if (row["stage"], row["source"]) == (stage, source)]
        distinct = len({row["ancestry"] for row in rows})
        for key, value in (("stage", stage), ("group", source)):
            a.need(csv_group[key] == value, "Source group CSV order")
        a.need(
            int(csv_group["trajectories"]) == len(rows)
            and int(csv_group["ancestries"]) == distinct
            and int(csv_group["rows"]) == sum(row["N"] for row in rows),
            "Source group CSV counts",
        )
        a.equal(
            float(csv_group["equal_trajectory_mean"]),
            mean([row["E"] for row in rows]),
            "Source equal trajectory mean",
        )
        weighted = math.fsum(row["N"] * row["E"] for row in rows) / sum(row["N"] for row in rows)
        a.equal(float(csv_group["sample_weighted_mean"]), weighted, "Source sample-weighted mean")
        saved = original["stages"][stage]["groups"][source]
        for key, field, endpoint in (
            ("equal_low", "equal_trajectory_interval", 0),
            ("equal_high", "equal_trajectory_interval", 1),
            ("weighted_low", "sample_weighted_interval", 0),
            ("weighted_high", "sample_weighted_interval", 1),
        ):
            a.equal(
                float(csv_group[key]), saved[field][endpoint][8], "Unchanged saved source interval"
            )
        for start in sorted({part["start_control"] for row in rows for part in row["bins"]}):
            parts = [
                (row, part)
                for row in rows
                for part in row["bins"]
                if part["start_control"] == start
            ]
            expected_curves.append(
                dict(
                    stage=stage,
                    group=source,
                    start_control=start,
                    # The historical aggregate CSV uses a nominal bin window.
                    # Individual random bins retain their actual executed ends.
                    stop_control_exclusive=start + 100,
                    trajectories=len(parts),
                    ancestries=len({row["ancestry"] for row, _ in parts}),
                    rows=sum(part["rows"] for _, part in parts),
                    equal_trajectory_mean=mean([part["E"] for _, part in parts]),
                    sample_weighted_mean=math.fsum(part["rows"] * part["E"] for _, part in parts)
                    / sum(part["rows"] for _, part in parts),
                )
            )
    a.need(len(curves) == len(expected_curves), "Complete source-bin CSV census")
    for row, expected in zip(curves, expected_curves, strict=True):
        for key, value in expected.items():
            if isinstance(value, str):
                a.need(row[key] == value, "Source-bin CSV owner " + key)
            elif isinstance(value, int):
                a.need(int(row[key]) == value, "Source-bin CSV count " + key)
            else:
                a.equal(float(row[key]), value, "Source-bin CSV scalar " + key)


def verify_reference_qualification(a: ReviewAudit, stage_b: dict[str, Any]) -> None:
    """Honor K1's asymmetric saved primary and confirmation summary layouts."""
    primary, confirmation = stage_b["primary"], stage_b["confirmation"]
    a.need(
        isinstance(primary, dict) and isinstance(primary.get("reference"), dict),
        "K1 primary stores policy-to-summary mapping with a reference entry",
    )
    a.need(
        isinstance(confirmation, dict)
        and "reference" not in confirmation
        and "episodes" in confirmation,
        "K1 confirmation stores the reference summary directly",
    )
    for cohort, summary in (("primary", primary["reference"]), ("confirmation", confirmation)):
        a.prove(
            summary,
            dict(episodes=16, successes=16, survived=16, steps=[1000] * 16),
            "Competent frozen reference " + cohort,
        )


def audit_statistics(
    a: ReviewAudit, inputs: dict[str, Path], config: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Independently derive the complete fixed-reference roster and statistics."""
    original = a.load(inputs["result"])
    old_config = a.load(inputs["config"])
    spread = a.load(inputs["spread"])
    a.need(
        config["reference_policy"] == "reference"
        and config["reference_policy_hash"] == REFERENCE_HASH,
        "Frozen single reference actor",
    )
    a.need(
        old_config["policy_order"] == spread["fixed_reference"]["policy_order"]
        and old_config["policy_order"][8] == "reference"
        and len(old_config["policy_order"]) == 9,
        "Original nine-policy roster and fixed column eight",
    )
    a.need(
        old_config["policies"]["reference"]["policy_hash"]
        == original["policy_hashes"]["reference"]
        == REFERENCE_HASH
        == spread["fixed_reference"]["policy_hash"],
        "Reference actor hash identity",
    )
    a.prove(
        old_config["metrics"],
        dict(early_controls=100, bin_controls=100, saturation_threshold=0.95),
        "Unchanged original metric recipe",
    )
    a.need(
        original["execution_complete"] is True
        and original["error"] is None
        and original["stages"]["B"]["reference_qualified"] is True,
        "Completed qualified original K1 evidence",
    )
    verify_reference_qualification(a, original["stages"]["B"])
    fixed = dict(spread["fixed_reference"], policy="reference")
    a.prove(result["reference"], fixed, "Unchanged fixed reference metadata")
    a.need(
        fixed["qualified"] is True and fixed["reference_column"] == 8,
        "Fixed qualified reference column",
    )
    old_reference = old_config["policies"]["reference"]
    interface = old_reference["audit"]["interface"]
    a.need(
        old_reference["file_sha256"]
        == old_config["reference_checkpoint_sha256"]
        == config["reference_checkpoint_sha256"]
        == fixed["checkpoint_sha256"],
        "Exact inherited reference checkpoint file identity",
    )
    a.prove(
        fixed["normalization"], old_reference["normalization"], "Original reference normalization"
    )
    for key in ("action_scale", "action_bias", "execution"):
        a.prove(fixed[key], interface[key], "Original reference execution interface " + key)
    a.need(
        old_config["trajectory_roster"]
        == [row["trajectory"] for row in original["stages"]["A"]["trajectories"]],
        "Original fourteen-trajectory roster",
    )
    episodes = original["stages"]["B"]["episodes"]
    tags = [row["tag"] for row in episodes]
    a.need(len(set(tags)) == len(tags), "Unique original episode identities")
    episodes_by_tag = {row["tag"]: row for row in episodes}
    expected = []
    for stage in ("A", "B"):
        for record in original["stages"][stage]["trajectories"]:
            source = record["trajectory"]
            a.need(
                list(record["policies"]) == old_config["policy_order"],
                "Exact original policy column order",
            )
            prior = spread["trajectories_in_original_order"][len(expected)]
            saved = source.get("outcome", episodes_by_tag.get(source["id"]))
            outcome = prior["outcome"]
            a.prove(saved, outcome, "Previously published outcome metadata/source join")
            a.prove(
                prior,
                dict(
                    stage=stage,
                    roster_index=len(expected),
                    trajectory=source["id"],
                    source=source["group"],
                    executed_rows_N=source["steps"],
                    E_saved_mse=record["policies"]["reference"]["mse"],
                    prediction_raw=record["raw"],
                    saved_control_bins=record["policies"]["reference"]["bins"],
                    first100_saved=record["policies"]["reference"]["early"],
                ),
                "Existing fixed-reference summary row/source join",
            )
            expected.append(derive_trajectory(a, record, stage, len(expected), outcome))
    a.need(
        Counter(row["source"] for row in expected) == SOURCE_COUNTS, "Original eight-source counts"
    )
    a.need(len({row["id"] for row in expected}) == 34, "No duplicate or selected trajectories")
    a.need(
        [row["N"] for row in expected[-4:]] == [8, 26, 22, 14],
        "All four actual short random lengths",
    )
    a.need(
        all(
            row["N"] == 1000
            and row["outcome"]["complete"]
            and row["outcome"]["success"]
            and row["outcome"]["survived"]
            for row in expected[:-4]
        ),
        "Fourteen complete walking MPPI and sixteen self trajectories",
    )
    a.need(
        all(
            row["outcome"]["complete"]
            and not row["outcome"]["success"]
            and not row["outcome"]["survived"]
            for row in expected[-4:]
        ),
        "Recorded random falls, no fabricated tail states",
    )
    a.prove(result["trajectories"], expected, "Independent trajectory reconstruction")
    groups = []
    for source in SOURCE_COUNTS:
        rows = [row for row in expected if row["source"] == source]
        groups.append(derive_group(rows, source, rows[0]["stage"]))
    groups.append(derive_group(expected[:14], "pooled_MPPI_14", "A"))
    a.prove(result["groups"], groups, "Independent population covariance and source summaries")
    for group in groups:
        decomposition = group["decomposition"]
        if decomposition is not None:
            a.equal(
                decomposition["var_between_full"],
                decomposition["covariance_diag_contribution"]
                + decomposition["covariance_offdiag_contribution"],
                "Covariance diagonal/off-diagonal full variance identity",
            )
            if group["two_trajectory_difference"] is not None:
                difference = group["two_trajectory_difference"]
                a.equal(
                    difference["full_delta"],
                    mean([part["delta_E"] for part in difference["bin_deltas"]]),
                    "Signed two-trajectory temporal cancellation identity",
                )
    audit_csv_joins(a, inputs, expected, original)
    totals = dict(
        trajectories=len(expected),
        rows=sum(row["N"] for row in expected),
        bins=sum(len(row["bins"]) for row in expected),
        actuator_entries=6 * len(expected),
    )
    a.prove(config["expected"], totals, "Frozen complete evidence counts")
    a.prove(result["totals"], totals, "Published complete evidence counts")
    a.prove(
        result["analysis_counts"],
        totals
        | dict(
            matrix_joins=34, actuator_joins=204, saturation_joins=68, curve_joins=71, group_joins=8
        ),
        "Complete original CSV join census",
    )
    return dict(
        totals=totals,
        source_counts=dict(SOURCE_COUNTS),
        groups=groups,
        reference_hash=REFERENCE_HASH,
    )


def verify_worker_close(a: ReviewAudit, receipt: dict[str, Any], result_sha: str) -> None:
    """A clean child exit and a complete bound result are both mandatory."""
    a.prove(
        receipt,
        dict(
            schema="walker-trajectory-review-exit-v1",
            exit_code=0,
            timeout=False,
            error=None,
            execution_complete=True,
            result_sha256=result_sha,
            child_started=True,
            child_reaped=True,
            kill_attempted=False,
            cleanup_errors=[],
            result_error=None,
        ),
        "Complete successful worker closure",
    )


def audit_run(a: ReviewAudit, run: Path, *, config_sha: str, seal_sha: str) -> dict[str, Any]:
    """Validate a one-shot run's seals, original inputs, saved result and costs."""
    a.bind(run / "config.json", config_sha)
    a.bind(run / "input_manifest.json", seal_sha)
    config, seal, manifest = [
        a.load(run / name) for name in ("config.json", "input_manifest.json", "manifest.json")
    ]
    source_sha = a.bind(run / "manifest.json")["sha256"]
    a.need(
        config["schema"] == "walker-trajectory-review-config-v1"
        and seal["schema"] == "walker-trajectory-review-inputs-v1",
        "Versioned frozen review recipe/input seal",
    )
    a.need(
        manifest["config_sha256"] == seal["config_sha256"] == config_sha
        and seal["source_manifest_sha256"] == source_sha,
        "Exact source/config/input seal ownership",
    )
    a.prove(config["reconciliation"], dict(atol=ATOL, rtol=RTOL), "Fixed reconciliation tolerances")
    for binding in manifest["files"]:
        actual = a.bind(contained(a, run / "source", binding["file"]), binding["sha256"])
        if "bytes" in binding:
            a.need(actual["bytes"] == binding["bytes"], "Exact frozen source byte count")
    a.need(
        set(seal["roles"]) == set(config["inputs"]) == {*CAPSULE_KEYS, "capsule", "spread"},
        "Exact nine portable input roles",
    )
    a.need(
        set(seal["files"]) == set(seal["roles"].values())
        and len(seal["files"]) == config["budgets"]["inputs"] == 9,
        "Exactly nine distinct sealed inputs, no uncounted aliases or extra files",
    )
    a.need(Path(seal["source_root"]).is_absolute(), "Explicit original machine-specific input root")
    inputs, bindings = {}, {}
    for role, relative in seal["roles"].items():
        a.need(relative in seal["files"], "Role has sealed file binding")
        path = contained(a, run, relative)
        a.need(path.is_relative_to(run / "inputs"), "Portable saved inputs remain under inputs")
        a.need(path.suffix in (".json", ".csv"), "Only saved JSON/CSV metadata is eligible")
        inputs[role] = path
        binding = a.bind(path, seal["files"][relative]["sha256"])
        bindings[role] = binding
        a.prove(
            binding,
            {key: seal["files"][relative][key] for key in ("sha256", "bytes")},
            "Exact portable input " + role,
        )
        a.prove(
            binding,
            {key: config["inputs"][role][key] for key in ("sha256", "bytes")},
            "Exact original recipe input " + role,
        )
    for relative, binding in seal["files"].items():
        a.prove(
            a.bind(contained(a, run, relative), binding["sha256"]),
            {key: binding[key] for key in ("sha256", "bytes")},
            "Complete input seal",
        )
    a.need(seal["budgets"] == config["budgets"], "Input seal preserves finite budgets")
    capsule = a.load(inputs["capsule"])
    a.need(capsule["study"] == "action_compatibility_k1", "Original K1 capsule study identity")
    for role, key in CAPSULE_KEYS.items():
        a.prove(
            bindings[role],
            {field: capsule["files"][key][field] for field in ("sha256", "bytes")},
            "Original immutable K1 capsule binding " + role,
        )
    spread = a.load(inputs["spread"])
    for key, role in [
        ("capsule.json", "capsule"),
        *(
            (value, role)
            for role, value in CAPSULE_KEYS.items()
            if role not in ("actuators", "saturation")
        ),
    ]:
        a.prove(
            bindings[role],
            {field: spread["source_hashes"][key][field] for field in ("sha256", "bytes")},
            "Existing fixed-reference summary source binding " + role,
        )
    result = a.load(run / "result.json")
    a.prove(
        result,
        dict(
            schema="walker-trajectory-review-result-v1",
            execution_complete=True,
            error=None,
            config_sha256=config_sha,
            input_manifest_sha256=seal_sha,
            source_manifest_sha256=source_sha,
        ),
        "Complete result owner",
    )
    findings = audit_statistics(a, inputs, config, result)
    work = dict(
        input_files=9,
        input_bytes=sum(binding["bytes"] for binding in bindings.values()),
        metric_records=34,
        bins=304,
        source_control_rows_covered=30070,
        **{key: 0 for key in ZERO_WORK},
    )
    a.prove(result["work"], work, "Complete saved-only work census")
    a.prove(
        result["platform"],
        {key: config["platform"][key] for key in ("python", "numpy")},
        "Pinned saved-metric arithmetic interpreter and NumPy",
    )
    a.need(bool(result["platform"]["host"]), "Explicit analysis host")
    a.prove(config["budgets"], {key: 0 for key in ZERO_WORK}, "Zero new science budgets")
    for actual, ceiling in (
        (result["wall_seconds"], config["budgets"]["wall_seconds"]),
        (result["peak_rss_bytes"], config["budgets"]["peak_rss_bytes"]),
        (work["input_bytes"], config["budgets"]["input_bytes"]),
        (result["storage_bytes_before_result"], config["budgets"]["storage_bytes"]),
    ):
        a.need(
            isinstance(actual, (float, int)) and math.isfinite(actual) and 0 <= actual <= ceiling,
            "Worker stayed inside finite budget",
        )
    launch, exit_receipt = [
        a.load(run / name) for name in ("launch_receipt.json", "worker_exit.json")
    ]
    a.prove(
        launch,
        dict(
            schema="walker-trajectory-review-launch-v1",
            config_sha256=config_sha,
            input_manifest_sha256=seal_sha,
            source_manifest_sha256=source_sha,
            timeout_seconds=300,
            retries_permitted=False,
        ),
        "One-shot immutable launch binding",
    )
    verify_worker_close(a, exit_receipt, a.bind(run / "result.json")["sha256"])
    a.prove(
        launch["command"][1:],
        [
            "-B",
            str(run / "source/scripts/train_mppi_gps.py"),
            "study",
            "trajectory-review",
            "worker",
            "--out",
            str(run),
        ],
        "Actual command executes the frozen source in the unique run",
    )
    lock = a.load(run / "worker.lock")
    a.need(type(lock["pid"]) is int and lock["pid"] > 0, "Recorded one-shot worker PID")
    a.prove(
        lock,
        dict(launch_receipt_sha256=a.bind(run / "launch_receipt.json")["sha256"]),
        "Worker actually consumed this exact immutable launch receipt",
    )
    owner_lock = a.load(run / "launch.lock")
    a.need(type(owner_lock["pid"]) is int and owner_lock["pid"] > 0, "Recorded one-shot owner PID")
    a.prove(owner_lock, dict(config_sha256=config_sha), "One-shot owner consumed frozen recipe")
    a.need(
        0 <= exit_receipt["wall_seconds"] <= 300
        and 0 <= exit_receipt["stored_bytes"] <= config["budgets"]["storage_bytes"],
        "External elapsed/storage budget",
    )
    a.need(
        set(exit_receipt["streams"]) == {"worker.stdout", "worker.stderr"},
        "Both worker streams retained",
    )
    for name, binding in exit_receipt["streams"].items():
        a.prove(
            a.bind(contained(a, run, name), binding["sha256"]),
            {field: binding[field] for field in ("sha256", "bytes")},
            "Worker stream binding " + name,
        )
    # The immutable exit receipt declares bytes immediately before its own
    # creation. Reports and independent audit outputs must live elsewhere.
    retained = [path for path in run.rglob("*") if path.is_file()]
    for path in retained:
        a.need(str(path.resolve()) in a.files, "No unbound files added to the immutable run")
    retained_bytes = sum(path.stat().st_size for path in retained)
    a.need(
        retained_bytes - (run / "worker_exit.json").stat().st_size == exit_receipt["stored_bytes"],
        "Independently recounted exact worker-close storage",
    )
    a.need(
        retained_bytes <= config["budgets"]["storage_bytes"], "Complete closed run storage ceiling"
    )
    findings.update(
        work=work,
        elapsed_seconds=result["wall_seconds"],
        peak_rss_bytes=result["peak_rss_bytes"],
        stored_bytes=exit_receipt["stored_bytes"],
        closed_run_bytes=retained_bytes,
    )
    return findings


def assert_independent_imports(a: ReviewAudit) -> None:
    forbidden = (
        "torch",
        "h5py",
        "mujoco",
        "src.mppi",
        "src.envs",
        "src.gps.mppi_gps.trajectory_review",
        "src.gps.mppi_gps.experiments",
        "src.gps.mppi_gps.reporting",
        "src.gps.mppi_gps.policy",
        "src.gps.mppi_gps.runtime",
        "src.gps.mppi_gps.durability",
    )
    a.need(
        not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in forbidden
        ),
        "Independent stdlib/NumPy-only audit imports",
    )
    allowed = {
        "src.gps",
        "src.gps.mppi_gps",
        "src.gps.mppi_gps.audits",
        "src.gps.mppi_gps.audits.trajectory_review",
    }
    a.need(
        not any(name.startswith("src.") and name not in allowed for name in sys.modules),
        "No other repository implementation participates in this independent audit",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--seal-sha", required=True)
    args = parser.parse_args(argv)
    run, out = args.run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        parser.error("Audit output must be outside immutable run evidence")
    out.mkdir(parents=True, exist_ok=False)
    started, audit = time.perf_counter(), ReviewAudit()
    error, findings = None, {}
    try:
        audit.bind(Path(__file__))
        findings = audit_run(audit, run, config_sha=args.config_sha, seal_sha=args.seal_sha)
        for path, binding in list(audit.files.items()):
            audit.need(
                audit.bind(Path(path)) == binding, "Every audited input unchanged after read"
            )
        assert_independent_imports(audit)
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    receipt = dict(
        schema="walker-trajectory-review-independent-audit-v1",
        passed=error is None,
        error=error,
        checks=audit.checks,
        bound_files=len(audit.files),
        new_controls=0,
        new_neural_rows=0,
        new_optimizer_steps=0,
        raw_array_reads=0,
        checkpoint_loads=0,
        wall_seconds=time.perf_counter() - started,
        scope="Independent original K1 capsule/input/source bindings, complete fixed-reference roster, saved scalar/actuator/bin/cumulative identities, CSV ownership joins, population covariance decomposition and zero-new-science accounting. No raw arrays, checkpoints, policy evaluation or physics.",
    )
    for name, value in (
        ("receipt.json", receipt),
        ("findings.json", findings),
        ("file_bindings.json", audit.files),
    ):
        with (out / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(passed=receipt["passed"], checks=audit.checks, error=error)))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
