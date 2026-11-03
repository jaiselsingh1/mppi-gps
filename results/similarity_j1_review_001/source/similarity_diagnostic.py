"""Post-hoc J1 outcome decomposition from compact saved metadata only.

Run with ``python -m src.gps.mppi_gps.reporting.similarity_diagnostic``.
This supplement never changes frozen scores, gates or protected-test status.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

MAX_INPUT = 16 * 1024**2
MAX_OUTPUT = 8 * 1024**2
MAX_WALL = 60
CANDIDATES = tuple("ABCDE")


def decompose_motion(components: dict[str, Any], horizon: int) -> dict[str, float]:
    """Separate observed-prefix error from the frozen missing-time term."""
    common = components["common_controls"]
    means = np.asarray(components["capped_component_means"], dtype=np.float64)
    if (
        type(horizon) is not int
        or horizon <= 0
        or type(common) is not int
        or not 0 < common <= horizon
        or means.shape != (4,)
        or not np.isfinite(means).all()
        or (means < 0).any()
        or (means > 9).any()
    ):
        raise ValueError("Motion decomposition requires a nonempty actual overlap and four means")
    prefix_mean = float(means.mean())
    fraction = common / horizon
    observed_term = fraction * prefix_mean
    missing_term = 4 * (1 - fraction)
    frozen = math.sqrt(observed_term + missing_term)
    if not math.isclose(frozen, float(components["error"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Saved motion components do not reconstruct the frozen score")
    return dict(
        overlap_component_mean=prefix_mean,
        overlap_only_motion=math.sqrt(prefix_mean),
        frozen_observed_term=observed_term,
        frozen_missing_term=missing_term,
        frozen_motion_reconstructed=frozen,
    )


def _ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite rank input")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def descriptive_spearman(x: list[float], y: list[float]) -> float | None:
    """Tied-rank correlation only; no ungrouped significance claim."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return None if denominator == 0 else float(rx @ ry / denominator)


def analyze_cases(cases: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    if not isinstance(cases, list) or not 0 < len(cases) <= 64:
        raise ValueError("Expected one finite J1 learning-case roster")
    rows = []
    seen = set()
    try:
        for case in cases:
            case_id = case["case_id"]
            if case_id in seen:
                raise ValueError("Duplicate learning-case identifier")
            seen.add(case_id)
            evaluations = case["evaluations"]
            if not isinstance(evaluations, list) or not 0 < len(evaluations) <= 64:
                raise ValueError("Missing or oversized standalone outcome panel")
            teacher = case["teacher"]
            reproduction = case["reproduction"]
            if any(
                type(value["controls"]) is not int
                or not 0 < value["controls"] <= teacher["horizon"]
                for value in evaluations
            ):
                raise ValueError("Standalone durations require positive actual control counts")
            decomposition = decompose_motion(case["motion_components"], teacher["horizon"])
            frozen = float(case["post_motion"])
            if not math.isclose(
                frozen, decomposition["frozen_motion_reconstructed"], rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("Saved case and motion component scores differ")
            if reproduction["controls"] != case["motion_components"]["common_controls"]:
                raise ValueError("Reproduction controls differ from saved common-prefix length")
            total_cost = sum(float(e["cost"]) for e in evaluations)
            tail_cost = sum(float(e["absorbing_cost"]) for e in evaluations)
            row = dict(
                case_id=case_id,
                group=case["group"],
                learner=case["learner"],
                source=case["source"],
                split=case["split"],
                standalone_attempts=len(evaluations),
                standalone_walking=sum(bool(e["walking"]) for e in evaluations),
                standalone_full_survival=sum(bool(e["survived"]) for e in evaluations),
                standalone_controls=sum(int(e["controls"]) for e in evaluations),
                standalone_mean_controls=float(np.mean([e["controls"] for e in evaluations])),
                standalone_total_cost=total_cost,
                standalone_total_absorbing_cost=tail_cost,
                standalone_tail_fraction=None if total_cost == 0 else tail_cost / total_cost,
                reproduction_controls=reproduction["controls"],
                reproduction_cost=float(reproduction["cost"]),
                reproduction_absorbing_cost=float(reproduction["absorbing_cost"]),
                frozen_motion=frozen,
                **decomposition,
                **{f"score_{name}": float(case["scores"][name]) for name in CANDIDATES},
            )
            if not all(
                math.isfinite(value) for value in row.values() if isinstance(value, (float, int))
            ):
                raise ValueError("Nonfinite saved outcome")
            rows.append(row)
    except KeyError as error:
        raise ValueError(f"Missing saved evidence: {error.args[0]}") from error

    summaries = {}
    for source in [*sorted({row["source"] for row in rows}), "all"]:
        selected = [row for row in rows if source == "all" or row["source"] == source]
        survival = [row["standalone_mean_controls"] for row in selected]
        total_cost = sum(row["standalone_total_cost"] for row in selected)
        reproduction_cost = sum(row["reproduction_cost"] for row in selected)
        summaries[source] = dict(
            cases=len(selected),
            acquisition_groups=len({row["group"] for row in selected}),
            standalone_mean_controls_range=[min(survival), max(survival)],
            standalone_mean_controls_ratio=max(survival) / min(survival),
            standalone_walking=sum(row["standalone_walking"] for row in selected),
            standalone_attempts=sum(row["standalone_attempts"] for row in selected),
            standalone_aggregate_tail_fraction=None
            if total_cost == 0
            else sum(row["standalone_total_absorbing_cost"] for row in selected) / total_cost,
            reproduction_aggregate_tail_fraction=None
            if reproduction_cost == 0
            else sum(row["reproduction_absorbing_cost"] for row in selected) / reproduction_cost,
            frozen_motion_range=[
                min(row["frozen_motion"] for row in selected),
                max(row["frozen_motion"] for row in selected),
            ],
            frozen_motion_population_std=float(np.std([r["frozen_motion"] for r in selected])),
            overlap_only_motion_range=[
                min(row["overlap_only_motion"] for row in selected),
                max(row["overlap_only_motion"] for row in selected),
            ],
            descriptive_score_survival_spearman={
                name: descriptive_spearman([row[f"score_{name}"] for row in selected], survival)
                for name in CANDIDATES
            },
        )
    return rows, dict(
        schema="walker-j1-posthoc-outcome-decomposition-v1",
        post_hoc=True,
        saved_only=True,
        predictive_validation=False,
        frozen_scores_and_gates_unchanged=True,
        formula="frozen_motion^2=(common/horizon)*overlap_only_motion^2+4*(1-common/horizon)",
        interpretation=(
            "As common/horizon approaches zero the frozen score approaches two. "
            "When observed-prefix mean component error is near four, elapsed duration has "
            "little influence on the frozen score. Overlap-only error is conditioned on "
            "surviving prefixes of unequal lengths and remains sensitive to synchronization; "
            "it is not competence or a validated predictor. Walking and duration stay separate."
        ),
        correlation_scope=(
            "Descriptive case-level ranks, with related learners sharing acquisition groups. "
            "No significance, protected discrimination or candidate advantage is claimed. "
            "Pooled-source correlations are context only and are not the primary MPPI comparison."
        ),
        outcomes=summaries,
    )


def _join_table(rows: list[dict], table: list[dict[str, str]]) -> None:
    indexed = {row["case_id"]: row for row in table}
    if len(indexed) != len(table) or set(indexed) != {row["case_id"] for row in rows}:
        raise ValueError("Saved compact case-table roster differs")
    for row in rows:
        saved = indexed[row["case_id"]]
        for field in ("group", "source", "split", "learner"):
            if saved[field] != row[field]:
                raise ValueError("Saved compact case-table identity differs")
        for field in ("standalone_attempts", "standalone_walking", "standalone_controls"):
            if int(saved[field]) != row[field]:
                raise ValueError("Saved compact case-table outcome differs")
        if float(saved["post_motion"]) != row["frozen_motion"]:
            raise ValueError("Saved compact case-table frozen motion differs")


def _plot(rows: list[dict]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for source in sorted({row["source"] for row in rows}):
        selected = [row for row in rows if row["source"] == source]
        for index, axis in enumerate(axes.flat):
            x = "reproduction_controls" if index < 2 else "standalone_mean_controls"
            y = "frozen_motion" if index % 2 == 0 else "overlap_only_motion"
            axis.scatter([row[x] for row in selected], [row[y] for row in selected], label=source)
            axis.set_xlabel(
                "Same-start executed controls" if index < 2 else "Mean standalone executed controls"
            )
            axis.set_ylabel(
                "Frozen motion error" if index % 2 == 0 else "Overlap-only motion error"
            )
            axis.grid(alpha=0.2)
    axes[0, 0].axhline(2, color="gray", linestyle="--", linewidth=1, label="zero-overlap limit")
    axes[0, 0].legend()
    figure.suptitle(
        "J1 development outcomes — post-hoc decomposition, no protected prediction test"
    )
    payload = io.BytesIO()
    figure.savefig(payload, format="png", dpi=150)
    plt.close(figure)
    return payload.getvalue()


def supplement(capsule: Path, out: Path) -> dict:
    started = time.monotonic()
    inputs = {}
    read_bytes = 0
    if out.exists():
        raise FileExistsError(out)
    if out.resolve().is_relative_to(capsule.resolve()):
        raise ValueError("Supplement output must remain outside the immutable capsule")

    def read(name: str) -> bytes:
        nonlocal read_bytes
        path = capsule / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(capsule.resolve())
        ):
            raise ValueError(f"Missing or unsafe saved evidence: {name}")
        before = path.stat()
        size = before.st_size
        if size + read_bytes > MAX_INPUT:
            raise ValueError("Supplement input reservation exhausted")
        payload = path.read_bytes()
        after = path.stat()

        def identity(value):
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if len(payload) != size or identity(before) != identity(after):
            raise ValueError("Saved supplement input changed")
        read_bytes += size
        inputs[name] = dict(bytes=size, sha256=hashlib.sha256(payload).hexdigest())
        return payload

    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite saved JSON: {value}")

    cases = json.loads(read("run/worker/cases.json"), parse_constant=reject)
    config = json.loads(read("run/config.json"), parse_constant=reject)
    compressed = read("report/cases.csv.gz")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        decoded = stream.read(MAX_INPUT + 1)
    if len(decoded) > MAX_INPUT:
        raise ValueError("Decompressed compact case table exceeds reservation")
    rows, summary = analyze_cases(cases)
    _join_table(rows, list(csv.DictReader(io.StringIO(decoded.decode()))))
    summary["frozen_prediction_configuration"] = config["prediction"]
    summary["frozen_scientific_source_commit"] = config["source_commit"]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    payloads = {
        "cases.csv": csv_buffer.getvalue().encode(),
        "summary.json": (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode(),
        "motion_and_duration.png": _plot(rows),
    }
    if sum(map(len, payloads.values())) > MAX_OUTPUT - 64 * 1024:
        raise ValueError("Supplement output reservation exhausted")
    if time.monotonic() - started > MAX_WALL:
        raise ValueError("Supplement wall reservation exhausted")
    out.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (out / name).write_bytes(payload)
    receipt = dict(
        schema="walker-j1-posthoc-supplement-receipt-v1",
        saved_only=True,
        post_hoc=True,
        actual_cases=len(rows),
        input_files=inputs,
        read_bytes=read_bytes,
        decompressed_table_bytes=len(decoded),
        output_files={
            name: dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            for name, payload in payloads.items()
        },
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_scientific_work=dict(
            native_controls=0,
            model_controls=0,
            physics_substeps=0,
            neural_rows=0,
            backward_rows=0,
            optimizer_steps=0,
            checkpoint_loads=0,
        ),
        wall_seconds=time.monotonic() - started,
        limits=dict(input_bytes=MAX_INPUT, output_bytes=MAX_OUTPUT, wall_seconds=MAX_WALL),
    )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(supplement(args.capsule, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
