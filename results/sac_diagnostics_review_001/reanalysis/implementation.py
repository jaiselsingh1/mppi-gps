"""Review preserved SAC diagnostic summaries without models, fitting or physics.

This retrospective calculation corrects tied ranks and states the population
behind the exploratory AUC. It does not turn post-fit competence into a
pre-update learning prediction or reconstruct evidence the original runs omit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

STUDIES = ("student_positive_control", "jitter_tolerance", "on_policy_disagreement")
SCORES = (
    "offline_action_mse",
    "on_policy_standing",
    "on_policy_standing_short",
    "on_policy_mid_stride",
    "on_policy_mid_stride_short",
)
MAX_SUMMARY_BYTES = 256 * 1024
ZERO_WORK = dict(
    native_controls=0,
    model_controls=0,
    physics_substeps=0,
    neural_rows=0,
    backward_rows=0,
    optimizer_steps=0,
    checkpoint_loads=0,
)


def average_ranks(values: list[float]) -> list[float]:
    """Return zero-based average ranks, assigning equal observations equal rank."""
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Ranks require finite observations")
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    first = 0
    while first < len(order):
        end = first + 1
        while end < len(order) and values[order[end]] == values[order[first]]:
            end += 1
        rank = (first + end - 1) / 2
        for index in order[first:end]:
            ranks[index] = rank
        first = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    """Tie-correct rank correlation; undefined/constant outcomes remain missing."""
    if len(x) != len(y):
        raise ValueError("Paired observations must have equal length")
    if len(x) < 2:
        return None
    a, b = average_ranks(x), average_ranks(y)
    a_mean, b_mean = math.fsum(a) / len(a), math.fsum(b) / len(b)
    da = [value - a_mean for value in a]
    db = [value - b_mean for value in b]
    denominator = math.sqrt(math.fsum(value * value for value in da)) * math.sqrt(
        math.fsum(value * value for value in db)
    )
    if denominator == 0:
        return None
    return math.fsum(left * right for left, right in zip(da, db, strict=True)) / denominator


def auc_lower_is_better(scores: list[float], positives: list[bool]) -> float | None:
    """Empirical pairwise AUC with half credit for tied scores."""
    if len(scores) != len(positives):
        raise ValueError("AUC observations must have equal length")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("AUC requires finite observations")
    yes = [value for value, positive in zip(scores, positives, strict=True) if positive]
    no = [value for value, positive in zip(scores, positives, strict=True) if not positive]
    if not yes or not no:
        return None
    return math.fsum(
        1.0 if left < right else 0.5 if left == right else 0.0 for left in yes for right in no
    ) / (len(yes) * len(no))


def analyze(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute review statistics from three unchanged, explicitly named summaries."""
    positive = summaries["student_positive_control"]
    jitter = summaries["jitter_tolerance"]
    on_policy = summaries["on_policy_disagreement"]
    students = on_policy["students"]
    walks = [float(row["walks"]) for row in students]
    if not students or any(value < 0 or value > 32 or not value.is_integer() for value in walks):
        raise ValueError("Expected saved integer walking counts out of 32")
    statistics = {}
    for key in SCORES:
        values = [float(row[key]) for row in students]
        statistics[key] = dict(
            spearman_tie_correct=spearman(values, walks),
            auc_any_walking=auc_lower_is_better(values, [value > 0 for value in walks]),
        )
    on_policy_rho = statistics["on_policy_standing"]["spearman_tie_correct"]
    offline_rho = statistics["offline_action_mse"]["spearman_tie_correct"]
    advantage = (
        abs(on_policy_rho) - abs(offline_rho)
        if on_policy_rho is not None and offline_rho is not None
        else None
    )
    positive_outcomes = {}
    for key, row in positive["table"].items():
        counts = row["standing"]
        if any(not isinstance(value, int) or not 0 <= value <= 32 for value in counts):
            raise ValueError("Invalid positive-control standing counts")
        positive_outcomes[key] = dict(walks=sum(counts), episodes=32 * len(counts), per_seed=counts)
    jitter_outcomes = []
    for row in jitter["conditions"]:
        counts = [student["standing_walks"] for student in row["students"]]
        if not counts or any(
            not isinstance(value, int) or not 0 <= value <= 32 for value in counts
        ):
            raise ValueError("Invalid jitter standing counts")
        if not math.isclose(
            sum(counts) / (32 * len(counts)), row["standing_walk_rate"], abs_tol=1e-12
        ):
            raise ValueError("Jitter aggregate disagrees with student counts")
        jitter_outcomes.append(
            dict(
                jitter=row["jitter"],
                signal_share=row["signal_share"],
                walks=sum(counts),
                episodes=32 * len(counts),
                per_seed=counts,
            )
        )
    return dict(
        status="RETROSPECTIVE_SAVED_SUMMARY_REVIEW",
        positive_control_outcomes=positive_outcomes,
        jitter_outcomes=jitter_outcomes,
        on_policy_statistics=statistics,
        auc_population=dict(
            unit="fitted student",
            positive_definition="at least one walking episode out of 32, not competence",
            positive_students=sum(value > 0 for value in walks),
            zero_walking_students=sum(value == 0 for value in walks),
            prospective_or_protected=False,
        ),
        corrected_original_on_policy_gate=dict(
            standing_correlation_advantage=advantage,
            minimum_absolute_correlation=0.7,
            minimum_advantage=0.15,
            passes=(
                on_policy_rho is not None
                and abs(on_policy_rho) >= 0.7
                and advantage is not None
                and advantage >= 0.15
            ),
        ),
        original_summary_statistics=dict(
            on_policy_spearman=on_policy["spearman_with_walking"],
            on_policy_readings=on_policy["readings"],
            positive_control_readings=positive["readings"],
            jitter_readings=jitter["readings"],
        ),
        recipe_contrast=dict(
            authority="observed source snapshot; original launch binding unavailable",
            positive_control_and_jitter=dict(
                architecture="17 ->128 tanh ->128 tanh ->6 tanh",
                unique_training_rows=9000,
                source_trajectories=9,
                optimizer_updates=10000,
                batch_size=256,
                learning_rate=0.0003,
                normalization="per training corpus mean/std, std floor 1e-6",
                learner_seeds=[0, 1, 2],
                standing_resets=32,
            ),
            j1=dict(
                architecture="17 ->128 tanh ->128 tanh ->6 tanh",
                unique_training_rows=1000,
                source_trajectories=1,
                optimizer_updates=2000,
                batch_size=256,
                learning_rate=0.0003,
                normalization="frozen W1 normalization",
                standing_resets=16,
            ),
            interpretation="Data size/diversity, update count, normalization and resets differ; resource cause is not isolated.",
        ),
        limits=[
            "Original scripts retain summaries and figures, not checkpoints, Adam/RNG, index tapes or executed student traces.",
            "On-policy analysis refitted students and checked one held-MSE scalar; exact original actor equality is unverified.",
            "On-policy scores are post-fit competence diagnostics; they do not predict that same update before fitting.",
            "Jitter arms are fresh fits, not fine-tuning of competent parents; forgetting under MPPI updates remains untested.",
            "Original Spearman used ordinal double argsort; this review uses average ranks for ties without changing originals.",
            "AUC is exploratory on the same 39 students and shared sources/seeds, with no protected confirmation or independence claim.",
        ],
    )


def run_review(capsule: Path, output: Path) -> dict[str, Any]:
    """Read three bounded sealed summaries and create a separate review directory."""
    started = time.monotonic()
    manifest_data = (capsule / "capsule.json").read_bytes()
    if len(manifest_data) > 1024 * 1024:
        raise ValueError("Oversized review manifest")
    manifest = json.loads(manifest_data)
    summaries, inputs = {}, {}
    read_bytes = len(manifest_data)
    for study in STUDIES:
        name = f"runs/2026-11-02_{study}_001/summary.json"
        path = capsule / name
        if path.is_symlink() or path.stat().st_size > MAX_SUMMARY_BYTES:
            raise ValueError("Review requires small ordinary saved JSON summaries")
        payload = path.read_bytes()
        binding = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        if binding != {key: manifest["files"][name][key] for key in binding}:
            raise ValueError("Saved review input hash/size mismatch: " + name)
        read_bytes += len(payload)
        summaries[study] = json.loads(payload)
        inputs[name] = binding
    report = analyze(summaries)
    output.mkdir(parents=True, exist_ok=False)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    (output / "summary.json").write_bytes(payload)
    receipt = dict(
        status="PASSED_SAVED_SUMMARY_REANALYSIS_ONLY",
        inputs=inputs,
        input_manifest_sha256=hashlib.sha256(manifest_data).hexdigest(),
        read_bytes=read_bytes,
        output=dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()),
        wall_seconds=time.monotonic() - started,
        new_scientific_work=ZERO_WORK,
    )
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_review(args.capsule, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
