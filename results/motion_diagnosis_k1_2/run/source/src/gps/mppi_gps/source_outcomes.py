"""Shared source-comparison outcomes; action fitting and execution stay separate."""

from __future__ import annotations


def source_screen(evaluations: dict, config: dict) -> dict:
    pairs = []
    for pair in range(len(config["seed_pairs"])):
        baseline = evaluations[f"mppi_s{pair}"]["summary"]
        candidate = evaluations[f"sac_reference_s{pair}"]["summary"]
        pairs.append(
            dict(
                pair=pair,
                walking_gain=candidate["successes"] - baseline["successes"],
                survival_gain=candidate["survived"] - baseline["survived"],
                walking_nonregression=candidate["successes"] >= baseline["successes"],
                survival_nonregression=candidate["survived"] >= baseline["survived"],
                relative_cost_gain=1 - candidate["mean_cost"] / baseline["mean_cost"],
                controls_delta=candidate["mean_steps"] - baseline["mean_steps"],
            )
        )
    gate = config["walking_screen"]
    checks = dict(
        positive_pairs=sum(
            row["walking_gain"] >= gate["minimum_additional_successes"] for row in pairs
        )
        >= gate["minimum_positive_pairs"],
        all_pair_walking_nonregression=all(row["walking_nonregression"] for row in pairs),
        all_pair_survival_nonregression=all(row["survival_nonregression"] for row in pairs),
    )
    return dict(pairs=pairs, checks=checks, walking_screen=all(checks.values()))


def add_work(before: dict, work: dict) -> dict:
    """Add disjoint Meter counters once; retain each phase separately in reports."""
    return {key: before.get(key, 0) + work.get(key, 0) for key in set(before) | set(work)}
