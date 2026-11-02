"""Finite learner-response selection for the existing Walker torque loop.

Only complete task outcomes enter these rules. This module performs no physics,
inference or fitting. Callers own paired acquisition, fixed learning updates,
fresh evaluation, frozen configurations and their Meter evidence.
"""

from __future__ import annotations

import math

SCHEMA = "walker-finite-post-update-feedback-v3.1"


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _summary(value):
    """Validate only outcome fields, never an action-agreement surrogate."""
    n = value["episodes"]
    if type(n) is not int or n < 1:
        raise ValueError("A nonempty complete cohort is required")
    for field in ("successes", "survived"):
        count = value[field]
        if type(count) is not int or not 0 <= count <= n:
            raise ValueError("Invalid cohort task counts")
    if value["successes"] > value["survived"]:
        raise ValueError("Success requires survival")
    cost = _number(value["mean_cost"], "mean_cost")
    steps = _number(value["mean_steps"], "mean_steps")
    if cost <= 0 or not 0 < steps <= 1000:
        raise ValueError("Positive Walker cost and 1..1000 mean controls required")
    if len(value.get("costs", [])) != n:
        raise ValueError("Retain every complete episode cost")
    costs = [_number(item, "episode cost") for item in value["costs"]]
    if not math.isclose(sum(costs) / n, cost, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Mean cost differs from complete episode costs")
    return n


def teacher_eligibility(independent, candidate, cost_tolerance=0.05):
    """Full task success and cost constraint, with no reference-path test."""
    n = _summary(independent)
    if _summary(candidate) != n:
        raise ValueError("Matched teacher cohort sizes required")
    tolerance = _number(cost_tolerance, "cost_tolerance")
    if tolerance < 0:
        raise ValueError("Nonnegative teacher cost tolerance required")
    checks = dict(
        independent_task_success=independent["successes"] == n,
        candidate_task_success=candidate["successes"] == n,
        complete_cost_guard=candidate["mean_cost"] <= independent["mean_cost"] * (1 + tolerance),
    )
    return dict(
        eligible=all(checks.values()),
        checks=checks,
        relative_teacher_cost_change=(candidate["mean_cost"] / independent["mean_cost"] - 1),
    )


def child_comparison(parent, independent, guided, minimum_gain=0.05):
    """Compare a fixed-update child to both controls on the same fresh starts.

    Pairing and input ownership must be checked by the caller; counts alone do
    not establish those properties. Per-start cost deltas are retained here.
    This development gate is not a statistical confidence guarantee.
    """
    n = _summary(parent)
    if _summary(independent) != n or _summary(guided) != n:
        raise ValueError("Matched standalone cohort sizes required")
    gain = _number(minimum_gain, "minimum_gain")
    if not 0 < gain < 1:
        raise ValueError("minimum_gain must lie strictly between zero and one")
    controls = {}
    for label, control in (("parent", parent), ("independent", independent)):
        relative = (control["mean_cost"] - guided["mean_cost"]) / control["mean_cost"]
        checks = dict(
            cost_gain=relative >= gain,
            success_nonregression=guided["successes"] >= control["successes"],
            survival_nonregression=guided["survived"] >= control["survived"],
            mean_steps_nonregression=guided["mean_steps"] >= control["mean_steps"],
        )
        controls[label] = dict(
            passed=all(checks.values()),
            checks=checks,
            relative_complete_cost_gain=relative,
            paired_cost_deltas_guided_minus_control=[
                float(g) - float(c) for g, c in zip(guided["costs"], control["costs"])
            ],
        )
    return dict(
        schema=SCHEMA,
        passed=all(v["passed"] for v in controls.values()),
        controls=controls,
        minimum_relative_cost_gain=gain,
        action_agreement_is_gate=False,
    )


def next_acquisition(primary, confirmation, *, parent_hash, child_hash, rho, completed_rounds=0):
    """Specify exact planner-facing feedback without constructing a planner.

    A rejected child never replaces its parent. The returned actor hash and rho
    bind the next guided selector; every acquisition also has a zero-rho control.
    Two rounds are a minimum for a loop claim, not sufficient on their own.
    """
    if not all(
        isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
        for h in (parent_hash, child_hash)
    ):
        raise ValueError("Explicit parent and child policy hashes required")
    rho = _number(rho, "rho")
    if rho <= 0 or type(completed_rounds) is not int or completed_rounds < 0:
        raise ValueError("Positive guided coupling and nonnegative round count required")
    if not isinstance(primary.get("passed"), bool):
        raise ValueError("Primary comparison result required")
    confirmed = bool(
        primary["passed"] and confirmation is not None and confirmation.get("passed") is True
    )
    return dict(
        schema=SCHEMA,
        advance=confirmed,
        actor_hash=child_hash if confirmed else parent_hash,
        optimizer_owner_hash=child_hash if confirmed else parent_hash,
        guided_rho=rho if confirmed else 0.0,
        independent_rho=0.0,
        independent_producer_policy_access=False,
        completed_rounds=completed_rounds + int(confirmed),
        second_round_required_for_loop_claim=completed_rounds + int(confirmed) < 2,
        stop_reason=None
        if confirmed
        else (
            "primary_child_did_not_improve"
            if not primary["passed"]
            else "confirmation_required_or_failed"
        ),
        next_round_requires_frozen_acquisition_protocol=confirmed,
    )
