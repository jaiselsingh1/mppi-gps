"""H4.1 decision arithmetic only; no acquisition, simulation or learning.

This is a development-informed hypothesis, not an empirically validated
controller. The worker must independently prove checkpoint/data ownership and
full paired outcomes. Weight proposals do not promote a policy for deployment.
"""
from __future__ import annotations

import math

SCHEMA = "walker-outcome-coupling-backoff-h4.1"


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Explicit SHA-256 policy owner required")
    return value


def _counts(score):
    n, successes, survived = (score[k] for k in ("episodes", "successes", "survived"))
    if any(type(v) is not int for v in (n, successes, survived)) or not 0 <= successes <= survived <= n or n < 1:
        raise ValueError("Complete cohort counts must satisfy 0 <= successes <= survived <= episodes")
    return n, successes, survived


def propose_weight(rho, guided, independent, *, guided_parent_hash,
                   independent_parent_hash, updated_guided_hash,
                   guided_seeds, independent_seeds):
    """Halve iff the matched guided response loses task success OR survival.

    Equal incoming policy IDs and distinct paired seeds are necessary checks,
    not proof of identical Adam/data/update state. Callers bind those objects
    before using this decision in an acquisition. Cost/MSE/agreement do not
    enter this rule, and a lower proposed weight is not a positive outcome.
    """
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho <= 0:
        raise ValueError("Finite strictly positive entering agreement weight required")
    entering = _hash(guided_parent_hash)
    if entering != _hash(independent_parent_hash):
        raise ValueError("Feedback responses require the same incoming learner")
    updated = _hash(updated_guided_hash)
    n, gs, gv = _counts(guided)
    ni, ins, inv = _counts(independent)
    if n != ni:
        raise ValueError("Matched response cohort sizes required")
    first, second = tuple(guided_seeds), tuple(independent_seeds)
    if (len(first) != n or first != second or len(set(first)) != n or
            any(type(seed) is not int or seed < 0 for seed in first)):
        raise ValueError("Explicit identical distinct paired evaluation seeds required")
    reasons = []
    if gs < ins:
        reasons.append("lost_task_success")
    if gv < inv:
        reasons.append("lost_full_survival")
    next_rho = rho / 2 if reasons else float(rho)
    if next_rho <= 0:
        raise ValueError("Backoff underflow; stop rather than substitute a new weight")
    return dict(schema=SCHEMA, entering_rho=float(rho), proposed_rho=next_rho,
                factor=0.5 if reasons else 1.0, trigger=bool(reasons), reasons=reasons,
                feedback_incoming_policy_hash=entering, next_acquisition_actor_hash=updated,
                next_acquisition_optimizer_owner_hash=updated,
                matched_evaluation_seeds=list(first),
                guided_counts=dict(episodes=n, successes=gs, survived=gv),
                independent_counts=dict(episodes=n, successes=ins, survived=inv),
                independent_control_rho=0.0, automatic_increase=False,
                current_action_agreement_used=False, deployment_promotion=False,
                empirical_benefit_established=False)
