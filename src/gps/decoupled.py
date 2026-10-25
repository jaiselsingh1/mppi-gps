"""Proposal ablations and cost-verified target synthesis, without physics calls.

The caller supplies one already-scored cloud and a callback which evaluates an
entire proposed plan with the SAME task/continuation objective as those scores.
Only verified full-plan improvement is asserted; fitting its first action does
not certify the neural policy's improvement or future closed-loop behavior.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Literal

import torch

from src.gps.torch_mppi import (
    softmin_weights, synthesize_plan, temperature_for_neff_fraction_with_details,
)

ProposalMode = Literal["seeded", "split_proposal"]
SynthesisMode = Literal["mean_verified", "medoid_fallback"]
VerifyPlans = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
SOURCE_NAMES = {0: "policy_nominal", 1: "verified_mean", 2: "single_anchor", 3: "verified_local_cluster"}


def _floating(value: torch.Tensor, name: str, device: torch.device | None = None,
              *, finite: bool = True) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating torch tensor.")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on the same device as the plans.")
    value = value.to(dtype=torch.float64)
    if finite and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} must be finite.")
    return value


def _bounds(low: torch.Tensor, high: torch.Tensor, plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    low, high = _floating(low, "low", plan.device), _floating(high, "high", plan.device)
    if low.shape != plan.shape[-1:] or high.shape != low.shape or not bool((low < high).all()):
        raise ValueError("Bounds must have shape (action,) and satisfy low < high.")
    return low, high


@torch.no_grad()
def make_proposals(
    policy_nominal: torch.Tensor,
    noise: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    proposal_mode: ProposalMode = "seeded",
    policy_sigma: float = .3,
    independent_sigma: float = .3,
    independent_center: Literal["zero", "hold"] = "zero",
    previous_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build (B,K,H,A) actions using the existing sampler's scaled noise.

    ``noise`` is ALREADY scaled by ``policy_sigma`` (including its established
    temporal filter). Seeded mode is exactly clip(policy_nominal + noise).
    Split mode uses the first K/2 draws around that nominal and rescales the
    remaining draws to fixed ``independent_sigma`` around zero/held action.
    K=256 therefore allocates128+128 without another noise draw or rollout.
    A held center must be supplied explicitly; it is never inferred from pi.
    No cross-state or previous MPPI plan is retained by this function.
    """
    nominal = _floating(policy_nominal, "policy_nominal")
    noise = _floating(noise, "noise", nominal.device)
    if nominal.ndim != 3 or min(nominal.shape[1:]) < 1 or noise.ndim != 4:
        raise ValueError("Expected policy_nominal(B,H,A) and noise(B,K,H,A).")
    if noise.shape[0] != nominal.shape[0] or noise.shape[2:] != nominal.shape[1:] or noise.shape[1] < 1:
        raise ValueError("Noise must match the nominal batch/horizon/action dimensions.")
    if proposal_mode not in {"seeded", "split_proposal"}:
        raise ValueError("Unknown proposal_mode.")
    if independent_center not in {"zero", "hold"}:
        raise ValueError("independent_center must be zero or hold.")
    for name, sigma in (("policy_sigma", policy_sigma), ("independent_sigma", independent_sigma)):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    low, high = _bounds(low, high, nominal)
    if proposal_mode == "seeded":
        if independent_center != "zero" or previous_action is not None:
            raise ValueError("Independent center settings require split_proposal.")
        raw = nominal[:, None] + noise
    else:
        if noise.shape[1] % 2:
            raise ValueError("split_proposal requires an even candidate count.")
        if independent_center == "hold":
            if previous_action is None:
                raise ValueError("hold requires an explicit previous_action(B,A).")
            held = _floating(previous_action, "previous_action", nominal.device)
            if held.shape != (nominal.shape[0], nominal.shape[2]):
                raise ValueError("previous_action must have shape (B,A).")
            center = held[:, None, :].expand_as(nominal)
        else:
            if previous_action is not None:
                raise ValueError("previous_action requires independent_center='hold'.")
            center = torch.zeros_like(nominal)
        half = noise.shape[1] // 2
        raw = torch.cat((nominal[:, None] + noise[:, :half],
                         center[:, None] + noise[:, half:] * (independent_sigma / policy_sigma)), dim=1)
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("Proposal synthesis overflowed before clipping.")
    return torch.maximum(torch.minimum(raw, high), low)


@dataclass(frozen=True)
class VerifiedSynthesis:
    """Device-resident plans and explicit provenance for a supervised target."""

    plans: torch.Tensor
    labels: torch.Tensor
    costs: torch.Tensor
    accepted: torch.Tensor
    source_codes: torch.Tensor
    anchor_indices: torch.Tensor
    mean_plans: torch.Tensor
    mean_costs: torch.Tensor
    weights: torch.Tensor
    temperatures: torch.Tensor
    n_eff: torch.Tensor
    temperature_saturated: torch.Tensor
    expected_candidate_cost: torch.Tensor
    improving_candidate_count: torch.Tensor
    mean_rejected: torch.Tensor
    mean_tied: torch.Tensor
    bimodal_collapse: torch.Tensor
    rescue_eligible: torch.Tensor
    rescued: torch.Tensor
    cluster_candidate_count: torch.Tensor
    verification_rollouts: int
    verification_callback_calls: int

    def diagnostics(self) -> dict:
        """Counts/rates with distinct, explicit denominators; no empty0/0 rate."""
        count = lambda mask: int(mask.sum().item())
        collapsed, eligible, rejected = count(self.bimodal_collapse), count(self.rescue_eligible), count(self.mean_rejected)
        rescued = count(self.rescued)
        collapse_rescued = count(self.bimodal_collapse & self.rescued)
        return {"states": len(self.accepted), "accepted": count(self.accepted),
                "mean_rejected_count": rejected, "mean_tied_count": count(self.mean_tied),
                "bimodal_collapse_count": collapsed,
                "bimodal_collapse_denominator": len(self.accepted),
                "bimodal_collapse_rate": collapsed / len(self.accepted) if len(self.accepted) else None,
                "medoid_rescued_collapse_count": collapse_rescued,
                "medoid_rescue_denominator": collapsed,
                "medoid_rescue_rate": collapse_rescued / collapsed if collapsed else None,
                "rescue_eligible_count": eligible, "rescued_count": rescued,
                "eligible_rescue_rate": rescued / eligible if eligible else None,
                "all_rejected_rescue_denominator": rejected,
                "all_rejected_rescue_rate": rescued / rejected if rejected else None,
                "verification_rollouts": self.verification_rollouts,
                "verification_callback_calls": self.verification_callback_calls,
                "source_names": dict(SOURCE_NAMES)}


@torch.no_grad()
def synthesize_verified(
    sampled: torch.Tensor,
    scores: torch.Tensor,
    nominal: torch.Tensor,
    nominal_cost: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    verify: VerifyPlans,
    *,
    synthesis_mode: SynthesisMode = "mean_verified",
    neff_fraction: float = .2,
    beta: float = 0.,
    cluster_radius: float | None = None,
) -> VerifiedSynthesis:
    """Verify the mean, optionally rescue it with one already-scored candidate.

    ``verify(plans[B',H,A], row_indices[B']) -> costs[B']`` MUST evaluate
    complete plans with the same frozen-policy continuation and task units as
    ``scores`` and ``nominal_cost``. The callback is called once for every mean.
    Default single-anchor fallback reuses an already-verified candidate score
    and calls no simulator. Candidate argmin ties choose the first index.

    With an explicit ``cluster_radius``, only improving candidates at strict
    full-plan distance<radius from the anchor may be averaged. Its actual cost
    is verified in one extra subset callback, with single-anchor fallback if
    that average fails. A small radius is NOT a single-basin guarantee. Leave
    it unset for the base compute-matched experiment.

    Nonfinite candidate scores are excluded; a row with no finite score raises.
    Nonfinite nominal/actions/verification costs raise instead of making an
    unverified target. Equality is rejected. Rejected rows return their old
    nominal and cost with accepted=False, never a purported improvement.
    """
    nominal = _floating(nominal, "nominal")
    sampled = _floating(sampled, "sampled", nominal.device)
    scores = _floating(scores, "scores", nominal.device, finite=False)
    baseline = _floating(nominal_cost, "nominal_cost", nominal.device)
    if nominal.ndim != 3 or min(nominal.shape[1:]) < 1 or sampled.ndim != 4:
        raise ValueError("Expected nominal(B,H,A) and sampled(B,K,H,A).")
    batch, candidates = sampled.shape[:2]
    if batch != len(nominal) or sampled.shape[2:] != nominal.shape[1:] or candidates < 1:
        raise ValueError("Sampled plans must match nominal batch/horizon/action dimensions.")
    if scores.shape != (batch, candidates) or baseline.shape != (batch,):
        raise ValueError("scores(B,K) and nominal_cost(B) must match sampled plans.")
    if synthesis_mode not in {"mean_verified", "medoid_fallback"}:
        raise ValueError("Unknown synthesis_mode.")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    if cluster_radius is not None and (not math.isfinite(cluster_radius) or cluster_radius <= 0):
        raise ValueError("cluster_radius must be finite and positive.")
    if cluster_radius is not None and synthesis_mode != "medoid_fallback":
        raise ValueError("cluster_radius requires medoid_fallback.")
    low, high = _bounds(low, high, nominal)
    if not bool(((sampled >= low) & (sampled <= high)).all()) or not bool(((nominal >= low) & (nominal <= high)).all()):
        raise ValueError("Nominal and sampled actions must already satisfy the action bounds.")
    row_indices = torch.arange(batch, device=nominal.device)
    temperatures, saturated = temperature_for_neff_fraction_with_details(scores, neff_fraction)
    weights = softmin_weights(scores, temperatures)
    mean = synthesize_plan(nominal, sampled, weights, low, high)

    def checked_verify(plans: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        result = _floating(verify(plans, indices), "verification costs", nominal.device)
        if result.shape != indices.shape:
            raise ValueError("Verification callback must return one scalar cost per supplied row.")
        return result

    mean_costs = checked_verify(mean, row_indices)
    mean_ok = mean_costs < baseline
    rejected, tied = ~mean_ok, mean_costs == baseline
    finite = torch.isfinite(scores)
    expected = (weights * torch.where(finite, scores, 0.)).sum(dim=1)
    collapse = (expected < baseline) & (mean_costs > baseline)
    improving = finite & (scores < baseline[:, None])
    eligible = rejected & improving.any(dim=1)
    plans, costs = nominal.clone(), baseline.clone()
    plans[mean_ok], costs[mean_ok] = mean[mean_ok], mean_costs[mean_ok]
    source = mean_ok.to(torch.int64)
    anchor_indices = torch.full((batch,), -1, device=nominal.device, dtype=torch.int64)
    cluster_counts = torch.zeros(batch, device=nominal.device, dtype=torch.int64)
    verification_rollouts, verification_calls = batch, 1

    if synthesis_mode == "medoid_fallback":
        action_delta_squared = (sampled[:, :, 0] - nominal[:, None, 0]).square().sum(dim=-1)
        objective = torch.where(improving, scores + beta * action_delta_squared, torch.inf)
        if bool((improving & ~torch.isfinite(objective)).any()):
            raise FloatingPointError("Anchor selection objective overflowed.")
        anchor = objective.argmin(dim=1)
        chosen_rows = row_indices[eligible]
        if len(chosen_rows):
            indices = anchor[eligible]
            anchor_plans = sampled[chosen_rows, indices]
            plans[eligible], costs[eligible] = anchor_plans, scores[chosen_rows, indices]
            source[eligible], anchor_indices[eligible] = 2, indices
            if cluster_radius is not None:
                distance = (sampled[eligible] - anchor_plans[:, None]).flatten(2).norm(dim=-1)
                local = improving[eligible] & (distance < cluster_radius)
                cluster_counts[eligible] = local.sum(dim=1)
                local_weights = torch.where(local, weights[eligible], 0.)
                mass = local_weights.sum(dim=1)
                average_rows = (local.sum(dim=1) > 1) & (mass > 0)
                if bool(average_rows.any()):
                    selected_rows = chosen_rows[average_rows]
                    selected_weights = local_weights[average_rows] / mass[average_rows, None]
                    cluster = torch.einsum("bk,bkha->bha", selected_weights, sampled[selected_rows])
                    cluster = torch.maximum(torch.minimum(cluster, high), low)
                    cluster_cost = checked_verify(cluster, selected_rows)
                    verification_rollouts += len(selected_rows)
                    verification_calls += 1
                    cluster_ok = cluster_cost < baseline[selected_rows]
                    accepted_rows = selected_rows[cluster_ok]
                    plans[accepted_rows], costs[accepted_rows] = cluster[cluster_ok], cluster_cost[cluster_ok]
                    source[accepted_rows] = 3

    accepted = costs < baseline
    rescued = (source == 2) | (source == 3)
    return VerifiedSynthesis(
        plans=plans, labels=plans[:, 0].clone(), costs=costs, accepted=accepted,
        source_codes=source, anchor_indices=anchor_indices, mean_plans=mean, mean_costs=mean_costs,
        weights=weights, temperatures=temperatures, n_eff=weights.square().sum(dim=1).reciprocal(),
        temperature_saturated=saturated, expected_candidate_cost=expected,
        improving_candidate_count=improving.sum(dim=1), mean_rejected=rejected, mean_tied=tied,
        bimodal_collapse=collapse, rescue_eligible=eligible, rescued=rescued,
        cluster_candidate_count=cluster_counts, verification_rollouts=verification_rollouts,
        verification_callback_calls=verification_calls)
