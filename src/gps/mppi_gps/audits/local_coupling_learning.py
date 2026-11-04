"""Independent saved-evidence primitives for the bounded L2 learning study.

This module imports no study, learner, planner, runtime or simulator producer.
Checkpoint evaluation uses the deployed CPU Torch linear/tanh action function,
without constructing a policy, restoring RNGs, differentiating or fitting.
The on-disk adapter is added only after the study's evidence contract is frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .trajectory_pi import (
    Audit,
    array_hash,
    audit_ledger,
    audit_stream,
    batch_cost,
    compare_outcome,
    gibbs,
    healthy,
    load_arrays,
    observations,
    outcome,
    policy_hash,
    tensor_arrays,
    typed_fingerprint,
)
from .trajectory_pi import (
    AuditFailure as AuditFailure,
)


class LearningAudit(Audit):
    """Use bounded deployed Torch semantics, with the existing audit row census."""

    def __init__(
        self,
        *,
        seconds: float = 3600,
        cpu_seconds: float = 3600,
        neural_rows: int = 8_000_000,
        output_bytes: int = 2 * 1024**3,
        memory_bytes: int = 4 * 1024**3,
    ):
        super().__init__(
            seconds=seconds,
            neural_rows=neural_rows,
            output_bytes=output_bytes,
            memory_bytes=memory_bytes,
        )
        self.cpu_limit = cpu_seconds
        self._torch_ready = False

    def check(self) -> None:
        super().check()
        if time.process_time() - self.cpu_started > self.cpu_limit:
            raise AuditFailure("Independent audit CPU budget exhausted")

    def predict(self, payload: dict, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float32)
        self.need(
            values.ndim >= 1 and values.shape[-1] == 17 and np.isfinite(values).all(),
            "Finite pre-action 17-dimensional inputs for deployed action map",
        )
        rows = int(np.prod(values.shape[:-1])) if values.ndim > 1 else 1
        self.need(
            self.requested_rows + rows <= self.neural_limit,
            "Independent deployed-head row budget exhausted before inference",
        )
        if not rows:
            return np.empty((*values.shape[:-1], 6), dtype=np.float32)
        self.requested_rows += rows
        try:
            import torch
            from torch.nn import functional as functional

            if not self._torch_ready:
                torch.set_num_threads(1)
                if torch.get_num_interop_threads() != 1:
                    torch.set_num_interop_threads(1)
                self._torch_ready = True
            arrays = tensor_arrays(payload["state_dict"])
            state = {name: torch.as_tensor(value, device="cpu") for name, value in arrays.items()}
            self.need(
                all(value.dtype == torch.float32 for value in state.values()),
                "Unchanged float32 deployed tensors and normalization",
            )
            with torch.no_grad():
                value = torch.as_tensor(values, device="cpu")
                value = (value - state["observation_mean"]) / state["observation_std"]
                for layer in (0, 2, 4):
                    self.check()
                    value = torch.tanh(
                        functional.linear(
                            value,
                            state[f"network.{layer}.weight"],
                            state[f"network.{layer}.bias"],
                        )
                    )
                result = value.detach().cpu().numpy().copy()
            self.need(
                result.shape == (*values.shape[:-1], 6)
                and np.isfinite(result).all()
                and np.all(np.abs(result) <= 1),
                "Finite bounded deployed Torch action with original call shape",
            )
        except BaseException:
            self.unknown_rows += rows
            raise
        self.completed_rows += rows
        return result


def compare_native_episode(
    audit: Audit,
    raw: dict[str, np.ndarray],
    saved: dict,
    horizon: int,
) -> dict:
    """Check saved N+1 contexts, prefix costs, one terminal and missing tail."""
    measured = outcome(audit, raw, horizon)
    compare_outcome(audit, saved, measured)
    return measured


def deployed_reconstruction(
    audit: LearningAudit,
    payload: dict,
    initial: np.ndarray,
    post_states: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute D on the executable prefix, including the fall-causing action."""
    _, mask, valid = batch_cost(post_states, actions)
    initial = np.asarray(initial)
    audit.need(
        initial.shape == (19,) and np.isfinite(initial).all() and healthy(initial),
        "Fixed complete bank begins from a live finite state",
    )
    pre = np.concatenate(
        (np.broadcast_to(initial, (len(post_states), 1, 19)), post_states[:, :-1]), axis=1
    )
    use = mask & valid[:, None]
    prediction = audit.predict(payload, observations(pre[use])).astype(np.float64)
    terms = np.zeros(mask.shape, dtype=np.float64)
    terms[use] = 0.5 * np.sum((actions[use].astype(np.float64) - prediction) ** 2, axis=-1)
    discrepancy = terms.sum(axis=1)
    discrepancy[~valid] = np.inf
    return discrepancy, mask, valid, prediction


def compare_bank(
    audit: Audit,
    *,
    initial_actions: np.ndarray,
    post_actions: np.ndarray,
    candidates: np.ndarray,
    post_states: np.ndarray,
    saved_cost: np.ndarray,
    saved_discrepancy: np.ndarray,
    saved_score: np.ndarray,
    saved_weights: np.ndarray,
    temperature: float,
    alpha: float,
    saved_lengths: np.ndarray,
    recomputed_discrepancy: np.ndarray | None = None,
    discrepancy_atol: float = 1e-6,
) -> dict:
    """Independently audit one retained stock-MPPI bank; impose no survival gate."""
    cost, mask, valid = batch_cost(post_states, candidates)
    k, horizon = mask.shape
    audit.need(
        initial_actions.shape == post_actions.shape == (horizon, 6)
        and np.isfinite(initial_actions).all()
        and np.isfinite(post_actions).all()
        and np.all(np.abs(initial_actions) <= 1)
        and np.all(np.abs(post_actions) <= 1),
        "Incoming and updated plans are finite bounded complete sequences",
    )
    audit.need(
        np.shape(saved_discrepancy) == (k,)
        and np.isfinite(saved_discrepancy[valid]).all()
        and np.all(saved_discrepancy[valid] >= 0),
        "Nonnegative finite-prefix reconstruction roster",
    )
    audit.close(saved_cost, cost, "Native prefix, terminal and absorbing bank cost", atol=1e-9)
    audit.close(saved_lengths, mask.sum(axis=1), "Inclusive first-fall action lengths", atol=0)
    if recomputed_discrepancy is not None:
        audit.close(
            saved_discrepancy,
            recomputed_discrepancy,
            "Deployed pre-action reconstruction with unchanged normalization",
            atol=discrepancy_atol,
        )
    audit.need(np.isfinite(alpha) and alpha >= 0, "Frozen nonnegative scalar agreement weight")
    expected_score = cost if alpha == 0 else cost + alpha * saved_discrepancy
    audit.close(saved_score, expected_score, "Actual total score C plus alpha D", atol=1e-9)
    weights, ess = gibbs(expected_score, temperature)
    audit.close(saved_weights, weights, "Recorded fixed-temperature Gibbs weights", atol=1e-12)
    perturbations = candidates - initial_actions[None]
    updated = np.clip(initial_actions + np.einsum("k,kha->ha", weights, perturbations), -1, 1)
    audit.close(post_actions, updated, "Stock effective-bounded-perturbation update", atol=1e-12)
    survived = healthy(post_states).all(axis=1)
    return dict(
        candidates=k,
        horizon=horizon,
        valid_candidates=int(valid.sum()),
        full_survivors=int(survived.sum()),
        executable_prediction_rows=int((mask & valid[:, None]).sum()),
        effective_sample_size=ess,
        fallen_weight_mass=float(weights[~survived].sum()),
        surviving_weight_mass=float(weights[survived].sum()),
        maximum_weight=float(weights.max()),
        action_displacement=float(np.linalg.norm(post_actions[0] - initial_actions[0])),
        full_plan_displacement=float(np.linalg.norm(post_actions - initial_actions)),
        D_independently_recomputed=recomputed_discrepancy is not None,
    )


class StockSampler:
    """Rebuild the frozen stock proposal from private RNG, without planner imports."""

    def __init__(self, config: dict):
        self.config = config
        self.horizon, self.candidates = int(config["H"]), int(config["K"])
        if config.get("noise_std") is not None or config.get("noise_cov") is not None:
            raise AuditFailure("L2 uses the unchanged isotropic noise_sigma recipe")
        self.spatial = np.linalg.cholesky(float(config["noise_sigma"]) ** 2 * np.eye(6))
        self.temporal = None
        cutoff = config.get("noise_lowpass_cutoff_hz")
        if cutoff is not None:
            from scipy.linalg import solve_discrete_lyapunov, toeplitz
            from scipy.signal import butter, tf2ss

            numerator, denominator = butter(
                config["noise_lowpass_order"],
                cutoff,
                btype="lowpass",
                fs=config["noise_lowpass_sample_rate_hz"],
                output="ba",
            )
            transition, input_map, output_map, feedthrough = tf2ss(numerator, denominator)
            stationary = solve_discrete_lyapunov(transition, input_map @ input_map.T)
            covariance = np.empty(self.horizon)
            covariance[0] = (
                output_map @ stationary @ output_map.T + feedthrough @ feedthrough.T
            ).item()
            previous = np.eye(transition.shape[0])
            for lag in range(1, self.horizon):
                current = transition @ previous
                covariance[lag] = (
                    output_map @ current @ stationary @ output_map.T
                    + output_map @ previous @ input_map @ feedthrough.T
                ).item()
                previous = current
            correlation = toeplitz(covariance / covariance[0])
            correlation = (correlation + correlation.T) * 0.5
            np.fill_diagonal(correlation, 1)
            self.temporal = np.linalg.cholesky(correlation)

    def candidates_from(self, incoming: np.ndarray, rng_before: dict) -> tuple[np.ndarray, dict]:
        if incoming.shape != (self.horizon, 6):
            raise AuditFailure("Complete incoming stock-MPPI plan required")
        sampler = np.random.Generator(np.random.PCG64(0))
        sampler.bit_generator.state = copy.deepcopy(rng_before)
        standard = sampler.standard_normal((self.candidates, self.horizon, 6))
        if self.temporal is not None:
            standard = np.einsum("ht,kta->kha", self.temporal, standard, optimize=True)
        noise = np.einsum("khi,ji->khj", standard, self.spatial)
        return np.clip(incoming[None] + noise, -1, 1), copy.deepcopy(sampler.bit_generator.state)


def compare_bank_statistics(
    audit: Audit,
    sampler: StockSampler,
    *,
    incoming: np.ndarray,
    updated: np.ndarray,
    cost: np.ndarray,
    discrepancy: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    lengths: np.ndarray,
    temperature: float,
    alpha: float,
    rng_before: dict,
    rng_after: dict,
) -> dict:
    """Check every recorded update; physical state/D coverage is sparse and explicit."""
    candidates, expected_rng = sampler.candidates_from(incoming, rng_before)
    audit.equal(rng_after, expected_rng, "Actual stock-noise private RNG transition")
    audit.need(
        all(
            np.shape(value) == (sampler.candidates,)
            for value in (cost, discrepancy, scores, weights, lengths)
        )
        and np.all((lengths >= 1) & (lengths <= sampler.horizon)),
        "Every candidate score and executable-prefix length is present",
    )
    finite = np.isfinite(cost)
    audit.need(
        np.all(np.isfinite(discrepancy[finite])) and np.all(discrepancy[finite] >= 0),
        "Finite reconstruction for finite native candidates",
    )
    expected_scores = cost if alpha == 0 else cost + alpha * discrepancy
    audit.close(scores, expected_scores, "Every bank uses actual C plus alpha D", atol=1e-9)
    expected_weights, ess = gibbs(expected_scores, temperature)
    audit.close(weights, expected_weights, "Every bank's fixed-temperature weights", atol=1e-12)
    expected_updated = np.clip(
        incoming + np.einsum("k,kha->ha", weights, candidates - incoming[None]), -1, 1
    )
    audit.close(updated, expected_updated, "Every carried-plan bounded update", atol=1e-12)
    return dict(
        candidates=candidates,
        effective_sample_size=ess,
        reconstructed_standard_normal_values=sampler.candidates * sampler.horizon * 6,
        physics_or_discrepancy_recomputed=False,
    )


def common_prefix_corpora(
    audit: Audit,
    independent: Mapping[str, dict[str, np.ndarray]],
    fixed: Mapping[str, dict[str, np.ndarray]],
    restart_order: Sequence[str],
) -> dict:
    """Join all paired executed source events without resampling or outcome selection."""
    order = list(restart_order)
    audit.need(
        len(order) == len(set(order)) and set(independent) == set(fixed) == set(order),
        "Every frozen restart contributes to the equal-prefix corpus",
    )
    values = {arm: dict(observations=[], actions=[]) for arm in ("independent", "fixed")}
    row_events, lengths = [], []
    for restart_index, name in enumerate(order):
        pair = {"independent": independent[name], "fixed": fixed[name]}
        count = min(len(raw["actions"]) for raw in pair.values())
        audit.need(count > 0, "No missing or fabricated restart prefix is fitting data")
        lengths.append(dict(restart=name, rows=count))
        for arm, raw in pair.items():
            states, actions = raw["states"], raw["actions"]
            audit.need(
                states.shape == (len(actions) + 1, 19)
                and actions.shape[1:] == (6,)
                and np.isfinite(states).all()
                and np.isfinite(actions).all()
                and np.all(np.abs(actions) <= 1),
                "Bounded executed N+1 source corpus: " + arm,
            )
            obs = observations(states[:count])
            if "observations" in raw:
                audit.close(
                    raw["observations"][:count], obs, "Exact pre-action source rows", atol=0
                )
            values[arm]["observations"].append(obs.astype(np.float64))
            values[arm]["actions"].append(actions[:count].astype(np.float64))
        row_events.extend((restart_index, control) for control in range(count))
    for arm in values.values():
        for key in arm:
            arm[key] = np.concatenate(arm[key])
    return dict(
        arms=values,
        row_events=np.asarray(row_events, dtype=np.int64),
        per_restart=lengths,
        rows=len(row_events),
    )


def check_paired_incoming(audit: Audit, independent: dict, fixed: dict) -> None:
    """R1 pair equality; across-round arms may differ and retain their own states."""
    for key in ("state_dict", "optimizer_state", "private_rng_state"):
        audit.equal(independent[key], fixed[key], "Paired incoming checkpoint identity: " + key)


def check_continuation(
    audit: Audit,
    *,
    parent: dict,
    child: dict,
    initial: dict,
    parent_globals: dict,
    child_globals: dict,
    indices: np.ndarray,
    rng_rows: list[dict],
    corpus_rows: int,
    updates: int,
    batch_size: int = 256,
) -> dict:
    """Check actual continuation proofs without reproducing any gradient arithmetic."""
    old, new = parent["optimizer_state"], child["optimizer_state"]
    audit.need(corpus_rows > 0 and updates > 0, "Positive frozen corpus and update budget")
    audit.equal(initial["state_dict"], parent["state_dict"], "Actual incoming actor tensors")
    audit.equal(initial["adam"], old["adam"], "Actual incoming Adam moments")
    audit.equal(initial["rng"], old["rng"], "Actual incoming private PCG")
    audit.equal(
        initial["private_rng_state"],
        parent["private_rng_state"],
        "Actual incoming private Torch RNG",
    )
    audit.equal(
        child["private_rng_state"], parent["private_rng_state"], "Private Torch continuation"
    )
    audit.equal(child_globals, parent_globals, "Deterministic fit retains full global RNG envelope")
    for name in ("observation_mean", "observation_std"):
        audit.equal(child["state_dict"][name], parent["state_dict"][name], "Frozen normalization")
    audit.need(
        indices.dtype == np.int64
        and indices.shape == (updates, batch_size)
        and len(rng_rows) == updates,
        "All actual minibatch indices and RNG transitions preserved",
    )
    sampler = np.random.Generator(np.random.PCG64(0))
    sampler.bit_generator.state = old["rng"]
    for i, row in enumerate(rng_rows):
        audit.equal(
            row["rng_before"], sampler.bit_generator.state, "Incoming minibatch private PCG"
        )
        expected = sampler.integers(0, corpus_rows, size=batch_size)
        audit.equal(indices[i], expected, "Exact private-PCG minibatch draw")
        audit.equal(row["rng_after"], sampler.bit_generator.state, "Outgoing minibatch private PCG")
        if "indices_fingerprint" in row:
            audit.need(
                row["indices_fingerprint"] == typed_fingerprint(expected), "Minibatch binding"
            )
    audit.equal(new["rng"], sampler.bit_generator.state, "Saved child PCG continues actual tape")
    audit.need(new["updates"] == old["updates"] + updates, "Fixed cumulative Adam update count")
    audit.equal(new["adam"]["param_groups"], old["adam"]["param_groups"], "Unchanged Adam groups")
    audit.need(set(new["adam"]["state"]) == set(old["adam"]["state"]), "Unchanged Adam slots")
    for key, after in new["adam"]["state"].items():
        before = old["adam"]["state"][key]
        audit.need(set(after) == {"step", "exp_avg", "exp_avg_sq"}, "Existing Adam state schema")
        audit.close(float(after["step"]), float(before["step"]) + updates, "Adam cursor", atol=0)
        for slot in ("exp_avg", "exp_avg_sq"):
            actual = tensor_arrays({slot: after[slot]})[slot]
            previous = tensor_arrays({slot: before[slot]})[slot]
            audit.need(
                actual.shape == previous.shape
                and actual.dtype == previous.dtype == np.float32
                and np.isfinite(actual).all(),
                "Finite continued Adam moments without reset: " + slot,
            )
        audit.need(
            np.all(tensor_arrays({"x": after["exp_avg_sq"]})["x"] >= 0), "Adam second moments"
        )
    audit.need(policy_hash(child["state_dict"]) == child["policy_hash"], "Child policy fingerprint")
    return dict(
        updates=updates,
        minibatch_fingerprint=typed_fingerprint(indices),
        incoming_optimizer_fingerprint=typed_fingerprint(old),
        outgoing_optimizer_fingerprint=typed_fingerprint(new),
        gradient_arithmetic_reproduced=False,
    )


def local_teacher_summary(outcomes: Sequence[dict]) -> dict:
    """Three-hundred-control task competence is not standalone walking."""
    if not outcomes:
        raise AuditFailure("Missing local teacher outcomes")
    costs = [float(row["complete_cost"]) for row in outcomes]
    if not np.isfinite(costs).all():
        raise AuditFailure("Finite local teacher complete costs required")
    return dict(
        episodes=len(outcomes),
        local_walking=sum(
            bool(row["full_survival"]) and float(row["velocity_mae"]) <= 0.3 for row in outcomes
        ),
        full_survival=sum(bool(row["full_survival"]) for row in outcomes),
        mean_complete_cost=float(np.mean(costs)),
    )


def select_teacher_arm(
    independent: dict,
    weak: dict,
    large: dict,
    action_rms: Mapping[str, float],
) -> dict:
    """Frozen competence/action-effect selection; no reconstruction or spectra input."""
    if any(row["episodes"] != 8 for row in (independent, weak, large)):
        raise AuditFailure("All 24 R1 acquisition episodes must close before selection")
    baseline_qualified = independent["local_walking"] >= 6
    arms = {}
    for name, candidate in (("weak", weak), ("large", large)):
        rms = float(action_rms[name])
        if not np.isfinite(rms) or rms < 0:
            raise AuditFailure("Finite common-prefix executed-action RMS required")
        gates = dict(
            independent_competent=baseline_qualified,
            candidate_competent=candidate["local_walking"] >= 6,
            local_walking_nonregression=(
                candidate["local_walking"] >= independent["local_walking"]
            ),
            full_survival_nonregression=(
                candidate["full_survival"] >= independent["full_survival"]
            ),
            complete_cost_noninferior=(
                candidate["mean_complete_cost"] <= 1.10 * independent["mean_complete_cost"]
            ),
            action_effect=rms >= 0.01,
        )
        arms[name] = dict(passed=all(gates.values()), gates=gates, action_rms=rms)
    selected = "weak" if arms["weak"]["passed"] else "large" if arms["large"]["passed"] else None
    return dict(selected=selected, arms=arms, independent_qualified=baseline_qualified)


def _panel_rows(panels: dict) -> tuple[list[str], list[str]]:
    names = ["primary", "confirmation"]
    if set(panels) != set(names):
        raise AuditFailure("Both fixed primary and disjoint confirmation panels required")
    streams = sorted(panels["primary"])
    if len(streams) != 3 or set(panels["confirmation"]) != set(streams):
        raise AuditFailure("Three paired minibatch streams on both panels required")
    for name in names:
        for row in panels[name].values():
            for panel in row.values():
                if not (
                    panel["episodes"] == 16
                    and 0 <= panel["walking"] <= panel["full_survival"] <= 16
                ):
                    raise AuditFailure(
                        "Every standalone panel contains sixteen consistent outcomes"
                    )
    return names, streams


def r1_walking_decision(panels: dict) -> dict:
    """Confirm the same streams on both panels, separately from retention/fit."""
    names, streams = _panel_rows(panels)
    per_stream = {}
    for stream in streams:
        entries = {}
        for name in names:
            row = panels[name][stream]
            parent, independent, positive = (
                row[key] for key in ("parent", "independent", "positive")
            )
            nonregression = all(
                positive[key] >= reference[key]
                for key in ("walking", "full_survival")
                for reference in (parent, independent)
            )
            entries[name] = dict(
                walking_vs_independent=positive["walking"] - independent["walking"],
                walking_vs_parent=positive["walking"] - parent["walking"],
                nonregression=nonregression,
                passed=(
                    nonregression
                    and positive["walking"] >= independent["walking"] + 2
                    and positive["walking"] >= parent["walking"] + 2
                ),
                individual_competence=positive["walking"] >= 15,
            )
        per_stream[stream] = dict(
            panels=entries,
            passed=all(row["passed"] for row in entries.values()),
            individual_competence=all(row["individual_competence"] for row in entries.values()),
        )
    joint = [stream for stream in streams if per_stream[stream]["passed"]]
    nonregression = all(
        row["nonregression"] for stream in per_stream.values() for row in stream["panels"].values()
    )
    return dict(
        passed=len(joint) >= 2 and nonregression,
        joint_streams=joint,
        per_stream=per_stream,
        all_stream_nonregression=nonregression,
        fit_improvement_is_walking=False,
    )


def r2_walking_decision(panels: dict) -> dict:
    """Use fresh reset-matched incoming actors for the difference in differences."""
    names, streams = _panel_rows(panels)
    per_stream = {}
    for stream in streams:
        entries = {}
        for name in names:
            row = panels[name][stream]
            parent_positive, parent_zero, positive, zero = (
                row[key]
                for key in ("parent_positive", "parent_independent", "positive", "independent")
            )
            gain_positive = positive["walking"] - parent_positive["walking"]
            gain_zero = zero["walking"] - parent_zero["walking"]
            difference_in_differences = gain_positive - gain_zero
            nonregression = all(
                positive[key] >= reference[key]
                for key in ("walking", "full_survival")
                for reference in (parent_positive, zero)
            )
            entries[name] = dict(
                own_walking_increment=gain_positive,
                independent_walking_increment=gain_zero,
                difference_in_differences=difference_in_differences,
                walking_vs_independent=positive["walking"] - zero["walking"],
                nonregression=nonregression,
                passed=(
                    nonregression
                    and gain_positive >= 2
                    and difference_in_differences >= 2
                    and positive["walking"] >= zero["walking"] + 2
                ),
            )
        per_stream[stream] = dict(
            panels=entries, passed=all(row["passed"] for row in entries.values())
        )
    joint = [stream for stream in streams if per_stream[stream]["passed"]]
    nonregression = all(
        row["nonregression"] for stream in per_stream.values() for row in stream["panels"].values()
    )
    return dict(
        passed=len(joint) >= 2 and nonregression,
        joint_streams=joint,
        per_stream=per_stream,
        all_stream_nonregression=nonregression,
        frozen_old_actor_control_present=False,
        interpretation="Bounded feedback-associated evidence; extra fitting is not isolated",
    )


def audit_corpus(audit: Audit, manifest_path: Path) -> dict:
    """Rebuild each stored physical row from the two bound eight-episode rosters."""
    manifest = audit.json(manifest_path)
    audit.need(manifest["schema"] == "local-coupling-corpora-1", "Paired corpus schema")
    cases = manifest["cases"]
    audit.need([row["slot"] for row in cases] == list(range(8)), "All eight ordered corpus cases")
    source = {"independent": {}, "fixed": {}}
    for row in cases:
        raw = [load_arrays(audit, audit.bound(spec)) for spec in row["traces"]]
        for key in ("states", "warmstarts", "integrations"):
            audit.equal(raw[0][key][0], raw[1][key][0], "Paired acquisition initial context")
        audit.equal(
            row["controls"], [len(value["actions"]) for value in raw], "Source prefix lengths"
        )
        audit.need(row["common"] == min(row["controls"]) > 0, "Every restart contributes labels")
        source["independent"][str(row["slot"])], source["fixed"][str(row["slot"])] = raw
    rebuilt = common_prefix_corpora(
        audit, source["independent"], source["fixed"], [str(i) for i in range(8)]
    )
    audit.need(
        manifest["rows"] == rebuilt["rows"]
        and manifest["row_ids_fingerprint"] == typed_fingerprint(rebuilt["row_events"]),
        "Exact common-prefix corpus count and event hash",
    )
    for name, arm in (("a", "independent"), ("b", "fixed")):
        raw = load_arrays(audit, audit.bound(manifest["files"][name]))
        audit.equal(raw["row_ids"], rebuilt["row_events"], "Every fitting source event")
        for key in ("observations", "actions"):
            audit.equal(raw[key], rebuilt["arms"][arm][key], "Exact physical fitting rows: " + key)
        expected_states = np.concatenate(
            [source[arm][str(case["slot"])]["states"][: case["common"]] for case in cases]
        )
        audit.equal(raw["states"], expected_states, "Exact physical pre-action corpus states")
    return manifest


def _closed_job(audit: Audit, directory: Path) -> dict:
    ledger = audit_ledger(audit, directory)
    audit.need(
        ledger["receipt"]["status"] == "COMPLETED"
        and ledger["failed"] == ledger["rejected"] == 0
        and not any(
            key.startswith("unknown_partial_") and value for key, value in ledger["counts"].items()
        ),
        "Closed successful phase with no unknown or rejected paid work",
    )
    return ledger


def audit_diagnostics(audit: LearningAudit, payload: dict, roster: dict) -> dict:
    measured = {}
    for name, row in roster.items():
        raw = load_arrays(audit, audit.bound(row["data"]))
        predictions = load_arrays(audit, audit.bound(row["predictions"]))["predictions"]
        obs, target = raw["observations"], raw["actions"]
        audit.need(row["torch_prediction_batch_size"] == 1024, "Preserved diagnostic call grouping")
        expected = np.concatenate(
            [audit.predict(payload, obs[i : i + 1024]) for i in range(0, len(obs), 1024)]
        )
        audit.close(
            predictions, expected, "Saved diagnostic deployed bounded predictions", atol=1e-6
        )
        residual = predictions.astype(np.float64) - target
        mse = float(np.mean(residual**2))
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        audit.need(row["rows"] == len(obs), "All diagnostic rows")
        audit.close(row["action_mse"], mse, "Saved diagnostic MSE", atol=1e-12)
        audit.close(row["rms_l2"], rms, "Saved diagnostic RMS-L2", atol=1e-12)
        measured[name] = dict(rows=len(obs), action_mse=mse, rms_l2=rms)
    return measured


def audit_fit_job(audit: LearningAudit, directory: Path, expected: dict, config: dict) -> dict:
    """Bind the actual fitter proof/tape/checkpoints, without redoing optimization."""
    job, result = audit.json(directory / "job.json"), audit.json(directory / "result.json")
    audit.need(
        result["schema"] == "local-coupling-fit-1" and result["status"] == "COMPLETED",
        "Closed fit result",
    )
    for key in ("round", "stream", "arm", "config", "incoming", "corpus"):
        audit.equal(job[key], result[key], "Actual fit job/result binding: " + key)
        if key in expected:
            audit.equal(job[key], expected[key], "Frozen fit parent/corpus/role: " + key)
    incoming = audit.bound(result["incoming"])
    parent, parent_envelope = audit_stream(audit, incoming.parent)
    outgoing = audit.bound(result["outgoing"])
    audit.need(
        outgoing.resolve() == (directory / "stream/stream.json").resolve(),
        "Actual child stream location",
    )
    child, child_envelope = audit_stream(audit, outgoing.parent)
    audit.equal(
        child_envelope["metadata"],
        dict(
            round=result["round"],
            stream=result["stream"],
            arm=result["arm"],
            config_sha256=result["config"]["sha256"],
            fit_updates=config["learning"]["updates"],
            incoming=result["incoming"],
        ),
        "Actual retained child owns the declared round, arm, fork and incoming stream",
    )
    raw = load_arrays(audit, audit.bound(result["corpus"]))
    for key, filename in (
        ("initial_state", "initial_state.pt"),
        ("fit", "fit.json"),
        ("index_tape", "index_tape.npz"),
        ("index_rng", "index_rng.jsonl"),
    ):
        audit.need(
            audit.bound(result[key]).resolve() == (directory / filename).resolve(),
            "Actual fit proof binding: " + key,
        )
    initial, fit = audit.package(directory / "initial_state.pt"), audit.json(directory / "fit.json")
    tape = load_arrays(audit, directory / "index_tape.npz")
    rng_rows = [
        json.loads(line) for line in (directory / "index_rng.jsonl").read_text().splitlines()
    ]
    updates = config["learning"]["updates"]
    continuation = check_continuation(
        audit,
        parent=parent,
        child=child,
        initial=initial,
        parent_globals=audit.package(incoming.parent / "global_rng.pt"),
        child_globals=audit.package(outgoing.parent / "global_rng.pt"),
        indices=tape["indices"],
        rng_rows=rng_rows,
        corpus_rows=len(raw["actions"]),
        updates=updates,
    )
    audit.equal(
        tape["updates"],
        np.arange(
            parent["optimizer_state"]["updates"] + 1,
            child["optimizer_state"]["updates"] + 1,
            dtype=np.int64,
        ),
        "Every cumulative minibatch cursor",
    )
    audit.equal(
        tape["fit_updates"],
        np.arange(1, updates + 1, dtype=np.int64),
        "Every new corpus minibatch cursor",
    )
    for i, row in enumerate(rng_rows):
        audit.need(
            row["update"] == int(tape["updates"][i]) and row["fit_update"] == i + 1,
            "RNG proof cumulative cursor",
        )
    settings = {
        key: config["learning"][key]
        for key in (
            "batch_size",
            "lr",
            "eta_theta",
            "gradient_clip",
            "prox_reduction",
            "anchor_refresh",
        )
    }
    state = child["optimizer_state"]
    audit.need(
        fit["parent_policy_hash"] == state["parent_policy_hash"] == parent["policy_hash"]
        and fit["child_policy_hash"] == state["policy_hash"] == child["policy_hash"]
        and fit["anchor_policy_hash"] == state["anchor_policy_hash"] == parent["policy_hash"]
        and policy_hash(state["anchor"]) == parent["policy_hash"]
        and parent["optimizer_state"]["policy_hash"] == parent["policy_hash"],
        "Actual checkpoint, Adam owner and new-corpus anchor lineage",
    )
    audit.equal(fit["fit_config"], settings, "Unchanged fitter settings")
    audit.equal(state["fit_config"], settings, "Unchanged saved optimizer settings")
    audit.need(
        fit["schema"] == "walker-adam-proximal-fit-v1"
        and fit["optimizer_continued"] is True
        and fit["resume_fit"] is False
        and fit["failed"] is False,
        "Existing continuation fitting procedure",
    )
    audit.need(
        fit["completed_updates"] == fit["fit_updates"] == result["updates"] == updates
        and fit["cumulative_updates"]
        == result["cumulative_updates"]
        == config.get("initial_cumulative_updates", 2000) + updates * result["round"],
        "Fixed fit budget and actual cumulative endpoint",
    )
    fit_data = dict(
        observations_sha256=array_hash(raw["observations"].astype(np.float32)),
        actions_sha256=array_hash(raw["actions"].astype(np.float32)),
        rows=len(raw["actions"]),
    )
    audit.equal(fit["fit_data"], fit_data, "Exact FP32 fitter corpus conversion")
    audit.equal(state["fit_data"], fit_data, "Child checkpoint binds actual new corpus")
    for role, payload, envelope in (
        ("before", parent, parent_envelope),
        ("after", child, child_envelope),
    ):
        expected_proof = dict(
            policy_hash=payload["policy_hash"],
            optimizer_fingerprint=typed_fingerprint(payload["optimizer_state"]),
            normalization=typed_fingerprint(
                {
                    "mean": tensor_arrays(payload["state_dict"])["observation_mean"],
                    "std": tensor_arrays(payload["state_dict"])["observation_std"],
                }
            ),
            private_torch=typed_fingerprint(payload["private_rng_state"]),
            global_rng=envelope["global_rng_fingerprint"],
        )
        audit.equal(result[role], expected_proof, "Actual fit/reload tensor and RNG proof: " + role)
    expected_work = {
        unit: dict(requested=value, completed=value, unknown_partial=0)
        for unit, value in (
            ("neural_rows", updates * 256),
            ("backward_rows", updates * 256),
            ("optimizer_steps", updates),
        )
    }
    audit.equal(fit["work"], expected_work, "Complete fixed learning work census")
    audit.need(
        len(fit["learning_curve"]) == updates
        and fit["parameter_count"] == fit["proximal_divisor"] == 19590
        and not fit["anchor_refreshes"],
        "Every loss and unchanged architecture",
    )
    for i, row in enumerate(fit["learning_curve"]):
        audit.need(
            row["fit_update"] == i + 1
            and row["update"] == parent["optimizer_state"]["updates"] + i + 1
            and row["proximal_loss"] == 0
            and row["total_loss"] == row["action_mse"]
            and all(
                np.isfinite(row[key]) and row[key] >= 0
                for key in ("action_mse", "total_loss", "unclipped_gradient_norm")
            ),
            "Finite ordinary-MSE learning curve",
        )
    diagnostics = dict(
        before=audit_diagnostics(audit, parent, result["diagnostics_before"]),
        after=audit_diagnostics(audit, child, result["diagnostics_after"]),
    )
    ledger = _closed_job(audit, directory)
    for key in ("native_controls", "model_controls", "physics_substeps"):
        audit.need(ledger["counts"].get("requested_" + key, 0) == 0, "No physical work in fitting")
    return dict(
        parent=parent,
        child=child,
        incoming=str(incoming),
        outgoing=str(outgoing),
        continuation=continuation,
        diagnostics=diagnostics,
        ledger=ledger,
        indices=tape["indices"],
    )


def audit_evaluation_job(audit: LearningAudit, directory: Path, expected: dict) -> dict:
    job, result = audit.json(directory / "job.json"), audit.json(directory / "result.json")
    audit.need(
        result["schema"] == "local-coupling-evaluation-1" and result["status"] == "COMPLETED",
        "Closed standalone evaluation result",
    )
    for key in ("config", "checkpoint", "reset_seed"):
        audit.equal(job[key], result[key], "Actual standalone job binding: " + key)
        if key in expected:
            audit.equal(job[key], expected[key], "Frozen standalone actor/reset: " + key)
    payload = audit.checkpoint(audit.bound(result["checkpoint"]))
    audit.need(result["policy_hash"] == payload["policy_hash"], "Actual direct-execution actor")
    raw = load_arrays(audit, audit.bound(result["trace"]))
    measured = compare_native_episode(audit, raw, result["outcome"], 1000)
    for control in range(len(raw["actions"])):
        prediction = audit.predict(payload, raw["observations"][control]).astype(np.float64)
        audit.close(
            raw["actions"][control], prediction, "Direct bounded deployed Torch action", atol=1e-6
        )
    context = {
        key: raw[plural][0]
        for key, plural in (
            ("state", "states"),
            ("warmstart", "warmstarts"),
            ("integration", "integrations"),
        )
    }
    audit.need(
        result["initial_context_fingerprint"] == typed_fingerprint(context),
        "Actual complete reset-context fingerprint",
    )
    ledger = _closed_job(audit, directory)
    for unit, value in (
        ("native_controls", len(raw["actions"])),
        ("neural_rows", len(raw["actions"])),
        ("physics_substeps", 4 * len(raw["actions"])),
    ):
        audit.need(
            ledger["counts"].get("completed_" + unit, 0) == value,
            "Complete standalone work: " + unit,
        )
    audit.need(
        ledger["counts"].get("requested_model_controls", 0) == 0,
        "Standalone execution has no MPPI assistance",
    )
    return dict(
        outcome=measured, context=context, policy_hash=payload["policy_hash"], ledger=ledger
    )


def compare_native_discrepancy(
    audit: Audit,
    *,
    actions: np.ndarray,
    native_predictions: np.ndarray,
    bank_predictions: np.ndarray,
    bank_discrepancy: float,
    saved_action_error: float,
    saved_D_tolerance: float,
) -> dict:
    """Propagate the fixed per-action tolerance through bounded squared residuals."""
    audit.need(
        actions.ndim == 2
        and actions.shape[1] == 6
        and 1 <= len(actions) <= 48
        and native_predictions.shape == bank_predictions.shape == actions.shape
        and all(
            np.isfinite(value).all() and np.all(np.abs(value) <= 1)
            for value in (actions, native_predictions, bank_predictions)
        ),
        "Bounded finite actual-prefix actions and native/source-bank predictions",
    )
    action_error = float(np.max(np.abs(native_predictions - bank_predictions)))
    tolerance = 1e-10 + 2 * actions.size * 1e-6
    native_D = float(0.5 * np.sum((actions - native_predictions) ** 2))
    source_D = float(0.5 * np.sum((actions - bank_predictions) ** 2))
    audit.need(action_error <= 1e-6, "Native/source-bank per-action prediction tolerance")
    audit.close(
        saved_action_error, action_error, "Saved native/source-bank action maximum", atol=1e-15
    )
    audit.close(
        saved_D_tolerance, tolerance, "Declared propagated bounded-residual D tolerance", atol=0
    )
    audit.close(
        bank_discrepancy,
        source_D,
        "Source-bank prediction slice reconstructs its original D",
        atol=1e-10,
    )
    audit.close(
        native_D,
        bank_discrepancy,
        "Native/source-bank D within propagated action-error bound",
        atol=tolerance,
    )
    return dict(
        controls=len(actions),
        action_max_error=action_error,
        native_D=native_D,
        bank_D=float(bank_discrepancy),
        D_absolute_error=abs(native_D - bank_discrepancy),
        D_tolerance=tolerance,
        action_tolerance=1e-6,
        accumulation_tolerance=1e-10,
    )


def audit_acquisition(
    audit: LearningAudit,
    directory: Path,
    expected: dict,
    config: dict,
    *,
    result_override: dict | None = None,
    native_prediction_checker=None,
    closed_job_checker=None,
) -> dict:
    """Audit every stock update and the declared sparse full-physics-state roster.

    Optional hooks are reserved for an independently certified numerical
    validation amendment. Defaults retain the original strict L2/L3.0 contract;
    callers must bind and audit any derived result before supplying it here.
    """
    import h5py

    job, saved_result = audit.json(directory / "job.json"), audit.json(directory / "manifest.json")
    result = saved_result if result_override is None else result_override
    audit.need(
        result["schema"] == "local-coupling-episode-1" and result["status"] == "COMPLETED",
        "Closed controller episode, including physical falls",
    )
    for key in ("round", "stream", "arm", "slot", "alpha", "seed", "checkpoint"):
        audit.equal(job[key], result[key], "Acquisition role/actor binding: " + key)
        if key in expected:
            audit.equal(job[key], expected[key], "Frozen acquisition role/actor: " + key)
    audit.equal(
        job["planner"], config["controller"]["planner"], "Unchanged standard local MPPI recipe"
    )
    payload = audit.checkpoint(audit.bound(job["checkpoint"]))
    audit.need(result["actor_hash"] == payload["policy_hash"], "Fixed actual acquisition actor")
    restarts = load_arrays(audit, audit.bound(job["restarts"]))
    context = {
        key: restarts[f"r{job['slot']}_{key}"]
        for key in ("state", "warmstart", "integration", "spec", "dimensions")
    }
    audit.need(
        result["initial_context_fingerprint"] == typed_fingerprint(context),
        "Complete frozen restart context",
    )
    raw = load_arrays(audit, audit.bound(result["trace"]))
    measured = compare_native_episode(audit, raw, result["outcome"], 300)
    audit.need(
        result["outcome"]["local_walking"]
        == (measured["full_survival"] and measured["velocity_mae"] <= 0.3),
        "Local walking is separate from standalone walking",
    )
    for key, plural in (
        ("state", "states"),
        ("warmstart", "warmstarts"),
        ("integration", "integrations"),
    ):
        audit.equal(raw[plural][0], context[key], "Actual complete restored initial boundary")
    sampler = StockSampler(job["planner"])
    rng = np.random.Generator(np.random.PCG64(job["seed"])).bit_generator.state
    incoming = np.zeros((sampler.horizon, 6), np.float64)
    n = len(raw["actions"])
    expected_calls = [(0, 0), (0, 1)] + [(t, 0) for t in range(1, n)]
    bank_path = audit.bound(result["bank_stats"])
    rows = full_banks = prediction_rows = 0
    full_files = {}
    with h5py.File(bank_path, "r") as source:
        audit.need(
            list(source) == [f"{i:06d}" for i in range(len(expected_calls))],
            "All actual refinement calls, including both initial passes",
        )
        for bank_index, (control, inner) in enumerate(expected_calls):
            group = source[f"{bank_index:06d}"]
            arrays = {name: group[name][:] for name in group}
            attrs = dict(group.attrs)
            survival = arrays["full_survival"]
            audit.need(
                survival.dtype == np.bool_
                and survival.shape == (sampler.candidates,)
                and np.all(~survival | (arrays["lengths"] == sampler.horizon)),
                "Every bank has explicit boolean full-survival evidence, including last-control falls",
            )
            audit.close(
                attrs["survival_fraction"],
                float(survival.mean()),
                "Every bank's explicit full-survival fraction",
                atol=0,
            )
            audit.close(
                attrs["fallen_weight"],
                float(arrays["weights"][~survival].sum()),
                "Every bank's explicit fallen weight mass",
                atol=1e-12,
            )
            audit.need(
                int(attrs["control"]) == control and int(attrs["inner"]) == inner,
                "Control versus refinement indexing",
            )
            audit.need(
                attrs["actor_hash"] == payload["policy_hash"]
                and attrs["alpha"] == job["alpha"]
                and attrs["lambda_value"] == 3.0,
                "Fixed policy, scalar alpha and actual standard temperature",
            )
            for key, plural in (
                ("initial_state", "states"),
                ("initial_warmstart", "warmstarts"),
                ("initial_integration", "integrations"),
            ):
                audit.equal(
                    arrays[key],
                    raw[plural][control],
                    "Bank uses the actual pre-action native context",
                )
            before, after = json.loads(attrs["rng_before"]), json.loads(attrs["rng_after"])
            audit.equal(before, rng, "Private planner stream carries between real refinements")
            audit.equal(
                arrays["U_before"], incoming, "Full plan carries without extra horizon shifts"
            )
            stats = compare_bank_statistics(
                audit,
                sampler,
                incoming=arrays["U_before"],
                updated=arrays["U_after"],
                cost=arrays["C"],
                discrepancy=arrays["D"],
                scores=arrays["J"],
                weights=arrays["weights"],
                lengths=arrays["lengths"],
                temperature=attrs["lambda_value"],
                alpha=attrs["alpha"],
                rng_before=before,
                rng_after=after,
            )
            audit.close(
                attrs["ESS"], stats["effective_sample_size"], "Saved actual bank ESS", atol=1e-10
            )
            prediction_rows += int(arrays["lengths"][np.isfinite(arrays["C"])].sum())
            capture = (control == 0) or (control in (149, 299) and inner == 0)
            audit.need(
                ("full_bank" in attrs) == capture, "Exactly the declared reached full-bank roster"
            )
            if capture:
                path = directory / attrs["full_bank"]
                bank = load_arrays(audit, path)
                audit.equal(
                    bank["full_survival"],
                    healthy(bank["posts"]).all(axis=1),
                    "Full-bank survival vector matches physical health through the final control",
                )
                for name, value in arrays.items():
                    audit.equal(
                        bank[name], value, "Full-bank and all-call statistics identity: " + name
                    )
                audit.equal(
                    bank["actions"],
                    stats["candidates"],
                    "Actual sampled actions from frozen stock RNG",
                )
                discrepancy, mask, valid, expected_predictions = deployed_reconstruction(
                    audit, payload, arrays["initial_state"], bank["posts"], bank["actions"]
                )
                audit.equal(bank["mask"], mask, "Inclusive first-fall physical prefix mask")
                use = mask & valid[:, None]
                saved_predictions = bank["predictions"]
                audit.need(
                    saved_predictions.shape == (int(use.sum()), 6),
                    "Saved deployed predictions contain only valid executable rows",
                )
                audit.close(
                    saved_predictions,
                    expected_predictions,
                    "Actual saved pre-action deployed outputs",
                    atol=1e-6,
                )
                terms = np.zeros(mask.shape)
                terms[use] = 0.5 * np.sum(
                    (bank["actions"][use] - saved_predictions.astype(np.float64)) ** 2, axis=1
                )
                from_predictions = terms.sum(1)
                from_predictions[~valid] = np.inf
                audit.close(
                    bank["D"],
                    from_predictions,
                    "D is built from the saved bounded deployed outputs",
                    atol=1e-10,
                )
                bank_stats = compare_bank(
                    audit,
                    initial_actions=arrays["U_before"],
                    post_actions=arrays["U_after"],
                    candidates=bank["actions"],
                    post_states=bank["posts"],
                    saved_cost=arrays["C"],
                    saved_discrepancy=arrays["D"],
                    saved_score=arrays["J"],
                    saved_weights=arrays["weights"],
                    temperature=attrs["lambda_value"],
                    alpha=attrs["alpha"],
                    saved_lengths=arrays["lengths"],
                    recomputed_discrepancy=discrepancy,
                )
                audit.close(
                    attrs["survival_fraction"],
                    bank_stats["full_survivors"] / sampler.candidates,
                    "Saved full-survival fraction",
                    atol=0,
                )
                audit.close(
                    attrs["fallen_weight"],
                    bank_stats["fallen_weight_mass"],
                    "Saved fallen weight mass",
                    atol=1e-12,
                )
                full_files[str(path.resolve())] = bank
                full_banks += 1
            if inner == 1 or control > 0:
                audit.equal(
                    raw["actions"][control],
                    arrays["U_after"][0],
                    "Executed action is the last refined plan's first control",
                )
                incoming = np.concatenate((arrays["U_after"][1:], arrays["U_after"][-1:]))
            else:
                incoming = arrays["U_after"]
            rng = after
            rows += 1
    audit.need(
        result["banks"] == rows == n + 1,
        "Actual native decisions plus one initial extra refinement",
    )
    replay_controls = replay_prediction_rows = 0
    native_discrepancy_checks = []
    checks = result["native_candidate_checks"]
    audit.need(
        [row["candidate"] for row in checks] == [0, 255],
        "Declared first-bank candidate verification roster",
    )
    for row in checks:
        path = audit.bound(row["source"])
        audit.need(
            path.resolve() == (directory / "full_t000_p0.npz").resolve(),
            "Candidate replay binds the initial actual bank",
        )
        bank = full_files[str(path.resolve())]
        candidate = int(row["candidate"])
        if not np.isfinite(bank["C"][candidate]):
            audit.need(
                row.get("status") == "UNAVAILABLE_INVALID_PREFIX"
                and set(row) == {"source", "candidate", "status"},
                "Invalid-prefix candidate verification remains explicitly unavailable",
            )
            native_discrepancy_checks.append(
                dict(candidate=candidate, status="UNAVAILABLE_INVALID_PREFIX")
            )
            continue
        audit.need("status" not in row, "Finite candidate has its actual native verification")
        replay = load_arrays(audit, audit.bound(row["trace"]))
        replay_outcome = compare_native_episode(audit, replay, row["outcome"], 48)
        length = len(replay["actions"])
        audit.equal(
            replay["actions"],
            bank["actions"][candidate, :length],
            "Exact saved candidate physical action prefix",
        )
        audit.close(
            replay["states"][1:],
            bank["posts"][candidate, :length],
            "Recorded native versus forecast state prefixes",
            atol=1e-10,
        )
        for key, plural in (
            ("state", "states"),
            ("warmstart", "warmstarts"),
            ("integration", "integrations"),
        ):
            audit.equal(replay[plural][0], context[key], "Candidate replay full initial context")
        prediction = load_arrays(audit, audit.bound(row["predictions"]))
        audit.equal(
            prediction["observations"],
            replay["observations"][:-1],
            "Native replay pre-action prediction input",
        )
        extra_prediction_rows = 0
        if (
            native_prediction_checker is not None
            and row.get("comparison_version") == "MATCHED_GROUP_L3_1"
        ):
            metrics, extra_prediction_rows = native_prediction_checker(
                audit, payload=payload, bank=bank, replay=replay, prediction=prediction, row=row
            )
        else:
            expected_prediction = audit.predict(payload, prediction["observations"]).astype(
                np.float64
            )
            audit.close(
                prediction["predictions"],
                expected_prediction,
                "Native replay deployed outputs",
                atol=1e-6,
            )
            d = float(0.5 * np.sum((replay["actions"] - prediction["predictions"]) ** 2))
            audit.close(row["D"], d, "Native replay D reduction", atol=1e-10)
            audit.close(prediction["D"], d, "Native replay D file", atol=1e-10)
            use = bank["mask"] & np.isfinite(bank["C"])[:, None]
            audit.need(
                int(use[candidate].sum()) == length,
                "Native replay preserves the source executable-prefix length",
            )
            offset = int(use[:candidate].sum())
            bank_prediction = bank["predictions"][offset : offset + length]
            audit.equal(
                prediction["expected_predictions"],
                bank_prediction,
                "Expected native predictions are the exact original full-bank slice",
            )
            metrics = compare_native_discrepancy(
                audit,
                actions=replay["actions"],
                native_predictions=prediction["predictions"],
                bank_predictions=bank_prediction,
                bank_discrepancy=float(bank["D"][candidate]),
                saved_action_error=float(prediction["action_max_error"]),
                saved_D_tolerance=float(prediction["D_tolerance"]),
            )
        audit.close(
            row["action_max_error"],
            metrics["action_max_error"],
            "Native verification action-error manifest",
            atol=1e-15,
        )
        audit.close(
            row["D_tolerance"],
            metrics["D_tolerance"],
            "Native verification propagated-D manifest",
            atol=0,
        )
        native_discrepancy_checks.append(dict(candidate=candidate, **metrics))
        audit.close(
            replay_outcome["complete_cost"],
            bank["C"][candidate],
            "Native versus forecast prefix cost",
            atol=1e-9,
        )
        replay_controls += length
        replay_prediction_rows += length + extra_prediction_rows
    if job.get("replay_episode"):
        replay = load_arrays(audit, audit.bound(result["native_episode_replay"]))
        for key in ("states", "warmstarts", "integrations", "observations", "actions", "costs"):
            audit.close(
                replay[key], raw[key], "Complete saved native episode replay: " + key, atol=1e-10
            )
        replay_controls += n
    else:
        audit.need("native_episode_replay" not in result, "No undeclared extra episode replay")
    ledger = (closed_job_checker or _closed_job)(audit, directory)
    expected_work = dict(
        model_controls=rows * sampler.candidates * sampler.horizon,
        native_controls=n + replay_controls,
        neural_rows=prediction_rows + replay_prediction_rows,
    )
    expected_work["physics_substeps"] = 4 * (
        expected_work["model_controls"] + expected_work["native_controls"]
    )
    for key, value in expected_work.items():
        audit.need(
            ledger["counts"].get("completed_" + key, 0) == value,
            "Complete controller and verification census: " + key,
        )
    audit.equal(result["work"], ledger["counts"], "Controller manifest actual receipt census")
    return dict(
        outcome={**measured, "local_walking": result["outcome"]["local_walking"]},
        raw=raw,
        actor_hash=payload["policy_hash"],
        banks=rows,
        full_banks=full_banks,
        native_discrepancy_checks=native_discrepancy_checks,
        ledger=ledger,
        manifest=result,
    )


def _teacher_summary(rows: list[dict]) -> dict:
    outcomes = [row["outcome"] for row in rows]
    return dict(
        local_walking=sum(bool(row["local_walking"]) for row in outcomes),
        full_survival=sum(bool(row["full_survival"]) for row in outcomes),
        complete_cost=float(np.mean([row["complete_cost"] for row in outcomes])),
        velocity_mae=float(np.mean([row["velocity_mae"] for row in outcomes])),
        controls=sum(row["controls"] for row in outcomes),
    )


def _action_change(first: list[dict], second: list[dict]) -> dict:
    cases, squares = [], []
    for slot, (a, b) in enumerate(zip(first, second, strict=True)):
        x, y = a["raw"]["actions"], b["raw"]["actions"]
        n = min(len(x), len(y))
        if n <= 0:
            raise AuditFailure("Every action-effect restart has a positive common prefix")
        difference = (x[:n] - y[:n]) ** 2
        cases.append(
            dict(slot=slot, common_controls=n, action_rms=float(np.sqrt(difference.mean())))
        )
        squares.append(difference.ravel())
    return dict(pooled_rms=float(np.sqrt(np.concatenate(squares).mean())), cases=cases)


def _close_tree(audit: Audit, actual, expected, message: str) -> None:
    if isinstance(expected, dict):
        audit.need(isinstance(actual, dict) and set(actual) == set(expected), message + " keys")
        for key, value in expected.items():
            _close_tree(audit, actual[key], value, message + "." + str(key))
    elif isinstance(expected, list):
        audit.need(isinstance(actual, list) and len(actual) == len(expected), message + " rows")
        for a, b in zip(actual, expected, strict=True):
            _close_tree(audit, a, b, message)
    elif isinstance(expected, float):
        audit.close(actual, expected, message, atol=1e-9)
    else:
        audit.need(actual == expected, message)


def _selection(audit: Audit, rosters: dict[str, list[dict]], saved: dict, config: dict) -> dict:
    baseline = _teacher_summary(rosters["independent"])
    comparisons = {}
    for arm in ("weak", "large"):
        current = _teacher_summary(rosters[arm])
        effect = _action_change(rosters["independent"], rosters[arm])
        gates = dict(
            baseline_competent=baseline["local_walking"] >= 6,
            positive_competent=current["local_walking"] >= 6,
            walking_nonregression=current["local_walking"] >= baseline["local_walking"],
            survival_nonregression=current["full_survival"] >= baseline["full_survival"],
            cost_noninferiority=current["complete_cost"] <= 1.10 * baseline["complete_cost"],
            effective_intervention=effect["pooled_rms"] >= 0.01,
        )
        comparisons[arm] = dict(
            outcomes=current, effect=effect, gates=gates, eligible=all(gates.values())
        )
    selected = next((arm for arm in ("weak", "large") if comparisons[arm]["eligible"]), None)
    alphas = {row["name"]: row["alpha"] for row in config["controller"]["arms"]}
    expected = dict(
        schema="local-coupling-selection-1",
        baseline=baseline,
        comparisons=comparisons,
        selected_condition=selected,
        alpha=alphas[selected] if selected else None,
        selected_before_fitting=True,
    )
    _close_tree(audit, saved, expected, "Selection uses competence/action effect before fitting")
    return expected


def _forks(audit: Audit, prepared: dict, config: dict) -> dict:
    rows = prepared["forks"]
    audit.need(
        [row["stream"] for row in rows] == [0, 1, 2], "Exactly three one-time jumped-PCG forks"
    )
    streams = {}
    original = None
    for row in rows:
        envelope_path = audit.bound(row["envelope"])
        payload, envelope = audit_stream(audit, envelope_path.parent)
        metadata = envelope["metadata"]
        audit.need(
            metadata["pcg_jump"] == row["stream"] + 1 and metadata["stream"] == row["stream"],
            "One declared R1 PCG jump",
        )
        incoming = audit.bound(metadata["incoming"])
        parent, parent_envelope = audit_stream(audit, incoming.parent)
        if original is None:
            original = parent
        else:
            audit.equal(parent, original, "All forks share one original inherited checkpoint")
        for name, spec in config["inputs"]["incoming_stream_files"].items():
            audit.bind(incoming.parent / name, spec)
        audit.equal(
            payload["state_dict"], parent["state_dict"], "Fork leaves actor/normalization unchanged"
        )
        audit.equal(
            payload["private_rng_state"],
            parent["private_rng_state"],
            "Fork leaves private Torch RNG unchanged",
        )
        audit.bind(
            envelope_path.parent / "global_rng.pt", audit.bind(incoming.parent / "global_rng.pt")
        )
        expected = copy.deepcopy(parent["optimizer_state"])
        pcg = np.random.PCG64(0)
        pcg.state = copy.deepcopy(expected["rng"])
        expected["rng"] = pcg.jumped(row["stream"] + 1).state
        audit.equal(
            payload["optimizer_state"],
            expected,
            "Only the declared private PCG state changes at fork creation",
        )
        audit.need(
            envelope["global_rng_fingerprint"] == parent_envelope["global_rng_fingerprint"],
            "Exact inherited global envelope",
        )
        streams[row["stream"]] = row["envelope"]
    return streams


def _accounting(
    audit: Audit, run: Path, result: dict, prepared: dict, out: Path, stage: str
) -> dict:
    counts = copy.deepcopy(prepared.get("engineering_work", {}))
    seen = set()
    phases = []
    clean = True
    roster = result["accounting"]["child_receipts"]
    for spec in roster:
        path = audit.bound(spec)
        audit.need(str(path) not in seen, "Each child receipt imported once")
        seen.add(str(path))
        ledger = audit_ledger(audit, path.parent)
        for key, value in ledger["counts"].items():
            counts[key] = counts.get(key, 0) + value
        clean = (
            clean
            and ledger["receipt"]["status"] == "COMPLETED"
            and ledger["failed"] == ledger["rejected"] == 0
            and not any(
                key.startswith("unknown_partial_") and value
                for key, value in ledger["counts"].items()
            )
        )
        phases.append(
            dict(receipt=spec, counts=ledger["counts"], status=ledger["receipt"]["status"])
        )
    audit.equal(
        counts, result["accounting"]["counts"], "Engineering plus every child phase, counted once"
    )
    raw = (run / "study_ledger.jsonl").read_bytes()
    snapshot = out / "input_study_ledger.jsonl"
    snapshot.write_bytes(raw)
    audit.bind(snapshot)
    events = [json.loads(line) for line in raw.splitlines()]
    audit.need(len(events) == len(roster), "Every coordinator import has one ledger event")
    for row, spec in zip(events, roster, strict=True):
        audit.need(
            row["event"] == "closed_child_included" and row["receipt"] == spec,
            "Coordinator import roster",
        )
        audit.equal(
            row["counts"], audit.json(audit.bound(spec))["counts"], "Imported original child census"
        )
    return dict(
        clean=bool(clean),
        counts=counts,
        phases=phases,
        engineering_included_once=True,
        coordinator_ledger_snapshot=str(snapshot),
        stage=stage,
    )


def _panel_summary(rows: list[dict]) -> dict:
    audit_rows = [row["outcome"] for row in rows]
    return dict(
        episodes=len(rows),
        walking=sum(row["walking"] for row in audit_rows),
        full_survival=sum(row["full_survival"] for row in audit_rows),
        complete_cost=float(np.mean([row["complete_cost"] for row in audit_rows])),
        controls=sum(row["controls"] for row in audit_rows),
    )


def _round(
    audit: LearningAudit,
    block: dict,
    config: dict,
    config_binding: dict,
    prepared: dict,
    forks: dict,
    acquisition: dict,
    previous: dict | None,
    selected: str,
) -> dict:
    round_index = block["round"]
    corpus = {}
    for row in block["corpora"]:
        manifest = audit_corpus(audit, audit.bound(row["manifest"]))
        audit.equal(manifest["config"], config_binding, "Corpus binds frozen study config")
        stream = row["stream"]
        corpus[stream] = manifest
        source_stream = -1 if round_index == 1 else stream
        arms = ("independent", selected if round_index == 1 else "positive")
        for case in manifest["cases"]:
            expected = [
                acquisition[(round_index, source_stream, arm, case["slot"])]["manifest"]["trace"]
                for arm in arms
            ]
            audit.equal(
                case["traces"],
                expected,
                "New-round-only labels use actual corresponding teacher episodes",
            )
    audit.need(
        set(corpus) == ({-1} if round_index == 1 else {0, 1, 2}), "Exact equal-corpus roster"
    )
    incoming = {(row["stream"], row["arm"]): row["envelope"] for row in block["incoming_streams"]}
    outgoing = {(row["stream"], row["arm"]): row["envelope"] for row in block["outgoing_streams"]}
    expected_keys = {(s, a) for s in range(3) for a in ("independent", "positive")}
    audit.need(
        set(incoming) == set(outgoing) == expected_keys and len(block["fits"]) == 6,
        "Exactly six matched fixed-budget fits",
    )
    fits = {}
    compact = {}
    for row in block["fits"]:
        key = (row["stream"], row["arm"])
        audit.need(key not in fits, "No repeated or selected fitting endpoint")
        expected_in = (
            forks[key[0]] if round_index == 1 else previous["outgoing"][f"{key[0]}:{key[1]}"]
        )
        audit.equal(
            incoming[key], expected_in, "Retain actual own-arm Adam/RNG parent across rounds"
        )
        data = corpus[-1 if round_index == 1 else key[0]]["files"][
            "a" if key[1] == "independent" else "b"
        ]
        fit = audit_fit_job(
            audit,
            audit.bound(row["result"]).parent,
            dict(
                round=round_index,
                stream=key[0],
                arm=key[1],
                config=config_binding,
                incoming=expected_in,
                corpus=data,
            ),
            config,
        )
        audit.need(
            str(audit.bound(outgoing[key])) == fit["outgoing"], "Retained actual child endpoint"
        )
        fits[key] = fit
        compact[f"{key[0]}:{key[1]}"] = dict(
            parent=fit["parent"]["policy_hash"],
            child=fit["child"]["policy_hash"],
            continuation=fit["continuation"],
            diagnostics=fit["diagnostics"],
        )
    for stream in range(3):
        a, b = fits[(stream, "independent")], fits[(stream, "positive")]
        if round_index == 1:
            check_paired_incoming(audit, a["parent"], b["parent"])
        audit.equal(a["indices"], b["indices"], "Paired actual minibatch row-index stream")
    groups = {}
    contexts = {}
    seen = set()
    for row in block["evaluations"]:
        stream, arm, kind, panel, seed = (
            row[k] for k in ("stream", "arm", "kind", "panel", "reset_seed")
        )
        key = (stream, arm, kind, panel, seed)
        audit.need(key not in seen, "No duplicate evaluation/retry")
        seen.add(key)
        base = config["seeds"][f"round{round_index}_{panel}_base"]
        audit.need(base <= seed < base + 16, "Fresh frozen evaluation seed and panel")
        if round_index == 1 and kind == "parent":
            expected_checkpoint = prepared["checkpoint"]
            audit.need(stream == -1 and arm == "parent", "Shared R1 original actor evaluated once")
        else:
            fit = fits[(stream, arm)]
            envelope = fit["outgoing"] if kind == "child" else fit["incoming"]
            path = Path(envelope).parent / "policy.pt"
            seal = audit.bind(path)
            expected_checkpoint = dict(
                path=str(path.resolve()), bytes=seal["bytes"], sha256=seal["sha256"]
            )
        value = audit_evaluation_job(
            audit,
            audit.bound(row["result"]).parent,
            dict(config=config_binding, checkpoint=expected_checkpoint, reset_seed=seed),
        )
        context_key = (panel, seed)
        if context_key in contexts:
            audit.equal(
                contexts[context_key],
                value["context"],
                "All actors use identical complete reset contexts",
            )
        else:
            contexts[context_key] = value["context"]
        groups.setdefault((stream, arm, kind, panel), []).append(value)
    actors = [
        (s, a, k)
        for s in range(3)
        for a in ("independent", "positive")
        for k in (("child",) if round_index == 1 else ("parent", "child"))
    ]
    if round_index == 1:
        actors.append((-1, "parent", "parent"))
    expected = {
        (s, a, k, p, config["seeds"][f"round{round_index}_{p}_base"] + i)
        for s, a, k in actors
        for p in ("primary", "confirmation")
        for i in range(16)
    }
    audit.need(
        seen == expected, "Every declared primary and confirmation evaluation, including failures"
    )
    panels = {name: {} for name in ("primary", "confirmation")}
    for name in panels:
        for stream in range(3):
            row = {
                arm: _panel_summary(groups[(stream, arm, "child", name)])
                for arm in ("independent", "positive")
            }
            if round_index == 1:
                row["parent"] = _panel_summary(groups[(-1, "parent", "parent", name)])
            else:
                for arm in ("independent", "positive"):
                    row["parent_" + arm] = _panel_summary(groups[(stream, arm, "parent", name)])
            panels[name][str(stream)] = row
    decision = r1_walking_decision(panels) if round_index == 1 else r2_walking_decision(panels)
    saved = block["criterion"]
    qualifying = []
    for stream, rows in decision["per_stream"].items():
        if round_index == 1:
            passed = all(
                row["walking_vs_independent"] >= 2 and row["walking_vs_parent"] >= 2
                for row in rows["panels"].values()
            )
        else:
            passed = all(
                row["own_walking_increment"] >= 2
                and row["difference_in_differences"] >= 2
                and row["walking_vs_independent"] >= 2
                for row in rows["panels"].values()
            )
        if passed:
            qualifying.append(int(stream))
    audit.need(
        saved["passed"] == decision["passed"]
        and saved["nonregression_all"] == decision["all_stream_nonregression"]
        and saved["winning_streams"] == qualifying,
        "Independent walking and confirmation criterion",
    )
    return dict(
        round=round_index,
        panels=panels,
        criterion=decision,
        fits=compact,
        outgoing={f"{s}:{a}": binding for (s, a), binding in outgoing.items()},
    )


def audit_native_prerequisite(audit: Audit, spec: dict, prepared: dict) -> dict:
    path = audit.bound(spec)
    directory = path.parent
    result, job = audit.json(path), audit.json(directory / "job.json")
    audit.need(
        result["schema"] == "local-coupling-native-prerequisite-1"
        and result["status"] == "COMPLETED",
        "Original native prerequisite completed before acquisition",
    )
    for key in ("restarts", "student_episode"):
        audit.equal(job[key], prepared[key], "Native prerequisite uses the frozen input: " + key)
    contexts = load_arrays(audit, audit.bound(prepared["restarts"]))
    source = load_arrays(audit, audit.bound(prepared["student_episode"]))
    capture = load_arrays(audit, audit.bound(result["capture"]))
    repeat = load_arrays(audit, audit.bound(result["repeat"]))
    measured = outcome(audit, capture, 4)
    outcome(audit, repeat, 4)
    audit.need(
        len(capture["actions"]) == 4,
        "Four captured native controls before the source student's fall",
    )
    audit.equal(
        capture["actions"], source["actions"][1:5], "Declared original four-control input tape"
    )
    for name in ("states", "warmstarts", "integrations", "observations", "actions", "costs"):
        audit.close(
            repeat[name], capture[name], "Strict saved same-host native replay: " + name, atol=1e-10
        )
    candidate = load_arrays(audit, audit.bound(result["candidate"]))
    for key, plural in (
        ("state", "states"),
        ("warmstart", "warmstarts"),
        ("integration", "integrations"),
    ):
        expected = contexts["r0_" + key]
        audit.equal(capture[plural][0], expected, "Native prerequisite initial context bytes")
        audit.equal(
            candidate["initial_" + key],
            expected,
            "Candidate prerequisite starts from identical context",
        )
        audit.equal(
            candidate["after_" + key],
            expected,
            "Candidate forecast leaves complete live context unchanged",
        )
    audit.equal(candidate["actions"], capture["actions"], "Single-candidate action tape")
    audit.close(
        candidate["posts"][0],
        capture["states"][1:],
        "Saved K1/H4 native and forecast states",
        atol=1e-10,
    )
    c, _, _ = batch_cost(candidate["posts"], candidate["actions"][None])
    audit.close(candidate["costs"], c, "K1/H4 forecast cost", atol=1e-9)
    audit.close(
        candidate["costs"][0],
        measured["complete_cost"],
        "K1/H4 native versus forecast complete cost",
        atol=1e-9,
    )
    audit.bound(result["model"])
    audit.bound(result["model_options"])
    audit.bind(directory / "ledger.jsonl")
    ledger_rows = [
        json.loads(line) for line in (directory / "ledger.jsonl").read_text().splitlines()
    ]
    model_events = [
        row for row in ledger_rows if row["event"] == "native_model_bound_before_physics"
    ]
    audit.need(len(model_events) == 1, "Compiled model/options sealed exactly once")
    event = model_events[0]
    audit.equal(event["model"], result["model"], "Actual compiled-model seal")
    audit.equal(event["options"], result["model_options"], "Actual runtime-options seal")
    physics = [
        row
        for row in ledger_rows
        if row["event"] == "requested"
        and (row["charges"].get("native_controls", 0) or row["charges"].get("model_controls", 0))
    ]
    audit.need(
        bool(physics) and event["id"] < min(row["id"] for row in physics),
        "Native model bound before the first charged control",
    )
    ledger = _closed_job(audit, directory)
    for unit, value in (
        ("native_controls", 8),
        ("model_controls", 4),
        ("physics_substeps", 48),
        ("neural_rows", 0),
    ):
        audit.need(
            ledger["counts"].get("completed_" + unit, 0) == value,
            "Complete native prerequisite census: " + unit,
        )
    return dict(
        passed=True,
        counts=ledger["counts"],
        scope="Saved strict replay/candidate/context algebra; no dynamics repeated by auditor",
    )


def _inspect(audit: LearningAudit, run: Path, out: Path, stage: str) -> dict:
    snapshot = run / "r1_result.json" if stage == "r1" else run / "result.json"
    audit.bind(snapshot)
    copied = out / "input_result.json"
    copied.write_bytes(snapshot.read_bytes())
    result = audit.json(copied)
    audit.need(result["schema"] == "local-coupling-learning-l2-result-1", "Frozen L2 result schema")
    config = audit.json(audit.bound(result["config"]))
    audit.need(
        config["launch_ready"] is True and config["version"] == "L2.0",
        "Frozen launch-ready L2 configuration",
    )
    prepared = audit.json(audit.bound(result["prepared_manifest"]))
    audit.need(prepared["schema"] == "local-coupling-prepared-1", "Prepared input roster")
    audit.need(
        prepared["config_sha256"] == result["config"]["sha256"],
        "Prepared fork/input roster binds the exact frozen configuration",
    )
    launch = audit.json(audit.bound(result["launch"]))
    for key in ("config", "prepared_manifest", "source_manifest"):
        audit.equal(launch[key], result[key], "Launch/result seal: " + key)
    source = audit.json(audit.bound(result["source_manifest"]))
    source_root = Path(launch["source_directory"])
    for row in source["files"]:
        audit.bind(source_root / row["path"], row)
    audit.bind(
        audit.bound(result["config"]),
        audit.bind(source_root / "configs/studies/local_coupling_learning_l2_0.json"),
    )
    for row in prepared["files"]:
        audit.bound(row)
    for key, original in (
        ("checkpoint", "checkpoint"),
        ("restarts", "restart_contexts"),
        ("held", "held_diagnostic"),
        ("student_episode", "student_episode"),
    ):
        audit.bind(audit.bound(prepared[key]), config["inputs"][original])
    environment = audit.json(audit.bound(launch["environment"]))
    for row in [environment["interpreter"], *environment["binary_files"]]:
        audit.bound(row)
    prerequisite = audit_native_prerequisite(audit, result["native_preflight"], prepared)
    forks = _forks(audit, prepared, config)
    prior = None
    prior_work = dict(
        cpu_seconds=0.0, wall_seconds=0.0, requested_neural_rows=0, completed_neural_rows=0
    )
    reuse = False
    if stage == "final" and result.get("r1_admission"):
        admission = audit.json(audit.bound(result["r1_admission"]))
        audit.need(
            admission["passed"] is True
            and admission["stage"] == "r1"
            and admission["unknown_partial_neural_rows"] == 0,
            "Passing immutable independent R1 admission",
        )
        prior_work = {key: admission[key] for key in prior_work}
        audit.cpu_limit = min(
            audit.cpu_limit, config["budgets"]["audit"]["cpu_seconds"] - prior_work["cpu_seconds"]
        )
        audit.seconds = min(
            audit.seconds, config["budgets"]["audit"]["wall_seconds"] - prior_work["wall_seconds"]
        )
        audit.neural_limit = min(
            audit.neural_limit,
            config["budgets"]["audit"]["neural_rows"] - prior_work["requested_neural_rows"],
        )
        audit.check()
        prior = audit.json(audit.bound(admission["findings"]))
        bindings = audit.json(audit.bound(admission["file_bindings"]))
        for path, binding in bindings.items():
            audit.bind(Path(path), binding)
        previous = audit.json(audit.bound(prior["input_result"]))
        audit.equal(previous["config"], result["config"], "R1 and final configuration identity")
        audit.equal(
            previous["rounds"][0],
            result["rounds"][0],
            "R1 results remain immutable after admission",
        )
        audit.equal(
            previous["acquisitions"],
            result["acquisitions"][:24],
            "R1 acquisition evidence is not overwritten",
        )
        ledger_snapshot = audit.bound(prior["accounting"]["coordinator_ledger_snapshot_binding"])
        audit.need(
            (run / "study_ledger.jsonl").read_bytes().startswith(ledger_snapshot.read_bytes()),
            "Final coordinator ledger retains the exact admitted prefix",
        )
        reuse = True
    acquisition = {}
    rosters = {}
    full_banks = 0
    all_banks = 0
    for row in result["acquisitions"]:
        key = tuple(row[k] for k in ("round", "stream", "arm", "slot"))
        audit.need(key not in acquisition, "One actual acquisition per frozen role")
        manifest_path = audit.bound(row["manifest"])
        if reuse and key[0] == 1:
            manifest = audit.json(manifest_path)
            value = dict(
                outcome=manifest["outcome"],
                raw=load_arrays(audit, audit.bound(manifest["trace"])),
                manifest=manifest,
                actor_hash=manifest["actor_hash"],
                banks=manifest["banks"],
                full_banks=0,
            )
        else:
            if key[0] == 1:
                actor = prepared["checkpoint"]
                alpha = next(
                    r["alpha"] for r in config["controller"]["arms"] if r["name"] == key[2]
                )
            else:
                previous_round = next(r for r in result["rounds"] if r["round"] == 1)
                envelope = next(
                    r["envelope"]
                    for r in previous_round["outgoing_streams"]
                    if (r["stream"], r["arm"]) == key[1:3]
                )
                path = Path(envelope["path"]).parent / "policy.pt"
                seal = audit.bind(path)
                actor = dict(path=str(path.resolve()), bytes=seal["bytes"], sha256=seal["sha256"])
                alpha = (
                    0.0
                    if key[2] == "independent"
                    else audit.json(audit.bound(result["selection"]))["alpha"]
                )
            expected = dict(
                round=key[0],
                stream=key[1],
                arm=key[2],
                slot=key[3],
                alpha=alpha,
                seed=config["seeds"]["planner_base"] + key[3],
                checkpoint=actor,
            )
            value = audit_acquisition(audit, manifest_path.parent, expected, config)
        acquisition[key] = value
        rosters.setdefault(key[:3], []).append((key[3], value))
        full_banks += value["full_banks"]
        all_banks += value["banks"]
    for key, values in rosters.items():
        audit.need(
            sorted(slot for slot, _ in values) == list(range(8)),
            "Complete eight-restart roster; falls do not abort other episodes",
        )
        rosters[key] = [value for _, value in sorted(values)]
    audit.need(
        set(key for key in rosters if key[0] == 1)
        == {(1, -1, arm) for arm in ("independent", "weak", "large")},
        "All 24 R1 acquisition episodes",
    )
    for slot in range(8):
        a = acquisition[(1, -1, "independent", slot)]
        for arm in ("weak", "large"):
            b = acquisition[(1, -1, arm, slot)]
            for name in ("states", "warmstarts", "integrations"):
                audit.equal(
                    a["raw"][name][0],
                    b["raw"][name][0],
                    "R1 arms begin from identical complete context",
                )
    selection = _selection(
        audit,
        {arm: rosters[(1, -1, arm)] for arm in ("independent", "weak", "large")},
        audit.json(audit.bound(result["selection"])),
        config,
    )
    round_summaries = []
    if selection["selected_condition"] is None:
        audit.need(
            not result["rounds"] and len(acquisition) == 24 and stage == "final",
            "No fitting or follow-up after absent competent effective teacher",
        )
    else:
        for block in result["rounds"]:
            index = block["round"]
            audit.need(
                index == len(round_summaries) + 1 and index <= 2,
                "At most two declared alternating learning rounds",
            )
            if index == 1 and reuse:
                summary = prior["rounds"][0]
            else:
                summary = _round(
                    audit,
                    block,
                    config,
                    result["config"],
                    prepared,
                    forks,
                    acquisition,
                    round_summaries[-1] if round_summaries else None,
                    selection["selected_condition"],
                )
            round_summaries.append(summary)
        audit.need(
            bool(round_summaries), "Selected teacher requires its fixed complete learning round"
        )
        r1 = round_summaries[0]["criterion"]
        if stage == "r1":
            audit.need(
                r1["passed"] and len(round_summaries) == 1 and len(acquisition) == 24,
                "Only confirmed closed R1 is admitted to R2",
            )
        elif r1["passed"]:
            audit.need(
                reuse and len(round_summaries) == 2 and len(acquisition) == 72,
                "Confirmed R1 requires the admitted bounded second round",
            )
        else:
            audit.need(
                len(round_summaries) == 1
                and len(acquisition) == 24
                and not result.get("r1_admission"),
                "No R2 after negative/unconfirmed R1",
            )
    teacher_feedback = []
    useful = False
    if len(round_summaries) == 2:
        r2 = round_summaries[1]
        for stream in range(3):
            zero = rosters[(2, stream, "independent")]
            positive = rosters[(2, stream, "positive")]
            reference = rosters[(1, -1, selection["selected_condition"])]
            for slot in range(8):
                for name in ("states", "warmstarts", "integrations", "actions", "costs"):
                    audit.equal(
                        zero[slot]["raw"][name],
                        rosters[(1, -1, "independent")][slot]["raw"][name],
                        "Policy-free independent discovery remains identical under paired contexts/noise",
                    )
            current = _teacher_summary(positive)
            references = [_teacher_summary(reference), _teacher_summary(zero)]
            competent = all(
                current["local_walking"] >= r["local_walking"]
                and current["full_survival"] >= r["full_survival"]
                and current["complete_cost"] <= 1.1 * r["complete_cost"]
                for r in references
            )
            effect = _action_change(reference, positive)
            teacher_feedback.append(
                dict(
                    stream=stream,
                    competent=competent,
                    effect=effect,
                    changed=effect["pooled_rms"] >= 0.01,
                )
            )
        _close_tree(
            audit,
            result["rounds"][1]["teacher_feedback"],
            teacher_feedback,
            "Saved R2 teacher-feedback evidence",
        )
        wins = {int(s) for s in r2["criterion"]["joint_streams"]}
        useful = bool(
            r2["criterion"]["passed"]
            and all(r["competent"] for r in teacher_feedback)
            and sum(r["changed"] and r["stream"] in wins for r in teacher_feedback) >= 2
        )
        audit.need(
            result["useful_feedback"] == useful,
            "Finite feedback-associated criterion; no frozen-old-actor causal control",
        )
    accounting = _accounting(audit, run, result, prepared, out, stage)
    ledger_path = Path(accounting["coordinator_ledger_snapshot"])
    seal = audit.bind(ledger_path)
    accounting["coordinator_ledger_snapshot_binding"] = dict(
        path=str(ledger_path), bytes=seal["bytes"], sha256=seal["sha256"]
    )
    audit.need(accounting["clean"], "All phase accounting closed with no unknown scientific work")
    audit.need(
        result["status"] != "FAILED_STOPPED",
        "An original failed execution is not a complete successful audit",
    )
    return dict(
        passed=True,
        stage=stage,
        original_status=result["status"],
        input_result=dict(
            path=str(copied.resolve()), **{k: audit.bind(copied)[k] for k in ("bytes", "sha256")}
        ),
        source_commit=source["git_commit"],
        selection=selection,
        rounds=round_summaries,
        teacher_feedback=teacher_feedback,
        useful_feedback=useful,
        accounting=accounting,
        all_actual_banks=all_banks,
        new_full_banks_independently_recomputed=full_banks,
        reused_r1_audit=reuse,
        prior_admission=result.get("r1_admission"),
        prior_audit_work=prior_work,
        native_prerequisite=prerequisite,
        numerical_contract=dict(
            full_bank_cost_atol=1e-9,
            full_bank_score_atol=1e-9,
            deployed_D_atol=1e-6,
            native_state_context_atol=1e-10,
            native_cost_atol=1e-9,
            prediction_atol=1e-6,
            native_D_atol_formula="1e-10 + 2*(6*actual_prefix_controls)*1e-6",
            relative_tolerance=0,
        ),
        native_discrepancy_checks=(prior.get("native_discrepancy_checks", []) if reuse else [])
        + [
            dict(
                round=key[0],
                stream=key[1],
                arm=key[2],
                slot=key[3],
                checks=value["native_discrepancy_checks"],
            )
            for key, value in acquisition.items()
            if "native_discrepancy_checks" in value
        ],
        broad_project_success=False,
        scope="Saved-only finite learning/feedback test; no new physics or gradients; sparse forecast-state coverage; no equal-compute cloning/DAgger or second-body result",
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def audit_run(
    run: Path,
    out: Path,
    *,
    stage: str = "final",
    seconds: float = 3600,
    cpu_seconds: float = 3600,
    neural_rows: int = 8_000_000,
    output_bytes: int = 2 * 1024**3,
    memory_bytes: int = 4 * 1024**3,
) -> dict:
    """Create one independent audit receipt; R1 work is reused, never silently refit."""
    run, out = Path(run).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    audit = LearningAudit(
        seconds=seconds,
        cpu_seconds=cpu_seconds,
        neural_rows=neural_rows,
        output_bytes=output_bytes,
        memory_bytes=memory_bytes,
    )
    findings = {}
    try:
        audit.need(stage in ("r1", "final"), "Declared audit stage")
        findings = _inspect(audit, run, out, stage)
    except BaseException as error:
        findings = dict(
            passed=False,
            stage=stage,
            error=repr(error),
            traceback=traceback.format_exc(),
            scope="Incomplete or failed independent audit; original study evidence retained, no missing measurement treated as zero",
        )
        copied = out / "input_result.json"
        if copied.is_file():
            original = json.loads(copied.read_text())
            findings["original_status"] = original.get("status")
            findings["original_error"] = original.get("error")
            findings["declared_paid_counts"] = original.get("accounting", {}).get("counts")
            findings["declared_counts_scope"] = (
                "Preserved original declaration only; incomplete audit does not certify these counters or fill missing measurements with zero"
            )
    findings["checks"] = audit.checks
    (out / "findings.json").write_text(
        json.dumps(findings, default=_plain, indent=2, allow_nan=False) + "\n"
    )
    (out / "file_bindings.json").write_text(
        json.dumps(audit.files, default=_plain, indent=2, allow_nan=False) + "\n"
    )

    def seal(path):
        import hashlib

        data = path.read_bytes()
        return dict(
            path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )

    receipt = dict(
        schema="local-coupling-independent-audit-1",
        stage=stage,
        passed=findings["passed"],
        wall_seconds=time.monotonic() - audit.started,
        cpu_seconds=time.process_time() - audit.cpu_started,
        requested_neural_rows=audit.requested_rows,
        completed_neural_rows=audit.completed_rows,
        unknown_partial_neural_rows=audit.unknown_rows,
        model_controls=0,
        native_controls=0,
        physics_substeps=0,
        backward_rows=0,
        optimizer_steps=0,
        findings=seal(out / "findings.json"),
        file_bindings=seal(out / "file_bindings.json"),
        original_failure_preserved=True,
        limits=dict(
            wall_seconds=seconds,
            cpu_seconds=cpu_seconds,
            neural_rows=neural_rows,
            output_bytes=output_bytes,
            memory_bytes=memory_bytes,
        ),
    )
    prior = findings.get("prior_audit_work", {})
    receipt["cumulative_completed_neural_rows"] = receipt["completed_neural_rows"] + prior.get(
        "completed_neural_rows", 0
    )
    receipt["cumulative_cpu_seconds"] = receipt["cpu_seconds"] + prior.get("cpu_seconds", 0)
    receipt["cumulative_wall_seconds"] = receipt["wall_seconds"] + prior.get("wall_seconds", 0)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > output_bytes:
        receipt.update(passed=False, error="Independent audit output ceiling exceeded")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("r1", "final"), default="final")
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--cpu-seconds", type=float, default=3600)
    parser.add_argument("--neural-rows", type=int, default=8_000_000)
    args = parser.parse_args(argv)
    receipt = audit_run(
        args.run,
        args.out,
        stage=args.stage,
        seconds=args.seconds,
        cpu_seconds=args.cpu_seconds,
        neural_rows=args.neural_rows,
    )
    print(json.dumps(receipt))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
