"""Deterministic mechanism falsifier, deliberately separate from task evidence.

Run with ``python -m src.gps.independent_preflight FRESH_DIRECTORY``. No actor
is fitted. A quadratic estimates complete measured returns of the declared
zero-action continuation, then reranks an unchanged five-action local bank.
The result tests plumbing and a possible causal relationship in a toy system;
it establishes neither MountainCar exploration nor general sample efficiency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


STEPS = 4
GOAL = 2.
ENERGY = .1
TERMINAL = 8.
ACTIONS = (-1., -.5, 0., .5, 1.)
CONTINUATION = "C_zero_v1: u=0 at every remaining step; no replanning, no memory"
VALIDATION_TOLERANCE = 1e-9
MINIMUM_PROMOTION_IMPROVEMENT = 1.


def _features(states):
    error = GOAL - np.asarray(states, dtype=np.float64)
    return np.stack((np.ones_like(error), error, error ** 2), axis=-1)


def _fit_value(states, targets):
    return np.linalg.lstsq(_features(states), np.asarray(targets), rcond=None)[0]


class PreflightFailure(RuntimeError):
    """Failure retains the complete result and its written artifact."""

    def __init__(self, result, path):
        self.result = result
        self.path = str(path)
        super().__init__("Preflight sentinels failed: " + ", ".join(result["failed_sentinels"]))


class _Probe:
    def __init__(self):
        self.started = time.perf_counter()
        self.transitions = []
        self.queries = []
        self.predictions = []
        self.bank = np.asarray(ACTIONS, dtype=np.float64)
        self.bank_digest = hashlib.sha256(self.bank.tobytes()).hexdigest()

    def begin(self, phase, purpose, kind, **metadata):
        query = {"id": len(self.queries), "phase": phase, "purpose": purpose,
                 "kind": kind, "controls": 0, "failed": False,
                 "wall_start_seconds": time.perf_counter() - self.started, **metadata}
        self.queries.append(query)
        return query

    def finish(self, query, **result):
        query.update(result)
        query["wall_stop_seconds"] = time.perf_counter() - self.started

    def step(self, state, action, remaining, query):
        # Every use of the model, including executed and candidate controls,
        # passes through this one transition recorder.
        state, action = float(state), float(action)
        if not np.isfinite(state) or not np.isfinite(action) or abs(action) > 1. or remaining < 1:
            query["failed"] = True
            raise ValueError("Invalid integrator query")
        next_state = state + action
        stage_cost = ENERGY * action ** 2
        terminal_cost = TERMINAL * (GOAL - next_state) ** 2 if remaining == 1 else 0.
        row = {"id": len(self.transitions), "query_id": query["id"],
               "phase": query["phase"], "purpose": query["purpose"],
               "kind": query["kind"], "remaining": int(remaining), "state": state,
               "action": action, "next_state": next_state, "stage_cost": stage_cost,
               "terminal_cost": terminal_cost, "cost": stage_cost + terminal_cost}
        self.transitions.append(row)
        query["controls"] += 1
        return next_state, row["cost"]

    def trajectory(self, state, actions, remaining, *, phase, purpose, kind, **metadata):
        actions = list(map(float, actions))
        if len(actions) != remaining:
            raise ValueError("Complete continuation requires all remaining controls")
        query = self.begin(phase, purpose, kind, initial_state=float(state),
                           remaining_start=int(remaining), **metadata)
        states = [float(state)]
        costs = []
        for index, action in enumerate(actions):
            state, cost = self.step(state, action, remaining - index, query)
            states.append(state)
            costs.append(cost)
        # A suffix at r=0 has an already-observed terminal state; its terminal
        # boundary cost is a target evaluation and consumes no extra control.
        total = sum(costs) if remaining else TERMINAL * (GOAL - state) ** 2
        result = {"query_id": query["id"], "states": states, "actions": actions,
                  "costs": costs, "return": float(total), "remaining_start": remaining,
                  "final_state": float(state), "success": abs(state - GOAL) <= 1e-12}
        self.finish(query, **{key: value for key, value in result.items() if key != "query_id"})
        return result

    def predict(self, states, coefficients, *, phase, purpose, remaining, value_id):
        states = np.atleast_1d(np.asarray(states, dtype=np.float64))
        predictions = _features(states) @ coefficients
        row = {"id": len(self.predictions), "phase": phase, "purpose": purpose,
               "value_id": value_id, "states": states.tolist(),
               "remaining": np.broadcast_to(remaining, states.shape).astype(int).tolist(),
               "predictions": predictions.tolist(), "rows": int(len(states))}
        self.predictions.append(row)
        return predictions

    def controller(self, *, phase, coefficients=None, value_id="absent"):
        execution = self.begin(phase, "closed_loop_execution", "executed",
                               controller="H1 independent fixed bank + " + value_id,
                               initial_state=0., remaining_start=STEPS)
        state = 0.
        states, actions, costs, decisions = [state], [], [], []
        for remaining in range(STEPS, 0, -1):
            proposal = self.begin(phase, "independent_local_bank", "simulated",
                                  remaining_start=remaining, state=state,
                                  bank_digest=self.bank_digest, actions=list(ACTIONS))
            endpoints, local = [], []
            for action in self.bank:
                endpoint, cost = self.step(state, action, remaining, proposal)
                endpoints.append(endpoint)
                local.append(cost)
            scores = np.asarray(local)
            if coefficients is not None and remaining > 1:
                scores = scores + self.predict(endpoints, coefficients, phase=phase,
                    purpose="local_tail_rerank", remaining=remaining - 1, value_id=value_id)
            selected = int(scores.argmin())
            action = float(self.bank[selected])
            self.finish(proposal, endpoints=endpoints, local_costs=local,
                        scores=scores.tolist(), selected_index=selected)
            decisions.append({"query_id": proposal["id"], "state": state,
                              "remaining": remaining, "selected_index": selected,
                              "action": action, "guidance_present": coefficients is not None,
                              "guidance_used": coefficients is not None and remaining > 1})
            state, cost = self.step(state, action, remaining, execution)
            states.append(state)
            actions.append(action)
            costs.append(cost)
        result = {"query_id": execution["id"], "states": states, "actions": actions,
                  "costs": costs, "return": float(sum(costs)), "final_state": state,
                  "success": abs(state - GOAL) <= 1e-12, "decisions": decisions}
        self.finish(execution, **{key: value for key, value in result.items() if key != "query_id"})
        return result


def run_preflight(outpath) -> dict:
    """Run the frozen small falsifier, write a fresh preflight.json, or raise.

    ``outpath`` may name its destination directory or an explicit
    ``preflight.json``. An existing result is never modified. Failed sentinels
    are also written before raising, so a failure cannot disappear on rerun.
    """
    destination = Path(outpath)
    if destination.suffix.lower() == ".json":
        if destination.name != "preflight.json":
            raise ValueError("Explicit result filename must be preflight.json")
    else:
        destination = destination / "preflight.json"
    if destination.exists():
        raise FileExistsError(f"Retaining existing preflight: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    probe = _Probe()

    # Fixed full-plan search is the discovery mechanism. Selection uses only
    # measured candidate outcomes, and its winner is freshly re-executed.
    acquired_candidates = [probe.trajectory(0., [action] * STEPS, STEPS,
        phase="discovery", purpose="independent_full_plan", kind="simulated",
        proposal_index=index) for index, action in enumerate(ACTIONS)]
    winner = int(np.argmin([candidate["return"] for candidate in acquired_candidates]))
    acquired = probe.trajectory(0., acquired_candidates[winner]["actions"], STEPS,
        phase="discovery", purpose="selected_plan_reexecution", kind="executed",
        source_query_id=acquired_candidates[winner]["query_id"])

    # Acquisition is not relabeled as zero-controller evidence. Each target is
    # a distinct, complete, freshly simulated continuation of frozen C_zero.
    training = []
    for index, state in enumerate(acquired["states"]):
        remaining = STEPS - index
        continuation = probe.trajectory(state, [0.] * remaining, remaining,
            phase="fitting", purpose="complete_C_zero_target", kind="simulated",
            continuation_identity=CONTINUATION, acquisition_state_index=index)
        training.append({"state": state, "remaining": remaining,
                         "target": continuation["return"],
                         "continuation_query_id": continuation["query_id"],
                         "continuation_identity": CONTINUATION})
    learning_started = time.perf_counter()
    coefficients = _fit_value([row["state"] for row in training],
                              [row["target"] for row in training])
    learning_seconds = time.perf_counter() - learning_started
    corrupted = -coefficients

    # All heldout states lie strictly between the acquired training states.
    # This is interpolation under a stated exact-quadratic feature assumption.
    heldout = []
    for state, remaining in zip((.25, .75, 1.25, 1.75), (4, 3, 2, 1)):
        continuation = probe.trajectory(state, [0.] * remaining, remaining,
            phase="validation", purpose="heldout_complete_C_zero_target", kind="simulated",
            continuation_identity=CONTINUATION)
        heldout.append({"state": state, "remaining": remaining,
                        "target": continuation["return"],
                        "continuation_query_id": continuation["query_id"]})
    heldout_states = [row["state"] for row in heldout]
    heldout_remaining = [row["remaining"] for row in heldout]
    heldout_targets = np.array([row["target"] for row in heldout])
    good_predictions = probe.predict(heldout_states, coefficients, phase="validation",
        purpose="heldout_prediction", remaining=heldout_remaining, value_id="quadratic_C_zero")
    bad_predictions = probe.predict(heldout_states, corrupted, phase="validation",
        purpose="heldout_prediction", remaining=heldout_remaining, value_id="corrupted")
    good_error = float(np.max(np.abs(good_predictions - heldout_targets)))
    bad_error = float(np.max(np.abs(bad_predictions - heldout_targets)))

    baseline = probe.controller(phase="qualification_baseline")
    bad = probe.controller(phase="qualification_corrupted", coefficients=corrupted,
                           value_id="corrupted")
    good = probe.controller(phase="qualification_learned", coefficients=coefficients,
                            value_id="quadratic_C_zero")

    def qualifies(error, trajectory):
        return bool(error <= VALIDATION_TOLERANCE and trajectory["success"] and
                    baseline["return"] - trajectory["return"] >= MINIMUM_PROMOTION_IMPROVEMENT)

    bad_promoted = qualifies(bad_error, bad)
    good_promoted = qualifies(good_error, good)
    events = [{"event": "initialize", "guidance": "absent", "weight": 0.},
              {"event": "promotion_check", "candidate": "corrupted",
               "promoted": bad_promoted, "max_prediction_error": bad_error,
               "closed_loop_return": bad["return"], "weight_after": float(bad_promoted)},
              {"event": "promotion_check", "candidate": "quadratic_C_zero",
               "promoted": good_promoted, "max_prediction_error": good_error,
               "closed_loop_return": good["return"], "weight_after": float(good_promoted)}]
    deployed = probe.controller(phase="deployment_learned",
        coefficients=coefficients if good_promoted else None,
        value_id="quadratic_C_zero" if good_promoted else "absent")

    # Inject a known corrupt artifact after promotion, then test outcome-based
    # monitoring on fresh complete paired executions. No partial rollout score
    # is allowed to qualify or retain influence.
    monitor_baseline = probe.controller(phase="monitor_baseline")
    monitor_bad = probe.controller(phase="monitor_corruption", coefficients=corrupted,
                                   value_id="corrupted_active_artifact")
    revoked = bool(good_promoted and monitor_bad["return"] > monitor_baseline["return"])
    events.append({"event": "monitor_after_injected_corruption", "revoked": revoked,
                   "baseline_return": monitor_baseline["return"],
                   "active_return": monitor_bad["return"],
                   "reason": "complete paired execution regressed",
                   "weight_after": 0. if revoked else float(good_promoted)})
    fallback = probe.controller(phase="deployment_after_revocation",
        coefficients=None if revoked else (coefficients if good_promoted else None),
        value_id="absent" if revoked or not good_promoted else "quadratic_C_zero")

    planning_queries = [row for row in probe.queries if row["purpose"] == "independent_local_bank"]
    sentinels = {
        "initial_guidance_absent": all(not row["guidance_present"] for row in baseline["decisions"]),
        "measured_nontrivial_success_acquired": bool(acquired["success"] and
            acquired["states"][0] != GOAL and acquired["return"] < baseline["return"]),
        "complete_frozen_continuation_targets": all(
            probe.queries[row["continuation_query_id"]]["controls"] == row["remaining"] and
            row["continuation_identity"] == CONTINUATION for row in training),
        "heldout_complete_return_prediction_valid": good_error <= VALIDATION_TOLERANCE,
        "independent_proposals_unchanged": all(row["bank_digest"] == probe.bank_digest and
            row["actions"] == list(ACTIONS) for row in planning_queries),
        "guidance_changes_first_action": good["actions"][0] != baseline["actions"][0],
        "executed_long_horizon_improvement": bool(good["success"] and
            baseline["return"] - good["return"] >= MINIMUM_PROMOTION_IMPROVEMENT),
        "corrupted_value_denied_promotion": not bad_promoted,
        "good_value_earns_promotion": good_promoted,
        "promoted_deployment_improves": bool(deployed["success"] and
            baseline["return"] - deployed["return"] >= MINIMUM_PROMOTION_IMPROVEMENT),
        "corrupted_active_value_revoked": revoked,
        "revocation_restores_baseline_execution": bool(revoked and
            fallback["actions"] == baseline["actions"] and fallback["return"] == baseline["return"]),
    }
    by_phase = {}
    for row in probe.transitions:
        counts = by_phase.setdefault(row["phase"], {"simulated_controls": 0, "executed_controls": 0})
        counts[row["kind"] + "_controls"] += 1
    counts = {"total_controls": len(probe.transitions),
              "simulated_controls": sum(row["kind"] == "simulated" for row in probe.transitions),
              "executed_controls": sum(row["kind"] == "executed" for row in probe.transitions),
              "queries": len(probe.queries), "failed_queries": sum(row["failed"] for row in probe.queries),
              "prediction_calls": len(probe.predictions),
              "prediction_rows": sum(row["rows"] for row in probe.predictions),
              "fitting_rows": len(training), "learning_updates": 1,
              "zero_control_terminal_target_queries": sum(row["controls"] == 0 for row in probe.queries),
              "by_phase": by_phase}
    result = {
        "schema": "independent-preflight-v1", "status": "passed" if all(sentinels.values()) else "failed",
        "scope": "deterministic mechanism integration falsifier; not main environment or efficiency evidence",
        "configuration": {"steps": STEPS, "initial_state": 0., "goal": GOAL,
            "transition": "x_next=x+u", "action_bounds": [-1., 1.], "energy_coefficient": ENERGY,
            "terminal_coefficient": TERMINAL, "local_horizon": 1, "proposal_actions": list(ACTIONS),
            "acquisition": "five independent constant full-horizon plans; best measured plan reexecuted",
            "validation_tolerance": VALIDATION_TOLERANCE,
            "minimum_promotion_improvement": MINIMUM_PROMOTION_IMPROVEMENT,
            "bank_digest": probe.bank_digest},
        "learning": {"object": "quadratic complete-return cost of frozen C_zero",
            "continuation_identity": CONTINUATION, "features": ["1", "goal-x", "(goal-x)^2"],
            "coefficients": coefficients.tolist(), "corrupted_coefficients": corrupted.tolist(),
            "targets": training, "heldout": heldout,
            "heldout_predictions": good_predictions.tolist(),
            "heldout_max_absolute_error": good_error, "corrupted_max_absolute_error": bad_error,
            "remaining_feature_omitted_because": "C_zero has zero running cost and the same final state for every remaining horizon",
            "trainable_actor": False, "learning_seconds": learning_seconds},
        "assumptions": ["Exact deterministic integrator and restartable recorded states",
            "Fixed full-horizon proposal bank covers a successful plan",
            "Zero-action continuation is executable for every queried suffix",
            "Quadratic function class is correct; heldout tests interpolate within acquired state coverage",
            "Toy trust tests use paired deterministic resets and do not establish statistical confidence"],
        "acquisition": {"candidates": acquired_candidates, "selected_index": winner, "reexecuted": acquired},
        "trajectories": {"baseline": baseline, "learned_qualification": good,
            "corrupted_qualification": bad, "learned_deployment": deployed,
            "monitor_baseline": monitor_baseline, "monitor_corruption": monitor_bad,
            "fallback_after_revocation": fallback},
        "trust_events": events, "headroom": baseline["return"] - acquired["return"],
        "executed_improvement": baseline["return"] - good["return"],
        "first_action_change": good["actions"][0] - baseline["actions"][0],
        "sentinels": sentinels, "failed_sentinels": [key for key, passed in sentinels.items() if not passed],
        "counts": counts, "raw_transitions": probe.transitions,
        "raw_queries": probe.queries, "raw_predictions": probe.predictions,
        "wall_seconds": time.perf_counter() - probe.started,
        "verdict": (("Only toy causal plumbing passes" if all(sentinels.values()) else
                     "Toy causal plumbing fails") +
                    "; general exploration, learned-value MPC superiority, and standalone-policy GPS remain untested."),
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if result["failed_sentinels"]:
        raise PreflightFailure(result, destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpath", type=Path)
    args = parser.parse_args()
    result = run_preflight(args.outpath)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "wall_seconds": result["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
