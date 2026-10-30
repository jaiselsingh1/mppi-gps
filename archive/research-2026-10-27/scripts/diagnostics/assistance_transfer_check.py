"""Exact finite-horizon mechanism checks; no Walker or training-run mutation.

The two-step cost is the repository's coordinated-action counterexample.
Exact gradients isolate objective ownership from critic and optimizer errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess


def assisted(p: float) -> float:
    """Policy a0=p; independent expert then executes a1=p."""
    return .25 * p * p + (p - 1.) ** 2


def standalone(p: float, q: float) -> float:
    """The phase-only policy actually executes a0=p and a1=q."""
    return assisted(p) + (q - p) ** 2


def panel(p: float, q: float) -> dict:
    teacher_tail = (p - 1.) ** 2
    policy_tail = teacher_tail + (q - p) ** 2
    gap = policy_tail - teacher_tail
    assert math.isclose(standalone(p, q), assisted(p) + gap, abs_tol=1e-14)
    return {"policy": {"p": p, "q": q}, "assisted_J1": assisted(p),
            "standalone_J2": standalone(p, q), "teacher_tail": teacher_tail,
            "policy_tail": policy_tail, "completion_gap": gap}


def differences(before: dict, after: dict) -> dict:
    keys = ("assisted_J1", "standalone_J2", "completion_gap")
    result = {"delta_" + key: after[key] - before[key] for key in keys}
    assert math.isclose(result["delta_standalone_J2"],
                        result["delta_assisted_J1"] + result["delta_completion_gap"],
                        abs_tol=1e-14)
    return result


def absorbing(gamma: float, remaining: int, cost: float = 5.) -> float:
    assert 0. <= gamma <= 1. and remaining >= 0
    return sum(gamma ** j * cost for j in range(remaining))


def run() -> dict:
    # An exact gradient step for J1; q has exactly zero assisted gradient.
    old_p, old_q, step = .5, 0., 1. / 15.
    grad = (2.5 * old_p - 2., 0.)
    before = panel(old_p, old_q)
    after = panel(old_p - step * grad[0], old_q)
    delta = differences(before, after)
    assert delta["delta_assisted_J1"] < 0. < delta["delta_standalone_J2"]
    assisted_step = {"operator": "one exact gradient-descent step on J1",
                     "learning_rate": step, "gradient": grad,
                     "before": before, "after": after, "differences": delta,
                     "critic": "exact analytic teacher continuation",
                     "expressivity": "independent p,q can represent optimum(.8,.8)",
                     "interpretation": "assistance gap, with no approximation or optimization error"}

    # Same initial policy, now update from the return of its own two actions.
    grad_full = (4.5 * old_p - 2. - 2. * old_q, 2. * (old_q - old_p))
    full_after = panel(old_p - .1 * grad_full[0], old_q - .1 * grad_full[1])
    full_delta = differences(before, full_after)
    assert full_delta["delta_standalone_J2"] < 0.
    full_step = {"operator": "one exact gradient-descent step on J2",
                 "learning_rate": .1, "gradient": grad_full,
                 "before": before, "after": full_after, "differences": full_delta}

    # Teacher data supplies a reset at phase1 with stored a0=1, not an action label.
    # Feedback policy a1=k*a0 can learn a suffix while preserving a0=0 behavior.
    def feedback(p: float, k: float) -> dict:
        result = panel(p, k * p)
        result["policy"] = {"p": p, "k": k, "law": "a0=p; a1=k*stored_a0"}
        result["teacher_reset_suffix_K1"] = (k - 1.) ** 2
        return result

    feedback_before, feedback_suffix = feedback(0., 0.), feedback(0., 1.)
    feedback_final = feedback(.8, 1.)
    assert feedback_suffix["teacher_reset_suffix_K1"] < feedback_before["teacher_reset_suffix_K1"]
    assert feedback_before["standalone_J2"] == feedback_suffix["standalone_J2"] == 1.
    assert math.isclose(feedback_final["standalone_J2"], .2, abs_tol=1e-14)
    reverse = {"archive": {"controller": "independent coordinated expert",
                           "executed_actions": [1., 1.], "complete_cost": .25,
                           "suffix_reset": {"phase": 1, "stored_a0": 1.}},
               "learning_signal": "only costs of policy-executed suffixes and full episodes",
               "action_regression_targets": False,
               "suffix_operator": "exact minimization of own-return K1(k)=(k-1)^2",
               "root_operator": "exact minimization of own-return J2(p,k=1)=1.25*p^2-2*p+1",
               "before": feedback_before, "after_suffix_update": feedback_suffix,
               "suffix_differences": differences(feedback_before, feedback_suffix),
               "after_root_update": feedback_final,
               "root_differences": differences(feedback_suffix, feedback_final)}

    # The same reset curriculum with another expressive policy can interfere.
    phase_before, phase_after = panel(0., 0.), panel(0., 1.)
    phase_before["teacher_reset_suffix_K1"] = 1.
    phase_after["teacher_reset_suffix_K1"] = 0.
    assert phase_after["standalone_J2"] > phase_before["standalone_J2"]
    interference = {"operator": "exact suffix-own-return minimization q:0->1, holding p=0",
                    "before": phase_before, "after": phase_after,
                    "differences": differences(phase_before, phase_after),
                    "interpretation": "teacher-reset improvement does not imply root improvement; class still represents optimum"}

    gamma, horizon, raw, fallen_at = .999, 5, [2., 3.], 2
    executed = sum(gamma ** t * cost for t, cost in enumerate(raw))
    fall_tail = gamma ** fallen_at * absorbing(gamma, horizon - fallen_at)
    total = executed + fall_tail
    expected = sum(gamma ** t * ([2., 3., 5., 5., 5.][t]) for t in range(horizon))
    assert math.isclose(total, expected, abs_tol=1e-13)
    assert absorbing(gamma, 0) == 0.
    assert absorbing(1., horizon - fallen_at) == 15.
    assert absorbing(0., horizon - fallen_at) == 5.
    termination = {"gamma": gamma, "N": horizon, "fall_after_controls": fallen_at,
                   "executed_costs_including_falling_transition_once": raw,
                   "executed_discounted_cost": executed, "finite_absorbing_tail": fall_tail,
                   "complete_cost": total, "expanded_N_control_cost": expected,
                   "healthy_N_control_cutoff_tail": 0.,
                   "semantics": "absorption begins after fall; no infinite tail; no expert after terminal or N"}

    return {"kind": "exact_analytic_mechanism_check", "status": "all_assertions_passed",
            "scope": "finite two-step deterministic toy; no Walker evidence or learned critic",
            "equations": {"cost": ".25*a0^2+(a0-1)^2+(a1-a0)^2",
                          "J1": ".25*p^2+(p-1)^2",
                          "J2": "J1+(q-p)^2",
                          "general_gap_identity": "J_N(pi)=J_L^E(pi)+gamma^L*E_dLpi[V_pi(x_L,N-L)-V_E_restart(x_L,N-L)]",
                          "reverse_objective": "K_b(pi)=E_x~teacher_executed_states_at_b[V_pi(x,N-b)]"},
            "assisted_update": assisted_step, "full_return_comparator": full_step,
            "reverse_handover_feedback": reverse, "reverse_handover_interference": interference,
            "finite_termination": termination,
            "predeclared_checks": {"assistance_only_implication": "falsified if exact J1 decreases and J2 increases",
                                   "feedback_reset_transfer": "fails if suffix k update changes p=0 root return or full optimum is not .2",
                                   "reset_improvement_implies_root_improvement": "falsified if exact K1 decreases while root J2 increases",
                                   "finite_accounting": "fails if expanded finite cost differs or healthy N cutoff adds a tail"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.out_dir.resolve()
    # Refuse to overwrite any artifact, including an empty pre-existing directory.
    destination.mkdir(parents=True, exist_ok=False)
    result = run()
    source = Path(__file__).resolve()
    root = source.parents[2]
    result["provenance"] = {"python": platform.python_version(),
                            "source": str(source.relative_to(root)),
                            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                            "update_count": {"assisted": 1, "full_return_comparator": 1,
                                             "reverse_feedback": 2, "reverse_phase_only": 1}}
    shutil.copy2(source, destination / source.name)
    (destination / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "results": str(destination / "results.json"),
                      "assisted_delta": result["assisted_update"]["differences"],
                      "reverse_final_cost": result["reverse_handover_feedback"]["after_root_update"]["standalone_J2"]}))


if __name__ == "__main__":
    main()
