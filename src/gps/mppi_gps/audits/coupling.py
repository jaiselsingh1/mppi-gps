"""Independent saved-only audit of the H4.1 outcome-triggered coupling study.

This reconciles saved arrays, native costs, checkpoint state and work receipts.
It performs no policy inference, optimization or simulator replay.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from .common import audit_main, comparison_path, curve_csv, paired_cohort, physical_features
from .native import Audit


def audit_h4(
    native_checker: ModuleType,
    audit: Audit,
    study: Path,
    config: dict[str, Any],
    seal: dict[str, Any],
    stage: dict[str, Any],
    result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    arms = ("independent", "fixed_high", "backoff")
    high = 0.1199341983507337
    audit.need(
        config["weights"] == dict(independent=0.0, fixed_high=high, backoff=high / 2),
        "Actual first outcome-triggered weight adjustment",
    )
    audit.need(
        config["roots"]
        == dict(
            acquisition=952000000, planner=952100000, primary=952200000, confirmation=952300000
        ),
        "Frozen first-round seed roster",
    )
    audit.need(
        config["reserved_second_round_roots"]
        == dict(
            acquisition=954000000, planner=954100000, primary=954200000, confirmation=954300000
        ),
        "Disjoint second-round reservation corrected prelaunch",
    )
    audit.need(
        config["common_rows"] == 6096
        and config["teacher_rows"] == 2000
        and (config["fit_checkpoints"] == [500, 1000, 2000]),
        "H4 fixed data/update roster",
    )
    parent = audit.checkpoint(study / "input.pt")
    parent_optimizer = parent["optimizer_state"]
    owner = parent["policy_hash"]
    audit.need(
        owner
        == config["parent_hash"]
        == "7bf162f6e8c9bb63718e44f8392838503528a6669e7553088a07f13be6036916",
        "Actual updated v3.1 guided child participates in new acquisition",
    )
    audit.need(
        result["policy_hashes"].get("parent") == owner, "Reported incoming parent policy identity"
    )
    inspection = audit.load(study / "inputs/parent_inspection.json")
    audit.need(
        native_checker.fingerprint(parent_optimizer)
        == inspection["optimizer_tree_sha256"]
        == result["matching"]["optimizer_tree_sha256"],
        "Same incoming full Adam/RNG state",
    )
    normal = {key: parent["state_dict"]["observation_" + key].numpy() for key in ("mean", "std")}
    audit.need(
        native_checker.fingerprint(normal) == inspection["normalization_sha256"],
        "Unchanged clipped-input normalization",
    )
    with np.load(study / "inputs/history.npz", allow_pickle=False) as f:
        history = {key: np.array(f[key]) for key in ("observations", "actions")}
    expected_data = dict(
        rows=6096,
        observations_sha256=native_checker.fit_hash(history["observations"]),
        actions_sha256=native_checker.fit_hash(history["actions"]),
    )
    audit.need(
        parent_optimizer["fit_data"] == inspection["fit_data"] == expected_data
        and parent_optimizer["updates"] == 12000,
        "Actual updated learner owns retained6096 guided history",
    )
    comparison = comparison_path(args, config)
    for canonical, digest in seal["historical_files"].items():
        path = comparison / Path(canonical).relative_to(Path(config["comparison_run"]))
        audit.bind(path, digest)
    with np.load(comparison / "inputs/guided.npz", allow_pickle=False) as f:
        audit.need(
            all((np.array_equal(history[key], f[key]) for key in history)),
            "Exact prior guided data retained in all future arms",
        )
    feedback = audit.load(study / "inputs/feedback.json")
    old = audit.load(study / "inputs/original_comparison_result.json")
    audit.need(
        feedback["trigger"]
        and feedback["reasons"] == ["lost_full_survival"]
        and (feedback["entering_rho"] == high)
        and (feedback["proposed_rho"] == high / 2)
        and (feedback["next_acquisition_actor_hash"] == owner),
        "Saved matched outcome signal causes weight halving",
    )
    audit.need(
        old["cases"]["primary"]["guided"]["survived"] == 0
        and old["cases"]["primary"]["independent"]["survived"] == 3,
        "Response-rule input is the actual previous survival loss",
    )
    seed_proof = audit.load(study / "inputs/seed_audit.json")
    audit.need(
        seed_proof["conflicts"] == [] and seed_proof["planner_configs_checked"],
        "Prelaunch first/second seed freshness audit",
    )
    traces = {}
    for tag in stage["episodes"]:
        _, d = audit.episode(stage, tag)
        traces[tag] = d
    teacher_tags = {tag for score in result["teacher_summaries"].values() for tag in score["tags"]}
    audit.need(
        len(teacher_tags) == 6
        and all((len(result["teacher_summaries"][arm]["tags"]) == 2 for arm in arms)),
        "All six fixed teacher attempts retained",
    )
    before = {
        (row["episode"], row["step"]): row["private"]
        for row in stage["events"]["independent_planner_before"]
    }
    after = {
        (row["episode"], row["step"]): row["private"]
        for row in stage["events"]["independent_planner_after"]
    }
    decisions = {(row["episode"], row["step"]): row for row in stage["events"]["decision"]}
    teacher_complete = {row["episode"]: row for row in stage["events"]["teacher_complete"]}
    model_expected = 0
    teacher_live = 0
    guidance = 0
    posthoc = 0
    for arm in arms:
        score = result["teacher_summaries"][arm]
        audit.summary(stage, score)
        for i, tag in enumerate(score["tags"]):
            event = stage["episodes"][tag]
            d = traces[tag]
            n = event["steps"]
            teacher_live += n
            rho = config["weights"][arm]
            audit.need(
                event["seed"] == config["roots"]["acquisition"] + i, "Matched teacher reset seed"
            )
            model_expected += (n + 1) * 256 * 48 + (2 if rho == 0 else 3) * n * 48
            changes = 0
            for t, info in enumerate(d["infos"]["rows"]):
                decision = info["controller"]
                b = before[tag, t]
                e = after[tag, t]
                audit.need(
                    decision["actor_hash"] == owner and decision["alpha"] == rho,
                    "Actual changed weight and updated actor at every teacher decision",
                )
                audit.need(
                    not decision["raw_diagnostics"]["coupling_active"]
                    and decision["raw_diagnostics"]["cost_track_mean"] == 0.0,
                    "Native independent producer has no policy coupling",
                )
                audit.need(
                    decision["enabled"] == (rho > 0)
                    and decision["influence_used"] == (rho > 0)
                    and (not decision["numerical_revocation"]),
                    "Declared intervention actually enabled without revocation",
                )
                audit.need(
                    np.array_equal(d["actions"][t], np.asarray(decision["guided_first_action"])),
                    "Recorded selected control was actually executed",
                )
                changes += int(
                    rho > 0
                    and np.linalg.norm(
                        np.asarray(decision["guided_first_action"])
                        - decision["baseline_first_action"]
                    )
                    > 1e-06
                )
                audit.need(
                    set(b) == set(e) == {"U", "rng_state", "initial_plan"},
                    "Complete planner private state schema",
                )
                audit.need(
                    np.asarray(b["U"]).shape == np.asarray(e["U"]).shape == (48, 6)
                    and np.isfinite(b["U"]).all()
                    and np.isfinite(e["U"]).all(),
                    "Finite full planner nominal snapshots",
                )
                audit.need(
                    b["rng_state"]["bit_generator"] == e["rng_state"]["bit_generator"] == "PCG64",
                    "Private planner RNG ownership",
                )
                audit.need(
                    b["initial_plan"] == (t == 0) and (not e["initial_plan"]),
                    "One first-decision refinement block",
                )
                if t:
                    audit.need(
                        b == after[tag, t - 1],
                        "All consecutive private before/after snapshots continuous",
                    )
                audit.need(
                    decisions[tag, t]["bank_digest"] == decision["bank_digest"],
                    "Decision trace/ledger bank identity",
                )
            record = result["teacher_records"][tag]
            audit.need(
                record["actor_hash"] == owner
                and record["rho"] == rho
                and (record["changed_vs_own_baseline"] == changes),
                "Teacher result binds updated actor and actual changed controls",
            )
            episode_end = int(event["id"].split("-")[1])
            complete_end = int(teacher_complete[tag]["id"].split("-")[1])
            for row in stage["events"]["measurement_complete"]:
                if row["measure_kind"] != "neural_rows" or row.get("episode") != tag:
                    continue
                sequence = int(row["id"].split("-")[1])
                if sequence < episode_end:
                    guidance += row["amount"]
                elif sequence < complete_end:
                    posthoc += row["amount"]
            audit.need(record["raw"] == event["raw"], "Complete actual teacher raw ref")
    audit.need(
        len(before) == len(after) == len(decisions) == teacher_live,
        "All teacher controls have full before/after/decision records",
    )
    for i in range(2):
        reference = result["teacher_summaries"]["independent"]["tags"][i]
        for arm in ("fixed_high", "backoff"):
            tag = result["teacher_summaries"][arm]["tags"][i]
            for key in ("states", "warmstarts", "integrations", "observations"):
                audit.need(
                    np.array_equal(traces[tag][key][0], traces[reference][key][0]),
                    "Same full teacher reset before trajectory divergence",
                )
            audit.need(
                before[tag, 0] == before[reference, 0],
                "Same entering private nominal and full RNG state",
            )
            for t in range(
                min(stage["episodes"][tag]["steps"], stage["episodes"][reference]["steps"])
            ):
                audit.need(
                    before[tag, t]["rng_state"] == before[reference, t]["rng_state"],
                    "Paired independent RNG stream remains aligned across different physical paths",
                )
    manipulation = {}
    for label in ("fixed_high_vs_independent", "backoff_vs_independent", "backoff_vs_fixed_high"):
        left, right = label.split("_vs_")
        values = []
        for i in range(2):
            first = traces[result["teacher_summaries"][left]["tags"][i]]
            second = traces[result["teacher_summaries"][right]["tags"][i]]
            n = min(len(first["actions"]), len(second["actions"]))
            delta = np.linalg.norm(first["actions"][:n] - second["actions"][:n], axis=1)
            changed = np.flatnonzero(delta > 1e-06)
            byte = np.flatnonzero(np.any(first["actions"][:n] != second["actions"][:n], axis=1))
            value = dict(
                shared_prefix_controls=n,
                changed_action_rows=len(changed),
                first_changed_control=int(changed[0]) if len(changed) else None,
                first_byte_differing_action_control=int(byte[0]) if len(byte) else None,
                first_change_norm=float(delta[changed[0]]) if len(changed) else None,
                max_action_difference_norm=float(delta.max()),
                byte_identical_actions=np.array_equal(first["actions"], second["actions"]),
                byte_identical_observations=np.array_equal(
                    first["observations"], second["observations"]
                ),
            )
            audit.need(
                result["manipulation"][label][i] == value,
                "Independently recounted actual trajectory influence",
            )
            values.append(value)
        manipulation[label] = values
    bank_count = 0
    if args.compact_hdf is not None:
        locator = json.loads(args.raw_locator.read_text())
        kept = set(locator["compact"]["groups"])
    else:
        wanted = {(tag, t) for tag, t in decisions if t in (0, 500, 999)}
        for label, pairs in result["manipulation"].items():
            left, right = label.split("_vs_")
            for i, pair in enumerate(pairs):
                for name in ("first_changed_control", "first_byte_differing_action_control"):
                    if pair[name] is not None:
                        wanted.add((result["teacher_summaries"][left]["tags"][i], pair[name]))
                        wanted.add((result["teacher_summaries"][right]["tags"][i], pair[name]))
        kept = {decisions[key]["evidence"]["group"] for key in wanted}
    for (tag, t), decision in decisions.items():
        if decision["evidence"]["group"] not in kept:
            continue
        d = audit.group(study / "evidence/raw.h5", decision["evidence"])
        bank_count += 1
        audit.need(
            np.array_equal(d["initial_state"], traces[tag]["states"][t])
            and np.array_equal(d["initial_warmstart"], traces[tag]["warmstarts"][t]),
            "Retained bank actual simulator state/warmstart",
        )
        audit.need(
            np.array_equal(d["reference_u"], np.asarray(before[tag, t]["U"]))
            and np.array_equal(d["selected_plan"][0], traces[tag]["actions"][t]),
            "Retained bank private reference and actual executed control",
        )
        audit.need(
            d["bank_actions"].shape == (257, 48, 6)
            and d["bank_post_states"].shape == (257, 48, 19),
            "Complete retained final-pass bank and verified native mean",
        )
        shifted = np.concatenate((d["bank_actions"][-1, 1:], d["bank_actions"][-1, -1:]), axis=0)
        audit.need(
            np.array_equal(np.asarray(after[tag, t]["U"]), shifted),
            "Protected producer advances its own native mean, not the selected guided plan",
        )
        pre = np.concatenate(
            (np.broadcast_to(d["initial_state"], (257, 1, 19)), d["bank_post_states"][:, :-1]),
            axis=1,
        )
        audit.need(
            np.array_equal(
                d["bank_observations"], physical_features(pre, "clipped"), equal_nan=True
            ),
            "Retained bank actual full states produce declared clipped student inputs",
        )
        audit.need(
            np.allclose(
                d["baseline_scores"],
                d["bank_raw_costs"] + d["proximity_costs"],
                atol=1e-10,
                rtol=1e-12,
                equal_nan=True,
            ),
            "Retained baseline objective components",
        )
        audit.need(
            np.allclose(
                d["guided_scores"],
                d["baseline_scores"] + 0.5 * decision["alpha"] * d["agreement_squared"],
                atol=1e-10,
                rtol=1e-12,
                equal_nan=True,
            ),
            "Actual changed weight enters retained trajectory objective",
        )
    raw = result["teacher_summaries"]["independent"]
    checks = {
        arm + "_task_success": score["episodes"] == 2
        and score["successes"] == 2
        and (score["survived"] == 2)
        and (score["mean_steps"] == 1000)
        for arm, score in result["teacher_summaries"].items()
    }
    checks.update(
        {
            arm + "_cost_guard": result["teacher_summaries"][arm]["mean_cost"]
            <= 1.05 * raw["mean_cost"]
            for arm in ("fixed_high", "backoff")
        }
    )
    admitted = all(checks.values())
    audit.need(
        result["teacher_admission"]
        == dict(passed=admitted, checks=checks, agreement_used=False, per_slot_substitution=False),
        "Actual full-task teacher gate; no agreement rejection",
    )
    curves = {}
    final_owners = {"parent": owner}
    training = 0
    diagnostics = 0
    if not admitted:
        audit.need(
            result["status"] == "teacher_admission_negative"
            and (not result["learning_curves"])
            and (not result.get("datasets")),
            "Failed teachers stop child fitting without substitution",
        )
        audit.need(
            stage["counts"]["completed_optimizer_steps"]
            == stage["counts"]["completed_backward_rows"]
            == 0,
            "No unadmitted data trained a child",
        )
    else:
        rng = np.random.Generator(np.random.PCG64(0))
        rng.bit_generator.state = deepcopy(parent_optimizer["rng"])
        indices = []
        at = {}
        for t in range(1, 2001):
            indices.append(rng.integers(0, 8096, size=256))
            if t in (500, 1000, 2000):
                at[t] = deepcopy(rng.bit_generator.state)
        stored = audit.group(study / "evidence/raw.h5", result["matching"]["sampled_indices"])
        audit.need(
            np.array_equal(stored["indices"], np.stack(indices))
            and result["matching"]["expected_terminal_pcg64"] == at[2000],
            "All three children share exact restored-PCG64 minibatches",
        )
        for arm in arms:
            with np.load(study / "datasets" / f"{arm}.npz", allow_pickle=False) as f:
                data = {key: np.array(f[key]) for key in ("observations", "actions")}
            canonical = audit.group(study / "evidence/raw.h5", result["datasets"][arm])
            for key, width in [("observations", 17), ("actions", 6)]:
                expected = np.concatenate(
                    [history[key]]
                    + [traces[tag][key] for tag in result["teacher_summaries"][arm]["tags"]]
                )
                audit.need(
                    data[key].shape == (8096, width)
                    and np.array_equal(data[key], expected)
                    and np.array_equal(data[key], canonical[key]),
                    "Exact common6096 plus own2000 executed teacher pairs",
                )
            previous = 0
            previous_owner = owner
            curve = []
            refresh = []
            for checkpoint in (500, 1000, 2000):
                checkpoint_data = audit.checkpoint(study / "checkpoints" / f"{arm}_{checkpoint}.pt")
                optimizer_state = checkpoint_data["optimizer_state"]
                fit_record = audit.load(study / "learning" / f"{arm}_{checkpoint}.json")
                audit.need(
                    optimizer_state["updates"] == 12000 + checkpoint
                    and optimizer_state["fit_updates"] == checkpoint
                    and (optimizer_state["rng"] == at[checkpoint]),
                    "H4 exact incoming Adam continuation and sampler position",
                )
                audit.need(
                    all(
                        (
                            float(item["step"]) == 12000 + checkpoint
                            for item in optimizer_state["adam"]["state"].values()
                        )
                    ),
                    "H4 Adam per-parameter cumulative counters",
                )
                audit.need(
                    checkpoint_data["policy_hash"]
                    == optimizer_state["policy_hash"]
                    == fit_record["child_policy_hash"],
                    "H4 checkpoint identities",
                )
                audit.need(
                    optimizer_state["fit_config"]
                    == dict(
                        batch_size=256,
                        lr=0.0003,
                        eta_theta=0.01,
                        gradient_clip=0.5,
                        prox_reduction="mean",
                        anchor_refresh=250,
                    ),
                    "H4 unchanged child objective",
                )
                audit.need(
                    optimizer_state["fit_data"]
                    == fit_record["fit_data"]
                    == dict(
                        rows=8096,
                        observations_sha256=native_checker.fit_hash(data["observations"]),
                        actions_sha256=native_checker.fit_hash(data["actions"]),
                    ),
                    "H4 actual targets/observations fitted",
                )
                audit.need(
                    fit_record["parent_policy_hash"] == previous_owner
                    and fit_record["optimizer_continued"]
                    and (fit_record["resume_fit"] == (previous > 0)),
                    "One new proximal block resumed at checkpoints",
                )
                audit.need(
                    native_checker.tensor_hash(optimizer_state["anchor"])
                    == optimizer_state["anchor_policy_hash"]
                    == fit_record["anchor_policy_hash"],
                    "Actual active proximal anchor packaged",
                )
                for key in ("mean", "std"):
                    audit.need(
                        torch.equal(
                            checkpoint_data["state_dict"]["observation_" + key],
                            parent["state_dict"]["observation_" + key],
                        ),
                        "H4 fixed original normalization",
                    )
                for kind, n in [
                    ("neural_rows", 256 * (checkpoint - previous)),
                    ("backward_rows", 256 * (checkpoint - previous)),
                    ("optimizer_steps", checkpoint - previous),
                ]:
                    audit.need(
                        fit_record["work"][kind]
                        == dict(requested=n, completed=n, unknown_partial=0),
                        "H4 exact individual fit work",
                    )
                refresh.extend(fit_record["anchor_refreshes"])
                curve.extend(fit_record["learning_curve"])
                previous = checkpoint
                previous_owner = checkpoint_data["policy_hash"]
            audit.need(
                [point["before_fit_update"] for point in refresh] == list(range(251, 2001, 250)),
                "H4 correct anchor refreshes across checkpoints",
            )
            audit.need(
                curve == result["learning_curves"][arm] and len(curve) == 2000,
                "H4 all2000 learning points",
            )
            for t, point in enumerate(curve, 1):
                audit.need(
                    point["update"] == 12000 + t and point["fit_update"] == t,
                    "H4 continuous global/block update chronology",
                )
            curves[arm] = curve
            final_owners[arm] = previous_owner
            audit.need(
                previous_owner == result["policy_hashes"][arm], "Final standalone child owner"
            )
        training = 1536000
        diagnostics = 12 * 8096
        audit.need(
            stage["counts"]["completed_optimizer_steps"] == 6000
            and stage["counts"]["completed_backward_rows"] == training,
            "H4 exact three2k fit budgets",
        )
    curve_csv(audit, study, curves)
    for tag, d in traces.items():
        if tag in teacher_tags:
            continue
        for info in d["infos"]["rows"]:
            c = info["controller"]
            audit.need(
                c.get("policy_hash") in final_owners.values()
                and c == dict(policy_hash=c["policy_hash"], feature_mode="state"),
                "H4 standalone torque-policy execution only",
            )
    for cohort, scores in result["cases"].items():
        root = (
            config["roots"]["acquisition"]
            if cohort == "acquisition_starts"
            else config["roots"][cohort]
        )
        paired_cohort(
            audit, stage, traces, scores, root, 2 if cohort == "acquisition_starts" else 16
        )
        audit.need(
            set(scores) == set(final_owners), "Correct parent-only or four-policy standalone roster"
        )
        for arm, value in scores.items():
            for tag in value["tags"]:
                audit.need(
                    all(
                        (
                            info["controller"]["policy_hash"] == final_owners[arm]
                            for info in traces[tag]["infos"]["rows"]
                        )
                    ),
                    "Final cohort policy owner per arm",
                )
    if admitted:
        for cohort in ("primary", "confirmation"):
            if cohort not in result["cases"]:
                continue
            scores = result["cases"][cohort]
            g = scores["backoff"]
            checks = dict(
                minimum_walking_successes=g["successes"] >= 1,
                more_walking_than_fixed=g["successes"] > scores["fixed_high"]["successes"],
            )
            comparisons = {}
            for arm in ("parent", "independent", "fixed_high"):
                control = scores[arm]
                gain = (control["mean_cost"] - g["mean_cost"]) / control["mean_cost"]
                part = dict(
                    success_nonregression=g["successes"] >= control["successes"],
                    survival_nonregression=g["survived"] >= control["survived"],
                    mean_controls_nonregression=g["mean_steps"] >= control["mean_steps"],
                    complete_cost=gain >= (0 if arm == "parent" else 0.05),
                )
                comparisons[arm] = dict(
                    checks=part,
                    relative_cost_gain=gain,
                    passed=all(part.values()),
                    paired_cost_deltas=(np.asarray(g["costs"]) - control["costs"]).tolist(),
                    paired_control_deltas=(np.asarray(g["steps"]) - control["steps"]).tolist(),
                )
            gate = dict(
                passed=all(checks.values())
                and all((value["passed"] for value in comparisons.values())),
                checks=checks,
                comparisons=comparisons,
            )
            audit.need(
                result[cohort + "_gate"] == gate,
                "H4 independently recomputed walking/survival/cost gate",
            )
        audit.need(
            ("confirmation" in result["cases"]) == result["primary_gate"]["passed"],
            "H4 conditional confirmation follows only frozen primary gate",
        )
        confirmed = result["primary_gate"]["passed"] and result.get("confirmation_gate", {}).get(
            "passed", False
        )
        audit.need(
            result["confirmed_benefit"] == confirmed and result["next_round_eligible"] == confirmed,
            "H4 confirmed benefit distinct from preliminary or stationary gain",
        )
        proposal = result["next_weight_proposal"]
        primary = result["cases"]["primary"]
        trigger = (
            primary["backoff"]["successes"] < primary["independent"]["successes"]
            or primary["backoff"]["survived"] < primary["independent"]["survived"]
        )
        audit.need(
            proposal["trigger"] == trigger
            and proposal["proposed_rho"] == config["weights"]["backoff"] / (2 if trigger else 1),
            "H4 prospective response rule recomputed from new matched children",
        )
        if confirmed:
            audit.need(
                not trigger and result["guaranteed_hold_if_confirmed"],
                "Passing criteria necessarily imply next hold",
            )
    gap_scores = result["cases"]["acquisition_starts"]
    gap_key = "paired_teacher_student_gap" if admitted else "paired_teacher_incoming_parent_gap"
    for arm in arms:
        student = gap_scores[arm if admitted else "parent"]
        teacher = result["teacher_summaries"][arm]
        recorded = result[gap_key][arm]
        audit.need(recorded["teacher_costs"] == teacher["costs"], "Paired gap actual teacher costs")
        audit.need(
            recorded["student_costs" if admitted else "parent_costs"] == student["costs"],
            "Paired gap actual student/parent costs",
        )
        audit.equal(
            recorded["costs" if admitted else "parent_minus_teacher_costs"],
            np.asarray(student["costs"]) - teacher["costs"],
            "Same-context complete-cost gap",
        )
    work = stage["counts"]
    standalone = sum(
        (row["steps"] for tag, row in stage["episodes"].items() if tag not in teacher_tags)
    )
    audit.need(
        work["completed_model_controls"] == model_expected
        and work["completed_live_controls"] == teacher_live + standalone,
        "H4 all model and live controls exactly recounted",
    )
    audit.need(
        posthoc == teacher_live, "Exactly one diagnostic actor forward per actual teacher row"
    )
    audit.need(
        work["completed_neural_rows"] == guidance + posthoc + training + diagnostics + standalone,
        "Every guidance/teaching/fit/diagnostic/standalone neural row reconciled",
    )
    audit.need(result["loop_closed"] is False, "No unsupported closed-loop/convergence claim")
    return dict(
        status=result["status"],
        teacher_summaries=result["teacher_summaries"],
        teacher_admission=result["teacher_admission"],
        manipulation=manipulation,
        retained_full_selection_banks_checked=bank_count,
        private_before_after_snapshots_checked=len(before) + len(after),
        cases=result["cases"],
        primary_gate=result.get("primary_gate"),
        exact_work=dict(work),
        neural_breakdown=dict(
            guidance=guidance,
            teacher_diagnostic=posthoc,
            training=training,
            full_data_diagnostic=diagnostics,
            standalone=standalone,
        ),
        gap=result[gap_key],
        model_array_scope="Only retained selected full banks inspected; all other query work and private snapshots recounted from original full ledger. No dynamics replay or model optimizer arithmetic replay.",
    )


def main(argv: list[str] | None = None) -> int:
    """Audit an explicitly supplied closed H4.1 run into a fresh directory."""
    return audit_main("h4", audit_h4, argv, source=Path(__file__))


if __name__ == "__main__":
    raise SystemExit(main())
