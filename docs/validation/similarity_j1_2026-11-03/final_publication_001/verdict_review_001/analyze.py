"""Saved development tables only: descriptive J1 closure, no new scientific execution."""
from __future__ import annotations

import hashlib
import json
import resource
import statistics
import time
from pathlib import Path

START = time.monotonic()
BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "audited_summary_read_002"
OUT = Path(__file__).resolve().parent
READS = 0
BINDINGS = {}


def load(name):
    global READS
    path = INPUT / name
    payload = path.read_bytes()
    READS += len(payload)
    if READS > 16 * 1024**2:
        raise ValueError("Saved-metadata review read limit")
    BINDINGS[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    return json.loads(payload)


def stats(values):
    values = list(values)
    return dict(count=len(values), mean=statistics.fmean(values), population_sd=statistics.pstdev(values), minimum=min(values), maximum=max(values), median=statistics.median(values))


worker = load("worker_result.json")
cases = load("cases.json")
models = load("score_models.json")
probe = load("probe_summary.json")
audit = load("audit.json")
assert audit["passed"] is audit["full_scientific_audit_passed"] is True
assert worker["status"] == "CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE"
assert worker["protected_test_revealed"] is False
assert audit["objective_passed"] is worker["objective_passed"] is False
assert {row["split"] for row in cases} == {"development"}
assert len(cases) == 32 and len({row["group"] for row in cases}) == 8
assert audit["worker_result_sha256"] == BINDINGS["worker_result.json"]["sha256"]
for name in ("cases.json", "score_models.json", "probe_summary.json"):
    assert audit["input_files"]["worker/" + name] == BINDINGS[name]

summary = {}
for source in ("mppi", "reference"):
    rows = [row for row in cases if row["source"] == source]
    evaluations = [outcome for row in rows for outcome in row["evaluations"]]
    teachers = {row["teacher"]["tag"]: row["teacher"] for row in rows}
    positive = [row for row in rows if row["walking_successes"] > 0]
    full = [row for row in rows if row["motion_components"]["full_window"]]
    mean_scores = {name: stats(row["scores"][name] for row in rows) for name in "ABCDE"}
    summary[source] = dict(
        lineage_groups=len({row["group"] for row in rows}), learning_cases=len(rows),
        teachers=dict(episodes=len(teachers), walking=sum(v["walking"] for v in teachers.values()), survived=sum(v["survived"] for v in teachers.values()), cost=stats(v["cost"] for v in teachers.values())),
        standalone=dict(episodes=len(evaluations), walking=sum(v["walking"] for v in evaluations), full_survival=sum(v["survived"] for v in evaluations), controls=stats(v["controls"] for v in evaluations), complete_cost=stats(v["cost"] for v in evaluations)),
        child_rates=dict(positive_children=len(positive), positive_source_groups=len({row["group"] for row in positive}), maximum_walking_per16=max(row["walking_successes"] for row in rows), cases=[dict(case_id=row["case_id"], walking=row["walking_successes"], trials=row["walking_trials"], full_survival=sum(v["survived"] for v in row["evaluations"]), post_motion=row["post_motion"], reproduction_controls=row["reproduction"]["controls"], reproduction_walking=row["reproduction"]["walking"]) for row in rows]),
        reproduction=dict(binary_successes=sum(row["reproduced"] for row in rows), full_windows=len(full), walking_at_teacher_start=sum(row["reproduction"]["walking"] for row in rows), motion=stats(row["post_motion"] for row in rows), controls=stats(row["reproduction"]["controls"] for row in rows), full_window_components=[dict(case_id=row["case_id"], **{key: row["motion_components"][key] for key in ("joint_angle_rms", "forward_velocity_rms", "height_rms", "final_relative_progress_error")}) for row in full]),
        pre_update_scores=mean_scores,
        C_components={name:stats(row["components"]["C"][name] for row in rows) for name in ("explained_mse", "residual_mse")},
    )

pairs = []
for learner in sorted({row["learner"] for row in cases}):
    a = next(row for row in cases if row["learner"] == learner and row["source"] == "mppi")
    b = next(row for row in cases if row["learner"] == learner and row["source"] == "reference")
    assert a["incoming_bindings"] == b["incoming_bindings"]
    assert a["incoming"] == b["incoming"] and a["evaluation_seeds"] == b["evaluation_seeds"]
    av, bv = a["evaluations"], b["evaluations"]
    pairs.append(dict(group=a["group"], learner=learner, walking_gain_reference_minus_mppi=b["walking_successes"]-a["walking_successes"], survival_gain_reference_minus_mppi=sum(v["survived"] for v in bv)-sum(v["survived"] for v in av), mean_cost_difference_reference_minus_mppi=statistics.fmean(v["cost"] for v in bv)-statistics.fmean(v["cost"] for v in av), motion_difference_reference_minus_mppi=b["post_motion"]-a["post_motion"]))

full_rows = [row for row in cases if row["motion_components"]["full_window"]]
failures = dict(fell_before_full_window=sum(not row["motion_components"]["full_window"] for row in cases), full_window_angle_gate_failed=sum(row["motion_components"]["joint_angle_rms"]>0.25 for row in full_rows), full_window_forward_velocity_gate_failed=sum(row["motion_components"]["forward_velocity_rms"]>0.3 for row in full_rows), full_window_height_gate_failed=sum(row["motion_components"]["height_rms"]>0.1 for row in full_rows), full_window_progress_gate_failed=sum(row["motion_components"]["final_relative_progress_error"]>0.5 for row in full_rows), note="Only full-window cases are classified by full-window error thresholds. Counts can overlap; missing post-fall progress stays missing.")
factorial = [dict(context=row["context"], nominal_change_mse=stats(row["nominal_action_squared_changes"]), carried_private_rng_mean_action_variance=statistics.fmean(row["carried_rng_action_variance"]), zero_nominal_private_rng_mean_action_variance=statistics.fmean(row["zero_rng_action_variance"])) for row in probe["factorial_contrasts"]]
gains = [{key:value for key,value in row.items() if key not in ("gain_per_normalized_observation", "gain_per_physical_observation", "gain_standardized", "held_residuals", "singular_values")} for row in probe["local_fits"]]

result = dict(
    schema="walker-j1-development-causal-review-v1",
    input_bindings=BINDINGS,
    audited_source=dict(source_commit="8c678e507dad1e22708f02a121886d7951d1da44", config_sha256=audit["config_sha256"], source_manifest_sha256=audit["source_manifest_sha256"], input_manifest_sha256=audit["input_manifest_sha256"], scientific_audit_passed=True, objective_passed=False, protected_test_opened=False),
    source_results=summary,
    paired_source_comparisons=pairs,
    paired_summary={name:dict(reference_higher=sum(row[name]>0 for row in pairs), reference_lower=sum(row[name]<0 for row in pairs), tied=sum(row[name]==0 for row in pairs)) for name in ("walking_gain_reference_minus_mppi", "survival_gain_reference_minus_mppi", "mean_cost_difference_reference_minus_mppi", "motion_difference_reference_minus_mppi")},
    reproduction_failures=failures,
    prediction_development_gate={key:value for key,value in models["prediction_models"].items() if key!="models"},
    constant_MPPI_walking_prediction={name:row["walking_constant_probability"] for name,row in models["prediction_models"]["models"].items()},
    probe=dict(decisions=probe["decisions"], physical_contexts=4, source_trajectories=2, gain_contexts=2, gain_context_innovation_fits=4, original_replay_checks=probe["original_replay_checks"], factorial=factorial, hidden_input_contrasts=probe["hidden_input_contrasts"], gains=gains, held_out_fit_errors_are_not_task_outcomes=True, full_column_rank_is_design_rank_not_controller_gain_rank=True),
    actual_worker_costs=worker["costs"],
    actual_audit_costs=audit["work"],
    established=["All eight MPPI and eight reference teachers walked; lower native teacher cost did not guarantee successful transfer to this learner/update.", "MPPI supervision produced no standalone walking in256 resets over16 fresh children. The reference source produced35 walking resets across8 children in6 groups, with maximum6/16 for any child. Neither source yielded binary motion reproduction.", "Primary MPPI motion variation was0.03849315344778441, below0.1, and MPPI walking labels were all failures. The declared development gate correctly prevented acquisition/fitting of the protected block.", "At the four fixed contexts, changing carried nominal or private innovations changes selected actions. The tested time/root-x/warmstart perturbations had zero or numerical-roundoff action effects.", "A rank17 perturbation design does not certify a nonzero, accurate or optimal feedback law: two fits were exactlyzero; the other two had held errors0.07117 and0.12683 without measured contact-regime changes."],
    supported_interpretations=["The frozen fresh-parent/2000-update MPPI benchmark cannot identify walking prediction and has insufficient declared motion variation. This is an identification failure of the version, not a protected comparison in which all scores lost to MSE.", "Teacher source affects observed student outcomes under matched incoming packages, tapes and update counts, but source bundles different states, targets and trajectory structure; this contrast does not identify which feature caused the difference.", "The reference result demonstrates occasional learning of walking with width128 and2000updates; it does not prove capacity is sufficient for all MPPI targets or establish reliable individual competence.", "The observed MPPI action variation motivates treating residuals cautiously; it does not prove action averaging, smoothing, memory deletion or transport invariance is safe."],
    unresolved=["Whether any pre-update candidate predicts a genuinely useful MPPI learning update better than MSE.", "How much teacher memory/randomness, coverage, action consequences, trajectory structure, finite fitting or policy capacity caused the transfer gap.", "Whether learned incoming packages create useful MPPI learning-outcome variation, or merely retain pre-existing walking or regress under new supervision.", "Whether any change improves a next standalone child while preserving teachers over multiple acquisition-learning rounds."],
    recommended_next_version="J1.1-B: development-only incoming-package/outcome-variation pilot with learned and original fresh packages, matched no-update controls, unchanged scientific interface and no protected launch.",
    no_new_scientific_work=dict(network_calls=0, native_controls=0, model_controls=0, neural_rows=0, fits=0, optimizer_steps=0),
    scope="Descriptive arithmetic on already audited development metadata. No new predictor fit, bootstrap, raw-array audit, policy inference, physics or experiment; counts of episodes/probe calls are not independent task replications.",
)
payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
assert len(payload)<2*1024**2 and time.monotonic()-START<60
with (OUT/"review.json").open("xb") as stream:
    stream.write(payload)
usage=resource.getrusage(resource.RUSAGE_SELF)
receipt=dict(schema="walker-j1-development-review-process-v1", passed=True, source_read_bytes=READS, wall_seconds=time.monotonic()-START, user_seconds=usage.ru_utime, system_seconds=usage.ru_stime, peak_rss_bytes=usage.ru_maxrss, output=dict(file="review.json",bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()), new_scientific_work=result["no_new_scientific_work"], limitations="Earlier read-only shell summaries are additional unmetered engineering review overhead, not free science.")
with (OUT/"receipt.json").open("x") as stream:
    json.dump(receipt,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
print(json.dumps(dict(passed=True, source_read_bytes=READS, summary={source:summary[source]["standalone"] for source in summary}, pairs=result["paired_summary"])))
