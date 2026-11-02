"""Recompute preliminary H5 closure arithmetic from one saved scalar probe.

No SSH, arrays, checkpoint decoding, inference, fitting or physics. This is a
metadata consistency review pending the independent full scientific audit.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
PROBE = ART / "engineering/028_closed_worker_probe/stdout.txt"
JSON_OUT = ART / "preliminary_closed_worker_review_001.json"
TABLE_OUT = ART / "preliminary_closed_worker_review_001.md"
ARMS = ("independent", "selector_only", "proposal_guided")


def main():
    started = time.perf_counter()
    if JSON_OUT.exists() or TABLE_OUT.exists():
        raise FileExistsError("Preliminary review is create-only")
    payload = PROBE.read_bytes()
    source = Path(__file__).read_bytes()
    saved = json.loads(payload)
    worker = saved["worker_result.json"]
    work = worker["work"]
    reservations = worker["reservation_census"]
    summaries = worker["teacher_summaries"]
    assert worker["status"] == "teacher_competence_negative_no_child_fits"
    assert worker["execution_complete"] is True and worker["comparison_complete"] is False
    assert worker["confirmed_benefit"] is False and worker["next_round_eligible"] is False
    assert worker.get("error") is None
    assert saved["worker_exit.json"]["exit_code"] == 0
    assert saved["worker_exit.json"]["unknown_partial_work_possible"] is False
    assert saved["worker_exit.json"]["stopping_reason"] is None
    assert saved["worker_exit.json"]["retries"] == 0
    assert set(summaries) == {"0", "1", "2"}
    aggregate, paired, stream_rows = {}, [], []
    for arm in ARMS:
        costs = []
        successes = survived = controls = episodes = 0
        for stream in ("0", "1", "2"):
            values = summaries[stream][arm]
            assert values["episodes"] == len(values["costs"]) == 4
            assert values["complete"] is True
            mean_cost = sum(values["costs"]) / 4
            assert mean_cost == values["mean_cost"]
            costs.extend(values["costs"])
            successes += values["successes"]
            survived += values["survived"]
            controls += values["controls"]
            episodes += values["episodes"]
            base = summaries[stream]["independent"]["mean_cost"]
            ratio = mean_cost / base
            walking = values["successes"] == 4
            cost_pass = mean_cost <= 1.05 * base
            assert worker["teacher_admission"][stream][arm] == dict(walking=walking, cost=cost_pass)
            stream_rows.append(
                dict(
                    stream=int(stream),
                    arm=arm,
                    episodes=4,
                    walking=values["successes"],
                    full_survival=values["survived"],
                    controls=values["controls"],
                    mean_complete_cost=mean_cost,
                    cost_ratio_to_paired_independent=ratio,
                    cost_change_percent=100.0 * (ratio - 1.0),
                    walking_gate=walking,
                    five_percent_cost_gate=cost_pass,
                )
            )
            if arm != "independent":
                paired.append(
                    dict(
                        stream=int(stream),
                        contrast=arm + "_minus_independent",
                        mean_complete_cost_difference=mean_cost - base,
                        ratio_of_mean_complete_cost=ratio,
                        cost_change_percent=100.0 * (ratio - 1.0),
                        reset_order_cost_differences=[
                            second - first
                            for first, second in zip(
                                summaries[stream]["independent"]["costs"],
                                values["costs"],
                                strict=True,
                            )
                        ],
                        pairing_scope="Declared matched acquisition-reset order; exact simulator reset-context equality awaits the independent raw audit.",
                    )
                )
        aggregate[arm] = dict(
            episodes=episodes,
            walking=successes,
            full_survival=survived,
            controls=controls,
            complete_cost_sum=sum(costs),
            mean_complete_cost=sum(costs) / episodes,
            complete_costs=costs,
            aggregation="Episode equal, three streams by four reset slots. These are three shared-parent streams, not twelve independent trained learners.",
        )
    episode_count = sum(row["episodes"] for row in aggregate.values())
    live = sum(row["controls"] for row in aggregate.values())
    assert episode_count == 36 and live == 34400
    model_controls = 24768 * live + 24576 * episode_count
    expected = dict(
        requested_controls=model_controls + live,
        model_controls=model_controls,
        live_controls=live,
        physics_substeps=4 * (model_controls + live),
        model_queries=6 * live + 2 * episode_count,
        live_queries=live,
        episodes=episode_count,
        backward_rows=3 * 2000 * 256,
        optimizer_steps=3 * 2000,
        checkpoint_loads=3,
    )
    for counter, value in expected.items():
        assert reservations["requested"]["total"][counter] == value
        assert reservations["completed"]["total"][counter] == value
    for scope in reservations["unknown_partial"].values():
        assert all(value == 0 for value in scope.values())
    assert not reservations["pending"] and not reservations["stopped"]
    assert all(value == 0 for value in reservations["errors"].values())
    assert work["completed_model_controls"] == model_controls
    assert work["completed_live_controls"] == live
    assert work["completed_controls"] == model_controls + live
    assert work["completed_physics_substeps"] == expected["physics_substeps"]
    assert work["completed_backward_rows"] == expected["backward_rows"]
    assert work["completed_optimizer_steps"] == expected["optimizer_steps"]
    assert work["completed_checkpoint_loads"] == expected["checkpoint_loads"]
    parent_forward_rows = expected["backward_rows"]
    neural = work["completed_neural_rows"]
    forward_calls = reservations["completed"]["total"]["prediction_queries"]
    per_arm = {}
    for arm in ARMS:
        row = reservations["completed"]["arm:" + arm]
        controls = aggregate[arm]["controls"]
        assert row["model_controls"] == 24768 * controls + 24576 * 12
        assert row["model_queries"] == 6 * controls + 24
        assert row["live_controls"] == controls
        assert row["physics_substeps"] == 4 * (row["model_controls"] + controls)
        assert row["optimizer_steps"] == row["backward_rows"] == row["checkpoint_loads"] == 0
        assert row["prediction_queries"] == 3 * controls + 12 + 12
        per_arm[arm] = dict(
            model_controls=row["model_controls"],
            live_controls=controls,
            requested_native_controls=row["model_controls"] + controls,
            physics_substeps=row["physics_substeps"],
            model_queries=row["model_queries"],
            reported_neural_rows=row["neural_rows"],
            posthoc_teacher_rows=controls,
            inferred_guidance_rows=row["neural_rows"] - controls,
            reported_forward_calls=row["prediction_queries"],
            inferred_guidance_forward_calls=row["prediction_queries"] - 12,
            posthoc_teacher_forward_calls=12,
        )
    shared_neural = neural - sum(row["reported_neural_rows"] for row in per_arm.values())
    shared_calls = forward_calls - sum(row["reported_forward_calls"] for row in per_arm.values())
    parent_metric_rows = shared_neural - parent_forward_rows
    parent_metric_calls = shared_calls - expected["optimizer_steps"]
    assert parent_metric_rows == 36000 and parent_metric_calls == 36
    audit_rows = neural - parent_forward_rows
    audit_calls = forward_calls - expected["optimizer_steps"]
    assert audit_rows == 1085983492 and audit_calls == 103308
    assert work["measurement_requests"] == forward_calls + 6000 + 6000 + 3
    state = dict(
        status="pending_independent_scientific_audit",
        metadata_arithmetic_consistent=True,
        scientific_audit_passed=False,
        source=dict(
            file=str(PROBE), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        ),
        review_source=dict(
            file=str(Path(__file__)), bytes=len(source), sha256=hashlib.sha256(source).hexdigest()
        ),
        source_review_read_bytes=len(payload) + len(source),
        inherited_probe=dict(
            read_bytes=saved["read_bytes"],
            wall_seconds=saved["wall_seconds"],
            scientific_calls=saved["scientific_calls"],
            remote_file_bindings=saved["files"],
        ),
        teacher_aggregate_by_arm=aggregate,
        teacher_per_stream=stream_rows,
        paired_complete_cost_comparisons=paired,
        teacher_totals=dict(episodes=36, walking=34, full_survival=34, controls=live),
        expected_and_reported_work=expected,
        per_arm_work=per_arm,
        parent_warmup=dict(
            fits_expected_by_protocol=3,
            updates_per_fit_expected_by_protocol=2000,
            reported_total_optimizer_steps=6000,
            reported_forward_rows=parent_forward_rows,
            reported_backward_rows=1536000,
            inferred_full_data_metric_rows=parent_metric_rows,
            inferred_full_data_metric_forward_calls=parent_metric_calls,
            expected_metric_events=12,
            expected_checkpoint_files=6,
            reported_checkpoint_loads=3,
            ownership_limit="Per-fit tapes, Adam/RNG/normalization and file roster require the independent audit; this scalar probe does not contain all fit/checkpoint records.",
        ),
        neural_accounting=dict(
            reported_worker_forward_rows=neural,
            reported_worker_forward_calls=forward_calls,
            parent_fitting_forward_rows=parent_forward_rows,
            parent_fitting_forward_calls=6000,
            teacher_posthoc_rows=live,
            teacher_posthoc_calls=36,
            inferred_guidance_rows=sum(row["inferred_guidance_rows"] for row in per_arm.values()),
            inferred_guidance_calls=sum(
                row["inferred_guidance_forward_calls"] for row in per_arm.values()
            ),
            expected_full_prediction_audit_rows=audit_rows,
            expected_full_prediction_audit_calls=audit_calls,
            expected_audit_checkpoint_loads=6,
            not_the_full_roster_checkpoint_load_ceiling=15,
            training_forward_or_gradient_reproduction_in_audit=False,
            audit_status="Expected paid audit roster from the closed worker; not yet claimed completed.",
        ),
        unmeasured=dict(
            child_fits=dict(count=0, reason="Frozen normal-teacher-competence stop"),
            standalone_policy_walking=None,
            standalone_policy_survival=None,
            student_cost=None,
            teacher_student_gap=None,
            child_reconstruction=None,
            child_held_error=None,
            primary_panel=None,
            confirmation_panel=None,
            useful_learning=None,
            scope="No child, standalone parent/student panels or paired gap episodes executed; missing outcomes are not zero-performance measurements.",
        ),
        closure=dict(
            worker_status=worker["status"],
            worker_execution_complete=worker["execution_complete"],
            learning_comparison_complete=False,
            confirmed_benefit=False,
            next_round_eligible=False,
            no_automatic_followup=True,
        ),
        reported_resource_accounting=dict(
            meter_wall_seconds=work["wall_seconds"],
            worker_wall_seconds=worker["wall_seconds"],
            outer_worker_wall_seconds=saved["worker_exit.json"]["wall_seconds"],
            outer_observed_peak_rss_bytes=saved["worker_exit.json"]["observed_peak_rss_bytes"],
            meter_evidence_storage_bytes=work["storage_bytes"],
            external_files_and_full_run_size="Not supplied by this probe; wait for audited external catalog.",
            allocation_core_wall_hours_upper=saved["worker_exit.json"]["wall_seconds"] * 16 / 3600,
            scope="Nested wall times must not be added. Sixteen-core allocation times elapsed wall is an allocation upper bound, not measured CPU core-hours. Billing, energy and engineering overhead remain separate/unknown.",
        ),
        causal_limits=[
            "The scalar worker declarations and arithmetic await the full independently recomputed raw-source/query/prediction/checkpoint/outcome audit.",
            "I passed its declared acquisition roster; S and P each lost one complete walking episode in different parent/reset contexts. This is a teacher-competence gate failure for this fixed placement recipe.",
            "P versus S is not a measured student-usefulness contrast because no child was fitted. Neither failed cloning, a bad agreement metric, nor a benefit from another representation or temporal objective follows.",
            "The local native-cost guard did not guarantee full-episode competence. Different competent motions were allowed; matching a reference path was not required.",
            "The guide stayed fixed within round one; no updated policy participated in another acquisition round, and no adaptive loop or convergence was measured.",
        ],
        review_wall_seconds=time.perf_counter() - started,
        review_scientific_calls=0,
        read_accounting_scope="Only the saved scalar probe and this helper source were read; no raw arrays, ledgers, original checkpoints or remote files were opened. Library/import/stat work is unmetered; JSON/Markdown writes follow the inner timer.",
    )
    table = [
        "# Preliminary closed-worker review — independent scientific audit pending",
        "",
        "These are recomputed scalar declarations, not an independent raw-evidence audit pass.",
        "",
        "| Arm | Walking /12 | Full survival /12 | Controls | Mean complete cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = aggregate[arm]
        table.append(
            f"| {arm} | {row['walking']} | {row['full_survival']} | {row['controls']:,} | {row['mean_complete_cost']:.6f} |"
        )
    table += [
        "",
        "| Stream | Guided arm | Mean cost / paired I mean | Cost change | Walking gate | 5% cost gate |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in stream_rows:
        if row["arm"] != "independent":
            table.append(
                f"| {row['stream']} | {row['arm']} | {row['cost_ratio_to_paired_independent']:.9f} | {row['cost_change_percent']:+.6f}% | {row['walking_gate']} | {row['five_percent_cost_gate']} |"
            )
    table += [
        "",
        f"Reported work: {model_controls:,} model controls + {live:,} live controls = {model_controls + live:,} native controls; {expected['physics_substeps']:,} physics substeps. NN work is separately {neural:,} forward rows, {1536000:,} backward rows and {6000:,} Adam updates, all learning from the three parent warm-ups.",
        "",
        f"Expected independent prediction audit: {audit_rows:,} rows in {audit_calls:,} forward calls and six checkpoint loads, with no training or native replay. This is an expected audit census, not completed audit work.",
        "",
        "Child fits, standalone walking/survival/cost, child reconstruction/held error and paired teacher–student gaps were not measured. No second round or automatic follow-up is permitted.",
    ]
    with JSON_OUT.open("x") as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.write("\n")
    TABLE_OUT.write_text("\n".join(table) + "\n")
    print(
        json.dumps(
            dict(
                json=str(JSON_OUT),
                tables=str(TABLE_OUT),
                status=state["status"],
                teacher_totals=state["teacher_totals"],
                expected_audit_rows=audit_rows,
                expected_audit_calls=audit_calls,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
