"""S1 reporting fixtures contain no fitting, policy inference or simulator execution."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.reporting import source
from src.gps.mppi_gps.reporting.source import (
    build_tables,
    episode_components,
    matched_reference_gaps,
    paired_bootstrap,
    render,
)


def _metric(value):
    return dict(
        mse=value,
        per_trajectory={
            "teacher_0": dict(
                rows=1000,
                mse=value,
                early=dict(mse=value),
                per_actuator_mse=[value] * 6,
                bins=[
                    dict(
                        start_control=start,
                        stop_control_exclusive=start + 100,
                        rows=100,
                        mse=value,
                        per_actuator_mse=[value] * 6,
                    )
                    for start in range(0, 1000, 100)
                ],
                predictions=dict(mean_abs=0.3, saturation_fraction=0.0),
            ),
        },
    )


def _episode(name, panel, seed, index):
    success = name == "reference" or (name.startswith("sac_reference") and index < 2)
    steps = (
        1000
        if success
        else 120
        if name.startswith("sac_reference")
        else 80
        if name.startswith("mppi")
        else 50
    )
    terminal = 0.0 if success else 20.0
    absorbing = 5.0 * (1000 - steps)
    return dict(
        tag=f"{panel}_{name}_{seed}",
        seed=seed,
        steps=steps,
        success=success,
        survived=success,
        mae=0.1 if success else 1.0,
        cost=0.5 * steps + terminal + absorbing,
        complete=True,
        terminal_cost=terminal,
        terminal_adjustment=0.0,
        absorbing_cost=absorbing,
    )


def _fixture(tmp_path):
    run = tmp_path / "run"
    (run / "learning").mkdir(parents=True)
    order = [
        "reference",
        *(
            name
            for pair in range(3)
            for name in (f"incoming_s{pair}", f"mppi_s{pair}", f"sac_reference_s{pair}")
        ),
    ]
    config = dict(
        schema="walker-student-source-s1-v1",
        evaluation_order=order,
        primary_seeds=list(range(16)),
        confirmation_seeds=list(range(100, 116)),
        bootstrap=dict(seed=3, draws=20),
    )
    (run / "config.json").write_text(json.dumps(config))
    (run / "input_manifest.json").write_text(
        json.dumps(dict(config_sha256=sha(run / "config.json")))
    )
    points, fits = {}, {}
    for pair in range(3):
        for arm in ("incoming", "mppi", "sac_reference"):
            name = f"{arm}_s{pair}"
            points[name] = dict(
                pair=pair,
                arm=arm,
                update=0 if arm == "incoming" else 2000,
                metrics={
                    split: _metric(
                        0.5 if arm == "incoming" else 0.1 if split.startswith(arm + "_") else 0.6
                    )
                    for split in source.SPLITS
                },
            )
            if arm != "incoming":
                file = f"learning/{name}.json"
                curve = [
                    dict(
                        update=update,
                        fit_update=update,
                        action_mse=1 / (update + 1),
                        proximal_loss=0.0,
                        total_loss=1 / (update + 1),
                        unclipped_gradient_norm=0.3,
                    )
                    for update in range(1, 2001)
                ]
                (run / file).write_text(
                    json.dumps(dict(learning_curve=curve, completed_updates=2000, failed=False))
                )
                fits[name] = dict(point=name, pair=pair, arm=arm, file=file, sha256=sha(run / file))
    evaluations = {}
    for panel in ("primary", "confirmation"):
        evaluations[panel] = {}
        for name in order:
            episodes = [
                _episode(name, panel, seed, index)
                for index, seed in enumerate(config[panel + "_seeds"])
            ]
            evaluations[panel][name] = dict(
                episodes=episodes,
                summary=dict(
                    episodes=16,
                    successes=sum(row["success"] for row in episodes),
                    survived=sum(row["survived"] for row in episodes),
                ),
            )
    teachers = {
        f"{arm}_{split}_{index}": dict(
            id=f"{arm}_{split}_{index}",
            source=arm,
            split=split,
            rows=1000,
            seed=900 + index,
            outcome=dict(
                complete=True,
                success=True,
                survived=True,
                cost=400.0 if arm == "mppi" else 500.0,
                mae=0.1,
            ),
        )
        for arm in ("mppi", "sac_reference")
        for split, count in (("train", 4), ("validation", 2))
        for index in range(count)
    }
    gates = {
        panel: dict(
            pairs=[
                dict(
                    pair=pair,
                    walking_gain=2,
                    survival_gain=2,
                    relative_cost_gain=0.1,
                    controls_delta=100,
                )
                for pair in range(3)
            ],
            checks=dict(
                positive_pairs=True, walking_nonregression=True, survival_nonregression=True
            ),
            walking_screen=True,
        )
        for panel in ("primary", "confirmation")
    }
    gates.update(
        confirmed_walking_benefit=True, competence={name: name == "reference" for name in order}
    )
    result = dict(
        schema="walker-student-source-s1-result-v1",
        execution_complete=True,
        status="complete",
        error=None,
        points=points,
        fits=fits,
        evaluations=evaluations,
        gates=gates,
        teacher_outcomes=teachers,
        phase_work=dict(fit=dict(completed_optimizer_steps=12000), eval=dict(episodes=320)),
        work=dict(completed_optimizer_steps=12000, completed_backward_rows=3072000),
        wall_seconds=1.0,
    )
    (run / "result.json").write_text(json.dumps(result))
    return run, result, config


def test_all_splits_episodes_components_and_unpaired_source_gaps_are_retained(tmp_path):
    _, result, config = _fixture(tmp_path)
    tables = build_tables(result)
    assert len(tables["fitting"]) == 9 * 4
    assert len(tables["trajectories"]) == 9 * 4
    assert len(tables["actuator_errors"]) == 9 * 4 * 6
    assert len(tables["reconstruction_control_bins"]) == 9 * 4 * 10
    assert len(tables["episodes"]) == 320
    assert len(tables["standalone"]) == 20
    assert len(tables["source_teachers"]) == 12
    assert len(tables["teacher_student_gaps"]) == 36
    assert len(tables["paired_cost_components"]) == 6
    for row in tables["episodes"]:
        assert row["raw_cost"] + row["terminal_cost"] + row["terminal_adjustment"] + row[
            "absorbing_cost"
        ] == pytest.approx(row["cost"])
    assert episode_components(dict(cost=4.0))["raw_cost"] is None
    reference_child = next(
        row
        for row in tables["standalone"]
        if row["point"] == "sac_reference_s0" and row["panel"] == "primary"
    )
    assert reference_child["walking_count"] == reference_child["survival_count"] == 2
    assert result["gates"]["confirmed_walking_benefit"]
    assert not result["gates"]["competence"]["sac_reference_s0"]
    gap = next(row for row in tables["teacher_student_gaps"] if row["point"] == "sac_reference_s0")
    assert gap["teacher_mean_cost"] == 500.0
    assert gap["student_minus_teacher_cost"] == reference_child["mean_cost"] - 500.0
    assert "unpaired" in gap["scope"]
    for row in tables["paired_cost_components"]:
        assert row["raw_cost_delta"] + row["terminal_cost_delta"] + row[
            "terminal_adjustment_delta"
        ] + row["absorbing_cost_delta"] == pytest.approx(row["cost_delta"])
    bootstrap = paired_bootstrap(result, config)
    assert bootstrap["primary"]["available"] and bootstrap["confirmation"]["available"]
    json.dumps(tables, allow_nan=False)


def test_full_render_binds_fixed_learning_bins_and_separates_benefit_from_competence(tmp_path):
    run, _, _ = _fixture(tmp_path)
    out = tmp_path / "report"
    receipt = render(run, out)
    assert receipt["schema"] == "walker-student-source-s1-report-v1"
    assert (
        receipt["new_forward_rows"]
        == receipt["new_controls"]
        == receipt["backward_rows"]
        == receipt["optimizer_steps"]
        == 0
    )
    with (out / "learning_bins.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 120
    assert sum(int(row["updates"]) for row in rows) == 12000
    assert rows[0]["start_fit_update"] == "1" and rows[0]["stop_fit_update"] == "100"
    text = (out / "report.md").read_text()
    assert "Recorded native attempts: **320/320**" in text
    assert "Declared confirmed source walking benefit: **True**" in text
    assert "a walking gain need not be competence" in text
    assert "whole-corpus source comparison" in text
    for name in ("reconstruction", "learning_curves", "standalone_outcomes", "cost_components"):
        for suffix in ("png", "pdf"):
            assert (out / f"{name}.{suffix}").stat().st_size > 0
    assert all(sha(out / name) == binding["sha256"] for name, binding in receipt["outputs"].items())
    with pytest.raises(FileExistsError):
        render(run, out)


def test_partial_cuda_failure_retains_partial_loss_bin_without_native_imputation(tmp_path):
    run, result, config = _fixture(tmp_path)
    name = "mppi_s0"
    fit = result["fits"][name]
    evidence = json.loads((run / fit["file"]).read_text())
    evidence.update(
        learning_curve=evidence["learning_curve"][:53], completed_updates=53, failed=True
    )
    (run / fit["file"]).write_text(json.dumps(evidence))
    fit["sha256"] = sha(run / fit["file"])
    result.update(
        schema="walker-student-source-s1-fit-v1",
        execution_complete=False,
        status="stopped_gpu",
        error="Synthetic CUDA failure",
        points={"incoming_s0": result["points"]["incoming_s0"]},
        fits={name: fit},
        evaluations={},
        gates={},
    )
    (run / "result.json").unlink()
    (run / "fit_result.json").write_text(json.dumps(result))
    out = tmp_path / "partial"
    render(run, out)
    with (out / "learning_bins.csv").open() as stream:
        bins = list(csv.DictReader(stream))
    assert len(bins) == 1 and bins[0]["updates"] == "53"
    assert bins[0]["stop_fit_update"] == "53"
    assert build_tables(result)["episodes"] == []
    assert not paired_bootstrap(result, config)["primary"]["available"]
    text = (out / "report.md").read_text()
    assert "fit_result.json" in text and "Synthetic CUDA failure" in text
    assert "Recorded native attempts: **0/320**" in text
    assert "Declared confirmed source walking benefit: **unavailable**" in text


def _capsule(tmp_path, run, report, interruption_summary=None, ledger_claims=None):
    capsule = tmp_path / "capsule"
    (capsule / "run").mkdir(parents=True)
    (capsule / "report").mkdir()
    result_name = "fit_result.json" if interruption_summary is not None else "result.json"
    for name in ("config.json", result_name, "input_manifest.json"):
        shutil.copy2(run / name, capsule / "run" / name)
    if interruption_summary is not None:
        shutil.copy2(interruption_summary, capsule / "run/interruption_summary.json")
        shutil.copy2(interruption_summary.parent / "receipt.json", capsule / "run/receipt.json")
    if ledger_claims is not None:
        shutil.copy2(ledger_claims, capsule / "run/ledger_claims.json")
        shutil.copy2(
            ledger_claims.parent / "full_run_file_manifest.json",
            capsule / "run/full_run_file_manifest.json",
        )
    result_path = run / result_name
    result = json.loads(result_path.read_text())
    if result.get("schema") == source.EVALUATION_SCHEMA:
        inherited = Path(result["inherited_fit"]["file"])
        (capsule / "run" / inherited).parent.mkdir(parents=True)
        shutil.copy2(run / inherited, capsule / "run" / inherited)
    for name in ("learning_bins.csv", "learning_bins_manifest.json", "receipt.json"):
        shutil.copy2(report / name, capsule / "report" / name)
    paths = [
        path
        for folder in ("run", "report")
        for path in (capsule / folder).rglob("*")
        if path.is_file()
    ]
    (capsule / "capsule.json").write_text(
        json.dumps(
            dict(
                files={
                    str(path.relative_to(capsule)): dict(
                        sha256=sha(path), bytes=path.stat().st_size
                    )
                    for path in paths
                }
            )
        )
    )
    return capsule


def _interruption_fixture(tmp_path):
    run, original, config = _fixture(tmp_path)
    fit = {
        key: original[key] for key in ("points", "fits", "teacher_outcomes", "work", "wall_seconds")
    }
    fit.update(
        schema="walker-student-source-s1-fit-v1", execution_complete=True, error=None, samplers={}
    )
    (run / "fit_result.json").write_text(json.dumps(fit))
    (run / "result.json").unlink()
    confirmation = original["evaluations"]["confirmation"]
    evaluations = dict(
        primary=original["evaluations"]["primary"],
        confirmation={
            "reference": confirmation["reference"],
            "incoming_s0": confirmation["incoming_s0"],
            "mppi_s0": dict(episodes=confirmation["mppi_s0"]["episodes"][:3]),
        },
    )
    partial = dict(
        point="mppi_s0",
        panel="confirmation",
        tag="confirmation_mppi_s0_103",
        seed=103,
        verified_steps=22,
        observed_native_cost=42.0,
        complete=False,
        success=None,
        survived=None,
        complete_cost=None,
    )
    completed = {
        (panel, name, row["seed"])
        for panel, actors in evaluations.items()
        for name, record in actors.items()
        for row in record["episodes"]
    }
    missing = [
        dict(
            panel=panel,
            point=name,
            seed=seed,
            tag=f"{panel}_{name}_{seed}",
            status="verified_complete"
            if (panel, name, seed) in completed
            else "partial"
            if (panel, name, seed) == ("confirmation", "mppi_s0", 103)
            else "not_started",
        )
        for panel in ("primary", "confirmation")
        for name in config["evaluation_order"]
        for seed in config[panel + "_seeds"]
    ]
    summary = dict(
        fit,
        schema="walker-student-source-s1-interrupted-summary-v1",
        execution_complete=False,
        comparison_complete=False,
        status="interrupted_infrastructure",
        error="Synthetic host restart",
        audit_complete=True,
        integrity_status="valid_scoped_prefix",
        fit_result_sha256=sha(run / "fit_result.json"),
        config_sha256=sha(run / "config.json"),
        input_manifest_sha256=sha(run / "input_manifest.json"),
        audit_receipt=dict(
            file="receipt.json", schema="walker-student-source-s1-interruption-audit-v1"
        ),
        evaluations=evaluations,
        partial_attempts=[partial],
        missing_roster=missing,
        gates=dict(withheld=True, reason="The frozen comparison was interrupted"),
        work=dict(
            known_durable_prefix=dict(episodes=195, partial_controls=22),
            unknown_unrecorded_tail="Not quantified; not zero",
        ),
    )
    folder = tmp_path / "postmortem"
    folder.mkdir()
    path = folder / "interruption_summary.json"
    _write_interruption(path, summary)
    return run, path, summary


def _write_interruption(path, summary):
    path.write_text(json.dumps(summary))
    (path.parent / "receipt.json").write_text(
        json.dumps(
            dict(
                schema="walker-student-source-s1-interruption-audit-v1",
                passed=True,
                summary_sha256=sha(path),
            )
        )
    )


def _failed_audit_fixture(tmp_path):
    run, path, summary = _interruption_fixture(tmp_path)
    original_episodes = [
        dict(row, event="episode_complete")
        for actors in summary["evaluations"].values()
        for record in actors.values()
        for row in record["episodes"]
    ]
    actors = [
        dict(
            event="source_actor_evaluated",
            cohort=panel,
            point=name,
            summary=record.get("summary", {}),
        )
        for panel, values in summary["evaluations"].items()
        for name, record in values.items()
        if len(record["episodes"]) == 16
    ]
    unavailable = {
        "primary_mppi_s1_7",
        *(f"primary_sac_reference_s2_{seed}" for seed in range(11, 16)),
        "confirmation_reference_100",
        "confirmation_reference_101",
    }
    for values in summary["evaluations"].values():
        for record in values.values():
            record["episodes"] = [
                row for row in record["episodes"] if row["tag"] not in unavailable
            ]
    for row in summary["missing_roster"]:
        if row["tag"] in unavailable:
            row["status"] = "unverified_raw"
    summary["integrity_status"] = "incomplete_retained_evidence"
    _write_interruption(path, summary)
    audit = path.parent / "receipt.json"
    receipt = json.loads(audit.read_text())
    receipt.update(passed=False, error="Synthetic missing native-array groups")
    audit.write_text(json.dumps(receipt))
    ledger = run / "evidence/eval/ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in [*original_episodes, *actors])
    )
    binding = dict(
        file="evidence/eval/ledger.jsonl", sha256=sha(ledger), bytes=ledger.stat().st_size
    )
    catalog = path.parent / "full_run_file_manifest.json"
    catalog.write_text(json.dumps(dict(files=[binding])))
    claims = path.parent / "ledger_claims.json"
    claims.write_text(
        json.dumps(
            dict(
                schema="walker-s1-surviving-ledger-claims-v1",
                source_ledger=binding,
                original_file_manifest=dict(file=catalog.name, sha256=sha(catalog)),
                episode_complete=original_episodes,
                actor_summaries=actors,
            )
        )
    )
    return run, path, claims


def test_failed_audit_display_requires_flag_and_withholds_native_comparisons(tmp_path):
    run, summary, claims = _failed_audit_fixture(tmp_path)
    original_audit_hash = sha(summary.parent / "receipt.json")
    with pytest.raises(ValueError, match="display-failed-audit"):
        render(run, tmp_path / "rejected", interruption_summary=summary)
    out = tmp_path / "forensic"
    receipt = render(
        run, out, interruption_summary=summary, display_failed_audit=True, ledger_claims=claims
    )
    assert receipt["audit_passed"] is False and receipt["failed_audit_display"] is True
    assert receipt["verified_against_original_ledger"] is True
    assert sha(summary.parent / "receipt.json") == original_audit_hash
    with (out / "verified_native_subset.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 187
    with (out / "unverified_ledger_episode_claims.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 8
    with (out / "worker_actor_summary_claims.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 12
    for name in (
        "matched_reference_gaps",
        "teacher_student_gaps",
        "paired_cost_components",
        "cost_components",
    ):
        with (out / f"{name}.csv").open() as stream:
            assert list(csv.DictReader(stream)) == []
    with (out / "standalone.csv").open() as stream:
        assert "mean_cost" not in next(csv.DictReader(stream))
    text = (out / "report.md").read_text()
    assert "integrity audit remains FAILED" in text and "nonrepresentative" in text
    assert "No actor/source rate" in text and "unknown unrecorded tail" in text


def test_failed_forensic_capsule_keeps_explicit_flag_and_source_receipt_proof(tmp_path):
    run, summary, claims = _failed_audit_fixture(tmp_path)
    full = tmp_path / "full_forensic"
    render(run, full, interruption_summary=summary, display_failed_audit=True, ledger_claims=claims)
    capsule = _capsule(tmp_path, run, full, summary, claims)
    with pytest.raises(ValueError, match="display-failed-audit"):
        render(capsule, tmp_path / "silently_accepted")
    out = tmp_path / "portable_forensic"
    receipt = render(capsule, out, display_failed_audit=True)
    assert receipt["audit_passed"] is False
    assert receipt["verified_against_original_ledger"] is False
    assert receipt["source_report_verified_against_original_ledger"] is True
    assert not (capsule / "run/evidence").exists()
    for name in ("reconstruction", "learning_curves", "standalone_outcomes", "cost_components"):
        assert sha(full / f"{name}.png") == sha(out / f"{name}.png")


@pytest.mark.parametrize("field", ["cost", "success"])
def test_ledger_claim_outcome_tampering_cannot_reuse_an_unrelated_correct_source_hash(
    tmp_path, field
):
    run, summary, path = _failed_audit_fixture(tmp_path)
    claims = json.loads(path.read_text())
    row = claims["episode_complete"][0]
    row[field] = row[field] + 1 if field == "cost" else not row[field]
    path.write_text(json.dumps(claims))
    with pytest.raises(ValueError, match="claim rows differ"):
        render(
            run,
            tmp_path / "tampered_claim",
            interruption_summary=summary,
            display_failed_audit=True,
            ledger_claims=path,
        )
    assert not (tmp_path / "tampered_claim").exists()


def test_interrupted_run_keeps_partial_outcomes_unknown_and_original_outputs_absent(tmp_path):
    run, summary, _ = _interruption_fixture(tmp_path)
    fit_hash = sha(run / "fit_result.json")
    out = tmp_path / "interrupted_report"
    receipt = render(run, out, interruption_summary=summary)
    assert not (run / "result.json").exists() and not (run / "owner_exit.json").exists()
    assert sha(run / "fit_result.json") == fit_hash
    assert receipt["interruption_summary_sha256"] == sha(summary)
    assert receipt["fit_result_sha256"] == fit_hash
    with (out / "episodes.csv").open() as stream:
        episodes = list(csv.DictReader(stream))
    assert len(episodes) == 196
    partial = next(row for row in episodes if row["complete"] == "False")
    assert partial["verified_steps"] == "22" and partial["observed_native_cost"] == "42.0"
    assert partial["success"] == partial["survived"] == partial["cost"] == ""
    with (out / "standalone.csv").open() as stream:
        observed = next(
            row
            for row in csv.DictReader(stream)
            if row["panel"] == "confirmation" and row["point"] == "mppi_s0"
        )
    assert observed["completed_episodes"] == "3" and observed["recorded_attempts"] == "4"
    with (out / "missing_roster.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 320
    bootstrap = json.loads((out / "descriptive_bootstrap.json").read_text())
    assert not bootstrap["primary"]["available"] and not bootstrap["confirmation"]["available"]
    text = (out / "report.md").read_text()
    assert "Interrupted experiment" in text and "not a completed S1 comparison" in text
    assert "unknown unrecorded tail" in text
    assert "Declared confirmed source walking benefit: **unavailable**" in text


def test_unstarted_actor_with_zero_observed_episodes_has_no_imputed_failure_rate(tmp_path):
    _, _, summary = _interruption_fixture(tmp_path)
    summary["evaluations"]["confirmation"]["sac_reference_s0"] = dict(
        episodes=[],
        summary=dict(
            episodes=0,
            successes=0,
            survived=0,
            mean_cost=None,
            expected_episodes=16,
            cohort_complete=False,
        ),
    )
    row = next(
        row
        for row in build_tables(summary)["standalone"]
        if row["panel"] == "confirmation" and row["point"] == "sac_reference_s0"
    )
    assert row["recorded_attempts"] == row["completed_episodes"] == 0
    assert row["walking_count"] is row["walking_fraction"] is row["mean_cost"] is None
    assert row["declared_episodes"] == 0 and row["declared_expected_episodes"] == 16


def test_interrupted_capsule_preserves_summary_and_fit_ownership_without_raw(tmp_path):
    run, summary, _ = _interruption_fixture(tmp_path)
    full = tmp_path / "full_interrupted"
    render(run, full, interruption_summary=summary)
    capsule = _capsule(tmp_path, run, full, summary)
    assert not (capsule / "run/result.json").exists()
    assert not (capsule / "run/learning").exists()
    out = tmp_path / "portable_interrupted"
    receipt = render(capsule, out)
    assert receipt["render_scope"] == "portable_learning_bins"
    assert receipt["interruption_summary_sha256"] == sha(summary)
    for name in ("reconstruction", "learning_curves", "standalone_outcomes", "cost_components"):
        assert sha(full / f"{name}.png") == sha(out / f"{name}.png")
    assert (full / "episodes.csv").read_bytes() == (out / "episodes.csv").read_bytes()
    manifest = json.loads((out / "learning_bins_manifest.json").read_text())
    assert manifest["interruption_summary_sha256"] == sha(summary)
    assert manifest["fit_result_sha256"] == sha(run / "fit_result.json")


@pytest.mark.parametrize("fault", ["audit_pending", "fit_ownership"])
def test_interruption_summary_rejects_pending_audit_or_changed_fit_fields(tmp_path, fault):
    run, path, summary = _interruption_fixture(tmp_path)
    if fault == "audit_pending":
        summary["audit_complete"] = False
    else:
        summary["points"]["incoming_s0"]["metrics"]["mppi_train"]["mse"] += 1
    _write_interruption(path, summary)
    with pytest.raises(ValueError, match="scoped-prefix audit|fit ownership"):
        render(run, tmp_path / "rejected", interruption_summary=path)
    assert not (tmp_path / "rejected").exists()


def test_clean_capsule_renders_without_full_loss_curves_or_simulator_arrays(tmp_path, monkeypatch):
    run, _, _ = _fixture(tmp_path)
    full = tmp_path / "full"
    render(run, full)
    capsule = _capsule(tmp_path, run, full)
    assert not (capsule / "run/learning").exists()
    assert not list(capsule.rglob("*.h5"))
    monkeypatch.setattr(
        source,
        "learning_bins",
        lambda *args: pytest.fail("Portable render cannot read full curves"),
    )
    portable = tmp_path / "portable"
    receipt = render(capsule, portable)
    assert receipt["render_scope"] == "portable_learning_bins"
    for name in ("reconstruction", "learning_curves", "standalone_outcomes", "cost_components"):
        assert sha(full / f"{name}.png") == sha(portable / f"{name}.png")
    for name in ("fitting", "episodes", "learning_bins", "teacher_student_gaps", "cost_components"):
        assert (full / f"{name}.csv").read_bytes() == (portable / f"{name}.csv").read_bytes()
    text = (portable / "report.md").read_text()
    assert "Portable scope" in text and "Full per-update fit JSONs" in text
    with (capsule / "report/learning_bins.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="capsule file binding"):
        render(capsule, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_missing_cpu_attempt_is_not_counted_as_a_failed_episode_or_bootstrapped(tmp_path):
    _, result, config = _fixture(tmp_path)
    result["evaluations"]["primary"]["sac_reference_s0"]["episodes"].pop()
    row = next(
        row
        for row in build_tables(result)["standalone"]
        if row["point"] == "sac_reference_s0" and row["panel"] == "primary"
    )
    assert row["recorded_attempts"] == row["completed_episodes"] == 15
    assert row["walking_count"] == 2 and row["walking_fraction"] == pytest.approx(2 / 15)
    assert not paired_bootstrap(result, config)["primary"]["available"]


def test_fresh_reference_gaps_pair_actual_reset_rows_without_replicating_the_control(tmp_path):
    _, result, config = _fixture(tmp_path)
    gaps, status = matched_reference_gaps(result, config)
    assert len(gaps) == len(status) == 6
    assert all(row["available"] for row in status)
    row = next(row for row in gaps if row["panel"] == "primary" and row["pair"] == 0)
    assert row["reference_walking_count"] == row["reference_survival_count"] == 16
    assert row["child_walking_count"] == row["child_survival_count"] == 2
    assert row["walking_count_gap"] == row["survival_count_gap"] == -14
    assert row["mean_steps_gap"] == -770
    assert row["mean_raw_cost_gap"] + row["mean_terminal_cost_gap"] + row[
        "mean_terminal_adjustment_gap"
    ] + row["mean_absorbing_cost_gap"] == pytest.approx(row["mean_cost_gap"])
    assert row["shared_reference_control"] is True
    assert "not three independent teacher replications" in row["scope"]
    assert all(row["source"] == "sac_reference" for row in gaps)


@pytest.mark.parametrize("fault", ["missing", "misaligned", "incomplete"])
def test_fresh_reference_gap_never_imputes_missing_or_misaligned_episodes(tmp_path, fault):
    _, result, config = _fixture(tmp_path)
    episodes = result["evaluations"]["primary"]["sac_reference_s0"]["episodes"]
    if fault == "missing":
        episodes.pop()
    elif fault == "misaligned":
        episodes[0]["seed"] = 99999
    else:
        episodes[0]["complete"] = False
    gaps, status = matched_reference_gaps(result, config)
    assert len(gaps) == 5
    assert not any(row["panel"] == "primary" and row["pair"] == 0 for row in gaps)
    unavailable = next(row for row in status if row["panel"] == "primary" and row["pair"] == 0)
    assert unavailable["available"] is False and unavailable["reason"]
    assert "walking_count_gap" not in unavailable


def test_full_curve_tampering_and_cli_lazy_import_contract(tmp_path):
    run, result, _ = _fixture(tmp_path)
    fit = next(iter(result["fits"].values()))
    (run / fit["file"]).write_text("{}")
    with pytest.raises(ValueError, match="learning-curve file binding"):
        render(run, tmp_path / "tampered")
    root = Path(__file__).resolve().parents[1]
    code = "import sys; from src.gps.mppi_gps.reporting.source import main; assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules; main(['--help'])"
    process = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "--run" in process.stdout and "--out" in process.stdout


def _evaluation_fixture(tmp_path):
    """Derived metadata redirects unchanged closed curves; no model is involved."""
    run, result, config = _fixture(tmp_path)
    original = json.loads(json.dumps(result))
    for name, point in original["points"].items():
        point.update(
            parameter_count=19590,
            optimizer_fingerprint="adam-" + name,
            checkpoint=dict(
                file="checkpoints/" + name + ".pt",
                sha256="checkpoint-" + name,
                policy_hash="policy-" + name,
                bytes=12,
            ),
        )
    for name, fit in original["fits"].items():
        fit["optimizer_fingerprint"] = "adam-" + name
    original = {
        key: original[key] for key in ("points", "fits", "teacher_outcomes", "work", "wall_seconds")
    }
    original.update(
        schema="walker-student-source-s1-fit-v1",
        samplers={"paired": "exact original tape owner"},
        execution_complete=True,
        error=None,
    )
    inherited = run / "inputs/s1"
    inherited.mkdir(parents=True)
    (inherited / "fit_result.json").write_text(json.dumps(original))
    shutil.move(str(run / "learning"), inherited / "learning")
    result.update(
        schema=source.EVALUATION_SCHEMA,
        points=json.loads(json.dumps(original["points"])),
        fits=json.loads(json.dumps(original["fits"])),
        samplers=original["samplers"],
        inherited_fit=dict(
            file="inputs/s1/fit_result.json", sha256=sha(inherited / "fit_result.json")
        ),
        inherited_work=dict(
            original_fit=original["work"],
            original_eval_known_prefix=dict(completed_neural_rows=55607),
            original_eval_unknown_tail="Unquantified; not zero",
        ),
        work=dict(
            completed_neural_rows=75000,
            completed_live_controls=75000,
            completed_optimizer_steps=0,
            completed_backward_rows=0,
            checkpoint_loads=10,
        ),
    )
    result["phase_work"] = dict(eval=result["work"])
    for point in result["points"].values():
        point["checkpoint"]["file"] = "inputs/s1/" + point["checkpoint"]["file"]
    for fit in result["fits"].values():
        fit["file"] = "inputs/s1/" + fit["file"]
    for panel, first in (("primary", 1270000000), ("confirmation", 1280000000)):
        config[panel + "_seeds"] = list(range(first, first + 16))
        for name, actor in result["evaluations"][panel].items():
            actor["episodes"] = [
                _episode(name, panel, seed, index)
                for index, seed in enumerate(config[panel + "_seeds"])
            ]
    config["schema"] = "walker-student-source-s1.1-v1"
    config["source_bindings"] = dict(s1_fit_result_sha256=result["inherited_fit"]["sha256"])
    (run / "config.json").write_text(json.dumps(config))
    (run / "input_manifest.json").write_text(
        json.dumps(dict(config_sha256=sha(run / "config.json")))
    )
    (run / "result.json").write_text(json.dumps(result))
    return run, result, config


def test_evaluation_only_report_reuses_fits_and_separates_original_and_new_work(tmp_path):
    run, result, config = _evaluation_fixture(tmp_path)
    out = tmp_path / "evaluation_report"
    receipt = render(run, out)
    assert receipt["schema"] == source.EVALUATION_REPORT_SCHEMA
    assert receipt["study"] == "S1.1"
    assert receipt["inherited_fit"] == result["inherited_fit"]
    assert receipt["new_forward_rows"] == receipt["optimizer_steps"] == 0
    accounting = json.loads((out / "cost_accounting.json").read_text())
    assert accounting["recorded_work"]["completed_optimizer_steps"] == 0
    assert accounting["inherited_work"]["original_fit"]["completed_optimizer_steps"] == 12000
    assert accounting["inherited_work"]["original_eval_unknown_tail"] == "Unquantified; not zero"
    with (out / "work_groups.csv").open() as stream:
        groups = {row["group"]: row for row in csv.DictReader(stream)}
    assert set(groups) == {
        "original_fit",
        "original_eval_known_prefix",
        "original_eval_unknown_tail",
        "s1_1_new_native",
    }
    assert groups["s1_1_new_native"]["completed_neural_rows"] == "75000"
    assert groups["original_eval_known_prefix"]["completed_neural_rows"] == "55607"
    with (out / "episodes.csv").open() as stream:
        episodes = list(csv.DictReader(stream))
    assert len(episodes) == 320
    assert {int(row["seed"]) for row in episodes} == set(
        config["primary_seeds"] + config["confirmation_seeds"]
    )
    bootstrap = json.loads((out / "descriptive_bootstrap.json").read_text())
    assert all(bootstrap[panel]["training_streams"] == 3 for panel in ("primary", "confirmation"))
    assert "zero new training replications" in bootstrap["training_stream_origin"]
    assert "original interrupted outcomes are not pooled" in bootstrap["native_scope"]
    assert len(build_tables(result, config)["matched_reference_gaps"]) == 6
    text = (out / "report.md").read_text()
    assert "# S1.1:" in text and "zero fitting, reconstruction inference" in text
    assert (
        "same three training realizations" in text
        and "repeated testing is not a new learning gain" in text
    )
    assert "original interrupted native attempts are not pooled" in text


@pytest.mark.parametrize(
    "fault", ["metric", "checkpoint_owner", "fit_owner", "closed_fit_hash", "recipe_binding"]
)
def test_evaluation_report_rejects_changed_inherited_evidence(tmp_path, fault):
    run, result, _ = _evaluation_fixture(tmp_path)
    if fault == "metric":
        result["points"]["mppi_s0"]["metrics"]["mppi_train"]["mse"] += 1
    elif fault == "checkpoint_owner":
        result["points"]["mppi_s0"]["checkpoint"]["policy_hash"] = "different owner"
    elif fault == "fit_owner":
        result["fits"]["mppi_s0"]["optimizer_fingerprint"] = "different Adam owner"
    elif fault == "closed_fit_hash":
        (run / "inputs/s1/fit_result.json").write_text("{}")
    else:
        binding = run / result["inherited_fit"]["file"]
        closed_fit = json.loads(binding.read_text())
        closed_fit["wall_seconds"] += 1
        binding.write_text(json.dumps(closed_fit))
        result["inherited_fit"]["sha256"] = sha(binding)
    (run / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="S1.1 .* (evidence|owner|ownership|binding)"):
        render(run, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_evaluation_capsule_uses_inherited_bins_without_original_full_curves(tmp_path, monkeypatch):
    run, _, _ = _evaluation_fixture(tmp_path)
    full = tmp_path / "full_evaluation"
    render(run, full)
    capsule = _capsule(tmp_path, run, full)
    assert (capsule / "run/inputs/s1/fit_result.json").is_file()
    assert not (capsule / "run/inputs/s1/learning").exists()
    assert not list(capsule.rglob("*.h5"))
    monkeypatch.setattr(
        source,
        "learning_bins",
        lambda *args: pytest.fail("S1.1 capsule may only use existing loss bins"),
    )
    out = tmp_path / "portable_evaluation"
    receipt = render(capsule, out)
    assert receipt["schema"] == source.EVALUATION_REPORT_SCHEMA
    assert receipt["render_scope"] == "portable_learning_bins"
    for name in ("reconstruction", "learning_curves", "standalone_outcomes", "cost_components"):
        assert sha(full / f"{name}.png") == sha(out / f"{name}.png")
    for name in ("learning_bins", "fitting", "episodes", "work_groups", "matched_reference_gaps"):
        assert (full / f"{name}.csv").read_bytes() == (out / f"{name}.csv").read_bytes()
    (capsule / "run/inputs/s1/fit_result.json").write_text("{}")
    with pytest.raises(ValueError, match="inherited closed-fit file binding"):
        render(capsule, tmp_path / "tampered")


def test_evaluation_report_wrapper_has_lazy_help():
    root = Path(__file__).resolve().parents[1]
    code = """import runpy, sys
sys.argv = ['report_student_source_s1_1.py', '--help']
try:
    runpy.run_path('scripts/diagnostics/report_student_source_s1_1.py', run_name='__main__')
except SystemExit as exit_status:
    assert exit_status.code == 0
assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
"""
    process = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "--run" in process.stdout and "--out" in process.stdout
