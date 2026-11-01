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


def _capsule(tmp_path, run, report):
    capsule = tmp_path / "capsule"
    (capsule / "run").mkdir(parents=True)
    (capsule / "report").mkdir()
    for name in ("config.json", "result.json", "input_manifest.json"):
        shutil.copy2(run / name, capsule / "run" / name)
    for name in ("learning_bins.csv", "learning_bins_manifest.json", "receipt.json"):
        shutil.copy2(report / name, capsule / "report" / name)
    paths = [path for folder in ("run", "report") for path in (capsule / folder).iterdir()]
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
