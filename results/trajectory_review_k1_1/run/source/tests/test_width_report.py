"""Saved synthetic W1 reports contain every paired seed and no real inference."""

from __future__ import annotations

import json

import pytest

from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.reporting.width import (
    build_tables,
    paired_bootstrap,
    render,
    teacher_student_gaps,
)


def synthetic_result():
    config = dict(
        schema="walker-student-width-w1-v1",
        updates=10000,
        batch_size=256,
        bootstrap_seed=5,
        bootstrap_draws=20,
    )
    result = dict(
        execution_complete=True,
        error=None,
        status="complete",
        points={},
        evaluations=dict(curve={}, primary={}, confirmation={}),
        gates={},
        teacher_outcomes=dict(train={"teacher": dict(cost=10, complete=True)}),
        work=dict(completed_optimizer_steps=60000),
        wall_seconds=2,
        stored_bytes=42,
    )
    for pair in range(3):
        for width in (128, 256):
            for update in (0, 2000, 5000, 10000):
                name = f"s{pair}_w{width}_u{update}"
                result["points"][name] = dict(
                    pair=pair,
                    width=width,
                    update=update,
                    policy_seed=100 + pair,
                    fit_seed=200 + pair,
                    parameter_count=19590 if width == 128 else 71942,
                    metrics={
                        split: dict(
                            mse=(1 if width == 128 else 0.8) / (update + 1),
                            per_trajectory={
                                "teacher": dict(rows=2, mse=0.25, early=dict(mse=0.25))
                            },
                        )
                        for split in ("train", "validation")
                    },
                )
                for cohort, count, root in (("curve", 4, 1000), ("primary", 16, 2000)):
                    if cohort == "primary" and update != 10000:
                        continue
                    cost = 100 + pair if width == 128 else 99 + pair
                    episodes = [
                        dict(
                            seed=root + index,
                            tag=f"{cohort}_{name}_{index}",
                            cost=cost,
                            steps=100,
                            mae=1.0,
                            success=False,
                            survived=False,
                            complete=True,
                            terminal_cost=20,
                            terminal_adjustment=0,
                            absorbing_cost=4500,
                        )
                        for index in range(count)
                    ]
                    result["evaluations"][cohort][name] = dict(
                        summary=dict(
                            episodes=count,
                            successes=0,
                            survived=0,
                            mean_cost=cost,
                            mean_steps=100,
                            mean_velocity_mae=1.0,
                        ),
                        episodes=episodes,
                    )
    return result, config


def test_all_points_and_pairs_are_reported_without_best_selection():
    result, _ = synthetic_result()
    tables = build_tables(result)
    assert len(tables["fitting"]) == len(tables["trajectories"]) == 48
    assert len(tables["standalone"]) == 30 and len(tables["episodes"]) == 192
    assert len(tables["paired_final"]) == 3
    assert {row["pair"] for row in tables["paired_final"]} == {0, 1, 2}
    assert all(row["primary_cost_256_minus_128"] == -1 for row in tables["paired_final"])
    gaps = teacher_student_gaps(result)
    assert len(gaps) == 6 and all(row["teacher_mean_cost"] == 10 for row in gaps)
    assert all("not paired" in row["scope"] for row in gaps)


def test_crossed_bootstrap_preserves_training_pair_and_reset_matching():
    result, config = synthetic_result()
    output = paired_bootstrap(result, config)
    assert output == paired_bootstrap(result, config)
    assert output["training_pairs"] == 3 and output["primary"]["reset_count"] == 16
    assert output["primary"]["intervals"]["cost_256_minus_128"] == [-1, -1]
    assert output["primary"]["intervals"]["walking_fraction_256_minus_128"] == [0, 0]
    assert output["confirmation"]["available"] is False
    assert output["fitting"]["train"]["mean_relative_mse_gain"] == pytest.approx(0.2)
    first = next(iter(result["evaluations"]["primary"].values()))
    first["episodes"][0]["seed"] = -1
    with pytest.raises(ValueError, match="paired reset order"):
        paired_bootstrap(result, config)


def test_saved_render_binds_inputs_outputs_figures_and_is_create_only(tmp_path):
    result, config = synthetic_result()
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(config))
    (run / "result.json").write_text(json.dumps(result))
    (run / "input_manifest.json").write_text(
        json.dumps(dict(config_sha256=sha(run / "config.json")))
    )
    out = tmp_path / "report"
    receipt = render(run, out)
    assert receipt["new_controls"] == receipt["new_forward_rows"] == receipt["optimizer_steps"] == 0
    for name in ("fitting_curves", "standalone_curves", "primary_outcomes"):
        for suffix in ("png", "pdf"):
            assert (out / f"{name}.{suffix}").stat().st_size > 0
    assert all(sha(out / name) == value["sha256"] for name, value in receipt["outputs"].items())
    text = (out / "report.md").read_text()
    assert "No new teacher acquisition" in text
    assert "do not match computation" in text
    assert "does not isolate" not in text  # The stated limit is precise, not a substitute verdict.
    assert "not isolate representational capacity" in text
    assert "not a matched-reset causal estimate" in text
    assert (out / "descriptive_bootstrap.json").is_file()
    with pytest.raises(FileExistsError):
        render(run, out)
    (run / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration binding"):
        render(run, tmp_path / "tampered")
