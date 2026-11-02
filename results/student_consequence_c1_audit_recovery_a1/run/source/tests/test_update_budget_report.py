"""W2 saved report fixtures never evaluate a policy, optimizer or simulator."""

from __future__ import annotations

import json

import pytest

from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.reporting.update_budget import build_tables, paired_bootstrap, render


def synthetic_result():
    config = dict(schema="walker-student-update-w2-v1", bootstrap_seed=5, bootstrap_draws=20)
    result = dict(
        execution_complete=True,
        error=None,
        status="complete",
        points={},
        evaluations=dict(curve={}, primary={}, confirmation={}),
        gates=dict(
            confirmed_retention_benefit=True,
            confirmed_cost_benefit=False,
            confirmed_walking_gain=False,
            competence={},
            primary=dict(pairs=[]),
        ),
        teacher_outcomes=dict(train={"teacher": dict(cost=10, complete=True)}),
        work=dict(completed_optimizer_steps=30000, completed_checkpoint_loads=3),
        wall_seconds=2,
        stored_bytes=42,
    )
    for pair in range(3):
        result["gates"]["primary"]["pairs"].append(dict(pair=pair, relative_controls_gain=0.2))
        for update in (0, 2000, 10000):
            name = f"s{pair}_u{update}"
            result["points"][name] = dict(
                pair=pair,
                width=128,
                update=update,
                policy_seed=100 + pair,
                fit_seed=200 + pair,
                parameter_count=19590,
                metrics={
                    split: dict(
                        mse=0.8 if update == 2000 else 1.0,
                        per_trajectory={"teacher": dict(rows=2, mse=0.25, early=dict(mse=0.25))},
                    )
                    for split in ("train", "validation")
                },
            )
            for cohort, count, root in (("curve", 4, 1000), ("primary", 16, 2000)):
                if cohort == "primary" and update == 0:
                    continue
                cost = 99 + pair if update == 2000 else 100 + pair
                steps = 120 if update == 2000 else 100
                episodes = [
                    dict(
                        seed=root + index,
                        tag=f"{cohort}_{name}_{index}",
                        cost=cost,
                        steps=steps,
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
                        mean_steps=steps,
                        mean_velocity_mae=1.0,
                    ),
                    episodes=episodes,
                )
    return result, config


def test_every_stream_endpoint_and_curve_remains_visible():
    result, _ = synthetic_result()
    tables = build_tables(result)
    assert len(tables["fitting"]) == len(tables["trajectories"]) == 18
    assert len(tables["standalone"]) == 15 and len(tables["episodes"]) == 132
    assert len(tables["paired_endpoints"]) == 3
    assert {row["update"] for row in tables["standalone"] if row["cohort"] == "primary"} == {
        2000,
        10000,
    }
    assert all(row["width"] == 128 for row in tables["fitting"])


def test_crossed_bootstrap_retains_direction_and_both_paired_axes():
    result, config = synthetic_result()
    output = paired_bootstrap(result, config)
    assert output == paired_bootstrap(result, config)
    assert output["training_streams"] == 3 and output["candidate_updates"] == 2000
    assert output["primary"]["intervals"]["cost_2k_minus_10k"] == [-1, -1]
    assert output["primary"]["intervals"]["controls_2k_minus_10k"] == [20, 20]
    assert output["primary"]["intervals"]["walking_fraction_2k_minus_10k"] == [0, 0]
    assert output["confirmation"]["available"] is False
    assert output["fitting"]["train"]["mean_relative_mse_gain"] == pytest.approx(0.2)
    first = next(iter(result["evaluations"]["primary"].values()))
    first["episodes"][0]["seed"] = -1
    with pytest.raises(ValueError, match="paired reset order"):
        paired_bootstrap(result, config)


def test_saved_render_is_create_only_and_separates_retention_from_walking(tmp_path):
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
    assert "Confirmed retention benefit: True" in text
    assert "Retention without walking is partial" in text
    assert "not six independent fits" in text
    assert "there is no omnibus task-pass label" in text
    assert "same endpoint to walk on at least 15/16" in text
    accounting = json.loads((out / "cost_accounting.json").read_text())
    assert accounting["training_streams"] == 3 and accounting["planned_optimizer_steps"] == 30000
    assert "included once" in accounting["prefix_accounting"]
    with pytest.raises(FileExistsError):
        render(run, out)
    (run / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration binding"):
        render(run, tmp_path / "tampered")
