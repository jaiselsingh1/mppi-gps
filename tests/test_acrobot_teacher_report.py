"""Synthetic independent teacher-accounting and geometry fixtures; no physics."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.reporting import acrobot_teacher as report


def test_report_accepts_public_dispatcher_arguments(capsys):
    with pytest.raises(SystemExit) as exit_info:
        report.main(["--help"])
    assert exit_info.value.code == 0
    assert "--cpu-seconds" in capsys.readouterr().out


def trace(n=500, *, upright=True):
    states = np.zeros((n + 1, 5))
    states[:, 0] = np.arange(n + 1) * 0.02
    if not upright:
        states[:, 1] = np.pi
    actions = np.zeros((n, 1))
    costs = np.zeros(n) if upright else np.full(n, 16.0)
    return states, actions, costs


def test_cost_geometry_and_two_second_success_are_independent():
    value = report.episode_metrics(*trace())
    assert value["success"] and value["complete"]
    assert value["height_seconds"] == value["longest_height_seconds"] == 10
    assert value["first_height_crossing_seconds"] == 0.02
    assert value["task_cost"] == value["final_velocity_norm"] == 0
    failed = report.episode_metrics(*trace(upright=False))
    assert not failed["success"] and failed["complete"]
    assert failed["task_cost"] == 8000 and failed["final_height"] == 0
    assert failed["first_height_crossing_seconds"] is None


def test_incomplete_prefix_cannot_pass_even_with_good_final_two_seconds():
    value = report.episode_metrics(*trace(499))
    assert not value["success"] and not value["complete"]
    assert value["missing_controls"] == 1


def test_success_uses_final_window_not_peak_or_inherited_initial_state():
    states, actions, _ = trace()
    states[1:401, 1] = np.pi
    tip = report.tip_geometry(states[1:])
    costs = np.sum((tip - [0, 0, 4]) ** 2, axis=1)
    value = report.episode_metrics(states, actions, costs)
    assert value["success"] and value["height_seconds"] == 2
    assert value["first_height_crossing_seconds"] == pytest.approx(8.02)
    states[401, 1] = np.pi
    costs[400] = 16
    assert not report.episode_metrics(states, actions, costs)["success"]


def test_height_success_keeps_velocity_separate_and_recomputes_every_cost():
    states, actions, costs = trace()
    states[:, 3:] = [3, 4]
    actions[:] = 0.5
    costs[:] = 0.2525
    value = report.episode_metrics(states, actions, costs)
    assert value["success"] and value["final_velocity_norm"] == 5
    assert value["velocity_cost"] == 125
    assert value["action_cost"] == pytest.approx(1.25)
    costs[50] += 1e-8
    with pytest.raises(ValueError, match="POST-action cost"):
        report.episode_metrics(states, actions, costs)


def test_pre_action_binding_and_n_plus_one_contract_are_enforced():
    states, actions, costs = trace()
    report.episode_metrics(states, actions, costs, observations=states[:-1, 1:])
    observations = states[1:, 1:].copy()
    observations[0, 1] = 0.1
    with pytest.raises(ValueError, match="PRE-action"):
        report.episode_metrics(states, actions, costs, observations=observations)
    with pytest.raises(ValueError, match="N\\+1"):
        report.episode_metrics(states[1:], actions, costs)
    states[-1, 0] += 0.001
    with pytest.raises(ValueError, match="time boundaries"):
        report.episode_metrics(states, actions, costs)


def config_and_rows():
    config = dict(
        recipes=[dict(name="a"), dict(name="b")],
        development=[dict(reset_seed=10, planner_seed=20), dict(reset_seed=11, planner_seed=21)],
        qualification=[dict(reset_seed=100 + i, planner_seed=200 + i) for i in range(10)],
        qualification_episode_count=10,
        qualification_successes_required=9,
    )
    rows = [
        dict(
            phase="development",
            recipe=recipe,
            complete=True,
            success=True,
            task_cost=cost,
            **seeds,
        )
        for recipe, cost in (("a", 10), ("b", 9))
        for seeds in config["development"]
    ]
    rows += [
        dict(
            phase="qualification",
            recipe="b",
            complete=True,
            success=i < 9,
            task_cost=9,
            **seeds,
        )
        for i, seeds in enumerate(config["qualification"])
    ]
    return config, rows


def test_selection_uses_development_only_and_disjoint_nine_of_ten_gate():
    config, rows = config_and_rows()
    value = report.selection_from_rows(rows, config)
    assert value["selected_recipe"] == "b"
    assert value["qualification_successes"] == 9 and value["gate_passed"]
    rows[-2]["success"] = False
    assert not report.selection_from_rows(rows, config)["gate_passed"]


def test_selection_rejects_missing_duplicates_wrong_recipe_and_seed_leakage():
    config, rows = config_and_rows()
    with pytest.raises(ValueError, match="Qualification roster"):
        report.selection_from_rows(rows[:-1], config)
    with pytest.raises(ValueError, match="Qualification roster"):
        report.selection_from_rows(rows + [rows[-1]], config)
    rows[-1]["recipe"] = "a"
    with pytest.raises(ValueError, match="Wrong qualified recipe"):
        report.selection_from_rows(rows, config)


def test_fixed_selection_does_not_hide_failed_development_episodes():
    config, rows = config_and_rows()
    for row in rows:
        if row["phase"] == "development":
            row["success"] = False
    value = report.selection_from_rows(rows, config)
    assert value["selected_recipe"] == "b"
    assert value["ranking"][0]["successes"] == 0


def synthetic_run(tmp_path):
    """A tiny complete saved lifecycle; generated arrays are not native evidence."""
    root = tmp_path / "run"
    root.mkdir()
    config, _ = config_and_rows()
    config["recipes"].append(dict(name="c"))
    for recipe in config["recipes"]:
        recipe["planner"] = dict(H=2, K=2, lam=3, initial_iterations=2)
    config.update(
        task=dict(frame_skip=2, controls=3, balance_seconds=0.04, balance_height=3.8),
        preflight=dict(controls=2),
        expected_environment=dict(python="synthetic"),
        threads=2,
        math_threads=1,
        bindings=[],
        limits=dict(
            native_controls=100,
            forecast_controls=1000,
            physics_substeps=10000,
            cpu_seconds=100,
            wall_seconds=100,
            rss_bytes=1000000,
            output_bytes=10000000,
        ),
    )
    config_path = tmp_path / "config.json"
    report._write_json(config_path, config)
    (root / "config.json").write_bytes(config_path.read_bytes())
    report._write_json(root / "inputs.json", dict(config_sha256=report._sha(config_path), files=[]))
    report._write_json(
        root / "environment.json",
        dict(
            versions=config["expected_environment"],
            physics_threads=2,
            math_threads=dict(OMP_NUM_THREADS="1"),
        ),
    )
    receipts = []
    with h5py.File(root / "evidence.h5", "x") as raw:
        raw.attrs["status"] = "QUALIFIED"
        preflight = raw.create_group("preflight")
        preflight.attrs["status"] = "PASSED"
        states, actions, costs = trace(2)
        for label in ("recorded", "restored"):
            group = preflight.create_group(label)
            for key, value in dict(
                states=states,
                costs=costs,
                warmstarts=np.zeros((3, 2)),
                integration=np.zeros((3, 7)),
            ).items():
                group.create_dataset(key, data=value)
        preflight.create_dataset("batch_states", data=states[1:])
        preflight.create_dataset("batch_cost", data=[costs.sum()])
        preflight.create_dataset("actions", data=actions)
        for phase in ("development", "qualification"):
            for recipe_index, recipe in enumerate(config["recipes"]):
                if phase == "qualification" and recipe_index != 0:
                    continue
                for trial, seeds in enumerate(config[phase]):
                    name = (
                        f"development/recipe_{recipe_index}/trial_{trial}"
                        if phase == "development"
                        else f"qualification/trial_{trial}"
                    )
                    group = raw.create_group(name)
                    states, actions, costs = trace(3)
                    for key, value in dict(
                        states=states,
                        actions=actions,
                        costs=costs,
                        warmstarts=np.zeros((4, 2)),
                        integration=np.zeros((4, 7)),
                        pre_action_observations=states[:-1, 1:],
                    ).items():
                        group.create_dataset(key, data=value)
                    group.attrs.update(
                        status="COMPLETED",
                        stage=phase,
                        recipe=recipe["name"],
                        **seeds,
                        completed_controls=3,
                        success=True,
                        task_cost=0,
                    )
                    plan = group.create_group("planning")
                    for t in range(3):
                        bank = plan.create_group(f"control_{t:04d}")
                        bank.create_dataset("pre_action_observation", data=states[t, 1:])
                        for p in range(2 if t == 0 else 1):
                            found = bank.create_group(f"pass_{p:03d}")
                            scores = np.array([0, 1])
                            weights = np.exp(-(scores - scores.min()) / 3)
                            weights /= weights.sum()
                            for key, value in dict(
                                candidate_costs=scores,
                                weights=weights,
                                nominal_before=np.zeros((2, 1)),
                                nominal_after=np.zeros((2, 1)),
                            ).items():
                                found.create_dataset(key, data=value)
                            found.attrs.update(lam=3, neff=1 / np.sum(weights**2))
                    metrics = report.episode_metrics(
                        states, actions, costs, horizon=3, hold_seconds=0.04
                    )
                    receipts.append(
                        dict(
                            stage=phase,
                            recipe_index=recipe_index,
                            trial_index=trial,
                            recipe=recipe["name"],
                            **seeds,
                            success=True,
                            controls=3,
                            task_cost=0,
                            first_tip_above_threshold_seconds=0.02,
                            time_balanced_seconds=0.06,
                            **{
                                key: metrics[key]
                                for key in (
                                    "first_complete_hold_start_seconds",
                                    "final_window_minimum_height",
                                    "final_window_velocity_rms",
                                    "terminal_angles_raw",
                                    "terminal_winding",
                                )
                            },
                        )
                    )
    (root / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in receipts))
    report._write_json(root / "selection.json", dict(selected_recipe_index=0))
    work = {}
    for field, count in dict(
        native_controls=48,
        verification_controls=6,
        forecast_controls=256,
        physics_substeps=620,
        model_queries=64,
        restores=1,
    ).items():
        for suffix in ("attempted", "completed"):
            work[field + "_" + suffix] = count
    work.update(network_forwards=0, backward_calls=0, optimizer_updates=0)
    evidence = {p.name: dict(sha256=report._sha(p), bytes=p.stat().st_size) for p in root.iterdir()}
    result = dict(
        status="QUALIFIED",
        selected_recipe_index=0,
        selected_recipe=config["recipes"][0],
        development=[row for row in receipts if row["stage"] == "development"],
        qualification=[row for row in receipts if row["stage"] == "qualification"],
        qualification_successes=10,
        evidence=evidence,
        work=work,
        unknown_partial_work=dict(
            model_queries=0,
            native_controls=0,
            verification_controls=0,
            forecast_controls=0,
            physics_substeps=0,
        ),
        resources=dict(cpu_seconds=1, wall_seconds=1, rss_bytes=1, output_bytes=1),
    )
    report._write_json(root / "result.json", result)
    return root, config_path


def test_small_saved_lifecycle_checks_all_passes_without_execution(tmp_path, monkeypatch):
    raw, config = synthetic_run(tmp_path)
    monkeypatch.setattr(report, "_plot", lambda path, *args: path.write_bytes(b"synthetic"))
    value = report.render(raw, config, tmp_path / "report")
    assert value["status"] == "PASSED", value
    assert value["recomputed_work"]["model_queries"] == 64
    assert value["recomputed_work"]["physics_substeps"] == 620
    assert value["selection"]["qualification_successes"] == 10
    assert value["new_physics_controls"] == value["new_network_forwards"] == 0
    assert (tmp_path / "report" / "episodes.csv").exists()


def test_tampered_saved_receipt_is_retained_as_failed_review(tmp_path, monkeypatch):
    raw, config = synthetic_run(tmp_path)
    value = json.loads((raw / "result.json").read_text())
    value["work"]["forecast_controls_completed"] += 1
    report._write_json(raw / "result.json", value)
    monkeypatch.setattr(report, "_plot", lambda path, *args: None)
    result = report.render(raw, config, tmp_path / "failed_report")
    assert result["status"] == "FAILED_OR_INCOMPLETE"
    assert "forecast_controls_completed" in result["error"]
    assert (tmp_path / "failed_report" / "independent_audit.json").exists()


def test_report_budget_exhaustion_preserves_failure_without_partial_certification(tmp_path):
    raw, config = synthetic_run(tmp_path)
    value = report.render(raw, config, tmp_path / "budget_report", cpu_seconds=0)
    assert value["status"] == "FAILED_OR_INCOMPLETE"
    assert "CPU ceiling exhausted" in value["error"]
    assert value["new_physics_controls"] == 0
