"""Independent saved-array energy and controller-binding fixtures; no physics."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.reporting import acrobot_energy_probe as report


def states_at_pose(n=600, q=(0.0, 0.0), velocity=(0.0, 0.0)):
    states = np.zeros((n + 1, 5))
    states[:, 0] = np.arange(n + 1) * 0.02
    states[:, 1:3] = q
    states[:, 3:] = velocity
    return states


def test_independent_energy_reference_and_signed_error():
    states = np.array([[0, 0, 0, 0, 0], [0, np.pi, 0, 0, 0], [0, np.pi, np.pi, 0, 0]])
    values = report.independent_components(states, np.zeros((3, 1)), kind="energy", weight=160 / 9)
    np.testing.assert_allclose(values["potential"], [58.86, 19.62, 29.43], atol=1e-13)
    np.testing.assert_array_equal(values["kinetic"], [0, 0, 0])
    assert values["energy_error"][1] == pytest.approx(-1)
    assert values["gate"][0] == 1 and values["weighted_energy"][0] == 0
    fast = states[1:].copy()
    fast[:, 3:] = [20, 30]
    above = report.independent_components(fast, np.zeros((2, 1)), kind="energy", weight=160 / 9)
    assert (above["energy_error"] > 0).all()
    assert (above["weighted_energy"] > 0).all()  # Overshoot is not silently clipped away.


def test_energy_matched_spin_is_not_height_success():
    velocity = np.sqrt(2 * (58.86 - 29.43) / (0.09530365719649561 + 0.25))
    states = states_at_pose(q=(np.pi, np.pi), velocity=(0, velocity))
    actions = np.zeros((600, 1))
    comp = report.independent_components(states[1:], actions, kind="energy", weight=160 / 9)
    row, _ = report.episode_measurements(
        states, actions, comp["running"], controls=600, kind="energy", weight=160 / 9
    )
    assert row["terminal_total_energy"] == pytest.approx(58.86)
    assert row["maximum_height"] == pytest.approx(2)
    assert not row["success"] and row["target_energy_low_height_fraction"] == 1
    assert row["terminal_kinetic_energy"] > 29


def test_height_success_does_not_add_velocity_criterion_or_double_terminal():
    states = states_at_pose(velocity=(3, 4))
    actions = np.ones((600, 1)) * 0.5
    comp = report.independent_components(states[1:], actions, kind="energy", weight=160 / 9)
    row, _ = report.episode_measurements(
        states, actions, comp["running"], controls=600, kind="energy", weight=160 / 9
    )
    assert row["success"]
    assert row["terminal_velocity_norm"] == 5
    assert row["episode_terminal_cost"] == pytest.approx(52)
    assert row["task_cost"] == pytest.approx(600 * 0.312 + 52)
    assert row["first_tip_above_threshold_seconds"] == 0.02
    assert row["longest_height_hold_seconds"] == 12


def test_final_window_pre_action_and_complete_trace_contracts():
    states = states_at_pose(500)
    states[1:401, 1] = np.pi
    actions = np.zeros((500, 1))
    comp = report.independent_components(states[1:], actions, kind="dense")
    row, _ = report.episode_measurements(
        states,
        actions,
        comp["running"],
        controls=500,
        kind="dense",
        observations=states[:-1, 1:],
    )
    assert row["success"] and row["time_above_height_seconds"] == 2
    short, _ = report.episode_measurements(
        states[:-1], actions[:-1], comp["running"][:-1], controls=500, kind="dense"
    )
    assert not short["success"] and not short["complete"]
    incorrect = states[:-1, 1:].copy()
    incorrect[0, 0] += 0.1
    with pytest.raises(ValueError, match="PRE observation"):
        report.episode_measurements(
            states, actions, comp["running"], controls=500, kind="dense", observations=incorrect
        )
    with pytest.raises(ValueError, match=r"N\+1"):
        report.episode_measurements(
            states[1:], actions, comp["running"], controls=500, kind="dense"
        )


def test_declared_numeric_bound_is_not_changed_after_error():
    audit = report.SavedAudit(Path("."), 10)
    expected = np.array([1e6])
    allowed = report.REPLAY_ATOL + report.ROUNDING_FACTOR * expected
    audit.same(expected + allowed * 0.5, expected, "Cost sum")
    with pytest.raises(ValueError, match="numeric"):
        audit.same(expected + allowed * 2, expected, "Bad cost sum")
    with pytest.raises(ValueError, match="numeric"):
        audit.same(expected + 1e-9, expected, "Replay keeps strict tolerance", replay=True)


def roster():
    config = dict(
        parts={
            "near-upright": dict(
                starts=[dict(reset_seed=i, planner_seed=i + 100) for i in range(4)]
            ),
            "energy": dict(
                starts=[dict(reset_seed=i, planner_seed=i + 100) for i in (10, 11)],
                variants=[dict(name=name) for name in ("a", "b", "c")],
            ),
        }
    )
    rows = [
        dict(
            part="near-upright",
            variant="old",
            reset_seed=i,
            planner_seed=i + 100,
            complete=True,
            success=False,
        )
        for i in range(4)
    ]
    rows += [
        dict(
            part="energy",
            variant=name,
            reset_seed=i,
            planner_seed=i + 100,
            complete=True,
            success=False,
        )
        for name in ("a", "b", "c")
        for i in (10, 11)
    ]
    return config, rows


def test_kill_rule_requires_two_in_one_variant_not_pooled_successes():
    config, rows = roster()
    rows[4]["success"] = rows[6]["success"] = True
    value = report.decision(rows, config)
    assert value["scientific_decision"] == "TEACHER_PROBLEM_STOP"
    assert value["selected_variant"] is None
    rows[7]["success"] = True
    value = report.decision(rows, config)
    assert value["selected_variant"] == "b" and value["promising_variants"] == ["b"]
    assert not value["teacher_nine_of_ten_qualified"] and not value["loop_tested"]
    rows[5]["success"] = True
    assert (
        report.decision(rows, config)["selected_variant"] == "a"
    )  # Declared order, not later cost.


@pytest.mark.parametrize("change", ["missing", "duplicate", "partial"])
def test_incomplete_roster_never_gets_certified(change):
    config, rows = roster()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = copy.deepcopy(rows[-2])
    else:
        rows[-1]["complete"] = False
    with pytest.raises(ValueError, match="roster|Incomplete"):
        report.decision(rows, config)


def synthetic_bank(tmp_path):
    import h5py

    cfg = dict(K=3, H=100, noise_sigma=0.3, lam=3.0, initial_iterations=3)
    variant = dict(name="fixture", kind="dense", planner=cfg)
    seed = 123
    generator = np.random.default_rng(seed)
    before = np.zeros((cfg["H"], 1))
    noise = np.einsum(
        "khi,ji->khj",
        generator.standard_normal((3, 100, 1)),
        np.linalg.cholesky(np.array([[0.3**2]])),
    )
    actions = np.clip(before[None] + noise, -1, 1)
    states = np.zeros((3, 100, 5))
    states[..., 0] = np.arange(1, 101)[None] * 0.02
    comp = report.independent_components(states, actions, kind="dense")
    scores = comp["running"].sum(1)
    weights = np.exp(-(scores - scores.min()) / 3)
    weights /= weights.sum()
    after = before + np.einsum("k,kha->ha", weights, actions - before[None])
    handle = h5py.File(tmp_path / "bank.h5", "w")
    group = handle.create_group("pass")
    for name, value in dict(
        candidate_costs=scores, weights=weights, nominal_before=before, nominal_after=after
    ).items():
        group.create_dataset(name, data=value)
    group.attrs.update(
        lam=3.0,
        neff=1 / np.sum(weights**2),
        rng_state_after_json=json.dumps(generator.bit_generator.state),
    )
    diag = group.create_group("components")
    indices = np.array([np.argmin(scores), 0])
    arrays = dict(
        running_sums=np.stack([comp[k].sum(1) for k in report.RUNNING_COLUMNS], 1),
        energy_sums=np.stack([comp[k].sum(1) for k in report.ENERGY_COLUMNS], 1),
        terminal=np.zeros(3),
        peak_height=np.full(3, 4.0),
        final_window_minimum_height=np.full(3, 4.0),
        height_qualified=np.ones(3, bool),
        selected_indices=indices,
        selected_states=states[indices],
        selected_actions=actions[indices],
    )
    for name, value in arrays.items():
        diag.create_dataset(name, data=value)
    diag.attrs["candidate_actions_sha256"] = report.array_sha(actions)
    return handle, group, variant, seed


def test_streamed_white_update_and_retained_cost_bindings(tmp_path):
    handle, group, variant, seed = synthetic_bank(tmp_path)
    try:
        audit = report.SavedAudit(tmp_path, 10)
        rows = []
        result = report._verify_bank(
            audit,
            group,
            variant,
            np.random.default_rng(seed),
            None,
            np.zeros(5),
            rows,
            dict(case="synthetic"),
        )
        np.testing.assert_array_equal(result, group["nominal_after"][:])
        assert rows[0]["candidate_action_hash_matches_regeneration"]
        assert audit.regeneration_cpu_seconds >= 0
        group["nominal_after"][0, 0] += 0.1
        with pytest.raises(ValueError, match="weighted update"):
            report._verify_bank(
                report.SavedAudit(tmp_path, 10),
                group,
                variant,
                np.random.default_rng(seed),
                None,
                np.zeros(5),
                [],
                {},
            )
    finally:
        handle.close()


def test_binary_regeneration_hash_is_diagnostic_not_fabricated_identity(tmp_path):
    handle, group, variant, seed = synthetic_bank(tmp_path)
    try:
        group["components"].attrs["candidate_actions_sha256"] = "0" * 64
        rows = []
        report._verify_bank(
            report.SavedAudit(tmp_path, 10),
            group,
            variant,
            np.random.default_rng(seed),
            None,
            np.zeros(5),
            rows,
            {},
        )
        assert rows[0]["candidate_action_hash_matches_regeneration"] is False
    finally:
        handle.close()


def test_missing_evidence_preserves_a_failed_saved_review(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(protocol="acrobot-energy-discovery-probe", version="1.1")))
    out = tmp_path / "failed"
    value = report.render(tmp_path / "missing", config, out)
    assert value["status"] == "FAILED_OR_INCOMPLETE"
    assert (out / "independent_audit.json").is_file()
    assert value["new_physics_controls"] == value["new_network_forwards"] == 0


def test_exhausted_cpu_and_public_cli_are_native_free(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text("{}")
    value = report.render(tmp_path, config, tmp_path / "exhausted", cpu_seconds=0)
    assert value["status"] == "FAILED_OR_INCOMPLETE"
    assert "CPU budget" in value["error"]
    with pytest.raises(SystemExit) as exit_info:
        report.main(["--help"])
    assert exit_info.value.code == 0 and "--base" in capsys.readouterr().out
    code = (
        "import sys; from src.gps.mppi_gps.reporting import acrobot_energy_probe; "
        "assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
