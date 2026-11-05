"""Saved-only contracts for fixed-point, capture and native planner evidence."""

import importlib
import sys

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from src.gps.mppi_gps.reporting import acrobot_attractor as report


def probe_arrays():
    a, b = report.analytic_map()
    states, controls = [np.zeros(5)], [np.zeros(1)]
    for h in (1e-6, 5e-7):
        for coordinate in range(5):
            for sign in (1, -1):
                x, u = np.zeros(5), np.zeros(1)
                if coordinate < 4:
                    x[coordinate + 1] = sign * h
                else:
                    u[0] = sign * h
                states.append(x)
                controls.append(u)
    held = np.random.Generator(np.random.PCG64(3023000999)).normal(0, 1e-5, (16, 5))
    states.extend(np.column_stack([np.zeros(16), held[:, :4]]))
    controls.extend(held[:, 4:])
    states, controls = np.array(states), np.array(controls)
    after = np.column_stack([np.full(37, 0.02), states[:, 1:] @ a.T + controls @ b.T])
    return states, controls, after


def lqr_packet():
    a, b = report.analytic_map()
    q = np.diag([0.0, 0.0, 0.01, 0.01])
    q[:2, :2] = [[4, 2], [2, 1]]
    r = np.array([[0.01]])
    p = solve_discrete_are(a, b, q, r)
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return dict(
        A=a, B=b, Q=q, R=r, P=p, K=k, unsaturated_rho=float(1 / (k @ np.linalg.solve(p, k.T))[0, 0])
    )


def test_two_finite_difference_steps_and_unseen_predictions_are_bound():
    x, u, n = probe_arrays()
    got = report.native_jacobians(x, u, n)
    assert got["finite_difference_error"] < 1e-12
    assert got["held_out_error"] < 1e-12
    broken = x.copy()
    broken[[1, 2]] = broken[[2, 1]]
    with pytest.raises(ValueError, match="FD perturbation"):
        report.native_jacobians(broken, u, n)
    broken = n.copy()
    broken[-1, 1] += 1e-5
    with pytest.raises(ValueError, match="Held-out"):
        report.native_jacobians(x, u, broken)


def test_riccati_and_gear_bound_reject_wrong_gain_or_native_map():
    packet = lqr_packet()
    native = report.native_jacobians(*probe_arrays())
    got = report.riccati_measurements(packet, native)
    assert got["spectral_radius"] < 1
    broken = dict(packet, K=packet["K"] * 0.5)
    with pytest.raises(ValueError, match="feedback gain"):
        report.riccati_measurements(broken, native)
    with pytest.raises(ValueError, match="first native"):
        report.riccati_measurements(packet, dict(native, B=native["B"] / 2))


def test_height_success_is_separate_from_empirical_capture():
    x = np.zeros((501, 5))
    x[:, 0] = np.arange(501) * 0.02
    u = np.zeros((500, 1))
    x[:, 3] = 0.1
    got = report.trace_measurements(x, u)
    assert got["height_success"] and not got["capture"]
    x[:, 3] = 0
    x[:, 1] = 2 * np.pi
    assert report.trace_measurements(x, u)["capture"]
    with pytest.raises(ValueError, match=r"N\+1"):
        report.trace_measurements(x[:-1], u)


def test_signed_energy_and_dense_costs_are_post_state_quantities():
    x = np.array([[0.02, 0.0, 0.0, 0.0, 0.0], [0.02, np.pi, 0.0, 0.0, 0.0]])
    u = np.zeros((2, 1))
    dense, td = report.costs(x, u, "dense")
    energy, te = report.costs(x, u, "energy")
    np.testing.assert_allclose(dense, [0, 16], rtol=0, atol=1e-12)
    assert energy[0] == 0 and energy[1] > 17 and td.sum() == 0 and te[1] == 100


def test_shell_selection_cannot_jump_over_failed_inner_shell():
    rows = [
        dict(kind="shell", factor=f, capture=f != 4.0)
        for f in (0.25, 1.0, 4.0, 16.0, 64.0, 256.0)
        for _ in range(24)
    ]
    assert report.selected_shell(rows, 3.0) == 3.0
    with pytest.raises(ValueError, match="24"):
        report.selected_shell(rows[:23], 3.0)


def test_gibbs_temperature_changes_preserve_affine_score_units():
    scores = np.linspace(0, 10, 256)
    lo, hi = 1e-6, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        w = np.exp(-(scores - scores.min()) / mid)
        w /= w.sum()
        if 1 / (w * w).sum() < 32:
            lo = mid
        else:
            hi = mid
    temperature = (lo + hi) / 2
    w = np.exp(-scores / temperature)
    w /= w.sum()
    a = report.weight_measurements(scores, w, temperature)
    b = report.weight_measurements(17 * scores - 63, w, 17 * temperature)
    assert a["within_aim"] and abs(a["ess"] - b["ess"]) < 1e-10
    with pytest.raises(ValueError, match="Gibbs"):
        report.weight_measurements(scores, np.roll(w, 1), temperature)


def test_module_has_no_simulator_or_worker_imports():
    before = set(sys.modules)
    importlib.reload(report)
    assert not ({"mujoco", "torch", "src.gps.mppi_gps.attractor_lqr"} & (set(sys.modules) - before))


def test_handoff_keeps_complete_context_and_first_basin_entry(tmp_path):
    class Group(dict):
        attrs = {"reset_at_join": False}

    p = np.eye(4)
    x = np.array([[0.0, 0.3, 0.0, 0.0, 0.0], [0.02, 0.1, 0.0, 0.0, 0.0]])
    prefix = Group(
        states=x,
        actions=np.zeros((1, 1)),
        warmstarts=np.zeros((2, 2)),
        integration=np.zeros((2, 28)),
    )
    suffix = Group(states=x[-1:], warmstarts=np.zeros((1, 2)), integration=np.zeros((1, 28)))
    got = report.check_handoff(report.Review(tmp_path, 10), prefix, suffix, p, 0.02)
    assert got["join_control"] == 1
    suffix["warmstarts"][0, 0] = 1
    with pytest.raises(ValueError, match="handoff context"):
        report.check_handoff(report.Review(tmp_path, 10), prefix, suffix, p, 0.02)


def test_report_retains_missing_input_failure_without_physics(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"protocol":"wrong"}')
    got = report.render(tmp_path, cfg, tmp_path / "report")
    assert got["status"] == "FAILED_OR_INCOMPLETE"
    assert got["new_physics_controls"] == 0
    assert (tmp_path / "report/independent_audit.json").exists()
