"""Pure arithmetic/bookkeeping fixtures; no engine, optimizer or actor calls."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional.shooting_reference import (
    BudgetedQueryExecutor, ShootingConfig, ShootingProblem, SolveLimit,
    derivative_blocks, finite_difference_inputs, protocol, tip_squared,
    tip_squared_gradient, tracking_action, tracking_gains, solve)
from src.gps.bidirectional.hetero_executor import SharedControlBudget


class MemoryArchive:
    def __init__(self):
        self.events = []
        self.arrays = {}

    def append(self, kind, **payload):
        self.events.append(dict(kind=kind, **payload))
        return str(len(self.events))

    def save_npz(self, path, **arrays):
        if path in self.arrays:
            raise FileExistsError(path)
        self.arrays[path] = {k:np.array(v, copy=True) for k,v in arrays.items()}
        return dict(file=path, sha256="0"*64, bytes=1)


class ArithmeticMap:
    """Linear array calculation; deliberately not a simulator or task model."""
    def __init__(self):
        self.calls = []
        self.A = np.array([[1., .1, .02, 0.], [0., 1., 0., .03],
                           [0., 0., .9, 0.], [0., 0., 0., .8]])
        self.B = np.array([.01, 0., .1, .2])

    def evaluate(self, states, actions, *, purpose, metadata=None):
        self.calls.append((np.array(states), np.array(actions), purpose))
        out = np.column_stack((states[:, 0]+.01, states[:, 1:]@self.A.T+actions*self.B))
        return SimpleNamespace(states=out, query_id=f"fake{len(self.calls)}")


def problem():
    model, archive = ArithmeticMap(), MemoryArchive()
    return ShootingProblem(np.zeros(5), model, archive, "fixture"), model, archive


def test_work_bound_has_no_unallocated_growth():
    p = protocol()
    assert sum(v for k,v in p["work_bound"].items() if k not in
               ("total", "maximum_heterogeneous_solver_batches")) == 1_600_000
    assert p["work_bound"]["maximum_heterogeneous_solver_batches"] == 480
    assert p["actor_rows"] == p["adam_steps"] == 0
    assert "not MPPI" in p["interpretation"]


@pytest.mark.parametrize("changes", [dict(T=801), dict(max_controls=1_600_001),
    dict(reset_seeds=(191000,)), dict(max_residual_calls=81), dict(final_min_capture_and_local=28)])
def test_protocol_rejects_budget_or_gate_changes(changes):
    with pytest.raises(ValueError):
        ShootingConfig(**changes)


def test_fk_gradient_matches_independent_scalar_difference():
    states = np.array([[.4, -.7, 1., 2.], [2., .3, -.1, .2]])
    gradient = tip_squared_gradient(states)
    for j in range(4):
        a, b = states.copy(), states.copy()
        a[:, j] += 1e-6; b[:, j] -= 1e-6
        np.testing.assert_allclose(gradient[:, j], (tip_squared(a)-tip_squared(b))/2e-6,
                                   rtol=1e-6, atol=1e-8)


def test_stencils_use_native_clock_and_bound_aware_actions():
    states = np.array([[0., 1., 2., 3., 4.], [7.99, -1., -2., -3., -4.]])
    actions = np.array([[-1.], [1.]])
    ss, uu, denominator = finite_difference_inputs(states, actions)
    assert ss.shape == (20, 5) and uu.shape == (20, 1)
    np.testing.assert_array_equal(ss[:, 0], np.repeat(states[:, 0], 10))
    assert np.max(abs(uu)) <= 1
    np.testing.assert_allclose(denominator, [1e-5, 1e-5], rtol=0, atol=1e-15)


def test_derivative_blocks_recover_fixed_linear_arithmetic_map():
    model = ArithmeticMap()
    states = np.array([[0., .1, .2, .3, .4], [1., -.1, -.2, -.3, -.4]])
    ss, uu, denominator = finite_difference_inputs(states, np.array([[-1.], [1.]]))
    out = model.evaluate(ss, uu, purpose="fixture").states
    A, B = derivative_blocks(out, denominator)
    np.testing.assert_allclose(A, np.broadcast_to(model.A, (2, 4, 4)), atol=1e-10)
    np.testing.assert_allclose(B, np.broadcast_to(model.B[None, :, None], (2, 4, 1)), atol=1e-10)


def test_capture_uses_post_boundaries701_through800_and_local600():
    p, _, _ = problem()
    states = np.zeros((801, 4))
    states[700, 0] = np.pi
    assert not np.any(p._capture(states))  # 700 is outside final100.
    states[701, 0] = np.pi
    residual = p._capture(states)
    assert residual[0] > 0 and np.count_nonzero(residual) == 1
    states[600, 0] = np.pi
    assert p._capture(states)[-1] > 0
    states[:] = 0; states[800, 2] = 1.
    assert p._capture(states)[199] > 0


def test_initial_guess_is_bounded_but_not_claimed_experience():
    p, model, _ = problem()
    for winding in (-1, 1):
        z = p.initial_guess(winding)
        low, high = p.bounds()
        assert np.all(z >= low) and np.all(z <= high)
        physical, u = p.unpack(z)
        assert physical.shape == (801, 4) and u.shape == (800, 1)
        assert not np.any(u)
    assert model.calls == []


def test_residual_callback_and_sparse_jacobian_are_consistent():
    p, model, archive = problem()
    z = np.zeros(4000)
    r = p.residual(z)
    J = p.jacobian(z)
    assert r.shape == (3401,) and J.shape == (3401, 4000)
    assert not np.any(r)
    assert len(model.calls[0][0]) == 800 and len(model.calls[1][0]) == 8000
    # Independent variable differences for representative state/control columns.
    for column in (0, 2, 1700, 3199, 3200, 3999):
        a, b = z.copy(), z.copy()
        a[column] += 1e-6; b[column] -= 1e-6
        measured = (p.residual(a)-p.residual(b))/2e-6
        np.testing.assert_allclose(J[:, column].toarray().ravel(), measured, atol=1e-8)
    assert all(not row.get("accepted_as_experience", False) for row in archive.events)


def test_active_capture_jacobian_matches_scalar_difference():
    p, _, _ = problem()
    z = np.zeros(4000)
    z[4*700:4*700+4] = [.4, .2, .6, -.2]
    z[4*599] = .9
    J = p.jacobian(z)
    for column in (4*700, 4*700+1, 4*700+2, 4*599):
        a, b = z.copy(), z.copy()
        a[column] += 1e-6; b[column] -= 1e-6
        measured = (p.residual(a)-p.residual(b))/2e-6
        np.testing.assert_allclose(J[:, column].toarray().ravel(), measured, rtol=1e-5, atol=1e-7)


def test_callbacks_stop_before_any_extra_array_query():
    p, model, _ = problem()
    p.residual_calls = 80
    with pytest.raises(SolveLimit): p.residual(np.zeros(4000))
    p.jacobian_calls = 40
    with pytest.raises(SolveLimit): p.jacobian(np.zeros(4000))
    assert model.calls == []


def test_tracking_gain_and_wrapped_error_are_deterministic_arithmetic():
    A = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    B = np.zeros((3, 4, 1))
    K = tracking_gains(A, B)
    assert not np.any(K)
    states, actions = np.zeros((4, 5)), np.zeros((3, 1))
    snapshot = SimpleNamespace(elapsed=0, state=np.array([0., 2*np.pi, 0., 0., 0.]))
    np.testing.assert_allclose(tracking_action(snapshot, states, actions, np.ones((3, 1, 4))), [0.], atol=1e-15)
    snapshot.state[3] = 100.
    np.testing.assert_array_equal(tracking_action(snapshot, states, actions, np.ones((3, 1, 4))), [-1.])


def test_ordinary_wrapper_cannot_spend_past_shared_cap():
    class Fake:
        def __init__(self):
            self.task, self.archive, self.calls = object(), MemoryArchive(), 0
            self.latest_query_id = None
        def step(self, action, metadata=None):
            self.calls += 1
            return "returned"
    fake = Fake(); budget = SharedControlBudget(1)
    wrapper = BudgetedQueryExecutor(fake, budget)
    assert wrapper.step(np.zeros(1)) == "returned"
    with pytest.raises(RuntimeError): wrapper.step(np.zeros(1))
    assert fake.calls == 1 and budget.totals()["requested_controls"] == 1


def test_rejection_before_new_query_does_not_reuse_prior_completion(tmp_path):
    ledger = tmp_path/"ledger.jsonl"
    ledger.write_text('{"kind":"query_requested","query_id":"old"}\n'
                      '{"kind":"query_error","query_id":"old","call_returned":true}\n')
    class Rejecting:
        latest_query_id = "old"
        task = object()
        archive = MemoryArchive()
        def step(self, action, metadata=None):
            raise ValueError("Rejected before any new query intent")
    Rejecting.archive.ledger_path = ledger
    budget = SharedControlBudget(1)
    wrapper = BudgetedQueryExecutor(Rejecting(), budget)
    with pytest.raises(ValueError, match="Rejected"):
        wrapper.step(np.zeros(1))
    counts = budget.totals()
    assert counts["requested_controls"] == counts["not_attempted_error_controls"] == 1
    assert counts["completed_controls"] == counts["unknown_error_controls"] == 0


def test_no_engine_or_actor_imports_in_reference_module_source():
    # Static ownership guard: actual engine imports live only inside runtime entry.
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/"src/gps/bidirectional/shooting_reference.py"
    tree = ast.parse(path.read_text())
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [name.name for node in top_imports for name in node.names]
    assert not any(name.startswith(("mujoco", "torch", "scipy")) for name in names)


def test_frozen_packet_verifies_exact_copied_sources_and_rejects_mutation(tmp_path, monkeypatch):
    from scripts.diagnostics import run_native_shooting_reference as cli
    origin = tmp_path/"origin"; origin.mkdir()
    (origin/"module.py").write_text("constant = 1\n")
    monkeypatch.setattr(cli, "ROOT", origin)
    monkeypatch.setattr(cli, "FILES", ("module.py",))
    monkeypatch.setattr(cli, "versions", lambda:{"synthetic": "fixed"})
    output = tmp_path/"packet"
    cli.freeze(output, ShootingConfig(), entry_epoch=1.)
    manifest, launch = cli.verify_frozen(output)
    assert manifest["source_fingerprint"] == launch["source_fingerprint"]
    with pytest.raises(FileExistsError):
        cli.freeze(output, ShootingConfig(), entry_epoch=1.)
    (output/"source/module.py").write_text("constant = 2\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        cli.verify_frozen(output)


def test_frozen_packet_rejects_runtime_package_drift(tmp_path, monkeypatch):
    from scripts.diagnostics import run_native_shooting_reference as cli
    origin = tmp_path/"origin"; origin.mkdir()
    (origin/"module.py").write_text("constant = 1\n")
    monkeypatch.setattr(cli, "ROOT", origin)
    monkeypatch.setattr(cli, "FILES", ("module.py",))
    monkeypatch.setattr(cli, "versions", lambda:{"synthetic": "fixed"})
    output = tmp_path/"packet"
    cli.freeze(output, ShootingConfig(), entry_epoch=1.)
    monkeypatch.setattr(cli, "versions", lambda:{"synthetic": "changed"})
    with pytest.raises(ValueError, match="package versions"):
        cli.verify_frozen(output)


def test_real_archive_solve_summary_does_not_collide_with_reserved_clock(tmp_path, monkeypatch):
    """Use the actual durable logger; no real solver or simulator is called."""
    from scipy import optimize
    from src.gps.bidirectional.evidence import Archive
    calls = []
    def fake_least_squares(fun, initial, **kwargs):
        calls.append("mocked_solver")
        fun(initial)  # ArithmeticMap only; no engine/query executor exists.
        return SimpleNamespace(success=False, status=0, message="fixed synthetic result", nfev=1, njev=0)
    monkeypatch.setattr(optimize, "least_squares", fake_least_squares)
    with Archive(tmp_path/"archive") as archive:
        p = ShootingProblem(np.zeros(5), ArithmeticMap(), archive, "synthetic")
        actions, record = solve(p, -1)
        assert actions.shape == (800, 1)
        assert "solve_wall_seconds" in record and "wall_seconds" not in record
    import json
    events = [json.loads(line) for line in (tmp_path/"archive/ledger.jsonl").read_text().splitlines()]
    end = next(row for row in events if row["kind"] == "shooting_solve_end")
    assert end["solve_wall_seconds"] >= 0 and end["wall_seconds"] >= 0
    assert end["accepted_as_experience"] is False and calls == ["mocked_solver"]
