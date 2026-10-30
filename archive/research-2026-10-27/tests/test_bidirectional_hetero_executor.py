"""Authored mocked H1 tests only: no engines, task imports, solver or NN calls."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional.hetero_executor import HeterogeneousH1Executor, SharedControlBudget


class MemoryArchive:
    """In-memory protocol fake; append is the durable-boundary stand-in."""
    def __init__(self):
        self.rows, self.arrays = [], {}
        self._sequence = 0

    def next_query_id(self):
        identifier = f"q{self._sequence:09d}"
        self._sequence += 1
        return identifier

    def append(self, kind, **payload):
        self.rows.append(dict(kind=kind, **deepcopy(payload)))
        return f"event-{len(self.rows)-1:09d}"

    def save_npz(self, name, **arrays):
        assert name not in self.arrays
        self.arrays[name] = {key: np.array(value, copy=True) for key, value in arrays.items()}
        assert not any(value.dtype.hasobject for value in self.arrays[name].values())
        return dict(file=name, sha256=hashlib.sha256(name.encode()).hexdigest(), bytes=1)

    def totals(self):
        return dict(requested_controls=sum(r["requested_controls"] for r in self.rows if r["kind"] == "query_requested"),
                    completed_controls=sum(r["completed_controls"] for r in self.rows
                        if r["kind"] == "query_complete" or (r["kind"] == "query_error" and r["call_returned"])))


def fake_task(**changes):
    task = SimpleNamespace(backend="cpu", task_hash="1"*64,
                           config=SimpleNamespace(to_dict=lambda: dict(T=800, dt=.01)))
    for key, value in changes.items():
        setattr(task, key, value)
    return task


def states_at(*elapsed):
    states = np.zeros((len(elapsed), 5))
    states[:, 0] = np.asarray(elapsed)*.01
    states[:, 1:5] = [.2, -.3, .4, -.5]
    return states


def mock_map(states, actions, *, initial_warmstart):
    """Independent affine fixture, not an Acrobot/transition approximation."""
    output = states[:, None, :].copy()
    output[:, 0, 0] += .01
    output[:, 0, 1] += 2*actions[:, 0, 0]
    output[:, 0, 2] -= actions[:, 0, 0]
    return output, None, None


def setup_executor(cap=20, simulator=mock_map, **options):
    archive, budget = MemoryArchive(), SharedControlBudget(cap)
    executor = HeterogeneousH1Executor(fake_task(), archive, budget=budget, simulator=simulator, **options)
    return executor, archive, budget


def test_heterogeneous_rows_save_native_clocks_zero_warmstart_and_request_before_call():
    seen = []
    executor, archive, budget = setup_executor()
    def simulator(states, actions, *, initial_warmstart):
        assert [row["kind"] for row in archive.rows] == ["query_requested"]
        request = archive.rows[0]
        assert request["requested_controls"] == request["requested_physics_frames"] == 3
        assert request["budget_requested_cumulative"] == budget.requested_controls == 3
        assert request["inputs"]["file"] in archive.arrays
        assert states.shape == (3, 5) and actions.shape == (3, 1, 1)
        np.testing.assert_array_equal(initial_warmstart, np.zeros((3, 2)))
        seen.append(True)
        return mock_map(states, actions, initial_warmstart=initial_warmstart)
    executor._simulator = simulator
    original_states, original_actions = states_at(0, 123, 799), np.array([[-1.], [.125], [1.]])
    before = original_states.copy()
    result = executor.evaluate(original_states, original_actions, purpose="central_state_fd", metadata={"axis": 2})
    assert seen == [True] and not result.states.flags.writeable
    np.testing.assert_array_equal(original_states, before)
    np.testing.assert_allclose(result.states[:, 0], [.01, 1.24, 8.], rtol=0, atol=1e-12)
    raw = archive.arrays[result.result["file"]]
    assert raw["states"].shape == (3, 1, 5) and raw["prestates"].shape == (3, 1, 5)
    np.testing.assert_array_equal(raw["initial_elapsed"], [0, 123, 799])
    np.testing.assert_array_equal(raw["elapsed"], [1, 124, 800])
    assert not raw["initial_warmstart_exact"].any() and not raw["endpoint_warmstart_exact"].any()
    assert str(raw["endpoint_restart_semantics"]) == "explicit-zero-solver-warmstart-intervention"
    assert bool(raw["dynamics_only"]) and str(raw["returns_semantics"]) == "none"
    assert not any("cost" in key or "score" in key for key in raw)
    assert [r["kind"] for r in archive.rows] == ["query_requested", "query_complete"]
    assert executor.ledger_totals == dict(requested_controls=3, completed_controls=3)
    assert budget.totals()["completed_controls"] == 3


@pytest.mark.parametrize("change", ["shape", "empty", "nan", "negative_clock", "terminal_clock", "off_grid", "action_shape", "action_nan", "bound"])
def test_invalid_native_rows_reject_before_reservation_and_simulator(change):
    calls = []
    def simulator(*args, **kwargs):
        calls.append(True)
        raise AssertionError("must not call simulator")
    executor, archive, budget = setup_executor(simulator=simulator)
    states, actions = states_at(0), np.zeros((1, 1))
    if change == "shape": states = states[:, 1:]
    elif change == "empty": states, actions = np.empty((0, 5)), np.empty((0, 1))
    elif change == "nan": states[0, 1] = np.nan
    elif change == "negative_clock": states[0, 0] = -.01
    elif change == "terminal_clock": states[0, 0] = 8.
    elif change == "off_grid": states[0, 0] = .005
    elif change == "action_shape": actions = np.zeros((1, 1, 1))
    elif change == "action_nan": actions[0, 0] = np.nan
    elif change == "bound": actions[0, 0] = np.nextafter(1., 2.)
    with pytest.raises(ValueError):
        executor.evaluate(states, actions, purpose="invalid")
    assert not calls and not archive.rows and not archive.arrays and budget.requested_controls == 0


def test_budget_is_shared_with_ordinary_executor_wrapper_and_never_refunds_errors():
    executor, archive, budget = setup_executor(cap=3)
    ordinary = budget.reserve(2, purpose="ordinary_query_executor")
    budget.complete(ordinary)
    executor.evaluate(states_at(1), np.zeros((1, 1)), purpose="heterogeneous")
    assert archive.rows[0]["budget_requested_cumulative"] == 3
    with pytest.raises(RuntimeError, match="budget"):
        executor.evaluate(states_at(2), np.zeros((1, 1)), purpose="over_cap")
    assert len(archive.rows) == 2 and budget.requested_controls == 3
    totals = budget.totals()
    assert totals["completed_controls"] == 3 and totals["reservations"] == 2


def test_capacity_above_8000_rejects_before_reservation_or_simulator():
    calls = []
    executor, archive, budget = setup_executor(cap=9000, simulator=lambda *a, **k: calls.append(True))
    with pytest.raises(ValueError, match="8000"):
        executor.evaluate(np.zeros((8001, 5)), np.zeros((8001, 1)), purpose="oversized")
    assert not calls and not archive.rows and not archive.arrays and budget.requested_controls == 0


@pytest.mark.parametrize("coercion", ["complex_states", "complex_actions", "numeric_strings"])
def test_non_real_native_arrays_cannot_silently_change_the_requested_physics(coercion):
    executor, archive, budget = setup_executor()
    states, actions = states_at(0), np.zeros((1, 1))
    if coercion == "complex_states":
        states = states.astype(complex); states[0, 1] += 1j
    elif coercion == "complex_actions":
        actions = np.array([[1j]])
    else:
        actions = np.array([[".5"]])
    with pytest.raises(ValueError, match="real numeric"):
        executor.evaluate(states, actions, purpose="invalid_coercion")
    assert not archive.rows and budget.requested_controls == 0


def test_interrupted_simulator_retains_unknown_partial_requested_controls():
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("synthetic interruption")
    executor, archive, budget = setup_executor(cap=2, simulator=interrupted)
    with pytest.raises(KeyboardInterrupt):
        executor.evaluate(states_at(0, 5), np.zeros((2, 1)), purpose="interrupted")
    assert [r["kind"] for r in archive.rows] == ["query_requested", "query_error"]
    error = archive.rows[-1]
    assert error["unknown_partial"] and error["partial_physics_unknown"]
    assert not error["call_returned"] and error["completed_controls"] == 0
    assert budget.totals()["unknown_error_controls"] == 2 and budget.requested_controls == 2
    with pytest.raises(RuntimeError, match="budget"):
        executor.evaluate(states_at(0), np.zeros((1, 1)), purpose="no_retry")


@pytest.mark.parametrize("corruption", ["shape", "clock", "nan", "complex", "input_mutation"])
def test_returned_invalid_output_is_charged_without_false_completion(corruption):
    def simulator(states, actions, *, initial_warmstart):
        result, _, _ = mock_map(states, actions, initial_warmstart=initial_warmstart)
        if corruption == "shape": result = result[:, 0]
        elif corruption == "clock": result[:, :, 0] += .01
        elif corruption == "nan": result[:, :, 1] = np.nan
        elif corruption == "complex": result = result.astype(complex)+1j
        else: states[:, 1] += 1.
        return result, None, None
    executor, archive, budget = setup_executor(simulator=simulator)
    original = states_at(2)
    before = original.copy()
    with pytest.raises(ValueError):
        executor.evaluate(original, np.zeros((1, 1)), purpose="bad_output")
    np.testing.assert_array_equal(original, before)
    assert archive.rows[-1]["kind"] == "query_error" and archive.rows[-1]["call_returned"]
    assert not archive.rows[-1]["unknown_partial"] and archive.rows[-1]["completed_controls"] == 1
    assert budget.totals()["completed_controls"] == 1 and budget.totals()["error_requested_controls"] == 1
    assert not any(r["kind"] == "query_complete" for r in archive.rows)


def test_request_append_failure_prevents_simulation_and_does_not_claim_unknown_physics():
    calls = []
    executor, archive, budget = setup_executor(simulator=lambda *a, **k: calls.append(True))
    def fail_append(*args, **kwargs):
        raise OSError("synthetic fsync failure")
    archive.append = fail_append
    with pytest.raises(OSError, match="fsync"):
        executor.evaluate(states_at(0), np.zeros((1, 1)), purpose="request_failure")
    assert not calls
    assert budget.totals()["not_attempted_error_controls"] == 1
    assert budget.totals()["unknown_error_controls"] == 0


def test_result_save_failure_preserves_known_returned_work():
    executor, archive, budget = setup_executor()
    original_save = archive.save_npz
    def fail_result(name, **arrays):
        if name.endswith("_result.npz"):
            raise OSError("synthetic result persistence failure")
        return original_save(name, **arrays)
    archive.save_npz = fail_result
    with pytest.raises(OSError, match="persistence"):
        executor.evaluate(states_at(0), np.zeros((1, 1)), purpose="result_failure")
    assert archive.rows[-1]["kind"] == "query_error" and archive.rows[-1]["completed_controls"] == 1
    assert budget.totals()["completed_controls"] == 1


def test_reservation_identity_resolution_and_pending_counts():
    budget, other = SharedControlBudget(4), SharedControlBudget(4)
    token = budget.reserve(2, purpose="pending")
    alien = other.reserve(2, purpose="pending")
    assert budget.totals()["inflight_or_interrupted_controls"] == 2
    with pytest.raises(ValueError, match="different"):
        budget.complete(alien)
    budget.error(token, call_returned=False, attempted=True)
    with pytest.raises(ValueError, match="resolved"):
        budget.complete(token)
    assert budget.totals()["unknown_error_controls"] == 2 and budget.requested_controls == 2


def test_bound_aware_mock_control_differences_keep_every_requested_action_native():
    executor, archive, budget = setup_executor(cap=4)
    h = 1e-5
    upper = executor.evaluate(states_at(0, 0), np.array([[1.], [1.-h]]), purpose="backward_at_upper_bound")
    lower = executor.evaluate(states_at(799, 799), np.array([[-1.+h], [-1.]]), purpose="forward_at_lower_bound")
    np.testing.assert_allclose((upper.states[0, 1:5]-upper.states[1, 1:5])/h, [2., -1., 0., 0.], atol=1e-10, rtol=0)
    np.testing.assert_allclose((lower.states[0, 1:5]-lower.states[1, 1:5])/h, [2., -1., 0., 0.], atol=1e-10, rtol=0)
    assert budget.requested_controls == budget.totals()["completed_controls"] == 4
    assert all(np.max(np.abs(packet["actions"])) <= 1. for packet in archive.arrays.values())


@pytest.mark.parametrize("bad_task", [fake_task(backend="warp"), fake_task(task_hash="missing"),
    fake_task(config=SimpleNamespace(to_dict=lambda: dict(T=799, dt=.01))),
    fake_task(config=SimpleNamespace(to_dict=lambda: dict(T=800, dt=.02)))])
def test_non_native_task_contract_rejects_at_construction(bad_task):
    with pytest.raises(ValueError):
        HeterogeneousH1Executor(bad_task, MemoryArchive(), budget=SharedControlBudget(1), simulator=mock_map)


def test_default_adapter_passes_heterogeneous_states_to_only_the_injected_cpu_env():
    seen = []
    def batch_rollout(states, actions, *, initial_warmstart):
        seen.append((states.shape, actions.shape, initial_warmstart.shape))
        return mock_map(states, actions, initial_warmstart=initial_warmstart)
    task = fake_task(env=SimpleNamespace(_use_warp=False, _frame_skip=1, batch_rollout=batch_rollout))
    executor = HeterogeneousH1Executor(task, MemoryArchive(), budget=SharedControlBudget(2))
    executor.evaluate(states_at(4, 798), np.zeros((2, 1)), purpose="default_adapter")
    assert seen == [((2, 5), (2, 1, 1), (2, 2))]
