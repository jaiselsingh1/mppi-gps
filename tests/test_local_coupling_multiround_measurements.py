"""Pure admission and fake-physics L3 measurement lifecycle; no real forwards."""

from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps import local_coupling_controller as controller
from src.gps.mppi_gps import local_coupling_multiround_measurements as measure
from src.gps.mppi_gps.provenance import write
from src.gps.mppi_gps.trajectory_pi_native import binding
from src.gps.mppi_gps.trajectory_pi_student import observations


def outcome(*, n=300, fell=False, mae=0.2, cost=100.0):
    survived = n == 300 and not fell
    return dict(
        controls=n,
        requested_controls=300,
        fell=fell,
        full_survival=survived,
        local_walking=survived and mae <= 0.3,
        velocity_mae=mae,
        complete_cost=cost,
    )


def test_admission_uses_actual_task_outcomes_and_fixed_anchor_only():
    candidate = outcome(mae=0.3, cost=110.0) | dict(
        reconstruction=float("inf"), spectral_score=float("nan"), candidate_survival=0.001
    )
    result = measure.teacher_admission(candidate, outcome())
    assert result["admitted"] and result["reasons"] == []
    assert result["score_or_reconstruction_gate"] is False
    rejected = measure.teacher_admission(outcome(cost=110.001), outcome())
    assert not rejected["admitted"]
    assert rejected["reasons"] == ["anchor_cost_noninferiority"]


def test_retention_without_walking_does_not_admit_new_teacher():
    result = measure.teacher_admission(outcome(mae=1.5), outcome())
    assert result["checks"]["full_local_survival"]
    assert not result["admitted"] and result["reasons"] == ["local_walking"]


def test_fall_keeps_retained_episode_and_explicitly_marks_fallback():
    anchor = dict(slot=3, outcome=outcome(), trace="anchor")
    retained = dict(slot=3, outcome=outcome(cost=105), trace="last-admitted")
    failed = dict(slot=3, outcome=outcome(n=57, fell=True, mae=0.5, cost=1500), trace="paid-fall")
    result = measure.admit_or_retain(failed, retained, anchor)
    assert result["selected"] is retained
    assert result["fallback_reused"] and not result["fresh_acquisition_admitted"]
    assert failed["trace"] == "paid-fall"
    admitted = dict(slot=3, outcome=outcome(cost=109), trace="new-admitted")
    replacement = measure.admit_or_retain(admitted, retained, anchor)
    assert replacement["selected"] is admitted and not replacement["fallback_reused"]


def test_contradictory_outcomes_or_invalid_fallback_are_evidence_errors():
    with pytest.raises(ValueError, match="contradict"):
        measure.teacher_admission(outcome(n=57, fell=True) | dict(full_survival=True), outcome())
    with pytest.raises(ValueError, match="fixed fallback"):
        measure.teacher_admission(outcome(), outcome(mae=0.5))
    anchor = dict(slot=0, outcome=outcome())
    with pytest.raises(ValueError, match="no longer satisfies"):
        measure.admit_or_retain(anchor, dict(slot=0, outcome=outcome(cost=111)), anchor)
    with pytest.raises(ValueError, match="one restart"):
        measure.admit_or_retain(anchor, dict(slot=1, outcome=outcome()), anchor)


def test_safeguards_are_not_mutable_after_freeze():
    with pytest.raises(ValueError, match="cannot be changed"):
        measure.teacher_admission(outcome(), outcome(), cost_ratio_max=1.20)


def config(tmp_path):
    path = tmp_path / "config.json"
    write(path, dict(schema="local-coupling-learning-l2-1", launch_ready=True))
    return binding(path)


def saved_trace(path, n=51, *, offset=0.0):
    states = np.zeros((n + 1, 19), dtype=np.float64)
    states[:, 0] = np.arange(n + 1) * 0.008
    states[:, 2], states[:, 10] = 1.2, 1.5
    states[1:, 5] = offset
    with h5py.File(path, "x") as raw:
        for key, value in dict(
            states=states,
            warmstarts=np.zeros((n + 1, 9)),
            integrations=np.zeros((n + 1, 91)),
            observations=observations(states),
            actions=np.full((n, 6), offset),
            costs=np.ones(n),
        ).items():
            raw[key] = value
    return binding(path)


def test_retained_corpora_preserve_all_full_episode_rows(tmp_path):
    spec = config(tmp_path)
    a, b = [], []
    for slot in range(8):
        for name, roster, offset in (("a", a, 0.1), ("b", b, -0.2)):
            roster.append(
                dict(
                    slot=slot, trace=saved_trace(tmp_path / f"{name}{slot}.h5", 300, offset=offset)
                )
            )
    result = measure.build_retained_corpora(a, b, tmp_path / "corpus", config_binding=spec)
    assert result["rows"] == 2400
    with np.load(result["files"]["a"]["path"], allow_pickle=False) as left:
        with np.load(result["files"]["b"]["path"], allow_pickle=False) as right:
            assert left["row_ids"].shape == (2400, 2)
            np.testing.assert_array_equal(left["row_ids"], right["row_ids"])
            np.testing.assert_allclose(left["actions"], 0.1, rtol=0, atol=0)
            np.testing.assert_allclose(right["actions"], -0.2, rtol=0, atol=0)
    b[0]["trace"] = saved_trace(tmp_path / "paid-fall.h5", 100)
    with pytest.raises(ValueError, match="Only full"):
        measure.build_retained_corpora(a, b, tmp_path / "rejected", config_binding=spec)
    assert not (tmp_path / "rejected").exists()


class FakeActor:
    """Array fixture, never a Torch policy or neural inference call."""

    obs_dim = 17
    hidden_dim = 128

    def eval(self):
        return self

    def hash(self):
        return "fake-bound-actor"

    def predict(self, obs, meter=None):
        return np.zeros((*obs.shape[:-1], 6), dtype=np.float32)


class FakeEnvironment:
    """Analytic arrays stand in for physics; no native work is charged."""

    action_dim = 6
    action_bounds = (-np.ones(6), np.ones(6))
    _dt = 0.008

    def batch_rollout(self, initial, actions, **kwargs):
        assert np.array_equal(initial, self.context["state"])
        posts = np.broadcast_to(initial, (*actions.shape[:2], 19)).copy()
        return posts, np.linspace(1.0, 2.0, len(actions)), None

    def step(self, action):
        raise AssertionError("A probe must never execute a native teacher control")

    def close(self):
        self.closed = True


def probe_fixture(tmp_path, monkeypatch, n=51):
    spec = config(tmp_path)
    actor = tmp_path / "fake-checkpoint.pt"
    actor.write_bytes(b"synthetic binding only, never loaded by torch")
    trace = saved_trace(tmp_path / "trace.h5", n, offset=0.1)
    evaluation = tmp_path / "evaluation.json"
    write(
        evaluation,
        dict(
            status="COMPLETED",
            checkpoint=binding(actor),
            policy_hash="fake-bound-actor",
            trace=trace,
            outcome=dict(controls=n),
        ),
    )
    restarts = tmp_path / "restarts.npz"
    with restarts.open("xb") as raw:
        np.savez_compressed(raw, r0_spec=np.asarray(1), r0_dimensions=np.array([9, 9, 6]))
    envs = []

    def environment(job, work):
        value = FakeEnvironment()
        envs.append(value)
        return value

    def restore(env, context, work):
        env.context = deepcopy(context)

    monkeypatch.setattr(measure, "_load_actor", lambda _: FakeActor())
    monkeypatch.setattr(measure, "_env", environment)
    monkeypatch.setattr(measure, "_restore", restore)
    monkeypatch.setattr(measure, "snapshot", lambda env: deepcopy(env.context))
    monkeypatch.setattr(controller, "snapshot", lambda env: deepcopy(env.context))
    job = dict(
        out=str(tmp_path / "probes"),
        config=spec,
        checkpoint=binding(actor),
        evaluation=binding(evaluation),
        restarts=binding(restarts),
        planner=dict(K=256, H=48, initial_iterations=2),
        task={},
        physics_threads=4,
        round=0,
        stream=1,
        arm="positive",
        probe_seeds=[2962000100, 2962000101, 2962000102],
        limits=dict(
            wall_seconds=30,
            output_bytes=64 * 1024**2,
            memory_bytes=4 * 1024**3,
            checkpoint_loads=1,
            policy_constructions=1,
        ),
    )
    return job, envs


def test_probes_use_pre_action_context_zero_plan_and_preserve_missing(tmp_path, monkeypatch):
    job, envs = probe_fixture(tmp_path, monkeypatch)
    result = measure.probe_job(job)
    saved = result["result"]
    assert saved["status"] == "COMPLETED"
    assert saved["observed_anchors"] == 2 and saved["missing_anchors"] == 1
    assert [x["control"] for x in saved["anchors"]] == [0, 50, 100]
    assert saved["anchors"][2]["status"] == "MISSING_AFTER_EXECUTED_PREFIX"
    assert not (Path(job["out"]) / "anchor_100").exists()
    assert saved["actual_native_controls"] == 0
    assert "requested_native_controls" not in result["receipt"]["counts"]
    assert "requested_neural_rows" not in result["receipt"]["counts"]
    assert len(envs) == 1 and envs[0].closed
    assert "zero carried plan" in saved["query_context"]
    for entry in saved["anchors"][:2]:
        assert entry["bank_count"] == 2 and len(entry["full_banks"]) == 2
        with h5py.File(entry["bank_stats"]["path"], "r") as banks:
            assert len(banks) == 2
            np.testing.assert_array_equal(banks["000000/U_before"][:], 0)
            np.testing.assert_array_equal(banks["000001/U_before"][:], banks["000000/U_after"][:])
        with np.load(entry["context"]["path"], allow_pickle=False) as context:
            assert context["state"][0] == entry["control"] * 0.008
            np.testing.assert_array_equal(context["observation"], observations(context["state"]))
            np.testing.assert_array_equal(context["student_action"], np.full(6, 0.1))
            assert context["gap_action_mse"] == np.mean(
                (context["mppi_action"] - context["student_action"]) ** 2
            )


def test_probe_bound_actor_mismatch_stops_before_any_query(tmp_path, monkeypatch):
    job, _ = probe_fixture(tmp_path, monkeypatch)
    other = tmp_path / "other-checkpoint.pt"
    other.write_bytes(b"different actor")
    job["checkpoint"] = binding(other)
    result = measure.probe_job(job)
    assert result["result"]["status"] == "FAILED_STOPPED"
    assert "actual completed standalone actor" in result["result"]["error"]
    assert result["receipt"]["counts"] == {}
    assert not list(Path(job["out"]).glob("anchor_*"))


def test_probe_budget_stop_does_not_retry_or_execute(tmp_path, monkeypatch):
    job, _ = probe_fixture(tmp_path, monkeypatch)
    job["limits"]["checkpoint_loads"] = 0
    result = measure.probe_job(job)
    assert result["result"]["status"] == "FAILED_STOPPED"
    assert "Work budget exhausted" in result["result"]["error"]
    assert result["receipt"]["counts"] == {}
    assert not list(Path(job["out"]).glob("anchor_*"))
