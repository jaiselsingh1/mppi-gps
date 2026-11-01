"""Synthetic S1.1 lifecycle faults; no neural, optimizer or simulator work."""

from __future__ import annotations

import inspect
import json
import sys
import time
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from src.gps.mppi_gps import provenance, runtime
from src.gps.mppi_gps.experiments import student_source_evaluation as study
from src.gps.mppi_gps.source_evaluation_inputs import audit_known_reset_ledgers
from src.gps.mppi_gps.source_outcomes import add_work


def budgets(**values):
    return dict(
        budgets=dict(
            live_controls=2,
            physics_substeps=8,
            neural_rows=2,
            checkpoint_loads=2,
            storage_bytes=1024**3,
            **values,
        )
    )


def test_meter_rotation_keeps_global_neural_and_physics_reservations(tmp_path):
    config = budgets()
    first = study.EvaluationMeter(
        tmp_path / "run/evidence/episodes/000", config, {}, time.monotonic() + 10, "episode"
    )
    calls = []
    for _ in range(2):
        first.measure("neural_rows", 1, lambda: calls.append("synthetic"))
        first._query(
            "live",
            1,
            4,
            dict(input=np.array(0)),
            lambda: 0,
            lambda value: dict(output=np.array(value)),
        )
    first.close()
    original = (first.out / "raw.h5").read_bytes()
    prior = first.totals()
    second = study.EvaluationMeter(
        tmp_path / "run/evidence/episodes/001", config, prior, time.monotonic() + 10, "episode"
    )
    with pytest.raises(runtime.BudgetExhausted, match="cumulative"):
        second.measure("neural_rows", 1, lambda: calls.append("forbidden"))
    with pytest.raises(runtime.BudgetExhausted, match="cumulative"):
        second._query("live", 1, 4, {}, lambda: calls.append("forbidden"), lambda value: {})
    second.close()
    assert calls == ["synthetic", "synthetic"]
    assert (first.out / "raw.h5").read_bytes() == original
    assert prior["requested_live_controls"] == prior["requested_neural_rows"] == 2
    assert second.totals()["requested_controls"] == 0


def test_rotating_live_wrapper_calls_native_step_once_without_reopening_prior(
    tmp_path, monkeypatch
):
    class Environment:
        _frame_skip, action_dim, state_dim = 4, 6, 19
        calls = 0

        def batch_rollout(self, *args, **kwargs):
            raise AssertionError("No planner calls allowed")

        def step(self, action):
            self.calls += 1
            return np.zeros(17), 0.0, False, {}

    monkeypatch.setattr(
        runtime,
        "snapshot",
        lambda env: dict(state=np.zeros(19), warmstart=np.zeros(9), integration=np.zeros(40)),
    )
    env = Environment()
    first = study.EvaluationMeter(
        tmp_path / "run/evidence/episodes/000", budgets(), {}, time.monotonic() + 10, "episode"
    )
    first.instrument(env)
    env.step(np.zeros(6))
    first.close()
    original = (first.out / "raw.h5").read_bytes()
    second = study.EvaluationMeter(
        tmp_path / "run/evidence/episodes/001",
        budgets(),
        first.totals(),
        time.monotonic() + 10,
        "episode",
    )
    second.instrument(env)
    env.step(np.zeros(6))
    second.close()
    assert env.calls == 2
    assert first.totals()["completed_live_controls"] == 1
    assert second.totals()["completed_live_controls"] == 1
    assert (first.out / "raw.h5").read_bytes() == original


@pytest.mark.parametrize(
    "phase,kind",
    [
        ("episode", "backward_rows"),
        ("episode", "optimizer_steps"),
        ("episode", "checkpoint_loads"),
        ("loads", "neural_rows"),
    ],
)
def test_phase_forbids_extra_components_before_callable(tmp_path, phase, kind):
    path = tmp_path / ("run/evidence/loads" if phase == "loads" else "run/evidence/episodes/000")
    meter = study.EvaluationMeter(path, budgets(), {}, time.monotonic() + 10, phase)
    calls = []
    with pytest.raises(ValueError, match="forbids"):
        meter.measure(kind, 1, lambda: calls.append("forbidden"))
    with pytest.raises(ValueError, match="native"):
        meter._query("model", 1, 4, {}, lambda: calls.append("forbidden"), lambda value: {})
    meter.close()
    assert not calls


def test_global_deadline_and_storage_guard_stop_before_callable(tmp_path, monkeypatch):
    meter = study.EvaluationMeter(
        tmp_path / "run/evidence/loads", budgets(), {}, time.monotonic() - 1, "loads"
    )
    calls = []
    with pytest.raises(runtime.BudgetExhausted, match="wall"):
        meter.measure("checkpoint_loads", 1, lambda: calls.append(1))
    meter.deadline = time.monotonic() + 10
    meter.static_bytes = meter.config["budgets"]["storage_bytes"]
    with pytest.raises(runtime.BudgetExhausted, match="storage"):
        meter.measure("checkpoint_loads", 1, lambda: calls.append(1))
    meter.close()
    assert not calls


@pytest.fixture
def worker_fixture(tmp_path, monkeypatch):
    """Four tiny fake episodes exercise orchestration without importing MuJoCo."""
    out = tmp_path / "run"
    out.mkdir()
    for name in ("config.json", "input_manifest.json", "manifest.json", "launch_receipt.json"):
        (out / name).write_text("{}")
    config = budgets()
    config["budgets"].update(episodes=4, live_controls=4, physics_substeps=16, neural_rows=4)
    config.update(
        evaluation_order=["reference", "incoming_s0"],
        primary_seeds=[1270000000],
        confirmation_seeds=[1280000000],
        episode_steps=1000,
        ack_timeout_seconds=60,
        poll_seconds=0.1,
        task=dict(task={}),
        competence=dict(minimum_walking_successes_per_panel=15),
        reference=dict(sha256="b" * 64, policy_hash="b" * 64),
        checkpoints=dict(
            incoming_s0=dict(
                sha256="a" * 64,
                policy_hash="a" * 64,
                optimizer_fingerprint=provenance.fingerprint(None),
            )
        ),
    )
    launch = dict(deadline_monotonic=time.monotonic() + 100, boot_id="boot")
    initial = dict(
        execution_complete=False,
        status="evaluating",
        error=None,
        points=dict(incoming_s0=dict(checkpoint=dict(file="incoming_s0.pt"))),
        evaluations=dict(primary={}, confirmation={}),
        segments=[],
        gates={},
        work={},
    )
    monkeypatch.setattr(study, "verify_inputs", lambda path: (config, launch))
    monkeypatch.setattr(study, "inherited_result", lambda path, cfg: deepcopy(initial))
    monkeypatch.setattr(study, "platform_identity", lambda cfg: dict(device="fake CPU"))
    monkeypatch.setattr(study, "read_boot_id", lambda: "boot")
    monkeypatch.setattr(study, "source_screen", lambda records, cfg: dict(walking_screen=False))
    state = dict(failure=None, published=[], env_closes=0)

    class Actor:
        def __init__(self, name):
            self.name = name

        def eval(self):
            return self

        def hash(self):
            return (
                config["reference"]["policy_hash"]
                if self.name == "reference"
                else config["checkpoints"][self.name]["policy_hash"]
            )

        def predict(self, observation, meter):
            return meter.measure("neural_rows", 1, lambda: np.zeros(6, dtype=np.float32))

    class Policy:
        @staticmethod
        def from_checkpoint(path, device):
            assert device == "cpu"
            return Actor(path.stem), None, {}

    class Reference:
        @staticmethod
        def from_checkpoint(path, meter):
            def load():
                if state["failure"] == "load":
                    raise ValueError("synthetic load failure")
                return Actor("reference")

            return meter.measure("checkpoint_loads", 1, load)

    class Environment:
        def __init__(self, **kwargs):
            pass

        def close(self):
            state["env_closes"] += 1
            if state["failure"] == "environment_close":
                raise OSError("synthetic environment cleanup failure")

    policy_module, reference_module, env_module = (
        ModuleType("policy"),
        ModuleType("diagnostic_policies"),
        ModuleType("walker2d"),
    )
    policy_module.Policy = Policy
    reference_module.LegacySACActor = Reference
    env_module.Walker2d = Environment
    import src.gps.mppi_gps as package

    monkeypatch.setitem(sys.modules, "src.gps.mppi_gps.policy", policy_module)
    monkeypatch.setattr(package, "policy", policy_module, raising=False)
    monkeypatch.setitem(sys.modules, "src.gps.mppi_gps.diagnostic_policies", reference_module)
    monkeypatch.setitem(sys.modules, "src.envs.walker2d", env_module)

    def fake_episode(env, controller, seed, steps, meter, tag):
        physical = np.zeros((2, 19))
        physical[:, 2] = [1.0, 0.7]
        physical[1, 0], physical[:, 10] = 0.008, 1.5
        warm, integration = np.zeros((2, 9)), np.zeros((2, 91))
        observation = np.concatenate((physical[0, 2:10], physical[0, 10:19]))
        meter.context.update(episode=tag, step=0)
        meter._counts["resets"] += 1
        meter.record("reset_complete", tag=tag, seed=seed)
        action, info = controller(physical[0], warm[0], observation, 0)
        meter._query(
            "live",
            1,
            4,
            dict(action=action),
            lambda: 0,
            lambda result: dict(output=np.array(result)),
        )
        if state["failure"] == "episode":
            raise ValueError("synthetic post-query failure")
        raw = meter.save_arrays(
            tag,
            states=physical,
            warmstarts=warm,
            integrations=integration,
            actions=action[None],
            observations=observation[None],
            costs=np.array([25.0]),
            velocities=np.array([1.5]),
            infos=dict(rows=[dict(controller=info)]),
        )
        episode = dict(
            tag=tag,
            seed=seed,
            steps=1,
            success=False,
            survived=False,
            mae=0.0,
            cost=5020.0,
            complete=True,
            terminal_cost=20.0,
            terminal_adjustment=0.0,
            absorbing_cost=4995.0,
            raw=raw,
        )
        meter._counts["episodes"] += 1
        meter.record("episode_complete", **episode)
        return SimpleNamespace(
            **episode, states=physical, warmstarts=warm, integrations=integration
        )

    monkeypatch.setattr(runtime, "execute_episode", fake_episode)
    original_seal = study.seal_segment

    def seal(*args):
        if state["failure"] == "seal":
            raise OSError("synthetic receipt commit failure")
        return original_seal(*args)

    def publish(run, path, identifier, ordinal, boot, deadline, journal, **kwargs):
        state["published"].append(identifier)
        if state["failure"] == "setup_ack" or (ordinal >= 0 and state["failure"] == "episode_ack"):
            raise TimeoutError("synthetic missing independent ACK")
        return dict(
            segment=str(path.relative_to(run)),
            receipt_sha256=provenance.sha(path / "receipt.json"),
            request=dict(file="synthetic", sha256="c" * 64),
            ack=dict(file="synthetic", sha256="d" * 64),
            mirror_commit_sha256="e" * 64,
            wait_seconds=0.0,
        )

    monkeypatch.setattr(study, "seal_segment", seal)
    monkeypatch.setattr(study, "publish_and_wait", publish)
    return out, config, state


@pytest.mark.parametrize(
    "failure,loads,controls,episodes",
    [
        ("load", 0, 0, 0),
        ("setup_ack", 2, 0, 0),
        ("episode", 2, 1, 0),
        ("seal", 2, 1, 1),
        ("episode_ack", 2, 1, 1),
    ],
)
def test_failed_lifecycle_preserves_paid_work_once(
    worker_fixture, failure, loads, controls, episodes
):
    out, config, state = worker_fixture
    state["failure"] = failure
    result = study.worker(out)
    assert not result["execution_complete"] and result["status"] == "stopped_partial"
    assert result["work"].get("completed_checkpoint_loads", 0) == loads
    assert result["work"]["completed_live_controls"] == controls
    assert result["work"]["episodes"] == episodes
    assert result["work"]["completed_neural_rows"] == controls
    assert result["phase_work"]["eval"] == result["work"]
    assert not result["segments"]
    assert json.loads((out / "result.json").read_text())["work"] == result["work"]
    if failure in ("setup_ack", "episode_ack"):
        assert (out / result["uncommitted_partial"]["directory"] / "receipt.json").is_file()
    if failure == "load":
        assert result["work"]["unknown_partial_checkpoint_loads"] == 1


def test_synthetic_roster_uses_local_cohort_length_and_no_final_walking_claim(worker_fixture):
    out, config, state = worker_fixture
    result = study.worker(out)
    assert result["execution_complete"] and len(result["segments"]) == 4
    assert result["work"]["completed_live_controls"] == result["work"]["episodes"] == 4
    assert result["work"]["completed_checkpoint_loads"] == 2
    assert result["gates"]["confirmed_walking_benefit"] is False
    assert result["actors_initial"] == result["actors_final"]
    assert all(
        row["episodes"][0]["segment"].endswith(row["episodes"][0]["tag"])
        for panel in result["evaluations"].values()
        for row in panel.values()
    )
    reference = json.loads((out / result["segments"][0]["segment"] / "receipt.json").read_text())
    assert reference["ownership"]["optimizer_fingerprint"] == provenance.fingerprint(None)
    assert state["env_closes"] == 1


def test_cleanup_failure_keeps_closed_costs_and_marks_execution_incomplete(worker_fixture):
    out, config, state = worker_fixture
    state["failure"] = "environment_close"
    result = study.worker(out)
    assert not result["execution_complete"] and result["error"]["operation"] == "environment_close"
    assert result["work"]["episodes"] == 4
    assert len(result["segments"]) == 4
    assert (out / "result.json").is_file()


def test_meter_cleanup_fault_keeps_actual_prefix_and_earlier_cause(worker_fixture, monkeypatch):
    out, config, state = worker_fixture
    original_close = study.EvaluationMeter.close

    def close(meter):
        original_close(meter)
        if meter.phase == "episode":
            raise OSError("synthetic post-close persistence failure")

    monkeypatch.setattr(study.EvaluationMeter, "close", close)
    result = study.worker(out)
    assert not result["execution_complete"]
    assert result["work"]["completed_live_controls"] == result["work"]["episodes"] == 1
    assert result["work"]["completed_checkpoint_loads"] == 2
    assert result["uncommitted_partial"]["ledger_closed"] is True
    assert result["cleanup_errors"][0]["operation"] == "meter_close"
    assert result["error"]["message"] == "synthetic post-close persistence failure"
    assert (out / "result.json").is_file()


def test_configured_episode_count_must_be_fully_executed(worker_fixture):
    out, config, state = worker_fixture
    config["budgets"]["episodes"] = 5
    result = study.worker(out)
    assert not result["execution_complete"]
    assert "roster" in result["error"]["message"]
    assert result["work"]["episodes"] == 4


def test_freshness_census_hashes_ledgers_and_rejects_hidden_recorded_seed(tmp_path):
    path = tmp_path / "runs/prior/evidence/ledger.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(event="reset_complete", seed=1270000000)) + "\n")
    excluded = tmp_path / "runs/prior/source/nested/evidence/ledger.jsonl"
    excluded.parent.mkdir(parents=True)
    excluded.write_text(json.dumps(dict(seed=1280000000)) + "\n")
    proof = audit_known_reset_ledgers(tmp_path, [1280000000])
    assert proof["passed"] and proof["files"][0]["sha256"] == provenance.sha(path)
    assert len(proof["files"]) == 1
    assert proof["files"][0]["bytes"] == path.stat().st_size
    assert not audit_known_reset_ledgers(tmp_path, [1270000000])["passed"]
    path.write_text(path.read_text() + '{"truncated":')
    invalid = audit_known_reset_ledgers(tmp_path, [1280000000])
    assert (
        not invalid["passed"]
        and invalid["errors"]
        and invalid["files"][0]["sha256"] == provenance.sha(path)
    )


def test_freshness_helper_is_extractable_standard_library_source(tmp_path):
    namespace = {}
    exec(
        "from __future__ import annotations\n" + inspect.getsource(audit_known_reset_ledgers),
        namespace,
    )
    assert namespace["audit_known_reset_ledgers"](tmp_path, [1270000000])["seeds"] == [1270000000]


def test_disjoint_cost_addition_does_not_charge_parent_twice():
    parent = dict(requested_live_controls=100, wall_seconds=2.0)
    latest = dict(requested_live_controls=1, wall_seconds=0.5)
    charged_before_ack = add_work(parent, latest)
    assert add_work(parent, latest) == charged_before_ack
    assert add_work(charged_before_ack, latest)["requested_live_controls"] == 102
