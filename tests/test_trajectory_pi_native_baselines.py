"""Synthetic orchestration tests; no native integration or neural inference."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
import pytest

from src.gps.mppi_gps import runtime
from src.gps.mppi_gps import trajectory_pi_native as native
from src.gps.mppi_gps import trajectory_pi_native_baselines as study


def physical_arrays(initial, actions):
    states = np.repeat(initial[None], len(actions) + 1, axis=0)
    time = np.arange(len(states)) * 0.008
    states[:, 0] = initial[0] + time
    states[:, 1] = initial[1] + 1.5 * time
    return dict(
        states=states,
        actions=actions,
        costs=np.zeros(len(actions)),
        warmstarts=np.zeros((len(states), 9)),
        integrations=np.zeros((len(states), 91)),
        observations=np.concatenate((states[:, 2:10], states[:, 10:19]), axis=1),
        native_fall_cost=np.zeros(len(actions)),
    )


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    context = np.zeros(19)
    context[2] = 1.25
    context[10] = 1.5
    actions = np.zeros((1000, 6))
    episodes = []
    for slot in (0, 1):
        path = tmp_path / f"original{slot}.npz"
        np.savez(path, **physical_arrays(context, actions))
        episodes.append(
            dict(
                slot=slot,
                original_episode=study.binding(path),
                original_prepared=study.binding(path),
            )
        )
    limits = dict(
        wall_seconds=600,
        output_bytes=64 * 1024**2,
        memory_bytes=2 * 1024**3,
        native_controls=4000,
        model_controls=2000,
        physics_substeps=24000,
        model_queries=2,
        environment_constructions=1,
        restores=6,
        external_mj_forward_calls=6,
        compiled_model_saves=1,
    )
    launch = dict(
        output_directory=str(tmp_path / "out"),
        config=dict(sha256="rv"),
        source_manifest=dict(sha256="source"),
    )
    config = dict(limits=limits, task={}, physics_threads=8, p1_config_sha256="science")
    inputs = dict(previous_prepared_manifest={}, origin_replay={}, episodes=episodes)
    monkeypatch.setattr(study, "verify_launch", lambda _: (launch, config, inputs))
    env = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)), bad=False)

    def construct(job, work):
        env.work = work
        return work.call("fake_environment", dict(environment_constructions=1), lambda: env)

    def restore(target, value, work):
        work.call("fake_restore", dict(restores=1, external_mj_forward_calls=1), lambda: None)
        target.context = {
            key: np.array(value[key], copy=True) for key in ("state", "warmstart", "integration")
        }

    def execute(target, work, path, horizon, *, actions):
        raw = physical_arrays(target.context["state"], actions)
        work.call(
            "fake_controls",
            dict(native_controls=horizon, physics_substeps=4 * horizon),
            lambda: None,
        )
        with h5py.File(path, "x") as data:
            for key, value in raw.items():
                data[key] = value
        return native.trace_outcome(raw["states"], raw["costs"], horizon, False)

    def batch(initial, controls, **kwargs):
        env.work.call(
            "fake_forecast",
            dict(model_controls=1000, physics_substeps=4000, model_queries=1),
            lambda: None,
        )
        states = physical_arrays(initial, controls[0])["states"][None, 1:]
        if env.bad:
            states[0, 10, 3] += 1e-5
        return states, np.zeros(1), None

    env.batch_rollout = batch
    env.close = lambda: None
    monkeypatch.setattr(native, "_env", construct)
    monkeypatch.setattr(native, "_restore", restore)
    monkeypatch.setattr(native, "execute_trace", execute)
    monkeypatch.setattr(
        runtime,
        "snapshot",
        lambda target: {key: np.array(value, copy=True) for key, value in target.context.items()},
    )
    monkeypatch.setattr(
        mujoco,
        "mj_saveModel",
        lambda model, filename, buffer: Path(filename).write_bytes(b"fake compiled model"),
    )
    return tmp_path, env


def test_complete_fixed_roster_and_context_proofs(synthetic):
    root, env = synthetic
    result = study.run_validation(root / "launch.json")
    assert result["status"] == "PASS"
    assert result["work"]["completed_native_controls"] == 4000
    assert result["work"]["completed_model_controls"] == 2000
    assert result["work"]["completed_physics_substeps"] == 24000
    assert len(result["episodes"]) == 2
    with np.load(root / "out/candidate_0.npz") as raw:
        assert np.array_equal(raw["initial_state"], raw["final_state"])
    assert (root / "out/launch.json").exists()


def test_first_candidate_failure_stops_without_second_source(synthetic):
    root, env = synthetic
    env.bad = True
    result = study.run_validation(root / "launch.json")
    assert result["status"] == "FAIL_STOPPED"
    assert result["work"]["completed_native_controls"] == 2000
    assert result["work"]["completed_model_controls"] == 1000
    assert len(result["episodes"]) == 1
    assert not (root / "out/capture_1.h5").exists()


def test_antecedent_counters_add_once_after_p0(tmp_path):
    import json

    from src.gps.mppi_gps.trajectory_pi_study import StudyWork, binding

    cfg = json.loads(
        (Path(__file__).parents[1] / "configs/studies/trajectory_pi_p1_1.json").read_text()
    )
    r0 = dict(requested_native_controls=1000, completed_native_controls=1000)
    rv = dict(
        requested_native_controls=4000,
        completed_native_controls=4000,
        requested_model_controls=2000,
        completed_model_controls=2000,
    )
    (tmp_path / "r0.json").write_text(json.dumps(dict(work=r0)))
    (tmp_path / "rv.json").write_text(json.dumps(dict(counts=rv)))
    p0 = dict(
        requested_native_controls=16,
        completed_native_controls=16,
        derived_cost_component_rows=1024000,
    )
    meter = StudyWork(
        tmp_path / "out",
        cfg,
        dict(work=p0),
        antecedents=[
            dict(phase="r0", receipt=binding(tmp_path / "r0.json"), counts=r0),
            dict(phase="rv", receipt=binding(tmp_path / "rv.json"), counts=rv),
        ],
    )
    assert meter.work.counts["completed_native_controls"] == 5016
    assert meter.work.counts["completed_model_controls"] == 2000
    assert meter.work.counts["derived_cost_component_rows"] == 1024000
    meter.work.close("FAILED")
    rows = [
        json.loads(line)
        for line in (tmp_path / "out/coordinator/ledger.jsonl").read_text().splitlines()
    ]
    assert rows[0]["event"] == "prerequisite_work_included" and rows[0]["counts"] == p0
    assert [r["phase"] for r in rows if r["event"] == "antecedent_work_included"] == ["r0", "rv"]


def test_outer_guard_charges_prior_active_resources(tmp_path, monkeypatch):
    import json
    import time

    from src.gps.mppi_gps import trajectory_pi_launch as launch

    cfg = json.loads(
        (Path(__file__).parents[1] / "configs/studies/trajectory_pi_p1_1.json").read_text()
    )
    monkeypatch.setattr(launch, "tree_usage", lambda _: dict(cpu_seconds=10.0, rss_bytes=1))
    guard = launch.Guard(
        tmp_path, cfg, antecedents=[dict(phase="r0", wall_seconds=14.0, cpu_seconds=10.0)]
    )
    assert time.monotonic() - guard.started >= 14
    assert guard.sample()["cpu_seconds"] == 10.0
