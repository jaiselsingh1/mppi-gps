"""Synthetic interrupted-prefix lifecycle; no native physics, network or fit."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps import local_coupling_prefix_resume as resume
from src.gps.mppi_gps import trajectory_pi_native as native
from src.gps.mppi_gps.provenance import write
from src.mppi.mppi import MPPI


def _put_trace(path, arrays):
    with h5py.File(path, "x") as target:
        for name, value in arrays.items():
            target.create_dataset(name, data=value, maxshape=(None, *value.shape[1:]), chunks=True)


def _arrays():
    states = np.zeros((301, 19))
    states[:, 0] = np.arange(301)
    states[:, 2] = 1.2
    states[:, 10] = 1.5
    return dict(
        states=states,
        warmstarts=np.broadcast_to(np.arange(301)[:, None] * 0.01, (301, 9)).copy(),
        integrations=np.broadcast_to(np.arange(301)[:, None] * 0.02, (301, 91)).copy(),
        observations=resume.observations(states),
        actions=np.broadcast_to((np.arange(300)[:, None] + 1) / 1000, (300, 6)).copy(),
        costs=np.arange(300) / 1000 + 0.01,
        native_fall_cost=np.zeros(300),
    )


def _bank(group, index, before, after, rng_before, rng_after, arrays, actor_hash="fixed-actor"):
    control, inner = (0, index) if index < 2 else (index - 1, 0)
    for name, value in dict(
        U_before=before,
        U_after=after,
        initial_state=arrays["states"][control],
        initial_warmstart=arrays["warmstarts"][control],
        initial_integration=arrays["integrations"][control],
        C=np.arange(256, dtype=float),
        D=np.zeros(256),
    ).items():
        group.create_dataset(name, data=value)
    group.attrs.update(
        control=control,
        inner=inner,
        alpha=0.0,
        actor_hash=actor_hash,
        rng_before=json.dumps(rng_before, sort_keys=True),
        rng_after=json.dumps(rng_after, sort_keys=True),
    )


def make_case(tmp_path, stream=1):
    slot = 0 if stream == 1 else 1
    count = resume.PREFIX_CONTROLS[stream, slot]
    original = tmp_path / "old"
    original.mkdir()
    arrays = _arrays()
    reference = tmp_path / "r1.h5"
    _put_trace(reference, arrays)
    prefix_arrays = {
        name: value[: count + 1 if name in resume.TRACE_FIELDS[:4] else count]
        for name, value in arrays.items()
    }
    _put_trace(original / "trace.h5", prefix_arrays)
    for name in ("checkpoint.pt", "config.json"):
        (tmp_path / name).write_text("synthetic immutable input\n")
    initial = {
        name: arrays[key][0]
        for name, key in (
            ("state", "states"),
            ("warmstart", "warmstarts"),
            ("integration", "integrations"),
        )
    }
    initial.update(spec=np.array(8191), dimensions=np.array([9, 9, 6]))
    np.savez(
        tmp_path / "restarts.npz", **{f"r{slot}_{name}": value for name, value in initial.items()}
    )
    seed = 2961000100 + stream
    rng = np.random.default_rng(seed)
    plan = np.zeros((48, 6))
    full_names = []
    with h5py.File(original / "banks.h5", "x") as banks:
        banks.attrs["synthetic_source"] = "unchanged"
        for index in range(count + 1):
            control, inner = (0, index) if index < 2 else (index - 1, 0)
            before = deepcopy(rng.bit_generator.state)
            rng.standard_normal((256, 48, 6))
            after = np.full((48, 6), 0.0005 if index == 0 else (control + 1) / 1000)
            group = banks.create_group(f"{index:06d}")
            _bank(group, index, plan, after, before, rng.bit_generator.state, arrays)
            if control == 0 or (control == 149 and inner == 0):
                name = f"full_t{control:03d}_p{inner}.npz"
                group.attrs["full_bank"] = name
                full_names.append(name)
                np.savez_compressed(
                    original / name,
                    actions=np.zeros((256, 48, 6)),
                    posts=np.zeros((256, 48, 19)),
                    C=np.arange(256, dtype=float),
                    D=np.zeros(256),
                )
            plan = after.copy() if index == 0 else resume._shift(after)
    context = {
        name: arrays[key][count]
        for name, key in (
            ("state", "states"),
            ("warmstart", "warmstarts"),
            ("integration", "integrations"),
        )
    }
    context.update(
        spec=initial["spec"],
        dimensions=initial["dimensions"],
        U_carried_after_one_shift=plan,
        control=np.array(count),
        bank_count=np.array(count + 1),
        initial_plan=np.array(False),
    )
    np.savez(tmp_path / "resume.npz", **context)
    rows = []

    def event(operation, charges, completed=True):
        index = len(rows)
        rows.append(
            dict(
                id=index,
                event="requested",
                operation=operation,
                charges=charges,
                wall_seconds=index / 1000,
            )
        )
        if completed:
            rows.append(
                dict(
                    id=index + 1,
                    event="completed",
                    request_id=index,
                    operation=operation,
                    charges=charges,
                    wall_seconds=(index + 1) / 1000,
                )
            )

    event("load_actor", dict(checkpoint_loads=1, policy_constructions=1))
    event("environment", dict(environment_constructions=1))
    event("restore", dict(restores=1, external_mj_forward_calls=1))
    for index in range(count + 1):
        event(
            "native_forecast", dict(model_controls=12288, model_queries=1, physics_substeps=49152)
        )
        event("neural_rows", dict(neural_rows=100))
        if index > 0:
            event("native_control", dict(native_controls=1, physics_substeps=4))
    event(
        "native_forecast",
        dict(model_controls=12288, model_queries=1, physics_substeps=49152),
        completed=False,
    )
    (original / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    limits = dict(
        wall_seconds=60,
        output_bytes=128 * 1024**2,
        memory_bytes=2 * 1024**3,
        model_controls=10**9,
        model_queries=10**6,
        native_controls=1000,
        physics_substeps=4 * 10**9,
        neural_rows=10**9,
        checkpoint_loads=10,
        policy_constructions=10,
        environment_constructions=10,
        restores=10,
        external_mj_forward_calls=10,
    )
    job = dict(
        out=str(original),
        round=4,
        stream=stream,
        arm="independent",
        slot=slot,
        alpha=0.0,
        seed=seed,
        checkpoint=resume.binding(tmp_path / "checkpoint.pt"),
        restarts=resume.binding(tmp_path / "restarts.npz"),
        controls=300,
        planner=dict(H=48, K=256, initial_iterations=2, noise_sigma=0.3, lam=3.0),
        task=dict(frame_skip=4),
        physics_threads=4,
        replay_episode=slot == 0,
        config=resume.binding(tmp_path / "config.json"),
        limits=limits,
    )
    write(original / "job.json", job)
    job.update(
        out=str(tmp_path / "new"),
        prefix_job=resume.binding(original / "job.json"),
        prefix_trace=resume.binding(original / "trace.h5"),
        prefix_banks=resume.binding(original / "banks.h5"),
        prefix_ledger=resume.binding(original / "ledger.jsonl"),
        resume_context=resume.binding(tmp_path / "resume.npz"),
        reference_R1_trace=resume.binding(reference),
        prefix_full_banks=[resume.binding(original / name) for name in full_names],
        resume_rng_state=deepcopy(rng.bit_generator.state),
    )
    return job, arrays


def install_fake_execution(monkeypatch, arrays, expected):
    acquired = []
    native_controls = []
    restores = []

    class Actor:
        def hash(self):
            return "fixed-actor"

    class Env:
        def __init__(self, work):
            self.work = work
            self.index = 0

        def get_state(self):
            return arrays["states"][self.index].copy()

        def get_warmstart(self):
            return arrays["warmstarts"][self.index].copy()

        def _get_obs(self):
            return arrays["observations"][self.index].copy()

        def step(self, action):
            def execute():
                index = self.index
                np.testing.assert_array_equal(action, arrays["actions"][index])
                native_controls.append(index)
                self.index += 1
                return None, arrays["costs"][index], False, dict(fall_cost=0.0)

            return self.work.call(
                "native_control", dict(native_controls=1, physics_substeps=4), execute
            )

        def close(self):
            pass

    class Planner:
        restore = MPPI.restore

        def __init__(self, env, cfg, *, seed, actor, alpha, work, directory):
            self.env, self.work = env, work
            self.H, self.nu = 48, 6
            self._rng = np.random.default_rng(seed)
            self.stats = h5py.File(directory / "banks.h5", "x")
            self.actor_hash = actor.hash()
            self.native_checks = []

        def penalty(self, state, posts, actions):
            raise AssertionError("Synthetic plan never calls a neural network")

        def plan_step(self, state, prior_cost, initial_warmstart):
            control = self.control
            acquired.append(control)
            assert self.inner == 0
            if len(acquired) == 1:
                np.testing.assert_array_equal(self.U, expected["planner_state"]["U"])
                assert self._rng.bit_generator.state == expected["planner_state"]["rng_state"]
                assert self._initial_plan is False
            np.testing.assert_array_equal(state, arrays["states"][control])
            np.testing.assert_array_equal(initial_warmstart, arrays["warmstarts"][control])
            before = deepcopy(self._rng.bit_generator.state)
            self.work.call(
                "native_forecast",
                dict(model_controls=12288, model_queries=1, physics_substeps=49152),
                lambda: None,
            )
            self._rng.standard_normal((256, 48, 6))
            after = np.full((48, 6), (control + 1) / 1000)
            group = self.stats.create_group(f"{self.bank_count:06d}")
            _bank(
                group, self.bank_count, self.U, after, before, self._rng.bit_generator.state, arrays
            )
            self.U = resume._shift(after)
            self.bank_count += 1
            return arrays["actions"][control].copy(), {}

    def restore(env, context, work):
        def apply():
            env.index = int(context["state"][0])
            for name, key in (
                ("state", "states"),
                ("warmstart", "warmstarts"),
                ("integration", "integrations"),
            ):
                np.testing.assert_array_equal(context[name], arrays[key][env.index])
            restores.append(env.index)

        work.call("restore", dict(restores=1, external_mj_forward_calls=1), apply)

    monkeypatch.setattr(
        resume,
        "_load_actor",
        lambda job, work: work.call(
            "load_actor", dict(checkpoint_loads=1, policy_constructions=1), Actor
        ),
    )
    monkeypatch.setattr(
        resume,
        "_env",
        lambda job, work: work.call(
            "environment", dict(environment_constructions=1), lambda: Env(work)
        ),
    )
    monkeypatch.setattr(resume, "_restore", restore)
    monkeypatch.setattr(resume, "RecordedMPPI", Planner)
    monkeypatch.setattr(
        resume,
        "verify_native",
        lambda env, work, directory, checks, actor: [dict(status="SYNTHETIC_ONLY")],
    )
    monkeypatch.setattr(
        native,
        "snapshot",
        lambda env: dict(
            state=env.get_state(),
            warmstart=env.get_warmstart(),
            integration=arrays["integrations"][env.index].copy(),
        ),
    )
    return acquired, native_controls, restores


@pytest.mark.parametrize("stream", [1, 2])
def test_exact_suffix_only_lifecycle_and_logical_ledger(tmp_path, monkeypatch, stream):
    job, arrays = make_case(tmp_path, stream)
    expected = resume.validate_prefix(job)
    original_bindings = [job[name] for name in resume.INPUT_FIELDS] + job["prefix_full_banks"]
    acquired, native_controls, restores = install_fake_execution(monkeypatch, arrays, expected)
    result = resume.resume_episode(job)
    assert result["result"]["status"] == "COMPLETED", result["result"]
    count = expected["controls"]
    assert acquired == list(range(count, 300))
    assert native_controls == list(range(count, 300)) + (
        list(range(300)) if job["replay_episode"] else []
    )
    assert restores == [count] + ([0] if job["replay_episode"] else [])
    assert result["result"]["banks"] == 301
    raw = native.load_trace(Path(job["out"]) / "trace.h5")
    for name in resume.TRACE_FIELDS:
        np.testing.assert_array_equal(raw[name], arrays[name])
    for value in original_bindings:
        assert resume.binding(Path(value["path"])) == value
    with (
        h5py.File(job["prefix_banks"]["path"], "r") as old,
        h5py.File(Path(job["out"]) / "banks.h5", "r") as new,
    ):
        resume._check_h5_prefix(old, new)
    receipt = result["receipt"]
    suffix = json.loads(Path(result["raw_suffix_receipt"]["path"]).read_text())
    assert receipt["counts"]["requested_native_controls"] == 300 + (
        300 if job["replay_episode"] else 0
    )
    assert receipt["counts"]["completed_model_queries"] == 301
    assert suffix["counts"]["completed_model_queries"] == 300 - count
    assert suffix["counts"]["completed_native_controls"] == len(native_controls)
    assert receipt["cpu_seconds"] == suffix["cpu_seconds"]
    assert receipt["prefix_cpu_seconds"] is None
    excluded = receipt["excluded_pending_requests"]
    assert len(excluded) == 1 and excluded[0]["charges"]["model_controls"] == 12288
    rows = [
        json.loads(line) for line in (Path(job["out"]) / "ledger.jsonl").read_text().splitlines()
    ]
    pending = {}
    counts = {}
    for index, row in enumerate(rows):
        assert row["id"] == index
        if row["event"] == "requested":
            pending[index] = row
            resume._add(counts, row["charges"], "requested_")
        elif row["event"] == "completed":
            request = pending.pop(row["request_id"])
            assert request["operation"] == row["operation"] and request["charges"] == row["charges"]
            resume._add(counts, row["charges"], "completed_")
    assert not pending and counts == receipt["counts"]
    assert rows[-1]["event"] == "closed" and rows[-1]["receipt"] == receipt
    assert result["result"]["continuation"]["prefix_banks"] == job["prefix_banks"]
    assert result["result"]["continuation"]["prefix_banks_count"] == count + 1
    with pytest.raises(FileExistsError):
        resume.resume_episode(job)


@pytest.mark.parametrize("corruption", ["state", "plan", "rng", "metadata", "binding"])
def test_rejects_corrupted_handoff_before_any_actor_or_physics(tmp_path, monkeypatch, corruption):
    job, _ = make_case(tmp_path)
    if corruption in ("state", "plan"):
        path = Path(job["resume_context"]["path"])
        with np.load(path) as raw:
            values = {key: raw[key].copy() for key in raw.files}
        field = "state" if corruption == "state" else "U_carried_after_one_shift"
        values[field].flat[0] += 0.01
        np.savez(path, **values)
        job["resume_context"] = resume.binding(path)
    elif corruption == "rng":
        job["resume_rng_state"]["state"]["state"] += 1
    elif corruption == "metadata":
        job["planner"]["noise_sigma"] = 0.4
    else:
        job["prefix_banks"]["sha256"] = "bad"
    monkeypatch.setattr(
        resume,
        "_load_actor",
        lambda *args: pytest.fail("No actor allowed before handoff validation"),
    )
    monkeypatch.setattr(
        resume, "_env", lambda *args: pytest.fail("No physics allowed before handoff validation")
    )
    with pytest.raises(ValueError):
        resume.validate_prefix(job)
    result = resume.resume_episode(job)
    assert result["result"]["status"] == "FAILED_STOPPED"
    assert result["receipt"]["counts"] == {}


def test_suffix_budget_stop_preserves_raw_receipt_and_unknown_pending(tmp_path, monkeypatch):
    job, arrays = make_case(tmp_path)
    expected = resume.validate_prefix(job)
    acquired, native_controls, _ = install_fake_execution(monkeypatch, arrays, expected)
    job["limits"]["model_controls"] = 0
    result = resume.resume_episode(job)
    assert result["result"]["status"] == "FAILED_STOPPED"
    assert "Work budget exhausted" in result["result"]["error"]
    assert acquired == [198] and native_controls == []
    assert len(result["excluded_pending_requests"]) == 1
    assert Path(result["raw_suffix_receipt"]["path"]).is_file()
    assert result["receipt"]["counts"]["completed_native_controls"] == 198
    raw = native.load_trace(Path(job["out"]) / "trace.h5")
    assert len(raw["actions"]) == 198


def test_ledger_does_not_discard_resolved_failures_or_multiple_unknowns(tmp_path):
    job, _ = make_case(tmp_path)
    path = Path(job["prefix_ledger"]["path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["operation"] = "neural_rows"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Only the final interrupted"):
        resume.completed_prefix_ledger(path, 198)
