"""Exercise S3's continuous runner and frozen-policy provenance checks."""

import copy
import inspect
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from scripts.runners import run_walker2d as baseline
from scripts.runners import run_walker_shadow as shadow
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.rl.sac import SacActor
from src.utils.config import MPPIConfig
from src.utils.trace_replay import load_episode


class ConstantPolicy:
    def __init__(self, value):
        self.value = value

    def act(self, obs, deterministic=True):
        assert deterministic
        return np.full(6, self.value)


class FailingPolicy:
    def __init__(self, valid_queries=12):
        self.calls = 0
        self.valid_queries = valid_queries

    def act(self, obs, deterministic=True):
        self.calls += 1
        if self.calls > self.valid_queries:
            raise RuntimeError("deliberate policy failure")
        return np.zeros(6)


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shadow_reference")
    cfg = MPPIConfig(K=8, H=6, lam=3.0, noise_sigma=0.3, initial_iterations=2)
    config = directory / "mppi.json"
    config.write_text(json.dumps(asdict(cfg)))
    task = json.loads((shadow._ROOT / "configs/walker2d_task.json").read_text())
    task["nthread"] = 1
    task_config = directory / "task.json"
    task_config.write_text(json.dumps(task))
    traces = directory / "traces.npz"
    baseline.main(config=config, task_config=task_config, episodes=2, steps=12,
                  seed=43, planner_seed=71, traces_output=traces, log_every=0)
    return {"trace_path": traces, "reference": load_episode(traces, 1), "cfg": cfg,
            "config_path": config, "task": task, "task_path": task_config}


def _run(reference_run, policies=None, **kwargs):
    return shadow.run_shadow(reference_run["reference"], reference_run["task"],
                             reference_run["cfg"], policies or {"zero": ConstantPolicy(0.0)},
                             steps=12, **kwargs)


def test_shadow_runner_matches_every_reference_field_across_twelve_decisions(reference_run):
    trace, candidates, report = _run(reference_run, {
        "zero": ConstantPolicy(0.0), "positive": ConstantPolicy(0.2),
        "negative": ConstantPolicy(-0.2),
    })
    assert report["status"] == "complete"
    assert report["shadow_only"] and report["tau"] == 0.0
    assert report["completed_steps"] == 12
    assert report["reference_trace_exact"]
    for field in shadow.TRACE_FIELDS:
        np.testing.assert_array_equal(trace[field], reference_run["reference"][field])
        assert report["reference_fields"][field]["exact"]
    np.testing.assert_array_equal(trace["episode_step_offsets"], [0, 12])
    np.testing.assert_array_equal(trace["episode_state_offsets"], [0, 13])
    np.testing.assert_array_equal(trace["reset_seeds"], [44])
    np.testing.assert_array_equal(trace["planner_seeds"], [72])
    for label in candidates:
        assert candidates[label]["plans"].shape == (12, 6, 6)
        assert report["policies"][label]["decisions"] == 12
    for step in range(12):
        baseline_costs = [report["decisions"][label][step]["baseline_cost"] for label in candidates]
        assert len(set(baseline_costs)) == 1


def test_runner_saves_actual_post_call_shift_and_advanced_rng_for_future_branches(reference_run):
    trace, _, _ = _run(reference_run)
    np.testing.assert_array_equal(trace["post_plan_nominals"][:, :-1], trace["plans"][:, 1:])
    np.testing.assert_array_equal(trace["post_plan_nominals"][:, -1], trace["plans"][:, -1])
    np.testing.assert_array_equal(trace["post_plan_nominals"][:-1], trace["pre_plan_nominals"][1:])
    np.testing.assert_array_equal(trace["post_planner_rng_states"][:-1], trace["planner_rng_states"][1:])
    assert not trace["post_initial_plan_flags"].any()
    assert not np.any(trace["post_planner_rng_states"] == trace["planner_rng_states"])
    # Independently restore a post-call snapshot, execute its just-planned
    # baseline action, then reproduce the *next* decision exactly.
    env = Walker2d(**reference_run["task"])
    controller = MPPI(env, reference_run["cfg"])
    try:
        step = 5
        env.reset(state=trace["states"][step])
        env.set_warmstart(trace["warmstarts"][step])
        controller.restore({"U": trace["post_plan_nominals"][step],
                            "rng_state": json.loads(str(trace["post_planner_rng_states"][step])),
                            "initial_plan": bool(trace["post_initial_plan_flags"][step])})
        env.step(trace["actions"][step])
        action, _ = controller.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
        np.testing.assert_array_equal(action, trace["actions"][step + 1])
        np.testing.assert_array_equal(controller.last_plan, trace["plans"][step + 1])
    finally:
        env.close()


def test_shadow_does_not_restore_tampered_reference_decisions(reference_run):
    modified = copy.deepcopy(reference_run)
    modified["reference"]["pre_plan_nominals"][3, 0, 0] += 0.2
    trace, _, report = _run(modified)
    np.testing.assert_array_equal(trace["actions"], reference_run["reference"]["actions"])
    np.testing.assert_array_equal(trace["pre_plan_nominals"], reference_run["reference"]["pre_plan_nominals"])
    assert not report["reference_trace_exact"]
    assert report["status"] == "parity_failed"
    assert report["reference_fields"]["pre_plan_nominals"]["first_divergence"] == 3


def test_programming_failure_reports_partial_evidence_and_stops(reference_run):
    callbacks = []
    with pytest.raises(RuntimeError, match="deliberate policy failure"):
        _run(reference_run, {"first": ConstantPolicy(0.0), "bad": FailingPolicy()},
             on_progress=lambda *args: callbacks.append(args))
    trace, candidates, report = callbacks[-1]
    assert report["status"] == "failed"
    assert report["completed_steps"] == 2
    assert "deliberate policy failure" in report["error"]
    assert len(trace["states"]) == 3
    assert len(trace["actions"]) == len(candidates["bad"]["plans"]) == 2
    assert len(candidates["first"]["plans"]) == len(report["decisions"]["first"]) == 2
    assert report["pending_decision"]["step"] == 2
    assert set(report["pending_decision"]["decisions"]) == {"first"}
    np.testing.assert_array_equal(report["pending_decision"]["state"], trace["states"][-1])
    np.testing.assert_array_equal(trace["actions"], reference_run["reference"]["actions"][:2])


def test_progress_snapshots_own_their_arrays_and_nested_decision_records(reference_run, monkeypatch):
    class StationaryWalker(Walker2d):
        # Keep this callback-ownership test alive for 101 controls cheaply.
        # Actual physics parity is tested separately above, not with this stub.
        def step(self, action):
            self.data.time += self._dt
            return self._get_obs(), 0.0, False, {}

    monkeypatch.setattr(shadow, "Walker2d", StationaryWalker)
    callbacks = []

    def progress(trace, candidates, report):
        callbacks.append((trace, candidates, report))
        if report["status"] == "running":
            trace["actions"][0, 0] = 999.0
            candidates["zero"]["plans"][0, 0, 0] = 999.0
            report["decisions"]["zero"][0]["baseline_cost"] = -999.0

    trace, candidates, report = shadow.run_shadow(
        reference_run["reference"], reference_run["task"], reference_run["cfg"],
        {"zero": ConstantPolicy(0.0)}, steps=101, on_progress=progress,
    )
    assert len(callbacks) == 2
    assert callbacks[0][2]["completed_steps"] == 100
    assert len(callbacks[0][2]["decisions"]["zero"]) == 100
    assert report["completed_steps"] == len(report["decisions"]["zero"]) == 101
    assert trace["actions"][0, 0] != 999.0
    assert candidates["zero"]["plans"][0, 0, 0] != 999.0
    assert report["decisions"]["zero"][0]["baseline_cost"] != -999.0


def test_shadow_command_has_no_active_execution_argument():
    assert not {"active", "mode", "execute_policy"} & inspect.signature(shadow.main).parameters.keys()


@pytest.fixture
def selection(tmp_path, reference_run):
    # Tiny, untrained actors isolate provenance plumbing from any S2 run files.
    with torch.random.fork_rng():
        torch.manual_seed(8)
        actor = SacActor(17, -np.ones(6), np.ones(6), hidden_dim=8)
    policy_state = {"format_version": 1, "actor": actor.state_dict(), "obs_dim": 17,
                    "hidden_dim": 8, "observation_transform": "identity",
                    "action_low": [-1.0] * 6, "action_high": [1.0] * 6}
    sources = {relative: shadow.sha256(shadow._ROOT / relative)
               for relative in shadow.PHYSICS_SOURCES}
    task_hash = shadow.sha256(reference_run["task_path"])
    manifest = {"task_sha256": task_hash, "obs_transform": "identity", "checkpoints": {}}
    for label, step in (("weak", 0), ("intermediate", 200000), ("strong", 800000)):
        checkpoint = tmp_path / f"{label}.pt"
        torch.save({"policy": policy_state, "task_sha256": task_hash, "step": step,
                    "source_sha256": sources}, checkpoint)
        manifest["checkpoints"][label] = {"checkpoint": str(checkpoint), "step": step,
                                          "sha256": shadow.sha256(checkpoint)}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_selection_loads_exact_frozen_policies_without_consuming_torch_rng(selection, reference_run):
    path, manifest = selection
    before = torch.random.get_rng_state()
    policies, loaded = shadow.load_selection(path, reference_run["task_path"])
    assert loaded == manifest
    assert set(policies) == {"weak", "intermediate", "strong"}
    assert torch.equal(before, torch.random.get_rng_state())
    for actor in policies.values():
        assert not actor.training
        assert all(not parameter.requires_grad for parameter in actor.parameters())
        assert np.isfinite(actor.act(np.zeros(17))).all()


@pytest.mark.parametrize("problem", ["task", "transform", "missing_category", "hash", "step", "physics"])
def test_selection_rejects_mismatched_identity_or_provenance(selection, reference_run, problem):
    path, manifest = selection
    if problem == "task":
        manifest["task_sha256"] = "0" * 64
    elif problem == "transform":
        manifest["obs_transform"] = "running_normalizer"
    elif problem == "missing_category":
        del manifest["checkpoints"]["strong"]
    elif problem == "hash":
        manifest["checkpoints"]["weak"]["sha256"] = "0" * 64
    elif problem == "step":
        manifest["checkpoints"]["weak"]["step"] = 123
    else:
        checkpoint = manifest["checkpoints"]["weak"]["checkpoint"]
        payload = torch.load(checkpoint, weights_only=True)
        payload["source_sha256"]["assets/walker2d.xml"] = "0" * 64
        torch.save(payload, checkpoint)
        manifest["checkpoints"]["weak"]["sha256"] = shadow.sha256(type(path)(checkpoint))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        shadow.load_selection(path, reference_run["task_path"])


def test_command_saves_auditable_outputs_and_refuses_nonempty_directory(selection, reference_run, tmp_path):
    selection_path, _ = selection
    output = tmp_path / "shadow"
    arguments = {"reference": reference_run["trace_path"], "reference_episode": 1,
                 "config": reference_run["config_path"], "task_config": reference_run["task_path"],
                 "policy_selection": selection_path, "steps": 12, "out_dir": output}
    shadow.main(**arguments)
    report = json.loads((output / "report.json").read_text())
    assert report["reference_trace_exact"] and report["status"] == "complete"
    saved = load_episode(output / "traces.npz", 0)
    for field in shadow.TRACE_FIELDS:
        np.testing.assert_array_equal(saved[field], reference_run["reference"][field])
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert all(shadow.sha256(output / relative) == expected for relative, expected in hashes.items())
    assert "source/src/mppi/gate.py" in hashes
    for label in ("weak", "intermediate", "strong"):
        assert len(json.loads((output / label / "decisions.json").read_text())["decisions"]) == 12
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        shadow.main(**arguments)
    assert (output / "report.json").read_bytes() == before


def test_command_returns_nonzero_after_saving_a_mismatch_report(selection, reference_run, tmp_path):
    with np.load(reference_run["trace_path"], allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    # Episode 1 begins at action 12, whereas its initial state is at state 13.
    arrays["actions"][15, 0] += 1e-9
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)
    output = tmp_path / "mismatch"
    with pytest.raises(SystemExit) as failure:
        shadow.main(tampered, output, reference_episode=1, steps=12,
                    config=reference_run["config_path"], task_config=reference_run["task_path"],
                    policy_selection=selection[0])
    assert failure.value.code == 1
    report = json.loads((output / "report.json").read_text())
    assert not report["reference_trace_exact"]
    assert report["status"] == "parity_failed"
    assert report["reference_fields"]["actions"]["first_divergence"] == 3
    assert (output / "artifact_hashes.json").is_file()


def test_command_saves_partial_failed_run_before_propagating_error(
    selection, reference_run, tmp_path, monkeypatch
):
    path, manifest = selection
    monkeypatch.setattr(shadow, "load_selection", lambda *_: ({"weak": FailingPolicy()}, manifest))
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="deliberate policy failure"):
        shadow.main(reference_run["trace_path"], output, reference_episode=1, steps=12,
                    config=reference_run["config_path"], task_config=reference_run["task_path"],
                    policy_selection=path)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["completed_steps"] == 2
    assert report["pending_decision"]["step"] == 2
    assert json.loads((output / "status.json").read_text()) == report
    saved = load_episode(output / "traces.npz", 0)
    np.testing.assert_array_equal(saved["actions"], reference_run["reference"]["actions"][:2])
    assert len(json.loads((output / "weak/decisions.json").read_text())["decisions"]) == 2
    assert (output / "manifest.json").is_file()


@pytest.mark.parametrize("problem", ["missing_snapshot", "no_policies", "zero_steps"])
def test_runner_rejects_incomplete_or_empty_protocol_before_simulation(reference_run, problem):
    reference = dict(reference_run["reference"])
    policies, steps = {"zero": ConstantPolicy(0.0)}, 12
    if problem == "missing_snapshot":
        del reference["planner_rng_states"]
    elif problem == "no_policies":
        policies = {}
    else:
        steps = 0
    with pytest.raises(ValueError):
        shadow.run_shadow(reference, reference_run["task"], reference_run["cfg"], policies, steps)
