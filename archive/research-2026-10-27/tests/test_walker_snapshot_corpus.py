"""Synthetic archive extraction only: no environment import or model query."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from src.gps import walker_snapshot_corpus as corpus
from src.gps.walker_owned_credit import QUALIFIED_PLANNER_SIGNATURE


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    task = tmp_path / "task.json"
    model = tmp_path / "model.xml"
    shutil.copy2(ROOT / "configs/walker2d_task.json", task)
    shutil.copy2(ROOT / "assets/walker2d.xml", model)
    config = tmp_path / "qualified.json"
    config.write_text(json.dumps(dict(QUALIFIED_PLANNER_SIGNATURE)))
    lengths = [1000, 1000, 600]
    resets, planners = [30, 35, 31], [10030, 10035, 10031]
    states, warms, episodes = [], [], []
    for index, (length, reset, planner) in enumerate(zip(lengths, resets, planners)):
        state = np.zeros((length + 1, 19), dtype=np.float64)
        state[:, 0] = np.arange(length + 1) * .008
        state[:, 1] = reset + np.arange(length + 1) / 1000.
        state[:, 2] = 1.25
        if index == 2:
            state[-1, 2] = .7
        states.append(state)
        warms.append(np.repeat((np.arange(length + 1)[:, None] + reset) / 1000., 9, axis=1))
        episodes.append({"reset_seed": reset, "planner_seed": planner, "steps": length,
                         "survived": index != 2})
    arrays = {"states": np.concatenate(states), "warmstarts": np.concatenate(warms),
        "actions": np.zeros((sum(lengths), 6)), "costs": np.zeros(sum(lengths)),
        "episode_step_offsets": np.cumsum([0] + lengths),
        "episode_state_offsets": np.cumsum([0] + [length + 1 for length in lengths]),
        "reset_seeds": np.array(resets), "planner_seeds": np.array(planners)}
    npz = tmp_path / "stream_a.npz"
    np.savez_compressed(npz, **arrays)
    report = {"task": {**json.loads(task.read_text()), "max_steps": 1000, "velocity_cost_weight": 2.},
        "mppi_config": dict(QUALIFIED_PLANNER_SIGNATURE), "episodes": episodes,
        "run": {"requested_episodes": 3, "complete": True, "elapsed_seconds": 1.5},
        "provenance": {"task_config_sha256": _sha(task), "config_sha256": _sha(config),
            "config_path": str(config), "packages": {"mujoco": "3.6.0", "numpy": "synthetic", "scipy": "synthetic"},
            "control_dt": .008, "physics_dt": .002, "frame_skip": 4, "backend": "mujoco_cpu", "nthread": 18,
            "source_sha256": {relative: _sha(ROOT / relative) for relative in
                ("src/envs/walker2d.py", "src/envs/mujoco_env.py", "src/mppi/mppi.py",
                 "src/utils/config.py", "assets/walker2d.xml")}}}
    metadata = tmp_path / "stream_a.json"
    metadata.write_text(json.dumps(report))
    return {"pair": {"npz": npz, "json": metadata, "source_root": ROOT}, "task": task,
            "model": model, "arrays": arrays, "report": report, "directory": tmp_path}


def _load(bundle, pairs=None):
    return corpus.load_corpus(pairs or [bundle["pair"]], bundle["task"], bundle["model"],
                              train_reset_seeds=(30, 31))


def test_every_fixed_episode_time_slot_is_retained_without_teacher_labels(bundle):
    result = _load(bundle)
    assert result["counts"]["context_slots"] == 9
    assert result["counts"]["active_contexts"] == 8
    assert result["counts"]["inactive_contexts"] == 1
    assert result["counts"]["source_episodes"] == 3
    assert result["counts"]["new_simulated_controls"] == 0
    assert result["counts"]["new_actor_inference_rows"] == 0
    assert result["metadata"]["fresh_current_qualification_established"] is False
    assert result["metadata"]["teacher_action_or_return_labels_exported"] is False
    assert result["metadata"]["all_historical_source_identity_certified"] is False
    runner = result["metadata"]["inputs"][0]["historical_runner_identity"]
    assert runner["expected_sha256"] is None and runner["matches"] is None
    missing = next(row for row in result["contexts"] if row["ancestry"]["reset_seed"] == 31
                   and row["ancestry"]["handover"] == 900)
    assert not missing["active"] and not missing["healthy"]
    assert missing["state"] is None and missing["warmstart"] is None
    assert missing["remaining"] == 100 and missing["source_array_index"] is None
    assert missing["fall_before_handover"] and missing["source_episode"]["steps"] == 600
    earlier = next(row for row in result["contexts"] if row["ancestry"]["reset_seed"] == 31
                   and row["ancestry"]["handover"] == 500)
    assert earlier["active"] and earlier["healthy"] and earlier["remaining"] == 500
    for row in result["contexts"]:
        assert not {"actions", "teacher_actions", "costs", "cost", "returns", "return", "targets"} & set(row)
        if row["active"]:
            np.testing.assert_array_equal(row["state"], bundle["arrays"]["states"][row["source_array_index"]])
            np.testing.assert_array_equal(row["warmstart"], bundle["arrays"]["warmstarts"][row["source_array_index"]])
            with pytest.raises(ValueError):
                row["state"].setflags(write=True)
            with pytest.raises(ValueError):
                row["warmstart"].setflags(write=True)
    work = result["metadata"]["historical_work"]
    assert work["executed_controls"] == 2600
    assert work["candidate_slots_reconstructed"] == (2600 + 3) * 256 * 48
    assert not work["durable_per_attempt_ledger_available"]


def test_entire_reset_ancestry_stays_in_one_split_across_planner_streams(bundle):
    other_arrays = dict(bundle["arrays"])
    other_arrays["planner_seeds"] = other_arrays["planner_seeds"] + 10000
    npz = bundle["directory"] / "stream_b.npz"
    np.savez_compressed(npz, **other_arrays)
    report = deepcopy(bundle["report"])
    for row in report["episodes"]:
        row["planner_seed"] += 10000
    metadata = bundle["directory"] / "stream_b.json"
    metadata.write_text(json.dumps(report))
    result = _load(bundle, [bundle["pair"], {"npz": npz, "json": metadata, "source_root": ROOT}])
    assert result["counts"]["context_slots"] == 18
    assert result["counts"]["train_contexts"] == 12
    assert result["counts"]["heldout_contexts"] == 6
    by_root = {}
    for context in result["contexts"]:
        by_root.setdefault(context["ancestry"]["root_cluster"], set()).add(context["split"])
    assert by_root == {"reset30": {"train"}, "reset31": {"train"}, "reset35": {"heldout"}}


@pytest.mark.parametrize("change", ["task_hash", "model_hash", "control_dt", "mujoco", "report_K32"])
def test_identity_and_engine_metadata_mismatches_fail_closed(bundle, change):
    report = deepcopy(bundle["report"])
    if change == "task_hash":
        report["provenance"]["task_config_sha256"] = "wrong"
    elif change == "model_hash":
        report["provenance"]["source_sha256"]["assets/walker2d.xml"] = "wrong"
    elif change == "control_dt":
        report["provenance"]["control_dt"] = .01
    elif change == "mujoco":
        report["provenance"]["packages"]["mujoco"] = "unknown"
    else:
        report["mppi_config"]["K"] = 32
    bundle["pair"]["json"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        _load(bundle)


@pytest.mark.parametrize("change", ["warm_shape", "state_dtype", "offsets", "nonfinite", "early_unhealthy"])
def test_raw_trace_schema_and_first_fall_boundaries_fail_closed(bundle, change):
    arrays = dict(bundle["arrays"])
    if change == "warm_shape":
        arrays["warmstarts"] = arrays["warmstarts"][:, :8]
    elif change == "state_dtype":
        arrays["states"] = arrays["states"].astype(np.float32)
    elif change == "offsets":
        arrays["episode_state_offsets"] = arrays["episode_state_offsets"].copy()
        arrays["episode_state_offsets"][1] += 1
    elif change == "nonfinite":
        arrays["states"] = arrays["states"].copy()
        arrays["states"][1, 3] = np.nan
    else:
        arrays["states"] = arrays["states"].copy()
        arrays["states"][500, 2] = .7
    np.savez_compressed(bundle["pair"]["npz"], **arrays)
    with pytest.raises(ValueError):
        _load(bundle)


def test_duplicate_files_and_undeclared_absent_training_roots_are_rejected(bundle):
    with pytest.raises(ValueError, match="more than once"):
        _load(bundle, [bundle["pair"], bundle["pair"]])
    with pytest.raises(ValueError, match="absent"):
        corpus.load_corpus([bundle["pair"]], bundle["task"], bundle["model"], train_reset_seeds=(999,))


def test_mppi_source_difference_is_allowed_but_cpu_step_difference_is_not(bundle):
    archive_root = bundle["directory"] / "archive_source"
    report = deepcopy(bundle["report"])
    for relative in report["provenance"]["source_sha256"]:
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    mppi = archive_root / "src/mppi/mppi.py"
    mppi.write_text(mppi.read_text() + "\n# Synthetic historical planner/backend implementation difference.\n")
    report["provenance"]["source_sha256"]["src/mppi/mppi.py"] = _sha(mppi)
    bundle["pair"]["source_root"] = archive_root
    bundle["pair"]["json"].write_text(json.dumps(report))
    result = _load(bundle)
    assert "src/mppi/mppi.py" in result["metadata"]["inputs"][0]["source_compatibility"]["allowed_state_only_differences"]
    cpu = archive_root / "src/envs/mujoco_env.py"
    cpu.write_text(cpu.read_text().replace("self.data.ctrl[:] = action", "self.data.ctrl[:] = action * 0.5", 1))
    report["provenance"]["source_sha256"]["src/envs/mujoco_env.py"] = _sha(cpu)
    bundle["pair"]["json"].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="method differs"):
        _load(bundle)


def test_source_runner_mismatch_not_hidden_or_mistaken_for_full_independence_proof(bundle):
    report = deepcopy(bundle["report"])
    report["provenance"]["source_sha256"]["scripts/runners/run_walker2d.py"] = "0" * 64
    bundle["pair"]["json"].write_text(json.dumps(report))
    result = _load(bundle)
    runner = result["metadata"]["inputs"][0]["historical_runner_identity"]
    assert runner["expected_sha256"] == "0" * 64
    assert runner["actual_sha256"] == _sha(ROOT / "scripts/runners/run_walker2d.py")
    assert runner["matches"] is False
    assert runner["certifies_all_historical_independence"] is False
    assert result["metadata"]["all_historical_source_identity_certified"] is False
    assert result["metadata"]["independence_provenance_assumption"] == (
        "historical metadata and two instrumented replays; runner source not recovered yet")
    assert result["counts"]["context_slots"] == 9 and result["counts"]["active_contexts"] == 8


def _recovered_runner(bundle, *, guided=False):
    path = bundle["directory"] / "recovered_original_runner.py"
    keyword = ", nominal=state" if guided else ""
    path.write_text("from src.mppi.mppi import MPPI\n"
        "def collect(env, cfg, state, warmstart, seed):\n"
        "    controller = MPPI(env, cfg)\n"
        "    controller.reset(seed=seed)\n"
        f"    return controller.plan_step(state, initial_warmstart=warmstart{keyword})\n")
    report = deepcopy(bundle["report"])
    report["provenance"]["source_sha256"]["scripts/runners/run_walker2d.py"] = _sha(path)
    bundle["pair"]["json"].write_text(json.dumps(report))
    bundle["pair"]["runner_source"] = path
    return path


def test_exact_recovered_runner_closes_source_identity_without_claiming_fresh_qualification(bundle):
    path = _recovered_runner(bundle)
    result = _load(bundle)
    runner = result["metadata"]["inputs"][0]["historical_runner_identity"]
    assert runner["matches"] is True and runner["recovered_source_identity"] is True
    assert runner["recovery_source"] == str(path)
    assert runner["independent_call_contract"]["plan_keywords"] == ["initial_warmstart"]
    assert result["metadata"]["all_historical_source_identity_certified"] is True
    assert result["metadata"]["fresh_current_qualification_established"] is False
    assert "not fresh current qualification" in result["metadata"]["independence_provenance_assumption"]


def test_wrong_recovered_runner_override_fails_even_though_state_only_gap_is_otherwise_allowed(bundle):
    path = _recovered_runner(bundle)
    path.write_text(path.read_text() + "\n# Changed recovery bytes\n")
    with pytest.raises(ValueError, match="recorded historical runner hash"):
        _load(bundle)


def test_hash_matched_runner_with_guided_plan_call_is_not_certified_independent(bundle):
    _recovered_runner(bundle, guided=True)
    with pytest.raises(ValueError, match="independent MPPI call contract"):
        _load(bundle)
