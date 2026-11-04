"""Exercise the actual eight-round coordinator with only synthetic saved jobs.

No actor, optimizer or simulator is constructed. The fixture retains real
create-only byte bindings, snapshots, teacher admission and 2,400-row joins;
scientific job bodies and their compact index tapes are explicitly mocked.
"""

import json
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps import local_coupling_controller as controller
from src.gps.mppi_gps import local_coupling_learning as learning
from src.gps.mppi_gps import local_coupling_multiround as loop
from src.gps.mppi_gps import local_coupling_multiround_measurements as measurements
from src.gps.mppi_gps.provenance import write
from src.gps.mppi_gps.trajectory_pi_accounting import Work
from src.gps.mppi_gps.trajectory_pi_native import binding
from src.gps.mppi_gps.trajectory_pi_student import observations

ROOT = Path(__file__).resolve().parents[1]


def recipe():
    return json.loads((ROOT / "configs/studies/local_coupling_multiround_l3_0.json").read_text())


def saved_trace(path, *, slot=0, controls=300, action=0.0):
    """Save numerical fixtures, not a simulated trajectory."""
    states = np.zeros((controls + 1, 19), dtype=np.float64)
    states[:, 0] = np.arange(controls + 1) * 0.008
    states[:, 1], states[:, 2], states[:, 10] = slot, 1.2, 1.5
    with h5py.File(path, "x") as raw:
        for key, value in dict(
            states=states,
            observations=observations(states),
            warmstarts=np.zeros((controls + 1, 9), dtype=np.float64),
            integrations=np.zeros((controls + 1, 91), dtype=np.float64),
            actions=np.full((controls, 6), action, dtype=np.float64),
            costs=np.full(controls, 100.0 / controls, dtype=np.float64),
        ).items():
            raw.create_dataset(key, data=value, compression="gzip")
    return binding(path)


def local_outcome(*, controls=300, mae=0.1, cost=100.0):
    survived = controls == 300
    return dict(
        controls=controls,
        requested_controls=300,
        fell=not survived,
        full_survival=survived,
        local_walking=survived and mae <= 0.3,
        velocity_mae=mae,
        complete_cost=cost,
    )


def envelope(directory, *, arm, generation, ancestor, pcg_draws=0):
    directory.mkdir(parents=True, exist_ok=False)
    # These are text markers. They cannot be loaded as real Torch checkpoints.
    write(directory / "policy.pt", dict(synthetic=True, arm=arm, generation=generation))
    write(directory / "global_rng.pt", dict(synthetic=True, global_token="retained-global"))
    write(
        directory / "stream.json",
        dict(
            synthetic=True,
            arm=arm,
            generation=generation,
            ancestor=ancestor,
            adam_updates=4000 + generation * 2000,
            pcg_draws=pcg_draws,
            private_torch_token="retained-private",
            normalization_token="retained-normalization",
            global_token="retained-global",
        ),
    )
    return binding(directory / "stream.json")


def fixture_launch(tmp_path):
    config = deepcopy(recipe())
    config["launch_ready"] = True
    # Only reduce synthetic evaluation cardinality; all eight rounds, both arms
    # and all eight restart acquisitions are exercised without shortening them.
    config["seeds"]["panel_size"] = 1
    write(tmp_path / "config.json", config)
    initial = [
        dict(
            stream=0,
            arm=arm,
            envelope=envelope(
                tmp_path / ("initial_" + arm), arm=arm, generation=0, ancestor="L2-" + arm
            ),
        )
        for arm in loop.ARMS
    ]
    anchors = []
    for slot in range(8):
        trace = saved_trace(tmp_path / f"anchor{slot}.h5", slot=slot)
        path = tmp_path / f"anchor{slot}.json"
        write(path, dict(slot=slot, outcome=local_outcome(), trace=trace))
        anchors.append(dict(slot=slot, manifest=binding(path), trace=trace))
    # Restarts are bound provenance for mocked acquisition; no restore is called.
    write(tmp_path / "restarts.json", dict(synthetic=True))
    write(tmp_path / "held.json", dict(synthetic=True))
    write(tmp_path / "anchor_corpus.json", dict(synthetic=True))
    write(
        tmp_path / "prepared.json",
        dict(
            initial_streams=initial,
            anchors=anchors,
            restarts=binding(tmp_path / "restarts.json"),
            held=binding(tmp_path / "held.json"),
            anchor_corpus=binding(tmp_path / "anchor_corpus.json"),
        ),
    )
    return dict(
        base_directory=str(tmp_path),
        config=binding(tmp_path / "config.json"),
        prepared_manifest=binding(tmp_path / "prepared.json"),
    ), config


def mock_jobs(monkeypatch, config, *, fault=None):
    calls = dict(acquisition=[], fits=[], evaluations=[], probes=[])

    def closed(job, kind, payload, *, filename="result.json", status="COMPLETED"):
        out = Path(job["out"])
        work = Work(out, dict(job["limits"], mock_jobs=1))
        work.call("synthetic_saved_" + kind, dict(mock_jobs=1), lambda: None)
        receipt = work.close("COMPLETED" if status == "COMPLETED" else "FAILED")
        result = dict(payload, status=status)
        write(out / filename, result)
        return dict(path=str(out), result=result, receipt=receipt)

    def generation(checkpoint):
        return loop.read(loop.bound(checkpoint))["generation"]

    def acquire(job):
        out = Path(job["out"])
        out.mkdir(parents=True, exist_ok=False)
        r, arm, slot = job["round"], job["arm"], job["slot"]
        assert generation(job["checkpoint"]) == r - 1
        calls["acquisition"].append(deepcopy(job))
        fell = r in (2, 5)
        # Round4 is retention without walking; round7 positive violates cost.
        mae = 0.5 if r == 4 else 0.1
        cost = 112.0 if r == 7 and arm == "positive" else 100.0
        n = 5 if fell else 300
        action = 0.1 + slot * 0.001 if arm == "independent" else 0.2 + r * 0.01 + slot * 0.001
        trace = saved_trace(out / "trace.h5", slot=slot, controls=n, action=action)
        invalid = fault == "execution" and r == 2 and arm == "independent" and slot == 0
        return closed(
            job,
            "acquisition",
            dict(
                slot=slot,
                checkpoint=job["checkpoint"],
                trace=trace,
                outcome=local_outcome(controls=n, mae=mae, cost=cost),
            ),
            filename="manifest.json",
            status="FAILED_STOPPED" if invalid else "COMPLETED",
        )

    def fit(job):
        out = Path(job["out"])
        out.mkdir(parents=True, exist_ok=False)
        before = loop.read(loop.bound(job["incoming"]))
        r, arm = job["round"], job["arm"]
        assert before["generation"] == r - 1 and before["arm"] == arm
        assert before["ancestor"] == "L2-" + arm
        assert before["adam_updates"] == 4000 + (r - 1) * 2000
        assert before["pcg_draws"] == (r - 1) * 2000
        assert config["learning"]["updates"] == 2000
        with np.load(loop.bound(job["corpus"]), allow_pickle=False) as raw:
            assert len(raw["actions"]) == 2400
        outgoing = envelope(
            out / "stream",
            arm=arm,
            generation=r,
            ancestor=before["ancestor"],
            pcg_draws=before["pcg_draws"] + 2000,
        )
        indices = np.array([[0, 3], [5, 7]], dtype=np.int64)
        if fault == "indices" and r == 2 and arm == "positive":
            indices[0, 0] = 1
        with (out / "index_tape.npz").open("xb") as target:
            np.savez(target, indices=indices)
        calls["fits"].append(
            dict(
                round=r,
                arm=arm,
                incoming=job["incoming"],
                outgoing=outgoing,
                requested_update_budget=2000,
                before=before,
                mock_only=True,
            )
        )
        return closed(
            job,
            "fit",
            dict(
                outgoing=outgoing,
                index_tape=binding(out / "index_tape.npz"),
                diagnostics_before=dict(own=dict(action_mse=0.5)),
                diagnostics_after=dict(own=dict(action_mse=0.7 if r == 3 else 0.5)),
            ),
        )

    def evaluate(job):
        out = Path(job["out"])
        out.mkdir(parents=True, exist_ok=False)
        g = generation(job["checkpoint"])
        role = job["role"]
        assert g == (0 if role["kind"] == "initial" else role["round"])
        calls["evaluations"].append(deepcopy(job))
        # All endpoints fail walking. Some retain balance before later regression.
        survived = role["round"] == 3
        return closed(
            job,
            "evaluation",
            dict(
                checkpoint=job["checkpoint"],
                outcome=dict(
                    walking=False,
                    full_survival=survived,
                    controls=1000 if survived else 1,
                    complete_cost=100.0 if survived else 5000.0,
                ),
            ),
        )

    def probe(job):
        Path(job["out"]).mkdir(parents=True, exist_ok=False)
        evaluation = loop.read(loop.bound(job["evaluation"]))
        assert evaluation["checkpoint"] == job["checkpoint"]
        calls["probes"].append(deepcopy(job))
        return closed(
            job, "probe", dict(anchors=[dict(control=0, status="MISSING_AFTER_EXECUTED_PREFIX")])
        )

    monkeypatch.setattr(controller, "run_episode", acquire)
    monkeypatch.setattr(learning, "fit_job", fit)
    monkeypatch.setattr(learning, "evaluate_job", evaluate)
    monkeypatch.setattr(measurements, "probe_job", probe)
    return calls


def test_all_eight_rounds_run_despite_zero_walking_regression_and_fallback(tmp_path, monkeypatch):
    launch, config = fixture_launch(tmp_path)
    calls = mock_jobs(monkeypatch, config)
    study = loop.StreamStudy(launch, 0)
    result = study.run()
    assert result["status"] == "COMPLETED_ALL_ROUNDS", result.get("error")
    assert [row["round"] for row in result["rounds"]] == list(range(1, 9))
    assert all(row["status"] == "COMPLETED" for row in result["rounds"])
    assert len(calls["acquisition"]) == 128 and len(calls["fits"]) == 16
    assert len(calls["evaluations"]) == 22 and len(calls["probes"]) == 18
    assert sum(row["requested_update_budget"] for row in calls["fits"]) == 32_000
    assert result["accounting"]["counts"] == dict(requested_mock_jobs=184, completed_mock_jobs=184)
    assert len(result["accounting"]["child_receipts"]) == 184
    assert len({row["path"] for row in result["accounting"]["child_receipts"]}) == 184
    assert all(
        not loop.read(loop.bound(row["result"]))["outcome"]["walking"]
        for block in result["rounds"]
        for row in block["evaluations"]
    )
    # Invalid teachers do not enter data, but their attempted evidence remains.
    for r in (2, 4, 5):
        block = result["rounds"][r - 1]
        assert all(row["fallback"] and row["new_rows"] == 0 for row in block["admissions"])
        assert all(Path(row["attempt"]["path"]).exists() for row in block["admissions"])
    assert all(row["age"] == 2 for row in result["rounds"][4]["admissions"])
    # A repeated accepted independent solve is not reported as new experience.
    stationary = [row for row in result["rounds"][2]["admissions"] if row["arm"] == "independent"]
    assert all(
        row["admitted"] and not row["new_experience"] and row["new_rows"] == 0 for row in stationary
    )
    for block in result["rounds"]:
        corpus = loop.read(loop.bound(block["corpus"]))
        assert corpus["rows"] == 2400
        for arm in loop.ARMS:
            acquired = [
                job
                for job in calls["acquisition"]
                if job["round"] == block["round"] and job["arm"] == arm
            ]
            before = next(
                row for row in calls["fits"] if row["round"] == block["round"] and row["arm"] == arm
            )
            expected = binding(Path(before["incoming"]["path"]).parent / "policy.pt")
            assert all(job["checkpoint"] == expected for job in acquired)
            assert all(job["seed"] == loop.planner_seed(config, 0, job["slot"]) for job in acquired)
    # All actual snapshots are create-only files; only the convenience pointer moves.
    final = loop.read(study.root / "result.json")
    assert final["status"] == "COMPLETED_ALL_ROUNDS"
    assert (study.root / "result.json").is_symlink()
    assert len(list((study.root / "progress").glob("*.json"))) > 128
    for arm in loop.ARMS:
        final_state = loop.read(loop.bound(study.incoming[arm]))
        assert final_state["generation"] == 8 and final_state["adam_updates"] == 20_000
        assert final_state["pcg_draws"] == 16_000
        assert final_state["ancestor"] == "L2-" + arm
        assert final_state["private_torch_token"] == "retained-private"
        assert final_state["global_token"] == "retained-global"


@pytest.mark.parametrize(
    "fault,reason",
    [("execution", "Invalid execution"), ("indices", "Paired minibatch streams diverged")],
)
def test_invalid_execution_or_broken_paired_stream_stops_without_scientific_retry(
    tmp_path, monkeypatch, fault, reason
):
    launch, config = fixture_launch(tmp_path)
    calls = mock_jobs(monkeypatch, config, fault=fault)
    result = loop.StreamStudy(launch, 0).run()
    assert result["status"] == "FAILED_STOPPED" and reason in result["error"]
    assert [row["round"] for row in result["rounds"]] == [1, 2]
    assert result["rounds"][0]["status"] == "COMPLETED"
    assert result["rounds"][1]["status"] == "RUNNING"
    assert not result["final_evaluations"]
    assert max(job["round"] for job in calls["acquisition"]) == 2
    assert len({row["path"] for row in result["accounting"]["child_receipts"]}) == len(
        result["accounting"]["child_receipts"]
    )


def test_fixed_planner_seed_and_per_stream_work_do_not_multiply_global_budget():
    config = recipe()
    assert config["rounds"] == 8 and config["streams"] == 3
    assert config["seeds"]["panel_size"] == 16
    for stream in range(3):
        for slot in range(8):
            assert loop.planner_seed(config, stream, slot) == 2961000000 + 100 * stream + slot
    changed = deepcopy(config)
    changed["seeds"]["planner_round_stride"] = 10000
    with pytest.raises(ValueError, match="fixes acquisition"):
        loop.planner_seed(changed, 0, 0)
    limits = loop.child_limits(config, {})
    glob, audit = config["budgets"]["global"], config["budgets"]["audit"]
    assert limits["optimizer_steps"] == 32_000
    assert 3 * limits["optimizer_steps"] == glob["optimizer_steps"]
    assert 3 * limits["backward_rows"] == glob["backward_rows"]
    assert 3 * limits["model_controls"] <= glob["model_controls"]
    assert 3 * limits["neural_rows"] <= glob["neural_rows"] - audit["neural_rows"]
    charged = loop.child_limits(
        config, dict(requested_optimizer_steps=2000, requested_neural_rows=123)
    )
    assert charged["optimizer_steps"] == limits["optimizer_steps"] - 2000
    assert charged["neural_rows"] == limits["neural_rows"] - 123


def test_exhausted_update_reservation_rejects_before_even_a_mock_operation(tmp_path):
    config = recipe()
    limits = loop.child_limits(config, dict(requested_optimizer_steps=32_000))
    called = []
    work = Work(tmp_path, limits)
    with pytest.raises(RuntimeError, match="optimizer_steps"):
        work.call("never_construct_optimizer", dict(optimizer_steps=1), lambda: called.append(True))
    receipt = work.close("FAILED")
    assert not called and receipt["counts"] == {}


def test_reaped_cpu_is_cumulative_and_added_only_while_coordinator_is_live():
    from src.gps.mppi_gps.experiments.local_coupling_multiround import include_reaped_cpu

    live = dict(cpu_seconds=30.0, rss_bytes=100, observed_pids=[100, 101, 103])
    receipts = [dict(reaped_child_cpu_seconds=10.0), dict(reaped_child_cpu_seconds=27.0)]
    value = include_reaped_cpu(live, 101, receipts)
    assert value["cpu_seconds"] == 57.0  # Cumulative max, not10+27 again.
    assert live["cpu_seconds"] == 30.0
    assert include_reaped_cpu(live, 101, [])["cpu_seconds"] == 30.0
    # The outer waited-child value already includes coordinator descendants.
    reaped = dict(cpu_seconds=80.0, rss_bytes=100, observed_pids=[100])
    assert include_reaped_cpu(reaped, 101, receipts)["cpu_seconds"] == 80.0
    assert include_reaped_cpu(reaped, None, receipts)["cpu_seconds"] == 80.0


@pytest.mark.parametrize(
    "resource_name,reason",
    [
        ("cpu", "Shared worker CPU reservation exhausted"),
        ("rss", "Shared aggregate RSS exhausted"),
    ],
)
def test_supervisor_shared_guard_rejects_before_any_mock_worker_body(
    tmp_path, monkeypatch, resource_name, reason
):
    from src.gps.mppi_gps import trajectory_pi_launch as launching
    from src.gps.mppi_gps.experiments import local_coupling_multiround as experiment

    launch, config = fixture_launch(tmp_path)
    (tmp_path / "source").mkdir()
    launch["source_directory"] = str(tmp_path / "source")
    monkeypatch.setattr(loop, "verify_launch", lambda path: launch)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTHONDONTWRITEBYTECODE",
    ):
        monkeypatch.setenv(name, "5")
    before = dict(cpu_seconds=0, rss_bytes=100, observed_pids=[100])
    exceeded = dict(cpu_seconds=0, rss_bytes=100, observed_pids=[100])
    if resource_name == "cpu":
        exceeded["cpu_seconds"] = config["budgets"]["worker"]["cpu_seconds"] + 1
        assert (
            exceeded["cpu_seconds"]
            < config["budgets"]["global"]["cpu_seconds"]
            - config["budgets"]["preparation"]["cpu_seconds"]
        )
    else:
        exceeded["rss_bytes"] = config["budgets"]["global"]["memory_bytes"] + 1
    values = iter((before, exceeded))
    monkeypatch.setattr(experiment, "live_tree_usage", lambda: next(values))
    phases = []

    def guarded_phase(name, argv, base, source, deadline, guard):
        phases.append(name)
        guard()  # The real shared guard, before even a mocked worker action.
        pytest.fail("Resource exhaustion must reject before the worker body")

    monkeypatch.setattr(launching, "run_phase", guarded_phase)
    with pytest.raises(RuntimeError, match=reason):
        experiment.supervise(tmp_path / "launch.json")
    assert phases == ["worker"] and not (tmp_path / "external_worker.json").exists()
    assert all(
        experiment.os.environ[name] == "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    )
