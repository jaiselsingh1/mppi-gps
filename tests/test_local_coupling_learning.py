"""L2 contracts on tiny saved traces, eight synthetic updates and a fake simulator."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.gps.mppi_gps import local_coupling_learning as learning
from src.gps.mppi_gps.policy import Policy
from src.gps.mppi_gps.provenance import fingerprint, write
from src.gps.mppi_gps.trajectory_pi_student import (
    SETTINGS,
    global_rng_state,
    initial_stream,
    load_stream,
    normalization_from_training,
    observations,
    save_stream,
)


@pytest.fixture(scope="module", autouse=True)
def restore_torch_runtime():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def limits(**changes):
    return (
        dict(
            wall_seconds=60,
            output_bytes=64 * 1024**2,
            memory_bytes=4 * 1024**3,
            checkpoint_loads=100,
            checkpoint_writes=100,
            policy_constructions=100,
            optimizer_constructions=100,
            neural_rows=100000,
            backward_rows=100000,
            optimizer_steps=100,
            environment_constructions=1,
            resets=1,
            external_mj_forward_calls=2,
            native_controls=1000,
            physics_substeps=4000,
        )
        | changes
    )


def synthetic_trace(path, n=5, offset=0.0, context=0.0):
    states = np.zeros((n + 1, 19), dtype=np.float64)
    states[:, 0] = np.arange(n + 1) * 0.008
    states[:, 2], states[:, 10] = 1.2, 1.5
    states[:, 4] = context
    states[1:, 5] = offset
    with h5py.File(path, "x") as stream:
        for name, value in dict(
            states=states,
            warmstarts=np.zeros((n + 1, 9)),
            integrations=np.zeros((n + 1, 91)),
            observations=observations(states),
            actions=np.full((n, 6), offset),
            costs=np.ones(n),
        ).items():
            stream[name] = value
    return learning.binding(path)


def rosters(path, *, zero_slot=None, changed_context=None):
    path.mkdir()
    result = []
    for arm in ("a", "b"):
        rows = []
        for slot in range(8):
            n = (4 if arm == "a" else 3) + (slot % 2)
            if (arm, slot) == zero_slot:
                n = 0
            context = 0.01 if (arm, slot) == changed_context else 0.0
            rows.append(
                dict(
                    slot=slot,
                    trace=synthetic_trace(
                        path / f"{arm}{slot}.h5",
                        n,
                        offset=0.1 if arm == "a" else -0.2,
                        context=context,
                    ),
                )
            )
        result.append(rows)
    return result


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Exactly four two-update fit jobs: eight Adam updates/2048 backward rows."""
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("l2-synthetic")
    states = np.zeros((11, 19), dtype=np.float64)
    states[:, 2] = 1.2
    states[:, 10] = 1.5
    states[:, 4] = np.linspace(-0.2, 0.2, 11)
    obs = observations(states)
    actions = np.full((11, 6), 0.1)
    parent = Policy(seed=1701, normalization=normalization_from_training(obs))
    actor, state, _ = initial_stream(parent, obs, actions, seed=1901)
    incoming = root / "incoming"
    save_stream(incoming, actor, state)
    config = dict(
        schema="local-coupling-learning-l2-1",
        launch_ready=True,
        inputs=dict(checkpoint=learning.binding(incoming / "policy.pt")),
        learning=SETTINGS | dict(updates=2, gradient_clip=0.5, device="cpu", torch_threads=1),
        evaluation=dict(horizon=3),
    )
    config_path = root / "config.json"
    write(config_path, config)
    bound = learning.binding(config_path)
    forked = learning.fork_streams(incoming, root / "forks", config_binding=bound, limits=limits())
    assert forked["result"]["status"] == "COMPLETED"
    a, b = rosters(root / "raw")
    corpus = learning.build_paired_corpora(a, b, root / "corpus", config_binding=bound)
    held = root / "held.npz"
    np.savez(held, states=states, observations=obs, actions=actions)
    jobs, results = {}, {}
    for round_index in (1, 2):
        for arm in ("a", "b"):
            source = (
                forked["result"]["streams"][0]["envelope"]
                if round_index == 1
                else results[(1, arm)]["result"]["outgoing"]
            )
            job = dict(
                config=bound,
                incoming=source,
                corpus=corpus["files"][arm],
                out=str(root / f"r{round_index}_{arm}"),
                round=round_index,
                stream=0,
                arm=arm,
                limits=limits(),
                diagnostics=dict(
                    own=corpus["files"][arm],
                    cross=corpus["files"]["b" if arm == "a" else "a"],
                    held=learning.binding(held),
                ),
            )
            jobs[(round_index, arm)] = job
            results[(round_index, arm)] = learning.fit_job(job)
            assert results[(round_index, arm)]["result"]["status"] == "COMPLETED", results[
                (round_index, arm)
            ]
    return dict(
        root=root,
        incoming=incoming,
        bound=bound,
        forks=forked,
        corpus=corpus,
        jobs=jobs,
        results=results,
        config=config,
    )


def test_import_has_no_torch_or_simulator_side_effects():
    code = "import sys; import src.gps.mppi_gps.local_coupling_learning; assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True)


def test_saved_terminal_observation_is_checked_without_any_network_or_physics(tmp_path):
    path = tmp_path / "trace.h5"
    learning._trace(synthetic_trace(path, n=2), 300)
    with h5py.File(path, "r+") as raw:
        raw["observations"][-1, 0] = np.nan
    with pytest.raises(ValueError, match="pre-action observations differ"):
        learning._trace(learning.binding(path), 300)


def test_corpus_uses_every_common_prefix_and_exact_pre_action_rows(pipeline):
    corpus = pipeline["corpus"]
    assert corpus["rows"] == 28
    a, b = [learning._arrays(Path(corpus["files"][arm]["path"])) for arm in ("a", "b")]
    assert np.array_equal(a["row_ids"], b["row_ids"])
    assert set(a["row_ids"][:, 0]) == set(range(8))
    assert all(row["common"] == min(row["controls"]) for row in corpus["cases"])
    for raw in (a, b):
        assert raw["actions"].dtype == np.float64
        assert np.array_equal(raw["observations"], observations(raw["states"]))


def test_all_three_pcg_forks_preserve_exact_operative_state_and_raw_globals(pipeline):
    actor, state, _ = load_stream(pipeline["incoming"])
    seen = []
    for row in pipeline["forks"]["result"]["streams"]:
        directory = Path(row["directory"])
        actual, continued, metadata = load_stream(directory)
        assert actual.hash() == actor.hash()
        assert fingerprint(continued["adam"]) == fingerprint(state["adam"])
        assert fingerprint(actual._generator.get_state()) == fingerprint(
            actor._generator.get_state()
        )
        assert fingerprint(actual.normalization) == fingerprint(actor.normalization)
        assert (directory / "global_rng.pt").read_bytes() == (
            pipeline["incoming"] / "global_rng.pt"
        ).read_bytes()
        generator = np.random.PCG64(0)
        generator.state = deepcopy(state["rng"])
        expected = generator.jumped(row["stream"] + 1).state
        assert continued["rng"] == expected
        assert metadata["pcg_jump"] == row["stream"] + 1
        seen.append(fingerprint(continued["rng"]))
    assert len(set(seen)) == 3
    assert pipeline["forks"]["receipt"]["counts"]["completed_checkpoint_writes"] == 9


def test_paired_tapes_and_round_two_actual_adam_rng_continuation(pipeline):
    for round_index in (1, 2):
        a = pipeline["results"][(round_index, "a")]
        b = pipeline["results"][(round_index, "b")]
        with (
            np.load(a["result"]["index_tape"]["path"]) as first,
            np.load(b["result"]["index_tape"]["path"]) as second,
        ):
            assert np.array_equal(first["indices"], second["indices"])
            assert first["indices"].shape == (2, 256)
        for arm, returned in (("a", a), ("b", b)):
            assert returned["result"]["updates"] == 2
            assert returned["receipt"]["counts"]["completed_optimizer_steps"] == 2
            assert returned["receipt"]["counts"]["completed_backward_rows"] == 512
            assert returned["result"]["cumulative_updates"] == round_index * 2
            if round_index == 2:
                prior = pipeline["results"][(1, arm)]["result"]
                assert (
                    returned["result"]["before"]["optimizer_fingerprint"]
                    == prior["after"]["optimizer_fingerprint"]
                )
                _, old_state, _ = load_stream(Path(prior["outgoing"]["path"]).parent)
                actual = torch.load(returned["result"]["initial_state"]["path"], weights_only=True)
                assert fingerprint(actual["adam"]) == fingerprint(old_state["adam"])
                assert actual["rng"] == old_state["rng"]
    assert (
        sum(
            row["receipt"]["counts"]["completed_optimizer_steps"]
            for row in pipeline["results"].values()
        )
        == 8
    )


def test_diagnostics_keep_torch_prediction_grouping_and_saved_float32_outputs(pipeline):
    result = pipeline["results"][(1, "a")]["result"]
    for stage in ("diagnostics_before", "diagnostics_after"):
        for row in result[stage].values():
            assert row["torch_prediction_batch_size"] == 1024
            with np.load(row["predictions"]["path"]) as raw:
                assert raw["predictions"].dtype == np.float32
                assert raw["predictions"].shape == (row["rows"], 6)


@pytest.mark.parametrize("bad", ["context", "zero"])
def test_missing_common_slot_or_mismatched_snapshot_stops_before_output(tmp_path, pipeline, bad):
    kwargs = dict(changed_context=("b", 4)) if bad == "context" else dict(zero_slot=("b", 4))
    a, b = rosters(tmp_path / "raw", **kwargs)
    with pytest.raises(ValueError, match="contexts differ|Every restart"):
        learning.build_paired_corpora(a, b, tmp_path / "out", config_binding=pipeline["bound"])
    assert not (tmp_path / "out").exists()


def test_round_two_cannot_restart_or_use_another_arms_child(tmp_path, pipeline):
    job = dict(pipeline["jobs"][(2, "a")], out=str(tmp_path / "wrong"), arm="b")
    result = learning.fit_job(job)
    assert result["result"]["status"] == "FAILED_STOPPED"
    assert "same-arm round-one child" in result["result"]["error"]
    assert result["receipt"]["counts"].get("completed_optimizer_steps", 0) == 0


def test_budget_rejects_before_entering_fitter_and_preserves_paid_loading(tmp_path, pipeline):
    job = dict(
        pipeline["jobs"][(1, "a")],
        out=str(tmp_path / "budget"),
        limits=limits(optimizer_constructions=0),
    )
    result = learning.fit_job(job)
    assert result["result"]["status"] == "FAILED_STOPPED"
    assert result["receipt"]["counts"]["completed_checkpoint_loads"] == 2
    assert result["receipt"]["counts"].get("requested_optimizer_steps", 0) == 0
    assert result["receipt"]["counts"].get("requested_backward_rows", 0) == 0
    with np.load(result["result"]["index_tape"]["path"]) as raw:
        assert raw["indices"].shape == (0, 256)


def test_torch_only_actor_and_native_recorder_on_fake_physics(tmp_path, pipeline, monkeypatch):
    from src.gps.mppi_gps import runtime
    from src.gps.mppi_gps import trajectory_pi_native as native
    from src.gps.mppi_gps.trajectory_pi_student import NumpyPolicy

    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy actor must never deploy")

    monkeypatch.setattr(NumpyPolicy, "from_checkpoint", forbidden)

    class FakeEnv:
        def __init__(self, work):
            self.work = work
            self.state = np.zeros(19)
            self.state[2], self.state[10] = 1.2, 1.5
            self.count = 0

        def reset(self):
            self.state[4] = np.random.uniform(-0.01, 0.01)
            return self._get_obs()

        def get_state(self):
            return self.state.copy()

        def _get_obs(self):
            return observations(self.state)

        def step(self, action):
            def calculate():
                self.count += 1
                self.state[0] += 0.008
                fell = self.count == 2
                self.state[2] = 0.6 if fell else 1.2
                return (
                    self._get_obs(),
                    21.0 if fell else 1.0,
                    fell,
                    dict(fall_cost=20.0 if fell else 0.0),
                )

            return self.work.call(
                "mock_native_control", dict(native_controls=1, physics_substeps=4), calculate
            )

        def close(self):
            pass

    def snapshot(env):
        return dict(state=env.state.copy(), warmstart=np.zeros(9), integration=np.zeros(91))

    monkeypatch.setattr(runtime, "snapshot", snapshot)
    monkeypatch.setattr(native, "snapshot", snapshot)
    monkeypatch.setattr(
        native,
        "_env",
        lambda job, work: work.call(
            "mock_environment", dict(environment_constructions=1), lambda: FakeEnv(work)
        ),
    )
    source = Path(pipeline["results"][(1, "a")]["result"]["outgoing"]["path"]).parent / "policy.pt"
    job = dict(
        config=pipeline["bound"],
        checkpoint=learning.binding(source),
        out=str(tmp_path / "evaluation"),
        task={},
        reset_seed=91,
        limits=limits(),
        physics_threads=4,
    )
    before = fingerprint(global_rng_state())
    returned = learning.evaluate_job(job)
    assert returned["result"]["status"] == "COMPLETED"
    assert fingerprint(global_rng_state()) == before
    assert returned["result"]["outcome"]["controls"] == 2
    assert returned["result"]["outcome"]["complete_cost"] == 27
    assert not returned["result"]["outcome"]["walking"]
    counts = returned["receipt"]["counts"]
    assert counts["completed_neural_rows"] == 2
    assert counts["completed_native_controls"] == 2
    assert counts["completed_physics_substeps"] == 8
    with h5py.File(returned["result"]["trace"]["path"]) as raw:
        assert raw["states"].shape == (3, 19)
        assert raw["warmstarts"].shape == (3, 9)
        assert raw["integrations"].shape == (3, 91)
        assert raw["observations"].shape == (3, 17)
        assert raw["actions"].shape == (2, 6)
        actual, _, _ = Policy.from_checkpoint(source, device="cpu")
        for index in range(2):
            assert np.array_equal(
                actual.predict(raw["observations"][index]).astype(np.float64), raw["actions"][index]
            )
