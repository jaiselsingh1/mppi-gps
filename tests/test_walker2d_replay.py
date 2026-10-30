"""Runner traces must let any recorded decision be replayed exactly."""

import json

import numpy as np
import pytest

from scripts.runners import run_walker2d as runner
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import compare_fields, load_episode, restore_decision

_SMALL_CONFIG = {
    "K": 8,
    "H": 6,
    "lam": 3.0,
    "noise_sigma": 0.3,
    "noise_lowpass_cutoff_hz": 23.4375,
    "noise_lowpass_sample_rate_hz": 125.0,
    "noise_lowpass_order": 2,
    "use_is_correction": False,
    "initial_iterations": 2,
}


@pytest.fixture(scope="module")
def traced_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("walker_replay")
    config = directory / "mppi.json"
    config.write_text(json.dumps(_SMALL_CONFIG))
    traces = directory / "traces.npz"
    runner.main(
        config=config,
        episodes=2,
        steps=12,
        seed=3,
        planner_seed=40,
        traces_output=traces,
        log_every=0,
    )
    return config, traces


def test_trace_decision_fields_align_with_actions(traced_run):
    _, traces = traced_run
    for index in range(2):
        episode = load_episode(traces, index)
        steps = len(episode["actions"])
        assert episode["pre_plan_nominals"].shape == (steps, 6, 6)
        assert episode["plans"].shape == (steps, 6, 6)
        assert episode["planner_rng_states"].shape == (steps,)
        np.testing.assert_array_equal(episode["initial_plan_flags"], np.arange(steps) == 0)
        np.testing.assert_array_equal(episode["actions"], episode["plans"][:, 0])
        np.testing.assert_array_equal(
            episode["pre_plan_nominals"][1:, :-1], episode["plans"][:-1, 1:]
        )
        np.testing.assert_array_equal(
            episode["pre_plan_nominals"][1:, -1], episode["plans"][:-1, -1]
        )
        np.testing.assert_array_equal(episode["pre_plan_nominals"][0], np.zeros((6, 6)))


@pytest.mark.parametrize("episode_index, step", [(0, 0), (0, 7), (1, 4)])
def test_restoring_a_recorded_decision_reproduces_the_rest_of_the_episode(
    traced_run, episode_index, step
):
    config, traces = traced_run
    episode = load_episode(traces, episode_index)
    task = json.loads((runner._REPO_ROOT / "configs/walker2d_task.json").read_text())
    env = Walker2d(**task)
    try:
        controller = MPPI(env, MPPIConfig(**json.loads(config.read_text())))
        restore_decision(env, controller, episode, step)
        replay = {"actions": [], "states": [episode["states"][step]], "costs": []}
        for _ in range(step, len(episode["actions"])):
            action, _ = controller.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
            _, cost, _, _ = env.step(action)
            replay["actions"].append(action)
            replay["states"].append(env.get_state())
            replay["costs"].append(cost)
    finally:
        env.close()

    reference = {
        "actions": episode["actions"][step:],
        "states": episode["states"][step:],
        "costs": episode["costs"][step:],
    }
    report = compare_fields(reference, replay, ("actions", "states", "costs"))
    assert all(field["exact"] for field in report.values()), report


def test_replay_checker_fails_on_a_mismatch_after_saving_its_report(traced_run, tmp_path):
    from scripts.diagnostics import replay_walker

    config, traces = traced_run
    with np.load(traces) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["actions"] = arrays["actions"].copy()
    arrays["actions"][3, 0] += 1e-9
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **arrays)

    replay_walker.main(
        reference=traces,
        reference_episode=0,
        out_dir=tmp_path / "exact",
        steps=12,
        config=config,
        restore_length=4,
    )
    assert json.loads((tmp_path / "exact" / "replay_report.json").read_text())[
        "reference_trace_exact"
    ]

    with pytest.raises(SystemExit) as error:
        replay_walker.main(
            reference=tampered,
            reference_episode=0,
            out_dir=tmp_path / "tampered",
            steps=12,
            config=config,
            restore_length=4,
        )
    assert error.value.code == 1
    report = json.loads((tmp_path / "tampered" / "replay_report.json").read_text())
    assert not report["reference_trace_exact"]
    assert report["fields"]["actions"]["first_divergence"] == 3
