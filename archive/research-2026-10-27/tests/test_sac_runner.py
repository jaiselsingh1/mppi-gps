"""Exercise the real collector, learner and saved-policy evaluation together."""

import copy
import json

import numpy as np
import pytest
import torch

from scripts.runners import eval_policy_walker2d as evaluation
from scripts.training import train_walker2d_sac as training
from src.rl import sac
from src.rl.walker import WalkerRL


@pytest.mark.parametrize("fall", [False, True])
def test_collector_and_actual_update_use_final_observation_and_terminal_mask(fall, monkeypatch):
    env = WalkerRL(max_episode_steps=1)
    replay = sac.ReplayBuffer(4, 17, 6, seed=2)
    agent = sac.SACAgent(17, -np.ones(6), np.ones(6),
                         sac.SACConfig(hidden_dim=8, batch_size=1))
    try:
        obs, _ = env.reset(seed=7)
        if fall:
            state = env.walker.get_state()
            state[3] = 0.99999
            state[10:] = 0.0
            state[12] = 5.0
            obs, _ = env.reset(options={"state": state})
        final, reward, terminated, truncated, info = training.collect_transition(
            env, replay, obs, np.zeros(6)
        )
        assert terminated == fall and truncated != fall
        reset, _ = env.reset(seed=8)
        np.testing.assert_array_equal(replay.next_obs[0], final.astype(np.float32))
        assert not np.array_equal(replay.next_obs[0], reset.astype(np.float32))
        assert replay.terminated[0, 0] == float(fall)
        assert replay.rewards[0, 0] == np.float32(reward)
        assert info["failure_tail"] == pytest.approx(-995.0 if fall else 0.0)

        captured = {}
        original_target = sac.sac_target

        def inspect_target(rewards, terminated, next_q, next_log_pi, gamma, alpha):
            target = original_target(rewards, terminated, next_q, next_log_pi, gamma, alpha)
            captured.update(rewards=rewards.clone(), terminated=terminated.clone(),
                            next_q=next_q.clone(), next_log_pi=next_log_pi.clone(),
                            target=target.clone(), alpha=alpha.detach().clone())
            return target

        monkeypatch.setattr(sac, "sac_target", inspect_target)
        actor_inputs = []
        hook = agent.actor.register_forward_pre_hook(
            lambda module, args: actor_inputs.append(args[0].detach().clone())
        )
        try:
            metrics = agent.update(replay, step=1)  # Critic update only.
        finally:
            hook.remove()
        assert all(np.isfinite(value) for value in metrics.values())
        if fall:
            assert not actor_inputs  # No policy/value/entropy after absorption.
            torch.testing.assert_close(captured["target"], captured["rewards"])
        else:
            assert len(actor_inputs) == 1
            np.testing.assert_array_equal(actor_inputs[0].numpy()[0], final.astype(np.float32))
            expected = captured["rewards"] + 0.995 * (
                captured["next_q"] - captured["alpha"] * captured["next_log_pi"]
            )
            torch.testing.assert_close(captured["target"], expected)
        assert metrics["target_q_mean"] == captured["target"].item()
    finally:
        env.close()


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("sac_runner") / "train"
    training.main(run_dir=directory, total_steps=12, learning_starts=4,
                  save_every=6, eval_every=6, eval_episodes=1, batch_size=4,
                  hidden_dim=8, buffer_size=32, max_episode_steps=3)
    return directory


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _assert_trace_report(trace_path, report):
    assert report["traces"]["sha256"] == evaluation.sha256(trace_path)
    env = WalkerRL()
    try:
        low, high = env.walker.action_bounds
        with np.load(trace_path, allow_pickle=False) as data:
            lengths = [episode["steps"] for episode in report["episodes"]]
            np.testing.assert_array_equal(data["episode_step_offsets"], np.cumsum([0, *lengths]))
            np.testing.assert_array_equal(data["episode_state_offsets"],
                                          np.cumsum([0, *[n + 1 for n in lengths]]))
            np.testing.assert_array_equal(data["reset_seeds"],
                                          [episode["reset_seed"] for episode in report["episodes"]])
            assert data["states"].shape == (sum(lengths) + len(lengths), env.walker.state_dim)
            assert data["warmstarts"].shape == (len(data["states"]), 9)
            assert data["actions"].shape == (sum(lengths), 6)
            assert data["contacts"].shape == (sum(lengths), 2)
            assert np.all((low <= data["actions"]) & (data["actions"] <= high))
            for index, episode in enumerate(report["episodes"]):
                a, b = data["episode_step_offsets"][index:index + 2]
                s, t = data["episode_state_offsets"][index:index + 2]
                states, costs = data["states"][s:t], data["costs"][a:b]
                assert len(states) == len(costs) + 1 == episode["steps"] + 1
                env.reset(seed=episode["reset_seed"])
                np.testing.assert_array_equal(states[0], env.walker.get_state())
                np.testing.assert_array_equal(data["warmstarts"][s], env.walker.get_warmstart())
                expected_mae = np.mean(np.abs(report["task"]["target_velocity"] - states[1:, 10]))
                assert episode["velocity_mae"] == expected_mae
                assert episode["total_cost"] == pytest.approx(costs.sum())
                assert episode["training_return"] == pytest.approx(data["training_rewards"][a:b].sum())
    finally:
        env.close()


def test_tiny_training_saves_loadable_ladder_and_finite_metrics(tiny_run):
    status = json.loads((tiny_run / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["step"] == status["total_steps"] == 12
    assert status["updates"] == 8
    assert status["episodes"] == 4
    assert status["latest_update"]
    assert all(np.isfinite(value) for value in status["latest_update"].values())
    checkpoints = _records(tiny_run / "checkpoints.jsonl")
    assert [record["step"] for record in checkpoints] == [0, 6, 12]
    for record in checkpoints:
        path = tiny_run / "checkpoints" / f"step_{record['step']:07d}.pt"
        assert evaluation.sha256(path) == record["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["step"] == record["step"]
        assert payload["gamma"] == 0.995
        assert payload["policy"]["observation_transform"] == "identity"
        actor = sac.load_policy(payload["policy"])
        assert np.isfinite(actor.act(np.zeros(17))).all()

    metrics = _records(tiny_run / "metrics.jsonl")
    evaluations = [record for record in metrics if record["type"] == "evaluation"]
    assert [record["step"] for record in evaluations] == [0, 6, 12]
    assert all(np.isfinite(value) for record in evaluations for value in record["summary"].values())
    episodes = _records(tiny_run / "episodes.jsonl")
    assert len(episodes) == 4
    assert all(episode["truncated"] and not episode["terminated"] for episode in episodes)
    assert all(episode["failure_tail"] == 0.0 for episode in episodes)
    assert all(0 <= episode["reset_seed"] < 40000 for episode in episodes)


def test_run_retains_exact_source_and_task_provenance(tiny_run):
    config = json.loads((tiny_run / "config.json").read_text())
    metadata = config["provenance"]
    assert len(metadata["git_head"]) == 40
    assert metadata["control_dt"] == 0.008
    assert metadata["physics_dt"] == 0.002
    assert metadata["frame_skip"] == 4
    assert metadata["torch_threads"] == metadata["simulator_threads"] == 1
    assert metadata["task_sha256"] == evaluation.sha256(training.DEFAULT_TASK)
    assert config["failure_tail"] == pytest.approx(-995.0)
    assert config["sac"]["gamma"] == 0.995
    assert config["task"] == json.loads((tiny_run / "task.json").read_text())
    expected_sources = {
        "src/envs/walker2d.py", "src/envs/mujoco_env.py", "src/rl/walker.py",
        "src/rl/sac.py", "scripts/training/train_walker2d_sac.py",
        "scripts/runners/eval_policy_walker2d.py", "scripts/runners/run_walker2d.py",
        "assets/walker2d.xml", "docs/rl_ladder.md",
    }
    assert expected_sources <= metadata["source_sha256"].keys()
    for relative, expected_hash in metadata["source_sha256"].items():
        assert evaluation.sha256(tiny_run / "source" / relative) == expected_hash
    assert (tiny_run / "source" / "pyproject.toml").is_file()
    assert (tiny_run / "source" / "uv.lock").is_file()


def test_saved_checkpoint_runs_local_evaluation(tiny_run, tmp_path):
    checkpoint = tiny_run / "checkpoints" / "step_0000012.pt"
    output = tmp_path / "evaluation.json"
    evaluation.main(checkpoint=checkpoint, output=output, episodes=2, steps=4, seed=50000)
    report = json.loads(output.read_text())
    assert report["training_step"] == 12
    assert report["checkpoint_sha256"] == evaluation.sha256(checkpoint)
    assert report["development_class"] == "unclassified"
    assert [episode["reset_seed"] for episode in report["episodes"]] == [50000, 50001]
    assert all(episode["steps"] == 4 for episode in report["episodes"])
    assert report["summary"]["velocity_mae"] >= 0
    assert report["summary"]["survival_rate"] == 1.0
    assert report["summary"]["mean_steps"] == 4.0
    assert all(np.isfinite(value) for value in report["summary"].values())
    _assert_trace_report(output.with_suffix(".npz"), report)


def test_monitoring_saves_reconstructable_raw_traces(tiny_run):
    for step in (0, 6, 12):
        report_path = tiny_run / "evaluations" / f"step_{step:07d}.json"
        report = json.loads(report_path.read_text())
        _assert_trace_report(report_path.with_suffix(".npz"), report)


def test_policy_evaluation_is_repeatable_without_consuming_rng_or_changing_weights(tiny_run):
    payload = torch.load(tiny_run / "checkpoints" / "step_0000012.pt", weights_only=True)
    actor = sac.load_policy(payload["policy"])
    before = copy.deepcopy(actor.state_dict())
    training_mode = actor.training
    torch_rng = torch.get_rng_state().clone()
    first = evaluation.evaluate_policy(actor, episodes=2, steps=4, seed=50000)
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    second = evaluation.evaluate_policy(actor, episodes=2, steps=4, seed=50000)
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    assert actor.training == training_mode
    for name, parameter in actor.state_dict().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    for a, b in zip(first["episodes"], second["episodes"]):
        assert {k: v for k, v in a.items() if k != "mean_action_seconds"} == {
            k: v for k, v in b.items() if k != "mean_action_seconds"
        }


@pytest.mark.parametrize("completed, mae, count, steps, expected", [
    (14, 0.1, 30, 1000, "weak"),
    (15, 0.1, 30, 1000, "intermediate"),
    (28, 0.1, 30, 1000, "intermediate"),
    (29, 0.2, 30, 1000, "strong"),
    (30, 0.2, 30, 1000, "strong"),
    (29, 0.200001, 30, 1000, "intermediate"),
    (29, 0.1, 29, 1000, "unclassified"),
    (30, 0.1, 30, 999, "unclassified"),
])
def test_development_classification_boundaries(completed, mae, count, steps, expected):
    report = {"episodes": [{"survived": i < completed} for i in range(count)],
              "summary": {"velocity_mae": mae}, "run": {"steps": steps}}
    assert evaluation.classify(report) == expected


def test_training_refuses_to_reuse_an_existing_run(tiny_run):
    original = (tiny_run / "status.json").read_bytes()
    with pytest.raises(FileExistsError):
        training.main(run_dir=tiny_run, total_steps=1)
    assert (tiny_run / "status.json").read_bytes() == original


def test_evaluation_refuses_to_overwrite_report(tiny_run, tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("existing evidence")
    checkpoint = tiny_run / "checkpoints" / "step_0000012.pt"
    original_checkpoint = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.main(checkpoint=checkpoint, output=output, episodes=1, steps=1)
    assert output.read_text() == "existing evidence"
    assert checkpoint.read_bytes() == original_checkpoint


@pytest.mark.parametrize("collision", ["checkpoint", "task", "report"])
def test_video_output_cannot_replace_inputs_or_report(tiny_run, tmp_path, collision):
    checkpoint = tiny_run / "checkpoints" / "step_0000012.pt"
    output = tmp_path / "report.json"
    video = {"checkpoint": checkpoint, "task": training.DEFAULT_TASK, "report": output}[collision]
    with pytest.raises(ValueError):
        evaluation.main(checkpoint=checkpoint, output=output, video=video, episodes=1, steps=1)
    assert not output.exists()


def test_evaluation_refuses_existing_video_before_rendering(tiny_run, tmp_path):
    checkpoint = tiny_run / "checkpoints" / "step_0000012.pt"
    output, video = tmp_path / "report.json", tmp_path / "existing.mp4"
    video.write_bytes(b"existing video")
    with pytest.raises(FileExistsError):
        evaluation.main(checkpoint=checkpoint, output=output, video=video, episodes=1, steps=1)
    assert video.read_bytes() == b"existing video"
    assert not output.exists()


def test_evaluation_rejects_a_different_task(tiny_run, tmp_path):
    checkpoint = tiny_run / "checkpoints" / "step_0000012.pt"
    task = json.loads(training.DEFAULT_TASK.read_text())
    task["target_velocity"] += 0.1
    changed_task, output = tmp_path / "task.json", tmp_path / "report.json"
    changed_task.write_text(json.dumps(task))
    with pytest.raises(ValueError, match="task hashes differ"):
        evaluation.main(checkpoint=checkpoint, output=output, task_config=changed_task,
                        episodes=1, steps=1)
    assert not output.exists()


def test_evaluation_rejects_a_different_physics_hash_before_running(tiny_run, tmp_path, monkeypatch):
    payload = torch.load(tiny_run / "checkpoints" / "step_0000012.pt", weights_only=True)
    payload["source_sha256"]["assets/walker2d.xml"] = "0" * 64
    checkpoint, output = tmp_path / "different_physics.pt", tmp_path / "report.json"
    torch.save(payload, checkpoint)

    def must_not_evaluate(*args, **kwargs):
        pytest.fail("A mismatched physics checkpoint must be rejected before evaluation")

    monkeypatch.setattr(evaluation, "evaluate_policy", must_not_evaluate)
    with pytest.raises(ValueError, match="physics differ: assets/walker2d.xml"):
        evaluation.main(checkpoint=checkpoint, output=output, episodes=1, steps=1)
    assert not output.exists()
    assert not output.with_suffix(".npz").exists()
