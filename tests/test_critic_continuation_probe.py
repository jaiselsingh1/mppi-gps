"""Independent accounting checks for the frozen-critic Monte Carlo diagnostic."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.diagnostics import critic_continuation_probe as probe
from src.rl.sac import SACAgent, SACConfig
from src.rl.walker import DEFAULT_TASK, WalkerRL


class FakeEnv:
    """Reward schedule with a real absorbing-tail convention and exact restores."""

    def __init__(self, costs, *, gamma, horizon, fall_at=None):
        self.costs, self.gamma, self.horizon, self.fall_at = costs, gamma, horizon, fall_at
        self.walker = SimpleNamespace(get_state=lambda: self.state.copy(), get_warmstart=lambda: self.warm.copy())

    def reset(self, *, options):
        self.state, self.warm = options["state"].copy(), options["warmstart"].copy()
        self.step_index = 0
        return np.zeros(17), {}

    def step(self, action):
        self.step_index += 1
        cost = self.costs[self.step_index - 1]
        terminated = self.step_index == self.fall_at
        truncated = self.step_index == self.horizon and not terminated
        tail = -self.gamma * 5 / (1 - self.gamma) if terminated else 0.0
        return np.zeros(17), -cost + tail, terminated, truncated, {"task_cost": cost, "failure_tail": tail}


class FakeActor:
    def __init__(self, log_prob=-2.0):
        self.log_prob, self.calls = log_prob, 0

    def get_action(self, observation):
        self.calls += 1
        action = torch.randn((1, 6)) * .01
        return action, torch.tensor([[self.log_prob]]), torch.zeros((1, 6))


def _rollout(env, actor, **kwargs):
    return probe.rollout_soft_return(env, actor, np.zeros(19), np.zeros(9), np.zeros(6),
                                     gamma=env.gamma, alpha=.25, max_steps=env.horizon, seed=77, **kwargs)


def test_fixed_action_has_no_entropy_and_fall_tail_is_charged_only_once():
    actor = FakeActor()
    env = FakeEnv([2.0, 3.0], gamma=.5, horizon=5, fall_at=2)
    result = _rollout(env, actor)
    # r0=-2, r1=-3-5 (the exact gamma=.5 failure tail); entropy at t1 only.
    assert result["soft_return"] == pytest.approx(-2 + .5 * (-3 - 5 + .5))
    assert result["discounted_adjusted_rewards"] == pytest.approx(-6)
    assert result["discounted_entropy_bonus"] == pytest.approx(.25)
    assert result["failure_tail_charged_by_environment"] == pytest.approx(-5)
    assert result["partial_soft_returns"]["5"] == result["soft_return"]
    assert result["steps"] == 2 and result["terminated"] and not result["truncated"]
    assert result["residual_discount_weight"] == 0.0 and actor.calls == 1


def test_first_step_fall_never_samples_policy_or_bootstraps():
    actor = FakeActor(log_prob=float("nan"))
    result = _rollout(FakeEnv([5.0], gamma=.995, horizon=3, fall_at=1), actor)
    assert result["soft_return"] == pytest.approx(-1000.0)
    assert result["discounted_entropy_bonus"] == 0.0 and actor.calls == 0


def test_timeout_retains_missing_tail_and_torch_pairing_restores_caller_rng():
    torch.manual_seed(912)
    before = torch.get_rng_state().clone()
    result = _rollout(FakeEnv([1, 1, 1], gamma=.5, horizon=3), FakeActor())
    torch.testing.assert_close(torch.get_rng_state(), before)
    assert result["soft_return"] == pytest.approx(-1 + .5 * (-1 + .5) + .25 * (-1 + .5))
    assert result["truncated"] and not result["terminated"]
    assert result["residual_discount_weight"] == pytest.approx(.125)
    identical = _rollout(FakeEnv([1, 1, 1], gamma=.5, horizon=3), FakeActor())
    assert result == identical


def test_fall_wins_over_simultaneous_horizon():
    result = _rollout(FakeEnv([1, 1], gamma=.5, horizon=2, fall_at=2), FakeActor())
    assert result["terminated"] and not result["truncated"]
    assert result["residual_discount_weight"] == 0.0


def test_bad_restore_and_early_timeout_are_rejected():
    env = FakeEnv([1], gamma=.5, horizon=3)
    env.walker.get_state = lambda: np.ones(19)
    with pytest.raises(RuntimeError, match="restore"):
        _rollout(env, FakeActor())
    env = FakeEnv([1], gamma=.5, horizon=1)
    with pytest.raises(RuntimeError, match="cutoff"):
        probe.rollout_soft_return(env, FakeActor(), np.zeros(19), np.zeros(9), np.zeros(6),
                                 gamma=.5, alpha=.25, max_steps=3, seed=1)


def test_monte_carlo_error_and_pairing_are_computed_from_replicates():
    assert probe.mc_summary([1, 3])["mc_standard_error"] == pytest.approx(1.0)
    ranking = probe.paired_ranking([1, 3, 5], [0, 2, 4], q_delta=2)
    assert ranking["mc_delta"] == 1.0 and ranking["paired_mc_standard_error"] == 0.0
    assert ranking["empirical_ranking_agrees"]
    assert not probe.paired_ranking([1, 3], [0, 2], q_delta=-2)["empirical_ranking_agrees"]
    with pytest.raises(ValueError):
        probe.paired_ranking([1], [1, 2], q_delta=0)


def test_anchor_selection_depends_only_on_uniform_rows_and_checks_manifest(tmp_path):
    path = tmp_path / "snapshots.npz"
    states = np.arange(12 * 19, dtype=float).reshape(12, 19)
    np.savez(path, states=states, warmstarts=np.zeros((12, 9)), recorded_actions=np.zeros((12, 6)))
    sidecar = {"numeric_sha256": probe.sha256(path), "snapshots": [{"id": i} for i in range(12)]}
    path.with_suffix(".json").write_text(json.dumps(sidecar))
    anchors, inputs = probe.load_anchors(path, 8)
    indices = np.linspace(0, 11, 8, dtype=int)
    assert inputs["selected_rows"] == indices.tolist()
    np.testing.assert_array_equal(anchors[-1]["state"], states[-1])
    assert len(anchors) == 8 and anchors[0]["teacher_action"].shape == (6,)
    sidecar["numeric_sha256"] = "wrong"
    path.with_suffix(".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="hash"):
        probe.load_anchors(path, 8)


def test_trace_selection_excludes_post_terminal_and_interepisode_boundaries(tmp_path):
    path = tmp_path / "trace.npz"
    np.savez(path, states=np.zeros((7, 19)), warmstarts=np.zeros((7, 9)),
             episode_state_offsets=np.array([0, 3, 7]))
    anchors, _ = probe.load_trace_anchors((path,), 8)
    assert [a["input_row"] for a in anchors] == [0, 1, 3, 4, 5]
    assert all(a["teacher_action"] is None for a in anchors)


def test_tiny_real_environment_pipeline_is_frozen_and_refuses_overwrite(tmp_path):
    torch.set_num_threads(1)
    cfg = SACConfig(hidden_dim=8, batch_size=2)
    agent = SACAgent(17, -np.ones(6), np.ones(6), cfg)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"agent": agent.state_dict(), "policy": agent.policy_state(), "gamma": cfg.gamma,
                "task_sha256": probe.sha256(DEFAULT_TASK), "step": 0, "seed": 0,
                "config": {"mppi_stream": {"terminal": "actor"}}}, checkpoint)
    env = WalkerRL(max_episode_steps=3)
    try:
        env.reset(seed=123)
        anchors = tmp_path / "snapshots.npz"
        np.savez(anchors, states=env.walker.get_state()[None], warmstarts=env.walker.get_warmstart()[None],
                 recorded_actions=np.zeros((1, 6)))
    finally:
        env.close()
    before = {key: value.clone() for key, value in agent.actor.state_dict().items()}
    out = tmp_path / "diagnostic"
    report = probe.run(checkpoint=checkpoint, output_dir=out, anchors_path=anchors,
                       state_count=1, repeats=2, max_steps=3)
    assert report["source_unchanged"] and len(report["cases"]) == 1
    assert report["resource_ledger"]["continuation_controls"] <= 12
    assert report["resource_ledger"]["cold_teacher_search_controls"] == 0
    assert json.loads((out / "status.json").read_text())["complete"]
    assert json.loads((out / "protocol.json").read_text())["inputs"]["selected_rows"] == [0]
    for key, value in agent.actor.state_dict().items():
        torch.testing.assert_close(value, before[key])
    with pytest.raises(FileExistsError):
        probe.run(checkpoint=checkpoint, output_dir=out, anchors_path=anchors,
                  state_count=1, repeats=2, max_steps=3)


@pytest.mark.parametrize("kwargs", [{"state_count": 9}, {"repeats": 9}, {"max_steps": 2501}, {"state_count": 0}])
def test_protocol_hard_limits_precede_io(tmp_path, kwargs):
    with pytest.raises(ValueError):
        probe.run(checkpoint=Path("missing"), output_dir=tmp_path / "out", **kwargs)
