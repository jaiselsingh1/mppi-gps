"""Step 19: MPPI episodes as a SAC data stream, coupled to the actor only through MPPI's terminal cost."""
import json

import numpy as np
import pytest

from scripts.training.train_walker2d_sac import main as train
from scripts.training.train_walker_mdgps import _new_actor, policy_state
from src.mppi.mppi import MPPI
from src.rl.mppi_stream import collect_block, mppi_episode
from src.rl.walker import DEFAULT_TASK, WalkerRL


def _job(**overrides) -> dict:
    job = {"task_config": str(DEFAULT_TASK), "gamma": 0.995, "max_episode_steps": 1000, "K": 8, "H": 6,
           "tail_steps": 5, "policy": None, "reset_seed": 200_001, "planner_seed": 300_001, "cap": 12}
    return {**job, **overrides}


def test_mppi_transitions_carry_walkerrl_rewards():
    result = mppi_episode(_job())
    env = WalkerRL(DEFAULT_TASK, 0.995, 1000)
    obs, _ = env.reset(seed=200_001)
    for t in range(result["steps"]):
        np.testing.assert_array_equal(result["obs"][t], obs)
        obs, reward, terminated, _, _ = env.step(result["action"][t])
        assert reward == result["reward"][t] and terminated == result["terminated"][t]
        np.testing.assert_array_equal(result["next_obs"][t], obs)


def test_the_actor_reaches_mppi_only_as_its_terminal_cost(monkeypatch):
    calls = []
    original = MPPI.plan_step

    def spy(self, state, nominal=None, nominal_first=None, prior_cost=None, coupling=None, initial_warmstart=None):
        calls.append((nominal, nominal_first, prior_cost, coupling))
        return original(self, state, nominal, nominal_first, prior_cost, coupling, initial_warmstart)

    monkeypatch.setattr(MPPI, "plan_step", spy)
    with_actor = mppi_episode(_job(policy=policy_state(_new_actor(0)), cap=4))
    assert calls and all(n is None and f is None and c is None and p is not None for n, f, p, c in calls)
    calls.clear()
    alone = mppi_episode(_job(cap=4))
    assert all(p is None for _, _, p, _ in calls)
    assert alone["simulated_controls"] == 8 * 6 * (2 + 3)
    assert with_actor["simulated_controls"] > alone["simulated_controls"]


def test_a_block_executes_exactly_the_requested_controls():
    seeds = iter(range(1, 100))

    def next_seeds() -> dict:
        i = next(seeds)
        return {"reset_seed": 200_000 + i, "planner_seed": 300_000 + i}

    transitions, stats = collect_block(map, _job(), steps=23, episode_cap=10, next_seeds=next_seeds)
    assert len(transitions) == 23 and stats["steps"] == 23
    assert sum(e["steps"] for e in stats["per_episode"]) == 23
    assert all(e["steps"] <= 10 for e in stats["per_episode"])
    assert len({e["reset_seed"] for e in stats["per_episode"]}) == stats["episodes"]


def _short_run(path, **mppi) -> tuple[list[dict], dict, dict]:
    train(path, total_steps=400, learning_starts=50, save_every=400, eval_every=400, eval_episodes=1,
          batch_size=32, hidden_dim=32, buffer_size=1000, max_episode_steps=60, **mppi)
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    return rows, json.loads((path / "status.json").read_text()), json.loads((path / "config.json").read_text())


@pytest.mark.parametrize("terminal", ["none", "actor"])
def test_mppi_controls_count_against_the_budget(tmp_path, terminal):
    rows, status, config = _short_run(tmp_path / "run", mppi_share=0.25, mppi_block=100, mppi_terminal=terminal,
                                      mppi_K=8, mppi_H=6, mppi_tail_steps=5, mppi_episode_cap=20, mppi_workers=1)
    blocks = [r for r in rows if r["type"] == "mppi_block"]
    assert [b["step"] for b in blocks] == [75, 175, 275, 375] and all(b["steps"] == 25 for b in blocks)
    assert status["status"] == "complete" and status["step"] == 400 and status["updates"] == 350
    assert config["mppi_stream"]["terminal"] == terminal


def test_without_a_share_training_is_plain_sac(tmp_path):
    rows, status, config = _short_run(tmp_path / "run")
    assert not any(r["type"] == "mppi_block" for r in rows) and config["mppi_stream"] is None
    assert status["status"] == "complete" and status["updates"] == 350
