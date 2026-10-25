"""Independent-oracle checks for the bounded raw-teacher diagnostic."""
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.diagnostics.independent_teacher_probe import (
    InstrumentedMPPI, collect_teacher, load_checkpoint, query_teacher, replay_difference,
    rollout_forced_action, save_snapshots, score_full_plans, teacher_config, validate_protocol,
)
from scripts.training.train_walker_mdgps import policy_state
from src.gps.policy import TanhMLPPolicy
from src.mppi.mppi import MPPI
from src.rl.walker import WalkerRL


@pytest.fixture
def walker_snapshot():
    env = WalkerRL(max_episode_steps=30)
    obs, _ = env.reset(seed=8_900_000)
    cfg = replace(teacher_config(), K=4, H=6)
    planner = InstrumentedMPPI(env.walker, cfg, seed=8_910_000)
    snapshot = {"id": 0, "episode": 0, "teacher_t": 0, "reset_seed": 8_900_000,
                "planner_seed": 8_910_000, "state": env.walker.get_state(),
                "warmstart": env.walker.get_warmstart(), "obs": obs, "planner": planner.snapshot()}
    yield env, cfg, planner, snapshot
    env.close()


def test_final_pass_nominal_is_the_first_pass_update_not_initial_zero(walker_snapshot):
    env, cfg, observed, snapshot = walker_snapshot
    reference = MPPI(env.walker, cfg, seed=8_910_000)
    reference._plan_iteration(snapshot["state"], None, None, snapshot["warmstart"])
    independently_computed_second_nominal = reference.U.copy()
    reference_action, _ = reference._plan_iteration(snapshot["state"], None, None, snapshot["warmstart"])
    result = query_teacher(env, observed, snapshot)
    assert result["optimization_iterations"] == 2
    np.testing.assert_array_equal(result["pass_nominals"][0], np.zeros((6, 6)))
    assert np.max(np.abs(independently_computed_second_nominal)) > 0
    np.testing.assert_array_equal(result["final_pass_nominal"], independently_computed_second_nominal)
    np.testing.assert_array_equal(result["action"], reference_action)
    np.testing.assert_allclose(result["mean_plan"],
                               np.einsum("k,kha->ha", result["sample_weights"], result["sample_plans"]), atol=1e-15)


def test_snapshot_replays_entire_samples_after_actor_loading_external_rollout_and_rng_changes(walker_snapshot, tmp_path):
    env, _, planner, snapshot = walker_snapshot
    reference = query_teacher(env, planner, snapshot)
    actor = TanhMLPPolicy(17, *env.walker.action_bounds)
    checkpoint = tmp_path / "actor.pt"
    torch.save(policy_state(actor), checkpoint)
    loaded = load_checkpoint(checkpoint)
    action = loaded.act(snapshot["obs"], deterministic=True)
    rollout_forced_action(snapshot, action, loaded, total_steps=8, short_steps=6)
    np.random.seed(53)
    np.random.normal(size=200)
    torch.manual_seed(74)
    torch.rand(50)
    repeated = query_teacher(env, planner, snapshot)
    differences = replay_difference(reference, repeated)
    assert differences["exact"]
    assert all(value == 0 for key, value in differences.items() if key != "exact")


def test_all_full_plans_use_identical_fullstate_warmstart_and_match_individual_rollouts(walker_snapshot):
    env, _, planner, snapshot = walker_snapshot
    query = query_teacher(env, planner, snapshot)
    report, plans = score_full_plans(env, snapshot, query)
    independently_scored = [env.walker.batch_rollout(snapshot["state"], plan[None],
                                                     initial_warmstart=snapshot["warmstart"])[1].item()
                            for plan in plans]
    np.testing.assert_array_equal(list(report["costs"].values()), independently_scored)
    assert report["best_sample_rescore_difference"] == 0.
    assert report["mean_minus_nominal"] == independently_scored[0] - independently_scored[1]


def test_snapshot_files_roundtrip_planner_rng_and_physics_exactly(walker_snapshot, tmp_path):
    env, _, planner, snapshot = walker_snapshot
    query = query_teacher(env, planner, snapshot)
    snapshot.update(recorded_action=query["action"], recorded_sample_costs=query["sample_costs"],
                    optimization_iterations=2)
    save_snapshots(tmp_path, [snapshot])
    metadata = json.loads((tmp_path / "snapshots.json").read_text())["snapshots"][0]
    with np.load(tmp_path / "snapshots.npz", allow_pickle=False) as data:
        restored = {"state": data["states"][0], "warmstart": data["warmstarts"][0],
                    "planner": {"U": data["planner_U"][0], "initial_plan": bool(data["initial_plan"][0]),
                                "rng_state": metadata["rng_state"]}}
    assert replay_difference(query, query_teacher(env, planner, restored))["exact"]


class ConstantActor:
    def act(self, obs, deterministic=True):
        return np.array([0.25])


class FakeWalker:
    _v_target, _dt, _w_unhealthy = 1.5, .008, 5.

    def __init__(self):
        self.t = 0

    def get_state(self):
        return np.array([self.t])

    def get_warmstart(self):
        return np.array([self.t + 1])


class OracleEnv:
    """Fall at step2 costs27 incl its terminal20; unused controls cost5."""
    fall_at = 2

    def __init__(self, max_episode_steps):
        self.limit = max_episode_steps
        self.walker = FakeWalker()

    def reset(self, options):
        assert options["state"].item() == 0
        assert options["warmstart"].item() == 1
        return np.zeros(1), {}

    def step(self, action):
        self.walker.t += 1
        fell = self.walker.t == self.fall_at
        cost = 27. if fell else float(action.item())
        return np.zeros(1), -cost - (995. if fell else 0.), fell, self.walker.t == self.limit and not fell, {
            "task_cost": cost, "x_velocity": 1.5}

    def close(self):
        pass


def test_forced_action_then_named_actor_finite_cost_has_one_terminal_charge_and_no_reward_tail():
    snapshot = {"state": np.zeros(1), "warmstart": np.ones(1)}
    report, trace = rollout_forced_action(snapshot, np.array([2.]), ConstantActor(),
                                          total_steps=6, short_steps=3, env_factory=OracleEnv)
    np.testing.assert_array_equal(trace["actions"], [[2.], [.25]])
    assert report["short"]["finite_horizon_task_cost"] == 2. + 27. + 5.
    assert report["long"]["finite_horizon_task_cost"] == 2. + 27. + 4 * 5.
    assert report["long"]["executed_controls"] == 2
    assert report["long"]["fell"] and not report["long"]["survived"]
    assert report["long"]["time_alive_seconds"] == .016
    assert report["long"]["velocity_mae"] == 0.


def test_healthy_administrative_cutoff_is_not_a_fall_or_absorbing_cost():
    class HealthyEnv(OracleEnv):
        fall_at = -1

    snapshot = {"state": np.zeros(1), "warmstart": np.ones(1)}
    report, _ = rollout_forced_action(snapshot, np.array([2.]), ConstantActor(),
                                      total_steps=6, short_steps=3, env_factory=HealthyEnv)
    for horizon in ("short", "long"):
        assert report[horizon]["survived"]
        assert not report[horizon]["fell"]
        assert report[horizon]["absorbing_cost"] == 0.
    assert report["long"]["finite_horizon_task_cost"] == 2. + 5 * .25
    assert report["short"]["finite_horizon_task_cost"] == 2. + 2 * .25
    assert report["administrative_cutoff"]


def test_teacher_collection_never_loads_an_actor_and_keeps_missing_anchors(monkeypatch):
    from scripts.diagnostics import independent_teacher_probe as probe

    def forbidden(*args, **kwargs):
        raise AssertionError("Actor used during independent teacher collection")

    monkeypatch.setattr(probe, "load_checkpoint", forbidden)
    snapshots, rows, missing = collect_teacher(1, 2, 8_900_000, 8_910_000, (0, 1),
                                               replace(teacher_config(), K=4, H=6))
    assert len(snapshots) == 2
    assert not missing
    assert rows[0]["healthy_at_collection_cutoff"]
    assert rows[0]["administrative_cutoff"]
    assert not rows[0]["fell"]
    # A deliberately forced early fall demonstrates missing planned anchors,
    # without altering/replacing the reset or asking for another episode.
    original = WalkerRL.step

    def early_fall(self, action):
        obs, reward, _, _, info = original(self, action)
        return obs, reward, True, False, info

    monkeypatch.setattr(WalkerRL, "step", early_fall)
    snapshots, rows, missing = collect_teacher(1, 2, 8_900_000, 8_910_000, (0, 1),
                                               replace(teacher_config(), K=4, H=6))
    assert len(snapshots) == 1 and rows[0]["fell"]
    assert missing == [{"episode": 0, "teacher_t": 1, "reason": "teacher_fell_before_anchor"}]


@pytest.mark.parametrize("seed,episodes", [(5_000_000, 1), (5_000_199, 1), (4_999_999, 2), (5_000_198, 3)])
def test_reserved_final_reset_panel_is_rejected_including_partial_overlap(seed, episodes):
    with pytest.raises(ValueError, match="reserved"):
        validate_protocol(episodes, 300, seed, 8_910_000, (0, 75, 150, 225))


def test_independent_teacher_rejects_policy_seed_or_tail(walker_snapshot):
    _, _, planner, snapshot = walker_snapshot
    for kwargs in ({"nominal": np.zeros((6, 6))}, {"nominal_first": np.zeros(6)},
                   {"prior_cost": lambda *args: 0.}, {"coupling": lambda *args: 0.}):
        with pytest.raises(ValueError, match="no policy inputs"):
            planner.plan_step(snapshot["state"], **kwargs)


def test_cli_budget_cannot_expand_to_an_unbounded_anchor_study():
    with pytest.raises(ValueError, match="four anchors"):
        validate_protocol(3, 300, 8_900_000, 8_910_000, (0, 1, 2, 3, 4))
