"""Bound score/nominal/replay fixtures; no native simulator or neural network."""

from __future__ import annotations

import h5py
import numpy as np

from src.gps.mppi_gps import coupling_warmstart_probe as module
from src.gps.mppi_gps import trajectory_pi_native as native
from src.gps.mppi_gps.provenance import fingerprint, write
from src.gps.mppi_gps.trajectory_pi_native import binding, load_trace


class FakeActor:
    """Synthetic bounded array function, not a Policy or Torch module."""

    obs_dim, hidden_dim = 17, 128
    observation_mean, observation_std = np.zeros(17), np.ones(17)

    def eval(self):
        return self

    def hash(self):
        return "synthetic-fixed-actor"

    def predict(self, observations, meter):
        count = len(observations)
        return meter.call(
            "synthetic_array_map",
            dict(neural_rows=count),
            lambda: np.zeros((count, 6), dtype=np.float32),
        )


class FakeEnv:
    """Array bookkeeping only. It implements no mechanics or integration."""

    action_dim, _dt = 6, 0.008
    action_bounds = np.full(6, -1.0), np.full(6, 1.0)

    def __init__(self, work=None):
        self.work = work
        self.state = np.zeros(19)
        self.state[2], self.state[10] = 1.2, 1.5
        self.warmstart = np.zeros(9)
        self.integration = np.zeros(91)

    def get_state(self):
        return self.state.copy()

    def get_warmstart(self):
        return self.warmstart.copy()

    def _get_obs(self):
        return module.observations(self.state)

    def batch_rollout(self, initial, actions, initial_warmstart=None):
        def arrays():
            count, horizon = actions.shape[:2]
            posts = np.broadcast_to(initial, (count, horizon, 19)).copy()
            posts[:, :, 0] += np.arange(1, horizon + 1)[None] * 0.008
            cost = (actions**2).sum((1, 2))
            return posts, cost, None

        count = int(np.prod(actions.shape[:2]))
        return self.work.call(
            "synthetic_forecast",
            dict(model_controls=count, model_queries=1, physics_substeps=count * 4),
            arrays,
        )

    def step(self, action):
        def arrays():
            self.state[0] += 0.008
            self.integration[0] += 1
            self.warmstart += 0.01
            fell = self.integration[0] >= 3
            if fell:
                self.state[2] = 0.7
            cost = float(np.sum(action**2)) + (20 if fell else 0)
            return self._get_obs(), cost, fell, dict(fall_cost=20.0 if fell else 0.0)

        if self.work is None:
            return arrays()
        return self.work.call(
            "synthetic_control", dict(native_controls=1, physics_substeps=4), arrays
        )

    def close(self):
        pass


def snapshot(env):
    return dict(
        state=env.state.copy(),
        warmstart=env.warmstart.copy(),
        integration=env.integration.copy(),
        spec=np.asarray(1),
        dimensions=np.asarray([9, 9, 6]),
    )


def restore(env, context, work):
    def assign():
        env.state = context["state"].copy()
        env.warmstart = context["warmstart"].copy()
        env.integration = context["integration"].copy()

    work.call("synthetic_restore", dict(restores=1, external_mj_forward_calls=1), assign)


def fixture_job(tmp_path, monkeypatch, kind="controller"):
    context = snapshot(FakeEnv())
    with (tmp_path / "context.npz").open("xb") as target:
        np.savez(target, **context)
    write(tmp_path / "config.json", dict(synthetic=True))
    write(tmp_path / "policy.pt", dict(synthetic=True))
    limits = dict(
        wall_seconds=60,
        output_bytes=64 * 1024**2,
        memory_bytes=4 * 1024**3,
        checkpoint_loads=1,
        policy_constructions=1,
        environment_constructions=1,
        model_controls=100000,
        physics_substeps=500000,
        model_queries=20,
        neural_rows=100000,
        native_controls=300,
        restores=3,
        external_mj_forward_calls=3,
    )
    job = dict(
        kind=kind,
        out=str(tmp_path / "out"),
        config=binding(tmp_path / "config.json"),
        checkpoint=binding(tmp_path / "policy.pt"),
        context=binding(tmp_path / "context.npz"),
        seed=1234,
        planner=dict(
            H=48, K=256, lam=3, noise_sigma=0.3, initial_iterations=2, use_is_correction=False
        ),
        task={},
        physics_threads=4,
        controls=300,
        alpha=1.2,
        initial_plan_mode="zero",
        initial_plan=None,
        limits=limits,
    )
    monkeypatch.setattr(module, "_load_actor", lambda _: FakeActor())
    monkeypatch.setattr(module, "snapshot", snapshot)
    monkeypatch.setattr(native, "snapshot", snapshot)
    monkeypatch.setattr(module, "_restore", restore)
    monkeypatch.setattr(
        module,
        "_env",
        lambda job, work: work.call(
            "synthetic_environment", dict(environment_constructions=1), lambda: FakeEnv(work)
        ),
    )
    return job, context


def test_mean_score_records_raw_sum_true_J_and_fall_prefix(tmp_path, monkeypatch):
    job, _ = fixture_job(tmp_path, monkeypatch)
    value = module.run_job(job)
    assert value["result"]["status"] == "COMPLETED", value["result"]
    result = value["result"]
    assert result["outcome"]["fell"] and result["outcome"]["controls"] == 3
    assert result["outcome"]["absorbing_tail_cost"] == 5 * 297
    assert result["banks"] == 4 and len(result["full_banks"]) == 2
    with h5py.File(result["bank_stats"]["path"], "r") as raw:
        for bank in raw.values():
            assert np.array_equal(bank["D"][:], bank["D_sum"][:])
            assert np.array_equal(bank["R"][:], bank["D_sum"][:] / bank["N"][:])
            assert np.array_equal(bank["J"][:], bank["C"][:] + job["alpha"] * bank["R"][:])
            assert bank.attrs["score_version"] == module.SCORE_VERSION
            assert len(bank.attrs["innovation_fingerprint"]) == 64
    trace = load_trace(native.Path(result["trace"]["path"]))
    assert np.array_equal(
        trace["action_gap_mse"], ((trace["actions"] - trace["policy_actions"]) ** 2).mean(1)
    )
    assert result["first_action_gap_mse"] == trace["action_gap_mse"][0]
    assert np.count_nonzero(trace["native_fall_cost"]) == 1
    assert value["receipt"]["counts"]["completed_native_controls"] == 3
    assert value["receipt"]["counts"]["completed_model_queries"] == 4
    assert value["receipt"]["counts"]["completed_neural_rows"] == 4 * 256 * 48 + 3


def test_same_seed_reuses_innovations_despite_different_initial_nominal(tmp_path, monkeypatch):
    job, context = fixture_job(tmp_path, monkeypatch)
    first = module.run_job(job)
    plan_path = tmp_path / "warm.npz"
    with plan_path.open("xb") as target:
        np.savez(
            target,
            U=np.full((48, 6), 0.1),
            target_context_fingerprint=np.asarray(fingerprint(context)),
            checkpoint_sha256=np.asarray(job["checkpoint"]["sha256"]),
            config_sha256=np.asarray(job["config"]["sha256"]),
        )
    warm = dict(
        job,
        out=str(tmp_path / "warm_out"),
        initial_plan_mode="shadow_carried",
        initial_plan=binding(plan_path),
    )
    second = module.run_job(warm)
    assert second["result"]["status"] == "COMPLETED", second["result"]
    for a, b in zip(first["result"]["full_banks"], second["result"]["full_banks"], strict=True):
        with np.load(a["path"]) as aa, np.load(b["path"]) as bb:
            assert np.array_equal(aa["innovation"], bb["innovation"])
            assert not np.array_equal(aa["U_before"], bb["U_before"])
            assert not np.array_equal(aa["actions"], bb["actions"])
    assert second["result"]["initial_two_passes"]


def test_alpha_contrast_shares_first_bank_not_later_updated_candidate_banks(tmp_path, monkeypatch):
    job, _ = fixture_job(tmp_path, monkeypatch)
    zero = module.run_job(dict(job, alpha=0))
    positive = module.run_job(dict(job, out=str(tmp_path / "positive"), alpha=30))
    assert zero["result"]["status"] == positive["result"]["status"] == "COMPLETED"
    first, second = [], []
    for target, value in ((first, zero), (second, positive)):
        for spec in value["result"]["full_banks"]:
            with np.load(spec["path"]) as raw:
                target.append({name: raw[name].copy() for name in raw.files})
    for key in ("actions", "posts", "C", "innovation", "U_before"):
        assert np.array_equal(first[0][key], second[0][key])
    assert np.array_equal(first[0]["J"], first[0]["C"])
    assert not np.array_equal(first[0]["weights"], second[0]["weights"])
    assert np.array_equal(first[1]["innovation"], second[1]["innovation"])
    # Once alpha changes pass0's nominal, common innovations no longer imply
    # identical candidates. That distinction must be visible in raw evidence.
    assert not np.array_equal(first[1]["U_before"], second[1]["U_before"])
    assert not np.array_equal(first[1]["actions"], second[1]["actions"])


def source_fixture(tmp_path, context):
    env = FakeEnv()
    arrays = dict(states=[], warmstarts=[], integrations=[], observations=[], actions=[], costs=[])
    for control in range(4):
        for key, value in (
            ("states", env.state),
            ("warmstarts", env.warmstart),
            ("integrations", env.integration),
            ("observations", env._get_obs()),
        ):
            arrays[key].append(value.copy())
        if control < 3:
            action = np.full(6, 0.2)
            _, cost, _, _ = env.step(action)
            arrays["actions"].append(action)
            arrays["costs"].append(cost)
    path = tmp_path / "source.h5"
    with h5py.File(path, "x") as target:
        for key, value in arrays.items():
            target.create_dataset(key, data=np.asarray(value, dtype=np.float64))
    current = dict(
        context,
        state=arrays["states"][2],
        warmstart=arrays["warmstarts"][2],
        integration=arrays["integrations"][2],
    )
    target_context = tmp_path / "target_context.npz"
    with target_context.open("xb") as target:
        np.savez(target, **current)
    return binding(path), binding(target_context), current


def test_shadow_preparation_replays_one_student_action_and_shifts_query_plan_once(
    tmp_path, monkeypatch
):
    job, context = fixture_job(tmp_path, monkeypatch, "prepare_warm")
    source, target, target_context = source_fixture(tmp_path, context)
    job.update(source_trace=source, context=target, control=2)
    value = module.run_job(job)
    result = value["result"]
    assert result["status"] == "COMPLETED", result
    assert not result["recommendation_was_executed"] and result["shadow_plan_shift_count"] == 1
    assert value["receipt"]["counts"]["completed_native_controls"] == 1
    assert value["receipt"]["counts"]["completed_model_queries"] == 2
    with (
        np.load(result["full_banks"][1]["path"]) as last,
        np.load(result["initial_plan"]["path"]) as plan,
    ):
        expected = last["U_after"].copy()
        expected[:-1] = expected[1:]
        expected[-1] = expected[-2]
        assert np.array_equal(expected, plan["U"])
        assert str(plan["target_context_fingerprint"]) == fingerprint(target_context)
    assert max(result["source_replay_max_errors"].values()) == 0
    assert result["actual_student_transition_action"] != result["shadow_recommendation"]


def test_bad_source_replay_stops_before_shadow_forecast(tmp_path, monkeypatch):
    job, context = fixture_job(tmp_path, monkeypatch, "prepare_warm")
    source, target, _ = source_fixture(tmp_path, context)
    with h5py.File(source["path"], "r+") as raw:
        raw["costs"][1] += 0.01
    job.update(source_trace=binding(native.Path(source["path"])), context=target, control=2)
    value = module.run_job(job)
    assert value["result"]["status"] == "FAILED_STOPPED"
    assert "cost replay mismatch" in value["result"]["error"]
    assert value["receipt"]["counts"]["completed_native_controls"] == 1
    assert value["receipt"]["counts"].get("requested_model_queries", 0) == 0


def test_work_cap_stops_before_forecast_and_preserves_paid_prefix(tmp_path, monkeypatch):
    job, _ = fixture_job(tmp_path, monkeypatch)
    job["limits"]["model_controls"] = 0
    value = module.run_job(job)
    assert value["result"]["status"] == "FAILED_STOPPED"
    assert "model_controls" in value["result"]["error"]
    assert value["receipt"]["counts"].get("requested_model_controls", 0) == 0
    assert value["receipt"]["counts"]["completed_neural_rows"] == 1
    assert native.Path(value["result"]["trace"]["path"]).is_file()


def test_shadow_plan_binding_rejects_other_target_before_native_execution(tmp_path, monkeypatch):
    job, context = fixture_job(tmp_path, monkeypatch)
    plan = tmp_path / "wrong.npz"
    with plan.open("xb") as target:
        np.savez(
            target,
            U=np.zeros((48, 6)),
            target_context_fingerprint=np.asarray("wrong"),
            checkpoint_sha256=np.asarray(job["checkpoint"]["sha256"]),
            config_sha256=np.asarray(job["config"]["sha256"]),
        )
    job.update(initial_plan_mode="shadow_carried", initial_plan=binding(plan))
    value = module.run_job(job)
    assert value["result"]["status"] == "FAILED_STOPPED"
    assert "target, checkpoint or protocol binding" in value["result"]["error"]
    assert value["receipt"]["counts"].get("requested_native_controls", 0) == 0
    assert fingerprint(context) != "wrong"


def test_prefix_length_includes_fall_action_without_implying_survival():
    raw = np.asarray([12.0, 6.0])
    masks = np.asarray([[True, True, True], [True, False, False]])
    mean, lengths = module.prefix_mean(raw, masks)
    assert np.array_equal(lengths, [3, 1]) and np.array_equal(mean, [4, 6])
    # A last-action fall retains all3 attempted actions and is uncensored,
    # while full_survival remains false. The worker stores these separately.
    posts = np.zeros((1, 3, 19))
    posts[:, :, 2] = 1.2
    posts[:, -1, 2] = 0.7
    assert not module.healthy(posts).all()


def test_module_import_is_lazy():
    import subprocess
    import sys

    value = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import src.gps.mppi_gps.coupling_warmstart_probe; "
                "assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert value.returncode == 0, value.stderr
