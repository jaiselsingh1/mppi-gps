"""Small independent update/continuation/precision checks, with a work census."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.gps.mppi_gps.full_pi_alternating import (
    Meter,
    arrays,
    context,
    health,
    instrument_environment,
    live,
    optimize,
    score_components,
)
from src.gps.mppi_gps.policy import Policy, fit_policy
from src.gps.mppi_gps.provenance import fingerprint
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

CENSUS = {}


@pytest.fixture(autouse=True)
def count_engineering(monkeypatch):
    from src.envs.mujoco_env import MuJoCoEnv
    from src.envs.walker2d import Walker2d

    def add(key, n=1):
        CENSUS[key] = CENSUS.get(key, 0) + n

    forward = Policy.forward

    def measured_forward(self, x):
        n = int(np.prod(x.shape[:-1])) or 1
        add("neural_forward_rows", n)
        CENSUS["last_forward_rows"] = n
        return forward(self, x)

    monkeypatch.setattr(Policy, "forward", measured_forward)
    step = torch.optim.Adam.step

    def measured_adam(self, *args, **kw):
        add("Adam_updates")
        return step(self, *args, **kw)

    monkeypatch.setattr(torch.optim.Adam, "step", measured_adam)
    backward = torch.Tensor.backward

    def measured_backward(self, *args, **kw):
        add("backward_rows", CENSUS.get("last_forward_rows", 0))
        return backward(self, *args, **kw)

    monkeypatch.setattr(torch.Tensor, "backward", measured_backward)
    batch = MuJoCoEnv.batch_rollout

    def measured_batch(self, state, u, **kw):
        n = u.shape[0] * u.shape[1]
        add("model_controls", n)
        add("physics_substeps", 4 * n)
        return batch(self, state, u, **kw)

    monkeypatch.setattr(MuJoCoEnv, "batch_rollout", measured_batch)
    native = Walker2d.step

    def measured_step(self, u):
        add("native_controls")
        add("physics_substeps", 4)
        return native(self, u)

    monkeypatch.setattr(Walker2d, "step", measured_step)
    yield
    if os.environ.get("PI_ENGINEERING_CENSUS"):
        Path(os.environ["PI_ENGINEERING_CENSUS"]).write_text(json.dumps(CENSUS, indent=2))


def limits():
    return dict(
        wall_seconds=600, cpu_seconds=600, output_bytes=256 * 1024**2, rss_bytes=4 * 1024**3
    )


class Zero:
    def predict(self, x, meter=None):
        return np.zeros((*np.shape(x)[:-1], 6), np.float32)

    def hash(self):
        return "zero-test"


class Toy:
    action_dim = 6
    action_bounds = (np.full(6, -1.0), np.full(6, 1.0))
    _dt = 0.008

    def batch_rollout(self, state, actions, **kwargs):
        k, h, _ = actions.shape
        posts = np.broadcast_to(state, (k, h, 19)).copy()
        posts[..., 2] = 1.25
        posts[..., 10] = 1.5 + 0.1 * actions[..., 0]
        costs = 10 + np.sum((actions - 0.2) ** 2, axis=(1, 2))
        return posts, costs, np.zeros((k, h, 0))


def toy_config():
    return dict(
        mppi=dict(
            K=8,
            H=6,
            lam=3.0,
            noise_sigma=0.3,
            initial_iterations=2,
            use_is_correction=False,
            lam_target_neff_fraction=0.05,
        )
    )


@pytest.mark.parametrize("alpha", [0.0, 0.12])
def test_exact_prototype_nominal_evolution(tmp_path, alpha):
    cfg = toy_config()
    init = dict(state=np.zeros(19), warmstart=np.zeros(9))
    init["state"][2] = 1.25
    U = np.full((6, 6), 0.1)
    rng = np.random.default_rng(123).bit_generator.state
    meter = Meter(tmp_path / "meter", limits())
    selected, _ = optimize(
        Toy(), meter, tmp_path / "solve", cfg, init, U, Zero(), alpha, rng, 4, 100.0
    )

    class Original(MPPI):
        def _sample_noise(self):
            noise = super()._sample_noise() * self.scale
            noise[0] = 0.0
            return noise

    original = Original(Toy(), MPPIConfig(**cfg["mppi"]), seed=123)
    original.U = U.copy()
    with np.load(tmp_path / "solve/iterations.npz") as saved:
        costs = []
        for i in range(4):
            original.scale = 0.1 ** (i / 3)
            current = original.U.copy()

            def prior(posts, actions):
                # Independent constant policy: D is .5 sum(actions**2).
                return 0.5 * alpha * np.sum(actions**2, axis=(1, 2))

            original._plan_iteration(init["state"], prior, None, init["warmstart"])
            np.testing.assert_array_equal(saved[f"p{i:03d}_U"], current)
            np.testing.assert_array_equal(saved[f"p{i:03d}_output_U"], original.U)
            np.testing.assert_array_equal(saved[f"p{i:03d}_weights"], original._last_weights)
            costs.append(float(original._last_costs[0] + 0.5 * alpha * np.sum(current**2)))
    assert selected["iteration"] == int(np.argmin(costs))
    assert (tmp_path / "solve/unscored_final_mean.npz").exists()
    assert original._initial_plan  # No plan_step or horizon shift occurred.
    meter.journal.close()


def test_score_uses_pre_action_and_fall_causer_without_rounding():
    start = np.zeros(19)
    start[2] = 1.25
    posts = np.broadcast_to(start, (1, 3, 19)).copy()
    posts[0, 1, 2] = 0.7

    class Half(Zero):
        def predict(self, x, meter=None):
            return np.full((*np.shape(x)[:-1], 6), 0.5, np.float32)

    controls = np.full((1, 3, 6), 0.5 + 1e-10)
    d, n, _ = score_components(start, posts, controls, Half(), None)
    assert n.tolist() == [2]
    assert d[0] > 0
    assert np.isclose(d[0], 6e-20, rtol=1e-5, atol=0)
    assert not health(posts)[0, 1]


def test_continued_adam_and_paired_sampling(tmp_path):
    torch.set_num_threads(1)
    x = np.arange(20 * 17, dtype=np.float32).reshape(20, 17) / 100
    u = np.zeros((20, 6), np.float32)
    parent = Policy(7)
    meter = Meter(tmp_path / "meter", limits())
    p, state, _ = fit_policy(
        parent, x, u, updates=3, eta_theta=0.0, prox_reduction="mean", meter=meter, seed=4
    )
    records = []
    child, continued, _ = fit_policy(
        p,
        x,
        u + 0.1,
        state,
        updates=4,
        eta_theta=0.0,
        prox_reduction="mean",
        meter=meter,
        index_observer=lambda v: records.append(v["indices"]),
    )
    expected = np.random.default_rng()
    expected.bit_generator.state = copy.deepcopy(state["rng"])
    np.testing.assert_array_equal(records, expected.integers(0, 20, size=(4, 256)))
    assert continued["updates"] == 7
    assert all(int(v["step"]) == 7 for v in continued["adam"]["state"].values())
    assert child.normalization.keys() == parent.normalization.keys()
    assert state["policy_hash"] == p.hash()
    assert fingerprint(state) != fingerprint(continued)
    assert meter.completed["optimizer_steps"] == 7
    assert meter.completed["backward_rows"] == 7 * 256
    meter.journal.close()


def test_durable_unknown_and_budget_rejection(tmp_path):
    meter = Meter(tmp_path, limits() | dict(model_controls=3))
    with pytest.raises(RuntimeError, match="ceiling"):
        meter.paid(dict(model_controls=4), lambda: None)
    assert not meter.requested
    with pytest.raises(ValueError):
        meter.paid(dict(model_controls=2), lambda: (_ for _ in ()).throw(ValueError("test")))
    assert meter.unknown_partial == dict(model_controls=2)
    rows = [json.loads(s) for s in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["request", "error"]
    meter.journal.close()


def test_independent_fall_cost_and_missing_tail():
    from src.gps.mppi_gps.full_pi_alternating_audit import physical_cost

    posts = np.zeros((1, 5, 19))
    posts[..., 2] = 1.25
    posts[..., 10] = 1.5
    posts[:, 2:, 2] = 0.7
    controls = np.zeros((1, 5, 6))
    cost, walking, active = physical_cost(posts, controls)
    np.testing.assert_array_equal(active, [[True, True, True, False, False]])
    np.testing.assert_array_equal(cost, [35.0])  # native5 + terminal20 + two missing*5
    assert not walking[0]


def test_supervisor_retains_peer_failure(tmp_path):
    import hashlib

    run = tmp_path / "run"
    source = run / "source/src/gps/mppi_gps"
    source.mkdir(parents=True)
    fake = source / "full_pi_alternating.py"
    fake.write_text("import sys,time\nif sys.argv[-1]=='1': raise SystemExit(1)\ntime.sleep(30)\n")
    (run / "config.json").write_text(json.dumps(dict(total_limits=dict(execution_wall_seconds=5))))
    (run / "source_manifest.json").write_text(
        json.dumps(
            {
                "src/gps/mppi_gps/full_pi_alternating.py": dict(
                    bytes=fake.stat().st_size, sha256=hashlib.sha256(fake.read_bytes()).hexdigest()
                )
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/studies/run_full_pi_alternating.py",
            "--run",
            str(run),
            "--commit",
            "synthetic",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    receipt = json.loads((run / "coordinator_result.json").read_text())
    assert receipt["stopped"] and 1 in receipt["exit_codes"]
    assert (run / "files_manifest.json").exists()


@pytest.mark.skipif(
    not os.environ.get("PI_NATIVE_FIXTURE"), reason="Explicit native validity fixture"
)
def test_complete_native_initializer_and_zero_residual(tmp_path):
    cfg = json.loads(Path("configs/studies/full_pi_alternating_v1.json").read_text())
    meter = Meter(tmp_path / "meter", limits())
    cfg["task"]["nthread"] = 2
    env = instrument_environment(cfg, meter)
    with np.load(os.environ["PI_NATIVE_FIXTURE"], allow_pickle=False) as src:
        data = {k: src[k] for k in ("states", "actions", "warmstarts", "integrations", "costs")}
        for key, terminal in (
            ("states", "terminal_state"),
            ("warmstarts", "terminal_warmstarts"),
            ("integrations", "terminal_integrations"),
        ):
            data[key] = np.concatenate((data[key], src[terminal][None]))
    actual, outcome = live(
        env, meter, tmp_path / "first.npz", tape=data["actions"], initial=context(data)
    )
    assert outcome["walking"]
    np.testing.assert_allclose(actual["states"], data["states"], rtol=0, atol=1e-10)
    posts, costs, _ = env.batch_rollout(
        data["states"][0], data["actions"][None], initial_warmstart=data["warmstarts"][0]
    )
    np.testing.assert_allclose(posts[0], actual["states"][1:], rtol=0, atol=1e-10)
    assert abs(costs[0] - outcome["cost"]) < 1e-10
    from src.gps.mppi_gps.full_pi_alternating_audit import physical_cost

    independent_cost, independent_walk, _ = physical_cost(posts, data["actions"][None])
    assert independent_walk[0]
    np.testing.assert_allclose(independent_cost, [outcome["cost"]], rtol=0, atol=1e-10)

    class Table:
        lookup = {
            o.tobytes(): u
            for o, u in zip(actual["observations"][:-1], actual["actions"], strict=True)
        }

        def predict(self, x, meter=None):
            return np.stack([self.lookup[o.tobytes()] for o in x])

    # Table returns physical f64 actions: do not round them to the real actor's
    # FP32 schema. This fixture establishes deterministic replay, not learnability.
    table = Table()
    D, lengths, _ = score_components(
        actual["states"][0], posts, actual["actions"][None], table, None
    )
    np.testing.assert_array_equal(D, [0.0])
    np.testing.assert_array_equal(lengths, [1000])
    for o, u in zip(actual["observations"][:-1], actual["actions"], strict=True):
        np.testing.assert_array_equal(table.predict(o[None])[0], u)
    again, outcome2 = live(
        env, meter, tmp_path / "again.npz", tape=actual["actions"], initial=context(actual)
    )
    assert outcome2["walking"]
    np.testing.assert_array_equal(again["states"], actual["states"])
    arrays(tmp_path / "zero_residual.npz", residual=np.zeros_like(actual["actions"]))
    env.close()
    meter.journal.close()
