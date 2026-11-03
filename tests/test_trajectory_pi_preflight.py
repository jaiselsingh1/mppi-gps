"""Synthetic P0 contract checks; no native physics or real policy fitting."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.mppi_gps import trajectory_pi_native as native
from src.gps.mppi_gps import trajectory_pi_preflight as preflight
from src.gps.mppi_gps import trajectory_pi_student as student
from src.gps.mppi_gps.provenance import fingerprint, sha
from src.gps.mppi_gps.trajectory_pi_accounting import Work


def _row(root: Path, name: str) -> dict:
    path = root / name
    return dict(path=name, bytes=path.stat().st_size, sha256=sha(path))


def _work(tmp_path: Path) -> Work:
    return Work(tmp_path, preflight._limits())


def test_roster_verifies_exact_bytes_and_rejects_mutation(tmp_path):
    (tmp_path / "a").write_bytes(b"old")
    rows = [_row(tmp_path, "a")]
    assert preflight._roster(tmp_path, rows)["a"] == (tmp_path / "a").resolve()
    (tmp_path / "a").write_bytes(b"new")
    with pytest.raises(ValueError, match="binding"):
        preflight._roster(tmp_path, rows)


@pytest.mark.parametrize("name", ["../escape", "/tmp/absolute", "a/../escape", "a//b", "./a"])
def test_roster_rejects_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        preflight._roster(tmp_path, [dict(path=name, bytes=0, sha256="0" * 64)])


def test_roster_rejects_duplicate_and_noncanonical_order(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(ValueError, match="repeated"):
        preflight._roster(tmp_path, [_row(tmp_path, "a")] * 2)
    with pytest.raises(ValueError, match="sorted"):
        preflight._roster(tmp_path, [_row(tmp_path, "b"), _row(tmp_path, "a")])


def test_comparisons_preserve_signed_zero_and_frozen_tolerance():
    assert preflight._comparison([1.0], [1.0 + 1e-11], "near")["passed"]
    with pytest.raises(ValueError, match="comparison failed"):
        preflight._comparison([1.0], [1.0 + 1e-8], "far")
    with pytest.raises(ValueError, match="comparison failed"):
        preflight._comparison(np.array([0.0]), np.array([-0.0]), "signed", exact=True)


def test_fall_accounting_does_not_duplicate_terminal_or_invent_states():
    result = preflight._fall_accounting()
    assert result["controls"] == 2
    assert result["complete_cost"] == 1 + 21 + 3 * 5
    assert not result["walking"] and not result["full_survival"]


def test_budget_enforcement_is_before_entry_and_keeps_failed_receipt(tmp_path):
    result = preflight._budget_enforcement(tmp_path)
    assert result["rejected_before_entry"] and result["synthetic_failure_retained"]
    assert result["actual_physics"] == result["actual_fitting"] == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "budget_fixture/ledger.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in rows] == ["rejected", "requested", "failed", "closed"]
    assert result["fixture_work"] == dict(
        requested_fixture_units=1, unknown_partial_fixture_units=1
    )


def _synthetic_event(alpha):
    initial = np.zeros(19)
    initial[2] = 1.2
    states = np.broadcast_to(initial, (2, 3, 19)).copy()
    actions = np.array([np.full((3, 6), 0.1), np.full((3, 6), 0.2)])
    costs = np.array([1.0, 2.0])
    d = 0.5 * np.sum((actions - 0.25) ** 2, axis=(1, 2))
    scores = costs + alpha * d
    weights = np.exp(-(scores - scores.min()))
    weights /= weights.sum()
    nominal = actions[0]
    update = nominal + np.einsum("k, kha -> ha", weights, actions - nominal[None])
    event = dict(
        iteration=0,
        states=states,
        actions=actions,
        current_actions=nominal,
        C=costs,
        D=d if alpha else None,
        J=scores,
        weights=weights,
        post_actions=update,
        **{"lambda": 1.0},
    )
    env = SimpleNamespace(
        action_bounds=(-1.0, 1.0),
        _is_healthy=lambda z, angle: (z > 0.8) & (z < 2) & (np.abs(angle) < 1),
    )
    actor = SimpleNamespace(
        predict=lambda values, meter=None: np.full((len(values), 6), 0.25, dtype=np.float32)
    )
    return event, env, actor, dict(state=initial, warmstart=np.zeros(9))


@pytest.mark.parametrize("alpha", [0.0, 0.12])
def test_manual_score_uses_pre_action_state_total_objective_and_unshifted_update(tmp_path, alpha):
    event, env, actor, context = _synthetic_event(alpha)
    # Sum per-control first, exactly as the executable reconstruction contract.
    if alpha:
        event["D"] = np.sum(0.5 * np.sum((event["actions"] - 0.25) ** 2, axis=-1), axis=1)
        event["J"] = event["C"] + alpha * event["D"]
    work = _work(tmp_path)
    result = preflight._manual_bank(event, env, actor, context, alpha, work)
    assert all(row["passed"] for row in result["comparisons"])
    shifted = deepcopy(event)
    shifted["post_actions"] = np.roll(event["post_actions"], 1, axis=0)
    shifted["post_actions"][0, 0] += 0.01
    with pytest.raises(ValueError, match="without_shift"):
        preflight._manual_bank(shifted, env, actor, context, alpha, work)
    work.close("COMPLETED")


def test_replay_bank_requires_identical_context_and_sampled_controls(tmp_path):
    event, env, _, context = _synthetic_event(0.0)
    work = _work(tmp_path)
    replay = preflight._ReplayBank(env, [event], context, work)
    states, costs, sensors = replay.batch_rollout(
        context["state"], event["actions"], initial_warmstart=context["warmstart"]
    )
    assert np.array_equal(states, event["states"]) and np.array_equal(costs, event["C"])
    assert sensors.shape == (2, 3, 0)
    with pytest.raises(ValueError, match="extra bank"):
        replay.batch_rollout(
            context["state"], event["actions"], initial_warmstart=context["warmstart"]
        )
    replay = preflight._ReplayBank(env, [event], context, work)
    with pytest.raises(ValueError, match="different bank"):
        replay.batch_rollout(
            context["state"], event["actions"] + 0.001, initial_warmstart=context["warmstart"]
        )
    work.close("COMPLETED")


def test_actual_binary_torch_version_is_checked_not_package_metadata(monkeypatch):
    import sys

    versions = dict(
        python="3.11.16",
        numpy=np.__version__,
        scipy="1.17.1",
        mujoco="3.6.0",
        torch="2.10.0+cpu",
        h5py="3.16.0",
    )
    monkeypatch.setattr(preflight.platform, "python_version", lambda: versions["python"])
    for name in ("h5py", "mujoco", "scipy"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__version__=versions[name]))
    fake = SimpleNamespace(
        __version__="2.10.0",
        cuda=SimpleNamespace(is_initialized=lambda: False),
        set_num_threads=lambda value: None,
        get_num_threads=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(ValueError, match="Actual imported runtime"):
        preflight._runtime(dict(resources=dict(runtime=versions)))
    fake.__version__ = "2.10.0+cpu"
    assert preflight._runtime(dict(resources=dict(runtime=versions))) == versions


def _fake_workflow(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    source = tmp_path / "source.json"
    source.write_text("{}")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}")
    frozen = dict(task={}, parents=[dict(prepared_directory="initial/stream0")])
    monkeypatch.setattr(preflight, "_validate_bindings", lambda *args: (frozen, {}))
    monkeypatch.setattr(preflight, "_runtime", lambda config: {"synthetic_only": True})
    monkeypatch.setattr(
        preflight,
        "_limits",
        lambda config=None: dict(
            wall_seconds=600,
            output_bytes=1024**2,
            memory_bytes=8 * 1024**3,
            numpy_constructions=1,
            checkpoint_loads=1,
        ),
    )
    env = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(native, "_env", lambda *args: env)
    monkeypatch.setattr(native.NumpyPolicy, "from_checkpoint", lambda path: object())
    monkeypatch.setattr(preflight, "_replays", lambda *args: ({}, [dict(passed=True)]))
    monkeypatch.setattr(preflight, "_zero_residual", lambda *args: dict(D=0.0))
    monkeypatch.setattr(preflight, "_solves", lambda *args: ([dict(passed=True)], {}))
    monkeypatch.setattr(preflight, "_continuation", lambda *args: dict(synthetic_updates=8))
    monkeypatch.setattr(preflight, "_fall_accounting", lambda: dict(complete_cost=37.0))
    monkeypatch.setattr(
        preflight, "_budget_enforcement", lambda out: dict(rejected_before_entry=True)
    )
    return config, prepared, source


def test_create_only_workflow_emits_all_eight_typed_passes_and_flat_work(monkeypatch, tmp_path):
    config, prepared, source = _fake_workflow(monkeypatch, tmp_path)
    result = preflight.run_preflight(config, prepared, source, tmp_path / "out")
    assert result["status"] == "PASS"
    assert {row["name"] for row in result["checks"]} == set(preflight.CHECK_NAMES)
    assert all(row["passed"] is True for row in result["checks"])
    assert result["config_sha256"] == sha(config)
    assert result["prepared_manifest_sha256"] == sha(prepared / "manifest.json")
    assert all(type(value) is int for value in result["work"].values())
    assert json.loads((tmp_path / "out/preflight.json").read_text())["status"] == "PASS"
    with pytest.raises(FileExistsError):
        preflight.run_preflight(config, prepared, source, tmp_path / "out")


@pytest.mark.parametrize(
    "function",
    [
        "_replays",
        "_zero_residual",
        "_solves",
        "_continuation",
        "_fall_accounting",
        "_budget_enforcement",
    ],
)
def test_failed_prerequisite_stops_and_preserves_prior_checks(monkeypatch, tmp_path, function):
    paths = _fake_workflow(monkeypatch, tmp_path)
    entered = []

    def fail(*args):
        entered.append(True)
        raise ValueError("declared negative fixture")

    monkeypatch.setattr(preflight, function, fail)
    result = preflight.run_preflight(*paths, tmp_path / "out")
    assert entered == [True] and result["status"] == "FAIL_STOPPED"
    assert any(row["status"] == "FAIL_STOPPED" for row in result["checks"])
    assert not all(row["passed"] for row in result["checks"])
    assert (tmp_path / "out/ledger.jsonl").exists()


def test_missing_source_binding_stops_before_runtime_or_physics(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"schema":"trajectory-pi-p1-1","launch_ready":true}')
    entered = []
    monkeypatch.setattr(preflight, "_runtime", lambda config: entered.append(True))
    result = preflight.run_preflight(
        config, tmp_path / "missing", tmp_path / "missing.json", tmp_path / "out"
    )
    assert result["status"] == "FAIL_STOPPED" and not entered
    assert result["work"]["completed_native_controls"] == 0
    assert result["work"]["completed_optimizer_steps"] == 0


def test_continuation_uses_only_synthetic_rows_and_actual_saved_stream(monkeypatch, tmp_path):
    import random
    import sys

    # The fake envelope restores its own dictionary, so keep real test-run RNGs
    # untouched while exercising the production disturbance/restoration path.
    monkeypatch.setattr(np.random, "seed", lambda seed: None)
    monkeypatch.setattr(random, "seed", lambda seed: None)

    globals_now = dict(value=0)
    saved, calls = {}, []

    class FakePolicy:
        normalization = dict(mean=np.zeros(17, np.float32), std=np.ones(17, np.float32))
        _generator = SimpleNamespace(get_state=lambda: np.array([7], dtype=np.int64))

        def __init__(self, cursor=0):
            self.cursor = cursor

        def hash(self):
            return str(self.cursor)

        def state_dict(self):
            return dict(weight=np.array([self.cursor], dtype=np.int64))

    def load(path, **kwargs):
        key = str(path)
        if key in saved:
            policy, state, global_state = saved[key]
            globals_now.update(global_state)
            return policy, deepcopy(state), {}
        globals_now.update(value=0)
        return FakePolicy(), dict(cursor=2000, parent_policy_hash="legacy"), {}

    def fit(parent, x, y, *, updates, seed, optimizer_state, resume_fit, meter):
        assert x.shape == (32, 17) and y.shape == (32, 6)
        assert seed == 2861000000 and np.max(np.abs(y)) < 0.7
        calls.append(dict(updates=updates, resume_fit=resume_fit))
        for unit, amount in (
            ("neural_rows", updates * 256),
            ("backward_rows", updates * 256),
            ("optimizer_steps", updates),
        ):
            meter.measure(unit, amount, lambda: None)
        cursor = parent.cursor + updates
        globals_now["value"] += updates
        return FakePolicy(cursor), dict(cursor=2000 + cursor, parent_policy_hash=parent.hash()), {}

    def save(path, policy, state, metadata):
        assert metadata == dict(synthetic_only=True)
        saved[str(path)] = policy, deepcopy(state), deepcopy(globals_now)
        return dict(global_rng_fingerprint=fingerprint(globals_now))

    fake_torch = SimpleNamespace(
        random=SimpleNamespace(
            default_generator=SimpleNamespace(manual_seed=lambda seed: globals_now.update(value=-1))
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(preflight, "_action_parity", lambda *args: dict(synthetic_fixture=True))
    monkeypatch.setattr(student, "load_stream", load)
    monkeypatch.setattr(student, "fit_round", fit)
    monkeypatch.setattr(student, "save_stream", save)
    monkeypatch.setattr(student, "global_rng_state", lambda: deepcopy(globals_now))
    monkeypatch.setattr(student, "restore_global_rng", lambda value: globals_now.update(value))
    work = _work(tmp_path / "meter")
    config = dict(
        parents=[dict(prepared_directory="initial/stream0")],
        seeds=dict(synthetic_continuation=2861000000),
        prerequisites=dict(action_function_parity_atol=1e-6),
    )
    result = preflight._continuation(config, tmp_path, tmp_path, work)
    assert result["synthetic_updates"] == 8
    assert calls == [
        dict(updates=4, resume_fit=False),
        dict(updates=2, resume_fit=False),
        dict(updates=2, resume_fit=True),
    ]
    assert work.counts["completed_optimizer_steps"] == 8
    assert work.counts["completed_backward_rows"] == 2048
    work.close("COMPLETED")


def test_complete_binding_contract_and_source_mutation_fail_before_execution(monkeypatch, tmp_path):
    import hashlib

    root, prepared = tmp_path / "source", tmp_path / "prepared"
    own = root / "src/gps/mppi_gps/trajectory_pi_preflight.py"
    own.parent.mkdir(parents=True)
    own.write_text("# frozen fixture source\n")
    original = Path(preflight.__file__).resolve().parents[3] / preflight.PROTOTYPE
    archive = root / preflight.PROTOTYPE
    archive.parent.mkdir(parents=True)
    archive.write_bytes(original.read_bytes())
    checkpoint = prepared / "initial/stream0/policy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen original checkpoint bytes")
    teacher_dir = prepared / "teachers"
    teacher_dir.mkdir()
    for index in range(2):
        (teacher_dir / f"slot{index}.npz").write_bytes(b"frozen complete teacher fixture")
    config = root / "configs/studies/trajectory_pi_p1.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            dict(
                schema="trajectory-pi-p1-1",
                launch_ready=True,
                streams=[0],
                parents=[
                    dict(
                        prepared_directory="initial/stream0",
                        checkpoint=dict(sha256=sha(checkpoint)),
                    )
                ],
                teachers=[dict(prepared_path=f"teachers/slot{index}.npz") for index in range(2)],
            )
        )
    )
    names = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()]
    rows = [_row(root, name) for name in sorted(names)]
    source = tmp_path / "source_manifest.json"
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    source.write_text(
        json.dumps(
            dict(
                schema="trajectory-pi-source-1",
                files=rows,
                roster_sha256=hashlib.sha256(encoded).hexdigest(),
            )
        )
    )
    names = [str(path.relative_to(prepared)) for path in prepared.rglob("*") if path.is_file()]
    (prepared / "manifest.json").write_text(
        json.dumps(
            dict(
                schema="trajectory-pi-prepared-1",
                config_sha256=sha(config),
                normalization_inherited=True,
                parent_refits=0,
                parents=[dict(stream=0)],
                teachers=[dict(slot=0), dict(slot=1)],
                files=[_row(prepared, name) for name in sorted(names)],
            )
        )
    )
    monkeypatch.setattr(preflight, "__file__", str(own))
    frozen, manifest = preflight._validate_bindings(config, prepared, source)
    assert frozen["launch_ready"] and manifest["parent_refits"] == 0
    own.write_text("# edited after freeze\n")
    with pytest.raises(ValueError, match="Immutable file binding"):
        preflight._validate_bindings(config, prepared, source)


def test_module_import_does_not_import_simulator_or_torch():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.gps.mppi_gps.trajectory_pi_preflight; "
            "assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_invalid_hypothetical_candidate_is_masked_before_inference(tmp_path):
    event, env, _, context = _synthetic_event(0.12)
    event["states"][1, 0, 0] = np.nan
    event["C"][1] = np.inf
    event["D"] = np.array([0.5 * 18 * (0.1 - 0.25) ** 2, np.inf])
    # Preserve exactly the per-control summation order used by the score.
    event["D"][0] = np.sum(0.5 * np.sum((event["actions"][0] - 0.25) ** 2, axis=-1))
    event["J"] = event["C"] + 0.12 * event["D"]
    event["weights"] = np.array([1.0, 0.0])
    event["post_actions"] = event["current_actions"].copy()
    rows = []

    def predict(values, meter=None):
        assert np.isfinite(values).all()
        rows.append(len(values))
        return np.full((len(values), 6), 0.25, dtype=np.float32)

    work = _work(tmp_path)
    result = preflight._manual_bank(
        event, env, SimpleNamespace(predict=predict), context, 0.12, work
    )
    assert rows == [3] and all(row["passed"] for row in result["comparisons"])
    work.close("COMPLETED")


def test_action_parity_meters_both_backends_on_saved_native_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr(native, "load_trace", lambda path: dict(observations=np.zeros((17, 17))))

    class Actor:
        def __init__(self, delta):
            self.delta = delta

        def predict(self, values, meter):
            return meter.measure(
                "neural_rows",
                len(values),
                lambda: np.full((len(values), 6), self.delta, np.float32),
            )

    parent, actor = Actor(0.0), Actor(1e-7)
    monkeypatch.setattr(student.NumpyPolicy, "from_policy", lambda policy: actor)
    work = _work(tmp_path / "meter")
    result = preflight._action_parity(parent, tmp_path, work, 1e-6)
    assert result["measured_neural_rows"] == work.counts["completed_neural_rows"] == 32
    assert result["maximum_absolute_error"] <= 1e-6
    actor.delta = 1e-3
    with pytest.raises(ValueError, match="parity failed"):
        preflight._action_parity(parent, tmp_path, work, 1e-6)
    with pytest.raises(ValueError, match="tolerance"):
        preflight._action_parity(parent, tmp_path, work, 1e-5)
    work.close("COMPLETED")
