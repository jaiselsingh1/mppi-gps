"""Saved-array/contract tests only: no simulator, actor, or recovery execution."""

import hashlib
import json
import sys

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.experience import (
    AbsentContrast,
    _fit_hash,
    calibrate_agreement,
    coverage_diagnostic,
    load_experience,
    recovery_acceptance,
)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(stream, tag, **arrays):
    group = stream.create_group("arrays/" + tag)
    group.attrs["tag"] = tag
    refs = {}
    for key, value in arrays.items():
        value = np.asarray(value)
        dataset = group.create_dataset(key, data=value)
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        dataset.attrs["numpy_dtype"] = value.dtype.str
        dataset.attrs["encoding"] = "array"
        dataset.attrs["sha256"] = digest
        refs[key] = dict(
            dtype=value.dtype.str, shape=list(value.shape), encoding="array", sha256=digest
        )
    return dict(file="raw.h5", group="arrays/" + tag, tag=tag, arrays=refs)


@pytest.fixture
def recorded_arrays(tmp_path):
    """A saved-data fixture, explicitly not a simulated research environment."""
    parent = tmp_path / "closed_parent"
    (parent / "evidence").mkdir(parents=True)
    _json(
        parent / "config.json",
        dict(schema="walker-cio-mppi-v1", mppi=dict(H=2), task=dict(nthread=4)),
    )
    _json(parent / "result.json", dict(error=None))
    source = parent / "source/bound.txt"
    source.parent.mkdir()
    source.write_text("fixture source binding, not executable model code")
    _json(
        parent / "manifest.json",
        dict(
            schema="walker-cio-source-v1",
            config_sha256=_sha(parent / "config.json"),
            files=[dict(file="bound.txt", sha256=_sha(source))],
        ),
    )
    rows, episodes, obs_parts, action_parts, feedback = [], {}, [], [], []
    rng = np.random.default_rng(7)
    with h5py.File(parent / "evidence/raw.h5", "x") as stream:
        for e in range(4):
            tag, n = f"discovery_{e:02d}", 1000
            states = np.zeros((n + 1, 19), dtype=float)
            states[:, 2] = 1.25
            states[:, 3] = np.arange(n + 1) * 1e-5
            states[:, 10] = 1.5
            obs = np.concatenate((states[:-1, 2:10], np.clip(states[:-1, 10:19], -10, 10)), axis=1)
            actions = rng.uniform(-1, 1, (n, 6))
            episodes[tag] = dict(observations=obs, actions=actions)
            obs_parts.append(obs)
            action_parts.append(actions)
            for t in range(n):
                rows.append(
                    dict(
                        event="independent_planner_before",
                        episode=tag,
                        step=t,
                        private=dict(
                            U=np.full((2, 6), t / 1000).tolist(),
                            initial_plan=t == 0,
                            rng_state=dict(
                                bit_generator="PCG64",
                                state=dict(state=t + 1, inc=1),
                                has_uint32=0,
                                uinteger=0,
                            ),
                        ),
                    )
                )
            raw = _ref(
                stream,
                tag,
                observations=obs,
                actions=actions,
                states=states,
                warmstarts=np.zeros((n + 1, 9)),
                integrations=np.zeros((n + 1, 91)),
                costs=np.ones(n),
            )
            rows.append(
                dict(
                    event="episode_complete",
                    tag=tag,
                    complete=True,
                    success=True,
                    steps=n,
                    seed=e,
                    raw=raw,
                )
            )
            for t in np.linspace(0, n - 1, 16, dtype=int):
                plan = np.tile(actions[t], (2, 1))
                evidence = _ref(
                    stream,
                    f"decision_{e}_{t}",
                    baseline_plan=plan,
                    selected_plan=plan,
                    selected_pre_action_observations=np.tile(obs[t], (1, 2, 1)),
                )
                rows.append(
                    dict(
                        event="decision",
                        episode=tag,
                        step=int(t),
                        actor_hash=None,
                        influence_used=False,
                        baseline_task_cost=float(4 + e + t / 1000),
                        evidence=evidence,
                    )
                )
        for e in range(4):
            tag = f"discovery_{e:02d}"
            for i in range(16):
                t = [0, 1, 125, 875][i % 4]
                obs = episodes[tag]["observations"][t].copy()
                obs[2] += 0.001 * (i + 1)
                action = rng.uniform(-1, 1, 6)
                obs_parts.append(obs[None])
                action_parts.append(action[None])
                digest = f"bank_{e}_{i}"
                selected = _ref(
                    stream,
                    digest,
                    selected_plan=np.tile(action, (2, 1)),
                    selected_pre_action_observations=np.tile(obs, (1, 2, 1)),
                )
                rows.append(dict(event="coupled_selection", bank_digest=digest, evidence=selected))
                feedback.append(dict(episode=tag, step=t, bank_digest=digest))
        obs, actions = np.concatenate(obs_parts), np.concatenate(action_parts)
        raw = _ref(stream, "bootstrap", observations=obs, actions=actions)
        rows.append(
            dict(
                event="teaching_dataset",
                tag="bootstrap",
                raw=raw,
                executed_episodes=list(episodes),
                feedback=feedback,
            )
        )
    rows.append(dict(event="runtime_close"))
    for i, row in enumerate(rows):
        row["id"] = f"event-{i:09d}"
    (parent / "evidence/ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _json(
        parent / "learning/bootstrap.json",
        dict(
            fit_data=dict(
                observations_sha256=_fit_hash(obs), actions_sha256=_fit_hash(actions), rows=4064
            )
        ),
    )
    return parent, episodes, feedback, obs, actions


def test_loader_preserves_exact_targets_history_phase_and_fit_binding(recorded_arrays):
    parent, episodes, feedback, obs, actions = recorded_arrays
    before = set(sys.modules)
    loaded = load_experience(parent)
    imported = set(sys.modules) - before
    assert not any(name.split(".")[0] in {"torch", "mujoco"} for name in imported)
    np.testing.assert_array_equal(loaded["observations"], obs)
    np.testing.assert_array_equal(loaded["actions"], actions)
    assert loaded["actions"].dtype == np.float64
    previous = loaded["previous_actions"]
    for e, episode in enumerate(loaded["episodes"]):
        np.testing.assert_array_equal(previous[e * 1000], np.zeros(6))
        np.testing.assert_array_equal(
            previous[e * 1000 + 1 : (e + 1) * 1000], episode["actions"][:-1]
        )
        np.testing.assert_array_equal(
            episode["private_states_by_step"][875]["U"], np.full((2, 6), 0.875)
        )
        assert not episode["private_states_by_step"][875]["initial_plan"]
    for i, row in enumerate(feedback):
        expected = (
            np.zeros(6)
            if row["step"] == 0
            else episodes[row["episode"]]["actions"][row["step"] - 1]
        )
        np.testing.assert_array_equal(previous[4000 + i], expected)
    np.testing.assert_array_equal(
        loaded["normalization"]["mean"], obs.mean(axis=0).astype(np.float32)
    )
    assert loaded["normalization"]["std"].dtype == np.float32
    assert loaded["provenance"]["float32_fit_data"]["rows"] == 4064
    assert len(loaded["trajectory_ids"]) == 4064
    assert len(loaded["calibration"]) == 64
    assert [r["step"] for r in loaded["calibration"][:16]] == list(
        np.linspace(0, 999, 16, dtype=int)
    )
    assert loaded["calibration"][0]["observations"].shape == (2, 17)
    np.testing.assert_array_equal(loaded["calibration"][0]["plan"][0], actions[0])


def test_loader_rejects_changed_saved_target_bytes(recorded_arrays):
    parent, *_ = recorded_arrays
    with h5py.File(parent / "evidence/raw.h5", "r+") as stream:
        stream["arrays/bootstrap/actions"][0, 0] += 0.01
    with pytest.raises(ValueError, match="target bytes"):
        load_experience(parent)


def test_loader_rejects_changed_frozen_source_and_open_ledger(recorded_arrays):
    parent, *_ = recorded_arrays
    source = parent / "source/bound.txt"
    saved = source.read_text()
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        load_experience(parent)
    source.write_text(saved)
    path = parent / "evidence/ledger.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="closed bootstrap"):
        load_experience(parent)


def test_loader_rejects_float32_fit_input_mismatch(recorded_arrays):
    parent, *_ = recorded_arrays
    path = parent / "learning/bootstrap.json"
    value = json.loads(path.read_text())
    value["fit_data"]["actions_sha256"] = "wrong"
    _json(path, value)
    with pytest.raises(ValueError, match="float32 fit inputs"):
        load_experience(parent)


def test_local_recovery_criterion_checks_both_endpoint_and_native_cost():
    norm = dict(mean=np.zeros(17), std=np.ones(17))
    nominal, costs = np.zeros((1000, 17)), np.full(1000, 0.2)
    observations, actual = np.zeros((35, 17)), np.full(34, 0.2)
    observations[2] = 0.5
    observations[-1] = 0.4
    result = recovery_acceptance(observations, actual, nominal, costs, 125, norm)
    assert result["accepted"] and result["start_rms"] == 0.5
    assert result["endpoint_limit"] == 0.45
    observations[-1] = 0.6
    assert not recovery_acceptance(observations, actual, nominal, costs, 125, norm)["accepted"]
    observations[-1] = 0.05
    actual[-8:] = 0.41
    assert not recovery_acceptance(observations, actual, nominal, costs, 125, norm)["accepted"]
    assert not recovery_acceptance(
        observations, np.full(34, 0.2), nominal, costs, 125, norm, done=True
    )["accepted"]
    assert not recovery_acceptance(observations[:3], actual[:2], nominal, costs, 125, norm)[
        "accepted"
    ]


def test_coverage_is_nearest_normalized_rms_with_held_trajectory_calibration():
    norm = dict(mean=np.zeros(17), std=np.ones(17))
    train = np.repeat(np.array([0.0, 1.0, 4.0, 5.0])[:, None], 17, axis=1)
    rollout = np.repeat(np.array([0.0, 10.0])[:, None], 17, axis=1)
    result = coverage_diagnostic(train, rollout, norm, ["a", "a", "b", "b"])
    np.testing.assert_allclose(result["distances"], [0, 5])
    np.testing.assert_allclose(result["calibration_distances"], [4, 3, 3, 4])
    assert result["leave_one_trajectory_out_q95"] == 4
    assert result["above_calibration_fraction"] == 0.5
    assert "descriptive" in result["scope"]
    assert coverage_diagnostic(train, np.empty((0, 17)), norm)["q95"] is None
    with pytest.raises(ValueError, match="at least two trajectories"):
        coverage_diagnostic(train, rollout, norm, ["a"] * 4)


class CalibrationMeter:
    def __init__(self):
        self.arrays, self.events, self.neural_rows = [], [], 0

    def measure(self, kind, amount, fn, **metadata):
        assert kind == "neural_rows"
        self.neural_rows += amount
        return fn()

    def save_arrays(self, tag, **arrays):
        self.arrays.append((tag, arrays))
        return dict(file="saved-array-test", group=tag)

    def record(self, event, **fields):
        json.dumps(fields, allow_nan=False)
        self.events.append((event, fields))


class CalibrationActor:
    def __init__(self, value=0.0):
        self.value = value

    def hash(self):
        return "fixed-test-owner"

    def predict(self, obs, meter):
        return meter.measure("neural_rows", len(obs), lambda: np.full((len(obs), 6), self.value))


def calibration_entries(cost=24.0):
    return dict(
        calibration=[
            dict(
                tag=f"teacher_{i // 16}",
                step=i % 16,
                plan=np.ones((2, 6)),
                observations=np.zeros((2, 17)),
                task_cost=cost,
            )
            for i in range(64)
        ]
    )


def test_agreement_calibration_uses_median_ratio_and_only_metered_actor_inference():
    meter = CalibrationMeter()
    result = calibrate_agreement(calibration_entries(), CalibrationActor(), meter)
    np.testing.assert_allclose(result["rhos"], [0.2, 0.8])
    assert meter.neural_rows == 128
    assert result["physics_queries_added"] == 0
    assert result["contrast_present"] and not result["duplicated_rhos"]
    np.testing.assert_array_equal(meter.arrays[0][1]["agreement_costs"], np.full(64, 6.0))


def test_agreement_calibration_retains_duplicate_clamped_contrasts():
    high = calibrate_agreement(calibration_entries(1e8), CalibrationActor(), CalibrationMeter())
    low = calibrate_agreement(calibration_entries(0.0), CalibrationActor(), CalibrationMeter())
    assert high["rhos"] == [10.0, 10.0] and high["duplicated_rhos"]
    assert low["rhos"] == [1e-4, 1e-4] and low["duplicated_rhos"]


def test_absent_agreement_contrast_records_evidence_and_invents_no_weight():
    meter = CalibrationMeter()
    with pytest.raises(AbsentContrast, match="below 1e-12"):
        calibrate_agreement(calibration_entries(), CalibrationActor(1.0), meter)
    assert not meter.events[-1][1]["contrast_present"]
    assert meter.events[-1][1]["rhos"] == []
