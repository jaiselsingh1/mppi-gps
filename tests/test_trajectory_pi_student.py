"""Synthetic contracts for unchanged W2 learning and deterministic PI actions."""

from __future__ import annotations

import copy
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.gps.mppi_gps.policy import Policy, fit_policy
from src.gps.mppi_gps.provenance import fingerprint, sha
from src.gps.mppi_gps.trajectory_pi_student import (
    NumpyPolicy,
    fit_round,
    freeze_legacy_parent,
    global_rng_state,
    initial_stream,
    load_stream,
    normalization_from_training,
    observations,
    save_stream,
)


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def data():
    rng = np.random.default_rng(801)
    obs = rng.normal(size=(19, 17))
    actions = np.tanh(rng.normal(size=(19, 6)))
    return obs, actions


def test_unchanged_native_features_and_w2_normalization():
    states = np.arange(19 * 3, dtype=np.float64).reshape(3, 19) - 20
    actual = observations(states)
    expected = np.concatenate((states[:, 2:10], np.clip(states[:, 10:19], -10, 10)), axis=1)
    assert np.array_equal(actual, expected)
    values = np.ones((6, 17), dtype=np.float64)
    values[:, 3] = np.arange(6)
    norm = normalization_from_training(values)
    assert np.array_equal(norm["mean"], values.mean(0).astype(np.float32))
    assert np.array_equal(norm["std"], np.maximum(values.std(0), 0.05).astype(np.float32))
    with pytest.raises(ValueError):
        normalization_from_training(values[:0])


def test_numpy_torch_parity_and_exact_scalar_chunk_layout_invariance():
    obs, _ = data()
    actor = Policy(seed=1903, normalization=normalization_from_training(obs))
    exported = NumpyPolicy.from_policy(actor)
    rng = np.random.default_rng(29)
    values = rng.normal(size=(256, 17)).astype(np.float32)
    values[0] *= 1000
    values[1] *= -1000
    whole = exported.predict(values)
    single = np.stack([exported.predict(row) for row in values])
    chunks = np.concatenate(
        [
            exported.predict(values[:17]),
            exported.predict(values[17:90]),
            exported.predict(values[90:]),
        ]
    )
    assert np.array_equal(whole, single)
    assert np.array_equal(whole, chunks)
    assert np.array_equal(whole, exported.predict(np.asfortranarray(values)))
    assert np.array_equal(whole, exported.predict(values.reshape(16, 16, 17)).reshape(256, 6))
    assert exported.policy_hash == actor.hash()
    assert np.max(np.abs(whole - actor.predict(values))) <= 1e-6
    assert np.all(np.abs(whole) <= 1)
    assert exported.predict(values[:0]).shape == (0, 6)
    payload = exported.payload()
    payload["state"]["network.0.weight"][0, 0] = 999
    assert np.array_equal(exported.predict(values), whole)


def test_exact_zero_residual_reproduces_deterministic_closed_trajectory():
    actor = NumpyPolicy.from_policy(Policy(seed=81))
    state = np.linspace(-0.2, 0.2, 17, dtype=np.float64)
    states, controls = [state.copy()], []
    for _ in range(30):
        action = actor.predict(state).astype(np.float64)
        controls.append(action)
        state = 0.97 * state
        state[:6] += 0.05 * action
        states.append(state.copy())
    predictions = actor.predict(np.asarray(states[:-1]))
    assert np.array_equal(predictions.astype(np.float64), controls)
    assert float(np.sum((predictions.astype(np.float64) - controls) ** 2)) == 0
    replay = np.asarray(states)[0].copy()
    for index, control in enumerate(controls):
        actual = actor.predict(replay).astype(np.float64)
        assert np.array_equal(actual, control)
        replay = 0.97 * replay
        replay[:6] += 0.05 * actual
        assert np.array_equal(replay, states[index + 1])


def test_initial_stream_materializes_empty_adam_without_updates():
    obs, actions = data()
    parent = Policy(seed=91)
    child, state, evidence = initial_stream(parent, obs, actions, seed=77)
    assert parent.hash() == child.hash()
    assert state["updates"] == state["fit_updates"] == 0
    assert state["adam"]["state"] == {}
    assert evidence["learning_curve"] == []
    assert all(value["requested"] == value["completed"] == 0 for value in evidence["work"].values())


def test_checkpoint_continuation_preserves_actual_adam_private_and_global_rng(tmp_path):
    obs, actions = data()
    parent = Policy(seed=91)
    whole, whole_state, _ = fit_round(parent, obs, actions, updates=8, seed=119)
    early, early_state, _ = fit_round(parent, obs, actions, updates=3, seed=119)
    before = global_rng_state()
    frozen = save_stream(tmp_path / "stream", early, early_state, {"point": 3})
    assert fingerprint(global_rng_state()) == fingerprint(before)
    torch.rand(3)
    np.random.normal(size=3)
    random.random()
    loaded, loaded_state, meta = load_stream(tmp_path / "stream")
    assert meta == {"point": 3}
    assert fingerprint(global_rng_state()) == frozen["global_rng_fingerprint"]
    assert fingerprint(loaded_state) == fingerprint(early_state)
    final, final_state, _ = fit_round(
        loaded, obs, actions, updates=5, seed=999999, optimizer_state=loaded_state, resume_fit=True
    )
    assert final.hash() == whole.hash()
    assert fingerprint(
        {key: value for key, value in final_state.items() if key != "parent_policy_hash"}
    ) == fingerprint(
        {key: value for key, value in whole_state.items() if key != "parent_policy_hash"}
    )
    # The saved actor really is the incoming parent of the resumed call; this
    # provenance field intentionally differs while operative training is exact.
    assert final_state["parent_policy_hash"] == loaded.hash()
    assert whole_state["parent_policy_hash"] == parent.hash()
    assert torch.equal(final._generator.get_state(), whole._generator.get_state())
    assert NumpyPolicy.from_checkpoint(tmp_path / "stream/policy.pt").policy_hash == loaded.hash()
    with pytest.raises(FileExistsError):
        save_stream(tmp_path / "stream", early, early_state)
    with pytest.raises(ValueError, match="same device"):
        load_stream(tmp_path / "stream", device="cuda")


def test_new_corpus_keeps_existing_sampler_and_adam_without_reseeding():
    obs, actions = data()
    parent = Policy(seed=91)
    actor, state, _ = fit_round(parent, obs, actions, updates=2, seed=3)
    changed = obs + 0.1
    first, first_state, _ = fit_round(
        actor, changed, actions, updates=2, seed=12, optimizer_state=state
    )
    second, second_state, _ = fit_policy(
        actor,
        changed,
        actions,
        updates=2,
        seed=99999,
        optimizer_state=state,
        batch_size=256,
        lr=3e-4,
        eta_theta=0,
        prox_reduction="mean",
        anchor_refresh=0,
    )
    assert first.hash() == second.hash()
    assert fingerprint(first_state) == fingerprint(second_state)
    assert first_state["updates"] == 4 and first_state["fit_updates"] == 2
    assert first_state["anchor_policy_hash"] == actor.hash()


def test_freeze_legacy_parent_copies_original_bytes_and_does_no_parent_refit(tmp_path):
    obs, actions = data()
    parent, state, _ = fit_round(Policy(seed=7), obs, actions, updates=2, seed=8)
    original = tmp_path / "old.pt"
    parent.checkpoint(original, optimizer_state=state, metadata={"historical": True})
    globals_before = fingerprint(global_rng_state())
    actor, inherited, envelope = freeze_legacy_parent(
        original, tmp_path / "frozen", global_seed=881, metadata={"study": "new"}
    )
    assert fingerprint(global_rng_state()) == globals_before
    assert original.read_bytes() == (tmp_path / "frozen/policy.pt").read_bytes()
    assert actor.hash() == parent.hash() and fingerprint(inherited) == fingerprint(state)
    assert envelope["historical_global_rng_available"] is False
    assert envelope["work"]["optimizer_steps"] == envelope["work"]["neural_rows"] == 0
    assert envelope["source_checkpoint"]["sha256"] == sha(original)
    loaded, loaded_state, meta = load_stream(tmp_path / "frozen")
    assert loaded.hash() == actor.hash() and fingerprint(loaded_state) == fingerprint(state)
    assert meta == {"historical": True}
    expected = torch.Generator(device="cpu").manual_seed(881).get_state()
    assert torch.equal(global_rng_state()["torch_cpu"], expected)


def test_invalid_export_and_stream_files_fail_before_forward(tmp_path):
    exported = NumpyPolicy.from_policy(Policy(seed=5)).payload()
    bad = copy.deepcopy(exported)
    bad["state"]["observation_std"][0] = 0
    with pytest.raises(ValueError):
        NumpyPolicy(bad)
    bad = copy.deepcopy(exported)
    bad["state"]["network.0.weight"][0, 0] += 0.001
    with pytest.raises(ValueError, match="hash"):
        NumpyPolicy(bad)
    with pytest.raises(ValueError, match="unchanged"):
        NumpyPolicy.from_policy(Policy(obs_dim=23))
    obs, actions = data()
    actor, state, _ = initial_stream(Policy(), obs, actions, seed=4)
    save_stream(tmp_path / "stream", actor, state)
    with (tmp_path / "stream/global_rng.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        load_stream(tmp_path / "stream")


def test_numpy_import_and_inference_load_no_torch_or_simulator():
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','mujoco'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from src.gps.mppi_gps.trajectory_pi_student import NumpyPolicy, observations
assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_existing_fit_and_export_import_no_simulator():
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc,sys,numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]=='mujoco': raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from src.gps.mppi_gps.policy import Policy
from src.gps.mppi_gps.trajectory_pi_student import fit_round,NumpyPolicy
actor,state,evidence=fit_round(Policy(seed=17),np.zeros((3,17)),np.zeros((3,6)),updates=1,seed=18)
assert evidence['work']['optimizer_steps']['completed']==1
assert NumpyPolicy.from_policy(actor).predict(np.zeros(17)).shape==(6,)
assert 'mujoco' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_policy_row_budget_precedes_contractions(monkeypatch):
    actor = NumpyPolicy.from_policy(Policy(seed=9))
    entered = []
    monkeypatch.setattr(np, "einsum", lambda *args, **kwargs: entered.append(True))

    class Meter:
        def measure(self, kind, rows, fn, **metadata):
            assert kind == "neural_rows" and rows == 12
            raise RuntimeError("Synthetic row budget exhausted")

    with pytest.raises(RuntimeError, match="budget"):
        actor.predict(np.zeros((12, 17)), meter=Meter())
    assert entered == []
