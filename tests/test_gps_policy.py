"""Policy/data/optimizer contracts; no environment or physics is constructed."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from src.gps.mppi_gps.policy import Policy, fit_policy, full_dataset_mse


class CountingMeter:
    def __init__(self):
        self.counts = dict(neural_rows=0, backward_rows=0, optimizer_steps=0)

    def measure(self, kind, count, function, *, purpose):
        assert purpose
        self.counts[kind] += count
        return function()


def equal_tree(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert set(first) == set(second)
        for key in first:
            equal_tree(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            equal_tree(a, b)
    else:
        assert first == second


def data():
    rng = np.random.default_rng(13)
    observations = rng.normal(0, 0.2, (9, 17)).astype(np.float32)
    actions = np.tanh(observations[:, :6] + 0.1).astype(np.float32)
    return observations, actions


def test_bounded_shapes_meter_and_private_initialization():
    torch.manual_seed(91)
    np.random.seed(92)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = deepcopy(np.random.get_state())
    policy = Policy(seed=3)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    after = np.random.get_state()
    assert (
        numpy_before[0] == after[0]
        and np.array_equal(numpy_before[1], after[1])
        and numpy_before[2:] == after[2:]
    )
    torch.manual_seed(9)
    np.random.seed(10)
    assert Policy(seed=3).hash() == policy.hash()
    assert Policy(seed=4).hash() != policy.hash()
    meter = CountingMeter()
    values = policy.predict(np.full((2, 3, 17), 100.0, dtype=np.float64), meter=meter)
    assert values.shape == (2, 3, 6) and np.isfinite(values).all() and np.all(np.abs(values) <= 1)
    assert policy.predict(np.zeros(17)).shape == (6,)
    assert policy.predict(np.empty((0, 17))).shape == (0, 6)
    assert meter.counts == dict(neural_rows=6, backward_rows=0, optimizer_steps=0)
    assert sum(p.numel() for p in policy.parameters()) == 19590


@pytest.mark.parametrize(
    "observations,actions",
    [
        (np.zeros((2, 18)), np.zeros((2, 6))),
        (np.zeros((2, 17)), np.zeros((3, 6))),
        (np.zeros((1, 2, 17)), np.zeros((1, 2, 6))),
        (np.empty((0, 17)), np.empty((0, 6))),
        (np.full((2, 17), np.nan), np.zeros((2, 6))),
        (np.zeros((2, 17)), np.full((2, 6), 1.01)),
    ],
)
def test_pre_action_matrix_validation_precedes_any_metered_forward(observations, actions):
    policy = Policy()
    before = policy.hash()
    meter = CountingMeter()
    with pytest.raises(ValueError):
        fit_policy(policy, observations, actions, updates=1, meter=meter)
    assert policy.hash() == before and not any(meter.counts.values())


def test_one_update_matches_explicit_loss_adam_and_private_minibatch():
    parent = Policy(seed=8)
    obs, actions = data()
    before = parent.hash()
    meter = CountingMeter()
    reference = Policy(seed=8)
    optimizer = torch.optim.Adam(reference.parameters(), lr=3e-4, foreach=False, fused=False)
    rows = np.random.default_rng(44).integers(0, len(obs), size=4)
    prediction = reference(torch.as_tensor(obs[rows]))
    mse = torch.mean((prediction - torch.as_tensor(actions[rows])) ** 2)
    mse.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5))
    optimizer.step()
    child, state, evidence = fit_policy(
        parent, obs, actions, updates=1, batch_size=4, eta_theta=0.1, seed=44, meter=meter
    )
    equal_tree(child.state_dict(), reference.state_dict())
    equal_tree(state["adam"], optimizer.state_dict())
    assert parent.hash() == before and evidence["learning_curve"][0]["proximal_loss"] == 0.0
    assert evidence["learning_curve"][0]["action_mse"] == float(mse.detach())
    assert evidence["learning_curve"][0]["unclipped_gradient_norm"] == gradient_norm
    assert meter.counts == dict(neural_rows=4, backward_rows=4, optimizer_steps=1)


def test_positive_proximal_block_resume_and_checkpoint_are_exact(tmp_path):
    normalization = dict(mean=np.arange(17) / 20, std=np.linspace(0.5, 2, 17))
    parent = Policy(seed=2, normalization=normalization)
    obs, actions = data()
    before = parent.hash()
    whole, whole_state, whole_evidence = fit_policy(
        parent, obs, actions, updates=4, batch_size=5, eta_theta=0.07, seed=41
    )
    first, first_state, first_evidence = fit_policy(
        parent, obs, actions, updates=2, batch_size=5, eta_theta=0.07, seed=41
    )
    path = tmp_path / "policy.pt"
    reference = first.checkpoint(path, first_state, {"role": "fixed unit checkpoint"})
    loaded, loaded_state, metadata = Policy.from_checkpoint(path)
    assert loaded.hash() == first.hash() == reference["policy_hash"] and metadata == {
        "role": "fixed unit checkpoint"
    }
    equal_tree(loaded_state, first_state)
    with pytest.raises(FileExistsError):
        first.checkpoint(path, first_state)
    resumed, resumed_state, resumed_evidence = fit_policy(
        loaded,
        obs,
        actions,
        loaded_state,
        updates=2,
        batch_size=5,
        eta_theta=0.07,
        seed=999,
        resume_fit=True,
    )
    equal_tree(resumed.state_dict(), whole.state_dict())
    equal_tree(resumed_state["adam"], whole_state["adam"])
    assert (
        resumed_state["rng"] == whole_state["rng"]
        and resumed_state["updates"] == resumed_state["fit_updates"] == 4
    )
    assert (
        first_evidence["learning_curve"] + resumed_evidence["learning_curve"]
        == whole_evidence["learning_curve"]
    )
    assert parent.hash() == before and resumed_state["anchor_policy_hash"] == before
    assert resumed.normalization["mean"].tobytes() == parent.normalization["mean"].tobytes()
    assert resumed.normalization["std"].tobytes() == parent.normalization["std"].tobytes()
    assert whole_evidence["learning_curve"][-1]["proximal_loss"] > 0


def test_new_block_keeps_adam_rng_but_resets_anchor_and_rejects_wrong_owner():
    obs, actions = data()
    parent = Policy(7)
    child, state, _ = fit_policy(parent, obs, actions, updates=2, batch_size=3, seed=20)
    new, new_state, evidence = fit_policy(child, obs, actions, state, updates=1, batch_size=3)
    assert new_state["updates"] == 3 and new_state["fit_updates"] == 1
    assert (
        new_state["anchor_policy_hash"] == child.hash()
        and evidence["learning_curve"][0]["proximal_loss"] == 0
    )
    with pytest.raises(ValueError, match="different parent"):
        fit_policy(parent, obs, actions, state, updates=1)
    with pytest.raises(ValueError, match="unchanged data"):
        fit_policy(
            child, obs + 0.01, actions, state, updates=1, batch_size=3, seed=20, resume_fit=True
        )
    with pytest.raises(ValueError, match="resume_fit"):
        fit_policy(new, obs, actions, updates=1, resume_fit=True)


def test_legacy_seventeen_checkpoint_without_width_and_pinned_hash(tmp_path):
    policy = Policy(seed=3)
    with torch.no_grad():
        for value in policy.parameters():
            value.zero_()
    # Frozen v1 hash of zero float32 weights/means and unit standard deviations.
    assert policy.hash() == "b911f6f175d86156ad2bb31d4f58fe719551b8823a04de76dac39fc0a025d858"
    payload = dict(
        schema="walker-tanh-policy-v1",
        seed=3,
        policy_hash=policy.hash(),
        state_dict=policy.state_dict(),
        private_rng_state=policy._generator.get_state(),
        training=True,
        optimizer_state=None,
        metadata={"role": "legacy17"},
    )
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    loaded, state, metadata = Policy.from_checkpoint(path)
    assert loaded.obs_dim == 17 and loaded.hash() == policy.hash() and state is None
    assert metadata == {"role": "legacy17"}


def test_twenty_three_features_normalization_bounds_and_checkpoint(tmp_path):
    normal = dict(mean=np.arange(23) / 20, std=np.linspace(0.5, 2, 23))
    policy = Policy(seed=5, obs_dim=23, normalization=normal)
    meter = CountingMeter()
    values = policy.predict(np.full((2, 3, 23), 100.0), meter=meter)
    assert values.shape == (2, 3, 6) and np.isfinite(values).all() and np.all(np.abs(values) <= 1)
    assert meter.counts == dict(neural_rows=6, backward_rows=0, optimizer_steps=0)
    assert sum(p.numel() for p in policy.parameters()) == 20358
    with pytest.raises(ValueError):
        policy.predict(np.zeros(17))
    with pytest.raises(ValueError):
        Policy(obs_dim=23, normalization=Policy().normalization)
    with pytest.raises(ValueError):
        Policy(obs_dim=18)
    path = tmp_path / "history.pt"
    policy.checkpoint(path, metadata={"features": "caller previous action"})
    loaded, _, metadata = Policy.from_checkpoint(path)
    assert loaded.obs_dim == 23 and loaded.hash() == policy.hash()
    assert metadata == {"features": "caller previous action"}
    assert np.array_equal(loaded.predict(np.zeros(23)), policy.predict(np.zeros(23)))


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_proximity_coefficient_matches_independent_flat_parameter_objective(reduction):
    obs, actions = data()
    parent = Policy(8)
    reference = Policy(8)
    anchor = torch.cat([p.detach().flatten().clone() for p in reference.parameters()])
    optimizer = torch.optim.Adam(reference.parameters(), lr=3e-4, foreach=False, fused=False)
    rng = np.random.default_rng(44)
    prox_values = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        rows = rng.integers(0, len(obs), size=4)
        mse = (
            (reference(torch.as_tensor(obs[rows])) - torch.as_tensor(actions[rows])).square().mean()
        )
        delta = torch.cat([p.flatten() for p in reference.parameters()]) - anchor
        prox = 2.0 * delta.square().sum() / (delta.numel() if reduction == "mean" else 1)
        (mse + prox).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
        optimizer.step()
        prox_values.append(float(prox.detach()))
    child, state, evidence = fit_policy(
        parent,
        obs,
        actions,
        updates=3,
        batch_size=4,
        eta_theta=4.0,
        seed=44,
        prox_reduction=reduction,
    )
    for actual, expected in zip(child.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=1e-6)
    assert evidence["parameter_count"] == 19590
    assert evidence["proximal_divisor"] == (19590 if reduction == "mean" else 1)
    np.testing.assert_allclose(
        [r["proximal_loss"] for r in evidence["learning_curve"]], prox_values, rtol=2e-6
    )
    assert state["fit_config"]["prox_reduction"] == reduction


@pytest.mark.parametrize("refresh,chunks", [(100, [99, 1, 1, 3]), (250, [250, 1, 2])])
def test_refresh_boundary_checkpoint_resume_is_exact_and_new_blocks_reset(
    tmp_path, refresh, chunks
):
    obs, actions = data()
    parent = Policy(2)
    settings = dict(
        batch_size=2, eta_theta=0.07, seed=41, prox_reduction="mean", anchor_refresh=refresh
    )
    whole, whole_state, whole_evidence = fit_policy(
        parent, obs, actions, updates=sum(chunks), **settings
    )
    child, state, rows, refreshes = parent, None, [], []
    for index, count in enumerate(chunks):
        child, state, evidence = fit_policy(
            child, obs, actions, state, updates=count, resume_fit=index > 0, **settings
        )
        rows.extend(evidence["learning_curve"])
        refreshes.extend(evidence["anchor_refreshes"])
        path = tmp_path / f"chunk{index}.pt"
        child.checkpoint(path, state)
        child, state, _ = Policy.from_checkpoint(path)
    equal_tree(child.state_dict(), whole.state_dict())
    equal_tree(state["adam"], whole_state["adam"])
    equal_tree(state["anchor"], whole_state["anchor"])
    assert state["rng"] == whole_state["rng"] and state["fit_updates"] == sum(chunks)
    assert (
        rows == whole_evidence["learning_curve"] and refreshes == whole_evidence["anchor_refreshes"]
    )
    assert (
        refreshes[0]["before_fit_update"] == refresh + 1 and rows[refresh]["proximal_loss"] == 0.0
    )
    assert state["anchor_policy_hash"] == whole_state["anchor_policy_hash"] != parent.hash()
    _, reset_state, reset_evidence = fit_policy(
        child, obs, actions, state, updates=1, batch_size=2, prox_reduction="sum", anchor_refresh=0
    )
    assert reset_state["anchor_policy_hash"] == child.hash() and reset_state["fit_updates"] == 1
    assert reset_evidence["learning_curve"][0]["proximal_loss"] == 0.0


def test_legacy_fit_config_resumes_with_sum_fixed_defaults():
    obs, actions = data()
    parent = Policy(2)
    whole, whole_state, _ = fit_policy(
        parent, obs, actions, updates=4, batch_size=3, eta_theta=0.0, seed=19
    )
    child, state, _ = fit_policy(
        parent, obs, actions, updates=2, batch_size=3, eta_theta=0.0, seed=19
    )
    del state["fit_config"]["prox_reduction"]
    del state["fit_config"]["anchor_refresh"]
    resumed, resumed_state, _ = fit_policy(
        child, obs, actions, state, updates=2, batch_size=3, eta_theta=0.0, seed=99, resume_fit=True
    )
    equal_tree(resumed.state_dict(), whole.state_dict())
    equal_tree(resumed_state["adam"], whole_state["adam"])
    assert resumed_state["rng"] == whole_state["rng"]


def test_full_dataset_mse_weights_last_batch_and_only_counts_forward_rows():
    obs, actions = data()
    policy = Policy(3)
    before = policy.hash()
    meter = CountingMeter()
    for value in policy.parameters():
        value.grad = None
    expected = float(
        np.mean((policy.predict(obs).astype(np.float64) - actions.astype(np.float64)) ** 2)
    )
    actual = full_dataset_mse(policy, obs, actions, meter=meter, batch_size=4)
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-12)
    assert meter.counts == dict(neural_rows=9, backward_rows=0, optimizer_steps=0)
    assert policy.hash() == before and all(p.grad is None for p in policy.parameters())
    with pytest.raises(ValueError):
        full_dataset_mse(policy, obs[:2], actions, meter=meter)
    assert meter.counts["neural_rows"] == 9
