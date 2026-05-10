from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gps_train import make_collection_bias
from src.gps.coupling import make_policy_filter_coupling
from src.utils.config import GPSConfig


class ZeroPolicy(torch.nn.Module):
    def __init__(self, act_dim: int = 1) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((obs.shape[0], self.act_dim), dtype=obs.dtype, device=obs.device) + self.anchor * 0.0


def test_make_collection_bias_warmup_is_inactive() -> None:
    cfg = GPSConfig(coupling_mode="track", coupling_warmup_iters=2)

    prior, coupling = make_collection_bias(ZeroPolicy(), cfg, it=0, policy_trust=1.0)

    assert prior is None
    assert coupling is None


def test_make_collection_bias_track_builds_tracking_prior_only() -> None:
    cfg = GPSConfig(coupling_mode="track", coupling_warmup_iters=0, lambda_policy_track=0.5)

    prior, coupling = make_collection_bias(ZeroPolicy(), cfg, it=0, policy_trust=1.0)

    assert prior is not None
    assert coupling is None
    states = np.zeros((2, 1, 4), dtype=np.float32)
    actions = np.ones((2, 1, 1), dtype=np.float32)
    np.testing.assert_allclose(prior(states, actions), np.full((2,), 0.5))


def test_make_collection_bias_filter_builds_tracking_prior_and_filter() -> None:
    cfg = GPSConfig(coupling_mode="filter", coupling_warmup_iters=0, lambda_policy_track=0.5)

    prior, coupling = make_collection_bias(ZeroPolicy(), cfg, it=0, policy_trust=1.0)

    assert prior is not None
    assert coupling is not None


def test_make_collection_bias_bc_collection_stays_plain_bc() -> None:
    cfg = GPSConfig(
        collection_mode="bc",
        coupling_mode="track",
        coupling_warmup_iters=0,
        lambda_policy_track=0.5,
    )

    prior, coupling = make_collection_bias(ZeroPolicy(), cfg, it=0, policy_trust=1.0)

    assert prior is None
    assert coupling is None


def test_make_collection_bias_rejects_removed_modes() -> None:
    cfg = GPSConfig(coupling_mode="cost", coupling_warmup_iters=10)

    try:
        make_collection_bias(ZeroPolicy(), cfg, it=0, policy_trust=1.0)
    except ValueError as exc:
        assert "track" in str(exc)
        assert "filter" in str(exc)
    else:
        raise AssertionError("removed coupling mode was accepted")


def test_policy_filter_masks_far_samples_without_biasing_kept_scores() -> None:
    coupling = make_policy_filter_coupling(
        ZeroPolicy(),
        min_fraction=0.0,
        keep_fraction=0.5,
        min_n_eff=0.0,
        max_weight=1.0,
    )
    states = np.zeros((4, 1, 4), dtype=np.float32)
    actions = np.array([[[0.0]], [[1.0]], [[2.0]], [[3.0]]], dtype=np.float32)
    base_score = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    result = coupling(
        states=states,
        actions=actions,
        costs=base_score,
        base_score=base_score,
        lam=0.1,
    )

    np.testing.assert_allclose(result["score"][:2], base_score[:2])
    assert np.isinf(result["score"][2])
    assert np.isinf(result["score"][3])


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
