"""The reviewer's exact multimodal saddle; no MuJoCo or training needed."""
import torch

from src.gps.decoupled import synthesize_verified


def saddle_cost(plans, _rows):
    u, v = plans[:, 0, 0], plans[:, 0, 1]
    return (1. + u.square()) * (v.square() - 1.).square()


def test_exact_symmetric_saddle_rejects_bad_mean_and_reuses_one_winning_mode():
    nominal = torch.zeros((1, 1, 2), dtype=torch.float64)
    sampled = torch.tensor([[[[1., 1.]], [[1., -1.]]]], dtype=torch.float64)
    scores = saddle_cost(sampled.flatten(0, 1), None).reshape(1, 2)
    baseline = saddle_cost(nominal, None)
    low, high = torch.full((2,), -2., dtype=torch.float64), torch.full((2,), 2., dtype=torch.float64)
    calls = []

    def verify(plans, rows):
        calls.append((plans.clone(), rows.clone()))
        return saddle_cost(plans, rows)

    rejected = synthesize_verified(sampled, scores, nominal, baseline, low, high, verify)
    torch.testing.assert_close(rejected.mean_plans, torch.tensor([[[1., 0.]]], dtype=torch.float64), rtol=0, atol=0)
    assert baseline.item() == 1.
    assert rejected.expected_candidate_cost.item() == 0.
    assert rejected.mean_costs.item() == 2.
    assert not rejected.accepted.item()
    assert rejected.bimodal_collapse.item()
    assert rejected.costs.item() == 1.

    calls.clear()
    rescued = synthesize_verified(sampled, scores, nominal, baseline, low, high, verify,
                                  synthesis_mode="medoid_fallback")
    torch.testing.assert_close(rescued.labels, torch.tensor([[1., 1.]], dtype=torch.float64), rtol=0, atol=0)
    assert rescued.accepted.item()
    assert rescued.costs.item() == 0. < .1
    assert rescued.source_codes.item() == 2
    assert rescued.anchor_indices.item() == 0  # Stable tie break, no arbitrary averaging.
    assert len(calls) == rescued.verification_callback_calls == rescued.verification_rollouts == 1
    assert rescued.diagnostics()["medoid_rescue_rate"] == 1.
    assert rescued.diagnostics()["medoid_rescue_denominator"] == 1
