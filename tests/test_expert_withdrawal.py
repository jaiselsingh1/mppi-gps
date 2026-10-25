"""Finite-objective guards and real tiny expert-withdrawal training runs."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.training import train_expert_withdrawal as training
from src.gps.expert_value import ExpertValueTrainer


def _tiny(**changes):
    cfg = replace(training.WithdrawalConfig(), total_steps=8, prefix_schedule=(8,),
                  updates_per_stage=2, pairs=2, train_starts=2, calibration_starts=1,
                  verification_starts=1, value_labels=2, value_updates=5, value_batch_size=2,
                  hidden_dim=8, workers=1, eval_episodes=2)
    return replace(cfg, **changes)


@pytest.mark.parametrize("changes", [
    {"prefix_schedule": ()}, {"prefix_schedule": (1, 4)},
    {"prefix_schedule": (1, 1, 8)}, {"prefix_schedule": (8, 1)},
    {"prefix_schedule": (0, 8)}, {"prefix_schedule": (True, 8)},
    {"seed": -1}, {"seed": True}, {"seed": 2 ** 32},
    {"train_starts": 20_001}, {"verification_starts": 20_001},
    {"value_labels": 1}, {"teacher_tolerance": -1.},
    {"gamma": 1.}, {"gamma": np.nan}, {"pairs": 0}, {"teacher_H": 1},
])
def test_stage_seed_and_acquisition_guards_reject_invalid_configuration(changes):
    with pytest.raises(ValueError):
        training.validate(_tiny(**changes))


def test_valid_seed_schedule_fits_disjoint_nonreserved_blocks():
    cfg = _tiny(seed=2, total_steps=16, prefix_schedule=(1, 16))
    training.validate(cfg)
    count = cfg.updates_per_stage * len(cfg.prefix_schedule)
    base = 11_000_000 + cfg.seed * 100_000
    blocks = [set(range(base + offset, base + offset + count * n))
              for offset, n in [(0, cfg.train_starts), (20_000, cfg.calibration_starts),
                                (40_000, cfg.verification_starts)]]
    blocks.append(set(range(base + 60_000, base + 60_000 + cfg.eval_episodes)))
    for i, block in enumerate(blocks):
        assert block.isdisjoint(range(5_000_000, 5_000_200))
        assert all(block.isdisjoint(other) for other in blocks[i + 1:])


def test_prefix_objective_requires_grounded_expert_and_has_exact_zero_terminal():
    prefix = SimpleNamespace(steps=2, terminated=False, gamma=.5, unhealthy_cost=5.,
                             cost=3., endpoint_obs=np.zeros(17))
    with pytest.raises(RuntimeError, match="grounded expert"):
        training.prefix_objective(prefix, total_steps=8, value=None)
    ungrounded = ExpertValueTrainer(horizon=8, hidden_dim=8).snapshot()
    with pytest.raises(RuntimeError, match="grounded"):
        training.prefix_objective(prefix, total_steps=8, value=ungrounded)
    prefix.steps = 8
    poison = SimpleNamespace(predict=lambda *a: pytest.fail("A full-horizon actor queried an expert value"))
    assert training.prefix_objective(prefix, total_steps=8, value=poison) == 3.
    prefix.steps = 9
    with pytest.raises(ValueError):
        training.prefix_objective(prefix, total_steps=8, value=poison)


def test_prefix_fall_adds_only_remaining_finite_absorption_without_value_query():
    prefix = SimpleNamespace(steps=2, terminated=True, gamma=.5, unhealthy_cost=5.,
                             cost=26., endpoint_obs=np.zeros(17))
    poison = SimpleNamespace(predict=lambda *a: pytest.fail("A fallen prefix queried an expert value"))
    expected = 26. + .5 ** 2 * sum(5. * .5 ** j for j in range(6))
    assert training.prefix_objective(prefix, total_steps=8, value=poison) == pytest.approx(expected)


def _rows(directory):
    return [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]


def _check_final_artifacts(directory, updates):
    status = json.loads((directory / "status.json").read_text())
    assert status["phase"] == "complete" and status["curriculum_complete"] and status["source_unchanged"]
    assert status["completed_updates"] == updates
    assert status["standalone_walking_gate"] is None  # tiny horizons cannot establish the walking gate
    assert len(list(directory.glob("policy_????.pt"))) == updates
    assert (directory / "policy_initial.pt").exists() and (directory / "policy_final.pt").exists()
    final = json.loads((directory / "final.json").read_text())
    assert final["standalone"] and final["summary"]["episodes"] == 2
    assert np.isfinite(final["summary"]["finite_discounted_cost"])
    protocol = json.loads((directory / "protocol.json").read_text())
    assert {"src/envs/base.py", "scripts/runners/run_walker2d.py"} <= protocol["source_sha256"].keys()
    for name, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((directory / "source" / name).read_bytes()).hexdigest() == expected
    return status


def test_full_horizon_only_cpu_run_has_no_expert_queries_or_value_fit(tmp_path, monkeypatch):
    prefixes, takeovers = [], []
    original_prefix, original_takeover = training.rollout_policy_prefix, training.takeover_objective

    def prefix(*args, **kwargs):
        result = original_prefix(*args, **kwargs)
        prefixes.append(result)
        return result

    def takeover(*args, **kwargs):
        result = original_takeover(*args, **kwargs)
        takeovers.append(result)
        return result

    monkeypatch.setattr(training, "rollout_policy_prefix", prefix)
    monkeypatch.setattr(training, "takeover_objective", takeover)
    monkeypatch.setattr(training, "independent_teacher_suffix", lambda *a, **k: pytest.fail("Full-N run queried MPPI"))
    monkeypatch.setattr(training.ExpertValueTrainer, "fit", lambda *a, **k: pytest.fail("Full-N run fit an expert value"))
    out = tmp_path / "full"
    training.run(out, _tiny())
    status = _check_final_artifacts(out, 2)
    rows = _rows(out)
    assert [row["prefix_steps"] for row in rows] == [8, 8]
    assert all(row["expert_value_fit"] is None and not row["expert_assistance"] for row in rows)
    assert all(row["teacher_reference_cost"] is None for row in rows)
    assert len(prefixes) == 20 and len(takeovers) == 12
    assert all(p.steps == 8 and p.actions.shape == (8, 6) for p in prefixes)
    assert all(t.prefix.steps == 8 and t.prefix.actions.shape == (8, 6) and t.suffix is None for t in takeovers)
    ledger = status["ledger"]
    assert ledger["policy_controls"] == (20 + 12) * 8
    assert ledger["standalone_evaluation_controls"] == 4 * 8
    assert all(ledger[key] == 0 for key in ("teacher_controls", "teacher_candidate_controls",
                                           "teacher_planner_calls", "teacher_optimization_iterations"))
    assert not list(out.glob("expert_data_*.npz")) and not list(out.glob("expert_value_*.pt"))


def test_mixed_cpu_run_withdraws_teacher_and_freezes_value_during_search(tmp_path, monkeypatch):
    fits, scores, takeovers = [], [], []
    original_fit = training.ExpertValueTrainer.fit
    original_predict = training.ExpertValueTrainer.snapshot
    original_takeover = training.takeover_objective

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        fits.append(result)
        return result

    def snapshot(*args, **kwargs):
        result = original_predict(*args, **kwargs)
        initial = {k: v.clone() for k, v in result.model.state_dict().items()}

        def predict(obs, remaining):
            value = result.predict(obs, remaining)
            assert not any(p.requires_grad for p in result.model.parameters())
            for key, tensor in result.model.state_dict().items():
                assert np.array_equal(tensor.numpy(), initial[key].numpy())
            scores.append(remaining)
            return value
        return SimpleNamespace(predict=predict)

    def takeover(*args, **kwargs):
        result = original_takeover(*args, **kwargs)
        takeovers.append(result)
        return result

    monkeypatch.setattr(training.ExpertValueTrainer, "fit", fit)
    monkeypatch.setattr(training.ExpertValueTrainer, "snapshot", snapshot)
    monkeypatch.setattr(training, "takeover_objective", takeover)
    out = tmp_path / "mixed"
    training.run(out, _tiny(total_steps=16, prefix_schedule=(1, 16), teacher_K=4, teacher_H=2,
                           teacher_initial_iterations=1, teacher_tolerance=1e6))
    status = _check_final_artifacts(out, 3)
    rows = _rows(out)
    assert [row["prefix_steps"] for row in rows] == [1, 16, 16]
    assert len(fits) == 1 and fits[0]["source"] == "independent_mppi_continuation"
    assert scores and set(scores) == {15}
    assert rows[0]["expert_assistance"] and rows[0]["expert_value_fit"] is not None
    assert all(not r["expert_assistance"] and r["expert_value_fit"] is None for r in rows[1:])
    assert sum(t.suffix is not None for t in takeovers) == 6
    assert all(t.suffix is None for t in takeovers if t.prefix_steps == 16)
    for key in ("teacher_controls", "teacher_candidate_controls", "teacher_planner_calls"):
        assert rows[0]["ledger_cumulative"][key] > 0
        assert rows[0]["ledger_cumulative"][key] == rows[-1]["ledger_cumulative"][key] == status["ledger"][key]
    assert len(list(out.glob("expert_data_*.npz"))) == len(list(out.glob("expert_value_*.pt"))) == 1
