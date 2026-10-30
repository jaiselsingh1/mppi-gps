"""Analysis sentinels use synthetic evidence; they perform no simulator work."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics import report_independent_search as reporting


def _write(path, value):
    path.write_text(json.dumps(value))


def _fixture(out, *, mc_cost=80., active=True):
    cfg = json.loads((Path(__file__).parents[1] / "configs/independent_search_pilot.json").read_text())
    _write(out / "config.json", cfg)
    (out / "controllers").mkdir()
    events = [dict(query_id=i, arm=arm, phase="training", role="acquisition", executed_controls=1)
              for i, arm in enumerate(["shared", "memory", "mc", "polo"])]
    (out / "queries.jsonl").write_text("".join(json.dumps(row)+"\n" for row in events))
    episodes, curves, gates = [], [], []
    for seed in cfg["seeds"]:
        for arm, cost in (("memory", 100.), ("mc", mc_cost), ("polo", 110.)):
            for cycle, final in ((-1, False), (0, False), (3, True)):
                curves.append(dict(seed=seed, arm=arm, cycle=cycle, final=final,
                                   mean_cost=cost+.5, success_rate=1., model="test" if active else None,
                                   changed_actions=3 if arm == "mc" and active else 0,
                                   evaluated_executed_controls=360, executed_controls=900,
                                   costs=[cost, cost+1], charged_controls=220+cycle,
                                   preflight_controls=212))
            for index in range(2):
                episodes.append(dict(id=f"{seed}_{arm}_{index}", seed=seed, arm=arm, cycle=3,
                                     role="final", root=seed*10+index, cost=cost+index, success=True))
            episodes.append(dict(id=f"{seed}_{arm}_acq", seed=seed, arm=arm, cycle=0,
                                 role="acquisition", root=100+seed, cost=100, success=True))
            gate = dict(seed=seed, arm=arm, cycle=0, promoted=arm == "mc" and active, revoked=False)
            if arm != "memory":
                gate["calibration"] = dict(passed=active, paired_rmse=0., paired_sign_accuracy=1., pairs=[])
                if active:
                    np.savez(out / "controllers" / f"heldout_s{seed}_{arm}_c0.npz",
                             targets=np.array([20., 10.]), predictions=np.array([20., 10.]),
                             remaining=np.array([180, 100]), supported=np.ones(2, dtype=bool),
                             nn_predictions=np.array([21., 11.]), groups=np.array([1, 2]))
            gates.append(gate)
    for filename, value in (("episodes.json", episodes), ("learning_curves.json", curves), ("gates.json", gates)):
        _write(out / filename, value)
    _write(out / "work.json", dict(actual=reporting._ledger(out / "queries.jsonl")["actual"],
                                  completed=True, source_unchanged=True, wall_seconds=.1))
    _write(out / "preflight.json", dict(status="passed", wall_seconds=.01,
                                        counts=dict(total_controls=212, simulated_controls=176,
                                                    executed_controls=36)))


def test_hierarchical_final_and_shared_work_accounting(tmp_path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.setattr(reporting, "_plot", lambda *_: None)
    result = reporting.report(tmp_path)
    assert result["criteria"]["full_mechanism_chain_established"]
    assert result["work"]["actual_including_preflight"]["total_controls"] == 216
    assert all(row["total_controls"] == 214 for row in result["work"]["standalone_per_arm"].values())
    assert result["per_arm"]["mc"]["final_executed_controls"] == 1080
    assert result["final_comparisons"]["memory"]["training_seeds"] == 3
    assert result["final_comparisons"]["memory"]["hierarchical_bootstrap_95"][1] < 0


def test_prediction_success_cannot_hide_executed_control_failure(tmp_path, monkeypatch):
    _fixture(tmp_path, mc_cost=150.)
    monkeypatch.setattr(reporting, "_plot", lambda *_: None)
    result = reporting.report(tmp_path)
    assert result["criteria"]["valid_measured_long_horizon_prediction_and_pair_signs"]
    assert not result["criteria"]["final_executed_superiority_to_both_controls"]
    assert not result["criteria"]["exploration_or_efficiency_claim_established"]
    assert "negative result" in result["verdict"]


def test_absent_model_does_not_imply_influence_or_prediction_validity(tmp_path, monkeypatch):
    _fixture(tmp_path, active=False)
    monkeypatch.setattr(reporting, "_plot", lambda *_: None)
    result = reporting.report(tmp_path)
    assert not result["criteria"]["valid_measured_long_horizon_prediction_and_pair_signs"]
    assert not result["criteria"]["learned_influence_promoted"]
    assert not result["criteria"]["deployed_final_actions_changed"]


def test_missing_ledger_event_is_not_silently_ignored(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps(dict(query_id=1, executed_controls=4))+"\n")
    with pytest.raises(ValueError, match="missing or repeated"):
        reporting._ledger(path)
