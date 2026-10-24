"""Step 20: the policy learns only from the MPPI runs that stayed up and that it already explains best."""
import json

import numpy as np
import pytest
import torch

import scripts.training.train_walker_mdgps as training
from scripts.training.train_walker_mdgps import (TrainConfig, _batch_act, _new_actor, _select_learnable, main,
                                                 policy_state)


def _run(obs, labels, survived):
    return {"obs": obs, "labels": labels, "survived": survived}


def test_each_start_state_keeps_the_runs_it_fits_best_among_those_that_stayed_up():
    actor = _new_actor(0)
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((20, 17))
    own = _batch_act(actor)(obs)
    near, far = own + 0.01, own + 0.5
    runs = [_run(obs, far, True), _run(obs, near, True), _run(obs, own, False),   # start state 1
            _run(obs, own, False), _run(obs, own, False), _run(obs, own, False)]  # start state 2: all fell
    kept, stats = _select_learnable(runs, actor, TrainConfig(labeler="teacher", teacher_runs=3, teacher_keep=1))
    assert len(kept) == 1 and kept[0]["labels"] is near
    assert stats["start_states_without_a_kept_run"] == 1 and stats["runs_kept"] == 1
    assert stats["kept_fit_error_median"] < stats["rejected_fit_error_median"]


@pytest.mark.parametrize("bad", [dict(labeler="dagger", teacher_runs=2, teacher_keep=1),
                                 dict(labeler="teacher", teacher_runs=2),
                                 dict(labeler="teacher", teacher_runs=2, teacher_keep=3),
                                 dict(labeler="teacher", teacher_keep=1),
                                 dict(labeler="teacher", teacher_selector="selector.pt")])
def test_selection_settings_are_checked(tmp_path, bad):
    with pytest.raises(ValueError):
        main(tmp_path / "run", TrainConfig(**bad))


def test_a_fixed_selector_ranks_the_runs_in_place_of_the_student(tmp_path, monkeypatch):
    path = tmp_path / "selector.pt"
    selector = _new_actor(7)
    torch.save(policy_state(selector), path)
    rankers = []
    original = training._select_learnable

    def spy(runs, actor, cfg):
        rankers.append({k: v.clone() for k, v in actor.state_dict().items()})
        return original(runs, actor, cfg)

    monkeypatch.setattr(training, "_select_learnable", spy)
    main(tmp_path / "run", TrainConfig(labeler="teacher", K=8, H=4, steps=10, episodes_per_iter=2, workers=1,
                                       grad_steps=5, eval_every=100, eval_episodes=1, iterations=2, teacher_runs=3,
                                       teacher_keep=1, teacher_selector=str(path)))
    assert len(rankers) == 2
    for ranker in rankers:
        assert all(torch.equal(ranker[k], v) for k, v in selector.state_dict().items())


def test_a_selecting_run_learns_from_the_kept_runs_and_reports_all_of_mppi(tmp_path):
    cfg = TrainConfig(labeler="teacher", K=8, H=4, steps=10, episodes_per_iter=2, workers=1, grad_steps=5,
                      eval_every=100, eval_episodes=1, iterations=1, teacher_runs=3, teacher_keep=1)
    main(tmp_path / "run", cfg)
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[0])
    assert row["runs_kept"] + row["start_states_without_a_kept_run"] == 2
    assert row["collect_completed"] <= 6 and row["labels"] == 10 * row["runs_kept"]
