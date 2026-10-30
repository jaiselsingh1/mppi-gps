"""Zero-physics protocol checks for the ending comparison driver.

    Task.transition and Task.rollout are forbidden throughout these checks.
    Acquisition states below are named mock contexts, not simulator evidence.
"""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/training/run_ending_actor_comparison.py"
CONFIG = ROOT / "configs/ending_actor_comparison.json"


@pytest.fixture
def driver(monkeypatch):
    specification = importlib.util.spec_from_file_location("ending_driver_pure_tests", DRIVER)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure driver tests must consume zero simulator controls")

    monkeypatch.setattr(module.Task, "transition", forbidden)
    monkeypatch.setattr(module.Task, "rollout", forbidden)
    monkeypatch.setattr(module.IndependentMPPI, "plan", forbidden)
    return module


def _configuration():
    return json.loads(CONFIG.read_text())


def _mock_experiment(driver):
    config = _configuration()
    experiment = driver.Experiment.__new__(driver.Experiment)
    experiment.cfg = config
    total = config["task"]["gate"] + config["task"]["delay"] + 1
    # No Task instance, reset, or transition is used to generate these contexts.
    experiment.task = SimpleNamespace(T=total, gate=config["task"]["gate"],
        delay=config["task"]["delay"], reset=lambda seed: np.array([0., .3]))
    experiment.paid = {}
    return experiment


def _mock_pool(experiment, seed):
    pool = {"train": [], "heldout": []}
    acquisition = experiment.cfg["acquisition"]
    for label in ("short", "long"):
        for index in range(acquisition["episodes_per_horizon"]):
            states = np.column_stack((np.arange(experiment.task.T + 1),
                                      np.full(experiment.task.T + 1, .3)))
            row = {"name": f"acq_{label}_{seed}_{index}", "reset_seed": 20000 + 100 * seed + index,
                   "states": states}
            key = "train" if index < acquisition["train_per_horizon"] else "heldout"
            pool[key].append(row)
    return pool


def _roots(contexts):
    # This is the exact root identity used by Experiment.fit, deliberately
    # collapsing paired short/long acquisition traces from the same reset.
    return {str(context["ancestry"].get("reset_seed", context["ancestry"].get("episode")))
            for context in contexts}


def test_actual_context_ancestry_is_root_disjoint_for_all_frozen_fits(driver):
    experiment = _mock_experiment(driver)
    all_training, all_heldout = set(), set()
    for seed in experiment.cfg["seeds"]:
        pool = _mock_pool(experiment, seed)
        assert _roots([{"ancestry": row} for row in pool["train"]]).isdisjoint(
            _roots([{"ancestry": row} for row in pool["heldout"]]))
        for update in range(experiment.cfg["search"]["updates"]):
            ordinary = experiment.starts(seed, update, pool, "train")
            extra = experiment.starts(seed, update, pool, "critic_train")
            heldout = experiment.starts(seed, update, pool, "critic_holdout")
            validation = experiment.starts(seed, update, pool, "validation")
            training_roots = _roots(ordinary + extra)
            heldout_roots = _roots(heldout)
            assert training_roots.isdisjoint(heldout_roots), (seed, update)
            assert training_roots.isdisjoint(_roots(validation)), (seed, update)
            assert len(ordinary) == len(extra) == len(heldout) == len(validation) == 4
            all_training.update(training_roots)
            all_heldout.update(heldout_roots)
    # A heldout reset is never a fitting root in a different update or seed.
    assert all_training.isdisjoint(all_heldout)


def test_shared_experience_and_audit_reserve_apply_to_every_arm(driver):
    experiment = _mock_experiment(driver)
    experiment.cfg = deepcopy(experiment.cfg)
    experiment.cfg["budget"]["controls_per_seed_arm"] = 1000
    experiment.paid = {"shared/global": 218, "shared/seed0": 282}
    before = deepcopy(experiment.paid)
    experiment.reserve("seed0/full_return", 0)
    assert experiment.paid == before
    with pytest.raises(driver.BudgetStop, match="audit reserve"):
        experiment.reserve("seed0/full_return", 1)
    with pytest.raises(driver.BudgetStop, match="audit reserve"):
        experiment.reserve("shared/global", 1)
    # Arm-specific use is charged only to that arm; common acquisition goes
    # into every arm's comparison denominator for the corresponding seed.
    experiment.paid = {"shared/global": 100, "shared/seed0": 100,
                       "seed0/full_return": 300}
    experiment.reserve("seed0/critic_prefix", 300)
    with pytest.raises(driver.BudgetStop):
        experiment.reserve("shared/seed0", 1)


def test_local_gap_threshold_has_a_cost_definition_bound_without_outcomes():
    config = _configuration()
    task, planner = config["task"], config["planner"]
    total = task["gate"] + task["delay"] + 1
    assert planner["H"] <= task["delay"]
    # Every eligible pre-gate H-window finishes before the ending transition.
    assert max(range(task["gate"])) + planner["H"] - 1 < total - 1
    # The independent +1 plan reaches exactly the greed-only optimistic path.
    # Its positive initial y remains above the good-lane threshold, eliminating
    # detour. Its only excess over the bound is effort <= .02 per slot.
    assert .2 > task["ending_y"]
    assert task["move"] > 0
    assert config["qualification"]["max_gap_per_control"] >= task["effort"]
    assert config["qualification"]["max_gap_per_control"] == .04
    assert config["search"]["maxnorm"] == .6
    assert config["search"]["lr"] == .4


def test_main_orders_qualification_then_shared_acquisition_then_actor_training():
    module = ast.parse(DRIVER.read_text())
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = {name: [] for name in ("qualify", "acquire", "train")}
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in calls:
            calls[node.func.attr].append(node.lineno)
    assert all(len(lines) == 1 for lines in calls.values())
    assert calls["qualify"][0] < calls["acquire"][0] < calls["train"][0]


def test_planner_has_no_actor_or_critic_input_in_driver():
    module = ast.parse(DRIVER.read_text())
    planner_calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "IndependentMPPI"]
    assert len(planner_calls) == 1
    assert len(planner_calls[0].args) == 1
    assert {argument.arg for argument in planner_calls[0].keywords} == {None, "seed"}
    assert set(_configuration()["planner"]) == {"K", "H", "temp", "sigma"}
