"""Synthetic multi-case lifecycle: no physics, neural forwards or fitting."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import similarity_study as study


def test_declared_seed_roles_and_partition_are_disjoint():
    config = json.loads(Path("configs/studies/similarity_j1_0.json").read_text())
    roles = {}
    partitions = {}
    for group in config["groups"]:
        values = {group["acquisition_seed"], group["planner_seed"]}
        for learner in group["learners"]:
            values |= {learner["policy_seed"], learner["fit_seed"]}
        values |= set(group["learners"][0]["evaluation_seeds"])
        assert not roles.get(group["split"], set()).intersection(values)
        roles.setdefault(group["split"], set()).update(values)
        partitions.setdefault(group["split"], []).append(group)
    assert not roles["development"].intersection(roles["protected"])
    assert all(len(v) == 8 for v in partitions.values())


def test_multiple_case_catalogs_close_and_zero_motion_variation_protects_final(
    tmp_path, monkeypatch
):
    import torch

    from src.gps.mppi_gps import diagnostic_policies, search, similarity_execution, similarity_probe

    torch.set_num_threads(1)
    config = json.loads(Path("configs/studies/similarity_j1_0.json").read_text())
    for name in ("config.json", "manifest.json", "input_manifest.json"):
        (tmp_path / name).write_text("{}")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    np.savez(prepared / "normalization.npz", mean=np.zeros(17), std=np.ones(17))
    monkeypatch.setattr(study, "verify", lambda _: config)
    monkeypatch.setattr(study, "runtime_gate", lambda _: None)
    monkeypatch.setattr(study, "input_gate", lambda _: None)
    arrays, seen_groups, fits = {}, [], []

    class Meter:
        def __init__(self, out, _):
            self.out, self.serial = out, 0

        def record(self, *_, **__):
            self.serial += 1
            return f"event-{self.serial:09d}"

        def reserve_external(self, amount):
            assert amount >= 0

        def save_arrays(self, tag, **values):
            arrays[tag] = values
            return {"token": tag}

        def close(self):
            pass

        def accounting(self):
            return {"completed": {"fits": len(fits)}, "unknown_partial": {}}

    class Actor:
        def predict(self, obs, **_):
            return np.zeros((*np.shape(obs)[:-1], 6), dtype=np.float32)

    states = np.zeros((1001, 19))
    states[:, 0] = np.arange(1001) * 0.008
    states[:, 1] = states[:, 0] * 1.5
    states[:, 2], states[:, 10] = 1.2, 1.5
    physical = dict(
        states=states,
        warmstarts=np.zeros((1001, 9)),
        integrations=np.zeros((1001, 91)),
        integration_spec=16383,
        observations=np.zeros((1001, 17)),
        actions=np.zeros((1000, 6)),
        done=np.zeros(1000, dtype=bool),
    )

    class Runner:
        def __init__(self, *_):
            self.env = object()

        def trace(self, tag, *_, **__):
            return dict(tag=tag, controls=1000, walking=True, survived=True, cost=0.0)

        def teacher(self, tag, *_):
            seen_groups.append(tag)
            return self.trace(tag)

        def load_trace(self, _):
            return physical

        def planner_context(self, *_):
            return {}

        def persist_probe(self, value):
            return value

        def restore(self, *_):
            raise AssertionError("The fake decision never invokes physics")

        def contacts(self, *_):
            raise AssertionError("The fake decision never invokes native contacts")

        def close(self):
            pass

    class Planner:
        def __init__(self, *_):
            pass

    def parent_prepare(seed, fitseed, *_):
        return dict(
            actor=Actor(),
            global_torch=torch.get_rng_state().clone(),
            bindings=dict(
                checkpoint={"file": f"parent_{seed}.pt"}, tape={"file": f"tape_{fitseed}.npy"}
            ),
        )

    def fit_case(parent, obs, actions, out, _):
        assert obs.shape == (1000, 17) and actions.shape == (1000, 6)
        fits.append(out.name)
        return dict(
            actor=Actor(),
            checkpoint={"file": str(out / "child.pt")},
            receipt_file=str(out / "fit_receipt.json"),
        )

    monkeypatch.setattr(similarity_execution, "JMeter", Meter)
    monkeypatch.setattr(similarity_execution, "NativeRunner", Runner)
    monkeypatch.setattr(similarity_execution, "parent_prepare", parent_prepare)
    monkeypatch.setattr(similarity_execution, "fit_case", fit_case)
    monkeypatch.setattr(
        diagnostic_policies.LegacySACActor, "from_checkpoint", lambda *_, **__: Actor()
    )
    monkeypatch.setattr(search, "IndependentMPPI", Planner)
    monkeypatch.setattr(similarity_probe, "run_decision", lambda *args, **_: {"request": args[4]})
    monkeypatch.setattr(similarity_probe, "summarize_probe", lambda *_, **__: {"passed": True})
    monkeypatch.setattr(study, "read_array", lambda _, ref, key, **__: arrays[ref["token"]][key])
    result = study.scientific_worker(tmp_path)
    assert result["status"] == "CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE", result
    assert (
        len(fits)
        == len(result_cases := json.loads((tmp_path / "worker/cases.json").read_text()))
        == 32
    )
    assert {r["split"] for r in result_cases} == {"development"}
    assert len(seen_groups) == 8
    assert result["protected_test_revealed"] is False
    assert result["objective_passed"] is False
    assert len(json.loads((tmp_path / "worker/traces.json").read_text())) > 32
    assert len((tmp_path / "worker/trace_catalog.jsonl").read_text().splitlines()) > 32
    original = copy.deepcopy(json.loads((tmp_path / "worker_result.json").read_text()))
    with pytest.raises(FileExistsError):
        study.scientific_worker(tmp_path)
    assert json.loads((tmp_path / "worker_result.json").read_text()) == original
