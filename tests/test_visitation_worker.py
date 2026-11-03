"""Synthetic V stage/gate tests; no reference/learner forwards or native physics."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.mppi_gps import visitation_worker as worker
from src.gps.mppi_gps.provenance import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def recipe():
    return json.loads((ROOT / "configs/studies/student_visitation_s1_2_v_1.json").read_text())


@pytest.fixture
def saved_contexts(tmp_path):
    config = recipe()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    context = dict(
        state=np.zeros(19),
        warmstart=np.zeros(9),
        integration=np.zeros(91),
        observation=np.zeros(17),
        spec=np.asarray(16383),
        dimensions=np.asarray([9, 9, 6]),
    )
    np.savez(prepared / "context.npz", **context)
    np.savez(
        prepared / "normalization.npz",
        mean=np.zeros(17, dtype=np.float32),
        std=np.ones(17, dtype=np.float32),
    )
    seeds = config["validity"]["positive_control_reference_reset_seeds"]
    anchors = config["validity"]["anchors"]
    catalog = dict(
        canonical_contexts={str(seed): dict(file="context.npz") for seed in seeds},
        parent_initial_contexts={
            str(seed): {str(s): dict(file="context.npz") for s in range(3)} for seed in seeds
        },
        anchors=deepcopy(anchors),
        episodes={
            a["source_tag"]: dict(file="unused.npz", controls=a["source_controls"]) for a in anchors
        },
        parents={str(s): dict(stream=s) for s in range(3)},
        normalization=dict(file="normalization.npz"),
    )
    return tmp_path, prepared, catalog, config


def outcome(anchor, stage, walking=False, survived=False):
    return dict(
        stage=stage,
        source_tag=anchor["source_tag"],
        stream=anchor["stream"],
        panel=anchor["panel"],
        source_case=anchor["source_case"],
        anchor_ordinal=anchor["anchor_ordinal"],
        t=anchor["control"],
        walking=walking,
        survived=survived,
    )


def screening_rows(config, primary=(0, 1), confirmation=(0, 1)):
    rows = []
    for anchor in config["validity"]["anchors"]:
        rows.append(outcome(anchor, "parent_replay"))
        selected = primary if anchor["panel"] == "primary" else confirmation
        walking = anchor["stream"] in selected and anchor["anchor_ordinal"] in (1, 2, 3)
        rows.append(outcome(anchor, "reference_takeover", walking, walking))
    return rows


def test_static_gate_compares_all96saved_parent_contexts(saved_contexts):
    _, prepared, catalog, config = saved_contexts
    result = worker.static_context_gate(prepared, catalog, config)
    assert result["passed"] and len(result["comparisons"]) == 96
    assert (
        result["native_controls"]
        == result["network_forwards"]
        == result["policy_constructions"]
        == 0
    )


@pytest.mark.parametrize(
    "field", ["state", "warmstart", "integration", "observation", "spec", "dimensions"]
)
def test_static_gate_cannot_ignore_one_parent_context(saved_contexts, field):
    _, prepared, catalog, config = saved_contexts
    context = dict(np.load(prepared / "context.npz"))
    context[field] = context[field].copy()
    context[field].flat[0] += 1
    np.savez(prepared / "changed.npz", **context)
    seed = next(iter(catalog["parent_initial_contexts"]))
    catalog["parent_initial_contexts"][seed]["2"]["file"] = "changed.npz"
    with pytest.raises(ValueError, match="context"):
        worker.static_context_gate(prepared, catalog, config)


def test_initial_success_cannot_admit_failed_noninitial_takeovers():
    config = recipe()
    rows = screening_rows(config, (), ())
    for row in rows:
        if row["stage"] == "reference_takeover" and row["anchor_ordinal"] == 0:
            row.update(walking=True, survived=True)
    result = worker.reference_screen(rows)
    assert not result["passed"] and result["same_qualifying_streams"] == []


def test_same_two_noninitial_streams_can_pass_both_panels():
    result = worker.reference_screen(screening_rows(recipe()))
    assert result["passed"] and result["same_qualifying_streams"] == [0, 1]


def test_different_qualifying_streams_do_not_confirm():
    result = worker.reference_screen(screening_rows(recipe(), (0, 1), (1, 2)))
    assert not result["passed"] and result["same_qualifying_streams"] == [1]


def test_survival_regression_blocks_walking_improvement():
    rows = screening_rows(recipe())
    for row in rows:
        if row["stream"] == 2 and row["stage"] == "parent_replay" and row["t"] > 0:
            row["survived"] = True
    result = worker.reference_screen(rows)
    assert not result["passed"] and result["regressions"]


def test_unpaired_anchor_cannot_be_counted_as_a_comparison():
    rows = screening_rows(recipe())
    next(row for row in rows if row["stage"] == "reference_takeover" and row["t"] > 0)[
        "source_tag"
    ] = "wrong"
    with pytest.raises(ValueError, match="not paired"):
        worker.reference_screen(rows)


@pytest.fixture
def staged_worker(saved_contexts, monkeypatch):
    run, _, catalog, config = saved_contexts
    events = []
    controls = dict(positive_failure=False, replay_failure=False)
    monkeypatch.setitem(
        sys.modules,
        "src.gps.mppi_gps.visitation_inputs",
        SimpleNamespace(verify_prepared=lambda path: catalog),
    )

    class Actor:
        _generator = SimpleNamespace(get_state=lambda: np.zeros(1, dtype=np.uint8))

        def hash(self):
            return "a" * 64

    def load_parent(run, descriptor, normalization, meter):
        events.append("load_parent")
        state = {"rng": {}}
        return (
            Actor(),
            state,
            dict(
                file="fake.pt",
                bytes=1,
                sha256="b" * 64,
                actor_hash="a" * 64,
                optimizer_sha256=fingerprint(state),
                private_rng_sha256=fingerprint(Actor._generator.get_state()),
            ),
        )

    def load_reference(*args):
        assert events.count("parent_replay") == 96
        events.append("load_reference")
        return Actor(), dict(file="ref.pt", bytes=1, sha256="c" * 64, actor_hash="a" * 64)

    monkeypatch.setattr(worker, "_load_parent", load_parent)
    monkeypatch.setattr(worker, "_load_reference", load_reference)

    class Native:
        def __init__(self, *args):
            self.catalog = catalog
            self.env = SimpleNamespace(close=lambda: events.append("close"))

        def episode(self, tag):
            return dict(tag=tag)

        def canonical(self, seed):
            return dict(seed=seed)

        def closed_loop(self, tag, actor, **kwargs):
            stage = kwargs["stage"]
            if stage == "reference_reset":
                assert events.count("parent_replay") == 96
            if stage == "reference_takeover":
                assert events.count("reference_reset") == 32
            events.append(stage)
            if controls["replay_failure"] and events.count("parent_replay") == 3:
                raise ValueError("synthetic replay mismatch")
            is_reset = stage == "reference_reset"
            walking = is_reset and not (
                controls["positive_failure"]
                and kwargs.get("seed")
                in config["validity"]["positive_control_reference_reset_seeds"][:2]
            )
            count = (
                1000
                if is_reset
                else (
                    catalog["episodes"][kwargs["source_tag"]]["controls"] - kwargs["t"]
                    if stage == "parent_replay"
                    else 1
                )
            )
            return dict(
                tag=tag,
                stage=stage,
                source_tag=kwargs.get("source_tag"),
                t=kwargs.get("t", 0),
                h=1000,
                actor=kwargs["actor_alias"],
                seed=kwargs.get("seed"),
                controls=count,
                walking=walking,
                survived=walking,
                cost=float(count),
                raw_cost=float(count),
                native_cost=float(count),
                native_terminal=0.0,
                absorbing_cost=0.0,
                mae=0.0,
                done=not walking,
                trace={},
            )

    monkeypatch.setattr(worker, "NativeVisitation", Native)
    meter = SimpleNamespace(reserve_external=lambda amount: None)
    return run, config, meter, events, controls


def test_complete_negative_screen_keeps_all96takeovers(staged_worker):
    run, config, meter, events, _ = staged_worker
    result = worker.runV(run, config, meter)
    assert result["status"] == "COMPLETED_V_FIXED_ROSTER"
    assert not result["reference_screen"]["passed"]
    assert events.count("parent_replay") == 96 and events.count("reference_reset") == 32
    assert events.count("reference_takeover") == 96
    rows = json.loads((run / "worker/outcomes.json").read_text())["rows"]
    assert len(rows) == 224 and not result["learning_launched"] and not result["followup_launched"]
    assert len((run / "worker/outcomes.jsonl").read_text().splitlines()) == 224


def test_negative_positive_control_finishes32then_stops_before_takeovers(staged_worker):
    run, config, meter, events, controls = staged_worker
    controls["positive_failure"] = True
    result = worker.runV(run, config, meter)
    assert result["status"] == "STOPPED_REFERENCE_POSITIVE_CONTROL_FAILED"
    assert events.count("reference_reset") == 32 and events.count("reference_takeover") == 0
    assert result["phase_status"]["takeovers"] == "not_started"
    assert len(json.loads((run / "worker/outcomes.json").read_text())["rows"]) == 128


def test_replay_failure_preserves_prefix_without_reference_work(staged_worker):
    run, config, meter, events, controls = staged_worker
    controls["replay_failure"] = True
    with pytest.raises(ValueError, match="synthetic replay mismatch"):
        worker.runV(run, config, meter)
    assert "load_reference" not in events and "reference_reset" not in events
    assert len(json.loads((run / "worker/outcomes.json").read_text())["rows"]) == 2
    result = json.loads((run / "worker/phase_status.json").read_text())
    assert result["status"] == "STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE"
    assert result["stopped_during"] == "parent_replay"


def test_work_census_matches_frozen_v_recipe():
    assert worker.work_limits() == recipe()["work_limits"]
    assert worker.work_limits()["optimizer_steps"] == worker.work_limits()["backward_rows"] == 0


@pytest.fixture
def parent_package(tmp_path):
    import torch

    from src.gps.mppi_gps.consequence_learning import FIT_CONFIG
    from src.gps.mppi_gps.policy import Policy
    from src.gps.mppi_gps.provenance import sha

    prepared = tmp_path / "prepared"
    (prepared / "parents").mkdir(parents=True)
    actor = Policy(seed=7)
    state = actor.state_dict()
    names = [name for name in state if name.startswith("network.")]
    incoming = dict(
        schema="walker-adam-proximal-fit-v1",
        adam=dict(
            state={
                i: dict(
                    step=torch.tensor(2000.0),
                    exp_avg=torch.zeros_like(state[name]),
                    exp_avg_sq=torch.zeros_like(state[name]),
                )
                for i, name in enumerate(names)
            },
            param_groups=[
                dict(params=list(range(6)), lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            ],
        ),
        rng=np.random.PCG64(3).state,
        updates=2000,
        fit_updates=2000,
        policy_hash=actor.hash(),
        parent_policy_hash="0" * 64,
        anchor_policy_hash=actor.hash(),
        anchor=state,
        parameter_names=names,
        fit_config=FIT_CONFIG,
        fit_data=dict(observations_sha256="1" * 64, actions_sha256="2" * 64, rows=4000),
    )
    payload = dict(
        schema="walker-tanh-policy-v1",
        seed=actor.seed,
        obs_dim=17,
        policy_hash=actor.hash(),
        state_dict=state,
        private_rng_state=actor._generator.get_state(),
        training=actor.training,
        optimizer_state=incoming,
        metadata={},
    )
    path = prepared / "parents/s0_parent.pt"
    torch.save(payload, path)
    global_rng = torch.get_rng_state().numpy()
    np.save(prepared / "parents/global.npy", global_rng)
    descriptor = dict(
        file="parents/s0_parent.pt",
        bytes=path.stat().st_size,
        sha256=sha(path),
        actor_hash=actor.hash(),
        continuation=dict(
            actor_hash=actor.hash(),
            optimizer_sha256=fingerprint(incoming),
            incoming_updates=2000,
            incoming_fit_updates=2000,
            private_rng_sha256=fingerprint(payload["private_rng_state"]),
            pcg_sha256=fingerprint(incoming["rng"]),
            global_rng_sha256=fingerprint(global_rng),
        ),
        global_cpu_sidecar=dict(file="parents/global.npy"),
    )
    operations = []

    def measure(kind, amount, fn, **kwargs):
        operations.append((kind, amount))
        return fn()

    meter = SimpleNamespace(measure=measure)
    return tmp_path, descriptor, actor.normalization, meter, operations, payload


def test_parent_is_decoded_once_and_full_continuation_bound_before_constructor(parent_package):
    import torch

    run, descriptor, normalization, meter, operations, payload = parent_package
    before = fingerprint(torch.get_rng_state())
    actor, incoming, record = worker._load_parent(run, descriptor, normalization, meter)
    assert operations == [("checkpoint_loads", 1), ("policy_constructions", 1)]
    assert record["continuation_verified_before_constructor"]
    assert (
        record["optimizer_sha256"]
        == fingerprint(payload["optimizer_state"])
        == fingerprint(incoming)
    )
    assert actor.hash() == descriptor["actor_hash"]
    assert fingerprint(torch.get_rng_state()) == before
    assert (
        fingerprint(actor._generator.get_state())
        == descriptor["continuation"]["private_rng_sha256"]
    )


@pytest.mark.parametrize("mismatch", ["optimizer", "private_rng", "normalization", "global_rng"])
def test_parent_binding_mismatch_prevents_constructor(parent_package, mismatch):
    run, descriptor, normalization, meter, operations, _ = parent_package
    if mismatch == "normalization":
        normalization["std"][0] = 2.0
    else:
        key = {
            "optimizer": "optimizer_sha256",
            "private_rng": "private_rng_sha256",
            "global_rng": "global_rng_sha256",
        }[mismatch]
        descriptor["continuation"][key] = "f" * 64
    with pytest.raises(ValueError):
        worker._load_parent(run, descriptor, normalization, meter)
    assert operations == [("checkpoint_loads", 1)]


def test_v_meter_enforces_declared_metadata_and_zero_learning_limits(tmp_path):
    from src.gps.mppi_gps.runtime import BudgetExhausted

    meter = worker.VisitationMeter(
        tmp_path / "evidence", wall_seconds=10, max_bytes=1024**3, rss_bytes=2 * 1024**3
    )
    try:
        before = meter.ledger_path.stat().st_size
        with pytest.raises(BudgetExhausted):
            meter.authorize(dict(optimizer_steps=1))
        with pytest.raises(BudgetExhausted):
            meter.record("ordinary", value="x" * 512)
        sequence = meter._sequence
        meter._sequence = 1_200_000
        with pytest.raises(BudgetExhausted, match="journal record ceiling"):
            meter.record("ordinary")
        meter._sequence = sequence
        meter._array_sequence = 300
        with pytest.raises(BudgetExhausted, match="array group ceiling"):
            meter.save_arrays("unpaid", value=np.zeros(1))
        meter._array_sequence = 0
        assert meter.ledger_path.stat().st_size == before
        assert meter.work == {}
    finally:
        meter.close()
    for line in meter.ledger_path.read_bytes().splitlines(keepends=True):
        event = json.loads(line)["event"]
        assert len(line) <= (2048 if event in ("runtime_open", "runtime_close") else 512)


def test_v_module_import_keeps_policy_and_simulator_lazy():
    import subprocess

    code = """import builtins,sys
old=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','mujoco','gymnasium','warp'}:
  raise AssertionError('unexpected import:'+name)
 return old(name,*args,**kwargs)
builtins.__import__=guard
import src.gps.mppi_gps.visitation_native
import src.gps.mppi_gps.visitation_worker
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_preverified_catalog_avoids_rehashing_after_heavy_imports(
    staged_worker, saved_contexts, monkeypatch
):
    run, config, meter, _, _ = staged_worker
    catalog = saved_contexts[2]
    catalog["_verification_accounting"] = dict(read_bytes=123, source_files=7)

    def forbidden_rehash(path):
        raise AssertionError("Input verification repeated after lightweight admission")

    monkeypatch.setitem(
        sys.modules,
        "src.gps.mppi_gps.visitation_inputs",
        SimpleNamespace(verify_prepared=forbidden_rehash),
    )
    result = worker.runV(run, config, meter, verified_catalog=catalog)
    assert result["status"] == "COMPLETED_V_FIXED_ROSTER"
    assert result["prepared_verification_accounting"] == dict(read_bytes=123, source_files=7)
    assert (
        json.loads((run / "worker/static_gate.json").read_text())[
            "prepared_verification_accounting"
        ]
        == result["prepared_verification_accounting"]
    )
