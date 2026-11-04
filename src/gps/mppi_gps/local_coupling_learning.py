"""Bound paired corpora, exact learning continuation and Torch-only execution.

The acquisition/cohort owner supplies frozen jobs and reserves their budgets.
This module reuses the existing fitter, stream envelope and native recorder;
importing it constructs neither a policy nor a simulator.
"""

from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import fingerprint, sha, write
from .trajectory_pi_accounting import Work
from .trajectory_pi_student import SETTINGS, observations


def binding(path: Path) -> dict:
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bound(spec: dict) -> Path:
    path = Path(spec["path"])
    _need(path.is_absolute() and path.is_file(), "Explicit available file binding required")
    _need(binding(path) == spec, "Bound input changed: " + str(path))
    return path


def _config(spec: dict) -> dict:
    value = json.loads(_bound(spec).read_text())
    _need(
        value.get("schema") in ("local-coupling-learning-l2-1", "local-coupling-learning-l3-1")
        and value.get("launch_ready") is True,
        "Frozen launch-ready local coupling configuration required",
    )
    return value


def _torch(config: dict) -> Any:
    import torch

    learner = config["learning"]
    _need(
        all(learner.get(key) == value for key, value in SETTINGS.items()),
        "Existing W2 fit settings required",
    )
    _need(
        learner["gradient_clip"] == 0.5
        and learner["device"] == "cpu"
        and learner["torch_threads"] == 1,
        "Fixed single-threaded CPU learner required",
    )
    _need(not torch.cuda.is_initialized(), "No GPU execution permitted")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def _arrays(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as source:
            return {key: source[key].copy() for key in source.files}
    import h5py

    with h5py.File(path, "r") as source:
        return {key: source[key][:] for key in source}


def _trace(spec: dict, maximum: int) -> dict:
    raw = _arrays(_bound(spec))
    n = len(raw["actions"])
    shapes = dict(
        states=(n + 1, 19),
        warmstarts=(n + 1, 9),
        integrations=(n + 1, 91),
        actions=(n, 6),
        costs=(n,),
    )
    _need(
        n <= maximum
        and all(
            raw[key].shape == shape and raw[key].dtype == np.float64 and np.isfinite(raw[key]).all()
            for key, shape in shapes.items()
        ),
        "Finite native N+1 context/control trace required",
    )
    _need(
        raw["observations"].shape in ((n, 17), (n + 1, 17))
        and raw["observations"].dtype == np.float64
        and np.array_equal(
            raw["observations"], observations(raw["states"])[: len(raw["observations"])]
        ),
        "Native pre-action observations differ",
    )
    _need(np.all(np.abs(raw["actions"]) <= 1), "Executed physical torques must be bounded")
    return raw


def build_paired_corpora(
    a_traces: list[dict],
    b_traces: list[dict],
    out: Path,
    *,
    config_binding: dict,
    max_controls: int = 300,
) -> dict:
    """Retain every common executed (slot,step), including failed prefixes."""
    _config(config_binding)
    _need(0 < max_controls <= 300, "Bounded local acquisition horizon required")
    pairs = []
    for roster in (a_traces, b_traces):
        indexed = {row["slot"]: row for row in roster}
        _need(
            len(roster) == len(indexed) == 8 and set(indexed) == set(range(8)),
            "All eight unique restart slots required",
        )
        pairs.append(indexed)
    inputs = {arm: dict(states=[], observations=[], actions=[]) for arm in ("a", "b")}
    slots, steps, cases = [], [], []
    for slot in range(8):
        raw = [_trace(roster[slot]["trace"], max_controls) for roster in pairs]
        for key in ("states", "warmstarts", "integrations"):
            _need(
                raw[0][key][0].tobytes() == raw[1][key][0].tobytes(),
                "Paired acquisition initial contexts differ",
            )
        n = min(len(value["actions"]) for value in raw)
        _need(n > 0, "Every restart must contribute a common executed prefix")
        cases.append(
            dict(
                slot=slot,
                controls=[len(value["actions"]) for value in raw],
                common=n,
                traces=[roster[slot]["trace"] for roster in pairs],
            )
        )
        slots.extend([slot] * n)
        steps.extend(range(n))
        for arm, value in zip(("a", "b"), raw, strict=True):
            for key in inputs[arm]:
                inputs[arm][key].append(value[key][:n])
    _need(bool(slots), "No common executed labels; no fit permitted")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    row_ids = np.column_stack((slots, steps)).astype(np.int64)
    files = {}
    for arm in inputs:
        arrays = {key: np.concatenate(value) for key, value in inputs[arm].items()}
        with (out / f"{arm}.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays, row_ids=row_ids)
        files[arm] = binding(out / f"{arm}.npz")
    result = dict(
        schema="local-coupling-corpora-1",
        config=config_binding,
        rows=len(slots),
        row_ids_fingerprint=fingerprint(row_ids),
        cases=cases,
        files=files,
        selection="All common executed prefixes; no survivor, MSE or spectral filter",
    )
    write(out / "manifest.json", result)
    return result


def _finish(out: Path, work: Work, result: dict) -> dict:
    try:
        receipt = work.close("COMPLETED" if result["status"] == "COMPLETED" else "FAILED")
    except BaseException as error:
        result.update(status="FAILED_STOPPED", accounting_error=repr(error))
        receipt = work.close("FAILED") if not work.closed else work.receipt()
    result["receipt"] = binding(out / "receipt.json")
    write(out / "result.json", result)
    return dict(path=str(out.resolve()), result=result, receipt=receipt)


def fork_streams(incoming_dir: Path, out: Path, *, config_binding: dict, limits: dict) -> dict:
    """One deterministic PCG jump roster; actor/Adam/private Torch/globals unchanged."""
    config = _config(config_binding)
    _torch(config)
    from .trajectory_pi_student import global_rng_state, load_stream, save_stream

    incoming_dir, out = Path(incoming_dir), Path(out)
    expected = config["inputs"]["checkpoint"]
    _need(
        all(
            binding(incoming_dir / "policy.pt")[key] == expected[key] for key in ("bytes", "sha256")
        ),
        "Incoming W2 package differs",
    )
    incoming = binding(incoming_dir / "stream.json")
    out.mkdir(parents=True, exist_ok=False)
    work = Work(out, limits)
    result = dict(
        schema="local-coupling-forks-1",
        status="RUNNING",
        config=config_binding,
        incoming=incoming,
        streams=[],
    )
    try:
        actor, state, _ = work.call(
            "incoming_stream",
            dict(checkpoint_loads=2, policy_constructions=1),
            lambda: load_stream(incoming_dir),
        )
        _need(state is not None, "Complete incoming Adam/private PCG required")
        before = dict(
            actor=actor.hash(),
            adam=fingerprint(state["adam"]),
            private_torch=fingerprint(actor._generator.get_state()),
            global_rng=fingerprint(global_rng_state()),
            normalization=fingerprint(actor.normalization),
        )
        for index in range(3):
            continued = deepcopy(state)
            generator = np.random.PCG64(0)
            generator.state = deepcopy(state["rng"])
            continued["rng"] = generator.jumped(index + 1).state
            directory = out / f"stream{index}"
            metadata = dict(
                study="L2.0",
                config_sha256=config_binding["sha256"],
                stream=index,
                pcg_jump=index + 1,
                incoming=incoming,
                incoming_rng_fingerprint=fingerprint(state["rng"]),
                global_envelope="Inherited exact package; no new global seed",
            )
            saved = work.call(
                "fork_stream",
                dict(checkpoint_writes=2),
                lambda: save_stream(directory, actor, continued, metadata),
            )
            # The new fork is not published until this function closes. Keep the
            # exact inherited envelope package, including its original encoding.
            source, target = incoming_dir / "global_rng.pt", directory / "global_rng.pt"
            work.call(
                "copy_global_envelope",
                dict(checkpoint_writes=1),
                lambda: shutil.copyfile(source, target),
            )
            saved["files"]["global_rng.pt"] = {
                key: binding(target)[key] for key in ("bytes", "sha256")
            }
            (directory / "stream.json").write_text(json.dumps(saved, indent=2) + "\n")
            _need(
                target.read_bytes() == source.read_bytes()
                and fingerprint(global_rng_state()) == before["global_rng"],
                "Fork changed inherited global envelope",
            )
            work.record(
                "pcg_fork_created",
                stream=index,
                jump=index + 1,
                rng_fingerprint=fingerprint(continued["rng"]),
            )
            result["streams"].append(
                dict(
                    stream=index,
                    directory=str(directory.resolve()),
                    envelope=binding(directory / "stream.json"),
                    optimizer_fingerprint=fingerprint(continued),
                    rng_fingerprint=fingerprint(continued["rng"]),
                )
            )
        _need(
            actor.hash() == before["actor"]
            and fingerprint(state["adam"]) == before["adam"]
            and fingerprint(actor._generator.get_state()) == before["private_torch"],
            "Fork mutated actor/Adam/private Torch state",
        )
        result.update(status="COMPLETED", inherited=before, new_global_seeds=0, pcg_forks=3)
    except BaseException as error:
        result.update(status="FAILED_STOPPED", error=repr(error))
    return _finish(out, work, result)


def _corpus(spec: dict) -> dict:
    raw = _arrays(_bound(spec))
    x, y = raw["observations"], raw["actions"]
    _need(
        x.ndim == 2
        and x.shape[1] == 17
        and y.shape == (len(x), 6)
        and len(x) > 0
        and x.dtype == y.dtype == np.float64
        and np.isfinite(x).all()
        and np.isfinite(y).all()
        and np.all(np.abs(y) <= 1),
        "Finite physical observation/target corpus required",
    )
    if "states" in raw:
        _need(
            np.array_equal(x, observations(raw["states"])), "Corpus pre-action state join differs"
        )
    return raw


def _diagnostics(actor: Any, roster: dict, out: Path, work: Work) -> dict:
    result = {}
    for name, spec in roster.items():
        _need(name.isidentifier(), "Simple diagnostic name required")
        raw = _corpus(spec)
        predictions = np.concatenate(
            [
                actor.predict(raw["observations"][start : start + 1024], meter=work)
                for start in range(0, len(raw["actions"]), 1024)
            ]
        )
        residual = predictions.astype(np.float64) - raw["actions"]
        path = out / (name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, predictions=predictions)
        result[name] = dict(
            data=spec,
            predictions=binding(path),
            rows=len(predictions),
            action_mse=float(np.mean(residual**2)),
            rms_l2=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
            torch_prediction_batch_size=1024,
        )
    return result


def incoming_lineage_matches(config: dict, job: dict, metadata: dict, state: dict) -> bool:
    """Validate the frozen owner without constructing another policy or optimizer."""
    multiround = config["schema"] == "local-coupling-learning-l3-1"
    if multiround and job["round"] == 1:
        original = next(
            row
            for row in config["inputs"]["initial_streams"]
            if (row["stream"], row["arm"]) == (job["stream"], job["arm"])
        )
        return (
            job["incoming"]["sha256"] == original["files"]["stream.json"]["sha256"]
            and metadata.get("stream") == job["stream"]
            and metadata.get("arm") == job["arm"]
            and state["updates"] == config["initial_cumulative_updates"]
        )
    if multiround:
        return (
            metadata.get("stream") == job["stream"]
            and metadata.get("arm") == job["arm"]
            and metadata.get("config_sha256") == job["config"]["sha256"]
            and metadata.get("round") == job["round"] - 1
            and metadata.get("fit_updates") == config["learning"]["updates"]
            and state["updates"]
            == config["initial_cumulative_updates"]
            + config["learning"]["updates"] * (job["round"] - 1)
        )
    return (
        metadata.get("stream") == job["stream"]
        and metadata.get("config_sha256") == job["config"]["sha256"]
        and (
            metadata.get("pcg_jump") == job["stream"] + 1
            if job["round"] == 1
            else metadata.get("round") == 1
            and metadata.get("arm") == job["arm"]
            and metadata.get("fit_updates") == config["learning"]["updates"]
        )
    )


def fit_job(job: dict) -> dict:
    """One fit block; round two loads its own child stream without another fork."""
    config = _config(job["config"])
    torch = _torch(config)
    from .trajectory_pi_student import fit_round, global_rng_state, load_stream, save_stream

    incoming_path, raw = _bound(job["incoming"]), _corpus(job["corpus"])
    _need(incoming_path.name == "stream.json", "Bound complete stream envelope required")
    updates = config["learning"]["updates"]
    multiround = config["schema"] == "local-coupling-learning-l3-1"
    rounds = config["rounds"] if multiround else 2
    _need(
        type(updates) is int
        and 0 < updates <= 2000
        and type(job["round"]) is int
        and 1 <= job["round"] <= rounds,
        "Frozen round/update count required",
    )
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "job.json", job)
    work = Work(out, job["limits"])
    result = dict(
        schema="local-coupling-fit-1",
        status="RUNNING",
        config=job["config"],
        incoming=job["incoming"],
        corpus=job["corpus"],
        round=job["round"],
        stream=job["stream"],
        arm=job["arm"],
    )
    indices, cumulative, block_steps = [], [], []
    rng_file = None
    try:
        actor, state, metadata = work.call(
            "incoming_stream",
            dict(checkpoint_loads=2, policy_constructions=1),
            lambda: load_stream(incoming_path.parent),
        )
        _need(
            incoming_lineage_matches(config, job, metadata, state),
            "Incoming stream is not the frozen same-arm continuation",
        )
        before = dict(
            policy_hash=actor.hash(),
            optimizer_fingerprint=fingerprint(state),
            normalization=fingerprint(actor.normalization),
            private_torch=fingerprint(actor._generator.get_state()),
            global_rng=fingerprint(global_rng_state()),
        )
        result["before"] = before
        result["diagnostics_before"] = _diagnostics(
            actor, job.get("diagnostics", {}), out / "before", work
        )
        rng_file = (out / "index_rng.jsonl").open("x")

        def initial(value: dict) -> None:
            def save() -> None:
                with (out / "initial_state.pt").open("xb") as stream:
                    torch.save(value, stream)
                    stream.flush()
                    os.fsync(stream.fileno())

            work.call("actual_fit_initial_state", dict(checkpoint_writes=1), save)

        def record(value: dict) -> None:
            indices.append(value["indices"].copy())
            cumulative.append(value["update"])
            block_steps.append(value["fit_update"])
            row = {key: value[key] for key in ("update", "fit_update", "rng_before", "rng_after")}
            row["indices_fingerprint"] = fingerprint(value["indices"])
            rng_file.write(json.dumps(row, sort_keys=True) + "\n")
            rng_file.flush()
            os.fsync(rng_file.fileno())

        child, continued, evidence = work.call(
            "fit_constructors",
            dict(policy_constructions=1, optimizer_constructions=1),
            lambda: fit_round(
                actor,
                raw["observations"],
                raw["actions"],
                updates=updates,
                seed=0,
                optimizer_state=state,
                resume_fit=False,
                meter=work,
                initial_observer=initial,
                index_observer=record,
            ),
        )
        write(out / "fit.json", evidence)
        saved = work.call(
            "save_continuation",
            dict(checkpoint_writes=2),
            lambda: save_stream(
                out / "stream",
                child,
                continued,
                dict(
                    round=job["round"],
                    stream=job["stream"],
                    arm=job["arm"],
                    config_sha256=job["config"]["sha256"],
                    fit_updates=updates,
                    incoming=job["incoming"],
                ),
            ),
        )
        loaded, loaded_state, _ = work.call(
            "reload_continuation",
            dict(checkpoint_loads=2, policy_constructions=1),
            lambda: load_stream(out / "stream"),
        )
        _need(
            loaded.hash() == child.hash()
            and fingerprint(loaded_state) == fingerprint(continued)
            and fingerprint(global_rng_state()) == saved["global_rng_fingerprint"]
            and fingerprint(child.normalization) == before["normalization"],
            "Saved/reloaded continuation or normalization differs",
        )
        result.update(
            status="COMPLETED",
            updates=updates,
            cumulative_updates=continued["updates"],
            after=dict(
                policy_hash=child.hash(),
                optimizer_fingerprint=fingerprint(continued),
                normalization=fingerprint(child.normalization),
                private_torch=fingerprint(child._generator.get_state()),
                global_rng=saved["global_rng_fingerprint"],
            ),
            outgoing=binding(out / "stream/stream.json"),
            initial_state=binding(out / "initial_state.pt"),
            fit=binding(out / "fit.json"),
            diagnostics_after=_diagnostics(loaded, job.get("diagnostics", {}), out / "after", work),
        )
    except BaseException as error:
        result.update(status="FAILED_STOPPED", error=repr(error))
        if hasattr(error, "fit_evidence"):
            write(out / "failed_fit.json", error.fit_evidence)
    finally:
        if rng_file is not None:
            rng_file.close()
        with (out / "index_tape.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                indices=np.asarray(indices, dtype=np.int64).reshape(
                    -1, config["learning"]["batch_size"]
                ),
                updates=np.asarray(cumulative, dtype=np.int64),
                fit_updates=np.asarray(block_steps, dtype=np.int64),
            )
        result["index_tape"] = binding(out / "index_tape.npz")
        if (out / "index_rng.jsonl").is_file():
            result["index_rng"] = binding(out / "index_rng.jsonl")
    return _finish(out, work, result)


def evaluate_job(job: dict) -> dict:
    """Fresh-reset standalone Torch Policy.predict, without any planner/fallback."""
    config = _config(job["config"])
    _torch(config)
    checkpoint = _bound(job["checkpoint"])
    _need(
        type(config["evaluation"]["horizon"]) is int
        and 0 < config["evaluation"]["horizon"] <= 1000
        and type(job["reset_seed"]) is int
        and 0 <= job["reset_seed"] < 2**32,
        "Bounded horizon and uint32 reset required",
    )
    from .policy import Policy
    from .runtime import snapshot
    from .trajectory_pi_native import _env, execute_trace

    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "job.json", job)
    work, env = Work(out, job["limits"]), None
    result = dict(
        schema="local-coupling-evaluation-1",
        status="RUNNING",
        config=job["config"],
        checkpoint=job["checkpoint"],
        reset_seed=job["reset_seed"],
        controller="Torch Policy.predict; direct bounded torques",
        role=job.get("role"),
    )
    try:
        actor, _, _ = work.call(
            "torch_checkpoint",
            dict(checkpoint_loads=1, policy_constructions=1),
            lambda: Policy.from_checkpoint(checkpoint, device="cpu"),
        )
        _need(actor.obs_dim == 17 and actor.hidden_dim == 128, "Existing W2 architecture required")
        env = _env(dict(task=job["task"], physics_threads=job.get("physics_threads", 1)), work)
        before = np.random.get_state()
        try:
            np.random.seed(job["reset_seed"])
            work.call("paired_reset", dict(resets=1, external_mj_forward_calls=2), env.reset)
        finally:
            np.random.set_state(before)
        initial = snapshot(env)
        outcome = execute_trace(
            env, work, out / "trace.h5", config["evaluation"]["horizon"], actor=actor
        )
        result.update(
            status="COMPLETED",
            policy_hash=actor.hash(),
            initial_context_fingerprint=fingerprint(
                {key: initial[key] for key in ("state", "warmstart", "integration")}
            ),
            outcome=outcome,
            trace=binding(out / "trace.h5"),
        )
    except BaseException as error:
        result.update(status="FAILED_STOPPED", error=repr(error))
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                result.update(status="FAILED_STOPPED", cleanup_error=repr(error))
    return _finish(out, work, result)
