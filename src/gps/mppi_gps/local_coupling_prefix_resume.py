"""Continue the two interrupted L3 alpha-zero native prefixes without replaying them.

This is an engineering continuation adapter, not a new controller. It restores
the last *post-action* simulator boundary and the once-shifted carried MPPI plan.
The immutable prefix remains in its original directory. A derived logical
ledger joins completed prefix reservations to new suffix reservations; the
interrupted forecast remains separately charged as infrastructure uncertainty.
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import MPPIConfig

from .local_coupling_controller import RecordedMPPI, local_outcome
from .local_coupling_numeric_validation import COMPARISON_VERSION, verify_native
from .provenance import fingerprint, write
from .trajectory_pi_accounting import Work
from .trajectory_pi_native import (
    Trace,
    _env,
    _restore,
    binding,
    execute_trace,
    load_trace,
    same_bytes,
)
from .trajectory_pi_student import observations

SCHEMA = "local-coupling-prefix-continuation-1"
PREFIX_CONTROLS = {(1, 0): 198, (2, 1): 151}
INPUT_FIELDS = (
    "prefix_job",
    "prefix_trace",
    "prefix_banks",
    "prefix_ledger",
    "resume_context",
    "reference_R1_trace",
)
JOB_FIELDS = (
    "round",
    "stream",
    "arm",
    "slot",
    "alpha",
    "seed",
    "checkpoint",
    "restarts",
    "controls",
    "planner",
    "task",
    "physics_threads",
    "replay_episode",
    "config",
)
TRACE_FIELDS = (
    "states",
    "warmstarts",
    "integrations",
    "observations",
    "actions",
    "costs",
    "native_fall_cost",
)


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bound(value: dict) -> Path:
    path = Path(value["path"])
    _need(binding(path) == value, "Continuation input changed: " + str(path))
    return path


def _add(counts: dict, values: dict, prefix: str = "") -> None:
    for key, value in values.items():
        _need(type(value) is int and value >= 0, "Nonnegative integer work charges required")
        field = prefix + key
        counts[field] = counts.get(field, 0) + value


def completed_prefix_ledger(path: Path, controls: int) -> tuple[list[dict], dict, list[dict]]:
    """Keep completed calls, and expose the one unmatched interrupted forecast.

    No failed/resolved work is silently removed. This adapter is intentionally
    restricted to prefixes stopped during their next native forecast.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    pending: dict[int, dict] = {}
    counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        _need(row["id"] == index, "Broken original prefix ledger sequence")
        if row["event"] == "requested":
            pending[index] = row
        elif row["event"] == "completed":
            request = pending.pop(row["request_id"], None)
            _need(
                request is not None
                and request["operation"] == row["operation"]
                and request["charges"] == row["charges"],
                "Broken original prefix request resolution",
            )
            _add(counts, row["charges"], "requested_")
            _add(counts, row["charges"], "completed_")
        else:
            raise ValueError("Unexpected original prefix ledger event: " + row["event"])
    _need(len(pending) == 1, "Exactly one unresolved original forecast required")
    request = next(iter(pending.values()))
    _need(
        request["id"] == len(rows) - 1
        and request["operation"] == "native_forecast"
        and request["charges"]
        == dict(model_controls=12288, model_queries=1, physics_substeps=49152),
        "Only the final interrupted H48/K256 forecast may be excluded",
    )
    _need(
        counts.get("completed_native_controls") == controls
        and counts.get("completed_model_queries") == controls + 1
        and counts.get("completed_model_controls") == (controls + 1) * 12288,
        "Completed prefix ledger does not match saved controls/banks",
    )
    excluded = [
        dict(
            request_id=request["id"],
            operation=request["operation"],
            charges=deepcopy(request["charges"]),
            event=deepcopy(request),
            accounting="original_infrastructure_unknown_upper_bound_not_logical_completed_work",
        )
    ]
    return rows[:-1], counts, excluded


def _shift(plan: np.ndarray) -> np.ndarray:
    value = plan.copy()
    value[:-1] = value[1:]
    value[-1] = value[-2].copy()
    return value


def validate_prefix(job: dict) -> dict:
    """Authenticate saved state/plan/RNG/indexing without any physics or NN call."""
    import h5py

    paths = {name: _bound(job[name]) for name in INPUT_FIELDS}
    original_job = json.loads(paths["prefix_job"].read_text())
    _need(
        all(job[name] == original_job[name] for name in JOB_FIELDS),
        "Scientific job metadata changed during prefix continuation",
    )
    controls = PREFIX_CONTROLS.get((job["stream"], job["slot"]))
    _need(
        controls is not None
        and job["round"] == 4
        and job["arm"] == "independent"
        and job["alpha"] == 0
        and job["controls"] == 300
        and job["planner"]["H"] == 48
        and job["planner"]["K"] == 256
        and job["planner"]["initial_iterations"] == 2,
        "Only the two declared interrupted L3 alpha-zero episodes may continue",
    )
    for name in ("checkpoint", "restarts", "config"):
        _bound(job[name])
    original_directory = paths["prefix_job"].parent
    _need(
        Path(original_job["out"]).resolve() == original_directory.resolve()
        and all(
            paths[name].parent.resolve() == original_directory.resolve()
            for name in ("prefix_trace", "prefix_banks", "prefix_ledger")
        )
        and Path(job["out"]).resolve() != original_directory.resolve(),
        "Prefix inputs must belong to the original job; output must be separate",
    )
    arrays = load_trace(paths["prefix_trace"])
    reference = load_trace(paths["reference_R1_trace"])
    _need(set(arrays) == set(TRACE_FIELDS), "Complete original trace fields required")
    for name in TRACE_FIELDS:
        count = controls + 1 if name in TRACE_FIELDS[:4] else controls
        _need(len(arrays[name]) == count and np.isfinite(arrays[name]).all(), "Bad prefix " + name)
        _need(
            name in reference
            and len(reference[name]) == (301 if name in TRACE_FIELDS[:4] else 300)
            and np.allclose(arrays[name], reference[name][:count], rtol=0, atol=1e-10),
            "Original prefix disagrees with fixed-seed R1 reference: " + name,
        )
    _need(
        same_bytes(arrays["observations"], observations(arrays["states"]))
        and np.all(arrays["native_fall_cost"] == 0),
        "Prefix observations/indexing or fall status invalid",
    )
    with np.load(_bound(job["restarts"]), allow_pickle=False) as raw:
        initial = {
            name: raw[f"r{job['slot']}_{name}"].copy()
            for name in ("state", "warmstart", "integration", "spec", "dimensions")
        }
    with np.load(paths["resume_context"], allow_pickle=False) as raw:
        resume = {name: raw[name].copy() for name in raw.files}
    for name, key in (
        ("state", "states"),
        ("warmstart", "warmstarts"),
        ("integration", "integrations"),
    ):
        _need(
            same_bytes(initial[name], arrays[key][0]), "Initial complete context mismatch: " + name
        )
        _need(
            same_bytes(resume[name], arrays[key][-1]), "Resume complete context mismatch: " + name
        )
    for name in ("spec", "dimensions"):
        _need(same_bytes(resume[name], initial[name]), "Resume metadata mismatch: " + name)
    _need(
        int(resume["control"]) == controls
        and int(resume["bank_count"]) == controls + 1
        and not bool(resume["initial_plan"]),
        "Resume control/bank/initial-plan metadata mismatch",
    )
    full_bindings = job["prefix_full_banks"]
    full_paths = [_bound(value) for value in full_bindings]
    _need(
        len({path.name for path in full_paths}) == len(full_paths)
        and all(path.parent.resolve() == original_directory.resolve() for path in full_paths),
        "Original full bank bindings must be unique and local to original job",
    )
    rng = np.random.default_rng(job["seed"])
    plan = np.zeros((48, 6))
    full_names: set[str] = set()
    actor_hash = None
    with h5py.File(paths["prefix_banks"], "r") as raw:
        _need(
            list(raw) == [f"{i:06d}" for i in range(controls + 1)],
            "Incomplete prefix bank sequence",
        )
        for index, group in enumerate(raw.values()):
            control, inner = (0, index) if index < 2 else (index - 1, 0)
            _need(
                int(group.attrs["control"]) == control
                and int(group.attrs["inner"]) == inner
                and float(group.attrs["alpha"]) == 0,
                "Prefix bank control/pass/alpha changed",
            )
            actor_hash = actor_hash or str(group.attrs["actor_hash"])
            _need(str(group.attrs["actor_hash"]) == actor_hash, "Original scoring actor changed")
            _need(
                json.loads(group.attrs["rng_before"]) == rng.bit_generator.state,
                "Prefix RNG-before does not follow original seed and stream",
            )
            # This authenticates random-stream consumption only. It runs no
            # optimizer, forecast, inference, dynamics or policy fitting.
            rng.standard_normal((256, 48, 6))
            _need(
                json.loads(group.attrs["rng_after"]) == rng.bit_generator.state,
                "Prefix RNG-after does not follow original sampling recipe",
            )
            _need(same_bytes(group["U_before"][:], plan), "Prefix carried plan changed")
            for field, key in (
                ("initial_state", "states"),
                ("initial_warmstart", "warmstarts"),
                ("initial_integration", "integrations"),
            ):
                _need(
                    same_bytes(group[field][:], arrays[key][control]),
                    "Prefix bank pre-action context mismatch: " + field,
                )
            after = group["U_after"][:]
            _need(after.shape == (48, 6) and np.isfinite(after).all(), "Invalid saved refined plan")
            if index != 0:
                _need(
                    same_bytes(after[0], arrays["actions"][control]),
                    "Prefix executed action mismatch",
                )
            plan = after.copy() if index == 0 else _shift(after)
            if "full_bank" in group.attrs:
                full_names.add(str(group.attrs["full_bank"]))
        _need(
            full_names == {path.name for path in full_paths}, "Missing or extra frozen full banks"
        )
    _need(
        same_bytes(plan, resume["U_carried_after_one_shift"]),
        "Resume carried plan must shift the final completed refinement exactly once",
    )
    if "resume_rng_state" in job:
        _need(job["resume_rng_state"] == rng.bit_generator.state, "Resume RNG state changed")
    old_rows, counts, excluded = completed_prefix_ledger(paths["prefix_ledger"], controls)
    first_bank = next(path for path in full_paths if path.name == "full_t000_p0.npz")
    with np.load(first_bank, allow_pickle=False) as raw:
        native_check = dict(
            context=initial,
            actions=raw["actions"][[0, 255]].copy(),
            posts=raw["posts"][[0, 255]].copy(),
            C=raw["C"][[0, 255]].copy(),
            D=raw["D"][[0, 255]].copy(),
        )
    return dict(
        paths=paths,
        arrays=arrays,
        reference=reference,
        initial=initial,
        resume={name: resume[name] for name in initial},
        planner_state=dict(U=plan, rng_state=deepcopy(rng.bit_generator.state), initial_plan=False),
        controls=controls,
        banks=controls + 1,
        actor_hash=actor_hash,
        full_paths=full_paths,
        native_check=native_check,
        prefix_rows=old_rows,
        prefix_counts=counts,
        excluded=excluded,
    )


def _copy_h5(source: Path, target: Any) -> None:
    import h5py

    with h5py.File(source, "r") as original:
        for name in original:
            original.copy(name, target)
        for name, value in original.attrs.items():
            target.attrs[name] = value
        _check_h5_prefix(original, target)
    target.flush()


def _check_h5_prefix(original: Any, target: Any) -> None:
    """Compare all copied arrays and attributes before the first suffix call."""
    import h5py

    _need(set(original) <= set(target), "Copied HDF5 prefix is missing objects")
    _need(set(original.attrs) == set(target.attrs), "Copied HDF5 attributes changed")
    for name, value in original.attrs.items():
        _need(
            same_bytes(np.asarray(value), np.asarray(target.attrs[name])),
            "Copied HDF5 attribute changed",
        )
    for name, value in original.items():
        if isinstance(value, h5py.Group):
            _check_h5_prefix(value, target[name])
        else:
            _need(
                same_bytes(value[:], target[name][: len(value)]),
                "Copied HDF5 dataset changed: " + name,
            )
            _need(
                set(value.attrs) == set(target[name].attrs),
                "Copied HDF5 dataset attributes changed",
            )
            for key, attribute in value.attrs.items():
                _need(
                    same_bytes(np.asarray(attribute), np.asarray(target[name].attrs[key])),
                    "Copied HDF5 dataset attribute value changed",
                )


def _load_actor(job: dict, work: Work) -> Any:
    from .policy import Policy

    actor = work.call(
        "load_actor",
        dict(checkpoint_loads=1, policy_constructions=1),
        lambda: Policy.from_checkpoint(job["checkpoint"]["path"], device="cpu")[0],
    )
    actor.eval()
    return actor


def _merge_ledger(directory: Path, prefix: dict | None, suffix: dict) -> dict:
    """Make a transparent derived episode receipt; preserve raw suffix bytes."""
    (directory / "receipt.json").rename(directory / "suffix_receipt.json")
    rows = [] if prefix is None else prefix["prefix_rows"]
    raw_suffix = [
        json.loads(line) for line in (directory / "suffix_ledger.jsonl").read_text().splitlines()
    ]
    combined = []
    for source, source_rows in (("immutable_prefix", rows), ("new_suffix", raw_suffix[:-1])):
        mapping: dict[int, int] = {}
        for row in source_rows:
            new = deepcopy(row)
            mapping[row["id"]] = len(combined)
            new["id"] = len(combined)
            new["continuation_source"] = source
            new["source_id"] = row["id"]
            if "request_id" in new:
                new["request_id"] = mapping[row["request_id"]]
            combined.append(new)
    counts = {} if prefix is None else dict(prefix["prefix_counts"])
    _add(counts, suffix["counts"])
    receipt = dict(
        suffix,
        counts=counts,
        schema=SCHEMA,
        timer_scope="new_suffix_only_original_prefix_CPU_in_original_global_worker_accounting",
        prefix_cpu_seconds=None,
        prefix_wall_seconds=None,
        suffix_ledger=binding(directory / "suffix_ledger.jsonl"),
        suffix_receipt=binding(directory / "suffix_receipt.json"),
        prefix_completed_counts={} if prefix is None else prefix["prefix_counts"],
        excluded_pending_requests=[] if prefix is None else prefix["excluded"],
    )
    combined.append(
        dict(id=len(combined), event="closed", wall_seconds=suffix["wall_seconds"], receipt=receipt)
    )
    with (directory / "ledger.jsonl").open("x") as stream:
        for row in combined:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write(directory / "receipt.json", receipt)
    return receipt


def resume_episode(job: dict) -> dict:
    """Resume only the unpaid native suffix and prescribed final verifications."""
    directory = Path(job["out"])
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "job.json", job)
    work = Work(directory, job["limits"], filename="suffix_ledger.jsonl")
    env = planner = trace = prefix = None
    result = dict(
        schema="local-coupling-episode-1",
        status="FAILED_STOPPED",
        **{
            name: job[name]
            for name in ("round", "stream", "arm", "slot", "alpha", "seed", "checkpoint")
        },
        comparison_version=COMPARISON_VERSION,
    )
    try:
        prefix = validate_prefix(job)
        work.check()
        actor = _load_actor(job, work)
        _need(actor.hash() == prefix["actor_hash"], "Continuation scoring actor hash changed")
        env = _env(dict(task=job["task"], physics_threads=job["physics_threads"]), work)
        _restore(env, prefix["resume"], work)
        planner = RecordedMPPI(
            env,
            MPPIConfig(**job["planner"]),
            seed=job["seed"],
            actor=actor,
            alpha=job["alpha"],
            work=work,
            directory=directory,
        )
        _copy_h5(prefix["paths"]["prefix_banks"], planner.stats)
        for source in prefix["full_paths"]:
            target = directory / source.name
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            _need(
                binding(target)["sha256"] == binding(source)["sha256"], "Copied full bank changed"
            )
        planner.bank_count = prefix["banks"]
        planner.restore(prefix["planner_state"])
        planner.native_checks = [
            dict(prefix["native_check"], source=binding(directory / "full_t000_p0.npz"))
        ]
        trace = Trace(directory / "trace.h5")
        _copy_h5(prefix["paths"]["prefix_trace"], trace.file)
        trace.flush()
        states = list(prefix["arrays"]["states"].copy())
        costs = list(prefix["arrays"]["costs"].copy())
        fell = False
        for control in range(prefix["controls"], job["controls"]):
            planner.control, planner.inner = control, 0
            state = env.get_state()
            action, _ = planner.plan_step(
                state,
                prior_cost=lambda p, u: planner.penalty(state, p, u),
                initial_warmstart=env.get_warmstart(),
            )
            _, cost, done, info = env.step(action.astype(np.float64))
            _need(
                np.isfinite(env.get_state()).all() and np.isfinite(cost),
                "Invalid executed native suffix physics",
            )
            trace.append(actions=action, costs=cost, native_fall_cost=info["fall_cost"])
            trace.boundary(env)
            trace.flush()
            states.append(env.get_state())
            costs.append(cost)
            if done:
                fell = True
                break
        trace.close()
        trace = None
        result.update(
            outcome=local_outcome(np.asarray(states), np.asarray(costs), job["controls"], fell),
            actor_hash=actor.hash(),
            initial_context_fingerprint=fingerprint(prefix["initial"]),
            banks=planner.bank_count,
        )
        _need(actor.hash() == planner.actor_hash == prefix["actor_hash"], "Scoring actor changed")
        planner.stats.close()
        raw = load_trace(directory / "trace.h5")
        for name in TRACE_FIELDS:
            _need(
                raw[name].shape == prefix["reference"][name].shape
                and np.allclose(raw[name], prefix["reference"][name], rtol=0, atol=1e-10),
                "Resumed full episode differs from fixed-seed R1 reference: " + name,
            )
        result["reference_R1_identity"] = dict(
            rtol=0, atol=1e-10, fields=list(TRACE_FIELDS), status="PASS"
        )
        result["native_candidate_checks"] = verify_native(
            env, work, directory, planner.native_checks, actor
        )
        if job.get("replay_episode"):
            _restore(env, prefix["initial"], work)
            replay_path = directory / "episode_replay.h5"
            execute_trace(env, work, replay_path, job["controls"], actions=raw["actions"])
            replay = load_trace(replay_path)
            for name in TRACE_FIELDS:
                _need(
                    raw[name].shape == replay[name].shape
                    and np.allclose(raw[name], replay[name], rtol=0, atol=1e-10),
                    "Complete resumed native episode replay mismatch: " + name,
                )
            result["native_episode_replay"] = binding(replay_path)
        # Bindings are checked again, with no old files opened for writing.
        for name in INPUT_FIELDS:
            _bound(job[name])
        for value in job["prefix_full_banks"]:
            _bound(value)
        result.update(
            trace=binding(directory / "trace.h5"),
            bank_stats=binding(directory / "banks.h5"),
            status="COMPLETED",
        )
    except BaseException as error:
        result["error"] = repr(error)
    finally:
        if trace is not None:
            trace.close()
        if planner is not None and planner.stats.id.valid:
            planner.stats.close()
        if env is not None:
            env.close()
    suffix = work.close("COMPLETED" if result["status"] == "COMPLETED" else "FAILED")
    receipt = _merge_ledger(directory, prefix, suffix)
    result["work"] = receipt["counts"]
    result["continuation"] = dict(
        schema=SCHEMA,
        **{name: job[name] for name in INPUT_FIELDS},
        prefix_full_banks=job["prefix_full_banks"],
        prefix_controls=None if prefix is None else prefix["controls"],
        prefix_banks_count=None if prefix is None else prefix["banks"],
        prefix_completed_counts={} if prefix is None else prefix["prefix_counts"],
        excluded_pending_requests=[] if prefix is None else prefix["excluded"],
        suffix_ledger=receipt["suffix_ledger"],
        suffix_receipt=receipt["suffix_receipt"],
        acquisition_replayed_prefix_controls=0,
        native_suffix_controls=len(costs) - prefix["controls"]
        if prefix is not None and "outcome" in result
        else None,
        prescribed_episode_replay_controls=300 if "native_episode_replay" in result else 0,
        timer_scope=receipt["timer_scope"],
        resume_rng_state=None if prefix is None else prefix["planner_state"]["rng_state"],
        carried_plan_fingerprint=None
        if prefix is None
        else fingerprint(prefix["planner_state"]["U"]),
    )
    write(directory / "manifest.json", result)
    return dict(
        path=str(directory),
        result=result,
        receipt=receipt,
        raw_suffix_receipt=receipt["suffix_receipt"],
        raw_suffix_ledger=receipt["suffix_ledger"],
        prefix_completed_counts=receipt["prefix_completed_counts"],
        excluded_pending_requests=receipt["excluded_pending_requests"],
    )
