"""Fail-closed prerequisites for the frozen full-trajectory PI comparison.

Importing this module performs no simulation, checkpoint decoding or fitting.
Only ``run_preflight`` executes the separately budgeted, create-only P0 assay.
Its synthetic continuation check never fits research observations or labels.
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import fingerprint, sha, write
from .trajectory_pi_accounting import Work

CHECK_NAMES = (
    "full_context_replay",
    "pre_action_index",
    "full_sequence_no_shift",
    "score_formula",
    "checkpoint_continuation",
    "fall_accounting",
    "budget_enforcement",
    "zero_residual_reproduction",
)
PROTOTYPE = "archive/prototype_sources/full_pi_2026_11_03.py"
PROTOTYPE_SHA256 = "350cdce35c43585eb7edcda11bc2fd9a65c542b0b336783362715703929ffea2"
ATOL = 1e-10


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _roster(root: Path, rows: list[dict]) -> dict[str, Path]:
    """Verify canonical relative file bindings without following escaped paths."""
    _need(isinstance(rows, list) and bool(rows), "Nonempty immutable file list required")
    paths = {}
    root = root.resolve()
    for row in rows:
        name = row["path"]
        _need(isinstance(name, str) and bool(name), "Relative file name required")
        relative = Path(name)
        path = (root / relative).resolve()
        _need(
            not relative.is_absolute()
            and not {".", ".."} & set(relative.parts)
            and str(relative) == name
            and path.is_relative_to(root)
            and name not in paths,
            "Unsafe or repeated immutable file name",
        )
        _need(
            path.is_file()
            and type(row["bytes"]) is int
            and path.stat().st_size == row["bytes"]
            and sha(path) == row["sha256"],
            "Immutable file binding differs: " + name,
        )
        paths[name] = path
    _need(list(paths) == sorted(paths), "Canonical sorted file list required")
    return paths


def _validate_bindings(
    config_path: Path, prepared: Path, source_manifest: Path
) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text())
    _need(
        config.get("schema") == "trajectory-pi-p1-1" and config.get("launch_ready") is True,
        "Committed launch-ready configuration required",
    )
    source = json.loads(source_manifest.read_text())
    _need(source.get("schema") == "trajectory-pi-source-1", "Immutable source list required")
    encoded = json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()
    _need(
        hashlib.sha256(encoded).hexdigest() == source["roster_sha256"], "Source roster seal differs"
    )
    root = Path(__file__).resolve().parents[3]
    paths = _roster(root, source["files"])
    own = "src/gps/mppi_gps/trajectory_pi_preflight.py"
    _need(paths.get(own) == Path(__file__).resolve(), "Execute the bound frozen P0 source")
    _need(
        "configs/studies/trajectory_pi_p1.json" in paths
        and sha(paths["configs/studies/trajectory_pi_p1.json"]) == sha(config_path),
        "Source and scientific configuration differ",
    )
    _need(
        PROTOTYPE in paths and sha(paths[PROTOTYPE]) == PROTOTYPE_SHA256,
        "Original optimizer archive differs",
    )
    manifest = json.loads((prepared / "manifest.json").read_text())
    _need(
        manifest.get("schema") == "trajectory-pi-prepared-1"
        and manifest.get("config_sha256") == sha(config_path)
        and manifest.get("normalization_inherited") is True
        and manifest.get("parent_refits") == 0,
        "Prepared inputs do not bind the unchanged incoming parents",
    )
    files = _roster(prepared, manifest["files"])
    _need(
        {row["stream"] for row in manifest["parents"]} == set(config["streams"])
        and {row["slot"] for row in manifest["teachers"]} == {0, 1},
        "Prepared roster differs",
    )
    for spec in config["parents"]:
        checkpoint = files[spec["prepared_directory"] + "/policy.pt"]
        _need(sha(checkpoint) == spec["checkpoint"]["sha256"], "Incoming parent bytes differ")
    for spec in config["teachers"]:
        _need(spec["prepared_path"] in files, "Missing complete initial teacher")
    return config, manifest


def _runtime(config: dict) -> dict:
    import h5py
    import mujoco
    import scipy
    import torch

    actual = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        mujoco=mujoco.__version__,
        torch=str(torch.__version__),
        h5py=h5py.__version__,
    )
    expected = config["resources"]["runtime"]
    _need(
        all(value == expected[name] for name, value in actual.items()),
        "Actual imported runtime differs from frozen CPU pins: " + repr(actual),
    )
    _need(not torch.cuda.is_initialized(), "P0 must not use an initialized CUDA runtime")
    torch.set_num_threads(1)
    _need(torch.get_num_threads() == 1, "Single-threaded inherited CPU fitting required")
    return actual


def _limits(config: dict | None = None) -> dict[str, int]:
    p = {} if config is None else config["prerequisites"]
    return dict(
        wall_seconds=min(int(p.get("wall_seconds", 600)), 600),
        output_bytes=min(int(p.get("output_bytes", 512 * 1024**2)), 512 * 1024**2),
        memory_bytes=8 * 1024**3,
        model_controls=min(int(p.get("model_controls", 2_000_000)), 2_000_000),
        native_controls=min(int(p.get("native_controls", 2500)), 2500),
        physics_substeps=4 * (2_000_000 + 2500),
        neural_rows=2_000_000,
        backward_rows=min(int(p.get("synthetic_backward_rows", 2048)), 2048),
        optimizer_steps=min(int(p.get("synthetic_optimizer_steps", 8)), 8),
        environment_constructions=1,
        restores=16,
        external_mj_forward_calls=16,
        checkpoint_loads=3,
        policy_constructions=5,
        optimizer_constructions=3,
        numpy_constructions=2,
        stream_writes=3,
        model_queries=5,
        pi_passes=4,
        replay_optimizer_passes=2,
        replay_sampling_controls=512_000,
    )


def _comparison(actual: Any, expected: Any, name: str, *, exact: bool = False) -> dict:
    from .trajectory_pi_native import same_bytes

    actual, expected = np.asarray(actual), np.asarray(expected)
    same_shape = actual.shape == expected.shape
    passed = (
        same_bytes(actual, expected)
        if exact
        else (
            same_shape
            and np.isfinite(actual).all()
            and np.isfinite(expected).all()
            and np.allclose(actual, expected, rtol=0, atol=ATOL)
        )
    )
    result = dict(
        name=name,
        passed=bool(passed),
        shape=list(actual.shape),
        exact_bytes=exact,
        rtol=0,
        atol=0 if exact else ATOL,
        maximum_absolute_error=float(np.max(np.abs(actual - expected)))
        if same_shape and actual.size and np.isfinite(actual).all() and np.isfinite(expected).all()
        else None,
    )
    _need(passed, "Frozen P0 comparison failed: " + name + " " + repr(result))
    return result


def _replays(config: dict, prepared: Path, out: Path, env: Any, work: Work) -> tuple[dict, list]:
    from .trajectory_pi_native import _restore, execute_trace, load_context, load_trace
    from .trajectory_pi_student import observations

    checks = []
    for spec in config["teachers"]:
        path = prepared / spec["prepared_path"]
        with np.load(path, allow_pickle=False) as raw:
            source = {name: raw[name].copy() for name in raw.files}
        context = load_context(path)
        _restore(env, context, work)
        trace = out / f"teacher_{spec['slot']}_replay.h5"
        outcome = execute_trace(env, work, trace, config["horizon"], actions=source["actions"])
        _need(
            outcome["controls"] == 1000 and outcome["walking"],
            "Initial teacher replay lost walking",
        )
        actual = load_trace(trace)
        for name in ("states", "warmstarts", "integrations", "actions", "costs"):
            checks.append(_comparison(actual[name], source[name], f"teacher{spec['slot']}/{name}"))
        checks.append(
            _comparison(
                actual["observations"][:-1],
                source["observations"],
                f"teacher{spec['slot']}/pre_action_observations",
            )
        )
        checks.append(
            _comparison(
                observations(actual["states"][:-1]),
                source["observations"],
                f"teacher{spec['slot']}/pre_action_features",
            )
        )
        _restore(env, context, work)
    first = prepared / config["teachers"][0]["prepared_path"]
    return load_context(first), checks


class _MeteredActor:
    def __init__(self, actor: Any, work: Work):
        self.actor, self.work = actor, work

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.actor.predict(values, meter=self.work)


def _zero_residual(env: Any, actor: Any, context: dict, out: Path, work: Work) -> dict:
    from .trajectory_pi_native import _restore, execute_trace, load_trace
    from .trajectory_pi_optimizer import FrozenReconstructionCost

    _restore(env, context, work)
    trace = out / "zero_residual_native.h5"
    outcome = execute_trace(env, work, trace, 16, actor=actor)
    _need(outcome["controls"] == 16 and not outcome["fell"], "Zero-residual prefix fell before16")
    raw = load_trace(trace)
    _restore(env, context, work)
    states, costs, _ = env.batch_rollout(
        context["state"], raw["actions"][None], initial_warmstart=context["warmstart"]
    )
    comparisons = [
        _comparison(states[0], raw["states"][1:], "zero_residual/post_states"),
        _comparison(costs, np.asarray([raw["costs"].sum()]), "zero_residual/cost"),
    ]
    cost = FrozenReconstructionCost(env, _MeteredActor(actor, work), context["state"], 0.12)
    matching = cost(states, raw["actions"][None])
    _need(
        np.array_equal(cost.last_D, np.zeros(1)) and np.array_equal(matching, np.zeros(1)),
        "Canonical bounded action function does not give exact zero residual",
    )
    with (out / "zero_residual_forecast.npz").open("xb") as stream:
        np.savez_compressed(stream, states=states, costs=costs, D=cost.last_D, matching=matching)
    return dict(
        controls=16, comparisons=comparisons, D=0.0, actual_actions_dtype=str(raw["actions"].dtype)
    )


def _manual_bank(
    event: dict, env: Any, actor: Any, context: dict, alpha: float, work: Work
) -> dict:
    from src.utils.rollout_data import pre_action_states, pre_fall_action_mask

    from .trajectory_pi_student import observations

    post, actions = event["states"], event["actions"]
    mask = pre_fall_action_mask(env._is_healthy(post[..., 2], post[..., 3]))
    pre = pre_action_states(context["state"], post)
    comparisons = [
        _comparison(actions[0], event["current_actions"], "sample0_is_current", exact=True),
        _comparison(
            pre[:, 0],
            np.broadcast_to(context["state"], pre[:, 0].shape),
            "candidate_pre_action_initial",
            exact=True,
        ),
    ]
    if alpha:
        valid = np.all(
            (np.isfinite(post).all(axis=-1) & np.isfinite(actions).all(axis=-1)) | ~mask,
            axis=1,
        )
        executable = mask & valid[:, None]
        prediction = actor.predict(observations(pre[executable]), meter=work).astype(np.float64)
        per_control = np.zeros(mask.shape, dtype=np.float64)
        per_control[executable] = 0.5 * np.sum((actions[executable] - prediction) ** 2, axis=-1)
        d = np.sum(per_control, axis=1)
        d[~valid] = np.inf
        comparisons.append(_comparison(event["D"], d, "half_summed_D", exact=True))
        comparisons.append(
            _comparison(event["J"], event["C"] + alpha * d, "C_plus_alpha_D", exact=True)
        )
    else:
        _need(event["D"] is None, "Alpha0 population reconstruction must remain unmeasured")
        comparisons.append(_comparison(event["J"], event["C"], "alpha0_is_native_C", exact=True))
    expected = np.clip(
        event["current_actions"]
        + np.einsum("k, kha -> ha", event["weights"], actions - event["current_actions"][None]),
        env.action_bounds[0],
        env.action_bounds[1],
    )
    comparisons.append(
        _comparison(event["post_actions"], expected, "weighted_update_without_shift", exact=True)
    )
    _need(np.allclose(event["weights"].sum(), 1.0, rtol=0, atol=1e-14), "Weights not normalized")
    scores = event["J"]
    weights = np.exp(-(scores - np.min(scores)) / event["lambda"])
    weights /= weights.sum()
    comparisons.append(_comparison(event["weights"], weights, "softmin_of_total_objective"))
    return dict(iteration=event["iteration"], alpha=alpha, comparisons=comparisons)


class _ReplayBank:
    """Recompute the archived optimizer on paid banks without native dynamics."""

    def __init__(self, env: Any, events: list[dict], context: dict, work: Work):
        self.env, self.events, self.context, self.work = env, events, context, work
        self.cursor = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def batch_rollout(self, start: np.ndarray, actions: np.ndarray, **kwargs: Any) -> tuple:
        from .trajectory_pi_native import same_bytes

        _need(self.cursor < len(self.events), "Archived optimizer requested an extra bank")
        event = self.events[self.cursor]
        _need(
            same_bytes(start, self.context["state"])
            and same_bytes(kwargs["initial_warmstart"], self.context["warmstart"])
            and same_bytes(actions, event["actions"]),
            "Archived optimizer generated a different bank",
        )
        self.cursor += 1
        return event["states"].copy(), event["C"].copy(), np.empty((*actions.shape[:2], 0))


def _prototype_replay(
    env: Any,
    events: list[dict],
    context: dict,
    initial: np.ndarray,
    config: Any,
    seed: int,
    solved: dict,
    work: Work,
) -> dict:
    from src.mppi.mppi import MPPI

    path = Path(__file__).resolve().parents[3] / PROTOTYPE
    _need(sha(path) == PROTOTYPE_SHA256, "Archived prototype bytes changed")
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "FullPI")
        or (isinstance(node, ast.FunctionDef) and node.name == "optimize")
    ]
    _need(len(nodes) == 2, "Reviewed archived optimizer nodes absent")
    scope = dict(np=np, MPPI=MPPI, FINAL_SCALE=0.1)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    exact_class, planners = scope["FullPI"], []

    def capture(*args: Any, **kwargs: Any) -> Any:
        planner = exact_class(*args, **kwargs)
        planners.append(planner)
        return planner

    scope["FullPI"] = capture
    replay = _ReplayBank(env, events, context, work)
    best, curve = work.call(
        "archived_optimizer_bank_replay",
        dict(replay_optimizer_passes=2, replay_sampling_controls=512_000),
        lambda: scope["optimize"](
            replay, config, context["state"], context["warmstart"], initial, seed, 2
        ),
    )
    _need(replay.cursor == 2 and len(planners) == 1, "Archived optimizer replay incomplete")
    comparisons = [
        _comparison(best, solved["best_actions"], "prototype_best_nominal", exact=True),
        _comparison(
            curve,
            np.asarray([row["C"] for row in solved["curve"]]),
            "prototype_native_curve",
            exact=True,
        ),
        _comparison(planners[0].U, solved["final_actions"], "prototype_final_U", exact=True),
    ]
    _need(
        fingerprint(planners[0].snapshot()["rng_state"])
        == fingerprint(solved["planner_snapshot"]["rng_state"]),
        "Prototype private PCG differs",
    )
    return dict(
        native_controls=0,
        model_controls=0,
        comparisons=comparisons,
        extra_sampling_controls=512_000,
        prototype_sha256=sha(path),
    )


def _solves(
    config: dict, prepared: Path, out: Path, env: Any, actor: Any, context: dict, work: Work
) -> tuple[list, dict]:
    from src.utils.config import MPPIConfig

    from .trajectory_pi_native import SolveRecorder, _restore, same_bytes
    from .trajectory_pi_optimizer import FrozenReconstructionCost, optimize_full

    with np.load(prepared / config["teachers"][0]["prepared_path"], allow_pickle=False) as raw:
        initial = raw["actions"].copy()
    cfg, results = MPPIConfig(**config["planner"]), []
    manual, equivalent = [], None
    seed = config["seeds"]["preflight_physical"]
    for arm in config["arms"]:
        alpha = arm["alpha"]
        path = out / ("solve_" + arm["name"])
        path.mkdir()
        _restore(env, context, work)
        recorder = SolveRecorder(path, actor, env, work, context, [0, 1])
        events, before, prior_after = [], None, None
        cost = FrozenReconstructionCost(env, _MeteredActor(actor, work), context["state"], alpha)

        def observer(event: dict) -> None:
            nonlocal before, prior_after
            recorder(event)
            if event["phase"] == "before":
                before = event
                _need(
                    same_bytes(event["initial_state"], context["state"])
                    and same_bytes(event["initial_warmstart"], context["warmstart"]),
                    "A full solve changed its fixed context",
                )
                if prior_after is not None:
                    _need(
                        same_bytes(event["U"], prior_after), "Horizon shifted between refinements"
                    )
            else:
                _need(before is not None, "Missing pre-refinement record")
                manual.append(_manual_bank(event, env, actor, context, alpha, work))
                prior_after = event["post_actions"].copy()
                if not alpha:
                    events.append(deepcopy(event))

        try:
            env._pi_in_solve = True
            solved = optimize_full(
                env,
                cfg,
                context["state"],
                context["warmstart"],
                initial,
                seed,
                2,
                prior_cost=cost,
                observer=observer,
            )
        finally:
            env._pi_in_solve = False
            recorder.close()
        _need(
            solved["counts"]["completed_passes"] == 2 and not solved["final_update_scored"],
            "Unexpected full-sequence pass/selection contract",
        )
        rows = solved["curve"]
        selected = int(np.argmin([row["J"] for row in rows]))
        _need(
            solved["best_iteration"] == selected and solved["best_J"] == rows[selected]["J"],
            "Selection was not best scored total objective",
        )
        write(
            path / "result.json",
            {
                key: value
                for key, value in solved.items()
                if key not in {"best_states", "best_actions", "final_actions"}
            },
        )
        if not alpha:
            equivalent = _prototype_replay(env, events, context, initial, cfg, seed, solved, work)
        results.append(
            dict(
                arm=arm["name"],
                alpha=alpha,
                counts=solved["counts"],
                reconstruction_work=cost.counts,
            )
        )
    return manual, dict(solves=results, prototype_equivalence=equivalent)


def _action_parity(parent: Any, out: Path, work: Work, tolerance: float) -> dict:
    from .trajectory_pi_native import load_trace
    from .trajectory_pi_student import NumpyPolicy

    _need(tolerance == 1e-6, "Frozen Torch/NumPy parity tolerance must remain1e-6")
    values = load_trace(out / "zero_residual_native.h5")["observations"][:-1]
    _need(values.shape == (16, 17), "Actual16-control P0 prefix observations required")
    actor = work.call(
        "numpy_continuation_parent",
        dict(numpy_constructions=1),
        lambda: NumpyPolicy.from_policy(parent),
    )
    numpy_actions = actor.predict(values, meter=work)
    torch_actions = parent.predict(values, meter=work)
    maximum = float(np.max(np.abs(numpy_actions - torch_actions)))
    _need(maximum <= tolerance, "Actual incoming parent's Torch/NumPy action parity failed")
    with (out / "action_function_parity.npz").open("xb") as stream:
        np.savez_compressed(
            stream, observations=values, numpy_actions=numpy_actions, torch_actions=torch_actions
        )
    return dict(
        rows=16,
        measured_neural_rows=32,
        maximum_absolute_error=maximum,
        maximum_absolute_error_limit=tolerance,
        native_action_function="canonical_numpy",
    )


def _continuation(config: dict, prepared: Path, out: Path, work: Work) -> dict:
    import torch

    from .trajectory_pi_student import (
        fit_round,
        global_rng_state,
        load_stream,
        restore_global_rng,
        save_stream,
    )

    initial = prepared / config["parents"][0]["prepared_directory"]
    parent, state, _ = work.call(
        "load_initial_continuation",
        dict(checkpoint_loads=1, policy_constructions=1),
        lambda: load_stream(initial, device="cpu"),
    )
    parent_hash, old_state = parent.hash(), fingerprint(state)
    parity = _action_parity(
        parent, out, work, config["prerequisites"]["action_function_parity_atol"]
    )
    globals_before = global_rng_state()
    rng = np.random.Generator(np.random.PCG64(config["seeds"]["synthetic_continuation"]))
    mean, std = parent.normalization["mean"], parent.normalization["std"]
    x = (mean + 0.3 * std * rng.normal(size=(32, 17))).astype(np.float32)
    y = rng.uniform(-0.7, 0.7, size=(32, 6)).astype(np.float32)
    with (out / "synthetic_continuation_data.npz").open("xb") as stream:
        np.savez_compressed(stream, observations=x, actions=y)
    seed = config["seeds"]["synthetic_continuation"]

    def fit(policy: Any, previous: dict, updates: int, resume: bool = False) -> tuple:
        return work.call(
            "synthetic_continuation_fit",
            dict(policy_constructions=1, optimizer_constructions=1),
            lambda: fit_round(
                policy,
                x,
                y,
                updates=updates,
                seed=seed,
                optimizer_state=previous,
                resume_fit=resume,
                meter=work,
            ),
        )

    def save(name: str, policy: Any, previous: dict) -> dict:
        return work.call(
            "continuation_checkpoint",
            dict(stream_writes=1),
            lambda: save_stream(out / name, policy, previous, metadata=dict(synthetic_only=True)),
        )

    full, full_state, full_evidence = fit(parent, state, 4)
    full_envelope = save("continuation_full", full, full_state)
    restore_global_rng(globals_before)
    split, split_state, split_evidence = fit(parent, state, 2)
    middle = save("continuation_middle", split, split_state)
    # Deliberately disturb every global CPU stream; loading must restore them.
    torch.random.default_generator.manual_seed(seed + 1)
    np.random.seed(seed + 1)
    import random

    random.seed(seed + 1)
    loaded, loaded_state, _ = work.call(
        "reload_continuation",
        dict(checkpoint_loads=1, policy_constructions=1),
        lambda: load_stream(out / "continuation_middle", device="cpu"),
    )
    _need(
        fingerprint(global_rng_state()) == middle["global_rng_fingerprint"],
        "Checkpoint did not restore its global RNG envelope",
    )
    resumed, resumed_state, resumed_evidence = fit(loaded, loaded_state, 2, True)
    final_envelope = save("continuation_resumed", resumed, resumed_state)

    def operative(value: dict) -> dict:
        return {key: item for key, item in value.items() if key != "parent_policy_hash"}

    _need(
        full.hash() == resumed.hash()
        and fingerprint(full.state_dict()) == fingerprint(resumed.state_dict())
        and fingerprint(full._generator.get_state()) == fingerprint(resumed._generator.get_state())
        and fingerprint(operative(full_state)) == fingerprint(operative(resumed_state))
        and full_envelope["global_rng_fingerprint"] == final_envelope["global_rng_fingerprint"],
        "Exact Adam/private/global RNG continuation failed",
    )
    _need(
        parent.hash() == parent_hash and fingerprint(state) == old_state,
        "Synthetic assay modified an incoming parent",
    )
    result = dict(
        synthetic_rows=32,
        action_function_parity=parity,
        synthetic_updates=8,
        legacy_parent_refitted=False,
        excluded_lineage_fields=["parent_policy_hash"],
        policy_hash=full.hash(),
        optimizer_operative_fingerprint=fingerprint(operative(full_state)),
        global_rng_fingerprint=full_envelope["global_rng_fingerprint"],
        fitting=[full_evidence, split_evidence, resumed_evidence],
    )
    write(out / "continuation.json", result)
    return result


def _fall_accounting() -> dict:
    from .trajectory_pi_native import trace_outcome

    states = np.zeros((3, 19), dtype=np.float64)
    value = trace_outcome(states, np.array([1.0, 21.0]), 5, True)
    _need(
        value["raw_native_cost"] == 22.0
        and value["absorbing_tail_cost"] == 15.0
        and value["complete_cost"] == 37.0
        and value["controls"] == 2
        and not value["full_survival"]
        and not value["walking"],
        "Fall cost must charge recorded terminal once and five per missing control",
    )
    return value


def _budget_enforcement(out: Path) -> dict:
    limits = dict(wall_seconds=10, output_bytes=1024**2, memory_bytes=8 * 1024**3, fixture_units=1)
    work, entered = Work(out / "budget_fixture", limits), []
    try:
        work.call("over_cap_must_not_enter", dict(fixture_units=2), lambda: entered.append(True))
    except RuntimeError:
        pass
    _need(not entered and not work.counts, "Budget was checked after entering paid call")

    def fail() -> None:
        raise ValueError("Declared synthetic failure; no research work")

    try:
        work.call("failed_fixture", dict(fixture_units=1), fail)
    except ValueError:
        pass
    _need(
        work.counts == dict(requested_fixture_units=1, unknown_partial_fixture_units=1),
        "Failed-call ownership disappeared",
    )
    receipt = work.close("EXPECTED_SYNTHETIC_FAILURE")
    return dict(
        rejected_before_entry=True,
        synthetic_failure_retained=True,
        fixture_work=receipt["counts"],
        actual_physics=0,
        actual_fitting=0,
    )


def run_preflight(
    config_path: str | Path, prepared_dir: str | Path, source_manifest: str | Path, out: str | Path
) -> dict:
    """Execute P0 once; any missing binding or failed prerequisite stops launch.

    An exception becomes a durable FAIL_STOPPED receipt, with paid prefixes and
    uncertain reservations retained. Reusing an output directory is forbidden.
    The caller additionally enforces the external 600-second process deadline.
    """
    from .trajectory_pi_native import NumpyPolicy, _env

    config_path, prepared_dir = Path(config_path), Path(prepared_dir)
    source_manifest, out = Path(source_manifest), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    work = Work(out, _limits())
    checks = [dict(name=name, passed=False, status="NOT_RUN") for name in CHECK_NAMES]
    result = dict(
        schema="trajectory-pi-preflight-1",
        status="FAIL_STOPPED",
        config_sha256=sha(config_path) if config_path.is_file() else None,
        source_manifest_sha256=sha(source_manifest) if source_manifest.is_file() else None,
        prepared_manifest_sha256=sha(prepared_dir / "manifest.json")
        if (prepared_dir / "manifest.json").is_file()
        else None,
        checks=checks,
        comparison_tolerances=dict(rtol=0, atol=ATOL),
    )
    env, current = None, None

    def passed(name: str, evidence: Any) -> None:
        row = next(item for item in checks if item["name"] == name)
        row.update(passed=True, status="PASS", evidence=evidence)
        work.record("prerequisite_passed", name=name)

    try:
        config, _ = _validate_bindings(config_path, prepared_dir, source_manifest)
        work.limits = _limits(config)
        result["versions"] = _runtime(config)
        env = _env(dict(task=config["task"], physics_threads=8), work)
        actor = work.call(
            "numpy_parent",
            dict(numpy_constructions=1, checkpoint_loads=1),
            lambda: NumpyPolicy.from_checkpoint(
                prepared_dir / config["parents"][0]["prepared_directory"] / "policy.pt"
            ),
        )
        current = "full_context_replay"
        context, comparisons = _replays(config, prepared_dir, out, env, work)
        passed(current, comparisons)
        passed(
            "pre_action_index",
            dict(
                source="all2000 actual teacher controls",
                first_index=0,
                observations="N pre-action features; terminal context separate",
            ),
        )
        current = "zero_residual_reproduction"
        passed(current, _zero_residual(env, actor, context, out, work))
        current = "full_sequence_no_shift"
        manual, solves = _solves(config, prepared_dir, out, env, actor, context, work)
        passed(current, solves)
        passed("score_formula", manual)
        current = "checkpoint_continuation"
        passed(current, _continuation(config, prepared_dir, out, work))
        current = "fall_accounting"
        passed(current, _fall_accounting())
        current = "budget_enforcement"
        passed(current, _budget_enforcement(out))
        work.check(reserve_output=128 * 1024)
        _need(all(row["passed"] is True for row in checks), "Incomplete prerequisite roster")
        result["status"] = "PASS"
    except Exception as error:
        result["error"] = repr(error)
        if current is not None:
            next(item for item in checks if item["name"] == current).update(
                status="FAIL_STOPPED", error=repr(error)
            )
        work.record("preflight_stopped", error=repr(error), failed_check=current)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                result.update(status="FAIL_STOPPED", cleanup_error=repr(error))
    try:
        receipt = work.close("COMPLETED" if result["status"] == "PASS" else "FAILED")
    except Exception as error:
        result.update(status="FAIL_STOPPED", accounting_error=repr(error))
        if not work.closed:
            receipt = work.close("FAILED")
        else:
            receipt = work.receipt()
    units = set(_limits()) - {"wall_seconds", "output_bytes", "memory_bytes"}
    counts = {
        prefix + unit: 0
        for unit in units
        for prefix in ("requested_", "completed_", "unknown_partial_")
    }
    counts.update(receipt["counts"])
    result.update(
        work=counts, accounting={key: value for key, value in receipt.items() if key != "counts"}
    )
    write(out / "preflight.json", result)
    return result
