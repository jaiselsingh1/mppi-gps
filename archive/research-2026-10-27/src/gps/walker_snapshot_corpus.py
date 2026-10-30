"""Pure import of historical Walker state/warmstart contexts.

This module constructs no environment, imports no simulator, evaluates no
actor, and exports no teacher action or return labels. Every requested
episode/time slot survives extraction, including unavailable post-fall slots.
Historical identity checks are not a fresh current qualification gate.
Recovered original runner bytes can additionally close historical source
provenance. Otherwise independence remains attributed to metadata and two
instrumented replays, with the runner identity gap reported explicitly.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.gps.walker_owned_credit import load_qualified_planner_config, snapshot_digest
from src.utils.config import MPPIConfig


ROOT = Path(__file__).resolve().parents[2]
HORIZON = 1000
CPU_METHODS = ("reset", "step", "get_state", "get_warmstart", "set_state")
TASK_DEFAULTS = {
    "frame_skip": 4, "cost_style": "gymnasium", "target_velocity": 1.5,
    "forward_reward_weight": 1., "healthy_reward": 1., "vel_cost_weight": 1.,
    "vertical_velocity_cost_weight": 0., "angle_cost_weight": .1,
    "unhealthy_cost_weight": 5., "ctrl_cost_weight": .001,
    "terminal_unhealthy_weight": 20., "apply_terminal_cost_on_done": True,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _readonly(value):
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _integer(value, name, minimum=0):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _task_signature(task):
    unknown = set(task) - set(TASK_DEFAULTS) - {"max_steps", "velocity_cost_weight"}
    if unknown:
        raise ValueError("Unrecognized historical Walker task fields: " + ", ".join(sorted(unknown)))
    signature = {**TASK_DEFAULTS, **{key: value for key, value in task.items() if key in TASK_DEFAULTS}}
    if task.get("velocity_cost_weight", signature["vel_cost_weight"]) != signature["vel_cost_weight"]:
        raise ValueError("Historical velocity-cost alias is inconsistent")
    if task.get("max_steps", HORIZON) != HORIZON or signature["cost_style"] != "target_velocity":
        raise ValueError("Corpus must use the declared 1000-control target-velocity task")
    return signature


def _methods(path):
    module = ast.parse(Path(path).read_text())
    classes = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MuJoCoEnv"]
    if len(classes) != 1:
        raise ValueError("Expected one archived MuJoCoEnv definition")
    return {node.name: node for node in classes[0].body if isinstance(node, ast.FunctionDef)}


def _ast_hash(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _verify_independent_runner(path):
    module = ast.parse(Path(path).read_text())
    raw_imports = [node for node in ast.walk(module) if isinstance(node, ast.ImportFrom)
                   and node.module == "src.mppi.mppi" and
                   any(alias.name == "MPPI" and alias.asname in (None, "MPPI") for alias in node.names)]
    forbidden = []
    for node in ast.walk(module):
        names = []
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.arg):
            names = [node.arg]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
        forbidden.extend(name for name in names if any(word in name.lower() for word in ("actor", "critic", "policy")))
    constructors = [node for node in ast.walk(module) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "MPPI"]
    plans = [node for node in ast.walk(module) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "plan_step"]
    resets = [node for node in ast.walk(module) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "reset"
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "controller"
              and {keyword.arg for keyword in node.keywords} == {"seed"}]
    if (not raw_imports or forbidden or not constructors or not plans or not resets
            or any(len(node.args) != 2 or node.keywords for node in constructors)
            or any(len(node.args) != 1 or {keyword.arg for keyword in node.keywords} != {"initial_warmstart"}
                   for node in plans)):
        raise ValueError("Recovered runner does not satisfy the raw independent MPPI call contract")
    return {"raw_mppi_import": True, "no_actor_critic_policy_references": True,
            "constructor_count": len(constructors), "plan_call_count": len(plans),
            "plan_keywords": ["initial_warmstart"], "private_seed_reset_present": True}


def _source_compatibility(source_root, provenance, model_path, runner_source=None):
    hashes = provenance["source_sha256"]
    source_root = Path(source_root).resolve()
    required = ("src/envs/walker2d.py", "src/envs/mujoco_env.py", "assets/walker2d.xml",
                "src/mppi/mppi.py", "src/utils/config.py")
    archived_hashes = {}
    current_hashes = {}
    for relative in required:
        archived = source_root / relative
        if not archived.is_file() or hashes.get(relative) != _sha(archived):
            raise ValueError(f"Archived source/model hash does not match metadata: {relative}")
        archived_hashes[relative] = hashes[relative]
        current = model_path if relative == "assets/walker2d.xml" else ROOT / relative
        current_hashes[relative] = _sha(current)
    for relative in ("assets/walker2d.xml", "src/envs/walker2d.py"):
        if archived_hashes[relative] != current_hashes[relative]:
            raise ValueError(f"Current task model or Walker source differs: {relative}")
    archived_methods = _methods(source_root / "src/envs/mujoco_env.py")
    current_methods = _methods(ROOT / "src/envs/mujoco_env.py")
    cpu = {}
    for name in CPU_METHODS:
        if name not in archived_methods or name not in current_methods:
            raise ValueError(f"Required CPU reset/step/state method missing: {name}")
        previous, current = _ast_hash(archived_methods[name]), _ast_hash(current_methods[name])
        if previous != current:
            raise ValueError(f"CPU reset/step/state method differs from archive: {name}")
        cpu[name] = previous
    setter = current_methods.get("set_warmstart")
    if setter is None:
        raise ValueError("Current solver warmstart setter is missing")
    direct_assignment = any(
        isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == "warmstart"
        and any(isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                and target.value.attr == "qacc_warmstart" for target in node.targets)
        for node in ast.walk(setter))
    physics_calls = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("mj_step", "mj_forward", "rollout") for node in ast.walk(setter))
    if not direct_assignment or physics_calls:
        raise ValueError("Current warmstart setter is not a direct solver-state restore")
    if "set_warmstart" in archived_methods and _ast_hash(archived_methods["set_warmstart"]) != _ast_hash(setter):
        raise ValueError("Existing archived warmstart setter differs")
    runner_relative = "scripts/runners/run_walker2d.py"
    expected_runner = hashes.get(runner_relative)
    archive_runner_path = source_root / runner_relative
    runner_path = Path(runner_source).resolve() if runner_source is not None else archive_runner_path
    actual_runner = _sha(runner_path) if runner_path.is_file() else None
    if runner_source is not None and (expected_runner is None or actual_runner != expected_runner):
        raise ValueError("Recovered runner source does not match the recorded historical runner hash")
    runner_matches = expected_runner == actual_runner if expected_runner is not None and actual_runner is not None else None
    runner_contract = _verify_independent_runner(runner_path) if runner_matches else None
    runner_identity = {"path": str(runner_path), "expected_sha256": expected_runner,
        "actual_sha256": actual_runner,
        "matches": runner_matches, "recovery_source": str(runner_path) if runner_source is not None else None,
        "archive_runner_path": str(archive_runner_path),
        "archive_runner_actual_sha256": _sha(archive_runner_path) if archive_runner_path.is_file() else None,
        "recovered_source_identity": runner_source is not None,
        "independent_call_contract": runner_contract,
        "certifies_all_historical_independence": False}
    verified_keys = set(required)
    if runner_matches:
        verified_keys.add(runner_relative)
    for relative in set(hashes) - verified_keys - {runner_relative}:
        path = (source_root / relative).resolve()
        if (path.is_relative_to(source_root) and not Path(relative).is_absolute()
                and path.is_file() and _sha(path) == hashes[relative]):
            verified_keys.add(relative)
    all_recorded = runner_matches is True and verified_keys == set(hashes)
    return {"archive_source_root": str(source_root), "archived_hashes": archived_hashes,
            "current_hashes": current_hashes, "cpu_method_ast_hashes": cpu,
            "historical_runner_identity": runner_identity,
            "all_recorded_source_hashes_verified": all_recorded,
            "unverified_recorded_source_keys": sorted(set(hashes) - verified_keys),
            "current_set_warmstart_ast_hash": _ast_hash(setter),
            "current_set_warmstart_added": "set_warmstart" not in archived_methods,
            "allowed_state_only_differences": [relative for relative in required
                if archived_hashes[relative] != current_hashes[relative]],
            "scope": "State-only restart compatibility; no planner continuation, all-source identity, or fresh qualification claim"}


def _healthy(states):
    return (states[..., 2] > .8) & (states[..., 2] < 2.) & (states[..., 3] > -1.) & (states[..., 3] < 1.)


def _pair_paths(pair):
    if isinstance(pair, Mapping):
        result = {name: Path(pair[name]).resolve() for name in ("npz", "json")}
        result["source_root"] = Path(pair.get("source_root", result["npz"].parents[2])).resolve()
        if "config" in pair:
            result["config"] = Path(pair["config"]).resolve()
        if "runner_source" in pair:
            result["runner_source"] = Path(pair["runner_source"]).resolve()
    else:
        if len(pair) != 2:
            raise ValueError("Each corpus pair requires NPZ and matching JSON paths")
        result = {"npz": Path(pair[0]).resolve(), "json": Path(pair[1]).resolve()}
        result["source_root"] = result["npz"].parents[2]
    return result


def load_corpus(pairs, task_path, model_path, *, times=(0, 500, 900),
                train_reset_seeds=tuple(range(30, 35))):
    """Return a state-only catalog; never select episodes based on outcomes.

    Pairs are (npz,json) tuples or mappings with npz/json and optional
    source_root/config/runner_source paths. A supplied recovered runner MUST
    match the recorded runner hash and its independent call contract. The
    ordinary archive root is npz.parents[2]; its
    config is the NPZ sibling named by provenance.config_path. Reset ids define
    whole-root splits across every planner stream and descendant handover.
    """
    task_path, model_path = Path(task_path).resolve(), Path(model_path).resolve()
    task_hash, model_hash = _sha(task_path), _sha(model_path)
    task = _task_signature(json.loads(task_path.read_text()))
    times = tuple(_integer(value, "handover time") for value in times)
    if not times or len(set(times)) != len(times) or any(value >= HORIZON for value in times):
        raise ValueError("Handover times must be distinct predeclared integers in [0,1000)")
    train_seeds = frozenset(_integer(seed, "training reset seed") for seed in train_reset_seeds)
    if not train_seeds or not pairs:
        raise ValueError("Require a declared nonempty training reset split and source pairs")
    inputs, contexts, seen_files, seen_episodes = [], [], set(), set()
    original_controls = original_candidate_slots = survived_count = 0
    original_wall = 0.
    for pair_index, pair in enumerate(pairs):
        paths = _pair_paths(pair)
        if paths["npz"] in seen_files:
            raise ValueError("A source NPZ cannot be imported more than once")
        seen_files.add(paths["npz"])
        report = json.loads(paths["json"].read_text())
        provenance = report["provenance"]
        if (provenance["task_config_sha256"] != task_hash or
                provenance["source_sha256"].get("assets/walker2d.xml") != model_hash
                or _task_signature(report["task"]) != task):
            raise ValueError("Historical task/model identity differs from current supplied bytes")
        if (provenance["packages"].get("mujoco") != "3.6.0"
                or not isinstance(provenance["packages"].get("numpy"), str)
                or not isinstance(provenance["packages"].get("scipy"), str)
                or not np.isclose(provenance["control_dt"], .008, rtol=0., atol=1e-15)
                or not np.isclose(provenance["physics_dt"], .002, rtol=0., atol=1e-15)
                or provenance["frame_skip"] != 4 or task["frame_skip"] != 4
                or not isinstance(provenance["backend"], str) or not provenance["backend"]):
            raise ValueError("Historical MuJoCo/control-boundary metadata is incompatible")
        source_config = paths.get("config", paths["npz"].parent / Path(provenance["config_path"]).name)
        if _sha(source_config) != provenance["config_sha256"]:
            raise ValueError("Historical tuned configuration bytes do not match metadata")
        configured = load_qualified_planner_config(source_config)
        try:
            reported_config = MPPIConfig(**report["mppi_config"])
        except TypeError as error:
            raise ValueError("Historical configuration metadata has unknown fields") from error
        if asdict(reported_config) != asdict(configured):
            raise ValueError("Historical report and tuned configuration disagree")
        compatibility = _source_compatibility(paths["source_root"], provenance, model_path,
                                               paths.get("runner_source"))
        input_hashes = {"npz": _sha(paths["npz"]), "json": _sha(paths["json"])}
        with np.load(paths["npz"], allow_pickle=False) as archive:
            states, warms = archive["states"], archive["warmstarts"]
            steps_offsets, state_offsets = archive["episode_step_offsets"], archive["episode_state_offsets"]
            resets, planners = archive["reset_seeds"], archive["planner_seeds"]
            if (states.dtype != np.float64 or warms.dtype != np.float64
                    or states.ndim != 2 or states.shape[1] != 19 or warms.shape != (len(states), 9)
                    or not np.isfinite(states).all() or not np.isfinite(warms).all()):
                raise ValueError("Corpus requires finite float64 FULLPHYSICS19 and aligned warmstart9")
            episodes = report["episodes"]
            count = len(episodes)
            if (count < 1 or report["run"].get("complete") is not True
                    or report["run"]["requested_episodes"] != count
                    or resets.shape != (count,) or planners.shape != (count,)
                    or resets.dtype.kind not in "iu" or planners.dtype.kind not in "iu"
                    or len(set(resets.tolist())) != count
                    or steps_offsets.shape != (count + 1,) or state_offsets.shape != (count + 1,)
                    or steps_offsets.dtype.kind not in "iu" or state_offsets.dtype.kind not in "iu"
                    or steps_offsets[0] != 0 or state_offsets[0] != 0
                    or np.any(np.diff(steps_offsets) < 1)
                    or not np.array_equal(np.diff(state_offsets), np.diff(steps_offsets) + 1)
                    or state_offsets[-1] != len(states)
                    or archive["actions"].shape != (int(steps_offsets[-1]), 6)
                    or archive["costs"].shape != (int(steps_offsets[-1]),)):
                raise ValueError("Episode metadata, seed identities, offsets or raw trace lengths disagree")
            source_episode_rows = []
            for episode_index, episode in enumerate(episodes):
                steps = _integer(episode["steps"], "source episode steps", 1)
                reset_seed, planner_seed = int(resets[episode_index]), int(planners[episode_index])
                survived = episode["survived"]
                if (type(survived) is not bool or steps > HORIZON
                        or episode["reset_seed"] != reset_seed or episode["planner_seed"] != planner_seed
                        or steps != steps_offsets[episode_index + 1] - steps_offsets[episode_index]
                        or (survived and steps != HORIZON)):
                    raise ValueError("Source episode completion or seed provenance is inconsistent")
                identity = (reset_seed, planner_seed)
                if identity in seen_episodes:
                    raise ValueError("Duplicate source reset/planner episode")
                seen_episodes.add(identity)
                lo, hi = int(state_offsets[episode_index]), int(state_offsets[episode_index + 1])
                episode_states = states[lo:hi]
                health = _healthy(episode_states)
                if (not health[:-1].all() or bool(health[-1]) != survived
                        or not np.isclose(episode_states[0, 0], 0., atol=1e-12)
                        or not np.allclose(np.diff(episode_states[:, 0]), .008, rtol=0., atol=1e-10)):
                    raise ValueError("Source clock or pre-first-fall state boundary is inconsistent")
                source_episode = {"episode_index": episode_index, "reset_seed": reset_seed,
                    "planner_seed": planner_seed, "steps": steps, "terminated": not survived,
                    "survived": survived, "state_offset": lo, "state_end_offset": hi}
                source_episode_rows.append(source_episode)
                original_controls += steps
                original_candidate_slots += configured.K * configured.H * (steps + configured.initial_iterations - 1)
                survived_count += survived
                for handover in times:
                    active = not (not survived and handover >= steps)
                    index = lo + handover if active else None
                    state = _readonly(states[index]) if active else None
                    warm = _readonly(warms[index]) if active else None
                    ancestry = {"source_pair": pair_index, "source_npz": str(paths["npz"]),
                        "source_episode_index": episode_index, "reset_seed": reset_seed,
                        "planner_seed": planner_seed, "handover": handover,
                        "root_cluster": f"reset{reset_seed}",
                        "role": "root" if handover == 0 else "guide"}
                    contexts.append({"id": f"pair{pair_index}:reset{reset_seed}:planner{planner_seed}:t{handover}",
                        "split": "train" if reset_seed in train_seeds else "heldout",
                        "role": ancestry["role"], "active": active, "state": state, "warmstart": warm,
                        "snapshot_digest": snapshot_digest(SimpleNamespace(state=state, warmstart=warm)) if active else None,
                        "ancestry": ancestry, "input_hashes": dict(input_hashes), "source_array_index": index,
                        "source_episode": dict(source_episode), "remaining": HORIZON - handover,
                        "healthy": bool(_healthy(state)) if active else False,
                        "fall_before_handover": not active,
                        "absence_reason": None if active else "source_fell_at_or_before_handover"})
        elapsed = float(report["run"]["elapsed_seconds"])
        if not np.isfinite(elapsed) or elapsed < 0.:
            raise ValueError("Historical wall time must be finite and nonnegative")
        original_wall += elapsed
        inputs.append({"pair_index": pair_index, "npz": str(paths["npz"]), "json": str(paths["json"]),
            "input_hashes": input_hashes, "config": str(source_config),
            "config_sha256": provenance["config_sha256"], "config_signature": asdict(configured),
            "source_compatibility": compatibility, "packages": dict(provenance["packages"]),
            "historical_runner_identity": compatibility["historical_runner_identity"],
            "backend": provenance["backend"], "nthread": provenance.get("nthread"),
            "control_dt": provenance["control_dt"], "physics_dt": provenance["physics_dt"],
            "frame_skip": provenance["frame_skip"], "source_episodes": source_episode_rows})
    present_seeds = {context["ancestry"]["reset_seed"] for context in contexts}
    if not train_seeds <= present_seeds:
        raise ValueError("Declared training reset seeds are absent from the supplied corpus")
    counts = {"source_pairs": len(inputs), "source_episodes": len(seen_episodes),
        "context_slots": len(contexts), "active_contexts": sum(row["active"] for row in contexts),
        "inactive_contexts": sum(not row["active"] for row in contexts),
        "train_contexts": sum(row["split"] == "train" for row in contexts),
        "heldout_contexts": sum(row["split"] == "heldout" for row in contexts),
        "active_train_contexts": sum(row["active"] and row["split"] == "train" for row in contexts),
        "active_heldout_contexts": sum(row["active"] and row["split"] == "heldout" for row in contexts),
        "source_survived_episodes": survived_count, "source_failed_episodes": len(seen_episodes) - survived_count,
        "new_simulated_controls": 0, "new_actor_inference_rows": 0}
    all_sources_certified = all(row["source_compatibility"]["all_recorded_source_hashes_verified"] for row in inputs)
    independence_provenance = (
        "original runner source matches recorded hashes and statically uses raw independent MPPI; source provenance is not fresh current qualification or full historical replay"
        if all_sources_certified else
        "historical metadata and two instrumented replays; runner source not recovered yet")
    return {"metadata": {"schema": "walker-state-only-corpus-v1", "inputs": inputs,
        "task_path": str(task_path), "task_sha256": task_hash, "task_signature": task,
        "model_path": str(model_path), "model_sha256": model_hash, "times": list(times),
        "train_reset_seeds": sorted(train_seeds), "heldout_reset_seeds": sorted(present_seeds - train_seeds),
        "split_unit": "entire reset ancestry across all planner streams and handovers",
        "fresh_current_qualification_established": False, "teacher_action_or_return_labels_exported": False,
        "all_historical_source_identity_certified": all_sources_certified,
        "independence_provenance_assumption": independence_provenance,
        "historical_work": {"executed_controls": original_controls,
            "candidate_slots_reconstructed": original_candidate_slots,
            "total_controls_reconstructed": original_controls + original_candidate_slots,
            "reported_wall_seconds": original_wall, "durable_per_attempt_ledger_available": False,
            "accounting_scope": "Reconstructed completed historical search; unknown failed/interrupted work is not certified"},
        "assumptions": ["Same XML/task/Walker and CPU reset-step-state semantics permit a state-only simulator restart",
            "Matched original runner source establishes the declared call protocol, not a full replay of all historical episodes",
            "FULLPHYSICS plus recorded qacc_warmstart suffices with this model's no-activation/no-mocap setup",
            "External applied forces are zero; the new actor overwrites control before its first step",
            "Historical MuJoCo3.6 metadata is verified; no installed engine or fresh current behavior is tested",
            "New complete candidate-owned rollouts must supply all future learning and acceptance returns"]},
        "contexts": contexts, "counts": counts}
