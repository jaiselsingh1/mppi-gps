"""Finite, metered qualification of the existing MuJoCo PointMass MPPI.

This actor-free physical baseline implements neither policy learning nor the
protected trajectory-memory research controller. Only native PointMass.step
and PointMass.batch_rollout integrate physics, through the ledger below.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_CONFIG = {
    "schema": "mujoco-point-mass-workflow-v1",
    "environment": {"frame_skip": 1, "nthread": 1, "use_warp": False,
                    "ctrl_cost_weight": 0.01, "random_goal": True},
    "mppi": {"K": 256, "H": 64, "noise_sigma": 0.5, "lam": 0.01,
             "use_is_correction": False, "initial_iterations": 1},
    "validate": {"reset_seed": 310000000, "planner_seed": 410000000,
                 "tape_seed": 510000000, "tape_steps": 8, "K": 4, "H": 8,
                 "parity_atol": 1e-12, "max_physics_controls": 80, "max_queries": 11},
    "qualify": {"reset_seeds": [310000001, 310000002],
                "planner_seeds": [410000001, 410000002], "steps": 300,
                "max_physics_controls_per_phase": 9831000,
                "max_queries_per_phase": 1200, "paid_phases": 2},
}


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=_json)
        stream.write("\n")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz(root, name, **arrays):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"file": name, "sha256": _sha(path), "bytes": path.stat().st_size}


def load_config_dict(value):
    canonical = lambda x: json.dumps(x, sort_keys=True, allow_nan=False)
    if canonical(value) != canonical(DEFAULT_CONFIG):
        raise ValueError("This qualification requires the exact frozen PointMass profile")
    return json.loads(canonical(value))


def load_config(path=None):
    """No stage retuning is implicit in this first physical qualification."""
    return load_config_dict(DEFAULT_CONFIG if path is None else json.loads(Path(path).read_text()))


def _freeze(out, config, stage):
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "config.json", config)
    _write(out / "inputs.json", {"stage": stage, "fixed_roster": config[stage],
                                "native_reset_distributions": "unchanged PointMass.reset"})
    paths = ["scripts/runners/run_point_mass.py", "scripts/training/train_bidirectional_gps.py", "src/envs/point_mass.py",
             "src/envs/mujoco_env.py", "src/envs/base.py", "src/mppi/mppi.py",
             "src/utils/config.py", "src/utils/math.py", "assets/point_mass.xml",
             "tests/test_mujoco_point_mass_workflow.py", "configs/bidirectional_mujoco.json"]
    paths += [str(p.relative_to(_ROOT)) for p in sorted((_ROOT / "assets/common").rglob("*")) if p.is_file()]
    for name in ("src/__init__.py", "src/envs/__init__.py", "src/mppi/__init__.py", "src/utils/__init__.py",
                 "scripts/__init__.py", "scripts/runners/__init__.py", "scripts/training/__init__.py"):
        if (_ROOT / name).is_file():
            paths.append(name)
    files = []
    for name in paths:
        source, target = _ROOT / name, out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append({"file": name, "sha256": _sha(source), "bytes": source.stat().st_size})
    manifest = {"schema": config["schema"], "stage": stage, "files": files,
                "config_sha256": _sha(out / "config.json"), "inputs_sha256": _sha(out / "inputs.json")}
    _write(out / "manifest.json", manifest)
    return manifest


def _verify(out, manifest):
    if (_sha(out / "config.json") != manifest["config_sha256"] or
            _sha(out / "inputs.json") != manifest["inputs_sha256"]):
        raise ValueError("Frozen configuration or input roster changed")
    for row in manifest["files"]:
        for path in (_ROOT / row["file"], out / "source" / row["file"]):
            if path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
                raise ValueError("Frozen source changed: " + row["file"])


class BoundaryLedger:
    """Append a durable request before every native rollout or live control."""
    def __init__(self, path, max_controls, max_queries):
        self.path = Path(path)
        self.path.mkdir()
        self.ledger_path = self.path / "ledger.jsonl"
        self.stream = self.ledger_path.open("x")
        self.max_controls, self.max_queries = max_controls, max_queries
        self.context, self.last_model = {}, None
        self.counts = dict(requested_physics_controls=0, completed_physics_controls=0,
                           requested_physics_frames=0, completed_physics_frames=0,
                           queries=0, failed_queries=0, rejected_queries=0, unknown_partial_controls=0,
                           reset_calls=0, restore_calls=0, neural_rows=0,
                           backward_calls=0, optimizer_steps=0)

    def append(self, kind, **fields):
        row = {"kind": kind, "wall_seconds": time.perf_counter(), **fields}
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False, default=_json) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def query(self, kind, controls, inputs, operation, pack):
        if (type(controls) is not int or controls <= 0 or
                self.counts["requested_physics_controls"] + controls > self.max_controls or
                self.counts["queries"] >= self.max_queries):
            self.counts["rejected_queries"] += 1
            self.append("query_rejected", query_kind=kind, attempted_controls=controls,
                        requested_physics_frames=0, reason="fixed_budget")
            raise RuntimeError("Fixed physical-query budget exhausted before execution")
        self.counts["queries"] += 1
        query = f"q{self.counts['queries']:07d}"
        raw = _npz(self.path, f"queries/{query}_inputs.npz", **inputs,
                   context_json=np.array(json.dumps(self.context, sort_keys=True)))
        for key in ("requested_physics_controls", "requested_physics_frames"):
            self.counts[key] += controls  # This profile fixes frame_skip=1.
        self.append("query_requested", query_id=query, query_kind=kind, inputs=raw,
                    requested_physics_controls=controls, requested_physics_frames=controls,
                    context=self.context.copy())
        started, returned = time.perf_counter(), False
        try:
            value = operation()
            returned = True
            result = _npz(self.path, f"queries/{query}_result.npz", **pack(value))
        except BaseException as error:
            if returned:
                self.counts["completed_physics_controls"] += controls
                self.counts["completed_physics_frames"] += controls
            else:
                self.counts["unknown_partial_controls"] += controls
            self.counts["failed_queries"] += 1
            self.append("query_error", query_id=query, query_kind=kind,
                        completed_physics_controls=controls if returned else 0,
                        completed_physics_frames=controls if returned else 0,
                        partial_physics_unknown=not returned, error_type=type(error).__name__,
                        error=str(error), query_wall_seconds=time.perf_counter() - started)
            raise
        self.counts["completed_physics_controls"] += controls
        self.counts["completed_physics_frames"] += controls
        self.append("query_complete", query_id=query, query_kind=kind, result=result,
                    completed_physics_controls=controls, completed_physics_frames=controls,
                    query_wall_seconds=time.perf_counter() - started)
        if kind == "model_bank":
            self.last_model = result
        return value

    def close(self):
        self.append("archive_close", work=self.counts)
        self.stream.close()


def _snapshot(env):
    return dict(state=env.get_state(), warmstart=env.get_warmstart(), goal=env.goal.copy())


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _restore(env, snapshot, ledger):
    env.reset(state=snapshot["state"].copy(), goal=snapshot["goal"].copy())
    env.set_warmstart(snapshot["warmstart"].copy())
    ledger.counts["restore_calls"] += 1
    ledger.append("state_restored", physics_frames=0, goal=snapshot["goal"])
    actual = _snapshot(env)
    if not all(_same(value, actual[key]) for key, value in snapshot.items()):
        raise ValueError("Native physics/goal/warmstart restoration differs")


def _reset(env, seed, ledger):
    ledger.append("reset_requested", seed=seed, physics_frames=0)
    np.random.seed(seed)
    obs = env.reset()
    ledger.counts["reset_calls"] += 1
    raw = _npz(ledger.path, f"resets/reset{ledger.counts['reset_calls']}.npz", **_snapshot(env), observation=obs)
    ledger.append("reset_complete", seed=seed, raw=raw, physics_frames=0,
                  sampler_calls={"uniform": 2, "normal": 1}, scalar_variates={"uniform": 4, "normal": 2})
    return obs


def _instrument(env, ledger):
    original_batch, original_step = env.batch_rollout, env.step

    def batch(initial_state, actions, initial_warmstart=None):
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[-1] != 2 or not np.isfinite(actions).all() or np.any(abs(actions) > 1):
            raise ValueError("Finite native PointMass action tapes in [-1,1] required")
        before = _snapshot(env)
        def pack(value):
            actual = _snapshot(env)
            if not all(_same(v, actual[k]) for k, v in before.items()):
                raise ValueError("Planning mutated the live simulator, goal or warmstart")
            states, costs, sensors = value
            return dict(states=states, costs=costs, sensordata=sensors,
                        running_costs=env.running_cost(states, actions, sensors),
                        terminal_costs=env.terminal_cost(states[:, -1], sensors[:, -1]),
                        terminal_cost_role=np.array("planning_horizon_heuristic"))
        return ledger.query("model_bank", int(np.prod(actions.shape[:2])),
                            dict(initial_state=np.asarray(initial_state), actions=actions,
                                 initial_warmstart=np.zeros(env.model.nv) if initial_warmstart is None else initial_warmstart,
                                 goal=env.goal.copy()),
                            lambda: original_batch(initial_state, actions, initial_warmstart=initial_warmstart), pack)

    def step(action):
        action = np.asarray(action)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
            raise ValueError("Finite native PointMass action in [-1,1] required")
        def pack(value):
            obs, cost, done, info = value
            return dict(**_snapshot(env), observation=obs, stage_cost=np.array(cost), done=np.array(done),
                        metrics_json=np.array(json.dumps(env.task_metrics(), sort_keys=True)))
        return ledger.query("executed_control", 1, dict(**_snapshot(env), action=action),
                            lambda: original_step(action), pack)

    env.batch_rollout, env.step = batch, step


def _dependencies():
    from src.envs.point_mass import PointMass
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig
    return PointMass, MPPI, MPPIConfig


def _validation(config, ledger, PointMass, MPPI, MPPIConfig):
    spec = config["validate"]
    env = PointMass(**config["environment"])
    try:
        _instrument(env, ledger)
        _reset(env, spec["reset_seed"], ledger)
        start = _snapshot(env)
        actions = np.random.default_rng(spec["tape_seed"]).uniform(-0.2, 0.2, (1, spec["tape_steps"], 2))
        ledger.append("fixed_tape_draw", seed=spec["tape_seed"], uniform_variates=actions.size, physics_frames=0)
        states, costs, _ = env.batch_rollout(start["state"], actions, initial_warmstart=start["warmstart"])
        _restore(env, start, ledger)
        live, running = [], []
        for action in actions[0]:
            _, cost, _, _ = env.step(action)
            live.append(env.get_state()); running.append(cost)
        terminal = float(env.terminal_cost(env.get_state()[None], env.data.sensordata[None])[0])
        parity = bool(np.allclose(states[0], live, rtol=0, atol=spec["parity_atol"]) and
                      np.allclose(costs[0], sum(running) + terminal, rtol=0, atol=spec["parity_atol"]))
        ledger.append("model_live_parity", passed=parity, atol=spec["parity_atol"],
                      exact_states=_same(states[0], np.asarray(live)), terminal_cost_once=terminal)
        if not parity:
            raise ValueError("MuJoCo model/live parity failed")
        cfg = dict(config["mppi"], K=spec["K"], H=spec["H"])
        planner = MPPI(env, MPPIConfig(**cfg), seed=spec["planner_seed"])
        initial, private = _snapshot(env), planner.snapshot()
        action, info = planner.plan_step(initial["state"], initial_warmstart=initial["warmstart"])
        first_plan = planner.last_plan
        first_data = {k: v.copy() for k, v in planner.get_rollout_data().items()}
        _restore(env, initial, ledger); planner.restore(private)
        repeated, repeated_info = planner.plan_step(initial["state"], initial_warmstart=initial["warmstart"])
        replay = _same(action, repeated) and _same(first_plan, planner.last_plan)
        replay &= all(_same(v, planner.get_rollout_data()[k]) for k, v in first_data.items())
        _npz(ledger.path, "planner_replay.npz", **initial, nominal=private["U"],
             rng_json=np.array(json.dumps(private["rng_state"], sort_keys=True)),
             initial_plan=np.array(private["initial_plan"]), action=action, repeated_action=repeated,
             plan=first_plan, repeated_plan=planner.last_plan)
        ledger.append("planner_replay", passed=bool(replay), first_info=info, repeated_info=repeated_info)
        if not replay:
            raise ValueError("Restored MuJoCo planner decision differs")
        return {"parity_pass": True, "planner_replay_pass": True}
    finally:
        env.close()


def _qualification(config, ledger, PointMass, MPPI, MPPIConfig):
    spec = config["qualify"]
    env = PointMass(**config["environment"])
    episodes = []
    try:
        _instrument(env, ledger)
        for episode, (seed, planner_seed) in enumerate(zip(spec["reset_seeds"], spec["planner_seeds"])):
            obs = _reset(env, seed, ledger)
            planner = MPPI(env, MPPIConfig(**config["mppi"]), seed=planner_seed)
            states, warms, observations = [env.get_state()], [env.get_warmstart()], [obs]
            actions, costs, successes, distances, speeds = [], [], [], [], []
            for step in range(spec["steps"]):
                ledger.context = {"episode": episode, "step": step}
                before, private = _snapshot(env), planner.snapshot()
                action, info = planner.plan_step(before["state"], initial_warmstart=before["warmstart"])
                data = planner.get_rollout_data()
                _npz(ledger.path, f"decisions/episode{episode}_step{step:04d}.npz", **before,
                     elapsed=np.array(step), nominal_before=private["U"], plan=planner.last_plan,
                     nominal_after=planner.U, action=action, weights=data["weights"], raw_scores=data["costs"],
                     rng_before_json=np.array(json.dumps(private["rng_state"], sort_keys=True)),
                     rng_after_json=np.array(json.dumps(planner.snapshot()["rng_state"], sort_keys=True)),
                     initial_plan=np.array(private["initial_plan"]),
                     model_result_json=np.array(json.dumps(ledger.last_model, sort_keys=True)),
                     info_json=np.array(json.dumps(info, sort_keys=True, default=_json)))
                obs, cost, done, _ = env.step(action)
                if done:
                    raise ValueError("PointMass unexpectedly terminated before its frozen horizon")
                metrics = env.task_metrics()
                states.append(env.get_state()); warms.append(env.get_warmstart()); observations.append(obs)
                actions.append(action); costs.append(cost); successes.append(metrics["success"])
                distances.append(metrics["tip_dist"]); speeds.append(metrics["qvel_norm"])
            terminal = float(env.terminal_cost(states[-1][None], env.data.sensordata[None])[0])
            hold = 0
            for success in reversed(successes):
                if not success:
                    break
                hold += 1
            raw = _npz(ledger.path, f"episodes/episode{episode}.npz", states=np.asarray(states),
                       warmstarts=np.asarray(warms), goal=env.goal.copy(), observations=np.asarray(observations),
                       actions=np.asarray(actions), running_costs=np.asarray(costs), terminal_cost=np.array(terminal),
                       success=np.asarray(successes), distance=np.asarray(distances), speed=np.asarray(speeds))
            row = dict(episode=episode, reset_seed=seed, planner_seed=planner_seed, steps=spec["steps"],
                       success=bool(successes[-1]), settling_steps_at_end_descriptive=hold,
                       final_distance=distances[-1], final_speed=speeds[-1],
                       running_cost=float(sum(costs)), terminal_cost=terminal,
                       complete_cost=float(sum(costs) + terminal), raw=raw)
            ledger.append("episode_complete", **row, terminal_count=1)
            episodes.append(row)
        return {"episodes": episodes, "qualification_pass": all(e["success"] for e in episodes)}
    finally:
        env.close()


def _paid_repeat(out):
    left, right = out / "producer", out / "verification"
    names = {p.relative_to(left) for p in left.rglob("*.npz")}
    if names != {p.relative_to(right) for p in right.rglob("*.npz")}:
        raise ValueError("Paid-repeat raw-file roster differs")
    for name in names:
        with np.load(left / name, allow_pickle=False) as a, np.load(right / name, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or not all(_same(a[k], b[k]) for k in a.files):
                raise ValueError("Paid-repeat array differs: " + str(name))
    return {"passed": True, "array_pairs": len(names), "query_arrays_and_planner_snapshots": True}


def _run(out_dir, config, stage):
    started = time.perf_counter()
    config = load_config() if config is None else load_config_dict(config)
    out = Path(out_dir).resolve()
    manifest = _freeze(out, config, stage)
    phases, error, audit, detail = [], None, None, {}
    _write(out / "launch.json", {"stage": stage, "started_unix": time.time(), "python": sys.executable,
                               "platform": platform.platform(), "cpu_count": os.cpu_count()})
    try:
        packages = {}
        for name in ("mujoco", "numpy", "scipy", "warp-lang", "mujoco-warp"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        _write(out / "runtime.json", {"python": platform.python_version(),
               "packages": packages,
               "backend": "mujoco_cpu", "nthread": 1, "frame_skip": 1})
        PointMass, MPPI, MPPIConfig = _dependencies()
        for name in (["validation"] if stage == "validate" else ["producer", "verification"]):
            _verify(out, manifest)
            spec = config["validate" if stage == "validate" else "qualify"]
            ledger = BoundaryLedger(out / name, spec.get("max_physics_controls", spec.get("max_physics_controls_per_phase")),
                                    spec.get("max_queries", spec.get("max_queries_per_phase")))
            try:
                phase = (_validation if stage == "validate" else _qualification)(config, ledger, PointMass, MPPI, MPPIConfig)
                if ledger.counts["requested_physics_controls"] != ledger.max_controls or ledger.counts["queries"] != ledger.max_queries:
                    raise ValueError("Actual finite query roster differs from declaration")
                _write(ledger.path / "result.json", phase)
                if not detail:
                    detail = phase
            finally:
                ledger.close()
                phases.append({"phase": name, **ledger.counts})
        if stage == "qualify":
            audit = _paid_repeat(out)
            _write(out / "audit.json", audit)
        _verify(out, manifest)
    except BaseException as failure:
        error = {"type": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc()}
        _write(out / "error.json", error)
    result = dict(schema=config["schema"], stage=stage, software_pass=error is None,
                  parity_pass=bool(error is None and detail.get("parity_pass", False)),
                  qualification_pass=bool(error is None and detail.get("qualification_pass", False)),
                  paid_repeat_pass=bool(audit and audit["passed"]), independent_mppi_only=True,
                  trajectory_memory_implemented=False, actor_training=False, policy_authority=False,
                  scientific_loop_pass=False, details=detail, phases=phases,
                  work={key: sum(p[key] for p in phases) for key in phases[0] if key != "phase"} if phases else {},
                  wall_seconds=time.perf_counter() - started)
    _write(out / "result.json", result)
    return result


def validate(out_dir, config=None):
    return _run(out_dir, config, "validate")


def qualify(out_dir, config=None):
    return _run(out_dir, config, "qualify")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "qualify"), default="validate")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    result = (validate if args.stage == "validate" else qualify)(args.out_dir, load_config(args.config))
    print(json.dumps(result, sort_keys=True))
    passed = result["software_pass"] and (result["parity_pass"] if args.stage == "validate" else result["qualification_pass"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
