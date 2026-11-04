"""Capture and strictly revalidate the two fixed control sequences on the worker.

There is no search, policy inference or fitting. Native stepping, restoration,
candidate rollouts and durable accounting use the existing implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .provenance import sha, write
from .trajectory_pi_accounting import Work, directory_bytes
from .trajectory_pi_replay_validation import binding, bound, compare

MODULE = "src.gps.mppi_gps.trajectory_pi_native_baselines"


def verify_launch(path: Path) -> tuple[dict, dict, dict]:
    launch = json.loads(path.read_text())
    if launch.get("schema") != "trajectory-pi-native-baselines-launch-1":
        raise ValueError("Frozen worker-native validation launch required")
    config = json.loads(bound(launch["config"]).read_text())
    inputs = json.loads(bound(launch["bindings"]).read_text())
    source = json.loads(bound(launch["source_manifest"]).read_text())
    root = Path(__file__).resolve().parents[3]
    if root != Path(launch["source_directory"]) or source["schema"] != "trajectory-pi-source-1":
        raise ValueError("Execute the bound worker source")
    roster = json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(roster).hexdigest() != source["roster_sha256"]:
        raise ValueError("Source roster seal changed")
    for row in source["files"]:
        p = (root / row["path"]).resolve()
        if (
            not p.is_relative_to(root)
            or p.stat().st_size != row["bytes"]
            or sha(p) != row["sha256"]
        ):
            raise ValueError("Frozen worker source changed")
    if config["schema"] != "trajectory-pi-native-baselines-1" or config["horizon"] != 1000:
        raise ValueError("Fixed-sequence worker validation required")
    science = root / config["p1_config_relative"]
    science_config = json.loads(science.read_text())
    if sha(science) != config["p1_config_sha256"] or science_config["task"] != config["task"]:
        raise ValueError("Scientific task/configuration binding changed")
    previous = json.loads(bound(inputs["previous_prepared_manifest"]).read_text())
    if previous["config_sha256"] != inputs["previous_config"]["sha256"]:
        raise ValueError("Original P1.0 prepared-input binding changed")
    bound(inputs["previous_config"])
    lineage = science_config["lineage"]
    for name in ("previous_config", "previous_prepared_manifest"):
        if any(inputs[name][key] != lineage[name][key] for key in ("bytes", "sha256")):
            raise ValueError("P1.1 changed its original frozen parent-input lineage")
    origin = inputs["origin_replay"]
    for name, expected in lineage["origin_replay"]["bindings"].items():
        if any(origin[name][key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError("P1.1 origin-host proof differs from its frozen selection")
    result = json.loads(bound(origin["result"]).read_text())
    audit = json.loads(bound(origin["audit_receipt"]).read_text())
    findings = json.loads(bound(origin["audit_findings"]).read_text())
    external = json.loads(bound(origin["external_summary"]).read_text())
    if not (
        result["status"] == "ORIGIN_REPLAY_PASS"
        and audit["passed"] is True
        and audit["origin_replay_passed"] is True
        and findings["origin_replay_passed"] is True
        and findings["conclusion"]["linux_replay_passed"] is False
        and external["replay"]["exit_code"] == external["audit"]["exit_code"] == 0
    ):
        raise ValueError("Audited origin-host success is required before worker revalidation")
    environment = json.loads(bound(inputs["worker_environment"]).read_text())
    actual = dict(
        machine=platform.machine(),
        python=platform.python_version(),
        versions={
            name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "scipy", "h5py")
        },
    )
    if any(actual[key] != environment[key] for key in actual):
        raise ValueError("Worker runtime differs from its frozen environment")
    pins = config["expected_runtime"]
    if (
        actual["machine"] != "x86_64"
        or actual["python"] != pins["python"]
        or any(actual["versions"][key] != pins[key] for key in actual["versions"])
    ):
        raise ValueError("Pinned native CPU worker required")
    if binding(Path(sys.executable)) != environment["interpreter"]:
        raise ValueError("Execute the frozen worker interpreter")
    if launch["output_directory"] != config["output_directory"]:
        raise ValueError("Frozen worker validation directory changed")
    for item in [environment["interpreter"], *environment["binary_files"]]:
        bound(item)
    for row in inputs["episodes"]:
        bound(row["original_episode"])
        bound(row["original_prepared"])
        spec = science_config["teachers"][row["slot"]]
        previous_row = next(
            item for item in previous["files"] if item["path"] == spec["prepared_path"]
        )
        if any(
            row["original_episode"][key] != spec["episode"][key]
            or row["original_prepared"][key] != previous_row[key]
            for key in ("bytes", "sha256")
        ):
            raise ValueError(
                "Use only the unchanged selected recordings/contexts/control sequences"
            )
    if [row["slot"] for row in inputs["episodes"]] != [0, 1]:
        raise ValueError("Exactly the two already-selected source sequences are required")
    return launch, config, inputs


def run_validation(path: Path) -> dict:
    launch, config, inputs = verify_launch(path)
    out = Path(launch["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "launch.json", launch)
    work = Work(out, config["limits"])
    result = dict(
        schema="trajectory-pi-native-baselines-result-1",
        status="FAIL_STOPPED",
        config_sha256=launch["config"]["sha256"],
        p1_config_sha256=config["p1_config_sha256"],
        source_manifest_sha256=launch["source_manifest"]["sha256"],
        previous_prepared_manifest=inputs["previous_prepared_manifest"],
        origin_replay=inputs["origin_replay"],
        episodes=[],
        launch=binding(out / "launch.json"),
    )
    env = None
    try:
        import mujoco

        from .runtime import snapshot
        from .trajectory_pi_native import (
            _env,
            _restore,
            execute_trace,
            load_context,
            load_trace,
            same_bytes,
        )

        env = _env(dict(task=config["task"], physics_threads=config["physics_threads"]), work)
        work.call(
            "freeze_compiled_model",
            dict(compiled_model_saves=1),
            lambda: mujoco.mj_saveModel(env.model, str(out / "compiled.mjb"), None),
        )
        options = {
            name: np.asarray(getattr(env.model.opt, name)).tolist()
            for name in dir(env.model.opt)
            if not name.startswith("_")
            and isinstance(getattr(env.model.opt, name), (int, float, np.number, np.ndarray))
        }
        write(out / "model_options.json", options)
        write(
            out / "model_bindings.json",
            dict(
                compiled=binding(out / "compiled.mjb"),
                options=binding(out / "model_options.json"),
                dimensions=[9, 9, 6],
                spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
            ),
        )
        result["model_bindings"] = binding(out / "model_bindings.json")
        work.record("native_model_bound", model_bindings=result["model_bindings"])
        for spec in inputs["episodes"]:
            slot = spec["slot"]
            episode = {key: spec[key] for key in ("slot", "original_episode", "original_prepared")}
            result["episodes"].append(episode)
            context = load_context(bound(spec["original_prepared"]))
            with np.load(bound(spec["original_prepared"]), allow_pickle=False) as old:
                controls = old["actions"].copy()
            if controls.shape != (1000, 6) or controls.dtype != np.float64:
                raise ValueError("Original full float64 control sequence required")
            _restore(env, context, work)
            capture = out / f"capture_{slot}.h5"
            outcome = execute_trace(env, work, capture, 1000, actions=controls)
            episode.update(capture=binding(capture), outcome=outcome)
            raw = load_trace(capture)
            episode["initial_context_exact"] = all(
                same_bytes(raw[name][0], context[key])
                for name, key in (
                    ("states", "state"),
                    ("warmstarts", "warmstart"),
                    ("integrations", "integration"),
                )
            )
            episode["actions_exact"] = same_bytes(raw["actions"], controls)
            if not (
                outcome["walking"] and episode["initial_context_exact"] and episode["actions_exact"]
            ):
                raise ValueError(
                    "Worker-native fixed sequence lost walking or its original context/actions"
                )
            _restore(env, load_context(capture), work)
            replay = out / f"replay_{slot}.h5"
            replay_outcome = execute_trace(env, work, replay, 1000, actions=controls)
            repeated = load_trace(replay)
            comparison = compare(raw, repeated)
            episode.update(
                replay=binding(replay),
                replay_outcome=replay_outcome,
                replay_comparisons=comparison,
                native_replay_passed=all(row["passed"] for row in comparison.values())
                and same_bytes(raw["actions"], repeated["actions"])
                and replay_outcome["walking"]
                and replay_outcome["full_survival"],
            )
            if not episode["native_replay_passed"]:
                raise ValueError("Strict same-worker native replay failed")
            native_context = load_context(capture)
            _restore(env, native_context, work)
            states, costs, _ = env.batch_rollout(
                native_context["state"],
                controls[None],
                initial_warmstart=native_context["warmstart"],
            )
            candidate = out / f"candidate_{slot}.npz"
            unchanged = snapshot(env)
            with candidate.open("xb") as target:
                np.savez_compressed(
                    target,
                    states=states,
                    costs=costs,
                    actions=controls[None],
                    initial_state=native_context["state"],
                    initial_warmstart=native_context["warmstart"],
                    initial_integration=native_context["integration"],
                    final_state=unchanged["state"],
                    final_warmstart=unchanged["warmstart"],
                    final_integration=unchanged["integration"],
                )
            episode["candidate"] = binding(candidate)
            state_error = float(np.max(np.abs(states[0] - raw["states"][1:])))
            cost_error = float(abs(costs[0] - outcome["complete_cost"]))
            episode["candidate_state_maximum_error"] = (
                state_error if np.isfinite(state_error) else None
            )
            episode["candidate_cost_error"] = cost_error if np.isfinite(cost_error) else None
            episode["candidate_scores_finite"] = bool(
                np.isfinite(states).all() and np.isfinite(costs).all()
            )
            episode["single_candidate_passed"] = bool(
                np.allclose(states[0], raw["states"][1:], rtol=0, atol=1e-10)
                and np.isclose(costs[0], outcome["complete_cost"], rtol=0, atol=1e-9)
                and all(
                    same_bytes(unchanged[key], native_context[key])
                    for key in ("state", "warmstart", "integration")
                )
            )
            if not episode["single_candidate_passed"]:
                raise ValueError("Strict worker native/candidate-rollout equivalence failed")
        result["status"] = "PASS"
    except BaseException as error:
        result["error"] = repr(error)
        work.record("native_validation_stopped", error=repr(error))
    finally:
        if env is not None:
            env.close()
    receipt = work.close("COMPLETED" if result["status"] == "PASS" else "FAILED")
    result["work"] = receipt["counts"]
    write(out / "result.json", result)
    return result


def supervise(path: Path) -> dict:
    from .trajectory_pi_launch import capacity_passed, observe_capacity, run_phase, tree_usage

    entered = time.monotonic()
    launch, config, _ = verify_launch(path)
    out, source = Path(launch["output_directory"]), Path(launch["source_directory"])
    logs = out.parent / "validation_launch"
    (logs / "supervisor").mkdir(parents=True, exist_ok=False)
    initial_cpu = tree_usage(os.getpid())["cpu_seconds"]
    capacity = observe_capacity(out.parent)
    write(logs / "supervisor/capacity.json", capacity)
    if not capacity_passed(capacity, {}, main=False):
        raise RuntimeError("Worker-native validation capacity gate failed before physics")

    def sample() -> dict:
        return tree_usage(os.getpid())

    def guard() -> dict:
        value = sample()
        if (
            time.monotonic() - entered > 600
            or directory_bytes(out.parent) + 64 * 1024 > 64 * 1024**2
        ):
            raise RuntimeError("Worker-native validation total wall/output budget exhausted")
        if value["rss_bytes"] > config["limits"]["memory_bytes"]:
            raise RuntimeError("Worker-native validation aggregate RAM budget exhausted")
        return value

    guard.sample = sample
    replay = run_phase(
        "native-validation",
        [sys.executable, "-B", "-m", MODULE, "--launch", str(path), "--worker"],
        logs,
        source,
        540,
        guard,
        poll_seconds=0.25,
    )
    write(out / "external_receipt.json", replay)
    remaining = max(0.0, 600 - (time.monotonic() - entered))
    audit = run_phase(
        "audit",
        [
            sys.executable,
            "-B",
            "-m",
            "src.gps.mppi_gps.audits.trajectory_pi_native_baselines",
            "--run",
            str(out),
            "--out",
            str(out / "audit"),
            "--seconds",
            str(min(60.0, remaining)),
        ],
        logs,
        source,
        min(60.0, remaining),
        guard,
        poll_seconds=0.25,
    )
    summary = dict(
        schema="trajectory-pi-native-validation-external-1",
        validation=replay,
        audit=audit,
        wall_seconds=time.monotonic() - entered,
        cpu_seconds=sample()["cpu_seconds"] - initial_cpu,
        output_bytes_before_summary=directory_bytes(out.parent),
        no_retries=True,
    )
    write(out / "external_summary.json", summary)
    guard()
    if (
        replay["exit_code"] == audit["exit_code"] == 0
        and replay["limit_error"] is None
        and audit["limit_error"] is None
    ):
        result = json.loads((out / "result.json").read_text())
        audit_receipt = json.loads((out / "audit/receipt.json").read_text())
        findings = json.loads((out / "audit/findings.json").read_text())
        if (
            result["status"] == "PASS"
            and audit_receipt["passed"] is True
            and findings["native_validation_passed"] is True
        ):
            proof = dict(
                result,
                schema="trajectory-pi-native-validation-1",
                result=binding(out / "result.json"),
                receipt=binding(out / "receipt.json"),
                native_audit=dict(
                    receipt=binding(out / "audit/receipt.json"),
                    findings=binding(out / "audit/findings.json"),
                ),
                external_summary=binding(out / "external_summary.json"),
            )
            proof["admission_wall_seconds"] = time.monotonic() - entered
            write(out / "manifest.json", proof)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if args.worker:
        return int(run_validation(args.launch)["status"] != "PASS")
    result = supervise(args.launch)
    return int(
        any(
            result[key]["exit_code"] != 0 or result[key]["limit_error"] is not None
            for key in ("validation", "audit")
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
