"""Frozen native check of the adapted external Drake controller.

No Drake import, fitting, MPPI search or new starts. Preparation supplies the
saved controller and local continuous-time certificate. Native results test
50Hz execution empirically, independently of that continuous-time claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

from src.gps.mppi_gps.acrobot_drake_reference import local_certificate, spong_control
from src.gps.mppi_gps.acrobot_energy import MODEL_SHA256


def run(config_path: Path, preparation_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if preparation["config_sha256"] != config_sha:
        raise ValueError("Preparation config binding differs")
    context_path = Path(config["attractor_context_config"])
    if (
        hashlib.sha256(context_path.read_bytes()).hexdigest()
        != config["attractor_context_config_sha256"]
    ):
        raise ValueError("Historical context config changed")
    context_config = json.loads(context_path.read_text())
    gain = np.asarray(preparation["K_physical"])
    value = np.asarray(preparation["P"])
    certificate = local_certificate(gain, value)
    if certificate != preparation["certificate"]:
        raise ValueError("Saved exact local certificate does not recompute")
    output.mkdir(parents=True, exist_ok=False)
    import h5py
    import mujoco

    from src.envs.acrobot import MODEL_PATH, Acrobot, balanced, tip_position
    from src.gps.mppi_gps.acrobot_teacher import _capture

    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("XML differs from bound model")
    expected = context_config["expected_environment"]
    runtime = {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "mujoco", "h5py")}
    runtime["python"] = platform.python_version()
    if runtime != expected:
        raise ValueError(f"Native runtime changed: {runtime!r}")
    roster = [
        (f"balance_{i}", "balance", row)
        for i, row in enumerate(context_config["lqr"]["historical_starts"])
    ]
    roster += [
        (f"swing_{i}", "hybrid", row)
        for i, row in enumerate(context_config["stages"]["swing-development"]["starts"])
    ]
    if len(roster) != 6:
        raise ValueError("Frozen native roster must contain exactly six starts")
    started = time.monotonic()
    completed_controls = 0
    rows = []
    status = "RUNNING"
    env = Acrobot(frame_skip=2, nthread=2)
    try:
        with h5py.File(output / "evidence.h5", "x") as raw:
            raw.attrs["config_sha256"] = config_sha
            raw.attrs["preparation_sha256"] = hashlib.sha256(
                preparation_path.read_bytes()
            ).hexdigest()
            for identifier, mode, case in roster:
                context = {
                    key: np.asarray(val, dtype=float)
                    for key, val in case["initial_context"].items()
                }
                mujoco.mj_setState(
                    env.model, env.data, context["integration"], mujoco.mjtState.mjSTATE_INTEGRATION
                )
                mujoco.mj_forward(env.model, env.data)
                mujoco.mj_setState(
                    env.model, env.data, context["integration"], mujoco.mjtState.mjSTATE_INTEGRATION
                )
                initial = _capture(mujoco, env)
                if any(initial[key].tobytes() != context[key].tobytes() for key in initial):
                    raise ValueError("Complete historical context restoration is not byte exact")
                group = raw.create_group(identifier)
                group.attrs["mode"] = mode
                group.attrs["reset_seed"] = case["reset_seed"]
                seen = {key: [val] for key, val in initial.items()}
                controls, costs, details = [], [], []
                try:
                    for _ in range(500):
                        if completed_controls >= config["limits"]["native_controls"]:
                            raise RuntimeError("Frozen native control budget exhausted")
                        if time.monotonic() - started > config["limits"]["native_seconds"]:
                            raise RuntimeError("Frozen native wall budget exhausted")
                        before = seen["states"][-1]
                        if not np.array_equal(before[1:], env._get_obs()):
                            raise ValueError("Pre-action state/observation mismatch")
                        level = np.inf if mode == "balance" else certificate["level"]["float"]
                        detail = spong_control(before, gain, value, level)
                        action = np.array([detail["action"]])
                        if not np.isfinite(action).all() or np.max(np.abs(action)) > 1:
                            raise ValueError("Invalid normalized control")
                        _, cost, done, _ = env.step(action)
                        completed_controls += 1
                        controls.append(action)
                        costs.append(cost)
                        details.append(detail)
                        current = _capture(mujoco, env)
                        for key, val in current.items():
                            seen[key].append(val)
                        if (
                            done
                            or not np.isfinite(cost)
                            or not all(np.isfinite(v).all() for v in current.values())
                        ):
                            raise ValueError("Invalid native physics")
                        if not np.isclose(
                            current["states"][0] - before[0], 0.02, rtol=0, atol=1e-10
                        ):
                            raise ValueError("Control time mismatch")
                finally:
                    for key, vals in seen.items():
                        group.create_dataset(key, data=np.asarray(vals))
                    group.create_dataset("actions", data=np.asarray(controls))
                    group.create_dataset("costs", data=np.asarray(costs))
                    for key in ("raw_torque", "energy", "balance", "local_value"):
                        group.create_dataset(key, data=np.asarray([x[key] for x in details]))
                    raw.flush()
                states = np.asarray(seen["states"])
                heights = tip_position(states[1:])[:, 2]
                row = {
                    "id": identifier,
                    "mode": mode,
                    "reset_seed": case["reset_seed"],
                    "controls": len(controls),
                    "height_hold_success": bool(balanced(states[1:], dt=0.02)),
                    "maximum_height": float(heights.max()),
                    "final2s_minimum_height": float(heights[-100:].min()),
                    "native_task_cost": float(np.sum(costs)),
                    "entered_continuous_region": bool(
                        any(d["local_value"] <= certificate["level"]["float"] for d in details)
                    ),
                    "clipped_fraction": float(np.mean([abs(d["raw_torque"]) > 2 for d in details])),
                }
                rows.append(row)
                with (output / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                if (
                    sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                    > config["limits"]["native_output_bytes"]
                ):
                    raise RuntimeError("Frozen output budget exhausted")
        status = "COMPLETED"
    finally:
        record = {
            "status": status,
            "runtime": runtime,
            "config_sha256": config_sha,
            "preparation_sha256": hashlib.sha256(preparation_path.read_bytes()).hexdigest(),
            "rows": rows,
            "counts": {
                "episodes_completed": len(rows),
                "native_controls": completed_controls,
                "physics_substeps": completed_controls * 2,
                "restores_attempted": len(rows) + (status != "COMPLETED"),
                "CARE_solves": 0,
                "SDP_solves": 0,
                "MPPI_queries": 0,
            },
            "wall_seconds": time.monotonic() - started,
            "certificate_scope": "continuous feedback only; these native results are empirical sample-and-hold/RK4 validation",
        }
        with (output / "results.json").open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.preparation, args.output)
