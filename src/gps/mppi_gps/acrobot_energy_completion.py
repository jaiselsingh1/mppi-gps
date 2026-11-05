"""Storage-only completion using the original immutable Acrobot worker source.

Execute this file directly, outside the frozen export, with --frozen-source.
Only standard-library modules load before the original source path is selected.
Complete trials0/1 remain in the failed run; repeat partial trial2, compare its
paid429-control prefix, then run trial3. No controller implementation lives here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

LIMITS = dict(
    cpu_seconds=635,
    wall_seconds=1400,
    output_bytes=128 * 1024**2,
    rss_bytes=4 * 1024**3,
    native_controls=1000,
    forecast_controls=38553600,
    physics_substeps=77109200,
)
OLD_WORK = dict(
    native_controls=1429,
    model_queries=1436,
    forecast_controls=55142400,
    physics_substeps=110287658,
    verification_controls=0,
    restores=0,
)
FIELDS = ("states", "warmstarts", "integration", "actions", "costs", "pre_action_observations")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_protocol(protocol: dict[str, Any]) -> None:
    require(
        protocol.get("protocol") == "acrobot-near-upright-storage-completion"
        and protocol.get("version") == "1.1R",
        "Unknown storage completion protocol",
    )
    require(protocol.get("limits") == LIMITS, "Completion resource limits changed")
    require(
        protocol.get("retained_trial_indices") == [0, 1]
        and protocol.get("completion_trial_indices") == [2, 3],
        "Only incomplete trials2/3 may run",
    )
    require(
        protocol.get("expected_previous_controls") == 1429
        and protocol.get("expected_previous_queries") == 1436
        and protocol.get("partial_controls") == 429
        and protocol.get("prefix_atol") == 1e-10,
        "Paid-work or prefix comparison contract changed",
    )
    shared = protocol.get("shared_limits", {})
    require(
        shared
        == dict(
            cpu_seconds=11160,
            output_bytes=2 * 1024**3,
            original_physics_substeps=1510411416,
            amended_physics_substeps=1543589874,
            additional_repeated_physics_substeps=33178458,
        ),
        "The explicitly amended shared census must remain declared",
    )
    require(
        protocol.get("no_new_scientific_conditions") is True,
        "Storage completion must introduce no scientific condition",
    )


def validate_previous(result: dict[str, Any], original: dict[str, Any]) -> None:
    require(
        result.get("status") == "STOPPED_RESOURCES" and result.get("part") == "near-upright",
        "Only the preserved near-upright resource stop is eligible",
    )
    require(
        str(result.get("error", "")).startswith(
            "ResourceLimit: Frozen output_bytes budget exhausted:"
        ),
        "The original failure must be storage-only",
    )
    require(
        not any(result.get("unknown_partial_work", {"missing": 1}).values()),
        "Original work includes unknown partial calls",
    )
    for field, count in OLD_WORK.items():
        require(
            all(
                result.get("work", {}).get(field + "_" + ending) == count
                for ending in ("attempted", "completed")
            ),
            "Original paid census changed: " + field,
        )
    require(
        all(
            result["work"].get(key) == 0
            for key in ("network_forwards", "backward_calls", "optimizer_updates")
        ),
        "Original run includes learning work",
    )
    rows = result.get("trials", [])
    require(
        [row.get("trial_index") for row in rows] == [0, 1]
        and all(row.get("controls") == 500 and row.get("recipe_index") == 0 for row in rows),
        "Exactly the two completed original episodes must be retained",
    )
    for row in rows:
        start = original["parts"]["near-upright"]["starts"][row["trial_index"]]
        require(
            all(row[key] == start[key] for key in ("reset_seed", "planner_seed")),
            "A retained episode has a different seed binding",
        )
    limits = original["parts"]["near-upright"]["limits"]
    require(
        result["resources"]["cpu_seconds"] + LIMITS["cpu_seconds"] <= limits["cpu_seconds"],
        "Completion would reset the paid CPU budget",
    )
    require(
        result["resources"]["wall_seconds"] + LIMITS["wall_seconds"] <= limits["wall_seconds"],
        "Completion would reset the paid wall budget",
    )
    require(
        result["resources"]["rss_bytes"] < limits["rss_bytes"],
        "Original stop included a memory breach",
    )


def compare_prefix(old: Any, new: Any, controls: int = 429, atol: float = 1e-10) -> dict[str, Any]:
    import numpy as np

    old_initial, new_initial = np.asarray(old["states"][0]), np.asarray(new["states"][0])
    initial_equal = (
        old_initial.shape == new_initial.shape
        and old_initial.dtype == new_initial.dtype
        and old_initial.tobytes() == new_initial.tobytes()
    )
    result = dict(
        status="PASSED" if initial_equal else "FAILED",
        initial_state_byte_identical=initial_equal,
        controls=controls,
        previous_paid_queries=432,
        rtol=0,
        atol=atol,
        fields={},
    )
    for name in FIELDS:
        count = controls + 1 if name in ("states", "warmstarts", "integration") else controls
        a, b = np.asarray(old[name][:count]), np.asarray(new[name][:count])
        valid_shape = a.shape == b.shape and len(a) == count
        finite = valid_shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(np.abs(a - b), initial=0)) if finite else None
        valid = finite and error <= atol
        result["fields"][name] = dict(
            rows=count,
            shape=list(a.shape),
            dtype=str(a.dtype),
            maximum_absolute_error=error,
            passed=bool(valid),
            previous_array_raw_bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            completion_array_raw_bytes_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
        )
        if not valid:
            result["status"] = "FAILED"
    return result


def load_frozen(source: Path, protocol: dict[str, Any]):
    require(
        not any(name == "src" or name.startswith("src.") for name in sys.modules),
        "Execute the wrapper directly in a fresh Python process, not as an imported src module",
    )
    source = source.resolve()
    require(
        digest(source / "source_manifest.json") == protocol["frozen_source_manifest_sha256"],
        "Original frozen source manifest changed",
    )
    manifest = json.loads((source / "source_manifest.json").read_text())
    require(
        manifest.get("source_commit") == protocol["frozen_worker_source_commit"],
        "The wrapper must use the original05fe source",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    module = importlib.import_module("src.gps.mppi_gps.acrobot_energy_probe")
    require(
        Path(module.__file__).resolve() == source / "src/gps/mppi_gps/acrobot_energy_probe.py",
        "Recovery imported a different worker implementation",
    )
    return module


def complete(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    require(
        digest(Path(__file__)) == args.wrapper_sha256,
        "Separately frozen completion wrapper changed",
    )
    probe = load_frozen(args.frozen_source, protocol)
    import h5py
    import numpy as np

    original_path = args.frozen_source / protocol["original_config_path"]
    require(
        digest(original_path) == protocol["original_config_sha256"],
        "Original configuration changed",
    )
    original = json.loads(original_path.read_text())
    require(
        digest(args.previous / "result.json") == protocol["previous_result_sha256"]
        and digest(args.previous / "evidence.h5") == protocol["previous_evidence_sha256"],
        "The preserved original failure changed",
    )
    previous = json.loads((args.previous / "result.json").read_text())
    validate_previous(previous, original)
    for name, binding in previous.get("evidence", {}).items():
        target = (args.previous / name).resolve()
        require(
            target.is_relative_to(args.previous.resolve())
            and target.is_file()
            and target.stat().st_size == binding["bytes"]
            and digest(target) == binding["sha256"],
            "Preserved original evidence changed: " + name,
        )

    args.out.mkdir(parents=True, exist_ok=False)
    budget = probe.Budget(LIMITS, args.out)
    result = dict(
        protocol="acrobot-near-upright-storage-completion",
        version="1.1R",
        part="near-upright-completion",
        status="STARTING",
        trials=[],
        prior_work=previous["work"],
        prior_resources=previous["resources"],
        original_failure_preserved=True,
        retained_original_trial_indices=[0, 1],
        completed_trial_indices=[],
        partial_repeat=dict(
            native_controls=429,
            model_queries=432,
            forecast_controls=16588800,
            physics_substeps=33178458,
        ),
        shared_limits=protocol["shared_limits"],
        teacher_qualified=False,
        learning_or_loop_launched=False,
    )
    try:
        with probe.resource_timers(LIMITS, budget):
            _, budget = probe._setup(original_path, args.out, budget, 2)
            inputs = json.loads((args.out / "inputs.json").read_text())
            old_inputs = json.loads((args.previous / "inputs.json").read_text())
            for key in ("config_sha256", "files", "frozen_source_manifest"):
                require(
                    inputs.get(key) == old_inputs.get(key),
                    "Original source binding changed: " + key,
                )
            environment = json.loads((args.out / "environment.json").read_text())
            require(
                environment == json.loads((args.previous / "environment.json").read_text()),
                "Native runtime differs from the failed worker",
            )
            require(
                digest(args.preflight / "result.json") == protocol["preflight_result_sha256"],
                "Shared preflight result changed",
            )
            result["preflight_binding"] = probe.verify_preflight(
                args.preflight, original_path, inputs
            )
            probe.write(args.out / "amendment.json", protocol)
            shutil.copy2(Path(__file__), args.out / "wrapper_source.py")
            probe.write(
                args.out / "recovery_bindings.json",
                dict(
                    wrapper_sha256=args.wrapper_sha256,
                    frozen_source_manifest_sha256=protocol["frozen_source_manifest_sha256"],
                    previous_result_sha256=protocol["previous_result_sha256"],
                    previous_evidence_sha256=protocol["previous_evidence_sha256"],
                    protocol_sha256=digest(args.protocol),
                    remaining_limits=LIMITS,
                ),
            )
            with (
                h5py.File(args.previous / "evidence.h5", "r") as old,
                h5py.File(args.out / "evidence.h5", "x") as new,
            ):
                for index in (0, 1, 2):
                    group = old[f"trials/variant_0/trial_{index}"]
                    n = 500 if index < 2 else 429
                    require(
                        len(group["actions"]) == n and len(group["states"]) == n + 1,
                        "Preserved native episode/prefix lengths changed",
                    )
                    require(
                        group.attrs["status"] == ("COMPLETED" if index < 2 else "STOPPED"),
                        "Preserved episode status changed",
                    )
                    for name in FIELDS:
                        count = n + 1 if name in ("states", "warmstarts", "integration") else n
                        require(
                            name in group
                            and len(group[name]) == count
                            and np.isfinite(group[name][:]).all(),
                            "Preserved complete context/prefix missing or nonfinite: " + name,
                        )
                    initial = np.asarray(
                        original["parts"]["near-upright"]["starts"][index]["initial_state"],
                        dtype=np.float64,
                    )
                    require(
                        group["states"][0].tobytes() == initial.tobytes(),
                        "Preserved restart context changed",
                    )

                import mujoco

                from src.mppi.mppi import MPPI

                new.attrs["integration_spec"] = old.attrs["integration_spec"]
                env = probe._new_env(original["parts"]["near-upright"]["variants"][0], 2)
                try:
                    for index in (2, 3):
                        start = original["parts"]["near-upright"]["starts"][index]
                        group = new.create_group(f"trials/variant_0/trial_{index}")
                        row = probe._run_trial(
                            mujoco,
                            env,
                            MPPI,
                            group,
                            budget,
                            original["parts"]["near-upright"]["variants"][0],
                            {key: start[key] for key in ("reset_seed", "planner_seed")},
                            "near-upright",
                            0,
                            index,
                            args.out,
                            controls=500,
                            initial_state=np.asarray(start["initial_state"], dtype=float),
                            metrics_fn=probe.native_metrics,
                            bank_callback=probe.record_bank,
                        )
                        result["trials"].append(row)
                        result["completed_trial_indices"].append(index)
                        if index == 2:
                            check = compare_prefix(old["trials/variant_0/trial_2"], group)
                            probe.write(args.out / "prefix_check.json", check)
                            require(
                                check["status"] == "PASSED",
                                "Repeated original prefix mismatched; trial3 must not run",
                            )
                finally:
                    env.close()
            result.update(
                status="COMPLETED_NEAR_UPRIGHT_RECOVERY",
                combined_near_upright_trials=previous["trials"] + result["trials"],
                successes=sum(row["success"] for row in previous["trials"] + result["trials"]),
            )
    except probe.ResourceLimit as error:
        result.update(status="STOPPED_RESOURCES", error=f"{type(error).__name__}: {error}")
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="STOPPED_INVALID_EXECUTION", error=f"{type(error).__name__}: {error}")
    result["cumulative_paid_work"] = {
        key: value + getattr(budget.work, key) for key, value in previous["work"].items()
    }
    return probe._finalize(args.out, result, budget)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen-source", "protocol", "previous", "preflight", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wrapper-sha256", required=True)
    args = parser.parse_args(argv)
    result = complete(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] == "COMPLETED_NEAR_UPRIGHT_RECOVERY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
