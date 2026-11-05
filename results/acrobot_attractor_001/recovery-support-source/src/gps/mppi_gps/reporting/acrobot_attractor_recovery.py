"""Recover a failed final metadata write using complete preserved LQR evidence.

The original native failure is immutable. This saved-only verifier repeats no
physics and changes no controller: it checks every declared trace before deriving
the missing radius/confirmation fields into a new directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.gps.mppi_gps.reporting import acrobot_attractor as audit


def validate_failure(result: dict, protocol: dict) -> None:
    audit.require(
        result.get("status") == "STOPPED_INVALID_EXECUTION", "Preserve the native failure"
    )
    audit.require(result.get("stage") == "lqr", "Only the LQR metadata finalization is eligible")
    audit.require(
        result.get("error") == protocol["original_error_exact"], "Exact frozen finalization failure"
    )
    audit.require(
        str(result["error"]).startswith("FileExistsError:"), "Not a physics or resource failure"
    )
    audit.require(
        not any(result["unknown_partial_work"].values()), "No unknown partial native work"
    )


def derive_packet(initial: dict, rows: list[dict], spec: dict) -> dict:
    """Keep every original numerical field; derive only the intended final fields."""
    factor = None
    for value in spec["shell_factors"]:
        found = [r for r in rows if r.get("kind") == "shell" and r.get("factor") == value]
        audit.require(len(found) == 24, "All24 directions per declared shell")
        if not all(r["capture"] for r in found):
            break
        factor = value
    audit.require(factor is not None, "A completed confirmation roster requires a nonempty radius")
    rho = factor * initial["unsaturated_rho"]
    confirmations = [r for r in rows if r["kind"].startswith("confirm_")]
    audit.require(len(confirmations) == 128, "All128 frozen confirmations must be present")
    p = np.asarray(initial["P"])
    packet = copy.deepcopy(initial)
    packet.update(
        rho=rho,
        selected_rho=rho,
        selected_shell_factor=factor,
        coordinate_projections=np.sqrt(rho * np.diag(np.linalg.inv(p))).tolist(),
        join_validation_passed=all(r["capture"] for r in confirmations),
        shells=[
            dict(
                factor=f, captures=sum(r["capture"] for r in rows if r.get("factor") == f), count=24
            )
            for f in spec["shell_factors"]
        ],
    )
    audit.require(
        all(packet[k] == v for k, v in initial.items()), "Initial controller packet is unchanged"
    )
    return packet


def verify_saved(
    review: audit.Review, config: Path, protocol: dict
) -> tuple[dict, dict, list[dict], dict]:
    import h5py

    run = review.base / "lqr"
    for name, binding in protocol["original_files"].items():
        expected = binding["sha256"]
        review.check((run / name).stat().st_size == binding["bytes"], "Frozen input bytes: " + name)
        review.check(
            audit.digest(review.consume(run / name)) == expected, "Frozen failed evidence: " + name
        )
    review.check(
        audit.digest(config) == protocol["original_files"]["config.json"]["sha256"],
        "Unchanged scientific configuration",
    )
    review.check(
        audit.digest(review.base / "source/source_manifest.json")
        == protocol["original_source_manifest_sha256"],
        "Original immutable worker source",
    )
    review.check(
        review.load(review.base / "source/source_manifest.json")["source_commit"]
        == protocol["original_worker_commit"],
        "Original worker commit",
    )
    cfg = review.load(config)
    result = audit.check_sources(review, run, config, cfg)
    validate_failure(result, protocol)
    initial = review.load(run / "lqr.json")
    spec = cfg["lqr"]
    rows, actual = [], {}
    with h5py.File(run / "evidence.h5", "r") as raw:
        review.check(raw.attrs["status"] == result["status"], "Original H5 failure status retained")
        lin = raw["linearization"]
        native = audit.native_jacobians(
            lin["probe_states"][:], lin["probe_actions"][:], lin["next_states"][:]
        )
        math = audit.riccati_measurements(initial, native)
        held = np.random.Generator(np.random.PCG64(spec["heldout_linearization_seed"])).normal(
            0, spec["heldout_scale"], (16, 5)
        )
        review.same(
            lin["probe_states"][21:, 1:],
            held[:, :4],
            "Private native held-out state bindings",
            exact=True,
        )
        review.same(
            lin["probe_actions"][21:],
            held[:, 4:],
            "Private native held-out action bindings",
            exact=True,
        )
        for case_id, group in raw["trials"].items():
            case = json.loads(group.attrs["case_json"])
            measured, states, _ = audit.check_trace(
                review, group, "dense", np.asarray(initial["K"])
            )
            review.check(
                group.attrs["status"] == "COMPLETED" and len(states) == 501,
                "Complete500-control saved LQR case",
            )
            review.check(case["id"] == case_id, "Native finite-case identity")
            review.check(
                bool(group.attrs["captured"]) == measured["capture"], "Independent capture outcome"
            )
            review.check(
                bool(group.attrs["height_success"]) == measured["height_success"],
                "Separate native height outcome",
            )
            review.same(states[0], case["state"], "Saved case initial state", exact=True)
            row = {k: v for k, v in case.items() if k != "initial_context"} | measured
            rows.append(row)
            actual[case_id] = (row, states[0])
            if case["kind"] == "historical":
                for key, value in case["initial_context"].items():
                    expected = np.asarray(value, dtype=float)
                    review.check(
                        group[key][0].tobytes() == expected.tobytes(),
                        "Byte-exact original context: " + key,
                    )
        calibration = audit.lqr_roster(spec, initial)
        for expected in calibration:
            row, state = actual[expected["id"]]
            review.check(
                row["kind"] == expected["kind"] and row.get("factor") == expected.get("factor"),
                "Declared calibration-case classification",
            )
            review.same(state, expected["state"], "Independently reconstructed calibration input")
        selected = audit.selected_shell(rows, initial["unsaturated_rho"], spec["shell_factors"])
        expected = audit.lqr_roster(spec, initial, selected)
        review.check(
            len(rows) == 362 and set(actual) == {r["id"] for r in expected},
            "Exactly362 complete frozen inputs",
        )
        for case in expected:
            review.same(
                actual[case["id"]][1],
                case["state"],
                "Private confirmation direction/radius and seed",
                exact=True,
            )
        replay, original = raw["replay"], raw["trials/historical_0"]
        for key in ("states", "warmstarts", "integration"):
            review.same(
                replay[key][:], original[key][:13], "Preserved12-control native replay: " + key
            )
            review.check(
                replay[key][0].tobytes() == original[key][0].tobytes(),
                "Byte-exact replay restoration: " + key,
            )
        for key in ("actions", "costs"):
            review.same(replay[key][:], original[key][:12], "Preserved native replay: " + key)
    journal = [json.loads(line) for line in (run / "trials.jsonl").read_text().splitlines()]
    review.check(
        len(journal) == len(rows) and {r["id"] for r in journal} == set(actual),
        "Complete original native journal",
    )
    for row in journal:
        recomputed = actual[row["id"]][0]
        review.check(
            row["captured"] == recomputed["capture"]
            and row["height_success"] == recomputed["height_success"],
            "Native journal agrees with recomputed outcomes",
        )
    native_controls = sum(row["controls"] for row in rows)
    expected_work = dict(
        native_controls=native_controls,
        verification_controls=49,
        forecast_controls=0,
        model_queries=0,
        physics_substeps=2 * (native_controls + 49),
        restores=1,
    )
    audit.check_work(review, result, expected_work, spec["limits"])
    packet = derive_packet(initial, rows, spec)
    return packet, result, rows, dict(math=math, recomputed_inherited_work=expected_work)


def recover(
    base: Path, config: Path, protocol_path: Path, previous: Path, out: Path
) -> dict[str, Any]:
    base, config, out = base.resolve(), config.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(protocol_path.read_text())
    result: dict[str, Any] = dict(
        status="STARTING",
        original_failure_preserved=True,
        new_physics_controls=0,
        new_network_forwards=0,
        new_fitting_updates=0,
    )
    review = audit.Review(base, 20, previous.resolve())
    try:
        review.check(
            protocol["protocol"] == "acrobot-attractor-lock-finalization-recovery"
            and protocol["version"] == "R1",
            "Frozen recovery version",
        )
        review.check(
            protocol["limits"]
            == dict(
                cpu_seconds=20,
                wall_seconds=120,
                output_bytes=8 * 1024**2,
                native_controls=0,
                physics_substeps=0,
                network_forwards=0,
                optimizer_updates=0,
            ),
            "Bounded saved-only recovery",
        )
        review.check(
            audit.digest(Path(__file__)) == protocol["helper_sha256"]
            and audit.digest(Path(audit.__file__)) == protocol["reviewer_sha256"],
            "Frozen independent verifier sources",
        )
        packet, original, rows, details = verify_saved(review, config, protocol)
        for name, binding in protocol["original_files"].items():
            review.check(
                audit.digest(base / "lqr" / name) == binding["sha256"],
                "Original evidence remains byte unchanged: " + name,
            )
        result.update(
            protocol="acrobot-transient-attractor",
            version="1.0",
            stage="lqr",
            status="VERIFIED_EMPIRICAL_JOIN_REGION"
            if packet["join_validation_passed"]
            else "FAILED_BASIN_CONFIRMATION",
            status_scope="Saved-only finalization recovery; original native execution status remains failed",
            original_status=original["status"],
            original_error=original["error"],
            lqr=packet,
            join_validation_passed=packet["join_validation_passed"],
            inherited_work=original["work"],
            work={key: 0 for key in original["work"]},
            historical_captures=sum(r["capture"] for r in rows if r["kind"] == "historical"),
            confirmation_captures=sum(
                r["capture"] for r in rows if r["kind"].startswith("confirm_")
            ),
            recovery_protocol_sha256=audit.digest(protocol_path),
            original_bindings=protocol["original_files"],
            **details,
        )
        audit.write_csv(out / "episodes.csv", rows)
        audit.write_json(
            out / "verification.json",
            dict(
                checks=review.checks,
                inputs=review.inputs,
                details=details,
                no_nonlinear_basin_certificate=True,
            ),
        )
        shutil.copyfile(config, out / "config.json")
        audit.write_json(out / "lqr.json", packet)
        review.check(
            sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) + 32768 < 8 * 1024**2,
            "Recovery output allocation",
        )
    except Exception as error:
        result = dict(
            status="FAILED_OR_INCOMPLETE",
            error=f"{type(error).__name__}: {error}",
            original_failure_preserved=True,
            new_physics_controls=0,
            new_network_forwards=0,
            new_fitting_updates=0,
        )
    result.update(recovery_cpu_seconds=time.process_time() - review.started, checks=review.checks)
    audit.write_json(out / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "config", "protocol", "previous", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = recover(args.base, args.config, args.protocol, args.previous, args.out)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "error",
                    "checks",
                    "historical_captures",
                    "confirmation_captures",
                    "recovery_cpu_seconds",
                )
                if k in result
            },
            indent=2,
        )
    )
    return (
        0
        if result.get("status") in {"VERIFIED_EMPIRICAL_JOIN_REGION", "FAILED_BASIN_CONFIRMATION"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
