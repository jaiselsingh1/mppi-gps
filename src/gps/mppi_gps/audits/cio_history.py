"""Independent, bounded audit of saved CIO history scalars and trace exports.

Only the standard library is used. Historical pickle/checkpoint contents are
hashed, never decoded. Trace export headers and short prefixes are checked
against report metadata; this is not simulator replay or source equivalence.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import resource
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

SCHEMA = "walker-cio-history-audit-1"
ZERO_WORK = {
    "native_controls": 0,
    "physics_substeps": 0,
    "neural_rows": 0,
    "backward_rows": 0,
    "optimizer_steps": 0,
    "mppi_queries": 0,
    "checkpoint_loads": 0,
    "pickle_decodes": 0,
}


def strict_json(payload: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError("Nonfinite JSON: " + value)

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=reject)


def numeric_fields(value: dict, prefix: str = "") -> dict:
    """Recursively enumerate finite scalar observations, excluding booleans."""
    result = {}
    for key, item in value.items():
        name = prefix + key
        if type(item) is dict:
            result.update(numeric_fields(item, name + "."))
        elif type(item) in (int, float):
            if not math.isfinite(item):
                raise ValueError("Nonfinite scalar: " + name)
            result[name] = item
    return result


class Audit:
    """Finite read/hash/check census, without historical runtime imports."""

    def __init__(self, limits: dict):
        self.limits = limits
        self.started = time.monotonic()
        self.cpu_started = time.process_time()
        self.checks = 0
        self.read_bytes = 0
        self.output_bytes = 0
        self.files: dict[str, dict] = {}
        self.trace_prefix_values = 0

    def need(self, condition: Any, message: str) -> None:
        self.checks += 1
        if not condition:
            raise ValueError(message)

    def check(self) -> None:
        self.need(time.monotonic() - self.started <= self.limits["wall_seconds"], "Wall budget")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == "darwin" else 1024
        self.need(peak <= self.limits["memory_bytes"], "Memory budget")

    def read(self, path: Path, expected: dict | None = None, *, retain: bool = True) -> bytes:
        self.check()
        self.need(
            path.is_absolute() and path.is_file() and not path.is_symlink(), "Regular absolute file"
        )
        before = path.stat()
        self.need(before.st_size <= self.limits["read_bytes"] - self.read_bytes, "Read budget")
        digest, pieces = hashlib.sha256(), []
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                self.read_bytes += len(block)
                digest.update(block)
                if retain:
                    pieces.append(block)
                self.check()
        after = path.stat()
        self.need(
            (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Changed file",
        )
        binding = {"bytes": before.st_size, "sha256": digest.hexdigest()}
        if expected is not None:
            self.need(
                binding == {key: expected[key] for key in binding}, "File binding: " + str(path)
            )
        if str(path) in self.files:
            self.need(self.files[str(path)] == binding, "Repeated binding: " + str(path))
        self.files[str(path)] = binding
        return b"".join(pieces)

    def close_float(self, actual: float, expected: float, message: str) -> None:
        self.need(
            math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            message,
        )

    def receipt(self) -> dict:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            **ZERO_WORK,
            "wall_seconds": time.monotonic() - self.started,
            "cpu_seconds": time.process_time() - self.cpu_started,
            "peak_process_rss_bytes": peak * (1 if sys.platform == "darwin" else 1024),
            "logical_read_and_hash_bytes": self.read_bytes,
            "unique_bound_files": len(self.files),
            "unique_bound_bytes": sum(item["bytes"] for item in self.files.values()),
            "trace_prefix_values": self.trace_prefix_values,
            "output_bytes": self.output_bytes,
            "physical_disk_read_bytes": None,
            "peak_rss_scope": "Process-lifetime high-water, including startup",
        }


def _npy(stream: Any, audit: Audit) -> tuple[dict, int]:
    audit.need(stream.read(6) == b"\x93NUMPY", "NPY magic")
    version = stream.read(2)
    audit.need(version in (b"\x01\x00", b"\x02\x00"), "NPY version")
    size = struct.unpack(
        "<H" if version[0] == 1 else "<I", stream.read(2 if version[0] == 1 else 4)
    )[0]
    audit.need(size <= 4096, "NPY header ceiling")
    header = ast.literal_eval(stream.read(size).decode("latin1"))
    audit.need(set(header) == {"descr", "fortran_order", "shape"}, "NPY fields")
    audit.need(header["fortran_order"] is False, "C-order trace export")
    audit.need(header["descr"] in ("<f8", "<i8"), "Trace dtype")
    audit.need(
        type(header["shape"]) is tuple and all(type(v) is int and v >= 0 for v in header["shape"]),
        "Trace shape",
    )
    return header, stream.tell()


def check_trace_export(
    payload: bytes, traces: list[dict], audit: Audit, sample_traces: int
) -> dict:
    """Check all offsets and up to three prefix rows of the first N traces."""
    headers = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = {"offsets.npy", "elapsed_seconds.npy", "joint_angles_radians.npy"}
        audit.need(set(archive.namelist()) == names, "Trace archive members")
        for info in archive.infolist():
            audit.need(info.file_size <= 16 * 1024**2, "Uncompressed trace ceiling")
        with archive.open("offsets.npy") as stream:
            header, _ = _npy(stream, audit)
            audit.need(
                header == {"descr": "<i8", "fortran_order": False, "shape": (len(traces) + 1,)},
                "Offsets header",
            )
            values = stream.read(8 * (len(traces) + 1))
            audit.need(len(values) == 8 * (len(traces) + 1), "Offsets length")
            offsets = list(struct.unpack("<" + "q" * (len(traces) + 1), values))
        expected = [0]
        for trace in traces:
            expected.append(expected[-1] + trace["controls"] + 1)
        audit.need(offsets == expected, "All N+1 trace offsets")
        for name, shape in [
            ("elapsed_seconds", (offsets[-1],)),
            ("joint_angles_radians", (offsets[-1], 6)),
        ]:
            with archive.open(name + ".npy") as stream:
                header, start = _npy(stream, audit)
                audit.need(
                    header == {"descr": "<f8", "fortran_order": False, "shape": shape},
                    "Trace array header",
                )
                headers[name] = header
                width = 6 if name == "joint_angles_radians" else 1
                for index, trace in enumerate(traces[:sample_traces]):
                    count = min(3, trace["controls"] + 1)
                    stream.seek(start + offsets[index] * width * 8)
                    data = stream.read(count * width * 8)
                    audit.need(len(data) == count * width * 8, "Trace prefix length")
                    values = struct.unpack("<" + "d" * (count * width), data)
                    audit.trace_prefix_values += len(values)
                    audit.need(all(math.isfinite(v) for v in values), "Finite trace prefix")
                    if width == 6:
                        audit.need(
                            list(values[:6]) == trace["start_state"][4:10],
                            "Angle prefix/start-state join",
                        )
                    else:
                        audit.need(
                            values[0] == 0 and all(b > a for a, b in zip(values, values[1:])),
                            "Elapsed prefix",
                        )
    return {
        "traces": len(traces),
        "states": offsets[-1],
        "sampled_traces": min(sample_traces, len(traces)),
        "headers": headers,
    }


def verify(config: dict, check: Audit) -> dict:
    for binding in config["auditor_sources"]:
        check.read(Path(binding["path"]), binding, retain=False)
    report = Path(config["report_directory"])
    frozen = Path(config["source_directory"])
    check.need(report.is_absolute() and frozen.is_absolute(), "Explicit machine paths")
    actual_names = {str(p.relative_to(report)) for p in report.rglob("*") if p.is_file()}
    check.need(actual_names == set(config["report_files"]), "Exact report file roster")
    payloads = {}
    for name, binding in config["report_files"].items():
        payloads[name] = check.read(report / name, binding, retain=Path(name).suffix != ".png")
    original = strict_json(payloads["config.json"])
    result = strict_json(payloads["result.json"])
    accounting = strict_json(payloads["accounting.json"])
    inventory = strict_json(payloads["inventory.json"])
    check.need(result["status"] == "SAVED_ONLY_REVIEW", "Saved-only report status")
    check.need(
        result["config_sha256"] == hashlib.sha256(payloads["config.json"]).hexdigest(),
        "Report configuration hash",
    )
    manifest = strict_json(check.read(frozen / "manifest.json", config["source_manifest"]))
    actual_source = {str(p.relative_to(frozen)) for p in frozen.rglob("*") if p.is_file()}
    check.need(
        actual_source == set(manifest["files"]) | {"manifest.json"}, "Exact frozen source roster"
    )
    for name, binding in manifest["files"].items():
        check.read(frozen / name, binding, retain=False)
    launch = strict_json(check.read(Path(config["launch"]["path"]), config["launch"]))
    exit_status = strict_json(check.read(Path(config["exit"]["path"]), config["exit"]))
    check.need(
        launch["source_manifest_sha256"] == config["source_manifest"]["sha256"],
        "Launch/source binding",
    )
    check.need(
        launch["science_calls_allowed"] == manifest["science_calls_allowed"] == 0,
        "Zero-work launch",
    )
    check.need(
        exit_status["exit_code"] == 0 and exit_status["timeout"] is False,
        "Successful renderer exit",
    )
    report_runs = {run["id"]: run for run in result["runs"]}
    check.need(
        len(report_runs) == len(result["runs"]) == len(original["runs"]), "Distinct run roster"
    )
    expected_csv, summaries, source_bytes = [], {}, 0
    for run in original["runs"]:
        key, inputs = run["id"], run["inputs"]
        saved, decoded = report_runs[key], {}
        for kind, binding in inputs.items():
            payload = check.read(Path(binding["path"]), binding, retain=kind != "teachers")
            source_bytes += binding["bytes"]
            if kind != "teachers":
                decoded[kind] = payload
        metadata = strict_json(decoded["run"])
        rows = [strict_json(line) for line in decoded["metrics"].splitlines() if line.strip()]
        rounds = [row["round"] for row in rows]
        check.need(
            rounds == list(range(len(rows))) and len(rows) == config["round_counts"][key],
            "Complete reported round range",
        )
        check.need(
            saved["metrics"] == rows and saved["reported_rounds"] == rounds,
            "Every original scalar row preserved",
        )
        check.need(
            saved["provenance"] == metadata and saved["source_inputs"] == inputs,
            "Original provenance/bindings preserved",
        )
        check.need(saved["final_teacher_round"] == rounds[-1], "Final teacher round")
        for row in rows:
            expected_csv.append({"run": key, **numeric_fields(row)})
        files = [
            {"path": str(p), "bytes": p.stat().st_size, "suffix": p.suffix}
            for p in sorted(Path(run["directory"]).iterdir())
            if p.is_file()
        ]
        check.need(inventory[key] == files, "Current source availability inventory")
        for field in [
            "student_joint_traces",
            "student_noise_joint_traces",
            "earlier_teacher_joint_traces",
        ]:
            check.need(saved[field]["available"] is False, "Missing evidence remains missing")
        traces = saved["final_teacher_traces"]
        check.need(len(traces) == config["trace_counts"][key], "Retained trace roster")
        check.need(len({t["id"] for t in traces}) == len(traces), "Distinct retained trace IDs")
        for trace in traces:
            check.need(
                trace["round"] == rounds[-1]
                and 0 < trace["controls"] <= trace["intended_controls"] <= 1000,
                "Retained trace metadata",
            )
            check.need(
                len(trace["start_state"]) == 19 and trace["reset_seed"] is None,
                "State/reset metadata",
            )
        trace_result = check_trace_export(
            payloads[key + "_teacher_angles.npz"], traces, check, config["prefix_traces"]
        )
        counters = {
            field: sum(row.get(field, 0) for row in rows)
            for field in ["executed_controls", "mppi_passes", "simulated_controls", "seconds"]
        }
        dev_controls = sum(
            row.get("dev", {}).get("mean_steps", 0) * metadata["settings"].get("dev_episodes", 0)
            for row in rows
        )
        check.close_float(dev_controls, round(dev_controls), "Integer saved development controls")
        summary = {
            "rounds": rounds,
            "recorded_loop_counters": counters,
            "development_controls_from_saved_means": round(dev_controls),
            "trace_export": trace_result,
        }
        if len(rows) > 1 and "gap_rms_after" in rows[1]:
            summary["gap_change_fraction_first_to_last_update"] = (
                rows[-1]["gap_rms_after"] / rows[1]["gap_rms_after"] - 1
            )
        summary["teacher_cost_change_fraction_round0_to_last"] = (
            rows[-1]["task_cost_per_control"] / rows[0]["task_cost_per_control"] - 1
        )
        fresh = strict_json(decoded["fresh"]) if "fresh" in decoded else None
        check.need(saved["standalone_fresh"] == fresh, "Fresh evaluation copied exactly")
        if fresh is not None:
            episodes = fresh["episodes"]
            recomputed = {
                "episodes": len(episodes),
                "survived": sum(episode["survived"] for episode in episodes),
                "controls": sum(episode["steps"] for episode in episodes),
            }
            check.need(fresh["summary"]["episodes"] == len(episodes), "Fresh episode count")
            check.close_float(
                fresh["summary"]["survival_rate"],
                recomputed["survived"] / len(episodes),
                "Fresh survival",
            )
            check.close_float(
                fresh["summary"]["mean_steps"],
                recomputed["controls"] / len(episodes),
                "Fresh controls",
            )
            check.close_float(
                fresh["summary"]["velocity_mae"],
                math.fsum(e["velocity_mae"] for e in episodes) / len(episodes),
                "Fresh MAE",
            )
            summary["fresh_evaluation"] = recomputed
        summaries[key] = summary
    reader = csv.DictReader(io.StringIO(payloads["round_metrics.csv"].decode()))
    csv_rows = list(reader)
    check.need(
        len(csv_rows) == len(expected_csv) == sum(config["round_counts"].values()),
        "All CSV round rows",
    )
    columns = set().union(*(row.keys() for row in expected_csv))
    check.need(set(reader.fieldnames or []) == columns, "All CSV numeric fields")
    for actual, expected in zip(csv_rows, expected_csv, strict=True):
        check.need(actual["run"] == expected["run"], "CSV run join")
        for column in columns - {"run"}:
            check.need(
                actual[column] == (str(expected[column]) if column in expected else ""),
                "CSV scalar/missingness: " + column,
            )
    check.need(
        accounting["source_hash_bytes"] == accounting["logical_source_read_bytes"] == source_bytes,
        "Renderer source-read census",
    )
    check.need(original["preparation_source_hash_bytes"] == source_bytes, "Preparation hash census")
    check.need(
        accounting["output_bytes"]
        == sum(value["bytes"] for value in config["report_files"].values()),
        "Self-inclusive renderer output bytes",
    )
    for field in ZERO_WORK.keys() - {"pickle_decodes"}:
        check.need(
            accounting[field] == result["accounting"][field] == 0, "Renderer zero-work counter"
        )
    for field, limit in [
        ("wall_seconds", "wall_seconds"),
        ("peak_process_rss_bytes", "memory_bytes"),
        ("output_bytes", "output_bytes"),
    ]:
        check.need(accounting[field] <= original["limits"][limit], "Renderer limit: " + field)
    check.check()
    return {
        "runs": summaries,
        "round_rows": len(csv_rows),
        "original_source_hash_bytes": source_bytes,
        "frozen_source_files": len(manifest["files"]),
        "auditor_source_files": len(config["auditor_sources"]),
        "derived_report_files": len(config["report_files"]),
        "renderer_accounting": accounting,
    }


def audit(config_path: Path, out: Path) -> dict:
    if config_path.stat().st_size > 64 * 1024:
        raise ValueError("Audit configuration ceiling")
    config_bytes = config_path.read_bytes()
    config = strict_json(config_bytes)
    if config.get("schema") != SCHEMA or config.get("launch_ready") is not True:
        raise ValueError("Frozen reviewed audit configuration required")
    limits = config["limits"]
    for key, maximum in {
        "wall_seconds": 120,
        "memory_bytes": 256 * 1024**2,
        "read_bytes": 256 * 1024**2,
        "output_bytes": 16 * 1024**2,
    }.items():
        if not 0 < limits[key] <= maximum:
            raise ValueError("Audit ceiling: " + key)
    if not 1 <= config["prefix_traces"] <= 8:
        raise ValueError("Trace-prefix ceiling")
    if len(config_bytes) > limits["output_bytes"]:
        raise ValueError("Audit configuration exceeds output budget")
    check = Audit(limits)
    check.check()
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_bytes(config_bytes)
    check.output_bytes = len(config_bytes)
    status, failure, values = "PASS", None, {}
    try:
        values = verify(config, check)
    except Exception as error:
        status, failure = "FAIL", f"{type(error).__name__}: {error}"
    result = {
        "schema": SCHEMA,
        "status": status,
        "failure": failure,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        **values,
        "files": check.files,
        "checks": check.checks,
        "limitations": [
            "No physics, inference, gradients, fitting, replay, pickle or checkpoint decoding",
            "Source bindings and exported prefix joins checked; not original trace extraction equivalence",
            "Availability claims limited to inventoried source directories",
            "Historical loop counters exclude evaluation and lost/interrupted work",
        ],
        "accounting": check.receipt(),
    }
    while True:
        payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        total = len(config_bytes) + len(payload)
        if result["accounting"]["output_bytes"] == total:
            break
        result["accounting"]["output_bytes"] = total
    if total > limits["output_bytes"]:
        raise ValueError("Audit output budget")
    with (out / "audit.json").open("xb") as stream:
        stream.write(payload)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit(args.config, args.out)
    print(
        json.dumps(
            {"status": result["status"], "checks": result["checks"], "failure": result["failure"]}
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
