"""Replay the compact saved report: python render_report.py --out /tmp/fresh-v-report.

Only capsule-bound metadata and losslessly compressed returned-prefix scalars
are read. No raw simulator arrays, checkpoint, policy, or fitting is required.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
CAPSULE = Path(__file__).resolve().parent
REPOSITORY = CAPSULE.parents[1]
MAX_CURVES = 224000 * 512
REPORT_SOURCE = "src/gps/mppi_gps/reporting/student_visitation.py"


def source_file(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Capsule path must be relative and contained")
    path = CAPSULE / relative
    if any(
        (CAPSULE / Path(*relative.parts[:n])).is_symlink()
        for n in range(1, len(relative.parts) + 1)
    ):
        raise ValueError("Capsule symlinks are not accepted")
    if not path.resolve().is_relative_to(CAPSULE) or not path.is_file():
        raise ValueError("Missing or escaped capsule file: " + name)
    return path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(value) -> tuple:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def copy_verified(name: str, destination: Path, bindings: dict, limit: int) -> None:
    binding, source = bindings[name], source_file(name)
    size = binding["bytes"]
    if type(size) is not int or not 0 <= size <= limit or source.stat().st_size != size:
        raise ValueError("Capsule file size differs or exceeds its bound: " + name)
    before, copied, checksum = source.stat(), 0, hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while chunk := incoming.read(min(1024**2, size - copied + 1)):
            copied += len(chunk)
            if copied > size:
                raise ValueError("Capsule file grew during replay")
            outgoing.write(chunk)
            checksum.update(chunk)
    if (
        identity(source.stat()) != identity(before)
        or copied != size
        or checksum.hexdigest() != binding["sha256"]
    ):
        raise ValueError("Capsule file changed or failed its hash: " + name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.exists() or out.is_relative_to(CAPSULE):
        raise ValueError(
            "Report output must be fresh and outside this immutable capsule"
        )
    manifest_path = source_file("capsule.json")
    if manifest_path.stat().st_size > 4 * 1024**2:
        raise ValueError("Capsule manifest exceeds its compact metadata bound")
    bindings = json.loads(manifest_path.read_text())["files"]
    if not {
        "run/config.json",
        "run/worker_result.json",
        "run/worker/outcomes.jsonl",
    } <= set(bindings):
        raise ValueError("The capsule lacks required V run metadata")
    with tempfile.TemporaryDirectory(prefix="walker-v-report-") as directory:
        snapshot = Path(directory)
        metadata_bytes = 0
        for name in bindings:
            if (
                name.startswith("run/")
                and "/source/" not in name
                and Path(name).suffix in {".json", ".jsonl"}
            ):
                metadata_bytes += bindings[name]["bytes"]
                if metadata_bytes > 32 * 1024**2:
                    raise ValueError("Compact run metadata exceeds 32 MiB")
                copy_verified(name, snapshot / name, bindings, 4 * 1024**2)
        audit_name = "audit/result/audit.json"
        copy_verified(audit_name, snapshot / audit_name, bindings, 4 * 1024**2)
        audit = json.loads((snapshot / audit_name).read_text())
        expected = audit["prefix_curve_file"]
        if (
            expected["file"] != "prefix_curves.jsonl"
            or type(expected["bytes"]) is not int
            or not 0 <= expected["bytes"] <= MAX_CURVES
        ):
            raise ValueError("Invalid audited prefix-curve binding")
        compressed = "audit/result/prefix_curves.jsonl.gz"
        copy_verified(compressed, snapshot / compressed, bindings, 128 * 1024**2)
        checksum, count = hashlib.sha256(), 0
        with (
            gzip.open(snapshot / compressed, "rb") as incoming,
            (snapshot / audit_name)
            .with_name("prefix_curves.jsonl")
            .open("xb") as outgoing,
        ):
            while chunk := incoming.read(min(1024**2, expected["bytes"] - count + 1)):
                count += len(chunk)
                if count > expected["bytes"]:
                    raise ValueError(
                        "Decompressed curves exceed the audited byte bound"
                    )
                checksum.update(chunk)
                outgoing.write(chunk)
        if count != expected["bytes"] or checksum.hexdigest() != expected["sha256"]:
            raise ValueError("Decompressed curves differ from the independent audit")
        version_name = "publication/report_version_002.json"
        copy_verified(version_name, snapshot / version_name, bindings, 4 * 1024**2)
        version = json.loads((snapshot / version_name).read_text())
        if (
            version.get("schema") != "walker-visitation-v-publication-report-version-v1"
            or type(version.get("version")) is not int
            or version["version"] != 2
            or version.get("scientific_source_commit")
            != "70075397bfb2966dff97e325c4104915f03c9581"
            or version.get("unchanged_scientific_admission_sha256")
            != "a8c0dc1f25c7aeae93c0161bb042c403887e1c65134dbb97c75ce41376daebdf"
            or bindings["run/admission.json"]["sha256"]
            != version["unchanged_scientific_admission_sha256"]
            or any(
                type(version.get(key)) is not int or version[key] != 0
                for key in ("new_physics", "new_neural_rows", "new_fitting")
            )
        ):
            raise ValueError(
                "The publication report version or unchanged scientific scope differs"
            )
        excerpt = "publication/source/" + REPORT_SOURCE
        for key, name in (
            ("source", excerpt),
            ("original_report_source", "run/source/" + REPORT_SOURCE),
        ):
            declared = version[key]
            if declared["file"] != name or any(
                declared[field] != bindings[name][field]
                for field in ("bytes", "sha256")
            ):
                raise ValueError("Publication report source binding differs")
        copy_verified(
            "run/source/" + REPORT_SOURCE,
            snapshot / "run/source" / REPORT_SOURCE,
            bindings,
            4 * 1024**2,
        )
        copy_verified(excerpt, snapshot / excerpt, bindings, 4 * 1024**2)
        current = REPOSITORY / REPORT_SOURCE
        if (
            current.is_symlink()
            or not current.is_file()
            or current.stat().st_size != bindings[excerpt]["bytes"]
            or digest(current) != bindings[excerpt]["sha256"]
        ):
            raise ValueError(
                "Current report implementation differs from the capsule's sealed source"
            )
        sys.path.insert(0, str(REPOSITORY))
        from src.gps.mppi_gps.reporting.student_visitation import report_run

        receipt = report_run(snapshot / "run", out, audit=snapshot / audit_name)
        print(
            json.dumps(
                dict(output=str(out), summaries=receipt.get("summaries")),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
