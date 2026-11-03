"""Verify and decode a compact J1 capsule into an explicit fresh replay directory.

Publication-only adapter. The immutable compact capsule is never modified, and
no simulator, policy, fitting, score or outcome calculation is imported.

The byte/hash/decompression algorithm is preserved from the sealed J1 publication
helper. This portable entry point enforces actual RSS and wall boundaries on both
supported platforms; Linux additionally installs an address-space limit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import resource
import signal
import sys
import time
from pathlib import Path

MAX_READ = 128 * 1024**2
MAX_OUTPUT = 128 * 1024**2
MAX_FILE = 64 * 1024**2
MAX_FILES = 10_000
MAX_WALL = 120
MAX_MEMORY = 512 * 1024**2


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Materialization path escaped its root")
    result = root / relative
    if any(path.is_symlink() for path in [result, *result.parents] if path.is_relative_to(root)):
        raise ValueError("Materialization cannot follow a symlink")
    return result


def rss_bytes() -> int:
    """Measure this exec image; Linux ru_maxrss may retain the fork parent's peak."""
    if sys.platform == "linux":
        fields = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            name, separator, value = line.partition(":")
            if not separator or name not in {"VmHWM", "VmRSS"}:
                continue
            words = value.split()
            if name in fields or len(words) != 2 or not words[0].isdecimal() or words[1] != "kB":
                raise RuntimeError("Linux current-exec RSS metadata is malformed")
            fields[name] = int(words[0]) * 1024
        if set(fields) != {"VmHWM", "VmRSS"} or min(fields.values()) <= 0:
            raise RuntimeError("Linux current-exec RSS metadata is unavailable")
        return max(fields.values())
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def install_limits() -> dict:
    """Install supported CLI limits without claiming macOS address-space control."""
    if sys.platform == "linux":
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        ceiling = MAX_MEMORY if hard == resource.RLIM_INFINITY else min(MAX_MEMORY, hard)
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        return dict(
            mode="linux_address_space_rss_wall",
            address_space_enforced=True,
            address_space_bytes=ceiling,
            rss_boundary="Current-exec /proc/self/status VmHWM/VmRSS before/after each read, decode and write",
        )
    if sys.platform == "darwin":
        return dict(
            mode="darwin_rss_wall",
            address_space_enforced=False,
            address_space_bytes=None,
            rss_boundary="Measured ru_maxrss bytes before/after each read, decode and write",
            reason="macOS RLIMIT_AS is not installed; bounded per-file RSS and wall guards apply",
        )
    raise ValueError("J1 publication materialization supports the pinned Linux and macOS hosts")


def materialize(
    source: Path,
    out: Path,
    manifest_sha256: str,
    exit_sha256: str,
    *,
    limit_enforcement: dict | None = None,
) -> dict:
    started = time.monotonic()
    for value in (source.absolute(), out.absolute()):
        if any(path.is_symlink() for path in [value, *value.parents]):
            raise ValueError("Materialization roots cannot contain a symlink")
    source, out = source.resolve(), out.resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise FileExistsError(
            "Materialization requires an explicit fresh directory outside the capsule"
        )
    out.mkdir(parents=True, exist_ok=False)
    read_bytes, output_bytes = 0, 0
    cache: dict[str, bytes] = {}
    files = {}

    def boundary() -> None:
        if time.monotonic() - started > MAX_WALL:
            raise TimeoutError("Materialization wall allocation exhausted")
        if rss_bytes() > MAX_MEMORY:
            raise MemoryError("Materialization measured peak RSS exceeds 512 MiB")

    def read(name: str) -> bytes:
        nonlocal read_bytes
        boundary()
        if name in cache:
            return cache[name]
        path = safe(source, name)
        amount = path.stat().st_size
        if amount > MAX_FILE or read_bytes + amount > MAX_READ:
            raise ValueError("Materialization source-read allocation exhausted")
        data = path.read_bytes()
        read_bytes += len(data)
        boundary()
        if len(data) != amount:
            raise ValueError("Compact source changed during materialization")
        if name in {"files_manifest.json", "collection_exit.json", "capsule.json"}:
            cache[name] = data
        return data

    result = dict(
        schema="walker-j1-verified-materialization-v1",
        passed=False,
        source=str(source),
        output=str(out),
        manifest_sha256=manifest_sha256,
        collection_exit_sha256=exit_sha256,
        new_scientific_work=0,
        immutable_capsule_modified=False,
        limit_enforcement=limit_enforcement
        or dict(
            mode="api_rss_wall",
            address_space_enforced=False,
            address_space_bytes=None,
            rss_boundary="Measured peak RSS before/after each read, decode and write",
            reason="Direct API: no process-global address-space limit is installed",
        ),
    )
    try:
        raw_manifest, raw_exit = (
            read("files_manifest.json"),
            read("collection_exit.json"),
        )
        if sha(raw_manifest) != manifest_sha256 or sha(raw_exit) != exit_sha256:
            raise ValueError("Materialization closing hashes differ from transferred evidence")
        manifest, closing = json.loads(raw_manifest), json.loads(raw_exit)
        if closing.get("passed") is not True or closing.get("manifest_sha256") != manifest_sha256:
            raise ValueError("Cannot materialize an unclosed compact collection")
        capsule = json.loads(read("capsule.json"))
        if capsule.get("mode") != "metadata":
            raise ValueError("Saved-report materialization requires the metadata capsule")
        expected = set(manifest["files"]) | {
            "files_manifest.json",
            "collection_exit.json",
        }
        if len(expected) > MAX_FILES:
            raise ValueError("Materialization file allocation exhausted")
        actual = set()
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("Compact source contains an alias")
            if path.is_file():
                actual.add(str(path.relative_to(source)))
        if actual != expected:
            raise ValueError("Compact source has missing or unexpected files")
        original_bindings = capsule.get("files", {})
        for name in sorted(expected):
            boundary()
            data = read(name)
            binding = manifest["files"].get(name)
            if binding is not None and (
                len(data) != binding["bytes"] or sha(data) != binding["sha256"]
            ):
                raise ValueError("Stored compact hash/size differs: " + name)
            original = original_bindings.get(name, {})
            destination_name = name
            if original.get("encoding") == "gzip_exact_original_bytes":
                amount = original.get("original_bytes")
                if (
                    type(amount) is not int
                    or not 0 <= amount <= MAX_FILE
                    or not name.endswith(".gz")
                ):
                    raise ValueError("Declared decompression size/path is invalid")
                if output_bytes + amount > MAX_OUTPUT - 2 * 1024**2:
                    raise ValueError(
                        "Materialization output allocation exhausted before decompression"
                    )
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                    decoded = stream.read(amount + 1)
                boundary()
                if len(decoded) != amount or sha(decoded) != original.get("original_sha256"):
                    raise ValueError(
                        "Bounded decompression differs from original hash/size: " + name
                    )
                data, destination_name = decoded, name[:-3]
            if output_bytes + len(data) > MAX_OUTPUT - 2 * 1024**2:
                raise ValueError("Materialization output allocation exhausted")
            destination = safe(out, destination_name)
            boundary()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(data)
            files[destination_name] = dict(bytes=len(data), sha256=sha(data), compact_file=name)
            output_bytes += len(data)
            boundary()
        result["passed"] = True
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        try:
            measured_rss = rss_bytes()
        except (OSError, RuntimeError, ValueError) as error:
            measured_rss = None
            result["passed"] = False
            result["rss_measurement_error"] = dict(type=type(error).__name__, message=str(error))
        result.update(
            files=files,
            source_read_bytes=read_bytes,
            output_bytes_before_receipt=output_bytes,
            wall_seconds=time.monotonic() - started,
            user_seconds=usage.ru_utime,
            system_seconds=usage.ru_stime,
            peak_rss_bytes=measured_rss,
            peak_rss_scope="Current exec image VmHWM/VmRSS"
            if sys.platform == "linux"
            else "Process ru_maxrss in platform-native units",
            resource_peak_rss_bytes=int(
                usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
            ),
            resource_peak_rss_scope="Accounting only; Linux ru_maxrss may retain a pre-exec/fork-parent peak",
            limits=dict(
                source_read_bytes=MAX_READ,
                output_bytes=MAX_OUTPUT,
                individual_decompressed_bytes=MAX_FILE,
                wall_seconds=MAX_WALL,
                peak_rss_bytes=MAX_MEMORY,
                address_space_bytes=result["limit_enforcement"]["address_space_bytes"],
            ),
            recipe="Copy exact identity files; decode only declared .gz originals after stored and original hash/size checks. No immutable capsule bytes are rewritten.",
        )
        with (out / "materialization_receipt.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def timeout(signum, frame) -> None:
    raise TimeoutError("Materialization wall allocation exhausted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--exit-sha256", required=True)
    args = parser.parse_args(argv)
    enforcement = install_limits()
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, MAX_WALL)
    try:
        result = materialize(
            args.capsule,
            args.out,
            args.manifest_sha256,
            args.exit_sha256,
            limit_enforcement=enforcement,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                out=str(args.out),
                source_read_bytes=result["source_read_bytes"],
                output_bytes=result["output_bytes_before_receipt"],
            )
        )
    )
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
