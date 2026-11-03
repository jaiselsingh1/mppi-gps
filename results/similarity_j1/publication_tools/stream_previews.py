"""Stream exact predeclared J1 atlas previews; never create a remote video/archive copy."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import resource
import signal
import sys
import tarfile
import time
from pathlib import Path

from collector import MAX_PREVIEW, MAX_RSS, MAX_WALL, Reader, closure_gate, deadline
from verify_capsule import verify

sys.dont_write_bytecode = True


class BoundedOutput:
    def __init__(self, stream):
        self.stream = stream
        self.bytes = 0
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        if self.bytes + len(data) > MAX_PREVIEW:
            raise ValueError(
                "J1 exact preview tar exceeds its 8 MiB transfer reservation"
            )
        amount = self.stream.write(data)
        if amount != len(data):
            raise IOError("Preview stream write was incomplete")
        self.bytes += amount
        self.digest.update(data)
        return amount

    def flush(self) -> None:
        self.stream.flush()


def stream_catalog(root: Path, manifest_sha256: str, exit_sha256: str, output) -> dict:
    started = time.monotonic()
    verified = verify(
        root, manifest_sha256, exit_sha256, external_preview_catalog_only=True
    )
    capsule_bytes = (root / "capsule.json").read_bytes()
    capsule = json.loads(capsule_bytes)
    if capsule.get("mode") != "previews":
        raise ValueError(
            "Preview streaming requires the closed fixed-role preview catalog"
        )
    run, art = Path(capsule["run_location"]), Path(capsule["artifacts_location"])
    reader = Reader([run, art, root])
    closure_gate(run, art, reader)
    manifest = reader.json(root / "files_manifest.json")
    sources = [
        (root / name, name, binding) for name, binding in manifest["files"].items()
    ]
    for name in ("files_manifest.json", "collection_exit.json"):
        path = root / name
        data = reader.read(path)
        sources.append(
            (path, name, dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        )
    for name, binding in capsule.get("files", {}).items():
        sources.append((Path(binding["source"]), name, binding))
    sink = BoundedOutput(output)
    with tarfile.open(fileobj=sink, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        for path, name, binding in sorted(sources, key=lambda row: row[1]):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Preview tar name escaped its fixed-role directory")
            data = reader.read(path)
            if (
                len(data) != binding["bytes"]
                or hashlib.sha256(data).hexdigest() != binding["sha256"]
            ):
                raise ValueError(
                    "Closed preview source changed before transfer: " + name
                )
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    sink.flush()
    reader.unchanged()
    return dict(
        schema="walker-j1-exact-preview-tar-stream-v1",
        passed=True,
        files=len(sources),
        tar_bytes=sink.bytes,
        tar_sha256=sink.digest.hexdigest(),
        physical_source_and_catalog_read_bytes=reader.read_bytes
        + verified["physical_read_bytes"]
        + len(capsule_bytes),
        catalog_verification_read_bytes=verified["physical_read_bytes"],
        wall_seconds=time.monotonic() - started,
        permanent_archive_created=False,
        remote_video_copies_created=False,
        altered_videos=False,
        new_scientific_work=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--exit-sha256", required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (MAX_RSS, MAX_RSS))
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, MAX_WALL)
    value = stream_catalog(
        args.catalog.resolve(),
        args.manifest_sha256,
        args.exit_sha256,
        sys.stdout.buffer,
    )
    print(json.dumps(value), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
