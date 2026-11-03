"""Create-only compact collection of a CLOSED J1 study; stdlib, saved bytes only.

No source import, network, physics, policy forward, fit, metric recomputation or
unclosed outcome inspection occurs here. Metadata and fixed-role previews are
separate explicit invocations. Large raw evidence remains at its original host.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import signal
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
MIB = 1024**2
MAX_READ = 128 * MIB
METADATA_READ_RESERVATION = 112 * MIB
MAX_COMPACT = 56 * MIB
MAX_PREVIEW = 8 * MIB
MAX_WALL = 600
MAX_RSS = 1024**3
MAX_FILES = 50_000
FIXED_PREVIEWS = (
    "development_00_mppi",
    "development_00_s0_incoming",
    "development_00_s0_mppi_reproduction",
)
RUN_FILES = (
    "config.json",
    "manifest.json",
    "input_manifest.json",
    "preparation.json",
    "worker_result.json",
    "worker_process_receipt.json",
    "worker_exit.json",
    "worker_coordinator_exit.json",
    "worker_ownership.json",
    "audit_ownership.json",
    "render_ownership.json",
    "admission.json",
    "finalizer_exit.json",
    "prepared/renderer.json",
    "worker/calibration.json",
    "worker/score_definitions.json",
    "worker/score_models_prelearning.json",
    "worker/score_models.json",
    "worker/prediction_model_seal.json",
    "worker/predictive_comparison.json",
    "worker/probe_summary.json",
    "worker/probe_results.json",
    "worker/cases.json",
    "worker/teachers.json",
    "worker/traces.json",
    "worker/case_outcomes.jsonl",
    "worker_stdout.txt",
)
RUN_GLOBS = (
    "worker/source_groups/*.json",
    "worker/score_seals/*.json",
    "worker/parents/*/bindings.json",
    "worker/fits/*/fit_receipt.json",
    "worker/fits/*/fit_failure.json",
    "worker/fits/*/learning_curve.json",
)
AUDIT_FILES = (
    "coordinator_exit.json",
    "exit.json",
    "stdout.txt",
    "result/audit.json",
    "result/process_receipt.json",
)
RENDER_FILES = (
    "coordinator_exit.json",
    "exit.json",
    "stdout.txt",
    "result/atlas_receipt.json",
    "result/process_receipt.json",
    "result/source_read_accounting.json",
    "result/index.html",
)


def new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def storage_census(run: Path, art: Path) -> dict:
    totals, count = {}, 0
    for root, name in ((run, "run_bytes"), (art, "artifacts_bytes")):
        amount = 0
        for path in root.rglob("*"):
            count += 1
            if count > MAX_FILES:
                raise ValueError("J1 storage census exceeds its finite file allocation")
            if path.is_symlink():
                raise ValueError("J1 storage census cannot account an aliased output")
            if path.is_file():
                amount += path.stat().st_size
        totals[name] = amount
    totals.update(
        total_bytes=sum(totals.values()),
        entries=count,
        filesystem_free_bytes=shutil.disk_usage(run).free,
    )
    return totals


def deadline(signum: int, frame: Any) -> None:
    raise TimeoutError("J1 compact collection exhausted its 600-second wall allocation")


class Reader:
    def __init__(self, roots: list[Path], *, limit: int = MAX_READ):
        self.roots = [root.resolve() for root in roots]
        self.read_bytes = 0
        self.limit = limit
        self.cache: dict[Path, bytes] = {}
        self.identities: dict[Path, tuple[int, int, int, int]] = {}

    def path(self, value: Path) -> Path:
        absolute = value.absolute()
        root = next(
            (root for root in self.roots if absolute.is_relative_to(root)), None
        )
        if root is None:
            raise ValueError("Collection read escaped its explicit study roots")
        for parent in [absolute, *absolute.parents]:
            if parent == root.parent:
                break
            if parent.is_symlink():
                raise ValueError("Collection cannot follow a symlink: " + str(parent))
        return absolute

    def read(self, path: Path) -> bytes:
        path = self.path(path)
        if path in self.cache:
            return self.cache[path]
        before = identity(path)
        amount = before[2]
        if (
            not path.is_file()
            or amount > 64 * MIB
            or self.read_bytes + amount > self.limit
        ):
            raise ValueError(
                "J1 metadata source-read or individual-file allocation exhausted"
            )
        data = path.read_bytes()
        self.read_bytes += len(data)
        if len(data) != amount or identity(path) != before:
            raise ValueError("J1 preserved input changed during collection")
        self.identities[path], self.cache[path] = before, data
        return data

    def json(self, path: Path) -> Any:
        return json.loads(self.read(path))

    def unchanged(self) -> None:
        if any(identity(path) != value for path, value in self.identities.items()):
            raise ValueError("Closed J1 metadata changed during collection")


def closed_receipt(value: dict) -> bool:
    # Scientific failure is collectable; an unreaped/unknown process is not a
    # substitute for proving the live process has finished.
    return (
        type(value.get("pid")) is int
        and value["pid"] > 0
        and type(value.get("exit_code")) is int
        and value.get("leader_reaped") is True
    )


def live_study_processes(
    run: Path,
    *,
    proc: Path = Path("/proc"),
    uid: int | None = None,
    nonces: tuple[str, ...] = (),
) -> list[int]:
    uid = os.getuid() if uid is None else uid
    found = []
    script = str(run / "source/scripts/train_mppi_gps.py").encode()
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            if directory.stat().st_uid != uid:
                continue
            args = (directory / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            has_run = any(
                args[index : index + 2] == [b"--run", str(run).encode()]
                for index in range(len(args) - 1)
            )
            matched_argv = has_run and (
                script in args
                or b"src.gps.mppi_gps.reporting.similarity" in args
                or any(arg.endswith(b"/operator_001/remote_program.py") for arg in args)
            )
            try:
                environment = (
                    (directory / "environ").read_bytes().split(b"\0") if nonces else []
                )
            except PermissionError:
                environment = []
            matched_nonce = any(
                ("J1_PHASE_NONCE=" + value).encode() in environment for value in nonces
            )
            if matched_argv or matched_nonce:
                found.append(int(directory.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def closure_gate(run: Path, art: Path, reader: Reader) -> dict:
    """Read ownership/exit metadata before any worker outcome or protected score."""
    nonces = tuple(
        reader.json(path)["nonce"]
        for path in (
            run / (phase + "_ownership.json") for phase in ("worker", "audit", "render")
        )
        if path.is_file()
    )
    active = live_study_processes(run, nonces=nonces)
    if active:
        raise ValueError(
            "J1 is still active; outcome collection prohibited: " + str(active)
        )
    root = art / "operator_001"
    launch = root / "launch/operator_receipt.json"
    if not launch.is_file():
        raise ValueError("Actual external worker closure receipt is missing")
    launched = reader.json(launch)
    if not closed_receipt(launched.get("process_receipt", {})):
        raise ValueError("Actual J1 worker coordinator has not been reaped")
    closure = dict(checked_owner_nonces=list(nonces))
    for phase, directory in (
        ("worker", None),
        ("audit", art / "audit_001"),
        ("render", art / "render_001"),
    ):
        owner_path = run / (phase + "_ownership.json")
        if not owner_path.is_file():
            closure[phase] = dict(
                available=False, status="unattempted_or_preownership_failure"
            )
            continue
        owner = reader.json(owner_path)
        coordinator_path = (
            run / "worker_coordinator_exit.json"
            if phase == "worker"
            else directory / "coordinator_exit.json"
        )
        if not coordinator_path.is_file():
            raise ValueError(
                "A declared J1 phase has no actual closed coordinator: " + phase
            )
        coordinator = reader.json(coordinator_path)
        if not closed_receipt(coordinator) or coordinator["pid"] != owner.get(
            "coordinator_pid"
        ):
            raise ValueError(
                "J1 phase closure does not bind its actual coordinator: " + phase
            )
        child_path = (
            run / "worker_exit.json" if phase == "worker" else directory / "exit.json"
        )
        child = reader.json(child_path) if child_path.is_file() else None
        if child is not None and not closed_receipt(child):
            raise ValueError("J1 owned child has no reaped status: " + phase)
        closure[phase] = dict(
            available=True,
            coordinator=coordinator,
            child=child,
            uncertainty=child is None
            or bool(coordinator.get("unknown_partial_work_possible"))
            or bool(child.get("unknown_partial_work_possible")),
        )
    return closure


def selected_files(
    run: Path, art: Path, config: dict
) -> tuple[list[tuple[Path, str]], list[str]]:
    result, missing = [], []

    def retain(path: Path, destination: str) -> None:
        if path.is_file():
            result.append((path, destination))
        else:
            missing.append(destination)

    for name in RUN_FILES:
        retain(run / name, "run/" + name)
    source_names = [
        config["protocol"],
        "configs/studies/similarity_j1_0.json",
        "docs/proposals/similarity_j1_user_request.txt",
    ]
    for role in ("xml", "historical_source_bindings"):
        if config.get("inputs", {}).get(role, {}).get("file"):
            source_names.append(config["inputs"][role]["file"])
    for name in source_names:
        retain(run / "source" / name, "run/source/" + name)
    for pattern in RUN_GLOBS:
        for path in sorted(run.glob(pattern)):
            retain(path, "run/" + str(path.relative_to(run)))
    for split in ("development", "protected"):
        name = "worker/score_seals/" + split + ".json"
        retain(run / name, "run/" + name)
    for group in config.get("groups", []):
        name = "worker/source_groups/" + group["id"] + ".json"
        retain(run / name, "run/" + name)
        for learner in group["learners"]:
            name = "worker/parents/" + learner["id"] + "/bindings.json"
            retain(run / name, "run/" + name)
            for source in ("mppi", "reference"):
                for filename in ("fit_receipt.json", "learning_curve.json"):
                    name = (
                        "worker/fits/" + learner["id"] + "_" + source + "/" + filename
                    )
                    retain(run / name, "run/" + name)
    for namespace, names in (("audit_001", AUDIT_FILES), ("render_001", RENDER_FILES)):
        for name in names:
            retain(
                art / namespace / name,
                ("audit/" if namespace == "audit_001" else "atlas/") + name,
            )
    for path in sorted((art / "report_001").glob("*")):
        if path.suffix in {".json", ".md", ".csv", ".png"}:
            retain(path, "report/" + path.name)
    if not (art / "report_001/report_receipt.json").is_file():
        missing.append("report/report_receipt.json")
    for path in sorted((art / "operator_001").rglob("*")):
        if (
            "__pycache__" not in path.parts
            and path.is_file()
            and path.suffix in {".json", ".md", ".py", ".txt"}
        ):
            retain(path, "operator/" + str(path.relative_to(art / "operator_001")))
    tools = Path(__file__).resolve().parent
    if tools.is_relative_to(art):
        for path in sorted(tools.rglob("*")):
            if (
                "__pycache__" not in path.parts
                and path.is_file()
                and path.suffix in {".json", ".md", ".py", ".txt"}
            ):
                retain(path, "publication_tools/" + str(path.relative_to(tools)))
    for phase in (
        "prepare",
        "launch",
        "audit-launch",
        "finalize",
        "render-launch",
        "report",
    ):
        if not (art / "operator_001" / phase / "operator_receipt.json").is_file():
            missing.append("operator/" + phase + "/operator_receipt.json")
    unique = dict((destination, path) for path, destination in result)
    return [
        (path, destination) for destination, path in sorted(unique.items())
    ], sorted(set(missing))


def compression_for(path: Path) -> str:
    if (
        path.name == "learning_curve.json"
        or path.suffix == ".csv"
        or (path.suffix == ".txt" and path.stat().st_size > MIB)
    ):
        return "gzip_exact_original_bytes"
    return "identity"


def save_selected(
    reader: Reader, source: Path, destination: Path, *, remaining: int
) -> dict:
    data = reader.read(source)
    encoding = compression_for(source)
    encoded = gzip.compress(data, mtime=0) if encoding != "identity" else data
    if len(encoded) > remaining:
        raise ValueError("J1 56 MiB compact payload reservation exhausted")
    destination = (
        destination.with_name(destination.name + ".gz")
        if encoding != "identity"
        else destination
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(encoded)
    return dict(
        source=str(source),
        path=str(destination),
        original_bytes=len(data),
        original_sha256=sha_bytes(data),
        bytes=len(encoded),
        sha256=sha_bytes(encoded),
        encoding=encoding,
    )


def inventory(
    run: Path, art: Path, out: Path, reader: Reader, copied: set[Path]
) -> list[dict]:
    audit_path = art / "audit_001/result/audit.json"
    audit = reader.json(audit_path) if audit_path.is_file() else {}
    input_hashes = audit.get("input_files", {})
    stream_hashes = audit.get("stream_checks", {}).get("hashes", {})
    result = []
    for root, namespace in ((run, "run"), (art, "artifacts")):
        for path in sorted(root.rglob("*")):
            if len(result) >= MAX_FILES:
                raise ValueError(
                    "J1 remote availability catalog exceeds its file allocation"
                )
            if path.is_relative_to(out) or not path.is_file() or path in copied:
                continue
            path = reader.path(path)
            rel = str(path.relative_to(root))
            if namespace == "run" and rel.startswith("source/"):
                continue  # Complete source identities are already in manifest.json.
            before = identity(path)
            binding = input_hashes.get(rel, {}) if namespace == "run" else {}
            checksum, origin = (
                binding.get("sha256"),
                "closed_audit_input_files" if binding.get("sha256") else None,
            )
            if namespace == "run" and rel.startswith("worker/evidence/"):
                checksum = stream_hashes.get(path.name)
                origin = "closed_audit_stream_checks" if checksum else None
            if checksum is not None and binding.get("bytes", before[2]) != before[2]:
                raise ValueError(
                    "Remote evidence size differs from its closed audit binding"
                )
            result.append(
                dict(
                    namespace=namespace,
                    file=rel,
                    location=str(path),
                    bytes=before[2],
                    available=True,
                    retained="remote_only",
                    sha256=checksum,
                    checksum_source=origin,
                    checksum_recomputed_by_collector=False,
                    checksum_unavailable_reason=None
                    if checksum
                    else "No closed whole-file hash binding; no large re-read authorized",
                    identity=dict(
                        device=before[0], inode=before[1], mtime_ns=before[3]
                    ),
                )
            )
    for namespace, root, name in (
        ("run", run, "worker/evidence/arrays.bin"),
        ("run", run, "worker/evidence/arrays.index.jsonl"),
        ("run", run, "worker/evidence/ledger.jsonl"),
        ("artifacts", art, "audit_001/result/stream_index.sqlite"),
    ):
        if not (root / name).is_file():
            result.append(
                dict(
                    namespace=namespace,
                    file=name,
                    location=str(root / name),
                    available=False,
                    retained="missing_or_uncreated",
                    bytes=None,
                    sha256=None,
                    checksum_source=None,
                    checksum_recomputed_by_collector=False,
                    checksum_unavailable_reason="Original container is missing or was never created",
                )
            )
    return result


def collect_metadata(
    run: Path, art: Path, out: Path, reader: Reader, config: dict
) -> dict:
    selected, missing = selected_files(run, art, config)
    files, payload = {}, 0
    for source, destination in selected:
        row = save_selected(
            reader, source, out / destination, remaining=MAX_COMPACT - 4 * MIB - payload
        )
        row["path"] = str(Path(row["path"]).relative_to(out))
        files[row["path"]] = row
        payload += row["bytes"]
    remote = inventory(run, art, out, reader, {source for source, _ in selected})
    external = dict(
        schema="walker-j1-remote-evidence-availability-v1",
        files=remote,
        host=config["locations"]["host"],
        source_location=str(run / "source"),
        source_identities="run/manifest.json; source bytes intentionally not recopied",
        excluded_raw_categories=[
            "arrays.bin",
            "arrays.index.jsonl",
            "ledger.jsonl",
            "checkpoints",
            "minibatch/RNG tapes",
            "audit SQLite",
            "atlas videos/scalars",
        ],
        full_raw_hash_rereads=0,
    )
    encoded = (json.dumps(external, indent=2, allow_nan=False) + "\n").encode()
    if payload + len(encoded) > MAX_COMPACT - 2 * MIB:
        raise ValueError(
            "Remote availability metadata exceeds its compact closing reservation"
        )
    with (out / "external_evidence.json").open("xb") as stream:
        stream.write(encoded)
    return dict(
        files=files,
        missing=missing,
        selected_payload_bytes=payload,
        remote_only_files=len(remote),
        remote_only_bytes=sum(v["bytes"] or 0 for v in remote),
    )


def collect_previews(run: Path, art: Path, out: Path, reader: Reader) -> dict:
    atlas_path = art / "render_001/result/atlas_receipt.json"
    if not atlas_path.is_file():
        return dict(
            status="UNAVAILABLE_NO_ATLAS", files={}, missing=list(FIXED_PREVIEWS)
        )
    atlas = reader.json(atlas_path)
    if atlas.get("passed") is not True:
        return dict(
            status="UNAVAILABLE_ATLAS_DID_NOT_PASS",
            files={},
            missing=list(FIXED_PREVIEWS),
        )
    records = {row["tag"]: row for row in atlas["traces"]}
    files, missing, payload = {}, [], 0
    for tag in FIXED_PREVIEWS:
        if tag not in records:
            missing.append(tag)
            continue
        directory = records[tag]["directory"]
        if Path(directory).name != directory:
            raise ValueError("Atlas preview directory escaped its role binding")
        source = art / "render_001/result" / directory
        for name in (
            "video.mp4",
            "frames.csv",
            "trace.json",
            "encoder_receipt.json",
            "encoder_stderr.txt",
            "motion_contacts_controls.png",
        ):
            path = source / name
            if not path.is_file():
                missing.append(tag + "/" + name)
                continue
            data = reader.read(path)
            if payload + len(data) > MAX_PREVIEW - 128 * 1024:
                raise ValueError(
                    "Fixed-role preview reservation exceeds 8 MiB; no replacement or transcoding"
                )
            destination = Path(tag) / name
            files[str(destination)] = dict(
                bytes=len(data),
                sha256=sha_bytes(data),
                source=str(path),
                encoding="identity",
            )
            payload += len(data)
    return dict(
        status="COLLECTED_FIXED_PREDECLARED_ROLES",
        files=files,
        missing=missing,
        selected_payload_bytes=payload,
        roles=list(FIXED_PREVIEWS),
        altered_or_transcoded_videos=False,
        transfer="Stream these exact source files directly; no remote preview/video copy was created",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("metadata", "previews"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--metadata-capsule", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    run, out = args.run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        raise ValueError("Compact output must be outside immutable run evidence")
    out.mkdir(parents=True, exist_ok=False)
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, MAX_WALL)
    # No simulator/Torch imports; a strict address-space ceiling is safe here.
    resource.setrlimit(resource.RLIMIT_AS, (MAX_RSS, MAX_RSS))
    result = dict(
        schema="walker-j1-compact-collector-v1",
        mode=args.mode,
        passed=False,
        automatic_followup=False,
        scientific_work=dict(
            native_controls=0,
            model_controls=0,
            neural_rows=0,
            gradients=0,
            optimizer_steps=0,
        ),
        source_commit=args.source_commit,
        config_sha256=args.config_sha256,
        original_failure_preserved=True,
    )
    reader = None
    try:
        reader = Reader(
            [run],
            limit=METADATA_READ_RESERVATION if args.mode == "metadata" else MAX_READ,
        )
        config_bytes = reader.read(run / "config.json")
        if sha_bytes(config_bytes) != args.config_sha256:
            raise ValueError("Collector configuration differs from reviewed deployment")
        config = json.loads(config_bytes)
        art = Path(config["locations"]["artifacts"]).resolve()
        if (
            config["source_commit"] != args.source_commit
            or str(run) != config["locations"]["run"]
            or not out.is_relative_to(art)
        ):
            raise ValueError(
                "Collector source/run/output differs from explicit frozen locations"
            )
        reader.roots.append(art)
        result.update(run_location=str(run), artifacts_location=str(art))
        result["closure"] = closure_gate(run, art, reader)
        prior_reads = 0
        if args.mode == "previews":
            if args.metadata_capsule is None:
                raise ValueError(
                    "Previews require the actual closed metadata capsule for shared read accounting"
                )
            metadata_root = args.metadata_capsule.resolve()
            metadata_bytes = reader.read(metadata_root / "capsule.json")
            metadata = json.loads(metadata_bytes)
            manifest_bytes = reader.read(metadata_root / "files_manifest.json")
            metadata_manifest = json.loads(manifest_bytes)
            metadata_exit = reader.json(metadata_root / "collection_exit.json")
            if (
                metadata_exit.get("passed") is not True
                or metadata_exit.get("manifest_sha256") != sha_bytes(manifest_bytes)
                or metadata.get("config_sha256") != args.config_sha256
                or metadata.get("source_commit") != args.source_commit
                or metadata_manifest["files"]["capsule.json"]["sha256"]
                != sha_bytes(metadata_bytes)
            ):
                raise ValueError(
                    "Preview shared accounting does not bind the closed J1 metadata capsule"
                )
            prior_reads = metadata["source_read_bytes"]
            if (
                type(prior_reads) is not int
                or not 0 <= prior_reads <= METADATA_READ_RESERVATION
            ):
                raise ValueError("Metadata source-read reservation is unavailable")
            reader.limit = MAX_READ - prior_reads
            if reader.read_bytes > reader.limit:
                raise ValueError(
                    "Shared J1 publication source-read reservation exhausted"
                )
        result["prior_source_read_bytes"] = prior_reads
        before = storage_census(run, art)
        reserved = MAX_COMPACT if args.mode == "metadata" else MAX_PREVIEW
        if (
            before["filesystem_free_bytes"] < reserved
            or before["total_bytes"] + reserved
            > config["limits"]["global_output_bytes"]
        ):
            raise ValueError(
                "J1 retained-output or physical capacity reservation is unavailable"
            )
        result["storage_before"] = before
        result.update(
            collect_metadata(run, art, out, reader, config)
            if args.mode == "metadata"
            else collect_previews(run, art, out, reader)
        )
        reader.unchanged()
        if live_study_processes(
            run, nonces=tuple(result["closure"]["checked_owner_nonces"])
        ):
            raise ValueError("A study process became active during collection")
        result["storage_after_before_closing_metadata"] = storage_census(run, art)
        result["passed"] = True
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(
            source_read_bytes=reader.read_bytes if reader else 0,
            source_read_limit_bytes=MAX_READ,
            total_shared_source_read_bytes=result.get("prior_source_read_bytes", 0)
            + (reader.read_bytes if reader else 0),
            wall_seconds=time.monotonic() - started,
            user_seconds=usage.ru_utime,
            system_seconds=usage.ru_stime,
            peak_rss_bytes=usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            limits=dict(
                wall_seconds=MAX_WALL,
                address_space_bytes=MAX_RSS,
                compact_bytes=MAX_COMPACT,
                ignored_preview_bytes=MAX_PREVIEW,
                paired_remote_compact_and_local_tracked_bytes=2 * MAX_COMPACT,
                total_compact_preview_receipt_reservation_bytes=128 * MIB,
            ),
        )
        new(out / "capsule.json", result)
        files = {}
        for path in sorted(out.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                files[str(path.relative_to(out))] = dict(
                    bytes=len(data), sha256=sha_bytes(data)
                )
        total = sum(v["bytes"] for v in files.values())
        final = dict(
            schema="walker-j1-compact-file-manifest-v1",
            files=files,
            bytes=total,
            status="COMPLETE_COLLECTION"
            if result["passed"]
            else "STOPPED_COLLECTION_PRESERVED",
            manifest_excludes_itself=True,
            transfer_method="Stream tar directly; no permanent duplicate archive. Verify per-file bytes and hashes after transfer.",
            source_read_bytes=result["source_read_bytes"],
            staged_verification_read_bytes=total,
            closing_receipt="collection_exit.json; final passed/status and manifest binding are recorded separately",
        )
        new(out / "files_manifest.json", final)
        final_total = total + (out / "files_manifest.json").stat().st_size
        limit = MAX_COMPACT if args.mode == "metadata" else MAX_PREVIEW
        passed = result["passed"] and final_total + 4096 <= limit
        closing = dict(
            schema="walker-j1-compact-collection-exit-v1",
            passed=passed,
            body_completed=result["passed"],
            bytes_before_exit=final_total,
            byte_limit=limit,
            manifest_sha256=sha_bytes((out / "files_manifest.json").read_bytes()),
            retries=0,
            exit_code=int(not passed),
            leader_completed=True,
            error=None
            if passed
            else result.get("error", "Final compact closing reservation unavailable"),
        )
        new(out / "collection_exit.json", closing)
        final_total += (out / "collection_exit.json").stat().st_size
        result["passed"] = passed
        print(
            json.dumps(
                dict(
                    passed=passed,
                    mode=args.mode,
                    out=str(out),
                    bytes=final_total,
                    files=len(files) + 2,
                    source_read_bytes=result["source_read_bytes"],
                    manifest_sha256=closing["manifest_sha256"],
                    collection_exit_sha256=sha_bytes(
                        (out / "collection_exit.json").read_bytes()
                    ),
                ),
                allow_nan=False,
            )
        )
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
