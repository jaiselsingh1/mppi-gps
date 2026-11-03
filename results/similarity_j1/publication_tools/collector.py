"""J1 compact collector002: admit the bound repaired publication, retain failure001.

The sealed001 collector implements byte budgets, raw availability, compression
and create-only publication. This small extension changes only publication
namespace selection after actual repair-parent/render/report closure.
"""

from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path

BASE_SHA256 = "39fd627ed345916b5acf03ab326be91e91431226a2e5bdabf66bb9545a25053b"
path = Path(__file__).with_name("collector_base_001.py")
if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA256:
    raise ValueError("Original publication collector bytes changed")
spec = importlib.util.spec_from_file_location("j1_collector_base_001", path)
assert spec is not None and spec.loader is not None
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
Reader, MAX_PREVIEW, MAX_RSS, MAX_WALL, deadline = (
    base.Reader,
    base.MAX_PREVIEW,
    base.MAX_RSS,
    base.MAX_WALL,
    base.deadline,
)
prior_closure, prior_selection, prior_previews = (
    base.closure_gate,
    base.selected_files,
    base.collect_previews,
)


def clean(value: dict) -> bool:
    return (
        base.closed_receipt(value)
        and value["exit_code"] == 0
        and value.get("unknown_partial_work_possible") is False
        and value.get("process_group_kill_requested") is False
        and value.get("stopping_reason") is None
    )


def active_repair_script(script: bytes) -> bool:
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            if directory.stat().st_uid == os.getuid() and script in (
                directory / "cmdline"
            ).read_bytes().split(b"\0"):
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False


def repair_closed(run: Path, art: Path, reader: Reader) -> dict:
    root = art / "publication_repair_002/execution_001"
    # Close proof is checked before opening any new atlas/report outcome.
    launch, parent, result = (
        reader.json(root / name)
        for name in ("launch_receipt.json", "parent_exit.json", "repair_result.json")
    )
    owners = {
        name: reader.json(root / (name + "_ownership.json"))
        for name in ("coordinate", "render", "report")
    }
    render_exit, report_exit = (
        reader.json(art / "render_002/exit.json"),
        reader.json(root / "report_exit.json"),
    )
    if (
        not all(clean(value) for value in (parent, render_exit, report_exit))
        or launch.get("passed") is not True
        or result.get("passed") is not True
    ):
        raise ValueError(
            "Repaired publication has not actually closed cleanly; no002 selection permitted"
        )
    bindings = launch["bindings"]
    if (
        parent != launch.get("parent_exit")
        or parent["pid"] != result.get("parent_pid")
        or parent["command"] != owners["coordinate"]["command"]
        or result.get("bindings") != bindings
    ):
        raise ValueError(
            "Repair result does not bind the actual closed publication parent"
        )
    for name, owner in owners.items():
        if owner.get("bindings") != bindings or any(
            owner.get(key) != launch.get(key)
            for key in ("wrapper_sha256", "protocol_sha256")
        ):
            raise ValueError("Repair ownership code/protocol/input hashes differ")
        if name != "coordinate" and owner.get("parent_pid") != parent["pid"]:
            raise ValueError("Publication child belongs to another parent")
    if (
        render_exit["command"] != owners["render"]["command"]
        or report_exit["command"] != owners["report"]["command"]
    ):
        raise ValueError("Actual publication child argv differs from its owner")
    active = base.live_study_processes(
        run, nonces=tuple(row["nonce"] for row in owners.values())
    )
    if active:
        raise ValueError("A repaired publication child remains active")
    script = next(
        arg.encode() for arg in parent["command"] if arg.endswith("/repair.py")
    )
    if active_repair_script(script):
        raise ValueError("The repaired publication launcher has not exited")
    for key, name in (
        ("wrapper_sha256", "repair.py"),
        ("protocol_sha256", "protocol.json"),
    ):
        if hashlib.sha256(
            reader.read(art / "publication_repair_002" / name)
        ).hexdigest() != launch.get(key):
            raise ValueError("Closed repair implementation/protocol changed")
    paths = {
        "config_sha256": run / "config.json",
        "source_manifest_sha256": run / "manifest.json",
        "input_manifest_sha256": run / "input_manifest.json",
        "worker_result_sha256": run / "worker_result.json",
        "admission_sha256": run / "admission.json",
        "scientific_audit_sha256": art / "audit_001/result/audit.json",
        "failed_render001_atlas_sha256": art / "render_001/result/atlas_receipt.json",
        "failed_render001_coordinator_sha256": art / "render_001/coordinator_exit.json",
        "failed_render001_child_sha256": art / "render_001/exit.json",
        "failed_render001_source_accounting_sha256": art
        / "render_001/result/source_read_accounting.json",
        "renderer_preflight_sha256": run / "prepared/renderer.json",
    }
    for key, path in paths.items():
        if hashlib.sha256(reader.read(path)).hexdigest() != bindings.get(key):
            raise ValueError("Closed scientific/publication binding differs: " + key)
    admission, finalizer, audit = (
        reader.json(run / "admission.json"),
        reader.json(run / "finalizer_exit.json"),
        reader.json(paths["scientific_audit_sha256"]),
    )
    if (
        not clean(finalizer)
        or admission.get("finalizer_pid") != finalizer["pid"]
        or admission.get("scientific_audit_passed") is not True
        or audit.get("full_scientific_audit_passed") is not True
    ):
        raise ValueError("Original scientific audit/finalizer admission is unavailable")
    process, atlas, report = (
        reader.json(art / "render_002/result/process_receipt.json"),
        reader.json(art / "render_002/result/atlas_receipt.json"),
        reader.json(art / "report_002/report_receipt.json"),
    )
    if (
        process.get("worker_pid") != render_exit["pid"]
        or process.get("coordinator_pid") != parent["pid"]
        or process.get("bindings") != bindings
        or process.get("protocol_sha256") != launch["protocol_sha256"]
    ):
        raise ValueError(
            "Actual repaired renderer PID/input/protocol association differs"
        )
    atlas_sha = hashlib.sha256(
        reader.read(art / "render_002/result/atlas_receipt.json")
    ).hexdigest()
    report_sha = hashlib.sha256(
        reader.read(art / "report_002/report_receipt.json")
    ).hexdigest()
    if (
        atlas.get("passed") is not True
        or process.get("result_sha256") != atlas_sha
        or result.get("render_atlas_sha256") != atlas_sha
        or result.get("report_sha256") != report_sha
        or report.get("saved_only") is not True
    ):
        raise ValueError(
            "Repaired renderer/reporter results do not bind the actual clean children"
        )
    manifest = reader.json(run / "manifest.json")
    reporter_sha = next(
        row["sha256"]
        for row in manifest["files"]
        if row["file"] == "src/gps/mppi_gps/reporting/similarity.py"
    )
    if report.get("source_sha256") != reporter_sha:
        raise ValueError("A changed reporter produced publication002")
    account = reader.json(art / "render_002/result/source_read_accounting.json")
    old_account = reader.json(art / "render_001/result/source_read_accounting.json")
    roster_sha = hashlib.sha256(
        json.dumps(
            account["actual_roster"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if (
        account["actual_roster"] != old_account["actual_roster"]
        or len(account["actual_roster"]) != 96
        or roster_sha != bindings["roster_sha256"]
        or account["codec"]
        != dict(
            traces=96,
            done_bytes=account["codec"]["done_bytes"],
            input_dtype="uint8",
            output_dtype="bool",
            other_arrays_unchanged=True,
        )
    ):
        raise ValueError(
            "Repaired publication changed the original roster or undeclared inputs"
        )
    if (
        account["source_read_bytes"] > account["source_read_limit_bytes"]
        or account["source_read_limit_bytes"] + account["gate_read_reservation_bytes"]
        > 2 * 1024**3
        or atlas["frames"] > 64000
    ):
        raise ValueError(
            "Repaired rendering exceeded its original finite read/frame reservation"
        )
    for key in (
        "native_controls",
        "physics_substeps",
        "neural_rows",
        "optimizer_steps",
    ):
        if atlas["work"].get(key) != 0:
            raise ValueError("Publication repair contains unapproved scientific work")
    prior_seconds = {
        phase: reader.json(art / "operator_001" / phase / "operator_receipt.json")[
            "stage_wall_seconds"
        ]
        for phase in ("launch", "render-launch")
    }
    if (
        prior_seconds != launch["prior_stage_seconds"]
        or sum(prior_seconds.values()) + launch["stage_wall_seconds"] > 28800
    ):
        raise ValueError(
            "Repair did not preserve the shared original eight-hour wall census"
        )
    total_render_bytes = sum(
        path.stat().st_size
        for root in (art / "render_001", art / "render_002")
        for path in root.rglob("*")
        if path.is_file()
    )
    if total_render_bytes > 2 * 1024**3:
        raise ValueError("Original plus repaired rendering exceeded two GiB")
    return dict(
        passed=True,
        bindings=bindings,
        parent_pid=parent["pid"],
        render_pid=render_exit["pid"],
        report_pid=report_exit["pid"],
        nonces=[owner["nonce"] for owner in owners.values()],
        actual_shared_worker_render_report_seconds=sum(prior_seconds.values())
        + launch["stage_wall_seconds"],
        combined_render_bytes=total_render_bytes,
        failed001_preserved=True,
    )


def closure_gate(run: Path, art: Path, reader: Reader) -> dict:
    original = prior_closure(run, art, reader)
    repaired = repair_closed(run, art, reader)
    original["publication_repair_002"] = repaired
    original["checked_owner_nonces"].extend(repaired["nonces"])
    return original


def selected_files(run: Path, art: Path, config: dict):
    files, missing = prior_selection(run, art, config)

    def historical(name: str) -> str:
        if name.startswith("atlas/"):
            return "failed_render_001/" + name[len("atlas/") :]
        if name.startswith("report/"):
            return "prior_report_001/" + name[len("report/") :]
        return name

    result = [(path, historical(name)) for path, name in files]
    missing = [historical(name) for name in missing]

    def retain(path: Path, name: str):
        if path.is_file():
            result.append((path, name))
        else:
            missing.append(name)

    for name in (
        "exit.json",
        "stdout_stderr.txt",
        "result/atlas_receipt.json",
        "result/process_receipt.json",
        "result/source_read_accounting.json",
        "result/index.html",
    ):
        retain(art / "render_002" / name, "atlas/" + name)
    for path in sorted((art / "report_002").glob("*")):
        if path.suffix in {".json", ".md", ".csv", ".png"}:
            retain(path, "report/" + path.name)
    for path in sorted((art / "publication_repair_002").rglob("*")):
        if (
            "__pycache__" not in path.parts
            and path.is_file()
            and path.suffix in {".json", ".md", ".py", ".txt"}
        ):
            retain(
                path,
                "publication_repair_002/"
                + str(path.relative_to(art / "publication_repair_002")),
            )
    unique = dict((name, path) for path, name in result)
    return [(path, name) for name, path in sorted(unique.items())], sorted(set(missing))


class RepairedAtlasRoot:
    """Confined namespace adapter for the sealed original fixed-role file reader."""

    def __init__(self, root: Path):
        self.root = root

    def __truediv__(self, relative: str) -> Path:
        if not relative.startswith("render_001/"):
            raise ValueError("Unexpected original atlas namespace access")
        return self.root / ("render_002/" + relative[len("render_001/") :])


def collect_previews(run: Path, art: Path, out: Path, reader: Reader) -> dict:
    return prior_previews(run, RepairedAtlasRoot(art), out, reader)


base.closure_gate = closure_gate
base.selected_files = selected_files
base.collect_previews = collect_previews


if __name__ == "__main__":
    raise SystemExit(base.main())
