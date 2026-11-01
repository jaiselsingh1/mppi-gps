"""Bind the closed S1 fits for evaluation only, without loading model tensors."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .provenance import sha, write


def audit_known_reset_ledgers(repo: Path, seeds: list[int]) -> dict:
    """Read known saved ledgers conservatively; this is no global freshness claim.

    The helper intentionally uses only the standard library so the coordinator
    can send the identical census to the worker before preparing a frozen run.
    Any proposed integer in a ledger counts as a collision, even outside a
    field named ``seed``. Original interrupted prefixes are read as saved.
    """
    import hashlib
    import json
    import os
    import re
    from pathlib import Path

    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Explicit uint32 reset seeds are required")
    selected = set(seeds)
    possible_seed = re.compile(b"|".join(str(seed).encode("ascii") for seed in sorted(selected)))
    files, excluded_files, collisions, errors = [], [], [], []

    def visit(value, address, path):
        if type(value) is int and value in selected:
            collisions.append(dict(file=path, address=address, seed=value))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, address + "/" + str(key), path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, address + "/" + str(index), path)

    for seed in selected:
        if seeds.count(seed) > 1:
            collisions.append(dict(file=None, address="proposed_seeds", seed=seed))
    paths = set()
    for directory in ("runs", "artifacts", "results", "archive"):
        base = Path(repo).resolve() / directory
        for parent, directories, filenames in os.walk(base):
            directories[:] = [name for name in directories if name != "source"]
            for name in filenames:
                path = Path(parent) / name
                if "ledger" in name and path.suffix in (".json", ".jsonl"):
                    paths.add(path)
    for path in sorted(paths):
        try:
            digest, size, count = hashlib.sha256(), 0, 0
            binding = dict(file=str(path), sha256=None, bytes=None, records=None)
            synthetic_root = next(
                (
                    part
                    for part in path.relative_to(Path(repo).resolve()).parts[:-1]
                    if part in {"test_evidence", "pytest_tmp"}
                ),
                None,
            )
            if synthetic_root is not None:
                binding.update(
                    reason="explicit_synthetic_fixture_root", synthetic_root=synthetic_root
                )
                excluded_files.append(binding)
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                binding.update(sha256=digest.hexdigest(), bytes=size)
                continue
            files.append(binding)
            if path.suffix == ".jsonl":
                with path.open("rb") as stream:
                    for index, line in enumerate(stream):
                        digest.update(line)
                        size += len(line)
                        if line.strip():
                            try:
                                value = json.loads(line)
                                # JSON integers contain their decimal bytes. Still parse
                                # every record for validity, but traverse only candidates.
                                if possible_seed.search(line):
                                    visit(value, "/" + str(index), str(path))
                                count += 1
                            except ValueError as error:
                                errors.append(
                                    dict(
                                        file=str(path),
                                        line=index + 1,
                                        type=type(error).__name__,
                                        message=str(error),
                                    )
                                )
            else:
                raw = path.read_bytes()
                digest.update(raw)
                size = len(raw)
                visit(json.loads(raw), "/0", str(path))
                count = 1
            binding.update(sha256=digest.hexdigest(), bytes=size, records=count)
        except (OSError, ValueError) as error:
            errors.append(dict(file=str(path), type=type(error).__name__, message=str(error)))
    return dict(
        passed=not collisions and not errors,
        seeds=list(seeds),
        files=files,
        excluded_files=excluded_files,
        collisions=collisions,
        errors=errors,
        scope="Exact integer census of readable saved *ledger*.json/jsonl under runs, artifacts, results and archive; frozen source copies excluded. Candidate files under explicit synthetic fixture roots named test_evidence or pytest_tmp are excluded from reset evidence and retained separately with hashes and byte counts. Unknown, external and unrecorded resets are outside this proof.",
    )


def prepare_inputs(source: Path, out: Path, config: dict) -> dict:
    source = source.resolve()
    expected = config["source_bindings"]
    originals = {
        "config.json": "s1_config_sha256",
        "manifest.json": "s1_source_manifest_sha256",
        "input_manifest.json": "s1_input_manifest_sha256",
        "fit_result.json": "s1_fit_result_sha256",
    }
    for name, key in originals.items():
        if sha(source / name) != expected[key]:
            raise ValueError("S1.1 inherited input differs: " + name)
    manifest = json.loads((source / "manifest.json").read_text())
    seal = json.loads((source / "input_manifest.json").read_text())
    for row in manifest["files"]:
        if sha(source / "source" / row["file"]) != row["sha256"]:
            raise ValueError("S1.1 original frozen source differs")
    for name, binding in seal["files"].items():
        if sha(source / name) != binding["sha256"]:
            raise ValueError("S1.1 original sealed input differs")
    fit = json.loads((source / "fit_result.json").read_text())
    exit_record = json.loads((source / "fit_exit.json").read_text())
    if (
        not fit["execution_complete"]
        or fit["error"]
        or exit_record["exit_code"] != 0
        or exit_record["fit_result_sha256"] != expected["s1_fit_result_sha256"]
        or len(fit["points"]) != 9
        or len(fit["fits"]) != 6
    ):
        raise ValueError("S1.1 requires the exact closed six-fit phase")
    for name, binding in config["checkpoints"].items():
        if (
            sha(source / binding["file"]) != binding["sha256"]
            or fit["points"][name]["checkpoint"]["sha256"] != binding["sha256"]
            or fit["points"][name]["optimizer_fingerprint"] != binding["optimizer_fingerprint"]
        ):
            raise ValueError("S1.1 actor/checkpoint owner differs: " + name)
    if sha(source / config["reference"]["file"]) != expected["reference_checkpoint_sha256"]:
        raise ValueError("S1.1 reference bytes differ")
    out.mkdir(parents=True, exist_ok=False)
    inherited = out / "s1"
    inherited.mkdir()
    for name in (*originals, "fit_exit.json", "launch_receipt.json"):
        shutil.copy2(source / name, inherited / name)
    for name in ("source", "inputs", "learning", "checkpoints"):
        shutil.copytree(source / name, inherited / name)
    shutil.copytree(source / "evidence/fit", inherited / "evidence/fit")
    catalog_source = source.parent.parent / "artifacts" / source.name
    catalog = catalog_source / "local_full_run_file_manifest.json"
    if sha(catalog) != expected["s1_original_run_manifest_sha256"]:
        raise ValueError("S1.1 original interrupted run catalog differs")
    old_files = {row["file"]: row for row in json.loads(catalog.read_text())["files"]}
    for path in inherited.rglob("*"):
        if path.is_file():
            relative = str(path.relative_to(inherited))
            if sha(path) != old_files[relative]["sha256"]:
                raise ValueError("S1.1 copied inherited bytes differ: " + relative)
    history = out / "s1_interruption"
    history.mkdir()
    for path, name in (
        (catalog, "original_run_manifest.json"),
        (catalog_source / "audit_interruption_001/interruption_summary.json", "summary.json"),
        (catalog_source / "audit_interruption_001/receipt.json", "receipt.json"),
        (catalog_source / "full_audit_001/receipt.json", "full_audit_receipt.json"),
    ):
        shutil.copy2(path, history / name)
    write(
        out / "inventory.json",
        dict(
            schema="walker-student-source-s1.1-inputs-v1",
            source_run=str(source),
            original_closed_fit_sha256=expected["s1_fit_result_sha256"],
            original_run_manifest_sha256=expected["s1_original_run_manifest_sha256"],
            new_fits=0,
            new_reconstruction_rows=0,
            scope="Exact closed S1 source, inputs, fits and continuation packages; no original evaluation arrays copied or repaired.",
        ),
    )
    return dict(files=sum(path.is_file() for path in out.rglob("*")))
