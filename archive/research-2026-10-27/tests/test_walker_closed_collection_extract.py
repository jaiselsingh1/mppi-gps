"""Synthetic archives only; never read the closed or live Walker study."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tarfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/"runs/2026-10-27_walker_owned_signal_001/validate_extract_closed_signal_001.py"
spec = importlib.util.spec_from_file_location("closed_signal_extractor_fixture", SCRIPT)
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


def encoded(value):
    return (json.dumps(value, sort_keys=True)+"\n").encode()


def h(data):
    return hashlib.sha256(data).hexdigest()


def fixture_archive(tmp_path, *, attack=None, exit_code=0):
    payload = {f"source/file{i:02d}.py":f"# inert synthetic source{i}\n".encode() for i in range(40)}
    manifest = encoded(dict(files={name:h(data) for name,data in payload.items()}))
    manifest_sha = h(manifest)
    payload.update({"manifest.json":manifest,
        "packet_received.json":encoded(dict(manifest_sha256=manifest_sha)),
        "launch_started.json":encoded(dict(pid=extractor.PID, root=extractor.ROOT,
            manifest_sha256=manifest_sha, proc_start_ticks=extractor.START_TICKS, boot_id=extractor.BOOT)),
        "launch_journal.jsonl":b'{"kind":"synthetic_launch"}\n',
        "execution.pid":str(extractor.PID).encode()+b"\n", "execution.log":b"synthetic inert log\n",
        "execution/status.json":encoded(dict(phase="COMPLETED_TENTATIVE_BLIND_CHECK"))})
    source = {name:dict(sha256=h(data),stat=dict(bytes=len(data), mode=stat.S_IFREG|0o644)) for name,data in payload.items()}
    directories = {name:dict(mode=stat.S_IFDIR|0o755) for name in (".", "source", "execution")}
    process = dict(pid=extractor.PID, expected_start_ticks=extractor.START_TICKS,
        boot_id=extractor.BOOT, present=False, terminal_basis="PID_ABSENT")
    closure = encoded(dict(schema="walker-owned-signal-closed-collection-v1",root=extractor.ROOT,
        source_manifest_sha256=manifest_sha, process_before=process,process_before_emission=process,process_after=process,
        all_source_tree_regular_files_preserved=True,failed_and_partial_tails_preserved_verbatim=True,
        stable_complete_hash_reads=2,complete_inventory_comparisons=2,outcome_verdict="NOT_ASSIGNED",
        new_simulator_controls=0,actor_inference_rows=0,fits=0,tests=0,file_count=len(source),
        source_bytes=sum(len(v) for v in payload.values()),source_files_matched=40,directory_count=3,
        closure_kind="COMPLETED_TENTATIVE_BLIND_CHECK",assay_terminal_status_recorded=True,
        assay_status_phase="COMPLETED_TENTATIVE_BLIND_CHECK",execution_directory_present=True,
        provenance={name:dict(sha256=h(payload[name]),bytes=len(payload[name])) for name in extractor.REQUIRED}))
    snapshot = encoded(dict(schema="walker-owned-signal-closure-inventory-v1",root=extractor.ROOT,
        source_manifest_sha256=manifest_sha,files=source,directories=directories,
        generated_files={"closure_record.json":dict(sha256=h(closure),bytes=len(closure))},
        inventory_source_file_count=len(source),inventory_source_bytes=sum(len(v) for v in payload.values()),verdict="NOT_ASSIGNED"))
    members = [(name,data,tarfile.REGTYPE) for name,data in sorted(payload.items())]
    members += [("closure_record.json",closure,tarfile.REGTYPE),("snapshot_manifest.json",snapshot,tarfile.REGTYPE)]
    if attack == "path":
        members.insert(0,("../escape",b"must not escape",tarfile.REGTYPE))
    elif attack == "symlink":
        members.insert(0,("source/linked",b"",tarfile.SYMTYPE))
    elif attack == "hash":
        members = [(name,b"tampered\n" if name=="source/file00.py" else data,kind) for name,data,kind in members]
    elif attack == "footer":
        members = members[:-1]
    archive = tmp_path/"synthetic.tar.gz"
    with tarfile.open(archive,"w:gz",format=tarfile.PAX_FORMAT) as tar:
        for name,data,kind in members:
            info=tarfile.TarInfo(name);info.size=len(data);info.type=kind
            if kind == tarfile.SYMTYPE:info.linkname="../../outside"
            tar.addfile(info,io.BytesIO(data))
    if attack == "gzip_trailer":
        archive.write_bytes(archive.read_bytes()[:-8])
    receipt = tmp_path/"ssh_receipt.json"
    receipt.write_bytes(encoded(dict(schema="walker-owned-signal-ssh-receipt-v1",exit_code=exit_code,
        archive_sha256=h(archive.read_bytes()),archive_bytes=archive.stat().st_size,
        collector_source_sha256=extractor.COLLECTOR_SHA,host="synthetic-host",command="no SSH was run")))
    return archive,receipt,manifest_sha


def test_valid_archive_promotes_once_without_overwriting(tmp_path):
    archive,receipt,manifest = fixture_archive(tmp_path)
    destination = tmp_path/"accepted"
    original = archive.read_bytes()
    result = extractor.validate_extract(archive,receipt,destination,expected_manifest_sha=manifest)
    assert result["accepted"] and result["status"] == "VALIDATED_ARCHIVE"
    assert result["source_files_matched"] == 40 and result["scientific_verdict"] == "NOT_ASSIGNED"
    assert (destination/"source/file00.py").read_bytes() == b"# inert synthetic source0\n"
    assert archive.read_bytes() == original and not list(tmp_path.glob("accepted.staging-*"))
    with pytest.raises(ValueError, match="never overwrite"):
        extractor.validate_extract(archive,receipt,destination,expected_manifest_sha=manifest)


@pytest.mark.parametrize("attack",["path","symlink","hash","footer","gzip_trailer"])
def test_bad_archive_retains_incomplete_staging_and_input(tmp_path,attack):
    archive,receipt,manifest = fixture_archive(tmp_path,attack=attack)
    original = archive.read_bytes(); destination=tmp_path/"accepted"
    with pytest.raises((ValueError,EOFError,tarfile.TarError)):
        extractor.validate_extract(archive,receipt,destination,expected_manifest_sha=manifest)
    assert not destination.exists() and archive.read_bytes() == original
    stages = list(tmp_path.glob("accepted.staging-*"))
    assert len(stages) == 1 and stages[0].is_dir() and not (tmp_path/"escape").exists()
    rows=[json.loads(line) for line in (tmp_path/"accepted.validation.jsonl").read_text().splitlines()]
    assert rows[-1]["status"] == "INCOMPLETE" and rows[-1]["accepted"] is False


def test_nonzero_ssh_exit_cannot_accept_even_well_formed_archive(tmp_path):
    archive,receipt,manifest = fixture_archive(tmp_path,exit_code=1)
    destination=tmp_path/"accepted"
    with pytest.raises(ValueError,match="successful SSH"):
        extractor.validate_extract(archive,receipt,destination,expected_manifest_sha=manifest)
    assert not destination.exists()
    assert json.loads((tmp_path/"accepted.validation.jsonl").read_text().splitlines()[-1])["status"] == "INCOMPLETE"
