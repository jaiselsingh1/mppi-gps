"""Synthetic bytes only; no report, actor, simulator, or study input execution."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]/"runs/2026-10-27_walker_owned_signal_001/analysis_source_002/report_walker_owned_signal_v002.py"
spec = importlib.util.spec_from_file_location("signal_report_v002", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    data = b'{"kind":"synthetic"}\n'
    (tmp_path/"events.jsonl").write_bytes(data)
    inventory = {"schema": "walker-owned-signal-closure-inventory-v1",
        "root": module.CLOSURE_ROOT,
        "source_manifest_sha256": module.SOURCE_MANIFEST_SHA256,
        "files": {"execution/events.jsonl": {"sha256": hashlib.sha256(data).hexdigest(),
            "stat": {"bytes": len(data)}}}}
    path = tmp_path/"inventory.json"
    path.write_text(json.dumps(inventory))
    return data, path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_events_digest_has_explicit_inventory_binding(tmp_path):
    data, inventory, digest = fixture(tmp_path)
    provenance = {}
    assert module.read_reporting_input(tmp_path, {"input_sha256": {}}, "events.jsonl",
        provenance, inventory, digest) == data
    assert provenance["closure_inventory_sha256"] == digest
    assert list(provenance["missing_binding_fallback"]) == ["events.jsonl"]


def test_existing_audit_mismatch_never_falls_back(tmp_path):
    _, inventory, digest = fixture(tmp_path)
    with pytest.raises(ValueError, match="changed after audit"):
        module.read_reporting_input(tmp_path, {"input_sha256": {"events.jsonl": "0"*64}},
            "events.jsonl", {}, inventory, digest)


@pytest.mark.parametrize("case", ["no_inventory", "wrong_hash", "wrong_root", "changed_events", "missing_other"])
def test_fallback_fails_closed(tmp_path, case):
    _, inventory, digest = fixture(tmp_path)
    relative = "events.jsonl"
    if case == "no_inventory":
        inventory = None
    elif case == "wrong_hash":
        digest = "0"*64
    elif case == "wrong_root":
        value = json.loads(inventory.read_text()); value["root"] = "/wrong"
        inventory.write_text(json.dumps(value)); digest = hashlib.sha256(inventory.read_bytes()).hexdigest()
    elif case == "changed_events":
        (tmp_path/relative).write_bytes(b"modified")
    else:
        relative = "round.json"; (tmp_path/relative).write_bytes(b"{}")
    with pytest.raises(ValueError):
        module.read_reporting_input(tmp_path, {"input_sha256": {}}, relative, {}, inventory, digest)
