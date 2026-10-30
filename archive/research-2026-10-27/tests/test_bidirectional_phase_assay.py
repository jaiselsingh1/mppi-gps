"""Source/config/evidence gates only; never construct an engine or actor."""
from copy import deepcopy
import json

import pytest

from src.gps.bidirectional.phase_assay import (
    DEFAULT, _audit_binds, _function_hash, input_provenance, load_config, protocol,
)


def test_phase_assay_counts_only_two_new_episodes_and_charges_reused_pair():
    spec = protocol(load_config())
    assert spec["max_controls_per_episode"] == 58851944
    assert spec["max_new_controls"] == spec["reused_baseline_controls"] == 117703888
    assert spec["matched_pair_controls"] == 235407776
    assert spec["actual_new_episodes"] == spec["reused_baseline_episodes"] == 2
    assert spec["actor_rows"] == spec["optimizer_steps"] == 0
    assert spec["baseline_condition"] == "short_knots"


def test_phase_assay_rejects_retuned_or_expanded_configuration(tmp_path):
    config = json.loads(json.dumps(DEFAULT)); config["search"]["K"] += 1
    path = tmp_path/"config.json"; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fixed phase32"):
        load_config(path)
    with pytest.raises(ValueError, match="Undeclared"):
        protocol(config)


def test_missing_raw_audit_stops_before_backend_or_source_access(tmp_path):
    with pytest.raises(ValueError, match="audit is pending"):
        input_provenance(tmp_path/"absent-baseline", tmp_path/"absent-backend.json")
    baseline = tmp_path/"baseline"; baseline.mkdir()
    (baseline/"audit_001.json").write_text(json.dumps(dict(status="PASS", arithmetic_valid=False)))
    with pytest.raises(ValueError, match="ALL raw baseline"):
        input_provenance(baseline, tmp_path/"absent-backend.json")


def test_audit_must_bind_all_inputs_under_same_original_run_prefix():
    files = {"manifest.json": {"sha256": "m"}, "config.json": {"sha256": "c"},
             "evidence/short_knots/ledger.jsonl": {"sha256": "l"}}
    audit = dict(input_sha256={"/old/run/"+key: row["sha256"] for key, row in files.items()})
    _audit_binds(audit, files)
    wrong = deepcopy(audit); wrong["input_sha256"]["/old/run/config.json"] = "changed"
    with pytest.raises(ValueError, match="exact baseline"):
        _audit_binds(wrong, files)
    split = dict(input_sha256={"/old/run/manifest.json": "m", "/other/run/config.json": "c",
                              "/old/run/evidence/short_knots/ledger.jsonl": "l"})
    with pytest.raises(ValueError, match="exact baseline"):
        _audit_binds(split, files)


def test_audit_accepts_nested_raw_bindings_but_rejects_hash_conflicts():
    files = {"manifest.json": {"sha256": "m"}, "config.json": {"sha256": "c"},
             "evidence/short_knots/ledger.jsonl": {"sha256": "l"}}
    audit = dict(input_sha256={"/old/run/manifest.json": "m", "/old/run/config.json": "c"},
                 archives=[dict(input_sha256={"/old/run/evidence/short_knots/ledger.jsonl": "l"})])
    _audit_binds(audit, files)
    conflicting = deepcopy(audit)
    conflicting["archives"].append(dict(input_sha256={"/old/run/config.json": "changed"}))
    with pytest.raises(ValueError, match="Conflicting"):
        _audit_binds(conflicting, files)


def test_executed_function_ast_binding_ignores_inventory_comments_but_not_behavior(tmp_path):
    a = tmp_path/"a.py"; b = tmp_path/"b.py"; c = tmp_path/"c.py"
    a.write_text("def execute_episode(x):\n    return x+1\n")
    b.write_text("# inventory only\ndef execute_episode(x):\n    return x+1\n")
    c.write_text("def execute_episode(x):\n    return x+2\n")
    assert _function_hash(a, "execute_episode") == _function_hash(b, "execute_episode")
    assert _function_hash(a, "execute_episode") != _function_hash(c, "execute_episode")
