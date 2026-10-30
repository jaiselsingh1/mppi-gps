"""Checkpoint/launch provenance seams; no actors, forwards or simulator calls."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.gps.bidirectional.evidence import Archive
from src.gps.bidirectional.workflow import checkpoint, source_fingerprint


def manifest(path, *, dependencies=None, reverse=False):
    rows=[{"file":"source/a.py","sha256":"a"*64},{"file":"source/b.py","sha256":"b"*64}]
    if reverse:rows.reverse()
    (path/"manifest.json").write_text(json.dumps({"files":rows,
        "dependency_versions":dependencies or {"numpy":"frozen","torch":"frozen"}}))


def test_resume_source_identity_ignores_inventory_order_but_binds_dependencies(tmp_path):
    manifest(tmp_path);first=source_fingerprint(tmp_path)
    manifest(tmp_path,reverse=True);assert source_fingerprint(tmp_path)==first
    manifest(tmp_path,dependencies={"numpy":"changed","torch":"frozen"})
    assert source_fingerprint(tmp_path)!=first


def test_round_commit_is_create_only_and_binds_serialized_continuation_state(tmp_path):
    manifest(tmp_path)
    state={"config_hash":"config","seed":0,"next_round":1,
        "learners":{"independent_data":{"private_rng":"saved0"},"bidirectional":{"private_rng":"saved1"}},
        "families":{"independent_data":0,"bidirectional":1},
        "memories":{"independent_data":["fragment0"],"bidirectional":["fragment1"]},
        "corpora":{"independent_data":["trace0"],"bidirectional":["trace1"]},
        "completed_results":[{"seed":-1,"passed":False}]}
    with Archive(tmp_path/"evidence") as archive:
        path=checkpoint(tmp_path,"seed0_round0_committed",state,archive)
        restored=torch.load(path,map_location="cpu",weights_only=False)
        assert restored==state and restored["source_fingerprint"]==source_fingerprint(tmp_path)
        before=path.read_bytes()
        with pytest.raises(FileExistsError):checkpoint(tmp_path,"seed0_round0_committed",{},archive)
        assert path.read_bytes()==before
        rows=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
        recorded=[r for r in rows if r["kind"]=="checkpoint"]
        assert len(recorded)==1
        assert recorded[0]["sha256"]==hashlib.sha256(before).hexdigest()
        assert archive.totals()["requested_controls"]==0


def test_outer_launch_reexecutes_module_from_frozen_tree_with_absolute_inputs(tmp_path,monkeypatch):
    import scripts.training.train_bidirectional_gps as cli

    class ReexecutionCaptured(Exception):pass

    out=tmp_path/"new";corpus=tmp_path/"corpus";corpus.mkdir()
    (corpus/"manifest.json").write_text("{}\n")
    captured={}
    def fake_freeze(path,config,stage):
        path.mkdir();(path/"source").mkdir()
        (path/"config.json").write_text(json.dumps(config))
        captured["stage"]=stage
    def fake_exec(executable,command):
        captured.update(executable=executable,command=command)
        raise ReexecutionCaptured
    monkeypatch.setattr(cli,"freeze_source",fake_freeze)
    monkeypatch.setattr(cli.os,"chdir",lambda path:captured.update(cwd=Path(path)))
    monkeypatch.setattr(cli.os,"execv",fake_exec)
    monkeypatch.setattr(cli.sys,"argv",["train_bidirectional_gps","--stage","validate",
        "--out",str(out),"--corpus",str(corpus)])
    with pytest.raises(ReexecutionCaptured):cli.main()
    command=captured["command"]
    assert command[:3]==[captured["executable"],"-m","scripts.training.train_bidirectional_gps"]
    assert "--internal-frozen" in command and captured["cwd"]==out/"source"
    assert command[command.index("--config")+1]==str(out/"config.json")
    assert command[command.index("--corpus")+1]==str(corpus)
    assert not (out/"stage_complete.json").exists()
    assert not list(out.rglob("ledger.jsonl"))
