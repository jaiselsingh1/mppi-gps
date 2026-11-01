"""Small H5 lifecycle fixtures require no ignored evidence or simulator."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.gps.mppi_gps.experiments import teacher_proposal as module
from src.gps.mppi_gps.inspection import inspect_run
from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.recipes import load_recipe
from src.gps.mppi_gps.reporting.teacher_proposal import render


def fixture(tmp_path, monkeypatch):
    config = load_recipe("teacher_proposal_h5_0")
    source = tmp_path / "source_owner"
    (source / "inputs").mkdir(parents=True)
    files = {}
    for label, binding in config["inputs"].items():
        path = source / binding["file"]
        # Intentionally not an NPZ: preparation must hash/copy, never decode.
        path.write_bytes(("synthetic-hash-only-" + label).encode())
        binding.update(bytes=path.stat().st_size, sha256=sha(path))
        files[binding["file"]] = sha(path)
    for name in ("config.json", "manifest.json", "launch_receipt.json"):
        (source / name).write_text("{}\n")
    (source / "input_manifest.json").write_text(json.dumps(dict(files=files)))
    config["input_owner"]["files"] = {
        name: dict(bytes=(source / name).stat().st_size, sha256=sha(source / name))
        for name in ("config.json", "manifest.json", "input_manifest.json", "launch_receipt.json")
    }
    probe = dict(
        **{key: config["runtime"][key] for key in ("host", "python", "platform", "distributions")},
        affinity=list(range(32)),
        storage=dict(free_bytes=1024**4),
        native_simulation_inference_or_fitting_calls=0,
    )
    probe_path = tmp_path / "runtime.json"
    probe_path.write_text(json.dumps(probe))
    monkeypatch.setattr(module, "load_recipe", lambda name: deepcopy(config))

    def freeze(out, recipe, command):
        out.mkdir()
        (out / "source").mkdir()
        (out / "source/tiny.py").write_text("# synthetic source\n")
        (out / "config.json").write_text(json.dumps(recipe))
        (out / "manifest.json").write_text(
            json.dumps(
                dict(
                    command=command,
                    files=[dict(file="tiny.py", sha256=sha(out / "source/tiny.py"))],
                    config_sha256=sha(out / "config.json"),
                    input_sha256=None,
                    policy_action_fitting=False,
                )
            )
        )

    monkeypatch.setattr(module, "freeze", freeze)
    seed_review = tmp_path / "seed_review.json"
    seed_review.write_text(
        json.dumps(dict(passed=True, metadata_only=True, seed_roster=module.seed_roster(config)))
    )
    return source, probe_path, config, seed_review


def test_create_only_frozen_preparation_hashes_inputs_and_reports_no_scientific_work(
    tmp_path, monkeypatch
):
    source, probe, config, seeds = fixture(tmp_path, monkeypatch)
    before = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    out = tmp_path / "prepared"
    result = module.prepare(out, source, probe, seeds)
    assert result["preparation_complete"] and not result["execution_complete"]
    assert result["work"] == dict(
        network_forwards=0, physics_controls=0, optimizer_steps=0, mppi_queries=0
    )
    assert result["launch_blockers"]
    for binding in config["inputs"].values():
        assert sha(out / binding["file"]) == binding["sha256"]
    assert before == {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    inspected = inspect_run(out, verify=True)
    assert (
        inspected["hash_mismatches"] == inspected["errors"] == inspected["missing_evidence"] == []
    )
    assert inspected["bulk_evidence"] == {}
    with pytest.raises(FileExistsError):
        module.prepare(out, source, probe, seeds)
    original = {str(p): sha(p) for p in out.rglob("*") if p.is_file()}
    receipt = render(out, tmp_path / "report")
    assert receipt["network_forwards"] == receipt["physics_controls"] == 0
    assert "prospective maxima" in (tmp_path / "report/report.md").read_text()
    assert original == {str(p): sha(p) for p in out.rglob("*") if p.is_file()}


@pytest.mark.parametrize("problem", ("input", "owner", "version", "free_space"))
def test_changed_input_owner_runtime_or_space_stops_before_creating_export(
    tmp_path, monkeypatch, problem
):
    source, probe, config, seeds = fixture(tmp_path, monkeypatch)
    if problem == "input":
        (source / config["inputs"]["train"]["file"]).write_bytes(b"different")
    elif problem == "owner":
        (source / "manifest.json").write_text("changed\n")
    else:
        data = json.loads(probe.read_text())
        if problem == "version":
            data["distributions"]["mujoco"] = "wrong-version"
        else:
            data["storage"]["free_bytes"] = 1
        probe.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.prepare(tmp_path / "out", source, probe, seeds)
    assert not (tmp_path / "out").exists()


def test_prepare_rejects_symlink_input_escape(tmp_path, monkeypatch):
    source, probe, config, seeds = fixture(tmp_path, monkeypatch)
    name = config["inputs"]["train"]["file"]
    original = (source / name).read_bytes()
    external = tmp_path / "outside"
    external.write_bytes(original)
    (source / name).unlink()
    (source / name).symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        module.prepare(tmp_path / "out", source, probe, seeds)


def test_eligibility_is_not_silently_cleared_by_an_implementation_or_metric(tmp_path):
    config = load_recipe("teacher_proposal_h5_0")
    config["storage_review"]["status"] = "bounded_array_stream_census_passed"
    assert len(module.launch_blockers(config)) == 2
    config["launch_authorized"] = True
    assert module.launch_blockers(config) == [
        "Weak-learner ordering exception has not been approved"
    ]
    config["eligibility"]["decision"] = "approved_single_round_exception"
    assert module.launch_blockers(config) == []
    config["eligibility"]["useful_learning_prerequisite_satisfied"] = True
    assert module.launch_blockers(config)


def test_unclosed_launch_is_unknown_work_instead_of_reusing_preparation_zeroes(
    tmp_path, monkeypatch
):
    source, probe, config, seeds = fixture(tmp_path, monkeypatch)
    out = tmp_path / "prepared"
    module.prepare(out, source, probe, seeds)
    (out / "worker_exit.json").write_text(json.dumps(dict(exit_code=-9)))
    view = inspect_run(out)
    assert view["outcomes"]["status"] == "launch_attempt_without_worker_result"
    assert view["costs"]["work"]["completion_unknown"]
    assert "worker_result.json" in view["missing_evidence"]
