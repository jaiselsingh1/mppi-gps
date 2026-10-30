"""Finite native phase-proposal assay with a paid, audited reused baseline.

No work occurs on import. Setup reads/hashes metadata and saved arrays only.
Only run_assay constructs a task, after all source/input/backend checks pass.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .phase_search import PhaseIndependentSearch, PhaseSearchConfig
from .search import SearchConfig
from .task import TaskConfig


ROOT = Path(__file__).resolve().parents[3]
CRITICAL = ("assets/acrobot.xml", "src/envs/acrobot.py", "src/envs/base.py",
            "src/envs/mujoco_env.py", "src/gps/bidirectional/task.py",
            "src/gps/bidirectional/search.py", "src/gps/bidirectional/backend.py",
            "src/gps/bidirectional/evidence.py")
WORKFLOW_FUNCTIONS = ("execute_episode", "root", "canonical", "identity", "summary")
COMPACT_NAMES = dict(baseline_audit="baseline_audit.json",
                     backend_report="backend/backend_qualification.json",
                     backend_manifest="backend/manifest.json")
DEFAULT = dict(schema="independent-phase-proposals-dev001", backend="warp",
               task=asdict(TaskConfig()), search=asdict(PhaseSearchConfig()),
               reset_seeds=[110100, 110101], planner_seeds=[30000, 30001])


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def _regular(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe relative input")
    result = Path(root)/path
    if result.is_symlink() or not result.is_file():
        raise ValueError("Missing regular input: "+str(result))
    return result


def _function_hash(path, name):
    tree = ast.parse(Path(path).read_text())
    matches = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(matches) != 1:
        raise ValueError("Missing/ambiguous source function: "+name)
    return hashlib.sha256(ast.dump(matches[0], include_attributes=False).encode()).hexdigest()


def load_config(path=None):
    expected = json.loads(json.dumps(DEFAULT))
    config = deepcopy(expected) if path is None else json.loads(Path(path).read_text())
    if config != expected:
        raise ValueError("Only the fixed phase32 comparison is declared")
    PhaseSearchConfig(**config["search"]); TaskConfig.from_dict(config["task"])
    return config


def protocol(config):
    if json.loads(json.dumps(config)) != json.loads(json.dumps(DEFAULT)):
        raise ValueError("Undeclared phase assay configuration")
    per_episode = 561*sum(min(144, r) for r in range(1, 801))+800
    return dict(schema=DEFAULT["schema"], reset_seeds=config["reset_seeds"], planner_seeds=config["planner_seeds"],
                phase_search=asdict(PhaseSearchConfig(**config["search"])), baseline_condition="short_knots",
                new_condition="independent_phases", max_controls_per_episode=per_episode,
                max_new_controls=2*per_episode, reused_baseline_controls=2*per_episode,
                matched_pair_controls=4*per_episode, actor_rows=0, optimizer_steps=0,
                actual_new_episodes=2, reused_baseline_episodes=2,
                shared_assessments="Backend qualification and full prior temporal acquisition remain separately paid ancestry",
                intervention="Only32 broad slots replaced after consuming original sampler draws; unchanged Gaussian/memory/mean/retention/cost/physics",
                stopping_rule="Exactly two new complete episodes, or stop on error; no retry, extra roots, retuning, guidance or actor training",
                interpretation="Post-negative development coverage assay; no horizon-gap, standalone-learning or efficiency claim",
                missing_witness="A strong complete long-independent-capture witness is still absent")


def _audit_binds(audit, files):
    # The whole-run auditor binds source/config at top level; each independent
    # archive retains its ledger/query/episode bindings in its nested report.
    inputs = {}
    for layer in [audit]+audit.get("archives", []):
        for key, value in layer.get("input_sha256", {}).items():
            if key in inputs and inputs[key] != value:
                raise ValueError("Conflicting baseline audit input hashes")
            inputs[key] = value
    manifest_sha = files["manifest.json"]["sha256"]
    prefixes = [str(Path(key).parent) for key, value in inputs.items()
                if Path(key).name == "manifest.json" and value == manifest_sha]
    if not any(all(inputs.get(str(Path(prefix)/relative)) == ref["sha256"]
                   for relative, ref in files.items()) for prefix in prefixes):
        raise ValueError("Audit does not bind the exact baseline source/config/ledger/raw packet")


def input_provenance(baseline_run, backend_report, *, baseline_audit=None):
    """Fail closed before construction; no model/actor/controller invocation."""
    baseline = Path(baseline_run).resolve(); backend_report = Path(backend_report).resolve()
    audit_path = baseline/"audit_001.json" if baseline_audit is None else Path(baseline_audit).resolve()
    if not audit_path.is_file():
        raise ValueError("Baseline raw arithmetic audit is pending; phase execution prohibited")
    audit = json.loads(audit_path.read_text())
    if (audit.get("status") != "PASS" or audit.get("arithmetic_valid") is not True
            or audit.get("errors") or audit.get("pending")):
        raise ValueError("Require PASS for ALL raw baseline conditions before phase execution")
    archives = audit.get("archives", [])
    if (len(archives) != 2 or {Path(a.get("archive", "")).name for a in archives}
            != {"short_knots", "independent_blocks"} or any(
                a.get("status") != "PASS" or a.get("arithmetic_valid") is not True
                or a.get("closed") is not True or a.get("errors") or a.get("pending") for a in archives)):
        raise ValueError("Require closed PASS for both original raw archives")
    files = {}

    def add(relative):
        path = _regular(baseline, relative)
        ref = dict(file=relative, sha256=_sha(path), bytes=path.stat().st_size)
        files[relative] = ref
        return path

    config = json.loads(add("config.json").read_text())
    manifest = json.loads(add("manifest.json").read_text())
    previous_protocol = json.loads(add("protocol.json").read_text())
    previous = json.loads(add("assay_result.json").read_text())
    add("stage_complete.json")
    if (_sha(baseline/"config.json") != manifest["config_sha256"]
            or config.get("schema") != "independent-temporal-proposals-dev001"
            or config.get("backend") != "warp" or config.get("task") != asdict(TaskConfig())
            or SearchConfig(**config["search"]) != SearchConfig()
            or previous.get("status") != "complete" or previous.get("source_unchanged") is not True
            or previous.get("actor_rows") != 0 or previous.get("optimizer_steps") != 0):
        raise ValueError("Baseline task/config/status differs from the fixed temporal assay")
    if (previous_protocol.get("reset_seeds") != DEFAULT["reset_seeds"]
            or previous_protocol.get("planner_seeds") != DEFAULT["planner_seeds"]
            or previous_protocol.get("max_controls") != 235407776
            or previous_protocol.get("conditions", {}).get("short_knots") != json.loads(json.dumps(asdict(SearchConfig())))):
        raise ValueError("Baseline reset/planner/proposal recipe differs")
    if set(previous.get("conditions", {})) != {"short_knots", "independent_blocks"}:
        raise ValueError("Require all four original temporal outcomes")
    if any(row.get("local") or row.get("capture") or not row.get("complete")
           for condition in previous["conditions"].values() for row in condition["outcomes"]):
        raise ValueError("This conditional assay requires the closed negative temporal outcome")
    source_map = {row["file"]: row["sha256"] for row in manifest["files"]}
    critical = {}
    for relative in CRITICAL:
        old = add("source/"+relative)
        expected = source_map.get("source/"+relative)
        if expected != _sha(old) or expected != _sha(ROOT/relative):
            raise ValueError("Baseline physics/search/source contract changed: "+relative)
        critical[relative] = expected
    old_workflow = add("source/src/gps/bidirectional/workflow.py")
    if _sha(old_workflow) != source_map.get("source/src/gps/bidirectional/workflow.py"):
        raise ValueError("Baseline workflow source changed")
    function_hashes = {}
    for name in WORKFLOW_FUNCTIONS:
        value = _function_hash(old_workflow, name)
        if value != _function_hash(ROOT/"src/gps/bidirectional/workflow.py", name):
            raise ValueError("Executed controller/helper function changed: "+name)
        function_hashes[name] = value
    for package in ("mujoco", "mujoco-warp", "warp-lang", "numpy", "scipy", "torch"):
        if manifest["dependency_versions"].get(package) != importlib.metadata.version(package):
            raise ValueError("Temporal runtime package changed: "+package)

    baselines = []
    audit_required = {name: ref for name, ref in files.items()
                      if name in ("config.json", "manifest.json") or name.startswith("source/")}
    for condition in ("short_knots", "independent_blocks"):
        relative = "evidence/"+condition+"/ledger.jsonl"
        ledger = add(relative)
        requests, completed, outcomes, episodes = {}, set(), {}, {}
        last = None
        for line in ledger.open():
            row = json.loads(line); last = row; kind = row["kind"]
            if kind == "query_requested":
                qid = row["query_id"]
                if qid in requests or row.get("task_hash") != TaskConfig().task_hash:
                    raise ValueError("Duplicate/wrong-task baseline query")
                requests[qid] = row["requested_controls"]
            elif kind == "query_complete":
                qid = row["query_id"]
                if qid not in requests or qid in completed or row["completed_controls"] != requests[qid]:
                    raise ValueError("Incomplete/duplicate baseline completion")
                completed.add(qid)
            elif kind in ("query_error", "query_rejected"):
                raise ValueError("Baseline has failed/rejected work")
            elif kind == "episode_saved":
                episodes[row["episode_id"]] = row
            elif kind == "episode_outcome":
                outcomes[row["context_id"]] = row
        accounting = previous["accounting"][condition]
        if (last is None or last["kind"] != "archive_close" or len(requests) != 4800
                or len(completed) != 4800 or sum(requests.values()) != 117703888
                or accounting != last.get("accounting")
                or any(accounting.get(k) != 0 for k in ("unknown_controls", "error_queries", "rejected_queries"))):
            raise ValueError("Baseline all-condition ledger is not complete/exact")
        audit_required[relative] = files[relative]
        for i, expected in enumerate(previous["conditions"][condition]["outcomes"]):
            context = "ordinary:"+str(DEFAULT["reset_seeds"][i])
            if (len(previous["conditions"][condition]["outcomes"]) != 2
                    or expected.get("context_id") != context or len(expected.get("query_ids", [])) != 800
                    or expected.get("task_hash") != TaskConfig().task_hash
                    or expected.get("actor_hash") is not None or not expected.get("deterministic")):
                raise ValueError("Wrong baseline root/owner/task/length")
            actual = outcomes.get(context)
            if actual is None or any(actual.get(k) != v for k, v in expected.items()):
                raise ValueError("Result not bound to baseline episode event")
            saved = episodes.get(expected["episode_id"])
            if saved is None:
                raise ValueError("Missing saved baseline episode")
            raw_relative = "evidence/"+condition+"/"+saved["raw"]["file"]
            raw_path = add(raw_relative)
            if files[raw_relative]["sha256"] != saved["raw"]["sha256"]:
                raise ValueError("Baseline episode bytes changed")
            audit_required[raw_relative] = files[raw_relative]
            with np.load(raw_path, allow_pickle=False) as raw:
                source = json.loads(str(raw["source_json"].item()))
                initial = json.loads(str(raw["initial_snapshot_json"].item()))
                if (source.get("reset_seed") != DEFAULT["reset_seeds"][i]
                        or source.get("planner_seed") != DEFAULT["planner_seeds"][i]
                        or initial.get("elapsed") != 0 or initial.get("context_id") != context
                        or initial.get("task_hash") != TaskConfig().task_hash
                        or not bool(raw["complete"].item()) or raw["actions"].shape != (800, 1)
                        or raw["query_ids"].tolist() != expected["query_ids"]):
                    raise ValueError("Baseline actual reset/planner/clock/trace recipe differs")
            if condition == "short_knots":
                baselines.append(expected)
    _audit_binds(audit, audit_required)
    backend = json.loads(backend_report.read_text())
    backend_manifest_path = backend_report.parent/"manifest.json"
    backend_manifest = json.loads(backend_manifest_path.read_text())
    if (not backend.get("qualified") or backend.get("task_hash") != TaskConfig().task_hash
            or backend.get("source_sha256") != critical["src/gps/bidirectional/backend.py"]
            or backend_manifest.get("dependency_versions") != manifest.get("dependency_versions")
            or backend_manifest.get("cuda_device") != manifest.get("cuda_device")):
        raise ValueError("Backend and temporal runtime/source gates do not match")
    return dict(schema="phase32-input-provenance-v1", baseline_run=str(baseline),
                files=list(files.values()), baseline_audit=dict(file=str(audit_path), sha256=_sha(audit_path), bytes=audit_path.stat().st_size),
                backend_report=dict(file=str(backend_report), sha256=_sha(backend_report), bytes=backend_report.stat().st_size),
                backend_manifest=dict(file=str(backend_manifest_path), sha256=_sha(backend_manifest_path), bytes=backend_manifest_path.stat().st_size),
                critical_source_sha256=critical, workflow_function_ast_sha256=function_hashes,
                baseline_outcomes=baselines, reused_baseline_controls=117703888,
                full_prior_temporal_controls=235407776, full_prior_temporal_wall_seconds=previous["wall_seconds"],
                backend_qualification_controls=backend.get("charged_requested_controls"),
                baseline_runtime=manifest["dependency_versions"], cuda_device=manifest["cuda_device"],
                no_model_or_actor_calls_in_setup=True, efficiency_claim=False)


def snapshot_inputs(provenance, out):
    out = Path(out); baseline = Path(provenance["baseline_run"])
    target = out/"inputs"; target.mkdir(exist_ok=False)
    for ref in provenance["files"]:
        source = _regular(baseline, ref["file"]); destination = target/"baseline"/ref["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if _sha(destination) != ref["sha256"] or destination.stat().st_size != ref["bytes"]:
            raise ValueError("Baseline changed while copying compact evidence")
    for name in ("baseline_audit", "backend_report", "backend_manifest"):
        ref = provenance[name]; destination = target/COMPACT_NAMES[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(ref["file"]).open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if _sha(destination) != ref["sha256"]:
            raise ValueError("Qualification input changed while copying")
    _write(out/"input_provenance.json", provenance)


def verify_inputs(out):
    out = Path(out); provenance = json.loads((out/"input_provenance.json").read_text())
    for ref in provenance["files"]:
        path = _regular(out/"inputs/baseline", ref["file"])
        if _sha(path) != ref["sha256"] or path.stat().st_size != ref["bytes"]:
            raise ValueError("Frozen baseline metadata changed")
    for name in ("baseline_audit", "backend_report", "backend_manifest"):
        if _sha(out/"inputs"/COMPACT_NAMES[name]) != provenance[name]["sha256"]:
            raise ValueError("Frozen qualification metadata changed")
    for relative, expected in provenance["critical_source_sha256"].items():
        if _sha(ROOT/relative) != expected:
            raise ValueError("Frozen current source no longer matches baseline")
    for name, expected in provenance["workflow_function_ast_sha256"].items():
        if _function_hash(ROOT/"src/gps/bidirectional/workflow.py", name) != expected:
            raise ValueError("Frozen executed workflow function changed: "+name)
    return provenance


def run_assay(config, out):
    from .evidence import Archive, QueryExecutor
    from .task import AcrobotCaptureTask
    from .workflow import execute_episode, root, summary
    from .temporal_assay import check_backend
    out = Path(out); started = time.perf_counter(); spec = protocol(config)
    provenance = verify_inputs(out)
    manifest = json.loads((out/"manifest.json").read_text())
    if not Path(__file__).resolve().is_relative_to(out/"source") or any(
            _sha(out/ref["file"]) != ref["sha256"] for ref in manifest["files"]):
        raise ValueError("Execute phase assay only from immutable frozen source")
    # This runtime/device check creates no task and performs no integration.
    check_backend(config, out/"inputs/backend/backend_qualification.json")
    _write(out/"protocol.json", spec)
    archive = Archive(out/"evidence/independent_phases"); task = None; outcomes = []
    try:
        task = AcrobotCaptureTask(TaskConfig.from_dict(config["task"]), backend=config["backend"])
        executor = QueryExecutor(task, archive, scope="independent_phases", max_requested_controls=spec["max_new_controls"])
        for i, (reset_seed, planner_seed) in enumerate(zip(config["reset_seeds"], config["planner_seeds"])):
            planner = PhaseIndependentSearch(executor, PhaseSearchConfig(**config["search"]), seed=planner_seed)
            _, _, outcome = execute_episode(executor, root(task, reset_seed, "ordinary"), f"independent_phases_root{i}_0000",
                planner=planner, source=dict(reset_seed=reset_seed, planner_seed=planner_seed,
                    reused_baseline_episode=provenance["baseline_outcomes"][i]["episode_id"],
                    representation="absolute-clock-phase32"))
            outcomes.append(outcome)
            _write(out/f"independent_phases_root{i}.json", outcome)
        if any(_sha(out/ref["file"]) != ref["sha256"] for ref in manifest["files"]):
            raise RuntimeError("Source changed during the finite assay")
    except BaseException as error:
        _write(out/"execution_failure.json", dict(error_type=type(error).__name__, error=str(error),
            complete_outcomes=outcomes, paid_accounting=archive.totals(), wall_seconds=time.perf_counter()-started))
        raise
    finally:
        archive.close()
        if task is not None:
            task.close()
    accounting = archive.totals()
    if accounting["requested_controls"] != spec["max_new_controls"] or accounting["unknown_controls"]:
        raise RuntimeError("Completed phase work differs from declared schedule")
    baseline = provenance["baseline_outcomes"]
    result = dict(schema=spec["schema"], status="complete", development_lead=sum(row["capture"] for row in outcomes) == 2,
        conditions=dict(short_knots=dict(**summary(baseline), outcomes=baseline, reused=True),
                        independent_phases=dict(**summary(outcomes), outcomes=outcomes, reused=False)),
        accounting=accounting, reused_baseline_controls=117703888, matched_pair_controls=235407776,
        unique_new_plus_full_prior_temporal_controls=accounting["requested_controls"]+provenance["full_prior_temporal_controls"],
        separately_paid_backend_qualification_controls=provenance["backend_qualification_controls"],
        original_task_qualified=False, standalone_learning_tested=False, efficiency_claim=False,
        actor_rows=0, optimizer_steps=0, source_unchanged=True, manifest_sha256=_sha(out/"manifest.json"),
        input_provenance_sha256=_sha(out/"input_provenance.json"), query_ledger_sha256=_sha(archive.ledger_path),
        wall_seconds=time.perf_counter()-started)
    _write(out/"assay_result.json", result)
    return result
