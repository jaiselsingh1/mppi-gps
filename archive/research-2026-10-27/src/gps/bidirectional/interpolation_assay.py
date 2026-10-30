"""Finite outcome-qualified interpolation of the closed precision assay's actors.

No work occurs on import. This tests step acceptance for existing actor
proposals, with no optimizer, model query, new teacher, or deployment claim.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time


CHECKPOINTS = {
    "parent": ("precision_precision_parent", "156b79fe77d5be05ffcd87f2efed1b00d9caf8d8d62a19eedb40c872f520af9f"),
    "raw": ("precision_precision_child_raw", "0d9bec3ea8df352562462f50caf956e6c04ddb6a27a15a468517718c806c56ff"),
    "guided": ("precision_precision_child_guided", "ef85d71ce6c53c1ceb33ccd339356d41158e3c01b7c4ebb772b4fc6fa2c19253"),
}
DEFAULT = dict(schema="bidirectional-ending-interpolation-assay-v0",
    ancestor_schema="bidirectional-ending-precision-assay-v0", source_arm="precision",
    task=dict(gate=12, delay=12, move=.2, ending_y=-.1, effort=.02, greed=.2,
              detour=.2, detour_steps=5, penalty=20.),
    alphas=[.0625, .125, .25, .5, 1.],
    interpolation="all_trainable_float_parameters_including_log_std;unchanged_buffers;parent_private_rng",
    qualification_roots=16, qualification_seed=151000, qualification_noise_seed=153000,
    evaluation_roots=64, evaluation_seed=152000, evaluation_noise_seed=154000,
    minimum_complete_gain=.001, capture_min=.9,
    max_controls=21600, max_actor_rows=21600, max_adam_steps=0,
    max_model_controls=0)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def load_config(path=None):
    config = deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    if config != DEFAULT:
        raise ValueError("Only the fixed interpolation assay is accepted")
    return config


def work_bound(config=None):
    config = DEFAULT if config is None else config
    if config != DEFAULT:
        raise ValueError("Undeclared interpolation assay")
    qualification = (1+2*len(config["alphas"]))*config["qualification_roots"]*25*2
    final = 4*config["evaluation_roots"]*25*2
    return dict(qualification_controls=qualification, final_controls=final,
        total_controls=qualification+final, executed_controls=qualification+final,
        actor_rows=qualification+final, model_controls=0, adam_steps=0,
        scope="All fixed qualification candidates are paid; final panels only if a candidate qualifies; no retries or reallocation")


def _safe_source(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe ancestor reference")
    path = Path(root)/relative
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing/nonregular ancestor input: "+str(path))
    return path


def _file_ref(root, relative):
    path = _safe_source(root, relative)
    return dict(file=str(relative), sha256=_sha(path), bytes=path.stat().st_size)


def input_provenance(parent_run):
    """Bind exact closed inputs without loading a policy or running a query."""
    root = Path(parent_run).resolve()
    names = ["manifest.json", "config.json", "assay_result.json", "input_provenance.json",
             "evidence/assay/ledger.jsonl", "work_ledger.jsonl"]
    names += ["checkpoints/"+label+".pt" for label, _ in CHECKPOINTS.values()]
    names += ["inputs/manifest.json", "inputs/config.json", "inputs/pilot_result.json"]
    refs = [_file_ref(root, name) for name in names]
    files = {row["file"]: row for row in refs}
    result = json.loads((root/"assay_result.json").read_text())
    config = json.loads((root/"config.json").read_text())
    manifest = json.loads((root/"manifest.json").read_text())
    ancestor = json.loads((root/"input_provenance.json").read_text())
    if (config.get("schema") != DEFAULT["ancestor_schema"] or result.get("schema") != config["schema"]
            or result.get("stage") != "failed_precision_child_improvement" or result.get("passed") is not False):
        raise ValueError("Require the exact closed negative precision assay")
    if config["task"] != DEFAULT["task"]:
        raise ValueError("Ancestor task differs")
    for name, field in (("manifest.json", "manifest_sha256"),
                        ("input_provenance.json", "input_provenance_sha256"),
                        ("evidence/assay/ledger.jsonl", "query_ledger_sha256"),
                        ("work_ledger.jsonl", "work_ledger_sha256")):
        if files[name]["sha256"] != result[field]:
            raise ValueError("Closed precision result binding changed: "+name)
    if files["config.json"]["sha256"] != manifest["config_sha256"]:
        raise ValueError("Ancestor frozen config changed")
    critical = {}
    for relative in ("src/ending/core.py", "src/gps/bidirectional/ending_pilot.py",
                     "src/gps/bidirectional/policy.py"):
        matches = [r for r in manifest["files"] if r["file"] == "source/"+relative]
        current = Path(__file__).resolve().parents[3]/relative
        if (len(matches) != 1 or _sha(_safe_source(root, "source/"+relative)) != matches[0]["sha256"]
                or _sha(current) != matches[0]["sha256"]):
            raise ValueError("Actor/dynamics/observation contract changed: "+relative)
        critical[relative] = matches[0]["sha256"]
    checkpoints = {}; last = None
    expected = {label: role for role, (label, _) in CHECKPOINTS.items()}
    with (root/"evidence/assay/ledger.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line); last = row
            if row.get("kind") == "checkpoint" and row.get("label") in expected:
                if row["label"] in checkpoints:
                    raise ValueError("Duplicate ancestor checkpoint")
                checkpoints[row["label"]] = row
    if last is None or last.get("kind") != "archive_close" or set(checkpoints) != set(expected):
        raise ValueError("Require clean closed precision ledger and all three checkpoints")
    for role, (label, expected_sha) in CHECKPOINTS.items():
        if files["checkpoints/"+label+".pt"]["sha256"] != expected_sha or checkpoints[label]["sha256"] != expected_sha:
            raise ValueError("Wrong fixed precision checkpoint: "+role)
    ancestor_files = {r["file"]: r for r in ancestor["files"]}
    for name in ("manifest.json", "config.json", "pilot_result.json"):
        ref = files["inputs/"+name]
        if ref["sha256"] != ancestor_files[name]["sha256"] or ref["bytes"] != ancestor_files[name]["bytes"]:
            raise ValueError("V3 paid ancestry metadata changed")
    v3 = json.loads((root/"inputs/pilot_result.json").read_text())
    if v3["accounting"] != ancestor["reused_cost_ancestry"] or v3["manifest_sha256"] != files["inputs/manifest.json"]["sha256"]:
        raise ValueError("V3 cost ancestry binding differs")
    return dict(schema="ending-interpolation-input-provenance-v1", parent_run=str(root), files=refs,
        checkpoints={role: dict(label=label, sha256=digest, original_event=checkpoints[label])
                     for role, (label, digest) in CHECKPOINTS.items()},
        critical_source_sha256=critical,
        precision_paid_ancestry=result["accounting"], precision_wall_seconds=result["assay_wall_seconds"],
        v3_paid_ancestry=ancestor["reused_cost_ancestry"], v3_wall_seconds=ancestor["reused_wall_seconds"],
        relationship="V3 plus precision assay are one shared paid lineage, counted once; candidate attributions overlap and are not independent discovery.",
        arithmetic_audit="Hash-bound closed source/metrics here; independent raw arithmetic audit remains separate",
        no_actor_forward_or_query_in_setup=True, efficiency_claim=False)


def snapshot_inputs(parent_run, out):
    provenance = input_provenance(parent_run); root = Path(provenance["parent_run"])
    target = Path(out)/"inputs"; target.mkdir(exist_ok=False)
    for row in provenance["files"]:
        destination = target/row["file"]; destination.parent.mkdir(parents=True, exist_ok=True)
        with _safe_source(root, row["file"]).open("rb") as src, destination.open("xb") as dst:
            for block in iter(lambda: src.read(1024*1024), b""):
                dst.write(block)
            dst.flush(); os.fsync(dst.fileno())
        if _sha(destination) != row["sha256"] or destination.stat().st_size != row["bytes"]:
            raise ValueError("Ancestor changed while copying; preserve failed preparation")
    _write(Path(out)/"input_provenance.json", provenance)
    return provenance


def verify_inputs(out):
    out = Path(out); value = json.loads((out/"input_provenance.json").read_text())
    expected = {role: digest for role, (_, digest) in CHECKPOINTS.items()}
    if {r: v["sha256"] for r, v in value["checkpoints"].items()} != expected:
        raise ValueError("Wrong pinned interpolation inputs")
    for row in value["files"]:
        path = _safe_source(out/"inputs", row["file"])
        if _sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Frozen ancestor input changed")
    return value


def interpolate_actor(parent, child, alpha):
    """No forward: interpolate parameters, preserve buffers and parent RNG.

    Parameter arithmetic uses float64 then serializes to the original dtype.
    Endpoints are copied exactly. Optimizer state is never constructed/used.
    """
    import torch
    from .policy import FamilyGaussianActor
    if type(alpha) not in (float, int) or isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Finite interpolation fraction in [0,1] required")
    before = parent.policy_state(); proposed = child.policy_state()
    for key in ("format_version", "policy_type", "obs_dim", "hidden_dim", "num_families", "action_low", "action_high"):
        if before[key] != proposed[key]:
            raise ValueError("Actor architecture/action metadata differs: "+key)
    parent_parameters = dict(parent.named_parameters()); child_parameters = dict(child.named_parameters())
    if set(parent_parameters) != set(child_parameters):
        raise ValueError("Parameter names differ")
    parent_buffers = dict(parent.named_buffers()); child_buffers = dict(child.named_buffers())
    if set(parent_buffers) != set(child_buffers):
        raise ValueError("Buffer names differ")
    for name, buffer in parent_buffers.items():
        other = child_buffers[name]
        if buffer.dtype != other.dtype or buffer.shape != other.shape or not torch.equal(buffer, other):
            raise ValueError("Buffer changed: "+name)
    output = FamilyGaussianActor.from_checkpoint(before)
    with torch.no_grad():
        for name, parameter in output.named_parameters():
            a, b = parent_parameters[name], child_parameters[name]
            if (not a.requires_grad or not b.requires_grad or not a.is_floating_point() or not b.is_floating_point()
                    or a.dtype != b.dtype or a.shape != b.shape or not bool(torch.isfinite(a).all())
                    or not bool(torch.isfinite(b).all())):
                raise ValueError("Require matching finite trainable float parameter: "+name)
            value = a if alpha == 0 else b if alpha == 1 else (a.double()+float(alpha)*(b.double()-a.double())).to(a.dtype)
            parameter.copy_(value)
    if not torch.equal(output._generator.get_state(), parent._generator.get_state()):
        raise RuntimeError("Interpolation changed the parent private sampling stream")
    return output


def _comparison(base, proposed, minimum, deterministic):
    if len(base) == 0 or len(base) != len(proposed) or [r.context_id for r in base] != [r.context_id for r in proposed]:
        raise ValueError("Require paired nonempty complete root panels")
    if len({r.context_id for r in base}) != len(base):
        raise ValueError("Duplicate root in panel")
    for row in list(base)+list(proposed):
        if not row.complete or row.deterministic is not deterministic or not math.isfinite(row.cost):
            raise ValueError("Require finite complete actual outcomes in the declared mode")
    gains = [a.cost-b.cost for a, b in zip(base, proposed)]
    mean_gain = math.fsum(gains)/len(gains)
    nonregression = sum(r.capture for r in proposed) >= sum(r.capture for r in base)
    return dict(mean_cost_gain=mean_gain, paired_cost_gains=gains,
        capture_nonregression=bool(nonregression), improved=bool(nonregression and mean_gain >= minimum))


def assess_panels(deterministic, stochastic, minimum=.001, capture_min=.9):
    if ("best_raw" in deterministic) != ("best_raw" in stochastic):
        raise ValueError("Best-raw control must be present in both modes")
    references = ("parent", "raw", "best_raw") if "best_raw" in deterministic else ("parent", "raw")
    benefit = {name: _comparison(deterministic[name], deterministic["guided"], minimum, True)
               for name in references}
    stochastic_benefit = {name: _comparison(stochastic[name], stochastic["guided"], minimum, False)
                         for name in references}
    capture = sum(r.capture for r in deterministic["guided"])/len(deterministic["guided"])
    return dict(passed=bool(capture >= capture_min and all(r["improved"] for r in
        list(benefit.values())+list(stochastic_benefit.values()))),
        deterministic_guided_vs=benefit, stochastic_guided_vs=stochastic_benefit,
        deterministic_capture_rate=capture)


def assess_raw(deterministic, stochastic, minimum=.001, capture_min=.9):
    """Independently qualify a raw-data step against its own common parent."""
    mean = _comparison(deterministic["parent"], deterministic["raw"], minimum, True)
    noisy = _comparison(stochastic["parent"], stochastic["raw"], minimum, False)
    capture = sum(r.capture for r in deterministic["raw"])/len(deterministic["raw"])
    return dict(passed=bool(capture >= capture_min and mean["improved"] and noisy["improved"]),
        deterministic_raw_vs_parent=mean, stochastic_raw_vs_parent=noisy,
        deterministic_capture_rate=capture)


def choose_alpha(rows):
    """Qualification-only choice; stable smaller-alpha tie, no final feedback."""
    if [row["alpha"] for row in rows] != DEFAULT["alphas"]:
        raise ValueError("All fixed alpha rows must be retained in order")
    eligible = [r for r in rows if r["criteria"]["passed"]]
    if any(not math.isfinite(r["guided_mean_complete_cost"]) for r in rows):
        raise ValueError("Nonfinite qualification score")
    return None if not eligible else min(eligible, key=lambda r: (r["guided_mean_complete_cost"], r["alpha"]))["alpha"]


def choose_raw_alpha(rows):
    """A distinct raw-data search control; None means explicit parent fallback."""
    if [row["alpha"] for row in rows] != DEFAULT["alphas"]:
        raise ValueError("All fixed alpha rows must be retained in order")
    eligible = [r for r in rows if r["raw_criteria"]["passed"]]
    if any(not math.isfinite(r["raw_mean_complete_cost"]) for r in rows):
        raise ValueError("Nonfinite raw qualification score")
    return None if not eligible else min(eligible, key=lambda r: (r["raw_mean_complete_cost"], r["alpha"]))["alpha"]


def _summary(rows):
    return dict(capture_rate=math.fsum(float(r.capture) for r in rows)/len(rows),
        mean_cost=math.fsum(r.cost for r in rows)/len(rows), outcomes=[asdict(r) for r in rows])


def run_assay(config, out):
    """Actual execution only from frozen source after all setup bindings pass."""
    if config != DEFAULT:
        raise ValueError("Undeclared interpolation assay")
    bound = work_bound(config)
    if (bound["total_controls"], bound["actor_rows"], bound["adam_steps"], bound["model_controls"]) != (
            config["max_controls"], config["max_actor_rows"], config["max_adam_steps"], config["max_model_controls"]):
        raise ValueError("Schedule and fixed caps differ")
    out = Path(out).resolve(); manifest = json.loads((out/"manifest.json").read_text())
    if (not Path(__file__).resolve().is_relative_to(out/"source")
            or _sha(out/"config.json") != manifest["config_sha256"]
            or any(_sha(out/r["file"]) != r["sha256"] for r in manifest["files"])):
        raise ValueError("Run only frozen assay source/config")
    provenance = verify_inputs(out)
    import numpy as np
    import torch
    from scripts.diagnostics.run_bidirectional_tests import DurableWork, WorkPlugin
    from .evidence import Archive
    from .policy import FamilyGaussianActor, policy_hash, controller_hash
    from .meter import attach_meter
    from .ending_pilot import CountedEnding, EndingContext, EndingExecutor, _actor_panel
    from .workflow import checkpoint

    start = time.perf_counter(); archive = Archive(out/"evidence"/"assay")
    archive.work_scope = "setup"; archive.budget_failed = False; archive.query_failed = False
    work = DurableWork(out/"work_ledger.jsonl"); plugin = WorkPlugin(work)
    result = dict(schema=config["schema"], stage="setup", passed=False, negative_retained=True,
        source_arm="precision", qualification=[], selected_alpha=None, raw_selected_alpha=None,
        no_training=True, no_model_queries=True, no_deployment_promotion=True,
        full_feedback_established=False, efficiency_claim=False)
    actor_rows = 0; installed = False; dynamics = None

    def phase(name):
        archive.work_scope = name; work.current_test = name; archive.append("assay_phase", phase=name)

    original_before = plugin._before
    def bounded_before(module, args):
        nonlocal actor_rows
        if isinstance(module, FamilyGaussianActor):
            shape = np.shape(args[0]); rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
            if actor_rows+rows > config["max_actor_rows"]:
                archive.budget_failed = True
                work.append("neural_rejected", rows=rows, reason="fixed_actor_row_cap", before_forward=True)
                raise RuntimeError("Fixed actor-row cap exhausted before forward")
            actor_rows += rows
        original_before(module, args)
    plugin._before = bounded_before

    def reject_adam(*args, **kwargs):
        archive.budget_failed = True
        work.append("optimizer_rejected", updates=1, reason="no_optimizer_in_assay", before_optimizer=True)
        raise RuntimeError("Optimizer work is prohibited in interpolation assay")

    def evaluate(actor, seeds, tag, deterministic, noise_seed):
        # Full checkpoint precedes every actual owner version's first forward.
        attach_meter(actor, archive, tag)
        rows = _actor_panel(executor, actor, seeds, tag, deterministic=deterministic, stochastic_seed=noise_seed)
        actor_id = policy_hash(actor); cid = controller_hash(actor, deterministic=deterministic, family=0)
        if any(r.actor_hash != actor_id or r.controller_hash != cid or r.semantics != "actor_suffix"
               or not r.complete or r.remaining_steps != 25 or r.deterministic is not deterministic for r in rows):
            raise RuntimeError("Complete actor-only outcome owner differs")
        return rows

    try:
        plugin.install(); installed = True; torch.optim.Adam.step = reject_adam
        dynamics = CountedEnding(archive, work, config["max_controls"], **config["task"])
        executor = EndingExecutor(EndingContext(dynamics), archive)
        actors = {}
        for role, (label, expected_sha) in CHECKPOINTS.items():
            path = out/"inputs/checkpoints"/(label+".pt")
            if _sha(path) != expected_sha:
                raise ValueError("Fixed checkpoint bytes changed before actor load")
            state = torch.load(path, map_location="cpu", weights_only=False)
            actors[role] = FamilyGaussianActor.from_checkpoint(state["learner"]["policy"])
        source_hashes = {name: policy_hash(actor) for name, actor in actors.items()}
        checkpoint(out, "interpolation_source_parent", dict(actor=actors["parent"].policy_state(),
            source_checkpoint_sha256=CHECKPOINTS["parent"][1], promotion_claim=False), archive)
        archive.append("input_reuse", provenance_sha256=_sha(out/"input_provenance.json"),
            source_actor_hashes=source_hashes, precision_paid_ancestry=provenance["precision_paid_ancestry"],
            v3_paid_ancestry=provenance["v3_paid_ancestry"], efficiency_claim=False)
        candidates = {}
        # Construct/save every fixed candidate before querying any qualification outcome.
        for alpha in config["alphas"]:
            label = "a"+str(alpha).replace(".", "p")
            candidates[alpha] = {}
            for role in ("raw", "guided"):
                candidate = interpolate_actor(actors["parent"], actors[role], alpha)
                candidates[alpha][role] = candidate
                checkpoint(out, "interpolation_"+label+"_"+role, dict(actor=candidate.policy_state(),
                    alpha=alpha, parameter_names=[n for n, _ in candidate.named_parameters()],
                    parent_actor_hash=source_hashes["parent"], source_child_actor_hash=source_hashes[role],
                    source_checkpoint_sha256=CHECKPOINTS[role][1], promotion_claim=False), archive)
        phase("qualification_parent")
        seeds = range(config["qualification_seed"], config["qualification_seed"]+config["qualification_roots"])
        noise = config["qualification_noise_seed"]
        parent_mean = evaluate(actors["parent"], seeds, "interpolation_qual_parent_mean", True, noise)
        parent_stoch = evaluate(actors["parent"], seeds, "interpolation_qual_parent_stochastic", False, noise)
        result["qualification_parent"] = dict(mean=_summary(parent_mean), stochastic=_summary(parent_stoch))
        for alpha in config["alphas"]:
            phase("qualification_alpha_"+str(alpha)); panels = {"parent": parent_mean}; stochastic = {"parent": parent_stoch}
            for role, actor in candidates[alpha].items():
                tag = "interpolation_qual_a"+str(alpha).replace(".", "p")+"_"+role
                panels[role] = evaluate(actor, seeds, tag+"_mean", True, noise)
                stochastic[role] = evaluate(actor, seeds, tag+"_stochastic", False, noise)
            row = dict(alpha=alpha, criteria=assess_panels(panels, stochastic,
                config["minimum_complete_gain"], config["capture_min"]),
                raw_criteria=assess_raw(panels, stochastic, config["minimum_complete_gain"], config["capture_min"]),
                guided_mean_complete_cost=math.fsum(r.cost for r in panels["guided"])/len(panels["guided"]),
                raw_mean_complete_cost=math.fsum(r.cost for r in panels["raw"])/len(panels["raw"]),
                mean_panels={n: _summary(p) for n, p in panels.items() if n != "parent"},
                stochastic_panels={n: _summary(p) for n, p in stochastic.items() if n != "parent"},
                actor_hashes={n: policy_hash(a) for n, a in candidates[alpha].items()})
            result["qualification"].append(row); archive.append("interpolation_qualification", result=row)
        selected = choose_alpha(result["qualification"])
        raw_selected = choose_raw_alpha(result["qualification"])
        result.update(selected_alpha=selected, raw_selected_alpha=raw_selected,
            best_raw_parent_fallback=raw_selected is None,
            raw_selection_rule="raw_gain_vs_parent_and_capture_nonregression_both_modes;minimum_raw_mean_complete_cost;smaller_alpha_tie;otherwise_parent")
        archive.append("interpolation_selection", selected_alpha=selected,
            raw_selected_alpha=raw_selected, best_raw_parent_fallback=raw_selected is None,
            raw_rule=result["raw_selection_rule"],
            rule="qualified_guided_mean_complete_cost_minimum;smaller_alpha_tie", final_evidence_not_seen=True)
        if selected is None:
            result["stage"] = "failed_all_interpolation_qualification"
        else:
            phase("final_fresh_evaluation")
            seeds = range(config["evaluation_seed"], config["evaluation_seed"]+config["evaluation_roots"])
            noise = config["evaluation_noise_seed"]
            best_raw = actors["parent"] if raw_selected is None else candidates[raw_selected]["raw"]
            roster = {"parent": actors["parent"], **candidates[selected], "best_raw": best_raw}
            result["best_raw_actor_hash"] = policy_hash(best_raw)
            result["best_raw_query_policy"] = "Execute and charge a distinct full panel even when owner duplicates parent/raw_same_alpha"
            panels = {n: evaluate(a, seeds, "interpolation_final_"+n+"_mean", True, noise) for n, a in roster.items()}
            stochastic = {n: evaluate(a, seeds, "interpolation_final_"+n+"_stochastic", False, noise) for n, a in roster.items()}
            criteria = assess_panels(panels, stochastic, config["minimum_complete_gain"], config["capture_min"])
            result.update(passed=criteria["passed"], negative_retained=not criteria["passed"],
                stage="complete" if criteria["passed"] else "failed_fresh_interpolation_improvement",
                final=dict(criteria=criteria, mean_panels={n: _summary(p) for n, p in panels.items()},
                    stochastic_panels={n: _summary(p) for n, p in stochastic.items()}))
        if {n: policy_hash(a) for n, a in actors.items()} != source_hashes:
            raise RuntimeError("Source actor parameters changed during the assay")
    except BaseException as error:
        result.update(passed=False, stage="execution_error", negative_retained=True,
            error_type=type(error).__name__, error=str(error))
        archive.append("assay_error", error_type=type(error).__name__, error=str(error))
    finally:
        if installed:
            plugin.uninstall()
        work.close(); archive.close()
    events = [json.loads(line) for line in work.path.read_text().splitlines()]
    result["accounting"] = dict(
        requested_scalar_controls=sum(e["controls"] for e in events if e["kind"] == "scalar_transition_requested"),
        completed_scalar_controls=sum(e["controls"] for e in events if e["kind"] == "scalar_transition_complete"),
        reserved_query_controls=0 if dynamics is None else dynamics.reserved,
        actor_rows=sum(e["rows"] for e in events if e["kind"] == "neural_requested"),
        actor_calls=sum(e["kind"] == "neural_requested" for e in events),
        adam_attempts=sum(e["kind"] == "optimizer_update_requested" for e in events),
        adam_completed=sum(e["kind"] == "optimizer_update_complete" for e in events),
        optimizer_rejected_attempts=sum(e["kind"] == "optimizer_rejected" for e in events),
        diagnostic_backward_attempts=sum(e["kind"] == "autograd_grad_requested" for e in events),
        gradient_backward_attempts=sum(e["kind"] == "autograd_backward_requested" for e in events),
        global_autograd_grad_calls=sum(e["kind"] == "autograd_grad_requested" for e in events),
        global_autograd_backward_calls=sum(e["kind"] == "autograd_backward_requested" for e in events),
        model_controls=0, new_mujoco_controls=0, new_warp_controls=0, query_totals=archive.totals(),
        overlap="Global hooks and attached per-actor meter observe the same forwards; never sum them")
    fresh = result["accounting"]; old = provenance["precision_paid_ancestry"]; v3 = provenance["v3_paid_ancestry"]
    result["cost_crosswalk"] = dict(new_assay=fresh, precision_ancestor=old, v3_ancestor=v3,
        unique_lineage_requested_scalar_controls=fresh["requested_scalar_controls"]+old["requested_scalar_controls"]+v3["requested_scalar_controls"],
        unique_lineage_actor_rows=fresh["actor_rows"]+old["actor_rows"]+v3["actor_rows"],
        unique_lineage_adam_attempts=fresh["adam_attempts"]+old["adam_attempts"]+v3["adam_attempts"],
        scope="One shared paid ancestry; all alpha candidates included. Alpha/arm standalone attributions overlap, not independent repetitions.")
    unchanged = all(_sha(out/r["file"]) == r["sha256"] for r in manifest["files"])
    if not unchanged:
        result.update(passed=False, stage="source_changed", negative_retained=True)
    result.update(config_sha256=_sha(out/"config.json"), manifest_sha256=_sha(out/"manifest.json"),
        input_provenance_sha256=_sha(out/"input_provenance.json"), source_unchanged=unchanged,
        query_ledger_sha256=_sha(archive.ledger_path), work_ledger_sha256=_sha(work.path),
        assay_wall_seconds=time.perf_counter()-start, declared_work_bound=bound,
        reused_ancestry_wall_seconds=provenance["precision_wall_seconds"]+provenance["v3_wall_seconds"],
        no_retuning=True, old_negative_unchanged=True,
        scope="Cheap Ending step-acceptance mechanism only; no full two-round bidirectional, exploration, efficiency, or physical-task success claim")
    _write(out/"assay_result.json", result)
    return result
