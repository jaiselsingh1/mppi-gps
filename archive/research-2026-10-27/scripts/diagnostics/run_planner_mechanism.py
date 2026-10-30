"""Bounded frozen-value mechanism assay; never imports or starts a trainer run."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import tyro

from src.gps.planner_mechanism_diagnostic import (
    capture, endpoint_branch, first_action_branch, normalize_config,
)

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Options:
    study_dir: Path
    inputs_dir: Path | None = None
    workers: int = 6
    smoke: bool = False


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def seeds_for(root: int, discovery=2, confirmation=4) -> tuple[list[int], list[int], list[int]]:
    """Different roots/roles never reuse the planner's future stream."""
    if discovery < 1 or confirmation < 1 or discovery > 8 or confirmation > 8:
        raise ValueError("Bounded positive seed counts required")
    block = 700_000_000 + (int(root) - 42_000_000) * 100
    a = list(range(block, block + discovery))
    b = list(range(block + 10, block + 10 + confirmation))
    c = list(range(block + 20, block + 20 + confirmation))
    if len(set(a + b + c)) != len(a + b + c):
        raise AssertionError("Discovery and confirmation streams overlap")
    return a, b, c


def select_oracle(indices: list[int], full_costs: dict[int, list[float]], discovery: int) -> int:
    """The selector cannot access confirmation outcomes through this interface."""
    if not indices or discovery < 1:
        raise ValueError("Need measured discovery returns")
    if any(len(full_costs[i]) < discovery for i in indices):
        raise ValueError("Missing requested discovery returns")
    means = {i: float(np.mean(np.asarray(full_costs[i])[:discovery])) for i in indices}
    if not all(np.isfinite(list(means.values()))):
        raise ValueError("Nonfinite discovery return")
    return min(indices, key=lambda i: (means[i], i))


def _job(item):
    kind, args = item
    if kind == "capture":
        return capture(**args)
    if kind == "endpoint":
        return endpoint_branch(**args)
    if kind == "action":
        return first_action_branch(**args)
    raise ValueError(kind)


def _source_hashes() -> dict[str, str]:
    paths = [p for folder in ("src", "scripts", "configs", "assets")
             for p in (ROOT / folder).rglob("*")
             if p.is_file() and p.suffix in {".py", ".json", ".xml"}
             and "__pycache__" not in p.parts
             and p.name != "report_planner_mechanism.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def _ledger(out: Path) -> dict:
    totals, groups = {}, {}
    for category in ("captures", "endpoints", "replanning", "handoff", "guided"):
        counts = {}
        for p in sorted((out / category).glob("*.pkl")):
            row = pickle.loads(p.read_bytes())
            for key, value in row.get("ledger", {}).items():
                counts[key] = counts.get(key, 0) + int(value)
                totals[key] = totals.get(key, 0) + int(value)
        groups[category] = counts
    return dict(total=totals, by_category=groups,
        interpretation="Executed controls and modeled candidate/memory/verification controls are distinct. Planner calls are not controls. Shared bank capture charged once.")


def run(opt: Options) -> None:
    if not 1 <= opt.workers <= 12:
        raise ValueError("Use at most twelve diagnostic workers")
    out = opt.study_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if opt.inputs_dir is not None:
        origin = opt.inputs_dir.resolve()
        if (out / "inputs").exists():
            raise FileExistsError("Do not replace frozen input copies")
        shutil.copytree(origin / "inputs", out / "inputs")
        shutil.copy2(origin / "input_manifest.json", out / "input_manifest.json")
    manifest = json.loads((out / "input_manifest.json").read_text())
    cohorts = []
    packets = {}
    for order, (name, record) in enumerate(manifest["cohorts"].items()):
        folder = out / "inputs" / name
        for relative, expected in record["files"].items():
            if sha(folder / relative) != expected:
                raise ValueError(f"Frozen input changed: {name}/{relative}")
        config = json.loads((folder / "protocol.json").read_text())["config"]
        cfg = normalize_config(config)
        if opt.smoke:
            cfg.total_steps = 64
        model_file = next(folder.glob("value_*.pkl"))
        controller_file = next(folder.glob("iteration_*.pkl"))
        controller = pickle.loads(controller_file.read_bytes())
        value = pickle.loads(model_file.read_bytes())
        reference = pickle.loads((folder / "reference_controller.pkl").read_bytes())
        if value is None:
            raise ValueError(f"No frozen value for {name}")
        # Both sides use the SAME incumbent archive within each cohort.
        archive = controller["guided"]["archive"]
        roots = list(range(42_000_000 + order * 100, 42_000_000 + order * 100 + (1 if opt.smoke else 4)))
        if opt.smoke:
            roots = [root + 1_000_000 for root in roots]
        cohort = dict(name=name, roots=roots, config=asdict(cfg),
            capture_step=0 if opt.smoke else 128,
            model_file=str(model_file.relative_to(out)), model_sha256=sha(model_file),
            controller_file=str(controller_file.relative_to(out)),
            model_metrics=value.get("metrics"),
            source_completed_cycles=record["completed_cycles"],
            source_study=record["study"], archive_choice="frozen guided incumbent, identical for all diagnostic branches",
            reference_choice="cohort's own frozen canonical-restart bootstrap controller")
        cohorts.append(cohort)
        packets[name] = dict(cfg=cfg, archive=archive, value=value, reference=reference)
    discovery, confirmation = (1, 1) if opt.smoke else (2, 4)
    protocol = dict(version=1, smoke=opt.smoke, cohorts=cohorts,
        discovery_repeats=discovery, confirmation_repeats=confirmation,
        candidate_count=3, workers=opt.workers, diagnostic_alpha=.1,
        selection="Local baseline + actual bounded model choice if distinct, otherwise explicitly exploratory full-score alternative + seeded uniform distinct plan.",
        endpoint_semantics="Commit H controls then canonical-restart own fixed reference until finite end; model frozen.",
        replanning_semantics="Execute ONE chosen action then memory-only MPC with retained raw POST-bank nominal and paired fresh future RNG.",
        handoff_semantics="Single selected-plan nominal handoff for baseline and independently selected oracle winner; subsequent production nominal rule unchanged.",
        guided_semantics="UNQUALIFIED alpha=.1 common-paired-bank frozen model at every subsequent control; never promoted into a training run.",
        isolation="Separate copied inputs and checkout. Neither active run is edited, resumed, extended or fed these labels.",
        interpretation="Four fresh roots per frozen model, one decision/root, no backfill. Common paired bank for BOTH models; not reproduction of original coupled positive-alpha extra-mean scoring. Conditional suffix mechanism assay, not eight-second walking qualification or training-seed robustness.",
        seed_rule="Per-root discovery +0..1, endpoint confirmation +10..13, first-action confirmation +20..23, base700m+(root-42m)*100; disjoint by construction.",
        source_sha256=_source_hashes())
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Resume must preserve all diagnostic inputs/configuration/source")
    else:
        write_json(protocol_path, protocol)
        for relative in protocol["source_sha256"]:
            target = out / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    started = time.monotonic()
    def status(phase, completed=0, total=0):
        write_json(out / "status.json", dict(status="running", phase=phase, completed=completed,
            total=total, elapsed_seconds=time.monotonic()-started, driver_pid=os.getpid(),
            work=_ledger(out), smoke=opt.smoke))
    def execute(pool, category, items):
        result = {}
        pending = {}
        for key, job in items.items():
            path = out / category / (key + ".pkl")
            if path.exists():
                result[key] = pickle.loads(path.read_bytes())
            else:
                pending[pool.submit(_job, job)] = (key, path)
        status(category, len(result), len(items))
        for future in as_completed(pending):
            key, path = pending[future]
            value = future.result()
            save(path, value)
            result[key] = value
            status(category, len(result), len(items))
        return result
    import multiprocessing as mp
    try:
        with ProcessPoolExecutor(max_workers=opt.workers, mp_context=mp.get_context("spawn")) as pool:
            capture_jobs = {}
            for c in cohorts:
                data = packets[c["name"]]
                for root in c["roots"]:
                    key = f"{c['name']}_root_{root}"
                    capture_jobs[key] = ("capture", dict(root=root, cfg=data["cfg"], archive=data["archive"],
                        value_packet=data["value"], kind=c["name"], seed=610_000_000+root,
                        capture_step=c["capture_step"]))
            captures = execute(pool, "captures", capture_jobs)
            endpoint_jobs = {}
            for c in cohorts:
                data = packets[c["name"]]
                for root in c["roots"]:
                    key = f"{c['name']}_root_{root}"
                    cap = captures[key]
                    if cap["status"] != "captured":
                        continue
                    dseeds, eseeds, _ = seeds_for(root, discovery, confirmation)
                    for role in cap["selection"]["roles"]:
                        i = role["index"]
                        endpoint_jobs[key+f"_candidate_{i}"] = ("endpoint", dict(captured=cap,
                            candidate_index=i, cfg=data["cfg"], reference=data["reference"],
                            seeds=dseeds+eseeds, value_packet=data["value"], kind=c["name"]))
            endpoints = execute(pool, "endpoints", endpoint_jobs)
            selections = {c["name"]:{} for c in cohorts}
            for c in cohorts:
                for root in c["roots"]:
                    key = f"{c['name']}_root_{root}"
                    cap = captures[key]
                    if cap["status"] != "captured":
                        selections[c["name"]][str(root)] = dict(status=cap["status"], no_backfill=True)
                        continue
                    indices = [r["index"] for r in cap["selection"]["roles"]]
                    values = {i:endpoints[key+f"_candidate_{i}"]["full_costs"] for i in indices}
                    oracle = select_oracle(indices, values, discovery)
                    selections[c["name"]][str(root)] = dict(status="captured", oracle_index=oracle,
                        baseline_index=int(cap["bank"]["baseline_index"]), candidate_indices=indices,
                        discovery_means={str(i):float(np.mean(values[i][:discovery])) for i in indices},
                        selection_seed_count=discovery, confirmation_used_for_selection=False)
            write_json(out / "selection.json", selections)
            replanning, handoff, guided = {}, {}, {}
            for c in cohorts:
                data = packets[c["name"]]
                for root in c["roots"]:
                    key = f"{c['name']}_root_{root}"
                    cap = captures[key]
                    if cap["status"] != "captured":
                        continue
                    _, _, rseeds = seeds_for(root, discovery, confirmation)
                    select = selections[c["name"]][str(root)]
                    for i in select["candidate_indices"]:
                        args = dict(captured=cap, candidate_index=i, cfg=data["cfg"],
                                    archive=data["archive"], seeds=rseeds)
                        replanning[key+f"_candidate_{i}"] = ("action", args)
                    for i in dict.fromkeys([select["baseline_index"], select["oracle_index"]]):
                        args = dict(captured=cap, candidate_index=i, cfg=data["cfg"],
                                    archive=data["archive"], seeds=rseeds, handoff_selected_nominal=True)
                        handoff[key+f"_candidate_{i}"] = ("action", args)
                    actual = int(cap["bank"]["selected_index"])
                    guided[key] = ("action", dict(captured=cap, candidate_index=actual, cfg=data["cfg"],
                        archive=data["archive"], seeds=rseeds, value_packet=data["value"], kind=c["name"], alpha=.1))
            execute(pool, "replanning", replanning)
            execute(pool, "handoff", handoff)
            execute(pool, "guided", guided)
        changed = [relative for relative, expected in protocol["source_sha256"].items()
                   if sha(ROOT/relative) != expected or sha(out/"source"/relative) != expected]
        if changed:
            raise ValueError(f"Diagnostic source changed: {changed}")
        write_json(out / "status.json", dict(status="complete", phase="complete",
            elapsed_seconds=time.monotonic()-started, driver_pid=os.getpid(),
            source_unchanged=True, work=_ledger(out), smoke=opt.smoke))
    except BaseException as error:
        write_json(out / "status.json", dict(status="failed", phase="failure", error=repr(error),
            elapsed_seconds=time.monotonic()-started, driver_pid=os.getpid(),
            work=_ledger(out), unfinished_work="Any work not durably written is an uncounted lower bound."))
        raise


if __name__ == "__main__":
    run(tyro.cli(Options))
