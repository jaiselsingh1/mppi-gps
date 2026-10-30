"""Descriptive, post-audit return decomposition at the qualified48-control horizon.

No engine, actor, audit implementation or training module is imported. No new
target, inference, selection, pass criterion or experiment is introduced.
CLI: RUN --audit CLOSED_AUDIT.json --plan PLAN.json --out NEW.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import time

import numpy as np


HORIZON, GAMMA, ABSORBING = 48, .999, 5.
COMPONENTS = ("near", "far_actual", "far_absorbing", "total")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def absorption(start, stop):
    return float(ABSORBING*np.sum(GAMMA**np.arange(start, stop, dtype=np.int64)))


def split_return(raw_costs, remaining, terminated):
    """All discounts are relative to this actual candidate query's initial state."""
    costs = np.asarray(raw_costs, dtype=np.float64)
    require(type(remaining) is int and remaining > 0 and type(terminated) is bool,
            "Explicit finite horizon and termination are required")
    require(costs.ndim == 1 and np.isfinite(costs).all() and np.all(costs >= 0), "Invalid raw costs")
    n = len(costs)
    require(n <= remaining and (n == remaining or terminated), "Healthy incomplete query has no complete credit")
    h = min(HORIZON, remaining)
    discounted = GAMMA**np.arange(n)*costs
    near_actual = float(discounted[:h].sum())
    near_absorbing = absorption(n, h) if terminated and n < h else 0.
    far_actual = float(discounted[h:].sum())
    far_absorbing = absorption(max(n, h), remaining) if terminated else 0.
    total = float(discounted.sum())+(absorption(n, remaining) if terminated else 0.)
    near = near_actual+near_absorbing
    residual = near+far_actual+far_absorbing-total
    require(abs(residual) <= 1e-9+1e-12*abs(total), "Components do not reconstruct the complete return")
    scale = max(1., absorption(0, remaining))
    return dict(remaining=remaining, executed_steps=n, terminated=terminated,
        healthy_complete=not terminated and n == remaining, executed_beyond_H=n > HORIZON,
        near_actual=near_actual, near_absorbing=near_absorbing, near=near,
        far_actual=far_actual, far_absorbing=far_absorbing, far_total=far_actual+far_absorbing,
        total=total, reconstruction_residual=residual, normalization_scale=scale,
        normalized={key:value/scale for key,value in
                    (("near",near),("far_actual",far_actual),("far_absorbing",far_absorbing),("total",total))})


class BoundReader:
    def __init__(self, root, expected):
        self.root, self.expected, self.consumed = Path(root).resolve(), expected, {}

    def data(self, relative):
        part = PurePosixPath(relative)
        require(not part.is_absolute() and ".." not in part.parts and bool(part.parts), "Unsafe evidence path")
        path = (self.root/relative).resolve()
        require(path.is_relative_to(self.root), "Evidence escapes the closed run")
        require(relative in self.expected, "Input absent from independent audit: "+relative)
        raw = path.read_bytes()
        require(sha(raw) == self.expected[relative], "Audited input hash changed: "+relative)
        self.consumed[relative] = sha(raw)
        return raw

    def obj(self, relative):
        return json.loads(self.data(relative))


def coverage(refs, queries):
    unique = list(dict.fromkeys(refs))
    rows = [queries[ref] for ref in unique]
    n = len(rows)
    count = lambda key: sum(bool(row[key]) for row in rows)
    return dict(logical_query_uses=len(refs), unique_physical_queries=n, cache_alias_uses=len(refs)-n,
        fraction_denominator="unique physical queries; checkpoint reuse is not independent",
        executed_beyond_H=count("executed_beyond_H"),
        fraction_executed_beyond_H=None if not n else count("executed_beyond_H")/n,
        healthy_complete=count("healthy_complete"),
        fraction_healthy_complete=None if not n else count("healthy_complete")/n,
        physical_falls=count("terminated"),
        declared_horizon_counts=dict(sorted(Counter(str(row["remaining"]) for row in rows).items())),
        falling_executed_steps_counts=dict(sorted(Counter(str(row["executed_steps"]) for row in rows if row["terminated"]).items())))


def panel_components(refs, contexts, queries):
    require(len(refs) == len(contexts) and len(set(refs)) == len(refs), "Invalid physical-query panel")
    matrix = np.asarray([[queries[ref]["normalized"][key] for key in COMPONENTS] for ref in refs])
    roles = np.asarray([context["role"] for context in contexts])
    require(set(roles) == {"root", "guide"}, "The frozen equal-role mixture is required")
    role_values = {role:matrix[roles == role].mean(axis=0) for role in ("root", "guide")}
    mixed = (role_values["root"]+role_values["guide"])/2
    clusters = {}
    roots = np.asarray([context["ancestry"]["reset_seed"] for context in contexts])
    for root in sorted(set(roots)):
        local = roots == root
        require(np.any(local & (roles == "root")) and np.any(local & (roles == "guide")), "Incomplete root cluster")
        clusters[str(root)] = dict(zip(COMPONENTS, ((matrix[local & (roles == "root")].mean(axis=0)
            +matrix[local & (roles == "guide")].mean(axis=0))/2).tolist()))
    convert = lambda values:dict(zip(COMPONENTS, values.tolist()))
    return dict(mixture=convert(mixed), role_scores={role:convert(values) for role,values in role_values.items()},
                root_clusters=clusters)


def gains(before, after):
    result = {key:float(before[key]-after[key]) for key in COMPONENTS}
    result["beyond_H"] = result["far_actual"]+result["far_absorbing"]
    result["beyond_H_fraction_of_total_gain"] = result["beyond_H"]/result["total"] if result["total"] > 0 else None
    require(np.isclose(result["near"]+result["beyond_H"], result["total"], atol=1e-10, rtol=1e-10), "Gain decomposition mismatch")
    return result


def compare(before, after, queries):
    bycontext = []
    for previous, current in zip(before["query_refs"], after["query_refs"]):
        a, b = queries[previous], queries[current]
        require(a["context_id"] == b["context_id"] and a["remaining"] == b["remaining"], "Unmatched checkpoint contexts")
        both_early = a["terminated"] and b["terminated"] and max(a["executed_steps"], b["executed_steps"]) <= HORIZON
        difference = gains(a["normalized"], b["normalized"])
        if both_early:
            require(abs(difference["beyond_H"]) <= 1e-12, "Two pre-horizon falls cannot change the constant far tail")
        bycontext.append(dict(context_id=a["context_id"], previous_query=previous, current_query=current,
            same_cached_query=previous == current,
            both_fall_before_H=both_early and max(a["executed_steps"],b["executed_steps"]) < HORIZON,
            both_fall_by_H_without_measured_far_cost=bool(both_early), normalized_gains=difference))
    mixed = gains(before["components"]["mixture"],after["components"]["mixture"])
    scope = ("no_positive_complete_return_gain" if mixed["total"] <= 1e-12 else
             "positive_gain_includes_beyond_H_component" if mixed["beyond_H"] > 1e-12 else
             "positive_gain_near_only_or_far_component_regresses")
    return dict(previous_checkpoint=before["checkpoint_index"], current_checkpoint=after["checkpoint_index"],
        descriptive_gain_scope=scope, arithmetic_zero_tolerance=1e-12, mixture=mixed,
        root_only=gains(before["components"]["role_scores"]["root"],after["components"]["role_scores"]["root"]),
        root_clusters={root:gains(before["components"]["root_clusters"][root], after["components"]["root_clusters"][root])
                       for root in before["components"]["root_clusters"]}, contexts=bycontext,
        both_early_fall_contexts=sum(row["both_fall_by_H_without_measured_far_cost"] for row in bycontext))


def analyze(run_dir, audit_path, plan_path):
    started = time.perf_counter()
    audit_raw, plan_raw = Path(audit_path).read_bytes(), Path(plan_path).read_bytes()
    audit, plan = json.loads(audit_raw), json.loads(plan_raw)
    require(plan.get("schema") == "walker-owned-credit-decomposition-plan-v1" and plan.get("lookahead_controls") == HORIZON
        and plan.get("gamma") == GAMMA and plan.get("outcomes_inspected_to_prepare_this_plan") is False, "Undeclared descriptive decomposition")
    require(audit.get("schema") == "walker-owned-signal-independent-audit-v1" and audit.get("phase") in ("PASS","FAIL")
        and audit.get("evidence_arithmetic_valid") is True and audit.get("errors") == [] and audit.get("pending") == []
        and audit.get("action_verification_requested") is True, "Completed valid independent raw/action audit required")
    journal = audit.get("audit_inference_journal", {})
    require(journal.get("inflight_rows") == 0 and journal.get("error_rows") == 0
        and journal.get("incomplete_final_line") is False and journal.get("completed_rows") == journal.get("requested_rows"),
        "Independent action audit did not finish cleanly")
    reader = BoundReader(run_dir, audit["input_sha256"])
    config, contexts, status = reader.obj("config.json"), reader.obj("contexts.json"), reader.obj("status.json")
    require(config["gamma"] == GAMMA and config["total_steps"] == 1000
        and status["phase"] == "COMPLETED_TENTATIVE_BLIND_CHECK", "Closed frozen assay required")
    held_contexts = contexts["heldout"]
    require(len(held_contexts) == 25 and {c["ancestry"]["reset_seed"] for c in held_contexts} == set(range(35,40)),
        "All25 held contexts/five reset clusters required")
    queries = {}
    for detail in audit["queries"]:
        ref = detail["query_ref"]
        require(ref not in queries, "Repeated physical query row in audit")
        result = reader.obj(f"queries/{ref}/result.json")
        require(result["query_ref"] == ref and result["complete"] is True and result["gamma"] == GAMMA
            and result["cutoff"] is False and result["continuation_owner"] == "candidate", "Incomplete/unowned decomposition input")
        require(all(result[key] == detail[key] for key in ("owner_hash","context_id","phase","remaining")),
                "Query identity differs from independent audit")
        with np.load(io.BytesIO(reader.data(result["raw_path"])), allow_pickle=False) as raw:
            split = split_return(raw["raw_costs"], result["remaining"], result["terminated"])
        require(split["executed_steps"] == result["executed_steps"]
            and np.isclose(split["total"], result["cost"], atol=1e-9, rtol=1e-11), "Complete cost reconstruction differs from audited result")
        require(split["executed_steps"] == detail["executed_steps"]
            and np.isclose(split["total"], detail["cost"], atol=1e-9, rtol=1e-11), "Raw return differs from independent audit")
        queries[ref] = dict(query_ref=ref, owner_hash=result["owner_hash"], context_id=result["context_id"],
            phase=result["phase"], seed=detail["seed"], raw_path=result["raw_path"], **split)
    seeds, logical_held_refs = [], []
    for audited_seed in audit["seeds"]:
        seed = audited_seed["seed"]
        if audited_seed["distribution_failed"]:
            seeds.append(dict(seed=seed, distribution_failed=True, checkpoints=[], comparisons=None,
                              interpretation="No held checkpoint decomposition: frozen distribution stop retained."))
            continue
        panels = []
        for audited_panel in audited_seed["held"]:
            index = audited_panel["index"]
            panel = reader.obj(f"held/seed{seed}_checkpoint{index}.json")
            require(panel["query_refs"] == audited_panel["query_refs"] and panel["owner_hash"] == audited_panel["owner_hash"]
                and panel["context_ids"] == [c["id"] for c in held_contexts], "Held panel differs from audited references")
            refs = panel["query_refs"]
            require(all(queries[ref]["phase"] == "blind_verification" for ref in refs), "Foreign continuation phase")
            components = panel_components(refs, held_contexts, queries)
            require(np.isclose(components["mixture"]["total"], audited_panel["score"], atol=1e-10, rtol=1e-10), "Frozen mixture reconstruction differs")
            panels.append(dict(checkpoint_index=index, owner_hash=panel["owner_hash"], cache_alias=panel["cache_alias"],
                query_refs=refs, components=components, coverage=coverage(refs, queries)))
            logical_held_refs.extend(refs)
        require([panel["checkpoint_index"] for panel in panels] == [0,1,2,3], "All four frozen checkpoints required")
        seeds.append(dict(seed=seed, distribution_failed=False, checkpoints=panels,
            comparisons=dict(initial_to_final=compare(panels[0],panels[-1],queries),
                increments=[compare(a,b,queries) for a,b in zip(panels,panels[1:])])) )
    require({seed["seed"] for seed in seeds} == set(config["seeds"]), "Missing frozen seed")
    held_unique = [ref for ref,query in queries.items() if query["phase"] == "blind_verification"]
    require(set(held_unique) == set(logical_held_refs), "Audited held query omitted from descriptive panels")
    return dict(schema="walker-owned-credit-decomposition-v1", descriptive_only=True,
        created_utc=datetime.now(timezone.utc).isoformat(), lookahead_controls=HORIZON, gamma=GAMMA,
        audit_sha256=sha(audit_raw), plan_sha256=sha(plan_raw), analysis_source_sha256=sha(Path(__file__).read_bytes()),
        independently_audited_signal_phase=audit["phase"], frozen_criteria_unchanged=True,
        consumed_audited_input_sha256=reader.consumed,
        coverage=dict(all_unique_complete_queries=coverage(list(queries),queries),
            held_unique_physical_queries=coverage(held_unique,queries),
            held_checkpoint_uses=coverage(logical_held_refs,queries)),
        control_units=dict(outer_control_slot_seconds=.008, lookahead_seconds=48*.008,
            mujoco_integration_frames_per_confirmed_returned_control=4, physics_frame_seconds=.002,
            confirmed_complete_query_controls=sum(query["executed_steps"] for query in queries.values()),
            integration_frames_derived_from_qualified_recipe=4*sum(query["executed_steps"] for query in queries.values()),
            failed_call_scope="A failed call can have0..4 completed integration frames, unknown, and retains one charged requested control slot.",
            additional_integration_frames_from_analysis=0),
        seeds=seeds, queries=list(queries.values()), additional_physics_controls=0,
        additional_actor_inferences=0, additional_fits=0, wall_seconds=time.perf_counter()-started,
        limitations=["Positive component gains mean lower complete cost; fractions can be negative or exceed1 when components offset.",
            "After both controllers fall by48, identical far analytical absorption cannot establish a beyond-lookahead gain.",
            "Measured execution past48 establishes that trajectory coverage only; analytical absorption remains distinct.",
            "The actor changes later actions too: far-cost gains do not causally prove earlier preparatory actions improved.",
            "This is contribution/accounting decomposition, not an action-time causal claim or a hypothetical-plan-to-executed-controller claim.",
            "Cached query reuse is the same evidence, not independent sampling.",
            "This descriptive result does not change frozen selection/verdicts or establish standalone walking, MPPI feedback, or efficiency."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Descriptive output already exists; preserve it")
    result = analyze(args.run_dir, args.audit, args.plan)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(args.out), sha256=sha(args.out.read_bytes()),
                          wall_seconds=result["wall_seconds"], additional_controls=0, additional_inferences=0)))


if __name__ == "__main__":
    main()
