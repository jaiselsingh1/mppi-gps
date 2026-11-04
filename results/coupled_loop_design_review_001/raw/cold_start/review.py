"""Saved-only cold-start/refinement review; no policy or simulator imports.

This script reads only the immutable snapshot roster in the bound review
configuration. First and second refinements have the same simulator context,
but consume successive random draws. Their differences are descriptive and do
not identify the causal effect of warm initialization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import h5py
import numpy as np


def binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def write(path, value):
    with Path(path).open("x") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")


def array_hash(value):
    value = np.ascontiguousarray(value)
    prefix = json.dumps([value.dtype.str, list(value.shape)]).encode()
    return hashlib.sha256(prefix + value.tobytes()).hexdigest()


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def summary(values):
    array = np.asarray(values, dtype=float)
    return dict(n=len(array), mean=float(array.mean()), median=float(np.median(array)),
                minimum=float(array.min()), maximum=float(array.max()),
                decreases=int(np.count_nonzero(array < 0)),
                equal=int(np.count_nonzero(array == 0)),
                increases=int(np.count_nonzero(array > 0))) if len(array) else dict(n=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cpu-cap", type=int)
    args = parser.parse_args()
    cpu_started, wall_started = time.process_time(), time.monotonic()
    config = json.loads(args.config.read_text())
    cap = min(config["limits"]["cold_start_cpu_seconds"], args.cpu_cap or config["limits"]["cold_start_cpu_seconds"])
    resource.setrlimit(resource.RLIMIT_CPU, (int(cap), int(cap) + 1))
    out = Path(__file__).resolve().parent
    inputs = {}
    checked_bytes = 0

    def bound(spec):
        nonlocal checked_bytes
        if time.process_time() - cpu_started > cap:
            raise RuntimeError("Saved-only cold-start CPU cap exhausted")
        key = str(Path(spec["path"]).resolve())
        if key not in inputs:
            actual = binding(spec["path"])
            if actual != dict(spec, path=key):
                raise ValueError("Bound review evidence changed: " + key)
            inputs[key] = actual
            checked_bytes += actual["bytes"]
        return Path(key)

    def existing(path):
        spec = binding(path)
        inputs[spec["path"]] = spec
        return Path(spec["path"])

    inputs[str(args.config.resolve())] = binding(args.config)
    inputs[str(Path(__file__).resolve())] = binding(__file__)
    original_root = Path(config["original_run"])
    launch = json.loads(existing(original_root / "launch.json").read_text())
    science = json.loads(bound(launch["config"]).read_text())
    prepared = json.loads(bound(launch["prepared_manifest"]).read_text())
    bound(prepared["restarts"])
    source_root = Path(launch["source_directory"])
    source_files = {}
    for name in ("src/mppi/mppi.py", "src/gps/mppi_gps/local_coupling_multiround.py",
                 "src/gps/mppi_gps/local_coupling_multiround_measurements.py",
                 "src/gps/mppi_gps/local_coupling_controller.py"):
        source_files[name] = binding(existing(source_root / name))
    rows, arrays, probe_rows, missing = [], [], [], []
    banks_for_context_rng = {}
    all_states = set()

    def record_pair(bank_spec, role, *, full_paths=None, manifest=None, acquisition=None):
        path = bound(bank_spec)
        with h5py.File(path, "r") as raw:
            groups = [raw["000000"], raw["000001"]]
            if [(int(g.attrs["control"]), int(g.attrs["inner"])) for g in groups] != [(0, 0), (0, 1)]:
                raise ValueError("Saved first decision must contain exactly ordered initial refinements")
            pair = [{key: g[key][:] for key in ("U_before", "U_after", "C", "J", "weights",
                         "full_survival", "initial_state", "initial_warmstart", "initial_integration")}
                    for g in groups]
            attrs = [{key: (value.item() if isinstance(value, np.generic) else value)
                      for key, value in g.attrs.items()} for g in groups]
        contexts_equal = all(same(pair[0][key], pair[1][key]) for key in
                             ("initial_state", "initial_warmstart", "initial_integration"))
        context_hash = hashlib.sha256("".join(array_hash(pair[0][key]) for key in
            ("initial_state", "initial_warmstart", "initial_integration")).encode()).hexdigest()
        nominal_chain = same(pair[0]["U_after"], pair[1]["U_before"])
        zero_start = bool(np.all(pair[0]["U_before"] == 0))
        rng_chain = json.loads(attrs[0]["rng_after"]) == json.loads(attrs[1]["rng_before"])
        same_random = json.loads(attrs[0]["rng_before"]) == json.loads(attrs[1]["rng_before"])
        if not contexts_equal or not nominal_chain or not zero_start or not rng_chain or same_random:
            raise ValueError("Initial zero/warm refinement context, plan or RNG chain differs")
        for index in (0, 1):
            p, a = pair[index], attrs[index]
            zero = bool(np.all(p["U_before"] == 0))
            rng_key = json.dumps(json.loads(a["rng_before"]), sort_keys=True)
            key = context_hash, rng_key
            banks_for_context_rng.setdefault(key, []).append(dict(
                role=role, refinement=index, zero_nominal=zero,
                U_before_sha256=array_hash(p["U_before"]), bank_stats=bank_spec))
        if full_paths is not None:
            for index, full in enumerate(full_paths):
                full_path = bound(full)
                with np.load(full_path, allow_pickle=False) as data:
                    for key, value in pair[index].items():
                        if not same(value, data[key]):
                            raise ValueError("Compact/full saved bank differs: " + key)
        result = dict(
            **{key: value for key, value in role.items() if key != "seed"},
            bank_stats_path=str(path), initial_context_sha256=context_hash,
            exact_same_initial_context=contexts_equal,
            zero_first_U_before=zero_start, second_U_before_equals_first_U_after=nominal_chain,
            first_rng_after_equals_second_rng_before=rng_chain,
            same_random_numbers_between_refinements=same_random,
            first_rng_before=attrs[0]["rng_before"], second_rng_before=attrs[1]["rng_before"],
            first_rng_after=attrs[0]["rng_after"], second_rng_after=attrs[1]["rng_after"],
            C_min_first=float(np.min(pair[0]["C"])), C_min_second=float(np.min(pair[1]["C"])),
            survival_first=float(pair[0]["full_survival"].mean()),
            survival_second=float(pair[1]["full_survival"].mean()),
            ESS_first=float(attrs[0]["ESS"]), ESS_second=float(attrs[1]["ESS"]),
            fall_weight_first=float(attrs[0]["fallen_weight"]),
            fall_weight_second=float(attrs[1]["fallen_weight"]),
            lambda_first=float(attrs[0]["lambda_value"]), lambda_second=float(attrs[1]["lambda_value"]),
            first_action_RMS_first=float(np.sqrt(np.mean(pair[0]["U_after"][0] ** 2))),
            first_action_RMS_second=float(np.sqrt(np.mean(pair[1]["U_after"][0] ** 2))),
            first_action_refinement_delta_RMS=float(np.sqrt(np.mean(
                (pair[1]["U_after"][0] - pair[0]["U_after"][0]) ** 2))),
            full_nominal_refinement_delta_RMS=float(np.sqrt(np.mean(
                (pair[1]["U_after"] - pair[0]["U_after"]) ** 2))),
            first_U_before_sha256=array_hash(pair[0]["U_before"]),
            first_U_after_sha256=array_hash(pair[0]["U_after"]),
            second_U_before_sha256=array_hash(pair[1]["U_before"]),
            second_U_after_sha256=array_hash(pair[1]["U_after"]),
            alpha=float(attrs[0]["alpha"]), seed=int(manifest["seed"]) if manifest else role["seed"],
        )
        for name in ("C_min", "survival", "ESS", "fall_weight", "first_action_RMS"):
            result[name + "_delta_second_minus_first"] = result[name + "_second"] - result[name + "_first"]
        if acquisition is not None:
            result.update(manifest_path=acquisition["manifest"]["path"],
                          episode_controls=manifest["outcome"]["controls"],
                          full_episode_survival=manifest["outcome"]["full_survival"],
                          local_walking=manifest["outcome"]["local_walking"],
                          episode_complete_cost=manifest["outcome"]["complete_cost"],
                          checkpoint_sha256=manifest["checkpoint"]["sha256"])
            with h5py.File(bound(manifest["trace"]), "r") as trace:
                result["executed_first_action_equals_second_update"] = same(
                    trace["actions"][0], pair[1]["U_after"][0])
            if not result["executed_first_action_equals_second_update"]:
                raise ValueError("Executed first action differs from final initial refinement")
        return result, pair

    for roster in config["snapshot_roster"]:
        snapshot = json.loads(bound(roster["snapshot"]).read_text())
        completed = [b for b in snapshot["rounds"] if b["status"] == "COMPLETED"]
        if [b["round"] for b in completed] != roster["completed_rounds"]:
            raise ValueError("Immutable snapshot completed-round roster differs")
        for block in completed:
            for acquisition in block["acquisitions"]:
                manifest = json.loads(bound(acquisition["manifest"]).read_text())
                if manifest["status"] != "COMPLETED":
                    raise ValueError("Only logically completed physical episodes belong in this review")
                checks = manifest["native_candidate_checks"]
                full0 = next(check["source"] for check in checks if "source" in check)
                full1_path = Path(full0["path"]).with_name("full_t000_p1.npz")
                full1 = binding(full1_path)
                role = dict(kind="acquisition", stream=roster["stream"], round=block["round"],
                            arm=acquisition["arm"], slot=acquisition["slot"])
                row, pair = record_pair(manifest["bank_stats"], role, full_paths=[full0, full1],
                                        manifest=manifest, acquisition=acquisition)
                arrays.append({key: np.stack([part[key] for part in pair]) for key in
                               ("U_before", "U_after", "initial_state", "initial_warmstart", "initial_integration")})
                rows.append(row)
                all_states.add(row["initial_context_sha256"])
        # These queries are measurement-only. Include all saved observed probe
        # anchors when searching for an actually controlled zero/warm contrast.
        probes = snapshot["initial_probes"] + [p for b in completed for p in b["probes"]]
        for probe in probes:
            result = json.loads(bound(probe["result"]).read_text())
            for anchor in result["anchors"]:
                if anchor["status"] != "OBSERVED":
                    missing.append(dict(probe=probe["result"], anchor=anchor))
                    continue
                role = dict(kind="measurement_only_probe", stream=roster["stream"],
                            round=result["round"], arm=probe["arm"],
                            control=anchor["control"], seed=anchor["seed"])
                row, _ = record_pair(anchor["bank_stats"], role, full_paths=anchor["full_banks"])
                probe_rows.append(row)

    true_contrasts = [dict(initial_context_sha256=key[0], rng_before=key[1], banks=values)
                      for key, values in banks_for_context_rng.items()
                      if {v["zero_nominal"] for v in values} == {True, False}]
    with (out / "refinement_arrays.npz").open("xb") as target:
        np.savez_compressed(target, **{key: np.stack([value[key] for value in arrays]) for key in arrays[0]})
    for filename, data in (("episodes.csv", rows), ("measurement_probes.csv", probe_rows)):
        if data:
            with (out / filename).open("x", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    groups = {}
    for arm in ("independent", "positive"):
        for population in ("all_completed_attempts", "full_300_control_survivors"):
            selected = [r for r in rows if r["arm"] == arm and
                        (population == "all_completed_attempts" or r["full_episode_survival"])]
            groups[arm + "/" + population] = dict(
                episodes=len(selected), local_walking=sum(r["local_walking"] for r in selected),
                measures={name: summary([r[name] for r in selected]) for name in (
                    "C_min_first", "C_min_second", "C_min_delta_second_minus_first",
                    "survival_first", "survival_second", "survival_delta_second_minus_first",
                    "ESS_first", "ESS_second", "ESS_delta_second_minus_first",
                    "first_action_RMS_first", "first_action_RMS_second",
                    "first_action_refinement_delta_RMS", "fall_weight_first", "fall_weight_second")})
    report = dict(
        schema="saved-cold-start-refinement-review-1", status="COMPLETED_SAVED_ONLY_REVIEW",
        review_config=binding(args.config), source_commit=config["source_commit"],
        complete_acquisition_episodes=len(rows), full_300_control_survivors=sum(r["full_episode_survival"] for r in rows),
        observed_probe_anchors=len(probe_rows), missing_probe_anchors=len(missing),
        acquisition_initial_contexts=len(all_states), fixed_acquisition_slots=8,
        acquisition_contexts_match_eight_fixed_slots=(len(all_states) == 8),
        all_zero_first_nominals=all(r["zero_first_U_before"] for r in rows),
        all_second_nominals_are_unshifted_first_updates=all(r["second_U_before_equals_first_U_after"] for r in rows),
        all_refinements_same_complete_context=all(r["exact_same_initial_context"] for r in rows),
        all_refinements_use_different_successive_rng=all(not r["same_random_numbers_between_refinements"] for r in rows),
        true_same_context_same_rng_zero_vs_warm_contrasts=true_contrasts,
        grouped_summary=groups,
        interpretation=[
            "Each acquisition cold-starts its carried H48 torque plan at exact zero, performs two same-context initial refinements, executes the second update's first action, then shifts once per executed control.",
            "Pass two uses pass one's full unshifted updated plan, but consumes the next random draw. Its observed changes combine plan refinement and changed samples; they do not isolate a warm-start causal effect.",
            "The acquisition block always starts at the same eight prepared student contexts. Current standalone evaluation states feed independent scratch-MPPI probes only; neither those queried actions nor those contexts enter the retained acquisition corpus.",
            "Data retention chooses the latest competent full300-control episode per fixed slot, otherwise retaining old experience. This is not corrective acquisition at newly visited states.",
            "Repeated independent solves reuse the fixed stream-slot planner seed, so repeated rounds are not independent cold-start replications.",
        ],
        minimal_future_test="One saved complete restart context, one fixed actor/alpha and private RNG snapshot, compare exactly one H48/K256 refinement from zero U versus a predefined valid nonzero carried U, sharing the exact innovations and native runtime; freeze both initial plans and record the full candidate banks and executed continuation. Do not interpret the existing successive-draw two-pass comparison as that test.",
        limitations=[
            "A saved-data design review, not an independent audit or a new controller experiment.",
            "Candidate cost minima, sample survival and ESS describe sampled banks; they are not executed-policy outcomes.",
            "Only the bound immutable snapshot roster is included; incomplete and later rounds are not assigned zero outcomes.",
            "No initial-plan causal contrast or additional teacher exploration was launched.",
        ],
        source_bindings=source_files,
    )
    write(out / "result.json", report)
    write(out / "missing_probe_anchors.json", missing)
    write(out / "inputs.json", dict(schema="saved-cold-start-input-roster-1", files=list(inputs.values())))
    outputs = [binding(path) for path in out.iterdir() if path.is_file() and path.name != "accounting.json"]
    write(out / "accounting.json", dict(
        schema="saved-review-accounting-1", cpu_limit_seconds=cap,
        process_cpu_seconds=time.process_time()-cpu_started,
        wall_seconds=time.monotonic()-wall_started, verified_input_bytes=checked_bytes,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        outputs=outputs, output_bytes=sum(x["bytes"] for x in outputs),
        physics_substeps=0, model_queries=0, neural_forwards=0, policy_updates=0,
        paid_preparation="Source/schema inspection and command overhead were not individually metered; charge the separate5-second command reserve. The first failed analysis retains its distinct120-second hard upper bound and unknown actual CPU.",
    ))
    print(json.dumps(dict(episodes=len(rows), probe_anchors=len(probe_rows),
        same_rng_zero_warm_contrasts=len(true_contrasts), cpu_seconds=time.process_time()-cpu_started)))


if __name__ == "__main__":
    main()
