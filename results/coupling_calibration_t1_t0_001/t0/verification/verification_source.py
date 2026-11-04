"""Independent saved arithmetic for T0; never import its producer or renderer.

Run only after the supervisor has closed all4 preparations and16 controllers.
The parent retains a resource receipt even if the bounded child fails. This
does not replay physics, evaluate a policy, or close the separate L3 audit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def file_binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def verify(base, report):
    import h5py
    import numpy as np

    inputs = {}

    def bind(value):
        path = Path(value["path"] if isinstance(value, dict) else value).resolve()
        if str(path) not in inputs:
            inputs[str(path)] = file_binding(path)
        if isinstance(value, dict):
            check(inputs[str(path)] == value, "Changed saved byte binding: " + str(path))
        return path

    def load(value):
        return json.loads(bind(value).read_text())

    def same(a, b):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

    def innovation_hash(value):
        digest = hashlib.sha256(json.dumps(["array", list(value.shape), str(value.dtype)]).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    launch = load(base / "launch.json")
    config, calibration = load(launch["config"]), load(launch["calibration"])
    supervisor = load(base / "t0/result.json")
    check(supervisor["status"] == "COMPLETED" and len(supervisor["records"]) == 20,
          "Wait for the closed20-job supervisor roster before verification")
    alpha = launch["selected_alpha"]
    check(alpha == calibration["selected_alpha"] and alpha > 0, "Frozen calibrated coefficient changed")
    # This checks recorded-prediction arithmetic only. It does not regenerate
    # neural outputs or certify the historical L3 replay/audit.
    calibration_banks, ratios = [], {"all": [], "competitive": []}
    check(len(config["banks"]) == 6 and config["competitive"]["count"] == 32,
          "Exactly six frozen banks and fixed competitive32 required")
    for bank_id, entry in enumerate(config["banks"]):
        with np.load(bind(entry["bank"]), allow_pickle=False) as source:
            raw = {name: source[name].copy() for name in
                   ("C", "D", "lengths", "actions", "mask", "predictions")}
        with h5py.File(bind(entry["stats"]), "r") as source:
            group = source[entry["group"]]
            check(group.attrs["actor_hash"] == config["actor_hash"], "T1 bank belongs to another actor")
            check(same(group["C"][:], raw["C"]) and same(group["D"][:], raw["D"])
                  and same(group["lengths"][:], raw["lengths"]), "T1 full/compact bank differs")
        valid = np.isfinite(raw["C"])
        used = raw["mask"] & valid[:, None]
        check(raw["predictions"].shape == (int(used.sum()), 6), "T1 saved prediction row grouping differs")
        terms = np.zeros(raw["mask"].shape)
        terms[used] = 0.5*np.sum((raw["actions"][used]-raw["predictions"])**2, axis=-1)
        reconstructed = terms.sum(axis=1)
        discrepancy = float(np.max(np.abs(reconstructed[valid]-raw["D"][valid])))
        check(discrepancy <= 1e-10 and np.array_equal(raw["mask"].sum(1), raw["lengths"]),
              "T1 D_sum arithmetic or executable-prefix lengths differ")
        residual = raw["D"][valid]/raw["lengths"][valid]
        costs = raw["C"][valid]
        selected = dict(all=np.arange(len(costs)),
                        competitive=np.argsort(costs, kind="stable")[:32])
        measured = {}
        for basis, indices in selected.items():
            cstd, rstd = float(np.std(costs[indices], ddof=0)), float(np.std(residual[indices], ddof=0))
            check(cstd > 1e-12 and rstd > 1e-12 and np.isfinite([cstd, rstd]).all(),
                  "Degenerate frozen calibration spread")
            ratio = rstd/cstd
            ratios[basis].append(ratio)
            measured[basis] = dict(cstd=cstd, rstd=rstd, unit_ratio=ratio, candidates=len(indices))
        calibration_banks.append(dict(bank=bank_id, binding=entry["bank"], group=entry["group"],
            D_sum_max_arithmetic_error=discrepancy, finite_candidates=int(valid.sum()), spread=measured))
    calibration_targets = {}
    for basis, bank_ratios in ratios.items():
        median_ratio = float(np.median(bank_ratios))
        check(abs(median_ratio-calibration["median_unit_ratios"][basis]) <= 1e-12,
              "T1 median spread ratio differs")
        for target in config["targets"]:
            selected_alpha = float(target/median_ratio)
            check(abs(selected_alpha-calibration["alphas"][basis][str(target)]) <= 1e-12,
                  "T1 frozen target-to-alpha calculation differs")
            achieved = [selected_alpha*ratio for ratio in bank_ratios]
            check(abs(float(np.median(achieved))-target) <= 1e-12, "T1 median target not achieved")
            calibration_targets[f"{basis}/{target}"] = dict(alpha=selected_alpha,
                achieved_per_bank=achieved, achieved_bank_equal_median=float(np.median(achieved)))
    records = [row for row in supervisor["records"] if row["kind"] == "controller"]
    check(len(records) == 16, "All16 completed controller episodes required")
    expected = {(index, mode, coefficient) for index in range(4)
                for mode in ("zero", "shadow_carried") for coefficient in (0.0, alpha)}
    check({(row["index"], row["mode"], row["alpha"]) for row in records} == expected,
          "The frozen controller roster differs")
    task = config["task"]
    horizon, threshold = config["controls"], config["decision"]["action_rms_threshold"]
    check(horizon == 300 and threshold == 0.01, "Frozen endpoint/threshold changed")
    episodes, traces, banks, first_banks = [], {}, {}, {}
    work_counts = {}
    all_manifests = []
    for row in supervisor["records"]:
        manifest = load(row["manifest"])
        job = load(row["job"])
        check(row["exit_code"] == 0 and manifest["status"] == "COMPLETED", "Unclosed execution record")
        check(manifest["checkpoint"] == config["checkpoint"] and manifest["actor_hash"] == config["actor_hash"],
              "Different fixed actor binding")
        check(job["context"] == config["contexts"][row["index"]]["context"], "Context roster changed")
        bind(manifest["checkpoint"])
        bind(job["context"])
        receipt = load(Path(row["manifest"]["path"]).parent / "receipt.json")
        check(receipt["counts"] == manifest["work"], "Manifest/receipt work mismatch")
        check(not any(k.startswith("unknown_partial_") and value for k, value in receipt["counts"].items()),
              "Unresolved original execution work")
        for key, value in receipt["counts"].items():
            work_counts[key] = work_counts.get(key, 0) + value
        all_manifests.append(row["manifest"])
        if row["kind"] != "controller":
            bind(manifest["source_transition_replay"])
            bind(manifest["initial_plan"])
            continue
        key = (row["index"], row["mode"], row["alpha"])
        check(job["seed"] == config["seeds"]["controller_base"] + row["index"], "Paired private seed changed")
        with h5py.File(bind(manifest["trace"]), "r") as source:
            trace = {name: source[name][:] for name in source}
        states, actions, costs = trace["states"], trace["actions"], trace["costs"]
        n = len(actions)
        check(1 <= n <= horizon and states.shape == (n+1, 19) and actions.shape == (n, 6),
              "Invalid actual native N+1 prefix")
        check(all(np.isfinite(value).all() for value in trace.values()), "Nonfinite saved native arrays")
        check(states.dtype == actions.dtype == costs.dtype == np.float64 and np.all(np.abs(actions) <= 1),
              "Native float64 bounded torques required")
        obs = np.concatenate((states[:, 2:10], np.clip(states[:, 10:19], -10, 10)), axis=1)
        check(same(obs, trace["observations"]), "Incorrect pre-action feature/state pairing")
        good = (states[:, 2] > 0.8) & (states[:, 2] < 2) & (states[:, 3] > -1) & (states[:, 3] < 1)
        check(good[:-1].all(), "Post-fall states were appended")
        fell = not bool(good[-1])
        check(n == horizon or fell, "Unexplained incomplete episode")
        terminal = np.zeros(n)
        if fell:
            terminal[-1] = task["terminal_unhealthy_weight"]
        check(np.array_equal(trace["native_fall_cost"], terminal), "Native fall terminal not charged exactly once")
        # Native target-velocity running cost is reconstructed independently
        # from post-action physical states and actual executed controls.
        running = (task["vel_cost_weight"] * np.abs(task["target_velocity"]-states[1:, 10])
                   + task["vertical_velocity_cost_weight"] * states[1:, 11]**2
                   + task["angle_cost_weight"] * states[1:, 3]**2
                   + task["unhealthy_cost_weight"] * (~good[1:])
                   + task["ctrl_cost_weight"] * np.sum(actions**2, axis=1))
        cost_error = float(np.max(np.abs(costs-(running+terminal))))
        check(cost_error <= 1e-9, "Native task components disagree with saved costs")
        survival = n == horizon and not fell
        velocity_mae = float(np.mean(np.abs(states[1:, 10]-task["target_velocity"])))
        raw_cost = float(np.sum(costs))
        tail = float(task["unhealthy_cost_weight"]*(horizon-n)) if fell else 0.0
        outcome = dict(controls=n, requested_controls=horizon, fell=fell, full_survival=survival,
            walking=False, local_walking=bool(survival and velocity_mae <= 0.3),
            velocity_mae=velocity_mae, mean_velocity=float(np.mean(states[1:, 10])),
            raw_native_cost=raw_cost, absorbing_tail_cost=tail, complete_cost=raw_cost+tail)
        for name, value in outcome.items():
            saved = manifest["outcome"][name]
            if isinstance(value, float):
                check(abs(saved-value) <= 1e-9, "Outcome mismatch: " + name)
            else:
                check(saved == value, "Outcome mismatch: " + name)
        policy = trace["policy_actions"]
        check(policy.shape == actions.shape and np.all(np.abs(policy) <= 1), "Invalid stored deployed policy actions")
        gap = ((actions-policy)**2).mean(axis=1)
        check(np.allclose(gap, trace["action_gap_mse"], rtol=0, atol=1e-12), "Saved action-gap values differ")
        check(abs(float(gap[0])-manifest["first_action_gap_mse"]) <= 1e-12 and
              abs(float(gap.mean())-manifest["mean_executed_action_gap_mse"]) <= 1e-12,
              "Manifest deployed-policy gap differs")
        check(np.array_equal(actions[0], manifest["first_action"]), "Manifest first executed torque differs")
        with np.load(bind(manifest["executed_initial_plan"]), allow_pickle=False) as data:
            initial_plan = data["U"].copy()
        if row["mode"] == "zero":
            check(np.all(initial_plan == 0), "Zero arm has a nonzero initial plan")
        else:
            with np.load(bind(manifest["initial_plan"]), allow_pickle=False) as data:
                check(same(initial_plan, data["U"]), "Bound shadow nominal changed")
        compact = []
        with h5py.File(bind(manifest["bank_stats"]), "r") as data:
            check(len(data) == n+1 == manifest["banks"], "Two initial passes/one later pass census differs")
            for index, group in enumerate(data.values()):
                expected_control, expected_inner = (0, index) if index < 2 else (index-1, 0)
                check(int(group.attrs["control"]) == expected_control and int(group.attrs["inner"]) == expected_inner,
                      "Bank pre-action/initial-refinement indexing differs")
                compact.append(dict(innovation=str(group.attrs["innovation_fingerprint"]),
                    rng_before=json.loads(group.attrs["rng_before"]), rng_after=json.loads(group.attrs["rng_after"])))
        full = []
        for index, spec in enumerate(manifest["full_banks"]):
            with np.load(bind(spec), allow_pickle=False) as data:
                bank = {name: data[name].copy() for name in ("innovation", "actions", "posts", "C", "U_before", "U_after")}
            check(innovation_hash(bank["innovation"]) == compact[index]["innovation"],
                  "Raw first-bank innovations do not match the recorded fingerprint")
            full.append(bank)
        check(len(full) == 2 and same(full[0]["U_before"], initial_plan)
              and same(full[1]["U_before"], full[0]["U_after"]), "Initial plan/refinement chaining differs")
        check(same(full[1]["U_after"][0], actions[0]), "Executed first torque not from second initial refinement")
        episodes.append(dict(index=row["index"], mode=row["mode"], alpha=row["alpha"], **outcome,
            native_fall_terminal=float(terminal.sum()), native_cost_component_max_error=cost_error,
            first_teacher_policy_action_mse=float(gap[0]), mean_teacher_policy_action_mse=float(gap.mean()),
            trace=manifest["trace"], manifest=row["manifest"]))
        traces[key], banks[key], first_banks[key] = trace, compact, full
    pairs, identities, regressions = [], [], []
    for case in range(4):
        group = [row for row in episodes if row["index"] == case]
        for left, right in itertools.combinations(group, 2):
            a, b = (case, left["mode"], left["alpha"]), (case, right["mode"], right["alpha"])
            aa, bb = traces[a], traces[b]
            for name in ("states", "warmstarts", "integrations", "observations"):
                check(same(aa[name][0], bb[name][0]), "Paired complete initial context differs")
            n, m = min(len(aa["actions"]), len(bb["actions"])), min(len(banks[a]), len(banks[b]))
            check(banks[a][:m] == banks[b][:m], "Common bank-prefix innovations/private RNG differ")
            maximum = np.max(np.abs(aa["states"][1:n+1]-bb["states"][1:n+1]), axis=1)
            divergent = np.flatnonzero(maximum > 1e-10)
            pairs.append(dict(index=case, left_mode=left["mode"], right_mode=right["mode"],
                left_alpha=left["alpha"], right_alpha=right["alpha"], common_controls=n,
                contrast="alpha_same_initialization" if left["mode"] == right["mode"] else
                    "initialization_same_alpha" if left["alpha"] == right["alpha"] else "joint_alpha_and_initialization",
                first_action_rms=float(np.sqrt(np.mean((aa["actions"][0]-bb["actions"][0])**2))),
                common_prefix_action_rms=float(np.sqrt(np.mean((aa["actions"][:n]-bb["actions"][:n])**2))),
                common_banks=m, matching_innovation_and_rng_prefix=True,
                post_initial_state_max_abs=float(maximum.max()),
                first_divergent_boundary_gt1e_10=int(divergent[0]+1) if len(divergent) else None))
        for mode in ("zero", "shadow_carried"):
            a, b = (case, mode, 0.0), (case, mode, alpha)
            check(all(same(first_banks[a][0][name], first_banks[b][0][name])
                      for name in ("actions", "posts", "C", "U_before", "innovation")),
                  "Within-mode first cloud differs before alpha weighting")
            identities.append(dict(index=case, mode=mode, first_pass_actions_posts_C_U_innovation_byte_exact=True))
            negative = next(row for row in group if row["mode"] == mode and row["alpha"] == 0)
            positive = next(row for row in group if row["mode"] == mode and row["alpha"] == alpha)
            regressions.append(dict(index=case, mode=mode,
                earlier_fall=bool(positive["fell"] and (not negative["fell"] or positive["controls"] < negative["controls"])),
                lost_local_walking=bool(negative["local_walking"] and not positive["local_walking"]),
                lost_full_survival=bool(negative["full_survival"] and not positive["full_survival"])))
    alpha_pairs = [row for row in pairs if row["contrast"] == "alpha_same_initialization"]
    check(len(pairs) == 24 and len(alpha_pairs) == 8 and len(identities) == 8, "Comparison roster differs")
    effective = any(row["common_prefix_action_rms"] >= threshold for row in alpha_pairs)
    destabilizing = any(row["earlier_fall"] or row["lost_local_walking"] or row["lost_full_survival"] for row in regressions)
    decision = dict(effective=effective, destabilizing=destabilizing,
        verdict="destabilizing" if destabilizing else "ineffective" if not effective else "effective_without_declared_competence_regression",
        executed_action_rms_threshold=threshold, matched_regressions=regressions)
    comparison = dict(available=False)
    if report is not None and report.is_file():
        reporter = load(report)
        differences = []
        for name in ("effective", "destabilizing", "verdict", "executed_action_rms_threshold", "matched_regressions"):
            if reporter["decision"][name] != decision[name]:
                differences.append(dict(role="decision", field=name))
        for row in episodes:
            saved = next(value for value in reporter["episodes"] if
                (value["index"], value["mode"], value["alpha"]) == (row["index"], row["mode"], row["alpha"]))
            for name in ("controls", "fell", "full_survival", "local_walking", "complete_cost", "raw_native_cost",
                         "absorbing_tail_cost", "velocity_mae", "native_fall_terminal",
                         "first_teacher_policy_action_mse", "mean_teacher_policy_action_mse"):
                value = row[name]
                equal = abs(saved[name]-value) <= 1e-9 if type(value) is float else saved[name] == value
                if not equal:
                    differences.append(dict(role="episode", index=row["index"], mode=row["mode"], alpha=row["alpha"], field=name))
        for row in pairs:
            saved = next(value for value in reporter["pairs"] if all(value[name] == row[name] for name in
                ("index", "left_mode", "right_mode", "left_alpha", "right_alpha")))
            for name in ("common_controls", "first_action_rms", "common_prefix_action_rms",
                         "post_initial_state_max_abs", "first_divergent_boundary_gt1e_10"):
                value = row[name]
                equal = abs(saved[name]-value) <= 1e-12 if type(value) is float else saved[name] == value
                if not equal:
                    differences.append(dict(role="pair", index=row["index"], field=name))
        comparison = dict(available=True, binding=inputs[str(report.resolve())], differences=differences)
        check(not differences, "Independent saved measurements differ from the reporter")
    return dict(schema="t0-independent-saved-cross-check-1", status="PASSED_SAVED_ONLY_CROSS_CHECK",
        decision=decision, episodes=episodes, pairs=pairs, first_cloud_identities=identities,
        T1_saved_prediction_arithmetic=dict(banks=calibration_banks, targets=calibration_targets,
            scope="Same-actor saved prediction arithmetic; no new forwards or historical L3 audit closure"),
        original_work_counts=work_counts, original_supervisor_cpu_seconds=supervisor["cpu_seconds"],
        reporter_comparison=comparison, inputs=list(inputs.values()),
        limitations=["No native replay or new physics; this checks the saved measurements and bindings.",
            "No neural forwards; deployed actions are checked for saved indexing and arithmetic, not independently regenerated.",
            "Common-prefix later innovation evidence is its saved RNG/fingerprint records; raw innovations are independently hashed for both first banks.",
            "Four selected local contexts, not independent whole-task replications, learning evidence or closure of the separate L3 audit."],
        new_work=dict(physics_substeps=0, model_queries=0, neural_forwards=0, policy_updates=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    if args.child:
        resource.setrlimit(resource.RLIMIT_CPU, (17, 18))
        result = None
        started, cpu = time.monotonic(), time.process_time()
        try:
            result = verify(args.base.resolve(), args.report)
        except BaseException as error:
            result = dict(schema="t0-independent-saved-cross-check-1", status="FAILED_SAVED_ONLY_CROSS_CHECK",
                          error=repr(error), new_work=dict(physics_substeps=0, neural_forwards=0, policy_updates=0))
        finally:
            result["verification_source"] = file_binding(__file__)
            result["work"] = dict(cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-started,
                                  child_cpu_cap_seconds=17, physics_substeps=0, neural_forwards=0, policy_updates=0)
            save(out / "result.json", result)
        return int(result["status"] != "PASSED_SAVED_ONLY_CROSS_CHECK")
    resource.setrlimit(resource.RLIMIT_CPU, (3, 18))
    started, cpu = time.monotonic(), time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    code, error = None, None
    try:
        argv = [sys.executable, str(Path(__file__).resolve()), "--base", str(args.base), "--child"]
        if args.report:
            argv += ["--report", str(args.report)]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (out / "stdout.log").open("x") as stdout, (out / "stderr.log").open("x") as stderr:
            code = subprocess.run(argv, stdout=stdout, stderr=stderr, env=env, check=False).returncode
    except BaseException as exception:
        error = repr(exception)
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu = after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        save(out / "resource_receipt.json", dict(schema="saved-verification-resource-1", returncode=code,
            error=error, child_cpu_seconds=child_cpu, parent_cpu_seconds=time.process_time()-cpu,
            total_measured_cpu_seconds=child_cpu+time.process_time()-cpu, wall_seconds=time.monotonic()-started,
            aggregate_cpu_ceiling_seconds=20, child_cpu_cap_seconds=17, command_reserve_seconds=3,
            physics_substeps=0, model_queries=0, neural_forwards=0, policy_updates=0))
    return int(code != 0)


if __name__ == "__main__":
    raise SystemExit(main())
