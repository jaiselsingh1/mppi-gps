"""Compare two closed saved measurement JSONs; no producer/import or raw replay."""

import hashlib
import json
import resource
import time
from pathlib import Path


def binding(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 3))
    out = Path(__file__).resolve().parent
    started, cpu = time.monotonic(), time.process_time()
    result = dict(schema="t0-independent-reporter-comparison-1", status="FAILED_COMPARISON")
    try:
        own_path, report_path = out / "result.json", out.parent / "t0_report_001/result.json"
        own, report = json.loads(own_path.read_text()), json.loads(report_path.read_text())
        result["inputs"] = [binding(own_path), binding(report_path)]
        if own["status"] != "PASSED_SAVED_ONLY_CROSS_CHECK" or report["status"] != "COMPLETED":
            raise ValueError("Both independent verification and reporter must be closed")
        differences, maxima = [], dict(episode_scalars=0.0, pair_scalars=0.0)
        for name in ("effective", "destabilizing", "verdict", "executed_action_rms_threshold", "matched_regressions"):
            if own["decision"][name] != report["decision"][name]:
                differences.append(dict(role="decision", field=name))
        for row in own["episodes"]:
            saved = next(value for value in report["episodes"] if all(value[name] == row[name]
                for name in ("index", "mode", "alpha")))
            for name in ("controls", "fell", "full_survival", "local_walking", "complete_cost",
                         "raw_native_cost", "absorbing_tail_cost", "velocity_mae", "native_fall_terminal",
                         "first_teacher_policy_action_mse", "mean_teacher_policy_action_mse"):
                value = row[name]
                if type(value) is float:
                    error = abs(saved[name]-value)
                    maxima["episode_scalars"] = max(maxima["episode_scalars"], error)
                    equal = error <= 1e-9
                else:
                    equal = saved[name] == value
                if not equal:
                    differences.append(dict(role="episode", index=row["index"], mode=row["mode"], alpha=row["alpha"], field=name))
        for row in own["pairs"]:
            identity = frozenset(((row["left_mode"], row["left_alpha"]),
                                  (row["right_mode"], row["right_alpha"])))
            saved = next(value for value in report["pairs"] if value["index"] == row["index"]
                and frozenset(((value["left_mode"], value["left_alpha"]),
                               (value["right_mode"], value["right_alpha"]))) == identity)
            for name in ("common_controls", "first_action_rms", "common_prefix_action_rms",
                         "post_initial_state_max_abs", "first_divergent_boundary_gt1e_10"):
                value = row[name]
                if type(value) is float:
                    error = abs(saved[name]-value)
                    maxima["pair_scalars"] = max(maxima["pair_scalars"], error)
                    equal = error <= 1e-12
                else:
                    equal = saved[name] == value
                if not equal:
                    differences.append(dict(role="pair", index=row["index"], field=name))
        alpha_pairs = [row for row in own["pairs"] if row["contrast"] == "alpha_same_initialization"]
        bad_pair = next(row for row in alpha_pairs if row["index"] == 1 and row["left_mode"] == "shadow_carried")
        bad_episodes = [row for row in own["episodes"] if row["index"] == 1 and row["mode"] == "shadow_carried"]
        result.update(differences=differences, maximum_absolute_scalar_differences=maxima,
            compared_episodes=len(own["episodes"]), compared_pairs=len(own["pairs"]),
            alpha_first_action_rms_range=[min(row["first_action_rms"] for row in alpha_pairs),
                                          max(row["first_action_rms"] for row in alpha_pairs)],
            alpha_common_prefix_rms_range=[min(row["common_prefix_action_rms"] for row in alpha_pairs),
                                           max(row["common_prefix_action_rms"] for row in alpha_pairs)],
            bad_case_pair=bad_pair,
            bad_case_episodes=[{name: row[name] for name in ("index", "mode", "alpha", "controls",
                "local_walking", "full_survival", "complete_cost", "raw_native_cost", "absorbing_tail_cost",
                "first_teacher_policy_action_mse", "mean_teacher_policy_action_mse")} for row in bad_episodes])
        if differences:
            raise ValueError("Saved reporter differs from independent measurements")
        result["status"] = "PASSED_SAVED_REPORTER_COMPARISON"
    except BaseException as error:
        result["error"] = repr(error)
    finally:
        result["source"] = binding(Path(__file__))
        result["work"] = dict(process_cpu_seconds=time.process_time()-cpu,
            wall_seconds=time.monotonic()-started, cpu_cap_seconds=2,
            physics_substeps=0, model_queries=0, neural_forwards=0, policy_updates=0)
        result["pair_join"] = "Unordered condition identities; compared RMS and absolute-deviation measures are symmetric. Original failed ordered-join comparison remains preserved."
        with (out / "reporter_comparison_after_v2.json").open("x") as target:
            json.dump(result, target, indent=2, allow_nan=False)
            target.write("\n")
    return int(result["status"] != "PASSED_SAVED_REPORTER_COMPARISON")


if __name__ == "__main__":
    raise SystemExit(main())
