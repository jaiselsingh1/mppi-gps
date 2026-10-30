"""Read-only post-hoc comparison of saved MPPI plans and realized continuations.

This reproduces the original scratch analysis without running a simulator.
Overlapping windows from two development episodes are descriptive evidence,
not independent trials or causal tests of replacing an MPPI action.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.envs.walker2d import Walker2d
from src.utils.trace_replay import load_episode

_S1 = _ROOT / "runs/2026-10-11_walker_instrumentation"
_S3 = _ROOT / "runs/2026-10-11_s3_shadow_001"
_ALIGNED_FIELDS = ("states", "warmstarts", "actions", "costs", "plans",
                   "pre_plan_nominals", "planner_rng_states", "initial_plan_flags")


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifact_path(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Artifact path escapes its bundle: {relative}")
    return path


def load_inputs(trace_path: Path, shadow_dir: Path, policy_label: str) -> dict:
    """Verify evidence hashes and decision alignment before using saved costs."""
    trace_path, shadow_dir = Path(trace_path), Path(shadow_dir)
    if not policy_label or Path(policy_label).name != policy_label:
        raise ValueError("policy_label must be a single directory name.")
    hash_path = shadow_dir / "artifact_hashes.json"
    hashes = json.loads(hash_path.read_text())
    required = {"manifest.json", "report.json", "traces.npz", "inputs/task.json",
                "inputs/mppi.json", f"{policy_label}/decisions.json"}
    if not required.issubset(hashes):
        raise ValueError(f"Missing hashed inputs: {sorted(required - hashes.keys())}")
    for relative, expected in hashes.items():
        path = _artifact_path(shadow_dir, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Shadow artifact hash mismatch: {relative}")
    manifest = json.loads((shadow_dir / "manifest.json").read_text())
    report = json.loads((shadow_dir / "report.json").read_text())
    if (report.get("status") != "complete" or report.get("shadow_only") is not True
            or report.get("reference_trace_exact") is not True or report.get("tau") != 0.0):
        raise ValueError("Require a completed, exact, tau=0 shadow audit.")
    if sha256(trace_path) != manifest["reference_sha256"]:
        raise ValueError("Reference trace hash differs from the shadow manifest.")
    task = json.loads((shadow_dir / "inputs/task.json").read_text())
    config = json.loads((shadow_dir / "inputs/mppi.json").read_text())
    if (hashes["inputs/task.json"] != manifest["task_sha256"]
            or hashes["inputs/mppi.json"] != manifest["config_sha256"]
            or task != manifest["task"]
            or any(manifest["mppi_config"].get(key) != value for key, value in config.items())):
        raise ValueError("Archived configuration disagrees with the manifest.")
    horizon = config["H"]
    absorbing_cost = task["unhealthy_cost_weight"]
    if task.get("cost_style") != "target_velocity":
        raise ValueError("This analysis requires the frozen target_velocity task.")
    sources = manifest["provenance"]["source_sha256"]
    for relative, expected in sources.items():
        if hashes.get(f"source/{relative}") != expected:
            raise ValueError(f"Archived source hash mismatch: {relative}")
    # Health rules and episode slicing come from these local modules; require
    # their exact audited versions, but do not construct or step an environment.
    for relative in ("src/envs/walker2d.py", "src/utils/trace_replay.py"):
        if sources.get(relative) != sha256(_ROOT / relative):
            raise ValueError(f"Current analysis dependency differs from audit: {relative}")
    if policy_label not in manifest["selection"]["checkpoints"]:
        raise ValueError("Policy is absent from the frozen selection.")
    reference = load_episode(trace_path, manifest["reference_episode"])
    shadow = load_episode(shadow_dir / "traces.npz", 0)
    count = len(reference["costs"])
    if report["completed_steps"] != count or report["requested_steps"] < count:
        raise ValueError("Shadow audit is not aligned with the full reference episode.")
    for key in _ALIGNED_FIELDS:
        if key not in reference or key not in shadow or not np.array_equal(reference[key], shadow[key]):
            raise ValueError(f"Reference/shadow field alignment mismatch: {key}")
    for key in ("reset_seed", "planner_seed"):
        if reference[key] != shadow[key] or reference[key] != manifest[key]:
            raise ValueError(f"Reference/shadow seed mismatch: {key}")
    if reference["plans"].shape != (count, horizon, reference["actions"].shape[1]):
        raise ValueError("Saved plan horizon/action dimensions disagree with configuration.")
    if not np.array_equal(reference["actions"], reference["plans"][:, 0]):
        raise ValueError("Executed actions do not match the first saved plan action.")
    decisions = json.loads((shadow_dir / policy_label / "decisions.json").read_text())["decisions"]
    if report["policies"][policy_label]["decisions"] != count:
        raise ValueError("Policy report decision count disagrees with the trace.")
    return {"episode": reference, "decisions": decisions, "horizon": horizon,
            "absorbing_cost": absorbing_cost, "reported_fall": report["fell"],
            "provenance": {"trace": str(trace_path.resolve()),
                "trace_sha256": sha256(trace_path), "shadow_dir": str(shadow_dir.resolve()),
                "artifact_hashes_sha256": sha256(hash_path),
                "input_sha256": {key: hashes[key] for key in sorted(required)},
                "audited_source_sha256": sources}}


def analyze_episode(episode: dict, decisions: list[dict], horizon: int,
                    absorbing_cost: float, reported_fall: bool) -> tuple[list[dict], dict]:
    """Sum actions t..t+H-1; pad only a verified terminal fall, never a timeout."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if not np.isfinite(absorbing_cost) or absorbing_cost < 0:
        raise ValueError("absorbing_cost must be finite and nonnegative.")
    costs = np.asarray(episode["costs"], dtype=float)
    states = np.asarray(episode["states"], dtype=float)
    count = len(costs)
    if (costs.shape != (count,) or count == 0 or states.shape != (count + 1, 19)
            or not np.isfinite(costs).all() or not np.isfinite(states).all()
            or len(decisions) != count):
        raise ValueError("Require finite Walker states and one cost/decision per transition.")
    healthy = Walker2d._is_healthy(states[:, 2], states[:, 3])
    if not healthy[:-1].all():
        raise ValueError("Trace has an unhealthy initial state or actions after its first fall.")
    fell = bool(not healthy[-1])
    if type(reported_fall) is not bool or reported_fall != fell:
        raise ValueError("Final-state health disagrees with the shadow fall report.")
    rows, skipped = [], 0
    for step, decision in enumerate(decisions):
        if type(decision["step"]) is not int or decision["step"] != step:
            raise ValueError("Decision indices must align with every trace transition.")
        baseline, policy = decision["baseline_cost"], decision["policy_cost"]
        accepted = decision["accepted"]
        if not np.isfinite(baseline) or (policy is not None and not np.isfinite(policy)):
            raise ValueError("Saved scores must be finite, or null for an invalid policy candidate.")
        if type(accepted) is not bool or (accepted and policy is None):
            raise ValueError("Accepted candidates must have a valid score.")
        if (type(decision["baseline_fell"]) is not bool
                or decision["policy_fell"] is not None and type(decision["policy_fell"]) is not bool):
            raise ValueError("Predicted fall flags must be boolean or null for an invalid policy.")
        expected = (policy is not None and decision["policy_fell"] is not None
                    and policy <= baseline
                    and (not decision["policy_fell"] or decision["baseline_fell"]))
        if accepted != expected:
            raise ValueError("Saved acceptance disagrees with the fixed tau=0 rule.")
        observed = min(horizon, count - step)
        padding = horizon - observed
        if padding and not fell:
            skipped += 1
            continue
        # Live costs already include the falling transition's terminal charge.
        # Adding a terminal charge here would double-count the fall.
        realized = float(costs[step:step + observed].sum() + padding * absorbing_cost)
        if not np.isfinite(realized):
            raise ValueError("Realized window cost is non-finite.")
        rows.append({"step": step, "baseline_cost": float(baseline),
                     "realized_cost": realized,
                     "policy_cost": None if policy is None else float(policy),
                     "accepted": accepted, "observed_steps": observed,
                     "absorbing_steps": padding,
                     "window_type": "absorbing_padded" if padding else "fully_observed"})
    counts = {"decisions": count, "fell": fell, "included": len(rows),
              "fully_observed": sum(row["absorbing_steps"] == 0 for row in rows),
              "absorbing_padded": sum(row["absorbing_steps"] > 0 for row in rows),
              "skipped_incomplete_nonfall": skipped,
              "accepted_total": sum(decision["accepted"] for decision in decisions),
              "accepted_included": sum(row["accepted"] for row in rows)}
    return rows, counts


def summarize(rows: list[dict]) -> dict:
    """Describe overlapping windows; empty/zero-denominator quantities are null."""
    plan = np.asarray([row["baseline_cost"] for row in rows], dtype=float)
    real = np.asarray([row["realized_cost"] for row in rows], dtype=float)
    gap = plan - real
    ratios = real[plan != 0] / plan[plan != 0]
    accepted = [row for row in rows if row["accepted"]]
    beating = sum(row["policy_cost"] <= row["realized_cost"] for row in accepted)
    return {"n": len(rows), "median_gap": float(np.median(gap)) if len(rows) else None,
            "mean_gap": float(gap.mean()) if len(rows) else None,
            "fraction_realized_cheaper": float(np.mean(gap > 0)) if len(rows) else None,
            "median_ratio": float(np.median(ratios)) if len(ratios) else None,
            "ratio_n": len(ratios), "accepted": len(accepted),
            "accepted_beating_realized": beating,
            "fraction_accepted_beating_realized": beating / len(accepted) if accepted else None}


def compare(success_trace: Path, failure_trace: Path, success_shadow: Path,
            failure_shadow: Path, policy_label: str = "strong") -> dict:
    rows, episodes, inputs = [], {}, {}
    semantics = None
    for label, trace, directory in (("success", success_trace, success_shadow),
                                     ("failure", failure_trace, failure_shadow)):
        data = load_inputs(trace, directory, policy_label)
        current = {"horizon": data["horizon"], "absorbing_cost": data["absorbing_cost"]}
        if semantics is not None and current != semantics:
            raise ValueError("Episodes use different scoring horizons or absorbing costs.")
        semantics = current
        episode_rows, counts = analyze_episode(data["episode"], data["decisions"],
                data["horizon"], data["absorbing_cost"], data["reported_fall"])
        rows.extend(dict(row, episode=label) for row in episode_rows)
        episodes[label] = {**counts, "summary": summarize(episode_rows)}
        inputs[label] = data["provenance"]
    return {"analysis": "post_hoc_plan_vs_realized", "policy_label": policy_label,
            **semantics, "groups": {"all": summarize(rows),
                "fully_observed": summarize([r for r in rows if r["absorbing_steps"] == 0]),
                "absorbing_padded": summarize([r for r in rows if r["absorbing_steps"] > 0])},
            "episodes": episodes, "inputs": inputs,
            "source_sha256": {"scripts/diagnostics/compare_walker_plan_realized.py": sha256(Path(__file__)),
                "src/envs/walker2d.py": sha256(_ROOT / "src/envs/walker2d.py"),
                "src/utils/trace_replay.py": sha256(_ROOT / "src/utils/trace_replay.py")},
            "limits": "Descriptive, post-hoc, overlapping windows from two development episodes; not independent trials. Policy-plan versus raw-MPPI realized cost is not the causal benefit of taking one policy action then returning to MPPI. Fully observed windows include a falling transition when it ends exactly at the window boundary. Padded windows use only the declared absorbing task cost after verified termination, never after non-fall truncation. No new simulation or policy evaluation was run.",
            "windows": rows}


def main(success_trace: Path = _S1 / "replay_reset30_planner20030/traces.npz",
         failure_trace: Path = _S1 / "replay_reset31_planner20031/traces.npz",
         success_shadow: Path = _S3 / "success", failure_shadow: Path = _S3 / "failure",
         policy_label: str = "strong", output: Path | None = None) -> None:
    """Print a summary; optionally save a complete report to a new JSON file."""
    if output is not None:
        output = Path(output)
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        protected = (Path(success_trace).parent, Path(failure_trace).parent,
                     Path(success_shadow), Path(failure_shadow))
        if any(output.resolve().is_relative_to(path.resolve()) for path in protected):
            raise ValueError("Output must be outside the input evidence directories.")
    report = compare(success_trace, failure_trace, success_shadow, failure_shadow, policy_label)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("analysis", "policy_label", "horizon",
                      "absorbing_cost", "groups", "episodes", "limits")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
