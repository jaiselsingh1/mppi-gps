"""Recompute external-reference outcomes from saved traces, with no simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def summarize(directory: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for planner in ("sampling", "ilqg"):
        for context in range(8):
            path = directory / f"{planner}_{context}.csv"
            if not path.exists():
                continue
            with path.open() as stream:
                rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
            timestep = rows[1]["time"] - rows[0]["time"]
            expected_controls = round(10 / timestep)
            complete = (
                len(rows) == expected_controls + 1
                and rows[-1]["control"] == expected_controls
            )
            # Project cost convention: the executed action at t is charged
            # alongside its resulting state at t+1. Never use the next action.
            native_post_cost = 0.0
            official_pre_cost = 0.0
            official_reward_sum = 0.0
            for before, after in zip(rows, rows[1:]):
                velocity2 = after["v0"] ** 2 + after["v1"] ** 2
                tip_x = math.sin(after["q0"]) + math.sin(after["q0"] + after["q1"])
                native_post_cost += (
                    (4 - after["tip_height"]) ** 2
                    + tip_x**2
                    + 0.01 * (velocity2 + before["u"] ** 2)
                )
                distance = math.hypot(4 - after["tip_height"], tip_x)
                # Exact dm_control dense swingup tolerance: target radius .2m,
                # margin1, gaussian sigmoid, value_at_margin .1.
                official_reward_sum += (
                    1.0 if distance <= 0.2
                    else math.exp(
                        -0.5 * ((distance - 0.2) * math.sqrt(-2 * math.log(0.1))) ** 2
                    )
                )
                before_x = math.sin(before["q0"]) + math.sin(before["q0"] + before["q1"])
                official_pre_cost += (
                    25 * ((4 - before["tip_height"]) ** 2 + before_x**2)
                    + 0.5 * (before["v0"] ** 2 + before["v1"] ** 2)
                    + 0.025 * before["u"] ** 2
                )
            final = [row["tip_height"] for row in rows if row["time"] >= 8 - 1e-10]
            summaries.append(
                {
                    "planner": planner,
                    "context": context,
                    "complete": complete,
                    "saved_state_rows": len(rows),
                    "executed_controls": max(0, len(rows) - 1),
                    "peak_height": max(row["tip_height"] for row in rows),
                    "final_2s_minimum_height": min(final) if final else None,
                    "final_hold_success": complete and bool(final) and min(final) > 3.8,
                    "first_crossing_time": next(
                        (row["time"] for row in rows if row["tip_height"] > 3.8), None
                    ),
                    "native_post_action_cost_sum": native_post_cost,
                    "native_post_action_cost_time_integral": timestep * native_post_cost,
                    "mjpc_pre_action_cost_sum": official_pre_cost,
                    "dm_control_dense_swingup_reward_sum": official_reward_sum,
                    "dm_control_dense_swingup_reward_mean": official_reward_sum / max(1, len(rows)-1),
                    "last_counted_step": rows[-1]["steps"],
                    "max_abs_control": max(abs(row["u"]) for row in rows[:-1]),
                    "csv_sha256": file_hash(path),
                    "candidates_sha256": file_hash(path.with_suffix(".candidates.f64")),
                    "candidate_bytes": path.with_suffix(".candidates.f64").stat().st_size,
                }
            )
    result = {
        "no_physics_or_inference": True,
        "cost_indexing": "native cost uses x[t+1],u[t]; MJPC diagnostic uses x[t],u[t]",
        "height_observations": "saved control observations, including terminal; not a continuous-time certificate",
        "trials": summaries,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.directory, args.output)


if __name__ == "__main__":
    main()
