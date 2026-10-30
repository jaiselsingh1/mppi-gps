"""Saved-only primary survival, tracking and native cost decomposition.

This is a portable version of the guided-data survival detail proof. It also
accepts L1 and H4.1 closed runs with the same native episode evidence schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .common import assert_independent_imports, require
from .native import Audit, sha


def primary_details(run: Path) -> dict[str, Any]:
    result = json.loads((run / "result.json").read_text())
    primary_tags = {tag for score in result["cases"]["primary"].values() for tag in score["tags"]}
    events = {}
    with (run / "evidence/ledger.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["event"] == "episode_complete" and row["tag"] in primary_tags:
                events[row["tag"]] = row
    out = {}
    with h5py.File(run / "evidence/raw.h5", "r") as f:
        for arm, score in result["cases"]["primary"].items():
            rows = []
            for tag in score["tags"]:
                event = events[tag]
                g = f[event["raw"]["group"]]
                values = {
                    key: np.asarray(g[key][()])
                    for key in [
                        "states",
                        "actions",
                        "costs",
                        "absorbing_cost",
                        "terminal_adjustment",
                        "terminal_cost",
                    ]
                }
                for key, value in values.items():
                    require(
                        hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                        == event["raw"]["arrays"][key]["sha256"],
                        "Saved-only proof failed",
                    )
                s = values["states"]
                u = values["actions"].astype(float)
                height = s[-1, 2] > 0.8 and s[-1, 2] < 2
                angle = s[-1, 3] > -1 and s[-1, 3] < 1
                classification = (
                    "full_survival"
                    if height and angle
                    else "height_and_angle"
                    if not height and (not angle)
                    else "height"
                    if not height
                    else "angle"
                )
                costs = dict(
                    tracking=float((2 * np.abs(1.5 - s[1:, 10])).sum()),
                    vertical=float((0.5 * s[1:, 11] ** 2).sum()),
                    angle=float((0.1 * s[1:, 3] ** 2).sum()),
                    control=float((0.001 * np.sum(u * u, axis=1)).sum()),
                    unhealthy=0.0 if classification == "full_survival" else 5.0,
                    fall_terminal=float(values["terminal_cost"]),
                    terminal_adjustment=float(values["terminal_adjustment"]),
                    absorbing=float(values["absorbing_cost"]),
                )
                require(
                    np.isclose(sum(costs.values()), event["cost"], atol=1e-09, rtol=1e-12),
                    "Saved-only proof failed",
                )
                mae = float(np.abs(s[1:, 10] - 1.5).mean())
                survived = len(u) == 1000 and classification == "full_survival"
                require(
                    event["mae"] == mae
                    and event["survived"] == survived
                    and (event["success"] == (survived and mae <= 0.3)),
                    "Saved-only proof failed",
                )
                rows.append(
                    dict(
                        tag=tag,
                        seed=event["seed"],
                        steps=len(u),
                        success=event["success"],
                        survived=survived,
                        velocity_mae=mae,
                        mean_x_velocity=float(s[1:, 10].mean()),
                        final_height=float(s[-1, 2]),
                        final_angle=float(s[-1, 3]),
                        termination=classification,
                        complete_cost=event["cost"],
                        cost_components=costs,
                    )
                )
            out[arm] = rows
    return dict(
        schema="walker-primary-native-outcome-details-v1",
        cases=out,
        new_controls=0,
        new_neural_rows=0,
        new_backward_rows=0,
        new_optimizer_steps=0,
    )


def main(argv: list[str] | None = None) -> int:
    """Write primary native cost details to a fresh JSON output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    result = primary_details(run)
    result["input_result_sha256"] = sha(run / "result.json")
    result["input_ledger_sha256"] = sha(run / "evidence/ledger.jsonl")
    result["script_sha256"] = sha(Path(__file__))
    assert_independent_imports(Audit(run, run / "source"))
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
