"""Protocol rule 8 (docs/experiment_log.md): is a learned arm still showing promise?

Reads each seed's metrics.jsonl and compares the mean dev time alive over the
arm's last three dev evaluations with its first three. From iteration 29 on, an
arm is futile if that rise is under `min_rise` seconds and no seed has completed
more than `max_completions` of its dev episodes. Prints a JSON verdict.
"""
import json
from pathlib import Path

import numpy as np
import tyro


def verdict(runs: list[list[dict]], start: int = 29, min_rise: float = 0.3, max_completions: int = 2) -> dict:
    evals = [{row["iteration"]: row["eval_dev"] for row in rows if "eval_dev" in row} for rows in runs]
    shared = sorted(set.intersection(*(set(e) for e in evals)))
    if len(shared) < 6 or shared[-1] < start:
        return {"decided": False, "evaluations": shared}
    mean_alive = [float(np.mean([e[i]["mean_steps"] * 0.008 for e in evals])) for i in shared]
    rise = float(np.mean(mean_alive[-3:]) - np.mean(mean_alive[:3]))
    best = max(round(e[i]["survival_rate"] * e[i]["episodes"]) for e in evals for i in shared)
    return {"decided": True, "iteration": shared[-1], "mean_dev_alive": mean_alive, "rise": rise,
            "best_seed_completions": best, "futile": rise < min_rise and best <= max_completions}


def main(metrics: tuple[Path, ...], start: int = 29, min_rise: float = 0.3, max_completions: int = 2) -> None:
    runs = [[json.loads(line) for line in path.read_text().splitlines() if line.strip()] for path in metrics]
    print(json.dumps(verdict(runs, start, min_rise, max_completions)))


if __name__ == "__main__":
    tyro.cli(main)
