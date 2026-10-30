"""Post hoc (not preregistered): paired scorer versus ridge strength on the fork."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, "/Users/jaisel/Documents/mppi-gps")
cfg = json.load(open("/Users/jaisel/Documents/mppi-gps/configs/credit_testbed.json"))
RUN = "/Users/jaisel/Documents/mppi-gps/runs/2026-10-26_credit_testbed_001"

def job(args):
    d, noise, seed = args
    from src.credit import study, estimators as est
    from scripts.credit.run_credit_stack import _unit_rng, _planner_seed
    key = f"fork:{d}:{noise:g}"
    task = study.fork_task(cfg["fork"], d, noise)
    setup = json.load(open(f"{RUN}/setup/fork_{d}_{noise:g}.json"))
    alone, exact = setup["closed_loop"]["alone"]["mean_model_cost"], setup["closed_loop"]["exact_term"]["mean_model_cost"]
    units, rows = [], []
    ctrl = task.controller()
    for n in (16, 64, 256):
        while len(units) < n:
            u = len(units)
            units.append(study.collect_unit(task, ctrl, _unit_rng(cfg, key, seed, 1, u), _planner_seed(cfg, seed, 1, u), 3))
        pairs = study.unit_pairs(task, units)
        for ridge in (1.0, 0.1, 0.01):
            V = est.fit_paired(task.view, pairs, ridge)
            eps = study.evaluate(task, task.controller(terminal=V), 40, cfg["common"]["seeds"])
            j = study.summarize(task, eps)["mean_model_cost"]
            rows.append({"d": d, "noise": noise, "seed": seed, "outcomes": 4 * n, "ridge": ridge, "terminal_share": (alone - j) / (alone - exact)})
    return rows

if __name__ == "__main__":
    jobs = [(d, n, s) for d in (20, 140) for n in (0.0, 2.0) for s in (0, 1)]
    with ProcessPoolExecutor(8) as pool:
        rows = [r for out in pool.map(job, jobs) for r in out]
    json.dump(rows, open(f"{RUN}/posthoc_paired_ridge.json", "w"), indent=1)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["d"], r["noise"], r["outcomes"], r["ridge"])].append(r["terminal_share"])
    for k in sorted(agg):
        print(k, np.round(agg[k], 2))
