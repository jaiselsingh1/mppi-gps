"""Post hoc (not preregistered): commit-H versus first-action exact headroom on the saved banks."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/Users/jaisel/Documents/mppi-gps")
from scripts.credit.run_credit_stack import build_task
cfg = json.load(open("/Users/jaisel/Documents/mppi-gps/configs/credit_testbed.json"))
run = Path("/Users/jaisel/Documents/mppi-gps/runs/2026-10-26_credit_testbed_001")
out = {}
for key in [f"fork:{d}:0" for d in (-10, 1, 20, 60, 140)] + [f"car:{p:g}" for p in (0.004, 0.0015, 0.001, 0.0007)]:
    task = build_task(cfg, key, run)
    bank = pickle.load(open(run / "banks" / f"{key.replace(':', '_')}.pkl", "rb"))
    res = {}
    for source in ("alone", "exact_term"):
        h_commit, h_first, agree = [], [], []
        for d in bank:
            if d.source != source:
                continue
            o = d.options
            s1 = o.states[:, :1, :].astype(float); a0 = o.actions[:, :1, :].astype(float)
            q1 = task.env.running_cost(s1, a0)[:, 0] + task.exact(s1[:, 0, :])
            m = o.mean_index
            h_commit.append(d.q_exact[m] - d.q_exact.min()); h_first.append(q1[m] - q1.min())
            agree.append(float(np.argmin(d.q_exact) == np.argmin(q1)))
        h_commit, h_first = np.array(h_commit), np.array(h_first)
        res[source] = {"n": len(h_commit), "commit_share_ge5": float(np.mean(h_commit >= 5)), "first_share_ge5": float(np.mean(h_first >= 5)),
                       "commit_median": float(np.median(h_commit)), "first_median": float(np.median(h_first)),
                       "same_best_option": float(np.mean(agree))}
    out[key] = res
    print(key, json.dumps(res), flush=True)
json.dump(out, open(run / "posthoc_first_action_headroom.json", "w"), indent=1)
