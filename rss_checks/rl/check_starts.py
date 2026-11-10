"""Check posloop's start picking without MPPI: round-0 copy, round-1 policy runs, both modes."""
import json, sys
from pathlib import Path
ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks")); sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np


def main():
    import multiprocessing as mp
    from scipy.spatial import cKDTree
    from pos_check import load, obs17, walked
    from posloop import Student, run_policy, standing, leave_threshold, pick_starts
    X0, Ss, As = load("pos"); keep = [k for k in range(len(Ss)) if walked(Ss[k])]
    O = np.concatenate([obs17(np.concatenate([X0[k][None], Ss[k][:-1]])) for k in keep]); A = np.concatenate([As[k] for k in keep])
    pol = Student(O, seed=0); pol.fit(O, A, 10_000); thr = leave_threshold(X0, Ss, keep, pol.m, pol.s)
    with mp.get_context("spawn").Pool(16) as pool:
        runs = pool.map(run_policy, [(pol.layers(), pol.m, pol.s, x) for x in standing(62_000_000 + 1000 + np.arange(32))])
    tree = cKDTree((O - pol.m) / pol.s); out = dict(threshold=thr)
    for mode in ("uniform", "left"):
        st, lv, src, ctl = pick_starts(runs, np.random.default_rng(63_000_001), mode, tree, thr, pol.m, pol.s)
        out[mode] = dict(n_left=sum(c == "left" for c in src), start_ctl_median=float(np.median(ctl)),
                         start_to_fall_median=float(np.median([q[0] - c for q, c in zip(runs, ctl)])),
                         share_before_leaving=None)
    # old posloop rule, first 16 runs: same picks as uniform mode?
    rng = np.random.default_rng(63_000_001); old = []
    for lasted, _, S in runs:
        hi = max(0, lasted - 25); old.append(int(rng.integers(0, min(hi, len(S) - 1) + 1)))
    _, lv, _, _ = pick_starts(runs, np.random.default_rng(0), "left", tree, thr, pol.m, pol.s)
    _, _, _, cu = pick_starts(runs, np.random.default_rng(63_000_001), "uniform")
    out["uniform_matches_old_rule"] = old == cu
    out["uniform"]["share_before_leaving"] = float(np.mean([l is not None and c < l for c, l in zip(cu, lv)]))
    out["lasted_median"] = float(np.median([q[0] for q in runs])); out["leave_median"] = float(np.median([l for l in lv if l is not None]))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
