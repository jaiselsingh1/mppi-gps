"""PREREG_rootcause.md phase 2 FIX data: MPPI-16k walks with noise on the executed actions (DART), clean labels.
  rootcause_dart.py FIRST COUNT DART -> runs/root/dart<DART>_<FIRST>.npz (xbar, labels, executed)"""
import json, sys, time
from pathlib import Path
ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks")); sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from ksweep import cfg_of, envs_of


def main():
    import gpu_walks as gw
    first, count, dart = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    X = np.load(ROOT / "rss_checks/rl/standing_starts.npz")["states"]
    jobs = [dict(k=k, x0=X[k], length=400, dart=dart, seed=900 + k) for k in range(first, first + count)]
    t0 = time.perf_counter(); got = gw.lockstep_walks(envs_of(count), cfg_of(16384), jobs, backend="warp")
    np.savez(ROOT / f"runs/root/dart{dart:g}_{first}.npz", **{f"{f}_{g['k']}": g[f] for g in got for f in ("xbar", "labels", "executed")})
    print(json.dumps(dict(dart=dart, lengths=[len(g["xbar"]) for g in got], s=round(time.perf_counter() - t0))), flush=True)


if __name__ == "__main__":
    main()
