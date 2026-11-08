"""Practice-start pool for the theory-aligned arm (rss_checks; PREREG_loop.md addendum ~01:55): states of plain
MPPI's walks from the task's standing starts (bench no-tie arm, runs/rc/simple/none_seed1-5, walks that do the
task by the judge's rule), full states [t, qpos, qvel] after each control; 20,000 sampled uniformly (seed 0).
These approximate d^{pi*}_rho, the optimal policy's state visitation from the task's starts."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path.home() / "mppi-gps-claude-sac"
X, n, kept = [], 0, 0
for s in range(1, 6):
    for r in range(16):
        f = ROOT / f"runs/rc/simple/none_seed{s}/runs_{r}.npz"
        if not f.exists():
            continue
        W = np.load(f)
        for k in W.files:
            x = W[k]; n += 1
            if len(x) < 1000 or np.mean(np.abs(x[100:1000, 10] - 1.5)) > 0.3:
                continue
            X.append(x[:1000]); kept += 1
X = np.concatenate(X)
pick = np.random.default_rng(0).choice(len(X), 20000, replace=False)
out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, x=X[pick])
print(f"walks {n}, kept {kept}, states {len(X)}, pool 20000; mean vx {X[pick, 10].mean():.2f}, mean z {X[pick, 2].mean():.2f}")
