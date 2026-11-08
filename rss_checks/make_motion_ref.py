"""MPPI's walking transitions for the motion reward (rss_checks; PREREG_motion.md).

Source: the bench's no-tie arm (runs/rc/simple/none_seed1-5, rounds 0-15). That's plain MPPI from 6
training starts a round, never pulled toward the policy, no executed-action noise. Only walks that
do the task are kept (Kendall 2026-11-04: keep only trajectories that do the task), by the judge's
rule: last 1,000 controls and mean |vx - 1.5| <= 0.3 after control 100.
Observation as our_walker.obs_of: qpos[1:], qvel clipped at +-10 (state x = [t, qpos(9), qvel(9)]).
Writes out/ref.npz (o, o2) and out/ref.json (provenance).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path.home() / "mppi-gps-claude-sac"


def obs(x):
    return np.concatenate([x[:, 2:10], np.clip(x[:, 10:19], -10.0, 10.0)], 1).astype(np.float32)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    O, O2, kept, seen = [], [], [], 0
    for s in range(1, 6):
        d = ROOT / f"runs/rc/simple/none_seed{s}"
        for r in range(16):
            f = d / f"runs_{r}.npz"
            if not f.exists():
                continue
            W = np.load(f)
            for k in W.files:
                x = W[k]; seen += 1
                if len(x) < 1000 or abs(np.mean(x[100:1000, 10]) - 1.5) > 0.3 or np.mean(np.abs(x[100:1000, 10] - 1.5)) > 0.3:
                    continue
                o = obs(x[:1000]); O.append(o[:-1]); O2.append(o[1:]); kept.append([s, r, k])
    o, o2 = np.concatenate(O), np.concatenate(O2)
    np.savez_compressed(out / "ref.npz", o=o, o2=o2)
    info = dict(source="runs/rc/simple/none_seed1-5 rounds 0-15 (plain MPPI, no tie, no executed noise)",
                rule="1,000 controls and mean |vx - 1.5| <= 0.3 after control 100", walks_seen=seen,
                walks_kept=len(kept), transitions=int(len(o)), kept=kept,
                obs_mean=o.mean(0).round(3).tolist(), torso_height_mean=float(o[:, 0].mean()), vx_mean=float(o[:, 8].mean()))
    (out / "ref.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "kept"}))


if __name__ == "__main__":
    main()
