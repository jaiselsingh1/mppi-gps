"""PREREG_rootcause.md phase 2 FIX evaluation (CPU): copies of the noise-injected MPPI-16k walks (DART: executed with
noise, MPPI's clean actions as labels), and the orbit measure for every copy.
Orbit measure: from mid-walk states of the teacher's own runs, the copy's distance (pooled sd, rms per input) to the
NEAREST state anywhere in its teacher's runs, over time (does the copy stay on the teacher's orbit, at any phase?).
  rootcause_fix.py -> runs/root/fix.json"""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import obs17
from rootcause import OUT, T, cpu_run, fit_pl, fit_s1, mlp, mppi_data, sac_data, standing


def dart_data():
    W = []
    for f in (0, 4):
        Z = np.load(OUT / f"dart_{f}.npz")
        for k in range(f, f + 4):
            W.append((Z[f"xbar_{k}"], obs17(Z[f"xbar_{k}"]), Z[f"labels_{k}"], Z[f"executed_{k}"]))
    return W


def main():
    import multiprocessing as mp
    from scipy.spatial import cKDTree
    S, M, D = sac_data(), mppi_data(), dart_data(); res = {}
    res["dart_walks"] = dict(lengths=[len(x) for x, _, _, _ in D], vx_err=[float(np.mean(np.abs(x[100:, 10] - 1.5))) for x, _, _, _ in D if len(x) > 100],
                             label_vs_executed_rms=float(np.sqrt(np.mean(np.concatenate([(l - e) ** 2 for _, _, l, e in D])))))
    print(json.dumps(dict(dart_walks=res["dart_walks"])), flush=True)
    sacO = np.concatenate([o for _, o, _ in S[:9]]); sacA = np.concatenate([a for _, _, a in S[:9]])
    mpO = np.concatenate([o for _, o, _ in M]); mpA = np.concatenate([a for _, _, a in M])
    dO = np.concatenate([o for _, o, _, _ in D]); dA = np.concatenate([l for _, _, l, _ in D])
    pooled = np.concatenate([sacO, mpO]).std(0) + 1e-6
    # spread of each dataset on the common scale: distance between different runs at the same control, controls 0-399
    def between(Os):
        L = min(400, min(len(o) for o in Os)); Z = [(o[:L] - 0) / pooled for o in Os]
        return float(np.mean([np.sqrt(np.mean((Z[i] - Z[j]) ** 2, 1)).mean() for i in range(len(Z)) for j in range(i + 1, len(Z))]))
    res["spread_common"] = dict(sac_8x400=between([S[i][1] for i in range(8)]), sac_one_x8=0.0, mppi16k=between([o for _, o, _ in M]),
                                mppi16k_dart=between([o for _, o, _, _ in D]))
    print(json.dumps(dict(spread_common=res["spread_common"])), flush=True)
    EV = standing(61_000_000 + np.arange(32)); nets = {}
    arms = dict(FIX_dart_pl=("pos", dO, dA, fit_pl), FIX_dart_s1=("pos", dO, dA, fit_s1))
    with mp.get_context("spawn").Pool(24) as pool:
        for name, (space, O, A, fit) in arms.items():
            res[name] = []
            for seed in (0, 1, 2):
                net = fit(O, A, seed); nets[(name, seed)] = net
                g = pool.map(cpu_run, [(space, net, x, T) for x in EV]); lasted = np.array([q[0] for q in g])
                res[name].append(dict(seed=seed, rows=int(len(O)), fit_mse_per_joint=float(np.mean((mlp(net, O) - A) ** 2)),
                                      walks=int(sum(q[0] == T and q[1] <= 0.3 for q in g)), survived=int((lasted == T).sum()),
                                      lasted_median=float(np.median(lasted))))
                print(json.dumps(dict(arm=name, **res[name][-1])), flush=True)
        # orbit measure
        others = dict(sac_copy=("torque", fit_s1(sacO, sacA, 0), S, [o for _, o, _ in S]),
                      sac_break=("torque", fit_s1(np.concatenate([S[0][1][:400]] * 8), np.concatenate([S[0][2][:400]] * 8), 0), S, [o for _, o, _ in S]),
                      mppi_copy=("pos", fit_pl(mpO, mpA, 0), M, [o for _, o, _ in M]),
                      dart_copy=("pos", nets[("FIX_dart_pl", 0)], M, [o for _, o, _ in M]))
        res["orbit"] = {}
        for name, (space, net, runs, teacherO) in others.items():
            tree = cKDTree(np.concatenate(teacherO) / pooled)
            starts = [r[0][t] for r in runs[:8] for t in (150, 250, 340)]
            g = pool.map(cpu_run, [(space, net, x, 300) for x in starts]); curves = []
            for lasted, _, Oc in g:
                curves.append(tree.query(Oc / pooled)[0] / np.sqrt(17))
            at = lambda c: float(np.median([c_[c] for c_ in curves if len(c_) > c])) if any(len(c_) > c for c_ in curves) else None
            res["orbit"][name] = dict(lasted=[int(q[0]) for q in g], up_300=int(sum(q[0] == 300 for q in g)), of=len(g),
                                      dist_to_teacher_states={str(c): at(c) for c in (0, 25, 50, 100, 200)})
            print(json.dumps({name: {k: v for k, v in res["orbit"][name].items() if k != "lasted"}}), flush=True)
    json.dump(res, open(OUT / "fix.json", "w"), indent=1)


if __name__ == "__main__":
    main()
