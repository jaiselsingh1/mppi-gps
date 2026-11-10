"""PREREG_s2gps.md (PLAN_mppigps.md S2; rss_checks/rl; mppi-gps venv): the MPPI-GPS loop.
Round 0: pi copies 16 noise-injected MPPI-16k walks (noise 0.10 and 0.15, clean labels; 6,400 rows), 10,000 updates.
Each round r: MPPI (K 16,384, H 48, lambda 3, sigma 0.15; fresh plan at the start of each walk, its own carried plan after,
never set from pi) walks 400 controls from 4 fresh standing starts, executing its action + N(0, 0.15^2) noise, labels = its
clean action; ARM gps: MPPI's rollout cost + w_r sum_t ||u_t - pi(x_t)||^2 over its horizon, w_r = W_MAX * score(pi),
score = mean fraction of the 1,000 controls pi stays up on the 32 held-out starts (Kendall: the worse the policy, the less
MPPI should copy it); ARM nopull: w = 0. New rows join all data; pi refit 2,000 updates from its current weights; pi scored
alone on 32 held-out standing starts. Saves data and weights every round (resumable); stops at STOP_AT box time.
  s2_gps.py ARM ROUNDS  -> runs/root/s2_<ARM>/"""
import json, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import obs17

K, L, N_WALK, NOISE, W_MAX, STOP_AT = 16384, 400, 4, 0.15, 1.0, "07:30"


def d0():
    O, A = [], []
    for n in (0.1, 0.15):
        for f in (0, 4):
            Z = np.load(ROOT / f"runs/root/dart{n:g}_{f}.npz")
            for k in range(f, f + 4):
                O.append(obs17(Z[f"xbar_{k}"])); A.append(Z[f"labels_{k}"])
    return np.concatenate(O), np.concatenate(A)


def main():
    import multiprocessing as mp, torch
    import gpu_walks as gw
    from ksweep import cfg_of, envs_of
    from posloop import Student, standing
    from rootcause import T, cpu_run
    arm, rounds = sys.argv[1], int(sys.argv[2]); d = ROOT / f"runs/root/s2_{arm}"; d.mkdir(parents=True, exist_ok=True); logf = d / "rounds.jsonl"
    O0, A0 = d0(); pol = Student(O0, seed=0)
    done = [json.loads(l) for l in open(logf)] if logf.exists() else []
    if done:
        Z = np.load(d / "data.npz"); O, A = Z["O"], Z["A"]; pol.net.load_state_dict(torch.load(d / "pi.pt")); r0 = done[-1]["round"] + 1
        print(json.dumps(dict(resumed_at=r0, rows=int(len(O)))), flush=True)
    else:
        O, A = O0, A0; pol.fit(O, A, 10_000); r0 = 0
    EV = standing(61_000_000 + np.arange(32)); pool = mp.get_context("spawn").Pool(16); cache = {}

    def evaluate():
        g = pool.map(cpu_run, [("pos", (pol.layers(), pol.m, pol.s), x, T) for x in EV]); lasted = np.array([q[0] for q in g])
        return dict(walks=int(sum(q[0] == T and q[1] <= 0.3 for q in g)), survived=int((lasted == T).sum()),
                    lasted_median=float(np.median(lasted)), score=float(np.mean(lasted) / T))

    def save(row):
        np.savez(d / "data.npz", O=O, A=A); torch.save(pol.net.state_dict(), d / "pi.pt")
        with open(logf, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "walk_lengths"}), flush=True)
    if r0 == 0:
        ev = evaluate(); save(dict(round=0, arm=arm, rows=int(len(O)), judge=ev)); r0 = 1; last = ev
    else:
        last = done[-1]["judge"]
    for r in range(r0, rounds + 1):
        if datetime.now().strftime("%H:%M") >= STOP_AT and datetime.now().hour < 12:
            print(json.dumps(dict(stopped_at=datetime.now().strftime("%H:%M"), round=r)), flush=True); break
        t0 = time.perf_counter(); w = W_MAX * last["score"] if arm == "gps" else 0.0
        X = standing(73_000_000 + 100 * r + np.arange(N_WALK))
        jobs = [dict(k=k, x0=X[k], length=L, dart=NOISE, seed=74_000_000 + 100 * r + k) for k in range(N_WALK)]
        kw = dict(pull="action", w_eff=w, net=(pol.layers(), pol.m, pol.s), observation=obs17) if w > 0 else {}
        got = gw.lockstep_walks(envs_of(N_WALK), cfg_of(K), jobs, backend="warp", warp_cache=cache, **kw)
        On, An = [], []
        for g in got:
            x, lab = g["xbar"], g["labels"]; n = len(x) if len(x) == L else max(0, len(x) - 25)
            On.append(obs17(x[:n])); An.append(lab[:n])
        On, An = np.concatenate(On), np.concatenate(An)
        gap0 = float(np.mean(np.sum((pol.act(On) - An) ** 2, 1)))
        O, A = np.concatenate([O, On]), np.concatenate([A, An]); pol.fit(O, A, 2_000)
        gap1 = float(np.mean(np.sum((pol.act(On) - An) ** 2, 1)))
        last = evaluate()
        save(dict(round=r, arm=arm, w=w, walk_lengths=[len(g["xbar"]) for g in got], mppi_walked=int(sum(len(g["xbar"]) == L and float(np.mean(np.abs(g["xbar"][100:, 10] - 1.5))) <= 0.3 for g in got)),
                  new_rows=int(len(On)), rows=int(len(O)), gap_before=gap0, gap_after=gap1, judge=last, s=round(time.perf_counter() - t0)))
    pool.close()


if __name__ == "__main__":
    main()
