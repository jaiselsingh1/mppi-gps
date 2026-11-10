"""PREREG_posloop.md (rss_checks/rl; mppi-gps venv): the minimal loop in position space (MPPI explores, the policy copies).
Action space: the chosen PD position model (runs/pos/chosen.json). Policy: 2 x 256 tanh MLP with a tanh output (the
targets live in [-1, 1]), inputs standardised with round 0's statistics.
Round 0: the policy copies MPPI's walks from standing (runs/pos/walks_pos.npz, the walks that walked), 10,000 updates.
Each round r >= 1:
  1. the policy runs from 16 fresh standing starts; one state per run, uniform in [0, lasted - 25] controls (at least
     0.2 s before a fall);
  2. MPPI (the chosen setting, fresh plan, never seeded) runs 375 controls (3 s) from each of those states, with its
     rollout cost plus w sum_t ||u_t - pi(x_t)||^2 (the 2015 action term, in target units; w = 0: no similarity);
  3. MPPI's states while up (dropping the last 25 controls before a fall) and its actions join the data; the policy is
     refit lightly: 2,000 Adam updates (lr 1e-3, batch 512) from its current weights on all data so far.
Measured each round: the policy alone on 32 fresh standing starts (walks, survived, median time up); MPPI's 3 s survival;
the mean squared MPPI-policy action gap on MPPI's new data, before and after the refit.
PREREG_posloop_left.md: START = left picks MPPI's start at the first control where the policy's run leaves the data
(nearest state among all data so far, standardised, farther than the leave-one-walk-out 95th percentile on MPPI's walks).
PREREG_posloop_torque.md: ARM = torque runs the same loop on the torque model (assets/walker2d.xml) from MPPI's torque
walks (runs/pos/walks_torque.npz), with the loop config's own noise sd.
PREREG_posloop_k4096.md: K overrides MPPI's sample count (default: the config's 256).
  posloop.py NAME W ROUNDS [START=uniform|left N_RUN=16 ARM=pos|torque K L=375]   -> runs/pos/loop_<NAME>/rounds.jsonl"""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import OUT, chosen, load, obs17, walked, xml_of

N_RUN, N_EVAL, L_MPPI = 16, 32, 375


def standing(seeds):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets/walker2d.xml")); q0 = m.qpos0.copy(); X = []
    for s in seeds:
        r = np.random.default_rng(int(s)); X.append(np.r_[0.0, q0 + r.uniform(-5e-3, 5e-3, m.nq), r.uniform(-5e-3, 5e-3, m.nv)])
    return np.array(X)


def mlp(layers, m, s, o):
    h = (o - m) / s
    for W, b in layers:
        h = np.tanh(h @ W.T + b)
    return h


def run_policy(job):
    """CPU closed loop of the numpy policy from one state; returns (lasted, mean |vx - 1.5| after 100 or None, states)."""
    layers, m, s, x0 = job[:4]; arm = job[4] if len(job) > 4 else "pos"
    import mujoco
    mm = mujoco.MjModel.from_xml_path(xml_of(arm)); d = mujoco.MjData(mm)
    mujoco.mj_setState(mm, d, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(mm, d); S, vx = [], []
    for t in range(1000):
        S.append(np.r_[0.0, d.qpos, d.qvel])
        d.ctrl[:] = mlp(layers, m, s, obs17(S[-1][None]))[0]
        for _ in range(4):
            mujoco.mj_step(mm, d)
        vx.append(d.qvel[0])
        if not (0.8 < d.qpos[1] < 2.0 and abs(d.qpos[2]) < 1.0):
            return t + 1, None, np.array(S)
    return 1000, float(np.mean(np.abs(np.array(vx[100:]) - 1.5))), np.array(S)


class Student:
    def __init__(self, O, seed):
        import torch
        torch.manual_seed(seed); self.torch = torch
        self.m, self.s = O.mean(0), O.std(0) + 1e-6
        self.net = torch.nn.Sequential(torch.nn.Linear(17, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(),
                                       torch.nn.Linear(256, 6), torch.nn.Tanh())
        self.g = torch.Generator().manual_seed(seed)

    def fit(self, O, A, updates):
        torch = self.torch; opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        x = torch.tensor((O - self.m) / self.s, dtype=torch.float32); y = torch.tensor(A, dtype=torch.float32)
        for _ in range(updates):
            b = torch.randint(0, len(x), (512,), generator=self.g); l = ((self.net(x[b]) - y[b]) ** 2).mean(); opt.zero_grad(); l.backward(); opt.step()

    def layers(self):
        return [(l.weight.detach().numpy().astype(np.float64), l.bias.detach().numpy().astype(np.float64))
                for l in self.net if isinstance(l, self.torch.nn.Linear)]

    def act(self, O):
        return mlp(self.layers(), self.m, self.s, O)


def leave_threshold(X0, Ss, keep, m, s):
    """95th percentile of MPPI's walk states' nearest state on the other walks (standardised obs; PREREG_copyfall.md)."""
    from scipy.spatial import cKDTree
    Ow = {k: (obs17(np.concatenate([X0[k][None], Ss[k][:-1]])) - m) / s for k in keep}; d = []
    for k in keep:
        d.append(cKDTree(np.concatenate([Ow[j] for j in keep if j != k])).query(Ow[k][::5])[0])
    return float(np.percentile(np.concatenate(d), 95))


def pick_starts(runs, rng, mode, tree=None, thr=None, m=None, s=None):
    """MPPI's start on each policy run: uniform in [0, fall - 25] (posloop), or the first control where the run leaves
    the data, never later than fall - 25 (uniform if it never leaves). Returns starts, leave controls, sources, start controls."""
    starts, leave, src, ctl = [], [], [], []
    for lasted, _, S in runs:
        hi = min(max(0, lasted - 25), len(S) - 1); j = None
        if mode == "left":
            out = np.nonzero(tree.query((obs17(S) - m) / s)[0] > thr)[0]
            if len(out):
                j = min(int(out[0]), hi); leave.append(int(out[0])); src.append("left")
            else:
                leave.append(None)
        if j is None:
            j = int(rng.integers(0, hi + 1)); src.append("uniform")
        starts.append(S[j]); ctl.append(j)
    return starts, leave, src, ctl


def main():
    import multiprocessing as mp
    from scipy.spatial import cKDTree
    import gpu_walks as gw
    from scripts.diagnostics.coupled_mppi import CONFIG
    from src.envs.walker2d import Walker2d
    from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
    name, w, rounds = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "uniform"; n_run = int(sys.argv[5]) if len(sys.argv) > 5 else N_RUN
    arm = sys.argv[6] if len(sys.argv) > 6 else "pos"; k_mppi = int(sys.argv[7]) if len(sys.argv) > 7 else None
    L = int(sys.argv[8]) if len(sys.argv) > 8 else L_MPPI   # MPPI controls per start
    d = OUT / f"loop_{name}"; d.mkdir(parents=True, exist_ok=True); logf = d / "rounds.jsonl"
    cfg = json.loads(CONFIG.read_text())
    if arm == "pos":
        cfg["noise_sigma"] = chosen()["sigma"]
    if k_mppi:
        cfg["K"] = k_mppi
    warp_cache = {}   # one Warp rollout batch reused across rounds
    X0, Ss, As = load(arm); keep = [k for k in range(len(Ss)) if walked(Ss[k])]
    O = np.concatenate([obs17(np.concatenate([X0[k][None], Ss[k][:-1]])) for k in keep]); A = np.concatenate([As[k] for k in keep])
    pol = Student(O, seed=0); pol.fit(O, A, 10_000)
    thr = leave_threshold(X0, Ss, keep, pol.m, pol.s) if mode == "left" else None
    EV = standing(61_000_000 + np.arange(N_EVAL)); pool = mp.get_context("spawn").Pool(16)

    def evaluate():
        g = pool.map(run_policy, [(pol.layers(), pol.m, pol.s, x, arm) for x in EV])
        lasted = np.array([q[0] for q in g])
        return dict(walks=int(sum(q[0] == 1000 and q[1] <= 0.3 for q in g)), survived=int((lasted == 1000).sum()), lasted_median=float(np.median(lasted)))
    ev = evaluate(); print(json.dumps(dict(round=0, rows=int(len(O)), **ev)), flush=True)
    with open(logf, "w") as f:
        f.write(json.dumps(dict(round=0, w=w, mode=mode, n_run=n_run, arm=arm, K=cfg["K"], L=L, noise_sigma=cfg["noise_sigma"], threshold=thr, rows=int(len(O)), judge=ev)) + "\n")
    obsf = lambda X: obs17(X)
    for r in range(1, rounds + 1):
        t0 = time.perf_counter()
        runs = pool.map(run_policy, [(pol.layers(), pol.m, pol.s, x, arm) for x in standing(62_000_000 + 1000 * r + np.arange(n_run))])
        rng = np.random.default_rng(63_000_000 + r)
        tree = cKDTree((O - pol.m) / pol.s) if mode == "left" else None
        starts, leave, src, ctl = pick_starts(runs, rng, mode, tree, thr, pol.m, pol.s)
        jobs = [dict(k=k, x0=starts[k], length=L, dart=0.0, seed=10_000 * r + k) for k in range(n_run)]
        envs = [Walker2d(**NATIVE_TASK, model_path=xml_of(arm), nthread=1) for _ in jobs]
        got = gw.lockstep_walks(envs, cfg, jobs, backend="warp", pull="action", w_eff=w, net=(pol.layers(), pol.m, pol.s),
                                observation=obsf, warp_cache=warp_cache)
        On, An, up3 = [], [], 0
        for g, x0 in zip(got, starts):
            S, U = g["states"], g["executed"]; ok = len(S) == L and ((0.8 < S[:, 2]) & (S[:, 2] < 2) & (np.abs(S[:, 3]) < 1)).all()
            up3 += ok; n = len(S) if ok else max(0, len(S) - 25)
            if n:
                On.append(obs17(np.concatenate([x0[None], S[:-1]]))[:n]); An.append(U[:n])
        On = np.concatenate(On) if On else np.zeros((0, 17)); An = np.concatenate(An) if An else np.zeros((0, 6))
        gap0 = float(np.mean(np.sum((pol.act(On) - An) ** 2, 1))) if len(On) else None
        O, A = np.concatenate([O, On]), np.concatenate([A, An]); pol.fit(O, A, 2_000)
        gap1 = float(np.mean(np.sum((pol.act(On) - An) ** 2, 1))) if len(On) else None
        ev = evaluate()
        lv = [q if q is not None else q2[0] for q, q2 in zip(leave, runs)] if mode == "left" else None
        row = dict(round=r, w=w, mode=mode, mppi_up_3s=int(up3), of=n_run, new_rows=int(len(On)), rows=int(len(O)), gap_before=gap0, gap_after=gap1,
                   policy_lasted_median_runs=float(np.median([q[0] for q in runs])), n_left=int(sum(c == "left" for c in src)),
                   leave_median=(float(np.median(lv)) if lv else None), start_to_fall_median=float(np.median([q[0] - c for q, c in zip(runs, ctl)])),
                   judge=ev, s=round(time.perf_counter() - t0))
        with open(logf, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(dict(round=r, mppi_up=int(up3), n_left=row["n_left"], leave=row["leave_median"], gap=(round(gap0, 4) if gap0 else None, round(gap1, 4) if gap1 else None), **ev, s=row["s"])), flush=True)
    pool.close()


if __name__ == "__main__":
    main()
