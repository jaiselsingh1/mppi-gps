"""Can the policy copy MPPI if it may pick any of MPPI's good samples? (rss_checks; readings in
PREREG_goodset.md, written before this runs.)

Same walks and test as diag_noise.py: 2 standing starts x 3 plain-MPPI walks, executed-action noise
0 and 0.1, same seeds. Every control additionally records MPPI's 64 lowest-cost sampled first
actions and their costs (from that control's final update). MPPI's label is the weighted average of
those samples; here the policy may instead match whichever good sample is nearest to its own output.

Per (start, noise) and fit seed 1-3, four policies (same network, 3,000 Adam updates, batch 256):
  copy    MPPI's planned action, the weighted average (diag_noise's label; seed 1 reproduces it)
  best    the single lowest-cost sample's first action
  set_lam nearest of the samples within lam of the best (MPPI's own temperature: weight >= e^-1)
  set_32  nearest of the 32 lowest-cost samples
Test as diag_noise: takeovers from the walks' states every 0.4 s between 2 and 6 s, 4 s alone; and
the 32 standing starts, 8 s.
"""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
import numpy as np, torch

import os
QUICK = os.environ.get("QUICK") == "1"  # plumbing only: 60 controls, 50 updates, fit seed 1; not a reading
M = 64
LENGTH, UPDATES, SEEDS = (60, 50, (1,)) if QUICK else (1000, 3000, (1, 2, 3))
LEVELS = (0.0, 0.1)
ARMS = ("copy", "best", "set_lam", "set_32")


def record_samples():
    """Patch MPPI so each planner keeps (first actions, cost - min, lam) of its 64 best samples per update."""
    from src.mppi import mppi as mm
    reg = []
    init0, noise0, soft0 = mm.MPPI.__init__, mm.MPPI._sample_noise, mm.MPPI._softmin_weights

    def init(self, *a, **k):
        init0(self, *a, **k); self._gs, self._gs_a0 = [], None; reg.append(self)

    def noise(self):
        n = noise0(self)
        self._gs_a0 = np.clip(self.U[0][None] + n[:, 0, :], self.action_low, self.action_high)
        return n

    def soft(self, S, lam):
        out = soft0(self, S, lam)
        if self._gs_a0 is not None and len(S) == len(self._gs_a0):
            i = np.argsort(S)[:M]
            self._gs.append((self._gs_a0[i].astype(np.float32), (S[i] - S.min()).astype(np.float32), float(lam)))
            self._gs_a0 = None
        return out

    mm.MPPI.__init__, mm.MPPI._sample_noise, mm.MPPI._softmin_weights = init, noise, soft
    return reg


def collect(out):
    f = out / "walks.npz"
    if f.exists():
        return
    import gpu_walks as gw
    from scripts.diagnostics import mppi_loop as ml
    from scripts.diagnostics.coupled_mppi import CONFIG
    cfg = json.loads(CONFIG.read_text())
    reg = record_samples()
    env = ml.make_env("walker", 1)
    starts = []
    for j in range(2):
        st = np.random.get_state(); np.random.seed(90_000_000 + 1000 + j); env.reset(); np.random.set_state(st); starts.append(env.get_state())
    jobs = [dict(k=6 * s + 3 * li + r, x0=starts[s], length=LENGTH, dart=lv, seed=5_000_000 + 100 * s + 10 * li + r)
            for s in range(2) for li, lv in enumerate(LEVELS) for r in range(3)]
    envs = [ml.make_env("walker", 1) for _ in jobs]
    got = gw.lockstep_walks(envs, cfg, jobs, backend="warp")
    first = int(cfg.get("initial_iterations", 1))
    save = {}
    for g, p in zip(got, reg):
        T = len(g["labels"]); rec = p._gs[first - 1:first - 1 + T]
        assert len(rec) == T, (len(p._gs), T)
        A = np.stack([r[0] for r in rec]); C = np.stack([r[1] for r in rec])
        # the label is the weighted average of all K samples; check it against the top-64 average as a sanity
        save.update({f"x_{g['k']}": g["xbar"], f"u_{g['k']}": g["labels"], f"a_{g['k']}": A, f"c_{g['k']}": C,
                     f"lam_{g['k']}": np.array([r[2] for r in rec])})
    np.savez_compressed(f, **save)


def fit_and_test(args):
    out, s, li, arm, seed = args
    from scripts.diagnostics import mppi_loop as ml
    from scripts.diagnostics.feedback_rollouts import step
    from scripts.diagnostics.noise_cutoff import observation
    from scripts.diagnostics.policy_survival import standing_starts
    from simple_loop import mlp
    torch.set_num_threads(1)
    env = ml.make_env("walker", 1)
    W = np.load(out / "walks.npz")
    ks = [6 * s + 3 * li + r for r in range(3)]
    X = np.concatenate([W[f"x_{k}"] for k in ks]); U = np.concatenate([W[f"u_{k}"] for k in ks])
    A = np.concatenate([W[f"a_{k}"] for k in ks]); C = np.concatenate([W[f"c_{k}"] for k in ks])
    lam = np.concatenate([W[f"lam_{k}"] for k in ks])
    o = observation(X); mean, std = o.mean(0), o.std(0) + 1e-6
    if arm == "set_lam":
        mask = C <= lam[:, None]
    elif arm == "set_32":
        mask = np.zeros(C.shape, bool); mask[:, :32] = True
    else:
        mask = None
    torch.manual_seed(seed); net = mlp(); opt = torch.optim.Adam(net.parameters(), lr=3e-4); g = torch.Generator().manual_seed(seed)
    Xt = torch.as_tensor(((o - mean) / std).astype(np.float32))
    Ut = torch.as_tensor((A[:, 0] if arm == "best" else U).astype(np.float32))
    At = torch.as_tensor(A.astype(np.float32)); Mt = None if mask is None else torch.as_tensor(mask)

    def loss_of(idx):
        y = net(Xt[idx])
        if Mt is None:
            return ((y - Ut[idx]) ** 2).mean()
        d = ((y[:, None, :] - At[idx]) ** 2).mean(-1)
        return torch.where(Mt[idx], d, torch.full_like(d, float("inf"))).min(1).values.mean()

    for _ in range(UPDATES):
        i = torch.randint(0, len(Xt), (256,), generator=g); loss = loss_of(i)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        y = net(Xt).numpy()
        fit_err = float(loss_of(torch.arange(len(Xt))))
    if mask is None:
        chosen = Ut.numpy()
    else:
        d = ((y[:, None, :] - A) ** 2).mean(-1); d[~mask] = np.inf; chosen = A[np.arange(len(A)), d.argmin(1)]
    moved = float(((chosen - U) ** 2).mean())  # how far the matched targets sit from MPPI's label

    def run(x, n):
        alive, lasted, sp = np.ones(len(x), bool), np.full(len(x), n), np.zeros(len(x))
        for t in range(n):
            with torch.no_grad():
                u = net(torch.as_tensor(((observation(x) - mean) / std).astype(np.float32))).numpy()
            x = step(env, x, np.clip(u.astype(np.float64), -1, 1))
            fell = alive & ~env._is_healthy(x[:, 2], x[:, 3]); lasted[fell], alive = t + 1, alive & ~fell
            if t >= 100: sp += x[:, 10]
        return alive, lasted, sp / max(n - 100, 1)

    take = np.array([W[f"x_{k}"][t] for k in ks for t in (range(250, 751, 50) if not QUICK else range(10, 51, 20)) if len(W[f"x_{k}"]) > t])
    a_t, l_t, _ = run(take, 500 if not QUICK else 20)
    a_s, l_s, sp = run(standing_starts(env).copy(), 1000 if not QUICK else 20)
    return dict(start=s, noise=LEVELS[li], arm=arm, fit_seed=seed, walk_lengths=[int(len(W[f"x_{k}"])) for k in ks],
                states=int(len(X)), set_size=None if mask is None else float(mask.sum(1).mean()),
                fit_error=fit_err, target_vs_label=moved, takeovers=int(len(take)), takeover_4s=float(a_t.mean()),
                takeover_median_s=float(np.median(l_t)) * 0.008, standing_up8s=int(a_s.sum()),
                standing_walks=int((a_s & (np.abs(sp - 1.5) <= 0.3)).sum()), standing_median_s=float(np.median(l_s)) * 0.008)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    collect(out)
    W = np.load(out / "walks.npz")
    for k in range(12):  # sanity: top-64 weighted average vs MPPI's label (should be close, not equal)
        A, C, lam, U = W[f"a_{k}"], W[f"c_{k}"], W[f"lam_{k}"], W[f"u_{k}"]
        w = np.exp(-C / lam[:, None]); w /= w.sum(1, keepdims=True)
        print(json.dumps(dict(walk=k, length=int(len(U)), top64_avg_vs_label=float(((np.einsum("tk,tka->ta", w, A) - U) ** 2).mean()),
                              within_lam=float((C <= lam[:, None]).sum(1).mean()))), flush=True)
    jobs = [(out, s, li, arm, seed) for s in range(2) for li in range(len(LEVELS)) for arm in ARMS for seed in SEEDS]
    rows = []
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:  # spawn: the parent has initialised CUDA (Warp)
        for r in pool.imap_unordered(fit_and_test, jobs):
            rows.append(r); print(json.dumps(r), flush=True)
            (out / "result.json").write_text(json.dumps(sorted(rows, key=lambda r: (r["start"], r["noise"], r["arm"], r["fit_seed"])), indent=1) + "\n")
    for s in range(2):
        for lv in LEVELS:
            for arm in ARMS:
                v = [r for r in rows if r["start"] == s and r["noise"] == lv and r["arm"] == arm]
                print("start %d noise %.1f %-7s takeover 4 s %s  median %s  standing up %s" % (
                    s, lv, arm, [round(r["takeover_4s"], 2) for r in v], [round(r["takeover_median_s"], 2) for r in v],
                    [r["standing_up8s"] for r in v]), flush=True)


if __name__ == "__main__":
    main()
