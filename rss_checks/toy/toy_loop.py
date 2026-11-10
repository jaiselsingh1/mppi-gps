"""PREREG_gate.md: the plain loop (arm A) on the stock toys, CPU float64.
  toy_loop.py parity TOY | norm TOY | rstar TOY | loop TOY SEED ROUNDS [stop_if_invalid]"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import butter, lfilter
from toy_task import Toy, T

OUT = Path(os.environ.get("TOYOUT", Path(__file__).resolve().parent / "out")); OUT.mkdir(parents=True, exist_ok=True)
HELD = np.arange(32); K, PASSES, SIGMAS, CUT, FS_HZ = 64, 2, (0.3, 0.1), 9.375, 50.0
EPS, N_COPIES, LR, BATCH, FIT_STEPS = 0.1, 4, 1e-3, 1024, 500
_b, _a = butter(2, CUT, btype="lowpass", fs=FS_HZ); _VAR = float(np.sum(lfilter(_b, _a, np.r_[1.0, np.zeros(20000)]) ** 2))


def noise(rng, n, length, sigma, burn=200):
    f = lfilter(_b, _a, rng.standard_normal((n, length + burn, 2)), axis=1)[:, burn:]
    return f / np.sqrt(_VAR) * sigma


def make_pi(W, mean, sd):
    def pi(o):
        h = (o - mean) / sd
        for i, (w, b) in enumerate(W):
            z = h @ w + b; h = z / (1.0 + np.exp(-z)) if i < len(W) - 1 else np.tanh(z)
        return h
    return pi


def closed(toy, pi, x, tg, D=None, keep=False):
    """u_t = clip(pi(obs) + D[:, t]); returns the stock return per world (and the observations before each control)."""
    W = len(x); R = np.zeros(W); O = np.zeros((W, T, toy.nobs)) if keep else None
    for t in range(T):
        o = toy.obs(x, tg)
        if keep:
            O[:, t] = o
        u = pi(o) if D is None else pi(o) + D[:, t]
        u = np.clip(u, -1.0, 1.0); x = toy.step(x, u); R += toy.reward(x, u, tg)
    return (R, O) if keep else R


def norm_stats(toy):
    f = OUT / f"norm_{toy.name}.npz"
    if f.exists():
        z = np.load(f); return z["mean"], z["sd"]
    x, tg = toy.starts(HELD); rng = np.random.default_rng(123); E = noise(rng, len(x), T, 0.3)
    _, O = closed(toy, lambda o: np.zeros((len(o), 2)), x, tg, E, keep=True)
    mean, sd = O.reshape(-1, toy.nobs).mean(0), O.reshape(-1, toy.nobs).std(0); np.savez(f, mean=mean, sd=sd)
    return mean, sd


def rstar(toy, Kr=1024, H=50):
    x, tg = toy.starts(HELD); N = len(x); rng = np.random.default_rng(7); U = np.zeros((N, H, 2)); R = np.zeros(N); t0 = time.perf_counter()
    for t in range(T):
        E = noise(rng, N * Kr, H, 0.3, burn=50).reshape(N, Kr, H, 2); E[:, 0] = 0.0
        C = np.clip(U[:, None] + E, -1.0, 1.0)
        S = toy.open_loop(np.repeat(x, Kr, 0), C.reshape(N * Kr, H, 2)).reshape(N, Kr, H, 5)
        G = toy.reward(S, C, tg[:, None, None, :]).sum(-1)                       # (N, Kr) summed reward over the horizon
        lo, hi = G.min(1, keepdims=True), G.max(1, keepdims=True)
        w = np.exp(-10.0 * (hi - G) / np.maximum(hi - lo, 1e-12)); w /= w.sum(1, keepdims=True)
        U = U + (w[:, :, None, None] * E).sum(1)
        u = np.clip(U[:, 0], -1.0, 1.0); x = toy.step(x, u); R += toy.reward(x, u, tg)
        U = np.concatenate([U[:, 1:], U[:, -1:]], 1)
        if t % 100 == 99:
            print(f"  rstar {toy.name} control {t + 1}: mean return so far {R.mean():.1f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    out = dict(toy=toy.name, K=Kr, H=H, rstar=float(R.mean()), se=float(R.std(ddof=1) / np.sqrt(N)), per_start=R.tolist(), seconds=round(time.perf_counter() - t0))
    json.dump(out, open(OUT / f"rstar_{toy.name}.json", "w")); print(json.dumps({k: v for k, v in out.items() if k != "per_start"}), flush=True)


def loop(toy, seed, rounds, stop_if_invalid):
    import torch
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2); torch.manual_seed(seed)
    mean, sd = norm_stats(toy)
    net = torch.nn.Sequential(torch.nn.Linear(toy.nobs, 32), torch.nn.SiLU(), torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Linear(32, 32), torch.nn.SiLU(),
                              torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Linear(32, 2), torch.nn.Tanh())
    export = lambda: [(l.weight.detach().numpy().T.copy(), l.bias.detach().numpy().copy()) for l in net if isinstance(l, torch.nn.Linear)]
    xh, tgh = toy.starts(HELD); pi = make_pi(export(), mean, sd); logf = OUT / f"loop_{toy.name}_seed{seed}.jsonl"
    J0 = float(closed(toy, pi, xh, tgh).mean()); open(logf, "w").write(json.dumps(dict(round=0, held=J0)) + "\n"); print(f"{toy.name} seed {seed} round 0 held-out {J0:.1f}", flush=True)
    rng = np.random.default_rng(10_000 + seed); mt, st = torch.tensor(mean), torch.tensor(sd)
    for r in range(1, rounds + 1):
        t0 = time.perf_counter(); idx = 100 + 700 * (seed - 1) + 32 * r + np.arange(32); x0, tg = toy.starts(idx); N = len(x0)
        D = np.zeros((N, T, 2)); G0 = None
        for p in range(PASSES):
            E = np.concatenate([noise(rng, N * (K // 2), T, s).reshape(N, K // 2, T, 2) for s in SIGMAS], 1); E[:, 0] = 0.0
            G = closed(toy, pi, np.repeat(x0, K, 0), np.repeat(tg, K, 0), (D[:, None] + E).reshape(N * K, T, 2)).reshape(N, K)
            if G0 is None:
                G0 = G[:, 0].copy()
            D = D + E[np.arange(N), G.argmax(1)]
        G1, O = closed(toy, pi, x0, tg, D, keep=True); Ghalf = closed(toy, pi, x0, tg, 0.5 * D)
        # fit: targets clip(pi_k(x) + delta_t) on the path and at 4 copies moved by 0.1 sd
        o = O.reshape(-1, toy.nobs); d = D.reshape(-1, 2)
        oo = np.concatenate([o] + [o + rng.standard_normal(o.shape) * EPS * sd for _ in range(N_COPIES)]); dd = np.concatenate([d] * (N_COPIES + 1))
        tgt = np.clip(pi(oo) + dd, -1.0, 1.0); Xt, Yt = torch.tensor(oo), torch.tensor(tgt)
        opt = torch.optim.Adam(net.parameters(), lr=LR); g = torch.Generator().manual_seed(1000 * seed + r)
        f = lambda X: net((X - mt) / st)
        with torch.no_grad():
            l0 = float(((f(Xt) - Yt) ** 2).sum(-1).mean())
        for _ in range(FIT_STEPS):
            b = torch.randint(0, len(Xt), (BATCH,), generator=g); loss = ((f(Xt[b]) - Yt[b]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            l1 = float(((f(Xt) - Yt) ** 2).sum(-1).mean())
        assert np.isfinite(l1), "NaN loss"
        pi = make_pi(export(), mean, sd)
        Gf = closed(toy, pi, x0, tg); Jh = float(closed(toy, pi, xh, tgh).mean())
        imp = float((G1 - G0).mean()); row = dict(round=r, policy=float(G0.mean()), traj=float(G1.mean()), fitted=float(Gf.mean()), half=float(Ghalf.mean()),
                                               improvement=imp, signal_share=float(np.mean(G1 - G0 > 1.0)), kept_share=float((Gf - G0).mean() / imp) if abs(imp) > 1e-9 else None,
                                               half_share=float((Ghalf - G0).mean() / imp) if abs(imp) > 1e-9 else None, corr_rms=float(np.sqrt((D ** 2).mean())),
                                               loss_before=l0, loss_after=l1, held=Jh, seconds=round(time.perf_counter() - t0))
        open(logf, "a").write(json.dumps(row) + "\n")
        print(f"{toy.name} seed {seed} round {r}: policy {row['policy']:.1f} traj {row['traj']:.1f} fitted {row['fitted']:.1f} half {row['half']:.1f} signal {row['signal_share']:.2f} "
              f"held-out {Jh:.1f} ({row['seconds']} s)", flush=True)
        if r == 1 and stop_if_invalid and row["signal_share"] < 0.5:
            print(f"{toy.name} seed {seed}: V2 not met at round 1 (signal share {row['signal_share']:.2f}); stopped after round 1", flush=True); break


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]; toy = Toy(name, nthread=int(os.environ.get("NTHREAD", "4")))
    if mode == "parity":
        w = toy.parity(); print(name, json.dumps(w), "PASSED" if max(w.values()) <= 1e-8 else "FAILED")
    elif mode == "norm":
        print(name, norm_stats(toy))
    elif mode == "rstar":
        rstar(toy)
    else:
        loop(toy, int(sys.argv[3]), int(sys.argv[4]), len(sys.argv) > 5)
