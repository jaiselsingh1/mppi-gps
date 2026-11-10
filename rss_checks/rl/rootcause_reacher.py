"""PREREG_rootcause.md phase 4: does the mechanism hold on the reacher toy (dm_control reacher-easy, CPU)?
Teacher: MPPI-MPC as in toy_loop.rstar (K 1,024, H 50, sigma 0.3, 9.375 Hz). Data from 8 training starts (bank 100-107),
1,000 controls each: "thin" = the teacher as it runs; "spread" = executed with N(0, 0.1^2) noise, clean labels (same rows).
Copies: 2 x 256 tanh, tanh output, z-scored, Adam 1e-3, batch 512, 10,000 updates, 3 seeds. Measured: the copy's return
from the same 8 starts and from the 32 held-out starts, vs the teacher's; the copy's distance to the teacher's states.
  rootcause_reacher.py -> runs/root/reacher.json"""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT / "rss_checks" / "toy"))
import numpy as np
from toy_loop import closed, noise
from toy_task import T, Toy

TRAIN, HELD = np.arange(100, 108), np.arange(32)


def teacher(toy, x, tg, dart, seed, Kr=1024, H=50):
    N = len(x); rng = np.random.default_rng(seed); U = np.zeros((N, H, 2)); R = np.zeros(N); O, A = [], []
    for t in range(T):
        E = noise(rng, N * Kr, H, 0.3, burn=50).reshape(N, Kr, H, 2); E[:, 0] = 0.0
        C = np.clip(U[:, None] + E, -1.0, 1.0)
        S = toy.open_loop(np.repeat(x, Kr, 0), C.reshape(N * Kr, H, 2)).reshape(N, Kr, H, 5)
        G = toy.reward(S, C, tg[:, None, None, :]).sum(-1)
        lo, hi = G.min(1, keepdims=True), G.max(1, keepdims=True)
        w = np.exp(-10.0 * (hi - G) / np.maximum(hi - lo, 1e-12)); w /= w.sum(1, keepdims=True)
        U = U + (w[:, :, None, None] * E).sum(1)
        u = np.clip(U[:, 0], -1.0, 1.0); O.append(toy.obs(x, tg)); A.append(u.copy())
        ue = np.clip(u + dart * rng.standard_normal(u.shape), -1.0, 1.0) if dart else u
        x = toy.step(x, ue); R += toy.reward(x, ue, tg)
        U = np.concatenate([U[:, 1:], U[:, -1:]], 1)
    return R, np.stack(O, 1), np.stack(A, 1)


def fit(O, A, seed):
    import torch
    torch.manual_seed(seed); m, s = O.mean(0), O.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(O.shape[1], 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(),
                              torch.nn.Linear(256, A.shape[1]), torch.nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), 1e-3); x = torch.tensor((O - m) / s, dtype=torch.float32); y = torch.tensor(A, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(10_000):
        b = torch.randint(0, len(x), (512,), generator=g); l = ((net(x[b]) - y[b]) ** 2).mean(); opt.zero_grad(); l.backward(); opt.step()
    L = [(l.weight.detach().numpy().astype(np.float64), l.bias.detach().numpy().astype(np.float64)) for l in net if isinstance(l, torch.nn.Linear)]

    def pi(o):
        h = (o - m) / s
        for W, b in L:
            h = np.tanh(h @ W.T + b)
        return h
    return pi, float(np.mean((pi(O) - A) ** 2))


def main():
    from scipy.spatial import cKDTree
    toy = Toy("reacher", nthread=8); t0 = time.perf_counter(); res = {}
    x, tg = toy.starts(TRAIN); xh, tgh = toy.starts(HELD)
    data = {name: teacher(toy, x, tg, dart, 31 + i) for i, (name, dart) in enumerate((("thin", 0.0), ("spread", 0.1)))}
    Rt_train = data["thin"][0]
    res["teacher"] = dict(train_return=float(Rt_train.mean()), spread_run_return=float(data["spread"][0].mean()), s=round(time.perf_counter() - t0))
    print(json.dumps(res["teacher"]), flush=True)
    sd = np.concatenate([data["thin"][1].reshape(-1, 6), data["spread"][1].reshape(-1, 6)]).std(0) + 1e-6
    tree = cKDTree(data["thin"][1].reshape(-1, 6) / sd)
    for name, (_, O, A) in data.items():
        res[name] = []
        for seed in (0, 1, 2):
            pi, mse = fit(O.reshape(-1, 6), A.reshape(-1, 2), seed)
            Rtr, Otr = closed(toy, pi, x, tg, keep=True); Rh = closed(toy, pi, xh, tgh)
            dist = tree.query(Otr.reshape(-1, 6) / sd)[0].reshape(len(x), T) / np.sqrt(6)
            res[name].append(dict(seed=seed, fit_mse=mse, train_return=float(Rtr.mean()), share_of_teacher_train=float(Rtr.mean() / Rt_train.mean()),
                                  heldout_return=float(Rh.mean()), dist_to_teacher_states={str(c): float(np.median(dist[:, c])) for c in (50, 200, 500, 999)}))
            print(json.dumps(dict(data=name, **res[name][-1])), flush=True)
    res["s"] = round(time.perf_counter() - t0)
    json.dump(res, open(ROOT / "runs/root/reacher.json", "w"), indent=1)


if __name__ == "__main__":
    main()
