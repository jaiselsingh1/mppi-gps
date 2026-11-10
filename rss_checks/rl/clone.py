"""PREREG_clone.md (rss_checks/rl; mppi-gps venv): can a policy copied from MPPI's walks walk? Position targets vs torques.
Students: plain regression from the 17-d observation to MPPI's executed action on the walks that walked
(runs/pos/walks_<arm>.npz), 2 x 256 tanh MLP, standardised inputs, MSE, Adam 1e-3, batch 512, 10,000 updates, 3 seeds.
Deployment: CPU MuJoCo, deterministic, 32 fresh standing starts (OurWalker's reset rule, seeds 60_000_000 + i), 1,000
controls, on the arm's own model (position: the chosen PD model). Walks = up 1,000 controls and mean |vx - 1.5| <= 0.3
after control 100.
  clone.py -> runs/pos/clone.json"""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np
from pos_check import OUT, load, obs17, walked, xml_of

SEEDS, N_START = (0, 1, 2), 32


def starts():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets/walker2d.xml")); q0 = m.qpos0.copy(); X = []
    for i in range(N_START):
        r = np.random.default_rng(60_000_000 + i)
        X.append(np.r_[0.0, q0 + r.uniform(-5e-3, 5e-3, m.nq), r.uniform(-5e-3, 5e-3, m.nv)])
    return np.array(X)


def train(arm, seed):
    import torch
    X0, Ss, As = load(arm)
    keep = [k for k in range(len(Ss)) if walked(Ss[k])]
    O = np.concatenate([obs17(np.concatenate([X0[k][None], Ss[k][:-1]])) for k in keep]); A = np.concatenate([As[k] for k in keep])
    m, s = O.mean(0), O.std(0) + 1e-6
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(17, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(), torch.nn.Linear(256, 6))
    opt = torch.optim.Adam(net.parameters(), 1e-3); xt = torch.tensor((O - m) / s, dtype=torch.float32); yt = torch.tensor(A, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(10_000):
        b = torch.randint(0, len(xt), (512,), generator=g); loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fit = float(((net(xt) - yt) ** 2).mean())
    W = [(l.weight.detach().numpy().T.astype(np.float64), l.bias.detach().numpy().astype(np.float64)) for l in net if isinstance(l, torch.nn.Linear)]
    return dict(arm=arm, seed=seed, walks_used=len(keep), rows=int(len(O)), train_mse=fit, m=m, s=s, W=W)


def run(job):
    arm, st, x0 = job
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml_of(arm)); d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d); vx = []
    for t in range(1000):
        h = (obs17(np.r_[0.0, d.qpos, d.qvel][None]) - st["m"]) / st["s"]
        for i, (W, b) in enumerate(st["W"]):
            h = h @ W + b; h = np.tanh(h) if i < len(st["W"]) - 1 else h
        d.ctrl[:] = np.clip(h[0], -1, 1)
        for _ in range(4):
            mujoco.mj_step(m, d)
        vx.append(d.qvel[0])
        if not (0.8 < d.qpos[1] < 2.0 and abs(d.qpos[2]) < 1.0):
            return t + 1, None
    return 1000, float(np.mean(np.abs(np.array(vx[100:]) - 1.5)))


def main():
    import multiprocessing as mp
    X = starts(); students = [train(arm, sd) for arm in ("pos", "torque") for sd in SEEDS]
    jobs = [(st["arm"], st, x) for st in students for x in X]
    with mp.get_context("spawn").Pool(30) as pool:
        got = pool.map(run, jobs, chunksize=4)
    out = []
    for i, st in enumerate(students):
        g = got[i * N_START:(i + 1) * N_START]
        lasted = [q[0] for q in g]; walks = sum(q[0] == 1000 and q[1] <= 0.3 for q in g)
        out.append(dict(arm=st["arm"], seed=st["seed"], walks_used=st["walks_used"], rows=st["rows"], train_mse=st["train_mse"],
                        walks=int(walks), survived=int(sum(l == 1000 for l in lasted)), lasted_median=float(np.median(lasted)),
                        speed_err=[q[1] for q in g if q[1] is not None]))
        print(json.dumps({k: v for k, v in out[-1].items() if k != "speed_err"}), flush=True)
    json.dump(out, open(OUT / "clone.json", "w"), indent=1)


if __name__ == "__main__":
    main()
