"""PREREG_pos.md (rss_checks/rl; mppi-gps venv): MPPI on Walker with torque actions vs PD position-target actions.
  pos_check.py gpucheck              Warp vs CPU MuJoCo on the position model (10-control rollouts)
  pos_check.py walks ARM N           N MPPI walks (the loop's config, unchanged) from the judge's standing starts,
                                     1,000 controls, seeds 0..N-1 -> runs/pos/walks_<ARM>.npz
  pos_check.py analyze               Q1 walking, Q2 action predictability, Q3 fragility -> runs/pos/readings.json
ARM: torque (assets/walker2d.xml) | pos (assets/walker2d_pos_off_kp100.xml: PD servos about the standing pose, 1 rad per unit, kp 100, kv 10, +-100 N m)."""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rss_checks"))
sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np

OUT = ROOT / "runs/pos"
XMLS = dict(torque=str(ROOT / "assets/walker2d.xml"), pos=str(ROOT / "assets/walker2d_pos_off_kp100.xml"))
GAMMA, H = 0.995, 125


def xml_of(arm):
    if arm == "pos" and chosen():
        return str(ROOT / "assets" / chosen()["xml"])
    return XMLS[arm]


def make_env(arm):
    from src.envs.walker2d import Walker2d
    from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
    return Walker2d(**NATIVE_TASK, model_path=xml_of(arm), nthread=1)


def gpucheck():
    import mujoco, gpu_walks as gw
    for arm in ("torque", "pos"):
        m = mujoco.MjModel.from_xml_path(XMLS[arm]); X = np.load(ROOT / "rss_checks/rl/standing_starts.npz")["states"][:4]
        U = np.random.default_rng(0).uniform(-1, 1, (64, 10, 6)); x0 = np.repeat(X, 16, 0)
        g = gw.WarpRollouts(m, 64, 10, 4)(x0[:, 1:10], x0[:, 10:19], np.zeros((64, 9)), U)
        c = np.zeros_like(g)
        for i in range(64):
            d = mujoco.MjData(m); mujoco.mj_setState(m, d, x0[i], mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d)
            for t in range(10):
                d.ctrl[:] = U[i, t]
                for _ in range(4):
                    mujoco.mj_step(m, d)
                c[i, t] = np.r_[0.0, d.qpos, d.qvel]
        print(arm, "max |warp - cpu| over 10 controls: qpos %.2e qvel %.2e" % (np.abs(g[..., 1:10] - c[..., 1:10]).max(), np.abs(g[..., 10:] - c[..., 10:]).max()), flush=True)


def chosen():
    f = OUT / "chosen.json"
    return json.load(open(f)) if f.exists() else None


def calib(name, xml, sigma, n):
    """Calibration (PREREG_pos.md amendment): n walks with a given position model and noise sd; nothing else changes."""
    import gpu_walks as gw
    from scripts.diagnostics.coupled_mppi import CONFIG
    from src.envs.walker2d import Walker2d
    from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
    cfg = json.loads(CONFIG.read_text()); cfg["noise_sigma"] = float(sigma)
    X = np.load(ROOT / "rss_checks/rl/standing_starts.npz")["states"][:n]
    jobs = [dict(k=k, x0=X[k], length=1000, dart=0.0, seed=k) for k in range(n)]
    envs = [Walker2d(**NATIVE_TASK, model_path=str(ROOT / "assets" / xml), nthread=1) for _ in jobs]; t0 = time.perf_counter()
    got = gw.lockstep_walks(envs, cfg, jobs, backend="warp")
    Ss = [g["states"] for g in got]
    res = dict(name=name, xml=xml, sigma=float(sigma), walks=int(sum(walked(S) for S in Ss)), survived=int(sum(len(S) == 1000 for S in Ss)),
               lengths=[len(S) for S in Ss], vx=[float(S[100:, 10].mean()) if len(S) > 100 else None for S in Ss], s=round(time.perf_counter() - t0))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "calib.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)


def walks(arm, n):
    import gpu_walks as gw
    from scripts.diagnostics.coupled_mppi import CONFIG
    cfg = json.loads(CONFIG.read_text()); X = np.load(ROOT / "rss_checks/rl/standing_starts.npz")["states"][:n]
    if arm == "pos" and chosen():
        cfg["noise_sigma"] = chosen()["sigma"]
    jobs = [dict(k=k, x0=X[k], length=1000, dart=0.0, seed=k) for k in range(n)]
    envs = [make_env(arm) for _ in jobs]; t0 = time.perf_counter()
    got = gw.lockstep_walks(envs, cfg, jobs, backend="warp")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"walks_{arm}.npz", s=round(time.perf_counter() - t0), X0=X,
             **{f"states_{g['k']}": g["states"] for g in got}, **{f"actions_{g['k']}": g["executed"] for g in got})
    print(json.dumps(dict(arm=arm, walks=n, s=round(time.perf_counter() - t0))), flush=True)


def load(arm):
    Z = np.load(OUT / f"walks_{arm}.npz"); n = len(Z["X0"])
    return Z["X0"], [Z[f"states_{k}"] for k in range(n)], [Z[f"actions_{k}"] for k in range(n)]


def obs17(X):
    return np.concatenate([X[..., 2:10], np.clip(X[..., 10:19], -10, 10)], -1)


def walked(S):
    if len(S) < 1000 or not ((0.8 < S[:, 2]) & (S[:, 2] < 2) & (np.abs(S[:, 3]) < 1)).all():
        return False
    return float(np.abs(S[100:, 10] - 1.5).mean()) <= 0.3


def r2_by_walk(X0, Ss, As, folds=4, seed=0):
    """Held-out R^2 of a 2 x 256 tanh MLP (standardised obs -> executed action), folds by whole walk."""
    import torch
    torch.manual_seed(seed); n = len(Ss); idx = np.arange(n); out = []
    O = [obs17(np.concatenate([x0[None], S[:-1]])) for x0, S in zip(X0, Ss)]
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        Otr, Atr = np.concatenate([O[i] for i in tr]), np.concatenate([As[i] for i in tr])
        Ote, Ate = np.concatenate([O[i] for i in te]), np.concatenate([As[i] for i in te])
        m, s = Otr.mean(0), Otr.std(0) + 1e-6
        net = torch.nn.Sequential(torch.nn.Linear(17, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(), torch.nn.Linear(256, 6))
        opt = torch.optim.Adam(net.parameters(), 1e-3); xt = torch.tensor((Otr - m) / s, dtype=torch.float32); yt = torch.tensor(Atr, dtype=torch.float32)
        for _ in range(4000):
            b = torch.randint(0, len(xt), (512,)); loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p = net(torch.tensor((Ote - m) / s, dtype=torch.float32)).numpy()
        out.append(float(1 - ((Ate - p) ** 2).sum() / ((Ate - Ate.mean(0)) ** 2).sum()))
    return out


def replay_window(args):
    """Open-loop replay of an action window from a state; returns (discounted window reward, end state, alive at the end)."""
    arm, x0, U = args
    import mujoco
    from e1_pi import step_reward
    m = mujoco.MjModel.from_xml_path(xml_of(arm)); d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d)
    G, alive = 0.0, np.ones(1, bool)
    for t in range(len(U)):
        d.ctrl[:] = U[t]
        for _ in range(4):
            mujoco.mj_step(m, d)
        x = np.r_[0.0, d.qpos, d.qvel]; r, alive = step_reward(x[None], U[t][None], alive); G += GAMMA ** t * r[0]
    return G, x, bool(alive[0])


def fragility(arm, X0, Ss, As, rng):
    """Windows of 1 s at controls 100, 300, 500, 700 of walks that stayed up; MPPI's executed actions replayed open loop,
    exactly and after small changes; end-state distance to the exact replay in units of the walks' state sd."""
    import multiprocessing as mp
    from scipy.signal import butter, filtfilt, lfilter
    b, a = butter(2, 23.4375, btype="lowpass", fs=125.0); var = float(np.sum(lfilter(b, a, np.r_[1.0, np.zeros(20000)]) ** 2))
    bb, aa = butter(2, 5.0, btype="lowpass", fs=125.0)
    sd = np.concatenate(Ss)[:, 1:].std(0) + 1e-6
    jobs, keys = [], []
    for k, (S, A) in enumerate(zip(Ss, As)):
        if len(S) < 1000:
            continue
        for t0 in (100, 300, 500, 700):
            x0 = S[t0 - 1]; U = A[t0:t0 + H]; dev = U - U.mean(0); rms = np.sqrt((dev ** 2).mean(0))
            conds = {"exact": U, "smooth5": np.clip(filtfilt(bb, aa, U, axis=0), -1, 1), "shrink0.5": U.mean(0) + 0.5 * dev}
            for kk in (0.1, 0.25):
                for j in range(4):
                    e = lfilter(b, a, rng.standard_normal((H + 200, 6)), axis=0)[200:] / np.sqrt(var)
                    conds[f"noise{kk}_{j}"] = np.clip(U + e * kk * rms, -1, 1)
            for c, Uc in conds.items():
                jobs.append((arm, x0, Uc)); keys.append((k, t0, c))
    with mp.get_context("spawn").Pool(30) as pool:
        got = pool.map(replay_window, jobs, chunksize=8)
    res = {}
    for (k, t0, c), g in zip(keys, got):
        res.setdefault((k, t0), {})[c] = g
    rows = {}
    for (k, t0), cs in res.items():
        Ge, xe, _ = cs["exact"]
        for c, (G, x, al) in cs.items():
            name = c.split("_")[0]
            rows.setdefault(name, []).append((float(np.linalg.norm((x[1:] - xe[1:]) / sd)), G - Ge, al))
    return {c: dict(windows=len(v), end_dist_median=float(np.median([q[0] for q in v])), window_return_change_mean=float(np.mean([q[1] for q in v])),
                    alive_1s=float(np.mean([q[2] for q in v]))) for c, v in rows.items()}


def analyze():
    out = {}
    for arm in ("torque", "pos"):
        X0, Ss, As = load(arm)
        r2 = r2_by_walk(X0, Ss, As)
        out[arm] = dict(walks=int(sum(walked(S) for S in Ss)), of=len(Ss), survived=int(sum(len(S) == 1000 for S in Ss)),
                        vx_mean=float(np.mean([S[100:, 10].mean() for S in Ss if len(S) == 1000])) if any(len(S) == 1000 for S in Ss) else None,
                        r2_folds=r2, r2_mean=float(np.mean(r2)), fragility=fragility(arm, X0, Ss, As, np.random.default_rng(5)))
        print(json.dumps({arm: out[arm]}), flush=True)
    json.dump(out, open(OUT / "readings.json", "w"), indent=1)


if __name__ == "__main__":
    {"gpucheck": lambda: gpucheck(), "walks": lambda: walks(sys.argv[2], int(sys.argv[3])), "analyze": lambda: analyze(),
     "calib": lambda: calib(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))}[sys.argv[1]]()
