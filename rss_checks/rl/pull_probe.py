"""Pull-strength probe (rss_checks/rl; .venv-rl; PREREG_pull.md "Probe first"). MPPI pulled toward a standing-still
policy (runs/rc/motion2/task_motion_seed0: PPO on task + motion reward, 32/32 up, 0 walk; amended ~07:24,
self_seed2 final turned out to walk) from 8 fresh standing starts, 1,000 controls, w in {0, 0.3, 1, 3, 10}, and toward
a walking policy (runs/loop/pool_seed0 final) at w in {1, 10}. Per config: walks up all 1,000 controls, mean vx after
control 100 (up walks), mean per-step ||u - pi||^2 on the executed path, seconds. Also each policy's own mean vx from
the same starts. Usage: pull_probe.py OUT [w ...] (extra w values toward the standing policy, for the grid extension)."""
import json, pickle, subprocess, sys, os
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    own_only = "--own-only" in sys.argv
    extra = [float(v) for v in sys.argv[2:] if v != "--own-only"]
    stand = os.environ.get("STAND", "runs/rc/motion2/task_motion_seed0")
    import jax, jax.numpy as jnp, mujoco
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from our_walker import XML, healthy_of, obs_of
    from policy_export import export_policy, standing_starts

    net = ppo_networks.make_ppo_networks(17, 6, preprocess_observations_fn=running_statistics.normalize)
    make_inf = ppo_networks.make_inference_fn(net)
    pool = np.load(ROOT / "runs/loop/pool_mppi.npz")["x"][:4000]
    O = np.concatenate([pool[:, 2:10], np.clip(pool[:, 10:19], -10, 10)], 1)
    X0 = standing_starts(79_000_000 + np.arange(8))
    model = mujoco.MjModel.from_xml_path(XML)
    pols, checks, own = {}, {}, {}
    for name, run in (("stand", stand), ("walk", "runs/loop/pool_seed0")):
        S = pickle.load(open(ROOT / run / "state.pkl", "rb"))
        P, err = export_policy(S["params"], O, make_inf, jax, jnp); pols[name] = P; checks[name] = dict(run=run, round=S.get("round", S.get("chunk")), export_max_abs_err=err)
        pol = jax.jit(make_inf(S["params"], deterministic=True)); key = jax.random.PRNGKey(0)
        ds = [mujoco.MjData(model) for _ in X0]
        for d, x in zip(ds, X0):
            mujoco.mj_setState(model, d, x, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(model, d)
        alive = np.ones(len(ds), bool); vx = [[] for _ in ds]
        for t in range(1000):
            u = np.clip(np.asarray(pol(jnp.asarray(np.stack([obs_of(d) for d in ds]).astype(np.float32)), key)[0], np.float64), -1, 1)
            for i, d in enumerate(ds):
                if alive[i]:
                    d.ctrl[:] = u[i]
                    for _ in range(4):
                        mujoco.mj_step(model, d)
                    if t >= 100:
                        vx[i].append(d.qvel[0])
                    alive[i] = healthy_of(d.qpos)
        own[name] = dict(up=int(alive.sum()), vx=float(np.mean([np.mean(v) for v, a in zip(vx, alive) if a])) if alive.any() else None)
    print(json.dumps(dict(checks=checks, own=own)), flush=True)
    assert all(c["export_max_abs_err"] <= 1e-4 for c in checks.values()), "policy export check failed"
    if own_only:
        return
    assert own["stand"]["up"] == 8 and own["stand"]["vx"] < 1.0, "the non-walking policy falls or walks"
    cfgs = [("stand", w) for w in [0.0, 0.3, 1.0, 3.0, 10.0] + extra] + [("walk", 1.0), ("walk", 10.0)]
    if extra:
        cfgs = [("stand", w) for w in extra]
    procs = []
    for name, w in cfgs:
        tag = f"{name}_w{w:g}"; inp = out / f"in_{tag}.npz"
        np.savez(inp, x=X0, length=1000, beta=0.0, gamma=0.995, seed0=41_000_000, w_pull=w, **pols[name])
        f = open(out / f"{tag}.log", "w")
        procs.append((name, w, tag, subprocess.Popen([str(Path.home() / "mppi-gps/.venv/bin/python"), str(ROOT / "rss_checks/loop_mppi.py"), str(inp), str(out / f"out_{tag}.npz")],
                                                    stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "2"})))
    rows = []
    for name, w, tag, p in procs:
        p.wait()
        if p.returncode != 0:
            rows.append(dict(cfg=tag, error=(out / f"{tag}.log").read_text()[-1500:])); continue
        M = np.load(out / f"out_{tag}.npz"); fell = M["fell"]
        vxs = [M[f"states_{k}"][100:, 10].mean() for k in range(len(fell)) if not fell[k]]
        info = json.loads((out / f"{tag}.log").read_text().strip().splitlines()[-1])
        rows.append(dict(cfg=tag, policy=name, w=w, up=int((~fell).sum()), vx=float(np.mean(vxs)) if vxs else None,
                         gap=float(M["gap"].mean()), s=info["s"]))
        print(json.dumps(rows[-1]), flush=True)
    (out / ("summary_extra.json" if extra else "summary.json")).write_text(json.dumps(dict(checks=checks, own=own, rows=rows), indent=1))


if __name__ == "__main__":
    main()
