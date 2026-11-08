"""Does the motion term flip which behaviour PPO's objective prefers? (rss_checks; .venv-rl; PREREG_motion.md)
For each chosen policy (deterministic, from the 32 standing starts, CPU MuJoCo, up to 1,000 controls), compute
the discounted (gamma 0.995, PPO's) value under (a) the task reward alone and (b) the motion reward r_m alone,
with each motion run's own discriminator. A run that is still up at 1,000 gets a tail of
mean(last 200 rewards)/(1 - gamma), as PPO bootstraps at truncation. Then the combined objective
V_task + w V_motion ranks the behaviours; w* is the weight at which the standing policy and a lurching policy
tie. Trains nothing."""
import json, pickle, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np

GAMMA = 0.995
POLICIES = ["task_seed2", "task_seed0", "task_seed1", "task_motion_seed0", "task_motion_seed1", "task_motion_seed2",
            "motion_seed0", "motion_seed1", "motion_seed2"]
DS = [f"{a}_seed{s}" for a in ("task_motion", "motion") for s in range(3)]


def value(r, survived):
    tail = r[-200:].mean() / (1 - GAMMA) if survived else 0.0
    return float((GAMMA ** np.arange(len(r)) * r).sum() + GAMMA ** len(r) * tail)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    import jax, jax.numpy as jnp, mujoco
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from motion_rl import d_apply
    from our_walker import STARTS, XML, cost_terms, healthy_of, obs_of

    M = ROOT / "runs/rc/motion2"
    S = {n: pickle.load(open(M / n / "state.pkl", "rb")) for n in POLICIES}
    ref = np.load(M / "ref.npz")
    mu = np.concatenate([ref["o"].mean(0)] * 2); sd = np.concatenate([ref["o"].std(0) + 1e-3] * 2)
    dj = jax.jit(d_apply)
    net = ppo_networks.make_ppo_networks(17, 6, preprocess_observations_fn=running_statistics.normalize)
    make_inf = ppo_networks.make_inference_fn(net)
    starts = np.load(STARTS)["states"][:32]
    model = mujoco.MjModel.from_xml_path(XML)
    rows = {}
    for n in POLICIES:
        f = jax.jit(make_inf(S[n]["params"], deterministic=True))
        datas = [mujoco.MjData(model) for _ in starts]
        for d, x in zip(datas, starts):
            mujoco.mj_setState(model, d, x, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(model, d)
        alive = np.ones(32, bool); RT = [[] for _ in range(32)]; TR = [[] for _ in range(32)]
        for t in range(1000):
            o = np.stack([obs_of(d) for d in datas])
            u = np.clip(np.asarray(f(jnp.asarray(o), jax.random.PRNGKey(0))[0], dtype=np.float64), -1, 1)
            for i, d in enumerate(datas):
                if not alive[i]:
                    continue
                d.ctrl[:] = u[i]
                for _ in range(4):
                    mujoco.mj_step(model, d)
                h = healthy_of(d.qpos)
                RT[i].append(5.0 - cost_terms(d.qpos, d.qvel, u[i], h) - (0.0 if h else 20.0))
                TR[i].append(np.concatenate([o[i], obs_of(d)]))
                if not h:
                    alive[i] = False
            if not alive.any():
                break
        Vt, Vm, up = [], {k: [] for k in DS}, []
        for i in range(32):
            rt = np.array(RT[i]); X = jnp.asarray(((np.array(TR[i]) - mu) / sd).astype(np.float32))
            Vt.append(value(rt, alive[i])); up.append(len(rt) * 0.008)
            for k in DS:
                rm = np.maximum(0.0, 1.0 - 0.25 * (np.asarray(dj(S[k]["dparams"], X)) - 1.0) ** 2)
                Vm[k].append(value(rm, alive[i]))
        rows[n] = dict(chunk=S[n]["chunk"], time_up_s=float(np.median(up)), survived=int(alive.sum()),
                       V_task=float(np.mean(Vt)), V_motion={k: float(np.mean(v)) for k, v in Vm.items()},
                       V_motion_mean=float(np.mean([np.mean(v) for v in Vm.values()])))
        print(f"{n:18s} c{S[n]['chunk']}  up {np.median(up):5.2f}s ({int(alive.sum()):2d}/32)  V_task {np.mean(Vt):7.1f}  "
              f"V_motion {rows[n]['V_motion_mean']:7.1f}  -> task+5*motion {np.mean(Vt) + 5 * rows[n]['V_motion_mean']:7.1f}", flush=True)
    stand = rows["task_seed2"]
    flips = {}
    for n, r in rows.items():
        if n == "task_seed2":
            continue
        dt = stand["V_task"] - r["V_task"]
        per = {k: (dt / (r["V_motion"][k] - stand["V_motion"][k])) if r["V_motion"][k] > stand["V_motion"][k] else None for k in DS}
        flips[n] = dict(task_prefers_standing=dt > 0, w_star_per_D=per)
    (out / "reward_rank.json").write_text(json.dumps(dict(rows=rows, flips=flips), indent=1) + "\n")
    print("w* (weight at which this policy ties the standing policy; above it the combined objective prefers it):")
    for n, fl in flips.items():
        ws = [v for v in fl["w_star_per_D"].values() if v is not None]
        print(f"  {n:18s} task alone prefers standing: {fl['task_prefers_standing']}  w* over the 6 Ds: "
              + (f"min {min(ws):.2f} median {np.median(ws):.2f} max {max(ws):.2f}" if ws else "n/a"), flush=True)


if __name__ == "__main__":
    main()
