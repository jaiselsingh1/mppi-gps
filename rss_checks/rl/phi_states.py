"""Off-path start states for checking the MPPI-value potential (rss_checks; .venv-rl; PREREG_motion.md ~23:57).
From the 32 standing starts on CPU MuJoCo (as our_walker.judge): 24 states of the standing policy
(task_seed2's latest) at 2 s; 24 of a falling policy (task_seed0's latest) 0.2 s before it falls; 24 of
zero actions 0.2 s before the fall. Full-physics states [t, qpos, qvel], ready for MPPI restarts.
Writes <out>/off_states.npz (x, group)."""
import pickle, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import numpy as np


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    import jax, jax.numpy as jnp, mujoco
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from our_walker import STARTS, XML, healthy_of, obs_of

    net = ppo_networks.make_ppo_networks(17, 6, preprocess_observations_fn=running_statistics.normalize)
    make_inf = ppo_networks.make_inference_fn(net)
    M = ROOT / "runs/rc/motion2"

    def policy(name):
        S = pickle.load(open(M / name / "state.pkl", "rb"))
        f = jax.jit(make_inf(S["params"], deterministic=True))
        print(name, "chunk", S["chunk"], flush=True)
        return lambda o: np.asarray(f(jnp.asarray(o), jax.random.PRNGKey(0))[0])

    starts = np.load(STARTS)["states"][:32]
    model = mujoco.MjModel.from_xml_path(XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    def run(act, T=1000):
        datas = [mujoco.MjData(model) for _ in starts]
        for d, x in zip(datas, starts):
            mujoco.mj_setState(model, d, x, mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(model, d)
        hist = np.zeros((T + 1, len(starts), nstate)); alive = np.ones(len(starts), bool); fell = np.full(len(starts), -1)
        for i, d in enumerate(datas):
            mujoco.mj_getState(model, d, hist[0, i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        for t in range(T):
            u = np.clip(act(np.stack([obs_of(d) for d in datas])), -1, 1)
            for i, d in enumerate(datas):
                if not alive[i]:
                    continue
                d.ctrl[:] = u[i]
                for _ in range(4):
                    mujoco.mj_step(model, d)
                mujoco.mj_getState(model, d, hist[t + 1, i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
                if not healthy_of(d.qpos):
                    alive[i] = False; fell[i] = t + 1
            if not alive.any():
                break
        return hist, fell

    X, G = [], []
    h, fell = run(policy("task_seed2"), T=250)
    ok = [i for i in range(32) if fell[i] < 0][:24]
    X += [h[250, i] for i in ok]; G += ["standing"] * len(ok)
    print("standing policy: up at 2 s from", len(ok), "of 32", flush=True)
    for name, act in (("policy_prefall", policy("task_seed0")), ("zero_prefall", lambda o: np.zeros((len(o), 6)))):
        h, fell = run(act)
        idx = [i for i in range(32) if fell[i] >= 26][:24]
        X += [h[fell[i] - 25, i] for i in idx]; G += [name] * len(idx)
        print(name, ": falls in", int((fell > 0).sum()), "of 32, median fall step", int(np.median(fell[fell > 0])) if (fell > 0).any() else -1, "; kept", len(idx), flush=True)
    np.savez(out / "off_states.npz", x=np.array(X), group=np.array(G))
    print("saved", len(X), "states", flush=True)


if __name__ == "__main__":
    main()
