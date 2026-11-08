"""Brax PPO policy -> numpy weights for loop_mppi.py's pull (rss_checks/rl; PREREG_pull.md), and fresh standing starts.
Policy: obs normalised by the shared running statistics, 4 swish hidden layers, linear output (loc, scale);
the deterministic action is tanh(loc)."""
import numpy as np


def export_policy(params, O, make_inf, jax, jnp):
    """Returns (P, max abs error vs Brax's deterministic action on obs O at full-precision matmul)."""
    nrm = params[0]; pp = params[1]["params"]
    P = dict(obs_mean=np.asarray(nrm.mean, np.float64), obs_std=np.asarray(nrm.std, np.float64))
    for i in range(5):
        P[f"PW{i}"] = np.asarray(pp[f"hidden_{i}"]["kernel"], np.float64); P[f"Pb{i}"] = np.asarray(pp[f"hidden_{i}"]["bias"], np.float64)
    assert f"hidden_5" not in pp, "unexpected policy depth"
    h = (O - P["obs_mean"]) / P["obs_std"]
    for i in range(4):
        z = h @ P[f"PW{i}"] + P[f"Pb{i}"]; h = z / (1 + np.exp(-z))
    an = np.tanh((h @ P["PW4"] + P["Pb4"])[:, :6])
    with jax.default_matmul_precision("highest"):
        pol = jax.jit(make_inf(params, deterministic=True))
        aj = np.asarray(pol(jnp.asarray(O.astype(np.float32)), jax.random.PRNGKey(0))[0])
    return P, float(np.max(np.abs(an - aj)))


def standing_starts(seeds):
    """Full-physics states from OurWalker's reset (the task's start distribution), one per seed."""
    import mujoco
    from our_walker import OurWalker
    env = OurWalker(); n = mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    X = []
    for s in seeds:
        env.reset(seed=int(s)); x = np.zeros(n); mujoco.mj_getState(env.model, env.data, x, mujoco.mjtState.mjSTATE_FULLPHYSICS); X.append(x)
    return np.array(X)
