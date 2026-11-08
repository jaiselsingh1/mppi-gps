"""Fit the MPPI-value potential Phi and run its offline checks (rss_checks; .venv-rl; PREREG_motion.md ~23:57).
Labels: the bench's no-tie MPPI walks (runs/rc/simple/none_seed1-5); each non-terminal state's value is
the discounted (gamma 0.995) sum of the rewards after it, with the fall penalty, plus a tail for walks
that end upright. Fit on seeds 1-4, held out seed 5 (C1). Off-path check against MPPI's measured returns
from <dir>/off_states.npz + off_returns.npz (C2, C3). Writes <dir>/phi.npz (weights, normalizer) and
<dir>/phi_checks.json."""
import json, sys
from pathlib import Path

ROOT = Path.home() / "mppi-gps-claude-sac"
import numpy as np

GAMMA = 0.995


def rewards(x):
    z, pitch, vx, vz = x[:, 2], x[:, 3], x[:, 10], x[:, 11]
    bad = ~((z > 0.8) & (z < 2.0) & (np.abs(pitch) < 1.0))
    c = 2 * np.abs(1.5 - vx) + 0.5 * vz**2 + 0.1 * pitch**2 + 5.0 * bad
    return 5.0 - c - 20.0 * bad, bad


def obs(x):
    return np.concatenate([x[:, 2:10], np.clip(x[:, 10:19], -10.0, 10.0)], 1).astype(np.float32)


def walk_labels(x, full_length=1000):
    """States after each control -> (obs of non-terminal states, their values, fell)."""
    r, bad = rewards(x)
    T = len(x); fell = bool(bad[-1])
    V = np.zeros(T)
    V[-1] = 0.0 if fell else r[-200:].mean() / (1 - GAMMA)
    for i in range(T - 2, -1, -1):
        V[i] = r[i + 1] + GAMMA * V[i + 1]
    keep = slice(0, T - 1) if fell else slice(0, T)
    return obs(x)[keep], V[keep], fell


def main():
    d = Path(sys.argv[1])
    import jax, jax.numpy as jnp, optax
    from scipy.stats import spearmanr

    data = {s: ([], [], 0, 0) for s in range(1, 6)}
    for s in range(1, 6):
        O, V, n, nf = [], [], 0, 0
        for r in range(16):
            f = ROOT / f"runs/rc/simple/none_seed{s}/runs_{r}.npz"
            if not f.exists():
                continue
            W = np.load(f)
            for k in W.files:
                o, v, fell = walk_labels(W[k]); O.append(o); V.append(v); n += 1; nf += fell
        data[s] = (np.concatenate(O), np.concatenate(V), n, nf)
    Otr = np.concatenate([data[s][0] for s in (1, 2, 3, 4)]); Vtr = np.concatenate([data[s][1] for s in (1, 2, 3, 4)])
    Ote, Vte = data[5][0], data[5][1]
    mu, sd = Otr.mean(0), Otr.std(0) + 1e-3
    scale = 1000.0

    def init(key, sizes=(17, 256, 256, 1)):
        ps = []
        for m, n in zip(sizes[:-1], sizes[1:]):
            key, k = jax.random.split(key)
            ps.append((jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m), jnp.zeros(n)))
        return ps

    def apply(ps, o):
        x = (o - mu) / sd
        for i, (Wt, b) in enumerate(ps):
            x = x @ Wt + b
            if i < len(ps) - 1:
                x = jax.nn.relu(x)
        return x[..., 0] * scale

    ps = init(jax.random.PRNGKey(0)); opt = optax.adam(3e-4); st = opt.init(ps)
    Xt, Yt = jnp.asarray(Otr), jnp.asarray(Vtr / scale)

    @jax.jit
    def step(ps, st, key):
        i = jax.random.randint(key, (1024,), 0, Xt.shape[0])
        loss, g = jax.value_and_grad(lambda p: jnp.mean((apply(p, Xt[i]) / scale - Yt[i]) ** 2))(ps)
        up, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, up), st, loss

    key = jax.random.PRNGKey(1)
    for t in range(20000):
        key, k = jax.random.split(key); ps, st, loss = step(ps, st, k)
    f = jax.jit(apply)
    pte = np.asarray(f(ps, jnp.asarray(Ote)))
    c1 = float(spearmanr(pte, Vte).statistic)
    r2 = float(1 - ((pte - Vte) ** 2).sum() / ((Vte - Vte.mean()) ** 2).sum())
    S, R = np.load(d / "off_states.npz"), np.load(d / "off_returns.npz")
    po = np.asarray(f(ps, jnp.asarray(obs(S["x"]))))
    G, grp = R["G"], R["group"]
    c2 = float(spearmanr(po, G).statistic)
    groups = {g: dict(n=int((grp == g).sum()), phi_mean=float(po[grp == g].mean()), mppi_return_mean=float(G[grp == g].mean()),
                      mppi_stays_up=int((R["fell_at"][grp == g] < 0).sum())) for g in ("standing", "policy_prefall", "zero_prefall")}
    order_phi = sorted(groups, key=lambda g: -groups[g]["phi_mean"]); order_true = sorted(groups, key=lambda g: -groups[g]["mppi_return_mean"])
    res = dict(train_states=int(len(Otr)), heldout_states=int(len(Ote)), walks={s: [data[s][2], data[s][3]] for s in data},
               final_train_loss=float(loss), C1_spearman_heldout=c1, heldout_r2=r2, C1_pass=c1 >= 0.8,
               C2_spearman_offpath=c2, C2_pass=c2 >= 0.6, C3_order_phi=order_phi, C3_order_true=order_true,
               C3_match=order_phi == order_true, groups=groups,
               offpath_points=[dict(group=str(g), phi=float(p), mppi_return=float(r)) for g, p, r in zip(grp, po, G)])
    (d / "phi_checks.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez(d / "phi.npz", mu=mu, sd=sd, scale=scale, **{f"W{i}": np.asarray(W_) for i, (W_, _) in enumerate(ps)},
             **{f"b{i}": np.asarray(b) for i, (_, b) in enumerate(ps)})
    print(json.dumps({k: v for k, v in res.items() if k != "offpath_points"}, indent=1))


if __name__ == "__main__":
    main()
