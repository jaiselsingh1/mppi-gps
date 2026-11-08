"""Is the critic export mismatch numerical precision? Compare numpy (float64) with Brax's critic under JAX's default
GPU matmul precision and under 'highest', on the smoke run's latest params and recorded-like observations."""
import pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path.home() / "mppi-gps-claude-sac"; sys.path.insert(0, str(ROOT / "rss_checks" / "rl"))
import jax, jax.numpy as jnp
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
S = pickle.load(open(ROOT / "runs/loop/smoke/state.pkl", "rb")); p = S["params"]
net = ppo_networks.make_ppo_networks(17, 6, preprocess_observations_fn=running_statistics.normalize)
ref = np.load(ROOT / "runs/rc/motion2/ref.npz")["o"][:1000].astype(np.float64)
nrm = p[0]; vp = p[2]["params"]
h = (ref - np.asarray(nrm.mean, np.float64)) / np.asarray(nrm.std, np.float64)
for i in range(5):
    z = h @ np.asarray(vp[f"hidden_{i}"]["kernel"], np.float64) + np.asarray(vp[f"hidden_{i}"]["bias"], np.float64); h = z / (1 + np.exp(-z))
vn = (h @ np.asarray(vp["hidden_5"]["kernel"], np.float64) + np.asarray(vp["hidden_5"]["bias"], np.float64))[:, 0]
for prec in ("default", "highest"):
    with jax.default_matmul_precision(prec):
        vj = np.asarray(jax.jit(lambda q, o: net.value_network.apply(q[0], q[2], o))(p, jnp.asarray(ref.astype(np.float32)))).reshape(-1)
    print(prec, "max rel err vs numpy float64: %.2e" % (np.max(np.abs(vn - vj)) / np.max(np.abs(vj))))
