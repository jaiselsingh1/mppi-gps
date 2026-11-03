"""Post hoc for policy_picks: consistency ceiling of a state-determined label on the bank's states."""
import json, sys
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.diagnostics.action_spectrum import FIRST, neighbour_index, state_dependence
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.policy_picks import fit, load_bank, relabel
from scripts.diagnostics.sac_continuation import K1, load_teachers

rng = np.random.default_rng(0)
pooled = np.concatenate([t["observations"][FIRST:] for t in load_teachers(K1)])
mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
walks = load_bank(Path("runs/2026-11-03_sample_bank_001"))
index = neighbour_index([(observation(e["states"])[FIRST:] - mu) / sd for e in walks])
r2 = lambda y: state_dependence(np.concatenate([a[FIRST:] for a in y]), index, rng)["r2"]
pi0 = fit(walks, [e["actions"] for e in walks], 0, False)
pred = [pi0.predict(observation(e["states"]).astype(np.float32)) for e in walks]
labels, _ = relabel(walks, pi0, 0.2, 10.0)
steps = [l - p for l, p in zip(labels, pred)]
centre = [e["nominal_first"] for e in walks]
out = dict(
    policy_prediction=r2(pred),
    mppi_action=r2([e["actions"] for e in walks]),
    mppi_sampling_centre=r2(centre),
    relabel_h02_lam10_round1=r2(labels),
    relabel_step_from_policy=r2(steps),
    step_share_of_label_variance=float(np.concatenate(steps).var(0).sum() / np.concatenate(labels).var(0).sum()),
)
print(json.dumps(out, indent=2))
Path("runs/2026-11-03_policy_picks_001/posthoc_ceiling.json").write_text(json.dumps(out, indent=2) + "\n")
