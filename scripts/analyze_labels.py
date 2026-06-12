"""Label-achievability and budget-ratchet diagnostics for a GPS run.

Uses the per-step diagnostics saved by collect_episodes
(step_diag_iter_*.npz: beta, dagger, episode, cost_gap) plus the replay
buffer and a checkpoint to answer two adversarial questions from the
literature review:

1. Label achievability: are BC residuals ||label - pi(obs)|| concentrated in
   the (policy-driven episode, beta=0) bin — i.e., recovery states labeled
   with actions the policy has never matched (the PLATO drift-gap concern)?
2. Budget ratchet: does the cumulative certified cost gap trend upward
   within episodes (per-step budgets composing through the state)?

Usage: PYTHONPATH=. python scripts/analyze_labels.py --run-dir runs/walker_gps6
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import GPSConfig, PolicyConfig


def main(run_dir: str = "runs/walker_gps6", env_name: str = "walker2d") -> None:
    run = Path(run_dir)
    gps = GPSConfig.load(env_name)
    policy = DeterministicPolicy(gps.obs_dim, gps.act_dim, PolicyConfig())
    policy.load_state_dict(
        torch.load(run / "checkpoint_latest.pt", map_location="cpu"), strict=False
    )
    policy.eval()

    replay = np.load(run / "replay_latest.npz")
    obs, acts, tags = replay["obs"], replay["acts"], replay["tags"]
    with torch.no_grad():
        mu = policy.forward(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    resid = np.linalg.norm(acts - mu, axis=-1)

    diags = sorted(run.glob("step_diag_iter_*.npz"))
    if not diags:
        print("no step diagnostics found (run predates instrumentation)")
        print(f"overall residual: mean={resid.mean():.3f} p90={np.percentile(resid, 90):.3f}")
        return

    beta = np.concatenate([np.load(p)["beta"] for p in diags])
    dagger = np.concatenate([np.load(p)["dagger"] for p in diags])
    gap = np.concatenate([np.load(p)["cost_gap"] for p in diags])
    episode = np.concatenate(
        [np.load(p)["episode"] + 100 * i for i, p in enumerate(diags)]
    )
    n = min(len(beta), len(resid))
    # replay is trimmed newest-last; diagnostics cover all iters — align tails
    beta, dagger, gap, episode, resid_a = (
        beta[-n:], dagger[-n:], gap[-n:], episode[-n:], resid[-n:]
    )

    print("== label-achievability bins (BC residual ||label - pi(obs)||) ==")
    for dname, dmask in (("planner-driven", ~dagger), ("policy-driven", dagger)):
        for bname, bmask in (("beta=0", beta == 0.0), ("beta>0", beta > 0.0)):
            m = dmask & bmask
            if m.sum() == 0:
                print(f"{dname:>15} {bname}: (empty)")
                continue
            r = resid_a[m]
            print(f"{dname:>15} {bname}: n={m.sum():6d} mean={r.mean():.3f} "
                  f"p90={np.percentile(r, 90):.3f}")

    print("== budget ratchet (cumulative certified gap within episodes) ==")
    slopes = []
    for e in np.unique(episode):
        m = episode == e
        if m.sum() < 20:
            continue
        cum = np.cumsum(gap[m])
        t = np.arange(len(cum))
        slopes.append(np.polyfit(t, cum, 1)[0])
    print(f"episodes={len(slopes)} mean within-episode gap slope="
          f"{np.mean(slopes):.4f} cost/step (positive = ratchet)")


if __name__ == "__main__":
    tyro.cli(main)
