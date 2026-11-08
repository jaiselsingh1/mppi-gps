"""The similarity: a cost on where MPPI's rollouts go, and the walk the others agree with.

Each MPPI rollout pays w x sum over its states of the share-weighted squared distance to the
nearest state of each reference set (observed coordinates, in sd units of the training walks).
The loop's references are last round's medoid walk (share 1 - s) and the states the deployed
policy visits alone while it stays up (share s), s = the policy's task score.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def reference_trees(refs, obs, sd):
    """refs: [(states (N, n) or None, share)] -> [(KD tree on observed sd-scaled states, share)]."""
    return [
        (cKDTree(ref[:, obs] / sd), share)
        for ref, share in refs
        if share > 0 and ref is not None and len(ref)
    ]


def similarity_cost(trees, w, obs, sd):
    """MPPI's prior cost: (states (K, H, n), actions) -> (K,) extra cost per rollout."""

    def prior(states, actions):
        flat = states.reshape(-1, states.shape[-1])[:, obs] / sd
        total = 0.0
        for tree, share in trees:
            d, _ = tree.query(flat)
            total = total + share * d**2
        return w * np.reshape(total, states.shape[:2]).sum(1)

    return prior


def set_distance(a, b, obs, sd):
    """rms distance from each state of a to the nearest state of b (observed, sd units)."""
    d, _ = cKDTree(b[:, obs] / sd).query(a[:, obs] / sd)
    return float(np.sqrt((d**2).mean()))


def medoid(walks, obs, sd):
    """The walk closest on average to the others (symmetrized set distance) and its mean distance."""
    n = len(walks)
    pair = np.array(
        [[set_distance(walks[i], walks[j], obs, sd) if i != j else 0.0 for j in range(n)]
         for i in range(n)]
    )  # fmt: skip
    m = int(np.argmin((pair + pair.T).sum(1)))
    return m, float(np.mean([pair[i, m] for i in range(n) if i != m]))
