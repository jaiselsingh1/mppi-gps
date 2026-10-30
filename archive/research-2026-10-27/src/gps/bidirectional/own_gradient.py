"""Frozen-parent action-gradient linearization, not teacher-action regression."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import torch

from .learning import PolicyLearner
from .policy import FamilyGaussianActor, policy_hash


SEMANTICS = "frozen_parent_own_occupancy_action_gradient"


@dataclass(frozen=True)
class OwnGradientBatch:
    observations: np.ndarray
    families: np.ndarray
    states: np.ndarray
    saved_means: np.ndarray
    gradients: np.ndarray
    actor_hash: str
    task_hash: str
    row_references: tuple[dict, ...]
    kind: str
    scale: float = 25.
    source: str = SEMANTICS


def validate_batch(batch, expected_actor, expected_task, obs_dim=8, act_dim=1):
    """Validate exact own-state/owner/query ancestry before a training forward."""
    if not isinstance(batch, OwnGradientBatch) or batch.source != SEMANTICS:
        raise ValueError("Own-occupancy gradient evidence required; teacher batches are not accepted")
    if (batch.actor_hash != expected_actor or batch.task_hash != expected_task
            or not isinstance(expected_actor, str) or not expected_actor
            or not isinstance(expected_task, str) or not expected_task
            or batch.kind not in ("full", "local", "null") or batch.scale != 25.):
        raise ValueError("Frozen gradient owner/task/kind/scale differs")
    obs, families, states, means, gradients = map(np.asarray, (
        batch.observations, batch.families, batch.states, batch.saved_means, batch.gradients))
    n = len(states)
    if (n != 200 or obs_dim != 8 or act_dim != 1 or obs.shape != (n, 8) or families.shape != (n,)
            or families.dtype.kind not in "iu" or np.any(families != 0) or states.shape != (n, 2)
            or means.shape != (n, 1) or gradients.shape != (n, 1)
            or not all(np.isfinite(a).all() for a in (obs, states, means, gradients))
            or np.any(np.abs(means) > 1) or np.any(np.abs(states[:, 1]) > 1)
            or np.any(states[:, 0] != np.floor(states[:, 0])) or np.any(states[:, 0] < 0)
            or np.any(states[:, 0] >= 25)):
        raise ValueError("Aligned finite 200-row Ending own-gradient arrays required")
    expected_obs = np.zeros((n, 8)); expected_obs[:, 0] = states[:, 1]
    expected_obs[:, 1] = states[:, 0]/25; expected_obs[:, 2] = 1-expected_obs[:, 1]
    if not np.array_equal(obs, expected_obs) or (batch.kind == "null" and np.any(gradients != 0)):
        raise ValueError("Gradient state/observation alignment or null direction differs")
    if not isinstance(batch.row_references, (tuple, list)) or len(batch.row_references) != n:
        raise ValueError("Every slope row needs explicit actual and model query ancestry")
    for i, ref in enumerate(batch.row_references):
        remaining = 25-int(states[i, 0])
        if (not isinstance(ref, Mapping) or ref.get("actor_hash") != expected_actor or ref.get("task_hash") != expected_task
                or ref.get("elapsed") != int(states[i, 0]) or ref.get("semantics") != SEMANTICS
                or not np.array_equal(np.asarray(ref.get("state")), states[i])
                or not isinstance(ref.get("source_episode"), str) or not ref["source_episode"]):
            raise ValueError("Gradient row does not bind its own state and frozen continuation")
        sets = []
        for key in ("center_query_ids", "minus_query_ids", "plus_query_ids"):
            ids = ref.get(key)
            if (not isinstance(ids, (tuple, list)) or len(ids) != remaining or len(set(ids)) != remaining
                    or any(not isinstance(v, str) or not v for v in ids)):
                raise ValueError("Complete unique query suffixes are required for each gradient row")
            sets.append(set(ids))
        if any(sets[a].intersection(sets[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Actual center and counterfactual branches cannot reuse one control")
        for key in ("center_raw", "slope_raw"):
            raw = ref.get(key)
            if (not isinstance(raw, Mapping) or not isinstance(raw.get("file"), str) or not raw["file"]
                    or not isinstance(raw.get("sha256"), str) or not raw["sha256"]
                    or type(raw.get("bytes")) is not int or raw["bytes"] < 0):
                raise ValueError("Gradient row raw evidence reference missing")
    return dict(observations=obs.copy(), families=families.copy(), states=states.copy(),
        saved_means=means.copy(), gradients=gradients.copy(), weights=np.ones(n),
        scale=batch.scale, kind=batch.kind)


class OwnGradientLearner(PolicyLearner):
    """Only the auxiliary interface differs; complete own-MC PPO is unchanged."""
    def _auxiliary(self, aux):
        if aux is None:
            raise ValueError("This assay requires a full/local/null own-gradient batch")
        values = validate_batch(aux, policy_hash(self.actor), getattr(self, "own_gradient_task_hash", None),
            self.actor.obs_dim, self.actor.act_dim)
        self.actor._inputs(values["observations"], values["families"])
        return dict(observations=self._tensor(values["observations"]),
            families=self._tensor(values["families"], True), weights=self._tensor(values["weights"]),
            saved_means=self._tensor(values["saved_means"]).detach(),
            gradients=self._tensor(values["gradients"]).detach(), scale=values["scale"])

    def _aux_loss(self, aux, index):
        mean, _ = self.actor(aux["observations"][index], aux["families"][index])
        bounded = self.actor._bounded(mean)
        return ((bounded-aux["saved_means"][index])*aux["gradients"][index]).sum(dim=-1).mean()/aux["scale"]


def clone_gradient_learner(parent, task_hash):
    """No forward: copy the complete optimizer and private streams into subclass."""
    if not isinstance(task_hash, str) or not task_hash:
        raise ValueError("Explicit task identity required")
    actor = FamilyGaussianActor.from_checkpoint(parent.actor.policy_state(), device=parent.device)
    child = OwnGradientLearner(actor, parent.cfg)
    child.load_state_dict(deepcopy(parent.state_dict()))
    child.own_gradient_task_hash = task_hash
    return child
