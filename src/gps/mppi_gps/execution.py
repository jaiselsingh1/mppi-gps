"""Native teacher/student execution, imported without constructing a simulator.

The simulator, planner and Torch remain lazy execution-boundary imports.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .metrics import legacy_summary as summary
from .provenance import sha, write_json

if TYPE_CHECKING:
    from .policy import Policy
    from .runtime import Episode, Meter


class Authority:
    """Small version-bound support record, separate from planner state."""

    def __init__(
        self,
        actor: Policy,
        observations: np.ndarray,
        radius: float | None,
        meter: Meter,
        task_hash: str,
    ) -> None:
        from scipy.spatial import cKDTree

        self.owner = actor.hash()
        self.task_hash = task_hash
        self.radius = radius
        self.meter = meter
        self.valid = True
        self.promoted = False
        normal = actor.normalization
        self.mean = np.asarray(normal["mean"])[:17]
        self.std = np.asarray(normal["std"])[:17]
        rows = np.asarray(observations)
        rows = rows[np.linspace(0, len(rows) - 1, min(512, len(rows)), dtype=int)]
        self.tree = cKDTree((rows - self.mean) / self.std)
        self.evidence = meter.save_arrays(
            "authority_support", observations=rows, mean=self.mean, std=self.std
        )
        meter.record(
            "provisional_candidate_registered",
            policy_hash=self.owner,
            task_hash=task_hash,
            support=self.evidence,
            radius=radius,
            provisional_only=True,
        )

    def supports(self, observations: np.ndarray) -> np.ndarray:
        rows = np.asarray(observations)
        if self.radius is None:
            return np.ones(rows.shape[:-1], dtype=bool)
        distance, _ = self.tree.query(((rows - self.mean) / self.std).reshape(-1, 17))
        return (distance / np.sqrt(17) <= self.radius).reshape(rows.shape[:-1])

    def allowed(self, actor: Policy, observation: np.ndarray, provisional: bool) -> bool:
        if actor.hash() != self.owner:
            self.revoke("policy_version_changed")
        return bool(
            self.valid and (self.promoted or provisional) and self.supports(np.asarray(observation))
        )

    def revoke(self, reason: str) -> None:
        if self.valid:
            self.meter.record("guidance_revoked", policy_hash=self.owner, reason=reason)
        self.valid = False
        self.promoted = False


class PhysicalStudy:
    """Shared native execution boundary for diagnostic and coupled controllers."""

    def __init__(self, out: str | Path, cfg: Mapping[str, Any], meter: Meter) -> None:
        import torch

        from src.envs.walker2d import Walker2d
        from src.utils.config import MPPIConfig

        self.out, self.cfg, self.meter = Path(out), cfg, meter
        torch.set_num_threads(4)
        if (
            cfg["policy"]["hidden"] != 128
            or cfg["episode_steps"] != 1000
            or cfg["gates"]["velocity_mae"] != 0.3
        ):
            raise ValueError(
                "V2 physical studies fix 128 hidden units and the 1,000-control native walking criterion"
            )
        self.env = Walker2d(**cfg["task"])
        meter.instrument(self.env)
        self.mppi_cfg = MPPIConfig(**cfg["mppi"])
        self.records = []
        self.task_hash = hashlib.sha256(
            json.dumps(cfg["task"], sort_keys=True).encode()
        ).hexdigest()

    def checkpoint(
        self, actor: Policy, state: dict[str, Any] | None, tag: str, **metadata: Any
    ) -> str:
        path = self.out / "checkpoints" / (tag + ".pt")
        actor.checkpoint(path, optimizer_state=state, metadata=dict(tag=tag, **metadata))
        self.meter.record(
            "checkpoint",
            file=str(path.relative_to(self.out)),
            sha256=sha(path),
            policy_hash=actor.hash(),
        )
        return str(path.relative_to(self.out))

    def actor_eval(
        self, actor: Policy, root: int, count: int, tag: str, feature_mode: str = "state"
    ) -> tuple[list[Episode], dict[str, Any]]:
        from .runtime import execute_episode

        traces = []
        for index in range(count):
            previous = np.zeros(6)

            def control(state, warm, obs, t):
                nonlocal previous
                features = (
                    obs
                    if actor.obs_dim == 17
                    else np.concatenate(
                        (obs, previous if feature_mode == "history" else np.zeros(6))
                    )
                )
                action = actor.predict(features, meter=self.meter)
                previous = action.copy()
                return action, dict(policy_hash=actor.hash(), feature_mode=feature_mode)

            self.meter.context = dict(role="actor_only", cohort=tag, feature_mode=feature_mode)
            episode = execute_episode(
                self.env,
                control,
                root + index,
                self.cfg["episode_steps"],
                self.meter,
                f"{tag}_{index:02d}",
            )
            traces.append(episode)
            print(
                json.dumps(
                    dict(
                        event="actor_episode",
                        tag=episode.tag,
                        success=episode.success,
                        steps=episode.steps,
                        cost=episode.cost,
                    )
                ),
                flush=True,
            )
        value = summary(traces)
        self.records.append(
            dict(
                kind="actor",
                stage=tag,
                policy_hash=actor.hash(),
                feature_mode=feature_mode,
                **value,
            )
        )
        return traces, value

    def teacher(
        self,
        seed: int,
        planner_seed: int,
        tag: str,
        actor: Policy | None = None,
        rho: float = 0.0,
        authority: Authority | None = None,
        provisional: bool = True,
    ) -> tuple[Episode, int, float | None]:
        from .runtime import execute_episode
        from .search import CoupledSelector, IndependentMPPI

        planner = IndependentMPPI(self.env, self.mppi_cfg, planner_seed)
        selector = CoupledSelector(alpha=rho, eta_u=self.cfg["coupling"]["eta_u"])
        changes = [0]

        def control(state, warm, obs, t):
            self.meter.record(
                "independent_planner_before", episode=tag, step=t, private=planner.private_state()
            )
            bank = planner.propose(state, warm)
            enabled = bool(
                rho > 0
                and actor is not None
                and authority is not None
                and authority.allowed(actor, obs, provisional)
            )
            plan, info = selector.select(
                bank,
                actor=actor,
                meter=self.meter,
                enabled=enabled,
                supported=None if authority is None else authority.supports,
            )
            change = float(
                np.linalg.norm(
                    np.asarray(info["guided_first_action"])
                    - np.asarray(info["baseline_first_action"])
                )
            )
            changes[0] += int(enabled and change > self.cfg["gates"]["action_change_atol"])
            if info.get("numerical_revocation") and authority is not None:
                authority.revoke("numerical_failure")
            self.meter.record("decision", episode=tag, step=t, action_change_norm=change, **info)
            self.meter.record(
                "independent_planner_after", episode=tag, step=t, private=planner.private_state()
            )
            return plan[0], info

        self.meter.context = dict(role="teacher", cohort=tag, rho=float(rho))
        episode = execute_episode(
            self.env, control, seed, self.cfg["episode_steps"], self.meter, tag
        )
        agreement = None
        agreement_evidence = None
        if actor is not None:
            prediction = actor.predict(episode.observations, meter=self.meter)
            agreement = float(np.mean(np.sum((episode.actions - prediction) ** 2, axis=1)))
            agreement_evidence = self.meter.save_arrays(
                "executed_agreement_" + tag,
                policy_actions=prediction,
                executed_actions=episode.actions,
                squared_disagreement=np.sum((episode.actions - prediction) ** 2, axis=1),
            )
        self.meter.record(
            "teacher_complete",
            episode=tag,
            rho=float(rho),
            changed_decisions=changes[0],
            complete_cost=episode.cost,
            success=episode.success,
            executed_agreement=agreement,
            agreement_evidence=agreement_evidence,
        )
        print(
            json.dumps(
                dict(
                    event="teacher_episode",
                    tag=tag,
                    rho=float(rho),
                    success=episode.success,
                    cost=episode.cost,
                    changed_decisions=changes[0],
                    executed_agreement=agreement,
                )
            ),
            flush=True,
        )
        return episode, changes[0], agreement

    def fit(
        self,
        parent: Policy,
        state: dict[str, Any] | None,
        dataset: tuple[np.ndarray, np.ndarray],
        tag: str,
        seed: int,
    ) -> tuple[Policy, dict[str, Any]]:
        from .policy import fit_policy

        options = dict(self.cfg["learning"], seed=seed)
        child, new_state, evidence = fit_policy(
            parent, *dataset, optimizer_state=deepcopy(state), meter=self.meter, tag=tag, **options
        )
        write_json(self.out / "learning" / (tag + ".json"), evidence)
        self.checkpoint(
            child,
            new_state,
            tag,
            feature_mode="state",
            parent_hash=parent.hash(),
            deployment_qualified=False,
        )
        self.records.append(
            dict(
                kind="fit",
                stage=tag,
                parent_hash=parent.hash(),
                child_hash=child.hash(),
                updates=options["updates"],
            )
        )
        return child, new_state

    def authority(self, actor: Policy, radius: float | None = None) -> Authority:
        anchors = self.training_observations
        return Authority(actor, anchors, radius, self.meter, self.task_hash)

    def close(self) -> None:
        self.env.close()
