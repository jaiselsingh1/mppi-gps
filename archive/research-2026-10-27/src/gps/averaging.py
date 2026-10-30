"""Averaged iterates of MPPI-GPS policies.

Stochastic mirror descent's guarantee is for the averaged iterate, not the
last one. For policies, the average of iterates is the mixture policy; acting
with its mean is acting with the mean of the members' actions
(`AveragedPolicy`). `weight_average` averages the networks' weights instead,
which gives one network but has no such interpretation.
"""
import copy

import numpy as np
import torch

from src.rl.sac import SacActor


class AveragedPolicy:
    """Deterministic policy that acts with the mean of several policies' actions."""

    def __init__(self, policies: list[SacActor]) -> None:
        if not policies:
            raise ValueError("AveragedPolicy needs at least one policy.")
        self.policies = policies

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if not deterministic:
            raise ValueError("AveragedPolicy only acts deterministically.")
        return np.mean([policy.act(obs, deterministic=True) for policy in self.policies], axis=0)


def weight_average(states: list[dict]) -> dict:
    """Policy state (load_policy format) whose actor weights are the mean of `states`'."""
    if not states:
        raise ValueError("weight_average needs at least one policy state.")
    first = states[0]
    for state in states[1:]:
        if {k: v for k, v in state.items() if k != "actor"} != {k: v for k, v in first.items() if k != "actor"}:
            raise ValueError("policy states differ outside their actor weights.")
        if state["actor"].keys() != first["actor"].keys():
            raise ValueError("policy states have different actor parameters.")
    averaged = copy.deepcopy(first)
    averaged["actor"] = {name: torch.stack([state["actor"][name] for state in states]).mean(dim=0)
                         for name in first["actor"]}
    return averaged
