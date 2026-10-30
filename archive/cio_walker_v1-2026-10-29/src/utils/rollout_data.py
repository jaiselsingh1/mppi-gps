"""Align rollout data with the states at which actions were chosen.

MuJoCoEnv.batch_rollout returns the state *after* each action: states[:, t]
is s_{t+1}. Pairing states[:, t] with actions[:, t] would score or supervise a
policy on the result of its action instead of the state where it must choose
that action. These helpers rebuild s_0..s_{H-1} and mark actions chosen inside
the healthy prefix.
"""

import numpy as np


def pre_action_states(
    initial_state: np.ndarray,
    rollout_states: np.ndarray,
) -> np.ndarray:
    """Return (K, H, nstate) states s_0..s_{H-1} at which actions 0..H-1 act.

    initial_state is the planning state, shape (nstate,) or (K, nstate).
    rollout_states is batch_rollout output, shape (K, H, nstate).
    """
    rollout_states = np.asarray(rollout_states)
    if rollout_states.ndim != 3:
        raise ValueError("rollout_states must have shape (K, H, nstate).")
    K, _, nstate = rollout_states.shape
    initial = np.asarray(initial_state, dtype=rollout_states.dtype)
    if initial.shape == (nstate,):
        initial = np.broadcast_to(initial, (K, nstate))
    elif initial.shape != (K, nstate):
        raise ValueError("initial_state must have shape (nstate,) or (K, nstate).")
    return np.concatenate([initial[:, None, :], rollout_states[:, :-1, :]], axis=1)


def pre_fall_action_mask(post_action_healthy: np.ndarray) -> np.ndarray:
    """Mark actions chosen from a state on the healthy prefix.

    post_action_healthy[:, t] is the health of s_{t+1}. The planning state s_0
    is the live state and is assumed healthy, because live episodes end at the
    first fall. Action t is kept when s_1..s_t are healthy, so the action that
    causes the first fall is kept and every later action is excluded.
    """
    healthy = np.asarray(post_action_healthy, dtype=bool)
    if healthy.ndim != 2:
        raise ValueError("post_action_healthy must have shape (K, H).")
    mask = np.ones_like(healthy)
    if healthy.shape[1] > 1:
        mask[:, 1:] = np.logical_and.accumulate(healthy[:, :-1], axis=1)
    return mask


def episode_action_pairs(
    states: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair one live episode's pre-action states with its executed actions.

    Runner traces hold the initial state plus one post-action state per action,
    so states has exactly one more row than actions.
    """
    states = np.asarray(states)
    actions = np.asarray(actions)
    if len(states) != len(actions) + 1:
        raise ValueError(
            "states must hold the initial state plus one post-action state per action."
        )
    return states[:-1], actions
