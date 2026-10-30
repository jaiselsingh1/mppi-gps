"""Load runner traces and restore individual MPPI decisions exactly.

A decision at step t of an episode depends on the simulator state, the solver
warm-start, and the planner snapshot (nominal U, RNG state, refinement flag).
Runner traces store all three, aligned so that states[t] is the pre-action
state of action t.
"""

import json
from pathlib import Path

import numpy as np

from src.envs.mujoco_env import MuJoCoEnv
from src.mppi.mppi import MPPI

_STEP_FIELDS = ("actions", "contacts", "contact_normal_forces", "costs")
_STATE_FIELDS = ("states", "warmstarts")
_DECISION_FIELDS = ("pre_plan_nominals", "plans", "planner_rng_states", "initial_plan_flags")


def load_episode(trace_path: Path | str, episode: int) -> dict:
    """Slice one episode out of a runner trace file."""
    with np.load(trace_path, allow_pickle=False) as data:
        step_offsets = data["episode_step_offsets"]
        state_offsets = data["episode_state_offsets"]
        if not 0 <= episode < len(step_offsets) - 1:
            raise IndexError(f"episode {episode} is not in {trace_path}.")
        steps = slice(int(step_offsets[episode]), int(step_offsets[episode + 1]))
        states = slice(int(state_offsets[episode]), int(state_offsets[episode + 1]))
        result = {
            "reset_seed": int(data["reset_seeds"][episode]),
            "planner_seed": int(data["planner_seeds"][episode]),
        }
        for key in _STATE_FIELDS:
            result[key] = data[key][states]
        for key in _STEP_FIELDS + _DECISION_FIELDS:
            if key in data.files:
                result[key] = data[key][steps]
    if len(result["states"]) != len(result["actions"]) + 1:
        raise ValueError("Trace must hold the initial state plus one state per action.")
    return result


def decision_snapshot(episode: dict, step: int) -> dict:
    """Planner snapshot entering decision `step` of a loaded episode."""
    if "planner_rng_states" not in episode:
        raise KeyError("Trace has no planner snapshots; replay from the episode start.")
    return {
        "U": episode["pre_plan_nominals"][step],
        "rng_state": json.loads(str(episode["planner_rng_states"][step])),
        "initial_plan": bool(episode["initial_plan_flags"][step]),
    }


def restore_decision(env: MuJoCoEnv, controller: MPPI, episode: dict, step: int) -> np.ndarray:
    """Put env and controller exactly where they were before action `step`."""
    env.set_state(episode["states"][step])
    env.set_warmstart(episode["warmstarts"][step])
    controller.restore(decision_snapshot(episode, step))
    return env.get_state()


def compare_fields(reference: dict, replay: dict, fields: tuple[str, ...]) -> dict:
    """Exact per-field comparison with the first diverging row and max error."""
    report = {}
    for key in fields:
        expected = np.asarray(reference[key])
        actual = np.asarray(replay[key])
        length = min(len(expected), len(actual))
        diverging = [
            index for index in range(length) if not np.array_equal(expected[index], actual[index])
        ]
        numeric = np.issubdtype(expected.dtype, np.number) and np.issubdtype(
            actual.dtype, np.number
        )
        report[key] = {
            "exact": len(expected) == len(actual) and not diverging,
            "reference_length": len(expected),
            "replay_length": len(actual),
            "first_divergence": diverging[0] if diverging else None,
            "max_abs_error": (
                float(np.max(np.abs(expected[:length] - actual[:length])))
                if numeric and length
                else None
            ),
        }
    return report
