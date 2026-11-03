"""S1.2-V closed-loop reset/restoration execution, with durable actual prefixes.

The source replay controller is the retained parent, including its recorded
falls. It is never an open-loop action replay or a call to C1's success-only
baseline runner. Simulator construction remains inherited and lazy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .consequence_native import NativeConsequence, load_npz


def same_context(actual: dict, expected: dict, observation: Any) -> None:
    """Keep snapshot layout exact and numerical comparisons at the frozen tolerance."""
    for key in ("spec", "dimensions"):
        if (
            np.asarray(actual[key]).shape != np.asarray(expected[key]).shape
            or np.asarray(actual[key]).dtype != np.asarray(expected[key]).dtype
            or not np.array_equal(actual[key], expected[key])
        ):
            raise ValueError("S1.2-V context structure differs: " + key)
    for key in ("state", "warmstart", "integration"):
        left, right = np.asarray(actual[key]), np.asarray(expected[key])
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or not np.allclose(left, right, rtol=0, atol=1e-10)
        ):
            raise ValueError("S1.2-V context differs: " + key)
    left, right = np.asarray(observation), np.asarray(expected["observation"])
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or not np.allclose(left, right, rtol=0, atol=1e-10)
    ):
        raise ValueError("S1.2-V context differs: observation")


def episode_context(source: dict, index: int) -> dict:
    return dict(
        state=source["states"][index],
        warmstart=source["warmstarts"][index],
        integration=source["integrations"][index],
        observation=source["observations"][index],
        spec=source["spec"],
        dimensions=source["dimensions"],
    )


def exact_action(prediction: Any, recorded: Any | None = None) -> np.ndarray:
    """Validate the original float32 batch-one output, then promote for native physics."""
    value = np.asarray(prediction)
    if (
        value.dtype != np.float32
        or value.shape != (6,)
        or not np.isfinite(value).all()
        or np.any(np.abs(value) > 1)
    ):
        raise ValueError("S1.2-V requires one finite bounded float32 action")
    promoted = value.astype(np.float64)
    if recorded is not None:
        expected = np.asarray(recorded)
        if (
            expected.dtype != np.float64
            or expected.shape != (6,)
            or value.tobytes() != expected.astype(np.float32).tobytes()
            or promoted.tobytes() != expected.tobytes()
        ):
            raise ValueError("S1.2-V retained-parent action is not byte exact")
    return promoted


def compare_return(source: dict, index: int, cost: float, done: bool, info: dict) -> None:
    for actual, expected, label in (
        (cost, source["native_costs"][index], "native cost"),
        (info["x_velocity"], source["x_velocities"][index], "velocity"),
        (info["fall_cost"], source["fall_costs"][index], "fall terminal"),
    ):
        if not np.isclose(actual, expected, rtol=0, atol=1e-10):
            raise ValueError("S1.2-V retained-parent replay differs: " + label)
    if bool(done) != bool(source["done"][index]):
        raise ValueError("S1.2-V retained-parent done differs")


class NativeVisitation(NativeConsequence):
    """Reuse native construction and accounting, supplying a restored actor runner."""

    def episode(self, tag: str) -> dict[str, np.ndarray]:
        return load_npz(self.prepared / self.catalog["episodes"][tag]["file"])

    def canonical(self, seed: int) -> dict[str, np.ndarray]:
        return load_npz(self.prepared / self.catalog["canonical_contexts"][str(seed)]["file"])

    def closed_loop(
        self,
        alias: str,
        actor: Any,
        *,
        actor_alias: str,
        checkpoint_sha256: str,
        actor_hash: str,
        stage: str,
        source: dict | None = None,
        source_tag: str | None = None,
        t: int = 0,
        seed: int | None = None,
        expected_replay: bool = False,
        canonical: dict | None = None,
        horizon: int = 1000,
    ) -> dict:
        if source is None:
            if seed is None or canonical is None or expected_replay:
                raise ValueError("S1.2-V own reset needs its bound canonical context")
            observation = self._reset(seed)
            current = self._snapshot()
            same_context(current, canonical, observation)
        else:
            if not 0 <= t < len(source["actions"]):
                raise ValueError("S1.2-V restore anchor is not an executed pre-action state")
            context = episode_context(source, t)
            current = self._restore(context)
            observation = self.env._get_obs().copy()
            same_context(current, context, observation)
        self.recorder.begin(alias, horizon, current, observation)
        metadata = dict(
            kind=stage,
            actor=actor_alias,
            source=source_tag,
            t=t,
            seed=seed,
            integration_spec=self.spec,
            dimensions=[9, 9, 6],
        )
        costs, velocities, terminals = [], [], []
        done = False
        try:
            for ordinal in range(horizon):
                position = t + ordinal
                if expected_replay and position >= len(source["actions"]):
                    raise ValueError("S1.2-V source ended before its recorded fall/horizon")
                self.meter.after_prediction = lambda value: self.recorder.prediction(
                    value, actor_hash=checkpoint_sha256
                )
                try:
                    predicted = actor.predict(observation, meter=self.meter)
                finally:
                    self.meter.after_prediction = None
                action = exact_action(
                    predicted, source["actions"][position] if expected_replay else None
                )
                work = dict(native_controls=1, physics_substeps=4)
                self.meter.authorize(work)
                actual_pre = self._snapshot()
                actual_observation = self.env._get_obs().copy()
                if expected_replay:
                    same_context(actual_pre, episode_context(source, position), actual_observation)
                self.recorder.intent(action, snapshot=actual_pre, observation=actual_observation)
                self.meter.charge(work)
                try:
                    observation, cost, done, info = self.env.step(action)
                except BaseException:
                    self.meter.uncertain("native_controls", 1)
                    self.meter.uncertain("physics_substeps", 4)
                    raise
                self.meter.returned("native_controls", 1)
                self.meter.returned("physics_substeps", 4)
                try:
                    current = self._snapshot()
                    self.recorder.completed(
                        current,
                        observation,
                        cost=float(cost),
                        velocity=float(info["x_velocity"]),
                        fall=float(info["fall_cost"]),
                        done=bool(done),
                    )
                except BaseException:
                    for kind, amount in work.items():
                        self.meter.returned_unpersisted[kind] = (
                            self.meter.returned_unpersisted.get(kind, 0) + amount
                        )
                    raise
                costs.append(float(cost))
                velocities.append(float(info["x_velocity"]))
                terminals.append(float(info["fall_cost"]))
                if expected_replay:
                    same_context(current, episode_context(source, position + 1), observation)
                    compare_return(source, position, cost, done, info)
                if done:
                    break
            if expected_replay:
                expected_controls = min(horizon, len(source["actions"]) - t)
                if len(costs) != expected_controls:
                    raise ValueError("S1.2-V retained-parent returned prefix length differs")
                if not done and expected_controls < horizon:
                    # Only an actual recorded terminal permits a short prefix.
                    raise ValueError("S1.2-V short source lacks its recorded fall")
            absorbing = 5.0 * (horizon - len(costs)) if done else 0.0
            reference = self.recorder.close(absorbing, metadata)
        except BaseException as error:
            self.recorder.fail(error, meta=metadata)
            raise
        native_cost = float(np.sum(costs, dtype=np.float64))
        terminal = float(np.sum(terminals, dtype=np.float64))
        mae = float(np.mean(np.abs(np.asarray(velocities, dtype=np.float64) - 1.5)))
        survived = len(costs) == horizon and not done
        return dict(
            tag=alias,
            stage=stage,
            source_tag=source_tag,
            t=t,
            h=horizon,
            actor=actor_alias,
            actor_hash=actor_hash,
            checkpoint_sha256=checkpoint_sha256,
            seed=seed,
            trace=reference,
            controls=len(costs),
            done=bool(done),
            native_cost=native_cost,
            native_terminal=terminal,
            raw_cost=native_cost - terminal,
            absorbing_cost=absorbing,
            cost=native_cost + absorbing,
            mae=mae,
            survived=survived,
            walking=bool(horizon == 1000 and survived and mae <= 0.3),
            replay_verified=bool(expected_replay),
        )
