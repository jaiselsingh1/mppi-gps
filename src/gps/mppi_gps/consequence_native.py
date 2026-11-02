"""Native C1 replay/assay/execution. Importing this module constructs nothing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .consequence_storage import C1TraceRecorder
from .provenance import sha, write
from .runtime import restore, snapshot


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def source_snapshot(source: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return dict(
        state=source["states"][index],
        warmstart=source["warmstarts"][index],
        integration=source["integrations"][index],
        spec=source["integration_spec"],
        dimensions=source["dimensions"],
    )


def same_replay(
    actual: dict[str, Any],
    source: dict[str, np.ndarray],
    index: int,
    observation: np.ndarray,
    cost: float | None = None,
) -> None:
    for key, target in (
        ("state", "states"),
        ("warmstart", "warmstarts"),
        ("integration", "integrations"),
    ):
        if not np.allclose(actual[key], source[target][index], rtol=0, atol=1e-10):
            raise ValueError(f"C1 replay mismatch: {key} boundary {index}")
    if not np.allclose(observation, source["observations"][index], rtol=0, atol=1e-10):
        raise ValueError(f"C1 replay mismatch: observation boundary {index}")
    if cost is not None and not np.isclose(cost, source["costs"][index - 1], rtol=0, atol=1e-10):
        raise ValueError(f"C1 replay mismatch: native cost control {index - 1}")


class NativeConsequence:
    def __init__(self, run: Path, config: dict[str, Any], meter: Any):
        import mujoco

        from src.envs.walker2d import Walker2d

        self.run, self.config, self.meter = run, config, meter
        self.prepared = run / "prepared"
        self.catalog = json.loads((self.prepared / "catalog.json").read_text())
        task = json.loads((self.prepared / "inputs/task.json").read_text())
        if sha(run / "source/assets/walker2d.xml") != task["xml_sha256"]:
            raise ValueError("C1 XML differs from the inherited task")
        meter.reserve_work("environment_constructions")
        meter.record("environment_requested", mjdata_objects=17)
        self.env = Walker2d(**task["task"], use_warp=False)
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        if (self.env.model.nq, self.env.model.nv, self.env.model.nu) != (9, 9, 6):
            raise ValueError("C1 native dimensions changed")
        if mujoco.mj_stateSize(self.env.model, self.spec) != 91:
            raise ValueError("C1 inherited integration width changed")
        meter.returned("environment_constructions", 1)
        meter.record("environment_complete", integration_size=91, mjdata_objects=17)
        self.recorder = C1TraceRecorder(meter, meter.out.parent / "active")
        self.baseline_costs: dict[str, np.ndarray] = {}

    def episode(self, tag: str) -> dict[str, np.ndarray]:
        return load_npz(self.prepared / self.catalog["episodes"][tag]["file"])

    def _snapshot(self) -> dict[str, np.ndarray]:
        self.meter.reserve_work("snapshot_reads")
        try:
            result = snapshot(self.env)
        except BaseException:
            self.meter.uncertain("snapshot_reads", 1)
            raise
        self.meter.returned("snapshot_reads", 1)
        return result

    def _restore(self, context: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        def operation():
            return restore(self.env, context)

        # runtime.restore performs one explicit mj_forward and one verification snapshot.
        self.meter.reserve_work("external_mj_forward_calls")
        self.meter.reserve_work("snapshot_reads")
        try:
            actual = self.meter.measure(
                "restores", 1, operation, purpose="canonical_byte_exact_restore"
            )
        except BaseException:
            self.meter.uncertain("external_mj_forward_calls", 1)
            self.meter.uncertain("snapshot_reads", 1)
            raise
        self.meter.returned("external_mj_forward_calls", 1)
        self.meter.returned("snapshot_reads", 1)
        return actual

    def _reset(self, seed: int) -> np.ndarray:
        def operation():
            state = np.random.get_state()
            try:
                np.random.seed(seed)
                return self.env.reset()
            finally:
                np.random.set_state(state)

        self.meter.reserve_work("external_mj_forward_calls", 2)
        try:
            observation = self.meter.measure("resets", 1, operation, purpose="paired_reset")
        except BaseException:
            self.meter.uncertain("external_mj_forward_calls", 2)
            raise
        self.meter.returned("external_mj_forward_calls", 2)
        return observation

    def trace(
        self,
        alias: str,
        *,
        source: dict[str, np.ndarray] | None = None,
        source_tag: str | None = None,
        t: int = 0,
        horizon: int = 1000,
        first: np.ndarray | None = None,
        actor: Any = None,
        actor_alias: str | None = None,
        seed: int | None = None,
        baseline: bool = False,
    ) -> dict[str, Any]:
        if actor is None:
            current = self._restore(source_snapshot(source, t))
            observation = self.env._get_obs().copy()
        else:
            observation = self._reset(seed)
            current = self._snapshot()
        self.recorder.begin(alias, horizon, current, observation)
        trace_metadata = dict(
            kind="baseline" if baseline else "evaluation" if actor is not None else "assay",
            source=source_tag,
            t=t,
            actor=actor_alias,
            seed=seed,
            integration_spec=self.spec,
            dimensions=[9, 9, 6],
        )
        costs, velocities, falls = [], [], []
        done = False
        try:
            if baseline:
                same_replay(current, source, 0, observation)
            elif actor is not None and source is not None:
                # Source-context evaluations reuse the teacher's exact reset
                # seed. Check its saved initial boundary; never infer equality.
                same_replay(current, source, 0, observation)
            for index in range(horizon):
                if actor is None:
                    action = (
                        first if index == 0 and first is not None else source["actions"][t + index]
                    )
                else:
                    self.meter.after_prediction = lambda value: self.recorder.prediction(
                        value, actor_alias=actor_alias
                    )
                    try:
                        # Keep deployment's original one-dimensional input shape.
                        action = actor.predict(observation, meter=self.meter)
                    finally:
                        self.meter.after_prediction = None
                # Match the original instrumented native interface: both
                # physics and native control-cost arithmetic receive float64.
                action = np.asarray(action, dtype=np.float64)
                work = dict(native_controls=1, physics_substeps=4)
                self.meter.authorize(work)
                actual_pre = self._snapshot()
                actual_observation = self.env._get_obs().copy()
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
                    self.meter.returned_unpersisted["native_controls"] = (
                        self.meter.returned_unpersisted.get("native_controls", 0) + 1
                    )
                    self.meter.returned_unpersisted["physics_substeps"] = (
                        self.meter.returned_unpersisted.get("physics_substeps", 0) + 4
                    )
                    try:
                        self.meter.record("native_error", alias=alias, ordinal=index, returned=True)
                    except BaseException:
                        pass
                    raise
                costs.append(float(cost))
                velocities.append(float(info["x_velocity"]))
                falls.append(float(info["fall_cost"]))
                if baseline:
                    same_replay(current, source, index + 1, observation, float(cost))
                    if done:
                        raise ValueError(
                            "C1 successful source unexpectedly terminated during replay"
                        )
                if done:
                    break
            absorbing = 5.0 * (horizon - len(costs)) if done else 0.0
            reference = self.recorder.close(absorbing, trace_metadata)
        except BaseException as error:
            self.recorder.fail(error, meta=trace_metadata)
            raise
        native_cost = float(np.sum(costs, dtype=np.float64))
        terminal = float(np.sum(falls, dtype=np.float64))
        raw = native_cost - terminal
        mae = float(np.mean(np.abs(np.asarray(velocities) - 1.5)))
        survived = len(costs) == horizon and not done
        result = dict(
            tag=alias,
            source_tag=source_tag,
            t=t,
            h=horizon,
            trace=reference,
            raw_cost=raw,
            native_cost=native_cost,
            absorbing_cost=absorbing,
            cost=native_cost + absorbing,
            controls=len(costs),
            done=bool(done),
            native_terminal=terminal,
            survived=survived,
            walking=bool(horizon == 1000 and survived and mae <= 0.3),
            mae=mae,
            actor=actor_alias,
            seed=seed,
        )
        if baseline:
            self.baseline_costs[source_tag] = np.asarray(costs, dtype=np.float64)
            result["native_costs"] = costs
        return result

    def baselines(self) -> dict[str, Any]:
        results = []
        for tag in self.catalog["episodes"]:
            result = self.trace(
                "baseline_" + tag, source=self.episode(tag), source_tag=tag, baseline=True
            )
            if not result["walking"] or not result["survived"] or result["controls"] != 1000:
                raise ValueError("C1 replay failed the full walking source outcome gate")
            results.append(result)
        receipt = dict(
            schema="walker-c1-replay-gate-v1",
            passed=True,
            episodes=results,
            tolerance=dict(rtol=0, atol=1e-10),
            native_controls=12000,
            substitution_controls_before_gate=0,
        )
        write(self.run / "worker/replay_gate.json", receipt)
        return receipt

    def assays(self) -> list[dict[str, Any]]:
        rows = []
        with (self.run / "worker/assays.jsonl").open("x") as output:
            for tag, descriptor in self.catalog["episodes"].items():
                source = self.episode(tag)
                for t in range(1000):
                    h = min(100, 1000 - t)
                    result = self.trace(
                        f"assay_{tag}_{t:04d}",
                        source=source,
                        source_tag=tag,
                        t=t,
                        horizon=h,
                        first=source["policy_actions"][t],
                    )
                    baseline_cost = float(
                        np.sum(self.baseline_costs[tag][t : t + h], dtype=np.float64)
                    )
                    result.update(
                        stream=descriptor["stream"],
                        case=descriptor["case"],
                        baseline_cost=baseline_cost,
                        delta=result["cost"] - baseline_cost,
                    )
                    output.write(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
                    output.flush()
                    # The trace archive and its compact completion are already fsynced.
                    rows.append({key: value for key, value in result.items() if key != "trace"})
        return rows
