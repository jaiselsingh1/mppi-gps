"""Finite J1 native, planning and fixed-update learning boundaries.

Simulator/Torch imports remain local to explicitly paid execution. Historical
checkpoint keys, planner math and C1 array/trace hash contracts are unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .consequence_execution import ConsequenceMeter
from .consequence_storage import C1TraceRecorder
from .proposal_evidence import BinaryEvidenceMeter, read_array
from .provenance import fingerprint, sha, write
from .runtime import BudgetExhausted


def work_limits() -> dict[str, int]:
    return dict(
        native_controls=1_400_000,
        model_controls=210_000_000,
        physics_substeps=850_000_000,
        neural_rows=100_000_000,
        prediction_calls=100_000_000,
        backward_rows=32_768_000,
        optimizer_steps=128_000,
        fits=64,
        environment_constructions=1,
        policy_constructions=128,
        optimizer_constructions=128,
        checkpoint_loads=512,
        model_queries=80_000,
        live_queries=1_400_000,
        resets=2_000,
        restores=2_000,
        snapshot_reads=3_000_000,
        external_mj_forward_calls=20_000,
        contact_measurements=1_500_000,
        contact_force_queries=10_000_000,
    )


class JMeter(ConsequenceMeter):
    """J1-only larger quotas; no historical meter limits or encodings change."""

    def __init__(self, out: Path, config: Mapping[str, Any]):
        maxima = work_limits()
        supplied = dict(config.get("work_limits", maxima))
        if set(supplied) != set(maxima) or any(
            type(v) is not int or not 0 <= v <= maxima[k] for k, v in supplied.items()
        ):
            raise ValueError("J1 work limits must declare every bounded category")
        self.config, self.limits = dict(config), supplied
        self.work: dict[str, int] = {}
        self.completed_work: dict[str, int] = {}
        self.unknown_work: dict[str, int] = {}
        self.returned_unpersisted: dict[str, int] = {}
        budgets = config.get("limits", config)
        self.deadline = time.monotonic() + budgets.get("worker_wall_seconds", 8 * 3600)
        self.rss_bytes = budgets.get("worker_peak_rss_bytes", 8 * 1024**3)
        self._serial = 0
        self.after_prediction = None
        self.max_ledger_rows = 16_000_000
        self.max_array_groups = 70_000
        raw_limit = budgets.get("worker_evidence_bytes", 7 * 1024**3)
        self.external_limit = budgets.get("worker_external_bytes", 512 * 1024**2)
        if not 0 < raw_limit <= 7 * 1024**3:
            raise ValueError("J1 evidence allowance exceeds seven GiB")
        if not 0 <= self.external_limit <= 512 * 1024**2:
            raise ValueError("J1 external allowance exceeds512MiB")
        BinaryEvidenceMeter.__init__(
            self,
            Path(out),
            max_controls=supplied["native_controls"] + supplied["model_controls"],
            max_neural_rows=supplied["neural_rows"],
            max_bytes=raw_limit,
            reserved_bytes=self.external_limit,
            max_event_bytes=2048,
            max_header_bytes=8192,
            max_arrays_per_group=32,
        )

    def record(self, event: str, **fields: Any) -> str:
        if getattr(self, "_sequence", 0) >= self.max_ledger_rows and event != "runtime_close":
            raise BudgetExhausted("J1 ledger row ceiling exhausted")
        return BinaryEvidenceMeter.record(self, event, **fields)

    def save_arrays(self, tag: str, **arrays: Any) -> dict:
        if self._array_sequence >= self.max_array_groups:
            raise BudgetExhausted("J1 array group ceiling exhausted")
        return BinaryEvidenceMeter.save_arrays(self, tag, **arrays)

    def reserve_external(self, amount: int) -> None:
        self.check()
        actual = sum(
            path.stat().st_size
            for path in self.out.parent.rglob("*")
            if path.is_file() and not path.is_relative_to(self.out)
        )
        if actual + amount > self.external_limit:
            raise BudgetExhausted("J1 external-file reservation exceeded")

    def totals(self) -> dict[str, Any]:
        result = super().totals()
        for prefix, values in (
            ("requested", self.work),
            ("completed", self.completed_work),
            ("unknown_partial", self.unknown_work),
        ):
            result[prefix + "_model_controls"] = values.get("model_controls", 0)
            result[prefix + "_controls"] = values.get("model_controls", 0) + values.get(
                "native_controls", 0
            )
        result.update(
            model_queries=self.work.get("model_queries", 0),
            neural_rows=self.completed_work.get("neural_rows", 0),
            backward_rows=self.completed_work.get("backward_rows", 0),
            optimizer_steps=self.completed_work.get("optimizer_steps", 0),
            physics_frames=self.completed_work.get("physics_substeps", 0),
            queries=self.work.get("model_queries", 0) + self.work.get("live_queries", 0),
        )
        return result


def paid(
    meter: JMeter,
    counts: Mapping[str, int],
    operation: Callable[[], Any],
    purpose: str = "",
) -> Any:
    """Persist intent before entering one operation; never infer zero partial work."""
    counts = dict(counts)
    meter.authorize(counts)
    meter._serial += 1
    identifier = meter._serial
    meter.record("paid_requested", operation=identifier, counts=counts, purpose=purpose[:60])
    meter.charge(counts)
    returned = False
    try:
        result = operation()
        returned = True
        for kind, amount in counts.items():
            meter.returned(kind, amount)
        meter.record("paid_complete", operation=identifier, counts=counts)
        return result
    except BaseException:
        target = meter.returned_unpersisted if returned else meter.unknown_work
        for kind, amount in counts.items():
            target[kind] = target.get(kind, 0) + amount
        try:
            meter.record("paid_error", operation=identifier, counts=counts, returned=returned)
        except BaseException:
            pass
        raise


def foot_contacts(
    model: Any, data: Any, meter: JMeter | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Actual post-step contact cache; this function never calls mj_forward."""
    import mujoco

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("foot_geom", "foot_left_geom")
    ]
    if min([floor, *feet]) < 0:
        raise ValueError("Walker contact geometry names differ")
    count = sum(
        {data.contact[i].geom1, data.contact[i].geom2} == {floor, foot}
        for i in range(data.ncon)
        for foot in feet
    )

    def operation() -> tuple[np.ndarray, np.ndarray]:
        present, normal = np.zeros(2, dtype=bool), np.zeros(2, dtype=np.float64)
        force = np.zeros(6)
        for ordinal in range(data.ncon):
            contact = data.contact[ordinal]
            pair = {contact.geom1, contact.geom2}
            for index, foot in enumerate(feet):
                if pair == {floor, foot}:
                    present[index] = True
                    mujoco.mj_contactForce(model, data, ordinal, force)
                    normal[index] += max(0.0, force[0])
        return present, normal

    return (
        operation()
        if meter is None
        else paid(
            meter,
            {"contact_measurements": 1, "contact_force_queries": count},
            operation,
            "actual_contact_cache",
        )
    )


class NativeRunner:
    def __init__(self, run: Path, config: Mapping[str, Any], meter: JMeter):
        import torch

        from src.envs.walker2d import Walker2d

        if torch.get_num_threads() != 1:
            raise ValueError("J1 requires the declared one Torch CPU thread")
        self.run, self.config, self.meter = Path(run), dict(config), meter
        xml = self.run / "source/assets/walker2d.xml"
        if config.get("xml_sha256") is not None and sha(xml) != config["xml_sha256"]:
            raise ValueError("J1 frozen Walker XML changed")
        self.env = paid(
            meter,
            {"environment_constructions": 1},
            lambda: Walker2d(**config["task"], use_warp=False),
            "walker_native_cpu",
        )
        if (self.env.model.nq, self.env.model.nv, self.env.model.nu) != (9, 9, 6):
            raise ValueError("J1 native Walker dimensions differ")
        original = self.env.batch_rollout

        def batch(state: Any, actions: Any, **kwargs: Any) -> Any:
            values = np.asarray(actions)
            if values.ndim != 3 or values.shape[2] != 6:
                raise ValueError("J1 model action shape differs")
            controls = int(values.shape[0] * values.shape[1])
            return paid(
                meter,
                {"model_controls": controls, "physics_substeps": 4 * controls, "model_queries": 1},
                lambda: original(state, actions, **kwargs),
                "native_batch_rollout",
            )

        self.env.batch_rollout = batch
        self.recorder = C1TraceRecorder(meter, meter.out.parent / "active")

    def snapshot(self) -> dict:
        from .runtime import snapshot

        return paid(self.meter, {"snapshot_reads": 1}, lambda: snapshot(self.env), "full_snapshot")

    def restore(self, snapshot: Mapping[str, Any]) -> dict:
        from .runtime import restore

        return paid(
            self.meter,
            {"restores": 1, "external_mj_forward_calls": 1, "snapshot_reads": 1},
            lambda: restore(self.env, snapshot),
            "byte_exact_restore",
        )

    def reset(self, seed: int) -> dict:
        def operation() -> None:
            original = np.random.get_state()
            try:
                np.random.seed(seed)
                self.env.reset()
            finally:
                np.random.set_state(original)

        paid(self.meter, {"resets": 1, "external_mj_forward_calls": 2}, operation, "seeded_reset")
        return self.snapshot()

    def contacts(self, env: Any = None) -> dict:
        target = self.env if env is None else env
        present, forces = foot_contacts(target.model, target.data, self.meter)
        return dict(presence=present, normal_forces=forces)

    def trace(
        self,
        tag: str,
        controller: Callable[[np.ndarray, int, Mapping[str, Any]], Any],
        *,
        seed: int | None = None,
        initialsnapshot: Mapping[str, Any] | None = None,
        replay_source: Mapping[str, Any] | None = None,
        horizon: int = 1000,
    ) -> dict:
        if (seed is None) == (initialsnapshot is None):
            raise ValueError("Exactly one seed or full initial snapshot is required")
        current = self.reset(seed) if seed is not None else self.restore(initialsnapshot)
        observation = self.env._get_obs().copy()
        self.recorder.begin(tag, horizon, current, observation)
        contacts, forces, costs, velocities, terminals = [], [], [], [], []
        done = False

        def compare(position: int, cost: float | None = None) -> None:
            if replay_source is None:
                return
            for key, source in (
                ("state", "states"),
                ("warmstart", "warmstarts"),
                ("integration", "integrations"),
            ):
                expected = np.asarray(replay_source[source])[position]
                if not np.allclose(current[key], expected, rtol=0, atol=1e-10):
                    raise ValueError("J1 teacher replay differs: " + source)
            if not np.allclose(
                observation, replay_source["observations"][position], rtol=0, atol=1e-10
            ):
                raise ValueError("J1 teacher replay observation differs")
            if cost is not None:
                expected = replay_source.get("native_costs", replay_source.get("costs"))
                if not np.isclose(cost, expected[position - 1], rtol=0, atol=1e-10):
                    raise ValueError("J1 teacher replay native cost differs")

        try:
            compare(0)
            for index in range(horizon):
                action = np.asarray(controller(observation.copy(), index, current))
                if (
                    action.shape != (6,)
                    or action.dtype.kind != "f"
                    or not np.isfinite(action).all()
                    or np.any(np.abs(action) > 1)
                ):
                    raise ValueError("J1 direct controller returned invalid action")
                action = action.astype(np.float64)
                if (
                    replay_source is not None
                    and action.tobytes()
                    != np.asarray(replay_source["actions"][index], dtype=np.float64).tobytes()
                ):
                    raise ValueError("J1 teacher replay action differs")
                self.meter.authorize(
                    {"native_controls": 1, "physics_substeps": 4, "live_queries": 1}
                )
                self.recorder.intent(action, snapshot=current, observation=observation)
                observation, cost, done, info = paid(
                    self.meter,
                    {"native_controls": 1, "physics_substeps": 4, "live_queries": 1},
                    lambda: self.env.step(action),
                    "direct_native_control",
                )
                current = self.snapshot()
                try:
                    self.recorder.completed(
                        current,
                        observation,
                        cost=float(cost),
                        velocity=float(info["x_velocity"]),
                        fall=float(info["fall_cost"]),
                        done=bool(done),
                    )
                except BaseException:
                    for key, amount in (("native_controls", 1), ("physics_substeps", 4)):
                        self.meter.returned_unpersisted[key] = (
                            self.meter.returned_unpersisted.get(key, 0) + amount
                        )
                    raise
                present, normal = foot_contacts(self.env.model, self.env.data, self.meter)
                if replay_source is not None and "contacts" in replay_source:
                    if not np.array_equal(present, replay_source["contacts"][index]):
                        raise ValueError("J1 teacher replay contact presence differs")
                    if not np.allclose(
                        normal, replay_source["contact_normal_forces"][index], rtol=0, atol=1e-10
                    ):
                        raise ValueError("J1 teacher replay contact force differs")
                contacts.append(present)
                forces.append(normal)
                costs.append(float(cost))
                velocities.append(float(info["x_velocity"]))
                terminals.append(float(info["fall_cost"]))
                compare(index + 1, float(cost))
                if replay_source is not None and bool(done) != bool(replay_source["done"][index]):
                    raise ValueError("J1 teacher replay termination differs")
                if done:
                    break
            tail = 5.0 * (horizon - len(costs)) if done else 0.0
            trace = self.recorder.close(
                tail,
                dict(
                    study="J1",
                    measurement="actual_native_execution",
                    integration_spec=int(current["spec"]),
                    dimensions=np.asarray(current["dimensions"]).tolist(),
                ),
            )
        except BaseException as error:
            self.recorder.fail(error, meta=dict(study="J1"))
            raise
        contact_trace = self.meter.save_arrays(
            tag + "_contacts",
            contacts=np.asarray(contacts, dtype=bool),
            contact_normal_forces=np.asarray(forces, dtype=np.float64),
            control_ordinals=np.arange(len(costs), dtype=np.int64),
        )
        native = float(np.sum(costs, dtype=np.float64))
        terminal = float(np.sum(terminals, dtype=np.float64))
        mae = float(np.mean(np.abs(np.asarray(velocities) - 1.5)))
        survived = len(costs) == horizon and not done
        return dict(
            tag=tag,
            trace=trace,
            contact_trace=contact_trace,
            controls=len(costs),
            horizon=horizon,
            done=bool(done),
            native_cost=native,
            raw_cost=native - terminal,
            native_terminal=terminal,
            absorbing_cost=tail,
            cost=native + tail,
            mae=mae,
            survived=survived,
            walking=bool(horizon == 1000 and survived and mae <= 0.3),
            seed=seed,
            replay_verified=replay_source is not None,
        )

    def load_trace(self, outcome: Mapping[str, Any]) -> dict[str, Any]:
        result = {
            name: read_array(self.meter.out, outcome["trace"], name, max_bytes=16 * 1024**2).copy()
            for name in outcome["trace"]["arrays"]
        }
        if outcome.get("contact_trace") is not None:
            result.update(
                {
                    name: read_array(
                        self.meter.out, outcome["contact_trace"], name, max_bytes=16 * 1024**2
                    ).copy()
                    for name in outcome["contact_trace"]["arrays"]
                    if name != "control_ordinals"
                }
            )
        metadata = json.loads(str(result["metadata"].item()))
        result["metadata"] = metadata
        result["integration_spec"] = np.array(metadata["integration_spec"], dtype=np.int64)
        result["spec"] = result["integration_spec"].copy()
        result["dimensions"] = np.asarray(metadata["dimensions"], dtype=np.int64)
        return result

    def planner_context(self, outcome: Mapping[str, Any], index: int) -> dict:
        source = self.load_trace(outcome)
        if not 0 <= index < len(source["actions"]):
            raise ValueError("The declared planner anchor is not available")
        ref = outcome["planner_trace"]
        arrays = {
            name: read_array(self.meter.out, ref, name, max_bytes=16 * 1024**2)
            for name in ref["arrays"]
        }

        def private(suffix: str) -> dict:
            return dict(
                U=arrays["nominals_" + suffix][index].copy(),
                rng_state=json.loads(str(arrays["rng_" + suffix][index])),
                initial_plan=bool(arrays["initial_" + suffix][index]),
            )

        return dict(
            snapshot=dict(
                state=source["states"][index].copy(),
                warmstart=source["warmstarts"][index].copy(),
                integration=source["integrations"][index].copy(),
                spec=source["spec"].copy(),
                dimensions=source["dimensions"].copy(),
            ),
            planner=private("before"),
            original_action=source["actions"][index].copy(),
            original_selected_plan=arrays["selected_plans"][index].copy(),
            original_planner_after=private("after"),
            original_bank_digest=str(arrays["bank_digests"][index]),
            original_selected_task_cost=float(arrays["selected_task_costs"][index]),
            source_tag=outcome["tag"],
            source_control=index,
        )

    def persist_probe(self, probe: Mapping[str, Any]) -> dict:
        """Keep all pass arrays while retaining a small in-memory summary."""
        result = copy.deepcopy(
            {key: value for key, value in probe.items() if key not in {"passes", "snapshot"}}
        )
        tag = probe["request"]["id"]
        result["context_ref"] = self.meter.save_arrays(
            tag + "_context",
            **probe["snapshot"],
            observation=probe["observation"],
            action=probe["action"],
            selected_plan=probe["selected_plan"],
            planner_before=probe["planner_before"],
            planner_after=probe["planner_after"],
        )
        result["pass_refs"] = [
            self.meter.save_arrays(tag + f"_pass_{index:02d}", **row)
            for index, row in enumerate(probe["passes"])
        ]
        return result

    def teacher(self, tag: str, seed: int, planner_seed: int) -> dict:
        from src.utils.config import MPPIConfig

        from .search import CoupledSelector, IndependentMPPI

        planner = IndependentMPPI(self.env, MPPIConfig(**self.config["mppi"]), planner_seed)
        selector = CoupledSelector(alpha=0.0, eta_u=0.01)
        before, after, selected, decisions = [], [], [], []

        def controller(observation: Any, ordinal: int, current: Mapping[str, Any]) -> np.ndarray:
            before.append(planner.private_state())
            bank = planner.propose(current["state"], current["warmstart"])
            # The selector's numerical path is identical with no logging meter.
            # Its ordinary meter path archives every bank (~5 MiB/decision),
            # which belongs only to the explicitly bounded repeated probes.
            # Model work remains paid by the environment's batch wrapper.
            plan, info = selector.select(bank, actor=None, meter=None, enabled=False)
            after.append(planner.private_state())
            selected.append(plan.copy())
            decisions.append(info)
            return plan[0]

        outcome = self.trace(tag, controller, seed=seed)
        outcome["planner_trace"] = self.meter.save_arrays(
            tag + "_planner",
            nominals_before=np.asarray([v["U"] for v in before]),
            nominals_after=np.asarray([v["U"] for v in after]),
            rng_before=np.asarray([json.dumps(v["rng_state"], sort_keys=True) for v in before]),
            rng_after=np.asarray([json.dumps(v["rng_state"], sort_keys=True) for v in after]),
            initial_before=np.asarray([v["initial_plan"] for v in before], dtype=bool),
            initial_after=np.asarray([v["initial_plan"] for v in after], dtype=bool),
            selected_plans=np.asarray(selected),
            selected_actions=np.asarray(selected)[:, 0],
            control_ordinals=np.arange(len(selected), dtype=np.int64),
            bank_digests=np.asarray([v["bank_digest"] for v in decisions], dtype="U64"),
            raw_scores=np.asarray([v["raw_score"] for v in decisions], dtype=np.float64),
            selected_task_costs=np.asarray(
                [v["selected_task_cost"] for v in decisions], dtype=np.float64
            ),
            selected_scores=np.asarray([v["selected_score"] for v in decisions], dtype=np.float64),
            selected_origins=np.asarray([v["selected_origin"] for v in decisions]),
        )
        return outcome

    def close(self) -> None:
        self.env.close()


def _relative_file(meter: JMeter, path: Path) -> str:
    root = meter.out.parent.parent.resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError("J1 files must remain inside their run")
    return str(path.resolve().relative_to(root))


def _array_binding(path: Path, array: np.ndarray, meter: JMeter) -> dict:
    return dict(
        file=_relative_file(meter, path),
        bytes=path.stat().st_size,
        sha256=sha(path),
        array_sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        dtype=str(array.dtype),
        shape=list(array.shape),
    )


def parent_prepare(
    seed: int, fitseed: int, norm: Mapping[str, Any], out: Path, meter: JMeter
) -> dict:
    import torch

    from .policy import FIT_SCHEMA, Policy

    if torch.get_num_threads() != 1:
        raise ValueError("J1 fresh parents require one Torch thread")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    global_state = torch.get_rng_state().clone()
    actor = paid(
        meter,
        {"policy_constructions": 1},
        lambda: Policy(seed=seed, device="cpu", normalization=norm, obs_dim=17, hidden_dim=128),
        "fresh_width128_parent",
    )
    optimizer = paid(
        meter,
        {"optimizer_constructions": 1},
        lambda: torch.optim.Adam(
            actor.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
            fused=False,
        ),
        "empty_adam0",
    )
    config = dict(
        batch_size=256,
        lr=3e-4,
        eta_theta=0.0,
        gradient_clip=0.5,
        prox_reduction="sum",
        anchor_refresh=0,
    )
    package = dict(
        schema=FIT_SCHEMA,
        adam=copy.deepcopy(optimizer.state_dict()),
        rng=copy.deepcopy(np.random.PCG64(fitseed).state),
        updates=0,
        fit_updates=0,
        policy_hash=actor.hash(),
        parent_policy_hash=actor.hash(),
        anchor_policy_hash=actor.hash(),
        anchor={name: value.detach().cpu().clone() for name, value in actor.state_dict().items()},
        parameter_names=[name for name, _ in actor.named_parameters()],
        fit_config=config,
        fit_data=dict(observations_sha256=None, actions_sha256=None, rows=0),
    )
    generator = np.random.Generator(np.random.PCG64(fitseed))
    before = []
    after = []
    indices = []
    for _ in range(2000):
        before.append(copy.deepcopy(generator.bit_generator.state))
        indices.append(generator.integers(0, 1000, size=256))
        after.append(copy.deepcopy(generator.bit_generator.state))
    tape = np.asarray(indices, dtype=np.int64)
    meter.reserve_external(tape.nbytes + 2 * 1024**2)
    with (out / "minibatches.npy").open("xb") as stream:
        np.save(stream, tape, allow_pickle=False)
    with (out / "global_torch_rng.npy").open("xb") as stream:
        np.save(stream, global_state.numpy(), allow_pickle=False)
    checkpoint = actor.checkpoint(
        out / "incoming.pt",
        optimizer_state=package,
        metadata=dict(study="J1", fresh_parent=True, updates=0),
    )
    checkpoint["file"] = _relative_file(meter, out / "incoming.pt")
    if not torch.equal(torch.get_rng_state(), global_state):
        raise ValueError("Parent preparation changed global Torch RNG")
    bindings = dict(
        schema="walker-j1-incoming-package-v1",
        seed=seed,
        fitseed=fitseed,
        checkpoint=checkpoint,
        actor_hash=actor.hash(),
        optimizer_fingerprint=fingerprint(package),
        private_rng_fingerprint=fingerprint(actor._generator.get_state()),
        global_rng_fingerprint=fingerprint(global_state),
        tape=_array_binding(out / "minibatches.npy", tape, meter),
        global_cpu_rng=_array_binding(out / "global_torch_rng.npy", global_state.numpy(), meter),
        initial_pcg64=package["rng"],
        terminal_pcg64=copy.deepcopy(generator.bit_generator.state),
        updates=2000,
        batch_size=256,
        unique_rows=1000,
        adam_state_empty=not bool(package["adam"]["state"]),
    )
    write(out / "bindings.json", bindings)
    return dict(
        actor=actor,
        package=package,
        bindings=bindings,
        tape=tape,
        rng_before=before,
        rng_after=after,
        global_torch=global_state,
    )


def fit_case(
    parent: Mapping[str, Any], observations: Any, actions: Any, out: Path, meter: JMeter
) -> dict:
    import torch

    from .policy import fit_policy

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    x = np.asarray(observations)
    y = np.asarray(actions)
    if x.shape != (1000, 17) or y.shape != (1000, 6):
        raise ValueError("Every J1 learning case requires1000unique executed rows")
    actor, package = parent["actor"], parent["package"]
    owner_hash = actor.hash()
    owner_optimizer = fingerprint(package)
    if fingerprint(torch.get_rng_state()) != fingerprint(parent["global_torch"]):
        raise ValueError("Caller did not restore the matched global Torch RNG")
    actual = []

    def initial(value: Mapping[str, Any]) -> None:
        if (
            fingerprint(value["state_dict"]) != fingerprint(actor.state_dict())
            or fingerprint(value["adam"]) != fingerprint(package["adam"])
            or value["rng"] != package["rng"]
            or not torch.equal(value["private_rng_state"], actor._generator.get_state())
        ):
            raise ValueError("Initial fit tensors/Adam/RNG differ from incoming package")

    def indices(value: Mapping[str, Any]) -> None:
        index = len(actual)
        if (
            index >= 2000
            or value["update"] != index + 1
            or value["fit_update"] != index + 1
            or not np.array_equal(value["indices"], parent["tape"][index])
            or value["rng_before"] != parent["rng_before"][index]
            or value["rng_after"] != parent["rng_after"][index]
        ):
            raise ValueError("Fitting minibatch/RNG differs from its presealed tape")
        actual.append(
            hashlib.sha256(np.asarray(value["indices"], dtype=np.int64).tobytes()).hexdigest()
        )

    meter.reserve_external(4 * 1024**2)

    def execute_fit() -> Any:
        try:
            return fit_policy(
                actor,
                x,
                y,
                optimizer_state=package,
                updates=2000,
                batch_size=256,
                lr=3e-4,
                eta_theta=0,
                seed=parent["bindings"]["fitseed"],
                meter=meter,
                tag=out.name,
                resume_fit=False,
                prox_reduction="sum",
                anchor_refresh=0,
                initial_observer=initial,
                index_observer=indices,
            )
        except BaseException as error:
            write(
                out / "fit_failure.json",
                dict(
                    schema="walker-j1-fit-failure-v1",
                    error_type=type(error).__name__,
                    message=str(error),
                    fit_evidence=getattr(error, "fit_evidence", None),
                    observed_minibatch_hashes=actual,
                    child_checkpoint_available=False,
                    parent_actor_hash=owner_hash,
                    parent_optimizer_fingerprint=owner_optimizer,
                ),
            )
            raise

    child, state, evidence = paid(
        meter,
        {"fits": 1, "policy_constructions": 1, "optimizer_constructions": 1},
        execute_fit,
        "fixed2000update_case",
    )
    if (
        len(actual) != 2000
        or state["rng"] != parent["bindings"]["terminal_pcg64"]
        or state["updates"] != 2000
        or state["fit_updates"] != 2000
        or actor.hash() != owner_hash
        or fingerprint(package) != owner_optimizer
    ):
        raise ValueError("Fitting continuation/tape/parent ownership differs")
    if not torch.equal(torch.get_rng_state(), parent["global_torch"]):
        raise ValueError("Fitting changed the matched global Torch RNG")
    meter.reserve_external(4 * 1024**2)
    checkpoint = child.checkpoint(
        out / "child.pt",
        optimizer_state=state,
        metadata=dict(study="J1", parent_hash=owner_hash, updates=2000),
    )
    checkpoint["file"] = _relative_file(meter, out / "child.pt")
    write(out / "learning_curve.json", evidence)
    receipt = dict(
        schema="walker-j1-learning-case-fit-v1",
        incoming_actor_hash=owner_hash,
        incoming_optimizer_fingerprint=owner_optimizer,
        child_actor_hash=child.hash(),
        child_optimizer_fingerprint=fingerprint(state),
        checkpoint=checkpoint,
        updates=2000,
        tape_sha256=parent["bindings"]["tape"]["sha256"],
        observed_minibatch_hashes=actual,
        all2000minibatches_and_rng_matched=True,
        parent_unchanged=True,
        global_rng_unchanged=True,
    )
    write(out / "fit_receipt.json", receipt)
    return dict(
        actor=child,
        package=state,
        checkpoint=checkpoint,
        curve=evidence,
        receipt=receipt,
        receipt_file=_relative_file(meter, out / "fit_receipt.json"),
        learning_curve_file=_relative_file(meter, out / "learning_curve.json"),
    )
