"""H5.0 native adapter; imported only after the frozen launch gate passes.

No policy, simulator or worker is constructed on import. The public worker
rechecks authorization, source/input ownership and runtime before paid work.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .proposal_storage import CONTRACT, check_native_dimensions
from .provenance import fingerprint, sha, write

ARM_CODES = dict(independent="I", selector_only="S", proposal_guided="P")


def reference(arrays: dict[str, Any], key: str) -> dict[str, Any]:
    return dict(reference=arrays, key=key)


class NativeOperations:
    def __init__(self, out: Path, config: dict[str, Any], meter: Any):
        import mujoco
        import torch

        from src.envs.walker2d import Walker2d

        from .policy import Policy

        self.out, self.config, self.meter = out, config, meter
        self.torch, self.Policy = torch, Policy
        torch.set_num_threads(config["policy"]["torch_threads"])
        torch.manual_seed(config["learning"]["global_torch_seed"])
        self.global_rng = torch.random.get_rng_state().clone()
        if self.global_rng.numel() > CONTRACT["global_rng_bytes"]:
            raise ValueError("Global Torch RNG sidecar exceeds its frozen byte contract")
        self.actor_registry: dict[str, Any] = {}
        self.checkpoint_records: list[dict[str, Any]] = []
        self.fit_records: list[dict[str, Any]] = []
        self._serial = 0
        self.datasets = {}
        for label in ("train", "validation", "normalization"):
            with np.load(out / config["inputs"][label]["file"], allow_pickle=False) as values:
                self.datasets[label] = {key: values[key].copy() for key in values.files}
        self.normalization = self.datasets["normalization"]
        self.normalization_hash = fingerprint(self.normalization)
        task_owner = json.loads((out / config["inputs"]["task"]["file"]).read_text())
        if task_owner != dict(task=config["task"], xml_sha256=config["xml_sha256"]):
            raise ValueError("Native task differs from the exact W2 task/input binding")
        xml = out / "source/assets/walker2d.xml"
        if sha(xml) != config["xml_sha256"]:
            raise ValueError("Walker model XML binding changed")
        meter.record("native_construction_requested", model_constructions=1, mjdata_objects=17)
        self.env = Walker2d(**config["task"], use_warp=False)
        try:
            size = mujoco.mj_stateSize(self.env.model, mujoco.mjtState.mjSTATE_INTEGRATION)
            if (self.env.model.nq, self.env.model.nv, self.env.model.nu) != (9, 9, 6):
                raise ValueError("Native joint/actuator dimensions changed")
            check_native_dimensions(
                state=self.env.state_dim,
                action=self.env.action_dim,
                integration=int(size),
                sensors=self.env.model.nsensordata,
            )
            meter.record(
                "native_construction_complete",
                integration_size=size,
                native_threads=config["task"]["nthread"],
                torch_threads=torch.get_num_threads(),
            )
            meter.instrument(self.env)
        except BaseException:
            self.env.close()
            raise

    def _context(self, **values: Any) -> None:
        self.meter.context = dict(phase="worker", **values)

    def _global_unchanged(self) -> None:
        if not self.torch.equal(self.global_rng, self.torch.random.get_rng_state()):
            raise ValueError("An undeclared operation consumed the global Torch CPU RNG")

    def _checkpoint(self, actor: Any, state: Any, name: str, fitting_evidence=None, tape=None):
        path = self.out / "checkpoints" / (name + ".pt")
        self.meter.reserve_external(2 * 1024**2)
        descriptor = actor.checkpoint(path, optimizer_state=state, metadata=dict(name=name))
        rng = self.meter.save_arrays(name + "_global_torch_rng", state=self.global_rng.numpy())
        record = dict(
            file=str(path.relative_to(self.out)),
            sha256=descriptor["sha256"],
            bytes=descriptor["bytes"],
            global_rng=reference(rng, "state"),
            optimizer_sha256=fingerprint(state),
            fitting_evidence=fitting_evidence,
            fit_tape=tape,
        )
        self.actor_registry[actor.hash()] = record
        self.checkpoint_records.append(dict(actor_hash=actor.hash(), name=name, **record))
        self.meter.record("checkpoint", actor_hash=actor.hash(), **record)
        self._global_unchanged()
        return path

    def _reload(self, path: Path):
        actor, state, _ = self.meter.measure(
            "checkpoint_loads",
            1,
            lambda: self.Policy.from_checkpoint(path, device="cpu"),
            purpose="canonical_fixed_endpoint_reload",
        )
        self._global_unchanged()
        return actor, state

    def _metric(self, actor: Any, x: np.ndarray, y: np.ndarray, tag: str, xref, yref):
        from .policy import full_dataset_mse

        value = full_dataset_mse(
            actor,
            x,
            y,
            meter=self.meter,
            batch_size=self.config["learning"]["prediction_batch_size"],
        )
        self.meter.record(
            "policy_metric_complete",
            actor_hash=actor.hash(),
            tag=tag,
            observations=xref,
            targets=yref,
            rows=len(x),
            batch_size=1024,
            action_mse=value,
        )
        return value

    def _historical_metrics(self, actor: Any, tag: str):
        result = {}
        for label in ("train", "validation"):
            data = self.datasets[label]
            path = self.config["inputs"][label]["file"]
            result[label] = self._metric(
                actor,
                data["observations"],
                data["actions"],
                tag + "/" + label,
                dict(file=path, key="observations"),
                dict(file=path, key="actions"),
            )
        return result

    def _fit(self, parent: Any, incoming: Any, data: dict, seed: int, tag: str, updates: int):
        from .policy import fit_policy

        owner = (
            parent.hash(),
            fingerprint(incoming),
            fingerprint(parent.normalization),
            fingerprint(parent._generator.get_state()),
        )
        self.meter.reserve_external(8 * 1024**2)
        initial = {}
        actual_indices = []
        rng = np.random.Generator(np.random.PCG64(seed))
        if incoming is not None:
            rng.bit_generator.state = copy.deepcopy(incoming["rng"])
        initial_pcg64 = copy.deepcopy(rng.bit_generator.state)
        tape = np.stack([rng.integers(0, len(data["actions"]), size=256) for _ in range(updates)])
        tape_ref = self.meter.save_arrays(tag + "_expected_minibatch_tape", indices=tape)

        def observe_initial(value):
            for group in value["adam"]["param_groups"]:
                for key, expected in self.config["learning"]["adam"].items():
                    actual = list(group[key]) if isinstance(group[key], tuple) else group[key]
                    if actual != expected:
                        raise ValueError("Observed Adam setting differs before fitting: " + key)
            if fingerprint(value["state_dict"]) != fingerprint(parent.state_dict()):
                raise ValueError("Observed child tensors differ from the exact incoming parent")
            if not self.torch.equal(value["private_rng_state"], parent._generator.get_state()):
                raise ValueError("Observed child private Torch RNG differs from its parent")
            if value["rng"] != initial_pcg64:
                raise ValueError("Observed minibatch generator differs before the first update")
            if incoming is not None and (
                fingerprint(value["adam"]) != fingerprint(incoming["adam"])
                or value["rng"] != incoming["rng"]
            ):
                raise ValueError(
                    "Observed child Adam/private PCG64 differs before its first update"
                )
            initial.update(value)
            self.meter.record(
                "fit_initial_state",
                tag=tag,
                parent_hash=parent.hash(),
                policy_state_sha256=fingerprint(value["state_dict"]),
                adam_sha256=fingerprint(value["adam"]),
                rng=value["rng"],
                private_rng_sha256=fingerprint(value["private_rng_state"]),
                expected_tape=tape_ref,
                **self.meter._fields({}),
            )

        def update_observer(row):
            actual_indices.append(row["indices"].copy())
            self.meter.record("fit_update", tag=tag, **row, **self.meter._fields({}))

        try:
            actor, state, evidence = fit_policy(
                parent,
                data["observations"],
                data["actions"],
                optimizer_state=copy.deepcopy(incoming),
                updates=updates,
                batch_size=256,
                lr=0.0003,
                eta_theta=0,
                seed=seed,
                prox_reduction="mean",
                anchor_refresh=0,
                resume_fit=False,
                meter=self.meter,
                tag=tag,
                initial_observer=observe_initial,
                update_observer=update_observer,
            )
        except BaseException as error:
            if hasattr(error, "fit_evidence"):
                error.fit_evidence.update(
                    sampled_indices=tape_ref,
                    expected_terminal_rng=rng.bit_generator.state,
                    incoming_optimizer_sha256=owner[1],
                )
                write(self.out / "learning" / (tag + "_partial.json"), error.fit_evidence)
            raise
        if owner != (
            parent.hash(),
            fingerprint(incoming),
            fingerprint(parent.normalization),
            fingerprint(parent._generator.get_state()),
        ):
            raise ValueError("Fitting mutated the common incoming policy/optimizer/normalization")
        if fingerprint(actor.normalization) != self.normalization_hash:
            raise ValueError("Training normalization was refitted or changed")
        if incoming is not None and (
            fingerprint(initial["adam"]) != fingerprint(incoming["adam"])
            or initial["rng"] != incoming["rng"]
            or not self.torch.equal(initial["private_rng_state"], parent._generator.get_state())
        ):
            raise ValueError("The child did not inherit exact Adam/private RNG ownership")
        if not np.array_equal(tape, np.asarray(actual_indices)):
            raise ValueError("Observed fitting indices differ from the declared private stream")
        if state["rng"] != rng.bit_generator.state:
            raise ValueError("Fitting did not consume the exact declared minibatch stream")
        if state["updates"] != (0 if incoming is None else incoming["updates"]) + updates:
            raise ValueError("Cumulative Adam update count changed")
        if state["fit_updates"] != updates or state["anchor_policy_hash"] != parent.hash():
            raise ValueError("Changed data must begin a new block at the incoming parent")
        evidence.update(
            incoming_optimizer_sha256=owner[1],
            expected_terminal_rng=state["rng"],
            sampled_indices=tape_ref,
            initial_adam_sha256=fingerprint(initial["adam"]),
            initial_private_rng_sha256=fingerprint(initial["private_rng_state"]),
            initial_policy_state_sha256=fingerprint(initial["state_dict"]),
        )
        name = "learning/" + tag + ".json"
        write(self.out / name, evidence)
        checkpoint = self._checkpoint(actor, state, tag, name, tape_ref)
        loaded, loaded_state = self._reload(checkpoint)
        if loaded.hash() != actor.hash() or fingerprint(loaded_state) != fingerprint(state):
            raise ValueError("Checkpoint reload changed the complete endpoint")
        self.fit_records.append(
            dict(
                tag=tag,
                file=name,
                policy_hash=loaded.hash(),
                checkpoint=copy.deepcopy(self.actor_registry[loaded.hash()]),
            )
        )
        return dict(
            actor=loaded,
            state=loaded_state,
            owner=fingerprint(loaded_state),
            private_rng_owner=fingerprint(loaded._generator.get_state()),
            stream=self.meter.context.get("stream"),
            arm=self.meter.context.get("arm"),
        )

    def parent(self, stream):
        tag = f"s{stream['id']}_parent"
        self._context(role="parent_fit", stream=stream["id"])
        actor = self.Policy(
            stream["policy_seed"],
            device="cpu",
            normalization=self.normalization,
            obs_dim=17,
            hidden_dim=128,
        )
        self._checkpoint(actor, None, tag + "_initial")
        self._historical_metrics(actor, tag + "_initial")
        parent = self._fit(
            actor,
            None,
            self.datasets["train"],
            stream["fit_seed"],
            tag,
            self.config["learning"]["parent_updates"],
        )
        self._historical_metrics(parent["actor"], tag)
        return parent

    def _teacher(self, parent, stream, arm, index, gap=False):
        from src.utils.config import MPPIConfig

        from .guided_proposals import TwoStreamMPPI
        from .runtime import execute_episode

        prefix = "gap_" if gap else ""
        seed = stream["gap_seeds" if gap else "acquisition_seeds"][index]
        controller = TwoStreamMPPI(
            self.env,
            MPPIConfig(**self.config["mppi"]),
            stream[prefix + "protected_planner_seeds"][index],
            stream["gap_second_planner_seeds" if gap else "second_planner_seeds"][index],
            alpha=self.config["guidance"]["alpha"],
            eta_u=self.config["guidance"]["eta_u"],
        )
        actor, owner = parent["actor"], parent["owner"]
        tag = f"{prefix}teacher_s{stream['id']}_{arm}_{index:02d}"
        self._context(role="teacher", stream=stream["id"], arm=arm, actor_hash=actor.hash())

        def action(state, warm, obs, step):
            plan, info = controller.decide(
                state,
                actor,
                self.meter,
                arm=ARM_CODES[arm],
                warmstart=warm,
            )
            # Full array descriptors are already in the selection ledger/index.
            # Keep a resolvable compact pointer in each episode's controller info.
            pointer = {key: info["evidence"][key] for key in ("file", "group", "tag")}
            return plan[0], dict(policy_hash=actor.hash(), arm=arm, selection=pointer)

        episode = self.meter.run_episode(
            lambda: execute_episode(self.env, action, seed, 1000, self.meter, tag)
        )
        prediction = actor.predict(episode.observations, meter=self.meter)
        error = np.sum((episode.actions - prediction) ** 2, axis=1)
        saved = self.meter.save_arrays(
            tag + "_agreement",
            policy_actions=prediction,
            squared_disagreement=error,
        )
        self.meter.record(
            "teacher_agreement_complete",
            actor_hash=actor.hash(),
            episode_raw=episode.raw,
            predictions=reference(saved, "policy_actions"),
            squared_disagreement=reference(saved, "squared_disagreement"),
            rows=episode.steps,
            action_agreement=float(np.mean(error)),
        )
        if (
            fingerprint(parent["state"]) != owner
            or actor.hash() != parent["state"]["policy_hash"]
            or fingerprint(actor._generator.get_state()) != parent["private_rng_owner"]
        ):
            raise ValueError("Acquisition mutated the incoming learner")
        self._global_unchanged()
        return episode

    def acquire(self, parent, stream, arm, index):
        return self._teacher(parent, stream, arm, index)

    def gap_teacher(self, parent, stream, arm, index):
        return self._teacher(parent, stream, arm, index, gap=True)

    def fit(self, parent, episodes, stream, arm):
        tag = f"s{stream['id']}_{arm}_child"
        self._context(role="child_fit", stream=stream["id"], arm=arm)
        data = {
            key: np.concatenate([getattr(episode, key) for episode in episodes])
            for key in ("observations", "actions")
        }
        if len(data["actions"]) != 4000:
            raise ValueError("The fixed child dataset must contain all4,000 unique executed rows")
        if any(len(episode.tag) > CONTRACT["child_row_tag_characters"] for episode in episodes):
            raise ValueError("Teacher row identity exceeds its declared serialization bound")
        saved = self.meter.save_arrays(
            tag + "_data",
            **data,
            row_episode=np.repeat([episode.tag for episode in episodes], 1000),
            row_step=np.tile(np.arange(1000), 4),
        )
        xref, yref = reference(saved, "observations"), reference(saved, "actions")
        self._metric(parent["actor"], *data.values(), tag + "/incoming", xref, yref)
        child = self._fit(
            parent["actor"],
            parent["state"],
            data,
            stream["fit_seed"],
            tag,
            self.config["learning"]["child_updates"],
        )
        self._metric(child["actor"], *data.values(), tag + "/endpoint", xref, yref)
        self._historical_metrics(child["actor"], tag)
        return child

    def evaluate(self, actor, seed, tag):
        from .runtime import execute_episode

        arm = next((name for name in ARM_CODES if f"_{name}_" in tag), None)
        self._context(
            role="actor_only",
            cohort=tag.split("_", 1)[0],
            arm=arm,
            stream=actor["stream"],
            actor_hash=actor["actor"].hash(),
        )

        def action(state, warm, obs, step):
            return actor["actor"].predict(obs, meter=self.meter), dict(
                policy_hash=actor["actor"].hash()
            )

        episode = self.meter.run_episode(
            lambda: execute_episode(self.env, action, seed, 1000, self.meter, tag)
        )
        if (
            fingerprint(actor["state"]) != actor["owner"]
            or fingerprint(actor["actor"]._generator.get_state()) != actor["private_rng_owner"]
        ):
            raise ValueError("Standalone evaluation mutated the optimizer/RNG state")
        self._global_unchanged()
        return episode

    def paired(self, first, second):
        if first.seed != second.seed or any(
            not np.array_equal(getattr(first, name)[0], getattr(second, name)[0])
            for name in ("states", "warmstarts", "integrations", "observations")
        ):
            raise ValueError("Paired physical reset contexts differ")

    def record(self, event, **fields):
        self.meter.record(event, **fields)

    def close(self):
        self.env.close()


def worker(out: Path) -> int:
    """Native worker boundary. The launch coordinator installs the watchdog."""
    from .experiments.teacher_proposal import launch_blockers
    from .lifecycle import verify
    from .proposal_round import run_round

    verify(out)
    config = json.loads((out / "config.json").read_text())
    blockers = launch_blockers(config)
    if blockers:
        raise ValueError("H5.0 launch is blocked: " + "; ".join(blockers))
    ownership = json.loads((out / "launch_ownership.json").read_text())
    if (
        ownership["coordinator_pid"] != os.getppid()
        or ownership["nonce"] != os.environ.get("H5_LAUNCH_NONCE")
        or ownership["watchdog_wall_seconds"] != config["budgets"]["worker_wall_seconds"]
        or any(
            ownership[key] != sha(out / file)
            for key, file in (
                ("config_sha256", "config.json"),
                ("source_manifest_sha256", "manifest.json"),
                ("input_manifest_sha256", "input_manifest.json"),
            )
        )
    ):
        raise ValueError("Native worker lacks a live matching outer-watchdog owner")
    for name, binding in json.loads((out / "input_manifest.json").read_text())["files"].items():
        if sha(out / name) != binding["sha256"]:
            raise ValueError("Frozen input changed: " + name)
    # Kept lazy: unauthorized workers cannot import an inference/simulator stack.
    from .proposal_execution import NativeMeter, verify_runtime

    verify_runtime(config)
    started = time.perf_counter()
    result, operations = {}, None
    with NativeMeter(out / "evidence", config) as meter:
        try:
            operations = NativeOperations(out, config, meter)
            result = run_round(config, operations)
        except BaseException as error:
            result.update(
                execution_complete=False,
                status="worker_error",
                error=dict(type=type(error).__name__, message=str(error)),
            )
            meter.record("h5_worker_error", error=result["error"])
            if operations is not None:
                from .runtime import snapshot

                result["failure_simulator_context"] = meter.save_arrays(
                    "worker_failure_context", **snapshot(operations.env)
                )
        finally:
            if operations is not None:
                result.update(
                    actor_registry=operations.actor_registry,
                    fit_records=operations.fit_records,
                    checkpoint_records=operations.checkpoint_records,
                    policy_hashes={
                        item["name"]: item["actor_hash"] for item in operations.checkpoint_records
                    },
                    parents={
                        item["name"]: item
                        for item in operations.checkpoint_records
                        if item["name"].endswith("_parent")
                    },
                )
                try:
                    operations.close()
                except BaseException as error:
                    result.update(
                        execution_complete=False,
                        confirmed_benefit=False,
                        status="worker_close_error",
                        close_error=dict(type=type(error).__name__, message=str(error)),
                    )
            result["work"] = meter.totals()
            result["reservation_census"] = meter.reservations.snapshot(include_events=False)
    result["wall_seconds"] = time.perf_counter() - started
    write(out / "worker_result.json", result)
    print(json.dumps(dict(status=result.get("status"), error=result.get("error"))), flush=True)
    return 0 if result.get("execution_complete") else 1
