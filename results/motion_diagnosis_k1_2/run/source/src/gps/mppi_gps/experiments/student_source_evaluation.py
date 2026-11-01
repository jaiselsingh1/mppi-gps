"""S1.1: fresh standalone evaluation of fixed S1 actors with closed episode files."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..durability import durable_json, file_binding, fsync_directory, read_boot_id, seal_segment
from ..evaluation_exchange import OwnerJournal, publish_and_wait
from ..lifecycle import ROOT, freeze, verify
from ..metrics import summary
from ..provenance import fingerprint, sha, write
from ..recipes import load_recipe
from ..runtime import BudgetExhausted, Meter
from ..source_outcomes import add_work, source_screen


def check_budget(config: dict, prior: dict, current: dict, kind: str, amount: int) -> None:
    keys = dict(
        neural_rows="neural_rows",
        checkpoint_loads="checkpoint_loads",
        live_controls="live_controls",
        physics_substeps="physics_substeps",
    )
    if kind not in keys or type(amount) is not int or amount < 0:
        raise ValueError("S1.1 forbids this work kind")
    counter = "requested_" + kind
    if prior.get(counter, 0) + current.get(counter, 0) + amount > config["budgets"][keys[kind]]:
        raise BudgetExhausted("S1.1 cumulative work exhausted before " + kind)


class EvaluationMeter(Meter):
    def __init__(self, out: Path, config: dict, prior: dict, deadline: float, phase: str):
        self.config, self.prior, self.deadline, self.phase = config, prior, deadline, phase
        self.final_work = None
        super().__init__(
            out,
            max_controls=config["budgets"]["live_controls"],
            max_neural_rows=config["budgets"]["neural_rows"],
        )
        run = out.parents[2] if phase == "episode" else out.parents[1]
        self.static_bytes = sum(
            p.stat().st_size for p in run.rglob("*") if p.is_file() and not p.is_relative_to(out)
        )

    def boundary(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExhausted("S1.1 global wall deadline exhausted")
        if (
            self.static_bytes
            + self._raw.id.get_filesize()
            + self.ledger_path.stat().st_size
            + 64 * 1024**2
            > self.config["budgets"]["storage_bytes"]
        ):
            raise BudgetExhausted("S1.1 global storage ceiling exhausted")

    def measure(self, kind, amount, fn, **metadata):
        self.boundary()
        if kind != ("checkpoint_loads" if self.phase == "loads" else "neural_rows"):
            raise ValueError("S1.1 phase forbids this measurement")
        check_budget(self.config, self.prior, self._counts, kind, amount)
        return super().measure(kind, amount, fn, **metadata)

    def _query(self, mode, controls, frame_skip, *args, **kwargs):
        self.boundary()
        if self.phase != "episode" or mode != "live" or controls != 1 or frame_skip != 4:
            raise ValueError("S1.1 permits only the fixed native CPU controller")
        check_budget(self.config, self.prior, self._counts, "live_controls", controls)
        check_budget(
            self.config, self.prior, self._counts, "physics_substeps", controls * frame_skip
        )
        return super()._query(mode, controls, frame_skip, *args, **kwargs)

    def close(self):
        if not self.closed:
            super().close()
            self.final_work = super().totals()

    def totals(self):
        return dict(self.final_work) if self.final_work is not None else super().totals()


def prepare(
    out: Path, source: Path, remote_freshness: Path, preflight_out: Path | None = None
) -> dict:
    from ..source_evaluation_inputs import audit_known_reset_ledgers, prepare_inputs
    from ..width_inputs import audit_reset_seeds

    if preflight_out is not None and preflight_out.resolve().is_relative_to(out.resolve()):
        raise ValueError("Preflight diagnostics must be outside the create-only frozen run")
    config = load_recipe("student_source_evaluation")
    config["source_run"] = str(source.resolve())
    seeds = config["primary_seeds"] + config["confirmation_seeds"]
    proof = audit_reset_seeds(ROOT, seeds)
    proof["known_reset_ledgers"] = audit_known_reset_ledgers(ROOT, seeds)
    proof["passed"] = proof["passed"] and proof["known_reset_ledgers"]["passed"]
    remote = json.loads(remote_freshness.read_text())
    if preflight_out is not None:
        write(
            preflight_out,
            dict(
                schema="walker-s1.1-preparation-preflight-v1",
                local=proof,
                remote=remote,
                remote_proof_binding=file_binding(remote_freshness),
                new_neural_rows=0,
                new_native_controls=0,
            ),
        )
    if (
        not proof["passed"]
        or not remote["passed"]
        or remote["seeds"] != seeds
        or not remote["known_reset_ledgers"]["passed"]
        or remote["known_reset_ledgers"]["seeds"] != seeds
    ):
        raise ValueError("S1.1 fresh-reset proof failed")
    if config["task"]["xml_sha256"] != sha(ROOT / "assets/walker2d.xml"):
        raise ValueError("S1.1 native XML changed")
    freeze(out, config, "student-source-evaluation-s1.1")
    prepare_inputs(source, out / "inputs", config)
    write(out / "inputs/local_seed_audit.json", proof)
    write(out / "inputs/remote_seed_audit.json", remote)
    write(
        out / "input_manifest.json",
        dict(
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            files={
                str(p.relative_to(out)): file_binding(p)
                for p in sorted((out / "inputs").rglob("*"))
                if p.is_file()
            },
            planned_work=config["budgets"],
        ),
    )
    return dict(
        prepared=str(out),
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
    )


def platform_identity(config: dict) -> dict:
    import torch

    expected = config["platform"]
    identity = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        **{key: importlib.metadata.version(key) for key in ("mujoco", "numpy", "scipy", "h5py")},
        torch=torch.__version__,
    )
    if (
        sys.platform != "linux"
        or identity["python"] != config["expected_python"]
        or any(
            identity[key] != expected[key] for key in ("torch", "mujoco", "numpy", "scipy", "h5py")
        )
        or torch.version.cuda is not None
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("S1.1 pinned CPU runtime differs")
    torch.set_num_threads(1)
    return identity


def verify_inputs(out: Path) -> tuple[dict, dict]:
    verify(out)
    config = json.loads((out / "config.json").read_text())
    seal = json.loads((out / "input_manifest.json").read_text())
    if (
        config["schema"] != "walker-student-source-s1.1-v1"
        or seal["config_sha256"] != sha(out / "config.json")
        or seal["source_manifest_sha256"] != sha(out / "manifest.json")
    ):
        raise ValueError("S1.1 input/source ownership differs")
    for name, binding in seal["files"].items():
        path = (out / name).resolve()
        if not path.is_relative_to(out.resolve()) or file_binding(path) != binding:
            raise ValueError("S1.1 sealed input changed: " + name)
    launch = json.loads((out / "launch_receipt.json").read_text())
    if (
        launch["config_sha256"] != sha(out / "config.json")
        or launch["input_manifest_sha256"] != sha(out / "input_manifest.json")
        or launch["boot_id"] != read_boot_id()
    ):
        raise ValueError("S1.1 owner/boot seal differs")
    return config, launch


def inherited_result(out: Path, config: dict) -> dict:
    fit_path = out / "inputs/s1/fit_result.json"
    fit = json.loads(fit_path.read_text())
    points, fits = deepcopy(fit["points"]), deepcopy(fit["fits"])
    for point in points.values():
        point["checkpoint"]["file"] = "inputs/s1/" + point["checkpoint"]["file"]
    for record in fits:
        record["file"] = "inputs/s1/" + record["file"]
        record["initial"]["file"] = "inputs/s1/" + record["initial"]["file"]
    old = json.loads((out / "inputs/s1_interruption/summary.json").read_text())
    combined = old["work"]["known_durable_prefix"]
    previous_eval = {
        key: value - fit["work"].get(key, 0)
        for key, value in combined.items()
        if isinstance(value, (int, float)) and key != "wall_seconds_lower_bound"
    }
    previous_eval.update(
        wall_seconds=None,
        snapshots=None,
        wall_seconds_lower_bound=old["durable_prefix"]["last_record"]["wall_seconds"],
    )
    return dict(
        schema="walker-student-source-s1.1-result-v1",
        execution_complete=False,
        status="evaluating",
        error=None,
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        inherited_fit=dict(
            file="inputs/s1/fit_result.json",
            sha256=sha(fit_path),
            source_bindings=config["source_bindings"],
        ),
        points=points,
        fits=fits,
        samplers=fit["samplers"],
        teacher_outcomes=fit["teacher_outcomes"],
        inherited_work=dict(
            original_fit=fit["work"],
            original_eval_known_prefix=previous_eval,
            original_eval_unknown_tail=old["work"]["unknown_unrecorded_tail"],
        ),
        evaluations=dict(primary={}, confirmation={}),
        gates={},
        segments=[],
        work={},
    )


def worker(out: Path) -> dict:
    from .. import policy, runtime
    from ..diagnostic_policies import LegacySACActor

    config, launch = verify_inputs(out)
    started = time.monotonic()
    deadline, boot = launch["deadline_monotonic"], launch["boot_id"]
    result = inherited_result(out, config)
    result["platform"] = platform_identity(config)
    durable_json(
        out / "worker_started.json",
        dict(
            schema="walker-s1.1-worker-start-v1",
            platform=result["platform"],
            boot_id=boot,
            pid=os.getpid(),
            launch_receipt_sha256=sha(out / "launch_receipt.json"),
            config_sha256=sha(out / "config.json"),
            input_manifest_sha256=sha(out / "input_manifest.json"),
        ),
    )
    journal = OwnerJournal(out / "owner_ledger.jsonl")
    cumulative, last_ack, meter, env = {}, None, None, None
    mirror_wait = 0.0

    def guard():
        if read_boot_id() != boot or time.monotonic() >= deadline:
            raise RuntimeError("S1.1 boot/one-shot deadline stop")

    def ownership(ordinal, point, panel=None, seed=None, tag=None, context=None):
        binding = (
            config["reference"] if point == "reference" else config["checkpoints"].get(point, {})
        )
        return dict(
            ordinal=ordinal,
            point=point,
            panel=panel,
            seed=seed,
            tag=tag,
            boot_id=boot,
            boot_id_after=read_boot_id(),
            owner_pid=os.getpid(),
            launch_receipt_sha256=sha(out / "launch_receipt.json"),
            worker_started_sha256=sha(out / "worker_started.json"),
            config_sha256=sha(out / "config.json"),
            input_manifest_sha256=sha(out / "input_manifest.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            checkpoint_sha256=binding.get("sha256"),
            policy_hash=binding.get("policy_hash"),
            optimizer_fingerprint=binding.get("optimizer_fingerprint", fingerprint(None)),
            initial_context_fingerprint=context,
            cumulative_before=dict(cumulative),
        )

    def acknowledge(path, identifier, ordinal):
        nonlocal last_ack, mirror_wait
        receipt_sha = sha(path / "receipt.json")
        journal.record(
            "segment_closed",
            ordinal=ordinal,
            identifier=identifier,
            receipt_sha256=receipt_sha,
            cumulative=dict(cumulative),
        )
        descriptor = publish_and_wait(
            out,
            path,
            identifier,
            ordinal,
            boot,
            deadline,
            journal,
            timeout_seconds=config["ack_timeout_seconds"],
            poll_seconds=config["poll_seconds"],
        )
        mirror_wait += descriptor["wait_seconds"]
        last_ack = descriptor["ack"]["sha256"]
        return descriptor

    try:
        guard()
        actors, states = {}, {}
        path = out / "evidence/loads"
        journal.record("segment_started", ordinal=-1, identifier="loads", previous_ack_sha256=None)
        meter = EvaluationMeter(path, config, cumulative, deadline, "loads")
        for name in config["evaluation_order"]:
            meter.context = dict(role="source_checkpoint_load", point=name)
            if name == "reference":
                actors[name] = LegacySACActor.from_checkpoint(
                    out / "inputs/s1/inputs/reference.pt", meter=meter
                )
                states[name] = None
            else:
                cp = result["points"][name]["checkpoint"]
                actors[name], states[name], _ = meter.measure(
                    "checkpoint_loads",
                    1,
                    lambda: policy.Policy.from_checkpoint(out / cp["file"], device="cpu"),
                    point=name,
                )
            actors[name].eval()
            expected = config["reference"] if name == "reference" else config["checkpoints"][name]
            if actors[name].hash() != expected["policy_hash"] or (
                name != "reference"
                and fingerprint(states[name]) != expected["optimizer_fingerprint"]
            ):
                raise ValueError("S1.1 loaded actor/Adam owner differs")
        meter.close()
        setup_work = meter.totals()
        own = ownership(-1, "loads", tag="loads")
        cumulative = add_work(cumulative, setup_work)
        for name in ("raw.h5", "ledger.jsonl"):
            with (path / name).open("rb") as stream:
                os.fsync(stream.fileno())
        fsync_directory(path)
        durable_json(
            path / "receipt.json",
            dict(
                schema="walker-durable-loads-v1",
                ownership=own,
                episode=dict(role="checkpoint_loads"),
                work=setup_work,
                cumulative=dict(cumulative),
                files={n: file_binding(path / n) for n in ("raw.h5", "ledger.jsonl")},
            ),
        )
        result["setup"] = acknowledge(path, "loads", -1)
        meter = None

        def actor_owners():
            return {
                name: dict(
                    policy_hash=actor.hash(),
                    optimizer_fingerprint=fingerprint(states[name]),
                    private_rng_fingerprint=fingerprint(actor._generator.get_state())
                    if hasattr(actor, "_generator")
                    else fingerprint(None),
                )
                for name, actor in actors.items()
            }

        result["actors_initial"] = actor_owners()
        from src.envs.walker2d import Walker2d

        env = Walker2d(**config["task"]["task"])
        paired = {}
        ordinal = 0
        for panel in ("primary", "confirmation"):
            for name in config["evaluation_order"]:
                episodes = []
                for index, seed in enumerate(config[panel + "_seeds"]):
                    guard()
                    if ordinal >= config["budgets"]["episodes"]:
                        raise BudgetExhausted("S1.1 fixed episode roster exhausted")
                    tag = f"{panel}_{name}_{index:02d}"
                    identifier = f"{ordinal:03d}_{tag}"
                    journal.record(
                        "segment_started",
                        ordinal=ordinal,
                        identifier=identifier,
                        previous_ack_sha256=last_ack,
                    )
                    path = out / "evidence/episodes" / identifier
                    meter = EvaluationMeter(path, config, cumulative, deadline, "episode")
                    meter.context = dict(role="source_standalone", point=name, cohort=panel)
                    actor = actors[name]

                    def controller(physical, warm, observation, step):
                        expected = np.concatenate(
                            (physical[2:10], np.clip(physical[10:19], -10, 10))
                        )
                        if not np.array_equal(expected, observation):
                            raise ValueError("S1.1 unchanged native features differ")
                        return actor.predict(observation, meter=meter), dict(
                            policy_hash=result["actors_initial"][name]["policy_hash"],
                            controller="standalone_torque",
                        )

                    episode = runtime.execute_episode(
                        env, controller, seed, config["episode_steps"], meter, tag
                    )
                    context = fingerprint(
                        dict(
                            state=episode.states[0],
                            warm=episode.warmstarts[0],
                            integration=episode.integrations[0],
                        )
                    )
                    if paired.setdefault(seed, context) != context:
                        raise ValueError("S1.1 full reset context pairing differs")
                    if actor_owners() != result["actors_initial"]:
                        raise ValueError("S1.1 evaluation mutated actor/continuation state")
                    own = ownership(ordinal, name, panel, seed, tag, context)
                    meter.close()
                    work = meter.totals()
                    record = {
                        key: getattr(episode, key)
                        for key in (
                            "tag",
                            "seed",
                            "steps",
                            "success",
                            "survived",
                            "mae",
                            "cost",
                            "complete",
                            "terminal_cost",
                            "terminal_adjustment",
                            "absorbing_cost",
                            "raw",
                        )
                    }
                    after = add_work(cumulative, work)
                    seal_segment(path, record, own, work, after)
                    cumulative = after
                    descriptor = acknowledge(path, identifier, ordinal)
                    result["segments"].append(
                        dict(ordinal=ordinal, point=name, panel=panel, tag=tag, **descriptor)
                    )
                    meter = None
                    episodes.append(episode)
                    ordinal += 1
                result["evaluations"][panel][name] = dict(
                    summary=summary(episodes),
                    episodes=[
                        {
                            key: getattr(e, key)
                            for key in (
                                "tag",
                                "seed",
                                "steps",
                                "success",
                                "survived",
                                "mae",
                                "cost",
                                "complete",
                                "terminal_cost",
                                "terminal_adjustment",
                                "absorbing_cost",
                                "raw",
                            )
                        }
                        | dict(segment=result["segments"][-len(episodes) + j]["segment"])
                        for j, e in enumerate(episodes)
                    ],
                    **result["actors_initial"][name],
                )
                print(
                    json.dumps(
                        dict(
                            event="actor_panel_complete",
                            point=name,
                            panel=panel,
                            summary=result["evaluations"][panel][name]["summary"],
                        )
                    ),
                    flush=True,
                )
            result["gates"][panel] = source_screen(result["evaluations"][panel], config)
        guard()
        if ordinal != config["budgets"]["episodes"]:
            raise ValueError("S1.1 fixed episode roster was not completed")
        if (
            sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
            > config["budgets"]["storage_bytes"]
        ):
            raise BudgetExhausted("S1.1 closed evidence storage ceiling exhausted")
        result["actors_final"] = actor_owners()
        result["gates"].update(
            confirmed_walking_benefit=all(
                result["gates"][p]["walking_screen"] for p in ("primary", "confirmation")
            ),
            competence={
                name: all(
                    result["evaluations"][p][name]["summary"]["successes"]
                    >= config["competence"]["minimum_walking_successes_per_panel"]
                    for p in ("primary", "confirmation")
                )
                for name in config["evaluation_order"]
            },
        )
        result.update(execution_complete=True, status="evaluation_complete")
    except BaseException as error:
        result.update(
            status="stopped_partial",
            error=dict(
                type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
            ),
        )
        journal.record("worker_error", error=result["error"])
    finally:

        def retain_cleanup_error(error, operation):
            detail = dict(
                operation=operation,
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
            result.setdefault("cleanup_errors", []).append(detail)
            result.update(execution_complete=False, status="stopped_partial")
            if result["error"] is None:
                result["error"] = detail

        if meter is not None:
            try:
                meter.close()
            except BaseException as error:
                retain_cleanup_error(error, "meter_close")
            result["uncommitted_partial"] = dict(
                directory=str(meter.out.relative_to(out)),
                work=meter.totals(),
                ledger_closed=meter.closed,
            )
            # A receipt may already have charged this closed Meter before an
            # ACK failure. Its prior owner plus actual work is counted once.
            cumulative = add_work(meter.prior, meter.totals())
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                retain_cleanup_error(error, "environment_close")
        result.update(
            work=cumulative,
            phase_work=dict(eval=cumulative),
            mirror_wait_seconds=mirror_wait,
            wall_seconds=time.monotonic() - started,
            worker_wall_seconds=time.monotonic() - started,
            boot_id=boot,
        )
        journal.record(
            "worker_finished", execution_complete=result["execution_complete"], work=cumulative
        )
        journal.close()
        durable_json(out / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("prepare")
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--source", type=Path, required=True)
    item.add_argument("--remote-freshness", type=Path, required=True)
    item.add_argument("--preflight-out", type=Path)
    item = commands.add_parser("worker")
    item.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    result = (
        prepare(out, args.source, args.remote_freshness, args.preflight_out)
        if args.command == "prepare"
        else worker(out)
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in ("prepared", "execution_complete", "error", "work", "gates")
            }
        ),
        flush=True,
    )
    return int(bool(result.get("error")))
