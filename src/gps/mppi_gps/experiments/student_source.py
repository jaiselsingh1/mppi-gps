"""Matched source-corpus learning on CUDA, followed by native CPU evaluation.

The two create-only phases use separate interpreters and durable meters. They
share one sealed recipe, cumulative time/storage/work ceilings and immutable
CPU checkpoints. No planner, critic, relabelling or outcome selection occurs.
"""

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

from ..compatibility_metrics import agreement_metrics
from ..lifecycle import ROOT, freeze, verify
from ..metrics import summary
from ..provenance import default, fingerprint, sha, write
from ..recipes import load_recipe
from ..runtime import BudgetExhausted, Meter


def source_screen(evaluations: dict, config: dict) -> dict:
    pairs = []
    for pair in range(len(config["seed_pairs"])):
        baseline = evaluations[f"mppi_s{pair}"]["summary"]
        candidate = evaluations[f"sac_reference_s{pair}"]["summary"]
        pairs.append(
            dict(
                pair=pair,
                walking_gain=candidate["successes"] - baseline["successes"],
                survival_gain=candidate["survived"] - baseline["survived"],
                walking_nonregression=candidate["successes"] >= baseline["successes"],
                survival_nonregression=candidate["survived"] >= baseline["survived"],
                relative_cost_gain=1 - candidate["mean_cost"] / baseline["mean_cost"],
                controls_delta=candidate["mean_steps"] - baseline["mean_steps"],
            )
        )
    gate = config["walking_screen"]
    checks = dict(
        positive_pairs=sum(
            row["walking_gain"] >= gate["minimum_additional_successes"] for row in pairs
        )
        >= gate["minimum_positive_pairs"],
        all_pair_walking_nonregression=all(row["walking_nonregression"] for row in pairs),
        all_pair_survival_nonregression=all(row["survival_nonregression"] for row in pairs),
    )
    return dict(pairs=pairs, checks=checks, walking_screen=all(checks.values()))


def combine_work(fit: dict, evaluation: dict) -> dict:
    """Sum each phase's numeric counters once; timing remains separately visible."""
    keys = set(fit) | set(evaluation)
    return {key: fit.get(key, 0) + evaluation.get(key, 0) for key in keys}


class SourceMeter(Meter):
    def __init__(
        self, out: Path, config: dict, phase: str, deadline: float, prior: dict | None = None
    ):
        self.config, self.phase, self.deadline = config, phase, deadline
        self.prior = prior or {}
        self.static_bytes = 0
        self.final_work = None
        super().__init__(
            out,
            max_controls=config["budgets"]["live_controls"],
            max_neural_rows=config["budgets"]["neural_rows"]
            - self.prior.get("requested_neural_rows", 0),
        )
        self.refresh_overhead()

    def refresh_overhead(self) -> None:
        run = self.out.parents[1]
        self.static_bytes = sum(
            p.stat().st_size
            for p in run.rglob("*")
            if p.is_file() and not p.is_relative_to(self.out)
        )

    def check_limits(self) -> None:
        if time.monotonic() > self.deadline:
            raise BudgetExhausted("S1 cumulative worker wall ceiling exhausted")
        if (
            self._raw.id.get_filesize()
            + self.ledger_path.stat().st_size
            + self.static_bytes
            + 64 * 1024**2
            > self.config["budgets"]["storage_bytes"]
        ):
            raise BudgetExhausted("S1 cumulative storage ceiling exhausted")

    def measure(self, kind, amount, fn, **metadata):
        self.check_limits()
        key = {
            "neural_rows": "neural_rows",
            "backward_rows": "backward_rows",
            "optimizer_steps": "optimizer_steps",
            "checkpoint_loads": "checkpoint_loads",
        }.get(kind)
        if (
            key is None
            or (self.phase == "eval" and kind in {"backward_rows", "optimizer_steps"})
            or (self.phase == "fit" and kind == "checkpoint_loads")
        ):
            raise ValueError("Forbidden S1 phase measurement: " + str(kind))
        requested = "requested_" + kind
        if (
            self.prior.get(requested, 0) + self._counts.get(requested, 0) + amount
            > self.config["budgets"][key]
        ):
            raise BudgetExhausted("S1 work ceiling exhausted before " + kind)
        return super().measure(kind, amount, fn, **metadata)

    def _query(self, mode, controls, frame_skip, *args, **kwargs):
        self.check_limits()
        if self.phase != "eval" or mode != "live":
            raise ValueError("S1 permits only native CPU evaluation controls")
        if (
            self._counts["requested_physics_substeps"] + controls * frame_skip
            > self.config["budgets"]["physics_substeps"]
        ):
            raise BudgetExhausted("S1 physics ceiling exhausted")
        return super()._query(mode, controls, frame_skip, *args, **kwargs)

    def save_arrays(self, tag, **arrays):
        # Permit post-call/failure persistence after a deadline; reserve closing
        # space before starting work. No new arithmetic is admitted afterward.
        estimate = sum(
            np.asarray(v).nbytes if not isinstance(v, dict) else len(json.dumps(v, default=default))
            for v in arrays.values()
        )
        if (
            hasattr(self, "_raw")
            and self._raw.id.valid
            and self._raw.id.get_filesize()
            + self.ledger_path.stat().st_size
            + self.static_bytes
            + 2 * estimate
            + 128 * 1024
            > self.config["budgets"]["storage_bytes"]
        ):
            raise BudgetExhausted("S1 persistence storage ceiling exhausted")
        return super().save_arrays(tag, **arrays)

    def totals(self):
        return dict(self.final_work) if self.final_work is not None else super().totals()

    def close(self):
        if not self.closed:
            super().close()
            self.final_work = super().totals()


def verify_phase(out: Path) -> tuple[dict, dict, float]:
    verify(out)
    config = json.loads((out / "config.json").read_text())
    seal = json.loads((out / "input_manifest.json").read_text())
    if seal["config_sha256"] != sha(out / "config.json") or seal["source_manifest_sha256"] != sha(
        out / "manifest.json"
    ):
        raise ValueError("S1 source/configuration seal changed")
    for name, binding in seal["files"].items():
        path = (out / name).resolve()
        if (
            not path.is_relative_to(out.resolve())
            or sha(path) != binding["sha256"]
            or path.stat().st_size != binding["bytes"]
        ):
            raise ValueError("S1 sealed input changed: " + name)
    launch = json.loads((out / "launch_receipt.json").read_text())
    if launch["config_sha256"] != seal["config_sha256"] or launch["input_manifest_sha256"] != sha(
        out / "input_manifest.json"
    ):
        raise ValueError("S1 external launch ownership changed")
    return config, seal, float(launch["deadline_monotonic"])


def platform_identity(phase: str, config: dict) -> dict:
    import torch

    names = ("numpy", "scipy", "h5py") + (("mujoco",) if phase == "eval" else ())
    identity = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch.__version__,
        **{n: importlib.metadata.version(n) for n in names},
    )
    expected = config["expected_runtime"][phase]
    if (
        sys.platform != "linux"
        or identity["python"] != config["expected_runtime"]["python"]
        or any(identity[k] != v for k, v in expected.items() if k != "cuda")
    ):
        raise ValueError("S1 pinned " + phase + " runtime differs")
    torch.set_num_threads(1)
    if phase == "fit":
        if (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or not torch.cuda.is_available()
            or torch.version.cuda != expected["cuda"]
        ):
            raise ValueError("S1 deterministic CUDA runtime unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        identity.update(
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0),
            deterministic_algorithms=True,
            allow_tf32=False,
        )
        if identity["device"] != "NVIDIA GeForce RTX 5090":
            raise ValueError("S1 CUDA device differs")
    return identity


def prepare(out: Path, diagnostic: Path) -> dict:
    from ..source_inputs import prepare_inputs
    from ..width_inputs import audit_reset_seeds

    config = load_recipe("student_source")
    config["diagnostic_run"] = str(diagnostic.resolve())
    task = json.loads((diagnostic / "inputs/task.json").read_text())
    if task["xml_sha256"] != sha(ROOT / "assets/walker2d.xml"):
        raise ValueError("S1 task XML differs")
    config.update(task=task["task"], xml_sha256=task["xml_sha256"])
    seed_proof = audit_reset_seeds(ROOT, config["primary_seeds"] + config["confirmation_seeds"])
    selected = {v for pair in config["seed_pairs"] for v in pair.values()}

    def contains(value):
        if type(value) is int:
            return value in selected
        if isinstance(value, dict):
            return any(contains(v) for v in value.values())
        return isinstance(value, list) and any(contains(v) for v in value)

    training_collisions = [
        row["file"]
        for row in seed_proof["historical_configurations"]
        if contains(json.loads(Path(row["file"]).read_text()))
    ]
    if not seed_proof["passed"] or len(selected) != 6 or training_collisions:
        raise ValueError("S1 proposed seeds are not locally fresh")
    seed_proof.update(training_seeds=sorted(selected), training_collisions=training_collisions)
    out = freeze(out, config, "student-source-s1")
    inventory = prepare_inputs(diagnostic, out / "inputs", config)
    write(out / "inputs/seed_audit.json", seed_proof)
    write(
        out / "input_manifest.json",
        dict(
            files={
                str(p.relative_to(out)): dict(sha256=sha(p), bytes=p.stat().st_size)
                for p in sorted((out / "inputs").rglob("*"))
                if p.is_file()
            },
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            planned_work=dict(
                optimizer_steps=12000,
                backward_rows=3072000,
                neural_rows=3500000,
                diagnostic_forward_rows=108000,
                live_controls=320000,
                episodes=320,
                checkpoint_loads=10,
                model_controls=0,
            ),
        ),
    )
    return dict(
        prepared=str(out),
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        input_schema=inventory["schema"],
    )


def fit_worker(out: Path) -> dict:
    import torch

    from .. import policy
    from ..source_inputs import load_arrays

    config, _, deadline = verify_phase(out)
    identity = platform_identity("fit", config)
    inventory = json.loads((out / "inputs/inventory.json").read_text())
    data = {
        f"{arm}_{split}": load_arrays(out / "inputs", f"{arm}_{split}.npz", inventory)
        for arm in config["fit_arms"]
        for split in ("train", "validation")
    }
    normal = load_arrays(out / "inputs", "normalization.npz", inventory)
    result = dict(
        schema="walker-student-source-s1-fit-v1",
        execution_complete=False,
        error=None,
        points={},
        fits=[],
        samplers={},
        platform=identity,
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        teacher_outcomes=inventory["teachers"],
    )
    meter = SourceMeter(out / "evidence/fit", config, "fit", deadline)
    started = time.perf_counter()
    try:

        def save_point(actor, state, pair, arm, update):
            name = f"incoming_s{pair}" if update == 0 else f"{arm}_s{pair}"
            meter.context = dict(
                role="source_reconstruction", point=name, pair=pair, arm=arm, update=update
            )
            predictions, metrics = {}, {}
            for split, rows in data.items():
                prediction = np.concatenate(
                    [
                        actor.predict(
                            rows["observations"][i : i + config["prediction_batch_size"]],
                            meter=meter,
                        )
                        for i in range(
                            0, len(rows["observations"]), config["prediction_batch_size"]
                        )
                    ]
                )
                predictions[split + "_predictions"] = prediction
                metrics[split] = dict(
                    mse=float(np.mean((prediction.astype(np.float64) - rows["actions"]) ** 2)),
                    per_trajectory={
                        str(tag): agreement_metrics(
                            prediction[rows["row_episode"] == tag],
                            rows["actions"][rows["row_episode"] == tag],
                        )
                        for tag in dict.fromkeys(rows["row_episode"].tolist())
                    },
                )
            meter.refresh_overhead()
            meter.check_limits()
            checkpoint = actor.checkpoint(
                out / "checkpoints" / (name + ".pt"),
                optimizer_state=state,
                metadata=dict(
                    pair=pair,
                    arm=arm,
                    updates=update,
                    width=128,
                    feature_mode="clipped",
                    deployment_qualified=False,
                ),
            )
            checkpoint["file"] = str(Path(checkpoint["file"]).relative_to(out))
            result["points"][name] = dict(
                pair=pair,
                arm=arm,
                update=update,
                checkpoint=checkpoint,
                optimizer_fingerprint=fingerprint(state),
                parameter_count=sum(p.numel() for p in actor.parameters()),
                metrics=metrics,
                predictions=meter.save_arrays("reconstruction_" + name, **predictions),
            )
            meter.refresh_overhead()
            meter.record("source_checkpoint", point=name, **checkpoint)
            print(
                json.dumps(
                    dict(
                        event="source_checkpoint",
                        point=name,
                        metrics={k: v["mse"] for k, v in metrics.items()},
                    )
                ),
                flush=True,
            )

        for pair, seeds in enumerate(config["seed_pairs"]):
            rng = np.random.Generator(np.random.PCG64(seeds["fit_seed"]))
            first = deepcopy(rng.bit_generator.state)
            indices = np.stack(
                [
                    rng.integers(
                        0, config["training_rows_per_arm"], size=config["learning"]["batch_size"]
                    )
                    for _ in range(config["learning"]["updates"])
                ]
            )
            result["samplers"][str(pair)] = dict(
                seed=seeds["fit_seed"],
                initial_rng=first,
                final_rng=deepcopy(rng.bit_generator.state),
                minibatches=meter.save_arrays(f"sampler_s{pair}", indices=indices),
            )
            parent = policy.Policy(
                seeds["policy_seed"], device="cuda:0", normalization=normal, hidden_dim=128
            )
            parent_hash = parent.hash()
            save_point(parent, None, pair, "incoming", 0)
            initial_owners = []
            for arm in config["fit_arms"]:
                name = f"{arm}_s{pair}"
                initial = {}
                meter.context = dict(role="source_fit", point=name, pair=pair, arm=arm)

                def observe(payload):
                    path = out / "learning" / (name + "_initial.pt")
                    path.parent.mkdir(exist_ok=True)
                    with path.open("xb") as stream:
                        torch.save(payload, stream)
                    initial.update(
                        file=str(path.relative_to(out)),
                        sha256=sha(path),
                        policy_hash=policy._hash_state(payload["state_dict"]),
                        adam_fingerprint=fingerprint(payload["adam"]),
                        rng_fingerprint=fingerprint(payload["rng"]),
                        private_rng_fingerprint=fingerprint(payload["private_rng_state"]),
                    )
                    initial["raw"] = meter.save_arrays(
                        "initial_" + name,
                        **{k: v.numpy() for k, v in payload["state_dict"].items()},
                        private_rng_state=payload["private_rng_state"].numpy(),
                    )
                    meter.refresh_overhead()
                    meter.record("source_fit_initial", point=name, initial=initial)
                    owners = {
                        key: initial[key]
                        for key in (
                            "policy_hash",
                            "adam_fingerprint",
                            "rng_fingerprint",
                            "private_rng_fingerprint",
                        )
                    }
                    if (
                        payload["adam"]["state"]
                        or payload["rng"] != result["samplers"][str(pair)]["initial_rng"]
                        or initial["policy_hash"] != parent_hash
                        or (initial_owners and owners != initial_owners[0])
                    ):
                        raise ValueError("S1 paired initial state differs before the first update")

                try:
                    child, state, evidence = policy.fit_policy(
                        parent,
                        data[arm + "_train"]["observations"],
                        data[arm + "_train"]["actions"],
                        updates=config["learning"]["updates"],
                        batch_size=config["learning"]["batch_size"],
                        lr=config["learning"]["lr"],
                        eta_theta=0.0,
                        seed=seeds["fit_seed"],
                        meter=meter,
                        tag=name,
                        prox_reduction="mean",
                        anchor_refresh=0,
                        initial_observer=observe,
                    )
                except BaseException as error:
                    evidence = getattr(error, "fit_evidence", dict(failed=True, learning_curve=[]))
                    path = out / "learning" / (name + "_failed.json")
                    write(path, evidence)
                    result["fits"].append(
                        dict(
                            point=name,
                            pair=pair,
                            arm=arm,
                            file=str(path.relative_to(out)),
                            sha256=sha(path),
                            failed=True,
                            initial=initial,
                            completed_updates=evidence.get("completed_updates", 0),
                        )
                    )
                    raise
                owners = {
                    k: initial[k]
                    for k in (
                        "policy_hash",
                        "adam_fingerprint",
                        "rng_fingerprint",
                        "private_rng_fingerprint",
                    )
                }
                initial_owners.append(owners)
                if initial["policy_hash"] != parent_hash or (
                    len(initial_owners) == 2 and initial_owners[0] != owners
                ):
                    raise ValueError("S1 paired actual initial learning state differs")
                path = out / "learning" / (name + ".json")
                write(path, evidence)
                result["fits"].append(
                    dict(
                        point=name,
                        pair=pair,
                        arm=arm,
                        file=str(path.relative_to(out)),
                        sha256=sha(path),
                        initial=initial,
                        optimizer_fingerprint=fingerprint(state),
                    )
                )
                save_point(child, state, pair, arm, config["learning"]["updates"])
                if (
                    state["rng"] != result["samplers"][str(pair)]["final_rng"]
                    or parent.hash() != parent_hash
                ):
                    raise ValueError("S1 sampling or parent ownership changed")
                del child, state
        meter.check_limits()
        result["execution_complete"] = True
    except BaseException as error:
        result["error"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
        meter.record("source_fit_stopped", error=result["error"])
    finally:
        meter.close()
        result.update(work=meter.totals(), wall_seconds=time.perf_counter() - started)
        write(out / "fit_result.json", result)
    return result


def eval_worker(out: Path) -> dict:
    from src.envs.walker2d import Walker2d

    from .. import policy, runtime
    from ..diagnostic_policies import LegacySACActor

    config, _, deadline = verify_phase(out)
    identity = platform_identity("eval", config)
    fit = json.loads((out / "fit_result.json").read_text())
    if not fit["execution_complete"] or fit["error"]:
        raise ValueError("S1 cannot evaluate an incomplete fit phase")
    if (
        fit["schema"] != "walker-student-source-s1-fit-v1"
        or fit["config_sha256"] != sha(out / "config.json")
        or fit["input_manifest_sha256"] != sha(out / "input_manifest.json")
        or set(fit["points"]) != set(config["evaluation_order"]) - {"reference"}
        or len(fit["fits"]) != 6
    ):
        raise ValueError("S1 complete fit-phase ownership differs")
    fit_exit = json.loads((out / "fit_exit.json").read_text())
    if fit_exit["exit_code"] != 0 or fit_exit["fit_result_sha256"] != sha(out / "fit_result.json"):
        raise ValueError("S1 external fit-phase exit binding differs")
    result = dict(
        schema="walker-student-source-s1-result-v1",
        execution_complete=False,
        error=None,
        status="evaluating",
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        fit_result_sha256=sha(out / "fit_result.json"),
        points=fit["points"],
        fits=fit["fits"],
        samplers=fit["samplers"],
        platform=dict(fit=fit["platform"], eval=identity),
        teacher_outcomes=fit["teacher_outcomes"],
        evaluations=dict(primary={}, confirmation={}),
        gates={},
    )
    meter = SourceMeter(out / "evidence/eval", config, "eval", deadline, fit["work"])
    env = None
    started = time.perf_counter()
    try:
        actors, states = {}, {}
        for name in config["evaluation_order"]:
            meter.context = dict(role="source_checkpoint_load", point=name)
            if name == "reference":
                actors[name] = LegacySACActor.from_checkpoint(
                    out / "inputs/reference.pt", meter=meter
                )
                states[name] = None
                if actors[name].hash() != config["source_bindings"]["reference_policy_hash"]:
                    raise ValueError("S1 reference actor differs")
            else:
                entry = fit["points"][name]["checkpoint"]
                path = out / entry["file"]
                if not path.resolve().is_relative_to(out.resolve()):
                    raise ValueError("S1 checkpoint escapes the frozen run")
                if sha(path) != entry["sha256"]:
                    raise ValueError("S1 trained checkpoint bytes changed")
                actor, state, _ = meter.measure(
                    "checkpoint_loads",
                    1,
                    lambda: policy.Policy.from_checkpoint(path, device="cpu"),
                    point=name,
                )
                if (
                    actor.hash() != entry["policy_hash"]
                    or fingerprint(state) != fit["points"][name]["optimizer_fingerprint"]
                ):
                    raise ValueError("S1 CPU checkpoint continuation changed")
                actors[name], states[name] = actor, state
            actors[name].eval()
        env = Walker2d(**config["task"])
        paired = {}
        for panel in ("primary", "confirmation"):
            for name in config["evaluation_order"]:
                actor, state = actors[name], states[name]
                owner, optimizer_owner = actor.hash(), fingerprint(state)
                episodes = []
                for index, seed in enumerate(config[panel + "_seeds"]):
                    if meter._counts["episodes"] >= config["budgets"]["episodes"]:
                        raise BudgetExhausted("S1 episode roster ceiling exhausted")
                    meter.context = dict(role="source_standalone", point=name, cohort=panel)

                    def controller(physical, warm, observation, step):
                        expected = np.concatenate(
                            (physical[2:10], np.clip(physical[10:19], -10, 10))
                        )
                        if not np.array_equal(expected, observation):
                            raise ValueError("S1 native feature alignment differs")
                        return actor.predict(observation, meter=meter), dict(
                            policy_hash=owner, controller="standalone_torque"
                        )

                    episode = runtime.execute_episode(
                        env,
                        controller,
                        seed,
                        config["episode_steps"],
                        meter,
                        f"{panel}_{name}_{index:02d}",
                    )
                    initial = fingerprint(
                        dict(
                            state=episode.states[0],
                            warm=episode.warmstarts[0],
                            integration=episode.integrations[0],
                        )
                    )
                    if paired.setdefault(seed, initial) != initial:
                        raise ValueError("S1 reset pairing differs")
                    episodes.append(episode)
                    meter.check_limits()
                if actor.hash() != owner or fingerprint(state) != optimizer_owner:
                    raise ValueError("S1 evaluation mutated learner state")
                result["evaluations"][panel][name] = dict(
                    summary=summary(episodes),
                    episodes=[
                        {
                            k: getattr(e, k)
                            for k in (
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
                        for e in episodes
                    ],
                    policy_hash=owner,
                    optimizer_fingerprint=optimizer_owner,
                )
                meter.record(
                    "source_actor_evaluated",
                    point=name,
                    cohort=panel,
                    summary=result["evaluations"][panel][name]["summary"],
                )
            result["gates"][panel] = source_screen(result["evaluations"][panel], config)
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
        result.update(execution_complete=True, status="source_comparison_complete")
    except BaseException as error:
        result.update(
            error=dict(
                type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
            ),
            status="stopped_partial",
        )
        meter.record("source_eval_stopped", error=result["error"])
    finally:
        if env is not None:
            env.close()
        meter.close()
        result.update(
            phase_work=dict(fit=fit["work"], eval=meter.totals()),
            work=combine_work(fit["work"], meter.totals()),
            wall_seconds=fit["wall_seconds"] + time.perf_counter() - started,
        )
        write(out / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--diagnostic", type=Path, required=True)
    for phase in ("fit", "evaluate"):
        item = sub.add_parser(phase)
        item.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        prepare(args.out, args.diagnostic)
        if args.command == "prepare"
        else (
            fit_worker(args.out.resolve())
            if args.command == "fit"
            else eval_worker(args.out.resolve())
        )
    )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "prepared",
                    "execution_complete",
                    "status",
                    "error",
                    "work",
                    "gates",
                    "config_sha256",
                    "input_manifest_sha256",
                )
            },
            default=default,
        ),
        flush=True,
    )
    return int(bool(result.get("error")))


if __name__ == "__main__":
    raise SystemExit(main())
