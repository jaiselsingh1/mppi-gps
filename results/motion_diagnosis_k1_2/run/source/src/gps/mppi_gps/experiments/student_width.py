"""One frozen 128/256 tanh-width contrast, with fresh paired learning streams.

No planner is constructed. Preparation reads sealed K1 inputs; fitting remains
simulator-free. Native evaluation begins only after all six fixed fits finish.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ..compatibility_metrics import agreement_metrics
from ..lifecycle import ROOT, freeze, verify
from ..metrics import summary
from ..provenance import default, fingerprint, sha, write
from ..recipes import load_recipe
from ..runtime import BudgetExhausted, Meter


def point_name(pair: int, width: int, update: int) -> str:
    return f"s{pair}_w{width}_u{update}"


def reservation(config: dict[str, Any]) -> dict[str, int]:
    actors = len(config["widths"]) * len(config["seed_pairs"])
    diagnostic = actors * len(config["checkpoints"]) * (config["rows"] + config["validation_rows"])
    fitting = actors * config["updates"] * config["batch_size"]
    episodes = actors * (
        len(config["checkpoints"]) * len(config["curve_seeds"])
        + len(config["primary_seeds"])
        + len(config["confirmation_seeds"])
    )
    controls = episodes * config["episode_steps"]
    return dict(
        requested_controls=controls,
        neural_rows=fitting + diagnostic + controls,
        backward_rows=fitting,
        optimizer_steps=actors * config["updates"],
        episodes=episodes,
        model_controls=0,
        diagnostic_forward_rows=diagnostic,
    )


def contrast(points: dict[str, Any], evaluations: dict[str, Any], config: dict[str, Any]) -> dict:
    """Predeclared seed-paired screens; gates never select checkpoints or seeds."""
    fitting, validation, task = [], [], []
    for pair in range(len(config["seed_pairs"])):
        keys = [point_name(pair, width, config["updates"]) for width in config["widths"]]
        small, wide = [points[key]["metrics"] for key in keys]
        fitting.append(1 - wide["train"]["mse"] / small["train"]["mse"])
        validation.append(1 - wide["validation"]["mse"] / small["validation"]["mse"])
        if evaluations:
            small_eval, wide_eval = [evaluations[key]["summary"] for key in keys]
            task.append(
                dict(
                    relative_cost_gain=1 - wide_eval["mean_cost"] / small_eval["mean_cost"],
                    walking_gain=wide_eval["successes"] - small_eval["successes"],
                    walking_nonregression=wide_eval["successes"] >= small_eval["successes"],
                    survival_nonregression=wide_eval["survived"] >= small_eval["survived"],
                    steps_nonregression=wide_eval["mean_steps"] >= small_eval["mean_steps"],
                )
            )
    gates = config["gates"]
    fit_checks = dict(
        mean_train_gain=float(np.mean(fitting)) >= gates["minimum_train_mse_gain"],
        fitting_pairs=sum(value > 0 for value in fitting) >= gates["minimum_fitting_pairs"],
    )
    task_checks = (
        {}
        if not task
        else dict(
            mean_cost_gain=float(np.mean([row["relative_cost_gain"] for row in task]))
            >= gates["minimum_cost_gain"],
            cost_pairs=sum(row["relative_cost_gain"] > 0 for row in task)
            >= gates["minimum_cost_pairs"],
            walking_pairs=sum(row["walking_gain"] >= gates["minimum_walking_gain"] for row in task)
            >= gates["minimum_walking_pairs"],
            all_pair_walking_nonregression=all(row["walking_nonregression"] for row in task),
            all_pair_survival_nonregression=all(row["survival_nonregression"] for row in task),
            all_pair_steps_nonregression=all(row["steps_nonregression"] for row in task),
        )
    )
    return dict(
        relative_train_mse_gains=fitting,
        relative_validation_mse_gains=validation,
        fitting_checks=fit_checks,
        reconstruction_screen=all(fit_checks.values()),
        validation_guard=float(np.mean(validation)) >= 0,
        fitting_screen=all(fit_checks.values()) and float(np.mean(validation)) >= 0,
        task_pairs=task,
        task_checks=task_checks,
        task_screen=bool(task_checks) and all(task_checks.values()),
    )


class WidthMeter(Meter):
    """Enforce the finite learning/live budget before each external operation."""

    def __init__(self, out: Path, budgets: dict[str, int]):
        self.budgets = budgets
        self.deadline = time.perf_counter() + budgets["wall_seconds"]
        self.final_work = None
        # Reserve 1 GiB for source/input/checkpoint/learning/result files.
        self.non_evidence_reservation = 1024**3
        super().__init__(
            out, max_controls=budgets["requested_controls"], max_neural_rows=budgets["neural_rows"]
        )

    def check_limits(self) -> None:
        if time.perf_counter() > self.deadline:
            raise BudgetExhausted("W1 wall ceiling exhausted")
        if (
            self._raw.id.get_filesize()
            + self.ledger_path.stat().st_size
            + self.non_evidence_reservation
            >= self.budgets["storage_bytes"]
        ):
            raise BudgetExhausted("W1 storage ceiling exhausted")

    def save_arrays(self, tag, **arrays):
        self.check_limits()
        reserve = 262144 * (1 + len(arrays))
        for value in arrays.values():
            reserve += (
                len(json.dumps(value, default=default).encode())
                if isinstance(value, dict)
                else np.asarray(value).nbytes
            )
        stored = self._raw.id.get_filesize() + self.ledger_path.stat().st_size
        if stored + reserve + self.non_evidence_reservation > self.budgets["storage_bytes"]:
            raise BudgetExhausted("W1 array persistence exceeds reserved storage")
        return super().save_arrays(tag, **arrays)

    def measure(self, kind, amount, fn, **metadata):
        self.check_limits()
        if kind not in {"neural_rows", "backward_rows", "optimizer_steps"}:
            raise ValueError("Unregistered W1 operation")
        if kind in {"backward_rows", "optimizer_steps"}:
            if self._counts["requested_" + kind] + amount > self.budgets[kind]:
                self._counts["rejected_measurements"] += 1
                self.record(
                    "measurement_rejected",
                    measure_kind=kind,
                    amount=amount,
                    **self._fields(metadata),
                )
                raise BudgetExhausted("W1 learning reservation exhausted before operation")
        return super().measure(kind, amount, fn, **metadata)

    def _query(self, mode, *args, **kwargs):
        self.check_limits()
        if mode != "live":
            raise ValueError("W1 forbids new model-rollout controls")
        return super()._query(mode, *args, **kwargs)

    def totals(self):
        return dict(self.final_work) if self.final_work is not None else super().totals()

    def close(self):
        if not self.closed:
            super().close()
            self.final_work = super().totals()


def prepare(out: Path, k1: Path) -> dict[str, Any]:
    from ..width_inputs import audit_reset_seeds, extract_benchmark

    cfg = load_recipe("student_width")
    if cfg["gradient_clip"] != 0.5 or cfg["normalization_floor"] != 0.05:
        raise ValueError("W1 clip/floor recipe differs from the shared fitting/input contracts")
    cfg["k1_run"] = str(k1.resolve())
    cfg["task"] = json.loads((k1 / "config.json").read_text())["task"]
    cfg["xml_sha256"] = sha(ROOT / "assets/walker2d.xml")
    cfg["authorization"] = (
        "User: Can you go ahead with the next tests?; 31 October 2026. One recommended width contrast, not the queued planning/coupling studies."
    )
    planned = reservation(cfg)
    for key, value in planned.items():
        if key in cfg["budgets"] and value > cfg["budgets"][key]:
            raise ValueError("W1 roster exceeds ceiling: " + key)
    seeds = cfg["curve_seeds"] + cfg["primary_seeds"] + cfg["confirmation_seeds"]
    seed_proof = audit_reset_seeds(ROOT, seeds)
    if not seed_proof["passed"]:
        raise ValueError("W1 evaluation reset collision: " + str(seed_proof["collisions"]))
    out = freeze(out, cfg, command="student-width-w1")
    provenance = extract_benchmark(k1, out / "inputs")
    write(out / "inputs/task.json", dict(task=cfg["task"], xml_sha256=cfg["xml_sha256"]))
    write(out / "inputs/seed_audit.json", seed_proof)
    write(
        out / "input_manifest.json",
        dict(
            files={str(p.relative_to(out)): sha(p) for p in sorted((out / "inputs").iterdir())},
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            planned_work=planned,
        ),
    )
    return dict(
        prepared=str(out),
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        planned_work=planned,
        benchmark_schema=provenance["schema"],
    )


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stream:
        return {key: np.array(stream[key]) for key in stream.files}


def episode_record(episode) -> dict[str, Any]:
    keys = (
        "cost",
        "success",
        "survived",
        "steps",
        "mae",
        "seed",
        "tag",
        "complete",
        "terminal_cost",
        "terminal_adjustment",
        "absorbing_cost",
        "raw",
    )
    return {key: getattr(episode, key) for key in keys}


def worker(out: Path) -> int:
    import torch

    from .. import policy, runtime

    verify(out)
    cfg = json.loads((out / "config.json").read_text())
    seal = json.loads((out / "input_manifest.json").read_text())
    if (
        sha(out / "config.json") != seal["config_sha256"]
        or sha(out / "manifest.json") != seal["source_manifest_sha256"]
    ):
        raise ValueError("W1 freeze changed")
    for path, digest in seal["files"].items():
        if sha(out / path) != digest:
            raise ValueError("W1 input changed: " + path)
    if (out / "evidence").exists():
        raise FileExistsError("W1 worker is create-only")
    torch.set_num_threads(cfg["torch_threads"])
    data = {split: read_npz(out / f"inputs/{split}.npz") for split in ("train", "validation")}
    normal = read_npz(out / "inputs/normalization.npz")
    provenance = json.loads((out / "inputs/provenance.json").read_text())
    actors, states = {}, {}
    result = dict(
        schema=cfg["schema"],
        execution_complete=False,
        error=None,
        status="started",
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        points={},
        fits=[],
        samplers={},
        evaluations=dict(curve={}, primary={}, confirmation={}),
        gates={},
        teacher_outcomes={key: value["outcomes"] for key, value in provenance["splits"].items()},
    )
    started = time.perf_counter()
    meter = WidthMeter(out / "evidence", cfg["budgets"])
    env = None
    try:

        def save_point(actor, state, pair, width, update):
            name = point_name(pair, width, update)
            meter.context = dict(
                role="width_saved_reconstruction", point=name, pair=pair, width=width, update=update
            )
            predictions, metrics = {}, {}
            for split, rows in data.items():
                prediction = np.concatenate(
                    [
                        actor.predict(
                            rows["observations"][start : start + cfg["prediction_batch_size"]],
                            meter=meter,
                        )
                        for start in range(
                            0, len(rows["observations"]), cfg["prediction_batch_size"]
                        )
                    ]
                )
                predictions[split + "_predictions"] = prediction
                per_trajectory = {
                    tag: agreement_metrics(
                        prediction[rows["row_episode"] == tag],
                        rows["actions"][rows["row_episode"] == tag],
                    )
                    for tag in dict.fromkeys(rows["row_episode"].tolist())
                }
                metrics[split] = dict(
                    mse=float(np.mean((prediction.astype(np.float64) - rows["actions"]) ** 2)),
                    per_trajectory=per_trajectory,
                )
            overhead = sum(
                p.stat().st_size
                for p in out.rglob("*")
                if p.is_file() and "evidence" not in p.relative_to(out).parts
            )
            if overhead + 16777216 > meter.non_evidence_reservation:
                raise BudgetExhausted("W1 non-evidence storage reservation exhausted")
            checkpoint = actor.checkpoint(
                out / "checkpoints" / (name + ".pt"),
                optimizer_state=state,
                metadata=dict(
                    pair=pair,
                    width=width,
                    updates=update,
                    feature_mode="clipped",
                    fresh_initialization=True,
                    deployment_qualified=False,
                ),
            )
            checkpoint["file"] = str(Path(checkpoint["file"]).relative_to(out))
            point = dict(
                pair=pair,
                width=width,
                update=update,
                policy_seed=cfg["seed_pairs"][pair]["policy_seed"],
                fit_seed=cfg["seed_pairs"][pair]["fit_seed"],
                checkpoint=checkpoint,
                optimizer_fingerprint=fingerprint(state),
                parameter_count=sum(p.numel() for p in actor.parameters()),
                metrics=metrics,
                predictions=meter.save_arrays("reconstruction_" + name, **predictions),
            )
            result["points"][name] = point
            actors[name], states[name] = actor, state
            meter.record("width_checkpoint", point=name, **checkpoint)
            print(
                json.dumps(
                    dict(
                        event="width_checkpoint",
                        point=name,
                        train_mse=metrics["train"]["mse"],
                        validation_mse=metrics["validation"]["mse"],
                    )
                ),
                flush=True,
            )

        for pair, seeds in enumerate(cfg["seed_pairs"]):
            rng = np.random.Generator(np.random.PCG64(seeds["fit_seed"]))
            endpoints = {"0": deepcopy(rng.bit_generator.state)}
            batches = []
            for update in range(1, cfg["updates"] + 1):
                batches.append(rng.integers(0, cfg["rows"], size=cfg["batch_size"]))
                if update in cfg["checkpoints"]:
                    endpoints[str(update)] = deepcopy(rng.bit_generator.state)
            result["samplers"][str(pair)] = dict(
                seed=seeds["fit_seed"],
                minibatches=meter.save_arrays(
                    f"paired_minibatches_s{pair}", indices=np.stack(batches)
                ),
                rng_at_checkpoints=endpoints,
            )
            for width in cfg["widths"]:
                actor = policy.Policy(
                    seed=seeds["policy_seed"],
                    device=cfg["device"],
                    normalization=normal,
                    hidden_dim=width,
                )
                state, previous = None, 0
                save_point(actor, state, pair, width, 0)
                normal_hash = fingerprint(actor.normalization)
                for update in cfg["checkpoints"][1:]:
                    name = point_name(pair, width, update)
                    meter.context = dict(
                        role="width_fitting", point=name, pair=pair, width=width, update=update
                    )
                    actor, state, fit = policy.fit_policy(
                        actor,
                        data["train"]["observations"],
                        data["train"]["actions"],
                        optimizer_state=state,
                        updates=update - previous,
                        batch_size=cfg["batch_size"],
                        lr=cfg["lr"],
                        eta_theta=cfg["eta_theta"],
                        seed=seeds["fit_seed"],
                        meter=meter,
                        tag=name,
                        resume_fit=previous > 0,
                        prox_reduction=cfg["prox_reduction"],
                        anchor_refresh=cfg["anchor_refresh"],
                    )
                    if (
                        state["rng"] != endpoints[str(update)]
                        or state["updates"] != update
                        or state["fit_updates"] != update
                        or fingerprint(actor.normalization) != normal_hash
                    ):
                        raise ValueError("W1 matched continuation changed")
                    file = out / "learning" / (name + ".json")
                    write(file, fit)
                    result["fits"].append(
                        dict(point=name, file=str(file.relative_to(out)), sha256=sha(file))
                    )
                    save_point(actor, state, pair, width, update)
                    previous = update
        if any(name == "mujoco" or name.startswith("mujoco.") for name in sys.modules):
            raise ValueError("W1 fitting imported simulator")
        work = meter.totals()
        if (
            work["completed_optimizer_steps"] != 60000
            or work["completed_backward_rows"] != 15360000
        ):
            raise ValueError("W1 fixed training budget changed")
        from src.envs.walker2d import Walker2d

        env = Walker2d(**cfg["task"])
        meter.instrument(env)
        if env._use_warp:
            raise ValueError("W1 requires native CPU execution")
        paired_resets = {}

        def evaluate(name, seeds, cohort):
            actor, state = actors[name], states[name]
            owner, optimizer_owner = actor.hash(), fingerprint(state)
            episodes = []
            for index, seed in enumerate(seeds):
                if meter.totals()["episodes"] >= cfg["budgets"]["episodes"]:
                    raise BudgetExhausted("W1 episode roster exhausted before reset")
                meter.context = dict(
                    role="width_standalone",
                    point=name,
                    cohort=cohort,
                    pair=result["points"][name]["pair"],
                    width=actor.hidden_dim,
                )

                def controller(physical, warm, observation, step):
                    expected = np.concatenate((physical[2:10], np.clip(physical[10:19], -10, 10)))
                    if not np.array_equal(expected, observation):
                        raise ValueError("W1 native pre-action feature alignment changed")
                    return actor.predict(observation, meter=meter), dict(
                        policy_hash=owner, feature_mode="clipped", width=actor.hidden_dim
                    )

                episode = runtime.execute_episode(
                    env,
                    controller,
                    seed,
                    cfg["episode_steps"],
                    meter,
                    f"{cohort}_{name}_{index:02d}",
                )
                initial = fingerprint(
                    dict(
                        state=episode.states[0],
                        warm=episode.warmstarts[0],
                        integration=episode.integrations[0],
                    )
                )
                if paired_resets.setdefault(seed, initial) != initial:
                    raise ValueError("W1 reset pairing changed")
                episodes.append(episode)
                meter.check_limits()
            if actor.hash() != owner or fingerprint(state) != optimizer_owner:
                raise ValueError("W1 evaluation changed actor/Adam state")
            result["evaluations"][cohort][name] = dict(
                summary=summary(episodes),
                episodes=[episode_record(e) for e in episodes],
                policy_hash=owner,
                optimizer_fingerprint=optimizer_owner,
            )
            print(
                json.dumps(
                    dict(event="width_evaluation", point=name, cohort=cohort, **summary(episodes))
                ),
                flush=True,
            )

        for name in result["points"]:
            evaluate(name, cfg["curve_seeds"], "curve")
        final = [
            point_name(pair, width, cfg["updates"])
            for pair in range(len(cfg["seed_pairs"]))
            for width in cfg["widths"]
        ]
        for name in final:
            evaluate(name, cfg["primary_seeds"], "primary")
        primary = contrast(result["points"], result["evaluations"]["primary"], cfg)
        trigger = primary["task_screen"] or any(
            result["evaluations"]["primary"][name]["summary"]["successes"]
            >= cfg["gates"]["competence_minimum_successes"]
            for name in final
        )
        result["gates"] = dict(
            primary=primary,
            confirmation_triggered=trigger,
            confirmation=None,
            confirmed_task_benefit=False,
            competence={},
        )
        if trigger:
            for name in final:
                evaluate(name, cfg["confirmation_seeds"], "confirmation")
            confirmed = contrast(result["points"], result["evaluations"]["confirmation"], cfg)
            result["gates"]["confirmation"] = confirmed
            result["gates"]["confirmed_task_benefit"] = (
                primary["task_screen"] and confirmed["task_screen"]
            )
        for name in final:
            result["gates"]["competence"][name] = bool(
                trigger
                and result["evaluations"]["primary"][name]["summary"]["successes"]
                >= cfg["gates"]["competence_minimum_successes"]
                and result["evaluations"]["confirmation"][name]["summary"]["successes"]
                >= cfg["gates"]["competence_minimum_successes"]
            )
        result["execution_complete"] = True
        result["status"] = "complete"
    except BaseException as error:
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        result["status"] = "stopped"
        if hasattr(error, "fit_evidence"):
            write(out / "interrupted_fit.json", error.fit_evidence)
    finally:
        if env is not None:
            env.close()
        meter.close()
        result["work"] = meter.totals()
        result["wall_seconds"] = time.perf_counter() - started
        result["stored_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        write(out / "result.json", result)
    return 0 if result["execution_complete"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--k1", type=Path, required=True)
    item = sub.add_parser("worker")
    item.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.out.resolve(), args.k1.resolve()), indent=2))
        return 0
    return worker(args.out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
