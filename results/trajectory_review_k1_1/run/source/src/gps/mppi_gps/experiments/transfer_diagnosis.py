"""Bounded saved-distribution analysis and single-action continuation assay.

No policy or planner is constructed. Simulator imports occur only after D1.
The complete unmodified replay gate precedes every substituted-action branch.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ..lifecycle import ROOT, freeze, verify
from ..provenance import default, sha, write
from ..recipes import load_recipe
from ..runtime import BudgetExhausted, Meter


class TransferMeter(Meter):
    """Reserve native work and support queries before crossing each boundary."""

    def __init__(self, out: Path, config: dict[str, Any], static_bytes: int = 0):
        self.config = config
        self.static_bytes = static_bytes
        self.stage: str | None = None
        self.stage_started = time.perf_counter()
        self.final_work: dict[str, Any] | None = None
        super().__init__(out, max_controls=config["d2_budgets"]["live_controls"], max_neural_rows=0)
        self._counts.update(support_queries=0, completed_support_queries=0, branches=0)

    def begin_stage(self, stage: str) -> None:
        if stage not in {"d1", "d2"}:
            raise ValueError("Unknown transfer stage")
        self.stage, self.stage_started = stage, time.perf_counter()
        self.record("transfer_stage_start", stage=stage)
        self.check_limits()

    def check_limits(self, prospective_bytes: int = 0) -> None:
        if self.stage:
            limits = self.config[self.stage + "_budgets"]
            if time.perf_counter() - self.stage_started > limits["wall_seconds"]:
                raise BudgetExhausted(f"{self.stage.upper()} wall ceiling exhausted")
            if self.stage == "d1":
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
                if peak_bytes > limits["memory_bytes"]:
                    raise BudgetExhausted("D1 process peak memory ceiling exhausted")
        if hasattr(self, "_raw") and self._raw.id.valid:
            used = self._raw.id.get_filesize() + self.ledger_path.stat().st_size
            # Reserve output metadata/HDF5 object overhead before each save.
            if (
                used + self.static_bytes + prospective_bytes + 64 * 1024**2
                > self.config["budgets"]["storage_bytes"]
            ):
                raise BudgetExhausted("Transfer storage ceiling exhausted")

    def stage_summary(self) -> dict[str, Any]:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(
            worker_wall_seconds=time.perf_counter() - self.stage_started,
            peak_memory_bytes=int(peak if sys.platform == "darwin" else peak * 1024),
        )

    def support_queries(self, amount: int, **metadata: Any) -> None:
        self.check_limits()
        if self.stage != "d1" or type(amount) is not int or amount < 0:
            raise ValueError("Explicit D1 support query reservation required")
        if self._counts["support_queries"] + amount > self.config["d1_budgets"]["support_queries"]:
            raise BudgetExhausted("D1 support query ceiling exhausted")
        self._counts["support_queries"] += amount
        self.record("support_queries_reserved", amount=amount, **metadata)

    def measure(self, kind: str, amount: int, fn: Any, **metadata: Any) -> Any:
        raise ValueError("Transfer diagnosis forbids neural/learning measurements")

    def save_arrays(self, tag: str, **arrays: Any) -> dict[str, Any]:
        estimate = sum(
            np.asarray(value).nbytes
            if not isinstance(value, dict)
            else len(json.dumps(value, default=default))
            for value in arrays.values()
        )
        # Work boundaries enforce wall/memory limits. Completed native work and
        # partial traces must still be persisted after a boundary exhausts them.
        # check_limits reserves 64 MiB for these closing writes before new work.
        if hasattr(self, "_raw") and self._raw.id.valid:
            used = self._raw.id.get_filesize() + self.ledger_path.stat().st_size
            if (
                used + self.static_bytes + 2 * estimate + 128 * 1024
                > self.config["budgets"]["storage_bytes"]
            ):
                raise BudgetExhausted("Transfer persistence storage ceiling exhausted")
        return super().save_arrays(tag, **arrays)

    def _query(self, mode: str, controls: int, frame_skip: int, *args: Any, **kwargs: Any):
        self.check_limits()
        if self.stage != "d2" or mode != "live":
            raise ValueError("Only D2 native live integration is permitted")
        if (
            self._counts["requested_physics_substeps"] + controls * frame_skip
            > self.config["d2_budgets"]["physics_substeps"]
        ):
            raise BudgetExhausted("D2 physics substep ceiling exhausted")
        return super()._query(mode, controls, frame_skip, *args, **kwargs)

    def start_branch(self, **metadata: Any) -> None:
        self.check_limits()
        if self._counts["branches"] >= self.config["d2_budgets"]["branches"]:
            raise BudgetExhausted("D2 branch ceiling exhausted")
        self._counts["branches"] += 1
        self.record("transfer_branch_start", **metadata)

    def totals(self) -> dict[str, Any]:
        return dict(self.final_work) if self.final_work is not None else super().totals()

    def close(self) -> None:
        if not self.closed:
            super().close()
            self.final_work = super().totals()


def replay_check(actual: dict, expected: dict, *, atol: float, rtol: float) -> dict:
    """Numerical continuation agreement; snapshot restoration is separately exact."""
    maxima, passed = {}, True
    for key in ("states", "warmstarts", "integrations", "costs"):
        left, right = np.asarray(actual[key]), np.asarray(expected[key])
        same_shape = left.shape == right.shape
        maxima[key] = float(np.max(np.abs(left - right))) if same_shape and left.size else None
        passed = (
            passed
            and same_shape
            and bool(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False))
        )
    return dict(passed=passed, max_abs=maxima, atol=atol, rtol=rtol)


def prefix_outcomes(trace: dict, baseline: dict, horizons: list[int]) -> list[dict]:
    """Additive finite-window cost and observed poststate deviations, without padding."""
    rows, count = [], len(trace["costs"])
    for horizon in horizons:
        executed = min(horizon, count)
        done = bool(np.any(trace["done_flags"][:executed]))
        native = float(np.sum(trace["costs"][:executed], dtype=np.float64))
        terminal = float(np.sum(trace["fall_costs"][:executed], dtype=np.float64))
        absorbing = float(5 * (horizon - executed)) if done else 0.0
        base_cost = float(np.sum(baseline["costs"][:horizon], dtype=np.float64))
        row = dict(
            horizon=horizon,
            executed=executed,
            done=done,
            native_cost=native,
            raw_cost=native - terminal,
            terminal_cost=terminal,
            absorbing_cost=absorbing,
            complete_cost=native + absorbing,
            delta_cost=native + absorbing - base_cost,
        )
        for name, key in (
            ("state_rms", "states"),
            ("warmstart_rms", "warmstarts"),
            ("integration_rms", "integrations"),
        ):
            row[name] = (
                float(np.sqrt(np.mean((trace[key][horizon] - baseline[key][horizon]) ** 2)))
                if count >= horizon
                else None
            )
        rows.append(row)
    return rows


def run_branch(env: Any, meter: TransferMeter, anchor: dict, branch: dict) -> tuple[dict, dict]:
    from ..runtime import restore, snapshot

    name = branch["name"]
    baseline = name == "teacher"
    identity = dict(
        teacher_id=anchor["teacher_id"], anchor=anchor["step"], name=name, baseline=baseline
    )
    meter.context = dict(stage="d2", **identity)
    meter.start_branch(**identity)
    template = snapshot(env)
    initial = dict(anchor["initial"], spec=template["spec"], dimensions=template["dimensions"])
    restore(env, initial)
    if not np.array_equal(env._get_obs(), anchor["observation"]):
        raise ValueError("Restored observation differs from the bound prediction input")
    trace = {
        key: []
        for key in ("observations", "actions", "costs", "velocities", "done_flags", "fall_costs")
    }
    trace.update(
        states=[initial["state"]],
        warmstarts=[initial["warmstart"]],
        integrations=[initial["integration"]],
    )
    failure = None
    done = False
    try:
        for index, teacher_action in enumerate(anchor["teacher_actions"]):
            meter.context = dict(stage="d2", step=index, **identity)
            action = np.asarray(branch["action"] if index == 0 else teacher_action, dtype=float)
            observation = env._get_obs().copy()
            _, cost, done, info = env.step(action)
            current = snapshot(env)
            trace["observations"].append(observation)
            trace["actions"].append(action.copy())
            trace["costs"].append(float(cost))
            trace["velocities"].append(float(info["x_velocity"]))
            trace["done_flags"].append(bool(done))
            trace["fall_costs"].append(float(info.get("fall_cost", 0.0)))
            for key, source in (
                ("states", "state"),
                ("warmstarts", "warmstart"),
                ("integrations", "integration"),
            ):
                trace[key].append(current[source])
            if done:
                break
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        arrays = {key: np.asarray(value) for key, value in trace.items()}
        raw = meter.save_arrays(f"branch_{anchor['teacher_id']}_{anchor['step']}_{name}", **arrays)
        record = dict(
            **identity,
            steps=len(trace["costs"]),
            done=bool(done),
            raw=raw,
            complete=failure is None,
            error=failure,
            action_mse=float(np.mean((branch["action"] - anchor["teacher_actions"][0]) ** 2)),
            action_provenance=branch.get("provenance", {}),
            policy_hash=branch.get("policy_hash"),
        )
        meter.record("transfer_branch_error" if failure else "transfer_branch_integrated", **record)
    return record, arrays


def run_d2(inputs: Path, meter: TransferMeter, config: dict, result: dict) -> None:
    from src.envs.walker2d import Walker2d

    from ..runtime import snapshot
    from ..transfer_inputs import load_fork_inputs

    anchors = load_fork_inputs(inputs, random_seed=config["random_seed"])
    if [(a["teacher_id"], a["step"]) for a in anchors] != [
        (teacher, step) for teacher in config["teacher_ids"] for step in config["anchors"]
    ]:
        raise ValueError("D2 anchor roster differs from the frozen recipe")
    for anchor in anchors:
        if [b["name"] for b in anchor["branches"]] != config["branch_names"]:
            raise ValueError("D2 branch roster differs from the frozen recipe")
        if len(anchor["teacher_actions"]) != config["branch_controls"]:
            raise ValueError("D2 continuation length changed")
    result["action_roster"] = meter.save_arrays(
        "d2_action_roster",
        teacher_ids=np.asarray([a["teacher_id"] for a in anchors]),
        anchors=np.asarray([a["step"] for a in anchors]),
        observations=np.asarray([a["observation"] for a in anchors]),
        branch_names=np.asarray(config["branch_names"]),
        first_actions=np.asarray([[b["action"] for b in a["branches"]] for a in anchors]),
    )
    if not config["task"]["apply_terminal_cost_on_done"]:
        raise ValueError("D2 requires the native once-only fall terminal charge")
    env = Walker2d(**config["task"])
    meter.instrument(env)
    meter.record("native_environment_constructed", initial=snapshot(env), integration_steps=0)
    baselines = {}
    try:
        for anchor in anchors:
            record, trace = run_branch(env, meter, anchor, anchor["branches"][0])
            record["replay"] = replay_check(
                trace, anchor["expected"], atol=config["replay_atol"], rtol=config["replay_rtol"]
            )
            record["prefixes"] = prefix_outcomes(trace, trace, config["horizons"])
            result["baselines"].append(record)
            meter.record(
                "transfer_baseline_complete",
                teacher_id=anchor["teacher_id"],
                anchor=anchor["step"],
                replay_passed=record["replay"]["passed"],
                record=record,
            )
            if record["done"] or not record["replay"]["passed"]:
                raise ValueError(
                    "D2 unmodified replay gate failed; no substituted action permitted"
                )
            baselines[(anchor["teacher_id"], anchor["step"])] = trace
        result["baseline_gate_passed"] = True
        meter.record("transfer_baseline_gate", passed=True, baselines=len(result["baselines"]))
        for anchor in anchors:
            baseline = baselines[(anchor["teacher_id"], anchor["step"])]
            for branch in anchor["branches"][1:]:
                record, trace = run_branch(env, meter, anchor, branch)
                record["prefixes"] = prefix_outcomes(trace, baseline, config["horizons"])
                result["branches"].append(record)
                meter.record("transfer_branch_complete", **record)
        result["execution_complete"] = True
    finally:
        env.close()


def runtime_identity() -> dict:
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "mujoco", "h5py", "torch")
        },
    )


def prepare(out: Path, k1: Path, w2: Path) -> dict:
    from ..transfer_inputs import prepare_inputs

    config = load_recipe("transfer_diagnosis")
    for label, parent in (("k1", k1), ("w2", w2)):
        for name, expected in config["parent_metadata_sha256"][label].items():
            if sha(parent / name) != expected:
                raise ValueError(
                    "Parent metadata differs from the published capsule: " + label + "/" + name
                )
    task = json.loads((w2 / "inputs/task.json").read_text())
    if task["xml_sha256"] != sha(ROOT / "assets/walker2d.xml"):
        raise ValueError("Task XML differs from the input lineage")
    config.update(
        task=task["task"],
        xml_sha256=task["xml_sha256"],
        k1_run=str(k1.resolve()),
        w2_run=str(w2.resolve()),
    )
    out = freeze(out, config, "transfer-diagnosis")
    inventory = prepare_inputs(k1, w2, out / "inputs")
    write(
        out / "input_manifest.json",
        dict(
            files={
                str(path.relative_to(out)): dict(sha256=sha(path), bytes=path.stat().st_size)
                for path in sorted((out / "inputs").rglob("*"))
                if path.is_file()
            },
            config_sha256=sha(out / "config.json"),
            source_manifest_sha256=sha(out / "manifest.json"),
            planned_work=dict(
                support_queries=config["expected"]["support_queries"],
                native_controls=24000,
                physics_substeps=96000,
                branches=240,
                neural_rows=0,
                backward_rows=0,
                optimizer_steps=0,
                model_controls=0,
            ),
        ),
    )
    return dict(
        prepared=str(out),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        counts=inventory["counts"],
    )


def worker(out: Path) -> dict:
    from ..transfer_metrics import analyze_inputs

    started = time.perf_counter()
    verify(out)
    seal = json.loads((out / "input_manifest.json").read_text())
    if seal["source_manifest_sha256"] != sha(out / "manifest.json") or seal["config_sha256"] != sha(
        out / "config.json"
    ):
        raise ValueError("Transfer source/config seal changed")
    for name, binding in seal["files"].items():
        path = (out / name).resolve()
        if (
            not path.is_relative_to(out.resolve())
            or sha(path) != binding["sha256"]
            or path.stat().st_size != binding["bytes"]
        ):
            raise ValueError("Transfer input binding changed: " + name)
    config = json.loads((out / "config.json").read_text())
    identity = runtime_identity()
    expected = config["expected_runtime"]
    if (
        identity["python"] != expected["python"]
        or identity["packages"] != expected["packages"]
        or sys.platform != "linux"
    ):
        raise ValueError("D1/D2 must use the frozen native Linux CPU runtime")
    static_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    result = dict(
        schema="walker-transfer-diagnosis-result-v1",
        execution_complete=False,
        error=None,
        status="incomplete",
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        runtime=identity,
        d1={},
        d2=dict(
            evidence_directory="evidence",
            baseline_gate_passed=False,
            execution_complete=False,
            baselines=[],
            branches=[],
        ),
    )
    meter = TransferMeter(out / "evidence", config, static_bytes)
    try:
        meter.begin_stage("d1")
        result["d1"] = analyze_inputs(out / "inputs", meter, config)
        result["d1"]["evidence_directory"] = "evidence"
        meter.check_limits()
        stage_summary = meter.stage_summary()
        result["d1"].update(stage_summary)
        if meter._counts["support_queries"] != config["expected"]["support_queries"]:
            raise ValueError("D1 support query census differs from the frozen roster")
        meter._counts["completed_support_queries"] = meter._counts["support_queries"]
        meter.record(
            "transfer_stage_complete",
            stage="d1",
            support_queries=meter._counts["support_queries"],
            **stage_summary,
        )
        meter.begin_stage("d2")
        run_d2(out / "inputs", meter, config, result["d2"])
        meter.check_limits()
        stage_summary = meter.stage_summary()
        result["d2"].update(stage_summary)
        meter.record("transfer_stage_complete", stage="d2", **stage_summary)
        result.update(execution_complete=True, status="diagnosis_complete")
    except BaseException as error:
        result["error"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
        result["status"] = (
            "stopped_invalid" if not result["d2"]["baseline_gate_passed"] else "stopped_partial"
        )
        meter.record(
            "transfer_stopped", error=result["error"], stage=meter.stage, **meter.stage_summary()
        )
    finally:
        meter.close()
        result["work"] = meter.totals()
        result["wall_seconds"] = time.perf_counter() - started
        write(out / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--k1", type=Path, required=True)
    item.add_argument("--w2", type=Path, required=True)
    item = sub.add_parser("worker")
    item.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        prepare(args.out, args.k1, args.w2)
        if args.command == "prepare"
        else worker(args.out.resolve())
    )
    print(json.dumps(result, default=default, allow_nan=False), flush=True)
    return int(bool(result.get("error")))


if __name__ == "__main__":
    raise SystemExit(main())
