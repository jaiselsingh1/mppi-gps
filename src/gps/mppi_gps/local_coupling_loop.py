"""Frozen L2 lifecycle: actual local teachers, matched learning and feedback.

This coordinator composes the unchanged planner, native recorder and W2 fitter.
The two learning blocks are conditional on prespecified scientific outcomes;
all episodes within an authorized controller roster run, including failures.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .provenance import write
from .trajectory_pi_accounting import directory_bytes
from .trajectory_pi_native import binding, load_trace

MODULE = "src.gps.mppi_gps.local_coupling_loop"


def bound(spec: dict) -> Path:
    path = Path(spec["path"])
    if binding(path) != spec:
        raise ValueError("Frozen file changed: " + str(path))
    return path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def native_preflight(job: dict) -> dict:
    """One four-control capture/repeat and K1/H4 path check, with raw evidence."""
    import mujoco

    from .runtime import snapshot
    from .trajectory_pi_accounting import Work
    from .trajectory_pi_native import _env, _restore, execute_trace

    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "job.json", job)
    work = Work(out, job["limits"])
    env = None
    result = dict(schema="local-coupling-native-prerequisite-1", status="FAILED_STOPPED")
    try:
        env = _env(dict(task=job["task"], physics_threads=4), work)
        model = out / "compiled.mjb"
        work.call(
            "save_compiled_model",
            dict(compiled_model_saves=1),
            lambda: mujoco.mj_saveModel(env.model, str(model), None),
        )
        options = {
            name: np.asarray(getattr(env.model.opt, name)).tolist()
            for name in dir(env.model.opt)
            if not name.startswith("_")
            and isinstance(getattr(env.model.opt, name), (int, float, np.number, np.ndarray))
        }
        write(out / "model_options.json", options)
        work.record(
            "native_model_bound_before_physics",
            model=binding(model),
            options=binding(out / "model_options.json"),
        )
        with np.load(bound(job["restarts"]), allow_pickle=False) as raw:
            context = {
                key: raw[f"r0_{key}"].copy()
                for key in ("state", "warmstart", "integration", "spec", "dimensions")
            }
        with np.load(bound(job["student_episode"]), allow_pickle=False) as raw:
            controls = raw["actions"][1:5].copy()
        traces = []
        for name in ("capture", "repeat"):
            _restore(env, context, work)
            execute_trace(env, work, out / f"{name}.h5", 4, actions=controls)
            traces.append(load_trace(out / f"{name}.h5"))
        for name in ("states", "warmstarts", "integrations", "observations", "actions", "costs"):
            if not np.allclose(traces[0][name], traces[1][name], rtol=0, atol=1e-10):
                raise ValueError("Strict same-host prerequisite failed: " + name)
        _restore(env, context, work)
        posts, costs, _ = env.batch_rollout(
            context["state"], controls[None], initial_warmstart=context["warmstart"]
        )
        after = snapshot(env)
        with (out / "candidate.npz").open("xb") as target:
            np.savez_compressed(
                target,
                posts=posts,
                costs=costs,
                actions=controls,
                **{f"initial_{key}": context[key] for key in ("state", "warmstart", "integration")},
                **{f"after_{key}": after[key] for key in ("state", "warmstart", "integration")},
            )
        if not np.allclose(posts[0], traces[0]["states"][1:], rtol=0, atol=1e-10) or not np.isclose(
            costs[0], traces[0]["costs"].sum(), rtol=0, atol=1e-9
        ):
            raise ValueError("Native/candidate prerequisite mismatch")
        if any(
            not np.array_equal(context[key], after[key])
            for key in ("state", "warmstart", "integration")
        ):
            raise ValueError("Candidate forecast mutated complete live context")
        result.update(
            status="COMPLETED",
            model=binding(model),
            model_options=binding(out / "model_options.json"),
            capture=binding(out / "capture.h5"),
            repeat=binding(out / "repeat.h5"),
            candidate=binding(out / "candidate.npz"),
        )
    except BaseException as error:
        result["error"] = repr(error)
    finally:
        if env is not None:
            env.close()
    receipt = work.close("COMPLETED" if result["status"] == "COMPLETED" else "FAILED")
    write(out / "result.json", result)
    return dict(path=str(out), result=result, receipt=receipt)


def teacher_summary(episodes: list[dict]) -> dict:
    if sorted(row["slot"] for row in episodes) != list(range(8)):
        raise ValueError("Complete eight-restart executed roster required")
    outcomes = [row["outcome"] for row in episodes]
    return dict(
        local_walking=sum(row["local_walking"] for row in outcomes),
        full_survival=sum(row["full_survival"] for row in outcomes),
        complete_cost=float(np.mean([row["complete_cost"] for row in outcomes])),
        velocity_mae=float(np.mean([row["velocity_mae"] for row in outcomes])),
        controls=sum(row["controls"] for row in outcomes),
    )


def action_change(first: list[dict], second: list[dict]) -> dict:
    cases, squares = [], []
    for a, b in zip(
        sorted(first, key=lambda r: r["slot"]), sorted(second, key=lambda r: r["slot"]), strict=True
    ):
        if a["slot"] != b["slot"]:
            raise ValueError("Paired restart ownership differs")
        x, y = load_trace(bound(a["trace"]))["actions"], load_trace(bound(b["trace"]))["actions"]
        n = min(len(x), len(y))
        if not n:
            raise ValueError("No common executed source prefix")
        delta = (x[:n] - y[:n]) ** 2
        squares.append(delta.ravel())
        cases.append(
            dict(slot=a["slot"], common_controls=n, action_rms=float(np.sqrt(delta.mean())))
        )
    return dict(pooled_rms=float(np.sqrt(np.concatenate(squares).mean())), cases=cases)


def select_teacher(rosters: dict[str, list[dict]], config: dict) -> dict:
    """Only executed competence and intervention effectiveness select alpha."""
    baseline = teacher_summary(rosters["independent"])
    comparisons = {}
    rule = config["selection"]
    for arm in rule["priority"]:
        measured = teacher_summary(rosters[arm])
        effect = action_change(rosters["independent"], rosters[arm])
        gates = dict(
            baseline_competent=baseline["local_walking"] >= rule["baseline_required_local_walking"],
            positive_competent=measured["local_walking"] >= rule["positive_required_local_walking"],
            walking_nonregression=measured["local_walking"] >= baseline["local_walking"],
            survival_nonregression=measured["full_survival"] >= baseline["full_survival"],
            cost_noninferiority=measured["complete_cost"]
            <= rule["complete_cost_ratio_max"] * baseline["complete_cost"],
            effective_intervention=effect["pooled_rms"] >= rule["action_rms_min"],
        )
        comparisons[arm] = dict(
            outcomes=measured, effect=effect, gates=gates, eligible=all(gates.values())
        )
    selected = next((arm for arm in rule["priority"] if comparisons[arm]["eligible"]), None)
    alphas = {row["name"]: row["alpha"] for row in config["controller"]["arms"]}
    return dict(
        schema="local-coupling-selection-1",
        baseline=baseline,
        comparisons=comparisons,
        selected_condition=selected,
        alpha=alphas[selected] if selected else None,
        selected_before_fitting=True,
    )


def panel_counts(records: list[dict], *, stream: int, arm: str, kind: str, panel: str) -> dict:
    rows = [
        row
        for row in records
        if (row["stream"], row["arm"], row["kind"], row["panel"]) == (stream, arm, kind, panel)
    ]
    if len(rows) != 16:
        raise ValueError("Complete frozen sixteen-reset evaluation panel required")
    outcomes = [read(bound(row["result"]))["outcome"] for row in rows]
    return dict(
        walking=sum(row["walking"] for row in outcomes),
        full_survival=sum(row["full_survival"] for row in outcomes),
        complete_cost=float(np.mean([row["complete_cost"] for row in outcomes])),
        controls=sum(row["controls"] for row in outcomes),
    )


def walking_criterion(records: list[dict], round_index: int) -> dict:
    winners, details, safe = [], [], True
    for stream in range(3):
        wins = []
        for panel in ("primary", "confirmation"):
            zero = panel_counts(
                records, stream=stream, arm="independent", kind="child", panel=panel
            )
            positive = panel_counts(
                records, stream=stream, arm="positive", kind="child", panel=panel
            )
            if round_index == 1:
                parent_zero = parent_positive = panel_counts(
                    records, stream=-1, arm="parent", kind="parent", panel=panel
                )
            else:
                parent_zero = panel_counts(
                    records, stream=stream, arm="independent", kind="parent", panel=panel
                )
                parent_positive = panel_counts(
                    records, stream=stream, arm="positive", kind="parent", panel=panel
                )
            gain_positive = positive["walking"] - parent_positive["walking"]
            gain_zero = zero["walking"] - parent_zero["walking"]
            nonregression = all(
                positive[key] >= other[key]
                for key in ("walking", "full_survival")
                for other in (zero, parent_positive)
            )
            safe = safe and nonregression
            win = gain_positive >= 2 and positive["walking"] >= zero["walking"] + 2
            if round_index == 2:
                win = win and gain_positive - gain_zero >= 2
            wins.append(win)
            details.append(
                dict(
                    stream=stream,
                    panel=panel,
                    independent=zero,
                    positive=positive,
                    incoming_independent=parent_zero,
                    incoming_positive=parent_positive,
                    walking_gain_positive=gain_positive,
                    walking_gain_independent=gain_zero,
                    difference_in_gain=gain_positive - gain_zero,
                    nonregression=nonregression,
                    qualifying_gain=win,
                )
            )
        if all(wins):
            winners.append(stream)
    return dict(
        passed=bool(safe and len(winners) >= 2),
        winning_streams=winners,
        nonregression_all=safe,
        details=details,
    )


def child_limits(config: dict, counts: dict) -> dict:
    limits = config["budgets"]["global"]
    result = dict(
        wall_seconds=config["budgets"]["worker"]["wall_seconds"],
        output_bytes=config["budgets"]["worker"]["output_bytes"],
        memory_bytes=limits["memory_bytes"],
    )
    for unit in (
        "model_controls",
        "native_controls",
        "physics_substeps",
        "neural_rows",
        "backward_rows",
        "optimizer_steps",
    ):
        result[unit] = limits[unit] - counts.get("requested_" + unit, 0)
    for unit in (
        "checkpoint_loads",
        "checkpoint_writes",
        "policy_constructions",
        "optimizer_constructions",
        "environment_constructions",
        "model_queries",
        "restores",
        "external_mj_forward_calls",
        "resets",
        "compiled_model_saves",
        "pi_passes",
    ):
        result[unit] = 100_000
    return result


class Study:
    def __init__(self, launch: dict):
        self.launch = launch
        self.base = Path(launch["base_directory"])
        self.root = self.base / "worker"
        self.root.mkdir(exist_ok=False)
        self.config = read(bound(launch["config"]))
        self.prepared = read(bound(launch["prepared_manifest"]))
        self.started = time.monotonic()
        self.counts = dict(self.prepared.get("engineering_work", {}))
        self.receipts: list[dict] = []
        self.snapshot_index = 0
        self.ledger = (self.root / "study_ledger.jsonl").open("x")
        self.result = dict(
            schema="local-coupling-learning-l2-result-1",
            status="RUNNING",
            config=launch["config"],
            prepared_manifest=launch["prepared_manifest"],
            source_manifest=launch["source_manifest"],
            launch=binding(self.base / "launch.json"),
            acquisitions=[],
            rounds=[],
            accounting={},
        )
        write(self.root / "config.json", self.config)
        write(self.root / "launch.json", launch)

    def check(self):
        budget = self.config["budgets"]
        if time.monotonic() - self.started > budget["worker"]["wall_seconds"]:
            raise RuntimeError("Worker wall budget exhausted")
        if time.process_time() > budget["worker"]["cpu_seconds"]:
            raise RuntimeError("Worker CPU budget exhausted")
        if directory_bytes(self.base) + 1024**2 > budget["worker"]["output_bytes"]:
            raise RuntimeError("Execution output reserve exhausted")
        for unit, maximum in budget["global"].items():
            if "requested_" + unit in self.counts and self.counts["requested_" + unit] > maximum:
                raise RuntimeError("Global work ceiling exhausted: " + unit)

    def include(self, path: Path, receipt: dict):
        seal = binding(path)
        if any(row["path"] == seal["path"] for row in self.receipts):
            raise ValueError("Duplicate child work import")
        self.receipts.append(seal)
        for key, value in receipt["counts"].items():
            self.counts[key] = self.counts.get(key, 0) + value
        self.ledger.write(
            json.dumps(
                dict(event="closed_child_included", receipt=seal, counts=receipt["counts"]),
                sort_keys=True,
            )
            + "\n"
        )
        self.ledger.flush()
        os.fsync(self.ledger.fileno())
        self.check()

    def job(self, function, job):
        self.check()
        job["config"] = self.launch["config"]
        job["limits"] = child_limits(self.config, self.counts)
        remaining = (
            self.config["budgets"]["worker"]["output_bytes"]
            - directory_bytes(self.base)
            - 2 * 1024**2
        )
        if remaining < 64 * 1024**2:
            raise RuntimeError("Insufficient reserved output for the next complete job")
        job["limits"]["output_bytes"] = min(64 * 1024**2, remaining)
        value = function(job)
        self.include(Path(value["path"]) / "receipt.json", value["receipt"])
        if value["result"]["status"] != "COMPLETED":
            raise RuntimeError("Closed job failed: " + str(value["result"]))
        return value["result"]

    def checkpoint(self):
        self.result["accounting"] = dict(
            child_receipts=self.receipts,
            counts=self.counts,
            worker_process_cpu_seconds=time.process_time(),
            worker_wall_seconds=time.monotonic() - self.started,
        )
        # Immutable progress records preserve every paid prefix. Only the
        # convenience pointer changes; no result or frozen source is replaced.
        target = self.root / "progress" / f"{self.snapshot_index:06d}.json"
        write(target, self.result)
        temporary = self.root / "result.next"
        temporary.symlink_to(target.relative_to(self.root))
        os.replace(temporary, self.root / "result.json")
        self.snapshot_index += 1

    def acquire(self, round_index, stream, arm, alpha, checkpoint):
        from .local_coupling_controller import run_episode

        rows = []
        for slot in range(8):
            out = (
                self.root
                / "acquisition"
                / f"r{round_index}"
                / ("shared" if stream == -1 else f"stream{stream}")
                / arm
                / f"slot{slot}"
            )
            value = self.job(
                run_episode,
                dict(
                    out=str(out),
                    round=round_index,
                    stream=stream,
                    arm=arm,
                    slot=slot,
                    alpha=alpha,
                    seed=self.config["seeds"]["planner_base"] + slot,
                    checkpoint=checkpoint,
                    restarts=self.prepared["restarts"],
                    controls=300,
                    planner=self.config["controller"]["planner"],
                    task=self.config["controller"]["task"],
                    physics_threads=4,
                    replay_episode=slot == 0,
                ),
            )
            self.result["acquisitions"].append(
                dict(
                    round=round_index,
                    stream=stream,
                    arm=arm,
                    slot=slot,
                    manifest=binding(out / "manifest.json"),
                )
            )
            rows.append(value)
            self.checkpoint()
            print(
                json.dumps(
                    dict(
                        stage="acquisition",
                        round=round_index,
                        stream=stream,
                        arm=arm,
                        slot=slot,
                        **value["outcome"],
                    )
                ),
                flush=True,
            )
        return rows

    def evaluate(self, block, stream, arm, kind, checkpoint):
        from .local_coupling_learning import evaluate_job

        for panel in ("primary", "confirmation"):
            seed_base = self.config["seeds"][f"round{block['round']}_{panel}_base"]
            for episode in range(16):
                seed = seed_base + episode
                out = (
                    self.root
                    / "evaluations"
                    / f"r{block['round']}"
                    / f"s{stream}_{arm}_{kind}"
                    / panel
                    / str(seed)
                )
                self.job(
                    evaluate_job,
                    dict(
                        out=str(out),
                        checkpoint=checkpoint,
                        reset_seed=seed,
                        task=self.config["controller"]["task"],
                        physics_threads=4,
                        role=dict(
                            round=block["round"], stream=stream, arm=arm, kind=kind, panel=panel
                        ),
                    ),
                )
                block["evaluations"].append(
                    dict(
                        stream=stream,
                        arm=arm,
                        kind=kind,
                        panel=panel,
                        reset_seed=seed,
                        result=binding(out / "result.json"),
                    )
                )
            self.checkpoint()

    def learning_round(self, round_index, rosters, incoming):
        from .local_coupling_learning import build_paired_corpora, fit_job

        block = dict(
            round=round_index,
            corpora=[],
            fits=[],
            evaluations=[],
            incoming_streams=[],
            outgoing_streams=[],
        )
        self.result["rounds"].append(block)
        corpora = {}
        for stream in range(3):
            corpus_stream = -1 if round_index == 1 else stream
            if corpus_stream not in corpora:
                a, b = rosters[corpus_stream]
                out = (
                    self.root
                    / "corpus"
                    / f"r{round_index}"
                    / ("shared" if corpus_stream == -1 else f"stream{stream}")
                )
                corpus = build_paired_corpora(
                    [dict(slot=row["slot"], trace=row["trace"]) for row in a],
                    [dict(slot=row["slot"], trace=row["trace"]) for row in b],
                    out,
                    config_binding=self.launch["config"],
                )
                corpora[corpus_stream] = corpus
                block["corpora"].append(
                    dict(stream=corpus_stream, manifest=binding(out / "manifest.json"))
                )
            for arm, data_arm, cross in (("independent", "a", "b"), ("positive", "b", "a")):
                stream_spec = incoming[(stream, arm)]
                block["incoming_streams"].append(dict(stream=stream, arm=arm, envelope=stream_spec))
                if round_index == 2:
                    self.evaluate(
                        block,
                        stream,
                        arm,
                        "parent",
                        binding(Path(stream_spec["path"]).parent / "policy.pt"),
                    )
                corpus = corpora[corpus_stream]
                out = self.root / "fits" / f"r{round_index}" / f"s{stream}_{arm}"
                value = self.job(
                    fit_job,
                    dict(
                        out=str(out),
                        incoming=stream_spec,
                        corpus=corpus["files"][data_arm],
                        round=round_index,
                        stream=stream,
                        arm=arm,
                        diagnostics=dict(
                            own=corpus["files"][data_arm],
                            cross=corpus["files"][cross],
                            held=self.prepared["held"],
                        ),
                    ),
                )
                block["fits"].append(
                    dict(stream=stream, arm=arm, result=binding(out / "result.json"))
                )
                block["outgoing_streams"].append(
                    dict(stream=stream, arm=arm, envelope=value["outgoing"])
                )
                self.evaluate(
                    block,
                    stream,
                    arm,
                    "child",
                    binding(Path(value["outgoing"]["path"]).parent / "policy.pt"),
                )
        if round_index == 1:
            self.evaluate(block, -1, "parent", "parent", self.prepared["checkpoint"])
        block["criterion"] = walking_criterion(block["evaluations"], round_index)
        self.checkpoint()
        print(
            json.dumps(dict(stage="round_closed", round=round_index, criterion=block["criterion"])),
            flush=True,
        )
        return block

    def run(self):
        try:
            self.job(
                native_preflight,
                dict(
                    out=str(self.root / "native_preflight"),
                    restarts=self.prepared["restarts"],
                    student_episode=self.prepared["student_episode"],
                    task=self.config["controller"]["task"],
                ),
            )
            self.result["native_preflight"] = binding(self.root / "native_preflight/result.json")
            rosters = {}
            for arm in self.config["controller"]["arms"]:
                rosters[arm["name"]] = self.acquire(
                    1, -1, arm["name"], arm["alpha"], self.prepared["checkpoint"]
                )
            selection = select_teacher(rosters, self.config)
            write(self.root / "selection.json", selection)
            self.result["selection"] = binding(self.root / "selection.json")
            if selection["selected_condition"] is None:
                self.result["status"] = "NO_COMPETENT_EFFECTIVE_TEACHER_CONTRAST"
            else:
                incoming = {
                    (s, arm): binding(Path(self.prepared["forks"][s]["directory"]) / "stream.json")
                    for s in range(3)
                    for arm in ("independent", "positive")
                }
                first = self.learning_round(
                    1,
                    {-1: (rosters["independent"], rosters[selection["selected_condition"]])},
                    incoming,
                )
                if not first["criterion"]["passed"]:
                    self.result["status"] = "ROUND1_WALKING_NOT_CONFIRMED"
                else:
                    self.result["status"] = "ROUND1_CLOSED_AWAITING_AUDIT"
                    self.checkpoint()
                    r1_snapshot = self.root / "r1_result.json"
                    write(r1_snapshot, self.result)
                    self.result["r1_snapshot"] = binding(r1_snapshot)
                    prefix = self.root / "r1_ledger_prefix.jsonl"
                    with prefix.open("xb") as target:
                        target.write((self.root / "study_ledger.jsonl").read_bytes())
                    self.result["r1_ledger_prefix"] = binding(prefix)
                    admission = invoke_audit(self.base, stage="r1")
                    if not admission.get("passed"):
                        raise ValueError("Independent R1 admission audit failed")
                    self.result["r1_admission"] = binding(self.base / "audit/r1/receipt.json")
                    incoming = {
                        (r["stream"], r["arm"]): r["envelope"] for r in first["outgoing_streams"]
                    }
                    second_rosters = {}
                    for s in range(3):
                        second_rosters[s] = tuple(
                            self.acquire(
                                2,
                                s,
                                arm,
                                alpha,
                                binding(Path(incoming[(s, arm)]["path"]).parent / "policy.pt"),
                            )
                            for arm, alpha in (
                                ("independent", 0.0),
                                ("positive", selection["alpha"]),
                            )
                        )
                    second = self.learning_round(2, second_rosters, incoming)
                    checks = []
                    for s in range(3):
                        current = teacher_summary(second_rosters[s][1])
                        references = [
                            teacher_summary(rosters[selection["selected_condition"]]),
                            teacher_summary(second_rosters[s][0]),
                        ]
                        safety = all(
                            current["local_walking"] >= r["local_walking"]
                            and current["full_survival"] >= r["full_survival"]
                            and current["complete_cost"] <= 1.1 * r["complete_cost"]
                            for r in references
                        )
                        effect = action_change(
                            rosters[selection["selected_condition"]], second_rosters[s][1]
                        )
                        checks.append(
                            dict(
                                stream=s,
                                competent=safety,
                                effect=effect,
                                changed=effect["pooled_rms"] >= 0.01,
                            )
                        )
                    second["teacher_feedback"] = checks
                    wins = second["criterion"]["winning_streams"]
                    useful = (
                        second["criterion"]["passed"]
                        and all(r["competent"] for r in checks)
                        and sum(r["changed"] and r["stream"] in wins for r in checks) >= 2
                    )
                    self.result["useful_feedback"] = bool(useful)
                    self.result["status"] = (
                        "TWO_USEFUL_ROUNDS" if useful else "ROUND2_NOT_CONFIRMED"
                    )
        except BaseException as error:
            self.result.update(status="FAILED_STOPPED", error=repr(error))
        finally:
            self.checkpoint()
            self.ledger.close()
        return self.result


def invoke_audit(base: Path, *, stage: str) -> dict:
    launch = read(base / "launch.json")
    config = read(bound(launch["config"]))
    limits = config["budgets"]["audit"]
    used = (
        read(base / "audit/r1/receipt.json")
        if stage == "final" and (base / "audit/r1/receipt.json").exists()
        else {}
    )
    external = (
        read(base / "audit/r1/external.json")
        if used and (base / "audit/r1/external.json").exists()
        else used
    )
    reserve = limits["report_publication_reserve"]
    remaining_wall = (
        limits["wall_seconds"] - reserve["wall_seconds"] - external.get("wall_seconds", 0)
    )
    remaining_cpu = limits["cpu_seconds"] - reserve["cpu_seconds"] - external.get("cpu_seconds", 0)
    remaining_rows = limits["neural_rows"] - used.get("completed_neural_rows", 0)
    if stage == "r1":
        cap = limits["intermediate_r1_max"]
        remaining_wall = min(remaining_wall, cap["wall_seconds"])
        remaining_cpu = min(remaining_cpu, cap["cpu_seconds"])
        remaining_rows = min(remaining_rows, cap["neural_rows"])
    command = [
        sys.executable,
        "-B",
        "-m",
        "src.gps.mppi_gps.audits.local_coupling_learning",
        "--run",
        str(base / "worker"),
        "--out",
        str(base / "audit" / stage),
        "--stage",
        stage,
        "--seconds",
        str(remaining_wall),
        "--cpu-seconds",
        str(remaining_cpu),
        "--neural-rows",
        str(remaining_rows),
    ]
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    completed = subprocess.run(
        command, cwd=launch["source_directory"], timeout=remaining_wall, check=False
    )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(
        base / "audit" / stage / "external.json",
        dict(
            argv=command,
            exit_code=completed.returncode,
            wall_seconds=time.monotonic() - started,
            cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            scope="Enclosing child process including imports; nested internal audit timers are not added again.",
        ),
    )
    receipt = base / "audit" / stage / "receipt.json"
    value = read(receipt) if receipt.exists() else dict(passed=False)
    if completed.returncode != 0:
        value["passed"] = False
    return value


def verify_launch(path: Path) -> dict:
    launch = read(path)
    if launch.get("schema") != "local-coupling-l2-launch-1":
        raise ValueError("Frozen L2 launch required")
    config = read(bound(launch["config"]))
    if config.get("launch_ready") is not True:
        raise ValueError("Unfrozen scientific configuration")
    source = read(bound(launch["source_manifest"]))
    root = Path(launch["source_directory"])
    if Path(__file__).resolve() != root / "src/gps/mppi_gps/local_coupling_loop.py":
        raise ValueError("Run the immutable source export")
    for row in source["files"]:
        actual = binding(root / row["path"])
        if any(actual[k] != row[k] for k in ("bytes", "sha256")):
            raise ValueError("Source export changed")
    prepared = read(bound(launch["prepared_manifest"]))
    for row in prepared["files"]:
        bound(row)
    environment = read(bound(launch["environment"]))
    for row in [environment["interpreter"], *environment["binary_files"]]:
        bound(row)
    if binding(Path(sys.executable)) != environment["interpreter"]:
        raise ValueError("Wrong execution interpreter")
    return launch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args(argv)
    launch = verify_launch(args.launch)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    limit = read(bound(launch["config"]))["budgets"]["worker"]["cpu_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 5))
    result = Study(launch).run()
    return int(result["status"] == "FAILED_STOPPED")


if __name__ == "__main__":
    raise SystemExit(main())
