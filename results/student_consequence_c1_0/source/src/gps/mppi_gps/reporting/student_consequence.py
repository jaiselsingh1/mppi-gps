"""Render C1's saved scalar evidence without inference, physics or raw reads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

MAX_OUTPUT = 512 * 1024**2
MAX_RSS = 2 * 1024**3
CURVE_FIELDS = (
    "update",
    "fit_update",
    "action_mse",
    "weighted_action_mse",
    "proximal_loss",
    "total_loss",
    "unclipped_gradient_norm",
)
OUTCOME_FIELDS = (
    "row_kind",
    "panel",
    "stream",
    "arm",
    "actor",
    "tag",
    "seed",
    "episodes",
    "walking_successes",
    "full_survivals",
    "walking",
    "survived",
    "controls",
    "mean_steps",
    "cost",
    "mean_cost",
    "native_cost",
    "raw_cost",
    "native_terminal",
    "absorbing_cost",
    "mae",
    "mean_velocity_mae",
    "roster_status",
    "source",
)
ALIAS = re.compile(r"s([0-2])_(parent|uniform|cost_effect|shuffled)(?:_child)?")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rss() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class _Reader:
    def __init__(self):
        self.started = time.monotonic()
        self.inputs: dict[str, dict[str, Any]] = {}
        self.read_bytes = 0
        self.partial_lines: list[dict[str, Any]] = []

    def check(self) -> None:
        if time.monotonic() - self.started > 1800 or _rss() > MAX_RSS:
            raise ValueError("C1 report time/RSS ceiling exhausted")

    def load(self, path: Path, *, lines: bool = False) -> Any:
        self.check()
        if path.suffix not in {".json", ".jsonl"}:
            raise ValueError("C1 reporter accepts saved JSON only")
        size = path.stat().st_size
        if size > 128 * 1024**2 or self.read_bytes + size > 512 * 1024**2:
            raise ValueError("C1 saved report JSON exceeds bounded read allowance")
        payload = path.read_bytes()
        if len(payload) != size:
            raise ValueError("Saved C1 JSON changed during read")
        self.read_bytes += len(payload)
        self.inputs[str(path)] = dict(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )

        def reject(value):
            raise ValueError("Nonfinite saved JSON: " + value)

        if lines:
            rows = []
            for line in payload.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    self.partial_lines.append(
                        dict(
                            file=str(path),
                            bytes=len(line),
                            sha256=hashlib.sha256(line).hexdigest(),
                            availability="incomplete_final_JSONL_fragment",
                        )
                    )
                    break
                if line.strip():
                    rows.append(json.loads(line, parse_constant=reject))
            return rows
        return json.loads(payload, parse_constant=reject)


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _identity(alias: str) -> dict[str, Any]:
    match = ALIAS.fullmatch(alias)
    return dict(stream=int(match[1]), arm=match[2]) if match else dict(stream="", arm="")


def _csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _save(path, stream.getvalue().encode())


def _save(path: Path, payload: bytes) -> None:
    current = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    if current + len(payload) > MAX_OUTPUT - 65536:
        raise ValueError("C1 report output ceiling exhausted before write")
    with path.open("xb") as stream:
        stream.write(payload)


def _table(rows: list[dict], columns: tuple[str, ...]) -> str:
    if not rows:
        return "No saved measurements are available.\n"

    def show(value):
        if value is None or value == "":
            return "missing"
        return str(value).replace("|", "\\|").replace("\n", " ")

    return (
        "\n".join(
            [
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
                *["| " + " | ".join(show(row.get(key)) for key in columns) + " |" for row in rows],
            ]
        )
        + "\n"
    )


def _curves(documents: list[tuple[str, Any]]) -> list[dict]:
    points: dict[tuple[str, int], dict] = {}

    def visit(value, alias, source):
        if isinstance(value, dict):
            named = value.get("tag", value.get("alias", value.get("name", alias)))
            if isinstance(named, str) and ALIAS.fullmatch(named):
                alias = named
            curve = value.get("learning_curve")
            if isinstance(curve, list):
                if len(curve) > 2000:
                    raise ValueError("Saved C1 curve exceeds its fixed update budget")
                if not alias:
                    raise ValueError("Saved C1 curve lacks its actor alias")
                for index, point in enumerate(curve):
                    if not isinstance(point, dict):
                        raise ValueError("Saved C1 curve point must be an object")
                    if any(key in point and not _finite(point[key]) for key in CURVE_FIELDS):
                        raise ValueError("Saved C1 curve contains nonfinite values")
                    update = point.get("fit_update", point.get("update"))
                    if type(update) is not int or update < 1:
                        raise ValueError("Saved C1 curve update identity missing")
                    row = dict(
                        actor=alias, **_identity(alias), **{k: point.get(k) for k in CURVE_FIELDS}
                    )
                    key = alias, update
                    if key in points:
                        if source in points[key]["sources"]:
                            raise ValueError("Repeated learning point within one saved input")
                        if any(points[key].get(k) != row.get(k) for k in CURVE_FIELDS):
                            raise ValueError("Conflicting saved copies of a C1 learning point")
                        points[key]["sources"].add(source)
                    else:
                        row.update(
                            sources={source},
                            point_index=index,
                            failed=bool(value.get("failed", False)),
                        )
                        points[key] = row
                return
            for key, child in value.items():
                next_alias = key if isinstance(key, str) and ALIAS.fullmatch(key) else alias
                visit(child, next_alias, source)
        elif isinstance(value, list):
            for child in value:
                visit(child, alias, source)

    for source, document in documents:
        inferred = Path(source).parent.name
        visit(document, inferred if ALIAS.fullmatch(inferred) else "", source)
    return [
        dict(row, sources=";".join(sorted(row["sources"]))) for _, row in sorted(points.items())
    ]


def _outcomes(panels: dict, evaluations: list[dict], replay: dict) -> list[dict]:
    rows = []
    for panel, streams in panels.items():
        expected = 4 if panel == "source" else 16
        for stream, actors in streams.items():
            for arm, saved in actors.items():
                count = saved.get("episodes")
                if type(count) is not int or count < 1:
                    raise ValueError("Saved C1 cohort lacks observed episode count")
                if (
                    any(
                        type(saved.get(k)) is not int or not 0 <= saved[k] <= count
                        for k in ("successes", "survived")
                    )
                    or saved["successes"] > saved["survived"]
                ):
                    raise ValueError("Saved C1 walking/survival counts are invalid")
                rows.append(
                    dict(
                        row_kind="summary",
                        panel=panel,
                        stream=stream,
                        arm=arm,
                        actor=f"s{stream}_{arm}",
                        episodes=count,
                        walking_successes=saved.get("successes"),
                        full_survivals=saved.get("survived"),
                        controls=saved.get("controls"),
                        mean_steps=saved.get("mean_steps"),
                        mean_cost=saved.get("mean_cost"),
                        mean_velocity_mae=saved.get("mean_velocity_mae"),
                        roster_status="fixed_roster_complete"
                        if count == expected
                        else "observed_prefix_only",
                        source="saved_panel_summary",
                    )
                )
    for panel, events in [("teacher_replay", replay.get("episodes", [])), (None, evaluations)]:
        for saved in events:
            alias = saved.get("actor") or "teacher"
            identity = _identity(alias)
            rows.append(
                dict(
                    row_kind="episode",
                    panel=panel or saved.get("panel", "unknown"),
                    stream=saved.get("stream", identity["stream"]),
                    arm=saved.get("arm", "teacher" if panel else identity["arm"]),
                    actor=alias,
                    tag=saved.get("tag"),
                    seed=saved.get("seed"),
                    walking=saved.get("walking"),
                    survived=saved.get("survived"),
                    controls=saved.get("controls"),
                    cost=saved.get("cost"),
                    native_cost=saved.get("native_cost"),
                    raw_cost=saved.get("raw_cost"),
                    native_terminal=saved.get("native_terminal"),
                    absorbing_cost=saved.get("absorbing_cost"),
                    mae=saved.get("mae"),
                    roster_status="saved_episode",
                    source="replay_gate" if panel else "saved_evaluation",
                )
            )
    return rows


def _metrics(values: dict) -> list[dict]:
    rows = []
    for actor, groups in values.items():
        for dataset, metric in [
            ("own_corpus", groups.get("own", {})),
            *groups.get("historical", {}).items(),
        ]:
            if metric:
                rows.append(
                    dict(
                        actor=actor,
                        dataset=dataset,
                        rows=metric.get("rows"),
                        action_mse=metric.get("action_mse"),
                        weighted_action_mse=metric.get("weighted_action_mse"),
                        batch_size=metric.get("batch_size"),
                        source="saved_metric",
                    )
                )
    return rows


def _figures(out: Path, curves: list[dict], outcomes: list[dict], reader: _Reader) -> list[str]:
    summaries = [
        row
        for row in outcomes
        if row["row_kind"] == "summary"
        and row["panel"] in {"primary", "confirmation"}
        and row["roster_status"] == "fixed_roster_complete"
    ]
    if not curves and not summaries:
        return []
    reader.check()
    with tempfile.TemporaryDirectory(prefix="c1-report-mpl-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            from matplotlib import pyplot as plt

            artifacts = []
            if curves:
                fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
                for actor in sorted({row["actor"] for row in curves}):
                    data = [row for row in curves if row["actor"] == actor]
                    for axis, key in zip(
                        axes,
                        ("action_mse", "weighted_action_mse", "unclipped_gradient_norm"),
                        strict=True,
                    ):
                        present = [row for row in data if _finite(row.get(key))]
                        if present:
                            axis.plot(
                                [r.get("fit_update") or r["update"] for r in present],
                                [r[key] for r in present],
                                linewidth=0.7,
                                alpha=0.75,
                                label=actor,
                            )
                        axis.set_ylabel(key.replace("_", " "))
                        axis.grid(alpha=0.2)
                axes[0].legend(fontsize=7, ncol=3)
                axes[-1].set_xlabel("Recorded child fit update (all points; no smoothing)")
                fig.suptitle("C1 saved minibatch losses and gradient norm")
                fig.tight_layout()
                for extension in ("svg", "png"):
                    name = "learning_curves." + extension
                    buffer = io.BytesIO()
                    fig.savefig(
                        buffer,
                        format=extension,
                        dpi=160,
                        metadata={"Date": None} if extension == "svg" else {},
                    )
                    _save(out / name, buffer.getvalue())
                    artifacts.append(name)
                plt.close(fig)
            if summaries:
                labels = sorted({row["actor"] for row in summaries})
                fig, axes = plt.subplots(2, 2, figsize=(12, 7))
                for panel, marker in (("primary", "o"), ("confirmation", "x")):
                    data = {r["actor"]: r for r in summaries if r["panel"] == panel}
                    for axis, key, title in zip(
                        axes.flat,
                        ("walking_successes", "full_survivals", "mean_cost", "mean_steps"),
                        (
                            "Walking successes",
                            "Full survival",
                            "Mean complete cost",
                            "Mean executed controls",
                        ),
                        strict=True,
                    ):
                        present = [
                            (i, data[actor][key])
                            for i, actor in enumerate(labels)
                            if actor in data and _finite(data[actor].get(key))
                        ]
                        if present:
                            axis.plot(
                                [p[0] for p in present],
                                [p[1] for p in present],
                                marker,
                                label=panel,
                            )
                        axis.set_title(title)
                        axis.set_xticks(
                            range(len(labels)), labels, rotation=65, ha="right", fontsize=7
                        )
                        axis.grid(alpha=0.2)
                axes[0, 0].legend()
                fig.suptitle("Saved paired panels; retention without walking is not competence")
                fig.tight_layout()
                for extension in ("svg", "png"):
                    name = "standalone_outcomes." + extension
                    buffer = io.BytesIO()
                    fig.savefig(
                        buffer,
                        format=extension,
                        dpi=160,
                        metadata={"Date": None} if extension == "svg" else {},
                    )
                    _save(out / name, buffer.getvalue())
                    artifacts.append(name)
                plt.close(fig)
            return artifacts
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def render(run: Path, out: Path, audit: Path | None = None) -> dict[str, Any]:
    """Create a bounded saved-only report; missing measurements stay missing."""
    run, out = Path(run), Path(out)
    if out.exists():
        raise FileExistsError("C1 report output is create-only")
    root = run / "run" if (run / "run/config.json").is_file() else run
    reader = _Reader()
    documents = {}
    for relative in (
        "config.json",
        "manifest.json",
        "worker_result.json",
        "worker_exit.json",
        "input_manifest.json",
        "prerequisites.json",
        "prepared/preparation_receipt.json",
        "worker/replay_gate.json",
        "worker/contrast_gate.json",
        "worker/metrics.json",
        "worker/actor_registry.json",
        "worker/fit_records.json",
        "worker/outcomes.json",
    ):
        path = root / relative
        if path.is_file():
            documents[relative] = reader.load(path)
    result = documents.get("worker_result.json", {})
    config = documents.get("config.json", {})
    replay = documents.get("worker/replay_gate.json", {})
    saved_outcomes = documents.get("worker/outcomes.json", {})
    panels = result.get("student_outcomes", saved_outcomes.get("panels", {}))
    gates = result.get("gates", saved_outcomes.get("gates", {}))
    metrics = result.get("metrics", documents.get("worker/metrics.json", {}))
    curve_docs = [
        ("worker_result.json", result),
        ("worker/fit_records.json", documents.get("worker/fit_records.json", {})),
    ]
    for path in sorted((root / "worker/learning").glob("*/*.json")):
        curve_docs.append((str(path.relative_to(root)), reader.load(path)))
    curves = _curves(curve_docs)
    evaluations = []
    if (root / "worker/evaluations.jsonl").is_file():
        evaluations = reader.load(root / "worker/evaluations.jsonl", lines=True)
    outcomes = _outcomes(panels, evaluations, replay)
    metric_rows = _metrics(metrics)
    audit_value, audit_path = {}, None
    if audit is not None:
        audit_path = Path(audit)
        if audit_path.is_dir():
            candidates = [
                audit_path / name
                for name in (
                    "audit.json",
                    "receipt.json",
                    "result/audit.json",
                    "result/receipt.json",
                )
            ]
            audit_path = next((path for path in candidates if path.is_file()), candidates[0])
        audit_value = reader.load(audit_path)
    audit_binding = audit_value.get("worker_result_sha256")
    association = "worker_result_hash_not_declared"
    if audit_binding is not None:
        worker_binding = reader.inputs.get(str(root / "worker_result.json"), {})
        if audit_binding != worker_binding.get("sha256"):
            raise ValueError("C1 audit receipt binds a different worker result")
        association = "worker_result_sha256_verified"
    config_association = "config_hash_not_declared"
    if audit_value.get("config_sha256") is not None:
        if audit_value["config_sha256"] != reader.inputs.get(str(root / "config.json"), {}).get(
            "sha256"
        ):
            raise ValueError("C1 audit receipt binds a different configuration")
        config_association = "config_sha256_verified"
    status = result.get("status", "NO_WORKER_RESULT_INCOMPLETE")
    complete = status == "COMPLETED_FIXED_C1_ROSTER"
    degenerate = status == "STOPPED_DEGENERATE_PLANNED_CONTRAST"
    audited = (
        (
            audit_value.get("full_scientific_audit_passed") is True
            or audit_value.get("scientific_round_audit_complete") is True
        )
        and "c1" in str(audit_value.get("schema", "")).lower()
        and association == "worker_result_sha256_verified"
        and config_association == "config_sha256_verified"
    )
    if reader.partial_lines:
        complete = audited = False
    if degenerate:
        verdict = "The planned weighting contrast was degenerate. No child fitting or standalone learning comparison was measured; this is not a failed-learning result."
    elif not complete:
        verdict = "Execution is incomplete or stopped at a prerequisite/infrastructure gate. Available prefixes are retained; absent child or standalone measurements are missing, not zero or evidence of failed learning."
    elif not audited:
        verdict = "The fixed roster is saved, but a completed C1 audit is not bound here. Learning claims remain provisional."
    elif gates.get("confirmed"):
        verdict = "Saved audited gates confirm the declared weighting walking gain. This is finite learner evidence; no coupling loop or follow-up is established."
    elif gates.get("uniform_parent_confirmed"):
        verdict = "Uniform teaching produced the declared confirmed walking gain; cost-effect weighting did not confirm its additional gain. These are separate results."
    else:
        verdict = "The declared weighting walking gain was not confirmed. Reconstruction or survival gains remain separate partial outcomes; competence requires the frozen walking criteria."
    location = config.get("locations", {})
    external_run = (
        str(Path(location["remote_workspace"]) / "runs" / location["run_id"])
        if location.get("remote_workspace") and location.get("run_id")
        else str(root)
    )
    gaps = result.get("teacher_student_gap", saved_outcomes.get("source_context_gaps", []))
    contrast = result.get("contrast_gate", documents.get("worker/contrast_gate.json", {}))
    out.mkdir(parents=True, exist_ok=False)
    _csv(out / "curves.csv", ("actor", "stream", "arm", *CURVE_FIELDS, "failed", "sources"), curves)
    _csv(out / "outcomes.csv", OUTCOME_FIELDS, outcomes)
    _csv(
        out / "metrics.csv",
        ("actor", "dataset", "rows", "action_mse", "weighted_action_mse", "batch_size", "source"),
        metric_rows,
    )
    figures = _figures(out, curves, outcomes, reader)
    cost = dict(
        schema="walker-c1-saved-cost-report-v1",
        worker=result.get("costs", {}),
        worker_process=documents.get("worker_exit.json"),
        preparation=documents.get("prepared/preparation_receipt.json"),
        planned_limits=config.get("limits", {}),
        inherited_source_run=location.get("source_run"),
        inherited_work=result.get("inherited_work"),
        worker_process_lifetime_cpu=result.get("process_lifetime_cpu"),
        worker_process_lifetime_io=result.get("process_lifetime_io"),
        prerequisites=documents.get("prerequisites.json"),
        audit=audit_value or None,
        report_scope="Saved JSON, figures and scalar arithmetic only; zero neural, physics or optimizer work. Missing accounting is not zero.",
    )
    _save(out / "costs.json", (json.dumps(cost, indent=2, allow_nan=False) + "\n").encode())
    summaries = [row for row in outcomes if row["row_kind"] == "summary"]
    teachers = [row for row in outcomes if row["panel"] == "teacher_replay"]
    text = [
        "# C1.0: cost-effect-weighted fitting",
        f"Status: `{status}`. Completed C1 audit bound: **{audited}**.",
        verdict,
        "An incomplete final JSONL fragment is retained by hash; only complete saved lines are shown. The current evidence is incomplete."
        if reader.partial_lines
        else "",
        _table(
            [audit_value] if audit_value else [],
            (
                "passed",
                "full_scientific_audit_passed",
                "scientific_contrast_completed",
                "scientific_status",
            ),
        ),
        "## Teacher replay",
        _table(
            teachers,
            (
                "tag",
                "walking",
                "survived",
                "controls",
                "cost",
                "raw_cost",
                "native_terminal",
                "absorbing_cost",
            ),
        ),
        "## Standalone students",
        "Walking and full survival are separate. Individual competence requires at least 15/16 walking on both fresh panels; survival alone is retention. Source-context results use previously observed teacher resets; primary and confirmation use fresh paired resets. No returning to one reference path is required.",
        _table(
            summaries,
            (
                "panel",
                "actor",
                "episodes",
                "walking_successes",
                "full_survivals",
                "mean_cost",
                "controls",
                "roster_status",
            ),
        ),
        "## Fitting and held trajectories",
        "Ordinary action MSE is distinct from weighted training loss. Every saved JSON curve point is retained without smoothing in curves.csv; no missing point is fabricated. Raw-only curve prefixes require a separately audited scalar extract; this renderer never decodes their arrays.",
        _table(
            metric_rows,
            ("actor", "dataset", "rows", "action_mse", "weighted_action_mse", "batch_size"),
        ),
        f"Recorded learning points: **{len(curves)}**. Recorded standalone episode rows: **{len(evaluations)}**.",
        "## Gates and paired gaps",
        "Saved gates (not reselected by this renderer):",
        "```json",
        json.dumps(gates, indent=2, allow_nan=False),
        "```",
        "Contrast gate:",
        "```json",
        json.dumps(contrast, indent=2, allow_nan=False),
        "```",
        "Teacher–student gaps below are only from matched source starting contexts; they are not a fresh generalization panel.",
        _table(
            gaps,
            (
                "actor",
                "source_tag",
                "teacher_cost",
                "student_cost",
                "cost_gap",
                "teacher_walking",
                "student_walking",
                "teacher_survived",
                "student_survived",
            ),
        ),
        "## Accounting and evidence",
        "Complete saved requested/completed/unknown-partial work, worker process metadata, preparation and audit scope are preserved in costs.json. Budget ceilings are not charges for completed work.",
        f"Host: `{location.get('remote_host', 'not declared')}`. New raw arrays/index/ledger and child checkpoints remain under `{external_run}/worker/`; incoming parents and inputs under `{external_run}/prepared/`. Inherited source evidence: `{location.get('source_run', 'not bound in saved configuration')}`. This renderer never opens them.",
        "Fixed-tape effects are not feedback Q, recovery, intrinsic importance or a proof of teachability. This learner diagnostic does not establish MPPI–policy coupling or a CIO–GPS reproduction.",
    ]
    for figure in figures:
        if figure.endswith(".png"):
            text.extend([f"![Saved C1 evidence]({figure})"])
    _save(out / "report.md", ("\n\n".join(text) + "\n").encode())
    reader.check()
    outputs = {
        str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=_sha(p))
        for p in sorted(out.iterdir())
        if p.is_file()
    }
    if sum(v["bytes"] for v in outputs.values()) > MAX_OUTPUT - 65536:
        raise ValueError("C1 report output ceiling exhausted")
    receipt = dict(
        schema="walker-c1-saved-report-receipt-v1",
        saved_status=status,
        C1_audit_bound=audited,
        C1_audit_association=association,
        C1_config_association=config_association,
        inputs=reader.inputs,
        source_code=dict(
            file=str(Path(__file__)),
            bytes=Path(__file__).stat().st_size,
            sha256=_sha(Path(__file__)),
        ),
        outputs=outputs,
        table_rows=dict(curves=len(curves), outcomes=len(outcomes), metrics=len(metric_rows)),
        figures=figures,
        new_neural_rows=0,
        new_native_controls=0,
        new_optimizer_steps=0,
        saved_JSON_read_bytes=reader.read_bytes,
        incomplete_JSONL_fragments=reader.partial_lines,
        output_hash_read_bytes=sum(v["bytes"] for v in outputs.values()),
        wall_seconds=time.monotonic() - reader.started,
        peak_rss_bytes=_rss(),
        limits=dict(wall_seconds=1800, peak_rss_bytes=MAX_RSS, output_bytes=MAX_OUTPUT),
        receipt_excludes_itself=True,
    )
    payload = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    if len(payload.encode()) > 65536:
        raise ValueError("C1 report receipt exceeds its reserved closure space")
    (out / "receipt.json").write_text(payload)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    render(args.run, args.out, args.audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
