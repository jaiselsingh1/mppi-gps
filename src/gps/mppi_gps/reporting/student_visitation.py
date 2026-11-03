"""Render saved V-only outcomes and actual returned prefixes; never infer."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import resource
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

MAX_READ = 512 * 1024**2
MAX_OUTPUT = 256 * 1024**2
MAX_WALL = 1800
MAX_RSS = 2 * 1024**3
OUTCOME_FIELDS = (
    "kind",
    "tag",
    "actor",
    "stream",
    "panel",
    "source_tag",
    "anchor_ordinal",
    "control",
    "seed",
    "controls",
    "walking",
    "survived",
    "done",
    "mae",
    "raw_cost",
    "native_cost",
    "native_terminal",
    "absorbing_cost",
    "cost",
)
CURVE_FIELDS = (
    "tag",
    "kind",
    "control",
    "elapsed_seconds",
    "velocity",
    "height",
    "pitch",
    "native_cost",
    "terminal_cost",
    "done",
)


def _rss() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class SavedReader:
    def __init__(self):
        self.started = time.monotonic()
        self.read_bytes = 0
        self.inputs = {}

    def check(self) -> None:
        if time.monotonic() - self.started > MAX_WALL or _rss() > MAX_RSS:
            raise ValueError("V report wall/RSS limit exhausted")

    def load(self, path: Path, *, lines: bool = False) -> Any:
        self.check()
        if path.suffix not in {".json", ".jsonl"} or path.is_symlink():
            raise ValueError("V report accepts ordinary saved JSON only")
        before = path.stat()
        size = before.st_size
        if size > 256 * 1024**2 or self.read_bytes + size > MAX_READ:
            raise ValueError("V report read reservation exhausted")
        with path.open("rb") as stream:
            payload = stream.read(size)
        self.read_bytes += len(payload)
        if len(payload) != size or path.stat() != before:
            raise ValueError("V saved report input changed")
        self.inputs[str(path)] = dict(bytes=size, sha256=hashlib.sha256(payload).hexdigest())

        def reject(value):
            raise ValueError("Nonfinite V JSON: " + value)

        if lines:
            if payload and not payload.endswith(b"\n"):
                raise ValueError("V report refuses incomplete final JSONL rows")
            return [json.loads(row, parse_constant=reject) for row in payload.splitlines() if row]
        return json.loads(payload, parse_constant=reject)


def _save(out: Path, name: str, payload: bytes) -> None:
    used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if used + len(payload) > MAX_OUTPUT - 65536:
        raise ValueError("V report output reservation exhausted")
    with (out / name).open("xb") as stream:
        stream.write(payload)


def _csv(out: Path, name: str, fields: tuple[str, ...], rows: list[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _save(out, name, stream.getvalue().encode())


def _table(rows: list[dict], fields: tuple[str, ...]) -> str:
    def show(value):
        return "missing" if value is None else str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(fields) + " |",
            "| " + " | ".join("---" for _ in fields) + " |",
            *["| " + " | ".join(show(row.get(key)) for key in fields) + " |" for row in rows],
        ]
    )


def _summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        population = (
            "noninitial_primary" if (row.get("anchor_ordinal") or 0) > 0 else "initial_secondary"
        )
        if row["kind"] == "reference_reset":
            population = "own_reset_positive_control"
        groups[(row["kind"], row.get("panel"), row.get("stream"), population)].append(row)
    result = []
    for (kind, panel, stream, population), group in groups.items():
        result.append(
            dict(
                kind=kind,
                panel=panel,
                stream=stream,
                population=population,
                episodes=len(group),
                walking=sum(row["walking"] is True for row in group),
                survival=sum(row["survived"] is True for row in group),
                controls=sum(row["controls"] for row in group),
                native_cost=sum(row["native_cost"] for row in group),
                native_terminal=sum(row["native_terminal"] for row in group),
                absorbing_cost=sum(row["absorbing_cost"] for row in group),
                cost=sum(row["cost"] for row in group),
            )
        )
    return sorted(
        result,
        key=lambda row: (row["kind"], str(row["panel"]), str(row["stream"]), row["population"]),
    )


def _figures(out: Path, curves: list[dict], reader: SavedReader) -> list[str]:
    if not curves:
        return []
    reader.check()
    with tempfile.TemporaryDirectory(prefix="s12-v-report-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            from matplotlib import pyplot as plt

            by_tag = defaultdict(list)
            for point in curves:
                by_tag[point["tag"]].append(point)
            names = []
            colors = {
                "parent_replay": "#687b9a",
                "reference_reset": "#2d9c80",
                "reference_takeover": "#b05d35",
            }
            for window, suffix in ((None, "full"), (100, "first100")):
                fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
                shown = set()
                for points in by_tag.values():
                    selected = [p for p in points if window is None or p["control"] < window]
                    if not selected:
                        continue
                    kind = selected[0]["kind"]
                    label = kind if kind not in shown else None
                    shown.add(kind)
                    for axis, key in zip(
                        axes, ("velocity", "height", "pitch", "native_cost"), strict=True
                    ):
                        axis.plot(
                            [p["control"] for p in selected],
                            [p[key] for p in selected],
                            color=colors[kind],
                            alpha=0.18,
                            linewidth=0.6,
                            label=label,
                        )
                        axis.set_ylabel(key.replace("_", " "))
                axes[0].axhline(1.5, color="black", linestyle="--", linewidth=0.7)
                axes[0].legend()
                axes[-1].set_xlabel("Actual control index within continued branch")
                fig.suptitle("All available returned prefixes; each trace stops at its actual fall")
                fig.tight_layout()
                name = "prefixes_" + suffix + ".png"
                fig.savefig(out / name, dpi=130)
                plt.close(fig)
                reader.check()
                if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > MAX_OUTPUT - 65536:
                    raise ValueError("V report figure output limit exhausted")
                names.append(name)
            return names
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def render(run: Path, out: Path, audit: Path | None = None) -> dict[str, Any]:
    """Create a saved-only report; missing and unattempted stages remain missing."""
    run, out = Path(run).resolve(), Path(out).resolve()
    root = run / "run" if (run / "run/config.json").is_file() else run
    if out.exists() or any(
        out.is_relative_to(root / name) for name in ("worker", "prepared", "source")
    ):
        raise ValueError("V report output must be fresh and outside preserved evidence")
    reader = SavedReader()
    documents = {}
    for name in ("config.json", "worker_result.json", "worker_exit.json", "coordinator_exit.json"):
        if (root / name).is_file():
            documents[name] = reader.load(root / name)
    for name in (
        "static_gate.json",
        "parent_replay_gate.json",
        "reference_positive_control_gate.json",
        "reference_usefulness_screen.json",
        "accounting.json",
    ):
        if (root / "worker" / name).is_file():
            documents["worker/" + name] = reader.load(root / "worker" / name)
    audit_path = Path(audit) if audit is not None else root / "audit/audit.json"
    if audit_path.is_dir():
        audit_path = audit_path / "audit.json"
    audit_value = reader.load(audit_path) if audit_path.is_file() else None
    outcome_path = root / "worker/outcomes.jsonl"
    outcomes = reader.load(outcome_path, lines=True) if outcome_path.is_file() else []
    if audit_value is not None and "outcomes_sha256" in audit_value and outcome_path.is_file():
        if reader.inputs[str(outcome_path)]["sha256"] != audit_value["outcomes_sha256"]:
            raise ValueError("V saved outcomes differ from their independent audit binding")
    for row in outcomes:
        row["kind"] = row.get("stage", row.get("kind"))
        row["control"] = row.get("t", row.get("control", 0))
    curves_path = audit_path.parent / "prefix_curves.jsonl"
    curves = reader.load(curves_path, lines=True) if curves_path.is_file() else []
    if audit_value is not None and "prefix_curve_file" in audit_value and curves_path.is_file():
        binding = audit_value["prefix_curve_file"]
        if reader.inputs[str(curves_path)] != {key: binding[key] for key in ("bytes", "sha256")}:
            raise ValueError("V saved prefix curves differ from their independent audit binding")
    if len(outcomes) > 224 or len(curves) > 224000:
        raise ValueError("V saved roster exceeds frozen census")
    for point in curves:
        if any(
            type(point[key]) not in (int, float) or not math.isfinite(point[key])
            for key in ("velocity", "height", "pitch", "native_cost")
        ):
            raise ValueError("Nonfinite V prefix scalar")
    out.mkdir(parents=True, exist_ok=False)
    _csv(out, "outcomes.csv", OUTCOME_FIELDS, outcomes)
    _csv(out, "actual_prefixes.csv", CURVE_FIELDS, curves)
    summaries = _summaries(outcomes)
    summary_fields = (
        "kind",
        "panel",
        "stream",
        "population",
        "episodes",
        "walking",
        "survival",
        "controls",
        "native_cost",
        "native_terminal",
        "absorbing_cost",
        "cost",
    )
    _csv(out, "outcome_summary.csv", summary_fields, summaries)
    figures = _figures(out, curves, reader)
    receipt = dict(
        schema="walker-student-visitation-s1.2-v-saved-report-v1",
        saved_only=True,
        audit_passed=None if audit_value is None else audit_value.get("passed"),
        no_automatic_learning=True,
        outcomes=len(outcomes),
        prefix_points=len(curves),
        read_bytes=reader.read_bytes,
        input_files=reader.inputs,
        new_scientific_work=dict(
            neural_rows=0,
            prediction_calls=0,
            native_controls=0,
            physics_substeps=0,
            backward_rows=0,
            optimizer_steps=0,
            checkpoint_tensor_loads=0,
            resets=0,
            restores=0,
            mppi_queries=0,
        ),
        available_documents=sorted(documents),
        missing_evidence=[
            name for name in ("worker_result.json", "worker_exit.json") if name not in documents
        ],
        accounting=dict(
            worker=documents.get(
                "worker/accounting.json", documents.get("worker_result.json", {}).get("costs")
            ),
            audit=None if audit_value is None else audit_value.get("work"),
            worker_process_cpu=documents.get("worker_result.json", {}).get("process_lifetime_cpu"),
        ),
        closure_reason=None if audit_value is None else audit_value.get("closure_reason"),
    )
    text = [
        "# S1.2 V: reference behavior at preserved student-visited states",
        "Saved evidence only. This block tests reference behavior and the exact replay/control contract; it performs no learning and makes no teachability or MPPI-coupling claim.",
        "Independent audit: "
        + ("missing" if audit_value is None else str(audit_value.get("passed")))
        + ". Closure: "
        + str(receipt["closure_reason"])
        + ".",
        "Walking requires 1,000 actual controls, no fall and full-window velocity MAE at most 0.3. Longer survival without walking is retention only. Native terminal cost is charged once; each missing control after an actual fall contributes five to the absorbing tail. Unattempted stages and missing post-fall states are never filled with zeros.",
        _table(summaries, summary_fields),
        "Only the 72 noninitial anchors enter gain/nonregression admission. The 24 initial anchors are secondary. The same two or more streams must gain at least two walking outcomes in both panels, with no walking or full-survival regression in any stream. Own-reset positive controls require at least 15/16 walking on each panel.",
        "The 96 anchors are nested within 24 preserved parent episodes, with four anchors per episode. They are paired continuations, not 96 independent task replications. The 32 own-reset controls use the declared matched canonical reset contexts. Successful reference continuations would establish behavior under this assay, not useful labels or successful learning.",
        "The all-prefix figures show every available executed trace and stop each trace at its actual fall. CSV files preserve all available outcomes and returned controls.",
        *[f"![{name}]({name})" for name in figures],
        "Complete saved accounting is in report_receipt.json; a missing cost record remains missing. Conditional learning requires separate review and authorization.",
    ]
    _save(out, "report.md", ("\n\n".join(text) + "\n").encode())
    receipt.update(wall_seconds=time.monotonic() - reader.started, peak_rss_bytes=_rss())
    _save(
        out,
        "report_receipt.json",
        (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )
    return receipt


report_run = render
