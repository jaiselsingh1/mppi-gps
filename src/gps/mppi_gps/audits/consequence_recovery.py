"""Close C1's preserved audit evidence without repeating scientific execution.

The original numerical certification is retained, not reproduced. This module
uses only the standard library: checkpoint files and array archives are hashed
as bytes, never decoded, and no actor or simulator is constructed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import resource
import stat
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


class RecoveryError(ValueError):
    """A failed recovery check stops the new attempt and retains its census."""


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    _need(type(value) is int and value >= minimum, "Invalid integer: " + label)
    return value


def _finite(value: Any, label: str) -> float:
    _need(type(value) in (int, float) and math.isfinite(value), "Nonfinite " + label)
    return float(value)


def _json(payload: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(
        payload,
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            RecoveryError("Nonfinite JSON constant: " + value)
        ),
    )


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _rss() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _identity(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    _need(stat.S_ISREG(value.st_mode), "Only ordinary input files are eligible")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _binding(value: Any) -> dict[str, Any]:
    _need(isinstance(value, dict), "File binding must be an object")
    size = _integer(value.get("bytes"), "binding bytes")
    digest = value.get("sha256")
    _need(
        isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
        "Invalid whole-file SHA256 binding",
    )
    return dict(bytes=size, sha256=digest)


class _Reader:
    def __init__(self, protocol: dict[str, Any]):
        self.started = time.monotonic()
        self.limits = protocol["limits"]
        self.read_bytes = 0
        self.hashed_files = self.journal_lines = self.json_documents = 0
        self.roster: dict[tuple[str, str], dict[str, Any]] = {}
        self.identities: dict[Path, tuple[int, ...]] = {}
        self.roots = {}
        for key, location in (
            ("run", protocol["locations"]["original_run"]),
            ("artifacts", protocol["locations"]["original_artifacts"]),
        ):
            root = Path(location)
            _need(root.is_absolute() and root.is_dir() and not root.is_symlink(), "Invalid root")
            self.roots[key] = root.resolve(strict=True)
        _need(self.roots["run"] != self.roots["artifacts"], "Input roots must be distinct")
        self.check()

    def check(self) -> None:
        _need(
            time.monotonic() - self.started < self.limits["wall_seconds"],
            "Recovery wall budget exhausted",
        )
        _need(_rss() <= self.limits["rss_bytes"], "Recovery RSS budget exhausted")

    def reserve(self, amount: int) -> None:
        self.check()
        _need(
            self.read_bytes + amount <= self.limits["read_bytes"], "Recovery read budget exhausted"
        )

    def path(self, root: str, relative: str) -> Path:
        _need(root in self.roots and isinstance(relative, str), "Unknown input root/path")
        parts = PurePosixPath(relative)
        _need(
            bool(parts.parts) and not parts.is_absolute() and ".." not in parts.parts,
            "Input path escapes its root",
        )
        path = self.roots[root].joinpath(*parts.parts)
        current = path
        while current != self.roots[root]:
            _need(not current.is_symlink(), "Symlink input is ineligible")
            current = current.parent
        _need(path.resolve(strict=True).is_relative_to(self.roots[root]), "Input path escapes root")
        _identity(path)
        return path

    def add(self, root: str, relative: str, value: dict, origin: str) -> None:
        binding = _binding(value)
        path = self.path(root, relative)
        _need(path.stat().st_size == binding["bytes"], "Bound file size differs: " + relative)
        key = root, relative
        if key in self.roster:
            _need(
                all(self.roster[key][field] == binding[field] for field in ("bytes", "sha256")),
                "Conflicting file bindings: " + relative,
            )
            self.roster[key]["origins"].append(origin)
        else:
            _need(len(self.roster) < self.limits["max_files"], "Recovery file census exhausted")
            self.roster[key] = dict(binding, origins=[origin])

    def _open(self, path: Path):
        self.check()
        identity = _identity(path)
        old = self.identities.setdefault(path, identity)
        _need(identity == old, "Input changed before a repeated read")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != identity:
            os.close(descriptor)
            raise RecoveryError("Input changed while opening")
        return os.fdopen(descriptor, "rb"), identity

    def read(self, root: str, relative: str, *, maximum: int | None = None) -> bytes:
        path = self.path(root, relative)
        size = path.stat().st_size
        _need(size <= (maximum or self.limits["max_json_bytes"]), "Metadata file exceeds bound")
        self.reserve(size)
        stream, identity = self._open(path)
        with stream:
            _need(identity[2] == size, "Metadata size changed before opening")
            payload = stream.read(size)
        self.read_bytes += len(payload)
        _need(self.read_bytes <= self.limits["read_bytes"], "Read budget exhausted")
        _need(len(payload) == size and _identity(path) == identity, "Input changed during read")
        self.check()
        return payload

    def json(self, root: str, relative: str) -> Any:
        value = _json(self.read(root, relative))
        self.json_documents += 1
        return value

    def anchored(self, value: dict, origin: str) -> Any:
        self.add(value["root"], value["file"], value, origin)
        payload = self.read(value["root"], value["file"])
        _need(hashlib.sha256(payload).hexdigest() == value["sha256"], "Anchor SHA256 differs")
        value = _json(payload)
        self.json_documents += 1
        return value

    def hash_roster(self) -> dict[str, Any]:
        self.reserve(sum(value["bytes"] for value in self.roster.values()))
        verified = {}
        for (root, relative), binding in sorted(self.roster.items()):
            path = self.path(root, relative)
            stream, identity = self._open(path)
            digest = hashlib.sha256()
            count = 0
            with stream:
                _need(identity[2] == binding["bytes"], "Bound size changed before hashing")
                remaining = binding["bytes"]
                while remaining:
                    request = min(1024**2, remaining)
                    self.reserve(request)
                    payload = stream.read(request)
                    count += len(payload)
                    self.read_bytes += len(payload)
                    _need(bool(payload), "Input ended during hashing")
                    remaining -= len(payload)
                    digest.update(payload)
            _need(
                _identity(path) == identity and count == binding["bytes"], "Input changed in hash"
            )
            _need(digest.hexdigest() == binding["sha256"], "Input SHA256 differs: " + relative)
            self.hashed_files += 1
            verified[root + "/" + relative] = dict(binding)
        return verified

    def lines(self, root: str, relative: str, *, lifecycle: bool = False) -> Iterator[dict]:
        path = self.path(root, relative)
        self.reserve(path.stat().st_size)
        stream, identity = self._open(path)
        maximum = self.limits["max_journal_line_bytes"]
        lifecycle_maximum = self.limits.get("max_lifecycle_line_bytes", maximum)
        with stream:
            self.reserve(identity[2])
            remaining = identity[2]
            while remaining:
                request = min(remaining, (lifecycle_maximum if lifecycle else maximum) + 1)
                self.reserve(request)
                payload = stream.readline(request)
                self.read_bytes += len(payload)
                remaining -= len(payload)
                _need(bool(payload), "Journal ended during parsing")
                _need(payload.endswith(b"\n"), "Partial or excessive journal line")
                row = _json(payload)
                self.journal_lines += 1
                allowed = (
                    lifecycle_maximum
                    if lifecycle and row.get("event") in {"runtime_open", "runtime_close"}
                    else maximum
                )
                _need(len(payload) <= allowed, "Journal line exceeds frozen bound")
                yield row
        _need(_identity(path) == identity, "Journal changed during parsing")

    def unchanged(self) -> None:
        for path, identity in self.identities.items():
            _need(_identity(path) == identity, "Input changed before recovery closure")


def _zero_counts(value: dict) -> bool:
    return isinstance(value, dict) and all(type(v) is int and v == 0 for v in value.values())


def _predictions(reader: _Reader, relative: str, expected: dict, actors: set[str]) -> dict:
    pending = None
    calls = rows = events = 0
    for row in reader.lines("artifacts", relative):
        _need(
            type(row.get("id")) is int and row["id"] == events,
            "Prediction journal event order differs",
        )
        events += 1
        if row.get("event") == "prediction_requested":
            amount = _integer(row.get("rows"), "prediction rows", 1)
            _need(
                pending is None and row.get("actor_hash") in actors, "Prediction ownership differs"
            )
            _need(
                type(row.get("call")) is int and row["call"] == calls,
                "Prediction call order differs",
            )
            _need(row.get("requested_rows_after") == rows + amount, "Prediction row census differs")
            pending = row["actor_hash"], row.get("scope"), amount
        elif row.get("event") == "prediction_complete":
            _need(
                pending == (row.get("actor_hash"), row.get("scope"), row.get("rows")),
                "Prediction completion lacks its matching request",
            )
            rows += pending[2]
            calls += 1
            _need(row.get("completed_rows_after") == rows, "Completed prediction census differs")
            pending = None
        else:
            raise RecoveryError("Unexpected/error prediction journal event")
    _need(pending is None, "Unclosed prediction request")
    _need(
        (calls, rows, events)
        == (
            expected["prediction_calls"],
            expected["prediction_rows"],
            expected["prediction_events"],
        ),
        "Prediction journal totals differ",
    )
    return dict(calls=calls, rows=rows, events=events, unknown_rows=0)


def _worker_journal(reader: _Reader, receipt: dict, frame_skip: int) -> dict:
    requested: Counter[str] = Counter()
    completed: Counter[str] = Counter()
    inventory: Counter[str] = Counter()
    pending: dict[int, tuple[str, int]] = {}
    native = None
    environment = False
    operations = events = 0
    for row in reader.lines("run", "worker/evidence/ledger.jsonl", lifecycle=True):
        _need(row.get("id") == f"event-{events:09d}", "Worker journal event order differs")
        events += 1
        event = row["event"]
        inventory[event] += 1
        if event == "work_requested":
            operation = _integer(row.get("operation"), "work operation", 1)
            _need(operation == operations + 1 and operation not in pending, "Work order differs")
            operations = operation
            category, amount = row["category"], _integer(row.get("amount"), "work amount")
            pending[operation] = category, amount
            requested[category] += amount
            if category == "neural_rows":
                requested["prediction_calls"] += 1
        elif event == "work_complete":
            pair = pending.pop(row.get("operation"), None)
            _need(pair == (row.get("category"), row.get("amount")), "Work completion differs")
            completed[pair[0]] += pair[1]
            if pair[0] == "neural_rows":
                completed["prediction_calls"] += 1
        elif event == "trace_intent":
            _need(native is None, "Overlapping native intent")
            native = row["alias"], row["ordinal"]
            requested["native_controls"] += 1
            requested["physics_substeps"] += frame_skip
        elif event == "trace_returned":
            _need(native == (row["alias"], row["ordinal"]), "Native return differs")
            native = None
            completed["native_controls"] += 1
            completed["physics_substeps"] += frame_skip
        elif event in {
            "index_before",
            "index_after",
            "expected_index_before",
            "expected_index_after",
        }:
            category = (
                "expected_index_draws" if event.startswith("expected") else "actual_index_draws"
            )
            (requested if event.endswith("before") else completed)[category] += 256
        elif event in {"shuffle_requested", "shuffle_complete"}:
            counts = requested if event.endswith("requested") else completed
            counts["shuffle_calls"] += 1
            counts["shuffle_entries"] += _integer(row["rows"], "shuffle entries")
        elif event == "environment_requested":
            _need(not environment, "Duplicate environment request")
            environment = True
            requested["environment_constructions"] += 1
        elif event == "environment_complete":
            _need(environment, "Environment completion lacks request")
            environment = False
            completed["environment_constructions"] += 1
        elif event in {
            "fit_construction_requested",
            "fit_construction_complete",
            "endpoint_save_requested",
            "endpoint_save_complete",
        }:
            category = (
                "policy_constructions" if event.startswith("fit_construction") else "endpoint_saves"
            )
            (requested if event.endswith("requested") else completed)[category] += 1
        elif event.endswith("_error"):
            raise RecoveryError("Worker journal contains an error event")
    _need(not pending and native is None and not environment, "Unclosed worker work intent")
    _need(dict(inventory) == receipt["stream"]["event_inventory"], "Event inventory differs")
    # The frozen native boundary charges these helpers without adding duplicate
    # per-control events. Reconstruct their existing fixed geometry, not calls.
    for counts in (requested, completed):
        counts["external_mj_forward_calls"] = counts["restores"] + 2 * counts["resets"]
        counts["snapshot_reads"] = (
            2 * counts["native_controls"] + counts["restores"] + counts["resets"]
        )
    expected = receipt["accounting"]
    _need(dict(requested) == expected["independent_requested"], "Requested worker census differs")
    _need(dict(completed) == expected["independent_completed"], "Completed worker census differs")
    _need(requested == completed, "Worker has unmatched paid work")
    return dict(events=events, requested=dict(requested), completed=dict(completed))


def _close(actual: float, expected: float, label: str) -> None:
    _need(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9), label + " differs")


def _scalar_outcomes(
    reader: _Reader, result: dict, receipt: dict, registry: dict, expected: dict, config: dict
) -> dict:
    saved = reader.json("run", "worker/outcomes.json")
    replay = reader.json("run", "worker/replay_gate.json")
    _need(replay.get("passed") is True, "Original replay gate did not pass")
    teachers = {row["source_tag"]: row for row in replay["episodes"]}
    _need(len(teachers) == expected["teacher_replays"], "Teacher roster differs")
    _need(
        all(r["walking"] is True and r["survived"] is True for r in teachers.values()),
        "Teacher failed",
    )
    payload = reader.read("run", "worker/evaluations.jsonl")
    _need(payload.endswith(b"\n"), "Incomplete evaluation metadata")
    evaluations = [_json(line) for line in payload.splitlines()]
    _need(len(evaluations) == expected["standalone"], "Standalone episode roster differs")
    groups: dict[tuple[str, int, str], list[dict]] = {}
    identities = set()
    gaps = []
    required = {
        (alias, panel, case)
        for alias in registry
        for panel in ("primary", "confirmation", "source")
        for case in range(
            config["source_episodes_per_stream"]
            if panel == "source"
            else len(config["seeds"][panel])
        )
    }
    _need(len(required) == expected["standalone"], "Frozen episode census differs")
    for row in evaluations:
        key = row["actor"], row["panel"], row["case"]
        _need(key not in identities and row["actor"] in registry, "Duplicate/unknown episode")
        identities.add(key)
        actor = registry[row["actor"]]
        _need((row["stream"], row["arm"]) == (actor["stream"], actor["arm"]), "Actor join differs")
        _need(row["panel"] in {"primary", "confirmation", "source"}, "Unknown evaluation panel")
        _need(key in required, "Episode is outside the fixed roster")
        _need(
            row["tag"] == f"eval_{row['actor']}_{row['panel']}_{row['case']:02d}",
            "Episode tag join differs",
        )
        if row["panel"] != "source":
            _need(row["seed"] == config["seeds"][row["panel"]][row["case"]], "Reset seed differs")
        else:
            _need(
                row["source_tag"] == f"teacher_s{row['stream']}_independent_{row['case']:02d}",
                "Source starting-context join differs",
            )
        controls = _integer(row["controls"], "episode controls")
        horizon = _integer(row["h"], "episode horizon", 1)
        _need(horizon == config["source_controls_per_episode"], "Evaluation horizon differs")
        _need(controls <= horizon, "Episode controls exceed horizon")
        _need(type(row["done"]) is bool and type(row["survived"]) is bool, "Invalid outcome flags")
        _need(row["survived"] == (not row["done"] and controls == horizon), "Survival differs")
        _need(
            type(row["walking"]) is bool
            and row["walking"]
            == (row["survived"] and _finite(row["mae"], "velocity error") <= 0.3),
            "Walking outcome differs",
        )
        _close(row["native_cost"], row["raw_cost"] + row["native_terminal"], "Native cost")
        _close(row["cost"], row["native_cost"] + row["absorbing_cost"], "Complete cost")
        _close(
            row["native_terminal"],
            config["task"]["terminal_unhealthy_weight"] if row["done"] else 0,
            "Native terminal charge",
        )
        _close(
            row["absorbing_cost"], 5 * (horizon - controls) if row["done"] else 0, "Absorbing tail"
        )
        groups.setdefault((row["panel"], row["stream"], row["arm"]), []).append(row)
        if row["panel"] == "source":
            teacher = teachers[row["source_tag"]]
            gaps.append(
                dict(
                    actor=row["actor"],
                    source_tag=row["source_tag"],
                    cost_gap=row["cost"] - teacher["cost"],
                )
            )
    _need(identities == required, "Fixed standalone roster is incomplete")
    panels: dict[str, dict] = {}
    for (panel, stream, arm), rows in groups.items():
        rows.sort(key=lambda row: row["case"])
        _need([r["case"] for r in rows] == list(range(len(rows))), "Panel cases are incomplete")
        summary = dict(
            episodes=len(rows),
            successes=sum(r["walking"] for r in rows),
            survived=sum(r["survived"] for r in rows),
            controls=sum(r["controls"] for r in rows),
            mean_cost=sum(r["cost"] for r in rows) / len(rows),
            mean_steps=sum(r["controls"] for r in rows) / len(rows),
            mean_velocity_mae=sum(r["mae"] for r in rows) / len(rows),
        )
        panels.setdefault(panel, {}).setdefault(str(stream), {})[arm] = summary
        actual = saved["panels"][panel][str(stream)][arm]
        _need(set(actual) == set(summary), "Outcome summary fields differ")
        for field, value in summary.items():
            _close(value, actual[field], "Outcome summary " + field)
    _need(
        panels == result["student_outcomes"] or saved["panels"] == result["student_outcomes"],
        "Result outcomes differ",
    )
    _need(saved["panels"] == receipt["outcomes"]["panels"], "Prior audit outcomes differ")
    _need(saved["gates"] == result["gates"] == receipt["outcomes"]["gates"], "Saved gates differ")
    _need(len(gaps) == len(saved["source_context_gaps"]), "Source gap count differs")
    saved_gaps = {(r["actor"], r["source_tag"]): r for r in saved["source_context_gaps"]}
    for row in gaps:
        _close(
            row["cost_gap"],
            saved_gaps[(row["actor"], row["source_tag"])]["cost_gap"],
            "Paired source gap",
        )
    # Fresh walking criteria are integer comparisons, independent of cost/MSE.
    streams = sorted({r["stream"] for r in registry.values()})
    qualifiers, uniform_qualifiers = {}, {}
    for panel in ("primary", "confirmation"):
        qualifiers[panel], uniform_qualifiers[panel] = [], []
        regressions = uniform_regressions = False
        for stream in streams:
            actors = panels[panel][str(stream)]
            weighted = actors["cost_effect"]
            controls = [actors[arm] for arm in ("parent", "uniform", "shuffled")]
            if all(weighted["successes"] >= other["successes"] + 2 for other in controls):
                qualifiers[panel].append(stream)
            regressions |= any(
                weighted[k] < other[k] for other in controls for k in ("successes", "survived")
            )
            if actors["uniform"]["successes"] >= actors["parent"]["successes"] + 2:
                uniform_qualifiers[panel].append(stream)
            uniform_regressions |= any(
                actors["uniform"][k] < actors["parent"][k] for k in ("successes", "survived")
            )
        if regressions:
            qualifiers[panel] = []
        if uniform_regressions:
            uniform_qualifiers[panel] = []
    confirmed = len(set(qualifiers["primary"]) & set(qualifiers["confirmation"])) >= 2
    uniform_confirmed = (
        len(set(uniform_qualifiers["primary"]) & set(uniform_qualifiers["confirmation"])) >= 2
    )
    _need(
        confirmed == result["gates"]["confirmed"] == receipt["weighting_confirmed"],
        "Walking gate differs",
    )
    _need(
        uniform_confirmed == result["gates"]["uniform_parent_confirmed"],
        "Uniform walking gate differs",
    )
    return dict(
        episodes=len(evaluations),
        walking=sum(r["walking"] for r in evaluations),
        full_survival=sum(r["survived"] for r in evaluations),
        controls=sum(r["controls"] for r in evaluations),
        paired_gaps=len(gaps),
        weighting_confirmed=confirmed,
        uniform_parent_confirmed=uniform_confirmed,
        teacher_controls=sum(row["controls"] for row in teachers.values()),
    )


def _curves(reader: _Reader, result: dict, expected: dict, config: dict) -> dict:
    records = reader.json("run", "worker/fit_records.json")
    _need(
        len(records) == expected["fit_blocks"] and records == result["fit_records"],
        "Fit records differ",
    )
    count = 0
    for alias, record in records.items():
        evidence = record["fitting_evidence"]
        curve = evidence["learning_curve"]
        _need(len(curve) == config["learning"]["updates"], "Per-child update budget differs")
        _need(
            record["tag"] == alias and record["completed_updates"] == len(curve),
            "Fit updates differ",
        )
        _need(
            record["sampled_updates"] == len(curve) and evidence.get("failed") is False,
            "Incomplete fit",
        )
        for index, point in enumerate(curve, 1):
            _need(point["fit_update"] == index, "Learning curve update order differs")
            for key, value in point.items():
                _finite(value, "learning point " + key)
        count += len(curve)
    _need(count == expected["curve_rows"], "Learning point count differs")
    return dict(fit_blocks=len(records), complete_curve_rows=count)


def _assays(reader: _Reader, relative: str, expected: dict, config: dict, registry: dict) -> dict:
    payload = reader.read("artifacts", relative)
    rows = csv.DictReader(io.StringIO(payload.decode()))
    seen = set()
    controls = requested = falls = 0
    terminal = absorbing = 0.0
    sources = {
        f"teacher_s{row['stream']}_independent_{case:02d}"
        for row in registry.values()
        for case in range(config["source_episodes_per_stream"])
    }
    source_controls = config["source_controls_per_episode"]
    _need(expected["assays"] == len(sources) * source_controls, "Frozen assay census differs")
    for row in rows:
        key = row["source_tag"], int(row["t"])
        _need(
            key not in seen and key[0] in sources and 0 <= key[1] < source_controls,
            "Repeated/unknown assay source/control",
        )
        seen.add(key)
        horizon, actual = int(row["h"]), int(row["controls"])
        _need(
            1 <= actual <= horizon
            and horizon == min(100, source_controls - key[1])
            and row["done"] in {"True", "False"},
            "Invalid assay geometry",
        )
        controls += actual
        requested += horizon
        falls += row["done"] == "True"
        terminal += float(row["native_terminal"])
        absorbing += float(row["absorbing_cost"])
        done = row["done"] == "True"
        _close(
            float(row["native_terminal"]),
            config["task"]["terminal_unhealthy_weight"] if done else 0,
            "Assay native terminal",
        )
        _close(
            float(row["absorbing_cost"]),
            5 * (horizon - actual) if done else 0,
            "Assay absorbing tail",
        )
        _need(done or actual == horizon, "Unclosed successful assay window")
        _close(
            float(row["cost"]),
            float(row["native_cost"]) + float(row["absorbing_cost"]),
            "Assay complete cost",
        )
        _close(float(row["delta"]), float(row["cost"]) - float(row["baseline_cost"]), "Assay delta")
    _need(len(seen) == expected["assays"], "Assay scalar row census differs")
    return dict(
        rows=len(seen),
        requested_controls=requested,
        controls=controls,
        falls=falls,
        native_terminal_cost=terminal,
        absorbing_tail_cost=absorbing,
    )


def _validate_protocol(protocol: dict) -> None:
    _need(protocol.get("schema") == "walker-c1-audit-recovery-protocol-v1", "Wrong recovery schema")
    _need(
        protocol.get("study") == "C1.0-A1" and protocol.get("user_authorized") is True,
        "Recovery is not authorized",
    )
    _need(protocol.get("automatic_followup") is False, "Automatic follow-up is forbidden")
    _need(
        bool(protocol.get("new_scientific_work")) and _zero_counts(protocol["new_scientific_work"]),
        "Recovery must declare zero new scientific work",
    )
    limits = protocol["limits"]
    caps = dict(
        wall_seconds=900,
        rss_bytes=1024**3,
        read_bytes=8 * 1024**3,
        output_bytes=64 * 1024**2,
        max_files=2000,
        max_json_bytes=16 * 1024**2,
        max_journal_line_bytes=512,
        max_lifecycle_line_bytes=2048,
    )
    for key, maximum in caps.items():
        value = limits.get(key, maximum)
        _need(
            type(value) in (int, float) and math.isfinite(value) and 0 < value <= maximum,
            "Invalid/increased recovery limit: " + key,
        )
    for key in (
        "max_files",
        "max_json_bytes",
        "max_journal_line_bytes",
        "output_bytes",
        "read_bytes",
    ):
        _integer(limits[key], key, 1)
    expected = protocol["expected_counts"]
    for key in (
        "input_files",
        "actors",
        "teacher_replays",
        "assays",
        "fit_blocks",
        "curve_rows",
        "standalone",
        "prediction_calls",
        "prediction_rows",
        "prediction_events",
    ):
        _integer(expected[key], key, 1)
    cumulative = protocol["cumulative"]
    for key in ("prior_wall_upper_seconds", "audit_wall_limit_seconds"):
        _need(_finite(cumulative[key], key) >= 0, "Negative cumulative wall limit")
    for key in ("prior_read_upper_bytes", "audit_read_limit_bytes"):
        _integer(cumulative[key], key)
    _need(
        cumulative["audit_wall_limit_seconds"] <= 7200
        and cumulative["audit_read_limit_bytes"] <= 64 * 1024**3,
        "Original audit limits increased",
    )
    _need(
        cumulative["prior_wall_upper_seconds"] + limits["wall_seconds"]
        <= cumulative["audit_wall_limit_seconds"],
        "Cumulative audit time reservation exceeded",
    )
    _need(
        cumulative["prior_read_upper_bytes"] + limits["read_bytes"]
        <= cumulative["audit_read_limit_bytes"],
        "Cumulative audit read reservation exceeded",
    )


def recover(protocol: dict[str, Any], out: Path) -> dict[str, Any]:
    """Verify a new recovery attempt; only its coordinator can admit closure."""
    out = Path(out)
    # This guard precedes every write, including a failure receipt. Recovery
    # cannot use either original evidence tree as an output or error directory.
    resolved_out = out.resolve(strict=False)
    for location in (
        protocol["locations"]["original_run"],
        protocol["locations"]["original_artifacts"],
    ):
        root = Path(location).resolve(strict=False)
        _need(not resolved_out.is_relative_to(root), "Recovery output is inside original evidence")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    reader = None
    result: dict[str, Any] = dict(
        schema="walker-c1-audit-recovery-result-v1",
        recovery_checks_passed=False,
        admitted_original_scientific_checks=False,
        retained_numerical_certification=False,
        original_overall_audit_closure_passed=False,
        original_child_exit_code=None,
        original_audit_coordinator_failed=True,
        new_scientific_work=dict(protocol.get("new_scientific_work", {})),
        scope="Fresh saved-output integrity, scalar/accounting checks and recovery worker completion; retained original numerical certification. No new numerical replay, actor evaluation or retrospective original process exit.",
    )
    try:
        result["protocol_sha256"] = hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        _validate_protocol(protocol)
        reader = _Reader(protocol)
        expected = protocol["expected_counts"]
        anchors = {
            name: reader.anchored(value, "frozen anchor " + name)
            for name, value in protocol["anchors"].items()
        }
        receipt, failure, catalog = (
            anchors[k] for k in ("scientific_receipt", "failure", "audit_output_catalog")
        )
        _need(
            receipt.get("schema") == "walker-student-consequence-c1.0-audit-v1",
            "Wrong original audit",
        )
        _need(
            all(
                receipt.get(k) is True
                for k in (
                    "passed",
                    "full_scientific_audit_passed",
                    "scientific_contrast_completed",
                    "evidence_integrity_passed",
                )
            ),
            "Original numerical certification did not pass",
        )
        _need(
            all(
                receipt.get(k) == 0
                for k in (
                    "physics_controls",
                    "optimizer_steps",
                    "backward_rows",
                    "unknown_partial_prediction_rows",
                )
            ),
            "Original audit work is incomplete",
        )
        _need(
            receipt["stream"].get("structural_passed") is True
            and receipt["stream"].get("complete_journal") is True
            and receipt["stream"].get("unknown_partial_controls") == 0
            and receipt["stream"].get("pending_trace_alias") is None,
            "Original saved stream was not completely certified",
        )
        _need(
            failure.get("schema") == "walker-c1-audit-launcher-failure-v1"
            and failure.get("overall_audit_closure_passed") is False
            and failure.get("audit_coordinator_failed") is True
            and failure.get("original_audit_exit_exists") is False
            and failure.get("original_child_exit_code") is None
            and failure.get("coordinator_exit_code") == 1,
            "Original failure was not preserved",
        )
        _need(
            _binding(failure["scientific_receipt"])
            == _binding(protocol["anchors"]["scientific_receipt"]),
            "Failure/receipt binding differs",
        )
        _need(
            _binding(catalog["failure_companion"]) == _binding(protocol["anchors"]["failure"]),
            "Output catalog/failure binding differs",
        )
        receipt_relative = PurePosixPath(protocol["anchors"]["scientific_receipt"]["file"])
        audit_relative = receipt_relative.parent.parent
        _need(
            not (reader.roots["artifacts"] / str(audit_relative) / "exit.json").exists(),
            "Original missing exit receipt has been replaced",
        )
        _need(
            Path(catalog["audit_outer"]).resolve()
            == reader.roots["artifacts"] / str(audit_relative),
            "Audit-output root differs",
        )
        _need(
            catalog["file_count"] == len(catalog["files"])
            and catalog["file_bytes"] == sum(v["bytes"] for v in catalog["files"].values()),
            "Audit-output catalog census differs",
        )
        _need(
            protocol["cumulative"]["prior_wall_upper_seconds"]
            >= catalog["accounting"]["previous_wall_seconds"]
            + catalog["accounting"]["helper_wall_seconds_through_hashing"],
            "Prior wall debit understates known work",
        )
        _need(
            protocol["cumulative"]["prior_read_upper_bytes"]
            >= catalog["accounting"]["cumulative_read_bytes"],
            "Prior read debit understates known work",
        )
        _need(
            len(receipt["input_files"]) == expected["input_files"], "Original input census differs"
        )
        for relative, binding in receipt["input_files"].items():
            reader.add("run", relative, binding, "original scientific receipt input")
        for relative, binding in catalog["files"].items():
            reader.add(
                "artifacts", str(audit_relative / relative), binding, "closed audit output catalog"
            )
        for key in ("traceback", "gnu_time"):
            reader.add("artifacts", failure[key]["file"], failure[key], "original failed launcher")
        _need(
            "Pinned Linux worker RSS could not be measured"
            in reader.read("artifacts", failure["traceback"]["file"]).decode(),
            "Original traceback differs",
        )
        _need(
            "Exit status: 1" in reader.read("artifacts", failure["gnu_time"]["file"]).decode(),
            "Original coordinator exit differs",
        )
        worker = reader.json("run", "worker_result.json")
        config = reader.json("run", "config.json")
        _need(
            worker["status"] == "COMPLETED_FIXED_C1_ROSTER" and worker.get("error") is None,
            "Original worker did not complete",
        )
        _need(
            worker.get("parent_refits") == worker.get("teacher_reacquisitions") == 0
            and worker.get("followup_launched") is False,
            "Original forbidden work differs",
        )
        for name, key in (
            ("config.json", "config_sha256"),
            ("worker_result.json", "worker_result_sha256"),
        ):
            _need(
                receipt["input_files"][name]["sha256"] == receipt[key],
                "Original result/config binding differs",
            )
        registry = reader.json("run", "worker/actor_registry.json")
        _need(len(registry) == expected["actors"], "Actor registry count differs")
        paths = set()
        for alias, row in registry.items():
            _need(
                alias == f"s{row['stream']}_{row['arm']}"
                and row["arm"] in {"parent", "uniform", "cost_effect", "shuffled"},
                "Actor registry join differs",
            )
            path = reader.path("run", row["file"])
            _need(row["file"] not in paths, "Checkpoint paths are not distinct")
            paths.add(row["file"])
            reader.add(
                "run",
                row["file"],
                dict(bytes=path.stat().st_size, sha256=row["sha256"]),
                "original audited actor registry",
            )
        for name, digest in receipt["stream"]["file_hashes"].items():
            reader.add(
                "run",
                "worker/evidence/" + name,
                dict(bytes=receipt["accounting"]["evidence_file_bytes"][name], sha256=digest),
                "original saved-stream hash",
            )
        # Reserve the entire hash union and both journal parses before large reads.
        journals = (
            reader.path("run", "worker/evidence/ledger.jsonl"),
            reader.path("artifacts", str(audit_relative / "result/predictions.jsonl")),
        )
        reader.reserve(
            sum(v["bytes"] for v in reader.roster.values())
            + sum(p.stat().st_size for p in journals)
            + 32 * 1024**2
        )
        verified = reader.hash_roster()
        predictions = _predictions(
            reader,
            str(audit_relative / "result/predictions.jsonl"),
            expected,
            {v["actor_hash"] for v in registry.values()},
        )
        _need(
            (
                receipt["prediction_calls"],
                receipt["prediction_rows"],
                receipt["completed_prediction_rows"],
            )
            == (predictions["calls"], predictions["rows"], predictions["rows"]),
            "Prior prediction receipt census differs",
        )
        for key in ("independent_unknown_partial", "independent_returned_unpersisted"):
            _need(_zero_counts(receipt["accounting"][key]), "Original partial work remains")
        _need(
            receipt["accounting"]["complete_reconciliation"] is True,
            "Original reconciliation incomplete",
        )
        journal = _worker_journal(
            reader, receipt, _integer(config["task"]["frame_skip"], "frame skip", 1)
        )
        _need(
            worker["costs"]["requested"] == journal["requested"]
            and worker["costs"]["completed"] == journal["completed"],
            "Worker result work differs",
        )
        outcomes = _scalar_outcomes(reader, worker, receipt, registry, expected, config)
        fitting = _curves(reader, worker, expected, config)
        assays = _assays(
            reader, str(audit_relative / "result/assay_scalars.csv"), expected, config, registry
        )
        _need(
            outcomes["teacher_controls"] + outcomes["controls"] + assays["controls"]
            == journal["completed"]["native_controls"],
            "Saved teacher/assay/student geometry differs from paid native controls",
        )
        _need(
            receipt["teacher_replays"] == expected["teacher_replays"]
            and receipt["assays"] == expected["assays"]
            and receipt["standalone_episodes"] == expected["standalone"],
            "Scientific roster differs",
        )
        reader.unchanged()
        catalog_payload = _canonical(
            dict(schema="walker-c1-recovery-verified-files-v1", files=verified)
        )
        _need(
            len(catalog_payload) + 65536 <= protocol["limits"]["output_bytes"],
            "Recovery output budget exhausted",
        )
        with (out / "verified_files.json").open("xb") as stream:
            stream.write(catalog_payload)
            stream.flush()
            os.fsync(stream.fileno())
        result.update(
            recovery_checks_passed=True,
            admitted_original_scientific_checks=True,
            retained_numerical_certification=True,
            verified_file_count=len(verified),
            files_catalog=dict(
                file="verified_files.json",
                bytes=len(catalog_payload),
                sha256=hashlib.sha256(catalog_payload).hexdigest(),
            ),
            prediction_journal=predictions,
            original_worker_journal=journal,
            saved_outcomes=outcomes,
            saved_fitting=fitting,
            saved_assays=assays,
            original_scientific_receipt=_binding(protocol["anchors"]["scientific_receipt"]),
        )
    except BaseException as error:
        result.update(error=dict(type=type(error).__name__, message=str(error)))
    result["accounting"] = dict(
        read_bytes=0 if reader is None else reader.read_bytes,
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=_rss(),
        output_bytes_before_receipt=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),
    )
    result["work"] = dict(
        result["accounting"],
        hashed_files=0 if reader is None else reader.hashed_files,
        parsed_journal_lines=0 if reader is None else reader.journal_lines,
        parsed_JSON_documents=0 if reader is None else reader.json_documents,
        verified_files=result.get("verified_file_count", 0),
    )
    payload = _canonical(result)
    maximum = protocol.get("limits", {}).get("output_bytes", 64 * 1024**2)
    _need(
        result["accounting"]["output_bytes_before_receipt"] + len(payload) <= maximum,
        "Recovery final receipt exceeds output budget",
    )
    with (out / "recovery.json").open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return result
