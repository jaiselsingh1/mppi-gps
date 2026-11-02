"""Independently reconcile copied H5 scalar evidence and the closed audit.

This is a bounded standard-library metadata/report review. It does not repeat
the scientific audit, open external arrays/checkpoints, or perform inference,
physics, fitting, SSH or Git operations. Three source texts are read only to
explain the final-close storage timing and are checked against the manifest.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import math
import re
import struct
import time
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
REPO = ART.parents[1]
BASE = ART / "received_bundle_001/unpacked"
PACKET = BASE / "collection/packet"
REPORT = BASE / "report_render/result"
OUT = ART / "final_independent_review_001.json"
MD = ART / "final_independent_review_001.md"
ARMS = ("independent", "selector_only", "proposal_guided")
HEX = re.compile(r"[0-9a-f]{64}\Z")
EPISODE = re.compile(r"teacher_s([0-2])_(independent|selector_only|proposal_guided)_([0-3][0-9])\Z")


class Review:
    def __init__(self):
        self.started = time.perf_counter()
        self.cache = {}
        self.inputs = {}
        self.read_bytes = 0
        self.checks = 0
        self.notes = []
        if OUT.exists() or MD.exists():
            raise FileExistsError("Independent review is create-only")

    def need(self, condition, message):
        self.checks += 1
        if not condition:
            raise ValueError(message)

    def read(self, path):
        path = Path(path).resolve()
        allowed_source = {
            (REPO / name).resolve()
            for name in (
                "src/gps/mppi_gps/runtime.py",
                "src/gps/mppi_gps/proposal_evidence.py",
                "src/gps/mppi_gps/proposal_native.py",
            )
        }
        self.need(
            path.is_relative_to(BASE.resolve())
            or path in allowed_source
            or path == Path(__file__).resolve(),
            "Review input escapes the permitted copied metadata/source scope",
        )
        if path not in self.cache:
            self.need(path.is_file() and not path.is_symlink(), "Missing/symlinked review input")
            before = path.stat()
            self.need(before.st_size <= 16 * 1024**2, "Oversized compact review input")
            with path.open("rb") as stream:
                payload = stream.read(16 * 1024**2 + 1)
            self.need(len(payload) <= 16 * 1024**2, "Compact input grew beyond its read bound")
            after = path.stat()
            self.need(
                (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "Compact input changed during read",
            )
            self.read_bytes += len(payload)
            self.need(self.read_bytes <= 128 * 1024**2, "Review read ceiling exceeded")
            self.cache[path] = payload
            self.inputs[str(path)] = dict(
                bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
            )
        return self.cache[path]

    def load(self, path):
        def reject(value):
            raise ValueError("Nonfinite copied JSON constant: " + value)

        return json.loads(self.read(path), parse_constant=reject)

    def bound(self, path, row):
        self.read(path)
        actual = self.inputs[str(Path(path).resolve())]
        self.need(
            actual == {key: row[key] for key in ("bytes", "sha256")},
            "Copied file binding differs: " + str(path),
        )

    def table(self, name):
        return list(csv.DictReader(io.StringIO(self.read(REPORT / name).decode())))

    def same_number(self, first, second, message):
        self.need(math.isclose(first, second, rel_tol=0.0, abs_tol=1e-12), message)

    def verify_files(self):
        packet_manifest = self.load(PACKET / "packet_manifest.json")
        files = packet_manifest["files"]
        actual = {str(path.relative_to(PACKET)) for path in PACKET.rglob("*") if path.is_file()}
        self.need(actual == set(files) | {"packet_manifest.json"}, "Packet roster differs")
        for name, row in files.items():
            self.need(
                not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Escaping packet path",
            )
            self.bound(PACKET / name, row)
        collection = self.load(BASE / "collection/receipt.json")
        self.need(collection["collection_complete"] is True, "Collector did not close")
        self.need(collection["packet_file_count"] == len(actual), "Collector file count differs")
        self.need(
            collection["packet_bytes"] == sum(len(self.read(PACKET / name)) for name in actual),
            "Collector byte count differs",
        )
        self.bound(BASE / "collection/packet.tar.gz", collection["archive"])
        receipt = self.load(REPORT / "receipt.json")
        report_files = {
            str(path.relative_to(REPORT)) for path in REPORT.iterdir() if path.is_file()
        }
        self.need(
            report_files == set(receipt["outputs"]) | {"receipt.json"}, "Report file roster differs"
        )
        for name, row in receipt["outputs"].items():
            self.bound(REPORT / name, row)
        for label, row in receipt["inputs"].items():
            path = PACKET / ("diagnostics.json" if label == "saved_diagnostics" else "run/" + label)
            self.bound(path, row)
        reporter = BASE / "helpers/report_teacher_proposal_edc0.py"
        self.need(
            hashlib.sha256(self.read(reporter)).hexdigest() == receipt["source_sha256"],
            "Saved report source differs",
        )
        self.need(
            receipt["source_read_bytes"] == sum(row["bytes"] for row in receipt["inputs"].values()),
            "Report source read sum differs",
        )
        for name, count in receipt["table_rows"].items():
            self.need(len(self.table(name)) == count, "Report table count differs: " + name)
        self.need(
            self.read(REPORT / "learning_curves.png").startswith(b"\x89PNG\r\n\x1a\n"),
            "PNG header differs",
        )
        self.need(
            self.read(REPORT / "learning_curves.pdf").startswith(b"%PDF"), "PDF header differs"
        )
        for key in (
            "raw_arrays_decoded",
            "checkpoints_loaded",
            "network_forwards",
            "physics_controls",
            "optimizer_steps",
        ):
            self.need(receipt[key] == 0, "Report performed scientific work")
        return packet_manifest, collection, receipt

    def closed(self, value, label):
        self.need(value["exit_code"] == 0 and value["stopping_reason"] is None, label + " failed")
        self.need(
            value["unknown_partial_work_possible"] is False and value["retries"] == 0,
            label + " closure incomplete",
        )

    def run(self):
        packet_manifest, collection, report_receipt = self.verify_files()
        config = self.load(PACKET / "run/config.json")
        manifest = self.load(PACKET / "run/manifest.json")
        seal = self.load(PACKET / "run/input_manifest.json")
        worker = self.load(PACKET / "run/worker_result.json")
        worker_exit = self.load(PACKET / "run/worker_exit.json")
        audit = self.load(PACKET / "audit/result/receipt.json")
        audit_exit = self.load(PACKET / "audit/exit.json")
        ownership = self.load(PACKET / "audit/ownership.json")
        diag = self.load(PACKET / "diagnostics.json")
        external = self.load(PACKET / "external_catalog.json")
        audit_outputs = self.load(BASE / "external_audit_outputs_catalog.json")
        self.closed(worker_exit, "Worker outer watchdog")
        self.closed(audit_exit, "Audit outer watchdog")
        self.closed(self.load(BASE / "report_render/exit.json"), "Report outer watchdog")
        self.need(
            audit["scientific_round_audit_complete"] is True
            and audit["prediction_replay_complete"] is True,
            "Full scientific audit incomplete",
        )
        self.need(
            worker["status"] == "teacher_competence_negative_no_child_fits"
            and worker["execution_complete"] is True,
            "Unexpected worker closure",
        )
        self.need(
            worker["comparison_complete"] is False
            and worker["confirmed_benefit"] is False
            and worker["next_round_eligible"] is False,
            "Unmeasured learning was promoted",
        )
        self.need(
            worker["evaluations"] == worker["paired_teacher_student_gap"] == {},
            "Unrequested standalone/gap roster",
        )
        self.need(
            config["launch_authorized"] is True
            and config["eligibility"]["useful_learning_prerequisite_satisfied"] is False
            and config["eligibility"]["second_round_authorized"] is False,
            "Authorization or prerequisite claim differs",
        )
        for name, key in (
            ("config.json", "config_sha256"),
            ("manifest.json", "source_manifest_sha256"),
            ("input_manifest.json", "input_manifest_sha256"),
            ("worker_result.json", "worker_result_sha256"),
        ):
            self.need(
                ownership[key] == self.inputs[str((PACKET / "run" / name).resolve())]["sha256"],
                "Audit ownership binding differs",
            )
        self.need(
            manifest["config_sha256"] == seal["config_sha256"] == packet_manifest["config_sha256"],
            "Frozen config chain differs",
        )
        self.need(
            seal["source_manifest_sha256"] == ownership["source_manifest_sha256"],
            "Frozen source seal differs",
        )
        self.need(
            diag["audit_receipt"]["sha256"]
            == self.inputs[str((PACKET / "audit/result/receipt.json").resolve())]["sha256"],
            "Diagnostic audit binding differs",
        )
        self.need(
            diag["source_ledger"]["sha256"] == audit["file_hashes"]["ledger.jsonl"],
            "Diagnostic ledger binding differs",
        )
        self.need(
            audit["outcomes"]
            == dict(
                teacher_complete=True,
                learning_contrast_measured=False,
                standalone_panels_measured=False,
                confirmed_benefit=False,
                loop_closed=False,
            ),
            "Audited outcome scope differs",
        )

        checkpoints = {row["name"]: row for row in worker["checkpoint_records"]}
        self.need(
            set(checkpoints)
            == {f"s{s}_{point}" for s in range(3) for point in ("parent_initial", "parent")},
            "Unexpected checkpoint roster",
        )
        source_rows = {row["file"]: row["sha256"] for row in manifest["files"]}
        prep = audit["preparation"]["bindings"]
        for name, digest in source_rows.items():
            self.need(
                external["files"]["source/" + name]["sha256"]
                == prep["source/" + name]["sha256"]
                == digest,
                "Frozen source binding differs",
            )
        for row in checkpoints.values():
            declared = external["files"][row["file"]]
            self.need(
                all(declared[key] == row[key] for key in ("bytes", "sha256")),
                "Checkpoint external catalog differs",
            )
        self.need(
            audit["audit_checkpoint_loads"] == len(checkpoints) == 6,
            "Actual checkpoint audit count differs",
        )
        self.need(
            audit["checkpoint_read_bytes"] == sum(row["bytes"] for row in checkpoints.values()),
            "Checkpoint read byte count differs",
        )
        for name, row in external["files"].items():
            self.need(
                type(row["bytes"]) is int
                and row["bytes"] >= 0
                and HEX.fullmatch(row["sha256"]) is not None,
                "Malformed external declaration",
            )
        for name in ("arrays.bin", "arrays.index.jsonl", "ledger.jsonl"):
            self.need(
                external["files"]["evidence/" + name]["sha256"] == audit["file_hashes"][name],
                "External raw catalog differs from the full audit",
            )

        events = diag["events"]
        self.need(
            len({row["id"] for row in events}) == len(events) == 84,
            "Duplicate or missing scalar event",
        )
        episodes = {row["tag"]: row for row in events if row["event"] == "episode_complete"}
        metrics = {row["tag"]: row for row in events if row["event"] == "policy_metric_complete"}
        agreement_events = [row for row in events if row["event"] == "teacher_agreement_complete"]
        self.need(
            len(episodes) == 36 and len(metrics) == 12 and len(agreement_events) == 36,
            "Saved scalar roster differs",
        )
        episode_rows, falls = [], []
        for tag, row in episodes.items():
            match = EPISODE.fullmatch(tag)
            self.need(match is not None, "Unexpected teacher episode tag")
            sid, arm, reset = int(match[1]), match[2], int(match[3])
            self.need(
                reset < 4 and row["seed"] == config["streams"][sid]["acquisition_seeds"][reset],
                "Acquisition reset binding differs",
            )
            n = row["steps"]
            self.need(row["complete"] is True and 0 < n <= 1000, "Incomplete teacher event")
            self.need(
                row["success"] == (row["survived"] and n == 1000 and row["mae"] <= 0.3),
                "Walking qualification differs",
            )
            absorbing = 0.0 if row["survived"] else 5.0 * (1000 - n)
            self.same_number(row["absorbing_cost"], absorbing, "Absorbing tail differs")
            self.same_number(
                row["terminal_cost"], 0.0 if row["survived"] else 20.0, "Fall terminal differs"
            )
            self.same_number(row["terminal_adjustment"], 0.0, "Native terminal was charged again")
            raw = row["raw"]["arrays"]
            for name, shape in (
                ("observations", [n, 17]),
                ("actions", [n, 6]),
                ("states", [n + 1, 19]),
                ("warmstarts", [n + 1, 9]),
                ("integrations", [n + 1, 91]),
                ("costs", [n]),
                ("velocities", [n]),
            ):
                self.need(
                    raw[name]["shape"] == shape and raw[name]["dtype"] == "<f8",
                    "Episode array descriptor shape/dtype differs",
                )
            for name in ("terminal_cost", "terminal_adjustment", "absorbing_cost"):
                self.need(
                    raw[name]["sha256"] == hashlib.sha256(struct.pack("<d", row[name])).hexdigest(),
                    "Saved scalar/array descriptor binding differs",
                )
            detail = dict(
                tag=tag,
                stream=sid,
                arm=arm,
                reset_index=reset,
                seed=row["seed"],
                controls=n,
                walking=row["success"],
                full_survival=row["survived"],
                mae=row["mae"],
                complete_cost=row["cost"],
                executed_prefix_cost_including_native_terminal=row["cost"]
                - row["absorbing_cost"]
                - row["terminal_adjustment"],
                native_terminal_cost=row["terminal_cost"],
                terminal_adjustment=row["terminal_adjustment"],
                absorbing_tail_cost=row["absorbing_cost"],
            )
            episode_rows.append(detail)
            if not row["survived"]:
                falls.append(detail)
        self.need(
            {(row["stream"], row["arm"], row["reset_index"], row["controls"]) for row in falls}
            == {(1, "selector_only", 0, 201), (2, "proposal_guided", 0, 199)},
            "Unexpected fall roster",
        )

        cohorts, totals = [], {}
        for sid in range(3):
            base = worker["teacher_summaries"][str(sid)]["independent"]["mean_cost"]
            for arm in ARMS:
                rows = [episodes[f"teacher_s{sid}_{arm}_{index:02d}"] for index in range(4)]
                costs = [row["cost"] for row in rows]
                derived = dict(
                    episodes=4,
                    successes=sum(row["success"] for row in rows),
                    survived=sum(row["survived"] for row in rows),
                    controls=sum(row["steps"] for row in rows),
                    mean_cost=sum(costs) / 4,
                    costs=costs,
                    complete=True,
                )
                self.need(
                    derived == worker["teacher_summaries"][str(sid)][arm],
                    "Recomputed teacher summary differs",
                )
                gates = dict(
                    walking=all(row["success"] for row in rows),
                    cost=derived["mean_cost"] <= 1.05 * base,
                )
                self.need(
                    gates == worker["teacher_admission"][str(sid)][arm], "Teacher gate differs"
                )
                cohorts.append(
                    dict(
                        stream=sid,
                        arm=arm,
                        **derived,
                        cost_ratio_vs_independent=derived["mean_cost"] / base,
                        gates=gates,
                    )
                )
        for arm in ARMS:
            rows = [row for row in episode_rows if row["arm"] == arm]
            totals[arm] = dict(
                episodes=len(rows),
                walking=sum(row["walking"] for row in rows),
                full_survival=sum(row["full_survival"] for row in rows),
                controls=sum(row["controls"] for row in rows),
                mean_complete_cost=sum(row["complete_cost"] for row in rows) / len(rows),
                mean_trajectory_mae=sum(row["mae"] for row in rows) / len(rows),
                complete_cost_sum=sum(row["complete_cost"] for row in rows),
                absorbing_cost_sum=sum(row["absorbing_tail_cost"] for row in rows),
            )

        metrics_rows = []
        for sid in range(3):
            for split, n in (("train", 4000), ("validation", 2000)):
                values = []
                for point in ("parent_initial", "parent"):
                    row = metrics[f"s{sid}_{point}/{split}"]
                    self.need(
                        row["rows"] == n and row["batch_size"] == 1024,
                        "Metric row/batch binding differs",
                    )
                    self.need(
                        row["actor_hash"] == checkpoints[f"s{sid}_{point}"]["actor_hash"],
                        "Metric checkpoint actor differs",
                    )
                    self.need(
                        row["observations"]
                        == dict(file=config["inputs"][split]["file"], key="observations")
                        and row["targets"]
                        == dict(file=config["inputs"][split]["file"], key="actions"),
                        "Metric target/source differs",
                    )
                    values.append(row["action_mse"])
                metrics_rows.append(
                    dict(
                        stream=sid,
                        split=split,
                        rows=n,
                        initial_action_mse=values[0],
                        parent_2k_action_mse=values[1],
                        change=values[1] - values[0],
                        relative_change_percent=100 * (values[1] / values[0] - 1),
                    )
                )
        curve_rows = []
        for record in worker["fit_records"]:
            evidence = self.load(PACKET / "run" / record["file"])
            self.need(
                evidence["completed_updates"]
                == evidence["cumulative_updates"]
                == evidence["fit_updates"]
                == 2000
                and evidence["failed"] is False,
                "Parent fit block incomplete",
            )
            self.need(
                evidence["fit_data"]["rows"] == 4000 and evidence["parameter_count"] == 19590,
                "Parent fit geometry differs",
            )
            self.need(
                evidence["child_policy_hash"] == checkpoints[record["tag"]]["actor_hash"],
                "Parent endpoint owner differs",
            )
            self.need(
                evidence["parent_policy_hash"]
                == evidence["anchor_policy_hash"]
                == checkpoints[record["tag"] + "_initial"]["actor_hash"],
                "Parent initial/anchor owner differs",
            )
            self.need(
                evidence["optimizer_continued"] is False
                and evidence["resume_fit"] is False
                and evidence["anchor_refreshes"] == [],
                "Fresh parent fitting continuation differs",
            )
            self.need(len(evidence["learning_curve"]) == 2000, "Missing parent learning points")
            for index, point in enumerate(evidence["learning_curve"], 1):
                self.need(
                    point["update"] == point["fit_update"] == index,
                    "Parent update sequence differs",
                )
                curve_rows.append(dict(artifact=record["file"], tag=record["tag"], **point))
        self.need(
            len(curve_rows) == 6000 and len(worker["fit_records"]) == 3,
            "Unexpected child fit or missing parent fit",
        )
        self.need(
            audit["fitting"]
            == dict(
                complete_fit_blocks=3,
                replayed_minibatch_indices=1536000,
                private_rng_checks=3,
                training_forward_rows=0,
                adam_updates_reproduced=0,
            ),
            "Audit fit ownership/replay census differs",
        )

        by_raw = {json.dumps(row["raw"], sort_keys=True): row for row in episodes.values()}
        agreements = []
        for event in agreement_events:
            row = by_raw[json.dumps(event["episode_raw"], sort_keys=True)]
            match = EPISODE.fullmatch(row["tag"])
            self.need(
                event["rows"] == row["steps"]
                and event["actor_hash"] == checkpoints[f"s{match[1]}_parent"]["actor_hash"],
                "Posthoc teacher/parent/row join differs",
            )
            predictions = event["predictions"]["reference"]["arrays"][event["predictions"]["key"]]
            residual = event["squared_disagreement"]["reference"]["arrays"][
                event["squared_disagreement"]["key"]
            ]
            self.need(
                predictions["shape"] == [row["steps"], 6] and predictions["dtype"] == "<f4",
                "Actual prediction dtype/shape changed",
            )
            self.need(
                residual["shape"] == [row["steps"]] and residual["dtype"] == "<f8",
                "Teacher error reduction descriptor differs",
            )
            agreements.append(
                dict(
                    episode=row["tag"],
                    stream=int(match[1]),
                    arm=match[2],
                    controls=row["steps"],
                    D=event["action_agreement"],
                )
            )
        agreement_aggregates = []
        for arm in ARMS:
            rows = [row for row in agreements if row["arm"] == arm]
            agreement_aggregates.append(
                dict(
                    arm=arm,
                    trajectories=len(rows),
                    trajectory_equal_D=sum(row["D"] for row in rows) / len(rows),
                    control_weighted_D=sum(row["D"] * row["controls"] for row in rows)
                    / sum(row["controls"] for row in rows),
                )
            )
        changed_pairs = []
        for sid in range(3):
            for index in range(4):
                first = episodes[f"teacher_s{sid}_independent_{index:02d}"]
                for arm in ARMS[1:]:
                    second = episodes[f"teacher_s{sid}_{arm}_{index:02d}"]
                    self.need(first["seed"] == second["seed"], "Comparison resets differ")
                    comparisons = {}
                    for key in ("observations", "actions", "states"):
                        one, two = first["raw"]["arrays"][key], second["raw"]["arrays"][key]
                        comparisons[key] = dict(
                            independent_sha256=one["sha256"],
                            guided_sha256=two["sha256"],
                            changed=one["sha256"] != two["sha256"],
                            independent_shape=one["shape"],
                            guided_shape=two["shape"],
                        )
                    changed_pairs.append(
                        dict(
                            stream=sid,
                            reset_index=index,
                            guided_arm=arm,
                            seed=first["seed"],
                            arrays=comparisons,
                        )
                    )
        self.need(
            len(changed_pairs) == 24
            and all(
                all(row["changed"] for row in pair["arrays"].values()) for pair in changed_pairs
            ),
            "Some matched experiences did not change",
        )

        mechanism_rows = []
        for sid, arms in diag["mechanism_summary"].items():
            for arm, values in arms.items():
                n = worker["teacher_summaries"][sid][arm]["controls"]
                self.need(values["decisions"] == n, "Mechanism decision count differs")
                for key in (
                    "selected_origin_counts",
                    "baseline_verification_status_counts",
                    "guided_verification_status_counts",
                ):
                    self.need(
                        sum(values[key].values()) == n
                        and all(
                            type(value) is int and value >= 0 for value in values[key].values()
                        ),
                        "Mechanism counts do not reconcile",
                    )
                self.need(
                    set(values["baseline_verification_status_counts"]) <= {"ok", "guard_fallback"}
                    and set(values["guided_verification_status_counts"])
                    <= {"ok", "guard_fallback"},
                    "Unexpected native/numerical verification error",
                )
                group = collections.Counter()
                for origin, count in values["selected_origin_counts"].items():
                    label = (
                        "protected_bank"
                        if ":protected:" in origin
                        else "proposal_bank"
                        if ":proposal:" in origin
                        else origin
                    )
                    group[label] += count
                chosen = "baseline" if arm == "independent" else "guided"
                self.need(
                    group["protected_guard_fallback"]
                    == values[chosen + "_verification_status_counts"].get("guard_fallback", 0),
                    "Fallback selection/status count differs",
                )
                mechanism_rows.append(
                    dict(
                        stream=int(sid),
                        arm=arm,
                        controls=n,
                        selected_origin_groups=dict(group),
                        selected_origins=values["selected_origin_counts"],
                        baseline_verification_status_counts=values[
                            "baseline_verification_status_counts"
                        ],
                        guided_verification_status_counts=values[
                            "guided_verification_status_counts"
                        ],
                    )
                )
        self.need(
            len(mechanism_rows) == 9 and sum(row["controls"] for row in mechanism_rows) == 34400,
            "Mechanism cohort roster differs",
        )

        for row in self.table("episode_outcomes.csv"):
            event = episodes[row["tag"]]
            for key in ("seed", "steps"):
                self.need(int(row[key]) == event[key], "Episode CSV integer differs")
            for key in ("cost", "mae", "terminal_cost", "terminal_adjustment", "absorbing_cost"):
                self.same_number(float(row[key]), event[key], "Episode CSV scalar differs")
            for key in ("success", "survived", "complete"):
                self.need(row[key] == str(event[key]), "Episode CSV outcome differs")
            self.same_number(
                float(row["executed_prefix_cost"]),
                event["cost"] - event["absorbing_cost"] - event["terminal_adjustment"],
                "Prefix CSV differs",
            )
        for row in self.table("full_data_metrics.csv"):
            self.same_number(
                float(row["action_mse"]), metrics[row["tag"]]["action_mse"], "Metric CSV differs"
            )
            self.need(row["reduction"] == "mean_over_rows_and_six_actions", "Metric units differ")
        agreed = {row["episode"]: row for row in agreements}
        for row in self.table("teacher_action_agreement.csv"):
            self.same_number(
                float(row["action_agreement"]), agreed[row["episode"]]["D"], "Teacher D CSV differs"
            )
            self.need(
                row["reduction"] == "mean_over_time_of_sum_six_squared_action_errors",
                "Teacher D units differ",
            )
        curves_by_key = {(row["artifact"], row["update"]): row for row in curve_rows}
        for row in self.table("learning_curves.csv"):
            actual = curves_by_key[(row["artifact"], int(row["update"]))]
            for key in (
                "fit_update",
                "action_mse",
                "proximal_loss",
                "total_loss",
                "unclipped_gradient_norm",
            ):
                self.same_number(float(row[key]), actual[key], "Learning CSV point differs")
        for row in self.table("mechanism_summary.csv"):
            values = diag["mechanism_summary"][row["stream"]][row["arm"]]
            self.need(int(row["decisions"]) == values["decisions"], "Mechanism CSV controls differ")
            for key in (
                "selected_origin_counts",
                "baseline_verification_status_counts",
                "guided_verification_status_counts",
            ):
                self.need(json.loads(row[key]) == values[key], "Mechanism CSV map differs")
        self.need(
            self.table("student_outcomes.csv") == self.table("paired_gap.csv") == [],
            "Unmeasured outcomes were filled with zeros",
        )

        work = worker["work"]
        for key, value in audit["structure"]["work"].items():
            self.need(work[key] == value, "Audited ledger and worker work differ")
        controls = sum(row["controls"] for row in episode_rows)
        expected_models = 24768 * controls + 24576 * 36
        self.need(
            expected_models == work["completed_model_controls"] == 852903936,
            "Model controls differ",
        )
        self.need(work["completed_live_controls"] == controls == 34400, "Live controls differ")
        self.need(
            work["completed_physics_substeps"] == (expected_models + controls) * 4 == 3411753344,
            "Physics substeps differ",
        )
        self.need(
            audit["audit_neural_rows"]
            == audit["audit_completed_neural_rows"]
            == work["completed_neural_rows"] - 1536000
            == 1085983492,
            "Independent prediction row census differs",
        )
        self.need(
            audit["audit_prediction_calls"]
            == worker["reservation_census"]["completed"]["total"]["prediction_queries"] - 6000
            == 103308,
            "Prediction call census differs",
        )
        self.need(
            audit["audit_unknown_partial_neural_rows"] == 0
            and not any(audit["audit_scope_unknown_partial_neural_rows"].values()),
            "Audit unknown partial work remains",
        )
        self.need(
            audit["audit_scope_requested_neural_rows"]
            == audit["audit_scope_completed_neural_rows"],
            "Scope predictions incomplete",
        )
        self.need(
            audit["new_native_controls"] == audit["new_optimizer_steps"] == 0,
            "Audit introduced native/learning intervention",
        )
        self.need(
            audit["scientific"]["selection_checks"]
            == audit["scientific"]["covered_teacher_controls"]
            == 34400,
            "Teacher controls lack selection audit",
        )
        self.need(
            audit["scientific"]["teacher_posthoc_checks"] == 36
            and audit["scientific"]["full_metric_checks"] == 12
            and audit["scientific"]["episodes"] == 36,
            "Scientific audit outcome/metric roster differs",
        )
        rng_draws = 2 * (controls + 36) * 256 * 48 * 6
        self.need(
            audit["scientific"]["independently_recreated_standard_normal_draws"]
            == rng_draws
            == 5077794816,
            "Sampler RNG census differs",
        )
        self.need(
            audit_outputs["files"]["result/work.jsonl"]["bytes"]
            == audit["journal_bytes"]
            == 41167444,
            "Audit journal size differs",
        )
        raw_files = {
            name: external["files"]["evidence/" + name]
            for name in ("arrays.bin", "arrays.index.jsonl", "ledger.jsonl")
        }
        closed_storage = sum(row["bytes"] for row in raw_files.values())
        storage_delta = closed_storage - work["storage_bytes"]
        self.need(
            closed_storage == 206888154706 and storage_delta == 1624,
            "Closed raw storage census differs",
        )
        source_matches = {}
        for name in (
            "src/gps/mppi_gps/runtime.py",
            "src/gps/mppi_gps/proposal_evidence.py",
            "src/gps/mppi_gps/proposal_native.py",
        ):
            payload = self.read(REPO / name)
            self.need(
                hashlib.sha256(payload).hexdigest() == source_rows[name],
                "Storage timing explanation source differs from frozen source",
            )
            source_matches[name] = source_rows[name]
        self.notes.append(
            "The final raw-file total exceeds worker.work.storage_bytes by1,624B. Frozen-source-matched worker code snapshots totals inside the meter context; exiting it then appends exactly one runtime_close ledger record. The source/metadata ordering explains this closing-metadata delta; this review did not read the external ledger tail. Both original values are retained."
        )
        self.notes.append(
            "The saved report uses the exact newer reporter hash de1ad2c0048ed64f5627a955c737dd79f94f46c6b0b929148f8be36643e66318; the frozen scientific source contains an older reporter. This is a separate saved-data reporting revision, not a scientific-source change."
        )
        self.notes.append(
            "The packet source manifest, audit ownership/preparation and external source catalog agree on all523 source files. Attribution to Git commit ff67495dae33ee43952677d0c0165a43f8d469b0 was established in the separate prelaunch source/Git review; this compact packet alone does not encode/recompute that Git association."
        )

        source_bytes = self.read(Path(__file__))
        review = dict(
            schema="walker-h5-final-independent-compact-review-v1",
            status="compact_evidence_review_passed_after_closed_scientific_audit",
            passed=True,
            full_scientific_audit_receipt_passed=True,
            review_repeated_full_scientific_audit=False,
            scope="Independent recomputation of saved scalar aggregates, outcome/gate logic, descriptor joins and compact report/accounting bindings. Raw/prediction/checkpoint verification is attributed to the separately completed hash-bound scientific audit, not repeated here.",
            checks=self.checks,
            teacher_aggregate_by_arm=totals,
            teacher_per_stream=cohorts,
            teacher_episode_rows=sorted(episode_rows, key=lambda row: row["tag"]),
            fall_details=falls,
            parent_metric_initial_to_2k=metrics_rows,
            parent_metric_units="Mean squared bounded native-control error over all dataset rows and six actions; predictions float32, observed controls float64; stored full-data reductions audited separately.",
            teacher_D_aggregates=agreement_aggregates,
            teacher_D_per_episode=agreements,
            teacher_D_units="Executed-time mean of the sum of six squared bounded native-control errors, with no division by six. Actual guide predictions float32, recorded actions float64, squared-disagreement array float64. Primary aggregation is trajectory equal; control-weighted values are separately labeled and never teaching gates.",
            changed_experience_pairs=changed_pairs,
            changed_experience_pair_count=24,
            changed_experience_interpretation="All24 matched I-vs-S/P episode pairs have different saved observation/action/state hashes already covered by the full audit. This establishes changed realized experience, not teachability, student benefit or a particular divergence time.",
            mechanism_cohorts=mechanism_rows,
            mechanism_scope="Counts are exact scalar-extracted source-ledger summaries and match rendered tables; actual inner updates, protected independence, selection/control joins and nontraining predictions were independently verified by the scientific audit.",
            audit_structure=audit["structure"],
            audit_scientific=audit["scientific"],
            audit_fitting=audit["fitting"],
            audit_outcomes=audit["outcomes"],
            worker_work=work,
            audit_work={
                key: value
                for key, value in audit.items()
                if key not in ("preparation", "structure", "scientific", "fitting", "outcomes")
            },
            actual_combined_nn=dict(
                forward_rows=work["completed_neural_rows"] + audit["audit_neural_rows"],
                forward_calls=109308 + audit["audit_prediction_calls"],
                backward_rows=1536000,
                adam_updates=6000,
                worker_checkpoint_loads=3,
                audit_checkpoint_loads=6,
                checkpoint_files_created=6,
            ),
            rng_accounting=dict(
                worker_standard_normal_draws_inferred=rng_draws,
                audit_standard_normal_draws_recreated=rng_draws,
                actual_worker_training_indices=1536000,
                auxiliary_expected_tape_indices=1536000,
                audit_indices_replayed=1536000,
                index_and_noise_draws_are_not_neural_rows=True,
            ),
            raw_evidence_files=raw_files,
            closed_raw_evidence_bytes=closed_storage,
            worker_preclose_evidence_bytes=work["storage_bytes"],
            closing_metadata_delta_bytes=storage_delta,
            frozen_sources_supporting_storage_timing=source_matches,
            external_declared_file_count=len(external["files"]),
            external_declared_bytes=sum(row["bytes"] for row in external["files"].values()),
            external_catalog_scope="536 declared external raw/source/checkpoint/input files, not an independently enumerated complete filesystem mirror. Checksums are reconciled to closed-audit declarations here; external file bytes were not transferred or reopened by this review.",
            external_audit_outputs=audit_outputs,
            worker_outer_closure=worker_exit,
            audit_outer_closure=audit_exit,
            report_receipt=report_receipt,
            no_measured_student_benefit=dict(
                child_fits=0,
                learning_contrast_measured=False,
                standalone_panels_measured=False,
                paired_gap_measured=False,
                standalone_walking=None,
                standalone_survival=None,
                standalone_cost=None,
                paired_gap=None,
                scope="No student performance zeros were fabricated; teacher stop prevented these measurements.",
            ),
            provenance_notes=self.notes,
            input_bindings=self.inputs,
            review_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            review_logical_file_read_bytes=self.read_bytes,
            review_wall_seconds=time.perf_counter() - self.started,
            review_scientific_work=dict(
                raw_array_decodes=0,
                checkpoints_loaded=0,
                neural_rows=0,
                native_controls=0,
                gradients=0,
                adam_updates=0,
                ssh_calls=0,
            ),
            cost_scope="Review counters include each unique copied input/source payload once through a bounded read cache. Earlier interactive inspection reads, import/kernel/stat work and final output writes are separate unmetered engineering overhead. Worker, audit, collection and render wall timers overlap internally and are not summed here. Billing/energy remain unknown.",
        )
        with OUT.open("x") as stream:
            json.dump(review, stream, indent=2, allow_nan=False)
            stream.write("\n")
        lines = [
            "# Final independent compact-evidence review",
            "",
            "The closed scientific audit passed, and copied scalar evidence/report/accounting bindings reconcile. This metadata review did not repeat raw-array or prediction replay.",
            "",
            "| Arm | Walking /12 | Full survival /12 | Controls | Mean complete cost | Trajectory-equal D |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            row = totals[arm]
            d = next(
                value["trajectory_equal_D"] for value in agreement_aggregates if value["arm"] == arm
            )
            lines.append(
                f"| {arm} | {row['walking']} | {row['full_survival']} | {row['controls']:,} | {row['mean_complete_cost']:.6f} | {d:.6f} |"
            )
        lines += [
            "",
            "| Stream | Split | Initial full-data MSE | 2k-parent MSE | Change |",
            "|---|---|---:|---:|---:|",
        ]
        for row in metrics_rows:
            lines.append(
                f"| {row['stream']} | {row['split']} | {row['initial_action_mse']:.6f} | {row['parent_2k_action_mse']:.6f} | {row['relative_change_percent']:+.3f}% |"
            )
        lines += [
            "",
            f"Audit replay: {audit['audit_neural_rows']:,} prediction rows /{audit['audit_prediction_calls']:,} calls /6 checkpoint loads; {audit['wall_seconds']:.6f}s; {audit['source_read_bytes']:,} archive-read bytes. No new physics or Adam updates. Worker training remains the3×2k parent warm-ups only.",
            "",
            "All24 matched I-vs-S/P experience pairs differ in saved observation/action/state hashes. All9 mechanism cohorts reconcile to34,400 executed teacher controls. Neither result establishes teachability or student usefulness.",
            "",
            f"Closed raw storage is {closed_storage:,}B; the original worker pre-close snapshot is {work['storage_bytes']:,}B. The1,624B difference is the post-snapshot runtime-close metadata write by frozen-source control flow; raw tail bytes were not reopened here.",
            "",
            "Child learning, standalone policy walking/survival/cost and paired teacher–student gaps were not measured. No follow-up or loop claim is admitted.",
            "",
            "Provenance qualifications:",
            *["- " + note for note in self.notes],
        ]
        with MD.open("x") as stream:
            stream.write("\n".join(lines) + "\n")
        print(
            json.dumps(
                dict(
                    passed=True,
                    json=str(OUT),
                    markdown=str(MD),
                    checks=self.checks,
                    logical_read_bytes=self.read_bytes,
                    teacher_aggregate=totals,
                    teacher_D_aggregates=agreement_aggregates,
                    audit_rows=audit["audit_neural_rows"],
                    closed_raw_bytes=closed_storage,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    Review().run()
