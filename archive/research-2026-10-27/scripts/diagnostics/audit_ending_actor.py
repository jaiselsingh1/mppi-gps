"""Independent raw-evidence auditor for the bounded ending/actor experiment.

No task, planner, learner, or trainer is imported. Recomputing each recorded
transition and analytic-bound step is additional paid verification work. The
append-only audit ledger keeps that work separate from the experimental ledger.
Existing evidence and audit outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


COMPONENTS = ("effort", "greed", "detour", "ending")
PHYSICS_FIELDS = {"states", "actions", "costs", "components"}
NONPHYSICS_KINDS = {"fit", "predict", "prediction", "model", "checkpoint"}


def _read(path):
    return json.loads(Path(path).read_text())


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _rows(path):
    with Path(path).open() as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                yield number, json.loads(line)


class RawAudit:
    def __init__(self, run_dir, audit_handle):
        self.run = Path(run_dir).resolve()
        self.config = _read(self.run / "config.json")
        self.task = self.config["task"]
        self.T = int(self.task["gate"])+int(self.task["delay"])+1
        self.handle = audit_handle
        self.checks = Counter()
        self.errors = []
        self.failed_check_count = 0
        self.maximum_errors = defaultdict(float)
        self.main_controls = Counter()
        self.audit_controls = Counter()
        self.main_by_phase = Counter()
        self.audit_by_phase = Counter()
        self.kind_counts = Counter()
        self.query_ids = set()
        self.query_info = {}
        self.artifact_hashes = {}
        self.audit_rows = 0
        self.physics_queries = 0
        self.bound_queries = 0
        self.external_unverified_controls = 0
        self.prediction_rows = 0
        self.fit_updates = 0
        self.audit_prediction_rows = 0
        self.actor_queries = {}
        self.planner_queries = {}
        self.executions = {}
        self.previous_planner = {}
        self.models = {}
        self.predictions = {}
        self.ledger_rows = []
        self.next_query = {}
        self.started = time.perf_counter()

    def check(self, condition, category, detail):
        self.checks[category] += 1
        if not bool(condition):
            self.failed_check_count += 1
            if len(self.errors) < 100:
                self.errors.append(f"{category}: {detail}")

    def equal(self, left, right, category, detail, tolerance=1e-12):
        left, right = np.asarray(left), np.asarray(right)
        correct = left.shape == right.shape
        if correct and left.size:
            correct = bool(np.isfinite(left).all() and np.isfinite(right).all())
            if correct:
                error = float(np.max(np.abs(left.astype(float)-right.astype(float))))
                self.maximum_errors[category] = max(self.maximum_errors[category], error)
                correct = error <= tolerance
        self.check(correct, category, detail)

    def path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.run / path

    def charge(self, row, controls, kind, started, **extra):
        scope = row["scope"]
        phase = row.get("phase", "unspecified")
        self.audit_controls[scope] += int(controls)
        self.audit_by_phase[f"{scope}/{phase}"] += int(controls)
        event = dict(id=self.audit_rows, source_query_id=row["id"], scope=scope,
                     phase=phase, kind=kind, controls=int(controls),
                     source_file=row.get("file"), wall_seconds=time.perf_counter()-started,
                     **extra)
        self.handle.write(json.dumps(_plain(event), separators=(",", ":"), allow_nan=False)+"\n")
        self.audit_rows += 1

    def state_range(self, states, detail):
        self.check(np.isfinite(states).all(), "finite_states", detail)
        clocks = states[..., 0]
        self.check(np.all((0 <= clocks) & (clocks <= self.T) & (clocks == np.floor(clocks))),
                   "state_clock_range", detail)
        self.check(np.all(np.abs(states[..., 1]) <= 1.), "state_position_range", detail)

    def expected(self, states, actions):
        """Independent vector arithmetic for already-recorded transition slots."""
        clock, y = states[..., 0], states[..., 1]
        action = actions[..., 0]
        active = clock < self.T
        movable = active & (clock < self.task["gate"])
        after_y = np.where(movable, np.clip(y+self.task["move"]*action, -1., 1.), y)
        after_t = np.minimum(clock+1., self.T)
        good = after_y <= self.task["ending_y"]
        ending = active & (clock == self.T-1)
        parts = np.stack((
            np.where(movable, self.task["effort"]*action**2, 0.),
            np.where(movable, .5*self.task["greed"]*(1.-after_y), 0.),
            np.where(active & (clock >= self.task["gate"])
                     & (clock < self.task["gate"]+self.task["detour_steps"]) & good,
                     self.task["detour"], 0.),
            np.where(ending & ~good, self.task["penalty"], 0.),
        ), axis=-1)
        flags = dict(ending_event=ending, terminated=after_t >= self.T,
                     ending_good=ending & good, good_lane=good)
        return np.stack((after_t, after_y), axis=-1), parts, flags

    def metadata(self, data):
        for name in ("metadata", "metadata_json"):
            if name in data and data[name].shape == () and data[name].dtype.kind in "US":
                return json.loads(str(data[name]))
        return {}

    def physics(self, row, data):
        started = time.perf_counter()
        states, actions = data["states"], data["actions"]
        detail = str(row["id"])
        if states.ndim < 2 or states.shape[-1] != 2 or actions.ndim < 2 or actions.shape[-1] != 1:
            self.check(False, "query_array_shapes", detail)
            self.charge(row, row["controls"], "failed_physics_verification", started,
                        failed=True, reason="invalid state/action dimensions; charged attempted slots")
            return
        h = actions.shape[-2]
        batch = actions.shape[:-2]
        count = int(np.prod(batch, dtype=np.int64)) if batch else 1
        slots = count*h
        self.physics_queries += 1
        shapes_ok = (states.shape == batch+(h+1, 2)
                     and data["costs"].shape == batch+(h,)
                     and data["components"].shape == batch+(h, 4))
        self.check(shapes_ok, "query_array_shapes", detail)
        self.check(slots == row["controls"], "query_slot_accounting", detail)
        self.check(np.isfinite(actions).all() and np.all(np.abs(actions) <= 1.),
                   "bounded_actions", detail)
        if not shapes_ok:
            self.charge(row, slots, "failed_physics_verification", started, failed=True,
                        reason="inconsistent trajectory shapes")
            return
        self.state_range(states, detail)
        before = states[..., :-1, :]
        expected_states, parts, flags = self.expected(before, actions)
        self.equal(states[..., 1:, :], expected_states, "transition_equations", detail)
        self.equal(data["components"], parts, "cost_component_equations", detail)
        self.equal(data["costs"], parts.sum(axis=-1), "total_cost_equations", detail)
        self.equal(data["costs"], data["components"].sum(axis=-1), "raw_component_sum", detail)
        for name, expected in flags.items():
            if name in data:
                self.equal(data[name], expected, "transition_flags", detail+" "+name, tolerance=0.)
        if "flags" in data:
            self.equal(data["flags"], np.stack([flags[n] for n in
                       ("ending_event", "terminated", "ending_good", "good_lane")], axis=-1),
                       "transition_flags", detail, tolerance=0.)
        if "returns" in data:
            self.equal(data["returns"], np.cumsum(data["costs"][..., ::-1], axis=-1)[..., ::-1],
                       "complete_suffix_sums", detail)
        if "totals" in data:
            self.equal(data["totals"], data["costs"].sum(axis=-1), "query_totals", detail)
        padding = int(np.count_nonzero(before[..., 0] >= self.T))
        metadata = {**self.metadata(data), **row}
        if "padding_controls" in metadata:
            self.check(metadata["padding_controls"] == padding, "padding_accounting", detail)
        for name in ("horizon", "h"):
            if name in metadata:
                self.check(int(metadata[name]) == h, "recorded_horizon", detail)
        remaining = self.T-states[..., 0, 0]
        for name in ("remaining", "remaining_start"):
            if name in metadata:
                supplied = np.broadcast_to(metadata[name], batch if batch else ())
                self.equal(supplied, remaining, "recorded_remaining", detail)
        ended = states[..., -1, 0] >= self.T
        for name in ("complete", "terminated"):
            if name in metadata:
                supplied = np.broadcast_to(metadata[name], batch if batch else ())
                self.equal(supplied, ended, "query_cutoff_flags", detail+" "+name, tolerance=0.)
        if "truncated" in metadata:
            supplied = np.broadcast_to(metadata["truncated"], batch if batch else ())
            self.check(not np.any(supplied & ended), "query_cutoff_flags", detail+" truncated and terminated")
        if metadata.get("cutoff") in {"terminal", "termination", "episode_end"}:
            self.check(np.all(ended), "query_cutoff_flags", detail+" terminal cutoff")
        if type(metadata.get("cutoff")) is bool:
            self.check(metadata["cutoff"] == bool(np.all(~ended)), "query_cutoff_flags", detail+" prefix cutoff")
        if "initial" in data:
            initial = np.broadcast_to(data["initial"], batch+(2,))
            self.equal(initial, states[..., 0, :], "query_initial_state", detail)
        self.query_info[str(row["id"])] = dict(row, horizon=h, slots=slots, complete=bool(np.all(ended)))
        if row["kind"] == "actor":
            theta = np.asarray(data["theta"], dtype=float)
            self.check(theta.shape == (3,), "actor_parameter_shape", detail)
            phi = np.stack((np.ones(before.shape[:-1]), before[..., 1], before[..., 0]/self.T), axis=-1)
            self.equal(actions[..., 0], np.tanh(phi@theta), "actor_action_equations", detail)
            self.check(self.actor_hash(theta) == row["controller_hash"], "actor_controller_hash", detail)
            self.actor_queries[detail] = dict(row, states=states.copy(), costs=data["costs"].copy(),
                theta=theta.copy(), total=float(np.sum(data["costs"])),
                success=bool(np.all(ended) and states[-1, 1] <= self.task["ending_y"]))
        elif row["kind"] == "mppi":
            self.planner(row, data)
        elif row["kind"] == "mppi_execute":
            parent = self.planner_queries[str(row["planner_query"])]
            self.equal(states[0], parent["initial"], "executed_planner_context", detail)
            self.equal(actions[0], parent["action"], "executed_planner_action", detail)
            self.equal(states[1], parent["first_next_state"], "executed_planner_successor", detail)
            self.executions[str(row["planner_query"])] = dict(row, states=states.copy(), cost=float(data["costs"].sum()))
        self.charge(row, slots, "physics_verification", started, padding_controls=padding,
                    batch_count=count, horizon=h, file_sha256=self.artifact_hashes[row["file"]])

    @staticmethod
    def actor_hash(theta):
        return hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest()

    def planner(self, row, data):
        """Pure RNG/weight/nominal reconstruction, with no additional physics."""
        spec, actions = row["planner"], data["actions"]
        count, h = spec["K"], actions.shape[1]
        initial, nominal = data["states"][0, 0], data["nominal_pre"]
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = json.loads(str(data["rng_pre"]))
        proposals = np.clip(nominal[None, :h]+rng.normal(0., spec["sigma"], (count, h, 1)), -1., 1.)
        proposals[0] = nominal[:h]
        for i, value in enumerate((0., 1., -1., .5, -.5), start=1):
            proposals[i] = value
        self.equal(actions[:count], proposals, "planner_private_rng_proposals", row["id"])
        self.check(rng.bit_generator.state == json.loads(str(data["rng_post"])),
                   "planner_private_rng_post", row["id"])
        totals = data["costs"].sum(axis=1)
        weights = np.exp(-(totals[:count]-totals[:count].min())/spec["temp"])
        weights /= weights.sum()
        self.equal(data["weights"], weights, "planner_exponential_weights", row["id"])
        mean = np.clip(np.sum(proposals*weights[:, None, None], axis=0), -1., 1.)
        self.equal(actions[-1], mean, "planner_verified_mean", row["id"])
        selected = int(data["selected"])
        self.check(selected == int(np.argmin(totals)), "planner_discrete_selection", row["id"])
        post = np.zeros((spec["H"], 1))
        post[:h-1] = actions[selected, 1:]
        self.equal(data["nominal_post"], post, "planner_retained_nominal", row["id"])
        self.check(row["controls"] == (count+1)*h and row["candidate_controls"] == count*h
                   and row["verification_controls"] == h, "planner_work_categories", row["id"])
        key = (row["scope"], row["episode"])
        previous = self.previous_planner.get(key)
        if previous:
            self.equal(nominal, previous["nominal"], "planner_nominal_continuity", row["id"])
            self.equal(initial, previous["first_next_state"], "planner_state_continuity", row["id"])
            self.check(json.loads(str(data["rng_pre"])) == previous["rng"],
                       "planner_rng_continuity", row["id"])
        else:
            seed = int(re.fullmatch(r"shared/seed(\d+)", row["scope"]).group(1))
            seeded = np.random.Generator(np.random.PCG64(seed*100000+row["reset_seed"]))
            self.check(json.loads(str(data["rng_pre"])) == seeded.bit_generator.state,
                       "planner_initial_private_seed", row["id"])
            self.equal(nominal, np.zeros_like(nominal), "planner_actor_independent_initialization", row["id"])
            initial_rng = np.random.Generator(np.random.PCG64(row["reset_seed"]))
            self.equal(initial, [0., initial_rng.uniform(.2, .4)], "planner_root_reset", row["id"])
        packet = dict(row, initial=initial.copy(), action=actions[selected, 0].copy(),
                      first_next_state=data["states"][selected, 1].copy(), selected_total=float(totals[selected]))
        self.planner_queries[str(row["id"])] = packet
        self.previous_planner[key] = dict(nominal=post.copy(), rng=rng.bit_generator.state,
                                         first_next_state=packet["first_next_state"])

    def model(self, relative):
        if relative not in self.models:
            path = self.path(relative)
            self.check(path.is_file(), "critic_packet_archived", str(path))
            with np.load(path, allow_pickle=False) as packet:
                value = {name: packet[name] for name in packet.files}
            meta = {name: value[name].item() for name in
                    ("total_steps", "gate", "ending_y", "target_actor_hash", "width", "ridge", "support_radius", "target_semantics")}
            meta["feature_schema"] = "[t/T,y/2]; intercept,linear,postgate,badlane,postgate*badlane,RBF"
            hasher = hashlib.sha256(json.dumps(meta, sort_keys=True).encode())
            for name in ("centers", "weights", "train_features"):
                array = np.ascontiguousarray(value[name], dtype="<f8")
                hasher.update(str(array.shape).encode())
                hasher.update(array.tobytes())
            hasher.update(json.dumps(value["train_groups"].tolist(), ensure_ascii=True).encode())
            self.check(hasher.hexdigest() == str(value["digest"]), "critic_packet_digest", relative)
            self.models[relative] = value
        return self.models[relative]

    def value_equations(self, packet, states):
        x = np.asarray(states)/[int(packet["total_steps"]), 2.]
        squared = np.sum((x[:, None]-packet["centers"][None])**2, axis=-1)
        post = (states[:, 0] >= int(packet["gate"])).astype(float)
        bad = (states[:, 1] > float(packet["ending_y"])).astype(float)
        phi = np.column_stack((np.ones(len(x)), x, post, bad, post*bad,
                               np.exp(-squared/(2*float(packet["width"])**2))))
        values = np.clip(phi@packet["weights"], 0., float(packet["cost_upper_bound"]))
        distances = np.sqrt(np.min(np.sum((x[:, None]-packet["train_features"][None])**2, axis=-1), axis=1))
        terminal = states[:, 0] == int(packet["total_steps"])
        values[terminal] = 0.
        supported = (distances <= float(packet["support_radius"])) | terminal
        return values, distances, supported

    def prediction(self, row, data):
        started = time.perf_counter()
        packet = self.model(row["model"])
        values, distances, supported = self.value_equations(packet, data["states_to_predict"])
        self.audit_prediction_rows += len(values)
        self.equal(data["values"], values, "critic_prediction_equations", row["id"])
        self.equal(data["distances"], distances, "critic_support_distances", row["id"])
        self.equal(data["supported"], supported, "critic_support_flags", row["id"], tolerance=0.)
        self.check(row["prediction_rows"] == len(values), "critic_prediction_row_accounting", row["id"])
        self.predictions[str(row["id"])] = dict(row, states=data["states_to_predict"].copy(),
            values=data["values"].copy(), supported=data["supported"].copy(),
            targets=data.get("targets"))
        self.charge(row, 0, "critic_prediction_verification", started, prediction_rows=len(values))

    def development(self, row):
        """Check copied raw development transitions; expose absent bound records."""
        started = time.perf_counter()
        simulator, arithmetic = 0, 0
        paths = sorted(self.path(row.get("raw", "development")).glob("*.json"))
        self.check(bool(paths), "development_raw_files", str(row["id"]))
        original_task, original_T = self.task, self.T
        try:
            for path in paths:
                record = _read(path)
                task = dict(original_task)
                # These two task overrides are explicit in the frozen test source.
                # Old fixture records omitted per-call configuration; no outcome is invented.
                if "test_rollout_scalar_parity" in record["test"]:
                    task.update(gate=2, delay=1, detour_steps=1)
                if "test_private_rng_final_window" in record["test"]:
                    task.update(effort=0., greed=0., detour=0., penalty=0.)
                calls_count = 0
                for index, call in enumerate(record["model_calls"]):
                    self.task = call.get("task", task)
                    self.T = int(self.task["gate"])+int(self.task["delay"])+1
                    states = np.asarray(call["states"], dtype=float)
                    actions = np.asarray(call["actions"], dtype=float)
                    if actions.shape == states.shape[:-1]:
                        actions = actions[..., None]
                    expected_states, parts, flags = self.expected(states, actions)
                    count = int(np.prod(states.shape[:-1], dtype=int))
                    calls_count += count
                    where = f"{path.name} call {index}"
                    self.check(count == call["controls"], "development_slot_accounting", where)
                    self.equal(call["next_states"], expected_states, "development_transition_equations", where)
                    actual_parts = np.stack([call["components"][n] for n in COMPONENTS], axis=-1)
                    self.equal(actual_parts, parts, "development_component_equations", where)
                    self.equal(call["total"], parts.sum(axis=-1), "development_total_equations", where)
                    for name, expected in flags.items():
                        self.equal(call["flags"][name], expected, "development_flag_equations", where, tolerance=0.)
                self.check(calls_count == record["simulator_controls"], "development_call_totals", path.name)
                simulator += calls_count
                arithmetic += record["arithmetic_verification_controls"]
        finally:
            self.task, self.T = original_task, original_T
        self.check(simulator+arithmetic == row["controls"], "development_global_accounting", str(row["id"]))
        self.external_unverified_controls += arithmetic
        self.charge(row, simulator+arithmetic, "development_verification", started,
                    raw_transition_controls_verified=simulator,
                    arithmetic_controls_conservatively_charged_without_raw_outputs=arithmetic,
                    configuration_source="per-call task if available; otherwise frozen test-name overrides")

    def bound(self, row, data):
        started = time.perf_counter()
        required = {"initial", "h", "bound", "stage_bounds", "max_y_by_step"}
        self.check(required <= set(data), "bound_fields", str(row["id"]))
        if not required <= set(data):
            self.charge(row, row["controls"], "failed_bound_verification", started, failed=True)
            return
        initial = np.asarray(data["initial"], dtype=float)
        h = int(data["h"])
        self.check(initial.shape == (2,) and h >= 0, "bound_shapes", str(row["id"]))
        if initial.shape != (2,) or h < 0:
            self.charge(row, row["controls"], "failed_bound_verification", started, failed=True)
            return
        self.state_range(initial, str(row["id"]))
        self.bound_queries += 1
        clock, y = initial
        positions, stages = [], []
        for _ in range(h):
            cost = 0.
            if clock < self.task["gate"]:
                y = min(1., y+self.task["move"])
                cost = .5*self.task["greed"]*(1.-y)
            clock = min(float(self.T), clock+1.)
            positions.append(y)
            stages.append(cost)
        self.check(row["controls"] == h, "bound_slot_accounting", str(row["id"]))
        self.equal(data["stage_bounds"], stages, "bound_stage_equations", str(row["id"]))
        self.equal(data["max_y_by_step"], positions, "bound_reachability_equations", str(row["id"]))
        self.equal(data["bound"], sum(stages), "bound_sum", str(row["id"]))
        eligible = bool(initial[0]+h <= self.task["gate"])
        if row.get("eligible") or row.get("optimizer_gap_eligible"):
            self.check(eligible, "bound_qualification_window", str(row["id"]))
        self.query_info[str(row["id"])] = dict(scope=row["scope"], kind="bound", file=row["file"],
            slots=h, pre_gate_eligible=eligible)
        self.charge(row, h, "bound_verification", started, pre_gate_eligible=eligible,
                    file_sha256=self.artifact_hashes[row["file"]])

    def manifest(self):
        manifest = _read(self.run / "manifest.json")
        self.check(isinstance(manifest.get("files"), list), "manifest_schema", "files must be a list")
        for entry in manifest.get("files", []):
            archive = self.path(entry["archive"])
            matches = archive.exists() and _hash(archive) == entry["sha256"]
            self.check(matches, "archived_source_hashes", entry["path"])

    def ledger(self):
        for number, row in _rows(self.run / "ledger.jsonl"):
            detail = f"line {number}, query {row.get('id')}"
            self.check({"id", "scope", "phase", "kind", "controls"} <= row.keys(), "ledger_schema", detail)
            if not {"id", "scope", "phase", "kind", "controls"} <= row.keys():
                continue
            key = str(row["id"])
            self.check(key not in self.query_ids, "unique_query_ids", key)
            self.query_ids.add(key)
            if self.ledger_rows:
                self.next_query[str(self.ledger_rows[-1]["id"])] = key
            self.ledger_rows.append(row)
            self.check(type(row["controls"]) is int and row["controls"] >= 0,
                       "ledger_control_counts", detail)
            if type(row["controls"]) is not int or row["controls"] < 0:
                continue
            self.main_controls[row["scope"]] += row["controls"]
            self.main_by_phase[f"{row['scope']}/{row['phase']}"] += row["controls"]
            self.kind_counts[row["kind"]] += 1
            self.prediction_rows += int(row.get("prediction_rows", 0))
            self.fit_updates += int(row.get("fit_updates", 0))
            attempted = time.perf_counter()
            before_audit_rows = self.audit_rows
            try:
                if row.get("file"):
                    path = self.path(row["file"])
                    self.check(path.is_file(), "query_artifact_exists", str(path))
                    if not path.is_file():
                        continue
                    self.artifact_hashes[row["file"]] = _hash(path)
                    if "sha256" in row:
                        self.check(row["sha256"] == self.artifact_hashes[row["file"]], "query_artifact_hash", detail)
                    with np.load(path, allow_pickle=False) as archive:
                        data = {name: archive[name] for name in archive.files}
                    if PHYSICS_FIELDS <= data.keys():
                        self.physics(row, data)
                    elif row["kind"] == "bound":
                        self.bound(row, data)
                    elif row["kind"] == "predict" and "states_to_predict" in data:
                        self.prediction(row, data)
                    else:
                        self.check(row["controls"] == 0 and row["kind"] in NONPHYSICS_KINDS,
                                   "unexpected_nonphysics_artifact", detail)
                    self.query_info.setdefault(key, dict(row))
                elif row["controls"] > 0 and row["kind"] == "development":
                    self.development(row)
                else:
                    self.check(row["controls"] == 0, "missing_physics_artifact", detail)
                    self.query_info[key] = dict(row)
            except Exception as error:
                self.check(False, "query_audit_exception", f"{detail}: {type(error).__name__}: {error}")
                if self.audit_rows == before_audit_rows:
                    self.charge(row, row["controls"], "failed_query_verification", attempted,
                                failed=True, reason=str(error), controls_are_conservative_attempt_charge=True)

    def references(self):
        path = self.run / "records.jsonl"
        if not path.exists():
            return
        reference_names = {"query", "query_id", "queries", "query_ids", "score_queries",
                           "full_query", "prefix_query", "tail_query", "reference_query"}

        def visit(value, location):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in reference_names or key.endswith("_query_id") or key.endswith("_query_ids"):
                        items = item if isinstance(item, list) else [item]
                        for identifier in items:
                            if identifier is not None and isinstance(identifier, (str, int)):
                                self.check(str(identifier) in self.query_ids, "record_query_references",
                                           f"{location}.{key}: {identifier}")
                    elif isinstance(item, (dict, list)):
                        visit(item, location+"."+key)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{location}[{index}]")

        for number, row in _rows(path):
            visit(row, f"records line {number}")

    def trace(self, identifier, theta=None, complete=True):
        tr = self.actor_queries[str(identifier)]
        if complete:
            self.check(tr["complete"], "complete_actor_outcome", str(identifier))
        if theta is not None:
            self.equal(tr["theta"], theta, "scored_actor_identity", str(identifier), tolerance=0.)
        return tr

    def prefix_score(self, ref, model_file, base_hash, candidate=None):
        prefix = self.trace(ref["prefix"], candidate, complete=False)
        model = self.model(model_file)
        self.check(str(model["target_actor_hash"]) == base_hash, "scorer_base_continuation", ref["prefix"])
        # Each prefix is immediately followed by its recorded prediction query.
        prediction = self.predictions[self.next_query[str(ref["prefix"])]]
        self.equal(prediction["states"], prefix["states"][-1:], "prefix_prediction_context", ref["prefix"])
        self.equal(ref["predicted"], prediction["values"][0], "prefix_prediction_reference", ref["prefix"])
        self.check(ref["supported"] == bool(prediction["supported"][0]), "prefix_support_reference", ref["prefix"])
        self.check(prediction["model"] == model_file, "prefix_model_version", ref["prefix"])
        if ref["fallback"] is not None:
            tail = self.trace(ref["fallback"])
            self.equal(tail["states"][0], prefix["states"][-1], "base_fallback_handoff", ref["prefix"])
            self.check(tail["controller_hash"] == base_hash and not ref["supported"],
                       "exact_frozen_base_fallback", ref["prefix"])
            used = tail["total"]
        else:
            self.check(ref["supported"], "supported_value_or_exact_fallback", ref["prefix"])
            used = float(prediction["values"][0])
        self.equal(ref["used_tail"], used, "used_tail_reference", ref["prefix"])
        return prefix["total"]+used

    def fit_records(self, records):
        updates = {(r["scope"], r["update"]): r for r in records if r["kind"] == "update"}
        for row in self.ledger_rows:
            if row["kind"] != "fit":
                continue
            packet = self.model(row["model"])
            traces = [self.trace(q) for q in row["target_queries"]]
            states = np.concatenate([t["states"][:-1] for t in traces])
            returns = np.concatenate([np.cumsum(t["costs"][::-1])[::-1] for t in traces])
            groups = np.concatenate([[str(t["ancestry"].get("reset_seed", t["ancestry"].get("episode")))]
                                     *len(t["costs"]) for t in traces])
            self.equal(packet["target_states"], states, "critic_target_state_provenance", row["model"])
            self.equal(packet["target_returns"], returns, "critic_measured_return_provenance", row["model"])
            self.equal(packet["train_features"], states/[self.T, 2.], "critic_training_only_features", row["model"])
            self.check(np.array_equal(packet["train_groups"], groups), "critic_root_groups", row["model"])
            self.check(packet["target_query_ids"].tolist() == row["target_queries"],
                       "critic_target_query_ids", row["model"])
            self.check(all(t["controller_hash"] == row["base_hash"] for t in traces)
                       and str(packet["target_actor_hash"]) == row["base_hash"],
                       "critic_frozen_target_actor", row["model"])
            self.check(row["digest"] == str(packet["digest"]), "critic_fit_digest", row["model"])
            update = updates[(row["scope"], row["update"])]
            self.check(row["base_hash"] == update["base_hash"], "critic_epoch_identity", row["model"])
            self.check(all(t["ancestry"].get("role") in {"train", "critic_train"} for t in traces),
                       "critic_training_role_exclusion", row["model"])
            hold = [r for r in records if r["kind"] == "critic_holdout"
                    and r["scope"] == row["scope"] and r["update"] == row["update"]]
            self.check(len(hold) == 1, "critic_heldout_record", row["model"])
            if hold:
                htraces = [self.trace(q) for q in hold[0]["queries"]]
                held_groups = {str(t["ancestry"].get("reset_seed", t["ancestry"].get("episode"))) for t in htraces}
                self.check(not set(groups) & held_groups, "critic_whole_root_holdout", row["model"])
                self.check(all(t["controller_hash"] == row["base_hash"] and
                               t["ancestry"].get("role") == "critic_holdout" for t in htraces),
                           "critic_heldout_continuation", row["model"])
                hs = np.concatenate([t["states"][:-1] for t in htraces])
                ht = np.concatenate([np.cumsum(t["costs"][::-1])[::-1] for t in htraces])
                predictions = [p for p in self.predictions.values() if p["scope"] == row["scope"]
                               and p["phase"] == "critic_holdout" and p["update"] == row["update"]]
                self.check(len(predictions) == 1, "critic_heldout_prediction", row["model"])
                if predictions:
                    self.equal(predictions[0]["states"], hs, "critic_heldout_state_provenance", row["model"])
                    self.equal(predictions[0]["targets"], ht, "critic_heldout_return_provenance", row["model"])
                    self.equal(hold[0]["mae"], np.mean(abs(predictions[0]["values"]-ht)),
                               "critic_heldout_error", row["model"])

    def update_records(self, records):
        fit_by_key = {(r["scope"], r["update"]): r for r in self.ledger_rows if r["kind"] == "fit"}
        previous = {}
        directions = {}
        sc = self.config["search"]
        for row in records:
            if row["kind"] != "update":
                continue
            scope, update = row["scope"], row["update"]
            base, candidates, eps = map(np.asarray, (row["base"], row["candidates"], row["eps"]))
            expected_seed = 1000000+row["seed"]*1000+update
            exact_eps = np.random.default_rng(expected_seed).standard_normal((sc["pairs"], 3))
            self.check(row["direction_seed"] == expected_seed, "candidate_direction_seed", scope)
            self.equal(eps, exact_eps, "candidate_direction_draws", scope, tolerance=0.)
            expected = np.empty_like(candidates)
            expected[0::2], expected[1::2] = base+sc["sigma"]*eps, base-sc["sigma"]*eps
            self.equal(candidates, expected, "candidate_antithetic_parameters", scope, tolerance=0.)
            self.equal(base, previous.get(scope, self.config["actor"]["initial_parameters"]),
                       "actor_update_continuity", scope, tolerance=0.)
            self.check(self.actor_hash(base) == row["base_hash"], "actor_base_hash", scope)
            direction_key = (row["seed"], update)
            if direction_key in directions:
                self.equal(eps, directions[direction_key], "matched_arm_perturbations", scope, tolerance=0.)
            directions[direction_key] = eps
            scores = np.asarray(row["scores"])
            if scope.endswith("/full_return"):
                calculated = [[self.trace(q, candidates[i])["total"] for q in refs]
                              for i, refs in enumerate(row["score_queries"])]
            else:
                model_file = fit_by_key[(scope, update)]["model"]
                calculated = [[self.prefix_score(ref, model_file, row["base_hash"], candidates[i]) for ref in refs]
                              for i, refs in enumerate(row["score_queries"])]
            self.equal(scores, calculated, "candidate_score_references", scope)
            means = scores.mean(axis=1)
            gradient = np.sum((means[0::2]-means[1::2])[:, None]*eps, axis=0)/(2*sc["pairs"]*sc["sigma"]*sc["cost_scale"])
            raw = -sc["lr"]*gradient
            norm = float(np.linalg.norm(raw))
            step = raw*(sc["maxnorm"]/norm) if norm > sc["maxnorm"] else raw
            proposal = row["proposal"]
            for name, value in (("gradient", gradient), ("unclipped_step", raw), ("step", step), ("parameters", base+step)):
                self.equal(proposal[name], value, "raw_cost_es_update", scope+" "+name)
            self.check(np.linalg.norm(step) <= sc["maxnorm"]+1e-12, "bounded_actor_step", scope)
            eligibility = np.linalg.norm(candidates-base, axis=1) <= sc["maxnorm"]
            best = int(np.argmin(np.where(eligibility, means, np.inf))) if eligibility.any() else None
            base_train = [self.trace(q, base) for q in row["base_train_queries"]]
            base_validation = [self.trace(q, base) for q in row["base_validation_queries"]]
            valid = []
            for choice in row["checked"]:
                theta = np.asarray(choice["parameters"])
                expected_parameters = base+step if choice["label"] == "bounded_ES" else candidates[best]
                self.equal(theta, expected_parameters, "evaluated_proposal_identity", scope)
                self.check(np.linalg.norm(theta-base) <= sc["maxnorm"]+1e-12, "bounded_actor_step", scope)
                train = [self.trace(q, theta) for q in choice["train_queries"]]
                validation = [self.trace(q, theta) for q in choice["validation_queries"]]
                for old, new in ((base_train, train), (base_validation, validation)):
                    self.check(len(old) == len(new), "paired_acceptance_roots", scope)
                    for a, b in zip(old, new):
                        self.equal(a["states"][0], b["states"][0], "paired_acceptance_starts", scope, tolerance=0.)
                tc, vc = np.mean([t["total"] for t in train]), np.mean([t["total"] for t in validation])
                ts, vs = np.mean([t["success"] for t in train]), np.mean([t["success"] for t in validation])
                margin = self.config["acceptance"]["cost_margin"]
                accepted = (tc < np.mean([t["total"] for t in base_train])-margin
                            and vc < np.mean([t["total"] for t in base_validation])-margin
                            and ts >= np.mean([t["success"] for t in base_train])
                            and vs >= np.mean([t["success"] for t in base_validation]))
                self.equal(choice["train_cost"], tc, "acceptance_raw_returns", scope)
                self.equal(choice["validation_cost"], vc, "acceptance_raw_returns", scope)
                self.equal(choice["validation_success"], vs, "acceptance_raw_successes", scope)
                self.check(choice["accepted_eligible"] == bool(accepted), "strict_full_return_acceptance", scope)
                if accepted:
                    valid.append(choice)
            winner = min(valid, key=lambda c:c["validation_cost"]) if valid else None
            self.check(row["winner"] == (winner["label"] if winner else None), "lowest_valid_proposal", scope)
            parameters = winner["parameters"] if winner else base
            self.equal(row["parameters"], parameters, "deployed_update_parameters", scope)
            self.check(row["actor_hash"] == self.actor_hash(parameters), "deployed_actor_hash", scope)
            previous[scope] = parameters

    def diagnostic_records(self, records):
        fits = {(r["scope"], r["update"]): r for r in self.ledger_rows if r["kind"] == "fit"}
        updates = {(r["scope"], r["update"]): r for r in records if r["kind"] == "update"}
        for row in records:
            if row["kind"] not in {"diagnostic", "proposal_diagnostic"}:
                continue
            update = updates[(row["scope"], row["update"])]
            model = row.get("model", fits[(row["scope"], row["update"])]["model"])
            if row["kind"] == "proposal_diagnostic":
                arrays = [row["A"], row["B"], row["C"]]
                groups = [(row["A_queries"], row["B_tail_queries"], row["prefix_references"], row["parameters"])]
                self.check(row["decision_already_fixed"], "diagnostics_excluded_from_acceptance", row["scope"])
                self.check(row["A_queries"] == next(c for c in update["checked"] if c["label"] == row["label"])["validation_queries"][:2],
                           "proposed_change_actual_returns", row["scope"])
            else:
                arrays = [row[n] for n in ("full_candidate", "exact_hybrid", "approximate_hybrid")]
                groups = list(zip(row["A_queries"], row["B_tail_queries"], row["prefix_references"], row["parameters"]))
            expected_a, expected_b, expected_c = [], [], []
            for a_refs, b_refs, prefix_refs, parameters in groups:
                a_values, b_values, c_values = [], [], []
                for aq, bq, ref in zip(a_refs, b_refs, prefix_refs):
                    a = self.trace(aq, parameters)
                    b = self.trace(bq)
                    prefix = self.trace(ref["prefix"], parameters, complete=False)
                    self.check(b["controller_hash"] == row["base_hash"], "diagnostic_exact_base_continuation", bq)
                    self.equal(b["states"][0], prefix["states"][-1], "diagnostic_B_handoff", bq)
                    self.equal(a["states"][0], prefix["states"][0], "diagnostic_A_start", aq)
                    a_values.append(a["total"]); b_values.append(prefix["total"]+b["total"])
                    c_values.append(self.prefix_score(ref, model, row["base_hash"], parameters))
                expected_a.append(a_values); expected_b.append(b_values); expected_c.append(c_values)
            if row["kind"] == "proposal_diagnostic":
                expected_a, expected_b, expected_c = expected_a[0], expected_b[0], expected_c[0]
            for actual, expected in zip(arrays, (expected_a, expected_b, expected_c)):
                self.equal(actual, expected, "diagnostic_A_B_C_return_references", row["scope"])
            if row["kind"] == "diagnostic":
                a, b, c = map(np.asarray, arrays)
                for name, expected in (("continuation_error", b-a), ("critic_error", c-b), ("combined_error", c-a),
                        ("full_pair_differences", a[0::2]-a[1::2]), ("exact_hybrid_pair_differences", b[0::2]-b[1::2]),
                        ("approximate_hybrid_pair_differences", c[0::2]-c[1::2])):
                    self.equal(row[name], expected, "diagnostic_error_decomposition", row["scope"]+" "+name)

    def record_checks(self):
        path = self.run / "records.jsonl"
        if not path.exists():
            return
        records = [r for _, r in _rows(path)]
        planner_episodes = {r["name"]: r for r in records if r["kind"] == "planner_episode"}
        for identifier, trace in self.actor_queries.items():
            ancestry = trace["ancestry"]
            if "episode" in ancestry:
                source = planner_episodes[ancestry["episode"]]
                self.equal(trace["states"][0], source["states"][ancestry["time"]],
                           "reached_state_ancestry", identifier, tolerance=0.)
            elif "parent_query" in ancestry:
                self.equal(trace["states"][0], self.actor_queries[ancestry["parent_query"]]["states"][-1],
                           "continuation_state_ancestry", identifier, tolerance=0.)
            else:
                rng = np.random.Generator(np.random.PCG64(ancestry["reset_seed"]))
                self.equal(trace["states"][0], [0., rng.uniform(.2, .4)], "actor_root_reset", identifier)
        self.fit_records(records)
        self.update_records(records)
        self.diagnostic_records(records)
        for row in records:
            if row["kind"] == "planner_episode":
                executions = [self.executions[q] for q in row["queries"]]
                states = np.concatenate((executions[0]["states"][:1], np.array([e["states"][1] for e in executions])))
                self.equal(row["states"], states, "planner_episode_raw_trace", row["name"])
                self.equal(row["total"], sum(e["cost"] for e in executions), "planner_episode_outcome", row["name"])
                self.check(row["success"] == bool(states[-1, 1] <= self.task["ending_y"]), "planner_episode_ending", row["name"])
            elif row["kind"] == "evaluation":
                traces = [self.trace(q, row["parameters"]) for q in row["queries"]]
                costs = [t["total"] for t in traces]; successes = [t["success"] for t in traces]
                self.equal(row["individual_cost"], costs, "evaluation_actual_costs", row["scope"])
                self.equal(row["individual_success"], successes, "evaluation_actual_endings", row["scope"])
                self.equal(row["cost"], np.mean(costs), "evaluation_actual_costs", row["scope"])
                self.equal(row["success"], np.mean(successes), "evaluation_actual_endings", row["scope"])
        summary = _read(self.run / "summary.json")
        self.check(summary["main_controls"] == sum(self.main_controls.values()),
                   "summary_control_total", "unique main work")
        self.check(summary["query_count"] == len(self.query_ids), "summary_query_total", "ledger rows")
        self.check(summary["scope_main_controls"] == dict(self.main_controls), "summary_scope_totals", "scope work")

    def budgets(self):
        global_scopes = {scope for scope in self.main_controls if scope in {"shared", "global", "shared/global"}}
        global_main = sum(self.main_controls[s] for s in global_scopes)
        global_audit = sum(self.audit_controls[s] for s in global_scopes)
        per_arm = {}
        cap = int(self.config.get("budget", {}).get("controls_per_seed_arm", 1_000_000))
        for scope in self.main_controls:
            match = re.fullmatch(r"seed(\d+)/([^/]+)", scope)
            if not match:
                self.check(scope in global_scopes or re.fullmatch(r"shared/seed\d+", scope),
                           "scope_schema", scope)
                continue
            shared = f"shared/seed{match.group(1)}"
            main = self.main_controls[scope]+self.main_controls[shared]+global_main
            verification = self.audit_controls[scope]+self.audit_controls[shared]+global_audit
            per_arm[scope] = dict(own_main_controls=self.main_controls[scope],
                shared_seed_main_controls=self.main_controls[shared], global_shared_main_controls=global_main,
                main_standalone_controls=main, audit_standalone_controls=verification,
                all_paid_standalone_controls=main+verification, cap=cap,
                within_cap=main+verification <= cap)
            self.check(main+verification <= cap, "all_paid_per_seed_arm_budget", scope)
        return per_arm


def audit(run_dir):
    run = Path(run_dir).resolve()
    destination, ledger_path = run / "audit.json", run / "audit_ledger.jsonl"
    if destination.exists() or ledger_path.exists():
        raise FileExistsError("Retaining existing audit.json/audit_ledger.jsonl; never overwrite audit work")
    started = time.perf_counter()
    with ledger_path.open("x", buffering=1) as handle:
        review = RawAudit(run, handle)
        try:
            review.manifest()
            review.ledger()
            review.references()
            review.record_checks()
            per_arm = review.budgets()
        except BaseException as error:
            review.check(False, "audit_exception", f"{type(error).__name__}: {error}")
            per_arm = {}
        result = dict(schema="ending-actor-raw-audit-v1", passed=not review.failed_check_count,
            errors=review.errors, failed_check_count=review.failed_check_count,
            checks=dict(review.checks), maximum_numeric_errors=dict(review.maximum_errors),
            main_unique_controls=sum(review.main_controls.values()),
            audit_unique_verification_controls=sum(review.audit_controls.values()),
            all_paid_actual_controls=sum(review.main_controls.values())+sum(review.audit_controls.values()),
            main_controls_by_scope=dict(review.main_controls), audit_controls_by_scope=dict(review.audit_controls),
            main_controls_by_scope_phase=dict(review.main_by_phase),
            audit_controls_by_scope_phase=dict(review.audit_by_phase), per_seed_arm_standalone=per_arm,
            ledger_rows=len(review.query_ids), physics_queries=review.physics_queries,
            analytic_bound_queries=review.bound_queries, audit_ledger_rows=review.audit_rows,
            development_controls_conservatively_charged_without_raw_replay=review.external_unverified_controls,
            main_prediction_rows=review.prediction_rows, main_fit_updates=review.fit_updates,
            audit_prediction_rows=review.audit_prediction_rows,
            query_kind_counts=dict(review.kind_counts), query_file_hashes=review.artifact_hashes,
            audit_wall_seconds=time.perf_counter()-started,
            audit_script_sha256=_hash(Path(__file__)), simulator_api_calls=0,
            new_actor_training_updates=0, new_critic_fits=0,
            scope="Independent recorded transition/bound algebra and provenance; no learning/evaluation rerun",
            limitations=["External development work without the common raw NPZ schema is charged again but not certified as replayed.",
                         "Passing verifies raw records and declared budgets, not generalization or causal superiority."])
        with destination.open("x") as output:
            json.dump(_plain(result), output, indent=2, allow_nan=False)
            output.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    result = audit(parser.parse_args().run_dir)
    print(json.dumps({key: result[key] for key in ("passed", "failed_check_count", "main_unique_controls",
          "audit_unique_verification_controls", "all_paid_actual_controls", "audit_wall_seconds", "errors")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
