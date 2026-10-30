"""Independently audit a completed pilot using recorded evidence.

Only saved evidence, transition algebra, hashing, and private RNG reconstruction
are used. Algebraic transition recomputations count as additional verification
controls, separately from the frozen main ledger. No Task, Search, Value, fit
routine, or experiment driver is imported.
The audit does not change frozen inputs or regenerate experimental outcomes.
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


COUNTERS = ("independent_controls", "memory_controls", "verification_controls",
            "scratch_controls", "executed_controls", "planner_calls", "failed_queries",
            "prediction_rows", "fit_updates")
EPISODE_KINDS = {"plan", "selection", "guidance", "execute", "failed_plan",
                 "failed_guidance", "failed_execute"}
ROLES = dict(bootstrap=1, acquisition=2, collection=3, branch=4, calibration=5,
             qualification=6, retention=7, evaluation=8, final=9)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pure_proposals(seed, count, horizon):
    """Independent reconstruction of the declared private-RNG proposal recipe."""
    rng = np.random.default_rng(seed)
    result = np.empty((count, horizon, 1), dtype=np.float64)
    cuts = (0, count // 2, count // 2 + count // 4, count)
    for start, stop, block in zip(cuts[:-1], cuts[1:], (4, 16, horizon)):
        drawn = rng.uniform(-1., 1., (stop-start, (horizon+block-1)//block, 1))
        result[start:stop] = np.repeat(drawn, block, axis=1)[:, :horizon]
    result[0] = 0.
    return result


def transition_equations(states, actions, goal):
    """Arithmetic check of recorded boundaries; this is not a simulator call."""
    action = np.clip(np.asarray(actions)[..., 0], -1., 1.)
    position, velocity = states[..., 0], states[..., 1]
    speed = np.clip(velocity + .0015*action - .0025*np.cos(3.*position), -.07, .07)
    position_after = np.clip(position + speed, -1.2, .6)
    speed = np.where((position_after <= -1.2) & (speed < 0.), 0., speed)
    after = np.stack((position_after, speed), axis=-1)
    terminal = position >= goal
    return np.where(terminal[..., None], states, after), np.where(terminal, 0., 1.+.02*action**2)


def features(states, remaining, total_steps):
    return np.column_stack((np.asarray(states)/[1.8, .07], np.asarray(remaining)/total_steps))


class Audit:
    def __init__(self, out):
        self.out = Path(out)
        self.cfg = read(self.out / "config.json")
        self.errors = []
        self.checks = Counter()
        self.max_errors = defaultdict(float)
        self.totals = Counter()
        self.per_arm = defaultdict(Counter)
        self.per_cycle_training = defaultdict(Counter)
        self.fit_events = {}
        self.paired_events = []
        self.unavailable_rejected_models = []
        self.proposal_comparisons = {}
        self.cross_arm_proposal_matches = 0
        self.plans_checked = 0
        self.verification_controls = 0
        self.metadata = {p.stem: read(p) for p in (self.out / "episodes").glob("*.json")}
        self.raw = {}
        self.controllers = {p.stem.removeprefix("controller_"): read(p)
                            for p in (self.out / "controllers").glob("controller_*.json")}
        self.archives = {p.stem.removeprefix("archive_"): read(p)
                         for p in (self.out / "controllers").glob("archive_*.json")}
        self.archive_row_sets = {key: set(value["row_ids"]) for key, value in self.archives.items()}
        self.gates = read(self.out / "gates.json")
        self.gate_by_key = {(g["seed"], g["arm"], g["cycle"]): g for g in self.gates}

    def check(self, condition, category, detail):
        self.checks[category] += 1
        if not bool(condition):
            if len(self.errors) < 100:
                self.errors.append(f"{category}: {detail}")

    def equal(self, left, right, category, detail, tolerance=1e-12):
        left, right = np.asarray(left), np.asarray(right)
        good = left.shape == right.shape
        if good and left.size:
            good = bool(np.isfinite(left).all() and np.isfinite(right).all())
            if good:
                error = float(np.max(np.abs(left-right)))
                self.max_errors[category] = max(self.max_errors[category], error)
                good = error <= tolerance
        self.check(good, category, detail)

    def load_raw(self, identifier):
        if identifier not in self.raw:
            with np.load(self.out / "episodes" / f"{identifier}.npz", allow_pickle=False) as z:
                self.raw[identifier] = {key: z[key] for key in ("states", "actions", "costs", "returns")}
        return self.raw[identifier]

    def source_check(self):
        manifest = read(self.out / "manifest.json")
        root = Path(__file__).resolve().parents[2]
        for relative, expected in manifest["source_hashes"].items():
            for label, path in (("archived", self.out / "source" / relative), ("runtime", root / relative)):
                self.check(path.exists() and digest(path) == expected, "source_hash", f"{label} {relative}")
        self.check(self.cfg == read(self.out / "source/configs/independent_search_pilot.json"),
                   "configuration", "saved configuration differs from frozen source")

    def episode(self, identifier, rows):
        meta = self.metadata[identifier]
        raw = self.load_raw(identifier)
        r = meta["remaining_start"]
        self.check(len(raw["actions"]) == r and len(raw["states"]) == r+1,
                   "complete_episodes", identifier)
        after, costs = transition_equations(raw["states"][:-1], raw["actions"], self.cfg["goal"])
        self.verification_controls += len(raw["actions"])
        self.equal(after, raw["states"][1:], "transition_algebra", identifier)
        self.equal(costs, raw["costs"], "stage_cost_algebra", identifier)
        self.equal(np.cumsum(raw["costs"][::-1])[::-1], raw["returns"], "measured_suffix_returns", identifier)
        self.equal(raw["returns"][0], meta["cost"], "episode_summary", identifier)
        self.check(meta["success"] == bool(np.any(raw["states"][:, 0] >= self.cfg["goal"])),
                   "episode_summary", identifier+" success")
        self.check([rows[0]["query_id"], rows[-1]["query_id"]+1] == meta["ledger_event_range"],
                   "ledger_episode_ranges", identifier)
        controller = self.controllers[meta["controller"]]
        by_step = defaultdict(dict)
        for row in rows:
            self.check(row["controller"] == meta["controller"] and row["archive"] == controller["archive"],
                       "controller_context", identifier)
            self.check(row["root"] == meta["root"] and row["role"] == meta["role"],
                       "episode_context", identifier)
            self.check(row["remaining"] == r-row["step"], "episode_context", identifier+" remaining")
            self.check(row["kind"] not in by_step[row["step"]], "unique_step_events", identifier)
            by_step[row["step"]][row["kind"]] = row
        with np.load(self.out / "episodes" / f"{identifier}.npz", allow_pickle=False) as z:
            for t in range(r):
                event = by_step[t]
                execution = event["execute"]
                self.equal(execution["state"], raw["states"][t], "execute_ledger", identifier)
                self.equal(execution["next_state"], raw["states"][t+1], "execute_ledger", identifier)
                self.equal(execution["action"], raw["actions"][t], "execute_ledger", identifier)
                self.equal(execution["cost"], raw["costs"][t], "execute_ledger", identifier)
                if meta["forced"] and t == 0:
                    self.check(set(event) == {"execute"} and z["selected_indices"][t] == -1,
                               "forced_action_semantics", identifier)
                    continue
                plan, selection = event["plan"], event["selection"]
                horizon = min(meta["horizon"], r-t)
                proposal_seed = int(np.random.SeedSequence([meta["stream"], t]).generate_state(1)[0])
                self.check(plan["proposal_seed"] == proposal_seed == z["proposal_seeds"][t],
                           "private_stream_seeds", identifier)
                proposals = pure_proposals(proposal_seed, self.cfg["K"], horizon)
                expected_digest = hashlib.sha256(proposals.tobytes()).hexdigest()
                self.check(plan["independent_digest"] == expected_digest, "independent_proposal_digest", identifier)
                self.plans_checked += 1
                count = self.cfg["K"]+1+len(plan["memory_row_ids"])
                bank_actions = z["bank_first_actions"][t, :count]
                self.equal(proposals[:, 0], bank_actions[:self.cfg["K"]], "raw_bank_first_actions", identifier)
                scores = z["bank_heuristic_scores"][t, :self.cfg["K"]]
                weights = np.exp(-(scores-scores.min())/self.cfg["temperature"])
                weights /= weights.sum()
                mean = np.sum(proposals*weights[:, None, None], axis=0)
                self.equal(mean, plan["mean_plan"], "independent_weighted_mean", identifier)
                self.equal(mean[0], bank_actions[self.cfg["K"]], "mean_first_action", identifier)
                for j, row_id in enumerate(plan["memory_row_ids"]):
                    episode_id, index = row_id.rsplit(":", 1)
                    self.check(row_id in self.archive_row_sets[controller["archive"]],
                               "retrieved_archive_membership", row_id)
                    self.equal(self.load_raw(episode_id)["actions"][int(index)],
                               bank_actions[self.cfg["K"]+1+j], "memory_first_action", row_id)
                baseline, selected = int(z["baseline_indices"][t]), int(z["selected_indices"][t])
                self.check(plan["baseline_index"] == baseline == int(np.argmin(z["bank_local_scores"][t, :count])),
                           "baseline_selection", identifier)
                self.check(selection["selected_index"] == selected, "selected_index", identifier)
                self.equal(selection["action"], bank_actions[selected], "selected_bank_execution", identifier)
                self.equal(raw["actions"][t], bank_actions[selected], "selected_bank_execution", identifier)
                changed = not np.array_equal(bank_actions[selected], bank_actions[baseline])
                self.check(selection["changed"] == changed, "changed_action_labels", identifier)
                self.check(plan["independent_controls"] == self.cfg["K"]*horizon
                           and plan["verification_controls"] == horizon
                           and plan["memory_controls"] == len(plan["memory_row_ids"])*horizon
                           and plan["planner_calls"] == 1,
                           "query_category_accounting", identifier)
                if controller["value"] is None:
                    self.check(selected == baseline and "guidance" not in event,
                               "absent_guidance", identifier)
                else:
                    guidance = event["guidance"]
                    supported = np.asarray(guidance["supported"], dtype=bool)
                    q = np.asarray(guidance["predicted_scores"], dtype=float)
                    ranked = np.where(supported, q, np.inf)
                    expected = int(np.argmin(ranked)) if supported[baseline] else baseline
                    if ranked[baseline] <= ranked[expected]+1e-12 or not np.isfinite(q).all():
                        expected = baseline
                    self.check(expected == selected == guidance["selected"], "guidance_selection", identifier)
                    scratch = len(np.unique(bank_actions, axis=0)) if controller["kind"] == "mc" else 0
                    self.check(guidance["scratch_controls"] == scratch,
                               "scratch_accounting", identifier)
                # Matching stream/step/horizon must reproduce banks even after state distributions diverge.
                key = (meta["seed"], meta["stream"], t, horizon)
                prior = self.proposal_comparisons.get(key)
                if prior and prior[0] != meta["arm"]:
                    self.check(prior[1] == expected_digest, "cross_arm_proposal_matching", identifier)
                    self.cross_arm_proposal_matches += 1
                else:
                    self.proposal_comparisons[key] = (meta["arm"], expected_digest)

    def ledger(self):
        current_id, rows = None, []
        row_count = 0
        with (self.out / "queries.jsonl").open() as handle:
            for row_count, line in enumerate(handle, start=1):
                row = json.loads(line)
                self.check(row["query_id"] == row_count-1, "append_only_query_ids", str(row_count))
                for key in COUNTERS:
                    value = row.get(key, 0)
                    self.check(type(value) is int and value >= 0, "nonnegative_integer_counters", key)
                    self.totals[key] += value
                    self.per_arm[row["arm"]][key] += value
                    if row["phase"] == "training":
                        self.per_cycle_training[(row["seed"], row["arm"], row["cycle"])][key] += value
                if row["kind"] == "fit":
                    self.fit_events[(row["seed"], row["arm"], row["cycle"])] = row
                if row["kind"] == "paired_result":
                    self.paired_events.append(row)
                if row["kind"] not in EPISODE_KINDS:
                    continue
                identifier = row["episode"]
                if current_id is not None and identifier != current_id:
                    self.episode(current_id, rows)
                    rows = []
                current_id = identifier
                rows.append(row)
        if rows:
            self.episode(current_id, rows)
        self.query_rows = row_count
        for counters in [self.totals, *self.per_arm.values(), *self.per_cycle_training.values()]:
            counters["simulated_controls"] = sum(counters[key] for key in COUNTERS[:4])
            counters["total_controls"] = counters["simulated_controls"]+counters["executed_controls"]
        work = read(self.out / "work.json")
        for key, value in self.totals.items():
            self.check(work["actual"][key] == value, "aggregate_work_ledger", key)
        for key, counters in self.per_cycle_training.items():
            if key[1] != "shared":
                self.check(counters["total_controls"] <= self.cfg["cycle_training_controls"],
                           "frozen_cycle_budget", str(key))

    def archive_and_controller_checks(self):
        for expected, archive in self.archives.items():
            roots = defaultdict(list)
            episodes = [self.metadata[i] for i in archive["episode_ids"]]
            for ep_index, meta in enumerate(episodes):
                self.check(meta["phase"] == "training" and meta["role"] in
                           {"bootstrap", "acquisition", "collection", "branch"},
                           "archive_role_exclusion", meta["id"])
                gate = self.gate_by_key.get((meta["seed"], meta["arm"], meta["cycle"]))
                if gate and meta["role"] in {"collection", "branch"}:
                    self.check(meta["root"] in gate["train_roots"] and meta["root"] not in gate["heldout_roots"],
                               "archive_heldout_exclusion", meta["id"])
                roots[str(meta["root"])].extend((ep_index, t) for t in range(meta["remaining_start"]))
            quotas = dict.fromkeys(sorted(roots), 0)
            budget = min(self.cfg["memory_rows"], sum(map(len, roots.values())))
            while budget:
                for root in quotas:
                    if quotas[root] < len(roots[root]) and budget:
                        quotas[root] += 1
                        budget -= 1
            chosen = []
            for root, quota in quotas.items():
                if quota:
                    chosen += [roots[root][j] for j in np.linspace(0, len(roots[root])-1, quota, dtype=np.int64)]
            row_ids = [f"{episodes[e]['id']}:{t}" for e, t in chosen]
            self.check(archive["row_ids"] == row_ids, "archive_exact_rows", expected)
            hasher = hashlib.sha256(json.dumps({"goal": self.cfg["goal"], "rows": row_ids}, sort_keys=True).encode())
            for meta in episodes:
                hasher.update(json.dumps({"root": str(meta["root"]), "id": meta["id"],
                                          "remaining_start": meta["remaining_start"]}, sort_keys=True).encode())
                for key in ("states", "actions", "returns"):
                    hasher.update(np.asarray(self.load_raw(meta["id"])[key], dtype=np.float64).tobytes())
            self.check(hasher.hexdigest() == expected == archive["digest"], "archive_digest", expected)
        for expected, controller in self.controllers.items():
            body = controller["archive"]+controller["kind"]+str(controller["horizon"])+(controller["value"] or "off")
            self.check(hashlib.sha256(body.encode()).hexdigest() == expected == controller["version"],
                       "controller_version", expected)
            self.check(controller["archive"] in self.archives, "controller_archive", expected)
            if controller["value"] is not None:
                with np.load(self.out / "controllers" / f"value_{controller['value']}.npz", allow_pickle=False) as z:
                    target, kind = str(z["target_version"]), str(z["kind"])
                    hasher = hashlib.sha256((target+kind).encode())
                    for key in ("centers", "weights", "train_features"):
                        hasher.update(z[key].tobytes())
                    self.check(hasher.hexdigest() == controller["value"], "value_digest", expected)
                    self.check(target == controller["target_version"] and target in self.controllers,
                               "declared_value_continuation", expected)

    def roots_and_targets(self):
        ordinary_roots = defaultdict(set)
        for meta in self.metadata.values():
            if not meta["forced"]:
                ordinary_roots[meta["role"]].add(meta["root"])
                index = int(re.match(r"s\d+_\w+_c-?\d+_\w+_(\d+)_\d+$", meta["id"]).group(1))
                reset_cycle = 9 if meta["role"] == "final" else meta["cycle"]
                reset_seed = 60_000_000+meta["seed"]*1_000_000+(reset_cycle+1)*100_000+ROLES[meta["role"]]*10_000+index
                self.check(meta["root"] == reset_seed and meta["stream"] == reset_seed+100_000_000,
                           "role_seed_definition", meta["id"])
                self.equal(self.load_raw(meta["id"])["states"][0],
                           [np.random.default_rng(reset_seed).uniform(-.6, -.4), 0.],
                           "reset_distribution", meta["id"])
        roles = sorted(ordinary_roots)
        for i, left in enumerate(roles):
            for right in roles[i+1:]:
                self.check(not ordinary_roots[left] & ordinary_roots[right], "disjoint_role_roots", left+" / "+right)
        for key, event in self.fit_events.items():
            seed, arm, cycle = key
            gate = self.gate_by_key[key]
            self.check(not set(gate["train_roots"]) & set(gate["heldout_roots"]), "whole_root_split", str(key))
            with np.load(self.out / "controllers" / f"targets_s{seed}_{arm}_c{cycle}.npz", allow_pickle=False) as z:
                ids, groups = z["ids"], z["groups"]
                self.check(set(map(int, groups)) <= set(gate["train_roots"]), "fitting_root_ancestry", str(key))
                for j, raw_id in enumerate(ids):
                    identifier, index = str(raw_id).rsplit(":", 1)
                    index = int(index)
                    meta = self.metadata[identifier]
                    data = self.load_raw(identifier)
                    self.equal(z["states"][j], data["states"][index], "target_state_provenance", str(raw_id))
                    self.check(z["remaining"][j] == meta["remaining_start"]-index
                               and groups[j] == meta["root"], "target_context_provenance", str(raw_id))
                    self.check(meta["controller"] == event["target_version"] == gate["target_controller"],
                               "frozen_target_controller", str(raw_id))
                    self.check(not meta["forced"] or index > 0, "forced_predecision_exclusion", str(raw_id))
                    if arm == "mc":
                        self.equal(z["targets"][j], data["returns"][index], "target_complete_return", str(raw_id))
                    else:
                        self.check(index % self.cfg["bank_stride"] == 0, "optimized_backup_source", str(raw_id))
                model = self.out / "controllers" / f"value_{event['value_digest']}.npz"
                # Rejected models may have no deployed controller file, but fit records preserve their digest.
                if model.exists():
                    with np.load(model, allow_pickle=False) as v:
                        self.equal(v["train_features"], features(z["states"], z["remaining"], self.cfg["steps"]),
                                   "training_only_support", str(key))
                else:
                    self.unavailable_rejected_models.append(dict(seed=seed, arm=arm, cycle=cycle,
                                                                 digest=event["value_digest"]))
            with np.load(self.out / "controllers" / f"heldout_s{seed}_{arm}_c{cycle}.npz", allow_pickle=False) as h:
                self.check(set(map(int, h["groups"])) == set(gate["heldout_roots"]),
                           "heldout_root_ancestry", str(key))
        for row in self.paired_events:
            p = row["result"]
            point = self.metadata[p["point_episode"]]
            self.check(point["root"] == p["root"] and p["target_controller"] in self.controllers,
                       "paired_continuation_identity", p["point_episode"])
        self.root_counts = {role: len(roots) for role, roots in ordinary_roots.items()}


def audit(out, destination=None):
    start = time.perf_counter()
    out = Path(out).resolve()
    if not read(out / "work.json").get("completed") or not read(out / "status.json").get("completed"):
        raise RuntimeError("Audit requires a completed frozen trial; no partial outcome is certified")
    destination = Path(destination) if destination else out / "independent_audit.json"
    if destination.exists():
        raise FileExistsError(f"Retaining existing audit: {destination}")
    review = Audit(out)
    review.source_check()
    review.ledger()
    review.archive_and_controller_checks()
    review.roots_and_targets()
    result = dict(schema="independent-evidence-audit-v1", passed=not review.errors,
                  errors=review.errors, checks=dict(review.checks), maximum_numeric_errors=dict(review.max_errors),
                  episodes=len(review.metadata), query_rows=review.query_rows,
                  reconstructed_independent_banks=review.plans_checked,
                  cross_arm_bank_matches=review.cross_arm_proposal_matches,
                  rejected_models_without_persisted_weights=review.unavailable_rejected_models,
                  root_counts=review.root_counts, raw_category_totals=dict(review.totals),
                  audit_wall_seconds=time.perf_counter()-start, simulator_api_calls=0,
                  audit_work=dict(verification_controls=review.verification_controls,
                                  simulated_controls=review.verification_controls,
                                  executed_controls=0, total_controls=review.verification_controls,
                                  model_verification_queries=len(review.metadata), failed_queries=0,
                                  training_updates=0),
                  limitations=["Audits the recorded deterministic equations and accounting; does not establish simulator realism.",
                               "Optimized-backup intermediate fits are not retrained; their saved source and target provenance is checked.",
                               "Prediction validity and causal superiority remain experimental results, not audit conclusions."],
                  verdict="Raw evidence checks pass" if not review.errors else "Raw evidence checks failed")
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    result = audit(args.out_dir, args.destination)
    print(json.dumps({key: result[key] for key in ("passed", "episodes", "query_rows",
          "reconstructed_independent_banks", "audit_wall_seconds", "audit_work", "errors")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
