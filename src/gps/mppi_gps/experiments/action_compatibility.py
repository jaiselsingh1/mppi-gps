"""Frozen, no-learning Walker action-compatibility diagnostic (K1)."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..compatibility_inputs import extract_inputs
from ..compatibility_metrics import (
    aggregate_trajectory_scores,
    agreement_metrics,
    neighborhood_summary,
    prepare_neighborhoods,
)
from ..lifecycle import ROOT, freeze, verify
from ..metrics import summary
from ..policy import Policy
from ..provenance import default, fingerprint, sha, write
from ..recipes import load_recipe
from ..runtime import BudgetExhausted, Meter


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stream:
        return {key: np.array(stream[key], copy=True) for key in stream.files}


def save_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def ancestry_id(row: dict[str, Any]) -> str:
    value = row["ancestry"]
    return value["reset_cluster"] if isinstance(value, dict) else str(value)


class DiagnosticMeter(Meter):
    """Enforce no learning/model calls in addition to existing work reservations."""

    def __init__(
        self,
        out: Path,
        *,
        controls: int,
        forward_rows: int,
        wall_seconds: int,
        storage_bytes: int = 7516192768,
    ):
        self.deadline = time.perf_counter() + wall_seconds
        self.storage_limit = storage_bytes
        self.final_work = None
        super().__init__(out, max_controls=controls, max_neural_rows=forward_rows)

    def close(self):
        if not self.closed:
            super().close()
            self.final_work = super().totals()

    def totals(self):
        return dict(self.final_work) if self.final_work is not None else super().totals()

    def check_deadline(self) -> None:
        if time.perf_counter() > self.deadline:
            raise BudgetExhausted("Frozen stage wall reservation exhausted")
        if self._raw.id.get_filesize() >= self.storage_limit:
            raise BudgetExhausted("Frozen stage storage reservation exhausted")

    def save_arrays(self, tag, **arrays):
        # Reserve conservative HDF metadata plus all uncompressed payload bytes
        # before persisting. Existing Meter still owns durable array contracts.
        reserve = 262144 * (1 + len(arrays))
        for value in arrays.values():
            reserve += (
                len(json.dumps(value, default=default).encode())
                if isinstance(value, dict)
                else np.asarray(value).nbytes
            )
        if self._raw.id.get_filesize() + reserve > self.storage_limit:
            raise BudgetExhausted("Array reservation would exceed frozen stage storage")
        return super().save_arrays(tag, **arrays)

    def measure(self, kind, amount, fn, **metadata):
        self.check_deadline()
        if kind not in {"neural_rows", "checkpoint_loads"}:
            raise ValueError("Diagnostic forbids training and unregistered work kinds")
        return super().measure(kind, amount, fn, **metadata)

    def _query(self, mode, *args, **kwargs):
        self.check_deadline()
        if mode != "live":
            raise ValueError("Diagnostic forbids model-rollout controls")
        return super()._query(mode, *args, **kwargs)


def seed_audit(repo: Path, seeds: list[int]) -> dict[str, Any]:
    selected = set(seeds)
    collisions, files = [], []

    def visit(value, address, owner):
        if type(value) is int and value in selected:
            collisions.append(dict(file=owner, address=address, seed=value))
        elif type(value) is int and any(word in address.lower() for word in ("seed", "root")):
            # Conservatively cover root+offset generators, rather than checking
            # only roots literally present in historical configurations.
            for seed in selected:
                if value <= seed < value + 1_000_000:
                    collisions.append(
                        dict(
                            file=owner,
                            address=address,
                            seed=seed,
                            conservative_root=value,
                            reserved_offsets=1_000_000,
                        )
                    )
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, address + "/" + str(key), owner)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, address + f"/{index}", owner)

    for path in sorted((repo / "runs").glob("*/config.json")):
        value = json.loads(path.read_text())
        visit(value, "", str(path))
        files.append(dict(file=str(path), sha256=sha(path)))
    if len(seeds) != len(selected) or collisions:
        raise ValueError("Proposed reset seeds collide with historical configuration values")
    return dict(
        passed=True,
        seeds=seeds,
        historical_configurations=files,
        collisions=collisions,
        scope="Local historical config census plus conservative million-offset seed/root ranges; all observed reset seeds in the four canonical closed ledgers additionally checked. No claim about unavailable external runs.",
    )


def reservation(config: dict[str, Any], teacher_rows: int, policies: int) -> dict[str, int]:
    # Auxiliary/retained datasets are conservatively charged even when memoized.
    a = (teacher_rows + 6096 + 6096 + 96) * policies
    primary = len(config["primary_seeds"]) * len(config["primary_policy_order"])
    confirmation = len(config["confirmation_seeds"])
    random = 4
    episodes = primary + confirmation + random
    controls = episodes * config["episode_steps"]
    b = (primary + confirmation) * config["episode_steps"]
    b += (16 + random) * config["episode_steps"] * policies
    return dict(
        stage_a_forward_rows=a,
        stage_b_forward_rows=b,
        combined_forward_rows=a + b,
        live_controls=controls,
        physics_substeps=4 * controls,
        episodes=episodes,
        model_controls=0,
        backward_rows=0,
        optimizer_steps=0,
    )


def prepare(out: Path, artifact_root: Path) -> dict[str, Any]:
    """Audit/copy inputs and seal all choices without evaluating an actual policy."""
    from ..diagnostic_policies import LegacySACActor

    cfg = load_recipe("action_compatibility")
    reference_audit = json.loads((artifact_root / "reference_audit.json").read_text())
    if (
        reference_audit.get("passed") is not True
        or reference_audit.get("status") != "interface_eligible_not_newly_qualified"
        or reference_audit.get("selection", {}).get("checkpoint_sha256")
        != cfg["reference_checkpoint_sha256"]
    ):
        raise ValueError("Reference metadata audit does not admit the preselected candidate")
    staged = artifact_root / "prepared_inputs"
    inventory = extract_inputs(ROOT, staged)
    policy_rows = {}
    (staged / "checkpoints").mkdir()
    for name, entry in inventory["policies"].items():
        source = Path(entry["checkpoint"])
        target = staged / "checkpoints" / f"{name}.pt"
        shutil.copy2(source, target)
        if sha(target) != entry["file_sha256"]:
            raise ValueError("Copied policy checkpoint differs")
        policy_rows[name] = dict(entry, checkpoint=str(target.relative_to(staged)))
    parent, _, _ = Policy.from_checkpoint(staged / policy_rows["h4_parent"]["checkpoint"])
    for index, seed in enumerate(cfg["untrained_seeds"]):
        actor = Policy(seed=seed, normalization=parent.normalization)
        target = staged / "checkpoints" / f"untrained_{index}.pt"
        binding = actor.checkpoint(
            target, metadata=dict(role="untrained", updates=0, normalization_owner=parent.hash())
        )
        policy_rows[f"untrained_{index}"] = dict(
            checkpoint=str(target.relative_to(staged)),
            file_sha256=binding["sha256"],
            policy_hash=actor.hash(),
            feature="clipped",
            normalization=actor.normalization,
            architecture="17-128tanh-128tanh-6tanh",
            seed=seed,
            training="none; incoming H4 normalization copied without refit",
        )
    reference = ROOT / "runs/2026-10-11_s2_sac_ladder_seed0_001/checkpoints/step_1000000.pt"
    if sha(reference) != cfg["reference_checkpoint_sha256"]:
        raise ValueError("Preselected final SAC checkpoint changed; no fallback permitted")
    target = staged / "checkpoints/reference.pt"
    shutil.copy2(reference, target)
    actor = LegacySACActor.from_checkpoint(target)  # Decode only; no forward call.
    if actor.hash() != reference_audit["selection"]["actor_hash"]:
        raise ValueError("Decoded reference differs from eligibility audit")
    policy_rows["reference"] = dict(
        checkpoint="checkpoints/reference.pt",
        file_sha256=sha(target),
        policy_hash=actor.hash(),
        feature="clipped",
        normalization=actor.normalization,
        architecture="17-256ReLU-256ReLU-SACheads; deterministic tanh(mean)",
        training="independent SAC final 1M; gamma .995 native negative cost plus absorbing reward tail",
        eligible=True,
        qualified=False,
        audit=reference_audit,
    )
    policy_order = (
        [f"untrained_{i}" for i in range(3)] + list(inventory["policies"]) + ["reference"]
    )
    teachers = inventory["trajectories"]
    arrays = [load_npz(staged / row["file"]) for row in teachers]
    flat = {
        key: np.concatenate([row[key] for row in arrays])
        for key in ("observations", "actions", "states")
    }
    ids = np.concatenate([np.repeat(row["id"], row["steps"]) for row in teachers])
    ancestry = np.concatenate([np.repeat(ancestry_id(row), row["steps"]) for row in teachers])
    steps = np.concatenate([np.arange(row["steps"]) for row in teachers])
    geometry = prepare_neighborhoods(
        flat["observations"],
        ids,
        steps,
        ancestry,
        **parent.normalization,
        config=cfg["neighborhoods"],
    )
    save_npz(staged / "geometry.npz", geometry)
    random = np.random.Generator(np.random.PCG64(cfg["synthetic_random_seed"]))
    permutation = np.random.Generator(np.random.PCG64(cfg["permutation_seed"]))
    save_npz(
        staged / "negative_controls.npz",
        dict(
            random_labels=random.uniform(-1, 1, flat["actions"].shape),
            permutation_indices=np.concatenate(
                [
                    permutation.permutation(row["steps"])
                    + sum(previous["steps"] for previous in teachers[:index])
                    for index, row in enumerate(teachers)
                ]
            ),
        ),
    )
    seeds = cfg["primary_seeds"] + cfg["confirmation_seeds"]
    old_seeds = {row["seed"] for row in teachers + inventory["historical_standalone"]}
    for observed in inventory.get("source_reset_seeds", {}).values():
        old_seeds.update(observed)
    if old_seeds.intersection(seeds):
        raise ValueError("Reset contexts overlap preserved trajectory roster")
    write(staged / "seed_audit.json", seed_audit(ROOT, seeds))
    shutil.copy2(artifact_root / "reference_audit.json", staged / "reference_audit.json")
    shutil.copy2(artifact_root / "data_audit.json", staged / "data_audit.json")
    h4_cfg = json.loads(
        (ROOT / "runs/2026-10-30_adaptive_coupling_h4_1_r1_001/config.json").read_text()
    )
    cfg.update(
        task=h4_cfg["task"],
        mppi=h4_cfg["mppi"],
        policy=dict(hidden=128),
        gates=dict(velocity_mae=0.3),
        policy_order=policy_order,
        policies=policy_rows,
        trajectory_roster=teachers,
        historical_standalone=inventory["historical_standalone"],
        datasets=inventory["datasets"],
        inventory_sha256=sha(staged / "inventory.json"),
        missing_evidence=inventory["missing_evidence"],
        platform=dict(python=sys.version, platform=platform.platform(), machine=platform.machine()),
        basis_commit="c58f0c434e0f5bacb06099a33c3dd06922bdb2c0",
        implementation_provenance="Exact current source bytes sealed in manifest; new diagnostic implementation is additional to the basis commit.",
        scoring_precision="float64 residuals against original executed actions; network inputs float32. Source fitting MSE cast targets float32 and may differ in final bits.",
        inference="Current Policy.predict batches of1024; legacy actor retains original batch-one execution; exact-float32 input memoization pinned to each actor identity.",
    )
    planned = reservation(cfg, len(flat["actions"]), len(policy_order))
    if (
        planned["stage_a_forward_rows"] > cfg["budgets"]["stage_a_forward_rows"]
        or planned["stage_b_forward_rows"] > cfg["budgets"]["stage_b_forward_rows"]
        or planned["live_controls"] > cfg["budgets"]["reserved_live_controls"]
        or planned["episodes"] != cfg["budgets"]["maximum_episodes"]
    ):
        raise ValueError("Complete roster exceeds frozen reservation")
    cfg["planned_work"] = planned
    frozen = freeze(out, cfg, "action-compatibility-k1")
    shutil.move(str(staged), frozen / "inputs")
    write(
        frozen / "input_manifest.json",
        dict(
            files={
                str(path.relative_to(frozen)): sha(path)
                for path in sorted((frozen / "inputs").rglob("*"))
                if path.is_file()
            },
            config_sha256=sha(frozen / "config.json"),
            source_manifest_sha256=sha(frozen / "manifest.json"),
            planned_work=planned,
            new_predictions_before_freeze=0,
            source_evidence_ownership="Historical originals remain unchanged; curated copies retain exact source descriptors and existing audit pins.",
        ),
    )
    receipt = dict(
        run=str(frozen),
        config_sha256=sha(frozen / "config.json"),
        input_manifest_sha256=sha(frozen / "input_manifest.json"),
        planned_work=planned,
        geometry_sha256=sha(frozen / "inputs/geometry.npz"),
        predictions=0,
        physics=0,
    )
    write(artifact_root / "preparation_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


class PredictionStore:
    """Reuse only exact float32 inputs tied to the same frozen actor identity."""

    def __init__(self, actors: dict[str, Any], batch_size: int, conflict_atol: float = 0.000002):
        self.actors, self.batch_size = actors, batch_size
        self.conflict_atol = conflict_atol
        self.cache = {name: {} for name in actors}
        self.reuse_sources = []
        self.maximum_cache_rounding_difference = 0.0

    def seed(self, name, observations, predictions, source):
        x, y = np.asarray(observations, dtype=np.float32), np.asarray(predictions)
        if x.shape != (len(y), 17) or y.shape != (len(x), 6) or not np.isfinite(y).all():
            raise ValueError("Saved predictions have incompatible alignment")
        for row, value in zip(x, y, strict=True):
            key = row.tobytes()
            if key in self.cache[name]:
                difference = float(np.max(np.abs(self.cache[name][key] - value)))
                self.maximum_cache_rounding_difference = max(
                    self.maximum_cache_rounding_difference, difference
                )
                if difference > self.conflict_atol:
                    raise ValueError("Saved predictions conflict at identical frozen policy input")
            else:
                self.cache[name][key] = value.copy()
        self.reuse_sources.append(dict(policy=name, rows=len(x), source=source))

    def predict(self, name, observations, meter):
        x = np.asarray(observations, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 17 or not np.isfinite(x).all():
            raise ValueError("Finite aligned clipped17 input rows required")
        keys = [row.tobytes() for row in x]
        absent = list(dict.fromkeys(key for key in keys if key not in self.cache[name]))
        for start in range(0, len(absent), self.batch_size):
            part = absent[start : start + self.batch_size]
            rows = np.stack([np.frombuffer(key, dtype=np.float32) for key in part])
            predictions = self.actors[name].predict(rows, meter=meter)
            if (
                predictions.shape != (len(rows), 6)
                or not np.isfinite(predictions).all()
                or np.any(np.abs(predictions) > 1)
            ):
                raise ValueError("Policy violated native bounded torque interface")
            for key, value in zip(part, predictions, strict=True):
                self.cache[name][key] = value.copy()
        return np.stack([self.cache[name][key] for key in keys]), len(absent)


def score_trajectory(row, data, names, store, meter, cfg, *, negative=None):
    scores, predictions = {}, []
    for name in names:
        prediction, new_rows = store.predict(name, data["observations"], meter)
        predictions.append(prediction)
        value = agreement_metrics(prediction, data["actions"], **cfg["metrics"])
        value.update(new_forward_rows=new_rows, reused_forward_rows=len(prediction) - new_rows)
        if negative is not None:
            value["synthetic_random_labels"] = agreement_metrics(
                prediction, negative["random"], **cfg["metrics"]
            )
            value["permuted_recorded_labels"] = agreement_metrics(
                prediction, negative["permuted"], **cfg["metrics"]
            )
        scores[name] = value
    raw = meter.save_arrays(
        "prediction_matrix_" + row["id"],
        predictions=np.stack(predictions, axis=1),
        executed_actions=data["actions"],
        observations=data["observations"],
    )
    return dict(trajectory=row, policies=scores, raw=raw)


def group_scores(rows, names, cfg):
    groups = {}
    for group in dict.fromkeys(row["trajectory"]["group"] for row in rows):
        selected = [row for row in rows if row["trajectory"]["group"] == group]
        values = np.array([[row["policies"][name]["mse"] for name in names] for row in selected])
        rng = np.random.Generator(np.random.PCG64(cfg["bootstrap_seed"]))
        aggregate = aggregate_trajectory_scores(
            values,
            np.array([row["trajectory"]["steps"] for row in selected]),
            np.array([ancestry_id(row["trajectory"]) for row in selected]),
            rng=rng,
            bootstrap_draws=cfg["bootstrap_draws"],
        )
        replicates = aggregate.pop("replicates_equal_trajectory_mean")
        aggregate.pop("replicates_sample_weighted_mean")
        contrasts = {}
        for name in names[3:]:
            index = names.index(name)
            delta = replicates[:, index] - np.mean(replicates[:, :3], axis=1)
            contrasts[name + "_minus_mean_untrained"] = dict(
                delta=float(np.mean(values[:, index] - np.mean(values[:, :3], axis=1))),
                interval=np.percentile(delta, [2.5, 97.5]).tolist(),
            )
        groups[group] = dict(policies=names, **aggregate, contrasts=contrasts)
    return groups


def historical_failure_trace(data):
    """Describe existing executions; own actions are not expert labels off-path."""
    x = data["observations"]
    return dict(
        rows=len(x),
        actions_mean_abs=float(np.mean(np.abs(data["actions"]))),
        saturation_fraction=float(np.mean(np.abs(data["actions"]) >= 0.95)),
        clipped_velocity_rows=int(np.sum(np.any(np.abs(data["states"][:, 10:19]) > 10, axis=1))),
        scope="Saved standalone action/state trace and recorded outcome; no expert labels or new continuation queried.",
    )


def stage_a(out, cfg, actors, store, meter):
    inputs = out / "inputs"
    rows, data_rows = [], []
    negative = load_npz(inputs / "negative_controls.npz")
    offset = 0
    for entry in cfg["trajectory_roster"]:
        data = load_npz(inputs / entry["file"])
        data_rows.append(data)
        for name, binding in entry.get("saved_predictions", {}).items():
            if binding["policy_hash"] != actors[name].hash():
                raise ValueError("Saved prediction owner differs from roster")
            store.seed(name, data["observations"], data[binding["key"]], binding)
        count = len(data["actions"])
        random = negative["random_labels"][offset : offset + count]
        indices = negative["permutation_indices"][offset : offset + count] - offset
        rows.append(
            score_trajectory(
                entry,
                data,
                cfg["policy_order"],
                store,
                meter,
                cfg,
                negative=dict(random=random, permuted=data["actions"][indices]),
            )
        )
        offset += count
    dataset_scores = {}
    for name, relative in cfg["datasets"].items():
        if not isinstance(relative, str):
            continue
        data = load_npz(inputs / relative)
        if not {"observations", "actions"}.issubset(data):
            continue
        entry = dict(
            id="retained_" + name,
            group=name,
            steps=len(data["actions"]),
            ancestry="mixed_retained_history",
        )
        dataset_scores[name] = score_trajectory(entry, data, cfg["policy_order"], store, meter, cfg)
    flat = {
        key: np.concatenate([row[key] for row in data_rows])
        for key in (
            "actions",
            "states",
            "planner_nominal",
            "planner_rng_fingerprint",
            "bank_digest",
            "planner_context_available",
        )
    }
    geometry = load_npz(inputs / "geometry.npz")
    context = {
        key: flat[key]
        for key in (
            "planner_nominal",
            "planner_rng_fingerprint",
            "bank_digest",
            "planner_context_available",
        )
    }
    context["rho"] = np.concatenate(
        [np.repeat(row["controller"]["rho"], row["steps"]) for row in cfg["trajectory_roster"]]
    )
    neighborhood = neighborhood_summary(geometry, flat["actions"], flat["states"], context)
    boundaries = np.cumsum([0] + [row["steps"] for row in cfg["trajectory_roster"]])
    for collision in neighborhood["exact_incoming_h4_effective_input_collisions"]["details"]:
        selected = collision["row_indices"]
        collision["guidance_weights"] = sorted(set(context["rho"][selected].tolist()))
        snapshots = {}
        for field in ("warmstarts", "integrations"):
            values = []
            for index in selected:
                source_index = int(np.searchsorted(boundaries[1:], index, side="right"))
                data = data_rows[source_index]
                if field in data:
                    values.append(data[field][index - boundaries[source_index]])
            snapshots[field] = dict(
                available_rows=len(values),
                total_rows=len(selected),
                all_identical=all(np.array_equal(values[0], value) for value in values[1:])
                if len(values) == len(selected)
                else None,
            )
        collision["saved_physical_context"] = snapshots
    historical = []
    for row in cfg["historical_standalone"]:
        data = load_npz(inputs / row["file"])
        historical.append(dict(trajectory=row, diagnostics=historical_failure_trace(data)))
    return dict(
        trajectories=rows,
        groups=group_scores(rows, cfg["policy_order"], cfg),
        retained_datasets=dataset_scores,
        neighborhoods=neighborhood,
        historical_standalone=historical,
        saved_prediction_reuse=list(store.reuse_sources),
        maximum_cache_rounding_difference=store.maximum_cache_rounding_difference,
        teaching_utility_tested=False,
        new_physics=0,
    )


def evaluate_one(study, actor, seed, tag, meter):
    from ..runtime import execute_episode

    owner = actor.hash()

    def control(state, warm, obs, step):
        return actor.predict(obs, meter=meter), dict(
            policy_hash=owner, standalone=True, feature_mode="clipped"
        )

    meter.context = dict(role="standalone", policy_hash=owner, cohort=tag)
    episode = execute_episode(study.env, control, seed, 1000, meter, tag)
    if actor.hash() != owner:
        raise ValueError("Standalone execution mutated frozen actor")
    return episode


def episode_record(episode, policy, cohort):
    return dict(
        policy=policy,
        cohort=cohort,
        seed=episode.seed,
        tag=episode.tag,
        steps=episode.steps,
        cost=episode.cost,
        success=episode.success,
        survived=episode.survived,
        mae=episode.mae,
        complete=episode.complete,
        terminated=not episode.survived,
        raw=episode.raw,
        terminal_cost=episode.terminal_cost,
        terminal_adjustment=episode.terminal_adjustment,
        absorbing_cost=episode.absorbing_cost,
    )


def stage_b(out, cfg, actors, store, meter):
    import torch

    from ..execution import PhysicalStudy
    from ..runtime import execute_episode

    study = PhysicalStudy(out, cfg, meter)
    torch.set_num_threads(cfg["torch_threads"])
    primary = {name: [] for name in cfg["primary_policy_order"]}
    records, reference_traces = [], []
    try:
        for index, seed in enumerate(cfg["primary_seeds"]):
            initial = None
            for name in cfg["primary_policy_order"]:
                episode = evaluate_one(
                    study, actors[name], seed, f"k1_primary_{name}_{index:02d}", meter
                )
                if initial is None:
                    initial = (episode.states[0], episode.integrations[0], episode.warmstarts[0])
                elif not all(
                    np.array_equal(x, y)
                    for x, y in zip(
                        initial,
                        (episode.states[0], episode.integrations[0], episode.warmstarts[0]),
                        strict=True,
                    )
                ):
                    raise ValueError("Primary native reset pairing failed")
                primary[name].append(episode)
                records.append(episode_record(episode, name, "primary"))
                store.seed(name, episode.observations, episode.actions, dict(own_log=episode.raw))
                if name == "reference":
                    reference_traces.append(episode)
        scores = {name: summary(episodes) for name, episodes in primary.items()}
        first = (
            scores["reference"]["successes"] >= cfg["reference_qualification"]["walking_successes"]
        )
        confirmation = []
        if first:
            for index, seed in enumerate(cfg["confirmation_seeds"]):
                episode = evaluate_one(
                    study,
                    actors["reference"],
                    seed,
                    f"k1_confirmation_reference_{index:02d}",
                    meter,
                )
                confirmation.append(episode)
                records.append(episode_record(episode, "reference", "confirmation"))
        qualified = (
            first
            and sum(e.success for e in confirmation)
            >= cfg["reference_qualification"]["walking_successes"]
        )
        scored = []
        if qualified:
            for episode in reference_traces:
                row = dict(
                    id=episode.tag,
                    group="qualified_reference",
                    steps=episode.steps,
                    ancestry=f"native-reset-{episode.seed}",
                    seed=episode.seed,
                )
                scored.append(
                    score_trajectory(
                        row,
                        dict(observations=episode.observations, actions=episode.actions),
                        cfg["policy_order"],
                        store,
                        meter,
                        cfg,
                    )
                )
            rng = np.random.Generator(np.random.PCG64(cfg["random_execution_seed"]))
            for index, seed in enumerate(cfg["primary_seeds"][:4]):
                meter.context = dict(role="random_action", cohort="negative_control")

                def random_controller(state, warm, obs, step):
                    return rng.uniform(-1, 1, 6), dict(
                        distribution="independent_uniform[-1,1]^6",
                        seed=cfg["random_execution_seed"],
                    )

                episode = execute_episode(
                    study.env, random_controller, seed, 1000, meter, f"k1_random_{index:02d}"
                )
                records.append(episode_record(episode, "random_actions", "negative_control"))
                row = dict(
                    id=episode.tag,
                    group="executed_random_actions",
                    steps=episode.steps,
                    ancestry=f"native-reset-{seed}",
                    seed=seed,
                )
                scored.append(
                    score_trajectory(
                        row,
                        dict(observations=episode.observations, actions=episode.actions),
                        cfg["policy_order"],
                        store,
                        meter,
                        cfg,
                    )
                )
        return dict(
            primary=scores,
            confirmation=summary(confirmation) if confirmation else None,
            reference_qualified=bool(qualified),
            reference_status="provisionally_qualified" if qualified else "unqualified_candidate",
            episodes=records,
            trajectories=scored,
            groups=group_scores(scored, cfg["policy_order"], cfg) if scored else {},
            random_collection_triggered=bool(qualified),
            no_fallback_candidate=True,
        )
    finally:
        study.close()


def worker(out: Path) -> int:
    import torch

    from ..diagnostic_policies import LegacySACActor

    started = time.perf_counter()
    verify(out)
    cfg = json.loads((out / "config.json").read_text())
    seal = json.loads((out / "input_manifest.json").read_text())
    for name, expected in seal["files"].items():
        if sha(out / name) != expected:
            raise ValueError("Frozen diagnostic input changed: " + name)
    torch.set_num_threads(cfg["torch_threads"])
    result = dict(
        schema=cfg["schema"],
        execution_complete=False,
        error=None,
        scientific_scope=cfg["claim_scope"],
        protocol_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        stages={},
    )
    meters, actors, states = {}, {}, {}
    try:
        meters["A"] = DiagnosticMeter(
            out / "evidence/stage_a",
            controls=0,
            forward_rows=cfg["budgets"]["stage_a_forward_rows"],
            wall_seconds=cfg["budgets"]["stage_a_wall_seconds"],
            storage_bytes=cfg["budgets"]["stage_a_storage_bytes"],
        )
        for name in cfg["policy_order"]:
            path = out / "inputs" / cfg["policies"][name]["checkpoint"]
            if name == "reference":
                actors[name] = LegacySACActor.from_checkpoint(path, meter=meters["A"])
            else:
                actors[name], state, _ = meters["A"].measure(
                    "checkpoint_loads",
                    1,
                    lambda path=path: Policy.from_checkpoint(path),
                    purpose="frozen_roster_load",
                )
                states[name] = fingerprint(state)
            if actors[name].hash() != cfg["policies"][name]["policy_hash"]:
                raise ValueError("Frozen policy identity differs")
        identities = {name: actor.hash() for name, actor in actors.items()}
        store = PredictionStore(actors, cfg["prediction_batch_size"], cfg["cache_conflict_atol"])
        result["stages"]["A"] = stage_a(out, cfg, actors, store, meters["A"])
        meters["A"].close()
        write(out / "stage_a_result.json", result["stages"]["A"])
        meters["B"] = DiagnosticMeter(
            out / "evidence/stage_b",
            controls=cfg["budgets"]["reserved_live_controls"],
            forward_rows=cfg["budgets"]["stage_b_forward_rows"],
            wall_seconds=cfg["budgets"]["stage_b_wall_seconds"],
            storage_bytes=cfg["budgets"]["stage_b_storage_bytes"],
        )
        result["stages"]["B"] = stage_b(out, cfg, actors, store, meters["B"])
        if identities != {name: actor.hash() for name, actor in actors.items()}:
            raise ValueError("Diagnostic changed frozen actor parameters")
        result["final_policy_states"] = {
            name: meters["B"].save_arrays(
                "final_policy_" + name,
                **{key: value.detach().cpu().numpy() for key, value in actor.state_dict().items()},
            )
            for name, actor in actors.items()
        }
        result["policy_hashes"] = identities
        result["optimizer_state_fingerprints"] = states
        result["execution_complete"] = True
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise_again = isinstance(error, (KeyboardInterrupt, SystemExit))
    else:
        raise_again = False
    finally:
        for meter in meters.values():
            if not meter.closed:
                meter.close()
        result["work"] = {name: meter.totals() for name, meter in meters.items()}
        result["wall_seconds"] = time.perf_counter() - started
        result["stored_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if result["stored_bytes"] > cfg["budgets"]["storage_bytes"]:
            result["error"] = dict(
                type="StorageBudgetExceeded", message="Frozen storage ceiling exceeded"
            )
            result["execution_complete"] = False
        write(out / "result.json", result)
    if raise_again:
        return 130
    return int(result["error"] is not None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--out", required=True, type=Path)
    item.add_argument("--artifact-root", required=True, type=Path)
    item = sub.add_parser("worker")
    item.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.out.resolve(), args.artifact_root.resolve())
        return 0
    return worker(args.out.resolve())
