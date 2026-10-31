"""Saved-only input inventory for the bounded Kendall compatibility diagnostic.

No simulator, policy construction, forward prediction or optimizer is imported.
Whole historical raw HDF files are not rehashed. Small inputs and compact
exports are sealed, and extracted arrays are checked against saved descriptors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .provenance import sha, write

V1 = "2026-10-29_walker_cio_mppi_001"
V2 = "2026-10-29_cio_feedback_v2_001"
V3 = "2026-10-30_guided_data_comparison_v3_001"
L1 = "2026-10-30_student_learnability_l1_001"
H4 = "2026-10-30_adaptive_coupling_h4_1_r1_001"
POLICY_PATHS = {
    "h4_parent": (H4, "input.pt"),
    "h4_independent": (H4, "checkpoints/independent_2000.pt"),
    "h4_fixed_high": (H4, "checkpoints/fixed_high_2000.pt"),
    "h4_backoff": (H4, "checkpoints/backoff_2000.pt"),
    "l1_clipped": (L1, "checkpoints/clipped_10000.pt"),
}


def native_observations(states: np.ndarray) -> np.ndarray:
    """Native pre-action clipped feature map; no simulator reconstruction."""
    value = np.asarray(states)
    if value.ndim != 2 or value.shape[1] != 19 or not np.isfinite(value).all():
        raise ValueError("Finite (N,19) physical states required")
    return np.concatenate((value[:, 2:10], np.clip(value[:, 10:19], -10, 10)), axis=1)


def validate_pairs(states: np.ndarray, observations: np.ndarray, actions: np.ndarray) -> None:
    """Check actual alignment/bounds without changing or clipping action labels."""
    n = len(states)
    if observations.shape != (n, 17) or actions.shape != (n, 6):
        raise ValueError("Unaligned saved observations/actions")
    if not np.array_equal(native_observations(states), observations):
        raise ValueError("Saved observations are not native pre-action observations")
    if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
        raise ValueError("Executed torque labels must be finite and bounded")


def membership(tag: str) -> dict[str, str]:
    """Training exposure, not an assertion of population holdout independence."""
    result = {}
    for policy in POLICY_PATHS:
        if tag.startswith("discovery_"):
            status = "fitted_original_discovery"
        elif tag.startswith("v2_r0_trial_guided1_"):
            status = (
                "trajectory_held_out_same_resets_as_fitted_independent"
                if policy == "l1_clipped"
                else "fitted_retained_guided_history"
            )
        elif tag.startswith("v2_r0_trial_raw_"):
            status = (
                "fitted_added_independent"
                if policy == "l1_clipped"
                else "trajectory_held_out_same_resets_as_fitted_guided"
            )
        elif tag.startswith("h4_r1_"):
            arm = tag[len("h4_r1_") :].rsplit("_", 1)[0]
            if policy == "h4_parent":
                status = "not_fitted_but_acquisition_conditioned_on_this_guide"
            elif policy == "l1_clipped":
                status = "not_fitted_new_reset_context"
            elif policy == "h4_" + arm:
                status = "fitted_new_acquisition"
            else:
                status = "trajectory_held_out_same_resets_as_fitted_h4_arm"
        else:
            raise ValueError("Unknown teacher source")
        result[policy] = status
    return result


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _roster(repo: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    teachers = []
    for i in range(4):
        teachers.append(dict(id=f"discovery_{i:02d}", group="original_discovery", run=V1))
    for arm, group in (("raw", "later_independent"), ("guided1", "later_guided")):
        for i in range(2):
            teachers.append(dict(id=f"v2_r0_trial_{arm}_{i}", group=group, run=V2))
    for arm in ("independent", "fixed_high", "backoff"):
        for i in range(2):
            teachers.append(dict(id=f"h4_r1_{arm}_{i:02d}", group=f"h4_{arm}", run=H4))
    standalone = []
    for run, selected in (
        (H4, ("parent", "independent", "fixed_high", "backoff")),
        (L1, ("clipped",)),
    ):
        result = _read(repo / "runs" / run / "result.json")
        for arm in selected:
            policy = f"h4_{arm}" if run == H4 else "l1_clipped"
            for tag in result["cases"]["primary"][arm]["tags"]:
                standalone.append(dict(id=tag, run=run, policy=policy, cohort="primary"))
    return teachers, standalone


def _scan(run: Path, tags: set[str], teacher_tags: set[str], context: bool) -> dict[str, Any]:
    episodes, decisions, private, controllers, agreements = {}, {}, {}, {}, {}
    reset_seeds: set[int] = set()
    last = None
    with (run / "evidence/ledger.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            event = row["event"]
            tag = row.get("episode", row.get("tag", ""))
            if event == "reset_requested" and type(row.get("seed")) is int:
                reset_seeds.add(row["seed"])
            if event == "episode_complete" and row.get("tag") in tags:
                if row["tag"] in episodes:
                    raise ValueError("Duplicate saved episode")
                episodes[row["tag"]] = row
            if tag in teacher_tags and event == "decision":
                controllers.setdefault(
                    tag,
                    {k: row.get(k) for k in ("actor_hash", "alpha", "enabled", "eta_u")},
                )
                entry = decisions.setdefault(tag, {})
                if context:
                    entry[row["step"]] = row["bank_digest"]
                elif not entry:
                    entry.update(
                        {k: row.get(k) for k in ("actor_hash", "alpha", "enabled", "eta_u")}
                    )
            if tag in teacher_tags and event == "independent_planner_before":
                if context:
                    state = row["private"]
                    private.setdefault(tag, {})[row["step"]] = dict(
                        nominal=np.asarray(state["U"], dtype=np.float64),
                        rng=hashlib.sha256(
                            json.dumps(
                                state["rng_state"], sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest(),
                        initial_plan=bool(state["initial_plan"]),
                    )
                else:
                    private[tag] = private.get(tag, 0) + 1
            if tag in teacher_tags and event == "teacher_complete":
                reference = row.get("agreement_evidence")
                if reference is not None:
                    agreements[tag] = reference
            last = row
    if last is None or last["event"] != "runtime_close":
        raise ValueError("Only closed saved runs are admitted")
    if set(episodes) != tags:
        raise ValueError("Saved episode roster is incomplete")
    return dict(
        episodes=episodes,
        decisions=decisions,
        private=private,
        controllers=controllers,
        agreements=agreements,
        reset_seeds=sorted(reset_seeds),
    )


def _policy(path: Path) -> dict[str, Any]:
    # Decoding existing checkpoint metadata incurs no inference or fitting.
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    expected = {
        "network.0.weight": (128, 17),
        "network.2.weight": (128, 128),
        "network.4.weight": (6, 128),
    }
    if payload["schema"] != "walker-tanh-policy-v1" or payload["obs_dim"] != 17:
        raise ValueError("Unexpected frozen policy interface")
    if any(tuple(state[k].shape) != shape for k, shape in expected.items()):
        raise ValueError("Unexpected frozen architecture")
    return dict(
        checkpoint=str(path),
        file_sha256=sha(path),
        policy_hash=payload["policy_hash"],
        architecture=[17, 128, 128, 6],
        activations=["tanh", "tanh", "tanh"],
        feature="clipped",
        action_scaling="direct six tanh torques in [-1,1]",
        execution="deterministic; no planner prefix/fallback",
        normalization={
            "mean": state["observation_mean"].numpy().tolist(),
            "std": state["observation_std"].numpy().tolist(),
        },
        normalization_source="own frozen checkpoint; never refit for diagnostic",
        metadata=payload.get("metadata"),
        fit_data=payload["optimizer_state"]["fit_data"],
    )


def _collect(repo: Path, context: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    teachers, standalone = _roster(repo)
    scans = {}
    for run in (V1, V2, H4, L1):
        wanted = {r["id"] for r in teachers + standalone if r["run"] == run}
        teacher_tags = {r["id"] for r in teachers if r["run"] == run}
        scans[run] = _scan(repo / "runs" / run, wanted, teacher_tags, context)
    sources = {}
    for run in (V1, V2, V3, L1, H4):
        for name in ("config.json", "result.json", "manifest.json"):
            path = repo / "runs" / run / name
            sources[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path), recomputed=True)
    for run, name in (
        (L1, "inputs/physical_rows.npz"),
        (L1, "inputs/clipped.npz"),
        (L1, "inputs/provenance.json"),
        (V3, "inputs/teachers.h5"),
        (H4, "inputs/history.npz"),
        (H4, "exports/complete_executed_evidence.h5"),
    ):
        path = repo / "runs" / run / name
        sources[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path), recomputed=True)
    for run in (L1, V3, H4):
        seal = _read(repo / "runs" / run / "input_manifest.json")
        for name, expected in seal["files"].items():
            source = str(repo / "runs" / run / name)
            if source in sources and sources[source]["sha256"] != expected:
                raise ValueError("Historical sealed input changed: " + source)
    compact = _read(repo / "runs" / H4 / "exports/full_local_evidence_locator.json")["compact"]
    compact_path = str(repo / "runs" / H4 / "exports" / compact["file"])
    if sources[compact_path]["sha256"] != compact["sha256"]:
        raise ValueError("H4 compact export changed")
    raw_l1 = repo / "runs" / L1 / "evidence/raw.h5"
    sources[str(raw_l1)] = dict(
        bytes=raw_l1.stat().st_size,
        sha256="48461e6c55172eec07fbb57757804db326937fd50e87f40ddfde49b6810e066e",
        recomputed=False,
        scope="Existing independent audit file binding; extracted arrays separately verified",
    )
    raw_h4 = repo / "runs" / H4 / "evidence/raw.h5"
    sources[str(raw_h4)] = dict(
        bytes=raw_h4.stat().st_size,
        sha256="5805a0ea194b0cf2228c4a98830f6156887856f6eb46d1790e76942712c7a48c",
        recomputed=False,
        scope="Existing full local locator; only six saved guide-prediction groups read, not a repeat whole-raw audit",
    )
    for run, digest in (
        (V1, "257b008eb453bd3d8a6e6bf623229af1fc43f466fb16101dc26ee8fc9a676bc5"),
        (V2, "43593c4aad640ae789f160f7ea6186f4631cf4237a8765ac5e8dd6f004dfc147"),
        (L1, "f3b196af8430094dcb1aeb52b0b9b05f892b721a593bf7d8ceb4e6ffc14d149e"),
        (H4, "c229397f0e3118da76a3db3ba1e287f8e1353fe26c975ae1e1e7b443e04b5f2d"),
    ):
        path = repo / "runs" / run / "evidence/ledger.jsonl"
        sources[str(path)] = dict(
            bytes=path.stat().st_size,
            sha256=digest,
            recomputed=False,
            scope="Existing closed independent-audit receipt; streamed episode/context selection, no repeat whole-file audit",
        )
    for entry in teachers + standalone:
        event = scans[entry["run"]]["episodes"][entry["id"]]
        entry.update(
            file=("teachers/" if entry in teachers else "standalone/") + entry["id"] + ".npz",
            steps=event["steps"],
            seed=event["seed"],
            outcome={k: event[k] for k in ("complete", "success", "survived", "cost", "mae")},
            source=dict(
                run=str(repo / "runs" / entry["run"]), ledger_event=event["id"], raw=event["raw"]
            ),
            ancestry=dict(
                reset_seed=event["seed"],
                reset_cluster=f"native-reset-{event['seed']}",
                scope="Historical development trajectory; paired arms share reset clusters",
            ),
        )
        if entry in teachers:
            cfg = _read(repo / "runs" / entry["run"] / "config.json")
            rho = 0.0
            if entry["group"] in ("later_guided", "h4_fixed_high"):
                rho = 0.1199341983507337
            elif entry["group"] == "h4_backoff":
                rho = 0.05996709917536685
            entry["controller"] = dict(
                name="independent task-only proposal + external selector",
                H=cfg["mppi"]["H"],
                K=cfg["mppi"]["K"],
                rho=rho,
                eta_u=0.01,
                effective_guidance_enabled=rho > 0,
            )
            actual = scans[entry["run"]]["controllers"][entry["id"]]
            effective = actual["alpha"] if actual["enabled"] else 0.0
            if effective != rho or actual["eta_u"] != 0.01:
                raise ValueError("Recorded controller differs from declared roster")
            entry["controller"].update(
                actor_hash=actual["actor_hash"], declared_alpha=actual["alpha"]
            )
            if entry["run"] == H4 and entry["id"] in scans[H4]["agreements"]:
                entry["saved_predictions"] = dict(
                    h4_parent=dict(
                        key="saved_predictions_h4_parent",
                        policy_hash=actual["actor_hash"],
                        source_file=str(raw_h4),
                        source_ref=scans[H4]["agreements"][entry["id"]],
                        rows=entry["steps"],
                    )
                )
            entry["membership"] = membership(entry["id"])
            entry["context_rows"] = (
                len(scans[entry["run"]]["private"].get(entry["id"], {}))
                if context
                else scans[entry["run"]]["private"].get(entry["id"], 0)
            )
            if entry["steps"] != 1000 or not all(
                entry["outcome"][k] for k in ("complete", "success", "survived")
            ):
                raise ValueError(
                    "Teacher roster must contain complete successful executed episodes"
                )
    result = dict(
        schema="walker-kendall-saved-input-inventory-v1",
        trajectories=teachers,
        historical_standalone=standalone,
        policies={
            name: _policy(repo / "runs" / run / filename)
            for name, (run, filename) in POLICY_PATHS.items()
        },
        datasets={
            "history": "datasets/history.npz",
            "auxiliary": "datasets/auxiliary.npz",
            "l1_training": "datasets/l1_training.npz",
        },
        source_files=sources,
        source_reset_seeds={
            str(repo / "runs" / run): scans[run]["reset_seeds"] for run in (V1, V2, H4, L1)
        },
        missing_evidence=[
            "Original discovery full raw HDF, endpoint/warmstart/integration arrays are remote-only locally; complete pre-action states/actions survive in audited L1 physical_rows. Original planner nominal/RNG and bank digests remain inline in local ledger.",
            "Full model banks are not copied; H4 compact stores selected bank snapshots only. No new same-state queries.",
        ],
        saved_prediction_scope="Historical full_dataset_mse stores aggregate scalars. Six H4 teacher_complete references preserve incoming-guide policy_actions on those exact executed observations (6000 reusable rows). Standalone actions provide own-logged deterministic predictions. All other new teacher-matrix forward rows must be metered.",
        new_forward_rows=0,
        new_controls=0,
        new_training_updates=0,
    )
    return result, scans


def inventory(repo_root: str | Path) -> dict[str, Any]:
    """Read the exact fixed teacher/policy roster without any predictions."""
    return _collect(Path(repo_root).resolve(), context=False)[0]


def _decode(group: h5py.Group, key: str) -> np.ndarray:
    dataset = group[key]
    value = np.asarray(dataset[()])
    expected = dataset.attrs.get("sha256")
    if expected and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != expected:
        raise ValueError("Saved array bytes changed: " + key)
    return value


def _from_hdf(path: Path, group_name: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as stream:
        group = stream[group_name]
        full_states = _decode(group, "states")
        result = dict(
            states=full_states[:-1],
            terminal_state=full_states[-1],
            observations=_decode(group, "observations"),
            actions=_decode(group, "actions"),
        )
        for key in ("warmstarts", "integrations"):
            if key in group:
                value = _decode(group, key)
                result[key] = value[:-1]
                result["terminal_" + key] = value[-1]
        for key in ("costs", "velocities"):
            if key in group:
                result[key] = _decode(group, key)
        return result


def _save(path: Path, values: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **values)
    return dict(
        file_sha256=sha(path),
        bytes=path.stat().st_size,
        arrays={
            k: dict(
                shape=list(v.shape),
                dtype=str(v.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
            )
            for k, v in values.items()
        },
    )


def extract_inputs(repo_root: str | Path, destination: str | Path) -> dict[str, Any]:
    """Create-only curated inputs; original evidence and checkpoint bytes stay intact."""
    repo, destination = Path(repo_root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report, scans = _collect(repo, context=True)
    with np.load(repo / "runs" / L1 / "inputs/physical_rows.npz", allow_pickle=False) as stream:
        physical = {k: np.array(stream[k]) for k in stream.files}
    with np.load(repo / "runs" / L1 / "inputs/clipped.npz", allow_pickle=False) as stream:
        clipped = {k: np.array(stream[k]) for k in stream.files}
    extracted = {}
    for entry in report["trajectories"] + report["historical_standalone"]:
        teacher = "membership" in entry
        tag, run, n = entry["id"], entry["run"], entry["steps"]
        if run == V1:
            mask = (physical["row_kind"] == "executed_discovery") & (physical["row_episode"] == tag)
            values = dict(
                states=physical["states"][mask],
                observations=clipped["observations"][mask],
                actions=physical["actions"][mask],
            )
            row_steps = physical["row_step"][mask]
            for key in ("observations", "actions"):
                digest = hashlib.sha256(np.ascontiguousarray(values[key]).tobytes()).hexdigest()
                if digest != entry["source"]["raw"]["arrays"][key]["sha256"]:
                    raise ValueError("Original executed rows differ from saved source descriptors")
        else:
            path = repo / "runs" / (V3 if run == V2 else run)
            if run == V2:
                source = path / "inputs/teachers.h5"
                group_name = tag
            elif run == H4:
                source = path / "exports/complete_executed_evidence.h5"
                group_name = entry["source"]["raw"]["group"]
            else:
                source = path / "evidence/raw.h5"
                group_name = entry["source"]["raw"]["group"]
            values = _from_hdf(source, group_name)
            row_steps = np.arange(n)
        validate_pairs(values["states"], values["observations"], values["actions"])
        for key in ("observations", "actions", "costs", "velocities"):
            if key in values:
                actual_hash = hashlib.sha256(
                    np.ascontiguousarray(values[key]).tobytes()
                ).hexdigest()
                if actual_hash != entry["source"]["raw"]["arrays"][key]["sha256"]:
                    raise ValueError("Extracted bytes differ from source ledger: " + key)
        if len(values["actions"]) != n or not np.array_equal(row_steps, np.arange(n)):
            raise ValueError("Executed row order/count changed")
        values["row_steps"] = row_steps
        entry["availability"] = {
            key: key in values for key in ("terminal_state", "warmstarts", "integrations")
        }
        if teacher:
            private = scans[run]["private"].get(tag, {})
            digests = scans[run]["decisions"].get(tag, {})
            values.update(
                planner_nominal=np.full((n, 48, 6), np.nan),
                planner_rng_fingerprint=np.full(n, "", dtype="U64"),
                bank_digest=np.full(n, "", dtype="U64"),
                planner_initial_plan=np.zeros(n, dtype=bool),
                planner_context_available=np.zeros(n, dtype=bool),
            )
            for step, state in private.items():
                if not 0 <= step < n or state["nominal"].shape != (48, 6):
                    raise ValueError("Planner snapshot alignment changed")
                values["planner_nominal"][step] = state["nominal"]
                values["planner_rng_fingerprint"][step] = state["rng"]
                values["planner_initial_plan"][step] = state["initial_plan"]
                values["planner_context_available"][step] = True
            for step, digest in digests.items():
                values["bank_digest"][step] = digest
            cache = entry.get("saved_predictions", {}).get("h4_parent")
            if cache:
                if cache["policy_hash"] != report["policies"]["h4_parent"]["policy_hash"]:
                    raise ValueError("Saved teacher guide predictions have a different owner")
                with h5py.File(cache["source_file"], "r") as stream:
                    group = stream[cache["source_ref"]["group"]]
                    predictions = _decode(group, "policy_actions")
                    recorded_actions = _decode(group, "executed_actions")
                if not np.array_equal(recorded_actions, values["actions"]):
                    raise ValueError("Saved guide prediction rows have different executed labels")
                expected = cache["source_ref"]["arrays"]["policy_actions"]["sha256"]
                if hashlib.sha256(predictions.tobytes()).hexdigest() != expected:
                    raise ValueError("Saved guide prediction descriptor changed")
                if predictions.shape != (n, 6) or not np.isfinite(predictions).all():
                    raise ValueError("Saved guide predictions are invalid")
                values[cache["key"]] = predictions
        entry["extracted"] = _save(destination / entry["file"], values)
        if teacher:
            extracted[tag] = values
    common = {
        k: physical[k][:4096] for k in ("states", "actions", "row_kind", "row_episode", "row_step")
    }
    history_states = np.concatenate(
        [common["states"]] + [extracted[f"v2_r0_trial_guided1_{i}"]["states"] for i in range(2)]
    )
    with np.load(repo / "runs" / H4 / "inputs/history.npz", allow_pickle=False) as stream:
        history = {k: np.array(stream[k]) for k in stream.files}
    validate_pairs(history_states, history["observations"], history["actions"])
    history.update(
        states=history_states,
        row_kind=np.concatenate((common["row_kind"], np.full(2000, "executed_added_guided"))),
        row_episode=np.concatenate(
            (common["row_episode"], np.repeat([f"v2_r0_trial_guided1_{i}" for i in range(2)], 1000))
        ),
        row_steps=np.concatenate((common["row_step"], np.tile(np.arange(1000), 2))),
    )
    l1_data = dict(
        states=physical["states"],
        observations=clipped["observations"],
        actions=physical["actions"],
        row_kind=physical["row_kind"],
        row_episode=physical["row_episode"],
        row_steps=physical["row_step"],
    )
    auxiliary_mask = np.isin(
        physical["row_kind"], ["hypothetical_perturbed_resolve", "executed_recovery"]
    )
    auxiliary = {k: v[auxiliary_mask] for k, v in l1_data.items()}
    for name, values in (("history", history), ("l1_training", l1_data), ("auxiliary", auxiliary)):
        report.setdefault("dataset_bindings", {})[name] = _save(
            destination / report["datasets"][name], values
        )
    report["context_fingerprint_contract"] = (
        "SHA256 of sorted compact JSON of saved PCG64 state; distinct from fit/array hashes"
    )
    report["source_ledger_scope"] = (
        "One streaming selection pass per ledger during extraction; old whole-file/raw audits reused. No model evaluation or action predictions."
    )
    write(destination / "inventory.json", report)
    return report
