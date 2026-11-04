"""Read-only T0 outcomes, pairing proofs and episode-respecting diagnostics.

No policy, planner or simulator module is imported. Spectral kernels are
extracted from preserved source, without importing its SAC execution helpers.
These supporting measurements never admit data or change the frozen decision.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..provenance import fingerprint, sha, write

NORMALIZATION = Path(
    "/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_student_update_w2_001/inputs/normalization.npz"
)
KERNEL_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "results/sac_diagnostics_review_001/source/scripts/diagnostics/action_spectrum.py"
)
TRACE_FIELDS = (
    "states",
    "warmstarts",
    "integrations",
    "observations",
    "actions",
    "costs",
    "native_fall_cost",
    "policy_actions",
    "action_gap_mse",
)


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same(a: Any, b: Any) -> bool:
    first, second = np.asarray(a), np.asarray(b)
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and first.tobytes() == second.tobytes()
    )


class Inputs:
    """Authenticate every input opened by the renderer, once per path."""

    def __init__(self) -> None:
        self.bindings: dict[str, dict] = {}

    def bound(self, value: Path | dict) -> Path:
        path = Path(value["path"] if isinstance(value, dict) else value).resolve()
        if str(path) not in self.bindings:
            self.bindings[str(path)] = dict(
                path=str(path), bytes=path.stat().st_size, sha256=sha(path)
            )
        if isinstance(value, dict):
            _need(self.bindings[str(path)] == value, "Changed report input: " + str(path))
        return path

    def json(self, value: Path | dict) -> dict:
        return json.loads(self.bound(value).read_text())

    def npz(self, value: Path | dict) -> dict[str, np.ndarray]:
        with np.load(self.bound(value), allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}

    def trace(self, value: dict) -> dict[str, np.ndarray]:
        import h5py

        with h5py.File(self.bound(value), "r") as archive:
            return {name: archive[name][:] for name in archive}


def episode_outcome(raw: dict, horizon: int = 300, terminal_weight: float = 20) -> dict:
    """Recompute native outcomes; terminal cost is already in native costs."""
    _need(set(TRACE_FIELDS) <= set(raw), "Missing executed trace fields")
    n = len(raw["actions"])
    _need(1 <= n <= horizon, "Actual, nonempty executed prefix required")
    for name in TRACE_FIELDS:
        count = n + 1 if name in TRACE_FIELDS[:4] else n
        _need(
            len(raw[name]) == count and np.isfinite(raw[name]).all(), "Invalid trace field: " + name
        )
    states, actions = raw["states"], raw["actions"]
    _need(
        states.shape == (n + 1, 19) and actions.shape == (n, 6),
        "Native N+1 Walker indexing required",
    )
    obs = np.concatenate((states[:, 2:10], np.clip(states[:, 10:19], -10, 10)), axis=1)
    _need(_same(obs, raw["observations"]), "Incorrect saved pre-action observations")
    _need(
        raw["policy_actions"].shape == actions.shape
        and np.all(np.abs(actions) <= 1)
        and np.all(np.abs(raw["policy_actions"]) <= 1),
        "Bounded deployed torque arrays required",
    )
    healthy = (states[:, 2] > 0.8) & (states[:, 2] < 2) & (states[:, 3] > -1) & (states[:, 3] < 1)
    _need(healthy[:-1].all(), "Post-fall states must not be appended")
    fell = not bool(healthy[-1])
    _need(n == horizon or fell, "An incomplete healthy trace is not a closed episode")
    expected_fall = np.zeros(n)
    if fell:
        expected_fall[-1] = terminal_weight
    _need(
        np.array_equal(raw["native_fall_cost"], expected_fall),
        "Native fall terminal must occur exactly once",
    )
    gaps = np.mean((actions - raw["policy_actions"]) ** 2, axis=1)
    _need(
        np.allclose(gaps, raw["action_gap_mse"], rtol=0, atol=1e-12),
        "Saved deployed-policy gap differs",
    )
    velocity_mae = float(np.mean(np.abs(states[1:, 10] - 1.5)))
    survival = n == horizon and not fell
    native = float(np.sum(raw["costs"]))
    tail = float(5 * (horizon - n)) if fell else 0.0
    return dict(
        controls=n,
        requested_controls=horizon,
        fell=fell,
        full_survival=survival,
        local_walking=bool(survival and velocity_mae <= 0.3),
        velocity_mae=velocity_mae,
        raw_native_cost=native,
        native_fall_terminal=float(expected_fall.sum()),
        native_cost_without_fall_terminal=native - float(expected_fall.sum()),
        absorbing_tail_cost=tail,
        complete_cost=native + tail,
        first_teacher_policy_action_mse=float(gaps[0]),
        mean_teacher_policy_action_mse=float(gaps.mean()),
        mean_prefix_reconstruction_R=float(3 * gaps.mean()),
    )


def pair_difference(left: dict, right: dict, left_raw: dict, right_raw: dict) -> dict:
    """Compare real common prefixes only, retaining failures and state divergence."""
    n = min(len(left_raw["actions"]), len(right_raw["actions"]))
    _need(n >= 1, "Common executed action required")
    _need(
        _same(left_raw["states"][0], right_raw["states"][0]),
        "Paired episodes must start at the same state",
    )
    differences = left_raw["states"][1 : n + 1] - right_raw["states"][1 : n + 1]
    maxima = np.max(np.abs(differences), axis=1)
    crossed = np.flatnonzero(maxima > 1e-10)
    same_alpha = left["alpha"] == right["alpha"]
    same_mode = left["mode"] == right["mode"]
    kind = (
        "alpha_same_initialization"
        if same_mode
        else "initialization_same_alpha"
        if same_alpha
        else "joint_alpha_and_initialization"
    )
    return dict(
        index=left["index"],
        left_mode=left["mode"],
        right_mode=right["mode"],
        left_alpha=left["alpha"],
        right_alpha=right["alpha"],
        contrast=kind,
        common_controls=n,
        left_controls=len(left_raw["actions"]),
        right_controls=len(right_raw["actions"]),
        first_action_rms=float(
            np.sqrt(np.mean((left_raw["actions"][0] - right_raw["actions"][0]) ** 2))
        ),
        common_prefix_action_rms=float(
            np.sqrt(np.mean((left_raw["actions"][:n] - right_raw["actions"][:n]) ** 2))
        ),
        post_initial_state_max_abs=float(maxima.max()),
        first_divergent_boundary_gt1e_10=int(crossed[0] + 1) if len(crossed) else None,
        states_diverge_beyond_initial=bool(len(crossed)),
        no_padded_tails=True,
    )


def frozen_decision(episodes: list[dict], pairs: list[dict], threshold: float = 0.01) -> dict:
    alpha_pairs = [row for row in pairs if row["contrast"] == "alpha_same_initialization"]
    _need(
        len(episodes) == 16 and len(alpha_pairs) == 8,
        "Complete frozen roster required before decision",
    )
    regressions = []
    for case in range(4):
        for mode in ("zero", "shadow_carried"):
            values = [row for row in episodes if row["index"] == case and row["mode"] == mode]
            _need(len(values) == 2, "Exactly two alpha conditions per initialization")
            base = next(row for row in values if row["alpha"] == 0)
            positive = next(row for row in values if row["alpha"] > 0)
            earlier_fall = positive["fell"] and (
                not base["fell"] or positive["controls"] < base["controls"]
            )
            loss_walk = base["local_walking"] and not positive["local_walking"]
            loss_survival = base["full_survival"] and not positive["full_survival"]
            regressions.append(
                dict(
                    index=case,
                    mode=mode,
                    earlier_fall=bool(earlier_fall),
                    lost_local_walking=bool(loss_walk),
                    lost_full_survival=bool(loss_survival),
                )
            )
    effective = any(row["common_prefix_action_rms"] >= threshold for row in alpha_pairs)
    destabilizing = any(
        row["earlier_fall"] or row["lost_local_walking"] or row["lost_full_survival"]
        for row in regressions
    )
    verdict = (
        "destabilizing"
        if destabilizing
        else "ineffective"
        if not effective
        else "effective_without_declared_competence_regression"
    )
    return dict(
        effective=effective,
        destabilizing=destabilizing,
        verdict=verdict,
        executed_action_rms_threshold=threshold,
        alpha_pairs=8,
        matched_regressions=regressions,
        following="Stop this block; no fitting or automatic follow-up."
        if destabilizing or not effective
        else "A T3 protocol may be proposed for review; learnability remains untested.",
        scope="300-control local controller outcomes; not standalone policy walking, learning, convergence or superiority.",
    )


def _kernels(path: Path) -> dict:
    """Load function definitions only; top-level SAC imports never execute."""
    from scipy import signal

    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"spectrum", "lowpass"}
    ]
    _need(
        {node.name for node in functions} == {"spectrum", "lowpass"},
        "Preserved spectral kernels missing",
    )
    namespace = dict(np=np, signal=signal, FS=125.0, FIRST=0, SEGMENT=300)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def neighbour_bands(
    observations: list[np.ndarray],
    bands: dict[str, list[np.ndarray]],
    *,
    whole_episode: bool,
    k: int = 10,
) -> list[dict]:
    """Same-source neighbors; finite support and whole-trajectory exclusion explicit."""
    from scipy.spatial.distance import cdist

    if not observations:
        return []
    x = np.concatenate(observations)
    episode = np.concatenate(
        [np.full(len(value), index) for index, value in enumerate(observations)]
    )
    step = np.concatenate([np.arange(len(value)) for value in observations])
    distances = cdist(x, x, "sqeuclidean")
    same_episode = episode[:, None] == episode[None]
    excluded = (
        same_episode if whole_episode else same_episode & (np.abs(step[:, None] - step[None]) < 100)
    )
    distances[excluded] = np.inf
    neighbor = np.full((len(x), k), -1, dtype=int)
    valid = np.isfinite(distances).sum(1) >= k
    for row in np.flatnonzero(valid):
        neighbor[row] = np.lexsort((np.arange(len(x)), distances[row]))[:k]
    rows = []
    for band_name, episodes in bands.items():
        y = np.concatenate(episodes)
        for held in range(len(observations)):
            query = (episode == held) & valid
            target = y[query]
            denominator = float(np.sum((target - target.mean(0)) ** 2)) if len(target) else 0.0
            prediction = y[neighbor[query]].mean(1) if len(target) else target
            r2 = (
                float(1 - np.sum((target - prediction) ** 2) / denominator)
                if denominator > 0
                else None
            )
            rows.append(
                dict(
                    episode=held,
                    exclusion="whole_restart_trajectory"
                    if whole_episode
                    else "same_episode_within100_controls",
                    band=band_name,
                    k=k,
                    total_queries=int((episode == held).sum()),
                    finite_neighbor_queries=int(query.sum()),
                    missing_neighbor_queries=int(((episode == held) & ~valid).sum()),
                    target_sum_squared_deviation=denominator,
                    r2=r2,
                    status="OBSERVED"
                    if r2 is not None
                    else "MISSING_INSUFFICIENT_SUPPORT_OR_ZERO_VARIANCE",
                )
            )
    return rows


def supporting_diagnostics(
    episodes: list[dict], traces: dict, mean: np.ndarray, std: np.ndarray, kernels: dict
) -> tuple[list[dict], list[dict], dict]:
    spectral, neighbors, arrays = [], [], {}
    for mode in ("zero", "shadow_carried"):
        for alpha in sorted({row["alpha"] for row in episodes}):
            roster = sorted(
                [row for row in episodes if row["mode"] == mode and row["alpha"] == alpha],
                key=lambda row: row["index"],
            )
            states, low, high, full, eligible, skipped = [], [], [], [], [], []
            for row in roster:
                raw = traces[row["key"]]
                action = raw["actions"]
                n = len(action)
                metadata = dict(
                    index=row["index"],
                    mode=mode,
                    alpha=alpha,
                    controls=n,
                    burn=0,
                    cutoff_hz=5.0,
                    sample_rate_hz=125.0,
                    nperseg=min(300, n),
                    noverlap=min(300, n) // 2,
                )
                if n < 32:
                    skipped.append((row["index"], n))
                    spectral.append(
                        dict(
                            metadata,
                            status="MISSING_PREFIX_SHORTER_THAN32",
                            high_band_variance=None,
                            high_variance_share=None,
                            above5hz_psd_bin_power_share=None,
                            teacher_policy_residual_variance=None,
                        )
                    )
                    continue
                kernels["SEGMENT"] = min(300, n)
                frequency, power = kernels["spectrum"]([action])
                lower = kernels["lowpass"]([action], 5.0)[0]
                upper = action - lower
                total_variance = float(np.var(action, axis=0).sum())
                high_variance = float(np.var(upper, axis=0).sum())
                spectral.append(
                    dict(
                        metadata,
                        status="OBSERVED",
                        high_band_variance=high_variance,
                        high_variance_share=high_variance / total_variance
                        if total_variance > 0
                        else None,
                        above5hz_psd_bin_power_share=float(power[frequency > 5].sum() / power.sum())
                        if power.sum() > 0
                        else None,
                        teacher_policy_residual_variance=float(
                            np.var(action - raw["policy_actions"], axis=0).sum()
                        ),
                    )
                )
                key = f"case{row['index']}_{mode}_{alpha:.12g}"
                arrays[key + "_frequency"] = frequency
                arrays[key + "_power"] = power
                states.append((raw["observations"][:-1] - mean) / std)
                low.append(lower)
                high.append(upper)
                full.append(action)
                eligible.append(row["index"])
            for index, count in skipped:
                for exclusion in ("same_episode_within100_controls", "whole_restart_trajectory"):
                    for band in ("full", "low", "high"):
                        neighbors.append(
                            dict(
                                index=index,
                                mode=mode,
                                alpha=alpha,
                                available_restart_episodes=len(states),
                                exclusion=exclusion,
                                band=band,
                                k=10,
                                total_queries=count,
                                finite_neighbor_queries=0,
                                missing_neighbor_queries=count,
                                target_sum_squared_deviation=None,
                                r2=None,
                                status="MISSING_PREFIX_SHORTER_THAN32",
                            )
                        )
            if states:
                for heldout in (False, True):
                    for value in neighbour_bands(
                        states, dict(full=full, low=low, high=high), whole_episode=heldout
                    ):
                        index = eligible[value.pop("episode")]
                        neighbors.append(
                            dict(
                                index=index,
                                mode=mode,
                                alpha=alpha,
                                available_restart_episodes=len(states),
                                **value,
                            )
                        )
    return spectral, neighbors, arrays


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _spread_ratio(c: np.ndarray, r: np.ndarray, indices: np.ndarray, alpha: float) -> float | None:
    if len(indices) < 2 or not np.isfinite(c[indices]).all() or not np.isfinite(r[indices]).all():
        return None
    denominator = float(np.std(c[indices]))
    return float(np.std(alpha * r[indices]) / denominator) if denominator > 0 else None


def render(
    base: Path,
    out: Path,
    *,
    normalization: Path | None = None,
    kernel_source: Path | None = None,
    cpu_seconds: float = 160.0,
) -> dict:
    """Render a completed assay; never launch or repair an incomplete roster."""
    import h5py

    started, cpu = time.monotonic(), time.process_time()
    inputs = Inputs()
    base, out = base.resolve(), out.resolve()
    supervisor = inputs.json(base / "t0/result.json")
    _need(supervisor["status"] == "COMPLETED", "All native T0 jobs must finish before reporting")
    launch = inputs.json(supervisor["launch"])
    config = inputs.json(launch["config"])
    calibration = inputs.json(launch["calibration"])
    inventory = inputs.json(launch["inventory"])
    alpha = float(calibration["selected_alpha"])
    _need(
        alpha == launch["selected_alpha"] == calibration["alphas"]["competitive"]["0.3"],
        "Frozen coefficient selection changed",
    )
    _need(
        config["actor_hash"] == inventory["actor"]["actor_hash"] == calibration["actor_hash"],
        "Actor mismatch across phases",
    )
    norm_path = inputs.bound(normalization or NORMALIZATION)
    norm = inputs.npz(norm_path)
    mean, std = norm["mean"], norm["std"]
    _need(
        mean.shape == std.shape == (17,)
        and mean.dtype == std.dtype == np.float32
        and np.isfinite(mean).all()
        and np.isfinite(std).all()
        and np.all(std > 0),
        "Frozen W2 normalization buffers required",
    )
    norm_hash = fingerprint(dict(mean=mean, std=std))
    kernel_path = inputs.bound(kernel_source or KERNEL_SOURCE)
    kernels = _kernels(kernel_path)
    records = supervisor["records"]
    _need(len(records) == 20, "Four warm queries plus16 controller episodes required")
    manifests = [(entry, inputs.json(entry["manifest"])) for entry in records]
    _need(
        all(
            entry["exit_code"] == 0
            and value["status"] == "COMPLETED"
            and value["actor_hash"] == config["actor_hash"]
            and value["normalization_fingerprint"] == norm_hash
            for entry, value in manifests
        ),
        "Incomplete result, actor or normalization binding",
    )
    warm = {entry["index"]: value for entry, value in manifests if entry["kind"] == "prepare_warm"}
    _need(set(warm) == set(range(4)), "Fixed four warm preparations required")
    episodes, traces, banks = [], {}, {}
    bank_rows = []
    for entry, manifest in manifests:
        if entry["kind"] != "controller":
            continue
        index, mode, coefficient = entry["index"], entry["mode"], float(entry["alpha"])
        _need(
            index in range(4)
            and mode in ("zero", "shadow_carried")
            and coefficient in (0.0, alpha),
            "Unexpected controller condition",
        )
        key = (index, mode, coefficient)
        _need(key not in traces, "Duplicate controller condition")
        _need(
            manifest["alpha"] == coefficient
            and manifest["initial_plan_mode"] == mode
            and manifest["checkpoint"] == config["checkpoint"],
            "Controller manifest condition mismatch",
        )
        raw = inputs.trace(manifest["trace"])
        outcome = episode_outcome(raw, terminal_weight=config["task"]["terminal_unhealthy_weight"])
        for field in (
            "controls",
            "fell",
            "full_survival",
            "local_walking",
            "complete_cost",
            "raw_native_cost",
            "absorbing_tail_cost",
            "velocity_mae",
        ):
            _need(
                np.isclose(outcome[field], manifest["outcome"][field], rtol=0, atol=1e-9),
                "Saved outcome differs: " + field,
            )
        context = inputs.npz(config["contexts"][index]["context"])
        for field, array in (
            ("state", "states"),
            ("warmstart", "warmstarts"),
            ("integration", "integrations"),
        ):
            _need(
                _same(context[field], raw[array][0]), "Controller initial complete context differs"
            )
        initial = inputs.npz(manifest["executed_initial_plan"])["U"]
        _need(
            initial.shape == (48, 6)
            and initial.dtype == np.float64
            and manifest["initial_two_passes"] is True,
            "Same two-pass H48 initialization required",
        )
        _need(
            fingerprint(initial) == manifest["initial_plan_fingerprint"],
            "Initial nominal fingerprint differs",
        )
        if mode == "zero":
            _need(np.all(initial == 0), "Zero initialization used a warm nominal")
        else:
            _need(
                warm[index]["shadow_plan_shift_count"] == 1
                and warm[index]["recommendation_was_executed"] is False,
                "Shadow nominal provenance differs",
            )
            _need(
                _same(initial, inputs.npz(warm[index]["initial_plan"])["U"]),
                "Shadow initial nominal differs",
            )
        row = dict(index=index, mode=mode, alpha=coefficient, **outcome)
        episodes.append(dict(row, key=key))
        traces[key] = raw
        groups = []
        with h5py.File(inputs.bound(manifest["bank_stats"]), "r") as archive:
            _need(
                len(archive) == len(raw["actions"]) + 1 == manifest["banks"],
                "Two initial passes plus one per later control required",
            )
            for bank_id, (name, group) in enumerate(archive.items()):
                expected_control, expected_inner = (0, bank_id) if bank_id < 2 else (bank_id - 1, 0)
                _need(
                    name == f"{bank_id:06d}"
                    and int(group.attrs["control"]) == expected_control
                    and int(group.attrs["inner"]) == expected_inner,
                    "Incorrect bank/pre-action indexing",
                )
                _need(
                    group.attrs["actor_hash"] == config["actor_hash"]
                    and float(group.attrs["alpha"]) == coefficient
                    and float(group.attrs["lambda_value"]) == 3.0,
                    "Candidate scoring actor/alpha/temperature differs",
                )
                weights, survival, lengths = (
                    group["weights"][:],
                    group["full_survival"][:],
                    group["N"][:],
                )
                _need(
                    weights.shape == survival.shape == lengths.shape == (256,)
                    and np.isfinite(weights).all()
                    and np.all(weights >= 0)
                    and np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0),
                    "Invalid candidate weight population",
                )
                c, r = group["C"][:], group["R"][:]
                valid = group["valid"][:]
                finite = np.flatnonzero(np.isfinite(c) & np.isfinite(r))
                task_finite = np.flatnonzero(np.isfinite(c))
                competitive = task_finite[np.argsort(c[task_finite], kind="stable")[:32]]
                bank_rows.append(
                    dict(
                        index=index,
                        mode=mode,
                        alpha=coefficient,
                        bank=bank_id,
                        control=expected_control,
                        inner=expected_inner,
                        survival_fraction=float(survival.mean()),
                        fallen_weight=float(weights[~survival].sum()),
                        censored_weight=float(weights[lengths < 48].sum()),
                        last_control_fall_weight=float(
                            weights[(~survival) & (lengths == 48)].sum()
                        ),
                        invalid_weight=float(weights[~valid].sum()),
                        ess=float(1 / np.sum(weights**2)),
                        zero_length_candidates=int((lengths == 0).sum()),
                        zero_R_valid_candidates=int((valid & (r == 0)).sum()),
                        finite_C_candidates=int(np.isfinite(c).sum()),
                        realized_all_finite_spread_ratio=_spread_ratio(c, r, finite, coefficient),
                        realized_task_top32_spread_ratio=_spread_ratio(
                            c, r, competitive, coefficient
                        ),
                        temperature=3.0,
                        innovation_fingerprint=str(group.attrs["innovation_fingerprint"]),
                    )
                )
                groups.append(
                    dict(
                        name=name,
                        innovation=str(group.attrs["innovation_fingerprint"]),
                        rng_before=str(group.attrs["rng_before"]),
                        rng_after=str(group.attrs["rng_after"]),
                    )
                )
        banks[key] = dict(groups=groups, manifest=manifest)
        _need(time.process_time() - cpu <= cpu_seconds, "Saved-only report CPU budget exhausted")
    _need(len(episodes) == 16 and len(traces) == 16, "Complete16-condition roster required")
    pairs, innovation_rows, cloud_rows = [], [], []
    for case in range(4):
        roster = sorted(
            [row for row in episodes if row["index"] == case],
            key=lambda row: (row["mode"], row["alpha"]),
        )
        _need(
            len(roster) == 4 and len({row["key"] for row in roster}) == 4,
            "Four conditions per source restart required",
        )
        common_banks = min(len(banks[row["key"]]["groups"]) for row in roster)
        for bank in range(common_banks):
            proof = [banks[row["key"]]["groups"][bank] for row in roster]
            _need(
                all(value == proof[0] for value in proof),
                "Paired innovation stream differs across conditions",
            )
        innovation_rows.append(
            dict(
                index=case,
                compared_conditions=4,
                common_banks=common_banks,
                matching_innovation_and_RNG_records=True,
                compared_only_common_bank_prefix=True,
            )
        )
        for left, right in itertools.combinations(roster, 2):
            pairs.append(pair_difference(left, right, traces[left["key"]], traces[right["key"]]))
        for mode in ("zero", "shadow_carried"):
            source = [row for row in roster if row["mode"] == mode]
            raw_banks = []
            for row in source:
                manifest = banks[row["key"]]["manifest"]
                full = next(
                    spec
                    for spec in manifest["full_banks"]
                    if Path(spec["path"]).name == "full_t000_p0.npz"
                )
                raw_banks.append(inputs.npz(full))
            checks = {
                name: _same(raw_banks[0][name], raw_banks[1][name])
                for name in ("actions", "posts", "C", "U_before", "innovation")
            }
            _need(
                all(checks.values()),
                "Same-mode first candidate cloud differs before alpha weighting",
            )
            cloud_rows.append(
                dict(
                    index=case,
                    mode=mode,
                    bank="000000",
                    **checks,
                    scope="First pass only; subsequent plans/clouds may differ despite paired innovations",
                )
            )
    spectral, neighbors, spectral_arrays = supporting_diagnostics(
        episodes, traces, mean.astype(float), std.astype(float), kernels
    )
    decision = frozen_decision(episodes, pairs, config["decision"]["action_rms_threshold"])
    _need(time.process_time() - cpu <= cpu_seconds, "Saved-only report CPU budget exhausted")
    out.mkdir(parents=True, exist_ok=False)
    public_episodes = [
        {key: value for key, value in row.items() if key != "key"} for row in episodes
    ]
    for name, rows in (
        ("episodes.csv", public_episodes),
        ("pair_differences.csv", pairs),
        ("candidate_banks.csv", bank_rows),
        ("innovation_pairing.csv", innovation_rows),
        ("first_cloud_identity.csv", cloud_rows),
        ("spectra.csv", spectral),
        ("neighbour_r2.csv", neighbors),
    ):
        _csv(out / name, rows)
    with (out / "spectral_arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **spectral_arrays)
    result = dict(
        schema="coupling-controller-saved-report-1",
        status="COMPLETED",
        decision=decision,
        episodes=public_episodes,
        pairs=pairs,
        innovation_pairing=innovation_rows,
        first_cloud_identity=cloud_rows,
        supporting_diagnostics=dict(
            source=str(kernel_path),
            extracted_functions=["spectrum", "lowpass"],
            adaptation="Separate episodes; burn0; Welch nperseg=min(300,N), noverlap=nperseg//2; min32 controls; fixed5Hz fourth-order zero-phase Butterworth. High band=actions-lowpass(actions). Teacher-policy residual variance separately labelled. No pooled-state renormalization.",
            neighbour="10 finite neighbors; distance-exclusion100 and whole-restart-trajectory-heldout controls; per-held-episode R2; negative and missing retained.",
            comparison="Different from original K1:1000 controls,burn100,Welch450/225,pooled source normalization. These are supporting explanations, not comparable thresholds or admission gates.",
        ),
        saved_supervisor_accounting=supervisor,
        limitations=[
            "Four current-student restart contexts, not whole-task standalone policy evaluations or learning evidence.",
            "Shadow preparation paid two preceding-state MPPI forecasts. Its recommendation was not executed: the actual student action connects that predecessor to the target context. This is a shadow nominal, not a teacher-owned MPC history.",
            "T1 selected its coefficient from three states of the first source student episode. T0 case0 at control100 overlaps that calibration episode/state; cases1..3 were outside the T1 bank roster. This is not held-out confirmation.",
            "Realized candidate spread ratios, survival mass, spectra and neighbor R2 are descriptive; none changes alpha, admits demonstrations or replaces executed outcomes.",
            "The report performs saved-only consistency checks, not closure or rerun of the failed L3 independent audit.",
        ],
        source_bindings=list(inputs.bindings.values()),
        work=dict(
            cpu_seconds=time.process_time() - cpu,
            wall_seconds=time.monotonic() - started,
            cpu_limit_seconds=cpu_seconds,
            physics=0,
            neural_forwards=0,
            policy_updates=0,
            audit_reruns=0,
        ),
    )
    table = "\n".join(
        f"| {row['index']} | {row['mode']} | {row['alpha']:.8g} | {row['controls']} | {int(row['local_walking'])} / {int(row['full_survival'])} | {row['complete_cost']:.6f} | {row['mean_teacher_policy_action_mse']:.6f} |"
        for row in public_episodes
    )
    with (out / "report.md").open("x") as stream:
        stream.write(
            "# T0: executed local controller contrast\n\n"
            f"**Frozen decision: {decision['verdict']}.** All16 planned controller episodes and four shadow preparations completed. "
            f"Effective alpha intervention: {decision['effective']}; declared competence regression: {decision['destabilizing']}.\n\n"
            "| Restart | Initialization | Alpha | Controls | Local walk / survive | Complete cost | Executed policy-gap MSE |\n"
            "|---|---|---:|---:|---|---:|---:|\n" + table + "\n\n"
            "Outcomes cover300 controls, not1,000-control standalone policy walking. Native fall terminal is already included once in native cost; an actual fall adds five per missing control. All24 condition-pair contrasts use their real common action prefixes, without padded tails.\n\n"
            "Paired innovation fingerprints and RNG records match across the common bank prefix of all four conditions at each restart. Within each initialization, alpha0 and positive alpha share byte-identical first-pass sampled actions, states and native costs. Later nominal updates and candidate clouds may differ despite paired innovations. See `pair_differences.csv`, `innovation_pairing.csv` and `first_cloud_identity.csv`.\n\n"
            "Shadow nominals required paid preceding-state MPPI queries; the actual student action, rather than the teacher recommendation, connects to the target. Zero/shadow and alpha effects are separate contrasts; no task benefit is assumed. Case0 overlaps the T1 calibration episode/state. Cases1..3 were outside its bank roster; this is not held-out confirmation.\n\n"
            "`candidate_banks.csv` preserves survival/fall/censoring mass, ESS and realized all-finite/top32 spread ratios. `spectra.csv` and `neighbour_r2.csv` are supporting episode-respecting diagnostics using fixed W2 normalization,5Hz bands and whole-restart-held-out controls, with negative/missing results retained. Their shortened Welch windows and burn0 differ from K1. They neither tune alpha nor admit data.\n\n"
            f"{decision['following']} No fitting, independent-audit closure, convergence or superiority is established here.\n"
        )
    write(out / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--kernel-source", type=Path)
    parser.add_argument("--cpu-seconds", type=float, default=160.0)
    args = parser.parse_args(argv)
    render(
        args.base,
        args.out,
        normalization=args.normalization,
        kernel_source=args.kernel_source,
        cpu_seconds=args.cpu_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
