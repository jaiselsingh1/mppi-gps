"""Array-only outcome, pairing and episode-boundary reporting fixtures."""

from __future__ import annotations

import ast
import json

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.provenance import fingerprint, write
from src.gps.mppi_gps.reporting import coupling_controller_check as report


def trace(n=2, *, fell=True, action=0.02):
    states = np.zeros((n + 1, 19))
    states[:, 0] = np.arange(n + 1) * 0.008
    states[:, 1] = np.arange(n + 1) * action
    states[:, 2], states[:, 10] = 1.2, 1.5
    if fell:
        states[-1, 2] = 0.7
    fall = np.zeros(n)
    fall[-1] = 20 if fell else 0
    return dict(
        states=states,
        warmstarts=np.zeros((n + 1, 9)),
        integrations=np.zeros((n + 1, 91)),
        observations=np.concatenate((states[:, 2:10], states[:, 10:19]), axis=1),
        actions=np.full((n, 6), action),
        policy_actions=np.zeros((n, 6)),
        costs=np.ones(n) + fall,
        native_fall_cost=fall,
        action_gap_mse=np.full(n, action**2),
    )


def test_native_terminal_is_not_added_twice_and_fall_tail_is_explicit():
    value = report.episode_outcome(trace())
    assert value["raw_native_cost"] == 22
    assert value["native_fall_terminal"] == 20
    assert value["absorbing_tail_cost"] == 1490
    assert value["complete_cost"] == 1512
    assert not value["full_survival"] and not value["local_walking"]
    last = report.episode_outcome(trace(300))
    assert last["fell"] and last["absorbing_tail_cost"] == 0
    assert not last["full_survival"]
    healthy = report.episode_outcome(trace(300, fell=False))
    assert healthy["full_survival"] and healthy["local_walking"]


def test_cost_and_pre_action_contracts_reject_fabrication():
    raw = trace()
    raw["native_fall_cost"][0] = 20
    with pytest.raises(ValueError, match="exactly once"):
        report.episode_outcome(raw)
    raw = trace()
    raw["observations"][0, 0] += 0.01
    with pytest.raises(ValueError, match="pre-action"):
        report.episode_outcome(raw)


def test_common_prefix_excludes_missing_tails_and_checks_state_divergence():
    left, right = trace(2, action=0), trace(5, action=0.02)
    a, b = dict(index=0, mode="zero", alpha=0), dict(index=0, mode="zero", alpha=3.0)
    value = report.pair_difference(a, b, left, right)
    assert value["common_controls"] == 2
    assert value["first_action_rms"] == pytest.approx(0.02)
    assert value["common_prefix_action_rms"] == pytest.approx(0.02)
    assert value["first_divergent_boundary_gt1e_10"] == 1
    assert value["states_diverge_beyond_initial"] and value["no_padded_tails"]


def test_frozen_decision_uses_all_eight_alpha_pairs_and_actual_regressions():
    episodes = [
        dict(
            index=i,
            mode=m,
            alpha=a,
            fell=False,
            full_survival=True,
            local_walking=True,
            controls=300,
        )
        for i in range(4)
        for m in ("zero", "shadow_carried")
        for a in (0.0, 3.0)
    ]
    pairs = [
        dict(contrast="alpha_same_initialization", common_prefix_action_rms=0.01 if i == 7 else 0)
        for i in range(8)
    ]
    assert report.frozen_decision(episodes, pairs)["effective"]
    assert not report.frozen_decision(episodes, pairs)["destabilizing"]
    episodes[-1].update(fell=True, full_survival=False, local_walking=False, controls=200)
    result = report.frozen_decision(episodes, pairs)
    assert result["destabilizing"] and result["verdict"] == "destabilizing"
    with pytest.raises(ValueError, match="Complete frozen roster"):
        report.frozen_decision(episodes[:-1], pairs)


def test_whole_restart_holdout_retains_negative_r2_and_missing_neighbors():
    x = np.arange(40, dtype=float)[:, None]
    y = np.arange(40, dtype=float)[:, None]
    rows = report.neighbour_bands([x, x], dict(low=[y, y + 100]), whole_episode=True)
    assert len(rows) == 2 and all(row["finite_neighbor_queries"] == 40 for row in rows)
    assert all(row["r2"] < 0 for row in rows)
    missing = report.neighbour_bands([x[:3], x[:3]], dict(high=[y[:3], y[:3]]), whole_episode=True)
    assert all(row["r2"] is None and row["missing_neighbor_queries"] == 3 for row in missing)


def test_ast_spectral_loading_never_executes_original_imports(tmp_path):
    tree = ast.parse(report.KERNEL_SOURCE.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"spectrum", "lowpass"}
    ]
    source = tmp_path / "preserved.py"
    source.write_text(
        "raise AssertionError('SAC imports must never execute')\n"
        + "\n".join(ast.unparse(node) for node in functions)
    )
    kernels = report._kernels(source)
    action = np.sin(np.arange(40)[:, None] / 4) * np.ones((1, 6))
    kernels["SEGMENT"] = 40
    f, power = kernels["spectrum"]([action])
    assert len(f) == len(power) == 21
    assert kernels["lowpass"]([action], 5)[0].shape == (40, 6)


def test_short_episodes_are_missing_not_filtered_or_temporally_joined():
    episodes = [
        dict(index=i, mode=m, alpha=a, key=(i, m, a))
        for i in range(4)
        for m in ("zero", "shadow_carried")
        for a in (0.0, 3.0)
    ]
    traces = {row["key"]: trace(20) for row in episodes}
    spectra, neighbors, arrays = report.supporting_diagnostics(
        episodes, traces, np.zeros(17), np.ones(17), {}
    )
    assert len(spectra) == 16 and len(neighbors) == 96 and not arrays
    assert all(row["status"] == "MISSING_PREFIX_SHORTER_THAN32" for row in spectra)
    assert all(row["r2"] is None and row["missing_neighbor_queries"] == 20 for row in neighbors)


def test_realized_spread_is_descriptive_and_degenerate_is_missing():
    assert report._spread_ratio(np.arange(4.0), np.arange(4.0) * 2, np.arange(4), 3) == 6
    assert report._spread_ratio(np.ones(4), np.arange(4.0), np.arange(4), 3) is None


def _spec(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=report.sha(path))


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, value)
    return _spec(path)


def _npz(path, **values):
    with path.open("xb") as stream:
        np.savez(stream, **values)
    return _spec(path)


def saved_lifecycle(tmp_path):
    """Sixteen tiny, closed synthetic array traces; no mechanics or actor."""
    base = tmp_path / "run"
    base.mkdir()
    checkpoint = _json(base / "checkpoint.json", dict(synthetic=True))
    norm_path = base / "norm.npz"
    _npz(norm_path, mean=np.zeros(17, dtype=np.float32), std=np.ones(17, dtype=np.float32))
    norm_hash = fingerprint(
        dict(mean=np.zeros(17, dtype=np.float32), std=np.ones(17, dtype=np.float32))
    )
    contexts = []
    warm = []
    for i in range(4):
        state = trace()["states"][0]
        context = _npz(
            base / f"context{i}.npz",
            state=state,
            warmstart=np.zeros(9),
            integration=np.zeros(91),
            spec=np.array(1),
            dimensions=np.array([9, 9, 6]),
        )
        contexts.append(dict(context=context))
        plan = _npz(base / f"warm{i}.npz", U=np.full((48, 6), 0.1))
        warm.append(
            dict(initial_plan=plan, shadow_plan_shift_count=1, recommendation_was_executed=False)
        )
    config = _json(
        base / "config.json",
        dict(
            actor_hash="fake",
            checkpoint=checkpoint,
            contexts=contexts,
            task=dict(terminal_unhealthy_weight=20),
            decision=dict(action_rms_threshold=0.01),
        ),
    )
    calibration = _json(
        base / "calibration.json",
        dict(actor_hash="fake", selected_alpha=3.0, alphas=dict(competitive={"0.3": 3.0})),
    )
    inventory = _json(base / "inventory.json", dict(actor=dict(actor_hash="fake")))
    launch = _json(
        base / "launch.json",
        dict(config=config, calibration=calibration, inventory=inventory, selected_alpha=3.0),
    )
    full = _npz(
        base / "full_t000_p0.npz",
        actions=np.zeros((2, 3, 6)),
        posts=np.zeros((2, 3, 19)),
        C=np.arange(2.0),
        U_before=np.zeros((48, 6)),
        innovation=np.ones((2, 3, 6)),
    )
    records = []
    common = dict(
        status="COMPLETED",
        actor_hash="fake",
        normalization_fingerprint=norm_hash,
        checkpoint=checkpoint,
    )
    for i in range(4):
        spec = _json(base / f"warm_manifest{i}.json", dict(common, **warm[i]))
        records.append(dict(kind="prepare_warm", index=i, exit_code=0, manifest=spec))
    for i in range(4):
        for mode in ("zero", "shadow_carried"):
            for alpha in (0.0, 3.0):
                name = f"case{i}_{mode}_{alpha}"
                raw = trace(
                    action=(0.1 if mode == "shadow_carried" else 0) + (0.02 if alpha else 0)
                )
                trace_path = base / (name + ".h5")
                with h5py.File(trace_path, "x") as archive:
                    for key, value in raw.items():
                        archive.create_dataset(key, data=value)
                banks_path = base / (name + "_banks.h5")
                with h5py.File(banks_path, "x") as archive:
                    for bank in range(3):
                        group = archive.create_group(f"{bank:06d}")
                        group.attrs.update(
                            control=0 if bank < 2 else 1,
                            inner=bank if bank < 2 else 0,
                            actor_hash="fake",
                            alpha=alpha,
                            lambda_value=3.0,
                            innovation_fingerprint=f"same_case{i}_bank{bank}",
                            rng_before=f"before{bank}",
                            rng_after=f"after{bank}",
                        )
                        for key, value in dict(
                            weights=np.ones(256) / 256,
                            full_survival=np.ones(256, dtype=bool),
                            N=np.full(256, 48),
                            valid=np.ones(256, dtype=bool),
                            C=np.arange(256.0),
                            R=np.arange(256.0) / 100,
                        ).items():
                            group.create_dataset(key, data=value)
                initial = np.full(
                    (48, 6), 0.1 if mode == "shadow_carried" else 0.0, dtype=np.float64
                )
                plan = _npz(base / (name + "_initial.npz"), U=initial)
                manifest = dict(
                    common,
                    alpha=alpha,
                    initial_plan_mode=mode,
                    outcome=report.episode_outcome(raw),
                    trace=_spec(trace_path),
                    bank_stats=_spec(banks_path),
                    banks=3,
                    full_banks=[full],
                    executed_initial_plan=plan,
                    initial_plan_fingerprint=fingerprint(initial),
                    initial_two_passes=True,
                )
                manifest_spec = _json(base / (name + ".json"), manifest)
                records.append(
                    dict(
                        kind="controller",
                        index=i,
                        mode=mode,
                        alpha=alpha,
                        exit_code=0,
                        manifest=manifest_spec,
                    )
                )
    _json(base / "t0/result.json", dict(status="COMPLETED", launch=launch, records=records))
    return base, norm_path


def test_complete_saved_only_lifecycle_writes24_unpadded_pairs(tmp_path):
    base, norm = saved_lifecycle(tmp_path)
    out = tmp_path / "report"
    result = report.render(base, out, normalization=norm)
    assert len(result["episodes"]) == 16 and len(result["pairs"]) == 24
    assert result["decision"]["effective"] and not result["decision"]["destabilizing"]
    assert len(result["first_cloud_identity"]) == 8 and len(result["innovation_pairing"]) == 4
    assert all(row["common_controls"] == 2 for row in result["pairs"])
    assert result["work"]["physics"] == result["work"]["neural_forwards"] == 0
    assert (out / "candidate_banks.csv").is_file()
    with pytest.raises(FileExistsError):
        report.render(base, out, normalization=norm)


def test_running_supervisor_cannot_render_or_create_outputs(tmp_path):
    base = tmp_path / "run"
    _json(base / "t0/result.json", dict(status="RUNNING"))
    out = tmp_path / "report"
    with pytest.raises(ValueError, match="finish"):
        report.render(base, out)
    assert not out.exists()


def test_changed_innovation_is_not_misreported_as_paired(tmp_path):
    base, norm = saved_lifecycle(tmp_path)
    path = base / "case0_zero_3.0_banks.h5"
    with h5py.File(path, "r+") as archive:
        archive["000001"].attrs["innovation_fingerprint"] = "different"
    manifest_path = base / "case0_zero_3.0.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bank_stats"] = _spec(path)
    manifest_path.write_text(json.dumps(manifest))
    supervisor_path = base / "t0/result.json"
    supervisor = json.loads(supervisor_path.read_text())
    next(
        row
        for row in supervisor["records"]
        if row.get("mode") == "zero" and row.get("index") == 0 and row.get("alpha") == 3.0
    )["manifest"] = _spec(manifest_path)
    supervisor_path.write_text(json.dumps(supervisor))
    with pytest.raises(ValueError, match="innovation stream"):
        report.render(base, tmp_path / "report", normalization=norm)
