"""Frozen saved-array spectrum/conflict review; no simulator or policy execution."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

# Reserve five CPU seconds for the shell/read-only preparation commands. The
# adapter's hard CPU limit includes imports and all native numerical kernels.
resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
started = time.monotonic()
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent
bindings = {}


def fingerprint_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([list(array.shape), str(array.dtype)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve()
    if str(path) not in bindings:
        data = path.read_bytes()
        bindings[str(path)] = dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    row = bindings[str(path)]
    if expected is not None:
        assert all(row[key] == expected[key] for key in ("bytes", "sha256")), str(path)
    return path


def bound(spec):
    return bind(spec["path"], spec)


def read_json(path):
    return json.loads(bind(path).read_text())


def read_bound(spec):
    return read_json(bound(spec))


def arrays(spec):
    with np.load(bound(spec), allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def kernels(path, names, constants):
    """Compile exact archived function bodies; never import their CLI/actors."""
    tree = ast.parse(bind(path).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    scope = dict(np=np, signal=signal, **constants)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope


def safe_scalar(value):
    value = float(value)
    return value if np.isfinite(value) else None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def guarded_neighbours(original, states, actions, i, pool):
    result = original(states, actions, i, pool)
    query_steps = np.arange(len(states[i]))
    eligible = np.zeros(len(query_steps), dtype=np.int64)
    for j in pool:
        if i == j:
            eligible += (np.abs(query_steps[:, None] - np.arange(len(states[j]))[None, :]) >= 100).sum(1)
        else:
            eligible += len(states[j])
    missing = eligible < 10
    # Original nearest-1 disagreement remains finite here. Its K-mean is not
    # meaningful where there are fewer than K eligible neighbours.
    if missing.any():
        result["mean_action"][missing] = np.nan
    result["missing_k_mean"] = int(missing.sum())
    assert np.isfinite(result["distance"]).all()
    return result


def measure(obs, action, mu, sd, burn):
    segment = 300 - burn
    constants = dict(FS=125.0, FIRST=burn, SEPARATION=100, K=10, SEGMENT=segment)
    spectrum = kernels(ROOT / "action_spectrum.original.py", ("spectrum", "lowpass", "neighbour_index", "state_dependence"), constants)
    conflict = kernels(ROOT / "label_conflict.original.py", ("squared_distances", "neighbours", "pairs", "matched", "flat", "averaging"), dict(SEPARATION=100, K=10))
    original_neighbours = conflict["neighbours"]
    conflict["neighbours"] = lambda states, actions, i, pool: guarded_neighbours(original_neighbours, states, actions, i, pool)
    observations = [(value[burn:] - mu) / sd for value in obs]
    actions = [value[burn:] for value in action]
    f, power = spectrum["spectrum"](action)
    all_power = float(power.sum())
    above = float(power[f > 5.0].sum())
    full = np.concatenate(actions)
    low = np.concatenate(spectrum["lowpass"](action, 5.0))
    high = full - low
    output = dict(
        burn_controls=burn, rows=len(full), episodes=8,
        welch_nperseg=segment, welch_noverlap=segment // 2,
        frequency_resolution_hz=125.0 / segment,
        above_5hz_welch_power_share=safe_scalar(above / all_power),
        high_residual_variance_share=safe_scalar(high.var(0).sum() / full.var(0).sum()),
        low_variance_share=safe_scalar(low.var(0).sum() / full.var(0).sum()),
        neighbour={},
        spectrum=[dict(frequency_hz=float(x), mean_torque_psd=float(y)) for x, y in zip(f, power, strict=True)],
    )
    normalized_flat = np.concatenate(observations)
    walk_ids = np.repeat(np.arange(8), segment)
    steps = np.tile(np.arange(segment), 8)
    for label, exclusion in (("gap100", 100), ("episode_held_out", 1_000_000)):
        spectrum["SEPARATION"] = exclusion
        index = spectrum["neighbour_index"](observations)
        forbidden = (walk_ids[:, None] == walk_ids[index]) & (np.abs(steps[:, None] - steps[index]) < exclusion)
        assert not forbidden.any(), "Forbidden temporal/source neighbour"
        distance = np.sqrt(np.maximum(((normalized_flat[:, None, :] - normalized_flat[index]) ** 2).sum(-1), 0))
        bands = {}
        for band, target in (("full", full), ("low_5hz", low), ("high_5hz", high)):
            result = spectrum["state_dependence"](target, index, np.random.default_rng(0))
            bands[band] = {key: safe_scalar(value) for key, value in result.items()}
        output["neighbour"][label] = dict(
            **bands, median_nearest_distance=float(np.median(distance[:, 0])),
            max_nearest_distance=float(distance[:, 0].max()),
            missing_neighbours=0,
        )
    result = conflict["pairs"](observations, actions)
    same = conflict["flat"](result, "same")
    other = conflict["flat"](result, "other")
    edges = np.quantile(np.concatenate([same[0], other[0]]), np.linspace(0, 1, 11))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    comparison = conflict["matched"](edges, same, other)
    mean_ratio = comparison["mean_ratio"]
    averaging_missing = sum(row["missing_k_mean"] for row in result["same"])
    if averaging_missing:
        # Preserve missingness; no forbidden neighbour enters a numeric result.
        averaging = dict(same_walk=None, reason="Fewer than K finite same-episode neighbours", missing_query_rows=averaging_missing)
    else:
        averaging = conflict["averaging"](result, actions)
    output["conflict"] = dict(
        other_over_same_equal_distance_bin_mean=safe_scalar(mean_ratio) if mean_ratio is not None else None,
        bins=comparison["bins"],
        bin_rule="Original ten within-cell pooled distance quantiles; descriptive, not exact matching or fixed-support cross-arm comparison",
        same_nearest_distance_median=float(np.median(same[0])),
        other_nearest_distance_median=float(np.median(other[0])),
        averaging=averaging,
        no_validated_gait_conflict_threshold=True,
    )
    return output


def diagnostic_mse(fit, phase, label):
    row = fit["diagnostics_" + phase][label]
    data, saved = arrays(row["data"]), arrays(row["predictions"])
    target = data["actions"].astype(np.float64)
    predictions = saved["predictions"].astype(np.float64)
    assert target.shape == predictions.shape
    mse = float(np.mean((target - predictions) ** 2))
    assert abs(mse - row["action_mse"]) <= 1e-10
    return dict(action_mse=mse, saved_action_mse=row["action_mse"], rows=len(target), new_inference=False)


def main():
    config = read_json(ROOT / "analysis_config.json")
    review = read_bound(config["review_config"])
    snapshots = [(entry, read_bound(entry["snapshot"])) for entry in review["snapshot_roster"]]
    envelope_spec = snapshots[0][1]["initial_streams"][0]["envelope"]
    envelope = read_bound(envelope_spec)
    checkpoint_spec = dict(envelope["files"]["policy.pt"], path=str(bound(envelope_spec).parent / "policy.pt"))
    checkpoint = bound(checkpoint_spec)
    import torch

    # Deserialization alone: no Policy construction and no network forwards.
    torch.set_num_threads(1)
    def no_forward(*args, **kwargs):
        raise RuntimeError("No neural forward is permitted in this saved-array review")
    torch.nn.Module._call_impl = no_forward
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    mu = state["observation_mean"].numpy().astype(np.float64)
    sd = state["observation_std"].numpy().astype(np.float64)
    assert mu.shape == sd.shape == (17,) and np.all(sd > 0)
    np.savez_compressed(ROOT / "fixed_normalization.npz", mean=mu, std=sd)
    cache, rows, metrics = {}, [], {}
    for roster, snapshot in snapshots:
        stream = roster["stream"]
        complete = [block for block in snapshot["rounds"] if block["status"] == "COMPLETED"]
        assert [block["round"] for block in complete] == roster["completed_rounds"]
        for block in complete:
            corpus = read_bound(block["corpus"])
            assert corpus["rows"] == 2400
            for arm, key, trace_index in (("independent", "a", 0), ("positive", "b", 1)):
                data = arrays(corpus["files"][key])
                observations, actions = [], []
                source_episodes = []
                for slot in range(8):
                    selected = data["row_ids"][:, 0] == slot
                    assert np.array_equal(data["row_ids"][selected, 1], np.arange(300))
                    observations.append(data["observations"][selected].astype(np.float64))
                    actions.append(data["actions"][selected].astype(np.float64))
                    source_episodes.append(corpus["cases"][slot]["traces"][trace_index])
                assert len({source["sha256"] for source in source_episodes}) == 8
                digest = fingerprint_arrays(data["observations"], data["actions"], data["row_ids"])
                reused = digest in cache
                if not reused:
                    cache[digest] = dict(first_stream=stream, first_round=block["round"], first_arm=arm)
                    metrics[digest] = dict(
                        content_hash=digest, first_occurrence=cache[digest],
                        primary=measure(observations, actions, mu, sd, burn=0),
                        burn100=measure(observations, actions, mu, sd, burn=100),
                    )
                    write_json(ROOT / f"corpus_{digest}.json", metrics[digest])
                fit_spec = next(row["result"] for row in block["fits"] if row["arm"] == arm)
                fit = read_bound(fit_spec)
                saved = {phase: {label: diagnostic_mse(fit, phase, label) for label in ("own", "held", "anchor")} for phase in ("before", "after")}
                row = dict(stream=stream, round=block["round"], arm=arm, corpus_content_hash=digest, reused_saved_array_metrics=reused, first_occurrence=cache[digest], source_episodes=source_episodes, fit=fit_spec, reconstruction=saved)
                rows.append(row)
                write_json(ROOT / "progress.json", dict(completed_cells=len(rows), unique_corpora=len(cache), last_cell=dict(stream=stream, round=block["round"], arm=arm), cpu_seconds=time.process_time(), wall_seconds=time.monotonic() - started))
    write_json(ROOT / "raw_metrics.json", dict(cells=rows, corpora=metrics))
    flat = []
    for cell in rows:
        for window in ("primary", "burn100"):
            metric = metrics[cell["corpus_content_hash"]][window]
            flat.append(dict(
                stream=cell["stream"], round=cell["round"], arm=cell["arm"], window=window,
                corpus_content_hash=cell["corpus_content_hash"], reused_saved_array_metrics=cell["reused_saved_array_metrics"],
                above5_power_share=metric["above_5hz_welch_power_share"],
                high_variance_share=metric["high_residual_variance_share"],
                r2_full_gap100=metric["neighbour"]["gap100"]["full"]["r2"],
                r2_low_gap100=metric["neighbour"]["gap100"]["low_5hz"]["r2"],
                r2_high_gap100=metric["neighbour"]["gap100"]["high_5hz"]["r2"],
                r2_full_episode_held_out=metric["neighbour"]["episode_held_out"]["full"]["r2"],
                r2_low_episode_held_out=metric["neighbour"]["episode_held_out"]["low_5hz"]["r2"],
                r2_high_episode_held_out=metric["neighbour"]["episode_held_out"]["high_5hz"]["r2"],
                conflict_other_over_same=metric["conflict"]["other_over_same_equal_distance_bin_mean"],
                mean_nearest_distance_episode_held_out=metric["neighbour"]["episode_held_out"]["median_nearest_distance"],
                same_episode_K_mean_missing=metric["conflict"]["averaging"].get("missing_query_rows", 0),
                own_mse_before=cell["reconstruction"]["before"]["own"]["action_mse"],
                own_mse_after=cell["reconstruction"]["after"]["own"]["action_mse"],
                held_mse_before=cell["reconstruction"]["before"]["held"]["action_mse"],
                held_mse_after=cell["reconstruction"]["after"]["held"]["action_mse"],
                anchor_mse_after=cell["reconstruction"]["after"]["anchor"]["action_mse"],
            ))
    with (ROOT / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return dict(status="COMPLETED", cells=len(rows), unique_corpora=len(cache), cached_references=len(rows) - len(cache), windows=["primary", "burn100"], normalization_checkpoint=checkpoint_spec, normalization_fingerprint=fingerprint_arrays(mu, sd))


try:
    result = main()
except BaseException as error:
    result = dict(status="FAILED_STOPPED", error=repr(error), traceback=traceback.format_exc())
write_json(ROOT / "file_bindings.json", bindings)
result.update(
    cpu_seconds=time.process_time(), command_cpu_reserve_seconds=5,
    cumulative_cpu_upper_seconds=time.process_time() + 5,
    wall_seconds=time.monotonic() - started,
    physics_substeps=0, model_queries=0, native_controls=0,
    neural_forward_rows=0, policy_updates=0,
    source=bindings[str(bind(__file__))], config=bindings[str(bind(ROOT / "analysis_config.json"))],
    rlimit_cpu_seconds=45, aggregate_subcap_cpu_seconds=50,
)
write_json(ROOT / "receipt.json", result)
print(json.dumps(result, indent=2))
sys.exit(result["status"] != "COMPLETED")
