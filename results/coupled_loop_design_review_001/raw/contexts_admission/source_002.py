"""Saved-only restart/admission/support review; creates a new result directory.

No project execution module, Torch, Policy, planner or simulator is imported.
Exact-row novelty is reported both for stored FP64 corpora and actual FP32
learner inputs/targets. Support distance is standardized Euclidean RMS over the
fixed 17 clipped observations, using the unchanged original W2 mean/std.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

CPU_STARTED, WALL_STARTED = time.process_time(), time.monotonic()
resource.setrlimit(resource.RLIMIT_CPU, (350, 360))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Reader:
    def __init__(self):
        self.bindings = {}
        self.json_cache = {}
        self.data_cache = {}
        self.trace_cache = {}

    def bound(self, spec):
        path = Path(spec["path"]).resolve()
        if str(path) not in self.bindings:
            self.bindings[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
        if self.bindings[str(path)] != spec:
            raise ValueError("Changed bound input: " + str(path))
        return path

    def capture(self, path):
        path = Path(path).resolve()
        if str(path) not in self.bindings:
            self.bindings[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
        return self.bindings[str(path)]

    def json(self, spec):
        path = self.bound(spec)
        if str(path) not in self.json_cache:
            self.json_cache[str(path)] = json.loads(path.read_text())
        return self.json_cache[str(path)]

    def data(self, spec):
        self.bound(spec)
        key = spec["sha256"]
        if key not in self.data_cache:
            with np.load(spec["path"], allow_pickle=False) as raw:
                self.data_cache[key] = {name: raw[name].copy() for name in raw.files}
        return self.data_cache[key]

    def trace(self, spec):
        self.bound(spec)
        key = spec["sha256"]
        if key not in self.trace_cache:
            with h5py.File(spec["path"], "r") as raw:
                self.trace_cache[key] = {name: raw[name][:] for name in raw}
        return self.trace_cache[key]


def observations(states):
    return np.concatenate((states[..., 2:10], np.clip(states[..., 10:19], -10, 10)), axis=-1)


def exact(array):
    array = np.asarray(array)
    header = json.dumps([str(array.dtype), list(array.shape[1:])]).encode()
    return [header + np.ascontiguousarray(row).tobytes() for row in array]


def row_keys(data, cast=None):
    obs = np.asarray(data["observations"], dtype=cast)
    actions = np.asarray(data["actions"], dtype=cast)
    return set(exact(obs)), set(a + b for a, b in zip(exact(obs), exact(actions)))


def array_digest(data):
    digest = hashlib.sha256()
    for name in ("observations", "actions"):
        array = np.ascontiguousarray(data[name])
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(values):
    values = np.asarray(values)
    return dict(n=len(values), min=float(values.min()), median=float(np.median(values)), mean=float(values.mean()), p95=float(np.quantile(values, 0.95)), max=float(values.max()))


def main(config_path, out):
    out.mkdir(parents=True, exist_ok=False)
    reader = Reader()
    cfg = reader.json(reader.capture(config_path))
    reader.capture(Path(__file__))
    original = Path(cfg["original_run"])
    prepared = reader.json(reader.capture(original / "prepared_manifest.json"))
    science = reader.json(dict(path=str(original / "source/configs/studies/local_coupling_multiround_l3_0.json"), bytes=22664, sha256=prepared["config_sha256"]))
    l2path = Path(science["inputs"]["restarts"]["path"]).parents[1]
    l2cfgpath = l2path / "source/configs/studies/local_coupling_learning_l2_0.json"
    l2cfg = reader.json(reader.capture(l2cfgpath))

    # These contexts are inherited from the single source student rollout.
    restart_data = reader.data(prepared["restarts"])
    original_restarts = reader.data(l2cfg["inputs"]["restart_contexts"])
    source = reader.data(l2cfg["inputs"]["student_episode"])
    checkpoint = l2cfg["inputs"]["checkpoint"]
    reader.bound(checkpoint)
    p1stream = reader.json(l2cfg["inputs"]["incoming_stream_files"]["stream.json"])
    w2checkpoint = p1stream["source_checkpoint"]
    reader.bound(w2checkpoint)
    assert checkpoint["sha256"] == w2checkpoint["sha256"]
    w2root = Path(w2checkpoint["path"]).parents[1]
    w2inputs = reader.json(reader.capture(w2root / "input_manifest.json"))
    train_spec = dict(path=str(w2root / "inputs/train.npz"), bytes=(w2root / "inputs/train.npz").stat().st_size, sha256=w2inputs["files"]["inputs/train.npz"])
    norm_path = w2root / "inputs/normalization.npz"
    norm_spec = dict(path=str(norm_path), bytes=norm_path.stat().st_size, sha256=w2inputs["files"]["inputs/normalization.npz"])
    train = reader.data(train_spec)
    norm = reader.data(norm_spec)
    mean = norm.get("mean", norm.get("observation_mean"))
    std = norm.get("std", norm.get("observation_std"))
    assert mean is not None and std is not None and mean.shape == std.shape == (17,)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    assert train["observations"].shape == (4000, 17)
    np.savez(out / "normalization_used.npz", mean=mean, std=std)
    context_rows = []
    for slot, control in enumerate(l2cfg["inputs"]["restart_source_control_indices"]):
        checks = {}
        for singular, plural in (("state", "states"), ("warmstart", "warmstarts"), ("integration", "integrations")):
            old = original_restarts[f"r{slot}_{singular}"]
            current = restart_data[f"r{slot}_{singular}"]
            recorded = source[plural][control]
            checks[singular] = old.dtype == current.dtype == recorded.dtype and old.shape == current.shape == recorded.shape and old.tobytes() == current.tobytes() == recorded.tobytes()
            assert checks[singular], (slot, singular)
        context_rows.append(dict(slot=slot, source_control=control, source_time_seconds=control * 0.008, source_episode=l2cfg["inputs"]["student_episode"], source_checkpoint=checkpoint, original_w2_checkpoint=w2checkpoint, state_byte_exact=checks["state"], warmstart_byte_exact=checks["warmstart"], integration_byte_exact=checks["integration"], pre_action=True))
    write(out / "restart_provenance.json", dict(contexts=context_rows, independent_source_episodes=1, source_episode_controls=len(source["actions"]), source_episode_fell=bool(source["done"][-1]), context_selection="Early first-half states at fixed controls1,17,34,51,67,84,101,118 of one W2s0-u2000 rollout; not eight independent task starts."))

    admission_rows, novelty_rows, support_rows, support_summaries, latest_rows, pending = [], [], [], [], [], []
    dataset_rosters = []
    trace_pair_cache = {}

    def experience(spec):
        manifest = reader.json(spec)
        trace_spec = manifest.get("trace", reader.capture(Path(spec["path"]).parent / "trace.h5"))
        if trace_spec["sha256"] not in trace_pair_cache:
            raw = reader.trace(trace_spec)
            trace_pair_cache[trace_spec["sha256"]] = dict(observations=raw["observations"][:-1], actions=raw["actions"])
        return trace_pair_cache[trace_spec["sha256"]]

    for roster in cfg["snapshot_roster"]:
        stream = roster["stream"]
        snapshot = reader.json(roster["snapshot"])
        complete = [r for r in snapshot["rounds"] if r["round"] in roster["completed_rounds"] and r["status"] == "COMPLETED"]
        assert [r["round"] for r in complete] == roster["completed_rounds"]
        latest = complete[-1]
        pending.append(dict(stream=stream, latest_complete_round=latest["round"], uncompleted_planned_rounds=[r for r in range(1, snapshot["planned_rounds"] + 1) if r not in roster["completed_rounds"]], source_snapshot=roster["snapshot"]))
        for arm in ("independent", "positive"):
            initial_spec = next(entry["files"]["stream.json"] for entry in science["inputs"]["initial_streams"] if entry["stream"] == stream and entry["arm"] == arm)
            initial_env = reader.json(initial_spec)
            l2_fit_spec = reader.capture(Path(initial_spec["path"]).parents[1] / "result.json")
            l2_fit = reader.json(l2_fit_spec)
            assert l2_fit["arm"] == arm and l2_fit["stream"] == stream
            previous = reader.data(l2_fit["corpus"])
            ancestry = [("W2_initial_2000_updates", train_spec), ("L2_round1_2000_updates", l2_fit["corpus"])]
            sets = {}
            for mode, cast in (("stored_raw", None), ("learner_float32", np.float32)):
                po, pr = row_keys(previous, cast)
                wo, wr = row_keys(train, cast)
                sets[mode] = dict(previous_obs=po, previous_rows=pr, union_obs=wo | po, union_rows=wr | pr)
            for round_record in complete:
                round_number = round_record["round"]
                fit_spec = next(entry["result"] for entry in round_record["fits"] if entry["arm"] == arm)
                fit = reader.json(fit_spec)
                corpus = reader.data(fit["corpus"])
                assert corpus["observations"].shape == (2400, 17) and corpus["actions"].shape == (2400, 6)
                ancestry.append((f"L3_round{round_number}", fit["corpus"]))
                for mode, cast in (("stored_raw", None), ("learner_float32", np.float32)):
                    oo, rr = row_keys(corpus, cast)
                    old = sets[mode]
                    observation = np.asarray(corpus["observations"], dtype=cast)
                    target = np.asarray(corpus["actions"], dtype=cast)
                    previous_obs = np.asarray(previous["observations"], dtype=cast)
                    previous_targets = np.asarray(previous["actions"], dtype=cast)
                    novelty_rows.append(dict(stream=stream, arm=arm, round=round_number, representation=mode, rows=len(observation), unique_observation_rows=len(oo), unique_input_target_rows=len(rr), new_unique_observations_vs_previous=len(oo - old["previous_obs"]), new_unique_rows_vs_previous=len(rr - old["previous_rows"]), new_unique_observations_vs_ancestral_union=len(oo - old["union_obs"]), new_unique_rows_vs_ancestral_union=len(rr - old["union_rows"]), positional_observations_changed=int(np.count_nonzero(np.any(observation != previous_obs, axis=1))) if observation.shape == previous_obs.shape else None, positional_targets_changed=int(np.count_nonzero(np.any(target != previous_targets, axis=1))) if target.shape == previous_targets.shape else None, input_target_arrays_byte_identical=observation.dtype == previous_obs.dtype and target.dtype == previous_targets.dtype and observation.tobytes() == previous_obs.tobytes() and target.tobytes() == previous_targets.tobytes(), input_target_array_sha256=array_digest(dict(observations=observation, actions=target)), corpus_sha256=fit["corpus"]["sha256"]))
                    old.update(previous_obs=oo, previous_rows=rr, union_obs=old["union_obs"] | oo, union_rows=old["union_rows"] | rr)
                for admission in round_record["admissions"]:
                    if admission["arm"] != arm:
                        continue
                    before, chosen = experience(admission["previous"]), experience(admission["chosen"])
                    before_o, before_r = row_keys(before)
                    chosen_o, chosen_r = row_keys(chosen)
                    identical = array_digest(before) == array_digest(chosen)
                    classification = "fallback" if admission["fallback"] else ("admitted_identical" if identical else "admitted_changed")
                    admission_rows.append(dict(stream=stream, arm=arm, round=round_number, slot=admission["slot"], classification=classification, admitted=admission["admitted"], fallback=admission["fallback"], declared_new_experience=admission["new_experience"], declared_new_rows=admission["new_rows"], exact_arrays_identical_to_previous=identical, exact_new_unique_input_target_rows=len(chosen_r - before_r), exact_new_unique_observation_rows=len(chosen_o - before_o), previous_experience_fingerprint=admission["previous_experience_fingerprint"], chosen_experience_fingerprint=admission["chosen_experience_fingerprint"], chosen_manifest=admission["chosen"]["path"]))
                previous = corpus

            # Every actual ancestor's input corpus is represented, not held data
            # or an inferred mixture of acquisition attempts that were rejected.
            seen_hashes = set()
            ancestor_observations = []
            for label, spec in ancestry:
                if spec["sha256"] not in seen_hashes:
                    seen_hashes.add(spec["sha256"])
                    ancestor_observations.append(reader.data(spec)["observations"])
            union = np.unique(np.concatenate(ancestor_observations), axis=0)
            current = previous["observations"]
            dataset_rosters.append(dict(stream=stream, arm=arm, latest_complete_round=latest["round"], ancestry=[dict(role=label, data=spec) for label, spec in ancestry], current_rows=len(current), ancestral_input_rows_before_deduplication=sum(len(value) for value in ancestor_observations), ancestral_unique_observation_rows=len(union), excluded_held_rows=2000, original_normalization=norm_spec, incoming_l2_envelope=initial_spec, initial_l2_policy_hash=initial_env["policy_hash"]))
            queries, query_metadata = [], []
            for entry in latest["evaluations"]:
                if entry["arm"] != arm or entry["kind"] != "child":
                    continue
                result = reader.json(entry["result"])
                outcome = result["outcome"]
                if not outcome["fell"]:
                    latest_rows.append(dict(stream=stream, arm=arm, round=latest["round"], panel=entry["panel"], reset_seed=entry["reset_seed"], fell=False, controls=outcome["controls"], terminal_fall_support=None))
                    continue
                trace = reader.trace(result["trace"])
                n = len(trace["actions"])
                assert n == outcome["controls"] and len(trace["states"]) == n + 1
                assert np.array_equal(observations(trace["states"]), trace["observations"])
                for label, index in (("last_pre_action", n - 1), ("terminal_fall", n)):
                    queries.append(trace["observations"][index])
                    query_metadata.append(dict(stream=stream, arm=arm, round=latest["round"], panel=entry["panel"], reset_seed=entry["reset_seed"], kind=label, control_boundary=index, executed_controls=n, state_height=float(trace["states"][index, 2]), state_pitch=float(trace["states"][index, 3]), terminal_fall_height=float(trace["states"][-1, 2]), terminal_fall_pitch=float(trace["states"][-1, 3]), trace_sha256=result["trace"]["sha256"]))
            if queries:
                query = np.asarray(queries, dtype=np.float64)
                z = (query - np.asarray(mean, dtype=np.float64)) / np.asarray(std, dtype=np.float64)
                nearest = {}
                for support_name, support in (("current2400", current), ("ancestral_union", union)):
                    distance = cdist(z, (support - np.asarray(mean, dtype=np.float64)) / np.asarray(std, dtype=np.float64), "sqeuclidean") / 17
                    indices = distance.argmin(axis=1)
                    nearest[support_name + "_index"] = indices
                    nearest[support_name + "_distance"] = np.sqrt(distance[np.arange(len(query)), indices])
                np.savez_compressed(out / f"support_s{stream}_{arm}.npz", observations=query, **nearest)
                for index, metadata in enumerate(query_metadata):
                    row = dict(metadata, current2400_distance=float(nearest["current2400_distance"][index]), ancestral_union_distance=float(nearest["ancestral_union_distance"][index]), current2400_nearest_row=int(nearest["current2400_index"][index]), ancestral_union_nearest_row=int(nearest["ancestral_union_index"][index]), current_support_rows=len(current), ancestral_unique_support_rows=len(union))
                    support_rows.append(row)
                for kind in ("last_pre_action", "terminal_fall"):
                    mask = np.asarray([value["kind"] == kind for value in query_metadata])
                    support_summaries.append(dict(stream=stream, arm=arm, round=latest["round"], kind=kind, current2400=summary(nearest["current2400_distance"][mask]), ancestral_union=summary(nearest["ancestral_union_distance"][mask])))

    csv_write(out / "admissions.csv", admission_rows)
    csv_write(out / "corpus_novelty.csv", novelty_rows)
    csv_write(out / "fall_support.csv", support_rows)
    write(out / "dataset_rosters.json", dataset_rosters)
    write(out / "pending_rounds.json", pending)
    write(out / "support_summary.json", support_summaries)
    admission_counts = {}
    for stream in range(3):
        for arm in ("independent", "positive"):
            rows = [row for row in admission_rows if row["stream"] == stream and row["arm"] == arm]
            admission_counts[f"s{stream}_{arm}"] = dict(Counter(row["classification"] for row in rows))
    result = dict(schema="saved-loop-contexts-admission-review-1", status="COMPLETED", config=reader.capture(config_path), source=reader.capture(Path(__file__)), snapshots=cfg["snapshot_roster"], restart_source_episodes=1, restart_contexts=8, admission_counts=admission_counts, completed_admission_rows=len(admission_rows), corpus_novelty_rows=len(novelty_rows), fall_support_queries=len(support_rows), support_summaries=support_summaries, latest_nonfall_episodes=latest_rows, limitations=["Saved-data descriptive review; not an independent experiment audit and not a new admission gate.", "Nearest standardized distance describes support; no calibrated threshold establishes recovery, causal failure or teachability.", "Ancestral support means data supplied to each ancestor fit, not proof every unique row was sampled by its minibatch stream.", "Current round is latest complete within each frozen snapshot; different streams can be at different completed rounds. Pending rounds are not zero outcomes.", "Exact byte novelty is shown separately for stored raw FP64 and actual learner FP32 inputs/targets. Exact temporal neighbors may be correlated; these are counts, not independent tasks.", "The eight acquisition contexts originate from a single fixed W2s0-u2000 student episode and do not cover eight independent source trajectories."], new_physics=0, new_model_queries=0, new_neural_forwards=0, new_policy_updates=0)
    write(out / "source_bindings.json", list(reader.bindings.values()))
    write(out / "result.json", result)
    costs = dict(cpu_seconds=time.process_time() - CPU_STARTED, wall_seconds=time.monotonic() - WALL_STARTED, cpu_limit_seconds=360, peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, output_bytes=sum(path.stat().st_size for path in out.rglob("*") if path.is_file()), new_physics=0, new_model_queries=0, new_neural_forwards=0, new_policy_updates=0, checkpoint_deserializations=0)
    write(out / "accounting.json", costs)
    print(json.dumps(dict(output=str(out), costs=costs, admissions=admission_counts, support_queries=len(support_rows))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.config.resolve(), args.out.resolve())
    except BaseException as error:
        write(args.out.resolve() / "failure.json", dict(error=repr(error), cpu_seconds=time.process_time()-CPU_STARTED, wall_seconds=time.monotonic()-WALL_STARTED, new_physics=0, new_neural_forwards=0, new_policy_updates=0))
        raise
