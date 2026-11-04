"""Metadata-only closure after all metric outputs were written; no recomputation."""
from pathlib import Path
import csv
import hashlib
import json
import resource
import statistics
import time

resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
started = time.monotonic()
root = Path(__file__).resolve().parent
progress = json.loads((root / "progress.json").read_text())
raw = json.loads((root / "raw_metrics.json").read_text())
config = json.loads((root / "analysis_config.json").read_text())
assert len(raw["cells"]) == progress["completed_cells"] == 32
assert len(raw["corpora"]) == progress["unique_corpora"] == 19
rows = list(csv.DictReader((root / "metrics.csv").open()))
assert len(rows) == 64
assert {(int(row["stream"]), int(row["round"]), row["arm"]) for row in rows} == {
    (row["stream"], row["round"], row["arm"]) for row in raw["cells"]
}
sources = [root / "adapter.py", root / "action_spectrum.original.py", root / "label_conflict.original.py"]
for source, expected in zip(sources, [config["adapter"], *config["archived_sources"]], strict=True):
    data = source.read_bytes()
    assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"]
summaries = []
keys = ["above5_power_share", "r2_full_gap100", "r2_low_gap100", "r2_high_gap100", "r2_full_episode_held_out", "r2_low_episode_held_out", "r2_high_episode_held_out", "conflict_other_over_same", "held_mse_after", "own_mse_after", "mean_nearest_distance_episode_held_out"]
for window in ("primary", "burn100"):
    for round_index in range(1, 7):
        for arm in ("independent", "positive"):
            selected = [row for row in rows if row["window"] == window and int(row["round"]) == round_index and row["arm"] == arm]
            if selected:
                summaries.append(dict(window=window, round=round_index, arm=arm, streams=len(selected), **{key: statistics.mean(float(row[key]) for row in selected) for key in keys}))
with (root / "summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)

def binding(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

output = dict(
    schema="l3-saved-spectrum-conflict-review-completion-1",
    status="METRICS_COMPLETE_METADATA_RECOVERED",
    original_adapter_exit_status=1,
    original_error="Final receipt JSON serialization of Path, after raw_metrics.json and metrics.csv were written",
    original_error_log=binding(root / "stderr.log"),
    scientific_measurements_repeated=0,
    cells=32, unique_corpora=19, cached_references=13,
    windows=["primary", "burn100"],
    raw_metrics=binding(root / "raw_metrics.json"),
    metrics_csv=binding(root / "metrics.csv"), summary_csv=binding(root / "summary.csv"),
    file_bindings=binding(root / "file_bindings.json"),
    original_adapter=binding(root / "adapter.py"),
    closure_source=binding(Path(__file__)),
    config=binding(root / "analysis_config.json"),
    paid_cpu_seconds_lower_bound_at_last_progress=progress["cpu_seconds"],
    final_adapter_cpu_seconds=None,
    original_adapter_cpu_hard_upper_seconds=355,
    command_and_metadata_closure_cpu_reservation_seconds=5,
    conservative_aggregate_cpu_charge_seconds=360,
    exact_final_cpu_counter_unavailable=True,
    closure_cpu_seconds=time.process_time(),
    closure_wall_seconds=time.monotonic() - started,
    physics_substeps=0, model_queries=0, native_controls=0,
    neural_forward_rows=0, policy_updates=0,
    limitations=[
        "Descriptive saved-array diagnostics, not a learning/admission gate or a causal claim",
        "Burn100 sensitivity marks 144 of1600 same-episode query K-means missing; nearest1 metrics retain a finite eligible neighbour",
        "Frequency windows are2.4s primary/1.6s sensitivity; gait-frequency resolution is limited",
        "Repeated corpora are cached references, not independent task replications",
        "Distance bins are original within-cell quantiles; pairwise differences remain susceptible to support-distribution changes",
        "No new SAC reference predictions or reference-matched ranking",
    ],
)
(root / "receipt.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
print("PRIMARY_COMMON3STREAMS_R1_R5")
for row in summaries:
    if row["window"] == "primary" and row["round"] in (1, 5, 6):
        print(json.dumps(row))
