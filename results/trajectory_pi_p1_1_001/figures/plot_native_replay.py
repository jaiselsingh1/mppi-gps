"""Render saved Walker joint angles only; no simulator, actor or optimizer imports."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
JOINTS = (
    "thigh_joint",
    "leg_joint",
    "foot_joint",
    "thigh_left_joint",
    "leg_left_joint",
    "foot_left_joint",
)


def render(base: Path, directory: Path) -> dict:
    wall, cpu = time.monotonic(), time.process_time()
    directory.mkdir(parents=True, exist_ok=True)
    figure_path = directory / "native_replay_joint_angles.png"
    provenance_path = directory / "native_replay_joint_angles.provenance.json"
    if figure_path.exists() or provenance_path.exists():
        raise FileExistsError("Saved-only figure output is create-only")
    bindings, logical_reads, decoded_bytes = {}, 0, 0

    def read(path: Path) -> bytes:
        nonlocal logical_reads
        payload = path.read_bytes()
        logical_reads += len(payload)
        bindings[str(path.resolve())] = dict(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
        return payload

    validation = base / "remote_evidence/native_validation"
    result = json.loads(read(validation / "result.json"))
    if result["status"] != "PASS" or len(result["episodes"]) != 2:
        raise ValueError("The plotted native baseline result must have passed")
    traces, overlays = [], {}
    for slot in (0, 1):
        path = base / "previous_prepared/originals" / f"teacher_{slot}.npz"
        with np.load(io.BytesIO(read(path)), allow_pickle=False) as raw:
            original = raw["states"].copy()
            if len(original) == 1000:
                original = np.concatenate((original, raw["terminal_state"][None]))
        values = {"Original macOS": original}
        for label, role in (("Native capture", "capture"), ("Native repeat", "replay")):
            path = validation / f"{role}_{slot}.h5"
            with h5py.File(io.BytesIO(read(path)), "r") as raw:
                values[label] = raw["states"][:]
        if any(v.shape != (1001, 19) or not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Plot all1001 finite actual boundaries without invented tails")
        episode = result["episodes"][slot]
        if not (episode["outcome"]["walking"] and episode["replay_outcome"]["walking"]):
            raise ValueError("Saved native walking qualification is required")
        overlays[str(slot)] = (
            values["Native capture"].dtype == values["Native repeat"].dtype
            and values["Native capture"].tobytes() == values["Native repeat"].tobytes()
        )
        if not overlays[str(slot)]:
            raise ValueError("Repeat does not overlay the capture exactly")
        decoded_bytes += sum(value.nbytes for value in values.values())
        traces.append(values)

    os.environ.setdefault("MPLCONFIGDIR", str(base / "engineering/matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axes = plt.subplots(6, 2, figsize=(12, 13), sharex=True, sharey="row")
    styles = {
        "Original macOS": dict(color="#666666", linewidth=0.9, alpha=0.85),
        "Native capture": dict(color="#0072B2", linewidth=1.4),
        "Native repeat": dict(color="#D55E00", linewidth=0.8, linestyle="--"),
    }
    for slot, values in enumerate(traces):
        for joint, name in enumerate(JOINTS):
            axis = axes[joint, slot]
            for label, states in values.items():
                axis.plot(states[:, 0], states[:, 4 + joint], label=label, **styles[label])
            axis.grid(alpha=0.15)
            if slot == 0:
                axis.set_ylabel(name + "\nangle (rad)")
            if joint == 0:
                axis.set_title(f"Source slot{slot}: native capture and repeat both walk", pad=8)
            if joint == 5:
                axis.set_xlabel("actual simulator time (s)")
            axis.set_xlim(0, 8)
    figure.suptitle(
        "Walker joint angles: original source and worker-native replay", fontsize=14, y=0.992
    )
    figure.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.971),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.5,
        0.008,
        "Saved-only · unchanged original controls and initial contexts · native repeat overlays capture exactly",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.12, right=0.985, bottom=0.05, top=0.925, hspace=0.32, wspace=0.10)
    with figure_path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    plt.close(figure)
    recipe = Path(__file__).resolve()
    recipe_payload = read(recipe)
    png = figure_path.read_bytes()
    logical_reads += len(png)
    receipt = dict(
        schema="trajectory-pi-native-replay-joint-figure-1",
        status="SAVED_ONLY_COMPLETE",
        recipe=dict(
            path=str(recipe),
            bytes=len(recipe_payload),
            sha256=hashlib.sha256(recipe_payload).hexdigest(),
        ),
        inputs=bindings,
        figure=dict(
            path=str(figure_path.resolve()), bytes=len(png), sha256=hashlib.sha256(png).hexdigest()
        ),
        fullphysics_state_columns=[4, 5, 6, 7, 8, 9],
        qpos_columns=[3, 4, 5, 6, 7, 8],
        joint_names=list(JOINTS),
        time_coordinate="actual saved FULLPHYSICS column0",
        native_repeat_exact_bytes=overlays,
        all_native_slots_walk=True,
        interpretation="Two fixed source trajectories, not independent replications or student policy outcomes.",
        versions=dict(
            numpy=np.__version__, h5py=h5py.__version__, matplotlib=matplotlib.__version__
        ),
        logical_read_bytes=logical_reads,
        decoded_state_array_bytes=decoded_bytes,
        wall_seconds=time.monotonic() - wall,
        cpu_seconds=time.process_time() - cpu,
        new_native_controls=0,
        new_model_controls=0,
        new_neural_rows=0,
        new_gradients=0,
        new_optimizer_updates=0,
        physical_disk_io_bytes=None,
    )
    with provenance_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "runs/2026-11-03_trajectory_pi_p1_1_001"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/trajectory_pi_p1_1_001/figures")
    args = parser.parse_args()
    result = render(args.base, args.out)
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "status", "figure", "wall_seconds")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
