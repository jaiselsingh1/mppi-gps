"""Figures and kinematic animations from closed Acrobot probe evidence only.

This is a figure renderer, not the independent scientific auditor. It reads the
complete original near-upright cases 0/1, the create-only completion cases 2/3,
and all six energy-cost episodes. It never imports a simulator or executes a
controller. Original interrupted prefixes remain preserved and are not counted
twice in the completed-roster figure.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.gps.mppi_gps.acrobot_energy import (
    components,
    energy_components,
    terminal_cost,
    tip_position,
)

MAX_OUTPUT_BYTES = 12 * 1024**2
MAX_CPU_SECONDS = 12.0
COLORS = ("#2367b8", "#d66b17", "#29885a", "#b93659")


@dataclass(frozen=True)
class SavedTrial:
    part: str
    variant: str
    variant_index: int
    trial_index: int
    reset_seed: int
    source: Path
    source_group: str
    states: np.ndarray
    actions: np.ndarray
    costs: np.ndarray
    objective_cost: float
    energy_weight: float | None
    planner_seconds: float


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def central_trial(trials: list[SavedTrial]) -> SavedTrial:
    """Use central cost order statistics, breaking the even-panel tie by index.

    Selection is within one objective only. With two trials both are central,
    so the lower original trial index is illustrated, irrespective of success.
    All trial trajectories remain present in the scientific dashboard.
    """
    if not trials:
        raise ValueError("A nonempty completed trial set is required.")
    ordered = sorted(trials, key=lambda trial: (trial.objective_cost, trial.trial_index))
    central = (ordered[(len(ordered) - 1) // 2], ordered[len(ordered) // 2])
    return min(central, key=lambda trial: (trial.trial_index, str(trial.source)))


def frame_indices(controls: int, maximum: int = 60) -> list[int]:
    """Include both saved endpoint states without interpolating new states."""
    if controls < 1 or maximum < 2:
        raise ValueError("Positive controls and at least two frames are required.")
    stride = max(1, math.ceil(controls / (maximum - 1)))
    indices = list(range(0, controls + 1, stride))
    if indices[-1] != controls:
        indices.append(controls)
    return indices


def _load(base: Path) -> tuple[list[list[SavedTrial]], list[dict[str, Any]]]:
    import h5py

    groups: list[list[SavedTrial]] = [[], [], [], []]
    inputs = []
    parts = (
        ("near-upright", (0, 1)),
        ("near-upright-completion", (2, 3)),
        ("energy", None),
    )
    for directory, near_indices in parts:
        result_path = base / directory / "result.json"
        result = json.loads(result_path.read_text())
        status = result.get("status", "")
        if status in ("", "STARTING", "RUNNING"):
            raise ValueError("Raw worker output is not closed: " + directory)
        if directory == "energy" and status not in (
            "PROMISING_TEACHER_PROBE_ONLY",
            "TEACHER_PROBLEM_STOP",
        ):
            raise ValueError("All six energy episodes must be complete before rendering.")
        config = json.loads((base / directory / "config.json").read_text())
        part = "energy" if directory == "energy" else "near-upright"
        specification = config["parts"][part]
        evidence_path = base / directory / "evidence.h5"
        with h5py.File(evidence_path, "r") as evidence:
            if evidence.attrs.get("status") != status:
                raise ValueError("Closed HDF5/result status binding differs: " + directory)
            for variant_index, variant in enumerate(specification["variants"]):
                indices = range(2) if near_indices is None else near_indices
                for index in indices:
                    path = f"trials/variant_{variant_index}/trial_{index}"
                    raw = evidence[path]
                    if raw.attrs.get("status") != "COMPLETED":
                        raise ValueError("A requested completed trial is missing: " + path)
                    states, actions, costs = (
                        raw["states"][:],
                        raw["actions"][:],
                        raw["costs"][:],
                    )
                    count = specification["controls"]
                    if (
                        states.shape != (count + 1, 5)
                        or actions.shape != (count, 1)
                        or costs.shape != (count,)
                        or not all(np.isfinite(value).all() for value in (states, actions, costs))
                    ):
                        raise ValueError("Incomplete or nonfinite saved native measurements.")
                    if not np.allclose(np.diff(states[:, 0]), 0.02, rtol=0, atol=1e-10):
                        raise ValueError("Saved state timing differs from 50 Hz controls.")
                    if np.max(np.abs(actions)) > 1:
                        raise ValueError("Saved actions exceed the native torque interface.")
                    energy_weight = variant.get("energy_weight")
                    objective = float(costs.sum())
                    if energy_weight is not None:
                        objective += float(terminal_cost(states[-1]))
                    if not np.isclose(objective, raw.attrs["task_cost"], rtol=0, atol=1e-10):
                        raise ValueError("Within-objective illustration cost differs from receipt.")
                    trial = SavedTrial(
                        part=part,
                        variant=variant["name"],
                        variant_index=variant_index,
                        trial_index=index,
                        reset_seed=int(raw.attrs["reset_seed"]),
                        source=evidence_path,
                        source_group=path,
                        states=states,
                        actions=actions,
                        costs=costs,
                        objective_cost=objective,
                        energy_weight=energy_weight,
                        planner_seconds=variant["planner"]["H"] * 0.02,
                    )
                    groups[variant_index + 1 if part == "energy" else 0].append(trial)
        inputs.extend(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
            for path in (result_path, evidence_path, base / directory / "config.json")
        )
    if [len(group) for group in groups] != [4, 2, 2, 2]:
        raise ValueError("Expected four near-upright and three two-seed energy panels.")
    return groups, inputs


def _title(trials: list[SavedTrial]) -> str:
    trial = trials[0]
    if trial.energy_weight is None:
        return "Dense cost, near upright"
    return f"Energy, {trial.planner_seconds:g} s planner, weight {trial.energy_weight:.2f}"


class _Budget:
    def __init__(self, out: Path, cpu_seconds: float) -> None:
        self.out = out
        self.started = time.process_time()
        self.cpu_seconds = cpu_seconds

    def check(self, additional_bytes: int = 0) -> None:
        if time.process_time() - self.started > self.cpu_seconds:
            raise RuntimeError("Saved-figure CPU reservation exhausted.")
        used = sum(path.stat().st_size for path in self.out.rglob("*") if path.is_file())
        if used + additional_bytes + 64 * 1024 > MAX_OUTPUT_BYTES:
            raise RuntimeError("Saved-figure 12 MiB output reservation exhausted.")

    def write(self, path: Path, payload: bytes) -> None:
        self.check(len(payload))
        with path.open("xb") as target:
            target.write(payload)


def _dashboard(groups: list[list[SavedTrial]], out: Path, budget: _Budget) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out / "mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 4, figsize=(15, 9), sharex="col")
    for column, trials in enumerate(groups):
        successes = 0
        for trial in trials:
            budget.check()
            time_values = trial.states[:, 0] - trial.states[0, 0]
            height = tip_position(trial.states)[:, 2]
            energy = energy_components(trial.states)
            gate = components(
                trial.states, np.zeros((len(trial.states), 1)), trial.energy_weight or 0
            )["gate"]
            successes += bool(np.all(height[-100:] >= 3.8))
            color = COLORS[trial.trial_index % len(COLORS)]
            axes[0, column].plot(
                time_values,
                height,
                color=color,
                label=f"start {trial.trial_index}, seed {trial.reset_seed}",
                lw=0.8,
            )
            for key, style in (("total_energy", "-"), ("potential", "--"), ("kinetic", ":")):
                axes[1, column].plot(time_values, energy[key], color=color, ls=style, lw=0.8)
            axes[2, column].plot(time_values, gate, color=color, lw=0.8)
            axes[3, column].plot(
                time_values[1:], trial.actions[:, 0], color=color, lw=0.6, alpha=0.8
            )
        end = float(trials[0].states[-1, 0] - trials[0].states[0, 0])
        for row in range(4):
            axes[row, column].axvspan(end - 2, end, color="#95a3b8", alpha=0.12)
            axes[row, column].grid(alpha=0.16)
            axes[row, column].tick_params(labelsize=7)
        axes[0, column].axhline(3.8, color="#27384e", ls=":", lw=1)
        axes[0, column].set_ylim(-0.05, 4.1)
        axes[0, column].set_title(
            f"{_title(trials)}\nFinal-two-second height successes: {successes}/{len(trials)}",
            fontsize=9,
        )
        axes[1, column].axhline(58.86, color="#27384e", ls=":", lw=0.8)
        axes[2, column].set_ylim(-0.03, 1.03)
        axes[3, column].set_ylim(-1.05, 1.05)
        axes[3, column].set_xlabel("Recorded time (s)", fontsize=8)
    for row, title in enumerate(
        (
            "Tip height (m)",
            "Energy (J)\nsolid E; dashed PE; dotted KE",
            "Proximity gate g\n(diagnostic for dense cost)",
            "Executed native control",
        )
    ):
        axes[row, 0].set_ylabel(title, fontsize=8)
    figure.suptitle("Acrobot balance and energy discovery: every completed probe", fontsize=13)
    figure.text(
        0.5,
        0.008,
        "Shading: final 2 s. Near-upright cases 0/1 are original; 2/3 use the declared completion. "
        "Plots are saved-data reductions, not new simulation.",
        ha="center",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.875))
    for column in range(4):
        handles, labels = axes[0, column].get_legend_handles_labels()
        position = axes[0, column].get_position()
        figure.legend(
            handles,
            labels,
            fontsize=6,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(position.x0, 0.945),
            bbox_transform=figure.transFigure,
        )
    for name, kwargs in (("probe_dashboard.png", {"dpi": 110}), ("probe_dashboard.pdf", {})):
        stream = io.BytesIO()
        figure.savefig(stream, format=name.rsplit(".", 1)[1], **kwargs)
        budget.write(out / name, stream.getvalue())
    plt.close(figure)


def _animation(trial: SavedTrial, title: str, path: Path, budget: _Budget) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    indices = frame_indices(len(trial.actions))
    frames = []
    font = ImageFont.load_default()

    def point(x: float, z: float) -> tuple[int, int]:
        return int(200 + 75 * x), int(385 - 75 * z)

    for index in indices:
        budget.check()
        state = trial.states[index]
        shoulder, elbow = state[1], state[2]
        joint = (math.sin(shoulder), 2 + math.cos(shoulder))
        tip = (joint[0] + math.sin(shoulder + elbow), joint[1] + math.cos(shoulder + elbow))
        image = Image.new("RGB", (400, 455), "#f8fafc")
        draw = ImageDraw.Draw(image)
        elapsed = state[0] - trial.states[0, 0]
        draw.text((10, 12), title, font=font, fill="#18243b")
        draw.text(
            (10, 30),
            f"Start {trial.trial_index}, seed {trial.reset_seed}; {elapsed:.2f} s",
            font=font,
            fill="#334155",
        )
        for x in range(12, 390, 16):
            draw.line((x, point(0, 3.8)[1], x + 8, point(0, 3.8)[1]), fill="#94a3b8")
        target = point(0, 4)
        draw.ellipse(
            (target[0] - 12, target[1] - 12, target[0] + 12, target[1] + 12),
            outline="#159e90",
            width=2,
        )
        points = point(0, 2), point(*joint), point(*tip)
        draw.line((points[0], points[1]), fill="#3572c6", width=7)
        draw.line((points[1], points[2]), fill="#df8d2d", width=7)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#18243b")
        draw.text(
            (10, 430), "Recorded-state kinematics; zero new physics", font=font, fill="#334155"
        )
        frames.append(image.quantize(colors=16, method=Image.Quantize.FASTOCTREE))
    durations = [
        max(10, round((trial.states[b, 0] - trial.states[a, 0]) * 1000 / 10) * 10)
        for a, b in zip(indices, indices[1:])
    ] + [100]
    stream = io.BytesIO()
    frames[0].save(
        stream,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    budget.write(path, stream.getvalue())
    return {
        "file": path.name,
        "part": trial.part,
        "variant": trial.variant,
        "trial_index": trial.trial_index,
        "reset_seed": trial.reset_seed,
        "source": str(trial.source),
        "source_group": trial.source_group,
        "frame_indices": indices,
        "frame_times_seconds": (trial.states[indices, 0] - trial.states[0, 0]).tolist(),
        "selection": "Within-objective central cost order statistics; even-panel tie uses lower original index.",
        "encoded_last_frame_hold_milliseconds": 100,
        "interpolation": False,
        "new_physics_controls": 0,
    }


def render(base: Path, out: Path, cpu_seconds: float = MAX_CPU_SECONDS) -> dict[str, Any]:
    """Create a bounded dashboard and four illustrative saved-state animations."""
    if not 0 < cpu_seconds <= MAX_CPU_SECONDS:
        raise ValueError("Figure CPU ceiling must lie in (0, 12] seconds.")
    base, out = base.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    budget = _Budget(out, cpu_seconds)
    receipt: dict[str, Any] = {
        "status": "STARTING",
        "role": "Saved-data figure renderer, not independent scientific audit",
        "renderer_source_sha256": _sha(Path(__file__)),
        "new_physics_controls": 0,
        "new_network_forwards": 0,
        "new_optimizer_updates": 0,
        "cpu_ceiling_seconds": cpu_seconds,
        "output_ceiling_bytes": MAX_OUTPUT_BYTES,
    }
    try:
        groups, inputs = _load(base)
        budget.check()
        _dashboard(groups, out, budget)
        receipt["animations"] = [
            _animation(central_trial(group), _title(group), out / f"saved_motion_{i}.gif", budget)
            for i, group in enumerate(groups)
        ]
        receipt.update(
            status="PASSED",
            inputs=inputs,
            completed_trial_counts=[len(group) for group in groups],
            note="Animations depict recorded state boundaries only; they neither replay physics nor establish teacher competence.",
        )
        budget.check()
    except Exception as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    receipt["cpu_seconds"] = time.process_time() - budget.started
    receipt["artifacts"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(out.iterdir())
        if path.is_file()
    }
    receipt["output_bytes_before_receipt"] = sum(
        path.stat().st_size for path in out.rglob("*") if path.is_file()
    )
    with (out / "figure_receipt.json").open("x") as target:
        json.dump(receipt, target, indent=2, allow_nan=False)
        target.write("\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cpu-seconds", type=float, default=MAX_CPU_SECONDS)
    args = parser.parse_args(argv)
    result = render(args.base, args.out, args.cpu_seconds)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "cpu_seconds", "error") if key in result}
        )
    )
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
