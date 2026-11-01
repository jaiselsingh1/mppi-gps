"""Report saved Walker motions and frozen segment diagnostics without simulation."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ..provenance import sha, write

LIMITS = dict(
    wall_seconds=600,
    peak_rss_bytes=512 * 1024**2,
    storage_bytes=128 * 1024**2,
    raw_read_bytes=64 * 1024**2,
)
PAIR_IDS = ("v2_r0_trial_raw_0", "v2_r0_trial_raw_1")
JOINTS = ("rootx", "rootz", "rooty", "thigh", "leg", "foot", "thigh_left", "leg_left", "foot_left")
FIGURES = (
    "all_trajectory_errors",
    "within_recurrence",
    "cross_trajectory_minima",
    "near_equal_pair",
)


class _Budget:
    def __init__(self, out: Path, limits: dict) -> None:
        self.out, self.limits = out, limits
        self.started = time.perf_counter()
        self.raw_read_bytes = 0
        self.peak_rss_bytes = 0

    def check(self) -> None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(peak if sys.platform == "darwin" else peak * 1024)
        self.peak_rss_bytes = max(self.peak_rss_bytes, peak)
        used = sum(path.stat().st_size for path in self.out.iterdir() if path.is_file())
        if (
            time.perf_counter() - self.started > self.limits["wall_seconds"]
            or peak > self.limits["peak_rss_bytes"]
            or used + 16384 > self.limits["storage_bytes"]
            or self.raw_read_bytes > self.limits["raw_read_bytes"]
        ):
            raise ValueError("Frozen report budget exhausted; retain partial output")


class _RawFile(io.RawIOBase):
    """Count actual compact-archive reads, including its integrity hash pass."""

    def __init__(self, path: Path, budget: _Budget) -> None:
        self.stream, self.budget = path.open("rb"), budget
        self.file_bytes = path.stat().st_size

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.file_bytes - self.tell()
        remaining = self.budget.limits["raw_read_bytes"] - self.budget.raw_read_bytes
        if size > remaining:
            raise ValueError("Compact raw read would exceed the frozen report budget")
        value = self.stream.read(size)
        self.budget.raw_read_bytes += len(value)
        return value

    def readinto(self, target) -> int:
        value = self.read(len(target))
        target[: len(value)] = value
        return len(value)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.stream.seek(offset, whence)

    def tell(self) -> int:
        return self.stream.tell()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        if not self.closed:
            self.stream.close()
        super().close()


def _read_json(path: Path, inputs: dict) -> dict:
    data = path.read_bytes()
    inputs[str(path)] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return json.loads(data)


def _validate(result: dict) -> None:
    if (
        result.get("schema") != "walker-motion-diagnosis-result-v1"
        or result.get("execution_complete") is not True
        or result.get("error") is not None
    ):
        raise ValueError("Explicitly completed motion-diagnosis result required")
    rows = result["trajectories"]
    if len(rows) != result["counts"]["trajectories"] or not rows:
        raise ValueError("Saved motion roster is incomplete")
    if len(result["cross_pairs"]) != len(rows) * (len(rows) - 1) // 2:
        raise ValueError("Every original trajectory pair must remain present")
    if not set(PAIR_IDS).issubset({row["id"] for row in rows}):
        raise ValueError("The declared near-equal pair is missing")
    for index, row in enumerate(rows):
        cursor = 0
        if row["index"] != index or not 0 < row["N"] <= 1000:
            raise ValueError("Original order and executed lengths must be explicit")
        for part in row["curves"]:
            if (
                part["start_control"] != cursor
                or part["rows"] <= 0
                or part["stop_control_exclusive"] - cursor != part["rows"]
            ):
                raise ValueError("Derived curves must preserve all executed rows without padding")
            for owner, width in (("physical", 17), ("actions", 6)):
                for statistic in ("mean", "min", "max"):
                    if np.asarray(part[owner][statistic]).shape != (width,):
                        raise ValueError("Malformed saved motion curve")
            cursor = part["stop_control_exclusive"]
        if cursor != row["N"]:
            raise ValueError("Missing or fabricated post-fall curve rows")


def _best_columns(best: dict | None) -> dict:
    if best is None:
        return dict(status="no_eligible_comparison")
    return (
        {key: value for key, value in best.items() if key != "squared_contributions"}
        | {"R2_" + key: value for key, value in best["squared_contributions"].items()}
        | dict(status="saved_minimum")
    )


def build_tables(result: dict) -> dict[str, list[dict]]:
    """Export every saved pair/minimum/contribution; no ranking is reselected."""
    tables = {name: [] for name in ("trajectories", "curve_bins", "within_best", "cross_pairs")}
    sources = {row["id"]: row["source"] for row in result["trajectories"]}
    for row in result["trajectories"]:
        identity = {key: row[key] for key in ("index", "id", "source", "N")}
        tables["trajectories"].append(
            identity
            | {key: row[key] for key in ("E", "first100_E", "terminal_state_available")}
            | {f"actuator_mse_{j}": value for j, value in enumerate(row["per_actuator_mse"])}
        )
        tables["within_best"].append(
            identity
            | dict(eligible_pairs=row["within"]["eligible_pairs"])
            | _best_columns(row["within"]["best"])
        )
        for part in row["curves"]:
            flat = identity | {
                key: part[key] for key in ("start_control", "stop_control_exclusive", "rows")
            }
            for owner in ("physical", "actions"):
                for statistic, values in part[owner].items():
                    flat.update({f"{owner}_{statistic}_{j}": x for j, x in enumerate(values)})
            flat.update({"displacement_" + key: x for key, x in part["displacement"].items()})
            for statistic, values in part["reference_error"].items():
                if statistic == "E":
                    flat["reference_E"] = values
                else:
                    flat.update({f"reference_{statistic}_{j}": x for j, x in enumerate(values)})
            tables["curve_bins"].append(flat)
    for pair in result["cross_pairs"]:
        tables["cross_pairs"].append(
            {key: value for key, value in pair.items() if key != "best"}
            | dict(first_source=sources[pair["first"]], second_source=sources[pair["second"]])
            | _best_columns(pair["best"])
        )
    return tables


def _csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _load_raw(run: Path, result: dict, budget: _Budget) -> dict[int, dict] | None:
    binding = result["motion_archive"]
    path = (run / binding["file"]).resolve()
    if not path.is_relative_to(run):
        raise ValueError("Compact motion archive escapes its run")
    if not path.exists():
        return None
    if path.stat().st_size != binding["bytes"]:
        raise ValueError("Compact motion archive size differs")
    with _RawFile(path, budget) as stream:
        digest = hashlib.sha256()
        while block := stream.read(min(1024**2, binding["bytes"] + 1)):
            digest.update(block)
        if digest.hexdigest() != binding["sha256"]:
            raise ValueError("Compact motion archive hash differs")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            output = {}
            for row in result["trajectories"]:
                index, count = row["index"], row["N"]
                arrays = {
                    name: archive[f"{name}_{index:03d}"]
                    for name in ("states", "actions", "reference")
                }
                for name, width in (("states", 19), ("actions", 6), ("reference", 6)):
                    if arrays[name].shape != (count, width) or not np.isfinite(arrays[name]).all():
                        raise ValueError("Unaligned or nonfinite compact motion array")
                key = f"terminal_{index:03d}"
                if (key in archive.files) != row["terminal_state_available"]:
                    raise ValueError("Terminal availability changed")
                if row["terminal_state_available"]:
                    arrays["terminal"] = archive[key]
                    if (
                        arrays["terminal"].shape != (19,)
                        or not np.isfinite(arrays["terminal"]).all()
                    ):
                        raise ValueError("Invalid compact terminal state")
                arrays["error"] = np.square(arrays["actions"] - arrays["reference"])
                output[index] = arrays
                budget.check()
    return output


def _plotting():
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="walker-motion-mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    return plt, PdfPages


def _save(plt, figure, out: Path, stem: str, budget: _Budget) -> None:
    figure.savefig(out / (stem + ".png"), dpi=140, bbox_inches="tight")
    figure.savefig(out / (stem + ".pdf"), bbox_inches="tight")
    plt.close(figure)
    budget.check()


def _error_heatmap(plt, result: dict, out: Path, budget: _Budget) -> None:
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    rows = result["trajectories"]
    norm = Normalize(0, 4)
    figure, axis = plt.subplots(figsize=(18, max(5, len(rows) * 0.25)))
    axis.set_facecolor("0.86")
    for index, row in enumerate(rows):
        for part in row["curves"]:
            axis.add_patch(
                Rectangle(
                    (part["start_control"], index - 0.45),
                    part["rows"],
                    0.9,
                    facecolor=plt.cm.viridis(norm(part["reference_error"]["E"])),
                    linewidth=0,
                )
            )
    axis.set_yticks(
        range(len(rows)), [f"{r['source']}/{r['id']} (N={r['N']})" for r in rows], fontsize=6
    )
    axis.set(
        xlim=(0, 1000),
        ylim=(len(rows) - 0.5, -0.5),
        xlabel="Absolute control index",
        title="All recorded trajectories: saved 25-control reference error; gray is missing",
    )
    figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axis,
        label="Existing mean squared action-reference error E (bounded scale)",
    )
    _save(plt, figure, out, "all_trajectory_errors", budget)


def _within_maps(plt, result: dict, out: Path, budget: _Budget) -> None:
    rows = result["trajectories"]
    width = min(5, len(rows))
    height = math.ceil(len(rows) / width)
    figure, axes = plt.subplots(height, width, figsize=(4 * width, 3.1 * height), squeeze=False)
    finite = [
        v for row in rows for line in row["within"]["R_matrix"] for v in line if v is not None
    ]
    maximum = max(finite, default=1)
    for axis, row in zip(axes.flat, rows):
        within = row["within"]
        axis.set_title(
            f"{row['source']}\n{row['id']} N={row['N']}; queries={within['eligible_pairs']}",
            fontsize=6,
        )
        if within["best"] is None:
            axis.set_facecolor("0.86")
            axis.text(
                0.5,
                0.5,
                "No eligible within-trajectory comparison",
                ha="center",
                va="center",
                fontsize=6,
                transform=axis.transAxes,
                wrap=True,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            values = np.array(
                [[np.nan if x is None else x for x in line] for line in within["R_matrix"]]
            )
            image = axis.imshow(
                np.ma.masked_invalid(values),
                vmin=0,
                vmax=maximum,
                cmap="viridis",
                origin="lower",
                interpolation="nearest",
            )
            axis.set_facecolor("0.86")
            ticks = list(range(0, len(within["starts"]), 10))
            axis.set_xticks(ticks, [within["starts"][j] for j in ticks], fontsize=5)
            axis.set_yticks(ticks, [within["starts"][j] for j in ticks], fontsize=5)
        axis.set_xlabel("First window start", fontsize=6)
        axis.set_ylabel("Second window start", fontsize=6)
    for axis in list(axes.flat)[len(rows) :]:
        axis.set_visible(False)
    figure.subplots_adjust(
        top=0.98 - 0.65 / figure.get_figheight(), hspace=0.8, wspace=0.55, right=0.9
    )
    if finite:
        figure.colorbar(
            image, ax=axes.ravel().tolist(), label="Frozen physical-state distance R", fraction=0.02
        )
    figure.suptitle(
        "All trajectories, absolute 25-control grid; excluded/short comparisons stay missing"
    )
    _save(plt, figure, out, "within_recurrence", budget)


def _cross_maps(plt, result: dict, out: Path, budget: _Budget) -> None:
    count = len(result["trajectories"])
    labels = [row["id"] for row in result["trajectories"]]
    figure, axes = plt.subplots(1, 2, figsize=(22, 11))
    maximum = max(pair["best"]["R"] for pair in result["cross_pairs"])
    for axis, kind, title in zip(
        axes, ("full", "partial"), ("Full H=25", "Partial H<25; separate horizon"), strict=True
    ):
        values = np.full((count, count), np.nan)
        for pair in result["cross_pairs"]:
            if pair["kind"] == kind:
                i, j = pair["first_index"], pair["second_index"]
                values[i, j] = values[j, i] = pair["best"]["R"]
        image = axis.imshow(
            np.ma.masked_invalid(values),
            vmin=0,
            vmax=maximum,
            cmap="viridis",
            interpolation="nearest",
        )
        axis.set_facecolor("0.86")
        axis.set_xticks(range(count), labels, rotation=90, fontsize=5)
        axis.set_yticks(range(count), labels, fontsize=5)
        axis.set_title(title)
    figure.subplots_adjust(wspace=0.65, right=0.92)
    figure.colorbar(
        image,
        ax=axes.tolist(),
        label="Saved minimum R; do not pool/rank different horizons",
        fraction=0.025,
    )
    figure.suptitle(
        "Every cross-trajectory pair, original roster order; gray cells are ineligible for that panel"
    )
    _save(plt, figure, out, "cross_trajectory_minima", budget)


def _near_pair(plt, result: dict, raw: dict, out: Path, budget: _Budget) -> None:
    rows = [next(row for row in result["trajectories"] if row["id"] == name) for name in PAIR_IDS]
    angle_values = [raw[row["index"]]["states"][:, 3:10] for row in rows]
    angle_limits = (
        min(float(x.min()) for x in angle_values),
        max(float(x.max()) for x in angle_values),
    )
    figure, axes = plt.subplots(3, 4, figsize=(23, 9), squeeze=False)
    figure.subplots_adjust(top=0.84, wspace=0.5, hspace=0.5, right=0.91)
    for column in range(4):
        owner = column % 2
        row, arrays = rows[owner], raw[rows[owner]["index"]]
        stop = 100 if column >= 2 else 1000
        matrices = (arrays["states"][:, 3:10], arrays["actions"], arrays["error"])
        labels = (JOINTS[2:], JOINTS[3:], JOINTS[3:])
        limits = (angle_limits, (-1, 1), (0, 4))
        for index, (axis, values, names, bounds) in enumerate(
            zip(axes[:, column], matrices, labels, limits, strict=True)
        ):
            visible = values[:stop]
            image = axis.imshow(
                visible.T,
                origin="upper",
                aspect="auto",
                interpolation="nearest",
                extent=(0, len(visible), len(names) - 0.5, -0.5),
                vmin=bounds[0],
                vmax=bounds[1],
                cmap="viridis",
            )
            axis.set(xlim=(0, stop), xlabel="Absolute control index")
            axis.set_facecolor("0.86")
            axis.set_yticks(range(len(names)), names, fontsize=6)
            if index == 0:
                axis.set_title(
                    f"{row['id']}\n{'first 100 controls' if column >= 2 else 'full record'}; E={row['E']:.6f}",
                    fontsize=8,
                )
            if column == 3:
                figure.colorbar(
                    image,
                    ax=axes[index, :].tolist(),
                    fraction=0.015,
                    label=(
                        "Native hinge qpos",
                        "Executed bounded control",
                        "Squared reference residual",
                    )[index],
                )
    figure.suptitle(
        "Fixed near-equal full-score pair: actual successive coordinates/actions/errors; no alignment"
    )
    _save(plt, figure, out, "near_equal_pair", budget)


def _panels(row: dict, raw: dict | None, zoom: bool) -> list[tuple[str, tuple, object]]:
    if raw is not None:
        states = raw["states"]
        if "terminal" in raw:
            states = np.vstack((states, raw["terminal"]))
        qpos, qvel = states[:, 1:10], states[:, 10:19]
        return [
            (
                "Root x−x0\n(m)",
                JOINTS[:1],
                (qpos[:, 0] - qpos[0, 0])[:, None],
            ),
            ("Root height\n(m)", JOINTS[1:2], qpos[:, 1:2]),
            ("Pitch / joints\n(rad)", JOINTS[2:], qpos[:, 2:]),
            ("Root velocity\n(m/s)", JOINTS[:2], qvel[:, :2]),
            ("Angular velocity\n(rad/s)", JOINTS[2:], qvel[:, 2:]),
            ("Control u\n[-1,1]", JOINTS[3:], raw["actions"]),
            ("Reference error\n(u−π)²", JOINTS[3:], raw["error"]),
        ]
    curves = [part for part in row["curves"] if not zoom or part["start_control"] < 100]

    def extract(owner, positions, key=None):
        output = []
        for part in curves:
            statistics = part[owner]
            output.append(
                {
                    name: [statistics[key or name][j] for j in positions]
                    for name in ("mean", "min", "max")
                }
            )
        return output

    roots = [
        {
            name: [part["displacement"][name], part["physical"][name][0]]
            for name in ("mean", "min", "max")
        }
        for part in curves
    ]
    error = [
        {
            "mean": part["reference_error"]["per_actuator_mse"],
            "min": part["reference_error"]["min"],
            "max": part["reference_error"]["max"],
        }
        for part in curves
    ]
    return [
        ("Root x−x0\n(m)", JOINTS[:1], [{k: v[:1] for k, v in p.items()} for p in roots]),
        ("Root height\n(m)", JOINTS[1:2], [{k: v[1:] for k, v in p.items()} for p in roots]),
        ("Pitch / joints\n(rad)", JOINTS[2:], extract("physical", range(1, 8))),
        ("Root velocity\n(m/s)", JOINTS[:2], extract("physical", range(8, 10))),
        ("Angular velocity\n(rad/s)", JOINTS[2:], extract("physical", range(10, 17))),
        ("Control u\n[-1,1]", JOINTS[3:], extract("actions", range(6))),
        ("Reference error\n(u−π)²", JOINTS[3:], error),
    ]


def _atlas(plt, PdfPages, result: dict, raw: dict | None, out: Path, budget: _Budget) -> None:
    name = "full_motion_atlas.pdf" if raw is not None else "derived_curve_atlas.pdf"
    with PdfPages(out / name) as atlas:
        for row in result["trajectories"]:
            arrays = None if raw is None else raw[row["index"]]
            figure, axes = plt.subplots(7, 2, figsize=(15, 15), squeeze=False, sharey="row")
            for column, maximum in enumerate((1000, 100)):
                panels = _panels(row, arrays, zoom=column == 1)
                curves = [
                    part for part in row["curves"] if column == 0 or part["start_control"] < 100
                ]
                edges = [part["start_control"] for part in curves] + [
                    curves[-1]["stop_control_exclusive"]
                ]
                for axis, (label, names, values) in zip(axes[:, column], panels, strict=True):
                    for j, component in enumerate(names):
                        color = plt.cm.tab10(j)
                        if arrays is not None:
                            axis.plot(
                                np.arange(len(values)),
                                values[:, j],
                                color=color,
                                linewidth=0.65,
                                label=component,
                            )
                        else:
                            mean, low, high = (
                                [part[key][j] for part in values] for key in ("mean", "min", "max")
                            )
                            _summary_trace(axis, edges, mean, low, high, color, component)
                    if row["N"] < maximum:
                        axis.axvspan(row["N"], maximum, facecolor="0.9", linewidth=0)
                    axis.set(xlim=(0, maximum), ylabel=label)
                    axis.tick_params(labelsize=6)
                    axis.legend(fontsize=4.5, ncol=3, loc="upper right")
                axes[0, column].set_title(
                    "Full absolute control range"
                    if column == 0
                    else "First 100 controls; every raw row if available"
                )
                axes[-1, column].set_xlabel("Absolute control index (one control = 0.008 s)")
            mode = (
                "Actual saved samples, no downsampling"
                if raw is not None
                else "Saved 25-control mean/min/max; pointwise motion is not reconstructed"
            )
            terminal = "available in source" if row["terminal_state_available"] else "missing"
            figure.suptitle(
                f"{row['source']} / {row['id']} — N={row['N']}\n{mode}; terminal {terminal}",
                fontsize=11,
                y=0.965,
            )
            figure.subplots_adjust(top=0.91, hspace=0.45, wspace=0.27)
            atlas.savefig(figure)
            plt.close(figure)
            del axis, axes, figure
            gc.collect()
            budget.check()


def _summary_trace(axis, edges, mean, low, high, color, label):
    """Draw recorded block statistics without Matplotlib's default zero endpoints."""
    trace = axis.stairs(mean, edges, baseline=None, color=color, linewidth=0.7, label=label)
    axis.fill_between(edges, low + low[-1:], high + high[-1:], step="post", color=color, alpha=0.12)
    return trace


def _copy_asset(capsule: Path, name: str, out: Path, inputs: dict, manifest: dict) -> None:
    relative = "report/" + name
    expected = manifest["files"][relative]
    path = (capsule / relative).resolve()
    if not path.is_relative_to(capsule) or not path.is_file():
        raise ValueError("Original full-motion asset is missing or escapes its capsule")
    data = path.read_bytes()
    binding = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    if any(binding[key] != expected[key] for key in binding):
        raise ValueError("Original full-motion asset binding differs")
    inputs[str(path)] = binding
    with (out / name).open("xb") as stream:
        stream.write(data)


def render(run: Path, out: Path) -> dict:
    """Use compact raw once, or derived curves plus bound original raw-view assets."""
    run, out = run.resolve(), out.resolve()
    capsule = run if (run / "run/result.json").is_file() else run.parent
    evidence_root = capsule if (capsule / "capsule.json").is_file() else run
    if out.is_relative_to(evidence_root):
        raise ValueError("Report output must be outside immutable run or capsule evidence")
    if (run / "run/result.json").is_file():
        run = run / "run"
    inputs = {}
    config, result = (_read_json(run / name, inputs) for name in ("config.json", "result.json"))
    _validate(result)
    limits = config.get("report_budgets", LIMITS)
    if limits != LIMITS:
        raise ValueError("Unexpected report budget; version it before execution")
    out.mkdir(parents=True, exist_ok=False)
    budget = _Budget(out, limits)
    receipt = dict(
        schema="walker-motion-diagnosis-report-v1",
        execution_complete=False,
        error=None,
        inputs=inputs,
        source_sha256=sha(Path(__file__)),
        limits=limits,
        scientific_calls=dict(
            network_forwards=0, physics_controls=0, optimizer_steps=0, mppi_queries=0
        ),
        counts=result["counts"],
        motion_archive=result["motion_archive"],
    )
    try:
        tables = build_tables(result)
        for name, rows in tables.items():
            _csv(out / (name + ".csv"), rows)
        raw = _load_raw(run, result, budget)
        plt, PdfPages = _plotting()
        _error_heatmap(plt, result, out, budget)
        _within_maps(plt, result, out, budget)
        _cross_maps(plt, result, out, budget)
        _atlas(plt, PdfPages, result, None, out, budget)
        if raw is not None:
            _near_pair(plt, result, raw, out, budget)
            _atlas(plt, PdfPages, result, raw, out, budget)
            receipt["motion_source"] = "compact_raw_archive; actual samples rendered"
            receipt["copied_original_assets"] = []
        else:
            manifest = _read_json(capsule / "capsule.json", inputs)
            names = ["full_motion_atlas.pdf", "near_equal_pair.png", "near_equal_pair.pdf"]
            for name in names:
                _copy_asset(capsule, name, out, inputs, manifest)
                budget.check()
            receipt["motion_source"] = (
                "saved_derived_curves; exact original raw atlas/pair figures copied"
            )
            receipt["copied_original_assets"] = names
        description = (
            "Actual samples were rendered from the hash-bound compact raw archive."
            if raw is not None
            else "The compact raw archive is external. This report renders saved block summaries and copies the hash-verified original full-motion atlas and near-equal-pair figures; it does not rerender their raw samples."
        )
        count = len(result["trajectories"])
        terminals = sum(row["terminal_state_available"] for row in result["trajectories"])
        (out / "report.md").write_text(f"""# K1.2 saved-motion diagnosis

{description}

Every original trajectory remains present, with source IDs, executed lengths and absolute control order. The full-motion atlas has one page per trajectory and shows all saved physical/action samples plus the first 100 controls. Its {terminals} available terminal states are physical observations only; no terminal actions or missing discovery endpoints are invented. Gray regions are unobserved, never zeros. Native qpos contains root displacement, height, pitch and six joint angles; nine velocities remain unclipped. No simulator, geometric stick model, phase alignment, FFT, new inference or learning is involved.

The derived atlas shows each saved 25-control mean and min/max band on its actual interval, including partial tails. These summaries do not recover the within-bin sequence or display terminal points. Full H=25 and shorter cross-pair windows are separate panels; every saved pair minimum and its state/action/error contributions remains in the CSV. Each minimum is a selected fragment: longer trajectories supply more candidate windows, and its query count remains explicit. Within-trajectory maps retain excluded cells and no-comparison trajectories as missing. Bins, frames, actuators and selected windows are not independent task replications. R is the frozen physical-state distance, not teachability or a newly selected policy ranking. Segment action MSE and own-state reference errors remain distinct.

The fixed later-independent pair is shown at its own recorded successive states without temporal warping. Similarity, recurrence and short matched windows do not establish gait phase, a stable limit cycle, consequential errors, recovery, competent student walking or useful coupling. Different successful/nonperiodic solutions remain valid. No new student execution is performed here.

[Full saved-motion atlas](full_motion_atlas.pdf) · [Derived block-summary atlas](derived_curve_atlas.pdf)

![All {count} error profiles](all_trajectory_errors.png)
![Every within-trajectory map](within_recurrence.png)
![All cross-trajectory pairs, horizons separate](cross_trajectory_minima.png)
![Fixed near-equal pair, actual recorded samples](near_equal_pair.png)
""")
        budget.check()
        receipt.update(
            execution_complete=True, tables={name: len(rows) for name, rows in tables.items()}
        )
    except BaseException as error:
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        receipt.update(
            raw_read_bytes=budget.raw_read_bytes,
            peak_rss_bytes=budget.peak_rss_bytes,
            outputs={
                path.name: dict(sha256=sha(path), bytes=path.stat().st_size)
                for path in sorted(out.iterdir())
                if path.is_file()
            },
        )
        receipt["storage_bytes_before_receipt"] = sum(
            item["bytes"] for item in receipt["outputs"].values()
        )
        receipt["wall_seconds"] = time.perf_counter() - budget.started
        write(out / "receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = render(args.run, args.out)
    print(json.dumps(dict(execution_complete=receipt["execution_complete"], report=str(args.out))))
    return 0
