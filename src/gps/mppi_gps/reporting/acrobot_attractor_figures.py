"""Read saved Acrobot LQR probes; depict finite capture tests, not basin proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read_saved(base: Path) -> tuple[list[dict], dict, dict]:
    """Independently recompute finite height/angle/speed capture from arrays."""
    import h5py

    packet = json.loads((base / "lqr.json").read_text())
    config = json.loads((base / "config.json").read_text())
    rows = []
    with h5py.File(base / "evidence.h5", "r") as data:
        for name, group in data["trials"].items():
            case = json.loads(group.attrs["case_json"])
            states, actions, raw = group["states"][:], group["actions"][:], group["raw_actions"][:]
            if group.attrs["status"] != "COMPLETED" or states.shape != (501, 5):
                raise ValueError("A finite canonical probe is incomplete: " + name)
            if not all(np.isfinite(value).all() for value in (states, actions, raw)):
                raise ValueError("Saved native arrays contain invalid values: " + name)
            error = states[:, 1:].copy()
            error[:, :2] = (error[:, :2] + np.pi) % (2 * np.pi) - np.pi
            height = 2 + np.cos(states[:, 1]) + np.cos(states[:, 1] + states[:, 2])
            angle = float(np.abs(error[-100:, :2]).max())
            velocity = float(np.abs(error[-100:, 2:]).max())
            height_success = bool((height[-100:] >= 3.8).all())
            captured = height_success and angle <= 0.02 and velocity <= 0.05
            if captured != bool(group.attrs["captured"]):
                raise ValueError("Saved capture differs from recomputation: " + name)
            if height_success != bool(group.attrs["height_success"]):
                raise ValueError("Saved height success differs from recomputation: " + name)
            if not np.array_equal(states[0], case["state"]):
                raise ValueError("Recorded restart differs: " + name)
            rows.append(
                dict(
                    case=case,
                    states=states,
                    actions=actions,
                    raw=raw,
                    height=height,
                    captured=captured,
                    height_success=height_success,
                    final_angle_max=angle,
                    final_velocity_max=velocity,
                    clipped_fraction=float(np.mean(np.abs(raw) > 1)),
                )
            )
    return rows, packet, config


def summarize(rows: list[dict], packet: dict) -> dict:
    by_kind = {}
    for kind in sorted({r["case"]["kind"] for r in rows}):
        selected = [r for r in rows if r["case"]["kind"] == kind]
        by_kind[kind] = dict(tested=len(selected), captured=sum(r["captured"] for r in selected))
    shells = []
    selected_factor = None
    contiguous = True
    for factor in sorted({r["case"]["factor"] for r in rows if r["case"]["kind"] == "shell"}):
        selected = [r for r in rows if r["case"].get("factor") == factor]
        count = sum(r["captured"] for r in selected)
        shells.append(dict(factor=factor, tested=len(selected), captured=count))
        contiguous = contiguous and len(selected) == 24 and count == 24
        if contiguous:
            selected_factor = factor
    axes = []
    for coordinate, unit in enumerate(("rad", "rad", "rad/s", "rad/s")):
        selected = [
            r
            for r in rows
            if r["case"]["kind"] == "marginal" and r["case"]["coordinate"] == coordinate
        ]
        passed = sorted(r["case"]["value"] for r in selected if r["captured"])
        failed = sorted(r["case"]["value"] for r in selected if not r["captured"])
        axes.append(
            dict(
                coordinate=coordinate,
                unit=unit,
                captured_test_values=passed,
                failed_test_values=failed,
            )
        )
    rho = float(selected_factor * packet["unsaturated_rho"]) if selected_factor else None
    p = np.asarray(packet["P"])
    bounds = np.sqrt(rho * np.diag(np.linalg.inv(p))).tolist() if rho else None
    initial = np.array([r["states"][0, 1:] for r in rows if r["captured"]])
    historical = []
    for row in rows:
        case = row["case"]
        if case["kind"] == "historical":
            historical.append(
                dict(
                    id=case["id"],
                    captured=row["captured"],
                    height_success=row["height_success"],
                    clipped_fraction=row["clipped_fraction"],
                    final_angle_max=row["final_angle_max"],
                    final_velocity_max=row["final_velocity_max"],
                )
            )
    return dict(
        groups=by_kind,
        shell_calibration=shells,
        selected_contiguous_shell_factor=selected_factor,
        candidate_join_rho=rho,
        marginal_tests=axes,
        candidate_level_coordinate_projections=bounds,
        projections_are_not_a_joint_capture_box=True,
        captured_initial_coordinate_min=initial.min(0).tolist(),
        captured_initial_coordinate_max=initial.max(0).tolist(),
        historical=sorted(historical, key=lambda r: r["id"]),
        total_cases=len(rows),
        capture_count=sum(r["captured"] for r in rows),
        interpretation="Finite deterministic shell/grid and random held-out probes; not a nonlinear basin proof. Coordinate extrema describe tested correlated states only.",
    )


def historical_figure(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    selected = sorted(
        (r for r in rows if r["case"]["kind"] == "historical"), key=lambda r: r["case"]["id"]
    )
    fig, axes = plt.subplots(4, 4, figsize=(14, 9), sharex="col")
    for j, row in enumerate(selected):
        states = row["states"]
        t = states[:, 0] - states[0, 0]
        axes[0, j].plot(t, row["height"], color="#1765ab", lw=1)
        axes[0, j].axhline(3.8, color="black", ls=":", lw=0.8)
        axes[0, j].set_ylim(-0.1, 4.2)
        axes[0, j].set_title(
            f"{row['case']['id']}: {'capture' if row['captured'] else 'failure'}\n"
            f"saturated {100 * row['clipped_fraction']:.1f}%",
            fontsize=9,
        )
        for k, color in enumerate(("#1765ab", "#df8028")):
            axes[1, j].plot(t, states[:, 1 + k], color=color, lw=0.9)
            axes[2, j].plot(t, states[:, 3 + k], color=color, lw=0.9)
        axes[3, j].plot(t[:-1], 2.0 * row["actions"][:, 0], color="#1765ab", lw=0.8)
        raw_axis = axes[3, j].twinx()
        raw_axis.plot(t[:-1], 2.0 * row["raw"][:, 0], color="#777777", alpha=0.5, lw=0.7)
        raw_axis.set_yscale("symlog", linthresh=1)
        raw_axis.tick_params(axis="y", labelsize=6, colors="#777777")
        axes[3, j].set_ylim(-2.2, 2.2)
        axes[3, j].set_xlabel("Time (s)")
        for i in range(4):
            axes[i, j].axvspan(t[-1] - 2, t[-1], color="#b4c5d9", alpha=0.15)
            axes[i, j].grid(alpha=0.15)
            axes[i, j].tick_params(labelsize=7)
    for ax, label in zip(
        axes[:, 0],
        (
            "Tip height (m)",
            "Raw joint angles (rad)",
            "Joint velocities (rad/s)",
            "Motor torque (Nm)",
        ),
    ):
        ax.set_ylabel(label, fontsize=8)
    fig.suptitle("Upright LQR on all four historical starts: one captured", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Joint 0 blue, joint 1 orange. Bottom: blue physical torque 2u Nm; gray unclipped 2u command on symlog right axes (Nm). "
        "Shading is final 2 s. Saved native traces; zero new physics.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(out / "historical_lqr.png", dpi=120, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out / "historical_lqr.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def basin_figure(rows: list[dict], packet: dict, summary: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    initial = np.array([r["states"][0, 1:] for r in rows])
    captured = np.array([r["captured"] for r in rows])
    historical = np.array([r["case"]["kind"] == "historical" for r in rows])
    covariance = summary["candidate_join_rho"] * np.linalg.inv(packet["P"])
    for ax, dims, labels in zip(
        axes,
        ((0, 1), (2, 3)),
        (
            ("Shoulder angle (rad)", "Elbow angle (rad)"),
            ("Shoulder velocity (rad/s)", "Elbow velocity (rad/s)"),
        ),
    ):
        for value, color, marker, label in (
            (False, "#ce4944", "x", "Failed tested start"),
            (True, "#269465", "o", "Captured tested start"),
        ):
            mask = captured == value
            ax.scatter(
                initial[mask, dims[0]],
                initial[mask, dims[1]],
                c=color,
                marker=marker,
                s=18,
                alpha=0.6,
                linewidths=0.8,
                label=label,
            )
        ax.scatter(
            initial[historical, dims[0]],
            initial[historical, dims[1]],
            c=np.where(captured[historical], "#269465", "#ce4944"),
            marker="*",
            edgecolors="black",
            s=140,
            linewidths=0.8,
            label="Historical starts",
        )
        sub = covariance[np.ix_(dims, dims)]
        values, vectors = np.linalg.eigh(sub)
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        ax.add_patch(
            Ellipse(
                (0, 0),
                2 * np.sqrt(values[1]),
                2 * np.sqrt(values[0]),
                angle=angle,
                fill=False,
                ls="--",
                color="#444444",
                lw=1,
                label="Selected level-set projection",
            )
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(alpha=0.15)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.27), fontsize=7, frameon=False, ncol=2)
    fig.suptitle(
        "Finite LQR capture probes: projections hide conditioning on other coordinates",
        y=1.02,
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.035,
        "All 362 tested starts shown; duplicate coordinates overlap. Dashed level set and axis extrema are NOT a captured box or a nonlinear basin proof.\n"
        "Selected shell 4: calibration 24/24; held-out boundary 64/64 and interior 64/64. Shell 16: 18/24.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    fig.savefig(out / "basin_projections.png", dpi=120, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out / "basin_projections.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    start, cpu = time.monotonic(), time.process_time()
    args.out.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    rows, packet, _ = read_saved(args.base)
    summary = summarize(rows, packet)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    historical_figure(rows, args.out)
    basin_figure(rows, packet, summary, args.out)
    inputs = {
        name: dict(bytes=(args.base / name).stat().st_size, sha256=sha(args.base / name))
        for name in ("evidence.h5", "lqr.json", "config.json")
    }
    receipt = dict(
        status="PASSED",
        source_sha256=sha(Path(__file__)),
        inputs=inputs,
        cpu_seconds=time.process_time() - cpu,
        wall_seconds=time.monotonic() - start,
        new_physics_controls=0,
        new_network_forwards=0,
        new_fitting_updates=0,
        note="Descriptive saved-array figure/recomputation only; original metadata finalization failure remains recorded.",
    )
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
