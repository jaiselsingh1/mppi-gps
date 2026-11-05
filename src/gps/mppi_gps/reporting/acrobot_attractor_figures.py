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


def _controller_trace(group: object, label: str) -> dict:
    """Read one executed N+1 trace and the final refinement used at each step."""
    states, actions, costs = group["states"][:], group["actions"][:], group["costs"][:]
    if states.shape != (len(actions) + 1, 5) or actions.shape != (len(costs), 1):
        raise ValueError("Incomplete executed controller trace: " + label)
    if not all(np.isfinite(x).all() for x in (states, actions, costs)):
        raise ValueError("Nonfinite executed controller trace: " + label)
    ess, temperatures = [], []
    if "planning" in group:
        for step in range(len(actions)):
            bank = group[f"planning/control_{step:04d}"]
            names = sorted(name for name in bank if name.startswith("pass_"))
            if not names:
                raise ValueError("Missing final executed refinement: " + label)
            last = bank[names[-1]]
            ess.append(float(last.attrs["neff"]))
            temperatures.append(float(last.attrs["lam"]))
    height = 2 + np.cos(states[:, 1]) + np.cos(states[:, 1] + states[:, 2])
    return dict(
        label=label,
        states=states,
        actions=actions,
        costs=costs,
        height=height,
        ess=np.asarray(ess),
        temperatures=np.asarray(temperatures),
    )


def _trace_description(trace: dict) -> dict:
    return dict(
        controls=len(trace["actions"]),
        duration_seconds=float(trace["states"][-1, 0] - trace["states"][0, 0]),
        running_cost=float(trace["costs"].sum()),
        final_two_second_height_hold=bool(
            len(trace["actions"]) >= 100 and (trace["height"][-100:] >= 3.8).all()
        ),
        maximum_height=float(trace["height"][1:].max()),
        ess_minimum=float(trace["ess"].min()) if len(trace["ess"]) else None,
        ess_median=float(np.median(trace["ess"])) if len(trace["ess"]) else None,
        ess_maximum=float(trace["ess"].max()) if len(trace["ess"]) else None,
    )


def _relative_value(trace: dict, p: np.ndarray, rho: float) -> np.ndarray:
    error = trace["states"][:, 1:].copy()
    error[:, :2] = (error[:, :2] + np.pi) % (2 * np.pi) - np.pi
    return np.einsum("ni,ij,nj->n", error, p, error) / rho


def render_controller_comparison(base: Path, previous: Path, locked_path: Path, out: Path) -> dict:
    """Compare closed saved controller stages; never synthesize a missing suffix.

    Near-upright columns use the same dense objective and 500-control duration.
    Swing-up columns have their own energy objective and an explicitly recorded
    MPPI-prefix/LQR-suffix duration. Their costs are not compared to dense cost.
    The exact initial contexts, not later diverging banks, define the pairing.
    """
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start, cpu = time.monotonic(), time.process_time()
    base, previous, locked_path, out = (
        Path(x).resolve() for x in (base, previous, locked_path, out)
    )
    out.mkdir(parents=True, exist_ok=False)
    locked = json.loads(locked_path.read_text())
    p, rho = np.asarray(locked["P"], dtype=float), float(locked["rho"])
    if not locked.get("join_validation_passed") or rho <= 0 or p.shape != (4, 4):
        raise ValueError("A saved validated fixed-point lock is required.")
    inputs = {locked_path}
    near_pairs, swing, rows = [], [], []
    near_file = base / "near/evidence.h5"
    inputs.update((near_file, base / "near/result.json", base / "near/config.json"))
    with h5py.File(near_file, "r") as new:
        for index in range(4):
            source = (
                previous
                / ("near-upright" if index < 2 else "near-upright-completion")
                / "evidence.h5"
            )
            inputs.add(source)
            group_name = f"trials/variant_0/trial_{index}"
            current = _controller_trace(new[group_name], f"ESS32 start {index}")
            with h5py.File(source, "r") as old:
                baseline = _controller_trace(old[group_name], f"lambda3 start {index}")
            if not np.array_equal(current["states"][0], baseline["states"][0]):
                raise ValueError("Near-upright restart mismatch.")
            if len(current["actions"]) != 500 or len(baseline["actions"]) != 500:
                raise ValueError("Near-upright duration differs from the frozen contrast.")
            near_pairs.append((baseline, current))
            rows.append(
                dict(
                    stage="near",
                    index=index,
                    objective="same original dense cost",
                    lambda3=_trace_description(baseline),
                    ess32=_trace_description(current),
                )
            )
    swing_file = base / "swing-development/evidence.h5"
    stage_result = base / "swing-development/result.json"
    inputs.update((swing_file, stage_result, base / "swing-development/config.json"))
    metadata = json.loads(stage_result.read_text())
    if len(metadata["trials"]) != 2:
        raise ValueError("Require both closed development attempts, including failures.")
    old_energy_file = previous / "energy/evidence.h5"
    old_energy_result = previous / "energy/result.json"
    inputs.update((old_energy_file, old_energy_result))
    historical_energy = json.loads(old_energy_result.read_text())
    with h5py.File(swing_file, "r") as data, h5py.File(old_energy_file, "r") as old_energy:
        for index in range(2):
            group = data[f"trials/variant_0/trial_{index}"]
            prefix = _controller_trace(group, f"MPPI prefix {index}")
            old_group = old_energy[f"trials/variant_0/trial_{index}"]
            old_prefix = _controller_trace(old_group, f"Historical energy prefix {index}")
            if not np.array_equal(prefix["states"][0], old_prefix["states"][0]):
                raise ValueError("Swing-up historical restart differs from current prefix.")
            suffix = (
                _controller_trace(group["lqr_suffix"], f"LQR suffix {index}")
                if "lqr_suffix" in group
                else None
            )
            info = next(x for x in metadata["trials"] if x["trial_index"] == index)
            joined = bool(info["joined"])
            if joined != (suffix is not None):
                raise ValueError("Join claim and saved suffix differ; no suffix is fabricated.")
            if suffix is not None and not np.array_equal(prefix["states"][-1], suffix["states"][0]):
                raise ValueError("LQR suffix reset/discontinuity at the join.")
            old_info = next(
                x
                for x in historical_energy["trials"]
                if x["recipe_index"] == 0 and x["trial_index"] == index
            )
            old_values, prefix_values = (
                _relative_value(old_prefix, p, rho),
                _relative_value(prefix, p, rho),
            )
            old_entries = np.flatnonzero(old_values[1:] <= 1)
            first_bank = {}
            for field in ("candidate_costs", "nominal_before"):
                current_bank = group["planning/control_0000/pass_000"][field][:]
                baseline_bank = old_group["planning/control_0000/pass_000"][field][:]
                first_bank[field + "_maximum_absolute_difference"] = (
                    float(np.max(np.abs(current_bank - baseline_bank)))
                    if current_bank.shape == baseline_bank.shape
                    else None
                )
            swing.append((old_prefix, prefix, suffix, info))
            rows.append(
                dict(
                    stage="swing-development",
                    index=index,
                    objective="energy-shaped prefix and same energy-cost LQR suffix",
                    historical_fixed_temperature=dict(
                        **_trace_description(old_prefix),
                        task_cost_with_one_terminal=float(old_info["task_cost"]),
                        minimum_V_over_already_frozen_rho=float(old_values.min()),
                        first_entry_control=int(old_entries[0] + 1) if len(old_entries) else None,
                        scope="Retrospective grading under independently frozen R1 join region; did not select that region.",
                    ),
                    current_prefix_minimum_V_over_frozen_rho=float(prefix_values.min()),
                    initial_bank_comparison=first_bank,
                    mppi_prefix=_trace_description(prefix),
                    lqr_suffix=_trace_description(suffix) if suffix is not None else None,
                    joined=joined,
                    join_control=info["join_control"],
                    hybrid_task_cost=float(info["task_cost"]),
                    running_cost=float(info["native_running_cost_sum"]),
                    single_final_terminal_cost=float(info["episode_terminal_cost"]),
                )
            )
    fig, axes = plt.subplots(4, 6, figsize=(18, 10), squeeze=False, sharex="col")
    colors = ("#777777", "#1765ab")
    for column, (baseline, current) in enumerate(near_pairs):
        for trace, color in zip((baseline, current), colors):
            times = trace["states"][:, 0] - trace["states"][0, 0]
            axes[0, column].plot(times, trace["height"], color=color, lw=0.9)
            axes[1, column].plot(times[1:], np.cumsum(trace["costs"]), color=color, lw=0.9)
            axes[2, column].plot(
                times[:-1], 2 * trace["actions"][:, 0], color=color, alpha=0.8, lw=0.7
            )
            axes[3, column].plot(times[:-1], trace["ess"], color=color, lw=0.8)
        axes[0, column].set_title(
            f"Near start {column}: both 10 s\nC dense: {baseline['costs'].sum():.1f} → {current['costs'].sum():.1f}",
            fontsize=8,
        )
        axes[1, column].set_ylabel("Cumulative dense cost", fontsize=7)
        for row in range(4):
            axes[row, column].axvspan(8, 10, color="#e8edf4", alpha=0.5)
    for j, (old_prefix, prefix, suffix, info) in enumerate(swing):
        column = 4 + j
        for trace, color in ((old_prefix, "#777777"), (prefix, "#1765ab"), (suffix, "#269465")):
            if trace is None:
                continue
            times = trace["states"][:, 0] - prefix["states"][0, 0]
            axes[0, column].plot(times, trace["height"], color=color, lw=1)
            axes[1, column].plot(
                times,
                np.log10(np.maximum(_relative_value(trace, p, rho), 1e-300)),
                color=color,
                lw=1,
            )
            axes[2, column].plot(times[:-1], 2 * trace["actions"][:, 0], color=color, lw=0.8)
            if len(trace["ess"]):
                axes[3, column].plot(times[:-1], trace["ess"], color=color, lw=0.9)
        axes[1, column].axhline(0, color="black", ls="--", lw=0.8)
        axes[1, column].set_ylabel("log10(V / join threshold)", fontsize=7)
        old_cost = rows[column]["historical_fixed_temperature"]["task_cost_with_one_terminal"]
        suffix_duration = len(suffix["actions"]) * 0.02 if suffix is not None else 0
        axes[0, column].set_title(
            f"Swing start {j}: MPPI {len(prefix['actions']) * 0.02:.2f} s, LQR {suffix_duration:.2f} s\n"
            f"C energy: {old_cost:.1f} → {info['task_cost']:.1f}; {'joined' if info['joined'] else 'no join'}",
            fontsize=8,
        )
        if suffix is not None:
            join_time = prefix["states"][-1, 0] - prefix["states"][0, 0]
            for row in range(4):
                axes[row, column].axvline(join_time, color="#269465", ls=":", lw=1)
        else:
            axes[0, column].text(
                0.03,
                0.1,
                "No LQR suffix recorded",
                transform=axes[0, column].transAxes,
                fontsize=7,
                color="#555555",
            )
    for column in range(6):
        axes[0, column].axhline(3.8, color="black", ls=":", lw=0.7)
        axes[0, column].set_ylim(-0.1, 4.2)
        axes[2, column].set_ylim(-2.2, 2.2)
        axes[3, column].axhline(32, color="black", ls="--", lw=0.7)
        axes[3, column].set_yscale("log")
        axes[3, column].set_ylim(0.8, 300)
        axes[3, column].set_xlabel("Executed time (s)", fontsize=7)
        for row in range(4):
            axes[row, column].grid(alpha=0.15)
            axes[row, column].tick_params(labelsize=6)
    for row, label in (
        (0, "Tip height (m)"),
        (2, "Motor torque 2u (Nm)"),
        (3, "Final-pass effective samples"),
    ):
        axes[row, 0].set_ylabel(label, fontsize=8)
    fig.suptitle(
        "Controller evidence: temperature changes weight spread; basin entry remains a separate physical outcome",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.012,
        "Gray: historical λ=3 (near) or λ=.15 (swing); blue: adaptive ESS32 MPPI; green: LQR only if recorded. Near cost/duration are identical definitions.\n"
        "Near dense and swing energy objectives have different definitions/scales; totals are not ranked across them. ESS is undefined during LQR. No new physics.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(
            out / f"controller_comparison.{suffix}", dpi=120, bbox_inches="tight", pad_inches=0.1
        )
    plt.close(fig)
    summary = dict(
        rows=rows,
        pairing_scope="Identical restart states; only the original first bank is paired. Later candidates/contexts change with executed controls.",
        missing_post_prefix_states_fabricated=False,
        nonlinear_basin_proof=False,
        logarithm_plot_floor=1e-300,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    receipt = dict(
        status="PASSED",
        source_sha256=sha(Path(__file__)),
        inputs={
            str(path): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(inputs)
        },
        cpu_seconds=time.process_time() - cpu,
        wall_seconds=time.monotonic() - start,
        new_physics_controls=0,
        new_network_forwards=0,
        new_fitting_updates=0,
    )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--locked", type=Path)
    args = parser.parse_args(argv)
    if args.previous is not None or args.locked is not None:
        if args.previous is None or args.locked is None:
            parser.error("Controller comparison requires both --previous and --locked.")
        print(
            json.dumps(
                render_controller_comparison(args.base, args.previous, args.locked, args.out)
            )
        )
        return 0
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
