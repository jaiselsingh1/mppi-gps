"""Create-only plots of saved fixed-reference trajectory agreement; no inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from pathlib import Path

from ..provenance import sha, write


def _csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def build_tables(result: dict) -> dict[str, list[dict]]:
    """Flatten the saved reduction without reweighting or recomputing predictions."""
    tables = {
        name: []
        for name in (
            "trajectories",
            "ordered_bins",
            "actuators",
            "marginals",
            "groups",
            "bin_covariance",
            "two_trajectory_differences",
        )
    }
    lookup = {row["id"]: row for row in result["trajectories"]}
    for row in result["trajectories"]:
        identity = {key: row[key] for key in ("index", "stage", "source", "id", "ancestry", "N")}
        tables["trajectories"].append(
            identity
            | {key: row[key] for key in ("E", "D", "bin_sd", "full_horizon")}
            | {"outcome_" + key: value for key, value in row["outcome"].items()}
        )
        for part in row["bins"]:
            tables["ordered_bins"].append(
                identity
                | {key: value for key, value in part.items() if key != "per_actuator_mse"}
                | {f"mse_{j}": value for j, value in enumerate(part["per_actuator_mse"])}
            )
        for j, error in enumerate(row["per_actuator_mse"]):
            value = identity | dict(
                actuator=j,
                mse=error,
                contribution_to_E=error / 6,
                error_share=None
                if row["actuator_error_shares"] is None
                else row["actuator_error_shares"][j],
            )
            for owner, marginal in row["marginals"].items():
                value.update(
                    {
                        owner + "_" + key: marginal[key][j]
                        for key in (
                            "mean_per_actuator",
                            "mean_abs_per_actuator",
                            "saturation_fraction_per_actuator",
                        )
                    }
                )
            tables["actuators"].append(value)
        for owner, marginal in row["marginals"].items():
            tables["marginals"].append(
                identity
                | dict(owner=owner, threshold=row["saturation_threshold"])
                | {key: marginal[key] for key in ("mean_abs", "rms", "saturation_fraction")}
            )
    for group in result["groups"]:
        decomposition = group["decomposition"]
        tables["groups"].append(
            {
                key: group[key]
                for key in (
                    "source",
                    "stage",
                    "count",
                    "descriptive_mixture",
                    "balanced",
                    "E_mean",
                    "E_sd",
                    "E_min",
                    "E_max",
                    "E_spread",
                )
            }
            | dict(
                N_values=json.dumps(group["N_values"]),
                trajectory_ids=json.dumps(group["trajectory_ids"]),
            )
            | (
                {}
                if decomposition is None
                else {
                    key: value for key, value in decomposition.items() if key != "covariance_matrix"
                }
            )
        )
        if decomposition is not None:
            bins = lookup[group["trajectory_ids"][0]]["bins"]
            for i, values in enumerate(decomposition["covariance_matrix"]):
                for j, value in enumerate(values):
                    tables["bin_covariance"].append(
                        dict(
                            source=group["source"],
                            first_start=bins[i]["start_control"],
                            first_stop=bins[i]["stop_control_exclusive"],
                            second_start=bins[j]["start_control"],
                            second_stop=bins[j]["stop_control_exclusive"],
                            covariance_ddof0=value,
                        )
                    )
        difference = group.get("two_trajectory_difference")
        if difference is not None:
            tables["two_trajectory_differences"].extend(
                dict(
                    source=group["source"],
                    first=difference["first"],
                    second=difference["second"],
                    full_delta=difference["full_delta"],
                    **part,
                )
                for part in difference["bin_deltas"]
            )
    return tables


def _validate(result: dict, config: dict) -> None:
    if result["schema"] != "walker-trajectory-review-result-v1":
        raise ValueError("Expected the declared trajectory-review result")
    if result.get("execution_complete") is not True or result.get("error") is not None:
        raise ValueError("Incomplete saved analysis cannot be rendered as complete")
    if not math.isfinite(config["control_seconds"]) or config["control_seconds"] <= 0:
        raise ValueError("Explicit positive control duration required")
    rows = result["trajectories"]
    if (
        len(rows) != result["totals"]["trajectories"]
        or sum(row["N"] for row in rows) != result["totals"]["rows"]
        or sum(len(row["bins"]) for row in rows) != result["totals"]["bins"]
        or 6 * len(rows) != result["totals"]["actuator_entries"]
    ):
        raise ValueError("Incomplete saved trajectory roster")
    for index, row in enumerate(rows):
        cursor = 0
        if row["index"] != index or len(row["per_actuator_mse"]) != 6:
            raise ValueError("Preserved trajectory order and six actuators required")
        for part in row["bins"]:
            if (
                part["start_control"] != cursor
                or part["stop_control_exclusive"] - cursor != part["rows"]
                or part["rows"] <= 0
                or len(part["per_actuator_mse"]) != 6
            ):
                raise ValueError("Bins must preserve contiguous executed order without padding")
            cursor = part["stop_control_exclusive"]
        if cursor != row["N"]:
            raise ValueError("Missing or padded saved controls")


def _figures(result: dict, config: dict, out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="walker-trajectory-mpl-"))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="walker-trajectory-font-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch, Rectangle

    rows = result["trajectories"]
    labels = [f"{row['source']}/{row['id']} (N={row['N']})" for row in rows]
    full_controls, seconds = config["full_controls"], config["control_seconds"]

    def save(figure, name):
        figure.savefig(out / (name + ".png"), dpi=140, bbox_inches="tight")
        figure.savefig(out / (name + ".pdf"), bbox_inches="tight")
        plt.close(figure)

    def time_axis(axis):
        axis.set_xlim(0, full_controls)
        axis.set_xlabel("Executed control index")
        axis.secondary_xaxis(
            "top", functions=(lambda x: x * seconds, lambda x: x / seconds)
        ).set_xlabel("Seconds from trajectory start")

    maximum = max(part["E"] for row in rows for part in row["bins"])
    norm = Normalize(0, max(maximum, 1e-12))
    figure, axis = plt.subplots(figsize=(16, max(5, len(rows) * 0.23)))
    axis.set_facecolor("0.86")
    for i, row in enumerate(rows):
        for part in row["bins"]:
            axis.add_patch(
                Rectangle(
                    (part["start_control"], i - 0.45),
                    part["rows"],
                    0.9,
                    facecolor=plt.cm.viridis(norm(part["E"])),
                    linewidth=0,
                )
            )
    axis.set_ylim(len(rows) - 0.5, -0.5)
    axis.set_yticks(range(len(rows)), labels, fontsize=6)
    time_axis(axis)
    axis.set_title("Individual saved bin means E; one common color scale, no time warping")
    axis.legend(
        handles=[Patch(facecolor="0.86", label="No observed post-fall data")],
        loc="lower right",
        fontsize=7,
    )
    figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axis,
        label="Mean squared bounded-control error E",
    )
    save(figure, "individual_bin_errors")

    figure, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    # Reserve space above both panels for their secondary time axes and the suptitle.
    figure.subplots_adjust(top=0.74)
    groups = list(dict.fromkeys(row["source"] for row in rows))
    colors = {name: plt.cm.tab10(i % 10) for i, name in enumerate(groups)}
    for axis, reference in zip(axes, (False, True), strict=True):
        for row in rows:
            if row["full_horizon"] and (row["source"] == "qualified_reference") == reference:
                axis.plot(
                    [p["stop_control_exclusive"] for p in row["bins"]],
                    [p["cumulative_E"] for p in row["bins"]],
                    color=colors[row["source"]],
                    alpha=0.75,
                    linewidth=0.9,
                    label=f"{row['source']}/{row['id']}",
                )
        time_axis(axis)
        axis.set_title(
            "Reference self logs (consistency control)"
            if reference
            else "All successful MPPI trajectories"
        )
        axis.legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    axes[0].set_ylabel("Cumulative row-weighted mean E (not sum D)")
    figure.suptitle(
        "Full 1,000-control trajectories only; short random episodes remain in the other tables/figure"
    )
    save(figure, "cumulative_agreement")

    figure, axes = plt.subplots(1, 3, figsize=(17, max(5, len(rows) * 0.23)), sharey=True)
    values = [[value / 6 for value in row["per_actuator_mse"]] for row in rows]
    plots = [
        axes[0].imshow(
            values, aspect="auto", vmin=0, vmax=max(max(v) for v in values), cmap="viridis"
        )
    ]
    for axis, owner in zip(axes[1:], ("actions", "predictions"), strict=True):
        plots.append(
            axis.imshow(
                [row["marginals"][owner]["saturation_fraction_per_actuator"] for row in rows],
                aspect="auto",
                vmin=0,
                vmax=1,
                cmap="magma",
            )
        )
        axis.set_title(owner + ": marginal saturation only")
    axes[0].set_title("Actuator contribution to E: MSE / 6")
    axes[0].set_yticks(range(len(rows)), labels, fontsize=5)
    for axis in axes:
        axis.set_xticks(range(6))
        axis.set_xlabel("Bounded-control component")
    figure.colorbar(plots[0], ax=axes[0], label="Contribution in E units")
    figure.colorbar(
        plots[-1],
        ax=axes[1:].tolist(),
        label=f"Marginal fraction |u| >= {rows[0]['saturation_threshold']}",
    )
    figure.suptitle(
        "No per-trajectory renormalization; marginal saturation does not identify conditional error"
    )
    save(figure, "actuator_and_saturation")

    figure, axes = plt.subplots(1, 2, figsize=(17, 6))
    series = (
        (0, "var_between_full", -0.17, "C0", "Variance of full E"),
        (0, "mean_between_bin_var", 0.17, "C1", "Mean variance at each bin"),
        (1, "covariance_diag_contribution", -0.17, "C2", "Diagonal / B²"),
        (1, "covariance_offdiag_contribution", 0.17, "C3", "Signed off-diagonal / B²"),
    )
    for index, group in enumerate(result["groups"]):
        value = group["decomposition"]
        if value is None:
            for axis in axes:
                axis.text(
                    index,
                    0,
                    "not\ndecomposed",
                    rotation=90,
                    va="bottom",
                    ha="center",
                    fontsize=7,
                    color="0.45",
                )
            continue
        for panel, field, offset, color, label in series:
            axes[panel].bar(
                index + offset,
                value[field],
                0.34,
                color=color,
                label=label if index == 0 else None,
            )
        axes[1].scatter(
            index,
            value["var_between_full"],
            marker="x",
            color="black",
            label="Their sum = full variance" if index == 0 else None,
        )
    for axis in axes:
        axis.axhline(0, color="0.4", linewidth=0.6)
        axis.set_xticks(
            range(len(result["groups"])),
            [g["source"] + (" *" if g["descriptive_mixture"] else "") for g in result["groups"]],
            rotation=40,
            ha="right",
            fontsize=7,
        )
        axis.set_ylabel("Population variance/covariance of saved bin means (E²)")
        axis.legend(fontsize=7)
    figure.suptitle(
        "Exact descriptive covariance identity (ddof=0); * pooled MPPI is a dependent source mixture"
    )
    save(figure, "variance_decomposition")


def render(run: Path, out: Path) -> dict:
    run, out = run.resolve(), out.resolve()
    evidence_root = run.parent if (run.parent / "capsule.json").is_file() else run
    if out.is_relative_to(evidence_root):
        raise ValueError("Report output must be outside immutable run or capsule evidence")
    if (run / "run/result.json").is_file():
        run = run / "run"
    inputs = {name: sha(run / name) for name in ("config.json", "result.json")}
    config, result = [json.loads((run / name).read_text()) for name in inputs]
    _validate(result, config)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    tables = build_tables(result)
    for name, rows in tables.items():
        _csv(out / (name + ".csv"), rows)
    _figures(result, config, out)
    text = """# Fixed-reference along-trajectory review

This is a saved-metadata analysis. The chosen policy is evaluated at each trajectory's own recorded successive states; no reference rollout, network evaluation or simulation is performed here. E is the existing sum of six squared action errors divided by 6N. D=6NE is the corresponding duration-dependent sum. Actions use the unchanged bounded control scale. All executed rows, including a fall-causing action, remain present; post-fall observations stay missing.

Individual identities, ancestry, source, ordered 100-control means and cumulative means are retained. A short random episode occupies only its observed rectangle width. Its one partial bin cannot supply a temporal decomposition. Full-horizon MPPI/reference sums and short random sums must not be compared as matched-duration evidence. Reference-self zeros are a consistency control.

The variance decomposition is exact algebra over saved bin means, with ddof=0. The full-trajectory variance equals the sum of all bin covariances divided by B². Signed off-diagonal terms can offset diagonal variation. The two-trajectory table preserves first-minus-second bin differences and their full difference. Bins, actuators, paired source contexts and frames are not independent task replications; there are no p-values or a chosen small-variance threshold. This does not recover within-bin per-control variance or establish a causal explanation.

Actuator contributions use MSE/6 on a common scale. Marginal action/prediction magnitude and saturation are reported separately; they do not identify error conditional on saturation. No phase alignment, per-trajectory renormalization, new weighting, metric selection or ranking rule is introduced.

Agreement on teacher states is separate from walking, survival and complete cost when a student chooses its own actions and visits new states. The repository's `docs/student_source_s1_1_results.md` supplies that separate fixed-update evaluation context. Variation here is not teachability, task importance or a beneficial trajectory-optimization mechanism. The implemented planner guidance is finite-horizon candidate reweighting, not whole-episode CIO optimization.

![Individual ordered bins](individual_bin_errors.png)
![Cumulative means](cumulative_agreement.png)
![Actuator errors and separate marginal saturation](actuator_and_saturation.png)
![Variance decomposition](variance_decomposition.png)
"""
    (out / "report.md").write_text(text)
    if inputs != {name: sha(run / name) for name in inputs}:
        raise ValueError("Saved report input changed during rendering")
    receipt = dict(
        schema="walker-trajectory-review-report-v1",
        inputs=inputs,
        source_sha256=sha(Path(__file__)),
        tables={name: len(rows) for name, rows in tables.items()},
        outputs={
            p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
            for p in sorted(out.iterdir())
            if p.is_file()
        },
        wall_seconds=time.perf_counter() - started,
        scientific_calls=0,
    )
    write(out / "receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = render(args.run, args.out)
    print(json.dumps(dict(report=str(args.out), outputs=len(receipt["outputs"]))))
    return 0
