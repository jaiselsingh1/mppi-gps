"""Saved-only source/replay comparison; no dynamics, policy or producer imports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np


def _load(path: Path) -> tuple[dict, dict]:
    payload = path.read_bytes()
    if path.suffix == ".npz":
        with np.load(io.BytesIO(payload), allow_pickle=False) as raw:
            values = {name: raw[name].copy() for name in raw.files}
        for name, terminal in (
            ("states", "terminal_state"),
            ("warmstarts", "terminal_warmstarts"),
            ("integrations", "terminal_integrations"),
        ):
            if len(values[name]) == len(values["actions"]) and terminal in values:
                values[name] = np.concatenate((values[name], values[terminal][None]))
    else:
        import h5py

        with h5py.File(io.BytesIO(payload), "r") as raw:
            values = {name: raw[name][:] for name in raw}
    n = len(values["actions"])
    for name, shape in (
        ("states", (n + 1, 19)),
        ("warmstarts", (n + 1, 9)),
        ("integrations", (n + 1, 91)),
        ("actions", (n, 6)),
        ("costs", (n,)),
    ):
        if values[name].shape != shape or not np.isfinite(values[name]).all():
            raise ValueError("Finite actual N+1 replay evidence required: " + name)
    return values, dict(
        path=str(path.resolve()), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )


def _outcome(raw: dict, horizon: int) -> dict:
    post, costs = raw["states"][1:], raw["costs"]
    healthy = (post[:, 2] > 0.8) & (post[:, 2] < 2) & (np.abs(post[:, 3]) < 1)
    survival = bool(len(costs) == horizon and healthy.all())
    mae = float(np.mean(np.abs(post[:, 10] - 1.5))) if len(costs) else None
    return dict(
        controls=len(costs),
        requested_controls=horizon,
        raw_cost=float(costs.sum()),
        all_observed_controls_healthy=bool(healthy.all()),
        full_survival=survival,
        walking=bool(horizon == 1000 and survival and mae is not None and mae <= 0.3),
        velocity_mae=mae,
        first_unhealthy_control=int(np.flatnonzero(~healthy)[0]) if (~healthy).any() else None,
        recorded_fall_terminal_cost=float(raw["native_fall_cost"].sum())
        if "native_fall_cost" in raw
        else None,
    )


def replay_report(source: str | Path, replay: str | Path, out: str | Path) -> dict:
    """Render actual saved boundaries and controls; missing tails stay missing.

    CSV row i compares boundary x_i, and action/cost i when actually present.
    Boundary zero precedes the first control. Figures never interpolate a tail.
    This diagnoses replay agreement; it establishes no causal platform claim.
    """
    entered, cpu = time.monotonic(), time.process_time()
    source, replay, out = Path(source), Path(replay), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    original, source_binding = _load(source)
    actual, replay_binding = _load(replay)
    horizon = len(original["actions"])
    fields = (
        "boundary_index source_time replay_time source_control_present replay_control_present "
        "states_linf warmstarts_linf integrations_linf actions_linf costs_abs"
    ).split()
    rows, differences, first = [], {}, {}
    for name in ("states", "warmstarts", "integrations", "actions", "costs"):
        size = min(len(original[name]), len(actual[name]))
        delta = np.abs(original[name][:size] - actual[name][:size])
        differences[name] = delta if name == "costs" else delta.max(axis=1)
        values = differences[name]
        first[name] = {
            str(limit): int(np.flatnonzero(values > limit)[0]) if np.any(values > limit) else None
            for limit in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 0.01, 1.0)
        }
    for index in range(max(len(original["states"]), len(actual["states"]))):
        row = dict(
            boundary_index=index,
            source_time=float(original["states"][index, 0])
            if index < len(original["states"])
            else None,
            replay_time=float(actual["states"][index, 0])
            if index < len(actual["states"])
            else None,
            source_control_present=index < horizon,
            replay_control_present=index < len(actual["actions"]),
        )
        for name, value in differences.items():
            row[name + ("_abs" if name == "costs" else "_linf")] = (
                float(value[index]) if index < len(value) else None
            )
        rows.append(row)
    with (out / "boundary_differences.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for joint, axis in enumerate(axes.flat):
        for name, raw in (("source", original), ("replay", actual)):
            axis.plot(raw["states"][:, 0], raw["states"][:, 4 + joint], label=name, linewidth=0.8)
        axis.set_ylabel(f"joint {joint + 1} (rad)")
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("actual simulator time (s)")
    axes[-1, 1].set_xlabel("actual simulator time (s)")
    figure.suptitle("Saved source and replay joint angles")
    figure.tight_layout()
    figure.savefig(out / "joint_angles.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(11, 5))
    for name, values in differences.items():
        # Exact zeros are omitted on the logarithmic axis, never replaced by noise.
        axis.plot(
            original["states"][: len(values), 0] - original["states"][0, 0],
            np.where(values > 0, values, np.nan),
            label=name,
        )
    axis.axhline(1e-10, color="black", linestyle="--", label="frozen absolute tolerance1e-10")
    axis.set(
        yscale="log",
        xlabel="control-boundary time from start (s)",
        ylabel="maximum absolute difference",
        title="Saved replay errors; exact zeros omitted",
    )
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(out / "replay_errors.png", dpi=150)
    plt.close(figure)
    result = dict(
        schema="trajectory-pi-replay-report-1",
        status="SAVED_ONLY_COMPLETE",
        source=source_binding,
        replay=replay_binding,
        outcomes=dict(source=_outcome(original, horizon), replay=_outcome(actual, horizon)),
        initial_context_bytes_equal={
            name: original[name][0].dtype == actual[name][0].dtype
            and original[name][0].tobytes() == actual[name][0].tobytes()
            for name in ("states", "warmstarts", "integrations")
        },
        compared_rows={name: len(value) for name, value in differences.items()},
        maximum_absolute_error={
            name: float(value.max()) if len(value) else None for name, value in differences.items()
        },
        first_exceedance_index=first,
        missing_tail_policy="Unobserved rows stay blank; no interpolation or post-fall fabrication",
        wall_seconds=time.monotonic() - entered,
        cpu_seconds=time.process_time() - cpu,
        input_read_bytes=source_binding["bytes"] + replay_binding["bytes"],
        physical_disk_io_bytes=None,
        new_native_controls=0,
        new_model_controls=0,
        new_neural_rows=0,
        new_gradients=0,
        new_optimizer_updates=0,
        output_bytes_before_receipt=sum(path.stat().st_size for path in out.iterdir()),
    )
    with (out / "replay_report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    value = replay_report(args.source, args.replay, args.out)
    compact = {key: value[key] for key in ("schema", "status", "maximum_absolute_error")}
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
