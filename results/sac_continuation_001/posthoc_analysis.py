"""Post hoc analyses of the SAC continuation run, outside the frozen protocol.

Reads the run's results.npz and the K1 inputs it used and writes posthoc.json next
to this file. No physics, inference or fitting happens here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.diagnostics.sac_continuation import (
    K1,
    VX,
    WINDOW,
    describe,
    gait_windows,
    load_sac_rollouts,
    load_teachers,
    spearman,
)
from src.gps.mppi_gps.motion_diagnosis import segment_distances

RUN = Path("runs/2026-11-02_sac_continuation_001")
ARMS = ("sac", "mppi1", "rand1", "mppi25")
HORIZON = 500


def main() -> None:
    z = np.load(RUN / "results.npz")
    teachers, sac = load_teachers(K1), load_sac_rollouts(K1)
    grid, tags = z["grid"], z["tags"].astype(str)
    when = np.tile(grid, len(teachers))
    discovery = np.char.startswith(tags, "discovery")
    fell = {arm: z[f"{arm}_fell_at"] >= 0 for arm in ARMS}
    cost = {arm: z[f"{arm}_cost"] for arm in ARMS}

    # Falls: where, when, and whether the first action mattered.
    robust = fell["sac"] & fell["mppi1"] & fell["rand1"]
    standing = ~fell["sac"] & (z["mae"] > 1.3)
    slow = ~fell["sac"] & ~z["walks"] & ~standing
    falls = dict(
        per_arm={arm: int(f.sum()) for arm, f in fell.items()},
        sac_fall_controls_after_handover=sorted(z["sac_fell_at"][fell["sac"]].tolist()),
        sac_mppi1_rand1_all_fall=[[t, int(c)] for t, c in zip(tags[robust], when[robust])],
        flips={
            arm: dict(
                rescues=int((~fell[arm] & fell["sac"]).sum()),
                breaks=int((fell[arm] & ~fell["sac"]).sum()),
            )
            for arm in ("mppi1", "rand1", "mppi25")
        },
        sac_falls_by_restore=dict(
            exact=[int(fell["sac"][~discovery].sum()), int((~discovery).sum())],
            nineteen_value=[int(fell["sac"][discovery].sum()), int(discovery.sum())],
        ),
        standing=[[t, int(c)] for t, c in zip(tags[standing], when[standing])],
        standing_final_vx=z["sac_states"][standing, -1, VX].tolist(),
        slow_walking_mae=z["mae"][slow].tolist(),
        walking_mae=describe(z["mae"][z["walks"]]),
    )

    # A25 by construction: the exact 25-control replay lands on the next grid start.
    n = len(grid)
    f25, fsac = fell["mppi25"].reshape(-1, n), fell["sac"].reshape(-1, n)
    a25_identity = dict(
        mppi25_fall_equals_next_cell_sac_fall=int((f25[:, :-1] == fsac[:, 1:]).sum()),
        compared=int(f25[:, :-1].size),
    )

    # How much one action moves a 500-control continuation, on cells where no
    # one-action arm fell.
    quiet = ~(fell["sac"] | fell["mppi1"] | fell["rand1"])
    quiet25 = ~(fell["sac"] | fell["mppi25"])

    def spread(x: np.ndarray) -> dict:
        return dict(describe(x), median_abs=float(np.median(np.abs(x))))

    one_action = dict(
        cells=int(quiet.sum()),
        A1=spread(z["A1"][quiet]),
        rand1_minus_sac=spread(z["rand1_minus_sac"][quiet]),
        mppi1_minus_rand1=spread(cost["mppi1"][quiet] - cost["rand1"][quiet]),
        spearman_A1_vs_rand1_minus_sac=spearman(z["A1"][quiet], z["rand1_minus_sac"][quiet]),
        A25_cells=int(quiet25.sum()),
        A25=spread(z["A25"][quiet25]),
        sac_cost_per_control_walking=float(np.median(cost["sac"][z["walks"]]) / HORIZON),
        control_cost_per_control=float(np.median(z["control_cost"]) / HORIZON),
    )

    # What the gait distance is made of, at each MPPI window's nearest SAC window.
    library, _ = gait_windows(sac)
    queries = np.stack([t["states"][g : g + WINDOW, 2:19] for t in teachers for g in grid])
    parts = np.empty((len(queries), 3))
    for i, window in enumerate(queries):
        tiled = np.ascontiguousarray(np.broadcast_to(window, library.shape))
        distances, components = segment_distances(tiled, library)
        parts[i] = components[np.argmin(distances)]
    gait = dict(
        R_squared_share_height_angles_velocities=(parts / parts.sum(1, keepdims=True))
        .mean(0)
        .tolist(),
        mppi_mean_vx_after_100=[float(t["states"][100:, VX].mean()) for t in teachers],
        sac_mean_vx_after_100=describe(np.array([r["states"][100:, VX].mean() for r in sac])),
        R_mppi_at_robust_falls=z["R_mppi"][robust].tolist(),
    )

    prefix = z["prefix_dev"]
    fidelity = dict(
        prefix_end_state_max_dev=dict(
            exact=float(np.nanmax(prefix[~discovery])),
            nineteen_value=float(np.nanmax(prefix[discovery])),
        ),
        control_replay_max_dev=float(np.nanmax(z["replay_dev"])),
    )

    simulated = sum(
        int(np.where(z[f"{arm}_fell_at"] >= 0, z[f"{arm}_fell_at"] + 1, HORIZON).sum())
        for arm in ARMS
    ) + int(np.where(z["control_fell_at"] >= 0, z["control_fell_at"] + 1, HORIZON).sum())

    out = Path(__file__).with_name("posthoc.json")
    report = dict(
        run=str(RUN),
        falls=falls,
        a25_identity=a25_identity,
        one_action=one_action,
        gait=gait,
        fidelity=fidelity,
        simulated_controls=simulated,
    )
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
