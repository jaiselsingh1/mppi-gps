"""SAC continuation map: can the walking SAC oracle carry on from MPPI's walks?

Draft diagnostic; see docs/sac_continuation_protocol.md. Reads the K1 run's saved
MPPI teachers, SAC rollouts and SAC checkpoint (actor and twin critic) and writes
only to --out. No MPPI query, fitting or optimizer step happens here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import mujoco
import numpy as np
import torch
from torch import nn

from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK, LegacySACActor
from src.gps.mppi_gps.motion_diagnosis import segment_distances

K1 = Path("runs/2026-10-31_action_compatibility_k1_001")
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS
INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION
TARGET_VELOCITY = 1.5
WALK_MAE = 0.3
WINDOW = 25  # K1.2 fragment length, in controls
GAIT_WINDOW_FIRST = 100  # SAC windows start after its startup transient
VX = 10  # qvel[0] in the 19-value physics state


def load_teachers(k1: Path) -> list[dict]:
    teachers = []
    for path in sorted((k1 / "inputs" / "teachers").glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            teachers.append(
                dict(
                    id=path.stem,
                    states=z["states"],
                    observations=z["observations"],
                    actions=z["actions"],
                    integrations=z["integrations"] if "integrations" in z.files else None,
                    warmstarts=z["warmstarts"] if "warmstarts" in z.files else None,
                    costs=z["costs"] if "costs" in z.files else None,
                )
            )
    return teachers


def load_sac_rollouts(k1: Path) -> list[dict]:
    rollouts = []
    with h5py.File(k1 / "evidence" / "stage_b" / "raw.h5", "r") as f:
        for name in sorted(f["arrays"], key=lambda n: n.rsplit("_", 1)[-1]):
            if "episode_k1_primary_reference_" not in name:
                continue
            g = f["arrays"][name]
            rollouts.append(
                dict(
                    id="sac_" + name.rsplit("_", 1)[-1],
                    states=g["states"][:],
                    observations=g["observations"][:],
                    actions=g["actions"][:].astype(np.float64),
                    integrations=g["integrations"][:],
                    warmstarts=g["warmstarts"][:],
                    costs=g["costs"][:],
                )
            )
    return rollouts


def load_critic(checkpoint: Path) -> list[nn.Module]:
    agent = torch.load(checkpoint, map_location="cpu", weights_only=True)["agent"]
    nets = []
    for key in ("q1", "q2"):
        net = nn.Sequential(
            nn.Linear(23, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        net.load_state_dict({k.removeprefix("net."): v for k, v in agent[key].items()})
        nets.append(net.eval().requires_grad_(False))
    return nets


def critic_value(nets: list[nn.Module], obs: np.ndarray, act: np.ndarray) -> np.ndarray:
    """min(q1, q2) in SAC's reward units (reward = -cost, gamma 0.995, soft Q)."""
    x = torch.as_tensor(np.concatenate([obs, act], axis=1), dtype=torch.float32)
    with torch.no_grad():
        return torch.minimum(nets[0](x), nets[1](x)).squeeze(1).numpy().astype(np.float64)


def act(actor: LegacySACActor, obs: np.ndarray) -> np.ndarray:
    """Deterministic tanh(mean) in K1's batch-one order, so SAC replays itself exactly."""
    rows = torch.as_tensor(np.asarray(obs, dtype=np.float32))
    result = np.empty((len(rows), 6))
    with torch.no_grad():
        for k, row in enumerate(rows):
            mean, _ = actor(row.unsqueeze(0))
            action = torch.tanh(mean) * actor.action_scale + actor.action_bias
            result[k] = action.squeeze(0).numpy()
    return result


def full_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(model, FULL))
    mujoco.mj_getState(model, data, state, FULL)
    return state


def observation(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate([data.qpos[1:], np.clip(data.qvel, -10.0, 10.0)])


def restore(model: mujoco.MjModel, data: mujoco.MjData, traj: dict, t: int) -> bool:
    """Exact integration state + warm start where saved (as runtime.restore does).

    Returns True only for an exact restore; the 19-value fallback leaves the
    solver warm start to one mj_forward.
    """
    if traj["integrations"] is None:
        mujoco.mj_resetData(model, data)
        mujoco.mj_setState(model, data, traj["states"][t], FULL)
        mujoco.mj_forward(model, data)
        return False
    mujoco.mj_setState(model, data, traj["integrations"][t], INTEGRATION)
    mujoco.mj_forward(model, data)
    mujoco.mj_setState(model, data, traj["integrations"][t], INTEGRATION)
    data.qacc_warmstart[:] = traj["warmstarts"][t]
    return bool(np.array_equal(full_state(model, data), traj["states"][t]))


def rollouts(
    env: Walker2d,
    actor: LegacySACActor,
    starts: list[tuple[dict, int]],
    prefixes: list[np.ndarray],
    horizon: int,
) -> dict[str, np.ndarray]:
    """Run every start in lockstep: replay its prefix, then the SAC actor.

    The score is the native per-control cost through the first fall, plus the
    native terminal cost once and the unhealthy cost for each remaining control,
    so every arm is scored over exactly ``horizon`` controls.
    """
    model = env.model
    n = len(starts)
    datas = [mujoco.MjData(model) for _ in range(n)]
    exact = np.array([restore(model, d, traj, t) for d, (traj, t) in zip(datas, starts)])
    states = np.full((n, horizon + 1, 19), np.nan)
    actions = np.full((n, horizon, 6), np.nan)
    states[:, 0] = [full_state(model, d) for d in datas]
    fell_at = np.full(n, -1)
    alive = np.ones(n, dtype=bool)
    for c in range(horizon):
        index = np.flatnonzero(alive)
        if len(index) == 0:
            break
        u = np.empty((len(index), 6))
        need = []
        for j, i in enumerate(index):
            if c < len(prefixes[i]):
                u[j] = prefixes[i][c]
            else:
                need.append(j)
        if need:
            u[need] = act(actor, np.stack([observation(datas[index[j]]) for j in need]))
        for j, i in enumerate(index):
            data = datas[i]
            data.ctrl[:] = u[j]
            for _ in range(env._frame_skip):
                mujoco.mj_step(model, data)
            states[i, c + 1] = full_state(model, data)
            actions[i, c] = u[j]
            if not env._is_healthy(data.qpos[1], data.qpos[2]):
                fell_at[i] = c
                alive[i] = False

    running = env.running_cost(np.nan_to_num(states[:, 1:]), np.nan_to_num(actions))
    steps = np.arange(horizon)
    last = np.where(fell_at >= 0, fell_at, horizon - 1)
    running = np.where(steps[None, :] <= last[:, None], running, 0.0)
    fell = fell_at >= 0
    tail = np.where(fell, env._w_term + env._w_unhealthy * (horizon - 1 - fell_at), 0.0)
    return dict(
        states=states,
        actions=actions,
        running=running,
        cost=running.sum(axis=1) + tail,
        fell_at=fell_at,
        exact_restore=exact,
    )


def walking(states: np.ndarray, fell_at: np.ndarray, settle: int) -> tuple[np.ndarray, np.ndarray]:
    """K1's walking rule on the continuation: survive, velocity MAE <= 0.3 after settle."""
    error = np.abs(states[:, settle + 1 :, VX] - TARGET_VELOCITY)
    seen = np.isfinite(error).sum(axis=1)
    mae = np.where(seen > 0, np.nansum(error, axis=1) / np.maximum(seen, 1), np.nan)
    return (fell_at < 0) & (mae <= WALK_MAE), mae


def gait_windows(rollouts_: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    windows, owner = [], []
    for k, r in enumerate(rollouts_):
        physical = r["states"][:, 2:19]
        for s in range(GAIT_WINDOW_FIRST, len(r["actions"]) - WINDOW + 1, WINDOW):
            windows.append(physical[s : s + WINDOW])
            owner.append(k)
    return np.stack(windows), np.array(owner)


def nearest_gait(
    queries: np.ndarray, library: np.ndarray, owner: np.ndarray, exclude: np.ndarray | None = None
) -> np.ndarray:
    """K1.2 window distance R from each query to its nearest SAC window."""
    result = np.full(len(queries), np.nan)
    for q, window in enumerate(queries):
        if not np.isfinite(window).all():
            continue
        keep = np.ones(len(library), dtype=bool) if exclude is None else owner != exclude[q]
        tiled = np.broadcast_to(window, library[keep].shape)
        distances, _ = segment_distances(np.ascontiguousarray(tiled), library[keep])
        result[q] = distances.min()
    return result


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def describe(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    q = np.percentile(x, [10, 50, 90])
    return dict(
        n=int(len(x)),
        mean=float(x.mean()),
        p10=float(q[0]),
        median=float(q[1]),
        p90=float(q[2]),
        frac_negative=float((x < 0).mean()),
    )


def plot(
    path: Path,
    ids: list[str],
    grid: np.ndarray,
    outcome: np.ndarray,
    a25: np.ndarray,
    control_walk: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    status = ["#d03b3b", "#fab219", "#0ca30c"]  # falls, survives not walking, walks
    labels = ["SAC falls", "SAC survives, not walking", "SAC keeps walking"]
    diverging = LinearSegmentedColormap.from_list(
        "helps_hurts", ["#1c5cab", "#86b6ef", "#f0efec", "#ec8a89", "#b42c2c"]
    )
    rows = len(ids)
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 1.2 + 0.62 * rows), facecolor=surface, constrained_layout=True
    )
    extent = (grid[0] - 12.5, grid[-1] + 12.5, rows - 0.5, -0.5)
    for ax in axes:
        ax.set_facecolor(surface)
        ax.set_yticks(range(rows), ids, fontsize=8, color=ink)
        ax.tick_params(colors=muted, length=0)
        ax.set_xlabel("MPPI control where SAC takes over", color=muted, fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].imshow(
        outcome,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=ListedColormap(status),
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
    )
    axes[0].set_title(
        f"Can SAC carry on from MPPI's walk?  (SAC from its own states: {control_walk:.0%} walk)",
        loc="left",
        fontsize=10,
        color=ink,
    )
    axes[0].legend(
        handles=[Patch(color=c, label=s) for c, s in zip(status, labels)],
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        frameon=False,
        fontsize=8,
        labelcolor=ink,
    )

    limit = np.nanpercentile(np.abs(a25), 95) if np.isfinite(a25).any() else 1.0
    image = axes[1].imshow(
        a25,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=diverging,
        vmin=-limit,
        vmax=limit,
    )
    axes[1].set_title(
        "A25 = cost(25 MPPI controls, then SAC) − cost(SAC now);  "
        "blue: MPPI's actions help SAC, red: they hurt",
        loc="left",
        fontsize=10,
        color=ink,
    )
    bar = fig.colorbar(image, ax=axes[1], fraction=0.025, pad=0.01)
    bar.ax.tick_params(colors=muted, labelsize=8)
    bar.outline.set_visible(False)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k1", type=Path, default=K1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--settle", type=int, default=100)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="plumbing only: two SAC rollouts stand in for the MPPI teachers",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    sac_rollouts = load_sac_rollouts(args.k1)
    if args.smoke:
        teachers = [dict(r, id=r["id"] + "_as_teacher") for r in sac_rollouts[:2]]
        sac_rollouts = sac_rollouts[2:]
        args.horizon, args.settle, args.stride = 120, 20, 475
        out = args.out or Path("runs") / "sac_continuation_smoke"
    else:
        teachers = load_teachers(args.k1)
        out = args.out or Path("runs") / f"{time.strftime('%Y-%m-%d')}_sac_continuation_001"
    out.mkdir(parents=True, exist_ok=args.smoke)

    env = Walker2d(**NATIVE_TASK)
    checkpoint = args.k1 / "inputs" / "checkpoints" / "reference.pt"
    actor = LegacySACActor.from_checkpoint(checkpoint)
    critic = load_critic(checkpoint)
    rng = np.random.default_rng(args.seed)
    horizon, chunk = args.horizon, args.chunk

    grid = np.arange(0, len(teachers[0]["actions"]) - chunk + 1, args.stride)
    starts = [(traj, int(t)) for traj in teachers for t in grid]
    tags = np.array([traj["id"] for traj, _ in starts])
    random_actions = rng.uniform(-1.0, 1.0, size=(len(starts), 1, 6))
    arms = {
        "sac": [np.empty((0, 6))] * len(starts),
        "mppi1": [traj["actions"][t : t + 1] for traj, t in starts],
        "rand1": list(random_actions),
        "mppi25": [traj["actions"][t : t + chunk] for traj, t in starts],
    }
    results = {arm: rollouts(env, actor, starts, pre, horizon) for arm, pre in arms.items()}

    control_starts = [(r, int(t)) for r in sac_rollouts for t in grid]
    control = rollouts(
        env, actor, control_starts, [np.empty((0, 6))] * len(control_starts), horizon
    )

    # Fidelity: SAC replaying itself from its own exact states, and MPPI prefix replay.
    replay_dev = np.array(
        [
            np.max(
                np.abs(
                    control["states"][k, 1 : WINDOW + 1, 1:]
                    - r["states"][t + 1 : t + WINDOW + 1, 1:]
                )
            )
            if t + WINDOW < len(r["states"])
            else np.nan
            for k, (r, t) in enumerate(control_starts)
        ]
    )
    prefix_dev = np.array(
        [
            np.max(
                np.abs(results["mppi25"]["states"][k, chunk, 1:] - traj["states"][t + chunk, 1:])
            )
            if t + chunk < len(traj["states"])
            else np.nan
            for k, (traj, t) in enumerate(starts)
        ]
    )
    prefix_cost_dev = np.array(
        [
            np.max(np.abs(results["mppi25"]["running"][k, :chunk] - traj["costs"][t : t + chunk]))
            if traj["costs"] is not None and results["mppi25"]["fell_at"][k] < 0
            else np.nan
            for k, (traj, t) in enumerate(starts)
        ]
    )

    walks, mae = walking(results["sac"]["states"], results["sac"]["fell_at"], args.settle)
    survives = results["sac"]["fell_at"] < 0
    control_walks, _ = walking(control["states"], control["fell_at"], args.settle)
    a1 = results["mppi1"]["cost"] - results["sac"]["cost"]
    r1 = results["rand1"]["cost"] - results["sac"]["cost"]
    a25 = results["mppi25"]["cost"] - results["sac"]["cost"]
    flip_up = (results["mppi25"]["fell_at"] < 0) & ~survives
    flip_down = (results["mppi25"]["fell_at"] >= 0) & survives

    # Critic advantage at every saved MPPI row, and at the grid starts for comparison.
    critic_rows = {}
    for traj in teachers:
        obs = traj["observations"].astype(np.float32)
        critic_rows[traj["id"]] = critic_value(critic, obs, traj["actions"]) - critic_value(
            critic, obs, act(actor, obs)
        )
    critic_grid = np.array([critic_rows[traj["id"]][t] for traj, t in starts])

    library, owner = gait_windows(sac_rollouts)
    mppi_windows = np.stack([traj["states"][t : t + WINDOW, 2:19] for traj, t in starts])
    end_windows = results["sac"]["states"][:, horizon - WINDOW + 1 :, 2:19]
    r_mppi = nearest_gait(mppi_windows, library, owner)
    r_end = nearest_gait(end_windows, library, owner)
    control_end = control["states"][:, horizon - WINDOW + 1 :, 2:19]
    control_owner = np.array([k for k in range(len(sac_rollouts)) for _ in grid])
    r_control_end = nearest_gait(control_end, library, owner, exclude=control_owner)

    per_trajectory = []
    for traj in teachers:
        m = tags == traj["id"]
        per_trajectory.append(
            dict(
                id=traj["id"],
                exact_restore=bool(results["sac"]["exact_restore"][m].all()),
                starts=int(m.sum()),
                sac_survives=float(survives[m].mean()),
                sac_walks=float(walks[m].mean()),
                A1=describe(a1[m]),
                rand1_minus_sac=describe(r1[m]),
                A25=describe(a25[m]),
                mppi25_rescues=int(flip_up[m].sum()),
                mppi25_breaks=int(flip_down[m].sum()),
                R_mppi_to_sac_gait=describe(r_mppi[m]),
                R_end_to_sac_gait=describe(r_end[m]),
            )
        )
    summary = dict(
        protocol="docs/sac_continuation_protocol.md",
        smoke=args.smoke,
        k1_run=str(args.k1),
        horizon=horizon,
        settle=args.settle,
        stride=args.stride,
        chunk=chunk,
        seed=args.seed,
        starts=len(starts),
        control_starts=len(control_starts),
        overall=dict(
            sac_survives=float(survives.mean()),
            sac_walks=float(walks.mean()),
            A1=describe(a1),
            rand1_minus_sac=describe(r1),
            A25=describe(a25),
            mppi25_rescues=int(flip_up.sum()),
            mppi25_breaks=int(flip_down.sum()),
            spearman_critic_vs_minus_A1=spearman(critic_grid, -a1),
            critic_advantage_all_rows=describe(np.concatenate(list(critic_rows.values()))),
            R_mppi_to_sac_gait=describe(r_mppi),
            R_end_to_sac_gait=describe(r_end),
            spearman_R_mppi_vs_sac_cost=spearman(r_mppi, results["sac"]["cost"]),
        ),
        control=dict(
            sac_survives=float((control["fell_at"] < 0).mean()),
            sac_walks=float(control_walks.mean()),
            R_end_to_other_sac_rollouts=describe(r_control_end),
        ),
        fidelity=dict(
            exact_restores=int(results["sac"]["exact_restore"].sum()),
            control_exact_restores=int(control["exact_restore"].sum()),
            control_replay_max_state_dev=describe(replay_dev),
            mppi25_prefix_end_state_dev=describe(prefix_dev),
            mppi25_prefix_cost_dev=describe(prefix_cost_dev),
        ),
        per_trajectory=per_trajectory,
        seconds=time.perf_counter() - started,
    )

    np.savez_compressed(
        out / "results.npz",
        grid=grid,
        tags=tags,
        **{
            f"{arm}_{key}": value.astype(np.float32) if key == "states" else value
            for arm, res in results.items()
            for key, value in res.items()
            if key != "actions"
        },
        control_cost=control["cost"],
        control_fell_at=control["fell_at"],
        walks=walks,
        mae=mae,
        A1=a1,
        A25=a25,
        rand1_minus_sac=r1,
        critic_grid=critic_grid,
        R_mppi=r_mppi,
        R_end=r_end,
        R_control_end=r_control_end,
        replay_dev=replay_dev,
        prefix_dev=prefix_dev,
        prefix_cost_dev=prefix_cost_dev,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ids = [traj["id"] for traj in teachers]
    outcome = (survives.astype(int) + walks.astype(int)).reshape(len(ids), len(grid))
    plot(
        out / "continuation_map.png",
        ids,
        grid,
        outcome,
        a25.reshape(len(ids), len(grid)),
        float(control_walks.mean()),
    )
    print(
        json.dumps(
            dict(
                out=str(out),
                overall=summary["overall"],
                control=summary["control"],
                fidelity=summary["fidelity"],
                seconds=summary["seconds"],
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
