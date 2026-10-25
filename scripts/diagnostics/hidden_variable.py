"""Step 30: what does the carried-over plan choose, physically?

At Step 22's teacher anchors where MPPI's labels split into two clusters
(Step 23's test), take one recorded or regenerated warm start from each
cluster: the plans whose queries sit farthest apart along the first principal
axis. From the anchor's exact state (re-recorded with Step 22's seeds and
checked against its saved labels), run MPPI on from each plan for `steps`
controls with the same planner seed, no action noise. Compare the two branches:
- which foot touches down first after the anchor, and when;
- the share of controls whose foot contacts (right, left) differ;
- the joint-angle distance between the branches, against the distance after
  swapping one branch's legs (a leg swap) and the smallest distance over time
  shifts of up to `max_shift` controls (a phase shift).
Also, per bimodal anchor, the antisymmetric (opposite-leg) share of the
difference between the two clusters' mean labels.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

from scripts.diagnostics.label_modes import two_modes
from scripts.diagnostics.label_variance_split import SplitConfig, record_episode

_REPO_ROOT = Path(__file__).resolve().parents[2]
SWAP = [3, 4, 5, 0, 1, 2]


def antisymmetric_share(diff: np.ndarray) -> float:
    right, left = diff[:3], diff[3:]
    same, opposite = (right + left) / 2, (right - left) / 2
    return float(opposite @ opposite / (same @ same + opposite @ opposite))


def first_touchdown(contacts: np.ndarray) -> tuple[str, int]:
    """contacts (T, 2) booleans (right, left): the first foot to go from air to ground."""
    for t in range(1, len(contacts)):
        for foot, name in enumerate(("right", "left")):
            if contacts[t, foot] and not contacts[t - 1, foot]:
                return name, t
    return "none", len(contacts)


def branch_distances(a: np.ndarray, b: np.ndarray, max_shift: int) -> dict[str, float]:
    """Mean joint-angle distance between two branches (T, 6): as recorded, legs of b swapped, best time shift."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    plain = float(np.mean(np.linalg.norm(a - b, axis=1)))
    swapped = float(np.mean(np.linalg.norm(a - b[:, SWAP], axis=1)))
    shifted = min(float(np.mean(np.linalg.norm(a[s:] - b[:n - s], axis=1))) if s >= 0 else
                  float(np.mean(np.linalg.norm(a[:n + s] - b[-s:], axis=1)))
                  for s in range(-max_shift, max_shift + 1) if n - abs(s) > max_shift)
    return {"plain": plain, "legs_swapped": swapped, "best_time_shift": shifted}


def _branch(job: dict) -> dict:
    import mujoco

    from scripts.runners.run_walker2d import _foot_contacts
    from scripts.training.train_walker_mdgps import _walker
    from src.mppi.mppi import MPPI
    from scripts.diagnostics.label_variance_split import _planner_cfg

    cfg = job["cfg"]
    env = _walker()
    planner = MPPI(env, _planner_cfg(cfg), seed=0)
    env.reset(state=job["state"])
    env.set_warmstart(job["warmstart"])
    planner.restore({"U": job["U"], "initial_plan": False,
                     "rng_state": np.random.default_rng(job["seed"]).bit_generator.state})
    ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
           for name in ("floor", "foot_geom", "foot_left_geom")]
    angles, contacts, fell = [], [], False
    for _ in range(job["steps"]):
        action, _ = planner.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
        _, _, done, _ = env.step(action)
        angles.append(env.data.qpos[3:9].copy())
        contacts.append(_foot_contacts(env.model, env.data, *ids)[0])
        if done:
            fell = True
            break
    return {"key": job["key"], "angles": np.asarray(angles), "contacts": np.asarray(contacts, dtype=bool),
            "fell": fell}


def main(
    out_dir: Path,
    raw: Path = Path("runs/2026-10-24_label_variance_001/raw.npz"),
    steps: int = 125,
    max_shift: int = 25,
    seed: int = 9_500_000,
    workers: int = 16,
) -> None:
    data = np.load(raw)
    cfg = SplitConfig(**{k: v for k, v in json.loads((raw.parent / "config.json").read_text()).items()
                         if k in SplitConfig.__dataclass_fields__ and k not in ("populations",)})
    labels, warm_U = data["teacher_executed"], data["teacher_warm_U"]
    anchors = []
    for i, cloud in enumerate(labels.reshape(len(labels), -1, 6)):
        centred = cloud - cloud.mean(axis=0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        if two_modes(centred @ axis)["bimodal"]:
            per_warm = (labels[i] - cloud.mean(axis=0)) @ axis          # (W, R)
            means = per_warm.mean(axis=1)
            low, high = int(np.argmin(means)), int(np.argmax(means))
            split = np.median(centred @ axis)
            diff = cloud[centred @ axis > split].mean(axis=0) - cloud[centred @ axis <= split].mean(axis=0)
            anchors.append({"anchor": i, "episode": int(data["teacher_episode"][i]), "t": int(data["teacher_t"][i]),
                            "warm": (low, high), "antisymmetric_share": antisymmetric_share(diff)})
    out_dir.mkdir(parents=True, exist_ok=False)
    episodes = sorted({a["episode"] for a in anchors})
    with Pool(workers) as pool:
        records = dict(zip(episodes, pool.starmap(record_episode, [(cfg, "teacher", e) for e in episodes])))
        for a in anchors:
            recorded = records[a["episode"]]["labels"][a["t"]]
            if not np.array_equal(recorded, data["teacher_recorded_label"][a["anchor"]]):
                raise RuntimeError(f"re-recorded label differs at anchor {a['anchor']}")
        jobs = [{"cfg": cfg, "key": (a["anchor"], side), "state": records[a["episode"]]["states"][a["t"]],
                 "warmstart": records[a["episode"]]["warmstarts"][a["t"]], "U": warm_U[a["anchor"], w],
                 "seed": seed + a["anchor"], "steps": steps}
                for a in anchors for side, w in zip(("low", "high"), a["warm"])]
        branches = {r["key"]: r for r in pool.map(_branch, jobs)}
    rows = []
    for a in anchors:
        low, high = branches[(a["anchor"], "low")], branches[(a["anchor"], "high")]
        n = min(len(low["contacts"]), len(high["contacts"]))
        foot_low, t_low = first_touchdown(low["contacts"])
        foot_high, t_high = first_touchdown(high["contacts"])
        rows.append({**{k: a[k] for k in ("anchor", "episode", "t", "antisymmetric_share")},
                     "first_touchdown": [foot_low, foot_high], "touchdown_steps": [t_low, t_high],
                     "contact_mismatch": float(np.mean(np.any(low["contacts"][:n] != high["contacts"][:n], axis=1))),
                     "fell": [low["fell"], high["fell"]],
                     **branch_distances(low["angles"], high["angles"], max_shift)})
    feet = [r["first_touchdown"] for r in rows]
    both = [f for f in feet if "none" not in f]
    summary = {
        "bimodal_anchors": len(rows),
        "median_antisymmetric_share": float(np.median([r["antisymmetric_share"] for r in rows])),
        "different_first_foot": float(np.mean([f[0] != f[1] for f in both])) if both else float("nan"),
        "median_touchdown_gap_steps": float(np.median([abs(r["touchdown_steps"][0] - r["touchdown_steps"][1])
                                                      for r in rows])),
        "median_contact_mismatch": float(np.median([r["contact_mismatch"] for r in rows])),
        "median_distance_plain": float(np.median([r["plain"] for r in rows])),
        "median_distance_legs_swapped": float(np.median([r["legs_swapped"] for r in rows])),
        "median_distance_best_shift": float(np.median([r["best_time_shift"] for r in rows])),
        "swap_beats_plain": float(np.mean([r["legs_swapped"] < r["plain"] for r in rows])),
        "shift_explains_most": float(np.mean([r["best_time_shift"] < 0.5 * r["plain"] for r in rows])),
        "branches_fell": int(sum(sum(r["fell"]) for r in rows)),
    }
    (out_dir / "anchors.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
