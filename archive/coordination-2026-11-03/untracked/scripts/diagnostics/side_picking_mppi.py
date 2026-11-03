"""Side-picking MPPI: does averaging only one group of samples make MPPI's labels learnable?

flat_directions suggested MPPI's executed action sits between better options, so
the weighted average of samples from both sides lands in between and which side
dominates varies at random. ridge_check found that dip at about two-thirds of
states and about 5x shallower than the fit said. This tests the planner-side
fix directly: split each update's samples in two, keep one group by a fixed
rule, and average only within it. Writes only to --out.

Design, fixed before the first run:

- Planner: standard Walker MPPI (configs/walker2d_best.json), warm-started, K = 256.
- Split: each update, the samples are split at their weighted mean along the
  principal direction of their weighted noise over the first 8 controls
  (0.064 s), using MPPI's own softmin weights. The update averages only the
  chosen group, its weights renormalized.
- Rules: "heavier side" (the group with more total weight: MPPI's own
  preference, no policy involved); "policy side (faller)" (the group whose
  weighted-mean first action is nearer smooth-MPPI student 0's action);
  "policy side (walker)" (nearer the SAC clone's action; positive control).
- Episodes, baseline, measures and readings as mirrored_mppi (reset seeds 0-5,
  1,000 controls; survival, K1 walking, cost, consistency and students on
  surviving episodes). Also: mean n_eff after the split, the share of updates
  that keep the heavier side, and how far the executed first action moves from
  the ordinary average.
- Prediction, written before running: the dip is shallow next to MPPI's
  temperature (about 0.2 cost units over +-0.3 against lambda 3), so "heavier
  side" will not raise consistency; the policy rules may raise it about as much
  as the torque-matching term did (<= 20%), not to 90%.
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

import numpy as np

from scripts.diagnostics.coupled_mppi import CONFIG, SEEDS, STEPS, WALK_MAE
from scripts.diagnostics.mirrored_mppi import run_episode, summarize
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts, load_teachers
from scripts.diagnostics.smooth_teacher_students import ARMS, SOURCE
from scripts.diagnostics.student_failure import fit_students
from scripts.diagnostics.student_positive_control import fit_s1
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.math import effective_sample_size
from src.utils.trace_replay import load_episode

PREFIX = 8
VARIANTS = {
    "heavier side": None,
    "policy side (faller)": "faller",
    "policy side (walker)": "walker",
}


class SidePickingMPPI(MPPI):
    """MPPI that averages only one of two groups of samples each update."""

    def __init__(self, env, cfg, policy=None):
        super().__init__(env, cfg)
        self.policy = policy
        self.log: list[tuple[bool, float, float]] = []

    def plan_step(self, state, **kwargs):
        self._state = state
        return super().plan_step(state, **kwargs)

    def _sample_noise(self) -> np.ndarray:
        self._noise = super()._sample_noise()
        return self._noise

    def _softmin_weights(self, S: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
        w, _ = super()._softmin_weights(S, lam)
        x = self._noise[:, :PREFIX].reshape(self.K, -1)
        d = x - w @ x
        axis = np.linalg.eigh((d * w[:, None]).T @ d)[1][:, -1]
        side = d @ axis > 0
        groups = (side, ~side)
        mass = np.array([w[g].sum() for g in groups])
        heavier = int(np.argmax(mass))
        first = np.clip(self.U[0] + self._noise[:, 0], self.action_low, self.action_high)
        if self.policy is None:
            pick = heavier
        else:
            target = self.policy.predict(observation(self._state[None]).astype(np.float32))[0]
            gaps = [
                np.linalg.norm(w[g] @ first[g] / m - target) if m > 1e-12 else np.inf
                for g, m in zip(groups, mass)
            ]
            pick = int(np.argmin(gaps))
        out = np.where(groups[pick], w, 0.0)
        out /= out.sum()
        self.log.append(
            (pick == heavier, effective_sample_size(out), float(np.linalg.norm((out - w) @ first)))
        )
        return out, effective_sample_size(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("runs/2026-11-02_smooth_teacher_students_001")
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS), action="append", help="default: all")
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    env = Walker2d(**NATIVE_TASK)
    pooled = np.concatenate([t["observations"][100:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    config = MPPIConfig(**json.loads(CONFIG.read_text()))
    seeds = SEEDS[:2] if args.quick else SEEDS
    steps = 160 if args.quick else STEPS
    policies = {}

    baseline = []
    for k in range(10):
        e = load_episode(SOURCE / ARMS["default"] / "traces.npz", k)
        if e["reset_seed"] in seeds:
            vx = e["states"][1:, 10]
            survived = len(e["actions"]) == STEPS
            baseline.append(
                dict(
                    seed=e["reset_seed"],
                    states=e["states"][:-1],
                    actions=e["actions"],
                    survived=survived,
                    k1_walks=bool(survived and np.mean(np.abs(vx - 1.5)) <= WALK_MAE),
                    walks=survived,
                    cost=float(e["costs"].sum()),
                )
            )
    cells = [dict(variant="unchanged", **summarize(env, baseline, mu, sd, rng, args.quick))]
    print(cells[-1], flush=True)
    for name in args.variant or VARIANTS:
        which = VARIANTS[name]
        if which == "faller" and which not in policies:
            policies[which] = fit_students(args.previous / "summary.json")[0]
        if which == "walker" and which not in policies:
            sac = load_sac_rollouts(K1)
            policies[which] = fit_s1(
                np.concatenate([r["observations"] for r in sac[:9]]).astype(np.float32),
                np.concatenate([r["actions"] for r in sac[:9]]).astype(np.float32),
                0,
            )
        controller = SidePickingMPPI(env, config, policies.get(which))
        episodes = [run_episode(env, env, controller, None, s, steps) for s in seeds]
        log = np.array(controller.log)
        if args.quick:
            for e in episodes:
                e["walks"] = True  # let the short plumbing run exercise the analysis
        cells.append(
            dict(
                variant=name,
                share_heavier_side=float(log[:, 0].mean()),
                mean_n_eff=float(log[:, 1].mean()),
                mean_action_shift=float(log[:, 2].mean()),
                **summarize(env, episodes, mu, sd, rng, args.quick),
            )
        )
        print(cells[-1], flush=True)
        np.savez_compressed(
            args.out / f"{name.split(' (')[0].replace(' ', '_')}_{which or 'none'}.npz",
            **{f"states_{e['seed']}": e["states"] for e in episodes},
            **{f"actions_{e['seed']}": e["actions"] for e in episodes},
        )

    base_cost = cells[0]["survivor_cost"]
    readings = {}
    for c in cells[1:]:
        readings[f"much more consistent: {c['variant']}"] = bool(c.get("consistency", 0.0) >= 0.5)
        readings[f"learnable: {c['variant']}"] = bool(
            c["survives"] >= 5
            and c["survivor_cost"] is not None
            and base_cost is not None
            and c["survivor_cost"] <= 1.2 * base_cost
            and c.get("consistency", 0.0) >= 0.9
            and c.get("student_walk_rate", 0.0) >= 0.5
        )
    summary = dict(cells=cells, readings=readings, seconds=time.perf_counter() - started)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(readings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
