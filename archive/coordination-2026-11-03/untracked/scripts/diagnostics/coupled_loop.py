"""Coupled loop: MPPI explores near the current policy, and the policy learns from what MPPI tried there.

policy_picks showed that labels picked by the policy from unchanged MPPI's tries
get their consistency only from the policy: MPPI tried things around its own
plan, about 1.2 sd per torque from the policy, so its tries said nothing
state-determined about what is better near the policy. The matching term's job
here is coverage, not consistent torques: it keeps MPPI's exploration near the
policy while the task cost keeps it walking. Each round MPPI is coupled to the
newest policy (never seeded from it), everything it evaluates is recorded, and
the policy refits. Writes only to --out.

Design, fixed before the first run:

- Start: the policy is the seed-0 S1 clone (10,000 updates) of the sample bank's
  9 unchanged MPPI walks (policy_picks round 0; it does not walk).
- Each round: standard Walker MPPI plus coupled_mppi's torque-matching term
  toward the current policy at w = 0.3 (walked 6/6 at -3% cost toward a falling
  student), reset seeds 0-5, 1,000 controls, recording as sample_bank.
- Arms, each with its own chain of policies and MPPI runs, 3 rounds:
  "MPPI's torques" (fit on the coupled MPPI's executed actions, the 2015 recipe)
  and "policy picks" (fit on policy_picks labels, h 0.3, lam 10, from MPPI's
  recorded tries, relabeled each round with the current policy).
- Training data: all rounds so far of the arm, surviving episodes, all controls.
  Students: S1 recipe, 10,000 updates, seeds 0 and 1; seed 0 becomes the policy.
- Measures per round: MPPI survival, K1 walking and cost of survivors;
  coverage, the mean distance between the policy's action and MPPI's sampling
  centre (unchanged MPPI: 0.85); label consistency and consistency of the label
  minus the policy's action (the "better" part; unchanged MPPI with policy
  picks: -0.15); label beats the policy's action in rollouts (as policy_picks);
  students walking from S1's 32 standing starts; takeover survival from the
  round's MPPI states.
- Readings: "coupling puts MPPI's tries near the policy" if round-1 coverage is
  <= 0.43; "MPPI's tries teach something state-determined" if the "better" part
  reaches consistency >= 0.2; "fixed" if an arm's students walk >= 50% by round
  3 while MPPI survives >= 5/6 at <= 1.2x unchanged cost (402.7).
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

from scripts.diagnostics.action_spectrum import FIRST, neighbour_index, state_dependence
from scripts.diagnostics.coupled_mppi import CONFIG, SEEDS, STEPS, WALK_MAE, matching_cost
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.policy_picks import better, fit, load_bank, relabel
from scripts.diagnostics.sac_continuation import K1, load_teachers
from scripts.diagnostics.sample_bank import RecordingMPPI
from scripts.diagnostics.smooth_teacher_students import EVAL_SEEDS
from scripts.diagnostics.stability_terminal import takeover_from
from scripts.diagnostics.student_positive_control import standing
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.utils.config import MPPIConfig

WEIGHT = 0.3
H, LAM = 0.3, 10.0
ROUNDS = 3
STUDENT_SEEDS = (0, 1)
TAKEOVERS = tuple(range(100, 901, 50))
UNCHANGED_COST = 402.668


def run(env, controller, target, seed, steps) -> dict:
    np.random.seed(seed)
    env.reset()
    controller.reset(seed=10_000 + seed)
    keep: dict[str, list] = {}
    done, total = False, 0.0
    for _ in range(steps):
        state = env.get_state()
        warm = env.get_warmstart()
        action, _ = controller.plan_step(
            state,
            prior_cost=matching_cost(env, target, state, WEIGHT),
            initial_warmstart=warm,
        )
        for k, v in dict(
            states=state, warmstarts=warm, actions=action, **controller.record
        ).items():
            keep.setdefault(k, []).append(v)
        _, cost, done, _ = env.step(action)
        total += cost
        if done:
            break
    out = {k: np.asarray(v) for k, v in keep.items()}
    vx = np.append(out["states"][1:, 10], env.get_state()[10])
    out["survived"] = np.array(not done and len(out["actions"]) == steps)
    out["walks"] = np.array(bool(out["survived"]) and np.mean(np.abs(vx - 1.5)) <= WALK_MAE)
    out["cost"] = np.array(total)
    return out


def consistency(episodes, labels, mu, sd, rng) -> float:
    index = neighbour_index([(observation(e["states"])[FIRST:] - mu) / sd for e in episodes])
    return state_dependence(np.concatenate([y[FIRST:] for y in labels]), index, rng)["r2"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    env = Walker2d(**NATIVE_TASK)
    pooled = np.concatenate([t["observations"][FIRST:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    config = MPPIConfig(**json.loads(CONFIG.read_text()))
    seeds = SEEDS[:2] if args.quick else SEEDS
    steps = 160 if args.quick else STEPS
    rounds = 1 if args.quick else ROUNDS
    student_seeds = STUDENT_SEEDS[:1] if args.quick else STUDENT_SEEDS
    eval_seeds = EVAL_SEEDS[:2] if args.quick else EVAL_SEEDS

    bank = load_bank(args.bank, every=args.quick)[: 2 if args.quick else None]
    start = fit(bank, [e["actions"] for e in bank], 0, args.quick)

    rows = []
    for arm in ("MPPI's torques", "policy picks"):
        target, data = start, []
        for r in range(1, rounds + 1):
            controller = RecordingMPPI(env, config)
            episodes = [run(env, controller, target, s, steps) for s in seeds]
            survivors = [e for e in episodes if bool(e["survived"])]
            data += survivors
            p_round = [
                target.predict(observation(e["states"]).astype(np.float32)) for e in survivors
            ]
            if arm == "MPPI's torques":
                labels_round = [e["actions"] for e in survivors]
                labels_all = [e["actions"] for e in data]
            else:
                labels_round, _ = relabel(survivors, target, H, LAM)
                labels_all, _ = relabel(data, target, H, LAM)
            row = dict(
                arm=arm,
                round=r,
                survives=len(survivors),
                k1_walks=int(sum(bool(e["walks"]) for e in episodes)),
                survivor_cost=(
                    float(np.mean([float(e["cost"]) for e in survivors])) if survivors else None
                ),
            )
            if len(survivors) >= 2:
                centre = np.concatenate([e["nominal_first"] for e in survivors])
                p = np.concatenate(p_round)
                row.update(
                    coverage=float(np.linalg.norm(p - centre, axis=1).mean()),
                    label_consistency=consistency(survivors, labels_round, mu, sd, rng),
                    better_part_consistency=consistency(
                        survivors, [y - q for y, q in zip(labels_round, p_round)], mu, sd, rng
                    ),
                    **better(env, survivors, labels_round, target),
                )
                students = [fit(data, labels_all, s, args.quick) for s in student_seeds]
                walks = [int(sum(standing(env, s, x) for x in eval_seeds)) for s in students]
                starts = [
                    e["states"][t] for e in survivors for t in TAKEOVERS if t < len(e["states"])
                ]
                row.update(
                    student_walks=walks,
                    student_walk_rate=float(sum(walks) / (len(students) * len(eval_seeds))),
                    takeover_survival=float(
                        np.mean(
                            [
                                takeover_from(env, students[0], x)
                                for x in starts[: 4 if args.quick else None]
                            ]
                        )
                    ),
                )
                target = students[0]
            rows.append(row)
            print(row, flush=True)
            np.savez_compressed(
                args.out / f"{'torques' if arm.startswith('MPPI') else 'picks'}_round{r}.npz",
                **{f"states_{i}": e["states"] for i, e in enumerate(episodes)},
                **{f"actions_{i}": e["actions"] for i, e in enumerate(episodes)},
            )
            if len(survivors) < 2:
                break

    first = [r for r in rows if r["round"] == 1 and "coverage" in r]
    summary = dict(
        rows=rows,
        readings=dict(
            coupling_puts_tries_near_policy=any(r["coverage"] <= 0.43 for r in first),
            tries_teach_something_state_determined=any(
                r.get("better_part_consistency", -1.0) >= 0.2 for r in rows
            ),
            fixed={
                arm: any(
                    r["arm"] == arm
                    and r["round"] == rounds
                    and r.get("student_walk_rate", 0.0) >= 0.5
                    and r["survives"] >= 5
                    and r["survivor_cost"] is not None
                    and r["survivor_cost"] <= 1.2 * UNCHANGED_COST
                    for r in rows
                )
                for arm in ("MPPI's torques", "policy picks")
            },
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["readings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
