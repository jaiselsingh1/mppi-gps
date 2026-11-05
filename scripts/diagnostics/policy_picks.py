"""Policy picks: labels from MPPI's evaluated samples, chosen near what the policy already does.

Side-picking inside MPPI would stop it exploring. Here MPPI runs unchanged (the
sample bank) and the choice moves to the learner: at each state MPPI visited,
the label is the cost-weighted average of MPPI's own samples, weighted toward
the policy's current action and with MPPI's sampling centre (its carried plan,
the part set by history) divided out. The policy refits on these labels and the
step repeats, so each label is "close to the policy but better" as judged by
MPPI's own rollouts. MPPI is never seeded from or steered by the policy.
Reads the sample bank and writes only to --out.

Design, fixed before the first run:

- Data: the sample bank's walking episodes (K1 rule), all controls.
- Round 0: labels are MPPI's executed actions (plain cloning). Students: S1
  recipe, 10,000 updates, seeds 0 and 1; the seed-0 student is "the policy".
- Rounds 1-3: at each control, with the policy's action p, MPPI's samples' first
  actions u_k (pre-clip r_k), costs J_k and pre-update first action n, weights
  are exp(-(J_k - min J)/lam - |u_k - p|^2/(2h^2) + |r_k - n|^2/(2 sigma^2)),
  sigma = 0.3 (MPPI's first-action sampling sd); the label is their weighted
  mean of u_k. With h = sigma this is exactly MPPI's own update with its
  samples reweighted as if drawn around the policy's action instead of around
  its carried plan. Students are refit from scratch on the new labels.
- Settings (h, lam): (0.3, 3), (0.3, 10), (0.2, 3), (0.2, 10). lam 3 is MPPI's.
  (A plumbing run showed h 0.1 leaves about one sample per label in 6-D, so
  narrower kernels were dropped before the first real run.)
- Measures per round: label consistency (10-nearest-state R^2, controls 100+);
  label distance to the policy and to MPPI's action; weight n_eff; "better":
  at 90 controls (every 100th from 100 in each walk), the cost of MPPI's plan
  with its first action replaced by the label, by the policy's action and by
  MPPI's own action, from the restored state; students' walking from S1's 32
  standing starts; takeover survival (controls of 100) from the walks' states at
  controls 100, 150, ..., 900.
- Readings: "the policy can pick consistent labels" if some setting reaches
  label consistency >= 0.5 by round 3 while the label beats the policy's own
  action at >= 60% of checked controls; "learnable" if students walk >= 50% or
  takeover survival reaches 1.5x round 0's.
- Prediction, written before running: consistency rises well above MPPI's 4%
  because the labels follow a smooth policy; whether students walk is open,
  since the labels exist only at MPPI's states.
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
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.sac_continuation import K1, load_teachers
from scripts.diagnostics.smooth_teacher_students import EVAL_SEEDS, RECIPE
from scripts.diagnostics.stability_terminal import takeover_from
from scripts.diagnostics.student_positive_control import standing
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps import policy
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

SIGMA = 0.3
SETTINGS = ((0.3, 3.0), (0.3, 10.0), (0.2, 3.0), (0.2, 10.0))
ROUNDS = 3
STUDENT_SEEDS = (0, 1)
UPDATES = 10_000
CHECKS = tuple(range(100, 1000, 100))
TAKEOVERS = tuple(range(100, 901, 50))


def load_bank(folder: Path, every: bool = False) -> list[dict]:
    walks = []
    for path in sorted(folder.glob("episode_*.npz")):
        with np.load(path) as data:
            e = {k: data[k] for k in data.files}
        if every or bool(e["walks"]):
            walks.append(e)
    return walks


def relabel(walks, target, h, lam):
    labels, n_eff = [], []
    for e in walks:
        p = target.predict(observation(e["states"]).astype(np.float32))
        u = e["prefix"][:, :, 0].astype(np.float64)
        drift = e["raw_first"] - e["nominal_first"][:, None]
        cost = e["costs"] - e["costs"].min(axis=1, keepdims=True)
        logw = (
            -cost / lam
            - ((u - p[:, None]) ** 2).sum(-1) / (2 * h**2)
            + (drift**2).sum(-1) / (2 * SIGMA**2)
        )
        w = np.exp(logw - logw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        labels.append(np.einsum("tk,tka->ta", w, u))
        n_eff.append(1.0 / (w**2).sum(axis=1))
    return labels, np.concatenate(n_eff)


def fit(walks, labels, seed, quick):
    obs = np.concatenate([observation(e["states"]) for e in walks]).astype(np.float32)
    acts = np.concatenate(labels).astype(np.float32)
    normalization = dict(mean=obs.mean(0), std=np.maximum(obs.std(0), 1e-6))
    student, _, _ = policy.fit_policy(
        policy.Policy(seed, "cpu", normalization, hidden_dim=128),
        obs,
        acts,
        updates=50 if quick else UPDATES,
        seed=seed,
        **RECIPE,
    )
    student.eval()
    return student


def better(env, walks, labels, target):
    rows = []
    for e, lab in zip(walks, labels):
        for t in CHECKS:
            if t >= len(e["actions"]):
                continue
            mine = target.predict(observation(e["states"][t : t + 1]).astype(np.float32))[0]
            seqs = np.repeat(e["plan"][t][None], 3, axis=0)
            seqs[:, 0] = np.clip([lab[t], mine, e["actions"][t]], -1.0, 1.0)
            _, cost, _ = env.batch_rollout(
                e["states"][t], seqs, initial_warmstart=e["warmstarts"][t]
            )
            rows.append(cost)
    rows = np.array(rows)
    return dict(
        label_beats_policy=float(np.mean(rows[:, 0] < rows[:, 1])),
        median_label_minus_policy=float(np.median(rows[:, 0] - rows[:, 1])),
        median_label_minus_mppi=float(np.median(rows[:, 0] - rows[:, 2])),
    )


def judge(env, walks, labels, target, index, rng, students, quick) -> dict:
    y = np.concatenate([lab[FIRST:] for lab in labels])
    starts = [e["states"][t] for e in walks for t in TAKEOVERS if t < len(e["states"])]
    seeds = EVAL_SEEDS[:2] if quick else EVAL_SEEDS
    walks_per_student = [int(sum(standing(env, s, x) for x in seeds)) for s in students]
    return dict(
        consistency=state_dependence(y, index, rng)["r2"],
        **better(env, walks, labels, target),
        student_walks=walks_per_student,
        student_walk_rate=float(sum(walks_per_student) / (len(students) * len(seeds))),
        takeover_survival=float(
            np.mean([takeover_from(env, students[0], x) for x in starts[: 4 if quick else None]])
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"), help="sample bank"
    )
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    env = Walker2d(**NATIVE_TASK)
    pooled = np.concatenate([t["observations"][FIRST:] for t in load_teachers(K1)])
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    walks = load_bank(args.bank, every=args.quick)[: 2 if args.quick else None]
    if len(walks) < 2:
        raise SystemExit("need at least two walking episodes in the bank")
    index = neighbour_index([(observation(e["states"])[FIRST:] - mu) / sd for e in walks])
    seeds = STUDENT_SEEDS[:1] if args.quick else STUDENT_SEEDS

    labels = [e["actions"] for e in walks]
    students = [fit(walks, labels, s, args.quick) for s in seeds]
    round0 = dict(
        setting="MPPI's actions",
        round=0,
        **judge(env, walks, labels, students[0], index, rng, students, args.quick),
    )
    rows = [round0]
    print(rows[-1], flush=True)
    for h, lam in SETTINGS[:1] if args.quick else SETTINGS:
        target = students[0]
        for r in range(1, ROUNDS + 1):
            labels, n_eff = relabel(walks, target, h, lam)
            p = np.concatenate(
                [target.predict(observation(e["states"]).astype(np.float32)) for e in walks]
            )
            y = np.concatenate(labels)
            mppi = np.concatenate([e["actions"] for e in walks])
            nominal = np.concatenate([e["nominal_first"] for e in walks])
            new = [fit(walks, labels, s, args.quick) for s in seeds]
            rows.append(
                dict(
                    setting=f"h {h:g}, lam {lam:g}",
                    round=r,
                    mean_n_eff=float(n_eff.mean()),
                    share_n_eff_below_2=float(np.mean(n_eff < 2)),
                    label_to_policy=float(np.linalg.norm(y - p, axis=1).mean()),
                    label_to_mppi=float(np.linalg.norm(y - mppi, axis=1).mean()),
                    policy_to_sampling_centre=float(np.linalg.norm(p - nominal, axis=1).mean()),
                    **judge(env, walks, labels, target, index, rng, new, args.quick),
                )
            )
            print(rows[-1], flush=True)
            target = new[0]

    last = [r for r in rows if r["round"] == ROUNDS or (args.quick and r["round"] == 1)]
    summary = dict(
        walks=len(walks),
        rows=rows,
        readings=dict(
            policy_can_pick_consistent_labels=any(
                r["consistency"] >= 0.5 and r["label_beats_policy"] >= 0.6 for r in last
            ),
            learnable=any(
                r["student_walk_rate"] >= 0.5
                or r["takeover_survival"] >= 1.5 * round0["takeover_survival"]
                for r in rows[1:]
            ),
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["readings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
