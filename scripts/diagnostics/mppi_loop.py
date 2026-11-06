"""MPPI loop: MPPI explores, a similarity on where its rollouts go steers it, the policy learns.

Why this design (all measured 2026-11-05/06; see search_probe, getup_probe, mppi_reshape_probe,
mode_count, replay_ablation):
- The whole-trajectory refinement the controller loop used cannot make structural changes to a walk
  (at most 7% of the way toward another walk's states under every noise schedule, adaptive noise,
  weaker feedback, spline knots and windows), so every similarity term acting there measured as
  useless. MPPI (MPC) is the strong optimizer: it gets the walker up from lying down (3 of 3; the
  refinement 0 of 3) and made every walk.
- A similarity on the states MPPI's rollouts reach, added to its rollout cost, reshapes a walk
  beyond MPPI's own run-to-run variation, and pulling six walks toward one shared walk cuts their
  spread (B -22%, the visited-state distance -28% after one round, -39% after three) for 5-11% more
  task cost. Kendall (2026-11-02): guide MPPI by where its rollouts go, not by actions step by step.
- Six independent MPPI walks hold a spread of behaviours; a policy fit to them falls (plain mode,
  0 of 6 seeds); the locked stride of the cycle mode removed the spread by hand (6 of 6), which does
  not exist on tasks without a gait. The general version: choose one behaviour and make MPPI agree
  with it.
- Policy side, kept as verified in the controller loop: the learner keeps refitting while the
  deployed copy changes only when the task score improves (--learner); the targets lean on the
  deployed policy by its score (--mix); slope targets (LQR gains, weight 3); refit updates fall from
  2,000 to 200 with the score.

Each round r:
1. Reference. The medoid of round r-1's walks (the walk closest on average to the others, visited-
   state distance) and the states the deployed policy visits alone from the training starts while
   it stays up, weighted 1 - s and s by the deployed task score s (Kendall: the worse the policy does,
   the less MPPI copies it). Each MPPI rollout pays w_eff x (share-weighted) squared distance from
   each of its states to the nearest reference state (observed coordinates, sd units of the round-0
   walks), w_eff fixed once (re-normalizing each round would strengthen it as the walks converge)
   so the pull at round 0's typical distance to their medoid equals --weight x the task
   cost per control (--weight 0: the no-similarity control, MPPI's walks cloned each round).
2. MPPI (configs/walker2d_best.json, as the walks were made) drives one walk of --length controls
   from each training start, in parallel; executed actions get Gaussian noise --dart (MPPI corrects
   it; its planned actions are the labels), so the data holds recoveries.
3. Slope targets: capped LQR gains along each executed walk.
4. The learner refits (mixed targets, updates by its score, learning rate falling to 0); the
   deployed copy is replaced when the learner's task score is at least its own.
5. Judged on the deployed copy: walker_task's judge (32 standing starts, 51 mid-walk, 51 held-out
   mid-walk, stride), the task score; logged: the walks' spread (mode_count's B, W) and cost over the
   bank walks', the policy-walk gap before and after the refit.

Readings fixed before the runs (Walker, 3 seeds per arm, --weight 1 and the --weight 0 control):
"the loop teaches Walker to walk without a locked stride" if at least 2 of 3 --weight 1 seeds walk
from standing (deployed, >= 16 of 32 in >= 4 of the last 5 rounds) with a competent learner (at least
half the deployed score in those rounds); "the similarity is what does it" if the --weight 0 control
has at least 2 fewer such seeds. Kill rule, enforced here: stop at --kill-round if no round so far
walked from >= 16 of 32 standing starts. Writes only to --out.

Validation seeds 4-6 (both arms, added 2026-11-06 08:40 after seeds 1-3 showed 2 of 3 --weight 1
seeds walking by round 12 and 0 of 3 controls by round 15; same settings and per-seed criterion),
readings fixed before they ran, on all six seeds: "the loop works across seeds" if at least 4 of 6
--weight 1 seeds walk; "the similarity is what does it" if the control has at least 3 fewer.

--dagger (the DAgger baseline with MPPI as the expert, Kendall's bar; added 2026-11-06 09:50):
everything as the --weight 0 control (starting policy, round 0's plain MPPI walks, judge, task
score, kill rule, gate, mixed targets, slopes, refit schedule) except how data is gathered and
kept. Each round the learner drives alone from each training start, then from fresh standing
draws each time it falls; MPPI, its plan carried along each run as in a walk, labels every state
it visits, one walk's MPPI steps per start (equal MPPI compute per round); slope targets are
capped LQR gains along the executed runs. Data is aggregated (round 0's MPPI walks, then every
round's labels), and the refit uses all of it. Reading fixed before the runs (seeds 1-3): "the
loop beats DAgger" if DAgger has at least 2 fewer walking seeds (same per-seed criterion, or
stopped by the kill rule) than --weight 1 on seeds 1-3.

--task getup (added 2026-11-06 17:30; getup_task): the first task with no repeating cycle: from
lying on the ground (varied poses), get up and stand (4 s, --length 500), the dense get-up cost, no
fall rule; the judge counts the 32 held-out lying starts the policy gets up from ("success", which
the kill rule counts in place of standing walks). Readings fixed before the runs (3 seeds per arm,
--weight 1 and the --weight 0 control, --rounds 30, kill at round 15 if no round got up from >= 16
of 32): "the loop teaches get-up" if at least 2 of 3 --weight 1 seeds get up from >= 16 of 32 in >=
4 of the last 5 rounds with a competent learner (at least half the deployed score in those
rounds); "the similarity is what does it" if the control has at least 2 fewer such seeds. Run only
after getup_task's checks pass (MPPI gate, cloning check).

--gate-noise (added 2026-11-06 12:00): the learner replaces the deployed copy when its task score
with Gaussian noise of this sd on every executed action (fixed paired draws) is at least the
deployed copy's; the mix share, the reference share and the refit schedule still use the
noiseless score. Why: every walking checkpoint of seeds 1-3 (and every locked-stride walker) loses
most of its score under the data's own executed-action noise (sd 0.1: 0.11-0.31), and the
noiseless gate swapped seed 1's noise-robust round-12 policy (0.72 under noise) for a cheaper,
fragile one (0.21) that no refit could then hold (refit_probe, noise_score_probe). Side-by-side
test (seeds 1-3 resumed with --gate-noise 0.1 from the last round before each first walked, 8 /
11 / 13, to round 20), reading fixed before it ran: the noise gate "buys robustness without losing
walking" if all 3 walk from >= 16 of 32 standing starts at round 20 and, in at least 2 of 3 seeds,
the round-20 deployed copy beats the original run's round-20 deployed copy both on the task score
under sd 0.05 action noise and on 1.0 m/s shoves from 100 fresh starts (robustness_probe).
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

_ENV = None


def make_env(task, threads):
    if task == "walker":
        from src.envs.walker2d import Walker2d
        from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

        return Walker2d(**NATIVE_TASK, nthread=threads)
    if task == "hopper":
        from src.envs.hopper import HOPPER_TASK, Hopper

        return Hopper(**HOPPER_TASK, nthread=threads)
    if task == "getup":
        from src.envs.walker_getup import WalkerGetUp
        from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

        return WalkerGetUp(**NATIVE_TASK, nthread=threads)
    raise ValueError(task)


def _init_worker(task, threads):
    global _ENV
    torch.set_num_threads(1)
    _ENV = make_env(task, threads)


def _walk(job):
    """One MPPI walk with the state-set pull and executed-action noise (runs in a worker)."""
    from scipy.spatial import cKDTree

    from scripts.diagnostics.feedback_rollouts import linearize, lqr, step
    from scripts.diagnostics.feedback_tube import capped
    from scripts.diagnostics.similarity_loop import R
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    env, obs, sd = _ENV, job["obs"], job["sd"]
    trees = [
        (cKDTree(ref[:, obs] / sd), share)
        for ref, share in job["refs"]
        if share > 0 and ref is not None and len(ref)
    ]
    w = job["w_eff"]

    def prior(states, actions):  # (K, H, n) -> (K,)
        flat = states.reshape(-1, states.shape[-1])[:, obs] / sd
        total = 0.0
        for tree, share in trees:
            d, _ = tree.query(flat)
            total = total + share * d**2
        return w * np.reshape(total, states.shape[:2]).sum(1)

    planner = MPPI(env, MPPIConfig(**job["cfg"]))
    planner.reset(seed=job["seed"])
    rng = np.random.default_rng(job["seed"] + 17)
    x = np.array(job["x0"], dtype=float)
    xbar, labels, executed, states = [], [], [], []
    for _ in range(job["length"]):
        env.reset(state=x)
        u, _ = planner.plan_step(
            x,
            prior_cost=prior if (w > 0 and trees) else None,
            initial_warmstart=env.get_warmstart(),
        )
        u = np.asarray(u, dtype=float)
        ue = (
            np.clip(u + job["dart"] * rng.standard_normal(u.shape), -1.0, 1.0) if job["dart"] else u
        )
        xbar.append(x.copy())
        labels.append(u)
        executed.append(ue)
        x = step(env, x[None], ue[None])[0]
        states.append(x.copy())
        if not env._is_healthy(x[2:3], x[3:4]).all():
            break
    xbar, labels, executed, states = map(np.array, (xbar, labels, executed, states))
    K = capped(lqr(*linearize(env, xbar, executed, job["lqr_scale"]), R, job["free"]), job["cap"])
    return dict(xbar=xbar, labels=labels, executed=executed, states=states, K=K, k=job["k"])


def _dagger(job):
    """DAgger data for one worker (--dagger): the learner drives alone (no expert action is
    executed), from job["x0s"] in turn, the next start each time it falls; MPPI, its plan carried
    along each run as in a walk, labels every state the learner visits, until job["budget"]
    labels (a walk's MPPI steps). Slope targets: capped LQR gains along each executed run."""
    from scripts.diagnostics.feedback_rollouts import linearize, lqr, step
    from scripts.diagnostics.feedback_tube import capped
    from scripts.diagnostics.similarity_loop import R
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    env, spec, net = _ENV, job["spec"], job["net"]
    mean = torch.tensor(job["mean"], dtype=torch.float32)
    std = torch.tensor(job["std"], dtype=torch.float32)
    planner = MPPI(env, MPPIConfig(**job["cfg"]))
    xbar, labels, executed, states, K = [], [], [], [], []
    runs = 0
    for x0 in job["x0s"]:
        if len(labels) >= job["budget"]:
            break
        planner.reset(seed=job["seed"] + 7919 * runs)
        x = np.array(x0, dtype=float)
        run_x, run_e = [], []
        for _ in range(job["length"]):
            if len(labels) >= job["budget"]:
                break
            env.reset(state=x)
            u, _ = planner.plan_step(x, initial_warmstart=env.get_warmstart())
            with torch.no_grad():
                o = torch.tensor(spec.observation(x[None]).astype(np.float32))
                a = net((o - mean) / std).numpy()[0].astype(np.float64)
            a = np.clip(a, -1.0, 1.0)
            run_x.append(x.copy())
            run_e.append(a)
            labels.append(np.asarray(u, dtype=float))
            x = step(env, x[None], a[None])[0]
            states.append(x.copy())
            if not env._is_healthy(x[2:3], x[3:4]).all():
                break
        runs += 1
        if run_x:
            run_x, run_e = np.array(run_x), np.array(run_e)
            K.append(
                capped(lqr(*linearize(env, run_x, run_e, job["lqr_scale"]), R, job["free"]), job["cap"])
            )  # fmt: skip
            xbar.append(run_x)
            executed.append(run_e)
    return dict(
        xbar=np.concatenate(xbar), labels=np.array(labels), executed=np.concatenate(executed),
        states=np.array(states), K=np.concatenate(K), runs=runs, k=job["k"],
    )  # fmt: skip


def set_distance(a, b, obs, sd):
    from scipy.spatial import cKDTree

    d, _ = cKDTree(b[:, obs] / sd).query(a[:, obs] / sd)
    return float(np.sqrt((d**2).mean()))


def main() -> int:
    from scripts.diagnostics import controller_loop as cl
    from scripts.diagnostics.coupled_mppi import CONFIG
    from scripts.diagnostics.mode_count import measure
    from scripts.diagnostics.pulls import policy_states
    from scripts.diagnostics.slope_fit import fit_rows
    from scripts.diagnostics.stride_cycle import Section
    from scripts.diagnostics.trajectory_pi import step_costs

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default="walker", choices=("walker", "hopper", "getup"))
    parser.add_argument("--weight", type=float, required=True, help="similarity (0: control)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--dart", type=float, default=0.1, help="executed-action noise sd")
    parser.add_argument("--fit-updates", type=int, default=2000)
    parser.add_argument("--copy-low", type=int, default=200)
    parser.add_argument("--no-mix", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--kill-round", type=int, default=15)
    parser.add_argument(
        "--dagger", action="store_true",
        help="the DAgger baseline: the learner drives, MPPI labels its states, data aggregated "
        "(needs --weight 0)",
    )  # fmt: skip
    parser.add_argument("--dagger-draws", type=int, default=60, help="restart draws per worker")
    parser.add_argument(
        "--gate-noise", type=float, default=0.0,
        help="accept the learner on the task score with this executed-action noise sd (0: noiseless)",
    )  # fmt: skip
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    parser.add_argument(
        "--resume", action="store_true", help="continue an existing --out from its last saved round"
    )
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.dagger and args.weight != 0:
        parser.error("--dagger is the baseline without the similarity: use --weight 0")
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=args.resume)
    if (args.out / "killed.json").exists():
        print("this run was already stopped by the kill rule", flush=True)
        return 0
    torch.set_num_threads(args.threads)
    task = argparse.Namespace(
        threads=args.threads, quick=args.quick, bank=args.bank, budget=1.5, rounds=args.rounds,
        judge_every=1, seed=args.seed,
    )  # fmt: skip
    if args.task == "walker":
        out = cl.walker_task(task, None)
        legs = {}
    elif args.task == "getup":  # no stride: the stride section and its measures are off
        from scripts.diagnostics.getup_task import getup_task

        out = getup_task(task)
        legs = None
    else:
        from scripts.diagnostics.hopper_task import THIGH, THIGH_VEL, hopper_task

        out = hopper_task(task)
        legs = dict(thigh=THIGH, thigh_vel=THIGH_VEL)
    env, spec, starts, current, scales, policy = out[:6]
    c0, judge, task_score = out[7], out[8], out[13]
    obs = spec.observed
    bank = [s for _, s in current]
    sd = np.concatenate(bank)[:, obs].std(0) + 1e-6
    lqr_scale = np.concatenate(bank)[:, 1:].std(0) + 1e-6
    section = Section(bank, **legs) if not args.quick and legs is not None else None
    task_per_control = float(np.mean(c0)) / len(current[0][0])
    cfg = json.loads(CONFIG.read_text())
    length = 60 if args.quick else args.length

    pool = mp.get_context("spawn").Pool(args.workers, _init_worker, (args.task, 1))

    def walks_of(r, refs, w_eff):
        jobs = [
            dict(
                k=k,
                x0=starts[k],
                length=length,
                cfg=cfg,
                refs=refs,
                w_eff=w_eff,
                obs=obs,
                sd=sd,
                dart=args.dart,
                seed=1_000_000 * args.seed + 1000 * r + k,
                lqr_scale=lqr_scale,
                free=spec.free,
                cap=cl.CAP,
            )  # fmt: skip
            for k in range(len(starts))
        ]
        return sorted(pool.map(_walk, jobs), key=lambda w: w["k"])

    def draw(seed):
        saved = np.random.get_state()
        try:
            np.random.seed(seed)
            env.reset()
        finally:
            np.random.set_state(saved)
        return env.get_state()

    def dagger_of(r, p):
        """DAgger's round-r data: the learner p drives from each training start, then fresh
        standing draws (seeds 40,000,000 + 100,000 seed + 1,000 r + 100 k + i) each time it falls;
        MPPI labels its states, a walk's MPPI steps per worker (equal MPPI compute)."""
        net = copy.deepcopy(p.network).cpu()
        mean = p.observation_mean.detach().cpu().numpy()
        std = p.observation_std.detach().cpu().numpy()
        jobs = [
            dict(
                k=k,
                x0s=[starts[k]] + [
                    draw(40_000_000 + 100_000 * args.seed + 1000 * r + 100 * k + i)
                    for i in range(args.dagger_draws)
                ],
                net=net, mean=mean, std=std, spec=spec, length=length, budget=length, cfg=cfg,
                seed=1_000_000 * args.seed + 1000 * r + k, lqr_scale=lqr_scale, free=spec.free,
                cap=cl.CAP,
            )  # fmt: skip
            for k in range(len(starts))
        ]
        return sorted(pool.map(_dagger, jobs), key=lambda w: w["k"])

    def rows_gap(p, X, U):
        """Per-action MSE between labels and the policy at their states (cl.gap's measure)."""
        pred = p.predict(spec.observation(X).astype(np.float32)).astype(np.float64)
        return float(np.mean((pred - U) ** 2))

    def medoid(walks):
        n = len(walks)
        pair = np.array(
            [[set_distance(walks[i], walks[j], obs, sd) if i != j else 0.0 for j in range(n)]
             for i in range(n)]
        )  # fmt: skip
        m = int(np.argmin((pair + pair.T).sum(1)))
        return m, float(np.mean([pair[i, m] for i in range(n) if i != m]))

    def gap_of(p, walks):
        return float(
            np.mean([cl.gap(p, starts[w["k"]], w["states"], w["labels"], spec) for w in walks])
        )

    def walk_info(walks):
        info = dict(
            walks_up=int(sum(len(w["states"]) == length for w in walks)),
            cost_over_bank=[
                round(float(step_costs(env, w["states"][None], w["executed"][None]).sum()) / c0[w["k"]], 3)
                for w in walks
            ],
        )  # fmt: skip
        if section is not None:
            m = measure(section, [w["states"] for w in walks])
            info.update(spread_B=m.get("B"), spread_W=m.get("W"))
        return info

    learner = policy
    deployed = copy.deepcopy(policy)
    start_round = 1
    if args.resume and (args.out / "rows.json").exists() and (args.out / "walks_0.npz").exists():
        # continue after the last round whose rows, walks and policies were all saved; every
        # round's random draws are fixed by (seed, round), so the run goes on as it would have
        rows = json.loads((args.out / "rows.json").read_text())
        last = rows[-1]["round"]
        while last > 0 and not all(
            (args.out / f).exists()
            for f in (f"walks_{last}.npz", f"policy_{last}.pt", f"learner_{last}.pt")
            + ((f"dagger_{last}.npz",) if args.dagger else ())
        ):
            last -= 1
        rows = [x for x in rows if x["round"] <= last]
        if last > 0:
            deployed.load_state_dict(torch.load(args.out / f"policy_{last}.pt"))
            learner.load_state_dict(torch.load(args.out / f"learner_{last}.pt"))
            deployed.eval()
            learner.eval()
        deployed_score = rows[-1]["deployed_score"]
        learner_score = rows[-1].get("learner_score", deployed_score)
        deployed_gate = rows[-1].get("deployed_gate")
        with np.load(args.out / f"walks_{last}.npz") as z:
            walks = [dict(k=k, states=z[f"states_{k}"]) for k in range(len(starts))]
        with np.load(args.out / "walks_0.npz") as z:
            walks0 = [z[f"states_{k}"] for k in range(len(starts))]
        start_round = last + 1
        print("resumed after round", last, flush=True)
    else:
        learner_score = deployed_score = task_score(deployed)
        deployed_gate = None
        walks = walks_of(0, [], 0.0)  # round 0: plain MPPI, the first reference
        rows = [dict(round=0, deployed_score=deployed_score, **walk_info(walks), **judge(deployed))]
        print(rows[-1], round(time.perf_counter() - started), "s", flush=True)
        np.savez_compressed(
            args.out / "walks_0.npz", **{f"states_{w['k']}": w["states"] for w in walks}
        )
        if args.dagger:  # DAgger's first iteration: the expert's own walks
            np.savez_compressed(
                args.out / "dagger_0.npz",
                X=np.concatenate([w["xbar"] for w in walks]),
                U=np.concatenate([w["labels"] for w in walks]),
                K=np.concatenate([w["K"] for w in walks]),
            )
        (args.out / "rows.json").write_text(json.dumps(rows, indent=1) + "\n")
        walks0 = [w["states"] for w in walks]
    def success(x):  # the task's count of judge starts done (Walker: standing walks)
        return x.get("success", x.get("standing_walks", 0))

    walked = any(success(x) >= 16 for x in rows)
    if args.dagger:  # aggregated data: round 0's MPPI walks and every round's labels since
        parts = []
        for i in range(start_round):
            with np.load(args.out / f"dagger_{i}.npz") as z:
                parts.append((z["X"], z["U"], z["K"]))
        D = [np.concatenate(c) for c in zip(*parts)]
    # the pull's strength is fixed once, from round 0's walks: at their typical distance to their
    # medoid it equals --weight x the task cost per control. Re-normalizing every round would
    # strengthen it as the walks converge (1 / distance^2), a ratchet on the task cost.
    _, d_typ0 = medoid(walks0)
    w_eff = args.weight * task_per_control / max(d_typ0, 1e-9) ** 2
    for r in range(start_round, args.rounds + 1):
        s = float(min(max(deployed_score, 0.0), 1.0))
        if args.dagger:
            m = d_typ = own = None
            walks = dagger_of(r, learner)
            new = [np.concatenate([w[key] for w in walks]) for key in ("xbar", "labels", "K")]
            D = [np.concatenate([a, b]) for a, b in zip(D, new)]
            gap_before = rows_gap(learner, new[0], new[1])
            X, U, Ks = D
        else:
            m, d_typ = medoid([w["states"] for w in walks])
            own = []
            for k in range(len(starts)):  # states the deployed policy visits alone while up
                seen, up = policy_states(env, spec, deployed, starts[k], length)
                own.append(seen[up])
            own = np.concatenate(own) if any(len(o) for o in own) else None
            refs = [(walks[m]["states"], 1.0 - s), (own, s)]
            walks = walks_of(r, refs, w_eff)
            gap_before = gap_of(learner, walks)
            X = np.concatenate([w["xbar"] for w in walks])
            U = np.concatenate([w["labels"] for w in walks])
            Ks = np.concatenate([w["K"] for w in walks])
        SC = np.broadcast_to(lqr_scale, (len(U), len(lqr_scale))).copy()
        if not args.no_mix and s > 0:
            U, Ks = cl.mixed_targets(deployed, s, spec, X, U, Ks, SC)
        fit_r = int(round(args.fit_updates * (1.0 - learner_score) + args.copy_low * learner_score))
        learner = fit_rows(
            learner, X, U, Ks, SC, cl.SLOPE, updates=fit_r, seed=r + 1000 * args.seed, spec=spec,
            lr_decay=True,
        )  # fmt: skip
        learner.eval()
        learner_score = task_score(learner)
        if args.gate_noise:  # accept on the task done under the data's executed-action noise
            if deployed_gate is None:
                deployed_gate = task_score(deployed, noise=args.gate_noise)
            learner_gate = task_score(learner, noise=args.gate_noise)
            accepted = bool(learner_gate >= deployed_gate)
        else:
            learner_gate = None
            accepted = bool(learner_score >= deployed_score)
        if accepted:
            deployed, deployed_score = copy.deepcopy(learner), learner_score
            deployed_gate = learner_gate
        row = dict(
            round=r,
            deployed_score=deployed_score,
            learner_score=learner_score,
            accepted=accepted,
            **(dict(learner_gate=learner_gate, deployed_gate=deployed_gate) if args.gate_noise else {}),
            fit_updates=fit_r,
            reference_medoid=m,
            reference_policy_share=s,
            policy_states=0 if own is None else int(len(own)),
            w_eff=w_eff,
            typical_distance_to_medoid=d_typ,
            gap_before_fit=gap_before,
            gap_after_fit=rows_gap(learner, new[0], new[1]) if args.dagger else gap_of(learner, walks),
            **(
                dict(
                    labels=int(len(new[1])),
                    policy_runs=int(sum(w["runs"] for w in walks)),
                    controls_per_run=float(len(new[1]) / max(sum(w["runs"] for w in walks), 1)),
                    rows_total=int(len(D[0])),
                )
                if args.dagger
                else walk_info(walks)
            ),
            **judge(deployed),
        )
        rows.append(row)
        print({k: v for k, v in row.items()}, round(time.perf_counter() - started), "s", flush=True)
        torch.save(deployed.state_dict(), args.out / f"policy_{r}.pt")
        torch.save(learner.state_dict(), args.out / f"learner_{r}.pt")
        np.savez_compressed(
            args.out / f"walks_{r}.npz",
            **{f"states_{w['k']}": w["states"] for w in walks},
            **{f"labels_{w['k']}": w["labels"] for w in walks},
        )
        if args.dagger:
            np.savez_compressed(args.out / f"dagger_{r}.npz", X=new[0], U=new[1], K=new[2])
        (args.out / "rows.json").write_text(json.dumps(rows, indent=1) + "\n")
        walked = walked or success(row) >= 16
        if r == args.kill_round and not walked and not args.quick:
            (args.out / "killed.json").write_text(
                json.dumps(dict(round=r, rule="no round succeeded from >= 16 of 32 judge starts"))
            )
            print("stopped by the kill rule at round", r, flush=True)
            break
    pool.close()
    summary = dict(args={k: str(v) for k, v in vars(args).items()}, rows=rows,
                   seconds=time.perf_counter() - started)  # fmt: skip
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
