"""The MPPI loop: MPPI explores, a similarity on where its rollouts go steers it, the policy learns.

Each round r:
1. Reference. Last round's medoid walk (the walk closest on average to the others) with share
   1 - s, and the states the deployed policy visits alone from the training starts while it stays
   up with share s, where s is the deployed policy's task score: the worse the policy does, the
   less MPPI follows it, and the more MPPI is pulled to agree with itself.
2. MPPI walks. MPPI (configs/walker2d_best.json) drives one walk of --length controls from each
   training start, in parallel, with the similarity in its rollout cost (similarity.py): each
   rollout pays w_eff x the share-weighted squared distance from its states to the nearest
   reference state. w_eff is fixed once from round 0 so that, at round 0's typical distance to
   its medoid, the pull equals --weight x the task cost per control (--weight 0: no similarity).
   Executed actions get Gaussian noise --dart; MPPI's planned actions are the labels.
3. Fit. Slope targets are capped LQR gains along each executed walk. Targets are mixed
   (1 - s) MPPI + s deployed policy for actions and slopes; the learner continues from its own
   weights with 2,000 updates falling to 200 as its score rises (learning rate decaying to 0).
4. Swap-in. The deployed copy is replaced only when the learner's task score is at least its own.
5. Judge the deployed copy (the task's judge) and log the walks' spread, cost and policy gap.

Round 0 is plain MPPI from the training starts (the first reference). Every round's random draws
are fixed by (seed, round), so --resume continues an interrupted run exactly. Kill rule: stop at
--kill-round if no round so far reached 16 of 32 on the task's success count.

Baseline: --dagger (with --weight 0) keeps everything else and changes how data is gathered:
the learner drives alone from each training start (fresh standing draws each time it falls) and
MPPI, its plan carried along each run, labels every state it visits; one walk's MPPI steps per
start (equal MPPI compute); data is aggregated over rounds.

Run: python scripts/run_loop.py --weight 1 --seed 1 --out runs/<name>  (writes only to --out:
rows.json, summary.json, walks_r.npz, policy_r.pt, learner_r.pt per round).
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from src.mppi_loop.dynamics import CAP, R, capped, linearize, lqr, policy_states, step, step_costs
from src.mppi_loop.fit import SLOPE, fit_rows, gap, mixed_targets
from src.mppi_loop.measures import Section, measure
from src.mppi_loop.similarity import medoid, reference_trees, similarity_cost

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "walker2d_best.json"
_ENV = None


def make_env(task, threads):
    if task == "walker":
        from src.envs.walker2d import Walker2d
        from src.mppi_loop.tasks.walker import WALKER_TASK

        return Walker2d(**WALKER_TASK, nthread=threads)
    if task == "hopper":
        from src.envs.hopper import HOPPER_TASK, Hopper

        return Hopper(**HOPPER_TASK, nthread=threads)
    raise ValueError(task)


def task_setup(task, args):
    if task == "walker":
        from src.mppi_loop.tasks import walker

        return walker.setup(args)
    if task == "hopper":
        from src.mppi_loop.tasks import hopper

        return hopper.setup(args)
    raise ValueError(task)


def _init_worker(task, threads):
    global _ENV
    torch.set_num_threads(1)
    _ENV = make_env(task, threads)


def mppi_walk(job):
    """One MPPI walk with the similarity in its cost and executed-action noise (in a worker)."""
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    env, w = _ENV, job["w_eff"]
    trees = reference_trees(job["refs"], job["obs"], job["sd"])
    prior = similarity_cost(trees, w, job["obs"], job["sd"])
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


def dagger_run(job):
    """DAgger data for one worker: the learner drives alone from job["x0s"] in turn (the next
    start each time it falls); MPPI, its plan carried along each run, labels every state it
    visits until job["budget"] labels. Slope targets: capped LQR gains along each executed run."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default="walker", choices=("walker", "hopper"))
    parser.add_argument("--weight", type=float, required=True, help="similarity (0: none)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--dart", type=float, default=0.1, help="executed-action noise sd")
    parser.add_argument("--fit-updates", type=int, default=2000)
    parser.add_argument("--copy-low", type=int, default=200)
    parser.add_argument("--no-mix", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--kill-round", type=int, default=15)
    parser.add_argument("--dagger", action="store_true", help="the DAgger baseline (--weight 0)")
    parser.add_argument("--dagger-draws", type=int, default=60, help="restart draws per worker")
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    parser.add_argument(
        "--resume", action="store_true", help="continue --out after its last saved round"
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
    task = task_setup(args.task, argparse.Namespace(
        threads=args.threads, quick=args.quick, bank=args.bank, seed=args.seed,
    ))  # fmt: skip
    env, spec, starts, current = task.env, task.spec, task.starts, task.current
    policy, c0, judge, task_score = task.policy, task.c0, task.judge, task.task_score
    obs = spec.observed
    bank = [s for _, s in current]
    sd = np.concatenate(bank)[:, obs].std(0) + 1e-6
    lqr_scale = np.concatenate(bank)[:, 1:].std(0) + 1e-6
    section = Section(bank, **task.legs) if not args.quick and task.legs is not None else None
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
                cap=CAP,
            )  # fmt: skip
            for k in range(len(starts))
        ]
        return sorted(pool.map(mppi_walk, jobs), key=lambda w: w["k"])

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
        standing draws (seeds 40,000,000 + 100,000 seed + 1,000 r + 100 k + i) each time it falls."""
        net = copy.deepcopy(p.network).cpu()
        mean = p.observation_mean.detach().cpu().numpy()
        std = p.observation_std.detach().cpu().numpy()
        jobs = [
            dict(
                k=k,
                x0s=[starts[k]]
                + [
                    draw(40_000_000 + 100_000 * args.seed + 1000 * r + 100 * k + i)
                    for i in range(args.dagger_draws)
                ],
                net=net,
                mean=mean,
                std=std,
                spec=spec,
                length=length,
                budget=length,
                cfg=cfg,
                seed=1_000_000 * args.seed + 1000 * r + k,
                lqr_scale=lqr_scale,
                free=spec.free,
                cap=CAP,
            )  # fmt: skip
            for k in range(len(starts))
        ]
        return sorted(pool.map(dagger_run, jobs), key=lambda w: w["k"])

    def rows_gap(p, X, U):
        """Per-action MSE between labels and the policy at their states (fit.gap's measure)."""
        pred = p.predict(spec.observation(X).astype(np.float32)).astype(np.float64)
        return float(np.mean((pred - U) ** 2))

    def gap_of(p, walks):
        return float(
            np.mean([gap(p, starts[w["k"]], w["states"], w["labels"], spec) for w in walks])
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

    def success(x):  # the task's count of judge starts done (Walker: standing walks)
        return x.get("success", x.get("standing_walks", 0))

    learner = policy
    deployed = copy.deepcopy(policy)
    start_round = 1
    if args.resume and (args.out / "rows.json").exists() and (args.out / "walks_0.npz").exists():
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
        with np.load(args.out / f"walks_{last}.npz") as z:
            walks = [dict(k=k, states=z[f"states_{k}"]) for k in range(len(starts))]
        with np.load(args.out / "walks_0.npz") as z:
            walks0 = [z[f"states_{k}"] for k in range(len(starts))]
        start_round = last + 1
        print("resumed after round", last, flush=True)
    else:
        learner_score = deployed_score = task_score(deployed)
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
    _, d_typ0 = medoid(walks0, obs, sd)
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
            m, d_typ = medoid([w["states"] for w in walks], obs, sd)
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
            U, Ks = mixed_targets(deployed, s, spec, X, U, Ks, SC)
        fit_r = int(round(args.fit_updates * (1.0 - learner_score) + args.copy_low * learner_score))
        learner = fit_rows(
            learner, X, U, Ks, SC, SLOPE, updates=fit_r, seed=r + 1000 * args.seed, spec=spec,
            lr_decay=True,
        )  # fmt: skip
        learner.eval()
        learner_score = task_score(learner)
        accepted = bool(learner_score >= deployed_score)
        if accepted:
            deployed, deployed_score = copy.deepcopy(learner), learner_score
        row = dict(
            round=r,
            deployed_score=deployed_score,
            learner_score=learner_score,
            accepted=accepted,
            fit_updates=fit_r,
            reference_medoid=m,
            reference_policy_share=s,
            policy_states=0 if own is None else int(len(own)),
            w_eff=w_eff,
            typical_distance_to_medoid=d_typ,
            gap_before_fit=gap_before,
            gap_after_fit=rows_gap(learner, new[0], new[1])
            if args.dagger
            else gap_of(learner, walks),
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
