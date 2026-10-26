"""Credit test stack: can a global learner supply MPPI's missing ending?

Phases, each writing JSON under --out-dir (workers skip finished outputs):

  setup  exact values; testbed validity (V0) and representation (R0) on the
         reserved starts; decision banks from MPPI alone and MPPI with the
         exact terminal value; M-a: do MPPI's own options lead to different
         complete outcomes (exact, and branch-and-continue counterfactuals).
  sweep  round 1: pools of complete outcomes from MPPI alone; every estimator
         fitted on N outcomes; M-b decision metrics on the bank and M-c
         closed-loop costs on the reserved starts (select and terminal modes).
  api    MountainCar only: later rounds whose data come from each estimator's
         own controller (approximate policy iteration), at a fixed budget.
  report tables, N-needed and figures.

Protocol and predictions: docs/credit_testbed.md.
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.credit import estimators as est
from src.credit import study
from src.credit.control import branch
from src.credit.exact import CarValue, solve_car
from src.envs.mountain_car import MountainCar

REPO = Path(__file__).resolve().parents[2]
SOURCES = [
    "src/mppi/mppi.py",
    "src/envs/delayed_fork.py",
    "src/envs/mountain_car.py",
    "src/credit/exact.py",
    "src/credit/control.py",
    "src/credit/estimators.py",
    "src/credit/study.py",
    "scripts/credit/run_credit_stack.py",
]


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def task_keys(cfg: dict) -> list[str]:
    keys = [f"fork:{d}:{n:g}" for d in cfg["fork"]["delays_past_horizon"] for n in cfg["fork"]["noise_std"]]
    return keys + [f"car:{p:g}" for p in cfg["car"]["powers"]]


def build_task(cfg: dict, key: str, out_dir: Path) -> study.Task:
    kind, *rest = key.split(":")
    if kind == "fork":
        return study.fork_task(cfg["fork"], int(rest[0]), float(rest[1]))
    power = float(rest[0])
    spec = cfg["car"]
    cache = out_dir / "exact" / f"car_{power:g}.npz"
    env = MountainCar(power=power, **spec["env"])
    if cache.exists():
        z = np.load(cache)
        exact = CarValue(env, z["positions"], z["velocities"], z["grid"], z["actions"], int(z["iterations"]), float(z["residual"]))
    else:
        exact = solve_car(env, **spec["dp"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, positions=exact.positions, velocities=exact.velocities, grid=exact.grid,
                 actions=exact.actions, iterations=exact.iterations, residual=exact.residual)
    return study.car_task(spec, power, exact=exact)


def _family(key: str) -> str:
    return key.split(":")[0]


def _seeds(cfg: dict) -> dict:
    return cfg["common"]["seeds"]


def _safe(key: str) -> str:
    return key.replace(":", "_")


def _optimal_steps(task: study.Task) -> dict:
    """Simulate the DP-greedy policy from the middle reserved start (accuracy check)."""
    env, exact = task.env, task.exact
    start = task.reserved_start(0)
    s = env.reset(start, noise=np.zeros_like(env._noise))
    if hasattr(env, "set_time"):
        env.set_time(0)
    total, steps, done = 0.0, 0, False
    while not done:
        s, c, done, _ = env.step(np.array([exact.action(s)]))
        total += c
        steps += 1
    return {"start": start.tolist(), "dp_value": float(exact(start)), "greedy_cost": total, "greedy_steps": steps}


# ---------------------------------------------------------------------------
# setup: V0, R0, banks, M-a


def _branch_outcomes(task, controller, episodes, times, rng):
    """Commit-H branches of four of MPPI's options at the given decisions."""
    rows = []
    for i, ep in enumerate(episodes):
        n = len(ep.trajectory.actions)
        for t in times:
            if t >= n:
                continue
            opts = ep.trajectory.options[t]
            q = opts.model_cost + task.exact(opts.endpoints)
            picks = {
                "mean": opts.mean_index,
                "exact_best": int(np.argmin(q)),
                "exact_worst": int(np.argmax(q)),
                "random": int(rng.integers(len(q) - 1)),
            }
            outcome = {}
            for name, j in picks.items():
                br = branch(task.env, controller, ep, t, lambda o, c, j=j: j)
                outcome[name] = br.trajectory.outcome
            rows.append({"episode": i, "t": int(t), "q": {k: float(q[j]) for k, j in picks.items()},
                         "outcome": outcome, "headroom": float(q[opts.mean_index] - q.min()),
                         "spread": float(np.ptp(q))})
    return rows


def setup_worker(cfg: dict, key: str, out_dir: str) -> str:
    out_dir = Path(out_dir)
    path = out_dir / "setup" / f"{_safe(key)}.json"
    if path.exists():
        return f"{key}: exists"
    t0 = time.time()
    task = build_task(cfg, key, out_dir)
    fam = cfg[_family(key)]
    common = cfg["common"]
    seeds = _seeds(cfg)
    n_eval, n_bank = common["eval_episodes"], fam["bank_episodes"]
    ridge = common["ridge"]
    oracle = study.oracle_fit(task, study.representation_states(task), ridge=1e-3)
    controllers = {
        "alone": task.controller(),
        "exact_term": task.controller(terminal=task.exact),
        "exact_sel": task.controller(scorer=est.value_scorer(task.exact)),
        "oracle_term": task.controller(terminal=oracle),
        "oracle_sel": task.controller(scorer=est.value_scorer(oracle)),
        "greedy_sel": task.controller(scorer=est.greedy_scorer()),
        "random_sel": task.controller(scorer=est.random_scorer(seeds["reserved_planner"])),
    }
    closed, banked = {}, {}
    for name, ctrl in controllers.items():
        record = "full" if name in ("alone", "exact_term") else "none"
        eps = study.evaluate(task, ctrl, n_eval, seeds, record="none")
        if record == "full":
            banked[name] = study.evaluate(task, ctrl, n_bank, seeds, record="full")
        closed[name] = study.summarize(task, eps)
    bank = study.decision_bank(task, {"alone": banked["alone"], "exact_term": banked["exact_term"]}, stride=fam["bank_stride"])
    with open(out_dir / "banks" / f"{_safe(key)}.pkl", "wb") as fh:
        pickle.dump(bank, fh)
    reference_metrics = {
        "exact": study.decision_metrics(bank, est.value_scorer(task.exact)),
        "oracle": study.decision_metrics(bank, est.value_scorer(oracle)),
        "greedy": study.decision_metrics(bank, est.greedy_scorer()),
        "random": study.decision_metrics(bank, est.random_scorer(1)),
    }
    # M-a on the alone bank: exact headroom of MPPI's own options, by phase.
    phases = []
    for ep_i, ep in enumerate(banked["alone"]):
        for t, opts in enumerate(ep.trajectory.options):
            q = opts.model_cost + task.exact(opts.endpoints)
            row = {"episode": ep_i, "t": t, "headroom": float(q[opts.mean_index] - q.min()), "spread": float(np.ptp(q))}
            if _family(key) == "fork":
                gate, H = task.env.gate, task.cfg.H
                # Window: the option's horizon end reaches the gate, so it can lock a lane.
                row["phase"] = "post_gate" if t >= gate else ("window" if t >= gate - H else "pre_window")
            else:
                row["phase"] = "all"
            phases.append(row)
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    times = fam["branch_times"]
    n_cf = fam["counterfactual_episodes"]
    cf_alone = _branch_outcomes(task, controllers["alone"], banked["alone"][:n_cf], times, rng)
    cf_exact = _branch_outcomes(task, controllers["exact_term"], banked["exact_term"][:n_cf], times, rng)
    for eps in banked.values():
        for ep in eps:
            ep.trajectory.options, ep.trajectory.snapshots = [], []
    _json(path, {
        "key": key,
        "task": task.name,
        "meta": task.meta,
        "delay_past_horizon": task.delay_past_horizon,
        "optimal": _optimal_steps(task),
        "closed_loop": closed,
        "bank_size": len(bank),
        "reference_decision_metrics": reference_metrics,
        "option_headroom": phases,
        "counterfactual_alone": cf_alone,
        "counterfactual_exact": cf_exact,
        "seconds": time.time() - t0,
    })
    return f"{key}: setup {time.time() - t0:.0f}s, bank {len(bank)}"


# ---------------------------------------------------------------------------
# sweep: round 1 from MPPI alone


def _planner_seed(cfg, seed, rnd, u):
    return _seeds(cfg)["train"] + 100_000 * seed + 1_000 * rnd + u


def _unit_rng(cfg, key, seed, rnd, u):
    return np.random.default_rng([_seeds(cfg)["train"], zlib.crc32(key.encode()), seed, rnd, u])


def _evaluate_fitted(task, cfg, bank, fitted: dict) -> dict:
    seeds, n_eval = _seeds(cfg), cfg["common"]["eval_episodes"]
    out = {}
    for name, f in fitted.items():
        scorer = study.scorer_for(name, f)
        row = {"decision": study.decision_metrics(bank, scorer)}
        row["select"] = study.summarize(task, study.evaluate(task, task.controller(scorer=scorer), n_eval, seeds))
        if name in study.VALUE_ESTIMATORS:
            row["terminal"] = study.summarize(task, study.evaluate(task, task.controller(terminal=f), n_eval, seeds))
        out[name] = row
    return out


def _strip(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k not in ("costs",)}


def sweep_worker(cfg: dict, key: str, seed: int, out_dir: str) -> str:
    out_dir = Path(out_dir)
    path = out_dir / "sweep" / f"{_safe(key)}_seed{seed}.json"
    if path.exists():
        return f"{key} seed {seed}: exists"
    t0 = time.time()
    task = build_task(cfg, key, out_dir)
    fam = cfg[_family(key)]
    with open(out_dir / "banks" / f"{_safe(key)}.pkl", "rb") as fh:
        bank = pickle.load(fh)
    alone = task.controller(name="alone")
    units, rows = [], []
    for n in fam["units"]:
        while len(units) < n:
            u = len(units)
            units.append(study.collect_unit(task, alone, _unit_rng(cfg, key, seed, 1, u), _planner_seed(cfg, seed, 1, u),
                                            cfg["common"]["branches_per_episode"]))
        fitted = study.fit_all(task, units, cfg["common"]["ridge"])
        res = _evaluate_fitted(task, cfg, bank, fitted)
        outcomes = int(sum(u.outcomes for u in units))
        informative = sum(_distinct_outcome(task, u, br) for u in units for br in u.branches)
        rows.append({"units": n, "outcomes": outcomes, "branches_with_other_lane_or_goal": informative,
                     "estimators": {k: {m: (_strip(v) if m != "decision" else v) for m, v in r.items()} for k, r in res.items()},
                     "seconds": time.time() - t0})
        _json(path.with_suffix(".partial.json"), {"key": key, "seed": seed, "rows": rows})
    if _family(key) == "car":
        with open(out_dir / "pools" / f"{_safe(key)}_seed{seed}.pkl", "wb") as fh:
            pickle.dump(units[: cfg["car"]["api_units"]], fh)
    _json(path, {"key": key, "seed": seed, "rows": rows, "seconds": time.time() - t0})
    path.with_suffix(".partial.json").unlink(missing_ok=True)
    return f"{key} seed {seed}: sweep {time.time() - t0:.0f}s"


def _distinct_outcome(task, unit, br) -> bool:
    """Did the branch end differently in kind from the recorded episode (lane or goal)?"""
    last_a = unit.episode.trajectory.states[-1]
    last_b = br.trajectory.states[-1]
    if hasattr(task.env, "lane_b"):
        return bool(task.env.lane_b(last_a[1]) != task.env.lane_b(last_b[1]))
    return bool((last_a[2] > 0.5) != (last_b[2] > 0.5))


# ---------------------------------------------------------------------------
# api: MountainCar rounds from each estimator's own controller

API_CHAINS = {"mc": "terminal", "td0.9": "terminal", "td0": "terminal", "paired": "terminal", "rd": "select"}


def _fit_one(task, name, units, ridge):
    traj = study.unit_trajectories(task, units)
    if name == "mc":
        return est.fit_mc(task.view, traj, ridge)
    if name.startswith("td"):
        return est.fit_lstd(task.view, traj, float(name[2:]), ridge)
    if name == "paired":
        return est.fit_paired(task.view, study.unit_pairs(task, units), ridge)
    return est.fit_return_decomposition(task.view, study.unit_episodes(task, units), ridge)


def _controller_for(task, name, fitted):
    if API_CHAINS[name] == "terminal":
        return task.controller(terminal=fitted)
    return task.controller(scorer=study.scorer_for(name, fitted))


def api_worker(cfg: dict, key: str, name: str, seed: int, out_dir: str) -> str:
    out_dir = Path(out_dir)
    path = out_dir / "api" / f"{_safe(key)}_{name}_seed{seed}.json"
    if path.exists():
        return f"{key} {name} seed {seed}: exists"
    t0 = time.time()
    task = build_task(cfg, key, out_dir)
    spec, ridge = cfg["car"], cfg["common"]["ridge"]
    with open(out_dir / "banks" / f"{_safe(key)}.pkl", "rb") as fh:
        bank = pickle.load(fh)
    with open(out_dir / "pools" / f"{_safe(key)}_seed{seed}.pkl", "rb") as fh:
        units = pickle.load(fh)
    cumulative = int(sum(u.outcomes for u in units))
    rows = []
    for rnd in range(1, spec["api_rounds"] + 1):
        fitted = _fit_one(task, name, units, ridge)
        ctrl = _controller_for(task, name, fitted)
        scorer = study.scorer_for(name, fitted)
        closed = study.summarize(task, study.evaluate(task, ctrl, cfg["common"]["eval_episodes"], _seeds(cfg)))
        rows.append({"round": rnd, "cumulative_outcomes": cumulative, "mode": API_CHAINS[name],
                     "decision": study.decision_metrics(bank, scorer), "closed_loop": _strip(closed),
                     "data_success": float(np.mean([u.episode.trajectory.states[-1, 2] > 0.5 for u in units])),
                     "seconds": time.time() - t0})
        _json(path.with_suffix(".partial.json"), {"rows": rows})
        if rnd == spec["api_rounds"]:
            break
        units = [
            study.collect_unit(task, ctrl, _unit_rng(cfg, key, seed, rnd + 1, u), _planner_seed(cfg, seed, rnd + 1, u),
                               cfg["common"]["branches_per_episode"])
            for u in range(spec["api_units"])
        ]
        cumulative += int(sum(u.outcomes for u in units))
    _json(path, {"key": key, "estimator": name, "seed": seed, "rows": rows, "seconds": time.time() - t0})
    path.with_suffix(".partial.json").unlink(missing_ok=True)
    return f"{key} {name} seed {seed}: api {time.time() - t0:.0f}s"


# ---------------------------------------------------------------------------


def _manifest(cfg_path: Path, out_dir: Path) -> None:
    hashes = {s: hashlib.sha256((REPO / s).read_bytes()).hexdigest() for s in SOURCES}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    _json(out_dir / f"manifest_{int(time.time())}.json", {
        "config": json.loads(cfg_path.read_text()), "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "source_sha256": hashes, "git_head": commit, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


def _run(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, *args) for fn, args in jobs]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["setup", "sweep", "api", "report"])
    ap.add_argument("--config", default=str(REPO / "configs" / "credit_testbed.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--only", default=None, help="comma-separated task keys")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    out_dir = Path(args.out_dir)
    for sub in ("setup", "banks", "sweep", "pools", "api", "exact"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    keys = task_keys(cfg) if args.only is None else args.only.split(",")
    if args.phase == "report":
        from scripts.credit.report_credit_stack import report

        report(cfg, out_dir)
        return
    _manifest(cfg_path, out_dir)
    if args.phase == "setup":
        # Solve each MountainCar value once before the workers share its cache.
        for key in keys:
            if _family(key) == "car":
                build_task(cfg, key, out_dir)
        _run([(setup_worker, (cfg, k, str(out_dir))) for k in keys], args.workers)
    elif args.phase == "sweep":
        jobs = []
        for k in keys:
            n_seeds = cfg["common"][f"data_seeds_{_family(k)}"]
            jobs += [(sweep_worker, (cfg, k, s, str(out_dir))) for s in range(n_seeds)]
        jobs.sort(key=lambda j: _family(j[1][1]) != "car")  # longest jobs first
        _run(jobs, args.workers)
    elif args.phase == "api":
        jobs = [(api_worker, (cfg, k, name, s, str(out_dir)))
                for k in keys if _family(k) == "car"
                for name in API_CHAINS for s in range(cfg["common"]["data_seeds_car"])]
        _run(jobs, args.workers)


if __name__ == "__main__":
    main()
