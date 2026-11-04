"""One bounded saved-evidence diagnosis and gated native H48 MPPI test; no fitting."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import resource
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = Path(
    "/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/runs/2026-11-03_full_pi_alternating_bounded_001"
)
DT = 0.008


def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            d,
            indent=2,
            default=lambda x: x.tolist()
            if isinstance(x, np.ndarray)
            else int(x)
            if isinstance(x, np.integer)
            else float(x)
            if isinstance(x, np.floating)
            else str(x),
        )
        + "\n"
    )


def bind(p):
    b = p.read_bytes()
    return dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def obs(s):
    return np.concatenate((s[..., 2:10], np.clip(s[..., 10:19], -10, 10)), axis=-1)


def healthy(s):
    return (s[..., 2] > 0.8) & (s[..., 2] < 2) & (s[..., 3] > -1) & (s[..., 3] < 1)


def active(posts):
    h = healthy(posts)
    return np.concatenate(
        (np.ones((*h.shape[:-1], 1), bool), np.logical_and.accumulate(h[..., :-1], axis=-1)),
        axis=-1,
    )


def weights(c, lam):
    w = np.exp(-(c - np.min(c)) / lam)
    return w / w.sum()


class Stop(RuntimeError):
    pass


class Limit(Stop):
    pass


class Gate(Stop):
    pass


class Budget:
    def __init__(self, out):
        self.out = out
        self.pre = json.loads(
            (ROOT / "artifacts/local_coupling_diagnosis_2026-11-03/budget_start.json").read_text()
        )
        self.t = time.monotonic()
        self.wall_pre = time.time() - self.pre["prep_wall_start_upper_epoch"]
        self.counts = {}
        self.unknown = {}
        self.stage = "analysis"
        self.log = (out / "ledger.jsonl").open("x")
        self.seq = 0

    def cpu(self):
        r = resource.getrusage(resource.RUSAGE_SELF)
        c = resource.getrusage(resource.RUSAGE_CHILDREN)
        return (
            self.pre["prep_cpu_reservation_seconds"]
            + self.pre.get("prior_process_cpu_seconds", 0)
            + r.ru_utime
            + r.ru_stime
            + c.ru_utime
            + c.ru_stime
        )

    def wall(self):
        return self.wall_pre + time.monotonic() - self.t

    def check(self):
        cpu_cap = 3000 if self.stage == "science" else 3420
        wall_cap = 3240 if self.stage == "science" else 3360
        if self.cpu() > cpu_cap or self.wall() > wall_cap:
            raise Limit("Frozen " + self.stage + " compute reservation exhausted")
        if sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) > 3 * 1024**3:
            raise Limit("3GiB output ceiling")

    def event(self, kind, **kw):
        self.log.write(
            json.dumps(
                dict(event=kind, wall=self.wall(), cpu=self.cpu(), **kw),
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x),
            )
            + "\n"
        )
        self.log.flush()

    def paid(self, counts, fn):
        self.check()
        ceilings = dict(
            model_controls=88780800,
            native_controls=19200,
            neural_rows=88788000,
            physics_substeps=355200000,
        )
        for key, value in counts.items():
            if self.counts.get(key, 0) + value > ceilings.get(key, float("inf")):
                raise Limit("Frozen work ceiling: " + key)
        self.seq += 1
        seq = self.seq
        self.event("request", id=seq, counts=counts)
        try:
            r = fn()
        except BaseException:
            for k, v in counts.items():
                self.unknown[k] = self.unknown.get(k, 0) + v
            self.event("error", id=seq, counts=counts)
            raise
        for k, v in counts.items():
            self.counts[k] = self.counts.get(k, 0) + v
        self.event("complete", id=seq, counts=counts)
        return r

    def measure(self, kind, n, fn, **kw):
        return self.paid({kind: int(n)}, fn)

    def close(self):
        write(
            self.out / "work.json",
            dict(
                completed=self.counts,
                unknown_partial=self.unknown,
                charged_cpu_seconds=self.cpu(),
                charged_elapsed_seconds=self.wall(),
                preparation_reservation=self.pre,
                new_fitting=0,
            ),
        )
        self.log.close()


def part_a(out, budget):
    from src.utils.math import temperature_for_neff_fraction

    rows = []
    for slot in range(2):
        for arm in ("zero", "fixed"):
            base = OLD / f"stream_0/r01_{arm}/search_{slot}"
            it = json.loads((base / "iterations.json").read_text())
            with np.load(base / "iterations.npz", allow_pickle=False) as z:
                for p in (0, 1, 50, 99):
                    budget.check()
                    key = f"p{p:03d}_"
                    c = z[key + "C"]
                    d = z[key + "D"]
                    w = z[key + "weights"]
                    lengths = z[key + "lengths"]
                    u = z[key + "U"]
                    output = z[key + "output_U"]
                    lam = it[p]["temperature"]
                    ad = 0.12 * d
                    edges = np.array([0, 25, 50, 100, 200, 400, 600, 800, 1000])
                    fallen = lengths < 1000
                    hist = np.histogram(lengths[fallen], bins=edges)[0]
                    w0 = weights(c, lam)
                    wa = weights(c + ad, lam)
                    l0 = temperature_for_neff_fraction(c, 0.05)
                    la = temperature_for_neff_fraction(c + ad, 0.05)
                    wr0 = weights(c, l0)
                    wra = weights(c + ad, la)
                    r = dict(
                        slot=slot,
                        arm=arm,
                        pass_index=p,
                        raw_bank=p != 1,
                        fall_histogram_edges_controls=edges,
                        fall_histogram=hist,
                        full_survival_fraction=float((~fallen).mean()),
                        ESS=float(1 / (w @ w)),
                        fallen_weight=float(w[fallen].sum()),
                        surviving_weight=float(w[~fallen].sum()),
                        lambda_actual=lam,
                        std_C=float(c.std()),
                        std_alpha_D=float(ad.std()),
                        alphaD_over_C=float(ad.std() / c.std()),
                        alphaD_over_lambda=float(ad.std() / lam),
                        actual_displacement_L2=float(np.linalg.norm(output - u)),
                        actual_displacement_RMS=float(np.sqrt(np.mean((output - u) ** 2))),
                        same_bank_fixed_lambda_TV=float(0.5 * np.abs(wa - w0).sum()),
                        same_bank_fixed_lambda_max_weight_change=float(np.max(np.abs(wa - w0))),
                        same_bank_ESS_recipe_TV=float(0.5 * np.abs(wra - wr0).sum()),
                        counterfactual_lambda_zero=l0,
                        counterfactual_lambda_alpha=la,
                        direct_update=None,
                    )
                    actual_expected = weights(c + (0.12 if arm == "fixed" else 0) * d, lam)
                    assert np.allclose(actual_expected, w, atol=1e-12, rtol=0)
                    if p != 1:
                        with np.load(base / f"bank_{p:03d}.npz", allow_pickle=False) as bank:
                            actions = bank["actions"]
                            assert np.array_equal(bank["C"], c) and np.array_equal(bank["D"], d)
                            assert np.array_equal(healthy(bank["posts"]).all(1), ~fallen)
                            du0 = np.einsum("k,kha->ha", w0, actions) - u
                            dua = np.einsum("k,kha->ha", wa, actions) - u
                            delta = dua - du0
                            recipe_delta = np.einsum("k,kha->ha", wra - wr0, actions)
                            assert np.allclose(
                                np.einsum("k,kha->ha", w, actions), output, atol=1e-12, rtol=0
                            )
                            r["direct_update"] = dict(
                                task_only_L2=float(np.linalg.norm(du0)),
                                alpha_increment_L2=float(np.linalg.norm(delta)),
                                total_alpha_L2=float(np.linalg.norm(dua)),
                                increment_over_task_norm=float(
                                    np.linalg.norm(delta) / np.linalg.norm(du0)
                                ),
                                increment_projection_on_total=float(
                                    np.sum(delta * dua) / np.sum(dua * dua)
                                ),
                                ESS_recipe_alpha_increment_L2=float(np.linalg.norm(recipe_delta)),
                            )
                    if p < 99:
                        r["next_actual_nominal_survives"] = bool(
                            healthy(z[f"p{p + 1:03d}_nominal_posts"]).all()
                        )
                    else:
                        r["next_actual_nominal_survives"] = None
                    rows.append(r)
    # Only pass0 has an identical entering nominal/bank across arms. Later arm contrasts are not direct alpha effects.
    pairs = []
    for slot in range(2):
        base0 = OLD / f"stream_0/r01_zero/search_{slot}"
        basea = OLD / f"stream_0/r01_fixed/search_{slot}"
        with np.load(base0 / "iterations.npz") as z, np.load(basea / "iterations.npz") as a:
            for p in (0, 1, 50, 99):
                pairs.append(
                    dict(
                        slot=slot,
                        pass_index=p,
                        entering_nominal_equal=bool(
                            np.array_equal(z[f"p{p:03d}_U"], a[f"p{p:03d}_U"])
                        ),
                        arm_output_difference_L2=float(
                            np.linalg.norm(z[f"p{p:03d}_output_U"] - a[f"p{p:03d}_output_U"])
                        ),
                    )
                )
    result = dict(
        rows=rows,
        paired_arm_contrasts=pairs,
        missing="Pass1 full banks were not preserved. Counterfactual update contributions are missing; no controls or physics regenerated. Pass99 updated mean was unscored, so its fate is missing.",
        native_scope="Saved native MuJoCo batch states, with prior frozen native replay audit; no new A physics. Counterfactual weightings do not simulate their alternative updated means.",
    )
    write(out / "part_a.json", result)
    return result


def positive_control(out):
    u = np.array([(a, b) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0)])
    c = 0.5 * np.sum((u - 0.5) ** 2, 1)
    d = 0.5 * np.sum((u + 0.5) ** 2, 1)
    result = []
    for alpha in (0.0, 0.12, 1.0):
        w = weights(
            c + alpha * d, 0.5
        )  # Independent product weights for separable two-step linear-integrator cost.
        q = np.array([-1.0, 0.0, 1.0])
        one = np.exp(-(0.5 * (q - 0.5) ** 2 + alpha * 0.5 * (q + 0.5) ** 2) / 0.5)
        one /= one.sum()
        independent = np.outer(one, one).reshape(-1)
        assert np.allclose(w, independent, atol=1e-15)
        result.append(
            dict(
                alpha=alpha,
                weights=w,
                mean_controls=w @ u,
                TV_from_zero=float(0.5 * np.abs(w - weights(c, 0.5)).sum()),
            )
        )
    assert result[-1]["TV_from_zero"] > 0.1 and np.max(np.abs(result[-1]["mean_controls"])) < 1e-15
    write(
        out / "positive_control.json",
        dict(
            passed=True,
            linear_dynamics="x_next=x+u; x0=0, two steps",
            cost="C=.5 sum(u-.5)^2; pi=-.5 so D=.5 sum(u+.5)^2; lambda=.5",
            arms=result,
        ),
    )
    return result


def part_c(out, episodes, policy, budget):
    from scipy import signal

    def functions(name, names, constants):
        p = ROOT / "archive/prototype_sources/local_coupling_2026-11-03" / name
        tree = ast.parse(p.read_text())
        tree = ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
            type_ignores=[],
        )
        space = dict(np=np, signal=signal, **constants)
        exec(compile(tree, str(p), "exec"), space)
        return space

    spectrum = functions(
        "action_spectrum.py",
        {"spectrum", "lowpass", "neighbour_index", "state_dependence"},
        dict(FIRST=0, SEGMENT=128, SEPARATION=100, K=10, FS=125.0),
    )
    conflict = functions(
        "label_conflict.py",
        {"squared_distances", "neighbours", "pairs", "matched", "flat"},
        dict(SEPARATION=100, K=10, BINS=10),
    )
    common = [
        i for i in range(8) if all(episodes[(i, a)]["survived"] for a in ("zero", "weak", "large"))
    ]
    if len(common) < 2:
        return dict(reached=False, reason="fewer than2 common complete paired300-control episodes")
    os0 = [episodes[(i, "zero")]["arrays"]["observations"][:-1] for i in common]
    pooled = np.concatenate(os0)
    mu = pooled.mean(0)
    sd = np.maximum(pooled.std(0), 1e-8)
    rows = []
    pair_results = {}
    for arm in ("zero", "weak", "large"):
        budget.check()
        obslist = [episodes[(i, arm)]["arrays"]["observations"][:-1] for i in common]
        actions = [episodes[(i, arm)]["arrays"]["actions"] for i in common]
        states = [(x - mu) / sd for x in obslist]
        index = spectrum["neighbour_index"](states)
        full = np.concatenate(actions)
        low = np.concatenate(spectrum["lowpass"](actions, 5.0))
        high = full - low
        f, p = spectrum["spectrum"](actions)
        pred = policy.predict(np.concatenate(obslist), meter=budget).astype(float)
        row = dict(
            arm=arm,
            starts=common,
            rows=len(full),
            above5Hz_variance_share=float(p[f > 5].sum() / p.sum()),
            low_R2=spectrum["state_dependence"](low, index, np.random.default_rng(0))["r2"],
            high_R2=spectrum["state_dependence"](high, index, np.random.default_rng(0))["r2"],
            student_vs_label_MSE=float(np.mean((pred - full) ** 2)),
        )
        held = dict(low=[], high=[])
        for fold in range(len(common)):
            train = [j for j in range(len(common)) if j != fold]
            train_baseline = np.concatenate([os0[j] for j in train])
            fm = train_baseline.mean(0)
            fs = np.maximum(train_baseline.std(0), 1e-8)
            fold_states = [(x - fm) / fs for x in obslist]
            low_list = spectrum["lowpass"](actions, 5.0)
            high_list = [u - lo for u, lo in zip(actions, low_list)]
            for band, labels in [("low", low_list), ("high", high_list)]:
                held[band].append(
                    conflict["neighbours"](fold_states, labels, fold, train)["mean_action"]
                )
        for band, target in [("low", low), ("high", high)]:
            prediction = np.concatenate(held[band])
            row["held_restart_" + band + "_R2"] = float(
                1 - np.sum((prediction - target) ** 2) / np.sum((target - target.mean(0)) ** 2)
            )
        rows.append(row)
        pair_results[arm] = conflict["pairs"](states, actions)
    edges = np.unique(
        np.quantile(
            np.concatenate(
                [conflict["flat"](p, "same")[0] for p in pair_results.values()]
                + [conflict["flat"](p, "other")[0] for p in pair_results.values()]
            ),
            np.linspace(0, 1, 11),
        )
    )
    edges[-1] = np.nextafter(edges[-1], np.inf)
    for row in rows:
        row["label_conflict"] = conflict["matched"](
            edges,
            conflict["flat"](pair_results[row["arm"]], "same"),
            conflict["flat"](pair_results[row["arm"]], "other"),
        )
    zero, large = rows[0], rows[-1]
    support = (
        large["above5Hz_variance_share"] <= 0.9 * zero["above5Hz_variance_share"]
        and large["held_restart_low_R2"] >= zero["held_restart_low_R2"] + 0.05
        and large["student_vs_label_MSE"] <= zero["student_vs_label_MSE"]
    )
    result = dict(
        reached=True,
        rows=rows,
        classification="iii_diagnostic_support_only"
        if support
        else "ii_diagnostic_pattern_not_established",
        adaptations="Exact archived function definitions; FIRST0, Welch128/64 for300 controls; 5Hz bands; K10/100-control temporal exclusion; common complete starts and shared alpha0 observation scaling. Startup is included, unlike original steady1000-control diagnostics. No CLI main, actor replacement or fitting invoked.",
    )
    write(out / "part_c.json", result)
    return result


def run_b(out, budget):
    import mujoco
    import torch

    from src.envs.walker2d import Walker2d
    from src.gps.mppi_gps.policy import Policy
    from src.gps.mppi_gps.runtime import restore, snapshot
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT / "configs/walker2d_best.json").read_text())
    task = json.loads((ROOT / "configs/walker2d_task.json").read_text())
    task["nthread"] = 4
    env = budget.paid(dict(environment_constructions=1), lambda: Walker2d(**task))
    policy = budget.paid(
        dict(checkpoint_loads=1), lambda: Policy.from_checkpoint(OLD / "parent0.pt")
    )[0]
    policy.eval()
    with np.load(OLD / "stream_0/r00_zero/eval_4130000000.npz", allow_pickle=False) as z:
        source = {k: z[k] for k in z.files}
    n = len(source["actions"])
    indices = np.linspace(1, n // 2, 8, dtype=int)
    restarts = []
    for t in indices:
        restarts.append(
            dict(
                state=source["states"][t],
                warmstart=source["warmstarts"][t],
                integration=source["integrations"][t],
                spec=np.array(int(mujoco.mjtState.mjSTATE_INTEGRATION)),
                dimensions=np.array([9, 9, 6]),
            )
        )
    np.savez_compressed(
        out / "restarts.npz",
        indices=indices,
        **{f"r{i}_{k}": v for i, r in enumerate(restarts) for k, v in r.items()},
    )
    write(
        out / "input_bindings.json",
        dict(
            checkpoint=bind(OLD / "parent0.pt"),
            student_episode=bind(OLD / "stream_0/r00_zero/eval_4130000000.npz"),
            indices=indices,
            mppi=bind(ROOT / "configs/walker2d_best.json"),
            task=bind(ROOT / "configs/walker2d_task.json"),
            source_controls=n,
        ),
    )
    raw_batch = env.batch_rollout

    def batch(state, actions, **kw):
        return budget.paid(
            dict(
                model_controls=int(actions.shape[0] * actions.shape[1]),
                physics_substeps=int(4 * actions.shape[0] * actions.shape[1]),
                model_queries=1,
            ),
            lambda: raw_batch(state, actions, **kw),
        )

    env.batch_rollout = batch

    def restore_start(r):
        return budget.paid(dict(restores=1), lambda: restore(env, r))

    def step(a):
        return budget.paid(dict(native_controls=1, physics_substeps=4), lambda: env.step(a))

    def discrepancy(state, posts, actions):
        mask = active(posts)
        pre = np.concatenate((np.broadcast_to(state, (len(posts), 1, 19)), posts[:, :-1]), 1)
        prediction = policy.predict(obs(pre[mask]), meter=budget).astype(float)
        terms = np.zeros(mask.shape)
        terms[mask] = 0.5 * np.sum((actions[mask] - prediction) ** 2, 1)
        return terms.sum(1), mask.sum(1)

    audits = []
    episodes = {}
    result = dict(
        gates={"candidate_survival": None, "action_change": None, "noninferior_survival": None},
        classification="unclassified",
        status="RUNNING",
    )
    bank_records = []
    try:
        budget.stage = "science"
        restore_start(restarts[0])
        cal = MPPI(env, MPPIConfig(**cfg), seed=4510000000)
        cu = np.clip(cal.U[None] + cal._sample_noise(), -1, 1)
        posts, c, sensor = batch(
            restarts[0]["state"], cu, initial_warmstart=restarts[0]["warmstart"]
        )
        d, length = discrepancy(restarts[0]["state"], posts, cu)
        alpha_large = float(0.5 * c.std() / d.std())
        np.savez_compressed(
            out / "calibration.npz", actions=cu, posts=posts, C=c, D=d, lengths=length
        )
        audits.append((restarts[0], cu[[0, 255]], posts[[0, 255]], "calibration"))
        write(
            out / "calibration.json",
            dict(
                alpha_large=alpha_large,
                std_C=float(c.std()),
                std_D=float(d.std()),
                survival_fraction=float(healthy(posts).all(1).mean()),
                alphaD_over_C=float(alpha_large * d.std() / c.std()),
                checkpoint_hash=policy.hash(),
            ),
        )
        result["alpha_large"] = alpha_large
        if not np.isfinite(alpha_large) or alpha_large <= 0.12:
            raise Gate("Calibration coefficient is not finite and larger than.12")
        if np.mean(healthy(posts).all(1)) < 0.5:
            result["gates"]["candidate_survival"] = False
            raise Gate("Calibration bank survival below.5")

        class Recorder(MPPI):
            def _plan_iteration(self, state, prior_cost, coupling, initial_warmstart):
                before = self.U.copy()
                a, info = super()._plan_iteration(state, prior_cost, coupling, initial_warmstart)
                p = self._last_states
                u = self._last_actions
                c = self._last_costs
                d = self.last_D
                survive = healthy(p).all(1)
                row = dict(
                    start=self.start_id,
                    arm=self.arm,
                    control=self.control,
                    inner=self.inner,
                    survival=float(survive.mean()),
                    ESS=float(1 / np.sum(self._last_weights**2)),
                    fallen_weight=float(self._last_weights[~survive].sum()),
                    lambda_value=info["lam"],
                    C_std=float(c.std()),
                    alphaD_std=float((self.alpha * d).std()),
                    U=before,
                    output_U=self.U.copy(),
                    C=c.copy(),
                    D=d.copy(),
                    weights=self._last_weights.copy(),
                    lengths=active(p).sum(1),
                )
                bank_records.append(row)
                if self.control == 0 or row["survival"] < 0.5:
                    name = f"bank_r{self.start_id}_{self.arm}_t{self.control:03d}_p{self.inner}.npz"
                    np.savez_compressed(
                        out / name,
                        actions=u,
                        posts=p,
                        C=c,
                        D=d,
                        weights=self._last_weights,
                        U=before,
                        output_U=self.U,
                    )
                    audits.append((snapshot(env), u[[0, 255]].copy(), p[[0, 255]].copy(), name))
                self.inner += 1
                if row["survival"] < 0.5:
                    result["gates"]["candidate_survival"] = False
                    result["failed_call"] = {
                        k: row[k] for k in ("start", "arm", "control", "inner", "survival")
                    }
                    raise Gate("Per-call candidate survival below.5")
                return a, info

        for arm, alpha in [("zero", 0.0), ("weak", 0.12), ("large", alpha_large)]:
            for i, r in enumerate(restarts):
                restore_start(r)
                planner = Recorder(env, MPPIConfig(**cfg), seed=4510000000 + i)
                planner.start_id = i
                planner.arm = arm
                planner.alpha = alpha
                planner.inner = 0
                planner.control = 0
                done = False
                trace = {
                    k: []
                    for k in (
                        "states",
                        "warmstarts",
                        "integrations",
                        "observations",
                        "actions",
                        "costs",
                    )
                }
                error = None

                def capture():
                    snap = snapshot(env)
                    for k, field in [
                        ("states", "state"),
                        ("warmstarts", "warmstart"),
                        ("integrations", "integration"),
                    ]:
                        trace[k].append(snap[field])
                    trace["observations"].append(obs(snap["state"]))

                capture()
                try:
                    for t in range(300):
                        planner.control = t
                        planner.inner = 0
                        state = env.get_state()

                        def prior(p, u):
                            planner.last_D, _ = discrepancy(state, p, u)
                            return alpha * planner.last_D

                        a, info = planner.plan_step(
                            state, prior_cost=prior, initial_warmstart=env.get_warmstart()
                        )
                        _, cost, done, _ = step(a)
                        trace["actions"].append(a)
                        trace["costs"].append(cost)
                        capture()
                        if done:
                            break
                except Stop as e:
                    error = repr(e)
                finally:
                    arrays = {k: np.asarray(v) for k, v in trace.items()}
                    np.savez_compressed(out / f"episode_r{i}_{arm}.npz", **arrays)
                    episode = dict(
                        start=i,
                        arm=arm,
                        alpha=alpha,
                        controls=len(trace["actions"]),
                        survived=bool(not done and len(trace["actions"]) == 300),
                        fell=done,
                        error=error,
                        arrays=arrays,
                    )
                    episodes[(i, arm)] = episode
                    write(
                        out / f"episode_r{i}_{arm}.json",
                        {k: v for k, v in episode.items() if k != "arrays"},
                    )
                    print(
                        json.dumps({k: v for k, v in episode.items() if k != "arrays"}), flush=True
                    )
                if error:
                    raise Gate(error)
                if arm != "zero":
                    base_success = sum(episodes[(j, "zero")]["survived"] for j in range(8))
                    success = sum(episodes[(j, arm)]["survived"] for j in range(i + 1))
                    if success + 7 - i < base_success:
                        result["gates"]["noninferior_survival"] = False
                        raise Gate("Nonzero arm cannot match completed baseline survival count")
        result["gates"]["candidate_survival"] = True
        result["survival_counts"] = {
            a: sum(episodes[(i, a)]["survived"] for i in range(8))
            for a in ("zero", "weak", "large")
        }
        result["gates"]["noninferior_survival"] = all(
            result["survival_counts"][a] >= result["survival_counts"]["zero"]
            for a in ("weak", "large")
        )
        changes = {}
        for a in ("weak", "large"):
            errors = []
            per = []
            for i in range(8):
                u = episodes[(i, "zero")]["arrays"]["actions"]
                v = episodes[(i, a)]["arrays"]["actions"]
                m = min(len(u), len(v))
                e = (u[:m] - v[:m]) ** 2
                errors.append(e.ravel())
                per.append(float(np.sqrt(e.mean())))
            changes[a] = dict(
                pooled_RMS=float(np.sqrt(np.concatenate(errors).mean())), per_restart_RMS=per
            )
        result["action_changes"] = changes
        result["gates"]["action_change"] = changes["large"]["pooled_RMS"] >= 0.01
        if not all(result["gates"].values()):
            raise Gate("Completed B gates did not all pass")
        result["part_c"] = part_c(out, episodes, policy, budget)
        result["classification"] = result["part_c"].get("classification", "unclassified")
        result["status"] = "COMPLETE"
    except Stop as e:
        result.update(status="STOPPED", reason=str(e))
    finally:
        budget.stage = "audit"
        audit_errors = []
        native_max = 0.0
        candidate_max = 0.0
        try:
            for (i, arm), e in episodes.items():
                restore_start(restarts[i])
                z = e["arrays"]
                for t, a in enumerate(z["actions"]):
                    step(a)
                    delta = float(np.max(np.abs(env.get_state() - z["states"][t + 1])))
                    native_max = max(native_max, delta)
                    if delta > 1e-10:
                        raise AssertionError("Native episode replay mismatch")
            for r, actions, posts, name in audits:
                for u, p in zip(actions, posts):
                    restore_start(r)
                    for t, a in enumerate(u):
                        _, _, done, _ = step(a)
                        delta = float(np.max(np.abs(env.get_state() - p[t])))
                        candidate_max = max(candidate_max, delta)
                        if delta > 1e-10:
                            raise AssertionError("Native candidate replay mismatch")
                        if done:
                            break
        except BaseException as e:
            audit_errors.append(repr(e))
        write(
            out / "native_audit.json",
            dict(
                passed=not audit_errors,
                native_trace_max_abs=native_max,
                candidate_prefix_max_abs=candidate_max,
                candidate_banks=len(audits),
                episodes=len(episodes),
                errors=audit_errors,
            ),
        )
        if audit_errors:
            result["status"] = "INVALID"
            result["classification"] = "unclassified"
        if bank_records:
            np.savez_compressed(
                out / "planning_records.npz",
                **{
                    f"p{i:05d}_{k}": v
                    for i, row in enumerate(bank_records)
                    for k, v in row.items()
                    if isinstance(v, np.ndarray)
                },
            )
            write(
                out / "planning_records.json",
                [
                    {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                    for r in bank_records
                ],
            )
        result["episodes"] = [
            {k: v for k, v in e.items() if k != "arrays"} for e in episodes.values()
        ]
        write(out / "part_b.json", result)
        env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-analysis", type=Path)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(out)
    result = {}
    try:
        if args.reuse_analysis:
            import shutil

            for name in ("part_a.json", "positive_control.json"):
                shutil.copyfile(args.reuse_analysis / name, out / name)
            write(
                out / "analysis_reuse.json",
                {
                    name: dict(
                        source=str(args.reuse_analysis / name), **bind(args.reuse_analysis / name)
                    )
                    for name in ("part_a.json", "positive_control.json")
                },
            )
            result["part_a"] = json.loads((out / "part_a.json").read_text())
        else:
            result["part_a"] = part_a(out, budget)
            positive_control(out)
        result["part_b"] = run_b(out, budget)
    except BaseException as e:
        write(out / "exception.json", dict(error=repr(e), traceback=traceback.format_exc()))
        raise
    finally:
        budget.close()
    print(
        json.dumps(
            {k: v for k, v in result.get("part_b", {}).items() if k not in ("episodes", "part_c")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
