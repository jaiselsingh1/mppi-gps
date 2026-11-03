"""Independent saved-result reductions and a fixed, metered native replay subset.

This reader does not call the production objective, outcome or selection code.
It shares only the existing policy loader, snapshot restore and resource meter.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .full_pi_alternating import Meter, binding, write
from .provenance import fingerprint


def physical_cost(posts, actions):
    healthy = (
        (posts[..., 2] > 0.8)
        & (posts[..., 2] < 2.0)
        & (posts[..., 3] > -1.0)
        & (posts[..., 3] < 1.0)
    )
    active = np.concatenate(
        (
            np.ones((*healthy.shape[:-1], 1), bool),
            np.logical_and.accumulate(healthy[..., :-1], axis=-1),
        ),
        axis=-1,
    )
    native = (
        2 * np.abs(posts[..., 10] - 1.5)
        + 0.5 * posts[..., 11] ** 2
        + 0.1 * posts[..., 3] ** 2
        + 0.001 * np.sum(actions**2, axis=-1)
        + 5 * (~healthy)
    )
    total = np.where(active, native, 0.0).sum(-1) + 20 * (~healthy).any(-1) + 5 * (~active).sum(-1)
    valid = np.isfinite(posts).all(-1) & np.isfinite(actions).all(-1)
    total = np.where((valid | ~active).all(-1), total, np.inf)
    walks = healthy.all(-1) & (np.abs(posts[..., 10] - 1.5).mean(-1) <= 0.3)
    return total, walks, active


class Audit:
    def __init__(self, run, out, replay=True):
        self.run, self.out, self.replay = Path(run), Path(out), replay
        self.out.mkdir(parents=True, exist_ok=False)
        self.cfg = json.loads((self.run / "config.json").read_text())
        self.meter = Meter(
            self.out,
            dict(
                wall_seconds=7200,
                cpu_seconds=18000,
                rss_bytes=6 * 1024**3,
                output_bytes=2 * 1024**3,
                neural_rows=100000000,
                native_controls=20000,
                model_controls=0,
                physics_substeps=80000,
            ),
        )
        self.checks, self.bytes = 0, 0

    def require(self, predicate, text):
        self.checks += 1
        if not bool(predicate):
            raise AssertionError(text)

    def close(self, a, b, text, atol=1e-8, rtol=1e-12):
        self.require(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False), text)

    def load(self, path):
        self.bytes += Path(path).stat().st_size
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}

    def policy(self, path):
        from .policy import Policy

        self.bytes += Path(path).stat().st_size
        return self.meter.paid(
            dict(checkpoint_loads=1, policy_constructions=1), lambda: Policy.from_checkpoint(path)
        )

    def D(self, start, posts, actions, actor):
        _, _, active = physical_cost(posts, actions)
        pre = np.concatenate((np.broadcast_to(start, (len(posts), 1, 19)), posts[:, :-1]), axis=1)
        x = pre[active]
        obs = np.concatenate(
            (x[:, 2:10], np.maximum(-10.0, np.minimum(10.0, x[:, 10:19]))), axis=-1
        )
        predictions = actor.predict(obs, meter=self.meter).astype(np.float64)
        residual = actions[active] - predictions
        terms = np.zeros(active.shape)
        terms[active] = np.einsum("ij,ij->i", residual, residual) / 2
        return terms.sum(1), active.sum(1)

    def trace(self, path, summary=None, actor=None):
        d = self.load(path)
        n = len(d["actions"])
        self.require(0 < n <= 1000, "trace length")
        for k in ("states", "warmstarts", "integrations", "observations"):
            self.require(len(d[k]) == n + 1, "N+1 " + k)
            self.require(np.isfinite(d[k]).all(), "finite " + k)
        obs = np.concatenate(
            (d["states"][:, 2:10], np.clip(d["states"][:, 10:19], -10, 10)), axis=1
        )
        self.require(np.array_equal(obs, d["observations"]), "exact observation map")
        self.require(np.max(np.abs(d["actions"])) <= 1, "bounded actions")
        s = d["states"][1:]
        healthy = (s[:, 2] > 0.8) & (s[:, 2] < 2) & (s[:, 3] > -1) & (s[:, 3] < 1)
        done = ~healthy
        self.require(np.array_equal(done, d["done"]), "native done")
        self.require(not done[:-1].any(), "no post-fall physical states")
        self.require(n == 1000 or bool(done[-1]), "no missing healthy tail")
        costs = (
            2 * abs(s[:, 10] - 1.5)
            + 0.5 * s[:, 11] ** 2
            + 0.1 * s[:, 3] ** 2
            + 0.001 * np.sum(d["actions"] ** 2, axis=1)
            + 5 * done
            + 20 * done
        )
        self.close(costs, d["costs"], "native running+terminal")
        self.close(d["fall_costs"], 20 * done, "terminal exactly once")
        self.close(d["velocities"], s[:, 10], "velocity alignment", atol=0, rtol=0)
        outcome = dict(
            controls=n,
            cost=float(costs.sum() + 5 * (1000 - n)),
            survived=n == 1000 and not done[-1],
            walking=n == 1000 and not done[-1] and np.mean(abs(s[:, 10] - 1.5)) <= 0.3,
        )
        if summary:
            for k, v in outcome.items():
                self.close(v, summary[k], "trace outcome " + k)
        if actor:
            pred = actor.predict(obs[:-1], meter=self.meter).astype(np.float64)
            self.close(pred, d["actions"], "standalone actor output", atol=2e-6, rtol=0)
        return d, outcome

    def ledger(self, root):
        expected, completed, pending = {}, {}, {}
        rows = (root / "ledger.jsonl").read_text().splitlines()
        self.bytes += (root / "ledger.jsonl").stat().st_size
        for line in rows:
            event = json.loads(line)
            if event["event"] == "request":
                self.require(event["operation"] not in pending, "unique operation")
                pending[event["operation"]] = event["counts"]
                for k, v in event["counts"].items():
                    expected[k] = expected.get(k, 0) + v
            elif event["event"] == "complete":
                self.require(
                    pending.pop(event["operation"]) == event["counts"], "paired completion"
                )
                for k, v in event["counts"].items():
                    completed[k] = completed.get(k, 0) + v
        work = json.loads((root / "work.json").read_text())
        self.require(expected == work["requested"], "requested work joins ledger")
        self.require(completed == work["completed"], "completed work joins ledger")
        self.require(not pending and not work["unknown_partial"], "no unclosed work")
        self.require(json.loads(rows[-1])["event"] == "closed", "closed worker ledger")
        for k, v in expected.items():
            if k in self.cfg["worker_limits"]:
                self.require(v <= self.cfg["worker_limits"][k], "work ceiling " + k)
        return completed

    def solve(self, directory, actor, query, alpha, base):
        rows = json.loads((directory / "iterations.json").read_text())
        info = json.loads((directory / "selection.json").read_text())
        its = self.load(directory / "iterations.npz")
        initial = np.asarray(info["initial_context"]["state"])
        choices, rawchoices = [], []
        for i, row in enumerate(rows):
            self.require(row["incoming_policy_hash"] == actor.hash(), "correct current policy")
            self.close(row["scale"], 0.1 ** (i / max(len(rows) - 1, 1)), "noise annealing")
            pre = f"p{i:03d}_"
            U, posts = its[pre + "U"], its[pre + "nominal_posts"]
            c, w, _ = physical_cost(posts[None], U[None])
            self.close(c[0], row["C"], "every nominal C")
            if query:
                d = 0.0
                self.require(not row["D_measured"], "query D explicitly unmeasured")
            else:
                dd, _ = self.D(initial, posts[None], U[None], actor)
                d = dd[0]
                # Existing Torch GEMM grouping has FP32 batch-size rounding.
                self.close(d, row["D"], "every nominal D", atol=5e-4, rtol=1e-6)
            self.close(row["J"], row["C"] + alpha * row["D"], "declared J")
            S = its[pre + "C"] + alpha * its[pre + "D"]
            weights = np.zeros_like(S)
            good = np.isfinite(S)
            weights[good] = np.exp(-(S[good] - min(S[good])) / row["temperature"])
            weights /= weights.sum()
            self.close(weights, its[pre + "weights"], "independent Gibbs weights", atol=1e-10)
            self.close(1 / np.sum(weights**2), row["ESS"], "ESS")
            if i + 1 < len(rows):
                self.require(
                    np.array_equal(its[pre + "output_U"], its[f"p{i + 1:03d}_U"]),
                    "unshifted next nominal",
                )
            admitted = bool(w[0]) and (base is None or c[0] <= 1.2 * base)
            self.require(admitted == row["admitted"], "output guard")
            rawchoices.append((row["J"], i))
            if query or admitted:
                choices.append((row["J"], i))
        best = min(choices)[1]
        self.require(info["selected"]["iteration"] == best, "best admitted historical nominal")
        self.require(
            info["raw_best"]["iteration"] == min(rawchoices)[1], "raw best historical nominal"
        )
        chosen = self.load(directory / "selected.npz")
        self.require(
            np.array_equal(chosen["actions"], its[f"p{best:03d}_U"]), "no best-sample substitution"
        )
        self.require(
            np.array_equal(chosen["posts"], its[f"p{best:03d}_nominal_posts"]), "selected states"
        )
        for bank in directory.glob("bank_*.npz"):
            i = int(bank.stem.split("_")[1])
            b = self.load(bank)
            c, _, active = physical_cost(b["posts"], b["actions"])
            self.close(c, b["C"], "stored full-bank C")
            self.require(np.array_equal(active.sum(1), b["lengths"]), "stored prefix lengths")
            if not query:
                d, _ = self.D(initial, b["posts"], b["actions"], actor)
                self.close(d, b["D"], "stored full-bank D", atol=5e-4, rtol=1e-6)
            mean = np.einsum("k,kta->ta", b["weights"], b["actions"])
            self.close(
                mean, its[f"p{i:03d}_output_U"], "weighted-mean identity", atol=1e-12, rtol=0
            )
        return info, rows

    def execute(self):
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        result = dict(passed=False, partial=True)
        try:
            manifest = json.loads((self.run / "files_manifest.json").read_text())
            for name, b in manifest.items():
                self.bytes += (self.run / name).stat().st_size
                self.require(binding(self.run / name) == b, "sealed file " + name)
            totals = {}
            for stream in range(3):
                root = self.run / f"stream_{stream}"
                summary = json.loads((root / "result.json").read_text())
                self.require(summary["status"] == "COMPLETE", "complete stream")
                self.require(len(summary["rounds"]) == 14, "every scheduled arm/round")
                counts = self.ledger(root)
                for k, v in counts.items():
                    totals[k] = totals.get(k, 0) + v
                parent, parent_opt, _ = self.policy(self.run / f"parent{stream}.pt")
                for arm in ("zero", "fixed"):
                    previous = parent_opt
                    actor = parent
                    for r in range(7):
                        cell = root / f"r{r:02d}_{arm}"
                        row = json.loads((cell / "result.json").read_text())
                        child, opt, _ = self.policy(cell / "policy.pt")
                        self.require(child.hash() == row["policy_hash"], "endpoint identity")
                        for name, value in parent.normalization.items():
                            self.require(
                                np.array_equal(value, child.normalization[name]),
                                "fixed normalization",
                            )
                        self.require(opt["updates"] == 2000 + 250 * r, "cumulative Adam updates")
                        if r:
                            self.require(
                                row["incoming_optimizer_fingerprint"] == fingerprint(previous),
                                "Adam/RNG ownership chain",
                            )
                            self.require(
                                row["parent_policy_hash"] == actor.hash(),
                                "updated actor enters next solve",
                            )
                            fit = self.load(cell / "fit_arrays.npz")
                            rng = np.random.default_rng()
                            rng.bit_generator.state = previous["rng"]
                            expected = rng.integers(0, 2000, size=(250, 256))
                            self.require(
                                np.array_equal(expected, fit["indices"]),
                                "actual continued minibatch tape",
                            )
                            self.require(
                                fingerprint(rng.bit_generator.state) == fingerprint(opt["rng"]),
                                "private RNG endpoint",
                            )
                            self.require(
                                all(
                                    int(v["step"]) == 2000 + 250 * r
                                    for v in opt["adam"]["state"].values()
                                ),
                                "every Adam step",
                            )
                            self.close(
                                np.mean((fit["before"] - fit["targets"]) ** 2),
                                row["pre_mse"],
                                "pre-fit reconstruction",
                            )
                            self.close(
                                np.mean((fit["after"] - fit["targets"]) ** 2),
                                row["post_mse"],
                                "post-fit reconstruction",
                            )
                            xs, us = [], []
                            for slot in range(2):
                                trace, outcome = self.trace(
                                    cell / f"teacher_{slot}.npz", row["teacher"][slot]
                                )
                                self.require(outcome["walking"], "retained teacher walking")
                                base = json.loads(
                                    (root / f"initializer_{slot}.npz.json").read_text()
                                )["cost"]
                                self.require(outcome["cost"] <= 1.2 * base, "original cost guard")
                                info, _ = self.solve(
                                    cell / f"search_{slot}", actor, False, row["alpha"], base
                                )
                                selected = self.load(cell / f"search_{slot}/selected.npz")
                                self.require(
                                    np.array_equal(trace["actions"], selected["actions"]),
                                    "actual teacher labels",
                                )
                                self.close(
                                    trace["states"][1:],
                                    selected["posts"],
                                    "batch/native selected trace",
                                    atol=1e-10,
                                    rtol=0,
                                )
                                xs.append(trace["observations"][:-1])
                                us.append(trace["actions"])
                            self.require(
                                np.array_equal(np.concatenate(xs), fit["observations"]),
                                "training observations owned",
                            )
                            self.require(
                                np.array_equal(np.concatenate(us), fit["targets"]),
                                "training physical targets owned",
                            )
                        for reset, expected_outcome in zip(
                            self.cfg["primary_resets"], row["evaluation"], strict=True
                        ):
                            self.trace(cell / f"eval_{reset}.npz", expected_outcome, child)
                        if r == 6:
                            for reset, e in zip(
                                self.cfg["confirmation_resets"], row["confirmation"], strict=True
                            ):
                                self.trace(cell / f"confirm_{reset}.npz", e, child)
                        first = self.load(cell / f"eval_{self.cfg['primary_resets'][0]}.npz")
                        for q in row["queries"]:
                            t = q["anchor"]
                            self.require(
                                q["available"] == (t < len(first["actions"])),
                                "missing anchor handling",
                            )
                            if q["available"]:
                                info, _ = self.solve(cell / f"query_{t}", child, True, 0.0, None)
                                selected = self.load(cell / f"query_{t}/selected.npz")
                                self.close(
                                    np.mean((selected["actions"][0] - first["actions"][t]) ** 2),
                                    q["mse"],
                                    "student-visited discrepancy",
                                )
                                self.close(
                                    info["initial_context"]["state"],
                                    first["states"][t],
                                    "actual query anchor",
                                    atol=0,
                                    rtol=0,
                                )
                                self.require(
                                    q["paired_cost_gap"] is None
                                    if t
                                    else q["paired_cost_gap"] is not None,
                                    "matched horizon only",
                                )
                        previous, actor = opt, child
                # Exact pairing of private draws, not just equality of seeds.
                for r in range(1, 7):
                    z = self.load(root / f"r{r:02d}_zero/fit_arrays.npz")
                    f = self.load(root / f"r{r:02d}_fixed/fit_arrays.npz")
                    self.require(np.array_equal(z["indices"], f["indices"]), "paired learner tapes")
                    for slot in range(2):
                        zrows = json.loads(
                            (root / f"r{r:02d}_zero/search_{slot}/iterations.json").read_text()
                        )
                        frows = json.loads(
                            (root / f"r{r:02d}_fixed/search_{slot}/iterations.json").read_text()
                        )
                        self.require(
                            all(
                                a["rng_before"] == b["rng_before"]
                                for a, b in zip(zrows, frows, strict=True)
                            ),
                            "paired planner innovations",
                        )
            for k, v in totals.items():
                if k in self.cfg["total_limits"]:
                    self.require(v <= self.cfg["total_limits"][k], "total scientific ceiling " + k)
            if self.replay:
                self.replays()
            result.update(passed=True, partial=False, scientific_work=totals)
        except BaseException as error:
            result.update(error=repr(error), traceback=traceback.format_exc())
        finally:
            self.meter.flush()
            result.update(
                checks=self.checks,
                logical_input_bytes=self.bytes,
                audit_work=self.meter.completed,
                unknown_partial=self.meter.unknown_partial,
            )
            write(self.out / "audit.json", result)
            self.meter.event("closed", passed=result["passed"])
            self.meter.journal.close()
        return result

    def replays(self):
        import mujoco

        from src.envs.walker2d import Walker2d

        from .runtime import restore

        env = self.meter.paid(
            dict(environment_constructions=1),
            lambda: Walker2d(**(self.cfg["task"] | dict(nthread=1))),
        )
        targets = []
        for r in (1, 6):
            for arm in ("zero", "fixed"):
                cell = self.run / f"stream_0/r{r:02d}_{arm}"
                targets.append((cell / "teacher_0.npz", None, 1000))
        for arm in ("zero", "fixed"):
            cell = self.run / f"stream_0/r06_{arm}"
            actor, _, _ = self.policy(cell / "policy.pt")
            targets.append((cell / f"eval_{self.cfg['primary_resets'][0]}.npz", actor, 100))
        for path, actor, length in targets:
            d = self.load(path)
            init = dict(
                state=d["states"][0],
                warmstart=d["warmstarts"][0],
                integration=d["integrations"][0],
                spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
                dimensions=np.array([9, 9, 6]),
            )
            self.meter.paid(dict(restores=1, external_forwards=1), lambda: restore(env, init))
            for t in range(min(length, len(d["actions"]))):
                action = (
                    d["actions"][t]
                    if actor is None
                    else actor.predict(env._get_obs(), meter=self.meter).astype(np.float64)
                )
                self.meter.paid(
                    dict(native_controls=1, physics_substeps=4), lambda: env.step(action)
                )
                self.close(
                    env.get_state(),
                    d["states"][t + 1],
                    "fixed exact native replay",
                    atol=1e-10,
                    rtol=0,
                )
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    result = Audit(a.run, a.out).execute()
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                checks=result["checks"],
                wall=time.monotonic() - start,
                error=result.get("error"),
            )
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
