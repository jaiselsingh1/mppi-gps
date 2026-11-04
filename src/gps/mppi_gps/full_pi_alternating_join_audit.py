"""Additional independent continuation/fit-prediction joins, no physics or fits.

The original frozen audit runs first and remains preserved. This supplemental
reader strengthens ownership checks; it changes no scientific decision rule.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .full_pi_alternating import Meter, binding, write
from .provenance import fingerprint


def audit(run, out, prior):
    import torch

    from .policy import Policy

    run, out, prior = Path(run), Path(out), Path(prior)
    original = json.loads(prior.read_text())
    if not original["passed"]:
        raise ValueError("Original audit must pass before this supplemental reader")
    out.mkdir(parents=True, exist_ok=False)
    remaining = 100000000 - original["audit_work"].get("neural_rows", 0)
    prior_work = json.loads((prior.parent / "work.json").read_text())
    remaining_wall = 7200 - prior_work["wall_seconds"]
    if remaining_wall <= 0:
        raise RuntimeError("Combined frozen audit wall allowance exhausted")
    meter = Meter(
        out,
        dict(
            wall_seconds=remaining_wall,
            cpu_seconds=3600,
            rss_bytes=4 * 1024**3,
            output_bytes=64 * 1024**2,
            neural_rows=remaining,
            optimizer_steps=0,
            backward_rows=0,
            model_controls=0,
            native_controls=0,
        ),
    )
    torch.set_num_threads(1)
    checks, reads = 0, 0

    def require(condition, description):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(description)

    def load(path):
        nonlocal reads
        reads += path.stat().st_size
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}

    def actor(path):
        nonlocal reads
        reads += path.stat().st_size
        return meter.paid(
            dict(checkpoint_loads=1, policy_constructions=1), lambda: Policy.from_checkpoint(path)
        )[0]

    result = dict(passed=False, original_audit=binding(prior), scope="Additional saved-only joins")
    try:
        templates = [load(run / f"teacher{s}.npz") for s in range(2)]
        for stream in range(3):
            parent = actor(run / f"parent{stream}.pt")
            for arm in ("zero", "fixed"):
                previous = parent
                plans = [t["actions"] for t in templates]
                rngs = [
                    np.random.default_rng(4110000000 + 100 * stream + s).bit_generator.state
                    for s in range(2)
                ]
                for r in range(7):
                    cell = run / f"stream_{stream}/r{r:02d}_{arm}"
                    child = actor(cell / "policy.pt")
                    row = json.loads((cell / "result.json").read_text())
                    if r:
                        f = load(cell / "fit_arrays.npz")
                        a = previous.predict(f["observations"], meter=meter).astype(np.float64)
                        b = child.predict(f["observations"], meter=meter).astype(np.float64)
                        require(
                            bool(np.allclose(a, f["before"], atol=2e-6, rtol=0)),
                            "Actual incoming fit predictions",
                        )
                        require(
                            bool(np.allclose(b, f["after"], atol=2e-6, rtol=0)),
                            "Actual outgoing fit predictions",
                        )
                        for slot in range(2):
                            root = cell / f"search_{slot}"
                            rows = json.loads((root / "iterations.json").read_text())
                            info = json.loads((root / "selection.json").read_text())
                            its = load(root / "iterations.npz")
                            require(
                                np.array_equal(its["p000_U"], plans[slot]),
                                "Carried full trajectory",
                            )
                            require(
                                fingerprint(rows[0]["rng_before"]) == fingerprint(rngs[slot]),
                                "Carried planner RNG",
                            )
                            require(
                                rows[0]["incoming_policy_hash"] == previous.hash(),
                                "Retained updated actor in next search",
                            )
                            plans[slot] = load(root / "selected.npz")["actions"]
                            rngs[slot] = info["output_rng_state"]
                    with np.load(cell / "eval_4130000000.npz", allow_pickle=False) as f:
                        for q in row["queries"]:
                            t = q["anchor"]
                            if not q["available"]:
                                continue
                            root = cell / f"query_{t}"
                            info = json.loads((root / "selection.json").read_text())
                            its = load(root / "iterations.npz")
                            rows = json.loads((root / "iterations.json").read_text())
                            require(
                                np.array_equal(its["p000_U"], templates[0]["actions"]),
                                "Full original query template, no suffix",
                            )
                            expected = np.random.default_rng(
                                4120000000 + 10000 * stream + 100 * r + t
                            ).bit_generator.state
                            require(
                                fingerprint(rows[0]["rng_before"]) == fingerprint(expected),
                                "Declared independent query RNG",
                            )
                            for key, field in (
                                ("state", "states"),
                                ("warmstart", "warmstarts"),
                                ("integration", "integrations"),
                            ):
                                require(
                                    np.array_equal(info["initial_context"][key], f[field][t]),
                                    "Complete query context " + key,
                                )
                    previous = child
        result["passed"] = True
    except BaseException as error:
        import traceback

        result.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        meter.flush()
        result.update(
            checks=checks,
            logical_input_bytes=reads,
            audit_work=meter.completed,
            original_plus_supplement_neural_rows=original["audit_work"].get("neural_rows", 0)
            + meter.completed.get("neural_rows", 0),
            source=binding(Path(__file__)),
        )
        write(out / "audit.json", result)
        meter.event("closed", passed=result["passed"])
        meter.journal.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    a = p.parse_args()
    r = audit(a.run, a.out, a.prior)
    print(json.dumps({k: r[k] for k in ("passed", "checks")}))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
