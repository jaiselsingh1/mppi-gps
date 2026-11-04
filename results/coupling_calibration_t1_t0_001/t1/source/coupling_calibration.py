"""Actor-bound, saved-bank calibration of a per-executed-step residual."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .bank_review import local_bank_update
from .provenance import write
from .trajectory_pi_native import binding


def competitive_indices(costs: np.ndarray, count: int = 32) -> np.ndarray:
    """Fixed top task-only weights at fixed temperature: stable cost ordering."""
    valid = np.flatnonzero(np.isfinite(costs))
    return valid[np.argsort(costs[valid], kind="stable")[:count]]


def calibration_ratio(costs: np.ndarray, residual: np.ndarray, indices: np.ndarray) -> float:
    if len(indices) < 2:
        raise ValueError("At least two competitive candidates required")
    c, r = float(np.std(costs[indices])), float(np.std(residual[indices]))
    if min(c, r) <= 1e-12 or not np.isfinite([c, r]).all():
        raise ValueError("Calibration spread is degenerate")
    return r / c


def calibrate(config_path: Path, out: Path) -> dict:
    import h5py

    started, cpu = time.monotonic(), time.process_time()
    config = json.loads(config_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    write(out / "config_binding.json", binding(config_path))
    if binding(Path(config["checkpoint"]["path"])) != config["checkpoint"]:
        raise ValueError("Frozen incoming actor changed")
    data, ratios = [], {"all": [], "competitive": []}
    for entry in config["banks"]:
        for key in ("bank", "stats"):
            if binding(Path(entry[key]["path"])) != entry[key]:
                raise ValueError("Frozen bank binding changed")
        with h5py.File(entry["stats"]["path"], "r") as raw:
            group = raw[entry["group"]]
            if group.attrs["actor_hash"] != config["actor_hash"]:
                raise ValueError("Saved discrepancy belongs to a different actor")
            recorded = {
                key: group[key][:] for key in ("C", "D", "lengths", "U_before", "full_survival")
            }
        with np.load(entry["bank"]["path"], allow_pickle=False) as raw:
            arrays = {
                key: raw[key].copy()
                for key in ("C", "D", "lengths", "actions", "U_before", "full_survival")
            }
        if any(not np.array_equal(value, arrays[key]) for key, value in recorded.items()):
            raise ValueError("Full-bank arrays do not match the bound H5 group")
        c, d, n = arrays["C"], arrays["D"], arrays["lengths"]
        valid = np.isfinite(c)
        if not np.isfinite(d[valid]).all() or np.any(n[valid] <= 0):
            raise ValueError("Missing executable finite candidate residual")
        residual = np.zeros_like(d)
        residual[valid] = d[valid] / n[valid]
        arrays["D"] = residual
        indices = dict(all=np.flatnonzero(valid), competitive=competitive_indices(c))
        baseline = local_bank_update(arrays, 0.0, config["temperature"])
        bank_ratios = {}
        for basis, selected in indices.items():
            bank_ratios[basis] = calibration_ratio(c, residual, selected)
            ratios[basis].append(bank_ratios[basis])
        data.append((entry, arrays, indices, baseline, bank_ratios))
    if len(data) != 6:
        raise ValueError("Exactly six frozen same-actor banks are required")
    alphas = {
        basis: {str(target): float(target / np.median(values)) for target in config["targets"]}
        for basis, values in ratios.items()
    }
    rows = []
    saved = {}
    for bank_id, (entry, arrays, indices, base, bank_ratios) in enumerate(data):
        for basis, values in alphas.items():
            for target, alpha in values.items():
                actual = local_bank_update(arrays, alpha, config["temperature"])
                w, w0 = actual["weights"], base["weights"]
                survived, n = arrays["full_survival"], arrays["lengths"]
                selected = indices[basis]
                rows.append(
                    dict(
                        bank=bank_id,
                        context=entry["context_index"],
                        group=entry["group"],
                        basis=basis,
                        target=float(target),
                        alpha=alpha,
                        achieved_ratio=alpha * bank_ratios[basis],
                        competitive_size=len(indices["competitive"]),
                        competitive_baseline_mass=float(w0[indices["competitive"]].sum()),
                        selected_baseline_mass=float(w0[selected].sum()),
                        baseline_top=int(np.argmax(w0)),
                        top=int(np.argmax(w)),
                        top_changed=bool(np.argmax(w0) != np.argmax(w)),
                        total_variation=float(0.5 * np.abs(w - w0).sum()),
                        baseline_ess=base["effective_sample_size"],
                        ess=actual["effective_sample_size"],
                        baseline_fallen_mass=float(w0[~survived].sum()),
                        baseline_censored_mass=float(w0[n < 48].sum()),
                        fallen_mass=float(w[~survived].sum()),
                        censored_mass=float(w[n < 48].sum()),
                        last_control_fall_mass=float(w[(~survived) & (n == 48)].sum()),
                        invalid_mass=float(w[~actual["valid"]].sum()),
                        first_action_rms=float(
                            np.sqrt(
                                np.mean((actual["updated_plan"][0] - base["updated_plan"][0]) ** 2)
                            )
                        ),
                        baseline_survival_fraction=float(survived.mean()),
                    )
                )
                saved[f"b{bank_id}_{basis}_{target}_weights"] = w
        for key in ("C", "D", "lengths", "full_survival"):
            saved[f"b{bank_id}_{key}"] = arrays[key]
        saved[f"b{bank_id}_baseline_weights"] = base["weights"]
    with (out / "per_bank.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / "measurements.npz", **saved)
    result = dict(
        schema="coupling-calibration-t1-1",
        status="COMPLETED",
        config=binding(config_path),
        actor_hash=config["actor_hash"],
        checkpoint=config["checkpoint"],
        alphas=alphas,
        median_unit_ratios={key: float(np.median(value)) for key, value in ratios.items()},
        selected_alpha=alphas["competitive"]["0.3"],
        selection="Frozen competitive32, target .3; not chosen from execution or fitting.",
        results=binding(out / "per_bank.csv"),
        measurements=binding(out / "measurements.npz"),
        work=dict(
            cpu_seconds=time.process_time() - cpu,
            wall_seconds=time.monotonic() - started,
            physics_substeps=0,
            neural_rows=0,
            optimizer_updates=0,
        ),
    )
    write(out / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(calibrate(args.config, args.out), indent=2))


if __name__ == "__main__":
    main()
