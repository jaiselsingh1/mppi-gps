"""Recover omitted rejected-model packets from immutable original fit targets.

No simulator calls, new targets, selection, or metric-based tuning. The exact
original digest must match; otherwise preserve the failed recovery and stop.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from src.gps.independent_learning import Value, fit_value


def recover(out):
    out = Path(out)
    dest = out / "postrun_audit" / "recovered_models"
    dest.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((out / "config.json").read_text())
    gates = json.loads((out / "gates.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    source = Path(__file__).resolve().parents[2] / "src/gps/independent_learning.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_hashes"]["src/gps/independent_learning.py"]:
        raise ValueError("Recovery requires unchanged original learning implementation")
    started = time.perf_counter()
    records = []
    try:
        for gate in gates:
            if gate["arm"] == "memory":
                continue
            expected = gate["model"]
            original = out / "controllers" / f"value_{expected}.npz"
            if original.exists():
                with np.load(original, allow_pickle=False) as data:
                    packet = {key: data[key] for key in data.files}
                for key in ("target_version", "kind"):
                    packet[key] = str(packet[key].item())
                for key in ("total_steps", "width", "support_radius"):
                    packet[key] = packet[key].item()
                if Value(**packet).digest != expected:
                    raise ValueError("An originally archived model has an invalid digest")
                records.append(dict(arm=gate["arm"], seed=gate["seed"], cycle=gate["cycle"],
                                    digest=expected, recovered=False, matched=True, fit_updates=0))
                continue
            target_path = out / "controllers" / f"targets_s{gate['seed']}_{gate['arm']}_c{gate['cycle']}.npz"
            with np.load(target_path, allow_pickle=False) as data:
                xs, rs, ys, groups = (data[k] for k in ("states", "remaining", "targets", "groups"))
                if ys.ndim == 2:
                    ys = ys[-1]  # Exact saved final optimized backup, not a new backup.
                value = fit_value(xs, rs, ys, groups, total_steps=cfg["steps"],
                                  version=gate["target_controller"], kind=gate["arm"],
                                  **{key: cfg[key] for key in ("centers", "width", "ridge", "support_radius")})
            matched = value.digest == expected
            record = dict(arm=gate["arm"], seed=gate["seed"], cycle=gate["cycle"],
                          digest=expected, actual_digest=value.digest, recovered=True, matched=matched,
                          fit_updates=1, fit_rows=len(xs), target_file=str(target_path.relative_to(out)))
            records.append(record)
            if not matched:
                raise ValueError(f"Exact reconstruction failed: {record}")
            np.savez_compressed(dest / f"value_{expected}.npz", **value.packet())
    finally:
        result = dict(records=records, matched=all(r["matched"] for r in records),
                      fit_updates=sum(r["fit_updates"] for r in records),
                      fit_rows=sum(r.get("fit_rows", 0) for r in records),
                      simulated_controls=0, executed_controls=0, prediction_rows=0,
                      wall_seconds=time.perf_counter()-started,
                      reason="Original runner omitted packets rejected before qualification; exact deterministic reconstruction only",
                      no_new_targets=True, no_tuning=True)
        (dest / "recovery.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.out_dir)))
