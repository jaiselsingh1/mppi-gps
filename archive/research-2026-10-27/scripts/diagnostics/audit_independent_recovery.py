"""Pure hash/provenance audit of recovered packets; no fits or model queries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def audit(out):
    start = time.perf_counter()
    out = Path(out).resolve()
    destination = out / "postrun_audit/recovered_models/independent_recovery_check.json"
    if destination.exists():
        raise FileExistsError(destination)
    recovery = json.loads((destination.parent / "recovery.json").read_text())
    cfg = json.loads((out / "config.json").read_text())
    gates = {(g["seed"], g["arm"], g["cycle"]): g
             for g in json.loads((out / "gates.json").read_text()) if g["arm"] != "memory"}
    errors, records = [], []
    updates, fitting_rows = 0, 0
    for row in recovery["records"]:
        key = (row["seed"], row["arm"], row["cycle"])
        gate = gates[key]
        expected = row["digest"]
        packet = ((destination.parent if row["recovered"] else out / "controllers")
                  / f"value_{expected}.npz")
        targets = out / "controllers" / f"targets_s{key[0]}_{key[1]}_c{key[2]}.npz"
        checks = {}
        with np.load(packet, allow_pickle=False) as v, np.load(targets, allow_pickle=False) as t:
            hasher = hashlib.sha256((str(v["target_version"])+str(v["kind"])).encode())
            for name in ("centers", "weights", "train_features"):
                hasher.update(v[name].tobytes())
            checks["digest_matches_original_gate"] = hasher.hexdigest() == expected == gate["model"]
            checks["frozen_continuation_version"] = str(v["target_version"]) == gate["target_controller"]
            checks["declared_model_kind"] = str(v["kind"]) == row["arm"]
            checks["frozen_hyperparameters"] = all(float(v[name]) == cfg[name]
                for name in ("width", "support_radius")) and int(v["total_steps"]) == cfg["steps"]
            expected_features = np.column_stack((t["states"]/[1.8, .07], t["remaining"]/cfg["steps"]))
            checks["support_uses_original_target_rows"] = np.array_equal(v["train_features"], expected_features)
            checks["centers_are_original_training_rows"] = all(
                np.any(np.all(expected_features == center, axis=1)) for center in v["centers"])
            checks["heldout_roots_excluded"] = (set(map(int, t["groups"])) <= set(gate["train_roots"])
                and not set(map(int, t["groups"])) & set(gate["heldout_roots"]))
            checks["finite_packet"] = all(np.isfinite(v[name]).all()
                                         for name in ("centers", "weights", "train_features"))
            if row["recovered"]:
                updates += row["fit_updates"]
                fitting_rows += row["fit_rows"]
                checks["recovery_row_accounting"] = row["fit_rows"] == len(t["states"])
        for name, passed in checks.items():
            if not passed:
                errors.append(f"{key}: {name}")
        records.append(dict(seed=key[0], arm=key[1], cycle=key[2], digest=expected,
                            recovered=row["recovered"], checks=checks))
    if len(records) != len(gates):
        errors.append("Recovery records do not cover all fitted seed/arm/cycle models")
    result = dict(schema="independent-recovery-provenance-check-v1", passed=not errors,
                  errors=errors, records=records, original_runner_archival_gap=True,
                  recovery_fit_updates_reconciled=updates, recovery_fitting_rows_reconciled=fitting_rows,
                  audit_work=dict(total_controls=0, simulated_controls=0, executed_controls=0,
                                  prediction_rows=0, fit_updates=0),
                  audit_wall_seconds=time.perf_counter()-start,
                  scope="Hashes and exact original-target provenance only; no fitting, prediction or transitions")
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    result = audit(parser.parse_args().out_dir)
    print(json.dumps({key: result[key] for key in
          ("passed", "errors", "recovery_fit_updates_reconciled", "recovery_fitting_rows_reconciled", "audit_wall_seconds")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
