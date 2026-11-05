"""Zero-integration Drake EOM check on the already-paid native parity roster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.gps.mppi_gps.acrobot_drake_reference import mechanics


def run(preparation_path: Path, cases_path: Path, output: Path) -> None:
    from pydrake.examples import AcrobotPlant

    preparation = json.loads(preparation_path.read_text())
    cases = json.loads(cases_path.read_text())["cases"]
    if len(cases) != 32:
        raise ValueError("Use exactly the existing 32 native parity cases")
    plant = AcrobotPlant()
    context = plant.CreateDefaultContext()
    plant.get_mutable_parameters(context).SetFromVector(preparation["drake_parameter_vector"])
    rows = []
    for case in cases:
        state = np.asarray(case["state"])
        context.get_mutable_continuous_state_vector().SetFromVector([state[0] + np.pi, *state[1:]])
        torque = case["torque_nm"]
        plant.get_input_port(0).FixValue(context, [torque])
        actual = plant.EvalTimeDerivatives(context).CopyToVector()
        mass, bias, _ = mechanics(state)
        expected = np.r_[state[2:], np.linalg.solve(mass, [0.0, torque] - bias)]
        error = float(np.max(np.abs(actual - expected)))
        rows.append(
            {
                "index": case["index"],
                "state": state.tolist(),
                "torque_nm": torque,
                "drake_derivative": actual.tolist(),
                "analytic_derivative": expected.tolist(),
                "max_error": error,
                "saved_native_one_step_error": case["max_error"],
            }
        )
    passed = all(row["max_error"] <= 1e-9 for row in rows)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "passed": passed,
        "atol": 1e-9,
        "rtol": 0,
        "preparation_sha256": hashlib.sha256(preparation_path.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "rows": rows,
        "counts": {
            "drake_EOM_evaluations": 32,
            "analytic_EOM_evaluations": 32,
            "physics_substeps": 0,
            "CARE_solves": 0,
            "SDP_solves": 0,
        },
        "maximum_error": max(row["max_error"] for row in rows),
    }
    (output / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise ValueError("Drake nonlinear EOM parity failed; no native controller execution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.preparation, args.cases, args.output)
