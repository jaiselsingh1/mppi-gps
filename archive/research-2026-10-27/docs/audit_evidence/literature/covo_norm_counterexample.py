"""Check CoVO-MPC's unrestricted Euclidean contraction assertion.

This is a mathematical source check, not an implementation test or a refutation
of the paper's commuting covariance design. Run from the repository root with
`.venv/bin/python docs/audit_evidence/literature/covo_norm_counterexample.py`.
"""

import json
from pathlib import Path

import numpy as np


def main() -> None:
    d = np.diag([100.0, 1.0])
    covariance = np.array([[1.0, 0.9], [0.9, 1.0]])
    temperature = 2.0
    update = np.linalg.inv(np.eye(2) + 2.0 * covariance @ d / temperature)
    _, singular_values, right_vectors = np.linalg.svd(update)
    initial_error = right_vectors[0]
    final_error = update @ initial_error
    cost_before = float(initial_error @ d @ initial_error)
    cost_after = float(final_error @ d @ final_error)
    assert np.all(np.linalg.eigvalsh(d) > 0)
    assert np.all(np.linalg.eigvalsh(covariance) > 0)
    assert np.linalg.norm(final_error) > np.linalg.norm(initial_error)
    assert cost_after < cost_before
    result = {
        "audit_date": "2026-10-23",
        "sources": [
            "https://proceedings.mlr.press/v242/yi24b/yi24b.pdf#page=5",
            "https://arxiv.org/pdf/2401.07369v1#page=17",
        ],
        "claim_checked": "For arbitrary SPD D and Sigma, norm_2((I + 2 Sigma D / lambda)^-1) <= 1",
        "D": d.tolist(),
        "Sigma": covariance.tolist(),
        "lambda": temperature,
        "update_matrix": update.tolist(),
        "operator_norm_2": float(singular_values[0]),
        "initial_error": initial_error.tolist(),
        "final_error": final_error.tolist(),
        "initial_cost": cost_before,
        "final_cost": cost_after,
        "euclidean_contraction_assertion_holds": False,
        "quadratic_cost_decreases": True,
        "scope": "Population quadratic tilted-Gaussian mean update; noncommuting SPD matrices. Does not test finite samples or refute commuting covariance design.",
    }
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
