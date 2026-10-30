"""Independently check the Gaussian density in the pinned external PO-MPC code.

Run with --repo pointing to alvaro-serra/pompc at
e0e1efea2f1e33ea98568ace960641d19b36a770. The original method body is
executed with deterministic NumPy stand-ins for JAX operations. No model,
paper result, or upstream implementation is changed. This check is expected
to FAIL for the inspected revision; it is not part of mppi-gps's test suite.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def check(repo: Path) -> dict:
    path = repo / "pompc/world_model.py"
    source = path.read_text()
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sample_actions")
    method.decorator_list = []
    method.returns = None
    for arg in method.args.posonlyargs + method.args.args + method.args.kwonlyargs:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    noise = np.array([0.3, -0.4], dtype=np.float64)
    namespace = {
        "jnp": np,
        "jax": SimpleNamespace(random=SimpleNamespace(normal=lambda key, shape: noise.copy())),
        "nn": SimpleNamespace(relu=lambda x: np.maximum(x, 0.0)),
    }
    exec(compile(module, str(path), "exec"), namespace)
    # Raw policy output: two means, then two raw log-standard-deviations.
    model = SimpleNamespace(policy_model=SimpleNamespace(
        apply_fn=lambda params, z: np.array([0.2, -0.2, 0.0, 0.0])))
    action, _, log_std, actual, mu = namespace["sample_actions"](
        model, np.zeros(1), {}, key=None)
    unsquashed = mu + noise * np.exp(log_std)
    # Independent scalar normal densities, followed by the exact same Jacobian
    # regularization as the released implementation.
    expected = sum(
        -0.5 * ((float(x) - float(m)) / math.exp(float(ls))) ** 2
        - float(ls) - 0.5 * math.log(2 * math.pi)
        for x, m, ls in zip(unsquashed, mu, log_std)
    ) - sum(math.log(max(1 - float(a) ** 2, 0) + 1e-6) for a in action)
    return {
        "repository": "https://github.com/alvaro-serra/pompc",
        "commit": "e0e1efea2f1e33ea98568ace960641d19b36a770",
        "source_path": "pompc/world_model.py",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "check": "original method body versus independent scalar Normal density",
        "action_dimension": 2,
        "noise": noise.tolist(),
        "actual_log_density": float(actual),
        "expected_log_density": expected,
        "difference": float(actual) - expected,
        "passed": math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12),
        "scope": "external implementation discrepancy; upstream left unmodified",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.repo)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    assert result["passed"], "Released PO-MPC Gaussian log density disagrees with independent formula"
