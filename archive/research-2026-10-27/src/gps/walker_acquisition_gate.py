"""Pure evidence gate for fresh independent Walker acquisition competence.

No simulator, actor, model checkpoint or training module is imported. The
legacy append-pickle stream is decoded with an explicit NumPy-only class
allowlist and must bind exactly to all converted JSON rows. Source hashes,
runtime metadata and completed outcome audit are independent requirements.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import pickle

import numpy as np

from scripts.diagnostics.audit_walker_qualification import audit_rows


REQUIRED_FILES = (
    "assets/walker2d.xml", "configs/walker2d_task.json", "configs/walker2d_best.json",
    "src/envs/base.py", "src/envs/mujoco_env.py", "src/envs/walker2d.py",
    "src/rl/walker.py", "src/mppi/mppi.py", "src/utils/math.py", "src/utils/config.py",
    "diag/step31.py", "diag/step32.py", "scripts/training/train_coupled_value.py",
    "scripts/training/train_replanning_value.py", "src/gps/expert_continuation.py",
    "src/gps/replanning_rollout.py",
)
REQUIRED_PACKAGES = ("mujoco", "numpy", "scipy")
TUNED_RECIPE = dict(K=256, H=48, lam=3.0, noise_sigma=.3, use_is_correction=False,
    noise_std=None, noise_cov=None, noise_lowpass_cutoff_hz=23.4375,
    noise_lowpass_sample_rate_hz=125., noise_lowpass_order=2, initial_iterations=2,
    lam_target_neff_fraction=None)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _path(root, relative):
    if not isinstance(relative, str):
        raise ValueError("path must be a relative string")
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError("unsafe relative path")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("path escapes its declared root")
    return result


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool) or str(key) in result:
                raise ValueError("invalid or colliding raw dictionary keys")
            result[str(key)] = _plain(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"unsupported raw value type: {type(value).__name__}")


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


class _SummaryUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy" and name in ("dtype", "ndarray"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(np._core.multiarray, name)
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return np._core.numeric._frombuffer
        raise pickle.UnpicklingError(f"forbidden raw pickle class: {module}.{name}")


def _raw_rows(data):
    stream, rows = io.BytesIO(data), []
    while stream.tell() < len(data):
        item = _SummaryUnpickler(stream).load()  # A partial final pickle is an error.
        if isinstance(item, dict) and "rows" in item:
            item = item["rows"]
        rows.extend(item if isinstance(item, list) else [item])
    return rows


def _expanded_config(root):
    """Read dataclass literal defaults without executing runtime source."""
    tree = ast.parse((root / "src/utils/config.py").read_text())
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MPPIConfig")
    defaults = {node.target.id: ast.literal_eval(node.value) for node in config.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None}
    supplied = json.loads((root / "configs/walker2d_best.json").read_text())
    if not isinstance(supplied, dict) or set(supplied)-set(defaults):
        raise ValueError("tuned config contains unknown fields")
    return dict(defaults, **supplied)


def _matches_tuned_recipe(config):
    """No unreviewed defaults; bools and structural integers stay distinct."""
    if set(config) != set(TUNED_RECIPE):
        return False
    for key, expected in TUNED_RECIPE.items():
        actual = config[key]
        if expected is None:
            valid_type = actual is None
        elif type(expected) is bool:
            valid_type = type(actual) is bool
        elif type(expected) is int:
            valid_type = type(actual) is int
        else:
            # Literal/JSON numeric 3 and 3.0 are equivalent temperatures;
            # booleans never stand in for numeric parameters.
            valid_type = type(actual) in (int, float) and np.isfinite(actual)
        if not valid_type or actual != expected:
            return False
    return True


def validate_acquisition_evidence(evidence_dir, runtime_root, *, expected_runtime=None):
    """Return ready/phase/reasons; missing evidence is PENDING, never success.

    ``runtime_root`` is a read-only snapshot of the actual execution source.
    ``expected_runtime`` is independently collected execution metadata, not a
    source_verified flag. All files listed in the manifest are checked, in
    addition to the mandatory critical-source set. No files are written.
    """
    directory, runtime = Path(evidence_dir).resolve(), Path(runtime_root).resolve()
    pending, failures, proof = [], [], dict(source_identity_verified=False,
        raw_stream_binding_verified=False, runtime_metadata_verified=False,
        fully_expanded_config_verified=False, raw_acquisition_states_certified=False)
    result = dict(schema="walker-acquisition-evidence-gate-v1", ready=False, phase="PENDING",
                  reasons=[], proof=proof, additional_physics_controls=0,
                  limitations=["This certifies completed outcome summaries and source/runtime binding, not raw reached-state trajectories.",
                    "The legacy stream can omit failed/in-flight uncommitted work; recorded costs remain a lower bound.",
                    "A single reset panel is not training-seed replication or a learner-performance result."])

    def read(name):
        path = directory / name
        if not path.is_file():
            pending.append(f"missing_input: {name}")
            return None
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError) as error:
            failures.append(f"invalid_json: {name}: {error}")
            return None

    records, status = read("records.json"), read("status.json")
    if records is not None and status is not None:
        try:
            rows = records["rows"] if isinstance(records, dict) else records
            if not isinstance(rows, list) or not isinstance(status, dict):
                raise ValueError("records need a rows list and status an object")
            outcome = audit_rows(rows, finished=status.get("finished") is True)
            result["outcome_audit"] = outcome
            if outcome["status"] == "FAIL" or (status.get("terminated") is True and not outcome["finished"]):
                failures.append("terminal_panel_failure: completed/terminated qualification did not pass")
            elif outcome["status"] != "PASS":
                pending.append("outcome_panel_pending: successful finish and all 256 valid unique rows required")
        except (ValueError, TypeError, KeyError) as error:
            failures.append(f"invalid_outcome_records: {error}")

    manifest, binding = read("runtime_manifest.json"), read("raw_stream_binding.json")
    if isinstance(manifest, dict):
        files = manifest.get("files", {})
        if not isinstance(files, dict):
            failures.append("invalid_source_manifest: files must be an object")
            files = {}
        missing = sorted(set(REQUIRED_FILES)-set(files))
        if missing:
            pending.append("missing_critical_sources: " + ", ".join(missing))
        comparisons = []
        for relative, digest in files.items():
            try:
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid SHA-256 digest")
                archived = _path(directory / "archived_source", relative)
                actual = _path(runtime, relative)
                if not archived.is_file() or not actual.is_file():
                    pending.append(f"missing_source_file: {relative}")
                    continue
                a, b = _sha(archived.read_bytes()), _sha(actual.read_bytes())
                matched = a == digest == b
                comparisons.append(dict(path=relative, manifest=digest, archived=a, runtime=b, matched=matched))
                if not matched:
                    failures.append(f"source_hash_mismatch: {relative}")
            except (ValueError, OSError) as error:
                failures.append(f"invalid_source_path_or_hash: {relative}: {error}")
        result["source_checks"] = comparisons
        proof["source_identity_verified"] = bool(not missing and len(comparisons) == len(files)
            and all(row["matched"] for row in comparisons))
        try:
            if proof["source_identity_verified"]:
                actual_config = _expanded_config(runtime)
                archived_config = _expanded_config(directory / "archived_source")
                proof["fully_expanded_config_verified"] = (actual_config == archived_config and
                    _matches_tuned_recipe(actual_config) and _matches_tuned_recipe(archived_config))
                result["expanded_tuned_config"] = actual_config
                if not proof["fully_expanded_config_verified"]:
                    failures.append("expanded_tuned_config_mismatch: full defaults/overrides or frozen K256 recipe differ")
        except (ValueError, OSError, StopIteration, TypeError) as error:
            failures.append(f"expanded_tuned_config_invalid: {error}")
        if expected_runtime is None:
            pending.append("runtime_metadata_unconfirmed: independently collected expected_runtime required")
        else:
            try:
                comparisons = {key: manifest.get(key) == expected_runtime.get(key) == value for key, value in
                               (("backend", "mujoco_cpu"), ("control_dt", .008), ("frame_skip", 4))}
                packages, expected_packages = manifest.get("packages", {}), expected_runtime.get("packages", {})
                for name in REQUIRED_PACKAGES:
                    comparisons[f"package:{name}"] = (isinstance(packages.get(name), str) and
                        bool(packages[name]) and packages[name] == expected_packages.get(name))
                proof["runtime_metadata_verified"] = all(comparisons.values())
                result["runtime_metadata_checks"] = comparisons
                if not proof["runtime_metadata_verified"]:
                    failures.append("runtime_metadata_mismatch: packages/backend/control_dt/frame_skip not independently matched")
            except (AttributeError, TypeError) as error:
                failures.append(f"runtime_metadata_invalid: {error}")
    elif manifest is not None:
        failures.append("invalid_source_manifest: expected an object")

    if isinstance(binding, dict) and records is not None:
        try:
            if binding.get("records_sha256") != _sha((directory / "records.json").read_bytes()):
                failures.append("records_binding_hash_mismatch")
            raw_path = _path(directory, binding.get("raw_stream_file", "validation.pkl"))
            if not raw_path.is_file():
                pending.append("missing_raw_stream: " + str(binding.get("raw_stream_file", "validation.pkl")))
            else:
                raw = raw_path.read_bytes()
                if binding.get("raw_stream_sha256") != _sha(raw):
                    failures.append("raw_stream_hash_mismatch")
                else:
                    raw_rows = _raw_rows(raw)
                    rows = records["rows"] if isinstance(records, dict) else records
                    equal = _canonical(raw_rows) == _canonical(rows)
                    result["raw_stream_rows"] = len(raw_rows)
                    if not equal:
                        failures.append("raw_records_canonical_mismatch: every append row and its order must match JSON")
                    proof["raw_stream_binding_verified"] = bool(equal and
                        binding.get("records_sha256") == _sha((directory / "records.json").read_bytes()))
        except (ValueError, TypeError, OSError, pickle.UnpicklingError, EOFError, KeyError) as error:
            failures.append(f"raw_stream_decode_or_binding_invalid: {error}")
    elif binding is not None and not isinstance(binding, dict):
        failures.append("invalid_raw_binding: expected an object with matching records")

    result["reasons"] = failures + pending
    result["ready"] = bool(not failures and not pending and all(proof[key] for key in
        ("source_identity_verified", "raw_stream_binding_verified", "runtime_metadata_verified", "fully_expanded_config_verified"))
        and result.get("outcome_audit", {}).get("status") == "PASS")
    result["phase"] = "FAIL" if failures else "PASS" if result["ready"] else "PENDING"
    return result
