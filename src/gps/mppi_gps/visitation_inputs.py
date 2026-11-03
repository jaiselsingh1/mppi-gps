"""Create-only S1.2 V inputs from selected, authenticated saved C1 bytes.

This module never decodes a checkpoint, constructs a policy, or imports a
simulator. The retained A1 certification authenticates the complete C1 index;
preparation verifies that index now and hashes only the selected raw arrays.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .consequence_inputs import (
    FORMAT,
    MIB,
    SHA256,
    ConsequenceInputError,
    _Budget,
    _copy_file,
    _json,
    _need,
    _safe_file,
    _write_arrays,
    _write_json,
)

ARRAY_NAMES = (
    "observations",
    "states",
    "warmstarts",
    "integrations",
    "actions",
    "native_costs",
    "x_velocities",
    "fall_costs",
    "done",
    "control_ordinals",
    "row_digests",
    "initial_digest",
    "metadata",
)
BOUNDARY_NAMES = ARRAY_NAMES[:4]
METADATA_PATHS = {
    "c1_config": "config.json",
    "c1_evaluations": "worker/evaluations.jsonl",
    "c1_actor_registry": "worker/actor_registry.json",
    "c1_prerequisites": "prerequisites.json",
    "c1_prepared_manifest": "prepared/prepared_input_manifest.json",
}


class VisitationInputError(ConsequenceInputError):
    """A missing, changed, inconsistent, or over-budget V input stops preparation."""


class _PreparationBudget(_Budget):
    def __init__(self, limits: Mapping[str, Any]):
        output = limits.get("output_bytes", 512 * MIB)
        _need(
            type(output) is int and 65536 < output <= 512 * MIB, "V output ceiling exceeds 512 MiB"
        )
        super().__init__(
            limits.get("input_read_bytes", 512 * MIB),
            min(output, 256 * MIB),
            limits.get("wall_seconds", 600),
            limits.get("peak_rss_bytes", 512 * MIB),
        )
        self.output_limit = output


def _read_bound(budget: _Budget, binding: Mapping[str, Any], kind: str) -> bytearray:
    return budget.whole(Path(binding["path"]).resolve(), binding, kind)


def _array_hash(value: np.ndarray) -> dict[str, Any]:
    return dict(
        shape=list(value.shape),
        dtype=value.dtype.str,
        sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    )


def _read_array(raw: Path, reference: Mapping[str, Any], key: str, budget: _Budget) -> np.ndarray:
    item = reference["arrays"][key]
    _need(
        set(item) == {"sha256", "shape", "dtype", "encoding", "offset", "byte_length"},
        "Selected array descriptor differs",
    )
    dtype, shape = np.dtype(item["dtype"]), item["shape"]
    offset, length = item["offset"], item["byte_length"]
    _need(
        not dtype.hasobject and dtype.fields is None and dtype.kind in "biufSU",
        "Unsafe selected array dtype",
    )
    _need(
        isinstance(shape, list)
        and len(shape) <= 3
        and all(type(n) is int and n >= 0 for n in shape),
        "Invalid selected array shape",
    )
    _need(
        type(offset) is int
        and type(length) is int
        and min(offset, length) >= 0
        and math.prod(shape) * dtype.itemsize == length
        and length <= 16 * MIB
        and offset + length <= raw.stat().st_size,
        "Selected array range differs",
    )
    _need(SHA256.fullmatch(str(item["sha256"])) is not None, "Invalid selected array hash")
    with raw.open("rb", buffering=0) as stream:
        stream.seek(offset)
        payload = budget.read(stream, length, "selected_array_C_bytes")
    _need(
        len(payload) == length and hashlib.sha256(payload).hexdigest() == item["sha256"],
        "Selected array C-byte SHA256 differs",
    )
    return np.frombuffer(payload, dtype=dtype).reshape(shape)


def _authenticate_index(
    path: Path,
    binding: Mapping[str, Any],
    selected: Mapping[str, Any],
    budget: _Budget,
) -> None:
    payload = budget.whole(path, binding, "complete_C1_index")
    requested, completed = set(), set()
    for line in payload.splitlines(keepends=True):
        budget.check()
        _need(len(line) <= 8192 and line.endswith(b"\n"), "Overlong or truncated C1 index row")
        row = _json(line)
        group = row.get("group")
        if group not in selected:
            continue
        if row.get("event") == "arrays_requested":
            _need(group not in requested, "Duplicate selected C1 array intent")
            _need(
                {k: v for k, v in row.items() if k != "event"} == selected[group],
                "Evaluation reference differs from authenticated C1 index",
            )
            requested.add(group)
        elif row.get("event") == "arrays_complete":
            maximum = max(
                a["offset"] + a["byte_length"] for a in selected[group]["arrays"].values()
            )
            _need(
                group in requested
                and group not in completed
                and row
                == dict(event="arrays_complete", format=FORMAT, group=group, end_offset=maximum),
                "Selected C1 array completion differs",
            )
            completed.add(group)
        else:
            raise VisitationInputError("Unexpected selected C1 index event")
    _need(requested == completed == set(selected), "Selected C1 index evidence is missing")


def _checked_episode(
    raw: Path,
    record: Mapping[str, Any],
    budget: _Budget,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference, n = record["trace"], record["controls"]
    _need(
        type(n) is int and 4 <= n <= 1000 and set(reference["arrays"]) == set(ARRAY_NAMES),
        "Incomplete C1 episode array roster",
    )
    arrays = {name: _read_array(raw, reference, name, budget) for name in ARRAY_NAMES}
    shapes = dict(
        observations=(n + 1, 17),
        states=(n + 1, 19),
        warmstarts=(n + 1, 9),
        integrations=(n + 1, 91),
        actions=(n, 6),
        native_costs=(n,),
        x_velocities=(n,),
        fall_costs=(n,),
        done=(n,),
        control_ordinals=(n,),
        row_digests=(n, 32),
        initial_digest=(32,),
    )
    for key, shape in shapes.items():
        value = arrays[key]
        expected = (
            "|u1"
            if key in {"done", "row_digests", "initial_digest"}
            else ("<u4" if key == "control_ordinals" else "<f8")
        )
        _need(
            value.shape == shape and value.dtype.str == expected and np.isfinite(value).all(),
            "C1 episode shape/dtype/nonfinite mismatch: " + key,
        )
    metadata_array = arrays.pop("metadata")
    _need(
        metadata_array.shape == () and metadata_array.dtype.kind in "SU", "Missing trace metadata"
    )
    metadata = _json(metadata_array.item())
    _need(
        metadata["alias"] == record["tag"]
        and metadata["actor"] == record["actor"]
        and metadata["seed"] == record["seed"]
        and metadata["horizon"] == 1000
        and metadata["completed_controls"] == n
        and metadata["availability"] == "closed_actual_prefix"
        and metadata["unknown_partial_controls"] == 0
        and metadata["unknown_partial_physics_substeps"] == 0
        and metadata["integration_spec"] == 16383
        and metadata["dimensions"] == [9, 9, 6],
        "C1 trace metadata/context binding differs",
    )
    _need(
        np.array_equal(arrays["control_ordinals"], np.arange(n, dtype="<u4"))
        and np.all(np.abs(arrays["actions"]) <= 1)
        and np.array_equal(arrays["actions"], arrays["actions"].astype("<f4").astype("<f8"))
        and not arrays["done"][:-1].any()
        and int(arrays["done"][-1]) == int(record["done"])
        and np.isin(arrays["done"], (0, 1)).all(),
        "C1 action/fall/ordinal binding differs",
    )
    _need(
        np.array_equal(
            arrays["observations"],
            np.concatenate(
                (arrays["states"][:, 2:10], np.clip(arrays["states"][:, 10:19], -10, 10)), axis=1
            ),
        ),
        "C1 recorded observation does not match the unchanged feature arithmetic",
    )
    digest = hashlib.sha256()
    for key in BOUNDARY_NAMES:
        digest.update(arrays[key][0].tobytes())
    previous = digest.digest()
    _need(previous == arrays["initial_digest"].tobytes(), "C1 initial boundary digest differs")
    for index in range(n):
        digest = hashlib.sha256(previous)
        for key in BOUNDARY_NAMES:
            digest.update(arrays[key][index + 1].tobytes())
        for key in ARRAY_NAMES[4:10]:
            digest.update(arrays[key][index].tobytes())
        previous = digest.digest()
        _need(previous == arrays["row_digests"][index].tobytes(), "C1 actual-prefix digest differs")
    arrays["metadata"] = metadata_array
    arrays["spec"] = np.array(16383, dtype="<i8")
    arrays["dimensions"] = np.array([9, 9, 6], dtype="<i8")
    return arrays, metadata


def _initial(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return dict(
        state=arrays["states"][0],
        warmstart=arrays["warmstarts"][0],
        integration=arrays["integrations"][0],
        observation=arrays["observations"][0],
        spec=arrays["spec"],
        dimensions=arrays["dimensions"],
    )


def _check_context(actual: Mapping[str, np.ndarray], canonical: Mapping[str, np.ndarray]) -> bool:
    for key in ("spec", "dimensions"):
        _need(np.array_equal(actual[key], canonical[key]), "Initial context structure differs")
    exact = True
    for key in ("state", "warmstart", "integration", "observation"):
        _need(
            actual[key].dtype == canonical[key].dtype
            and actual[key].shape == canonical[key].shape
            and np.allclose(actual[key], canonical[key], rtol=0, atol=1e-10),
            "Saved parent initial contexts disagree: " + key,
        )
        exact = exact and actual[key].tobytes() == canonical[key].tobytes()
    return exact


def prepare(config: Mapping[str, Any], run: Path) -> dict[str, Any]:
    """Create selected V input files; retain a failure receipt on any gate failure.

    ``input_preparation`` declares ``c1_run`` and exact ``{path,bytes,sha256}``
    bindings for ``review_recipe``, ``a1_verified_files``, ``recovery_admission``,
    ``reference`` and ``xml``. Limits are bounded by 600s/512MiB reads/RSS/output.
    The caller must separately freeze and verify execution-source/runtime gates.
    """
    settings, out = config["input_preparation"], Path(run) / "prepared"
    _need(not out.exists(), "V prepared output already exists; retries cannot replace evidence")
    budget = _PreparationBudget(settings.get("limits", {}))
    out.mkdir(parents=True, exist_ok=False)
    try:
        return _prepare(settings, out, budget)
    except BaseException as error:
        try:
            _write_json(
                out,
                "failure.json",
                dict(
                    schema="walker-visitation-input-failure-v1",
                    passed=False,
                    error_type=type(error).__name__,
                    error=str(error)[:2000],
                    accounting=budget.summary(),
                    scientific_work=0,
                ),
                budget,
                emergency=True,
            )
        except BaseException:
            pass
        raise


def _prepare(settings: Mapping[str, Any], out: Path, budget: _Budget) -> dict[str, Any]:
    review = _json(_read_bound(budget, settings["review_recipe"], "review_recipe"))
    _need(
        review["schema"] == "walker-student-visitation-s1.2-review-v2",
        "Unsupported V review recipe",
    )
    source = Path(settings["c1_run"]).resolve()
    _need(
        source.is_dir() and not out.resolve().is_relative_to(source),
        "Output cannot modify the C1 run",
    )
    admission = _json(_read_bound(budget, settings["recovery_admission"], "A2_admission"))
    _need(
        settings["recovery_admission"]["sha256"] == review["c1_recovery_receipt"]["sha256"]
        and admission["schema"] == "walker-c1-certificate-admission-v1"
        and admission["audit_recovery_resolved"] is True
        and admission["retained_A1_integrity_certification"] is True
        and admission["retained_original_numerical_certification"] is True,
        "C1 compensating admission binding is unresolved",
    )
    certified = _json(_read_bound(budget, settings["a1_verified_files"], "A1_verified_files"))[
        "files"
    ]
    for name in ("review_recipe", "a1_verified_files", "recovery_admission"):
        binding = settings[name]
        _copy_file(
            Path(binding["path"]).resolve(), out, "metadata/" + name + ".json", binding, budget
        )
    metadata = {}
    for name, relative in METADATA_PATHS.items():
        binding = review["input_bindings"][name]
        payload = budget.whole(_safe_file(source, relative), binding, "C1_metadata")
        metadata[name] = (
            [_json(line) for line in payload.splitlines()]
            if name == "c1_evaluations"
            else _json(payload)
        )
        _copy_file(_safe_file(source, relative), out, "metadata/c1/" + relative, binding, budget)
    _need(metadata["c1_prerequisites"]["passed"] is True, "Original C1 prerequisites did not pass")
    evaluations = metadata["c1_evaluations"]
    return _finish(settings, out, budget, review, source, certified, metadata, evaluations)


def _finish(
    settings: Mapping[str, Any],
    out: Path,
    budget: _Budget,
    review: Mapping[str, Any],
    source: Path,
    certified: Mapping[str, Any],
    metadata: Mapping[str, Any],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    roster = [
        row
        for stream in range(3)
        for split in ("train", "held")
        for row in review["source_episodes"][str(stream)][split]
    ]
    _need(
        len(roster) == 96 and len({row["tag"] for row in roster}) == 96,
        "Expected all 96 retained parent episodes",
    )
    seeds = review["validity"]["positive_control_reference_reset_seeds"]
    _need(
        seeds == list(range(2810000000, 2810000016)) + list(range(2820000000, 2820000016)),
        "The 32 declared C1 reset seeds differ",
    )
    for stream in range(3):
        rows = [
            row
            for split in ("train", "held")
            for row in review["source_episodes"][str(stream)][split]
        ]
        _need(
            len(rows) == 32 and [row["seed"] for row in rows] == seeds,
            "Per-parent source/reset roster differs",
        )
    by_tag = {}
    for record in evaluations:
        if record.get("arm") == "parent":
            _need(record["tag"] not in by_tag, "Duplicate parent evaluation")
            by_tag[record["tag"]] = record
    _need(set(by_tag) == {row["tag"] for row in roster}, "C1 parent evaluation roster differs")
    selected = {}
    for row in roster:
        record = by_tag[row["tag"]]
        actor = metadata["c1_actor_registry"][record["actor"]]
        parent = review["parents"][str(record["stream"])]
        _need(
            record["seed"] == row["seed"]
            and record["controls"] == row["controls"]
            and record["trace"]["group"] == row["array_group"]
            and record["trace"]["tag"] == row["tag"]
            and actor["sha256"] == parent["sha256"]
            and actor["actor_hash"] == parent["actor_hash"]
            and actor["file"] == parent["file"]
            and actor["arm"] == "parent",
            "Episode/checkpoint/source row join differs",
        )
        _need(
            record["actor"] == f"s{record['stream']}_parent"
            and record["tag"]
            == f"eval_s{record['stream']}_parent_{record['panel']}_{record['case']:02d}"
            and record["seed"]
            == (2810000000 if record["panel"] == "primary" else 2820000000) + record["case"],
            "Source episode actor/seed/panel identity differs",
        )
        selected[record["trace"]["group"]] = record["trace"]
    raw = _safe_file(source, "worker/evidence/arrays.bin")
    raw_binding = certified["run/worker/evidence/arrays.bin"]
    budget.track(raw, raw_binding["bytes"])
    index_binding = certified["run/worker/evidence/arrays.index.jsonl"]
    _authenticate_index(
        _safe_file(source, "worker/evidence/arrays.index.jsonl"), index_binding, selected, budget
    )
    anchors = review["validity"]["anchors"]
    _need(
        len(anchors) == 96 and len({(a["source_tag"], a["control"]) for a in anchors}) == 96,
        "Expected 96 distinct frozen V anchors",
    )
    selected_tags = {a["source_tag"] for a in anchors}
    _need(len(selected_tags) == 24, "Expected 24 unique V source episodes")
    catalog = dict(
        schema="walker-visitation-prepared-catalog-v1",
        status="complete",
        episodes={},
        initial_contexts={},
        parent_initial_contexts={},
        canonical_contexts={},
        anchors=anchors,
        parents={},
    )
    comparisons, contexts = [], {}
    for stream in range(3):
        for row in roster:
            record = by_tag[row["tag"]]
            if record["stream"] != stream:
                continue
            arrays, trace_metadata = _checked_episode(raw, record, budget)
            context = _initial(arrays)
            descriptor = _write_arrays(out, "initial/" + row["tag"] + ".npz", context, budget)
            catalog["initial_contexts"][row["tag"]] = descriptor | dict(
                stream=stream, seed=record["seed"], source_tag=row["tag"]
            )
            catalog["parent_initial_contexts"].setdefault(str(record["seed"]), {})[str(stream)] = (
                catalog["initial_contexts"][row["tag"]]
            )
            if stream == 0:
                contexts[record["seed"]] = {k: v.copy() for k, v in context.items()}
                canonical = _write_arrays(out, f"canonical/{record['seed']}.npz", context, budget)
                catalog["canonical_contexts"][str(record["seed"])] = canonical | dict(
                    source_tag=row["tag"]
                )
            exact = _check_context(context, contexts[record["seed"]])
            comparisons.append(
                dict(
                    source_tag=row["tag"],
                    seed=record["seed"],
                    stream=stream,
                    passed=True,
                    all_array_bytes_equal=exact,
                )
            )
            if row["tag"] in selected_tags:
                episode = _write_arrays(out, "episodes/" + row["tag"] + ".npz", arrays, budget)
                catalog["episodes"][row["tag"]] = episode | dict(
                    stream=stream,
                    actor=record["actor"],
                    seed=record["seed"],
                    controls=record["controls"],
                    source_record=record,
                    source_trace=record["trace"],
                    trace_metadata=trace_metadata,
                    array_hashes={k: _array_hash(v) for k, v in arrays.items()},
                )
    for anchor in anchors:
        record = by_tag[anchor["source_tag"]]
        n, ordinal = record["controls"], anchor["anchor_ordinal"]
        expected = [0, (n - 1) // 3, 2 * (n - 1) // 3, n - 1]
        _need(
            ordinal in range(4)
            and anchor["control"] == expected[ordinal]
            and anchor["array_group"] == record["trace"]["group"]
            and anchor["stream"] == record["stream"]
            and anchor["source_controls"] == n
            and anchor["source_case"] == record["case"]
            and anchor["panel"] == record["panel"]
            and (
                record["case"] in range(4)
                if record["panel"] == "primary"
                else record["case"] in range(4, 8)
            )
            and (ordinal == 0 or anchor["control"] > 0)
            and anchor["screen_role"]
            == ("secondary_initial" if ordinal == 0 else "primary_noninitial"),
            "Frozen V anchor/source binding differs",
        )
    manifest = {f["file"]: f for f in metadata["c1_prepared_manifest"]["files"]}
    for stream in range(3):
        parent = review["parents"][str(stream)]
        filename = f"parents/s{stream}_parent.pt"
        binding = manifest[filename]
        _need(
            binding["sha256"] == parent["sha256"]
            and binding["bytes"] == parent["continuation"]["checkpoint_bytes"],
            "Complete parent checkpoint binding differs",
        )
        copied = _copy_file(
            _safe_file(source, "prepared/" + filename), out, filename, binding, budget
        )
        sidecar = parent["global_cpu_sidecar"]
        copied_rng = _copy_file(
            _safe_file(source, "prepared/" + sidecar["file"]), out, sidecar["file"], sidecar, budget
        )
        catalog["parents"][str(stream)] = copied | dict(
            actor_hash=parent["actor_hash"],
            continuation=parent["continuation"],
            global_cpu_sidecar=copied_rng,
        )
    for key, filename in (("normalization", "normalization.npz"), ("task", "task.json")):
        binding = manifest["inputs/" + filename]
        if key == "normalization":
            _need(
                binding["sha256"] == review["normalization"]["sha256"],
                "Normalization binding differs",
            )
        catalog[key] = _copy_file(
            _safe_file(source, "prepared/inputs/" + filename),
            out,
            "inputs/" + filename,
            binding,
            budget,
        )
    for key, filename in (("reference", "reference.pt"), ("xml", "walker.xml")):
        binding = settings[key]
        expected = (
            review["reference"]["checkpoint_sha256"]
            if key == "reference"
            else metadata["c1_config"]["xml_sha256"]
        )
        _need(binding["sha256"] == expected, "Reference/XML whole-file binding differs")
        catalog[key] = _copy_file(
            Path(binding["path"]).resolve(), out, "inputs/" + filename, binding, budget
        )
    catalog["reference"]["actor_hash"] = review["reference"]["actor_hash"]
    catalog["source"] = dict(
        c1_run=str(source),
        raw=raw_binding,
        index=index_binding,
        a1_verified_files=settings["a1_verified_files"],
        recovery_admission=settings["recovery_admission"],
        current_verification="Full authenticated index; selected raw C bytes only. Full raw SHA is retained A1 point-in-time certification.",
    )
    expected_controls = sum(by_tag[a["source_tag"]]["controls"] - a["control"] for a in anchors)
    _need(
        expected_controls
        == review["finite_census"]["validity"][
            "expected_unchanged_parent_replay_controls_from_bound_metadata"
        ],
        "Unchanged-parent replay control census differs",
    )
    receipt = dict(
        schema="walker-visitation-static-input-receipt-v1",
        passed=True,
        parent_episodes=96,
        unique_V_source_episodes=24,
        anchors=96,
        noninitial_anchors=72,
        canonical_contexts=32,
        comparisons=comparisons,
        replay_rtol=0,
        replay_atol=1e-10,
        expected_parent_replay_controls=expected_controls,
        checkpoint_decodes=0,
        new_neural_rows=0,
        new_native_controls=0,
        new_optimizer_steps=0,
        caveat="Static source checks only; native byte-exact restoration, replay and runtime gates remain unverified.",
    )
    catalog["static_receipt"] = _write_json(out, "static_receipt.json", receipt, budget)
    _write_json(out, "catalog.json", catalog, budget)
    budget.unchanged()
    _write_json(
        out,
        "prepared_input_manifest.json",
        dict(
            schema="walker-visitation-prepared-input-manifest-v1",
            files=list(budget.output_files.values()),
            manifest_excludes_itself_and_final_receipt=True,
        ),
        budget,
    )
    census = dict(
        schema="walker-visitation-input-preparation-census-v1",
        passed=True,
        accounting=budget.summary(),
        source_files_unchanged=True,
        source_full_raw_rehashed=False,
        selected_source_groups=96,
        static_receipt=catalog["static_receipt"],
    )
    _write_json(out, "preparation_census.json", census, budget, emergency=True)
    return census


def verify_prepared(prepared: Path) -> dict[str, Any]:
    """Boundedly hash prepared files before execution, without scientific imports.

    The returned ``_verification_accounting`` describes this additional read
    pass; the caller must retain and charge it, rather than count it as free.
    This verifies the sealed metadata, not the native replay or actor output.
    """
    root = Path(prepared).resolve()
    budget = _PreparationBudget({})
    path = _safe_file(root, "prepared_input_manifest.json")
    budget.track(path)
    with path.open("rb", buffering=0) as stream:
        payload = budget.read(stream, min(path.stat().st_size, 4 * MIB + 1), "prepared_manifest")
    _need(len(payload) <= 4 * MIB, "Prepared manifest exceeds metadata bound")
    manifest = _json(payload)
    _need(
        manifest["schema"] == "walker-visitation-prepared-input-manifest-v1",
        "Unsupported prepared manifest",
    )
    names = set()
    for binding in manifest["files"]:
        _need(binding["file"] not in names, "Duplicate prepared input file")
        names.add(binding["file"])
        budget.whole(_safe_file(root, binding["file"]), binding, "prepared_input_verification")
    _need(
        {"catalog.json", "static_receipt.json"} <= names, "Missing prepared catalog/static receipt"
    )
    catalog_path = _safe_file(root, "catalog.json")
    binding = next(row for row in manifest["files"] if row["file"] == "catalog.json")
    catalog = _json(budget.whole(catalog_path, binding, "prepared_catalog"))
    _need(
        catalog["schema"] == "walker-visitation-prepared-catalog-v1"
        and catalog["status"] == "complete"
        and len(catalog["episodes"]) == 24
        and len(catalog["anchors"]) == 96
        and len(catalog["canonical_contexts"]) == 32
        and len(catalog["initial_contexts"]) == 96
        and all(set(v) == {"0", "1", "2"} for v in catalog["parent_initial_contexts"].values()),
        "Incomplete V prepared inputs",
    )
    budget.unchanged()
    catalog["_verification_accounting"] = budget.summary()
    return catalog
