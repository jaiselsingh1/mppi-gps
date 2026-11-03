"""Bounded J2 publication from audited JSON only; no arrays, actors or simulator."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import time
import zlib
from pathlib import Path
from typing import Any

MAX_READ_BYTES = 1024**3
MAX_FILE_BYTES = 16 * 1024**2
MAX_OUTPUT_BYTES = 32 * 1024**2
MAX_WALL_SECONDS = 60
ZERO_WORK = dict(
    native_controls=0,
    model_controls=0,
    physics_substeps=0,
    neural_rows=0,
    backward_rows=0,
    optimizer_steps=0,
    checkpoint_loads=0,
)


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _identity(stat: Any) -> tuple[int, ...]:
    return tuple(
        getattr(stat, key)
        for key in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def _json(payload: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            _need(key not in result, "Duplicate saved J2 JSON key")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError("Nonfinite saved J2 JSON: " + value)

    def finite_float(value: str) -> float:
        result = float(value)
        _need(math.isfinite(result), "Nonfinite saved J2 JSON number")
        return result

    return json.loads(
        payload, object_pairs_hook=pairs, parse_constant=reject, parse_float=finite_float
    )


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _number(value: Any) -> float:
    _need(type(value) in (int, float) and math.isfinite(value), "Invalid saved J2 number")
    return float(value)


class SavedInputs:
    def __init__(self, root: Path):
        _need(root.is_dir() and not root.is_symlink(), "Ordinary J2 worker directory required")
        self.root = root.resolve()
        self.started = time.monotonic()
        self.cpu_started = time.process_time()
        self.read_bytes = 0
        self.decoded_bytes = 0
        self.files: dict[str, dict[str, Any]] = {}
        self.stored_files: dict[str, dict[str, Any]] = {}
        self.identities: dict[Path, tuple[int, ...]] = {}

    def check(self) -> None:
        _need(
            time.monotonic() - self.started <= MAX_WALL_SECONDS, "J2 report wall budget exhausted"
        )

    def path(self, name: str) -> Path:
        relative = Path(name)
        _need(
            not relative.is_absolute() and ".." not in relative.parts,
            "J2 report path escapes worker",
        )
        path = self.root / relative
        _need(
            all(
                not parent.is_symlink()
                for parent in [path, *path.parents]
                if parent != self.root.parent
            ),
            "J2 report refuses source symlinks",
        )
        _need(path.resolve().is_relative_to(self.root), "J2 report path escapes worker")
        return path

    def load(self, name: str, binding: dict | None = None) -> Any:
        return self.external(self.path(name), name, binding)

    def external(self, path: Path, key: str, binding: dict | None = None) -> Any:
        self.check()
        _need(path.suffix == ".json", "Saved J2 JSON required")
        stored = path if path.exists() else path.with_name(path.name + ".gz")
        _need(
            stored.is_file() and not stored.is_symlink(),
            "Saved J2 JSON required",
        )
        before = stored.stat()
        _need(
            before.st_size <= MAX_FILE_BYTES
            and self.read_bytes + self.decoded_bytes + before.st_size <= MAX_READ_BYTES,
            "J2 report read budget exhausted",
        )
        with stored.open("rb") as stream:
            payload = stream.read(before.st_size + 1)
        self.read_bytes += len(payload)
        _need(
            len(payload) == before.st_size and _identity(stored.stat()) == _identity(before),
            "J2 report input changed",
        )
        self.stored_files[key] = dict(
            path=str(stored),
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            compressed=stored != path,
        )
        if stored != path:
            maximum = min(MAX_FILE_BYTES, MAX_READ_BYTES - self.read_bytes - self.decoded_bytes)
            _need(maximum > 0, "J2 report decoded read budget exhausted")
            decoder = zlib.decompressobj(wbits=31)
            try:
                payload = decoder.decompress(payload, maximum)
            except zlib.error as error:
                raise ValueError("Invalid compressed J2 JSON") from error
            self.decoded_bytes += len(payload)
            _need(decoder.eof, "J2 report decoded budget exhausted or truncated gzip")
            _need(not decoder.unused_data, "Trailing data after compressed J2 JSON")
        actual = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        if binding is not None:
            _need(
                actual == {key: binding[key] for key in actual},
                "J2 report input differs from independent audit: " + key,
            )
        self.files[key] = actual
        self.identities[stored] = _identity(before)
        return _json(payload)

    def verify_unchanged(self) -> None:
        self.check()
        for path, before in self.identities.items():
            _need(
                not path.is_symlink() and _identity(path.stat()) == before,
                "J2 report source changed",
            )


class Outputs:
    def __init__(self, root: Path, inputs: SavedInputs):
        _need(not root.exists(), "J2 report requires a fresh output directory")
        root.mkdir(parents=True)
        self.root, self.inputs = root, inputs
        self.bytes = 0
        self.files: dict[str, dict[str, Any]] = {}

    def save(self, name: str, payload: bytes) -> None:
        self.inputs.check()
        _need(
            self.bytes + len(payload) <= MAX_OUTPUT_BYTES - 128 * 1024,
            "J2 report output budget exhausted",
        )
        with (self.root / name).open("xb") as stream:
            stream.write(payload)
        self.files[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.bytes += len(payload)

    def table(self, name: str, rows: list[dict], fields: list[str]) -> None:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        self.save(name, stream.getvalue().encode())


def _same_numbers(actual: dict, expected: dict, names: dict[str, str]) -> None:
    for left, right in names.items():
        _need(
            math.isclose(_number(actual[left]), _number(expected[right]), rel_tol=0, abs_tol=1e-10),
            "J2 report case/audit join differs: " + left,
        )


def _case_row(summary: dict, audit: dict, config: dict) -> dict:
    evaluations = [row for row in summary["episodes"] if row["kind"] == "evaluation"]
    probes = [row for row in summary["episodes"] if row["kind"] == "probe"]
    _need(
        [row["seed"] for row in evaluations] == config["seeds"]["evaluation"]
        and [row["seed"] for row in probes] == config["seeds"]["probe"],
        "J2 report execution panels differ",
    )
    walks = [
        sum(bool(row["walking"]) for row in evaluations[start : start + 16]) for start in (0, 16)
    ]
    row = dict(
        case_id=summary["case_id"],
        seed=summary["seed"],
        condition=summary["condition"]["id"],
        jitter=summary["condition"]["kind"],
        signal_share=summary["condition"]["share"],
        checkpoint_sha256=summary["checkpoint"]["sha256"],
        policy_hash=summary["checkpoint"]["policy_hash"],
        evaluation_episodes=len(evaluations),
        walking=sum(walks),
        walking_panel_1=walks[0],
        walking_panel_2=walks[1],
        two_panel_competence=min(walks) >= 15,
        full_survival=sum(bool(value["survived"]) for value in evaluations),
        actual_controls=sum(value["controls"] for value in evaluations),
        right_censored_episodes=sum(bool(value["right_censored"]) for value in evaluations),
        diagnostic_error=summary["postfit"]["standing100"],
        diagnostic_short_error=summary["postfit"]["standing25"],
        offline_error=summary["fitting"]["held_clean_mse"],
        train_mse=summary["fitting"]["train_mse"],
        mean_probe_controls=summary["postfit"]["probe_mean_controls"],
        probe_falls=summary["postfit"]["probe_falls"],
        score_timing="post-fit diagnostic",
        before_update_scores_graded=False,
    )
    for source, target in [
        ("native_cost", "native_cost"),
        ("raw_cost", "raw_cost"),
        ("terminal_cost", "terminal_cost"),
        ("absorbing_cost", "absorbing_cost"),
        ("cost", "complete_cost"),
    ]:
        row[target] = math.fsum(_number(value[source]) for value in evaluations)
    row["mean_controls"] = row["actual_controls"] / len(evaluations)
    row["mean_observed_seconds"] = math.fsum(
        value["observed_seconds"] for value in evaluations
    ) / len(evaluations)
    for name in "ABC":
        row["preupdate_" + name + "_ungraded"] = _number(summary["pre_update_scores"][name])
    _need(
        summary["pre_update_scores"]["D"] is None and summary["pre_update_scores"]["E"] is None,
        "J2 inapplicable scores must stay missing",
    )
    row.update(preupdate_D="inapplicable", preupdate_E="inapplicable")
    _need(
        (row["case_id"], row["seed"], row["condition"])
        == (audit["case_id"], audit["seed"], audit["condition"]),
        "J2 report case identity differs",
    )
    _same_numbers(
        row,
        audit,
        {
            key: key
            for key in [
                "walking",
                "full_survival",
                "actual_controls",
                "complete_cost",
                "native_cost",
                "absorbing_cost",
                "diagnostic_error",
                "diagnostic_short_error",
                "offline_error",
                "mean_probe_controls",
                "probe_falls",
            ]
        },
    )
    return row


CASE_FIELDS = "case_id seed condition jitter signal_share checkpoint_sha256 policy_hash evaluation_episodes walking walking_panel_1 walking_panel_2 two_panel_competence full_survival actual_controls mean_controls mean_observed_seconds right_censored_episodes native_cost raw_cost terminal_cost absorbing_cost complete_cost diagnostic_error diagnostic_short_error offline_error train_mse mean_probe_controls probe_falls preupdate_A_ungraded preupdate_B_ungraded preupdate_C_ungraded preupdate_D preupdate_E score_timing before_update_scores_graded".split()
SEED_FIELDS = "seed comparison_executed diagnostic_spearman offline_spearman improvement diagnostic_auc_any_walking offline_auc_any_walking mean_probe_controls_spearman negative_probe_fall_count_spearman".split()


def _seed_rows(primary: dict | None, seeds: list[int]) -> list[dict]:
    if primary is None:
        return [dict(seed=seed, comparison_executed=False) for seed in seeds]
    rows = []
    _need([row["seed"] for row in primary["per_seed"]] == seeds, "J2 audited seed roster differs")
    for value in primary["per_seed"]:
        row = {key: value.get(key) for key in SEED_FIELDS if key in value}
        row.update(value.get("secondary_duration_controls", {}))
        row["comparison_executed"] = True
        rows.append(row)
    return rows


def _figures(
    outputs: Outputs,
    cases: list[dict],
    seeds: list[dict],
    curves: list[tuple[str, list[int], list[float]]],
    conditions: list[str],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    names = []

    def finish(name: str, figure: Any) -> None:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(figure)
        outputs.save(name, buffer.getvalue())
        names.append(name)

    with matplotlib.rc_context({"path.simplify": False, "agg.path.chunksize": 0}):
        figure, axis = plt.subplots(figsize=(10, 5))
        for case_id, updates, values in curves:
            axis.plot(updates, values, linewidth=0.45, alpha=0.65, label=case_id)
        if not curves:
            axis.text(0.5, 0.5, "No actor fit executed", ha="center", transform=axis.transAxes)
        axis.set(
            xlabel="Actual Adam update",
            ylabel="Recorded minibatch action MSE",
            title=f"All {len(curves)} actual fits; every recorded update",
        )
        finish("learning_curves.png", figure)

    figure, axis = plt.subplots(figsize=(11, 5))
    for seed in seeds:
        by_condition = {
            row["condition"]: row["walking"] for row in cases if row["seed"] == seed["seed"]
        }
        values = [by_condition.get(condition, math.nan) for condition in conditions]
        axis.plot(
            range(len(conditions)), values, marker="o", linewidth=0.8, label=str(seed["seed"])
        )
    axis.set(
        xticks=range(len(conditions)),
        xticklabels=conditions,
        ylim=(-0.5, 32.5),
        ylabel="Walking successes / 32 resets",
        title="Every actual child; missing conditions stay unexecuted",
    )
    axis.tick_params(axis="x", rotation=45)
    axis.legend(title="Fresh learner stream", fontsize=7)
    finish("condition_walking.png", figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for index, row in enumerate(seeds):
        for offset, key, color in [
            (-0.18, "offline_spearman", "tab:orange"),
            (0.18, "diagnostic_spearman", "tab:blue"),
        ]:
            value = row.get(key)
            if value is not None:
                axis.bar(
                    index + offset,
                    value,
                    width=0.34,
                    color=color,
                    label=key if index == 0 else None,
                )
    if not any(row.get("diagnostic_spearman") is not None for row in seeds):
        axis.text(
            0.5,
            0.5,
            "Primary comparison unexecuted or ineligible",
            ha="center",
            transform=axis.transAxes,
        )
    axis.set(
        xticks=range(len(seeds)),
        xticklabels=[str(row["seed"]) for row in seeds],
        ylim=(-1.05, 1.05),
        ylabel="Within-stream utility: negative Spearman(error, walking)",
        title="Six learner streams are the paired units",
    )
    axis.tick_params(axis="x", rotation=30)
    if axis.get_legend_handles_labels()[0]:
        axis.legend()
    finish("seed_utility.png", figure)
    return names


def report_run(worker: Path, audit_path: Path, out: Path) -> dict:
    """Publish only an independently audited complete or qualification-stopped run."""
    inputs = SavedInputs(worker)
    audit = inputs.external(audit_path, "independent_audit_result.json")
    _need(audit.get("passed") is True, "J2 report requires a passed independent audit")
    bindings = audit["input_files"]
    config = inputs.load("config.json", bindings["config.json"])
    result = inputs.load("result.json", bindings["result.json"])
    work = inputs.load("work.json", bindings["work.json"])
    _need(result["status"] == audit["worker_status"], "J2 worker/audit statuses differ")
    _need(
        result["status"]
        in {
            "COMPLETED_FIXED_J2_ROSTER",
            "STOPPED_CLEAN_POSITIVE_CONTROL",
            "STOPPED_REFERENCE_QUALIFICATION",
        },
        "J2 worker did not reach an audited scientific stopping point",
    )
    _need(result["work"] == work, "J2 work sidecar differs")
    primary = audit.get("primary")
    _need(
        audit["primary_comparison_executed"] == (primary is not None),
        "J2 primary execution flag differs",
    )
    _need(
        (primary is not None) == (result["status"] == "COMPLETED_FIXED_J2_ROSTER"),
        "J2 primary result disagrees with qualification stop",
    )
    indexed = {row["case_id"]: row for row in audit["records"]}
    _need(
        len(indexed) == len(audit["records"]) == len(result["cases"]),
        "J2 case/audit roster differs",
    )
    rows, fits = [], []
    seen = set()
    for entry in result["cases"]:
        case_id = entry["case_id"]
        _need(
            case_id not in seen
            and case_id in indexed
            and entry["summary"] == f"cases/{case_id}/summary.json",
            "J2 report duplicate or missing case",
        )
        seen.add(case_id)
        summary = inputs.load(entry["summary"], bindings[entry["summary"]])
        row = _case_row(summary, indexed[case_id], config)
        _need(
            (row["seed"], row["condition"]) == (entry["seed"], entry["condition"]),
            "J2 case roster identity differs",
        )
        rows.append(row)
        fits.append((row, f"cases/{case_id}/fit.json"))
    _need(result["actual_fits"] == len(rows), "J2 actual fit count differs")
    seed_rows = _seed_rows(primary, config["seeds"]["training"])
    outputs = Outputs(out, inputs)
    outputs.table("cases.csv", rows, CASE_FIELDS)
    outputs.table("per_seed_primary.csv", seed_rows, SEED_FIELDS)
    curve_buffer = io.BytesIO()
    curves = []
    curve_points = 0
    with gzip.GzipFile(fileobj=curve_buffer, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "case_id",
                    "seed",
                    "condition",
                    "checkpoint_sha256",
                    "update",
                    "fit_update",
                    "action_mse",
                    "point_json",
                ]
            )
            for row, name in fits:
                fit = inputs.load(name, bindings[name])
                _need(
                    fit["checkpoint"]["sha256"] == row["checkpoint_sha256"],
                    "J2 curve/endpoint checkpoint differs",
                )
                curve = fit["learning_curve"]
                _need(len(curve) == config["learning"]["updates"], "J2 curve omits actual updates")
                updates, losses = [], []
                for index, point in enumerate(curve, 1):
                    _need(
                        point["update"] == point["fit_update"] == index,
                        "J2 curve update sequence differs",
                    )
                    loss = _number(point["action_mse"])
                    _need(loss >= 0, "Negative J2 curve MSE")
                    writer.writerow(
                        [
                            row["case_id"],
                            row["seed"],
                            row["condition"],
                            row["checkpoint_sha256"],
                            point["update"],
                            point["fit_update"],
                            loss,
                            json.dumps(point, sort_keys=True, allow_nan=False),
                        ]
                    )
                    updates.append(index)
                    losses.append(loss)
                curves.append((row["case_id"], updates, losses))
                curve_points += len(curve)
                stream.flush()
                _need(
                    curve_buffer.tell() < MAX_OUTPUT_BYTES - outputs.bytes - 128 * 1024,
                    "J2 report output budget exhausted",
                )
    outputs.save("learning_curves.csv.gz", curve_buffer.getvalue())
    figures = _figures(
        outputs, rows, seed_rows, curves, [value["id"] for value in config["conditions"]]
    )
    primary_text = (
        "Primary comparison was not executed after the declared qualification stop. Missing statistics remain missing."
        if primary is None
        else f"Primary eligibility: {primary['eligible']}; finite post-fit diagnosis confirmed: {primary['confirmed']}. "
        "The gate is taken from the independently audited primary result, without selecting another score."
    )
    totals = {
        key: sum(row[key] for row in rows)
        for key in [
            "walking",
            "full_survival",
            "actual_controls",
            "native_cost",
            "absorbing_cost",
            "complete_cost",
        ]
    }
    report = f"""# J2.0 saved-evidence report

Worker status: `{result["status"]}`. Independent audit passed.

{primary_text}

Reference qualification walking per 16-reset panel:
`{audit.get("reference_qualification_walking", "not recorded")}`.
Clean-child qualification:
`{json.dumps(result.get("prerequisites", {}).get("clean"), sort_keys=True)}`.

The {len(rows)} actual fits produced {totals["walking"]} walking successes and
{totals["full_survival"]} full-survival episodes in {sum(row["evaluation_episodes"] for row in rows)}
standalone attempts, with {totals["actual_controls"]} executed controls.
Native cost sums to {totals["native_cost"]:.10g}, absorbing-tail cost to
{totals["absorbing_cost"]:.10g}, and complete cost to {totals["complete_cost"]:.10g}.
These totals describe the actual roster; unexecuted conditions have no outcomes.

[Every case](cases.csv) separates walking, survival, observed duration, censoring,
native/terminal/tail costs, reconstruction, checkpoint identity and probe duration.
Full survivors are censored at eight seconds. Probe gaps pool actual unequal
pre-fall prefixes; duration/fall controls are secondary and do not change admission.

[Six grouped primary units](per_seed_primary.csv) copy the audited seed-level
utilities and secondary duration controls. Conditions and reset episodes are not
independent students. Any-walking AUC concerns at least one success, not competence.
The case table separately records the frozen 15/16-in-each-panel convention.

![All actual fitting curves](learning_curves.png)

All {curve_points} curve points are preserved in [learning_curves.csv.gz](learning_curves.csv.gz),
including each complete original scalar record. The plotted minibatch losses are
fitting measurements, not held reconstruction or walking.

![Every actual child's walking](condition_walking.png)

![Audited within-stream utilities](seed_utility.png)

The primary score is measured after fitting on the exact evaluated checkpoint.
This result does not establish pre-update teachability, a better learning
mechanism, MPPI guidance or a closed loop. Before-update A/B/C are ungraded;
D/E remain explicitly inapplicable. No follow-up is authorized by this report.

[Cost accounting](cost.json) retains the worker and audit work separately from
this saved-only publication. No checkpoint, raw array, model or simulator was
loaded by the report. Final report I/O, wall time and source hashes are in
[receipt.json](receipt.json). A lossless compact mirror may supply `name.json.gz`
when `name.json` is absent; decoded bytes must retain the original audit hash.
"""
    outputs.save("report.md", report.encode())
    saved_json_read_bytes = inputs.read_bytes
    source_path = Path(__file__)
    source_before = source_path.stat()
    _need(
        inputs.read_bytes + inputs.decoded_bytes + source_before.st_size <= MAX_READ_BYTES,
        "J2 report source-hash read budget exhausted",
    )
    with source_path.open("rb") as stream:
        source_payload = stream.read(source_before.st_size + 1)
    inputs.read_bytes += len(source_payload)
    _need(
        len(source_payload) == source_before.st_size
        and _identity(source_path.stat()) == _identity(source_before),
        "J2 reporter source changed while hashing",
    )
    outputs.save(
        "cost.json",
        _encoded(
            dict(
                worker=work,
                audit={
                    key: audit.get(key)
                    for key in [
                        "wall_seconds",
                        "read_bytes",
                        "decoded_array_bytes",
                        "logical_read_bytes",
                        "new_neural_rows",
                        "requested_neural_rows",
                        "unknown_neural_rows",
                        "functional_network_calls",
                        "checkpoint_loads",
                        "new_physics_controls",
                        "new_gradient_rows",
                        "new_optimizer_steps",
                    ]
                },
                report=dict(
                    read_bytes=inputs.read_bytes,
                    decoded_bytes=inputs.decoded_bytes,
                    logical_read_bytes=inputs.read_bytes + inputs.decoded_bytes,
                    saved_json_read_bytes=saved_json_read_bytes,
                    reporter_source_read_bytes=len(source_payload),
                    new_scientific_work=ZERO_WORK,
                    final_totals="receipt.json",
                ),
                complete_engineering_cost=None,
                scope="Named worker, audit and saved-only report phases; nested times are not added",
            )
        ),
    )
    inputs.verify_unchanged()
    receipt = dict(
        schema="walker-j2-saved-report-v1",
        passed=True,
        worker_status=result["status"],
        audit_passed=True,
        primary_comparison_executed=primary is not None,
        primary=primary,
        actual_cases=len(rows),
        actual_fits=len(rows),
        curve_points=curve_points,
        figures=figures,
        input_files=inputs.files,
        stored_files=inputs.stored_files,
        input_binding_scope="input_files bind original decoded JSON; stored_files bind actual compressed/plain bytes; logical_read_bytes adds stored reads, gzip-decoded bytes and reporter source",
        read_bytes=inputs.read_bytes,
        decoded_bytes=inputs.decoded_bytes,
        logical_read_bytes=inputs.read_bytes + inputs.decoded_bytes,
        reporter_source=dict(
            bytes=len(source_payload), sha256=hashlib.sha256(source_payload).hexdigest()
        ),
        frozen_source_bindings=config.get("source_bindings", {}),
        wall_seconds=time.monotonic() - inputs.started,
        self_cpu_seconds=time.process_time() - inputs.cpu_started,
        limits=dict(
            read_bytes=MAX_READ_BYTES,
            logical_read_bytes=MAX_READ_BYTES,
            file_bytes=MAX_FILE_BYTES,
            decoded_file_bytes=MAX_FILE_BYTES,
            output_bytes=MAX_OUTPUT_BYTES,
            wall_seconds=MAX_WALL_SECONDS,
        ),
        output_files=outputs.files,
        output_bytes=0,
        output_hash_scope="All listed outputs except this receipt; output_bytes includes receipt",
        new_scientific_work=ZERO_WORK,
    )
    for _ in range(8):
        size = outputs.bytes + len(_encoded(receipt))
        if receipt["output_bytes"] == size:
            break
        receipt["output_bytes"] = size
    payload = _encoded(receipt)
    _need(
        outputs.bytes + len(payload) == receipt["output_bytes"] <= MAX_OUTPUT_BYTES,
        "J2 report closing output budget exhausted",
    )
    inputs.check()
    with (out / "receipt.json").open("xb") as stream:
        stream.write(payload)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = report_run(args.worker, args.audit, args.out)
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "passed",
                    "worker_status",
                    "actual_cases",
                    "curve_points",
                    "read_bytes",
                    "output_bytes",
                    "wall_seconds",
                ]
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
