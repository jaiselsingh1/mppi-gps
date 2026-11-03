"""Bounded local source/receipt review; never opens scientific worker output."""

from __future__ import annotations

import hashlib
import io
import json
import resource
import subprocess
import tarfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

START = time.monotonic()
ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
RUN = BASE / "freeze_001/frozen_run"
COMMIT = "8c678e507dad1e22708f02a121886d7951d1da44"
READ_LIMIT = 96 * 1024**2
OUTPUT_LIMIT = 2 * 1024**2
REQUEST_HASH = "33d8f030a283f1455a994e9427f69b225c96f827e899b1639891ad8671822eec"
READS = 0
BINDINGS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    global READS
    path = Path(path).resolve()
    require(time.monotonic() - START <= 60, "Review wall limit")
    require(path.is_file() and path.stat().st_size <= READ_LIMIT - READS, "Read allocation")
    payload = path.read_bytes()
    READS += len(payload)
    BINDINGS[str(path)] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    return payload


def load(name):
    return json.loads(read(BASE / name))


def digest(data):
    return hashlib.sha256(data).hexdigest()


config_bytes = read(RUN / "config.json")
config = json.loads(config_bytes)
manifest_bytes = read(RUN / "manifest.json")
manifest = json.loads(manifest_bytes)
require(config["source_commit"] == COMMIT, "Wrong frozen source commit")
require(manifest["config_sha256"] == digest(config_bytes), "Config/source seal mismatch")
require(config["automatic_followup"] is False and config["guidance_launch_authorized"] is False, "Guidance authorized")
source_roster = {}
for row in manifest["files"]:
    require(row["file"] not in source_roster, "Duplicate source file")
    path = (RUN / "source" / row["file"]).resolve()
    require(path.is_relative_to(RUN / "source"), "Escaped frozen source")
    payload = read(path)
    require(digest(payload) == row["sha256"], "Frozen source changed: " + row["file"])
    source_roster[row["file"]] = dict(bytes=len(payload), sha256=digest(payload))
require(source_roster[config["protocol"]]["sha256"] == config["protocol_sha256"], "Protocol seal changed")
require(source_roster["configs/studies/similarity_j1_0.json"]["sha256"] == config["recipe_sha256"], "Recipe seal changed")
attached = read(Path("/Users/jaisel/.codex/attachments/9a70937b-132f-451b-9107-77d80488fb7f/Pasted text.txt"))
tracked_request = read(ROOT / config["user_request"]["file"])
require(attached == tracked_request, "The checked-in request differs from the attachment")
require(digest(attached) == REQUEST_HASH == config["user_request"]["sha256"], "Exact request hash changed")
require(source_roster[config["user_request"]["file"]] == dict(bytes=6534, sha256=REQUEST_HASH), "Frozen request binding differs")

# Authenticate all 975 frozen source payloads against the local immutable Git
# object, without fetching or reading any current scientific outcome.
command = ["git", "archive", "--format=tar", COMMIT, *source_roster]
archive = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=20)
READS += len(archive.stdout) + len(archive.stderr)
require(READS <= READ_LIMIT, "Local Git source-read allocation")
with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as stream:
    git_roster = {}
    for member in stream.getmembers():
        if member.isdir():
            continue
        require(member.isfile() and member.name in source_roster, "Unexpected Git archive member")
        value = stream.extractfile(member).read()
        git_roster[member.name] = dict(bytes=len(value), sha256=digest(value))
require(git_roster == source_roster, "Frozen source does not equal published Git bytes")

freeze = load("freeze_001/receipt.json")
deploy = load("deploy_001/stdout.json")
deployment_transport = load("deploy_001/receipt.json")
preflight = load("preflight_001/stdout.json")
prepare = load("remote_prepare_001/stdout.json")
prepare_transport = load("remote_prepare_001/receipt.json")
prepared_read = load("prepared_read_001/stdout.json")
prepared = prepared_read["files"]
ci = load("ci_001/receipt.json")
ci_status = load("ci_001/status.json")
ci_log = read(BASE / "ci_001/logs.txt").decode()
engineering = load("validation_001/summary.json")
wrapper_failure = load("validation_001/export_validation_001/original_wrapper_failure.json")
orphan_closure = load("validation_001/export_validation_001/original_validator_group_closure.json")
replacement_observer = load("validation_001/export_validation_001/replacement_observer.json")
recovery = load("validation_001/export_validation_001/recovery_001/summary.json")
synthetic = load("synthetic_fit_001/result.json")
synthetic_exit = load("synthetic_fit_001/process_receipt.json")
operator = load("operator_001/validation_001/receipt.json")
launch_intent = load("remote_launch_001/protocol.json")

for receipt in (freeze, deploy, prepare):
    require(receipt["passed"] is True and receipt["source_commit"] == COMMIT, "Publication/preparation source differs")
    require(receipt["config_sha256"] == digest(config_bytes), "Preparation config differs")
    require(receipt["source_manifest_sha256"] == digest(manifest_bytes), "Preparation manifest differs")
require(ci["passed"] and ci["source_commit"] == ci_status["headSha"] == COMMIT, "Final CI source changed")
require(ci_status["conclusion"] == "success" and digest(ci_log.encode()) == ci["logs_sha256"], "CI closure/log binding")
require("1668 passed, 3 skipped, 1 xfailed" in ci_log, "CI expected test count missing")
require('"commands": 51' in ci_log, "CI clean-export command count missing")
require(prepared["preparation.json"]["passed"] is True, "Preparation did not pass")
for role, filename in (("reference", "reference.pt"), ("normalization", "normalization.npz")):
    pin = config["inputs"][role]
    bound = prepared["input_manifest.json"]["files"]["prepared/" + filename]
    require(all(pin[key] == bound[key] == preflight["files"][role][key] for key in ("bytes", "sha256")), "Input pin differs")
renderer = prepared["prepared/renderer.json"]
require(renderer["encoder_functional_probe"]["passed"] is True, "Encoder functional probe failed")
require(renderer["mode"] == "kinematic_schematic", "Rendering mode changed")
require(preflight["free_bytes"] >= config["limits"]["global_output_bytes"], "Capacity prerequisite failed")

groups = config["groups"]
splits = Counter(group["split"] for group in groups)
require(splits == dict(development=8, protected=8), "Source-group split differs")
require(len({g["id"] for g in groups}) == 16 and all(len(g["learners"]) == 2 for g in groups), "Source/learner roster")
roles = {key: set() for key in ("acquisition_seed", "planner_seed", "policy_seed", "fit_seed", "evaluation")}
evaluation_panels = []
for group in groups:
    for key in ("acquisition_seed", "planner_seed"):
        roles[key].add(group[key])
    for learner in group["learners"]:
        for key in ("policy_seed", "fit_seed"):
            roles[key].add(learner[key])
        require(len(learner["evaluation_seeds"]) == len(set(learner["evaluation_seeds"])) == 16, "Reset panel length")
        roles["evaluation"].update(learner["evaluation_seeds"])
        evaluation_panels.append(learner["evaluation_seeds"])
seen = set()
for role, seeds in roles.items():
    require(not seen.intersection(seeds), "Seed roles overlap")
    seen.update(seeds)
    require(sorted(seeds) == prepared["preparation.json"]["seed_roles"][role], "Prepared seed role differs")
require(len(roles["acquisition_seed"]) == len(roles["planner_seed"]) == 16, "Source seeds reused")
require(len(roles["policy_seed"]) == len(roles["fit_seed"]) == 32, "Learner seeds reused")
dev_resets = {seed for g in groups if g["split"] == "development" for l in g["learners"] for seed in l["evaluation_seeds"]}
protected_resets = {seed for g in groups if g["split"] == "protected" for l in g["learners"] for seed in l["evaluation_seeds"]}
require(not dev_resets.intersection(protected_resets), "Development/protected reset leakage")

k, h, n = config["mppi"]["K"], config["mppi"]["H"], 1000
fits, streams, teacher_sources, probes = 64, 32, 32, 208
source_model = 16 * ((n + 1) * k * h + n * 2 * h)
probe_model = probes * (k * h + 2 * h)
native = teacher_sources * n * 2 + streams * n + fits * n + fits * 16 * n
model = source_model + probe_model
updates, backward = fits * 2000, fits * 2000 * 256
neural = backward + fits * n + streams * n + fits * n + fits * 16 * n + 16 * n
calls = updates + fits + streams * n + fits * n + fits * 16 * n + 16 * n
maximum = dict(fits=fits, optimizer_steps=updates, backward_rows=backward, neural_rows=neural, prediction_calls=calls, native_controls=native, model_controls=model, physics_substeps=4 * (native + model), model_queries=16 * (n + 1 + 2 * n) + probes * 3, live_queries=native, resets=teacher_sources + fits * 16, restores=32 + streams + fits + probes, snapshot_reads=native + 1184 + probes + 180, external_mj_forward_calls=2 * (teacher_sources + fits * 16) + 32 + streams + fits + probes + 180, contact_measurements=native + probes)
for name, value in maximum.items():
    require(value <= config["work_limits"][name], "Planned work exceeds declared limit: " + name)
require(config["learning"]["updates"] == 2000 and config["learning"]["hidden_dim"] == 128 and config["learning"]["unique_rows"] == 1000, "Learner intervention changed")

note = dict(
    schema="walker-j1-independent-prelaunch-review-v1",
    created_at_utc=datetime.now(timezone.utc).isoformat(),
    scope="Read-only local review of frozen prelaunch source and compact engineering/preparation receipts after the science launch was initiated; this is not the later independent scientific audit or a review of any J1 outcome.",
    prelaunch_bindings_passed=True,
    scientific_audit_performed=False,
    scientific_audit_passed=None,
    protected_labels_read=False,
    network_calls=0,
    new_research_experiments=0,
    new_scientific_work=dict(native_controls=0, model_controls=0, neural_rows=0, backward_rows=0, optimizer_steps=0, fits=0),
    invariants=dict(source_commit=COMMIT, config_sha256=digest(config_bytes), source_manifest_sha256=digest(manifest_bytes), protocol_sha256=config["protocol_sha256"], recipe_sha256=config["recipe_sha256"], exact_user_request_sha256=REQUEST_HASH, exact_user_request_bytes=6534, attachment_equals_frozen_checked_in_request=True, published_git_source_equals_frozen_source=True, source_files=len(source_roster), source_payload_bytes=sum(r["bytes"] for r in source_roster.values()), automatic_followup=False, guidance_launch_authorized=False),
    source_bindings=source_roster,
    seed_contract=dict(split_groups=dict(splits), sources_per_group=2, fresh_learner_streams_per_group=2, incoming_packages=32, maximum_learning_cases=64, seed_roles={key: sorted(value) for key, value in roles.items()}, evaluation_panels=evaluation_panels, unique_evaluation_reset_ids=len(roles["evaluation"]), panels_are_shared_within_group=True, development_protected_disjoint=True, scope="Role-separated IDs and preparation bindings checked; no new historical-state or student outcome was inspected."),
    ceilings=dict(derived_upper_work=maximum, declared_work_limits=config["work_limits"], limits=config["limits"], formulas=dict(teacher_model_controls="16 * ((1000+1)*256*48 + 1000*2*48) = 198340608; initial two passes and one thereafter, plus raw-mean and selector-mean verifications", probe_model_controls="208 * (256*48 + 2*48) = 2575872; all noninitial with two mean verifications", native_controls="32*1000 teacher + 32*1000 teacher replay + 32*1000 incoming + 64*1000 reproduction + 64*16*1000 standalone = 1184000", physics_substeps="4 * (200916480 model + 1184000 native) = 808401920", neural_rows="64*2000*256 fitting + 64*1000 pre-score + 32*1000 incoming + 64*1000 reproduction + 64*16*1000 standalone + 16*1000 reference = 33968000", prediction_calls="128000 fitting + 64 pre-score batches + 1136000 execution = 1264064", probe_roster="24 factorial + 168 extra gain perturbations + 4 original repeats + 8 hidden time/root-x + 4 warmstart = 208; four gain baselines reused within factorial", rendering="32 teacher + 32 incoming + 64 reproduction + 64 first standalone = 192 traces; at most192*251=48192 frames/position-forwards with zero added terminal holds"), contact_force_queries="Data-dependent contact multiplicity; hard cap10000000, no invented zero count", stop_scope="Falls, short-source ineligibility, development gate and any prerequisite/budget stop can reduce actual work; formulas are full-roster upper allocations, not measured scientific costs"),
    primary_gates=dict(source="protected MPPI only", groups=8, mp_cases=16, positive_cases_at_least=4, zero_cases_at_least=4, positive_groups_and_streams_at_least=3, motion_std_at_least=0.1, simultaneous_bootstrap_draws=9999, all_failure_walking="Unidentified, even if fitting or motion improves", no_guidance_from_partial_motion=True),
    preparation=dict(remote_runtime=preflight, renderer=renderer, prepared_metadata=prepared, remote_prepare=prepare, transport_receipt=prepare_transport, independent_raw_remote_file_hashes_not_read="Compact receipt JSON inspected; later saved-evidence audit must authenticate actual remote files and all scientific gates"),
    engineering_record=dict(final_published_ci=dict(receipt=ci, status=ci_status, tests=dict(passed=1668, skipped=3, expected_failure=1), clean_export_commands=51, scope="Fresh locked CPU GitHub run on exact published commit; test physics/forwards/fitting are engineering, not J1 learning outcomes"), initial_local_validation=dict(all_commands_passed=engineering["all_commands_passed"], source_changes_during_validation=engineering["source_changes_during_validation"], retained=True), original_clean_export_wrapper_failure=wrapper_failure, original_observer=replacement_observer, original_owned_group_closure=orphan_closure, local_recovery_summary=recovery, synthetic_calibration=dict(result=synthetic, process_receipt=synthetic_exit, scope="One known synthetic fixed2000-update wrapper calibration, not a research learning case or evidence of walking"), operator_synthetic_validation=operator, deployment=dict(remote_receipt=deploy, transport_receipt=deployment_transport), launch_intent=launch_intent, known_unknowns=["Original orphaned validator's actual exit status and CPU/rusage remain unavailable; group closure does not recover them.", "Original observer saw1032385821 logical output bytes before stopping; this exceeds the536870912 engineering ceiling. The violation and later disposal/recovery are preserved, not retroactively relabelled within-budget.", "Initial local full validation changed source while running and failed format; it is not the final-source validation certificate.", "Full-suite, clean-export, CI and repeated development tests include engineering native/model/neural/fitting work that was not individually instrumented; no complete primitive total can be claimed for that work.", "Nested child/operator/transport/CI wall and CPU measures are not additive; GitHub infrastructure CPU and transient storage details are not fully available.", "Earlier assistant read-only review commands were not given a primitive I/O/CPU meter. This note's own script receipt counts only its bounded review process."]),
    verdict="Frozen source, request, finite roster, declared work formulas, fresh published CI and compact preparation bindings close this prelaunch review. No conclusion about a score, walking, reproduction, simulator replay in the research block or scientific success follows. The later independent scientific audit and actual complete costs remain required.",
    input_bindings=BINDINGS,
)
payload = (json.dumps(note, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
require(len(payload) <= OUTPUT_LIMIT, "Review output ceiling")
with (OUT / "prelaunch_review.json").open("xb") as stream:
    stream.write(payload)
usage = resource.getrusage(resource.RUSAGE_SELF)
child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = dict(schema="walker-j1-independent-prelaunch-review-process-v1", passed=True, output=dict(file="prelaunch_review.json", bytes=len(payload), sha256=digest(payload)), program=dict(file="review.py", bytes=Path(__file__).stat().st_size, sha256=digest(Path(__file__).read_bytes())), source_read_bytes=READS, git_archive_stdout_bytes=len(archive.stdout), wall_seconds=time.monotonic() - START, own_user_seconds=usage.ru_utime, own_system_seconds=usage.ru_stime, git_child_user_seconds=child_usage.ru_utime, git_child_system_seconds=child_usage.ru_stime, peak_rss_bytes=usage.ru_maxrss, limits=dict(wall_seconds=60, read_bytes=READ_LIMIT, output_bytes=OUTPUT_LIMIT), new_research_experiments=0, new_scientific_work=note["new_scientific_work"], protected_labels_read=False, network_calls=0, receipt_write_follows=True)
with (OUT / "receipt.json").open("x") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
print(json.dumps(dict(passed=True, output=str(OUT / "prelaunch_review.json"), source_read_bytes=READS, wall_seconds=receipt["wall_seconds"])))
