"""Metadata-only publication checks; never imports the simulator or a policy."""
from __future__ import annotations

import hashlib
import json
import re
import resource
import subprocess
import time
from pathlib import Path

START = time.monotonic()
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
checks = []


def need(value: bool, label: str) -> None:
    if not value:
        raise ValueError(label)
    checks.append(label)


review = HERE / "saved_review_002"
manifest = json.loads((review / "final_review_manifest.json").read_text())
for row in manifest["files"]:
    path = review / row["file"]
    need(path.stat().st_size == row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "immutable independent-review artifact: " + row["file"])
for name in ("README.md", "docs/full_pi_abc_coordination.md", "docs/local_coupling_fitting_f1_0_proposal.md"):
    path = ROOT / name
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        if target.startswith(("https:", "http:", "#")):
            continue
        need((path.parent / target.split("#", 1)[0]).exists(), "document link: " + name + " -> " + target)
need(hashlib.sha256((ROOT / "configs/studies/trajectory_pi_p1_1.json").read_bytes()).hexdigest() == "4c26199f9c0bdf18d1057200feb010c5c8d7f417eda48224ee43e58940690089", "P1.1 configuration remains unchanged")
changed = subprocess.check_output(["git", "diff", "743447ad178da408cb89881e859d715070ff81b3", "--name-only", "--", "src", "configs", "results/trajectory_pi_p1_001", "results/trajectory_pi_p1_1_001", "results/trajectory_pi_replay_r0_1_001"], cwd=ROOT, text=True)
need(not changed.strip(), "scientific implementation/configuration and replay capsules unchanged")
usage = resource.getrusage(resource.RUSAGE_SELF)
receipt = dict(schema="walker-abc-publication-validation-1", passed=True, checks=checks, process_cpu_seconds_including_imports=usage.ru_utime + usage.ru_stime, wall_seconds=time.monotonic() - START, research_calls=dict(native_controls=0, model_controls=0, neural_forward_rows=0, optimizer_steps=0), scope="Publication metadata only. No tests, physics, fitting, actor or planner invocation.")
(HERE / "publication_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
rows = []
for path in sorted(HERE.rglob("*")):
    if path.is_file() and path.name != "coordination_manifest.json":
        rows.append(dict(path=str(path.relative_to(HERE)), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(HERE / "coordination_manifest.json").write_text(json.dumps(dict(schema="walker-abc-coordination-manifest-1", files=rows, self_excluded=True, no_experiment_launch=True), indent=2) + "\n")
print(json.dumps(dict(passed=True, checks=len(checks), process_cpu_seconds=receipt["process_cpu_seconds_including_imports"], wall_seconds=receipt["wall_seconds"], files=len(rows))))
