"""Bounded full-trajectory PI: preparation, prerequisites, execution and review.

Help, source freezing and metadata inspection do not import the simulator or
construct a neural network. Research execution requires a sealed launch file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from ..provenance import sha, write

ROOT = Path(__file__).resolve().parents[4]


def freeze_source(out: Path) -> dict:
    """Export committed tracked bytes to a create-only, read-only source tree."""
    changed = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True
    ).strip()
    if changed:
        raise ValueError("Commit tracked changes before exporting frozen source")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in sorted(filter(None, names)):
        source = ROOT / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Only regular committed files may be exported: {name}")
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o444)
        rows.append(dict(path=name, bytes=target.stat().st_size, sha256=sha(target)))
    value = dict(
        schema="trajectory-pi-source-1",
        git_commit=commit,
        files=rows,
        roster_sha256=hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    manifest = out.parent / "source_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    write(manifest, value)
    manifest.chmod(0o444)
    for directory, _, _ in os.walk(out, topdown=False):
        Path(directory).chmod(0o555)
    return dict(source=str(out.resolve()), manifest=str(manifest.resolve()), commit=commit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("freeze", help="export committed source, without executing science")
    item.add_argument("--out", type=Path, required=True)
    item = commands.add_parser(
        "prepare", help="bind and copy existing inputs; no fitting or physics"
    )
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item = commands.add_parser(
        "prepare-native", help="copy audited worker-native inputs and exact parent streams"
    )
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--validation-manifest", type=Path, required=True)
    item.add_argument("--previous-prepared", type=Path, required=True)
    item = commands.add_parser(
        "native-baselines", help="one bounded capture/replay/candidate validation"
    )
    item.add_argument("--launch", type=Path, required=True)
    item = commands.add_parser("preflight", help="run the separately bounded prerequisite block")
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--prepared", type=Path, required=True)
    item.add_argument("--source-manifest", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item = commands.add_parser("run", help="execute only a fully sealed, passing launch")
    item.add_argument("--launch", type=Path, required=True)
    item = commands.add_parser(
        "audit", help="independent saved-evidence audit; no new physics/fits"
    )
    item.add_argument("--run", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--stopped", action="store_true")
    item = commands.add_parser("report", help="render saved measurements and joint-angle traces")
    item.add_argument("--run", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item = commands.add_parser("replay-report", help="compare preserved source/replay arrays only")
    item.add_argument("--source", type=Path, required=True)
    item.add_argument("--replay", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item = commands.add_parser("origin-replay", help="one frozen, bounded origin-host replay")
    item.add_argument("--launch", type=Path, required=True)
    item = commands.add_parser("inspect", help="read compact metadata, never reopen checkpoints")
    item.add_argument("--run", type=Path, required=True)
    item.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        value = freeze_source(args.out)
    elif args.command == "prepare":
        from ..trajectory_pi_inputs import prepare

        value = prepare(args.config, args.out)
    elif args.command == "prepare-native":
        from ..trajectory_pi_inputs import prepare_native

        value = prepare_native(
            args.config, args.out, args.validation_manifest, args.previous_prepared
        )
    elif args.command == "native-baselines":
        from ..trajectory_pi_native_baselines import main as native_main

        return native_main(["--launch", str(args.launch)])
    elif args.command == "preflight":
        from ..trajectory_pi_preflight import run_preflight

        value = run_preflight(args.config, args.prepared, args.source_manifest, args.out)
    elif args.command == "run":
        from ..trajectory_pi_study import run_loop

        value = run_loop(args.launch)
    elif args.command == "audit":
        from ..audits.trajectory_pi import main as audit_main

        arguments = ["--run", str(args.run), "--out", str(args.out)]
        return int(audit_main(arguments + (["--stopped"] if args.stopped else [])) or 0)
    elif args.command == "report":
        from ..reporting.trajectory_pi import render

        value = render(args.run, args.out)
    elif args.command == "replay-report":
        from ..reporting.trajectory_pi_replay import replay_report

        value = replay_report(args.source, args.replay, args.out)
    elif args.command == "origin-replay":
        from ..trajectory_pi_replay_validation import main as origin_main

        return origin_main(["--launch", str(args.launch)])
    else:
        from ..reporting.trajectory_pi import inspect

        value = inspect(args.run, verify=args.verify)
    # Receipts remain on disk; keep terminal output compact.
    print(
        json.dumps(
            {
                key: value.get(key)
                for key in ("schema", "status", "source", "manifest", "commit", "output", "errors")
                if key in value
            },
            indent=2,
        )
    )
    failed = args.command in {"preflight", "run"} and value.get("status") in {
        "FAIL_STOPPED",
        "FAILED_STOPPED",
        "PRELAUNCH_STOPPED",
    }
    return 1 if failed or value.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
