"""Check portable evidence, historical manifests and local documentation links.

This command uses only the standard library and never executes research workers.
Use --frozen-run explicitly to verify source/input seals in local full runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_binding(root: Path, name: str, binding: dict[str, Any], errors: list[str]) -> None:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        errors.append(f"Path escapes evidence root: {name}")
        return
    if not path.is_file():
        errors.append(f"Missing: {path}")
        return
    if "bytes" in binding and path.stat().st_size != binding["bytes"]:
        errors.append(f"Size mismatch: {path}")
    if digest(path) != binding["sha256"]:
        errors.append(f"Hash mismatch: {path}")


def check_capsules(root: Path, errors: list[str]) -> int:
    count = 0
    for path in sorted((root / "results").glob("*/capsule.json")):
        capsule = json.loads(path.read_text())
        if isinstance(capsule.get("files"), dict):
            bindings = capsule["files"].items()
        elif isinstance(capsule.get("capsule_files"), list):
            # The preserved SAC continuation capsule used repository-relative
            # paths. Read its original contract without rewriting its manifest.
            prefix = path.parent.relative_to(root)
            converted = {}
            for binding in capsule["capsule_files"]:
                name = Path(binding["path"])
                if not name.is_relative_to(prefix):
                    errors.append(f"Capsule binding escapes its directory: {name}")
                    continue
                relative_name = str(name.relative_to(prefix))
                if relative_name in converted:
                    errors.append(f"Duplicate capsule binding: {name}")
                    continue
                converted[relative_name] = binding
            bindings = converted.items()
        else:
            errors.append(f"Unrecognized capsule file roster: {path.relative_to(root)}")
            continue
        for name, binding in bindings:
            check_binding(path.parent, name, binding, errors)
            count += 1
    if not count:
        errors.append("No capsule files found")
    return count


def check_archives(root: Path, errors: list[str]) -> int:
    """Validate the two source archives with explicit per-file byte seals."""
    count = 0
    for path in sorted((root / "archive").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        for binding in manifest.get("files", []):
            name = binding.get("file")
            if name is None and manifest.get("schema") == "controller-loop-handoff-snapshot-v1":
                name = binding.get("snapshot_path")
            if not isinstance(name, str):
                errors.append(f"Missing archive payload path: {path.relative_to(root)}")
                continue
            check_binding(path.parent, name, binding, errors)
            count += 1
    if not count:
        errors.append("No sealed historical source files found")
    return count


def _anchors(path: Path) -> set[str]:
    anchors = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', path.read_text()))
    duplicates: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not re.match(r"^#{1,6}\s", line):
            continue
        heading = re.sub(r"^#+\s+|\s+#+$", "", line).strip().lower()
        heading = re.sub(r"<[^>]+>", "", heading)
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        suffix = duplicates.get(slug, 0)
        duplicates[slug] = suffix + 1
        anchors.add(slug if suffix == 0 else f"{slug}-{suffix}")
    return anchors


def check_links(root: Path, errors: list[str]) -> int:
    # Frozen audit reports and archive notes are exact evidence, not maintained docs.
    pages = [root / "README.md", root / "results/README.md"]
    pages += sorted((root / "docs").rglob("*.md"))
    pages += sorted((root / "results").glob("*/README.md"))
    count = 0
    for page in pages:
        for target in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", page.read_text()):
            target = target.strip().strip("<>")
            # Optional Markdown title follows the path after whitespace and a quote.
            target = re.split(r'\s+["\']', target, maxsplit=1)[0]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            destination = (page.parent / unquote(parsed.path)).resolve() if parsed.path else page
            count += 1
            if not destination.exists():
                errors.append(f"Broken link in {page.relative_to(root)}: {target}")
            elif parsed.fragment and destination.suffix == ".md":
                if unquote(parsed.fragment) not in _anchors(destination):
                    errors.append(f"Missing anchor in {page.relative_to(root)}: {target}")
    return count


def check_frozen(run: Path, errors: list[str]) -> int:
    """Verify recorded source/config/input bytes without raw arrays or execution."""
    manifest = json.loads((run / "manifest.json").read_text())
    count = 0
    for binding in manifest["files"]:
        check_binding(run / "source", binding["file"], binding, errors)
        count += 1
    check_binding(run, "config.json", {"sha256": manifest["config_sha256"]}, errors)
    if manifest.get("input_sha256"):
        check_binding(run, "input.pt", {"sha256": manifest["input_sha256"]}, errors)
    seal_path = run / "input_manifest.json"
    if seal_path.is_file():
        for name, binding in json.loads(seal_path.read_text()).get("files", {}).items():
            binding = {"sha256": binding} if isinstance(binding, str) else binding
            check_binding(run, name, binding, errors)
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--frozen-run", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    errors: list[str] = []
    counts = {
        "capsule_files": check_capsules(args.root, errors),
        "historical_files": check_archives(args.root, errors),
        "local_links": check_links(args.root, errors),
        "frozen_files": sum(check_frozen(run, errors) for run in args.frozen_run),
    }
    print(json.dumps(dict(counts=counts, errors=errors), indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
