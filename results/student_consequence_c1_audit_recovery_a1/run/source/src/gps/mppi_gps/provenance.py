"""Small provenance contracts shared by preparation, execution and reports.

File hashes, typed optimizer fingerprints, raw-array hashes and fitting-data
hashes intentionally have different encodings. Do not interchange them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


def sha(path: str | Path) -> str:
    """Hash file bytes without interpreting their contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default(value: Any) -> Any:
    """JSON default hook used by the saved-data studies."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def jsonable(value: Any) -> Any:
    """Legacy loop JSON conversion, including its string dictionary keys."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    return value


def write(path: str | Path, value: Any) -> None:
    """Create a new study JSON artifact; never replace existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=default)
        stream.write("\n")


def write_json(path: str | Path, value: Any) -> None:
    """Create a legacy-loop artifact with its original conversion contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint(value: Any) -> str:
    """Preserve the typed recursive optimizer/checkpoint fingerprint format."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if hasattr(item, "detach"):
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(json.dumps(["array", list(array.shape), str(array.dtype)]).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit((type(key).__name__, key))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(
                json.dumps([type(item).__name__, item], default=default, allow_nan=False).encode()
            )

    visit(value)
    return digest.hexdigest()


def selected_events(
    run: str | Path, select: Callable[[dict[str, Any]], bool]
) -> list[dict[str, Any]]:
    """Read a fully closed historical ledger and enforce its sequence IDs."""
    result = []
    with (Path(run) / "evidence/ledger.jsonl").open() as stream:
        last = None
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row["id"] != f"event-{index:09d}":
                raise ValueError("Historical ledger sequence changed")
            if select(row):
                result.append(row)
            last = row
    if last is None or last["event"] != "runtime_close":
        raise ValueError("Only closed historical evidence is admitted")
    return result
