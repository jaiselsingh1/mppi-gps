"""Compatibility entry point for the historical SAC continuation workflow.

Implementation: src.gps.mppi_gps.experiments.sac_continuation. Numerical and
legacy rank behavior are retained. Use a fresh output when intentionally
replaying this historical protocol; cleanup never launches it.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MODULE = "src.gps.mppi_gps.experiments.sac_continuation"


def main(argv: list[str] | None = None) -> int:
    return importlib.import_module(_MODULE).main(argv)


def __getattr__(name: str):
    return getattr(importlib.import_module(_MODULE), name)


if __name__ == "__main__":
    raise SystemExit(main())
