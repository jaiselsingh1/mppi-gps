"""Compatibility entry point for saved-only K1 diagnostic reporting."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.gps.mppi_gps.reporting.action_compatibility import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
