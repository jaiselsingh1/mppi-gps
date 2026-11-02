"""Compatibility entry point; implementation lives in the research library."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.gps.mppi_gps.reporting.guided_data_comparison import (
    main,
    sha,
    write,
)

__all__ = [
    "sha",
    "write",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
