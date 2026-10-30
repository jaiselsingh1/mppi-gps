"""Compatibility entry point; implementation lives in the research library."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.gps.mppi_gps.reporting.walker_experiments import (
    ROOT,
    load_run,
    main,
    outcome_panel,
    plot_coupling,
    plot_student,
)

__all__ = [
    "ROOT",
    "load_run",
    "outcome_panel",
    "plot_student",
    "plot_coupling",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
