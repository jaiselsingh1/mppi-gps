"""Compatibility entry point; implementation lives in the research library."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.gps.mppi_gps.experiments.guided_data_comparison import (
    ROOT,
    default,
    default_config,
    fingerprint,
    main,
    planned_work,
    prepare,
    selected_events,
    sha,
    summary,
    task_gate,
    validate_episode,
    worker,
    write,
)

__all__ = [
    "ROOT",
    "sha",
    "default",
    "write",
    "fingerprint",
    "default_config",
    "planned_work",
    "task_gate",
    "summary",
    "validate_episode",
    "selected_events",
    "prepare",
    "worker",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
