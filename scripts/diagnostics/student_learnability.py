"""Compatibility entry point; implementation lives in the research library."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.gps.mppi_gps.experiments.student_learnability import (
    ROOT,
    check_seed_freshness,
    default_config,
    feature_benefit,
    features,
    fingerprint,
    main,
    normalization,
    planned_work,
    prepare,
    selected_events,
    sha,
    summary,
    worker,
    write,
)

__all__ = [
    "ROOT",
    "features",
    "normalization",
    "default_config",
    "planned_work",
    "feature_benefit",
    "check_seed_freshness",
    "prepare",
    "worker",
    "main",
    "sha",
    "write",
    "fingerprint",
    "summary",
    "selected_events",
]

if __name__ == "__main__":
    raise SystemExit(main())
