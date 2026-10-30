"""Compatibility entry point; implementation lives in the research library."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.gps.mppi_gps.experiments.adaptive_coupling import (
    ARMS,
    HIGH_RHO,
    PARENT_HASH,
    ROOT,
    admission,
    changed_trajectory,
    check_environment,
    check_teacher_decisions,
    default_config,
    environment_record,
    evaluate_cohort,
    export,
    fingerprint,
    main,
    paired_resets,
    planned_work,
    prepare,
    seed_audit,
    sha,
    summary,
    task_gate,
    worker,
    write,
)

__all__ = [
    "ROOT",
    "PARENT_HASH",
    "HIGH_RHO",
    "ARMS",
    "environment_record",
    "check_environment",
    "default_config",
    "planned_work",
    "admission",
    "task_gate",
    "changed_trajectory",
    "check_teacher_decisions",
    "paired_resets",
    "evaluate_cohort",
    "seed_audit",
    "prepare",
    "worker",
    "export",
    "main",
    "sha",
    "write",
    "fingerprint",
    "summary",
]

if __name__ == "__main__":
    raise SystemExit(main())
