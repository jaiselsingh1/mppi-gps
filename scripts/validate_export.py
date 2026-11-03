"""Compatibility entry point for clean-export validation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gps.mppi_gps.validation.export import main
from src.gps.mppi_gps.validation.reports import (
    verify_report_outputs as verify_report_outputs,
)
from src.gps.mppi_gps.validation.reports import (
    verify_student_consequence_report as verify_student_consequence_report,
)
from src.gps.mppi_gps.validation.reports import (
    verify_teacher_proposal_report as verify_teacher_proposal_report,
)

if __name__ == "__main__":
    raise SystemExit(main())
