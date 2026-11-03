"""Compatibility entry point for clean-export validation.

Canonical implementation: src.gps.mppi_gps.validation.export.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gps.mppi_gps.validation.export import main
from src.gps.mppi_gps.validation.reports import (
    CLOSURE_SCIENCE_KEYS as CLOSURE_SCIENCE_KEYS,
)
from src.gps.mppi_gps.validation.reports import (
    SIMILARITY_CLOSING_HASHES as SIMILARITY_CLOSING_HASHES,
)
from src.gps.mppi_gps.validation.reports import (
    SIMILARITY_SCIENCE_KEYS as SIMILARITY_SCIENCE_KEYS,
)
from src.gps.mppi_gps.validation.reports import (
    _closure_binding as _closure_binding,
)
from src.gps.mppi_gps.validation.reports import (
    _closure_json as _closure_json,
)
from src.gps.mppi_gps.validation.reports import (
    _closure_payload as _closure_payload,
)
from src.gps.mppi_gps.validation.reports import (
    _closure_require as _closure_require,
)
from src.gps.mppi_gps.validation.reports import (
    _closure_zero as _closure_zero,
)
from src.gps.mppi_gps.validation.reports import (
    similarity_export_commands as similarity_export_commands,
)
from src.gps.mppi_gps.validation.reports import (
    verify_consequence_recovery_capsules as verify_consequence_recovery_capsules,
)
from src.gps.mppi_gps.validation.reports import (
    verify_report_outputs as verify_report_outputs,
)
from src.gps.mppi_gps.validation.reports import (
    verify_similarity_report as verify_similarity_report,
)
from src.gps.mppi_gps.validation.reports import (
    verify_student_consequence_report as verify_student_consequence_report,
)
from src.gps.mppi_gps.validation.reports import (
    verify_student_visitation_report as verify_student_visitation_report,
)
from src.gps.mppi_gps.validation.reports import (
    verify_teacher_proposal_report as verify_teacher_proposal_report,
)

if __name__ == "__main__":
    raise SystemExit(main())
