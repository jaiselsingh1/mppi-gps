"""Compatibility entry point for the focused frozen W2 update-budget contrast."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gps.mppi_gps.experiments.student_update_budget import main

if __name__ == "__main__":
    raise SystemExit(main())
