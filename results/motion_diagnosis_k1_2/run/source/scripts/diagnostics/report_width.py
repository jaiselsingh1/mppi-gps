"""Render saved W1 width diagnostics without new inference or physics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gps.mppi_gps.reporting.width import main

if __name__ == "__main__":
    raise SystemExit(main())
