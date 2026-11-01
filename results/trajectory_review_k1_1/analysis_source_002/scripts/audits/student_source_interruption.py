"""Launch the separate saved-only S1 interruption postmortem."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gps.mppi_gps.audits.source_interruption import main

if __name__ == "__main__":
    raise SystemExit(main())
