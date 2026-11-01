"""Compatibility entry point for the S1.1 one-shot CPU owner."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gps.mppi_gps.source_evaluation_launch import main

if __name__ == "__main__":
    raise SystemExit(main())
