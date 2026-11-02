"""C1's bounded independent audit compatibility entrypoint."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gps.mppi_gps.experiments.student_consequence import main

if __name__ == "__main__":
    raise SystemExit(main(["audit", *sys.argv[1:]]))
