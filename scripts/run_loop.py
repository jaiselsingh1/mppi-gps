"""Run the MPPI loop (src/mppi_loop/loop.py). Example:

python scripts/run_loop.py --weight 1 --seed 1 --out runs/mppi_loop_w1_seed1
python scripts/run_loop.py --weight 0 --seed 1 --out runs/mppi_loop_w0_seed1      # no similarity
python scripts/run_loop.py --weight 0 --dagger --seed 1 --out runs/mppi_loop_dagger_seed1
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mppi_loop.loop import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
