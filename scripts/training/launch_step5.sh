#!/usr/bin/env bash
# Step 5 launcher (docs/experiment_log.md, 2026-10-21 entry): picks the Step 4
# cell with the preregistered rule, then starts MPPI-GPS (B) and DAgger for
# training seeds 0-2 and the Step 4 policy-tail test, all in the background.
# Usage, from the repository root on the run host:
#   scripts/training/launch_step5.sh runs/<grid-run>/report.json
set -euo pipefail
report="$1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
selection=$(.venv/bin/python -m scripts.diagnostics.walker_headroom select --report-path "$report")
echo "$selection" > "$(dirname "$report")/selection.json"
read -r K H RHO < <(echo "$selection" | .venv/bin/python -c \
  "import json, sys; s = json.load(sys.stdin); print(s['K'], s['H'], s['neff_fraction'])")
echo "selected K=$K H=$H n_eff/K=$RHO"
common=(--cfg.K "$K" --cfg.H "$H" --cfg.iterations 25 --cfg.episodes-per-iter 10 --cfg.steps 400 --cfg.workers 3)
for seed in 0 1 2; do
  for labeler in mdgps dagger; do
    nohup nice -n 10 .venv/bin/python -m scripts.training.train_walker_mdgps \
      --out-dir "runs/2026-10-22_step5_${labeler}_seed${seed}" --cfg.labeler "$labeler" --cfg.seed "$seed" \
      "${common[@]}" > "runs/step5_${labeler}_seed${seed}.log" 2>&1 &
  done
done
nohup nice -n 10 .venv/bin/python -m scripts.diagnostics.walker_headroom run \
  --out-dir runs/2026-10-22_walker_headroom_tail_001 --mode tail --horizons "$H" --samples "$K" \
  --neff-fraction "$RHO" --workers 3 > runs/headroom_tail_001.log 2>&1 &
echo "launched"
