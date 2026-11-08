#!/bin/sh
# PREREG_pull.md: pull (MPPI pulled toward the policy, weight W) vs fresh (same, w = 0), seeds 0-2.
# Round 0 is the same computation in both arms (no policy yet, w = 0): it runs once in fresh and pull starts from
# fresh's round-0 checkpoint. Usage: launch_pull.sh W
W=$1; [ -n "$W" ] || { echo "usage: launch_pull.sh W"; exit 1; }
cd ~/mppi-gps-claude-sac
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 OMP_NUM_THREADS=2
export CUDA_MPS_PIPE_DIRECTORY=$HOME/mppi-gps-claude-sac/mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/mppi-gps-claude-sac/mps/log
PY=".venv-rl/bin/python -u rss_checks/rl/loop_rl.py --n_walks 16 --walk_len 500"
for s in 0 1 2; do
  (
    f=runs/pull/fresh_seed$s; p=runs/pull/pull_seed$s; mkdir -p $f $p
    if [ ! -e $f/state.pkl ]; then $PY --fresh_mppi 0 --seed $s --rounds 1 --out $f >> $f.log 2>&1; rm -f $f/done; fi
    if [ ! -e $p/state.pkl ]; then cp $f/state.pkl $p/state.pkl; cp $f/rounds.jsonl $p/rounds.jsonl; fi
    [ -e $f/done ] || $PY --fresh_mppi 0 --seed $s --rounds 15 --out $f >> $f.log 2>&1 &
    sleep 30
    [ -e $p/done ] || $PY --fresh_mppi $W --seed $s --rounds 15 --out $p >> $p.log 2>&1 &
    wait
  ) &
  sleep 20
done
wait
