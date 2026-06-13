#!/bin/bash
# Relaunch unfinished long-running jobs after a container restart.
# Invoked by the SessionStart hook (and safe to run manually); checks for a
# resumable TD3 training state and restarts it detached if not running.
cd "$(dirname "$0")/.." || exit 0

if [ -f runs/walker_td3_matched2/train_state.pt ]; then
  if ! pgrep -f "train_walker_rl" > /dev/null 2>&1; then
    PYTHONPATH="$PWD" nohup uv run python scripts/train_walker_rl.py \
      --match-task-cost --total-steps 1500000 --resume \
      --out-dir runs/walker_td3_matched2 >> /tmp/walker_td3m2.log 2>&1 &
    echo "resumed walker TD3 training (pid $!)"
  fi
fi
