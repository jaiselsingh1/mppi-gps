#!/bin/bash
# Relaunch unfinished long-running jobs after a container restart.
# Invoked by the SessionStart hook (and safe to run manually); checks for a
# resumable TD3 training state and restarts it detached if not running.
cd "$(dirname "$0")/.." || exit 0

# Don't relaunch if training already finished (sentinel) — avoids the
# hook resurrecting a completed run and spawning duplicate processes.
if [ -f runs/walker_td3_matched2/.done ]; then
  exit 0
fi

if [ -f runs/walker_td3_matched2/train_state.pt ] && [ ! -f runs/walker_td3_matched2/.done ]; then
  if ! pgrep -f "train_walker_rl" > /dev/null 2>&1; then
    PYTHONPATH="$PWD" nohup uv run python scripts/train_walker_rl.py \
      --match-task-cost --total-steps 1500000 --resume \
      --out-dir runs/walker_td3_matched2 >> /tmp/walker_td3m2.log 2>&1 &
    echo "resumed walker TD3 training (pid $!)"
  fi
fi

# Resume an interrupted GPS run if a resume command was recorded and it has
# not finished. The launching command is written to runs/<name>/.resume_cmd.
for rc in runs/*/.resume_cmd; do
  [ -f "$rc" ] || continue
  d=$(dirname "$rc")
  [ -f "$d/.done" ] && continue
  if ! pgrep -f "$(basename "$d")" > /dev/null 2>&1; then
    log="/tmp/$(basename "$d").log"
    PYTHONPATH="$PWD" nohup bash -c "$(cat "$rc")" >> "$log" 2>&1 &
    echo "resumed GPS run $(basename "$d") (pid $!)"
  fi
done
