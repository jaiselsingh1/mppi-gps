#!/bin/sh
# Task screen (PREREG_screen.md): PPO (published config, seeds 0-1) and MPPI (5 episodes) per task, two queues.
cd ~/mppi-gps-claude-sac
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 OMP_NUM_THREADS=2
TASKS="AcrobotSwingup AcrobotSwingupSparse FingerTurnHard BallInCup CartpoleSwingupSparse PendulumSwingup HopperHop"
( for t in $TASKS; do for s in 0 1; do d=runs/screen/ppo/${t}_seed$s
    [ -e $d/done ] && continue; rm -rf $d
    .venv-rl/bin/python -u rss_checks/rl/screen_ppo.py --task $t --seed $s --out $d > $d.log 2>&1
  done; done; echo ppo-queue-done > runs/screen/ppo.done ) &
( for t in $TASKS; do d=runs/screen/mppi/$t
    [ -e $d/episodes.jsonl ] && [ $(wc -l < $d/episodes.jsonl) -ge 5 ] && continue; rm -rf $d
    .venv-rl/bin/python -u rss_checks/rl/screen_mppi.py --task $t --episodes 5 --out $d > $d.log 2>&1
  done; echo mppi-queue-done > runs/screen/mppi.done ) &
wait
