#!/bin/sh
# PREREG_loop.md: the loop, seeds 0-2, 15 rounds; reference plain PPO (motion2 task_seed0-2) resumed to chunk 15.
cd ~/mppi-gps-claude-sac
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 OMP_NUM_THREADS=2
for s in 0 1 2; do d=runs/loop/seed$s; [ -e $d/done ] && continue; mkdir -p $d
  .venv-rl/bin/python -u rss_checks/rl/loop_rl.py --seed $s --rounds 15 --out $d >> $d.log 2>&1 &
  sleep 20; done
for s in 0 1 2; do d=runs/rc/motion2/task_seed$s; [ -e $d/done ] && continue
  .venv-rl/bin/python -u rss_checks/rl/motion_rl.py --w 0 --seed $s --chunks 15 --out $d >> $d.log 2>&1 &
  sleep 20; done
wait
