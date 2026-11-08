#!/bin/sh
# PREREG_loop.md addendum ~01:55: theory-aligned arm (MPPI-state restart pool) and the self-state control, seeds 0-2;
# plain PPO reference seeds 1-2 continue (seed 0's first walking chunk is settled).
cd ~/mppi-gps-claude-sac
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 OMP_NUM_THREADS=2
for s in 0 1 2; do
  d=runs/loop/pool_seed$s; [ -e $d/done ] || { mkdir -p $d; .venv-rl/bin/python -u rss_checks/rl/loop_rl.py --pool runs/loop/pool_mppi.npz --seed $s --rounds 15 --out $d >> $d.log 2>&1 & sleep 20; }
  d=runs/loop/self_seed$s; [ -e $d/done ] || { mkdir -p $d; .venv-rl/bin/python -u rss_checks/rl/loop_rl.py --self_pool --seed $s --rounds 15 --out $d >> $d.log 2>&1 & sleep 20; }
done
for s in 1 2; do d=runs/rc/motion2/task_seed$s; [ -e $d/done ] && continue
  .venv-rl/bin/python -u rss_checks/rl/motion_rl.py --w 0 --seed $s --chunks 15 --out $d >> $d.log 2>&1 & sleep 20; done
wait
