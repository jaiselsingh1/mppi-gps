#!/bin/sh
# Stage 1 of PREREG_motion.md: 3 arms x seeds 0-2, all at once in one capped scope (resumable: skips finished runs).
# A watcher applies the pre-set kill rule (motion seed never up 8 s from any start by chunk 8).
cd ~/mppi-gps-claude-sac
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in 0 1 2; do
  for arm in task task_motion motion; do
    d=runs/rc/motion2/${arm}_seed$s
    [ -e $d/done ] && continue
    case $arm in task) f="--w 0";; task_motion) f="--w 5 --task 1";; motion) f="--w 5 --task 0";; esac
    mkdir -p $d
    sh -c "echo \$\$ > $d.pid; exec .venv-rl/bin/python rss_checks/rl/motion_rl.py $f --seed $s --chunks 15 --out $d" >> $d.log 2>&1 < /dev/null &
    sleep 20
  done
done
( while pgrep -f "rss_checks/rl/motion_rl.py" > /dev/null; do
    for s in 0 1 2; do
      d=runs/rc/motion2/motion_seed$s; [ -e $d.killed ] && continue; [ -e $d/chunks.jsonl ] || continue
      n=$(wc -l < $d/chunks.jsonl)
      if [ "$n" -ge 8 ] && ! grep -q '"survived": [1-9]' $d/chunks.jsonl; then
        kill $(cat $d.pid) 2>/dev/null && echo "$(date) killed $d at chunk $n (never up 8 s)" > $d.killed
      fi
    done
    sleep 300
  done ) >> runs/rc/motion2/watch.log 2>&1 &
wait
