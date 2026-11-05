#!/bin/sh
set -eu
probe_root=/home/jaisel/mppi-gps-codex-acrobot/2026-11-05_energy_discovery_v1_1_001
for part in near-upright energy; do
    test ! -e "$probe_root/$part"
    test ! -e "$probe_root/$part.launcher.pid"
done
for part in near-upright energy; do
    setsid nohup sh -c '
        part=$1
        probe_root=$2
        cd "$probe_root/source" || exit 90
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 \
          /usr/bin/time -v -o "$probe_root/$part.process.txt" \
          /home/jaisel/mppi-gps/.venv/bin/python -m src.gps.mppi_gps.acrobot_energy_probe execute \
          --config configs/studies/acrobot_energy_discovery_v1_1.json \
          --part "$part" --preflight "$probe_root/preflight" --out "$probe_root/$part"
        result=$?
        printf "%s\n" "$result" > "$probe_root/$part.exit"
        exit "$result"
    ' acrobot-probe "$part" "$probe_root" \
      > "$probe_root/$part.stdout" 2> "$probe_root/$part.stderr" < /dev/null &
    printf "%s\n" "$!" > "$probe_root/$part.launcher.pid"
    printf "launched %s wrapper %s\n" "$part" "$!"
done
