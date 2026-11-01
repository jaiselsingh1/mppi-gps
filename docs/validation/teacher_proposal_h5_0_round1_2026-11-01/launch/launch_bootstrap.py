"""Start one frozen, authenticated H5 coordinator; no retries or direct worker."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/home/jaisel/mppi-gps-workspace')
RUN = ROOT/'runs/2026-11-01_teacher_proposal_h5_0_r1_001'
OUT = ROOT/'artifacts/2026-11-01_teacher_proposal_h5_0_r1_001/operator_001'
PYTHON = '/home/jaisel/mppi-gps/.venv/bin/python'
cfg = json.loads((RUN/'config.json').read_text())
assert cfg['launch_authorized'] is True
assert cfg['eligibility']['decision'] == 'approved_single_round_exception'
assert cfg['eligibility']['useful_learning_prerequisite_satisfied'] is False
assert cfg['eligibility']['second_round_authorized'] is False
assert cfg['budgets']['worker_wall_seconds'] == 21600
assert not any((RUN/name).exists() for name in ('launch_ownership.json','worker_result.json','worker_exit.json','evidence'))
OUT.mkdir(parents=True, exist_ok=False)
command = [PYTHON,'-B',str(RUN/'source/scripts/train_mppi_gps.py'),
           'study','teacher-proposal','run','--out',str(RUN)]
def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
intent = {'schema':'walker-h5-scientific-launch-intent-v1','command':command,
          'cwd':str(RUN/'source'),'created_unix_seconds':time.time(),
          'config_sha256':hashlib.sha256((RUN/'config.json').read_bytes()).hexdigest(),
          'worker_wall_seconds':21600,'research_rounds_authorized':1,'second_round_authorized':False,
          'outer_coordinator_owns_worker_watchdog':True,'retry_authorized':False}
write(OUT/'intent.json',intent)
environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
environment.pop('PYTHONPATH',None)
with (OUT/'coordinator_stdout.txt').open('xb') as stdout:
    process=subprocess.Popen(command,cwd=RUN/'source',env=environment,
                             stdin=subprocess.DEVNULL,stdout=stdout,stderr=subprocess.STDOUT,
                             start_new_session=True)
try:
    stat=Path(f'/proc/{process.pid}/stat').read_text()
    birth_ticks=int(stat.rsplit(') ',1)[1].split()[19])
except FileNotFoundError:
    birth_ticks=None
record=dict(intent,coordinator_pid=process.pid,coordinator_birth_ticks=birth_ticks,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            coordinator_poll=process.poll(),worker_started_by_bootstrap_directly=False,
            operator_directory=str(OUT))
write(OUT/'bootstrap_receipt.json',record)
print(json.dumps(record,indent=2))
