"""One-shot external owner for the frozen, prepublished D1/D2 worker."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

RUN = Path('/home/jaisel/mppi-gps-workspace/runs/2026-10-31_transfer_diagnosis_d1d2_001')
COMMIT = '95f0a55e490204f9ca898b6f8891c56e95e7a824'
CONFIG_SHA = 'be462b70ac7b5a0b420b1b137c998f6a78f3a52c198cbbffaf06609f9aa0031a'
SEAL_SHA = 'bf93b0a479b073bfe32d8d05e92b207506fda9398d5e82e83d15c40cc3c12033'
PYTHON = '/home/jaisel/mppi-gps/.venv/bin/python'

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def write(name,value):
    with (RUN/name).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

if __name__=='__main__':
    assert Path(__file__).resolve().parent == RUN
    assert sha(RUN/'config.json')==CONFIG_SHA
    assert sha(RUN/'input_manifest.json')==SEAL_SHA
    manifest=json.loads((RUN/'manifest.json').read_text())
    seal=json.loads((RUN/'input_manifest.json').read_text())
    assert sha(RUN/'manifest.json')==seal['source_manifest_sha256']
    for item in manifest['files']:
        assert sha(RUN/'source'/item['file'])==item['sha256'],item['file']
    for name,binding in seal['files'].items():
        assert sha(RUN/name)==binding['sha256'],name
        assert (RUN/name).stat().st_size==binding['bytes'],name
    with (RUN/'launch.lock').open('x') as stream:
        stream.write(str(os.getpid())+'\n')
    assert not (RUN/'result.json').exists() and not (RUN/'evidence').exists()
    env=os.environ.copy()
    settings={'MUJOCO_GL':'disable','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    env.update(settings)
    command=[PYTHON,'-B','scripts/train_mppi_gps.py','study','transfer-diagnosis','worker','--out',str(RUN)]
    receipt={'implementation_commit':COMMIT,'config_sha256':CONFIG_SHA,'input_manifest_sha256':SEAL_SHA,'source_manifest_sha256':sha(RUN/'manifest.json'),'launcher_sha256':sha(Path(__file__)),'host':'robopil5090','command':command,'cwd':str(RUN/'source'),'environment':settings,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launcher_pid':os.getpid(),'prelaunch_ci':'https://github.com/jaiselsingh1/mppi-gps/actions/runs/41259416102','worker_attempt':1}
    write('launch_receipt.json',receipt)
    started=time.perf_counter()
    with (RUN/'worker.stdout.txt').open('xb') as stdout,(RUN/'worker.stderr.txt').open('xb') as stderr:
        worker=subprocess.Popen(command,cwd=RUN/'source',env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
        write('worker_started.json',{'pid':worker.pid,'launcher_pid':os.getpid()})
        code=worker.wait()
    write('launch_exit.json',{'exit_code':code,'wall_seconds':time.perf_counter()-started,'worker_pid':worker.pid,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker_attempt':1,'stdout_sha256':sha(RUN/'worker.stdout.txt'),'stderr_sha256':sha(RUN/'worker.stderr.txt')})
