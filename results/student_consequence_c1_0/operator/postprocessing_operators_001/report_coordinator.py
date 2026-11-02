"""Bound the saved-only C1 report using its frozen watchdog implementation."""
import argparse,json,os,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True);args=parser.parse_args()
run,out,audit=args.run.resolve(),args.out.resolve(),args.audit.resolve()
sys.path.insert(0,str(run/'source'))
from src.gps.mppi_gps.lifecycle import verify
from src.gps.mppi_gps.proposal_launch import bounded_process
from src.gps.mppi_gps.provenance import write
verify(run)
config=json.loads((run/'config.json').read_text())
assert (run/'worker_exit.json').is_file() and (audit/'result/audit.json').is_file()
out.mkdir(parents=True,exist_ok=False)
environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',PYTHONPATH=str(run/'source'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MUJOCO_GL','CUDA_VISIBLE_DEVICES'):environment[key]=config['runtime'][key]
old=os.sched_getaffinity(0);os.sched_setaffinity(0,config['runtime']['cpu_affinity'])
try:
 receipt=bounded_process([sys.executable,str(run/'source/scripts/train_mppi_gps.py'),'study','student-consequence','report','--run',str(run),'--out',str(out/'result'),'--audit',str(audit/'result')],cwd=run/'source',environment=environment,output=out/'stdout.txt',wall_seconds=config['limits']['report_wall_seconds'],stdout_bytes=8*1024**2,peak_rss_bytes=config['limits']['report_peak_rss_bytes'])
finally:os.sched_setaffinity(0,old)
write(out/'exit.json',receipt)
raise SystemExit(int(receipt['exit_code']!=0))
