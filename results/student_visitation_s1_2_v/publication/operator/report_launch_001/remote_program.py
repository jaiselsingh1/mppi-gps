import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,os,resource,time
run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_visitation_s1_2_v_002');art=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-02_student_visitation_s1_2_v_002')
sys.path.insert(0,str(run/'source'))
from src.gps.mppi_gps.proposal_launch import bounded_process
from src.gps.mppi_gps.experiments.student_visitation import _clean_exit
from src.gps.mppi_gps.provenance import write,sha
config=json.loads((run/'config.json').read_text());assert config['source_commit']=='70075397bfb2966dff97e325c4104915f03c9581'
started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(run/'source'),MPLBACKEND='Agg')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MUJOCO_GL','CUDA_VISIBLE_DEVICES'):env[key]=config['runtime'][key]
receipt=bounded_process([sys.executable,str(run/'source/scripts/train_mppi_gps.py'),'study','visitation','report','--run',str(run),'--out',str(art/'report_001'),'--audit',str(art/'audit_001/result/audit.json')],cwd=run/'source',environment=env,output=art/'report_stdout.txt',wall_seconds=config['limits']['report_wall_seconds']-1,stdout_bytes=1024**2,peak_rss_bytes=config['limits']['report_peak_rss_bytes'])
write(art/'report_exit.json',receipt)
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(schema='walker-visitation-v-report-operator-v1',passed=_clean_exit(receipt),stage_wall_seconds=time.monotonic()-started,child_user_seconds=usage.ru_utime-before.ru_utime,child_system_seconds=usage.ru_stime-before.ru_stime,new_neural_rows=0,new_physics=0,new_fitting=0)
if result['passed']:
 result['report_receipt_sha256']=sha(art/'report_001/report_receipt.json')
 result['report_receipt']=json.loads((art/'report_001/report_receipt.json').read_text())
write(art/'report_operator_receipt.json',result);print(json.dumps(result));sys.exit(0 if result['passed'] else 1)
