"""Create an immutable, pure-NumPy verifier packet (no experiment work)."""
from pathlib import Path
import argparse,hashlib,json,shutil

DRIVER=r'''"""Frozen arithmetic-only verifier; all producers and neural work prohibited."""
import argparse,contextlib,hashlib,importlib.abc,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=('test','audit','repair'));p.add_argument('--run');p.add_argument('--label');a=p.parse_args()
root=Path(__file__).resolve().parent
m=json.loads((root/'manifest.json').read_text())
for row in m['files']:
 f=root/row['file'];assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
blocked=[]
class NoWork(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if (fullname.split('.')[0] in ('torch','mujoco','warp','mujoco_warp','scipy') or fullname in ('src.gps.bidirectional.return_curriculum','src.gps.bidirectional.deployment_adjoint','src.gps.bidirectional.deployment_loop') or fullname.startswith(('src.envs','src.ending','src.gps.bidirectional.policy','src.gps.bidirectional.learning','src.gps.bidirectional.search','src.gps.bidirectional.ending_pilot','src.gps.bidirectional.conditional_feedback','src.gps.bidirectional.early_context_loop','src.gps.bidirectional.geometry_return','src.gps.bidirectional.coherent_return','src.gps.bidirectional.first_action_completion'))):
   blocked.append(fullname);raise ImportError('Arithmetic audit prohibits '+fullname)
sys.meta_path.insert(0,NoWork());os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(root/'source'));os.chdir(root/'source');start=time.perf_counter()
if a.mode=='test':
 import pytest
 with (root/'test_stdout.txt').open('x') as out,(root/'test_stderr.txt').open('x') as err:
  with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
   status=int(pytest.main(['tests','-q','-p','no:cacheprovider','--basetemp',str(root/'test_tmp')]))
 r=dict(passed=status==0,pytest_exit_code=status,blocked_import_attempts=blocked,new_scalar_controls=0,new_actor_rows=0,new_fits=0,wall_seconds=time.perf_counter()-start)
 with (root/'test_receipt.json').open('x') as f:json.dump(r,f,indent=2,sort_keys=True)
 print(json.dumps(r));sys.exit(status)
assert a.run and a.label and json.loads((root/'test_receipt.json').read_text())['passed']
run=Path(a.run).resolve();target=run/('audit_002.json' if a.mode=='repair' else 'audit_001.json');receipt=root/('execution_'+a.label+'.json');assert not target.exists() and not receipt.exists()
with (root/('begin_'+a.label+'.json')).open('x') as f:json.dump(dict(mode=a.mode,run=str(run),utc_seconds=time.time(),new_scalar_controls=0,new_actor_rows=0),f,indent=2)
try:
 if a.mode=='repair':
  from src.gps.bidirectional.deployment_audit_repair import repair
  r=repair(run)
 else:
  from src.gps.bidirectional.return_curriculum_audit import audit
  r=audit(run)
except Exception as ex:
 with (root/('failure_'+a.label+'.json')).open('x') as f:json.dump(dict(error_type=type(ex).__name__,error=str(ex),wall_seconds=time.perf_counter()-start,new_scalar_controls=0,new_actor_rows=0,new_fits=0),f,indent=2)
 raise
with target.open('x') as f:json.dump(r,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
rr=dict(audit_file=str(target),audit_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),status=r['status'],scientific_success=r['scientific_success'],blocked_import_attempts=blocked,new_scalar_controls=0,new_actor_rows=0,new_fits=0,wall_seconds=time.perf_counter()-start,analytic_policy_jacobian_rows_rechecked=r['counts'].get('analytic_policy_jacobian_rows_rechecked',0),metric_solves_rechecked=r['counts'].get('metric_solves_rechecked',0))
with receipt.open('x') as f:json.dump(rr,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(rr,indent=2));sys.exit(0 if r['status']=='PASS' else 2)
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve();out.mkdir(exist_ok=False)
 files=['src/__init__.py','src/gps/__init__.py','src/gps/bidirectional/__init__.py',
  'src/gps/bidirectional/ending_audit.py','src/gps/bidirectional/ending_completion_audit.py',
  'src/gps/bidirectional/return_curriculum_audit.py','src/gps/bidirectional/deployment_audit_helpers.py',
  'src/gps/bidirectional/deployment_audit_repair.py',
  'tests/test_bidirectional_return_curriculum_audit.py','tests/test_bidirectional_deployment_audit.py']
 for name in files:
  dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dst)
 (out/'verify.py').write_text(DRIVER);shutil.copy2(__file__,out/'freeze_packet.py')
 manifest=dict(schema='independent-deployment-auditor-v1',files=[])
 for file in sorted(out.rglob('*')):
  if file.is_file():manifest['files'].append(dict(file=str(file.relative_to(out)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print(out)

if __name__=='__main__':main()
