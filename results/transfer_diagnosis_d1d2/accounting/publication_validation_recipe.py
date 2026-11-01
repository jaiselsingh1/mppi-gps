from pathlib import Path
import hashlib,json,shutil,subprocess,sys,tempfile,time,os
root=Path('/Users/jaisel/Documents/mppi-gps')
receipt=root/'artifacts/2026-10-31_transfer_diagnosis_d1d2_001/publication_validation_001'
receipt.mkdir(exist_ok=False)
export=Path(tempfile.mkdtemp(prefix='walker-transfer-final-clean-export-'))
paths=sorted(set(subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().strip('\0').split('\0')))
for relative in paths:
 if not relative:continue
 source=root/relative
 if source.is_file():
  target=export/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
assert not (export/'runs').exists() and not (export/'artifacts').exists()
python=str(root/'.venv/bin/python')
checks=[('pytest',[python,'-B','-m','pytest','-q']),('ruff',[str(root/'.venv/bin/ruff'),'check','src','scripts','tests']),('repository',[python,'-B','scripts/check_repository.py']),('cli',[python,'-B','scripts/train_mppi_gps.py','study','transfer-diagnosis','--help'])]
for capsule in sorted((export/'results').glob('*/capsule.json')):
 checks.append(('inspect_'+capsule.parent.name,[python,'-B','scripts/train_mppi_gps.py','inspect','--run',str(capsule.parent),'--verify','--json']))
checks.append(('portable_render',[python,'-B','scripts/diagnostics/report_transfer_diagnosis.py','--run','results/transfer_diagnosis_d1d2','--out',str(receipt/'portable_report')]))
results=[]
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['XDG_CACHE_HOME']=str(receipt/'cache')
for name,cmd in checks:
 started=time.perf_counter();result=subprocess.run(cmd,cwd=export,text=True,capture_output=True,env=env)
 stdout=receipt/(name+'.stdout.txt');stderr=receipt/(name+'.stderr.txt');stdout.write_text(result.stdout);stderr.write_text(result.stderr)
 item={'name':name,'command':cmd,'cwd':str(export),'exit_code':result.returncode,'wall_seconds':time.perf_counter()-started,'stdout_sha256':hashlib.sha256(stdout.read_bytes()).hexdigest(),'stderr_sha256':hashlib.sha256(stderr.read_bytes()).hexdigest(),'stdout_tail':result.stdout.splitlines()[-3:]}
 results.append(item);print(json.dumps(item),flush=True)
 if result.returncode:
  (receipt/'clean_export.json').write_text(json.dumps({'export':str(export),'results':results,'passed':False},indent=2)+'\n');sys.exit(result.returncode)
original=root/'artifacts/2026-10-31_transfer_diagnosis_d1d2_001/report_001'
figures={p.name:hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((receipt/'portable_report'/p.name).read_bytes()).hexdigest() for p in original.glob('*.png')}
assert len(figures)==9 and all(figures.values())
(receipt/'clean_export.json').write_text(json.dumps({'export':str(export),'files':len(paths),'excluded_ignored_evidence':True,'results':results,'nine_pngs_match_full_report':figures,'passed':True},indent=2)+'\n')
print(json.dumps({'passed':True,'export':str(export),'nine_pngs_match':all(figures.values())}))
