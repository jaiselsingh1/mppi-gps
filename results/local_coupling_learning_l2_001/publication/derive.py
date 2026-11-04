"""L2.0 saved-output publication derivation; no simulator/policy/fit imports."""
from pathlib import Path
import csv, hashlib, json, resource, shutil, time
import h5py
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(120,120))
start, cpu = time.monotonic(), time.process_time()
root=Path('/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi')
base=root/'runs/2026-11-03_local_coupling_learning_l2_001'
pub=base/'publication';pub.mkdir(exist_ok=False)
cap=root/'results/local_coupling_learning_l2_001'
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f: json.dump(x,f,indent=2);f.write('\n')
def bind(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
r=read(base/'worker/result.json');s=read(base/'supervisor_result.json');a=read(base/'audit/final/receipt.json');find=read(base/'audit/final/findings.json')
assert r['status']=='ROUND1_WALKING_NOT_CONFIRMED' and a['passed'] and len(r['acquisitions'])==24
assert len(r['rounds'])==1 and len(r['rounds'][0]['evaluations'])==224 and len(r['rounds'][0]['fits'])==6
assert all(not x['qualifying_gain'] for x in r['rounds'][0]['criterion']['details'])
groups={}
for spec in r['accounting']['child_receipts']:
 rel=Path(spec['path']).relative_to(base)
 phase=str(rel.parts[1]);counts=read(spec['path'])['counts'];g=groups.setdefault(phase,{})
 for k,v in counts.items():g[k]=g.get(k,0)+v
phases={x: {k:s[x][k] for k in ('wall_seconds','process_tree_cpu_seconds','supervisor_cpu_seconds','waited_children_cpu_seconds','exit_code','peak_sampled_tree_rss_bytes')} for x in ('worker','audit_external','report_external')}
work=dict(schema='l2-publication-accounting-1',scientific_status=r['status'],phases=groups,worker_plus_preparation=r['accounting']['counts'],audit=a,
 external_phases=phases,active_pool_wall_seconds=s['wall_seconds'],worker_internal_cpu_seconds=r['accounting']['worker_process_cpu_seconds'],
 active_tree_cpu_seconds=sum(p['process_tree_cpu_seconds'] for p in phases.values()),
 preparation_cpu_wall_charged_reservation=1800,
 total_cpu_with_preparation_reservation_before_publication=1800+sum(p['process_tree_cpu_seconds'] for p in phases.values()),
 total_wall_with_preparation_reservation_before_publication=1800+s['wall_seconds'],
 total_forward_rows=r['accounting']['counts']['completed_neural_rows']+a['completed_neural_rows'],
 count_scope='Worker totals include prepared 8 synthetic updates/2048 backward rows/2671 forwards and copied-stream preparation. Do not add child phases to their included totals. Audit forwards are additional; publication is saved-only.',
 retries='One sandbox launcher failed on ps before any worker/physics/fit. Same sealed launch then ran with process access. No scientific job or study was retried.',
 uncertainty='Preparation/CLI/hash/test and denied-launch overhead is conservatively charged as the frozen1800CPU/1800wall reservation, not claimed as exactly measured. Physical diskI/O unavailable; retained logical bytes separately catalogued.',
 skipped_conditional_round2=True,unused_budgets_are_not_spent=True,historical_costs='P1/R0/P1.1/six-round/A-B-C preserved separately; not imported into this new study.')
write(pub/'accounting.json',work)
diag=[];first_actions={};initial=[]
for row in r['acquisitions']:
 m=read(row['manifest']['path']);d=Path(row['manifest']['path']).parent
 vals={k:[] for k in ('survival_fraction','fallen_weight','ESS','lambda_value')}
 with h5py.File(m['bank_stats']['path'],'r') as f:
  for key in sorted(f):
   for k in vals: vals[k].append(float(f[key].attrs[k]))
  first=dict(f[sorted(f)[0]].attrs)
 with h5py.File(m['trace']['path'],'r') as f: first_actions[(m['arm'],m['slot'])]=f['actions'][0]
 diag.append(dict(arm=m['arm'],slot=m['slot'],banks=m['banks'],first_bank_survival_fraction=float(first['survival_fraction']),
  statistics={k:dict(mean=float(np.mean(v)),minimum=float(np.min(v)),maximum=float(np.max(v))) for k,v in vals.items()},outcome=m['outcome']))
 if m['arm']!='independent':
  other=base/f"worker/acquisition/r1/shared/independent/slot{m['slot']}/full_t000_p0.npz"
  with np.load(other,allow_pickle=False) as z,np.load(d/'full_t000_p0.npz',allow_pickle=False) as p:
   checks={k:bool(np.array_equal(z[k],p[k],equal_nan=True)) for k in ('actions','posts','C','D','mask','U_before','initial_state','initial_warmstart','initial_integration')}
  initial.append(dict(arm=m['arm'],slot=m['slot'],same_initial_candidate_bank=all(checks.values()),checks=checks))
assert all(x['same_initial_candidate_bank'] for x in initial)
first_rms={arm:float(np.sqrt(np.mean([(first_actions[(arm,i)]-first_actions[('independent',i)])**2 for i in range(8)]))) for arm in ('weak','large')}
write(pub/'controller_diagnostics.json',dict(schema='l2-saved-controller-diagnostics-1',episodes=diag,paired_initial_banks=initial,first_executed_same_state_rms=first_rms,
 aggregation='All actual banks, executable prefixes; means are bank-equal within episode. First bank and first actual action are paired at identical context. Later same-seed banks share innovations but diverge with carried plans/states; pooled whole-rollout action RMS is not same-state action effect.',new_physics=0,new_neural_rows=0))
with (base/'report/standalone_outcomes.csv').open() as f: ev=list(csv.DictReader(f))
summary=[]
for key in sorted({(x['stream'],x['arm'],x['kind'],x['panel']) for x in ev}):
 rows=[x for x in ev if (x['stream'],x['arm'],x['kind'],x['panel'])==key]
 summary.append(dict(stream=int(key[0]),arm=key[1],kind=key[2],panel=key[3],episodes=len(rows),walking=sum(x['walking']=='True' for x in rows),full_survival=sum(x['full_survival']=='True' for x in rows),mean_controls=float(np.mean([int(x['controls']) for x in rows])),mean_complete_cost=float(np.mean([float(x['complete_cost']) for x in rows])),mean_velocity_mae=float(np.mean([float(x['velocity_mae']) for x in rows]))))
write(pub/'standalone_summary.json',summary)
write(pub/'independent_review.json',dict(schema='l2-independent-saved-review-summary-1',reviewer='/root/l2_saved_review',scope='Independent read-only review of frozen recipe, nativeprerequisite,24 closed teacher rosters,224 actual outcomes,final audit/report and bindings; no code edits, tests, neural inference, physics or fitting.',passed=True,classification='Competent different localexperience plus retention without walkinggain; R2correctly nottriggered.',limitations=['Eightcontexts fromone originalepisode','Three minibatchforks fromone actor','300control localteacher,1000control student','Whole-rollout actionRMS includes state divergence','No old-actorR2control, noequalcomputeLPcloning/DAgger, nosecondbody']))
# Catalogue closed study/source/evidence; publication directory is excluded explicitly.
files={};links={}
for p in sorted(base.rglob('*')):
 if p.is_relative_to(pub):continue
 if p.is_symlink():links[str(p.relative_to(base))]=str(p.readlink());continue
 if p.is_file():files[str(p.relative_to(base))]=bind(p)
write(pub/'raw_file_manifest.json',dict(schema='l2-closed-study-files-1',base=str(base),scope='All closed run files excluding publication/ derivatives; symlinks separately recorded, no duplicate physical-file charge for pointers.',files=files,symlinks=links,file_count=len(files),total_logical_file_bytes=sum(x['bytes'] for x in files.values()),availability='LOCAL_ONLY_NO_REMOTE_MIRROR_ASSERTED'))
shutil.copyfile(__file__,pub/'derive.py')
write(pub/'derivation_receipt.json',dict(schema='l2-publication-derivation-1',cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,model_controls=0,native_controls=0,physics_substeps=0,neural_rows=0,backward_rows=0,optimizer_steps=0,input_result=bind(base/'worker/result.json'),source=bind(pub/'derive.py'),files_hashed=len(files),bytes_hashed=sum(x['bytes'] for x in files.values()),limits=dict(cpu_seconds=120),new_scientific_execution=False))
# Exact compact evidence copies; checkpoints,rawarrays,H5,bulkledgers/sourceexport stay external.
copy=[]
for name in ('worker/result.json','worker/config.json','worker/selection.json','supervisor_result.json','external_worker.json'):
 copy.append(Path(name))
for area in ('audit/final','report','publication'):
 copy.extend(p.relative_to(base) for p in (base/area).rglob('*') if p.is_file())
for area in ('worker/acquisition','worker/fits','worker/evaluations','worker/native_preflight','worker/corpus'):
 copy.extend(p.relative_to(base) for p in (base/area).rglob('*.json') if p.stat().st_size<100000)
copy.append(Path('engineering/sandbox_launch_001/attempt.json'))
for rel in sorted(set(copy)):
 dest=cap/rel;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copyfile(base/rel,dest)
print(json.dumps(dict(first_action_rms=first_rms,phases=groups,raw_files=len(files),raw_bytes=sum(x['bytes'] for x in files.values()),derivation=read(pub/'derivation_receipt.json'),copied=len(set(copy))),indent=2))
