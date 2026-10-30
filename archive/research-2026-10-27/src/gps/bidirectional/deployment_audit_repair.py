"""Additive verifier-schema repair, inheriting already checked raw arithmetic."""
from collections import defaultdict
import json,time
from pathlib import Path
from .ending_audit import Evidence,require,checkpoint_numpy,policy_identity
from .deployment_audit_helpers import audit_steps

def repair(run):
 start=time.perf_counter();e=Evidence(run);old=e.json('audit_001.json');old_sha=e.hashes['audit_001.json'];errors=[];counts=defaultdict(int,old['counts'])
 require(old['status']=='FAIL' and old['errors']==["deployment own-tangent and actual update evidence: KeyError: 'learning_form'"] and not old['pending'],'Not the exact legacy barrier verifier failure')
 def unchanged(path):
  path=Path(path);relative=str(path.resolve().relative_to(e.root)) if path.is_absolute() else str(path)
  file=e.file(relative,old['input_sha256'][relative]);return file
 for file in ['config.json','curriculum_result.json','manifest.json','stage_complete.json','evidence/curriculum/ledger.jsonl','work_ledger.jsonl']:unchanged(file)
 c=json.loads(unchanged('config.json').read_text());r=json.loads(unchanged('curriculum_result.json').read_text());require(c['schema']=='bidirectional-deployment-adjoint-dev0','Repair restricted to legacy barrier schema')
 keep={'checkpoint','episode_outcome','adjoint_step_requested','adjoint_gradient_requested','adjoint_gradient_complete','adjoint_step_complete','projected_parameter_update_requested','projected_parameter_update_complete'}
 rows=[]
 for line in unchanged('evidence/curriculum/ledger.jsonl').open():
  row=json.loads(line)
  if row['kind'] in keep:rows.append(row)
 checkpoints={};episodes={};outcomes={};base=Path('evidence/curriculum')
 for row in rows:
  if row['kind']=='checkpoint':
   file=unchanged(row['file']);require(e.hashes[str(file.relative_to(e.root))]==row['sha256'],'Checkpoint reference differs');saved=checkpoint_numpy(file);policy_identity(saved['learner']['policy']);checkpoints[row['label']]=saved
  if row['kind']=='episode_outcome':
   ref=row['raw'];unchanged(base/ref['file']);episodes[row['episode']]=e.npz(base,ref);outcomes[row['episode']]=row['outcome']
 try:audit_steps(e,base,c,r,rows,episodes,outcomes,checkpoints,counts)
 except Exception as ex:errors.append('Repaired own-tangent adapter: '+type(ex).__name__+': '+str(ex))
 new_gradient_refs={str(base/row['raw']['file']):row['raw']['sha256'] for row in rows if row['kind']=='adjoint_gradient_complete'}
 for name,value in e.hashes.items():
  if name!='audit_001.json':require(old['input_sha256'].get(name,new_gradient_refs.get(name))==value,'Helper input was not bound by original audit or unchanged ledger')
 return dict(schema='independent-deployment-audit-schema-repair-v1',status='FAIL' if errors else 'PASS',errors=errors,pending=[],evidence_arithmetic_valid=not errors,scientific_success=bool(old['detail'].get('independent_success') and not errors),counts=dict(counts),detail=old['detail'],input_sha256={**old['input_sha256'],**e.hashes},audit_wall_seconds=time.perf_counter()-start,new_simulator_controls=0,new_actor_rows=0,new_optimizer_steps=0,
  inherited_full_audit=dict(file='audit_001.json',sha256=old_sha,status=old['status'],errors=old['errors'],wall_seconds=old['audit_wall_seconds']),
  limits=old['limits']+['Original complete raw transition/cost/work/causal checks are inherited from hash-bound audit001. This repair rechecks all episode/checkpoint inputs and the complete missing adapter, without decoding every model-query array again. No scientific configuration or run is changed.',
   'Barrier/full-network derivative values remain source-bound; the active physics-derivative row counter denotes the recorded work, not independent neural/gradient replay.'])
