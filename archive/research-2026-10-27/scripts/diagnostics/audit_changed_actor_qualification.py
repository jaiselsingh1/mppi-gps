"""Independent arithmetic audit of the finite changed-controller qualification."""
import argparse,hashlib,importlib.abc,json,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--auditor',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 root=a.run.resolve();packet=a.auditor.resolve();out=a.out.resolve();out.mkdir(exist_ok=False);start=time.perf_counter();(out/'source.py').write_bytes(Path(__file__).read_bytes())
 manifest=json.loads((packet/'manifest.json').read_text())
 for r in manifest['files']:assert sha(packet/r['file'])==r['sha256']
 blocked=[]
 class NoWork(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0] in ('torch','mujoco','warp','mujoco_warp','scipy') or fullname.startswith(('src.ending','src.envs','src.gps.bidirectional.policy','src.gps.bidirectional.deployment_adjoint','src.gps.bidirectional.deployment_loop')):
    blocked.append(fullname);raise ImportError('Pure qualification audit prohibits '+fullname)
 sys.meta_path.insert(0,NoWork());sys.path.insert(0,str(packet/'source'))
 import numpy as np
 from src.gps.bidirectional.ending_audit import Evidence,same,require,packed,transition_arithmetic,checkpoint_numpy,policy_identity,resolutions,controller_identity,COMPONENTS
 from src.gps.bidirectional.deployment_audit_helpers import head_geometry
 from src.gps.bidirectional.return_curriculum_audit import compare
 e=Evidence(root);r=e.json('receipt.json');refs=r['input_sha256'];source_parent=next(Path(k) for k in refs if k.endswith('parent_inputs/checkpoints/round1_parent.pt'));source_changed=next(Path(k) for k in refs if k.endswith('checkpoints/initial_guided_u00.pt'))
 for file,expected in refs.items():require(sha(file)==expected,'Qualification input changed')
 known={}
 for name,file in [('parent',source_parent),('changed',source_changed)]:
  policy=checkpoint_numpy(file)['learner']['policy'];known[name]=policy_identity(policy)
 changed_policy=policy
 task=json.loads((source_changed.parent.parent/'config.json').read_text())['task'];base=Path('evidence/qualification');rows=e.rows(base/'ledger.jsonl');work=e.rows('work_ledger.jsonl');pending=[]
 req={z['query_id']:z for z in rows if z['kind']=='query_requested'};done={z['query_id']:z for z in rows if z['kind']=='query_complete'}
 require(set(req)==set(done) and len(req)==800,'Qualification query closure differs')
 global_counts={}
 for name,unit in [('scalar_transition','controls'),('neural','rows'),('autograd_grad','calls'),('optimizer_update','updates')]:
  before,after=resolutions(work,name,unit,pending);global_counts[name]=sum(z[unit] for z in before.values())
 require(not pending and global_counts==dict(scalar_transition=800,neural=960,autograd_grad=9,optimizer_update=0),'Qualification global accounting differs')
 used=set();panels={k:[] for k in known}
 for z in rows:
  if z['kind']!='episode_outcome':continue
  raw=e.npz(base,z['raw']);s,u=raw['states'],raw['actions'];owner=packed(raw,'owner_json');o=z['outcome'];name='parent' if z['episode'].startswith('fresh_parent') else 'changed'
  require(owner['actor_hash']==known[name] and owner['controller_hash']==controller_identity(owner) and owner['deterministic'],'Wrong deployment actor')
  expected,parts,cost,flags=transition_arithmetic(s[:-1],u,task);same(s[1:],expected,'Actual transition differs');same(raw['raw_costs'],cost,'Return differs')
  same(raw['cost_components'],np.stack([parts[k] for k in COMPONENTS],-1),'Components differ')
  same(o['cost'],cost.sum(),'Outcome differs');require(o['capture']==bool(flags['ending_good'][-1]) and s[-1,0]==25,'Incomplete outcome')
  for i,qid in enumerate(raw['query_ids'].tolist()):
   require(qid not in used,'Actual control reused');used.add(qid);qi=e.npz(base,req[qid]['inputs']);qo=e.npz(base,done[qid]['result'])
   require(req[qid]['mode']=='live' and req[qid]['owner']==owner,'Wrong raw query owner');same(qi['initial_state'],s[i],'Prestate differs');same(qi['actions'],u[i],'Action differs');same(qo['states'],s[i+1],'Poststate differs')
  panels[name].append(o)
 require(used==set(req),'Unattributed actual controls')
 for name,panel in panels.items():
  require([z['context_id'] for z in panel]==[f'ordinary:{i}' for i in range(8100000,8100016)],'Fresh roots differ')
  require(panel==r['summaries'][name]['outcomes'],'Saved qualification panel differs')
 criterion=compare(panels['parent'],panels['changed']);require(criterion==r['criterion'],'Qualification decision differs')
 prefix=e.rows('source_prefix.jsonl');gr=[z for z in prefix if z['kind']=='adjoint_gradient_requested'][-1];records=[]
 for ref in gr['actual_own_refs']:
  file=source_changed.parent.parent/'evidence/curriculum'/ref['file'];require(sha(file)==ref['sha256'],'Derivative own evidence changed')
  with np.load(file,allow_pickle=False) as f:records.append({k:f[k].copy() for k in f.files})
 with np.load(root/'changed_geometry.npz',allow_pickle=False) as f:geometry={k:f[k].copy() for k in f.files}
 g,A,J=head_geometry(geometry,records,changed_policy,task)
 same(g,geometry['cost_gradient'],'Analytic cost derivative differs',atol=2e-10);same(A,geometry['ending_gradients'],'Analytic endpoints differ',atol=2e-10);same(J,geometry['control_jacobian'],'Analytic action derivatives differ',atol=2e-10)
 require(np.allclose(g,geometry['autograd_cost'],atol=3e-6,rtol=3e-5) and np.allclose(A,geometry['autograd_ending'],atol=3e-6,rtol=3e-5),'Saved changed-actor autograd disagreement')
 result=dict(status='PASS',criterion=criterion,counts=global_counts,new_scalar_controls=0,new_actor_rows=0,new_optimizer_steps=0,analytic_policy_jacobian_rows_rechecked=80,wall_seconds=time.perf_counter()-start,blocked_import_attempts=blocked,input_sha256=e.hashes,source_sha256=sha(out/'source.py'),auditor_manifest_sha256=sha(packet/'manifest.json'),geometry_sha256=sha(root/'changed_geometry.npz'),limits='Arithmetic and saved analytic/autograd comparison; neural forwards are input/source/checkpoint bound, not repeated. Fresh qualification only, no MPC or next-student claim.')
 with (out/'audit.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))

if __name__=='__main__':main()
