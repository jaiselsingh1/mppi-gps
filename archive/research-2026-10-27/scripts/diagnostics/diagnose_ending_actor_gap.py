"""Read-only component, preparation and credit diagnosis of curriculum003."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    a=parser.parse_args();start=time.perf_counter();root=a.run.resolve();out=a.out.resolve();out.mkdir(exist_ok=False)
    (out/'analyze.py').write_bytes(Path(__file__).read_bytes());refs={}
    def read(p):refs[str(p.relative_to(root))]=sha(p);return json.loads(p.read_text())
    result=read(root/'curriculum_result.json');audit=read(root/'audit_001.json');assert audit['status']=='PASS'
    base=root/'evidence/curriculum';assert sha(base/'ledger.jsonl')==result['query_ledger_sha256']
    owner=result['rounds'][1]['parent_actor_hash'];events={};fits=[];requests={}
    for line in (base/'ledger.jsonl').open():
        e=json.loads(line)
        if e['kind']=='episode_outcome':events[e['episode']]=e
        if e['kind']=='curriculum_update_requested':requests[e['id']]=e
        if e['kind']=='curriculum_fit_requested' and e['behavior_actor_hash']==owner:fits.append(e)
    def load(ref):
        p=base/ref['file'];assert sha(p)==ref['sha256'];refs[str(p.relative_to(root))]=ref['sha256']
        with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    summaries={}
    for role,stem in [('RAW','final_teacher_RAW_'),('actor','final_teacher_actor_')]:
        selected=[v for k,v in events.items() if k.startswith(stem)];assert len(selected)==64
        raws=[load(e['raw']) for e in selected];states=np.stack([r['states'] for r in raws]);actions=np.stack([r['actions'] for r in raws])
        parts=np.stack([r['cost_components'].sum(0) for r in raws])
        summaries[role]=dict(mean_components=dict(zip(('effort','greed','detour','ending'),parts.mean(0).tolist())),
            mean_y_by_time=states[:,:,1].mean(0).tolist(),mean_actions_by_time=actions[:,:,0].mean(0).tolist(),
            mean_gate_y=float(states[:,12,1].mean()),gate_y_range=[float(states[:,12,1].min()),float(states[:,12,1].max())],
            first_goal_boundary_mean=float(np.mean([np.flatnonzero(s[:,1]<=-.1)[0] for s in states])),
            mean_complete_cost=float(parts.sum(1).mean()),capture=sum(bool(r['captured']) for r in raws))
    contrasts=[]
    for f in fits:
        req=requests[f['update_request_id']];raws=[]
        for path in f['query_refs'][:16]:
            p=Path(path);refs[str(p.relative_to(root))]=sha(p)
            with np.load(p,allow_pickle=False) as z:raws.append({k:z[k] for k in z.files})
        returns=np.stack([np.cumsum(r['raw_costs'][::-1])[::-1] for r in raws]).reshape(4,4,25)
        adv=returns-(returns.sum(1,keepdims=True)-returns)/3
        contrasts.append(dict(arm=req['arm'],update=req['update'],actor_hash=owner,
            ordinary_stochastic_mean_cost=float(returns[:,:,0].mean()),captures=sum(bool(r['captured']) for r in raws),
            episodes=16,advantage_squared_weight_active=float(np.square(adv[:,:,:12]).sum()),
            advantage_squared_weight_action_invariant=float(np.square(adv[:,:,12:]).sum()),
            limitation='Advantage weight energy, not parameter-gradient variance or a causal ablation.'))
    output=dict(schema='ending-actor-cost-gap-diagnosis-v1',actor_hash=owner,summaries=summaries,owned_stochastic_training=contrasts,
        source_sha256=sha(out/'analyze.py'),input_sha256=refs,wall_seconds=time.perf_counter()-start,
        new_controls=0,new_actor_rows=0,new_optimizer_steps=0,
        limits='Raw actual episodes only. Early arrival, objective distribution and noisy credit are distinct hypotheses; this diagnosis alone does not attribute update regression.')
    with (out/'diagnosis.json').open('x') as f:json.dump(output,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k!='input_sha256'},indent=2))

if __name__=='__main__':main()
