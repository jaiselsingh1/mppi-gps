"""Bounded changed-controller derivative/layout and fresh-mean qualification."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve();out=args.out.resolve();out.mkdir(exist_ok=False);start=time.perf_counter()
    sources=[Path(__file__).resolve(),Path('src/gps/bidirectional/deployment_adjoint.py').resolve()]
    refs={str(p):sha(p) for p in sources}
    for i,p in enumerate(sources):(out/f'source_{i}_{p.name}').write_bytes(p.read_bytes())
    from scripts.diagnostics.run_bidirectional_tests import DurableWork,WorkPlugin
    from src.gps.bidirectional.policy import FamilyGaussianActor,policy_hash
    from src.gps.bidirectional.deployment_adjoint import head_tangent_geometry,trajectory_tangent
    from src.gps.bidirectional.ending_pilot import CountedEnding,EndingContext,EndingExecutor
    from src.gps.bidirectional.conditional_feedback import _mean_panel
    from src.gps.bidirectional.return_curriculum import panel_summary,comparison
    from src.gps.bidirectional.evidence import Archive
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source_req=None;prefix=bytearray();request_count=0
    with (root/'evidence/curriculum/ledger.jsonl').open('rb') as f:
        for line in f:
            prefix.extend(line);row=json.loads(line)
            if row['kind']=='adjoint_gradient_requested':
                request_count+=1
                if request_count==2:source_req=row;break
    assert source_req is not None;(out/'source_prefix.jsonl').write_bytes(prefix)
    base=root/'evidence/curriculum';records=[]
    for i,ref in enumerate(source_req['actual_own_refs']):
        p=base/ref['file'];assert sha(p)==ref['sha256'];refs[str(p)]=ref['sha256']
        with np.load(p,allow_pickle=False) as z:
            owner=json.loads(str(z['owner_json']));records.append(dict(states=z['states'].copy(),actions=z['actions'].copy(),query_ids=z['query_ids'].copy(),
                actor_hash=owner['actor_hash'],deterministic=True,continuation='uninterrupted_actor',ordinary=i<4))
    changed=root/'checkpoints/initial_guided_u00.pt';parent=root/'parent_inputs/checkpoints/round1_parent.pt'
    refs[str(changed)]=sha(changed);refs[str(parent)]=sha(parent);task=json.loads((root/'config.json').read_text())['task']
    archive=Archive(out/'evidence/qualification');archive.work_scope='changed_actor_qualification';work=DurableWork(out/'work_ledger.jsonl');plugin=WorkPlugin(work);nn=grads=0
    old_before=plugin._before
    def before(module,args):
        nonlocal nn
        if isinstance(module,FamilyGaussianActor):
            shape=np.shape(args[0]);n=int(np.prod(shape[:-1])) if len(shape)>1 else 1
            if nn+n>960:raise RuntimeError('NN cap before call')
            nn+=n
        return old_before(module,args)
    plugin._before=before;result=dict(passed=False,stage='setup')
    try:
        plugin.install();original=torch.autograd.grad
        def grad(*a,**kw):
            nonlocal grads
            if grads>=9:raise RuntimeError('Gradient cap before call')
            grads+=1;return original(*a,**kw)
        torch.autograd.grad=grad
        archive.append('qualification_actor_constructions_requested',actors=2,orthogonal_initializers_per_actor=3)
        actor=FamilyGaussianActor.from_checkpoint(torch.load(changed,map_location='cpu',weights_only=False)['learner']['policy'])
        baseline=FamilyGaussianActor.from_checkpoint(torch.load(parent,map_location='cpu',weights_only=False)['learner']['policy'])
        assert policy_hash(actor)==source_req['actor_hash']
        archive.append('changed_geometry_requested',actor_hash=policy_hash(actor),neural_rows=160,gradient_calls=9,analytic_jacobian_rows=80)
        geometry=head_tangent_geometry(actor,records,task,serial=True);pieces=[trajectory_tangent(actor,r,task) for r in records]
        parameters=(actor.fc_mean.weight,actor.fc_mean.bias)
        def flat(loss,retain):return np.concatenate([g.detach().cpu().numpy().ravel() for g in torch.autograd.grad(loss,parameters,retain_graph=retain)])
        g=flat(sum(v[0] for v in pieces[:4])/4,True);A=np.stack([flat(v[1],i<7) for i,v in enumerate(pieces)])
        assert np.allclose(g,geometry['cost_gradient'],atol=3e-6,rtol=3e-5)
        assert np.allclose(A,geometry['ending_gradients'],atol=3e-6,rtol=3e-5)
        np.savez_compressed(out/'changed_geometry.npz',**geometry,autograd_cost=g,autograd_ending=A)
        archive.append('changed_geometry_complete',max_cost_difference=float(abs(g-geometry['cost_gradient']).max()),max_ending_difference=float(abs(A-geometry['ending_gradients']).max()))
        dynamics=CountedEnding(archive,work,800,**task);executor=EndingExecutor(EndingContext(dynamics),archive)
        panels={}
        for name,policy in [('parent',baseline),('changed',actor)]:panels[name],_=_mean_panel(executor,policy,list(range(8100000,8100016)),'fresh_'+name)
        criterion=comparison(panels['parent'],panels['changed']);result.update(passed=criterion['passed'],stage='complete',criterion=criterion,
            summaries={n:panel_summary(v) for n,v in panels.items()},geometry_passed=True)
    except Exception as error:result.update(error_type=type(error).__name__,error=str(error))
    finally:plugin.uninstall();archive.close();work.close()
    result.update(accounting=archive.totals(),actor_rows=nn,gradient_calls=grads,optimizer_steps=0,wall_seconds=time.perf_counter()-start,
        input_sha256=refs,source_unchanged=all(sha(p)==refs[str(p)] for p in sources),work_ledger_sha256=sha(work.path),
        limits='Fresh16-root and derivative diagnostic only; not a qualified MPC or next-student result. No parameter setting was selected.')
    with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','summaries')},indent=2));raise SystemExit(0 if result['passed'] else 2)

if __name__=='__main__':main()
