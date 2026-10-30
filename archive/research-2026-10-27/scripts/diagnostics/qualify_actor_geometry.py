"""Bounded80-row neural replay of measured mean-head trajectory geometry."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();root=args.run.resolve();out=args.out.resolve();out.mkdir(exist_ok=False);started=time.perf_counter()
    from scripts.diagnostics.run_bidirectional_tests import DurableWork,WorkPlugin
    from src.gps.bidirectional.policy import FamilyGaussianActor,policy_hash
    from src.gps.bidirectional.deployment_adjoint import head_tangent_geometry,metric_feasible_descent
    import torch
    torch.set_num_threads(1)
    paths=[Path(__file__).resolve(),Path('src/gps/bidirectional/deployment_adjoint.py').resolve(),Path('src/gps/bidirectional/policy.py').resolve()]
    refs={str(p):sha(p) for p in paths}
    for i,p in enumerate(paths):(out/f'source_{i}_{p.name}').write_bytes(p.read_bytes())
    prefix=bytearray();request=None;complete=None
    with (root/'evidence/curriculum/ledger.jsonl').open('rb') as f:
        for line in f:
            prefix.extend(line);row=json.loads(line)
            if row['kind']=='adjoint_gradient_requested':request=row
            if row['kind']=='adjoint_gradient_complete':complete=row;break
    assert request is not None and complete['request_id']==request['id'];(out/'source_prefix.jsonl').write_bytes(prefix)
    base=root/'evidence/curriculum';cp=root/'parent_inputs/checkpoints/round1_parent.pt'
    refs[str(cp)]=sha(cp);records=[]
    for i,ref in enumerate(request['actual_own_refs']):
        path=base/ref['file'];assert sha(path)==ref['sha256'];refs[str(path)]=ref['sha256']
        with np.load(path,allow_pickle=False) as z:
            owner=json.loads(str(z['owner_json']));records.append(dict(states=z['states'].copy(),actions=z['actions'].copy(),query_ids=z['query_ids'].copy(),
                actor_hash=owner['actor_hash'],deterministic=True,continuation='uninterrupted_actor',ordinary=i<4,raw=ref))
    path=base/complete['raw']['file'];assert sha(path)==complete['raw']['sha256'];refs[str(path)]=complete['raw']['sha256']
    with np.load(path,allow_pickle=False) as z:oldg=z['cost_gradient'].copy();oldA=z['ending_gradients'].copy();bounds=z['upper_bounds'].copy()
    task=json.loads((root/'config.json').read_text())['task'];work=DurableWork(out/'work_ledger.jsonl');plugin=WorkPlugin(work);nn=0
    original=plugin._before
    def before(module,values):
        nonlocal nn
        if isinstance(module,FamilyGaussianActor):
            shape=np.shape(values[0]);n=int(np.prod(shape[:-1])) if len(shape)>1 else 1
            if nn+n>80:raise RuntimeError('Neural replay cap before forward')
            nn+=n
        return original(module,values)
    plugin._before=before;result=dict(passed=False,new_controls=0,new_actor_rows=None,new_optimizer_steps=0)
    try:
        plugin.install();work.append('replay_actor_construction_requested',actor_constructions=1,orthogonal_initializers_per_actor=3,checkpoint_sha256=refs[str(cp)])
        saved=torch.load(cp,map_location='cpu',weights_only=False)['learner']['policy'];actor=FamilyGaussianActor.from_checkpoint(saved)
        assert policy_hash(actor)==request['actor_hash']
        work.append('analytic_geometry_requested',neural_rows=80,analytic_policy_jacobian_rows=80,analytic_dynamics_jacobian_rows=80)
        geometry=head_tangent_geometry(actor,records,task)
        assert np.allclose(geometry['cost_gradient'],oldg[-129:],atol=3e-6,rtol=3e-5)
        assert np.allclose(geometry['ending_gradients'],oldA[:,-129:],atol=3e-6,rtol=3e-5)
        direction,F,info=metric_feasible_descent(geometry['cost_gradient'],geometry['ending_gradients'],bounds,geometry['control_jacobian'])
        np.savez_compressed(out/'geometry.npz',**geometry,metric=F,direction=direction,upper_bounds=bounds)
        result.update(passed=True,projection=info,cost_gradient_max_difference=float(abs(geometry['cost_gradient']-oldg[-129:]).max()),
            ending_gradient_max_difference=float(abs(geometry['ending_gradients']-oldA[:,-129:]).max()))
        work.append('analytic_geometry_complete',analytic_policy_jacobian_rows=80,analytic_dynamics_jacobian_rows=80,linear_solves=1,halfspace_checks=512)
    except Exception as error:result.update(error_type=type(error).__name__,error=str(error))
    finally:plugin.uninstall();work.close()
    result.update(new_actor_rows=nn,wall_seconds=time.perf_counter()-started,input_sha256=refs,source_unchanged=all(sha(p)==refs[str(p)] for p in paths),
        work_ledger_sha256=sha(work.path),source_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        limitation='80paid NN replay rows; numeric gradient/metric diagnostic only. No changed actor was executed and descent predictions are not measured gains.')
    with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2));raise SystemExit(0 if result['passed'] else 2)

if __name__=='__main__':main()
