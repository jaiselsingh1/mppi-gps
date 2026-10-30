"""Independent numeric/actual-outcome audit adapter; no producers or Torch."""
import json,math
import numpy as np
from .ending_audit import require,same,packed
from .ending_completion_audit import numeric_digest

FRACTIONS=(1.,.5,.25,.125)

def barrier(slack):
    return -.001*math.log(slack) if slack>=.01 else .001*(.5*((slack-.02)/.01)**2-.5-math.log(.01))

def euclidean_replay(g,A,b):
    d=-.03*g/max(np.linalg.norm(g),1e-12);norms=np.einsum('ij,ij->i',A,A)
    for _ in range(64):
        for row,rhs,norm in zip(A,b,norms):
            excess=float(row@d-rhs)
            if excess>0 and norm>1e-20:d-=excess/norm*row
    valid=bool(max(A@d-b)<=1e-7 and g@d< -1e-9 and np.linalg.norm(d)<=.03+1e-7)
    return d if valid else np.zeros_like(d),valid

def metric_replay(g,A,b,J):
    F=J.T@J/len(J)+1e-5*np.eye(len(g));v=np.linalg.solve(F,np.column_stack((g,A.T)))
    z=v[:,0];d=-.05*z/max(float(np.sqrt(z@F@z)),1e-12)
    for _ in range(64):
        for row,inverse,rhs in zip(A,v[:,1:].T,b):
            excess=float(row@d-rhs);denom=float(row@inverse)
            if excess>0 and denom>1e-20:d-=excess/denom*inverse
        norm=float(np.sqrt(max(0.,d@F@d)))
        if norm>.05:d*=.05/norm
    valid=bool(np.sqrt(max(0.,d@F@d))<=.05+1e-7 and max(A@d-b)<=1e-7 and g@d< -1e-9)
    return d if valid else np.zeros_like(d),F,valid

def head_geometry(raw,records,policy,task):
    """Recompute analytic derivatives from SAVED activations, no NN forward."""
    W1=policy['actor']['fc1.weight'].astype(float);W2=policy['actor']['fc2.weight'].astype(float);w=policy['actor']['fc_mean.weight'][0].astype(float)
    h1=raw['h1'].astype(float);h2=raw['h2'].astype(float);require(h1.shape==h2.shape==(80,128),'Missing captured hidden states')
    offset=0;costs=[];ends=[];J=[];observations=[];actions=[]
    for i,r in enumerate(records):
        s=r['states'];a=r['actions'];t0=int(s[0,0]);state=np.zeros(129);cost=np.zeros(129)
        for j,t in enumerate(range(t0,12)):
            u=float(a[j,0]);factor=1-u*u
            obs=np.zeros(8);obs[0]=s[j,1];obs[1]=t/25;obs[2]=1-t/25;observations.append(obs);actions.append(u)
            hidden1=(1-h1[offset]**2)*W1[:,0];hidden2=(1-h2[offset]**2)*(W2@hidden1)
            du=factor*np.r_[h2[offset],1.]+factor*float(w@hidden2)*state;J.append(du)
            state=float(-1.<s[j,1]+task['move']*u<1.)*(state+task['move']*du)
            cost+=2*task['effort']*u*du-.5*task['greed']*state;offset+=1
        if i<4:costs.append(cost)
        ends.append(state)
    same(raw['observations'],observations,'Geometry feature rows do not match actual pre-action states')
    same(raw['mean_actions'],actions,'Geometry anchors were not actor-owned controls')
    return np.mean(costs,axis=0),np.stack(ends),np.stack(J)

def actual_acceptance(base,proposed,base_raws,proposed_raws):
    require(len(base)==len(proposed)==8 and [r['context_id'] for r in base]==[r['context_id'] for r in proposed],'Unpaired training test')
    gain=np.mean([a['cost']-b['cost'] for a,b in zip(base[:4],proposed[:4])])
    bv=max([0.]+[float(r['states'][-1,1])+.101 for r in base_raws[4:]])
    pv=max([0.]+[float(r['states'][-1,1])+.101 for r in proposed_raws[4:]])
    return bool(gain>=1e-6 and all(r['capture'] for r in proposed[:4]) and
        sum(r['capture'] for r in proposed[:4])>=sum(r['capture'] for r in base[:4]) and pv<=bv+1e-6),bv,pv

def audit_steps(e,base,c,r,rows,episodes,outcomes,checkpoints,counts):
    partial=r['stage']=='execution_error'
    serial=c['schema']=='bidirectional-deployment-action-metric-dev3-scalar-layout'
    metric=c['schema'] in ('bidirectional-deployment-action-metric-dev2','bidirectional-deployment-action-metric-dev3-scalar-layout');projected=c['schema']=='bidirectional-deployment-projected-adjoint-dev1'
    mode='metric_serial' if serial else 'metric' if metric else 'projected' if projected else 'barrier'
    requests={z['id']:z for z in rows if z['kind']=='adjoint_step_requested'}
    gradients={z['step_request_id']:z for z in rows if z['kind']=='adjoint_gradient_requested'}
    done={z['request_id']:z for z in rows if z['kind']=='adjoint_gradient_complete'}
    complete={z['request_id']:z for z in rows if z['kind']=='adjoint_step_complete'}
    if not partial:require(len(requests)==len(complete)==192,'Scheduled matched adjoint attempts missing')
    else:
        require(set(complete)<=set(requests),'Completion without a requested step')
        counts['interrupted_adjoint_step_requests']=len(requests)-len(complete)
    common=checkpoints['deployment_source_parent']['learner']
    source=e.file('parent_inputs/checkpoints/round1_parent.pt',c['parent_checkpoint_sha256'])
    from .ending_audit import checkpoint_numpy
    original=checkpoint_numpy(source)['learner'];require(common['policy']['policy_hash']==c['parent_actor_hash'],'Different initial actor')
    require(numeric_digest(common['policy'])==numeric_digest(original['policy']) and not common['optimizer']['state'] and common['updates']==0,
        'Common reset changed the actor or retained hidden Adam momentum')
    require(numeric_digest(common['numpy_rng'])==numeric_digest(original['numpy_rng']) and numeric_digest(common['torch_rng'])==numeric_digest(original['torch_rng']),
        'Initializer private streams changed')
    group_counts={};accepted=0
    for identifier,req in requests.items():
        if identifier not in complete:continue
        arm=req['arm'];step=req['step'];stem=f'{arm}_s{step:02d}';group_counts[arm]=group_counts.get(arm,0)+1
        prior_label=f'{arm}_u{step-1:02d}' if step else 'deployment_source_parent' if arm=='initial_guided' else arm.rsplit('_',1)[0]+'_parent'
        prior=checkpoints[prior_label]['learner'];owner=prior['policy']['policy_hash'];finish=complete[identifier]
        require(req['learner_digest']==numeric_digest(prior) and req['actor_hash']==owner and req['teacher_action_targets'] is False,'Gradient owner or common parent changed')
        require(req['ordinary']==[True]*4+[False]*4 and len(req['contexts'])==8,'Changed objective/context weighting')
        base_names=[stem+f'_base_c{i}' for i in range(8)];base_raws=[episodes[n] for n in base_names];base_outcomes=[outcomes[n] for n in base_names]
        require(finish['base_outcomes']==base_outcomes,'Base outcomes not actually executed')
        for i,(raw,context) in enumerate(zip(base_raws,req['contexts'])):
            same(raw['states'][0],context['state'],'Actual training start differs from declared context')
            require(packed(raw,'owner_json')['actor_hash']==owner and outcomes[base_names[i]]['deterministic'],'Wrong base continuation')
            if i<4:
                seed=int(context['context_id'].split(':')[1]);rng=np.random.Generator(np.random.PCG64(seed));same(context['state'],[0.,rng.uniform(.2,.4)],'Ordinary reset changed')
            else:
                context_source=req['source_contexts'][i-4];source=context_source['source']
                if 'raw' in source:observed=e.npz(base,source['raw'])['states']
                else:
                    path=e.file(source['file'],source['sha256'])
                    with np.load(path,allow_pickle=False) as z:observed=z['states'].copy()
                expected=observed[4].copy();delta=.025 if (i-4)%2 else -.025;expected[1]=np.clip(expected[1]+delta,-1,1)
                same(context['state'],expected,'Acquired state was fabricated or action-labeled')
                require(context_source['perturbation']==delta and context_source['state_index']==4,'Changed acquisition reset')
        gr=gradients[identifier];gd=done[gr['id']];raw=e.npz(base,gd['raw'])
        require(gr['actor_hash']==str(raw['actor_hash'])==owner and gr['active_jacobian_rows']==80 and gr.get('learning_form','barrier')==mode,'Tangent owner/derivative contract differs')
        require(len(gr['actual_own_refs'])==8,'Missing own gradient evidence')
        for ref,n in zip(gr['actual_own_refs'],base_names):
            saved=e.npz(base,ref);same(saved['states'],episodes[n]['states'],'Tangent evaluated a hypothetical or foreign trajectory')
            same(saved['actions'],episodes[n]['actions'],'Tangent uses another actor action')
            require(np.array_equal(saved['query_ids'],episodes[n]['query_ids']) and packed(saved,'owner_json')['actor_hash']==owner,'Foreign/reused target evidence')
        # dev2/dev3 inherited the barrier template's backward_calls=1 field.
        # Global before-call hooks independently establish that the metric
        # implementation ran analytic derivatives and no backward here. Keep
        # this metadata defect visible; never charge the template as real work.
        require(gr.get('grad_calls',0)==(9 if projected else 0) and (gr['backward_calls'] in (0,1) if metric else gr['backward_calls']==(0 if projected else 1)),'Gradient work declaration differs')
        if metric and gr['backward_calls']==1:counts['known_incorrect_backward_intent_fields']+=1
        if metric or projected:
            bounds=[]
            for i,z in enumerate(base_raws):
                slack=-.101-float(z['states'][-1,1]);bounds.append(max(slack,0.) if i<4 else (slack if slack>=0 else -min(-slack,.01)))
            same(raw['upper_bounds'],bounds,'Ending constraint targets changed')
            if metric:
                g,A,J=head_geometry(raw,base_raws,prior['policy'],c['task'])
                same(raw['cost_gradient'],g,'Analytic ordinary head-cost derivative differs',atol=2e-10)
                same(raw['ending_gradients'],A,'Analytic ending derivatives differ',atol=2e-10)
                same(raw['control_jacobian'],J,'Closed-loop action metric Jacobian differs',atol=2e-10)
                direction,F,valid=metric_replay(g,A,np.asarray(bounds),J);same(raw['metric'],F,'Metric changed',atol=2e-10)
                counts['analytic_policy_jacobian_rows_rechecked']+=80;counts['metric_solves_rechecked']+=1
            else:
                g=raw['cost_gradient'];A=raw['ending_gradients'];direction,valid=euclidean_replay(g,A,np.asarray(bounds))
            same(raw['direction'],direction,'Constrained update direction/invalid fallback differs',atol=2e-10)
            require(gd['details'][0]['valid']==valid,'Invalid numerical direction labeled usable')
            direction=raw['direction']
        else:
            values=[]
            for i,z in enumerate(base_raws):
                running=float(z['cost_components'][:,:2].sum());b=barrier(-.101-float(z['states'][-1,1]))
                values.append((running if i<4 else 0.)+b*(1. if i<4 else .25))
            same(gd['surrogate_loss'],sum(values)/4,'Constraint surrogate was relabeled task return',atol=8e-6)
        trials=finish['trials'];require(1<=len(trials)<=4,'Extra or missing fraction trials')
        selected=None
        full=checkpoints[stem+'_f0']['learner']
        if metric or projected:
            expected={name:value.copy() for name,value in prior['policy']['actor'].items()}
            if metric:
                expected['fc_mean.weight']+=direction[:-1].astype(np.float32).reshape(1,128);expected['fc_mean.bias']+=direction[-1:].astype(np.float32)
            else:
                offset=0
                for name in ('fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc_mean.weight','fc_mean.bias'):
                    shape=expected[name].shape;n=expected[name].size;expected[name]+=direction[offset:offset+n].astype(np.float32).reshape(shape);offset+=n
            for name,value in expected.items():same(full['policy']['actor'][name],value,'Actual full parameter proposal differs from measured direction',atol=2e-7)
        for j,trial in enumerate(trials):
            cp=checkpoints[stem+f'_f{j}']['learner'];fraction=FRACTIONS[j]
            require(trial['fraction']==fraction and trial['actor_hash']==cp['policy']['policy_hash'],'Changed fraction schedule/actor')
            for name,value in prior['policy']['actor'].items():
                expected=value+np.float32(fraction)*(full['policy']['actor'][name]-value) if value.dtype.kind=='f' else value
                same(cp['policy']['actor'][name],expected,'Backtracking trial was not its declared interpolation',atol=2e-7)
            names=[stem+f'_f{j}_c{i}' for i in range(8)];proposed=[outcomes[n] for n in names];raws=[episodes[n] for n in names]
            require(trial['outcomes']==proposed and all(v['actor_hash']==trial['actor_hash'] for v in proposed),'Changed-actor outcome ownership differs')
            allowed,bv,pv=actual_acceptance(base_outcomes,proposed,base_raws,raws)
            same(trial['acquired_violation'],pv,'Acquired feasibility measurement differs');same(finish['base_violation'],bv,'Base feasibility differs')
            require(trial['admissible']==allowed,'Loss/prediction substituted for actual acceptance')
            if allowed:
                require(j==len(trials)-1,'Continued testing after accepted fraction');selected=j
        require(finish['accepted']==(selected is not None) and finish['fraction']==(None if selected is None else FRACTIONS[selected]),'Update acceptance relabeled')
        final=checkpoints[f'{arm}_u{step:02d}']['learner'];expected=prior if selected is None else checkpoints[stem+f'_f{selected}']['learner']
        require(numeric_digest(final)==numeric_digest(expected),'Rollback/accepted optimizer and RNG state differs')
        accepted+=int(selected is not None);counts['adjoint_update_attempts']+=1;counts['analytic_dynamics_jacobian_rows_rechecked']+=80
    if not partial:require(group_counts==dict(initial_guided=64,round0_raw=32,round0_guided=32,round1_raw=32,round1_guided=32),'Unequal/skipped matched schedules')
    if metric or projected:
        requests=[z for z in rows if z['kind']=='projected_parameter_update_requested'];complete=[z for z in rows if z['kind']=='projected_parameter_update_complete']
        expected=counts['adjoint_update_attempts']
        require(len(requests)==len(complete)==expected and {z['id'] for z in requests}=={z['request_id'] for z in complete},'Unmetered manual optimization')
        counts['manual_parameter_updates']=expected;counts['projection_halfspace_checks']=expected*512
    counts['adjoint_updates_accepted']=accepted
