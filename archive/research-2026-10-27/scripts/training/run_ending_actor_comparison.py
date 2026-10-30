#!/usr/bin/env python3
"""One frozen, finite comparison; new run directories only. No imitation."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ending.core import Task, IndependentMPPI, pure_actor_actions, capacity_witness_theta, COMPONENTS
from src.ending.learning import generate_candidates, propose_update, complete_returns, fit_value, compare_scores


def clean(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [clean(v) for v in x]
    return x


def digest(theta):
    return hashlib.sha256(np.asarray(theta, dtype='<f8').tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + '\n')


class BudgetStop(RuntimeError): pass


class Experiment:
    def __init__(self, cfg, run):
        self.cfg, self.run, self.task = cfg, run, Task(**cfg['task'])
        self.counter, self.paid, self.rows = 0, {}, []
        self.started = time.perf_counter()
        self.ledger = (run/'ledger.jsonl').open('x', buffering=1)
        self.records = (run/'records.jsonl').open('x', buffering=1)

    def reserve(self, scope, controls):
        # Every main control reserves one further independent audit control.
        trial = self.paid.copy()
        trial[scope] = trial.get(scope, 0) + controls
        for seed in self.cfg['seeds']:
            shared = trial.get(f'shared/seed{seed}', 0) + trial.get('shared/global', 0)
            for arm in self.cfg['arms']:
                if 2*(shared + trial.get(f'seed{seed}/{arm}', 0)) > self.cfg['budget']['controls_per_seed_arm']:
                    raise BudgetStop(f'budget including audit reserve: seed={seed}, arm={arm}')

    def record(self, kind, **kw):
        self.records.write(json.dumps(clean(dict(kind=kind, **kw)), sort_keys=True)+'\n')

    def save(self, scope, phase, kind, arrays=None, controls=0, **meta):
        self.reserve(scope, controls)
        ident = f'q{self.counter:07d}'
        self.counter += 1
        relative = None
        if arrays is not None:
            relative = f'queries/{ident}.npz'
            np.savez_compressed(self.run/relative, **arrays)
        row = dict(id=ident, scope=scope, phase=phase, kind=kind, controls=int(controls), file=relative, **meta)
        self.ledger.write(json.dumps(clean(row), sort_keys=True)+'\n')
        self.rows.append(row)
        self.paid[scope] = self.paid.get(scope, 0) + controls
        return ident

    def trace(self, scope, phase, theta, start, ancestry, length=None, **meta):
        state = np.array(start, float)
        remain = self.task.T-int(state[0])
        h = remain if length is None else min(length, remain)
        self.reserve(scope, h)
        states, acts, costs, comp, flags = [state.copy()], [], [], [], []
        begin = time.perf_counter()
        for _ in range(h):
            action = pure_actor_actions(theta, state, self.task)
            state, components, cost, flag = self.task.transition(state, action)
            states.append(state.copy()); acts.append(action); costs.append(cost)
            comp.append([components[n] for n in COMPONENTS])
            flags.append([flag[n] for n in ('ending_event','terminated','ending_good','good_lane')])
        arrays = dict(states=np.array(states), actions=np.array(acts).reshape(h,1), costs=np.array(costs),
                      components=np.array(comp).reshape(h,4), flags=np.array(flags).reshape(h,4), theta=np.array(theta))
        qid = self.save(scope, phase, 'actor', arrays, h, ancestry=ancestry, controller_hash=digest(theta),
                        remaining_start=remain, horizon=h, complete=h==remain,
                        terminated=h==remain, cutoff=h<remain, wall_seconds=time.perf_counter()-begin, **meta)
        return dict(id=qid, states=arrays['states'], costs=arrays['costs'], total=float(np.sum(costs)),
                    success=bool(state[0]==self.task.T and state[1]<=self.task.ending_y), endpoint=state, ancestry=ancestry)

    def planner_episode(self, seed, reset_seed, horizon, phase, name):
        scope = f'shared/seed{seed}'
        state = self.task.reset(reset_seed)
        pconf = self.cfg['planner'].copy(); pconf['H'] = horizon
        planner = IndependentMPPI(self.task, **pconf, seed=seed*100000+reset_seed)
        states, actions, costs, comp, flags, gaps, query_ids = [state.copy()], [], [], [], [], [], []
        for step in range(self.task.T):
            h = min(horizon, self.task.T-step)
            self.reserve(scope, (pconf['K']+1)*h+1+(h if step<self.task.gate else 0))
            begin = time.perf_counter(); result = planner.plan(state)
            pre = np.broadcast_to(state,(len(result['plans']),1,2))
            arr = dict(states=np.concatenate((pre,result['states']),axis=1), actions=result['plans'],
                       costs=result['stage_costs'], components=np.stack([result['stage_components'][n] for n in COMPONENTS],axis=-1),
                       flags=np.stack([result['flags'][n] for n in ('ending_event','terminated','ending_good','good_lane')],axis=-1),
                       nominal_pre=result['nominal_pre'], nominal_post=result['nominal_post'], weights=result['weights'],
                       totals=result['totals'], selected=result['selected'],
                       rng_pre=json.dumps(result['rng_pre'],sort_keys=True), rng_post=json.dumps(result['rng_post'],sort_keys=True))
            qid=self.save(scope,phase,'mppi',arr,(pconf['K']+1)*h,episode=name,step=step,
                          controller_version='independent-mppi-v1',planner=pconf,reset_seed=reset_seed,
                          wall_seconds=time.perf_counter()-begin, candidate_controls=pconf['K']*h,verification_controls=h)
            query_ids.append(qid)
            if step<self.task.gate:
                bound=self.task.local_lower_bound(state,h)
                self.save(scope,phase,'bound',dict(initial=state,h=h,**{k:bound[k] for k in ('bound','stage_bounds','max_y_by_step')}),h,episode=name)
                gaps.append(float((result['totals'][result['selected']]-bound['bound'])/h))
            action=result['action']; state,components,cost,flag=self.task.transition(state,action)
            states.append(state.copy()); actions.append(action); costs.append(cost)
            comp.append([components[n] for n in COMPONENTS]); flags.append([flag[n] for n in ('ending_event','terminated','ending_good','good_lane')])
            self.save(scope,phase,'mppi_execute',dict(states=np.array(states[-2:]), actions=action.reshape(1,1), costs=np.array([cost]),
                       components=np.array(comp[-1:]),flags=np.array(flags[-1:])),1,episode=name,planner_query=qid)
        episode=dict(name=name,seed=seed,reset_seed=reset_seed,horizon=horizon,states=np.array(states),
                     total=float(np.sum(costs)),success=bool(state[1]<=self.task.ending_y),
                     local=bool(states[self.task.gate][1]>=self.cfg['qualification']['local_y']),
                     gap=float(np.mean(gaps)),queries=query_ids)
        self.record('planner_episode',**episode)
        return episode

    def qualify(self):
        results=[]
        for seed in self.cfg['seeds']:
            short=[]; long=[]; witness=[]; reached=[]
            for index in range(self.cfg['qualification']['roots']):
                root=10000+seed*100+index
                short.append(self.planner_episode(seed,root,self.cfg['planner']['H'],'qualification',f'qual_short_{seed}_{index}'))
                long.append(self.planner_episode(seed,root,self.task.T,'qualification',f'qual_long_{seed}_{index}'))
                witness.append(self.trace(f'shared/seed{seed}','qualification_capacity',capacity_witness_theta(),self.task.reset(root),dict(reset_seed=root,role='qualification_capacity')))
                # Capacity/reference from an actually reached early state, not an invented reset.
                reached.append(self.trace(f'shared/seed{seed}','qualification_reached',capacity_witness_theta(),short[-1]['states'][3],dict(episode=short[-1]['name'],time=3,role='qualification_reached')))
            q=self.cfg['qualification']
            metrics=dict(seed=seed,local_rate=np.mean([r['local'] for r in short]),short_success=np.mean([r['success'] for r in short]),
                         reference_success=np.mean([r['success'] for r in long]),capacity_success=np.mean([r['success'] for r in witness]),
                         reached_success=np.mean([r['success'] for r in reached]),local_gap=np.mean([r['gap'] for r in short]),
                         short_cost=np.mean([r['total'] for r in short]),reference_cost=np.mean([r['total'] for r in long]))
            metrics['passed']=bool(metrics['local_rate']>=q['local_rate'] and metrics['short_success']<=q['short_success_max']
                and metrics['reference_success']>=q['reference_success_min'] and metrics['capacity_success']>=q['capacity_success_min']
                and metrics['reached_success']>=q['capacity_success_min'] and metrics['local_gap']<=q['max_gap_per_control']
                and metrics['short_cost']-metrics['reference_cost']>=q['cost_margin']
                and self.task.delay>=self.cfg['planner']['H'])
            self.record('qualification',**metrics); results.append(metrics)
            print(json.dumps(clean(dict(phase='qualification',**metrics))),flush=True)
        write_json(self.run/'qualification.json',results)
        return all(r['passed'] for r in results)

    def acquire(self,seed):
        pool={'train':[],'heldout':[]}
        for label,h in [('short',self.cfg['planner']['H']),('long',self.task.T)]:
            for i in range(self.cfg['acquisition']['episodes_per_horizon']):
                episode=self.planner_episode(seed,20000+100*seed+i,h,'acquisition',f'acq_{label}_{seed}_{i}')
                pool['train' if i<self.cfg['acquisition']['train_per_horizon'] else 'heldout'].append(episode)
        return pool

    def starts(self,seed,update,pool,role):
        offset={'train':30000,'validation':40000,'critic_train':50000,'critic_holdout':60000}[role]
        roots=[dict(state=self.task.reset(offset+seed*1000+update*10+i),ancestry=dict(reset_seed=offset+seed*1000+update*10+i,role=role)) for i in range(2)]
        if role.startswith('critic'):
            guide=3 if update%2 else 6
        else:
            guide=self.cfg['curriculum']['guide_times'][min(update//self.cfg['curriculum']['updates_per_stage'],2)]
        source=pool['train' if role in ('train','critic_train') else 'heldout']
        if guide:
            half=len(source)//2
            for j in range(2):
                episode=source[j*half+(update%half)]
                roots.append(dict(state=episode['states'][guide],ancestry=dict(episode=episode['name'],reset_seed=episode['reset_seed'],time=guide,role=role)))
        else:
            for i in range(2,4):
                root=offset+seed*1000+update*10+i
                roots.append(dict(state=self.task.reset(root),ancestry=dict(reset_seed=root,role=role)))
        return roots

    def measure(self,scope,phase,theta,starts,**kw):
        return [self.trace(scope,phase,theta,r['state'],r['ancestry'],**kw) for r in starts]

    def fit(self,scope,seed,update,base,pool,train_starts):
        rows=self.measure(scope,'critic_targets',base,train_starts+self.starts(seed,update,pool,'critic_train'),update=update)
        states=np.concatenate([r['states'][:-1] for r in rows]); targets=np.concatenate([complete_returns(r['costs']) for r in rows])
        groups=np.concatenate([[r['ancestry'].get('reset_seed',r['ancestry'].get('episode'))]*len(r['costs']) for r in rows]).astype(str)
        c=self.cfg['critic']
        begin=time.perf_counter()
        model=fit_value(states,targets,groups,total_steps=self.task.T,gate=self.task.gate,ending_y=self.task.ending_y,
                        target_actor_hash=digest(base),centers=c['centers'],width=c['width'],ridge=c['ridge'],support_radius=c['support_radius'])
        # Archive every model before qualification, ranking, or actor acceptance.
        relative=f'models/{scope.replace("/","_")}_{update:03d}.npz'
        np.savez_compressed(self.run/relative,**model.packet(),target_states=states,target_returns=targets,target_query_ids=np.array([r['id'] for r in rows]))
        self.save(scope,'critic_fit','fit',controls=0,update=update,model=relative,digest=model.digest,base_hash=digest(base),
                  target_queries=[r['id'] for r in rows],training_rows=len(states),fit_updates=1,wall_seconds=time.perf_counter()-begin)
        held=self.measure(scope,'critic_holdout',base,self.starts(seed,update,pool,'critic_holdout'),update=update)
        hs=np.concatenate([r['states'][:-1] for r in held]); ht=np.concatenate([complete_returns(r['costs']) for r in held]); pred=model.predict(hs)
        self.save(scope,'critic_holdout','predict',dict(states_to_predict=hs,targets=ht,**pred),0,update=update,prediction_rows=len(hs),model=relative)
        self.record('critic_holdout',scope=scope,update=update,mae=np.mean(abs(pred['values']-ht)),model=relative,queries=[r['id'] for r in held],base_hash=digest(base))
        return model,relative

    def score_prefix(self,scope,phase,theta,base,model,model_file,starts,update):
        scored=[]; refs=[]
        for context in starts:
            prefix=self.trace(scope,phase,theta,context['state'],context['ancestry'],length=self.cfg['critic']['prefix'],update=update)
            pred=model.predict(prefix['endpoint'][None]); value=float(pred['values'][0]); tail=None
            self.save(scope,phase,'predict',dict(states_to_predict=prefix['endpoint'][None],**pred),0,update=update,prediction_rows=1,model=model_file)
            if not pred['supported'][0]:
                tail=self.trace(scope,phase+'_fallback_base',base,prefix['endpoint'],dict(parent_query=prefix['id'],role='unsupported_exact_base_tail'),update=update)
                value=tail['total']
            scored.append(prefix['total']+value); refs.append(dict(prefix=prefix['id'],fallback=None if tail is None else tail['id'],predicted=float(pred['values'][0]),used_tail=value,supported=bool(pred['supported'][0])))
        return scored,refs

    def evaluate(self,scope,seed,theta,update,final=False):
        n=self.cfg['evaluation']['final_roots' if final else 'development_roots']
        offset=900000 if final else 800000
        traces=[self.trace(scope,'final_evaluation' if final else 'evaluation',theta,self.task.reset(offset+seed*100+i),dict(reset_seed=offset+seed*100+i,role='final' if final else 'development'),update=update) for i in range(n)]
        row=dict(scope=scope,seed=seed,update=update,final=final,parameters=theta,actor_hash=digest(theta),
                 success=np.mean([r['success'] for r in traces]),cost=np.mean([r['total'] for r in traces]),
                 individual_success=[r['success'] for r in traces],individual_cost=[r['total'] for r in traces],
                 main_controls=self.paid.get(scope,0)+self.paid.get(f'shared/seed{seed}',0)+self.paid.get('shared/global',0),queries=[r['id'] for r in traces])
        row['charged_controls_with_audit_reserve']=2*row['main_controls']
        self.record('evaluation',**row)
        return row

    def diagnostic(self,scope,seed,update,base,candidates,eps,scores,model,model_file):
        diff=np.abs(np.mean(scores,axis=1)[0::2]-np.mean(scores,axis=1)[1::2]); diff[0]=-1
        pairs=[0,int(np.argmax(diff))]
        indices=np.array([[2*p,2*p+1] for p in pairs]).ravel()
        starts=[dict(state=self.task.reset(700000+seed*1000+update*10+i),ancestry=dict(reset_seed=700000+seed*1000+update*10+i,role='diagnostic_only')) for i in range(2)]
        A=[];B=[];C=[];ref=[];a_queries=[];b_queries=[]
        for ix in indices:
            cs,cref=self.score_prefix(scope,'diagnostic_prediction',candidates[ix],base,model,model_file,starts,update)
            # C and its model already archived before oracle A/B is queried.
            aa=self.measure(scope,'diagnostic_A_candidate',candidates[ix],starts,update=update)
            bb=[];bt=[]
            for context,r in zip(starts,cref):
                with np.load(self.run/f'queries/{r["prefix"]}.npz') as packet:
                    endpoint=packet['states'][-1]; prefixcost=float(packet['costs'].sum())
                tail=self.trace(scope,'diagnostic_B_exact_base',base,endpoint,dict(parent_query=r['prefix'],role='diagnostic_only'),update=update)
                bb.append(prefixcost+tail['total'])
                bt.append(tail['id'])
            A.append([r['total'] for r in aa]);B.append(bb);C.append(cs);ref.append(cref)
            a_queries.append([r['id'] for r in aa]);b_queries.append(bt)
        self.record('diagnostic',scope=scope,update=update,indices=indices,base_hash=digest(base),model=model_file,
                    parameters=candidates[indices],prefix_references=ref,A_queries=a_queries,B_tail_queries=b_queries,
                    **compare_scores(A,B,C,eps=eps[pairs],sigma=self.cfg['search']['sigma'],cost_scale=self.cfg['search']['cost_scale']))

    def train(self,seed,arm,pool):
        scope=f'seed{seed}/{arm}'; theta=np.array(self.cfg['actor']['initial_parameters'],float)
        self.evaluate(scope,seed,theta,0)
        final_update=0
        for update in range(self.cfg['search']['updates']):
            # Require a conservative whole-update reserve before spending anything.
            self.reserve(scope,16000)
            base=theta.copy(); train=self.starts(seed,update,pool,'train'); validation=self.starts(seed,update,pool,'validation')
            rngseed=1000000+seed*1000+update
            sc=self.cfg['search']; sample=generate_candidates(base,rngseed,pairs=sc['pairs'],sigma=sc['sigma'])
            candidates=sample['candidates']; eps=sample['eps']
            baseline=self.measure(scope,'accept_base_train',base,train,update=update)
            baseval=self.measure(scope,'accept_base_validation',base,validation,update=update)
            refs=[];model=None;modelfile=None
            if arm=='full_return':
                measurements=[self.measure(scope,'candidate_complete',candidate,train,update=update,candidate=i) for i,candidate in enumerate(candidates)]
                scores=np.array([[r['total'] for r in rows] for rows in measurements]);refs=[[r['id'] for r in rows] for rows in measurements]
            else:
                model,modelfile=self.fit(scope,seed,update,base,pool,train)
                scored=[self.score_prefix(scope,'candidate_prefix',candidate,base,model,modelfile,train,update) for candidate in candidates]
                scores=np.array([r[0] for r in scored]);refs=[r[1] for r in scored]
            proposal=propose_update(base,eps,scores,sc['sigma'],lr=sc['lr'],maxnorm=sc['maxnorm'],cost_scale=sc['cost_scale'])
            eligible=np.linalg.norm(candidates-base,axis=1)<=sc['maxnorm']
            best=int(np.argmin(np.where(eligible,scores.mean(axis=1),np.inf))) if eligible.any() else None
            choices=[('bounded_ES',proposal['parameters'])]+([] if best is None else [('best_eligible_perturbation',candidates[best])])
            checked=[]
            for label,parameters in choices:
                # Both arms apply the same actual full-policy acceptance rule.
                if arm=='full_return' and label=='best_eligible_perturbation':
                    tr=measurements[best]
                else: tr=self.measure(scope,'accept_candidate_train',parameters,train,update=update,proposal=label)
                va=self.measure(scope,'accept_candidate_validation',parameters,validation,update=update,proposal=label)
                tc=np.mean([r['total'] for r in tr]); vc=np.mean([r['total'] for r in va]); vs=np.mean([r['success'] for r in va])
                ok=bool(tc<np.mean([r['total'] for r in baseline])-self.cfg['acceptance']['cost_margin']
                    and vc<np.mean([r['total'] for r in baseval])-self.cfg['acceptance']['cost_margin']
                    and np.mean([r['success'] for r in tr])>=np.mean([r['success'] for r in baseline])
                    and vs>=np.mean([r['success'] for r in baseval]))
                checked.append(dict(label=label,parameters=parameters,train_cost=tc,validation_cost=vc,validation_success=vs,accepted_eligible=ok,
                                    train_queries=[r['id'] for r in tr],validation_queries=[r['id'] for r in va]))
            accepted=[r for r in checked if r['accepted_eligible']]
            winner=min(accepted,key=lambda r:r['validation_cost']) if accepted else None
            # Diagnostic data are obtained after proposals/acceptance decisions and never fed back.
            if model is not None and update%self.cfg['diagnostic']['every']==0:
                self.diagnostic(scope,seed,update,base,candidates,eps,scores,model,modelfile)
                # Evaluate the actual proposed changes on held-out original roots.
                # Their A values are already measured acceptance data; B/C are
                # post-decision diagnostics, never additional acceptance inputs.
                for check in checked:
                    ps,pr=self.score_prefix(scope,'proposal_diagnostic',check['parameters'],base,model,modelfile,validation[:2],update)
                    bvalues=[];tailids=[]
                    for ref in pr:
                        with np.load(self.run/f'queries/{ref["prefix"]}.npz') as p:
                            endpoint=p['states'][-1];pcost=float(p['costs'].sum())
                        tail=self.trace(scope,'proposal_diagnostic_B',base,endpoint,dict(parent_query=ref['prefix'],role='diagnostic_only'),update=update)
                        bvalues.append(pcost+tail['total']);tailids.append(tail['id'])
                    avals=[]
                    for query in check['validation_queries'][:2]:
                        with np.load(self.run/f'queries/{query}.npz') as p:avals.append(float(p['costs'].sum()))
                    self.record('proposal_diagnostic',scope=scope,update=update,label=check['label'],base_hash=digest(base),
                                parameters=check['parameters'],A=avals,B=bvalues,C=ps,prefix_references=pr,
                                A_queries=check['validation_queries'][:2],B_tail_queries=tailids,
                                base_A_queries=[r['id'] for r in baseval[:2]],decision_already_fixed=True)
            theta=np.array(winner['parameters'] if winner else base)
            self.record('update',scope=scope,seed=seed,update=update,base=base,base_hash=digest(base),candidates=candidates,eps=eps,
                        direction_seed=rngseed,scores=scores,score_queries=refs,proposal=proposal,checked=checked,
                        winner=None if winner is None else winner['label'],parameters=theta,actor_hash=digest(theta),
                        score_range=float(np.ptp(scores.mean(axis=1))),paired_signal=float(np.mean(abs(scores.mean(axis=1)[0::2]-scores.mean(axis=1)[1::2]))),
                        training_starts=train,validation_starts=validation,base_train_queries=[r['id'] for r in baseline],base_validation_queries=[r['id'] for r in baseval])
            final_update=update+1
            if final_update%self.cfg['evaluation']['every']==0:
                ev=self.evaluate(scope,seed,theta,final_update)
                print(json.dumps(clean(dict(phase='training',scope=scope,update=final_update,success=ev['success'],cost=ev['cost'],main_controls=ev['main_controls']))),flush=True)
        return self.evaluate(scope,seed,theta,final_update,final=True)


def freeze(run,config):
    paths=[config,Path(__file__).resolve(),ROOT/'scripts/diagnostics/audit_ending_actor.py',ROOT/'scripts/diagnostics/report_ending_actor.py',
           ROOT/'docs/ending_actor_protocol.md',ROOT/'docs/ending_actor_prior_art.md']
    paths+=sorted((ROOT/'src/ending').glob('*.py'))+sorted((ROOT/'tests').glob('test_ending_*.py'))
    paths+=sorted((ROOT/'artifacts/ending_actor_development').glob('*.json'))
    files=[]
    for path in paths:
        relative=path.relative_to(ROOT);archive=Path('source')/relative
        (run/archive).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,run/archive)
        files.append(dict(path=str(path),archive=str(archive),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(run/'manifest.json',dict(files=files,python=sys.version,
               dependencies={name:importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib','pytest')},
               freeze_time_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'configs/ending_actor_comparison.json');parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());run=args.run_dir.resolve()
    run.mkdir(parents=True,exist_ok=False);(run/'queries').mkdir();(run/'models').mkdir()
    write_json(run/'config.json',cfg);freeze(run,args.config.resolve())
    exp=Experiment(cfg,run)
    exp.save('shared/global','development','development',controls=cfg['development_controls'],raw='development/',
             failed_control_queries=1,failed_query_controls=0,
             description='211 simulator + 7 analytic controls in core tests; one terminal-plan rejection before physics; learning and driver tests zero')
    development=run/'development';development.mkdir()
    raw_development=ROOT/'artifacts/ending_actor_development'
    unique_raw=sorted(raw_development.glob('core_*.json'))
    for index,path in enumerate(unique_raw):
        shutil.copy2(path,development/f'core_{index}.json')
    results=[];status='running'
    try:
        qualified=exp.qualify()
        if not qualified: status='qualification_failed'
        else:
            # Finish common acquisition before either arm trains.
            pools={seed:exp.acquire(seed) for seed in cfg['seeds']}
            for seed in cfg['seeds']:
                for arm in cfg['arms']: results.append(exp.train(seed,arm,pools[seed]))
            status='completed'
    except BudgetStop as error:
        status='budget_stopped';exp.record('stop',reason=str(error));print(str(error),flush=True)
    except BaseException as error:
        status='error';exp.record('stop',reason=repr(error));raise
    finally:
        write_json(run/'summary.json',dict(status=status,final=results,main_controls=sum(exp.paid.values()),scope_main_controls=exp.paid,
                   main_wall_seconds=time.perf_counter()-exp.started,query_count=exp.counter,requires_independent_audit=True))
        exp.ledger.close();exp.records.close()


if __name__=='__main__':main()
