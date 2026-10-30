#!/usr/bin/env python3
"""Read-only paid-cost learning curves and a candid finite-study verdict."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def paired_interval(values, repetitions, seed=9817):
    x=np.array(values,float);rng=np.random.default_rng(seed)
    means=x[rng.integers(len(x),size=(repetitions,len(x)))].mean(axis=1)
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def report(run):
    start=time.perf_counter();cfg=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
    audit=json.loads((run/'audit.json').read_text()) if (run/'audit.json').exists() else {}
    records=[json.loads(l) for l in (run/'records.jsonl').read_text().splitlines() if l]
    quals=json.loads((run/'qualification.json').read_text()) if (run/'qualification.json').exists() else []
    evaluations=[r for r in records if r['kind']=='evaluation'];updates=[r for r in records if r['kind']=='update'];diag=[r for r in records if r['kind']=='diagnostic']
    def paid(r):return r['charged_controls_with_audit_reserve']
    result=dict(status=summary['status'],main_controls=summary['main_controls'],audit=audit,
                main_wall_seconds=summary['main_wall_seconds'],qualification=quals,
                fit_calls=sum(r['kind']=='fit' for r in [json.loads(l) for l in (run/'ledger.jsonl').read_text().splitlines()]),
                models=len(list((run/'models').glob('*.npz'))),diagnostic_panels=len(diag))
    result['raw_model_count_matches_all_fits']=result['models']==result['fit_calls']
    finals=[r for r in evaluations if r['final']]
    result['final']=[]
    for arm in cfg['arms']:
        rows=[r for r in finals if r['scope'].endswith('/'+arm)]
        passed=bool(len(rows)==len(cfg['seeds']) and all(r['success']>=cfg['evaluation']['success_rate'] and
                      next(q['short_cost'] for q in quals if q['seed']==r['seed'])-r['cost']>=cfg['evaluation']['cost_margin_over_short'] for r in rows))
        result['final'].append(dict(arm=arm,ending_learned=passed,success=[r['success'] for r in rows],cost=[r['cost'] for r in rows],
                                    paid_controls=[paid(r) for r in rows],accepted=sum(r['winner'] is not None for r in updates if r['scope'].endswith('/'+arm))))
    dev=[r for r in evaluations if not r['final']]
    scopes=[f'seed{s}/{a}' for s in cfg['seeds'] for a in cfg['arms']]
    panel={scope:sorted([r for r in dev if r['scope']==scope],key=paid) for scope in scopes}
    if all(panel.values()):
        common=min(max(paid(r) for r in rows) for rows in panel.values())
        feasible=all(any(paid(r)<=common for r in rows) for rows in panel.values())
        if feasible:
            selected={scope:max([r for r in rows if paid(r)<=common],key=paid) for scope,rows in panel.items()}
            costdif=[selected[f'seed{s}/critic_prefix']['cost']-selected[f'seed{s}/full_return']['cost'] for s in cfg['seeds']]
            successdif=[selected[f'seed{s}/critic_prefix']['success']-selected[f'seed{s}/full_return']['success'] for s in cfg['seeds']]
            ci=paired_interval(costdif,cfg['inference']['paired_seed_bootstrap_replicates']);si=paired_interval(successdif,cfg['inference']['paired_seed_bootstrap_replicates'])
            helps=bool(np.mean(costdif)<=-cfg['inference']['critic_help_cost_margin'] and np.mean(successdif)>=cfg['inference']['critic_help_success_margin'] and ci[1]<0 and si[0]>0)
            result['common_budget']=dict(controls=common,selected={k:dict(update=v['update'],controls=paid(v),cost=v['cost'],success=v['success']) for k,v in selected.items()},
                                         critic_minus_full_cost=costdif,cost_ci95=ci,critic_minus_full_success=successdif,success_ci95=si,critic_helps=helps)
    ledger=[json.loads(l) for l in (run/'ledger.jsonl').read_text().splitlines()]
    result['prediction_rows']=sum(r.get('prediction_rows',0) for r in ledger)
    result['fit_wall_seconds']=sum(r.get('wall_seconds',0) for r in ledger if r['kind']=='fit')
    result['failed_control_queries']=sum(r.get('failed_control_queries',0) for r in ledger)
    result['heldout_pre_gate_errors']=[]
    for row in ledger:
        if row['phase']=='critic_holdout' and row['kind']=='predict':
            with np.load(run/row['file']) as p:
                pre=p['states_to_predict'][:,0]<cfg['task']['gate']
                error=abs(p['values']-p['targets'])
                result['heldout_pre_gate_errors'].append(dict(scope=row['scope'],update=row['update'],pre_gate_mae=float(error[pre].mean()) if pre.any() else None,all_mae=float(error.mean())))
    if diag:
        def pooled_error(key):return float(np.sqrt(np.mean(np.concatenate([np.array(d[key]).ravel() for d in diag])**2)))
        def signs(pred,target):
            p=np.concatenate([np.array(d[pred]).ravel() for d in diag]);t=np.concatenate([np.array(d[target]).ravel() for d in diag])
            mask=abs(t)>1e-9
            return dict(accuracy=float(np.mean(np.sign(p[mask])==np.sign(t[mask]))) if mask.any() else None,nonzero_pairs=int(mask.sum()))
        result['diagnostics']=dict(critic_rmse=pooled_error('critic_error'),continuation_rmse=pooled_error('continuation_error'),
            critic_pair_sign=signs('approximate_hybrid_pair_differences','exact_hybrid_pair_differences'),
            continuation_pair_sign=signs('exact_hybrid_pair_differences','full_pair_differences'),
            combined_pair_sign=signs('approximate_hybrid_pair_differences','full_pair_differences'))
    proposed=[r for r in records if r['kind']=='proposal_diagnostic']
    if proposed:
        base=[];a=[];b=[];c=[]
        for row in proposed:
            aa=[]
            for q in row['base_A_queries']:
                with np.load(run/f'queries/{q}.npz') as p:aa.append(float(p['costs'].sum()))
            base.append(np.mean(aa));a.append(np.mean(row['A']));b.append(np.mean(row['B']));c.append(np.mean(row['C']))
        base,a,b,c=map(np.asarray,(base,a,b,c));mask=abs(a-base)>1e-9
        result['proposed_change_diagnostics']=dict(count=len(proposed),critic_rmse=float(np.sqrt(np.mean((c-b)**2))),
            continuation_rmse=float(np.sqrt(np.mean((b-a)**2))),actual_improving=int(np.sum(a<base-1e-9)),
            predicted_improving=int(np.sum(c<base-1e-9)),improvement_sign_accuracy=float(np.mean(np.sign(c[mask]-base[mask])==np.sign(a[mask]-base[mask]))) if mask.any() else None,
            raw=[dict(scope=r['scope'],update=r['update'],label=r['label'],A=r['A'],B=r['B'],C=r['C']) for r in proposed])
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    colors={'full_return':'#2563eb','critic_prefix':'#dc2626'}
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            rr=panel[f'seed{seed}/{arm}']
            for ax,metric in zip(axes,['success','cost']):
                ax.step([paid(r) for r in rr],[r[metric] for r in rr],where='post',color=colors[arm],alpha=.55,label=arm if seed==0 else None)
    axes[0].set_ylabel('Actor-only ending success');axes[0].set_ylim(-.03,1.03);axes[1].set_ylabel('Complete actor cost')
    for ax in axes:ax.set_xlabel('Paid controls, including audit and shared data');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(run/'learning_curves.png',dpi=170);plt.close(fig)
    if diag:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        for scope in scopes:
            dd=[d for d in diag if d['scope']==scope]
            axes[0].plot([d['update'] for d in dd],[d['critic_rmse'] for d in dd],alpha=.6)
            axes[1].plot([d['update'] for d in dd],[d['continuation_rmse'] for d in dd],alpha=.6)
        axes[0].set_title('C−B: critic approximation');axes[1].set_title('B−A: continuation mismatch')
        for ax in axes:ax.set_xlabel('Update');ax.set_ylabel('Return RMSE');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(run/'diagnostics.png',dpi=170);plt.close(fig)
    result['report_wall_seconds']=time.perf_counter()-start
    (run/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# Ending actor comparison results','',f"Status: **{summary['status']}**. Frozen single-task, three-seed mechanism test.",'',
           '| Actor scoring | Final success per seed | Mean final cost | Accepted updates | Ending criterion |','|---|---|---:|---:|---|']
    for r in result['final']:
        lines.append(f"| {r['arm']} | {', '.join(f'{v:.3f}' for v in r['success']) or 'not trained'} | {np.mean(r['cost']) if r['cost'] else float('nan'):.3f} | {r['accepted']} | {'passed' if r['ending_learned'] else 'failed'} |")
    lines+=['',f"Main controls: {summary['main_controls']:,}; main wall time: {summary['main_wall_seconds']:.3f}s. Audit counts/wall time and standalone arm allocations are in audit.json. Fits: {result['fit_calls']}; archived models: {result['models']}; fitting wall time: {result['fit_wall_seconds']:.3f}s; prediction rows: {result['prediction_rows']:,}.",
            '', '![Paid-cost learning curves](learning_curves.png)','']
    if 'common_budget' in result:
        cb=result['common_budget'];lines += [f"At the predeclared common budget {cb['controls']:,}, critic-minus-full mean cost is {np.mean(cb['critic_minus_full_cost']):.3f} (paired-seed bootstrap 95% interval {cb['cost_ci95']}); success difference {np.mean(cb['critic_minus_full_success']):.3f} ({cb['success_ci95']}). Critic-help criterion: **{'passed' if cb['critic_helps'] else 'failed'}**.",'']
    if 'diagnostics' in result:
        d=result['diagnostics'];lines += [f"Diagnostic C−B RMSE: {d['critic_rmse']:.3f}; B−A RMSE: {d['continuation_rmse']:.3f}. Nonzero pair sign accuracies: C vs B {d['critic_pair_sign']}; B vs A {d['continuation_pair_sign']}; C vs A {d['combined_pair_sign']}.",'','![Error decomposition](diagnostics.png)','']
    lines += ['Qualification, individual roots, complete query traces, actor candidates and every critic are retained. This study separates policy learning from action imitation. The task deliberately opposes a locally attractive lane and a delayed desired ending; it is not evidence that a policy preserves walking while improving settling. Independent longer search supplies successful reset coverage. No general exploration gain or global-to-local GPS improvement is established by these two arms. See the frozen protocol and prior-art note for assumptions and limits.']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('audit','heldout_pre_gate_errors')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);report(p.parse_args().run_dir)
