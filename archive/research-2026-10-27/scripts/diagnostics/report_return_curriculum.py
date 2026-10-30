"""Render closed curriculum evidence; never import a work producer."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--deployment',action='store_true',help='Label deployment-adjoint actor updates and external MC control accurately')
    parser.add_argument('--latest-audit',action='store_true',help='Bind latest additive audit, retaining hashes for its predecessors')
    args=parser.parse_args();started=time.perf_counter();out=args.out.resolve();out.mkdir(exist_ok=False)
    source=out/'report_source.py';source.write_bytes(Path(__file__).read_bytes())
    inputs=[];runs=[]
    for root in args.run:
        root=root.resolve();result=root/'curriculum_result.json';audit=root/'audit_001.json'
        history=sorted(root.glob('audit_[0-9][0-9][0-9].json'))
        if args.latest_audit and history:audit=history[-1]
        r=json.loads(result.read_text());a=json.loads(audit.read_text()) if audit.exists() else None
        assert r['stage'] in ('complete_measured_loop_with_negative_links','complete_conditional_two_round_loop')
        inputs.append(dict(run=str(root),result_sha256=sha(result),audit_sha256=sha(audit) if a else None,
            query_ledger_sha256=r['query_ledger_sha256'],work_ledger_sha256=r['work_ledger_sha256'],audit_file=audit.name,
            audit_history=[dict(file=p.name,sha256=sha(p)) for p in history]))
        runs.append((root,r,a))
    write(out/'started.json',dict(inputs=inputs,source_sha256=sha(source),new_controls=0,new_actor_rows=0,new_optimizer_work=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'raw':'#3465a4','guided':'#d07900','parent':'#555555'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    phase=[];comparisons=[];charges=[]
    fig,axes=plt.subplots(len(runs),3,figsize=(15,3.7*len(runs)),squeeze=False,layout='constrained')
    for i,(root,r,a) in enumerate(runs):
        run=root.name;initial=r['initial'];rounds=r['rounds'];final=r['final']
        for prefix,learning in [('initial',initial['learning'])]+[(f'round{j}',z['children']) for j,z in enumerate(rounds)]:
            for arm,data in learning.items():
                for p in data['curves']:
                    phase.append(dict(run=run,arm=prefix+'_'+arm,phase=p['phase'],updates=p['updates'],
                        capture_rate=p['summary']['capture_rate'],mean_cost=p['summary']['mean_cost'],
                        campaign_charged_controls=p['charged_controls'],selected_phase=data.get('selected_phase')))
        for stage,panels in [('initial',initial['qualification'])]+[(f'round{j}',z['student_qualification']) for j,z in enumerate(rounds)]+[('final',final['actors'])]:
            for arm,s in panels.items():comparisons.append(dict(run=run,type='standalone',stage=stage,arm=arm,
                mean_cost=s['mean_cost'],capture_rate=s['capture_rate'],roots=len(s['outcomes'])))
        for stage,p in [(f'round{j}',z['teacher_qualification']) for j,z in enumerate(rounds)]+[('final',final['teacher'])]:
            for arm,s in p['summaries'].items():comparisons.append(dict(run=run,type='controller',stage=stage,arm=arm,
                mean_cost=s['mean_cost'],capture_rate=s['capture_rate'],roots=len(s['outcomes'])))
        charges.append(dict(run=run,model_controls=r['model_controls'],executed_controls=r['executed_controls'],
            total_controls=r['accounting']['completed_controls'],actor_rows=r['actor_rows_requested'],adam_updates=r['adam_attempts'],
            diagnostic_backwards=r['diagnostic_backward_attempts'],queries=r['accounting']['queries'],
            errors=r['accounting']['error_queries'],rejected=r['accounting']['rejected_queries'],unknown_controls=r['accounting']['unknown_controls'],
            body_wall_seconds=r['body_wall_seconds'],audit_wall_seconds=a['audit_wall_seconds'] if a else None,
            audit_status=a['status'] if a else 'PENDING',scientific_success=r['passed']))
        ax=axes[i,0]
        for prefix,style in [('initial','-'),('round0','--'),('round1',':')]:
            for arm in ('raw','guided'):
                ps=[p for p in phase if p['run']==run and p['arm']==prefix+'_'+arm]
                label=('MC control' if prefix=='initial' and arm=='raw' else 'new learner' if prefix=='initial' else arm+' data') if args.deployment else ('control' if arm=='raw' else 'discovered/guided')
                ax.plot([p['updates'] for p in ps],[p['mean_cost'] for p in ps],style,marker='o',color=colors[arm],alpha=.9,label=prefix+' '+label)
        ax.set(xlabel='Actor updates in this arm' if args.deployment else 'Own-return updates in this arm',ylabel='Complete mean cost (lower better)',title=run.rsplit('_',1)[-1]+': development curves',yscale='log')
        ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15)
        ax=axes[i,1];stages=['initial','round0','round1','final']
        capture_rows=[]
        for arm in ('parent','raw','guided'):
            ps=[next(p for p in comparisons if p['run']==run and p['type']=='standalone' and p['stage']==s and p['arm']==arm) for s in stages]
            ax.plot(stages,[p['mean_cost'] for p in ps],marker='o',color=colors[arm],label=arm,
                ls='--' if arm=='guided' else '-',markerfacecolor='white' if arm=='guided' else colors[arm])
            capture_rows.append([round(100*p['capture_rate']) for p in ps])
        for j,stage in enumerate(stages):
            caps=[v[j] for v in capture_rows];label=(str(caps[0])+'% each') if len(set(caps))==1 else '/'.join(map(str,caps))+'%'
            ax.text(j,.98,label,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
        ax.set(title='Fresh standalone: capture parent/raw/guided',ylabel='Complete mean cost');ax.legend(fontsize=8,loc='center right');ax.grid(alpha=.15)
        ax=axes[i,2];stages=['round0','round1','final']
        for arm in ('raw','actor','guided'):
            ps=[next(p for p in comparisons if p['run']==run and p['type']=='controller' and p['stage']==s and p['arm']==arm) for s in stages]
            ax.plot(stages,[p['mean_cost'] for p in ps],marker='o',label=arm,color=colors.get(arm,'#555555'),
                ls='--' if arm=='guided' else '-',markerfacecolor='white' if arm=='guided' else colors.get(arm,'#555555'))
        fallback=all(z['teacher_qualification']['handover_episodes']==0 for z in rounds) and final['teacher']['handover_episodes']==0
        ax.set(title='Executed controllers: guided = RAW fallback' if fallback else 'Executed strong search / actor / hybrid',ylabel='Complete mean cost');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(out/'learning_curves.png',dpi=150);fig.savefig(out/'learning_curves.svg');plt.close(fig)
    for name,data in [('phase_curves.csv',phase),('fresh_comparisons.csv',comparisons),('actual_charges.csv',charges)]:
        with (out/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    # A trace animation is rendered from recorded states only, never a replay.
    import numpy as np
    from matplotlib.animation import FuncAnimation,PillowWriter
    root,r,a=runs[-1];events={}
    for line in (root/'evidence/curriculum/ledger.jsonl').open():
        event=json.loads(line)
        if event['kind']=='episode_outcome' and event['episode'] in ('final_teacher_RAW_00','final_teacher_GUIDE_00','final_teacher_actor_00_000'):
            events[event['episode']]=event
    trace_refs=[];fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained');artists=[]
    for name,role in [('final_teacher_RAW_00','RAW'),('final_teacher_GUIDE_00','hybrid'),('final_teacher_actor_00_000','actor')]:
        event=events[name];ref=event['raw'];path=root/'evidence/curriculum'/ref['file'];assert sha(path)==ref['sha256']
        with np.load(path,allow_pickle=False) as z:states=z['states'].copy()
        label=role+' (RAW fallback)' if role=='hybrid' and r['final']['teacher']['handover_episodes']==0 else role
        line,=ax.plot([],[],label=label,lw=2,color={'RAW':'#3465a4','hybrid':'#d07900','actor':'#555555'}[role],
            ls='--' if role=='hybrid' else '-');dot,=ax.plot([],[],marker='o',color=line.get_color())
        artists.append((line,dot,states));trace_refs.append(dict(run=str(root),episode=name,raw=ref))
    ax.axhspan(-1,-.1,color='green',alpha=.08);ax.axhline(-.1,color='green',ls='--',label='ending boundary')
    ax.axvline(12,color='gray',ls=':');ax.set(xlim=(0,25),ylim=(-1.05,1.05),xlabel='Executed control boundary',ylabel='Recorded physical y',
        title='Representative final root: recorded trajectories');ax.legend(ncol=4,fontsize=8)
    def frame(t):
        for line,dot,states in artists:
            line.set_data(states[:t+1,0],states[:t+1,1]);dot.set_data(states[t:t+1,0],states[t:t+1,1])
        return [a for pair in artists for a in pair[:2]]
    FuncAnimation(fig,frame,frames=26,interval=160).save(out/'recorded_trajectories.gif',writer=PillowWriter(fps=6));plt.close(fig)
    write(out/'receipt.json',dict(schema='closed-curriculum-evidence-report-v1',inputs=inputs,source_sha256=sha(source),
        trace_refs=trace_refs,files=[dict(file=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()],
        wall_seconds=time.perf_counter()-started,new_simulator_controls=0,new_actor_rows=0,new_optimizer_steps=0,
        deployment_labels=args.deployment,limits='Development curves are selected during learning; fresh assessments are separate. One seed; no efficiency or physical-task claim. Animation uses recorded states only.'))
    print(str(out/'learning_curves.png'))


if __name__=='__main__':main()
