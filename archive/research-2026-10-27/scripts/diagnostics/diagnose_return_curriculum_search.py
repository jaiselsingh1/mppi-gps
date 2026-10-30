"""Separate coverage from candidate-quality failures using saved forecasts only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();started=time.perf_counter();root=args.run.resolve();out=args.out.resolve();out.mkdir(exist_ok=False)
    (out/'analyze.py').write_bytes(Path(__file__).read_bytes());result=json.loads((root/'curriculum_result.json').read_text())
    assert result['stage']!='execution_error'
    base=root/'evidence/curriculum';assert sha(base/'ledger.jsonl')==result['query_ledger_sha256']
    inputs={str(p):sha(p) for p in (root/'curriculum_result.json',base/'ledger.jsonl',root/'manifest.json')}
    write(out/'started.json',dict(run=str(root),inputs=inputs,source_sha256=sha(out/'analyze.py'),new_controls=0,new_actor_rows=0,new_fits=0))
    records=[];eligibility=[]
    for line in (base/'ledger.jsonl').open():
        event=json.loads(line)
        if event['kind']=='curriculum_suffix_eligibility':eligibility.append(event)
        if event['kind']=='curriculum_hybrid_selection':records.append(event)
    inputs_arrays=[];rows=[];candidates=[]
    for event in records:
        index=0 if event['episode'].startswith('round0_') else 1;trust=eligibility[index]
        anchors=np.asarray([a for a,o in zip(trust['support_anchors'],trust['outcomes']) if o['capture']]);items=[]
        for k,ref in enumerate(event['candidate_refs']):
            path=base/ref['file'];digest=sha(path);assert digest==ref['sha256'];inputs_arrays.append(dict(file=str(path),sha256=digest,bytes=path.stat().st_size))
            with np.load(path,allow_pickle=False) as z:
                cost=float(z['cost']);state=z['tail_states'][0].copy();capture=bool(z['tail_states'][-1,1]<=-.1)
            obs=np.zeros(8);obs[0]=state[1];obs[1]=state[0]/25;obs[2]=1-obs[1]
            distance=float(np.linalg.norm(anchors-obs,axis=1).min());supported=distance<=.15
            row=dict(episode=event['episode'],candidate=k,cost=cost,capture=capture,handoff_y=float(state[1]),
                minimum_support_distance=distance,supported=supported,gain_vs_RAW=event['RAW_cost']-cost)
            items.append(row);candidates.append(row)
        capturing=[v for v in items if v['capture']];supported=[v for v in capturing if v['supported']]
        best=min(capturing,key=lambda v:v['cost']) if capturing else None;best_supported=min(supported,key=lambda v:v['cost']) if supported else None
        rows.append(dict(episode=event['episode'],RAW_cost=event['RAW_cost'],candidates=len(items),capturing_candidates=len(capturing),
            supported_capturing_candidates=len(supported),best_capturing_gain=None if best is None else best['gain_vs_RAW'],
            best_supported_gain=None if best_supported is None else best_supported['gain_vs_RAW'],
            improving_capturing_candidates=sum(v['capture'] and v['gain_vs_RAW']>0 for v in items),
            improving_supported_candidates=sum(v['capture'] and v['supported'] and v['gain_vs_RAW']>0 for v in items),
            selected=event['selected'] is not None))
    for name,data in [('root_forecasts.csv',rows),('candidate_forecasts.csv',candidates)]:
        with (out/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['no_candidates']);writer.writeheader();writer.writerows(data)
    unchanged=all(sha(Path(p))==digest for p,digest in inputs.items());assert unchanged
    write(out/'diagnosis.json',dict(schema='saved-policy-completion-candidate-diagnosis-v1',run=str(root),
        teacher_roots=len(rows),candidate_controllers=len(candidates),
        roots_with_any_capturing_forecast_improvement=sum(v['improving_capturing_candidates']>0 for v in rows),
        roots_with_supported_improvement=sum(v['improving_supported_candidates']>0 for v in rows),
        selected_handovers=sum(v['selected'] for v in rows),
        maximum_unrestricted_capturing_gain=max((v['gain_vs_RAW'] for v in candidates if v['capture']),default=None),
        maximum_supported_capturing_gain=max((v['gain_vs_RAW'] for v in candidates if v['capture'] and v['supported']),default=None),
        inputs=inputs,array_inputs=inputs_arrays,original_inputs_unchanged=unchanged,
        source_sha256=sha(out/'analyze.py'),outputs={n:sha(out/n) for n in ('root_forecasts.csv','candidate_forecasts.csv')},
        wall_seconds=time.perf_counter()-started,new_simulator_controls=0,new_actor_rows=0,new_optimizer_steps=0,
        limits='Hypothetical saved forecasts only. Improvement outside tested support is not an executed-control gain or permission for a handover. No learned function or simulator is executed.'))
    print(json.dumps({k:v for k,v in json.loads((out/'diagnosis.json').read_text()).items() if k not in ('inputs','array_inputs','outputs')},indent=2))


if __name__=='__main__':main()
