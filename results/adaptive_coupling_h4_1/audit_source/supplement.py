"""Saved-only paired-gap, first-divergence and storage clarification checks."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

START=time.perf_counter()
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/2026-10-30_adaptive_coupling_h4_1_r1_001'
OUT=Path(__file__).resolve().parent/'h4_receipt_001'
r=json.loads((RUN/'result.json').read_text())
locator=json.loads((RUN/'exports/full_local_evidence_locator.json').read_text())
bindings=json.loads((OUT/'file_bindings.json').read_text())
receipt=json.loads((OUT/'receipt.json').read_text())
assert receipt['passed']
for key,path in [('full_raw',RUN/'evidence/raw.h5'),('full_ledger',RUN/'evidence/ledger.jsonl')]:
    assert locator[key]['file']==str(path)
    assert locator[key]['sha256']==bindings[str(path)]['sha256']
    assert locator[key]['bytes']==bindings[str(path)]['bytes']==path.stat().st_size
assert hashlib.sha256((RUN/'result.json').read_bytes()).hexdigest()==locator['result_sha256']
events={};decisions={};private={}
wanted=set()
for i,pair in enumerate(r['manipulation']['backoff_vs_fixed_high']):
    t=pair['first_byte_differing_action_control']
    for arm in ('fixed_high','backoff'):
        wanted.add((r['teacher_summaries'][arm]['tags'][i],t))
with (RUN/'evidence/ledger.jsonl').open() as stream:
    for line in stream:
        if '"event":"episode_complete"' in line:
            row=json.loads(line);events[row['tag']]=row
        elif '"event":"decision"' in line:
            row=json.loads(line);key=(row['episode'],row['step'])
            if key in wanted:decisions[key]=row
        elif '"event":"independent_planner_before"' in line:
            row=json.loads(line);key=(row['episode'],row['step'])
            if key in wanted:private[key]=row['private']

def group(stream,tag):
    return stream[events[tag]['raw']['group']]

gaps={};first_divergences=[];limits=defaultdict(Counter)
with h5py.File(RUN/'evidence/raw.h5','r') as f:
    acquisition=r['cases']['acquisition_starts']
    for arm,teacher in r['teacher_summaries'].items():
        for i,teacher_tag in enumerate(teacher['tags']):
            original=group(f,teacher_tag)
            for policy in ('parent',arm):
                tag=acquisition[policy]['tags'][i];other=group(f,tag)
                for key in ('states','warmstarts','integrations','observations'):
                    assert np.array_equal(original[key][0],other[key][0]),(arm,policy,i,key)
        before=np.asarray(acquisition['parent']['costs'])-teacher['costs']
        after=np.asarray(acquisition[arm]['costs'])-teacher['costs']
        widened=after-before
        assert (widened>0).all()
        gaps[arm]=dict(full_initial_snapshots_equal=True,
            before_parent_minus_teacher_costs=before.tolist(),
            after_child_minus_teacher_costs=after.tolist(),
            gap_increases=widened.tolist(),mean_before_gap=float(before.mean()),
            mean_after_gap=float(after.mean()),mean_gap_increase=float(widened.mean()),
            scope='Identical full acquisition reset states, warmstarts and integration snapshots; diagnostic training starts, not fresh held generalization.')
    for i,pair in enumerate(r['manipulation']['backoff_vs_fixed_high']):
        t=pair['first_byte_differing_action_control']
        names=[r['teacher_summaries'][arm]['tags'][i] for arm in ('fixed_high','backoff')]
        left,right=[group(f,tag) for tag in names]
        for key in ('states','warmstarts','integrations','observations'):
            assert np.array_equal(left[key][t],right[key][t]),(i,key)
        assert private[names[0],t]==private[names[1],t]
        d0,d1=[decisions[tag,t] for tag in names]
        assert d0['bank_digest']==d1['bank_digest']
        b0,b1=[f[d['evidence']['group']] for d in (d0,d1)]
        equal_keys=['initial_state','initial_warmstart','bank_actions','bank_post_states','bank_raw_costs',
                    'reference_u','baseline_scores','agreement_squared','support','bank_observations']
        for key in equal_keys:
            assert np.array_equal(b0[key][()],b1[key][()],equal_nan=True),(i,key)
        base=b0['baseline_scores'][()]
        assert np.allclose(b0['guided_scores'][()]-base,2*(b1['guided_scores'][()]-base),atol=1e-10,rtol=1e-12,equal_nan=True)
        control_distance=float(np.linalg.norm(left['actions'][t]-right['actions'][t]))
        assert control_distance>1e-6
        first_divergences.append(dict(pair=i,control=t,bank_digest=d0['bank_digest'],
            full_simulator_snapshots_equal=True,private_nominal_and_rng_equal=True,
            actual_bank_arrays_equal=equal_keys,
            same_actor_hash=d0['actor_hash'],high_weight=d0['alpha'],backoff_weight=d1['alpha'],
            objective_influence_ratio_high_to_backoff=2.0,selected_first_action_difference_norm=control_distance,
            interpretation='Same updated actor, physical state, warmstart, private planner state and candidate bank; only the agreement coefficient differs before the first changed executed control.'))
    for arm,score in r['cases']['primary'].items():
        for tag in score['tags']:
            z,angle=group(f,tag)['states'][-1,2:4];cause=[]
            if not .8<z<2:cause.append('height')
            if not -1<angle<1:cause.append('angle')
            limits[arm]['+'.join(cause) or 'full_survival']+=1

primary=r['cases']['primary'];p=primary['parent'];b=primary['backoff']
cost_deltas=np.asarray(b['costs'])-p['costs'];control_deltas=np.asarray(b['steps'])-p['steps']
assert (cost_deltas>0).all() and (control_deltas<0).all()
reductions={arm:(r['full_mse']['parent_on_'+arm]-r['full_mse'][arm+'_2000'])/r['full_mse']['parent_on_'+arm] for arm in ('independent','fixed_high','backoff')}
assert not r['next_weight_proposal']['trigger'] and r['next_weight_proposal']['factor']==1
result=dict(passed=True,local_full_raw_locator_matches_independent_audit_hash=True,
    full_raw=locator['full_raw'],full_ledger=locator['full_ledger'],paired_gap_before_after=gaps,
    first_divergence_causal_proofs=first_divergences,primary_termination_limits=dict(limits),
    backoff_vs_parent=dict(all16_complete_costs_worse=True,all16_controls_shorter=True,
        cost_deltas=cost_deltas.tolist(),control_deltas=control_deltas.tolist()),
    recorded_full_dataset_mse_reduction_fractions=reductions,
    trigger_limitation='All matched child success/survival counts tie at zero; the rule proposes hold despite regression against the incoming parent. Proposal only; no next acquisition executed.',
    input_result_sha256=hashlib.sha256((RUN/'result.json').read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    new_controls=0,new_neural_rows=0,new_backward_rows=0,new_optimizer_steps=0,
    wall_seconds=time.perf_counter()-START)
with (OUT/'supplemental_native_detail.json').open('x') as stream:
    json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(result))
